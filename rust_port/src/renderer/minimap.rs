//! Port of `CMinimap` (MatrixMinimap.cpp/hpp).
//!
//! Ported pieces:
//!   - Background: offscreen orthographic landscape render into a 512×512
//!     texture at init (`bake_background` → `TerrainRenderer::bake_minimap`),
//!     mirroring `CMinimap::RenderBackground` (MatrixMinimap.cpp:855-1199).
//!   - World→map / map→world transforms with center + scale (pan/zoom
//!     "button" API in MatrixMinimap.hpp:137-169) and optional rotation
//!     around the minimap center (MatrixMinimap.cpp:159-180, guarded by
//!     `MINIMAP_SUPPORT_ROTATION`).
//!   - Timed event overlays plus off-screen arrow indicators
//!     (MatrixMinimap.cpp:332-395, 638-678).
//!   - Building markers colored by side (MatrixMinimap.cpp:679-763).
//!   - Camera frustum projected onto the water plane and drawn as a 4-line
//!     LINESTRIP in `MINIMAP_CAM_COLOR` (MatrixMinimap.cpp:794-841).
//!   - Layout from `CInterface.cpp:557` — 145×145 at panel offset (+13, +51)
//!     with the bottom-anchor rule from `CInterface.cpp:176-183`.
//!   - Per-frame `m_Center = camera.GetXYStrategy()` (MatrixMap.cpp:1261).
//!
//! Still intentionally skipped:
//!   - In-robot `DrawRadar` — arcade mode isn't ported.
//!   - Disk-cached background PNG (irrelevant in-browser).

use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::game::map::{GameMap, GLOBAL_SCALE};
use crate::renderer::camera::Camera;
use crate::renderer::terrain::TerrainRenderer;
use crate::renderer::texture::{create_texture_from_rgba, decode_texture_bytes};

/// Matches `MINIMAP_SIZE` in MatrixMinimap.hpp:14.
const TEX_SIZE: u32 = 512;

/// Layout constants in the SR2 1024×768 base design, pulled from the "if"
/// record in robots.dat (MiniM block → mmp/Static + CInterface.cpp:557):
///   MiniM.xPos = 0, MiniM.yPos = 559 (bottom-anchored)
///   Static mmp.xSize/ySize = 176×209 — this is the panel art's display size
///   Minimap landscape renders at (xPos+13, yPos+51, 145, 145) inside mmp
const MINIMAP_SIZE_PX: f32 = 145.0;
const MINIMAP_OFFSET_X: f32 = 13.0;
/// Y-offset from the top of the MiniM parent panel — matches `m_yPos + 51`
/// at CInterface.cpp:557 (panel-local).
const MINIMAP_OFFSET_Y_IN_PANEL: f32 = 51.0;
const MINIMAP_PANEL_W: f32 = 176.0;
const MINIMAP_PANEL_H: f32 = 209.0;
/// Panel Y in the 1024×768 base (MiniM.yPos from robots.dat). With the
/// bottom-anchor rule at CInterface.cpp:180-183 the final Y becomes
/// `screen_h - (768 - 559) = screen_h - 209`.
const MINIMAP_PANEL_Y_BASE: f32 = 559.0;
const DESIGN_HEIGHT_PX: f32 = 768.0;

/// `MINIMAP_CAM_COLOR = 0xFF30FFFF` (MatrixMinimap.cpp:19), D3DCOLOR ARGB.
const MINIMAP_CAM_COLOR: [f32; 4] = [48.0 / 255.0, 1.0, 1.0, 1.0];

/// MatrixMinimap.hpp:11-12 — scale clamp range for the zoom buttons.
const MINIMAP_MIN_SCALE: f32 = 1.0;
const MINIMAP_MAX_SCALE: f32 = 3.0;
/// Multiplicative zoom factors — mirror MatrixMinimap.cpp:1346 / 1352:
///   ButtonZoomIn:  SetOutParams(GetScale() * 1.8)
///   ButtonZoomOut: SetOutParams(GetScale() * 0.5)
/// Clamped to [MIN, MAX], so from 1.0 the reachable levels are 1.0, 1.8, 3.0.
const MINIMAP_ZOOM_IN_FACTOR: f32 = 1.8;
const MINIMAP_ZOOM_OUT_FACTOR: f32 = 0.5;
const MINIMAP_MAX_EVENTS: usize = 32;
const MINIMAP_EVENT_TTL_MS: f32 = 1000.0;
const MINIMAP_EVENT_R1: f32 = 15.0;
const MINIMAP_EVENT_R2: f32 = 1.0;
const MINIMAP_FLASH_PERIOD_MS: f32 = 100.0;
const MINIMAP_OUT_SCALE: f32 = 0.8;
const MINIMAP_OUT_INDICATOR_R: f32 = 15.0;
const MINIMAP_BUILDING_R: f32 = 8.0;
const MINIMAP_BUILDING_BASE_R: f32 = 8.0;

/// Water plane Z — from `renderer/water.rs`, matches `WATER_LEVEL` in the
/// original. Used both for the heightmap bake (coast cutoff) and for the
/// frustum projection plane in `frustum_on_water`.
const WATER_LEVEL: f32 = -2.0;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct MMVertex {
    pos: [f32; 2], // clip space
    uv: [f32; 2],
    color: [f32; 4],
}

/// Normalized UV sub-rect inside the icon atlas — matches `CMinimap::SMMTex`
/// (MatrixMinimap.hpp:58-66): texture + (u0,v0)/(u1,v1) of the sprite.
#[derive(Copy, Clone, Default)]
struct IconUv {
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
}

/// Static image sub-rect parsed from a UI block's sNormal* params.
/// (sNormalX, sNormalY, display_w, display_h, texture_w, texture_h).
/// UV normalization matches CIFaceElement.cpp:60-66.
#[derive(Copy, Clone, Default)]
struct SubRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    tex_w: f32,
    tex_h: f32,
}

/// Button UI element port of CIFaceButton (CInterface.cpp:430-467). Stores
/// the panel-local draw rect and the interface1 sub-rect UV for the
/// `sNormal` state (we don't bother animating focused/pressed).
#[derive(Copy, Clone, Default)]
struct Button {
    local: [f32; 4], // xPos, yPos, xSize, ySize (panel-local design-space)
    uv: IconUv,
}

#[derive(Copy, Clone)]
struct MinimapEvent {
    pos_in_world: [f32; 2],
    ttl: f32,
    color1: u32,
    color2: u32,
}

/// Result of a minimap left-press — see `Minimap::click`.
pub enum MinimapClick {
    None,
    BeginDrag([f32; 2]),
    ZoomIn,
    ZoomOut,
}
impl SubRect {
    fn to_uv(self) -> IconUv {
        IconUv {
            u0: self.x / self.tex_w,
            v0: self.y / self.tex_h,
            u1: (self.x + self.w) / self.tex_w,
            v1: (self.y + self.h) / self.tex_h,
        }
    }
}

pub struct Minimap {
    tex_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    bg_bind: wgpu::BindGroup,
    icon_bind: wgpu::BindGroup,
    panel_bind: wgpu::BindGroup,
    white_bind: wgpu::BindGroup,

    /// Parsed layout for the `mmp` static inside the `MiniM` panel block of
    /// robots.dat → "if" record. Source rect is in `interface1` pixel coords;
    /// we normalize at render time by the atlas dimensions.
    panel_src: SubRect,
    /// Normalized (u0,v0,u1,v1) for the MiniM panel art on `interface1`.
    panel_uv: IconUv,

    // Dynamic per-frame vertex streams.
    quad_vbuf: wgpu::Buffer, // TriangleList, grows via write_buffer
    quad_capacity: usize,    // vertex slots allocated
    line_vbuf: wgpu::Buffer, // LineStrip, fixed 5 verts for the frustum loop

    // Background texture kept alive because the bind group holds only a view.
    _bg_tex: wgpu::Texture,
    bg_depth_tex: wgpu::Texture,
    _white_tex: wgpu::Texture,

    /// Per-side minimap color (index = side id, 0..=4 from robots.dat `Side`
    /// block, MatrixMap.cpp:1015). Parsed once at init.
    side_colors: [[f32; 4]; 8],

    /// Zoom-in / zoom-out buttons parsed from the `MiniM` panel children
    /// (Button{Name=zi}/{Name=zo}). Drawn on top of the panel art; clicking
    /// them calls zoom_in/zoom_out (MatrixMinimap.hpp:192-193).
    zi_button: Button,
    zo_button: Button,

    /// Target scale tracked by the buttons, matches `m_TgtScale`
    /// (MatrixMinimap.hpp:101). Clamped to [MINIMAP_MIN_SCALE, MINIMAP_MAX_SCALE].
    tgt_scale: f32,

    /// Sub-rects for the icon atlas parsed from the `Minimap` block.
    /// `Minimap` block in robots.dat (MatrixMinimap.cpp:50-65).
    icon_arrow: IconUv,
    #[allow(dead_code)]
    icon_flyer: IconUv,
    #[allow(dead_code)]
    icon_turret: IconUv,
    #[allow(dead_code)]
    icon_robot: IconUv,
    icon_base: IconUv,
    icon_factory: IconUv,
    icon_point: IconUv,
    events: Vec<MinimapEvent>,
    time_ms: f32,

    /// Set once `bake_background` has run. Matches the original's caching:
    /// `RenderBackground` runs once at map load; subsequent frames just sample
    /// `m_Texture`.
    baked: bool,

    // ── Layout / state, mirrors CMinimap members ────────────────────────────
    pos_x: f32,
    pos_y: f32,
    size_x: f32,
    size_y: f32,
    center: [f32; 2], // m_Center
    scale: f32,       // m_Scale
    angle: f32,       // applied rotation (radians, world-space)

    // Cached per-frame
    delta: [f32; 2], // m_Delta

    // Map bounds baked at construction.
    map_sx: i32,
    map_sy: i32,
}

impl Minimap {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        map: &GameMap,
        matrix_data: &Storage,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Self {
        // ── Background render target ────────────────────────────────────────
        // 512×512 matching MINIMAP_SIZE. `bake_background` will fill this the
        // first time the app issues the minimap bake pass; until then it's a
        // cleared texture, same as the original's post-`RenderBackground` state.
        let bg_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("minimap bg"),
            size: wgpu::Extent3d {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Matches surface format so `TerrainRenderer::bake_minimap` can
            // reuse its color pipeline targets without a second pipeline.
            format: surface_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let bg_view = bg_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let bg_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("minimap bg depth"),
            size: wgpu::Extent3d {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let white_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("minimap white"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &white_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let white_view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("minimap sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("minimap bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let make_bind = |view: &wgpu::TextureView, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            })
        };
        let bg_bind = make_bind(&bg_view, "minimap bg bind");
        let white_bind = make_bind(&white_view, "minimap white bind");

        // ── Icon atlas ───────────────────────────────────────────────────────
        // Port of `CMinimap::Init` (MatrixMinimap.cpp:42-74): read the
        // `Minimap` block from robots.dat, load the icon atlas referenced by
        // each entry (all entries point at the same radarIcons texture), and
        // cache normalized UV rects for the sprites we actually draw.
        let (icon_tex_size, icon_rects) = parse_minimap_block(matrix_data);
        let icons_path = "Matrix/Textures/Minimap/radarIcons";
        let icons_rgba = read_texture(icons_path)
            .and_then(|bytes| decode_texture_bytes(&bytes))
            .unwrap_or_else(|| {
                log::warn!(
                    "minimap: icon atlas {} not found in bundle/pkg; falling back to 1×1 white",
                    icons_path
                );
                image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]))
            });
        let icons_view = create_texture_from_rgba(device, queue, &icons_rgba);
        let icon_bind = make_bind(&icons_view, "minimap icon bind");

        let atlas_w = icons_rgba.width() as f32;
        let atlas_h = icons_rgba.height() as f32;
        let default_w = icon_tex_size as f32; // 128 or whatever the dat reports
        let default_icon = IconUv {
            u0: 0.0,
            v0: 0.0,
            u1: 1.0,
            v1: 1.0,
        };
        let icon_arrow = icon_rects
            .get("arrow")
            .copied()
            .map(|r| r.normalize(atlas_w, atlas_h))
            .unwrap_or(default_icon);
        let icon_flyer = icon_rects
            .get("flyer")
            .copied()
            .map(|r| r.normalize(atlas_w, atlas_h))
            .unwrap_or(default_icon);
        let icon_turret = icon_rects
            .get("turret")
            .copied()
            .map(|r| r.normalize(atlas_w, atlas_h))
            .unwrap_or(default_icon);
        let icon_robot = icon_rects
            .get("robot")
            .copied()
            .map(|r| r.normalize(atlas_w, atlas_h))
            .unwrap_or(default_icon);
        let icon_base = icon_rects
            .get("base")
            .copied()
            .map(|r| r.normalize(atlas_w, atlas_h))
            .unwrap_or(default_icon);
        let icon_factory = icon_rects
            .get("factory")
            .copied()
            .map(|r| r.normalize(atlas_w, atlas_h))
            .unwrap_or(icon_base);
        let icon_point = icon_rects
            .get("point")
            .copied()
            .map(|r| r.normalize(atlas_w, atlas_h))
            .unwrap_or(icon_base);
        let _ = default_w; // silence unused warning

        // ── Panel art from robots.dat `if` / MiniM / Static `mmp` ──────────
        // Sub-rect layout mirrors CIFaceElement.cpp:60-66: u = xTexPos / tex_w,
        // sub size = element's xSize × ySize.
        let panel_src = parse_mmp_static(matrix_data).unwrap_or(SubRect {
            x: 0.0,
            y: 88.0,
            w: 176.0,
            h: 209.0,
            tex_w: 512.0,
            tex_h: 512.0,
        });
        let panel_path = "Matrix/Iface/interface1";
        let panel_rgba = read_texture(panel_path)
            .and_then(|bytes| decode_texture_bytes(&bytes))
            .unwrap_or_else(|| {
                log::warn!(
                    "minimap: {} not found in bundle/pkg; falling back to solid frame",
                    panel_path
                );
                image::RgbaImage::from_pixel(1, 1, image::Rgba([16, 18, 28, 235]))
            });
        let panel_view = create_texture_from_rgba(device, queue, &panel_rgba);
        let panel_bind = make_bind(&panel_view, "minimap panel bind");
        let panel_uv = panel_src.to_uv();

        let side_colors = parse_side_colors_mm(matrix_data);

        let zi_button = parse_minim_button(matrix_data, "zi").unwrap_or(Button {
            local: [149.0, 187.0, 12.0, 12.0],
            uv: IconUv { u0: 446.0/512.0, v0: 44.0/512.0, u1: 458.0/512.0, v1: 56.0/512.0 },
        });
        let zo_button = parse_minim_button(matrix_data, "zo").unwrap_or(Button {
            local: [10.0, 187.0, 12.0, 12.0],
            uv: IconUv { u0: 398.0/512.0, v0: 44.0/512.0, u1: 410.0/512.0, v1: 56.0/512.0 },
        });

        // ── Pipelines ───────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("minimap shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("minimap pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MMVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 2,
                },
            ],
        };
        let color_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let make_pipeline = |topo: wgpu::PrimitiveTopology, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(color_target.clone())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: topo,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let tex_pipeline =
            make_pipeline(wgpu::PrimitiveTopology::TriangleList, "minimap tri pipeline");
        let line_pipeline =
            make_pipeline(wgpu::PrimitiveTopology::LineStrip, "minimap line pipeline");

        let quad_capacity = 6 * 256; // panel + bg + buttons + markers/events
        let quad_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minimap quad vbuf"),
            size: (quad_capacity * std::mem::size_of::<MMVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let line_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("minimap line vbuf"),
            contents: bytemuck::cast_slice(&[MMVertex::default(); 5]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Start with neutral params; the caller can rotate to map angle later
        // via `set_angle` — mirrors `CMinimap::Init` (MatrixMinimap.cpp:70-73).
        let map_cx = map.world_width() * 0.5;
        let map_cy = map.world_height() * 0.5;

        Self {
            tex_pipeline,
            line_pipeline,
            bg_bind,
            white_bind,
            quad_vbuf,
            quad_capacity,
            line_vbuf,
            _bg_tex: bg_tex,
            bg_depth_tex,
            _white_tex: white_tex,
            icon_bind,
            panel_bind,
            panel_src,
            panel_uv,
            side_colors,
            zi_button,
            zo_button,
            tgt_scale: 1.0,
            icon_arrow,
            icon_flyer,
            icon_turret,
            icon_robot,
            icon_base,
            icon_factory,
            icon_point,
            events: Vec::with_capacity(MINIMAP_MAX_EVENTS),
            time_ms: 0.0,
            baked: false,
            pos_x: 0.0,
            pos_y: 0.0,
            size_x: MINIMAP_SIZE_PX,
            size_y: MINIMAP_SIZE_PX,
            center: [map_cx, map_cy],
            scale: 1.0,
            angle: 0.0,
            delta: [0.0, 0.0],
            map_sx: map.size_x as i32,
            map_sy: map.size_y as i32,
        }
    }

    /// Port of `CMinimap::SetAngle` (MatrixMinimap.cpp:159-180). We store just
    /// the scalar angle; the 2D rotation is applied on the fly in map-space.
    pub fn set_angle(&mut self, angle: f32) {
        self.angle = angle;
    }

    /// Port of `CMinimap::AddEvent` (MatrixMinimap.cpp:332-360).
    pub fn add_event(&mut self, x: f32, y: f32, color1: u32, color2: u32) {
        let ev = MinimapEvent {
            pos_in_world: [x, y],
            ttl: 1.0,
            color1,
            color2,
        };
        if self.events.len() < MINIMAP_MAX_EVENTS {
            self.events.push(ev);
            return;
        }
        if let Some((idx, _)) = self
            .events
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.ttl.partial_cmp(&b.ttl).unwrap_or(std::cmp::Ordering::Equal))
        {
            self.events[idx] = ev;
        }
    }

    /// Port of `CMinimap::Takt` / `PauseTakt` (MatrixMinimap.cpp:362-393).
    pub fn takt(&mut self, dt_ms: f32) {
        self.time_ms += dt_ms;
        if self.time_ms > 1_000_000.0 {
            self.time_ms -= 1_000_000.0;
        }

        let dt = dt_ms / MINIMAP_EVENT_TTL_MS;
        let mut i = 0;
        while i < self.events.len() {
            self.events[i].ttl -= dt;
            if self.events[i].ttl < 0.0 {
                self.events.swap_remove(i);
            } else {
                i += 1;
            }
        }

        let mul = 1.0 - 0.995f32.powf(dt_ms);
        self.scale += (self.tgt_scale - self.scale) * mul;
    }

    /// Port of `CMinimap::RenderBackground` (MatrixMinimap.cpp:855-1199).
    /// Renders the landscape orthographically from above into `bg_tex` so
    /// subsequent frames can sample it as a static texture. Call once, in the
    /// frame's command encoder, before any render pass that uses this minimap.
    ///
    /// Original camera setup (MatrixMinimap.cpp:986-1000):
    ///   - eye at (mapCX, mapCY, 1300), target (mapCX, mapCY, 1299), up=(0,-1,0)
    ///   - view X/Y columns scaled by 1/fsz (fsz = max map dim × GLOBAL_SCALE)
    ///   - ortho projection 1×1, near 1, far 10000
    /// Terrain vertices here are pre-centered on the map origin, so the
    /// translation terms for mapCX/mapCY drop out.
    pub fn bake_background(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &mut TerrainRenderer,
        map: &GameMap,
    ) {
        if self.baked {
            return;
        }

        // The bake must run in its own command submission: wgpu's
        // `queue.write_buffer` calls are all applied at the start of a submit
        // (last-write-wins for overlapping ranges), so sharing an encoder
        // with the main render would make both passes read the same — main —
        // `view_proj` in the terrain uniform buffer. Isolating the bake in
        // its own `submit` ensures the ortho VP is actually what the GPU
        // sees during the bake pass.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Minimap Bake Encoder"),
        });

        let sz = map.size_x.max(map.size_y) as f32;
        let fsz = sz * GLOBAL_SCALE;
        let s = 2.0 / fsz; // maps world span `fsz` → clip [-1, 1]
        let near = 1.0f32;
        let far = 10000.0f32;
        let eye_z = 1300.0f32;
        let inv_depth = 1.0 / (far - near);

        // Column-major Z-up-world → clip:
        //   clip.x =  s*(wx)              → right
        //   clip.y = -s*(wy)              → original up=(0,-1,0) flip
        //   clip.z = (eye_z - near - wz)/(far - near) → LH depth
        //   clip.w = 1
        let vp = glam::Mat4::from_cols_array(&[
            s, 0.0, 0.0, 0.0,
            0.0, -s, 0.0, 0.0,
            0.0, 0.0, -inv_depth, 0.0,
            0.0, 0.0, (eye_z - near) * inv_depth, 1.0,
        ]);

        // Clear to black, matching MatrixMinimap.cpp:983:
        //   Clear(D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(0,0,0), ...)
        // The water passes fill every tile inside the fsz×fsz ortho footprint,
        // so anything still black after the bake is outside the rendered area.
        let clear_color = wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };

        let color_view = self._bg_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self
            .bg_depth_tex
            .create_view(&wgpu::TextureViewDescriptor::default());

        terrain.bake_minimap(
            device,
            &mut encoder,
            queue,
            &color_view,
            &depth_view,
            vp,
            clear_color,
        );

        // The original bake path follows up with `RenderObjectToBackground`
        // for buildings. We do not have a top-down mesh pass yet, so stamp the
        // closest currently-available equivalent: the same minimap atlas icons
        // used by the live overlay, written directly into the baked texture.
        let mut bg_markers = Vec::with_capacity(map.buildings.len() * 6);
        let sz = map.size_x.max(map.size_y) as f32;
        let fsz = sz * GLOBAL_SCALE;
        let x0 = (map.size_x as f32 * GLOBAL_SCALE - fsz) * 0.5;
        let y0 = (map.size_y as f32 * GLOBAL_SCALE - fsz) * 0.5;
        let stamp_r = 6.0 / TEX_SIZE as f32;
        for b in &map.buildings {
            let u = (b.x - x0) / fsz;
            let v = (b.y - y0) / fsz;
            let px = 2.0 * u - 1.0;
            let py = 1.0 - 2.0 * v;
            let icon = if b.kind == 0 {
                self.icon_base
            } else {
                self.icon_factory
            };
            let mut color = self
                .side_colors
                .get(b.side as usize)
                .copied()
                .unwrap_or([0.85, 0.85, 0.85, 1.0]);
            color[3] = 0.35;
            Self::push_marker_clip(&mut bg_markers, [px, py], stamp_r, icon, color);
        }
        if !bg_markers.is_empty() {
            let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("minimap bg buildings"),
                contents: bytemuck::cast_slice(&bg_markers),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Minimap Background Buildings"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.tex_pipeline);
            pass.set_bind_group(0, &self.icon_bind, &[]);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.draw(0..bg_markers.len() as u32, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        self.baked = true;
    }

    // ── Coord transforms ────────────────────────────────────────────────────

    /// Port of `CMinimap::World2Map` (MatrixMinimap.cpp:276-291).
    fn world_to_map(&self, wx: f32, wy: f32) -> [f32; 2] {
        let sz = self.map_sx.max(self.map_sy) as f32;
        let fsz_inv = (1.0 / GLOBAL_SCALE) / sz;
        let half_w = self.map_sx as f32 * GLOBAL_SCALE * 0.5;
        let half_h = self.map_sy as f32 * GLOBAL_SCALE * 0.5;
        let fx = (wx - self.center[0]) * self.scale + self.center[0] - half_w;
        let fy = (wy - self.center[1]) * self.scale + self.center[1] - half_h;
        [
            self.delta[0] + self.pos_x + fx * fsz_inv * self.size_x + self.size_x * 0.5,
            self.delta[1] + self.pos_y + fy * fsz_inv * self.size_y + self.size_y * 0.5,
        ]
    }

    /// Rotate a minimap-space point by `self.angle` around the minimap center.
    /// Matches the `m_Rotation` transform (translate-to-origin, rotate, translate-back).
    fn apply_rotation(&self, p: [f32; 2]) -> [f32; 2] {
        if self.angle == 0.0 {
            return p;
        }
        let cx = self.pos_x + self.size_x * 0.5;
        let cy = self.pos_y + self.size_y * 0.5;
        let (s, c) = self.angle.sin_cos();
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        [cx + c * dx - s * dy, cy + s * dx + c * dy]
    }

    /// Inverse of `apply_rotation` — minimap-space screen point rotated back
    /// into un-rotated minimap space. Mirrors `D3DXMatrixInverse(&ri, &m_Rotation)`
    /// followed by `D3DXVec2TransformCoord(&pos, &in, &ri)` at
    /// MatrixMinimap.cpp:258-259.
    fn inverse_rotation(&self, p: [f32; 2]) -> [f32; 2] {
        if self.angle == 0.0 {
            return p;
        }
        let cx = self.pos_x + self.size_x * 0.5;
        let cy = self.pos_y + self.size_y * 0.5;
        let (s, c) = (-self.angle).sin_cos();
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        [cx + c * dx - s * dy, cy + s * dx + c * dy]
    }

    /// Port of `CMinimap::Map2World` (MatrixMinimap.cpp:250-274). Takes a
    /// screen-pixel point on the minimap and returns the world-space XY the
    /// camera would need for its link point to show that spot.
    fn map_to_world(&self, mx: f32, my: f32) -> [f32; 2] {
        let p = self.inverse_rotation([mx, my]);
        let sz = self.map_sx.max(self.map_sy) as f32;
        let fsz = GLOBAL_SCALE * sz;
        let fx = (p[0] - self.delta[0] - self.pos_x - self.size_x * 0.5) * fsz / self.size_x;
        let fy = (p[1] - self.delta[1] - self.pos_y - self.size_y * 0.5) * fsz / self.size_y;
        let half_w = self.map_sx as f32 * GLOBAL_SCALE * 0.5;
        let half_h = self.map_sy as f32 * GLOBAL_SCALE * 0.5;
        [
            (fx - self.center[0] + half_w) / self.scale + self.center[0],
            (fy - self.center[1] + half_h) / self.scale + self.center[1],
        ]
    }

    fn minimap_contains(&self, cursor_x: f32, cursor_y: f32) -> bool {
        if cursor_x < self.pos_x
            || cursor_y < self.pos_y
            || cursor_x > self.pos_x + self.size_x
            || cursor_y > self.pos_y + self.size_y
        {
            return false;
        }
        true
    }

    /// Port of `CMinimap::CalcMinimap2World` + `ButtonClick`
    /// (MatrixMinimap.cpp:1356-1379). Returns `Some(world_xy)` when the
    /// cursor is inside the minimap rect; the caller passes that straight
    /// to `camera.set_xy_strategy(tgt)` — the same two-line op the original
    /// performs at MatrixMinimap.cpp:1361.
    pub fn click_to_world(&self, cursor_x: f32, cursor_y: f32) -> Option<[f32; 2]> {
        if !self.minimap_contains(cursor_x, cursor_y) {
            return None;
        }
        Some(self.map_to_world(cursor_x, cursor_y))
    }

    fn button_hit(&self, btn: &Button, cx: f32, cy: f32) -> bool {
        let ui_scale = self.size_x / MINIMAP_SIZE_PX;
        let panel_x = 0.0;
        let panel_y = self.pos_y - MINIMAP_OFFSET_Y_IN_PANEL * ui_scale;
        let bx = panel_x + btn.local[0] * ui_scale;
        let by = panel_y + btn.local[1] * ui_scale;
        let bw = btn.local[2] * ui_scale;
        let bh = btn.local[3] * ui_scale;
        cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh
    }

    /// Dispatch a left-press at `(cx, cy)` in surface pixels. Checks buttons
    /// first, then the minimap rect. Main-rect presses begin drag immediately,
    /// matching the original minimap click/drag camera recentering.
    pub fn click(&mut self, cx: f32, cy: f32) -> MinimapClick {
        if self.button_hit(&self.zi_button, cx, cy) {
            self.zoom_in();
            return MinimapClick::ZoomIn;
        }
        if self.button_hit(&self.zo_button, cx, cy) {
            self.zoom_out();
            return MinimapClick::ZoomOut;
        }
        if let Some(tgt) = self.click_to_world(cx, cy) {
            MinimapClick::BeginDrag(tgt)
        } else {
            MinimapClick::None
        }
    }

    /// Port of `CMinimap::ButtonZoomIn` (MatrixMinimap.cpp:1344-1348).
    /// `scale *= 1.8`, clamped to [MINIMAP_MIN_SCALE, MINIMAP_MAX_SCALE].
    pub fn zoom_in(&mut self) {
        self.tgt_scale = (self.tgt_scale * MINIMAP_ZOOM_IN_FACTOR)
            .clamp(MINIMAP_MIN_SCALE, MINIMAP_MAX_SCALE);
    }

    /// Port of `CMinimap::ButtonZoomOut` (MatrixMinimap.cpp:1350-1354).
    /// `scale *= 0.5`, clamped the same way.
    pub fn zoom_out(&mut self) {
        self.tgt_scale = (self.tgt_scale * MINIMAP_ZOOM_OUT_FACTOR)
            .clamp(MINIMAP_MIN_SCALE, MINIMAP_MAX_SCALE);
    }

    /// Port of `CMinimap::BeforeDraw` (MatrixMinimap.cpp:182-248) — computes
    /// `m_Delta` so the map square stays pinned to the minimap rect when the
    /// world is scaled, then emits the four corner vertices.
    fn before_draw(&mut self) -> [MMVertex; 6] {
        self.delta = [0.0, 0.0];

        let sz = self.map_sx.max(self.map_sy);
        let fsz = sz as f32 * GLOBAL_SCALE;
        let x0 = (self.map_sx as f32 - sz as f32) * 0.5 * GLOBAL_SCALE;
        let y0 = (self.map_sy as f32 - sz as f32) * 0.5 * GLOBAL_SCALE;

        let lt = self.world_to_map(x0, y0);
        let rb = self.world_to_map(x0 + fsz, y0 + fsz);
        if lt[0] > self.pos_x {
            self.delta[0] = self.pos_x - lt[0];
        }
        if lt[1] > self.pos_y {
            self.delta[1] = self.pos_y - lt[1];
        }
        if rb[0] < self.pos_x + self.size_x {
            self.delta[0] = self.pos_x + self.size_x - rb[0];
        }
        if rb[1] < self.pos_y + self.size_y {
            self.delta[1] = self.pos_y + self.size_y - rb[1];
        }

        // Recompute with delta applied, then rotate. Matches the vertex layout
        // in MatrixMinimap.cpp:230-244 (tri-strip BL/TL/BR/TR, re-emitted as
        // two triangles below because we use TriangleList).
        let lt = self.world_to_map(x0, y0);
        let rb = self.world_to_map(x0 + fsz, y0 + fsz);
        let p_bl = self.apply_rotation([lt[0], rb[1]]);
        let p_tl = self.apply_rotation([lt[0], lt[1]]);
        let p_br = self.apply_rotation([rb[0], rb[1]]);
        let p_tr = self.apply_rotation([rb[0], lt[1]]);

        let v = |p: [f32; 2], uv: [f32; 2]| MMVertex {
            pos: p,
            uv,
            color: [1.0, 1.0, 1.0, 1.0],
        };

        [
            v(p_bl, [0.0, 1.0]),
            v(p_tl, [0.0, 0.0]),
            v(p_tr, [1.0, 0.0]),
            v(p_bl, [0.0, 1.0]),
            v(p_tr, [1.0, 0.0]),
            v(p_br, [1.0, 1.0]),
        ]
    }

    /// Emit a textured sprite centered on `px` with half-side `radius`.
    /// Vertex order and UV assignment mirror MatrixMinimap.cpp:746-754
    /// (tri-strip BL/TL/BR/TR → two triangles here because we use
    /// TriangleList), so the atlas sub-rect orients the sprite the same way.
    fn push_marker(
        out: &mut Vec<MMVertex>,
        px: [f32; 2],
        radius: f32,
        icon: IconUv,
        color: [f32; 4],
    ) {
        let (x, y, r) = (px[0], px[1], radius);
        let v = |p: [f32; 2], uv: [f32; 2]| MMVertex { pos: p, uv, color };
        let p_bl = [x - r, y + r];
        let p_tl = [x - r, y - r];
        let p_br = [x + r, y + r];
        let p_tr = [x + r, y - r];
        let uv_bl = [icon.u0, icon.v1];
        let uv_tl = [icon.u0, icon.v0];
        let uv_br = [icon.u1, icon.v1];
        let uv_tr = [icon.u1, icon.v0];
        out.push(v(p_bl, uv_bl));
        out.push(v(p_tl, uv_tl));
        out.push(v(p_tr, uv_tr));
        out.push(v(p_bl, uv_bl));
        out.push(v(p_tr, uv_tr));
        out.push(v(p_br, uv_br));
    }

    fn push_marker_clip(
        out: &mut Vec<MMVertex>,
        clip: [f32; 2],
        radius_clip: f32,
        icon: IconUv,
        color: [f32; 4],
    ) {
        let (x, y, r) = (clip[0], clip[1], radius_clip);
        let v = |p: [f32; 2], uv: [f32; 2]| MMVertex { pos: p, uv, color };
        let uv_bl = [icon.u0, icon.v1];
        let uv_tl = [icon.u0, icon.v0];
        let uv_br = [icon.u1, icon.v1];
        let uv_tr = [icon.u1, icon.v0];
        out.push(v([x - r, y - r], uv_bl));
        out.push(v([x - r, y + r], uv_tl));
        out.push(v([x + r, y + r], uv_tr));
        out.push(v([x - r, y - r], uv_bl));
        out.push(v([x + r, y + r], uv_tr));
        out.push(v([x + r, y - r], uv_br));
    }

    fn push_oriented_marker(
        out: &mut Vec<MMVertex>,
        px: [f32; 2],
        radius: f32,
        dir: [f32; 2],
        icon: IconUv,
        color: [f32; 4],
    ) {
        let dlen = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-6);
        let dx = dir[0] / dlen;
        let dy = dir[1] / dlen;
        let p0 = [px[0] + radius * (dx + dy), px[1] - radius * (dx - dy)];
        let p1 = [px[0] + radius * (dx - dy), px[1] + radius * (dx + dy)];
        let p2 = [px[0] - radius * (dx - dy), px[1] - radius * (dx + dy)];
        let p3 = [px[0] - radius * (dx + dy), px[1] + radius * (dx - dy)];
        let v = |p: [f32; 2], uv: [f32; 2]| MMVertex { pos: p, uv, color };
        out.push(v(p0, [icon.u0, icon.v1]));
        out.push(v(p1, [icon.u0, icon.v0]));
        out.push(v(p3, [icon.u1, icon.v0]));
        out.push(v(p0, [icon.u0, icon.v1]));
        out.push(v(p3, [icon.u1, icon.v0]));
        out.push(v(p2, [icon.u1, icon.v1]));
    }

    /// Render. `pass` must target the swapchain with LoadOp::Load, no depth.
    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        screen_w: f32,
        screen_h: f32,
        map: &GameMap,
        camera: &Camera,
    ) {
        if screen_w <= 1.0 || screen_h <= 1.0 {
            return;
        }

        // Port of the `if/MiniM` panel layout from robots.dat + the bottom-
        // anchor rule at CInterface.cpp:176-183:
        //   MiniM.xPos = 0, MiniM.yPos = 559 in 1024×768 base
        //   → final (0, screen_h - (768 - 559)) = (0, screen_h - 209).
        // The minimap landscape renders at panel-local (+13, +51, 145, 145)
        // from CInterface.cpp:557's `SetOutParams(xPos+13, yPos+51, 145,145)`.
        // Every pixel value scales by `screen_h / 768` so the 1024×768 design
        // doesn't shrink to invisibility on modern resolutions.
        let ui_scale = screen_h / DESIGN_HEIGHT_PX;
        let panel_w = MINIMAP_PANEL_W * ui_scale;
        let panel_h = MINIMAP_PANEL_H * ui_scale;
        let panel_x = 0.0;
        let panel_y =
            screen_h - (DESIGN_HEIGHT_PX - MINIMAP_PANEL_Y_BASE) * ui_scale;
        self.size_x = MINIMAP_SIZE_PX * ui_scale;
        self.size_y = MINIMAP_SIZE_PX * ui_scale;
        self.pos_x = panel_x + MINIMAP_OFFSET_X * ui_scale;
        self.pos_y = panel_y + MINIMAP_OFFSET_Y_IN_PANEL * ui_scale;

        // MatrixMap.cpp:1261: `SetOutParams(m_Camera.GetXYStrategy())` is
        // called every frame, so the minimap is always centered on what the
        // camera is looking at. The near/far edges get pinned back inside the
        // rect by `m_Delta` in BeforeDraw.
        let (cx, cy) = camera.strategy_xy();
        self.center = [cx, cy];

        // Quad layout in `quad_vbuf`:
        //   [ 0.. 6] MiniM panel art (`mmp` sub-rect of interface1)
        //   [ 6..12] landscape background (bake output)
        //   [12..18] zoom-in button (interface1 sub-rect)
        //   [18..24] zoom-out button (interface1 sub-rect)
        //   [24..  ] building markers
        let mut head = [MMVertex::default(); 24];
        // MiniM panel art quad (at panel_x, panel_y, panel_w × panel_h).
        fill_uv_rect_quad(
            &mut head[0..6],
            [panel_x, panel_y, panel_w, panel_h],
            self.panel_uv,
            [1.0, 1.0, 1.0, 1.0],
            screen_w,
            screen_h,
        );
        // Minimap landscape quad (with rotation + delta math from before_draw).
        let bg_verts = self.before_draw();
        for (i, v) in bg_verts.iter().enumerate() {
            head[6 + i] = MMVertex {
                pos: pixel_to_clip(v.pos, screen_w, screen_h),
                uv: v.uv,
                color: v.color,
            };
        }
        // Zoom buttons (panel-local coords scaled by ui_scale).
        let place_button = |btn: &Button| -> [f32; 4] {
            [
                panel_x + btn.local[0] * ui_scale,
                panel_y + btn.local[1] * ui_scale,
                btn.local[2] * ui_scale,
                btn.local[3] * ui_scale,
            ]
        };
        fill_uv_rect_quad(
            &mut head[12..18],
            place_button(&self.zi_button),
            self.zi_button.uv,
            [1.0, 1.0, 1.0, 1.0],
            screen_w,
            screen_h,
        );
        fill_uv_rect_quad(
            &mut head[18..24],
            place_button(&self.zo_button),
            self.zo_button.uv,
            [1.0, 1.0, 1.0, 1.0],
            screen_w,
            screen_h,
        );
        queue.write_buffer(&self.quad_vbuf, 0, bytemuck::cast_slice(&head));
        // `panel_src` is loaded at construction; we keep it around so future
        // layout changes can reference the raw sub-rect without reparsing.
        let _ = self.panel_src;

        // Building markers. Mirrors MatrixMinimap.cpp:700-762 — kind 0 = Base
        // → MMT_BASE (radius MINIMAP_BUILDING_BASE_R=8), all other kinds use
        // MMT_FACTORY (radius MINIMAP_BUILDING_R=8). Tint is `m_Color & alpha
        // | GetSideColorMM(side)`. Radii scale with UI so sprites stay
        // proportional to the enlarged minimap rect.
        let marker_r = MINIMAP_BUILDING_R * ui_scale;
        let mut markers: Vec<MMVertex> =
            Vec::with_capacity((map.buildings.len() + self.events.len()) * 6);
        let mut point_markers: Vec<MMVertex> = Vec::with_capacity(self.events.len() * 6);
        for b in &map.buildings {
            let mut px_px = self.world_to_map(b.x, b.y);
            px_px = self.apply_rotation(px_px);
            px_px[0] = px_px[0].floor() - 0.5;
            px_px[1] = px_px[1].floor() - 0.5;
            let icon = if b.kind == 0 {
                self.icon_base
            } else {
                self.icon_factory
            };
            let color = self
                .side_colors
                .get(b.side as usize)
                .copied()
                .unwrap_or([0.85, 0.85, 0.85, 1.0]);
            let radius = if b.kind == 0 {
                MINIMAP_BUILDING_BASE_R * ui_scale
            } else {
                marker_r
            };
            Self::push_marker(&mut markers, px_px, radius, icon, color);
        }
        for ev in &self.events {
            let pos = self.apply_rotation(self.world_to_map(ev.pos_in_world[0], ev.pos_in_world[1]));
            if pos[0] < self.pos_x
                || pos[0] > self.pos_x + self.size_x
                || pos[1] < self.pos_y
                || pos[1] > self.pos_y + self.size_y
            {
                let clipped = self.clip_to_rect(pos);
                let dir = [
                    self.pos_x + self.size_x * 0.5 - clipped[0],
                    self.pos_y + self.size_y * 0.5 - clipped[1],
                ];
                let flash = flash_color(ev.color1, ev.color2, self.time_ms);
                Self::push_oriented_marker(
                    &mut markers,
                    clipped,
                    MINIMAP_OUT_INDICATOR_R * ui_scale,
                    dir,
                    self.icon_arrow,
                    flash,
                );
            } else {
                let k = 1.0 - ev.ttl;
                let r = lerp(MINIMAP_EVENT_R1, MINIMAP_EVENT_R2, k) * ui_scale;
                let color = lerp_color_u32(ev.color1, ev.color2, k);
                Self::push_marker(&mut point_markers, pos, r, self.icon_point, color);
            }
        }
        // Clamp to capacity so we never overrun the vertex buffer.
        markers.extend(point_markers);
        let max_marker_verts = self
            .quad_capacity
            .saturating_sub(24)
            .min(markers.len() / 6 * 6);
        let markers = &markers[..max_marker_verts];

        // Upload markers immediately after the 6-vert background.
        let markers_clip: Vec<MMVertex> = markers
            .iter()
            .map(|v| MMVertex {
                pos: pixel_to_clip(v.pos, screen_w, screen_h),
                uv: v.uv,
                color: v.color,
            })
            .collect();
        if !markers_clip.is_empty() {
            queue.write_buffer(
                &self.quad_vbuf,
                (24 * std::mem::size_of::<MMVertex>()) as u64,
                bytemuck::cast_slice(&markers_clip),
            );
        }

        // Camera frustum loop — 4 world-space points on z=WATER_LEVEL, closed.
        let quad = camera.frustum_bounds_on_plane_zup(WATER_LEVEL);
        let mut loop_verts = [MMVertex::default(); 5];
        for i in 0..4 {
            // Camera returns centered world coords; the minimap math uses
            // uncentered. Shift by (map_cx, map_cy) before projecting.
            let wx = quad[i].x + self.map_sx as f32 * GLOBAL_SCALE * 0.5;
            let wy = quad[i].y + self.map_sy as f32 * GLOBAL_SCALE * 0.5;
            let mut px = self.world_to_map(wx, wy);
            px = self.apply_rotation(px);
            // Clamp to the minimap rect so the loop never leaks into the HUD.
            let min_x = self.pos_x;
            let max_x = self.pos_x + self.size_x;
            let min_y = self.pos_y;
            let max_y = self.pos_y + self.size_y;
            px[0] = px[0].clamp(min_x, max_x);
            px[1] = px[1].clamp(min_y, max_y);
            loop_verts[i] = MMVertex {
                pos: pixel_to_clip(px, screen_w, screen_h),
                uv: [0.5, 0.5],
                color: MINIMAP_CAM_COLOR,
            };
        }
        loop_verts[4] = loop_verts[0];
        queue.write_buffer(&self.line_vbuf, 0, bytemuck::cast_slice(&loop_verts));

        // ── Draws ──
        pass.set_pipeline(&self.tex_pipeline);
        pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));

        // 1) MiniM panel art from interface1 (sub-rect `mmp`). No scissor —
        //    the panel frame extends beyond the 145×145 inner landscape rect.
        pass.set_bind_group(0, &self.panel_bind, &[]);
        pass.draw(0..6, 0..1);

        // 2) Landscape + markers + frustum are clipped to the inner rect so
        //    zoom > 1 doesn't spill the world outside the panel frame. Ports
        //    the `SetViewport(m_Viewport)` clip in MatrixMinimap.cpp:595.
        let sx = self.pos_x.floor().max(0.0) as u32;
        let sy = self.pos_y.floor().max(0.0) as u32;
        let sw_u = (self.size_x.ceil() as i32)
            .max(0)
            .min(screen_w as i32 - sx as i32) as u32;
        let sh_u = (self.size_y.ceil() as i32)
            .max(0)
            .min(screen_h as i32 - sy as i32) as u32;
        if sw_u > 0 && sh_u > 0 {
            pass.set_scissor_rect(sx, sy, sw_u, sh_u);
        }

        pass.set_bind_group(0, &self.bg_bind, &[]);
        pass.draw(6..12, 0..1);

        if !markers_clip.is_empty() {
            pass.set_bind_group(0, &self.icon_bind, &[]);
            pass.draw(24..(24 + markers_clip.len() as u32), 0..1);
        }

        pass.set_pipeline(&self.line_pipeline);
        pass.set_bind_group(0, &self.white_bind, &[]);
        pass.set_vertex_buffer(0, self.line_vbuf.slice(..));
        pass.draw(0..5, 0..1);

        // Reset scissor before drawing the zoom buttons — they live on the
        // panel art, *outside* the inner landscape rect.
        pass.set_scissor_rect(0, 0, screen_w as u32, screen_h as u32);
        pass.set_pipeline(&self.tex_pipeline);
        pass.set_bind_group(0, &self.panel_bind, &[]);
        pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
        pass.draw(12..24, 0..1);
    }

    fn clip_to_rect(&self, input: [f32; 2]) -> [f32; 2] {
        let half_w = self.size_y * 0.5;
        let half_h = self.size_x * 0.5;
        let cx = self.pos_x + half_w;
        let cy = self.pos_y + half_h;
        let mut dir = [cx - input[0], cy - input[1]];
        loop {
            if dir[0] > half_w {
                dir[1] = dir[1] / dir[0] * half_w;
                dir[0] = half_w;
                continue;
            }
            if dir[0] < -half_w {
                dir[1] = -dir[1] / dir[0] * half_w;
                dir[0] = -half_w;
                continue;
            }
            if dir[1] > half_h {
                dir[0] = dir[0] / dir[1] * half_h;
                dir[1] = half_h;
                continue;
            }
            if dir[1] < -half_h {
                dir[0] = -dir[0] / dir[1] * half_h;
                dir[1] = -half_h;
                continue;
            }
            break;
        }
        [cx - dir[0] * MINIMAP_OUT_SCALE, cy - dir[1] * MINIMAP_OUT_SCALE]
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn unpack_argb(c: u32) -> [f32; 4] {
    [
        ((c >> 16) & 0xFF) as f32 / 255.0,
        ((c >> 8) & 0xFF) as f32 / 255.0,
        (c & 0xFF) as f32 / 255.0,
        ((c >> 24) & 0xFF) as f32 / 255.0,
    ]
}

fn lerp_color_u32(c1: u32, c2: u32, t: f32) -> [f32; 4] {
    let a = unpack_argb(c1);
    let b = unpack_argb(c2);
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
        lerp(a[3], b[3], t),
    ]
}

fn flash_color(c1: u32, c2: u32, time_ms: f32) -> [f32; 4] {
    let t = ((std::f32::consts::TAU * time_ms / MINIMAP_FLASH_PERIOD_MS).sin() * 0.5) + 0.5;
    lerp_color_u32(c1, c2, t)
}

/// Map top-left-origin pixel position → clip space (Y-up in clip).
fn pixel_to_clip(px: [f32; 2], sw: f32, sh: f32) -> [f32; 2] {
    [2.0 * px[0] / sw - 1.0, 1.0 - 2.0 * px[1] / sh]
}

/// Fill a 6-vertex slot with a textured pixel-space rect (`[x, y, w, h]`,
/// top-left origin) mapped to a sub-rect of the currently-bound texture.
/// Triangle order: BL / TL / TR + BL / TR / BR, matching CIFaceElement.cpp:
/// 74-79 when it emits its pre-transformed triangle strip.
fn fill_uv_rect_quad(
    dst: &mut [MMVertex],
    rect: [f32; 4],
    uv: IconUv,
    color: [f32; 4],
    sw: f32,
    sh: f32,
) {
    let [x, y, w, h] = rect;
    let to_clip = |p: [f32; 2]| pixel_to_clip(p, sw, sh);
    let v = |p: [f32; 2], uv: [f32; 2]| MMVertex {
        pos: to_clip(p),
        uv,
        color,
    };
    dst[0] = v([x, y + h], [uv.u0, uv.v1]);
    dst[1] = v([x, y], [uv.u0, uv.v0]);
    dst[2] = v([x + w, y], [uv.u1, uv.v0]);
    dst[3] = v([x, y + h], [uv.u0, uv.v1]);
    dst[4] = v([x + w, y], [uv.u1, uv.v0]);
    dst[5] = v([x + w, y + h], [uv.u1, uv.v1]);
}

/// Pixel-space rect parsed from a `Minimap` blockpar value, before we know
/// the atlas size. Ports the `x,y,w,h` storage of `CMinimap::SMMTex::Load`
/// (MatrixMinimap.cpp:21-40) before the normalization step.
#[derive(Copy, Clone, Debug)]
struct PxRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}
impl PxRect {
    fn normalize(self, atlas_w: f32, atlas_h: f32) -> IconUv {
        IconUv {
            u0: self.x / atlas_w,
            v0: self.y / atlas_h,
            u1: (self.x + self.w) / atlas_w,
            v1: (self.y + self.h) / atlas_h,
        }
    }
}

/// Resolve a Button inside `if/MiniM` by its Name parameter. Mirrors
/// `CIFaceButton` loading in CInterface.cpp:430-467 — only the `sNormal`
/// state is ported since we don't animate button hover/press yet.
fn parse_minim_button(matrix_data: &Storage, name: &str) -> Option<Button> {
    let minim = matrix_data.block_record("if", "MiniM")?;
    let names = matrix_data.get_buf(&minim, "2")?;
    let records = matrix_data.get_buf(&minim, "3")?;
    for i in 0..names.arrays_count() {
        if names.get_as_wstr(i) != "Button" {
            continue;
        }
        let child = records.get_as_wstr(i);
        if matrix_data.block_param(&child, "Name").as_deref() != Some(name) {
            continue;
        }
        let x = parse_f32_param(matrix_data, &child, "xPos")?;
        let y = parse_f32_param(matrix_data, &child, "yPos")?;
        let w = parse_f32_param(matrix_data, &child, "xSize")?;
        let h = parse_f32_param(matrix_data, &child, "ySize")?;
        let sx = parse_f32_param(matrix_data, &child, "sNormalX")?;
        let sy = parse_f32_param(matrix_data, &child, "sNormalY")?;
        let tw = parse_f32_param(matrix_data, &child, "sNormalWidth")?;
        let th = parse_f32_param(matrix_data, &child, "sNormalHeight")?;
        return Some(Button {
            local: [x, y, w, h],
            uv: SubRect {
                x: sx,
                y: sy,
                w,
                h,
                tex_w: tw,
                tex_h: th,
            }
            .to_uv(),
        });
    }
    None
}

/// Resolve the `if / MiniM / Static {Name=mmp}` sub-rect — the panel art
/// that frames the minimap on screen. Ports CInterface.cpp:545-596 reading
/// `sNormalX`, `sNormalY`, `xSize`, `ySize`, `sNormalWidth`, `sNormalHeight`.
fn parse_mmp_static(matrix_data: &Storage) -> Option<SubRect> {
    let minim = matrix_data.block_record("if", "MiniM")?;
    // Walk children of MiniM; find the Static whose "Name" param equals "mmp".
    let names = matrix_data.get_buf(&minim, "2")?;
    let records = matrix_data.get_buf(&minim, "3")?;
    for i in 0..names.arrays_count() {
        if names.get_as_wstr(i) == "Static" {
            let child = records.get_as_wstr(i);
            if matrix_data
                .block_param(&child, "Name")
                .as_deref()
                == Some("mmp")
            {
                let x = parse_f32_param(matrix_data, &child, "sNormalX")?;
                let y = parse_f32_param(matrix_data, &child, "sNormalY")?;
                let w = parse_f32_param(matrix_data, &child, "xSize")?;
                let h = parse_f32_param(matrix_data, &child, "ySize")?;
                let tw = parse_f32_param(matrix_data, &child, "sNormalWidth")?;
                let th = parse_f32_param(matrix_data, &child, "sNormalHeight")?;
                return Some(SubRect {
                    x,
                    y,
                    w,
                    h,
                    tex_w: tw,
                    tex_h: th,
                });
            }
        }
    }
    None
}

fn parse_f32_param(matrix_data: &Storage, rec: &str, key: &str) -> Option<f32> {
    matrix_data
        .block_param(rec, key)
        .and_then(|s| s.trim().parse::<f32>().ok())
}

/// Parse the `Minimap` block from robots.dat. Each entry's value is a
/// comma-separated `texture,x,y,w,h` string (MatrixMinimap.cpp:21-40).
/// Returns the raw pixel rects; the caller normalizes against the atlas size.
fn parse_minimap_block(
    matrix_data: &Storage,
) -> (u32, std::collections::HashMap<&'static str, PxRect>) {
    let mut rects = std::collections::HashMap::new();
    let Some(mm_rec) = matrix_data.block_record("da", "Minimap") else {
        log::warn!("minimap: no `Minimap` block in robots.dat — icons will be blank");
        return (128, rects);
    };
    let keys: &[&'static str] = &["point", "arrow", "flyer", "robot", "turret", "base", "factory"];
    for key in keys {
        let Some(val) = matrix_data.block_param(&mm_rec, key) else {
            continue;
        };
        let parts: Vec<&str> = val.split(',').map(|s| s.trim()).collect();
        if parts.len() < 5 {
            continue;
        }
        let x = parts[1].parse::<f32>().ok();
        let y = parts[2].parse::<f32>().ok();
        let w = parts[3].parse::<f32>().ok();
        let h = parts[4].parse::<f32>().ok();
        if let (Some(x), Some(y), Some(w), Some(h)) = (x, y, w, h) {
            rects.insert(*key, PxRect { x, y, w, h });
        }
    }
    // The icon atlas (radarIcons.dds) is 128×128 in shipping data; the
    // caller normalizes with the *actual* image dims, but we return 128 as
    // a sensible fallback when the atlas fails to load.
    (128, rects)
}

/// Parse the per-side minimap colors from robots.dat. The `Side` block
/// entries are keyed by side id and hold values of the form
/// "Name,R,G,B,Minimap,R_MM,G_MM,B_MM" (MatrixMap.cpp:1014-1020).
fn parse_side_colors_mm(matrix_data: &Storage) -> [[f32; 4]; 8] {
    let mut out = [[0.85f32, 0.85, 0.85, 1.0]; 8];
    let Some(side) = matrix_data.block_record("da", "Side") else {
        return out;
    };
    let Some(keys) = matrix_data.get_buf(&side, "0") else {
        return out;
    };
    let Some(vals) = matrix_data.get_buf(&side, "1") else {
        return out;
    };
    for i in 0..keys.arrays_count() {
        let id: usize = match keys.get_as_wstr(i).parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if id >= out.len() {
            continue;
        }
        // Value format: "Name,R,G,B,Minimap,R_MM,G_MM,B_MM".
        let raw = vals.get_as_wstr(i);
        let parts: Vec<&str> = raw.split(',').collect();
        if parts.len() < 8 {
            continue;
        }
        let r = parts[5].trim().parse::<f32>().unwrap_or(255.0) / 255.0;
        let g = parts[6].trim().parse::<f32>().unwrap_or(255.0) / 255.0;
        let b = parts[7].trim().parse::<f32>().unwrap_or(255.0) / 255.0;
        out[id] = [r, g, b, 1.0];
    }
    out
}

const SHADER_SRC: &str = r#"
struct VIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@vertex
fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    out.pos = vec4<f32>(in.pos, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let t = textureSample(tex, samp, in.uv);
    return t * in.color;
}
"#;
