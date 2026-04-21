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
//!   - Building markers colored by side (MatrixMinimap.cpp:679-763).
//!   - Camera frustum projected onto the water plane and drawn as a 4-line
//!     LINESTRIP in `MINIMAP_CAM_COLOR` (MatrixMinimap.cpp:794-841).
//!   - Layout from `CInterface.cpp:557` — 145×145 at panel offset (+13, +51)
//!     with the bottom-anchor rule from `CInterface.cpp:176-183`.
//!   - Per-frame `m_Center = camera.GetXYStrategy()` (MatrixMap.cpp:1261).
//!
//! Intentionally skipped (feature gaps, not fidelity):
//!   - Events / ping overlays (MatrixMinimap.cpp:332-360, 638-678) — require
//!     a live event system that hasn't been ported yet.
//!   - In-robot `DrawRadar` — arcade mode isn't ported.
//!   - Icon atlas from `robots.dat` `Minimap` block — solid squares stand in
//!     for `MMT_BASE` / `MMT_FACTORY` sprites until the atlas loader lands.
//!   - Disk-cached background PNG (irrelevant in-browser).

use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::game::map::{GameMap, GLOBAL_SCALE};
use crate::renderer::camera::Camera;
use crate::renderer::terrain::TerrainRenderer;
use crate::renderer::texture::{create_texture_from_rgba, decode_texture_bytes};

/// Matches `MINIMAP_SIZE` in MatrixMinimap.hpp:14.
const TEX_SIZE: u32 = 512;

/// Pixel dimensions from `CInterface.cpp:557`:
///   SetOutParams(m_xPos+13, m_yPos+51, 145, 145, ...)
/// The outer panel is anchored bottom-left of a 1024×768 design; the minimap
/// sits 13 px from the panel's left edge and 51 px from its top edge.
const MINIMAP_SIZE_PX: f32 = 145.0;
/// Offset from bottom-left of the screen in the base 1024×768 layout:
///   x = panel_x + 13, y = panel_y + 51, panel_y ≈ 570 in 768-space.
/// With the right/bottom anchoring rule in CInterface.cpp:176-183, the
/// final position becomes (13, screen_h - (768 - 621)) = (13, screen_h - 147).
const MINIMAP_OFFSET_X: f32 = 13.0;
const MINIMAP_OFFSET_FROM_BOTTOM: f32 = 147.0;

/// `MINIMAP_CAM_COLOR = 0xFF30FFFF` (MatrixMinimap.cpp:19), D3DCOLOR ARGB.
const MINIMAP_CAM_COLOR: [f32; 4] = [48.0 / 255.0, 1.0, 1.0, 1.0];

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

pub struct Minimap {
    tex_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    bg_bind: wgpu::BindGroup,
    icon_bind: wgpu::BindGroup,
    white_bind: wgpu::BindGroup,

    // Dynamic per-frame vertex streams.
    quad_vbuf: wgpu::Buffer, // TriangleList, grows via write_buffer
    quad_capacity: usize,    // vertex slots allocated
    line_vbuf: wgpu::Buffer, // LineStrip, fixed 5 verts for the frustum loop

    // Background texture kept alive because the bind group holds only a view.
    _bg_tex: wgpu::Texture,
    bg_depth_tex: wgpu::Texture,
    _white_tex: wgpu::Texture,

    /// Sub-rects for MMT_BASE / MMT_FACTORY / MMT_POINT, parsed from the
    /// `Minimap` block in robots.dat (MatrixMinimap.cpp:50-65).
    icon_base: IconUv,
    icon_factory: IconUv,
    // Reserved for the event-ping overlay (MatrixMinimap.cpp:638-678);
    // kept on the struct so bake code stays complete even before events land.
    #[allow(dead_code)]
    icon_point: IconUv,

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
        let icon_base = icon_rects
            .get("base")
            .copied()
            .map(|r| r.normalize(atlas_w, atlas_h))
            .unwrap_or_else(|| IconUv {
                u0: 0.0,
                v0: 0.0,
                u1: 1.0,
                v1: 1.0,
            });
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

        let quad_capacity = 6 * 128; // 128 textured quads (bg + markers)
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
            icon_base,
            icon_factory,
            icon_point,
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

        // Pin to bottom-left per the UI layout (CInterface.cpp:557):
        //   (m_xPos + 13, m_yPos + 51, 145, 145)
        // with the bottom anchor rule at CInterface.cpp:180-183.
        self.pos_x = MINIMAP_OFFSET_X;
        self.pos_y = screen_h - MINIMAP_OFFSET_FROM_BOTTOM;

        // MatrixMap.cpp:1261: `SetOutParams(m_Camera.GetXYStrategy())` is
        // called every frame, so the minimap is always centered on what the
        // camera is looking at. The near/far edges get pinned back inside the
        // rect by `m_Delta` in BeforeDraw.
        let (cx, cy) = camera.strategy_xy();
        self.center = [cx, cy];

        // Background first, two triangles.
        let bg_verts = self.before_draw();
        let mut bg_clip = [MMVertex::default(); 6];
        for (i, v) in bg_verts.iter().enumerate() {
            bg_clip[i] = MMVertex {
                pos: pixel_to_clip(v.pos, screen_w, screen_h),
                uv: v.uv,
                color: v.color,
            };
        }
        queue.write_buffer(&self.quad_vbuf, 0, bytemuck::cast_slice(&bg_clip));

        // Building markers. Mirrors MatrixMinimap.cpp:700-762 — kind 0 = Base
        // → MMT_BASE (radius MINIMAP_BUILDING_BASE_R=8), all other kinds use
        // MMT_FACTORY (radius MINIMAP_BUILDING_R=8). Tint is `m_Color & alpha
        // | GetSideColorMM(side)`.
        let mut markers: Vec<MMVertex> = Vec::with_capacity(map.buildings.len() * 6);
        for b in &map.buildings {
            let px_px = self.world_to_map(b.x, b.y);
            let px_px = self.apply_rotation(px_px);
            let (icon, r) = if b.kind == 0 {
                (self.icon_base, 8.0)
            } else {
                (self.icon_factory, 8.0)
            };
            let color = side_color_mm(b.side);
            Self::push_marker(&mut markers, px_px, r, icon, color);
        }
        // Clamp to capacity so we never overrun the vertex buffer.
        let max_marker_verts = self
            .quad_capacity
            .saturating_sub(6)
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
                (6 * std::mem::size_of::<MMVertex>()) as u64,
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
        // 1) Background quad (textured)
        pass.set_pipeline(&self.tex_pipeline);
        pass.set_bind_group(0, &self.bg_bind, &[]);
        pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
        pass.draw(0..6, 0..1);

        // 2) Building markers — sampled from the icon atlas (bind `icon_bind`)
        //    so each sprite uses its real UV sub-rect from `radarIcons.dds`.
        if !markers_clip.is_empty() {
            pass.set_bind_group(0, &self.icon_bind, &[]);
            pass.draw(6..(6 + markers_clip.len() as u32), 0..1);
        }

        // 3) Camera frustum outline (line strip)
        pass.set_pipeline(&self.line_pipeline);
        pass.set_bind_group(0, &self.white_bind, &[]);
        pass.set_vertex_buffer(0, self.line_vbuf.slice(..));
        pass.draw(0..5, 0..1);
    }
}

/// Map top-left-origin pixel position → clip space (Y-up in clip).
fn pixel_to_clip(px: [f32; 2], sw: f32, sh: f32) -> [f32; 2] {
    [2.0 * px[0] / sw - 1.0, 1.0 - 2.0 * px[1] / sh]
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

/// Side color table for the minimap. Mirrors `MatrixMap.cpp:1014-1020` —
/// the original reads these per-map from the CMAP `sides` block; we inline
/// the SR2 defaults until the Rust port parses that block.
fn side_color_mm(side: u8) -> [f32; 4] {
    let rgb = match side {
        1 => [1.00, 0.86, 0.15], // yellow (player)
        2 => [0.30, 0.55, 1.00], // blue
        3 => [1.00, 0.30, 0.30], // red
        4 => [0.40, 0.95, 0.40], // green
        5 => [0.95, 0.40, 0.95], // magenta
        _ => [0.75, 0.75, 0.75], // neutral
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
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
