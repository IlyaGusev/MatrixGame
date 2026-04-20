//! Sky rendering — ports DrawSky (MatrixMap.cpp:2020-2189).
//!
//! Two parts:
//!  * **Skybox pass**: four textured walls laid out as vertical strips of a
//!    single panoramic texture (Fore / Rite / Back / Left), drawn with a
//!    rotation-only view + shallow perspective (`CalcSkyMatrix`).
//!  * **Gradient pass**: screen-space fade along the horizon line computed
//!    from camera direction (MatrixVisiCalc.cpp:609-626). The top color
//!    fades from transparent to the sky color so the gradient blends into
//!    the skybox instead of clipping against it.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::game::common::unpack_rgb;
use crate::renderer::camera::Camera;
use crate::renderer::texture::{
    create_texture_from_rgba, create_texture_from_rgba_mipped, decode_texture_bytes,
};

/// MAX_VIEW_DISTANCE from MatrixCamera.cpp:13.
const MAX_VIEW_DISTANCE: f32 = 4000.0;
/// SH1 = g_ScreenY * 0.270416... (MatrixMap.cpp:2144).
const SH1_FRAC: f32 = 0.270_416_68;
/// SH2 = g_ScreenY * 0.07 (MatrixMap.cpp:2145).
const SH2_FRAC: f32 = 0.07;

/// Matches CAM_HFOV from MatrixCamera.hpp:51 — used only by the skybox
/// perspective so skybox walls project onto the frustum exactly once per face.
const SKY_HFOV: f32 = std::f32::consts::PI / 3.0;

/// Sky face names in the order the Rust port's skybox geometry expects them:
/// Fore (+Y), Rite (+X), Back (-Y), Left (-X). Matches the C++ SKY_FORE /
/// SKY_RITE / SKY_BACK / SKY_LEFT enum ordering (MatrixMap.cpp:2071-2098)
/// and the `Sky` block key names in `cfg/robots/data.txt`.
const FACE_CONFIG_KEYS: [&str; 4] = ["Fore", "Right", "Back", "Left"];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GradientVertex {
    position: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BoxVertex {
    /// XY of the cube corner in the skybox's local space. Z is recovered in
    /// the shader from `bottom_w` and the per-frame `cut_dn` uniform.
    xy: [f32; 2],
    /// 0 for a top corner, 1 for a bottom corner. The shader uses this to
    /// interpolate both Z (top=1.0, bottom=geo_dn) and V (top=v0,
    /// bottom=v0+(v1-v0)*cut_dn).
    bottom_w: f32,
    /// Normalized U for this corner.
    u: f32,
    /// Normalized V at the top edge of the face (unscaled).
    v0: f32,
    /// Normalized V at the configured bottom edge of the face (pre-cut_dn).
    v1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SkyboxUniforms {
    sky_view_proj: [[f32; 4]; 4],
    /// `cut_dn` in .x; .yzw unused. Padded to vec4 for std140 alignment.
    /// See `compute_cut_dn` / MatrixMap.cpp:2064.
    cut_dn: [f32; 4],
}

pub struct Sky {
    gradient_pipeline: wgpu::RenderPipeline,
    gradient_vertex_buffer: wgpu::Buffer,
    sky_color: [f32; 3],
    water_color: [f32; 3],
    skybox: Option<Skybox>,
}

struct Skybox {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Static yaw offset — sum of the sky config's `Angle` (hardcoded per sky
    /// name here) and the map's `SkyAngle` property.
    base_angle: f32,
    /// Radians per millisecond. Ports `m_SkyDeltaAngle` (MatrixMap.cpp:2495):
    /// the original advances `m_SkyAngle += m_SkyDeltaAngle * step` each Takt
    /// so clouds / stars drift over time. Value comes from the `DeltaAngle`
    /// Sky config entry; we use a gentle default since data.txt isn't parsed.
    delta_angle: f32,
    /// Accumulator advanced by `Sky::takt`.
    current_angle: f32,
}

impl Sky {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        sky_color_rgba: u32,
        water_color_rgba: u32,
        sky_name: &str,
        sky_angle: f32,
        matrix_data: Option<&Storage>,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Self {
        let sky_color = unpack_rgb(sky_color_rgba);
        let water_color = unpack_rgb(water_color_rgba);

        let gradient_pipeline = build_gradient_pipeline(device, config);
        let gradient_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sky Gradient VB"),
            contents: bytemuck::cast_slice(
                &[GradientVertex {
                    position: [0.0, 0.0],
                    color: [0.0; 4],
                }; 10],
            ),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let sky_cfg = matrix_data
            .and_then(|stor| resolve_sky_config(stor, sky_name))
            .or_else(|| {
                log::warn!(
                    "sky: no config for '{}' in robots.dat; skybox disabled",
                    sky_name
                );
                None
            });
        let skybox = sky_cfg
            .and_then(|cfg| load_skybox(device, queue, config, &cfg, sky_angle, read_texture));

        Self {
            gradient_pipeline,
            gradient_vertex_buffer,
            sky_color,
            water_color,
            skybox,
        }
    }

    /// Advance the skybox rotation. Mirrors `m_SkyAngle += m_SkyDeltaAngle *
    /// step` in `CMatrixMap::Takt` (MatrixMap.cpp:2495).
    pub fn takt(&mut self, dt_ms: f32) {
        if let Some(skybox) = &mut self.skybox {
            skybox.current_angle += skybox.delta_angle * dt_ms;
        }
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
    ) {
        let has_skybox = self.skybox.is_some();

        if let Some(skybox) = &self.skybox {
            skybox.render(queue, pass, camera);
        }

        self.render_gradient(queue, pass, camera, has_skybox);
    }

    fn render_gradient<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
        has_skybox: bool,
    ) {
        let sh_frac = compute_sky_height_frac(camera);
        let bot_ndc = 1.0 - 2.0 * sh_frac;
        let mid_ndc = 1.0 - 2.0 * (sh_frac - SH2_FRAC);
        let mut top_ndc = 1.0 - 2.0 * (sh_frac - SH1_FRAC);
        if top_ndc < 1.0 {
            top_ndc = 1.0;
        }

        let [sr, sg, sb] = self.sky_color;
        // DrawSky:2117 — with-skybox branch sets TOP color alpha to 0 so the
        // gradient fades into the skybox instead of drawing a hard line. The
        // no-skybox branch sets TOP color to 0 (fully transparent black).
        let top_color = if has_skybox {
            [sr, sg, sb, 0.0]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        };
        let sky_opaque = [sr, sg, sb, 1.0];
        let [wr, wg, wb] = self.water_color;
        let water_opaque = [wr, wg, wb, 1.0];
        let water_top = bot_ndc.clamp(-1.0, 1.0);

        let verts = [
            GradientVertex {
                position: [-1.0, top_ndc],
                color: top_color,
            },
            GradientVertex {
                position: [1.0, top_ndc],
                color: top_color,
            },
            GradientVertex {
                position: [-1.0, mid_ndc],
                color: sky_opaque,
            },
            GradientVertex {
                position: [1.0, mid_ndc],
                color: sky_opaque,
            },
            GradientVertex {
                position: [-1.0, bot_ndc],
                color: sky_opaque,
            },
            GradientVertex {
                position: [1.0, bot_ndc],
                color: sky_opaque,
            },
            GradientVertex {
                position: [-1.0, water_top],
                color: water_opaque,
            },
            GradientVertex {
                position: [1.0, water_top],
                color: water_opaque,
            },
            GradientVertex {
                position: [-1.0, -1.0],
                color: water_opaque,
            },
            GradientVertex {
                position: [1.0, -1.0],
                color: water_opaque,
            },
        ];
        queue.write_buffer(
            &self.gradient_vertex_buffer,
            0,
            bytemuck::cast_slice(&verts),
        );

        pass.set_pipeline(&self.gradient_pipeline);
        pass.set_vertex_buffer(0, self.gradient_vertex_buffer.slice(..));
        pass.draw(0..6, 0..1);
        pass.draw(6..10, 0..1);
    }
}

impl Skybox {
    fn render<'a>(&'a self, queue: &wgpu::Queue, pass: &mut wgpu::RenderPass<'a>, camera: &Camera) {
        let sky_view_proj = build_sky_view_proj(camera, self.base_angle + self.current_angle);
        let cut_dn = compute_cut_dn(camera.eye_pos().z);
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&SkyboxUniforms {
                sky_view_proj: sky_view_proj.to_cols_array_2d(),
                cut_dn: [cut_dn, 0.0, 0.0, 0.0],
            }),
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        // 4 walls × 6 indices (2 triangles). All baked into a single VB so we
        // match the `CInstDraw::BeginDraw + 4× AddVerts + ActualDraw` batch.
        pass.draw(0..24, 0..1);
    }
}

fn build_gradient_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sky Gradient Shader"),
        source: wgpu::ShaderSource::Wgsl(GRADIENT_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Sky Gradient Layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Sky Gradient Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GradientVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: 8,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn load_skybox(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    cfg: &SkyConfig,
    sky_angle: f32,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<Skybox> {
    // All four faces point at the same texture in every shipped Sky block
    // (only the UV rectangle differs). Load the Fore face's texture and
    // sample the other faces' UV strips from the same atlas.
    let texture_path = &cfg.faces[0].texture;
    let data = read_texture(texture_path)?;
    let rgba = decode_texture_bytes(&data)?;
    let tex_w = rgba.width() as f32;
    let tex_h = rgba.height() as f32;
    // Skybox textures are low-res panoramics — mipmap them so the horizon band
    // filters cleanly when tilted.
    let texture_view = if rgba.width() >= 4 && rgba.height() >= 4 {
        create_texture_from_rgba_mipped(device, queue, &rgba, 6)
    } else {
        create_texture_from_rgba(device, queue, &rgba)
    };
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Skybox Sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Skybox Shader"),
        source: wgpu::ShaderSource::Wgsl(SKYBOX_SHADER.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Skybox BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Skybox Layout"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Skybox Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<BoxVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    // xy
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    // bottom_w
                    wgpu::VertexAttribute {
                        offset: 8,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32,
                    },
                    // u
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32,
                    },
                    // v0
                    wgpu::VertexAttribute {
                        offset: 16,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32,
                    },
                    // v1
                    wgpu::VertexAttribute {
                        offset: 20,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                // DrawSky:2029 — alpha blend disabled on the skybox pass.
                blend: None,
                write_mask: wgpu::ColorWrites::COLOR,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        // DrawSky:2043 — z-test and z-write disabled. We force clip.z = clip.w
        // in the shader so the skybox writes the far-plane depth, keeping
        // everything else in front of it without needing to poke at state.
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let verts = build_skybox_vertices(cfg, tex_w, tex_h);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Skybox VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Skybox UB"),
        contents: bytemuck::bytes_of(&SkyboxUniforms {
            sky_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            cut_dn: [0.5, 0.0, 0.0, 0.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Skybox BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    log::info!(
        "skybox: loaded '{}' (angle = {:.3} + {:.3} rad, drift = {:.6} rad/ms)",
        texture_path,
        cfg.base_angle,
        sky_angle,
        cfg.delta_angle
    );
    Some(Skybox {
        pipeline,
        vertex_buffer,
        uniform_buffer,
        bind_group,
        base_angle: cfg.base_angle + sky_angle,
        delta_angle: cfg.delta_angle,
        current_angle: 0.0,
    })
}

struct SkyConfig {
    /// Static yaw in radians, matches the `Angle` parameter of a `Sky` block
    /// in `cfg/robots/data.txt` (converted via `GRAD2RAD` by the original).
    base_angle: f32,
    /// Yaw rate in radians per millisecond, matches `DeltaAngle`.
    delta_angle: f32,
    /// Per-face texture + UV rect (one entry per FACE_CONFIG_KEYS index).
    faces: [SkyFaceConfig; 4],
}

struct SkyFaceConfig {
    /// Texture path with `/` separators (relative to pkg root).
    texture: String,
    /// Normalized UV rectangle (u0, v0, u1, v1). `v1` may exceed 1.0 for the
    /// Right face — the original relies on CLAMP addressing, which pins the
    /// extra strip to the texture edge.
    uv: [f32; 4],
}

/// Parse a `Sky` block from the serialized `robots.dat` BlockPar. `stor` is
/// the root CStorage, `sky_name` is the `SkyName` property on the map (e.g.
/// `Default`, `BlueMoon`). Returns `None` if the sky block is missing — the
/// caller then draws the gradient pass only.
fn resolve_sky_config(stor: &Storage, sky_name: &str) -> Option<SkyConfig> {
    let sky_root = stor.block_record("da", "Sky")?;
    let sky_rec = stor.block_record(&sky_root, sky_name)?;

    let parse_deg = |s: Option<String>| -> f32 {
        s.and_then(|v| v.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(0.0)
    };
    let base_angle = parse_deg(stor.block_param(&sky_rec, "Angle"));
    let delta_angle = parse_deg(stor.block_param(&sky_rec, "DeltaAngle"));

    let faces = std::array::from_fn(|i| {
        let raw = stor
            .block_param(&sky_rec, FACE_CONFIG_KEYS[i])
            .unwrap_or_default();
        parse_face(&raw)
    });

    Some(SkyConfig {
        base_angle,
        delta_angle,
        faces,
    })
}

/// Parse a face entry of the form `<path>,<u0>,<v0>,<u1>,<v1>` where the UV
/// components are pixel offsets into the underlying texture. Falls back to a
/// full-texture quad if the string is empty or malformed.
///
/// The shipped skies all use 1024-wide/tall panoramic textures, but we don't
/// have the image dimensions at parse time. The original C++ divides by
/// `tex->GetSizeX()/Y()` after load (MatrixMapPrepare.cpp:1160). We defer
/// that division and keep the raw ints here, normalising later once we've
/// decoded the texture.
fn parse_face(raw: &str) -> SkyFaceConfig {
    let mut parts = raw.splitn(5, ',');
    let path = parts.next().unwrap_or("").trim().replace('\\', "/");
    let mut uv = [0.0_f32, 0.0, 1.0, 1.0];
    for slot in &mut uv {
        if let Some(p) = parts.next() {
            if let Ok(v) = p.trim().parse::<f32>() {
                *slot = v;
            }
        }
    }
    SkyFaceConfig { texture: path, uv }
}

fn build_skybox_vertices(cfg: &SkyConfig, tex_w: f32, tex_h: f32) -> Vec<BoxVertex> {
    // Z-up world-space cube centered on the camera: Y is forward, X is right,
    // Z is up (matches the rest of the port's coordinate system).
    //
    // Only XY is baked here. The shader derives Z (top=1.0, bottom=geo_dn)
    // and the bottom V from the `cut_dn` uniform so the horizon tracks camera
    // height, matching the per-frame recompute in DrawSky (MatrixMap.cpp:2064-2076).

    // Face corner XY positions, indexed by FACE_CONFIG_KEYS: Fore, Right,
    // Back, Left. Each face is (tl, tr, bl, br).
    let positions: [[[f32; 2]; 4]; 4] = [
        // Fore: +Y wall.
        [[-1.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, 1.0]],
        // Right: +X wall.
        [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]],
        // Back: -Y wall.
        [[1.0, -1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, -1.0]],
        // Left: -X wall.
        [[-1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]],
    ];

    let mut verts = Vec::with_capacity(24);
    for (i, p) in positions.iter().enumerate() {
        let [u0, v0, u1, v1] = cfg.faces[i].uv;
        let nu0 = u0 / tex_w;
        let nu1 = u1 / tex_w;
        let nv0 = v0 / tex_h;
        let nv1 = v1 / tex_h;
        push_face(&mut verts, p[0], p[1], p[2], p[3], nu0, nu1, nv0, nv1);
    }

    verts
}

#[allow(clippy::too_many_arguments)]
fn push_face(
    out: &mut Vec<BoxVertex>,
    xy_tl: [f32; 2],
    xy_tr: [f32; 2],
    xy_bl: [f32; 2],
    xy_br: [f32; 2],
    u_left: f32,
    u_right: f32,
    v0: f32,
    v1: f32,
) {
    let tl = BoxVertex { xy: xy_tl, bottom_w: 0.0, u: u_left,  v0, v1 };
    let tr = BoxVertex { xy: xy_tr, bottom_w: 0.0, u: u_right, v0, v1 };
    let bl = BoxVertex { xy: xy_bl, bottom_w: 1.0, u: u_left,  v0, v1 };
    let br = BoxVertex { xy: xy_br, bottom_w: 1.0, u: u_right, v0, v1 };
    // Two triangles per face (TL, BL, TR) and (TR, BL, BR).
    out.push(tl);
    out.push(bl);
    out.push(tr);
    out.push(tr);
    out.push(bl);
    out.push(br);
}

/// Matches `cut_dn = (200 + m_Camera.GetFrustumCenter().z) * 0.5 /
/// MAX_VIEW_DISTANCE + 0.5` from MatrixMap.cpp:2064. Called per-frame so the
/// skybox's horizon row tracks the camera's altitude.
fn compute_cut_dn(eye_z: f32) -> f32 {
    (200.0 + eye_z) * 0.5 / MAX_VIEW_DISTANCE + 0.5
}

fn build_sky_view_proj(camera: &Camera, sky_angle: f32) -> Mat4 {
    // The rendering view_proj in the port is `proj * look_at_rh(y_up_eye,
    // y_up_target, Y) * z_to_y` (camera.rs:363-378). For the skybox we want
    // the same transform chain with the camera's translation stripped so the
    // cube follows the camera. We rebuild the Y-up look-at with eye at the
    // origin pointing along the camera's forward direction.
    let fwd = camera.forward();
    let y_up_fwd = Vec3::new(fwd.x, fwd.z, -fwd.y);
    let view_rot = Mat4::look_at_rh(Vec3::ZERO, y_up_fwd, Vec3::Y);

    let yaw = Mat4::from_rotation_z(sky_angle);
    let z_to_y = Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);

    // DrawSky:2048 — near=0.01, far=3. A narrow FOV replays the original's
    // `CAM_HFOV` (60°) so walls project 1-to-1 per face.
    let proj = Mat4::perspective_rh(SKY_HFOV, camera.aspect, 0.01, 3.0);

    proj * view_rot * z_to_y * yaw
}

/// Ports the m_SkyHeight computation from MatrixVisiCalc.cpp:609-626.
/// Returns the horizon line as a fraction of screen height (0=top, 1=bottom).
fn compute_sky_height_frac(camera: &Camera) -> f32 {
    let dir = camera.forward();
    let mut proj = Vec3::new(dir.x, dir.y, 0.0);
    let len = proj.length();
    if len < 0.0001 {
        return -1.0;
    }
    proj /= len;
    let eye = camera.eye_pos();
    let mut target = eye + proj * MAX_VIEW_DISTANCE;
    target.z -= eye.z * 1.5;

    let clip = camera.view_proj() * Vec4::new(target.x, target.y, target.z, 1.0);
    if clip.w.abs() < 0.0001 {
        return -1.0;
    }
    let ndc_y = clip.y / clip.w;
    (1.0 - ndc_y) * 0.5
}

const GRADIENT_SHADER: &str = r#"
struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex fn vs_main(@location(0) pos: vec2<f32>, @location(1) col: vec4<f32>) -> VOut {
    var out: VOut;
    out.clip_pos = vec4<f32>(pos, 0.0, 1.0);
    out.color = col;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

const SKYBOX_SHADER: &str = r#"
struct U {
    sky_view_proj: mat4x4<f32>,
    // .x = cut_dn. Ports MatrixMap.cpp:2064 — recomputed each frame from
    // camera altitude so the skybox's horizon row tracks the view.
    cut_dn: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_main(
    @location(0) xy: vec2<f32>,
    @location(1) bottom_w: f32,
    @location(2) u_coord: f32,
    @location(3) v0: f32,
    @location(4) v1: f32,
) -> VOut {
    var out: VOut;
    let cut = u.cut_dn.x;
    // Top corners at z=1.0, bottom corners at geo_dn = 1 - 2*cut_dn
    // (MatrixMap.cpp:2067). bottom_w is 0 or 1, interpolating between them.
    let z = 1.0 - 2.0 * cut * bottom_w;
    let pos = vec3<f32>(xy, z);
    // Bottom V scales with cut_dn too (MatrixMap.cpp:2072): top = v0,
    // bottom = v0 + (v1 - v0) * cut_dn.
    let v = v0 + (v1 - v0) * cut * bottom_w;
    let clip = u.sky_view_proj * vec4<f32>(pos, 1.0);
    // Pin to far plane so the depth test never rejects the skybox.
    out.clip_pos = vec4<f32>(clip.xy, clip.w, clip.w);
    out.uv = vec2<f32>(u_coord, v);
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(t, s, in.uv).rgb, 1.0);
}
"#;
