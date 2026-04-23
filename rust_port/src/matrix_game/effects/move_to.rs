//! Port of `CMatrixEffectMoveto` (Effects/MatrixEffectMoveTo.{cpp,hpp}).
//!
//! Spawned on right-click move orders. Each instance is 6 camera-facing
//! billboards that spiral inward toward the destination over
//! `MOVETO_TTL = 400 ms` while the per-billboard z oscillates — a
//! classic RTS "move ping".
//!
//! Per-billboard placement follows `SPointMoveTo::Change`
//! (MatrixEffectMoveTo.cpp:22-48):
//!   angle   a = (i & ~1) * π / 3     — 3 pairs of billboards at 0°, 60°, 120°
//!   radial  r = MOVETO_R * k         — spirals in from 20 units to 0
//!   tangent d = MOVETO_R * 0.3 * sin(π·k)  (sign per `i & 1`)
//!   height  z = MOVETO_Z * sin(π·k / 2)    — 0 → 10 → 0 arc
//!   pos     = center + r·(s,c) + d·(c,-s); z clamped to terrain floor
//!
//! We don't have the C++ `TEXTURE_PATH_MOVETO` billboard texture
//! bundled; the fragment shader draws a soft radial white-yellow disk
//! which reads the same way visually (a bright ground ping).

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::matrix_game::map::GameMap;

/// MatrixEffectMoveTo.cpp:10 — effect lifetime in ms.
const MOVETO_TTL_MS: f32 = 400.0;
/// MOVETO_S (:11) — billboard world-space scale (half-size).
const MOVETO_S: f32 = 2.0;
/// MOVETO_R (:12) — outer spiral radius.
const MOVETO_R: f32 = 20.0;
/// MOVETO_Z (:13) — height of the vertical bounce at its apex.
const MOVETO_Z: f32 = 10.0;
/// Number of billboards per effect (`m_Pts[6]`, MatrixEffectMoveTo.hpp:25).
const PTS: usize = 6;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceData {
    center: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuadVertex {
    corner: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    color: [f32; 4],
    size: [f32; 4],
}

/// A single live move-to effect. The C++ stores `m_TTL` decrementing to
/// zero; we store the same (ms left until auto-removal).
#[derive(Copy, Clone)]
struct Effect {
    center: Vec3,
    ttl_ms: f32,
}

/// Port of `SPointMoveTo::Change` (MatrixEffectMoveTo.cpp:22-48). Returns
/// the world-space center of billboard `i` at animation phase `k ∈ [0, 1]`.
/// `k = m_TTL / MOVETO_TTL` per the C++ at :107 — k=1 when fresh, 0 at
/// expiry.
fn billboard_pos(map: &GameMap, center: Vec3, i: usize, k: f32) -> Vec3 {
    let pi = std::f32::consts::PI;
    let a = ((i & !1) as f32) * pi / 3.0;
    let (s, c) = a.sin_cos();
    // Tangential jitter — sign alternates per pair (`SET_SIGN_FLOAT(d, i & 1)`).
    let mut d = MOVETO_R * 0.3 * (pi * k).sin();
    if i & 1 == 1 {
        d = -d;
    }
    let z_osc = (pi * k / 2.0).sin();
    let x = center.x + MOVETO_R * s * k + d * c;
    let y = center.y + MOVETO_R * c * k - d * s;
    // Terrain clamp: billboard z never drops below the click-point z,
    // matching the C++ `if (lz < host->m_Pos.z) lz = host->m_Pos.z` at :44.
    let mut lz = map.get_z(x, y);
    if lz < center.z {
        lz = center.z;
    }
    Vec3::new(x, y, lz + MOVETO_Z * z_osc)
}

pub struct MoveToRenderer {
    pipeline: wgpu::RenderPipeline,
    quad_vb: wgpu::Buffer,
    quad_ib: wgpu::Buffer,
    instance_vb: wgpu::Buffer,
    instance_cap: usize,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    effects: Vec<Effect>,
    /// Instances uploaded this frame (≤ instance_cap).
    instance_count: u32,
}

impl MoveToRenderer {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        use wgpu::util::DeviceExt;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MoveTo BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MoveTo Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MoveTo PL"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("MoveTo Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<QuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<InstanceData>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        }],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    // Additive (`D3DBLEND_ONE` destblend for billboards
                    // in the original — see MatrixEffect.cpp for the
                    // billboard common state).
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let quad_verts = [
            QuadVertex {
                corner: [-1.0, -1.0],
            },
            QuadVertex {
                corner: [1.0, -1.0],
            },
            QuadVertex { corner: [1.0, 1.0] },
            QuadVertex {
                corner: [-1.0, 1.0],
            },
        ];
        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MoveTo Quad VB"),
            contents: bytemuck::cast_slice(&quad_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_indices = [0u16, 1, 2, 0, 2, 3];
        let quad_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MoveTo Quad IB"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // 6 billboards × up to ~16 concurrent effects = 96 instances.
        let instance_cap = 128usize;
        let instance_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MoveTo Instance VB"),
            size: (instance_cap * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MoveTo UB"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                cam_right: [1.0, 0.0, 0.0, 0.0],
                cam_up: [0.0, 0.0, 1.0, 0.0],
                color: [1.0, 0.92, 0.5, 1.0],
                size: [MOVETO_S, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MoveTo BG"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            quad_vb,
            quad_ib,
            instance_vb,
            instance_cap,
            uniform_buffer,
            bind_group,
            effects: Vec::new(),
            instance_count: 0,
        }
    }

    /// Port of `CMatrixEffect::CreateMoveto(pos)` (called per selected
    /// robot inside `CMatrixSideUnit::PGShowPlace` at
    /// MatrixSide.cpp:8553,8561). Pushes a fresh effect at `pos`.
    pub fn spawn(&mut self, pos: Vec3) {
        self.effects.push(Effect {
            center: pos,
            ttl_ms: MOVETO_TTL_MS,
        });
    }

    /// Per-frame takt — port of `CMatrixEffectMoveto::Takt`
    /// (MatrixEffectMoveTo.cpp:93-113). Decrement TTL, drop expired.
    pub fn takt(&mut self, step_ms: f32) {
        for e in &mut self.effects {
            e.ttl_ms -= step_ms;
        }
        self.effects.retain(|e| e.ttl_ms > 0.0);
    }

    pub fn is_active(&self) -> bool {
        !self.effects.is_empty()
    }

    /// Rebuild the instance buffer + uniforms. Called each frame before
    /// `render`.
    pub fn upload(
        &mut self,
        queue: &wgpu::Queue,
        map: &GameMap,
        view_proj: glam::Mat4,
        cam_right: Vec3,
        cam_up: Vec3,
        map_center: glam::Vec2,
    ) {
        if self.effects.is_empty() {
            self.instance_count = 0;
            return;
        }
        let mut instances: Vec<InstanceData> = Vec::with_capacity(self.effects.len() * PTS);
        for e in &self.effects {
            let k = (e.ttl_ms / MOVETO_TTL_MS).clamp(0.0, 1.0);
            for i in 0..PTS {
                if instances.len() >= self.instance_cap {
                    break;
                }
                let p = billboard_pos(map, e.center, i, k);
                instances.push(InstanceData {
                    center: [p.x - map_center.x, p.y - map_center.y, p.z],
                    _pad: 0.0,
                });
            }
        }
        self.instance_count = instances.len() as u32;
        queue.write_buffer(&self.instance_vb, 0, bytemuck::cast_slice(&instances));

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                cam_right: [cam_right.x, cam_right.y, cam_right.z, 0.0],
                cam_up: [cam_up.x, cam_up.y, cam_up.z, 0.0],
                // Warm white-yellow for a readable ground ping.
                color: [1.0, 0.92, 0.5, 1.0],
                size: [MOVETO_S, 0.0, 0.0, 0.0],
            }),
        );
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.instance_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.quad_vb.slice(..));
        pass.set_vertex_buffer(1, self.instance_vb.slice(..));
        pass.set_index_buffer(self.quad_ib.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..6, 0, 0..self.instance_count);
    }
}

// Camera-facing billboard shader with a soft radial disk. Pattern
// matches `SelectionRingRenderer`'s shader — `corner` ∈ [-1, 1]² drives
// a 2D distance-from-center, which we map through a smoothstep to give
// a bright core + soft falloff. Color × alpha × radial mask is then
// combined via the pipeline's additive blend.
const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up:    vec4<f32>,
    color:     vec4<f32>,
    size:      vec4<f32>,
};

@group(0) @binding(0) var<uniform> U: Uniforms;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) corner: vec2<f32>,
    @location(1) center: vec3<f32>,
) -> VOut {
    var o: VOut;
    let r = U.cam_right.xyz * (corner.x * U.size.x);
    let u = U.cam_up.xyz    * (corner.y * U.size.x);
    let wp = center + r + u;
    o.pos = U.view_proj * vec4<f32>(wp, 1.0);
    o.uv = corner;
    return o;
}

@fragment
fn fs_main(v: VOut) -> @location(0) vec4<f32> {
    let d = length(v.uv);
    // Bright core that falls off at the edge. `1 - smoothstep(0.0, 1.0, d)`
    // gives a soft disk; squaring the result sharpens the core a bit.
    let m = 1.0 - smoothstep(0.0, 1.0, d);
    let a = m * m * U.color.a;
    return vec4<f32>(U.color.rgb * a, a);
}
"#;
