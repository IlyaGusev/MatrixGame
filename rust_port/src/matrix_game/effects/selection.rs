//! Port of `CMatrixEffectSelection` (MatrixEffectSelection.{cpp,hpp}).
//!
//! Each selected object gets a ring of `m_SelCnt = ceil(2πr / SEL_PP_DIST)`
//! billboard "dots" around it. Dots repel each other, cling to the
//! terrain via `g_MatrixMap->GetZ`, and are clamped back into the ring
//! when they drift outward. Color fades via `LIC` between `m_ColorFrom`
//! and `m_ColorTo` over `SEL_COLOR_CHANGE_TIME = 200ms`.
//!
//! CBillboard itself (camera-facing textured quad with a draw-queue
//! submission via `Sort`) isn't ported yet — we draw the dots
//! directly as camera-facing quads in this module's own pipeline,
//! with a soft-circular alpha mask in the fragment shader instead of
//! the `BBT_SELDOT` texture. Visual parity is the same; revisit when
//! the CBillboard subsystem arrives.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

// ── Constants (MatrixEffectSelection.hpp:10-15) ──────────────────────────

pub const SEL_COLOR_CHANGE_TIME_MS: f32 = 200.0;
pub const SEL_SPEED: f32 = 0.1;
pub const SEL_SIZE: f32 = 1.5;
pub const SEL_PP_DIST: f32 = 5.5;
/// Floor on dot count. The C++ uses `if (m_SelCnt < 10) m_SelCnt = 10;`
/// (MatrixEffectSelection.cpp:26).
const MIN_SEL_CNT: usize = 10;

// ── Effect state (ports m_Points + m_Color* + m_Radius) ──────────────────

/// Port of `SSelPoint::m_Pos[0]` (MatrixEffectSelection.hpp:22) — a
/// single dot's offset from the ring center. `SEL_BLUR_CNT=1` in the
/// shipping game so we drop the secondary blur slots; revisit if a
/// build enables them.
#[derive(Clone, Copy)]
struct SelPoint {
    /// World-space offset from the ring's `center`. Matches the
    /// C++ `m_Points[i].m_Pos[0]` which is stored as a delta so the
    /// ring center can move (e.g. if the selected object moves).
    rel_pos: Vec3,
}

pub struct SelectionEffect {
    /// `m_Pos` — the ring's world-space anchor
    /// (MatrixEffectSelection.hpp:34). Selected building / unit moves
    /// this via `SetPos`.
    center: Vec3,
    /// `m_Radius` (MatrixEffectSelection.hpp:35).
    radius: f32,
    /// `m_Color_current` / `m_ColorTo` / `m_ColorFrom` / `m_ColorTime`
    /// (MatrixEffectSelection.hpp:38-41). Color lerp state.
    color_current: u32,
    color_to: u32,
    color_from: u32,
    color_time: f32,
    points: Vec<SelPoint>,
}

impl SelectionEffect {
    /// Port of the ctor (MatrixEffectSelection.cpp:16-65). Places
    /// `m_SelCnt` dots evenly around the ring at initial radius; the
    /// animation redistributes them each frame. Initial `color_current`
    /// is 0 (opaque black); `SetColor` kicks off the fade to `color`.
    pub fn new(center: Vec3, radius: f32, color: u32) -> Self {
        let cnt = {
            let raw = (std::f32::consts::TAU * radius / SEL_PP_DIST) as usize;
            raw.max(MIN_SEL_CNT)
        };
        let mut points = Vec::with_capacity(cnt);
        let da = std::f32::consts::TAU / cnt as f32;
        for i in 0..cnt {
            let (s, c) = (i as f32 * da).sin_cos();
            points.push(SelPoint {
                rel_pos: Vec3::new(s * radius, c * radius, 0.0),
            });
        }
        let mut eff = Self {
            center,
            radius,
            color_current: 0,
            color_to: color,
            color_from: color,
            color_time: SEL_COLOR_CHANGE_TIME_MS,
            points,
        };
        // Match C++ ctor's trailing `SetColor(color)` (line 63).
        eff.set_color(color);
        eff
    }

    /// `SetPos` (MatrixEffectSelection.hpp:61-66). Ring center moves;
    /// dots are stored relative so their positions stay valid.
    pub fn set_pos(&mut self, pos: Vec3) {
        self.center = pos;
    }

    /// `SetRadius` (MatrixEffectSelection.hpp:67-70).
    pub fn set_radius(&mut self, r: f32) {
        self.radius = r;
    }

    /// `SetColor` (MatrixEffectSelection.hpp:77-82). Primes the fade
    /// from the current color to `color`.
    pub fn set_color(&mut self, color: u32) {
        self.color_from = self.color_current;
        self.color_to = color;
        self.color_time = SEL_COLOR_CHANGE_TIME_MS;
    }

    /// Current lerped color, sampled each frame by the renderer.
    pub fn color(&self) -> u32 {
        self.color_current
    }

    /// Port of `CMatrixEffectSelection::Takt(float step)`
    /// (MatrixEffectSelection.cpp:96-206). `step_ms` is the frame
    /// delta in ms (note the C++ `float step` is ALSO in ms here —
    /// confirmed via the divisor `INVERT(SEL_COLOR_CHANGE_TIME)` of
    /// 1/200, giving t ∈ [0,1] over 200ms).
    ///
    /// `get_z(x, y)` samples terrain elevation so dots cling to the
    /// ground — matches `g_MatrixMap->GetZ(newpos.x, newpos.y)` at
    /// MatrixEffectSelection.cpp:139.
    pub fn takt(&mut self, step_ms: f32, get_z: impl Fn(f32, f32) -> f32) {
        let dtime = 0.1 * step_ms;

        // Color lerp.  LIC(to, from, t) with t = time/CHANGE_TIME:
        //   t = 1 → pure `from` (freshly-started), t = 0 → pure `to`.
        if self.color_time >= 0.0 {
            let t = (self.color_time / SEL_COLOR_CHANGE_TIME_MS).clamp(0.0, 1.0);
            self.color_current = lerp_color(self.color_to, self.color_from, t);
            self.color_time -= step_ms;
            if self.color_time < 0.0 {
                self.color_current = self.color_to;
            }
        }

        // Dot animation (MatrixEffectSelection.cpp:113-148). Each dot
        // repels from every other (sum-of-normals), plus a tangential
        // push (`cross(rel_pos, Z) * SEL_SPEED`) so the ring rotates.
        // Then clamped to terrain z and the ring radius.
        let n = self.points.len();
        for i in 0..n {
            let mut dir = Vec3::ZERO;
            let pi = self.points[i].rel_pos;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = pi - self.points[j].rel_pos;
                let len = d.length();
                if len > 1e-6 {
                    dir += d / len;
                }
            }
            let len = dir.length();
            if len > 1e-6 {
                dir /= len;
            }
            // Tangent = rel_pos × Z, scaled by SEL_SPEED.
            let move_dir = {
                let t = pi.cross(Vec3::Z);
                let ml = t.length();
                if ml > 1e-6 { t / ml } else { Vec3::ZERO }
            };
            dir += move_dir * SEL_SPEED;

            self.points[i].rel_pos += dir * dtime;

            // Terrain clamp (MatrixEffectSelection.cpp:138-142). The
            // `+ SEL_SIZE` keeps the dot a hair above ground; `- center.z`
            // brings the sampled z back into the local-offset frame.
            let world_xy = self.center + self.points[i].rel_pos;
            let z = get_z(world_xy.x, world_xy.y);
            self.points[i].rel_pos.z = (z + SEL_SIZE - self.center.z).max(0.0);

            // Radius clamp (MatrixEffectSelection.cpp:146-147).
            let r = self.points[i].rel_pos.length();
            if r > self.radius {
                self.points[i].rel_pos *= self.radius / r;
            }
        }
    }

    /// Iterate world-space dot positions (for upload to the GPU).
    pub fn world_positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        let c = self.center;
        self.points.iter().map(move |p| c + p.rel_pos)
    }
}

/// Port of `LIC(to, from, t)` from Common.hpp — linear interpolation
/// with t=1 selecting `from` and t=0 selecting `to`. Channel-wise on
/// ARGB u32s.
fn lerp_color(to: u32, from: u32, t: f32) -> u32 {
    let ch = |c: u32, sh: u32| ((c >> sh) & 0xFF) as f32;
    let mix = |a: f32, b: f32| {
        (a * (1.0 - t) + b * t).clamp(0.0, 255.0) as u32
    };
    let r = mix(ch(to, 16), ch(from, 16));
    let g = mix(ch(to, 8), ch(from, 8));
    let b = mix(ch(to, 0), ch(from, 0));
    let a = mix(ch(to, 24), ch(from, 24));
    (a << 24) | (r << 16) | (g << 8) | b
}

// ── Renderer (dot billboards) ────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuadVertex {
    corner: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceData {
    center: [f32; 3],
    _pad: f32,
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

pub struct SelectionRingRenderer {
    pipeline: wgpu::RenderPipeline,
    quad_vb: wgpu::Buffer,
    quad_ib: wgpu::Buffer,
    instance_vb: wgpu::Buffer,
    instance_cap: usize,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    effect: Option<SelectionEffect>,
    /// Dots currently uploaded to the GPU (≤ instance_cap).
    dot_count: u32,
}

impl SelectionRingRenderer {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Selection Ring BGL"),
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
            label: Some("Selection Ring Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Selection Ring PL"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Selection Ring Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    // Per-vertex: quad corner (-1..1).
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<QuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    // Per-instance: dot world center.
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
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

        // Static unit quad.
        let quad_verts = [
            QuadVertex { corner: [-1.0, -1.0] },
            QuadVertex { corner: [ 1.0, -1.0] },
            QuadVertex { corner: [ 1.0,  1.0] },
            QuadVertex { corner: [-1.0,  1.0] },
        ];
        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Selection Quad VB"),
            contents: bytemuck::cast_slice(&quad_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_indices = [0u16, 1, 2, 0, 2, 3];
        let quad_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Selection Quad IB"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Instance buffer sized for up to ~200 dots (4x the max a
        // reasonably large selection radius would produce). Grown on
        // demand in update().
        let instance_cap = 256usize;
        let instance_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Selection Instance VB"),
            size: (instance_cap * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Selection Ring UB"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                cam_right: [1.0, 0.0, 0.0, 0.0],
                cam_up:    [0.0, 0.0, 1.0, 0.0],
                color:     [0.0, 1.0, 0.0, 0.8],
                size:      [SEL_SIZE, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Selection Ring BG"),
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
            effect: None,
            dot_count: 0,
        }
    }

    /// Drop the current effect (Kill / UnSelect).
    pub fn clear(&mut self) {
        self.effect = None;
        self.dot_count = 0;
    }

    /// Bring the effect into existence, or rebind its center/radius/color.
    /// Ports `CMatrixEffect::CreateSelection` + subsequent SetPos/SetRadius
    /// calls (MatrixEffectSelection.hpp:61-82) as a single reconciler.
    pub fn set_selection(&mut self, center: Vec3, radius: f32, color: u32) {
        if self.effect.is_none() {
            self.effect = Some(SelectionEffect::new(center, radius, color));
        } else {
            let e = self.effect.as_mut().unwrap();
            e.set_pos(center);
            e.set_radius(radius);
            if e.color_to != color {
                e.set_color(color);
            }
        }
    }

    /// Advance the dot animation by `step_ms`.
    pub fn takt(&mut self, step_ms: f32, get_z: impl Fn(f32, f32) -> f32) {
        if let Some(e) = &mut self.effect {
            e.takt(step_ms, get_z);
        }
    }

    /// Upload the current dot centers to the instance buffer and the
    /// current view matrix bases to the uniform. Called once per frame
    /// before `render()`.
    ///
    /// `map_center` is the `(world_width/2, world_height/2)` offset the
    /// scene renderers subtract from every world position before upload
    /// (see BuildingsRenderer, ObjectsRenderer). The selection shader
    /// multiplies by `view_proj` which expects that same centered
    /// space, so we apply the shift here too.
    pub fn upload(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        cam_right: Vec3,
        cam_up: Vec3,
        map_center: glam::Vec2,
    ) {
        let Some(e) = &self.effect else {
            self.dot_count = 0;
            return;
        };
        let centers: Vec<InstanceData> = e.world_positions()
            .take(self.instance_cap)
            .map(|p| InstanceData {
                center: [p.x - map_center.x, p.y - map_center.y, p.z],
                _pad: 0.0,
            })
            .collect();
        self.dot_count = centers.len() as u32;
        queue.write_buffer(&self.instance_vb, 0, bytemuck::cast_slice(&centers));

        let col = e.color();
        let a = ((col >> 24) & 0xFF) as f32 / 255.0;
        let r = ((col >> 16) & 0xFF) as f32 / 255.0;
        let g = ((col >> 8) & 0xFF) as f32 / 255.0;
        let b = (col & 0xFF) as f32 / 255.0;
        let alpha = if a == 0.0 { 0.9 } else { a };
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                cam_right: [cam_right.x, cam_right.y, cam_right.z, 0.0],
                cam_up:    [cam_up.x,    cam_up.y,    cam_up.z,    0.0],
                color:     [r, g, b, alpha],
                size:      [SEL_SIZE, 0.0, 0.0, 0.0],
            }),
        );
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.dot_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.quad_vb.slice(..));
        pass.set_vertex_buffer(1, self.instance_vb.slice(..));
        pass.set_index_buffer(self.quad_ib.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..6, 0, 0..self.dot_count);
    }
}

const SHADER: &str = include_str!("../../../shaders/selection_ring.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_count_is_floor_2pi_r_over_sel_pp_dist() {
        let e = SelectionEffect::new(Vec3::ZERO, 74.0, 0x00FF00);
        let expected = (std::f32::consts::TAU * 74.0 / SEL_PP_DIST) as usize;
        assert_eq!(e.points.len(), expected.max(MIN_SEL_CNT));
    }

    #[test]
    fn dot_count_is_clamped_to_minimum() {
        let e = SelectionEffect::new(Vec3::ZERO, 1.0, 0x00FF00);
        assert_eq!(e.points.len(), MIN_SEL_CNT);
    }

    #[test]
    fn color_fade_finishes_after_change_time() {
        let mut e = SelectionEffect::new(Vec3::ZERO, 10.0, 0x00FF00);
        e.set_color(0xFF0000);
        // Halfway through: color is a mix.
        e.takt(100.0, |_, _| 0.0);
        let mid = e.color();
        assert_ne!(mid, 0x00FF00);
        assert_ne!(mid, 0xFF0000);
        // Past the fade window: lands at color_to.
        e.takt(150.0, |_, _| 0.0);
        assert_eq!(e.color() & 0x00FFFFFF, 0xFF0000);
    }

    #[test]
    fn takt_clamps_dots_to_radius() {
        // Place a dot beyond the ring; takt should pull it back.
        let mut e = SelectionEffect::new(Vec3::ZERO, 20.0, 0);
        for p in e.points.iter_mut() {
            p.rel_pos = Vec3::new(100.0, 0.0, 0.0);
        }
        e.takt(16.0, |_, _| 0.0);
        for p in &e.points {
            let r = p.rel_pos.length();
            assert!(r <= 20.0 + 0.001, "dot drifted past radius: {}", r);
        }
    }
}
