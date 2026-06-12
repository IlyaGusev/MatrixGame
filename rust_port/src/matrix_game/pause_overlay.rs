//! Port of `CMatrixMap::Render`'s `IsPaused()` block at
//! MatrixMap.cpp:2430-2454. When the game is paused, the original
//! draws a fullscreen translucent black quad on top of the world and
//! below the UI. The "Pause" hint widget itself comes from
//! `MatrixLogic.cpp:2611` (`m_PauseHint = CMatrixHint::Build(TEMPLATE_PAUSE)`)
//! and rides the standard hint render path — see `form_game.rs`'s
//! interface upload, which substitutes the cached Pause hint while
//! `is_paused` is true.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Tint applied to the dim quad — black at α≈40%. The C++ ships
/// `0x14000000` (α≈8%) at MatrixMap.cpp:2439; that's imperceptible
/// here so we crank it to read as a clear "world is paused" dim
/// while still keeping the playfield legible.
const PAUSE_TINT_RGBA: [f32; 4] = [0.0, 0.0, 0.0, 0.4];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OverlayVertex {
    /// Clip-space XY in `[-1, 1]`.
    pos: [f32; 2],
    color: [f32; 4],
}

pub struct PauseOverlay {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
}

impl PauseOverlay {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pause Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pause Overlay PL"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pause Overlay Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<OverlayVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as u64,
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Static fullscreen triangle covering NDC.
        let verts = [
            OverlayVertex {
                pos: [-1.0, -1.0],
                color: PAUSE_TINT_RGBA,
            },
            OverlayVertex {
                pos: [3.0, -1.0],
                color: PAUSE_TINT_RGBA,
            },
            OverlayVertex {
                pos: [-1.0, 3.0],
                color: PAUSE_TINT_RGBA,
            },
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pause Overlay VB"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            pipeline,
            vertex_buffer,
        }
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..3, 0..1);
    }
}

const SHADER: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};
@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(pos, 0.0, 1.0);
    o.color = color;
    return o;
}
@fragment
fn fs_main(v: VOut) -> @location(0) vec4<f32> { return v.color; }
"#;
