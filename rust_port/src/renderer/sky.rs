//! Sky rendering — ports DrawSky (MatrixMap.cpp:2020-2179, no-skybox branch).
//!
//! Screen-space gradient: fades from transparent (above horizon) to opaque sky
//! color at the horizon line. m_SkyHeight is computed from camera direction by
//! projecting a far point to screen space (MatrixVisiCalc.cpp:609-626).

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::game::common::unpack_rgb;
use crate::renderer::camera::Camera;

/// MAX_VIEW_DISTANCE from MatrixCamera.cpp:13.
const MAX_VIEW_DISTANCE: f32 = 4000.0;
/// SH1 = g_ScreenY * 0.270416... (MatrixMap.cpp:2144).
const SH1_FRAC: f32 = 0.270416666666667;
/// SH2 = g_ScreenY * 0.07 (MatrixMap.cpp:2145).
const SH2_FRAC: f32 = 0.07;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SkyVertex {
    position: [f32; 2],
    color: [f32; 4],
}

pub struct Sky {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    sky_color: [f32; 3],
    water_color: [f32; 3],
}

impl Sky {
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        sky_color_rgba: u32,
        water_color_rgba: u32,
    ) -> Self {
        let sky_color = unpack_rgb(sky_color_rgba);
        let water_color = unpack_rgb(water_color_rgba);

        // 10 verts = sky gradient (6) + separator + water band (4). Use non-strip for clarity.
        let initial = [SkyVertex {
            position: [0.0, 0.0],
            color: [0.0; 4],
        }; 10];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sky VB"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(SKY_SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SkyVertex>() as u64,
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
        });

        Self {
            pipeline,
            vertex_buffer,
            sky_color,
            water_color,
        }
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
    ) {
        let sh_frac = compute_sky_height_frac(camera);

        // Screen frac (0=top, 1=bottom) → NDC y (1=top, -1=bottom): ndc_y = 1 - 2*frac
        let bot_ndc = 1.0 - 2.0 * sh_frac;
        let mid_ndc = 1.0 - 2.0 * (sh_frac - SH2_FRAC);
        let mut top_ndc = 1.0 - 2.0 * (sh_frac - SH1_FRAC);

        // "if (v[0].p.y > 0) v[0].p.y = 0;" (MatrixMap.cpp:2158) — pin top strip to
        // screen top when horizon pulls it inside the viewport.
        if top_ndc < 1.0 {
            top_ndc = 1.0;
        }

        let [sr, sg, sb] = self.sky_color;
        let sky_opaque = [sr, sg, sb, 1.0];
        let sky_transparent = [sr, sg, sb, 0.0];
        let [wr, wg, wb] = self.water_color;
        let water_opaque = [wr, wg, wb, 1.0];

        // Clamp horizon to keep the water band geometry valid even when the horizon
        // sits off-screen (e.g., top-down view where bot_ndc > 1).
        let water_top = bot_ndc.clamp(-1.0, 1.0);

        let verts = [
            // Sky gradient strip (draw call 1): transparent top → opaque sky at horizon.
            SkyVertex {
                position: [-1.0, top_ndc],
                color: sky_transparent,
            },
            SkyVertex {
                position: [1.0, top_ndc],
                color: sky_transparent,
            },
            SkyVertex {
                position: [-1.0, mid_ndc],
                color: sky_opaque,
            },
            SkyVertex {
                position: [1.0, mid_ndc],
                color: sky_opaque,
            },
            SkyVertex {
                position: [-1.0, bot_ndc],
                color: sky_opaque,
            },
            SkyVertex {
                position: [1.0, bot_ndc],
                color: sky_opaque,
            },
            // Water band strip (draw call 2): horizon → screen bottom, solid water color.
            // Fills the below-horizon backdrop so shoreline alpha blends water-over-water
            // instead of water-over-sky (eliminates the halo between water and terrain).
            SkyVertex {
                position: [-1.0, water_top],
                color: water_opaque,
            },
            SkyVertex {
                position: [1.0, water_top],
                color: water_opaque,
            },
            SkyVertex {
                position: [-1.0, -1.0],
                color: water_opaque,
            },
            SkyVertex {
                position: [1.0, -1.0],
                color: water_opaque,
            },
        ];
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&verts));

        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..6, 0..1);
        pass.draw(6..10, 0..1);
    }
}

/// Ports the m_SkyHeight computation from MatrixVisiCalc.cpp:609-626.
/// Returns the horizon line as a fraction of screen height (0=top, 1=bottom).
fn compute_sky_height_frac(camera: &Camera) -> f32 {
    let dir = camera.forward();
    let mut proj = Vec3::new(dir.x, dir.y, 0.0);
    let len = proj.length();
    if len < 0.0001 {
        // Camera looking straight up/down: original sets m_SkyHeight = -100 (off-screen).
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

const SKY_SHADER: &str = r#"
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
