//! Port of `SPOT_TURRET` landscape decal (MatrixEffect.cpp:359-368,
//! MatrixObjectBuilding.cpp:1640). The C++ creates a flat textured
//! quad on the terrain at each free turret slot when the build picker
//! is open, using `Matrix\Textures\LandSpots\spot_turr2` (a circular
//! white-on-transparent decal).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::matrix_game::map::GameMap;
use crate::matrix_lib::three_g::texture::{create_texture_from_rgba, decode_texture_bytes};

/// One free turret slot to mark on the terrain. World-space position
/// + radius (in world units). Match the C++ `CreateLandscapeSpot(...,
/// scale=6, SPOT_TURRET)` — scale is in move-cells, so the world-space
/// radius is `scale * GLOBAL_SCALE_MOVE`.
#[derive(Debug, Clone, Copy)]
pub struct SlotMarker {
    pub pos: glam::Vec2,
    pub pos_z: f32,
    pub radius: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuadVertex {
    /// `[-1, 1]` in both axes, mapped to UV `[0, 1]` in fs.
    corner: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceData {
    /// World-space center XYZ + radius in `w`.
    center_radius: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    /// Tint applied to the spot — port of
    /// `m_SpotProperties[SPOT_TURRET].color = 0x80FFFFFF` (translucent
    /// white) at MatrixEffect.cpp:360.
    color: [f32; 4],
}

const SHADER: &str = r#"
struct VIn {
    @location(0) corner: vec2<f32>,
    @location(1) center_radius: vec4<f32>,
};
struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct U {
    view_proj: mat4x4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var s_diffuse: sampler;

@vertex fn vs_main(in: VIn) -> VOut {
    let center = in.center_radius.xyz;
    let radius = in.center_radius.w;
    // Flat ground quad: corner offsets in the XY world plane.
    let world = vec4<f32>(
        center.x + in.corner.x * radius,
        center.y + in.corner.y * radius,
        center.z,
        1.0
    );
    var out: VOut;
    out.clip_position = u.view_proj * world;
    out.uv = (in.corner + vec2<f32>(1.0, 1.0)) * 0.5;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_diffuse, s_diffuse, in.uv);
    return vec4<f32>(u.color.rgb * tex.rgb, u.color.a * tex.a);
}
"#;

pub struct SlotMarkerRenderer {
    pipeline: wgpu::RenderPipeline,
    quad_vb: wgpu::Buffer,
    quad_ib: wgpu::Buffer,
    instance_vb: wgpu::Buffer,
    instance_cap: u32,
    instance_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl SlotMarkerRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        let tex_path = "Matrix/Textures/LandSpots/spot_turr2";
        let Some(tex_bytes) = read_texture(tex_path) else {
            log::warn!("slot_marker: missing texture {tex_path} in bundle");
            return None;
        };
        log::debug!(
            "slot_marker: loaded {} bytes for {tex_path}",
            tex_bytes.len()
        );
        let Some(rgba) = decode_texture_bytes(&tex_bytes) else {
            log::warn!("slot_marker: decode failed for {tex_path}");
            return None;
        };
        log::debug!(
            "slot_marker: decoded {}x{} for {tex_path}",
            rgba.width(),
            rgba.height()
        );
        let view = create_texture_from_rgba(device, queue, &rgba);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SlotMarker BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SlotMarker UB"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                // 0x80FFFFFF — translucent white (MatrixEffect.cpp:360).
                color: [1.0, 1.0, 1.0, 0.5],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SlotMarker BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SlotMarker Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SlotMarker PL"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SlotMarker Pipeline"),
            layout: Some(&pl),
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
                            format: wgpu::VertexFormat::Float32x4,
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
                    // Standard alpha blend so the decal blends with terrain.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
                // Don't write — but DO test against terrain so the spot
                // doesn't show through hills.
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
            label: Some("SlotMarker Quad VB"),
            contents: bytemuck::cast_slice(&quad_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_indices = [0u16, 1, 2, 0, 2, 3];
        let quad_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SlotMarker Quad IB"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instance_cap = 32u32;
        let instance_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SlotMarker Instance VB"),
            size: (instance_cap as u64) * std::mem::size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            pipeline,
            quad_vb,
            quad_ib,
            instance_vb,
            instance_cap,
            instance_count: 0,
            uniform_buffer,
            bind_group,
        })
    }

    pub fn sync(&mut self, queue: &wgpu::Queue, map: &GameMap, markers: &[SlotMarker]) {
        let n = (markers.len() as u32).min(self.instance_cap);
        self.instance_count = n;
        if n == 0 {
            return;
        }
        // Subtract map center — view_proj is in centered coords, same
        // convention every other in-world renderer (cannons, robots,
        // buildings) uses for instance transforms.
        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;
        let data: Vec<InstanceData> = markers
            .iter()
            .take(n as usize)
            .map(|m| InstanceData {
                center_radius: [m.pos.x - cx, m.pos.y - cy, m.pos_z, m.radius],
            })
            .collect();
        queue.write_buffer(&self.instance_vb, 0, bytemuck::cast_slice(&data));
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        view_proj: glam::Mat4,
    ) {
        if self.instance_count == 0 {
            return;
        }
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                // 0x80FFFFFF — translucent white (MatrixEffect.cpp:360).
                color: [1.0, 1.0, 1.0, 0.5],
            }),
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.quad_vb.slice(..));
        pass.set_vertex_buffer(1, self.instance_vb.slice(..));
        pass.set_index_buffer(self.quad_ib.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..6, 0, 0..self.instance_count);
    }
}
