//! Stencil shadow rendering — the render half of `CVOShadowStencil`
//! (ShadowStencil.cpp:412) plus the `CMatrixMap::DrawShadows`
//! composition (MatrixMap.cpp:1865-2000).
//!
//! Volume geometry is built CPU-side per object unit by
//! `matrix_lib::three_g::shadow_stencil::ShadowStencil` and accumulated
//! here in (map-centered) world space each frame. The CMAP-baked
//! projected shadows of decorative objects (object.rs) join the same
//! composition through their stencil-mark pipeline (`fs_stencil` in
//! object_shadow.wgsl).

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

const STENCIL_SHADER: &str = include_str!("../../shaders/shadow_stencil.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct StencilUniform {
    view_proj: [[f32; 4]; 4],
    darken_color: [f32; 4],
}

pub struct StencilShadowRenderer {
    verts: Vec<[f32; 3]>,
    inds: Vec<u32>,
    vb: Option<wgpu::Buffer>,
    vb_cap: usize,
    ib: Option<wgpu::Buffer>,
    ib_cap: usize,
    ub: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline_volume: wgpu::RenderPipeline,
    pipeline_darken: wgpu::RenderPipeline,
    index_count: u32,
}

impl StencilShadowRenderer {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Stencil Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(STENCIL_SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Stencil Shadow BGL"),
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Stencil Shadow PL"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stencil Shadow UB"),
            contents: bytemuck::bytes_of(&StencilUniform {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                darken_color: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Stencil Shadow BG"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ub.as_entire_binding(),
            }],
        });

        // Volume pass: z-pass stencil counting. D3D front (CW) increments,
        // back decrements (MatrixMap.cpp:1884-1898, two-sided branch), depth
        // test on / writes off, color writes off.
        let pipeline_volume = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Stencil Shadow Volume Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_volume"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_volume"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::Always,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::IncrementWrap,
                    },
                    back: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::Always,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::DecrementWrap,
                    },
                    read_mask: 0xFF,
                    write_mask: 0xFF,
                },
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Darken pass: fullscreen quad, passes where ref(1) <= stencil
        // (D3DCMP_LESSEQUAL, MatrixMap.cpp:1993-1995), src-alpha blend,
        // no depth test (D3DRS_ZENABLE FALSE).
        let pipeline_darken = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Stencil Shadow Darken Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_darken"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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
                format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::LessEqual,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::Keep,
                    },
                    back: wgpu::StencilFaceState {
                        compare: wgpu::CompareFunction::LessEqual,
                        fail_op: wgpu::StencilOperation::Keep,
                        depth_fail_op: wgpu::StencilOperation::Keep,
                        pass_op: wgpu::StencilOperation::Keep,
                    },
                    read_mask: 0xFF,
                    write_mask: 0,
                },
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            verts: Vec::new(),
            inds: Vec::new(),
            vb: None,
            vb_cap: 0,
            ib: None,
            ib_cap: 0,
            ub,
            bind_group,
            pipeline_volume,
            pipeline_darken,
            index_count: 0,
        }
    }

    /// Drop last frame's accumulated volumes. Call before the per-object
    /// sync passes push this frame's geometry.
    pub fn begin_frame(&mut self) {
        self.verts.clear();
        self.inds.clear();
    }

    /// Append one object's volume mesh (`ShadowStencil::geometry()`),
    /// transformed by the unit's world matrix — the `Render(objma)`
    /// world transform of ShadowStencil.cpp:421.
    pub fn push_volume(&mut self, world: Mat4, verts: &[[f32; 3]], inds: &[u16]) {
        let base = self.verts.len() as u32;
        self.verts.extend(
            verts
                .iter()
                .map(|v| world.transform_point3(Vec3::from(*v)).to_array()),
        );
        self.inds.extend(inds.iter().map(|&i| base + i as u32));
    }

    /// Upload the frame's volumes + darken color. `shadow_color` is the
    /// CMAP DATA_SHADOWCOLOR 0xAARRGGBB; the darken quad's color is
    /// `shadow_color` (vertex diffuse, MatrixMapPrepare.cpp:2009) times
    /// the 0xC0C0C0C0 texture factor (MatrixMap.cpp:1979).
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        shadow_color: u32,
    ) {
        let f = 0xC0 as f32 / 255.0;
        let a = ((shadow_color >> 24) & 0xFF) as f32 / 255.0 * f;
        let r = ((shadow_color >> 16) & 0xFF) as f32 / 255.0 * f;
        let g = ((shadow_color >> 8) & 0xFF) as f32 / 255.0 * f;
        let b = (shadow_color & 0xFF) as f32 / 255.0 * f;
        queue.write_buffer(
            &self.ub,
            0,
            bytemuck::bytes_of(&StencilUniform {
                view_proj: view_proj.to_cols_array_2d(),
                darken_color: [r, g, b, a],
            }),
        );

        self.index_count = self.inds.len() as u32;
        if self.inds.is_empty() {
            return;
        }
        let vb_bytes = self.verts.len() * 12;
        if self.vb.is_none() || vb_bytes > self.vb_cap {
            self.vb_cap = vb_bytes.next_power_of_two();
            self.vb = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Stencil Shadow VB"),
                size: self.vb_cap as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        queue.write_buffer(
            self.vb.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&self.verts),
        );
        let ib_bytes = self.inds.len() * 4;
        if self.ib.is_none() || ib_bytes > self.ib_cap {
            self.ib_cap = ib_bytes.next_power_of_two();
            self.ib = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Stencil Shadow IB"),
                size: self.ib_cap as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        queue.write_buffer(
            self.ib.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&self.inds),
        );
    }

    /// Stencil-count all volume triangles (one two-sided draw).
    pub fn render_volumes<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.index_count == 0 {
            return;
        }
        let (Some(vb), Some(ib)) = (&self.vb, &self.ib) else {
            return;
        };
        pass.set_pipeline(&self.pipeline_volume);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    /// True when any volume or proj silhouette marked the stencil this
    /// frame (proj marks are counted by the caller).
    pub fn has_volumes(&self) -> bool {
        self.index_count > 0
    }

    /// Fullscreen darken where stencil >= 1.
    pub fn render_darken<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline_darken);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_stencil_reference(1);
        pass.draw(0..3, 0..1);
    }
}
