//! Decorative objects — palms, rocks, trees, grass, etc.
//!
//! Loads .vo meshes (game/vo_loader.rs) for each object type_id referenced by
//! the map, then draws all instances of each type as one instanced draw call.
//! Alpha-tested sampling handles foliage texture cutouts without z-ordering.

use std::collections::{BTreeMap, HashMap};

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::effects::point_light::PointLightSystem;
use crate::game::common::{unpack_rgb, FOG_END, FOG_START};
use crate::game::map::{GameMap, ObjectInstance, ObjectShadow};
use crate::game::vo_loader::{self, ShadowKind};
use crate::renderer::camera::Camera;
use crate::renderer::texture::{
    create_solid_texture, create_texture_from_rgba, decode_texture_bytes,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

/// Per-instance data: world transform expressed as 4 rows of a mat4, laid out
/// as sequential vertex attributes with step_mode=Instance.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceData {
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
    row3: [f32; 4],
    terrain_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    camera_pos: [f32; 4],
    time_ms: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaterialUniform {
    flags: [u32; 4],
    scroll: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowProjUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowTextureUniform {
    u_row: [f32; 4],
    v_row: [f32; 4],
}

struct MeshBatch {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instance_buffer: wgpu::Buffer,
    num_instances: u32,
    bind_group: wgpu::BindGroup,
    objects: Vec<ObjectInstance>,
    center: [f32; 2],
}

struct ShadowBatch {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    bind_group: wgpu::BindGroup,
}

struct SurfaceShadowSource {
    indices: Vec<u32>,
    diffuse_view: wgpu::TextureView,
}

pub struct ObjectsRenderer {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    batches: Vec<MeshBatch>,
    shadow_batches: Vec<ShadowBatch>,
    uniform_buffer: wgpu::Buffer,
    shadow_uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    time_ms: f32,
    last_point_light_revision: u64,
}

impl ObjectsRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        stor: &Storage,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        if map.objects.is_empty() {
            return None;
        }

        let strings = stor.get_buf("strings", "String")?;

        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;
        let mut by_type: BTreeMap<u32, Vec<&ObjectInstance>> = BTreeMap::new();
        for obj in &map.objects {
            by_type.entry(obj.type_id).or_default().push(obj);
        }

        let [sr, sg, sb] = unpack_rgb(map.sky_color);
        let fog_color = [sr, sg, sb, 1.0];
        let [ar, ag, ab] = unpack_rgb(map.ambient_color_obj);
        let [lr, lg, lb] = unpack_rgb(map.light_main_color_obj);
        let ambient_color = [ar, ag, ab, 1.0];
        let light_color = [lr, lg, lb, 1.0];
        let light_dir = [
            map.light_main_dir[0],
            map.light_main_dir[1],
            map.light_main_dir[2],
            0.0,
        ];

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Objects UB"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                ambient_color,
                light_color,
                light_dir,
                camera_pos: [0.0, 0.0, 0.0, 1.0],
                time_ms: [0.0, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shadow_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Object Shadows UB"),
            contents: bytemuck::bytes_of(&ShadowProjUniform {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let objects_bgl = create_objects_bgl(device);
        let shadow_bgl = create_shadow_bgl(device);
        let shadow_gen_bgl = create_shadow_gen_bgl(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let pipeline = create_objects_pipeline(device, config, &objects_bgl);
        let shadow_pipeline = create_shadow_pipeline(device, config, &shadow_bgl);
        let shadow_gen_pipeline = create_shadow_texture_pipeline(device, &shadow_gen_bgl);

        let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
        let fallback_tex = create_solid_texture(device, queue, [200, 200, 200, 255]);
        let black_tex = create_solid_texture(device, queue, [0, 0, 0, 255]);
        let transparent_tex = create_solid_texture(device, queue, [0, 0, 0, 0]);

        let mut batches = Vec::new();
        let mut shadow_batches = Vec::new();
        let mut loaded_types = 0usize;
        let mut failed_types = 0usize;

        for (type_id, instances) in &by_type {
            let id_str = if (*type_id as usize) < strings.arrays_count() {
                strings.get_as_wstr(*type_id as usize)
            } else {
                continue;
            };
            let Some(paths) = vo_loader::resolve_paths(&id_str) else {
                failed_types += 1;
                continue;
            };
            let Some(vo_bytes) = read_texture(&paths.vo_path) else {
                log::debug!("objects: VO not found: {}", paths.vo_path);
                failed_types += 1;
                continue;
            };
            let mesh = match vo_loader::parse_vo(&vo_bytes) {
                Ok(m) => m,
                Err(e) => {
                    log::warn!("objects: parse VO {} failed: {}", paths.vo_path, e);
                    failed_types += 1;
                    continue;
                }
            };

            let object_dir = paths
                .vo_path
                .rsplit_once('/')
                .map(|(dir, _)| format!("{dir}/"));
            let vertices: Vec<Vertex> = mesh
                .vertices
                .iter()
                .map(|v| Vertex {
                    position: v.position,
                    normal: v.normal,
                    uv: v.uv,
                })
                .collect();
            let inst_data: Vec<InstanceData> = instances
                .iter()
                .map(|obj| instance_matrix(obj, cx, cy, map, None))
                .collect();
            let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Objects Inst VB"),
                contents: bytemuck::cast_slice(&inst_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

            let use_vo_surface_materials = mesh.surfaces.len() > 1;
            let mut shadow_surfaces = Vec::new();

            for surface in &mesh.surfaces {
                let surface_material = if use_vo_surface_materials {
                    surface.texture_ref.as_deref().map(|spec| {
                        vo_loader::parse_material_spec_with_prefix(spec, object_dir.as_deref())
                    })
                } else {
                    None
                };
                let material = if use_vo_surface_materials {
                    vo_loader::merge_materials(&paths.material, surface_material.as_ref())
                } else {
                    paths.material.clone()
                };

                let diffuse_view = resolve_texture(
                    material.diffuse.as_ref(),
                    device,
                    queue,
                    &mut tex_cache,
                    read_texture,
                )
                .unwrap_or_else(|| fallback_tex.clone());
                let gloss_view = resolve_texture(
                    material.gloss.as_ref(),
                    device,
                    queue,
                    &mut tex_cache,
                    read_texture,
                )
                .unwrap_or_else(|| black_tex.clone());
                let back_view = resolve_texture(
                    material.back.as_ref(),
                    device,
                    queue,
                    &mut tex_cache,
                    read_texture,
                )
                .unwrap_or_else(|| black_tex.clone());
                let mask_view = resolve_texture(
                    material.mask.as_ref(),
                    device,
                    queue,
                    &mut tex_cache,
                    read_texture,
                )
                .unwrap_or_else(|| transparent_tex.clone());

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Objects Mesh VB"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Objects Mesh IB"),
                    contents: bytemuck::cast_slice(&surface.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let mat_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Objects Material UB"),
                    contents: bytemuck::bytes_of(&MaterialUniform {
                        flags: [
                            material.gloss.is_some() as u32,
                            material.back.is_some() as u32,
                            material.mask.is_some() as u32,
                            0,
                        ],
                        scroll: [material.scroll[0], material.scroll[1], 0.0, 0.0],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Objects BG"),
                    layout: &objects_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&diffuse_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&gloss_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&back_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&mask_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: mat_uniform.as_entire_binding(),
                        },
                    ],
                });

                batches.push(MeshBatch {
                    vertex_buffer,
                    index_buffer,
                    num_indices: surface.indices.len() as u32,
                    instance_buffer: instance_buffer.clone(),
                    num_instances: inst_data.len() as u32,
                    bind_group,
                    objects: instances.iter().map(|obj| (*obj).clone()).collect(),
                    center: [cx, cy],
                });
                shadow_surfaces.push(SurfaceShadowSource {
                    indices: surface.indices.clone(),
                    diffuse_view: diffuse_view.clone(),
                });
            }

            if matches!(
                paths.shadow.kind,
                ShadowKind::ProjectedStatic | ShadowKind::ProjectedDynamic
            ) {
                for obj in instances
                    .iter()
                    .filter_map(|obj| obj.shadow.as_ref().map(|shadow| (&**obj, shadow)))
                {
                    if let Some(batch) = build_shadow_batch(
                        device,
                        queue,
                        &shadow_pipeline,
                        &shadow_gen_pipeline,
                        &shadow_bgl,
                        &shadow_gen_bgl,
                        &sampler,
                        &shadow_uniform_buffer,
                        &vertices,
                        &shadow_surfaces,
                        obj.0,
                        obj.1,
                        map,
                        paths.shadow.texture_size.max(32),
                    ) {
                        shadow_batches.push(batch);
                    }
                }
            }

            loaded_types += 1;
        }

        log::info!(
            "objects: {} mesh types loaded, {} skipped, {} total instances drawn, {} projected shadows",
            loaded_types,
            failed_types,
            batches.iter().map(|b| b.num_instances).sum::<u32>(),
            shadow_batches.len(),
        );

        if batches.is_empty() && shadow_batches.is_empty() {
            return None;
        }

        Some(Self {
            pipeline,
            shadow_pipeline,
            batches,
            shadow_batches,
            uniform_buffer,
            shadow_uniform_buffer,
            fog_color,
            ambient_color,
            light_color,
            light_dir,
            time_ms: 0.0,
            last_point_light_revision: 0,
        })
    }

    pub fn takt(
        &mut self,
        dt_ms: f32,
        queue: &wgpu::Queue,
        map: &GameMap,
        point_lights: &PointLightSystem,
    ) {
        self.time_ms += dt_ms;
        if self.time_ms > 1_000_000.0 {
            self.time_ms -= 1_000_000.0;
        }

        let revision = point_lights.revision();
        if revision != self.last_point_light_revision {
            for batch in &mut self.batches {
                let [cx, cy] = batch.center;
                let inst_data: Vec<InstanceData> = batch
                    .objects
                    .iter()
                    .map(|obj| instance_matrix(obj, cx, cy, map, Some(point_lights)))
                    .collect();
                queue.write_buffer(&batch.instance_buffer, 0, bytemuck::cast_slice(&inst_data));
            }
            self.last_point_light_revision = revision;
        }
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
        view_proj: glam::Mat4,
    ) {
        let eye = camera.eye_pos();
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color: self.fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                ambient_color: self.ambient_color,
                light_color: self.light_color,
                light_dir: self.light_dir,
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                time_ms: [self.time_ms, 0.0, 0.0, 0.0],
            }),
        );
        queue.write_buffer(
            &self.shadow_uniform_buffer,
            0,
            bytemuck::bytes_of(&ShadowProjUniform {
                view_proj: view_proj.to_cols_array_2d(),
            }),
        );

        if !self.shadow_batches.is_empty() {
            pass.set_pipeline(&self.shadow_pipeline);
            for batch in &self.shadow_batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..batch.num_indices, 0, 0..1);
            }
        }

        if !self.batches.is_empty() {
            pass.set_pipeline(&self.pipeline);
            for batch in &self.batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, batch.instance_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..batch.num_indices, 0, 0..batch.num_instances);
            }
        }
    }
}

fn create_objects_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Objects BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_shadow_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Object Shadows BGL"),
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
    })
}

fn create_shadow_gen_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Object Shadow Texture BGL"),
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
    })
}

fn create_objects_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Objects Shader"),
        source: wgpu::ShaderSource::Wgsl(OBJECTS_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Objects PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Objects Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<InstanceData>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 48,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 64,
                            shader_location: 7,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                },
            ],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
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
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_shadow_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Object Shadows Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Object Shadows PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Object Shadows Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ShadowVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x2,
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
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            strip_index_format: Some(wgpu::IndexFormat::Uint32),
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
    })
}

fn create_shadow_texture_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Object Shadow Texture Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_TEXTURE_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Object Shadow Texture PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Object Shadow Texture Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
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
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_shadow_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    _shadow_pipeline: &wgpu::RenderPipeline,
    shadow_gen_pipeline: &wgpu::RenderPipeline,
    shadow_bgl: &wgpu::BindGroupLayout,
    shadow_gen_bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    shadow_uniform_buffer: &wgpu::Buffer,
    vertices: &[Vertex],
    surfaces: &[SurfaceShadowSource],
    obj: &ObjectInstance,
    shadow: &ObjectShadow,
    map: &GameMap,
    texture_size: u32,
) -> Option<ShadowBatch> {
    if shadow.vertices.is_empty() || shadow.indices.len() < 3 || surfaces.is_empty() {
        return None;
    }

    let shadow_texture = build_shadow_texture(
        device,
        queue,
        shadow_gen_pipeline,
        shadow_gen_bgl,
        sampler,
        vertices,
        surfaces,
        obj,
        shadow,
        map,
        texture_size,
    )?;

    let shadow_vertices: Vec<ShadowVertex> = shadow
        .vertices
        .iter()
        .map(|v| ShadowVertex {
            position: [
                v.position[0] - map.world_width() * 0.5,
                v.position[1] - map.world_height() * 0.5,
                v.position[2],
            ],
            uv: v.uv,
        })
        .collect();
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Object Shadow VB"),
        contents: bytemuck::cast_slice(&shadow_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Object Shadow IB"),
        contents: bytemuck::cast_slice(&shadow.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Object Shadow BG"),
        layout: shadow_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadow_texture),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });

    Some(ShadowBatch {
        vertex_buffer,
        index_buffer,
        num_indices: shadow.indices.len() as u32,
        bind_group,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_shadow_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    vertices: &[Vertex],
    surfaces: &[SurfaceShadowSource],
    obj: &ObjectInstance,
    shadow: &ObjectShadow,
    map: &GameMap,
    texture_size: u32,
) -> Option<wgpu::TextureView> {
    let projection = shadow_texture_projection(obj, shadow, map)?;
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Object Shadow Texture UB"),
        contents: bytemuck::bytes_of(&projection),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Object Shadow Source VB"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Object Shadow Texture"),
        size: wgpu::Extent3d {
            width: texture_size,
            height: texture_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&Default::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Object Shadow Texture Depth"),
        size: wgpu::Extent3d {
            width: texture_size,
            height: texture_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Object Shadow Encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Object Shadow Texture Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));

        for surface in surfaces {
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Object Shadow Source IB"),
                contents: bytemuck::cast_slice(&surface.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Object Shadow Texture BG"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&surface.diffuse_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..surface.indices.len() as u32, 0, 0..1);
        }
    }
    queue.submit([encoder.finish()]);

    Some(texture_view)
}

fn shadow_texture_projection(
    obj: &ObjectInstance,
    shadow: &ObjectShadow,
    map: &GameMap,
) -> Option<ShadowTextureUniform> {
    let rotation = object_rotation(obj);
    let inv_rotation = rotation.transpose();

    let mut light = inv_rotation
        * Vec3::new(
            map.light_main_dir[0],
            map.light_main_dir[1],
            map.light_main_dir[2],
        );
    if light.length_squared() < 1e-6 {
        return None;
    }
    light = light.normalize();

    let mut up = inv_rotation * Vec3::Z;
    if up.length_squared() < 1e-6 || up.normalize().dot(light).abs() > 0.98 {
        up = Vec3::Y;
    }
    let right = up.cross(light).normalize_or_zero();
    if right.length_squared() < 1e-6 {
        return None;
    }
    let up = light.cross(right).normalize_or_zero();
    if up.length_squared() < 1e-6 {
        return None;
    }

    let dim_x = shadow.dimensions[0].abs().max(0.001);
    let dim_y = shadow.dimensions[1].abs().max(0.001);
    let campos = Vec3::from_array(shadow.camera_pos);
    let u_row = Vec4::new(
        right.x / dim_x,
        right.y / dim_x,
        right.z / dim_x,
        -right.dot(campos) / dim_x,
    );
    let v_row = Vec4::new(
        up.x / dim_y,
        up.y / dim_y,
        up.z / dim_y,
        -up.dot(campos) / dim_y,
    );
    Some(ShadowTextureUniform {
        u_row: u_row.to_array(),
        v_row: v_row.to_array(),
    })
}

/// Build the same static-object transform order used by the original renderer:
/// Rx * Ry * Rz, then uniform scale, then translation into centered render space.
fn instance_matrix(
    obj: &ObjectInstance,
    cx: f32,
    cy: f32,
    map: &GameMap,
    point_lights: Option<&PointLightSystem>,
) -> InstanceData {
    let s = obj.scale.max(0.0001);
    let [terrain_r, terrain_g, terrain_b] =
        unpack_rgb(map.static_object_color_with_lighting(obj.x, obj.y, point_lights));
    let m = object_rotation(obj) * s;

    InstanceData {
        row0: [m.x_axis.x, m.y_axis.x, m.z_axis.x, obj.x - cx],
        row1: [m.x_axis.y, m.y_axis.y, m.z_axis.y, obj.y - cy],
        row2: [m.x_axis.z, m.y_axis.z, m.z_axis.z, obj.z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [terrain_r, terrain_g, terrain_b, 1.0],
    }
}

fn object_rotation(obj: &ObjectInstance) -> Mat3 {
    let (sx, cxr) = obj.angle_x.sin_cos();
    let (sy, cyr) = obj.angle_y.sin_cos();
    let (sz, cz) = obj.angle_z.sin_cos();
    let rx = Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, cxr, sx, 0.0, -sx, cxr]);
    let ry = Mat3::from_cols_array(&[cyr, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cyr]);
    let rz = Mat3::from_cols_array(&[cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0]);
    rx * ry * rz
}

fn resolve_texture(
    tex_path: Option<&String>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<wgpu::TextureView> {
    let path = tex_path?;
    if let Some(cached) = cache.get(path) {
        return Some(cached.clone());
    }
    let Some(data) = read_texture(path) else {
        return None;
    };
    let Some(rgba) = decode_texture_bytes(&data) else {
        return None;
    };
    let view = create_texture_from_rgba(device, queue, &rgba);
    cache.insert(path.clone(), view.clone());
    Some(view)
}

const OBJECTS_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    ambient_color: vec4<f32>,
    light_color: vec4<f32>,
    light_dir: vec4<f32>,
    camera_pos: vec4<f32>,
    time_ms: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var t_gloss: texture_2d<f32>;
@group(0) @binding(3) var t_back: texture_2d<f32>;
@group(0) @binding(4) var t_mask: texture_2d<f32>;
@group(0) @binding(5) var s_diffuse: sampler;
struct M {
    flags: vec4<u32>,
    scroll: vec4<f32>,
};
@group(0) @binding(6) var<uniform> m: M;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
    @location(7) terrain_color: vec4<f32>,
};
struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) view_dist: f32,
    @location(3) terrain_color: vec3<f32>,
    @location(4) world_pos: vec3<f32>,
};

@vertex fn vs_main(in: VIn) -> VOut {
    let p = vec4<f32>(in.position, 1.0);
    let world = vec4<f32>(dot(in.m0, p), dot(in.m1, p), dot(in.m2, p), dot(in.m3, p));
    let clip = u.view_proj * world;
    var out: VOut;
    out.clip_position = clip;
    out.uv = in.uv;
    let n = vec3<f32>(dot(in.m0.xyz, in.normal), dot(in.m1.xyz, in.normal), dot(in.m2.xyz, in.normal));
    out.normal = normalize(n);
    out.view_dist = clip.w;
    out.terrain_color = in.terrain_color.rgb;
    out.world_pos = world.xyz;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_diffuse, s_diffuse, in.uv);
    if (tex.a < 0.5) { discard; }

    let n = normalize(in.normal);
    let l = normalize(-u.light_dir.xyz);
    let ndotl = max(dot(n, l), 0.0);
    let lighting = clamp(u.ambient_color.rgb + u.light_color.rgb * ndotl, vec3<f32>(0.0), vec3<f32>(1.0));
    var rgb = tex.rgb * in.terrain_color * lighting;
    let scroll_uv = in.uv + m.scroll.xy * u.time_ms.x;

    if (m.flags.z != 0u) {
        let mask = textureSample(t_mask, s_diffuse, scroll_uv);
        let back = textureSample(t_back, s_diffuse, scroll_uv);
        let back_rgb = back.rgb * in.terrain_color * lighting;
        let blend = max(mask.a, max(mask.r, max(mask.g, mask.b)));
        rgb = mix(rgb, back_rgb, clamp(blend, 0.0, 1.0));
    }

    if (m.flags.x != 0u) {
        let gloss = textureSample(t_gloss, s_diffuse, in.uv).rgb;
        let view_dir = normalize(u.camera_pos.xyz - in.world_pos);
        let fresnel = pow(1.0 - max(dot(view_dir, n), 0.0), 4.0);
        let reflection = mix(u.fog_color.rgb, vec3<f32>(1.0), fresnel);
        rgb += gloss * reflection;
    }

    let fog_f = clamp((u.fog_params.y - in.view_dist) / (u.fog_params.y - u.fog_params.x), 0.0, 1.0);
    rgb = mix(u.fog_color.rgb, rgb, fog_f);
    return vec4<f32>(rgb, 1.0);
}
"#;

const SHADOW_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_shadow: texture_2d<f32>;
@group(0) @binding(2) var s_shadow: sampler;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    out.clip_position = u.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let shadow = textureSample(t_shadow, s_shadow, in.uv).a;
    if (shadow <= 0.01) { discard; }
    return vec4<f32>(0.0, 0.0, 0.0, shadow * 0.45);
}
"#;

const SHADOW_TEXTURE_SHADER: &str = r#"
struct U {
    u_row: vec4<f32>,
    v_row: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var s_diffuse: sampler;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_main(in: VIn) -> VOut {
    let p = vec4<f32>(in.position, 1.0);
    let uv = vec2<f32>(dot(u.u_row, p) + 0.5, dot(u.v_row, p) + 0.5);
    var out: VOut;
    out.clip_position = vec4<f32>(1.0 - uv.x * 2.0, uv.y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_diffuse, s_diffuse, in.uv);
    if (tex.a < 0.5) { discard; }
    return vec4<f32>(1.0, 1.0, 1.0, tex.a);
}
"#;
