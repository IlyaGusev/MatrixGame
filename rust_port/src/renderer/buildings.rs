//! Starting-building renderer — ports the draw path of `CMatrixBuilding`
//! (MatrixObjectBuilding.cpp:955-976) plus the group-unit iteration of
//! `CVectorObjectGroup::Draw` (VectorObject.cpp).
//!
//! Each CMAP `buildings/*` row produces one `BuildingInstance`. We group the
//! instances by `kind` (BUILDING_BASE..BUILDING_REPAIR), load the matching
//! `Matrix\Building\bN.cvo` (MatrixObjectBuilding.cpp:158-163), parse it into
//! a list of sub-meshes via `vo_loader::parse_cvo`, and create one GPU batch
//! per sub-mesh. Every sub-mesh shares the building's world transform — the
//! original uses a group-wide `m_GroupToWorldMatrix` pointer and each unit
//! composes it with its local animation matrix. We currently draw frame-0 of
//! each sub-VO, which matches the at-rest pose the original ships with.
//!
//! Shadow generation is deferred: `CMatrixBuilding::RNeed` (MatrixObjectBuilding.cpp:
//! 167-246) has the stencil/proj branches fully commented out in the shipped
//! code, so the original skips them for buildings too.

use std::collections::{BTreeMap, HashMap};

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec4};
use wgpu::util::DeviceExt;

use crate::effects::point_light::PointLightSystem;
use crate::game::common::{unpack_rgb, FOG_END, FOG_START};
use crate::game::map::{BuildingInstance, GameMap};
use crate::game::vo_loader::{self, CvoGroup, MaterialSpec};
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

struct MeshBatch {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instance_buffer: wgpu::Buffer,
    num_instances: u32,
    bind_group: wgpu::BindGroup,
    /// Buildings feeding this batch — kept so the instance transforms can be
    /// re-uploaded when point-light colors change (matches the terrain-tint
    /// refresh the object renderer does in its `takt`).
    buildings: Vec<BuildingInstance>,
    center: [f32; 2],
}

pub struct BuildingsRenderer {
    pipeline: wgpu::RenderPipeline,
    batches: Vec<MeshBatch>,
    uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    time_ms: f32,
    last_point_light_revision: u64,
}

impl BuildingsRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        if map.buildings.is_empty() {
            return None;
        }

        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;

        let mut by_kind: BTreeMap<u8, Vec<&BuildingInstance>> = BTreeMap::new();
        for b in &map.buildings {
            by_kind.entry(b.kind).or_default().push(b);
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
            label: Some("Buildings UB"),
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

        let bgl = create_bgl(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let pipeline = create_pipeline(device, config, &bgl);

        let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
        let fallback_tex = create_solid_texture(device, queue, [200, 200, 200, 255]);
        let black_tex = create_solid_texture(device, queue, [0, 0, 0, 255]);
        let transparent_tex = create_solid_texture(device, queue, [0, 0, 0, 0]);

        let mut batches = Vec::new();
        let mut loaded_kinds = 0usize;
        let mut missing_kinds = 0usize;

        for (kind, instances) in &by_kind {
            let cvo_path = match building_cvo_path(*kind) {
                Some(p) => p,
                None => {
                    missing_kinds += 1;
                    continue;
                }
            };
            let Some(cvo_bytes) = read_texture(&cvo_path) else {
                log::warn!("buildings: CVO not found: {}", cvo_path);
                missing_kinds += 1;
                continue;
            };
            let group: CvoGroup = vo_loader::parse_cvo(&cvo_path, &cvo_bytes);
            if group.units.is_empty() {
                log::warn!("buildings: CVO has no units: {}", cvo_path);
                missing_kinds += 1;
                continue;
            }

            let inst_data: Vec<InstanceData> = instances
                .iter()
                .map(|b| instance_matrix(b, cx, cy, map, None))
                .collect();
            let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Buildings Inst VB"),
                contents: bytemuck::cast_slice(&inst_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

            for unit in &group.units {
                let Some(vo_bytes) = read_texture(&unit.model_path) else {
                    log::debug!("buildings: sub-VO not found: {}", unit.model_path);
                    continue;
                };
                let mesh = match vo_loader::parse_vo(&vo_bytes) {
                    Ok(m) => m,
                    Err(e) => {
                        log::warn!("buildings: parse {} failed: {}", unit.model_path, e);
                        continue;
                    }
                };

                let vertices: Vec<Vertex> = mesh
                    .vertices
                    .iter()
                    .map(|v| Vertex {
                        position: v.position,
                        normal: v.normal,
                        uv: v.uv,
                    })
                    .collect();

                // Use frame 0's surface partition — the original's at-rest
                // pose (CVectorObjectAnim defaults to anim 0, frame 0).
                let Some(frame0) = mesh.frames.first() else {
                    continue;
                };

                // When the CVO unit declares no `Texture=…`, the original
                // falls back to the VO's own embedded surface texture name —
                // `vo->GetSurfaceFileName(0)` (VectorObject.cpp:2509). We
                // read that from the surface's `texture_ref` instead, and
                // otherwise keep the unit's declared diffuse. Gloss/mask/back
                // overrides from the CVO still win, mirroring the composed
                // skin the original builds at VectorObject.cpp:2513.
                let cvo_dir = cvo_path
                    .rsplit_once('/')
                    .map(|(d, _)| format!("{d}/"));
                for surf in &frame0.surfaces {
                    if surf.indices.is_empty() {
                        continue;
                    }

                    let material = if unit.material.diffuse.is_some() {
                        unit.material.clone()
                    } else if let Some(spec) = surf.texture_ref.as_deref() {
                        let surface_mat = vo_loader::parse_material_spec_with_prefix(
                            spec,
                            cvo_dir.as_deref(),
                        );
                        vo_loader::merge_materials(&surface_mat, Some(&unit.material))
                    } else {
                        unit.material.clone()
                    };

                    let (diffuse_view, alpha_test) =
                        resolve_diffuse(&material, device, queue, &mut tex_cache, read_texture)
                            .unwrap_or_else(|| (fallback_tex.clone(), false));
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

                    let vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Buildings Mesh VB"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Buildings Mesh IB"),
                        contents: bytemuck::cast_slice(&surf.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    let mat_uniform =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Buildings Material UB"),
                            contents: bytemuck::bytes_of(&MaterialUniform {
                                flags: [
                                    material.gloss.is_some() as u32,
                                    material.back.is_some() as u32,
                                    material.mask.is_some() as u32,
                                    alpha_test as u32,
                                ],
                                scroll: [material.scroll[0], material.scroll[1], 0.0, 0.0],
                            }),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Buildings BG"),
                        layout: &bgl,
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
                        num_indices: surf.indices.len() as u32,
                        instance_buffer: instance_buffer.clone(),
                        num_instances: inst_data.len() as u32,
                        bind_group,
                        buildings: instances.iter().map(|b| (*b).clone()).collect(),
                        center: [cx, cy],
                    });
                }
            }

            loaded_kinds += 1;
        }

        log::info!(
            "buildings: {} kinds loaded, {} skipped, {} total draw batches ({} instances)",
            loaded_kinds,
            missing_kinds,
            batches.len(),
            batches.iter().map(|b| b.num_instances).sum::<u32>(),
        );

        if batches.is_empty() {
            return None;
        }

        Some(Self {
            pipeline,
            batches,
            uniform_buffer,
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
                    .buildings
                    .iter()
                    .map(|b| instance_matrix(b, cx, cy, map, Some(point_lights)))
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

/// Maps `EBuildingType` (MatrixObjectBuilding.hpp:59-69) to the archive path
/// the original engine feeds into `CVectorObjectGroup::Load`
/// (MatrixObjectBuilding.cpp:158-163). Unknown kinds return `None`.
fn building_cvo_path(kind: u8) -> Option<String> {
    let name = match kind {
        0 => "b0", // BUILDING_BASE
        1 => "b1", // BUILDING_TITAN
        2 => "b2", // BUILDING_PLASMA
        3 => "b3", // BUILDING_ELECTRONIC
        4 => "b4", // BUILDING_ENERGY
        5 => "b5", // BUILDING_REPAIR
        _ => return None,
    };
    Some(format!("Matrix/Building/{name}.cvo"))
}

/// Build the per-building instance transform — ports
/// `CMatrixBuilding::RNeed` (MatrixObjectBuilding.cpp:115-148).
/// The original matrix is row-major: `[rot | 0; pos | 1]`; in column-major
/// glam we compose `translate(pos) * rotZ(angle*90°)` and flatten into the
/// row-layout instance the shared shader expects.
fn instance_matrix(
    b: &BuildingInstance,
    cx: f32,
    cy: f32,
    map: &GameMap,
    point_lights: Option<&PointLightSystem>,
) -> InstanceData {
    let theta = (b.angle as f32) * std::f32::consts::FRAC_PI_2;
    let (s, c) = theta.sin_cos();
    let rot = Mat3::from_cols(
        Vec4::new(c, s, 0.0, 0.0).truncate(),
        Vec4::new(-s, c, 0.0, 0.0).truncate(),
        Vec4::new(0.0, 0.0, 1.0, 0.0).truncate(),
    );
    let [terrain_r, terrain_g, terrain_b] =
        unpack_rgb(map.static_object_color_with_lighting(b.x, b.y, point_lights));
    InstanceData {
        row0: [rot.x_axis.x, rot.y_axis.x, rot.z_axis.x, b.x - cx],
        row1: [rot.x_axis.y, rot.y_axis.y, rot.z_axis.y, b.y - cy],
        row2: [rot.x_axis.z, rot.y_axis.z, rot.z_axis.z, b.build_z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [terrain_r, terrain_g, terrain_b, 1.0],
    }
}

fn resolve_diffuse(
    material: &MaterialSpec,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<(wgpu::TextureView, bool)> {
    let path = material.diffuse.as_ref()?;
    let view = resolve_texture(Some(path), device, queue, cache, read_texture)?;
    let alpha_test =
        vo_loader::resolve_alpha_test_with_txt(path, material.alpha_test, read_texture);
    Some((view, alpha_test))
}

fn resolve_texture(
    tex_path: Option<&String>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<wgpu::TextureView> {
    let path = tex_path?;
    if let Some(v) = cache.get(path) {
        return Some(v.clone());
    }
    let data = read_texture(path)?;
    let rgba = decode_texture_bytes(&data)?;
    let view = create_texture_from_rgba(device, queue, &rgba);
    cache.insert(path.clone(), view.clone());
    Some(view)
}

fn create_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Buildings BGL"),
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

fn create_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Buildings Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Buildings PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Buildings Pipeline"),
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

const SHADER: &str = r#"
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
    if (m.flags.w != 0u && tex.a < 8.0 / 255.0) { discard; }

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
