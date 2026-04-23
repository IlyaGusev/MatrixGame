//! Minimal chassis-only renderer for `CMatrixRobotAI`.
//!
//! Ports the chassis branch of `CMatrixRobot::RNeed` at
//! MatrixObjectRobot.cpp:299-357: each robot's `m_Unit[MRT_CHASSIS]`
//! calls `LoadObject(Matrix\\Robot\\ChassisN.vo)` where N is the
//! `ERobotUnitKind`. Armor / weapons / head are deferred — they
//! compose additional sub-VOs the original anchors to bones on the
//! chassis, and a faithful port needs the per-frame anchor table
//! from `CVectorObjectAnim::m_Animation`.
//!
//! Instance buffers are rebuilt each frame from the live arena so
//! robots that get spawned mid-session immediately render.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{unpack_rgb, FOG_END, FOG_START};
use crate::matrix_game::effects::point_light::PointLightSystem;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{MapStatic, Objects, ObjectType};
use crate::matrix_game::units::{ChassisKind, Robot};
use crate::matrix_lib::three_g::texture::{
    create_solid_texture, create_texture_from_rgba, decode_texture_bytes,
};
use crate::matrix_lib::three_g::vector_object::{self, MaterialSpec};

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
    /// Unused for robots (no sub-unit animation) — kept only because
    /// we reuse the buildings shader, which reads this attribute.
    unit_offset: [f32; 4],
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

/// One draw call — a single chassis surface. A chassis may have
/// multiple surfaces (e.g. track + body split into two), so a chassis
/// kind produces N batches.
struct MeshBatch {
    chassis: ChassisKind,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    /// Shared instance buffer with sibling batches of the same chassis
    /// kind (the GPU uploads the same instance list regardless of
    /// which surface is being drawn).
    bind_group: wgpu::BindGroup,
}

/// Per-chassis instance buffer + counter. Writing to `buffer` each
/// frame is cheap — robot counts are small (<20 in practice).
struct ChassisInstances {
    buffer: wgpu::Buffer,
    capacity: u32,
    num_instances: u32,
}

pub struct RobotsRenderer {
    pipeline: wgpu::RenderPipeline,
    batches: Vec<MeshBatch>,
    /// Indexed by `ChassisKind as usize`.
    chassis_instances: Vec<ChassisInstances>,
    uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    time_ms: f32,
    /// World-center offset applied to every instance so local-
    /// frame vertices share the terrain renderer's origin
    /// (matches the pattern `BuildingsRenderer` uses).
    center: [f32; 2],
}

const MAX_ROBOTS_PER_CHASSIS: u32 = 64;

impl RobotsRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;

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
            label: Some("Robots UB"),
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

        let chassis_list = [
            ChassisKind::Pneumatic,
            ChassisKind::Wheel,
            ChassisKind::Track,
            ChassisKind::AntiGravity,
            ChassisKind::Hovercraft,
        ];

        let mut chassis_instances: Vec<ChassisInstances> = Vec::with_capacity(chassis_list.len());
        for _ in 0..chassis_list.len() {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Robots Inst VB"),
                size: (MAX_ROBOTS_PER_CHASSIS as u64)
                    * std::mem::size_of::<InstanceData>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            chassis_instances.push(ChassisInstances {
                buffer,
                capacity: MAX_ROBOTS_PER_CHASSIS,
                num_instances: 0,
            });
        }

        let mut batches = Vec::new();
        for chassis in &chassis_list {
            let n = chassis_kind_index(*chassis) as u32 + 1;
            let vo_path = format!("Matrix/Robot/Chassis{}.vo", n);
            let Some(vo_bytes) = read_texture(&vo_path) else {
                log::warn!("robots: VO not found: {}", vo_path);
                continue;
            };
            let mesh = match vector_object::parse_vo(&vo_bytes) {
                Ok(m) => m,
                Err(e) => {
                    log::warn!("robots: parse {} failed: {}", vo_path, e);
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

            // Frame 0 surface partition (default at-rest pose).
            let Some(frame0) = mesh.frames.first() else {
                continue;
            };
            let vo_dir = vo_path.rsplit_once('/').map(|(d, _)| format!("{d}/"));

            // Fallback diffuse: `Matrix/Robot/ChassisN` (the texture
            // suffix the C++ LoadObject strips off). Original surfaces
            // that don't declare their own `Texture=` inherit from the
            // VO's embedded texture_ref; if that is still missing, use
            // this top-level skin.
            let top_diffuse = format!("Matrix/Robot/Chassis{}", n);
            let top_gloss = format!("Matrix/Robot/Chassis{}_gloss", n);

            // Instance buffer ptr for bind-group-free reuse.
            let inst = &chassis_instances[chassis_kind_index(*chassis)];
            let _ = inst;

            for surf in &frame0.surfaces {
                if surf.indices.is_empty() {
                    continue;
                }

                let mut material: MaterialSpec = MaterialSpec {
                    diffuse: Some(top_diffuse.clone()),
                    gloss: Some(top_gloss.clone()),
                    ..Default::default()
                };
                if let Some(spec) = surf.texture_ref.as_deref() {
                    let surface_mat = vector_object::parse_material_spec_with_prefix(
                        spec,
                        vo_dir.as_deref(),
                    );
                    material = vector_object::merge_materials(&surface_mat, Some(&material));
                }

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
                        label: Some("Robots Mesh VB"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Robots Mesh IB"),
                    contents: bytemuck::cast_slice(&surf.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let mat_uniform =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Robots Material UB"),
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
                    label: Some("Robots BG"),
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
                    chassis: *chassis,
                    vertex_buffer,
                    index_buffer,
                    num_indices: surf.indices.len() as u32,
                    bind_group,
                });
            }
        }

        log::info!("robots: {} chassis batches loaded", batches.len());
        if batches.is_empty() {
            return None;
        }

        Some(Self {
            pipeline,
            batches,
            chassis_instances,
            uniform_buffer,
            fog_color,
            ambient_color,
            light_color,
            light_dir,
            time_ms: 0.0,
            center: [cx, cy],
        })
    }

    pub fn takt(&mut self, dt_ms: f32) {
        self.time_ms += dt_ms;
        if self.time_ms > 1_000_000.0 {
            self.time_ms -= 1_000_000.0;
        }
    }

    /// Walk live `Robot`s and upload one instance per chassis kind.
    pub fn sync_robots(
        &mut self,
        queue: &wgpu::Queue,
        objs: &Objects,
        map: &GameMap,
        point_lights: &PointLightSystem,
    ) {
        let [cx, cy] = self.center;

        // Bucket by chassis kind first so we can write each buffer once.
        let mut buckets: Vec<Vec<InstanceData>> =
            (0..self.chassis_instances.len()).map(|_| Vec::new()).collect();

        for id in objs.iter_live() {
            let Some(obj) = objs.get(id) else { continue };
            if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
                continue;
            }
            let robot: &Robot = unsafe {
                &*(obj as *const dyn MapStatic as *const Robot)
            };
            let idx = chassis_kind_index(robot.chassis);
            if idx >= buckets.len() {
                continue;
            }
            if buckets[idx].len() as u32 >= self.chassis_instances[idx].capacity {
                continue;
            }
            buckets[idx].push(robot_instance(robot, cx, cy, map, Some(point_lights)));
        }

        for (idx, bucket) in buckets.iter().enumerate() {
            let inst = &mut self.chassis_instances[idx];
            inst.num_instances = bucket.len() as u32;
            if !bucket.is_empty() {
                queue.write_buffer(&inst.buffer, 0, bytemuck::cast_slice(bucket));
            }
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
            let idx = chassis_kind_index(batch.chassis);
            let inst = &self.chassis_instances[idx];
            if inst.num_instances == 0 {
                continue;
            }
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, inst.buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..batch.num_indices, 0, 0..inst.num_instances);
        }
    }
}

fn chassis_kind_index(c: ChassisKind) -> usize {
    match c {
        ChassisKind::Pneumatic => 0,
        ChassisKind::Wheel => 1,
        ChassisKind::Track => 2,
        ChassisKind::AntiGravity => 3,
        ChassisKind::Hovercraft => 4,
    }
}

/// Build one instance transform for a live `Robot`. The original robot
/// matrix composes `rotate(angle) * translate(pos_x, pos_y, pos_z)`
/// (MatrixObjectRobot.cpp:359-480). We only have yaw-0 since AI isn't
/// ported; position comes from `Robot::pos_{x,y,z}`, which `logic_takt`
/// lifts for the platform-rise animation.
fn robot_instance(
    r: &Robot,
    cx: f32,
    cy: f32,
    map: &GameMap,
    point_lights: Option<&PointLightSystem>,
) -> InstanceData {
    let [terrain_r, terrain_g, terrain_b] =
        unpack_rgb(map.static_object_color_with_lighting(r.pos_x, r.pos_y, point_lights));
    InstanceData {
        row0: [1.0, 0.0, 0.0, r.pos_x - cx],
        row1: [0.0, 1.0, 0.0, r.pos_y - cy],
        row2: [0.0, 0.0, 1.0, r.pos_z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [terrain_r, terrain_g, terrain_b, 1.0],
        unit_offset: [0.0, 0.0, 0.0, 0.0],
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
        vector_object::resolve_alpha_test_with_txt(path, material.alpha_test, read_texture);
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
        label: Some("Robots BGL"),
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
        label: Some("Robots Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Robots PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Robots Pipeline"),
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
                        wgpu::VertexAttribute {
                            offset: 80,
                            shader_location: 8,
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

const SHADER: &str = include_str!("../../shaders/object_building.wgsl");
