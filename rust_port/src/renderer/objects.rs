//! Decorative objects — palms, rocks, trees, grass, etc.
//!
//! Loads .vo meshes (game/vo_loader.rs) for each object type_id referenced by
//! the map, then draws all instances of each type as one instanced draw call.
//! Alpha-tested sampling handles foliage texture cutouts without z-ordering.

use std::collections::{BTreeMap, HashMap};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::game::common::{FOG_START, FOG_END, unpack_rgb};
use crate::game::map::{GameMap, ObjectInstance};
use crate::game::vo_loader::{self, VoMesh};
use crate::renderer::camera::Camera;
use crate::renderer::texture::{decode_texture_bytes, create_texture_from_rgba, create_solid_texture};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
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
}

struct MeshBatch {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instance_buffer: wgpu::Buffer,
    num_instances: u32,
    bind_group: wgpu::BindGroup,
}

pub struct ObjectsRenderer {
    pipeline: wgpu::RenderPipeline,
    batches: Vec<MeshBatch>,
    uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
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
        if map.objects.is_empty() { return None; }

        let strings = stor.get_buf("strings", "String")?;

        // Group instances by type_id so each mesh is loaded once and drawn instanced.
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
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
        let fallback_tex = create_solid_texture(device, queue, [200, 200, 200, 255]);

        let mut batches = Vec::new();
        let mut loaded_types = 0usize;
        let mut failed_types = 0usize;

        for (type_id, instances) in &by_type {
            let id_str = if (*type_id as usize) < strings.arrays_count() {
                strings.get_as_wstr(*type_id as usize)
            } else { continue };
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

            let vertices: Vec<Vertex> = mesh.vertices.iter().map(|v| Vertex {
                position: v.position,
                normal: v.normal,
                uv: v.uv,
            }).collect();
            let indices_u32: Vec<u32> = mesh.indices.iter().map(|&i| i as u32).collect();

            let tex_view = resolve_texture(&paths.texture_path, &mesh, device, queue, &mut tex_cache, read_texture)
                .unwrap_or_else(|| fallback_tex.clone());

            let inst_data: Vec<InstanceData> = instances.iter().map(|obj| {
                instance_matrix(obj, cx, cy, map)
            }).collect();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Objects Mesh VB"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Objects Mesh IB"),
                contents: bytemuck::cast_slice(&indices_u32),
                usage: wgpu::BufferUsages::INDEX,
            });
            let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Objects Inst VB"),
                contents: bytemuck::cast_slice(&inst_data),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Objects BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&tex_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });

            batches.push(MeshBatch {
                vertex_buffer,
                index_buffer,
                num_indices: indices_u32.len() as u32,
                instance_buffer,
                num_instances: inst_data.len() as u32,
                bind_group,
            });
            loaded_types += 1;
        }

        log::info!("objects: {} mesh types loaded, {} skipped, {} total instances drawn",
            loaded_types, failed_types,
            batches.iter().map(|b| b.num_instances).sum::<u32>());

        if batches.is_empty() { return None; }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Objects Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                            wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                            wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                            wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<InstanceData>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { offset: 0,  shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
                            wgpu::VertexAttribute { offset: 16, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },
                            wgpu::VertexAttribute { offset: 32, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },
                            wgpu::VertexAttribute { offset: 48, shader_location: 6, format: wgpu::VertexFormat::Float32x4 },
                            wgpu::VertexAttribute { offset: 64, shader_location: 7, format: wgpu::VertexFormat::Float32x4 },
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
        });

        Some(Self { pipeline, batches, uniform_buffer, fog_color, ambient_color, light_color, light_dir })
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        _camera: &Camera,
        view_proj: glam::Mat4,
    ) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            fog_color: self.fog_color,
            fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            ambient_color: self.ambient_color,
            light_color: self.light_color,
            light_dir: self.light_dir,
        }));

        pass.set_pipeline(&self.pipeline);
        for b in &self.batches {
            pass.set_bind_group(0, &b.bind_group, &[]);
            pass.set_vertex_buffer(0, b.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, b.instance_buffer.slice(..));
            pass.set_index_buffer(b.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..b.num_indices, 0, 0..b.num_instances);
        }
    }
}

/// Build the same static-object transform order used by the original renderer:
/// Rx * Ry * Rz, then uniform scale, then translation into centered render space.
fn instance_matrix(obj: &ObjectInstance, cx: f32, cy: f32, map: &GameMap) -> InstanceData {
    let s = obj.scale.max(0.0001);
    let [terrain_r, terrain_g, terrain_b] = unpack_rgb(map.static_object_color(obj.x, obj.y));

    let (sx, cxr) = obj.angle_x.sin_cos();
    let (sy, cyr) = obj.angle_y.sin_cos();
    let (sz, cz) = obj.angle_z.sin_cos();

    let rx = glam::Mat3::from_cols_array(&[
        1.0, 0.0, 0.0,
        0.0, cxr, sx,
        0.0, -sx, cxr,
    ]);
    let ry = glam::Mat3::from_cols_array(&[
        cyr, 0.0, -sy,
        0.0, 1.0, 0.0,
        sy, 0.0, cyr,
    ]);
    let rz = glam::Mat3::from_cols_array(&[
        cz, sz, 0.0,
        -sz, cz, 0.0,
        0.0, 0.0, 1.0,
    ]);
    let m = rx * ry * rz * s;

    InstanceData {
        row0: [m.x_axis.x, m.y_axis.x, m.z_axis.x, obj.x - cx],
        row1: [m.x_axis.y, m.y_axis.y, m.z_axis.y, obj.y - cy],
        row2: [m.x_axis.z, m.y_axis.z, m.z_axis.z, obj.z],
        row3: [0.0,     0.0,    0.0, 1.0],
        terrain_color: [terrain_r, terrain_g, terrain_b, 1.0],
    }
}

/// Try paths in order: explicit texture_path from the object Id, then the VO's
/// embedded `surfs/texs` reference. Caches decoded textures by path.
fn resolve_texture(
    id_tex_path: &Option<String>,
    mesh: &VoMesh,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<wgpu::TextureView> {
    let mut candidates: Vec<String> = Vec::new();
    if let Some(p) = id_tex_path { candidates.push(p.clone()); }
    if let Some(p) = &mesh.texture_ref {
        // ".\Matrix\Obj\palm\palm00" → "Matrix/Obj/palm/palm00"
        candidates.push(p.trim_start_matches(".\\").replace('\\', "/"));
    }

    for path in &candidates {
        if let Some(cached) = cache.get(path) {
            return Some(cached.clone());
        }
        let Some(data) = read_texture(path) else { continue };
        let Some(rgba) = decode_texture_bytes(&data) else { continue };
        let view = create_texture_from_rgba(device, queue, &rgba);
        cache.insert(path.clone(), view.clone());
        return Some(view);
    }
    None
}

const SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    ambient_color: vec4<f32>,
    light_color: vec4<f32>,
    light_dir: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var s_diffuse: sampler;

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
};

@vertex fn vs_main(in: VIn) -> VOut {
    // model matrix is row-major: compose by dotting each row with (pos, 1).
    let p = vec4<f32>(in.position, 1.0);
    let world = vec4<f32>(dot(in.m0, p), dot(in.m1, p), dot(in.m2, p), dot(in.m3, p));
    let clip = u.view_proj * world;
    var out: VOut;
    out.clip_position = clip;
    out.uv = in.uv;
    // Rotate normal by the upper-left 3x3 of the model matrix (assume uniform scale).
    let n = vec3<f32>(dot(in.m0.xyz, in.normal), dot(in.m1.xyz, in.normal), dot(in.m2.xyz, in.normal));
    out.normal = normalize(n);
    out.view_dist = clip.w;
    out.terrain_color = in.terrain_color.rgb;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_diffuse, s_diffuse, in.uv);
    // Standard alpha-test: discard fragments whose alpha is below 0.5.
    if (tex.a < 0.5) { discard; }

    let n = normalize(in.normal);
    let l = normalize(-u.light_dir.xyz);
    let ndotl = max(dot(n, l), 0.0);
    let lighting = clamp(u.ambient_color.rgb + u.light_color.rgb * ndotl, vec3<f32>(0.0), vec3<f32>(1.0));
    var rgb = tex.rgb * in.terrain_color * lighting;

    let fog_f = clamp((u.fog_params.y - in.view_dist) / (u.fog_params.y - u.fog_params.x), 0.0, 1.0);
    rgb = mix(u.fog_color.rgb, rgb, fog_f);
    return vec4<f32>(rgb, 1.0);
}
"#;
