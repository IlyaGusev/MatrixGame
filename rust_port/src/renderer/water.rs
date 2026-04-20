//! Water rendering — ports CMatrixWater + BuildWater + DrawWater.
//!
//! Shoreline alpha: per-group 64×64 alpha texture (ports BuildWater from
//! MatrixMapGroup.cpp:366-452), packed into an atlas, sampled per-fragment.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::game::map::{GameMap, InshorePreSpawn, GLOBAL_SCALE};
use crate::renderer::camera::Camera;

use crate::game::common::{unpack_rgb, FOG_END, FOG_START};

/// Ports INSHORE_SPEED (MatrixWater.hpp:17) — animation `t` advances by
/// `INSHORE_SPEED * step_ms * wave.speed` each tick.
const INSHORE_SPEED: f32 = 0.0005;
/// Max simultaneous waves per group (INSHOREWAVES_CNT, MatrixWater.hpp:19).
const INSHOREWAVES_CNT: usize = 7;

const WATER_LEVEL: f32 = -2.0;
const WATER_SIZE: usize = 16;
const WATER_ALPHA_SIZE: usize = 64;
const MAP_GROUP_SIZE: i32 = 10;
const WATER_TEXTURE_SCALE: f32 = 1.0 / 16.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WaterVertex {
    position: [f32; 3],
    normal: [f32; 3],
    water_uv: [f32; 2],
    alpha_uv: [f32; 2],
}

impl WaterVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
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
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WaterUniforms {
    view_proj: [[f32; 4]; 4],
    normal_mat: [[f32; 4]; 4],
    water_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    params: [f32; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
}

pub struct Water {
    water_instances: Vec<WaterInstance>,
    water_draws: Vec<WaterDraw>,
    fog_color: [f32; 4],
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    ocean_vertex_buffer: wgpu::Buffer,
    ocean_index_buffer: wgpu::Buffer,
    ocean_num_indices: u32,
    ocean_capacity_instances: usize,

    uniform_buffer: wgpu::Buffer,
    solid_bind_group: wgpu::BindGroup,
    solid_pipeline: wgpu::RenderPipeline,
    pipeline: wgpu::RenderPipeline,

    water_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    water_scale: f32,
    water_normal_len: f32,
    accum_ms: f32,
    angle: i32,
    phase_offsets: [u16; WATER_SIZE * WATER_SIZE],
    map_group_w: i32,
    map_group_h: i32,
    map_half_w: f32,
    map_half_h: f32,
    group_world: f32,
    has_group: std::collections::HashSet<(i32, i32)>,

    inshore: Option<InshoreSystem>,
}

/// Shoreline foam runtime. One per-group pool of active `Wave`s is advanced
/// each tick; every visible frame all live waves are packed into a single
/// instance buffer and drawn as textured quads.
///
/// Mirrors SInshorewave + SPreInshorewave from MatrixWater.hpp plus the
/// per-group bookkeeping in CMatrixMapGroup::GraphicTakt (MatrixMapGroup.cpp:
/// 752-789). Wave spawn, motion, and alpha fade all match the original.
struct InshoreSystem {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    prespawns: Vec<Vec<PreSpawn>>,
    active_waves: Vec<Vec<Wave>>,
    rng: u32,
    color_rgb: [f32; 3],
    map_half_w: f32,
    map_half_h: f32,
    draw_count: u32,
}

#[derive(Clone, Copy)]
struct PreSpawn {
    pos: [f32; 2],
    dir: [f32; 2],
    used: bool,
}

#[derive(Clone, Copy)]
struct Wave {
    pos: [f32; 2],
    dir: [f32; 2],
    len: f32,
    t: f32,
    speed: f32,
    scale: f32,
    slot: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InshoreVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

/// Per-wave instance data: encodes the 2x2 rotation+scale, the XY translation,
/// Z (water level), and the fade color. The vertex shader rebuilds a 4×4
/// world matrix from these six floats.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InshoreInstance {
    x_axis: [f32; 2], // (dir.x*scale, dir.y*scale)
    y_axis: [f32; 2], // (-dir.y*scale, dir.x*scale)
    translation: [f32; 3],
    _pad0: f32,
    color: [f32; 4], // rgba, a = fade
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InshoreUniforms {
    view_proj: [[f32; 4]; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
}

#[derive(Clone, Copy)]
struct WaterInstance {
    x0: f32,
    y0: f32,
}

struct WaterDraw {
    bind_group: wgpu::BindGroup,
    index_start: u32,
    index_count: u32,
}

struct WaterPreset {
    /// Preset name actually resolved (may differ from `map.water_name` if we
    /// fell back to the first configured preset).
    resolved_name: String,
    water_path: String,
    mirror_path: String,
    inshore_path: String,
}

/// Port of `MatrixMapPrepare.cpp:1208-1219` + `MatrixWater.cpp:213-216`:
/// look up `Water/<water_name>` in the serialized BlockPar tree and read
/// its `water` / `mirror` params. Falls back to the first configured preset
/// when the named one is absent (matches the original's
/// `m_WaterName = BlockGet(PAR_SOURCE_WATER)->BlockGetName(0)` rescue), and
/// to the shipped `water_blue` paths when no BlockPar data is available at
/// all (e.g. robots.dat missing on native).
fn resolve_water_preset(matrix_data: Option<&Storage>, water_name: &str) -> WaterPreset {
    let fallback = || WaterPreset {
        resolved_name: water_name.to_string(),
        water_path: "Matrix/Textures/Water/1".to_string(),
        mirror_path: "Matrix/Textures/Water/MIRROR".to_string(),
        inshore_path: "Matrix/Textures/Water/inshorewave".to_string(),
    };

    let Some(stor) = matrix_data else {
        return fallback();
    };
    let Some(water_root) = stor.block_record("da", "Water") else {
        return fallback();
    };

    let (resolved_name, preset_record) = match stor.block_record(&water_root, water_name) {
        Some(rec) => (water_name.to_string(), rec),
        None => match stor.first_block(&water_root) {
            Some((name, rec)) => {
                log::warn!(
                    "water: preset '{}' not in Water block; falling back to '{}'",
                    water_name,
                    name
                );
                (name, rec)
            }
            None => return fallback(),
        },
    };

    let fix_path = |p: Option<String>| p.map(|s| s.replace('\\', "/"));
    let water_path = fix_path(stor.block_param(&preset_record, "water"))
        .unwrap_or_else(|| "Matrix/Textures/Water/1".to_string());
    let mirror_path = fix_path(stor.block_param(&preset_record, "mirror"))
        .unwrap_or_else(|| "Matrix/Textures/Water/MIRROR".to_string());
    let inshore_path = fix_path(stor.block_param(&preset_record, "inshore"))
        .unwrap_or_else(|| "Matrix/Textures/Water/inshorewave".to_string());
    WaterPreset {
        resolved_name,
        water_path,
        mirror_path,
        inshore_path,
    }
}

impl Water {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        stor: &Storage,
        matrix_data: Option<&Storage>,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        let groups_buf = stor.get_buf("groups", "Data")?;
        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;

        let [wr, wg, wb] = unpack_rgb(map.water_color);
        let water_color = [wr, wg, wb, 1.0];
        let [lr, lg, lb] = unpack_rgb(map.light_main_color);
        let light_color = [lr, lg, lb, 1.0];
        let ld = map.light_main_dir;
        let light_dir = [ld[0], ld[1], ld[2], 0.0];
        let water_scale = GLOBAL_SCALE * MAP_GROUP_SIZE as f32 / WATER_SIZE as f32;
        let water_preset = resolve_water_preset(matrix_data, &map.water_name);

        let mut water_groups: Vec<(i32, i32)> = Vec::new();
        for gi in 0..groups_buf.arrays_count() {
            let raw = groups_buf.get_bytes(gi);
            if raw.len() < 4 {
                continue;
            }
            let gx = u16::from_le_bytes([raw[0], raw[1]]) as i32;
            let gy = u16::from_le_bytes([raw[2], raw[3]]) as i32;
            let w = MAP_GROUP_SIZE.min(map.size_x as i32 - gx);
            let h = MAP_GROUP_SIZE.min(map.size_y as i32 - gy);
            let mut min_z = f32::INFINITY;
            for py in 0..=h {
                for px in 0..=w {
                    min_z = min_z.min(map.point((gx + px) as usize, (gy + py) as usize).z);
                }
            }
            if min_z < 0.0 {
                water_groups.push((gx, gy));
            }
        }
        if water_groups.is_empty() {
            return None;
        }

        let up_level: f32 = -1.0;
        let down_level: f32 = -20.1;
        let alpha_step = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE / WATER_ALPHA_SIZE as f32;
        let mut alpha_images = Vec::with_capacity(water_groups.len());

        for &(gx, gy) in &water_groups {
            let group_x0 = gx as f32 * GLOBAL_SCALE;
            let group_y0 = gy as f32 * GLOBAL_SCALE;
            let mut alpha_image =
                image::GrayImage::new(WATER_ALPHA_SIZE as u32, WATER_ALPHA_SIZE as u32);

            for j in 0..WATER_ALPHA_SIZE {
                for i in 0..WATER_ALPHA_SIZE {
                    let wx = (i as f32 + 0.5) * alpha_step + group_x0;
                    let wy = (j as f32 + 0.5) * alpha_step + group_y0;
                    let wz = sample_height_for_water(map, wx, wy);

                    let zz: u8 = if wz < down_level {
                        255
                    } else if wz > up_level {
                        0
                    } else {
                        (255.0 - ((wz - down_level) / (up_level - down_level) * 255.0)) as u8
                    };

                    alpha_image.put_pixel(i as u32, j as u32, image::Luma([zz]));
                }
            }
            alpha_images.push(alpha_image);
        }

        let mut water_instances = Vec::new();
        for &(gx, gy) in &water_groups {
            let group_x0 = gx as f32 * GLOBAL_SCALE;
            let group_y0 = gy as f32 * GLOBAL_SCALE;
            water_instances.push(WaterInstance {
                x0: group_x0 - cx,
                y0: group_y0 - cy,
            });
        }

        let map_group_w = ((map.size_x as f32 + MAP_GROUP_SIZE as f32 - 1.0)
            / MAP_GROUP_SIZE as f32)
            .ceil() as i32;
        let map_group_h = ((map.size_y as f32 + MAP_GROUP_SIZE as f32 - 1.0)
            / MAP_GROUP_SIZE as f32)
            .ceil() as i32;
        let group_world = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE;
        let mut has_group = std::collections::HashSet::new();
        for gi in 0..groups_buf.arrays_count() {
            let raw = groups_buf.get_bytes(gi);
            if raw.len() < 4 {
                continue;
            }
            let gx = u16::from_le_bytes([raw[0], raw[1]]) as i32 / MAP_GROUP_SIZE;
            let gy = u16::from_le_bytes([raw[2], raw[3]]) as i32 / MAP_GROUP_SIZE;
            has_group.insert((gx, gy));
        }
        let phase_offsets = build_phase_offsets();
        let (lattice_z, lattice_normals) =
            build_wave_lattice(0, map.water_normal_len, &phase_offsets);
        let all_verts =
            build_instance_vertices(&water_instances, &lattice_z, &lattice_normals, water_scale);
        let all_idxs = build_instance_indices(water_instances.len());

        let load_tex = |path: &str, fb: [u8; 4]| -> wgpu::TextureView {
            read_texture(path)
                .and_then(|d| super::texture::decode_texture_bytes(&d))
                .map(|rgba| super::texture::create_texture_from_rgba(device, queue, &rgba))
                .unwrap_or_else(|| super::texture::create_solid_texture(device, queue, fb))
        };
        log::info!(
            "water: requested='{}' resolved='{}' water='{}' mirror='{}'",
            map.water_name,
            water_preset.resolved_name,
            water_preset.water_path,
            water_preset.mirror_path
        );
        let water_tex = load_tex(&water_preset.water_path, [30, 80, 120, 255]);
        let mirror_tex = load_tex(&water_preset.mirror_path, [100, 120, 140, 128]);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water VB"),
            contents: bytemuck::cast_slice(&all_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water IB"),
            contents: bytemuck::cast_slice(&all_idxs),
            usage: wgpu::BufferUsages::INDEX,
        });
        let ocean_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ocean VB"),
            contents: &[],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let ocean_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ocean IB"),
            contents: &[],
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        let [sr, sg, sb] = unpack_rgb(map.sky_color);
        let fog_color = [sr, sg, sb, 1.0];
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water UB"),
            contents: bytemuck::bytes_of(&WaterUniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                normal_mat: glam::Mat4::IDENTITY.to_cols_array_2d(),
                water_color,
                light_color,
                light_dir,
                params: [1.0, 0.0, 0.0, 0.0],
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let alpha_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water BGL"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
            ],
        });
        let white_alpha_tex = {
            let white = [255u8; WATER_ALPHA_SIZE * WATER_ALPHA_SIZE];
            let tex = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("Water Solid Alpha"),
                    size: wgpu::Extent3d {
                        width: WATER_ALPHA_SIZE as u32,
                        height: WATER_ALPHA_SIZE as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &white,
            );
            tex.create_view(&Default::default())
        };

        let make_bind_group = |alpha_view: &wgpu::TextureView, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&water_tex),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&mirror_tex),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(alpha_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&alpha_sampler),
                    },
                ],
            })
        };
        let solid_bind_group = make_bind_group(&white_alpha_tex, "Water Solid BG");
        let idxs_per_instance = (WATER_SIZE * WATER_SIZE * 6) as u32;
        let mut water_draws = Vec::with_capacity(alpha_images.len());
        for (group_idx, alpha_image) in alpha_images.iter().enumerate() {
            let alpha_tex = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("Water Alpha"),
                    size: wgpu::Extent3d {
                        width: WATER_ALPHA_SIZE as u32,
                        height: WATER_ALPHA_SIZE as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                alpha_image.as_raw(),
            );
            let alpha_view = alpha_tex.create_view(&Default::default());
            let bind_group = make_bind_group(&alpha_view, "Water Alpha BG");
            water_draws.push(WaterDraw {
                bind_group,
                index_start: group_idx as u32 * idxs_per_instance,
                index_count: idxs_per_instance,
            });
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"),
            source: wgpu::ShaderSource::Wgsl(WATER_SHADER.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let solid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Solid Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main_solid"),
                buffers: &[WaterVertex::desc()],
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
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[WaterVertex::desc()],
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
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let inshore = build_inshore_system(
            device,
            queue,
            config,
            map,
            &water_preset,
            read_texture,
            fog_color,
        );

        Some(Self {
            water_instances,
            water_draws,
            fog_color,
            vertex_buffer,
            index_buffer,
            num_indices: all_idxs.len() as u32,
            ocean_vertex_buffer,
            ocean_index_buffer,
            ocean_num_indices: 0,
            ocean_capacity_instances: 0,
            uniform_buffer,
            solid_bind_group,
            solid_pipeline,
            pipeline,
            water_color,
            light_color,
            light_dir,
            water_scale,
            water_normal_len: map.water_normal_len,
            accum_ms: 0.0,
            angle: 0,
            phase_offsets,
            map_group_w,
            map_group_h,
            map_half_w: map.world_width() * 0.5,
            map_half_h: map.world_height() * 0.5,
            group_world,
            has_group,
            inshore,
        })
    }

    pub fn takt(
        &mut self,
        dt_ms: f32,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &Camera,
        map: &GameMap,
    ) {
        if let Some(inshore) = self.inshore.as_mut() {
            // CMatrixMap::Takt (MatrixMap.cpp:2505-2510) only iterates
            // `m_VisibleGroups` — we replay the same check here via a full
            // port of CalcMapGroupVisibility/CheckCandidate so off-screen
            // waves freeze mid-animation instead of churning.
            let visible = visible_groups_mask(camera, map);
            inshore.takt(dt_ms, &visible, map.group_w);
        }

        self.accum_ms += dt_ms;
        let mut steps = 0;
        while self.accum_ms >= 60.0 {
            self.accum_ms -= 60.0;
            steps += 1;
        }
        if steps == 0 {
            return;
        }

        self.angle += steps;
        let (lattice_z, lattice_normals) =
            build_wave_lattice(self.angle, self.water_normal_len, &self.phase_offsets);

        let water_verts = build_instance_vertices(
            &self.water_instances,
            &lattice_z,
            &lattice_normals,
            self.water_scale,
        );
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&water_verts));
    }

    pub fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        camera: &Camera,
        view_proj: glam::Mat4,
        view: glam::Mat4,
    ) {
        let world = glam::Mat4::from_scale(glam::Vec3::splat(self.water_scale));
        let normal_mat = (view * world).inverse().transpose();

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&WaterUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                normal_mat: normal_mat.to_cols_array_2d(),
                water_color: self.water_color,
                light_color: self.light_color,
                light_dir: self.light_dir,
                params: [1.0, 0.0, 0.0, 0.0],
                fog_color: self.fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
        );

        let ocean_instances = self.collect_visible_ocean_instances(camera);
        if !ocean_instances.is_empty() {
            let (lattice_z, lattice_normals) =
                build_wave_lattice(self.angle, self.water_normal_len, &self.phase_offsets);
            self.ensure_ocean_capacity(device, ocean_instances.len());
            let ocean_verts = build_instance_vertices(
                &ocean_instances,
                &lattice_z,
                &lattice_normals,
                self.water_scale,
            );
            let ocean_idxs = build_instance_indices(ocean_instances.len());
            self.ocean_num_indices = ocean_idxs.len() as u32;
            queue.write_buffer(
                &self.ocean_vertex_buffer,
                0,
                bytemuck::cast_slice(&ocean_verts),
            );
            queue.write_buffer(
                &self.ocean_index_buffer,
                0,
                bytemuck::cast_slice(&ocean_idxs),
            );
            pass.set_pipeline(&self.solid_pipeline);
            pass.set_bind_group(0, &self.solid_bind_group, &[]);
            pass.set_vertex_buffer(0, self.ocean_vertex_buffer.slice(..));
            pass.set_index_buffer(self.ocean_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.ocean_num_indices, 0, 0..1);
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for draw in &self.water_draws {
            pass.set_bind_group(0, &draw.bind_group, &[]);
            pass.draw_indexed(
                draw.index_start..(draw.index_start + draw.index_count),
                0,
                0..1,
            );
        }

        if let Some(inshore) = self.inshore.as_mut() {
            inshore.render(device, queue, pass, view_proj, self.fog_color);
        }
    }

    fn ensure_ocean_capacity(&mut self, device: &wgpu::Device, instance_count: usize) {
        if instance_count <= self.ocean_capacity_instances {
            return;
        }
        let verts_per_instance = (WATER_SIZE + 1) * (WATER_SIZE + 1);
        let idxs_per_instance = WATER_SIZE * WATER_SIZE * 6;
        let new_capacity = instance_count.next_power_of_two().max(16);
        self.ocean_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ocean VB"),
            size: (new_capacity * verts_per_instance * std::mem::size_of::<WaterVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ocean_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ocean IB"),
            size: (new_capacity * idxs_per_instance * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ocean_capacity_instances = new_capacity;
    }

    fn collect_visible_ocean_instances(&self, camera: &Camera) -> Vec<WaterInstance> {
        let quad3 = camera.frustum_bounds_on_plane_zup(WATER_LEVEL);
        let mut quad = quad3.map(|p| glam::Vec2::new(p.x, p.y));

        // MatrixVisiCalc.cpp expands the projected top edge by one group width
        // before testing visible groups. Without that padding, ocean tiles can
        // pop out at the left/right screen edges when the top frustum edge is
        // nearly parallel to the water plane.
        let top_dir = quad[1] - quad[0];
        let top_len = top_dir.length();
        if top_len > 1e-5 {
            let ex = top_dir / top_len * (self.group_world * 2.0);
            quad[0] -= ex;
            quad[1] += ex;
        }

        let mut min_x = quad[0].x;
        let mut max_x = quad[0].x;
        let mut min_y = quad[0].y;
        let mut max_y = quad[0].y;
        for p in &quad[1..] {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
        min_x -= self.group_world;
        max_x += self.group_world;
        min_y -= self.group_world;
        max_y += self.group_world;

        // Frustum points are in centered render-space coordinates. Convert them back to the
        // original uncentered map/world group space before selecting group indices, matching
        // the original visibility code that works from GetPos0() world coordinates.
        let world_min_x = min_x + self.map_half_w;
        let world_max_x = max_x + self.map_half_w;
        let world_min_y = min_y + self.map_half_h;
        let world_max_y = max_y + self.map_half_h;

        let iminx = (world_min_x / self.group_world).floor() as i32 - 1;
        let imaxx = (world_max_x / self.group_world).ceil() as i32 + 1;
        let iminy = (world_min_y / self.group_world).floor() as i32 - 1;
        let imaxy = (world_max_y / self.group_world).ceil() as i32 + 1;

        let mut out = Vec::new();
        for gy in iminy..=imaxy {
            for gx in iminx..=imaxx {
                let x0 = gx as f32 * self.group_world - self.map_half_w;
                let y0 = gy as f32 * self.group_world - self.map_half_h;
                let rect_min = glam::Vec2::new(x0, y0);
                let rect_max = glam::Vec2::new(x0 + self.group_world, y0 + self.group_world);
                if !quad_intersects_rect(&quad, rect_min, rect_max) {
                    continue;
                }
                out.push(WaterInstance { x0, y0 });
            }
        }
        out
    }
}

fn sample_height_for_water(map: &GameMap, wx: f32, wy: f32) -> f32 {
    let scaledx = wx / GLOBAL_SCALE;
    let scaledy = wy / GLOBAL_SCALE;
    let ix = scaledx as i32;
    let iy = scaledy as i32;

    if ix >= 0 && iy >= 0 && ix < map.size_x as i32 && iy < map.size_y as i32 {
        let unit = map.unit(ix as usize, iy as usize);
        if unit.flags & crate::game::common::CELLFLAG_BRIDGE != 0 {
            let kx = scaledx - ix as f32;
            let ky = scaledy - iy as f32;

            let z0 = map.point(ix as usize, iy as usize).z;
            let z1 = map.point(ix as usize + 1, iy as usize).z;
            let z2 = map.point(ix as usize, iy as usize + 1).z;
            let z3 = map.point(ix as usize + 1, iy as usize + 1).z;

            let top = z0 + (z1 - z0) * kx;
            let bottom = z2 + (z3 - z2) * kx;
            return top + (bottom - top) * ky;
        }
    }

    map.get_z(wx, wy)
}

fn build_phase_offsets() -> [u16; WATER_SIZE * WATER_SIZE] {
    let mut seed = 1u32;
    let mut out = [0u16; WATER_SIZE * WATER_SIZE];
    for slot in &mut out {
        seed = seed.wrapping_mul(214013).wrapping_add(2531011);
        *slot = (((seed >> 16) & 0x7fff) as u16) % 255;
    }
    out
}

fn build_wave_lattice(
    angle: i32,
    water_normal_len: f32,
    phase_offsets: &[u16; WATER_SIZE * WATER_SIZE],
) -> (Vec<f32>, Vec<[f32; 3]>) {
    const AMP: f32 = 512.0 / 2500.0;
    const SIN_TABLE_SIZE: i32 = 256;

    let mut h = vec![0.0f32; WATER_SIZE * WATER_SIZE];
    for (idx, phase) in phase_offsets.iter().enumerate() {
        let phase_idx = ((angle + *phase as i32) & (SIN_TABLE_SIZE - 1)) as f32;
        h[idx] = AMP * (phase_idx * std::f32::consts::TAU / SIN_TABLE_SIZE as f32).sin();
    }

    let mut z = vec![0.0f32; (WATER_SIZE + 1) * (WATER_SIZE + 1)];
    let mut k1 = 0usize;
    for j in 0..WATER_SIZE {
        let row_first = h[k1];
        for i in 0..WATER_SIZE {
            z[j * (WATER_SIZE + 1) + i] = h[k1];
            k1 += 1;
        }
        z[j * (WATER_SIZE + 1) + WATER_SIZE] = row_first;
    }
    for i in 0..=WATER_SIZE {
        z[WATER_SIZE * (WATER_SIZE + 1) + i] = z[i];
    }

    let mut normals = vec![[0.0, 0.0, water_normal_len]; (WATER_SIZE + 1) * (WATER_SIZE + 1)];
    for j in 0..=WATER_SIZE {
        for i in 0..=WATER_SIZE {
            let c = j * (WATER_SIZE + 1) + i;
            let il = if i > 0 { i - 1 } else { WATER_SIZE - 1 };
            let ir = if i < WATER_SIZE { i + 1 } else { 1 };
            let ju = if j > 0 { j - 1 } else { WATER_SIZE - 1 };
            let jd = if j < WATER_SIZE { j + 1 } else { 1 };

            let cl = j * (WATER_SIZE + 1) + il;
            let cr = j * (WATER_SIZE + 1) + ir;
            let cu = ju * (WATER_SIZE + 1) + i;
            let cd = jd * (WATER_SIZE + 1) + i;

            let mut n = glam::Vec3::new(z[cl] - z[cr], z[cu] - z[cd], 1.0);
            n = n.normalize_or_zero() * water_normal_len;
            normals[c] = [n.x, n.y, n.z];
        }
    }

    (z, normals)
}

fn build_instance_vertices(
    instances: &[WaterInstance],
    lattice_z: &[f32],
    lattice_normals: &[[f32; 3]],
    water_scale: f32,
) -> Vec<WaterVertex> {
    let mut verts = Vec::with_capacity(instances.len() * (WATER_SIZE + 1) * (WATER_SIZE + 1));
    for inst in instances {
        let mut tv = 0.0f32;
        for j in 0..=WATER_SIZE {
            let mut tu = 0.0f32;
            for i in 0..=WATER_SIZE {
                let idx = j * (WATER_SIZE + 1) + i;
                verts.push(WaterVertex {
                    position: [
                        inst.x0 + i as f32 * water_scale,
                        inst.y0 + j as f32 * water_scale,
                        WATER_LEVEL + lattice_z[idx] * water_scale,
                    ],
                    normal: lattice_normals[idx],
                    water_uv: [tu, tv],
                    alpha_uv: [i as f32 / WATER_SIZE as f32, j as f32 / WATER_SIZE as f32],
                });
                tu += WATER_TEXTURE_SCALE;
            }
            tv += WATER_TEXTURE_SCALE;
        }
    }
    verts
}

fn build_instance_indices(instance_count: usize) -> Vec<u32> {
    let verts_per_instance = ((WATER_SIZE + 1) * (WATER_SIZE + 1)) as u32;
    let mut idxs = Vec::with_capacity(instance_count * WATER_SIZE * WATER_SIZE * 6);
    for inst in 0..instance_count as u32 {
        let base = inst * verts_per_instance;
        let stride = (WATER_SIZE + 1) as u32;
        for j in 0..WATER_SIZE as u32 {
            for i in 0..WATER_SIZE as u32 {
                let tl = base + j * stride + i;
                idxs.push(tl);
                idxs.push(tl + stride);
                idxs.push(tl + 1);
                idxs.push(tl + 1);
                idxs.push(tl + stride);
                idxs.push(tl + stride + 1);
            }
        }
    }
    idxs
}

fn quad_intersects_rect(
    quad: &[glam::Vec2; 4],
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
) -> bool {
    let rect = [
        glam::Vec2::new(rect_min.x, rect_min.y),
        glam::Vec2::new(rect_max.x, rect_min.y),
        glam::Vec2::new(rect_max.x, rect_max.y),
        glam::Vec2::new(rect_min.x, rect_max.y),
    ];

    if quad
        .iter()
        .any(|&p| p.x >= rect_min.x && p.x <= rect_max.x && p.y >= rect_min.y && p.y <= rect_max.y)
    {
        return true;
    }
    if rect.iter().any(|&p| point_in_convex_quad(p, quad)) {
        return true;
    }
    for i in 0..4 {
        let a0 = quad[i];
        let a1 = quad[(i + 1) & 3];
        for j in 0..4 {
            let b0 = rect[j];
            let b1 = rect[(j + 1) & 3];
            if segments_intersect(a0, a1, b0, b1) {
                return true;
            }
        }
    }
    false
}

fn point_in_convex_quad(p: glam::Vec2, quad: &[glam::Vec2; 4]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) & 3];
        let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if cross.abs() < 1e-4 {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if sign * cross < 0.0 {
            return false;
        }
    }
    true
}

/// Port of CMatrixMap::CalcMapGroupVisibility (MatrixVisiCalc.cpp:534-779),
/// scoped to producing a `visible[group_index] -> bool` mask. Rebuilds the
/// four ground-plane frustum points (`frustum_bounds_on_plane_zup` already
/// implements pos[0..3] construction with horizon + disp correction), walks
/// groups inside the quad's bounding box, and runs the original
/// CheckCandidate test: does the 2D group rect touch the projected quad, or
/// does the group's 3D AABB pass the frustum plane test?
fn visible_groups_mask(camera: &Camera, map: &GameMap) -> Vec<bool> {
    let mut out = vec![false; map.group_w * map.group_h];
    let pos = {
        let p3 = camera.frustum_bounds_on_plane_zup(map.min_z);
        [
            glam::Vec2::new(p3[0].x + map.world_width() * 0.5, p3[0].y + map.world_height() * 0.5),
            glam::Vec2::new(p3[1].x + map.world_width() * 0.5, p3[1].y + map.world_height() * 0.5),
            glam::Vec2::new(p3[2].x + map.world_width() * 0.5, p3[2].y + map.world_height() * 0.5),
            glam::Vec2::new(p3[3].x + map.world_width() * 0.5, p3[3].y + map.world_height() * 0.5),
        ]
    };
    // frustum_bounds_on_plane_zup returns CENTERED render-space coords (same
    // as WaterInstance.x0). The original CalcMapGroupVisibility works in
    // uncentered world space. Shifting by half-map once here keeps the
    // rest of the math in uncentered coords, matching C++ directly.

    let group_world = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE;
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (pos[0].x, pos[0].x, pos[0].y, pos[0].y);
    for p in &pos[1..] {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let iminx = ((min_x / group_world).floor() as i32 - 1).max(0);
    let imaxx = ((max_x / group_world) as i32 + 1).min(map.group_w as i32 - 1);
    let iminy = ((min_y / group_world).floor() as i32 - 1).max(0);
    let imaxy = ((max_y / group_world) as i32 + 1).min(map.group_h as i32 - 1);

    // Prebuild frustum planes once — CheckCandidate falls back to 3D AABB
    // test when the 2D quad check fails. Plane normals pack as Ax+By+Cz+D.
    let planes = camera.frustum_planes();
    let cx = map.world_width() * 0.5;
    let cy = map.world_height() * 0.5;

    for gy in iminy..=imaxy {
        for gx in iminx..=imaxx {
            let gi = gy as usize * map.group_w + gx as usize;
            let bounds = map.group_bounds[gi];
            let p0 = glam::Vec2::new(gx as f32 * group_world, gy as f32 * group_world);
            let p1 = p0 + glam::Vec2::splat(group_world);
            if check_candidate(&pos, p0, p1, bounds, &planes, cx, cy) {
                out[gi] = true;
            }
        }
    }
    out
}

/// Port of CMatrixMap::CheckCandidate (MatrixVisiCalc.cpp:470-531).
///
/// Quick-out if any projected frustum point lands inside the group rect; then
/// a 4-edge PointLineCatch walk checking whether any group corner is on the
/// right side of every quad edge (i.e. inside the convex projected polygon);
/// finally a 3D AABB-vs-frustum-planes test. `pos` is in uncentered world
/// space so the group rect and frustum planes both share that coordinate
/// frame — we translate AABB corners by (-cx, -cy) before the plane test
/// since camera.frustum_planes() is derived from the centered view_proj.
fn check_candidate(
    pos: &[glam::Vec2; 4],
    p0: glam::Vec2,
    p1: glam::Vec2,
    bounds: crate::game::map::GroupBounds,
    planes: &[[f32; 4]; 4],
    cx: f32,
    cy: f32,
) -> bool {
    // Step 1: frustum projection point inside group rect?
    for pt in pos {
        if pt.x >= p0.x && pt.x <= p1.x && pt.y >= p0.y && pt.y <= p1.y {
            return true;
        }
    }

    // Step 2: separating-axis test against the projected quad's 4 edges.
    // For every edge walk, if every one of the rect's 4 corners lies on the
    // OUTSIDE side of that edge, the rect is separated from the quad and we
    // can early-out as invisible. Mirrors the goto-driven loop in
    // MatrixVisiCalc.cpp:481-502 (edges walked as (3,0), (0,1), (1,2), (2,3)
    // via `i0 = NPOS-1; i1 = 0; ... i0 = i1++`). Only if NO edge separates
    // does the routine fall through to IsInFrustum.
    //
    // The original comment on `PointLineCatch` (Math3D.hpp:137) claims it
    // returns "true if on the right side", but that is right-side in D3D's
    // left-handed XY plane. Our Z-up right-handed math has `cross > 0` = LEFT
    // side. Since pos[] is walked LT→RT→RB→LB which is *clockwise* in our
    // coord frame, the interior of the quad sits on the side where `cross <
    // 0`. Flip the `PointLineCatch` result so "inside" corresponds to the
    // polygon's interior for both coord conventions.
    let corners = [
        p0,
        p1,
        glam::Vec2::new(p0.x, p1.y),
        glam::Vec2::new(p1.x, p0.y),
    ];
    for edge in 0..4 {
        let a = pos[(edge + 3) & 3];
        let b = pos[edge];
        let any_inside = corners
            .iter()
            .any(|&c| !point_line_catch(a, b, c));
        if !any_inside {
            return false;
        }
    }

    // Step 3: full 3D AABB vs. frustum plane set. camera.frustum_planes is in
    // centered render space, so translate the box there.
    let min = glam::Vec3::new(p0.x - cx, p0.y - cy, bounds.min_z);
    let max = glam::Vec3::new(p1.x - cx, p1.y - cy, bounds.max_z);
    for plane in planes {
        let px = if plane[0] >= 0.0 { max.x } else { min.x };
        let py = if plane[1] >= 0.0 { max.y } else { min.y };
        let pz = if plane[2] >= 0.0 { max.z } else { min.z };
        if plane[0] * px + plane[1] * py + plane[2] * pz + plane[3] < 0.0 {
            return false;
        }
    }
    true
}

/// Math3D.hpp PointLineCatch — true iff `p` lies on the right side of the
/// directed segment (s -> e), or exactly on the segment line.
fn point_line_catch(s: glam::Vec2, e: glam::Vec2, p: glam::Vec2) -> bool {
    let d1 = e - s;
    let d2 = p - e;
    d1.x * d2.y - d1.y * d2.x > 0.0
}

fn segments_intersect(a0: glam::Vec2, a1: glam::Vec2, b0: glam::Vec2, b1: glam::Vec2) -> bool {
    fn orient(a: glam::Vec2, b: glam::Vec2, c: glam::Vec2) -> f32 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }
    let o1 = orient(a0, a1, b0);
    let o2 = orient(a0, a1, b1);
    let o3 = orient(b0, b1, a0);
    let o4 = orient(b0, b1, a1);
    (o1 >= 0.0 && o2 <= 0.0 || o1 <= 0.0 && o2 >= 0.0)
        && (o3 >= 0.0 && o4 <= 0.0 || o3 <= 0.0 && o4 >= 0.0)
}

impl InshoreSystem {
    /// Per-tick wave advancement. Direct port of CMatrixMapGroup::GraphicTakt
    /// (MatrixMapGroup.cpp:752-789). Groups flagged false in `visible` are
    /// skipped entirely, so their waves pause mid-animation — this mirrors
    /// CMatrixMap::Takt walking `m_VisibleGroups` only.
    fn takt(&mut self, dt_ms: f32, visible: &[bool], group_w: usize) {
        // Original caps per-step advance at 0.5 to keep long frames from
        // skipping the entire fade window in one shot (MatrixMapGroup.cpp:762).
        let time = (INSHORE_SPEED * dt_ms).min(0.5);

        for (gi, waves) in self.active_waves.iter_mut().enumerate() {
            let prespawns = &mut self.prespawns[gi];
            if visible.get(gi).copied() != Some(true) {
                continue;
            }
            let _ = group_w; // kept for symmetry with caller; prespawn layout already matches
            let mut i = 0;
            while i < waves.len() {
                waves[i].t += time * waves[i].speed;
                // Scale grows from 13 to 20 over the first 70% of the life.
                waves[i].scale = 13.0 + kscale(waves[i].t, 0.0, 0.7) * 7.0;
                if waves[i].t > 1.0 {
                    let slot = waves[i].slot;
                    if slot < prespawns.len() {
                        prespawns[slot].used = false;
                    }
                    waves.swap_remove(i);
                    continue;
                }
                i += 1;
            }

            if waves.len() >= INSHOREWAVES_CNT || prespawns.is_empty() {
                continue;
            }
            // Random prespawn slot; if already in use, skip this tick (same
            // one-shot behaviour as the original — eventually a free slot
            // is hit).
            let idx = (irnd(&mut self.rng, prespawns.len() as u32)) as usize;
            if !prespawns[idx].used {
                let ps = prespawns[idx];
                waves.push(Wave::spawn(idx, ps.pos, ps.dir, &mut self.rng));
                prespawns[idx].used = true;
            }
        }
    }

    fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        view_proj: glam::Mat4,
        fog_color: [f32; 4],
    ) {
        // Pack every live wave into the instance buffer each frame. Counts
        // peak around 160 groups × 7 waves ≈ 1100 instances — well within a
        // single draw call's budget.
        let mut instances: Vec<InshoreInstance> = Vec::new();
        let cr = self.color_rgb;
        for waves in &self.active_waves {
            for w in waves {
                // Alpha fade: rise from 0→1 over t∈[0, 0.6], fall from 1→0
                // over t∈[0.6, 1.0] (MatrixWater.cpp:156).
                let a = (kscale(w.t, 0.0, 0.6) - kscale(w.t, 0.6, 1.0)).clamp(0.0, 1.0);
                if a <= 0.0 {
                    continue;
                }
                let xx = w.pos[0] + w.dir[0] * w.t * w.len - self.map_half_w;
                let yy = w.pos[1] + w.dir[1] * w.t * w.len - self.map_half_h;
                instances.push(InshoreInstance {
                    x_axis: [w.dir[0] * w.scale, w.dir[1] * w.scale],
                    y_axis: [-w.dir[1] * w.scale, w.dir[0] * w.scale],
                    translation: [xx, yy, WATER_LEVEL],
                    _pad0: 0.0,
                    color: [cr[0], cr[1], cr[2], a],
                });
            }
        }
        self.draw_count = instances.len() as u32;
        if self.draw_count == 0 {
            return;
        }

        self.ensure_instance_capacity(device, instances.len());
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instances),
        );
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&InshoreUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.draw(0..4, 0..self.draw_count);
    }

    fn ensure_instance_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count <= self.instance_capacity {
            return;
        }
        let new_cap = count.next_power_of_two().max(32);
        self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Inshore Instance VB"),
            size: (new_cap * std::mem::size_of::<InshoreInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_capacity = new_cap;
    }
}

impl Wave {
    /// Port of SInshorewave::Create (MatrixWater.cpp:15-48). `len ∈ [10, 15]`,
    /// `speed ∈ [0.7, 1.0]`, direction jittered by up to ±0.2 rad. `t` starts
    /// at 0; `scale` seeded at 10 and re-derived each tick.
    fn spawn(slot: usize, pos: [f32; 2], dir: [f32; 2], rng: &mut u32) -> Self {
        let len = 10.0 + frnd(rng, 5.0);
        let a = fsrnd(rng, 0.2);
        let (sn, cs) = a.sin_cos();
        Self {
            pos,
            dir: [cs * dir[0] - sn * dir[1], sn * dir[0] + cs * dir[1]],
            len,
            t: 0.0,
            speed: frnd(rng, 0.3) + 0.7,
            scale: 10.0,
            slot,
        }
    }
}

/// Port of Math3D.hpp KSCALE — clamped linear map from [k1, k2] into [0, 1].
fn kscale(k: f32, k1: f32, k2: f32) -> f32 {
    if k < k1 {
        0.0
    } else if k > k2 {
        1.0
    } else {
        (k - k1) / (k2 - k1)
    }
}

/// Seed mirroring `srand((unsigned)time(NULL))` (MatrixGame.cpp:90). The
/// original seeds once at app start; we seed once per `InshoreSystem` so the
/// stream is fresh per map load but not reproducible frame-to-frame.
fn inshore_rng_seed() -> u32 {
    crate::platform::now_secs() as u32
}

/// MSVC CRT `rand()` — the engine links against this, so `FRND`/`FSRND`/
/// `IRND` (Math3D.hpp:25-28) ultimately pull from this 15-bit stream.
/// Formula from MS CRT: `seed = seed*214013 + 2531011; return (seed >> 16) & 0x7fff`.
const RAND_MAX: u32 = 0x7fff;

fn msvc_rand(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(214013).wrapping_add(2531011);
    (*state >> 16) & RAND_MAX
}

/// `RND(0, x)` from Math3D.hpp:25 — uniform double in `[0, x]`. All the
/// jitter helpers below fan out from this so the distribution matches the
/// original: 15-bit granularity, inclusive upper bound.
fn rnd(state: &mut u32, x: f64) -> f64 {
    (msvc_rand(state) as f64) * (1.0 / RAND_MAX as f64) * x.abs()
}

fn frnd(state: &mut u32, x: f32) -> f32 {
    rnd(state, x as f64) as f32
}

fn fsrnd(state: &mut u32, x: f32) -> f32 {
    frnd(state, 2.0 * x) - x
}

/// `IRND(n) = Double2Int(RND(0, n - 0.55))` (Math3D.hpp:28). `Double2Int`
/// uses x87 `fistp` with the default round-to-nearest-even mode, so port
/// to `round_ties_even` rather than `as u32` (truncation) or `round()`
/// (round-half-away-from-zero) — both would shift the distribution.
fn irnd(state: &mut u32, n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let r = rnd(state, n as f64 - 0.55).round_ties_even();
    (r as i64).clamp(0, (n - 1) as i64) as u32
}

/// Build the per-group prespawn pool + pipeline. The inshore texture path
/// is taken from the already-resolved `WaterPreset` so that `water`,
/// `mirror`, and `inshore` all come from the same preset record — matching
/// the original's use of the rewritten `m_WaterName` (MatrixWater.cpp:44,
/// MatrixMapPrepare.cpp:1208-1219). Returns `None` and logs a warning if
/// texture decode fails — water still renders, just without shoreline foam.
fn build_inshore_system(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    map: &GameMap,
    water_preset: &WaterPreset,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    fog_color: [f32; 4],
) -> Option<InshoreSystem> {
    if map.inshore_prespawns.iter().all(|v| v.is_empty()) {
        log::info!("water: no inshore prespawns — shoreline foam disabled");
        return None;
    }

    let tex_path = water_preset.inshore_path.as_str();

    let tex_bytes = read_texture(tex_path)?;
    let rgba = super::texture::decode_texture_bytes(&tex_bytes)?;
    let tex_view = super::texture::create_texture_from_rgba(device, queue, &rgba);

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Inshore Sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Inshore BGL"),
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

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Inshore UB"),
        contents: bytemuck::bytes_of(&InshoreUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            fog_color,
            fog_params: [FOG_START, FOG_END, 0.0, 0.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Inshore BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&tex_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    // Static quad VB — 4 verts forming a triangle strip over [-1,-1]..[1,1].
    // Matches SInshorewave::PrepareVB (MatrixWater.cpp:50-78) vertex layout.
    let verts = [
        InshoreVertex { position: [-1.0, -1.0, 0.0], uv: [0.0, 0.0] },
        InshoreVertex { position: [1.0, -1.0, 0.0], uv: [1.0, 0.0] },
        InshoreVertex { position: [-1.0, 1.0, 0.0], uv: [0.0, 1.0] },
        InshoreVertex { position: [1.0, 1.0, 0.0], uv: [1.0, 1.0] },
    ];
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Inshore VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Inshore Instance VB"),
        size: (32 * std::mem::size_of::<InshoreInstance>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Inshore Shader"),
        source: wgpu::ShaderSource::Wgsl(INSHORE_SHADER.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Inshore PL"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Inshore Pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<InshoreVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                        wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                    ],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<InshoreInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0,  shader_location: 2, format: wgpu::VertexFormat::Float32x2 }, // x_axis
                        wgpu::VertexAttribute { offset: 8,  shader_location: 3, format: wgpu::VertexFormat::Float32x2 }, // y_axis
                        wgpu::VertexAttribute { offset: 16, shader_location: 4, format: wgpu::VertexFormat::Float32x3 }, // translation
                        wgpu::VertexAttribute { offset: 32, shader_location: 5, format: wgpu::VertexFormat::Float32x4 }, // color
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
                // MatrixWater.cpp:107 — premultiplied alpha blend on foam quads.
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
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            // Original: ZWRITEENABLE=FALSE, ZENABLE on default (LessEqual).
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let prespawns: Vec<Vec<PreSpawn>> = map
        .inshore_prespawns
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|p: &InshorePreSpawn| PreSpawn {
                    pos: p.pos,
                    dir: p.dir,
                    used: false,
                })
                .collect()
        })
        .collect();
    let active_waves = vec![Vec::new(); prespawns.len()];

    let [cr, cg, cb] = unpack_rgb(map.inshorewave_color);

    log::info!(
        "water: inshore foam ready ({} tex, {} prespawn points, color=0x{:06X})",
        tex_path,
        prespawns.iter().map(|v| v.len()).sum::<usize>(),
        map.inshorewave_color & 0xFFFFFF
    );

    Some(InshoreSystem {
        pipeline,
        vertex_buffer,
        instance_buffer,
        instance_capacity: 32,
        uniform_buffer,
        bind_group,
        prespawns,
        active_waves,
        rng: inshore_rng_seed(),
        color_rgb: [cr, cg, cb],
        map_half_w: map.world_width() * 0.5,
        map_half_h: map.world_height() * 0.5,
        draw_count: 0,
    })
}

const INSHORE_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_foam: texture_2d<f32>;
@group(0) @binding(2) var s_foam: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_dist: f32,
};

@vertex fn vs_main(
    @location(0) quad_pos: vec3<f32>,
    @location(1) quad_uv: vec2<f32>,
    @location(2) x_axis: vec2<f32>,
    @location(3) y_axis: vec2<f32>,
    @location(4) translation: vec3<f32>,
    @location(5) color: vec4<f32>,
) -> VOut {
    // Rebuild the per-wave world transform from the packed axis vectors.
    // Matches CVectorObjectAnim/MatrixWater.cpp:148-154 row-major layout:
    //   X row = dir.xy * scale
    //   Y row = (-dir.y, dir.x) * scale
    //   Z row = (0,0,1)
    //   Translation = (xx, yy, WATER_LEVEL)
    let world_xy = quad_pos.x * x_axis + quad_pos.y * y_axis;
    let world = vec3<f32>(world_xy.x + translation.x, world_xy.y + translation.y, translation.z);
    let clip = u.view_proj * vec4<f32>(world, 1.0);
    var out: VOut;
    out.clip_pos = clip;
    out.uv = quad_uv;
    out.color = color;
    out.view_dist = clip.w;
    return out;
}

@fragment fn fs_main(f: VOut) -> @location(0) vec4<f32> {
    // SetColor/Alpha MODULATE TEXTURE × TFACTOR (MatrixWater.cpp:109-110):
    // final = tex.rgb * color.rgb, final.a = tex.a * color.a.
    let tex = textureSample(t_foam, s_foam, f.uv);
    let rgb = tex.rgb * f.color.rgb;
    let alpha = tex.a * f.color.a;
    // Fog applies to the pre-alpha color in the original fixed-function pass.
    let fog_f = clamp((u.fog_params.y - f.view_dist) / (u.fog_params.y - u.fog_params.x), 0.0, 1.0);
    let fogged = mix(u.fog_color.rgb, rgb, fog_f);
    return vec4<f32>(fogged, alpha);
}
"#;

const WATER_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    water_color: vec4<f32>,
    light_color: vec4<f32>,
    light_dir: vec4<f32>,
    params: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_water: texture_2d<f32>;
@group(0) @binding(2) var t_mirror: texture_2d<f32>;
@group(0) @binding(3) var s: sampler;
@group(0) @binding(4) var t_alpha: texture_2d<f32>;
@group(0) @binding(5) var s_alpha: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) water_uv: vec2<f32>,
    @location(1) cam_normal: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) alpha_uv: vec2<f32>,
    @location(4) view_dist: f32,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) water_uv: vec2<f32>,
    @location(3) alpha_uv: vec2<f32>,
) -> VOut {
    var out: VOut;
    let clip = u.view_proj * vec4<f32>(position, 1.0);
    out.clip_pos = clip;
    out.water_uv = water_uv;
    out.alpha_uv = alpha_uv;

    out.cam_normal = (u.normal_mat * vec4<f32>(normal, 0.0)).xyz;
    out.world_normal = normal;
    out.view_dist = clip.w;
    return out;
}

// Solid (ocean) pass: pin clip-space depth to the far plane (NDC z = 1.0) so the
// LessEqual depth test passes only on pixels where no terrain wrote depth (i.e.,
// water-only cells + empty groups). This fills shoreline "gap" pixels with solid
// water while leaving the alpha gradient to blend over real sea-floor terrain.
@vertex fn vs_main_solid(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) water_uv: vec2<f32>,
    @location(3) alpha_uv: vec2<f32>,
) -> VOut {
    var out: VOut;
    let clip = u.view_proj * vec4<f32>(position, 1.0);
    out.clip_pos = vec4<f32>(clip.xy, clip.w, clip.w);
    out.water_uv = water_uv;
    out.alpha_uv = alpha_uv;

    out.cam_normal = (u.normal_mat * vec4<f32>(normal, 0.0)).xyz;
    out.world_normal = normal;
    out.view_dist = clip.w;
    return out;
}

@fragment fn fs_main(f: VOut) -> @location(0) vec4<f32> {
    let n = normalize(f.world_normal);
    let ndotl = max(dot(n, -u.light_dir.xyz), 0.0);
    let diffuse = u.water_color.rgb + u.light_color.rgb * ndotl;

    let water = textureSample(t_water, s, f.water_uv);
    let stage1 = clamp(water.rgb * diffuse * 2.0, vec3<f32>(0.0), vec3<f32>(1.0));

    let mirror_uv = f.cam_normal.xy * 0.5 + 0.5;
    let mirror = textureSample(t_mirror, s, mirror_uv);
    let final_rgb = mirror.rgb * mirror.a + stage1 * (1.0 - mirror.a);

    let fog_f = clamp((u.fog_params.y - f.view_dist) / (u.fog_params.y - u.fog_params.x), 0.0, 1.0);
    let fogged = mix(u.fog_color.rgb, final_rgb, fog_f);

    let alpha = textureSample(t_alpha, s_alpha, f.alpha_uv).r;

    return vec4<f32>(fogged, alpha);
}
"#;
