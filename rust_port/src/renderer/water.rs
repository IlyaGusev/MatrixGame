//! Water rendering — ports CMatrixWater + BuildWater + DrawWater.
//!
//! Shoreline alpha: per-group 64×64 alpha texture (ports BuildWater from
//! MatrixMapGroup.cpp:366-452), packed into an atlas, sampled per-fragment.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::renderer::camera::Camera;
use crate::game::map::{GameMap, GLOBAL_SCALE};

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
}

pub struct Water {
    water_instances: Vec<WaterInstance>,
    water_draws: Vec<WaterDraw>,
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
    water_path: &'static str,
    mirror_path: &'static str,
}

fn resolve_water_preset(water_name: &str) -> WaterPreset {
    let lower = water_name.to_ascii_lowercase();
    if lower.contains("black") {
        WaterPreset {
            water_path: "Matrix/Textures/Water/1BLACK",
            mirror_path: "Matrix/Textures/Water/MIRRORBLACK",
        }
    } else if lower.contains("purple") {
        WaterPreset {
            water_path: "Matrix/Textures/Water/1PURPLE",
            mirror_path: "Matrix/Textures/Water/MIRRORPURPLE",
        }
    } else {
        WaterPreset {
            water_path: "Matrix/Textures/Water/1",
            mirror_path: "Matrix/Textures/Water/MIRROR",
        }
    }
}

impl Water {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        stor: &Storage,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        let groups_buf = stor.get_buf("groups", "Data")?;
        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;

        let wc = map.water_color;
        let water_color = [
            ((wc >> 16) & 0xFF) as f32 / 255.0,
            ((wc >> 8) & 0xFF) as f32 / 255.0,
            (wc & 0xFF) as f32 / 255.0,
            1.0,
        ];
        let lc = map.light_main_color;
        let light_color = [
            ((lc >> 16) & 0xFF) as f32 / 255.0,
            ((lc >> 8) & 0xFF) as f32 / 255.0,
            (lc & 0xFF) as f32 / 255.0,
            1.0,
        ];
        let ld = map.light_main_dir;
        let light_dir = [ld[0], ld[1], ld[2], 0.0];
        let water_scale = GLOBAL_SCALE * MAP_GROUP_SIZE as f32 / WATER_SIZE as f32;
        let water_preset = resolve_water_preset(&map.water_name);

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

        let map_group_w =
            ((map.size_x as f32 + MAP_GROUP_SIZE as f32 - 1.0) / MAP_GROUP_SIZE as f32).ceil()
                as i32;
        let map_group_h =
            ((map.size_y as f32 + MAP_GROUP_SIZE as f32 - 1.0) / MAP_GROUP_SIZE as f32).ceil()
                as i32;
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
            "water: preset='{}' water='{}' mirror='{}'",
            map.water_name,
            water_preset.water_path,
            water_preset.mirror_path
        );
        let water_tex = load_tex(water_preset.water_path, [30, 80, 120, 255]);
        let mirror_tex = load_tex(water_preset.mirror_path, [100, 120, 140, 128]);

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
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water UB"),
            contents: bytemuck::bytes_of(&WaterUniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                normal_mat: glam::Mat4::IDENTITY.to_cols_array_2d(),
                water_color,
                light_color,
                light_dir,
                params: [1.0, 0.0, 0.0, 0.0],
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

        Some(Self {
            water_instances,
            water_draws,
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
        })
    }

    pub fn takt(&mut self, dt_ms: f32, _device: &wgpu::Device, queue: &wgpu::Queue) {
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
            queue.write_buffer(&self.ocean_vertex_buffer, 0, bytemuck::cast_slice(&ocean_verts));
            queue.write_buffer(&self.ocean_index_buffer, 0, bytemuck::cast_slice(&ocean_idxs));
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
            pass.draw_indexed(draw.index_start..(draw.index_start + draw.index_count), 0, 0..1);
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
        let quad = quad3.map(|p| glam::Vec2::new(p.x, p.y));
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
                // Build the test rect in the same centered render-space as the frustum quad.
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

fn quad_intersects_rect(quad: &[glam::Vec2; 4], rect_min: glam::Vec2, rect_max: glam::Vec2) -> bool {
    let rect = [
        glam::Vec2::new(rect_min.x, rect_min.y),
        glam::Vec2::new(rect_max.x, rect_min.y),
        glam::Vec2::new(rect_max.x, rect_max.y),
        glam::Vec2::new(rect_min.x, rect_max.y),
    ];

    if quad.iter().any(|&p| p.x >= rect_min.x && p.x <= rect_max.x && p.y >= rect_min.y && p.y <= rect_max.y) {
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

const WATER_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    water_color: vec4<f32>,
    light_color: vec4<f32>,
    light_dir: vec4<f32>,
    params: vec4<f32>,
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
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) water_uv: vec2<f32>,
    @location(3) alpha_uv: vec2<f32>,
) -> VOut {
    var out: VOut;
    out.clip_pos = u.view_proj * vec4<f32>(position, 1.0);
    out.water_uv = water_uv;
    out.alpha_uv = alpha_uv;

    out.cam_normal = (u.normal_mat * vec4<f32>(normal, 0.0)).xyz;
    out.world_normal = normal;
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

    let alpha = textureSample(t_alpha, s_alpha, f.alpha_uv).r;

    return vec4<f32>(final_rgb, alpha);
}
"#;
