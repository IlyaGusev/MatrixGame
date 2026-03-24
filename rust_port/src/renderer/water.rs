//! Water rendering — ports CMatrixWater + BuildWater + DrawWater.
//!
//! Shoreline alpha: per-group 64×64 alpha texture (ports BuildWater from
//! MatrixMapGroup.cpp:366-452), packed into an atlas, sampled per-fragment.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::game::common::{CELLFLAG_BRIDGE, CELLFLAG_WATER};
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
    alpha_tile: [f32; 4],
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
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    ocean_instances: Vec<WaterInstance>,
    ocean_vertex_buffer: wgpu::Buffer,
    ocean_index_buffer: wgpu::Buffer,
    ocean_num_indices: u32,

    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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
    alpha_atlas_inv_w: f32,
    alpha_atlas_inv_h: f32,
}

#[derive(Clone, Copy)]
struct WaterInstance {
    x0: f32,
    y0: f32,
    alpha_tile: [f32; 4],
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

        let atlas_cols = (water_groups.len() as f32).sqrt().ceil() as usize;
        let atlas_rows = water_groups.len().div_ceil(atlas_cols);
        let atlas_w = (atlas_cols * WATER_ALPHA_SIZE) as u32;
        let atlas_h = (atlas_rows * WATER_ALPHA_SIZE) as u32;
        let mut alpha_atlas = image::GrayImage::new(atlas_w, atlas_h);

        let up_level: f32 = -1.0;
        let down_level: f32 = -20.1;
        let alpha_step = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE / WATER_ALPHA_SIZE as f32;

        for (group_idx, &(gx, gy)) in water_groups.iter().enumerate() {
            let tile_x = (group_idx % atlas_cols) * WATER_ALPHA_SIZE;
            let tile_y = (group_idx / atlas_cols) * WATER_ALPHA_SIZE;
            let group_x0 = gx as f32 * GLOBAL_SCALE;
            let group_y0 = gy as f32 * GLOBAL_SCALE;

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

                    alpha_atlas.put_pixel(
                        (tile_x + i) as u32,
                        (tile_y + j) as u32,
                        image::Luma([zz]),
                    );
                }
            }
        }

        let mut water_instances = Vec::new();
        for (group_idx, &(gx, gy)) in water_groups.iter().enumerate() {
            let group_x0 = gx as f32 * GLOBAL_SCALE;
            let group_y0 = gy as f32 * GLOBAL_SCALE;
            let tile_x = (group_idx % atlas_cols) * WATER_ALPHA_SIZE;
            let tile_y = (group_idx / atlas_cols) * WATER_ALPHA_SIZE;
            let alpha_tile = [
                tile_x as f32 / atlas_w as f32,
                tile_y as f32 / atlas_h as f32,
                WATER_ALPHA_SIZE as f32 / atlas_w as f32,
                WATER_ALPHA_SIZE as f32 / atlas_h as f32,
            ];
            water_instances.push(WaterInstance {
                x0: group_x0 - cx,
                y0: group_y0 - cy,
                alpha_tile,
            });
        }

        let ocean_instances = build_solid_water_instances(map, groups_buf);
        let phase_offsets = build_phase_offsets();
        let (lattice_z, lattice_normals) =
            build_wave_lattice(0, map.water_normal_len, &phase_offsets);
        let all_verts =
            build_instance_vertices(&water_instances, &lattice_z, &lattice_normals, water_scale);
        let all_idxs = build_instance_indices(water_instances.len());
        let ocean_verts =
            build_instance_vertices(&ocean_instances, &lattice_z, &lattice_normals, water_scale);
        let ocean_idxs = build_instance_indices(ocean_instances.len());

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

        let alpha_tex = {
            let tex = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("Water Alpha Atlas"),
                    size: wgpu::Extent3d {
                        width: atlas_w,
                        height: atlas_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                alpha_atlas.as_raw(),
            );
            tex.create_view(&Default::default())
        };

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
            contents: bytemuck::cast_slice(&ocean_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let ocean_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ocean IB"),
            contents: bytemuck::cast_slice(&ocean_idxs),
            usage: wgpu::BufferUsages::INDEX,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water UB"),
            contents: bytemuck::bytes_of(&WaterUniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                normal_mat: glam::Mat4::IDENTITY.to_cols_array_2d(),
                water_color,
                light_color,
                light_dir,
                params: [1.0, 0.0, 1.0 / atlas_w as f32, 1.0 / atlas_h as f32],
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water BG"),
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
                    resource: wgpu::BindingResource::TextureView(&alpha_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&alpha_sampler),
                },
            ],
        });

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
                entry_point: Some("vs_main"),
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
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
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
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Some(Self {
            water_instances,
            vertex_buffer,
            index_buffer,
            num_indices: all_idxs.len() as u32,
            ocean_instances,
            ocean_vertex_buffer,
            ocean_index_buffer,
            ocean_num_indices: ocean_idxs.len() as u32,
            uniform_buffer,
            bind_group,
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
            alpha_atlas_inv_w: 1.0 / atlas_w as f32,
            alpha_atlas_inv_h: 1.0 / atlas_h as f32,
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

        let ocean_verts = build_instance_vertices(
            &self.ocean_instances,
            &lattice_z,
            &lattice_normals,
            self.water_scale,
        );
        queue.write_buffer(
            &self.ocean_vertex_buffer,
            0,
            bytemuck::cast_slice(&ocean_verts),
        );
    }

    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
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
                params: [1.0, 0.0, self.alpha_atlas_inv_w, self.alpha_atlas_inv_h],
            }),
        );

        pass.set_bind_group(0, &self.bind_group, &[]);

        if self.ocean_num_indices > 0 {
            pass.set_pipeline(&self.solid_pipeline);
            pass.set_vertex_buffer(0, self.ocean_vertex_buffer.slice(..));
            pass.set_index_buffer(self.ocean_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.ocean_num_indices, 0, 0..1);
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}

fn sample_height_for_water(map: &GameMap, wx: f32, wy: f32) -> f32 {
    let sx = wx / GLOBAL_SCALE;
    let sy = wy / GLOBAL_SCALE;
    let ix = sx.floor() as i32;
    let iy = sy.floor() as i32;

    if ix >= 0 && (ix as usize) < map.size_x && iy >= 0 && (iy as usize) < map.size_y {
        let flags = map.point(ix as usize, iy as usize).flags;
        if flags & CELLFLAG_BRIDGE != 0 {
            let kx = sx - ix as f32;
            let ky = sy - iy as f32;
            let ux = ix as usize;
            let uy = iy as usize;
            let z0 = map.point(ux, uy).z;
            let z1 = map.point(ux + 1, uy).z;
            let z2 = map.point(ux, uy + 1).z;
            let z3 = map.point(ux + 1, uy + 1).z;
            return ky * (kx * z3 + (1.0 - kx) * z2) + (1.0 - ky) * (kx * z1 + (1.0 - kx) * z0);
        }
        if flags & CELLFLAG_WATER != 0 {
            return -1000.0;
        }

        let local_x = wx - ix as f32 * GLOBAL_SCALE;
        let local_y = wy - iy as f32 * GLOBAL_SCALE;
        let p0 = map.point(ix as usize, iy as usize).z;
        let p1 = map.point(ix as usize + 1, iy as usize).z;
        let p2 = map.point(ix as usize, iy as usize + 1).z;
        let p3 = map.point(ix as usize + 1, iy as usize + 1).z;

        if local_y < local_x {
            let a1 = (p1 - p0) / GLOBAL_SCALE;
            let b1 = (p3 - p1) / GLOBAL_SCALE;
            let c1 = p0;
            return a1 * local_x + b1 * local_y + c1;
        }
        let a2 = (p3 - p2) / GLOBAL_SCALE;
        let b2 = (p2 - p0) / GLOBAL_SCALE;
        let c2 = p0;
        return a2 * local_x + b2 * local_y + c2;
    }

    -1000.0
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
                    alpha_tile: inst.alpha_tile,
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

fn build_solid_water_instances(
    map: &GameMap,
    groups_buf: &crate::assets::storage::DataBuf,
) -> Vec<WaterInstance> {
    let group_world = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE;
    let gsx =
        ((map.size_x as f32 + MAP_GROUP_SIZE as f32 - 1.0) / MAP_GROUP_SIZE as f32).ceil() as i32;
    let gsy =
        ((map.size_y as f32 + MAP_GROUP_SIZE as f32 - 1.0) / MAP_GROUP_SIZE as f32).ceil() as i32;

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

    let mut instances = Vec::new();
    let border = 5;
    for gy in -border..gsy + border {
        for gx in -border..gsx + border {
            let on_map = gx >= 0 && gx < gsx && gy >= 0 && gy < gsy;
            if !on_map || !has_group.contains(&(gx, gy)) {
                instances.push(WaterInstance {
                    x0: gx as f32 * group_world - map.world_width() * 0.5,
                    y0: gy as f32 * group_world - map.world_height() * 0.5,
                    alpha_tile: [0.0, 0.0, 0.0, 0.0],
                });
            }
        }
    }
    instances
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
    @location(3) alpha_atlas_uv: vec2<f32>,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) water_uv: vec2<f32>,
    @location(3) alpha_uv: vec2<f32>,
    @location(4) alpha_tile: vec4<f32>,
) -> VOut {
    var out: VOut;
    out.clip_pos = u.view_proj * vec4<f32>(position, 1.0);
    out.water_uv = water_uv;

    let half_texel = vec2(u.params.z * 0.5, u.params.w * 0.5);
    let inset_uv = clamp(alpha_uv, vec2(0.0), vec2(1.0)) * (alpha_tile.zw - 2.0 * half_texel) + alpha_tile.xy + half_texel;
    out.alpha_atlas_uv = select(inset_uv, vec2(0.0), alpha_tile.z < 0.001);

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

    let has_alpha_tile = step(0.001, f.alpha_atlas_uv.x + f.alpha_atlas_uv.y + 0.001);
    let tex_alpha = textureSample(t_alpha, s_alpha, f.alpha_atlas_uv).r;
    let alpha = mix(1.0, tex_alpha, has_alpha_tile);

    return vec4<f32>(final_rgb, alpha);
}
"#;
