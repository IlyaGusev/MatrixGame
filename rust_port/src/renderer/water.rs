//! Water rendering — ports CMatrixWater + BuildWater + DrawWater.
//!
//! Shoreline alpha: per-group 64×64 alpha texture (ports BuildWater from
//! MatrixMapGroup.cpp:366-452), packed into an atlas, sampled per-fragment.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::game::map::{GameMap, GLOBAL_SCALE};

const WATER_LEVEL: f32 = -2.0;
const WATER_SIZE: usize = 16;
const WATER_ALPHA_SIZE: usize = 64;
const MAP_GROUP_SIZE: i32 = 10;
const WATER_TEXTURE_SCALE: f32 = 1.0 / 16.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WaterVertex {
    position: [f32; 3],     // Z-up world coords
    water_uv: [f32; 2],    // tu/tv for water texture
    alpha_uv: [f32; 2],    // UV into alpha atlas (0..1 within this group's tile)
    alpha_tile: [f32; 4],   // xy=atlas offset, zw=atlas scale for this group
}

impl WaterVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 20, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 28, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    // Ocean tiles (solid water outside map)
    ocean_vertex_buffer: wgpu::Buffer,
    ocean_index_buffer: wgpu::Buffer,
    ocean_num_indices: u32,

    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,

    water_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    water_scale: f32,
    time: f32,
    alpha_atlas_inv_w: f32,
    alpha_atlas_inv_h: f32,
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
        let water_color = [((wc >> 16) & 0xFF) as f32 / 255.0, ((wc >> 8) & 0xFF) as f32 / 255.0, (wc & 0xFF) as f32 / 255.0, 1.0];
        let lc = map.light_main_color;
        let light_color = [((lc >> 16) & 0xFF) as f32 / 255.0, ((lc >> 8) & 0xFF) as f32 / 255.0, (lc & 0xFF) as f32 / 255.0, 1.0];
        let ld = map.light_main_dir;
        let light_dir = [ld[0], ld[1], ld[2], 0.0];
        let water_scale = GLOBAL_SCALE * MAP_GROUP_SIZE as f32 / WATER_SIZE as f32;

        // ── Collect water groups ──
        let mut water_groups: Vec<(i32, i32)> = Vec::new();
        for gi in 0..groups_buf.arrays_count() {
            let raw = groups_buf.get_bytes(gi);
            if raw.len() < 4 { continue; }
            let gx = u16::from_le_bytes([raw[0], raw[1]]) as i32;
            let gy = u16::from_le_bytes([raw[2], raw[3]]) as i32;
            let w = MAP_GROUP_SIZE.min(map.size_x as i32 - gx);
            let h = MAP_GROUP_SIZE.min(map.size_y as i32 - gy);
            let mut has_water = false;
            'check: for py in 0..=h { for px in 0..=w {
                if map.point((gx + px) as usize, (gy + py) as usize).z < 0.0 { has_water = true; break 'check; }
            }}
            if has_water { water_groups.push((gx, gy)); }
        }
        if water_groups.is_empty() { return None; }

        // ── Build per-group 64×64 alpha textures into an atlas ──
        // Ports BuildWater (MatrixMapGroup.cpp:366-452)
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

            // Original BuildWater lines 400-449
            for j in 0..WATER_ALPHA_SIZE {
                for i in 0..WATER_ALPHA_SIZE {
                    // Original: wx = (i+0.5) * (MAP_GROUP_SIZE*GLOBAL_SCALE/WATER_ALPHA_SIZE) + p0.x
                    let wx = (i as f32 + 0.5) * alpha_step + group_x0;
                    let wy = (j as f32 + 0.5) * alpha_step + group_y0;
                    let wz = sample_height(map, wx, wy);

                    let zz: u8 = if wz < down_level { 255 }
                        else if wz > up_level { 0 }
                        else { (255.0 - ((wz - down_level) / (up_level - down_level) * 255.0)) as u8 };

                    alpha_atlas.put_pixel(
                        (tile_x + i) as u32,
                        (tile_y + j) as u32,
                        image::Luma([zz]),
                    );
                }
            }
        }

        log::info!("water: {} groups, alpha atlas {}x{}", water_groups.len(), atlas_w, atlas_h);

        // ── Build per-group water meshes with alpha atlas UVs ──
        let mut all_verts: Vec<WaterVertex> = Vec::new();
        let mut all_idxs: Vec<u32> = Vec::new();

        for (group_idx, &(gx, gy)) in water_groups.iter().enumerate() {
            let group_x0 = gx as f32 * GLOBAL_SCALE;
            let group_y0 = gy as f32 * GLOBAL_SCALE;
            let base = all_verts.len() as u32;

            // Alpha atlas tile for this group
            let tile_x = (group_idx % atlas_cols) * WATER_ALPHA_SIZE;
            let tile_y = (group_idx / atlas_cols) * WATER_ALPHA_SIZE;
            let alpha_tile = [
                tile_x as f32 / atlas_w as f32,
                tile_y as f32 / atlas_h as f32,
                WATER_ALPHA_SIZE as f32 / atlas_w as f32,
                WATER_ALPHA_SIZE as f32 / atlas_h as f32,
            ];

            let mut tv = 0.0f32;
            for j in 0..=WATER_SIZE {
                let mut tu = 0.0f32;
                for i in 0..=WATER_SIZE {
                    let wx = i as f32 * water_scale + group_x0;
                    let wy = j as f32 * water_scale + group_y0;
                    all_verts.push(WaterVertex {
                        position: [wx - cx, wy - cy, WATER_LEVEL],
                        water_uv: [tu, tv],
                        alpha_uv: [i as f32 / WATER_SIZE as f32, j as f32 / WATER_SIZE as f32],
                        alpha_tile,
                    });
                    tu += WATER_TEXTURE_SCALE;
                }
                tv += WATER_TEXTURE_SCALE;
            }

            let stride = (WATER_SIZE + 1) as u32;
            for j in 0..WATER_SIZE as u32 { for i in 0..WATER_SIZE as u32 {
                let tl = base + j * stride + i;
                all_idxs.push(tl); all_idxs.push(tl + stride); all_idxs.push(tl + 1);
                all_idxs.push(tl + 1); all_idxs.push(tl + stride); all_idxs.push(tl + stride + 1);
            }}
        }

        // ── Ocean tiles (solid water outside map, border=5) ──
        let (ocean_verts, ocean_idxs) = build_ocean_tiles(cx, cy, map, water_scale, groups_buf);

        // ── Load textures ──
        let load_tex = |key: &str, fb: [u8; 4]| -> wgpu::TextureView {
            read_texture(key)
                .and_then(|d| super::texture::decode_texture_bytes(&d))
                .map(|rgba| super::texture::create_texture_from_rgba(device, queue, &rgba))
                .unwrap_or_else(|| super::texture::create_solid_texture(device, queue, fb))
        };
        let water_tex = load_tex("water_tex1", [30, 80, 120, 255]);
        let mirror_tex = load_tex("water_tex2", [100, 120, 140, 128]);

        // Upload alpha atlas as R8 texture
        let alpha_tex = {
            let tex = device.create_texture_with_data(queue, &wgpu::TextureDescriptor {
                label: Some("Water Alpha Atlas"),
                size: wgpu::Extent3d { width: atlas_w, height: atlas_h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }, wgpu::util::TextureDataOrder::LayerMajor, alpha_atlas.as_raw());
            tex.create_view(&Default::default())
        };

        // ── GPU resources ──
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water VB"), contents: bytemuck::cast_slice(&all_verts), usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water IB"), contents: bytemuck::cast_slice(&all_idxs), usage: wgpu::BufferUsages::INDEX,
        });
        let ocean_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ocean VB"), contents: bytemuck::cast_slice(&ocean_verts), usage: wgpu::BufferUsages::VERTEX,
        });
        let ocean_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ocean IB"), contents: bytemuck::cast_slice(&ocean_idxs), usage: wgpu::BufferUsages::INDEX,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water UB"),
            contents: bytemuck::bytes_of(&WaterUniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                normal_mat: glam::Mat4::IDENTITY.to_cols_array_2d(),
                water_color, light_color, light_dir,
                params: [water_scale, 0.0, 1.0 / atlas_w as f32, 1.0 / atlas_h as f32],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat, address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let alpha_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge, address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water BGL"), entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water BG"), layout: &bgl, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&water_tex) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&mirror_tex) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&alpha_tex) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&alpha_sampler) },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"), source: wgpu::ShaderSource::Wgsl(WATER_SHADER.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"), layout: Some(&pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[WaterVertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, front_face: wgpu::FrontFace::Ccw, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                // Push water slightly behind terrain to prevent Z-fighting at shoreline
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 1.0, clamp: 0.0 },
            }),
            multisample: Default::default(), multiview_mask: None, cache: None,
        });

        Some(Self {
            vertex_buffer, index_buffer, num_indices: all_idxs.len() as u32,
            ocean_vertex_buffer, ocean_index_buffer, ocean_num_indices: ocean_idxs.len() as u32,
            uniform_buffer, bind_group, pipeline,
            water_color, light_color, light_dir, water_scale, time: 0.0,
            alpha_atlas_inv_w: 1.0 / atlas_w as f32, alpha_atlas_inv_h: 1.0 / atlas_h as f32,
        })
    }

    pub fn takt(&mut self, dt_ms: f32, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        self.time += dt_ms / 1000.0;
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, queue: &wgpu::Queue, view_proj: glam::Mat4, view: glam::Mat4) {
        let world = glam::Mat4::from_scale(glam::Vec3::splat(self.water_scale));
        let normal_mat = (view * world).inverse().transpose();

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&WaterUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            normal_mat: normal_mat.to_cols_array_2d(),
            water_color: self.water_color,
            light_color: self.light_color,
            light_dir: self.light_dir,
            params: [self.water_scale, self.time, self.alpha_atlas_inv_w, self.alpha_atlas_inv_h],
        }));

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);

        // Solid pass: ocean tiles (opaque)
        if self.ocean_num_indices > 0 {
            pass.set_vertex_buffer(0, self.ocean_vertex_buffer.slice(..));
            pass.set_index_buffer(self.ocean_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.ocean_num_indices, 0, 0..1);
        }

        // Alpha pass: per-group water with shoreline alpha texture
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}

/// Bilinear height sampling (ports g_MatrixMap->GetZ)
fn sample_height(map: &GameMap, wx: f32, wy: f32) -> f32 {
    let sx = (wx / GLOBAL_SCALE).clamp(0.0, map.size_x as f32);
    let sy = (wy / GLOBAL_SCALE).clamp(0.0, map.size_y as f32);
    let ix = (sx as usize).min(map.size_x.saturating_sub(1));
    let iy = (sy as usize).min(map.size_y.saturating_sub(1));
    let kx = sx - ix as f32;
    let ky = sy - iy as f32;
    let z00 = map.point(ix, iy).z;
    let z10 = map.point((ix + 1).min(map.size_x), iy).z;
    let z01 = map.point(ix, (iy + 1).min(map.size_y)).z;
    let z11 = map.point((ix + 1).min(map.size_x), (iy + 1).min(map.size_y)).z;
    z00 * (1.0 - kx) * (1.0 - ky) + z10 * kx * (1.0 - ky) + z01 * (1.0 - kx) * ky + z11 * kx * ky
}

/// Build ocean tiles for solid water outside the map (border=5 groups).
fn build_ocean_tiles(
    cx: f32, cy: f32, map: &GameMap, water_scale: f32,
    groups_buf: &crate::assets::storage::DataBuf,
) -> (Vec<WaterVertex>, Vec<u32>) {
    let group_world = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE;
    let gsx = ((map.size_x as f32 + MAP_GROUP_SIZE as f32 - 1.0) / MAP_GROUP_SIZE as f32).ceil() as i32;
    let gsy = ((map.size_y as f32 + MAP_GROUP_SIZE as f32 - 1.0) / MAP_GROUP_SIZE as f32).ceil() as i32;

    let mut has_group = std::collections::HashSet::new();
    for gi in 0..groups_buf.arrays_count() {
        let raw = groups_buf.get_bytes(gi);
        if raw.len() < 4 { continue; }
        let gx = u16::from_le_bytes([raw[0], raw[1]]) as i32 / MAP_GROUP_SIZE;
        let gy = u16::from_le_bytes([raw[2], raw[3]]) as i32 / MAP_GROUP_SIZE;
        has_group.insert((gx, gy));
    }

    let mut verts = Vec::new();
    let mut idxs = Vec::new();
    let border = 5;

    for gy in -border..gsy + border {
        for gx in -border..gsx + border {
            let on_map = gx >= 0 && gx < gsx && gy >= 0 && gy < gsy;
            if on_map && has_group.contains(&(gx, gy)) { continue; }

            let base = verts.len() as u32;
            let x0 = gx as f32 * group_world - cx;
            let y0 = gy as f32 * group_world - cy;

            for j in 0..=WATER_SIZE {
                for i in 0..=WATER_SIZE {
                    verts.push(WaterVertex {
                        position: [x0 + i as f32 * water_scale, y0 + j as f32 * water_scale, WATER_LEVEL],
                        water_uv: [i as f32 * WATER_TEXTURE_SCALE, j as f32 * WATER_TEXTURE_SCALE],
                        alpha_uv: [0.0, 0.0],
                        alpha_tile: [0.0, 0.0, 0.0, 0.0], // no alpha tile → shader uses 1.0
                    });
                }
            }

            let stride = (WATER_SIZE + 1) as u32;
            for j in 0..WATER_SIZE as u32 { for i in 0..WATER_SIZE as u32 {
                let tl = base + j * stride + i;
                idxs.push(tl); idxs.push(tl + stride); idxs.push(tl + 1);
                idxs.push(tl + 1); idxs.push(tl + stride); idxs.push(tl + stride + 1);
            }}
        }
    }

    log::info!("ocean: {} solid tiles, {} verts", verts.len() / ((WATER_SIZE+1)*(WATER_SIZE+1)), verts.len());
    (verts, idxs)
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

fn wave_height(world_xy: vec2<f32>, t: f32) -> f32 {
    let p = world_xy * 0.05;
    return 0.205 * (
        sin(p.x * 1.7 + p.y * 2.3 + t * 0.8) * 0.5 +
        sin(p.x * 3.1 - p.y * 1.1 + t * 0.5) * 0.3 +
        sin(p.x * 0.5 + p.y * 4.7 + t * 0.3) * 0.2
    );
}

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) water_uv: vec2<f32>,
    @location(2) alpha_uv: vec2<f32>,
    @location(3) alpha_tile: vec4<f32>,
) -> VOut {
    var out: VOut;
    let ws = u.params.x;
    let t = u.params.y;

    let world_xy = position.xy;
    let h = wave_height(world_xy, t);
    var pos = position;
    pos.z += h * ws;

    out.clip_pos = u.view_proj * vec4<f32>(pos, 1.0);
    out.water_uv = water_uv;

    // Alpha atlas UV: local UV * tile_scale + tile_offset
    // Inset by half a texel to prevent bilinear filtering from bleeding across tile edges.
    // params.z = 1.0 / alpha_atlas_width, params.w = 1.0 / alpha_atlas_height
    let half_texel = vec2(u.params.z * 0.5, u.params.w * 0.5);
    let inset_uv = clamp(alpha_uv, vec2(0.0), vec2(1.0)) * (alpha_tile.zw - 2.0 * half_texel) + alpha_tile.xy + half_texel;
    out.alpha_atlas_uv = select(inset_uv, vec2(0.0), alpha_tile.z < 0.001);

    // Normal from wave derivatives (Z-up)
    let eps = 0.5;
    let hL = wave_height(world_xy - vec2(eps, 0.0), t);
    let hR = wave_height(world_xy + vec2(eps, 0.0), t);
    let hU = wave_height(world_xy - vec2(0.0, eps), t);
    let hD = wave_height(world_xy + vec2(0.0, eps), t);
    let wave_normal = normalize(vec3<f32>(hL - hR, hU - hD, 1.0));
    out.cam_normal = (u.normal_mat * vec4<f32>(wave_normal, 0.0)).xyz;
    out.world_normal = wave_normal;
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

    // Shoreline alpha from per-group 64×64 texture (ports BuildWater + Stage 0)
    // Ocean tiles have alpha_tile.zw = 0 → alpha_atlas_uv = (0,0), and we use alpha=1
    let has_alpha_tile = step(0.001, f.alpha_atlas_uv.x + f.alpha_atlas_uv.y + 0.001);
    let tex_alpha = textureSample(t_alpha, s_alpha, f.alpha_atlas_uv).r;
    // For ocean tiles (no alpha tile), use full opacity. For shoreline, use texture alpha.
    let alpha = mix(1.0, tex_alpha, has_alpha_tile);

    return vec4<f32>(final_rgb, alpha);
}
"#;
