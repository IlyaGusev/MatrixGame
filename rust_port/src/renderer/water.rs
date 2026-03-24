//! Water rendering — ports CMatrixWater.
//!
//! Architecture: per-group water meshes baked in world space (like terrain bottom).
//! Animation: shader-based using time uniform + world position for wave phase.
//! This avoids buffer update issues on WebGL and ensures seamless group boundaries
//! because wave phase is derived from world position, not per-group local coords.
//!
//! Rendering: ports WaterAlpha_t3 texture stages.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::assets::storage::Storage;
use crate::game::map::{GameMap, GLOBAL_SCALE};

const WATER_LEVEL: f32 = -2.0;
const WATER_SIZE: usize = 16;
const MAP_GROUP_SIZE: i32 = 10;

const WATER_TEXTURE_SCALE: f32 = 1.0 / 16.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WaterVertex {
    position: [f32; 3],   // world position (y = WATER_LEVEL, animated in shader)
    water_uv: [f32; 2],   // tu/tv for water texture
    depth_alpha: f32,
    _pad: [f32; 2],
}

impl WaterVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 20, shader_location: 2, format: wgpu::VertexFormat::Float32 },
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
    light_color: [f32; 4],   // RGB from LightMainColor
    light_dir: [f32; 4],     // world-space direction (remapped to Y-up)
    params: [f32; 4],        // x=water_scale, y=time, z=0, w=0
}

pub struct Water {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    num_indices: u32,

    water_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    water_scale: f32,
    time: f32,
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
        let water_scale = GLOBAL_SCALE * MAP_GROUP_SIZE as f32 / WATER_SIZE as f32;

        // Light from map properties (ports MatrixFormGame.cpp lines 59-76)
        let lc = map.light_main_color;
        let light_color = [
            ((lc >> 16) & 0xFF) as f32 / 255.0,
            ((lc >> 8) & 0xFF) as f32 / 255.0,
            (lc & 0xFF) as f32 / 255.0,
            1.0,
        ];
        // Original light_dir is in X-right Y-forward Z-up. Remap to our Y-up: (x, z, -y)
        let ld = map.light_main_dir;
        let light_dir = [ld[0], ld[2], -ld[1], 0.0];

        let up_level: f32 = -1.0;
        let down_level: f32 = -20.1;

        let mut all_verts: Vec<WaterVertex> = Vec::new();
        let mut all_idxs: Vec<u32> = Vec::new();

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
            if !has_water { continue; }

            let group_x0 = gx as f32 * GLOBAL_SCALE;
            let group_y0 = gy as f32 * GLOBAL_SCALE;
            let base = all_verts.len() as u32;

            let mut tv = 0.0f32;
            for j in 0..=WATER_SIZE {
                let mut tu = 0.0f32;
                for i in 0..=WATER_SIZE {
                    let wx = i as f32 * water_scale + group_x0;
                    let wy = j as f32 * water_scale + group_y0;

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
                    let wz = z00 * (1.0 - kx) * (1.0 - ky) + z10 * kx * (1.0 - ky)
                           + z01 * (1.0 - kx) * ky + z11 * kx * ky;
                    let alpha = if wz < down_level { 1.0 }
                        else if wz > up_level { 0.0 }
                        else { 1.0 - (wz - down_level) / (up_level - down_level) };

                    all_verts.push(WaterVertex {
                        position: [wx - cx, WATER_LEVEL, wy - cy],
                        water_uv: [tu, tv],
                        depth_alpha: alpha,
                        _pad: [0.0; 2],
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

        if all_verts.is_empty() { return None; }
        log::info!("water: {} verts, {} tris", all_verts.len(), all_idxs.len() / 3);

        let load_tex = |key: &str, fb: [u8; 4]| -> wgpu::TextureView {
            read_texture(key)
                .and_then(|d| super::terrain::decode_texture_bytes(&d))
                .map(|rgba| super::terrain::create_texture_from_rgba(device, queue, &rgba))
                .unwrap_or_else(|| super::terrain::create_solid_texture(device, queue, fb))
        };
        let water_tex = load_tex("water_tex1", [30, 80, 120, 255]);
        let mirror_tex = load_tex("water_tex2", [100, 120, 140, 128]);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water VB"), contents: bytemuck::cast_slice(&all_verts), usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water IB"), contents: bytemuck::cast_slice(&all_idxs), usage: wgpu::BufferUsages::INDEX,
        });
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water UB"),
            contents: bytemuck::bytes_of(&WaterUniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                normal_mat: glam::Mat4::IDENTITY.to_cols_array_2d(),
                water_color, light_color, light_dir,
                params: [water_scale, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat, address_mode_v: wgpu::AddressMode::Repeat,
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
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water BG"), layout: &bgl, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&water_tex) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&mirror_tex) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
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
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::LessEqual, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(), multiview_mask: None, cache: None,
        });

        Some(Self {
            vertex_buffer, index_buffer, uniform_buffer, bind_group, pipeline,
            num_indices: all_idxs.len() as u32, water_color, light_color, light_dir, water_scale, time: 0.0,
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
            params: [self.water_scale, self.time, 0.0, 0.0],
        }));

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}

/// Wave phase uses WORLD position (position.xz) so adjacent groups get seamless waves.
/// Wave amplitude and speed match original: r=512/2500=0.205, period ~61 seconds per cycle,
/// but multiple frequencies create visible movement.
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
struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) water_uv: vec2<f32>,
    @location(1) cam_normal: vec3<f32>,
    @location(2) depth_alpha: f32,
    @location(3) world_normal: vec3<f32>,
};

// Wave function using world position for seamless tiling across groups.
// Ports FillVB wave: h = r * sin(angle + phase), r=0.205, multiple frequencies.
fn wave_height(world_xz: vec2<f32>, t: f32) -> f32 {
    // Original: h = r * sin(angle + phase), r = 512/2500 = 0.205, one cycle per ~61s.
    // Multiple low-amplitude waves to approximate the 16x16 random-phase grid.
    let p = world_xz * 0.05;
    return 0.205 * (
        sin(p.x * 1.7 + p.y * 2.3 + t * 0.8) * 0.5 +
        sin(p.x * 3.1 - p.y * 1.1 + t * 0.5) * 0.3 +
        sin(p.x * 0.5 + p.y * 4.7 + t * 0.3) * 0.2
    );
}

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) water_uv: vec2<f32>,
    @location(2) depth_alpha: f32,
) -> VOut {
    var out: VOut;
    let ws = u.params.x;
    let t = u.params.y;

    // Animate Y using world XZ position for seamless cross-group waves
    let world_xz = position.xz;
    let h = wave_height(world_xz, t);
    var pos = position;
    pos.y += h * ws;

    out.clip_pos = u.view_proj * vec4<f32>(pos, 1.0);
    out.water_uv = water_uv;

    // Normal from wave derivatives
    let eps = 0.5;
    let hL = wave_height(world_xz - vec2(eps, 0.0), t);
    let hR = wave_height(world_xz + vec2(eps, 0.0), t);
    let hU = wave_height(world_xz - vec2(0.0, eps), t);
    let hD = wave_height(world_xz + vec2(0.0, eps), t);
    let wave_normal = normalize(vec3<f32>(hL - hR, 1.0, hU - hD));
    out.cam_normal = (u.normal_mat * vec4<f32>(wave_normal, 0.0)).xyz;

    out.world_normal = wave_normal;
    out.depth_alpha = depth_alpha;
    return out;
}

@fragment fn fs_main(f: VOut) -> @location(0) vec4<f32> {
    // D3D lighting: diffuse = ambient + light_color * max(0, dot(N, -light_dir))
    // ambient = WaterColor (from D3DRS_AMBIENT), default material = (1,1,1)
    let n = normalize(f.world_normal);
    let ndotl = max(dot(n, -u.light_dir.xyz), 0.0);
    let diffuse = u.water_color.rgb + u.light_color.rgb * ndotl;

    // Stage 1: MODULATE2X(water_texture, diffuse)
    // D3DTA_DIFFUSE = diffuse from lighting (ambient + directional)
    let water = textureSample(t_water, s, f.water_uv);
    let stage1 = clamp(water.rgb * diffuse * 2.0, vec3<f32>(0.0), vec3<f32>(1.0));

    // Stage 2: BLENDTEXTUREALPHA(mirror, stage1) with camera-space normal UVs
    // D3D TCI_CAMERASPACENORMAL maps [-1,1] normal range to [0,1] UV range
    let mirror_uv = f.cam_normal.xy * 0.5 + 0.5;
    let mirror = textureSample(t_mirror, s, mirror_uv);
    let final_rgb = mirror.rgb * mirror.a + stage1 * (1.0 - mirror.a);

    return vec4<f32>(final_rgb, f.depth_alpha);
}
"#;
