//! Terrain renderer — ports BuildTexUnions + BuildBottom from the original C++ engine.

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::assets::storage::{DataBuf, Storage};
use crate::game::map::{GameMap, GLOBAL_SCALE};

const TEX_BOTTOM_SIZE: usize = 64;
const MAP_GROUP_SIZE: i32 = 10;

/// Bottom vertex — ports SMatrixMapVertexBottom: position, color, tc[0] (atlas), tc[1] (macro).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
    uv: [f32; 2],       // tc[0]: atlas texture coords
    macro_uv: [f32; 2], // tc[1]: macrotexture coords
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 28, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 36, shader_location: 3, format: wgpu::VertexFormat::Float32x2 },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

struct DrawBatch {
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    index_format: wgpu::IndexFormat,
}

// ── Terrain Renderer ────────────────────────────────────────────────────────

// Water rendering moved to water.rs
//
// Remaining dead struct — cleaning up:
struct _WaterStateRemoved {
    _phases: Vec<i32>,
    _amplitude: f32,
    _angle: i32,
    _base_verts: Vec<Vertex>,
    /// Pre-built indices
    indices: Vec<u32>,
    /// Per-vertex cell index for wave lookup
    cell_map: Vec<usize>,
    /// World scale factor (original: GLOBAL_SCALE * MAP_GROUP_SIZE / WATER_SIZE)
    wave_z_scale: f32,
}

pub struct TerrainRenderer {
    pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    batches: Vec<DrawBatch>,
    overlay_batches: Vec<DrawBatch>,
    water: Option<super::water::Water>,
    uniform_buffer: wgpu::Buffer,
    depth_texture: wgpu::TextureView,
}

impl TerrainRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        stor: &Storage,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Self {
        let tex_union_dim = map.tex_union_dim;
        let atlas_size = TEX_BOTTOM_SIZE * tex_union_dim;
        let ts_inv = 1.0 / atlas_size as f64;

        // ── Build texture union atlases (ports BuildTexUnions) ──────────
        let atlas_views = build_tex_union_atlases(device, queue, stor, tex_union_dim, read_texture);
        log::info!("built {} texture union atlases ({}x{})", atlas_views.len(), atlas_size, atlas_size);

        // ── Parse groups/Data and build bottom geometry (ports BuildBottom) ──
        let groups_buf = stor.get_buf("groups", "Data");
        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;

        // Group geometry by texture union index
        let mut batches_by_tex: HashMap<u32, (Vec<Vertex>, Vec<u32>)> = HashMap::new();

        if let Some(grp) = groups_buf {
            for gi in 0..grp.arrays_count() {
                let raw = grp.get_bytes(gi);
                if raw.len() < 4 { continue; }
                let mut off = 0;

                let gx = rd_u16(raw, &mut off) as i32;
                let gy = rd_u16(raw, &mut off) as i32;

                let w = MAP_GROUP_SIZE.min(map.size_x as i32 - gx);
                let h = MAP_GROUP_SIZE.min(map.size_y as i32 - gy);

                // Group center offset (original: m_Matrix._41, _42)
                let group_cx = ((w >> 1) + gx) as f32 * GLOBAL_SCALE;
                let group_cy = ((h >> 1) + gy) as f32 * GLOBAL_SCALE;

                let geom_count = rd_u32(raw, &mut off);

                // Read geometries (texture + indices)
                struct BottomGeom { texture: u32, idx_offset: usize, idx_count: usize }
                let mut geoms = Vec::new();
                let mut all_indices: Vec<u16> = Vec::new();

                for _ in 0..geom_count {
                    if off + 8 > raw.len() { break; }
                    let texture = rd_u32(raw, &mut off);
                    let idx_bytes = rd_u32(raw, &mut off) as usize;

                    let idx_start = all_indices.len();
                    let idx_count = idx_bytes / 2;
                    for _ in 0..idx_count {
                        if off + 2 > raw.len() { break; }
                        all_indices.push(rd_u16(raw, &mut off));
                    }
                    geoms.push(BottomGeom { texture, idx_offset: idx_start, idx_count });
                }

                // Read vertices (SCompileBottomVert: WORD x, y, tx, ty = 8 bytes each)
                if off + 4 > raw.len() { continue; }
                let vert_bytes = rd_u32(raw, &mut off) as usize;
                let vert_count = vert_bytes / 8; // sizeof(SCompileBottomVert) = 8

                let mut vertices = Vec::with_capacity(vert_count);
                for _ in 0..vert_count {
                    if off + 8 > raw.len() { break; }
                    let vx = rd_u16(raw, &mut off) as i32;
                    let vy = rd_u16(raw, &mut off) as i32;
                    let tx = rd_u16(raw, &mut off);
                    let ty = rd_u16(raw, &mut off);

                    let pt = map.point(vx as usize, vy as usize);

                    // Original: (vx - gx - w/2) * GLOBAL_SCALE + group_cx
                    // Simplifies to: vx * GLOBAL_SCALE
                    let world_x = vx as f32 * GLOBAL_SCALE - cx;
                    let world_y = vy as f32 * GLOBAL_SCALE - cy;

                    let r = pt.r as f32 / 255.0;
                    let g = pt.g as f32 / 255.0;
                    let b = pt.b as f32 / 255.0;

                    // Bottom UVs are stored in atlas texel space in the source data.
                    // Sampling at texel corners is tolerated by the original D3D9 path
                    // but produces tile-edge bleed in wgpu; bias to texel centers.
                    let u = ts_inv * (tx as f64 + 0.5);
                    let v = ts_inv * (ty as f64 + 0.5);

                    // Macrotexture UVs: macrotexturestep * vx, macrotexturestep * vy
                    // macrotexturestep = 1.0 / m_MacrotextureSize
                    let macro_step = 1.0 / map.macro_texture_size as f32;
                    let mu = macro_step * vx as f32;
                    let mv = macro_step * vy as f32;

                    vertices.push(Vertex {
                        position: [world_x, pt.z, world_y],
                        color: [r, g, b, 1.0],
                        uv: [u as f32, v as f32],
                        macro_uv: [mu, mv],
                    });
                }

                // Add each geometry to its texture batch
                for geom in &geoms {
                    let (verts, idxs) = batches_by_tex.entry(geom.texture).or_default();
                    let base = verts.len() as u32;
                    verts.extend_from_slice(&vertices);
                    for i in geom.idx_offset..geom.idx_offset + geom.idx_count {
                        idxs.push(base + all_indices[i] as u32);
                    }
                }
            }
        }

        log::info!("parsed {} group batches", batches_by_tex.len());

        // ── Create GPU resources ────────────────────────────────────────
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Uniforms"),
            contents: bytemuck::bytes_of(&Uniforms { view_proj: glam::Mat4::IDENTITY.to_cols_array_2d() }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let surface_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let sampler_wrap_v = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        // Load macrotexture
        let macro_view = if let Some(data) = map
            .macro_texture_path
            .as_deref()
            .and_then(read_texture)
            .or_else(|| read_texture("macrotexture"))
        {
            if let Some(rgba) = decode_texture_bytes(&data) {
                log::info!("terrain: loaded macrotexture ({}x{})", rgba.width(), rgba.height());
                create_texture_from_rgba_mipped(device, queue, &rgba, 6)
            } else { create_solid_texture(device, queue, [180, 180, 180, 128]) }
        } else {
            log::warn!("terrain: macrotexture not found");
            create_solid_texture(device, queue, [180, 180, 180, 128])
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Original: macrotexture uses WRAP addressing (MatrixRenderPipeline.cpp:1208-1209)
        let macro_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        // Fallback white texture
        let white_view = create_solid_texture(device, queue, [200, 200, 200, 255]);

        let mut batches = Vec::new();
        let mut total_tris = 0u32;

        for (tex_idx, (verts, idxs)) in &batches_by_tex {
            if idxs.is_empty() { continue; }

            let tex_view = atlas_views.get(*tex_idx as usize).unwrap_or(&white_view);

            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(verts), usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(idxs), usage: wgpu::BufferUsages::INDEX,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(tex_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&macro_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&macro_sampler) },
                ],
            });

            total_tris += idxs.len() as u32 / 3;
            batches.push(DrawBatch {
                bind_group: bg,
                vertex_buffer: vb,
                index_buffer: ib,
                num_indices: idxs.len() as u32,
                index_format: wgpu::IndexFormat::Uint32,
            });
        }

        log::info!("terrain bottom: {} draw batches, {} triangles", batches.len(), total_tris);

        // ── Parse surfacesM overlays (ports CTerSurface::LoadM) ─────────
        let overlay_batches = build_surface_overlays(
            device, queue, &uniform_buffer, &bgl, &surface_sampler, &sampler_wrap_v, &macro_view, &macro_sampler,
            stor, map, read_texture,
        );

        // ── Water (has its own pipeline/shader/bindings) ──
        let water = super::water::Water::new(
            device, queue, config, map, stor, read_texture,
        );

        // ── Pipelines ──────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"), source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], immediate_size: 0,
        });

        // Bottom pipeline: opaque, Z-write enabled
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bottom Pipeline"), layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[Vertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main_opaque"),
                targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, front_face: wgpu::FrontFace::Ccw, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(), multiview_mask: None, cache: None,
        });

        // Overlay pipeline: alpha blended, Z-write disabled (original: ZWRITEENABLE=FALSE, ALPHABLENDENABLE=TRUE)
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Overlay Pipeline"), layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[Vertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main_alpha"),
                targets: &[Some(wgpu::ColorTargetState { format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::LessEqual, stencil: Default::default(), bias: wgpu::DepthBiasState { constant: -1, slope_scale: 0.0, clamp: 0.0 } }),
            multisample: Default::default(), multiview_mask: None, cache: None,
        });

        let depth_texture = create_depth_texture(device, config);

        Self { pipeline, overlay_pipeline, batches, overlay_batches, water, uniform_buffer, depth_texture }
    }

    pub fn resize(&mut self, device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) {
        self.depth_texture = create_depth_texture(device, config);
    }

    /// Per-frame water animation update (ports CMatrixWater::Takt).
    pub fn takt(&mut self, dt_ms: f32, device: &wgpu::Device, queue: &wgpu::Queue) {
        if let Some(water) = &mut self.water {
            water.takt(dt_ms, device, queue);
        }
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, queue: &wgpu::Queue, view_proj: glam::Mat4, view_mat: glam::Mat4) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&Uniforms { view_proj: view_proj.to_cols_array_2d() }));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Terrain Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.4, g: 0.6, b: 0.9, a: 1.0 }), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });

        // Draw bottom geometry (opaque, Z-write on)
        pass.set_pipeline(&self.pipeline);
        for batch in &self.batches {
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
            pass.draw_indexed(0..batch.num_indices, 0, 0..1);
        }

        // Draw surface overlays (alpha blended, Z-write off — original: DrawAll)
        if !self.overlay_batches.is_empty() {
            pass.set_pipeline(&self.overlay_pipeline);
            for batch in &self.overlay_batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
                pass.draw_indexed(0..batch.num_indices, 0, 0..1);
            }
        }

        // Draw water (own pipeline — ports DrawWater + WaterAlpha_t3)
        if let Some(water) = &self.water {
            water.render(&mut pass, queue, view_proj, view_mat);
        }
    }
}

// ── Build surface overlays (faithful port of CTerSurface::LoadM) ────────────

fn build_surface_overlays(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    uniform_buffer: &wgpu::Buffer,
    bgl: &wgpu::BindGroupLayout,
    sampler_clamp: &wgpu::Sampler,
    sampler_wrap_v: &wgpu::Sampler,
    macro_view: &wgpu::TextureView,
    macro_sampler: &wgpu::Sampler,
    stor: &Storage,
    map: &GameMap,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Vec<DrawBatch> {
    let srfm = match stor.get_buf("surfacesM", "Data") {
        Some(b) if b.arrays_count() > 0 => b,
        _ => return vec![],
    };
    let strings = match stor.get_buf("strings", "String") { Some(b) => b, _ => return vec![] };

    let cx = map.world_width() * 0.5;
    let cy = map.world_height() * 0.5;

    // Load overlay textures
    let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
    let white = create_solid_texture(device, queue, [255, 255, 255, 255]);

    // Group surfaces by texture, sorted by m_Index (draw order)
    struct SurfData { index: i32, tex_path: String, wrap_y: bool, color: [f32; 4], verts: Vec<Vertex>, indices: Vec<u16> }
    let mut surfaces: Vec<SurfData> = Vec::new();

    for i in 0..srfm.arrays_count() {
        let raw = srfm.get_bytes(i);
        if raw.len() < 32 { continue; }
        let mut off = 0;

        let ids = rd_i32(raw, &mut off);
        let index = rd_i32(raw, &mut off);       // m_Index: draw order
        let color_dw = rd_u32(raw, &mut off);     // m_Color
        let vcnt = rd_u32(raw, &mut off) as usize;
        let idxsz = rd_u32(raw, &mut off) as usize;
        let _grpsc = rd_u32(raw, &mut off) as usize;
        let disp_x = rd_f32(raw, &mut off);       // m_DispX
        let disp_y = rd_f32(raw, &mut off);       // m_DispY

        let tex_path = if ids >= 0 && (ids as usize) < strings.arrays_count() {
            strings.get_as_wstr(ids as usize).split('?').next().unwrap_or("").replace('\\', "/")
        } else { continue };

        // ARGB color
        let r = ((color_dw >> 16) & 0xFF) as f32 / 255.0;
        let g = ((color_dw >> 8) & 0xFF) as f32 / 255.0;
        let b = (color_dw & 0xFF) as f32 / 255.0;
        let a = ((color_dw >> 24) & 0xFF) as f32 / 255.0;

        // Vertex format: 3×f32 pos, u32 color, 2×f32 uv, 2×f32 uv_macro = 32 bytes
        let needed = off + vcnt * 32 + idxsz;
        if needed > raw.len() { continue; }

        let mut verts = Vec::with_capacity(vcnt);
        let mut wrap_y = false;
        for _ in 0..vcnt {
            let px = rd_f32(raw, &mut off);
            let py = rd_f32(raw, &mut off);
            let pz = rd_f32(raw, &mut off);
            let vcol = rd_u32(raw, &mut off);
            let tu = rd_f32(raw, &mut off);
            let tv = rd_f32(raw, &mut off);
            let _tum = rd_f32(raw, &mut off);
            let _tvm = rd_f32(raw, &mut off);

            let vr = ((vcol >> 16) & 0xFF) as f32 / 255.0;
            let vg = ((vcol >> 8) & 0xFF) as f32 / 255.0;
            let vb = (vcol & 0xFF) as f32 / 255.0;
            let va = ((vcol >> 24) & 0xFF) as f32 / 255.0;
            if tv < 0.0 || tv > 1.0 {
                wrap_y = true;
            }

            // Apply disp offset + center the map
            // Original: position is relative to group center, then m_Matrix._41/_42 applied
            // Here disp_x/disp_y IS the group center offset
            verts.push(Vertex {
                position: [px + disp_x - cx, pz + 0.05, py + disp_y - cy],
                color: [vr * r, vg * g, vb * b, va * a],
                uv: [tu, tv],
                macro_uv: [_tum, _tvm],
            });
        }

        // Preserve the original triangle strip exactly. The C++ renderer draws these
        // surfaces with D3DPT_TRIANGLESTRIP, including restart markers.
        let idx_count = idxsz / 2;
        let mut strip = Vec::with_capacity(idx_count);
        for _ in 0..idx_count {
            if off + 2 > raw.len() { break; }
            strip.push(rd_u16(raw, &mut off));
        }

        // Load texture if not cached
        if !tex_cache.contains_key(&tex_path) {
            if let Some(data) = read_texture(&tex_path) {
                if let Some(rgba) = decode_texture_bytes(&data) {
                    tex_cache.insert(tex_path.clone(), create_texture_from_rgba_mipped(device, queue, &rgba, 6));
                }
            }
        }

        surfaces.push(SurfData { index, tex_path, wrap_y, color: [r, g, b, a], verts, indices: strip });
    }

    // Sort by draw index (original: binary insertion sort in BeforeDraw)
    surfaces.sort_by_key(|s| s.index);

    let mut overlay_batches = Vec::new();
    let mut overlay_tris = 0u32;

    // Preserve strict surface draw order like the original CTerSurface::DrawAll path.
    for surf in &surfaces {
        if surf.indices.len() < 3 {
            continue;
        }

        let tex_view = tex_cache.get(surf.tex_path.as_str()).unwrap_or(&white);

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&surf.verts), usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&surf.indices), usage: wgpu::BufferUsages::INDEX,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(tex_view) },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(if surf.wrap_y { sampler_wrap_v } else { sampler_clamp }),
                },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(macro_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(macro_sampler) },
            ],
        });

        let prim_count = surf.indices.len().saturating_sub(2) as u32;
        overlay_tris += prim_count;
        overlay_batches.push(DrawBatch {
            bind_group: bg,
            vertex_buffer: vb,
            index_buffer: ib,
            num_indices: surf.indices.len() as u32,
            index_format: wgpu::IndexFormat::Uint16,
        });
    }

    log::info!("terrain overlays: {} batches, {} triangles, {} textures", overlay_batches.len(), overlay_tris, tex_cache.len());
    overlay_batches
}

/// Decode DDS (DXT1) or standard image formats.
pub fn decode_texture_bytes(data: &[u8]) -> Option<image::RgbaImage> {
    if data.len() > 128 && &data[0..4] == b"DDS " {
        return decode_dds_dxt1(data);
    }
    image::load_from_memory(data).ok().map(|img| img.to_rgba8())
}

fn decode_dds_dxt1(data: &[u8]) -> Option<image::RgbaImage> {
    if data.len() < 128 { return None; }
    let height = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let width = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let fourcc = &data[84..88];
    let pixel_data = &data[128..];

    let is_dxt3 = fourcc == b"DXT3";
    let is_dxt5 = fourcc == b"DXT5";
    let block_bytes = if fourcc == b"DXT1" { 8usize } else if is_dxt3 || is_dxt5 { 16 } else { return None };

    let bw = ((width + 3) / 4) as usize;
    let bh = ((height + 3) / 4) as usize;
    let mut img = image::RgbaImage::new(width, height);

    for by in 0..bh {
        for bx in 0..bw {
            let block_start = (by * bw + bx) * block_bytes;
            if block_start + block_bytes > pixel_data.len() { break; }

            // Color block is at offset 8 for DXT3/5, offset 0 for DXT1
            let color_off = if block_bytes == 16 { 8 } else { 0 };
            let mut colors = decode_dxt_color_block(
                &pixel_data[block_start + color_off..block_start + color_off + 8],
                fourcc == b"DXT1",
            );

            // DXT3: explicit 4-bit alpha per pixel in first 8 bytes
            if is_dxt3 {
                let alpha_block = &pixel_data[block_start..block_start + 8];
                for py in 0..4usize {
                    // Each row: 2 bytes = 4 pixels × 4 bits
                    let row_bits = u16::from_le_bytes([alpha_block[py * 2], alpha_block[py * 2 + 1]]);
                    for px in 0..4usize {
                        let a4 = ((row_bits >> (px * 4)) & 0xF) as u8;
                        colors[py * 4 + px][3] = (a4 << 4) | a4; // expand 4-bit to 8-bit
                    }
                }
            }
            // DXT5: interpolated alpha (first 8 bytes) — simplified, read endpoints
            if is_dxt5 {
                let a0 = pixel_data[block_start] as u16;
                let a1 = pixel_data[block_start + 1] as u16;
                let mut alpha_lut = [0u8; 8];
                alpha_lut[0] = a0 as u8;
                alpha_lut[1] = a1 as u8;
                if a0 > a1 {
                    for i in 2..8u16 { alpha_lut[i as usize] = ((a0 * (8 - i) + a1 * (i - 1)) / 7) as u8; }
                } else {
                    for i in 2..6u16 { alpha_lut[i as usize] = ((a0 * (6 - i) + a1 * (i - 1)) / 5) as u8; }
                    alpha_lut[6] = 0;
                    alpha_lut[7] = 255;
                }
                // 48 bits of 3-bit indices for 16 pixels
                let bits: u64 = pixel_data[block_start + 2] as u64
                    | (pixel_data[block_start + 3] as u64) << 8
                    | (pixel_data[block_start + 4] as u64) << 16
                    | (pixel_data[block_start + 5] as u64) << 24
                    | (pixel_data[block_start + 6] as u64) << 32
                    | (pixel_data[block_start + 7] as u64) << 40;
                for p in 0..16 {
                    let idx = ((bits >> (p * 3)) & 7) as usize;
                    colors[p][3] = alpha_lut[idx];
                }
            }

            for py in 0..4u32 {
                for px in 0..4u32 {
                    let x = bx as u32 * 4 + px;
                    let y = by as u32 * 4 + py;
                    if x < width && y < height {
                        img.put_pixel(x, y, image::Rgba(colors[py as usize * 4 + px as usize]));
                    }
                }
            }
        }
    }
    Some(img)
}

fn decode_dxt_color_block(block: &[u8], allow_dxt1_transparency: bool) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let expand5 = |v: u16| -> u8 { let x = ((v & 0x1F) as u8); (x << 3) | (x >> 2) };
    let expand6 = |v: u16| -> u8 { let x = ((v & 0x3F) as u8); (x << 2) | (x >> 4) };

    let col0 = [expand5(c0 >> 11), expand6(c0 >> 5), expand5(c0), 255];
    let col1 = [expand5(c1 >> 11), expand6(c1 >> 5), expand5(c1), 255];

    let (col2, col3) = if !allow_dxt1_transparency || c0 > c1 {
        ([((2*col0[0] as u16 + col1[0] as u16)/3) as u8, ((2*col0[1] as u16 + col1[1] as u16)/3) as u8, ((2*col0[2] as u16 + col1[2] as u16)/3) as u8, 255],
         [((col0[0] as u16 + 2*col1[0] as u16)/3) as u8, ((col0[1] as u16 + 2*col1[1] as u16)/3) as u8, ((col0[2] as u16 + 2*col1[2] as u16)/3) as u8, 255])
    } else {
        let avg = [
            ((col0[0] as u16 + col1[0] as u16) / 2) as u8,
            ((col0[1] as u16 + col1[1] as u16) / 2) as u8,
            ((col0[2] as u16 + col1[2] as u16) / 2) as u8,
            255,
        ];
        let transparent = [avg[0], avg[1], avg[2], 0];
        (avg, transparent)
    };

    let palette = [col0, col1, col2, col3];
    let mut result = [[0u8; 4]; 16];
    for row in 0..4 {
        let bits = block[4 + row] as u32;
        for col in 0..4 { result[row * 4 + col] = palette[((bits >> (col * 2)) & 3) as usize]; }
    }
    result
}

// ── Build texture union atlases (faithful port of BuildTexUnions) ────────────

fn build_tex_union_atlases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    stor: &Storage,
    tex_union_dim: usize,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Vec<wgpu::TextureView> {
    let tuc = match stor.get_buf("texunions", "Data") { Some(b) => b, None => return vec![] };
    let botc = match stor.get_buf("bottom", "Data") { Some(b) => b, None => return vec![] };
    let strings = match stor.get_buf("strings", "String") { Some(b) => b, None => return vec![] };
    let bmpc = stor.get_buf("bitmaps", "Bitmap"); // inline alpha masks

    let atlas_px = TEX_BOTTOM_SIZE * tex_union_dim;
    let union_size = tex_union_dim * tex_union_dim;

    let mut atlas_views = Vec::new();
    let mut src_cache: HashMap<usize, image::RgbaImage> = HashMap::new();
    let mut bmp_cache: HashMap<usize, image::RgbaImage> = HashMap::new();

    // Helper: load source texture by string table index
    let load_src = |id: usize, cache: &mut HashMap<usize, image::RgbaImage>, strings: &crate::assets::storage::DataBuf, read_texture: &dyn Fn(&str) -> Option<Vec<u8>>| {
        if !cache.contains_key(&id) {
            if id < strings.arrays_count() {
                let path = strings.get_as_wstr(id).split('?').next().unwrap_or("").replace('\\', "/").to_string();
                if let Some(data) = read_texture(&path) {
                    if let Ok(img) = image::load_from_memory(&data) {
                        cache.insert(id, img.to_rgba8());
                    }
                }
            }
        }
    };

    let mut atlas = image::RgbaImage::new(atlas_px as u32, atlas_px as u32);

    for i in 0..tuc.arrays_count() {
        // Original code reuses one CBitmap across atlases and only clears it for the last atlas.
        // Preserve that behavior instead of zero-initializing every atlas from scratch.
        if i == tuc.arrays_count() - 1 {
            for p in atlas.pixels_mut() { *p = image::Rgba([0, 0, 0, 255]); }
        }

        let un_data = tuc.get_bytes(i);
        let un: Vec<i32> = un_data.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // ── Pass 1: fill each slot with base texture + overlay blending ──
        for k in 0..un.len().min(union_size) {
            if un[k] < 0 { continue; }

            let bot_idx = un[k] as usize;
            if bot_idx >= botc.arrays_count() { continue; }

            let bot_raw = botc.get_bytes(bot_idx);
            let bot_elem_size = 4; // ST_INT32
            let bot_count = bot_raw.len() / bot_elem_size;
            if bot_count == 0 { continue; }

            let bot: Vec<i32> = bot_raw.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let xx = (k % tex_union_dim) * TEX_BOTTOM_SIZE;
            let yy = (k / tex_union_dim) * TEX_BOTTOM_SIZE;

            // First entry: base texture string table index
            let base_id = bot[0] as usize;
            load_src(base_id, &mut src_cache, strings, read_texture);

            if let Some(src) = src_cache.get(&base_id) {
                blit_tile(&mut atlas, xx, yy, src);
            }

            // Remaining entries come in pairs: (ids, ibm)
            let mut bi = 1;
            while bi + 1 < bot.len() {
                let ids = bot[bi];
                let ibm = bot[bi + 1];
                bi += 2;

                // Load inline bitmap mask
                let ibm_idx = ibm as usize;
                if !bmp_cache.contains_key(&ibm_idx) {
                    if let Some(bmp_buf) = bmpc {
                        if ibm_idx < bmp_buf.arrays_count() {
                            let png_data = bmp_buf.get_bytes(ibm_idx);
                            if let Ok(img) = image::load_from_memory(png_data) {
                                bmp_cache.insert(ibm_idx, img.to_rgba8());
                            }
                        }
                    }
                }

                if ids >= 0 {
                    // MergeByMask: blend overlay texture using mask alpha
                    let overlay_id = ids as usize;
                    load_src(overlay_id, &mut src_cache, strings, read_texture);

                    if let (Some(overlay), Some(mask)) = (src_cache.get(&overlay_id), bmp_cache.get(&ibm_idx)) {
                        merge_by_mask(&mut atlas, xx, yy, overlay, mask);
                    }
                } else {
                    // MergeWithAlpha: blend bitmap directly using its own alpha
                    if let Some(mask_img) = bmp_cache.get(&ibm_idx) {
                        merge_with_alpha(&mut atlas, xx, yy, mask_img);
                    }
                }
            }
        }

        // ── Pass 2: extend edges of empty slots from neighboring filled slots ──
        let tsz = atlas_px as i32;
        let tbs = TEX_BOTTOM_SIZE as i32;
        let dim = tex_union_dim as i32;

        for k in 0..un.len().min(union_size) {
            if un[k] >= 0 { continue; }

            let xx = (k % tex_union_dim) as i32 * tbs;
            let yy = (k / tex_union_dim) as i32 * tbs;

            let lp = xx > 0 && un[k - 1] >= 0;
            let tp = yy > 0 && un[k - tex_union_dim] >= 0;
            let rp = xx < tsz - tbs && un[k + 1] >= 0;
            let bp = yy < tsz - tbs && un[k + tex_union_dim] >= 0;

            // Extend from left neighbor
            if lp {
                let mut up = 0i32;
                let mut down = tbs;
                for u in 0..(tbs / 2 - 2) {
                    copy_col(&mut atlas, xx + u, yy + up, down - up, xx - 1, yy + up);
                    if tp { up += 1; }
                    if bp { down -= 1; }
                }
            }
            // Extend from top neighbor
            if tp {
                let mut left = 0i32;
                let mut rite = tbs;
                for u in 0..(tbs / 2 - 2) {
                    copy_row(&mut atlas, xx + left, yy + u, rite - left, xx + left, yy - 1);
                    if lp { left += 1; }
                    if rp { rite -= 1; }
                }
            }
            // Extend from right neighbor
            if rp {
                let mut up = 0i32;
                let mut down = tbs;
                for u in 1..=(tbs / 2 - 2) {
                    copy_col(&mut atlas, xx + tbs - u, yy + up, down - up, xx + tbs, yy + up);
                    if tp { up += 1; }
                    if bp { down -= 1; }
                }
            }
            // Extend from bottom neighbor
            if bp {
                let mut left = 0i32;
                let mut rite = tbs;
                for u in 1..=(tbs / 2 - 2) {
                    copy_row(&mut atlas, xx + left, yy + tbs - u, rite - left, xx + left, yy + tbs);
                    if lp { left += 1; }
                    if rp { rite -= 1; }
                }
            }
        }

        // Original BuildTexUnions builds the terrain atlas into an RGB bitmap,
        // so bottom terrain tiles do not carry per-source alpha into sampling/mips.
        for p in atlas.pixels_mut() {
            p.0[3] = 255;
        }

        // Original: LoadFromBitmap(bm, D3DFMT_DXT1, 6) — 6 mip levels
        let view = create_texture_from_rgba_mipped(device, queue, &atlas, 6);
        atlas_views.push(view);
    }

    atlas_views
}

/// Copy a 64x64 tile onto the atlas at (dx, dy).
fn blit_tile(atlas: &mut image::RgbaImage, dx: usize, dy: usize, src: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(src.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(src.height() as usize);
    for py in 0..h {
        for px in 0..w {
            atlas.put_pixel((dx + px) as u32, (dy + py) as u32, *src.get_pixel(px as u32, py as u32));
        }
    }
}

/// MergeByMask: dst = dst * (1 - mask.luma) + overlay * mask.luma
fn merge_by_mask(atlas: &mut image::RgbaImage, dx: usize, dy: usize, overlay: &image::RgbaImage, mask: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(overlay.width() as usize).min(mask.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(overlay.height() as usize).min(mask.height() as usize);
    for py in 0..h {
        for px in 0..w {
            let ax = (dx + px) as u32;
            let ay = (dy + py) as u32;
            let dst = atlas.get_pixel(ax, ay).0;
            let src = overlay.get_pixel(px as u32, py as u32).0;
            let m = mask.get_pixel(px as u32, py as u32).0;
            // Original MergeByMask (CBitmap.cpp:1280-1283):
            //   mask=0 → show overlay (bm2), mask=255 → show background (bm1)
            // So mask value = how much BACKGROUND to keep (inverted from typical alpha)
            let alpha = (255 - m[0]) as u16; // invert: 0=keep dst, 255=show overlay
            let inv = 255 - alpha;
            atlas.put_pixel(ax, ay, image::Rgba([
                ((dst[0] as u16 * inv + src[0] as u16 * alpha) / 255) as u8,
                ((dst[1] as u16 * inv + src[1] as u16 * alpha) / 255) as u8,
                ((dst[2] as u16 * inv + src[2] as u16 * alpha) / 255) as u8,
                255,
            ]));
        }
    }
}

/// MergeWithAlpha: dst = dst * (1 - src.alpha) + src.rgb * src.alpha
fn merge_with_alpha(atlas: &mut image::RgbaImage, dx: usize, dy: usize, src: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(src.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(src.height() as usize);
    for py in 0..h {
        for px in 0..w {
            let ax = (dx + px) as u32;
            let ay = (dy + py) as u32;
            let dst = atlas.get_pixel(ax, ay).0;
            let s = src.get_pixel(px as u32, py as u32).0;
            let alpha = s[3] as u16;
            let inv = 255 - alpha;
            atlas.put_pixel(ax, ay, image::Rgba([
                ((dst[0] as u16 * inv + s[0] as u16 * alpha) / 255) as u8,
                ((dst[1] as u16 * inv + s[1] as u16 * alpha) / 255) as u8,
                ((dst[2] as u16 * inv + s[2] as u16 * alpha) / 255) as u8,
                255,
            ]));
        }
    }
}

/// Copy a single column of pixels (for edge extension).
fn copy_col(atlas: &mut image::RgbaImage, dx: i32, dy: i32, h: i32, sx: i32, sy: i32) {
    let aw = atlas.width() as i32;
    let ah = atlas.height() as i32;
    for i in 0..h {
        let fx = sx.clamp(0, aw - 1);
        let fy = (sy + i).clamp(0, ah - 1);
        let tx = dx.clamp(0, aw - 1);
        let ty = (dy + i).clamp(0, ah - 1);
        let p = *atlas.get_pixel(fx as u32, fy as u32);
        atlas.put_pixel(tx as u32, ty as u32, p);
    }
}

/// Copy a single row of pixels (for edge extension).
fn copy_row(atlas: &mut image::RgbaImage, dx: i32, dy: i32, w: i32, sx: i32, sy: i32) {
    let aw = atlas.width() as i32;
    let ah = atlas.height() as i32;
    for i in 0..w {
        let fx = (sx + i).clamp(0, aw - 1);
        let fy = sy.clamp(0, ah - 1);
        let tx = (dx + i).clamp(0, aw - 1);
        let ty = dy.clamp(0, ah - 1);
        let p = *atlas.get_pixel(fx as u32, fy as u32);
        atlas.put_pixel(tx as u32, ty as u32, p);
    }
}

// ── GPU helpers ─────────────────────────────────────────────────────────────

pub fn create_solid_texture(device: &wgpu::Device, queue: &wgpu::Queue, color: [u8; 4]) -> wgpu::TextureView {
    let pixels: Vec<u8> = color.repeat(4);
    let texture = device.create_texture_with_data(queue, &wgpu::TextureDescriptor {
        label: Some("solid"), size: wgpu::Extent3d { width: 2, height: 2, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
    }, wgpu::util::TextureDataOrder::LayerMajor, &pixels);
    texture.create_view(&Default::default())
}

pub fn create_texture_from_rgba(device: &wgpu::Device, queue: &wgpu::Queue, img: &image::RgbaImage) -> wgpu::TextureView {
    create_texture_from_rgba_mipped(device, queue, img, 1)
}

/// Create texture with mipmap levels (original uses 6 for atlas textures).
/// Ports CTextureManaged::LoadFromBitmap(bm, D3DFMT_DXT1, levels).
/// Uses create_texture_with_data with all mip levels concatenated, which works on WebGL.
pub fn create_texture_from_rgba_mipped(device: &wgpu::Device, queue: &wgpu::Queue, img: &image::RgbaImage, levels: u32) -> wgpu::TextureView {
    let w = img.width();
    let h = img.height();
    let max_levels = (w.max(h) as f32).log2().floor() as u32 + 1;
    let mip_count = if levels <= 1 { 1 } else { levels.min(max_levels) };

    // Generate all mip levels CPU-side and concatenate into one buffer
    let mut all_data: Vec<u8> = Vec::new();
    all_data.extend_from_slice(img.as_raw()); // level 0

    let mut current = img.clone();
    for _level in 1..mip_count {
        let mw = (current.width() / 2).max(1);
        let mh = (current.height() / 2).max(1);
        let mut mip = image::RgbaImage::new(mw, mh);
        for y in 0..mh {
            for x in 0..mw {
                let sx = x * 2;
                let sy = y * 2;
                let p00 = current.get_pixel(sx.min(current.width()-1), sy.min(current.height()-1)).0;
                let p10 = current.get_pixel((sx+1).min(current.width()-1), sy.min(current.height()-1)).0;
                let p01 = current.get_pixel(sx.min(current.width()-1), (sy+1).min(current.height()-1)).0;
                let p11 = current.get_pixel((sx+1).min(current.width()-1), (sy+1).min(current.height()-1)).0;
                let a00 = p00[3] as f32 / 255.0;
                let a10 = p10[3] as f32 / 255.0;
                let a01 = p01[3] as f32 / 255.0;
                let a11 = p11[3] as f32 / 255.0;

                // Generate mips in premultiplied-alpha space to avoid shoreline
                // fringe/checker artifacts from straight-alpha averaging.
                let out_a = (p00[3] as u16 + p10[3] as u16 + p01[3] as u16 + p11[3] as u16) as f32 / (4.0 * 255.0);
                let premul_r = ((p00[0] as f32 * a00) + (p10[0] as f32 * a10) + (p01[0] as f32 * a01) + (p11[0] as f32 * a11)) * 0.25;
                let premul_g = ((p00[1] as f32 * a00) + (p10[1] as f32 * a10) + (p01[1] as f32 * a01) + (p11[1] as f32 * a11)) * 0.25;
                let premul_b = ((p00[2] as f32 * a00) + (p10[2] as f32 * a10) + (p01[2] as f32 * a01) + (p11[2] as f32 * a11)) * 0.25;

                let (out_r, out_g, out_b) = if out_a > 1e-6 {
                    (
                        (premul_r / out_a).clamp(0.0, 255.0) as u8,
                        (premul_g / out_a).clamp(0.0, 255.0) as u8,
                        (premul_b / out_a).clamp(0.0, 255.0) as u8,
                    )
                } else {
                    (0, 0, 0)
                };
                mip.put_pixel(x, y, image::Rgba([
                    out_r,
                    out_g,
                    out_b,
                    (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
                ]));
            }
        }
        all_data.extend_from_slice(mip.as_raw());
        current = mip;
    }

    // Upload all mip levels at once — works on WebGL unlike per-level write_texture
    let texture = device.create_texture_with_data(queue, &wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }, wgpu::util::TextureDataOrder::MipMajor, &all_data);

    texture.create_view(&Default::default())
}

fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth"), size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    texture.create_view(&Default::default())
}

// ── Binary helpers ──────────────────────────────────────────────────────────

fn rd_u32(d: &[u8], o: &mut usize) -> u32 { let v = u32::from_le_bytes([d[*o], d[*o+1], d[*o+2], d[*o+3]]); *o += 4; v }
fn rd_i32(d: &[u8], o: &mut usize) -> i32 { let v = i32::from_le_bytes([d[*o], d[*o+1], d[*o+2], d[*o+3]]); *o += 4; v }
fn rd_f32(d: &[u8], o: &mut usize) -> f32 { let v = f32::from_le_bytes([d[*o], d[*o+1], d[*o+2], d[*o+3]]); *o += 4; v }
fn rd_u16(d: &[u8], o: &mut usize) -> u16 { let v = u16::from_le_bytes([d[*o], d[*o+1]]); *o += 2; v }

// ── Shader ──────────────────────────────────────────────────────────────────

/// Ports TerBotM from MatrixRenderPipeline.cpp:
///   Stage 0: SELECT(atlas_texture) via tc[0]
///   Stage 1: BLENDTEXTUREALPHA(macro_texture, current) via tc[1]
///   Stage 2: MODULATE(diffuse_vertex_color, current)
const SHADER: &str = r#"
struct Uniforms { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var t_atlas: texture_2d<f32>;
@group(0) @binding(2) var s_atlas: sampler;    // ClampToEdge for atlas
@group(0) @binding(3) var t_macro: texture_2d<f32>;
@group(0) @binding(4) var s_macro: sampler;    // Repeat/Wrap for macrotexture

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) macro_uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) macro_uv: vec2<f32>,
};

@vertex fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.uv = in.uv;
    out.macro_uv = in.macro_uv;
    return out;
}

fn shade_terrain(in: VOut) -> vec4<f32> {
    // Stage 0: SELECT(atlas_texture) — ClampToEdge sampler
    let atlas = textureSample(t_atlas, s_atlas, in.uv);

    // Stage 1: BLENDTEXTUREALPHA(macro_texture, current) — Wrap sampler
    // D3DTOP_BLENDTEXTUREALPHA: Result = Arg1 * tex.a + Arg2 * (1 - tex.a)
    let macro_tex = textureSample(t_macro, s_macro, in.macro_uv);
    let blended = macro_tex.rgb * macro_tex.a + atlas.rgb * (1.0 - macro_tex.a);

    // Stage 2: MODULATE(diffuse_vertex_color, current)
    return vec4<f32>(blended * in.color.rgb, atlas.a);
}

@fragment fn fs_main_opaque(in: VOut) -> @location(0) vec4<f32> {
    let shaded = shade_terrain(in);
    return vec4<f32>(shaded.rgb, 1.0);
}

@fragment fn fs_main_alpha(in: VOut) -> @location(0) vec4<f32> {
    return shade_terrain(in);
}
"#;
