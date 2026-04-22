//! Terrain renderer — ports BuildBottom (MatrixMapGroup.cpp) + Draw (MatrixMap.cpp).
//!
//! File structure mirrors original:
//!   terrain.rs      ← MatrixMapGroup.cpp (BuildBottom) + MatrixMap.cpp (Draw)
//!   ter_surface.rs  ← MatrixTerSurface.cpp (LoadM, Draw)
//!   texture.rs      ← GPU texture utilities
//!   game/map_prepare.rs ← MatrixMapPrepare.cpp (BuildTexUnions)
//!   game/bitmap.rs  ← MatrixLib/Bitmap/CBitmap.cpp (MergeByMask etc)
//!   game/common.rs  ← Common.hpp (constants, binary helpers)

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::matrix_lib::base::storage::Storage;
use crate::matrix_game::effects::point_light::{PointLightRenderer, PointLightSystem};
use crate::matrix_game::common::{
    rd_u16, rd_u32, unpack_rgb, CELLFLAG_DOWN, FOG_END, FOG_START, MAP_GROUP_SIZE, TEX_BOTTOM_SIZE,
};
use crate::matrix_game::map::{GameMap, GLOBAL_SCALE};
use crate::matrix_game::map_prepare::build_tex_union_atlases;
use crate::matrix_game::camera::Camera;
use crate::matrix_game::ter_surface::{
    build_surface_overlays, GlossOverlayBatch, GlossResources, GlossVertex, OverlayBatch,
};
use crate::matrix_lib::three_g::texture::*;
use crate::matrix_game::water::visible_groups_mask;

/// Bottom vertex — ports SMatrixMapVertexBottom.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub macro_uv: [f32; 2],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4], // x = fog_start, y = fog_end
}

/// Uniforms for the gloss pass. Carries view matrix so the vertex shader can
/// transform normals to camera space (matching D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GlossUniforms {
    view_proj: [[f32; 4]; 4],
    normal_mat: [[f32; 4]; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
}

pub struct DrawBatch {
    pub bind_group: wgpu::BindGroup,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub index_format: wgpu::IndexFormat,
    pub cpu_vertices: Option<Vec<Vertex>>,
    pub point_coords: Option<Vec<(usize, usize)>>,
}

pub struct TerrainRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Depth-only pipeline for the `-1` texture sentinel. See
    /// MatrixMapGroup.cpp:593-606.
    depth_only_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    gloss_pipeline: wgpu::RenderPipeline,
    batches: Vec<DrawBatch>,
    depth_only_batches: Vec<DrawBatch>,
    overlay_batches: Vec<OverlayBatch>,
    gloss_batches: Vec<GlossOverlayBatch>,
    sky: super::sky::Sky,
    clear_color: wgpu::Color,
    fog_color: [f32; 4],
    objects: Option<super::object::ObjectsRenderer>,
    buildings: Option<super::object_building::BuildingsRenderer>,
    point_lights: PointLightRenderer,
    water: Option<super::water::Water>,
    uniform_buffer: wgpu::Buffer,
    gloss_uniform_buffer: wgpu::Buffer,
    depth_texture: wgpu::TextureView,
    last_point_light_revision: u64,
}

impl TerrainRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        stor: &Storage,
        matrix_data: &Storage,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Self {
        let tex_union_dim = map.tex_union_dim;
        let atlas_size = TEX_BOTTOM_SIZE * tex_union_dim;
        let ts_inv = 1.0 / atlas_size as f64;

        // BuildTexUnions (MatrixMapPrepare.cpp:108)
        let atlas_views = build_tex_union_atlases(device, queue, stor, tex_union_dim, read_texture);
        log::info!(
            "built {} texture union atlases ({}x{})",
            atlas_views.len(),
            atlas_size,
            atlas_size
        );

        // BuildBottom (MatrixMapGroup.cpp:231) — parse groups/Data
        let groups_buf = stor.get_buf("groups", "Data");
        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;

        type TerrainBatchParts = (Vec<Vertex>, Vec<u32>, Vec<(usize, usize)>);
        let mut batches_by_tex: std::collections::HashMap<u32, TerrainBatchParts> =
            std::collections::HashMap::new();

        if let Some(grp) = groups_buf {
            for gi in 0..grp.arrays_count() {
                let raw = grp.get_bytes(gi);
                if raw.len() < 4 {
                    continue;
                }
                let mut off = 0;

                let gx = rd_u16(raw, &mut off) as i32;
                let gy = rd_u16(raw, &mut off) as i32;
                let _w = MAP_GROUP_SIZE.min(map.size_x as i32 - gx);
                let _h = MAP_GROUP_SIZE.min(map.size_y as i32 - gy);

                let geom_count = rd_u32(raw, &mut off);

                struct BottomGeom {
                    texture: u32,
                    idx_offset: usize,
                    idx_count: usize,
                }
                let mut geoms = Vec::new();
                let mut all_indices: Vec<u16> = Vec::new();

                for _ in 0..geom_count {
                    if off + 8 > raw.len() {
                        break;
                    }
                    let texture = rd_u32(raw, &mut off);
                    let idx_bytes = rd_u32(raw, &mut off) as usize;
                    let idx_start = all_indices.len();
                    let idx_count = idx_bytes / 2;
                    for _ in 0..idx_count {
                        if off + 2 > raw.len() {
                            break;
                        }
                        all_indices.push(rd_u16(raw, &mut off));
                    }
                    geoms.push(BottomGeom {
                        texture,
                        idx_offset: idx_start,
                        idx_count,
                    });
                }

                if off + 4 > raw.len() {
                    continue;
                }
                let vert_bytes = rd_u32(raw, &mut off) as usize;
                let vert_count = vert_bytes / 8;

                let macro_step = 1.0 / map.macro_texture_size as f32;

                let mut vertices = Vec::with_capacity(vert_count);
                let mut point_coords = Vec::with_capacity(vert_count);
                for _ in 0..vert_count {
                    if off + 8 > raw.len() {
                        break;
                    }
                    let vx = rd_u16(raw, &mut off) as i32;
                    let vy = rd_u16(raw, &mut off) as i32;
                    let tx = rd_u16(raw, &mut off);
                    let ty = rd_u16(raw, &mut off);

                    let pt = map.point(vx as usize, vy as usize);
                    let world_x = vx as f32 * GLOBAL_SCALE - cx;
                    let world_y = vy as f32 * GLOBAL_SCALE - cy;

                    let r = pt.r as f32 / 255.0;
                    let g = pt.g as f32 / 255.0;
                    let b = pt.b as f32 / 255.0;

                    let u = ts_inv * (tx as f64 + 0.5);
                    let v = ts_inv * (ty as f64 + 0.5);
                    let mu = macro_step * vx as f32;
                    let mv = macro_step * vy as f32;

                    // Down-cell handling (MatrixMapGroup.cpp:314-335)
                    let mut pos_x = world_x;
                    let mut pos_y = world_y;
                    let mut pos_z = pt.z;

                    let vxi = vx as usize;
                    let vyi = vy as usize;
                    let down = vxi >= 1
                        && vyi >= 1
                        && vxi < map.size_x
                        && vyi < map.size_y
                        && (map.point(vxi, vyi).flags & CELLFLAG_DOWN) != 0
                        && (map.point(vxi - 1, vyi).flags & CELLFLAG_DOWN) != 0
                        && (map.point(vxi - 1, vyi - 1).flags & CELLFLAG_DOWN) != 0
                        && (map.point(vxi, vyi - 1).flags & CELLFLAG_DOWN) != 0;

                    if down {
                        let n = map.normal(vxi, vyi);
                        pos_x -= n.x * 0.5;
                        pos_y -= n.y * 0.5;
                        pos_z -= n.z * 0.5;
                    }

                    vertices.push(Vertex {
                        position: [pos_x, pos_y, pos_z],
                        color: [r, g, b, 1.0],
                        uv: [u as f32, v as f32],
                        macro_uv: [mu, mv],
                    });
                    point_coords.push((vxi, vyi));
                }

                for geom in &geoms {
                    let (verts, idxs, coords) = batches_by_tex.entry(geom.texture).or_default();
                    let base = verts.len() as u32;
                    verts.extend_from_slice(&vertices);
                    coords.extend_from_slice(&point_coords);
                    for &index in all_indices
                        .iter()
                        .skip(geom.idx_offset)
                        .take(geom.idx_count)
                    {
                        idxs.push(base + index as u32);
                    }
                }
            }
        }

        log::info!("parsed {} group batches", batches_by_tex.len());

        let [sr, sg, sb] = unpack_rgb(map.sky_color);
        let fog_color = [sr, sg, sb, 1.0];

        // GPU resources
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Uniforms"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
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

        // Load macrotexture. When the map has no `MacroTexture` property set
        // (training.cmap leaves it blank) we must not paint a gray/semi-opaque
        // placeholder — the terrain shader blends `macro.rgb*macro.a +
        // atlas.rgb*(1-macro.a)`, so any non-zero macro alpha washes the
        // ground toward the placeholder color. Alpha=0 makes the stage a
        // no-op, mirroring the original's behavior of simply not binding a
        // macrotexture in that case.
        let macro_view = if let Some(data) = map
            .macro_texture_path
            .as_deref()
            .and_then(read_texture)
            .or_else(|| read_texture("macrotexture"))
        {
            if let Some(rgba) = decode_texture_bytes(&data) {
                log::info!(
                    "terrain: loaded macrotexture ({}x{})",
                    rgba.width(),
                    rgba.height()
                );
                create_texture_from_rgba_mipped(device, queue, &rgba, 6)
            } else {
                log::warn!("terrain: macrotexture decode failed; disabling macro stage");
                create_solid_texture(device, queue, [0, 0, 0, 0])
            }
        } else {
            log::info!("terrain: no macrotexture for this map; disabling macro stage");
            create_solid_texture(device, queue, [0, 0, 0, 0])
        };

        let macro_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain BGL"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let transparent = create_solid_texture(device, queue, [0, 0, 0, 0]);

        let mut batches = Vec::new();
        let mut depth_only_batches = Vec::new();
        let mut total_tris = 0u32;
        // Some batches bind the atlas and write color; the -1 sentinel batch
        // just needs geometry — no atlas sample. Reuse the first available
        // atlas view for the sentinel bind group so the layout is satisfied.
        let sentinel_atlas = atlas_views.first();
        for (tex_idx, (verts, idxs, point_coords)) in &batches_by_tex {
            if idxs.is_empty() {
                continue;
            }
            let is_sentinel = *tex_idx == u32::MAX;
            let tex_view = if is_sentinel {
                match sentinel_atlas {
                    Some(v) => v,
                    None => {
                        log::warn!("terrain: -1 batch with no atlases available; skipping");
                        continue;
                    }
                }
            } else {
                match atlas_views.get(*tex_idx as usize) {
                    Some(v) => v,
                    None => {
                        log::warn!(
                            "terrain: batch texture index {} out of range (have {} atlases); skipping",
                            tex_idx,
                            atlas_views.len()
                        );
                        continue;
                    }
                }
            };
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(idxs),
                usage: wgpu::BufferUsages::INDEX,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(tex_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&macro_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&macro_sampler),
                    },
                ],
            });
            total_tris += idxs.len() as u32 / 3;
            let batch = DrawBatch {
                bind_group: bg,
                vertex_buffer: vb,
                index_buffer: ib,
                num_indices: idxs.len() as u32,
                index_format: wgpu::IndexFormat::Uint32,
                cpu_vertices: Some(verts.clone()),
                point_coords: Some(point_coords.clone()),
            };
            if is_sentinel {
                depth_only_batches.push(batch);
            } else {
                batches.push(batch);
            }
        }
        log::info!(
            "terrain bottom: {} draw batches, {} triangles",
            batches.len(),
            total_tris
        );

        // Reflection texture for the gloss pass — falls back to a warm highlight
        // if the asset isn't found (matches TEXTURE_PATH_REFLECTION from
        // StringConstants.hpp:124).
        let reflection_view = if let Some(rgba) =
            read_texture("Matrix/Textures/reflection").and_then(|data| decode_texture_bytes(&data))
        {
            create_texture_from_rgba_mipped(device, queue, &rgba, 6)
        } else {
            create_solid_texture(device, queue, [230, 220, 200, 255])
        };
        let reflection_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Gloss Reflection Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let gloss_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gloss Uniforms"),
            contents: bytemuck::bytes_of(&GlossUniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                normal_mat: glam::Mat4::IDENTITY.to_cols_array_2d(),
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let gloss_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gloss BGL"),
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
            ],
        });

        // Surface overlays (MatrixTerSurface.cpp)
        let overlays = build_surface_overlays(
            device,
            queue,
            &uniform_buffer,
            &bgl,
            &surface_sampler,
            &sampler_wrap_v,
            &transparent,
            &macro_sampler,
            stor,
            map,
            read_texture,
            Some(GlossResources {
                bgl: &gloss_bgl,
                uniform_buffer: &gloss_uniform_buffer,
                reflection_view: &reflection_view,
                reflection_sampler: &reflection_sampler,
            }),
        );
        let overlay_batches = overlays.base;
        let gloss_batches = overlays.gloss;

        // Decorative objects (palms / rocks / etc.)
        let objects =
            super::object::ObjectsRenderer::new(device, queue, config, map, stor, read_texture);
        // Starting buildings from the CMAP `buildings/*` table.
        let buildings =
            super::object_building::BuildingsRenderer::new(device, queue, config, map, read_texture);

        // Water (MatrixWater.cpp)
        let water =
            super::water::Water::new(device, queue, config, map, stor, matrix_data, read_texture);
        let point_lights = PointLightRenderer::new(device, config);

        // Pipelines
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bottom Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main_opaque"),
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

        // Depth-only pipeline for groups whose geometry carries a -1 texture
        // sentinel. Ports MatrixMapGroup.cpp:593-606: draws the triangles with
        // color writes disabled so they only populate the depth buffer —
        // occluding water / distant geometry without contributing any color.
        let depth_only_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bottom Depth-Only Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main_opaque"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
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

        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Overlay Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main_alpha"),
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
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: -1,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Gloss pipeline — ports TerSurfGlossMW (MatrixRenderPipeline.cpp:1460).
        // Runs as a second additive pass on top of the base overlay, weighted by
        // atlas alpha so unused texels don't leak into the highlight.
        let gloss_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gloss Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(GLOSS_SHADER.into()),
        });
        let gloss_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gloss Overlay Layout"),
                bind_group_layouts: &[&gloss_bgl],
                immediate_size: 0,
            });
        let gloss_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gloss Overlay Pipeline"),
            layout: Some(&gloss_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &gloss_shader,
                entry_point: Some("vs_main"),
                buffers: &[GlossVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gloss_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let depth_texture = create_depth_texture(device, config);

        let sky = super::sky::Sky::new(
            device,
            queue,
            config,
            map.sky_color,
            map.water_color,
            &map.sky_name,
            map.sky_angle,
            matrix_data,
            read_texture,
        );
        let clear_color = wgpu::Color {
            r: sr as f64,
            g: sg as f64,
            b: sb as f64,
            a: 1.0,
        };

        Self {
            pipeline,
            depth_only_pipeline,
            overlay_pipeline,
            gloss_pipeline,
            batches,
            depth_only_batches,
            overlay_batches,
            gloss_batches,
            sky,
            clear_color,
            fog_color,
            objects,
            point_lights,
            water,
            uniform_buffer,
            gloss_uniform_buffer,
            depth_texture,
            last_point_light_revision: 0,
            buildings,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) {
        self.depth_texture = create_depth_texture(device, config);
    }

    pub fn takt(
        &mut self,
        dt_ms: f32,
        map: &GameMap,
        point_lights: &PointLightSystem,
        camera: &Camera,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let revision = point_lights.revision();
        if revision != self.last_point_light_revision {
            for batch in &mut self.batches {
                let (Some(cpu_vertices), Some(point_coords)) =
                    (batch.cpu_vertices.as_mut(), batch.point_coords.as_ref())
                else {
                    continue;
                };
                for (vertex, &(px, py)) in cpu_vertices.iter_mut().zip(point_coords.iter()) {
                    let point = map.point(px, py);
                    let lum = point_lights.point_lum(px, py, map.size_x);
                    vertex.color[0] = ((point.r as i32 + lum[0]).clamp(0, 255) as f32) / 255.0;
                    vertex.color[1] = ((point.g as i32 + lum[1]).clamp(0, 255) as f32) / 255.0;
                    vertex.color[2] = ((point.b as i32 + lum[2]).clamp(0, 255) as f32) / 255.0;
                }
                queue.write_buffer(&batch.vertex_buffer, 0, bytemuck::cast_slice(cpu_vertices));
            }
            self.last_point_light_revision = revision;
        }

        if let Some(water) = &mut self.water {
            water.takt(dt_ms, device, queue, camera, map);
        }
        if let Some(objects) = &mut self.objects {
            objects.takt(dt_ms, queue, map, point_lights);
        }
        if let Some(buildings) = &mut self.buildings {
            buildings.takt(dt_ms, queue, map, point_lights);
        }
        self.sky.takt(dt_ms);
        self.point_lights.sync(device, map, point_lights);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        _device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        queue: &wgpu::Queue,
        camera: &Camera,
        view_proj: glam::Mat4,
        view_mat: glam::Mat4,
        map: &GameMap,
    ) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color: self.fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
        );

        // Gloss pass needs a camera-space normal matrix matching
        // D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR: inverse-transpose of the
        // rendering view matrix. `view_mat` is the Z-up look-at used by the
        // water renderer for the same purpose.
        if !self.gloss_batches.is_empty() {
            let normal_mat = view_mat.inverse().transpose();
            queue.write_buffer(
                &self.gloss_uniform_buffer,
                0,
                bytemuck::bytes_of(&GlossUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    normal_mat: normal_mat.to_cols_array_2d(),
                    fog_color: self.fog_color,
                    fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                }),
            );
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Terrain Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture,
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

        // Sky gradient (ports DrawSky, MatrixMap.cpp:2020). Drawn before landscape
        // so terrain/water overwrite it where geometry exists.
        self.sky.render(queue, &mut pass, camera);

        // Bottom geometry (opaque)
        pass.set_pipeline(&self.pipeline);
        for batch in &self.batches {
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
            pass.draw_indexed(0..batch.num_indices, 0, 0..1);
        }

        // Depth-only pass for groups whose -1 texture sentinel signals
        // "carve a hole but occlude everything behind" (MatrixMapGroup.cpp:
        // 593-606). Populating the depth buffer here keeps later water /
        // object draws from showing through these triangles.
        if !self.depth_only_batches.is_empty() {
            pass.set_pipeline(&self.depth_only_pipeline);
            for batch in &self.depth_only_batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
                pass.draw_indexed(0..batch.num_indices, 0, 0..1);
            }
        }

        // Per-group visibility mask, matches CMatrixMap::m_VisibleGroups
        // after CalcMapGroupVisibility (MatrixVisiCalc.cpp:534-779). Surfaces
        // the C++ attached to a specific group via `g->AddSurface(this)` only
        // draw when at least one of their owning groups is visible.
        let visible = visible_groups_mask(camera, map);
        let overlay_visible = |groups: &[u32]| -> bool {
            if groups.is_empty() {
                return true;
            }
            groups
                .iter()
                .any(|&gi| visible.get(gi as usize).copied().unwrap_or(false))
        };

        // Surface overlays (alpha blended, triangle strips)
        if !self.overlay_batches.is_empty() {
            pass.set_pipeline(&self.overlay_pipeline);
            for batch in &self.overlay_batches {
                if !overlay_visible(&batch.groups) {
                    continue;
                }
                let draw = &batch.draw;
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), draw.index_format);
                pass.draw_indexed(0..draw.num_indices, 0, 0..1);
            }
        }

        // Gloss pass: adds gloss*reflection weighted by atlas alpha on top of
        // the already-composited overlay (ports the stage 5 ADD(TEMP, CURRENT)
        // step of TerSurfGlossMW).
        if !self.gloss_batches.is_empty() {
            pass.set_pipeline(&self.gloss_pipeline);
            for batch in &self.gloss_batches {
                if !overlay_visible(&batch.groups) {
                    continue;
                }
                let draw = &batch.draw;
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..draw.num_indices, 0, 0..1);
            }
        }

        // Decorative objects plus their projected shadow pass.
        if let Some(objects) = &self.objects {
            objects.render(queue, &mut pass, camera, view_proj);
        }
        // Starting buildings — drawn after objects so their shadow projection
        // (when we add it) can overlay object silhouettes the way the original
        // does (MatrixMap.cpp DrawLandscape ordering).
        if let Some(buildings) = &self.buildings {
            buildings.render(queue, &mut pass, camera, view_proj);
        }

        // Visible additive point-light pass on terrain-conforming geometry.
        self.point_lights.render(queue, &mut pass, view_proj);

        // Water
        if let Some(water) = &mut self.water {
            water.render(_device, &mut pass, queue, camera, view_proj, view_mat);
        }
    }

    /// Ports `CMinimap::RenderBackground` (MatrixMinimap.cpp:855-1199).
    ///
    /// Renders the landscape orthographically from above into `color_view` /
    /// `depth_view`. Matches the original pass set it calls via
    /// `DrawLandscape(true)` + `DrawLandscapeSurfaces(true)` + water alpha:
    /// opaque bottom → depth-only → surfaces → gloss → per-group water alpha.
    /// Sky, objects, buildings, point lights are intentionally skipped — the
    /// original sets `MMFLAG_DISABLE_DRAW_OBJECT_LIGHTS` for the bake and
    /// stamps buildings via `RenderObjectToBackground` separately.
    pub fn bake_minimap(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        view_proj: glam::Mat4,
        clear_color: wgpu::Color,
    ) {
        // Repoint both uniform buffers at the orthographic VP. `normal_mat`
        // for gloss is left at identity since the ortho top-down view has no
        // meaningful camera-space reflection direction — matches the flat
        // look of RenderBackground's pass.
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color: self.fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
        );
        if !self.gloss_batches.is_empty() {
            queue.write_buffer(
                &self.gloss_uniform_buffer,
                0,
                bytemuck::bytes_of(&GlossUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    normal_mat: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    fog_color: self.fog_color,
                    fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                }),
            );
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Minimap Bake"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
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

        // Bottom (opaque terrain).
        pass.set_pipeline(&self.pipeline);
        for batch in &self.batches {
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
            pass.draw_indexed(0..batch.num_indices, 0, 0..1);
        }

        // Depth-only sentinel batches (MatrixMapGroup.cpp:593-606).
        if !self.depth_only_batches.is_empty() {
            pass.set_pipeline(&self.depth_only_pipeline);
            for batch in &self.depth_only_batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
                pass.draw_indexed(0..batch.num_indices, 0, 0..1);
            }
        }

        // Surface overlays — no visibility culling, the whole map is in view.
        if !self.overlay_batches.is_empty() {
            pass.set_pipeline(&self.overlay_pipeline);
            for batch in &self.overlay_batches {
                let draw = &batch.draw;
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), draw.index_format);
                pass.draw_indexed(0..draw.num_indices, 0, 0..1);
            }
        }

        // Gloss.
        if !self.gloss_batches.is_empty() {
            pass.set_pipeline(&self.gloss_pipeline);
            for batch in &self.gloss_batches {
                let draw = &batch.draw;
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..draw.num_indices, 0, 0..1);
            }
        }

        // Water solid + fill + alpha — fills the non-terrain area inside the
        // fsz×fsz ortho footprint with proper animated water, matching the
        // double loop in RenderBackground (MatrixMinimap.cpp:1055-1112).
        if let Some(water) = &mut self.water {
            water.bake_minimap_all(device, queue, &mut pass, view_proj);
        }
    }
}

/// Terrain shader — ports TerBotM (MatrixRenderPipeline.cpp:1198-1213).
const SHADER: &str = include_str!("../../shaders/terrain.wgsl");
/// Gloss overlay shader — ports TerSurfGlossMW (MatrixRenderPipeline.cpp:1460).
/// Runs as an additive pass on top of the already-composited base overlay: we
/// add `gloss.rgb * reflection.rgb` and pipe `atlas.a` through as the source
/// alpha so the hardware blend does `final += gloss*refl*atlas.a`, matching
/// the stage 5 `ADD(TEMP, CURRENT)` with SrcAlpha/InvSrcAlpha blending in the
/// original single-pass pipeline.
const GLOSS_SHADER: &str = include_str!("../../shaders/terrain_gloss.wgsl");