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

use crate::assets::storage::Storage;
use crate::effects::point_light::{PointLightRenderer, PointLightSystem};
use crate::game::common::{
    rd_u16, rd_u32, unpack_rgb, CELLFLAG_DOWN, FOG_END, FOG_START, MAP_GROUP_SIZE, TEX_BOTTOM_SIZE,
};
use crate::game::map::{GameMap, GLOBAL_SCALE};
use crate::game::map_prepare::build_tex_union_atlases;
use crate::renderer::camera::Camera;
use crate::renderer::ter_surface::{
    build_surface_overlays, GlossBatch, GlossResources, GlossVertex,
};
use crate::renderer::texture::*;

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
    overlay_pipeline: wgpu::RenderPipeline,
    gloss_pipeline: wgpu::RenderPipeline,
    batches: Vec<DrawBatch>,
    overlay_batches: Vec<DrawBatch>,
    gloss_batches: Vec<GlossBatch>,
    sky: super::sky::Sky,
    clear_color: wgpu::Color,
    fog_color: [f32; 4],
    objects: Option<super::objects::ObjectsRenderer>,
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
        matrix_data: Option<&Storage>,
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

        let mut batches_by_tex: std::collections::HashMap<
            u32,
            (Vec<Vertex>, Vec<u32>, Vec<(usize, usize)>),
        > = std::collections::HashMap::new();

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
                    for i in geom.idx_offset..geom.idx_offset + geom.idx_count {
                        idxs.push(base + all_indices[i] as u32);
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

        // Load macrotexture
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
                create_solid_texture(device, queue, [180, 180, 180, 128])
            }
        } else {
            log::warn!("terrain: macrotexture not found");
            create_solid_texture(device, queue, [180, 180, 180, 128])
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

        let white_view = create_solid_texture(device, queue, [200, 200, 200, 255]);
        let transparent = create_solid_texture(device, queue, [0, 0, 0, 0]);

        let mut batches = Vec::new();
        let mut total_tris = 0u32;
        for (tex_idx, (verts, idxs, point_coords)) in &batches_by_tex {
            if idxs.is_empty() {
                continue;
            }
            let tex_view = atlas_views.get(*tex_idx as usize).unwrap_or(&white_view);
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
            batches.push(DrawBatch {
                bind_group: bg,
                vertex_buffer: vb,
                index_buffer: ib,
                num_indices: idxs.len() as u32,
                index_format: wgpu::IndexFormat::Uint32,
                cpu_vertices: Some(verts.clone()),
                point_coords: Some(point_coords.clone()),
            });
        }
        log::info!(
            "terrain bottom: {} draw batches, {} triangles",
            batches.len(),
            total_tris
        );

        // Reflection texture for the gloss pass — falls back to a warm highlight
        // if the asset isn't found (matches TEXTURE_PATH_REFLECTION from
        // StringConstants.hpp:124).
        let reflection_view = if let Some(rgba) = read_texture("Matrix/Textures/reflection")
            .and_then(|data| decode_texture_bytes(&data))
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
            super::objects::ObjectsRenderer::new(device, queue, config, map, stor, read_texture);

        // Water (MatrixWater.cpp)
        let water = super::water::Water::new(
            device,
            queue,
            config,
            map,
            stor,
            matrix_data,
            read_texture,
        );
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
            overlay_pipeline,
            gloss_pipeline,
            batches,
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
        self.sky.takt(dt_ms);
        self.point_lights.sync(device, map, point_lights);
    }

    pub fn render(
        &mut self,
        _device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        queue: &wgpu::Queue,
        camera: &Camera,
        view_proj: glam::Mat4,
        view_mat: glam::Mat4,
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

        // Surface overlays (alpha blended, triangle strips)
        if !self.overlay_batches.is_empty() {
            pass.set_pipeline(&self.overlay_pipeline);
            for batch in &self.overlay_batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
                pass.draw_indexed(0..batch.num_indices, 0, 0..1);
            }
        }

        // Gloss pass: adds gloss*reflection weighted by atlas alpha on top of
        // the already-composited overlay (ports the stage 5 ADD(TEMP, CURRENT)
        // step of TerSurfGlossMW).
        if !self.gloss_batches.is_empty() {
            pass.set_pipeline(&self.gloss_pipeline);
            for batch in &self.gloss_batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..batch.num_indices, 0, 0..1);
            }
        }

        // Decorative objects plus their projected shadow pass.
        if let Some(objects) = &self.objects {
            objects.render(queue, &mut pass, camera, view_proj);
        }

        // Visible additive point-light pass on terrain-conforming geometry.
        self.point_lights.render(queue, &mut pass, view_proj);

        // Water
        if let Some(water) = &mut self.water {
            water.render(_device, &mut pass, queue, camera, view_proj, view_mat);
        }
    }
}

/// Terrain shader — ports TerBotM (MatrixRenderPipeline.cpp:1198-1213).
const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var t_atlas: texture_2d<f32>;
@group(0) @binding(2) var s_atlas: sampler;
@group(0) @binding(3) var t_macro: texture_2d<f32>;
@group(0) @binding(4) var s_macro: sampler;

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
    @location(3) view_dist: f32,
};

@vertex fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    let clip = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.clip_position = clip;
    out.color = in.color;
    out.uv = in.uv;
    out.macro_uv = in.macro_uv;
    out.view_dist = clip.w;
    return out;
}

// Linear fog (D3DFOG_LINEAR): factor=1 keeps original color, factor=0 replaces with fog_color.
// Use clip-space w (== view-space distance along forward axis) as the fog distance,
// matching D3DFOG_TABLE + WFOG semantics.
fn apply_fog(color: vec3<f32>, clip_w: f32) -> vec3<f32> {
    let f = clamp((uniforms.fog_params.y - clip_w) / (uniforms.fog_params.y - uniforms.fog_params.x), 0.0, 1.0);
    return mix(uniforms.fog_color.rgb, color, f);
}

fn shade_terrain(in: VOut) -> vec4<f32> {
    let atlas = textureSample(t_atlas, s_atlas, in.uv);
    let macro_tex = textureSample(t_macro, s_macro, in.macro_uv);
    let blended = macro_tex.rgb * macro_tex.a + atlas.rgb * (1.0 - macro_tex.a);
    return vec4<f32>(blended * in.color.rgb, atlas.a);
}

@fragment fn fs_main_opaque(in: VOut) -> @location(0) vec4<f32> {
    let shaded = shade_terrain(in);
    let fogged = apply_fog(shaded.rgb, in.view_dist);
    return vec4<f32>(fogged, 1.0);
}

@fragment fn fs_main_alpha(in: VOut) -> @location(0) vec4<f32> {
    let shaded = shade_terrain(in);
    let fogged = apply_fog(shaded.rgb, in.view_dist);
    return vec4<f32>(fogged, shaded.a);
}
"#;

/// Gloss overlay shader — ports TerSurfGlossMW (MatrixRenderPipeline.cpp:1460).
/// Runs as an additive pass on top of the already-composited base overlay: we
/// add `gloss.rgb * reflection.rgb` and pipe `atlas.a` through as the source
/// alpha so the hardware blend does `final += gloss*refl*atlas.a`, matching
/// the stage 5 `ADD(TEMP, CURRENT)` with SrcAlpha/InvSrcAlpha blending in the
/// original single-pass pipeline.
const GLOSS_SHADER: &str = r#"
struct GU {
    view_proj: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: GU;
@group(0) @binding(1) var t_atlas: texture_2d<f32>;
@group(0) @binding(2) var s_atlas: sampler;
@group(0) @binding(3) var t_gloss: texture_2d<f32>;
@group(0) @binding(4) var t_refl: texture_2d<f32>;
@group(0) @binding(5) var s_refl: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) cam_normal: vec3<f32>,
    @location(2) view_dist: f32,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VOut {
    var out: VOut;
    let clip = u.view_proj * vec4<f32>(position, 1.0);
    out.clip_pos = clip;
    out.uv = uv;
    out.cam_normal = (u.normal_mat * vec4<f32>(normal, 0.0)).xyz;
    out.view_dist = clip.w;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let atlas = textureSample(t_atlas, s_atlas, in.uv);
    // Gloss texture uses the same UVs as the atlas (same clamp/wrap rules),
    // so we reuse the atlas sampler rather than carving out a second binding.
    let gloss = textureSample(t_gloss, s_atlas, in.uv);
    // Reflection UV from camera-space normal (sphere map, matches water mirror
    // sampling in water.rs).
    let refl_uv = normalize(in.cam_normal).xy * 0.5 + 0.5;
    let refl = textureSample(t_refl, s_refl, refl_uv);
    let spec = gloss.rgb * refl.rgb;
    // Fog attenuates specular the same way it attenuates the diffuse base.
    let f = clamp((u.fog_params.y - in.view_dist) / (u.fog_params.y - u.fog_params.x), 0.0, 1.0);
    let spec_fogged = spec * f;
    return vec4<f32>(spec_fogged, atlas.a);
}
"#;
