//! Runtime projected-shadow system for dynamic objects (robots, buildings).
//!
//! Ports the engine's `SHADOW_PROJ_*` path: each shadow is a 2D silhouette
//! texture baked from the model's mesh, projected onto the terrain mesh as a
//! triangle list under the object.
//!
//! C++ references:
//! * `CVectorObject::CalcShadowProjMatrix` (VectorObject.cpp:1085) — light-space
//!   AABB → projector axes (vpos, vx, vy, vz).
//! * `ShadowProjBuildGeomInt` (MatrixShadowManager.cpp:89) — build the terrain
//!   geometry that receives the shadow.
//! * `CVectorObject::CalcShadowTexture` (VectorObject.cpp:1156) — bake the
//!   silhouette texture.
//! * `CMatrixMap::DrawShadowsProjFast` (MatrixMap.cpp:1818) — render-state
//!   block (alpha blend src*alpha / inv-src*alpha, no z-write, alpha texture
//!   modulated by `m_ShadowColor` factor — we bake this into the shader).
//!
//! Static objects (palms, rocks) have their projected shadow geometry baked
//! into the CMAP file and are still handled by `object.rs::ObjectsRenderer`.
//! This module covers the *dynamic* case where geometry must be rebuilt
//! against the current terrain at runtime.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::matrix_game::common::CELLFLAG_LAND;
use crate::matrix_game::map::{GameMap, GLOBAL_SCALE};

/// `SHADOW_ALTITUDE` constant from `VectorObject.hpp:22`. Applied as a direct
/// `+Z` offset to terrain corners in `ShadowProjBuildGeomInt`
/// (MatrixShadowManager.cpp:155).
const SHADOW_ALTITUDE: f32 = 0.7;

/// `CalcShadowProjMatrix(addsize=6.0)` — VectorObject.cpp:325. Padding around
/// the light-space AABB so the shadow projector covers a slight skirt past the
/// model's silhouette.
const ADD_SIZE: f32 = 6.0;

/// Projection data for one shadow. Stores both the local-space form (used by
/// the texture baker, which receives raw local-space vertices) and the
/// world-space form (used by the ground-projection geometry, which receives
/// world-space terrain points). The two forms differ only by the object's
/// world rotation.
#[derive(Clone, Copy, Debug)]
pub struct ShadowProjData {
    /// Local-space basis: `+X` = light-view right, `+Y` = light-view down,
    /// `+Z` = light direction.
    pub x_local: Vec3,
    pub y_local: Vec3,
    pub z_local: Vec3,
    /// World-space versions of the same basis vectors.
    pub x_world: Vec3,
    pub y_world: Vec3,
    pub z_world: Vec3,
    /// AABB center, in object-local space.
    pub center_local: Vec3,
    /// AABB center, in world space (= world_matrix * center_local).
    pub center_world: Vec3,
    /// AABB extent on the projector U axis (full width, world units).
    pub dim_x: f32,
    /// AABB extent on the projector V axis (full height, world units).
    pub dim_y: f32,
    /// AABB extent along the light direction (used to push the texture
    /// projector eye behind the model).
    pub dim_z: f32,
    /// World translation of the object (`objma._41`, `objma._42`). Used by the
    /// projector geometry pass to center the cell window — matches
    /// `MatrixShadowManager.cpp:93-94` which reads the cell from `objma._41 /
    /// GLOBAL_SCALE` rather than from the AABB center.
    pub world_translation_xy: [f32; 2],
}

/// Per-batch GPU resources for one rendered shadow. Cloning is cheap —
/// every wgpu handle inside is reference-counted, so we clone these
/// per-frame from the per-object cache in [`crate::matrix_game::object_robot::RobotsRenderer`]
/// to skip the geometry rebuild for stationary robots.
#[derive(Clone)]
pub struct ShadowBatch {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub bind_group: wgpu::BindGroup,
}

/// Mesh vertex consumed by the silhouette baker. Layout matches the shared
/// per-renderer `Vertex` (position + normal + uv).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShadowMeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// One sub-mesh participating in the bake.
pub struct ShadowMeshSurface {
    pub indices: Vec<u32>,
    pub diffuse: wgpu::TextureView,
    pub alpha_test: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ProjUniform {
    view_proj: [[f32; 4]; 4],
    /// Map's `ShadowColor` (DATA_SHADOWCOLOR / MatrixMap.cpp:1830) unpacked
    /// from 0xAARRGGBB into linear `(r, g, b, a)` floats. Mirrors D3D's
    /// `D3DRS_TEXTUREFACTOR` semantics for `DrawShadowsProjFast`:
    /// `colour = TFACTOR.rgb` and `alpha = TFACTOR.a * silhouette.a`.
    shadow_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TextureUniform {
    u_row: [f32; 4],
    v_row: [f32; 4],
    alpha_test: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

const PROJ_SHADER: &str = include_str!("../../shaders/object_shadow.wgsl");
const BAKE_SHADER: &str = include_str!("../../shaders/object_shadow_texture.wgsl");

/// Shared GPU state. One instance per renderer (BuildingsRenderer /
/// RobotsRenderer) is fine — pipelines are stateless.
pub struct ShadowSystem {
    proj_pipeline: wgpu::RenderPipeline,
    bake_pipeline: wgpu::RenderPipeline,
    proj_bgl: wgpu::BindGroupLayout,
    bake_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    proj_uniform_buffer: wgpu::Buffer,
}

impl ShadowSystem {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let proj_bgl = create_proj_bgl(device);
        let bake_bgl = create_bake_bgl(device);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let proj_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShadowSystem proj UB"),
            contents: bytemuck::bytes_of(&ProjUniform {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                shadow_color: [0.0, 0.0, 0.0, 0.45],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let proj_pipeline = create_proj_pipeline(device, config, &proj_bgl);
        let bake_pipeline = create_bake_pipeline(device, &bake_bgl);

        Self {
            proj_pipeline,
            bake_pipeline,
            proj_bgl,
            bake_bgl,
            sampler,
            proj_uniform_buffer,
        }
    }

    /// Refresh the shared view-projection uniform and the map's
    /// `ShadowColor`. Call once per frame before `render`. `shadow_color` is
    /// the CMAP DATA_SHADOWCOLOR DWORD packed as 0xAARRGGBB (matches D3D's
    /// `D3DRS_TEXTUREFACTOR` byte order — see MatrixMap.cpp:1830).
    pub fn update_view(&self, queue: &wgpu::Queue, view_proj: Mat4, shadow_color: u32) {
        let a = ((shadow_color >> 24) & 0xFF) as f32 / 255.0;
        let r = ((shadow_color >> 16) & 0xFF) as f32 / 255.0;
        let g = ((shadow_color >> 8) & 0xFF) as f32 / 255.0;
        let b = (shadow_color & 0xFF) as f32 / 255.0;
        queue.write_buffer(
            &self.proj_uniform_buffer,
            0,
            bytemuck::bytes_of(&ProjUniform {
                view_proj: view_proj.to_cols_array_2d(),
                shadow_color: [r, g, b, a],
            }),
        );
    }

    /// Compute projector axes from the model's mesh and current world
    /// transform. Returns `None` if the mesh is empty or the light direction
    /// is degenerate.
    pub fn calc_proj(
        &self,
        mesh_local_vertices: &[Vec3],
        light_world: Vec3,
        world_matrix: Mat4,
    ) -> Option<ShadowProjData> {
        calc_shadow_proj_data(mesh_local_vertices, light_world, world_matrix)
    }

    /// Bake the silhouette texture using the precomputed projection.
    ///
    /// `vertices` and `surfaces` describe the model in object-local space.
    pub fn bake_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[ShadowMeshVertex],
        surfaces: &[ShadowMeshSurface],
        proj: &ShadowProjData,
        texture_size: u32,
    ) -> Option<wgpu::TextureView> {
        if vertices.is_empty() || surfaces.is_empty() {
            return None;
        }

        // Faithful port of `CVectorObject::CalcShadowTexture`
        // (VectorObject.cpp:1156-1175): build mView in local space via
        // `LookAtLH(local_center - vz, local_center, -vy)`, then scale rows
        // 0/1 by `+1/|vx|` / `+1/|vy|`.
        //
        // mView's xaxis = -x_local (LookAt orientation), so row 0 after
        // scaling is `-x_local/|vx|` for xyz and `+x_local·center/|vx|` for w.
        // Same pattern for V on `-y_local`.
        let u_row = Vec4::new(
            -proj.x_local.x / proj.dim_x,
            -proj.x_local.y / proj.dim_x,
            -proj.x_local.z / proj.dim_x,
            proj.x_local.dot(proj.center_local) / proj.dim_x,
        );
        let v_row = Vec4::new(
            -proj.y_local.x / proj.dim_y,
            -proj.y_local.y / proj.dim_y,
            -proj.y_local.z / proj.dim_y,
            proj.y_local.dot(proj.center_local) / proj.dim_y,
        );

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowSystem silhouette tex"),
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowSystem silhouette depth"),
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());

        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShadowSystem bake VB"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ShadowSystem bake encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ShadowSystem bake pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
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
            pass.set_pipeline(&self.bake_pipeline);
            pass.set_vertex_buffer(0, vbuf.slice(..));

            for surface in surfaces {
                if surface.indices.is_empty() {
                    continue;
                }
                let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ShadowSystem bake IB"),
                    contents: bytemuck::cast_slice(&surface.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ShadowSystem bake UB"),
                    contents: bytemuck::bytes_of(&TextureUniform {
                        u_row: u_row.to_array(),
                        v_row: v_row.to_array(),
                        alpha_test: [surface.alpha_test as u32, 0, 0, 0],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ShadowSystem bake BG"),
                    layout: &self.bake_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ub.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&surface.diffuse),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });
                pass.set_bind_group(0, &bg, &[]);
                pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..surface.indices.len() as u32, 0, 0..1);
            }
        }
        queue.submit([encoder.finish()]);

        Some(view)
    }

    /// Build the projected shadow geometry over the terrain in a `map_radius`
    /// cell window centered on the object's world position. UVs map each
    /// terrain corner into the silhouette texture using the same projector
    /// the bake used.
    ///
    /// `map_center` is the renderer's `(cx, cy)` half-world offset (subtracted
    /// from world positions to match the centered render space all other
    /// renderers in this port use).
    #[allow(clippy::too_many_arguments)]
    pub fn build_geometry(
        &self,
        device: &wgpu::Device,
        map: &GameMap,
        proj: &ShadowProjData,
        texture: &wgpu::TextureView,
        map_radius: i32,
        map_center: [f32; 2],
    ) -> Option<ShadowBatch> {
        if proj.dim_x < 1e-3 || proj.dim_y < 1e-3 {
            return None;
        }

        // Faithful port of `ShadowProjBuildGeomInt`
        // (MatrixShadowManager.cpp:108-115): mView in world space is
        // `LookAtLH(center - vz_world, center, -vy_world)`, then rows 0/1
        // scaled by `_sx = -1/|vx|`, `_sy = -1/|vy|`.
        //
        // mView's xaxis = -x_world. After `*_sx` the row becomes
        // `+x_world/|vx|` for xyz and `-x_world·center_world/|vx|` for w.
        // Same pattern for V on `-y_world`.
        let tu_row = Vec4::new(
            proj.x_world.x / proj.dim_x,
            proj.x_world.y / proj.dim_x,
            proj.x_world.z / proj.dim_x,
            -proj.x_world.dot(proj.center_world) / proj.dim_x,
        );
        let tv_row = Vec4::new(
            proj.y_world.x / proj.dim_y,
            proj.y_world.y / proj.dim_y,
            proj.y_world.z / proj.dim_y,
            -proj.y_world.dot(proj.center_world) / proj.dim_y,
        );

        // Cell window centered on the object's world translation, matching
        // `Float2Int(objma._41 / GLOBAL_SCALE) - mapradius`
        // (MatrixShadowManager.cpp:93-96). `Float2Int` uses x87 `fistp` which
        // rounds to nearest even — `round_ties_even` is the Rust equivalent.
        let cx_cell = (proj.world_translation_xy[0] / GLOBAL_SCALE).round_ties_even() as i32;
        let cy_cell = (proj.world_translation_xy[1] / GLOBAL_SCALE).round_ties_even() as i32;
        let msx_raw = cx_cell - map_radius;
        let msy_raw = cy_cell - map_radius;
        let mex = (msx_raw + map_radius * 2).min(map.size_x as i32);
        let mey = (msy_raw + map_radius * 2).min(map.size_y as i32);
        let msx = msx_raw.max(0);
        let msy = msy_raw.max(0);
        if msx >= mex || msy >= mey {
            return None;
        }

        let nx = (mex - msx + 1) as usize;
        let ny = (mey - msy + 1) as usize;

        struct GridVert {
            p: Vec3,
            uv: [f32; 2],
            outside: bool,
        }
        let mut grid: Vec<GridVert> = Vec::with_capacity(nx * ny);
        let mut prj_off = true;
        for y in msy..=mey {
            for x in msx..=mex {
                let idx = (y as usize) * (map.size_x + 1) + (x as usize);
                let p = &map.points[idx];
                let pos = Vec3::new(
                    x as f32 * GLOBAL_SCALE,
                    y as f32 * GLOBAL_SCALE,
                    p.z + SHADOW_ALTITUDE,
                );
                let v4 = Vec4::new(pos.x, pos.y, pos.z, 1.0);
                let tu = tu_row.dot(v4) + 0.5;
                let tv = tv_row.dot(v4) + 0.5;
                let outside = !(0.0..=1.0).contains(&tu) || !(0.0..=1.0).contains(&tv);
                if !outside {
                    prj_off = false;
                }
                grid.push(GridVert {
                    p: pos,
                    uv: [tu, tv],
                    outside,
                });
            }
        }
        if prj_off {
            return None;
        }

        let mut shadow_verts: Vec<ShadowVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut vmap: Vec<i32> = vec![-1; nx * ny];

        for y in 0..(ny - 1) {
            for x in 0..(nx - 1) {
                let i00 = y * nx + x;
                let i10 = y * nx + (x + 1);
                let i01 = (y + 1) * nx + x;
                let i11 = (y + 1) * nx + (x + 1);

                if grid[i00].outside
                    && grid[i10].outside
                    && grid[i01].outside
                    && grid[i11].outside
                {
                    continue;
                }

                let ux = (x as i32 + msx) as usize;
                let uy = (y as i32 + msy) as usize;
                if ux >= map.size_x || uy >= map.size_y {
                    continue;
                }
                let unit = &map.units[uy * map.size_x + ux];
                if (unit.flags & CELLFLAG_LAND) == 0 {
                    continue;
                }

                let mut emit = |gi: usize| -> u32 {
                    if vmap[gi] >= 0 {
                        return vmap[gi] as u32;
                    }
                    let p = &grid[gi];
                    let v_idx = shadow_verts.len() as u32;
                    shadow_verts.push(ShadowVertex {
                        position: [
                            p.p.x - map_center[0],
                            p.p.y - map_center[1],
                            p.p.z,
                        ],
                        uv: p.uv,
                    });
                    vmap[gi] = v_idx as i32;
                    v_idx
                };
                let a = emit(i00);
                let b = emit(i10);
                let c = emit(i01);
                let d = emit(i11);
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }

        if indices.is_empty() {
            return None;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShadowSystem ground VB"),
            contents: bytemuck::cast_slice(&shadow_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShadowSystem ground IB"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShadowSystem proj BG"),
            layout: &self.proj_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.proj_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        Some(ShadowBatch {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            bind_group,
        })
    }

    /// Render every batch with the projection pipeline. Caller is responsible
    /// for `update_view` having been called this frame.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, batches: &'a [ShadowBatch]) {
        if batches.is_empty() {
            return;
        }
        pass.set_pipeline(&self.proj_pipeline);
        for batch in batches {
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..batch.num_indices, 0, 0..1);
        }
    }
}

fn calc_shadow_proj_data(
    mesh_local_vertices: &[Vec3],
    light_world: Vec3,
    world_matrix: Mat4,
) -> Option<ShadowProjData> {
    if mesh_local_vertices.is_empty() {
        return None;
    }
    let light_world = light_world.normalize_or_zero();
    if light_world.length_squared() < 1e-6 {
        return None;
    }

    // Rotation portion of the world matrix (rigid transforms only here).
    let world_rot = Mat3::from_cols(
        world_matrix.x_axis.truncate(),
        world_matrix.y_axis.truncate(),
        world_matrix.z_axis.truncate(),
    );
    let world_rot_t = world_rot.transpose();

    // Light into local space — port of `D3DXVec3TransformNormal(&lv, &light,
    // &iobjma)` (MatrixShadowManager.cpp:323).
    let light_local = (world_rot_t * light_world).normalize_or_zero();
    if light_local.length_squared() < 1e-6 {
        return None;
    }

    // Faithful port of `CVectorObject::CalcShadowProjMatrix`
    // (VectorObject.cpp:1085-1114). Build `ml = LookAtLH(-lv, 0, +Z)`,
    // measure the mesh AABB in that frame, then map the four reference
    // corners back through `inverse(ml)` to get pd in local space.
    //
    // Glam's `Mat4::look_at_lh(eye, target, up)` produces the same view
    // matrix D3D's row-vector LookAtLH does (column-major / row-major are
    // transposes — both apply identically to `Mat4 * Vec4` here).
    let ml = Mat4::look_at_lh(-light_local, Vec3::ZERO, Vec3::Z);

    let mut vmin = Vec3::splat(f32::INFINITY);
    let mut vmax = Vec3::splat(f32::NEG_INFINITY);
    for v in mesh_local_vertices {
        let p_lv = ml.transform_point3(*v);
        vmin = vmin.min(p_lv);
        vmax = vmax.max(p_lv);
    }
    vmin -= Vec3::splat(ADD_SIZE);
    vmax += Vec3::splat(ADD_SIZE);

    // pd's four reference points in light-view space, mapped back to local.
    let ml_inv = ml.inverse();
    let local_vpos = ml_inv.transform_point3(vmin);
    let local_vx_corner = ml_inv.transform_point3(Vec3::new(vmax.x, vmin.y, vmin.z));
    let local_vy_corner = ml_inv.transform_point3(Vec3::new(vmin.x, vmax.y, vmin.z));
    let local_vz_corner = ml_inv.transform_point3(Vec3::new(vmin.x, vmin.y, vmax.z));

    let vx_local = local_vx_corner - local_vpos;
    let vy_local = local_vy_corner - local_vpos;
    let vz_local = local_vz_corner - local_vpos;

    let dim_x = vx_local.length().max(1e-3);
    let dim_y = vy_local.length().max(1e-3);
    let dim_z = vz_local.length().max(1e-3);

    // Unit basis vectors derived from pd: x_local = vx_local/dim_x, etc.
    // These are the same axes the C++ mView in CalcShadowTexture /
    // ShadowProjBuildGeomInt uses — `LookAtLH(center - vz, center, -vy)`
    // gives `xaxis = -x_local`, `yaxis = -y_local`, `zaxis = +z_local`.
    let x_local = vx_local / dim_x;
    let y_local = vy_local / dim_y;
    let z_local = vz_local / dim_z;

    // Center of the AABB bottom face in local space. The C++ uses
    // `(pd.vx + pd.vy) * 0.5 + pd.vpos` (MatrixShadowManager.cpp:101) for
    // the projector "look-at" target.
    let center_local = local_vpos + (vx_local + vy_local) * 0.5;

    // World-space versions: rotate the basis, transform center as a point.
    let x_world = world_rot * x_local;
    let y_world = world_rot * y_local;
    let z_world = world_rot * z_local;
    let center_world = world_matrix.transform_point3(center_local);

    Some(ShadowProjData {
        x_local,
        y_local,
        z_local,
        x_world,
        y_world,
        z_world,
        center_local,
        center_world,
        dim_x,
        dim_y,
        dim_z,
        world_translation_xy: [world_matrix.w_axis.x, world_matrix.w_axis.y],
    })
}

fn create_proj_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ShadowSystem proj BGL"),
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
    })
}

fn create_bake_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ShadowSystem bake BGL"),
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
    })
}

fn create_proj_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ShadowSystem proj shader"),
        source: wgpu::ShaderSource::Wgsl(PROJ_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ShadowSystem proj PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ShadowSystem proj pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ShadowVertex>() as u64,
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
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_bake_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ShadowSystem bake shader"),
        source: wgpu::ShaderSource::Wgsl(BAKE_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ShadowSystem bake PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ShadowSystem bake pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ShadowMeshVertex>() as u64,
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
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
