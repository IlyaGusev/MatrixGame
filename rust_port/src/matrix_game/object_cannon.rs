//! Port of `CMatrixCannon` (MatrixObjectCannon.{cpp,hpp}) — the
//! turret/cannon game object + its renderer. Mirrors the layout the
//! C++ uses in `CMatrixCannon::RNeed` at MatrixObjectCannon.cpp:
//! 189-260: three sub-VOs per cannon (Basis, Turret{N}, Shaft{N})
//! mounted at matrix-IDs 20..22 that compose the full turret mesh.
//!
//! Scope: the factory interface needs cannons to render on the
//! building after `try_place_turret` commits. The AI / firing /
//! damage paths are out of scope for the constructor port — they
//! land with `CMatrixCannon::Takt` / `CMatrixCannon::Damage` in a
//! separate pass.
//!
//! Asset paths (StringConstants.hpp:188 + MatrixObjectCannon.cpp:
//! 209-229):
//!   * `Matrix/Cannon/Basis.vo` — shared base plate
//!   * `Matrix/Cannon/Turret{1-4}.vo` — rotating head
//!   * `Matrix/Cannon/Shaft{1-4}.vo` — barrel
//!
//! The C++ also loads per-kind textures from the VO's referenced
//! material; we re-use the existing `MaterialSpec` resolver (same as
//! `BuildingsRenderer`) so any textures the VO references are picked
//! up automatically.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{unpack_rgb, FOG_END, FOG_START};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectId, ObjectType, Objects, MR_ALL, MR_MATRIX,
};
use crate::matrix_game::logic::Rnd;
use crate::matrix_game::shadow::{
    ShadowBatch, ShadowMeshSurface, ShadowMeshVertex, ShadowSystem,
};
use crate::matrix_lib::three_g::texture::{
    create_solid_texture, create_texture_from_rgba, decode_texture_bytes,
};
use crate::matrix_lib::three_g::vector_object::{self, MaterialSpec, VoMesh};

/// Port of `ECannonState` (MatrixObjectCannon.hpp). Drives whether
/// the cannon is mid-construction (HP ramping, invulnerable) or live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannonState {
    /// `CANNON_IDLE` — fully built and operational.
    Idle,
    /// `CANNON_UNDER_CONSTRUCTION` — HP ramps from 0 to `hit_point_max`
    /// over `turret_build_time_ms`. Invulnerable while in this state.
    UnderConstruction,
    /// `CANNON_DIP` — destroyed, mid-explosion. Not driven yet (the
    /// damage path lands with the AI/firing port).
    #[allow(dead_code)]
    Dip,
}

/// Port of the Cannon game object. Instance-level state.
pub struct Cannon {
    core: ObjectCore,
    rchange: u32,
    object_state: u32,
    ablaze_ttl: i32,
    shorted_ttl: i32,

    /// World-space anchor XY (matches `m_Pos` on CMatrixCannon).
    pub pos: glam::Vec2,
    /// Platform height in world-Z — taken from the owning base's
    /// `build_z + BASE_PLATFORM_TOP_OFFSET` at spawn time.
    pub pos_z: f32,
    /// Rotation about Z — the C++ `m_Angle` drives the entire cannon
    /// matrix (MatrixObjectCannon.cpp:258).
    pub angle: f32,
    /// Owning side (PLAYER_SIDE for the local-player factories).
    pub side: i32,
    /// Which cannon variant 1..=4 (`m_Num` in C++).
    pub kind: i32,
    /// Parent building (for the AI "defend this base" logic later).
    pub parent: Option<ObjectId>,
    /// Slot index on the parent (0..turrets_max-1). Drives the
    /// geometric offset from the base centre.
    pub slot: i32,

    /// Hit points — seeded from `g_Config.m_CannonsProps[kind-1].m_Hitpoint`
    /// at spawn time (MatrixObjectCannon.cpp:201-202). When the cannon
    /// is `UnderConstruction`, this ramps from 0 toward `hit_point_max`
    /// in `tick_construction`.
    pub hit_point: f32,
    pub hit_point_max: f32,

    /// `m_CurrState` (MatrixObjectCannon.hpp).
    pub state: CannonState,
    /// `m_Invulnerability` flag — port of `SetInvulnerability`
    /// (MatrixObjectCannon.cpp:1486+ / MatrixSide.cpp:631). Damage paths
    /// must early-return while this is true.
    pub invulnerable: bool,
}

impl Cannon {
    pub fn new(
        pos: glam::Vec2,
        pos_z: f32,
        angle: f32,
        side: i32,
        kind: i32,
        parent: Option<ObjectId>,
        slot: i32,
    ) -> Self {
        let hp = crate::matrix_game::config::global()
            .turrets
            .cannons
            .get((kind - 1).max(0) as usize)
            .map(|c| c.hitpoint)
            .unwrap_or(100.0);
        let core = ObjectCore {
            obj_type: ObjectType::Cannon,
            geo_center: glam::Vec3::new(pos.x, pos.y, pos_z),
            radius: 20.0,
            matrix: glam::Mat4::from_translation(glam::Vec3::new(pos.x, pos.y, pos_z)),
            ..Default::default()
        };
        Self {
            core,
            rchange: MR_ALL,
            object_state: 0,
            ablaze_ttl: 0,
            shorted_ttl: 0,
            pos,
            pos_z,
            angle,
            side,
            kind,
            parent,
            slot,
            hit_point: hp,
            hit_point_max: hp,
            state: CannonState::Idle,
            invulnerable: false,
        }
    }

    /// Flip into `UnderConstruction`: HP=0, invulnerable. Called once
    /// at placement-confirmation time. Mirrors MatrixSide.cpp:629-631
    /// + 649.
    pub fn begin_construction(&mut self) {
        self.state = CannonState::UnderConstruction;
        self.invulnerable = true;
        self.hit_point = 0.0;
    }

    /// Build-stack timer notifies the cannon every tick — `progress` is
    /// in `[0, 1]`. Ramps HP and, on completion, flips to Idle and drops
    /// invulnerability. Port of MatrixObjectBuilding.cpp:1764-1813.
    pub fn tick_construction(&mut self, progress: f32) {
        if !matches!(self.state, CannonState::UnderConstruction) {
            return;
        }
        let p = progress.clamp(0.0, 1.0);
        self.hit_point = self.hit_point_max * p;
        if p >= 1.0 {
            self.state = CannonState::Idle;
            self.invulnerable = false;
            self.hit_point = self.hit_point_max;
        }
    }
}

impl MapStatic for Cannon {
    fn core(&self) -> &ObjectCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut ObjectCore {
        &mut self.core
    }
    fn rchange(&self) -> u32 {
        self.rchange
    }
    fn rchange_set(&mut self, b: u32) {
        self.rchange |= b;
    }
    fn rchange_clear(&mut self, b: u32) {
        self.rchange &= !b;
    }
    fn object_state(&self) -> u32 {
        self.object_state
    }
    fn object_state_set(&mut self, b: u32) {
        self.object_state |= b;
    }
    fn object_state_clear(&mut self, b: u32) {
        self.object_state &= !b;
    }
    fn ablaze_ttl(&self) -> i32 {
        self.ablaze_ttl
    }
    fn set_ablaze_ttl(&mut self, t: i32) {
        self.ablaze_ttl = t;
    }
    fn shorted_ttl(&self) -> i32 {
        self.shorted_ttl
    }
    fn set_shorted_ttl(&mut self, t: i32) {
        self.shorted_ttl = t;
    }
    fn r_need(&mut self, need: u32) {
        if need & self.rchange & MR_MATRIX != 0 {
            self.rchange &= !MR_MATRIX;
            let (s, c) = self.angle.sin_cos();
            self.core.matrix = glam::Mat4::from_cols(
                glam::Vec4::new(c, s, 0.0, 0.0),
                glam::Vec4::new(-s, c, 0.0, 0.0),
                glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
                glam::Vec4::new(self.pos.x, self.pos.y, self.pos_z, 1.0),
            );
            self.core.inv_matrix = self.core.matrix.inverse();
        }
        // Graph / shadows not yet driven by per-instance state —
        // clear the bits so `r_need` doesn't spin.
        self.rchange &= !(crate::matrix_game::map_static::MR_GRAPH
            | crate::matrix_game::map_static::MR_SHADOW_PROJ_GEOM
            | crate::matrix_game::map_static::MR_SHADOW_PROJ_TEX
            | crate::matrix_game::map_static::MR_SHADOW_STENCIL
            | crate::matrix_game::map_static::MR_MINIMAP);
    }
    fn takt(&mut self, _cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {}
    fn logic_takt(&mut self, _cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {}
    fn side(&self) -> i32 {
        self.side
    }
}

// ── Renderer ───────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceData {
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
    row3: [f32; 4],
    terrain_color: [f32; 4],
    unit_offset: [f32; 4],
    /// Per-side tint — same role as the buildings / robots instance
    /// data. Cannons also carry `m_Side` in the C++ and get the
    /// `GetSideColorTexture` treatment on team-marker surfaces.
    side_color: [f32; 4],
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
    camera_pos: [f32; 4],
    time_ms: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaterialUniform {
    flags: [u32; 4],
    scroll: [f32; 4],
}

/// One sub-mesh draw — basis / turret / shaft for a given kind.
struct CannonSurface {
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    bind_group: wgpu::BindGroup,
}

struct CannonMesh {
    vertex_buffer: wgpu::Buffer,
    surfaces: Vec<CannonSurface>,
}

/// Per-cannon-kind GPU resources.
struct KindGpu {
    /// `Matrix/Cannon/Basis.vo` (shared across kinds, but we keep a
    /// copy per kind so each kind's instance buffer is contiguous).
    basis: Option<CannonMesh>,
    /// `Matrix/Cannon/Turret{N}.vo`.
    turret: Option<CannonMesh>,
    /// `Matrix/Cannon/Shaft{N}.vo` — the barrel.
    shaft: Option<CannonMesh>,
    /// Mesh source for the silhouette baker — concatenates Basis + Turret +
    /// Shaft (with mount offsets applied) into a single vertex pool plus
    /// per-surface index lists. Empty when none of the sub-meshes loaded.
    shadow_source: Option<KindShadowSource>,
}

struct KindShadowSource {
    vertices: Vec<ShadowMeshVertex>,
    surfaces: Vec<ShadowMeshSurface>,
}

pub struct CannonsRenderer {
    pipeline: wgpu::RenderPipeline,
    kinds: Vec<KindGpu>,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    /// Cached `m_ShadowColor` from the map (DATA_SHADOWCOLOR), packed as
    /// 0xAARRGGBB and forwarded to `ShadowSystem::update_view` each frame.
    shadow_color: u32,
    time_ms: f32,
    center: [f32; 2],
    /// Per-kind instance slot offsets computed each frame by
    /// `sync_cannons` — index `k` is the start slot for cannons of
    /// kind `k+1` in the shared instance buffer, and `k+4` is the
    /// count.
    draws: Vec<(i32, u32, u32)>, // (kind, offset, count)
    /// Slot-marker draws — Basis-only stamps placed at `(offset, count)`
    /// in the instance buffer. Rendered separately from `draws` because
    /// they only emit the `Basis` sub-mesh (no Turret / Shaft).
    marker_draw: Option<(u32, u32)>,
    /// Projected-shadow infrastructure (pipelines, sampler, shared UB).
    shadow_system: ShadowSystem,
    /// Per-cannon baked silhouette texture, keyed by `ObjectId`. Bake uses
    /// the cannon's actual world matrix on first sight so the texture's
    /// orientation matches the projection's axes. Stale entries are evicted
    /// at the start of every `sync_cannons`.
    shadow_textures: HashMap<ObjectId, wgpu::TextureView>,
    /// Per-cannon ground-projection mesh, rebuilt per frame from the
    /// live cannon list inside `sync_cannons`.
    shadow_batches: Vec<ShadowBatch>,
}

const MAX_LIVE_CANNONS: u32 = 64;

impl CannonsRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;

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
            label: Some("Cannons UB"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                ambient_color,
                light_color,
                light_dir,
                camera_pos: [0.0, 0.0, 0.0, 1.0],
                time_ms: [0.0, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = create_bgl(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let pipeline = create_pipeline(device, config, &bgl);

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cannons Inst VB"),
            size: (MAX_LIVE_CANNONS as u64) * std::mem::size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
        let fallback_tex = create_solid_texture(device, queue, [200, 200, 200, 255]);
        let black_tex = create_solid_texture(device, queue, [0, 0, 0, 255]);
        let transparent_tex = create_solid_texture(device, queue, [0, 0, 0, 0]);

        // Basis is shared across all 4 kinds in the C++ (each kind
        // still needs its own `CannonMesh` because the texture refs
        // baked into the material cache can diverge). We load it
        // once and clone the mesh data cheaply.
        let basis_bytes = read_texture("Matrix/Cannon/Basis.vo");

        let shadow_system = ShadowSystem::new(device, config);

        let mut kinds: Vec<KindGpu> = Vec::with_capacity(4);
        let mut loaded = 0;
        for kind in 1..=4 {
            // Sub-mesh mount chain — port of `CMatrixCannon::RNeed`'s
            // `tm = m_Unit[i].m_Graph->GetMatrixById(20)` walk
            // (MatrixObjectCannon.cpp:282-329):
            //   Basis lives at the cannon origin.
            //   Turret mounts at Basis.matrix(20).translation.
            //   Shaft mounts at Basis.matrix(20) + Turret.matrix(20)
            //                                       (translations added).
            // We bake the cumulative offset into vertex positions at
            // load time — the per-instance transform is unchanged, so
            // the renderer doesn't need a second uniform.
            let basis_offset = glam::Vec3::ZERO;
            let basis = basis_bytes.as_deref().and_then(|b| {
                load_mesh(
                    b,
                    "Matrix/Cannon/Basis.vo",
                    basis_offset,
                    device,
                    queue,
                    &bgl,
                    &sampler,
                    &uniform_buffer,
                    &mut tex_cache,
                    read_texture,
                    &fallback_tex,
                    &black_tex,
                    &transparent_tex,
                )
            });
            let basis_mount = basis_bytes
                .as_deref()
                .and_then(|b| read_mount_offset(b, 20, "Basis"))
                .unwrap_or(glam::Vec3::ZERO);
            let turret_offset = basis_mount;
            let turret_bytes = read_texture(&format!("Matrix/Cannon/Turret{}.vo", kind));
            let turret = turret_bytes.as_deref().and_then(|b| {
                load_mesh(
                    b,
                    &format!("Matrix/Cannon/Turret{}.vo", kind),
                    turret_offset,
                    device,
                    queue,
                    &bgl,
                    &sampler,
                    &uniform_buffer,
                    &mut tex_cache,
                    read_texture,
                    &fallback_tex,
                    &black_tex,
                    &transparent_tex,
                )
            });
            let turret_mount = turret_bytes
                .as_deref()
                .and_then(|b| read_mount_offset(b, 20, &format!("Turret{kind}")))
                .unwrap_or(glam::Vec3::ZERO);
            let shaft_offset = basis_mount + turret_mount;
            let shaft = {
                let path = format!("Matrix/Cannon/Shaft{}.vo", kind);
                read_texture(&path).and_then(|b| {
                    load_mesh(
                        &b,
                        &path,
                        shaft_offset,
                        device,
                        queue,
                        &bgl,
                        &sampler,
                        &uniform_buffer,
                        &mut tex_cache,
                        read_texture,
                        &fallback_tex,
                        &black_tex,
                        &transparent_tex,
                    )
                })
            };
            if basis.is_some() || turret.is_some() || shaft.is_some() {
                loaded += 1;
            }

            // Concatenate the silhouette source rows from every loaded
            // sub-mesh into one mesh in the kind's local frame. Sub-mesh
            // index lists are remapped to point into the pooled vertex
            // buffer.
            let mut shadow_vertices: Vec<ShadowMeshVertex> = Vec::new();
            let mut shadow_surfaces: Vec<ShadowMeshSurface> = Vec::new();
            for sub in [basis.as_ref(), turret.as_ref(), shaft.as_ref()]
                .into_iter()
                .flatten()
            {
                let base = shadow_vertices.len() as u32;
                shadow_vertices.extend(sub.shadow_vertices.iter().copied());
                for s in &sub.shadow_surfaces {
                    shadow_surfaces.push(ShadowMeshSurface {
                        indices: s.indices.iter().map(|i| i + base).collect(),
                        diffuse: s.diffuse.clone(),
                        alpha_test: s.alpha_test,
                    });
                }
            }
            let shadow_source = if shadow_vertices.is_empty() {
                None
            } else {
                Some(KindShadowSource {
                    vertices: shadow_vertices,
                    surfaces: shadow_surfaces,
                })
            };

            kinds.push(KindGpu {
                basis: basis.map(|m| m.mesh),
                turret: turret.map(|m| m.mesh),
                shaft: shaft.map(|m| m.mesh),
                shadow_source,
            });
        }

        if loaded == 0 {
            log::warn!("cannons: no cannon meshes loaded");
            return None;
        }
        log::info!("cannons: loaded {} kinds", loaded);

        Some(Self {
            pipeline,
            kinds,
            instance_buffer,
            instance_capacity: MAX_LIVE_CANNONS,
            uniform_buffer,
            fog_color,
            ambient_color,
            light_color,
            light_dir,
            shadow_color: map.shadow_color,
            time_ms: 0.0,
            center: [cx, cy],
            draws: Vec::new(),
            marker_draw: None,
            shadow_system,
            shadow_textures: HashMap::new(),
            shadow_batches: Vec::new(),
        })
    }

    /// Per-frame time advance for shader-driven effects (matches the
    /// building/robot renderers). Noop for now — cannons carry no
    /// time-based animation in the constructor port.
    pub fn takt(&mut self, dt_ms: f32) {
        self.time_ms += dt_ms;
    }

    /// Walk live cannons and populate the instance buffer. The
    /// optional `ghost` parameter renders a single extra cannon at the
    /// turret-build placement preview position, tinted by validity
    /// (green = can build, red = can't). Mirrors the
    /// `m_CannonForBuild.m_Cannon->SetTerainColor(0xFF{00FF00,FF0000})`
    /// path at MatrixSide.cpp:554-558.
    pub fn sync_cannons(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        objs: &mut Objects,
        map: &GameMap,
        ghost: Option<GhostCannon>,
        markers: &[TurretSlotMarker],
    ) {
        let [cx, cy] = self.center;
        self.draws.clear();
        self.marker_draw = None;
        self.shadow_batches.clear();
        let mut instance_data: Vec<InstanceData> = Vec::with_capacity(16);

        // Group cannons by kind so draws are contiguous.
        let mut by_kind: [Vec<InstanceData>; 4] = [vec![], vec![], vec![], vec![]];
        let light_world = Vec3::new(
            map.light_main_dir[0],
            map.light_main_dir[1],
            map.light_main_dir[2],
        );
        let mut alive_shadow_ids: std::collections::HashSet<ObjectId> =
            std::collections::HashSet::new();
        for id in objs.iter_live() {
            let Some(obj) = objs.get(id) else { continue };
            if !matches!(obj.core().obj_type, ObjectType::Cannon) {
                continue;
            }
            let c: &Cannon = unsafe { &*(obj as *const dyn MapStatic as *const Cannon) };
            let k = ((c.kind - 1).max(0) as usize).min(3);
            by_kind[k].push(cannon_instance(c, cx, cy, map));

            // Per-instance shadow: bake the silhouette using THIS cannon's
            // world rotation so the texture's projector axes match the
            // ground-projection's axes. Bake is one-shot (cached by
            // ObjectId); the ground geometry rebuilds per frame.
            if let Some(kind_gpu) = self.kinds.get(k) {
                if let Some(src) = kind_gpu.shadow_source.as_ref() {
                    let world_uc = cannon_world_uncentered(c);
                    let local_pts: Vec<Vec3> = src
                        .vertices
                        .iter()
                        .map(|v| Vec3::from_array(v.position))
                        .collect();
                    if let Some(proj) =
                        self.shadow_system.calc_proj(&local_pts, light_world, world_uc)
                    {
                        let texture = self
                            .shadow_textures
                            .entry(id)
                            .or_insert_with(|| {
                                self.shadow_system
                                    .bake_texture(
                                        device,
                                        queue,
                                        &src.vertices,
                                        &src.surfaces,
                                        &proj,
                                        64,
                                    )
                                    .unwrap_or_else(|| {
                                        let dummy =
                                            device.create_texture(&wgpu::TextureDescriptor {
                                                label: Some("cannon shadow dummy"),
                                                size: wgpu::Extent3d {
                                                    width: 1,
                                                    height: 1,
                                                    depth_or_array_layers: 1,
                                                },
                                                mip_level_count: 1,
                                                sample_count: 1,
                                                dimension: wgpu::TextureDimension::D2,
                                                format: wgpu::TextureFormat::Rgba8Unorm,
                                                usage: wgpu::TextureUsages::TEXTURE_BINDING
                                                    | wgpu::TextureUsages::COPY_DST,
                                                view_formats: &[],
                                            });
                                        dummy.create_view(&Default::default())
                                    })
                            })
                            .clone();
                        if let Some(batch) = self.shadow_system.build_geometry(
                            device, map, &proj, &texture, 6, [cx, cy],
                        ) {
                            self.shadow_batches.push(batch);
                        }
                        alive_shadow_ids.insert(id);
                    }
                }
            }
        }
        if let Some(g) = ghost {
            let k = ((g.kind - 1).max(0) as usize).min(3);
            by_kind[k].push(ghost_instance(&g, cx, cy));
        }

        let mut offset: u32 = 0;
        for (k, list) in by_kind.iter().enumerate() {
            let count = list.len() as u32;
            if count == 0 {
                continue;
            }
            if offset + count > self.instance_capacity {
                break;
            }
            for inst in list {
                instance_data.push(*inst);
            }
            self.draws.push((k as i32 + 1, offset, count));
            offset += count;
        }

        // Slot markers — Basis-only mini-cannons at each free slot.
        // Append after the main cannons in the instance buffer; the
        // render loop emits a separate Basis-only pass for them.
        if !markers.is_empty() {
            let marker_start = offset;
            let mut marker_count: u32 = 0;
            for m in markers {
                if offset + 1 > self.instance_capacity {
                    break;
                }
                instance_data.push(marker_instance(m, cx, cy));
                offset += 1;
                marker_count += 1;
            }
            if marker_count > 0 {
                self.marker_draw = Some((marker_start, marker_count));
            }
        }

        if !instance_data.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instance_data),
            );
        }

        // Drop silhouette textures for cannons that didn't render this frame
        // (destroyed). Without this the cache leaks one texture per dead
        // cannon for the renderer's lifetime.
        self.shadow_textures
            .retain(|id, _| alive_shadow_ids.contains(id));
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
        view_proj: glam::Mat4,
    ) {
        let eye = camera.eye_pos();
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color: self.fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                ambient_color: self.ambient_color,
                light_color: self.light_color,
                light_dir: self.light_dir,
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                time_ms: [self.time_ms, 0.0, 0.0, 0.0],
            }),
        );

        // Project cannon shadows first so the cannons overdraw their bases
        // (matches `DrawShadowsProjFast` running before objects in the
        // original frame order — MatrixMap.cpp:2283).
        self.shadow_system
            .update_view(queue, view_proj, self.shadow_color);
        self.shadow_system.render(pass, &self.shadow_batches);

        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        for (kind, offset, count) in &self.draws {
            let Some(kgpu) = self.kinds.get(((kind - 1).max(0) as usize).min(3)) else {
                continue;
            };
            for mesh in [&kgpu.basis, &kgpu.turret, &kgpu.shaft]
                .iter()
                .flat_map(|m| m.as_ref())
            {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                for surface in &mesh.surfaces {
                    pass.set_bind_group(0, &surface.bind_group, &[]);
                    pass.set_index_buffer(
                        surface.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.draw_indexed(0..surface.num_indices, 0, *offset..(*offset + *count));
                }
            }
        }
        // Slot-marker pass — render the full cannon mesh (Basis +
        // Turret + Shaft) at each free slot so the player sees what
        // would mount there, tinted via `terrain_color`. Picks the
        // first kind whose meshes loaded so this works on builds where
        // a kind failed to parse.
        if let Some((m_offset, m_count)) = self.marker_draw {
            for kgpu in &self.kinds {
                if kgpu.basis.is_none() && kgpu.turret.is_none() && kgpu.shaft.is_none() {
                    continue;
                }
                for mesh in [&kgpu.basis, &kgpu.turret, &kgpu.shaft]
                    .iter()
                    .flat_map(|m| m.as_ref())
                {
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    for surface in &mesh.surfaces {
                        pass.set_bind_group(0, &surface.bind_group, &[]);
                        pass.set_index_buffer(
                            surface.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(
                            0..surface.num_indices,
                            0,
                            m_offset..(m_offset + m_count),
                        );
                    }
                }
                break;
            }
        }
    }
}

/// Uncentered world matrix for a cannon — raw map coordinates (no half-world
/// shift). Used by the shadow system since `map.points[]` are uncentered.
fn cannon_world_uncentered(c: &Cannon) -> Mat4 {
    let (s, co) = c.angle.sin_cos();
    let rot = Mat3::from_cols(
        Vec3::new(co, s, 0.0),
        Vec3::new(-s, co, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    Mat4::from_translation(Vec3::new(c.pos.x, c.pos.y, c.pos_z)) * Mat4::from_mat3(rot)
}

fn cannon_instance(c: &Cannon, cx: f32, cy: f32, _map: &GameMap) -> InstanceData {
    let (s, co) = c.angle.sin_cos();
    let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(c.side);
    // Under-construction tint — port of MatrixObjectCannon.cpp:546
    // (`SetRenderState(D3DRS_TEXTUREFACTOR, 0xFF00FF00)`). The C++
    // applies a flat green factor to mark the cannon as still being
    // built; the live `m_TerainColor` only kicks in once CANNON_IDLE.
    let terrain_color = if matches!(c.state, CannonState::UnderConstruction) {
        [0.0, 1.0, 0.0, 1.0]
    } else {
        [1.0, 1.0, 1.0, 1.0]
    };
    InstanceData {
        row0: [co, -s, 0.0, c.pos.x - cx],
        row1: [s, co, 0.0, c.pos.y - cy],
        row2: [0.0, 0.0, 1.0, c.pos_z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color,
        unit_offset: [0.0, 0.0, 0.0, 0.0],
        side_color: [sr, sg, sb, 1.0],
    }
}

/// Placement-preview snapshot consumed by `sync_cannons`. Carries the
/// world-space pose + tint of the ghost cannon the player sees while
/// `BUILDING_TURRET` is active. The C++ keeps this on
/// `m_CannonForBuild.m_Cannon` (a real `CMatrixCannon` instance held
/// out of the live arena); the Rust port keeps it as plain data on
/// `TurretBuild` and rebuilds the instance each frame.
#[derive(Debug, Clone, Copy)]
pub struct GhostCannon {
    pub kind: i32,
    pub pos: glam::Vec2,
    pub pos_z: f32,
    pub angle: f32,
    /// True = green tint (can build), false = red tint (can't build).
    pub can_build: bool,
    pub side: i32,
}

/// One free turret slot to mark on the terrain while the build picker
/// is open. Rendered as a low-alpha green Basis-only mini-cannon —
/// stand-in for the C++ `CreatePlacesShow` SPOT_TURRET landscape decals
/// (MatrixObjectBuilding.cpp:1617).
#[derive(Debug, Clone, Copy)]
pub struct TurretSlotMarker {
    pub pos: glam::Vec2,
    pub pos_z: f32,
    pub angle: f32,
}

fn marker_instance(m: &TurretSlotMarker, cx: f32, cy: f32) -> InstanceData {
    let (s, co) = m.angle.sin_cos();
    InstanceData {
        row0: [co, -s, 0.0, m.pos.x - cx],
        row1: [s, co, 0.0, m.pos.y - cy],
        row2: [0.0, 0.0, 1.0, m.pos_z],
        row3: [0.0, 0.0, 0.0, 1.0],
        // Cyan tint so the slot marker reads as "buildable here" even
        // against grass / sand. Dim enough not to be confused with a
        // live cannon.
        terrain_color: [0.4, 0.9, 1.0, 1.0],
        unit_offset: [0.0, 0.0, 0.0, 0.0],
        side_color: [0.4, 0.9, 1.0, 1.0],
    }
}

fn ghost_instance(g: &GhostCannon, cx: f32, cy: f32) -> InstanceData {
    let (s, co) = g.angle.sin_cos();
    let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(g.side);
    let tint = if g.can_build {
        [0.4, 1.0, 0.4, 1.0] // green — port of 0xFF00FF00
    } else {
        [1.0, 0.4, 0.4, 1.0] // red — port of 0xFFFF0000
    };
    InstanceData {
        row0: [co, -s, 0.0, g.pos.x - cx],
        row1: [s, co, 0.0, g.pos.y - cy],
        row2: [0.0, 0.0, 1.0, g.pos_z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: tint,
        unit_offset: [0.0, 0.0, 0.0, 0.0],
        side_color: [sr, sg, sb, 1.0],
    }
}

/// Read the matrix-id 20 from a VO and return its translation column.
/// In `CMatrixCannon::RNeed` (MatrixObjectCannon.cpp:321-327) this is
/// the mount-point for the next sub-mesh in the chain. The C++ reads
/// `tm->_41 / _42 / _43` (row-major D3DXMATRIX translation column),
/// which our `[f32; 16]` row-major slice exposes at indices 12-14.
fn read_mount_offset(bytes: &[u8], id: u32, label: &str) -> Option<glam::Vec3> {
    let mesh = vector_object::parse_vo(bytes).ok()?;
    let ids: Vec<(u32, String)> = mesh
        .matrices
        .iter()
        .map(|m| (m.id, m.name.clone()))
        .collect();
    log::debug!("cannons: {label} matrix ids: {:?}", ids);
    let Some(m) = mesh.matrix_by_id(id, 0) else {
        log::warn!("cannons: {label} has no matrix id {id}");
        return None;
    };
    log::debug!(
        "cannons: {label} matrix {id} = [{}, {}, {}]",
        m[12], m[13], m[14]
    );
    Some(glam::Vec3::new(m[12], m[13], m[14]))
}

#[allow(clippy::too_many_arguments)]
// `mount_offset`: world-space translation baked into every vertex of
// this mesh. Composes the cannon's hierarchical sub-mesh chain
// (Basis → Turret → Shaft) without needing per-sub-mesh uniforms. Zero
// for the first mesh in a chain.
/// Loaded sub-mesh + the silhouette source rows it contributed (for the
/// shared per-kind shadow bake). The shadow rows hold the SAME mount-offset
/// vertex positions the renderer uses, so the silhouette captures the full
/// composite cannon in its assembled local frame.
struct LoadedMesh {
    mesh: CannonMesh,
    shadow_vertices: Vec<ShadowMeshVertex>,
    shadow_surfaces: Vec<ShadowMeshSurface>,
}

fn load_mesh(
    bytes: &[u8],
    path: &str,
    mount_offset: glam::Vec3,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform_buffer: &wgpu::Buffer,
    tex_cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    fallback_tex: &wgpu::TextureView,
    black_tex: &wgpu::TextureView,
    transparent_tex: &wgpu::TextureView,
) -> Option<LoadedMesh> {
    let mesh: VoMesh = vector_object::parse_vo(bytes)
        .map_err(|e| log::warn!("cannons: parse {path} failed: {e}"))
        .ok()?;
    let vo_dir = path.rsplit_once('/').map(|(d, _)| format!("{d}/"));

    let vertices: Vec<Vertex> = mesh
        .vertices
        .iter()
        .map(|v| Vertex {
            position: [
                v.position[0] + mount_offset.x,
                v.position[1] + mount_offset.y,
                v.position[2] + mount_offset.z,
            ],
            normal: v.normal,
            uv: v.uv,
        })
        .collect();
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cannons Mesh VB"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let frame = mesh.frames.first()?;
    let mut surfaces = Vec::with_capacity(frame.surfaces.len());
    let shadow_vertices: Vec<ShadowMeshVertex> = vertices
        .iter()
        .map(|v| ShadowMeshVertex {
            position: v.position,
            normal: v.normal,
            uv: v.uv,
        })
        .collect();
    let mut shadow_surfaces: Vec<ShadowMeshSurface> = Vec::new();
    for surf in &frame.surfaces {
        if surf.indices.is_empty() {
            continue;
        }
        let mut material = MaterialSpec::default();
        if let Some(spec) = surf.texture_ref.as_deref() {
            material = vector_object::parse_material_spec_with_prefix(spec, vo_dir.as_deref());
        }

        let (diffuse_view, alpha_test) =
            resolve_diffuse(&material, device, queue, tex_cache, read_texture)
                .unwrap_or_else(|| (fallback_tex.clone(), false));
        shadow_surfaces.push(ShadowMeshSurface {
            indices: surf.indices.clone(),
            diffuse: diffuse_view.clone(),
            alpha_test,
        });
        let gloss_view = resolve_texture(
            material.gloss.as_ref(),
            device,
            queue,
            tex_cache,
            read_texture,
        )
        .unwrap_or_else(|| black_tex.clone());
        let back_view = resolve_texture(
            material.back.as_ref(),
            device,
            queue,
            tex_cache,
            read_texture,
        )
        .unwrap_or_else(|| black_tex.clone());
        let mask_view = resolve_texture(
            material.mask.as_ref(),
            device,
            queue,
            tex_cache,
            read_texture,
        )
        .unwrap_or_else(|| transparent_tex.clone());

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cannons Mesh IB"),
            contents: bytemuck::cast_slice(&surf.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let mat_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cannons Material UB"),
            contents: bytemuck::bytes_of(&MaterialUniform {
                flags: [
                    material.gloss.is_some() as u32,
                    material.back.is_some() as u32,
                    material.mask.is_some() as u32,
                    alpha_test as u32,
                ],
                scroll: [material.scroll[0], material.scroll[1], 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cannons BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&diffuse_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gloss_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&back_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: mat_uniform.as_entire_binding(),
                },
            ],
        });
        surfaces.push(CannonSurface {
            index_buffer,
            num_indices: surf.indices.len() as u32,
            bind_group,
        });
    }

    Some(LoadedMesh {
        mesh: CannonMesh {
            vertex_buffer,
            surfaces,
        },
        shadow_vertices,
        shadow_surfaces,
    })
}

fn resolve_diffuse(
    material: &MaterialSpec,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<(wgpu::TextureView, bool)> {
    let path = material.diffuse.as_ref()?;
    let view = resolve_texture(Some(path), device, queue, cache, read_texture)?;
    let alpha_test =
        vector_object::resolve_alpha_test_with_txt(path, material.alpha_test, read_texture);
    Some((view, alpha_test))
}

fn resolve_texture(
    tex_path: Option<&String>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<wgpu::TextureView> {
    let path = tex_path?;
    if let Some(v) = cache.get(path) {
        return Some(v.clone());
    }
    let data = read_texture(path)?;
    let rgba = decode_texture_bytes(&data)?;
    let view = create_texture_from_rgba(device, queue, &rgba);
    cache.insert(path.clone(), view.clone());
    Some(view)
}

fn create_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Cannons BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    // Reuse the buildings shader — identical vertex+instance layout.
    let shader_src = include_str!("../../shaders/object_building.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Cannons Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Cannons PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });

    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
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
    };
    let instance_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceData>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 64,
                shader_location: 7,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 80,
                shader_location: 8,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 96,
                shader_location: 9,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Cannons Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout, instance_layout],
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
    })
}
