//! Starting-building renderer — ports the draw path of `CMatrixBuilding`
//! (MatrixObjectBuilding.cpp:955-976) plus the group-unit iteration of
//! `CVectorObjectGroup::Draw` (VectorObject.cpp).
//!
//! Each CMAP `buildings/*` row produces one `BuildingInstance`. We group the
//! instances by `kind` (BUILDING_BASE..BUILDING_REPAIR), load the matching
//! `Matrix\Building\bN.cvo` (MatrixObjectBuilding.cpp:158-163), parse it into
//! a list of sub-meshes via `vector_object::parse_cvo`, and create one GPU batch
//! per sub-mesh. Every sub-mesh shares the building's world transform — the
//! original uses a group-wide `m_GroupToWorldMatrix` pointer and each unit
//! composes it with its local animation matrix. We currently draw frame-0 of
//! each sub-VO, which matches the at-rest pose the original ships with.
//!
//! Shadow generation is deferred: `CMatrixBuilding::RNeed` (MatrixObjectBuilding.cpp:
//! 167-246) has the stencil/proj branches fully commented out in the shipped
//! code, so the original skips them for buildings too.

use std::collections::{BTreeMap, HashMap};

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec4};
use wgpu::util::DeviceExt;

use crate::matrix_game::effects::point_light::PointLightSystem;
use crate::matrix_game::common::{unpack_rgb, FOG_END, FOG_START};
use crate::matrix_game::map::{BuildingInstance, GameMap};
use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectId, Objects, ObjectType, MR_ALL,
};
use crate::matrix_game::rnd::Rnd;
use crate::matrix_lib::three_g::vector_object::{self, CvoGroup, MaterialSpec};
use crate::matrix_game::camera::Camera;
use crate::matrix_lib::three_g::texture::{
    create_solid_texture, create_texture_from_rgba, decode_texture_bytes,
};

// ── Game-object side of CMatrixBuilding ─────────────────────────────────
//
// Renderer (BuildingsRenderer, below) is per-type instanced. The `Building`
// game-object carries per-instance logical state (side, kind, state
// machine, hit points, capture progress). Rendering and game-object live
// in the same file to mirror the C++ `MatrixObjectBuilding.{cpp,hpp}`
// layout.
//
// Scope-minimal port: data model + state enums + MapStatic trait impl
// with noop tick bodies. Takt / LogicTakt / Damage / Capture require
// effects / sound / per-side / progress-bar subsystems and land with
// those.

/// Port of `EBuildingType` (MatrixObjectBuilding.hpp:59-69). Discriminants
/// match the C++ so `BuildingInstance::kind` (already parsed from CMAP)
/// can be cast directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BuildingType {
    Base       = 0,
    Titan      = 1,
    Plasma     = 2,
    Electronic = 3,
    Energy     = 4,
    Repair     = 5,
}

impl BuildingType {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => BuildingType::Base,
            1 => BuildingType::Titan,
            2 => BuildingType::Plasma,
            3 => BuildingType::Electronic,
            4 => BuildingType::Energy,
            5 => BuildingType::Repair,
            _ => return None,
        })
    }
}

/// Port of `EBaseState` (MatrixObjectBuilding.hpp:71-82). The ctor seeds
/// `m_State = BASE_CLOSING` (MatrixObjectBuilding.cpp:43) — the base
/// immediately enters its closing animation so it settles at
/// `BASE_CLOSED` on first Takt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseState {
    Closed      = 0,
    Opening     = 1,
    Opened      = 2,
    Closing     = 3,
    /// `BUILDING_DIP` — "dying in progress". Reached after a base is
    /// destroyed; triggers the multi-second explosion sequence.
    Dip         = 4,
    /// `BUILDING_DIP_EXPLODED` — explosion sequence finished; ruin
    /// remains but the building object is otherwise inert.
    DipExploded = 5,
}

/// Default floor height of an opening base before it rises to ground
/// level (MatrixObjectBuilding.hpp:12). Used by the `BASE_OPENING`
/// animation; stored here so the value lives alongside the type
/// definition when the animation code lands.
pub const BASE_FLOOR_Z: f32 = -63.0;

/// Port of `CMatrixBuilding`. Minimal field set for now — the
/// capture / selection / progress-bar / turret-placement state lands
/// with its owning subsystems.
pub struct Building {
    core: ObjectCore,
    rchange: u32,
    object_state: u32,
    ablaze_ttl: i32,
    shorted_ttl: i32,

    /// `m_Pos` — XY in world units (MatrixObjectBuilding.hpp:179). The
    /// `m_Core->m_Matrix` translation is derived from this in `RNeed`.
    pub pos: glam::Vec2,
    /// `m_Angle` — one of 0/1/2/3 for 0/90/180/270 deg
    /// (MatrixObjectBuilding.cpp:120-137).
    pub angle: i32,
    /// `m_Side` — 0=neutral, 1=player, 2-8=AI factions
    /// (MatrixObjectBuilding.hpp:182, MatrixSide.hpp).
    pub side: i32,
    pub kind: BuildingType,
    pub state: BaseState,
    /// `m_BaseFloor` (MatrixObjectBuilding.hpp:204). Controls the
    /// opening / closing animation offset. Ctor default 0.2.
    pub base_floor: f32,
    /// `m_BuildZ` (MatrixObjectBuilding.hpp:205). Terrain floor under
    /// the building; seeded from `BuildingInstance::build_z` at spawn.
    pub build_z: f32,

    /// Hit-point trio (MatrixObjectBuilding.hpp:228-230). Initialised by
    /// `InitMaxHitpoint(hp)`; subsequent `Damage` calls reduce
    /// `hit_point` by table entries and flip to BUILDING_DIP on death.
    pub hit_point: f32,
    pub hit_point_max: f32,
    /// `m_MaxHitPointInversed` — 1 / hit_point_max, precomputed so the
    /// ratio used for progress-bar fill doesn't divide every frame.
    pub max_hit_point_inv: f32,

    /// `m_defHitPoint` (MatrixObjectBuilding.hpp:176). The "default"
    /// hit points loaded from robots.dat per-kind; used when
    /// respawning a captured base with full health.
    pub def_hit_point: i32,

    /// `m_TurretsHave` (MatrixObjectBuilding.hpp:185).
    pub turrets_have: i32,
    /// `m_TurretsMax` (MatrixObjectBuilding.hpp:184). Each building
    /// type carries 4 slots; the ctor sets this to the per-kind
    /// `EBuildingTurrets` enum value.
    pub turrets_max: i32,

    /// `m_UnderAttackTime` (MatrixObjectBuilding.hpp:171). Game-time
    /// ms past which the base is considered "quiet" (no recent
    /// attacker) — controls the attacked-warning UI.
    pub under_attack_time: i32,
    /// `m_CaptureMeNextTime` (MatrixObjectBuilding.hpp:172). Throttle
    /// on capture candidacy announcements.
    pub capture_me_next_time: i32,

    /// `m_ResourcePeriod` (MatrixObjectBuilding.hpp:165, union member).
    /// Ms until the next resource payout; counted down in Takt.
    pub resource_period: i32,

    /// `m_ShadowType` / `m_ShadowSize`
    /// (MatrixObjectBuilding.hpp:240-241). Raw integers from the CMAP
    /// so the renderer can pick the shadow mode without a second
    /// enum-lookup pass.
    pub shadow_type: i32,
    pub shadow_size: i32,

    /// `m_Capturer` (MatrixObjectBuilding.hpp:223). Tracked by
    /// ObjectId so a freed robot reads as `None` through the arena's
    /// tombstone — matches C++ `m_Capturer->m_Object == NULL` check.
    pub capturer: Option<ObjectId>,
}

impl Building {
    /// Port of `CMatrixBuilding` ctor + `OnLoad` (MatrixObjectBuilding.cpp:
    /// 24-69, :1007+). Minimal: seeds type, pos, angle, side, kind, state,
    /// and the derived base_floor / build_z. Robot-spawn / capture /
    /// progress-bar state stays at default (noop).
    pub fn from_instance(inst: &BuildingInstance) -> Self {
        let core = ObjectCore {
            obj_type: ObjectType::Building,
            geo_center: glam::Vec3::new(inst.x, inst.y, inst.build_z),
            matrix: glam::Mat4::from_translation(glam::Vec3::new(
                inst.x, inst.y, inst.build_z,
            )),
            ..Default::default()
        };

        Self {
            core,
            rchange: MR_ALL,                   // m_RChange(0xffffffff)
            object_state: 0,
            ablaze_ttl: 0,
            shorted_ttl: 0,

            pos: glam::Vec2::new(inst.x, inst.y),
            angle: inst.angle as i32,
            side: inst.side as i32,
            kind: BuildingType::from_u8(inst.kind).unwrap_or(BuildingType::Base),
            state: BaseState::Closing,         // MatrixObjectBuilding.cpp:43
            base_floor: 0.2,                   // MatrixObjectBuilding.cpp:42
            build_z: inst.build_z,

            hit_point: 0.0,
            hit_point_max: 0.0,
            max_hit_point_inv: 0.0,

            def_hit_point: 0,                  // MatrixObjectBuilding.cpp:56

            turrets_have: 0,                   // MatrixObjectBuilding.cpp:54
            turrets_max: 4,                    // default per-kind turret cap
            under_attack_time: 0,
            capture_me_next_time: 0,
            resource_period: 0,                // MatrixObjectBuilding.cpp:53

            shadow_type: 0,                    // SHADOW_OFF sentinel; CMAP sets
            shadow_size: 128,                  // MatrixObjectBuilding.cpp:40
            capturer: None,                    // MatrixObjectBuilding.cpp:48
        }
    }

    /// Port of `InitMaxHitpoint(hp)` (MatrixObjectBuilding.hpp:273). Seeds
    /// current + max + inverse together so later divisions don't need
    /// a zero-check.
    pub fn init_max_hitpoint(&mut self, hp: f32) {
        self.hit_point = hp;
        self.hit_point_max = hp;
        self.max_hit_point_inv = if hp != 0.0 { 1.0 / hp } else { 0.0 };
    }

    /// Port of `Open()` (MatrixObjectBuilding.hpp:251-257). Ignored for
    /// DIP/DIP_EXPLODED states; otherwise flips the state machine.
    /// Sound effects deferred.
    pub fn open(&mut self) {
        if matches!(self.state, BaseState::Dip | BaseState::DipExploded) {
            return;
        }
        self.state = BaseState::Opening;
    }

    /// Port of `Close()` (MatrixObjectBuilding.hpp:258-265). Also
    /// guarded by the `BUILDING_SPAWNBOT` flag — can't close while a
    /// robot is spawning.
    pub fn close(&mut self) {
        if matches!(self.state, BaseState::Dip | BaseState::DipExploded) {
            return;
        }
        if self.object_state
            & crate::matrix_game::map_static::OBJECT_STATE_BUILDING_SPAWNBOT != 0
        {
            return;
        }
        self.state = BaseState::Closing;
    }
}

impl MapStatic for Building {
    fn core(&self) -> &ObjectCore { &self.core }
    fn core_mut(&mut self) -> &mut ObjectCore { &mut self.core }
    fn rchange(&self) -> u32 { self.rchange }
    fn rchange_set(&mut self, b: u32) { self.rchange |= b; }
    fn rchange_clear(&mut self, b: u32) { self.rchange &= !b; }
    fn object_state(&self) -> u32 { self.object_state }
    fn object_state_set(&mut self, b: u32) { self.object_state |= b; }
    fn object_state_clear(&mut self, b: u32) { self.object_state &= !b; }
    fn ablaze_ttl(&self) -> i32 { self.ablaze_ttl }
    fn set_ablaze_ttl(&mut self, t: i32) { self.ablaze_ttl = t; }
    fn shorted_ttl(&self) -> i32 { self.shorted_ttl }
    fn set_shorted_ttl(&mut self, t: i32) { self.shorted_ttl = t; }

    /// Port of `CMatrixBuilding::RNeed` (MatrixObjectBuilding.cpp:112-344).
    /// Rebuilds world matrix from `m_Pos` + `m_Angle`, rebuilds the
    /// CVectorObjectGroup mesh from the per-kind CVO, rebuilds shadow
    /// projections. Only the transform bit is trivially portable; the
    /// rest needs the per-instance mesh loader which isn't per-instance
    /// yet in the Rust port.
    fn r_need(&mut self, need: u32) {
        if need & self.rchange & crate::matrix_game::map_static::MR_MATRIX != 0 {
            self.rchange &= !crate::matrix_game::map_static::MR_MATRIX;
            // Angle is one of 0..3 (MatrixObjectBuilding.cpp:120-137).
            // Each increment adds 90°. The `m_Core->m_Matrix` is
            // translation × Rz(angle).
            let ang = (self.angle & 3) as f32 * std::f32::consts::FRAC_PI_2;
            let (s, c) = ang.sin_cos();
            self.core.matrix = glam::Mat4::from_cols(
                glam::Vec4::new(c, s, 0.0, 0.0),
                glam::Vec4::new(-s, c, 0.0, 0.0),
                glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
                glam::Vec4::new(self.pos.x, self.pos.y, self.build_z, 1.0),
            );
            self.core.inv_matrix = self.core.matrix.inverse();
        }
        // MR_GRAPH / MR_SHADOW_* / MR_MINIMAP — clear bits so the
        // next r_need doesn't spin. Real rebuilds will land with the
        // per-instance mesh / shadow manager.
        self.rchange &= !(crate::matrix_game::map_static::MR_GRAPH
            | crate::matrix_game::map_static::MR_SHADOW_PROJ_GEOM
            | crate::matrix_game::map_static::MR_SHADOW_PROJ_TEX
            | crate::matrix_game::map_static::MR_SHADOW_STENCIL
            | crate::matrix_game::map_static::MR_MINIMAP
        );
        let _ = need;
    }

    /// Port of `CMatrixBuilding::Takt` (MatrixObjectBuilding.cpp:351-439).
    /// Noop skeleton: BUILDING_NEW_INCOME → score-billboard (effects),
    /// `m_GGraph->Takt` (per-instance mesh anim), `m_capture` overlay
    /// creation (effects) — every branch needs subsystems not yet
    /// ported. Structure preserved so the body lands in-place later.
    fn takt(&mut self, _cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {}

    /// Port of `CMatrixBuilding::LogicTakt` (MatrixObjectBuilding.cpp:495-899).
    /// State-machine driver for BASE_OPENING → OPENED (≈2.5s), OPENED
    /// → CLOSED, DIP explosion cycle, capture progress, resource
    /// payouts, turret spawning, ... all require effects / sound /
    /// sides / per-side resources. Skeleton here — the real body
    /// arrives with those subsystems.
    fn logic_takt(&mut self, _cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {}

    fn side(&self) -> i32 { self.side }
    fn need_repair(&self) -> bool { self.hit_point < self.hit_point_max }
}

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

struct MeshBatch {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instance_buffer: wgpu::Buffer,
    num_instances: u32,
    bind_group: wgpu::BindGroup,
    /// Buildings feeding this batch — kept so the instance transforms can be
    /// re-uploaded when point-light colors change (matches the terrain-tint
    /// refresh the object renderer does in its `takt`).
    buildings: Vec<BuildingInstance>,
    center: [f32; 2],
}

pub struct BuildingsRenderer {
    pipeline: wgpu::RenderPipeline,
    batches: Vec<MeshBatch>,
    uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    time_ms: f32,
    last_point_light_revision: u64,
}

impl BuildingsRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        if map.buildings.is_empty() {
            return None;
        }

        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;

        let mut by_kind: BTreeMap<u8, Vec<&BuildingInstance>> = BTreeMap::new();
        for b in &map.buildings {
            by_kind.entry(b.kind).or_default().push(b);
        }

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
            label: Some("Buildings UB"),
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

        let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
        let fallback_tex = create_solid_texture(device, queue, [200, 200, 200, 255]);
        let black_tex = create_solid_texture(device, queue, [0, 0, 0, 255]);
        let transparent_tex = create_solid_texture(device, queue, [0, 0, 0, 0]);

        let mut batches = Vec::new();
        let mut loaded_kinds = 0usize;
        let mut missing_kinds = 0usize;

        for (kind, instances) in &by_kind {
            let cvo_path = match building_cvo_path(*kind) {
                Some(p) => p,
                None => {
                    missing_kinds += 1;
                    continue;
                }
            };
            let Some(cvo_bytes) = read_texture(&cvo_path) else {
                log::warn!("buildings: CVO not found: {}", cvo_path);
                missing_kinds += 1;
                continue;
            };
            let group: CvoGroup = vector_object::parse_cvo(&cvo_path, &cvo_bytes);
            if group.units.is_empty() {
                log::warn!("buildings: CVO has no units: {}", cvo_path);
                missing_kinds += 1;
                continue;
            }

            let inst_data: Vec<InstanceData> = instances
                .iter()
                .map(|b| instance_matrix(b, cx, cy, map, None))
                .collect();
            let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Buildings Inst VB"),
                contents: bytemuck::cast_slice(&inst_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

            for unit in &group.units {
                let Some(vo_bytes) = read_texture(&unit.model_path) else {
                    log::debug!("buildings: sub-VO not found: {}", unit.model_path);
                    continue;
                };
                let mesh = match vector_object::parse_vo(&vo_bytes) {
                    Ok(m) => m,
                    Err(e) => {
                        log::warn!("buildings: parse {} failed: {}", unit.model_path, e);
                        continue;
                    }
                };

                let vertices: Vec<Vertex> = mesh
                    .vertices
                    .iter()
                    .map(|v| Vertex {
                        position: v.position,
                        normal: v.normal,
                        uv: v.uv,
                    })
                    .collect();

                // Use frame 0's surface partition — the original's at-rest
                // pose (CVectorObjectAnim defaults to anim 0, frame 0).
                let Some(frame0) = mesh.frames.first() else {
                    continue;
                };

                // When the CVO unit declares no `Texture=…`, the original
                // falls back to the VO's own embedded surface texture name —
                // `vo->GetSurfaceFileName(0)` (VectorObject.cpp:2509). We
                // read that from the surface's `texture_ref` instead, and
                // otherwise keep the unit's declared diffuse. Gloss/mask/back
                // overrides from the CVO still win, mirroring the composed
                // skin the original builds at VectorObject.cpp:2513.
                let cvo_dir = cvo_path
                    .rsplit_once('/')
                    .map(|(d, _)| format!("{d}/"));
                for surf in &frame0.surfaces {
                    if surf.indices.is_empty() {
                        continue;
                    }

                    let material = if unit.material.diffuse.is_some() {
                        unit.material.clone()
                    } else if let Some(spec) = surf.texture_ref.as_deref() {
                        let surface_mat = vector_object::parse_material_spec_with_prefix(
                            spec,
                            cvo_dir.as_deref(),
                        );
                        vector_object::merge_materials(&surface_mat, Some(&unit.material))
                    } else {
                        unit.material.clone()
                    };

                    let (diffuse_view, alpha_test) =
                        resolve_diffuse(&material, device, queue, &mut tex_cache, read_texture)
                            .unwrap_or_else(|| (fallback_tex.clone(), false));
                    let gloss_view = resolve_texture(
                        material.gloss.as_ref(),
                        device,
                        queue,
                        &mut tex_cache,
                        read_texture,
                    )
                    .unwrap_or_else(|| black_tex.clone());
                    let back_view = resolve_texture(
                        material.back.as_ref(),
                        device,
                        queue,
                        &mut tex_cache,
                        read_texture,
                    )
                    .unwrap_or_else(|| black_tex.clone());
                    let mask_view = resolve_texture(
                        material.mask.as_ref(),
                        device,
                        queue,
                        &mut tex_cache,
                        read_texture,
                    )
                    .unwrap_or_else(|| transparent_tex.clone());

                    let vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Buildings Mesh VB"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Buildings Mesh IB"),
                        contents: bytemuck::cast_slice(&surf.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    let mat_uniform =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Buildings Material UB"),
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
                        label: Some("Buildings BG"),
                        layout: &bgl,
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
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: mat_uniform.as_entire_binding(),
                            },
                        ],
                    });

                    batches.push(MeshBatch {
                        vertex_buffer,
                        index_buffer,
                        num_indices: surf.indices.len() as u32,
                        instance_buffer: instance_buffer.clone(),
                        num_instances: inst_data.len() as u32,
                        bind_group,
                        buildings: instances.iter().map(|b| (*b).clone()).collect(),
                        center: [cx, cy],
                    });
                }
            }

            loaded_kinds += 1;
        }

        log::info!(
            "buildings: {} kinds loaded, {} skipped, {} total draw batches ({} instances)",
            loaded_kinds,
            missing_kinds,
            batches.len(),
            batches.iter().map(|b| b.num_instances).sum::<u32>(),
        );

        if batches.is_empty() {
            return None;
        }

        Some(Self {
            pipeline,
            batches,
            uniform_buffer,
            fog_color,
            ambient_color,
            light_color,
            light_dir,
            time_ms: 0.0,
            last_point_light_revision: 0,
        })
    }

    pub fn takt(
        &mut self,
        dt_ms: f32,
        queue: &wgpu::Queue,
        map: &GameMap,
        point_lights: &PointLightSystem,
    ) {
        self.time_ms += dt_ms;
        if self.time_ms > 1_000_000.0 {
            self.time_ms -= 1_000_000.0;
        }

        let revision = point_lights.revision();
        if revision != self.last_point_light_revision {
            for batch in &mut self.batches {
                let [cx, cy] = batch.center;
                let inst_data: Vec<InstanceData> = batch
                    .buildings
                    .iter()
                    .map(|b| instance_matrix(b, cx, cy, map, Some(point_lights)))
                    .collect();
                queue.write_buffer(&batch.instance_buffer, 0, bytemuck::cast_slice(&inst_data));
            }
            self.last_point_light_revision = revision;
        }
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

        pass.set_pipeline(&self.pipeline);
        for batch in &self.batches {
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, batch.instance_buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..batch.num_indices, 0, 0..batch.num_instances);
        }
    }
}

/// Maps `EBuildingType` (MatrixObjectBuilding.hpp:59-69) to the archive path
/// the original engine feeds into `CVectorObjectGroup::Load`
/// (MatrixObjectBuilding.cpp:158-163). Unknown kinds return `None`.
fn building_cvo_path(kind: u8) -> Option<String> {
    let name = match kind {
        0 => "b0", // BUILDING_BASE
        1 => "b1", // BUILDING_TITAN
        2 => "b2", // BUILDING_PLASMA
        3 => "b3", // BUILDING_ELECTRONIC
        4 => "b4", // BUILDING_ENERGY
        5 => "b5", // BUILDING_REPAIR
        _ => return None,
    };
    Some(format!("Matrix/Building/{name}.cvo"))
}

/// Build the per-building instance transform — ports
/// `CMatrixBuilding::RNeed` (MatrixObjectBuilding.cpp:115-148).
/// The original matrix is row-major: `[rot | 0; pos | 1]`; in column-major
/// glam we compose `translate(pos) * rotZ(angle*90°)` and flatten into the
/// row-layout instance the shared shader expects.
fn instance_matrix(
    b: &BuildingInstance,
    cx: f32,
    cy: f32,
    map: &GameMap,
    point_lights: Option<&PointLightSystem>,
) -> InstanceData {
    let theta = (b.angle as f32) * std::f32::consts::FRAC_PI_2;
    let (s, c) = theta.sin_cos();
    let rot = Mat3::from_cols(
        Vec4::new(c, s, 0.0, 0.0).truncate(),
        Vec4::new(-s, c, 0.0, 0.0).truncate(),
        Vec4::new(0.0, 0.0, 1.0, 0.0).truncate(),
    );
    let [terrain_r, terrain_g, terrain_b] =
        unpack_rgb(map.static_object_color_with_lighting(b.x, b.y, point_lights));
    InstanceData {
        row0: [rot.x_axis.x, rot.y_axis.x, rot.z_axis.x, b.x - cx],
        row1: [rot.x_axis.y, rot.y_axis.y, rot.z_axis.y, b.y - cy],
        row2: [rot.x_axis.z, rot.y_axis.z, rot.z_axis.z, b.build_z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [terrain_r, terrain_g, terrain_b, 1.0],
    }
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
        label: Some("Buildings BGL"),
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
                visibility: wgpu::ShaderStages::FRAGMENT,
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
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Buildings Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Buildings PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Buildings Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[
                wgpu::VertexBufferLayout {
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
                },
                wgpu::VertexBufferLayout {
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

const SHADER: &str = include_str!("../../shaders/object_building.wgsl");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_game::common::{TRACE_BUILDING, TRACE_ROBOT};
    use crate::matrix_game::map_static::{MapStatic, Objects, OBJECT_STATE_BUILDING_SPAWNBOT};

    fn inst(kind: u8, side: u8) -> BuildingInstance {
        BuildingInstance {
            x: 100.0, y: 100.0, build_z: 0.0,
            angle: 0, side, kind,
            shadow_kind: 0, shadow_size: 128,
        }
    }

    #[test]
    fn from_instance_carries_pos_side_kind_and_initial_state() {
        let b = Building::from_instance(&inst(0, 1));
        assert_eq!(b.pos, glam::Vec2::new(100.0, 100.0));
        assert_eq!(b.side, 1);
        assert_eq!(b.kind, BuildingType::Base);
        assert_eq!(b.state, BaseState::Closing);
        assert_eq!(b.base_floor, 0.2);
        assert_eq!(b.shadow_size, 128);
    }

    #[test]
    fn init_max_hitpoint_seeds_all_three() {
        let mut b = Building::from_instance(&inst(0, 1));
        b.init_max_hitpoint(400.0);
        assert_eq!(b.hit_point, 400.0);
        assert_eq!(b.hit_point_max, 400.0);
        assert!((b.max_hit_point_inv - 0.0025).abs() < 1e-9);
        // Zero HP → inverse is 0 instead of NaN/inf.
        b.init_max_hitpoint(0.0);
        assert_eq!(b.max_hit_point_inv, 0.0);
    }

    #[test]
    fn open_and_close_respect_dip_and_spawnbot_guards() {
        // Open/Close ignored in DIP / DIP_EXPLODED states.
        let mut b = Building::from_instance(&inst(0, 1));
        b.state = BaseState::Dip;
        b.open();
        assert_eq!(b.state, BaseState::Dip);
        b.close();
        assert_eq!(b.state, BaseState::Dip);

        // Close blocked while a robot is spawning
        // (MatrixObjectBuilding.hpp:261).
        b.state = BaseState::Opened;
        b.object_state_set(OBJECT_STATE_BUILDING_SPAWNBOT);
        b.close();
        assert_eq!(b.state, BaseState::Opened, "close blocked by SPAWNBOT");

        b.object_state_clear(OBJECT_STATE_BUILDING_SPAWNBOT);
        b.close();
        assert_eq!(b.state, BaseState::Closing);
        b.open();
        assert_eq!(b.state, BaseState::Opening);
    }

    #[test]
    fn building_fits_trace_building_mask_and_shows_up_in_find_objects() {
        // The whole point of pulling buildings into the arena:
        // FindObjects(TRACE_BUILDING) starts returning them.
        let mut objs = Objects::new();
        let b = Building::from_instance(&inst(0, 1));
        let id = objs.spawn(Box::new(b));
        assert!(objs.any_object_in_radius(
            glam::Vec2::new(100.0, 100.0), 50.0, 1.0, TRACE_BUILDING, None,
        ));
        // And doesn't match unrelated masks.
        assert!(!objs.any_object_in_radius(
            glam::Vec2::new(100.0, 100.0), 50.0, 1.0, TRACE_ROBOT, None,
        ));
        let _ = id;
    }
}