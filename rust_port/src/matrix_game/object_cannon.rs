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
use crate::matrix_game::logic::Rnd;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectId, ObjectType, Objects, MR_ALL, MR_MATRIX,
};
use crate::matrix_game::shadow::{ShadowBatch, ShadowMeshSurface, ShadowMeshVertex, ShadowSystem};
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

    /// Own arena id — set right after spawn (same pattern as
    /// `Building::self_id`). Needed to hand the weapons a handler.
    pub self_id: Option<ObjectId>,
    /// DIP wreck pieces (`m_Unit[]` in CANNON_DIP mode) — the basis
    /// stays put, turret + shaft tumble away (MatrixObjectCannon.cpp:
    /// 1501-1539, DIPTakt :670-775). The cannon frees itself when all
    /// pieces are gone.
    pub dip_units: Vec<CannonDipUnit>,
    /// `m_CurrState` (MatrixObjectCannon.hpp).
    pub state: CannonState,
    /// `m_Invulnerability` flag — port of `SetInvulnerability`
    /// (MatrixObjectCannon.cpp:1486+ / MatrixSide.cpp:631). Damage paths
    /// must early-return while this is true.
    pub invulnerable: bool,
    /// `m_ShowHitpointTime` — floating HP-bar visibility timer.
    pub show_hitpoint_time: i32,

    // ── Combat state (MatrixObjectCannon.hpp:120-160) ────────────────
    /// `m_Weapons[]` — created lazily on the first live logic takt
    /// (the C++ creates them in `RNeed(MR_Graph)` when the shaft mesh
    /// loads; we don't carry mesh data on the game object).
    pub weapons: Vec<crate::matrix_game::effects::weapon::WeaponId>,
    /// `m_AngleX` — barrel elevation.
    pub angle_x: f32,
    /// `m_Unit[1].m_Angle` — turret yaw relative to the base `angle`.
    pub turret_angle: f32,
    /// `m_TargetCore` — currently tracked target.
    pub target: Option<ObjectId>,
    /// `m_TargetDisp` — aim-miss displacement while shooting "wide".
    pub target_disp: glam::Vec3,
    /// `m_FireNextThinkTime` / `m_NullTargetTime` / `m_TimeFromFire`.
    pub fire_next_think_time: i64,
    pub null_target_time: i32,
    pub time_from_fire: i32,
    /// DOT bookkeeping (`m_NextTimeAblaze` / `m_NextTimeShorted` /
    /// `m_LastDelayDamageSide`).
    pub next_time_ablaze: i64,
    pub next_time_shorted: i64,
    pub last_delay_damage_side: i32,
}

/// Aiming/fire timing constants (MatrixObjectCannon.hpp:14-18).
pub const CANNON_FIRE_THINK_PERIOD: i32 = 100;
pub const CANNON_NULL_TARGET_TIME: i32 = 1000;
pub const CANNON_TIME_FROM_FIRE: i32 = 1000;
/// `CANNNON_MIN_DANGLE` = GRAD2RAD(2).
pub const CANNON_MIN_DANGLE: f32 = 2.0 * std::f32::consts::PI / 180.0;

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
        let k = ((kind - 1).max(0) as usize).min(3);
        let core = ObjectCore {
            obj_type: ObjectType::Cannon,
            // Mesh-AABB center like the C++ `JoinToGroup` — mid-height,
            // not the base plate. Robots aim here; the HP bar anchors
            // here.
            geo_center: glam::Vec3::new(pos.x, pos.y, pos_z + CANNON_GEO_Z[k]),
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
            self_id: None,
            dip_units: Vec::new(),
            state: CannonState::Idle,
            invulnerable: false,
            show_hitpoint_time: 0,
            weapons: Vec::new(),
            angle_x: 0.0,
            turret_angle: 0.0,
            target: None,
            target_disp: glam::Vec3::ZERO,
            fire_next_think_time: 0,
            null_target_time: 0,
            time_from_fire: 0,
            next_time_ablaze: 0,
            next_time_shorted: 0,
            last_delay_damage_side: 0,
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

    /// `GetStrength` (MatrixObjectCannon.cpp:22-25).
    pub fn get_strength(&self) -> f32 {
        let props = crate::matrix_game::config::global().turrets.cannons
            [((self.kind - 1).max(0) as usize).min(3)];
        props.strength * (0.4 + 0.6 * (self.hit_point / self.hit_point_max.max(1.0)))
    }

    /// `GetFireRadius` (MatrixObjectCannon.hpp:163).
    pub fn fire_radius(&self, objs: &Objects) -> f32 {
        self.weapons
            .first()
            .and_then(|&w| objs.weapons.get(w))
            .map(|w| w.weapon_dist())
            .unwrap_or(0.0)
    }

    /// World yaw of the barrels — `m_Unit[1].m_Angle + m_Angle`.
    fn world_yaw(&self) -> f32 {
        self.turret_angle + self.angle
    }

    /// Barrel pivot (`m_FireCenter`) — the shaft mount run through the
    /// cannon's base rotation.
    fn fire_center(&self) -> Vec3 {
        let p2 = SHAFT_PIVOT[((self.kind - 1).max(0) as usize).min(3)];
        let (sa, ca) = self.angle.sin_cos();
        Vec3::new(
            p2.x * ca - p2.y * sa + self.pos.x,
            p2.x * sa + p2.y * ca + self.pos.y,
            p2.z + self.pos_z,
        )
    }

    /// World position of a shaft-space point — the row-vector chain
    /// `v · rotX(angle_x) · rotZ(turret) + pivot, then · rotZ(angle) +
    /// pos` of `CMatrixCannon::RNeed` (MatrixObjectCannon.cpp:282-310).
    fn shaft_point_world(&self, bone: Vec3) -> Vec3 {
        let k = ((self.kind - 1).max(0) as usize).min(3);
        let (sx, cx) = self.angle_x.sin_cos();
        let v = Vec3::new(bone.x, bone.y * cx - bone.z * sx, bone.y * sx + bone.z * cx);
        let (st, ct) = self.turret_angle.sin_cos();
        let v = Vec3::new(v.x * ct - v.y * st, v.x * st + v.y * ct, v.z) + SHAFT_PIVOT[k];
        let (sa, ca) = self.angle.sin_cos();
        Vec3::new(
            v.x * ca - v.y * sa + self.pos.x,
            v.x * sa + v.y * ca + self.pos.y,
            v.z + self.pos_z,
        )
    }

    /// Muzzle of barrel `i` — its fire bone through the chain.
    fn fire_from_barrel(&self, i: usize) -> Vec3 {
        let k = ((self.kind - 1).max(0) as usize).min(3);
        self.shaft_point_world(FIRE_BONES[k][i.min(1)])
    }

    /// Barrel direction from yaw + elevation — equals the Y axis of
    /// the fire matrix the C++ reads (`weapm._21`).
    fn fire_dir_now(&self) -> Vec3 {
        let (sy, cy) = self.world_yaw().sin_cos();
        let (sx, cx) = self.angle_x.sin_cos();
        Vec3::new(-cx * sy, cx * cy, sx)
    }

    /// Create the weapon effects once the cannon is live. The C++ does
    /// this in `RNeed(MR_Graph)` per fire matrix (ids 50-59) of the
    /// shaft mesh (MatrixObjectCannon.cpp:233-247); barrel counts per
    /// model are baked here since the game object carries no mesh.
    fn ensure_weapons(&mut self, objs: &mut Objects) {
        if !self.weapons.is_empty() || self.self_id.is_none() {
            return;
        }
        let self_id = self.self_id.unwrap();
        let props = crate::matrix_game::config::global().turrets.cannons
            [((self.kind - 1).max(0) as usize).min(3)];
        if props.weapon == crate::matrix_game::effects::weapon::WEAPON_NONE {
            return;
        }
        let barrels = FIRE_BARRELS[((self.kind - 1).max(0) as usize).min(3)];
        for _ in 0..barrels {
            let mut w = crate::matrix_game::effects::weapon::WeaponEffect::new(
                props.weapon,
                0,
                crate::matrix_game::effects::weapon::WeaponHandler::Cannon(self_id),
            );
            w.set_owner(self_id, self.side);
            self.weapons.push(objs.weapons.create(w));
        }
    }

    /// Shared "run the weapon logic" block (MatrixObjectCannon.cpp:
    /// 1099-1124 and :1316-1340): tick each weapon, collect `IsFireWas`,
    /// refresh `m_TimeFromFire` on hits. The `OBJECT_CANNON_REF_
    /// PROTECTION` ref-count dance is unnecessary with the weapon slab.
    fn weapons_logic(
        &mut self,
        takt: i32,
        map: &GameMap,
        objs: &mut Objects,
        rng: &mut Rnd,
    ) -> bool {
        let mut firewas = false;
        for &w in &self.weapons {
            if let Some(we) = objs.weapons.get_mut(w) {
                we.reset_fire_count();
            }
            crate::matrix_game::effects::weapon::weapon_takt(objs, w, takt as f32, map, rng);
            if let Some(we) = objs.weapons.get_mut(w) {
                firewas |= we.is_fire_was();
                if we.is_hit_was() {
                    self.time_from_fire = CANNON_TIME_FROM_FIRE;
                }
            }
        }
        firewas
    }

    /// `FindTarget` (MatrixObjectCannon.cpp:789-821) — pick the enemy
    /// unit most aligned with the current barrel direction, preferring
    /// anything inside actual fire range, with a two-stage
    /// line-of-sight check.
    fn seek_target(&self, map: &GameMap, objs: &Objects) -> Option<ObjectId> {
        use crate::matrix_game::common::{
            TRACE_BUILDING, TRACE_FLYER, TRACE_LANDSCAPE, TRACE_OBJECT, TRACE_OBJECTSPHERE,
            TRACE_ROBOT,
        };
        use crate::matrix_game::map_static::Control;
        use crate::matrix_game::map_trace::{trace, TraceStop};

        let props = crate::matrix_game::config::global().turrets.cannons
            [((self.kind - 1).max(0) as usize).min(3)];
        let center = self.core.geo_center;
        let cdir = self.fire_dir_now();
        let dist_fire = {
            let f = self.fire_radius(objs);
            f * f
        };
        let mut dist_cur = props.seek_radius * props.seek_radius;
        let mut coss = -1.0f32;
        let mut target: Option<ObjectId> = None;

        let mut cands: Vec<(ObjectId, Vec3, i32)> = Vec::new();
        objs.find_objects_3d(
            center,
            props.seek_radius,
            1.0,
            TRACE_ROBOT | TRACE_FLYER,
            None,
            |_, id| {
                if let Some(o) = objs.get(id) {
                    cands.push((id, o.core().geo_center, o.side()));
                }
                Control::Continue
            },
        );
        for (id, geo, side) in cands {
            if side == self.side {
                continue;
            }
            let dir = geo - center;
            let distc = dir.length_squared();
            if distc > dist_fire && dist_cur < dist_fire {
                continue;
            }
            let matches = distc < dist_fire && dist_cur > dist_fire;
            let dot = dir.normalize_or_zero().dot(cdir);
            if matches || dot > coss {
                let (t1, _) = trace(
                    map,
                    objs,
                    center,
                    geo,
                    TRACE_OBJECTSPHERE | TRACE_ROBOT | TRACE_FLYER | TRACE_LANDSCAPE,
                    self.self_id,
                );
                if t1.object() == Some(id) {
                    let (t2, _) = trace(
                        map,
                        objs,
                        center,
                        geo,
                        TRACE_BUILDING | crate::matrix_game::common::TRACE_CANNON | TRACE_OBJECT,
                        self.self_id,
                    );
                    if t2 == TraceStop::None {
                        dist_cur = distc;
                        coss = dot;
                        target = Some(id);
                    }
                }
            }
        }
        target
    }
}

/// Fire matrices (bone ids 50-59) per shaft model — extracted from
/// `Matrix\Cannon\Shaft{1..4}.vo` (see examples/probe_cannon_barrels.rs).
/// Shaft1 + Shaft4 are twin-barrel; 2 + 3 single.
const FIRE_BARRELS: [usize; 4] = [2, 1, 1, 2];
/// Shaft-space fire-bone translations per kind (same probe); twin
/// barrels carry two entries, singles repeat theirs.
const FIRE_BONES: [[Vec3; 2]; 4] = [
    [
        Vec3::new(-3.27, 20.51, -0.35),
        Vec3::new(3.81, 20.82, -0.38),
    ],
    [
        Vec3::new(-0.11, 35.48, -0.76),
        Vec3::new(-0.11, 35.48, -0.76),
    ],
    [Vec3::new(-0.61, 24.66, 8.03), Vec3::new(-0.61, 24.66, 8.03)],
    [Vec3::new(-7.02, 21.88, 3.63), Vec3::new(6.96, 21.88, 3.63)],
];
/// Shaft pivot offset in cannon space — `Basis.matrix(20)` +
/// `Turret{n}.matrix(20)` translations (the mount chain of
/// `CMatrixCannon::RNeed`, MatrixObjectCannon.cpp:282-329; probed via
/// examples/probe_cannon_barrels.rs companions).
const SHAFT_PIVOT: [Vec3; 4] = [
    Vec3::new(-0.113, -1.509, 26.751),
    Vec3::new(-0.838, 7.652, 20.588),
    Vec3::new(0.058, -1.861, 23.593),
    Vec3::new(-0.113, -10.955, 22.409),
];
/// `Basis.vo` matrix-20 translation — the turret unit's mount in
/// cannon space (probed via examples/probe_basis_mount.rs).
/// `SHAFT_PIVOT[k] = BASIS_MOUNT + Turret{k+1}.matrix(20)`.
const BASIS_MOUNT: Vec3 = Vec3::new(-0.113, 0.103, 11.446);

/// Mesh-AABB center height + half-diagonal per kind — the C++ derives
/// `m_GeoCenter` / `m_Radius` from `CalcBounds` over Basis+Turret+Shaft
/// (`JoinToGroup`, MatrixMapStatic.cpp:160-178); probed via
/// examples/probe_cannon_bounds.rs. The trace sphere keeps the tighter
/// hand-tuned 20 (the AABB half-diagonal over-covers as a *final* hit
/// test, which is all our sphere pick is), but the HP bar lifts by the
/// real mesh radius like the original.
const CANNON_GEO_Z: [f32; 4] = [14.6, 11.9, 16.1, 12.7];
const CANNON_MESH_RADIUS: [f32; 4] = [37.3, 43.9, 39.1, 35.3];

/// Which of the cannon's sub-meshes a [`CannonDipUnit`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannonDipPart {
    Basis,
    Turret,
    Shaft,
}

/// One cannon wreck piece (see `Cannon::dip_units`).
pub struct CannonDipUnit {
    pub pos: Vec3,
    pub velocity: Vec3,
    pub ttl: f32,
    pub dp: f32,
    pub dy: f32,
    pub dr: f32,
    pub part: CannonDipPart,
    /// Base Z-rotation — keeps the basis at the cannon's yaw (its
    /// unit matrix is never recomputed; flying pieces snap to the
    /// pure spin like the robot wrecks).
    pub yaw: f32,
    /// Spin-clock freeze stamp, set when the piece lands.
    pub freeze_t: Option<f32>,
    pub smoke: crate::matrix_game::effects::smoke_and_fire::Smoke,
}

/// `DistOtrezokPoint` — distance from segment `[a, b]` to point `p`.
fn dist_segment_point(a: Vec3, b: Vec3, p: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1.0e-12)).clamp(0.0, 1.0);
    (a + ab * t - p).length()
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
    fn takt(&mut self, cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {
        // HP-bar timer (`PBTakt`, MatrixObjectCannon.cpp:855-868):
        // construction keeps the bar pinned on so the HP ramp shows.
        if self.state == CannonState::UnderConstruction {
            self.show_hitpoint_time = 1;
        } else if self.show_hitpoint_time > 0 {
            self.show_hitpoint_time = (self.show_hitpoint_time - cms).max(0);
        }
    }

    fn show_hitpoint(&mut self) {
        self.show_hitpoint_time = crate::matrix_game::common::HITPOINT_SHOW_TIME_MS;
    }

    /// `CMatrixCannon::BeforeDraw` PB block (MatrixObjectCannon.cpp:
    /// 502-510): anchor at the geo center (mesh mid-height), lifted by
    /// the mesh radius, PB_CANNON_WIDTH=100.
    fn hitpoint_bar(
        &self,
        _map: &crate::matrix_game::map::GameMap,
    ) -> Option<crate::matrix_game::map_static::HpBar> {
        if self.show_hitpoint_time <= 0 || self.hit_point <= 0.0 || self.state == CannonState::Dip
        {
            return None;
        }
        let k = ((self.kind - 1).max(0) as usize).min(3);
        Some(crate::matrix_game::map_static::HpBar {
            anchor: self.core.geo_center,
            width: 100.0,
            fill: (self.hit_point / self.hit_point_max.max(1.0)).clamp(0.0, 1.0),
            x_off: -50.0,
            y_off: -CANNON_MESH_RADIUS[k],
        })
    }

    /// Port of `CMatrixCannon::LogicTakt` (MatrixObjectCannon.cpp:
    /// 872-1345): DIP / construction gates, ablaze + shorted DOT
    /// loops, target think, turret aiming, fire control.
    fn logic_takt(&mut self, takt: i32, rng: &mut Rnd, objs: &mut Objects) {
        use crate::matrix_game::effects::weapon::{WEAPON_ABLAZE, WEAPON_SHORTED};
        use crate::matrix_game::map_static::{OBJECT_STATE_ABLAZE, OBJECT_STATE_SHORTED};
        use crate::matrix_game::map_trace::trace;

        let Some(map) = crate::matrix_game::map::current_map() else {
            return;
        };
        let now = crate::matrix_game::map::current_elapsed_ms();
        let Some(self_id) = self.self_id else { return };

        if self.state == CannonState::Dip {
            // DIPTakt (MatrixObjectCannon.cpp:670-775) — same piece
            // physics as the robot wrecks; end-of-life boom pops a
            // radius above the piece.
            self.dip_takt(takt, map, objs, rng);
            if self.dip_units.is_empty() {
                objs.remove_deferred(self_id);
            }
            return;
        }
        if self.state == CannonState::UnderConstruction {
            return;
        }
        // Parent-building side sync (MatrixObjectCannon.cpp:887-898,
        // minus the per-robot env scrub — env isn't ported).
        if let Some(parent) = self.parent {
            if let Some(p) = objs.get(parent) {
                let pside = p.side();
                if pside != self.side {
                    self.side = pside;
                }
            }
        }

        self.ensure_weapons(objs);

        // Ablaze DOT (MatrixObjectCannon.cpp:929-957). The fire
        // visuals are skipped; the WEAPON_ABLAZE self-damage every
        // OBJECT_ROBOT_ABLAZE_PERIOD within each 90ms window is the
        // mechanic.
        if self.object_state & OBJECT_STATE_ABLAZE != 0 {
            while now > self.next_time_ablaze {
                self.next_time_ablaze += 90; // OBJECT_ROBOT_ABLAZE_PERIOD_EFFECT
                let pos = self.core.geo_center;
                let side = self.last_delay_damage_side;
                {
                    use crate::matrix_game::effects::smoke_and_fire::{fsrnd, Fire};
                    let mut vrng = crate::matrix_game::logic::Rnd::new(
                        ((now as i32) ^ ((pos.x + pos.y) as i32)).max(1),
                    );
                    let r = self.core.radius;
                    let fp = pos + glam::Vec3::new(fsrnd(&mut vrng, r), fsrnd(&mut vrng, r), 0.0);
                    objs.pending_effects
                        .push(crate::matrix_game::effects::GameEffect::Fire(Fire::new(
                            fp, 100.0, 2500.0, 10.0, 2.5, false, 0.04,
                        )));
                }
                for _ in (0..90).step_by(10) {
                    if self.damage(WEAPON_ABLAZE, pos, glam::Vec3::Z, side, None, self_id, objs) {
                        return;
                    }
                    if self.state == CannonState::Dip {
                        return;
                    }
                }
            }
        }
        // Shorted DOT (MatrixObjectCannon.cpp:958-997) — and a shorted
        // cannon does nothing else this takt (the C++ `return` at :996).
        if self.object_state & OBJECT_STATE_SHORTED != 0 {
            while now > self.next_time_shorted {
                self.next_time_shorted += 50; // OBJECT_SHORTED_PERIOD
                let pos = self.core.geo_center;
                let side = self.last_delay_damage_side;
                if self.damage(
                    WEAPON_SHORTED,
                    pos,
                    glam::Vec3::Z,
                    side,
                    None,
                    self_id,
                    objs,
                ) {
                    return;
                }
                if self.state == CannonState::Dip {
                    return;
                }
            }
            return;
        }

        let props = crate::matrix_game::config::global().turrets.cannons
            [((self.kind - 1).max(0) as usize).min(3)];

        // Target think every CANNON_FIRE_THINK_PERIOD ms.
        let delta = self.fire_next_think_time - now;
        let itstime = delta < 0 || delta > CANNON_FIRE_THINK_PERIOD as i64;
        if itstime {
            self.fire_next_think_time = now + CANNON_FIRE_THINK_PERIOD as i64;
            self.target_disp = glam::Vec3::ZERO;
            self.target = self.seek_target(map, objs);
        }
        // Tombstone check between thinks.
        if let Some(t) = self.target {
            if !objs.is_valid(t) {
                self.target = None;
            }
        }

        // Aiming + final checks. `aimed_and_in_range` collapses the
        // C++ `goto no_target` web: every bail lands in the shared
        // "keep ticking the weapons" tail below.
        let mut open_fire = false;
        if let Some(target_id) = self.target {
            let tgtpos = objs
                .get(target_id)
                .map(|o| o.core().geo_center)
                .unwrap_or(self.core.geo_center)
                + self.target_disp;

            let fire_center = self.fire_center();
            let mul = if props.max_da == 0.0 {
                1.0 - 0.995f32.powi(takt)
            } else {
                0.0
            };

            // Yaw (MatrixObjectCannon.cpp:1145-1176).
            let dirx = -(tgtpos.x - fire_center.x);
            let diry = tgtpos.y - fire_center.y;
            let dang = dirx.atan2(diry);
            let cang = self.turret_angle + self.angle;
            let da = angle_dist(cang, dang);
            let mut matchz = false;
            if props.max_da == 0.0 {
                if da.abs() < CANNON_MIN_DANGLE {
                    matchz = true;
                    self.turret_angle = dang - self.angle;
                } else {
                    self.turret_angle += da * mul;
                }
            } else if da.abs() < props.max_da + 0.001 {
                self.turret_angle += da;
                matchz = true;
            } else if da < 0.0 {
                self.turret_angle -= props.max_da;
            } else {
                self.turret_angle += props.max_da;
            }

            // Pitch (MatrixObjectCannon.cpp:1183-1219).
            let horiz = glam::Vec2::new(dirx, diry).length();
            let mut dang = (tgtpos.z - fire_center.z).atan2(horiz);
            dang = dang.clamp(props.max_bottom_angle, props.max_top_angle);
            let da = angle_dist(self.angle_x, dang);
            let mut matchx = false;
            if props.max_da == 0.0 {
                if da.abs() < CANNON_MIN_DANGLE {
                    matchx = true;
                    self.angle_x = dang;
                } else {
                    self.angle_x += da * mul;
                }
            } else if da.abs() < props.max_da + 0.001 {
                self.angle_x += da;
                matchx = true;
            } else if da < 0.0 {
                self.angle_x -= props.max_da;
            } else {
                self.angle_x += props.max_da;
            }

            self.rchange |= MR_MATRIX;

            // Re-aim the weapon effects — per-barrel muzzles
            // (`m_FireFrom[i]` / `m_FireDir[i]`, :1226-1231).
            let fd = self.fire_dir_now();
            for (i, &w) in self.weapons.iter().enumerate() {
                let ff = self.fire_from_barrel(i);
                if let Some(we) = objs.weapons.get_mut(w) {
                    we.modify(ff, fd, glam::Vec3::ZERO);
                }
            }

            // Fire decision (:1246-1290).
            if itstime && matchx && matchz {
                let tgt_geo = tgtpos - self.target_disp;
                let dq = (tgt_geo - self.core.geo_center).length_squared();
                let ddq = self.fire_radius(objs);
                if dq <= ddq * ddq {
                    let mut aim_ok = true;
                    let tgt_radius = objs.get(target_id).map(|o| o.core().radius).unwrap_or(0.0);
                    for (i, &w) in self.weapons.iter().enumerate() {
                        let ff = self.fire_from_barrel(i);
                        let wdist = objs.weapons.get(w).map(|x| x.weapon_dist()).unwrap_or(0.0);
                        let (_, hp) = trace(
                            map,
                            objs,
                            ff,
                            ff + fd * wdist,
                            crate::matrix_game::common::TRACE_ALL,
                            self.self_id,
                        );
                        let dist = dist_segment_point(ff, hp, tgt_geo);
                        if dist > tgt_radius * 2.0 {
                            aim_ok = false;
                            break;
                        }
                    }
                    open_fire = aim_ok;
                }
            }
        }

        if open_fire {
            self.null_target_time = CANNON_NULL_TARGET_TIME;
            for &w in &self.weapons {
                if let Some(we) = objs.weapons.get_mut(w) {
                    we.fire_begin(glam::Vec3::ZERO, self.self_id);
                }
            }
            let _firewas = self.weapons_logic(takt, map, objs, rng);

            // Shoot-wide correction (:1342-1349): after a second
            // without confirmed hits, displace the aim point.
            self.time_from_fire -= takt;
            if self.time_from_fire <= 0 {
                let r = objs
                    .get(self.target.unwrap())
                    .map(|o| o.core().radius)
                    .unwrap_or(0.0)
                    * 0.5;
                let fsrnd = |rng: &mut Rnd, x: f32| (rng.float01() as f32 * 2.0 - 1.0) * x;
                self.target_disp = glam::Vec3::new(fsrnd(rng, r), fsrnd(rng, r), fsrnd(rng, r));
                self.time_from_fire = CANNON_TIME_FROM_FIRE;
            }
        } else {
            // `no_target` tail (:1056-1126).
            if self.null_target_time > 0 {
                self.time_from_fire = (self.time_from_fire - takt).max(0);
                self.null_target_time -= takt;
                if self.null_target_time <= 0 {
                    for &w in &self.weapons {
                        if let Some(we) = objs.weapons.get_mut(w) {
                            we.fire_end();
                        }
                    }
                    self.null_target_time = 0;
                    return;
                }
            } else {
                self.time_from_fire = CANNON_TIME_FROM_FIRE;
                self.target_disp = glam::Vec3::ZERO;
            }
            let _firewas = self.weapons_logic(takt, map, objs, rng);
        }
    }

    fn side(&self) -> i32 {
        self.side
    }

    fn is_live(&self) -> bool {
        self.state != CannonState::Dip
    }

    fn need_repair(&self) -> bool {
        // `NeedRepair` — below max HP and built.
        self.state == CannonState::Idle && self.hit_point < self.hit_point_max
    }

    /// Port of `CMatrixCannon::Damage` (MatrixObjectCannon.cpp:
    /// 1353-1541). Sounds, progress-bar updates, minimap flashes and
    /// the war-camera hook are not ported.
    fn damage(
        &mut self,
        weap: crate::matrix_game::effects::weapon::Weapon,
        _pos: Vec3,
        _dir: Vec3,
        attacker_side: i32,
        _attacker: Option<ObjectId>,
        self_id: ObjectId,
        objs: &mut Objects,
    ) -> bool {
        use crate::matrix_game::common::PLAYER_SIDE;
        use crate::matrix_game::effects::weapon::{
            weap_to_index, WEAPON_FLAMETHROWER, WEAPON_INSTANT_DEATH, WEAPON_LIGHTENING,
            WEAPON_REPAIR,
        };
        use crate::matrix_game::map_static::{OBJECT_STATE_ABLAZE, OBJECT_STATE_SHORTED};

        let cfg = crate::matrix_game::config::global();
        let now = crate::matrix_game::map::current_elapsed_ms();

        let instant = weap == WEAPON_INSTANT_DEATH;
        if !instant {
            if self.invulnerable {
                return false;
            }
            if self.state == CannonState::Dip {
                return true;
            }

            let friendly_fire = attacker_side != 0 && attacker_side == self.side;
            let mut damagek = if friendly_fire || self.side != PLAYER_SIDE {
                1.0
            } else {
                cfg.difficulty.k_damage_enemy_to_player
            };
            if friendly_fire && self.side == PLAYER_SIDE {
                damagek *= cfg.difficulty.k_friendly_fire;
            }

            let idx = weap_to_index(weap);
            let dmg = idx.map(|i| cfg.cannon_damages.table[i]).unwrap_or_default();
            if weap == WEAPON_REPAIR {
                self.hit_point += if friendly_fire {
                    dmg.friend_damage as f32
                } else {
                    dmg.damage as f32
                };
                if self.hit_point > self.hit_point_max {
                    self.hit_point = self.hit_point_max;
                }
                return false;
            }

            if self.hit_point > dmg.mindamage as f32 {
                self.hit_point -= damagek
                    * if friendly_fire {
                        dmg.friend_damage as f32
                    } else {
                        dmg.damage as f32
                    };
            }

            if weap == WEAPON_FLAMETHROWER {
                self.object_state |= OBJECT_STATE_ABLAZE;
                self.last_delay_damage_side = attacker_side;
                let ttl = (self.ablaze_ttl + 300).min(5000);
                self.ablaze_ttl = ttl;
                self.next_time_ablaze = now;
            } else if weap == WEAPON_LIGHTENING {
                self.last_delay_damage_side = attacker_side;
                self.object_state |= OBJECT_STATE_SHORTED;
                self.shorted_ttl += 500;
                self.next_time_shorted = now;
                for &w in &self.weapons {
                    if let Some(we) = objs.weapons.get_mut(w) {
                        we.fire_end();
                    }
                }
            } else {
                self.last_delay_damage_side = 0;
            }

            if self.hit_point > 0.0 {
                if !matches!(
                    weap,
                    crate::matrix_game::effects::weapon::WEAPON_ABLAZE
                        | crate::matrix_game::effects::weapon::WEAPON_SHORTED
                        | WEAPON_LIGHTENING
                        | WEAPON_FLAMETHROWER
                ) {
                    // Impact flash (MatrixObjectCannon.cpp:1464-1468).
                    objs.pending_explosions
                        .push(crate::matrix_game::map_static::ExplosionSpawn {
                            pos: _pos,
                            props: &crate::matrix_game::effects::explosion::EXPLOSION_ROBOT_HIT,
                            fire: false,
                        });
                }
                return false;
            }
            if attacker_side != 0 && !friendly_fire {
                objs.inc_side_stat(attacker_side, |s| s.turret_kill += 1);
            }
        }

        // inst_death (MatrixObjectCannon.cpp:1477-1539): boom at the
        // base origin, then the unit scatter — basis stays put with
        // its smoke column, turret + shaft tumble away.
        let origin = Vec3::new(self.pos.x, self.pos.y, self.pos_z);
        objs.pending_explosions
            .push(crate::matrix_game::map_static::ExplosionSpawn {
                pos: origin,
                props: &crate::matrix_game::effects::explosion::EXPLOSION_ROBOT_BOOM,
                fire: true,
            });
        self.state = CannonState::Dip;
        self.init_dip_scatter();
        for w in self.weapons.drain(..) {
            objs.weapons.release(w);
        }
        let _ = self_id;
        true
    }
}

impl Cannon {
    /// `DIPTakt` (MatrixObjectCannon.cpp:670-775) — wreck-piece
    /// physics: gravity, water splash (silent), landscape rest,
    /// WEAPON_DEBRIS on object hits, end-of-life boom a radius up.
    fn dip_takt(
        &mut self,
        cms: i32,
        map: &crate::matrix_game::map::GameMap,
        objs: &mut crate::matrix_game::map_static::Objects,
        rng: &mut Rnd,
    ) {
        use crate::matrix_game::common::{TRACE_ALL, WATER_LEVEL};
        use crate::matrix_game::effects::konus::Konus;
        use crate::matrix_game::effects::smoke_and_fire::fsrnd;
        use crate::matrix_game::effects::GameEffect;
        use crate::matrix_game::map_trace::{trace, TraceStop};

        let ms = cms as f32;
        let self_id = self.self_id;
        let radius = self.core.radius;
        let mut i = 0;
        while i < self.dip_units.len() {
            let u = &mut self.dip_units[i];
            u.ttl -= ms;
            u.smoke.takt(ms, rng);
            if u.ttl <= 0.0 {
                let mut pos = u.pos;
                pos.z += radius;
                objs.pending_explosions
                    .push(crate::matrix_game::map_static::ExplosionSpawn {
                        pos,
                        props: &crate::matrix_game::effects::explosion::EXPLOSION_ROBOT_BOOM_SMALL,
                        fire: true,
                    });
                let mut u = self.dip_units.swap_remove(i);
                u.smoke.set_ttl(1000.0);
                objs.pending_effects.push(GameEffect::Smoke(u.smoke));
                continue;
            }
            if u.velocity != Vec3::ZERO {
                let oldpos = u.pos;
                u.velocity.z -= 0.0002 * ms;
                u.pos += u.velocity * ms;
                let (o, hitpos) = trace(map, objs, oldpos, u.pos, TRACE_ALL, self_id);
                match o {
                    TraceStop::Water => {
                        if map.get_z(hitpos.x, hitpos.y) < WATER_LEVEL {
                            objs.pending_effects.push(GameEffect::Konus(Konus::new_splash(
                                hitpos,
                                Vec3::Z,
                                10.0,
                                5.0,
                                fsrnd(rng, std::f32::consts::PI),
                                1000.0,
                                true,
                                crate::matrix_lib::three_g::billboard::TexRef::Path(
                                    crate::matrix_game::effects::effects_renderer::TEXTURE_PATH_SPLASH,
                                ),
                            )));
                            let mut u = self.dip_units.swap_remove(i);
                            u.smoke.set_ttl(1000.0);
                            objs.pending_effects.push(GameEffect::Smoke(u.smoke));
                            continue;
                        }
                    }
                    TraceStop::Landscape => {
                        u.velocity = Vec3::ZERO;
                        u.pos = hitpos;
                        u.freeze_t =
                            Some(crate::matrix_game::map::current_elapsed_ms() as f32);
                    }
                    TraceStop::Object(hit_id) => {
                        let v = u.velocity;
                        u.pos = hitpos;
                        u.ttl = 1.0;
                        objs.apply_damage(
                            hit_id,
                            crate::matrix_game::effects::weapon::WEAPON_DEBRIS,
                            hitpos,
                            v,
                            0,
                            None,
                        );
                    }
                    TraceStop::None => {}
                }
            }
            let pos = self.dip_units[i].pos;
            self.dip_units[i].smoke.set_pos(pos);
            i += 1;
        }
    }

    /// The unit seeding of `inst_death` (MatrixObjectCannon.cpp:
    /// 1499-1539): basis keeps TTL + smoke but never moves; turret and
    /// shaft fly with random spins.
    fn init_dip_scatter(&mut self) {
        use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd, Smoke};
        let origin = Vec3::new(self.pos.x, self.pos.y, self.pos_z);
        let mut vrng = crate::matrix_game::logic::Rnd::new(
            ((self.pos.x as i32) << 8 ^ (self.pos.y as i32) ^ 0x4A11).max(1),
        );
        let (s, c) = self.angle.sin_cos();
        let yaw = |v: Vec3| Vec3::new(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
        let k = ((self.kind - 1).max(0) as usize).min(3);

        let base_yaw = self.angle;
        let mut push = |pos: Vec3, part: CannonDipPart, flies: bool, vrng: &mut Rnd| {
            let (v, spins) = if flies {
                (
                    Vec3::new(fsrnd(vrng, 0.08), fsrnd(vrng, 0.08), 0.1),
                    (fsrnd(vrng, 0.005), fsrnd(vrng, 0.005), fsrnd(vrng, 0.005)),
                )
            } else {
                (Vec3::ZERO, (0.0, 0.0, 0.0))
            };
            let ttl = frnd(vrng, 3000.0) + 2000.0;
            self.dip_units.push(CannonDipUnit {
                pos,
                velocity: v,
                ttl,
                dp: spins.0,
                dy: spins.1,
                dr: spins.2,
                part,
                yaw: if flies { 0.0 } else { base_yaw },
                freeze_t: None,
                smoke: Smoke::new(
                    pos,
                    ttl + 100_000.0,
                    1000.0,
                    100.0,
                    0xFF00_0000,
                    false,
                    1.0 / 30.0,
                ),
            });
        };
        push(origin, CannonDipPart::Basis, false, &mut vrng);
        push(
            origin + yaw(BASIS_MOUNT),
            CannonDipPart::Turret,
            true,
            &mut vrng,
        );
        push(
            origin + yaw(SHAFT_PIVOT[k]),
            CannonDipPart::Shaft,
            true,
            &mut vrng,
        );
    }
}

/// `AngleDist` — shortest signed angular distance `to - from`,
/// wrapped to `(-π, π]`.
fn angle_dist(from: f32, to: f32) -> f32 {
    let mut d = to - from;
    while d > std::f32::consts::PI {
        d -= 2.0 * std::f32::consts::PI;
    }
    while d < -std::f32::consts::PI {
        d += 2.0 * std::f32::consts::PI;
    }
    d
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
    /// Mount offsets baked into the turret / shaft vertex data; DIP
    /// piece transforms subtract `R·bake` so the piece tumbles around
    /// its own origin like the C++ unit graphs (which are unbaked).
    turret_bake: Vec3,
    shaft_bake: Vec3,
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
    /// Per-(kind, sub-mesh) instance ranges computed each frame by
    /// `sync_cannons` — each entry draws one sub-mesh (0 basis /
    /// 1 turret / 2 shaft) over a contiguous instance range, so the
    /// turret and barrel carry their own aim transforms.
    draws: Vec<(usize, usize, u32, u32)>, // (kind idx, part, offset, count)
    /// Slot-marker draws — Basis-only stamps placed at `(offset, count)`
    /// in the instance buffer. Rendered separately from `draws` because
    /// they only emit the `Basis` sub-mesh (no Turret / Shaft).
    marker_draw: Option<(u32, u32)>,
    /// DIP wreck-piece draws — one single-sub-mesh draw per flying
    /// part: (kind index, which sub-mesh, instance slot).
    dip_draws: Vec<(usize, CannonDipPart, u32)>,
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

/// Instance slots: 3 sub-mesh instances per live cannon, plus slot
/// markers and DIP wreck pieces.
const MAX_CANNON_INSTANCES: u32 = 256;

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
            size: (MAX_CANNON_INSTANCES as u64) * std::mem::size_of::<InstanceData>() as u64,
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
                turret_bake: basis_mount,
                shaft_bake: shaft_offset,
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
            instance_capacity: MAX_CANNON_INSTANCES,
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
            dip_draws: Vec::new(),
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
        self.dip_draws.clear();
        self.shadow_batches.clear();
        let mut instance_data: Vec<InstanceData> = Vec::with_capacity(16);
        let mut dip_pieces: Vec<(usize, CannonDipPart, InstanceData)> = Vec::new();

        // Group cannons by (kind, sub-mesh) so draws are contiguous;
        // basis / turret / shaft each get their own transform so the
        // turret visibly tracks `turret_angle` / `angle_x`.
        let mut by_kind: [[Vec<InstanceData>; 3]; 4] = Default::default();
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
            // DIP wrecks: each surviving piece gets its own
            // single-sub-mesh draw with a tumble transform; no shadow
            // (the C++ flips SHADOW_OFF at inst_death).
            if c.state == CannonState::Dip {
                let t = crate::matrix_game::map::current_elapsed_ms() as f32;
                let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(c.side);
                let Some(kgpu) = self.kinds.get(k) else { continue };
                for u in &c.dip_units {
                    if u.ttl <= 0.0 {
                        continue;
                    }
                    let te = u.freeze_t.unwrap_or(t);
                    let rot = glam::Quat::from_rotation_z(u.yaw)
                        * glam::Quat::from_euler(
                            glam::EulerRot::YXZ,
                            u.dy * te,
                            u.dp * te,
                            u.dr * te,
                        );
                    let bake = match u.part {
                        CannonDipPart::Basis => Vec3::ZERO,
                        CannonDipPart::Turret => kgpu.turret_bake,
                        CannonDipPart::Shaft => kgpu.shaft_bake,
                    };
                    let tr = u.pos - rot * bake - Vec3::new(cx, cy, 0.0);
                    let rx = rot * Vec3::X;
                    let ry = rot * Vec3::Y;
                    let rz = rot * Vec3::Z;
                    dip_pieces.push((
                        k,
                        u.part,
                        InstanceData {
                            row0: [rx.x, ry.x, rz.x, tr.x],
                            row1: [rx.y, ry.y, rz.y, tr.y],
                            row2: [rx.z, ry.z, rz.z, tr.z],
                            row3: [0.0, 0.0, 0.0, 1.0],
                            // DIP texture factor 0xFF808080
                            // (MatrixObjectCannon.cpp:535).
                            terrain_color: [0.502, 0.502, 0.502, 1.0],
                            unit_offset: [0.0, 0.0, 0.0, 0.0],
                            side_color: [sr, sg, sb, 1.0],
                        },
                    ));
                }
                continue;
            }
            let (tb, sb) = self
                .kinds
                .get(k)
                .map(|g| (g.turret_bake, g.shaft_bake))
                .unwrap_or((Vec3::ZERO, Vec3::ZERO));
            for part in 0..3 {
                by_kind[k][part].push(cannon_part_instance(c, cx, cy, part, tb, sb));
            }

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
                        self.shadow_system
                            .calc_proj(&local_pts, light_world, world_uc)
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
                            device,
                            map,
                            &proj,
                            &texture,
                            6,
                            [cx, cy],
                        ) {
                            self.shadow_batches.push(batch);
                        }
                        alive_shadow_ids.insert(id);
                    }
                }
            }
        }
        if let Some(g) = ghost {
            // Ghost previews at rest pose — the base transform serves
            // all three sub-meshes.
            let k = ((g.kind - 1).max(0) as usize).min(3);
            let inst = ghost_instance(&g, cx, cy);
            for part in 0..3 {
                by_kind[k][part].push(inst);
            }
        }

        let mut offset: u32 = 0;
        'pack: for (k, parts) in by_kind.iter().enumerate() {
            for (part, list) in parts.iter().enumerate() {
                let count = list.len() as u32;
                if count == 0 {
                    continue;
                }
                if offset + count > self.instance_capacity {
                    break 'pack;
                }
                for inst in list {
                    instance_data.push(*inst);
                }
                self.draws.push((k, part, offset, count));
                offset += count;
            }
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

        // DIP wreck pieces — appended last; each draws a single
        // sub-mesh with its own tumble transform.
        for (k, part, inst) in dip_pieces {
            if offset + 1 > self.instance_capacity {
                break;
            }
            instance_data.push(inst);
            self.dip_draws.push((k, part, offset));
            offset += 1;
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
        for (kind, part, offset, count) in &self.draws {
            let Some(kgpu) = self.kinds.get(*kind) else {
                continue;
            };
            let mesh = match part {
                1 => &kgpu.turret,
                2 => &kgpu.shaft,
                _ => &kgpu.basis,
            };
            let Some(mesh) = mesh.as_ref() else { continue };
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            for surface in &mesh.surfaces {
                pass.set_bind_group(0, &surface.bind_group, &[]);
                pass.set_index_buffer(surface.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..surface.num_indices, 0, *offset..(*offset + *count));
            }
        }
        // DIP wreck pieces — one sub-mesh per instance slot.
        for (k, part, slot) in &self.dip_draws {
            let Some(kgpu) = self.kinds.get(*k) else {
                continue;
            };
            let mesh = match part {
                CannonDipPart::Basis => &kgpu.basis,
                CannonDipPart::Turret => &kgpu.turret,
                CannonDipPart::Shaft => &kgpu.shaft,
            };
            let Some(mesh) = mesh.as_ref() else { continue };
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            for surface in &mesh.surfaces {
                pass.set_bind_group(0, &surface.bind_group, &[]);
                pass.set_index_buffer(surface.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..surface.num_indices, 0, *slot..(*slot + 1));
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

/// Per-sub-mesh world transform. The basis takes the base yaw only;
/// the turret adds `Rz(turret_angle)` about its mount; the shaft adds
/// `Rz(turret_angle)·Rx(angle_x)` about its pivot — the same chain
/// `shaft_point_world` uses for the fire math (RNeed mount chain,
/// MatrixObjectCannon.cpp:282-310), so barrels and muzzle flashes
/// stay aligned. The mount offsets are baked into the turret/shaft
/// vertex data, hence the `T(bake)·R·T(-bake)` sandwich.
fn cannon_part_instance(
    c: &Cannon,
    cx: f32,
    cy: f32,
    part: usize,
    turret_bake: Vec3,
    shaft_bake: Vec3,
) -> InstanceData {
    let base = Mat4::from_translation(Vec3::new(c.pos.x - cx, c.pos.y - cy, c.pos_z))
        * Mat4::from_rotation_z(c.angle);
    let m = match part {
        1 => {
            base * Mat4::from_translation(turret_bake)
                * Mat4::from_rotation_z(c.turret_angle)
                * Mat4::from_translation(-turret_bake)
        }
        2 => {
            base * Mat4::from_translation(shaft_bake)
                * Mat4::from_rotation_z(c.turret_angle)
                * Mat4::from_rotation_x(c.angle_x)
                * Mat4::from_translation(-shaft_bake)
        }
        _ => base,
    };
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
    let r = |i: usize| [m.x_axis[i], m.y_axis[i], m.z_axis[i], m.w_axis[i]];
    InstanceData {
        row0: r(0),
        row1: r(1),
        row2: r(2),
        row3: r(3),
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
        m[12],
        m[13],
        m[14]
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
