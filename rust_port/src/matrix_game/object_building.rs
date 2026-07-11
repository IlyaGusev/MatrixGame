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
//! Projected shadows are wired through the shared `ShadowSystem`
//! (matrix_game::shadow). Each instance gets a baked silhouette texture and a
//! one-shot ground-projection mesh built against the terrain at spawn — the
//! `SHADOW_PROJ_STATIC` path in MatrixObjectBuilding.cpp:194-243.

use std::collections::{BTreeMap, HashMap};

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{unpack_rgb, FOG_END, FOG_START, PLAYER_SIDE};
use crate::matrix_game::config::RobotUnitKind;
use crate::matrix_game::effects::point_light::PointLightSystem;
use crate::matrix_game::interface::constructor::RobotConfig;
use crate::matrix_game::logic::Rnd;
use crate::matrix_game::map::{BuildingInstance, GameMap};
use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectId, ObjectType, Objects, MR_ALL,
};
use crate::matrix_game::robot::{ChassisKind, Robot};
use crate::matrix_game::shadow::{ShadowBatch, ShadowMeshSurface, ShadowMeshVertex, ShadowSystem};
use crate::matrix_lib::three_g::texture::{
    create_solid_texture, create_texture_from_rgba, decode_texture_bytes,
};
use crate::matrix_lib::three_g::vector_object::{self, CvoGroup, MaterialSpec};

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
    Base = 0,
    Titan = 1,
    Plasma = 2,
    Electronic = 3,
    Energy = 4,
    Repair = 5,
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
    Closed = 0,
    Opening = 1,
    Opened = 2,
    Closing = 3,
    /// `BUILDING_DIP` — "dying in progress". Reached after a base is
    /// destroyed; triggers the multi-second explosion sequence.
    Dip = 4,
    /// `BUILDING_DIP_EXPLODED` — explosion sequence finished; ruin
    /// remains but the building object is otherwise inert.
    DipExploded = 5,
}

/// Default floor height of an opening base before it rises to ground
/// level (MatrixObjectBuilding.hpp:12). Used by the `BASE_OPENING`
/// animation; stored here so the value lives alongside the type
/// definition when the animation code lands.
pub const BASE_FLOOR_Z: f32 = -63.0;

/// `MAX_ZAHVAT_POINTS` (MatrixObjectBuilding.hpp:14) — capture
/// progress steps until ownership flips.
pub const MAX_ZAHVAT_POINTS: i32 = 14;

/// `CAPTURE_RADIUS` (MatrixObjectBuilding.hpp:27).
pub const CAPTURE_RADIUS: f32 = 50.0;
/// `CAPTURE_SEEK_ROBOT_PERIOD` (MatrixObjectBuilding.hpp:28).
pub const CAPTURE_SEEK_ROBOT_PERIOD: i64 = 500;
/// `DISTANCE_CAPTURE_ME` (MatrixObjectBuilding.hpp:30).
pub const DISTANCE_CAPTURE_ME: f32 = 300.0;

/// Port of `STrueColor` (MatrixObjectBuilding.hpp:93-99) — the
/// capture-progress paint state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrueColor {
    pub color: u32,
    pub colored_cnt: i32,
}

/// Port of `ECaptureStatus` (MatrixObjectBuilding.hpp:84-90).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStatus {
    Done,
    InProgress,
    TooFar,
    Busy,
}

/// Max queued build items per building (MatrixObjectBuilding.hpp:43).
/// `CBuildStack` rejects AddItem once the queue hits this.
pub const MAX_STACK_UNITS: usize = 6;

/// Fallback build time when `g_Config.m_Timings[UNIT_ROBOT]` isn't
/// loaded yet (e.g. tests that skip `load_config`). Matches the value
/// used by the shipping `robots.dat` so behaviour is preserved.
pub const UNIT_ROBOT_BUILD_TIME_MS: i32 = 5000;

// `robot_build_time_ms` / `turret_build_time_ms` are `g_Config.m_Timings`
// accessors — ported to `matrix_game::config` alongside the timing table.
// Re-exported here for the existing in-file call sites.
pub use crate::matrix_game::config::{robot_build_time_ms, turret_build_time_ms};

/// Port of `CBuildStack` (MatrixObjectBuilding.{cpp,hpp}). The C++
/// holds an intrusive list of `CMatrixMapStatic*` (robots / cannons /
/// flyers) with `m_NextStackItem` / `m_PrevStackItem` pointers. We
/// use `Vec<PendingItem>` — the list semantics aren't needed since
/// we only ever pop the top. Queued robots exist only as their
/// `RobotConfig` spec (from the constructor panel); the live `Robot`
/// is built at dequeue-time.
#[derive(Debug, Clone, Default)]
pub struct BuildStack {
    items: Vec<PendingItem>,
    timer: i32,
    /// Items dropped by `clear`/`delete_item`, awaiting the C++
    /// `DeleteItem` side effects (resource refund + queued-turret
    /// ghost teardown, MatrixObjectBuilding.cpp:1844-1880). Applied
    /// by the building's logic takt, which has arena access. The bool
    /// is "refund" — false on building death (~CBuildStack refunds
    /// nothing, :1882-1919).
    pending_teardown: Vec<(PendingItem, bool)>,
}

#[derive(Debug, Clone, Copy)]
pub struct PendingItem {
    pub kind: PendingKind,
    pub side: i32,
}

#[derive(Debug, Clone, Copy)]
#[allow(clippy::large_enum_variant)] // RobotConfig carries the full spec; boxing wouldn't pay
pub enum PendingKind {
    /// A fully-configured robot (chassis + armor + head + weapons)
    /// produced by the constructor panel. Ports the `CMatrixRobotAI*`
    /// the C++ `CConstructor::StackRobot` pushes onto `m_BS.AddItem`
    /// (CConstructor.cpp:219).
    Robot(RobotConfig),
    /// A turret (cannon). Port of the `CMatrixCannon*` pushed from
    /// the turret-build UI. `slot` is the 0-based slot index on the
    /// parent building; `turret_kind` is the RUK_TURRET_* variant.
    Turret { slot: i32, turret_kind: i32 },
}

impl PendingKind {
    pub fn build_time_ms(&self) -> i32 {
        match self {
            PendingKind::Robot(_) => robot_build_time_ms(),
            PendingKind::Turret { .. } => turret_build_time_ms(),
        }
    }
}

impl BuildStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn items(&self) -> usize {
        self.items.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &PendingItem> {
        self.items.iter()
    }
    /// Port of `CBuildStack::GetRobotsCnt` (MatrixObjectBuilding.cpp:
    /// 1938-1950) — only queued robots count toward the robot cap.
    pub fn robots_cnt(&self) -> usize {
        self.items
            .iter()
            .filter(|it| matches!(it.kind, PendingKind::Robot(_)))
            .count()
    }
    pub fn is_full(&self) -> bool {
        self.items.len() >= MAX_STACK_UNITS
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Port of `CBuildStack::ClearStack` (MatrixObjectBuilding.cpp:
    /// 1952-1954) — drains every queued item through `DeleteItem`,
    /// refunding resources and tearing down queued-turret ghosts.
    pub fn clear(&mut self) {
        self.pending_teardown
            .extend(self.items.drain(..).map(|it| (it, true)));
        self.timer = 0;
    }

    /// Death-path drain — `~CBuildStack` (MatrixObjectBuilding.cpp:
    /// 1882-1919): queued items are destroyed with NO refund.
    pub fn clear_no_refund(&mut self) {
        self.pending_teardown
            .extend(self.items.drain(..).map(|it| (it, false)));
        self.timer = 0;
    }

    /// Port of `CBuildStack::DeleteItem(no)` (MatrixObjectBuilding.cpp:
    /// 1844-1880). `no` is 1-based like the C++; cancelling the head
    /// resets the build timer.
    pub fn delete_item(&mut self, no: usize) -> bool {
        if no == 0 || no > self.items.len() {
            return false;
        }
        if no == 1 {
            self.timer = 0;
        }
        let it = self.items.remove(no - 1);
        self.pending_teardown.push((it, true));
        true
    }

    /// Port of `CBuildStack::AddItem` (MatrixObjectBuilding.cpp:
    /// 1832-1842). Pushes `item` to the tail of the queue if there's
    /// room. The C++ also creates a UI stack-icon on the player-side
    /// HUD (`g_IFaceList->CreateStackIcon`); that hook lands with the
    /// rest of the player-side interface integration.
    pub fn add_item(&mut self, item: PendingItem) -> bool {
        if self.is_full() {
            return false;
        }
        self.items.push(item);
        true
    }

    /// Read-only view of the head item (what `TickTimer` is currently
    /// producing). Maps to `m_Top` in C++.
    pub fn head(&self) -> Option<&PendingItem> {
        self.items.first()
    }

    /// Read-only view of the full queue, head first. Maps to walking
    /// the `m_Top` → `m_NextStackItem` chain in the C++.
    pub fn list(&self) -> &[PendingItem] {
        &self.items
    }

    /// Progress ratio of the head item in `[0.0, 1.0]`. Drives the
    /// progress-bar UI the C++ updates at MatrixObjectBuilding.cpp:1681.
    pub fn progress(&self) -> f32 {
        let Some(head) = self.items.first() else {
            return 0.0;
        };
        let total = head.kind.build_time_ms().max(1) as f32;
        (self.timer as f32 / total).clamp(0.0, 1.0)
    }

    /// Port of `CBuildStack::TickTimer` (MatrixObjectBuilding.cpp:
    /// 1665-1717) for the robot branch only. Advances the timer;
    /// when it hits `UNIT_ROBOT_BUILD_TIME_MS` and the parent base
    /// is CLOSED, dequeues the head, produces a `Robot` at the
    /// base's spawn position, inserts it into the arena, and calls
    /// `JoinToGroup` (not ported).
    ///
    /// Returns the new robot's `ObjectId` on production — caller can
    /// use it to attach point lights / effects for visibility.
    pub fn tick_timer(
        &mut self,
        cms: i32,
        objs: &mut Objects,
        parent_self_id: ObjectId,
        parent_state: BaseState,
        parent_pos: glam::Vec3,
        parent_angle_quad: i32,
    ) -> Option<ObjectId> {
        if self.items.is_empty() {
            self.timer = 0;
            return None;
        }
        self.timer += cms;

        let head = self.items[0];
        let total = head.kind.build_time_ms();

        // Per-tick HP ramp for in-progress turrets — port of
        // MatrixObjectBuilding.cpp:1764-1769:
        //   percent_done = m_Timer / g_Config.m_Timings[UNIT_TURRET];
        //   ca->SetHitPoint(ca->GetMaxHitPoint() * percent_done);
        // The C++ also invokes the per-build-stack progress-bar UI
        // here; we keep that on the existing `progress()` accessor.
        if let PendingKind::Turret { slot, .. } = head.kind {
            let progress = (self.timer as f32 / total.max(1) as f32).clamp(0.0, 1.0);
            ramp_turret_hp(objs, parent_self_id, slot, progress);
        }

        if self.timer < total {
            return None;
        }
        if matches!(head.kind, PendingKind::Robot(_)) && parent_state != BaseState::Closed {
            // Wait — the C++ gates only robot production on
            // BASE_CLOSED (MatrixObjectBuilding.cpp:1690); the turret
            // branch (:1776) completes regardless of base state.
            return None;
        }

        // Produce: pop, build item, insert.
        self.items.remove(0);
        self.timer = 0;

        match head.kind {
            PendingKind::Robot(cfg) => {
                let spawn_pos = glam::Vec3::new(parent_pos.x, parent_pos.y, parent_pos.z);
                let chassis = chassis_from_kind(cfg.chassis.kind).unwrap_or(ChassisKind::Track);
                // The C++ computes a balanced team but then
                // unconditionally overwrites it with
                // `m_NewBorn->m_Team = 0;` (CConstructor.cpp:121) —
                // every factory robot ships on team 0.
                let team = 0;
                let mut robot = Robot::new(spawn_pos, head.side, chassis);
                robot.robot_spawn(parent_self_id, parent_angle_quad, parent_pos.z);
                robot.set_team(team);
                robot.config = cfg;
                // Initialise HP from the summed structure of the
                // equipped chassis + armor + head. Mirrors
                // `CMatrixRobotAI::CalcRobotMass` →
                // `InitMaxHitpoint(hp)` at MatrixRobot.cpp:4150-4293.
                let hp = crate::matrix_game::interface::constructor::raw_hitpoint_of_live(
                    &cfg.chassis,
                    &cfg.hull,
                    &cfg.head,
                ) as f32;
                if hp > 0.0 {
                    robot.hit_point_max = hp;
                    robot.hit_point = hp;
                }
                // Port of CConstructor.cpp:218 — populate the display
                // name (m_Name) using the construction-name helper.
                robot.name = crate::matrix_game::interface::constructor::name_of_live(
                    &cfg.chassis,
                    &cfg.hull,
                    &cfg.head,
                    &robot_weapons_from_cfg(&cfg),
                );
                let id = objs.spawn(Box::new(robot));
                if let Some(obj) = objs.get_mut(id) {
                    let r: &mut crate::matrix_game::robot::Robot = unsafe {
                        &mut *(obj as *mut dyn MapStatic as *mut crate::matrix_game::robot::Robot)
                    };
                    r.self_id = Some(id);
                    // Seed map coords so gather_info doesn't read cell
                    // (0,0) before the robot's first move order.
                    if let Some(map) = crate::matrix_game::map::current_map() {
                        r.map_pos_calc(map);
                    }
                }
                objs.add_lt(id);
                // RobotSpawn rally (MatrixRobot.cpp:2216-2221): player
                // robots get a place near the base + a muted
                // attack-move there so they disperse after move-out.
                if head.side != crate::matrix_game::common::PLAYER_SIDE {
                    // AI robots pick a team via ClacSpawnTeam
                    // (MatrixRobot.cpp:2204-2205), applied by the side
                    // AI on its next takt.
                    objs.pending_ai_spawn.push(id);
                }
                if head.side == crate::matrix_game::common::PLAYER_SIDE {
                    objs.pending_spawn_rallies.push(id);
                    // S_ROBOT_BUILD_END(_ALT) 50/50
                    // (MatrixObjectBuilding.cpp:1693-1695).
                    let alt = (spawn_pos.x as i32 ^ spawn_pos.y as i32) & 1 == 1;
                    objs.queue_snd(if alt { "r_build_ea" } else { "r_build_e" });
                }
                Some(id)
            }
            PendingKind::Turret { slot, turret_kind } => {
                // Turret placement happens immediately on the parent
                // building — this branch finalises construction:
                // ramp HP to 100%, drop invulnerability, flip to IDLE.
                // Port of MatrixObjectBuilding.cpp:1779-1813.
                ramp_turret_hp(objs, parent_self_id, slot, 1.0);
                // S_TURRET_BUILD_<num> voice, player side
                // (MatrixObjectBuilding.cpp:1780-1786).
                if head.side == crate::matrix_game::common::PLAYER_SIDE {
                    objs.queue_snd(&format!("t_build_{}", (turret_kind - 1).clamp(0, 3)));
                }
                // STAT_TURRET_BUILD (MatrixObjectBuilding.cpp:1810).
                if head.side != 0 {
                    objs.inc_side_stat(head.side, |s| s.turret_build += 1);
                }
                log::debug!(
                    "build: turret completed (kind={} slot={}) side={}",
                    turret_kind,
                    slot,
                    head.side,
                );
                None
            }
        }
    }
}

/// Walk the live arena, find the in-construction cannon mounted on
/// `parent_id` at `slot`, and forward the build progress so its HP
/// ramps + state flips at completion. Mirrors the per-tick
/// `ca->SetHitPoint(...)` call inside `CBuildStack::TickTimer`'s
/// turret branch (MatrixObjectBuilding.cpp:1769).
fn ramp_turret_hp(objs: &mut Objects, parent_id: ObjectId, slot: i32, progress: f32) {
    use crate::matrix_game::map_static::ObjectType;
    use crate::matrix_game::object_cannon::Cannon;

    let live: Vec<ObjectId> = objs.iter_units().collect();
    for id in live {
        let Some(obj) = objs.get_mut(id) else {
            continue;
        };
        if !matches!(obj.core().obj_type, ObjectType::Cannon) {
            continue;
        }
        let c: &mut Cannon = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Cannon) };
        if c.parent != Some(parent_id) || c.slot != slot {
            continue;
        }
        c.tick_construction(progress);
        return;
    }
}

/// Helper: turn the chassis/armor/head/weapons of a `RobotConfig`
/// into the `[WeaponUnit; MAX_WEAPON_CNT]` shape `name_of_live`
/// expects (the latter is shared with the constructor preview, which
/// holds `WeaponUnit` values directly).
pub fn robot_weapons_from_cfg(
    cfg: &RobotConfig,
) -> [crate::matrix_game::interface::constructor::WeaponUnit;
       crate::matrix_game::object_robot::MAX_WEAPON_CNT] {
    let mut out = [crate::matrix_game::interface::constructor::WeaponUnit::empty();
        crate::matrix_game::object_robot::MAX_WEAPON_CNT];
    for (i, w) in cfg.weapon.iter().enumerate() {
        out[i] = crate::matrix_game::interface::constructor::WeaponUnit {
            pos: i as i32 + 1,
            unit: *w,
        };
    }
    out
}

/// Translate `ChassisKind` (the render-side enum, 0..=4) into the
/// kind-space values (`RUK_CHASSIS_*`, 1..=5) the constructor panel
/// uses. The C++ doesn't need this because it has one enum; we keep
/// two for historical reasons so this bridge lives here.
pub fn kind_from_chassis(c: ChassisKind) -> RobotUnitKind {
    match c {
        ChassisKind::Pneumatic => RobotUnitKind::CHASSIS_PNEUMATIC,
        ChassisKind::Wheel => RobotUnitKind::CHASSIS_WHEEL,
        ChassisKind::Track => RobotUnitKind::CHASSIS_TRACK,
        ChassisKind::Hovercraft => RobotUnitKind::CHASSIS_HOVERCRAFT,
        ChassisKind::AntiGravity => RobotUnitKind::CHASSIS_ANTIGRAVITY,
    }
}

pub fn chassis_from_kind(k: RobotUnitKind) -> Option<ChassisKind> {
    Some(match k.0 {
        1 => ChassisKind::Pneumatic,
        2 => ChassisKind::Wheel,
        3 => ChassisKind::Track,
        4 => ChassisKind::Hovercraft,
        5 => ChassisKind::AntiGravity,
        _ => return None,
    })
}

/// Base selection radius (MatrixObjectBuilding.hpp:25). Each kind adds
/// a small per-kind extra to tighten the pick against the visible
/// footprint — values copied from `CMatrixBuilding::Select`
/// (MatrixObjectBuilding.cpp:1474-1508).
pub const BUILDING_SELECTION_SIZE: f32 = 50.0;

/// Port of the ring-center + radius math in `CMatrixBuilding::Select`
/// (MatrixObjectBuilding.cpp:1466-1509). The C++ matrix offsets use
/// `_21/_22` (row-major D3D = y-axis-world in column-major glam) to
/// push the ring off the robot-pad towards the building centre, and
/// `_11/_12` (x-axis-world) to shear sideways for asymmetric kinds
/// like BASE.
///
/// Returns `(ring_center_world, ring_radius)`. `pos` is the
/// building's `m_Pos` (robot-pad anchor); `build_z` seeds the
/// ring plane Z at `m_Matrix._43 + 5`.
pub fn selection_placement(
    pos: glam::Vec2,
    build_z: f32,
    angle_quad: i32,
    kind: BuildingType,
) -> (glam::Vec3, f32) {
    // `m_Angle` is stored as 0..=3 quarter-rotations
    // (MatrixObjectBuilding.cpp:120-137). Rebuild the matrix basis:
    let ang = (angle_quad & 3) as f32 * std::f32::consts::FRAC_PI_2;
    let (s, c) = ang.sin_cos();
    // Column-major glam: x_axis/y_axis are the matrix columns; their
    // (x, y) components map 1:1 to D3D row-major `_11/_12` and
    // `_21/_22` respectively.
    let x_axis = glam::Vec2::new(c, s);
    let y_axis = glam::Vec2::new(-s, c);

    // MatrixObjectBuilding.cpp:1470-1472.
    let mut p = pos - y_axis * 60.0;
    let mut r = BUILDING_SELECTION_SIZE;

    match kind {
        BuildingType::Base => {
            // :1478-1483
            r += 24.0;
            p -= x_axis * 7.0;
            p += y_axis * 16.0;
        }
        BuildingType::Energy => {
            // :1486-1489
            r += 10.0;
            p -= y_axis * 13.0;
        }
        BuildingType::Plasma => {
            // :1492-1495
            r += 15.0;
            p -= y_axis * 17.0;
        }
        BuildingType::Titan => {
            // :1498-1501
            r += 15.0;
            p -= y_axis * 17.0;
        }
        BuildingType::Electronic => {
            // :1504-1507
            r += 17.0;
            p -= y_axis * 17.0;
        }
        // Repair — no per-kind extra; the C++ falls through with
        // just the base radius + the initial y-axis * 60 offset.
        BuildingType::Repair => {}
    }
    (glam::Vec3::new(p.x, p.y, build_z + 5.0), r)
}

/// One entry in `CMatrixBuilding::m_TurretsPlaces[]`
/// (MatrixObjectBuilding.hpp:32-37). Holds the slot's world XY
/// (derived from move-cell coords at map load), default angle, and
/// the kind of cannon currently mounted there (-1 = empty).
#[derive(Debug, Clone, Copy)]
pub struct TurretPlace {
    /// Move-cell coords (x, y) — same units the C++ uses for slot
    /// equality + the snap test in `IsInPlaces`.
    pub coord: (i32, i32),
    /// Slot center in world space — `coord * GLOBAL_SCALE_MOVE` precomputed.
    pub world: glam::Vec2,
    /// Slot rest angle (radians around Z). Newly-built cannons rotate
    /// to this angle during the placement preview
    /// (MatrixSide.cpp:576-579).
    pub angle: f32,
    /// `m_CannonType` — 1..=4 for occupied, -1 for empty.
    pub cannon_type: i32,
}

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

    /// `m_TurretsPlaces[MAX_PLACES]` (MatrixObjectBuilding.hpp:32-37).
    /// Per-slot world-space coord + rest-angle + currently-mounted
    /// cannon kind (-1 for empty). Seeded from the CMAP cannons records
    /// at map load (MatrixMapPrepare.cpp:852-857). Drives the build-
    /// preview slot-snap, the validity check, and the `cannon_type`
    /// occupancy flag the C++ uses to refuse double-builds on the
    /// same slot (MatrixObjectBuilding.cpp:1569-1576).
    pub turret_places: Vec<TurretPlace>,

    /// `m_UnderAttackTime` (MatrixObjectBuilding.hpp:171). Game-time
    /// ms past which the base is considered "quiet" (no recent
    /// attacker) — controls the attacked-warning UI.
    pub under_attack_time: i32,
    /// `m_CaptureMeNextTime` (MatrixObjectBuilding.hpp:172). Throttle
    /// on capture candidacy announcements.
    pub capture_me_next_time: i32,
    /// `m_CaptureSeekRobotNextTime` (MatrixObjectBuilding.hpp:222) —
    /// next game-time the building scans for an adjacent capturer.
    pub capture_seek_robot_next_time: i64,
    /// `m_InCaptureTime` (MatrixObjectBuilding.hpp:215) — ms left of
    /// "actively being captured"; while >0 the rollback is suppressed.
    pub in_capture_time: i32,
    /// `m_InCaptureNextTimeErase` / `m_InCaptureNextTimePaint`
    /// (MatrixObjectBuilding.hpp:218-221) — pacing for the erase /
    /// paint halves of the capture progress.
    pub in_capture_next_time_erase: i64,
    pub in_capture_next_time_paint: i64,
    /// `m_CaptureNextTimeRollback` (MatrixObjectBuilding.hpp:219).
    pub capture_next_time_rollback: i64,
    /// `m_TrueColor` (MatrixObjectBuilding.hpp:93-99) — capture paint
    /// progress (color + painted points).
    pub true_color: TrueColor,
    /// `m_capture` (MatrixObjectBuilding.hpp) — the Zahvat capture
    /// ring; created lazily for non-base factories.
    pub zahvat: Option<crate::matrix_game::effects::zahvat::Zahvat>,

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

    /// `m_NextExplosionTime` / `m_NextExplosionTimeSound` — union's
    /// DIP branch (MatrixObjectBuilding.hpp:159-161). Game-time ms
    /// when the next explosion effect / sound should fire during the
    /// multi-second DIP sequence. Set by `damage` on death; consumed
    /// by `logic_takt`'s DIP branch.
    pub next_explosion_time: i32,
    pub next_explosion_time_sound: i32,

    /// `m_ShowHitpointTime` (MatrixObjectBuilding.hpp:227). Ms left on
    /// the health-bar overlay; reseeded to HITPOINT_SHOW_TIME_MS on
    /// every hit. Decays to 0 inside `logic_takt`.
    pub show_hitpoint_time: i32,

    /// `m_BaseFloor` progress in [0, 1] — 0 = fully closed (base below
    /// ground), 1 = fully opened (platform raised). Animated by the
    /// BASE_OPENING / BASE_CLOSING state machine at `BASE_FLOOR_SPEED`
    /// per ms (MatrixObjectBuilding.cpp:812-833).
    pub base_floor_progress: f32,

    /// Port of `m_BS` (MatrixObjectBuilding.hpp:177). Queue of items
    /// to produce; advanced per-tick by `logic_takt`.
    pub build_stack: BuildStack,

    /// The arena id for *this* building — populated by
    /// `World::spawn_buildings` immediately after `Objects::spawn`
    /// returns. The C++ doesn't carry this explicitly because its
    /// objects are raw pointers; for the Rust port it's how the
    /// build-stack hands a parent reference to freshly-produced
    /// robots so `RobotSpawn` + the spawn-animation can read the
    /// base's `m_BaseFloor` / `m_State`.
    pub self_id: Option<ObjectId>,
}

impl Building {
    /// Port of `CMatrixBuilding` ctor + `OnLoad` (MatrixObjectBuilding.cpp:
    /// 24-69, :1007+). Minimal: seeds type, pos, angle, side, kind, state,
    /// and the derived base_floor / build_z. Robot-spawn / capture /
    /// progress-bar state stays at default (noop).
    pub fn from_instance(inst: &BuildingInstance) -> Self {
        let kind = BuildingType::from_u8(inst.kind).unwrap_or(BuildingType::Base);
        // Use the same center + radius as the selection ring so
        // picking matches the visible hit area. Ports the
        // `CMatrixBuilding::Select` offset math exactly.
        let (pick_center, radius) = selection_placement(
            glam::Vec2::new(inst.x, inst.y),
            inst.build_z,
            inst.angle as i32,
            kind,
        );
        let core = ObjectCore {
            obj_type: ObjectType::Building,
            geo_center: pick_center,
            radius,
            matrix: glam::Mat4::from_translation(glam::Vec3::new(inst.x, inst.y, inst.build_z)),
            ..Default::default()
        };

        Self {
            core,
            rchange: MR_ALL, // m_RChange(0xffffffff)
            object_state: 0,
            ablaze_ttl: 0,
            shorted_ttl: 0,

            pos: glam::Vec2::new(inst.x, inst.y),
            angle: inst.angle as i32,
            side: inst.side as i32,
            kind,
            state: BaseState::Closing, // MatrixObjectBuilding.cpp:43
            base_floor: 0.2,           // MatrixObjectBuilding.cpp:42
            build_z: inst.build_z,

            hit_point: 0.0,
            hit_point_max: 0.0,
            max_hit_point_inv: 0.0,

            def_hit_point: 0, // MatrixObjectBuilding.cpp:56

            // Pre-mounted factory cannons (prop=1 slots) count toward
            // `turrets_have` at load (MatrixMapPrepare.cpp:849).
            turrets_have: inst
                .turret_places
                .iter()
                .filter(|p| p.cannon_type > 0)
                .count() as i32,
            turrets_max: 4, // default per-kind turret cap
            turret_places: inst
                .turret_places
                .iter()
                .map(|p| {
                    let scale = GameMap::GLOBAL_SCALE_MOVE;
                    TurretPlace {
                        coord: p.coord,
                        world: glam::Vec2::new(p.coord.0 as f32 * scale, p.coord.1 as f32 * scale),
                        angle: p.angle,
                        cannon_type: p.cannon_type,
                    }
                })
                .collect(),
            under_attack_time: 0,
            capture_me_next_time: 0,
            capture_seek_robot_next_time: 0,
            in_capture_time: 0,
            in_capture_next_time_erase: 0,
            in_capture_next_time_paint: 0,
            capture_next_time_rollback: 0,
            true_color: TrueColor::default(),
            zahvat: None,
            resource_period: 0, // MatrixObjectBuilding.cpp:53

            shadow_type: 0,   // SHADOW_OFF sentinel; CMAP sets
            shadow_size: 128, // MatrixObjectBuilding.cpp:40
            capturer: None,   // MatrixObjectBuilding.cpp:48
            next_explosion_time: 0,
            next_explosion_time_sound: 0,
            show_hitpoint_time: 0,
            // MatrixObjectBuilding.cpp:43 seeds BASE_CLOSING + default
            // base_floor at 0.2 from the ctor; the actual progress
            // value starts at 0 and animates toward 0 on the first
            // logic tick of a freshly-spawned building.
            base_floor_progress: 0.0,
            build_stack: BuildStack::new(),
            self_id: None,
        }
    }

    /// Entry point for the build-robot UI action — ports the
    /// `m_Base->m_BS.AddItem(m_Build)` call at CConstructor.cpp:219.
    /// Returns true if the queue accepted the item, false if full.
    ///
    /// Accepts a full `RobotConfig` (ports the C++ where `m_Build` is
    /// a fully-configured `CMatrixRobotAI*` with armor/weapons/head
    /// already attached).
    pub fn queue_robot(&mut self, cfg: RobotConfig) -> bool {
        self.build_stack.add_item(PendingItem {
            kind: PendingKind::Robot(cfg),
            side: self.side,
        })
    }

    /// Back-compat entry point for the simplified "just a chassis"
    /// default used by the pre-constructor `buro` handler. Builds a
    /// minimal `RobotConfig` with the chosen chassis + a passive
    /// armor so the validation check passes.
    pub fn queue_default_robot(&mut self, chassis: ChassisKind) -> bool {
        let mut cfg = RobotConfig::new();
        cfg.chassis.kind = kind_from_chassis(chassis);
        cfg.hull.unit.kind = RobotUnitKind::ARMOR_PASSIVE;
        self.queue_robot(cfg)
    }

    /// Queue a turret build on the given free slot of this base.
    /// Returns false if the slot is out of range, already occupied, or
    /// the build queue is full. Ports the turret-build follow-through
    /// of `CInterface::BeginBuildTurret` → `m_BS.AddItem`.
    ///
    /// `slot` is the index into `turret_places[]`. The C++ tracks
    /// occupancy via `m_TurretsPlaces[i].m_CannonType`
    /// (MatrixObjectBuilding.cpp:1795 + 1551); we mirror that — the
    /// slot's `cannon_type` flips from -1 to the requested kind on
    /// reserve, and is cleared back to -1 on cannon destruction.
    pub fn queue_turret_slot(&mut self, slot: i32, turret_kind: i32) -> bool {
        if self.turrets_have >= self.turrets_max {
            return false;
        }
        let idx = slot as usize;
        let Some(place) = self.turret_places.get_mut(idx) else {
            return false;
        };
        if place.cannon_type > 0 {
            return false; // already occupied
        }
        let ok = self.build_stack.add_item(PendingItem {
            kind: PendingKind::Turret { slot, turret_kind },
            side: self.side,
        });
        if ok {
            // Optimistically reserve the slot — the turret is mounted
            // onto the building immediately (the C++ mounts it before
            // the timer completes; the timer only gates the cost +
            // visuals).
            place.cannon_type = turret_kind;
            self.turrets_have += 1;
        }
        ok
    }

    /// Back-compat shim: pick the first free slot. Used by the older
    /// non-snapping placement path that has no slot index yet.
    pub fn queue_turret(&mut self, turret_kind: i32) -> bool {
        let slot = self
            .turret_places
            .iter()
            .position(|p| p.cannon_type < 0)
            .map(|i| i as i32);
        match slot {
            Some(s) => self.queue_turret_slot(s, turret_kind),
            // No CMAP slots: fall back to the legacy "just append"
            // behaviour so test maps without cannon records still
            // accept turret builds.
            None => {
                if self.turrets_have >= self.turrets_max {
                    return false;
                }
                let s = self.turrets_have;
                let ok = self.build_stack.add_item(PendingItem {
                    kind: PendingKind::Turret {
                        slot: s,
                        turret_kind,
                    },
                    side: self.side,
                });
                if ok {
                    self.turrets_have += 1;
                }
                ok
            }
        }
    }

    /// Entry-point for `CMatrixBuilding::ShowHitpoint` (MatrixObjectBuilding.hpp:272).
    /// Resets the health-bar overlay timer. Called by the attacker's
    /// weapon effect when it connects.
    pub fn show_hitpoint(&mut self) {
        self.show_hitpoint_time = crate::matrix_game::common::HITPOINT_SHOW_TIME_MS;
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
        if self.object_state & crate::matrix_game::map_static::OBJECT_STATE_BUILDING_SPAWNBOT != 0 {
            return;
        }
        self.state = BaseState::Closing;
    }

    /// `IsBase()` (MatrixObjectBuilding.hpp:189).
    pub fn is_base(&self) -> bool {
        self.kind == BuildingType::Base
    }

    /// `CanBeCaptured()` (MatrixObjectBuilding.hpp:191).
    pub fn can_be_captured(&self) -> bool {
        self.object_state
            & crate::matrix_game::map_static::OBJECT_STATE_BUILDING_CAPTURE_IN_PROGRESS
            == 0
    }

    /// `SetCapturedBy(ms)` (MatrixObjectBuilding.hpp:193-197).
    pub fn set_captured_by(&mut self, by: Option<ObjectId>) {
        self.object_state |=
            crate::matrix_game::map_static::OBJECT_STATE_BUILDING_CAPTURE_IN_PROGRESS;
        self.capturer = by;
    }

    /// `ResetCaptured()` (MatrixObjectBuilding.hpp:198-202).
    pub fn reset_captured(&mut self) {
        self.object_state &=
            !crate::matrix_game::map_static::OBJECT_STATE_BUILDING_CAPTURE_IN_PROGRESS;
        self.capturer = None;
    }

    /// Port of `CMatrixBuilding::SetNeutral` (MatrixObjectBuilding.cpp:
    /// 1143-1151).
    pub fn set_neutral(&mut self) {
        self.true_color = TrueColor::default();
        self.side = 0;
        self.build_stack.clear();
        self.rchange |= crate::matrix_game::map_static::MR_MINIMAP;
    }

    /// Port of `CMatrixBuilding::Capture(by)` (MatrixObjectBuilding.cpp:
    /// 1154-1263) — one paint/erase step of the capture progress. The
    /// nearest-robot `FindObjects` scan is precomputed by the caller
    /// (`me_nearest`); `by_side` is the capturing robot's side; `now`
    /// is game time in ms.
    /// The second return is the capture-voice key to queue
    /// (`S_ENEMY_FACTORY_CAPTURED` on a neutral factory fully painted
    /// by the player, MatrixObjectBuilding.cpp:1191-1193;
    /// `S_PLAYER_FACTORY_CAPTURED` when a player factory is erased to
    /// neutral, :1248-1249) — returned because `self` is checked out
    /// of the arena, so callers own the sound queue.
    pub fn capture(
        &mut self,
        by_side: i32,
        me_nearest: bool,
        now: i64,
    ) -> (CaptureStatus, Option<&'static str>) {
        let cfg = crate::matrix_game::config::global();
        let erase = cfg.timings.capture_time_erase as i64;
        let paint = cfg.timings.capture_time_paint as i64;

        if self.in_capture_time <= 0 {
            self.in_capture_time = (erase + paint) as i32;
            self.in_capture_next_time_erase = now;
            self.in_capture_next_time_paint = now;
        }

        if self.in_capture_next_time_erase >= now || self.in_capture_next_time_paint >= now {
            return (CaptureStatus::Busy, None);
        }

        if !me_nearest {
            return (CaptureStatus::TooFar, None);
        }

        // C++ `0xFF000000 | g_MatrixMap->GetSideColor(side)` — the
        // real side color: the zahvat ring renders this value
        // directly (MatrixObjectBuilding.cpp:1183).
        let capturer_color =
            0xFF00_0000u32 | crate::matrix_game::map::side_color_u32(by_side);

        self.in_capture_time = (erase + paint) as i32;

        if self.side == 0 {
            if self.true_color.color == 0 || self.true_color.color == capturer_color {
                self.true_color.color = capturer_color;
                while self.in_capture_next_time_paint < now {
                    self.in_capture_next_time_paint += paint;
                    self.in_capture_next_time_erase = now;
                    if self.true_color.colored_cnt == MAX_ZAHVAT_POINTS {
                        let snd = (by_side == PLAYER_SIDE).then_some("s_ef_cap");
                        self.side = by_side;
                        self.build_stack.clear();
                        self.rchange |= crate::matrix_game::map_static::MR_MINIMAP;
                        return (CaptureStatus::Done, snd);
                    }
                    self.true_color.colored_cnt += 1;
                }
            } else {
                while self.in_capture_next_time_erase < now {
                    self.in_capture_next_time_erase += erase;
                    self.in_capture_next_time_paint = now;
                    if self.true_color.colored_cnt == 0 {
                        self.true_color.color = 0;
                        return (CaptureStatus::InProgress, None);
                    }
                    self.true_color.colored_cnt -= 1;
                }
            }
        } else if self.true_color.color == capturer_color {
            while self.in_capture_next_time_paint < now {
                self.in_capture_next_time_paint += paint;
                self.in_capture_next_time_erase = now;
                if self.true_color.colored_cnt == MAX_ZAHVAT_POINTS {
                    // "дозахват" — re-capture by the same color; no
                    // voice (commented out in the C++, :1229-1230).
                    self.side = by_side;
                    self.rchange |= crate::matrix_game::map_static::MR_MINIMAP;
                    return (CaptureStatus::Done, None);
                }
                self.true_color.colored_cnt += 1;
            }
        } else {
            while self.in_capture_next_time_erase < now {
                self.in_capture_next_time_erase += erase;
                self.in_capture_next_time_paint = now + paint;
                if self.true_color.colored_cnt == 0 {
                    let snd = (self.side == PLAYER_SIDE).then_some("s_pf_cap");
                    self.true_color.color = 0;
                    self.side = 0;
                    self.rchange |= crate::matrix_game::map_static::MR_MINIMAP;
                    self.build_stack.clear();
                    return (CaptureStatus::InProgress, snd);
                }
                self.true_color.colored_cnt -= 1;
            }
        }

        (CaptureStatus::InProgress, None)
    }
}

impl MapStatic for Building {
    fn core(&self) -> &ObjectCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut ObjectCore {
        &mut self.core
    }
    /// Ray pick against the building's VO AABB rotated by the 90°-step
    /// yaw — ports the mesh-precision of `CVectorObject::Pick` closely
    /// enough that turret fire lines no longer clip the fat bounding
    /// sphere of their own building. Sphere fallback when the renderer
    /// hasn't registered bounds (headless sims).
    fn pick(&self, origin: glam::Vec3, dir: glam::Vec3) -> Option<f32> {
        use crate::matrix_lib::three_g::vector_object::{
            has_building_boxes, pick_building_boxes,
        };
        if !has_building_boxes(self.kind as u8) {
            return crate::matrix_game::map_static::ray_sphere_pick(
                self.core.geo_center,
                self.core.radius,
                origin,
                dir,
            );
        }
        // World → building-local: untranslate, then rotate by -yaw.
        let ang = (self.angle & 3) as f32 * std::f32::consts::FRAC_PI_2;
        let (s, c) = ang.sin_cos();
        let base = glam::Vec3::new(self.pos.x, self.pos.y, self.core.matrix.w_axis.z);
        let to_local = |v: glam::Vec3| glam::Vec3::new(c * v.x + s * v.y, -s * v.x + c * v.y, v.z);
        let lo = to_local(origin - base);
        let ld = to_local(dir);
        pick_building_boxes(self.kind as u8, lo.to_array(), ld.to_array())
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
            | crate::matrix_game::map_static::MR_MINIMAP);
        let _ = need;
    }

    /// Port of `CMatrixBuilding::Takt` (MatrixObjectBuilding.cpp:351-439).
    /// Noop skeleton: BUILDING_NEW_INCOME → score-billboard (effects),
    /// `m_GGraph->Takt` (per-instance mesh anim), `m_capture` overlay
    /// creation (effects) — every branch needs subsystems not yet
    /// ported. Structure preserved so the body lands in-place later.
    fn takt(&mut self, _cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {}

    fn show_hitpoint(&mut self) {
        Building::show_hitpoint(self);
    }

    /// `CMatrixBuilding::BeforeDraw` PB block (MatrixObjectBuilding.cpp:
    /// 931-948): anchored over the roof (ground + 90 for the base /
    /// + 80 otherwise), PB_BUILDING_WIDTH=200 starting `radius` left
    /// of the projected point.
    fn hitpoint_bar(
        &self,
        map: &crate::matrix_game::map::GameMap,
    ) -> Option<crate::matrix_game::map_static::HpBar> {
        if self.show_hitpoint_time <= 0
            || self.hit_point <= 0.0
            || matches!(self.state, BaseState::Dip | BaseState::DipExploded)
        {
            return None;
        }
        let geo = self.core.geo_center;
        let anchor = if self.kind == BuildingType::Base {
            // `GetGeoCenter() - m_Matrix._21/22 * 20` — the building
            // world matrix here is translation-only, so rebuild the
            // rotated y-axis from the quarter-turn angle.
            let ang = (self.angle & 3) as f32 * std::f32::consts::FRAC_PI_2;
            let (s, c) = ang.sin_cos();
            let y_axis = glam::Vec2::new(-s, c);
            let p = glam::Vec2::new(geo.x - y_axis.x * 20.0, geo.y - y_axis.y * 20.0);
            glam::Vec3::new(p.x, p.y, map.get_z(p.x, p.y) + 90.0)
        } else {
            glam::Vec3::new(geo.x, geo.y, map.get_z(geo.x, geo.y) + 80.0)
        };
        Some(crate::matrix_game::map_static::HpBar {
            anchor,
            width: 200.0,
            fill: (self.hit_point / self.hit_point_max.max(1.0)).clamp(0.0, 1.0),
            // The C++ starts the bar `GetRadius()` (mesh-AABB
            // half-diagonal, ~75-100 for buildings) left of the
            // anchor; our core radius is the smaller selection-ring
            // value, which visibly right-shifts the 200px bar.
            // Center it instead.
            x_off: -100.0,
            y_off: 0.0,
        })
    }

    /// Port of `CMatrixBuilding::LogicTakt` (MatrixObjectBuilding.cpp:495-891):
    /// build-stack tick, capture ring + rollback + candidate loop,
    /// under-attack / show-hitpoint timers, the `m_BaseFloor`
    /// BASE_OPENING/CLOSING animation, and the DIP explosion sequence
    /// ending in the ruin swap. Per-side resource payouts are
    /// centralised in `MapLogic::accrue_resources`.
    fn logic_takt(&mut self, cms: i32, _rng: &mut Rnd, objs: &mut Objects) {
        use crate::matrix_game::common::BASE_FLOOR_SPEED;

        // Deferred `CBuildStack::DeleteItem` side effects
        // (MatrixObjectBuilding.cpp:1844-1880): refund each dropped
        // item's cost to its side and tear down queued-turret ghosts
        // (UnjoinGroup + StaticDelete → free slot, un-count turret).
        if !self.build_stack.pending_teardown.is_empty() {
            let items: Vec<(PendingItem, bool)> =
                self.build_stack.pending_teardown.drain(..).collect();
            for (it, refund) in items {
                match it.kind {
                    PendingKind::Robot(cfg) => {
                        if refund && it.side != 0 {
                            let price =
                                crate::matrix_game::interface::constructor::cost_of_config(&cfg);
                            objs.pending_refunds.push((it.side, price.resources));
                        }
                    }
                    PendingKind::Turret { slot, turret_kind } => {
                        if refund && it.side != 0 {
                            let price =
                                crate::matrix_game::config::global().turrets.cost_of(turret_kind);
                            objs.pending_refunds.push((it.side, price.resources));
                        }
                        // Silent ghost delete (the C++ StaticDelete +
                        // ReleaseMe slot bookkeeping).
                        if let Some(me) = self.self_id {
                            let ghost = objs.iter_units().find(|&oid| {
                                crate::matrix_game::logic::cannon_ref(objs, oid)
                                    .map(|c| {
                                        c.parent == Some(me)
                                            && c.slot == slot
                                            && matches!(
                                                c.state,
                                                crate::matrix_game::object_cannon::CannonState::UnderConstruction
                                            )
                                    })
                                    .unwrap_or(false)
                            });
                            if let Some(gid) = ghost {
                                objs.remove_deferred(gid);
                            }
                        }
                        if let Some(p) = self.turret_places.get_mut(slot as usize) {
                            p.cannon_type = -1;
                        }
                        if self.turrets_have > 0 {
                            self.turrets_have -= 1;
                        }
                    }
                }
            }
        }

        // Build-stack tick — port of `m_BS.TickTimer(cms)` call at
        // MatrixObjectBuilding.cpp:601-605: only for a live, owned
        // building (a dying/DIP factory must not finish its queue).
        let parent_pos = glam::Vec3::new(self.pos.x, self.pos.y, self.build_z);
        let can_build = self.side != 0
            && !matches!(self.state, BaseState::Dip | BaseState::DipExploded);
        if let Some(parent_id) = self.self_id.filter(|_| can_build) {
            if let Some(spawned) = self
                .build_stack
                .tick_timer(cms, objs, parent_id, self.state, parent_pos, self.angle)
            {
                log::debug!(
                    "build: factory side={} produced robot {:?} at ({:.0}, {:.0}, {:.0})",
                    self.side,
                    spawned,
                    parent_pos.x,
                    parent_pos.y,
                    parent_pos.z,
                );
                // Port of MatrixRobot.cpp:2183 + 2228 — set the
                // BUILDING_SPAWNBOT flag and open the base so the
                // platform starts rising with the robot on top.
                self.object_state |= crate::matrix_game::map_static::OBJECT_STATE_BUILDING_SPAWNBOT;
                self.open();
                // STAT_ROBOT_BUILD (MatrixRobot.cpp:2224).
                if self.side != 0 {
                    objs.inc_side_stat(self.side, |s| s.robot_build += 1);
                }
            }
        }

        // Zahvat capture ring — lazy creation for non-base factories
        // (MatrixObjectBuilding.cpp:410-437) + per-takt repaint (:392).
        if self.zahvat.is_none()
            && !self.is_base()
            && !matches!(self.state, BaseState::Dip | BaseState::DipExploded)
        {
            // The C++ branches on the core matrix Y axis; the four
            // cases map 1:1 onto the placement angle quadrant.
            let deg: f32 = match self.angle & 3 {
                0 => 102.0,
                1 => 192.0,
                2 => -79.0,
                _ => 12.0,
            };
            let (s, c) = ((self.angle & 3) as f32 * std::f32::consts::FRAC_PI_2).sin_cos();
            // Y axis of Rz(angle*90°) = (-sin, cos).
            let yaxis = glam::Vec2::new(-s, c);
            let pos = glam::Vec3::new(
                self.pos.x + yaxis.x * 2.7,
                self.pos.y + yaxis.y * 2.7,
                self.build_z + 0.8,
            );
            self.zahvat = Some(crate::matrix_game::effects::zahvat::Zahvat::new(
                pos,
                24.0,
                deg.to_radians(),
                MAX_ZAHVAT_POINTS as usize,
            ));
            if self.side == 0 {
                self.true_color = TrueColor::default();
            } else {
                self.true_color.color =
                    0xFF00_0000 | crate::matrix_game::map::side_color_u32(self.side);
                self.true_color.colored_cnt = MAX_ZAHVAT_POINTS;
            }
        }
        if let Some(z) = self.zahvat.as_mut() {
            z.update_data(self.true_color.color, self.true_color.colored_cnt);
        }

        // Pre-DIP pass: countdowns + capture announce / seek / rollback
        // (MatrixObjectBuilding.cpp:503-594).
        if !matches!(self.state, BaseState::Dip | BaseState::DipExploded) {
            use crate::matrix_game::robot::Robot;
            let now = crate::matrix_game::map::current_elapsed_ms();

            // :506-527 — announce ourselves as a capture candidate to
            // enemy robots nearby, and retract from robots that fell
            // out of range. (AI robots act on the candidates; player
            // robots ignore them outside automatic mode.)
            if (self.capture_me_next_time as i64) < now
                && self.capturer.is_none()
                && !(self.is_base() && self.turrets_have > 0)
            {
                self.capture_me_next_time = (now + 1002) as i32;
                if let Some(self_id) = self.self_id {
                    let my_side = self.side;
                    let my_pos = self.pos;
                    let live: Vec<_> = objs.iter_units().collect();
                    for id in live {
                        let Some(o) = objs.get_mut(id) else { continue };
                        if !matches!(o.core().obj_type, ObjectType::RobotAi) {
                            continue;
                        }
                        let r: &mut Robot =
                            unsafe { &mut *(o as *mut dyn MapStatic as *mut Robot) };
                        if !r.is_live() {
                            continue;
                        }
                        r.object_state_clear(crate::matrix_game::map_static::ROBOT_CAPTURE_INFORMED);
                        let gc = r.core().geo_center;
                        let d2 = (gc.x - my_pos.x).powi(2) + (gc.y - my_pos.y).powi(2);
                        if d2 < DISTANCE_CAPTURE_ME * DISTANCE_CAPTURE_ME && r.side != my_side {
                            r.add_capture_candidate(self_id, _rng);
                        }
                        if r.object_state() & crate::matrix_game::map_static::ROBOT_CAPTURE_INFORMED
                            == 0
                        {
                            r.remove_capture_candidate(self_id);
                        }
                    }
                }
            }

            // MatrixObjectBuilding.cpp:529 — under-attack warning
            // timer decays; floor at 0.
            self.under_attack_time = (self.under_attack_time - cms).max(0);

            // MatrixObjectBuilding.cpp:531-535 — health-bar overlay.
            if self.show_hitpoint_time > 0 {
                self.show_hitpoint_time = (self.show_hitpoint_time - cms).max(0);
            }

            if self.kind != BuildingType::Base {
                // :537-565 — proactive capture: the nearest enemy robot
                // standing in CAPTURE_RADIUS captures without an order.
                if self.capture_seek_robot_next_time < now {
                    let mut found: Option<(crate::matrix_game::map_static::ObjectId, i32)> = None;
                    let mut best = CAPTURE_RADIUS * CAPTURE_RADIUS;
                    for id in objs.iter_units() {
                        let Some(o) = objs.get(id) else { continue };
                        if !matches!(o.core().obj_type, ObjectType::RobotAi) {
                            continue;
                        }
                        if !o.is_live() {
                            continue;
                        }
                        let gc = o.core().geo_center;
                        let d2 = (gc.x - self.pos.x).powi(2) + (gc.y - self.pos.y).powi(2);
                        if d2 < best {
                            best = d2;
                            found = Some((id, o.side()));
                        }
                    }
                    if let Some((_, rside)) = found.filter(|(_, s)| *s != self.side) {
                        // The found robot IS the nearest within radius,
                        // so the Capture() inner scan succeeds.
                        let (_, snd) = self.capture(rside, true, now);
                        if let Some(snd) = snd {
                            objs.queue_snd(snd);
                        }
                        let cfg = crate::matrix_game::config::global();
                        let nt = 100
                            .min(cfg.timings.capture_time_erase)
                            .min(cfg.timings.capture_time_paint);
                        self.capture_seek_robot_next_time = now + nt as i64;
                    } else {
                        self.capture_seek_robot_next_time = now + CAPTURE_SEEK_ROBOT_PERIOD;
                    }
                }

                // :566-594 — capture-progress decay and rollback.
                if self.in_capture_time > 0 {
                    self.in_capture_time -= cms;
                    if self.in_capture_time <= 0 {
                        self.in_capture_time = 0;
                        self.capture_next_time_rollback = now;
                    }
                } else {
                    let rollback =
                        crate::matrix_game::config::global().timings.capture_time_rollback as i64;
                    if self.side == 0 && self.true_color.colored_cnt > 0 {
                        while self.capture_next_time_rollback < now
                            && self.true_color.colored_cnt > 0
                        {
                            self.capture_next_time_rollback += rollback;
                            self.true_color.colored_cnt -= 1;
                        }
                    } else if self.side != 0 && self.true_color.colored_cnt < MAX_ZAHVAT_POINTS {
                        while self.capture_next_time_rollback < now
                            && self.true_color.colored_cnt < MAX_ZAHVAT_POINTS
                        {
                            self.capture_next_time_rollback += rollback;
                            self.true_color.colored_cnt += 1;
                        }
                    }
                }
            }
            // Per-side resource payout (:600-667) is centralised in
            // `MapLogic::accrue_resources`.
        }

        // DIP explosion sequence (MatrixObjectBuilding.cpp:672-808):
        // `m_HitPoint` counts down as a timer; explosions pop every
        // BUILDING_EXPLOSION_PERIOD (10ms) at random points within the
        // radius, 4% of them the bigger Boom2. A dying BASE finishes
        // with a real WEAPON_BIGBOOM blast, then (like all kinds) the
        // building swaps to ruin map-objects and StaticDeletes itself.
        // The C++ keeps decrementing through BUILDING_DIP_EXPLODED
        // (the `m_HitPoint < 0` branch at :762 isn't state-gated).
        if matches!(self.state, BaseState::Dip | BaseState::DipExploded) {
            self.hit_point -= cms as f32;
            let now = crate::matrix_game::map::current_elapsed_ms();
            if self.next_explosion_time == 0 {
                self.next_explosion_time = now as i32;
            }
            let mut vrng = crate::matrix_game::logic::Rnd::new(
                ((now as i32) ^ ((self.pos.x + self.pos.y) as i32)).max(1),
            );
            use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd};
            // Rolling boom sound at IRND(100..500) ms
            // (m_NextExplosionTimeSound, MatrixObjectBuilding.cpp:768-772).
            if self.next_explosion_time_sound == 0 {
                self.next_explosion_time_sound = now as i32;
            }
            while now > self.next_explosion_time_sound as i64 {
                self.next_explosion_time_sound += 100 + (frnd(&mut vrng, 400.0) as i32);
                objs.queue_snd_at("expl_bb", self.core.geo_center);
            }
            let radius = self.core.radius.max(20.0);
            // Aim point sits 60 units behind the building along its
            // facing (`m_Pos - m_Core->m_Matrix._21/22 * 60`, :782-784);
            // candidates scatter around it and a 4-attempt Pick puts
            // the explosion on the mesh surface (+2 along the ray).
            let pos0 = glam::Vec3::new(
                self.pos.x - self.core.matrix.y_axis.x * 60.0,
                self.pos.y - self.core.matrix.y_axis.y * 60.0,
                self.core.matrix.w_axis.z,
            );
            while now > self.next_explosion_time as i64 {
                self.next_explosion_time += 10; // BUILDING_EXPLOSION_PERIOD
                let mut found = None;
                for _ in 0..4 {
                    let pos = pos0
                        + glam::Vec3::new(
                            fsrnd(&mut vrng, radius),
                            fsrnd(&mut vrng, radius),
                            frnd(&mut vrng, 2.0 * radius),
                        );
                    let dir = (pos0 - pos).normalize_or_zero();
                    if dir == glam::Vec3::ZERO {
                        continue;
                    }
                    if let Some(t) = crate::matrix_game::map_trace::pick_sphere(
                        pos,
                        dir,
                        self.core.geo_center,
                        self.core.radius,
                    ) {
                        found = Some(pos + dir * (t + 2.0));
                        break;
                    }
                }
                let Some(pos) = found else { continue };
                let props = if frnd(&mut vrng, 1.0) < 0.04 {
                    &crate::matrix_game::effects::explosion::EXPLOSION_BUILDING_BOOM2
                } else {
                    &crate::matrix_game::effects::explosion::EXPLOSION_BUILDING_BOOM
                };
                objs.pending_explosions
                    .push(crate::matrix_game::map_static::ExplosionSpawn {
                        pos,
                        props,
                        fire: false,
                    });
            }
            // downtime = -1000 (-3000 more for BASE) —
            // MatrixObjectBuilding.cpp:672-692.
            let downtime = if self.kind == BuildingType::Base {
                -1000.0 - 2000.0
            } else {
                -1000.0
            };
            if self.kind == BuildingType::Base
                && self.state == BaseState::Dip
                && self.hit_point < downtime + 100.0
            {
                // The base's final blast is a REAL bigboom (:681-687).
                if let (Some(self_id), Some(map)) =
                    (self.self_id, crate::matrix_game::map::current_map())
                {
                    use crate::matrix_game::effects::weapon::{
                        weapon_takt, WeaponEffect, WeaponHandler, WEAPON_BIGBOOM,
                    };
                    // S_EXPLOSION_BUILDING_BOOM4 (MatrixObjectBuilding.cpp:679).
                    objs.queue_snd_at("expl_bb4", self.core.geo_center);
                    let mut w = WeaponEffect::new(WEAPON_BIGBOOM, 0, WeaponHandler::None);
                    w.set_owner(self_id, self.side);
                    w.modify(self.core.geo_center, glam::Vec3::Z, glam::Vec3::ZERO);
                    let wid = objs.weapons.create(w);
                    if let Some(we) = objs.weapons.get_mut(wid) {
                        we.fire_begin(glam::Vec3::ZERO, Some(self_id));
                    }
                    weapon_takt(objs, wid, 1.0, map, _rng);
                    if let Some(we) = objs.weapons.get_mut(wid) {
                        we.fire_end();
                    }
                    objs.weapons.release(wid);
                }
                self.state = BaseState::DipExploded;
            }
            if self.hit_point < downtime {
                // Ruin swap + StaticDelete (MatrixObjectBuilding.cpp:
                use crate::matrix_game::object::MapObject;
                let n = self.kind as i32;
                if n != 0 {
                    // S_EXPLOSION_BUILDING_BOOM3 on the ruin swap for
                    // non-BASE kinds (MatrixObjectBuilding.cpp:695).
                    objs.queue_snd_at("expl_bb3", self.core.geo_center);
                }
                let z = crate::matrix_game::map::current_map()
                    .map(|m| m.get_z(self.pos.x, self.pos.y))
                    .unwrap_or(self.core.matrix.w_axis.z);
                // `OBJECT_PATH_BUILDINGS_RUINS + "b" + n` (:710-716);
                // non-BASE kinds add the `b<n>p.vo` part rendered with
                // the `?Trans` alpha-tested skin (:718-724).
                let namet = format!("Matrix/Building/ruins/b{n}");
                objs.spawn(Box::new(MapObject::init_as_base_ruins(
                    self.pos,
                    z,
                    self.angle,
                    &format!("{namet}.vo"),
                )));
                if n != 0 {
                    objs.spawn(Box::new(MapObject::init_as_base_ruins(
                        self.pos,
                        z,
                        self.angle,
                        &format!("{namet}p.vo"),
                    )));
                }
                // The 20-50 temporary smoke spawners scattered over the
                // ruin (:726-755). The C++ ray-picks the ruin mesh for
                // exact placement; the mesh is render-side here, so the
                // drain approximates with the building's radius.
                objs.pending_ruin_smoke.push((
                    glam::Vec3::new(self.pos.x, self.pos.y, z),
                    self.core.radius.max(20.0),
                ));
                if let Some(id) = self.self_id {
                    // `g_MatrixMap->StaticDelete(this)` (:757); deferred
                    // because our box is checked out by the takt driver.
                    objs.remove_deferred(id);
                }
                return;
            }
        }

        // BUILDING_BASE platform animation (MatrixObjectBuilding.cpp:
        // 810-854). Runs for BASE buildings only — the other kinds
        // have no open/close animation.
        if self.kind == BuildingType::Base {
            let old = self.base_floor_progress;

            if self.state == BaseState::Opening {
                self.base_floor_progress += BASE_FLOOR_SPEED * cms as f32;
                if self.base_floor_progress >= 1.0 {
                    self.base_floor_progress = 1.0;
                    self.state = BaseState::Opened;
                    // S_PLATFORM_UP_STOP (MatrixObjectBuilding.cpp:821).
                    objs.queue_snd_at("base_platform_up_stop", self.core.geo_center);
                }
            }
            if self.state == BaseState::Closing {
                self.base_floor_progress -= BASE_FLOOR_SPEED * cms as f32;
                if self.base_floor_progress <= 0.0 {
                    self.base_floor_progress = 0.0;
                    self.state = BaseState::Closed;
                    // S_DOORS_CLOSE_STOP (MatrixObjectBuilding.cpp:831).
                    objs.queue_snd_at("base_doors_close_stop", self.core.geo_center);
                }
            }

            // When the progress changed, the C++ also nudges the
            // sub-unit matrices on `m_GGraph` (platform + door
            // translations) and flags `MR_Matrix` dirty. Those
            // per-unit matrices live on the render-side
            // CVectorObjectGroup in the Rust port; flag
            // `MR_MATRIX` so the next `r_need` rebuilds the object
            // transform (even though the per-unit nudges are still
            // deferred).
            if (self.base_floor_progress - old).abs() > f32::EPSILON {
                self.rchange |= crate::matrix_game::map_static::MR_MATRIX;
            }
        }
    }

    /// Port of `CMatrixBuilding::Damage` (MatrixObjectBuilding.cpp:254-344).
    ///
    /// Implements: already-DIP early-out, friendly-fire detection,
    /// WEAPON_REPAIR heal path, `mindamage`-floored HP decrement with
    /// `friend_damage` column selection, HP≤0 → BASE → DIP transition
    /// with explosion-sequence timers primed at 0.
    ///
    /// Deferred: difficulty scaling (k_damage_enemy_to_player /
    /// k_friendly_fire), sound effects, per-side kill-stat increments,
    /// effect-spawner cleanup (`RemoveEffectSpawnerByObject`),
    /// `ReleaseMe` side-resource unbinding, and the progress-bar
    /// update on HP change.
    ///
    /// Returns `true` iff the call destroyed the building (matches the
    /// original's contract used by attacker-side code to stop tracking
    /// a now-dead target).
    fn damage(
        &mut self,
        weap: crate::matrix_game::effects::weapon::Weapon,
        _pos: glam::Vec3,
        _dir: glam::Vec3,
        attacker_side: i32,
        _attacker: Option<ObjectId>,
        _self_id: ObjectId,
        objs: &mut Objects,
    ) -> bool {
        use crate::matrix_game::effects::weapon::WEAPON_REPAIR;

        // MatrixObjectBuilding.cpp:258 — already dying? ignore.
        if matches!(self.state, BaseState::Dip | BaseState::DipExploded) {
            return true;
        }

        // Friendly-fire iff the attacker has a side and matches ours
        // (side 0 = neutral / world — not flagged as friendly).
        let friendly_fire = attacker_side != 0 && attacker_side == self.side;

        let entry = objs.building_damages.get(weap).unwrap_or_default();

        if weap == WEAPON_REPAIR {
            // MatrixObjectBuilding.cpp:265-276 — REPAIR restores HP,
            // clamped to max. friendly_fire selects the `friend_damage`
            // column (which in MatrixConfig defaults to `damage` when
            // absent).
            let amount = if friendly_fire {
                entry.friend_damage
            } else {
                entry.damage
            };
            self.hit_point = (self.hit_point + amount as f32).min(self.hit_point_max);
            // No m_PB.Modify needed — `hitpoint_bar` reads live HP.
            return false;
        }

        // `damagek` — difficulty scaling (MatrixObjectBuilding.cpp:
        // 261-262): only enemy-vs-player uses the scale factor; a
        // friendly hit on the player additionally scales by the
        // friendly-fire coefficient.
        let difficulty = crate::matrix_game::config::global().difficulty;
        let mut damagek = if friendly_fire || self.side != PLAYER_SIDE {
            1.0f32
        } else {
            difficulty.k_damage_enemy_to_player
        };
        if friendly_fire && self.side == PLAYER_SIDE {
            damagek *= difficulty.k_friendly_fire;
        }

        // MatrixObjectBuilding.cpp:281-292.
        if self.hit_point > entry.mindamage as f32 {
            let base = if friendly_fire {
                entry.friend_damage
            } else {
                entry.damage
            };
            self.hit_point -= damagek * base as f32;
            // `hitpoint_bar` reads live HP; just reseed the linger timer.
            self.show_hitpoint();
        }

        // MatrixObjectBuilding.cpp:294-304 — under-attack warning,
        // throttled by the idle timer.
        if self.side == PLAYER_SIDE && !friendly_fire {
            if self.under_attack_time == 0 {
                let pick = (self.pos.x as i32 ^ self.hit_point as i32).rem_euclid(3) + 1;
                objs.queue_snd(&format!("s_side_attacked_{pick}"));
            }
            self.under_attack_time = UNDER_ATTACK_IDLE_TIME_MS;
        }

        // MatrixObjectBuilding.cpp:308-341 — death transition.
        if self.hit_point <= 0.0 {
            // Death voice (player side, :317-324): base / factory
            // 70%-or-30% generic split.
            if self.side == PLAYER_SIDE {
                let key = if self.is_base() {
                    "s_base_dead"
                } else if (self.pos.y as i32) % 10 < 7 {
                    "s_fa_dead"
                } else {
                    "s_building_dead"
                };
                objs.queue_snd(key);
            }
            if attacker_side != 0 && !friendly_fire {
                objs.inc_side_stat(attacker_side, |s| s.building_kill += 1);
            }
            self.hit_point = -1.0;
            self.state = BaseState::Dip;
            // Schedule the first explosion to fire on the next logic
            // takt that reads the DIP state (logic_takt's DIP branch
            // still needs effects). `0` = "fire as soon as possible".
            self.next_explosion_time = 0;
            self.next_explosion_time_sound = 0;

            // ~CBuildStack (MatrixObjectBuilding.cpp:1882-1919):
            // queued items are destroyed with no refund; the ghost
            // teardown runs on the next logic takt.
            self.build_stack.clear_no_refund();

            // ReleaseMe cascade (MatrixObjectBuilding.cpp:1364-1431):
            // robots spawning here die, capture state referencing this
            // building clears, child cannons orphan.
            if let Some(me) = self.self_id {
                let ids: Vec<crate::matrix_game::map_static::ObjectId> =
                    objs.iter_units().collect();
                objs.pending_env_purge.push(me);
                for oid in ids {
                    if let Some(r) = crate::matrix_game::logic::robot_mut(objs, oid) {
                        if r.base == Some(me) {
                            r.base = None;
                            r.must_die = true;
                        }
                        r.remove_capture_candidate(me);
                        r.env.remove_from_list(me);
                        if r.get_capture_factory() == Some(me) {
                            r.stop_capture();
                        }
                    } else if let Some(c) = crate::matrix_game::logic::cannon_mut(objs, oid) {
                        if c.parent == Some(me) {
                            c.parent = None;
                            // Queued ghost turret: ~CBuildStack kills it
                            // outright (WEAPON_INSTANT_DEATH,
                            // MatrixObjectBuilding.cpp:1897/1918) — it
                            // must not survive as an invulnerable prop.
                            if matches!(
                                c.state,
                                crate::matrix_game::object_cannon::CannonState::UnderConstruction
                            ) {
                                objs.remove_deferred(oid);
                            }
                        }
                    }
                }
            }
            self.zahvat = None;
            return true;
        }

        false
    }

    fn side(&self) -> i32 {
        self.side
    }
    fn need_repair(&self) -> bool {
        self.hit_point < self.hit_point_max
    }
    fn is_live(&self) -> bool {
        !matches!(self.state, BaseState::Dip | BaseState::DipExploded)
    }
}

/// `UNDER_ATTACK_IDLE_TIME` (MatrixMapStatic.hpp:43). After any enemy
/// hit on a player-owned building, the "under attack" flag stays lit
/// for 120 seconds so the warning sound doesn't retrigger on every
/// bullet.
pub const UNDER_ATTACK_IDLE_TIME_MS: i32 = 120_000;

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
    /// Per-sub-unit translation applied in mesh-local space before
    /// the world matrix. Port of the per-unit
    /// `D3DXMatrixTranslation` updates at
    /// MatrixObjectBuilding.cpp:842-850. Zero-filled for all
    /// sub-units on non-BASE kinds + for the "body" sub-unit on
    /// BASE. Populated for BASE sub-unit IDs 1/2/3 (platform / left
    /// door / right door) each frame by `BuildingsRenderer::takt`.
    unit_offset: [f32; 4],
    /// Per-side diffuse colour applied on top of the texture —
    /// ports `CMatrixBuilding::Draw`'s `m_GGraph->Draw(coltex)` where
    /// `coltex` is `GetSideColorTexture(m_Side)` (MatrixObjectBuilding.cpp:
    /// 968-971). The original only modulates specific team-marker
    /// surfaces (selected via CVO material flags); without the skin
    /// stage info in the Rust port we tint the whole building at
    /// reduced strength in the shader to keep each faction's bases
    /// visually distinct.
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
    /// Per-instance liveness: flips true once the arena `Building` was
    /// StaticDelete'd at the end of its DIP sequence
    /// (MatrixObjectBuilding.cpp:757) — the instance then collapses to a
    /// degenerate transform and the ruin batch takes over.
    hidden: Vec<bool>,
    center: [f32; 2],
    /// Kind of building this batch's mesh belongs to. Needed so the
    /// per-frame animation takt only touches BUILDING_BASE instances.
    kind: BuildingType,
    /// Sub-unit id from the CVO (`Id` param at
    /// VectorObject.cpp:2415-2553). BASE's platform is id=1, left
    /// door 2, right door 3; other sub-units and non-BASE kinds get
    /// `None` here.
    unit_id: Option<i32>,
}

/// Preloaded ruin mesh (`Matrix\Building\ruins\b<n>[p].vo`,
/// MatrixObjectBuilding.cpp:707-724). Instances start empty; the sync pass
/// fills them from arena `MapObject`s whose `ruin_graph` matches `vo_path`.
struct RuinBatch {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    instance_buffer: wgpu::Buffer,
    /// Buffer slots — one per building of this kind on the map (every
    /// DIP death produces at most one ruin pair).
    capacity: u32,
    num_instances: u32,
    bind_group: wgpu::BindGroup,
    vo_path: String,
}

/// Degenerate transform for a dead / not-yet-spawned instance — every
/// vertex collapses to one point so nothing rasterizes.
fn hidden_instance() -> InstanceData {
    InstanceData {
        row0: [0.0; 4],
        row1: [0.0; 4],
        row2: [0.0; 4],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [0.0; 4],
        unit_offset: [0.0; 4],
        side_color: [0.0; 4],
    }
}

pub struct BuildingsRenderer {
    pipeline: wgpu::RenderPipeline,
    batches: Vec<MeshBatch>,
    /// Ruin meshes per building kind on the map, preloaded so the DIP
    /// death swap needs no runtime VO loading.
    ruin_batches: Vec<RuinBatch>,
    uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    /// Cached `m_ShadowColor` from the map (DATA_SHADOWCOLOR), packed as
    /// 0xAARRGGBB and forwarded to `ShadowSystem::update_view` each frame.
    shadow_color: u32,
    time_ms: f32,
    last_point_light_revision: u64,
    /// Projected shadow infrastructure (pipelines, sampler, shared UB).
    shadow_system: ShadowSystem,
    /// One ground-projection mesh per building instance. Built once at spawn —
    /// buildings don't move, so the geometry doesn't need to refresh per frame.
    shadow_batches: Vec<ShadowBatch>,
    /// Pos-key of the building owning each `shadow_batches` entry —
    /// lets the sync hide the shadow when the building dies.
    shadow_keys: Vec<(i32, i32)>,
    shadow_visible: Vec<bool>,
    /// Latched once the arena reports any live `Building`; before that the
    /// "missing from arena ⇒ dead" rule must stay off (render-only flows
    /// never spawn logic buildings).
    logic_buildings_seen: bool,
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

        // Per-kind pooled mesh data used to bake the silhouette texture and
        // size the projector. Concatenates every sub-unit's frame-0 mesh into
        // a single buffer (sub-units share the building's local frame at rest;
        // the floor / door slide animations are runtime-only so the static
        // bake at frame-0 matches the original's `m_GroupVO`-sourced shadow).
        struct ShadowKindMesh {
            vertices: Vec<ShadowMeshVertex>,
            surfaces: Vec<ShadowMeshSurface>,
        }
        let mut shadow_kinds: HashMap<u8, ShadowKindMesh> = HashMap::new();

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

            let base_inst_data: Vec<InstanceData> = instances
                .iter()
                .map(|b| instance_matrix(b, cx, cy, map, None))
                .collect();
            let kind_enum = BuildingType::from_u8(*kind).unwrap_or(BuildingType::Base);
            let mut pick_boxes: Vec<([f32; 3], [f32; 3])> = Vec::new();

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
                // Collect the sub-VO's local AABB for logic-side pick
                // tests (replaces the sphere approximation).
                pick_boxes.push((frame0.bounds_min, frame0.bounds_max));

                // When the CVO unit declares no `Texture=…`, the original
                // falls back to the VO's own embedded surface texture name —
                // `vo->GetSurfaceFileName(0)` (VectorObject.cpp:2509). We
                // read that from the surface's `texture_ref` instead, and
                // otherwise keep the unit's declared diffuse. Gloss/mask/back
                // overrides from the CVO still win, mirroring the composed
                // skin the original builds at VectorObject.cpp:2513.
                let cvo_dir = cvo_path.rsplit_once('/').map(|(d, _)| format!("{d}/"));
                // Silhouette source — only the first unit in the CVO group
                // contributes. Filtering on `id` alone is not enough: e.g.
                // `b0.cvo` (BASE) lists `b0_anim.vo` (the landing pod) as an
                // anonymous unit with no `Id`, and `b1.cvo`/`b2.cvo`/`b4.cvo`
                // each have an anonymous `platform.vo` with no `Id`. Both
                // would pool into the silhouette and project a phantom shadow
                // next to the body. The original's commented-out projected-
                // shadow path uses `m_GGraph->m_Unit[0]` — i.e. the first
                // unit — as the body, so we mirror that.
                let is_body = !shadow_kinds.contains_key(kind);
                let shadow_vertex_offset = if is_body {
                    let shadow_mesh = shadow_kinds.entry(*kind).or_insert_with(|| ShadowKindMesh {
                        vertices: Vec::new(),
                        surfaces: Vec::new(),
                    });
                    let off = shadow_mesh.vertices.len() as u32;
                    shadow_mesh
                        .vertices
                        .extend(vertices.iter().map(|v| ShadowMeshVertex {
                            position: v.position,
                            normal: v.normal,
                            uv: v.uv,
                        }));
                    Some(off)
                } else {
                    None
                };
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
                    // Mirror this body surface into the silhouette source,
                    // re-indexed to point into the pooled vertex array.
                    // Skipped for animated sub-units (platform / doors) —
                    // their `shadow_vertex_offset` is None.
                    if let Some(offset) = shadow_vertex_offset {
                        let shadow_mesh = shadow_kinds
                            .get_mut(kind)
                            .expect("body offset implies kind already inserted");
                        let remapped: Vec<u32> = surf.indices.iter().map(|i| i + offset).collect();
                        shadow_mesh.surfaces.push(ShadowMeshSurface {
                            indices: remapped,
                            diffuse: diffuse_view.clone(),
                            alpha_test,
                        });
                    }
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
                    let index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

                    // Each batch owns its own instance buffer so the
                    // per-sub-unit animation (platform rise, doors
                    // slide) can write different offsets per batch
                    // without stomping on sibling sub-units that
                    // share the kind.
                    let instance_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Buildings Inst VB"),
                            contents: bytemuck::cast_slice(&base_inst_data),
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        });
                    batches.push(MeshBatch {
                        vertex_buffer,
                        index_buffer,
                        num_indices: surf.indices.len() as u32,
                        instance_buffer,
                        num_instances: base_inst_data.len() as u32,
                        bind_group,
                        buildings: instances.iter().map(|b| (*b).clone()).collect(),
                        hidden: vec![false; instances.len()],
                        center: [cx, cy],
                        kind: kind_enum,
                        unit_id: unit.id,
                    });
                }
            }

            vector_object::set_building_boxes(*kind, pick_boxes);
            loaded_kinds += 1;
        }

        // Ruin meshes for every kind present on the map — the DIP death
        // sequence swaps the building for `ruins\b<n>.vo` (+ alpha-tested
        // `b<n>p.vo` for non-BASE kinds) with the `ruins\b<n>` skin
        // override (MatrixObjectBuilding.cpp:707-724). Preloading here
        // keeps the swap a visibility flip — instance data arrives from
        // arena ruin `MapObject`s in `sync_building_animation`.
        let mut ruin_batches = Vec::new();
        let ruins_dir = "Matrix/Building/ruins/";
        for (kind, instances) in &by_kind {
            let variants: &[&str] = if *kind == 0 { &[""] } else { &["", "p"] };
            for variant in variants {
                let vo_path = format!("{ruins_dir}b{kind}{variant}.vo");
                let Some(vo_bytes) = read_texture(&vo_path) else {
                    log::debug!("buildings: ruin VO not found: {}", vo_path);
                    continue;
                };
                let mesh = match vector_object::parse_vo(&vo_bytes) {
                    Ok(m) => m,
                    Err(e) => {
                        log::warn!("buildings: parse ruin {} failed: {}", vo_path, e);
                        continue;
                    }
                };
                let Some(frame0) = mesh.frames.first() else {
                    continue;
                };
                // Ruin-local AABB for logic-side pick tests, keyed by
                // the exact `ruin_graph` path (init_as_base_ruins).
                vector_object::merge_ruin_bounds(
                    &vo_path,
                    frame0.bounds_min,
                    frame0.bounds_max,
                );
                let indices: Vec<u32> = frame0
                    .surfaces
                    .iter()
                    .flat_map(|s| s.indices.iter().copied())
                    .collect();
                if indices.is_empty() {
                    continue;
                }
                let vertices: Vec<Vertex> = mesh
                    .vertices
                    .iter()
                    .map(|v| Vertex {
                        position: v.position,
                        normal: v.normal,
                        uv: v.uv,
                    })
                    .collect();
                // The C++ passes "ruins\b<n>" (+ "?Trans" on the p-part)
                // as a whole-skin texture override to LoadObject.
                let spec = if variant.is_empty() {
                    format!("b{kind}")
                } else {
                    format!("b{kind}?Trans")
                };
                let material = crate::matrix_lib::three_g::texture::parse_material_spec_with_prefix(
                    &spec,
                    Some(ruins_dir),
                );
                let (diffuse_view, alpha_test) =
                    resolve_diffuse(&material, device, queue, &mut tex_cache, read_texture)
                        .unwrap_or_else(|| (fallback_tex.clone(), false));
                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Ruins Mesh VB"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Ruins Mesh IB"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let mat_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Ruins Material UB"),
                    contents: bytemuck::bytes_of(&MaterialUniform {
                        flags: [0, 0, 0, alpha_test as u32],
                        scroll: [0.0, 0.0, 0.0, 0.0],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Ruins BG"),
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
                            resource: wgpu::BindingResource::TextureView(&black_tex),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&black_tex),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&transparent_tex),
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
                let capacity = instances.len() as u32;
                let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Ruins Inst VB"),
                    contents: bytemuck::cast_slice(&vec![hidden_instance(); capacity as usize]),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
                ruin_batches.push(RuinBatch {
                    vertex_buffer,
                    index_buffer,
                    num_indices: indices.len() as u32,
                    instance_buffer,
                    capacity,
                    num_instances: 0,
                    bind_group,
                    vo_path,
                });
            }
        }
        if !ruin_batches.is_empty() {
            log::info!("buildings: {} ruin meshes preloaded", ruin_batches.len());
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

        // Bake one shadow batch per building instance. Buildings don't move,
        // so this is a one-shot at spawn — equivalent to the C++
        // `MR_ShadowProjGeom | MR_ShadowProjTex` request fired off by
        // `CMatrixBuilding::OnLoad`. `map_radius` of 10 cells matches the
        // default the original passes to `ShadowProjBuild` for buildings
        // (MatrixObjectBuilding.cpp:214). Shadow texture size mirrors the
        // building's `m_ShadowSize` (default 128, MatrixObjectBuilding.cpp:40).
        let shadow_system = ShadowSystem::new(device, config);
        let mut shadow_batches: Vec<ShadowBatch> = Vec::new();
        let mut shadow_keys: Vec<(i32, i32)> = Vec::new();
        let light_world = Vec3::new(
            map.light_main_dir[0],
            map.light_main_dir[1],
            map.light_main_dir[2],
        );
        for b in &map.buildings {
            let Some(mesh) = shadow_kinds.get(&b.kind) else {
                continue;
            };
            if mesh.vertices.is_empty() || mesh.surfaces.is_empty() {
                continue;
            }
            let world_matrix = building_world_matrix(b);
            let local_pts: Vec<Vec3> = mesh
                .vertices
                .iter()
                .map(|v| Vec3::from_array(v.position))
                .collect();
            let Some(proj) = shadow_system.calc_proj(&local_pts, light_world, world_matrix) else {
                continue;
            };
            let texture_size = b.shadow_size.max(32) as u32;
            let Some(tex) = shadow_system.bake_texture(
                device,
                queue,
                &mesh.vertices,
                &mesh.surfaces,
                &proj,
                texture_size,
            ) else {
                continue;
            };
            if let Some(batch) =
                shadow_system.build_geometry(device, map, &proj, &tex, 10, [cx, cy])
            {
                shadow_batches.push(batch);
                shadow_keys.push(pos_key(b.x, b.y));
            }
        }
        log::info!(
            "buildings: {} projected shadows built",
            shadow_batches.len()
        );

        let shadow_visible = vec![true; shadow_batches.len()];
        Some(Self {
            pipeline,
            batches,
            ruin_batches,
            uniform_buffer,
            fog_color,
            ambient_color,
            light_color,
            light_dir,
            shadow_color: map.shadow_color,
            time_ms: 0.0,
            last_point_light_revision: 0,
            shadow_system,
            shadow_batches,
            shadow_keys,
            shadow_visible,
            logic_buildings_seen: false,
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
            // Skip batches with no building near the relit window.
            let [rx0, ry0, rx1, ry1] = point_lights.changed_rect();
            let gs = crate::matrix_game::map::GLOBAL_SCALE;
            let wx0 = (rx0 - 1) as f32 * gs;
            let wy0 = (ry0 - 1) as f32 * gs;
            let wx1 = (rx1 + 1) as f32 * gs;
            let wy1 = (ry1 + 1) as f32 * gs;
            for batch in &mut self.batches {
                let any_in_window = batch
                    .buildings
                    .iter()
                    .any(|b| b.x >= wx0 && b.x <= wx1 && b.y >= wy0 && b.y <= wy1);
                if !any_in_window {
                    continue;
                }
                let [cx, cy] = batch.center;
                let inst_data: Vec<InstanceData> = batch
                    .buildings
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        if batch.hidden[i] {
                            hidden_instance()
                        } else {
                            instance_matrix(b, cx, cy, map, Some(point_lights))
                        }
                    })
                    .collect();
                queue.write_buffer(&batch.instance_buffer, 0, bytemuck::cast_slice(&inst_data));
            }
            self.last_point_light_revision = revision;
        }
    }

    /// Sync per-sub-unit animation offsets (platform rise / doors
    /// slide) from the live `Building` objects in the arena. Ports
    /// `CMatrixBuilding::LogicTakt`'s per-unit matrix updates at
    /// MatrixObjectBuilding.cpp:836-852. Called once per frame
    /// AFTER `World::takt` so the offsets are computed from the
    /// current frame's `base_floor_progress`.
    pub fn sync_building_animation(
        &mut self,
        queue: &wgpu::Queue,
        objs: &Objects,
        map: &GameMap,
        point_lights: &PointLightSystem,
    ) {
        use crate::matrix_game::map_static::{MapStatic, ObjectType};

        // Collect live Building progress + current side keyed by
        // (pos_x, pos_y) so the renderer can match its static
        // `BuildingInstance` list.
        //
        // The C++ keeps `CMatrixBuilding::m_Pos` and the renderer's
        // CVectorObjectGroup sharing the same world transform, so
        // identical (pos.x, pos.y) is a safe key.
        let mut progress_by_pos: std::collections::HashMap<(i32, i32), f32> =
            std::collections::HashMap::new();
        let mut side_by_pos: std::collections::HashMap<(i32, i32), u8> =
            std::collections::HashMap::new();
        // Ruin map-objects spawned by finished DIP sequences, keyed by VO
        // path → world placements (x, y, z, angle_z).
        let mut ruins_by_vo: std::collections::HashMap<&str, Vec<(f32, f32, f32, f32)>> =
            std::collections::HashMap::new();
        for id in objs.iter_units() {
            if let Some(obj) = objs.get(id) {
                match obj.core().obj_type {
                    ObjectType::Building => {
                        let b: &Building =
                            unsafe { &*(obj as *const dyn MapStatic as *const Building) };
                        let key = pos_key(b.pos.x, b.pos.y);
                        side_by_pos.insert(key, b.side.clamp(0, 255) as u8);
                        if b.kind == BuildingType::Base {
                            progress_by_pos.insert(key, b.base_floor_progress);
                        }
                    }
                    ObjectType::MapObject => {
                        let mo: &crate::matrix_game::object::MapObject = unsafe {
                            &*(obj as *const dyn MapStatic
                                as *const crate::matrix_game::object::MapObject)
                        };
                        if let Some(vo) = mo.ruin_graph.as_deref() {
                            let w = mo.core().matrix.w_axis;
                            ruins_by_vo
                                .entry(vo)
                                .or_default()
                                .push((w.x, w.y, w.z, mo.angle_z));
                        }
                    }
                    _ => {}
                }
            }
        }
        if !side_by_pos.is_empty() {
            self.logic_buildings_seen = true;
        }

        // Dead-building hide (`StaticDelete(this)` at the end of the DIP
        // sequence, MatrixObjectBuilding.cpp:757): once logic buildings
        // exist, any render instance missing from the arena collapses to
        // a degenerate transform; ruin instances replace it below.
        if self.logic_buildings_seen {
            for batch in &mut self.batches {
                let mut dirty = false;
                for (i, b) in batch.buildings.iter().enumerate() {
                    let dead = !side_by_pos.contains_key(&pos_key(b.x, b.y));
                    if batch.hidden[i] != dead {
                        batch.hidden[i] = dead;
                        dirty = true;
                    }
                }
                if dirty {
                    let [cx, cy] = batch.center;
                    let inst_data: Vec<InstanceData> = batch
                        .buildings
                        .iter()
                        .enumerate()
                        .map(|(i, b)| {
                            if batch.hidden[i] {
                                hidden_instance()
                            } else {
                                instance_matrix(b, cx, cy, map, Some(point_lights))
                            }
                        })
                        .collect();
                    queue.write_buffer(
                        &batch.instance_buffer,
                        0,
                        bytemuck::cast_slice(&inst_data),
                    );
                }
            }
            for (i, key) in self.shadow_keys.iter().enumerate() {
                self.shadow_visible[i] = side_by_pos.contains_key(key);
            }
        }

        // Ruin instances. Count only grows (ruins are never removed), so
        // a count change is a sufficient dirty signal.
        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;
        for rb in &mut self.ruin_batches {
            let placements = ruins_by_vo.get(rb.vo_path.as_str());
            let count = placements.map(|p| p.len()).unwrap_or(0).min(rb.capacity as usize) as u32;
            if count == rb.num_instances {
                continue;
            }
            let mut inst_data: Vec<InstanceData> = Vec::with_capacity(count as usize);
            if let Some(placements) = placements {
                for &(x, y, z, angle_z) in placements.iter().take(count as usize) {
                    let (s, c) = angle_z.sin_cos();
                    let [tr, tg, tb] = unpack_rgb(
                        map.static_object_color_with_lighting(x, y, Some(point_lights)),
                    );
                    let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(0);
                    inst_data.push(InstanceData {
                        row0: [c, -s, 0.0, x - cx],
                        row1: [s, c, 0.0, y - cy],
                        row2: [0.0, 0.0, 1.0, z],
                        row3: [0.0, 0.0, 0.0, 1.0],
                        terrain_color: [tr, tg, tb, 1.0],
                        unit_offset: [0.0; 4],
                        side_color: [sr, sg, sb, 1.0],
                    });
                }
            }
            queue.write_buffer(&rb.instance_buffer, 0, bytemuck::cast_slice(&inst_data));
            rb.num_instances = count;
        }

        // Side re-tint on capture (CMatrixBuilding::Capture flips
        // m_Side; the C++ re-reads GetSideColor every Draw — our
        // instance buffers are baked, so rewrite any batch whose
        // stored side went stale).
        for batch in &mut self.batches {
            let mut dirty = false;
            for b in batch.buildings.iter_mut() {
                let key = ((b.x * 10.0) as i32, (b.y * 10.0) as i32);
                if let Some(&side) = side_by_pos.get(&key) {
                    if b.side != side {
                        b.side = side;
                        dirty = true;
                    }
                }
            }
            if dirty {
                let [cx, cy] = batch.center;
                let inst_data: Vec<InstanceData> = batch
                    .buildings
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        if batch.hidden[i] {
                            hidden_instance()
                        } else {
                            instance_matrix(b, cx, cy, map, Some(point_lights))
                        }
                    })
                    .collect();
                queue.write_buffer(&batch.instance_buffer, 0, bytemuck::cast_slice(&inst_data));
            }
        }

        // Early-out when nothing is animating.
        if progress_by_pos.is_empty() {
            return;
        }

        for batch in &mut self.batches {
            if batch.kind != BuildingType::Base {
                continue;
            }
            // Only platform + doors move (unit IDs 1, 2, 3).
            let Some(uid) = batch.unit_id else { continue };
            if !(1..=3).contains(&uid) {
                continue;
            }
            let [cx, cy] = batch.center;
            let inst_data: Vec<InstanceData> = batch
                .buildings
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    if batch.hidden[i] {
                        return hidden_instance();
                    }
                    let key = pos_key(b.x, b.y);
                    let p = progress_by_pos.get(&key).copied().unwrap_or(0.0);
                    let mut d = instance_matrix(b, cx, cy, map, Some(point_lights));
                    d.unit_offset = sub_unit_offset(uid, p);
                    d
                })
                .collect();
            queue.write_buffer(&batch.instance_buffer, 0, bytemuck::cast_slice(&inst_data));
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

        // Shadows go down first so the building meshes overdraw their bases.
        // This mirrors `DrawShadowsProjFast` running before the object pass in
        // the original's per-frame order (MatrixMap.cpp:2283).
        self.shadow_system
            .update_view(queue, view_proj, self.shadow_color);
        self.shadow_system
            .render_visible(pass, &self.shadow_batches, &self.shadow_visible);

        pass.set_pipeline(&self.pipeline);
        for batch in &self.batches {
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, batch.instance_buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..batch.num_indices, 0, 0..batch.num_instances);
        }
        for batch in &self.ruin_batches {
            if batch.num_instances == 0 {
                continue;
            }
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

/// Shared pos-derived identity between arena `Building`s and the renderer's
/// static `BuildingInstance` list (buildings never move; same convention as
/// the side/animation sync below).
fn pos_key(x: f32, y: f32) -> (i32, i32) {
    ((x * 10.0) as i32, (y * 10.0) as i32)
}

/// World matrix for shadow geometry — uncentered (raw map coords). The shadow
/// system uses raw `map.points[]` heights and only subtracts the renderer's
/// half-world center when emitting the final ground-vertex positions.
fn building_world_matrix(b: &BuildingInstance) -> Mat4 {
    let theta = (b.angle as f32) * std::f32::consts::FRAC_PI_2;
    let (s, c) = theta.sin_cos();
    let rot = Mat3::from_cols(
        Vec3::new(c, s, 0.0),
        Vec3::new(-s, c, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    let translation = Vec3::new(b.x, b.y, b.build_z);
    Mat4::from_translation(translation) * Mat4::from_mat3(rot)
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
    let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(b.side as i32);
    InstanceData {
        row0: [rot.x_axis.x, rot.y_axis.x, rot.z_axis.x, b.x - cx],
        row1: [rot.x_axis.y, rot.y_axis.y, rot.z_axis.y, b.y - cy],
        row2: [rot.x_axis.z, rot.y_axis.z, rot.z_axis.z, b.build_z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [terrain_r, terrain_g, terrain_b, 1.0],
        unit_offset: [0.0, 0.0, 0.0, 0.0],
        side_color: [sr, sg, sb, 1.0],
    }
}

/// Port of the per-unit `D3DXMatrixTranslation` applied in
/// `CMatrixBuilding::RNeed` (MatrixObjectBuilding.cpp:842-850) while
/// the base's floor is animating. `progress` is `m_BaseFloor` in
/// [0, 1]. Platform (unit 1) translates in Z by
/// `-(1 - p) * 63 - 3`; left door (unit 2) slides `+25` along X,
/// right door (unit 3) slides `-25`, both maxing at `p = 0.5`.
fn sub_unit_offset(unit_id: i32, progress: f32) -> [f32; 4] {
    let door_shift = (progress * 2.0).clamp(0.0, 1.0) * 25.0;
    match unit_id {
        1 => [0.0, 0.0, -(1.0 - progress) * 63.0 - 3.0, 0.0],
        2 => [door_shift, 0.0, 0.0, 0.0],
        3 => [-door_shift, 0.0, 0.0, 0.0],
        _ => [0.0, 0.0, 0.0, 0.0],
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
        use crate::matrix_game::map::TurretPlaceCmap;
        // Synthesise 4 empty turret slots in a cross pattern around the
        // building so `queue_turret` has somewhere to land in tests.
        let pad = 30.0_f32;
        let scale = GameMap::GLOBAL_SCALE_MOVE;
        let cell = |dx: f32, dy: f32, ang: f32| TurretPlaceCmap {
            coord: (
                ((100.0 + dx) / scale).floor() as i32,
                ((100.0 + dy) / scale).floor() as i32,
            ),
            angle: ang,
            cannon_type: -1,
        };
        BuildingInstance {
            x: 100.0,
            y: 100.0,
            build_z: 0.0,
            angle: 0,
            side,
            kind,
            shadow_kind: 0,
            shadow_size: 128,
            turrets_places_cnt: 4,
            turret_places: vec![
                cell(pad, pad, 0.0),
                cell(-pad, pad, std::f32::consts::FRAC_PI_2),
                cell(pad, -pad, -std::f32::consts::FRAC_PI_2),
                cell(-pad, -pad, std::f32::consts::PI),
            ],
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
    fn dip_finish_spawns_ruins_and_static_deletes_building() {
        // End of the DIP sequence (MatrixObjectBuilding.cpp:691-758):
        // ruin map-objects (`ruins\b<n>.vo` + `b<n>p.vo` for non-BASE)
        // spawn at the building's pos/angle and the building itself is
        // StaticDelete'd.
        use crate::matrix_game::logic::MapLogic;
        use crate::matrix_game::object::MapObject;

        let mut w = MapLogic::with_seed(1);
        let mut b = Building::from_instance(&inst(1, 2)); // Titan
        b.init_max_hitpoint(100.0);
        b.state = BaseState::Dip;
        b.hit_point = -1.0; // Damage's death transition seeds -1
        let id = w.objects.spawn(Box::new(b));
        {
            let obj = w.objects.get_mut(id).unwrap();
            let bm = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
            bm.self_id = Some(id);
        }
        w.objects.add_lt(id);

        // downtime = -BUILDING_EXPLOSION_TIME = -1000 for non-BASE; the
        // hit-point timer must cross it.
        w.takt(1100);

        assert!(w.objects.get(id).is_none(), "building StaticDelete'd");
        let mut ruins: Vec<String> = Vec::new();
        w.objects.for_each_live(|_, obj| {
            if matches!(obj.core().obj_type, ObjectType::MapObject) {
                let mo = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
                if let Some(g) = &mo.ruin_graph {
                    ruins.push(g.clone());
                }
                assert_eq!(obj.core().matrix.w_axis.truncate().truncate(), glam::Vec2::new(100.0, 100.0));
            }
        });
        ruins.sort();
        assert_eq!(
            ruins,
            vec![
                "Matrix/Building/ruins/b1.vo".to_string(),
                "Matrix/Building/ruins/b1p.vo".to_string(),
            ]
        );
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
    fn damage_decrements_hp_and_transitions_to_dip_on_death() {
        use crate::matrix_game::config::WeaponDamage;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_GUN};

        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(300.0);
        let mut objs = Objects::new();
        objs.building_damages.table[weap_to_index(WEAPON_GUN).unwrap()] = WeaponDamage {
            damage: 100,
            mindamage: 0,
            friend_damage: 50,
        };
        let id = objs.spawn(Box::new(b));

        // 3 hits of 100 damage, enemy side → HP 0, transitions to DIP
        // (3rd hit tips to exactly 0.0 which is `<=0` in the branch).
        for _ in 0..3 {
            objs.apply_damage(id, WEAPON_GUN, glam::Vec3::ZERO, glam::Vec3::Z, 2, None);
        }
        let got = objs.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.state, BaseState::Dip);
        assert_eq!(mb.hit_point, -1.0);
        assert_eq!(mb.next_explosion_time, 0);
    }

    #[test]
    fn damage_to_dip_building_is_noop_and_returns_true() {
        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(100.0);
        b.state = BaseState::Dip;
        let mut objs = Objects::new();
        let id = objs.spawn(Box::new(b));

        let died = objs.apply_damage(
            id,
            crate::matrix_game::effects::weapon::WEAPON_BIGBOOM,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            2,
            None,
        );
        assert!(died, "already-DIP returns true without modifying state");
        let got = objs.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.hit_point, 100.0, "HP untouched");
    }

    #[test]
    fn friendly_fire_selects_friend_damage_column() {
        // Same-side attacker deals `friend_damage` instead of `damage`.
        use crate::matrix_game::config::WeaponDamage;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_BIGBOOM};

        let mut b = Building::from_instance(&inst(0, 2)); // side 2 (enemy AI)
        b.init_max_hitpoint(1000.0);
        let mut objs = Objects::new();
        objs.building_damages.table[weap_to_index(WEAPON_BIGBOOM).unwrap()] = WeaponDamage {
            damage: 500,
            mindamage: 0,
            friend_damage: 100,
        };
        let id = objs.spawn(Box::new(b));

        // Same-side attacker (side 2) — friend_damage=100.
        objs.apply_damage(id, WEAPON_BIGBOOM, glam::Vec3::ZERO, glam::Vec3::Z, 2, None);
        let mb = unsafe { &*(objs.get(id).unwrap() as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.hit_point, 900.0);

        // Enemy attacker (different non-zero side) — full damage=500.
        objs.apply_damage(id, WEAPON_BIGBOOM, glam::Vec3::ZERO, glam::Vec3::Z, 3, None);
        let mb = unsafe { &*(objs.get(id).unwrap() as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.hit_point, 400.0);
    }

    #[test]
    fn weapon_repair_heals_up_to_max() {
        // WEAPON_REPAIR adds HP rather than removing it. Clamps at max.
        use crate::matrix_game::config::WeaponDamage;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_REPAIR};

        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(200.0);
        b.hit_point = 50.0;
        let mut objs = Objects::new();
        objs.building_damages.table[weap_to_index(WEAPON_REPAIR).unwrap()] = WeaponDamage {
            damage: 80,
            mindamage: 0,
            friend_damage: 80,
        };
        let id = objs.spawn(Box::new(b));

        objs.apply_damage(
            id,
            WEAPON_REPAIR,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        let mb = unsafe { &*(objs.get(id).unwrap() as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.hit_point, 130.0);

        // Over-heal clamps to max.
        objs.apply_damage(
            id,
            WEAPON_REPAIR,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        let mb = unsafe { &*(objs.get(id).unwrap() as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.hit_point, 200.0, "clamped at hit_point_max");
    }

    #[test]
    fn under_attack_timer_latches_on_enemy_hits_only() {
        use crate::matrix_game::config::WeaponDamage;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_GUN};

        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(1000.0);
        let mut objs = Objects::new();
        objs.building_damages.table[weap_to_index(WEAPON_GUN).unwrap()] = WeaponDamage {
            damage: 20,
            mindamage: 0,
            friend_damage: 10,
        };
        let id = objs.spawn(Box::new(b));

        // Friendly-fire: no warning.
        objs.apply_damage(
            id,
            WEAPON_GUN,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        let mb = unsafe { &*(objs.get(id).unwrap() as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.under_attack_time, 0);

        // Enemy fire: warning latches.
        objs.apply_damage(id, WEAPON_GUN, glam::Vec3::ZERO, glam::Vec3::Z, 2, None);
        let mb = unsafe { &*(objs.get(id).unwrap() as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.under_attack_time, UNDER_ATTACK_IDLE_TIME_MS);
    }

    #[test]
    fn opening_base_completes_and_latches_at_opened() {
        use crate::matrix_game::logic::MapLogic;
        let mut w = MapLogic::with_seed(1);
        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(500.0);
        b.open();
        assert_eq!(b.state, BaseState::Opening);
        assert_eq!(b.base_floor_progress, 0.0);
        let id = w.objects.spawn(Box::new(b));
        w.objects.add_lt(id);

        // BASE_FLOOR_SPEED=0.0008/ms → 1250ms to reach 1.0. 1000ms → 0.8.
        w.takt(1000);
        let got = w.objects.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert!(
            (mb.base_floor_progress - 0.8).abs() < 1e-3,
            "base_floor_progress after 1s should be 0.8, got {}",
            mb.base_floor_progress
        );
        assert_eq!(mb.state, BaseState::Opening);

        // Another 500ms pushes past 1.0 and latches at Opened.
        w.takt(500);
        let got = w.objects.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.base_floor_progress, 1.0);
        assert_eq!(mb.state, BaseState::Opened);
    }

    #[test]
    fn closing_base_completes_and_latches_at_closed() {
        use crate::matrix_game::logic::MapLogic;
        let mut w = MapLogic::with_seed(1);
        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(500.0);
        b.base_floor_progress = 1.0;
        b.close();
        assert_eq!(b.state, BaseState::Closing);
        let id = w.objects.spawn(Box::new(b));
        w.objects.add_lt(id);

        // 1250ms to close fully. 1000ms → 0.2.
        w.takt(1000);
        let got = w.objects.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert!((mb.base_floor_progress - 0.2).abs() < 1e-3);
        assert_eq!(mb.state, BaseState::Closing);

        w.takt(500);
        let got = w.objects.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.base_floor_progress, 0.0);
        assert_eq!(mb.state, BaseState::Closed);
    }

    #[test]
    fn non_base_kinds_do_not_animate_base_floor() {
        // BUILDING_TITAN etc. have no open/close animation — state
        // transitions issued via open()/close() stay in Opening/Closing
        // because the animation block is gated on kind==Base.
        use crate::matrix_game::logic::MapLogic;
        let mut w = MapLogic::with_seed(1);
        let mut b = Building::from_instance(&inst(1, PLAYER_SIDE as u8)); // kind=Titan
        b.init_max_hitpoint(500.0);
        b.open();
        let id = w.objects.spawn(Box::new(b));
        w.objects.add_lt(id);

        w.takt(2000);
        let got = w.objects.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.base_floor_progress, 0.0);
        assert_eq!(
            mb.state,
            BaseState::Opening,
            "state stays Opening for non-Base kinds"
        );
    }

    #[test]
    fn under_attack_timer_decays_to_zero() {
        use crate::matrix_game::logic::MapLogic;
        let mut w = MapLogic::with_seed(1);
        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(500.0);
        b.under_attack_time = 1000;
        let id = w.objects.spawn(Box::new(b));
        w.objects.add_lt(id);

        w.takt(400);
        let got = w.objects.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.under_attack_time, 600);

        w.takt(1000);
        let got = w.objects.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert_eq!(
            mb.under_attack_time, 0,
            "clamped at zero, no negative drift"
        );
    }

    #[test]
    fn damage_resets_show_hitpoint_timer() {
        use crate::matrix_game::common::HITPOINT_SHOW_TIME_MS;
        use crate::matrix_game::config::WeaponDamage;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_GUN};

        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(500.0);
        let mut objs = Objects::new();
        objs.building_damages.table[weap_to_index(WEAPON_GUN).unwrap()] = WeaponDamage {
            damage: 50,
            mindamage: 0,
            friend_damage: 25,
        };
        let id = objs.spawn(Box::new(b));

        objs.apply_damage(id, WEAPON_GUN, glam::Vec3::ZERO, glam::Vec3::Z, 2, None);
        let mb = unsafe { &*(objs.get(id).unwrap() as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.show_hitpoint_time, HITPOINT_SHOW_TIME_MS);
    }

    #[test]
    fn timers_freeze_in_dip_state() {
        // Dying buildings skip the pre-DIP block entirely (C++ guard
        // at MatrixObjectBuilding.cpp:502).
        use crate::matrix_game::logic::MapLogic;
        let mut w = MapLogic::with_seed(1);
        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(500.0);
        b.state = BaseState::Dip;
        b.under_attack_time = 1000;
        b.show_hitpoint_time = 500;
        let id = w.objects.spawn(Box::new(b));
        w.objects.add_lt(id);

        w.takt(2000);
        let got = w.objects.get(id).unwrap();
        let mb = unsafe { &*(got as *const dyn MapStatic as *const Building) };
        assert_eq!(mb.under_attack_time, 1000, "frozen in DIP");
        assert_eq!(mb.show_hitpoint_time, 500, "frozen in DIP");
    }

    #[test]
    fn queue_turret_slot_reserves_marks_cannon_type_and_blocks_double_build() {
        // The C++ marks `m_TurretsPlaces[i].m_CannonType` so subsequent
        // hovers over the same slot are rejected
        // (MatrixObjectBuilding.cpp:1569-1576).
        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.turrets_max = 4;
        assert_eq!(b.turret_places.len(), 4);
        for p in &b.turret_places {
            assert_eq!(p.cannon_type, -1, "fresh slots empty");
        }

        assert!(
            b.queue_turret_slot(2, 3),
            "slot 2 should accept turret kind 3"
        );
        assert_eq!(b.turret_places[2].cannon_type, 3);
        assert_eq!(b.turrets_have, 1);

        // Re-queue on same slot must fail.
        assert!(!b.queue_turret_slot(2, 1), "slot 2 already occupied");
        assert_eq!(b.turret_places[2].cannon_type, 3);
        assert_eq!(b.turrets_have, 1);
    }

    #[test]
    fn queue_turret_slot_rejects_when_max_reached() {
        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.turrets_max = 2; // simulate the EBuildingTurrets cap
        assert!(b.queue_turret_slot(0, 1));
        assert!(b.queue_turret_slot(1, 2));
        // 3rd build hits the turrets_max gate.
        assert!(!b.queue_turret_slot(2, 3));
        assert_eq!(b.turrets_have, 2);
    }

    #[test]
    fn turret_build_progress_ramps_cannon_hp_and_completes_to_idle() {
        use crate::matrix_game::logic::MapLogic;
        use crate::matrix_game::object_cannon::{Cannon, CannonState};

        let mut w = MapLogic::with_seed(1);
        let mut b = Building::from_instance(&inst(0, PLAYER_SIDE as u8));
        b.init_max_hitpoint(1000.0);
        b.state = BaseState::Closed; // turret builds complete regardless of state; Closed is the rest state
        let parent_id = w.objects.spawn(Box::new(b));
        w.objects.add_lt(parent_id);
        // Re-borrow to wire self_id + queue the turret.
        {
            let obj = w.objects.get_mut(parent_id).unwrap();
            let bb: &mut Building = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
            bb.self_id = Some(parent_id);
            assert!(bb.queue_turret_slot(0, 1));
        }

        // Spawn the cannon mid-construction (the form_game placement
        // path mirrors this in `try_place_turret`).
        let mut cannon = Cannon::new(
            glam::Vec2::new(100.0, 100.0),
            0.0,
            0.0,
            PLAYER_SIDE,
            1,
            Some(parent_id),
            0,
        );
        cannon.hit_point_max = 200.0;
        cannon.begin_construction();
        assert!(cannon.invulnerable);
        assert_eq!(cannon.hit_point, 0.0);
        assert_eq!(cannon.state, CannonState::UnderConstruction);
        let cannon_id = w.objects.spawn(Box::new(cannon));
        w.objects.add_lt(cannon_id);

        // Half-way through the build timer: HP ramped to ~50%, still
        // UnderConstruction.
        let total = crate::matrix_game::config::turret_build_time_ms();
        w.takt(total / 2);
        let half = w.objects.get(cannon_id).unwrap();
        let cm = unsafe { &*(half as *const dyn MapStatic as *const Cannon) };
        assert!(
            cm.hit_point > 80.0 && cm.hit_point < 130.0,
            "expected ~100, got {}",
            cm.hit_point
        );
        assert!(cm.invulnerable);
        assert_eq!(cm.state, CannonState::UnderConstruction);

        // Past the threshold: HP=max, IDLE, vulnerable.
        w.takt(total);
        let done = w.objects.get(cannon_id).unwrap();
        let cm = unsafe { &*(done as *const dyn MapStatic as *const Cannon) };
        assert_eq!(cm.state, CannonState::Idle);
        assert_eq!(cm.hit_point, cm.hit_point_max);
        assert!(!cm.invulnerable);
    }

    #[test]
    fn closed_gate_applies_to_robots_only() {
        // The C++ gates only the robot branch on BASE_CLOSED
        // (MatrixObjectBuilding.cpp:1690); turrets (:1776) complete
        // regardless — non-base factories never animate to Closed.
        let mut objs = Objects::new();
        let parent_id = objs.spawn(Box::new(Building::from_instance(&inst(1, 1))));

        let mut bs = BuildStack::new();
        assert!(bs.add_item(PendingItem {
            kind: PendingKind::Turret {
                slot: 0,
                turret_kind: 1
            },
            side: 1,
        }));
        let total = turret_build_time_ms();
        bs.tick_timer(
            total,
            &mut objs,
            parent_id,
            BaseState::Opened,
            glam::Vec3::ZERO,
            0,
        );
        assert!(bs.is_empty(), "turret completes while base not Closed");

        let mut bs = BuildStack::new();
        let mut cfg = RobotConfig::new();
        cfg.hull.unit.kind = RobotUnitKind::ARMOR_PASSIVE;
        assert!(bs.add_item(PendingItem {
            kind: PendingKind::Robot(cfg),
            side: 1,
        }));
        let total = robot_build_time_ms();
        let spawned = bs.tick_timer(
            total,
            &mut objs,
            parent_id,
            BaseState::Opened,
            glam::Vec3::ZERO,
            0,
        );
        assert!(spawned.is_none());
        assert_eq!(bs.items(), 1, "robot waits for BASE_CLOSED");
    }

    #[test]
    fn building_fits_trace_building_mask_and_shows_up_in_find_objects() {
        // The whole point of pulling buildings into the arena:
        // FindObjects(TRACE_BUILDING) starts returning them.
        let mut objs = Objects::new();
        let b = Building::from_instance(&inst(0, 1));
        let id = objs.spawn(Box::new(b));
        assert!(objs.any_object_in_radius(
            glam::Vec2::new(100.0, 100.0),
            50.0,
            1.0,
            TRACE_BUILDING,
            None,
        ));
        // And doesn't match unrelated masks.
        assert!(!objs.any_object_in_radius(
            glam::Vec2::new(100.0, 100.0),
            50.0,
            1.0,
            TRACE_ROBOT,
            None,
        ));
        let _ = id;
    }
}
