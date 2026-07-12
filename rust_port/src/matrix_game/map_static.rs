//! Port of `CMatrixMapStatic` (MatrixMapStatic.{cpp,hpp}).
//!
//! C++ keeps every live map object in a single intrusive linked list rooted
//! at `m_FirstLogicTemp` and walked by `ProceedLogic`. Objects opt in with
//! `AddLT()` (only when their subclass has per-takt work — `BEHF_STATIC`
//! palms never AddLT, for example). `ProceedLogic` snapshots the next
//! pointer before dispatching so a callee may remove itself from the list
//! during its own takt without breaking the walk
//! (`MatrixMapStatic.cpp:349`).
//!
//! Rust equivalent: a generational-index arena. `ObjectId` lookups returning
//! `None` reproduce the `SObjectCore::m_Object == NULL` tombstone that the
//! C++ effects use to detect owner death. The logic-temp list is stored as
//! `Option<ObjectId>` prev/next fields on each slot.
//!
//! Scope A: base class + tick driver. Concrete subclasses land later; the
//! `MapStatic` trait here is what they implement.
//!
//! Scope B adds `CMatrixMapObject` (see `object.rs`) — the first live
//! subclass wired through `ProceedLogic`.

use glam::{Mat4, Vec2, Vec3};

use crate::matrix_game::common::{
    TRACE_BUILDING, TRACE_CANNON, TRACE_FLYER, TRACE_OBJECT, TRACE_ROBOT, TRACE_SKIP_INVISIBLE,
};
use crate::matrix_game::config::{BuildingDamages, ObjectDamages};
use crate::matrix_game::logic::Rnd;
use crate::matrix_game::sound::{Interrupt, SndEvent, SndHandle, SoundLayer};

// ── Resource-change bits (MR_*) (MatrixMapStatic.hpp:17-25) ──────────────
//
// Subclasses set these via `rchange_set` to mark which derived resources
// need recomputing; `RNeed(mask)` rebuilds the intersection of requested +
// dirty bits and clears them. Default at construction is `ALL` (matches
// `m_RChange(0xffffffff)`, MatrixMapStatic.hpp:346).

pub const MR_GRAPH: u32 = 1 << 0;
pub const MR_MATRIX: u32 = 1 << 1;
pub const MR_POS: u32 = 1 << 2;
pub const MR_ROTATE: u32 = 1 << 3;
pub const MR_SHADOW_STENCIL: u32 = 1 << 4;
pub const MR_SHADOW_PROJ_GEOM: u32 = 1 << 6;
pub const MR_SHADOW_PROJ_TEX: u32 = 1 << 7;
pub const MR_MINIMAP: u32 = 1 << 8;
pub const MR_ALL: u32 = 0xFFFF_FFFF;

// ── Object state bits (MatrixMapStatic.hpp:46-85) ────────────────────────
//
// The low bits are shared across all subclasses; bits 10..22 are
// subclass-overlayed (ROBOT_FLAG_*, BUILDING_*, cannon, mesh). We expose
// the overlays as `#[allow(dead_code)]` constants so subclass ports can
// reference them by their original names.

pub const OBJECT_STATE_ABLAZE: u32 = 1 << 0;
pub const OBJECT_STATE_SHORTED: u32 = 1 << 1;
pub const OBJECT_STATE_INVISIBLE: u32 = 1 << 2;
pub const OBJECT_STATE_INTERFACE: u32 = 1 << 3;
pub const OBJECT_STATE_INVULNERABLE: u32 = 1 << 3;
pub const OBJECT_STATE_SHADOW_SPECIAL: u32 = 1 << 4;
pub const OBJECT_STATE_TRACE_INVISIBLE: u32 = 1 << 5;
pub const OBJECT_STATE_DIP: u32 = 1 << 6;

// Mesh-only (CMatrixMapObject).
#[allow(dead_code)]
pub const OBJECT_STATE_BURNED: u32 = 1 << 10;
#[allow(dead_code)]
pub const OBJECT_STATE_EXPLOSIVE: u32 = 1 << 11;
#[allow(dead_code)]
pub const OBJECT_STATE_NORMALIZENORMALS: u32 = 1 << 12;
#[allow(dead_code)]
pub const OBJECT_STATE_SPECIAL: u32 = 1 << 13;
#[allow(dead_code)]
pub const OBJECT_STATE_TERRON_EXPL: u32 = 1 << 14;
#[allow(dead_code)]
pub const OBJECT_STATE_TERRON_EXPL1: u32 = 1 << 15;
#[allow(dead_code)]
pub const OBJECT_STATE_TERRON_EXPL2: u32 = 1 << 16;

// Robot-only (`ROBOT_FLAG_*` — MatrixMapStatic.hpp:62-77).
/// `ROBOT_FLAG_SGROUP` (`SETBIT(12)`, MatrixMapStatic.hpp:74) —
/// selected as part of the current group (drives the selection ring).
pub const ROBOT_FLAG_SGROUP: u32 = 1 << 12;
/// `ROBOT_FLAG_SARCADE` (`SETBIT(13)`) — arcade-selected (FPS mode,
/// out of scope; kept for flag parity).
pub const ROBOT_FLAG_SARCADE: u32 = 1 << 13;
/// `ROBOT_FLAG_ONWATER` (`SETBIT(14)`, MatrixMapStatic.hpp:76). Set by
/// `Z_From_Pos` (MatrixObjectRobot.cpp:282) whenever the terrain height
/// at the robot's XY is below `WATER_LEVEL`. Read by `Seek`
/// (MatrixRobot.cpp:2420) to apply the chassis `m_SpeedWaterCorr`.
pub const ROBOT_FLAG_ONWATER: u32 = 1 << 14;
/// `ROBOT_FLAG_COLLISION` — set by LowLevelMove when a collision
/// correction kicked in this tick (MatrixRobot.cpp:2623).
#[allow(dead_code)]
pub const ROBOT_FLAG_COLLISION: u32 = 1 << 15;
/// `ROBOT_FLAG_DISABLE_MANUAL` (`SETBIT(16)`, MatrixMapStatic.hpp:78).
/// Set when a base capture commits the robot — blocks manual-control
/// handover and forces MustDie on order break (MatrixRobot.cpp:1267).
pub const ROBOT_FLAG_DISABLE_MANUAL: u32 = 1 << 16;
/// `ROBOT_FLAG_ROT_LEFT` / `ROBOT_FLAG_ROT_RIGHT` (`SETBIT(17/18)`,
/// MatrixMapStatic.hpp:79-80) — one-shot manual-steer marks consumed
/// (and cleared) by the arcade branch of the robot logic takt
/// (MatrixRobot.cpp:977-1002). Set from key polls and mouse-cam drag.
pub const ROBOT_FLAG_ROT_LEFT: u32 = 1 << 17;
pub const ROBOT_FLAG_ROT_RIGHT: u32 = 1 << 18;
/// `ROBOT_CAPTURE_INFORMED` (`SETBIT(22)`, MatrixMapStatic.hpp:85).
/// Transient mark used by the building's capture-candidate announce
/// scan (MatrixObjectBuilding.cpp:506-527).
pub const ROBOT_CAPTURE_INFORMED: u32 = 1 << 22;

// Building-only (overlays bits 10..=12 from MatrixMapStatic.hpp:88-90).
/// `BUILDING_NEW_INCOME` — a resource tick just fired; Takt emits a
/// billboard-score effect on the next frame (MatrixObjectBuilding.cpp:355).
#[allow(dead_code)]
pub const OBJECT_STATE_BUILDING_NEW_INCOME: u32 = 1 << 10;
/// `BUILDING_SPAWNBOT` — base door is open and a robot is being
/// delivered; prevents `Close()` mid-spawn (MatrixObjectBuilding.hpp:269-270).
#[allow(dead_code)]
pub const OBJECT_STATE_BUILDING_SPAWNBOT: u32 = 1 << 11;
/// `BUILDING_CAPTURE_IN_PROGRESS` — a robot of another side is
/// capturing the base (MatrixObjectBuilding.hpp:191).
#[allow(dead_code)]
pub const OBJECT_STATE_BUILDING_CAPTURE_IN_PROGRESS: u32 = 1 << 12;

/// `EObjectType` (MatrixMapStatic.hpp:29-39). Discriminants match the C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ObjectType {
    Empty = 0,
    MapObject = 2,
    RobotAi = 3,
    Building = 4,
    Cannon = 5,
    Flyer = 6,
}

/// Ports `SObjectCore` (MatrixMapStatic.hpp:133-199). Ref-counting is
/// dropped: the `ObjectId → Option<&Slot>` lookup reproduces the
/// `m_Object == NULL` tombstone (see module doc). `m_GeoCenter` /
/// `m_Radius` / `m_TerainColor` are populated by subclass `join_to_group`
/// (not implemented in scope A).
#[derive(Debug, Clone)]
pub struct ObjectCore {
    pub matrix: Mat4,         // m_Matrix
    pub inv_matrix: Mat4,     // m_IMatrix
    pub radius: f32,          // m_Radius
    pub geo_center: Vec3,     // m_GeoCenter
    pub obj_type: ObjectType, // m_Type
    pub terrain_color: u32,   // m_TerainColor  (0xFFFFFFFF default)
}

impl Default for ObjectCore {
    fn default() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
            inv_matrix: Mat4::IDENTITY,
            radius: 0.0,
            geo_center: Vec3::ZERO,
            obj_type: ObjectType::Empty,
            terrain_color: 0xFFFF_FFFF,
        }
    }
}

/// The `CMatrixMapStatic` contract. Subclasses implement this trait; the
/// [`Objects`] arena stores `Box<dyn MapStatic>`.
///
/// Scope A: only `takt` + `logic_takt` + `r_need` are hit by the tick
/// loop. The remaining virtuals (`pick`, `draw`, `calc_bounds`, …) are
/// here as default-noop methods so concrete subclasses can implement
/// them incrementally — matching the `= 0` pure virtuals at
/// `MatrixMapStatic.hpp:459-478`.
pub trait MapStatic {
    // --- Core accessors (inline getters on CMatrixMapStatic) ---
    fn core(&self) -> &ObjectCore;
    fn core_mut(&mut self) -> &mut ObjectCore;
    fn rchange(&self) -> u32;
    fn rchange_set(&mut self, bits: u32); // RChange(zn)  — OR-in
    fn rchange_clear(&mut self, bits: u32); // RNoNeed(zn)  — AND-NOT
    fn object_state(&self) -> u32;
    fn object_state_set(&mut self, bits: u32);
    fn object_state_clear(&mut self, bits: u32);

    // --- Ablaze / Shorted TTLs (MatrixMapStatic.hpp:254-263) ---
    fn ablaze_ttl(&self) -> i32;
    fn set_ablaze_ttl(&mut self, ttl: i32);
    fn shorted_ttl(&self) -> i32;
    fn set_shorted_ttl(&mut self, ttl: i32);

    // --- Pure virtuals (MatrixMapStatic.hpp:457-478) ---
    //
    // `takt` is the per-frame graphic takt; `logic_takt` is the fixed-step
    // 10ms takt. `r_need` rebuilds requested resources and clears the
    // matching MR_* bits in `m_RChange`. The `rng` is `g_MatrixMap->Rnd`
    // — a shared stream every subclass may advance; `objs` is the arena
    // with the current object's slot temporarily empty (see
    // [`Objects::proceed_logic`] / [`Objects::graphic_takt`]). Both are
    // passed explicitly so Takt/LogicTakt aren't hiding a thread-local
    // global.
    fn r_need(&mut self, need: u32);
    fn takt(&mut self, cms: i32, rng: &mut Rnd, objs: &mut Objects);
    fn logic_takt(&mut self, cms: i32, rng: &mut Rnd, objs: &mut Objects);

    /// `ShowHitpoint()` — arm the floating HP bar over the object for
    /// `HITPOINT_SHOW_TIME` (1000 ms). Called every frame for the
    /// object under the mouse cursor (MatrixMap.cpp:1150-1178).
    fn show_hitpoint(&mut self) {}

    /// Floating HP-bar placement — the `m_PB.Modify` blocks of the
    /// per-type `BeforeDraw`. `None` while the bar is hidden.
    fn hitpoint_bar(&self, _map: &crate::matrix_game::map::GameMap) -> Option<HpBar> {
        None
    }

    // Default-noop virtuals so subclasses can opt in as they're ported.
    fn before_draw(&mut self) {}
    fn free_dynamic_resources(&mut self) {}
    fn side(&self) -> i32 {
        -1
    }
    fn need_repair(&self) -> bool {
        false
    }

    /// Port of `virtual bool Pick(orig, dir, float *outt)`
    /// (MatrixMapStatic.hpp:462). Ray / object intersection. The
    /// default is a sphere test against `core().geo_center` +
    /// `core().radius` — a conservative approximation for subclasses
    /// that haven't ported their mesh-level pick. Subclasses (Building,
    /// Robot) can override with a tighter bounds test once per-instance
    /// mesh data is available.
    ///
    /// Returns the ray parameter `t` (distance along `dir` from
    /// `origin`) for the nearest hit, or `None` if the ray misses.
    /// `dir` must be normalized for the returned `t` to be a distance.
    fn pick(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        ray_sphere_pick(self.core().geo_center, self.core().radius, origin, dir)
    }

    /// Port of the `IsLive*` state checks `FitToMask` makes
    /// (MatrixMap.hpp:75-77): robots in `ROBOT_DIP`, cannons in
    /// `CANNON_DIP` and buildings in `BUILDING_DIP*` stop matching
    /// trace/search masks. Default: alive.
    fn is_live(&self) -> bool {
        true
    }

    /// Port of `IsSpecial()` (MatrixMapStatic.hpp:299) — map objects
    /// whose destruction is a win condition. Attack orders accept them
    /// alongside live units (MatrixSide.cpp:704/729/841).
    fn is_special(&self) -> bool {
        false
    }

    /// Port of `virtual bool Damage(EWeapon, pos, dir, attacker_side,
    /// attacker)` (MatrixMapStatic.hpp:464). Returns true iff the
    /// damage caused *this* object to be removed from play (C++
    /// `Damage` returns `true` only when `Init` reset mid-call, per
    /// comment at MatrixObject.cpp:1572). Default: ignore.
    ///
    /// `objs` is the arena sans this object's slot — same contract as
    /// `takt`/`logic_takt`. Callees use it to enroll themselves into
    /// the logic-temp list (AddLT) or spawn effects.
    #[allow(clippy::too_many_arguments)]
    fn damage(
        &mut self,
        _weap: crate::matrix_game::effects::weapon::Weapon,
        _pos: Vec3,
        _dir: Vec3,
        _attacker_side: i32,
        _attacker: Option<ObjectId>,
        _self_id: ObjectId,
        _objs: &mut Objects,
    ) -> bool {
        false
    }
}

// ── Helper predicates on &dyn MapStatic (matches `IsRobot()` etc.) ─────

#[allow(dead_code)]
pub fn is_robot(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::RobotAi)
}

#[allow(dead_code)]
pub fn is_building(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::Building)
}

#[allow(dead_code)]
pub fn is_cannon(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::Cannon)
}

#[allow(dead_code)]
pub fn is_flyer(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::Flyer)
}

#[allow(dead_code)]
pub fn is_map_object(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::MapObject)
}

/// Port of `CMatrixMapStatic::FitToMask` (MatrixMap.hpp:73-81). Returns
/// true when this object's type (and, for robots/cannons/buildings, its
/// live state) satisfies a `TRACE_*` mask.
///
/// The C++ guards on `IsLiveRobot` / `IsLiveCannon` / `IsLiveBuilding` —
/// those check the per-subclass state machine (`CurrState != ROBOT_DIP`
/// etc.). Those subclasses aren't ported yet, so we treat every
/// robot/cannon/building as "live" (the mask check alone decides
/// inclusion). When the state-machines land, gate this on an
/// `is_live()` trait method.
///
/// `TRACE_SKIP_INVISIBLE` additionally excludes objects with the
/// `OBJECT_STATE_TRACE_INVISIBLE` bit set — ported here because
/// `FindObjects` consumers set the flag in the mask.
/// World-units per decorative-grid cell (map is ~2000 world units →
/// a few hundred cells).
const MO_GRID_CELL: f32 = 128.0;

pub fn fit_to_mask(obj: &dyn MapStatic, mask: u32) -> bool {
    // `TRACE_SKIP_INVISIBLE`: objects opting out of trace visibility
    // (MatrixMapStatic.hpp:51, set by OTP_INVLOGIC="1") are filtered.
    if mask & TRACE_SKIP_INVISIBLE != 0 && obj.object_state() & OBJECT_STATE_TRACE_INVISIBLE != 0 {
        return false;
    }
    // `IsLiveRobot()` / `IsLiveCannon()` / `IsLiveBuilding()` — dead
    // (DIP) objects stop matching (MatrixMap.hpp:75-77).
    if !obj.is_live() {
        return false;
    }
    match obj.core().obj_type {
        ObjectType::RobotAi => mask & TRACE_ROBOT != 0,
        ObjectType::Cannon => mask & TRACE_CANNON != 0,
        ObjectType::Building => mask & TRACE_BUILDING != 0,
        ObjectType::MapObject => mask & TRACE_OBJECT != 0,
        ObjectType::Flyer => mask & TRACE_FLYER != 0,
        ObjectType::Empty => false,
    }
}

/// Return from a `find_objects` callback — matches the bool contract in
/// `ENUM_OBJECTS2D` (MatrixMap.hpp:84) where `true` means "keep
/// searching", `false` means "stop".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Continue,
    Break,
}

/// Ray-sphere intersection. Returns the nearest positive `t` (ray
/// parameter, equal to distance when `dir` is a unit vector) at which
/// `origin + t*dir` intersects the sphere centered at `center` with
/// radius `radius`. `None` if the ray misses or both roots are behind
/// the origin. Used as the default `MapStatic::pick` body for
/// subclasses that don't override with a per-mesh test.
pub fn ray_sphere_pick(center: Vec3, radius: f32, origin: Vec3, dir: Vec3) -> Option<f32> {
    if radius <= 0.0 {
        return None;
    }
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    // Two roots: -b - sqrt_d (entering) and -b + sqrt_d (exiting). We
    // want the nearest positive t — the entering hit unless the ray
    // origin is already inside the sphere.
    let t0 = -b - sqrt_d;
    let t1 = -b + sqrt_d;
    if t0 >= 0.0 {
        Some(t0)
    } else if t1 >= 0.0 {
        Some(t1)
    } else {
        None
    }
}

// ── Arena + logic-temp list ─────────────────────────────────────────────

/// Stable handle into [`Objects`]. A generational index so reused slots
/// don't silently alias a freed id — equivalent to checking
/// `core->m_Object != NULL` in the C++ ref-counted world.
/// Ordering is the Rust stand-in for the C++ heap-pointer tie-break
/// (`DWORD(pCurrBot) < DWORD(data->robot)` in CollisionCallback) — an
/// arbitrary but stable "only one of the pair yields" rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId {
    index: u32,
    generation: u32,
}

#[cfg(test)]
impl ObjectId {
    /// Test-only handle minting for unit tests that don't need an
    /// arena (e.g. env list bookkeeping).
    pub(crate) fn synthetic(index: u32) -> Self {
        Self {
            index,
            generation: 0,
        }
    }
}

struct Slot {
    generation: u32,
    obj: Option<Box<dyn MapStatic>>,
    in_lt: bool,
    prev_lt: Option<ObjectId>,
    next_lt: Option<ObjectId>,
}

/// Manual-control input snapshot for the arcaded (FPS-mode) robot.
/// Ports the `GetAsyncKeyState(g_Config.m_KeyActions[KA_UNIT_*])`
/// polls scattered through `CMatrixRobotAI::LogicTakt`
/// (MatrixRobot.cpp:958-974, 1024-1038, 3165-3169) plus the
/// `g_MatrixMap->m_TraceStopPos` cursor trace (MatrixMap.cpp:1149)
/// and the `g_IFaceList->m_InFocus == INTERFACE` gate. The frontend
/// refreshes this every frame; headless sims set it directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct ArcadeInput {
    pub left: bool,
    pub right: bool,
    pub forward: bool,
    pub backward: bool,
    /// `KA_FIRE` (VK_LBUTTON) held.
    pub fire: bool,
    /// `m_TraceStopPos` — world point under the cursor (trace skips
    /// the arcaded robot itself).
    pub cursor_world: glam::Vec3,
    /// `m_InFocus == INTERFACE` — cursor is over the 2D UI.
    pub cursor_on_interface: bool,
}

/// Arena of all map-static objects + the intrusive logic-temp list.
/// Ports the `static` members + `m_FirstLogicTemp/m_NextLogicTemp` linkage
/// from `CMatrixMapStatic` (MatrixMapStatic.hpp:265-268, .cpp:60-61).
pub struct Objects {
    slots: Vec<Slot>,
    free: Vec<u32>,
    first_lt: Option<ObjectId>,
    last_lt: Option<ObjectId>,

    /// Ports `g_MatrixMap->m_NextLogicObject` — the snapshot of the next
    /// list node taken before each `static_takt` dispatch, so a callee
    /// can `del_lt(self)` without breaking the walk
    /// (MatrixMapStatic.cpp:349).
    next_logic_object: Option<ObjectId>,

    /// Ports the `ms != g_MatrixMap->GetPlayerSide()->GetArcadedObject()`
    /// guard in `ProceedLogic` (MatrixMapStatic.cpp:350). Filled in when
    /// sides land; `None` means the guard is inactive.
    pub arcaded_object: Option<ObjectId>,

    /// Manual-control input for the arcaded robot (see [`ArcadeInput`]).
    pub arcade_input: ArcadeInput,

    /// Port of `g_Config.m_ObjectDamages` (MatrixConfig.hpp:574). The
    /// per-weapon damage table used by MapObject's Damage branches.
    /// Lives on `Objects` because it's a shared world-level resource
    /// that damage callees need from inside the take-the-box pattern.
    pub object_damages: ObjectDamages,
    /// Port of `g_Config.m_BuildingDamages` + `m_BuildingHitPoints`
    /// (MatrixConfig.cpp:507-535). Same shape as `object_damages` plus
    /// a per-EBuildingType max-HP vector.
    pub building_damages: BuildingDamages,

    /// Slab of live `CMatrixEffectWeapon`s (see effects/weapon.rs).
    /// Lives here so damage / projectile / robot-takt paths can reach
    /// it through their `&mut Objects`.
    pub weapons: crate::matrix_game::effects::weapon::Weapons,
    /// Gameplay effects spawned mid-takt; `MapLogic` drains this into
    /// its effect list each frame (effects/mod.rs `effects_takt`).
    pub pending_effects: Vec<crate::matrix_game::effects::GameEffect>,
    /// Per-side kill counters — port of the `IncStatValue(STAT_*)`
    /// calls in the Damage paths (`CMatrixSideUnit::m_Statistic`).
    /// Indexed by side id (0..=8).
    pub side_stats: [SideStats; 9],
    /// Landscape-spot spawn queue (`CreateLandscapeSpot` calls) —
    /// drained by the render side which owns the spot list + geometry.
    pub pending_spots: Vec<crate::matrix_game::effects::landscape_spot::SpotSpawn>,
    /// Deferred explosion spawns — `CreateExplosion` calls from damage
    /// paths (which carry no RNG/map); built in `effects_takt`.
    pub pending_explosions: Vec<ExplosionSpawn>,
    /// Deferred resource refunds `(side, [titan, elec, energy, plasma])`
    /// from cancelled build-queue items (`CBuildStack::DeleteItem`) —
    /// drained by `MapLogic::takt`, which owns the sides.
    pub pending_refunds: Vec<(i32, [i32; 4])>,
    /// Player robots freshly produced by a base — awaiting the
    /// `RobotSpawn` rally (AssignPlace + PGOrderAttack,
    /// MatrixRobot.cpp:2204-2223), which needs side/place access.
    pub pending_spawn_rallies: Vec<ObjectId>,
    /// Live-unit index (RobotAi/Building/Cannon/Flyer) — the hot logic
    /// scans and object traces walk this instead of every arena slot;
    /// on real maps decorative MapObjects outnumber units ~100:1 and a
    /// full-slot walk per trace was the battle-FPS killer.
    unit_ids: Vec<ObjectId>,
    /// Decorative `MapObject` index — consulted only when a trace mask
    /// asks for TRACE_OBJECT.
    mapobject_ids: Vec<ObjectId>,
    /// Spatial hash over the (static) decoratives: world-space cells of
    /// [`MO_GRID_CELL`] units, each listing the map objects whose
    /// bounding sphere overlaps it. Traces and radius queries visit a
    /// handful of cells instead of every decorative — a flying wreck
    /// piece used to run its per-slice TRACE_ALL against ~2000 pick()
    /// calls. Rebuilt lazily when a decorative spawns/despawns.
    mo_grid: std::cell::RefCell<std::collections::HashMap<(i32, i32), Vec<ObjectId>>>,
    mo_grid_dirty: std::cell::Cell<bool>,
    /// Freshly produced AI-side robots awaiting `ClacSpawnTeam`
    /// (MatrixRobot.cpp:2204-2205). Drained by the side-AI takt.
    pub pending_ai_spawn: Vec<ObjectId>,
    /// World-sound dispatch queue — one [`SndEvent`] per
    /// `CSound::Play/AddSound/ChangePos/StopPlay` call site in ported
    /// code. Drained by the app loop into the [`SoundMixer`]
    /// (`pump_sounds` in form_game.rs); without an audio backend the
    /// mixer is a no-op and the queue just empties.
    pub pending_sounds: Vec<SndEvent>,
    /// Deferred ruin-smoke requests `(center, radius)` — the 20-50
    /// temporary effect spawners the C++ scatters over a dead
    /// building's ruins (MatrixObjectBuilding.cpp:726-755). Drained
    /// by `MapLogic::takt` into `ambient_spawners`.
    pub pending_ruin_smoke: Vec<(glam::Vec3, f32)>,
    /// MMFLAG_FLYCAM mirror — gates the war-pair pushes below
    /// (MatrixRobot.cpp:1898, MatrixObjectCannon.cpp:1399).
    pub fly_cam: bool,
    /// `m_Camera.AddWarPair(this, attaker)` calls queued from the
    /// damage paths `(target, attacker)`; drained into the fly-cam's
    /// [`AutoFlyData`](crate::matrix_game::camera::AutoFlyData) each
    /// frame.
    pub pending_war_pairs: Vec<(ObjectId, ObjectId)>,
    /// Per-map-group flyer altitude envelope: max(terrain land max,
    /// static building/object tops). The static-scene slice of
    /// `m_GroupMaxZObjRobots` (GetZInterpolatedObjRobots,
    /// MatrixMap.cpp:512-546); built lazily on the first flyer takt.
    pub flyer_alt_grid: Vec<f32>,
    /// Deferred effect point-light spawns (`CreatePointLight`) — drained
    /// by the app loop into the terrain `PointLightSystem`.
    pub pending_point_lights: Vec<PendingLight>,
    /// Move-or-create commands for effect lights that follow a moving
    /// source (plasma bolt / flame), keyed by the owning weapon:
    /// `(key, pos, radius, color)`. Killed via `pending_light_kill`.
    pub pending_light_follow: Vec<(
        crate::matrix_game::effects::weapon::WeaponId,
        [f32; 3],
        f32,
        u32,
    )>,
    pub pending_light_kill: Vec<crate::matrix_game::effects::weapon::WeaponId>,
    /// BEHF_SPAWNER robot-spawn requests — object logic_takt can't reach
    /// the side/order layer, so MapLogic drains these (build robot from
    /// the `RobotSpawn` catalogue, spawn, order attack).
    pub pending_spawner_bots: Vec<SpawnerBotRequest>,
    /// Size of the debris mesh catalog (set by the renderer at init;
    /// 0 = no mesh debris, e.g. in tests).
    pub debris_catalog_len: usize,
    /// `deb_type` of each catalog entry (first comma-field of the
    /// `Models/Debris` par name) — explosions pick debris meshes only
    /// from entries matching their preset's `deb_type`
    /// (MatrixEffectExplosion.cpp:323-331).
    pub debris_types: Vec<i32>,
    /// `HitTo` notices for robots whose box is checked out when their
    /// own hitscan weapon lands (the C++ calls the fire-end handler
    /// synchronously; here the robot drains its entries right after
    /// `weapons_logic_takt`). `(shooter, victim (side, type), hit pos)`.
    pub pending_hit_notices: Vec<(ObjectId, Option<(ObjectId, i32, ObjectType)>, glam::Vec3)>,
    /// Deferred env-list purge — dead unit ids scrubbed from every
    /// robot's enemy list at a point where no robot box is checked
    /// out. The synchronous death-cascade purge misses robots whose
    /// box is out (e.g. the killer, mid-takt when its hitscan lands);
    /// a leaked id keeps `enemy_cnt() > 0` and wedges the side AI in
    /// permanent war (the C++ purge is pointer-walk synchronous and
    /// can't miss).
    pub pending_env_purge: Vec<ObjectId>,
    /// Self-despawn queue — an object can't `remove()` itself while
    /// its box is checked out by a takt driver, so DIP wrecks queue
    /// here and `flush_removals` (run by the takt drivers after each
    /// walk) finishes the job. Stands in for the C++
    /// `g_MatrixMap->StaticDelete(this)` calls from DIPTakt.
    pending_removals: Vec<ObjectId>,
    /// Destroyed special (win-target) map objects this takt. MapLogic
    /// drains these into `m_BeforeWinCount` / side-status bookkeeping
    /// (MatrixObject.cpp:203-212, :249-260, :1244-1258 — the object
    /// can't reach the map/sides from `damage`/`logic_takt`).
    pub pending_special_deaths: Vec<SpecialDeathKind>,
    /// `MMFLAG_TERRON_DEAD` (MatrixObject.cpp:171) — exempts the
    /// player from the JUST_DEAD scan in CheckStatus.
    pub terron_dead: bool,
    /// Buildings in the ROP_CAPTURING phase this takt — the app loop
    /// force-deselects them if the player has one selected
    /// (MatrixRobot.cpp:1286-1294: Select(NOTHING) + PLDropAllActions).
    pub pending_capture_deselect: Vec<ObjectId>,
}

/// Which special-object death path fired (the C++ branches differ:
/// BREAK/ANIM only set SS_JUST_WIN once the win counter is exhausted,
/// the terron sets it unconditionally).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialDeathKind {
    /// BEHF_BREAK / BEHF_ANIM special target (MatrixObject.cpp:203, :249).
    Target,
    /// Terron explosion completed (MatrixObject.cpp:1244).
    Terron,
}

/// One floating HP bar (`CMatrixProgressBar` placement). `anchor` is
/// projected to the screen; the bar's top-left lands at
/// `(screen.x + x_off, screen.y + y_off)`.
#[derive(Debug, Clone, Copy)]
pub struct HpBar {
    pub anchor: glam::Vec3,
    pub width: f32,
    /// 0..1 HP fraction (`m_HitPoint * m_MaxHitPointInversed`).
    pub fill: f32,
    pub x_off: f32,
    pub y_off: f32,
}

/// A queued `CreateExplosion(pos, props, fire)` call.
pub struct ExplosionSpawn {
    pub pos: glam::Vec3,
    pub props: &'static crate::matrix_game::effects::explosion::ExplosionProps,
    pub fire: bool,
}

/// A BEHF_SPAWNER robot-spawn request (MatrixObject.cpp:1383-1465).
#[derive(Clone, Copy)]
pub struct SpawnerBotRequest {
    pub pos: glam::Vec3,
    pub number: i32,
    pub pick: usize,
    /// Spawner's `m_SensRadius` — bounds the attack-target search
    /// (MatrixObject.cpp:1442, `FindObjects(pos, m_SensRadius, …)`).
    pub sens_radius: f32,
}

/// A queued `CreatePointLight` call: over phase 1 (`t1` ms) radius LERPs
/// `r1→r2` and colour LICs `c1→c2`; over the remainder (`ttl-t1`) radius
/// holds `r2` and colour fades to black. Single-phase lights (muzzle
/// flashes, plasma) set `t1==ttl`, so the whole life is one LERP.
#[derive(Clone, Copy)]
pub struct PendingLight {
    pub pos: [f32; 3],
    pub r1: f32,
    pub r2: f32,
    pub c1: u32,
    pub c2: u32,
    pub ttl: f32,
    pub t1: f32,
}

/// The kill/build-stat subset of `CMatrixSideUnit`'s statistics
/// (MatrixSide.hpp `EStat`). Accumulated here because the object code
/// can't reach the sides; `MapLogic::sync_side_stats` mirrors these
/// into each side's stat array every takt.
#[derive(Debug, Default, Clone, Copy)]
pub struct SideStats {
    pub robot_build: i32,
    pub robot_kill: i32,
    pub turret_build: i32,
    pub turret_kill: i32,
    pub building_kill: i32,
}

impl Default for Objects {
    fn default() -> Self {
        Self::new()
    }
}

impl Objects {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            first_lt: None,
            last_lt: None,
            next_logic_object: None,
            arcaded_object: None,
            arcade_input: ArcadeInput::default(),
            object_damages: ObjectDamages::default(),
            building_damages: BuildingDamages::default(),
            weapons: Default::default(),
            pending_effects: Vec::new(),
            side_stats: [SideStats::default(); 9],
            pending_spots: Vec::new(),
            pending_explosions: Vec::new(),
            pending_refunds: Vec::new(),
            pending_spawn_rallies: Vec::new(),
            pending_ai_spawn: Vec::new(),
            unit_ids: Vec::new(),
            mapobject_ids: Vec::new(),
            mo_grid: std::cell::RefCell::new(std::collections::HashMap::new()),
            mo_grid_dirty: std::cell::Cell::new(true),
            pending_sounds: Vec::new(),
            pending_ruin_smoke: Vec::new(),
            fly_cam: false,
            pending_war_pairs: Vec::new(),
            flyer_alt_grid: Vec::new(),
            pending_point_lights: Vec::new(),
            pending_light_follow: Vec::new(),
            pending_light_kill: Vec::new(),
            pending_spawner_bots: Vec::new(),
            debris_catalog_len: 0,
            debris_types: Vec::new(),
            pending_hit_notices: Vec::new(),
            pending_env_purge: Vec::new(),
            pending_removals: Vec::new(),
            pending_special_deaths: Vec::new(),
            terron_dead: false,
            pending_capture_deselect: Vec::new(),
        }
    }

    /// Queue a despawn that takes effect after the current takt walk.
    pub fn remove_deferred(&mut self, id: ObjectId) {
        self.pending_removals.push(id);
    }

    /// Apply queued despawns. Called by the takt drivers once all
    /// boxes are back in their slots.
    pub fn flush_removals(&mut self) {
        while let Some(id) = self.pending_removals.pop() {
            self.remove(id);
        }
    }

    /// Bump a side's kill stats — port of
    /// `g_MatrixMap->GetSideById(side)->IncStatValue(stat)`.
    pub fn inc_side_stat(&mut self, side: i32, f: impl FnOnce(&mut SideStats)) {
        if let Some(s) = self.side_stats.get_mut(side as usize) {
            f(s);
        }
    }

    /// `CSound::Play(snd)` — non-positional one-shot by its canonical
    /// Sounds-block key (MatrixSoundManager.cpp:80-260).
    pub fn queue_snd(&mut self, name: &str) {
        self.queue_snd_layer(name, SoundLayer::All);
    }

    /// `CSound::Play(snd, sl)`.
    pub fn queue_snd_layer(&mut self, name: &str, layer: SoundLayer) {
        self.pending_sounds.push(SndEvent::Play {
            key: name.to_string(),
            layer,
        });
    }

    /// `CSound::AddSound(snd, pos)` — positional with the Pos2Key
    /// cell dedup (SEF_INTERRUPT default).
    pub fn queue_snd_at(&mut self, name: &str, pos: Vec3) {
        self.pending_sounds.push(SndEvent::Add {
            key: name.to_string(),
            pos: [pos.x, pos.y, pos.z],
            layer: SoundLayer::All,
            ifl: Interrupt::Interrupt,
        });
    }

    /// `CSound::AddSound(snd, pos, SL_ALL, SEF_SKIP)` — dedup that
    /// refreshes the playing instance instead of restarting it.
    pub fn queue_snd_at_skip(&mut self, name: &str, pos: Vec3) {
        self.pending_sounds.push(SndEvent::Add {
            key: name.to_string(),
            pos: [pos.x, pos.y, pos.z],
            layer: SoundLayer::All,
            ifl: Interrupt::Skip,
        });
    }

    /// `id = CSound::Play(snd, pos, sl)` — positional immediate (no
    /// dedup); `handle` mirrors the C++ keeping the returned id.
    pub fn queue_snd_play_at(
        &mut self,
        name: &str,
        pos: Vec3,
        layer: SoundLayer,
        handle: Option<SndHandle>,
    ) {
        self.pending_sounds.push(SndEvent::PlayAt {
            key: name.to_string(),
            pos: [pos.x, pos.y, pos.z],
            layer,
            handle,
        });
    }

    /// `id = CSound::Play(id, snd, pos, sl)` — ambient retrigger
    /// (chassis loop, flyer vint, looped weapon hum).
    pub fn queue_snd_follow(&mut self, handle: SndHandle, name: &str, pos: Vec3, layer: SoundLayer) {
        self.pending_sounds.push(SndEvent::PlayHandle {
            handle,
            key: name.to_string(),
            pos: [pos.x, pos.y, pos.z],
            layer,
        });
    }

    /// `CSound::ChangePos(id, snd, pos)`.
    pub fn queue_snd_move(&mut self, handle: SndHandle, name: &str, pos: Vec3) {
        self.pending_sounds.push(SndEvent::ChangePos {
            handle,
            key: name.to_string(),
            pos: [pos.x, pos.y, pos.z],
        });
    }

    /// `CSound::StopPlay(id)`.
    pub fn queue_snd_stop(&mut self, handle: SndHandle) {
        self.pending_sounds.push(SndEvent::Stop { handle });
    }

    /// `CMatrixMap::SetMusicVolume` hook (MatrixMap.cpp:3583).
    pub fn queue_music_volume(&mut self, vol: f32) {
        self.pending_sounds.push(SndEvent::MusicVolume(vol));
    }

    /// Insert `obj` into the arena and return its handle. The ctor
    /// equivalent for `CMatrixMapStatic`-derived objects: after this
    /// returns, the subclass may call `add_lt` to opt into logic takts.
    pub fn spawn(&mut self, obj: Box<dyn MapStatic>) -> ObjectId {
        let ty = obj.core().obj_type;
        let id = if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.obj = Some(obj);
            slot.in_lt = false;
            slot.prev_lt = None;
            slot.next_lt = None;
            ObjectId {
                index,
                generation: slot.generation,
            }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 1,
                obj: Some(obj),
                in_lt: false,
                prev_lt: None,
                next_lt: None,
            });
            ObjectId {
                index,
                generation: 1,
            }
        };
        match ty {
            ObjectType::RobotAi | ObjectType::Building | ObjectType::Cannon | ObjectType::Flyer => {
                self.unit_ids.push(id)
            }
            ObjectType::MapObject => {
                self.mapobject_ids.push(id);
                self.mo_grid_dirty.set(true);
            }
            ObjectType::Empty => {}
        }
        id
    }

    /// Destroy the object at `id`. Ports `~CMatrixMapStatic`
    /// (MatrixMapStatic.cpp:87-104): removes from logic-temp list and
    /// releases the storage. Subsequent lookups return `None`
    /// (the tombstone).
    pub fn remove(&mut self, id: ObjectId) {
        if !self.is_valid(id) {
            return;
        }
        self.del_lt(id);
        if let Some(p) = self.unit_ids.iter().position(|&x| x == id) {
            self.unit_ids.swap_remove(p);
        } else if let Some(p) = self.mapobject_ids.iter().position(|&x| x == id) {
            self.mapobject_ids.swap_remove(p);
            self.mo_grid_dirty.set(true);
        }
        let slot = &mut self.slots[id.index as usize];
        slot.obj = None;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(id.index);
    }

    /// Live units only (robots/buildings/cannons/flyers) — the fast
    /// path for gameplay scans that never care about decoratives.
    pub fn iter_units(&self) -> impl Iterator<Item = ObjectId> + '_ {
        self.unit_ids.iter().copied()
    }

    /// Decorative / interactive map objects (terron, break props, …).
    pub fn iter_mapobjects(&self) -> impl Iterator<Item = ObjectId> + '_ {
        self.mapobject_ids.iter().copied()
    }

    fn ensure_mo_grid(&self) {
        if !self.mo_grid_dirty.get() {
            return;
        }
        let mut grid: std::collections::HashMap<(i32, i32), Vec<ObjectId>> =
            std::collections::HashMap::new();
        for &id in &self.mapobject_ids {
            let Some(obj) = self.get(id) else { continue };
            let c = obj.core().geo_center;
            let r = obj.core().radius.max(1.0);
            let x0 = ((c.x - r) / MO_GRID_CELL).floor() as i32;
            let x1 = ((c.x + r) / MO_GRID_CELL).floor() as i32;
            let y0 = ((c.y - r) / MO_GRID_CELL).floor() as i32;
            let y1 = ((c.y + r) / MO_GRID_CELL).floor() as i32;
            for gy in y0..=y1 {
                for gx in x0..=x1 {
                    grid.entry((gx, gy)).or_default().push(id);
                }
            }
        }
        *self.mo_grid.borrow_mut() = grid;
        self.mo_grid_dirty.set(false);
    }

    /// Decorative candidates near the segment `start → end` (grid
    /// cells the segment's AABB touches). May contain duplicates —
    /// callers keep-nearest semantics make re-tests harmless.
    fn mapobjects_near_segment(&self, start: Vec3, end: Vec3, out: &mut Vec<ObjectId>) {
        self.ensure_mo_grid();
        let grid = self.mo_grid.borrow();
        let x0 = (start.x.min(end.x) / MO_GRID_CELL).floor() as i32;
        let x1 = (start.x.max(end.x) / MO_GRID_CELL).floor() as i32;
        let y0 = (start.y.min(end.y) / MO_GRID_CELL).floor() as i32;
        let y1 = (start.y.max(end.y) / MO_GRID_CELL).floor() as i32;
        for gy in y0..=y1 {
            for gx in x0..=x1 {
                if let Some(v) = grid.get(&(gx, gy)) {
                    out.extend_from_slice(v);
                }
            }
        }
    }

    /// Decorative candidates near a circle at `pos` of `radius`.
    fn mapobjects_near_circle(&self, pos: Vec2, radius: f32, out: &mut Vec<ObjectId>) {
        self.ensure_mo_grid();
        let grid = self.mo_grid.borrow();
        let x0 = ((pos.x - radius) / MO_GRID_CELL).floor() as i32;
        let x1 = ((pos.x + radius) / MO_GRID_CELL).floor() as i32;
        let y0 = ((pos.y - radius) / MO_GRID_CELL).floor() as i32;
        let y1 = ((pos.y + radius) / MO_GRID_CELL).floor() as i32;
        for gy in y0..=y1 {
            for gx in x0..=x1 {
                if let Some(v) = grid.get(&(gx, gy)) {
                    out.extend_from_slice(v);
                }
            }
        }
    }

    pub fn is_valid(&self, id: ObjectId) -> bool {
        self.slots
            .get(id.index as usize)
            .map(|s| s.generation == id.generation && s.obj.is_some())
            .unwrap_or(false)
    }

    /// Like [`is_valid`] but returns true even when the slot's box is
    /// temporarily checked out (take-the-box pattern in
    /// `proceed_logic`/`graphic_takt`/`apply_damage`). List-membership
    /// ops (`add_lt`/`del_lt`/`in_lt`) must use this so a takt body
    /// can self-`add_lt` via the passed `&mut Objects` — matching the
    /// C++ where `this->AddLT()` works from inside one of its own
    /// methods. External read access (`get`/`get_mut`) keeps the
    /// stricter check to preserve the "object not accessible" tombstone.
    fn gen_matches(&self, id: ObjectId) -> bool {
        self.slots
            .get(id.index as usize)
            .map(|s| s.generation == id.generation)
            .unwrap_or(false)
    }

    /// True when the slot is alive but its box is temporarily checked
    /// out by a takt driver (take-the-box pattern) — i.e. the object
    /// exists but is unreachable through `get`/`get_mut` right now.
    pub fn is_checked_out(&self, id: ObjectId) -> bool {
        self.gen_matches(id) && !self.is_valid(id)
    }

    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read-only view of an object. `None` is the `m_Object == NULL`
    /// tombstone — callers must handle it.
    pub fn get(&self, id: ObjectId) -> Option<&dyn MapStatic> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.obj.as_deref()
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Option<&mut (dyn MapStatic + 'static)> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.obj.as_deref_mut()
    }

    pub fn for_each_live(&self, mut f: impl FnMut(ObjectId, &dyn MapStatic)) {
        for (i, slot) in self.slots.iter().enumerate() {
            let Some(obj) = slot.obj.as_deref() else {
                continue;
            };
            let id = ObjectId {
                index: i as u32,
                generation: slot.generation,
            };
            f(id, obj);
        }
    }

    /// Ports `InLT` (MatrixMapStatic.hpp:440). Works even while the
    /// object's box is checked out by a takt driver — the list
    /// membership flag lives on the slot, not in the box.
    pub fn in_lt(&self, id: ObjectId) -> bool {
        self.slots
            .get(id.index as usize)
            .map(|s| s.generation == id.generation && s.in_lt)
            .unwrap_or(false)
    }

    /// Ports `AddLT` (MatrixMapStatic.hpp:441). Idempotent. Uses
    /// `gen_matches` (not `is_valid`) so self-enroll during a takt
    /// body works even though the slot's box is currently checked out.
    pub fn add_lt(&mut self, id: ObjectId) {
        if !self.gen_matches(id) || self.in_lt(id) {
            return;
        }
        // LIST_ADD(this, m_FirstLogicTemp, m_LastLogicTemp,
        //          m_PrevLogicTemp, m_NextLogicTemp)
        let old_last = self.last_lt;
        {
            let slot = &mut self.slots[id.index as usize];
            slot.in_lt = true;
            slot.prev_lt = old_last;
            slot.next_lt = None;
        }
        if let Some(prev) = old_last {
            self.slots[prev.index as usize].next_lt = Some(id);
        } else {
            self.first_lt = Some(id);
        }
        self.last_lt = Some(id);
    }

    /// Ports `DelLT` (MatrixMapStatic.hpp:442). Idempotent.
    pub fn del_lt(&mut self, id: ObjectId) {
        if !self.in_lt(id) {
            return;
        }
        let (prev, next) = {
            let slot = &mut self.slots[id.index as usize];
            let pn = (slot.prev_lt, slot.next_lt);
            slot.in_lt = false;
            slot.prev_lt = None;
            slot.next_lt = None;
            pn
        };
        // If we're removing the snapshotted next cursor mid-walk, advance
        // it so ProceedLogic still lands on a live node.
        if self.next_logic_object == Some(id) {
            self.next_logic_object = next;
        }
        if let Some(p) = prev {
            self.slots[p.index as usize].next_lt = next;
        } else {
            self.first_lt = patch_head(self.first_lt, id, next);
        }
        if let Some(n) = next {
            self.slots[n.index as usize].prev_lt = prev;
        } else {
            self.last_lt = patch_head(self.last_lt, id, prev);
        }
    }

    /// Port of `CMatrixMapStatic::ProceedLogic` (MatrixMapStatic.cpp:338-362).
    /// Walks the logic-temp list, snapshotting the next link before each
    /// dispatch so a callee may remove itself (or the current tail) from
    /// the list without breaking iteration.
    ///
    /// Dispatch uses the "take the box out" pattern: we `take()` the
    /// object's `Box<dyn MapStatic>` out of its slot, invoke
    /// [`static_takt`] with `&mut Objects` (the arena sans that slot),
    /// then return the box. This lets takt bodies issue arena queries
    /// like [`Objects::find_objects`] without re-entrant borrow errors.
    /// Tombstone lookups of the taken id return `None` for the duration
    /// of the call — consistent with the C++ `m_Object == NULL` check.
    pub fn proceed_logic(&mut self, takts: i32, rng: &mut Rnd) {
        let mut cursor = self.first_lt;
        while let Some(id) = cursor {
            // Snapshot *before* dispatch, same line as the original:
            //   g_MatrixMap->m_NextLogicObject = ms->m_NextLogicTemp;
            let snapshotted_next = self.slots[id.index as usize].next_lt;
            self.next_logic_object = snapshotted_next;

            if Some(id) != self.arcaded_object {
                static_takt(self, id, takts, rng);
            }

            cursor = self.next_logic_object;
        }
        self.next_logic_object = None;
        self.flush_removals();
    }

    /// Take an object's box out of its slot — the same
    /// "take-the-box" pattern `proceed_logic` uses, exposed for
    /// callers that must run an object method needing `&mut Objects`
    /// (e.g. `Robot::big_boom`). MUST be paired with [`Self::put_obj`].
    pub fn take_obj(&mut self, id: ObjectId) -> Option<Box<dyn MapStatic>> {
        self.slots.get_mut(id.index as usize).and_then(|s| {
            if s.generation == id.generation {
                s.obj.take()
            } else {
                None
            }
        })
    }

    /// Return a box taken with [`Self::take_obj`].
    pub fn put_obj(&mut self, id: ObjectId, b: Box<dyn MapStatic>) {
        if let Some(s) = self.slots.get_mut(id.index as usize) {
            if s.generation == id.generation && s.obj.is_none() {
                s.obj = Some(b);
            }
        }
    }

    /// Used by tests + future sides code. Iterates the logic-temp list
    /// without invoking tiks — snapshot-safe (mid-iteration mutation via
    /// `del_lt` is undefined for this method; use `proceed_logic` for
    /// that contract).
    pub fn iter_logic(&self) -> LogicIter<'_> {
        LogicIter {
            objs: self,
            cursor: self.first_lt,
        }
    }

    /// All live objects (arena order). Cheap `O(slots)`; skips freed
    /// slots and generation mismatches.
    pub fn iter_live(&self) -> impl Iterator<Item = ObjectId> + '_ {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.obj.as_ref().map(|_| ObjectId {
                index: i as u32,
                generation: s.generation,
            })
        })
    }

    /// Port of `CMatrixMap::FindObjects` (MatrixMap.cpp:2617-2814). Enumerates
    /// live objects overlapping a circle of radius `radius` at `pos`, filtered
    /// by `mask` ([`fit_to_mask`]). The `oscale` factor scales the candidate
    /// object's own radius for the distance test — the C++ uses it to accept
    /// "near-misses" or tighter-fit checks depending on caller intent.
    ///
    /// `skip` excludes one object (typically `this` in the caller's context).
    ///
    /// `visit(center, id) -> Control::Continue|Break` receives each hit in
    /// arena order; the callback's matching-bool return mirrors the C++
    /// `callback` return type at MatrixMap.hpp:84.
    ///
    /// Differences from the original:
    /// * Instead of the C++ map-group index we scan the `unit_ids`
    ///   index (plus the `mo_grid` cells when the mask asks for
    ///   TRACE_OBJECT).
    /// * `m_IntersectFlagFindObjects` re-visit guard isn't needed —
    ///   each candidate is visited exactly once.
    /// * Flyer's `GetCarryingRobot` promotion (MatrixMap.cpp:2699-2710)
    ///   is redundant here: a carried robot stays in `unit_ids` and its
    ///   position tracks the flyer, so it is enumerated directly.
    pub fn find_objects(
        &self,
        pos: Vec2,
        radius: f32,
        oscale: f32,
        mask: u32,
        skip: Option<ObjectId>,
        mut visit: impl FnMut(Vec2, ObjectId) -> Control,
    ) -> bool {
        let mut hit = false;
        let mut cands: Vec<ObjectId> = self.unit_ids.clone();
        if mask & TRACE_OBJECT != 0 {
            self.mapobjects_near_circle(pos, radius, &mut cands);
        }
        for id in cands {
            let obj = match self.get(id) {
                Some(o) => o,
                None => continue,
            };
            if Some(id) == skip {
                continue;
            }
            if !fit_to_mask(obj, mask) {
                continue;
            }
            let center3 = obj.core().geo_center;
            let center = Vec2::new(center3.x, center3.y);
            // MatrixMap.cpp:2704 — `dist = length(center.xy - pos) - radius * oscale`,
            // hit when `dist < radius` (the candidate's own radius is
            // absorbed into the per-object test in FindObjectAny, so
            // here we use `obj.radius * oscale` as the subtract term).
            let dist = (center - pos).length() - obj.core().radius * oscale;
            if dist >= radius {
                continue;
            }
            hit = true;
            if visit(center, id) == Control::Break {
                return hit;
            }
        }
        hit
    }

    /// 3D-distance variant of [`find_objects`] — ports the
    /// `FindObjects(const D3DXVECTOR3 &pos, ...)` overload
    /// (MatrixMap.cpp:2814+). The weapon paths (missile seek, blast
    /// radii, flame proximity, repair seek) all use this one: an
    /// object only counts as in-range when
    /// `|geo_center - pos| - radius*oscale < radius` in full 3D.
    ///
    /// The callback's first argument is the SEARCH center (matching
    /// the C++ `callback(pos, ms, user)`), unlike `find_objects` which
    /// historically passes the object center.
    pub fn find_objects_3d(
        &self,
        pos: Vec3,
        radius: f32,
        oscale: f32,
        mask: u32,
        skip: Option<ObjectId>,
        mut visit: impl FnMut(Vec3, ObjectId) -> Control,
    ) -> bool {
        let mut hit = false;
        let mut cands: Vec<ObjectId> = self.unit_ids.clone();
        if mask & TRACE_OBJECT != 0 {
            self.mapobjects_near_circle(Vec2::new(pos.x, pos.y), radius, &mut cands);
        }
        for id in cands {
            let obj = match self.get(id) {
                Some(o) => o,
                None => continue,
            };
            if Some(id) == skip {
                continue;
            }
            if !fit_to_mask(obj, mask) {
                continue;
            }
            let dist = (obj.core().geo_center - pos).length() - obj.core().radius * oscale;
            if dist >= radius {
                continue;
            }
            hit = true;
            if visit(pos, id) == Control::Break {
                return hit;
            }
        }
        hit
    }

    /// Predicate-only variant — matches the C++ `FindObjects(..., NULL, 0)`
    /// pattern used by BEHF_SENS at MatrixObject.cpp:1501 ("is anything
    /// in the radius?"). Stops as soon as a hit is found.
    pub fn any_object_in_radius(
        &self,
        pos: Vec2,
        radius: f32,
        oscale: f32,
        mask: u32,
        skip: Option<ObjectId>,
    ) -> bool {
        self.find_objects(pos, radius, oscale, mask, skip, |_, _| Control::Break)
    }

    /// Arena ray-cast. Scans live objects matching `mask` and returns
    /// the nearest hit `(id, t)` — `t` measured along `dir` from
    /// `origin`. Ports the "iterate objects, call Pick, keep nearest"
    /// fragment that CMatrixMap::Trace runs on the `TRACE_*OBJECT`
    /// bits (MatrixMapTrace.cpp — linear scan variant; group-indexed
    /// path arrives with the spatial index port).
    ///
    /// `skip` excludes one id (typically the caller's own). `dir`
    /// should be unit-length for the returned `t` to be a distance.
    pub fn pick_object(
        &self,
        origin: Vec3,
        dir: Vec3,
        mask: u32,
        skip: Option<ObjectId>,
    ) -> Option<(ObjectId, f32)> {
        self.pick_object_within(origin, dir, f32::MAX, mask, skip)
    }

    /// [`Self::pick_object`] with a known segment length — bounds the
    /// decorative-grid walk to the segment's AABB (a mouse-pick ray or
    /// weapon trace is a few cells; an unbounded reach would touch the
    /// whole grid).
    pub fn pick_object_within(
        &self,
        origin: Vec3,
        dir: Vec3,
        max_t: f32,
        mask: u32,
        skip: Option<ObjectId>,
    ) -> Option<(ObjectId, f32)> {
        let mut best: Option<(ObjectId, f32)> = None;
        let mut test = |objs: &Objects, id: ObjectId| {
            let obj = match objs.get(id) {
                Some(o) => o,
                None => return,
            };
            if Some(id) == skip || !fit_to_mask(obj, mask) {
                return;
            }
            if let Some(t) = obj.pick(origin, dir) {
                if t < 0.0 {
                    return;
                }
                match best {
                    Some((_, bt)) if bt <= t => {}
                    _ => best = Some((id, t)),
                }
            }
        };
        for id in self.unit_ids.iter().copied() {
            test(self, id);
        }
        if mask & TRACE_OBJECT != 0 {
            let end = origin + dir * max_t.min(4000.0);
            let mut cands: Vec<ObjectId> = Vec::new();
            self.mapobjects_near_segment(origin, end, &mut cands);
            for id in cands {
                test(self, id);
            }
        }
        best
    }

    /// External entry point for `CMatrixMapStatic::Damage(...)`. Uses the
    /// take-the-box pattern so `damage()` receives `&mut Objects` as the
    /// arena-sans-self, allowing the callee to `add_lt` itself (the
    /// BEHF_BURN enrollment) or query other objects.
    ///
    /// Returns whatever the subclass `damage` returned — `true` means
    /// "this object was reset / removed mid-call" (the only case where
    /// the original returns true; see MatrixObject.cpp:1572).
    pub fn apply_damage(
        &mut self,
        target: ObjectId,
        weap: crate::matrix_game::effects::weapon::Weapon,
        pos: Vec3,
        dir: Vec3,
        attacker_side: i32,
        attacker: Option<ObjectId>,
    ) -> bool {
        let mut boxed = match self.slots.get_mut(target.index as usize).and_then(|s| {
            if s.generation == target.generation {
                s.obj.take()
            } else {
                None
            }
        }) {
            Some(b) => b,
            None => return false,
        };
        // `CSound::AddSound(SoundHit(weap), pos)` in every C++ Damage
        // entry (MatrixRobot.cpp:1880, MatrixObjectCannon.cpp:1385,
        // MatrixObjectBuilding.cpp:279, MatrixObject.cpp:111,
        // MatrixFlyer.cpp:1791). It sits AFTER the WEAPON_REPAIR
        // early-return and the DIP-state bail in each of them — repair
        // never plays a hit sound, and dead/DIP targets stay silent.
        let hit_snd = crate::matrix_game::effects::weapon::hit_sound_key(weap);
        // Decorative map objects only voice the burning DOT
        // (MatrixObject.cpp:107-112, whablaze on BEHF_BURN); the other
        // classes run the full `SoundHit` table.
        let hit_snd = if boxed.core().obj_type == ObjectType::MapObject
            && weap != crate::matrix_game::effects::weapon::WEAPON_ABLAZE
        {
            ""
        } else {
            hit_snd
        };
        if !hit_snd.is_empty()
            && weap != crate::matrix_game::effects::weapon::WEAPON_REPAIR
            && boxed.is_live()
        {
            // `SoundHit` is `AddSound(snd, pos, SL_ALL, SEF_SKIP)`
            // (MatrixEffectWeapon.cpp:846).
            self.queue_snd_at_skip(hit_snd, pos);
        }
        let result = boxed.damage(weap, pos, dir, attacker_side, attacker, target, self);
        let slot = &mut self.slots[target.index as usize];
        if slot.generation == target.generation && slot.obj.is_none() {
            slot.obj = Some(boxed);
        }
        result
    }

    /// Ports the per-object pass inside `CMatrixMapStatic::SortEndGraphicTakt`
    /// (MatrixMapStatic.cpp:755-765). Calls `takt(cms, rng)` on every
    /// live object. The C++ visibility pre-cull (the
    /// `objects_left/objects_rite` sorted window) is skipped — frustum
    /// culling lives in the renderers here, Takt bodies are cheap
    /// no-ops for BEHF_STATIC objects, and the object count is trivial.
    ///
    /// Unlike `proceed_logic`, this does NOT snapshot a `next` cursor:
    /// the C++ graphic takt walks a fixed index range filled during
    /// sort, so in-loop mutations (via `AddObject`/`DIP`) don't affect
    /// the current pass. Our `iter_live` is similarly snapshotted —
    /// `Vec` indices collected up-front, so spawning mid-takt is safe.
    pub fn graphic_takt(&mut self, cms: i32, rng: &mut Rnd) {
        // Collect IDs first so mutations inside `takt` (e.g. `add_lt`,
        // spawn) can't skip or revisit the current pass.
        let ids: Vec<ObjectId> = self.iter_live().collect();
        for id in ids {
            // Take-the-box pattern (see proceed_logic): the dispatched
            // object gets `&mut Objects` with its own slot temporarily
            // empty, so callees can freely query/mutate the arena
            // without aliasing.
            let mut boxed = match self.slots[id.index as usize].obj.take() {
                Some(b) => b,
                None => continue,
            };
            boxed.takt(cms, rng, self);
            // Only re-install if the slot wasn't overwritten. A callee
            // could have despawned this id (bumping generation + adding
            // to free list); in that case we drop the box.
            let slot = &mut self.slots[id.index as usize];
            if slot.generation == id.generation && slot.obj.is_none() {
                slot.obj = Some(boxed);
            }
        }
    }
}

/// If the head/tail pointer was pointing at the removed node, repoint it
/// at the replacement. Otherwise leave it alone — the removed node was
/// somewhere mid-list and the caller already patched its neighbours.
fn patch_head(
    current: Option<ObjectId>,
    removed: ObjectId,
    replacement: Option<ObjectId>,
) -> Option<ObjectId> {
    if current == Some(removed) {
        replacement
    } else {
        current
    }
}

pub struct LogicIter<'a> {
    objs: &'a Objects,
    cursor: Option<ObjectId>,
}

impl<'a> Iterator for LogicIter<'a> {
    type Item = ObjectId;
    fn next(&mut self) -> Option<ObjectId> {
        let id = self.cursor?;
        self.cursor = self.objs.slots[id.index as usize].next_lt;
        Some(id)
    }
}

/// Port of `CMatrixMapStatic::StaticTakt` (MatrixMapStatic.cpp:107-143).
/// Decrements ablaze/shorted TTLs (including the robots'
/// `SwitchAnimation(ANIMATION_STAY)` on SHORTED→clear,
/// MatrixMapStatic.cpp:133) then delegates to the subclass'
/// `logic_takt`.
pub(crate) fn static_takt(objs: &mut Objects, id: ObjectId, ms: i32, rng: &mut Rnd) {
    // Take-the-box pattern — see `proceed_logic`. This also handles the
    // case where `id` points at a freed slot (returns early, same as
    // the C++ tombstone branch).
    let mut boxed = match objs.slots.get_mut(id.index as usize).and_then(|s| {
        if s.generation == id.generation {
            s.obj.take()
        } else {
            None
        }
    }) {
        Some(b) => b,
        None => return,
    };

    if boxed.object_state() & OBJECT_STATE_ABLAZE != 0 {
        let ttl = (boxed.ablaze_ttl() - ms).max(0);
        boxed.set_ablaze_ttl(ttl);
        if ttl == 0 {
            boxed.object_state_clear(OBJECT_STATE_ABLAZE);
        }
    }
    if boxed.object_state() & OBJECT_STATE_SHORTED != 0 {
        let ttl = (boxed.shorted_ttl() - ms).max(0);
        boxed.set_shorted_ttl(ttl);
        if ttl == 0 {
            boxed.object_state_clear(OBJECT_STATE_SHORTED);
            // `if (IsRobot()) SwitchAnimation(ANIMATION_STAY)`
            // (MatrixMapStatic.cpp:133). No chassis VO = headless run,
            // where animation state is render-only anyway.
            if boxed.core().obj_type == ObjectType::RobotAi {
                let r: &mut crate::matrix_game::robot::Robot = unsafe {
                    &mut *(boxed.as_mut() as *mut dyn MapStatic
                        as *mut crate::matrix_game::robot::Robot)
                };
                if let Some(vo) = crate::matrix_lib::three_g::vector_object::chassis_vo(
                    r.chassis.kind_index(),
                ) {
                    r.switch_animation(&vo, crate::matrix_game::robot::Animation::Stay);
                }
            }
        }
    }
    boxed.logic_takt(ms, rng, objs);

    // Re-install unless the callee despawned us.
    let slot = &mut objs.slots[id.index as usize];
    if slot.generation == id.generation && slot.obj.is_none() {
        slot.obj = Some(boxed);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Counting stub that also lets each invocation optionally mutate the
    /// arena via a closure. The closure captures `&mut Objects`, which
    /// would collide with the &mut borrow `static_takt` holds on the
    /// target object — we work around that by parking the closure on a
    /// shared handle (`Rc<RefCell<...>>`) and only draining it from the
    /// outside after a `proceed_logic` call.
    struct Stub {
        core: ObjectCore,
        rchange: u32,
        state: u32,
        ablaze: i32,
        shorted: i32,
        log: Rc<RefCell<Vec<(&'static str, i32)>>>,
        name: &'static str,
    }

    impl Stub {
        fn new(name: &'static str, log: Rc<RefCell<Vec<(&'static str, i32)>>>) -> Self {
            Self {
                core: ObjectCore {
                    obj_type: ObjectType::MapObject,
                    ..Default::default()
                },
                rchange: MR_ALL,
                state: 0,
                ablaze: 0,
                shorted: 0,
                log,
                name,
            }
        }
    }

    impl MapStatic for Stub {
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
            self.state
        }
        fn object_state_set(&mut self, b: u32) {
            self.state |= b;
        }
        fn object_state_clear(&mut self, b: u32) {
            self.state &= !b;
        }
        fn ablaze_ttl(&self) -> i32 {
            self.ablaze
        }
        fn set_ablaze_ttl(&mut self, t: i32) {
            self.ablaze = t;
        }
        fn shorted_ttl(&self) -> i32 {
            self.shorted
        }
        fn set_shorted_ttl(&mut self, t: i32) {
            self.shorted = t;
        }
        fn r_need(&mut self, _need: u32) {}
        fn takt(&mut self, _cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {}
        fn logic_takt(&mut self, cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {
            self.log.borrow_mut().push((self.name, cms));
        }
    }

    fn mk_stub(
        name: &'static str,
        log: Rc<RefCell<Vec<(&'static str, i32)>>>,
    ) -> Box<dyn MapStatic> {
        Box::new(Stub::new(name, log))
    }

    #[test]
    fn add_lt_is_idempotent_and_del_lt_on_absent_is_noop() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        assert!(!objs.in_lt(a));
        objs.del_lt(a); // noop
        objs.add_lt(a);
        objs.add_lt(a); // idempotent
        assert!(objs.in_lt(a));
        assert_eq!(objs.iter_logic().collect::<Vec<_>>(), vec![a]);
        objs.del_lt(a);
        objs.del_lt(a); // idempotent
        assert!(!objs.in_lt(a));
        assert!(objs.iter_logic().next().is_none());
    }

    #[test]
    fn proceed_logic_visits_each_object_once() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        let b = objs.spawn(mk_stub("b", log.clone()));
        let c = objs.spawn(mk_stub("c", log.clone()));
        objs.add_lt(a);
        objs.add_lt(b);
        objs.add_lt(c);

        objs.proceed_logic(10, &mut Rnd::new(1));
        assert_eq!(log.borrow().clone(), vec![("a", 10), ("b", 10), ("c", 10)]);
    }

    #[test]
    fn fit_to_mask_covers_every_object_type() {
        use crate::matrix_game::common::{
            TRACE_BUILDING, TRACE_CANNON, TRACE_FLYER, TRACE_OBJECT, TRACE_ROBOT,
        };
        fn stub(kind: ObjectType, state: u32) -> Box<dyn MapStatic> {
            let log = Rc::new(RefCell::new(Vec::new()));
            let mut s = Stub::new("s", log);
            s.core.obj_type = kind;
            s.state = state;
            Box::new(s)
        }
        let rob = stub(ObjectType::RobotAi, 0);
        let cnn = stub(ObjectType::Cannon, 0);
        let bld = stub(ObjectType::Building, 0);
        let obj = stub(ObjectType::MapObject, 0);
        let fly = stub(ObjectType::Flyer, 0);
        let emp = stub(ObjectType::Empty, 0);
        assert!(fit_to_mask(&*rob, TRACE_ROBOT));
        assert!(!fit_to_mask(&*rob, TRACE_OBJECT));
        assert!(fit_to_mask(&*cnn, TRACE_CANNON));
        assert!(fit_to_mask(&*bld, TRACE_BUILDING));
        assert!(fit_to_mask(&*obj, TRACE_OBJECT));
        assert!(fit_to_mask(&*fly, TRACE_FLYER));
        assert!(!fit_to_mask(&*emp, u32::MAX)); // Empty always fails.
                                                // OR'd mask — robot also passes against a TRACE_ANYOBJECT-style mask.
        assert!(fit_to_mask(&*rob, TRACE_ROBOT | TRACE_OBJECT));
    }

    #[test]
    fn fit_to_mask_respects_skip_invisible() {
        use crate::matrix_game::common::{TRACE_OBJECT, TRACE_SKIP_INVISIBLE};
        let visible = Stub::new("v", Rc::new(RefCell::new(Vec::new())));
        let mut invis = Stub::new("i", Rc::new(RefCell::new(Vec::new())));
        invis.state = OBJECT_STATE_TRACE_INVISIBLE;
        // Matching type, skip-invisible set: invisible object is excluded.
        assert!(fit_to_mask(&visible, TRACE_OBJECT | TRACE_SKIP_INVISIBLE));
        assert!(!fit_to_mask(&invis, TRACE_OBJECT | TRACE_SKIP_INVISIBLE));
        // Without skip-invisible flag, invisible still passes type filter.
        assert!(fit_to_mask(&invis, TRACE_OBJECT));
    }

    #[test]
    fn find_objects_returns_all_hits_in_radius_and_honors_skip() {
        use crate::matrix_game::common::TRACE_OBJECT;
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let mut make = |x: f32, y: f32| -> ObjectId {
            let mut s = Stub::new("x", log.clone());
            s.core.obj_type = ObjectType::MapObject;
            s.core.geo_center = glam::Vec3::new(x, y, 0.0);
            objs.spawn(Box::new(s))
        };
        let a = make(0.0, 0.0);
        let b = make(5.0, 0.0);
        let c = make(20.0, 0.0); // outside radius 10
        let _d = make(3.0, 4.0); // distance 5, inside

        let mut hits = Vec::new();
        let got = objs.find_objects(glam::Vec2::ZERO, 10.0, 1.0, TRACE_OBJECT, None, |_, id| {
            hits.push(id);
            Control::Continue
        });
        assert!(got, "at least one hit");
        // `a`, `b`, `d` expected; `c` excluded.
        assert!(hits.contains(&a));
        assert!(hits.contains(&b));
        assert!(!hits.contains(&c));

        // Skip `a` → it's excluded from the enumeration.
        hits.clear();
        objs.find_objects(
            glam::Vec2::ZERO,
            10.0,
            1.0,
            TRACE_OBJECT,
            Some(a),
            |_, id| {
                hits.push(id);
                Control::Continue
            },
        );
        assert!(!hits.contains(&a));
    }

    #[test]
    fn find_objects_stops_on_break() {
        use crate::matrix_game::common::TRACE_OBJECT;
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        for _ in 0..5 {
            let mut s = Stub::new("x", log.clone());
            s.core.obj_type = ObjectType::MapObject;
            s.core.geo_center = glam::Vec3::ZERO;
            objs.spawn(Box::new(s));
        }
        let mut visited = 0;
        let got = objs.find_objects(glam::Vec2::ZERO, 10.0, 1.0, TRACE_OBJECT, None, |_, _| {
            visited += 1;
            Control::Break
        });
        assert!(got);
        assert_eq!(visited, 1, "Break stops after the first hit");

        // `any_object_in_radius` shortcut — semantically the same.
        assert!(objs.any_object_in_radius(glam::Vec2::ZERO, 10.0, 1.0, TRACE_OBJECT, None));
    }

    #[test]
    fn find_objects_filters_by_type_mask() {
        use crate::matrix_game::common::{TRACE_OBJECT, TRACE_ROBOT};
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let mut mapobj = Stub::new("o", log.clone());
        mapobj.core.obj_type = ObjectType::MapObject;
        mapobj.core.geo_center = glam::Vec3::ZERO;
        objs.spawn(Box::new(mapobj));
        // Asking for robots: no match.
        assert!(!objs.any_object_in_radius(glam::Vec2::ZERO, 10.0, 1.0, TRACE_ROBOT, None,));
        // Asking for objects: hit.
        assert!(objs.any_object_in_radius(glam::Vec2::ZERO, 10.0, 1.0, TRACE_OBJECT, None,));
    }

    #[test]
    fn ray_sphere_pick_hits_front_face_when_outside() {
        // Sphere at (10, 0, 0) radius 2. Ray from origin down +X.
        // Front face at x=8 → t=8.
        let c = Vec3::new(10.0, 0.0, 0.0);
        let o = Vec3::ZERO;
        let d = Vec3::X;
        let t = ray_sphere_pick(c, 2.0, o, d).unwrap();
        assert!((t - 8.0).abs() < 1e-5);
    }

    #[test]
    fn ray_sphere_pick_misses_off_axis() {
        let c = Vec3::new(10.0, 5.0, 0.0);
        let o = Vec3::ZERO;
        let d = Vec3::X;
        // Sphere centered 5 units off the ray axis with radius 2 — misses.
        assert!(ray_sphere_pick(c, 2.0, o, d).is_none());
    }

    #[test]
    fn ray_sphere_pick_returns_exit_when_inside_sphere() {
        let c = Vec3::ZERO;
        let o = Vec3::ZERO;
        let d = Vec3::X;
        // Origin at center → -b - sqrt_d = -radius, -b + sqrt_d = +radius.
        // Nearest positive t = +radius.
        let t = ray_sphere_pick(c, 3.0, o, d).unwrap();
        assert!((t - 3.0).abs() < 1e-5);
    }

    #[test]
    fn ray_sphere_pick_rejects_zero_radius() {
        assert!(ray_sphere_pick(Vec3::ZERO, 0.0, Vec3::new(-10.0, 0.0, 0.0), Vec3::X).is_none());
    }

    #[test]
    fn pick_object_returns_nearest_hit_matching_mask() {
        use crate::matrix_game::common::{TRACE_BUILDING, TRACE_OBJECT, TRACE_ROBOT};
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let mut mk = |x: f32, kind: ObjectType, r: f32| -> ObjectId {
            let mut s = Stub::new("x", log.clone());
            s.core.obj_type = kind;
            s.core.geo_center = Vec3::new(x, 0.0, 0.0);
            s.core.radius = r;
            objs.spawn(Box::new(s))
        };
        let _near = mk(5.0, ObjectType::MapObject, 1.0); // t=4
        let mid = mk(10.0, ObjectType::Building, 1.0); // t=9
        let _far = mk(20.0, ObjectType::Building, 1.0); // t=19

        // TRACE_OBJECT: should hit the MapObject at t≈4.
        let (_, t) = objs
            .pick_object(Vec3::ZERO, Vec3::X, TRACE_OBJECT, None)
            .unwrap();
        assert!((t - 4.0).abs() < 1e-5);

        // TRACE_BUILDING: should hit the NEAREST building (mid, t≈9)
        // not the farther one.
        let (hit_id, t) = objs
            .pick_object(Vec3::ZERO, Vec3::X, TRACE_BUILDING, None)
            .unwrap();
        assert_eq!(hit_id, mid);
        assert!((t - 9.0).abs() < 1e-5);

        // TRACE_ROBOT: nothing matches.
        assert!(objs
            .pick_object(Vec3::ZERO, Vec3::X, TRACE_ROBOT, None)
            .is_none());
    }

    #[test]
    fn graphic_takt_visits_every_live_object_ignoring_logic_temp_membership() {
        // Ports the SortEndGraphicTakt contract (MatrixMapStatic.cpp:755):
        // every visible static object gets Takt, regardless of whether
        // it opted into the logic-temp list. The Rust stub below tracks
        // takt calls — we add `a` and `c` to logic-temp but leave `b`
        // off, and assert all three get their `takt` called.
        type TaktLog = Rc<RefCell<Vec<(&'static str, &'static str, i32)>>>;
        struct TaktCounter {
            core: ObjectCore,
            rchange: u32,
            state: u32,
            ablaze: i32,
            shorted: i32,
            name: &'static str,
            log: TaktLog,
        }
        impl MapStatic for TaktCounter {
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
                self.state
            }
            fn object_state_set(&mut self, b: u32) {
                self.state |= b;
            }
            fn object_state_clear(&mut self, b: u32) {
                self.state &= !b;
            }
            fn ablaze_ttl(&self) -> i32 {
                self.ablaze
            }
            fn set_ablaze_ttl(&mut self, t: i32) {
                self.ablaze = t;
            }
            fn shorted_ttl(&self) -> i32 {
                self.shorted
            }
            fn set_shorted_ttl(&mut self, t: i32) {
                self.shorted = t;
            }
            fn r_need(&mut self, _: u32) {}
            fn takt(&mut self, cms: i32, _: &mut Rnd, _: &mut Objects) {
                self.log.borrow_mut().push(("takt", self.name, cms));
            }
            fn logic_takt(&mut self, cms: i32, _: &mut Rnd, _: &mut Objects) {
                self.log.borrow_mut().push(("logic", self.name, cms));
            }
        }
        fn mk(name: &'static str, log: TaktLog) -> Box<dyn MapStatic> {
            Box::new(TaktCounter {
                core: ObjectCore {
                    obj_type: ObjectType::MapObject,
                    ..Default::default()
                },
                rchange: 0,
                state: 0,
                ablaze: 0,
                shorted: 0,
                name,
                log,
            })
        }

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk("a", log.clone()));
        let _b = objs.spawn(mk("b", log.clone()));
        let c = objs.spawn(mk("c", log.clone()));
        objs.add_lt(a);
        objs.add_lt(c); // `b` stays off the logic list.

        let mut rng = Rnd::new(1);
        objs.graphic_takt(33, &mut rng);
        // All three reached by graphic_takt.
        let got: Vec<_> = log.borrow().iter().cloned().collect();
        assert_eq!(
            got,
            vec![("takt", "a", 33), ("takt", "b", 33), ("takt", "c", 33)]
        );

        log.borrow_mut().clear();
        objs.proceed_logic(10, &mut rng);
        let got: Vec<_> = log.borrow().iter().cloned().collect();
        // Only the logic-temp members get logic_takt.
        assert_eq!(got, vec![("logic", "a", 10), ("logic", "c", 10)]);
    }

    #[test]
    fn graphic_takt_with_nonpositive_step_in_world_is_noop() {
        // World::graphic_takt guards on step<=0; Objects::graphic_takt
        // here accepts any value but we never invoke it negatively in
        // production. Sanity-check that Objects::graphic_takt with 0
        // still iterates (ports the visible-takt contract — zero-delta
        // frame hitches shouldn't drop takts; it's the caller's job).
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let _a = objs.spawn(mk_stub("a", log.clone()));
        let mut rng = Rnd::new(1);
        objs.graphic_takt(0, &mut rng);
        // mk_stub's `takt` is a noop; this just verifies we don't crash.
    }

    #[test]
    fn arcaded_object_is_skipped() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        let b = objs.spawn(mk_stub("b", log.clone()));
        objs.add_lt(a);
        objs.add_lt(b);
        objs.arcaded_object = Some(b);

        objs.proceed_logic(5, &mut Rnd::new(1));
        assert_eq!(log.borrow().clone(), vec![("a", 5)]);
    }

    #[test]
    fn ablaze_ttl_clears_on_boundary() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        objs.get_mut(a)
            .unwrap()
            .object_state_set(OBJECT_STATE_ABLAZE);
        objs.get_mut(a).unwrap().set_ablaze_ttl(10);
        objs.add_lt(a);

        objs.proceed_logic(10, &mut Rnd::new(1));
        let o = objs.get(a).unwrap();
        assert_eq!(o.ablaze_ttl(), 0);
        assert_eq!(o.object_state() & OBJECT_STATE_ABLAZE, 0);
    }

    #[test]
    fn ablaze_ttl_does_not_clear_early() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        objs.get_mut(a)
            .unwrap()
            .object_state_set(OBJECT_STATE_ABLAZE);
        objs.get_mut(a).unwrap().set_ablaze_ttl(15);
        objs.add_lt(a);

        objs.proceed_logic(10, &mut Rnd::new(1));
        let o = objs.get(a).unwrap();
        assert_eq!(o.ablaze_ttl(), 5);
        assert_ne!(o.object_state() & OBJECT_STATE_ABLAZE, 0);
    }

    /// Exercises the `m_NextLogicObject` snapshot (MatrixMapStatic.cpp:349).
    /// A stub removes its *successor* mid-walk; `proceed_logic` must still
    /// land on a live node for the next iteration.
    #[test]
    fn del_lt_of_next_during_takt_does_not_crash_walk() {
        struct Remover {
            core: ObjectCore,
            rchange: u32,
            state: u32,
            ablaze: i32,
            shorted: i32,
            target: ObjectId,
        }
        impl MapStatic for Remover {
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
                self.state
            }
            fn object_state_set(&mut self, b: u32) {
                self.state |= b;
            }
            fn object_state_clear(&mut self, b: u32) {
                self.state &= !b;
            }
            fn ablaze_ttl(&self) -> i32 {
                self.ablaze
            }
            fn set_ablaze_ttl(&mut self, t: i32) {
                self.ablaze = t;
            }
            fn shorted_ttl(&self) -> i32 {
                self.shorted
            }
            fn set_shorted_ttl(&mut self, t: i32) {
                self.shorted = t;
            }
            fn r_need(&mut self, _n: u32) {}
            fn takt(&mut self, _c: i32, _r: &mut Rnd, _o: &mut Objects) {}
            fn logic_takt(&mut self, _c: i32, _r: &mut Rnd, objs: &mut Objects) {
                // Natural with the take-the-box pattern: `objs` is the
                // arena sans this slot, so mutating a different slot's
                // list membership is a plain safe call.
                objs.del_lt(self.target);
            }
        }

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();

        let a_id = objs.spawn(Box::new(Remover {
            core: ObjectCore {
                obj_type: ObjectType::MapObject,
                ..Default::default()
            },
            rchange: 0,
            state: 0,
            ablaze: 0,
            shorted: 0,
            target: ObjectId {
                index: 999,
                generation: 0,
            }, // patched below
        }));
        let b_id = objs.spawn(mk_stub("b", log.clone()));
        let c_id = objs.spawn(mk_stub("c", log.clone()));
        // Patch the remover's target now that `b_id` is known.
        {
            let obj = objs.get_mut(a_id).unwrap();
            let rem = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Remover) };
            rem.target = b_id;
        }
        objs.add_lt(a_id);
        objs.add_lt(b_id);
        objs.add_lt(c_id);

        objs.proceed_logic(10, &mut Rnd::new(1));
        // `b` should NOT be ticked (the remover nuked it mid-walk before
        // the cursor reached it); `c` should still tick because the
        // snapshot re-pointed to it.
        assert_eq!(log.borrow().clone(), vec![("c", 10)]);
        assert!(!objs.in_lt(b_id));
    }

    /// Ports the other half of the `m_NextLogicObject` contract: objects
    /// added *during* a walk are NOT visited this call
    /// (MatrixMapStatic.cpp:349 snapshots the next link before dispatch).
    #[test]
    fn add_lt_during_takt_visits_on_next_call_only() {
        struct Adder {
            core: ObjectCore,
            rchange: u32,
            state: u32,
            ablaze: i32,
            shorted: i32,
            new_id: ObjectId,
        }
        impl MapStatic for Adder {
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
                self.state
            }
            fn object_state_set(&mut self, b: u32) {
                self.state |= b;
            }
            fn object_state_clear(&mut self, b: u32) {
                self.state &= !b;
            }
            fn ablaze_ttl(&self) -> i32 {
                self.ablaze
            }
            fn set_ablaze_ttl(&mut self, t: i32) {
                self.ablaze = t;
            }
            fn shorted_ttl(&self) -> i32 {
                self.shorted
            }
            fn set_shorted_ttl(&mut self, t: i32) {
                self.shorted = t;
            }
            fn r_need(&mut self, _n: u32) {}
            fn takt(&mut self, _c: i32, _r: &mut Rnd, _o: &mut Objects) {}
            fn logic_takt(&mut self, _c: i32, _r: &mut Rnd, objs: &mut Objects) {
                objs.add_lt(self.new_id);
            }
        }

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        // Pre-create the "future" object but keep it out of the list.
        let future_id = objs.spawn(mk_stub("future", log.clone()));
        let adder_id = objs.spawn(Box::new(Adder {
            core: ObjectCore {
                obj_type: ObjectType::MapObject,
                ..Default::default()
            },
            rchange: 0,
            state: 0,
            ablaze: 0,
            shorted: 0,
            new_id: future_id,
        }));
        objs.add_lt(adder_id);

        objs.proceed_logic(10, &mut Rnd::new(1)); // 1st call: only `adder` runs.
        assert!(
            log.borrow().is_empty(),
            "future tick ran in 1st walk: {:?}",
            log.borrow()
        );
        assert!(objs.in_lt(future_id));

        objs.proceed_logic(7, &mut Rnd::new(1)); // 2nd call: `adder` + `future`.
        assert_eq!(log.borrow().clone(), vec![("future", 7)]);
    }
}
