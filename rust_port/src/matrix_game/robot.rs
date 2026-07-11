//! Minimal port of `CMatrixRobotAI` (MatrixRobot.{cpp,hpp}).
//!
//! CMatrixRobotAI is ~5000 lines (AI, pathfinding, state machine,
//! chassis/armor/weapon composition, animation, combat). This file
//! ports only the subset needed for a faithful build-robot flow:
//! place the robot at a spawn position with a side / chassis kind,
//! join the arena, and carry enough state that FindObjects +
//! selection queries do the right thing. AI and combat land with
//! their owning subsystems.

use crate::matrix_game::logic::Rnd;
use crate::matrix_game::logic::{self, ROBOT_MOVECELLS_PER_SIZE};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectId, ObjectType, Objects, MR_ALL, MR_MATRIX,
    ROBOT_FLAG_COLLISION, ROBOT_FLAG_DISABLE_MANUAL, ROBOT_FLAG_ROT_LEFT, ROBOT_FLAG_ROT_RIGHT,
};
use crate::matrix_game::map_trace::{self, MovePath, ROBOT_FOOTPRINT_HALF};
use crate::matrix_game::object_cannon::{Cannon, CannonState};

/// Port of `ERobotState` (MatrixRobot.hpp). Full enum is 20+
/// states (DIP, MOVE, PATROL, ATTACK, CARRYING, CAPTURING,
/// FALLING, EMBRYO…); the rest land with combat / capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RobotState {
    /// `ROBOT_IN_SPAWN` — base is rising, robot is attached to the
    /// platform and moves up with it (MatrixRobot.cpp:2227).
    InSpawn,
    /// `ROBOT_BASE_MOVEOUT` — platform has fully risen; robot is
    /// driving off the pad on a straight `m_Forward` run.
    /// MatrixRobot.cpp:785-811.
    BaseMoveOut,
    /// `ROBOT_SUCCESSFULLY_BUILD` — spawn complete, the base was
    /// closed, the robot hands off to normal idle AI
    /// (MatrixRobot.cpp:801).
    Idle,
    /// `ROBOT_BASE_CAPTURE` — riding the base platform down while
    /// capturing it (MatrixRobot.cpp:1300). The order loop still
    /// runs in this state (MatrixRobot.cpp:830).
    BaseCapture,
    /// `ROBOT_CARRYING` — dangling under a transport flyer
    /// (MatrixRobot.cpp:612-716).
    Carrying,
    /// `ROBOT_FALLING` — released by the flyer, gravity until ground
    /// contact (MatrixRobot.cpp:570-607).
    Falling,
    /// `ROBOT_DIP` — destroyed; the wreck decays for `dip_ttl` ms
    /// (the unit-scatter death animation isn't ported) and then
    /// despawns.
    Dip,
}

/// Port of `ERobotUnitKind` — chassis rows (MatrixRobot.hpp). Only
/// the five chassis variants are modelled here; armor / weapon / head
/// kinds land with their render paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChassisKind {
    Pneumatic = 0,
    Wheel = 1,
    Track = 2,
    AntiGravity = 3,
    Hovercraft = 4,
}

impl ChassisKind {
    /// Config-table / passability-bit (nsh) index — `m_Unit[0].m_Kind - 1`
    /// with RUK_CHASSIS_HOVERCRAFT=4, ANTIGRAVITY=5 (MatrixConfig.hpp:39-43).
    /// NOT the enum discriminant: Hovercraft sorts BEFORE AntiGravity.
    pub fn kind_index(self) -> usize {
        match self {
            ChassisKind::Pneumatic => 0,
            ChassisKind::Wheel => 1,
            ChassisKind::Track => 2,
            ChassisKind::Hovercraft => 3,
            ChassisKind::AntiGravity => 4,
        }
    }

    /// Mesh-AABB half-diagonal of an assembled robot on this chassis —
    /// the C++ `m_Radius` (JoinToGroup, MatrixMapStatic.cpp:178),
    /// probed per chassis via probe_robot_bounds.rs. Logic-owned so
    /// browser and headless sims agree (the renderer must not write
    /// core.radius, same rule as geo_center).
    pub fn mesh_radius(self) -> f32 {
        [20.6, 20.2, 22.1, 22.1, 18.4][self.kind_index()]
    }

    /// Default height offset above the spawn platform for each
    /// chassis (approximate, matches the per-chassis height tables
    /// scattered through MatrixRobot.cpp).
    pub fn spawn_z_offset(self) -> f32 {
        match self {
            ChassisKind::Pneumatic => 9.0,
            ChassisKind::Wheel => 8.0,
            ChassisKind::Track => 5.0,
            ChassisKind::AntiGravity => 5.0,
            ChassisKind::Hovercraft => 7.0,
        }
    }
}

/// Port of the per-instance fields of `CMatrixRobotAI`. The AI state
/// machine, weapon unit array, group membership, environment scan,
/// capture plans — all deferred. Fields present here are what the
/// Damage / selection / spatial-query paths read.
pub struct Robot {
    core: ObjectCore,
    rchange: u32,
    object_state: u32,
    ablaze_ttl: i32,
    shorted_ttl: i32,

    /// `m_PosX` / `m_PosY` — world-space placement (MatrixRobot.hpp).
    /// The C++ stores XY separately from `m_Core->m_Matrix._41/_42`
    /// because the takt advances XY for movement before rebuilding the
    /// full matrix. We keep the same split.
    pub pos_x: f32,
    pub pos_y: f32,
    /// `m_PosZ` — chassis height above ground.
    pub pos_z: f32,
    /// `m_Side` — 1 = player, 2..=8 = AI. Matches CMatrixSide convention.
    pub side: i32,
    pub chassis: ChassisKind,

    /// `m_HitPoint` / `m_HitPointMax` — robots use these too.
    pub hit_point: f32,
    pub hit_point_max: f32,

    /// Building that spawned this robot (`m_Base` in C++, used by the
    /// "return to base" AI path). `None` for free-floating robots.
    pub base: Option<ObjectId>,
    /// Port of `m_CurrState` (MatrixRobot.hpp). Drives the Takt
    /// dispatch between spawn-animation / move-out / idle.
    pub state: RobotState,
    /// Base build_z at spawn time — anchor for the platform-rise
    /// animation. Updated when base reassigns.
    pub base_anchor_z: f32,
    /// Chassis height delta stored so Takt can restore pos_z after
    /// the platform animation.
    pub chassis_dz: f32,
    /// Port of `CMatrixRobot::m_Forward` (MatrixRobot.hpp). Unit
    /// direction in the XY plane the robot drives during
    /// `ROBOT_BASE_MOVEOUT` (MatrixRobot.cpp:787 —
    /// `LowLevelMove(ms, m_Forward * 100, ...)`). Seeded from the
    /// base's angle at spawn time so the robot exits perpendicular
    /// to the platform front.
    pub forward: glam::Vec2,

    /// `m_ChassisData.m_LastSolePos` — last spot where a track/wheel
    /// sole decal was stamped (MatrixRobot.cpp:737).
    pub last_sole_pos: glam::Vec2,

    /// `IsInPosition()` — set by the "terron" map easter egg when this
    /// robot stands on the trigger cell (MatrixRobot.cpp:1952-1955).
    pub in_position: bool,

    /// Pneumatic foot-linking (`m_ChassisData.m_LinkPos`/`m_LinkPrevFrame`,
    /// MatrixObjectRobot.cpp:1517-1590). `link_prev_frame < 0` = not linked.
    pub link_pos: glam::Vec2,
    pub link_prev_frame: i32,

    /// Port of `m_OrdersList[MAX_ORDERS]` + `m_OrdersInPool`
    /// (MatrixRobot.hpp:249-250). Top of the list is the active
    /// order dispatched each LogicTakt.
    pub orders: OrderList,
    /// Port of `m_MovePath[MatrixPathMoveMax]` + `m_MovePathCur` +
    /// `m_MovePathCnt` (MatrixRobot.hpp:262-265). Computed by
    /// `update_move_path` (= ZoneMoveCalc in C++) whenever an active
    /// ROT_MOVE_TO has no path yet, and consumed by `move_by_move_path`
    /// (MatrixRobot.cpp:1708-1764).
    pub move_path: MovePath,
    /// Port of `m_ZoneCur` (MatrixRobot.hpp:258) — zone the robot
    /// currently stands in, updated by `zone_cur_find`.
    pub zone_cur: i32,
    /// Port of `m_ZoneDes` (MatrixRobot.hpp) — zone of the MOVE_TO
    /// destination, found via `ZoneFindNear`.
    pub zone_des: i32,
    /// Port of `m_ZonePath` + `m_ZonePathCnt` (MatrixRobot.hpp:260) —
    /// the zone-level route from `FindPathInZone`.
    pub zone_path: Vec<i32>,
    /// Port of `m_ZonePathNext` (MatrixRobot.hpp:261) — index of the
    /// next zone to enter; -1 when no zone route is active.
    pub zone_path_next: i32,
    /// Port of `m_DesX`/`m_DesY` (MatrixRobot.hpp:257). Move-cell
    /// coords of the currently-pending MOVE_TO destination.
    pub des_x: i32,
    pub des_y: i32,
    /// Port of `m_MapX`/`m_MapY` (MatrixRobot.hpp:255). Robot's own
    /// upper-left move-cell corner, recomputed each tick from pos.
    pub map_x: i32,
    pub map_y: i32,
    /// Port of `m_Velocity` (MatrixRobot.hpp — in CMatrixRobot
    /// ancestor). Per-10ms velocity integrated by LowLevelMove.
    /// Only XY used in the spawn/move-out flow.
    pub velocity: glam::Vec2,
    /// Port of `m_Speed` (MatrixRobot.hpp). Scalar magnitude set by
    /// `Seek` and zeroed by `LowLevelStop`; the `do_animation` tail
    /// branch (MatrixRobot.cpp:331) gates Stay vs Move on
    /// `fabs(m_Speed) <= 0.01`.
    pub speed: f32,
    /// Port of `m_HullForward` (MatrixRobot.hpp). The hull direction
    /// tracks m_Forward through a smoothed rotation; initial value
    /// = m_Forward at spawn (CConstructor.cpp:192).
    pub hull_forward: glam::Vec2,
    /// Arcade-mode `SetMaxSpeed(GetMaxSpeed() * SPEED_BOOST)`
    /// (MatrixSide.cpp:1316,1325,1348). The C++ mutates `m_maxSpeed`
    /// directly; we keep the chassis base speed in config and store
    /// the boost factor, so `max_speed()` = base × this.
    pub max_speed_boost: f32,
    /// `this == GetArcadedObject()` snapshot, refreshed at the top of
    /// each logic takt so deep helpers (rotate_hull, get_lost) can
    /// test it without threading `&Objects` through.
    pub is_arcaded: bool,
    /// `m_PrevTurnSign` + `m_HullRotCnt` (MatrixRobot.hpp) — hull-servo
    /// sound hysteresis while arcaded (MatrixRobot.cpp:2255-2276).
    prev_turn_sign: i32,
    hull_rot_cnt: i32,
    /// rotate_hull has no `&mut Objects`; it raises this and the logic
    /// takt drains it into `queue_snd_at` (PlayHullSound).
    hull_sound_pending: bool,
    /// Port of `m_MoveTestPos` + `m_MoveTestChange`
    /// (MatrixRobot.hpp:268-269). Dead-reckoning watchdog: if the
    /// robot hasn't moved >5 units in 2 seconds, the MOVE_TO path
    /// gets wiped and recomputed (MatrixRobot.cpp:1753-1761).
    pub move_test_pos: glam::Vec2,
    pub move_test_change_ms: i64,

    // ── Collision / group-speed state (MatrixRobot.hpp:243-245,312-313).
    /// `m_Cols` — collision counter for this tick; reset at the top of
    /// LowLevelMove (MatrixRobot.cpp:2568) and incremented by the
    /// RobotToObject / SphereRobotToAABB handlers.
    pub cols: i32,
    /// `m_ColsWeight` — long-window collision weight (MatrixRobot.hpp:243).
    pub cols_weight: i32,
    /// `m_ColsWeight2` — short-window collision weight, used to gate the
    /// "near-stop" branch in `CollisionCallback` (MatrixRobot.cpp:2923).
    pub cols_weight2: i32,
    /// `m_GroupSpeed` — group formation speed cap (MatrixRobot.hpp:312).
    /// Seek falls back to `m_maxSpeed` when this is ≤ 0.001
    /// (MatrixRobot.cpp:2418).
    pub group_speed: f32,
    /// `m_ColSpeed` — speed cap when another robot is in front
    /// (MatrixRobot.hpp:313). Reset to 100 at the top of
    /// RobotToObjectCollision (MatrixRobot.cpp:3070) when no far-robot
    /// collision was detected; clamped by the CollisionCallback when
    /// one is.
    pub col_speed: f32,
    /// Port of `m_Environment` (CInfo, Logic/MatrixEnvironment.h) —
    /// the robot's sensing state: enemy list, combat target, assigned
    /// road-network place (`env.place`) and player-assigned fallback
    /// cell (`env.place_add`).
    pub env: crate::matrix_game::logic::environment::Info,
    /// Port of `m_GroupLogic` (MatrixRobot.hpp:349-350) — index into
    /// the side's `player_groups` array; -1 = none.
    pub group_logic: i32,
    /// Snapshot of `side->m_PlayerGroup[m_GroupLogic].m_RoadPath` —
    /// the C++ reads the side's route live inside `ZonePathCalc`
    /// (MatrixRobot.cpp:1587-1588); the arena split makes that
    /// borrow impossible, so the PG layer hands each group member an
    /// `Arc` of the route whenever it changes.
    pub group_road_path: Option<std::sync::Arc<crate::matrix_game::road_network::RoadRoute>>,
    /// AI-side "can't reach destination" marker (MatrixRobot.cpp:
    /// 1601-1604) — the side AI reassigns team + logic group.
    pub zone_path_fail_reteam: bool,

    /// `m_CollAvoid` (MatrixRobot.hpp — per-instance). Unit vector the
    /// WallAvoid hint pushes Seek to steer along when hugging a wall.
    /// In the shipped C++ build `WallAvoid` has an unconditional `return;`
    /// at its top (MatrixRobot.cpp:3080), so this stays zero in practice
    /// — but we still carry and reset it each tick for faithfulness.
    pub coll_avoid: glam::Vec2,

    /// Port of `CMatrixRobot::m_Animation` (MatrixObjectRobot.hpp:
    /// 33-43). High-level symbolic state that `SwitchAnimation` uses
    /// to decide the transition graph.
    pub animation: Animation,
    /// Port of the first `SMatrixRobotUnit::m_Graph` animation
    /// cursor (MatrixObjectRobot.hpp). One `AnimState` per part —
    /// chassis is the active animation under `do_chassis_animation`,
    /// the rest get a constant-rate `Takt(cms)` per
    /// MatrixObjectRobot.cpp:759-776 with an idle reset at anim end.
    pub chassis_anim: crate::matrix_lib::three_g::vector_object::AnimState,
    pub armor_anim: crate::matrix_lib::three_g::vector_object::AnimState,
    pub head_anim: crate::matrix_lib::three_g::vector_object::AnimState,
    pub weapon_anims: [crate::matrix_lib::three_g::vector_object::AnimState; 5],

    /// Port of `CMatrixRobot::m_Name` (MatrixRobot.hpp). Display name
    /// composed from chassis/armor/head label parts + total damage
    /// suffix by `GetConstructionName` (CConstructor.cpp:1540-1572).
    pub name: String,

    /// Port of `CMatrixRobot::m_Team` (MatrixRobot.hpp). Team index
    /// (0..=2) the robot belongs to within its side. Set by
    /// `set_team` either when StackRobot/ProduceRobot assigns the
    /// new bot to a team, or when group reassignment moves it.
    pub team: i32,

    /// Cached configuration the robot was built from — chassis kind /
    /// armor / head / weapons / cached damage. Lets `do_animation`,
    /// AI, and the building progress UI access the per-component
    /// metadata without round-tripping through `g_Config`.
    pub config: crate::matrix_game::interface::constructor::RobotConfig,

    // ── Combat state (MatrixRobot.hpp) ───────────────────────────────
    /// Own arena id — set right after spawn (same pattern as
    /// `Building::self_id`); the weapons need it as handler target.
    pub self_id: Option<ObjectId>,
    /// `m_Weapons[]` + `m_WeaponsCnt` — created by
    /// [`Robot::robot_weapon_init`] on the first live takt.
    pub weapons: Vec<BotWeapon>,
    /// `m_WeaponDir` — world-space point the weapons aim at (set by
    /// the ROT_FIRE order processing, MatrixRobot.cpp:1152-1175).
    pub weapon_dir: glam::Vec3,
    /// `m_LastDelayDamageSide` + `m_NextTimeAblaze` /
    /// `m_NextTimeShorted` — DOT bookkeeping.
    pub last_delay_damage_side: i32,
    pub next_time_ablaze: i64,
    pub next_time_shorted: i64,
    /// `m_BombProtect` / `m_LightProtect` / `m_AimProtect` — head
    /// armor coefficients in `[0, 1]` (MatrixRobot.cpp:4281-4288).
    pub bomb_protect: f32,
    pub light_protect: f32,
    pub aim_protect: f32,
    /// `m_MaxFireDist` / `m_MinFireDist` / `m_RepairDist` — weapon
    /// reach extremes computed by `CalcRobotMass`
    /// (MatrixRobot.cpp:4294-4330); BIGBOOM excluded, REPAIR feeds
    /// only `repair_dist`.
    pub max_fire_dist: f32,
    pub min_fire_dist: f32,
    pub repair_dist: f32,
    /// `m_HaveRepair` (MatrixRobot.hpp:295): 0 = none, 1 = has a
    /// repairer, 2 = ALL weapons are repairers (bomb not counted).
    pub have_repair: i32,
    /// `m_Strength` (MatrixRobot.hpp:361) — cached by `GatherInfo`'s
    /// `CalcStrength` call each env refresh.
    pub strength: f32,
    /// `m_MiniMapFlashTime` (MatrixRobot.hpp) — under-attack blink
    /// countdown the minimap reads; set to FLASH_PERIOD (1000ms) on
    /// non-friendly damage.
    pub mini_map_flash_time: i32,
    /// `m_TimeWithBase` (MatrixRobot.cpp:361-371) — spawn watchdog:
    /// a robot attached to a base for >10s closes the base and dies.
    pub time_with_base: i32,
    /// `m_KeelWaterCount` / `m_ChassisData.m_DustCount`
    /// (MatrixObjectRobot.cpp:911-948) — fractional spawn accumulators.
    pub keelwater_count: f32,
    pub dust_count: f32,
    /// `m_CargoFlyer` — the transport carrying us (ROBOT_CARRYING).
    pub cargo_flyer: Option<ObjectId>,
    /// `m_FallingSpeed` (MatrixRobot.cpp:570-607).
    pub falling_speed: f32,
    /// `GetEnv()->m_LastHit*` timestamps `HitTo` maintains — kept for
    /// the AI port even though the env itself isn't here yet.
    pub last_hit_enemy_ms: i64,
    pub last_hit_friendly_ms: i64,
    pub last_hit_target_ms: i64,
    /// `ROBOT_MUST_DIE` flag — set by [`Robot::big_boom`]; the next
    /// logic takt routes it into `Damage(WEAPON_INSTANT_DEATH)`.
    pub must_die: bool,
    /// Set when completing a base capture consumed this robot
    /// (`StaticDelete(this)` at MatrixRobot.cpp:1317) — the takt must
    /// bail immediately.
    pub consumed_by_capture: bool,
    /// Port of `m_CaptureCandidates[MAX_CAPTURE_CANDIDATES]` +
    /// `m_CaptureCandidatesCnt` (MatrixRobot.hpp:252-253). Each entry
    /// is `(building, time-before-capture ms)`.
    pub capture_candidates: Vec<(ObjectId, i32)>,
    /// `m_ShowHitpointTime` — floating HP-bar visibility timer.
    pub show_hitpoint_time: i32,
    /// Wreck decay timer for the `Dip` state.
    pub dip_ttl: i32,
    /// DIP wreck pieces (`m_Unit[]` in death mode) — flying chunks
    /// with smoke trails that damage what they land on
    /// (MatrixRobot.cpp:178-283).
    pub dip_units: Vec<DipUnit>,
    /// Per-pylon weapon muzzle transforms, written back by the robot
    /// renderer each frame from the real part chain — the
    /// `m_Graph->GetMatrixById(1) * m_Unit.m_Matrix` lookup the C++
    /// fire control does (MatrixRobot.cpp:1390-1394). `pos` is the
    /// barrel-tip bone in world space, `dir` the barrel's Y axis.
    pub weapon_mounts: [Option<WeaponMount>; 5],
    /// Repair-gun glow anchors (weapon VO bones 1/2 in world space),
    /// renderer write-back like `weapon_mounts` — port of
    /// `SWeaponRepairData::Update` (MatrixRobot.cpp:63-73).
    pub repair_glows: [Option<(glam::Vec3, glam::Vec3)>; 5],
    /// Assembled-part world positions (renderer write-back, see
    /// [`DipPartSnapshot`]).
    pub part_snapshot: DipPartSnapshot,
}

/// One weapon's world-space muzzle (see `weapon_mounts`).
#[derive(Debug, Clone, Copy)]
pub struct WeaponMount {
    pub pos: glam::Vec3,
    pub dir: glam::Vec3,
}

/// One scattering wreck piece — the robot's own part mesh flying with
/// a tumble (`m_Unit[]` in ROBOT_DIP mode). The robot renderer draws
/// it via `part` + the spin/pos fields; the hull (MRT_ARMOR) gets no
/// unit at all — the C++ zeroes its TTL so it's never drawn
/// (MatrixRobot.cpp:2088-2093, MatrixObjectRobot.cpp:1049).
pub struct DipUnit {
    pub pos: glam::Vec3,
    pub velocity: glam::Vec3,
    pub ttl: f32,
    pub dp: f32,
    pub dy: f32,
    pub dr: f32,
    pub part: DipPart,
    pub invert: bool,
    /// Base Z-rotation. Non-zero only for the stay-in-place chassis,
    /// whose C++ unit matrix is never recomputed and so keeps the
    /// robot's death orientation; flying pieces snap to the pure
    /// YawPitchRoll spin (DIPTakt only updates moving units).
    pub yaw: f32,
    /// Spin-clock freeze stamp — set when the piece lands (zero
    /// velocity skips the C++ matrix update, freezing orientation).
    pub freeze_t: Option<f32>,
    pub smoke: crate::matrix_game::effects::smoke_and_fire::Smoke,
}

/// Which of the robot's parts a [`DipUnit`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DipPart {
    Chassis,
    Head,
    /// Pylon index into `config.weapon`.
    Weapon(usize),
}

/// World positions (+ weapon invert flags) of the assembled parts,
/// written back by the renderer each frame — the real part chain only
/// exists render-side. `init_dip_scatter` seeds each wreck piece at
/// its part's position like the C++ (`m_Unit[i].m_Matrix._41-43`).
#[derive(Debug, Default, Clone, Copy)]
pub struct DipPartSnapshot {
    pub chassis: Option<glam::Vec3>,
    pub head: Option<glam::Vec3>,
    pub weapons: [Option<(glam::Vec3, bool)>; 5],
}

/// Port of `SBotWeapon` (MatrixRobot.hpp:50-121) — one mounted weapon
/// with its heat bookkeeping. The effect itself lives in the
/// `Objects::weapons` slab.
pub struct BotWeapon {
    pub weapon: crate::matrix_game::effects::weapon::WeaponId,
    /// Index into `config.weapon` (which pylon this is).
    pub unit: usize,
    pub heat: i32,
    pub heat_period: i32,
    pub heat_mod: i32,
    pub cool_down_period: i32,
    pub cool_down_mod: i32,
    pub on: bool,
}

/// Port of `EAnimation` (MatrixObjectRobot.hpp:32-47).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Animation {
    Off,
    Stay,
    Move,
    BeginMove,
    EndMove,
    MoveBack,
    BeginMoveBack,
    EndMoveBack,
    Rotate,
}

impl Robot {
    /// Minimal ctor equivalent. The C++ constructor initialises
    /// thousands of lines of unit arrays / AI state; we only set
    /// what's needed to place a robot in the world.
    pub fn new(pos: glam::Vec3, side: i32, chassis: ChassisKind) -> Self {
        let core = ObjectCore {
            obj_type: ObjectType::RobotAi,
            // JoinToGroup semantics, logic-owned (the renderer never
            // writes these): geo_center mid-body ≈ pos_z + 9, radius =
            // per-chassis mesh-AABB half-diagonal like the C++
            // m_Radius. The old 6-unit sphere at z+3 made hitscan
            // beams pass clean over robots.
            geo_center: pos + glam::Vec3::new(0.0, 0.0, 9.0),
            radius: chassis.mesh_radius(),
            matrix: glam::Mat4::from_translation(pos),
            ..Default::default()
        };
        Self {
            core,
            rchange: MR_ALL,
            object_state: 0,
            ablaze_ttl: 0,
            shorted_ttl: 0,
            pos_x: pos.x,
            pos_y: pos.y,
            pos_z: pos.z,
            side,
            chassis,
            hit_point: 100.0,
            hit_point_max: 100.0,
            base: None,
            state: RobotState::Idle,
            base_anchor_z: pos.z,
            chassis_dz: chassis.spawn_z_offset(),
            forward: glam::Vec2::new(0.0, -1.0),
            last_sole_pos: glam::Vec2::new(-1.0e9, -1.0e9),
            link_pos: glam::Vec2::ZERO,
            link_prev_frame: -1,
            in_position: false,
            orders: OrderList::new(),
            move_path: MovePath::default(),
            zone_cur: -1,
            zone_des: -1,
            zone_path: Vec::new(),
            zone_path_next: -1,
            des_x: 0,
            des_y: 0,
            map_x: 0,
            map_y: 0,
            velocity: glam::Vec2::ZERO,
            speed: 0.0,
            hull_forward: glam::Vec2::new(0.0, -1.0),
            max_speed_boost: 1.0,
            is_arcaded: false,
            prev_turn_sign: 0,
            hull_rot_cnt: 0,
            hull_sound_pending: false,
            move_test_pos: glam::Vec2::ZERO,
            move_test_change_ms: 0,
            // MatrixRobot.cpp:2970 — m_ColSpeed starts at 100 and Seek
            // clamps to min(m_GroupSpeed, m_ColSpeed). Without this the
            // first-tick velocity underflows to 0.
            cols: 0,
            cols_weight: 0,
            cols_weight2: 0,
            group_speed: 0.0,
            col_speed: 100.0,
            coll_avoid: glam::Vec2::ZERO,
            env: crate::matrix_game::logic::environment::Info::new(),
            group_logic: -1,
            group_road_path: None,
            zone_path_fail_reteam: false,
            animation: Animation::Off,
            chassis_anim: Default::default(),
            armor_anim: Default::default(),
            head_anim: Default::default(),
            weapon_anims: Default::default(),
            name: String::new(),
            team: 0,
            config: crate::matrix_game::interface::constructor::RobotConfig::new(),
            self_id: None,
            weapons: Vec::new(),
            weapon_dir: glam::Vec3::ZERO,
            last_delay_damage_side: 0,
            next_time_ablaze: 0,
            next_time_shorted: 0,
            bomb_protect: 0.0,
            light_protect: 0.0,
            aim_protect: 0.0,
            max_fire_dist: 0.0,
            min_fire_dist: 0.0,
            repair_dist: 0.0,
            have_repair: 0,
            strength: 0.0,
            mini_map_flash_time: 0,
            time_with_base: 0,
            keelwater_count: 0.0,
            dust_count: 0.0,
            cargo_flyer: None,
            falling_speed: 0.0,
            last_hit_enemy_ms: 0,
            last_hit_friendly_ms: 0,
            last_hit_target_ms: 0,
            must_die: false,
            consumed_by_capture: false,
            capture_candidates: Vec::new(),
            show_hitpoint_time: 0,
            dip_ttl: 0,
            dip_units: Vec::new(),
            weapon_mounts: [None; 5],
            repair_glows: [None; 5],
            part_snapshot: DipPartSnapshot::default(),
        }
    }

    /// Port of `CMatrixRobot::SetTeam(int)` (MatrixRobot.hpp). Used by
    /// `CConstructor::ProduceRobot` / `BuildSpecialBot` to assign the
    /// freshly-spawned robot to one of three groups.
    pub fn set_team(&mut self, team: i32) {
        self.team = team.clamp(0, 2);
    }

    /// Port of `CMatrixRobot::m_Team` accessor.
    pub fn team(&self) -> i32 {
        self.team
    }

    /// Port of `CMatrixRobot::SwitchAnimation(EAnimation a)`
    /// (MatrixObjectRobot.cpp:1368-1506). Direct translation of
    /// the transition graph:
    ///   - MOVE : from STAY/ENDMOVE/OFF/ROTATE/(back variants) →
    ///     BeginMove (one-shot), then Move (looped). From
    ///     BeginMove → Move when the one-shot ends.
    ///   - STAY : from MOVE/BeginMove → EndMove (one-shot), then
    ///     Stay. From ROTATE/OFF → Stay directly.
    ///   - OFF  : `m_Animation = a`, no anim change.
    ///   - others: stubbed as direct-set for now.
    pub fn switch_animation(
        &mut self,
        vo: &crate::matrix_lib::three_g::vector_object::VoMesh,
        target: Animation,
    ) {
        // MatrixObjectRobot.cpp:1376 — ANIMATION_OFF is a no-op on
        // the cursor; just record the state.
        if matches!(target, Animation::Off) {
            self.animation = Animation::Off;
            return;
        }
        // MatrixObjectRobot.cpp:1383-1389 — on any non-OFF target
        // that differs from current, reset `m_NextAnimTime` to the
        // current game time so speed-based advancement starts fresh.
        if target != self.animation {
            self.chassis_anim.next_anim_time = crate::matrix_game::map::current_elapsed_ms() as f64;
        }

        if target == Animation::Move {
            // :1391-1421. Only triggers the begin-move one-shot
            // when we're transitioning from a standing / rotate
            // state. If the chassis has no "BeginMove" anim,
            // fall through directly to Move (looped).
            if matches!(
                self.animation,
                Animation::Stay
                    | Animation::EndMove
                    | Animation::Off
                    | Animation::Rotate
                    | Animation::MoveBack
                    | Animation::EndMoveBack
                    | Animation::BeginMoveBack
            ) {
                self.animation = Animation::BeginMove;
                // SetAnimByName returns true on "not found"
                // (MatrixObjectRobot.cpp:1404).
                if self.chassis_anim.set_anim_by_name(vo, "BeginMove", false) {
                    self.animation = Animation::Move;
                    self.chassis_anim.set_anim_by_name(vo, "Move", true);
                }
            } else if self.animation == Animation::BeginMove {
                // :1414 — if BeginMove one-shot finished, advance
                // to looped Move.
                if self.chassis_anim.is_anim_end(vo) {
                    self.animation = Animation::Move;
                    self.chassis_anim.set_anim_by_name(vo, "Move", true);
                }
            }
        } else if target == Animation::Stay {
            // :1454-1493.
            if matches!(self.animation, Animation::Move | Animation::BeginMove) {
                self.animation = Animation::EndMove;
                if self.chassis_anim.set_anim_by_name(vo, "EndMove", false) {
                    self.animation = Animation::Stay;
                    self.chassis_anim.set_anim_by_name(vo, "Stay", true);
                }
            } else if self.animation == Animation::EndMove {
                if self.chassis_anim.is_anim_end(vo) {
                    self.animation = Animation::Stay;
                    self.chassis_anim.set_anim_by_name(vo, "Stay", true);
                }
            } else if matches!(
                self.animation,
                Animation::MoveBack | Animation::BeginMoveBack
            ) {
                self.animation = Animation::EndMoveBack;
                if self.chassis_anim.set_anim_by_name(vo, "EndMoveBack", false) {
                    self.animation = Animation::Stay;
                    self.chassis_anim.set_anim_by_name(vo, "Stay", true);
                }
            } else if self.animation == Animation::EndMoveBack {
                if self.chassis_anim.is_anim_end(vo) {
                    self.animation = Animation::Stay;
                    self.chassis_anim.set_anim_by_name(vo, "Stay", true);
                }
            } else if matches!(self.animation, Animation::Rotate | Animation::Off) {
                self.chassis_anim.set_anim_by_name(vo, "Stay", true);
                self.animation = Animation::Stay;
            }
        } else {
            // Other targets: :1502-1504 — direct assignment, no
            // cursor change. Covers ROTATE / DIE / OFF fallbacks.
            self.animation = target;
        }
    }

    /// Port of `CMatrixRobotAI::SetBase(CMatrixBuilding*)`
    /// (MatrixRobot.hpp). Called by the build-stack before spawning
    /// so the AI can route back to its factory.
    pub fn set_base(&mut self, base: ObjectId) {
        self.base = Some(base);
        self.time_with_base = 0; // C++ SetBase resets the 10s watchdog
    }

    /// Port of `CMatrixRobotAI::RobotSpawn(pBase)` (MatrixRobot.cpp:
    /// 2178-2231). Minimal subset: record the parent, flip state to
    /// `InSpawn`, seed the platform-tracking anchor. The caller is
    /// responsible for calling `pBase.open()` — the C++ does that
    /// at line 2228 and we keep the call on the Building side so
    /// logic_takt can drive the state machine in one place.
    ///
    /// `base_angle_quad` is the base's `m_Angle` (0..=3). Direct port
    /// of the forward-by-angle switch at CConstructor.cpp:183-190
    /// which is how the original seeds `m_Build->m_Forward` before
    /// adding the robot to `m_BS`.
    pub fn robot_spawn(&mut self, base_id: ObjectId, base_angle_quad: i32, base_build_z: f32) {
        self.base = Some(base_id);
        self.state = RobotState::InSpawn;
        self.base_anchor_z = base_build_z;
        self.pos_z = base_build_z;
        // MatrixRobot.cpp:2187 — no manual orders while riding the
        // spawn pad; cleared at move-out completion (:802). Also
        // exempts the robot from AllocPlaceForOrderOnTop's base-close.
        self.object_state |= ROBOT_FLAG_DISABLE_MANUAL;

        self.forward = match base_angle_quad & 3 {
            0 => glam::Vec2::new(0.0, 1.0),
            1 => glam::Vec2::new(-1.0, 0.0),
            2 => glam::Vec2::new(0.0, -1.0),
            _ => glam::Vec2::new(1.0, 0.0),
        };
        // CConstructor.cpp:192 — m_HullForward initialized to m_Forward.
        self.hull_forward = self.forward;
    }

    // ─────────────────────────────────────────────────────────────
    // Orders API — ports of the `AllocPlaceForOrderOnTop` / `MoveTo`
    // / `MoveReturn` / `StopMoving` / `GetLost` methods from
    // MatrixRobot.cpp.

    /// Port of `CMatrixRobotAI::MoveTo(mx, my)` (MatrixRobot.cpp:4625-
    /// 4656). Drops any stale MOVE_TO / STOP_MOVE orders, pushes a
    /// fresh ROT_MOVE_TO on top, clears the move-path so the
    /// dispatcher recomputes it on the next LogicTakt.
    pub fn move_to(&mut self, mx: i32, my: i32) {
        if std::env::var_os("MG_TRACE_CAPTURE").is_some() {
            if let Some(o) = self.orders.iter().find(|o| {
                o.ty == OrderType::MoveTo && o.phase == OrderPhase::CaptureMoving
            }) {
                eprintln!(
                    "[capture-trace] move_to({mx},{my}) replaced capture-MoveTo({},{}) of {:?} at ({:.0},{:.0})",
                    o.p1, o.p2, self.self_id, self.pos_x, self.pos_y
                );
            }
        }
        self.orders.remove_type(OrderType::MoveTo);
        self.orders.remove_type(OrderType::MoveToBack);
        self.orders.remove_type(OrderType::StopMove);
        self.orders
            .push_top(Order::set(OrderType::MoveTo, mx as f32, my as f32, 0.0, 0));
        self.zone_des = -1;
        self.zone_path.clear();
        self.zone_path_next = -1;
        self.move_path.clear();
        self.des_x = mx;
        self.des_y = my;
    }

    /// Port of `CMatrixRobotAI::MoveToBack` (MatrixRobot.cpp:4658-4684).
    pub fn move_to_back(&mut self, mx: i32, my: i32) {
        self.orders.remove_type(OrderType::MoveTo);
        self.orders.remove_type(OrderType::MoveToBack);
        self.orders.remove_type(OrderType::StopMove);
        self.orders.push_top(Order::set(
            OrderType::MoveToBack,
            mx as f32,
            my as f32,
            0.0,
            0,
        ));
        self.zone_des = -1;
        self.zone_path.clear();
        self.zone_path_next = -1;
        self.move_path.clear();
        self.des_x = mx;
        self.des_y = my;
    }

    /// Port of `CMatrixRobotAI::MoveReturn` (MatrixRobot.cpp:4687-4699):
    /// an existing MOVE_RETURN turns into a MOVE_TO at the new coords;
    /// otherwise a fresh MOVE_RETURN lands on top.
    pub fn move_return(&mut self, mx: i32, my: i32) {
        for o in self.orders.iter_mut() {
            if o.ty == OrderType::MoveReturn {
                *o = Order::set(OrderType::MoveTo, mx as f32, my as f32, 0.0, 0);
                return;
            }
        }
        self.orders.push_top(Order::set(
            OrderType::MoveReturn,
            mx as f32,
            my as f32,
            0.0,
            0,
        ));
    }

    /// Port of `CMatrixRobotAI::MoveToHigh` (MatrixRobot.cpp:4701-4713).
    pub fn move_to_high(&mut self, mx: i32, my: i32) {
        self.stop_capture();
        self.orders.remove_type(OrderType::MoveReturn);
        self.move_to(mx, my);
    }

    /// Port of `CMatrixRobotAI::StopCapture` (MatrixRobot.cpp:4810-4834):
    /// converts CAPTURE_FACTORY orders into STOP_CAPTURE (keeping the
    /// target) and capture-phase MOVE_TOs into STOP_MOVE.
    pub fn stop_capture(&mut self) {
        let mut released: Vec<ObjectId> = Vec::new();
        for o in self.orders.iter_mut() {
            if o.ty == OrderType::CaptureFactory {
                // C++ Release()s the slot BEFORE reading GetStatic()
                // (:4826-4828), so the STOP_CAPTURE order carries a
                // NULL target and the building's capturer is reset.
                if let Some(t) = o.target {
                    released.push(t);
                }
                *o = Order::set_static(OrderType::StopCapture, None);
            } else if o.ty == OrderType::MoveTo && o.phase == OrderPhase::CaptureMoving {
                if let Some(t) = o.target {
                    released.push(t);
                }
                *o = Order::set(OrderType::StopMove, 0.0, 0.0, 0.0, 0);
            }
        }
        self.orders.released.extend(released);
    }

    /// Port of `CMatrixRobotAI::CaptureFactory` (MatrixRobot.cpp:4786-
    /// 4796).
    pub fn capture_factory(&mut self, factory: ObjectId) {
        self.orders.remove_type(OrderType::CaptureFactory);
        // Deviation from the C++: drop any pre-capture MoveReturn. The
        // C++ keeps it, and once the approach MoveTo is gone the stale
        // return point yanks the robot away mid-capture (it can end up
        // "Capturing" a factory from across the map, permanently
        // claiming it and deadlocking its side). Step-asides during
        // the approach re-save the capture destination, so nothing
        // legitimate is lost.
        self.orders.remove_type(OrderType::MoveReturn);
        self.orders
            .push_top(Order::set_static(OrderType::CaptureFactory, Some(factory)));
    }

    /// Port of `CMatrixRobotAI::GetCaptureFactory` (MatrixRobot.cpp:
    /// 4798-4808).
    /// Port of `SelectByGroup` (MatrixRobot.cpp:5240-5253) — the flag
    /// half; the selection-ring effect is owned by the selection
    /// renderer which keys off `is_selected`.
    pub fn select_by_group(&mut self) {
        self.object_state |= crate::matrix_game::map_static::ROBOT_FLAG_SGROUP;
    }

    /// Port of `UnSelect` (MatrixRobot.cpp:5270-5276).
    pub fn unselect(&mut self) {
        self.object_state &= !(crate::matrix_game::map_static::ROBOT_FLAG_SGROUP
            | crate::matrix_game::map_static::ROBOT_FLAG_SARCADE);
    }

    /// Port of `IsSelected` (MatrixRobot.cpp:5303-5310).
    pub fn is_selected(&self) -> bool {
        self.object_state
            & (crate::matrix_game::map_static::ROBOT_FLAG_SGROUP
                | crate::matrix_game::map_static::ROBOT_FLAG_SARCADE)
            != 0
    }

    /// `Carry(flyer, quick_connect)` attach half
    /// (MatrixObjectRobot.cpp:1290-1356).
    pub fn carry_attach(&mut self, flyer: ObjectId) {
        self.cargo_flyer = Some(flyer);
        self.state = RobotState::Carrying;
        self.falling_speed = 0.0;
    }

    /// The ROBOT_CARRYING physics (MatrixRobot.cpp:612-716): chase the
    /// flyer hook with a sprung lag, dangle-sway the up vector, keep
    /// the elevator field anchored. The C++ also pushes out of nearby
    /// objects via FindObjects/CollisionRobots — robots in transit are
    /// airborne for a couple of seconds, so that nudge is skipped.
    fn carrying_takt(
        &mut self,
        cms: i32,
        map: &crate::matrix_game::map::GameMap,
        objs: &mut Objects,
    ) {
        const CARRYING_DISTANCE: f32 = 20.0; // MatrixObjectRobot.hpp:63
        const CARRYING_SPEED: f64 = 0.996; // MatrixObjectRobot.hpp:64

        let Some(fid) = self.cargo_flyer else {
            self.state = RobotState::Idle;
            return;
        };
        let Some((fpos, fangle, elevator_active)) =
            crate::matrix_game::logic::flyer_ref(objs, fid).map(|f| {
                (
                    f.pos,
                    f.angle,
                    f.carry
                        .elevator
                        .as_ref()
                        .map(|e| e.activated)
                        .unwrap_or(false),
                )
            })
        else {
            self.carry_release();
            return;
        };
        let ms = cms as f32;
        let mul = (1.0 - CARRYING_SPEED.powf(ms as f64)) as f32;

        // Mass-factor reel-in once the field is up (MatrixRobot.cpp:618-625).
        let mass_factor = {
            let f = crate::matrix_game::logic::flyer_mut(objs, fid).unwrap();
            if elevator_active {
                f.carry.robot_mass_factor = (f.carry.robot_mass_factor + 0.0005 * ms).min(1.0);
            }
            f.carry.robot_mass_factor
        };

        self.falling_speed = (self.falling_speed - ms * 0.013).max(0.0);

        let geo = self.core.geo_center;
        let delta = fpos - glam::Vec3::new(0.0, 0.0, CARRYING_DISTANCE) - geo;
        let mut mv = delta * mul * mass_factor;
        mv.z -= self.falling_speed * ms * 0.013;
        self.pos_x += mv.x;
        self.pos_y += mv.y;
        self.pos_z += mv.z;
        // Separation push off nearby robots while dangling
        // (MatrixRobot.cpp:678-691): nearest robot within half-radii
        // shoves the carried body out along the connecting vector.
        {
            let my_r = self.core.radius;
            let my_c = glam::Vec3::new(self.pos_x, self.pos_y, self.core.geo_center.z);
            let mut nearest: Option<(glam::Vec3, f32, f32)> = None;
            for oid in objs.iter_units() {
                if Some(oid) == self.self_id {
                    continue;
                }
                let Some(o) = crate::matrix_game::logic::robot_ref(objs, oid) else {
                    continue;
                };
                if !o.is_live() {
                    continue;
                }
                let d = my_c - o.core.geo_center;
                let d2 = d.length_squared();
                let reach = my_r * 0.5 + o.core.radius * 0.5;
                if d2 < reach * reach && nearest.map(|(_, nd2, _)| d2 < nd2).unwrap_or(true) {
                    nearest = Some((d, d2, o.core.radius));
                }
            }
            if let Some((d, d2, or)) = nearest {
                let dist = d2.sqrt();
                let norm = if dist >= 1.0e-6 { d / dist } else { d.normalize_or_zero() };
                let push = norm * (or * 0.5 + my_r * 0.5 - dist);
                self.pos_x += push.x;
                self.pos_y += push.y;
                self.pos_z += push.z;
            }
        }
        let cz = map.get_z(self.pos_x, self.pos_y);
        if self.pos_z < cz {
            self.pos_z = cz;
        }
        self.core.geo_center = glam::Vec3::new(self.pos_x, self.pos_y, self.pos_z + 9.0);

        // Sway + heading follow (MatrixRobot.cpp:651-668).
        {
            let f = crate::matrix_game::logic::flyer_mut(objs, fid).unwrap();
            f.carry.robot_up.x += mv.x * 0.03 + f.carry.robot_up_back.x * mul;
            f.carry.robot_up.y += mv.y * 0.03 + f.carry.robot_up_back.y * mul;
            let l = glam::Vec2::new(f.carry.robot_up.x, f.carry.robot_up.y).length();
            if l > 1.0 {
                f.carry.robot_up.x /= l;
                f.carry.robot_up.y /= l;
            }
            f.carry.robot_up_back.x -= f.carry.robot_up.x * 0.08;
            f.carry.robot_up_back.y -= f.carry.robot_up.y * 0.08;
            f.carry.robot_up_back.z = 0.0;
            let mul2 = 0.999f64.powf(ms as f64) as f32;
            f.carry.robot_up.x *= mul2;
            f.carry.robot_up.y *= mul2;
            let l = glam::Vec2::new(f.carry.robot_up_back.x, f.carry.robot_up_back.y).length();
            if l > 0.4 {
                f.carry.robot_up_back.x *= 0.4 / l;
                f.carry.robot_up_back.y *= 0.4 / l;
            }
            let da = mul
                * mul
                * crate::matrix_game::flyer::angle_dist(fangle, f.carry.robot_angle);
            f.carry.robot_angle -= da;
            let (s, c) = f.carry.robot_angle.sin_cos();
            let rf = f.carry.robot_forward;
            self.forward = glam::Vec2::new(c * rf.x + s * rf.y, -s * rf.x + c * rf.y);

            // Elevator field create / re-anchor (MatrixRobot.cpp:705-713).
            let radius = self.core.radius.max(10.0);
            let fwd3 = glam::Vec3::new(self.forward.x, self.forward.y, 0.0);
            match f.carry.elevator.as_mut() {
                None => {
                    f.carry.elevator = Some(
                        crate::matrix_game::effects::elevator_field::ElevatorField::new(
                            fpos,
                            self.core.geo_center,
                            radius,
                            fwd3,
                        ),
                    );
                }
                Some(e) => {
                    e.update_data(
                        fpos,
                        glam::Vec3::new(self.pos_x, self.pos_y, self.pos_z),
                        radius,
                        fwd3,
                    );
                }
            }
        }
    }

    /// `Carry(NULL)` — release into free fall
    /// (MatrixObjectRobot.cpp:1322-1330).
    pub fn carry_release(&mut self) {
        self.cargo_flyer = None;
        // Never resurrect a dead robot into free fall — the C++ can't
        // hit this (death detaches the carry link first), but a robot
        // checked out of the arena mid-takt can still be reached here.
        if self.state == RobotState::Dip {
            return;
        }
        self.state = RobotState::Falling;
        self.velocity = glam::Vec2::ZERO;
        self.falling_speed = 0.0;
    }

    /// `GetMaxSpeed()` (MatrixRobot.hpp:339) — per-chassis config
    /// speed with the arcade SPEED_BOOST factor applied.
    pub fn max_speed(&self) -> f32 {
        chassis_max_speed(self.chassis) * self.max_speed_boost
    }

    /// `m_GroupSpeed = GetMaxSpeed()` (MatrixSide.cpp:2359) — the
    /// CalcMaxSpeed reset.
    pub fn reset_group_speed(&mut self) {
        self.group_speed = self.max_speed();
    }

    /// `m_GroupSpeed *= k` (MatrixSide.cpp:2444).
    pub fn scale_group_speed(&mut self, k: f32) {
        self.group_speed *= k;
    }

    /// Port of `IsAutomaticMode` (MatrixRobot.hpp:380-383).
    pub fn is_automatic_mode(&self) -> bool {
        matches!(
            self.state,
            RobotState::InSpawn | RobotState::BaseMoveOut | RobotState::BaseCapture
        )
    }

    /// Port of `CanBreakOrder` (MatrixRobot.hpp:385-406). The
    /// MMFLAG_FULLAUTO demo flag and the arcaded-object (FPS mode)
    /// branch are out of scope: non-player capturing robots never
    /// break, player robots only obey the automatic-mode gate.
    pub fn can_break_order(&self) -> bool {
        if (self.side != crate::matrix_game::common::PLAYER_SIDE
            || crate::matrix_game::map::full_auto())
            && self.get_capture_factory().is_some()
        {
            return false; // DO NOT BREAK CAPTURING!!! NEVER!!! (C++ comment)
        }
        // `!IsAutomaticMode() && (m_Side!=PLAYER_SIDE || arcaded!=this)`
        // (MatrixRobot.hpp:405).
        !self.is_automatic_mode() && !self.is_arcaded
    }

    /// Port of `IsDisableManual` (MatrixRobot.hpp:357).
    /// `m_ObjectState & ROBOT_FLAG_COLLISION` — set by LowLevelMove when
    /// the robot hit something this frame.
    pub fn is_colliding(&self) -> bool {
        self.object_state & crate::matrix_game::map_static::ROBOT_FLAG_COLLISION != 0
    }

    pub fn is_disable_manual(&self) -> bool {
        (self.object_state & crate::matrix_game::map_static::ROBOT_FLAG_DISABLE_MANUAL) != 0
    }

    /// Port of `HaveBomb` (MatrixRobot.cpp:4617-4623).
    pub fn have_bomb(&self, objs: &Objects) -> bool {
        self.weapons.iter().any(|bw| {
            objs.weapons
                .get(bw.weapon)
                .map(|we| we.weapon_type() == WEAPON_BIGBOOM)
                .unwrap_or(false)
        })
    }

    /// Active robot carries a repair weapon — gates the `orep` HUD
    /// button (`FindWeapon(WEAPON_REPAIR)`, CInterface.cpp:1304).
    pub fn have_repair(&self, objs: &Objects) -> bool {
        self.weapons.iter().any(|bw| {
            objs.weapons
                .get(bw.weapon)
                .map(|we| we.weapon_type() == WEAPON_REPAIR)
                .unwrap_or(false)
        })
    }

    pub fn get_capture_factory(&self) -> Option<ObjectId> {
        self.orders
            .iter()
            .find(|o| o.ty == OrderType::CaptureFactory)
            .and_then(|o| o.target)
    }

    /// Port of `CMatrixRobotAI::AddCaptureCandidate` (MatrixRobot.cpp:
    /// 4893-4911).
    pub fn add_capture_candidate(&mut self, b: ObjectId, rng: &mut Rnd) {
        const MAX_CAPTURE_CANDIDATES: usize = 4; // MatrixRobot.hpp:48
        if self.orders.has(OrderType::CaptureFactory) {
            return;
        }
        self.object_state |= crate::matrix_game::map_static::ROBOT_CAPTURE_INFORMED;
        if self.capture_candidates.len() >= MAX_CAPTURE_CANDIDATES {
            return;
        }
        if self.capture_candidates.iter().any(|(id, _)| *id == b) {
            return;
        }
        self.capture_candidates.push((b, rng.range(100, 5000)));
    }

    /// Port of `CMatrixRobotAI::RemoveCaptureCandidate` (MatrixRobot.cpp:
    /// 4913-4925) — swap-remove like the C++.
    pub fn remove_capture_candidate(&mut self, b: ObjectId) {
        if let Some(i) = self.capture_candidates.iter().position(|(id, _)| *id == b) {
            self.capture_candidates.swap_remove(i);
        }
    }

    /// Port of `CMatrixRobotAI::ClearCaptureCandidates` (MatrixRobot.cpp:
    /// 4883-4891).
    pub fn clear_capture_candidates(&mut self) {
        self.capture_candidates.clear();
    }

    /// Port of `CMatrixRobotAI::TaktCaptureCandidate` (MatrixRobot.cpp:
    /// 4836-4881): tick down each candidate's fuse; once expired, pick
    /// the best target (fewest turrets; neutral breaks ties) and issue
    /// the CAPTURE_FACTORY order.
    pub fn takt_capture_candidate(&mut self, ms: i32, objs: &Objects, rng: &mut Rnd) {
        use crate::matrix_game::object_building::Building;
        let _ = rng;

        let mut bc: Option<(ObjectId, i32, i32)> = None; // (id, turrets_have, side)
        let mut i = 0;
        while i < self.capture_candidates.len() {
            self.capture_candidates[i].1 -= ms;
            if self.capture_candidates[i].1 < 0 {
                let bid = self.capture_candidates[i].0;
                let info = objs.get(bid).and_then(|o| {
                    if matches!(o.core().obj_type, ObjectType::Building) {
                        let b: &Building =
                            unsafe { &*(o as *const dyn MapStatic as *const Building) };
                        Some((b.side, b.turrets_have, b.capturer.is_none()))
                    } else {
                        None
                    }
                });
                match info {
                    Some((side, turrets, free)) if side != self.side => {
                        if free {
                            bc = match bc {
                                Some((cid, cturrets, cside)) => {
                                    if cturrets > turrets {
                                        Some((bid, turrets, side))
                                    } else if cturrets == turrets && side == 0 {
                                        // Neutral buildings win ties
                                        // (MatrixRobot.cpp:4858).
                                        Some((bid, turrets, side))
                                    } else {
                                        Some((cid, cturrets, cside))
                                    }
                                }
                                None => Some((bid, turrets, side)),
                            };
                        }
                    }
                    _ => {
                        // Dead or already ours — drop the candidate
                        // (swap-remove, :4868-4870).
                        self.capture_candidates.swap_remove(i);
                        continue;
                    }
                }
            }
            i += 1;
        }

        if let Some((bid, _, _)) = bc {
            self.capture_factory(bid);
            self.clear_capture_candidates();
        }
    }

    /// Port of `CMatrixRobotAI::StopMoving` (inline helper used from
    /// MatrixRobot.cpp:1034,5246,...). Clears current dest + path,
    /// zeroes velocity. The C++ version also sets
    /// `ROBOT_FLAG_COLLISION` + rechange flags; we skip those —
    /// there's no animation layer consuming them yet.
    /// Port of `CMatrixRobotAI::GetMoveToCoords` (MatrixRobot.cpp:
    /// 5016-5027). Returns the `(mx, my)` of the first ROT_MOVE_TO
    /// order in the pool, or None.
    pub fn move_to_coords(&self) -> Option<(i32, i32)> {
        for o in self.orders.iter() {
            if o.ty == OrderType::MoveTo {
                return Some((o.p1 as i32, o.p2 as i32));
            }
        }
        None
    }

    /// Port of `CMatrixRobotAI::GetReturnCoords` (MatrixRobot.cpp
    /// counterpart). Returns the `(mx, my)` of the first
    /// ROT_MOVE_RETURN in the pool, or None.
    pub fn return_coords(&self) -> Option<(i32, i32)> {
        for o in self.orders.iter() {
            if o.ty == OrderType::MoveReturn {
                return Some((o.p1 as i32, o.p2 as i32));
            }
        }
        None
    }

    /// Port of `CMatrixRobotAI::RotateRobot(dest, &rangle)` at
    /// MatrixRobot.cpp:2302-2391. Rotates `m_Forward` by one
    /// LogicTakt slice toward `dest`, at angular speed
    /// `m_maxRotationSpeed * m_SyncMul` (radians). Also rotates
    /// `m_Velocity` to stay aligned with the new forward.
    ///
    /// Returns `(aligned, angle)` — `aligned = true` when the
    /// rotation step was large enough to complete the turn this
    /// tick (C++ returns true; Seek then takes the "already
    /// facing dest" path). `angle` is the pre-rotation angle
    /// between forward and dest, fed back into Seek's
    /// end-of-path taper.
    pub fn rotate_robot(&mut self, cms: i32, dest: glam::Vec2) -> (bool, f32) {
        let max_rot =
            crate::matrix_game::config::global().chassis.rotation_speed[self.chassis.kind_index()];
        let sync_mul = (cms as f32) / 10.0;
        let rot_speed = max_rot * sync_mul;

        let dest_dir = dest - glam::Vec2::new(self.pos_x, self.pos_y);
        let dest_dir_n = dest_dir.normalize_or_zero();
        let forward = self.forward.normalize_or_zero();

        // cos of angle between forward and dest direction.
        let cos1 = (forward.x * dest_dir_n.x + forward.y * dest_dir_n.y).clamp(-1.0, 1.0);
        let angle1 = cos1.acos();

        // `vec` = forward rotated 90° clockwise (= forward.y, -forward.x).
        // `rotDir = dest_dir_n · vec` — positive when dest is to the
        // right of forward, negative when to the left.
        let vec = glam::Vec2::new(forward.y, -forward.x).normalize_or_zero();
        let rot_dir = dest_dir_n.x * vec.x + dest_dir_n.y * vec.y;

        // MatrixRobot.cpp:2335-2339 — if we can complete the turn
        // this tick, snap and report aligned.
        if angle1 <= rot_speed {
            self.forward = dest_dir_n;
            let vel_len = self.velocity.length();
            self.velocity = self.forward * vel_len;
            return (true, angle1);
        }

        // MatrixRobot.cpp:2341-2351 — one step of the rotation.
        let step = if (angle1 - rot_speed).abs() < 0.001 {
            // Exactly one step away → snap.
            dest_dir_n
        } else if rot_dir > 0.0 {
            // Dest to the right → rotate CW (-rot_speed around Z).
            rotate_vec2(forward, -rot_speed)
        } else {
            // Dest to the left → rotate CCW.
            rotate_vec2(forward, rot_speed)
        };
        self.forward = step;
        // MatrixRobot.cpp:2375 — rotate velocity with forward.
        let vel_len = self.velocity.length();
        self.velocity = self.forward * vel_len;
        (false, angle1)
    }

    /// Port of `CMatrixRobotAI::StopMoving` (MatrixRobot.cpp:4715-
    /// 4732): converts MOVE_TO / MOVE_TO_BACK orders to STOP_MOVE in
    /// place (the STOP_MOVE dispatch case then runs `LowLevelStop` and
    /// pops it) and clears the zone / move path state. Note the C++
    /// does NOT zero velocity here — `LowLevelStop` is commented out
    /// at :4731.
    pub fn stop_moving(&mut self) {
        for o in self.orders.iter_mut() {
            if o.ty == OrderType::MoveTo || o.ty == OrderType::MoveToBack {
                if o.phase == OrderPhase::CaptureMoving
                    && std::env::var_os("MG_TRACE_CAPTURE").is_some()
                {
                    eprintln!(
                        "[capture-trace] stop_moving killed capture-MoveTo({},{}) of {:?} at ({:.0},{:.0})",
                        o.p1, o.p2, self.self_id, self.pos_x, self.pos_y
                    );
                }
                *o = Order::set(OrderType::StopMove, 0.0, 0.0, 0.0, 0);
            }
        }
        self.zone_des = -1;
        self.zone_path.clear();
        self.zone_path_next = -1;
        self.move_path.clear();
    }

    /// Port of `CMatrixRobotAI::LowLevelStop` (MatrixRobot.cpp:2657-
    /// 2662) — kill all motion immediately.
    pub fn low_level_stop(&mut self) {
        self.speed = 0.0;
        self.velocity = glam::Vec2::ZERO;
    }

    /// Port of `CMatrixRobotAI::RotateHull(dest)` (MatrixRobot.cpp:
    /// 2233-2299). Lerps `hull_forward` toward the unit direction from
    /// the robot's position to `dest` at rate `m_maxHullSpeed` (loaded
    /// from `Chars/Armor/ARMOR{N}_ROTATION_SPEED`). The C++ MAX_HULL_ANGLE
    /// = GRAD2RAD(360) clamp at lines 2288-2296 is dead code (fabs of an
    /// angle never reaches 2π) so we omit it.
    ///
    /// `cms` scales the step like `rotate_robot` — the C++ takt cadence
    /// is 10 ms, so `sync_mul = cms / 10` keeps real-time behaviour
    /// independent of the frame rate.
    pub fn rotate_hull(&mut self, dest: glam::Vec2, cms: i32) {
        let here = glam::Vec2::new(self.pos_x, self.pos_y);
        let dest_dir = dest - here;
        // Arcaded: a cursor closer than MIN_ROT_DIST keeps the current
        // hull heading so the hull doesn't spin wildly under the robot
        // (MatrixRobot.cpp:2237-2239, MIN_ROT_DIST=20).
        let dest_dir_n = if self.is_arcaded && dest_dir.length_squared() < 20.0 * 20.0 {
            self.hull_forward
        } else {
            if dest_dir.length_squared() < 1e-8 {
                return;
            }
            dest_dir.normalize()
        };

        // Hull-servo sound on turn-direction flips (MatrixRobot.cpp:
        // 2255-2276): HULL_ROT_S_ANGL=10°, HULL_ROT_TAKTS=30.
        if self.is_arcaded {
            let cos3 =
                (self.hull_forward.x * dest_dir_n.x + self.hull_forward.y * dest_dir_n.y)
                    .clamp(-1.0, 1.0);
            let angle3 = cos3.acos();
            let mut dchanged = false;
            if angle3.abs() >= (10.0_f32).to_radians() {
                let cosmos = dest_dir_n.x * (-self.hull_forward.y)
                    + dest_dir_n.y * self.hull_forward.x;
                if cosmos < 0.0 && self.prev_turn_sign == 1 {
                    self.prev_turn_sign = 0;
                    dchanged = true;
                } else if cosmos > 0.0 && self.prev_turn_sign == 0 {
                    self.prev_turn_sign = 1;
                    dchanged = true;
                }
            }
            self.hull_rot_cnt += 1;
            if dchanged && self.hull_rot_cnt >= 30 {
                self.hull_sound_pending = true;
                self.hull_rot_cnt = 0;
            }
        }

        // Per-armor max hull rotation speed
        // (`g_Config.m_ItemChars[ARMOR{N}_ROTATION_SPEED]`,
        // MatrixRobot.cpp:4200-4214).
        let armor_kind = self.config.hull.unit.kind.0;
        let armor_idx = (armor_kind - 1).max(0) as usize;
        let max_hull_speed = crate::matrix_game::config::global()
            .item_chars
            .armor_rotation_speed
            .get(armor_idx)
            .copied()
            .unwrap_or(0.0);
        if max_hull_speed <= 0.0 {
            // No armor / no rate configured — snap to chassis forward
            // so the hull at least doesn't drift away.
            self.hull_forward = dest_dir_n;
            return;
        }
        let sync_mul = (cms as f32) / 10.0;
        // Near-reversals (hull-to-target angle >= 160°) turn at 3× the
        // rate so the hull doesn't lag when the target is behind
        // (MatrixRobot.cpp:2282-2285).
        let cos3 = self.hull_forward.x * dest_dir_n.x + self.hull_forward.y * dest_dir_n.y;
        let speed = if cos3 <= (160.0_f32).to_radians().cos() {
            max_hull_speed * 3.0
        } else {
            max_hull_speed
        };
        let max_step = speed * sync_mul;

        // `Vec3Truncate(delta, max_step)` — keep delta as-is when its
        // length is below the cap, otherwise scale to `max_step`. Port
        // of MatrixRobot.cpp:2283-2285.
        let delta = dest_dir_n - self.hull_forward;
        let delta_len = delta.length();
        let step = if delta_len > max_step && delta_len > 1e-8 {
            delta * (max_step / delta_len)
        } else {
            delta
        };

        let new_hf = self.hull_forward + step;
        let l = new_hf.length();
        if l > 1e-6 {
            self.hull_forward = new_hf / l;
        }
    }

    /// Port of `CMatrixRobotAI::GetLost(v)` (MatrixRobot.cpp:5188-
    /// 5237). Issues a ROT_MOVE_TO to a random nearby cell roughly
    /// perpendicular to `v`, tagged with `ROP_GETING_LOST` so the
    /// AI knows to abandon the order when it arrives.
    pub fn get_lost(
        &mut self,
        cms: i32,
        map: &GameMap,
        objs: &Objects,
        v: glam::Vec2,
        rng: &mut Rnd,
    ) {
        // The manually-driven robot can't be knocked off course
        // (MatrixRobot.cpp:5190-5191).
        if self.is_arcaded {
            return;
        }
        // Forward more than 70° off `v` → just turn toward it this
        // takt instead of plotting a side-step (MatrixRobot.cpp:5201-5205).
        let f1 = self.forward.normalize_or_zero();
        let f2 = v.normalize_or_zero();
        let angle = (f1.x * f2.x + f1.y * f2.y).clamp(-1.0, 1.0).acos();
        if angle > 70.0_f32.to_radians() {
            self.rotate_robot(cms, glam::Vec2::new(self.pos_x, self.pos_y) + v);
            return;
        }

        // Left / right perpendiculars.
        let v_left = glam::Vec2::new(v.y, -v.x);
        let v_right = -v_left;

        // LERP between left and right by a random 0..1.
        let v_param = rnd_float01(rng);
        let lost_param = rnd_float01(rng);

        let lerp = v_left * (1.0 - v_param) + v_right * v_param;
        let lost_len = GET_LOST_MIN + (GET_LOST_MAX - GET_LOST_MIN) * lost_param;

        let lost = (lerp * lost_len) + v + glam::Vec2::new(self.pos_x, self.pos_y);
        let (mx, my) = map.world_to_move(lost.x, lost.y);
        let chassis = self.chassis.kind_index();
        // C++ passes `this` as skip and issues the MoveTo whether or
        // not the spiral found a free spot (:5228-5230).
        let (dx, dy) = logic::place_find_near(
            map,
            objs,
            chassis,
            ROBOT_MOVECELLS_PER_SIZE,
            mx,
            my,
            self.self_id,
        )
        .unwrap_or((mx, my));
        if !self
            .orders
            .has_with_phase(OrderType::MoveTo, OrderPhase::GetingLost)
        {
            self.move_to(dx, dy);
            // Tag the newly-pushed order with ROP_GETING_LOST
            // (MatrixRobot.cpp:5230-5234).
            for o in self.orders.iter_mut() {
                if o.ty == OrderType::MoveTo {
                    o.phase = OrderPhase::GetingLost;
                    break;
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Movement dispatch.

    /// Port of `CMatrixRobotAI::MapPosCalc` = `PlaceGet(kind-1,
    /// pos-20, pos-20, &map_x, &map_y)` (MatrixRobot.hpp:355,
    /// MatrixLogic.cpp:465-495): anchor at the upper-left corner of
    /// the 4×4 footprint, clamped to map bounds, and — critically —
    /// nudged off a stop-blocked cell to the first free of (x+1,y),
    /// (x,y+1), (x+1,y+1). Without the nudge, a robot hugging an
    /// obstacle feeds a blocked start cell into the pathfinder, whose
    /// first-step stop exemption then legitimises path segments
    /// through the obstacle footprint.
    pub(crate) fn map_pos_calc(&mut self, map: &GameMap) {
        let nsh = self.chassis.kind_index();
        let (mx, my) = map.world_to_move(self.pos_x, self.pos_y);
        let mut mx = (mx - ROBOT_FOOTPRINT_HALF).clamp(0, map.size_move_x as i32 - 1);
        let mut my = (my - ROBOT_FOOTPRINT_HALF).clamp(0, map.size_move_y as i32 - 1);
        let blocked =
            |x: i32, y: i32| -> Option<bool> { map.move_cell(x, y).map(|c| c.stop & (1u32 << nsh) != 0) };
        if blocked(mx, my) == Some(true) {
            if blocked(mx + 1, my) == Some(false) {
                mx += 1;
            } else if blocked(mx, my + 1) == Some(false) {
                my += 1;
            } else if blocked(mx + 1, my + 1) == Some(false) {
                mx += 1;
                my += 1;
            }
        }
        self.map_x = mx;
        self.map_y = my;
    }

    /// Port of the `ROT_MOVE_TO` dispatch at MatrixRobot.cpp:1020-
    /// 1112 — the AI/commanded path (the arcade sub-branch is handled
    /// upstream in `process_orders_list`). If there's no active
    /// move-path, computes one via A*; if there is, hands off to
    /// `move_by_move_path`.
    /// Port of `CMatrixRobotAI::ZoneCurFind` (MatrixRobot.cpp:1552-
    /// 1576) — adopt the zone of the cell we stand on when valid.
    fn zone_cur_find(&mut self, map: &GameMap) {
        if let Some(c) = map.move_cell(self.map_x, self.map_y) {
            if c.zone >= 0 {
                self.zone_cur = c.zone;
            }
        }
    }

    /// Port of `CMatrixRobotAI::ZonePathCalc` (MatrixRobot.cpp:1578-
    /// 1605). The player-group / team `m_RoadPath` route constraint
    /// lands with the player-group order layer; ungrouped robots pass
    /// no route — same as the C++ `else` branch at :1594.
    fn zone_path_calc(&mut self, map: &GameMap) {
        if self.zone_cur < 0 {
            return;
        }
        // Player-group road route, when the PG layer assigned one
        // (MatrixRobot.cpp:1587-1588).
        let route = self
            .group_road_path
            .as_deref()
            .filter(|r| r.list_cnt > 0)
            .map(|r| (r, 0usize));
        self.zone_path = logic::find_path_in_zone(
            map,
            self.chassis.kind_index(),
            self.zone_cur,
            self.zone_des,
            route,
        );
        if !self.zone_path.is_empty() {
            debug_assert!(self.zone_path.len() >= 2);
            self.zone_path_next = 1;
        } else {
            self.zone_path_next = -1;
        }
        // AI robots that can't reach their zone change team
        // (MatrixRobot.cpp:1601-1604) — flagged for the side AI takt.
        if self.side != crate::matrix_game::common::PLAYER_SIDE
            && self.side != 0
            && self.zone_cur != self.zone_des
            && self.zone_path.is_empty()
        {
            self.zone_path_fail_reteam = true;
        }
    }

    /// Port of `CMatrixRobotAI::ZoneMoveCalc` (MatrixRobot.cpp:1607-
    /// 1689): collect every other live robot's / cannon's claimed
    /// cells as path weights, run `FindLocalPath` through the active
    /// zone window, then `OptimizeMovePath` over the same weight
    /// stamps.
    fn zone_move_calc(&mut self, map: &GameMap, objs: &Objects, elapsed_ms: i64) {
        if self.zone_cur < 0 {
            return;
        }

        self.move_test_change_ms = elapsed_ms;

        let mut others: Vec<map_trace::Blocker> = Vec::new();
        for id in objs.iter_units() {
            if self.self_id == Some(id) {
                continue;
            }
            let Some(other_obj) = objs.get(id) else {
                continue;
            };
            match other_obj.core().obj_type {
                ObjectType::RobotAi => {
                    let other: &Robot =
                        unsafe { &*(other_obj as *const dyn MapStatic as *const Robot) };
                    if !other.is_live() {
                        continue;
                    }
                    // C++ :1626-1645: a moving robot blocks where it
                    // stands (current remaining waypoint, weight 30)
                    // and where it's heading (m_DesX/Y, weight 200);
                    // a standing robot only blocks its map cell.
                    if other.move_to_coords().is_some() {
                        let pos = other
                            .move_path
                            .pts
                            .get(other.move_path.cur)
                            .copied()
                            .map(Some)
                            .unwrap_or(None);
                        others.push(map_trace::Blocker {
                            pos,
                            dest: map_trace::MovePt::new(other.des_x, other.des_y),
                            size: ROBOT_MOVECELLS_PER_SIZE,
                        });
                    } else {
                        others.push(map_trace::Blocker {
                            pos: None,
                            dest: map_trace::MovePt::new(other.map_x, other.map_y),
                            size: ROBOT_MOVECELLS_PER_SIZE,
                        });
                    }
                }
                ObjectType::Cannon => {
                    // C++ :1646-1652 — every live cannon is a
                    // stationary weight-200 blocker.
                    let cannon: &Cannon =
                        unsafe { &*(other_obj as *const dyn MapStatic as *const Cannon) };
                    if matches!(
                        cannon.state,
                        CannonState::Dip | CannonState::UnderConstruction
                    ) {
                        continue; // IsLiveCannon (MatrixMapStatic.hpp:228)
                    }
                    use crate::matrix_game::common::float2int;
                    let cx = float2int(cannon.pos.x / GameMap::GLOBAL_SCALE_MOVE)
                        - ROBOT_FOOTPRINT_HALF;
                    let cy = float2int(cannon.pos.y / GameMap::GLOBAL_SCALE_MOVE)
                        - ROBOT_FOOTPRINT_HALF;
                    others.push(map_trace::Blocker {
                        pos: None,
                        dest: map_trace::MovePt::new(cx, cy),
                        size: ROBOT_MOVECELLS_PER_SIZE,
                    });
                }
                _ => {}
            }
        }

        // Zone window — `m_ZonePath + m_ZonePathNext - 1` when a zone
        // route is active, else just the current zone (:1657-1665).
        let window: Vec<i32> = if self.zone_path_next >= 0 {
            let from = (self.zone_path_next - 1).max(0) as usize;
            self.zone_path[from..].to_vec()
        } else {
            vec![self.zone_cur]
        };

        let chassis = self.chassis.kind_index();
        let zone_rect_of = |z: i32| logic::zone_move_rect(map, z);
        let result = logic::find_local_path(
            map,
            chassis,
            ROBOT_MOVECELLS_PER_SIZE,
            self.map_x,
            self.map_y,
            &window,
            &zone_rect_of,
            self.des_x,
            self.des_y,
            &others,
        );
        let mut path = result.path;
        logic::optimize_move_path(
            map,
            chassis,
            ROBOT_MOVECELLS_PER_SIZE,
            &mut path,
            &result.weight,
        );

        // :1681-1688 — reset the cursor and the dist bookkeeping.
        self.move_path.total_len = map_trace::path_total_length(&path);
        self.move_path.followed_len = 0.0;
        self.move_path.pts = path;
        self.move_path.cur = 0;
    }

    /// Port of the order-dispatch loop at MatrixRobot.cpp:1011-1342:
    /// `while (cnt < m_OrdersInPool) { switch(m_OrdersList[0]) {...}
    /// ProcessOrdersList(); cnt++; }`. Each iteration dispatches the
    /// top order then rotates it to the back; `continue` paths pop
    /// the top instead. The MOVE_TO case has an arcade sub-branch for
    /// the manually-driven robot (MatrixRobot.cpp:1024-1038).
    fn process_orders_list(
        &mut self,
        cms: i32,
        map: &GameMap,
        rng: &mut Rnd,
        objs: &mut Objects,
        elapsed_ms: i64,
    ) {
        use crate::matrix_game::object_building::Building;

        let mut cnt = 0isize;
        while cnt < self.orders.len() as isize {
            let top = *self.orders.top().unwrap();
            match top.ty {
                OrderType::MoveTo | OrderType::MoveToBack => {
                    // Manual drive (MatrixRobot.cpp:1024-1038): while
                    // arcaded, W/S hold LowLevelMove straight along
                    // (or against) m_Forward, bypassing pathfinding;
                    // neither key held → StopMoving.
                    if self.is_arcaded {
                        let inp = objs.arcade_input;
                        let here = glam::Vec2::new(self.pos_x, self.pos_y);
                        if inp.forward {
                            let dest = here + self.forward * self.max_speed();
                            self.low_level_move(cms, map, objs, dest, true, true, false, false);
                        } else if inp.backward {
                            let dest = here - self.forward * self.max_speed();
                            self.low_level_move(cms, map, objs, dest, true, true, false, true);
                        } else {
                            self.stop_moving();
                        }
                        self.map_pos_calc(map);
                        self.orders.rotate_top_to_back();
                        cnt += 1;
                        continue;
                    }
                    // :1041-1048 — frozen while a base-capture order
                    // is IN_POSITION (the base platform is carrying
                    // us).
                    let mut frozen = false;
                    for o in self.orders.iter() {
                        if o.ty == OrderType::CaptureFactory
                            && o.phase == OrderPhase::CaptureInPosition
                        {
                            if let Some(fid) = o.target {
                                if let Some(b) = objs.get(fid) {
                                    if matches!(b.core().obj_type, ObjectType::Building) {
                                        let b: &Building = unsafe {
                                            &*(b as *const dyn MapStatic as *const Building)
                                        };
                                        if b.is_base() {
                                            frozen = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !frozen {
                        self.dispatch_move_to(cms, map, rng, objs, elapsed_ms);
                    }
                }
                OrderType::MoveReturn => {
                    // :1113-1125.
                    if !self.orders.has(OrderType::MoveTo) {
                        let (mx, my) = (top.p1 as i32, top.p2 as i32);
                        if logic::place_is_empty(
                            map,
                            objs,
                            self.chassis.kind_index(),
                            ROBOT_MOVECELLS_PER_SIZE,
                            mx,
                            my,
                            self.self_id,
                        ) {
                            self.orders.pop_top();
                            self.move_to(mx, my);
                            continue;
                        }
                    }
                }
                OrderType::StopMove => {
                    // :1126-1141.
                    self.low_level_stop();
                    self.orders.pop_top();
                    self.object_state |= ROBOT_FLAG_COLLISION;
                    self.rchange |= MR_MATRIX;
                    continue;
                }
                OrderType::Fire => {
                    // :1143-1195 — aim at the order params and toggle
                    // weapon groups by fire type.
                    self.dispatch_fire_order(objs);
                }
                OrderType::StopFire => {
                    // :1197-1201.
                    self.low_level_stop_fire(objs);
                    self.orders.pop_top();
                    continue;
                }
                OrderType::CaptureFactory => {
                    // :1203-1329.
                    let was_empty_phase = top.phase == OrderPhase::Empty;
                    match self.dispatch_capture_factory(cms, map, rng, objs, top) {
                        // CAPTURING completed on a base — the robot is
                        // consumed (StaticDelete at :1317).
                        CaptureOutcome::RobotConsumed => return,
                        // Non-base capture done: `RemoveOrderFromTop();
                        // continue;` at :1323-1324 — no rotation, no
                        // cnt++, next order dispatches this takt.
                        CaptureOutcome::OrderPopped => continue,
                        CaptureOutcome::Normal => {}
                    }
                    // `cnt--` at :1238 — the Empty-phase setup pushed a
                    // MOVE_TO order; the compensation lets it dispatch
                    // within this same takt.
                    if was_empty_phase
                        && self
                            .orders
                            .top()
                            .map(|o| o.ty == OrderType::MoveTo)
                            .unwrap_or(false)
                    {
                        cnt -= 1;
                    }
                }
                OrderType::StopCapture => {
                    // :1331-1337.
                    self.orders.pop_top();
                    continue;
                }
                OrderType::Empty => {}
            }
            self.orders.rotate_top_to_back();
            cnt += 1;
        }
    }

    /// Port of the ROT_CAPTURE_FACTORY dispatch case
    /// (MatrixRobot.cpp:1203-1329).
    fn dispatch_capture_factory(
        &mut self,
        cms: i32,
        map: &GameMap,
        _rng: &mut Rnd,
        objs: &mut Objects,
        top: Order,
    ) -> CaptureOutcome {
        use crate::matrix_game::object_building::{BaseState, Building, CaptureStatus};
        const BASE_DIST: f32 = 70.0;

        let Some(fid) = top.target else { return CaptureOutcome::Normal };
        // Validate the factory still exists.
        let Some(obj) = objs.get(fid) else {
            return CaptureOutcome::Normal;
        };
        if !matches!(obj.core().obj_type, ObjectType::Building) {
            return CaptureOutcome::Normal;
        }
        let (f_pos, f_is_base, f_state, f_fwd) = {
            let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
            (
                b.pos,
                b.is_base(),
                b.state,
                // `factory->GetMatrix()._21/_22` — the building's
                // local Y axis in world space (forward).
                glam::Vec2::new(b.core().matrix.y_axis.x, b.core().matrix.y_axis.y),
            )
        };

        // `factory->SetCapturedBy(this)` (:1209).
        if let Some(bm) = objs.get_mut(fid) {
            let b: &mut Building = unsafe { &mut *(bm as *mut dyn MapStatic as *mut Building) };
            b.set_captured_by(self.self_id);
        }

        let phase = top.phase;
        match phase {
            OrderPhase::Empty => {
                // :1213-1238.
                if (self.pos_x - f_pos.x).abs() < 2.0 && (self.pos_y - f_pos.y).abs() < 2.0 {
                    if let Some(o) = self.orders.top_mut() {
                        o.set_phase(OrderPhase::CaptureSettingUp);
                    }
                    return CaptureOutcome::Normal;
                }
                if let Some(o) = self.orders.top_mut() {
                    o.set_phase(OrderPhase::CaptureMoving);
                }
                use crate::matrix_game::common::trunc_float;
                let (mut mx, mut my);
                if f_is_base {
                    let tgt = f_pos + f_fwd * 100.0;
                    mx = trunc_float(tgt.x / GameMap::GLOBAL_SCALE_MOVE) as i32
                        - ROBOT_MOVECELLS_PER_SIZE / 2;
                    my = trunc_float(tgt.y / GameMap::GLOBAL_SCALE_MOVE) as i32
                        - ROBOT_MOVECELLS_PER_SIZE / 2;
                } else {
                    mx = trunc_float(f_pos.x / GameMap::GLOBAL_SCALE_MOVE) as i32
                        - ROBOT_MOVECELLS_PER_SIZE / 2;
                    my = trunc_float(f_pos.y / GameMap::GLOBAL_SCALE_MOVE) as i32
                        - ROBOT_MOVECELLS_PER_SIZE / 2;
                }
                if let Some((nx, ny)) = logic::place_find_near(
                    map,
                    objs,
                    self.chassis.kind_index(),
                    ROBOT_MOVECELLS_PER_SIZE,
                    mx,
                    my,
                    self.self_id,
                ) {
                    mx = nx;
                    my = ny;
                }
                // `MoveTo` pushes a MOVE_TO on top — the capture
                // order slides down one slot, exactly like the C++
                // pool insert. The C++ `cnt--` compensates the extra
                // rotation; our rotation loop reads len() live so the
                // net behaviour matches.
                self.move_to(mx, my);
                if let Some(o) = self.orders.iter_mut().find(|o| o.ty == OrderType::MoveTo) {
                    // The capture-MoveTo is marked with the capture
                    // phase so StopCapture can find it (:4829).
                    o.set_phase(OrderPhase::CaptureMoving);
                }
            }
            OrderPhase::CaptureMoving => {
                // :1239-1251.
                let dist = glam::Vec2::new(f_pos.x - self.pos_x, f_pos.y - self.pos_y);
                let limit = if f_is_base {
                    (BASE_DIST + 60.0) * (BASE_DIST + 60.0)
                } else {
                    BASE_DIST * BASE_DIST
                };
                if !self
                    .orders
                    .has_with_phase(OrderType::MoveTo, OrderPhase::CaptureMoving)
                    && dist.length_squared() <= limit
                {
                    if let Some(o) = self.orders.top_mut() {
                        o.set_phase(OrderPhase::CaptureInPosition);
                    }
                } else if !self.orders.has(OrderType::MoveTo)
                    && !self.orders.has(OrderType::MoveReturn)
                    && dist.length_squared() > limit
                {
                    // Deviation from the C++ (which strands the order
                    // forever here): collisions/step-asides can destroy
                    // the approach MoveTo; once both it and any return
                    // path are gone while still out of range, re-plot
                    // via the Empty phase. The AI issues each capture
                    // exactly once and the claim blocks reassignment,
                    // so a stranded capturer deadlocks its whole side.
                    if let Some(o) = self.orders.top_mut() {
                        o.set_phase(OrderPhase::Empty);
                    }
                }
            }
            OrderPhase::CaptureInPosition => {
                // :1252-1278.
                if f_is_base {
                    // C++ SetBase(factory) runs every tick of this phase,
                    // resetting m_TimeWithBase — without it the 10s
                    // watchdog kills the capturer before the descent.
                    self.set_base(fid);
                    if f_state != BaseState::Opened && f_state != BaseState::Opening {
                        if let Some(bm) = objs.get_mut(fid) {
                            let b: &mut Building =
                                unsafe { &mut *(bm as *mut dyn MapStatic as *mut Building) };
                            b.open();
                        }
                        return CaptureOutcome::Normal;
                    } else if f_state != BaseState::Opened {
                        return CaptureOutcome::Normal;
                    }
                    self.object_state |= ROBOT_FLAG_DISABLE_MANUAL;
                    self.low_level_move(
                        cms,
                        map,
                        objs,
                        glam::Vec2::new(f_pos.x, f_pos.y),
                        false,
                        false,
                        false,
                        false,
                    );
                } else {
                    self.low_level_move(
                        cms,
                        map,
                        objs,
                        glam::Vec2::new(f_pos.x, f_pos.y),
                        true,
                        true,
                        false,
                        false,
                    );
                }
                if (self.pos_x - f_pos.x).abs() < 2.0 && (self.pos_y - f_pos.y).abs() < 2.0 {
                    if let Some(o) = self.orders.top_mut() {
                        o.set_phase(OrderPhase::CaptureSettingUp);
                    }
                }
            }
            OrderPhase::CaptureSettingUp => {
                // :1279-1285 — face the building's forward axis, hull
                // follows chassis.
                let dest = glam::Vec2::new(self.pos_x + f_fwd.x, self.pos_y + f_fwd.y);
                let (aligned, _) = self.rotate_robot(cms, dest);
                if aligned {
                    if let Some(o) = self.orders.top_mut() {
                        o.set_phase(OrderPhase::Capturing);
                    }
                    self.low_level_stop();
                }
                let hull = glam::Vec2::new(self.pos_x + self.forward.x, self.pos_y + self.forward.y);
                self.rotate_hull(hull, cms);
            }
            OrderPhase::Capturing => {
                // :1286-1327.
                // :1287-1294 — if the player has this building selected,
                // force-deselect (Select(NOTHING) + PLDropAllActions).
                // The player side isn't reachable here; the app loop
                // drains this queue each frame.
                objs.pending_capture_deselect.push(fid);
                if f_is_base {
                    if f_state != BaseState::Closed && f_state != BaseState::Closing {
                        self.state = RobotState::BaseCapture;
                        if let Some(bm) = objs.get_mut(fid) {
                            let b: &mut Building =
                                unsafe { &mut *(bm as *mut dyn MapStatic as *mut Building) };
                            b.close();
                        }
                    } else if f_state == BaseState::Closed {
                        // S_ENEMY_BASE_CAPTURED / S_PLAYER_BASE_CAPTURED
                        // (MatrixRobotAI.cpp:1354-1366).
                        let mut old_side = 0;
                        if let Some(bm) = objs.get_mut(fid) {
                            let b: &mut Building =
                                unsafe { &mut *(bm as *mut dyn MapStatic as *mut Building) };
                            old_side = b.side;
                            b.side = self.side;
                            b.build_stack.clear();
                        }
                        let player = crate::matrix_game::common::PLAYER_SIDE;
                        if self.side == player {
                            objs.queue_snd("s_eb_cap");
                        } else if old_side == player {
                            objs.queue_snd("s_pb_cap");
                        }
                        self.orders.pop_top();
                        self.consumed_by_capture = true;
                        if let Some(me) = self.self_id {
                            objs.remove_deferred(me);
                        }
                        return CaptureOutcome::RobotConsumed;
                    }
                } else {
                    let status = self.building_capture_step(fid, objs);
                    if status == CaptureStatus::Done {
                        self.orders.pop_top();
                        return CaptureOutcome::OrderPopped;
                    }
                }
            }
            _ => {}
        }
        CaptureOutcome::Normal
    }

    /// One `factory->Capture(this)` call (:1321) — precomputes the
    /// nearest-robot check the C++ does inside
    /// `CMatrixBuilding::Capture` via `FindObjects`, then runs the
    /// building's paint/erase state machine.
    fn building_capture_step(
        &mut self,
        fid: ObjectId,
        objs: &mut Objects,
    ) -> crate::matrix_game::object_building::CaptureStatus {
        use crate::matrix_game::object_building::{Building, CaptureStatus};

        let Some(obj) = objs.get(fid) else {
            return CaptureStatus::TooFar;
        };
        if !matches!(obj.core().obj_type, ObjectType::Building) {
            return CaptureStatus::TooFar;
        }
        let f_pos = {
            let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
            b.pos
        };

        // `FindRobotForCapture` (MatrixObjectBuilding.cpp:1125-1146):
        // capture proceeds only when *we* are the nearest robot within
        // CAPTURE_RADIUS of the building.
        const CAPTURE_RADIUS: f32 = 50.0; // MatrixObjectBuilding.hpp:27
        let mut nearest: Option<(ObjectId, f32)> = None;
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
            let d2 = (gc.x - f_pos.x).powi(2) + (gc.y - f_pos.y).powi(2);
            if d2 < best {
                best = d2;
                nearest = Some((id, d2));
            }
        }
        // The robot running this takt is checked out of the arena
        // (take-the-box), so its own distance must join the scan — the
        // C++ `FindObjects` sees the capturer like any other robot.
        let my_d2 =
            (self.core.geo_center.x - f_pos.x).powi(2) + (self.core.geo_center.y - f_pos.y).powi(2);
        let me_nearest = my_d2 < best || nearest.map(|(id, _)| Some(id) == self.self_id).unwrap_or(false);

        let now = crate::matrix_game::map::current_elapsed_ms();
        let Some(bm) = objs.get_mut(fid) else {
            return CaptureStatus::TooFar;
        };
        let b: &mut Building = unsafe { &mut *(bm as *mut dyn MapStatic as *mut Building) };
        let (status, snd) = b.capture(self.side, me_nearest, now);
        if let Some(snd) = snd {
            objs.queue_snd(snd);
        }
        status
    }

    /// Port of the ROT_MOVE_TO / ROT_MOVE_TO_BACK dispatch case
    /// (MatrixRobot.cpp:1020-1112) — the pathfinding (non-arcade)
    /// path; the manual-drive branch is handled in `process_orders_list`.
    fn dispatch_move_to(
        &mut self,
        cms: i32,
        map: &GameMap,
        rng: &mut Rnd,
        objs: &mut Objects,
        elapsed_ms: i64,
    ) {
        self.map_pos_calc(map);
        self.zone_cur_find(map);

        if self.zone_cur < 0 {
            // :1061-1066 — no zone under us: keep walking whatever
            // path remains.
            if !self.move_path.pts.is_empty() {
                self.move_by_move_path(cms, map, rng, objs, elapsed_ms);
            }
            return;
        }

        // :1075-1092 — once 70% of the local path is walked, resync
        // the zone cursor to where we actually are.
        if self.zone_path_next >= 0 && (self.zone_path_next as usize) < self.zone_path.len() {
            if self.move_path.followed_len > self.move_path.total_len * 0.7 {
                let mut i = self.zone_path_next as usize;
                while i < self.zone_path.len() {
                    if self.zone_path[i] == self.zone_cur {
                        self.zone_path_next = (i + 1) as i32;
                        self.move_path.cur = 0;
                        self.move_path.pts.clear();
                        break;
                    }
                    i += 1;
                }
                if i >= self.zone_path.len() {
                    self.zone_path.clear();
                    self.zone_path_next = -1;
                    self.move_path.cur = 0;
                    self.move_path.pts.clear();
                }
            }
        }

        if !self.move_path.pts.is_empty() {
            self.move_by_move_path(cms, map, rng, objs, elapsed_ms);
            return;
        }

        // :1098-1101 — resolve the destination zone lazily.
        if self.zone_des < 0 {
            self.zone_des =
                logic::zone_find_near(map, self.chassis.kind_index() as i32, self.des_x, self.des_y);
            self.zone_path.clear();
            self.zone_path_next = -1;
        }

        if self.zone_path.is_empty() {
            self.zone_path_calc(map);
        }

        self.zone_move_calc(map, objs, elapsed_ms);
        if self.move_path.pts.is_empty() {
            self.stop_moving();
        }
    }

    /// Port of `CMatrixRobotAI::MoveByMovePath(ms)`
    /// (MatrixRobot.cpp:1708-1764). One LogicTakt slice of a cell-level
    /// waypoint follow: pick the current `(sou, des)` segment, fire
    /// `LowLevelMove` which folds Seek + collision, then advance `cur`
    /// when the projected progress along the segment exceeds the
    /// segment length (stop on the final waypoint). The stuck-watchdog
    /// + terrain-Z + ONWATER flag update all live here too.
    fn move_by_move_path(
        &mut self,
        cms: i32,
        map: &GameMap,
        rng: &mut Rnd,
        objs: &mut Objects,
        elapsed_ms: i64,
    ) {
        let Some((sou_pt, des_pt)) = self.move_path.current_segment() else {
            self.stop_moving();
            return;
        };
        let (sou_x, sou_y) = map_trace::waypoint_to_world(sou_pt);
        let (des_x, des_y) = map_trace::waypoint_to_world(des_pt);
        let dest = glam::Vec2::new(des_x, des_y);

        // `globalend` — the next waypoint IS the order's destination
        // cell (MatrixRobot.cpp:1721). `end_path` additionally
        // requires it to be the final segment of this local path.
        let globalend = self.des_x == des_pt.x && self.des_y == des_pt.y;
        let last_seg = self.move_path.cur + 1 == self.move_path.pts.len() - 1;
        let end_path = globalend && last_seg;

        // C++: LowLevelMove(ms, dest, robot_coll=true, obst_coll=true,
        //                    end_path=globalend && last, back=false).
        // The wrapper runs Seek (sets m_Velocity), then sphere/AABB +
        // robot-robot corrections, then integrates `pos += velocity *
        // m_SyncMul + result_coll`.
        self.low_level_move(cms, map, objs, dest, true, true, end_path, false);

        // Port of the segment-advance test at MatrixRobot.cpp:1725-1750.
        // `vMeProj` is the projection of our offset onto the segment;
        // the C++ compares squared lengths (sign-blind).
        let seg = glam::Vec2::new(des_x - sou_x, des_y - sou_y);
        let seg_len_sq = seg.length_squared().max(1.0e-6);
        let me = glam::Vec2::new(self.pos_x - sou_x, self.pos_y - sou_y);
        let dot = me.dot(seg);
        let proj_len_sq = dot * dot / seg_len_sq;
        let reached = if globalend {
            (self.pos_x - des_x).powi(2) + (self.pos_y - des_y).powi(2) < 0.2
        } else {
            proj_len_sq >= seg_len_sq
        };
        if reached {
            self.move_path.cur += 1;
            if self.move_path.cur + 1 >= self.move_path.pts.len() {
                // :1739-1748 — chain into the next zone segment, or
                // finish the order.
                self.zone_path_next += 1;
                if self.zone_path_next >= 0
                    && (self.zone_path_next as usize) < self.zone_path.len()
                {
                    self.move_path.cur = 0;
                    self.move_path.pts.clear();
                } else {
                    self.pos_x = des_x;
                    self.pos_y = des_y;
                    self.stop_moving();
                }
            }
        }

        // Stuck-watchdog — MatrixRobot.cpp:1753-1761: wipe both the
        // zone route and the local path so the next takt recomputes.
        let dpos = glam::Vec2::new(self.pos_x, self.pos_y) - self.move_test_pos;
        if dpos.length_squared() > 25.0 {
            self.move_test_pos = glam::Vec2::new(self.pos_x, self.pos_y);
            self.move_test_change_ms = elapsed_ms;
        } else if elapsed_ms - self.move_test_change_ms > 2000 {
            self.zone_path.clear();
            self.zone_path_next = -1;
            self.move_path.clear();
            // Still pinned 6 s in: the recomputed path keeps jamming
            // into the same same-side pile in perfect force
            // equilibrium (desired velocities cancelled by collisions,
            // step-aside gated off because the blocker "moves").
            // Sidestep out via the C++'s own GetLost — the escape its
            // authors left commented in the collision callback
            // (MatrixRobot.cpp:2988). The AI / player-group layers
            // re-issue the real destination on their next takt.
            // Rewind the timer by the 2 s wipe threshold so the
            // per-takt wipe cadence stays exactly the C++'s.
            if elapsed_ms - self.move_test_change_ms > 6000 {
                self.move_test_change_ms = elapsed_ms - 2000;
                // Preserve the live destination across the sidestep,
                // like the C++ step-aside does (MatrixRobot.cpp:
                // 2969-2972) — the GetLost MoveTo replaces the current
                // one, and a capture-move destination is issued exactly
                // once by the AI, so losing it deadlocks the whole
                // side. A stale MoveReturn from before this MoveTo
                // must be refreshed, not kept: it would march the
                // robot back to its old spot instead.
                let tp = self.move_to_coords().unwrap_or((self.map_x, self.map_y));
                if let Some(o) = self
                    .orders
                    .iter_mut()
                    .find(|o| o.ty == OrderType::MoveReturn)
                {
                    o.p1 = tp.0 as f32;
                    o.p2 = tp.1 as f32;
                } else {
                    self.move_return(tp.0, tp.1);
                }
                let fwd = self.forward;
                self.get_lost(cms, map, &*objs, fwd, rng);
            }
        }

        // Port of `Z_From_Pos` (MatrixObjectRobot.cpp:277-296): sample
        // terrain height at the new pos, flip `ROBOT_FLAG_ONWATER` based
        // on `roboz < WATER_LEVEL`, and float hover/anti-grav chassis
        // back up to the water plane.
        self.pos_z = self.z_from_pos(map);
        self.core.geo_center.x = self.pos_x;
        self.core.geo_center.y = self.pos_y;
        self.core.geo_center.z = self.pos_z + 9.0;
        self.rchange |= MR_MATRIX;

        self.stamp_sole(map, objs);
    }

    /// Track / wheel tread decals (`soles`, MatrixRobot.cpp:732-772):
    /// stamp a directional spot every ~10 units of travel on land.
    /// Pneumatic footprints come from foot-linking, which isn't ported.
    fn stamp_sole(&mut self, map: &GameMap, objs: &mut Objects) {
        use crate::matrix_game::common::{trunc_float, CELLFLAG_BRIDGE, CELLFLAG_LAND};
        use crate::matrix_game::effects::landscape_spot::{SpotKind, SpotSpawn};
        if self.state == RobotState::InSpawn {
            return;
        }
        let (kind, scale) = match self.chassis {
            crate::matrix_game::robot::ChassisKind::Track => (SpotKind::SoleTrack, 2.0),
            crate::matrix_game::robot::ChassisKind::Wheel => (SpotKind::SoleWheel, 3.0),
            _ => return,
        };
        let gc = glam::Vec2::new(self.pos_x, self.pos_y);
        if (self.last_sole_pos - gc).length_squared() <= 100.0 {
            return;
        }
        self.last_sole_pos = gc;
        let x = trunc_float(self.pos_x / crate::matrix_game::map::GLOBAL_SCALE) as i32;
        let y = trunc_float(self.pos_y / crate::matrix_game::map::GLOBAL_SCALE) as i32;
        let Some(flags) = map.unit_flags(x, y) else {
            return;
        };
        if flags & CELLFLAG_LAND == 0 || flags & CELLFLAG_BRIDGE != 0 {
            return;
        }
        let angle = (-self.forward.x).atan2(self.forward.y);
        objs.pending_spots.push(SpotSpawn {
            pos: gc,
            angle,
            scale,
            kind,
        });
    }

    /// Port of `CMatrixRobotAI::LowLevelMove` (MatrixRobot.cpp:2563-2653).
    /// Direct translation of the composition:
    ///   1. `Seek(dest, rotate, end_path, back)` — produces `m_Velocity`.
    ///   2. `genetic_mutated_velocity = m_Velocity * m_SyncMul` (the
    ///      full tick's worth of linear motion).
    ///   3. If `robot_coll`: `r = RobotToObjectCollision(gmv, ms)`.
    ///   4. If `obst_coll`: `o = SphereRobotToAABBObstacleCollision(r, gmv)`
    ///      followed by `WallAvoid(o, dest)` only when `m_Cols==0`.
    ///   5. `result_coll = r + o` and `pos += gmv + result_coll`.
    ///
    /// `Decelerate` / `LowLevelDecelerate` (:2655) isn't wired in yet —
    /// the MOVE_TO path never calls it; `StopMoving` just zeroes
    /// `move_path` directly.
    #[allow(clippy::too_many_arguments)]
    fn low_level_move(
        &mut self,
        cms: i32,
        map: &GameMap,
        objs: &mut Objects,
        dest: glam::Vec2,
        robot_coll: bool,
        obst_coll: bool,
        end_path: bool,
        back: bool,
    ) {
        let rotate = self.seek(cms, map, dest, end_path, back);
        self.cols = 0;

        let sync_mul = (cms as f32) / 10.0;
        let gmv = self.velocity * sync_mul;

        let mut r = glam::Vec2::ZERO;
        let mut result_coll = glam::Vec2::ZERO;

        if robot_coll {
            r = self.robot_to_object_collision(cms, map, objs, gmv);
            result_coll = r;
        }
        if obst_coll {
            let o = sphere_robot_to_aabb_obstacle_collision(
                map,
                self.chassis.kind_index(),
                self.pos_x,
                self.pos_y,
                r,
                gmv,
            );
            if self.cols == 0 {
                self.wall_avoid(dest, o);
            }
            result_coll = r + o;
        }

        // MatrixRobot.cpp:2614 — accumulate world-space distance actually
        // travelled into the `m_MovePath`'s `m_MovePathDistFollow`, used
        // by the AI's revert-to-old-path heuristic.
        let disp = gmv + result_coll;
        self.move_path.followed_len += disp.length();

        self.pos_x += disp.x;
        self.pos_y += disp.y;
        self.rchange |= MR_MATRIX;

        // Pneumatic collision flag (MatrixRobot.cpp:2621-2632): while the
        // walker is rotating, being pushed by a collision, or arriving,
        // suppress the foot-linking body-correction; the instant that
        // clears, re-anchor `link_pos` (FirstLinkPneumatic) so the
        // correction can't drag the body back into the obstacle. Without
        // this, dropped robots get pinned inside the base building.
        if matches!(self.chassis, ChassisKind::Pneumatic) {
            let gsm = GameMap::GLOBAL_SCALE_MOVE;
            let near_dest = end_path
                && ((dest.x - self.pos_x).powi(2) + (dest.y - self.pos_y).powi(2)) < gsm * gsm;
            if rotate || result_coll != glam::Vec2::ZERO || self.cols_weight2 != 0 || near_dest {
                self.object_state |= crate::matrix_game::map_static::ROBOT_FLAG_COLLISION;
            } else if self.is_colliding() {
                self.object_state &= !crate::matrix_game::map_static::ROBOT_FLAG_COLLISION;
                self.link_prev_frame = -1; // re-anchor == FirstLinkPneumatic
            }
        }
    }

    /// Port of `CMatrixRobotAI::Seek(dest, rotate, end_path, back)`
    /// (MatrixRobot.cpp:2394-2458). Rotates the robot one-tick toward
    /// `dest` (unless in `BASE_MOVEOUT` or going `back`), applies slope
    /// + water speed corrections, and sets `m_Velocity` / `m_Speed`.
    ///
    /// Returns the `rotate` bit the C++ passes through as an out-param —
    /// true when RotateRobot did a non-completing turn this tick, which
    /// feeds the Pneumatic `ROBOT_FLAG_COLLISION` heuristic in
    /// `low_level_move`.
    fn seek(
        &mut self,
        cms: i32,
        map: &GameMap,
        dest: glam::Vec2,
        end_path: bool,
        back: bool,
    ) -> bool {
        let mut rangle = 0.0;
        let rotate;
        if !matches!(self.state, RobotState::BaseMoveOut) && !back {
            let (aligned, a) = self.rotate_robot(cms, dest);
            if aligned {
                rotate = false;
                rangle = 0.0;
            } else {
                rotate = true;
                rangle = a;
            }
        } else {
            rotate = false;
        }

        let forward = if back { -self.forward } else { self.forward };

        let dest_dir = dest - glam::Vec2::new(self.pos_x, self.pos_y);
        let dest_length = dest_dir.length();

        // MatrixRobot.cpp:2418 — `m_GroupSpeed` 0-init falls back to
        // `m_maxSpeed`. A fresh robot never entered a group, so this is
        // the default path.
        let max_speed = self.max_speed();
        if self.group_speed <= 0.001 {
            self.group_speed = max_speed;
        }

        // Water + slope correction (MatrixRobot.cpp:2420-2442).
        let cfg = crate::matrix_game::config::global();
        let chassis_idx = self.chassis.kind_index();
        let mut k = if (self.object_state & crate::matrix_game::map_static::ROBOT_FLAG_ONWATER) != 0
        {
            cfg.chassis.water_corr[chassis_idx]
        } else {
            1.0
        };

        // Slope = sin(pitch) along the *travel* direction. The C++ uses
        // `dot(up, m_Core->m_Matrix._21..._23)` — the Y-axis of the
        // robot's world matrix, which is the terrain-aligned forward.
        // We reconstruct the same quantity from the terrain gradient at
        // the current pos along `forward` (already flipped when back).
        let slope = terrain_slope_along(map, self.pos_x, self.pos_y, forward);
        let slope_up = cfg.chassis.slope_corr_up[chassis_idx];
        let slope_down = cfg.chassis.slope_corr_down[chassis_idx];
        if slope >= 0.0 {
            if slope_up > 0.0 && slope >= slope_up {
                k = 0.0;
            } else if slope_up > 0.0 {
                // LERPFLOAT(slope/slope_up, 1.0, 0.0) = 1 - slope/slope_up
                k *= lerp(slope / slope_up, 1.0, 0.0);
            }
        } else {
            // LERPFLOAT(-slope, 1.0, slope_down) — with |slope|∈[0,1]
            // this scales from 1 at flat to slope_down at full descent.
            k *= lerp(-slope, 1.0, slope_down);
        }

        let min_speed = self.group_speed.min(self.col_speed);
        if dest_length - min_speed < 0.001 {
            self.velocity = dest_dir * k;
            self.speed = dest_length * k;
        } else {
            let mut speed = k * min_speed;
            if end_path {
                let mut t = (dest_length / 20.0).min(1.0);
                t *= (1.0 - rangle / std::f32::consts::PI).min(1.0);
                speed *= t;
            }
            self.velocity = forward * speed;
            self.speed = speed;
        }

        // C++ Seek's *return* (`vel`) is always true at :2458 and its
        // caller's `!vel` branch is dead; what LowLevelMove consumes is
        // the `rotate` out-param (MatrixRobot.cpp:2566) — return that.
        rotate
    }

    /// Port of `CMatrixRobotAI::RobotToObjectCollision`
    /// (MatrixRobot.cpp:3055-3074) + `CollisionCallback` (:2892-3053).
    /// Per nearby robot (within the FindObjects `3R` search):
    ///   - separates overlapping pairs along the connecting vector,
    ///   - freshly-colliding pair moving the same way → the rear robot
    ///     stops this tick (`data->stop`) and clamps `m_ColSpeed` to
    ///     half the leader's group speed (:2921-2940),
    ///   - sustained jam against a same-side robot dead ahead moving
    ///     opposite (or idle) → that robot is told to step aside:
    ///     MoveReturn + PlaceFindNearReturn + MoveTo + AddBadCoord
    ///     (:2946-2983),
    ///   - far look (2R..4R): the group-speed clamp without the stop
    ///     (:3006-3034).
    /// Returns the displacement to add to this tick's movement, or
    /// `-vel` (full stop) when the callback raised `stop` (:3072).
    fn robot_to_object_collision(
        &mut self,
        cms: i32,
        map: &GameMap,
        objs: &mut Objects,
        vel: glam::Vec2,
    ) -> glam::Vec2 {
        const COLLIDE_BOT_R: f32 = 18.0;
        const COLLIDE_BOT_2R: f32 = COLLIDE_BOT_R + COLLIDE_BOT_R;
        // `CANNON_COLLIDE_R` (MatrixObjectCannon.hpp:20).
        const CANNON_COLLIDE_R: f32 = 20.0;
        // `const int tm = 2` (:2895) — collision-weight multiplier.
        const TM: i32 = 2;

        // `my_pos + vel` — where the robot would end up this tick without
        // any collision response. The callback uses this for the distance
        // check at :2902.
        let my_pos = glam::Vec2::new(self.pos_x + vel.x, self.pos_y + vel.y);
        let my_cur = glam::Vec2::new(self.pos_x, self.pos_y);
        let self_has_return = self.return_coords().is_some();

        let mut result = glam::Vec2::ZERO;
        let mut far_col = false;
        let mut stop = false;

        let ids: Vec<ObjectId> = objs.iter_units().collect();
        for id in ids {
            let Some(other_obj) = objs.get(id) else {
                continue;
            };
            match other_obj.core().obj_type {
                ObjectType::RobotAi => {
                    // Snapshot, then drop the borrow — the weight inc /
                    // step-aside below need `objs` mutably.
                    let other: &Robot =
                        unsafe { &*(other_obj as *const dyn MapStatic as *const Robot) };
                    // :2897 — automatic-mode robots (spawn / move-out /
                    // base-capture ride) don't take part in collisions.
                    if other.is_automatic_mode() {
                        continue;
                    }
                    let their_pos = glam::Vec2::new(other.pos_x, other.pos_y);
                    // FindObjects search radius (3R around the current
                    // position, :3068) bounds the far-look branch.
                    if (their_pos - my_cur).length()
                        >= COLLIDE_BOT_R * 3.0 + other.core.radius
                    {
                        continue;
                    }
                    let other_vel = other.velocity;
                    let other_group_speed = other.group_speed;
                    let other_side = other.side;
                    let other_map = (other.map_x, other.map_y);
                    let other_kind = other.chassis.kind_index();
                    let other_bad: Vec<(i32, i32)> = other.env.bad_coords().to_vec();

                    let dv = my_pos - their_pos;
                    let dist = dv.length();

                    if dist < COLLIDE_BOT_2R {
                        'pass: {
                            // :2908 — not moving → no fancy logic.
                            let vd = vel.length_squared();
                            if vd < 1.0e-8 {
                                break 'pass;
                            }
                            let vdir = vel / vd.sqrt();

                            // :2915-2919 — both parties accumulate weight;
                            // the short-window one clamps at 500*tm.
                            self.cols_weight += cms * TM;
                            self.cols_weight2 += cms * TM;
                            if let Some(o) = crate::matrix_game::logic::robot_mut(objs, id) {
                                o.cols_weight += cms * TM;
                                o.cols_weight2 += cms * TM;
                            }
                            if self.cols_weight2 > 500 * TM {
                                self.cols_weight2 = 500 * TM;
                            }

                            // :2921-2940 — fresh collision with a robot
                            // ahead moving roughly the same way: stop and
                            // match half the leader's group speed. The id
                            // comparison is the C++ pointer tie-break so
                            // only one of the pair yields.
                            if self.cols_weight2 < 200 * TM
                                && other_vel.length_squared() > 1.0e-8
                                && vdir.dot(-dv) > 0.0
                                && vdir.dot(other_vel) > 0.0
                                && !(other_vel.dot(dv) > 0.0
                                    && self.self_id.map_or(false, |me| id < me))
                            {
                                stop = true;
                                far_col = true;
                                self.col_speed = self.group_speed.min(other_group_speed * 0.5);
                            }

                            // :2946-2983 — sustained jam: once the long
                            // window saturates, a same-side robot dead
                            // ahead that is idle (or driving into us) is
                            // asked to step aside.
                            if self.cols_weight < 500 * TM {
                                break 'pass;
                            }
                            self.cols_weight = 500 * TM;
                            if dist <= 1.0e-3 {
                                break 'pass;
                            }
                            let to_other = -dv / dist;
                            // cos(90°) = 0 — the other must be in front.
                            if vdir.dot(to_other) <= 0.0 {
                                break 'pass;
                            }
                            if other_vel.length_squared() > 0.0 {
                                let dvn = other_vel.normalize();
                                // Moving, but not against us (within 45°
                                // of head-on) → the jam resolves itself.
                                if vdir.dot(dvn) >= -((45.0f32).to_radians().cos()) {
                                    self.cols_weight = 0;
                                    break 'pass;
                                }
                            }
                            if !self_has_return && other_side == self.side {
                                // The blocker remembers where to come back
                                // to (its move-to target, else its spot)…
                                if let Some(o) = crate::matrix_game::logic::robot_mut(objs, id)
                                {
                                    let capture_dest = o
                                        .orders
                                        .iter()
                                        .find(|x| {
                                            x.ty == OrderType::MoveTo
                                                && x.phase == OrderPhase::CaptureMoving
                                        })
                                        .map(|x| (x.p1 as i32, x.p2 as i32));
                                    if let Some(tp) = capture_dest {
                                        // A capture approach is sacred: a
                                        // stale MoveReturn from before the
                                        // capture order would march the
                                        // robot back and strand it — point
                                        // the return at the factory.
                                        match o
                                            .orders
                                            .iter_mut()
                                            .find(|x| x.ty == OrderType::MoveReturn)
                                        {
                                            Some(x) => {
                                                x.p1 = tp.0 as f32;
                                                x.p2 = tp.1 as f32;
                                            }
                                            None => o.move_return(tp.0, tp.1),
                                        }
                                    } else if o.return_coords().is_none() {
                                        let tp = o
                                            .move_to_coords()
                                            .unwrap_or((o.map_x, o.map_y));
                                        o.move_return(tp.0, tp.1);
                                    }
                                }
                                // …then sidesteps to the nearest free cell
                                // not on its bad-coord list.
                                if let Some(tp) = crate::matrix_game::logic::place_find_near_return(
                                    map,
                                    objs,
                                    other_kind,
                                    4,
                                    other_map.0,
                                    other_map.1,
                                    &other_bad,
                                ) {
                                    if let Some(o) =
                                        crate::matrix_game::logic::robot_mut(objs, id)
                                    {
                                        o.move_to(tp.0, tp.1);
                                        o.env.add_bad_coord(tp);
                                    }
                                }
                            }
                            self.cols_weight = 0;
                        }
                        // :2999-3003 — separation displacement, halved
                        // (the other side pushes back symmetrically).
                        if dist > 1.0e-3 {
                            let correction = (COLLIDE_BOT_2R - dist) * 0.5;
                            result += (dv / dist) * correction;
                        }
                        self.cols += 1;
                    } else if dist < COLLIDE_BOT_R * 4.0 {
                        // :3006-3034 — far look: group-speed clamp for an
                        // aligned robot ahead, without the stop.
                        let vd = vel.length_squared();
                        if vd >= 1.0e-8 && other_vel.length_squared() > 1.0e-8 {
                            let vdir = vel / vd.sqrt();
                            if vdir.dot(-dv) > 0.0
                                && vdir.dot(other_vel) > 0.0
                                && !(other_vel.dot(dv) > 0.0
                                    && self.self_id.map_or(false, |me| id < me))
                            {
                                far_col = true;
                                self.col_speed = self.group_speed.min(other_group_speed * 0.5);
                            }
                        }
                    }
                }
                ObjectType::Cannon => {
                    // Cannon branch of CollisionCallback (MatrixRobot.cpp:
                    // 3036-3053). Correction is UNHALVED — the cannon
                    // never moves, so the robot takes the whole push-out.
                    let cannon: &Cannon =
                        unsafe { &*(other_obj as *const dyn MapStatic as *const Cannon) };
                    let dv = my_pos - cannon.pos;
                    let dist = dv.length();
                    if dist < COLLIDE_BOT_R + CANNON_COLLIDE_R && dist > 1.0e-3 {
                        let correction = COLLIDE_BOT_R + CANNON_COLLIDE_R - dist;
                        result += (dv / dist) * correction;
                        self.cols += 1;
                    }
                }
                _ => {}
            }
        }

        // MatrixRobot.cpp:3070 — no far collision → reset m_ColSpeed to
        // the cap, so Seek's `min(m_GroupSpeed, m_ColSpeed)` doesn't
        // stick at a stale clamp after the colliding robot moves away.
        if !far_col {
            self.col_speed = 100.0;
        }
        // :3072 — `data.stop` cancels this tick's movement outright.
        if stop {
            return -vel;
        }
        result
    }

    /// Port of `CMatrixRobotAI::WallAvoid` (MatrixRobot.cpp:3076-3175).
    /// **In the shipped C++ build the very first executable statement
    /// of this function's body is `return;` at :3080** — the remaining
    /// ~90 lines are dead code. So WallAvoid's only observable side
    /// effect is `m_CollAvoid = (0,0,0)` and early-exit. We replicate
    /// that exactly; porting the dead body would be a no-op.
    #[allow(unused_variables)]
    fn wall_avoid(&mut self, dest: glam::Vec2, obstacle_corr: glam::Vec2) {
        self.coll_avoid = glam::Vec2::ZERO;
    }

    /// Port of `CMatrixRobot::Z_From_Pos` (MatrixObjectRobot.cpp:277-296).
    /// Samples terrain Z, flips `ROBOT_FLAG_ONWATER` when below
    /// `WATER_LEVEL`, and floats hovercraft / anti-grav chassis back up
    /// to the water plane.
    fn z_from_pos(&mut self, map: &GameMap) -> f32 {
        use crate::matrix_game::common::WATER_LEVEL;
        use crate::matrix_game::map_static::ROBOT_FLAG_ONWATER;
        let mut z = map.get_z(self.pos_x, self.pos_y);
        if z < WATER_LEVEL {
            self.object_state |= ROBOT_FLAG_ONWATER;
            if matches!(
                self.chassis,
                ChassisKind::Hovercraft | ChassisKind::AntiGravity
            ) {
                z = WATER_LEVEL;
            } else if z < WATER_LEVEL - 100.0 {
                // Sank into deep water — drown (MustDie,
                // MatrixObjectRobot.cpp:286-290).
                self.must_die = true;
            }
        } else {
            self.object_state &= !ROBOT_FLAG_ONWATER;
        }
        z
    }
}

/// GET_LOST_MIN = COLLIDE_BOT_R * 5 = 90
/// GET_LOST_MAX = COLLIDE_BOT_R * 10 = 180
/// (MatrixRobot.hpp:39-40)
const GET_LOST_MIN: f32 = 90.0;
const GET_LOST_MAX: f32 = 180.0;

/// Per-chassis move speed from `g_Config.m_ItemChars[CHASSIS{N}_MOVE_SPEED]`.
/// Delegates to the global config loaded at startup.
fn chassis_max_speed(c: ChassisKind) -> f32 {
    crate::matrix_game::config::global().chassis.move_speed[c.kind_index()]
}

/// `CMatrixMap::RndFloat(0,1)` equivalent — returns a uniform float
/// in `[0, 1)`. Ports of specific RndFloat call sites end up here
/// until the C++ RngPool is fully ported.
fn rnd_float01(rng: &mut Rnd) -> f32 {
    (rng.next() & 0x7fff) as f32 / 32768.0
}

/// Rotate a 2D vector by `angle` radians (CCW positive — matches
/// D3D's `D3DXMatrixRotationZ`).
fn rotate_vec2(v: glam::Vec2, angle: f32) -> glam::Vec2 {
    let (s, c) = angle.sin_cos();
    glam::Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

/// Port of the `LERPFLOAT(k, c1, c2)` macro from
/// `MatrixLib/3G/include/Math3D.hpp:22` — `((k) * (c2-c1) + c1)`.
#[inline]
fn lerp(k: f32, c1: f32, c2: f32) -> f32 {
    k * (c2 - c1) + c1
}

/// Port of the slope term in `Seek` (MatrixRobot.cpp:2425-2430). The
/// C++ reads `m_Core->m_Matrix._21/_22/_23` — the Y-axis of the
/// terrain-aligned world matrix — and dots it with world-up `(0,0,1)`.
/// That Y-axis is unit length; `_23` is literally `sin(pitch)` along
/// forward.
///
/// We reconstruct the same quantity from the terrain gradient at
/// `(x, y)` along the 2D `forward` direction. Central difference
/// gives `dz/ds = tan(pitch)`; `sin = tan / sqrt(1 + tan²)`.
fn terrain_slope_along(map: &GameMap, x: f32, y: f32, forward: glam::Vec2) -> f32 {
    const EPS: f32 = 1.0;
    let z_forward = map.get_z(x + EPS * forward.x, y + EPS * forward.y);
    let z_back = map.get_z(x - EPS * forward.x, y - EPS * forward.y);
    let tan_pitch = (z_forward - z_back) / (2.0 * EPS);
    tan_pitch / (1.0 + tan_pitch * tan_pitch).sqrt()
}

/// `COLLIDE_FIELD_R` (MatrixRobot.hpp:25). Half-extent in move-cells
/// of the square grid `SphereRobotToAABBObstacleCollision` scans for
/// corner AABBs around the robot.
const COLLIDE_FIELD_R: i32 = 3;
/// `COLLIDE_BOT_R` (MatrixRobot.hpp:26). Robot collision radius in
/// world units — sphere center is the robot's `(PosX, PosY)`.
const COLLIDE_BOT_R: f32 = 18.0;
/// `COLLIDE_SPHERE_R = GLOBAL_SCALE_MOVE/2.1` (MatrixRobot.hpp:27).
/// Corner-sphere radius for the `m_Sphere` rounded cell corners.
/// `GLOBAL_SCALE_MOVE = 10.0` so this is `10/2.1 ≈ 4.7619`.
const COLLIDE_SPHERE_R: f32 = GameMap::GLOBAL_SCALE_MOVE / 2.1_f32;

/// Port of `CMatrixRobotAI::SphereToAABBCheck`
/// (MatrixRobot.cpp:3511-3537). Tests whether a circle of radius
/// `COLLIDE_BOT_R` at `p` overlaps the closed AABB `[v_min, v_max]`.
/// On overlap returns `Some((distance_sq, dsx, dsy))` where:
///   - `distance_sq` — squared distance from `p` to the nearest point
///     on the AABB (0 when `p` is inside);
///   - `dsx` / `dsy` — per-axis squared distances (used by the corner
///     picker at :3459 to decide whether to push along X or Y).
fn sphere_to_aabb_check(
    p: glam::Vec2,
    v_min: glam::Vec2,
    v_max: glam::Vec2,
) -> Option<(f32, f32, f32)> {
    let mut distance = 0.0_f32;
    let mut dsx = 0.0_f32;
    let mut dsy = 0.0_f32;
    if p.x < v_min.x {
        let dx = p.x - v_min.x;
        distance += dx * dx;
        dsx += dx * dx;
    } else if p.x > v_max.x {
        let dx = p.x - v_max.x;
        distance += dx * dx;
        dsx += dx * dx;
    }
    if p.y < v_min.y {
        let dy = p.y - v_min.y;
        distance += dy * dy;
        dsy += dy * dy;
    } else if p.y > v_max.y {
        let dy = p.y - v_max.y;
        distance += dy * dy;
        dsy += dy * dy;
    }
    if distance <= COLLIDE_BOT_R * COLLIDE_BOT_R {
        Some((distance, dsx, dsy))
    } else {
        None
    }
}

/// Port of `CMatrixRobotAI::SphereToAABB` (MatrixRobot.cpp:3178-3509).
/// Resolves a single-cell correction: given the robot at `pos` and a
/// move-cell at `(cell_x, cell_y)` with `corner` bits returned by
/// `SMatrixMapMove::GetType(chassis)`, returns the push-out vector
/// needed to separate them (or `ZERO` if there is no overlap).
///
/// `corner` is the byte the pathfinder / collision grid reads:
///   - low 4 bits: `m_Sphere` corner mask → rounded/circular corner
///   - high 4 bits: `m_Zubchik` corner mask → triangular `zubchik`
///     corner (diagonal ramp off one of the cell's diagonals)
///
/// `0xff` means the cell is passable and the caller never invokes
/// this helper. `corner == 0` means a full rectangular cell.
fn sphere_to_aabb(
    map: &GameMap,
    chassis_idx: usize,
    pos: glam::Vec2,
    cell_x: i32,
    cell_y: i32,
    corner: u8,
) -> glam::Vec2 {
    const GLOBAL_SCALE_MOVE: f32 = GameMap::GLOBAL_SCALE_MOVE;

    let lu = glam::Vec2::new(
        cell_x as f32 * GLOBAL_SCALE_MOVE,
        cell_y as f32 * GLOBAL_SCALE_MOVE,
    );
    let rd = lu + glam::Vec2::new(GLOBAL_SCALE_MOVE, GLOBAL_SCALE_MOVE);

    let Some((dcol, _dsx_init, _dsy_init)) = sphere_to_aabb_check(pos, lu, rd) else {
        return glam::Vec2::ZERO;
    };

    // --- Spherical / zubchik corner handling (cpp:3218-3345). Only
    // runs when `corner > 0` (at least one rounded/triangular corner).
    if corner > 0 {
        // Pick which quadrant of the cell the robot is in, and extract
        // the single sphere (cr) / zubchik (zb) bit the C++ tests.
        let mut cr: u8 = 0;
        let mut zb: u8 = 0;
        let half = GLOBAL_SCALE_MOVE * 0.5;
        if pos.x > lu.x + half {
            if pos.y < lu.y + half {
                if (corner & 16) != 0 {
                    zb = corner & 16;
                } else {
                    cr = corner & 1;
                }
            } else {
                if (corner & 32) != 0 {
                    zb = corner & 32;
                } else {
                    cr = corner & 2;
                }
            }
        } else {
            if pos.y < lu.y + half {
                if (corner & 128) != 0 {
                    zb = corner & 128;
                } else {
                    cr = corner & 8;
                }
            } else {
                if (corner & 64) != 0 {
                    zb = corner & 64;
                } else {
                    cr = corner & 4;
                }
            }
        }

        if zb != 0 {
            // Triangular corner — project pos onto the diagonal, push
            // out radially if inside COLLIDE_BOT_R of the projection.
            if (corner & 16) != 0 || (corner & 64) != 0 {
                // Diagonal (LU → RD): v2 = (+S, +S)
                let v = pos - lu;
                let v2 = glam::Vec2::new(GLOBAL_SCALE_MOVE, GLOBAL_SCALE_MOVE);
                let proj = vec2_projection(v2, v);
                let point = lu + proj;
                let res = pos - point;
                let res_len = res.length();
                if point.x >= lu.x
                    && point.x <= rd.x
                    && point.y >= lu.y
                    && point.y <= rd.y
                    && res_len < COLLIDE_BOT_R
                {
                    let cor = COLLIDE_BOT_R - res_len;
                    return (res / res_len.max(1.0e-6)) * cor;
                }
            } else if (corner & 32) != 0 || (corner & 128) != 0 {
                // Diagonal (RU → LD): v2 = (-S, +S)
                let v = pos - glam::Vec2::new(rd.x, lu.y);
                let v2 = glam::Vec2::new(-GLOBAL_SCALE_MOVE, GLOBAL_SCALE_MOVE);
                let proj = vec2_projection(v2, v);
                let point = glam::Vec2::new(rd.x + proj.x, lu.y + proj.y);
                let res = pos - point;
                let res_len = res.length();
                if point.x >= lu.x
                    && point.x <= rd.x
                    && point.y >= lu.y
                    && point.y <= rd.y
                    && res_len < COLLIDE_BOT_R
                {
                    let cor = COLLIDE_BOT_R - res_len;
                    return (res / res_len.max(1.0e-6)) * cor;
                }
            }
            return glam::Vec2::ZERO;
        } else if cr != 0 {
            // Rounded (spherical) corner — push out from cell center
            // by `(COLLIDE_BOT_R + COLLIDE_SPHERE_R) - dist`.
            let center = lu + glam::Vec2::new(half, half);
            let v_dist = pos - center;
            let dist = v_dist.length();
            if dist < COLLIDE_BOT_R + COLLIDE_SPHERE_R {
                let correction = COLLIDE_BOT_R + COLLIDE_SPHERE_R - dist;
                return (v_dist / dist.max(1.0e-6)) * correction;
            }
            return glam::Vec2::ZERO;
        }

        // :3343 — corner > 15 means only zubchik bits set; we handled
        // those already. Fall through to rectangular resolution when
        // only sphere bits were set and we landed in a non-spheretile
        // quadrant.
        if corner > 15 {
            return glam::Vec2::ZERO;
        }
    }

    // --- Rectangular corner handling (:3349-3505). Pick the AABB
    // corner nearest the robot and resolve along the shorter axis.
    let mut min_corner = lu;
    let mut prev_min = (pos - lu).length_squared();
    let mut angle: i32 = 8;

    let a = (pos - glam::Vec2::new(rd.x, lu.y)).length_squared();
    if a < prev_min {
        prev_min = a;
        min_corner = glam::Vec2::new(rd.x, lu.y);
        angle = 1;
    }
    let a = (pos - rd).length_squared();
    if a < prev_min {
        prev_min = a;
        min_corner = rd;
        angle = 2;
    }
    let a = (pos - glam::Vec2::new(lu.x, rd.y)).length_squared();
    if a < prev_min {
        prev_min = a;
        min_corner = glam::Vec2::new(lu.x, rd.y);
        angle = 4;
    }
    let _ = prev_min;

    // Neighbor-cell consultation: the C++ reads the two adjacent
    // cells' `GetType(chassis)` to detect "inner corner" situations
    // (both neighbors blocked → no push; one blocked → switch from
    // single-axis push to full vector push via `_GOTCHA_`).
    let neighbor_type = |nx: i32, ny: i32| -> u8 {
        if nx < 0 || ny < 0 {
            return 0xff;
        }
        if (nx as usize) >= map.size_move_x || (ny as usize) >= map.size_move_y {
            return 0xff;
        }
        map.move_cell(nx, ny)
            .map(|c| c.get_type(chassis_idx))
            .unwrap_or(0xff)
    };
    let (a1, a2) = match angle {
        1 => (
            neighbor_type(cell_x + 1, cell_y),
            neighbor_type(cell_x, cell_y - 1),
        ),
        2 => (
            neighbor_type(cell_x + 1, cell_y),
            neighbor_type(cell_x, cell_y + 1),
        ),
        4 => (
            neighbor_type(cell_x - 1, cell_y),
            neighbor_type(cell_x, cell_y + 1),
        ),
        _ => (
            neighbor_type(cell_x - 1, cell_y),
            neighbor_type(cell_x, cell_y - 1),
        ),
    };
    let blocked_both = (a1 > 15 && a1 != 255) && (a2 > 15 && a2 != 255);
    let blocked_any = (a1 > 15 && a1 != 255) || (a2 > 15 && a2 != 255);
    if blocked_both {
        return glam::Vec2::ZERO;
    }
    let gotcha = blocked_any;

    let dx = pos.x - min_corner.x;
    let dy = pos.y - min_corner.y;
    let (dx2, dy2) = (dx * dx, dy * dy);

    if dx2 > dy2 {
        // Push primarily along +X.
        let mut result = glam::Vec2::new(dx, 0.0);
        let result2 = glam::Vec2::new(dx, dy);
        if gotcha {
            let g_len = result2.length().max(1.0e-6);
            let gotcha_n = result2 / g_len;
            if (angle == 1 && gotcha_n.x > 0.0 && gotcha_n.y < 0.0)
                || (angle == 2 && gotcha_n.x > 0.0 && gotcha_n.y > 0.0)
                || (angle == 4 && gotcha_n.x < 0.0 && gotcha_n.y > 0.0)
                || (angle == 8 && gotcha_n.x < 0.0 && gotcha_n.y < 0.0)
            {
                result = result2;
            }
        }
        let r_len = result.length().max(1.0e-6);
        let mag = COLLIDE_BOT_R - dcol.sqrt();
        return (result / r_len) * mag;
    } else if dx2 < dy2 {
        // Push primarily along +Y.
        let mut result = glam::Vec2::new(0.0, dy);
        let result2 = glam::Vec2::new(dx, dy);
        if gotcha {
            let g_len = result2.length().max(1.0e-6);
            let gotcha_n = result2 / g_len;
            if (angle == 1 && gotcha_n.x > 0.0 && gotcha_n.y < 0.0)
                || (angle == 2 && gotcha_n.x > 0.0 && gotcha_n.y > 0.0)
                || (angle == 4 && gotcha_n.x < 0.0 && gotcha_n.y > 0.0)
                || (angle == 8 && gotcha_n.x < 0.0 && gotcha_n.y < 0.0)
            {
                result = result2;
            }
        }
        let r_len = result.length().max(1.0e-6);
        let mag = COLLIDE_BOT_R - dcol.sqrt();
        return (result / r_len) * mag;
    }
    // dx2 == dy2 — exact corner. C++ falls through to returning
    // `(0,0,0)` in that degenerate case.
    glam::Vec2::ZERO
}

/// Port of `Vec3Projection(v2, v)` — scalar projection of `v` onto
/// `v2`, returned as a vector in `v2`'s direction. Matches
/// `MatrixLib/3G/include/Math3D.hpp`'s helper used by `SphereToAABB`.
fn vec2_projection(v2: glam::Vec2, v: glam::Vec2) -> glam::Vec2 {
    let denom = v2.length_squared().max(1.0e-6);
    v2 * (v.dot(v2) / denom)
}

/// Port of `CMatrixRobotAI::SphereRobotToAABBObstacleCollision`
/// (MatrixRobot.cpp:2718-2879). Iteratively resolves sphere-vs-cell
/// collisions within a `2*COLLIDE_FIELD_R` square of move cells
/// centred on the robot's prospective position (`pos + corr + vel`).
/// Runs up to 4 relaxation passes; each pass re-evaluates every
/// `GetType(chassis)` corner in the window and accumulates positional
/// corrections back into `robot_pos`. Returns the total displacement
/// `robot_pos - oldpos` — i.e. exactly the correction the caller adds
/// to `pos += gmv + result_coll`.
fn sphere_robot_to_aabb_obstacle_collision(
    map: &GameMap,
    chassis_idx: usize,
    pos_x: f32,
    pos_y: f32,
    corr: glam::Vec2,
    vel: glam::Vec2,
) -> glam::Vec2 {
    const GLOBAL_SCALE_MOVE: f32 = GameMap::GLOBAL_SCALE_MOVE;
    let inv_scale_move = 1.0 / GLOBAL_SCALE_MOVE;

    let mut robot_pos = glam::Vec2::new(pos_x + corr.x + vel.x, pos_y + corr.y + vel.y);
    let oldpos = robot_pos;

    // Corner cache: 6×6 of `GetType(chassis)` bytes (COLLIDE_FIELD_R=3).
    // C++ recomputes only when the (x0,y0) anchor changes across the
    // 4 relaxation passes.
    let side = (COLLIDE_FIELD_R + COLLIDE_FIELD_R) as usize;
    let mut corners: Vec<u8> = vec![0xff; side * side];
    let mut calc_for_x = i32::MIN;
    let mut calc_for_y = i32::MIN;

    let size_move_x = map.size_move_x as i32;
    let size_move_y = map.size_move_y as i32;

    for _pass in 0..4 {
        let x0_raw = (robot_pos.x * inv_scale_move).trunc() as i32 - COLLIDE_FIELD_R;
        let y0_raw = (robot_pos.y * inv_scale_move).trunc() as i32 - COLLIDE_FIELD_R;
        let x1_raw = x0_raw + COLLIDE_FIELD_R * 2;
        let y1_raw = y0_raw + COLLIDE_FIELD_R * 2;

        // Outside map at all — mirror C++ early-exit at :2752-2755.
        if x1_raw < 0 || y1_raw < 0 {
            break;
        }
        if x0_raw > size_move_x || y0_raw > size_move_y {
            break;
        }

        let x0 = x0_raw.max(0);
        let y0 = y0_raw.max(0);
        let x1 = x1_raw.min(size_move_x);
        let y1 = y1_raw.min(size_move_y);

        // Recompute the corner cache only when the window shifted.
        if calc_for_x != x0_raw || calc_for_y != y0_raw {
            for i in corners.iter_mut() {
                *i = 0xff;
            }
            for y in y0..y1 {
                for x in x0..x1 {
                    let lx = (x - x0_raw) as usize;
                    let ly = (y - y0_raw) as usize;
                    if lx >= side || ly >= side {
                        continue;
                    }
                    let ct = map
                        .move_cell(x, y)
                        .map(|c| c.get_type(chassis_idx))
                        .unwrap_or(0xff);
                    corners[ly * side + lx] = ct;
                }
            }
            calc_for_x = x0_raw;
            calc_for_y = y0_raw;
        }

        let mut col_cnt = 0;
        for y in y0..y1 {
            for x in x0..x1 {
                let lx = (x - x0_raw) as usize;
                let ly = (y - y0_raw) as usize;
                if lx >= side || ly >= side {
                    continue;
                }
                let corner = corners[ly * side + lx];
                if corner == 0xff {
                    continue;
                }
                let col = sphere_to_aabb(map, chassis_idx, robot_pos, x, y, corner);
                if col != glam::Vec2::ZERO {
                    robot_pos += col;
                    col_cnt += 1;
                }
            }
        }
        let _ = col_cnt;
    }

    robot_pos - oldpos
}

impl MapStatic for Robot {
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

    fn r_need(&mut self, _need: u32) {
        // MR_MATRIX rebuild — the C++ assembles
        // translate(pos.x, pos.y, pos.z) * rotate(angle) * scale.
        // We stub: accept any request, clear the dirty bits so the
        // next r_need doesn't spin.
        self.rchange = 0;
    }

    fn takt(&mut self, cms: i32, rng: &mut Rnd, objs: &mut Objects) {
        // HP-bar timer decay (`RobotTakt`, MatrixRobot.cpp:313-317).
        if self.show_hitpoint_time > 0 {
            self.show_hitpoint_time = (self.show_hitpoint_time - cms).max(0);
        }

        // Collision-weight decay (MatrixRobot.cpp:375-376).
        if self.cols_weight != 0 {
            self.cols_weight = (self.cols_weight - cms).max(0);
        }
        if self.cols_weight2 != 0 {
            self.cols_weight2 = (self.cols_weight2 - cms).max(0);
        } else {
            // Collisions stopped — forget accumulated step-aside cells
            // (MatrixRobot.cpp:377).
            self.env.clear_bad_coords();
        }

        // Keel-water / hovercraft dust (CMatrixRobot::Takt,
        // MatrixObjectRobot.cpp:911-948).
        const KEELWATER_SPAWN_FACTOR: f32 = 0.01;
        const DUST_SPAWN_FACTOR: f32 = 0.007;
        if !matches!(
            self.state,
            RobotState::Carrying | RobotState::Falling | RobotState::InSpawn | RobotState::BaseCapture
        ) {
            let on_water =
                self.object_state & crate::matrix_game::map_static::ROBOT_FLAG_ONWATER != 0;
            if on_water {
                self.keelwater_count += self.speed * KEELWATER_SPAWN_FACTOR * cms as f32;
            } else if self.chassis == ChassisKind::Hovercraft {
                self.dust_count += (self.speed + 1.0) * DUST_SPAWN_FACTOR * cms as f32;
            }
            while self.keelwater_count > 1.0 {
                // CreateBillboard(.., WATER_LEVEL, 5, 40, ttl 3000,
                // keelwater texture, m_Forward).
                // The C++ uses a ground-aligned quad growing 5→40
                // along m_Forward; a forward-aligned line billboard
                // with the same texture/TTL is the closest primitive
                // the port's queue offers.
                let p0 = glam::Vec3::new(
                    self.pos_x,
                    self.pos_y,
                    crate::matrix_game::common::WATER_LEVEL,
                );
                let f3 = glam::Vec3::new(self.forward.x, self.forward.y, 0.0);
                // Flat oriented ground quad growing over 3000 ms
                // (MatrixObjectRobot.cpp:926) — not a camera-facing line.
                objs.pending_effects
                    .push(crate::matrix_game::effects::GameEffect::Keelwater(
                        crate::matrix_game::effects::billboard_fx::KeelwaterFx::new(p0, f3),
                    ));
                self.keelwater_count -= 1.0;
            }
            if self.chassis == ChassisKind::Hovercraft {
                while self.dust_count > 1.0 {
                    let spd = self.velocity
                        * (cms as f32 / crate::matrix_game::logic::LOGIC_TAKT_PERIOD_MS as f32);
                    objs.pending_effects
                        .push(crate::matrix_game::effects::GameEffect::Dust(
                            crate::matrix_game::effects::dust::Dust::new(
                                glam::Vec2::new(
                                    self.core.geo_center.x,
                                    self.core.geo_center.y,
                                ),
                                spd,
                                500.0,
                                rng,
                            ),
                        ));
                    self.dust_count -= 1.0;
                }
            }
        }
    }

    fn show_hitpoint(&mut self) {
        self.show_hitpoint_time = crate::matrix_game::common::HITPOINT_SHOW_TIME_MS;
    }

    /// `CMatrixRobot::BeforeDraw` PB block (MatrixObjectRobot.cpp:
    /// 1007-1017): anchor 20 above the robot origin, bar lifted by the
    /// radius, PB_ROBOT_WIDTH=70 centered.
    fn hitpoint_bar(
        &self,
        _map: &crate::matrix_game::map::GameMap,
    ) -> Option<crate::matrix_game::map_static::HpBar> {
        if self.show_hitpoint_time <= 0 || self.hit_point <= 0.0 || self.state == RobotState::Dip
        {
            return None;
        }
        Some(crate::matrix_game::map_static::HpBar {
            anchor: glam::Vec3::new(self.pos_x, self.pos_y, self.pos_z + 20.0),
            width: 70.0,
            fill: (self.hit_point / self.hit_point_max.max(1.0)).clamp(0.0, 1.0),
            x_off: -35.0,
            // C++ lifts by GetRadius() — the mesh-AABB half-diagonal
            // (~18 for a typical build), not our hand-tuned 6-unit
            // trace sphere.
            y_off: -18.0,
        })
    }

    /// Minimal port of `CMatrixRobotAI::LogicTakt` covering only the
    /// spawn-animation branch (MatrixRobot.cpp:774-824). Reads the
    /// parent base's `BaseState` + `base_floor_progress` to lift the
    /// robot off the platform; once the base reaches `Opened`,
    /// transitions to `BaseMoveOut`, then (after a short timer) to
    /// `Idle` + closes the base — mirroring the C++ at :785-811.
    fn logic_takt(&mut self, cms: i32, rng: &mut Rnd, objs: &mut Objects) {
        use crate::matrix_game::map::current_map;
        use crate::matrix_game::object_building::{BaseState, Building, BASE_FLOOR_Z};
        let Some(map) = current_map() else {
            // No map scope active — dispatched outside of a
            // frame-level takt. Bail so we don't null-deref.
            return;
        };
        let elapsed_ms = crate::matrix_game::map::current_elapsed_ms();

        // `this == GetArcadedObject()` — cached for the helpers that
        // don't see `objs` (rotate_hull, get_lost).
        self.is_arcaded = self.self_id.is_some() && objs.arcaded_object == self.self_id;

        // SOrder::Release side effect (MatrixRobot.hpp:219-229): any
        // dropped order whose static target is a building resets that
        // building's capturer, re-opening it for capture.
        if !self.orders.released.is_empty() {
            let released: Vec<ObjectId> = self.orders.released.drain(..).collect();
            for t in released {
                if let Some(b) = crate::matrix_game::logic::building_mut(objs, t) {
                    b.reset_captured();
                }
            }
        }

        // Minimap blink countdown (MatrixRobot.cpp:406-409).
        if self.mini_map_flash_time > 0 {
            self.mini_map_flash_time -= cms;
        }

        // ROBOT_DIP — `DIPTakt` (MatrixRobot.cpp:178-283): pieces fly
        // with gravity, bounce nothing, splash into water, damage
        // objects they strike (WEAPON_DEBRIS) and pop a small boom at
        // end of life. The robot frees itself when all pieces are gone.
        if self.state == RobotState::Dip {
            self.dip_takt(cms, map, rng, objs);
            self.dip_ttl -= cms;
            if self.dip_units.is_empty() || self.dip_ttl <= -10_000 {
                if let Some(id) = self.self_id {
                    objs.remove_deferred(id);
                }
            }
            return;
        }

        // Weapons mount on the first live takt (`RobotWeaponInit` runs
        // at construction in the C++; the arena id isn't known until
        // after spawn here).
        self.robot_weapon_init(objs);

        // Base watchdog (MatrixRobot.cpp:361-371): stuck with a base
        // >10s → close it and self-destruct.
        if self.base.is_some() {
            self.time_with_base += cms;
            if self.time_with_base > 10_000 {
                if let Some(bid) = self.base {
                    if let Some(b) = crate::matrix_game::logic::building_mut(objs, bid) {
                        b.close();
                    }
                }
                self.time_with_base = 0;
                self.must_die = true;
            }
        } else {
            self.time_with_base = 0;
        }

        // MUST_DIE flag — set by BigBoom (MatrixRobot.cpp:4774).
        if self.must_die {
            if let Some(id) = self.self_id {
                let pos = self.core.geo_center;
                if self.damage(
                    crate::matrix_game::effects::weapon::WEAPON_INSTANT_DEATH,
                    pos,
                    glam::Vec3::Z,
                    0,
                    None,
                    id,
                    objs,
                ) {
                    return;
                }
            }
        }

        // Ablaze / shorted periodic self-damage (MatrixRobot.cpp:416-507).
        if self.dot_takt(objs) {
            return;
        }

        // ROBOT_FALLING (MatrixRobot.cpp:570-607): gravity until the
        // terrain catches us, then dust + back to normal duty.
        if self.state == RobotState::Falling {
            let dtime = cms as f32 * 0.013;
            self.falling_speed += dtime;
            self.pos_z -= self.falling_speed * dtime;
            let z = map.get_z(self.pos_x, self.pos_y);
            if self.pos_z < z {
                self.pos_z = z;
                self.falling_speed = 0.0;
                self.state = RobotState::Idle;
                self.map_pos_calc(map);
                // S_ROBOT_UPAL (MatrixRobot.cpp:604).
                objs.queue_snd_at(
                    "s_upal",
                    glam::Vec3::new(self.pos_x, self.pos_y, self.pos_z),
                );
                let gc = glam::Vec2::new(self.pos_x, self.pos_y);
                for _ in 0..4 {
                    objs.pending_effects
                        .push(crate::matrix_game::effects::GameEffect::Dust(
                            crate::matrix_game::effects::dust::Dust::new(
                                gc,
                                glam::Vec2::ZERO,
                                3000.0,
                                rng,
                            ),
                        ));
                }
            }
            self.core.geo_center =
                glam::Vec3::new(self.pos_x, self.pos_y, self.pos_z + 9.0);
            return;
        }

        // ROBOT_CARRYING (MatrixRobot.cpp:612-716): spring toward the
        // flyer's hook point; spin up the elevator field.
        if self.state == RobotState::Carrying {
            self.carrying_takt(cms, map, objs);
            return;
        }

        // A shorted robot is paralyzed — no orders, movement or
        // weapons until the TTL clears (MatrixRobot.cpp:717).
        if self.object_state & OBJECT_STATE_SHORTED != 0 {
            return;
        }

        match self.state {
            RobotState::Dip | RobotState::Carrying | RobotState::Falling => unreachable!(),
            RobotState::InSpawn => {
                // Reach into the base; downcast-via-raw-pointer is
                // safe because the arena slot holds a concrete Building
                // when obj_type == Building.
                let Some(base_id) = self.base else { return };
                let Some(obj) = objs.get(base_id) else {
                    // Base despawned mid-animation — fall through to idle.
                    self.state = RobotState::Idle;
                    return;
                };
                if !matches!(obj.core().obj_type, ObjectType::Building) {
                    self.state = RobotState::Idle;
                    return;
                }
                let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
                // Port of `CMatrixBuilding::GetFloorZ` (MatrixObjectBuilding.cpp:
                // 1002-1005) + the ROBOT_IN_SPAWN assignment at
                // MatrixObjectRobot.cpp:381 (`roboz = base->GetFloorZ()`).
                // `BASE_FLOOR_Z = -63`, so the platform-top tracks the
                // opening animation naturally.
                self.pos_z = b.build_z + (1.0 - b.base_floor_progress) * BASE_FLOOR_Z - 3.0 + 2.7;

                // Update core so the selection ring / robot light
                // follow the rising robot.
                self.core.geo_center.z = self.pos_z + 9.0;
                self.rchange |= MR_MATRIX;

                if b.state == BaseState::Opened {
                    // MatrixRobot.cpp:780-783 — transition to move-out.
                    self.state = RobotState::BaseMoveOut;
                }
            }
            RobotState::BaseMoveOut => {
                // Port of MatrixRobot.cpp:785-811. The C++ calls
                // `LowLevelMove(ms, m_Forward * 100, robot_coll=true,
                //               obst_coll=false)` at :787. `obst_coll=false`
                // is critical — the robot is still standing on the base's
                // own cell which is impassable to the chassis by design;
                // the sphere/AABB gate would trap it there otherwise.
                // Seek short-circuits rotation for BASE_MOVEOUT (:2398)
                // and just sets `m_Velocity = m_Forward * m_maxSpeed`.
                const BASE_DIST: f32 = 70.0;

                // Match C++: dest = pos + m_Forward*100, so the robot
                // drives along `m_Forward` until `dist_sq >= BASE_DIST^2`.
                let dest_far = glam::Vec2::new(
                    self.pos_x + self.forward.x * 100.0,
                    self.pos_y + self.forward.y * 100.0,
                );
                self.low_level_move(cms, map, objs, dest_far, true, false, false, false);

                let Some(base_id) = self.base else {
                    self.state = RobotState::Idle;
                    return;
                };
                let Some(obj) = objs.get(base_id) else {
                    self.state = RobotState::Idle;
                    self.base = None;
                    return;
                };
                if !matches!(obj.core().obj_type, ObjectType::Building) {
                    self.state = RobotState::Idle;
                    self.base = None;
                    return;
                }
                let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
                // Port of MatrixObjectRobot.cpp:412-422 — pick Z
                // from the terrain when the robot is on a land/water
                // cell, otherwise (i.e. still on the base's own
                // cell) use the base's floor Z.
                use crate::matrix_game::common::{CELLFLAG_LAND, CELLFLAG_WATER};
                let (ux, uy) = (
                    (self.pos_x / crate::matrix_game::map::GLOBAL_SCALE).floor() as i32,
                    (self.pos_y / crate::matrix_game::map::GLOBAL_SCALE).floor() as i32,
                );
                let on_land_or_water = map
                    .unit_flags(ux, uy)
                    .map(|f| (f & (CELLFLAG_LAND | CELLFLAG_WATER)) != 0)
                    .unwrap_or(false);
                if on_land_or_water {
                    self.pos_z = map.get_z(self.pos_x, self.pos_y);
                } else {
                    self.pos_z =
                        b.build_z + (1.0 - b.base_floor_progress) * BASE_FLOOR_Z - 3.0 + 2.7;
                }

                let dx = self.pos_x - b.pos.x;
                let dy = self.pos_y - b.pos.y;
                let dist_sq = dx * dx + dy * dy;

                self.core.geo_center.x = self.pos_x;
                self.core.geo_center.y = self.pos_y;
                self.core.geo_center.z = self.pos_z + 9.0;
                self.rchange |= MR_MATRIX;

                if dist_sq >= BASE_DIST * BASE_DIST {
                    // MatrixRobot.cpp:797-811 — far enough: close
                    // the base, issue a `GetLost` order so the
                    // robot wanders away from the pad, and hand
                    // off to idle where the order dispatcher picks
                    // up from here.
                    if let Some(obj_mut) = objs.get_mut(base_id) {
                        if matches!(obj_mut.core().obj_type, ObjectType::Building) {
                            let b_mut: &mut Building =
                                unsafe { &mut *(obj_mut as *mut dyn MapStatic as *mut Building) };
                            b_mut.object_state_clear(
                                crate::matrix_game::map_static::OBJECT_STATE_BUILDING_SPAWNBOT,
                            );
                            b_mut.close();
                        }
                    }
                    // RESETFLAG(ROBOT_FLAG_DISABLE_MANUAL) at :802 —
                    // the robot becomes orderable once clear of the pad.
                    self.object_state &= !ROBOT_FLAG_DISABLE_MANUAL;
                    self.base = None;
                    self.state = RobotState::Idle;
                    self.get_lost(cms, map, &*objs, self.forward, rng);
                }
            }
            RobotState::Idle | RobotState::BaseCapture => {
                // Manual steering + fire polling for the arcaded robot
                // (MatrixRobot.cpp:958-974): A/D (or mouse-cam drag)
                // raise the one-shot ROT flags; LMB held issues a FIRE
                // order aimed at the cursor unless it's over the UI;
                // LMB released converts it to STOP_FIRE.
                if self.is_arcaded {
                    let inp = objs.arcade_input;
                    if inp.left {
                        self.object_state |= ROBOT_FLAG_ROT_LEFT;
                    }
                    if inp.right {
                        self.object_state |= ROBOT_FLAG_ROT_RIGHT;
                    }
                    if inp.fire {
                        if !self.orders.has(OrderType::Fire) && !inp.cursor_on_interface {
                            self.fire(inp.cursor_world, 0);
                        }
                    } else if self.orders.has(OrderType::Fire) {
                        self.stop_fire();
                    }
                }

                // Consume the steer flags into a chassis rotation
                // toward a point 90° left/right of the current heading
                // (MatrixRobot.cpp:977-1002).
                if self.object_state & (ROBOT_FLAG_ROT_LEFT | ROBOT_FLAG_ROT_RIGHT) != 0 {
                    let mut ang = 0.0_f32;
                    if self.object_state & ROBOT_FLAG_ROT_LEFT != 0 {
                        ang -= std::f32::consts::FRAC_PI_2;
                    }
                    if self.object_state & ROBOT_FLAG_ROT_RIGHT != 0 {
                        ang += std::f32::consts::FRAC_PI_2;
                    }
                    let dest = glam::Vec2::new(self.pos_x, self.pos_y)
                        + rotate_vec2(self.forward * self.max_speed(), ang);
                    self.rotate_robot(cms, dest);
                    self.object_state &= !(ROBOT_FLAG_ROT_LEFT | ROBOT_FLAG_ROT_RIGHT);
                }

                // Port of `TaktCaptureCandidate` gating at
                // MatrixRobot.cpp:1005-1006: auto-capture decisions
                // run for non-player sides (and for the player side
                // under MMFLAG_AUTOMATIC_MODE, which we don't model).
                if self.side != crate::matrix_game::common::PLAYER_SIDE {
                    self.takt_capture_candidate(cms, objs, rng);
                }
                self.process_orders_list(cms, map, rng, objs, elapsed_ms);
                self.core.geo_center.x = self.pos_x;
                self.core.geo_center.y = self.pos_y;
                if self.consumed_by_capture {
                    // CAPTURING consumed the robot (StaticDelete at
                    // MatrixRobot.cpp:1317) — bail before animation.
                    return;
                }
                // BASE_CAPTURE rides the closing base's floor down —
                // `roboz = mu->m_Base->GetFloorZ()` (MatrixObjectRobot
                // .cpp:364-381), same tracking as IN_SPAWN's ride up.
                if self.state == RobotState::BaseCapture {
                    // Track the base via self.base like IN_SPAWN does —
                    // the C++ reads mu->m_Base from the map cell, which
                    // is order-pool independent.
                    let base = self
                        .base
                        .and_then(|fid| objs.get(fid))
                        .filter(|o| matches!(o.core().obj_type, ObjectType::Building))
                        .map(|o| unsafe {
                            &*(o as *const dyn MapStatic
                                as *const crate::matrix_game::object_building::Building)
                        });
                    if let Some(b) = base {
                        self.pos_z =
                            b.build_z + (1.0 - b.base_floor_progress) * BASE_FLOOR_Z - 3.0 + 2.7;
                        self.core.geo_center.z = self.pos_z + 9.0;
                        self.rchange |= MR_MATRIX;
                    }
                } else {
                    // C++ RNeed's default branch runs Z_From_Pos every
                    // frame (MatrixObjectRobot.cpp:439-441), so a robot
                    // pushed onto water drowns and a hover parked over
                    // water floats. Without this, stale pos_z lets
                    // robots stand on water.
                    self.pos_z = self.z_from_pos(map);
                    self.core.geo_center.z = self.pos_z + 9.0;
                }
            }
        }

        // Port of the `do_animation` label at MatrixRobot.cpp:328-
        // 353, reached via `goto do_animation` from every LogicTakt
        // branch. Picks STAY vs MOVE/MOVE_BACK based on m_Speed +
        // presence of MOVE_TO* orders. State-specific calls (e.g.
        // SwitchAnimation(MOVE) during BASE_MOVEOUT at
        // MatrixObjectRobot.cpp:424) are handled by the per-tick
        // SwitchAnimation call below because SwitchAnimation is
        // idempotent when target == current.
        let chassis_idx = self.chassis.kind_index();
        if let Some(vo) = crate::matrix_lib::three_g::vector_object::chassis_vo(chassis_idx) {
            if self.speed.abs() <= 0.01 {
                self.switch_animation(&vo, Animation::Stay);
            } else if self.orders.has(OrderType::MoveTo) {
                self.switch_animation(&vo, Animation::Move);
            } else if self.orders.has(OrderType::MoveToBack) {
                self.switch_animation(&vo, Animation::MoveBack);
            }
            // BASE_MOVEOUT explicitly forces MOVE (MatrixObjectRobot.cpp:424).
            if matches!(self.state, RobotState::BaseMoveOut) {
                self.switch_animation(&vo, Animation::Move);
            }
            // IN_SPAWN explicitly forces STAY (MatrixObjectRobot.cpp:390).
            if matches!(self.state, RobotState::InSpawn) {
                self.switch_animation(&vo, Animation::Stay);
            }
        }

        // Fire-control + heat (MatrixRobot.cpp:1364-1542) — runs after
        // order processing, before the animation tail.
        if self.state == RobotState::Idle {
            self.weapons_logic_takt(cms, map, objs, rng);
            // Hitscan weapons fired above land synchronously in the C++
            // (`FireHandler` → `HitTo`), but our box is checked out, so
            // the dispatch queued the notices instead — apply them now.
            if let (false, Some(me)) = (objs.pending_hit_notices.is_empty(), self.self_id) {
                let mut i = 0;
                while i < objs.pending_hit_notices.len() {
                    if objs.pending_hit_notices[i].0 == me {
                        let (_, hit, pos) = objs.pending_hit_notices.swap_remove(i);
                        self.hit_to(hit, pos, objs);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Hull tracking + automatic fire cutoff — port of the
        // env-target block at MatrixRobot.cpp:853-951. MAX_HULL_ANGLE
        // is GRAD2RAD(360) so the RotateRobot fallbacks there are dead
        // code: the hull simply tracks a live `env.target`; with no
        // live target the weapons stop (`StopFire()` at :948 — this is
        // what makes robots cease fire once their victim dies or a
        // captured building's turret flips to their own side) and the
        // hull follows chassis forward.
        if self.is_arcaded {
            // Manual aim (MatrixRobot.cpp:843-847): the hull tracks
            // the cursor trace while it's off the 2D UI. The env
            // auto-aim/auto-stop-fire below is skipped entirely for
            // the arcaded robot (:853, :946).
            if self.state == RobotState::Idle && !objs.arcade_input.cursor_on_interface {
                let cw = objs.arcade_input.cursor_world;
                self.rotate_hull(glam::Vec2::new(cw.x, cw.y), cms);
            }
            if self.hull_sound_pending {
                self.hull_sound_pending = false;
                // PlayHullSound (MatrixRobot.cpp:5379-5394) — keyed on
                // the armor kind.
                let armor_kind = self.config.hull.unit.kind.0;
                let name = match armor_kind {
                    1 => "s_hull_passive",
                    2 => "s_hull_active",
                    3 => "s_hull_fireproof",
                    4 => "s_hull_plasmic",
                    5 => "s_hull_nuclear",
                    _ => "s_hull_6",
                };
                // `CSound::Play(S_HULL_*, pos, SL_HULL)` — immediate
                // positional on the hull layer, not the dedup path.
                objs.queue_snd_play_at(
                    name,
                    self.core.geo_center,
                    crate::matrix_game::sound::SoundLayer::Hull,
                    None,
                );
            }
        } else if matches!(self.state, RobotState::Idle | RobotState::BaseCapture) {
            let here = glam::Vec2::new(self.pos_x, self.pos_y);
            let mut hull_target: Option<glam::Vec2> = None;
            if let Some(t) = self.env.target {
                if let Some(tr) = crate::matrix_game::logic::robot_ref(objs, t) {
                    // :855 — target robot must be SUCCESSFULLY_BUILD.
                    if tr.state == RobotState::Idle {
                        hull_target = Some(glam::Vec2::new(tr.pos_x, tr.pos_y));
                    }
                } else if let Some(c) = crate::matrix_game::logic::cannon_ref(objs, t) {
                    // :891 — IsLiveActiveCannon.
                    if !matches!(c.state, CannonState::Dip | CannonState::UnderConstruction) {
                        hull_target =
                            Some(glam::Vec2::new(c.core().geo_center.x, c.core().geo_center.y));
                    }
                } else if let Some(b) = crate::matrix_game::logic::building_ref(objs, t) {
                    // :924 — live building, unless mid-capture.
                    if b.is_live() && !self.orders.has(OrderType::CaptureFactory) {
                        hull_target = Some(glam::Vec2::new(self.weapon_dir.x, self.weapon_dir.y));
                    }
                }
            }
            match hull_target {
                Some(tp) => self.rotate_hull(tp, cms),
                None => {
                    self.stop_fire();
                    self.rotate_hull(here + self.forward, cms);
                }
            }
        }
    }

    fn side(&self) -> i32 {
        self.side
    }
    fn need_repair(&self) -> bool {
        self.hit_point < self.hit_point_max
    }

    fn is_live(&self) -> bool {
        self.state != RobotState::Dip
    }

    /// Port of `CMatrixRobotAI::Damage` (MatrixRobot.cpp:1827-2118).
    /// Not ported (visual / unported-subsystem hooks): hit sounds,
    /// explosion + crater effects, progress-bar + minimap flashes, the
    /// war camera, env target-switching on cannon hits, and the DIP
    /// unit-scatter animation.
    fn damage(
        &mut self,
        weap: crate::matrix_game::effects::weapon::Weapon,
        _pos: glam::Vec3,
        _dir: glam::Vec3,
        attacker_side: i32,
        _attacker: Option<ObjectId>,
        self_id: ObjectId,
        objs: &mut Objects,
    ) -> bool {
        use crate::matrix_game::common::PLAYER_SIDE;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_INSTANT_DEATH};

        if self.state == RobotState::Dip {
            return true;
        }

        let cfg = crate::matrix_game::config::global();
        let now = crate::matrix_game::map::current_elapsed_ms();

        let instant = weap == WEAPON_INSTANT_DEATH;
        if !instant {
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
            let dmg = idx.map(|i| cfg.robot_damages.table[i]).unwrap_or_default();

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

            // Cannon attacker → enemy list + retarget when idle
            // (MatrixRobot.cpp:1870-1878).
            if !friendly_fire {
                if let Some(att) = _attacker {
                    let attacker_is_enemy_cannon =
                        crate::matrix_game::logic::cannon_ref(objs, att)
                            .map(|c| {
                                !matches!(
                                    c.state,
                                    crate::matrix_game::object_cannon::CannonState::Dip
                                ) && c.side != self.side
                            })
                            .unwrap_or(false);
                    if attacker_is_enemy_cannon {
                        if !self.env.search_enemy(att) {
                            self.env.add_to_list(att);
                        }
                        let target_is_cannon = self
                            .env
                            .target_attack
                            .map(|t| {
                                crate::matrix_game::logic::cannon_ref(objs, t).is_some()
                            })
                            .unwrap_or(true);
                        let now_i = now as i32;
                        if target_is_cannon
                            && now_i - self.env.last_hit_target > 4000
                            && now_i - self.env.target_change > 1000
                        {
                            self.env.target_last = self.env.target_attack;
                            self.env.target_attack = Some(att);
                            self.env.target_change = now_i;
                        }
                    }
                }
            }

            if self.hit_point > dmg.mindamage as f32 {
                let mut damage = damagek
                    * if friendly_fire {
                        dmg.friend_damage as f32
                    } else {
                        dmg.damage as f32
                    };
                if weap == WEAPON_BIGBOOM {
                    damage -= damage * self.bomb_protect;
                }
                self.hit_point -= damage;
                // `m_MiniMapFlashTime = FLASH_PERIOD` (MatrixRobot.cpp:1896).
                if !friendly_fire {
                    self.mini_map_flash_time = 1000;
                }
            }

            if weap == WEAPON_FLAMETHROWER {
                self.object_state |= OBJECT_STATE_ABLAZE;
                self.last_delay_damage_side = attacker_side;
                self.ablaze_ttl = (self.ablaze_ttl + 300).min(5000);
                self.next_time_ablaze = now;
            } else if weap == WEAPON_LIGHTENING {
                self.low_level_stop_fire(objs);
                self.object_state |= OBJECT_STATE_SHORTED;
                self.last_delay_damage_side = attacker_side;
                let mut ttl = self.shorted_ttl + 500 - (500.0 * self.light_protect) as i32;
                let maxl = 3000 - (3000.0 * self.light_protect) as i32;
                if ttl > maxl {
                    ttl = maxl;
                }
                self.shorted_ttl = ttl;
                self.next_time_shorted = now;
            } else {
                self.last_delay_damage_side = 0;
            }

            if self.hit_point > 50.0
                && !matches!(
                    weap,
                    crate::matrix_game::effects::weapon::WEAPON_ABLAZE
                        | crate::matrix_game::effects::weapon::WEAPON_SHORTED
                        | WEAPON_LIGHTENING
                        | WEAPON_FLAMETHROWER
                )
            {
                // Impact flash (MatrixRobot.cpp:1937-1941).
                objs.pending_explosions
                    .push(crate::matrix_game::map_static::ExplosionSpawn {
                        pos: _pos,
                        props: &crate::matrix_game::effects::explosion::EXPLOSION_ROBOT_HIT,
                        fire: false,
                    });
            }
            if self.hit_point > 0.0 {
                return false;
            }
            if attacker_side != 0 && !friendly_fire {
                objs.inc_side_stat(attacker_side, |s| s.robot_kill += 1);
            }
        }

        // inst_death (MatrixRobot.cpp:1951-2116).
        //
        // Terron easter egg: an in-position robot's death permanently
        // reveals the hidden portrets (MatrixRobot.cpp:1952-1955).
        if self.in_position {
            crate::matrix_game::map::show_portrets();
        }
        //
        // Self-destruct bomb: enemy robots blow up when going down if
        // the blast favors them (danger > 0); the player's own robots
        // never auto-trigger (the C++ comments the call out).
        if self.side != PLAYER_SIDE {
            let boom = self.weapons.iter().find(|bw| {
                objs.weapons
                    .get(bw.weapon)
                    .map(|w| w.weapon_type() == WEAPON_BIGBOOM)
                    .unwrap_or(false)
            });
            if let Some(bw) = boom {
                let wdist = objs
                    .weapons
                    .get(bw.weapon)
                    .map(|w| w.weapon_dist())
                    .unwrap_or(0.0);
                let mut danger = 0.0f32;
                let here = glam::Vec2::new(self.pos_x, self.pos_y);
                for oid in objs.iter_units().collect::<Vec<_>>() {
                    if Some(oid) == self.self_id {
                        continue;
                    }
                    let Some(o) = objs.get(oid) else { continue };
                    if !o.is_live() {
                        continue;
                    }
                    match o.core().obj_type {
                        ObjectType::RobotAi => {
                            let r: &Robot =
                                unsafe { &*(o as *const dyn MapStatic as *const Robot) };
                            let d2 = (glam::Vec2::new(r.pos_x, r.pos_y) - here).length_squared();
                            if d2 < wdist * wdist {
                                let s = r.calc_strength(objs);
                                if r.side == self.side {
                                    danger -= s;
                                } else {
                                    danger += s;
                                }
                            }
                        }
                        ObjectType::Cannon => {
                            let c: &Cannon =
                                unsafe { &*(o as *const dyn MapStatic as *const Cannon) };
                            if c.state == CannonState::UnderConstruction {
                                continue;
                            }
                            let d2 = (c.pos - here).length_squared();
                            if d2 < wdist * wdist {
                                let s = c.get_strength();
                                if c.side == self.side {
                                    danger -= s;
                                } else {
                                    danger += s;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if danger > 0.0 {
                    let bw_weapon = bw.weapon;
                    if let Some(map) = crate::matrix_game::map::current_map() {
                        // Inline `BigBoom(nC)` without the MustDie tail
                        // — death continues right below.
                        let geo = self.core.geo_center;
                        let vel = glam::Vec3::new(self.velocity.x, self.velocity.y, 0.0)
                            / crate::matrix_game::logic::LOGIC_TAKT_PERIOD_MS as f32;
                        if let Some(we) = objs.weapons.get_mut(bw_weapon) {
                            we.modify(geo, glam::Vec3::new(self.pos_x, self.pos_y, 0.0), vel);
                            we.fire_begin(vel, Some(self_id));
                        }
                        // Use a throwaway RNG stream — the C++ Takt(1)
                        // path doesn't draw from the world RNG.
                        let mut rng = crate::matrix_game::logic::Rnd::new(1);
                        weapon_takt(objs, bw_weapon, 1.0, map, &mut rng);
                    }
                }
            }
        }
        self.must_die = false; // ResetMustDie

        // Close the base if we died on the pad — spawning OR capturing
        // (MatrixRobot.cpp:2026-2036 gates on IsDisableManual, not state).
        if self.is_disable_manual() {
            if let Some(base_id) = self.base {
                if let Some(obj_mut) = objs.get_mut(base_id) {
                    if matches!(obj_mut.core().obj_type, ObjectType::Building) {
                        let b: &mut crate::matrix_game::object_building::Building = unsafe {
                            &mut *(obj_mut as *mut dyn MapStatic
                                as *mut crate::matrix_game::object_building::Building)
                        };
                        if matches!(self.state, RobotState::InSpawn | RobotState::BaseMoveOut) {
                            b.object_state_clear(
                                crate::matrix_game::map_static::OBJECT_STATE_BUILDING_SPAWNBOT,
                            );
                        }
                        b.close();
                    }
                }
            }
        }

        // Death visuals (MatrixRobot.cpp:2000-2007): the big boom +
        // the crater under the wreck.
        let geo = self.core.geo_center;
        objs.pending_explosions
            .push(crate::matrix_game::map_static::ExplosionSpawn {
                pos: geo,
                props: &crate::matrix_game::effects::explosion::EXPLOSION_ROBOT_BOOM,
                fire: true,
            });
        // Visual randomness only — seeded off position/time so the
        // world RNG stream is untouched (the C++ uses libc rand()).
        let mut vrng = crate::matrix_game::logic::Rnd::new(
            ((now as i32) ^ (self.pos_x as i32) << 8 ^ (self.pos_y as i32)).max(1),
        );
        objs.pending_spots
            .push(crate::matrix_game::effects::landscape_spot::SpotSpawn {
                pos: glam::Vec2::new(geo.x, geo.y),
                angle: crate::matrix_game::effects::smoke_and_fire::fsrnd(
                    &mut vrng,
                    std::f32::consts::PI,
                ),
                scale: 4.0 + crate::matrix_game::effects::smoke_and_fire::frnd(&mut vrng, 2.0),
                kind: crate::matrix_game::effects::landscape_spot::SpotKind::Voronka,
            });
        let on_water = self.object_state & crate::matrix_game::map_static::ROBOT_FLAG_ONWATER != 0;

        // ReleaseMe cascade (MatrixRobot.cpp:5071-5160): drop
        // selection flags, break orders (releases a mid-capture
        // factory), clear capture candidates, erase this robot from
        // every other robot's environment, clear our own env.
        self.unselect();
        self.break_all_orders(objs);
        self.clear_capture_candidates();
        if let Some(me) = self.self_id {
            let others: Vec<ObjectId> = objs.iter_units().collect();
            for oid in others {
                if let Some(other) = crate::matrix_game::logic::robot_mut(objs, oid) {
                    other.env.remove_from_list(me);
                }
            }
            // Robots whose box is checked out right now (the killer,
            // mid-takt) are unreachable above — scrub them later.
            objs.pending_env_purge.push(me);
        }
        self.env.clear();
        self.env.reset();

        // Detach the transport before dying (Damage inst_death detaches
        // the carry data at MatrixRobot.cpp:1302-1310, ReleaseMe calls
        // Carry(NULL) at :5158-5161) — otherwise the flyer later drops
        // the corpse and carry_release resurrects it out of DIP.
        if let Some(fid) = self.cargo_flyer.take() {
            if let Some(f) = crate::matrix_game::logic::flyer_mut(objs, fid) {
                if f.carrying_robot() == self.self_id {
                    f.carry_detach();
                }
            }
        }

        self.state = RobotState::Dip;
        self.dip_ttl = 5000; // FRND(3000)+2000 unit TTL ceiling
        self.speed = 0.0;
        self.velocity = glam::Vec2::ZERO;
        self.init_dip_scatter(&mut vrng, on_water);
        for bw in self.weapons.drain(..) {
            objs.weapons.release(bw.weapon);
        }
        true
    }
}

// ── SOrder / m_OrdersList ───────────────────────────────────────────────
//
// Port of `SOrder` + `CMatrixRobotAI::m_OrdersList[MAX_ORDERS]`
// (MatrixRobot.hpp:123-231). The C++ stores orders in a ring-like array
// that's MoveMemory-shuffled on insert/remove; we use a fixed-capacity
// `Vec<Order>` with manual insert-at-top + remove-at-index for the same
// push-to-top semantics.
//
// Params carry:
//   ROT_MOVE_TO     : p1,p2 = destination move-cell X,Y
//   ROT_MOVE_TO_BACK: same as MOVE_TO, drives in reverse
//   ROT_MOVE_RETURN : p1,p2 = return-to move-cell X,Y
//   ROT_STOP_MOVE   : unused
//   ROT_FIRE        : p1..3 = world-space target; p4 = type flag

/// Port of `OrderType` (MatrixRobot.hpp:123-136). `Empty` stands in
/// for `ROT_EMPTY_ORDER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderType {
    Empty,
    MoveTo,
    MoveToBack,
    MoveReturn,
    StopMove,
    Fire,
    StopFire,
    CaptureFactory,
    StopCapture,
}

/// How a ROT_CAPTURE_FACTORY dispatch ended (MatrixRobot.cpp:1203-1329):
/// `RobotConsumed` = base captured, robot StaticDelete'd (:1317);
/// `OrderPopped` = non-base capture done, order popped + `continue` (:1323).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureOutcome {
    Normal,
    RobotConsumed,
    OrderPopped,
}

/// Port of `OrderPhase` (MatrixRobot.hpp:138-151).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderPhase {
    Empty,
    WaitingForParams,
    Moving,
    Firing,
    CaptureMoving,
    CaptureInPosition,
    CaptureSettingUp,
    Capturing,
    GetingLost,
}

#[derive(Debug, Clone, Copy)]
pub struct Order {
    pub ty: OrderType,
    pub phase: OrderPhase,
    pub p1: f32,
    pub p2: f32,
    pub p3: f32,
    pub p4: i32,
    /// Port of `SOrder::m_Static` (MatrixRobot.hpp:190) — the object
    /// a CAPTURE_FACTORY / STOP_CAPTURE order references. The C++
    /// ref-counts the core; the arena's generational ids make a plain
    /// copy safe (a dead id simply fails `objs.get`).
    pub target: Option<ObjectId>,
}

impl Default for Order {
    fn default() -> Self {
        Self {
            ty: OrderType::Empty,
            phase: OrderPhase::Empty,
            p1: 0.0,
            p2: 0.0,
            p3: 0.0,
            p4: 0,
            target: None,
        }
    }
}

impl Order {
    /// Port of `SOrder::SetOrder(type, p1, p2, p3, p4)` (MatrixRobot.hpp:198).
    pub fn set(ty: OrderType, p1: f32, p2: f32, p3: f32, p4: i32) -> Self {
        Self {
            ty,
            phase: OrderPhase::Empty,
            p1,
            p2,
            p3,
            p4,
            target: None,
        }
    }

    /// Port of `SOrder::SetOrder(type, CMatrixMapStatic*)`
    /// (MatrixRobot.hpp:206).
    pub fn set_static(ty: OrderType, target: Option<ObjectId>) -> Self {
        Self {
            ty,
            phase: OrderPhase::Empty,
            p1: 0.0,
            p2: 0.0,
            p3: 0.0,
            p4: 0,
            target,
        }
    }

    /// Port of `SOrder::SetPhase` (MatrixRobot.hpp:215).
    pub fn set_phase(&mut self, phase: OrderPhase) {
        self.phase = phase;
    }
}

/// `MAX_ORDERS` from MatrixRobot.hpp:33.
pub const MAX_ORDERS: usize = 5;

/// Port of `m_OrdersList[MAX_ORDERS]` + `m_OrdersInPool`.
/// `AllocPlaceForOrderOnTop` shifts every existing order +1 and returns
/// `&m_OrdersList[0]` — so the head of the list is always the most
/// recently-added order.
#[derive(Debug, Default, Clone)]
pub struct OrderList {
    orders: Vec<Order>,
    /// Static targets of dropped orders, pending the `SOrder::Release`
    /// side effect (MatrixRobot.hpp:219-229): if the target is a
    /// building, its `capturer` must be reset. Drained by the robot's
    /// logic takt (which has `Objects` access).
    pub released: Vec<ObjectId>,
}

impl OrderList {
    pub fn new() -> Self {
        Self {
            orders: Vec::with_capacity(MAX_ORDERS),
            released: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.orders.len()
    }
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
    pub fn is_full(&self) -> bool {
        self.orders.len() >= MAX_ORDERS
    }

    pub fn top(&self) -> Option<&Order> {
        self.orders.first()
    }
    pub fn top_mut(&mut self) -> Option<&mut Order> {
        self.orders.first_mut()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Order> {
        self.orders.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Order> {
        self.orders.iter_mut()
    }

    /// Port of `CMatrixRobotAI::AllocPlaceForOrderOnTop`
    /// (MatrixRobot.cpp:4554-4570).
    pub fn push_top(&mut self, order: Order) -> bool {
        if self.is_full() {
            return false;
        }
        self.orders.insert(0, order);
        true
    }

    /// Port of `CMatrixRobotAI::RemoveOrderFromTop`.
    pub fn pop_top(&mut self) -> Option<Order> {
        if self.orders.is_empty() {
            None
        } else {
            let o = self.orders.remove(0);
            if let Some(t) = o.target {
                self.released.push(t);
            }
            Some(o)
        }
    }

    /// Port of `CMatrixRobotAI::RemoveOrder(OrderType)`.
    /// Drop every order (the `while(m_OrdersInPool>0) RemoveOrder(...)`
    /// tail of BreakAllOrders).
    pub fn clear(&mut self) {
        for o in &self.orders {
            if let Some(t) = o.target {
                self.released.push(t);
            }
        }
        self.orders.clear();
    }

    pub fn remove_type(&mut self, ty: OrderType) {
        let released = &mut self.released;
        self.orders.retain(|o| {
            if o.ty == ty {
                if let Some(t) = o.target {
                    released.push(t);
                }
                false
            } else {
                true
            }
        });
    }

    /// Port of `CMatrixRobotAI::FindOrderLikeThat(OrderType)`.
    /// Port of `FindOrder` (MatrixRobot.cpp:4929-4940) — order of
    /// `ty` whose static target is `obj`.
    pub fn find_order(&self, ty: OrderType, obj: ObjectId) -> bool {
        self.orders
            .iter()
            .any(|o| o.ty == ty && o.target == Some(obj))
    }

    /// Port of `RemoveOrder(int pos)` (MatrixRobot.cpp:4582-4589).
    pub fn remove_at(&mut self, pos: usize) {
        if pos < self.orders.len() {
            let o = self.orders.remove(pos);
            if let Some(t) = o.target {
                self.released.push(t);
            }
        }
    }

    pub fn has(&self, ty: OrderType) -> bool {
        self.orders.iter().any(|o| o.ty == ty)
    }

    /// Port of `CMatrixRobotAI::FindOrderLikeThat(OrderType, OrderPhase)`.
    pub fn has_with_phase(&self, ty: OrderType, phase: OrderPhase) -> bool {
        self.orders.iter().any(|o| o.ty == ty && o.phase == phase)
    }

    /// Port of `CMatrixRobotAI::ProcessOrdersList` (MatrixRobot.cpp:
    /// 4604-4615) — rotate the top order to the back so each pool
    /// entry gets one dispatch per LogicTakt.
    pub fn rotate_top_to_back(&mut self) {
        if self.orders.len() > 1 {
            let top = self.orders.remove(0);
            self.orders.push(top);
        }
    }
}

#[cfg(test)]
mod orders_tests {
    use super::*;

    #[test]
    fn push_top_limits_to_max_orders() {
        let mut ol = OrderList::new();
        for _ in 0..MAX_ORDERS {
            assert!(ol.push_top(Order::set(OrderType::MoveTo, 0.0, 0.0, 0.0, 0)));
        }
        assert!(ol.is_full());
        assert!(!ol.push_top(Order::set(OrderType::MoveTo, 0.0, 0.0, 0.0, 0)));
    }

    #[test]
    fn push_top_is_most_recent_first() {
        let mut ol = OrderList::new();
        ol.push_top(Order::set(OrderType::MoveTo, 1.0, 0.0, 0.0, 0));
        ol.push_top(Order::set(OrderType::Fire, 2.0, 0.0, 0.0, 0));
        assert_eq!(ol.top().unwrap().ty, OrderType::Fire);
    }

    #[test]
    fn remove_type_drops_all_matches() {
        let mut ol = OrderList::new();
        ol.push_top(Order::set(OrderType::MoveTo, 1.0, 0.0, 0.0, 0));
        ol.push_top(Order::set(OrderType::Fire, 0.0, 0.0, 0.0, 0));
        ol.push_top(Order::set(OrderType::MoveTo, 2.0, 0.0, 0.0, 0));
        ol.remove_type(OrderType::MoveTo);
        assert_eq!(ol.len(), 1);
        assert_eq!(ol.top().unwrap().ty, OrderType::Fire);
    }
}

// ── Combat: weapons + damage (CMatrixRobotAI combat subset) ─────────────

use crate::matrix_game::effects::weapon::{
    weapon_takt, WeaponEffect, WeaponHandler, WEAPON_BIGBOOM, WEAPON_BOMB, WEAPON_FLAMETHROWER,
    WEAPON_GUN, WEAPON_HOMING_MISSILE, WEAPON_LASER, WEAPON_LIGHTENING, WEAPON_MAX_HEAT,
    WEAPON_PLASMA, WEAPON_REPAIR, WEAPON_VOLCANO,
};
use crate::matrix_game::map_static::{OBJECT_STATE_ABLAZE, OBJECT_STATE_SHORTED};

/// `BARREL_TO_SHOT_ANGLE` (MatrixRobot.hpp:31) — 15°.
pub const BARREL_TO_SHOT_ANGLE: f32 = 15.0 * std::f32::consts::PI / 180.0;

impl Robot {
    /// Port of `CMatrixRobotAI::RobotWeaponInit` (MatrixRobot.cpp:
    /// 4052-4142) + the head-modifier application from `CalcRobotMass`
    /// (MatrixRobot.cpp:4248-4305): create one weapon effect per
    /// mounted weapon unit, seed heat/cool mods from `m_Overheat`, and
    /// apply the head's COOL_DOWN / FIRE_DISTANCE / OVERHEAT /
    /// LIGHT_PROTECT / BOMB_PROTECT percentages.
    pub fn robot_weapon_init(&mut self, objs: &mut Objects) {
        let Some(self_id) = self.self_id else { return };
        if !self.weapons.is_empty() {
            return;
        }
        let cfg = crate::matrix_game::config::global();

        // Head modifiers (CalcRobotMass).
        let (mut cool_down, mut fire_dist, mut overheat) = (0.0f32, 0.0f32, 0.0f32);
        if !self.config.head.kind.is_empty() {
            let h = self.config.head.kind.as_index();
            let hc = &cfg.head_chars;
            cool_down = (hc.cool_down[h.min(3)] / 100.0).max(-1.0);
            fire_dist = (hc.fire_distance[h.min(3)] / 100.0).max(-1.0);
            overheat = (hc.overheat[h.min(3)] / 100.0).max(-1.0);
            self.light_protect = (hc.light_protect[h.min(3)] / 100.0).clamp(0.0, 1.0);
            self.bomb_protect = (hc.bomb_protect[h.min(3)] / 100.0).clamp(0.0, 1.0);
        }

        for (i, w) in self.config.weapon.iter().enumerate() {
            if w.kind.is_empty() {
                continue;
            }
            // RUK_WEAPON_* → (EWeapon, heat entry). The C++ wires
            // RUK_WEAPON_ELECTRIC with a NULL handler (MatrixRobot.cpp
            // :4118) — preserved.
            let (wtype, oh, handler) = match w.kind.0 {
                1 => (
                    WEAPON_VOLCANO,
                    cfg.overheat.volcano,
                    WeaponHandler::Robot(self_id),
                ),
                2 => (WEAPON_GUN, cfg.overheat.gun, WeaponHandler::Robot(self_id)),
                3 => (
                    WEAPON_HOMING_MISSILE,
                    cfg.overheat.homing_missile,
                    WeaponHandler::Robot(self_id),
                ),
                4 => (
                    WEAPON_FLAMETHROWER,
                    cfg.overheat.flamethrower,
                    WeaponHandler::Robot(self_id),
                ),
                5 => (
                    WEAPON_BOMB,
                    cfg.overheat.bomb,
                    WeaponHandler::Robot(self_id),
                ),
                6 => (
                    WEAPON_LASER,
                    cfg.overheat.laser,
                    WeaponHandler::Robot(self_id),
                ),
                7 => (
                    WEAPON_BIGBOOM,
                    crate::matrix_game::config::OverheatParams {
                        heat_mod: 10,
                        cool_period: 0,
                        cool_mod: 10,
                    },
                    WeaponHandler::Robot(self_id),
                ),
                8 => (
                    WEAPON_PLASMA,
                    cfg.overheat.plasma,
                    WeaponHandler::Robot(self_id),
                ),
                9 => (
                    WEAPON_LIGHTENING,
                    cfg.overheat.lightening,
                    WeaponHandler::None,
                ),
                10 => (
                    WEAPON_REPAIR,
                    crate::matrix_game::config::OverheatParams {
                        heat_mod: 10,
                        cool_period: 0,
                        cool_mod: 10,
                    },
                    WeaponHandler::Robot(self_id),
                ),
                _ => continue,
            };
            let mut we = WeaponEffect::new(wtype, 0, handler);
            we.set_owner(self_id, self.side);
            // Head bonuses (MatrixRobot.cpp:4299-4302).
            we.modify_cool_down(cool_down);
            we.modify_dist(fire_dist);
            // `Float2Int(heat + heat*overheat)` (MatrixRobot.cpp:4306)
            // — round-to-nearest, not truncation.
            let heat_mod = crate::matrix_game::common::float2int(
                oh.heat_mod as f32 + oh.heat_mod as f32 * overheat,
            );
            let wdist = we.weapon_dist();
            let id = objs.weapons.create(we);
            self.weapons.push(BotWeapon {
                weapon: id,
                unit: i,
                heat: 0,
                heat_period: 0,
                heat_mod,
                cool_down_period: 0,
                cool_down_mod: oh.cool_mod,
                on: true,
            });
            // Fire-distance extremes (CalcRobotMass,
            // MatrixRobot.cpp:4294-4330): BIGBOOM skipped, REPAIR
            // feeds repair_dist only.
            match wtype {
                WEAPON_BIGBOOM => {}
                WEAPON_REPAIR => {
                    self.repair_dist = if self.repair_dist == 0.0 {
                        wdist
                    } else {
                        self.repair_dist.min(wdist)
                    };
                }
                _ => {
                    self.min_fire_dist = if self.max_fire_dist == 0.0 {
                        wdist
                    } else {
                        self.min_fire_dist.min(wdist)
                    };
                    self.max_fire_dist = self.max_fire_dist.max(wdist);
                }
            }
        }

        // m_HaveRepair (CalcRobotMass, MatrixRobot.cpp:4226-4231):
        // 1 = repairer present, 2 = nothing but repairers (the
        // self-destruct bomb RUK_WEAPON_BOMB doesn't count).
        let mut have_repair = 0;
        let mut normal_weapon = false;
        for w in self.config.weapon.iter() {
            match w.kind.0 {
                0 => {}
                10 => have_repair = 1,
                7 => {}
                _ => normal_weapon = true,
            }
        }
        if have_repair != 0 && !normal_weapon {
            have_repair = 2;
        }
        self.have_repair = have_repair;

        // m_AimProtect (MatrixRobot.cpp:4287-4288).
        if !self.config.head.kind.is_empty() {
            let h = self.config.head.kind.as_index();
            self.aim_protect = (cfg.head_chars.aim_protect[h.min(3)] / 100.0).clamp(0.0, 1.0);
        }
    }

    /// Port of `CMatrixRobotAI::Fire` (MatrixRobot.cpp:4734-4742).
    pub fn fire(&mut self, fire_pos: glam::Vec3, ftype: i32) {
        self.orders.remove_type(OrderType::Fire);
        self.orders.push_top(Order::set(
            OrderType::Fire,
            fire_pos.x,
            fire_pos.y,
            fire_pos.z,
            ftype,
        ));
    }

    /// Port of `CMatrixRobotAI::StopFire` (MatrixRobot.cpp:4744-4752):
    /// converts pending FIRE orders into STOP_FIRE in place.
    pub fn stop_fire(&mut self) {
        for o in self.orders.iter_mut() {
            if o.ty == OrderType::Fire {
                *o = Order::set(OrderType::StopFire, 0.0, 0.0, 0.0, 0);
            }
        }
    }

    /// Port of `CMatrixRobotAI::BreakAllOrders` (MatrixRobot.cpp:
    /// 4942-4981): full stop, env reset, weapons off, release any
    /// factory this robot is mid-capturing, drop the order pool.
    pub fn break_all_orders(&mut self, objs: &mut Objects) {
        self.low_level_stop();
        self.env.reset();
        self.low_level_stop_fire(objs);

        let my_id = self.self_id;
        let captures: Vec<ObjectId> = self
            .orders
            .iter()
            .filter(|o| o.ty == OrderType::CaptureFactory)
            .filter_map(|o| o.target)
            .collect();
        let disable_manual = self.is_disable_manual();
        for fid in captures {
            let Some(b) = crate::matrix_game::logic::building_mut(objs, fid) else {
                continue;
            };
            if b.capturer.is_some() && b.capturer == my_id {
                b.reset_captured();
                if !disable_manual {
                    if b.is_base()
                        && !matches!(
                            b.state,
                            crate::matrix_game::object_building::BaseState::Closed
                        )
                    {
                        b.close();
                    }
                } else {
                    self.must_die = true;
                }
            }
        }
        self.orders.clear();
    }

    /// Port of `LowLevelStopFire` (MatrixRobot.cpp:5044-5052).
    pub fn low_level_stop_fire(&mut self, objs: &mut Objects) {
        for bw in &self.weapons {
            if let Some(we) = objs.weapons.get_mut(bw.weapon) {
                we.fire_end();
            }
        }
    }

    /// Port of `CMatrixRobotAI::HitTo` (MatrixRobot.cpp:4025-4050) —
    /// the FIRE_END_HANDLER target. The env target-switching half
    /// needs `SMatrixRobotAIEnv` (not ported); the hit timestamps are
    /// the part current systems read.
    pub fn hit_to(
        &mut self,
        hit: Option<(ObjectId, i32, ObjectType)>,
        _pos: glam::Vec3,
        objs: &mut Objects,
    ) {
        let now = crate::matrix_game::map::current_elapsed_ms();
        let now_i = now as i32;
        let Some((hid, side, ty)) = hit else { return };

        if self.env.target == Some(hid) {
            self.env.last_hit_enemy = now_i;
            self.env.last_hit_target = now_i;
            self.last_hit_enemy_ms = now;
            self.last_hit_target_ms = now;
        } else if ty == ObjectType::Building {
            // buildings don't count
        } else if side != self.side {
            self.env.last_hit_enemy = now_i;
            self.last_hit_enemy_ms = now;
        } else {
            self.env.last_hit_friendly = now_i;
            self.last_hit_friendly_ms = now;
        }

        // Retaliation (MatrixRobot.cpp:4038-4048): a robot we hit
        // learns about us, and if it was busy shooting a cannon it
        // switches onto us.
        if ty == ObjectType::RobotAi {
            let Some(me_id) = self.self_id else { return };
            if hid == me_id {
                return;
            }
            let victim_is_enemy = crate::matrix_game::logic::robot_ref(objs, hid)
                .map(|v| v.side != self.side)
                .unwrap_or(false);
            if victim_is_enemy {
                if let Some(victim) = crate::matrix_game::logic::robot_mut(objs, hid) {
                    if !victim.env.search_enemy(me_id) {
                        victim.env.add_to_list(me_id);
                    }
                }
            }
            let do_retarget = {
                let victim_ok = crate::matrix_game::logic::robot_ref(objs, hid)
                    .map(|v| {
                        v.env
                            .target_attack
                            .map(|t| crate::matrix_game::logic::cannon_ref(objs, t).is_some())
                            .unwrap_or(false)
                            && v.env.search_enemy(me_id)
                    })
                    .unwrap_or(false);
                victim_ok && now_i - self.env.target_change > 1000
            };
            if do_retarget {
                self.env.target_last = self.env.target_attack;
                if let Some(victim) = crate::matrix_game::logic::robot_mut(objs, hid) {
                    victim.env.target_attack = Some(me_id);
                }
                self.env.target_change = now_i;
            }
        }
    }

    /// Port of `CMatrixRobotAI::BigBoom` (MatrixRobot.cpp:4755-4774):
    /// trigger the self-destruct bomb then flag MUST_DIE.
    pub fn big_boom(&mut self, map: &GameMap, objs: &mut Objects, rng: &mut Rnd) {
        let geo = self.core.geo_center;
        let vel = glam::Vec3::new(self.velocity.x, self.velocity.y, 0.0)
            / crate::matrix_game::logic::LOGIC_TAKT_PERIOD_MS as f32;
        for bw in &self.weapons {
            let is_boom = objs
                .weapons
                .get(bw.weapon)
                .map(|w| w.weapon_type() == WEAPON_BIGBOOM)
                .unwrap_or(false);
            if !is_boom {
                continue;
            }
            if let Some(we) = objs.weapons.get_mut(bw.weapon) {
                we.modify(geo, glam::Vec3::new(self.pos_x, self.pos_y, 0.0), vel);
                we.fire_begin(vel, self.self_id);
            }
            weapon_takt(objs, bw.weapon, 1.0, map, rng);
            break;
        }
        self.must_die = true;
    }

    /// Port of `CalcStrength` (MatrixRobot.cpp:5311-5330).
    pub fn calc_strength(&self, objs: &Objects) -> f32 {
        let cfg = crate::matrix_game::config::global();
        let mut s = 0.0f32;
        for bw in &self.weapons {
            let Some(we) = objs.weapons.get(bw.weapon) else {
                continue;
            };
            use crate::matrix_game::config::RobotUnitKind as K;
            let kind = match we.weapon_type() {
                WEAPON_PLASMA => K::WEAPON_PLASMA,
                WEAPON_VOLCANO => K::WEAPON_MACHINEGUN,
                WEAPON_HOMING_MISSILE => K::WEAPON_MISSILE,
                WEAPON_BOMB => K::WEAPON_MORTAR,
                WEAPON_FLAMETHROWER => K::WEAPON_FLAMETHROWER,
                WEAPON_BIGBOOM => K::WEAPON_BOMB,
                WEAPON_LIGHTENING => K::WEAPON_ELECTRIC,
                WEAPON_LASER => K::WEAPON_LASER,
                WEAPON_GUN => K::WEAPON_CANNON,
                _ => continue,
            };
            s += cfg.weapon_strength_ai.for_kind(kind);
        }
        s * (0.4 + 0.6 * (self.hit_point / self.hit_point_max.max(1.0)))
    }

    /// `SetWeaponToArcadedCoeff` / `SetWeaponToDefaultCoeff`
    /// (MatrixRobot.cpp:5396-5409).
    pub fn set_weapons_arcade_coeff(&self, objs: &mut Objects, arcade: bool) {
        for bw in &self.weapons {
            if let Some(we) = objs.weapons.get_mut(bw.weapon) {
                if arcade {
                    we.set_arcade_coefficient();
                } else {
                    we.set_default_coefficient();
                }
            }
        }
    }

    /// Fire-control block of `LogicTakt` (MatrixRobot.cpp:1364-1495) +
    /// the cool-down block (:1497-1542).
    fn weapons_logic_takt(&mut self, ms: i32, map: &GameMap, objs: &mut Objects, rng: &mut Rnd) {
        let arcaded = objs.arcaded_object.is_some() && objs.arcaded_object == self.self_id;
        let vel3 = glam::Vec3::new(self.velocity.x, self.velocity.y, 0.0)
            / crate::matrix_game::logic::LOGIC_TAKT_PERIOD_MS as f32;
        let geo = self.core.geo_center;

        for i in 0..self.weapons.len() {
            let (wid, on, heat, heat_mod) = {
                let bw = &self.weapons[i];
                (bw.weapon, bw.on, bw.heat, bw.heat_mod)
            };
            if objs.weapons.get(wid).is_none() || !(on || arcaded) {
                continue;
            }
            let wtype = objs.weapons.get(wid).map(|w| w.weapon_type()).unwrap();

            // Real muzzle from the renderer's part chain when
            // available; geo-center fallback before the first frame.
            let mount = self
                .weapons
                .get(i)
                .and_then(|b| self.weapon_mounts.get(b.unit))
                .copied()
                .flatten();
            let (v_pos, barrel2) = match mount {
                Some(m) => (m.pos, glam::Vec2::new(m.dir.x, m.dir.y)),
                None => (self.core.geo_center, self.hull_forward),
            };
            // Full 3D barrel axis — the C++ passes the matrix Y axis
            // verbatim (m._21.._23, MatrixRobot.cpp:1393), z included,
            // which the mortar uses for its steep-drop branch.
            let barrel3 = match mount {
                Some(m) => m.dir,
                None => glam::Vec3::new(self.hull_forward.x, self.hull_forward.y, 0.0),
            };

            // Heat budget check (MatrixRobot.cpp:1373-1384).
            let modk = if wtype == WEAPON_HOMING_MISSILE || wtype == WEAPON_BOMB {
                1
            } else {
                2
            };
            if heat + heat_mod * modk > WEAPON_MAX_HEAT {
                if let Some(we) = objs.weapons.get_mut(wid) {
                    we.fire_end();
                }
                continue;
            }

            let fire_dist = objs
                .weapons
                .get(wid)
                .map(|w| w.weapon_dist())
                .unwrap_or(0.0);
            let v_dir = (self.weapon_dir - v_pos).normalize_or_zero();

            // Range check — AI robots only (:1403-1408).
            if !arcaded && (geo - self.weapon_dir).length_squared() > fire_dist * fire_dist {
                if let Some(we) = objs.weapons.get_mut(wid) {
                    we.fire_end();
                }
                continue;
            }

            match wtype {
                WEAPON_BOMB => {
                    // (:1413-1431) — bomb aims at m_WeaponDir itself.
                    if let Some(we) = objs.weapons.get_mut(wid) {
                        we.modify(v_pos, barrel3, self.weapon_dir);
                    }
                    weapon_takt(objs, wid, ms as f32, map, rng);
                    if objs
                        .weapons
                        .get_mut(wid)
                        .map(|w| w.is_fire_was())
                        .unwrap_or(false)
                    {
                        let bw = &mut self.weapons[i];
                        bw.heat = (bw.heat + bw.heat_mod).min(WEAPON_MAX_HEAT);
                    }
                    continue;
                }
                WEAPON_BIGBOOM => continue, // (:1433) — fired only on death
                WEAPON_HOMING_MISSILE => {
                    // (:1437-1450).
                    if let Some(we) = objs.weapons.get_mut(wid) {
                        we.modify(v_pos, v_dir, vel3);
                    }
                    weapon_takt(objs, wid, ms as f32, map, rng);
                    if objs
                        .weapons
                        .get_mut(wid)
                        .map(|w| w.is_fire_was())
                        .unwrap_or(false)
                    {
                        let bw = &mut self.weapons[i];
                        bw.heat = (bw.heat + bw.heat_mod).min(WEAPON_MAX_HEAT);
                    }
                    continue;
                }
                _ => {}
            }

            if let Some(we) = objs.weapons.get_mut(wid) {
                we.modify(v_pos, v_dir, vel3);
            }

            // Barrel-angle gate (:1465-1475) — 2D angle between the
            // barrel direction and the target direction.
            let dir1 = barrel2.normalize_or_zero();
            let dir2 = glam::Vec2::new(v_dir.x, v_dir.y).normalize_or_zero();
            let angle = dir1.dot(dir2).clamp(-1.0, 1.0).acos();
            if angle > BARREL_TO_SHOT_ANGLE && wtype != WEAPON_REPAIR {
                if let Some(we) = objs.weapons.get_mut(wid) {
                    we.fire_end();
                }
                continue;
            }

            weapon_takt(objs, wid, ms as f32, map, rng);
            if objs
                .weapons
                .get_mut(wid)
                .map(|w| w.is_fire_was())
                .unwrap_or(false)
            {
                let bw = &mut self.weapons[i];
                bw.heat = (bw.heat + bw.heat_mod).min(WEAPON_MAX_HEAT);
            }
        }

        // Cool-down (MatrixRobot.cpp:1497-1542).
        let cfg = crate::matrix_game::config::global();
        for bw in &mut self.weapons {
            let Some(we) = objs.weapons.get(bw.weapon) else {
                continue;
            };
            let wtype = we.weapon_type();
            if wtype == WEAPON_BIGBOOM {
                continue;
            }
            if bw.heat > 0 {
                bw.cool_down_period += ms;
            } else {
                bw.cool_down_period = 0;
            }
            bw.heat_period = 0;
            let period = cfg
                .overheat
                .for_weapon(wtype)
                .map(|o| o.cool_period)
                .unwrap_or(0);
            while bw.heat > 0 && bw.cool_down_period >= period {
                bw.heat = (bw.heat - bw.cool_down_mod).max(0);
                bw.cool_down_period -= period;
                if bw.cool_down_mod <= 0 {
                    break; // zero cool rate would never converge
                }
            }
        }
    }

    /// Ablaze / shorted DOT loops at the head of `LogicTakt`
    /// (MatrixRobot.cpp:416-507). Fire/lightning visuals are skipped;
    /// the periodic self-damage is the mechanic. Returns true when the
    /// robot died mid-loop.
    fn dot_takt(&mut self, objs: &mut Objects) -> bool {
        let now = crate::matrix_game::map::current_elapsed_ms();
        let Some(self_id) = self.self_id else {
            return false;
        };
        if self.object_state & OBJECT_STATE_ABLAZE != 0 {
            while now > self.next_time_ablaze {
                self.next_time_ablaze += 90; // OBJECT_ROBOT_ABLAZE_PERIOD_EFFECT
                let pos = self.core.geo_center;
                let side = self.last_delay_damage_side;
                // Flame tongue at a random surface point — the
                // shleif AddFire at MatrixRobot.cpp:444-449.
                {
                    use crate::matrix_game::effects::smoke_and_fire::{fsrnd, Fire};
                    let mut vrng = crate::matrix_game::logic::Rnd::new(
                        ((now as i32) ^ ((pos.x + pos.y) as i32)).max(1),
                    );
                    let r = self.core.radius;
                    let fp = pos
                        + glam::Vec3::new(
                            fsrnd(&mut vrng, r),
                            fsrnd(&mut vrng, r),
                            fsrnd(&mut vrng, r) * 0.5,
                        );
                    objs.pending_effects
                        .push(crate::matrix_game::effects::GameEffect::Fire(Fire::new(
                            fp,
                            100.0,
                            1500.0,
                            30.0,
                            2.5,
                            false,
                            1.0 / 35.0,
                        )));
                }
                for _ in (0..90).step_by(10) {
                    // OBJECT_ROBOT_ABLAZE_PERIOD
                    if self.damage(
                        crate::matrix_game::effects::weapon::WEAPON_ABLAZE,
                        pos,
                        glam::Vec3::Z,
                        side,
                        None,
                        self_id,
                        objs,
                    ) {
                        return true;
                    }
                    if self.state == RobotState::Dip {
                        return true;
                    }
                }
            }
        }
        if self.object_state & OBJECT_STATE_SHORTED != 0 {
            while now > self.next_time_shorted {
                self.next_time_shorted += 50; // OBJECT_SHORTED_PERIOD
                let pos = self.core.geo_center;
                let side = self.last_delay_damage_side;
                // Arc between two random surface points
                // (MatrixRobot.cpp:478-501).
                {
                    use crate::matrix_game::effects::lightening::Lightening;
                    use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd};
                    let mut vrng = crate::matrix_game::logic::Rnd::new(
                        ((now as i32) ^ ((pos.x * 3.0 + pos.y) as i32)).max(1),
                    );
                    let r = self.core.radius;
                    let rndp = |vr: &mut crate::matrix_game::logic::Rnd| {
                        pos + glam::Vec3::new(fsrnd(vr, r), fsrnd(vr, r), fsrnd(vr, r))
                    };
                    let d1 = rndp(&mut vrng);
                    let d2 = rndp(&mut vrng);
                    objs.pending_effects
                        .push(crate::matrix_game::effects::GameEffect::Lightening(
                            Lightening::new_shorted(
                                d1,
                                d2,
                                frnd(&mut vrng, 400.0) + 100.0,
                                0xFFFF_FFFF,
                            ),
                        ));
                }
                if self.damage(
                    crate::matrix_game::effects::weapon::WEAPON_SHORTED,
                    pos,
                    glam::Vec3::Z,
                    side,
                    None,
                    self_id,
                    objs,
                ) {
                    return true;
                }
                if self.state == RobotState::Dip {
                    return true;
                }
            }
        }
        false
    }

    /// `DIPTakt` wreck physics (MatrixRobot.cpp:178-283).
    fn dip_takt(&mut self, cms: i32, map: &GameMap, rng: &mut Rnd, objs: &mut Objects) {
        use crate::matrix_game::common::{TRACE_ALL, WATER_LEVEL};
        use crate::matrix_game::effects::konus::Konus;
        use crate::matrix_game::effects::smoke_and_fire::fsrnd;
        use crate::matrix_game::effects::GameEffect;
        use crate::matrix_game::map_trace::{trace, TraceStop};

        let ms = cms as f32;
        let self_id = self.self_id;
        let mut i = 0;
        while i < self.dip_units.len() {
            let u = &mut self.dip_units[i];
            u.ttl -= ms;
            u.smoke.takt(ms, rng);
            if u.ttl <= 0.0 {
                // End of life — small boom 10 above (:190-208).
                let mut pos = u.pos;
                pos.z += 10.0;
                objs.pending_explosions
                    .push(crate::matrix_game::map_static::ExplosionSpawn {
                        pos,
                        props: &crate::matrix_game::effects::explosion::EXPLOSION_ROBOT_BOOM_SMALL,
                        fire: true,
                    });
                let mut u = self.dip_units.swap_remove(i);
                u.smoke.set_ttl(1000.0);
                // Hand the dying smoke emitter to the world so it
                // finishes fading.
                objs.pending_effects.push(GameEffect::Smoke(u.smoke));
                continue;
            }
            if u.velocity != glam::Vec3::ZERO {
                let oldpos = u.pos;
                u.velocity.z -= 0.0002 * ms; // gravity (:215)
                u.pos += u.velocity * ms;
                let (o, hitpos) = trace(map, objs, oldpos, u.pos, TRACE_ALL, self_id);
                match o {
                    TraceStop::Water => {
                        if map.get_z(hitpos.x, hitpos.y) < WATER_LEVEL {
                            // Sinks silently: splash, no end-of-life
                            // boom (the C++ zeroes the TTL, and
                            // TTL<=0 units are skipped — :224-236).
                            objs.queue_snd_at("splash", hitpos);
                            objs.pending_effects.push(GameEffect::Konus(Konus::new_splash(
                                hitpos,
                                glam::Vec3::Z,
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
                        u.velocity = glam::Vec3::ZERO;
                        u.pos = hitpos;
                        u.freeze_t =
                            Some(crate::matrix_game::map::current_elapsed_ms() as f32);
                        // Landing off-map OR on a base footprint kills
                        // the piece silently (`UnitGetTest == NULL ||
                        // m_Base`, MatrixRobot.cpp:243-247).
                        if !map.contains_point(hitpos.x, hitpos.y)
                            || crate::matrix_game::logic::is_on_base(objs, hitpos.x, hitpos.y)
                        {
                            let mut u = self.dip_units.swap_remove(i);
                            u.smoke.set_ttl(1000.0);
                            objs.pending_effects.push(GameEffect::Smoke(u.smoke));
                            continue;
                        }
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

    /// Death scatter init — the unit TTL/velocity/spin seeding of the
    /// robot Damage death flow (MatrixRobot.cpp:2040-2110). Each piece
    /// is one of the robot's real parts starting at its assembled
    /// position (renderer snapshot); the hull gets no piece (the C++
    /// zeroes MRT_ARMOR's TTL so it never flies or draws). `onair`
    /// (death above the ground by more than the radius) launches every
    /// piece, chassis included, with random velocities.
    fn init_dip_scatter(&mut self, rng: &mut Rnd, on_water: bool) {
        use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd, Smoke};
        let geo = self.core.geo_center;
        let snap = self.part_snapshot;
        let chassis_pos = snap.chassis.unwrap_or(geo);

        let onair = crate::matrix_game::map::current_map()
            .map(|m| m.get_z(chassis_pos.x, chassis_pos.y) + self.core.radius < chassis_pos.z)
            .unwrap_or(false);
        let mut cstay = frnd(rng, 1.0) < 0.5;
        if on_water {
            cstay = false;
        }

        let mut push = |pos: glam::Vec3,
                        part: DipPart,
                        invert: bool,
                        v: glam::Vec3,
                        spins: (f32, f32, f32),
                        yaw: f32,
                        rng: &mut Rnd| {
            let ttl = frnd(rng, 3000.0) + 2000.0;
            self.dip_units.push(DipUnit {
                pos,
                velocity: v,
                ttl,
                dp: spins.0,
                dy: spins.1,
                dr: spins.2,
                part,
                invert,
                yaw,
                freeze_t: None,
                smoke: Smoke::new(
                    pos,
                    ttl + 100_000.0,
                    1600.0,
                    80.0,
                    0xCF30_3030,
                    false,
                    1.0 / 30.0,
                ),
            });
        };

        // Unit 0 — chassis (:2061-2080). When it stays put its unit
        // matrix is never recomputed, keeping the death orientation.
        let stay = cstay && !onair;
        let chassis_yaw = if stay {
            let f = self.forward.normalize_or_zero();
            (-f.x).atan2(f.y)
        } else {
            0.0
        };
        let (v, spins) = if stay {
            (glam::Vec3::ZERO, (0.0, 0.0, 0.0))
        } else {
            let spins = (fsrnd(rng, 0.0005), fsrnd(rng, 0.0005), fsrnd(rng, 0.0005));
            let v = if onair {
                glam::Vec3::new(fsrnd(rng, 0.08), fsrnd(rng, 0.08), fsrnd(rng, 0.1))
            } else {
                glam::Vec3::new(0.0, 0.0, 0.1)
            };
            (v, spins)
        };
        push(chassis_pos, DipPart::Chassis, false, v, spins, chassis_yaw, rng);

        // Head + weapons (:2086-2110); armor skipped.
        let mut parts: Vec<(glam::Vec3, DipPart, bool)> = Vec::new();
        if !self.config.head.kind.is_empty() {
            parts.push((snap.head.unwrap_or(geo), DipPart::Head, false));
        }
        for (p, w) in self.config.weapon.iter().enumerate() {
            if w.kind.is_empty() {
                continue;
            }
            let (pos, inv) = snap.weapons[p.min(4)].unwrap_or((geo, false));
            parts.push((pos, DipPart::Weapon(p), inv));
        }
        for (pos, part, inv) in parts {
            let spins = (fsrnd(rng, 0.005), fsrnd(rng, 0.005), fsrnd(rng, 0.005));
            let v = if onair {
                glam::Vec3::new(fsrnd(rng, 0.08), fsrnd(rng, 0.08), fsrnd(rng, 0.1))
            } else {
                glam::Vec3::new(fsrnd(rng, 0.08), fsrnd(rng, 0.08), 0.1)
            };
            push(pos, part, inv, v, spins, 0.0, rng);
        }
    }

    /// Wreck smoke trails into the billboard queue. The pieces
    /// themselves are the robot's part meshes, drawn by the robot
    /// renderer's DIP path.
    pub fn draw_dip(&self, q: &mut crate::matrix_lib::three_g::billboard::BillboardQueue) {
        for u in &self.dip_units {
            u.smoke.draw(q);
        }
    }

    /// ROT_FIRE / ROT_STOP_FIRE order processing (MatrixRobot.cpp:
    /// 1143-1205). Returns true when an order consumed the dispatch
    /// slot this takt.
    fn dispatch_fire_order(&mut self, objs: &mut Objects) -> bool {
        let Some(top) = self.orders.top().copied() else {
            return false;
        };
        match top.ty {
            OrderType::Fire => {
                // Arcaded: aim at the live cursor trace every takt
                // (MatrixRobot.cpp:1151-1160); AI/commanded: aim at
                // the order's params.
                self.weapon_dir = if self.is_arcaded {
                    objs.arcade_input.cursor_world
                } else {
                    glam::Vec3::new(top.p1, top.p2, top.p3)
                };
                let ftype = top.p4;
                let vel = glam::Vec3::new(self.velocity.x, self.velocity.y, 0.0)
                    / crate::matrix_game::logic::LOGIC_TAKT_PERIOD_MS as f32;
                let skip = self.self_id;
                for bw in &self.weapons {
                    let Some(we) = objs.weapons.get_mut(bw.weapon) else {
                        continue;
                    };
                    match we.weapon_type() {
                        WEAPON_REPAIR => {
                            if ftype == 2 {
                                we.fire_begin(vel, skip);
                            } else {
                                we.fire_end();
                            }
                        }
                        WEAPON_BOMB => {
                            if ftype == 0 {
                                we.fire_begin(glam::Vec3::new(top.p1, top.p2, top.p3), skip);
                            } else {
                                we.fire_end();
                            }
                        }
                        _ => {
                            if ftype == 0 {
                                we.fire_begin(vel, skip);
                            } else {
                                we.fire_end();
                            }
                        }
                    }
                }
                true // `break` — FIRE stays on top until STOP_FIRE
            }
            OrderType::StopFire => {
                self.low_level_stop_fire(objs);
                self.orders.pop_top();
                false // `continue` — let the next order dispatch
            }
            _ => false,
        }
    }
}
