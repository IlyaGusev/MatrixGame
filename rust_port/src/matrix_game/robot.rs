//! Minimal port of `CMatrixRobotAI` (MatrixRobot.{cpp,hpp}).
//!
//! CMatrixRobotAI is ~5000 lines (AI, pathfinding, state machine,
//! chassis/armor/weapon composition, animation, combat). This file
//! ports only the subset needed for a faithful build-robot flow:
//! place the robot at a spawn position with a side / chassis kind,
//! join the arena, and carry enough state that FindObjects +
//! selection queries do the right thing. AI and combat land with
//! their owning subsystems.

use crate::matrix_game::logic::{self, ROBOT_MOVECELLS_PER_SIZE};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectId, ObjectType, Objects, MR_ALL, MR_MATRIX,
};
use crate::matrix_game::map_trace::{self, MovePath, ROBOT_FOOTPRINT_HALF};
use crate::matrix_game::orders::{Order, OrderList, OrderPhase, OrderType};
use crate::matrix_game::rnd::Rnd;

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
    /// Port of `SMatrixRobotAIEnv::m_PlaceAdd`
    /// (MatrixRobot.hpp env struct). Set by `PGSetPlace` when the
    /// group-formation spiral search finds a free cell that doesn't
    /// map to a road-network `m_Place` entry — a direct move-cell
    /// upper-left corner the robot is heading for. The corresponding
    /// `m_Place` (road-network index) isn't ported since the road
    /// network itself isn't, so `m_Place >= 0` always resolves to
    /// `-1` here and we always fall back to `place_add`.
    ///
    /// Read by `PGAssignPlacePlayer` to avoid walking every group
    /// member onto the same destination, and by the move-to ping
    /// spawner to draw one ping per assigned cell.
    pub place_add: Option<(i32, i32)>,

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
    /// cursor (MatrixObjectRobot.hpp). Only the chassis unit is
    /// rendered right now so we carry exactly one `AnimState`;
    /// armor / weapon / head each get their own when they land.
    pub chassis_anim: crate::matrix_lib::three_g::animation::AnimState,

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
    pub config: crate::matrix_game::robot_units::RobotConfig,
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
            geo_center: pos + glam::Vec3::new(0.0, 0.0, 3.0),
            radius: 6.0,
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
            orders: OrderList::new(),
            move_path: MovePath::default(),
            des_x: 0,
            des_y: 0,
            map_x: 0,
            map_y: 0,
            velocity: glam::Vec2::ZERO,
            speed: 0.0,
            hull_forward: glam::Vec2::new(0.0, -1.0),
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
            place_add: None,
            animation: Animation::Off,
            chassis_anim: Default::default(),
            name: String::new(),
            team: 0,
            config: crate::matrix_game::robot_units::RobotConfig::new(),
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
        self.orders.remove_type(OrderType::MoveTo);
        self.orders.remove_type(OrderType::MoveToBack);
        self.orders.remove_type(OrderType::StopMove);
        self.orders
            .push_top(Order::set(OrderType::MoveTo, mx as f32, my as f32, 0.0, 0));
        self.des_x = mx;
        self.des_y = my;
        self.move_path.clear();
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
            crate::matrix_game::config::global().chassis.rotation_speed[self.chassis as usize];
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

    pub fn stop_moving(&mut self) {
        self.orders.remove_type(OrderType::MoveTo);
        self.orders.remove_type(OrderType::MoveToBack);
        self.move_path.clear();
        self.velocity = glam::Vec2::ZERO;
        self.speed = 0.0;
    }

    /// Port of `CMatrixRobotAI::GetLost(v)` (MatrixRobot.cpp:5188-
    /// 5237). Issues a ROT_MOVE_TO to a random nearby cell roughly
    /// perpendicular to `v`, tagged with `ROP_GETING_LOST` so the
    /// AI knows to abandon the order when it arrives.
    ///
    /// The C++ variant also has a "rotate if forward angle >70°"
    /// early-out; we skip the RotateRobot branch (rotation lands
    /// with the full Seek port).
    pub fn get_lost(&mut self, map: &GameMap, objs: &Objects, v: glam::Vec2, rng: &mut Rnd) {
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
        let chassis = self.chassis as usize;
        let Some((dx, dy)) = logic::place_find_near(
            map,
            objs,
            chassis,
            ROBOT_MOVECELLS_PER_SIZE,
            mx,
            my,
            4,
            None,
        ) else {
            return;
        };
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

    /// Port of `CMatrixRobotAI::MapPosCalc` — truncates pos_x/y to
    /// move-cell coords with a footprint-center offset so `map_x/y`
    /// points at the upper-left corner of the robot's 4×4 footprint.
    fn map_pos_calc(&mut self, map: &GameMap) {
        let (mx, my) = map.world_to_move(self.pos_x, self.pos_y);
        self.map_x = mx - ROBOT_FOOTPRINT_HALF;
        self.map_y = my - ROBOT_FOOTPRINT_HALF;
    }

    /// Port of the `ROT_MOVE_TO` dispatch at MatrixRobot.cpp:1020-
    /// 1112 — minus the arcade / capture-chain branches. If there's
    /// no active move-path, computes one via A*; if there is, hands
    /// off to `move_by_move_path`.
    fn dispatch_move_to(&mut self, cms: i32, map: &GameMap, objs: &Objects, elapsed_ms: i64) {
        self.map_pos_calc(map);

        // Recompute path if stale / empty.
        if !self.move_path.is_active() {
            let chassis = self.chassis as usize;
            let start = match logic::place_find_near(
                map,
                objs,
                chassis,
                ROBOT_MOVECELLS_PER_SIZE,
                self.map_x,
                self.map_y,
                4,
                None,
            ) {
                Some((x, y)) => map_trace::MovePt::new(x, y),
                None => {
                    self.stop_moving();
                    return;
                }
            };
            let goal = match logic::place_find_near(
                map,
                objs,
                chassis,
                ROBOT_MOVECELLS_PER_SIZE,
                self.des_x,
                self.des_y,
                4,
                None,
            ) {
                Some((x, y)) => map_trace::MovePt::new(x, y),
                None => {
                    self.stop_moving();
                    return;
                }
            };
            // Port of the `other_des` / `other_path_list` collection
            // at MatrixRobot.cpp:1613-1645. Each other live robot /
            // cannon contributes a `(pos, dest)` pair that raises the
            // per-cell A* weight inside a footprint window — routing
            // _prefers_ detours but doesn't hard-block them, matching
            // the C++ weight-30/200 scheme at MatrixLogic.cpp:1285,1297.
            let mut blockers: Vec<map_trace::Blocker> = Vec::new();
            for id in objs.iter_live() {
                let Some(other_obj) = objs.get(id) else {
                    continue;
                };
                if !matches!(other_obj.core().obj_type, ObjectType::RobotAi) {
                    continue;
                }
                let other: &Robot =
                    unsafe { &*(other_obj as *const dyn MapStatic as *const Robot) };
                let (omx, omy) = map.world_to_move(other.pos_x, other.pos_y);
                let pos =
                    map_trace::MovePt::new(omx - ROBOT_FOOTPRINT_HALF, omy - ROBOT_FOOTPRINT_HALF);
                // C++ :1626-1645: if MoveTo exists, blocker =
                // (current_pos, move_to_dest); otherwise blocker =
                // (none, current_pos) — robot is stationary at its
                // "destination". MoveReturn folds the same way.
                if let Some((dx, dy)) = other.move_to_coords() {
                    blockers.push(map_trace::Blocker {
                        pos: Some(pos),
                        dest: map_trace::MovePt::new(dx, dy),
                    });
                } else if let Some((dx, dy)) = other.return_coords() {
                    blockers.push(map_trace::Blocker {
                        pos: Some(pos),
                        dest: map_trace::MovePt::new(dx, dy),
                    });
                } else {
                    blockers.push(map_trace::Blocker {
                        pos: None,
                        dest: pos,
                    });
                }
            }

            let raw = map_trace::find_path(map, start, goal, chassis, &blockers);
            let Some(raw) = raw else {
                self.stop_moving();
                return;
            };
            let opt = map_trace::optimize_path(map, &raw, chassis);
            self.move_path.total_len = map_trace::path_total_length(&opt);
            self.move_path.followed_len = 0.0;
            self.move_path.pts = opt;
            self.move_path.cur = 0;
            self.move_test_pos = glam::Vec2::new(self.pos_x, self.pos_y);
            self.move_test_change_ms = elapsed_ms;
        }

        self.move_by_move_path(cms, map, objs, elapsed_ms);
    }

    /// Port of `CMatrixRobotAI::MoveByMovePath(ms)`
    /// (MatrixRobot.cpp:1708-1764). One LogicTakt slice of a cell-level
    /// waypoint follow: pick the current `(sou, des)` segment, fire
    /// `LowLevelMove` which folds Seek + collision, then advance `cur`
    /// when the projected progress along the segment exceeds the
    /// segment length (stop on the final waypoint). The stuck-watchdog
    /// + terrain-Z + ONWATER flag update all live here too.
    fn move_by_move_path(&mut self, cms: i32, map: &GameMap, objs: &Objects, elapsed_ms: i64) {
        let Some((sou_pt, des_pt)) = self.move_path.current_segment() else {
            self.stop_moving();
            return;
        };
        let (sou_x, sou_y) = map_trace::waypoint_to_world(sou_pt);
        let (des_x, des_y) = map_trace::waypoint_to_world(des_pt);
        let dest = glam::Vec2::new(des_x, des_y);

        let last_seg = self.move_path.cur + 1 == self.move_path.pts.len() - 1;
        // `globalend` in the C++ = `m_MovePathCur+1 == m_MovePathCnt-1`,
        // i.e. we're on the final leg AND there is no follow-up zone
        // segment. Zone chaining isn't ported yet, so `globalend` ==
        // `last_seg` always. Matches MatrixRobot.cpp:1723.
        let end_path = last_seg;

        // C++: LowLevelMove(ms, dest, robot_coll=true, obst_coll=true,
        //                    end_path=globalend && last, back=false).
        // The wrapper runs Seek (sets m_Velocity), then sphere/AABB +
        // robot-robot corrections, then integrates `pos += velocity *
        // m_SyncMul + result_coll`.
        self.low_level_move(cms, map, objs, dest, true, true, end_path, false);

        // Port of the segment-advance test at MatrixRobot.cpp:1725-1750.
        let seg = glam::Vec2::new(des_x - sou_x, des_y - sou_y);
        let seg_len_sq = seg.length_squared().max(1.0e-6);
        let me = glam::Vec2::new(self.pos_x - sou_x, self.pos_y - sou_y);
        // `proj` is the projected distance along the segment normalized
        // to segment length — equivalent to `(me·seg)/|seg|`.
        let proj = me.dot(seg) / seg_len_sq.sqrt();
        let seg_len = seg_len_sq.sqrt();
        let reached = if last_seg {
            (glam::Vec2::new(self.pos_x - des_x, self.pos_y - des_y)).length_squared() < 0.2
        } else {
            proj >= seg_len
        };
        if reached {
            self.move_path.cur += 1;
            if self.move_path.cur + 1 >= self.move_path.pts.len() {
                self.pos_x = des_x;
                self.pos_y = des_y;
                self.orders.remove_type(OrderType::MoveTo);
                self.move_path.clear();
                self.velocity = glam::Vec2::ZERO;
                self.speed = 0.0;
            }
        }

        // Stuck-watchdog — MatrixRobot.cpp:1753-1761.
        let dpos = glam::Vec2::new(self.pos_x, self.pos_y) - self.move_test_pos;
        if dpos.length_squared() > 25.0 {
            self.move_test_pos = glam::Vec2::new(self.pos_x, self.pos_y);
            self.move_test_change_ms = elapsed_ms;
        } else if elapsed_ms - self.move_test_change_ms > 2000 {
            self.move_path.clear();
        }

        // Port of `Z_From_Pos` (MatrixObjectRobot.cpp:277-296): sample
        // terrain height at the new pos, flip `ROBOT_FLAG_ONWATER` based
        // on `roboz < WATER_LEVEL`, and float hover/anti-grav chassis
        // back up to the water plane.
        self.pos_z = self.z_from_pos(map);
        self.core.geo_center.x = self.pos_x;
        self.core.geo_center.y = self.pos_y;
        self.core.geo_center.z = self.pos_z + 3.0;
        self.rchange |= MR_MATRIX;
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
        objs: &Objects,
        dest: glam::Vec2,
        robot_coll: bool,
        obst_coll: bool,
        end_path: bool,
        back: bool,
    ) {
        let _rotate = self.seek(cms, map, dest, end_path, back);
        self.cols = 0;

        let sync_mul = (cms as f32) / 10.0;
        let gmv = self.velocity * sync_mul;

        let mut r = glam::Vec2::ZERO;
        let mut result_coll = glam::Vec2::ZERO;

        if robot_coll {
            r = self.robot_to_object_collision(objs, gmv);
            result_coll = r;
        }
        if obst_coll {
            let o = sphere_robot_to_aabb_obstacle_collision(
                map,
                self.chassis as usize,
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
    }

    /// Port of `CMatrixRobotAI::Seek(dest, rotate, end_path, back)`
    /// (MatrixRobot.cpp:2394-2458). Rotates the robot one-tick toward
    /// `dest` (unless in `BASE_MOVEOUT` or going `back`), applies slope
    /// + water speed corrections, and sets `m_Velocity` / `m_Speed`.
    ///
    /// Returns the `rotate` bit the C++ passes through as an out-param —
    /// true when RotateRobot did a non-completing turn this tick, which
    /// feeds the Pneumatic `ROBOT_FLAG_COLLISION` heuristic in the
    /// full LowLevelMove (we don't use it yet).
    fn seek(
        &mut self,
        cms: i32,
        map: &GameMap,
        dest: glam::Vec2,
        end_path: bool,
        back: bool,
    ) -> bool {
        let mut rangle = 0.0;
        let mut rotate;
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
        let max_speed = chassis_max_speed(self.chassis);
        if self.group_speed <= 0.001 {
            self.group_speed = max_speed;
        }

        // Water + slope correction (MatrixRobot.cpp:2420-2442).
        let cfg = crate::matrix_game::config::global();
        let chassis_idx = self.chassis as usize;
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

        // C++ Seek always returns `true` at :2458.
        let _ = &mut rotate;
        true
    }

    /// Port of `CMatrixRobotAI::RobotToObjectCollision`
    /// (MatrixRobot.cpp:3059-3074). Calls `FindObjects` within `3R` to
    /// collect nearby robots / cannons and runs `CollisionCallback`
    /// (:2892-3003) on each hit. The callback:
    ///   - separates overlapping pairs along the connecting vector,
    ///   - stops forward progress when an aligned robot is ahead
    ///     (matching m_ColSpeed to half of theirs, :2933),
    ///   - increments `m_Cols` / `m_ColsWeight` / `m_ColsWeight2`
    ///     counters used by the AI's "get lost" heuristic.
    ///
    /// The Rust port carries the separation + m_Cols bits; the
    /// ColsWeight / stop-on-alignment machinery is faithful to the
    /// original but simplified to pairwise separation until the full
    /// group AI lands (m_ColSpeed still resets to 100 in the no-far
    /// branch at :3070 — the observable speed clamp).
    fn robot_to_object_collision(&mut self, objs: &Objects, vel: glam::Vec2) -> glam::Vec2 {
        const COLLIDE_BOT_R: f32 = 18.0;
        const COLLIDE_BOT_2R: f32 = COLLIDE_BOT_R + COLLIDE_BOT_R;

        // `my_pos + vel` — where the robot would end up this tick without
        // any collision response. The callback uses this for the distance
        // check at :2902.
        let my_pos = glam::Vec2::new(self.pos_x + vel.x, self.pos_y + vel.y);

        let mut result = glam::Vec2::ZERO;
        let mut far_col = false;

        for id in objs.iter_live() {
            let Some(other_obj) = objs.get(id) else {
                continue;
            };
            if !matches!(other_obj.core().obj_type, ObjectType::RobotAi) {
                continue;
            }
            let other: &Robot = unsafe { &*(other_obj as *const dyn MapStatic as *const Robot) };
            let their_pos = glam::Vec2::new(other.pos_x, other.pos_y);
            let dv = my_pos - their_pos;
            let dist = dv.length();
            if dist < COLLIDE_BOT_2R && dist > 1.0e-3 {
                let correction = (COLLIDE_BOT_2R - dist) * 0.5;
                result += (dv / dist) * correction;
                self.cols += 1;
                far_col = true;
            }
        }

        // MatrixRobot.cpp:3070 — no far collision → reset m_ColSpeed to
        // the cap, so Seek's `min(m_GroupSpeed, m_ColSpeed)` doesn't
        // stick at a stale clamp after the colliding robot moves away.
        if !far_col {
            self.col_speed = 100.0;
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
            }
            // C++ also calls `MustDie()` when `z < WATER_LEVEL - 100`
            // (drowned). Skipped until damage flow is ported.
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
    crate::matrix_game::config::global().chassis.move_speed[c as usize]
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

    fn takt(&mut self, _cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {}

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

        match self.state {
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
                self.core.geo_center.z = self.pos_z + 3.0;
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
                self.low_level_move(cms, map, &*objs, dest_far, true, false, false, false);

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
                self.core.geo_center.z = self.pos_z + 3.0;
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
                    self.base = None;
                    self.state = RobotState::Idle;
                    self.get_lost(map, &*objs, self.forward, rng);
                }
            }
            RobotState::Idle => {
                // Dispatch the top-of-pool order. For now only
                // ROT_MOVE_TO (and implicitly its GetingLost-phase
                // variant) is handled — the rest land with combat
                // / capture. Mirrors the while-loop around
                // MatrixRobot.cpp:1012.
                let top_ty = self.orders.top().map(|o| o.ty);
                if matches!(top_ty, Some(OrderType::MoveTo)) {
                    let o = *self.orders.top().unwrap();
                    self.des_x = o.p1 as i32;
                    self.des_y = o.p2 as i32;
                    // `dispatch_move_to` → `move_by_move_path` →
                    // `low_level_move` already runs `robot_to_object_
                    // collision` (the separation + stop-on-aligned
                    // branches), so no extra call is needed here.
                    self.dispatch_move_to(cms, map, &*objs, elapsed_ms);
                    self.core.geo_center.x = self.pos_x;
                    self.core.geo_center.y = self.pos_y;
                } else {
                    // No order in the pool — the C++ eventually hits
                    // `LowLevelStop` which zeroes m_Speed. That's
                    // what the do_animation block below reads to
                    // pick ANIMATION_STAY.
                    self.velocity = glam::Vec2::ZERO;
                    self.speed = 0.0;
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
        let chassis_idx = self.chassis as usize;
        if let Some(vo) = crate::matrix_lib::three_g::animation::chassis_vo(chassis_idx) {
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
    }

    fn side(&self) -> i32 {
        self.side
    }
    fn need_repair(&self) -> bool {
        self.hit_point < self.hit_point_max
    }
}
