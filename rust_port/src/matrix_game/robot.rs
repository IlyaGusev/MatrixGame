//! Minimal port of `CMatrixRobotAI` (MatrixRobot.{cpp,hpp}).
//!
//! CMatrixRobotAI is ~5000 lines (AI, pathfinding, state machine,
//! chassis/armor/weapon composition, animation, combat). This file
//! ports only the subset needed for a faithful build-robot flow:
//! place the robot at a spawn position with a side / chassis kind,
//! join the arena, and carry enough state that FindObjects +
//! selection queries do the right thing. AI and combat land with
//! their owning subsystems.

use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectId, Objects, ObjectType, MR_ALL, MR_MATRIX,
};
use crate::matrix_game::logic::{self, ROBOT_MOVECELLS_PER_SIZE};
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
    Pneumatic   = 0,
    Wheel       = 1,
    Track       = 2,
    AntiGravity = 3,
    Hovercraft  = 4,
}

impl ChassisKind {
    /// Default height offset above the spawn platform for each
    /// chassis (approximate, matches the per-chassis height tables
    /// scattered through MatrixRobot.cpp).
    pub fn spawn_z_offset(self) -> f32 {
        match self {
            ChassisKind::Pneumatic   => 9.0,
            ChassisKind::Wheel       => 8.0,
            ChassisKind::Track       => 5.0,
            ChassisKind::AntiGravity => 5.0,
            ChassisKind::Hovercraft  => 7.0,
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

    /// Port of `CMatrixRobot::m_Animation` (MatrixObjectRobot.hpp:
    /// 33-43). High-level symbolic state that `SwitchAnimation` uses
    /// to decide the transition graph.
    pub animation: Animation,
    /// Port of the first `SMatrixRobotUnit::m_Graph` animation
    /// cursor (MatrixObjectRobot.hpp). Only the chassis unit is
    /// rendered right now so we carry exactly one `AnimState`;
    /// armor / weapon / head each get their own when they land.
    pub chassis_anim: crate::matrix_lib::three_g::animation::AnimState,
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
            animation: Animation::Off,
            chassis_anim: Default::default(),
        }
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
            self.chassis_anim.next_anim_time =
                crate::matrix_game::map::current_elapsed_ms() as f64;
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
        self.orders.push_top(Order::set(
            OrderType::MoveTo, mx as f32, my as f32, 0.0, 0,
        ));
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
        let max_rot = crate::matrix_game::config::global()
            .chassis.rotation_speed[self.chassis as usize];
        let sync_mul = (cms as f32) / 10.0;
        let rot_speed = max_rot * sync_mul;

        let dest_dir = dest - glam::Vec2::new(self.pos_x, self.pos_y);
        let dest_dir_n = dest_dir.normalize_or_zero();
        let forward = self.forward.normalize_or_zero();

        // cos of angle between forward and dest direction.
        let cos1 = (forward.x * dest_dir_n.x + forward.y * dest_dir_n.y)
            .clamp(-1.0, 1.0);
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
    pub fn get_lost(
        &mut self,
        map: &GameMap,
        objs: &Objects,
        v: glam::Vec2,
        rng: &mut Rnd,
    ) {
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
            map, objs, chassis, ROBOT_MOVECELLS_PER_SIZE, mx, my, 4, None,
        ) else {
            return;
        };
        if !self.orders.has_with_phase(OrderType::MoveTo, OrderPhase::GetingLost) {
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
    fn dispatch_move_to(
        &mut self,
        cms: i32,
        map: &GameMap,
        objs: &Objects,
        elapsed_ms: i64,
    ) {
        self.map_pos_calc(map);

        // Recompute path if stale / empty.
        if !self.move_path.is_active() {
            let chassis = self.chassis as usize;
            let start = match logic::place_find_near(
                map, objs, chassis, ROBOT_MOVECELLS_PER_SIZE,
                self.map_x, self.map_y, 4, None,
            ) {
                Some((x, y)) => map_trace::MovePt::new(x, y),
                None => { self.stop_moving(); return; }
            };
            let goal = match logic::place_find_near(
                map, objs, chassis, ROBOT_MOVECELLS_PER_SIZE,
                self.des_x, self.des_y, 4, None,
            ) {
                Some((x, y)) => map_trace::MovePt::new(x, y),
                None => { self.stop_moving(); return; }
            };
            // Port of the `other_des` / `other_path_list` arguments
            // the C++ passes to `FindLocalPath` (MatrixRobot.cpp:1658-
            // 1664): collect every other live robot's current
            // position + their MoveTo destination as blockers so A*
            // routes around them instead of planning a path that
            // immediately collides. `objs.iter_live()` doesn't
            // include self (taken out of the arena by the
            // take-the-box tick pattern).
            let mut blockers: Vec<map_trace::MovePt> = Vec::new();
            for id in objs.iter_live() {
                let Some(other_obj) = objs.get(id) else { continue };
                if !matches!(other_obj.core().obj_type, ObjectType::RobotAi) { continue; }
                let other: &Robot = unsafe {
                    &*(other_obj as *const dyn MapStatic as *const Robot)
                };
                let (omx, omy) = map.world_to_move(other.pos_x, other.pos_y);
                blockers.push(map_trace::MovePt::new(
                    omx - ROBOT_FOOTPRINT_HALF, omy - ROBOT_FOOTPRINT_HALF,
                ));
                if let Some((dx, dy)) = other.move_to_coords() {
                    blockers.push(map_trace::MovePt::new(dx, dy));
                }
                if let Some((dx, dy)) = other.return_coords() {
                    blockers.push(map_trace::MovePt::new(dx, dy));
                }
            }

            let raw = map_trace::find_path(map, start, goal, chassis, &blockers);
            let Some(raw) = raw else {
                self.stop_moving();
                return;
            };
            let opt = map_trace::optimize_path(map, &raw, chassis, &blockers);
            self.move_path.total_len = map_trace::path_total_length(&opt);
            self.move_path.followed_len = 0.0;
            self.move_path.pts = opt;
            self.move_path.cur = 0;
            self.move_test_pos = glam::Vec2::new(self.pos_x, self.pos_y);
            self.move_test_change_ms = elapsed_ms;
        }

        self.move_by_move_path(cms, map, elapsed_ms);
    }

    /// Port of `CMatrixRobotAI::MoveByMovePath(ms)`
    /// (MatrixRobot.cpp:1708-1764). Drives one LogicTakt slice of a
    /// cell-level waypoint follow: seek toward `pts[cur+1]`, advance
    /// `cur` when the projected progress along the segment exceeds
    /// the segment length, stop when the last waypoint is reached.
    ///
    /// We use a simplified Seek (no rotation, no slope/water
    /// correction, no collision) — the LowLevelMove full port lands
    /// with Phase 3. Robot-robot separation stays in effect.
    fn move_by_move_path(&mut self, cms: i32, map: &GameMap, elapsed_ms: i64) {
        let Some((sou_pt, des_pt)) = self.move_path.current_segment() else {
            self.stop_moving();
            return;
        };
        let (sou_x, sou_y) = map_trace::waypoint_to_world(sou_pt);
        let (des_x, des_y) = map_trace::waypoint_to_world(des_pt);

        // Seek-equivalent velocity: forward direction = seg dir,
        // magnitude = maxSpeed. LowLevelMove integration:
        // `pos += velocity * ms / LOGIC_TAKT_PERIOD`.
        let seg = glam::Vec2::new(des_x - sou_x, des_y - sou_y);
        let seg_len = seg.length().max(1e-3);
        let _ = seg / seg_len; // keep local `dir` intent visible
        let max_speed = chassis_max_speed(self.chassis);
        let sync_mul = (cms as f32) / 10.0;
        let last_seg = self.move_path.cur + 1 == self.move_path.pts.len() - 1;

        // Port of `Seek` (MatrixRobot.cpp:2394-2458). Structure:
        //   1. `RotateRobot(dest)` — rotates m_Forward one tick
        //      toward dest at `m_maxRotationSpeed` rad/10ms. Returns
        //      true when the turn completes this tick + snaps to
        //      dest direction. Fills `rangle` (pre-rotation angle)
        //      for the end-of-path taper.
        //   2. Two velocity branches:
        //      a. `destLength < min_speed` → velocity = destDir * k
        //         (direct hop to dest — prevents overshoot).
        //      b. else → velocity = forward * m_Speed, tapered by
        //         `t = min(1, destLength/20) * min(1, 1-rangle/π)`
        //         on the final segment (end_path).
        let dest = glam::Vec2::new(des_x, des_y);
        let (_aligned, rangle) = self.rotate_robot(cms, dest);

        let dest_dir = glam::Vec2::new(des_x - self.pos_x, des_y - self.pos_y);
        let dest_length = dest_dir.length();
        let (vel, speed) = if dest_length - max_speed < 0.001 {
            (dest_dir, dest_length)
        } else {
            let mut speed = max_speed;
            if last_seg {
                let mut t = (dest_length / 20.0).min(1.0);
                t *= (1.0 - rangle / std::f32::consts::PI).min(1.0);
                speed *= t;
            }
            (self.forward * speed, speed)
        };
        self.pos_x += vel.x * sync_mul;
        self.pos_y += vel.y * sync_mul;
        self.velocity = vel;
        self.speed = speed;

        // Follow-length accounting (MatrixRobot.cpp:2617).
        self.move_path.followed_len +=
            (vel.length() * sync_mul).abs();

        // Segment-advance test. Port of MatrixRobot.cpp:1725-1750.
        let last_seg = self.move_path.cur + 1 == self.move_path.pts.len() - 1;
        let me = glam::Vec2::new(self.pos_x - sou_x, self.pos_y - sou_y);
        let proj = me.dot(seg) / seg_len;
        let reached = if last_seg {
            (glam::Vec2::new(self.pos_x - des_x, self.pos_y - des_y)).length_squared() < 0.2
        } else {
            proj >= seg_len
        };
        if reached {
            self.move_path.cur += 1;
            if self.move_path.cur + 1 >= self.move_path.pts.len() {
                // Reached final waypoint — snap and stop.
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

        // Clamp pos_z to terrain floor during movement so the robot
        // rides the heightmap after leaving the spawn pad. Port of
        // MatrixObjectRobot.cpp:416 (land branch).
        self.pos_z = map.get_z(self.pos_x, self.pos_y);
        self.core.geo_center.x = self.pos_x;
        self.core.geo_center.y = self.pos_y;
        self.core.geo_center.z = self.pos_z + 3.0;
        self.rchange |= MR_MATRIX;
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

impl MapStatic for Robot {
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
        // Snapshot pre-tick position so we can revert if the final
        // position ends up on an impassable cell for this chassis
        // (port of the `obst_coll=true` rejection in
        // `CMatrixRobotAI::LowLevelMove` at MatrixRobot.cpp:2587 +
        // `SphereRobotToAABBObstacleCollision` at :2718). The C++
        // version computes a per-corner correction vector; we
        // approximate with a "try it, revert on fail" guard that
        // prevents the robot from ever standing on a tile its
        // chassis can't occupy.
        let pre_x = self.pos_x;
        let pre_y = self.pos_y;

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
                let b: &Building = unsafe {
                    &*(obj as *const dyn MapStatic as *const Building)
                };
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
                // Port of MatrixRobot.cpp:785-811. `LowLevelMove(ms,
                // m_Forward * 100, true, false)` at :787 delegates
                // to `Seek` (:2394), which for ROBOT_BASE_MOVEOUT
                // short-circuits rotation (:2398) and sets
                // `m_Velocity = m_Forward * m_maxSpeed` at :2456.
                // Then LowLevelMove integrates
                // `m_PosX += m_Velocity * m_SyncMul` (:2619) where
                // `m_SyncMul = ms / LOGIC_TAKT_PERIOD` (:386,
                // LOGIC_TAKT_PERIOD = 10).
                //
                // AABB obstacle avoidance (WallAvoid / SphereToAABB)
                // is still deferred — needs per-cell CMatrixMapMove
                // tables; the spawn-pad cells are clear anyway.
                const BASE_DIST: f32 = 70.0;
                const LOGIC_TAKT_PERIOD: f32 = 10.0;

                let chassis_idx = self.chassis as usize;
                let max_speed = crate::matrix_game::config::global()
                    .chassis.move_speed[chassis_idx];
                let sync_mul = (cms as f32) / LOGIC_TAKT_PERIOD;
                let vel = self.forward * max_speed;
                self.pos_x += vel.x * sync_mul;
                self.pos_y += vel.y * sync_mul;
                // Seek normally sets these; BASE_MOVEOUT skips Seek
                // but the do_animation tail still reads `speed` to
                // pick MOVE vs STAY, so keep them in sync.
                self.velocity = vel;
                self.speed = max_speed;

                // Port of RobotToObjectCollision's separation branch
                // (MatrixRobot.cpp:2905-3003). For every live robot
                // within 2R (= 36) of our projected position, add a
                // correction equal to `(2R - dist) * 0.5` along the
                // outward normal. Each robot ticks independently so
                // the total per-pair separation integrates to `2R -
                // dist` over the two frames — matching the original.
                // We skip the ColsWeight bookkeeping (used by the AI
                // to issue `MoveReturn` orders); that lands with the
                // full order / pathfinding port.
                let sep = robot_separation(self.pos_x, self.pos_y, objs);
                self.pos_x += sep.x;
                self.pos_y += sep.y;

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
                let b: &Building = unsafe {
                    &*(obj as *const dyn MapStatic as *const Building)
                };
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
                    self.pos_z = b.build_z + (1.0 - b.base_floor_progress) * BASE_FLOOR_Z - 3.0 + 2.7;
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
                            let b_mut: &mut Building = unsafe {
                                &mut *(obj_mut as *mut dyn MapStatic as *mut Building)
                            };
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
                    self.dispatch_move_to(cms, map, &*objs, elapsed_ms);
                    // Apply robot-robot separation after move.
                    let sep = robot_separation(self.pos_x, self.pos_y, objs);
                    self.pos_x += sep.x;
                    self.pos_y += sep.y;
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

        // Revert gate — approximation of `SphereRobotToAABBObstacle
        // Collision` (MatrixRobot.cpp:2718) which in the original
        // is enabled via `obst_coll=true` only for MOVE_TO paths
        // (MatrixRobot.cpp:1723 `MoveByMovePath` calls LowLevelMove
        // with both flags true). BASE_MOVEOUT explicitly disables
        // it (`LowLevelMove(ms, forward*100, true, false)` at :787)
        // because the robot starts on the base's own cell, which
        // is impassable to the chassis by design. Applying the
        // gate there would pin the robot to the pad forever.
        let in_move_order = matches!(self.state, RobotState::Idle)
            && self.orders.has(OrderType::MoveTo);
        if in_move_order {
            let chassis_idx = self.chassis as usize;
            let (mx, my) = map.world_to_move(self.pos_x, self.pos_y);
            let corner_x = mx - ROBOT_FOOTPRINT_HALF;
            let corner_y = my - ROBOT_FOOTPRINT_HALF;
            // Only revert if pre-tick pos WAS passable — otherwise
            // (i.e. robot started the tick already on an impassable
            // cell, e.g. right after BASE_MOVEOUT handoff) we'd
            // trap the robot forever.
            let (pmx, pmy) = map.world_to_move(pre_x, pre_y);
            let pre_passable = logic::is_absence_wall(
                map, chassis_idx, ROBOT_MOVECELLS_PER_SIZE,
                pmx - ROBOT_FOOTPRINT_HALF, pmy - ROBOT_FOOTPRINT_HALF,
            );
            let now_passable = logic::is_absence_wall(
                map, chassis_idx, ROBOT_MOVECELLS_PER_SIZE, corner_x, corner_y,
            );
            if pre_passable && !now_passable {
                self.pos_x = pre_x;
                self.pos_y = pre_y;
                self.core.geo_center.x = self.pos_x;
                self.core.geo_center.y = self.pos_y;
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

    fn side(&self) -> i32 { self.side }
    fn need_repair(&self) -> bool { self.hit_point < self.hit_point_max }
}

/// Port of the robot-robot separation core of `CollisionCallback`
/// (MatrixRobot.cpp:2892-3003). Scans every live robot in the arena
/// (the C++ narrows with `FindObjects` + spatial hash; we scan
/// linearly since the robot count is small) and returns the total
/// position-space correction vector: for each other robot within
/// `2R`, push `(2R - dist) * 0.5` outward along the connecting
/// vector.
///
/// The self robot is already checked out of the arena by the
/// `proceed_logic` take-the-box pattern, so `objs.iter_live()`
/// naturally skips it.
fn robot_separation(self_x: f32, self_y: f32, objs: &Objects) -> glam::Vec2 {
    const COLLIDE_BOT_R: f32 = 18.0;
    const COLLIDE_BOT_2R: f32 = COLLIDE_BOT_R + COLLIDE_BOT_R;

    let my_pos = glam::Vec2::new(self_x, self_y);
    let mut result = glam::Vec2::ZERO;
    for id in objs.iter_live() {
        let Some(obj) = objs.get(id) else { continue };
        if !matches!(obj.core().obj_type, ObjectType::RobotAi) { continue; }
        let other: &Robot = unsafe { &*(obj as *const dyn MapStatic as *const Robot) };
        let v = my_pos - glam::Vec2::new(other.pos_x, other.pos_y);
        let dist = v.length();
        if dist < COLLIDE_BOT_2R && dist > 1.0e-3 {
            let correction = (COLLIDE_BOT_2R - dist) * 0.5;
            result += (v / dist) * correction;
        }
    }
    result
}
