//! Minimal port of `CMatrixRobotAI` (MatrixRobot.{cpp,hpp}).
//!
//! CMatrixRobotAI is ~5000 lines (AI, pathfinding, state machine,
//! chassis/armor/weapon composition, animation, combat). This file
//! ports only the subset needed for a faithful build-robot flow:
//! place the robot at a spawn position with a side / chassis kind,
//! join the arena, and carry enough state that FindObjects +
//! selection queries do the right thing. AI and combat land with
//! their owning subsystems.

use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectId, Objects, ObjectType, MR_ALL, MR_MATRIX,
};
use crate::matrix_game::rnd::Rnd;

/// Port of `ERobotState` (MatrixRobot.hpp). Full enum covers 20+
/// states (DIP, MOVE, PATROL, ATTACK, CARRYING, etc.); we only
/// model the spawn-animation pipeline for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RobotState {
    /// `ROBOT_IN_SPAWN` — base is rising, robot is attached to the
    /// platform and moves up with it (MatrixRobot.cpp:2227).
    InSpawn,
    /// `ROBOT_BASE_MOVEOUT` — platform has fully risen; robot is
    /// driving off the pad. We simplify to a timer since the AI
    /// movement path isn't ported (MatrixRobot.cpp:785).
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
    /// Time remaining in `BaseMoveOut`. Ports the distance-based
    /// threshold at MatrixRobot.cpp:797 with a simpler timer — the
    /// original moves the robot forward at 100 units/ms until it's
    /// `BASE_DIST` (≈50) units from the base, which takes ~500ms.
    pub moveout_timer_ms: i32,
    /// Base build_z at spawn time — anchor for the platform-rise
    /// animation. Updated when base reassigns.
    pub base_anchor_z: f32,
    /// Chassis height delta stored so Takt can restore pos_z after
    /// the platform animation.
    pub chassis_dz: f32,
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
            moveout_timer_ms: 0,
            base_anchor_z: pos.z,
            chassis_dz: chassis.spawn_z_offset(),
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
    pub fn robot_spawn(&mut self, base_id: ObjectId, base_build_z: f32) {
        self.base = Some(base_id);
        self.state = RobotState::InSpawn;
        self.base_anchor_z = base_build_z;
        // Sit flush with the (fully-down) platform initially. As the
        // base opens, `takt` lifts pos_z by base_floor_progress *
        // BASE_FLOOR_RISE_Z so the robot appears to rise out.
        self.pos_z = base_build_z;
    }
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
    fn logic_takt(&mut self, cms: i32, _rng: &mut Rnd, objs: &mut Objects) {
        use crate::matrix_game::object_building::{BaseState, Building, BASE_FLOOR_Z};

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
                    self.moveout_timer_ms = 500;
                }
            }
            RobotState::BaseMoveOut => {
                // MatrixRobot.cpp:785-811. The original moves the
                // robot forward by `LowLevelMove(ms, forward * 100)`
                // and checks `D3DXVec2LengthSq(dist)` against
                // BASE_DIST². We don't have pathfinding; simulate
                // with a 500ms timer after which we force-close the
                // base and declare the spawn done.
                self.moveout_timer_ms -= cms;
                if self.moveout_timer_ms <= 0 {
                    if let Some(base_id) = self.base {
                        if let Some(obj) = objs.get_mut(base_id) {
                            if matches!(obj.core().obj_type, ObjectType::Building) {
                                let b_mut: &mut Building = unsafe {
                                    &mut *(obj as *mut dyn MapStatic as *mut Building)
                                };
                                // MatrixRobot.cpp:809-811.
                                b_mut.object_state_clear(
                                    crate::matrix_game::map_static::OBJECT_STATE_BUILDING_SPAWNBOT,
                                );
                                b_mut.close();
                            }
                        }
                    }
                    self.base = None;
                    self.state = RobotState::Idle;
                }
            }
            RobotState::Idle => {
                // Full CMatrixRobotAI AI is deferred — robot just
                // stands on the spawn pad.
            }
        }
    }

    fn side(&self) -> i32 { self.side }
    fn need_repair(&self) -> bool { self.hit_point < self.hit_point_max }
}
