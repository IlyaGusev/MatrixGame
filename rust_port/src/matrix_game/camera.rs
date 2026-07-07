//! Camera — port of CMatrixCamera (MatrixCamera.cpp/hpp), strategy +
//! in-robot (arcade) modes.
//!
//! Key behaviors (from the original) implemented here:
//!   - Link point follows terrain z + fixed height (MatrixCamera.cpp:924-926,
//!     CalcLinkPoint).
//!   - Keyboard pan uses the horizontal forward direction derived from
//!     (angle_z, angle_x) — same as the original's frustum-bottom-plane
//!     normal projected to XY (MatrixCamera.cpp:1071-1086).
//!   - Keyboard pan updates link_point immediately (CAM_XY_LERP_OFF,
//!     MatrixCamera.cpp:1033-1040); no smoothing on XY during pan.
//!   - Middle-mouse drag rotates (MouseCam mode, MatrixFormGame.cpp:531, 636).
//!   - Edge-pan: cursor within 4 px of screen edges triggers move actions
//!     (MatrixFormGame.hpp:9 + MatrixFormGame.cpp:232-240).
//!   - Mouse-wheel zoom steps `dist_param` by ±0.225, clamped [0.25, 4.0]
//!     (MatrixCamera.hpp:275-289).
//!   - PageUp/PageDown adjust pitch (`angle_param` 0..1).
//!   - Backslash (`\`) resets angles (MatrixFormGame.cpp:1285).
//!   - Arcade mode (CAMERA_INROBOT): the link point chases the arcaded
//!     robot with a speed-lerped forward offset and the yaw locks to
//!     the robot heading (CalcLinkPoint, MatrixCamera.cpp:895-931);
//!     mode changes ease with `1 - 0.995^ms`.
//!
//! Behaviors deliberately omitted here: fly-cam (MMFLAG_FLYCAM).

use glam::{Mat4, Vec2, Vec3};

use crate::matrix_lib::base::storage::Storage;

/// MatrixCamera.hpp defines `CAM_HFOV = 60°`, but the original feeds that
/// value to `D3DXMatrixPerspectiveFovLH`, which takes a vertical FOV.
const CAM_FOV: f32 = 60.0 * std::f32::consts::PI / 180.0;

/// MAX_VIEW_DISTANCE from MatrixCamera.cpp:13. Used both as the render far
/// plane and the cap for water/ocean frustum projection (MatrixVisiCalc.cpp:
/// 560-604). Keeping the far plane at this limit matches the original and
/// avoids iterating water tiles that would be fully fogged anyway.
pub const MAX_VIEW_DISTANCE: f32 = 4000.0;

/// Strategy camera parameters (MatrixConfig.cpp:226-249).
///
/// These are the defaults the original hard-codes in `CMatrixConfig::SetDefaults`.
/// `apply_camera_config` overwrites any field that appears under
/// `Camera/Strategy` in `robots.dat` — matches MatrixConfig.cpp:1011-1050.
#[derive(Clone, Copy)]
struct CamParam {
    rot_speed_x: f32,   // pitch speed (rad/ms)
    rot_speed_z: f32,   // yaw speed (rad/ms)
    rot_angle_min: f32, // pitch at param=0 (most top-down)
    rot_angle_max: f32, // pitch at param=1 (most level)
    dist_min: f32,
    dist_max: f32,
    angle_param: f32, // default pitch parameter
    height: f32,      // link-point height above terrain
    wheel_step: f32,  // raw CamMouseWheelStep; zoom() scales it by 4.5
    move_speed: f32,  // pan speed (world units/ms)
}

const STRATEGY_DEFAULTS: CamParam = CamParam {
    rot_speed_x: 0.0005,
    rot_speed_z: 0.001,
    rot_angle_min: 60.0 * std::f32::consts::PI / 180.0,
    rot_angle_max: 20.0 * std::f32::consts::PI / 180.0, // original (MatrixConfig.cpp)
    dist_min: 70.0,
    dist_max: 250.0,
    angle_param: 0.4,
    height: 140.0,
    wheel_step: 0.05,
    move_speed: 1.05,
};

/// `CAMERA_INROBOT` defaults (MatrixConfig.cpp:236-244) — arcade mode.
/// Only `angle_param` (0.0 → most level) and `height` (40) differ.
const INROBOT_DEFAULTS: CamParam = CamParam {
    angle_param: 0.0,
    height: 40.0,
    ..STRATEGY_DEFAULTS
};

/// Camera mode indices — `CAMERA_STRATEGY` / `CAMERA_INROBOT`
/// (MatrixConfig.hpp:537-541).
const MODE_STRATEGY: usize = 0;
const MODE_INROBOT: usize = 1;

/// Per-frame arcade camera target — the slice of the arcaded robot's
/// state `CalcLinkPoint` reads (MatrixCamera.cpp:899-915).
#[derive(Clone, Copy, Debug)]
pub struct ArcadeCamTarget {
    /// Robot position, **uncentered** world coords, z = `Z_From_Pos()`.
    pub pos: Vec3,
    /// `m_Forward` (chassis heading).
    pub forward: Vec2,
    /// `m_Speed / GetMaxSpeed()` — lerps the chase offset 10..30.
    pub speed_ratio: f32,
}

// MatrixFormGame.hpp:9 sets MOUSE_BORDER=4. Keep a slightly larger margin
// to tolerate browser DPR rounding, but small enough to stay out of the way.
const MOUSE_EDGE: f32 = 8.0;

pub struct Camera {
    // Current (rendered) state — equals the target state except while
    // CAM_LINK_POINT_CHANGED easing runs (mode changes / arcade follow).
    angle_z: f32,
    angle_x: f32,
    dist: f32,
    link_point: Vec3,

    // Target state (user-controlled).
    xy_strategy: [f32; 2], // in uncentered world coords
    ang_strategy: f32,
    // Per-mode pitch / zoom params — `m_AngleParam[CAMERA_PARAM_CNT]`
    // / `m_DistParam[...]` (MatrixCamera.hpp:186-187).
    angle_param: [f32; 2],
    dist_param: [f32; 2],

    pub aspect: f32,
    pub near: f32,
    pub far: f32,

    actions: u32,
    mouse_cam: bool, // middle-button drag active
    last_mouse: [f32; 2],
    cursor: [f32; 2],
    screen: [f32; 2],

    map_cx: f32,
    map_cy: f32,
    strategy_init_angle: f32,

    // Per-mode params, seeded from the defaults and optionally patched
    // by apply_camera_config (robots.dat `Camera/Strategy` +
    // `Camera/InRobot` blocks).
    params: [CamParam; 2],
    /// `m_ModeIndex` — MODE_STRATEGY / MODE_INROBOT.
    mode_index: usize,
    /// `CAM_LINK_POINT_CHANGED` — ease current state toward the target
    /// with `1 - 0.995^ms` instead of copying (MatrixCamera.cpp:1010-1048).
    link_point_changed: bool,
    /// `CAM_XY_LERP_OFF` — while easing, snap the link point XY
    /// (keyboard pan / SetXYStrategy set this).
    xy_lerp_off: bool,
    /// `m_CamInRobotForward0/1` (MatrixConfig.cpp:248-249) — chase
    /// offset along the robot forward, lerped by speed ratio.
    in_robot_fwd: [f32; 2],
    /// The arcaded robot's follow target for this frame; `Some` puts
    /// the camera in MODE_INROBOT.
    arcade_target: Option<ArcadeCamTarget>,
    /// `g_Config.m_CamBaseAngleZ` — yaw offset added on top of the map's
    /// `m_CameraAngle` when entering strategy mode (MatrixCamera.cpp:696).
    base_angle_z: f32,

    // Terrain sampler: uncentered (x, y) world coords → terrain z.
    sample_terrain: Option<Box<dyn Fn(f32, f32) -> f32 + Send + Sync>>,
    sample_ground: Option<Box<dyn Fn(f32, f32) -> f32 + Send + Sync>>,
}

const ACT_MOVE_LEFT: u32 = 1 << 0;
const ACT_MOVE_RIGHT: u32 = 1 << 1;
const ACT_MOVE_FWD: u32 = 1 << 2;
const ACT_MOVE_BACK: u32 = 1 << 3;
const ACT_ROT_LEFT: u32 = 1 << 4;
const ACT_ROT_RIGHT: u32 = 1 << 5;
const ACT_ROT_UP: u32 = 1 << 6;
const ACT_ROT_DOWN: u32 = 1 << 7;

impl Camera {
    pub fn new(aspect: f32) -> Self {
        let p = STRATEGY_DEFAULTS;
        Self {
            angle_z: 0.0,
            angle_x: lerp_ang(&p, p.angle_param),
            dist: lerp_dist(&p, 1.0),
            link_point: Vec3::ZERO,
            xy_strategy: [0.0, 0.0],
            ang_strategy: 0.0,
            angle_param: [STRATEGY_DEFAULTS.angle_param, INROBOT_DEFAULTS.angle_param],
            dist_param: [1.0, 1.0],
            aspect,
            near: 1.0,
            far: MAX_VIEW_DISTANCE,
            actions: 0,
            mouse_cam: false,
            last_mouse: [0.0, 0.0],
            cursor: [-1.0, -1.0],
            screen: [1.0, 1.0],
            map_cx: 0.0,
            map_cy: 0.0,
            strategy_init_angle: 0.0,
            params: [STRATEGY_DEFAULTS, INROBOT_DEFAULTS],
            mode_index: MODE_STRATEGY,
            link_point_changed: false,
            xy_lerp_off: false,
            in_robot_fwd: [10.0, 30.0],
            arcade_target: None,
            base_angle_z: 0.0,
            sample_terrain: None,
            sample_ground: None,
        }
    }

    /// Port of MatrixConfig.cpp:1011-1050. Overwrites any camera param found
    /// in `robots.dat` → `Camera` / `Camera/Strategy`; missing keys keep their
    /// hardcoded defaults. Call before `init_strategy_angle` so the base yaw
    /// offset is picked up when the map's camera angle is applied.
    pub fn apply_camera_config(&mut self, matrix_data: &Storage) {
        let Some(cam_rec) = matrix_data.block_record("da", "Camera") else {
            return;
        };

        // Top-level Camera params (MatrixConfig.cpp:1017-1022).
        if let Some(v) = parse_f32(matrix_data.block_param(&cam_rec, "CamBaseAngleZ")) {
            self.base_angle_z = v.to_radians();
        }
        if let Some(v) = parse_f32(matrix_data.block_param(&cam_rec, "CamMoveSpeed")) {
            for p in &mut self.params {
                p.move_speed = v;
            }
        }
        if let Some(v) = parse_f32(matrix_data.block_param(&cam_rec, "CamInRobotForward0")) {
            self.in_robot_fwd[0] = v;
        }
        if let Some(v) = parse_f32(matrix_data.block_param(&cam_rec, "CamInRobotForward1")) {
            self.in_robot_fwd[1] = v;
        }

        // Per-mode sub-blocks (MatrixConfig.cpp:1024-1044). Angle keys
        // are in degrees; RotAngleMin is capped at 94° in the original
        // to prevent the pitch going past straight-up.
        for (block, idx) in [("Strategy", MODE_STRATEGY), ("InRobot", MODE_INROBOT)] {
            let Some(rec) = matrix_data.block_record(&cam_rec, block) else {
                continue;
            };
            let p = &mut self.params[idx];
            if let Some(v) = parse_f32(matrix_data.block_param(&rec, "CamRotSpeedX")) {
                p.rot_speed_x = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&rec, "CamRotSpeedZ")) {
                p.rot_speed_z = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&rec, "CamMouseWheelStep")) {
                p.wheel_step = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&rec, "CamRotAngleMin")) {
                p.rot_angle_min = v.min(94.0).to_radians();
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&rec, "CamRotAngleMax")) {
                p.rot_angle_max = v.to_radians();
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&rec, "CamDistMin")) {
                p.dist_min = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&rec, "CamDistMax")) {
                p.dist_max = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&rec, "CamAngleParam")) {
                p.angle_param = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&rec, "CamHeight")) {
                p.height = v;
            }
        }

        // Re-seed state derived from the params: default pitch param, default
        // zoom, and the clamped working values.
        for i in 0..2 {
            self.angle_param[i] = self.params[i].angle_param;
            self.dist_param[i] = 1.0;
        }
        self.angle_x = lerp_ang(&self.params[0], self.angle_param[0]);
        self.dist = lerp_dist(&self.params[0], self.dist_param[0]);
    }

    pub fn set_map(&mut self, world_width: f32, world_height: f32) {
        self.map_cx = world_width * 0.5;
        self.map_cy = world_height * 0.5;
        self.xy_strategy = [self.map_cx, self.map_cy];
        // Far plane tracks the original MAX_VIEW_DISTANCE; fog fades to full
        // opacity by 2800 so nothing beyond 4000 would be visually meaningful.
        self.far = MAX_VIEW_DISTANCE;
    }

    pub fn set_aspect(&mut self, width: f32, height: f32) {
        self.aspect = width / height;
        self.screen = [width, height];
    }

    pub fn init_strategy_angle(&mut self, angle: f32) {
        // MatrixCamera.cpp:696 — strategy yaw is `CamBaseAngleZ + map.CameraAngle`.
        let combined = angle_norm(self.base_angle_z + angle);
        self.strategy_init_angle = combined;
        self.ang_strategy = combined;
        self.angle_z = combined;
    }

    pub fn set_xy_strategy(&mut self, pos: [f32; 2]) {
        // `SetXYStrategy` sets CAM_XY_LERP_OFF (MatrixCamera.hpp:262).
        self.xy_lerp_off = true;
        self.xy_strategy = pos;
    }

    /// Feed (or clear) the arcade follow target for this frame. `Some`
    /// switches the camera to `CAMERA_INROBOT`.
    pub fn set_arcade_target(&mut self, t: Option<ArcadeCamTarget>) {
        self.arcade_target = t;
    }

    /// Uncentered world XY of the strategy link point — equivalent to
    /// `CMatrixCamera::GetXYStrategy()` (MatrixCamera.hpp). Used by the
    /// minimap to recenter every frame per MatrixMap.cpp:1261.
    pub fn strategy_xy(&self) -> (f32, f32) {
        (self.xy_strategy[0], self.xy_strategy[1])
    }

    pub fn set_terrain_sampler(&mut self, f: Box<dyn Fn(f32, f32) -> f32 + Send + Sync>) {
        self.sample_terrain = Some(f);
    }

    pub fn set_ground_sampler(&mut self, f: Box<dyn Fn(f32, f32) -> f32 + Send + Sync>) {
        self.sample_ground = Some(f);
    }

    // ── Input handlers ────────────────────────────────────────────────────

    /// Ports RotateByMouse (MatrixCamera.hpp:238-248). Yaw only turns
    /// in strategy mode; pitch works in both (per-mode param).
    pub fn rotate_by_mouse(&mut self, dx: f32, dy: f32) {
        let m = self.mode_index;
        let p = &self.params[m];
        if m == MODE_STRATEGY {
            self.ang_strategy = angle_norm(self.ang_strategy + p.rot_speed_z * dx * 10.0);
        }
        self.angle_param[m] = (self.angle_param[m] - p.rot_speed_x * dy * 5.0).clamp(0.0, 1.0);
    }

    /// Ports ZoomInStep/OutStep (MatrixCamera.hpp:275-289). `notches` is an
    /// integer-ish wheel delta (positive = zoom in, shrinks distance).
    /// The original steps by `CamMouseWheelStep * 4.5` (0.05 * 4.5 = 0.225).
    pub fn zoom(&mut self, notches: f32) {
        let m = self.mode_index;
        self.dist_param[m] =
            (self.dist_param[m] - self.params[m].wheel_step * 4.5 * notches).clamp(0.25, 4.0);
    }

    /// Enter/exit MouseCam rotate mode. The original binds this to the middle
    /// mouse button.
    /// Middle-drag rotate active (`IsMouseCam`).
    pub fn is_mouse_cam(&self) -> bool {
        self.mouse_cam
    }

    /// Last cursor X seen by [`Self::on_mouse_move`] — the mouse-cam
    /// robot steer reads the pre-update value to compute its dx.
    pub fn last_mouse_x(&self) -> f32 {
        self.last_mouse[0]
    }

    pub fn on_rotate_button(&mut self, pressed: bool, x: f32, y: f32) {
        self.mouse_cam = pressed;
        self.last_mouse = [x, y];
    }

    pub fn on_mouse_move(&mut self, x: f32, y: f32) {
        self.cursor = [x, y];
        if self.mouse_cam {
            let dx = x - self.last_mouse[0];
            let dy = y - self.last_mouse[1];
            self.rotate_by_mouse(dx, dy);
        }
        self.last_mouse = [x, y];
    }

    pub fn on_mouse_wheel(&mut self, notches: f32) {
        self.zoom(notches);
    }

    /// Mark the cursor as outside the window so edge-pan stops firing.
    /// The existing in-bounds guard in `takt` then skips the pan.
    pub fn on_cursor_left(&mut self) {
        self.cursor = [-1.0, -1.0];
    }

    pub fn on_key(&mut self, key: KeyAction, pressed: bool) {
        if let KeyAction::ResetAngles = key {
            // `ResetAngles` (MatrixCamera.cpp:693-705) — restore the
            // per-mode pitch defaults; the yaw only snaps back in
            // strategy mode (arcade yaw is robot-driven).
            if pressed {
                self.ang_strategy = self.strategy_init_angle;
                for i in 0..2 {
                    self.angle_param[i] = self.params[i].angle_param;
                }
                let m = self.mode_index;
                self.angle_x = lerp_ang(&self.params[m], self.angle_param[m]);
                self.dist = lerp_dist(&self.params[m], self.dist_param[m]);
                if m == MODE_STRATEGY {
                    self.angle_z = self.strategy_init_angle;
                }
            }
            return;
        }
        let bit = match key {
            KeyAction::MoveLeft => ACT_MOVE_LEFT,
            KeyAction::MoveRight => ACT_MOVE_RIGHT,
            KeyAction::MoveForward => ACT_MOVE_FWD,
            KeyAction::MoveBack => ACT_MOVE_BACK,
            KeyAction::RotLeft => ACT_ROT_LEFT,
            KeyAction::RotRight => ACT_ROT_RIGHT,
            KeyAction::RotUp => ACT_ROT_UP,
            KeyAction::RotDown => ACT_ROT_DOWN,
            KeyAction::ResetAngles => return,
        };
        if pressed {
            self.actions |= bit;
        } else {
            self.actions &= !bit;
        }
    }

    // ── Per-frame update ──────────────────────────────────────────────────

    /// Ports CMatrixCamera::Takt (MatrixCamera.cpp:933-1130) —
    /// strategy + in-robot (arcade) modes.
    pub fn takt(&mut self, dt_ms: f32) {
        // Mode pick + transition (MatrixCamera.cpp:969-1000). On any
        // change the camera eases (CAM_LINK_POINT_CHANGED); returning
        // to strategy re-seats the target 80 units behind the current
        // link point along the strategy yaw.
        let index = if self.arcade_target.is_some() {
            MODE_INROBOT
        } else {
            MODE_STRATEGY
        };
        if index != self.mode_index {
            self.link_point_changed = true;
            self.xy_lerp_off = false;
            self.mode_index = index;
            if index == MODE_STRATEGY {
                let sn = self.ang_strategy.sin();
                let cs = self.ang_strategy.cos();
                self.xy_strategy = [
                    self.link_point.x + self.map_cx - sn * 80.0,
                    self.link_point.y + self.map_cy + cs * 80.0,
                ];
            }
        }

        // Target link point + yaw — CalcLinkPoint (MatrixCamera.cpp:
        // 895-931). link_point is stored map-centered.
        let (newlp, newangz) = match (self.mode_index, self.arcade_target) {
            (MODE_INROBOT, Some(t)) => {
                let p = &self.params[MODE_INROBOT];
                let fwd_off = lerp_f(
                    t.speed_ratio,
                    self.in_robot_fwd[0],
                    self.in_robot_fwd[1],
                );
                let lp = Vec3::new(
                    t.pos.x - self.map_cx + t.forward.x * fwd_off,
                    t.pos.y - self.map_cy + t.forward.y * fwd_off,
                    t.pos.z + p.height,
                );
                (lp, angle_norm((-t.forward.x).atan2(t.forward.y) + std::f32::consts::PI))
            }
            _ => {
                let p = &self.params[MODE_STRATEGY];
                let bias = self
                    .sample_terrain
                    .as_ref()
                    .map(|f| f(self.xy_strategy[0], self.xy_strategy[1]))
                    .unwrap_or(0.0);
                (
                    Vec3::new(
                        self.xy_strategy[0] - self.map_cx,
                        self.xy_strategy[1] - self.map_cy,
                        p.height + bias,
                    ),
                    self.ang_strategy,
                )
            }
        };
        let m = self.mode_index;
        let newdist = lerp_dist(&self.params[m], self.dist_param[m]);
        let newangx = lerp_ang(&self.params[m], self.angle_param[m]);

        if self.link_point_changed {
            // Ease toward the target (MatrixCamera.cpp:1010-1048),
            // `mul = 1 - 0.995^ms`.
            let mul = 1.0 - 0.995_f32.powf(dt_ms);
            let mut dlp = newlp - self.link_point;
            let daz = angle_dist(self.angle_z, newangz);
            let dd = newdist - self.dist;
            let dax = angle_dist(self.angle_x, newangx);
            self.link_point += dlp * mul;
            self.angle_z = angle_norm(self.angle_z + daz * mul);
            self.dist += dd * mul;
            self.angle_x += dax * mul;
            if self.xy_lerp_off {
                dlp.x = 0.0;
                dlp.y = 0.0;
                self.link_point.x = newlp.x;
                self.link_point.y = newlp.y;
            }
            let half_deg = 0.5_f32.to_radians();
            if m == MODE_STRATEGY
                && dlp.length_squared() < 0.25
                && daz.abs() < half_deg
                && dax.abs() < half_deg
                && dd.abs() < 0.5
            {
                self.link_point_changed = false;
            }
        } else {
            self.link_point = newlp;
            self.angle_z = newangz;
            self.dist = newdist;
            self.angle_x = newangx;
        }

        let p = self.params[m];

        // Combine keyboard-held pan bits with per-frame edge-pan, without
        // touching self.actions (clearing it would wipe the keyboard hold and
        // stall movement between OS key-repeat events — the "laggy WASD" bug).
        // Pan + edge-scroll are strategy-only (MatrixCamera.cpp:1066,
        // MatrixFormGame.cpp:232-241).
        let mut move_bits = if m == MODE_STRATEGY {
            self.actions & (ACT_MOVE_LEFT | ACT_MOVE_RIGHT | ACT_MOVE_FWD | ACT_MOVE_BACK)
        } else {
            0
        };
        let [cx, cy] = self.cursor;
        let [w, h] = self.screen;
        if m == MODE_STRATEGY && cx >= 0.0 && cy >= 0.0 && cx < w && cy < h {
            if cx < MOUSE_EDGE {
                move_bits |= ACT_MOVE_LEFT;
            }
            if cx > w - MOUSE_EDGE {
                move_bits |= ACT_MOVE_RIGHT;
            }
            if cy < MOUSE_EDGE {
                move_bits |= ACT_MOVE_FWD;
            }
            if cy > h - MOUSE_EDGE {
                move_bits |= ACT_MOVE_BACK;
            }
        }

        // Yaw/pitch (MatrixCamera.cpp:1103-1127). Yaw keys are
        // strategy-only (the C++ asserts strategy mode); the pitch
        // keys drive the per-mode angle param in both modes. These
        // bits are cleared on key release via on_key, so we leave
        // self.actions untouched here.
        if m == MODE_STRATEGY {
            if self.actions & ACT_ROT_LEFT != 0 {
                self.ang_strategy = angle_norm(self.ang_strategy - p.rot_speed_z * dt_ms);
            }
            if self.actions & ACT_ROT_RIGHT != 0 {
                self.ang_strategy = angle_norm(self.ang_strategy + p.rot_speed_z * dt_ms);
            }
        }
        if self.actions & ACT_ROT_UP != 0 {
            self.angle_param[m] = (self.angle_param[m] + p.rot_speed_x * dt_ms).min(1.0);
        }
        if self.actions & ACT_ROT_DOWN != 0 {
            self.angle_param[m] = (self.angle_param[m] - p.rot_speed_x * dt_ms).max(0.0);
        }
        self.ang_strategy = angle_norm(self.ang_strategy);
        self.angle_z = angle_norm(self.angle_z);

        // Keyboard + edge-pan (MatrixCamera.cpp:1062-1086) uses the bottom
        // frustum plane normal projected onto XY.
        if move_bits != 0 {
            // Pan snaps the link-point XY while easing (SETFLAG
            // CAM_XY_LERP_OFF at MatrixCamera.cpp:1069).
            self.xy_lerp_off = true;
            let speed = p.move_speed * dt_ms;
            let dir = self.bottom_plane_xy_dir() * speed;
            // lDir/rDir match MatrixCamera.cpp:1076-1077 (left = 90° CW
            // rotation of `dir`; right = 90° CCW).
            let ldir = Vec2::new(dir.y, -dir.x);
            let rdir = Vec2::new(-dir.y, dir.x);
            let tdir = dir;
            let bdir = -dir;
            if move_bits & ACT_MOVE_LEFT != 0 {
                self.xy_strategy[0] += ldir.x;
                self.xy_strategy[1] += ldir.y;
            }
            if move_bits & ACT_MOVE_RIGHT != 0 {
                self.xy_strategy[0] += rdir.x;
                self.xy_strategy[1] += rdir.y;
            }
            if move_bits & ACT_MOVE_FWD != 0 {
                self.xy_strategy[0] += tdir.x;
                self.xy_strategy[1] += tdir.y;
            }
            if move_bits & ACT_MOVE_BACK != 0 {
                self.xy_strategy[0] += bdir.x;
                self.xy_strategy[1] += bdir.y;
            }
        }

        let to_center = Vec2::new(
            self.map_cx - self.xy_strategy[0],
            self.map_cy - self.xy_strategy[1],
        );
        let r = to_center.length();
        let rlim = 3.0 * self.map_cx.max(self.map_cy);
        if r > rlim && r > 0.0 {
            let corr = to_center / r * (r - rlim);
            self.xy_strategy[0] += corr.x;
            self.xy_strategy[1] += corr.y;
        }
    }

    // ── Derived transforms ────────────────────────────────────────────────

    fn eye_unclamped(&self) -> Vec3 {
        let lp = self.link_point;
        let sz = self.angle_z.sin();
        let cz = self.angle_z.cos();
        let sx = self.angle_x.sin();
        let cx = self.angle_x.cos();
        // Port of the original's matView inverse. From MatrixCamera.cpp:737-745
        // the view matrix is T(-LP) * Rz(-AZ) * Rx(AX) * T(0,0,-Dist) * S
        // (row-major D3D) with S = diag(1,-1,-1,1). Solving `eye * matView = 0`
        // places the camera at:
        //     eye = LP + (-Dist*sin(AX)*sin(AZ),
        //                  Dist*sin(AX)*cos(AZ),
        //                  Dist*cos(AX))
        // So sin(AX) is the horizontal offset and cos(AX) is the vertical —
        // AX = 0 → top-down view, AX = π/2 → level view. The atoll map's
        // `CamRotAngleMin=100` (clamped to 94°) is a near-level pitch in
        // this convention. The previous formula had sin/cos swapped, which
        // treated AX as elevation and pushed the eye past the zenith at
        // high AX, producing the left/right flip and the look_at singularity.
        Vec3::new(
            lp.x - sz * sx * self.dist,
            lp.y + cz * sx * self.dist,
            lp.z + cx * self.dist,
        )
    }

    /// Ground-clamp lift applied to the eye Z — port of the BeforeDraw clamp
    /// at MatrixCamera.cpp:748-755 (`m_MatViewInversed._43 < GetZ(...) + 10`).
    /// Returns how far the eye must be raised (0 when no clamp engages).
    fn ground_lift(&self, eye: Vec3) -> f32 {
        if let Some(sample_ground) = &self.sample_ground {
            let wx = eye.x + self.map_cx;
            let wy = eye.y + self.map_cy;
            let min_z = sample_ground(wx, wy) + 10.0;
            if eye.z < min_z {
                return min_z - eye.z;
            }
        }
        0.0
    }

    pub fn eye_pos(&self) -> Vec3 {
        let mut eye = self.eye_unclamped();
        eye.z += self.ground_lift(eye);
        eye
    }

    /// Direct port of the view matrix construction in MatrixCamera.cpp:737-745.
    /// Row-major D3D: `matView = T(-LP) * Rz(-AZ) * Rx(AX) * T(0,0,-Dist) * S`
    /// with S = diag(1,-1,-1,1). Transposed into glam's column-major convention
    /// the operation order reverses: `S * T(0,0,-Dist) * Rx(AX) * Rz(-AZ) * T(-LP)`.
    /// Constructing the matrix explicitly (rather than via `look_at_lh`) avoids
    /// the near-zenith singularity when the camera is nearly top-down.
    ///
    /// The terrain ground clamp (MatrixCamera.cpp:748-755) is applied here too:
    /// the original lifts `m_MatViewInversed._43` and re-inverts into m_MatView,
    /// so the *rendered* view is lifted. Lifting the eye by `dz` while keeping
    /// the rotation equals post-multiplying by `T(0, 0, -dz)`.
    pub fn view_matrix(&self) -> Mat4 {
        let s = Mat4::from_scale(Vec3::new(1.0, -1.0, -1.0));
        let t2 = Mat4::from_translation(Vec3::new(0.0, 0.0, -self.dist));
        let rx = Mat4::from_rotation_x(self.angle_x);
        let rz = Mat4::from_rotation_z(-self.angle_z);
        let t1 = Mat4::from_translation(-self.link_point);
        let view = s * t2 * rx * rz * t1;
        let lift = self.ground_lift(self.eye_unclamped());
        if lift > 0.0 {
            view * Mat4::from_translation(Vec3::new(0.0, 0.0, -lift))
        } else {
            view
        }
    }

    /// Map center X the camera subtracts from every world position
    /// before projection (so renderers + hit-tests operate in the
    /// same centered space). Equals `world_width / 2`.
    pub fn map_cx(&self) -> f32 {
        self.map_cx
    }
    /// Map center Y — see [`map_cx`].
    pub fn map_cy(&self) -> f32 {
        self.map_cy
    }

    pub fn view_proj(&self) -> Mat4 {
        let proj = Mat4::perspective_lh(CAM_FOV, self.aspect, self.near, self.far);
        proj * self.view_matrix()
    }

    /// Project an **uncentered** world point to screen pixels (origin
    /// top-left). `None` when the point sits behind the camera. Port
    /// of `CMatrixCamera::Project` for the HP-bar placement.
    pub fn project_to_screen(
        &self,
        world: Vec3,
        screen_w: f32,
        screen_h: f32,
    ) -> Option<glam::Vec2> {
        let centered = world - Vec3::new(self.map_cx, self.map_cy, 0.0);
        let clip = self.view_proj() * centered.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        Some(glam::Vec2::new(
            (clip.x / clip.w * 0.5 + 0.5) * screen_w,
            (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * screen_h,
        ))
    }

    /// Camera position in uncentered world coordinates — the
    /// `GetFrustumCenter()` the C++ uses as trace origin.
    pub fn eye_pos_world(&self) -> Vec3 {
        self.eye_pos() + Vec3::new(self.map_cx, self.map_cy, 0.0)
    }

    /// World-space camera basis vectors. Derived from `inverse(view)`
    /// where the view matrix sends world→view: the inverse columns
    /// are the view axes expressed in world. After the `S = diag(1,-1,-1)`
    /// sign flip in `view_matrix`, glam's `.y_axis` / `.z_axis` come
    /// out negated relative to the visible screen basis, so we flip
    /// them back here for callers that want "screen right" / "screen
    /// up" in world space (billboard orientation, selection-ring dots).
    pub fn camera_right_world(&self) -> Vec3 {
        let inv = self.view_matrix().inverse();
        inv.x_axis.truncate().normalize_or_zero()
    }
    pub fn camera_up_world(&self) -> Vec3 {
        let inv = self.view_matrix().inverse();
        // `S` flipped view Y; undo via negation so the result reads as
        // "up on the screen" in world space.
        (-inv.y_axis.truncate()).normalize_or_zero()
    }

    /// Unproject a screen-pixel (`sx`, `sy` in `[0, screen_w] × [0, screen_h]`
    /// with origin top-left) into a world-space ray. Returns
    /// `(origin, dir)` in **uncentered** world coordinates, so the
    /// result can be compared directly against a `MapStatic::core().geo_center`.
    ///
    /// Ports the idiomatic ray-cast portion of
    /// `CMatrixCamera::GetPickRay` (MatrixCamera.cpp). We go through
    /// `inverse(view_proj)` to avoid replicating the sign conventions
    /// in `view_matrix` by hand — correct by construction. The `near`
    /// point lives at NDC z=0 (wgpu / D3D LH depth), `far` at z=1.
    pub fn screen_to_world_ray(
        &self,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> (Vec3, Vec3) {
        // Screen → NDC. wgpu / D3D have X right, Y up in NDC; screen
        // Y is top-origin so we flip.
        let nx = 2.0 * sx / screen_w - 1.0;
        let ny = 1.0 - 2.0 * sy / screen_h;
        let inv_vp = self.view_proj().inverse();
        let near_w = inv_vp * glam::Vec4::new(nx, ny, 0.0, 1.0);
        let far_w = inv_vp * glam::Vec4::new(nx, ny, 1.0, 1.0);
        let near = near_w.truncate() / near_w.w;
        let far = far_w.truncate() / far_w.w;
        // The camera operates in *centered* world space (view_matrix
        // translates by `-link_point`, which itself subtracts map
        // center). Shift back to uncentered so callers can hit-test
        // against MapStatic::geo_center.
        let offset = Vec3::new(self.map_cx, self.map_cy, 0.0);
        let origin = near + offset;
        let dir = (far - near).normalize_or_zero();
        (origin, dir)
    }

    pub fn forward(&self) -> Vec3 {
        (self.link_point - self.eye_pos()).normalize_or_zero()
    }

    fn bottom_plane_xy_dir(&self) -> Vec2 {
        // Match MatrixCamera.cpp:791: bottom plane inward normal = LB × RB.
        let [_, _, rb, lb] = self.frustum_corner_dirs_zup();
        let n = lb.cross(rb).normalize_or_zero();
        let xy = Vec2::new(n.x, n.y);
        if xy.length_squared() < 1e-8 {
            let s = self.angle_z.sin();
            let c = self.angle_z.cos();
            Vec2::new(s, -c)
        } else {
            xy.normalize()
        }
    }

    /// Projects each frustum corner ray onto the horizontal plane z = `plane_z`.
    /// Ports the visibility quad construction in MatrixVisiCalc.cpp:560-640.
    /// The key horizon-case behavior is that top rays which do not hit the
    /// water plane are extended by `MAX_VIEW_DISTANCE` along the true frustum
    /// ray, not by projecting a horizontal ray from the near plane.
    pub fn frustum_bounds_on_plane_zup(&self, plane_z: f32) -> [Vec3; 4] {
        let center = self.eye_pos();
        let [lt, rt, rb, lb] = self.frustum_corner_dirs_zup();

        let project = |dir: Vec3, fallback_max_dist: bool| {
            let mut k = if fallback_max_dist || dir.z >= -0.001 {
                MAX_VIEW_DISTANCE
            } else {
                ((plane_z - center.z) / dir.z).min(MAX_VIEW_DISTANCE)
            };
            if !k.is_finite() {
                k = MAX_VIEW_DISTANCE;
            }
            Vec3::new(center.x + dir.x * k, center.y + dir.y * k, plane_z)
        };

        let top_forward = lt + rt;
        let camera_direction_2d = Vec2::new(top_forward.x, top_forward.y).length();
        let k_cam = if camera_direction_2d > 1e-5 {
            top_forward.z.abs() / camera_direction_2d
        } else {
            f32::INFINITY
        };
        let k_etalon = center.z / MAX_VIEW_DISTANCE;
        let horizon_case = top_forward.z > 0.0 || k_cam < k_etalon;

        let mut out = [
            project(lt, horizon_case),
            project(rt, horizon_case),
            project(rb, false),
            project(lb, false),
        ];

        // MatrixVisiCalc.cpp:602-604 always applies the disp in the horizon
        // branch, and MatrixVisiCalc.cpp:651-656 applies it in the non-horizon
        // branch only when FrustPlaneB.norm.z > 0, with
        // FrustPlaneB.norm = LB × RB (MatrixCamera.cpp:791). The previous
        // `rb × lb` had the opposite sign, so disp fired at steep angles
        // (cutting off the bottom of the screen) and not at near-level ones.
        let apply_disp = if horizon_case {
            true
        } else {
            lb.cross(rb).normalize_or_zero().z > 0.0
        };
        if apply_disp {
            let bottom_mid = (out[2] + out[3]) * 0.5;
            let disp = Vec2::new(center.x - bottom_mid.x, center.y - bottom_mid.y);
            out[2].x += disp.x;
            out[2].y += disp.y;
            out[3].x += disp.x;
            out[3].y += disp.y;
        }

        out
    }

    fn frustum_corner_dirs_zup(&self) -> [Vec3; 4] {
        let eye = self.eye_pos();
        let inv_vp = self.view_proj().inverse();
        let corners = [(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)];
        corners.map(|(sx, sy)| {
            let far4 = inv_vp * glam::Vec4::new(sx, sy, 1.0, 1.0);
            let far = (far4 / far4.w).truncate();
            (far - eye).normalize_or_zero()
        })
    }

    /// Extract the Left/Right/Top/Bottom frustum plane equations (Ax+By+Cz+D=0
    /// form, points inside satisfy `Ax+By+Cz+D >= 0`) from `view_proj` using
    /// the standard Gribb-Hartmann trick. Near/Far are omitted — matches the
    /// original's SPlane set which drops those planes too (MatrixCamera.hpp:
    /// 413-420, `IsInFrustum(mins,maxs)` only tests L/R/T/B).
    pub fn frustum_planes(&self) -> [[f32; 4]; 4] {
        let m = self.view_proj().to_cols_array();
        // Row-vectors of the matrix (glam stores column-major, so m[col*4+row]).
        let row = |r: usize| [m[r], m[4 + r], m[8 + r], m[12 + r]];
        let r0 = row(0);
        let r1 = row(1);
        let r3 = row(3);
        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        let normalize = |p: [f32; 4]| -> [f32; 4] {
            let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-8);
            [p[0] / n, p[1] / n, p[2] / n, p[3] / n]
        };
        [
            normalize(add(r3, r0)), // Left:  r3 + r0
            normalize(sub(r3, r0)), // Right: r3 - r0
            normalize(sub(r3, r1)), // Top:   r3 - r1
            normalize(add(r3, r1)), // Bottom:r3 + r1
        ]
    }

    /// 3D AABB vs camera frustum. Returns true if the box is not entirely
    /// outside any of the 4 side planes. Matches CMatrixCamera::IsInFrustum
    /// (MatrixCamera.hpp:413) which also skips near/far planes.
    pub fn is_box_in_frustum(&self, min: Vec3, max: Vec3) -> bool {
        for plane in self.frustum_planes() {
            // Pick the box corner farthest in the plane's positive direction.
            let px = if plane[0] >= 0.0 { max.x } else { min.x };
            let py = if plane[1] >= 0.0 { max.y } else { min.y };
            let pz = if plane[2] >= 0.0 { max.z } else { min.z };
            if plane[0] * px + plane[1] * py + plane[2] * pz + plane[3] < 0.0 {
                return false;
            }
        }
        true
    }
}

fn lerp_ang(p: &CamParam, param: f32) -> f32 {
    p.rot_angle_min + (p.rot_angle_max - p.rot_angle_min) * param
}
/// `LERPFLOAT(k, c1, c2)` (Math3D.hpp:22).
fn lerp_f(k: f32, c1: f32, c2: f32) -> f32 {
    c1 + (c2 - c1) * k
}
fn lerp_dist(p: &CamParam, param: f32) -> f32 {
    p.dist_min + (p.dist_max - p.dist_min) * param
}

fn angle_norm(a: f32) -> f32 {
    a.rem_euclid(std::f32::consts::TAU)
}

fn angle_dist(from: f32, to: f32) -> f32 {
    let d = angle_norm(to) - angle_norm(from);
    if d > std::f32::consts::PI {
        d - std::f32::consts::TAU
    } else if d < -std::f32::consts::PI {
        d + std::f32::consts::TAU
    } else {
        d
    }
}

/// Parse a numeric robots.dat parameter. The original uses `GetDouble()`,
/// which accepts the same numeric grammar Rust's `parse::<f32>` does; any
/// blank/missing value falls through to the hardcoded default.
fn parse_f32(raw: Option<String>) -> Option<f32> {
    raw.and_then(|s| s.trim().parse::<f32>().ok())
}

#[derive(Clone, Copy)]
pub enum KeyAction {
    MoveLeft,
    MoveRight,
    MoveForward,
    MoveBack,
    RotLeft,
    RotRight,
    RotUp,
    RotDown,
    ResetAngles,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_strategy_angle_is_normalized() {
        let mut cam = Camera::new(1.0);
        cam.base_angle_z = std::f32::consts::PI * 3.0;
        cam.init_strategy_angle(std::f32::consts::PI);
        assert!(cam.ang_strategy >= 0.0);
        assert!(cam.ang_strategy < std::f32::consts::TAU);
        assert_eq!(cam.ang_strategy, cam.angle_z);
    }

    #[test]
    fn angle_dist_takes_shortest_path() {
        let eps = 1e-5;
        let a = std::f32::consts::TAU - 0.1;
        let b = 0.1;
        assert!((angle_dist(a, b) - 0.2).abs() < eps);
        assert!((angle_dist(b, a) + 0.2).abs() < eps);
    }

    /// `project_to_screen` must be the exact inverse of
    /// `screen_to_world_ray` — a world point projected to the screen,
    /// re-cast as a ray, must pass through that point.
    #[test]
    fn project_roundtrips_with_pick_ray() {
        let (w, h) = (1920.0f32, 1080.0f32);
        let mut cam = Camera::new(w / h);
        cam.set_map(640.0, 640.0);
        cam.set_xy_strategy([300.0, 280.0]);
        // Settle the interpolators.
        for _ in 0..200 {
            cam.takt(16.0);
        }
        for world in [
            Vec3::new(300.0, 300.0, 5.0),
            Vec3::new(250.0, 350.0, 30.0),
            Vec3::new(380.0, 260.0, 0.0),
        ] {
            let p = cam
                .project_to_screen(world, w, h)
                .expect("point in front of camera");
            let (origin, dir) = cam.screen_to_world_ray(p.x, p.y, w, h);
            // Distance from the ray to the world point.
            let d = (world - origin).cross(dir).length();
            assert!(
                d < 0.05,
                "ray misses projected point by {d} (screen {p:?}, world {world:?})"
            );
        }
    }
}
