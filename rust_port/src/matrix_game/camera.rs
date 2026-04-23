//! Camera — port of CMatrixCamera (MatrixCamera.cpp/hpp) in strategy mode.
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
//!
//! Behaviors deliberately omitted here: arcade (in-robot) mode and fly-cam.

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
    wheel_step: f32,  // `dist_param` delta per mouse-wheel notch
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
    wheel_step: 0.225,
    move_speed: 1.05,
};

// MatrixFormGame.hpp:9 sets MOUSE_BORDER=4. Keep a slightly larger margin
// to tolerate browser DPR rounding, but small enough to stay out of the way.
const MOUSE_EDGE: f32 = 8.0;

pub struct Camera {
    // Current (rendered) state — for strategy mode this equals the target state;
    // smoothing only kicks in on mode changes (not implemented, no arcade mode).
    angle_z: f32,
    angle_x: f32,
    dist: f32,
    link_point: Vec3,

    // Target state (user-controlled).
    xy_strategy: [f32; 2], // in uncentered world coords
    ang_strategy: f32,
    angle_param: f32,
    dist_param: f32,

    // Projection.
    pub aspect: f32,
    pub near: f32,
    pub far: f32,

    // Input state.
    actions: u32,
    mouse_cam: bool, // middle-button drag active
    last_mouse: [f32; 2],
    cursor: [f32; 2],
    screen: [f32; 2],

    // Map bounds.
    map_cx: f32,
    map_cy: f32,
    strategy_init_angle: f32,

    // Mode + global params, seeded from STRATEGY_DEFAULTS and optionally
    // patched by apply_camera_config (robots.dat `Camera/Strategy` block).
    params: CamParam,
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
            angle_param: p.angle_param,
            dist_param: 1.0,
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
            params: p,
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

        // Top-level Camera params. We only care about the two the strategy
        // camera actually consumes — InRobotForward0/1 are arcade-mode only.
        if let Some(v) = parse_f32(matrix_data.block_param(&cam_rec, "CamBaseAngleZ")) {
            self.base_angle_z = v.to_radians();
        }
        if let Some(v) = parse_f32(matrix_data.block_param(&cam_rec, "CamMoveSpeed")) {
            self.params.move_speed = v;
        }

        // Strategy sub-block. Angle keys are in degrees; RotAngleMin is capped
        // at 94° in the original to prevent the pitch going past straight-up.
        if let Some(strat_rec) = matrix_data.block_record(&cam_rec, "Strategy") {
            if let Some(v) = parse_f32(matrix_data.block_param(&strat_rec, "CamRotSpeedX")) {
                self.params.rot_speed_x = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&strat_rec, "CamRotSpeedZ")) {
                self.params.rot_speed_z = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&strat_rec, "CamMouseWheelStep")) {
                self.params.wheel_step = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&strat_rec, "CamRotAngleMin")) {
                self.params.rot_angle_min = v.min(94.0).to_radians();
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&strat_rec, "CamRotAngleMax")) {
                self.params.rot_angle_max = v.to_radians();
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&strat_rec, "CamDistMin")) {
                self.params.dist_min = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&strat_rec, "CamDistMax")) {
                self.params.dist_max = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&strat_rec, "CamAngleParam")) {
                self.params.angle_param = v;
            }
            if let Some(v) = parse_f32(matrix_data.block_param(&strat_rec, "CamHeight")) {
                self.params.height = v;
            }
        }

        // Re-seed state derived from the params: default pitch param, default
        // zoom, and the clamped working values.
        self.angle_param = self.params.angle_param;
        self.dist_param = 1.0;
        self.angle_x = lerp_ang(&self.params, self.angle_param);
        self.dist = lerp_dist(&self.params, self.dist_param);
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
        self.xy_strategy = pos;
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

    /// Ports RotateByMouse (MatrixCamera.hpp:238-248).
    pub fn rotate_by_mouse(&mut self, dx: f32, dy: f32) {
        let p = &self.params;
        self.ang_strategy = angle_norm(self.ang_strategy + p.rot_speed_z * dx * 10.0);
        self.angle_param -= p.rot_speed_x * dy * 5.0;
        self.angle_param = self.angle_param.clamp(0.0, 1.0);
    }

    /// Ports ZoomInStep/OutStep (MatrixCamera.hpp:275-289). `notches` is an
    /// integer-ish wheel delta (positive = zoom in, shrinks distance).
    pub fn zoom(&mut self, notches: f32) {
        self.dist_param -= self.params.wheel_step * notches;
        self.dist_param = self.dist_param.clamp(0.25, 4.0);
    }

    /// Enter/exit MouseCam rotate mode. The original binds this to the middle
    /// mouse button.
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
        let p = &self.params;
        if let KeyAction::ResetAngles = key {
            if pressed {
                self.ang_strategy = self.strategy_init_angle;
                self.angle_param = p.angle_param;
                self.angle_z = self.strategy_init_angle;
                self.angle_x = lerp_ang(p, p.angle_param);
                self.dist = lerp_dist(p, self.dist_param);
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

    /// Ports CMatrixCamera::Takt (MatrixCamera.cpp:933-1130) for strategy mode.
    pub fn takt(&mut self, dt_ms: f32) {
        let p = &self.params;

        self.link_point.x = self.xy_strategy[0] - self.map_cx;
        self.link_point.y = self.xy_strategy[1] - self.map_cy;
        self.angle_z = self.ang_strategy;
        self.angle_x = lerp_ang(p, self.angle_param);
        self.dist = lerp_dist(p, self.dist_param);
        let bias = self
            .sample_terrain
            .as_ref()
            .map(|f| f(self.xy_strategy[0], self.xy_strategy[1]))
            .unwrap_or(0.0);
        self.link_point.z = p.height + bias;

        // Combine keyboard-held pan bits with per-frame edge-pan, without
        // touching self.actions (clearing it would wipe the keyboard hold and
        // stall movement between OS key-repeat events — the "laggy WASD" bug).
        let mut move_bits =
            self.actions & (ACT_MOVE_LEFT | ACT_MOVE_RIGHT | ACT_MOVE_FWD | ACT_MOVE_BACK);
        let [cx, cy] = self.cursor;
        let [w, h] = self.screen;
        if cx >= 0.0 && cy >= 0.0 && cx < w && cy < h {
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

        // Yaw/pitch (MatrixCamera.cpp:1103-1127). These bits are cleared on
        // key release via on_key, so we leave self.actions untouched here.
        if self.actions & ACT_ROT_LEFT != 0 {
            self.ang_strategy = angle_norm(self.ang_strategy - p.rot_speed_z * dt_ms);
        }
        if self.actions & ACT_ROT_RIGHT != 0 {
            self.ang_strategy = angle_norm(self.ang_strategy + p.rot_speed_z * dt_ms);
        }
        if self.actions & ACT_ROT_UP != 0 {
            self.angle_param = (self.angle_param + p.rot_speed_x * dt_ms).min(1.0);
        }
        if self.actions & ACT_ROT_DOWN != 0 {
            self.angle_param = (self.angle_param - p.rot_speed_x * dt_ms).max(0.0);
        }
        self.ang_strategy = angle_norm(self.ang_strategy);
        self.angle_z = angle_norm(self.angle_z);

        // Keyboard + edge-pan (MatrixCamera.cpp:1062-1086) uses the bottom
        // frustum plane normal projected onto XY.
        if move_bits != 0 {
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

    pub fn eye_pos(&self) -> Vec3 {
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
        let mut eye = Vec3::new(
            lp.x - sz * sx * self.dist,
            lp.y + cz * sx * self.dist,
            lp.z + cx * self.dist,
        );
        if let Some(sample_ground) = &self.sample_ground {
            let wx = eye.x + self.map_cx;
            let wy = eye.y + self.map_cy;
            let min_z = sample_ground(wx, wy) + 10.0;
            if eye.z < min_z {
                eye.z = min_z;
            }
        }
        eye
    }

    /// Direct port of the view matrix construction in MatrixCamera.cpp:737-745.
    /// Row-major D3D: `matView = T(-LP) * Rz(-AZ) * Rx(AX) * T(0,0,-Dist) * S`
    /// with S = diag(1,-1,-1,1). Transposed into glam's column-major convention
    /// the operation order reverses: `S * T(0,0,-Dist) * Rx(AX) * Rz(-AZ) * T(-LP)`.
    /// Constructing the matrix explicitly (rather than via `look_at_lh`) avoids
    /// the near-zenith singularity when the camera is nearly top-down.
    pub fn view_matrix(&self) -> Mat4 {
        let s = Mat4::from_scale(Vec3::new(1.0, -1.0, -1.0));
        let t2 = Mat4::from_translation(Vec3::new(0.0, 0.0, -self.dist));
        let rx = Mat4::from_rotation_x(self.angle_x);
        let rz = Mat4::from_rotation_z(-self.angle_z);
        let t1 = Mat4::from_translation(-self.link_point);
        s * t2 * rx * rz * t1
    }

    /// Map center X the camera subtracts from every world position
    /// before projection (so renderers + hit-tests operate in the
    /// same centered space). Equals `world_width / 2`.
    pub fn map_cx(&self) -> f32 { self.map_cx }
    /// Map center Y — see [`map_cx`].
    pub fn map_cy(&self) -> f32 { self.map_cy }

    pub fn view_proj(&self) -> Mat4 {
        let proj = Mat4::perspective_lh(CAM_FOV, self.aspect, self.near, self.far);
        proj * self.view_matrix()
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
        let far_w  = inv_vp * glam::Vec4::new(nx, ny, 1.0, 1.0);
        let near = near_w.truncate() / near_w.w;
        let far  = far_w.truncate()  / far_w.w;
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
fn lerp_dist(p: &CamParam, param: f32) -> f32 {
    p.dist_min + (p.dist_max - p.dist_min) * param
}

fn angle_norm(a: f32) -> f32 {
    a.rem_euclid(std::f32::consts::TAU)
}

#[allow(dead_code)]
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
}
