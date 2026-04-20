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

/// MatrixCamera.hpp defines `CAM_HFOV = 60°`, but the original feeds that
/// value to `D3DXMatrixPerspectiveFovLH`, which takes a vertical FOV.
const CAM_FOV: f32 = 60.0 * std::f32::consts::PI / 180.0;

/// MAX_VIEW_DISTANCE from MatrixCamera.cpp:13. Used both as the render far
/// plane and the cap for water/ocean frustum projection (MatrixVisiCalc.cpp:
/// 560-604). Keeping the far plane at this limit matches the original and
/// avoids iterating water tiles that would be fully fogged anyway.
pub const MAX_VIEW_DISTANCE: f32 = 4000.0;

/// Strategy camera parameters (MatrixConfig.cpp:226-249).
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

const STRATEGY_PARAMS: CamParam = CamParam {
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

const MOUSE_EDGE: f32 = 4.0; // MatrixFormGame.hpp:9 (MOUSE_BORDER)

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
        let p = &STRATEGY_PARAMS;
        Self {
            angle_z: 0.0,
            angle_x: lerp_ang(p.angle_param),
            dist: lerp_dist(1.0),
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
            sample_terrain: None,
            sample_ground: None,
        }
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
        self.strategy_init_angle = angle;
        self.ang_strategy = angle;
        self.angle_z = angle;
    }

    pub fn set_xy_strategy(&mut self, pos: [f32; 2]) {
        self.xy_strategy = pos;
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
        let p = &STRATEGY_PARAMS;
        self.ang_strategy += p.rot_speed_z * dx * 10.0;
        self.angle_param -= p.rot_speed_x * dy * 5.0;
        self.angle_param = self.angle_param.clamp(0.0, 1.0);
    }

    /// Ports ZoomInStep/OutStep (MatrixCamera.hpp:275-289). `notches` is an
    /// integer-ish wheel delta (positive = zoom in, shrinks distance).
    pub fn zoom(&mut self, notches: f32) {
        self.dist_param -= STRATEGY_PARAMS.wheel_step * notches;
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

    pub fn on_key(&mut self, key: KeyAction, pressed: bool) {
        let p = &STRATEGY_PARAMS;
        if let KeyAction::ResetAngles = key {
            if pressed {
                self.ang_strategy = self.strategy_init_angle;
                self.angle_param = p.angle_param;
                self.angle_z = self.strategy_init_angle;
                self.angle_x = lerp_ang(p.angle_param);
                self.dist = lerp_dist(self.dist_param);
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
        let p = &STRATEGY_PARAMS;

        self.link_point.x = self.xy_strategy[0] - self.map_cx;
        self.link_point.y = self.xy_strategy[1] - self.map_cy;
        self.angle_z = self.ang_strategy;
        self.angle_x = lerp_ang(self.angle_param);
        self.dist = lerp_dist(self.dist_param);
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
            self.ang_strategy -= p.rot_speed_z * dt_ms;
        }
        if self.actions & ACT_ROT_RIGHT != 0 {
            self.ang_strategy += p.rot_speed_z * dt_ms;
        }
        if self.actions & ACT_ROT_UP != 0 {
            self.angle_param = (self.angle_param + p.rot_speed_x * dt_ms).min(1.0);
        }
        if self.actions & ACT_ROT_DOWN != 0 {
            self.angle_param = (self.angle_param - p.rot_speed_x * dt_ms).max(0.0);
        }

        // Keyboard + edge-pan (MatrixCamera.cpp:1062-1086) uses the bottom
        // frustum plane normal projected onto XY.
        if move_bits != 0 {
            let speed = p.move_speed * dt_ms;
            let dir = self.bottom_plane_xy_dir() * speed;
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
        // Place the camera on the +Y side of the link point at angle_z=0,
        // matching the original's matView construction. Solving `p * matView
        // = 0` for `matView = Rz(-yaw) * Rx(pitch) * T(0,0,-dist)` (with the
        // Y+Z column negations from MatrixCamera.cpp:742-745) gives eye =
        // (lp.x - sin(yaw)*cos(pitch)*dist, lp.y + cos(yaw)*cos(pitch)*dist,
        // lp.z + sin(pitch)*dist). The previous formula had the X and Y
        // offsets negated, putting the camera on the opposite side of the
        // link point — invisible on symmetric maps (atoll), but on training
        // it appeared as a left-right mirror since every asymmetric feature
        // showed from the wrong side.
        let mut eye = Vec3::new(
            lp.x - sz * cx * self.dist,
            lp.y + cz * cx * self.dist,
            lp.z + sx * self.dist,
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

    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.eye_pos();
        let target = self.link_point;
        Mat4::look_at_rh(eye, target, Vec3::Z)
    }

    pub fn view_proj(&self) -> Mat4 {
        let eye = self.eye_pos();
        let target = self.link_point;

        // Z-up (x,y,z) → Y-up (x,z,-y) for glam's look_at_rh.
        let eye_yup = Vec3::new(eye.x, eye.z, -eye.y);
        let target_yup = Vec3::new(target.x, target.z, -target.y);
        let view = Mat4::look_at_rh(eye_yup, target_yup, Vec3::Y);

        let proj = Mat4::perspective_rh(CAM_FOV, self.aspect, self.near, self.far);

        let z_to_y = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        proj * view * z_to_y
    }

    pub fn forward(&self) -> Vec3 {
        (self.link_point - self.eye_pos()).normalize_or_zero()
    }

    fn bottom_plane_xy_dir(&self) -> Vec2 {
        let eye = self.eye_pos();
        let target = self.link_point;
        let eye_yup = Vec3::new(eye.x, eye.z, -eye.y);
        let target_yup = Vec3::new(target.x, target.z, -target.y);
        let view = Mat4::look_at_rh(eye_yup, target_yup, Vec3::Y);
        let proj = Mat4::perspective_rh(CAM_FOV, self.aspect, self.near, self.far);
        let inv_vp = (proj * view).inverse();

        let sample_corner = |sx: f32| {
            let near4 = inv_vp * glam::Vec4::new(sx, -1.0, 0.0, 1.0);
            let far4 = inv_vp * glam::Vec4::new(sx, -1.0, 1.0, 1.0);
            let near = (near4 / near4.w).truncate();
            let far = (far4 / far4.w).truncate();
            far - near
        };
        let lb = sample_corner(-1.0);
        let rb = sample_corner(1.0);
        let n = lb.cross(rb).normalize_or_zero();
        let xy = Vec2::new(n.x, -n.z);
        if xy.length_squared() < 1e-8 {
            let s = self.angle_z.sin();
            let c = self.angle_z.cos();
            // Negated from the old (-s, c) to match the new eye orientation —
            // at angle_z=0 the camera now sits at +Y and looks -Y, so the
            // bottom-frustum projection should point -Y.
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

        let bottom_norm_z = lb.cross(rb).normalize_or_zero().z;
        if bottom_norm_z > 0.0 {
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

fn lerp_ang(param: f32) -> f32 {
    let p = &STRATEGY_PARAMS;
    p.rot_angle_min + (p.rot_angle_max - p.rot_angle_min) * param
}
fn lerp_dist(param: f32) -> f32 {
    let p = &STRATEGY_PARAMS;
    p.dist_min + (p.dist_max - p.dist_min) * param
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
