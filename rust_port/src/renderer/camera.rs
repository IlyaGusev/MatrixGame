//! Camera — faithful port of CMatrixCamera from MatrixCamera.cpp/hpp.
//!
//! Original file mapping:
//!   MatrixCamera.hpp  → camera state, RotateByMouse, ZoomIn/OutStep
//!   MatrixCamera.cpp  → BeforeDraw (view matrix), Takt (update), CalcLinkPoint
//!   MatrixConfig.cpp  → SCamParam defaults
//!   MatrixFormGame.cpp → mouse/keyboard input routing

use glam::{Mat4, Vec3};

// Original: GRAD2RAD(60) = PI/3
const CAM_HFOV: f32 = std::f32::consts::PI / 3.0;

/// Camera parameters per mode (ports SCamParam from MatrixConfig.cpp).
struct CamParam {
    mouse_wheel_step: f32,
    rot_speed_x: f32,      // pitch speed (rad/ms per pixel)
    rot_speed_z: f32,      // yaw speed (rad/ms per pixel)
    rot_angle_min: f32,    // pitch min (radians) — more top-down
    rot_angle_max: f32,    // pitch max (radians) — more level
    dist_min: f32,
    dist_max: f32,
    angle_param: f32,      // default pitch interpolation (0.0-1.0)
    height: f32,           // camera height above ground
}

/// Default strategy camera params (from MatrixConfig.cpp lines 226-249).
const STRATEGY_PARAMS: CamParam = CamParam {
    mouse_wheel_step: 0.05,
    rot_speed_x: 0.0005,
    rot_speed_z: 0.001,
    rot_angle_min: 60.0 * std::f32::consts::PI / 180.0,  // 60 degrees
    rot_angle_max: 20.0 * std::f32::consts::PI / 180.0,  // 20 degrees
    dist_min: 70.0,
    dist_max: 250.0,
    angle_param: 0.4,
    height: 140.0,
};

/// Camera move speed (original: m_CamMoveSpeed = 1.05).
const CAM_MOVE_SPEED: f32 = 1.05;

pub struct Camera {
    // ── CMatrixCamera fields ──
    /// Link point: world position camera orbits around (original: m_LinkPoint)
    link_point: Vec3,
    /// Horizontal rotation angle (original: m_AngleZ)
    angle_z: f32,
    /// Vertical rotation angle (original: m_AngleX, computed from angle_param)
    angle_x: f32,
    /// Distance from camera to link point (original: m_Dist, computed from dist_param)
    dist: f32,
    /// Normalized pitch parameter 0.0-1.0 (original: m_AngleParam[CAMERA_STRATEGY])
    angle_param: f32,
    /// Normalized distance parameter 0.25-4.0 (original: m_DistParam[CAMERA_STRATEGY])
    dist_param: f32,
    /// Strategy mode XY position (original: m_XY_Strategy)
    xy_strategy: [f32; 2],
    /// Strategy mode angle Z (original: m_Ang_Strategy)
    ang_strategy: f32,

    // ── Projection ──
    pub aspect: f32,
    pub near: f32,
    pub far: f32,

    // ── Input state ──
    actions: u32,
    mouse_dragging: bool,
    pub last_mouse: [f32; 2],

    // ── Map bounds ──
    map_cx: f32,
    map_cy: f32,
    map_half_w: f32,
    map_half_h: f32,
}

// Action flags (original: CAM_ACTION_*)
const ACT_MOVE_LEFT: u32  = 1 << 0;
const ACT_MOVE_RIGHT: u32 = 1 << 1;
const ACT_MOVE_UP: u32    = 1 << 2;
const ACT_MOVE_DOWN: u32  = 1 << 3;
const ACT_ROT_LEFT: u32   = 1 << 4;
const ACT_ROT_RIGHT: u32  = 1 << 5;

impl Camera {
    pub fn new(aspect: f32) -> Self {
        let p = &STRATEGY_PARAMS;
        Self {
            link_point: Vec3::ZERO,
            angle_z: 0.0,
            angle_x: lerp_ang(p.angle_param),
            dist: lerp_dist(1.0), // start at default zoom
            angle_param: p.angle_param,
            dist_param: 1.0,
            xy_strategy: [0.0, 0.0],
            ang_strategy: 0.0,
            aspect,
            near: 1.0,
            far: 5000.0,
            actions: 0,
            mouse_dragging: false,
            last_mouse: [0.0, 0.0],
            map_cx: 0.0,
            map_cy: 0.0,
            map_half_w: 1000.0,
            map_half_h: 1000.0,
        }
    }

    /// Set map bounds for camera panning limits.
    pub fn set_map(&mut self, world_width: f32, world_height: f32) {
        self.map_cx = world_width * 0.5;
        self.map_cy = world_height * 0.5;
        self.map_half_w = world_width * 0.5;
        self.map_half_h = world_height * 0.5;
        // Start camera at map center
        self.xy_strategy = [self.map_cx, self.map_cy];
        self.far = world_width.max(world_height) * 3.0;
    }

    pub fn set_aspect(&mut self, width: f32, height: f32) {
        self.aspect = width / height;
    }

    /// Ports CMatrixCamera::RotateByMouse (MatrixCamera.hpp lines 238-248).
    pub fn rotate_by_mouse(&mut self, dx: f32, dy: f32) {
        let p = &STRATEGY_PARAMS;
        // Yaw: m_Ang_Strategy += rot_speed_z * dx * 10
        self.ang_strategy += p.rot_speed_z * dx * 10.0;
        // Pitch: m_AngleParam -= rot_speed_x * dy * 5
        self.angle_param -= p.rot_speed_x * dy * 5.0;
        self.angle_param = self.angle_param.clamp(0.0, 1.0);
    }

    /// Ports ZoomInStep/ZoomOutStep (MatrixCamera.hpp lines 275-289).
    pub fn zoom_step(&mut self, delta: f32) {
        let p = &STRATEGY_PARAMS;
        self.dist_param -= p.mouse_wheel_step * 4.5 * delta;
        self.dist_param = self.dist_param.clamp(0.25, 4.0);
    }

    /// Handle mouse button press/release.
    pub fn on_mouse_button(&mut self, pressed: bool, x: f32, y: f32) {
        self.mouse_dragging = pressed;
        self.last_mouse = [x, y];
    }

    /// Handle mouse move.
    pub fn on_mouse_move(&mut self, x: f32, y: f32) {
        if self.mouse_dragging {
            let dx = x - self.last_mouse[0];
            let dy = y - self.last_mouse[1];
            self.rotate_by_mouse(dx, dy);
        }
        self.last_mouse = [x, y];
    }

    /// Handle mouse wheel.
    pub fn on_mouse_wheel(&mut self, delta: f32) {
        self.zoom_step(delta);
    }

    /// Handle key press/release for camera movement.
    pub fn on_key(&mut self, key: KeyAction, pressed: bool) {
        let bit = match key {
            KeyAction::MoveLeft => ACT_MOVE_LEFT,
            KeyAction::MoveRight => ACT_MOVE_RIGHT,
            KeyAction::MoveUp => ACT_MOVE_UP,
            KeyAction::MoveDown => ACT_MOVE_DOWN,
            KeyAction::RotLeft => ACT_ROT_LEFT,
            KeyAction::RotRight => ACT_ROT_RIGHT,
        };
        if pressed { self.actions |= bit; } else { self.actions &= !bit; }
    }

    /// Per-frame update — ports CMatrixCamera::Takt(float ms) lines 933-1130.
    pub fn takt(&mut self, dt_ms: f32) {
        let p = &STRATEGY_PARAMS;

        // ── Keyboard rotation (original lines 1103-1127) ──
        if self.actions & ACT_ROT_LEFT != 0 {
            self.ang_strategy -= p.rot_speed_z * dt_ms;
        }
        if self.actions & ACT_ROT_RIGHT != 0 {
            self.ang_strategy += p.rot_speed_z * dt_ms;
        }

        // ── Keyboard movement (original lines 1062-1099) ──
        // Move in the direction the camera is facing on XY plane
        let speed = CAM_MOVE_SPEED * dt_ms;
        let sin_z = self.ang_strategy.sin();
        let cos_z = self.ang_strategy.cos();

        if self.actions & ACT_MOVE_UP != 0 {
            self.xy_strategy[0] += sin_z * speed;
            self.xy_strategy[1] -= cos_z * speed;
        }
        if self.actions & ACT_MOVE_DOWN != 0 {
            self.xy_strategy[0] -= sin_z * speed;
            self.xy_strategy[1] += cos_z * speed;
        }
        if self.actions & ACT_MOVE_LEFT != 0 {
            self.xy_strategy[0] -= cos_z * speed;
            self.xy_strategy[1] -= sin_z * speed;
        }
        if self.actions & ACT_MOVE_RIGHT != 0 {
            self.xy_strategy[0] += cos_z * speed;
            self.xy_strategy[1] += sin_z * speed;
        }

        // ── Clamp to map bounds ──
        self.xy_strategy[0] = self.xy_strategy[0].clamp(0.0, self.map_cx * 2.0);
        self.xy_strategy[1] = self.xy_strategy[1].clamp(0.0, self.map_cy * 2.0);

        // ── CalcLinkPoint for strategy mode (original lines 895-931) ──
        // Link point in centered Z-up coords (terrain vertices are offset by -cx, -cy)
        let target = Vec3::new(
            self.xy_strategy[0] - self.map_cx,
            self.xy_strategy[1] - self.map_cy,
            p.height,
        );

        // ── Smooth interpolation (original: mul = 1.0 - pow(0.995, ms)) ──
        let mul = 1.0 - 0.995_f32.powf(dt_ms);
        self.link_point = self.link_point + (target - self.link_point) * mul;
        self.angle_z = self.ang_strategy; // strategy mode: direct

        // ── Compute actual angle and distance from params ──
        self.angle_x = lerp_ang(self.angle_param);
        self.dist = lerp_dist(self.dist_param);
    }

    /// Compute eye position in Z-up world coords from camera angles and link point.
    /// Ports the math from BeforeDraw (MatrixCamera.cpp).
    pub fn eye_pos(&self) -> Vec3 {
        let lp = self.link_point;
        let sin_z = self.angle_z.sin();
        let cos_z = self.angle_z.cos();
        let sin_x = self.angle_x.sin();
        let cos_x = self.angle_x.cos();

        // Camera orbits link_point: offset by angle_z (yaw) and angle_x (pitch) at distance
        // In Z-up coords: X right, Y forward, Z up
        Vec3::new(
            lp.x + sin_z * cos_x * self.dist,
            lp.y - cos_z * cos_x * self.dist,
            lp.z + sin_x * self.dist,
        )
    }

    /// View matrix in Z-up coords using look_at.
    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.eye_pos();
        let target = self.link_point;
        // Z-up: up vector is (0, 0, 1)
        Mat4::look_at_rh(eye, target, Vec3::Z)
    }

    pub fn view_proj(&self) -> Mat4 {
        // Build view in Z-up coords
        let eye = self.eye_pos();
        let target = self.link_point;

        // Convert Z-up positions to Y-up for glam's look_at_rh (which assumes Y-up)
        // Z-up (x,y,z) → Y-up (x,z,-y)
        let eye_yup = Vec3::new(eye.x, eye.z, -eye.y);
        let target_yup = Vec3::new(target.x, target.z, -target.y);
        let view = Mat4::look_at_rh(eye_yup, target_yup, Vec3::Y);

        let half_hfov = CAM_HFOV * 0.5;
        let vfov = 2.0 * (half_hfov.tan() / self.aspect).atan();
        let proj = Mat4::perspective_rh(vfov, self.aspect, self.near, self.far);

        // Vertex data is in Z-up coords. The view matrix expects Y-up.
        // Apply Z-up→Y-up conversion to vertices before the view transform:
        // (x,y,z) → (x,z,-y)
        let z_to_y = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);

        proj * view * z_to_y
    }

    /// Forward direction in Z-up coords (unit vector from eye to target).
    /// Ports GetDir() (MatrixCamera.hpp:265).
    pub fn forward(&self) -> Vec3 {
        (self.link_point - self.eye_pos()).normalize_or_zero()
    }

    pub fn frustum_bounds_on_plane_zup(&self, plane_z: f32) -> [Vec3; 4] {
        let inv_vp = self.view_proj().inverse();
        let corners = [(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)];
        corners.map(|(sx, sy)| {
            let near4 = inv_vp * glam::Vec4::new(sx, sy, 0.0, 1.0);
            let far4 = inv_vp * glam::Vec4::new(sx, sy, 1.0, 1.0);
            let near = (near4 / near4.w).truncate();
            let far = (far4 / far4.w).truncate();
            let dir = far - near;
            if dir.z.abs() < 1e-5 {
                return far;
            }
            let t = (plane_z - near.z) / dir.z;
            near + dir * t
        })
    }
}

/// LERPFLOAT(t, a, b) = a + (b - a) * t
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
    MoveUp,
    MoveDown,
    RotLeft,
    RotRight,
}
