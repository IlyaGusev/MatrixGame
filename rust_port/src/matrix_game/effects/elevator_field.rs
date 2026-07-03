//! Port of `MatrixEffectElevatorField.{cpp,hpp}` — the spiral tractor
//! beam shown while a flyer lifts / lowers a robot. Seven bezier
//! helices run from the ground anchor up to the flyer; tracer
//! billboards stream along them.

use crate::matrix_lib::three_g::billboard::{BillboardQueue, TexRef};
use glam::Vec3;

/// `ELEVATORFIELD_CNT` (MatrixEffectElevatorField.hpp:8).
const FIELD_CNT: usize = 7;
/// `ELEVATORFIELD_BB_CNT`.
const BB_CNT: usize = 300;
/// `ELEVATORFIELD_SPAWN_PERIOD` (ms).
const SPAWN_PERIOD: f32 = 10.0;
/// `ELEVATORFIELD_BB_SIZE`.
const BB_SIZE: f32 = 3.0;


/// Control-point coefficients (MatrixEffectElevatorField.cpp:60-88).
const KOEFS: [(f32, f32); 5] = [
    (0.9, 0.0),
    (0.0, -0.67),
    (-0.2, 0.2),
    (0.32, 0.65),
    (0.81, 0.3),
];

#[derive(Debug, Clone, Copy)]
struct Tracer {
    helix: usize,
    t: f32,
    prev: Vec3,
}

#[derive(Debug, Clone)]
pub struct ElevatorField {
    /// Per-helix bezier control points (6 each).
    helices: Vec<[Vec3; 6]>,
    tracers: Vec<Tracer>,
    spawn_accum: f32,
    /// `m_Activated` — set once the first tracer finishes a pass; the
    /// carry physics reads it to start reeling the robot in.
    pub activated: bool,
}

impl ElevatorField {
    /// Ctor (MatrixEffectElevatorField.cpp:44-101). `pos0` = top
    /// (flyer), `pos1` = bottom (robot/ground), `fwd` = flyer forward.
    pub fn new(pos0: Vec3, pos1: Vec3, radius: f32, fwd: Vec3) -> Self {
        let dir = (pos0 - pos1).normalize_or_zero();
        let perp = fwd.cross(dir).normalize_or_zero();
        let r = radius;

        let base: Vec<Vec3> = KOEFS
            .iter()
            .map(|&(kd, kp)| pos1 + dir * (r * kd) + perp * (r * kp))
            .collect();

        let mut helices = Vec::with_capacity(FIELD_CNT);
        let da = std::f32::consts::TAU / FIELD_CNT as f32;
        for i in 0..FIELD_CNT {
            let ang = da * i as f32;
            let rot = glam::Quat::from_axis_angle(dir.normalize_or(Vec3::Z), ang);
            let mut pts = [Vec3::ZERO; 6];
            for (k, b) in base.iter().enumerate() {
                pts[k] = pos1 + rot * (*b - pos1);
            }
            pts[5] = pos0;
            helices.push(pts);
        }
        Self {
            helices,
            tracers: Vec::new(),
            spawn_accum: 0.0,
            activated: false,
        }
    }

    /// `UpdateData` (MatrixEffectElevatorField.cpp:103-146) — re-anchor
    /// the helices; live tracers keep their parameter.
    pub fn update_data(&mut self, pos0: Vec3, pos1: Vec3, radius: f32, fwd: Vec3) {
        let fresh = Self::new(pos0, pos1, radius, fwd);
        self.helices = fresh.helices;
    }

    fn helix_point(pts: &[Vec3; 6], t: f32) -> Vec3 {
        // De Casteljau over the 6 control points (degree-5 bezier) —
        // visually equivalent to the C++ CBezierKords chain.
        let mut p = *pts;
        for level in (1..6).rev() {
            for i in 0..level {
                p[i] = p[i].lerp(p[i + 1], t);
            }
        }
        p[0]
    }

    /// Takt (MatrixEffectElevatorField.cpp:148-213): spawn a tracer on
    /// the least-populated helix every 10ms, advance all tracers.
    pub fn takt(&mut self, step_ms: f32, rng: &mut crate::matrix_game::logic::Rnd) {
        self.spawn_accum += step_ms;
        while self.spawn_accum >= SPAWN_PERIOD {
            self.spawn_accum -= SPAWN_PERIOD;
            if self.tracers.len() >= BB_CNT {
                break;
            }
            let mut counts = [0usize; FIELD_CNT];
            for tr in &self.tracers {
                counts[tr.helix] += 1;
            }
            let helix = counts
                .iter()
                .enumerate()
                .min_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let start = Self::helix_point(&self.helices[helix], 0.0);
            self.tracers.push(Tracer {
                helix,
                t: 0.0,
                prev: start,
            });
        }
        let mut any_done = false;
        for tr in self.tracers.iter_mut() {
            tr.prev = Self::helix_point(&self.helices[tr.helix], tr.t.min(1.0));
            // dt = FRND(0.01)+0.001 → [0.001, 0.011] (MatrixEffectElevatorField.cpp:207),
            // frame-rate-normalised via step/10.
            tr.t += (rng.float01() as f32 * 0.01 + 0.001) * (step_ms / 10.0);
            if tr.t > 1.0 {
                any_done = true;
            }
        }
        if any_done {
            self.activated = true;
        }
        self.tracers.retain(|tr| tr.t <= 1.0);
    }

    /// Draw — one short line billboard per tracer with a fade-in over
    /// the first 20% of the path.
    pub fn draw(&self, q: &mut BillboardQueue) {
        for tr in &self.tracers {
            let p = Self::helix_point(&self.helices[tr.helix], tr.t.min(1.0));
            // Fixed 0.1-parameter trailing streak (CalcPoint(t-0.1)→CalcPoint(t),
            // MatrixEffectElevatorField.cpp:163-165), not the previous frame.
            let tail = Self::helix_point(&self.helices[tr.helix], (tr.t - 0.1).max(0.0));
            let alpha = ((tr.t / 0.2).clamp(0.0, 1.0) * 255.0) as u32;
            let color = (alpha << 24) | 0x00ff_ffff;
            // BBT_EFIELD — the tracer texture lives in the shared
            // billboard atlas (MatrixEffect.cpp:63).
            q.line(
                tail,
                p,
                BB_SIZE,
                color,
                TexRef::Bbt(crate::matrix_lib::three_g::billboard::BBT_EFIELD),
            );
        }
    }
}
