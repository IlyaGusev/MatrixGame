//! Port of `MatrixEffectLightening.cpp` — the electric-weapon bolt
//! (`CMatrixEffectLightening`) and the short-circuit arc on shocked
//! units (`CMatrixEffectShorted`). A chain of GLOWBEAM billboard-line
//! segments with per-frame random displacement, plus a GLOWBEAMEND cap
//! at the source.

use glam::{Quat, Vec3};

use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd};
use crate::matrix_game::logic::Rnd;
use crate::matrix_lib::three_g::billboard::{
    BillboardQueue, TexRef, BBT_GLOWBEAM, BBT_GLOWBEAMEND,
};

/// Constants (MatrixEffectLightening.hpp:11-15).
const LIGHTENING_SEGMENT_LENGTH: f32 = 30.0;
pub const LIGHTENING_WIDTH: f32 = 10.0;
const SHORTED_SEGMENT_LENGTH: f32 = 5.0;
pub const SHORTED_WIDTH: f32 = 5.0;

pub struct Lightening {
    pos0: Vec3,
    pos1: Vec3,
    dir: Vec3,
    perp: Vec3,
    len: f32,
    seg_len: f32,
    seg_cnt: usize,
    ttl: f32,
    ttl0: f32,
    width: f32,
    dispersion: f32,
    color: u32,
    /// Shorted variant — arc path + TTL-driven sweep instead of the
    /// straight jitter.
    shorted: bool,
    /// Segment endpoints recomputed each takt (`UpdateData`).
    points: Vec<Vec3>,
}

impl Lightening {
    pub fn new(pos0: Vec3, pos1: Vec3, ttl: f32, dispersion: f32, width: f32, color: u32) -> Self {
        let mut l = Self {
            pos0,
            pos1,
            dir: Vec3::Z,
            perp: Vec3::X,
            len: 1.0,
            seg_len: 1.0,
            seg_cnt: 1,
            ttl,
            ttl0: ttl,
            width,
            dispersion,
            color,
            shorted: false,
            points: Vec::new(),
        };
        l.set_pos(pos0, pos1);
        l
    }

    /// `CreateShorted(pos0, pos1, ttl, color)` —
    /// CMatrixEffectShorted ctor (dispersion 3, SHORTED_WIDTH).
    pub fn new_shorted(pos0: Vec3, pos1: Vec3, ttl: f32, color: u32) -> Self {
        let mut l = Self::new(pos0, pos1, ttl, 3.0, SHORTED_WIDTH, color);
        l.shorted = true;
        l.set_pos(pos0, pos1);
        l
    }

    /// `SetPos` (MatrixEffectLightening.cpp:37-91 / :230-285):
    /// recompute direction, perpendicular and segment count.
    pub fn set_pos(&mut self, pos0: Vec3, pos1: Vec3) {
        self.pos0 = pos0;
        self.pos1 = pos1;
        let d = pos1 - pos0;
        let len = d.length().max(1e-6);
        self.len = len;
        self.dir = d / len;
        self.perp = if self.dir.x == 0.0 && self.dir.y == 0.0 {
            self.dir.cross(Vec3::Y)
        } else {
            self.dir.cross(Vec3::Z)
        };
        let seg = if self.shorted {
            SHORTED_SEGMENT_LENGTH
        } else {
            LIGHTENING_SEGMENT_LENGTH
        };
        let mut cnt = (len / seg) as usize + 1;
        let last = len - (cnt - 1) as f32 * seg;
        if last < seg / 2.0 && cnt > 1 {
            cnt -= 1;
        }
        self.seg_cnt = cnt;
        self.seg_len = len / cnt as f32;
    }

    /// Per-frame path regeneration — `UpdateData`
    /// (MatrixEffectLightening.cpp:93-112 straight bolt, :192-228 arc).
    fn update_points(&mut self, rng: &mut Rnd) {
        self.points.clear();
        self.points.push(self.pos0);
        let axis = self.dir.normalize_or_zero();
        if self.shorted {
            let h = self.len * 0.3;
            let dang = std::f32::consts::PI / self.seg_cnt as f32;
            let mut ang = std::f32::consts::PI - dang;
            let sweep = -(self.ttl / self.ttl0) * std::f32::consts::PI;
            let rot = Quat::from_axis_angle(axis, sweep);
            for _ in 0..self.seg_cnt.saturating_sub(1) {
                let dx = ang.cos() * self.len * 0.5 + self.len * 0.5;
                let dy = ang.sin() * h;
                let mut p = self.pos0 + self.dir * dx + self.perp * dy;
                ang -= dang;
                p.x += fsrnd(rng, self.seg_len * 0.5);
                p.y += fsrnd(rng, self.seg_len * 0.5);
                p.z += fsrnd(rng, self.seg_len * 0.5);
                let p = rot * (p - self.pos0) + self.pos0;
                self.points.push(p);
            }
        } else {
            let mut posx = self.pos0;
            for _ in 0..self.seg_cnt.saturating_sub(1) {
                posx += self.dir * self.seg_len;
                let p = posx
                    + self.dir * fsrnd(rng, self.seg_len * 0.5)
                    + self.perp * frnd(rng, self.dispersion);
                let rot = Quat::from_axis_angle(axis, fsrnd(rng, std::f32::consts::PI));
                let p = rot * (p - self.pos0) + self.pos0;
                self.points.push(p);
            }
        }
        self.points.push(self.pos1);
    }

    pub fn takt(&mut self, step: f32, rng: &mut Rnd) -> bool {
        self.ttl -= step;
        if self.ttl < 0.0 {
            return false;
        }
        self.update_points(rng);
        true
    }

    pub fn draw(&self, q: &mut BillboardQueue) {
        // GLOWBEAMEND cap at the source.
        q.billboard(
            self.pos0,
            self.width * 0.5,
            0.0,
            self.color,
            TexRef::Bbt(BBT_GLOWBEAMEND),
        );
        for w in self.points.windows(2) {
            q.line(
                w[0],
                w[1],
                self.width,
                self.color,
                TexRef::Bbt(BBT_GLOWBEAM),
            );
        }
    }
}
