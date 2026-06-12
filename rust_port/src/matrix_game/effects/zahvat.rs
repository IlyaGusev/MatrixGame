//! Port of `MatrixEffectZahvat.{cpp,hpp}` — `CMatrixEffectZahvat`,
//! the ring of capture-progress dots in front of a factory.
//!
//! `cnt` billboards sit on a circle of `radius` starting at `angle`;
//! `UpdateData(color, count)` paints the first `count` dots with the
//! capturer's side color and the rest with the near-invisible
//! `ZAHVAT_SPOT_GRAY1`. The original's per-dot flash interpolation is
//! disabled in the shipped build (`Takt` body commented out) — the
//! dots are static colors.

use crate::matrix_lib::three_g::billboard::{BillboardQueue, TexRef};
use glam::Vec3;

/// `ZAHVAT_SPOT_SIZE` (MatrixEffectZahvat.hpp:9).
pub const ZAHVAT_SPOT_SIZE: f32 = 6.0;
/// `ZAHVAT_SPOT_GRAY1` (MatrixEffectZahvat.hpp:10).
pub const ZAHVAT_SPOT_GRAY1: u32 = 0x00ff_ffff;
/// `TEXTURE_PATH_ZAHVAT` (StringConstants.hpp:139).
pub const TEXTURE_PATH_ZAHVAT: &str = "Matrix/Textures/Billboard/zahspot";

#[derive(Debug, Clone)]
pub struct Zahvat {
    points: Vec<Vec3>,
    colors: Vec<u32>,
}

impl Zahvat {
    /// `CMatrixEffectZahvat::CMatrixEffectZahvat` (MatrixEffectZahvat.
    /// cpp:11-31).
    pub fn new(pos: Vec3, radius: f32, mut angle: f32, cnt: usize) -> Self {
        let da = 2.0 * std::f32::consts::PI / cnt as f32;
        let mut points = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            let (s, c) = angle.sin_cos();
            points.push(pos + Vec3::new(c * radius, s * radius, 0.0));
            angle += da;
        }
        Self {
            points,
            colors: vec![ZAHVAT_SPOT_GRAY1; cnt],
        }
    }

    /// `UpdateData` (MatrixEffectZahvat.cpp:39-53).
    pub fn update_data(&mut self, color: u32, count: i32) {
        let count = (count.max(0) as usize).min(self.colors.len());
        for (i, c) in self.colors.iter_mut().enumerate() {
            *c = if i < count { color } else { ZAHVAT_SPOT_GRAY1 };
        }
    }

    /// `Draw` — one billboard per dot.
    pub fn draw(&self, q: &mut BillboardQueue) {
        for (p, &c) in self.points.iter().zip(self.colors.iter()) {
            if c & 0xFF00_0000 == 0 {
                continue; // alpha 0 — uncaptured dots are invisible
            }
            // Flat XY-plane quad like the original (identity world
            // matrix, MatrixEffectBillboard.cpp:30-33) — a camera-
            // facing sprite would sink its lower half into the ground.
            q.quad_flat(*p, ZAHVAT_SPOT_SIZE, c, TexRef::Path(TEXTURE_PATH_ZAHVAT));
        }
    }
}
