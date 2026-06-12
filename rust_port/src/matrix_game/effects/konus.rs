//! Port of `MatrixEffectKonus.{cpp,hpp}` — the 7-sector cone used for
//! muzzle flashes / impact splashes, plus the `KonusSplash` water
//! variant.

use glam::Vec3;

use crate::matrix_lib::three_g::billboard::{BillboardQueue, QueuedTri, TexRef};
use crate::matrix_lib::three_g::math3d::vec_to_matrix_x;

/// `KONUS_NUM_SECTORS` (MatrixEffectKonus.hpp:13).
const KONUS_NUM_SECTORS: usize = 7;

/// `CMatrixEffectKonus`.
pub struct Konus {
    pos: Vec3,
    dir: Vec3,
    radius: f32,
    height: f32,
    angle: f32,
    ttl: f32,
    ttl0: f32,
    intense: bool,
    color: u32,
    tex: TexRef,
    splash: Option<SplashInit>,
}

/// Extra state for the `CMatrixEffectKonusSplash` Takt
/// (MatrixEffectKonus.cpp:165-190).
struct SplashInit {
    height: f32,
    radius: f32,
    pos: Vec3,
}

impl Konus {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pos: Vec3,
        dir: Vec3,
        radius: f32,
        height: f32,
        angle: f32,
        ttl: f32,
        intense: bool,
        tex: TexRef,
    ) -> Self {
        Self {
            pos,
            dir,
            radius,
            height,
            angle,
            ttl,
            ttl0: ttl,
            intense,
            color: 0xFFFF_FFFF,
            tex,
            splash: None,
        }
    }

    /// `CMatrixEffectKonusSplash` ctor.
    #[allow(clippy::too_many_arguments)]
    pub fn new_splash(
        pos: Vec3,
        dir: Vec3,
        radius: f32,
        height: f32,
        angle: f32,
        ttl: f32,
        intense: bool,
        tex: TexRef,
    ) -> Self {
        let mut k = Self::new(pos, dir, radius, height, angle, ttl, intense, tex);
        k.splash = Some(SplashInit {
            height,
            radius,
            pos,
        });
        k
    }

    /// `Modify` (used by the plasma muzzle cone tracking the barrel).
    pub fn modify(&mut self, pos: Vec3, dir: Vec3) {
        self.pos = pos;
        self.dir = dir;
    }

    /// Takt (MatrixEffectKonus.cpp:126-143 / :165-190 for splash).
    pub fn takt(&mut self, step: f32) -> bool {
        self.ttl -= step;
        if self.ttl < 0.0 {
            return false;
        }
        let t = self.ttl / self.ttl0;
        if let Some(ref init) = self.splash {
            use std::f32::consts::PI;
            self.height = init.height * (PI * (1.0 - t) * 1.5).sin();
            self.radius = init.radius * (PI * (1.0 - t) * 0.5).sin();
            self.pos = init.pos;
            self.pos.z += 0.5 * init.height * (PI * (1.0 - t)).sin();
        }
        let a = (255.0 * t) as u32;
        self.color = (self.color & 0x00FF_FFFF) | (a << 24);
        true
    }

    /// Tessellate the cone fan into the queue — the 9-vertex strip of
    /// `PrepareDX` (MatrixEffectKonus.cpp:37-65) transformed by the
    /// `UpdateMatrix` basis (:146-162).
    pub fn draw(&self, q: &mut BillboardQueue) {
        let (sa, ca) = self.angle.sin_cos();
        let rows = vec_to_matrix_x(self.pos, self.dir);
        // Local rows scaled: X by height, Y/Z by rotated radius.
        let rx = rows[0] * self.height;
        let ry = (rows[1] * ca + rows[2] * sa) * self.radius;
        let rz = (rows[1] * -sa + rows[2] * ca) * self.radius;
        let world = |p: Vec3| -> Vec3 { rx * p.x + ry * p.y + rz * p.z + self.pos };

        let tip = (self.pos, [0.5f32, 0.5f32]);
        let ring: Vec<(Vec3, [f32; 2])> = (0..=KONUS_NUM_SECTORS)
            .map(|i| {
                // i = 0 corresponds to the C++ v[1] at (1,1,0)... the
                // first/last ring points are (1, 1, 0) with uv (1, .5);
                // intermediate ones walk the unit circle.
                if i == 0 || i == KONUS_NUM_SECTORS {
                    (world(Vec3::new(1.0, 1.0, 0.0)), [1.0, 0.5])
                } else {
                    let angle = std::f32::consts::TAU / KONUS_NUM_SECTORS as f32 * i as f32;
                    let (ss, cc) = angle.sin_cos();
                    (
                        world(Vec3::new(1.0, cc, ss)),
                        [(cc + 1.0) * 0.5, (ss + 1.0) * 0.5],
                    )
                }
            })
            .collect();
        for w in ring.windows(2) {
            q.tris.push(QueuedTri {
                v: [tip.0, w[0].0, w[1].0],
                uv: [tip.1, w[0].1, w[1].1],
                color: self.color,
                tex: self.tex,
                additive: self.intense,
            });
        }
    }
}
