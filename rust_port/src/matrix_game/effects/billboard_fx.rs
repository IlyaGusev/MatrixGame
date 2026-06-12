//! Ports of the standalone TTL'd billboard effects
//! (`CMatrixEffectBillboardLine`, MatrixEffectBillboard.{cpp,hpp}) —
//! gun muzzle flashes, volcano tracer lines, etc.

use glam::Vec3;

use crate::matrix_lib::three_g::billboard::{BillboardQueue, TexRef};

/// `LIC(c1, c2, t)` — channel-wise ARGB lerp.
pub fn lic(c1: u32, c2: u32, t: f32) -> u32 {
    let ch = |c: u32, sh: u32| ((c >> sh) & 0xFF) as f32;
    let mix = |sh: u32| -> u32 {
        ((ch(c1, sh) + (ch(c2, sh) - ch(c1, sh)) * t).clamp(0.0, 255.0)) as u32
    };
    (mix(24) << 24) | (mix(16) << 16) | (mix(8) << 8) | mix(0)
}

/// `CMatrixEffectBillboardLine` — a line quad fading `color1 → color2`
/// over `ttl`.
pub struct BillboardLineFx {
    pub p0: Vec3,
    pub p1: Vec3,
    pub width: f32,
    color1: u32,
    color2: u32,
    ttl: f32,
    ttl0: f32,
    tex: TexRef,
}

impl BillboardLineFx {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p0: Vec3,
        p1: Vec3,
        width: f32,
        color1: u32,
        color2: u32,
        ttl: f32,
        tex: TexRef,
    ) -> Self {
        Self {
            p0,
            p1,
            width,
            color1,
            color2,
            ttl,
            ttl0: ttl,
            tex,
        }
    }

    pub fn takt(&mut self, step: f32) -> bool {
        self.ttl -= step;
        self.ttl >= 0.0
    }

    pub fn draw(&self, q: &mut BillboardQueue) {
        let k = 1.0 - self.ttl / self.ttl0;
        q.line(
            self.p0,
            self.p1,
            self.width,
            lic(self.color1, self.color2, k),
            self.tex,
        );
    }
}
