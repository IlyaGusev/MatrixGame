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

/// `CMatrixEffectBillboardScore` (MatrixEffectBillboard.cpp:282-451) — the
/// floating "+N" resource-income numbers over buildings: a row of digit /
/// icon billboards that rise and fade over `BBS_TTL`.
pub struct ScoreFx {
    pos: Vec3,
    /// `(bbt, scale, disp_x)` per glyph.
    glyphs: Vec<(usize, f32, f32)>,
    ttl: f32,
}

impl ScoreFx {
    pub fn new(text: &str, pos: Vec3) -> Self {
        use crate::matrix_lib::three_g::billboard::{BBT_SCORE0, BBT_SCOREPLUS};
        const BBS_DX: f32 = 6.0;
        const BBS_ICON_DX: f32 = 12.0;
        let n = text.chars().count() as f32;
        let mut disp_x = -BBS_DX / 2.0 * n;
        let mut addw = 0.0f32;
        let mut glyphs = Vec::new();
        for ch in text.chars() {
            // (bbt, scale, dx-advance, next-addw). Digits 1..9 = 40..48,
            // ScoreIcon1..4 = 49..52 (e/p/b/t), ScoreIcons = 53 (a).
            let (tex, scale, dx, next_addw) = match ch {
                '0' => (BBT_SCORE0, 2.5, BBS_DX + addw, 0.0),
                '1' => (40, 2.5, BBS_DX + addw - 3.0, 0.0),
                d @ '2'..='9' => (39 + (d as usize - '0' as usize), 2.5, BBS_DX + addw, 0.0),
                'e' => (49, 6.0, BBS_ICON_DX, 3.0),
                'p' => (50, 6.0, BBS_ICON_DX, 3.0),
                'b' => (51, 6.0, BBS_ICON_DX, 3.0),
                't' => (52, 6.0, BBS_ICON_DX, 3.0),
                'a' => (53, 12.0, BBS_ICON_DX * 2.0, 10.0),
                _ => (BBT_SCOREPLUS, 2.5, BBS_DX + addw, 0.0),
            };
            disp_x += dx;
            addw = next_addw;
            glyphs.push((tex, scale, disp_x));
        }
        Self {
            pos,
            glyphs,
            ttl: 2000.0, // BBS_TTL
        }
    }

    pub fn takt(&mut self, step: f32) -> bool {
        self.ttl -= step;
        self.pos.z += step * 0.02; // BBS_SPEED
        self.ttl >= 0.0
    }

    pub fn draw(&self, q: &mut BillboardQueue) {
        use glam::Vec2;
        // Fade over the last (1 - BBS_FADE) of life (KSCALE(t, 0.7, 1)).
        let t = 1.0 - self.ttl / 2000.0;
        let fade = ((t - 0.7) / 0.3).clamp(0.0, 1.0);
        let alpha = ((1.0 - fade) * 255.0) as u32;
        let color = (alpha << 24) | 0x00FF_FFFF;
        for &(tex, scale, dispx) in &self.glyphs {
            q.billboard_disp(self.pos, scale, color, TexRef::Bbt(tex), Vec2::new(dispx, 0.0));
        }
    }
}
