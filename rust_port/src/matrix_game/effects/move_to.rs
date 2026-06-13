//! Port of `CMatrixEffectMoveto` (Effects/MatrixEffectMoveTo.{cpp,hpp}).
//!
//! Six billboards spiral inward over `MOVETO_TTL = 400 ms` while the
//! per-billboard z oscillates — the green "go here" ping the original
//! shows on a move order. Per-billboard placement follows
//! `SPointMoveTo::Change` (MatrixEffectMoveTo.cpp:22-48):
//!
//!   radial  r = MOVETO_R * k         — spirals in from 20 units to 0
//!   tangent d = MOVETO_R * 0.3 * sin(π·k)  (sign per `i & 1`)
//!   height  z = MOVETO_Z * sin(π·k / 2)    — 0 → 10 → 0 arc
//!   pos     = center + r·(s,c) + d·(c,-s); z clamped to terrain floor
//!
//! Each point draws as a flat XY-plane quad (`SPointMoveTo`'s matrix is
//! pure scale + translation and `CMatrixEffectBillboard::Draw` uses it
//! verbatim as the world matrix — :50-53) with the
//! `Matrix/Textures/Billboard/moveto` texture and 0xFFFFFFFF color via
//! the shared billboard queue.

use glam::Vec3;

use crate::matrix_game::map::GameMap;
use crate::matrix_lib::three_g::billboard::{BillboardQueue, TexRef};

/// `TEXTURE_PATH_MOVETO` (StringConstants.hpp:142).
pub const TEXTURE_PATH_MOVETO: &str = "Matrix/Textures/Billboard/moveto";

/// MatrixEffectMoveTo.cpp:10 — effect lifetime in ms.
const MOVETO_TTL_MS: f32 = 400.0;
/// MOVETO_S (:11) — billboard world-space scale.
const MOVETO_S: f32 = 2.0;
/// MOVETO_R (:12) — outer spiral radius.
const MOVETO_R: f32 = 20.0;
/// MOVETO_Z (:13) — height of the vertical bounce at its apex.
const MOVETO_Z: f32 = 10.0;
/// Number of billboards per effect (`m_Pts[6]`, MatrixEffectMoveTo.hpp:25).
const PTS: usize = 6;

/// A single live move-to effect. The C++ stores `m_TTL` decrementing to
/// zero; we store the same (ms left until auto-removal).
#[derive(Copy, Clone)]
struct Effect {
    center: Vec3,
    ttl_ms: f32,
}

/// Port of `SPointMoveTo::Change` (MatrixEffectMoveTo.cpp:22-48). Returns
/// the world-space center of billboard `i` at animation phase `k ∈ [0, 1]`.
/// `k = m_TTL / MOVETO_TTL` per the C++ at :107 — k=1 when fresh, 0 at
/// expiry.
fn billboard_pos(map: &GameMap, center: Vec3, i: usize, k: f32) -> Vec3 {
    let pi = std::f32::consts::PI;
    let a = ((i & !1) as f32) * pi / 3.0;
    let (s, c) = a.sin_cos();
    // Tangential jitter — sign alternates per pair (`SET_SIGN_FLOAT(d, i & 1)`).
    let mut d = MOVETO_R * 0.3 * (pi * k).sin();
    if i & 1 == 1 {
        d = -d;
    }
    let z_osc = (pi * k / 2.0).sin();
    let x = center.x + MOVETO_R * s * k + d * c;
    let y = center.y + MOVETO_R * c * k - d * s;
    // Terrain clamp: billboard z never drops below the click-point z,
    // matching the C++ `if (lz < host->m_Pos.z) lz = host->m_Pos.z` at :44.
    let mut lz = map.get_z(x, y);
    if lz < center.z {
        lz = center.z;
    }
    Vec3::new(x, y, lz + MOVETO_Z * z_osc)
}

/// Holds the live move-to pings; drawing goes through the shared
/// billboard queue (the renderer-side pipeline the old stand-in disk
/// shader used is gone — the real `moveto` texture ships in the
/// bundle now).
#[derive(Default)]
pub struct MoveToRenderer {
    effects: Vec<Effect>,
}

impl MoveToRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `CreateMoveto` — one new ping at `pos`.
    pub fn spawn(&mut self, pos: Vec3) {
        self.effects.push(Effect {
            center: pos,
            ttl_ms: MOVETO_TTL_MS,
        });
    }

    /// `DeleteAllMoveto` (the PGOrder* paths clear old pings before
    /// dropping a new batch).
    pub fn clear(&mut self) {
        self.effects.clear();
    }

    /// TTL decay (`Takt`, MatrixEffectMoveTo.cpp:101-110).
    pub fn takt(&mut self, step_ms: f32) {
        for e in &mut self.effects {
            e.ttl_ms -= step_ms;
        }
        self.effects.retain(|e| e.ttl_ms > 0.0);
    }

    pub fn is_active(&self) -> bool {
        !self.effects.is_empty()
    }

    /// `SPointMoveTo::Draw` × 6 per live effect — white ground-aligned
    /// quads with the moveto texture through the shared queue.
    pub fn draw(&self, map: &GameMap, q: &mut BillboardQueue) {
        for e in &self.effects {
            let k = (e.ttl_ms / MOVETO_TTL_MS).clamp(0.0, 1.0);
            for i in 0..PTS {
                let pos = billboard_pos(map, e.center, i, k);
                q.quad_flat(pos, MOVETO_S, 0xFFFF_FFFF, TexRef::Path(TEXTURE_PATH_MOVETO));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttl_decays_and_expires() {
        let mut r = MoveToRenderer::new();
        r.spawn(Vec3::ZERO);
        assert!(r.is_active());
        r.takt(399.0);
        assert!(r.is_active());
        r.takt(2.0);
        assert!(!r.is_active());
    }

    #[test]
    fn billboards_spiral_inward_as_ttl_decays() {
        let map = GameMap::test_flat(16, 16, 0.0);
        let c = Vec3::new(60.0, 60.0, 0.0);
        let fresh = billboard_pos(&map, c, 0, 1.0);
        let dying = billboard_pos(&map, c, 0, 0.1);
        let rf = (fresh - c).truncate().length();
        let rd = (dying - c).truncate().length();
        assert!(rf > rd, "spiral should contract: {rf} vs {rd}");
    }
}
