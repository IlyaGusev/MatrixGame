//! Port of `MatrixLib/3G/CBillboard.{hpp,cpp}` — the camera-facing
//! quad system every effect draws through.
//!
//! The C++ keeps two global queues flushed once per frame from
//! `CBillboard::SortEndDraw`: depth-sorted alpha-blended billboards
//! (one shared "sortable" atlas texture) and texture-grouped additive
//! ("intense") billboards + billboard lines. The Rust port mirrors
//! that as a [`BillboardQueue`] of logical primitives that the wgpu
//! effects renderer expands into vertices each frame with the current
//! camera basis — same math, same two blend buckets, same draw order
//! (sorted alpha first, additive after; both depth-tested with depth
//! writes off).

use glam::{Vec2, Vec3};

// ── EBillboardTextureSort (MatrixEffect.hpp:17-78) ───────────────────────
//
// Indices into the BBT texture table loaded from the `Billboards/
// Textures` block of robots.dat. Sorted entries are atlas rects in the
// `TexSort` texture; intense entries are standalone textures drawn
// additively.

pub const BBT_SMOKE: usize = 0;
pub const BBT_SMOKE_INTENSE: usize = 1;
pub const BBT_FIRE: usize = 2;
pub const BBT_FIRE_INTENSE: usize = 3;
pub const BBT_POINTLIGHT: usize = 4;
pub const BBT_LASEREND: usize = 5;
pub const BBT_SELDOT: usize = 6;
pub const BBT_PATHDOT: usize = 7;
pub const BBT_GLOWBEAMEND: usize = 8;
pub const BBT_FLAME: usize = 9;
pub const BBT_PLASMA1: usize = 10;
pub const BBT_PLASMA2: usize = 11;
pub const BBT_INTENSE: usize = 12;
pub const BBT_LASER: usize = 13;
pub const BBT_GUNFIRE1: usize = 14;
pub const BBT_GUNFIRE2: usize = 15;
pub const BBT_GUNFIRE3: usize = 16;
pub const BBT_TRASSA: usize = 17;
pub const BBT_SHLEIF: usize = 18;
pub const BBT_FIRESTREAM1: usize = 19;
pub const BBT_FIRESTREAM2: usize = 20;
pub const BBT_FIRESTREAM3: usize = 21;
pub const BBT_FIRESTREAM4: usize = 22;
pub const BBT_FIRESTREAM5: usize = 23;
pub const BBT_GLOWBEAM: usize = 24;
pub const BBT_FLAMELINE: usize = 25;
pub const BBT_SPARK: usize = 26;
pub const BBT_EFIELD: usize = 27;
pub const BBT_FLAMEFRAME0: usize = 28;
pub const BBT_FLAMEFRAME1: usize = 29;
pub const BBT_FLAMEFRAME2: usize = 30;
pub const BBT_FLAMEFRAME3: usize = 31;
pub const BBT_FLAMEFRAME4: usize = 32;
pub const BBT_FLAMEFRAME5: usize = 33;
pub const BBT_FLAMEFRAME6: usize = 34;
pub const BBT_FLAMEFRAME7: usize = 35;
pub const BBT_REPGLOWEND: usize = 36;
pub const BBT_REPGLOW: usize = 37;
pub const BBT_SCOREPLUS: usize = 38;
pub const BBT_SCORE0: usize = 39;
// BBT_SCORE1..9 = 40..48, BBT_SCOREICON1..4 = 49..52, BBT_SCOREICONS = 53.
pub const BBT_LAST: usize = 54;

/// `(BBT index, param key in the Billboards/Textures block)` — port of
/// the `bb2tn[]` table (MatrixEffect.cpp:34-99).
pub const BBT_NAMES: [(usize, &str); BBT_LAST] = [
    (BBT_SMOKE, "Smoke"),
    (BBT_SMOKE_INTENSE, "Smoke_intense"),
    (BBT_FIRE, "Fire"),
    (BBT_FIRE_INTENSE, "Fire_intense"),
    (BBT_POINTLIGHT, "Pointlight"),
    (BBT_LASEREND, "Laserend"),
    (BBT_SELDOT, "Seldot"),
    (BBT_PATHDOT, "Pathdot"),
    (BBT_GLOWBEAMEND, "GlowBeamEnd"),
    (BBT_FLAME, "Flame"),
    (BBT_PLASMA1, "Plasma1"),
    (BBT_PLASMA2, "Plasma2"),
    (BBT_INTENSE, "Intense"),
    (BBT_LASER, "Laser"),
    (BBT_GUNFIRE1, "Gunfire1"),
    (BBT_GUNFIRE2, "Gunfire2"),
    (BBT_GUNFIRE3, "Gunfire3"),
    (BBT_TRASSA, "Trassa"),
    (BBT_SHLEIF, "Shleif"),
    (BBT_FIRESTREAM1, "Fire_stream1"),
    (BBT_FIRESTREAM2, "Fire_stream2"),
    (BBT_FIRESTREAM3, "Fire_stream3"),
    (BBT_FIRESTREAM4, "Fire_stream4"),
    (BBT_FIRESTREAM5, "Fire_stream5"),
    (BBT_GLOWBEAM, "GlowBeam"),
    (BBT_FLAMELINE, "Flame_line"),
    (BBT_SPARK, "Spark"),
    (BBT_EFIELD, "Efield"),
    (BBT_FLAMEFRAME0, "FlameFrame0"),
    (BBT_FLAMEFRAME1, "FlameFrame1"),
    (BBT_FLAMEFRAME2, "FlameFrame2"),
    (BBT_FLAMEFRAME3, "FlameFrame3"),
    (BBT_FLAMEFRAME4, "FlameFrame4"),
    (BBT_FLAMEFRAME5, "FlameFrame5"),
    (BBT_FLAMEFRAME6, "FlameFrame6"),
    (BBT_FLAMEFRAME7, "FlameFrame7"),
    (BBT_REPGLOWEND, "Repair_glow_end"),
    (BBT_REPGLOW, "Repair_glow"),
    (BBT_SCOREPLUS, "ScorePlus"),
    (BBT_SCORE0, "Score0"),
    (40, "Score1"),
    (41, "Score2"),
    (42, "Score3"),
    (43, "Score4"),
    (44, "Score5"),
    (45, "Score6"),
    (46, "Score7"),
    (47, "Score8"),
    (48, "Score9"),
    (49, "ScoreIcon1"),
    (50, "ScoreIcon2"),
    (51, "ScoreIcon3"),
    (52, "ScoreIcon4"),
    (53, "ScoreIcons"),
];

/// Which texture a primitive samples: a BBT table entry, or one of the
/// standalone effect textures the C++ pulls from the cache by path
/// (konus muzzle flash, splashes, the BigBoom shell, landscape spots).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TexRef {
    Bbt(usize),
    Path(&'static str),
}

/// A queued camera-facing quad — port of `CBillboard` (the per-frame
/// submission, not the retained object: every effect re-queues its
/// billboards each frame in this port, which is what the C++ Sort /
/// AddToDrawQueue calls amount to).
#[derive(Debug, Clone, Copy)]
pub struct QueuedBillboard {
    pub pos: Vec3,
    pub scale: f32,
    /// In-plane spin (`m_Rot` 2×2 part) — `(cos a, sin a)`.
    pub rot: Vec2,
    /// In-plane displacement (`m_Rot._41/_42`) — used by the explosion
    /// "intense" sprites that drift in screen space (`DisplaceTo`).
    pub disp: Vec2,
    /// ARGB, as everywhere in the original.
    pub color: u32,
    pub tex: TexRef,
}

/// A queued line billboard — port of `CBillboardLine`.
#[derive(Debug, Clone, Copy)]
pub struct QueuedLine {
    pub p0: Vec3,
    pub p1: Vec3,
    pub width: f32,
    pub color: u32,
    pub tex: TexRef,
}

/// A pre-transformed textured triangle (world space). Carries the
/// cone (Konus) and BigBoom-shell geometry whose vertices the C++
/// computes on the CPU too.
#[derive(Debug, Clone, Copy)]
pub struct QueuedTri {
    pub v: [Vec3; 3],
    pub uv: [[f32; 2]; 3],
    pub color: u32,
    pub tex: TexRef,
    pub additive: bool,
}

/// One frame's worth of effect primitives. Filled by the effect draw
/// pass, expanded + flushed by the effects renderer.
#[derive(Default)]
pub struct BillboardQueue {
    pub billboards: Vec<QueuedBillboard>,
    pub lines: Vec<QueuedLine>,
    pub tris: Vec<QueuedTri>,
}

impl BillboardQueue {
    pub fn clear(&mut self) {
        self.billboards.clear();
        self.lines.clear();
        self.tris.clear();
    }

    /// Plain unrotated billboard.
    pub fn billboard(&mut self, pos: Vec3, scale: f32, angle: f32, color: u32, tex: TexRef) {
        let (s, c) = angle.sin_cos();
        self.billboards.push(QueuedBillboard {
            pos,
            scale,
            rot: Vec2::new(c, s),
            disp: Vec2::ZERO,
            color,
            tex,
        });
    }

    pub fn line(&mut self, p0: Vec3, p1: Vec3, width: f32, color: u32, tex: TexRef) {
        self.lines.push(QueuedLine {
            p0,
            p1,
            width,
            color,
            tex,
        });
    }
}

/// Expand a [`QueuedBillboard`] into its 4 world-space corners using
/// the camera basis — port of the matrix build in
/// `CBillboard::SortEndDraw` (CBillboard.cpp:472-505). Corner order
/// matches the C++ base array: `(-1,-1) (-1,1) (1,-1) (1,1)`.
pub fn expand_billboard(b: &QueuedBillboard, right: Vec3, up: Vec3) -> [Vec3; 4] {
    let r11 = b.rot.x * b.scale;
    let r12 = b.rot.y * b.scale;
    let r21 = -b.rot.y * b.scale;
    let r22 = b.rot.x * b.scale;
    let row0 = right * r11 + up * r12;
    let row1 = right * r21 + up * r22;
    let pos = b.pos + right * b.disp.x + up * b.disp.y;
    [
        pos - row0 - row1,
        pos - row0 + row1,
        pos + row0 - row1,
        pos + row0 + row1,
    ]
}

/// Expand a [`QueuedLine`] into its 4 corners — port of the
/// `CBillboardLine` matrix build (CBillboard.cpp:621-666): X axis along
/// the line, Y axis = normalized `dir × viewdir` scaled by width.
/// Corner order matches base `(0,.5) (0,-.5) (1,.5) (1,-.5)`.
pub fn expand_line(l: &QueuedLine, cam_pos: Vec3) -> [Vec3; 4] {
    let dir = l.p1 - l.p0;
    let viewdir = (l.p0 + l.p1) * 0.5 - cam_pos;
    let left = dir.cross(viewdir).normalize_or_zero() * (l.width * 0.5);
    [l.p0 + left, l.p0 - left, l.p1 + left, l.p1 - left]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn billboard_expansion_is_centered_and_scaled() {
        let b = QueuedBillboard {
            pos: Vec3::new(10.0, 20.0, 30.0),
            scale: 2.0,
            rot: Vec2::new(1.0, 0.0),
            disp: Vec2::ZERO,
            color: 0xFFFFFFFF,
            tex: TexRef::Bbt(BBT_SMOKE),
        };
        let v = expand_billboard(&b, Vec3::X, Vec3::Z);
        let center = (v[0] + v[1] + v[2] + v[3]) * 0.25;
        assert!((center - b.pos).length() < 1e-5);
        // Half-extent along camera right = scale.
        assert!(((v[2] - v[0]).length() * 0.5 - 2.0).abs() < 1e-4);
    }

    #[test]
    fn line_expansion_spans_endpoints_with_width() {
        let l = QueuedLine {
            p0: Vec3::ZERO,
            p1: Vec3::new(10.0, 0.0, 0.0),
            width: 4.0,
            color: 0xFFFFFFFF,
            tex: TexRef::Bbt(BBT_LASER),
        };
        // Camera below the line looking up: viewdir ≈ +Z.
        let v = expand_line(&l, Vec3::new(5.0, 0.0, -100.0));
        // Width across the line = 4.
        assert!(((v[0] - v[1]).length() - 4.0).abs() < 1e-4);
        // Ends at the line endpoints (midpoints of corner pairs).
        assert!((((v[0] + v[1]) * 0.5) - l.p0).length() < 1e-4);
        assert!((((v[2] + v[3]) * 0.5) - l.p1).length() < 1e-4);
    }
}
