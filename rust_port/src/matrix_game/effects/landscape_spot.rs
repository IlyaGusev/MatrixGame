//! Port of `MatrixEffectLandscapeSpot.{cpp,hpp}` — terrain decals
//! (craters, plasma scorch). Geometry samples the heightmap grid under
//! a rotated quad and floats `SPOT_ALTITUDE` above it; alpha runs the
//! per-type fade program.

use glam::{Vec2, Vec3};

use crate::matrix_game::map::{GameMap, GLOBAL_SCALE};
use crate::matrix_lib::three_g::billboard::TexRef;

/// `SPOT_ALTITUDE` / `SPOT_SIZE` (MatrixEffectLandscapeSpot.cpp).
const SPOT_ALTITUDE: f32 = 0.7;
const SPOT_SIZE: f32 = 5.0;
/// `MAX_EFFECTS_LANDSPOTS` (MatrixEffect.hpp) — live-spot cap.
const MAX_SPOTS: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpotKind {
    /// `SPOT_VORONKA` — crater, 30s.
    Voronka,
    /// `SPOT_PLASMA_HIT` — scorch flash, 3s.
    PlasmaHit,
    /// `SPOT_CONSTANT` — black burn mark, 15s.
    Constant,
}

impl SpotKind {
    fn ttl(self) -> f32 {
        match self {
            SpotKind::Voronka => 30_000.0,
            SpotKind::PlasmaHit => 3_000.0,
            SpotKind::Constant => 15_000.0,
        }
    }
    pub fn texture(self) -> TexRef {
        match self {
            SpotKind::Voronka => TexRef::Path("Matrix/Textures/LandSpots/varonka"),
            SpotKind::PlasmaHit => TexRef::Path("Matrix/Textures/LandSpots/spot_hit"),
            SpotKind::Constant => TexRef::Path("Matrix/Textures/LandSpots/spot"),
        }
    }
}

/// Spawn request queued from gameplay code (`CreateLandscapeSpot`).
#[derive(Debug, Clone, Copy)]
pub struct SpotSpawn {
    pub pos: Vec2,
    pub angle: f32,
    pub scale: f32,
    pub kind: SpotKind,
}

pub struct SpotVertex {
    pub pos: Vec3,
    pub uv: [f32; 2],
}

pub struct Spot {
    pub kind: SpotKind,
    pub verts: Vec<SpotVertex>,
    pub indices: Vec<u32>,
    pub ttl: f32,
    ttl0: f32,
    pub color: u32,
}

impl Spot {
    /// `BuildLand` (MatrixEffectLandscapeSpot.cpp:209-445): grid
    /// sampling under the rotated `SPOT_SIZE * scale` quad with
    /// affine-inverse UVs; triangles with any in-quad corner kept.
    pub fn build(map: &GameMap, s: &SpotSpawn) -> Option<Self> {
        let (sa, ca) = s.angle.sin_cos();
        let ext = SPOT_SIZE * s.scale;
        let xc = ext * ca;
        let xs = ext * sa;
        // Quad corners (parallelogram) — :222-256.
        let c0 = s.pos + Vec2::new(-xc + xs, -xs - xc);
        let c1 = s.pos + Vec2::new(xc + xs, xs - xc);
        let c3 = s.pos + Vec2::new(-xc - xs, -xs + xc);
        let ex = c1 - c0;
        let ey = c3 - c0;
        let ex2 = ex.length_squared().max(1e-6);
        let ey2 = ey.length_squared().max(1e-6);

        // Covered grid rect.
        let r = ext * std::f32::consts::SQRT_2;
        let gx0 = ((s.pos.x - r) / GLOBAL_SCALE).floor() as i32;
        let gy0 = ((s.pos.y - r) / GLOBAL_SCALE).floor() as i32;
        let gx1 = ((s.pos.x + r) / GLOBAL_SCALE).ceil() as i32;
        let gy1 = ((s.pos.y + r) / GLOBAL_SCALE).ceil() as i32;
        let w = (gx1 - gx0 + 1).max(0) as usize;
        let h = (gy1 - gy0 + 1).max(0) as usize;
        if w < 2 || h < 2 || w * h > 4096 {
            return None;
        }

        let mut verts = Vec::with_capacity(w * h);
        let mut inside = vec![false; w * h];
        for gy in gy0..=gy1 {
            for gx in gx0..=gx1 {
                let wx = gx as f32 * GLOBAL_SCALE;
                let wy = gy as f32 * GLOBAL_SCALE;
                let d = Vec2::new(wx, wy) - c0;
                let u = d.dot(ex) / ex2;
                let v = d.dot(ey) / ey2;
                let idx = ((gy - gy0) as usize) * w + (gx - gx0) as usize;
                inside[idx] = (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v);
                verts.push(SpotVertex {
                    pos: Vec3::new(wx, wy, map.get_z(wx, wy) + SPOT_ALTITUDE),
                    uv: [u.clamp(-0.5, 1.5), v.clamp(-0.5, 1.5)],
                });
            }
        }
        let mut indices = Vec::new();
        for gy in 0..h - 1 {
            for gx in 0..w - 1 {
                let i00 = (gy * w + gx) as u32;
                let i10 = i00 + 1;
                let i01 = i00 + w as u32;
                let i11 = i01 + 1;
                let any_in = inside[i00 as usize]
                    || inside[i10 as usize]
                    || inside[i01 as usize]
                    || inside[i11 as usize];
                if !any_in {
                    continue;
                }
                indices.extend_from_slice(&[i00, i01, i10, i10, i01, i11]);
            }
        }
        if indices.is_empty() {
            return None;
        }
        let ttl = s.kind.ttl();
        Some(Self {
            kind: s.kind,
            verts,
            indices,
            ttl,
            ttl0: ttl,
            color: 0,
        })
    }

    /// Per-type fade programs (MatrixEffectLandscapeSpot.cpp:33-168).
    pub fn takt(&mut self, step: f32) -> bool {
        self.ttl -= step;
        if self.ttl <= 0.0 {
            return false;
        }
        let k = self.ttl / self.ttl0; // remaining fraction
        self.color = match self.kind {
            SpotKind::Constant => {
                let a = if k < 0.1 { k / 0.1 * 255.0 } else { 255.0 } as u32;
                a << 24 // black texture modulated by alpha
            }
            SpotKind::Voronka => {
                let a = if k > 0.97 {
                    (1.0 - k) / 0.03 * 255.0
                } else if k < 0.1 {
                    k / 0.1 * 255.0
                } else {
                    255.0
                } as u32;
                (a << 24) | 0x00FF_FFFF
            }
            SpotKind::PlasmaHit => {
                let t = 1.0 - k;
                let ks = |f: f32, to: f32| ((t - f) / (to - f)).clamp(0.0, 1.0);
                let a = ((1.0 - ks(0.5, 1.0)) * 255.0) as u32;
                let r = ((1.0 - ks(0.3, 1.0)) * 255.0) as u32;
                let g = ((1.0 - ks(0.3, 0.7)) * 255.0) as u32;
                let b = ((1.0 - ks(0.0, 1.0)) * 255.0) as u32;
                (a << 24) | (r << 16) | (g << 8) | b
            }
        };
        true
    }
}

/// The live spot list — `CMatrixEffectLandscapeSpot`'s static chain.
#[derive(Default)]
pub struct LandscapeSpots {
    pub spots: Vec<Spot>,
}

impl LandscapeSpots {
    pub fn spawn(&mut self, map: &GameMap, s: &SpotSpawn) {
        if let Some(spot) = Spot::build(map, s) {
            if self.spots.len() >= MAX_SPOTS {
                self.spots.remove(0);
            }
            self.spots.push(spot);
        }
    }

    pub fn takt(&mut self, step: f32) {
        self.spots.retain_mut(|s| s.takt(step));
    }
}
