//! Port of `MatrixEffectDust.{cpp,hpp}` — `CMatrixEffectDust`: five
//! dust-puff billboards drifting outward from a point, growing from
//! `DUST_R1` to `DUST_R2` and fading from `DUST_C1` to `DUST_C2`
//! over their TTL.

use crate::matrix_game::effects::billboard_fx::lic;
use crate::matrix_game::logic::Rnd;
use crate::matrix_game::map::GameMap;
use crate::matrix_lib::three_g::billboard::{BillboardQueue, TexRef};
use glam::{Vec2, Vec3};

const DUST_BILLS_CNT: usize = 5;
const DUST_R1: f32 = 5.0;
const DUST_R1_VAR: f32 = 3.0;
const DUST_R2: f32 = 25.0;
const DUST_R2_VAR: f32 = 7.0;
const DUST_C1: u32 = 0x4F88_8888;
const DUST_C2: u32 = 0x0077_7777;
const DUST_SPEED_K: f32 = 1.5;
/// `TEXTURE_PATH_DUST` (StringConstants.hpp:126 — note the capital D).
pub const TEXTURE_PATH_DUST: &str = "Matrix/Textures/Billboard/Dust";

#[derive(Debug, Clone, Copy)]
struct Puff {
    pos: Vec3,
    dir: Vec2,
    r1: f32,
    r2: f32,
    ttl: f32,
}

#[derive(Debug, Clone)]
pub struct Dust {
    puffs: [Puff; DUST_BILLS_CNT],
    add_speed: Vec2,
    ttl_total: f32,
}

impl Dust {
    /// `CreateDust(pos, adddir, ttl)` (MatrixEffectDust.cpp:19-44).
    pub fn new(pos: Vec2, adddir: Vec2, ttl: f32, rng: &mut Rnd) -> Self {
        let mut fsrnd = |k: f32| (rng.float01() as f32 * 2.0 - 1.0) * k;
        let puffs = std::array::from_fn(|_| {
            let dir = Vec2::new(fsrnd(1.0), fsrnd(1.0)).normalize_or(Vec2::X);
            Puff {
                pos: Vec3::new(pos.x, pos.y, 0.0),
                dir: dir * DUST_SPEED_K,
                r1: DUST_R1 + fsrnd(DUST_R1_VAR),
                r2: DUST_R2 + fsrnd(DUST_R2_VAR),
                ttl,
            }
        });
        Self {
            puffs,
            add_speed: adddir,
            ttl_total: ttl,
        }
    }

    /// `Takt` (MatrixEffectDust.cpp:76-105). Returns false when dead.
    pub fn takt(&mut self, time: f32, map: &GameMap) -> bool {
        let mul = 0.996f64.powf(time as f64) as f32;
        let mut dead = 0;
        for p in self.puffs.iter_mut() {
            p.pos.x += p.dir.x + self.add_speed.x;
            p.pos.y += p.dir.y + self.add_speed.y;
            p.pos.z = map.get_z(p.pos.x, p.pos.y) + 0.1;
            p.dir *= mul;
            p.ttl = (p.ttl - time).max(0.0);
            if p.ttl == 0.0 {
                dead += 1;
            }
        }
        dead < DUST_BILLS_CNT
    }

    pub fn draw(&self, q: &mut BillboardQueue) {
        for p in &self.puffs {
            if p.ttl <= 0.0 {
                continue;
            }
            // CMatrixEffectBillboard lerps scale r1→r2 and color c1→c2
            // by 1 - ttl/ttl_total.
            let t = 1.0 - p.ttl / self.ttl_total.max(1.0);
            let scale = p.r1 + (p.r2 - p.r1) * t;
            let color = lic(DUST_C1, DUST_C2, t);
            // Same flat-quad geometry as the C++ billboard ctor —
            // dust puffs hug the terrain (pos.z = GetZ + 0.1).
            q.quad_flat(p.pos, scale, color, TexRef::Path(TEXTURE_PATH_DUST));
        }
    }
}
