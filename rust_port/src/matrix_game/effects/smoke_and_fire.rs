//! Port of `MatrixEffectSmokeAndFire.cpp` — `CMatrixEffectSmoke` and
//! `CMatrixEffectFire`, the puff emitters behind trails, wreck smoke
//! and burning objects.

use glam::{Vec2, Vec3};

use crate::matrix_game::logic::Rnd;
use crate::matrix_lib::three_g::billboard::{
    BillboardQueue, TexRef, BBT_FIRE, BBT_FIRE_INTENSE, BBT_SMOKE, BBT_SMOKE_INTENSE,
};

/// Constants from MatrixEffect.hpp / MatrixEffectSmokeAndFire.hpp.
const MAX_PUFF_COUNT: usize = 256;
const PUFF_SCALE1: f32 = 4.0;
const PUFF_SCALE2: f32 = 16.0;
const PUFF_FIRE_SCALE: f32 = 3.5;
/// `SMOKE_SPEED` (MatrixEffect.hpp) — 1/25.
pub const SMOKE_SPEED: f32 = 0.04;

pub(crate) fn frnd(rng: &mut Rnd, x: f32) -> f32 {
    rng.float01() as f32 * x
}
pub(crate) fn fsrnd(rng: &mut Rnd, x: f32) -> f32 {
    (rng.float01() as f32 * 2.0 - 1.0) * x
}
/// `KSCALE(x, from, to)` (Common.hpp) — clamped 0..1 ramp.
pub(crate) fn kscale(x: f32, from: f32, to: f32) -> f32 {
    ((x - from) / (to - from)).clamp(0.0, 1.0)
}

struct SmokePuff {
    pos: Vec3,
    orig: Vec3,
    ttl: f32,
    angle: f32,
    scale: f32,
    color: u32,
}

/// `CMatrixEffectSmoke`.
pub struct Smoke {
    pos: Vec3,
    ttl: f32,
    puff_ttl: f32,
    spawn: f32,
    color: u32,
    intense: bool,
    speed: f32,
    time: f32,
    next_spawn_time: f32,
    puffs: Vec<SmokePuff>,
}

impl Smoke {
    pub fn new(
        pos: Vec3,
        ttl: f32,
        puff_ttl: f32,
        spawn: f32,
        color: u32,
        intense: bool,
        speed: f32,
    ) -> Self {
        Self {
            pos,
            ttl,
            puff_ttl,
            spawn,
            color,
            intense,
            speed,
            time: 0.0,
            next_spawn_time: 0.0,
            puffs: Vec::new(),
        }
    }

    pub fn set_pos(&mut self, pos: Vec3) {
        self.pos = pos;
    }
    pub fn set_ttl(&mut self, ttl: f32) {
        self.ttl = ttl;
    }

    /// `CMatrixEffectSmoke::Takt` (MatrixEffectSmokeAndFire.cpp:82-156).
    /// Returns false when the emitter expired with no live puffs.
    pub fn takt(&mut self, step: f32, rng: &mut Rnd) -> bool {
        let dtime = self.speed * step;
        self.time += step;
        self.ttl -= step;
        while self.spawn > 0.0 && self.time > self.next_spawn_time && self.ttl > 0.0 {
            if self.puffs.len() < MAX_PUFF_COUNT {
                self.puffs.push(SmokePuff {
                    pos: self.pos,
                    orig: self.pos,
                    ttl: self.puff_ttl,
                    angle: frnd(rng, std::f32::consts::PI),
                    scale: 5.0 - frnd(rng, 2.0),
                    color: self.color,
                });
            }
            self.next_spawn_time += self.spawn;
        }
        if self.puffs.is_empty() && self.ttl < 0.0 {
            return false;
        }
        self.puffs.retain_mut(|p| {
            p.ttl -= step;
            p.ttl >= 0.0
        });
        let base_alpha = (self.color >> 24) as f32;
        for p in self.puffs.iter_mut() {
            let life = p.ttl / self.puff_ttl;
            // NOTE: the C++ multiplies by m_Speed twice here (dtime
            // already carries it) — preserved.
            p.pos.z += dtime * self.speed * (5.0 + (p.pos.z - p.orig.z));
            p.scale = PUFF_SCALE2 + (PUFF_SCALE1 - PUFF_SCALE2) * life;
            let a = (base_alpha * life) as u32;
            p.color = (self.color & 0x00FF_FFFF) | (a << 24);
            // In-plane spin is view-only; folded in at draw time.
        }
        true
    }

    pub fn draw(&self, q: &mut BillboardQueue) {
        let tex = if self.intense {
            TexRef::Bbt(BBT_SMOKE_INTENSE)
        } else {
            TexRef::Bbt(BBT_SMOKE)
        };
        for (i, p) in self.puffs.iter().enumerate() {
            let life = p.ttl / self.puff_ttl;
            let mut spin = std::f32::consts::PI * (life * self.speed * 25.0);
            if i & 1 == 1 {
                spin = -spin;
            }
            q.billboard(p.pos, p.scale, p.angle + spin, p.color, tex);
        }
    }
}

struct FirePuff {
    pos: Vec3,
    orig: Vec3,
    ttl: f32,
    angle: f32,
    scale: f32,
    color: u32,
    dir: Vec2,
}

/// `CMatrixEffectFire`.
pub struct Fire {
    pos: Vec3,
    ttl: f32,
    puff_ttl: f32,
    spawn: f32,
    disp_factor: f32,
    intense: bool,
    speed: f32,
    time: f32,
    next_spawn_time: f32,
    puffs: Vec<FirePuff>,
}

impl Fire {
    pub fn new(
        pos: Vec3,
        ttl: f32,
        puff_ttl: f32,
        spawn: f32,
        disp_factor: f32,
        intense: bool,
        speed: f32,
    ) -> Self {
        Self {
            pos,
            ttl,
            puff_ttl,
            spawn,
            disp_factor,
            intense,
            speed,
            time: 0.0,
            next_spawn_time: 0.0,
            puffs: Vec::new(),
        }
    }

    pub fn set_pos(&mut self, pos: Vec3) {
        self.pos = pos;
    }
    pub fn set_ttl(&mut self, ttl: f32) {
        self.ttl = ttl;
    }

    /// `CMatrixEffectFire::Takt` (MatrixEffectSmokeAndFire.cpp:276-382).
    pub fn takt(&mut self, step: f32, rng: &mut Rnd) -> bool {
        let dtime = SMOKE_SPEED * step;
        self.time += step;
        self.ttl -= step;
        while self.spawn > 0.0 && self.time > self.next_spawn_time && self.ttl > 0.0 {
            if self.puffs.len() < MAX_PUFF_COUNT {
                self.puffs.push(FirePuff {
                    pos: self.pos,
                    orig: self.pos,
                    ttl: self.puff_ttl,
                    angle: frnd(rng, std::f32::consts::PI),
                    scale: PUFF_FIRE_SCALE,
                    color: 0xFFFF_FFFF,
                    dir: Vec2::ZERO,
                });
            }
            self.next_spawn_time += self.spawn;
        }
        if self.puffs.is_empty() && self.ttl < 0.0 {
            return false;
        }
        self.puffs.retain_mut(|p| {
            p.ttl -= step;
            p.ttl >= 0.0
        });
        let df = self.disp_factor;
        for p in self.puffs.iter_mut() {
            let life = 1.0 - p.ttl / self.puff_ttl;
            p.pos.z += dtime * self.speed * (10.0 + (p.pos.z - p.orig.z));
            p.scale = PUFF_FIRE_SCALE;
            // Flame color ramp (MatrixEffectSmokeAndFire.cpp:326-331).
            let a = (255.0 * (kscale(life, 0.0, 0.3) - kscale(life, 0.3, 1.0))).max(0.0) as u32;
            let r = ((1.0 - kscale(life, 0.5, 1.0)) * 255.0) as u32;
            let g = ((1.0 - kscale(life, 0.0, 1.0)) * 255.0) as u32;
            let b = 10 + ((1.0 - kscale(life, 0.0, 0.1)) * 245.0) as u32;
            p.color = (a << 24) | (r << 16) | (g << 8) | b;

            // Bounded lateral wander (:333-361).
            let (mut xl, mut xr) = (-df, df);
            let (mut yl, mut yr) = (-df, df);
            let dx = p.pos.x - p.orig.x;
            let dy = p.pos.y - p.orig.y;
            if dx > df {
                xr = df / dx;
            } else if dx < -df {
                xl = df / dx;
            }
            if dy > df {
                yr = df / dy;
            } else if dy < -df {
                yl = df / dy;
            }
            p.dir.x += 0.01 * dtime * (xl + (xr - xl) * rng.float01() as f32);
            p.dir.y += 0.01 * dtime * (yl + (yr - yl) * rng.float01() as f32);
            p.pos.x += p.dir.x;
            p.pos.y += p.dir.y;
        }
        true
    }

    pub fn draw(&self, q: &mut BillboardQueue) {
        let tex = if self.intense {
            TexRef::Bbt(BBT_FIRE_INTENSE)
        } else {
            TexRef::Bbt(BBT_FIRE)
        };
        for p in &self.puffs {
            q.billboard(p.pos, p.scale, p.angle, p.color, tex);
        }
    }
}
