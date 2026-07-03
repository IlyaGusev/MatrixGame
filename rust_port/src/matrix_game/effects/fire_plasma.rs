//! Port of `MatrixEffectFirePlasma.cpp` — the plasma-cannon bolt:
//! flight + hit delivery, the 5-sprite bolt visuals (PLASMA1 line +
//! 4 PLASMA2 billboards) and the impact smoke / scorch spot.

use crate::matrix_game::effects::weapon::{weapon_hit, WeaponId, FEHF_LASTHIT};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{ObjectId, Objects};
use crate::matrix_game::map_trace::{trace, TraceStop};
use glam::Vec3;

pub struct FirePlasma {
    pos: Vec3,
    dir: Vec3,
    end: Vec3,
    prev_dist: f32,
    /// `m_Speed` — 0.5 units per ms (MatrixEffectWeapon.cpp:308).
    speed: f32,
    hitmask: u32,
    skip: Option<ObjectId>,
    pub weapon: WeaponId,
}

impl FirePlasma {
    pub fn new(
        start: Vec3,
        end: Vec3,
        speed: f32,
        hitmask: u32,
        skip: Option<ObjectId>,
        weapon: WeaponId,
    ) -> Self {
        Self {
            pos: start,
            dir: (end - start).normalize_or_zero(),
            end,
            prev_dist: (end - start).length(),
            speed,
            hitmask,
            skip,
            weapon,
        }
    }

    /// `CMatrixEffectFirePlasma::Takt` (MatrixEffectFirePlasma.cpp:74-137).
    /// Returns false when the bolt finished (caller drops it +
    /// releases the weapon ref).
    pub fn takt(&mut self, step: f32, map: &GameMap, objs: &mut Objects) -> bool {
        let dtime = self.speed * step;
        let newpos = self.pos + self.dir * dtime;

        let (hito, mut hitpos) = trace(map, objs, self.pos, newpos, self.hitmask, self.skip);
        let mut hit = hito != TraceStop::None;

        self.pos = newpos;

        // Overshot the target point — finish at the endpoint.
        let newdist = (self.end - self.pos).length();
        if newdist > self.prev_dist {
            hit = true;
            hitpos = self.end;
        }
        self.prev_dist = newdist;

        if hit {
            weapon_hit(objs, self.weapon, hito, hitpos, FEHF_LASTHIT, 0.0);
            // Impact visuals (MatrixEffectFirePlasma.cpp:109-126).
            use crate::matrix_game::effects::landscape_spot::{SpotKind, SpotSpawn};
            use crate::matrix_game::effects::smoke_and_fire::Smoke;
            use crate::matrix_game::effects::GameEffect;
            match hito {
                TraceStop::Landscape => {
                    objs.pending_spots.push(SpotSpawn {
                        pos: glam::Vec2::new(hitpos.x, hitpos.y),
                        angle: (hitpos.x + hitpos.y).fract() * std::f32::consts::PI,
                        scale: 0.5 + (hitpos.x * 0.37 + hitpos.y * 0.61).fract(),
                        kind: SpotKind::PlasmaHit,
                    });
                    objs.pending_effects.push(GameEffect::Smoke(Smoke::new(
                        hitpos,
                        100.0,
                        1400.0,
                        41.0,
                        0xFF90_9090,
                        false,
                        0.04,
                    )));
                }
                TraceStop::Water => {
                    objs.pending_effects.push(GameEffect::Smoke(Smoke::new(
                        hitpos,
                        100.0,
                        1400.0,
                        41.0,
                        0xFFFF_FFFF,
                        false,
                        0.04,
                    )));
                }
                _ => {
                    objs.pending_effects.push(GameEffect::Smoke(Smoke::new(
                        hitpos,
                        100.0,
                        1400.0,
                        41.0,
                        0xFF3F_2F2F,
                        false,
                        0.04,
                    )));
                }
            }
            // Kill the follow light (MatrixEffectFirePlasma.cpp Kill on hit).
            objs.pending_light_kill.push(self.weapon);
            return false;
        }
        // Bolt glow that follows the projectile (CreatePointLight,
        // MatrixEffectFirePlasma.cpp:29 — radius 20, colour 0x202030).
        objs.pending_light_follow
            .push((self.weapon, [self.pos.x, self.pos.y, self.pos.z], 20.0, 0x0020_2030));
        true
    }

    /// Bolt sprites (MatrixEffectFirePlasma.cpp:15-36, 132-136):
    /// PLASMA1 line down the axis + 4 PLASMA2 pulses along it.
    pub fn draw(&self, q: &mut crate::matrix_lib::three_g::billboard::BillboardQueue) {
        use crate::matrix_lib::three_g::billboard::{
            TexRef, BBT_PLASMA1, BBT_PLASMA2, BBT_POINTLIGHT,
        };
        const FIRE_PLASMA_W: f32 = 3.0;
        const FIRE_PLASMA_L: f32 = 7.0;
        // Glow sprite riding the bolt (BBT_POINTLIGHT r=20, 0x80202030,
        // MatrixEffectFirePlasma.cpp:29).
        q.billboard(self.pos, 20.0, 0.0, 0x8020_2030, TexRef::Bbt(BBT_POINTLIGHT));
        q.line(
            self.pos - self.dir * FIRE_PLASMA_L * 0.7,
            self.pos + self.dir * FIRE_PLASMA_L * 0.7,
            FIRE_PLASMA_W * 1.4,
            0xFFFF_FFFF,
            TexRef::Bbt(BBT_PLASMA1),
        );
        for k in [-0.5f32, -0.166, 0.166, 0.5] {
            q.billboard(
                self.pos + self.dir * FIRE_PLASMA_L * k,
                FIRE_PLASMA_W,
                0.0,
                0xFFFF_FFFF,
                TexRef::Bbt(BBT_PLASMA2),
            );
        }
    }
}
