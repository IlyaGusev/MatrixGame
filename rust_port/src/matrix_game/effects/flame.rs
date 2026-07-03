//! Port of `MatrixEffectFlame.cpp` — the flamethrower's puff stream:
//! puff movement, terrain/water interaction, the 20ms proximity damage
//! sweep, the LASTHIT on stream death, and the billboard chains.
//! Point lights are intentionally dropped (see render pipeline notes).

use crate::matrix_game::common::{TRACE_ANYOBJECT, TRACE_LANDSCAPE, TRACE_WATER, WATER_LEVEL};
use crate::matrix_game::effects::weapon::{
    dispatch_fire_end_handler, weapon_hit, WeaponId, FEHF_LASTHIT,
};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{Control, ObjectId, ObjectType, Objects};
use crate::matrix_game::map_trace::TraceStop;
use glam::Vec3;

/// Constants from MatrixEffectFlame.hpp:11-17.
const FLAME_SCALE_FACTOR: f32 = 3.6;
const FLAME_DIR_SPEED: f32 = 0.36;
const FLAME_TIME_SEEK_PERIOD: f32 = 20.0;
const FLAME_NUM_BILLS: usize = 10;

/// `CFlamePuff`.
struct FlamePuff {
    time: f32,
    pos: Vec3,
    dir: Vec3,
    speed: Vec3,
    next_seek_time: f32,
    /// Smoke-trail clock (`m_NextSmokeTime`, MatrixEffectFlame.cpp:145-156).
    next_smoke_time: f32,
    scale: f32,
    /// Visuals: 10 trailing FLAME billboards + per-bill spin signs
    /// (MatrixEffectFlame.cpp:42-52, 228-262).
    bills: [Vec3; FLAME_NUM_BILLS],
    signs: [f32; FLAME_NUM_BILLS],
    cur_alpha: f32,
}

/// `CMatrixEffectFlame`.
pub struct Flame {
    /// `m_TTL` — puff lifetime, `weapon_dist * 8.333`
    /// (MatrixEffectWeapon.cpp:649).
    ttl: f32,
    hitmask: u32,
    skip: Option<ObjectId>,
    pub weapon: WeaponId,
    puffs: Vec<FlamePuff>,
    /// First takt hasn't run yet — don't die before the initial puff
    /// arrives from the weapon's queue.
    started: bool,
}

impl Flame {
    pub fn new(ttl: f32, hitmask: u32, skip: Option<ObjectId>, weapon: WeaponId) -> Self {
        Self {
            ttl,
            hitmask,
            skip,
            weapon,
            puffs: Vec::new(),
            started: false,
        }
    }

    /// `CMatrixEffectFlame::Takt` (MatrixEffectFlame.cpp:375-397) +
    /// `CFlamePuff::Takt` (:133-285). Returns false when the stream is
    /// out of puffs — the dtor then fires the handler with
    /// `FEHF_LASTHIT` (MatrixEffectFlame.cpp:315).
    pub fn takt(&mut self, step: f32, map: &GameMap, objs: &mut Objects) -> bool {
        // Drain puffs the owning weapon queued via `AddPuff`.
        if let Some(w) = objs.weapons.get_mut(self.weapon) {
            for p in w.pending_puffs.drain(..) {
                let mut signs = [1.0f32; FLAME_NUM_BILLS];
                for (i, sgn) in signs.iter_mut().enumerate() {
                    // FRND(1) > 0.5 sign pick — alternating is
                    // visually equivalent and RNG-stream-neutral.
                    if i & 1 == 1 {
                        *sgn = -1.0;
                    }
                }
                self.puffs.push(FlamePuff {
                    time: 0.0,
                    pos: p.pos,
                    dir: p.dir,
                    speed: p.speed,
                    next_seek_time: 0.0,
                    next_smoke_time: 0.0,
                    scale: 0.0,
                    bills: [p.pos; FLAME_NUM_BILLS],
                    signs,
                    cur_alpha: 0.0,
                });
            }
        }
        self.started = self.started || !self.puffs.is_empty();
        if !self.started {
            return true;
        }

        let ttl = self.ttl;
        let mut i = 0;
        while i < self.puffs.len() {
            let alive = {
                let p = &mut self.puffs[i];
                p.time += step;
                p.time <= ttl
            };
            if !alive {
                // remove (not swap_remove) to keep spawn order — the
                // FLAMELINE chain links each puff to its predecessor.
                self.puffs.remove(i);
                continue;
            }
            self.puff_takt(i, step, map, objs);
            i += 1;
        }

        if self.puffs.is_empty() {
            // ~CMatrixEffectFlame → handler(TRACE_STOP_NONE, 0, LASTHIT).
            let handler = objs
                .weapons
                .get(self.weapon)
                .map(|w| w.handler)
                .unwrap_or(crate::matrix_game::effects::weapon::WeaponHandler::None);
            dispatch_fire_end_handler(objs, handler, TraceStop::None, Vec3::ZERO, FEHF_LASTHIT);
            // Let the weapon spawn a fresh flame on the next burst.
            if let Some(w) = objs.weapons.get_mut(self.weapon) {
                w.flame_alive = false;
            }
            objs.pending_light_kill.push(self.weapon);
            return false;
        }
        // Flame glow following the leading puff (CreatePointLight,
        // MatrixEffectFlame.cpp:31 — radius 20, white).
        if let Some(p) = self.puffs.last() {
            objs.pending_light_follow
                .push((self.weapon, [p.pos.x, p.pos.y, p.pos.z], 20.0, 0x00FF_FFFF));
        }
        true
    }

    fn puff_takt(&mut self, idx: usize, step: f32, map: &GameMap, objs: &mut Objects) {
        let ttl_inv = 1.0 / self.ttl;
        let (pos, scale, do_seek) = {
            let p = &mut self.puffs[idx];
            let k = p.time * ttl_inv;
            let scale = k * FLAME_SCALE_FACTOR + 1.5;
            let mk = 1.0 - k;

            // Smoke trail after 30% of puff life, one puff per 100ms
            // (MatrixEffectFlame.cpp:143-156).
            let mut smoke_count = 0u32;
            if k > 0.3 {
                while p.time > p.next_smoke_time {
                    p.next_smoke_time += 100.0;
                    smoke_count += 1;
                }
            } else {
                p.next_smoke_time = p.time;
            }

            let oldpos = p.pos;
            p.pos += p.speed * step + step * p.dir * mk * mk * FLAME_DIR_SPEED;
            p.scale = scale * scale;

            let mut hit_env = false;
            let mut norm_z = 0.0f32;
            if self.hitmask & TRACE_WATER != 0 && (p.pos.z - p.scale) <= WATER_LEVEL {
                hit_env = true;
                norm_z += 1.0;
            }
            if self.hitmask & TRACE_LANDSCAPE != 0 {
                let z = map.get_z(p.pos.x, p.pos.y);
                if (p.pos.z - p.scale) <= z {
                    // Slide along the terrain (MatrixEffectFlame.cpp:
                    // 192-202): keep travel distance, lift z.
                    let l = (p.pos - oldpos).length();
                    p.pos.z = z + p.scale;
                    let n = (p.pos - oldpos).normalize_or_zero();
                    p.pos = oldpos + n * l;
                }
            }

            let do_seek = if p.time > p.next_seek_time {
                p.next_seek_time = p.time + FLAME_TIME_SEEK_PERIOD;
                self.hitmask & TRACE_ANYOBJECT != 0
            } else {
                false
            };
            // Water push-back happens with the object push-back below;
            // stash z-norm in the seek pass through `norm_seed`.
            let _ = hit_env;
            (p.pos, p.scale, (do_seek, norm_z, smoke_count))
        };
        let (do_seek, norm_seed_z, smoke_count) = do_seek;

        // Emit the queued smoke puffs at the current position.
        for _ in 0..smoke_count {
            objs.pending_effects.push(
                crate::matrix_game::effects::GameEffect::Smoke(
                    crate::matrix_game::effects::smoke_and_fire::Smoke::new(
                        pos, 300.0, 700.0, 400.0, 0x2F30_3030, false, 0.0,
                    ),
                ),
            );
        }

        // Water hit notification (MatrixEffectFlame.cpp:181-188).
        if norm_seed_z > 0.0 {
            weapon_hit(
                objs,
                self.weapon,
                TraceStop::Water,
                Vec3::new(pos.x, pos.y, WATER_LEVEL),
                0,
                0.0,
            );
        }

        let mut norm = Vec3::new(0.0, 0.0, norm_seed_z);
        let mut any_hit = norm_seed_z > 0.0;
        if do_seek {
            // FlameEnum (MatrixEffectFlame.cpp:118-131): norm
            // accumulates over non-robot victims as objects are
            // visited; each victim's hit point uses the norm so far.
            struct Victim {
                id: ObjectId,
                center: Vec3,
                radius: f32,
                is_robot: bool,
            }
            let mut victims: Vec<Victim> = Vec::new();
            objs.find_objects_3d(pos, scale, 0.7, self.hitmask, self.skip, |_, id| {
                if let Some(o) = objs.get(id) {
                    victims.push(Victim {
                        id,
                        center: o.core().geo_center,
                        radius: o.core().radius,
                        is_robot: matches!(o.core().obj_type, ObjectType::RobotAi),
                    });
                }
                Control::Continue
            });
            any_hit |= !victims.is_empty();
            for v in victims {
                if !v.is_robot {
                    norm += (pos - v.center).normalize_or_zero();
                }
                weapon_hit(
                    objs,
                    self.weapon,
                    TraceStop::Object(v.id),
                    pos + norm * v.radius,
                    0,
                    0.0,
                );
            }
        }

        if any_hit {
            let n = norm.normalize_or_zero();
            let p = &mut self.puffs[idx];
            let k = p.time * ttl_inv;
            let lin_scale = k * FLAME_SCALE_FACTOR + 1.5;
            p.pos += n * lin_scale * (0.02 * step);
        }

        // Visual chain update (MatrixEffectFlame.cpp:218-256): bill 0
        // pins to the puff; each follower trails the previous within
        // `scale * k1`.
        {
            let p = &mut self.puffs[idx];
            let k = p.time * ttl_inv;
            let k1 = 1.0 - ((k - 0.5) / 0.5).clamp(0.0, 1.0);
            let lin_scale = k * FLAME_SCALE_FACTOR + 1.5;
            p.cur_alpha = 255.0 * k1 * k1;
            p.bills[0] = p.pos;
            for i in 1..FLAME_NUM_BILLS {
                let delta = p.bills[i - 1] - p.bills[i];
                let x = delta.length();
                let lim = lin_scale * k1;
                if x > lim && x > 1e-6 {
                    p.bills[i] += delta * ((x - lim) / x);
                }
            }
        }
    }

    /// Draw — 10 FLAME billboards per puff with the blue-tinged tail
    /// gradient (MatrixEffectFlame.cpp:228-262).
    pub fn draw(&self, q: &mut crate::matrix_lib::three_g::billboard::BillboardQueue) {
        use crate::matrix_lib::three_g::billboard::{TexRef, BBT_FLAME, BBT_FLAMELINE};
        // FLAMELINE connecting each puff to its predecessor, forming a
        // continuous stream (MatrixEffectFlame.cpp:272-283). Alpha dims
        // when the gap is under scale*2.
        for w in self.puffs.windows(2) {
            let (prev, cur) = (&w[0], &w[1]);
            let width = cur.scale * 2.0;
            let l = (prev.pos - cur.pos).length();
            let a = if l < width && width > 1e-6 {
                cur.cur_alpha * (l * 0.5 / cur.scale)
            } else {
                cur.cur_alpha
            };
            let color = ((a as u32) << 24) | 0x00FF_FFFF;
            q.line(prev.pos, cur.pos, width, color, TexRef::Bbt(BBT_FLAMELINE));
        }
        for p in &self.puffs {
            let angle_base = std::f32::consts::TAU / 1024.0 * ((p.time as i32 & 1023) as f32);
            for i in 0..FLAME_NUM_BILLS {
                let angle = angle_base * p.signs[i];
                let color = if i == 0 {
                    ((p.cur_alpha as u32) << 24) | 0x00FF_FFFF
                } else {
                    let kk = 1.0 - i as f32 / FLAME_NUM_BILLS as f32;
                    let rg = (kk * 128.0) as u32;
                    let b = ((kk * 512.0) as u32).min(255);
                    ((p.cur_alpha as u32) << 24) | (rg << 16) | (rg << 8) | b
                };
                q.billboard(p.bills[i], p.scale, angle, color, TexRef::Bbt(BBT_FLAME));
            }
        }
    }
}
