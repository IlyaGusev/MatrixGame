//! Port of `MatrixEffectMovingObject.cpp` — the ballistic projectiles:
//! gun / turret shells (`MO_Gun_Takt`, `MO_Gun_cannon_Takt`), homing
//! missiles (`MO_Homing_Missile_Takt`) and mortar bombs
//! (`MO_Bomb_Takt`), including the projectile meshes, smoke trails,
//! billboard tracers and impact explosion / spot spawns.
//!
//! `CMatrixEffect::m_Dist2` (the squared distance from the muzzle that
//! `WeaponHit` range-checks against) is passed explicitly to
//! [`weapon_hit`] instead of through a static.

use crate::matrix_game::common::{TRACE_BUILDING, TRACE_CANNON, TRACE_FLYER, TRACE_ROBOT};
use crate::matrix_game::effects::weapon::{weapon_hit, WeaponId, FEHF_LASTHIT};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{Control, ObjectId, ObjectType, Objects};
use crate::matrix_game::map_trace::{trace, TraceStop};
use crate::matrix_lib::three_g::math3d::Trajectory;
use glam::Vec3;

/// Constants from MatrixEffectMovingObject.hpp:11-22.
const HOMING_RADIUS: f32 = 1000.0;
const HM_SEEK_TIME_PERIOD: f32 = 50.0;
const MISSILE_IMPACT_RADIUS: f32 = 18.0;
const MISSILE_IMPACT_RADIUS_SQ: f32 = MISSILE_IMPACT_RADIUS * MISSILE_IMPACT_RADIUS;
const BOMB_DAMAGE_RADIUS: f32 = 20.0;
const GUN_DAMAGE_RADIUS: f32 = 5.0;

/// Per-kind state out of the `SMOProps::common` union.
pub enum MoKind {
    /// `MO_Gun_Takt` — robot gun shell, velocity `dir * 22`.
    Gun { maxdist: f32, dist: f32 },
    /// `MO_Gun_cannon_Takt` — turret shell, velocity `dir * 27`.
    /// Identical gameplay to `Gun` (the C++ bodies differ only in
    /// tracer visuals).
    GunCannon { maxdist: f32, dist: f32 },
    /// `MO_Homing_Missile_Takt`.
    Missile {
        next_seek_time: f32,
        target: Option<ObjectId>,
    },
    /// `MO_Bomb_Takt` — `vel_scale` is the C++ `velocity.x` trick
    /// (parametric speed along the trajectory).
    Bomb {
        trajectory: Trajectory,
        pos: f32,
        vel_scale: f32,
    },
}

impl MoKind {
    pub fn new_gun(maxdist: f32) -> Self {
        MoKind::Gun { maxdist, dist: 0.0 }
    }
    pub fn new_gun_cannon(maxdist: f32) -> Self {
        MoKind::GunCannon { maxdist, dist: 0.0 }
    }
    pub fn new_missile() -> Self {
        MoKind::Missile {
            next_seek_time: 0.0,
            target: None,
        }
    }
    pub fn new_bomb(points: &[Vec3], vel_scale: f32) -> Self {
        MoKind::Bomb {
            trajectory: Trajectory::new(points),
            pos: 0.0,
            vel_scale,
        }
    }
}

/// Port of `SMOProps` + `CMatrixEffectMovingObject` (gameplay subset).
pub struct MovingObject {
    start_pos: Vec3,
    cur_pos: Vec3,
    velocity: Vec3,
    target: Vec3,
    time: f32,
    side: i32,
    hitmask: u32,
    skip: Option<ObjectId>,
    pub weapon: WeaponId,
    kind: MoKind,
    /// `next_fire_time` — trail emission clock (smoke/fire puffs for
    /// missile/bomb, tracer lines for guns).
    next_fire_time: f32,
    /// Orientation rows for the mesh draw (`VecToMatrixY`).
    rows: [Vec3; 4],
}

impl MovingObject {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        start: Vec3,
        target: Vec3,
        velocity: Vec3,
        side: i32,
        hitmask: u32,
        skip: Option<ObjectId>,
        weapon: WeaponId,
        kind: MoKind,
    ) -> Self {
        Self {
            start_pos: start,
            cur_pos: start,
            velocity,
            target,
            time: 0.0,
            side,
            hitmask,
            skip,
            weapon,
            kind,
            next_fire_time: 0.0,
            rows: crate::matrix_lib::three_g::math3d::vec_to_matrix_y(
                start,
                velocity.normalize_or_zero(),
            ),
        }
    }

    /// Emit the missile/bomb smoke+fire trail along this frame's path
    /// (MatrixEffectMovingObject.cpp:281-313 / :407-435).
    fn emit_trail(&mut self, newpos: Vec3, objs: &mut Objects, period: f32, with_fire: bool) {
        use crate::matrix_game::effects::smoke_and_fire::{Fire, Smoke};
        use crate::matrix_game::effects::GameEffect;
        if self.time <= self.next_fire_time {
            return;
        }
        let span = self.time - self.next_fire_time;
        let dt = period / span.max(1e-6);
        let mut t = 0.0f32;
        while self.time > self.next_fire_time {
            self.next_fire_time += period;
            let p = (newpos - self.cur_pos) * t + self.cur_pos;
            t += dt;
            if self.time < 200.0 {
                objs.pending_effects.push(GameEffect::Smoke(Smoke::new(
                    p,
                    300.0,
                    300.0,
                    400.0,
                    0xFFFF_FFFF,
                    true,
                    0.04,
                )));
            } else {
                objs.pending_effects.push(GameEffect::Smoke(Smoke::new(
                    p,
                    300.0,
                    600.0,
                    if with_fire { 800.0 } else { 400.0 },
                    0xFFFF_FFFF,
                    true,
                    0.04,
                )));
                if with_fire {
                    objs.pending_effects.push(GameEffect::Fire(Fire::new(
                        p, 300.0, 300.0, 800.0, 1.0, true, 0.04,
                    )));
                }
            }
        }
    }

    /// Impact explosion + crater for the shells/missiles
    /// (MatrixEffectMovingObject.cpp:344-360 etc).
    fn impact_visuals(&self, hito: TraceStop, hitpos: Vec3, map: &GameMap, objs: &mut Objects) {
        use crate::matrix_game::effects::landscape_spot::{SpotKind, SpotSpawn};
        if hito == TraceStop::Water {
            // Bomb on deep water: nothing (gun/missile don't splash here).
            let z = map.get_z(hitpos.x, hitpos.y);
            if z <= crate::matrix_game::common::WATER_LEVEL {
                return;
            }
        }
        let mut pos = hitpos;
        let fire = if hito == TraceStop::Landscape {
            objs.pending_spots.push(SpotSpawn {
                pos: glam::Vec2::new(hitpos.x, hitpos.y),
                angle: (hitpos.x + hitpos.y).fract() * std::f32::consts::PI,
                scale: 6.0 + (hitpos.x * 0.43 + hitpos.y * 0.29).fract() * 3.0,
                kind: SpotKind::Voronka,
            });
            pos.z = map.get_z(hitpos.x, hitpos.y) + 10.0;
            true
        } else {
            false
        };
        objs.pending_explosions
            .push(crate::matrix_game::map_static::ExplosionSpawn {
                pos,
                props: &crate::matrix_game::effects::explosion::EXPLOSION_MISSILE,
                fire,
            });
    }

    /// `CMatrixEffectMovingObject::Takt` (MatrixEffectMovingObject.cpp:
    /// 112-128): advance time, run the kind handler. Returns false at
    /// end of life.
    pub fn takt(&mut self, step: f32, map: &GameMap, objs: &mut Objects) -> bool {
        self.time += step;
        match &self.kind {
            MoKind::Gun { .. } | MoKind::GunCannon { .. } => self.gun_takt(step, map, objs),
            MoKind::Missile { .. } => self.missile_takt(step, map, objs),
            MoKind::Bomb { .. } => self.bomb_takt(step, map, objs),
        }
    }

    /// `MOEnum` (MatrixEffectMovingObject.cpp:193-202): every object in
    /// `radius` around `center` gets the weapon hit callback with
    /// `m_Dist2 = |center - startpos|²` and flags 0.
    fn mo_enum(&self, center: Vec3, radius: f32, objs: &mut Objects) -> bool {
        let mut hits: Vec<ObjectId> = Vec::new();
        objs.find_objects_3d(center, radius, 1.0, self.hitmask, self.skip, |_, id| {
            hits.push(id);
            Control::Continue
        });
        let dist2 = (center - self.start_pos).length_squared();
        let any = !hits.is_empty();
        for id in hits {
            weapon_hit(objs, self.weapon, TraceStop::Object(id), center, 0, dist2);
        }
        any
    }

    /// `MO_Gun_Takt` / `MO_Gun_cannon_Takt`
    /// (MatrixEffectMovingObject.cpp:504-679).
    fn gun_takt(&mut self, step: f32, map: &GameMap, objs: &mut Objects) -> bool {
        let dtime = 0.1 * step;
        let vel = self.velocity.length();
        let newpos = self.cur_pos + self.velocity * dtime;

        let (maxdist, dist) = match &mut self.kind {
            MoKind::Gun { maxdist, dist } | MoKind::GunCannon { maxdist, dist } => {
                *dist += vel * dtime;
                (*maxdist, *dist)
            }
            _ => unreachable!(),
        };

        let mut hit = false;
        if newpos.z - map.get_z(newpos.x, newpos.y) < GUN_DAMAGE_RADIUS + 10.0 {
            hit = self.mo_enum(newpos, GUN_DAMAGE_RADIUS, objs);
        }

        let (hito, hitpos) = trace(map, objs, self.cur_pos, newpos, self.hitmask, self.skip);
        if hito != TraceStop::None {
            hit = true;
        }
        // Tracer line (MatrixEffectMovingObject.cpp:546 / :642): SHLEIF
        // texture, robot gun 6px 0x8FFFFFFF / turret 4px 0x6FFFFFFF.
        {
            use crate::matrix_game::effects::billboard_fx::BillboardLineFx;
            use crate::matrix_game::effects::GameEffect;
            use crate::matrix_lib::three_g::billboard::{TexRef, BBT_SHLEIF};
            let (wd, col, ttl) = if matches!(self.kind, MoKind::GunCannon { .. }) {
                (4.0, 0x6FFF_FFFF, 500.0)
            } else {
                (6.0, 0x8FFF_FFFF, 1000.0)
            };
            objs.pending_effects
                .push(GameEffect::BillboardLine(BillboardLineFx::new(
                    self.cur_pos,
                    hitpos,
                    wd,
                    col,
                    col & 0x00FF_FFFF,
                    ttl,
                    TexRef::Bbt(BBT_SHLEIF),
                )));
        }
        self.cur_pos = hitpos;
        self.rows = crate::matrix_lib::three_g::math3d::vec_to_matrix_y(
            self.cur_pos,
            self.velocity.normalize_or_zero(),
        );

        if hit {
            let dist2 = (hitpos - self.start_pos).length_squared();
            weapon_hit(objs, self.weapon, hito, hitpos, FEHF_LASTHIT, dist2);
            self.impact_visuals(hito, hitpos, map, objs);
            false
        } else if dist > maxdist {
            let dist2 = (hitpos - self.start_pos).length_squared();
            weapon_hit(
                objs,
                self.weapon,
                TraceStop::None,
                hitpos,
                FEHF_LASTHIT,
                dist2,
            );
            false
        } else {
            true
        }
    }

    /// `MO_Homing_Missile_Takt` (MatrixEffectMovingObject.cpp:206-384).
    fn missile_takt(&mut self, step: f32, map: &GameMap, objs: &mut Objects) -> bool {
        let dtime = 0.1 * step;

        let dir = self.velocity.normalize_or_zero();
        let mut to_tgt = dir;

        let (do_seek, cur_target) = match &mut self.kind {
            MoKind::Missile {
                next_seek_time,
                target,
            } => {
                // Drop dead targets — the SObjectCore tombstone check.
                if let Some(t) = *target {
                    if !objs.is_valid(t) {
                        *target = None;
                    }
                }
                if self.time > *next_seek_time {
                    *next_seek_time = HM_SEEK_TIME_PERIOD + self.time;
                    (true, *target)
                } else {
                    (false, *target)
                }
            }
            _ => unreachable!(),
        };

        if do_seek {
            // HMEnum (MatrixEffectMovingObject.cpp:148-191).
            let mut maxcos = if cur_target.is_some() {
                dir.dot((self.target - self.cur_pos).normalize_or_zero())
            } else {
                0.0
            };
            let seekcenter = self.cur_pos + dir * HOMING_RADIUS;

            struct Cand {
                id: ObjectId,
                center: Vec3,
                kind: ObjectType,
                side: i32,
            }
            let mut cands: Vec<Cand> = Vec::new();
            objs.find_objects_3d(
                seekcenter,
                HOMING_RADIUS,
                1.0,
                TRACE_BUILDING | TRACE_ROBOT | TRACE_CANNON | TRACE_FLYER,
                self.skip,
                |_, id| {
                    if let Some(o) = objs.get(id) {
                        cands.push(Cand {
                            id,
                            center: o.core().geo_center,
                            kind: o.core().obj_type,
                            side: o.side(),
                        });
                    }
                    Control::Continue
                },
            );

            let target_kind = cur_target
                .and_then(|t| objs.get(t))
                .map(|o| o.core().obj_type);
            let mut new_target = cur_target;
            let mut found = false;
            for c in cands {
                if c.side == self.side {
                    continue; // no need friendly object
                }
                let cur_is_unit = matches!(
                    target_kind,
                    Some(ObjectType::Flyer | ObjectType::RobotAi | ObjectType::Cannon)
                );
                if cur_is_unit && c.kind == ObjectType::Building {
                    continue; // no need building if robot located
                }
                let select_new = target_kind == Some(ObjectType::Building)
                    && matches!(
                        c.kind,
                        ObjectType::Flyer | ObjectType::RobotAi | ObjectType::Cannon
                    );
                let to_dir = (c.center - self.cur_pos).normalize_or_zero();
                let cc = dir.dot(to_dir);
                // cos(10°) forward cone.
                if cc > 0.984_807_75 && (cc > maxcos || select_new || Some(c.id) == cur_target) {
                    let (t, _) = trace(
                        map,
                        objs,
                        self.cur_pos,
                        c.center + (c.center - self.cur_pos) * 0.1,
                        crate::matrix_game::common::TRACE_ALL,
                        self.skip,
                    );
                    if t == TraceStop::None || t.object() == Some(c.id) {
                        new_target = Some(c.id);
                        to_tgt = to_dir;
                        maxcos = cc;
                        found = true;
                    }
                }
            }
            if !found {
                new_target = None;
            } else if let Some(t) = new_target {
                if let Some(o) = objs.get(t) {
                    self.target = o.core().geo_center;
                }
            }
            if let MoKind::Missile { target, .. } = &mut self.kind {
                *target = new_target;
            }
        }

        self.velocity += to_tgt * 0.1 * dtime;
        // NOTE: the C++ "clamp to 1000" at MatrixEffectMovingObject.cpp:
        // 258-261 only rescales a local copy of the magnitude — the
        // actual velocity is never clamped. Faithfully unclamped.

        let newpos = self.cur_pos + self.velocity * dtime;

        // Trail — MISSILE_FIRE_PERIOD = 20ms, smoke + fire after 200ms.
        self.emit_trail(newpos, objs, 20.0, true);

        let (hito, hitpos) = trace(map, objs, self.cur_pos, newpos, self.hitmask, self.skip);
        let mut hit = hito != TraceStop::None;
        self.cur_pos = hitpos;
        self.rows = crate::matrix_lib::three_g::math3d::vec_to_matrix_y(
            self.cur_pos,
            self.velocity.normalize_or_zero(),
        );

        let distance = (self.cur_pos - self.target).length_squared();
        if hit || distance < MISSILE_IMPACT_RADIUS_SQ {
            hit |= self.mo_enum(self.cur_pos, MISSILE_IMPACT_RADIUS, objs);
            if hit {
                self.impact_visuals(hito, self.cur_pos, map, objs);
            }

            let dist2 = (self.cur_pos - self.start_pos).length_squared();
            // The C++ passes TRACE_STOP_NONE here — direct-hit damage
            // was already delivered by the radius pass above.
            weapon_hit(
                objs,
                self.weapon,
                TraceStop::None,
                self.cur_pos,
                FEHF_LASTHIT,
                dist2,
            );
            false
        } else if self.time > 10_000.0 {
            let dist2 = (self.cur_pos - self.start_pos).length_squared();
            weapon_hit(objs, self.weapon, hito, self.cur_pos, FEHF_LASTHIT, dist2);
            false
        } else {
            true
        }
    }

    /// `MO_Bomb_Takt` (MatrixEffectMovingObject.cpp:386-502).
    fn bomb_takt(&mut self, step: f32, map: &GameMap, objs: &mut Objects) -> bool {
        let newpos = match &mut self.kind {
            MoKind::Bomb {
                trajectory,
                pos,
                vel_scale,
            } => {
                *pos += *vel_scale * step;
                trajectory.calc_point(*pos)
            }
            _ => unreachable!(),
        };

        // Trail — BOMB_FIRE_PERIOD = 15ms, smoke only.
        self.emit_trail(newpos, objs, 15.0, false);

        let mut hit = false;
        if newpos.z - map.get_z(newpos.x, newpos.y) < BOMB_DAMAGE_RADIUS + 10.0 {
            hit = self.mo_enum(newpos, BOMB_DAMAGE_RADIUS, objs);
        }

        let (hito, hitpos) = trace(map, objs, self.cur_pos, newpos, self.hitmask, self.skip);
        if hito != TraceStop::None {
            hit = true;
        }

        // Trajectory endpoint reached → detonate (IsVec3Equal, :458).
        if let MoKind::Bomb { trajectory, .. } = &self.kind {
            let last = trajectory.calc_point(1.0);
            if (last - newpos).abs().max_element() < 0.001 {
                hit = true;
            }
        }

        let dir = (newpos - self.cur_pos).normalize_or_zero();
        self.cur_pos = newpos;
        self.rows = crate::matrix_lib::three_g::math3d::vec_to_matrix_y(self.cur_pos, dir);

        if hit {
            let dist2 = (hitpos - self.start_pos).length_squared();
            weapon_hit(objs, self.weapon, hito, hitpos, FEHF_LASTHIT, dist2);
            self.impact_visuals(hito, hitpos, map, objs);
            false
        } else {
            true
        }
    }

    /// Mesh draw — roket / mina / bullet VO at the `VecToMatrixY`
    /// orientation (MatrixEffectMovingObject.cpp:112-128).
    pub fn draw(&self, meshes: &mut crate::matrix_game::effects::explosion::MeshQueue) {
        use crate::matrix_game::effects::explosion::{MeshDraw, MeshKind};
        let kind = match self.kind {
            MoKind::Missile { .. } => MeshKind::Rocket,
            MoKind::Bomb { .. } => MeshKind::Mina,
            MoKind::Gun { .. } | MoKind::GunCannon { .. } => MeshKind::Bullet,
        };
        meshes.draws.push(MeshDraw {
            kind,
            rows: self.rows,
            color: 0xFFFF_FFFF,
        });
    }
}
