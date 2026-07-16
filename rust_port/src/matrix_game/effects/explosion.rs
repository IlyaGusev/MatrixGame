//! Port of `MatrixEffectExplosion.{cpp,hpp}` — the explosion presets
//! and the spark / intense-sprite / fire-debris / mesh-debris
//! kinetics, plus `CMatrixEffectFireAnim` (the 8-frame flame loop
//! `CreateExplosion(..., fire=true)` adds).
//!
//! The per-preset point-light flash is ported via
//! `PointLightSystem::add_transient_light_anim` (queued through
//! `Objects::pending_point_lights`); the per-preset sound key is
//! queued when the spawn drains in `effects_takt`.

use glam::Vec3;

use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd, kscale, Fire, Smoke};
use crate::matrix_game::logic::Rnd;
use crate::matrix_game::map::GameMap;
use crate::matrix_lib::three_g::billboard::{
    BillboardQueue, QueuedBillboard, QueuedLine, TexRef, BBT_FLAMEFRAME0, BBT_INTENSE, BBT_SPARK,
};

/// Landscape-spot kinds an explosion can stamp (ESpotType subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpotType {
    None,
    Voronka,
    PlasmaHit,
}

/// Port of `SExplosionProperties` (MatrixEffectExplosion.hpp:23-52),
/// minus the light/sound fields (see module doc).
#[derive(Clone, Copy)]
pub struct ExplosionProps {
    pub min_speed: f32,
    pub max_speed: f32,
    pub min_speed_z: f32,
    pub max_speed_z: f32,
    pub sparks_deb_cnt: i32,
    pub fire_deb_cnt: i32,
    pub deb_cnt: i32,
    pub deb_min_ttl: i32,
    pub deb_max_ttl: i32,
    pub intense_cnt: i32,
    pub deb_type: i32,
    pub voronka: SpotType,
    pub voronka_scale: f32,
    /// Attached point light (MatrixEffectExplosion.cpp preset `light*`
    /// fields): phase 1 (`m_Time < light_time1`) LERPs radius `r1→r2` and
    /// colour `c1→c2`; phase 2 holds radius `r2` and fades colour→black
    /// over `(light_time2-light_time1)/DEBRIS_SPEED` ms (:636-646).
    pub light: bool,
    pub light_radius1: f32,
    pub light_radius2: f32,
    pub light_color1: u32,
    pub light_color2: u32,
    pub light_time1: f32,
    pub light_time2: f32,
    /// Sounds-block key of the preset's `sound` field
    /// (MatrixEffectExplosion.cpp:13-304); empty = S_NONE.
    pub sound: &'static str,
}

/// The preset instances (MatrixEffectExplosion.cpp:13-304).
pub const EXPLOSION_NORMAL: ExplosionProps = ExplosionProps {
    min_speed: 0.0,
    max_speed: 7.0,
    min_speed_z: -15.0,
    max_speed_z: 15.0,
    sparks_deb_cnt: 10,
    fire_deb_cnt: 7,
    deb_cnt: 200,
    deb_min_ttl: 1000,
    deb_max_ttl: 4000,
    intense_cnt: 6,
    deb_type: 0,
    voronka: SpotType::Voronka,
    voronka_scale: 20.0,
    light: false,
    light_radius1: 0.0,
    light_radius2: 0.0,
    light_color1: 0x00000000,
    light_color2: 0x00000000,
    light_time1: 0.0,
    light_time2: 0.0,
    sound: "expl_norm",
};
pub const EXPLOSION_MISSILE: ExplosionProps = ExplosionProps {
    min_speed: 0.0,
    max_speed: 7.0,
    min_speed_z: -15.0,
    max_speed_z: 25.0,
    sparks_deb_cnt: 10,
    fire_deb_cnt: 2,
    deb_cnt: 0,
    deb_min_ttl: 1000,
    deb_max_ttl: 4000,
    intense_cnt: 6,
    deb_type: 1,
    voronka: SpotType::None,
    voronka_scale: 0.0,
    light: true,
    light_radius1: 10.0,
    light_radius2: 100.0,
    light_color1: 0x00000000,
    light_color2: 0xFFFF6F33,
    light_time1: 1.5,
    light_time2: 8.0,
    sound: "expl_missile",
};
pub const EXPLOSION_ROBOT_HIT: ExplosionProps = ExplosionProps {
    min_speed: 0.0,
    max_speed: 5.0,
    min_speed_z: -5.0,
    max_speed_z: 5.0,
    sparks_deb_cnt: 5,
    fire_deb_cnt: 0,
    deb_cnt: 1,
    deb_min_ttl: 1000,
    deb_max_ttl: 2000,
    intense_cnt: 0,
    deb_type: 1,
    voronka: SpotType::None,
    voronka_scale: 0.0,
    light: false,
    light_radius1: 2.0,
    light_radius2: 30.0,
    light_color1: 0xFFFFFFFF,
    light_color2: 0x11FFFF11,
    light_time1: 1.5,
    light_time2: 3.0,
    sound: "expl_rh",
};
pub const EXPLOSION_LASER_HIT: ExplosionProps = ExplosionProps {
    min_speed: 0.0,
    max_speed: 5.0,
    min_speed_z: -5.0,
    max_speed_z: 5.0,
    sparks_deb_cnt: 10,
    fire_deb_cnt: 0,
    deb_cnt: 0,
    deb_min_ttl: 1000,
    deb_max_ttl: 2000,
    intense_cnt: 0,
    deb_type: 0,
    voronka: SpotType::None,
    voronka_scale: 0.0,
    light: true,
    light_radius1: 2.0,
    light_radius2: 10.0,
    light_color1: 0xFFFFFFFF,
    light_color2: 0x11200505,
    light_time1: 1.5,
    light_time2: 3.0,
    sound: "expl_lh",
};
pub const EXPLOSION_BUILDING_BOOM: ExplosionProps = ExplosionProps {
    min_speed: 0.0,
    max_speed: 10.0,
    min_speed_z: -5.0,
    max_speed_z: 10.0,
    sparks_deb_cnt: 2,
    fire_deb_cnt: 0,
    deb_cnt: 1,
    deb_min_ttl: 1000,
    deb_max_ttl: 2000,
    intense_cnt: 4,
    deb_type: 1,
    voronka: SpotType::None,
    voronka_scale: 0.0,
    light: false,
    light_radius1: 2.0,
    light_radius2: 10.0,
    light_color1: 0xFFFFFFFF,
    light_color2: 0x11FF3111,
    light_time1: 1.5,
    light_time2: 3.0,
    sound: "",
};
pub const EXPLOSION_BUILDING_BOOM2: ExplosionProps = ExplosionProps {
    min_speed: 10.0,
    max_speed: 15.0,
    min_speed_z: -5.0,
    max_speed_z: 15.0,
    sparks_deb_cnt: 20,
    fire_deb_cnt: 1,
    deb_cnt: 10,
    deb_min_ttl: 1000,
    deb_max_ttl: 2000,
    intense_cnt: 3,
    deb_type: 1,
    voronka: SpotType::None,
    voronka_scale: 0.0,
    light: false,
    light_radius1: 2.0,
    light_radius2: 10.0,
    light_color1: 0xFFFFFFFF,
    light_color2: 0x11FF3111,
    light_time1: 1.5,
    light_time2: 3.0,
    sound: "expl_bb2",
};
pub const EXPLOSION_ROBOT_BOOM: ExplosionProps = ExplosionProps {
    min_speed: 0.0,
    max_speed: 7.0,
    min_speed_z: 0.0,
    max_speed_z: 10.0,
    sparks_deb_cnt: 50,
    fire_deb_cnt: 3,
    deb_cnt: 30,
    deb_min_ttl: 1000,
    deb_max_ttl: 4000,
    intense_cnt: 6,
    deb_type: 1,
    voronka: SpotType::None,
    voronka_scale: 0.0,
    light: true,
    light_radius1: 10.0,
    light_radius2: 70.0,
    light_color1: 0x00000000,
    light_color2: 0xFFFF6F33,
    light_time1: 1.5,
    light_time2: 8.0,
    sound: "expl_rb",
};
pub const EXPLOSION_ROBOT_BOOM_SMALL: ExplosionProps = ExplosionProps {
    min_speed: 0.0,
    max_speed: 7.0,
    min_speed_z: 0.0,
    max_speed_z: 10.0,
    sparks_deb_cnt: 20,
    fire_deb_cnt: 2,
    deb_cnt: 10,
    deb_min_ttl: 800,
    deb_max_ttl: 2000,
    intense_cnt: 4,
    deb_type: 1,
    voronka: SpotType::None,
    voronka_scale: 0.0,
    light: true,
    light_radius1: 10.0,
    light_radius2: 50.0,
    light_color1: 0x00000000,
    light_color2: 0xFFFF6F33,
    light_time1: 1.5,
    light_time2: 8.0,
    sound: "expl_rbs",
};
pub const EXPLOSION_BIG_BOOM: ExplosionProps = ExplosionProps {
    min_speed: 0.0,
    max_speed: 25.0,
    min_speed_z: -25.0,
    max_speed_z: 25.0,
    sparks_deb_cnt: 1000,
    fire_deb_cnt: 0,
    deb_cnt: 400,
    deb_min_ttl: 1000,
    deb_max_ttl: 4000,
    intense_cnt: 16,
    deb_type: 1,
    voronka: SpotType::Voronka,
    voronka_scale: 12.0,
    light: true,
    light_radius1: 10.0,
    light_radius2: 70.0,
    light_color1: 0x00000000,
    light_color2: 0xFFFF6F33,
    light_time1: 1.5,
    light_time2: 8.0,
    sound: "expl_bigboom",
};
pub const EXPLOSION_OBJECT: ExplosionProps = ExplosionProps {
    min_speed: 0.0,
    max_speed: 7.0,
    min_speed_z: 0.0,
    max_speed_z: 10.0,
    sparks_deb_cnt: 20,
    fire_deb_cnt: 2,
    deb_cnt: 20,
    deb_min_ttl: 1000,
    deb_max_ttl: 2000,
    intense_cnt: 8,
    deb_type: 1,
    voronka: SpotType::Voronka,
    voronka_scale: 10.0,
    light: true,
    light_radius1: 10.0,
    light_radius2: 70.0,
    light_color1: 0x00000000,
    light_color2: 0xFFFF6F33,
    light_time1: 1.5,
    light_time2: 8.0,
    sound: "expl_obj",
};

/// Constants (MatrixEffectExplosion.hpp:15-21).
const EXPLOSION_SPARK_WIDTH: f32 = 3.0;
/// `DEBRIS_SPEED` = 1/100.
const DEBRIS_SPEED: f32 = 0.01;
const INTENSE_INIT_SIZE: f32 = 4.0;
const INTENSE_END_SIZE: f32 = 60.0;
const INTENSE_DISPLACE_RADIUS: f32 = 10.0;
const INTENSE_TTL: f32 = 1300.0;

/// `SGradient` ramps for the intense sprites
/// (MatrixEffectExplosion.cpp:720-740).
fn calc_gradient(t: f32, g: &[(f32, f32)]) -> f32 {
    for w in g.windows(2) {
        let (v0, t0) = w[0];
        let (v1, t1) = w[1];
        if t >= t0 && t < t1 {
            return v0 + (v1 - v0) * ((t - t0) / (t1 - t0));
        }
    }
    g.last().map(|&(v, _)| v).unwrap_or(0.0)
}

/// Follow-light key allocator for fire-debris embers (synthetic
/// WeaponId space; see `WeaponId::synthetic`).
static EMBER_KEY: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

enum Deb {
    Fire {
        fire: Fire,
        pos: Vec3,
        v: Vec3,
        ttl: f32,
        /// Terrain ember glow following the piece (r 20, 0x22222202,
        /// MatrixEffectExplosion.cpp:362/794).
        light_key: u32,
    },
    Intense {
        ttm: f32,
        ttl: f32,
        v: Vec3,
        pos: Vec3,
        angle: f32,
        scale: f32,
        disp: glam::Vec2,
        color: u32,
    },
    Spark {
        pos: Vec3,
        prepos: Vec3,
        v: Vec3,
        ttl: f32,
        unttl: f32,
    },
    Mesh {
        pos: Vec3,
        v: Vec3,
        ttl: f32,
        angley: f32,
        anglep: f32,
        angler: f32,
        index: usize,
        scale: f32,
    },
}

/// A mesh-debris draw request consumed by the effect mesh renderer.
pub struct MeshDraw {
    /// Index into the debris catalog (`Models/Debris`) or the bullet
    /// meshes — namespaced by [`MeshKind`].
    pub kind: MeshKind,
    /// Row-vector world matrix `[x, y, z, pos]`.
    pub rows: [Vec3; 4],
    /// Modulate color (ARGB).
    pub color: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKind {
    Debris(usize),
    Rocket,
    Mina,
    Bullet,
}

/// Per-frame mesh queue (debris pieces + projectiles).
#[derive(Default)]
pub struct MeshQueue {
    pub draws: Vec<MeshDraw>,
}

/// Port of `CMatrixEffectExplosion`.
pub struct Explosion {
    props: &'static ExplosionProps,
    time: f32,
    debs: Vec<Deb>,
}

impl Explosion {
    /// `CreateExplosion` (MatrixEffect.cpp:452-501): lifts the center
    /// to ≥ ground+10, seeds all debris classes, stamps the voronka.
    /// `debris_types` is the per-entry deb_type of the `Models/Debris`
    /// catalog; mesh debris is drawn only from entries matching the
    /// preset's `deb_type` (MatrixEffectExplosion.cpp:323-331, 448).
    pub fn new(
        pos: Vec3,
        props: &'static ExplosionProps,
        map: &GameMap,
        rng: &mut Rnd,
        debris_types: &[i32],
    ) -> Self {
        let z = map.get_z(pos.x, pos.y);
        let pos = if pos.z < z + 10.0 {
            Vec3::new(pos.x, pos.y, z + 10.0)
        } else {
            pos
        };
        let mut debs: Vec<Deb> = Vec::new();
        let rnd_range =
            |rng: &mut Rnd, a: f32, b: f32| -> f32 { a + (b - a) * rng.float01() as f32 };

        for _ in 0..props.fire_deb_cnt {
            let r = rnd_range(rng, props.min_speed, props.max_speed);
            let a = fsrnd(rng, std::f32::consts::PI);
            debs.push(Deb::Fire {
                // Default FIRE_SPEED = 0.02 (MatrixEffect.hpp:139,
                // MatrixEffectExplosion.cpp:361) — 0.04 rose 2× too fast.
                fire: Fire::new(pos, 1_000_000.0, 1000.0, 50.0, 5.0, false, 0.02),
                light_key: EMBER_KEY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                pos,
                v: Vec3::new(
                    a.sin() * r * 0.5,
                    a.cos() * r * 0.5,
                    rnd_range(rng, props.min_speed_z * 0.5, props.max_speed_z * 0.5),
                ),
                ttl: rnd_range(rng, props.deb_min_ttl as f32, props.deb_max_ttl as f32),
            });
        }
        for _ in 0..props.intense_cnt {
            let a = fsrnd(rng, std::f32::consts::PI);
            debs.push(Deb::Intense {
                ttm: frnd(rng, 200.0),
                ttl: INTENSE_TTL,
                v: Vec3::new(
                    INTENSE_DISPLACE_RADIUS * a.cos(),
                    INTENSE_DISPLACE_RADIUS * a.sin(),
                    0.0,
                ),
                pos,
                angle: fsrnd(rng, std::f32::consts::PI),
                scale: INTENSE_INIT_SIZE,
                disp: glam::Vec2::ZERO,
                color: 0x1F80_7E1B,
            });
        }
        for _ in 0..props.sparks_deb_cnt {
            let r = rnd_range(rng, props.min_speed, props.max_speed);
            let a = fsrnd(rng, std::f32::consts::PI);
            let ttl = rnd_range(rng, props.deb_min_ttl as f32, props.deb_max_ttl as f32);
            debs.push(Deb::Spark {
                pos,
                prepos: pos,
                v: Vec3::new(
                    a.sin() * r,
                    a.cos() * r,
                    rnd_range(rng, props.min_speed_z, props.max_speed_z),
                ),
                ttl,
                unttl: 1.0 / ttl,
            });
        }
        // First matching index + count, like the C++ scan (the catalog
        // groups each type contiguously).
        let dt_idx = debris_types.iter().position(|t| *t == props.deb_type);
        let dt_cnt = debris_types
            .iter()
            .filter(|t| **t == props.deb_type)
            .count();
        for _ in 0..props.deb_cnt {
            let Some(dt_idx) = dt_idx else { break };
            let r = rnd_range(rng, props.min_speed, props.max_speed);
            let a = fsrnd(rng, std::f32::consts::PI);
            debs.push(Deb::Mesh {
                pos,
                v: Vec3::new(
                    a.sin() * r,
                    a.cos() * r,
                    rnd_range(rng, props.min_speed_z, props.max_speed_z),
                ),
                ttl: rnd_range(rng, props.deb_min_ttl as f32, props.deb_max_ttl as f32),
                angley: fsrnd(rng, 1.0),
                anglep: fsrnd(rng, 1.0),
                angler: fsrnd(rng, 1.0),
                index: dt_idx + (rng.next() as usize) % dt_cnt,
                scale: 1.0,
            });
        }
        Self {
            props,
            time: 0.0,
            debs,
        }
    }

    pub fn voronka(&self) -> (SpotType, f32) {
        (self.props.voronka, self.props.voronka_scale)
    }

    /// `~CMatrixEffectExplosion` parity for the DeleteFirst eviction
    /// path: the C++ destructor's RemoveDebris kills every ember
    /// follow-light (MatrixEffectExplosion.cpp:794). Dropping a capped
    /// explosion without this leaks the lights forever — the
    /// "10 FPS after 10 minutes" bug.
    pub fn kill_lights(&self, objs: &mut crate::matrix_game::map_static::Objects) {
        for deb in &self.debs {
            if let Deb::Fire { light_key, .. } = deb {
                objs.pending_light_kill.push(
                    crate::matrix_game::effects::weapon::WeaponId::synthetic(*light_key),
                );
            }
        }
    }

    /// `Takt` (MatrixEffectExplosion.cpp:625-810). Returns false when
    /// all debris expired.
    pub fn takt(
        &mut self,
        step: f32,
        map: &GameMap,
        rng: &mut Rnd,
        objs: &mut crate::matrix_game::map_static::Objects,
    ) -> bool {
        let dtime = DEBRIS_SPEED * step;
        self.time += dtime;

        // Lifetime pass.
        let mut i = 0;
        while i < self.debs.len() {
            let dead = match &mut self.debs[i] {
                Deb::Intense { ttm, ttl, .. } => {
                    if *ttm >= 0.0 {
                        *ttm -= step;
                        false
                    } else {
                        *ttl -= step;
                        *ttl < 0.0
                    }
                }
                Deb::Fire {
                    ttl,
                    fire,
                    light_key,
                    ..
                } => {
                    fire.takt(step, rng);
                    if *ttl >= 0.0 {
                        *ttl -= step;
                        if *ttl < 0.0 {
                            // Wind the emitter down instead of popping:
                            // the C++ detaches it with SetTTL(1000)
                            // (MatrixEffectExplosion.cpp:476-487) so
                            // live puffs finish naturally. We linger
                            // 1000 ms more (marked by ttl ∈ [-1000, 0)).
                            fire.set_ttl(1000.0);
                            *ttl = -0.001;
                            // Ember glow dies with the emitter
                            // (RemoveDebris light kill, :794).
                            objs.pending_light_kill.push(
                                crate::matrix_game::effects::weapon::WeaponId::synthetic(
                                    *light_key,
                                ),
                            );
                        }
                        false
                    } else {
                        *ttl -= step;
                        *ttl < -1000.0
                    }
                }
                Deb::Spark { ttl, .. } | Deb::Mesh { ttl, .. } => {
                    *ttl -= step;
                    *ttl < 0.0
                }
            };
            if dead {
                self.debs.swap_remove(i);
            } else {
                i += 1;
            }
        }
        if self.debs.is_empty() {
            return false;
        }

        // Kinetics pass.
        for deb in &mut self.debs {
            match deb {
                Deb::Intense {
                    ttm,
                    ttl,
                    v,
                    disp,
                    scale,
                    color,
                    ..
                } => {
                    if *ttm < 0.0 {
                        let t = 1.0 - *ttl / INTENSE_TTL;
                        let df = if t > 0.05 { 1.0 } else { t * 20.0 };
                        *disp = glam::Vec2::new(v.x * df, v.y * df);
                        // `SetScale(1 + (60/4 - 1)*df)` OVERWRITES the
                        // billboard scale — absolute 1→15
                        // (MatrixEffectExplosion.cpp:716,
                        // CBillboard.hpp:163), not a multiplier on the
                        // initial size.
                        *scale = 1.0 + (INTENSE_END_SIZE / INTENSE_INIT_SIZE - 1.0) * df;
                        let a = ((1.0 - kscale(t, 0.0, 1.0)) * 255.0) as u32;
                        let r = calc_gradient(t, &[(150.0, 0.0), (100.0, 0.5), (0.0, 1.01)]) as u32;
                        let g = calc_gradient(
                            t,
                            &[(34.0, 0.0), (106.0, 0.25), (29.0, 0.5), (0.0, 1.01)],
                        ) as u32;
                        let b = calc_gradient(
                            t,
                            &[(23.0, 0.0), (29.0, 0.25), (20.0, 0.5), (0.0, 1.01)],
                        ) as u32;
                        *color = (a << 24) | (r << 16) | (g << 8) | b;
                    }
                }
                Deb::Fire {
                    fire,
                    pos,
                    v,
                    ttl,
                    light_key,
                } => {
                    // Winding-down emitters (ttl < 0) hold still.
                    if *ttl >= 0.0 {
                        v.z -= 1.1 * dtime;
                        *pos += *v * dtime;
                        if ground_hit(map, pos, v, ttl, rng) {
                            debris_splash(objs, *pos);
                        } else if crate::matrix_game::logic::is_on_base(objs, pos.x, pos.y) {
                            // Debris landing on a base footprint dies
                            // (m_Base cells, MatrixEffectExplosion.cpp:784-788).
                            *ttl = 1.0;
                        }
                        fire.set_pos(*pos);
                        // Ember terrain glow follows the piece
                        // (r 20, 0x22222202, MatrixEffectExplosion.cpp:362).
                        objs.pending_light_follow.push((
                            crate::matrix_game::effects::weapon::WeaponId::synthetic(*light_key),
                            [pos.x, pos.y, pos.z],
                            20.0,
                            0x2222_2202,
                        ));
                    }
                }
                Deb::Spark {
                    pos,
                    prepos,
                    v,
                    ttl,
                    unttl,
                } => {
                    v.z -= 1.1 * dtime;
                    *pos += *v * dtime;
                    if ground_hit(map, pos, v, ttl, rng) {
                        debris_splash(objs, *pos);
                    } else if crate::matrix_game::logic::is_on_base(objs, pos.x, pos.y) {
                        *ttl = 1.0;
                    }
                    let k = *ttl * *unttl;
                    let len = 5.0 * k * k + 1.0;
                    let delta = *pos - *prepos;
                    let x = delta.length();
                    if x > len {
                        *prepos += delta * ((x - len) / x);
                    }
                }
                Deb::Mesh { pos, v, ttl, .. } => {
                    v.z -= 1.1 * dtime;
                    *pos += *v * dtime;
                    if ground_hit(map, pos, v, ttl, rng) {
                        debris_splash(objs, *pos);
                    } else if crate::matrix_game::logic::is_on_base(objs, pos.x, pos.y) {
                        *ttl = 1.0;
                    }
                }
            }
        }
        true
    }

    pub fn draw(&self, q: &mut BillboardQueue, meshes: &mut MeshQueue) {
        for deb in &self.debs {
            match deb {
                Deb::Fire { fire, .. } => fire.draw(q),
                Deb::Intense {
                    ttm,
                    pos,
                    angle,
                    scale,
                    disp,
                    color,
                    ..
                } => {
                    if *ttm < 0.0 {
                        let (s, c) = angle.sin_cos();
                        q.billboards.push(QueuedBillboard {
                            pos: *pos,
                            scale: *scale,
                            rot: glam::Vec2::new(c, s),
                            disp: *disp,
                            color: *color,
                            tex: TexRef::Bbt(BBT_INTENSE),
                        });
                    }
                }
                Deb::Spark { pos, prepos, .. } => {
                    q.lines.push(QueuedLine {
                        p0: *prepos,
                        p1: *pos,
                        width: EXPLOSION_SPARK_WIDTH,
                        color: 0xFFFF_FFFF,
                        tex: TexRef::Bbt(BBT_SPARK),
                    });
                }
                Deb::Mesh {
                    pos,
                    ttl,
                    angley,
                    anglep,
                    angler,
                    index,
                    scale,
                    ..
                } => {
                    let k = (*ttl / 3500.0).clamp(0.0, 1.0);
                    let cc = (k * 255.0) as u32;
                    let color = 0xFF00_0000 | (cc << 16) | (cc << 8) | cc;
                    // D3DXMatrixRotationYawPitchRoll(angley*t, anglep*t, angler*t).
                    let rot = glam::Quat::from_euler(
                        glam::EulerRot::ZXY,
                        *angley * self.time,
                        *anglep * self.time,
                        *angler * self.time,
                    );
                    let rows = [
                        rot * Vec3::X * *scale,
                        rot * Vec3::Y * *scale,
                        rot * Vec3::Z * *scale,
                        *pos,
                    ];
                    meshes.draws.push(MeshDraw {
                        kind: MeshKind::Debris(*index),
                        rows,
                        color,
                    });
                }
            }
        }
    }
}

/// Shared ground/water response (MatrixEffectExplosion.cpp:759-789):
/// bounce off the terrain normal with 0.7 damping; terminate in water
/// or on built-up cells. Returns true when the piece hit water (the
/// caller spawns the splash).
fn ground_hit(map: &GameMap, pos: &mut Vec3, v: &mut Vec3, ttl: &mut f32, _rng: &mut Rnd) -> bool {
    use crate::matrix_game::common::WATER_LEVEL;
    // Off-map debris dies immediately (MatrixEffectExplosion.cpp:784-788).
    if pos.x < 0.0 || pos.y < 0.0 || pos.x >= map.world_width() || pos.y >= map.world_height() {
        *ttl = 1.0;
        return false;
    }
    let z = map.get_z(pos.x, pos.y);
    if pos.z < WATER_LEVEL && z <= WATER_LEVEL {
        *ttl = 1.0;
        return true;
    }
    if z > pos.z {
        let vl = v.length();
        let n = map.get_normal(pos.x, pos.y, false);
        *v += Vec3::new(n[0], n[1], n[2]) * vl;
        *v *= 0.7;
    }
    false
}

/// Water-splash konus for a drowned debris piece
/// (`CreateKonusSplash`, MatrixEffectExplosion.cpp:770).
fn debris_splash(objs: &mut crate::matrix_game::map_static::Objects, pos: Vec3) {
    use crate::matrix_game::effects::konus::Konus;
    use crate::matrix_game::effects::GameEffect;
    // S_SPLASH rides along with the konus (MatrixEffect.cpp:803).
    objs.queue_snd_at("splash", pos);
    let ang = ((pos.x * 3.1 + pos.y * 1.7).abs() % std::f32::consts::TAU) - std::f32::consts::PI;
    objs.pending_effects.push(GameEffect::Konus(Konus::new_splash(
        pos,
        Vec3::Z,
        10.0,
        5.0,
        ang,
        1000.0,
        true,
        TexRef::Path(crate::matrix_game::effects::effects_renderer::TEXTURE_PATH_SPLASH),
    )));
}

/// Port of `CMatrixEffectFireAnim` — the looping 8-frame flame sprite
/// `CreateExplosion(..., fire=true)` leaves at the blast site
/// (`CreateFireAnim(NULL, pos, w1=35, w2=60, ttl=2000)`).
pub struct FireAnim {
    pos: Vec3,
    w1: f32,
    w2: f32,
    ttl: f32,
    ttl0: f32,
    time: f32,
}

impl FireAnim {
    pub fn new(pos: Vec3, w1: f32, w2: f32, ttl: f32) -> Self {
        Self {
            pos,
            w1,
            w2,
            ttl,
            ttl0: ttl,
            // `m_Frame = IRND(8)` — random start frame
            // (MatrixEffectSmokeAndFire.cpp:442). Derive from pos so
            // neighbours desync without RNG plumbing.
            time: ((pos.x * 13.7 + pos.y * 7.3).abs() % 8.0) * 100.0,
        }
    }

    pub fn takt(&mut self, step: f32) -> bool {
        self.time += step;
        self.ttl -= step;
        self.ttl >= 0.0
    }

    pub fn draw(&self, q: &mut BillboardQueue) {
        // CMatrixEffectFireAnim (MatrixEffectSmokeAndFire.cpp:431-509):
        // a vertical billboard-LINE, width w1 × height w2, at constant
        // full alpha. Grows 0→full over the first 10% of TTL
        // (FIREFRAME_TTL_POROG 0.9) then shrinks to 0 over the rest;
        // 8 FLAMEFRAME textures at 100 ms/frame from a random start.
        let frame = (self.time / 100.0) as usize % 8;
        let k = (self.ttl / self.ttl0).clamp(0.0, 1.0); // 1 → 0
        const POROG: f32 = 0.9;
        let (w, h) = if k < POROG {
            let kk = k / POROG;
            (self.w1 * kk, self.w2 * kk)
        } else {
            let kk = (k - POROG) / (1.0 - POROG); // 1 at birth → 0
            (self.w1 * (1.0 - kk), self.w2 * (1.0 - kk))
        };
        if w <= 0.0 || h <= 0.0 {
            return;
        }
        q.line(
            self.pos,
            self.pos + Vec3::new(0.0, 0.0, h),
            w,
            0xFFFF_FFFF,
            TexRef::Bbt(BBT_FLAMEFRAME0 + frame),
        );
    }
}

/// `Smoke` companion for `CreateExplosion(..., fire=true)`:
/// `CreateSmoke(NULL, pos, 1500, 1500, 100, 0xFF000000)`.
pub fn explosion_fire_smoke(pos: Vec3) -> Smoke {
    Smoke::new(pos, 1500.0, 1500.0, 100.0, 0xFF00_0000, false, 0.04)
}
