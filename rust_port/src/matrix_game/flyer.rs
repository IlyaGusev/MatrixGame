//! Port of `MatrixFlyer.{cpp,hpp}` — `CMatrixFlyer`, scoped to the
//! robot-delivery role (`FO_GIVE_BOT`): maintenance supply drops fly
//! in carrying a robot, release it mid-pass and leave the map.
//!
//! Manual/arcade piloting (FLYER_MANUAL, LogicTaktArcade, the strategy
//! trajectory mode and the render-to-texture UI clones) is out of
//! scope by project decision — those paths are dead or commented out
//! in the shipped C++ too (MatrixFlyer.cpp:1344-1350).

use crate::matrix_game::logic::Rnd;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{HpBar, MapStatic, ObjectCore, ObjectId, ObjectType, Objects};
use crate::matrix_lib::three_g::math3d::Trajectory;
use glam::{Vec2, Vec3};

/// `FLYER_ALT_EMPTY` / `FLYER_ALT_MIN` (MatrixFlyer.hpp).
pub const FLYER_ALT_EMPTY: f32 = 70.0;
pub const FLYER_ALT_MIN: f32 = 20.0;
/// `ENGINE_ANGLE_STAY` / `ENGINE_ANGLE_MOVE`, `YAW_ANGLE_STAY` /
/// `YAW_ANGLE_MOVE` (MatrixFlyer.hpp, GRAD2RAD already applied).
pub const ENGINE_ANGLE_STAY: f32 = std::f32::consts::FRAC_PI_2;
pub const ENGINE_ANGLE_MOVE: f32 = 0.0;
pub const YAW_ANGLE_STAY: f32 = 0.0;
pub const YAW_ANGLE_MOVE: f32 = -20.0 * std::f32::consts::PI / 180.0;
/// Default rotor spin rate (`DAngle`, MatrixFlyer.cpp:546).
pub const VINT_SPEED: f32 = -0.025;

/// `SCarryData` (MatrixFlyer.hpp) — the dangling-robot spring state.
#[derive(Debug, Clone)]
pub struct CarryData {
    pub robot_forward: Vec2,
    pub robot_up: Vec3,
    pub robot_up_back: Vec3,
    pub robot_angle: f32,
    pub robot_mass_factor: f32,
    pub elevator: Option<crate::matrix_game::effects::elevator_field::ElevatorField>,
}

impl Default for CarryData {
    fn default() -> Self {
        Self {
            robot_forward: Vec2::new(0.0, -1.0),
            robot_up: Vec3::Z,
            robot_up_back: Vec3::ZERO,
            robot_angle: 0.0,
            robot_mass_factor: 0.0,
            elevator: None,
        }
    }
}

pub struct Flyer {
    core: ObjectCore,
    rchange: u32,
    object_state: u32,

    pub self_id: Option<ObjectId>,
    pub side: i32,
    /// `m_FlyerKind` (EFlyerKind 0-3).
    pub kind: i32,
    pub pos: Vec3,
    /// `m_AngleZ` heading (radians; 0 = +Y like the robots).
    pub angle: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub engine_angle: f32,
    pub target_engine_angle: f32,
    pub target_yaw: f32,
    pub target_pitch: f32,
    pub move_speed: f32,
    /// Rotor spin accumulator (m_Units[vint].m_Vint.m_Angle).
    pub vint_angle: f32,

    pub hit_point: f32,
    pub hit_point_max: f32,
    /// `m_ShowHitpointTime` — hover sets it (MatrixMap.cpp:1176),
    /// LogicTakt decays it (MatrixFlyer.cpp:1305-1308).
    pub show_hitpoint_time: i32,

    trajectory: Option<Trajectory>,
    trajectory_pos: f32,
    trajectory_len_rev: f32,
    trajectory_target_angle: f32,

    pub carrying: Option<ObjectId>,
    pub carry: CarryData,
}

impl Flyer {
    /// The FO_GIVE_BOT slice of `ApplyOrder` (MatrixFlyer.cpp:
    /// 1154-1262) minus the robot creation (the caller owns that —
    /// arena spawning can't happen from inside a constructor here).
    /// `target` is the world-space drop point, `ang` the approach
    /// heading, `dist`/`height`/`land_height` the GiveBot config
    /// values (Distance already FSRND(150)-jittered by the caller).
    pub fn new_give_bot(
        map: &GameMap,
        target: Vec2,
        side: i32,
        ang: f32,
        dist: f32,
        height: f32,
        land_height: f32,
        kind: i32,
        hitpoint: f32,
    ) -> Self {
        let z = map.get_z(target.x, target.y);
        let p = Vec3::new(target.x, target.y, z + land_height);
        let dir = Vec3::new(ang.sin(), ang.cos(), 0.0);

        let pts = [
            p - dir * dist + Vec3::new(0.0, 0.0, height),
            p - dir * 100.0,
            p + dir * 100.0,
            p + dir * dist * 2.0 + Vec3::new(0.0, 0.0, height * 3.0),
        ];
        let trajectory = Trajectory::new_approx(&pts);
        let len = trajectory.calc_length().max(1.0);

        let mut core = ObjectCore {
            obj_type: ObjectType::Flyer,
            ..Default::default()
        };
        core.geo_center = pts[0];
        core.radius = 15.0; // FLYER_RADIUS

        Self {
            core,
            rchange: 0,
            object_state: 0,
            self_id: None,
            side,
            kind,
            pos: pts[0],
            angle: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            engine_angle: ENGINE_ANGLE_STAY,
            target_engine_angle: ENGINE_ANGLE_STAY,
            target_yaw: YAW_ANGLE_STAY,
            target_pitch: 0.0,
            move_speed: 0.0,
            vint_angle: 0.0,
            hit_point: hitpoint,
            hit_point_max: hitpoint,
            show_hitpoint_time: 0,
            trajectory: Some(trajectory),
            trajectory_pos: 0.0,
            trajectory_len_rev: 1.0 / len,
            trajectory_target_angle: ang,
            carrying: None,
            carry: CarryData::default(),
        }
    }

    pub fn carrying_robot(&self) -> Option<ObjectId> {
        self.carrying
    }

    /// `ProceedTrajectory` (MatrixFlyer.cpp:1640-1691) — the live part
    /// before the unconditional `return`.
    fn proceed_trajectory(&mut self, ms: f32) {
        let Some(tr) = self.trajectory.as_ref() else {
            return;
        };
        let ptp = self.trajectory_pos;
        self.trajectory_pos += ms * self.trajectory_len_rev * 0.09;

        let p = tr.calc_point(self.trajectory_pos);
        let fdir = p - self.pos;
        let dd = fdir.length();
        self.move_speed = dd / ms.max(0.001);

        let mut a = (-fdir.x).atan2(fdir.y);
        let aa = angle_dist(a, self.trajectory_target_angle);
        a += kscale(self.trajectory_pos, 0.8, 0.99) * aa;

        let da = angle_dist(self.angle, a);
        if da.abs() < 30.0f32.to_radians() {
            self.pos = p;
        } else {
            self.trajectory_pos = ptp;
        }

        let mul = (1.0 - 0.997f64.powf(ms as f64)) as f32 * da;
        self.angle += mul;

        self.target_engine_angle = lerp(ENGINE_ANGLE_STAY, ENGINE_ANGLE_MOVE - self.yaw, 0.5);
        self.target_yaw = lerp(YAW_ANGLE_STAY, YAW_ANGLE_MOVE, 0.5);
        self.target_pitch = 0.0;

        if self.trajectory_pos > 0.98 {
            self.trajectory = None; // CancelTrajectory
        }
    }

    /// `CalcFlyerZInPoint` (MatrixFlyer.cpp:249-262). The C++ samples
    /// `GetZInterpolatedObjRobots` — a spline over per-group max
    /// heights INCLUDING buildings/objects — so the flyer cruises the
    /// smoothed scene envelope instead of hugging terrain triangles
    /// (and clipping through tall structures). We sample the lazily
    /// built static envelope in `objs.flyer_alt_grid`; live robots
    /// (part of the C++ grid) are low enough to be covered by
    /// FLYER_ALT_MIN.
    fn calc_z_in_point(&self, map: &GameMap, objs: &Objects, x: f32, y: f32) -> f32 {
        let addz = sample_flyer_envelope(map, &objs.flyer_alt_grid, x, y);
        if addz > FLYER_ALT_EMPTY {
            addz + FLYER_ALT_MIN
        } else {
            FLYER_ALT_EMPTY + FLYER_ALT_MIN
        }
    }
}

/// Build the per-group static altitude envelope: terrain land max
/// folded with every live building / map object top (the static-scene
/// slice of `m_GroupMaxZObjRobots`).
pub fn ensure_flyer_alt_grid(objs: &mut Objects, map: &GameMap) {
    if !objs.flyer_alt_grid.is_empty() {
        return;
    }
    let mut grid = map.group_max_z_land.clone();
    let gs =
        crate::matrix_game::common::MAP_GROUP_SIZE as f32 * crate::matrix_game::map::GLOBAL_SCALE;
    let ids: Vec<crate::matrix_game::map_static::ObjectId> = objs.iter_live().collect();
    for id in ids {
        let Some(o) = objs.get(id) else { continue };
        if !matches!(
            o.core().obj_type,
            crate::matrix_game::map_static::ObjectType::Building
                | crate::matrix_game::map_static::ObjectType::MapObject
        ) {
            continue;
        }
        let c = o.core().geo_center;
        let top = c.z + o.core().radius;
        let gx = (c.x / gs).floor() as i32;
        let gy = (c.y / gs).floor() as i32;
        if gx < 0 || gy < 0 || gx >= map.group_w as i32 || gy >= map.group_h as i32 {
            continue;
        }
        let cell = &mut grid[gy as usize * map.group_w + gx as usize];
        *cell = cell.max(top);
    }
    objs.flyer_alt_grid = grid;
}

/// Bilinear sample of the flyer envelope in group-center coordinates —
/// the smoothing stand-in for the C++ 4×4 B-spline
/// (GetZInterpolatedObjRobots, MatrixMap.cpp:512-546).
fn sample_flyer_envelope(map: &GameMap, grid: &[f32], wx: f32, wy: f32) -> f32 {
    if grid.len() != map.group_w * map.group_h {
        return map.get_z(wx, wy);
    }
    let gs =
        crate::matrix_game::common::MAP_GROUP_SIZE as f32 * crate::matrix_game::map::GLOBAL_SCALE;
    let fx = wx / gs - 0.5;
    let fy = wy / gs - 0.5;
    let gx0 = fx.floor() as i32;
    let gy0 = fy.floor() as i32;
    let tx = fx - gx0 as f32;
    let ty = fy - gy0 as f32;
    let sample = |gx: i32, gy: i32| -> f32 {
        if gx < 0 || gy < 0 || gx >= map.group_w as i32 || gy >= map.group_h as i32 {
            return 0.0;
        }
        grid[gy as usize * map.group_w + gx as usize]
    };
    let a = sample(gx0, gy0);
    let b = sample(gx0 + 1, gy0);
    let c = sample(gx0, gy0 + 1);
    let d = sample(gx0 + 1, gy0 + 1);
    let ab = a + (b - a) * tx;
    let cd = c + (d - c) * tx;
    ab + (cd - ab) * ty
}

/// `AngleDist` — shortest signed angular distance.
pub fn angle_dist(from: f32, to: f32) -> f32 {
    let mut d = (to - from) % std::f32::consts::TAU;
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    } else if d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    d
}

/// `KSCALE(x, a, b)` — clamped 0..1 ramp of x across [a, b].
fn kscale(x: f32, a: f32, b: f32) -> f32 {
    ((x - a) / (b - a)).clamp(0.0, 1.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl MapStatic for Flyer {
    fn core(&self) -> &ObjectCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut ObjectCore {
        &mut self.core
    }
    fn rchange(&self) -> u32 {
        self.rchange
    }
    fn rchange_set(&mut self, bits: u32) {
        self.rchange |= bits;
    }
    fn rchange_clear(&mut self, bits: u32) {
        self.rchange &= !bits;
    }
    fn object_state(&self) -> u32 {
        self.object_state
    }
    fn object_state_set(&mut self, bits: u32) {
        self.object_state |= bits;
    }
    fn object_state_clear(&mut self, bits: u32) {
        self.object_state &= !bits;
    }
    fn ablaze_ttl(&self) -> i32 {
        0
    }
    fn set_ablaze_ttl(&mut self, _ttl: i32) {}
    fn shorted_ttl(&self) -> i32 {
        0
    }
    fn set_shorted_ttl(&mut self, _ttl: i32) {}
    fn r_need(&mut self, _need: u32) {}
    fn side(&self) -> i32 {
        self.side
    }
    fn show_hitpoint(&mut self) {
        // MatrixFlyer.hpp:342.
        self.show_hitpoint_time = crate::matrix_game::common::HITPOINT_SHOW_TIME_MS;
    }

    /// MatrixFlyer.cpp:718-722 — bar centered above the body
    /// (PB_FLYER_WIDTH = 100, lifted by FLYER_RADIUS*2).
    fn hitpoint_bar(&self, _map: &GameMap) -> Option<HpBar> {
        if self.show_hitpoint_time <= 0 || self.hit_point <= 0.0 {
            return None;
        }
        Some(HpBar {
            anchor: self.pos,
            width: 100.0,
            fill: (self.hit_point / self.hit_point_max.max(1.0)).clamp(0.0, 1.0),
            x_off: -50.0,
            y_off: -30.0, // FLYER_RADIUS * 2
        })
    }

    /// Port of `CMatrixFlyer::Damage` (MatrixFlyer.cpp:1782-1846).
    /// Sounds and the FLYER_IN_SPAWN base-close branch are N/A here —
    /// the delivery flyer never spawns from a base.
    fn damage(
        &mut self,
        weap: crate::matrix_game::effects::weapon::Weapon,
        pos: Vec3,
        _dir: Vec3,
        _attacker_side: i32,
        _attacker: Option<ObjectId>,
        self_id: ObjectId,
        objs: &mut Objects,
    ) -> bool {
        use crate::matrix_game::effects::weapon::{
            WEAPON_ABLAZE, WEAPON_CANNON2, WEAPON_FLAMETHROWER, WEAPON_LASER, WEAPON_LIGHTENING,
            WEAPON_REPAIR, WEAPON_SHORTED,
        };

        if weap == WEAPON_REPAIR {
            return false;
        }

        let cfg = crate::matrix_game::config::global();
        if let Some(dmg) = cfg.flyer_damages.get(weap) {
            if self.hit_point > dmg.mindamage as f32 {
                self.hit_point -= dmg.damage as f32;
            }
        }

        // WEAPON_LIGHTENING → FireEnd (:1806-1814). The delivery flyer
        // carries no weapons, so there is nothing to cease.

        if self.hit_point > 0.0 {
            // Hit shake `m_Pitch += FSRND(0.1)` (:1821-1822) — visual-
            // only randomness, local seeded stream like the robot's
            // damage visuals.
            if weap != WEAPON_LASER && weap != WEAPON_CANNON2 {
                use crate::matrix_game::effects::smoke_and_fire::fsrnd;
                let now = crate::matrix_game::map::current_elapsed_ms();
                let mut vrng = crate::matrix_game::logic::Rnd::new(
                    ((now as i32) ^ ((pos.x + pos.y) as i32)).max(1),
                );
                self.pitch += fsrnd(&mut vrng, 0.1);
            }
            if weap != WEAPON_ABLAZE
                && weap != WEAPON_SHORTED
                && weap != WEAPON_LIGHTENING
                && weap != WEAPON_FLAMETHROWER
            {
                objs.pending_explosions
                    .push(crate::matrix_game::map_static::ExplosionSpawn {
                        pos,
                        props: &crate::matrix_game::effects::explosion::EXPLOSION_ROBOT_HIT,
                        fire: false,
                    });
            }
            false
        } else {
            // Dead (:1829-1843): boom at the body, drop the cargo (the
            // C++ leaves the carried robot dangling — the Falling state
            // is the port's safe equivalent of losing the carrier), and
            // despawn.
            objs.pending_explosions
                .push(crate::matrix_game::map_static::ExplosionSpawn {
                    pos: self.pos,
                    props: &crate::matrix_game::effects::explosion::EXPLOSION_ROBOT_BOOM,
                    fire: false,
                });
            if let Some(rid) = self.carrying.take() {
                if let Some(r) = crate::matrix_game::logic::robot_mut(objs, rid) {
                    r.carry_release();
                }
            }
            objs.queue_snd_stop(crate::matrix_game::sound::SndHandle::Flyer(self_id));
            objs.remove_deferred(self_id);
            true
        }
    }

    fn takt(&mut self, cms: i32, rng: &mut Rnd, _objs: &mut Objects) {
        // Rotor spin (m_Vint.m_AngleSpeedMax) — pure visual.
        self.vint_angle += VINT_SPEED * cms as f32;
        // Tractor-beam tracers advance on the frame clock
        // (CMatrixEffectElevatorField::Takt).
        if let Some(e) = self.carry.elevator.as_mut() {
            e.takt(cms as f32, rng);
        }
    }

    /// `LogicTakt` (MatrixFlyer.cpp:1291-1382), delivery slice.
    fn logic_takt(&mut self, cms: i32, _rng: &mut Rnd, objs: &mut Objects) {
        let Some(map) = crate::matrix_game::map::current_map() else {
            return;
        };
        // HP-bar timer decay (MatrixFlyer.cpp:1305-1308).
        if self.show_hitpoint_time > 0 {
            self.show_hitpoint_time = (self.show_hitpoint_time - cms).max(0);
        }
        let ms = cms as f32;
        let mul = (1.0 - 0.998f64.powf(ms as f64)) as f32;

        // LogicTaktOrder (1265-1289).
        self.proceed_trajectory(ms);
        if self.trajectory.is_none() {
            // Done — leave the map. Drop the robot first if the pass
            // never crossed 0.5 (degenerate short trajectory).
            if let Some(rid) = self.carrying.take() {
                if let Some(r) = crate::matrix_game::logic::robot_mut(objs, rid) {
                    r.carry_release();
                }
            }
            if let Some(me) = self.self_id {
                objs.queue_snd_stop(crate::matrix_game::sound::SndHandle::Flyer(me));
                objs.remove_deferred(me);
            }
            return;
        }
        self.core.geo_center = self.pos;

        // `m_Sound = CSound::Play(m_Sound, S_FLYER_VINT_CONTINUE, m_Pos)`
        // (MatrixFlyer.cpp:1325) — looped rotor hum follows the flyer.
        if let Some(me) = self.self_id {
            objs.queue_snd_follow(
                crate::matrix_game::sound::SndHandle::Flyer(me),
                "fl_cont",
                self.pos,
                crate::matrix_game::sound::SoundLayer::All,
            );
        }

        if self.trajectory_pos > 0.5 {
            if let Some(rid) = self.carrying.take() {
                if let Some(r) = crate::matrix_game::logic::robot_mut(objs, rid) {
                    r.carry_release();
                }
                self.carry.elevator = None;
            }
        }

        // Engine / yaw / pitch easing (1356-1372).
        let d = angle_dist(self.engine_angle, self.target_engine_angle);
        self.engine_angle += d * mul;
        let d = angle_dist(self.yaw, self.target_yaw);
        self.yaw += d * mul;
        let d = angle_dist(self.pitch, self.target_pitch);
        self.pitch += d * mul;

        // Altitude tracking (1374-1382).
        ensure_flyer_alt_grid(objs, map);
        let newz = self.calc_z_in_point(map, objs, self.pos.x, self.pos.y);
        self.pos.z += (newz - self.pos.z) * mul;
        self.core.geo_center = self.pos;
    }
}
