//! Behavior tests for the weapon / damage mechanics port: damage-table
//! application, DOT status effects, weapon cooldown cadence, hitscan
//! trace, and an end-to-end projectile shot.

use glam::Vec3;

use crate::matrix_game::common::{PLAYER_SIDE, TRACE_ALL};
use crate::matrix_game::config::{self, Difficulty, GlobalConfig, OverheatParams, WeaponDamage};
use crate::matrix_game::effects::weapon::{
    weapon_takt, WeaponEffect, WeaponHandler, WEAPON_BIGBOOM, WEAPON_COUNT, WEAPON_FLAMETHROWER,
    WEAPON_GUN, WEAPON_LIGHTENING, WEAPON_REPAIR, WEAPON_VOLCANO,
};
use crate::matrix_game::logic::Rnd;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{
    MapStatic, ObjectId, Objects, OBJECT_STATE_ABLAZE, OBJECT_STATE_SHORTED,
};
use crate::matrix_game::map_trace::{trace, TraceStop};
use crate::matrix_game::robot::ChassisKind;
use crate::matrix_game::robot::{Robot, RobotState};

/// Every test writes the SAME global config, so parallel test threads
/// can't observe torn state.
fn install_test_config() {
    let mut cfg = GlobalConfig::default();
    let dmg = WeaponDamage {
        damage: 10,
        mindamage: 0,
        friend_damage: 4,
    };
    for i in 0..WEAPON_COUNT {
        cfg.robot_damages.table[i] = dmg;
        cfg.cannon_damages.table[i] = dmg;
        cfg.weapon_radius.table[i] = 500.0;
        cfg.weapon_cooldown.table[i] = 100;
    }
    let oh = OverheatParams {
        heat_mod: 100,
        cool_period: 100,
        cool_mod: 50,
    };
    cfg.overheat.volcano = oh;
    cfg.overheat.plasma = oh;
    cfg.overheat.laser = oh;
    cfg.overheat.homing_missile = oh;
    cfg.overheat.flamethrower = oh;
    cfg.overheat.bomb = oh;
    cfg.overheat.gun = oh;
    cfg.overheat.lightening = oh;
    cfg.difficulty = Difficulty::default();
    config::set_global(cfg);
}

fn spawn_robot(objs: &mut Objects, pos: Vec3, side: i32, hp: f32) -> ObjectId {
    let mut r = Robot::new(pos, side, ChassisKind::Track);
    r.hit_point = hp;
    r.hit_point_max = hp;
    let id = objs.spawn(Box::new(r));
    if let Some(obj) = objs.get_mut(id) {
        let r: &mut Robot = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Robot) };
        r.self_id = Some(id);
    }
    id
}

fn robot<'a>(objs: &'a Objects, id: ObjectId) -> &'a Robot {
    let o = objs.get(id).unwrap();
    unsafe { &*(o as *const dyn MapStatic as *const Robot) }
}

#[test]
fn robot_damage_subtracts_table_damage_and_floors_on_mindamage() {
    install_test_config();
    let mut objs = Objects::new();
    let id = spawn_robot(&mut objs, Vec3::ZERO, PLAYER_SIDE, 100.0);
    let dead = objs.apply_damage(id, WEAPON_GUN, Vec3::ZERO, Vec3::X, 2, None);
    assert!(!dead);
    assert_eq!(robot(&objs, id).hit_point, 90.0);
}

#[test]
fn robot_friendly_fire_uses_friend_damage_column() {
    install_test_config();
    let mut objs = Objects::new();
    // Non-player side so the player FF multiplier isn't in play.
    let id = spawn_robot(&mut objs, Vec3::ZERO, 2, 100.0);
    objs.apply_damage(id, WEAPON_GUN, Vec3::ZERO, Vec3::X, 2, None);
    assert_eq!(robot(&objs, id).hit_point, 96.0); // friend_damage = 4
}

#[test]
fn robot_repair_heals_and_clamps_to_max() {
    install_test_config();
    let mut objs = Objects::new();
    let id = spawn_robot(&mut objs, Vec3::ZERO, PLAYER_SIDE, 100.0);
    objs.apply_damage(id, WEAPON_GUN, Vec3::ZERO, Vec3::X, 2, None); // 90
    objs.apply_damage(id, WEAPON_REPAIR, Vec3::ZERO, Vec3::X, 0, None);
    assert_eq!(robot(&objs, id).hit_point, 100.0); // +10, clamped
}

#[test]
fn robot_flamethrower_marks_ablaze_and_caps_ttl_at_5000() {
    install_test_config();
    let mut objs = Objects::new();
    let id = spawn_robot(&mut objs, Vec3::ZERO, PLAYER_SIDE, 10_000.0);
    for _ in 0..30 {
        objs.apply_damage(id, WEAPON_FLAMETHROWER, Vec3::ZERO, Vec3::X, 2, None);
    }
    let r = robot(&objs, id);
    assert!(r.object_state() & OBJECT_STATE_ABLAZE != 0);
    assert_eq!(r.ablaze_ttl(), 5000);
    assert_eq!(r.last_delay_damage_side, 2);
}

#[test]
fn robot_lightening_shorts_with_light_protect_scaling() {
    install_test_config();
    let mut objs = Objects::new();
    let id = spawn_robot(&mut objs, Vec3::ZERO, PLAYER_SIDE, 10_000.0);
    {
        let o = objs.get_mut(id).unwrap();
        let r: &mut Robot = unsafe { &mut *(o as *mut dyn MapStatic as *mut Robot) };
        r.light_protect = 0.5;
    }
    objs.apply_damage(id, WEAPON_LIGHTENING, Vec3::ZERO, Vec3::X, 2, None);
    let r = robot(&objs, id);
    assert!(r.object_state() & OBJECT_STATE_SHORTED != 0);
    // ttl += 500 - 500*0.5 = 250; max 3000 - 3000*0.5 = 1500.
    assert_eq!(r.shorted_ttl(), 250);
    for _ in 0..20 {
        objs.apply_damage(id, WEAPON_LIGHTENING, Vec3::ZERO, Vec3::X, 2, None);
    }
    assert_eq!(robot(&objs, id).shorted_ttl(), 1500);
}

#[test]
fn robot_bigboom_damage_reduced_by_bomb_protect() {
    install_test_config();
    let mut objs = Objects::new();
    let id = spawn_robot(&mut objs, Vec3::ZERO, PLAYER_SIDE, 100.0);
    {
        let o = objs.get_mut(id).unwrap();
        let r: &mut Robot = unsafe { &mut *(o as *mut dyn MapStatic as *mut Robot) };
        r.bomb_protect = 0.5;
    }
    objs.apply_damage(id, WEAPON_BIGBOOM, Vec3::ZERO, Vec3::X, 2, None);
    assert_eq!(robot(&objs, id).hit_point, 95.0); // 10 * (1 - 0.5)
}

#[test]
fn robot_death_flips_to_dip_and_credits_attacker_side() {
    install_test_config();
    let mut objs = Objects::new();
    let id = spawn_robot(&mut objs, Vec3::ZERO, PLAYER_SIDE, 10.0);
    let dead = objs.apply_damage(id, WEAPON_GUN, Vec3::ZERO, Vec3::X, 2, None);
    assert!(dead);
    let r = robot(&objs, id);
    assert_eq!(r.state, RobotState::Dip);
    assert!(!r.is_live());
    assert_eq!(objs.side_stats[2].robot_kill, 1);
    // Dead robots stop matching trace/search masks.
    assert!(!objs.any_object_in_radius(
        glam::Vec2::ZERO,
        50.0,
        1.0,
        crate::matrix_game::common::TRACE_ROBOT,
        None
    ));
}

#[test]
fn trace_hits_flat_landscape() {
    let map = GameMap::test_flat(16, 16, 5.0);
    let objs = Objects::new();
    let start = Vec3::new(100.0, 100.0, 50.0);
    let end = Vec3::new(140.0, 120.0, -10.0);
    let (stop, pos) = trace(&map, &objs, start, end, TRACE_ALL, None);
    assert_eq!(stop, TraceStop::Landscape);
    assert!((pos.z - 5.0).abs() < 0.01, "hit z = {}", pos.z);
}

#[test]
fn trace_prefers_nearer_object_over_landscape() {
    install_test_config();
    let map = GameMap::test_flat(16, 16, 0.0);
    let mut objs = Objects::new();
    let id = spawn_robot(&mut objs, Vec3::new(120.0, 100.0, 10.0), 2, 100.0);
    let start = Vec3::new(100.0, 100.0, 13.0);
    let end = Vec3::new(200.0, 100.0, 13.0);
    let (stop, _) = trace(&map, &objs, start, end, TRACE_ALL, None);
    assert_eq!(stop.object(), Some(id));
}

#[test]
fn weapon_cooldown_loop_fires_once_per_period() {
    install_test_config();
    let map = GameMap::test_flat(16, 16, 0.0);
    let mut objs = Objects::new();
    let mut rng = Rnd::new(7);

    let mut w = WeaponEffect::new(WEAPON_GUN, 0, WeaponHandler::None);
    w.modify(
        Vec3::new(100.0, 100.0, 10.0),
        Vec3::X,
        Vec3::new(200.0, 100.0, 10.0),
    );
    let wid = objs.weapons.create(w);
    objs.weapons
        .get_mut(wid)
        .unwrap()
        .fire_begin(Vec3::ZERO, None);

    // Cooldown = 100ms. First shot fires immediately (m_Time starts
    // at 0 ≥ 0, dropping it to -100); each takt then adds 10ms back,
    // so the second shot lands on the 11th takt when m_Time reaches 0.
    for _ in 0..11 {
        weapon_takt(&mut objs, wid, 10.0, &map, &mut rng);
    }
    // Each shot spawns a shell + a muzzle-flash line effect.
    let shells = objs
        .pending_effects
        .iter()
        .filter(|e| matches!(e, crate::matrix_game::effects::GameEffect::MovingObject(_)))
        .count();
    assert_eq!(shells, 2);
    assert_eq!(objs.weapons.get(wid).unwrap().fire_count(), 2);
}

#[test]
fn gun_projectile_travels_and_damages_target() {
    install_test_config();
    let map = GameMap::test_flat(32, 32, 0.0);
    let mut objs = Objects::new();
    let mut rng = Rnd::new(7);

    let shooter = spawn_robot(&mut objs, Vec3::new(100.0, 100.0, 0.0), PLAYER_SIDE, 100.0);
    let target = spawn_robot(&mut objs, Vec3::new(200.0, 100.0, 0.0), 2, 100.0);

    let mut w = WeaponEffect::new(WEAPON_GUN, 0, WeaponHandler::None);
    w.set_owner(shooter, PLAYER_SIDE);
    // Muzzle at the shooter, target point = the victim's center.
    let tgt = objs.get(target).unwrap().core().geo_center;
    let muzzle = objs.get(shooter).unwrap().core().geo_center;
    w.modify(muzzle, (tgt - muzzle).normalize(), tgt);
    let wid = objs.weapons.create(w);
    objs.weapons
        .get_mut(wid)
        .unwrap()
        .fire_begin(tgt, Some(shooter));
    weapon_takt(&mut objs, wid, 10.0, &map, &mut rng);
    assert!(!objs.pending_effects.is_empty(), "no projectile spawned");

    // Drive the effect list until the shell lands (22 units/ms × 0.1
    // scale ⇒ ~45 frames at 10ms to cover 100 units).
    let mut effects = Vec::new();
    for _ in 0..100 {
        crate::matrix_game::effects::effects_takt(&mut effects, 10.0, &map, &mut objs, &mut rng);
        if effects.is_empty() && objs.pending_effects.is_empty() {
            break;
        }
    }
    let hp = robot(&objs, target).hit_point;
    assert!(hp < 100.0, "target took no damage (hp={hp})");
    // Shooter untouched (skip + friendly side).
    assert_eq!(robot(&objs, shooter).hit_point, 100.0);
}

#[test]
fn volcano_hitscan_damages_target_immediately() {
    install_test_config();
    let map = GameMap::test_flat(32, 32, 0.0);
    let mut objs = Objects::new();
    let mut rng = Rnd::new(7);

    let shooter = spawn_robot(&mut objs, Vec3::new(100.0, 100.0, 0.0), PLAYER_SIDE, 100.0);
    let target = spawn_robot(&mut objs, Vec3::new(160.0, 100.0, 0.0), 2, 100.0);

    let mut w = WeaponEffect::new(WEAPON_VOLCANO, 0, WeaponHandler::None);
    w.set_owner(shooter, PLAYER_SIDE);
    let tgt = objs.get(target).unwrap().core().geo_center;
    let muzzle = objs.get(shooter).unwrap().core().geo_center;
    w.modify(muzzle, (tgt - muzzle).normalize(), Vec3::ZERO);
    let wid = objs.weapons.create(w);
    objs.weapons
        .get_mut(wid)
        .unwrap()
        .fire_begin(Vec3::ZERO, Some(shooter));
    weapon_takt(&mut objs, wid, 10.0, &map, &mut rng);

    assert_eq!(robot(&objs, target).hit_point, 90.0);
    assert!(objs.weapons.get_mut(wid).unwrap().is_hit_was());
}

#[test]
fn visuals_fill_billboard_and_mesh_queues() {
    use crate::matrix_game::effects::explosion::MeshQueue;
    use crate::matrix_lib::three_g::billboard::BillboardQueue;

    install_test_config();
    let map = GameMap::test_flat(32, 32, 0.0);
    let mut objs = Objects::new();
    objs.debris_catalog_len = 4;
    // Shipped-data shape: types 0 (gaika) then 1 (mysor), contiguous.
    objs.debris_types = vec![0, 0, 1, 1];
    let mut rng = Rnd::new(7);

    let shooter = spawn_robot(&mut objs, Vec3::new(100.0, 100.0, 0.0), PLAYER_SIDE, 100.0);
    let target = spawn_robot(&mut objs, Vec3::new(200.0, 100.0, 0.0), 2, 10.0);

    // A gun shot that kills: projectile mesh + tracer lines + muzzle
    // flash + death explosion + crater + wreck scatter.
    let mut w = WeaponEffect::new(WEAPON_GUN, 0, WeaponHandler::None);
    w.set_owner(shooter, PLAYER_SIDE);
    let tgt = objs.get(target).unwrap().core().geo_center;
    let muzzle = objs.get(shooter).unwrap().core().geo_center;
    w.modify(muzzle, (tgt - muzzle).normalize(), tgt);
    let wid = objs.weapons.create(w);
    objs.weapons
        .get_mut(wid)
        .unwrap()
        .fire_begin(tgt, Some(shooter));
    weapon_takt(&mut objs, wid, 10.0, &map, &mut rng);

    let mut effects = Vec::new();
    for _ in 0..200 {
        crate::matrix_game::effects::effects_takt(&mut effects, 10.0, &map, &mut objs, &mut rng);
        if !objs.is_valid(target) || robot(&objs, target).state == RobotState::Dip {
            break;
        }
    }
    assert_eq!(robot(&objs, target).state, RobotState::Dip);

    // Run a couple more frames so the death explosion exists, then
    // draw everything into the CPU queues.
    crate::matrix_game::effects::effects_takt(&mut effects, 10.0, &map, &mut objs, &mut rng);
    let mut q = BillboardQueue::default();
    let mut mq = MeshQueue::default();
    for e in &effects {
        e.draw(&mut q, &mut mq);
    }
    // Death explosion sparks/fire/debris must be visible.
    assert!(
        !q.billboards.is_empty() || !q.lines.is_empty(),
        "explosion produced no billboards"
    );
    assert!(!mq.draws.is_empty(), "no debris/projectile meshes queued");
    // The kill stamped a crater spawn for the decal system.
    assert!(
        objs.pending_spots
            .iter()
            .any(|s| s.kind == crate::matrix_game::effects::landscape_spot::SpotKind::Voronka),
        "no voronka crater queued"
    );

    // The decal geometry builds on the flat map.
    let mut spots = crate::matrix_game::effects::landscape_spot::LandscapeSpots::default();
    let pend: Vec<_> = objs.pending_spots.drain(..).collect();
    for sp in &pend {
        spots.spawn(&map, sp);
    }
    assert!(!spots.spots.is_empty());
    assert!(!spots.spots[0].indices.is_empty());
}

#[test]
fn laser_beam_visual_tracks_while_firing_and_clears_on_fire_end() {
    use crate::matrix_game::effects::weapon::WEAPON_LASER;
    install_test_config();
    let map = GameMap::test_flat(32, 32, 0.0);
    let mut objs = Objects::new();
    let mut rng = Rnd::new(7);

    let mut w = WeaponEffect::new(WEAPON_LASER, 0, WeaponHandler::None);
    w.modify(Vec3::new(100.0, 100.0, 10.0), Vec3::X, Vec3::ZERO);
    let wid = objs.weapons.create(w);
    objs.weapons
        .get_mut(wid)
        .unwrap()
        .fire_begin(Vec3::ZERO, None);
    weapon_takt(&mut objs, wid, 10.0, &map, &mut rng);
    assert!(objs.weapons.get(wid).unwrap().beam.is_some(), "no beam");

    objs.weapons.get_mut(wid).unwrap().fire_end();
    assert!(
        objs.weapons.get(wid).unwrap().beam.is_none(),
        "beam not cleared"
    );
}
