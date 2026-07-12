//! Gameplay regression tests on real map data (converted from the
//! one-off reproducer examples — see SIM.md for the harness recipe).
//! Each reproduced bug keeps its reproducer here. Skip gracefully when
//! Data/ is absent.

use matrixgame_rs::matrix_game::camera::{AutoFlyData, Camera};
use matrixgame_rs::matrix_game::config::RobotUnitKind;
use matrixgame_rs::matrix_game::effects::weapon::{WEAPON_GUN, WEAPON_LASER};
use matrixgame_rs::matrix_game::effects::GameEffect;
use matrixgame_rs::matrix_game::logic::{
    building_mut, get_map_pos, is_absence_wall, robot_mut, robot_ref, MapLogic,
};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::{MapStatic, ObjectId, SpawnerBotRequest};
use matrixgame_rs::matrix_game::object_cannon::{Cannon, CannonState};
use matrixgame_rs::matrix_game::object_robot::RobotUnitType;
use matrixgame_rs::matrix_game::robot::{ChassisKind, OrderType, Robot, RobotState};
use matrixgame_rs::matrix_game::side::{CurrSel, LogicActionType};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use matrixgame_rs::matrix_lib::three_g::vector_object::parse_vo;

fn load_data() -> Option<(PkgArchive, Storage)> {
    let _ = env_logger::builder().is_test(true).try_init();
    let Ok(pkg_data) = std::fs::read("../Data/robots.pkg") else {
        eprintln!("skipping: ../Data/robots.pkg not present");
        return None;
    };
    let Ok(dat_bytes) = std::fs::read("../Data/robots.dat") else {
        eprintln!("skipping: ../Data/robots.dat not present");
        return None;
    };
    Some((
        PkgArchive::from_bytes(pkg_data).expect("parse pkg"),
        Storage::from_bytes(&dat_bytes).expect("parse robots.dat"),
    ))
}

fn load_map(pkg: &PkgArchive, name: &str) -> GameMap {
    let cmap = pkg.read_file(name).expect("read CMAP");
    GameMap::from_cmap_bytes(&cmap).expect("parse map")
}

fn first_map(pkg: &PkgArchive) -> GameMap {
    let name = pkg
        .list_files()
        .into_iter()
        .find(|f| f.ends_with(".CMAP"))
        .expect("no CMAP")
        .to_string();
    load_map(pkg, &name)
}

/// Armed robot (plasma gun on pylon 1) at a move-cell position.
fn spawn_armed(
    game: &mut MapLogic,
    map: &GameMap,
    mx: i32,
    my: i32,
    side: i32,
    chassis: ChassisKind,
    chassis_kind: i32,
    hp: f32,
) -> ObjectId {
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let (x, y) = (mx as f32 * gs, my as f32 * gs);
    let z = map.get_z(x, y);
    let mut r = Robot::new(glam::Vec3::new(x, y, z), side, chassis);
    r.state = RobotState::Idle;
    r.config.weapon[0].ty = RobotUnitType::Weapon;
    r.config.weapon[0].kind = RobotUnitKind(8);
    r.config.chassis.ty = RobotUnitType::Chassis;
    r.config.chassis.kind = RobotUnitKind(chassis_kind);
    r.hit_point = hp;
    r.hit_point_max = hp;
    let id = game.objects.spawn(Box::new(r));
    if let Some(r) = robot_mut(&mut game.objects, id) {
        r.self_id = Some(id);
    }
    game.objects.add_lt(id);
    id
}

fn install_weapon_matrices(pkg: &PkgArchive) {
    for idx in 1..=6usize {
        if let Ok(data) = pkg.read_file(&format!("MATRIX/ROBOT/ARMOR{}.VO", idx)) {
            let vo = parse_vo(&data).unwrap();
            let m = matrixgame_rs::matrix_game::map::weapon_matrix_from_vo(&vo);
            matrixgame_rs::matrix_game::map::set_weapon_matrix_for(RobotUnitKind(idx as i32), m);
        }
    }
}

/// Player attack-order pipeline end to end: select → PGOrderAttack →
/// the env/GatherInfo/FirePL chain must destroy the target.
#[test]
fn attack_order_opens_fire() {
    let Some((pkg, dat)) = load_data() else { return };
    let map = first_map(&pkg);
    let mut game = MapLogic::with_seed(1234);
    game.load_config(&dat);

    let mut spot = None;
    'outer: for my in (20..map.size_move_y as i32 - 20).step_by(4) {
        for mx in (20..map.size_move_x as i32 - 20).step_by(4) {
            if is_absence_wall(&map, 2, 4, mx, my) && is_absence_wall(&map, 2, 4, mx + 10, my) {
                spot = Some((mx, my));
                break 'outer;
            }
        }
    }
    let (mx, my) = spot.expect("no passable spot");

    let a = spawn_armed(&mut game, &map, mx, my, 1, ChassisKind::Track, 3, 400.0);
    let enemy = spawn_armed(&mut game, &map, mx + 10, my, 2, ChassisKind::Track, 3, 400.0);
    game.ensure_sides_from_objects();

    game.player_side
        .select_replace(vec![a], Some(a), CurrSel::RobotsSelected);
    game.sync_group_from_selection();
    {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        let tp = get_map_pos(&game.objects, enemy).unwrap();
        let no = game.sel_group_to_logic_group();
        game.pg_order_attack(&map, no, tp, Some(enemy));
    }

    for _ in 0..1500 {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.takt(10);
        if robot_ref(&game.objects, enemy).map(|e| e.hit_point <= 0.0).unwrap_or(true) {
            return; // enemy destroyed — pipeline works
        }
    }
    let r = robot_ref(&game.objects, a).unwrap();
    panic!(
        "enemy alive after 15s: env={} target_attack={:?} place={} orders={}",
        r.env.enemy_cnt(),
        r.env.target_attack,
        r.env.place,
        r.orders.len()
    );
}

/// "Robots go through static objects": a crowd shoved across a blocked
/// cluster must never have a center inside a chassis-blocked move-cell.
#[test]
fn robots_dont_cross_blocked_cells() {
    const NSH: usize = 3; // hovercraft chassis index
    let Some((pkg, dat)) = load_data() else { return };
    let map = first_map(&pkg);

    let blocked = |mx: i32, my: i32| -> bool {
        map.move_cell(mx, my).map(|c| c.get_type(NSH) != 0xff).unwrap_or(false)
    };

    // Rows crossing a blocked cluster: free run, 2..8 blocked, free run.
    let sy = map.size_move_y as i32;
    let sx = map.size_move_x as i32;
    let mut crossings: Vec<(i32, i32, i32)> = Vec::new(); // (y, x_start, x_end)
    for my in (10..sy - 10).step_by(7) {
        let mut mx = 10;
        while mx < sx - 30 {
            if !(0..6).all(|i| !blocked(mx + i, my)) {
                mx += 1;
                continue;
            }
            let bstart = mx + 6;
            if !blocked(bstart, my) {
                mx += 1;
                continue;
            }
            let mut blen = 0;
            while blen < 8 && blocked(bstart + blen, my) {
                blen += 1;
            }
            if blen >= 8 {
                mx = bstart + blen;
                continue;
            }
            let after = bstart + blen;
            if (0..6).all(|i| !blocked(after + i, my))
                && is_absence_wall(&map, NSH, 4, mx, my - 2)
                && is_absence_wall(&map, NSH, 4, after + 1, my - 2)
            {
                crossings.push((my, mx + 2, after + 3));
                mx = after + 6;
            } else {
                mx += 1;
            }
        }
    }
    if crossings.is_empty() {
        eprintln!("skipping: no suitable blocked clusters on this map");
        return;
    }

    for &(my, x0, x1) in crossings.iter().take(8) {
        let mut game = MapLogic::with_seed(42 + my);
        game.load_config(&dat);
        // Crowd: robots clustered before the obstacle, all ordered to
        // the same cell past it — they shove each other near the wall.
        let mut ids = Vec::new();
        for (dx, dy) in [(0, 0), (0, 5), (0, -5), (-5, 0), (-5, 5), (-5, -5)] {
            let (sx_, sy_) = (x0 + dx, my + dy);
            if is_absence_wall(&map, NSH, 4, sx_ - 2, sy_ - 2) {
                ids.push(spawn_armed(&mut game, &map, sx_, sy_, 1, ChassisKind::Hovercraft, 4, 400.0));
            }
        }
        game.ensure_sides_from_objects();
        game.player_side
            .select_replace(ids.clone(), Some(ids[0]), CurrSel::RobotsSelected);
        game.sync_group_from_selection();
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            let no = game.sel_group_to_logic_group();
            game.pg_order_move_to(&map, no, (x1, my));
        }
        for _ in 0..3000 {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(16);
            for &a in &ids {
                let Some(r) = robot_ref(&game.objects, a) else { continue };
                let cmx = (r.pos_x / GameMap::GLOBAL_SCALE_MOVE) as i32;
                let cmy = (r.pos_y / GameMap::GLOBAL_SCALE_MOVE) as i32;
                assert!(
                    !blocked(cmx, cmy),
                    "robot inside blocked cell ({cmx},{cmy}) at t={}ms (run y={my} x{x0}->{x1})",
                    game.elapsed_ms
                );
            }
        }
    }
}

/// A robot ordered onto a turret must stop firing once the turret dies.
#[test]
fn robot_stops_firing_after_turret_dies() {
    let Some((pkg, dat)) = load_data() else { return };
    let map = first_map(&pkg);
    let mut game = MapLogic::with_seed(1234);
    game.load_config(&dat);

    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let mut spot = None;
    'outer: for my in (20..map.size_move_y as i32 - 20).step_by(4) {
        for mx in (20..map.size_move_x as i32 - 20).step_by(4) {
            if is_absence_wall(&map, 2, 4, mx, my) && is_absence_wall(&map, 2, 4, mx + 12, my) {
                spot = Some((mx, my));
                break 'outer;
            }
        }
    }
    let (mx, my) = spot.expect("no passable spot");
    let a = spawn_armed(&mut game, &map, mx, my, 1, ChassisKind::Track, 3, 4000.0);

    // Enemy turret 12 cells away with LOW hp so it dies quickly.
    let (tx, ty) = ((mx + 12) as f32 * gs, my as f32 * gs);
    let tz = map.get_z(tx, ty);
    let mut c = Cannon::new(glam::Vec2::new(tx, ty), tz, 0.0, 2, 1, None, 0);
    c.hit_point = 60.0;
    c.hit_point_max = 60.0;
    let cid = game.objects.spawn(Box::new(c));
    if let Some(obj) = game.objects.get_mut(cid) {
        let cm: &mut Cannon = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Cannon) };
        cm.self_id = Some(cid);
    }
    game.objects.add_lt(cid);
    game.ensure_sides_from_objects();

    game.player_side
        .select_replace(vec![a], Some(a), CurrSel::RobotsSelected);
    game.sync_group_from_selection();
    {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        let tp = get_map_pos(&game.objects, cid).unwrap();
        let no = game.sel_group_to_logic_group();
        game.pg_order_attack(&map, no, tp, Some(cid));
    }

    let mut turret_died_at: Option<i64> = None;
    let mut shots_after_death = 0i32;
    let mut last_fire_count = 0i32;
    for _ in 0..3000 {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.takt(10);

        let cannon_state = game.objects.get(cid).map(|o| {
            let c: &Cannon = unsafe { &*(o as *const dyn MapStatic as *const Cannon) };
            c.state
        });
        let mut fire_count = 0;
        for w in game.objects.weapons.iter() {
            fire_count += w.fire_count();
        }

        if turret_died_at.is_none() {
            if matches!(cannon_state, Some(CannonState::Dip) | None) {
                turret_died_at = Some(game.elapsed_ms);
            }
            last_fire_count = fire_count;
        } else if game.elapsed_ms > turret_died_at.unwrap() + 1500 {
            // 1.5s grace for in-flight volleys, then count.
            shots_after_death += fire_count - last_fire_count;
            last_fire_count = fire_count;
        } else {
            last_fire_count = fire_count;
        }

        if let Some(t0) = turret_died_at {
            if game.elapsed_ms > t0 + 8000 {
                break;
            }
        }
    }

    assert!(turret_died_at.is_some(), "turret never died — attack never landed");
    let r = robot_ref(&game.objects, a).unwrap();
    assert!(
        shots_after_death == 0,
        "robot kept firing after the turret died: extra_shots={} target_attack={:?} fire_order={}",
        shots_after_death,
        r.env.target_attack,
        r.orders.has(OrderType::Fire)
    );
}

/// Killing a building must scatter temporary smoke spawners over the
/// ruins that emit Smoke effects (MatrixObjectBuilding.cpp:726-755).
#[test]
fn building_death_scatters_ruin_smoke() {
    let Some((pkg, dat)) = load_data() else { return };
    let map = load_map(&pkg, "MATRIX/MAP/TRAINING.CMAP");
    let mut game = MapLogic::with_seed(3);
    game.load_config(&dat);
    game.spawn_buildings(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);

    // Deplete the first building; the DIP state machine takes it from there.
    let ids: Vec<_> = game.objects.iter_units().collect();
    let bid = ids
        .into_iter()
        .find(|&id| building_mut(&mut game.objects, id).is_some())
        .expect("a building");
    if let Some(b) = building_mut(&mut game.objects, bid) {
        b.hit_point = 1.0;
    }
    {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.objects
            .apply_damage(bid, WEAPON_GUN, glam::Vec3::ZERO, glam::Vec3::Z, 2, None);
    }

    for _ in 0..(30 * 20) {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
        }
        if game.effects.iter().any(|e| matches!(e, GameEffect::Smoke(_))) {
            return; // ruin smoke seen
        }
    }
    panic!("no ruin smoke within 30s of building death");
}

/// An AI-side bot from a map robot-spawner must get its own Attack
/// logic group and go to war (MatrixObject.cpp:1452-1455).
#[test]
fn spawner_bot_gets_attack_group() {
    let Some((pkg, dat)) = load_data() else { return };
    let map = load_map(&pkg, "MATRIX/MAP/TRAINING.CMAP");
    let mut game = MapLogic::with_seed(7);
    game.load_config(&dat);
    game.spawn_buildings(&map);
    game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);

    let cat = matrixgame_rs::matrix_game::config::robot_spawn_config();
    assert!(cat.choose(1, 0).is_some(), "robots.dat has no spawner-bot catalogue");

    game.objects.pending_spawner_bots.push(SpawnerBotRequest {
        pos: glam::Vec3::new(700.0, 1500.0, 0.0),
        number: 1,
        pick: 0,
        sens_radius: 300.0,
    });
    let before: Vec<_> = game.objects.iter_units().collect();
    {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.takt(50);
    }
    let dispatched = game
        .objects
        .iter_units()
        .filter(|id| !before.contains(id))
        .filter_map(|id| robot_ref(&game.objects, id))
        .any(|r| {
            r.side != 1
                && r.team == -1
                && r.group_logic >= 0
                && game.side_by_id(r.side).map_or(false, |s| {
                    s.logic_groups[r.group_logic as usize].action.ty == LogicActionType::Attack
                })
        });
    assert!(dispatched, "no spawned bot got an Attack logic group");
}

/// Pre-placed TRAINING enemies (CMAP group 0 → team -1 → Defence
/// groups) must not attack the player unprovoked.
#[test]
fn preplaced_training_enemies_stay_passive() {
    let Some((pkg, dat)) = load_data() else { return };
    let map = load_map(&pkg, "MATRIX/MAP/TRAINING.CMAP");
    let mut game = MapLogic::with_seed(1234);
    game.load_config(&dat);
    game.spawn_buildings(&map);
    let robot_ids = game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.accrue_resources(100_000);

    let enemies = robot_ids
        .iter()
        .filter(|&&id| robot_ref(&game.objects, id).map_or(false, |r| r.side != 1))
        .count();
    assert!(enemies > 0, "TRAINING has pre-placed enemies");

    let player_buildings = |game: &MapLogic| -> (usize, f32) {
        let mut cnt = 0;
        let mut hp = 0.0;
        for id in game.objects.iter_units() {
            if let Some(b) = matrixgame_rs::matrix_game::logic::building_ref(&game.objects, id) {
                if b.side == 1 {
                    cnt += 1;
                    hp += b.hit_point;
                }
            }
        }
        (cnt, hp)
    };
    let (c0, hp0) = player_buildings(&game);

    // 10 sim-minutes at 50ms steps.
    for _ in 0..(10 * 60 * 20) {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.takt(50);
    }

    let (cnt, hp) = player_buildings(&game);
    assert!(
        game.game_over != Some(false) && cnt == c0 && hp >= hp0,
        "player attacked unprovoked: buildings {c0}->{cnt}, hp {hp0:.0}->{hp:.0}, game_over={:?}",
        game.game_over
    );
}

/// FLYCAM autopilot: damage feeds war pairs, the camera flies toward
/// the fight and stays finite.
#[test]
fn flycam_chases_war_pairs() {
    let Some((pkg, dat)) = load_data() else { return };
    let map = load_map(&pkg, "MATRIX/MAP/ATOLL.CMAP");
    let mut game = MapLogic::with_seed(7);
    game.load_config(&dat);

    // First dry spot with dry neighbors (ATOLL is mostly water).
    let mut land = None;
    'scan: for gy in (4..map.size_y - 4).step_by(2) {
        for gx in (4..map.size_x - 4).step_by(2) {
            let (x, y) = (gx as f32 * 20.0, gy as f32 * 20.0);
            if map.get_z(x, y) > 0.0 && map.get_z(x + 60.0, y) > 0.0 {
                land = Some((x, y));
                break 'scan;
            }
        }
    }
    let (fx, fy) = land.expect("no land found");

    let spawn = |game: &mut MapLogic, x: f32, y: f32, side: i32| {
        let z = map.get_z(x, y);
        let mut r = Robot::new(glam::Vec3::new(x, y, z), side, ChassisKind::Track);
        r.state = RobotState::Idle;
        game.objects.spawn(Box::new(r))
    };
    let a = spawn(&mut game, fx, fy, 1);
    let b = spawn(&mut game, fx + 60.0, fy, 2);

    game.objects.fly_cam = true;

    let mut cam = Camera::new(16.0 / 9.0);
    cam.set_map(map.size_x as f32 * 20.0, map.size_y as f32 * 20.0);
    cam.set_xy_strategy([1500.0, 1500.0]); // park far from the fight
    cam.takt(50.0);
    let start = cam.eye_pos_world();

    let mut afd = AutoFlyData::new(&cam);
    let mut pair_events = 0usize;
    for i in 0..600 {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
        }
        // Poke both robots so the Damage path emits war pairs.
        if i % 10 == 0 {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.objects.apply_damage(
                a,
                WEAPON_LASER,
                glam::Vec3::new(fx, fy, 10.0),
                glam::Vec3::X,
                2,
                Some(b),
            );
        }
        let pairs: Vec<_> = game.objects.pending_war_pairs.drain(..).collect();
        pair_events += pairs.len();
        for (t, at) in pairs {
            afd.add_war_pair(t, at, &game.objects);
        }
        afd.takt(
            50.0,
            &mut cam,
            &map,
            &game.objects,
            &mut game.rng,
            game.elapsed_ms as i32,
            false,
        );
        let e = cam.eye_pos_world();
        assert!(e.is_finite(), "camera went non-finite at tick {i}: {e}");
    }

    let end = cam.eye_pos_world();
    let fight = glam::Vec3::new(fx + 30.0, fy, map.get_z(fx + 30.0, fy));
    let d_start = (start - fight).truncate().length();
    let d_end = (end - fight).truncate().length();
    assert!(pair_events > 0, "damage never queued a war pair");
    assert!(
        d_end < d_start * 0.5,
        "camera did not fly toward the fight: {d_start:.0} -> {d_end:.0}"
    );
}

/// Standoff regression ("enemies interlocked, nobody shoots"): hostile
/// robots inside fire range must exchange damage within 3 sim-minutes.
fn assert_hostiles_engage(interlocked: bool) {
    let Some((pkg, dat)) = load_data() else { return };
    install_weapon_matrices(&pkg);
    let map = load_map(&pkg, "MATRIX/MAP/ATOLL.CMAP");

    let mut game = MapLogic::with_seed(7);
    game.load_config(&dat);
    game.spawn_buildings(&map);
    game.spawn_ruins(&map);
    game.spawn_robots(&map);

    let find_spot = |w: i32, h: i32| -> (i32, i32) {
        for my in (20..map.size_move_y as i32 - 20 - h).step_by(4) {
            for mx in (20..map.size_move_x as i32 - 20 - w).step_by(4) {
                let ok = (0..h).step_by(4).all(|dy| {
                    (0..w).step_by(4).all(|dx| is_absence_wall(&map, 2, 4, mx + dx, my + dy))
                });
                if ok {
                    return (mx, my);
                }
            }
        }
        panic!("no flat spot");
    };

    let mut ids: Vec<ObjectId> = Vec::new();
    let mut spawns: Vec<(ObjectId, i32, i32, i32)> = Vec::new();
    if interlocked {
        // Alternating sides, 3 move-cells apart — robots are 4 cells
        // wide, so hostiles physically touch.
        let (mx, my) = find_spot(36, 10);
        for i in 0..5 {
            ids.push(spawn_armed(&mut game, &map, mx + i * 6, my, 2, ChassisKind::Track, 3, 400.0));
            ids.push(spawn_armed(&mut game, &map, mx + i * 6 + 3, my + 3, 3, ChassisKind::Track, 3, 400.0));
        }
    } else {
        // Two columns ~240 units apart, ordered through each other.
        let (mx, my) = find_spot(28, 30);
        for i in 0..5 {
            let a = spawn_armed(&mut game, &map, mx + i * 5, my, 2, ChassisKind::Track, 3, 400.0);
            let b = spawn_armed(&mut game, &map, mx + i * 5, my + 24, 3, ChassisKind::Track, 3, 400.0);
            ids.push(a);
            ids.push(b);
            spawns.push((a, 2, mx + i * 5, my));
            spawns.push((b, 3, mx + i * 5, my + 24));
        }
    }
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.accrue_resources(100_000);

    if !interlocked {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        for &(id, side, sx, sy) in &spawns {
            let ty = if side == 2 { sy + 24 } else { sy - 24 };
            if let Some(r) = robot_mut(&mut game.objects, id) {
                r.move_to(sx, ty);
            }
        }
    }

    let engaged = |game: &MapLogic| {
        ids.iter().any(|&id| {
            robot_ref(&game.objects, id)
                .map(|r| !r.is_live() || r.hit_point < r.hit_point_max)
                .unwrap_or(true)
        })
    };
    for step in 0..(180 * 20) {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
            game.graphic_takt(50);
        }
        // Headless: drain queues the renderer would consume.
        game.objects.pending_sounds.clear();
        game.sound_queue.clear();
        game.objects.weapons.freed.clear();
        let _ = matrixgame_rs::matrix_game::interface::sound::drain();
        if step % 100 == 0 && engaged(&game) {
            return;
        }
    }
    assert!(engaged(&game), "STANDOFF: no damage exchanged in 3 sim-minutes");
}

#[test]
fn adjacent_hostiles_engage() {
    assert_hostiles_engage(true);
}

#[test]
fn crossing_columns_engage() {
    assert_hostiles_engage(false);
}
