//! Does a robot keep firing at a turret after it's destroyed?
//! Player robot ordered to attack an enemy turret with low HP; after
//! the turret dies we watch whether the robot's weapons keep firing.

use matrixgame_rs::matrix_game::logic::{robot_mut, robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::{MapStatic, ObjectId};
use matrixgame_rs::matrix_game::object_cannon::{Cannon, CannonState};
use matrixgame_rs::matrix_game::robot::{ChassisKind, Robot, RobotState};
use matrixgame_rs::matrix_game::side::CurrSel;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn spawn_robot(game: &mut MapLogic, map: &GameMap, mx: i32, my: i32, side: i32) -> ObjectId {
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let (x, y) = (mx as f32 * gs, my as f32 * gs);
    let z = map.get_z(x, y);
    let mut r = Robot::new(glam::Vec3::new(x, y, z), side, ChassisKind::Track);
    r.state = RobotState::Idle;
    r.config.weapon[0].ty = matrixgame_rs::matrix_game::object_robot::RobotUnitType::Weapon;
    r.config.weapon[0].kind = matrixgame_rs::matrix_game::config::RobotUnitKind(8);
    r.config.chassis.ty = matrixgame_rs::matrix_game::object_robot::RobotUnitType::Chassis;
    r.config.chassis.kind = matrixgame_rs::matrix_game::config::RobotUnitKind(3);
    r.hit_point = 4000.0;
    r.hit_point_max = 4000.0;
    let id = game.objects.spawn(Box::new(r));
    if let Some(rm) = robot_mut(&mut game.objects, id) {
        rm.self_id = Some(id);
    }
    game.objects.add_lt(id);
    id
}

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Warn)
        .init();
    let data = std::fs::read("../Data/robots.pkg").expect("robots.pkg");
    let pkg = PkgArchive::from_bytes(data).unwrap();
    let cmap_name = pkg
        .list_files()
        .into_iter()
        .find(|f| f.ends_with(".CMAP"))
        .unwrap()
        .to_string();
    let cmap = pkg.read_file(&cmap_name).unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let dat = std::fs::read("../Data/robots.dat").expect("robots.dat");
    let matrix_data = Storage::from_bytes(&dat).unwrap();

    let mut game = MapLogic::with_seed(1234);
    game.load_config(&matrix_data);

    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let mut spot = None;
    'outer: for my in (20..map.size_move_y as i32 - 20).step_by(4) {
        for mx in (20..map.size_move_x as i32 - 20).step_by(4) {
            if matrixgame_rs::matrix_game::logic::is_absence_wall(&map, 2, 4, mx, my)
                && matrixgame_rs::matrix_game::logic::is_absence_wall(&map, 2, 4, mx + 12, my)
            {
                spot = Some((mx, my));
                break 'outer;
            }
        }
    }
    let (mx, my) = spot.expect("no passable spot");

    let a = spawn_robot(&mut game, &map, mx, my, 1);

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
        let tp = matrixgame_rs::matrix_game::logic::get_map_pos(&game.objects, cid).unwrap();
        let no = game.sel_group_to_logic_group();
        game.pg_order_attack(&map, no, tp, Some(cid));
    }

    let mut turret_died_at: Option<i64> = None;
    let mut shots_after_death = 0i32;
    let mut last_fire_count = 0i32;
    for step_i in 0..3000 {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.takt(10);
        if step_i < 60 {
            let r = robot_ref(&game.objects, a).unwrap();
            let probe = game.rng.clone();
            println!(
                "takt {:>3}: rng_next={} rpos=({:.2},{:.2}) state={:?} orders={} fire={}",
                step_i,
                { let mut p = probe; p.next() },
                r.pos_x,
                r.pos_y,
                r.state,
                r.orders.len(),
                r.orders.has(matrixgame_rs::matrix_game::robot::OrderType::Fire)
            );
        }

        let cannon_state = game.objects.get(cid).map(|o| {
            let c: &Cannon = unsafe { &*(o as *const dyn MapStatic as *const Cannon) };
            (c.state, c.hit_point)
        });
        let mut fire_count = 0;
        for w in game.objects.weapons.iter() {
            fire_count += w.fire_count();
        }

        if turret_died_at.is_none() {
            match cannon_state {
                Some((CannonState::Dip, _)) | None => {
                    turret_died_at = Some(game.elapsed_ms);
                    last_fire_count = fire_count;
                    println!("turret died at t={}ms (fire_count={})", game.elapsed_ms, fire_count);
                }
                _ => {}
            }
        } else if game.elapsed_ms > turret_died_at.unwrap() + 1500 {
            // Give a 1.5s grace for in-flight volleys, then count.
            shots_after_death += fire_count - last_fire_count;
            last_fire_count = fire_count;
        } else {
            last_fire_count = fire_count;
        }

        if game.elapsed_ms % 500 < 10 {
            let r = robot_ref(&game.objects, a).unwrap();
            let enemies: Vec<_> = r.env.enemies.iter().map(|e| e.enemy).collect();
            let mut wfire = false;
            for w in game.objects.weapons.iter() {
                wfire |= w.is_fire();
            }
            println!(
                "t={:>6}ms cannon={:?} env={:?} tgt={:?} fire_order={} weapon_is_fire={} shots={}",
                game.elapsed_ms,
                cannon_state,
                enemies,
                r.env.target_attack,
                r.orders.has(matrixgame_rs::matrix_game::robot::OrderType::Fire),
                wfire,
                fire_count
            );
        }
        if let Some(t0) = turret_died_at {
            if game.elapsed_ms > t0 + 8000 {
                break;
            }
        }
    }

    let r = robot_ref(&game.objects, a).unwrap();
    println!(
        "after death+1.5s: extra_shots={} target_attack={:?} fire_order={} env_enemies={}",
        shots_after_death,
        r.env.target_attack,
        r.orders.has(matrixgame_rs::matrix_game::robot::OrderType::Fire),
        r.env.enemy_cnt(),
    );
    if shots_after_death > 0 {
        println!("BUG REPRODUCED: robot keeps firing after the turret died");
        std::process::exit(1);
    }
    println!("OK: robot stopped firing");
}
