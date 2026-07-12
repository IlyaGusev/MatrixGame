//! Repro probe for the "enemies interlocked, nobody shoots" screenshot:
//! spawn two hostile AI-side robot clumps directly adjacent (scenario A)
//! and two columns marched through each other (scenario B) on ATOLL,
//! then watch env/war/fire state. Hostile robots inside fire range must
//! exchange damage within a minute — anything else is the standoff bug.

use matrixgame_rs::matrix_game::logic::{robot_mut, robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::{MapStatic, ObjectId};
use matrixgame_rs::matrix_game::robot::{ChassisKind, OrderType, Robot, RobotState};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn spawn(game: &mut MapLogic, map: &GameMap, mx: i32, my: i32, side: i32) -> ObjectId {
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let x = mx as f32 * gs;
    let y = my as f32 * gs;
    let z = map.get_z(x, y);
    let mut r = Robot::new(glam::Vec3::new(x, y, z), side, ChassisKind::Track);
    r.state = RobotState::Idle;
    r.config.weapon[0].ty = matrixgame_rs::matrix_game::object_robot::RobotUnitType::Weapon;
    r.config.weapon[0].kind = matrixgame_rs::matrix_game::config::RobotUnitKind(8);
    r.config.chassis.ty = matrixgame_rs::matrix_game::object_robot::RobotUnitType::Chassis;
    r.config.chassis.kind = matrixgame_rs::matrix_game::config::RobotUnitKind(3);
    r.hit_point = 400.0;
    r.hit_point_max = 400.0;
    let id = game.objects.spawn(Box::new(r));
    if let Some(r) = robot_mut(&mut game.objects, id) {
        r.self_id = Some(id);
    }
    game.objects.add_lt(id);
    id
}

fn find_spot(map: &GameMap, w: i32, h: i32) -> (i32, i32) {
    for my in (20..map.size_move_y as i32 - 20 - h).step_by(4) {
        for mx in (20..map.size_move_x as i32 - 20 - w).step_by(4) {
            let mut ok = true;
            'grid: for dy in (0..h).step_by(4) {
                for dx in (0..w).step_by(4) {
                    if !matrixgame_rs::matrix_game::logic::is_absence_wall(
                        map, 2, 4, mx + dx, my + dy,
                    ) {
                        ok = false;
                        break 'grid;
                    }
                }
            }
            if ok {
                return (mx, my);
            }
        }
    }
    panic!("no flat spot");
}

fn report(game: &MapLogic, ids: &[(ObjectId, i32)]) {
    let mut hp = [0.0f32; 2];
    let mut env = [0i32; 2];
    let mut fire = [0i32; 2];
    let mut alive = [0i32; 2];
    let mut min_hostile_d = f32::MAX;
    for (i, &(ida, sa)) in ids.iter().enumerate() {
        let Some(ra) = robot_ref(&game.objects, ida) else { continue };
        if !ra.is_live() {
            continue;
        }
        let k = if sa == 2 { 0 } else { 1 };
        alive[k] += 1;
        hp[k] += ra.hit_point;
        env[k] += ra.env.enemy_cnt();
        fire[k] += ra.orders.has(OrderType::Fire) as i32;
        for &(idb, sb) in ids.iter().skip(i + 1) {
            if sb == sa {
                continue;
            }
            let Some(rb) = robot_ref(&game.objects, idb) else { continue };
            if !rb.is_live() {
                continue;
            }
            let d = ((ra.pos_x - rb.pos_x).powi(2) + (ra.pos_y - rb.pos_y).powi(2)).sqrt();
            min_hostile_d = min_hostile_d.min(d);
        }
    }
    let wars: Vec<String> = [2, 3]
        .iter()
        .filter_map(|&sid| game.side_by_id(sid))
        .map(|s| {
            let t: Vec<String> = s
                .teams
                .iter()
                .filter(|t| t.robot_cnt > 0)
                .map(|t| format!("{:?}war={}", t.action.ty, t.war))
                .collect();
            t.join(",")
        })
        .collect();
    println!(
        "t={:>6}ms alive={}/{} hp=({:.0},{:.0}) env=({},{}) fire=({},{}) min_d={:.0} teams={:?}",
        game.elapsed_ms, alive[0], alive[1], hp[0], hp[1], env[0], env[1], fire[0], fire[1],
        min_hostile_d, wars,
    );
}

fn run_scenario(name: &str, pkg: &PkgArchive, dat: &Storage, interlocked: bool) {
    println!("=== scenario: {name} ===");
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();

    let mut game = MapLogic::with_seed(7);
    game.load_config(dat);
    game.spawn_buildings(&map);
    game.spawn_ruins(&map);
    game.spawn_robots(&map);

    let (mx, my) = if interlocked {
        find_spot(&map, 36, 10)
    } else {
        find_spot(&map, 28, 30)
    };
    println!("spot: ({mx},{my})");

    // (id, side, spawn mx, spawn my)
    let mut ids: Vec<(ObjectId, i32)> = Vec::new();
    let mut spawns: Vec<(ObjectId, i32, i32, i32)> = Vec::new();
    if interlocked {
        // Alternating sides, 3 move-cells apart — robots are 4 cells
        // wide, so hostiles physically touch like the screenshot.
        for i in 0..5 {
            let a = spawn(&mut game, &map, mx + i * 6, my, 2);
            let b = spawn(&mut game, &map, mx + i * 6 + 3, my + 3, 3);
            ids.push((a, 2));
            ids.push((b, 3));
        }
    } else {
        // Two columns 40 cells (~400 units) apart, ordered through
        // each other.
        for i in 0..5 {
            let a = spawn(&mut game, &map, mx + i * 5, my, 2);
            let b = spawn(&mut game, &map, mx + i * 5, my + 24, 3);
            ids.push((a, 2));
            ids.push((b, 3));
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

    for step in 0..(180 * 20) {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
            game.graphic_takt(50);
        }
        game.objects.pending_sounds.clear();
        game.sound_queue.clear();
        game.objects.weapons.freed.clear();
        let _ = matrixgame_rs::matrix_game::interface::sound::drain();
        if step % 200 == 0 {
            report(&game, &ids);
        }
    }
    report(&game, &ids);

    let dmg: f32 = ids
        .iter()
        .filter_map(|&(id, _)| robot_ref(&game.objects, id))
        .map(|r| r.hit_point_max - r.hit_point.max(0.0))
        .sum();
    let dead = ids
        .iter()
        .filter(|&&(id, _)| {
            robot_ref(&game.objects, id).map(|r| !r.is_live()).unwrap_or(true)
        })
        .count();
    if dmg <= 0.0 && dead == 0 {
        println!("VERDICT {name}: STANDOFF — no damage exchanged in 3min");
    } else {
        println!("VERDICT {name}: engaged (damage={dmg:.0}, dead={dead})");
    }
}

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Warn)
        .init();
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let dat = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();

    for idx in 1..=6usize {
        if let Ok(data) = pkg.read_file(&format!("MATRIX/ROBOT/ARMOR{}.VO", idx)) {
            let vo = matrixgame_rs::matrix_lib::three_g::vector_object::parse_vo(&data).unwrap();
            let m = matrixgame_rs::matrix_game::map::weapon_matrix_from_vo(&vo);
            matrixgame_rs::matrix_game::map::set_weapon_matrix_for(
                matrixgame_rs::matrix_game::config::RobotUnitKind(idx as i32),
                m,
            );
        }
    }

    run_scenario("A-interlocked", &pkg, &dat, true);
    run_scenario("B-columns-crossing", &pkg, &dat, false);
}
