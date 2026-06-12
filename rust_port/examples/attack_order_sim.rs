//! Headless end-to-end check of the player attack-order pipeline:
//! select robots → PGOrderAttack an enemy → run the logic takt and
//! verify the env/GatherInfo/FirePL chain actually opens fire.

use matrixgame_rs::matrix_game::logic::{robot_mut, robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::ObjectId;
use matrixgame_rs::matrix_game::robot::{ChassisKind, Robot, RobotState};
use matrixgame_rs::matrix_game::side::CurrSel;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn spawn(game: &mut MapLogic, map: &GameMap, mx: i32, my: i32, side: i32) -> ObjectId {
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let x = mx as f32 * gs;
    let y = my as f32 * gs;
    let z = map.get_z(x, y);
    let mut r = Robot::new(glam::Vec3::new(x, y, z), side, ChassisKind::Track);
    r.state = RobotState::Idle;
    // Plasma gun on pylon 1 so weapons init.
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
        .expect("no CMAP")
        .to_string();
    let cmap = pkg.read_file(&cmap_name).unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let dat = std::fs::read("../Data/robots.dat").expect("robots.dat");
    let matrix_data = Storage::from_bytes(&dat).unwrap();

    let mut game = MapLogic::with_seed(1234);
    game.load_config(&matrix_data);

    // Find a passable spot: scan for a place where both cells stand.
    let mut spot = None;
    'outer: for my in (20..map.size_move_y as i32 - 20).step_by(4) {
        for mx in (20..map.size_move_x as i32 - 20).step_by(4) {
            if matrixgame_rs::matrix_game::logic::is_absence_wall(&map, 2, 4, mx, my)
                && matrixgame_rs::matrix_game::logic::is_absence_wall(&map, 2, 4, mx + 10, my)
            {
                spot = Some((mx, my));
                break 'outer;
            }
        }
    }
    let (mx, my) = spot.expect("no passable spot");
    println!("spot: ({}, {})", mx, my);

    let a = spawn(&mut game, &map, mx, my, 1);
    let enemy = spawn(&mut game, &map, mx + 10, my, 2);
    game.ensure_sides_from_objects();

    // Select robot a (mirror click) + sync group.
    game.player_side
        .select_replace(vec![a], Some(a), CurrSel::RobotsSelected);
    game.sync_group_from_selection();

    // Issue the attack order at the enemy.
    {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        let tp = matrixgame_rs::matrix_game::logic::get_map_pos(&game.objects, enemy).unwrap();
        let no = game.sel_group_to_logic_group();
        game.pg_order_attack(&map, no, tp, Some(enemy));
    }

    // Run 15 simulated seconds.
    for step in 0..1500 {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.takt(10);
        if step % 300 == 0 {
            let r = robot_ref(&game.objects, a).unwrap();
            let gl = r.group_logic;
            let order = if gl >= 0 {
                format!("{:?}", game.player_side.player_groups[gl as usize].order)
            } else {
                "-".into()
            };
            let ecnt = r.env.enemy_cnt();
            let ta = r.env.target_attack.is_some();
            let fire = r.orders.has(matrixgame_rs::matrix_game::robot::OrderType::Fire);
            let ehp = robot_ref(&game.objects, enemy).map(|e| e.hit_point);
            println!(
                "t={:>5}ms gl={} order={} env={} target_attack={} fire_order={} enemy_hp={:?}",
                game.elapsed_ms, gl, order, ecnt, ta, fire, ehp
            );
        }
        if robot_ref(&game.objects, enemy).map(|e| e.hit_point <= 0.0).unwrap_or(true) {
            println!("ENEMY DESTROYED at t={}ms — attack pipeline OK", game.elapsed_ms);
            return;
        }
    }
    let r = robot_ref(&game.objects, a).unwrap();
    println!(
        "FAIL: enemy alive after 15s. env={} target_attack={:?} place={} orders={}",
        r.env.enemy_cnt(),
        r.env.target_attack,
        r.env.place,
        r.orders.len()
    );
    std::process::exit(1);
}
