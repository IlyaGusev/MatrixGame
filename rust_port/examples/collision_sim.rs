//! Headless repro for "robots go through static objects": order a robot
//! across a blocked (static-object) region and verify its center never
//! enters a move-cell that is blocked for its chassis.

use matrixgame_rs::matrix_game::logic::{robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::ObjectId;
use matrixgame_rs::matrix_game::robot::{ChassisKind, Robot, RobotState};
use matrixgame_rs::matrix_game::side::CurrSel;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

const NSH: usize = 3; // hovercraft chassis index

fn spawn(game: &mut MapLogic, map: &GameMap, mx: i32, my: i32, side: i32) -> ObjectId {
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let x = mx as f32 * gs;
    let y = my as f32 * gs;
    let z = map.get_z(x, y);
    let mut r = Robot::new(glam::Vec3::new(x, y, z), side, ChassisKind::Hovercraft);
    r.state = RobotState::Idle;
    r.config.weapon[0].ty = matrixgame_rs::matrix_game::object_robot::RobotUnitType::Weapon;
    r.config.weapon[0].kind = matrixgame_rs::matrix_game::config::RobotUnitKind(8);
    r.config.chassis.ty = matrixgame_rs::matrix_game::object_robot::RobotUnitType::Chassis;
    r.config.chassis.kind = matrixgame_rs::matrix_game::config::RobotUnitKind(4);
    r.hit_point = 400.0;
    r.hit_point_max = 400.0;
    let id = game.objects.spawn(Box::new(r));
    if let Some(r) = matrixgame_rs::matrix_game::logic::robot_mut(&mut game.objects, id) {
        r.self_id = Some(id);
    }
    game.objects.add_lt(id);
    id
}

fn blocked(map: &GameMap, mx: i32, my: i32) -> bool {
    map.move_cell(mx, my)
        .map(|c| c.get_type(NSH) != 0xff)
        .unwrap_or(false)
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

    // Find rows crossing a blocked cluster: passable run, >=2 blocked
    // cells, passable run. Collect several crossings across the map.
    let sy = map.size_move_y as i32;
    let sx = map.size_move_x as i32;
    let mut crossings: Vec<(i32, i32, i32)> = Vec::new(); // (y, x_start, x_end)
    for my in (10..sy - 10).step_by(7) {
        let mut mx = 10;
        while mx < sx - 30 {
            // need 6 free cells, then blocked run 2..8, then 6 free
            let free_before = (0..6).all(|i| !blocked(&map, mx + i, my));
            if !free_before {
                mx += 1;
                continue;
            }
            let bstart = mx + 6;
            if !blocked(&map, bstart, my) {
                mx += 1;
                continue;
            }
            let mut blen = 0;
            while blen < 8 && blocked(&map, bstart + blen, my) {
                blen += 1;
            }
            if blen >= 8 {
                mx = bstart + blen;
                continue;
            }
            let after = bstart + blen;
            let free_after = (0..6).all(|i| !blocked(&map, after + i, my));
            if free_after
                && matrixgame_rs::matrix_game::logic::is_absence_wall(&map, NSH, 4, mx, my - 2)
                && matrixgame_rs::matrix_game::logic::is_absence_wall(
                    &map,
                    NSH,
                    4,
                    after + 1,
                    my - 2,
                )
            {
                crossings.push((my, mx + 2, after + 3));
                mx = after + 6;
            } else {
                mx += 1;
            }
        }
    }
    println!("found {} crossings", crossings.len());
    if crossings.is_empty() {
        println!("SKIP: no suitable blocked clusters found");
        return;
    }

    let mut total_viol = 0usize;
    let mut runs = 0usize;
    for &(my, x0, x1) in crossings.iter().take(20) {
        let mut game = MapLogic::with_seed(42 + my);
        game.load_config(&matrix_data);
        // Crowd: 6 robots clustered before the obstacle, all ordered to
        // the same cell past it — they shove each other near the wall.
        let mut ids = Vec::new();
        for (dx, dy) in [(0, 0), (0, 5), (0, -5), (-5, 0), (-5, 5), (-5, -5)] {
            let (sx_, sy_) = (x0 + dx, my + dy);
            if matrixgame_rs::matrix_game::logic::is_absence_wall(&map, NSH, 4, sx_ - 2, sy_ - 2) {
                ids.push(spawn(&mut game, &map, sx_, sy_, 1));
            }
        }
        let n = ids.len();
        game.ensure_sides_from_objects();
        game.player_side
            .select_replace(ids.clone(), Some(ids[0]), CurrSel::RobotsSelected);
        game.sync_group_from_selection();
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            let no = game.sel_group_to_logic_group();
            game.pg_order_move_to(&map, no, (x1, my));
        }
        runs += 1;
        let mut viol = 0usize;
        let mut worst: Option<(i32, f32, f32)> = None;
        for _step in 0..3000 {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(16); // realistic ~60fps frame
            for &a in &ids {
                let Some(r) = robot_ref(&game.objects, a) else {
                    continue;
                };
                let cmx = (r.pos_x / GameMap::GLOBAL_SCALE_MOVE) as i32;
                let cmy = (r.pos_y / GameMap::GLOBAL_SCALE_MOVE) as i32;
                if blocked(&map, cmx, cmy) {
                    viol += 1;
                    if worst.is_none() {
                        worst = Some((game.elapsed_ms as i32, r.pos_x, r.pos_y));
                    }
                }
            }
        }
        let reached = ids
            .iter()
            .filter(|&&a| {
                robot_ref(&game.objects, a)
                    .map(|r| {
                        ((r.pos_x / 10.0) as i32 - x1).abs() <= 8
                            && ((r.pos_y / 10.0) as i32 - my).abs() <= 8
                    })
                    .unwrap_or(false)
            })
            .count();
        if viol > 0 {
            println!(
                "run y={} x{}→{}: {} VIOLATIONS first at {:?} reached={}/{}",
                my, x0, x1, viol, worst, reached, n
            );
        } else {
            println!("run y={} x{}→{}: clean, reached={}/{}", my, x0, x1, reached, n);
        }
        total_viol += viol;
    }
    println!("== {} runs, {} total violations", runs, total_viol);
    if total_viol > 0 {
        std::process::exit(1);
    }
}
