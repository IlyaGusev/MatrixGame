//! AI-deadlock probe: run the full atoll battle headless for 20 game
//! minutes and log per-side robot activity. A side whose live robots
//! all sit still (no position change, no orders in flight) for 60 s+
//! is flagged as stalled, with a dump of each robot's order state.
//!
//!   cargo run --example ai_stall_probe

use matrixgame_rs::matrix_game::logic::{robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map_static::MapStatic;
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use std::collections::HashMap;

fn main() {
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let dat = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();

    // Weapon matrices normally load in RobotsRenderer::new — headless
    // must populate them or every AI-built robot comes out weaponless.
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

    let mut game = MapLogic::with_seed(42);
    game.load_config(&dat);
    game.spawn_buildings(&map);
    game.spawn_ruins(&map);
    game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.init_effect_spawners(&map);
    game.accrue_resources(100_000);
    let stor = Storage::from_bytes(&cmap).unwrap();
    game.spawn_map_objects(&map, &stor);

    // last position + last time it changed, per robot.
    let mut last_move: HashMap<String, (glam::Vec2, i64)> = HashMap::new();
    let mut stall_reported: HashMap<i32, i64> = HashMap::new();

    for step in 0..(20 * 60 * 20) {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
        }
        game.objects.pending_sounds.clear();
        game.sound_queue.clear();
        game.objects.weapons.freed.clear();
        let _ = matrixgame_rs::matrix_game::interface::sound::drain();

        if step % 20 != 0 {
            continue;
        }
        let now = game.elapsed_ms;
        // Track movement.
        let ids: Vec<_> = game.objects.iter_live().collect();
        let mut side_state: HashMap<i32, (usize, usize)> = HashMap::new(); // (robots, moved_recently)
        for id in ids {
            let Some(r) = robot_ref(&game.objects, id) else {
                continue;
            };
            if !r.is_live() {
                continue;
            }
            let key = format!("{id:?}");
            let pos = glam::Vec2::new(r.pos_x, r.pos_y);
            let e = last_move.entry(key).or_insert((pos, now));
            if (pos - e.0).length() > 2.0 {
                *e = (pos, now);
            }
            let moved_recently = now - e.1 < 60_000;
            let s = side_state.entry(r.side).or_insert((0, 0));
            s.0 += 1;
            if moved_recently {
                s.1 += 1;
            }
        }
        if step % 2400 == 0 {
            let mut sides: Vec<_> = side_state.iter().collect();
            sides.sort();
            println!("t={:>4}s sides: {:?}", now / 1000, sides);
        }
        for (&side, &(robots, moving)) in &side_state {
            if robots >= 2 && moving == 0 && now - stall_reported.get(&side).copied().unwrap_or(0) > 120_000 {
                stall_reported.insert(side, now);
                println!("\n=== STALL side {side} at t={}s: {robots} live robots, none moved for 60s ===", now / 1000);
                for cid in game.objects.iter_live() {
                    if let Some(c) = matrixgame_rs::matrix_game::logic::cannon_ref(&game.objects, cid) {
                        println!("  [cannon {cid:?} side={} pos=({:.0},{:.0}) state={:?}]", c.side, c.pos.x, c.pos.y, c.state);
                    }
                }
                if let Some(su) = game.side_by_id(side) {
                    for (ti, t) in su.teams.iter().enumerate() {
                        if t.robot_cnt == 0 {
                            continue;
                        }
                        println!(
                            "  team {ti}: robots={} groups={} action={:?}/r{} path={:?} target_region={} war={} stay={} wait_union={} l_ok={} region_mass={} action_time={}",
                            t.robot_cnt, t.group_cnt, t.action.ty, t.action.region, t.action.region_path,
                            t.target_region, t.war, t.stay, t.wait_union, t.l_ok,
                            t.region_mass, t.action_time,
                        );
                    }
                    for (gi, lg) in su.logic_groups.iter().enumerate() {
                        if lg.robots_cnt <= 0 {
                            continue;
                        }
                        println!(
                            "  group {gi}: team={} robots={} strength={:.0} war={} action={:?}/r{}",
                            lg.team, lg.robots_cnt, lg.strength, lg.war, lg.action.ty, lg.action.region,
                        );
                    }
                }
                let ids: Vec<_> = game.objects.iter_live().collect();
                for id in ids {
                    let Some(r) = robot_ref(&game.objects, id) else {
                        continue;
                    };
                    if r.side != side || !r.is_live() {
                        continue;
                    }
                    let orders: Vec<String> = r
                        .orders
                        .iter()
                        .map(|o| format!("{:?}/{:?}", o.ty, o.phase))
                        .collect();
                    let env: Vec<String> = r
                        .env
                        .enemies
                        .iter()
                        .map(|e| {
                            let alive = game
                                .objects
                                .get(e.enemy)
                                .map(|o| format!("{:?} side{} live={}", o.core().obj_type, o.side(), o.is_live()))
                                .unwrap_or_else(|| "STALE".into());
                            format!("{:?}({alive},ds={})", e.enemy, e.del_slowly)
                        })
                        .collect();
                    if r.orders.has(matrixgame_rs::matrix_game::robot::OrderType::MoveTo) {
                        println!(
                            "    des=({},{}) zone_cur={} zone_des={} zone_path={:?} next={} move_path_len={} cur={}",
                            r.des_x, r.des_y, r.zone_cur, r.zone_des,
                            r.zone_path, r.zone_path_next,
                            r.move_path.pts.len(), r.move_path.cur,
                        );
                    }
                    if !env.is_empty() {
                        println!("    env: {env:?}");
                        println!(
                            "    target={:?} target_attack={:?} max_fd={:.0} place_not_found={} no_break={}",
                            r.env.target, r.env.target_attack, r.max_fire_dist,
                            r.env.place_not_found, r.env.order_no_break,
                        );
                    }
                    println!(
                        "  robot {:?} pos=({:.0},{:.0}) state={:?} orders={orders:?} team={} logic_group={} env.place={} place_add={:?}",
                        id,
                        r.pos_x,
                        r.pos_y,
                        r.state,
                        r.team,
                        r.group_logic,
                        r.env.place,
                        r.env.place_add,
                    );
                }
            }
        }
    }
    println!("done at t={}s", game.elapsed_ms / 1000);

    // Micro-trace: find a robot with a stuck MoveTo order and follow it
    // closely for 200 takts.
    let now = game.elapsed_ms;
    let stuck: Vec<_> = game
        .objects
        .iter_live()
        .filter(|&id| {
            robot_ref(&game.objects, id).is_some_and(|r| {
                r.is_live()
                    && r.orders.has(matrixgame_rs::matrix_game::robot::OrderType::MoveTo)
                    && last_move
                        .get(&format!("{id:?}"))
                        .is_some_and(|&(_, t)| now - t > 60_000)
            })
        })
        .collect();
    let Some(&rid) = stuck.first() else {
        println!("no stuck MoveTo robots to trace");
        return;
    };
    println!("tracing {rid:?}");
    if let Some(r) = robot_ref(&game.objects, rid) {
        println!(
            "  des=({},{}) map=({},{}) zone_path={:?} next={} waypoints={:?}",
            r.des_x, r.des_y, r.map_x, r.map_y, r.zone_path, r.zone_path_next, r.move_path.pts,
        );
        // Passability grid around the robot (move cells, chassis mask).
        let kind = r.chassis.kind_index();
        let (cx, cy) = (r.map_x, r.map_y);
        println!("  passability around ({cx},{cy}) for chassis {kind} ('.'=free '#'=blocked 'R'=robot):");
        let robots: Vec<(i32, i32)> = game
            .objects
            .iter_live()
            .filter_map(|id| robot_ref(&game.objects, id).map(|r| (r.map_x, r.map_y)))
            .collect();
        for dy in -10..=10 {
            let mut row = String::new();
            for dx in -14..=14 {
                let (x, y) = (cx + dx, cy + dy);
                let free = matrixgame_rs::matrix_game::logic::is_absence_wall(&map, kind, 1, x, y);
                let has_robot = robots.iter().any(|&(rx, ry)| (rx - x).abs() < 4 && (ry - y).abs() < 4 && (rx, ry) != (cx, cy));
                row.push(if (x, y) == (cx, cy) {
                    '@'
                } else if has_robot {
                    'R'
                } else if free {
                    '.'
                } else {
                    '#'
                });
            }
            println!("    {row}");
        }
    }
    for step in 0..200 {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
        }
        game.objects.pending_sounds.clear();
        game.sound_queue.clear();
        game.objects.weapons.freed.clear();
        let _ = matrixgame_rs::matrix_game::interface::sound::drain();
        if step % 10 == 0 {
            if let Some(r) = robot_ref(&game.objects, rid) {
                println!(
                    "  step {step}: pos=({:.1},{:.1}) vel=({:.3},{:.3}) speed={:.3} cols={} w={} w2={} colspd={:.2} path={}/{} zone_next={} orders={:?}",
                    r.pos_x, r.pos_y, r.velocity.x, r.velocity.y, r.speed,
                    r.cols, r.cols_weight, r.cols_weight2, r.col_speed,
                    r.move_path.cur, r.move_path.pts.len(), r.zone_path_next,
                    r.orders.iter().map(|o| format!("{:?}/{:?}", o.ty, o.phase)).collect::<Vec<_>>(),
                );
            }
        }
    }
}
