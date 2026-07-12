//! Trace the AI capture-region deadlock found by game_sim
//! (--player ai --map ATOLL --seed 1): replay the run and, from just
//! before the stall onset, dump every capture order's live state each
//! game second — holder pos/state, target factory, distance vs the
//! BASE_DIST gate, companion MoveTo presence, and place assignment.
//!
//!   cargo run --example capture_stall_probe -- [map] [seed] [from_s] [to_s]
//!
//! Defaults: ATOLL 1 1050 1130 (the original finding's stall window).

use matrixgame_rs::matrix_game::logic::{building_ref, robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::robot::{OrderPhase, OrderType};
use matrixgame_rs::matrix_game::map_static::MapStatic;
use matrixgame_rs::matrix_game::side::Side;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Error)
        .init();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let map_name = a.first().cloned().unwrap_or_else(|| "ATOLL".into());
    let seed: i32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let from_s: i64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1050);
    let to_s: i64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1130);
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg
        .read_file(&format!("MATRIX/MAP/{}.CMAP", map_name.to_uppercase()))
        .unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let dat = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();
    for idx in 1..=6usize {
        if let Ok(data) = pkg.read_file(&format!("MATRIX/ROBOT/ARMOR{idx}.VO")) {
            let vo = matrixgame_rs::matrix_lib::three_g::vector_object::parse_vo(&data).unwrap();
            let m = matrixgame_rs::matrix_game::map::weapon_matrix_from_vo(&vo);
            matrixgame_rs::matrix_game::map::set_weapon_matrix_for(
                matrixgame_rs::matrix_game::config::RobotUnitKind(idx as i32),
                m,
            );
        }
    }

    let mut game = MapLogic::with_seed(seed);
    game.load_config(&dat);
    game.player_side = Side::new(100);
    game.spawn_buildings(&map);
    game.spawn_ruins(&map);
    game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.init_effect_spawners(&map);
    game.accrue_resources(100_000);
    let stor = Storage::from_bytes(&cmap).unwrap();
    game.spawn_map_objects(&map, &stor);

    let trace_from_ms = from_s * 1000;
    let end_ms = to_s * 1000;
    while game.elapsed_ms < end_ms {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
            game.graphic_takt(50);
        }
        game.objects.pending_sounds.clear();
        game.sound_queue.clear();
        game.objects.weapons.freed.clear();
        game.objects.pending_spots.clear();
        game.objects.pending_point_lights.clear();
        game.objects.pending_light_follow.clear();
        game.objects.pending_light_kill.clear();
        let _ = matrixgame_rs::matrix_game::interface::sound::drain();

        if game.elapsed_ms < trace_from_ms || game.elapsed_ms % 5000 >= 50 {
            continue;
        }
        let now = game.elapsed_ms;
        for id in game.objects.iter_live() {
            let Some(r) = robot_ref(&game.objects, id) else { continue };
            if !r.is_live() {
                continue;
            }
            let Some(cap) = r.orders.iter().find(|o| o.ty == OrderType::CaptureFactory) else {
                continue;
            };
            let tgt = cap.target;
            let (tpos, tside, tstate, tcap_by) = tgt
                .and_then(|t| building_ref(&game.objects, t))
                .map(|b| {
                    (
                        (b.pos.x, b.pos.y),
                        b.side,
                        format!("{:?} cc={} col={:08x}", b.state, b.true_color.colored_cnt, b.true_color.color),
                        b.capturer,
                    )
                })
                .unwrap_or(((-1.0, -1.0), -99, "GONE".into(), None));
            // Who is nearest to the building within CAPTURE_RADIUS?
            let mut near_txt = String::from("-");
            if tpos.0 >= 0.0 {
                let mut best = 50.0f32 * 50.0;
                for oid in game.objects.iter_live() {
                    if let Some(o) = robot_ref(&game.objects, oid) {
                        if !o.is_live() { continue; }
                        let gc = o.core().geo_center;
                        let d2 = (gc.x - tpos.0).powi(2) + (gc.y - tpos.1).powi(2);
                        if d2 < best {
                            best = d2;
                            near_txt = format!("{oid:?}@{:.0}", d2.sqrt());
                        }
                    }
                }
            }
            let dist = ((r.pos_x - tpos.0).powi(2) + (r.pos_y - tpos.1).powi(2)).sqrt();
            let mv: Vec<String> = r
                .orders
                .iter()
                .map(|o| format!("{:?}/{:?}({},{})", o.ty, o.phase, o.p1, o.p2))
                .collect();
            let mv = mv.join(" ");
            println!(
                "t={:>4}s side={} robot {id:?} pos=({:.0},{:.0}) st={:?} speed={:.2} cap_phase={:?} mv={mv} tgt={tgt:?} tside={tside} tstate={tstate} tcapby={tcap_by:?} dist={dist:.0} near={near_txt} des=({},{}) path={}/{} place={}",
                now / 1000, r.side, r.pos_x, r.pos_y, r.state, r.speed, cap.phase,
                r.des_x, r.des_y, r.move_path.cur, r.move_path.pts.len(), r.env.place,
            );
            if cap.phase == OrderPhase::CaptureMoving && !mv.contains("MoveTo") && dist > 60.0 {
                println!("    ^^ STUCK: CaptureMoving, no MoveTo, out of range");
            }
        }
        println!("---");
    }

    // Experiment: find a robot frozen in CaptureMoving with no MoveTo,
    // force a fresh MoveTo to its capture destination, and watch
    // whether pathfinding can serve it at all.
    let stuck: Vec<_> = game
        .objects
        .iter_live()
        .filter(|&id| {
            robot_ref(&game.objects, id).is_some_and(|r| {
                r.is_live()
                    && r.orders
                        .iter()
                        .any(|o| o.ty == OrderType::CaptureFactory && o.phase == OrderPhase::CaptureMoving)
                    && !r.orders.has(OrderType::MoveTo)
            })
        })
        .collect();
    println!("frozen capture robots: {stuck:?}");
    let Some(&rid) = stuck.first() else { return };
    let (des_x, des_y) = {
        let r = matrixgame_rs::matrix_game::logic::robot_mut(&mut game.objects, rid).unwrap();
        println!(
            "forcing MoveTo({},{}) on {rid:?} at ({:.0},{:.0})",
            r.des_x, r.des_y, r.pos_x, r.pos_y
        );
        let (dx, dy) = (r.des_x, r.des_y);
        r.move_to(dx, dy);
        (dx, dy)
    };
    let _ = (des_x, des_y);
    for step in 0..600 {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
            game.graphic_takt(50);
        }
        game.objects.pending_sounds.clear();
        game.sound_queue.clear();
        game.objects.weapons.freed.clear();
        let _ = matrixgame_rs::matrix_game::interface::sound::drain();
        if step % 40 == 0 {
            if let Some(r) = robot_ref(&game.objects, rid) {
                let orders: Vec<String> = r
                    .orders
                    .iter()
                    .map(|o| format!("{:?}/{:?}", o.ty, o.phase))
                    .collect();
                println!(
                    "  +{:>2}s pos=({:.0},{:.0}) speed={:.2} zone_cur={} zone_des={} zpath={} mpath={}/{} orders={orders:?}",
                    step / 20, r.pos_x, r.pos_y, r.speed, r.zone_cur, r.zone_des,
                    r.zone_path.len(), r.move_path.cur, r.move_path.pts.len(),
                );
            }
        }
    }
}
