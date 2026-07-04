pub mod gfx;
pub mod matrix_game;
pub mod matrix_lib;
pub mod platform;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", not(feature = "wasm-bench")))]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("=== MatrixGame WASM v74 (anim+move debug) ===");
    matrix_game::form_game::run();
}

/// Headless battle benchmark for profiling the game logic under wasm
/// (run in node — see bench_node.mjs). Mirrors
/// examples/heavy_battle_bench.rs.
#[cfg(all(target_arch = "wasm32", feature = "wasm-bench"))]
#[wasm_bindgen]
pub fn bench_battle(pkg_bytes: &[u8], dat_bytes: &[u8]) -> String {
    console_error_panic_hook::set_once();
    use matrix_game::config::RobotUnitKind;
    use matrix_game::interface::constructor::{RobotConfig, Unit};
    use matrix_game::logic::{robot_mut, MapLogic};
    use matrix_game::map::{GameMap, MapScope};
    use matrix_game::object_robot::RobotUnitType;
    use matrix_game::robot::{ChassisKind, Robot, RobotState};
    use matrix_lib::base::pack::PkgArchive;
    use matrix_lib::base::storage::Storage;

    let now_ms = || js_sys::Date::now();
    let mut out = String::new();

    let pkg = PkgArchive::from_bytes(pkg_bytes.to_vec()).expect("pkg");
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").expect("cmap");
    let map = GameMap::from_cmap_bytes(&cmap).expect("map");
    let matrix_data = Storage::from_bytes(dat_bytes).expect("dat");

    let mut game = MapLogic::with_seed(7);
    game.load_config(&matrix_data);

    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let mut spot = None;
    'outer: for my in (30..map.size_move_y as i32 - 60).step_by(4) {
        for mx in (30..map.size_move_x as i32 - 60).step_by(4) {
            let mut ok = true;
            for dy in (0..40).step_by(8) {
                for dx in (0..40).step_by(8) {
                    if !matrix_game::logic::is_absence_wall(&map, 2, 4, mx + dx, my + dy) {
                        ok = false;
                    }
                }
            }
            if ok {
                spot = Some((mx, my));
                break 'outer;
            }
        }
    }
    let (mx, my) = spot.expect("no flat spot");

    let mut cfg = RobotConfig::new();
    cfg.chassis.kind = RobotUnitKind(3);
    cfg.hull.unit.kind = RobotUnitKind(6);
    cfg.head.kind = RobotUnitKind(1);
    for (i, k) in [3, 3, 1, 5].iter().enumerate() {
        cfg.weapon[i] = Unit {
            ty: RobotUnitType::Weapon,
            kind: RobotUnitKind(*k),
            price: Default::default(),
        };
    }

    let mut spawn = |game: &mut MapLogic, side: i32, x: f32, y: f32| {
        let z = map.get_z(x, y);
        let mut r = Robot::new(glam::Vec3::new(x, y, z), side, ChassisKind::Track);
        r.state = RobotState::Idle;
        r.config = cfg;
        r.hit_point = 8000.0;
        r.hit_point_max = 8000.0;
        r.map_x = (x / gs) as i32;
        r.map_y = (y / gs) as i32;
        let id = game.objects.spawn(Box::new(r));
        if let Some(rm) = robot_mut(&mut game.objects, id) {
            rm.self_id = Some(id);
            rm.set_team(0);
        }
        game.objects.add_lt(id);
    };
    for i in 0..13 {
        let (dx, dy) = (i % 5, i / 5);
        spawn(&mut game, 2, (mx + dx * 6) as f32 * gs, (my + dy * 6) as f32 * gs);
        spawn(&mut game, 3, (mx + dx * 6) as f32 * gs, (my + 24 + dy * 6) as f32 * gs);
    }
    // Full real-game scenario on top: authored bases/turrets/robots +
    // playing AI sides, so the bench covers pathfinding, production,
    // turret seek and region logic like the browser session.
    game.spawn_buildings(&map);
    game.spawn_robots(&map);
    // Decorative map objects share the object arena — without them the
    // bench flattered every `iter_live()` walk in the hot loops.
    let cmap_stor = Storage::from_bytes(&cmap).expect("cmap storage");
    let (_ids, _stats) = game.spawn_map_objects(&map, &cmap_stor);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.accrue_resources(100_000);

    let mut sum = 0.0f64;
    let mut worst = 0.0f64;
    let mut frames = 0u64;
    for step in 0..(4 * 60 * 20) {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        let t0 = now_ms();
        game.takt(50);
        let ms = now_ms() - t0;
        sum += ms;
        worst = worst.max(ms);
        frames += 1;
        if step % 400 == 399 {
            use std::sync::atomic::Ordering;
            let ph: Vec<f64> = matrix_game::logic::TAKT_PHASE_US
                .iter()
                .map(|a| a.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames as f64)
                .collect();
            let robots = game
                .objects
                .iter_live()
                .filter(|&id| matrix_game::logic::robot_ref(&game.objects, id).is_some())
                .count();
            out.push_str(&format!(
                "t={:>6}ms takt avg={:.2}ms worst={:.2}ms robots={} effects={} | per-frame ms: gather={:.2} proceed={:.2} sides={:.2} fx={:.2} rest={:.2}\n",
                game.elapsed_ms,
                sum / frames as f64,
                worst,
                robots,
                game.effects.len(),
                ph[0], ph[1], ph[2], ph[3], ph[4],
            ));
            sum = 0.0;
            worst = 0.0;
            frames = 0;
        }
    }
    out
}
