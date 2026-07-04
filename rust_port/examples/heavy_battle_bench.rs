//! Heavy-battle CPU bench: two armed AI packs fight at close range on
//! the real map. Prints takt cost + effect histogram over time.

use matrixgame_rs::matrix_game::config::RobotUnitKind;
use matrixgame_rs::matrix_game::object_robot::RobotUnitType;
use matrixgame_rs::matrix_game::interface::constructor::{RobotConfig, Unit};
use matrixgame_rs::matrix_game::logic::{robot_mut, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::robot::{ChassisKind, Robot, RobotState};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use std::collections::HashMap;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Error)
        .init();
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let matrix_data =
        Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();

    let mut game = MapLogic::with_seed(7);
    game.load_config(&matrix_data);

    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let mut spot = None;
    'outer: for my in (30..map.size_move_y as i32 - 60).step_by(4) {
        for mx in (30..map.size_move_x as i32 - 60).step_by(4) {
            let mut ok = true;
            for dy in (0..40).step_by(8) {
                for dx in (0..40).step_by(8) {
                    if !matrixgame_rs::matrix_game::logic::is_absence_wall(&map, 2, 4, mx + dx, my + dy) {
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
    let (mx, my) = spot.expect("no big flat spot");

    // Missile + mortar + machinegun loadout — trail/smoke heavy.
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

    let mut spawn = |game: &mut MapLogic, side: i32, x: f32, y: f32, team: i32| {
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
            rm.set_team(team);
        }
        game.objects.add_lt(id);
    };

    for i in 0..13 {
        let (dx, dy) = (i % 5, i / 5);
        spawn(&mut game, 2, (mx + dx * 6) as f32 * gs, (my + dy * 6) as f32 * gs, 0);
        spawn(&mut game, 3, (mx + dx * 6) as f32 * gs, (my + 24 + dy * 6) as f32 * gs, 0);
    }
    game.ensure_sides_from_objects();

    let mut sum = 0.0f64;
    let mut worst = 0.0f64;
    let mut frames = 0u64;
    for step in 0..(3 * 60 * 20) {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        let t0 = std::time::Instant::now();
        game.takt(50);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        sum += ms;
        worst = worst.max(ms);
        frames += 1;
        if step % 400 == 399 {
            let mut hist: HashMap<&'static str, usize> = HashMap::new();
            for e in &game.effects {
                *hist.entry(e.kind_name()).or_default() += 1;
            }
            let robots = game
                .objects
                .iter_live()
                .filter(|&id| {
                    matrixgame_rs::matrix_game::logic::robot_ref(&game.objects, id).is_some()
                })
                .count();
            let mut h: Vec<_> = hist.into_iter().collect();
            h.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
            println!(
                "t={:>6}ms takt avg={:.2}ms worst={:.2}ms robots={} effects={} top={:?}",
                game.elapsed_ms,
                sum / frames as f64,
                worst,
                robots,
                game.effects.len(),
                &h[..h.len().min(6)]
            );
            {
                use matrixgame_rs::matrix_game::effects::EFFECT_TAKT_NS;
                use std::sync::atomic::Ordering;
                let names = ["MovingObject","FirePlasma","Flame","BigBoom","Konus","Smoke","Fire","BbLine","Keel","Score","Lightening","Explosion","FireAnim","Dust"];
                let mut rows: Vec<(f64,&str)> = names.iter().enumerate()
                    .map(|(i,&n)| (EFFECT_TAKT_NS[i].swap(0, Ordering::Relaxed) as f64/1.0e6, n))
                    .filter(|r| r.0 > 0.01).collect();
                rows.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
                println!("   effect ms this window: {:?}", &rows[..rows.len().min(6)]);
            }
            sum = 0.0;
            worst = 0.0;
            frames = 0;
        }
    }
}
