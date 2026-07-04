//! End-to-end enemy-AI smoke test on a real map: load robots.pkg +
//! robots.dat, spawn the map's buildings/robots, then run the full
//! logic loop for a few simulated minutes and check the AI sides
//! actually play (build robots, form teams, issue orders).

use matrixgame_rs::matrix_game::logic::{robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::MapStatic;
use matrixgame_rs::matrix_game::side::LogicActionType;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

#[test]
fn enemy_ai_plays_on_real_map() {
    let _ = env_logger::builder().is_test(true).try_init();

    let Ok(pkg_data) = std::fs::read("../Data/robots.pkg") else {
        eprintln!("skipping: ../Data/robots.pkg not present");
        return;
    };
    let Ok(dat_bytes) = std::fs::read("../Data/robots.dat") else {
        eprintln!("skipping: ../Data/robots.dat not present");
        return;
    };
    let pkg = PkgArchive::from_bytes(pkg_data).expect("parse pkg");
    let matrix_data = Storage::from_bytes(&dat_bytes).expect("parse robots.dat");

    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").expect("read CMAP");
    let map = GameMap::from_cmap_bytes(&cmap_data).expect("parse map");

    let mut game = MapLogic::new();
    game.load_config(&matrix_data);
    game.spawn_buildings(&map);
    game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.accrue_resources(100_000);

    let ai_ids: Vec<i32> = game.other_sides.iter().map(|s| s.id).collect();
    assert!(
        !ai_ids.is_empty(),
        "map must have at least one non-player side"
    );
    let count_robots = |game: &MapLogic, sid: i32| -> usize {
        game.objects
            .iter_live()
            .filter(|&id| {
                robot_ref(&game.objects, id)
                    .map(|r| r.is_live() && r.side == sid)
                    .unwrap_or(false)
            })
            .count()
    };
    let initial_ai_robots: usize = ai_ids.iter().map(|&sid| count_robots(&game, sid)).sum();
    println!(
        "AI sides: {:?}, initial AI robots: {}",
        ai_ids, initial_ai_robots
    );

    // ~4 simulated minutes at 50ms frames.
    for _ in 0..(4 * 60 * 20) {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.takt(50);
    }

    let final_ai_robots: usize = ai_ids.iter().map(|&sid| count_robots(&game, sid)).sum();
    let mut any_action = false;
    let mut any_group = false;
    for &sid in &ai_ids {
        let side = game.side_by_id(sid).unwrap();
        for tm in &side.teams {
            if tm.robot_cnt > 0 && tm.action.ty != LogicActionType::None {
                any_action = true;
            }
        }
        if side.logic_groups.iter().any(|lg| lg.robots_cnt > 0) {
            any_group = true;
        }
        println!(
            "side {}: robots={} strength={:.1} war_side={} team actions={:?}",
            sid,
            side.robots_cnt,
            side.strength,
            side.war_side,
            side.teams
                .iter()
                .map(|t| (t.robot_cnt, t.action.ty))
                .collect::<Vec<_>>()
        );
    }
    println!("AI robots after sim: {}", final_ai_robots);

    assert!(
        final_ai_robots > initial_ai_robots,
        "AI must build robots over 4 simulated minutes \
         (initial {initial_ai_robots}, final {final_ai_robots})"
    );
    assert!(any_group, "AI robots must be organised into logic groups");
    assert!(any_action, "AI teams must have picked actions");
}
