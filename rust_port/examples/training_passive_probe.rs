//! Training-map passivity probe: pre-placed enemy robots (CMAP group 0
//! → team -1 → GroupNoTeamRobot Defence groups) must not attack the
//! player unprovoked. Run with FORCE_TEAM0=1 to mimic the old bug and
//! confirm the probe detects aggression.

use matrixgame_rs::matrix_game::logic::{building_ref, robot_mut, robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let force_team0 = std::env::var("FORCE_TEAM0").is_ok();

    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/TRAINING.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let matrix_data = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();

    let mut game = MapLogic::with_seed(1234);
    game.load_config(&matrix_data);
    game.spawn_buildings(&map);
    let robot_ids = game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.accrue_resources(100_000);

    if force_team0 {
        for &id in &robot_ids {
            if let Some(r) = robot_mut(&mut game.objects, id) {
                if r.side != 1 {
                    r.set_team(0);
                    r.group_logic = -1;
                }
            }
        }
    }

    let start: Vec<_> = robot_ids
        .iter()
        .filter_map(|&id| {
            let r = robot_ref(&game.objects, id)?;
            (r.side != 1).then_some((id, r.pos_x, r.pos_y))
        })
        .collect();
    println!("enemy robots: {} (force_team0={force_team0})", start.len());

    let player_buildings = |game: &MapLogic| -> (usize, f32) {
        let mut cnt = 0;
        let mut hp = 0.0;
        for id in game.objects.iter_units() {
            if let Some(b) = building_ref(&game.objects, id) {
                if b.side == 1 {
                    cnt += 1;
                    hp += b.hit_point;
                }
            }
        }
        (cnt, hp)
    };
    let (c0, hp0) = player_buildings(&game);
    println!("player buildings: {c0}, total hp {hp0:.0}");

    // 10 sim-minutes at 50ms steps, report each minute.
    for minute in 1..=10 {
        for _ in 0..(60 * 20) {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
        }
        let (cnt, hp) = player_buildings(&game);
        let alive = start
            .iter()
            .filter(|&&(id, _, _)| robot_ref(&game.objects, id).is_some())
            .count();
        println!(
            "min {minute:2}: player buildings={cnt} hp={hp:.0} | enemy alive={alive} | game_over={:?} player_status={:?}",
            game.game_over, game.player_side.status
        );
    }

    let mut max_disp = 0.0f32;
    for &(id, x0, y0) in &start {
        if let Some(r) = robot_ref(&game.objects, id) {
            let d = ((r.pos_x - x0).powi(2) + (r.pos_y - y0).powi(2)).sqrt();
            max_disp = max_disp.max(d);
            if d > 50.0 {
                println!(
                    "  MOVED: robot team={} group_logic={} ({:.0},{:.0}) -> ({:.0},{:.0}) d={:.0} orders={:?}",
                    r.team,
                    r.group_logic,
                    x0,
                    y0,
                    r.pos_x,
                    r.pos_y,
                    d,
                    r.orders.iter().map(|o| o.ty).collect::<Vec<_>>()
                );
            }
        }
    }
    println!("max displacement={max_disp:.1}");
    let (cnt, hp) = player_buildings(&game);
    if game.game_over == Some(false) || cnt < c0 || hp < hp0 {
        println!("RESULT: PLAYER ATTACKED (buildings {c0}->{cnt}, hp {hp0:.0}->{hp:.0})");
    } else {
        println!("RESULT: PLAYER UNTOUCHED");
    }
}
