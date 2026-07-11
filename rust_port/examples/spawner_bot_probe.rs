//! Spawner-bot dispatch probe: an AI-side bot from a map robot-spawner
//! must get its own Attack logic group (the enemy half of
//! MatrixObject.cpp:1452-1455) and go to war with the player.

use matrixgame_rs::matrix_game::logic::{robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::SpawnerBotRequest;
use matrixgame_rs::matrix_game::side::LogicActionType;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/TRAINING.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let matrix_data = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();

    let mut game = MapLogic::with_seed(7);
    game.load_config(&matrix_data);
    game.spawn_buildings(&map);
    game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);

    let cat = matrixgame_rs::matrix_game::config::robot_spawn_config();
    println!("catalogue bots: {}", cat.bots.len());
    let Some(bot) = cat.choose(1, 0) else {
        println!("RESULT: NO CATALOGUE (robots.dat has no spawner bots)");
        return;
    };
    println!("catalogue bot side={}", bot.side);

    game.objects.pending_spawner_bots.push(SpawnerBotRequest {
        pos: glam::Vec3::new(700.0, 1500.0, 0.0),
        number: 1,
        pick: 0,
        sens_radius: 300.0,
    });
    let before: Vec<_> = game.objects.iter_units().collect();
    {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.takt(50);
    }
    let new_ids: Vec<_> = game
        .objects
        .iter_units()
        .filter(|id| !before.contains(id))
        .collect();
    for id in new_ids {
        let Some(r) = robot_ref(&game.objects, id) else {
            continue;
        };
        println!(
            "spawned bot: side={} team={} group_logic={}",
            r.side, r.team, r.group_logic
        );
        if r.side != 1 && r.group_logic >= 0 {
            let side = game.side_by_id(r.side).unwrap();
            let lg = &side.logic_groups[r.group_logic as usize];
            println!(
                "  group action={:?} region={} team={}",
                lg.action.ty, lg.action.region, lg.team
            );
            if lg.action.ty == LogicActionType::Attack && r.team == -1 {
                println!("RESULT: OK — Attack group assigned");
                return;
            }
        }
    }
    println!("RESULT: FAIL — no Attack group");
}
