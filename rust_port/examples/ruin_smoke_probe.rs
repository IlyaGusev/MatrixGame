//! Ruin-smoke probe: killing a building must scatter 20-50 temporary
//! smoke spawners over the ruins (MatrixObjectBuilding.cpp:726-755)
//! that emit Smoke effects and expire within 5-20 s.

use matrixgame_rs::matrix_game::effects::GameEffect;
use matrixgame_rs::matrix_game::logic::{building_mut, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/TRAINING.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let matrix_data = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();

    let mut game = MapLogic::with_seed(3);
    game.load_config(&matrix_data);
    game.spawn_buildings(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);

    // Deplete the first building; the DIP state machine takes it from
    // there during takts.
    let ids: Vec<_> = game.objects.iter_units().collect();
    let bid = ids
        .into_iter()
        .find(|&id| building_mut(&mut game.objects, id).is_some())
        .expect("a building");
    if let Some(b) = building_mut(&mut game.objects, bid) {
        println!("killing building side={} kind={:?}", b.side, b.kind);
        b.hit_point = 1.0;
    }
    {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.objects.apply_damage(
            bid,
            matrixgame_rs::matrix_game::effects::weapon::WEAPON_GUN,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            2,
            None,
        );
    }

    let mut smoke = 0usize;
    for step in 0..(30 * 20) {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
        }
        let n = game
            .effects
            .iter()
            .filter(|e| matches!(e, GameEffect::Smoke(_)))
            .count();
        if n > smoke {
            smoke = n;
            println!("t={:>5}ms smoke effects alive: {n}", (step + 1) * 50);
        }
    }
    println!(
        "RESULT: {}",
        if smoke > 0 { "OK — ruin smoke seen" } else { "FAIL — no smoke" }
    );
}
