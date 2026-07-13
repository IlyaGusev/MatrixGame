//! Smoke-test every MATRIX/MAP/*.CMAP: full logic spawn + 30 sim-seconds.
//! Catches maps that panic on load or during early AI/logic ticks.
//!
//!   cargo run --example map_smoke

use matrixgame_rs::matrix_game::logic::MapLogic;
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let dat = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();
    let mut maps: Vec<String> = pkg
        .list_files()
        .iter()
        .filter(|f| {
            let up = f.to_uppercase();
            up.starts_with("MATRIX/MAP/") && up.ends_with(".CMAP")
        })
        .map(|f| f.to_string())
        .collect();
    maps.sort();

    let mut failed = 0;
    for path in &maps {
        let start = std::time::Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let cmap = pkg.read_file(path).unwrap();
            let map = GameMap::from_cmap_bytes(&cmap).unwrap();
            let stor = Storage::from_bytes(&cmap).unwrap();
            matrixgame_rs::matrix_game::map::set_map_name(path);
            let mut game = MapLogic::with_seed(1);
            game.load_config(&dat);
            game.spawn_buildings(&map);
            game.spawn_robots(&map);
            game.ensure_sides_from_objects();
            game.apply_side_resources(&map);
            game.init_effect_spawners(&map);
            game.accrue_resources(100_000);
            game.spawn_map_objects(&map, &stor);
            for _ in 0..(30 * 20) {
                let _scope = MapScope::enter(&map, game.elapsed_ms);
                game.takt(50);
            }
            game.objects.iter_units().count()
        }));
        match result {
            Ok(units) => println!(
                "OK   {path} ({units} units, {:.1}s)",
                start.elapsed().as_secs_f32()
            ),
            Err(_) => {
                failed += 1;
                println!("FAIL {path}");
            }
        }
    }
    println!("\n{} maps, {} failed", maps.len(), failed);
    std::process::exit(if failed > 0 { 1 } else { 0 });
}
