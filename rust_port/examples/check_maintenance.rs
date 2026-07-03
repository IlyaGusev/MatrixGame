//! Verify the reinforcement cooldown is seeded at load and enforced.
use matrixgame_rs::matrix_game::logic::MapLogic;
use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();
    let cmap_name = pkg.list_files().into_iter().find(|f| f.ends_with(".CMAP")).unwrap().to_string();
    let cmap = pkg.read_file(&cmap_name).unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let dat = std::fs::read("../Data/robots.dat").unwrap();
    let matrix_data = Storage::from_bytes(&dat).unwrap();

    let mut game = MapLogic::with_seed(1);
    game.load_config(&matrix_data);
    game.spawn_buildings(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    println!("map maintenance_prc = {}", map.maintenance_prc);
    println!("maintenance_time at load = {} ms", game.maintenance_time);
    assert!(game.maintenance_time > 0, "cooldown must be seeded at load");
    game.takt(10_000);
    println!("after 10s: {} ms", game.maintenance_time);
    assert!(game.maintenance_time < 500_000 && game.maintenance_time > 0);
    println!("OK: reinforcement cooldown seeded and ticking");
}
