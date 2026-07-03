//! Verify RS_EFFECTS spawner records load from the CMAPs.
use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
fn main() {
    let data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();
    let mut names: Vec<String> = pkg.list_files().into_iter().filter(|f| f.ends_with(".CMAP")).map(|s| s.to_string()).collect();
    names.sort();
    let mut total = 0usize;
    for n in &names {
        let cmap = pkg.read_file(n).unwrap();
        let map = GameMap::from_cmap_bytes(&cmap).unwrap();
        if !map.effect_spawners.is_empty() {
            println!("{n}: {} spawners, first: {:?}", map.effect_spawners.len(), map.effect_spawners.first());
            total += map.effect_spawners.len();
        }
    }
    println!("total spawners across {} maps: {}", names.len(), total);
}
