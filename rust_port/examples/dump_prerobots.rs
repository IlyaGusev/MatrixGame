use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;

fn main() {
    let data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();
    let mut names: Vec<String> = pkg
        .list_files()
        .into_iter()
        .filter(|f| f.to_lowercase().contains("train") && f.ends_with(".CMAP"))
        .map(|s| s.to_string())
        .collect();
    names.sort();
    for cmap_name in &names {
        println!("=== {cmap_name} ===");
        let cmap = pkg.read_file(cmap_name).unwrap();
        let map = GameMap::from_cmap_bytes(&cmap).unwrap();
        for r in &map.robots {
            println!(
                "  side={} group={} pos=({:.0},{:.0})",
                r.side, r.group, r.x, r.y
            );
        }
    }
}
