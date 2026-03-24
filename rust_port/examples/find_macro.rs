use matrixgame_rs::assets::pkg_reader::PkgArchive;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();
    let files = pkg.list_files();
    let mut macros: Vec<_> = files.iter().filter(|f| f.to_lowercase().contains("macrotexture")).collect();
    macros.sort();
    for f in &macros { println!("  {}", f); }
}
