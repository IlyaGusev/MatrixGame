//! Extract a CMAP file from robots.pkg for browser testing.
use matrixgame_rs::assets::pkg_reader::PkgArchive;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").expect("read robots.pkg");
    let pkg = PkgArchive::from_bytes(pkg_data).expect("parse pkg");

    let map_name = "MATRIX/MAP/ATOLL.CMAP";
    let cmap_data = pkg.read_file(map_name).expect("read CMAP");

    std::fs::create_dir_all("assets").ok();
    std::fs::write("assets/atoll.cmap", &cmap_data).expect("write CMAP");
    println!(
        "Extracted {map_name}: {} bytes -> assets/atoll.cmap",
        cmap_data.len()
    );
}
