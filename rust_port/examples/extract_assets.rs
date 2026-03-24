//! Extract all textures needed by the ATOLL map for browser serving.
use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();

    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();

    // Get string table (texture paths)
    let strings = stor.get_buf("strings", "String").unwrap();
    println!("Extracting textures for ATOLL map...");

    let mut extracted = 0;
    for i in 0..strings.arrays_count() {
        let path = strings.get_as_wstr(i);
        let path = path.split('?').next().unwrap_or("").replace('\\', "/");
        if path.is_empty() { continue; }

        let pkg_key = path.to_uppercase();
        match pkg.read_file(&pkg_key) {
            Ok(data) => {
                let out_path = format!("assets/{}", path);
                let dir = std::path::Path::new(&out_path).parent().unwrap();
                std::fs::create_dir_all(dir).ok();
                std::fs::write(&out_path, &data).unwrap();
                println!("  {} ({} bytes)", out_path, data.len());
                extracted += 1;
            }
            Err(e) => {
                eprintln!("  SKIP {}: {}", path, e);
            }
        }
    }
    println!("Extracted {} texture files", extracted);
}
