//! Pack CMAP + textures into a single asset bundle for WASM.
//! Resolves missing extensions by trying .png and .dds in the pkg.
use matrixgame_rs::assets::bundle::AssetBundle;
use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();

    let map_name = "MATRIX/MAP/ATOLL.CMAP";
    let cmap_data = pkg.read_file(map_name).unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();

    let mut bundle = AssetBundle::new();
    bundle.add("map.cmap", cmap_data);

    let strings = stor.get_buf("strings", "String").unwrap();
    let mut tex_count = 0;
    for i in 0..strings.arrays_count() {
        let raw_path = strings.get_as_wstr(i);
        let path = raw_path.split('?').next().unwrap_or("").replace('\\', "/");
        if path.is_empty() || path.contains('*') {
            continue;
        }

        // Try exact path, then with .png, .dds extensions
        let pkg_key = path.to_uppercase();
        let candidates = vec![
            pkg_key.clone(),
            format!("{}.PNG", pkg_key),
            format!("{}.DDS", pkg_key),
        ];

        let mut found = false;
        for candidate in &candidates {
            if let Ok(data) = pkg.read_file(candidate) {
                // Store with the original path (no extension if it didn't have one)
                // so the renderer can look it up by the same key
                bundle.add(&path, data);
                tex_count += 1;
                found = true;
                println!("  {} -> {} ({} bytes)", path, candidate, bundle.to_bytes().len());
                break;
            }
        }
        if !found && path.to_lowercase().contains("ter") {
            eprintln!("  MISS: {}", path);
        }
    }

    // Add macrotexture
    // Map property: MacroTexture = Matrix\Macrotexture\05?SIM80 → filename is "05"
    if let Some(props) = stor.get_buf("properties", "Name") {
        if let Some(vals) = stor.get_buf("properties", "Value") {
            if let Some(idx) = props.find_as_wstr("MacroTexture") {
                let macro_path = vals.get_as_wstr(idx);
                let file_part = macro_path.split('\\').last().unwrap_or("").split('?').next().unwrap_or("");
                let pkg_key = format!("MATRIX/MACROTEXTURE/{}.PNG", file_part.to_uppercase());
                if let Ok(data) = pkg.read_file(&pkg_key) {
                    // Store under both the real path and the fallback alias
                    let real_path = macro_path.split('?').next().unwrap_or("").replace('\\', "/");
                    bundle.add(&real_path, data.clone());
                    bundle.add("macrotexture", data.clone());
                    println!("  macrotexture -> {} as '{}' + 'macrotexture' ({} bytes)", pkg_key, real_path, data.len());
                }
            }
        }
    }

    // Add water textures
    let water_files = [
        ("water_tex1", "MATRIX/TEXTURES/WATER/1.DDS"),
        ("water_tex2", "MATRIX/TEXTURES/WATER/MIRROR.DDS"),
    ];
    for (key, pkg_path) in &water_files {
        if let Ok(data) = pkg.read_file(pkg_path) {
            bundle.add(key, data.clone());
            tex_count += 1;
            println!("  {} -> {} ({} bytes)", key, pkg_path, data.len());
        }
    }

    let bytes = bundle.to_bytes();
    std::fs::create_dir_all("assets").ok();
    std::fs::write("assets/atoll.bundle", &bytes).unwrap();
    println!(
        "\nPacked {} textures + CMAP into assets/atoll.bundle ({:.1} MB)",
        tex_count,
        bytes.len() as f64 / 1024.0 / 1024.0
    );
}
