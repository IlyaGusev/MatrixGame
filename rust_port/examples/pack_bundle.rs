//! Pack CMAP + textures into a single asset bundle for WASM.
//! Resolves missing extensions by trying .png and .dds in the pkg.
use matrixgame_rs::assets::bundle::AssetBundle;
use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;
use matrixgame_rs::game::map::GameMap;
use matrixgame_rs::game::vo_loader;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();

    let map_name = "MATRIX/MAP/ATOLL.CMAP";
    let cmap_data = pkg.read_file(map_name).unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();

    let mut bundle = AssetBundle::new();
    bundle.add("map.cmap", cmap_data.clone());

    let strings = stor.get_buf("strings", "String").unwrap();
    let mut tex_count = 0;
    let mut extra_paths: Vec<String> = Vec::new();
    for i in 0..strings.arrays_count() {
        let raw_path = strings.get_as_wstr(i);
        let mut iter = raw_path.split('?');
        let base = iter.next().unwrap_or("");
        let path = base.replace('\\', "/");
        if path.is_empty() || path.contains('*') {
            continue;
        }

        // Surface ids also carry `?gloss=<name>` — pack the gloss sibling too.
        for param in iter {
            if let Some((k, v)) = param.split_once('=') {
                if k.eq_ignore_ascii_case("gloss") && !v.is_empty() {
                    let slash = path.rfind('/');
                    let dir = slash.map(|i| &path[..i]).unwrap_or("");
                    let gloss_path = if dir.is_empty() {
                        v.to_string()
                    } else {
                        format!("{}/{}", dir, v)
                    };
                    extra_paths.push(gloss_path);
                }
            }
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
                println!(
                    "  {} -> {} ({} bytes)",
                    path,
                    candidate,
                    bundle.to_bytes().len()
                );
                break;
            }
        }
        if !found && path.to_lowercase().contains("ter") {
            eprintln!("  MISS: {}", path);
        }
    }

    // Also pack the global reflection texture used by the gloss surface pass.
    extra_paths.push("Matrix/Textures/reflection".to_string());
    // Sky panorama textures referenced by the hardcoded sky config table in
    // `renderer/sky.rs::resolve_sky_texture`. Pack all of them so every sky
    // name resolves regardless of which map is loaded next.
    for name in [
        "blue",
        "blue_moon",
        "stars",
        "mars",
        "alien_blue",
        "dark_green",
        "black",
    ] {
        extra_paths.push(format!("Matrix/Textures/Sky/{}", name));
    }

    for extra in &extra_paths {
        let pkg_key = extra.to_uppercase();
        let candidates = [
            pkg_key.clone(),
            format!("{}.PNG", pkg_key),
            format!("{}.DDS", pkg_key),
        ];
        for candidate in &candidates {
            if let Ok(data) = pkg.read_file(candidate) {
                bundle.add(extra, data);
                tex_count += 1;
                println!("  extra {} -> {}", extra, candidate);
                break;
            }
        }
    }

    // Add macrotexture
    // Map property: MacroTexture = Matrix\Macrotexture\05?SIM80 → filename is "05"
    if let Some(props) = stor.get_buf("properties", "Name") {
        if let Some(vals) = stor.get_buf("properties", "Value") {
            if let Some(idx) = props.find_as_wstr("MacroTexture") {
                let macro_path = vals.get_as_wstr(idx);
                let file_part = macro_path
                    .split('\\')
                    .last()
                    .unwrap_or("")
                    .split('?')
                    .next()
                    .unwrap_or("");
                let pkg_key = format!("MATRIX/MACROTEXTURE/{}.PNG", file_part.to_uppercase());
                if let Ok(data) = pkg.read_file(&pkg_key) {
                    // Store under both the real path and the fallback alias
                    let real_path = macro_path
                        .split('?')
                        .next()
                        .unwrap_or("")
                        .replace('\\', "/");
                    bundle.add(&real_path, data.clone());
                    bundle.add("macrotexture", data.clone());
                    println!(
                        "  macrotexture -> {} as '{}' + 'macrotexture' ({} bytes)",
                        pkg_key,
                        real_path,
                        data.len()
                    );
                }
            }
        }
    }

    // Add water textures under the paths the renderer actually asks for
    // (matching resolve_water_preset in renderer/water.rs).
    let water_files = [
        ("Matrix/Textures/Water/1", "MATRIX/TEXTURES/WATER/1.DDS"),
        (
            "Matrix/Textures/Water/MIRROR",
            "MATRIX/TEXTURES/WATER/MIRROR.DDS",
        ),
        (
            "Matrix/Textures/Water/1BLACK",
            "MATRIX/TEXTURES/WATER/1BLACK.DDS",
        ),
        (
            "Matrix/Textures/Water/MIRRORBLACK",
            "MATRIX/TEXTURES/WATER/MIRRORBLACK.DDS",
        ),
        (
            "Matrix/Textures/Water/1PURPLE",
            "MATRIX/TEXTURES/WATER/1PURPLE.DDS",
        ),
        (
            "Matrix/Textures/Water/MIRRORPURPLE",
            "MATRIX/TEXTURES/WATER/MIRRORPURPLE.DDS",
        ),
    ];
    for (key, pkg_path) in &water_files {
        if let Ok(data) = pkg.read_file(pkg_path) {
            bundle.add(key, data.clone());
            tex_count += 1;
            println!("  {} -> {} ({} bytes)", key, pkg_path, data.len());
        }
    }

    // Pack object .vo meshes + their textures, one per unique object type_id.
    let map = GameMap::from_cmap_bytes(&cmap_data).unwrap();
    let mut obj_types: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for obj in &map.objects {
        obj_types.insert(obj.type_id);
    }
    let mut vo_count = 0;
    let mut obj_tex_count = 0;
    let mut vo_tex_seen = std::collections::HashSet::<String>::new();
    for type_id in &obj_types {
        if (*type_id as usize) >= strings.arrays_count() {
            continue;
        }
        let id_str = strings.get_as_wstr(*type_id as usize);
        let Some(paths) = vo_loader::resolve_paths(&id_str) else {
            continue;
        };
        let vo_key = paths.vo_path.to_uppercase();
        if let Ok(data) = pkg.read_file(&vo_key) {
            bundle.add(&paths.vo_path, data);
            vo_count += 1;
        } else {
            eprintln!("  MISS vo: {}", paths.vo_path);
            continue;
        }
        for t in [
            paths.material.diffuse.as_ref(),
            paths.material.gloss.as_ref(),
            paths.material.back.as_ref(),
            paths.material.mask.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if !vo_tex_seen.insert(t.clone()) {
                continue;
            }
            let k = t.to_uppercase();
            for cand in [k.clone(), format!("{}.DDS", k), format!("{}.PNG", k)] {
                if let Ok(data) = pkg.read_file(&cand) {
                    bundle.add(t, data);
                    obj_tex_count += 1;
                    break;
                }
            }
        }
    }
    println!(
        "  objects: {} vo files, {} textures packed",
        vo_count, obj_tex_count
    );

    let bytes = bundle.to_bytes();
    std::fs::create_dir_all("assets").ok();
    std::fs::write("assets/atoll.bundle", &bytes).unwrap();
    println!(
        "\nPacked {} textures + CMAP into assets/atoll.bundle ({:.1} MB)",
        tex_count,
        bytes.len() as f64 / 1024.0 / 1024.0
    );
}
