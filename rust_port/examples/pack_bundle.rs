//! Pack CMAP + textures into a single asset bundle for WASM.
//! Resolves missing extensions by trying .png and .dds in the pkg.
//!
//! Usage: `cargo run --example pack_bundle -- <map-name>`  (default: `atoll`).
//! Bare names resolve to `MATRIX/MAP/<NAME>.CMAP`. The output lands at
//! `assets/<name>.bundle`, matching the `?bundle=<url>` query the WASM loader
//! accepts.
use matrixgame_rs::gfx::bundle::AssetBundle;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::three_g::vector_object;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();

    let requested = std::env::args().nth(1).unwrap_or_else(|| "atoll".into());
    let short_name = requested
        .trim()
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
        .trim_end_matches(".cmap")
        .trim_end_matches(".CMAP")
        .to_lowercase();
    let map_name = if requested.contains('/') || requested.contains('\\') {
        requested.replace('\\', "/").to_uppercase()
    } else {
        format!("MATRIX/MAP/{}.CMAP", requested.to_uppercase())
    };
    println!("Packing map: {}", map_name);

    let cmap_data = pkg
        .read_file(&map_name)
        .unwrap_or_else(|e| panic!("could not read {}: {}", map_name, e));
    let stor = Storage::from_bytes(&cmap_data).unwrap();

    let mut bundle = AssetBundle::new();
    bundle.add("map.cmap", cmap_data.clone());

    // Global game data (Sky / Water / Side / ... block par tree).
    // Optional — the renderer falls back to defaults if missing.
    match std::fs::read("../Data/robots.dat") {
        Ok(data) => {
            println!("  robots.dat ({} bytes)", data.len());
            bundle.add("robots.dat", data);
        }
        Err(e) => eprintln!("  skip robots.dat: {}", e),
    }

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
    // Minimap icon atlas (CMinimap::Init reads sub-rects from the Minimap
    // block in robots.dat).
    extra_paths.push("Matrix/Textures/Minimap/radarIcons".to_string());
    // UI skin atlases — the "if" record in robots.dat references these by
    // name for every panel/button sub-rect. Minimap.rs currently uses
    // interface1 for the MiniM panel; pack 2/3 too so the rest of the HUD
    // can be ported without a second bundle bump.
    extra_paths.push("Matrix/Iface/interface1".to_string());
    extra_paths.push("Matrix/Iface/interface2".to_string());
    extra_paths.push("Matrix/Iface/interface3".to_string());
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
                    .next_back()
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

    // Water textures. Key names match the paths the `Water/*` BlockPar in
    // robots.dat returns (MatrixWater.cpp:215-216), so `resolve_water_preset`
    // in renderer/water.rs can look them up directly under the WASM bundle's
    // case-sensitive HashMap.
    let water_files = [
        ("Matrix/Textures/Water/1", "MATRIX/TEXTURES/WATER/1.DDS"),
        (
            "Matrix/Textures/Water/mirror",
            "MATRIX/TEXTURES/WATER/MIRROR.DDS",
        ),
        (
            "Matrix/Textures/Water/1black",
            "MATRIX/TEXTURES/WATER/1BLACK.DDS",
        ),
        (
            "Matrix/Textures/Water/mirrorblack",
            "MATRIX/TEXTURES/WATER/MIRRORBLACK.DDS",
        ),
        (
            "Matrix/Textures/Water/1purple",
            "MATRIX/TEXTURES/WATER/1PURPLE.DDS",
        ),
        (
            "Matrix/Textures/Water/mirrorpurple",
            "MATRIX/TEXTURES/WATER/MIRRORPURPLE.DDS",
        ),
        (
            // Shoreline foam quad — `Water/*/inshore` in robots.dat.
            "Matrix/Textures/Water/inshorewave",
            "MATRIX/TEXTURES/WATER/INSHOREWAVE.DDS",
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

    let try_pack_texture =
        |bundle: &mut AssetBundle,
         seen: &mut std::collections::HashSet<String>,
         t: &str|
         -> bool {
            if !seen.insert(t.to_string()) {
                return false;
            }
            let k = t.to_uppercase();
            for cand in [k.clone(), format!("{}.DDS", k), format!("{}.PNG", k)] {
                if let Ok(data) = pkg.read_file(&cand) {
                    bundle.add(t, data);
                    return true;
                }
            }
            false
        };

    for type_id in &obj_types {
        if (*type_id as usize) >= strings.arrays_count() {
            continue;
        }
        let id_str = strings.get_as_wstr(*type_id as usize);
        let Some(paths) = vector_object::resolve_paths(&id_str) else {
            continue;
        };
        let vo_key = paths.vo_path.to_uppercase();
        let vo_bytes = match pkg.read_file(&vo_key) {
            Ok(data) => {
                bundle.add(&paths.vo_path, data.clone());
                vo_count += 1;
                data
            }
            Err(_) => {
                eprintln!("  MISS vo: {}", paths.vo_path);
                continue;
            }
        };

        // Top-level (id-string) material textures.
        for t in [
            paths.material.diffuse.as_ref(),
            paths.material.gloss.as_ref(),
            paths.material.back.as_ref(),
            paths.material.mask.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if try_pack_texture(&mut bundle, &mut vo_tex_seen, t) {
                obj_tex_count += 1;
            }
        }

        // Per-surface materials carried inside the VO's own surfs/texs table.
        // `objects.rs` merges these with the top-level paths in
        // `build_per_surface_frame_ranges` → `parse_material_spec_with_prefix`,
        // so the bundle must include their textures too. Without this, VOs
        // with multi-material surfaces show untextured / fallback colors.
        let Ok(vo) = vector_object::parse_vo(&vo_bytes) else {
            continue;
        };
        let object_dir = paths
            .vo_path
            .rsplit_once('/')
            .map(|(dir, _)| format!("{dir}/"));
        for surface in &vo.surfaces {
            let Some(spec) = surface.texture_ref.as_deref() else {
                continue;
            };
            let m = vector_object::parse_material_spec_with_prefix(spec, object_dir.as_deref());
            for t in [
                m.diffuse.as_ref(),
                m.gloss.as_ref(),
                m.back.as_ref(),
                m.mask.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if try_pack_texture(&mut bundle, &mut vo_tex_seen, t) {
                    obj_tex_count += 1;
                }
            }
        }
    }
    println!(
        "  objects: {} vo files, {} textures packed",
        vo_count, obj_tex_count
    );

    // Pack each distinct building kind's CVO + every sub-VO + their textures.
    // Mirrors the renderer's load path in `renderer/buildings.rs`: for each
    // `BuildingInstance`, it fetches `Matrix/Building/bN.cvo`, parses it, and
    // reads each unit's `Model` + Texture*. The bundle needs all those assets
    // (plus the VO's own embedded surface texture) under their original keys.
    let mut building_kinds: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
    for b in &map.buildings {
        building_kinds.insert(b.kind);
    }
    let mut cvo_count = 0;
    let mut sub_vo_count = 0;
    let mut building_tex_count = 0;
    for kind in &building_kinds {
        let name = match kind {
            0 => "b0",
            1 => "b1",
            2 => "b2",
            3 => "b3",
            4 => "b4",
            5 => "b5",
            _ => continue,
        };
        let cvo_path = format!("Matrix/Building/{}.cvo", name);
        let cvo_key = cvo_path.to_uppercase();
        let cvo_bytes = match pkg.read_file(&cvo_key) {
            Ok(data) => {
                bundle.add(&cvo_path, data.clone());
                cvo_count += 1;
                data
            }
            Err(e) => {
                eprintln!("  MISS cvo: {} ({e})", cvo_path);
                continue;
            }
        };
        let group = vector_object::parse_cvo(&cvo_path, &cvo_bytes);
        for unit in &group.units {
            let vo_key = unit.model_path.to_uppercase();
            let vo_bytes = match pkg.read_file(&vo_key) {
                Ok(data) => {
                    bundle.add(&unit.model_path, data.clone());
                    sub_vo_count += 1;
                    data
                }
                Err(_) => {
                    eprintln!("  MISS building sub-vo: {}", unit.model_path);
                    continue;
                }
            };

            // CVO-level textures.
            for t in [
                unit.material.diffuse.as_ref(),
                unit.material.gloss.as_ref(),
                unit.material.back.as_ref(),
                unit.material.mask.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if try_pack_texture(&mut bundle, &mut vo_tex_seen, t) {
                    building_tex_count += 1;
                }
            }
            // VO's own embedded surface-texture fallback — the renderer uses
            // this when a CVO unit omits `Texture=`.
            if let Ok(vo) = vector_object::parse_vo(&vo_bytes) {
                let dir = unit
                    .model_path
                    .rsplit_once('/')
                    .map(|(d, _)| format!("{d}/"));
                for s in &vo.surfaces {
                    if let Some(spec) = s.texture_ref.as_deref() {
                        let m = vector_object::parse_material_spec_with_prefix(spec, dir.as_deref());
                        for t in [
                            m.diffuse.as_ref(),
                            m.gloss.as_ref(),
                            m.back.as_ref(),
                            m.mask.as_ref(),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            if try_pack_texture(&mut bundle, &mut vo_tex_seen, t) {
                                building_tex_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "  buildings: {} cvo files, {} sub-vos, {} textures packed",
        cvo_count, sub_vo_count, building_tex_count
    );

    // Pack sibling `.txt` files so the alpha-test override path in
    // `vector_object::resolve_alpha_test_with_txt` (which ports
    // Texture.cpp:113-136) works in the WASM build. There are ~119 of these
    // in robots.pkg, ~25 bytes each — trivial overhead. We add every .txt we
    // can find under the keys the loader will actually request (mixed-case
    // normalized paths) by looking at every bundled texture and probing for
    // a sibling.
    let texture_keys: Vec<String> = bundle.list_files().iter().map(|s| s.to_string()).collect();
    let mut txt_count = 0;
    for key in &texture_keys {
        let txt_key = match key.rsplit_once('.') {
            Some((stem, _)) => format!("{stem}.txt"),
            None => format!("{key}.txt"),
        };
        if bundle.read_file(&txt_key).is_some() {
            continue;
        }
        let pkg_key = txt_key.to_uppercase();
        if let Ok(data) = pkg.read_file(&pkg_key) {
            bundle.add(&txt_key, data);
            txt_count += 1;
        }
    }
    println!("  alpha-test txt siblings: {} packed", txt_count);

    let bytes = bundle.to_bytes();
    std::fs::create_dir_all("assets").ok();
    let out_path = format!("assets/{}.bundle", short_name);
    std::fs::write(&out_path, &bytes).unwrap();
    println!(
        "\nPacked {} textures + CMAP into {} ({:.1} MB)",
        tex_count,
        out_path,
        bytes.len() as f64 / 1024.0 / 1024.0
    );
}
