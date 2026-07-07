//! Pack CMAP + textures into a single asset bundle for WASM.
//! Resolves missing extensions by trying .png and .dds in the pkg.
//!
//! Usage: `cargo run --example pack_bundle -- <map-name>`  (default: `atoll`).
//! Bare names resolve to `MATRIX/MAP/<NAME>.CMAP`. The output lands at
//! `assets/<name>.bundle`, matching the `?bundle=<url>` query the WASM loader
//! accepts.
use matrixgame_rs::gfx::bundle::AssetBundle;
use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
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
    let dat_bytes = std::fs::read("../Data/robots.dat");
    let dat_stor: Option<Storage> = match &dat_bytes {
        Ok(data) => {
            println!("  robots.dat ({} bytes)", data.len());
            bundle.add("robots.dat", data.clone());
            Storage::from_bytes(data).ok()
        }
        Err(e) => {
            eprintln!("  skip robots.dat: {}", e);
            None
        }
    };

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
    // Text / LabelsText atlases referenced by `if/Main`'s statics
    // (CInterface.cpp:540+ loader reads `sNormal = Matrix\\IFace\\text_1`
    // for `lives`, `name`, `bopis`, `bresg`, `fresg`, `prog`, ...).
    extra_paths.push("Matrix/Iface/text_1".to_string());
    extra_paths.push("Matrix/Iface/text_2".to_string());
    // Base-selection sprites referenced by `if/Main`'s per-kind
    // platform / resource-label statics.
    for i in 1..=6 {
        extra_paths.push(format!("Matrix/Iface/base_{}", i));
    }
    // Software-cursor sprite sheets — the `Cursors` block in robots.dat
    // maps mode names (arrow/cross_*/star) to these 4x4 frame grids.
    for c in [
        "arrow_blue",
        "cross_blue",
        "cross_red",
        "cross_yellow",
        "star",
    ] {
        extra_paths.push(format!("Matrix/Textures/Cursors/{}", c));
    }
    // Progress-bar sprite used by `CMatrixProgressBar`
    // (MatrixProgressBar.cpp:22 `TEXTURE_PATH_PB`). Ships only as DDS.
    extra_paths.push("Matrix/Textures/pb".to_string());
    // Turret-build slot marker — `TEXTURE_PATH_TURRET_RADIUS` at
    // StringConstants.hpp:114, used by SPOT_TURRET landscape decals
    // (MatrixObjectBuilding.cpp:1640). Ships as DDS.
    extra_paths.push("Matrix/Textures/LandSpots/spot_turr2".to_string());
    // Weapon / damage effect assets: the BBT billboard table (sortable
    // atlas + every intense texture from `da/Billboards/Textures`),
    // landscape spot decals, standalone effect textures, projectile
    // meshes and the debris catalog.
    if let Some(dat_s) = dat_stor.as_ref() {
        if let Some(bb_rec) = dat_s.block_record("da", "Billboards") {
            if let Some(ts) = dat_s.block_param(&bb_rec, "TexSort") {
                extra_paths.push(ts);
            }
            if let Some(tex_rec) = dat_s.block_record(&bb_rec, "Textures") {
                if let (Some(_k), Some(v)) =
                    (dat_s.get_buf(&tex_rec, "0"), dat_s.get_buf(&tex_rec, "1"))
                {
                    for i in 0..v.arrays_count() {
                        let val = v.get_as_wstr(i);
                        let parts: Vec<&str> = val.split(',').collect();
                        if parts.first().map(|s| s.trim()) == Some("1") {
                            if let Some(path) = parts.get(1) {
                                extra_paths.push(path.trim().to_string());
                            }
                        }
                    }
                }
            }
        }
        if let Some(models_rec) = dat_s.block_record("da", "Models") {
            if let Some(deb_rec) = dat_s.block_record(&models_rec, "Debris") {
                if let (Some(_k), Some(v)) =
                    (dat_s.get_buf(&deb_rec, "0"), dat_s.get_buf(&deb_rec, "1"))
                {
                    for i in 0..v.arrays_count() {
                        extra_paths.push(v.get_as_wstr(i));
                    }
                }
            }
        }
    }
    for p in [
        "Matrix/Textures/LandSpots/varonka",
        "Matrix/Textures/LandSpots/spot_hit",
        "Matrix/Textures/LandSpots/spot",
        "Matrix/Textures/LandSpots/sole",
        "Matrix/Textures/LandSpots/sole_w",
        "Matrix/Textures/LandSpots/sole_p",
        "Matrix/Textures/Effects/gun_fire",
        "Matrix/Textures/Effects/splash",
        "Matrix/Textures/Effects/bigboom",
        "Matrix/Textures/Effects/gun_bullets1",
        "Matrix/Textures/Effects/gun_bullets2",
        "Matrix/Textures/Billboard/zahspot",
        "Matrix/Textures/Billboard/Dust",
        "Matrix/Textures/Billboard/keelwater",
        "Matrix/Textures/Billboard/moveto",
        "Matrix/Flyer/SH.VO",
        "Matrix/Flyer/AH.VO",
        "Matrix/Flyer/TH.VO",
        "Matrix/Flyer/BH.VO",
        "Matrix/Flyer/Screw_SH.VO",
        "Matrix/Flyer/Screw_ah.VO",
        "Matrix/Flyer/Screw_th.VO",
        "Matrix/Flyer/Screw_bh.VO",
        "Matrix/Flyer/sh",
        "Matrix/Flyer/th",
        "Matrix/Flyer/Bh",
        "Matrix/Flyer/screwsh",
        "Matrix/Flyer/screwah",
        "Matrix/Flyer/screwth",
        "Matrix/Flyer/screwbh",
        "Matrix/Flyer/sh_gloss",
        "Matrix/Flyer/th_gloss",
        "Matrix/Flyer/bh_gloss",
        "Matrix/Bullets/roket.vo",
        "Matrix/Bullets/mina.vo",
        "Matrix/Bullets/bullet.vo",
    ] {
        extra_paths.push(p.to_string());
    }
    // Hint chrome + inline resource icons — referenced by
    // `CMatrixHint::PreloadBitmaps` (MatrixHint.cpp:441-459). Sources
    // come from `da/Hints/0/Source` (border0) and every alias under
    // `da/Hints/Bitmaps` (res_* / face_* / exit* / stat_* / bhole*).
    // Resolve the aliases from the shipped robots.dat so this list
    // stays in sync with whatever the data team names the hint assets.
    if let Some(dat_s) = dat_stor.as_ref() {
        if let Some(hints_rec) = dat_s.block_record("da", "Hints") {
            if let Some(b0) = dat_s.block_record(&hints_rec, "0") {
                if let Some(path) = dat_s.block_param(&b0, "Source") {
                    extra_paths.push(path);
                }
            }
            if let Some(bmps_rec) = dat_s.block_record(&hints_rec, "Bitmaps") {
                if let (Some(keys), Some(vals)) =
                    (dat_s.get_buf(&bmps_rec, "0"), dat_s.get_buf(&bmps_rec, "1"))
                {
                    for i in 0..keys.arrays_count().min(vals.arrays_count()) {
                        extra_paths.push(vals.get_as_wstr(i));
                    }
                }
            }
        }
    }
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

    // Textures referenced by the effect meshes (bullets + debris VOs)
    // — packed under their extensionless resolved names so the WASM
    // bundle reader (exact-key lookup) finds them.
    {
        let mut fx_vo_paths: Vec<String> = vec![
            "Matrix/Bullets/roket.vo".to_string(),
            "Matrix/Bullets/mina.vo".to_string(),
            "Matrix/Bullets/bullet.vo".to_string(),
        ];
        if let Some(dat_s) = dat_stor.as_ref() {
            if let Some(models_rec) = dat_s.block_record("da", "Models") {
                if let Some(deb_rec) = dat_s.block_record(&models_rec, "Debris") {
                    if let Some(v) = dat_s.get_buf(&deb_rec, "1") {
                        for i in 0..v.arrays_count() {
                            fx_vo_paths.push(v.get_as_wstr(i));
                        }
                    }
                }
            }
        }
        for vo_path in fx_vo_paths {
            let Ok(vo_bytes) = pkg.read_file(&vo_path) else {
                eprintln!("  MISS fx vo: {}", vo_path);
                continue;
            };
            let Ok(vo) = vector_object::parse_vo(&vo_bytes) else {
                continue;
            };
            for surf in &vo.surfaces {
                let Some(spec) = surf.texture_ref.as_deref() else {
                    continue;
                };
                if let Some(t) =
                    matrixgame_rs::matrix_lib::three_g::texture::resolve_texture_name(spec, None)
                {
                    let key = t.replace('\\', "/");
                    let up = key.to_uppercase();
                    for cand in [up.clone(), format!("{}.DDS", up), format!("{}.PNG", up)] {
                        if let Ok(data) = pkg.read_file(&cand) {
                            bundle.add(&key, data);
                            break;
                        }
                    }
                }
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
        |bundle: &mut AssetBundle, seen: &mut std::collections::HashSet<String>, t: &str| -> bool {
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
        let Some(paths) = matrixgame_rs::matrix_game::object::resolve_paths(&id_str) else {
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
                        let m =
                            vector_object::parse_material_spec_with_prefix(spec, dir.as_deref());
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

    // Pack robot chassis meshes — `Matrix\Robot\ChassisN.vo` for N in 1..=5
    // (MatrixObjectRobot.cpp:352, path from OBJECT_PATH_ROBOT_CHASSIS +
    // m_Kind). The renderer needs the VO, its diffuse/gloss textures,
    // and the embedded surface-texture fallback the VO may reference.
    let mut chassis_vo_count = 0;
    let mut chassis_tex_count = 0;
    for n in 1..=5u32 {
        let vo_path = format!("Matrix/Robot/Chassis{}.vo", n);
        let vo_key = vo_path.to_uppercase();
        let vo_bytes = match pkg.read_file(&vo_key) {
            Ok(data) => {
                bundle.add(&vo_path, data.clone());
                chassis_vo_count += 1;
                data
            }
            Err(_) => {
                eprintln!("  MISS chassis vo: {}", vo_path);
                continue;
            }
        };
        // Texture stem (C++ LoadObject takes the name part before .vo and
        // loads diffuse/gloss via the material composition machinery).
        // `_e`-suffixed variants are the enemy-tint diffuse the C++ swaps
        // in for non-player robots (MatrixObjectRobot.cpp:335-348).
        let stem = format!("Matrix/Robot/Chassis{}", n);
        for suffix in ["", "_gloss", "_e", "_e_gloss"] {
            let t = format!("{}{}", stem, suffix);
            if try_pack_texture(&mut bundle, &mut vo_tex_seen, &t) {
                chassis_tex_count += 1;
            }
        }
        if let Ok(vo) = vector_object::parse_vo(&vo_bytes) {
            let dir = vo_path.rsplit_once('/').map(|(d, _)| format!("{d}/"));
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
                            chassis_tex_count += 1;
                        }
                    }
                }
            }
        }
    }
    println!(
        "  robots: {} chassis vo files, {} textures packed",
        chassis_vo_count, chassis_tex_count
    );

    // Pack robot armor / head / weapon meshes for the constructor
    // preview AND for live enemy units. Paths follow `OBJECT_PATH_ROBOT_*`
    // from StringConstants.hpp:177-180. We include the `_e` enemy-tint
    // variants the C++ swaps in for non-player robots
    // (MatrixObjectRobot.cpp:335-348) — needed for map-spawned enemies
    // to render with correct side-color blends instead of the player's
    // yellow team-marker paint.
    let mut robot_part_vo_count = 0;
    let mut robot_part_tex_count = 0;
    let part_specs: [(&str, u32); 3] = [("Armor", 6), ("Head", 4), ("Weapon", 10)];
    for (folder, max_kind) in part_specs {
        for n in 1..=max_kind {
            let vo_path = format!("Matrix/Robot/{}{}.vo", folder, n);
            let vo_key = vo_path.to_uppercase();
            let vo_bytes = match pkg.read_file(&vo_key) {
                Ok(data) => {
                    bundle.add(&vo_path, data.clone());
                    robot_part_vo_count += 1;
                    data
                }
                Err(_) => {
                    eprintln!("  MISS robot part vo: {}", vo_path);
                    continue;
                }
            };
            let stem = format!("Matrix/Robot/{}{}", folder, n);
            for suffix in ["", "_gloss", "_e", "_e_gloss"] {
                let t = format!("{}{}", stem, suffix);
                if try_pack_texture(&mut bundle, &mut vo_tex_seen, &t) {
                    robot_part_tex_count += 1;
                }
            }
            if let Ok(vo) = vector_object::parse_vo(&vo_bytes) {
                let dir = vo_path.rsplit_once('/').map(|(d, _)| format!("{d}/"));
                for s in &vo.surfaces {
                    if let Some(spec) = s.texture_ref.as_deref() {
                        let m =
                            vector_object::parse_material_spec_with_prefix(spec, dir.as_deref());
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
                                robot_part_tex_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "  robot parts: {} armor/head/weapon vo files, {} textures packed",
        robot_part_vo_count, robot_part_tex_count
    );

    // Pack cannon/turret meshes — `Matrix/Cannon/{Basis,Turret{1-4},Shaft{1-4}}.vo`
    // (MatrixObjectCannon.cpp:209,229). Referenced by `CannonsRenderer::new`.
    let mut cannon_vo_count = 0;
    let mut cannon_tex_count = 0;
    let mut cannon_paths: Vec<String> = vec!["Matrix/Cannon/Basis.vo".into()];
    for n in 1..=4 {
        cannon_paths.push(format!("Matrix/Cannon/Turret{}.vo", n));
        cannon_paths.push(format!("Matrix/Cannon/Shaft{}.vo", n));
    }
    for vo_path in &cannon_paths {
        let vo_key = vo_path.to_uppercase();
        let Ok(vo_bytes) = pkg.read_file(&vo_key) else {
            eprintln!("  MISS cannon vo: {}", vo_path);
            continue;
        };
        bundle.add(vo_path, vo_bytes.clone());
        cannon_vo_count += 1;
        let dir = vo_path.rsplit_once('/').map(|(d, _)| format!("{d}/"));
        if let Ok(vo) = vector_object::parse_vo(&vo_bytes) {
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
                            cannon_tex_count += 1;
                        }
                    }
                }
            }
        }
    }
    // Also pack the base Basis diffuse/gloss in case VOs don't self-reference.
    for t in ["Matrix/Cannon/Basis", "Matrix/Cannon/Basis_Gloss"] {
        if try_pack_texture(&mut bundle, &mut vo_tex_seen, t) {
            cannon_tex_count += 1;
        }
    }
    println!(
        "  cannons: {} vo files, {} textures packed",
        cannon_vo_count, cannon_tex_count
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
