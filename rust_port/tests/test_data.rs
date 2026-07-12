//! Data-loading checks against the shipped robots.pkg / robots.dat
//! (converted from one-off check_* / probe_* examples). Skip gracefully
//! when Data/ is absent.

use matrixgame_rs::matrix_game::config::{ItemCharsTable, TurretProps};
use matrixgame_rs::matrix_game::logic::MapLogic;
use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use matrixgame_rs::matrix_lib::three_g::vector_object;

fn load_pkg() -> Option<PkgArchive> {
    let Ok(data) = std::fs::read("../Data/robots.pkg") else {
        eprintln!("skipping: ../Data/robots.pkg not present");
        return None;
    };
    Some(PkgArchive::from_bytes(data).expect("parse pkg"))
}

fn load_dat() -> Option<Storage> {
    let Ok(bytes) = std::fs::read("../Data/robots.dat") else {
        eprintln!("skipping: ../Data/robots.dat not present");
        return None;
    };
    Some(Storage::from_bytes(&bytes).expect("parse robots.dat"))
}

/// BuildingInstance parsing + CVO resolution: every building kind on
/// ATOLL must resolve to a parseable CVO whose sub-VOs and diffuse
/// textures exist in the pkg (the renderer fallback-fills silently).
#[test]
fn buildings_and_cvos_resolve() {
    let Some(pkg) = load_pkg() else { return };
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").expect("read CMAP");
    let map = GameMap::from_cmap_bytes(&cmap).expect("parse map");
    assert!(!map.buildings.is_empty(), "ATOLL has authored buildings");

    let mut kinds: Vec<u8> = map.buildings.iter().map(|b| b.kind).collect();
    kinds.sort();
    kinds.dedup();
    for kind in kinds {
        let path = format!("MATRIX/BUILDING/B{}.CVO", kind);
        let bytes = pkg.read_file(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let group = vector_object::parse_cvo(&path, &bytes);
        assert!(!group.units.is_empty(), "{path} has sub-units");
        for u in &group.units {
            let vo = u.model_path.to_uppercase();
            assert!(pkg.read_file(&vo).is_ok(), "{path}: missing sub-VO {vo}");
            if let Some(tex) = u.material.diffuse.as_deref() {
                let dds = format!("{}.DDS", tex.to_uppercase());
                assert!(pkg.read_file(&dds).is_ok(), "{path}: missing texture {dds}");
            }
        }
    }
}

/// RS_EFFECTS spawner records load from the shipped CMAPs.
#[test]
fn effect_spawners_load() {
    let Some(pkg) = load_pkg() else { return };
    let names: Vec<String> = pkg
        .list_files()
        .into_iter()
        .filter(|f| f.ends_with(".CMAP"))
        .map(|s| s.to_string())
        .collect();
    let mut total = 0usize;
    for n in &names {
        let cmap = pkg.read_file(n).expect("read CMAP");
        let map = GameMap::from_cmap_bytes(&cmap).unwrap_or_else(|e| panic!("{n}: {e:?}"));
        total += map.effect_spawners.len();
    }
    assert!(total > 0, "no effect spawners found across {} maps", names.len());
}

/// The reinforcement cooldown must be seeded at load and tick down.
#[test]
fn maintenance_cooldown_seeded_and_ticking() {
    let (Some(pkg), Some(dat)) = (load_pkg(), load_dat()) else { return };
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").expect("read CMAP");
    let map = GameMap::from_cmap_bytes(&cmap).expect("parse map");

    let mut game = MapLogic::with_seed(1);
    game.load_config(&dat);
    game.spawn_buildings(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    let seeded = game.maintenance_time;
    assert!(seeded > 0, "cooldown must be seeded at load");
    game.takt(10_000);
    assert!(game.maintenance_time > 0 && game.maintenance_time < seeded);
}

/// ItemChars structure tables + turret HP populate from robots.dat.
#[test]
fn item_chars_and_turret_props_parse() {
    let Some(dat) = load_dat() else { return };
    let chars = ItemCharsTable::from_matrix_data(&dat).expect("ItemChars");
    assert!(chars.chassis_structure.iter().all(|&v| v > 0));
    assert!(chars.armor_structure.iter().all(|&v| v > 0));
    assert!(chars.head_hp_add.iter().any(|&v| v > 0));

    let turrets = TurretProps::from_matrix_data(&dat).expect("TurretProps");
    for (i, c) in turrets.cannons.iter().enumerate() {
        assert!(c.hitpoint > 0.0, "turret {} hitpoint", i + 1);
    }
}
