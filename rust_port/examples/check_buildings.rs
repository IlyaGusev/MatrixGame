//! Verify BuildingInstance parsing + CVO resolution against the shipped pkg.
use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::game::map::GameMap;
use matrixgame_rs::game::vo_loader;

fn main() {
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();

    println!("buildings: {}", map.buildings.len());
    for b in &map.buildings {
        println!(
            "  kind={} side={} angle={} pos=({:.1}, {:.1}, {:.1}) shadow=({},{})",
            b.kind, b.side, b.angle, b.x, b.y, b.build_z, b.shadow_kind, b.shadow_size
        );
    }

    // Resolve each kind's CVO + enumerate its sub-meshes with which assets
    // are actually present in the pkg. Catches missing textures / sub-VOs
    // before the renderer silently fallback-fills them.
    let mut kinds: Vec<u8> = map.buildings.iter().map(|b| b.kind).collect();
    kinds.sort();
    kinds.dedup();
    for kind in kinds {
        let path = format!("MATRIX/BUILDING/B{}.CVO", kind);
        let bytes = match pkg.read_file(&path) {
            Ok(b) => b,
            Err(e) => {
                println!("[kind {kind}] {path} not found: {e}");
                continue;
            }
        };
        let group = vo_loader::parse_cvo(&path, &bytes);
        println!("[kind {kind}] {path} → {} units:", group.units.len());
        for u in &group.units {
            let vo_upper = u.model_path.to_uppercase();
            let vo_ok = pkg.read_file(&vo_upper).is_ok();
            let tex = u.material.diffuse.as_deref().unwrap_or("(none)");
            let tex_upper = format!("{}.DDS", tex.to_uppercase());
            let tex_ok = pkg.read_file(&tex_upper).is_ok();
            println!(
                "  name='{}' id={:?} model={} (vo_ok={}) diffuse={} (dds_ok={})",
                u.name, u.id, u.model_path, vo_ok, tex, tex_ok
            );
        }
    }
}
