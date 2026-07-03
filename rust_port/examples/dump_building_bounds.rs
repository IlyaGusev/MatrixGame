//! Dump per-kind building AABBs exactly as the renderer registers them
//! (union of every CVO sub-VO's frame-0 bounds).

use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::three_g::vector_object;

fn main() {
    let data = std::fs::read("../Data/robots.pkg").expect("robots.pkg");
    let pkg = PkgArchive::from_bytes(data).unwrap();
    for kind in 0..5u8 {
        let cvo_path = format!("Matrix/Building/b{kind}.cvo");
        let Ok(bytes) = pkg.read_file(&cvo_path) else {
            println!("kind {kind}: no {cvo_path}");
            continue;
        };
        let group = vector_object::parse_cvo(&cvo_path, &bytes);
        let mut bb: Option<([f32; 3], [f32; 3])> = None;
        for unit in &group.units {
            let Ok(vo_bytes) = pkg.read_file(&unit.model_path) else {
                println!("  kind {kind}: missing {}", unit.model_path);
                continue;
            };
            let Ok(mesh) = vector_object::parse_vo(&vo_bytes) else {
                continue;
            };
            let Some(f0) = mesh.frames.first() else { continue };
            println!(
                "  kind {kind} unit {:<28} bounds {:?} .. {:?}",
                unit.model_path, f0.bounds_min, f0.bounds_max
            );
            let e = bb.get_or_insert((f0.bounds_min, f0.bounds_max));
            for i in 0..3 {
                e.0[i] = e.0[i].min(f0.bounds_min[i]);
                e.1[i] = e.1[i].max(f0.bounds_max[i]);
            }
        }
        if let Some((mn, mx)) = bb {
            println!("kind {kind}: UNION {mn:?} .. {mx:?}\n");
        }
    }
}
