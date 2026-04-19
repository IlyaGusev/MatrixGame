use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;

fn main() {
    let data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();

    let path = "MATRIX/OBJ/PALM/PALM00.VO";
    let vo_bytes = pkg.read_file(path).expect("PALM00.VO");
    println!("{} = {} bytes", path, vo_bytes.len());

    let stor = Storage::from_bytes(&vo_bytes).expect("parse VO storage");
    stor.dump_structure();

    // Peek at vertices (if they're 32-byte SVOVertex) and indices.
    if let Some(verts_buf) = stor.get_buf("verts", "data") {
        let n = verts_buf.arrays_count();
        println!("\nverts/data: {} arrays", n);
        for i in 0..n.min(3) {
            let b = verts_buf.get_bytes(i);
            println!(
                "  array {}: {} bytes ({} SVOVertex-sized entries)",
                i,
                b.len(),
                b.len() / 32
            );
            if b.len() >= 32 {
                let f =
                    |off: usize| f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]);
                println!(
                    "    v0 pos=({:.2},{:.2},{:.2}) n=({:.2},{:.2},{:.2}) uv=({:.3},{:.3})",
                    f(0),
                    f(4),
                    f(8),
                    f(12),
                    f(16),
                    f(20),
                    f(24),
                    f(28)
                );
            }
        }
    }
    if let Some(idxs) = stor.get_buf("idxs", "data") {
        let n = idxs.arrays_count();
        println!("\nidxs/data: {} arrays", n);
        for i in 0..n.min(3) {
            let b = idxs.get_bytes(i);
            println!(
                "  array {}: {} bytes ({} u16 indices)",
                i,
                b.len(),
                b.len() / 2
            );
        }
    }
    if let Some(surfs) = stor.get_buf("surfs", "texs") {
        println!("\nsurfs/texs: {} arrays", surfs.arrays_count());
        for i in 0..surfs.arrays_count().min(3) {
            println!("  tex[{}] = {:?}", i, surfs.get_as_wstr(i));
        }
    }
}
