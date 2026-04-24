//! Dump the `Hints/0` border layout + `Hints/Bitmaps` asset paths.
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let bytes = std::fs::read("../Data/robots.dat").expect("robots.dat");
    let stor = Storage::from_bytes(&bytes).expect("storage parse");

    let hints_rec = stor.block_record("da", "Hints").expect("Hints");
    println!("=== da/Hints children ===");
    if let (Some(n), Some(r)) = (stor.get_buf(&hints_rec, "2"), stor.get_buf(&hints_rec, "3")) {
        for i in 0..n.arrays_count() {
            println!("  {} -> {}", n.get_as_wstr(i), r.get_as_wstr(i));
        }
    }
    println!("\n=== da/Hints top-level params ===");
    if let (Some(k), Some(v)) = (stor.get_buf(&hints_rec, "0"), stor.get_buf(&hints_rec, "1")) {
        for i in 0..k.arrays_count() {
            println!("  {} = {}", k.get_as_wstr(i), v.get_as_wstr(i));
        }
    }

    // Border id 0 — referenced by all hint templates (`0|…`).
    if let Some(b0) = stor.block_record(&hints_rec, "0") {
        println!("\n=== da/Hints/0 ===");
        if let (Some(k), Some(v)) = (stor.get_buf(&b0, "0"), stor.get_buf(&b0, "1")) {
            for i in 0..k.arrays_count() {
                println!("  {} = {}", k.get_as_wstr(i), v.get_as_wstr(i));
            }
        }
    }

    // Hints/Bitmaps asset aliases.
    if let Some(bmps) = stor.block_record(&hints_rec, "Bitmaps") {
        println!("\n=== da/Hints/Bitmaps ===");
        if let (Some(k), Some(v)) = (stor.get_buf(&bmps, "0"), stor.get_buf(&bmps, "1")) {
            for i in 0..k.arrays_count() {
                println!("  {} = {}", k.get_as_wstr(i), v.get_as_wstr(i));
            }
        }
    }
}
