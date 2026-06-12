//! Dump the `Cannons/CannonN` sub-block from robots.dat.
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let bytes = std::fs::read("../Data/robots.dat").expect("robots.dat");
    let stor = Storage::from_bytes(&bytes).expect("storage parse");

    // Enumerate all children of `da` so we can see where cannon stats live.
    if let (Some(names), Some(recs)) = (stor.get_buf("da", "2"), stor.get_buf("da", "3")) {
        let n = names.arrays_count();
        println!("da/ children ({n}):");
        for i in 0..n {
            println!("  {} -> {}", names.get_as_wstr(i), recs.get_as_wstr(i));
        }
    }
    // The C++ loader reads cannons from `Models/Cannons`, not `Cannons`.
    let models_rec = stor.block_record("da", "Models").expect("no Models");
    let Some(cannons_rec) = stor.block_record(&models_rec, "Cannons") else {
        println!("no Models/Cannons");
        return;
    };
    println!("Models/Cannons -> rec={cannons_rec}");
    // List child blocks first.
    if let (Some(names), Some(recs)) = (
        stor.get_buf(&cannons_rec, "2"),
        stor.get_buf(&cannons_rec, "3"),
    ) {
        let n = names.arrays_count();
        println!("Cannons child blocks ({n}):");
        for i in 0..n {
            println!("  {} -> {}", names.get_as_wstr(i), recs.get_as_wstr(i));
        }
    }
    // Also list top-level params of the Cannons block.
    if let (Some(k), Some(v)) = (
        stor.get_buf(&cannons_rec, "0"),
        stor.get_buf(&cannons_rec, "1"),
    ) {
        let n = k.arrays_count();
        println!("Cannons top-level params ({n}):");
        for i in 0..n.min(30) {
            println!("  {} = {}", k.get_as_wstr(i), v.get_as_wstr(i));
        }
    }
    // Models/Cannons children are enumerated — use the "2"/"3" columns
    // to find whatever names the data team picked.
    let (names_buf, recs_buf) = (
        stor.get_buf(&cannons_rec, "2"),
        stor.get_buf(&cannons_rec, "3"),
    );
    let child_names: Vec<String> = names_buf
        .map(|b| (0..b.arrays_count()).map(|i| b.get_as_wstr(i)).collect())
        .unwrap_or_default();
    let child_recs: Vec<String> = recs_buf
        .map(|b| (0..b.arrays_count()).map(|i| b.get_as_wstr(i)).collect())
        .unwrap_or_default();
    for (name, rec) in child_names.iter().zip(child_recs.iter()) {
        println!("\n== {name} (rec={rec}) ==");
        if let (Some(k), Some(v)) = (stor.get_buf(&rec, "0"), stor.get_buf(&rec, "1")) {
            let n = k.arrays_count();
            for j in 0..n {
                println!("  {} = {}", k.get_as_wstr(j), v.get_as_wstr(j));
            }
        }
    }
}
