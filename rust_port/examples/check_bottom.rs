use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();
    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();

    let tuc = stor.get_buf("texunions", "Data").unwrap();
    let botc = stor.get_buf("bottom", "Data").unwrap();

    // Check first texunion
    let un_data = tuc.get_bytes(0);
    let un: Vec<i32> = un_data
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut overlay_counts = std::collections::HashMap::new();
    for k in 0..un.len().min(256) {
        if un[k] < 0 {
            continue;
        }
        let bot_raw = botc.get_bytes(un[k] as usize);
        let bot: Vec<i32> = bot_raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let overlay_count = (bot.len() - 1) / 2;
        *overlay_counts.entry(overlay_count).or_insert(0) += 1;
    }

    println!("Overlay count distribution:");
    let mut sorted: Vec<_> = overlay_counts.iter().collect();
    sorted.sort_by_key(|(k, _)| **k);
    for (count, num_tiles) in sorted {
        println!("  {} overlays: {} tiles", count, num_tiles);
    }
}
