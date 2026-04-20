use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();
    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();

    let tuc = stor.get_buf("texunions", "Data").unwrap();
    let botc = stor.get_buf("bottom", "Data").unwrap();
    let strings = stor.get_buf("strings", "String").unwrap();
    let bmpc = stor.get_buf("bitmaps", "Bitmap").unwrap();

    let un_data = tuc.get_bytes(0);
    let un: Vec<i32> = un_data
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Check tiles at positions (1,1), (1,2), (2,1), (2,2) — four meeting at boundary
    let dim = 16;
    for (label, k) in [
        ("(1,1)", dim + 1),
        ("(1,2)", dim + 2),
        ("(2,1)", 2 * dim + 1),
        ("(2,2)", 2 * dim + 2),
    ] {
        if un[k] < 0 {
            println!("{}: empty", label);
            continue;
        }
        let bot_raw = botc.get_bytes(un[k] as usize);
        let bot: Vec<i32> = bot_raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let base_id = bot[0] as usize;
        let base_name = if base_id < strings.arrays_count() {
            strings.get_as_wstr(base_id)
        } else {
            "??".to_string()
        };
        println!(
            "{}: base={} (id {}), {} overlays:",
            label,
            base_name,
            base_id,
            (bot.len() - 1) / 2
        );

        let mut bi = 1;
        while bi + 1 < bot.len() {
            let ids = bot[bi];
            let ibm = bot[bi + 1];
            bi += 2;
            let overlay_name = if ids >= 0 && (ids as usize) < strings.arrays_count() {
                strings.get_as_wstr(ids as usize)
            } else {
                format!("bitmap-only (ids={})", ids)
            };
            let mask_size = if (ibm as usize) < bmpc.arrays_count() {
                bmpc.get_bytes(ibm as usize).len()
            } else {
                0
            };
            println!(
                "  overlay: {} (ids={}, mask={}, mask_bytes={})",
                overlay_name, ids, ibm, mask_size
            );
        }
    }
}
