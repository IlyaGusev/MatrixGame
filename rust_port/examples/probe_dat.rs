use matrixgame_rs::assets::storage::Storage;

fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();

    // Exercise the two new BlockPar helpers directly — same chain that Sky::new runs.
    for name in [
        "Default",
        "BlueMoon",
        "Stars",
        "AlienBlue",
        "DarkGreen",
        "Black",
        "Mars",
        "Missing",
    ] {
        println!("=== Sky/{} ===", name);
        let Some(sky_root) = stor.block_record("da", "Sky") else {
            println!("  no Sky block");
            continue;
        };
        let Some(rec) = stor.block_record(&sky_root, name) else {
            println!("  not found");
            continue;
        };
        for key in [
            "Angle",
            "DeltaAngle",
            "Fore",
            "Right",
            "Back",
            "Left",
            "Reflection",
        ] {
            println!("  {} = {:?}", key, stor.block_param(&rec, key).as_deref());
        }
    }
}
