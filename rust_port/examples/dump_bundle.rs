use matrixgame_rs::gfx::bundle::AssetBundle;

fn main() {
    let data = std::fs::read("assets/atoll.bundle").unwrap();
    let bundle = AssetBundle::from_bytes(&data).unwrap();
    let mut keys: Vec<&str> = bundle.list_files();
    keys.sort();
    println!(
        "bundle: {} entries, {:.1} MB",
        keys.len(),
        data.len() as f64 / 1e6
    );
    for k in keys.iter().take(30) {
        println!("  KEY: {}", k);
    }
    println!("---");
    for k in keys
        .iter()
        .filter(|k| k.to_lowercase().contains("water") || k.to_lowercase().contains("mirror"))
    {
        println!("  WATER: {}", k);
    }
}
