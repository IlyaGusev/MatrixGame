use matrixgame_rs::gfx::bundle::AssetBundle;

fn main() {
    let raw = std::fs::read("assets/atoll.bundle").expect("read bundle");
    let b = AssetBundle::from_bytes(&raw).unwrap();
    for path in [
        "Matrix/Iface/base_2",
        "Matrix/Iface/text_1",
        "Matrix/Iface/interface1",
    ] {
        if let Some(data) = b.read_file(path) {
            let hdr: Vec<String> = data.iter().take(16).map(|b| format!("{:02x}", b)).collect();
            println!("{}  len={}  first16={}", path, data.len(), hdr.join(" "));
            // PNG magic: 89 50 4E 47 0D 0A 1A 0A
            // DDS magic: 44 44 53 20 (DDS )
            if data.len() >= 8 && &data[0..8] == b"\x89PNG\r\n\x1a\n" {
                println!("  → PNG");
            } else if data.len() >= 4 && &data[0..4] == b"DDS " {
                println!("  → DDS");
            } else {
                println!("  → UNKNOWN FORMAT");
            }
        } else {
            println!("{}  MISSING", path);
        }
    }
}
