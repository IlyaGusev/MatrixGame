use matrixgame_rs::assets::pkg_reader::PkgArchive;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();

    let files = pkg.list_files();
    let mut water: Vec<_> = files
        .iter()
        .filter(|f| f.to_lowercase().contains("water"))
        .collect();
    water.sort();
    println!("=== Water-related files ===");
    for f in &water {
        println!("  {f}");
    }

    // Also check data.txt for water config
    if let Ok(data) = pkg.read_file("MATRIX/ROBOT/DATA.TXT") {
        let s = String::from_utf8_lossy(&data);
        if let Some(pos) = s.to_lowercase().find("water") {
            let start = pos.saturating_sub(200);
            let end = (pos + 500).min(s.len());
            println!("\n=== data.txt around 'water' ===\n{}", &s[start..end]);
        }
    }

    // Check cfg.txt
    for path in &["MATRIX/CFG/ROBOTS/CFG.TXT", "MATRIX/CFG.TXT"] {
        if let Ok(data) = pkg.read_file(path) {
            let s = String::from_utf8_lossy(&data);
            if s.to_lowercase().contains("water") {
                println!("\n=== {} (water sections) ===", path);
                for line in s.lines() {
                    if line.to_lowercase().contains("water") {
                        println!("  {}", line);
                    }
                }
            }
        }
    }
}
