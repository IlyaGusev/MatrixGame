use matrixgame_rs::assets::pkg_reader::PkgArchive;

fn main() {
    let data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();

    let files = pkg.list_files();

    // Terrain texture configs
    let mut txt_files: Vec<_> = files
        .iter()
        .filter(|f| f.contains("TER/") && f.ends_with(".TXT"))
        .collect();
    txt_files.sort();
    println!("=== TER/*.TXT ({}) ===", txt_files.len());
    for f in &txt_files[..txt_files.len().min(5)] {
        let data = pkg.read_file(f).unwrap();
        println!("--- {f} ---\n{}", String::from_utf8_lossy(&data));
    }

    // TERTOP configs
    let mut tertop: Vec<_> = files
        .iter()
        .filter(|f| f.contains("TERTOP/") && f.ends_with(".TXT"))
        .collect();
    tertop.sort();
    println!("\n=== TERTOP/*.TXT ({}) ===", tertop.len());
    for f in &tertop[..tertop.len().min(3)] {
        let data = pkg.read_file(f).unwrap();
        println!("--- {f} ---\n{}", String::from_utf8_lossy(&data));
    }

    // CMAP header dump
    let mut cmaps: Vec<_> = files.iter().filter(|f| f.ends_with(".CMAP")).collect();
    cmaps.sort();
    println!("\n=== CMAP files ({}) ===", cmaps.len());
    for f in &cmaps[..cmaps.len().min(5)] {
        println!("  {f}");
    }
    if let Some(cmap) = cmaps.first() {
        let data = pkg.read_file(cmap).unwrap();
        println!("\nFirst 512 bytes of {} ({} total):", cmap, data.len());
        for (i, chunk) in data[..512.min(data.len())].chunks(16).enumerate() {
            print!("  {:04x}: ", i * 16);
            for b in chunk {
                print!("{:02x} ", b);
            }
            for _ in chunk.len()..16 {
                print!("   ");
            }
            print!(" ");
            for b in chunk {
                print!(
                    "{}",
                    if b.is_ascii_graphic() || *b == b' ' {
                        *b as char
                    } else {
                        '.'
                    }
                );
            }
            println!();
        }
    }

    // CFG files
    let mut cfgs: Vec<_> = files.iter().filter(|f| f.contains("CFG")).collect();
    cfgs.sort();
    println!("\n=== CFG files ({}) ===", cfgs.len());
    for f in &cfgs {
        println!("  {f}");
    }
    if let Some(cfg) = cfgs.first() {
        let data = pkg.read_file(cfg).unwrap();
        let s = String::from_utf8_lossy(&data);
        println!("\n--- {} ---\n{}", cfg, &s[..s.len().min(3000)]);
    }
}
