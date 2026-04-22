use matrixgame_rs::matrix_lib::base::pack::PkgArchive;

#[test]
fn list_pkg_contents() {
    let data = std::fs::read("../Data/robots.pkg").expect("failed to read robots.pkg");
    let pkg = PkgArchive::from_bytes(data).expect("failed to parse pkg");

    let mut files = pkg.list_files();
    files.sort();

    println!("=== {} files in robots.pkg ===", files.len());
    for f in &files[..files.len().min(50)] {
        println!("  {f}");
    }
    if files.len() > 50 {
        println!("  ... and {} more", files.len() - 50);
    }
}

#[test]
fn read_various_files() {
    let data = std::fs::read("../Data/robots.pkg").expect("failed to read robots.pkg");
    let pkg = PkgArchive::from_bytes(data).expect("failed to parse pkg");

    let files = pkg.list_files();
    let mut by_ext: std::collections::HashMap<String, Vec<String>> = Default::default();
    for f in &files {
        let ext = f.rsplit('.').next().unwrap_or("").to_string();
        by_ext.entry(ext).or_default().push(f.to_string());
    }

    // Print extension stats
    let mut exts: Vec<_> = by_ext.iter().map(|(k, v)| (k.clone(), v.len())).collect();
    exts.sort_by(|a, b| b.1.cmp(&a.1));
    println!("=== File types ===");
    for (ext, count) in &exts {
        println!("  .{ext}: {count} files");
    }

    // Try reading one file of each type
    println!("\n=== Reading samples ===");
    let mut ok = 0;
    let mut fail = 0;
    for (ext, paths) in &by_ext {
        let path = &paths[0];
        match pkg.read_file(path) {
            Ok(data) => {
                println!("  OK  .{ext}: {path} ({} bytes)", data.len());
                ok += 1;
            }
            Err(e) => {
                println!("  ERR .{ext}: {path}: {e}");
                fail += 1;
            }
        }
    }
    println!("\n{ok} ok, {fail} failed");
    assert_eq!(fail, 0, "some files failed to read");
}
