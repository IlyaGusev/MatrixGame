use matrixgame_rs::assets::pkg_reader::PkgArchive;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();

    for path in [
        "MATRIX/TERTOP/BEACH SAND.DDS",
        "MATRIX/TERTOP/BEACH SAND00.DDS",
    ] {
        let data = pkg.read_file(path).unwrap();
        println!("=== {} ({} bytes) ===", path, data.len());
        let w = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let h = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let fourcc = std::str::from_utf8(&data[84..88]).unwrap_or("????");
        println!("  {}x{}, fourcc={}", w, h, fourcc);

        if let Some(img) = matrixgame_rs::renderer::terrain::decode_texture_bytes(&data) {
            let mut min_a = 255u8;
            let mut max_a = 0u8;
            let mut sum_a = 0u64;
            let mut count = 0u64;
            for p in img.pixels() {
                let a = p.0[3];
                min_a = min_a.min(a);
                max_a = max_a.max(a);
                sum_a += a as u64;
                count += 1;
            }
            println!(
                "  alpha: min={}, max={}, avg={:.1}",
                min_a,
                max_a,
                sum_a as f64 / count as f64
            );

            let out = path
                .rsplit('/')
                .next()
                .unwrap()
                .replace(".DDS", "_decoded.png");
            img.save(format!("assets/{out}")).ok();
            println!("  saved to assets/{out}");
        }
    }
}
