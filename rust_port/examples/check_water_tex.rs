use matrixgame_rs::assets::pkg_reader::PkgArchive;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();

    for (name, path) in [("water", "MATRIX/TEXTURES/WATER/1.DDS"), ("mirror", "MATRIX/TEXTURES/WATER/MIRROR.DDS")] {
        let data = pkg.read_file(path).unwrap();
        println!("=== {} ({} bytes) ===", name, data.len());
        // DDS header
        let w = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let h = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let fourcc = std::str::from_utf8(&data[84..88]).unwrap_or("????");
        println!("  {}x{}, fourcc={}", w, h, fourcc);

        // Decode and check alpha
        if let Some(img) = matrixgame_rs::renderer::terrain::decode_texture_bytes(&data) {
            let mut min_a = 255u8;
            let mut max_a = 0u8;
            let mut sum_a = 0u64;
            let mut count = 0u64;
            for p in img.pixels() {
                let a = p.0[3];
                if a < min_a { min_a = a; }
                if a > max_a { max_a = a; }
                sum_a += a as u64;
                count += 1;
            }
            println!("  alpha: min={}, max={}, avg={:.1}", min_a, max_a, sum_a as f64 / count as f64);

            // Save for visual inspection
            img.save(format!("assets/{}_decoded.png", name)).ok();
            println!("  saved to assets/{}_decoded.png", name);
        }
    }
}
