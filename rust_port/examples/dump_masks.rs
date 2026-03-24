use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();
    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();

    if let Some(bmpc) = stor.get_buf("bitmaps", "Bitmap") {
        println!("{} bitmap masks", bmpc.arrays_count());
        for i in 0..bmpc.arrays_count().min(3) {
            let data = bmpc.get_bytes(i);
            println!("  mask[{}]: {} bytes", i, data.len());
            if let Ok(img) = image::load_from_memory(data) {
                let rgba = img.to_rgba8();
                println!("    size: {}x{}", rgba.width(), rgba.height());
                // Check channels
                let p = rgba.get_pixel(32, 32).0;
                println!("    center pixel: R={} G={} B={} A={}", p[0], p[1], p[2], p[3]);
                // Check if grayscale
                let mut is_gray = true;
                for px in rgba.pixels() {
                    if px.0[0] != px.0[1] || px.0[1] != px.0[2] { is_gray = false; break; }
                }
                println!("    grayscale: {}", is_gray);
                // Alpha stats
                let mut min_a = 255u8;
                let mut max_a = 0u8;
                for px in rgba.pixels() { min_a = min_a.min(px.0[3]); max_a = max_a.max(px.0[3]); }
                println!("    alpha range: {}..{}", min_a, max_a);
                rgba.save(format!("assets/mask_{}.png", i)).ok();
            }
        }
    }
}
