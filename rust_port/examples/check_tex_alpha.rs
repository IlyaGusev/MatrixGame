use matrixgame_rs::gfx::bundle::AssetBundle;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::three_g::texture::decode_texture_bytes;

fn main() {
    // Also peek at what the WASM bundle serves for the same textures.
    if let Ok(b) = std::fs::read("assets/atoll.bundle") {
        let bundle = AssetBundle::from_bytes(&b).unwrap();
        for key in ["Matrix/Obj/palm/palm00", "Matrix/Obj/tree/tree03"] {
            match bundle.read_file(key) {
                Some(data) => {
                    let is_dds = data.len() > 4 && &data[..4] == b"DDS ";
                    let is_png = data.len() > 8 && &data[..8] == b"\x89PNG\r\n\x1a\n";
                    println!(
                        "BUNDLE {}: {} bytes, dds={} png={}",
                        key,
                        data.len(),
                        is_dds,
                        is_png
                    );
                    if let Some(img) = decode_texture_bytes(data) {
                        let n = img.width() * img.height();
                        let lo = img.pixels().filter(|p| p.0[3] < 128).count() as u32;
                        let hi = n - lo;
                        println!(
                            "  alpha<128: {} ({}%), alpha>=128: {} ({}%)",
                            lo,
                            lo * 100 / n,
                            hi,
                            hi * 100 / n
                        );
                    }
                }
                None => println!("BUNDLE {}: MISSING", key),
            }
        }
    } else {
        println!("(no bundle at assets/atoll.bundle — run pack_bundle first)");
    }

    let data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();

    for tex_name in [
        "MATRIX/OBJ/PALM/PALM00.DDS",
        "MATRIX/OBJ/PALM/PALM00.PNG",
        "MATRIX/OBJ/TREE/TREE03.DDS",
    ] {
        let Ok(bytes) = pkg.read_file(tex_name) else {
            println!("{}: not found", tex_name);
            continue;
        };
        println!("\n{}: {} bytes", tex_name, bytes.len());
        if &bytes[0..4] == b"DDS " {
            let fourcc = std::str::from_utf8(&bytes[84..88]).unwrap_or("?");
            let w = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
            let h = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
            println!("  DDS: {}x{}, fourcc={:?}", w, h, fourcc);
        }
        let Some(img) = decode_texture_bytes(&bytes) else {
            println!("  decode failed");
            continue;
        };
        let mut alpha_hist = [0u32; 8];
        for p in img.pixels() {
            alpha_hist[(p.0[3] as usize) / 32] += 1;
        }
        let total = img.width() * img.height();
        println!("  decoded: {}x{}", img.width(), img.height());
        println!("  alpha histogram (buckets of 32):");
        for (i, &c) in alpha_hist.iter().enumerate() {
            let pct = 100.0 * c as f32 / total as f32;
            println!(
                "    {:>3}-{:>3}: {:>6} ({:5.1}%)",
                i * 32,
                (i + 1) * 32 - 1,
                c,
                pct
            );
        }
    }
}
