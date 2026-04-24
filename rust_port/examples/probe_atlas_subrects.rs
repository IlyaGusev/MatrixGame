//! Extract base_2.png from the bundle, decode it, and dump key
//! sub-rect pixel summaries — verifies what pihu / pich pylons
//! actually display for their default state.

use matrixgame_rs::gfx::bundle::AssetBundle;
use matrixgame_rs::matrix_lib::three_g::texture::decode_texture_bytes;

fn main() {
    let data = std::fs::read("assets/atoll.bundle").unwrap();
    let bundle = AssetBundle::from_bytes(&data).unwrap();
    for atlas in ["Matrix/Iface/base_1", "Matrix/Iface/base_2"] {
        let bytes = match bundle.read_file(atlas) {
            Some(b) => b,
            None => {
                println!("MISSING: {}", atlas);
                continue;
            }
        };
        let rgba = decode_texture_bytes(bytes).expect("decode");
        let w = rgba.width() as usize;
        let h = rgba.height() as usize;
        println!("{}: {}×{}", atlas, w, h);
        let pixels: &[u8] = &rgba;

        // Sample a sub-rect: print per-pixel alpha histogram + center
        // pixel RGB so we can tell if the area is non-empty.
        let sample = |label: &str, x: usize, y: usize, sw: usize, sh: usize| {
            let mut nonempty = 0u32;
            let mut total = 0u32;
            for py in y..(y + sh).min(h) {
                for px in x..(x + sw).min(w) {
                    let idx = (py * w + px) * 4;
                    if idx + 3 < pixels.len() {
                        let a = pixels[idx + 3];
                        if a > 16 {
                            nonempty += 1;
                        }
                        total += 1;
                    }
                }
            }
            let cx = x + sw / 2;
            let cy = y + sh / 2;
            let cidx = (cy * w + cx) * 4;
            let center = if cidx + 3 < pixels.len() {
                format!(
                    "rgba=({}, {}, {}, {})",
                    pixels[cidx],
                    pixels[cidx + 1],
                    pixels[cidx + 2],
                    pixels[cidx + 3]
                )
            } else {
                "out-of-bounds".into()
            };
            println!(
                "  {} ({},{},{}×{}): {}/{} pixels with alpha>16, center {}",
                label, x, y, sw, sh, nonempty, total, center
            );
        };

        if atlas.ends_with("base_2") {
            sample("pich [Normal]", 294, 0, 69, 48);
            sample("chas1 [Normal] (Pneumatic)", 0, 0, 69, 48);
            sample("chas2 [Normal]", 0, 256, 69, 48);
            sample("chas3 [Normal]", 207, 256, 69, 48);
            sample("hull6 [Normal]", 0, 187, 69, 69);
            // Check whole "chas1 row" at y=0
            sample("base_2 (0,0,256x48) chas1 strip", 0, 0, 256, 48);
        }
        if atlas.ends_with("base_1") {
            sample("pihe [Normal]", 88, 320, 38, 38);
            sample("pi1 [Normal]", 88, 358, 49, 49);
            sample("head1 [Normal]", 164, 320, 38, 38);
            sample("weap1 [Normal]", 186, 358, 49, 49);
        }
    }
}
