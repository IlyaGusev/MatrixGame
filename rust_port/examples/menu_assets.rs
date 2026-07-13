//! Extract the main-menu art used by index.html into assets/menu/:
//! decodes .GI images (SR2 shell format, types 0 and 2) from
//! Data/mainmenu.pkg and copies the per-map preview JPGs from
//! Data/robots.pkg.
//!
//!   cargo run --example menu_assets

use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;

fn u32_at(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn i32_at(d: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}

fn rgb565(lo: u8, hi: u8) -> [u8; 3] {
    let v = (lo as u16) | ((hi as u16) << 8);
    let r = ((v >> 11) & 0x1F) as u8;
    let g = ((v >> 5) & 0x3F) as u8;
    let b = (v & 0x1F) as u8;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

/// Decode a .GI image. Type 0 = raw 16bpp RGB565 or 32bpp BGRA;
/// type 2 = three RLE layers (opaque body, premultiplied outline,
/// 6-bit inverted alpha). Format per the community `ranger-tools` spec.
fn decode_gi(d: &[u8]) -> image::RgbaImage {
    assert_eq!(&d[0..2], b"gi", "not a GI file");
    let (sx, sy) = (i32_at(d, 8), i32_at(d, 12));
    let (fx, fy) = (i32_at(d, 16), i32_at(d, 20));
    let amask = u32_at(d, 36);
    let ftype = u32_at(d, 40);
    let layer_count = u32_at(d, 44) as usize;
    let (w, h) = ((fx - sx) as usize, (fy - sy) as usize);
    let mut out = vec![0u8; w * h * 4];

    match ftype {
        0 => {
            let off = u32_at(d, 64) as usize;
            if amask == 0xFF00_0000 {
                for i in 0..w * h {
                    let p = off + i * 4;
                    out[i * 4] = d[p + 2];
                    out[i * 4 + 1] = d[p + 1];
                    out[i * 4 + 2] = d[p];
                    out[i * 4 + 3] = d[p + 3];
                }
            } else {
                for i in 0..w * h {
                    let [r, g, b] = rgb565(d[off + i * 2], d[off + i * 2 + 1]);
                    out[i * 4..i * 4 + 4].copy_from_slice(&[r, g, b, 255]);
                }
            }
        }
        2 => {
            for li in 0..layer_count {
                let lh = 64 + 32 * li;
                let (off, size) = (u32_at(d, lh) as usize, u32_at(d, lh + 4) as usize);
                if size == 0 {
                    continue;
                }
                let start_x = (i32_at(d, lh + 8) - sx) as usize;
                let start_y = (i32_at(d, lh + 12) - sy) as usize;
                let (mut p, end) = (off + 16, off + size); // skip size/w/h/0 sub-header
                let (mut x, mut y) = (0usize, 0usize);
                while p < end {
                    let byte = d[p];
                    p += 1;
                    match byte {
                        0 | 0x80 => {
                            x = 0;
                            y += 1;
                        }
                        b if b > 0x80 => {
                            for _ in 0..(b & 0x7F) {
                                let idx = ((y + start_y) * w + start_x + x) * 4;
                                if li < 2 {
                                    let [r, g, b] = rgb565(d[p], d[p + 1]);
                                    p += 2;
                                    out[idx..idx + 4].copy_from_slice(&[r, g, b, 255]);
                                } else {
                                    let a = (63 - d[p]) << 2;
                                    p += 1;
                                    // Outline colors are premultiplied by alpha.
                                    if a != 0 && a != 255 {
                                        for c in 0..3 {
                                            let v = out[idx + c] as f32 / a as f32;
                                            out[idx + c] =
                                                (((v * 63.0).round() as u32) << 2).min(255) as u8;
                                        }
                                    }
                                    out[idx + 3] = a;
                                }
                                x += 1;
                            }
                        }
                        b => x += b as usize,
                    }
                }
            }
        }
        t => panic!("unsupported GI type {t}"),
    }
    image::RgbaImage::from_raw(w as u32, h as u32, out).unwrap()
}

/// Hypsometric shading with slope-based lighting and the map's own
/// water color — stand-in for maps that ship without a .JPG preview.
fn heightmap_preview(map: &GameMap) -> image::RgbImage {
    const N: u32 = 256;
    let (ww, wh) = (map.world_width(), map.world_height());
    let wc = map.water_color;
    let water = [(wc >> 16) as u8, (wc >> 8) as u8, wc as u8];
    image::RgbImage::from_fn(N, N, |px, py| {
        let x = (px as f32 + 0.5) / N as f32 * ww;
        let y = (py as f32 + 0.5) / N as f32 * wh;
        let z = map.get_z(x, y);
        if z <= -0.5 {
            let d = (-z / 20.0).clamp(0.0, 1.0);
            let k = 1.0 - 0.6 * d;
            image::Rgb([
                (water[0] as f32 * k) as u8,
                (water[1] as f32 * k) as u8,
                (water[2] as f32 * k) as u8,
            ])
        } else {
            let dzx = map.get_z(x + 8.0, y) - z;
            let dzy = map.get_z(x, y + 8.0) - z;
            let light = (0.75 + (dzx - dzy) * 0.02).clamp(0.35, 1.1);
            let t = (z / 90.0).clamp(0.0, 1.0);
            let base = [110.0 + 90.0 * t, 96.0 + 60.0 * t, 70.0 + 50.0 * t];
            image::Rgb([
                (base[0] * light).min(255.0) as u8,
                (base[1] * light).min(255.0) as u8,
                (base[2] * light).min(255.0) as u8,
            ])
        }
    })
}

fn main() {
    std::fs::create_dir_all("assets/menu/maps").unwrap();
    let menu_pkg =
        PkgArchive::from_bytes(std::fs::read("../Data/mainmenu.pkg").expect("read mainmenu.pkg"))
            .expect("parse mainmenu.pkg");

    let png_out = [
        ("DATA/FORMMAIN3/OLDCAPTION.GI", "assets/menu/logo.png"),
        ("DATA/FORMMAIN3/2SHIP1.GI", "assets/menu/ship.png"),
        ("DATA/FORMMAIN3/2GAALSHIP.GI", "assets/menu/gaalship.png"),
        ("DATA/FORMMAIN3/2PLANET.GI", "assets/menu/planet.png"),
        ("DATA/FORMMAIN3/CIRCLE.GI", "assets/menu/circle.png"),
        ("DATA/FORMMAIN3/MICROTEXT.GI", "assets/menu/microtext.png"),
    ];
    for (src, dst) in png_out {
        let img = decode_gi(&menu_pkg.read_file(src).expect(src));
        img.save(dst).unwrap();
        println!("{src} -> {dst} ({}x{})", img.width(), img.height());
    }

    // Background is opaque and huge — store as JPEG.
    let bg = decode_gi(&menu_pkg.read_file("DATA/FORMMAIN3/2BG.GI").unwrap());
    let bg = image::DynamicImage::ImageRgba8(bg).to_rgb8();
    let mut f = std::io::BufWriter::new(std::fs::File::create("assets/menu/bg.jpg").unwrap());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, 85)
        .encode_image(&bg)
        .unwrap();
    println!(
        "DATA/FORMMAIN3/2BG.GI -> assets/menu/bg.jpg ({}x{})",
        bg.width(),
        bg.height()
    );

    let robots_pkg =
        PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").expect("read robots.pkg"))
            .expect("parse robots.pkg");
    for name in [
        "ATOLL",
        "TRAINING",
        "CROSSFIRE",
        "ISLANDS",
        "REACTOR",
        "ASYLUM",
        "ARMAGEDD",
        "VIRUS",
    ] {
        let dst = format!("assets/menu/maps/{}.jpg", name.to_lowercase());
        if let Ok(data) = robots_pkg.read_file(&format!("MATRIX/MAP/{name}.JPG")) {
            std::fs::write(&dst, &data).unwrap();
            println!("MATRIX/MAP/{name}.JPG -> {dst} ({} bytes)", data.len());
        } else {
            // No shipped preview (e.g. ASYLUM) — shade the heightmap.
            let cmap = robots_pkg
                .read_file(&format!("MATRIX/MAP/{name}.CMAP"))
                .expect(name);
            let img = heightmap_preview(&GameMap::from_cmap_bytes(&cmap).unwrap());
            let mut f = std::io::BufWriter::new(std::fs::File::create(&dst).unwrap());
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, 88)
                .encode_image(&img)
                .unwrap();
            println!("MATRIX/MAP/{name}.CMAP -> {dst} (heightmap fallback)");
        }
    }
}
