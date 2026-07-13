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

    // Every CMAP in robots.pkg: preview jpg (shipped or heightmap
    // fallback) + a levels.txt manifest line `slug\tsize_x\tsize_y`
    // that index.html uses to build the map list.
    let robots_pkg =
        PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").expect("read robots.pkg"))
            .expect("parse robots.pkg");
    let mut cmaps: Vec<String> = robots_pkg
        .list_files()
        .iter()
        .filter(|f| {
            let up = f.to_uppercase();
            up.starts_with("MATRIX/MAP/") && up.ends_with(".CMAP")
        })
        .map(|f| f.to_string())
        .collect();
    cmaps.sort();
    let mut levels = String::new();
    for path in &cmaps {
        let stem = &path["MATRIX/MAP/".len()..path.len() - ".CMAP".len()];
        let slug = slugify(stem);
        let cmap = robots_pkg.read_file(path).expect(path);
        let map = GameMap::from_cmap_bytes(&cmap).unwrap_or_else(|e| panic!("{path}: {e}"));
        levels.push_str(&format!("{slug}\t{}\t{}\n", map.size_x, map.size_y));

        let dst = format!("assets/menu/maps/{slug}.jpg");
        let jpg_key = format!("MATRIX/MAP/{stem}.JPG");
        if let Ok(data) = robots_pkg.read_file(&jpg_key) {
            std::fs::write(&dst, &data).unwrap();
            println!("{jpg_key} -> {dst} ({} bytes)", data.len());
        } else {
            let img = heightmap_preview(&map);
            let mut f = std::io::BufWriter::new(std::fs::File::create(&dst).unwrap());
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, 88)
                .encode_image(&img)
                .unwrap();
            println!("{path} -> {dst} (heightmap fallback)");
        }
    }
    std::fs::write("assets/menu/levels.txt", &levels).unwrap();
    println!("assets/menu/levels.txt: {} maps", cmaps.len());

    // SR2HD quick-launch form art (the screen index.html recreates) +
    // the shell bitmap fonts it uses, from forms.pkg.
    let forms_pkg =
        PkgArchive::from_bytes(std::fs::read("../Data/forms.pkg").expect("read forms.pkg"))
            .expect("parse forms.pkg");
    for (src, dst) in [
        ("DATA/FORMLOADROBOT/2BG.GI", "assets/menu/form_bg.png"),
        ("DATA/FORMLOADROBOT/2SLOTNORMAL.GI", "assets/menu/slot_normal.png"),
        ("DATA/FORMLOADROBOT/2SLOTACTIVE.GI", "assets/menu/slot_active.png"),
        ("DATA/FORMLOADROBOT/2SLOTONMOUSE.GI", "assets/menu/slot_hover.png"),
        ("DATA/FORMLOADROBOT/2ICONBLAZER.GI", "assets/menu/icon_blazer.png"),
        ("DATA/FORMLOADROBOT/2ICONKELLER.GI", "assets/menu/icon_keller.png"),
        ("DATA/FORMLOADROBOT/2ICONTERRON.GI", "assets/menu/icon_terron.png"),
        ("DATA/FORMLOADROBOT/2ICONPLAYER.GI", "assets/menu/icon_player.png"),
    ] {
        let img = decode_gi(&forms_pkg.read_file(src).expect(src));
        img.save(dst).unwrap();
        println!("{src} -> {dst} ({}x{})", img.width(), img.height());
    }
    // RANGER_6 = the blocky chrome font (title/tabs/buttons, caps 6px);
    // VERDANA *_SMOOTH = the list/label font. Matches the shipped shell.
    // Vertical scrollbar art (common.pkg) for the list's scrollbar:
    // up/down arrow buttons + the thumb piece set (top/center/bottom).
    let common_pkg =
        PkgArchive::from_bytes(std::fs::read("../Data/common.pkg").expect("read common.pkg"))
            .expect("parse common.pkg");
    for (src, dst) in [
        ("DATA/SCROLLBAR/2V1_UPN.GI", "assets/menu/sb_up.png"),
        ("DATA/SCROLLBAR/2V1_DOWNN.GI", "assets/menu/sb_down.png"),
        ("DATA/SCROLLBAR/2V1_UPA.GI", "assets/menu/sb_up_a.png"),
        ("DATA/SCROLLBAR/2V1_DOWNA.GI", "assets/menu/sb_down_a.png"),
        ("DATA/SCROLLBAR/2V1_TOPN.GI", "assets/menu/sb_top.png"),
        ("DATA/SCROLLBAR/2V1_CENTERN.GI", "assets/menu/sb_center.png"),
        ("DATA/SCROLLBAR/2V1_BOTTOMN.GI", "assets/menu/sb_bottom.png"),
    ] {
        let img = decode_gi(&common_pkg.read_file(src).expect(src));
        img.save(dst).unwrap();
        println!("{src} -> {dst} ({}x{})", img.width(), img.height());
    }

    for font in [
        "RANGER_6",
        "VERDANA_09_2_SMOOTH",
        "VERDANA_10_2_SMOOTH",
        "VERDANA_11_3_SMOOTH",
    ] {
        let bytes = forms_pkg
            .read_file(&format!("DATA/FONT/{font}.AFT"))
            .expect(font);
        export_font(font, &bytes);
    }
}

/// Rasterize an AFT bitmap font into a white-on-transparent glyph
/// atlas (`font_<name>.png`) + metrics (`font_<name>.txt`: first line
/// `line_height ascent atlas_w atlas_h` in 1x units, then `code x y w
/// h bearing_x bearing_y advance` per glyph). The PNG is stored at 2x
/// (Catmull-Rom supersample) and index.html shows it via CSS mask
/// with `mask-size` at the 1x dims — much crisper on fractional-DPR
/// displays than a 1x atlas.
fn export_font(name: &str, bytes: &[u8]) {
    use matrixgame_rs::matrix_game::interface::text::AftFont;
    let font = AftFont::parse(bytes).expect("AFT parse");
    let codes = font.codepoints();
    let cell_w = codes
        .iter()
        .filter_map(|&c| font.glyph_data(c).map(|g| g.0))
        .max()
        .unwrap_or(1)
        + 1;
    let cell_h = codes
        .iter()
        .filter_map(|&c| font.glyph_data(c).map(|g| g.1))
        .max()
        .unwrap_or(1)
        + 1;
    let cols = 16u32;
    let rows = (codes.len() as u32).div_ceil(cols);
    let (aw, ah) = (cols * cell_w, rows * cell_h);
    let mut img = image::RgbaImage::new(aw, ah);
    let mut metrics = format!("{} {} {aw} {ah}\n", font.line_height, font.ascent);
    for (i, &code) in codes.iter().enumerate() {
        let (w, h, bx, by, adv, px) = font.glyph_data(code).unwrap();
        let (cx, cy) = (i as u32 % cols * cell_w, i as u32 / cols * cell_h);
        for y in 0..h {
            for x in 0..w {
                let a = px[(y * w + x) as usize];
                if a != 0 {
                    img.put_pixel(cx + x, cy + y, image::Rgba([255, 255, 255, a]));
                }
            }
        }
        metrics.push_str(&format!("{code} {cx} {cy} {w} {h} {bx} {by} {adv}\n"));
    }
    let img2x = image::imageops::resize(&img, aw * 2, ah * 2, image::imageops::FilterType::CatmullRom);
    let slug = name.to_lowercase();
    img2x.save(format!("assets/menu/font_{slug}.png")).unwrap();
    std::fs::write(format!("assets/menu/font_{slug}.txt"), metrics).unwrap();
    println!(
        "DATA/FONT/{name}.AFT -> assets/menu/font_{slug}.png ({} glyphs, lh={}, 2x atlas)",
        codes.len(),
        font.line_height
    );
}

/// Same rule as tools/lang_dat.py and pack_bundle.rs:
/// "MANSION (E)" → "mansion_e".
fn slugify(s: &str) -> String {
    let mut out = String::new();
    for c in s.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}
