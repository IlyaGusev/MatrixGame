//! Dump texture union atlases as PNG files for visual inspection.
use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;
use matrixgame_rs::game::map::GameMap;
use std::collections::HashMap;

const TEX_BOTTOM_SIZE: usize = 64;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();

    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();
    let map = GameMap::from_cmap_bytes(&cmap_data).unwrap();

    let tex_union_dim = map.tex_union_dim;
    let atlas_px = TEX_BOTTOM_SIZE * tex_union_dim;

    let tuc = stor.get_buf("texunions", "Data").unwrap();
    let botc = stor.get_buf("bottom", "Data").unwrap();
    let strings = stor.get_buf("strings", "String").unwrap();
    let bmpc = stor.get_buf("bitmaps", "Bitmap");

    let mut src_cache: HashMap<usize, image::RgbaImage> = HashMap::new();
    let mut bmp_cache: HashMap<usize, image::RgbaImage> = HashMap::new();

    let load_src = |id: usize, cache: &mut HashMap<usize, image::RgbaImage>| {
        if !cache.contains_key(&id) && id < strings.arrays_count() {
            let path = strings.get_as_wstr(id).split('?').next().unwrap_or("").replace('\\', "/");
            let pkg_key = path.to_uppercase();
            if let Ok(data) = pkg.read_file(&pkg_key) {
                if let Ok(img) = image::load_from_memory(&data) {
                    cache.insert(id, img.to_rgba8());
                }
            }
        }
    };

    for i in 0..tuc.arrays_count() {
        let mut atlas = image::RgbaImage::new(atlas_px as u32, atlas_px as u32);
        if i == tuc.arrays_count() - 1 {
            for p in atlas.pixels_mut() { *p = image::Rgba([0, 0, 0, 255]); }
        }

        let un_data = tuc.get_bytes(i);
        let un: Vec<i32> = un_data.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let union_size = tex_union_dim * tex_union_dim;

        for k in 0..un.len().min(union_size) {
            if un[k] < 0 { continue; }
            let bot_idx = un[k] as usize;
            if bot_idx >= botc.arrays_count() { continue; }

            let bot_raw = botc.get_bytes(bot_idx);
            let bot: Vec<i32> = bot_raw.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if bot.is_empty() { continue; }

            let xx = (k % tex_union_dim) * TEX_BOTTOM_SIZE;
            let yy = (k / tex_union_dim) * TEX_BOTTOM_SIZE;

            // Base texture
            let base_id = bot[0] as usize;
            load_src(base_id, &mut src_cache);
            if let Some(src) = src_cache.get(&base_id) {
                blit(&mut atlas, xx, yy, src);
            }

            // Overlay pairs
            let mut bi = 1;
            while bi + 1 < bot.len() {
                let ids = bot[bi];
                let ibm = bot[bi + 1] as usize;
                bi += 2;

                if !bmp_cache.contains_key(&ibm) {
                    if let Some(bmp_buf) = bmpc {
                        if ibm < bmp_buf.arrays_count() {
                            if let Ok(img) = image::load_from_memory(bmp_buf.get_bytes(ibm)) {
                                bmp_cache.insert(ibm, img.to_rgba8());
                            }
                        }
                    }
                }

                if ids >= 0 {
                    let oid = ids as usize;
                    load_src(oid, &mut src_cache);
                    if let (Some(overlay), Some(mask)) = (src_cache.get(&oid), bmp_cache.get(&ibm)) {
                        merge_by_mask(&mut atlas, xx, yy, overlay, mask);
                    }
                } else {
                    if let Some(mask_img) = bmp_cache.get(&ibm) {
                        merge_with_alpha(&mut atlas, xx, yy, mask_img);
                    }
                }
            }
        }

        // Edge extension pass
        let tsz = atlas_px as i32;
        let tbs = TEX_BOTTOM_SIZE as i32;
        for k in 0..un.len().min(union_size) {
            if un[k] >= 0 { continue; }
            let xx = (k % tex_union_dim) as i32 * tbs;
            let yy = (k / tex_union_dim) as i32 * tbs;
            let lp = xx > 0 && un[k - 1] >= 0;
            let tp = yy > 0 && un[k - tex_union_dim] >= 0;
            let rp = xx < tsz - tbs && un[k + 1] >= 0;
            let bp = yy < tsz - tbs && un[k + tex_union_dim] >= 0;

            if lp { for u in 0..(tbs/2-2) { copy_col(&mut atlas, xx+u, yy + if tp {u} else {0}, tbs - if tp {u} else {0} - if bp {u} else {0}, xx-1, yy + if tp {u} else {0}); } }
            if tp { for u in 0..(tbs/2-2) { copy_row(&mut atlas, xx + if lp {u} else {0}, yy+u, tbs - if lp {u} else {0} - if rp {u} else {0}, xx + if lp {u} else {0}, yy-1); } }
            if rp { for u in 1..=(tbs/2-2) { copy_col(&mut atlas, xx+tbs-u, yy + if tp {u} else {0}, tbs - if tp {u} else {0} - if bp {u} else {0}, xx+tbs, yy + if tp {u} else {0}); } }
            if bp { for u in 1..=(tbs/2-2) { copy_row(&mut atlas, xx + if lp {u} else {0}, yy+tbs-u, tbs - if lp {u} else {0} - if rp {u} else {0}, xx + if lp {u} else {0}, yy+tbs); } }
        }

        let path = format!("assets/atlas_{}.png", i);
        atlas.save(&path).unwrap();
        println!("Saved {} ({}x{})", path, atlas_px, atlas_px);
    }
}

fn blit(atlas: &mut image::RgbaImage, dx: usize, dy: usize, src: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(src.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(src.height() as usize);
    for py in 0..h { for px in 0..w {
        atlas.put_pixel((dx+px) as u32, (dy+py) as u32, *src.get_pixel(px as u32, py as u32));
    }}
}

fn merge_by_mask(atlas: &mut image::RgbaImage, dx: usize, dy: usize, overlay: &image::RgbaImage, mask: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(overlay.width() as usize).min(mask.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(overlay.height() as usize).min(mask.height() as usize);
    for py in 0..h { for px in 0..w {
        let (ax, ay) = ((dx+px) as u32, (dy+py) as u32);
        let d = atlas.get_pixel(ax, ay).0;
        let s = overlay.get_pixel(px as u32, py as u32).0;
        let a = mask.get_pixel(px as u32, py as u32).0[0] as u16;
        let inv = 255 - a;
        atlas.put_pixel(ax, ay, image::Rgba([
            ((d[0] as u16 * inv + s[0] as u16 * a) / 255) as u8,
            ((d[1] as u16 * inv + s[1] as u16 * a) / 255) as u8,
            ((d[2] as u16 * inv + s[2] as u16 * a) / 255) as u8, 255,
        ]));
    }}
}

fn merge_with_alpha(atlas: &mut image::RgbaImage, dx: usize, dy: usize, src: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(src.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(src.height() as usize);
    for py in 0..h { for px in 0..w {
        let (ax, ay) = ((dx+px) as u32, (dy+py) as u32);
        let d = atlas.get_pixel(ax, ay).0;
        let s = src.get_pixel(px as u32, py as u32).0;
        let a = s[3] as u16;
        let inv = 255 - a;
        atlas.put_pixel(ax, ay, image::Rgba([
            ((d[0] as u16 * inv + s[0] as u16 * a) / 255) as u8,
            ((d[1] as u16 * inv + s[1] as u16 * a) / 255) as u8,
            ((d[2] as u16 * inv + s[2] as u16 * a) / 255) as u8, 255,
        ]));
    }}
}

fn copy_col(img: &mut image::RgbaImage, dx: i32, dy: i32, h: i32, sx: i32, sy: i32) {
    let (w, ih) = (img.width() as i32, img.height() as i32);
    for i in 0..h {
        let p = *img.get_pixel(sx.clamp(0,w-1) as u32, (sy+i).clamp(0,ih-1) as u32);
        img.put_pixel(dx.clamp(0,w-1) as u32, (dy+i).clamp(0,ih-1) as u32, p);
    }
}

fn copy_row(img: &mut image::RgbaImage, dx: i32, dy: i32, w: i32, sx: i32, sy: i32) {
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    for i in 0..w {
        let p = *img.get_pixel((sx+i).clamp(0,iw-1) as u32, sy.clamp(0,ih-1) as u32);
        img.put_pixel((dx+i).clamp(0,iw-1) as u32, dy.clamp(0,ih-1) as u32, p);
    }
}
