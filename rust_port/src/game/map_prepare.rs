//! Map preparation — ports functions from MatrixMapPrepare.cpp.
//! BuildTexUnions (line 108), PointCalcNormals (line 20).

use std::collections::HashMap;

use crate::assets::storage::Storage;
use crate::game::bitmap::{blit_tile, copy_col, copy_row, merge_by_mask, merge_with_alpha};
use crate::game::common::TEX_BOTTOM_SIZE;
use crate::renderer::texture::create_texture_from_rgba_mipped;

/// Ports BuildTexUnions (MatrixMapPrepare.cpp:108-293).
pub fn build_tex_union_atlases(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    stor: &Storage,
    tex_union_dim: usize,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Vec<wgpu::TextureView> {
    let tuc = match stor.get_buf("texunions", "Data") {
        Some(b) => b,
        None => return vec![],
    };
    let botc = match stor.get_buf("bottom", "Data") {
        Some(b) => b,
        None => return vec![],
    };
    let strings = match stor.get_buf("strings", "String") {
        Some(b) => b,
        None => return vec![],
    };
    let bmpc = stor.get_buf("bitmaps", "Bitmap");

    let atlas_px = TEX_BOTTOM_SIZE * tex_union_dim;
    let union_size = tex_union_dim * tex_union_dim;

    let mut atlas_views = Vec::new();
    let mut src_cache: HashMap<usize, image::RgbaImage> = HashMap::new();
    let mut bmp_cache: HashMap<usize, image::RgbaImage> = HashMap::new();

    let load_src = |id: usize,
                    cache: &mut HashMap<usize, image::RgbaImage>,
                    strings: &crate::assets::storage::DataBuf,
                    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>| {
        if !cache.contains_key(&id) {
            if id < strings.arrays_count() {
                let path = strings
                    .get_as_wstr(id)
                    .split('?')
                    .next()
                    .unwrap_or("")
                    .replace('\\', "/")
                    .to_string();
                if let Some(data) = read_texture(&path) {
                    if let Ok(img) = image::load_from_memory(&data) {
                        cache.insert(id, img.to_rgba8());
                    }
                }
            }
        }
    };

    let mut atlas = image::RgbaImage::new(atlas_px as u32, atlas_px as u32);

    for i in 0..tuc.arrays_count() {
        if i == tuc.arrays_count() - 1 {
            for p in atlas.pixels_mut() {
                *p = image::Rgba([0, 0, 0, 255]);
            }
        }

        let un_data = tuc.get_bytes(i);
        let un: Vec<i32> = un_data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Pass 1: fill slots with base texture + overlay blending
        for k in 0..un.len().min(union_size) {
            if un[k] < 0 {
                continue;
            }
            let bot_idx = un[k] as usize;
            if bot_idx >= botc.arrays_count() {
                continue;
            }

            let bot_raw = botc.get_bytes(bot_idx);
            if bot_raw.len() < 4 {
                continue;
            }
            let bot: Vec<i32> = bot_raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let xx = (k % tex_union_dim) * TEX_BOTTOM_SIZE;
            let yy = (k / tex_union_dim) * TEX_BOTTOM_SIZE;

            let base_id = bot[0] as usize;
            load_src(base_id, &mut src_cache, strings, read_texture);
            if let Some(src) = src_cache.get(&base_id) {
                blit_tile(&mut atlas, xx, yy, src);
            }

            let mut bi = 1;
            while bi + 1 < bot.len() {
                let ids = bot[bi];
                let ibm = bot[bi + 1];
                bi += 2;

                let ibm_idx = ibm as usize;
                if !bmp_cache.contains_key(&ibm_idx) {
                    if let Some(bmp_buf) = bmpc {
                        if ibm_idx < bmp_buf.arrays_count() {
                            let png_data = bmp_buf.get_bytes(ibm_idx);
                            if let Ok(img) = image::load_from_memory(png_data) {
                                bmp_cache.insert(ibm_idx, img.to_rgba8());
                            }
                        }
                    }
                }

                if ids >= 0 {
                    let overlay_id = ids as usize;
                    load_src(overlay_id, &mut src_cache, strings, read_texture);
                    if let (Some(overlay), Some(mask)) =
                        (src_cache.get(&overlay_id), bmp_cache.get(&ibm_idx))
                    {
                        merge_by_mask(&mut atlas, xx, yy, overlay, mask);
                    }
                } else {
                    if let Some(mask_img) = bmp_cache.get(&ibm_idx) {
                        merge_with_alpha(&mut atlas, xx, yy, mask_img);
                    }
                }
            }
        }

        // Pass 2: extend edges of empty slots
        let tsz = atlas_px as i32;
        let tbs = TEX_BOTTOM_SIZE as i32;

        for k in 0..un.len().min(union_size) {
            if un[k] >= 0 {
                continue;
            }
            let xx = (k % tex_union_dim) as i32 * tbs;
            let yy = (k / tex_union_dim) as i32 * tbs;

            let lp = xx > 0 && un[k - 1] >= 0;
            let tp = yy > 0 && un[k - tex_union_dim] >= 0;
            let rp = xx < tsz - tbs && un[k + 1] >= 0;
            let bp = yy < tsz - tbs && un[k + tex_union_dim] >= 0;

            if lp {
                let (mut up, mut down) = (0i32, tbs);
                for u in 0..(tbs / 2 - 2) {
                    copy_col(&mut atlas, xx + u, yy + up, down - up, xx - 1, yy + up);
                    if tp {
                        up += 1;
                    }
                    if bp {
                        down -= 1;
                    }
                }
            }
            if tp {
                let (mut left, mut rite) = (0i32, tbs);
                for u in 0..(tbs / 2 - 2) {
                    copy_row(
                        &mut atlas,
                        xx + left,
                        yy + u,
                        rite - left,
                        xx + left,
                        yy - 1,
                    );
                    if lp {
                        left += 1;
                    }
                    if rp {
                        rite -= 1;
                    }
                }
            }
            if rp {
                let (mut up, mut down) = (0i32, tbs);
                for u in 1..=(tbs / 2 - 2) {
                    copy_col(
                        &mut atlas,
                        xx + tbs - u,
                        yy + up,
                        down - up,
                        xx + tbs,
                        yy + up,
                    );
                    if tp {
                        up += 1;
                    }
                    if bp {
                        down -= 1;
                    }
                }
            }
            if bp {
                let (mut left, mut rite) = (0i32, tbs);
                for u in 1..=(tbs / 2 - 2) {
                    copy_row(
                        &mut atlas,
                        xx + left,
                        yy + tbs - u,
                        rite - left,
                        xx + left,
                        yy + tbs,
                    );
                    if lp {
                        left += 1;
                    }
                    if rp {
                        rite -= 1;
                    }
                }
            }
        }

        for p in atlas.pixels_mut() {
            p.0[3] = 255;
        }

        let view = create_texture_from_rgba_mipped(device, queue, &atlas, 6);
        atlas_views.push(view);
    }

    atlas_views
}
