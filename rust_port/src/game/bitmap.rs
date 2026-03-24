//! Bitmap operations — ports CBitmap methods from MatrixLib/Bitmap/src/CBitmap.cpp.

use crate::game::common::TEX_BOTTOM_SIZE;

/// Copy a 64x64 tile onto the atlas at (dx, dy).
/// Ports CBitmap::Copy.
pub fn blit_tile(atlas: &mut image::RgbaImage, dx: usize, dy: usize, src: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(src.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(src.height() as usize);
    for py in 0..h {
        for px in 0..w {
            atlas.put_pixel((dx + px) as u32, (dy + py) as u32, *src.get_pixel(px as u32, py as u32));
        }
    }
}

/// Ports CBitmap::MergeByMask (CBitmap.cpp:1239-1288).
/// Original: mask=0 → show overlay (bm2), mask=255 → show background (bm1).
pub fn merge_by_mask(atlas: &mut image::RgbaImage, dx: usize, dy: usize, overlay: &image::RgbaImage, mask: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(overlay.width() as usize).min(mask.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(overlay.height() as usize).min(mask.height() as usize);
    for py in 0..h {
        for px in 0..w {
            let ax = (dx + px) as u32;
            let ay = (dy + py) as u32;
            let dst = atlas.get_pixel(ax, ay).0;
            let src = overlay.get_pixel(px as u32, py as u32).0;
            let m = mask.get_pixel(px as u32, py as u32).0;
            let alpha = (255 - m[0]) as u16;
            let inv = 255 - alpha;
            atlas.put_pixel(ax, ay, image::Rgba([
                ((dst[0] as u16 * inv + src[0] as u16 * alpha) / 255) as u8,
                ((dst[1] as u16 * inv + src[1] as u16 * alpha) / 255) as u8,
                ((dst[2] as u16 * inv + src[2] as u16 * alpha) / 255) as u8,
                255,
            ]));
        }
    }
}

/// Ports CBitmap::MergeWithAlpha.
pub fn merge_with_alpha(atlas: &mut image::RgbaImage, dx: usize, dy: usize, src: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(src.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(src.height() as usize);
    for py in 0..h {
        for px in 0..w {
            let ax = (dx + px) as u32;
            let ay = (dy + py) as u32;
            let dst = atlas.get_pixel(ax, ay).0;
            let s = src.get_pixel(px as u32, py as u32).0;
            let alpha = s[3] as u16;
            let inv = 255 - alpha;
            atlas.put_pixel(ax, ay, image::Rgba([
                ((dst[0] as u16 * inv + s[0] as u16 * alpha) / 255) as u8,
                ((dst[1] as u16 * inv + s[1] as u16 * alpha) / 255) as u8,
                ((dst[2] as u16 * inv + s[2] as u16 * alpha) / 255) as u8,
                255,
            ]));
        }
    }
}

/// Copy a single column of pixels (edge extension).
/// Ports CBitmap::Copy for single-pixel-wide strips.
pub fn copy_col(atlas: &mut image::RgbaImage, dx: i32, dy: i32, h: i32, sx: i32, sy: i32) {
    let aw = atlas.width() as i32;
    let ah = atlas.height() as i32;
    for i in 0..h {
        let fx = sx.clamp(0, aw - 1);
        let fy = (sy + i).clamp(0, ah - 1);
        let tx = dx.clamp(0, aw - 1);
        let ty = (dy + i).clamp(0, ah - 1);
        let p = *atlas.get_pixel(fx as u32, fy as u32);
        atlas.put_pixel(tx as u32, ty as u32, p);
    }
}

/// Copy a single row of pixels (edge extension).
pub fn copy_row(atlas: &mut image::RgbaImage, dx: i32, dy: i32, w: i32, sx: i32, sy: i32) {
    let aw = atlas.width() as i32;
    let ah = atlas.height() as i32;
    for i in 0..w {
        let fx = (sx + i).clamp(0, aw - 1);
        let fy = sy.clamp(0, ah - 1);
        let tx = (dx + i).clamp(0, aw - 1);
        let ty = dy.clamp(0, ah - 1);
        let p = *atlas.get_pixel(fx as u32, fy as u32);
        atlas.put_pixel(tx as u32, ty as u32, p);
    }
}
