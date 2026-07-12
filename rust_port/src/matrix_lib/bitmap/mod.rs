//! Bitmap operations — ports CBitmap methods from MatrixLib/Bitmap/src/CBitmap.cpp.

pub mod sharpen;

use crate::matrix_game::common::TEX_BOTTOM_SIZE;

/// Copy a 64x64 tile onto the atlas at (dx, dy).
/// Ports CBitmap::Copy.
pub fn blit_tile(atlas: &mut image::RgbaImage, dx: usize, dy: usize, src: &image::RgbaImage) {
    let w = TEX_BOTTOM_SIZE.min(src.width() as usize);
    let h = TEX_BOTTOM_SIZE.min(src.height() as usize);
    for py in 0..h {
        for px in 0..w {
            atlas.put_pixel(
                (dx + px) as u32,
                (dy + py) as u32,
                *src.get_pixel(px as u32, py as u32),
            );
        }
    }
}

/// Ports CBitmap::MergeByMask (CBitmap.cpp:1239-1288).
/// Original: mask=0 → show overlay (sou2), mask=255 → show background (sou1).
///
/// Byte-identical with the C++ blend: fast paths at the extremes plus the
/// `>>8` intermediate formula (commented in the original as "не совсем
/// точная формула" — it's a truncating /256 divide, not a real /255). Keeps
/// the atlas output bit-for-bit with the shipped tiles.
pub fn merge_by_mask(
    atlas: &mut image::RgbaImage,
    dx: usize,
    dy: usize,
    overlay: &image::RgbaImage,
    mask: &image::RgbaImage,
) {
    let w = TEX_BOTTOM_SIZE
        .min(overlay.width() as usize)
        .min(mask.width() as usize);
    let h = TEX_BOTTOM_SIZE
        .min(overlay.height() as usize)
        .min(mask.height() as usize);
    for py in 0..h {
        for px in 0..w {
            let ax = (dx + px) as u32;
            let ay = (dy + py) as u32;
            let dst = atlas.get_pixel(ax, ay).0;
            let src = overlay.get_pixel(px as u32, py as u32).0;
            let mb = mask.get_pixel(px as u32, py as u32).0[0];
            let rgba = if mb == 0 {
                // `*des = *sou2` — show overlay (CBitmap.cpp:1281).
                [src[0], src[1], src[2], 255]
            } else if mb == 255 {
                // `*des = *sou1` — keep background (CBitmap.cpp:1282).
                [dst[0], dst[1], dst[2], 255]
            } else {
                let m = mb as u32;
                let inv = 255 - m;
                [
                    (((dst[0] as u32 * m) >> 8) + ((src[0] as u32 * inv) >> 8)) as u8,
                    (((dst[1] as u32 * m) >> 8) + ((src[1] as u32 * inv) >> 8)) as u8,
                    (((dst[2] as u32 * m) >> 8) + ((src[2] as u32 * inv) >> 8)) as u8,
                    255,
                ]
            };
            atlas.put_pixel(ax, ay, image::Rgba(rgba));
        }
    }
}

/// Ports CBitmap::MergeWithAlpha.
pub fn merge_with_alpha(
    atlas: &mut image::RgbaImage,
    dx: usize,
    dy: usize,
    src: &image::RgbaImage,
) {
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
            atlas.put_pixel(
                ax,
                ay,
                image::Rgba([
                    ((dst[0] as u16 * inv + s[0] as u16 * alpha) / 255) as u8,
                    ((dst[1] as u16 * inv + s[1] as u16 * alpha) / 255) as u8,
                    ((dst[2] as u16 * inv + s[2] as u16 * alpha) / 255) as u8,
                    255,
                ]),
            );
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

#[cfg(test)]
mod tests {
    use super::*;

    /// mask=0 fast path must return the overlay pixel unchanged.
    #[test]
    fn merge_by_mask_fully_overlays_on_mask_zero() {
        let mut atlas = image::RgbaImage::from_pixel(1, 1, image::Rgba([10, 20, 30, 255]));
        let overlay = image::RgbaImage::from_pixel(1, 1, image::Rgba([200, 150, 100, 255]));
        let mask = image::RgbaImage::from_pixel(1, 1, image::Rgba([0, 0, 0, 255]));
        merge_by_mask(&mut atlas, 0, 0, &overlay, &mask);
        assert_eq!(atlas.get_pixel(0, 0).0, [200, 150, 100, 255]);
    }

    /// mask=255 fast path must keep the atlas pixel unchanged.
    #[test]
    fn merge_by_mask_keeps_background_on_mask_255() {
        let mut atlas = image::RgbaImage::from_pixel(1, 1, image::Rgba([10, 20, 30, 255]));
        let overlay = image::RgbaImage::from_pixel(1, 1, image::Rgba([200, 150, 100, 255]));
        let mask = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
        merge_by_mask(&mut atlas, 0, 0, &overlay, &mask);
        assert_eq!(atlas.get_pixel(0, 0).0, [10, 20, 30, 255]);
    }

    /// Intermediate mask must use the C++ `>>8` blend so atlas output stays
    /// byte-identical with the shipped tiles: (dst*m)>>8 + (src*(255-m))>>8.
    #[test]
    fn merge_by_mask_uses_shifted_blend_for_intermediate() {
        let mut atlas = image::RgbaImage::from_pixel(1, 1, image::Rgba([200, 0, 0, 255]));
        let overlay = image::RgbaImage::from_pixel(1, 1, image::Rgba([100, 0, 0, 255]));
        let mask = image::RgbaImage::from_pixel(1, 1, image::Rgba([128, 0, 0, 255]));
        merge_by_mask(&mut atlas, 0, 0, &overlay, &mask);
        // (200*128)>>8 + (100*127)>>8 = 100 + 49 = 149
        assert_eq!(atlas.get_pixel(0, 0).0, [149, 0, 0, 255]);
    }
}
