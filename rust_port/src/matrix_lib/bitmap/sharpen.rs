//! Port of `sharpen.cpp` / `asharpen.asm` (MatrixLib/Bitmap) — the
//! VirtualDub-derived 3×3 convolution sharpen. Sole game use is
//! `CMatrixMapStatic::RenderToTexture` (MatrixMapStatic.cpp:652), which
//! runs `sharpen_run(out, in, 16)` after each `Make2xSmaller` step when
//! reducing the 256×256 robot-icon render down to the 64×64 medium
//! icon texture.
//!
//! The original picks between three implementations; what shipped
//! machines actually execute is:
//!
//! * interior pixels — `asm_sharpen_run_MMX` (MMX is always present):
//!   per channel (alpha included)
//!   `min(255, max(0, c*((256+8*lv)>>2 & ~7) - s8*(lv>>2)) >> 6)`
//!   where `c` is the center and `s8` the sum of the 8 neighbours.
//! * the 1-pixel border — the C `do_conv` path: RGB
//!   `clamp((s8*(-lv) + c*(256+8*lv)) >> 8, 0, 255)` with
//!   clamp-to-edge neighbour replication, and **alpha forced to 0**
//!   (`do_conv` composes `r<<16|g<<8|b` only). `do_conv` also adds a
//!   bias read from one past the end of its 9-entry kernel (`m[9]`,
//!   uninitialised stack in the original) — treated as 0 here.
//!
//! For `lv=16` (the only level the game uses) the interior and border
//! formulas are arithmetically identical on RGB.
//!
//! `shaders/icon_sharpen.wgsl` mirrors this math on the GPU for the
//! icon bake (the render never leaves VRAM); this module is the
//! reference implementation the shader is tested against.

use image::RgbaImage;

/// Port of `CBitmap::Make2xSmaller` (CBitmap.cpp:294, BytePP==4 branch).
/// Exact per-channel `(p00 + p01 + p10 + p11) >> 2` — the asm's
/// carry/rotate dance reconstructs precisely this truncating average.
pub fn make_2x_smaller(src: &RgbaImage) -> RgbaImage {
    let (w, h) = (src.width() / 2, src.height() / 2);
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p00 = src.get_pixel(x * 2, y * 2).0;
            let p01 = src.get_pixel(x * 2 + 1, y * 2).0;
            let p10 = src.get_pixel(x * 2, y * 2 + 1).0;
            let p11 = src.get_pixel(x * 2 + 1, y * 2 + 1).0;
            let mut d = [0u8; 4];
            for c in 0..4 {
                d[c] = ((p00[c] as u32 + p01[c] as u32 + p10[c] as u32 + p11[c] as u32) >> 2)
                    as u8;
            }
            out.put_pixel(x, y, image::Rgba(d));
        }
    }
    out
}

/// Port of `sharpen_run` (sharpen.cpp:222). `lv` ∈ [0..64].
pub fn sharpen_run(src: &RgbaImage, lv: i32) -> RgbaImage {
    let (w, h) = (src.width() as i32, src.height() as i32);
    assert!(w >= 3 && h >= 3, "sharpen_run needs at least 3x3");
    let mut out = RgbaImage::new(w as u32, h as u32);

    // MMX multipliers: `neg a_mult; shr 2` and `(b_mult >> 2) & 0xfff8`.
    let a4 = (lv as u32) >> 2;
    let b4 = (((256 + 8 * lv) as u32) >> 2) & 0xfff8;
    // Scalar border multipliers.
    let m0 = -lv;
    let m4 = 256 + 8 * lv;

    let texel = |x: i32, y: i32| src.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32).0;

    for y in 0..h {
        for x in 0..w {
            let c = texel(x, y);
            let mut s8 = [0u32; 4];
            for (dx, dy) in [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ] {
                let n = texel(x + dx, y + dy);
                for ch in 0..4 {
                    s8[ch] += n[ch] as u32;
                }
            }
            let border = x == 0 || y == 0 || x == w - 1 || y == h - 1;
            let mut d = [0u8; 4];
            if border {
                for ch in 0..3 {
                    let v = (s8[ch] as i32 * m0 + c[ch] as i32 * m4) >> 8;
                    d[ch] = v.clamp(0, 255) as u8;
                }
                d[3] = 0;
            } else {
                for ch in 0..4 {
                    let v = (c[ch] as u32 * b4).saturating_sub(s8[ch] * a4) >> 6;
                    d[ch] = v.min(255) as u8;
                }
            }
            out.put_pixel(x as u32, y as u32, image::Rgba(d));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_2x_smaller_truncates() {
        let mut src = RgbaImage::new(2, 2);
        src.put_pixel(0, 0, image::Rgba([1, 10, 255, 0]));
        src.put_pixel(1, 0, image::Rgba([1, 11, 255, 0]));
        src.put_pixel(0, 1, image::Rgba([1, 12, 255, 0]));
        src.put_pixel(1, 1, image::Rgba([2, 13, 255, 0]));
        let out = make_2x_smaller(&src);
        // (1+1+1+2)>>2 = 1, (10+11+12+13)>>2 = 11, (255*4)>>2 = 255.
        assert_eq!(out.get_pixel(0, 0).0, [1, 11, 255, 0]);
    }

    /// A flat image is a fixed point of the filter: c*(256+8lv) - 8c*lv
    /// = 256c. Interior keeps all channels; border zeroes alpha.
    #[test]
    fn sharpen_flat_image_identity() {
        let src = RgbaImage::from_pixel(5, 5, image::Rgba([100, 50, 25, 200]));
        let out = sharpen_run(&src, 16);
        assert_eq!(out.get_pixel(2, 2).0, [100, 50, 25, 200]);
        assert_eq!(out.get_pixel(0, 0).0, [100, 50, 25, 0]);
        assert_eq!(out.get_pixel(4, 3).0, [100, 50, 25, 0]);
    }

    /// Interior pixel against the hand-computed MMX formula.
    #[test]
    fn sharpen_interior_known_value() {
        let mut src = RgbaImage::from_pixel(3, 3, image::Rgba([100, 0, 0, 255]));
        src.put_pixel(1, 1, image::Rgba([200, 0, 0, 255]));
        let out = sharpen_run(&src, 16);
        // c=200 b4=96, s8=800 a4=4: (200*96 - 800*4)>>6 = (19200-3200)>>6 = 250
        assert_eq!(out.get_pixel(1, 1).0[0], 250);
    }

    /// Overshoot clamps to 255, undershoot saturates at 0.
    #[test]
    fn sharpen_clamps() {
        let mut src = RgbaImage::from_pixel(3, 3, image::Rgba([0, 255, 0, 255]));
        src.put_pixel(1, 1, image::Rgba([255, 0, 0, 255]));
        let out = sharpen_run(&src, 16);
        assert_eq!(out.get_pixel(1, 1).0[0], 255); // (255*96-0)>>6 = 382 → 255
        assert_eq!(out.get_pixel(1, 1).0[1], 0); // (0 - 2040*4) → sat 0
    }

    /// For lv=16 the border (scalar) and interior (MMX) formulas agree
    /// on RGB: 4*(c*96 - s*4) >> 8 == (c*96 - s*4) >> 6.
    #[test]
    fn sharpen_lv16_border_matches_interior_formula() {
        for c in [0u32, 7, 100, 255] {
            for s in (0u32..=2040).step_by(97) {
                let scalar = ((s as i32 * -16 + c as i32 * 384) >> 8).clamp(0, 255);
                let mmx = ((c * 96).saturating_sub(s * 4) >> 6).min(255);
                assert_eq!(scalar as u32, mmx, "c={c} s={s}");
            }
        }
    }
}
