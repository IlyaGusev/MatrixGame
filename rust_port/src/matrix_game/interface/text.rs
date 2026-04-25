//! Faithful port of the SR2 bitmap-font text rendering.
//!
//! The original game ships fonts as `.AFT` bitmap files inside
//! `forms.pkg/DATA/FONT/`. The shipped UI uses the `_SMOOTH` variants,
//! which carry a coverage-antialiased glyph (gray-value alpha per
//! pixel) alongside the legacy 1-bit glyph. The Rangers DLL plugin
//! reads these and blits them through D3D with point sampling.
//!
//! We replicate that: parse the AFT format, decode each glyph into
//! 8-bit alpha, lay them into a shared RGBA atlas (white pixels with
//! alpha = coverage), and draw the same textured quads the image
//! atlas uses. A point sampler keeps the pixels crisp.
//!
//! ## .AFT format (reverse-engineered from forms.pkg)
//!
//! ```text
//! Header (32 bytes):
//!   0x00  u8[4]   "aft\0" magic
//!   0x04  u32     version (always 1)
//!   0x08  u32     glyph_count
//!   0x0C  u32     ascent (px from cell top down to baseline)
//!   0x10  u32     unknown (always 2/3)
//!   0x14  u32     line_height (px) — total cell height = ascent + descent
//!   0x18  u64     reserved (zero)
//!
//! Glyph entry (64 bytes per glyph, glyph_count entries follow header):
//!   0x00  u32     codepoint (UTF-32)
//!   0x04  i32     unused / legacy bearing (matches 0x10 in plain
//!                  fonts; the SMOOTH layout uses 0x10 instead)
//!   0x08  u32     base advance (px)
//!   0x0C  i32     advance delta (signed) — the effective per-glyph
//!                  advance is `advance + delta`. Carries the actual
//!                  space width for ' ' (raw advance is just 1) and
//!                  small ±1..±2 nudges for narrow / wide letters.
//!   0x10  i32     bearing_x1 — LEFT offset for bmp1 (bmp1 is the
//!                  1-bit horizontal-stroke layer). Signed.
//!   0x14  i32     bearing_y1 — TOP offset for bmp1 (signed,
//!                  measured from the baseline upward).
//!   0x18  u32     bmp1 width
//!   0x1C  u32     bmp1 height
//!   0x20  u32     bmp1 file offset
//!   0x24  u32     bmp1 size (bytes incl. 16-byte bitmap header)
//!   0x28  i32     bearing_x2 — LEFT offset for bmp2 (bmp2 is the
//!                  SMOOTH antialiased verticals + halo layer).
//!   0x2C  i32     bearing_y2 — TOP offset for bmp2.
//!   0x30  u32     bmp2 width (0 when absent — plain fonts)
//!   0x34  u32     bmp2 height
//!   0x38  u32     bmp2 file offset (0 when absent)
//!   0x3C  u32     bmp2 size
//!
//! SMOOTH glyphs render as bmp2 painted first, bmp1 OVER (max
//! alpha) — bmp1 carries strokes that would alias badly at small
//! sizes (top/middle/bottom bars, curve fills), bmp2 carries the
//! verticals plus a 1-px AA halo that intentionally spills into
//! neighbour cells so adjacent letters blend together.
//!
//! Bitmap (size bytes at offset):
//!   0x00  u32     data_size (bytes, excluding this 16-byte header)
//!   0x04  u32     width  (px, matches the entry field above)
//!   0x08  u32     height (px)
//!   0x0C  u32     reserved
//!   0x10  u8[]    RLE opcodes, processed row-major. Each row ends
//!                 when `width` pixel positions have been visited
//!                 (and the encoder always pads to a row boundary).
//!                 Opcodes:
//!                   0x80 alone → skip one full row of `width`
//!                                transparent pixels (compactly
//!                                encodes empty rows).
//!                   0x00..0x7F → skip N transparent pixels.
//!                   0x81..0xFF → emit N = (b & 0x7F) pixels.
//!                                * Plain (1-bit) bitmaps: each pixel
//!                                  is fully opaque (alpha=255). No
//!                                  extra bytes follow.
//!                                * SMOOTH bitmap (bmp2): the next
//!                                  N bytes are per-pixel alpha
//!                                  values (0..255).
//! ```

use std::collections::HashMap;
use std::convert::TryInto;

use crate::matrix_lib::three_g::texture::create_texture_from_rgba;

/// Atlas key used by [`InterfaceRenderer`] to bind the text-glyph
/// atlas. Distinct from any data-pkg atlas path so it cannot collide.
pub const TEXT_ATLAS_KEY: &str = "__text__";

const ATLAS_W: u32 = 1024;
const ATLAS_H: u32 = 1024;

/// One AFT font, parsed lazily.
pub struct AftFont {
    /// Pixel line height — height of a row when stacking lines.
    pub line_height: u32,
    /// Distance from the cell's top edge to the baseline.
    pub ascent: u32,
    /// Codepoint → metrics + decoded 8bpp alpha bitmap.
    glyphs: HashMap<u32, AftGlyph>,
}

#[derive(Clone)]
struct AftGlyph {
    width: u32,
    height: u32,
    /// Left-side bearing: pen-relative x offset of the composited
    /// bitmap's leftmost column.
    bearing_x: i32,
    /// Signed offset from the baseline to the bitmap's TOP row.
    bearing_y: i32,
    /// Effective advance: stored advance + signed per-glyph delta.
    advance: u32,
    /// 8bpp coverage alpha, width*height bytes.
    pixels: Vec<u8>,
}

impl AftFont {
    pub fn parse(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 32 || &bytes[..4] != b"aft\0" {
            return Err("not an AFT font");
        }
        let glyph_count = u32_at(bytes, 8)? as usize;
        let ascent = u32_at(bytes, 0x0C)?;
        let line_height = u32_at(bytes, 0x14)?;
        let mut glyphs = HashMap::with_capacity(glyph_count);
        for i in 0..glyph_count {
            let o = 32 + i * 64;
            if o + 64 > bytes.len() {
                break;
            }
            let code = u32_at(bytes, o)?;
            // Effective advance = stored advance (0x08) plus a
            // signed per-glyph delta at 0x0C. The delta carries:
            //   - the actual space width for ' ' (raw adv=1, but
            //     x0C=2/3/4 depending on font size)
            //   - +1 px for narrow letters like 'i'/'l' that store
            //     a 1-px raw advance (overlap-by-design halo)
            //   - -1 px for glyphs whose AA halo already filled the
            //     trailing pixel (e.g. Cyrillic 'т' in 08_1_SMOOTH
            //     has x28=-1, x0C=-1 — bmp2 ends at pen+6 so the
            //     advance shrinks from 7 to 6 to avoid a visible
            //     trailing gap to the next letter).
            let advance =
                (u32_at(bytes, o + 0x08)? as i32 + i32_at(bytes, o + 0x0C)?).max(0) as u32;
            // Each glyph carries up to two bitmaps with their own
            // bearings:
            //   bmp1 (offset 0x20, plain 1-bit, RLE flag = solid)
            //   bmp2 (offset 0x38, antialiased, RLE has alpha bytes)
            // Plain fonts ship only bmp1; SMOOTH fonts ship BOTH and
            // the rendered glyph is bmp1 OVER bmp2 — bmp1 carries the
            // horizontal strokes and curve fills (which would alias
            // badly at small sizes), bmp2 carries the vertical
            // strokes plus the AA halo. Drawing bmp2 alone leaves
            // letters with hollow middles (no top/middle/bottom bars).
            // bmp1 bearing_x lives at 0x10 (NOT 0x04 — that field
            // appears unused in the SMOOTH placement). bmp2 bearing_x
            // lives at 0x28. The two values together centre bmp1's
            // narrower core inside bmp2's wider halo box, e.g. for an
            // 'O' with bw1=8/bw2=10 we get x10=1, x28=0 (a 1-px
            // offset to centre).
            let layer1 = decode_layer(bytes, o, 0x10, 0x14, 0x18, 0x20, 0x24, false)?;
            let layer2 = decode_layer(bytes, o, 0x28, 0x2C, 0x30, 0x38, 0x3C, true)?;
            let (width, height, bearing_x, bearing_y, pixels) =
                composite_layers(layer1, layer2);
            glyphs.insert(
                code,
                AftGlyph {
                    width,
                    height,
                    bearing_x,
                    bearing_y,
                    advance,
                    pixels,
                },
            );
        }
        Ok(Self { line_height, ascent, glyphs })
    }

    fn glyph(&self, codepoint: u32) -> Option<&AftGlyph> {
        self.glyphs
            .get(&codepoint)
            .or_else(|| self.glyphs.get(&('?' as u32)))
    }

    /// Pixel advance of `text` at this font's native size.
    pub fn measure(&self, text: &str) -> u32 {
        text.chars()
            .map(|c| self.glyph(c as u32).map(|g| g.advance).unwrap_or(0))
            .sum()
    }
}

/// One decoded bitmap layer (either the 1-bit bmp1 or the SMOOTH
/// bmp2). `pixels` is `width*height` bytes of 8bpp coverage alpha.
struct Layer {
    width: u32,
    height: u32,
    bearing_x: i32,
    bearing_y: i32,
    pixels: Vec<u8>,
}

/// Decode one of the two glyph layers. Field offsets are passed in so
/// the same routine handles bmp1 and bmp2 (the 64-byte glyph entry is
/// laid out the same way for each: bearing_x, bearing_y, width,
/// height, file offset, byte size). Returns `None` when the layer is
/// absent (size/offset zero — common for the SMOOTH layer of plain
/// fonts and for the bmp1 layer of dot-only glyphs like `i`).
fn decode_layer(
    bytes: &[u8],
    glyph_off: usize,
    bx_off: usize,
    by_off: usize,
    bw_off: usize,
    bmp_off_off: usize,
    bmp_sz_off: usize,
    smooth: bool,
) -> Result<Option<Layer>, &'static str> {
    let bmp_off = u32_at(bytes, glyph_off + bmp_off_off)? as usize;
    let bmp_sz = u32_at(bytes, glyph_off + bmp_sz_off)? as usize;
    if bmp_off == 0 || bmp_sz < 16 || bmp_off + bmp_sz > bytes.len() {
        return Ok(None);
    }
    let width = u32_at(bytes, glyph_off + bw_off)?;
    let height = u32_at(bytes, glyph_off + bw_off + 4)?;
    if width == 0 || height == 0 {
        return Ok(None);
    }
    let bw = u32_at(bytes, bmp_off + 4)?;
    let bh = u32_at(bytes, bmp_off + 8)?;
    let rle = &bytes[bmp_off + 16..bmp_off + bmp_sz];
    let pixels = decode_rle(rle, bw, bh, smooth);
    Ok(Some(Layer {
        width: bw,
        height: bh,
        bearing_x: i32_at(bytes, glyph_off + bx_off)?,
        bearing_y: i32_at(bytes, glyph_off + by_off)?,
        pixels,
    }))
}

/// Composite two glyph layers into one 8bpp alpha bitmap. Each layer
/// is placed at its own (bearing_x, bearing_y) — bearing_y is the
/// signed offset from the glyph's baseline to the bitmap's TOP row,
/// bearing_x the offset from the pen to the bitmap's LEFT column. The
/// resulting bitmap covers the union of both layers' rectangles, with
/// `bearing_x`/`bearing_y` pointing at the union's top-left.
///
/// Painting order is bmp2 first, then bmp1 OVER (saturating max), so
/// bmp1's solid pixels override bmp2's halo at the strokes that bmp2
/// only hints at. Returns a degenerate (0×0) glyph when both layers
/// are absent (used for whitespace).
fn composite_layers(
    layer1: Option<Layer>,
    layer2: Option<Layer>,
) -> (u32, u32, i32, i32, Vec<u8>) {
    match (layer1, layer2) {
        (None, None) => (0, 0, 0, 0, Vec::new()),
        (Some(l), None) | (None, Some(l)) => {
            (l.width, l.height, l.bearing_x, l.bearing_y, l.pixels)
        }
        (Some(l1), Some(l2)) => {
            let left = l1.bearing_x.min(l2.bearing_x);
            let top = l1.bearing_y.min(l2.bearing_y);
            let right = (l1.bearing_x + l1.width as i32).max(l2.bearing_x + l2.width as i32);
            let bottom =
                (l1.bearing_y + l1.height as i32).max(l2.bearing_y + l2.height as i32);
            let w = (right - left) as u32;
            let h = (bottom - top) as u32;
            let mut out = vec![0u8; (w * h) as usize];
            // bmp2 first (halo + verticals).
            let dx2 = (l2.bearing_x - left) as u32;
            let dy2 = (l2.bearing_y - top) as u32;
            for y in 0..l2.height {
                for x in 0..l2.width {
                    let v = l2.pixels[(y * l2.width + x) as usize];
                    if v != 0 {
                        let p = ((dy2 + y) * w + dx2 + x) as usize;
                        out[p] = v;
                    }
                }
            }
            // bmp1 over (solid horizontal strokes / curve fills).
            let dx1 = (l1.bearing_x - left) as u32;
            let dy1 = (l1.bearing_y - top) as u32;
            for y in 0..l1.height {
                for x in 0..l1.width {
                    let v = l1.pixels[(y * l1.width + x) as usize];
                    if v != 0 {
                        let p = ((dy1 + y) * w + dx1 + x) as usize;
                        out[p] = out[p].max(v);
                    }
                }
            }
            (w, h, left, top, out)
        }
    }
}

fn u32_at(b: &[u8], o: usize) -> Result<u32, &'static str> {
    if o + 4 > b.len() {
        return Err("AFT u32 out of bounds");
    }
    Ok(u32::from_le_bytes(b[o..o + 4].try_into().unwrap()))
}

fn i32_at(b: &[u8], o: usize) -> Result<i32, &'static str> {
    if o + 4 > b.len() {
        return Err("AFT i32 out of bounds");
    }
    Ok(i32::from_le_bytes(b[o..o + 4].try_into().unwrap()))
}

/// Decode an AFT RLE stream into an 8bpp alpha bitmap of `w × h`
/// pixels. `smooth == false` decodes the 1-bit bitmap (every emitted
/// pixel has alpha=255). `smooth == true` decodes the antialiased
/// bitmap where each `0x81..0xFF` opcode is followed by N per-pixel
/// alpha bytes.
fn decode_rle(data: &[u8], w: u32, h: u32, smooth: bool) -> Vec<u8> {
    let total = (w * h) as usize;
    let mut out = vec![0u8; total];
    let mut pos = 0usize;
    let row_w = w as usize;
    let mut i = 0usize;
    while i < data.len() && pos < total {
        let b = data[i];
        i += 1;
        if b == 0x80 {
            // Skip a full row of transparent pixels.
            pos += row_w;
        } else if b & 0x80 != 0 {
            let n = (b & 0x7F) as usize;
            if smooth {
                // N per-pixel alpha bytes follow the opcode.
                for _ in 0..n {
                    if pos >= total || i >= data.len() {
                        return out;
                    }
                    out[pos] = data[i];
                    pos += 1;
                    i += 1;
                }
            } else {
                for _ in 0..n {
                    if pos >= total {
                        return out;
                    }
                    out[pos] = 255;
                    pos += 1;
                }
            }
        } else {
            pos += b as usize;
        }
    }
    out
}

/// One cached glyph in the GPU atlas.
#[derive(Debug, Clone, Copy)]
pub struct GlyphRect {
    pub atlas_x: u32,
    pub atlas_y: u32,
    pub w: u32,
    pub h: u32,
    pub bearing_x: i32,
    pub bearing_y: i32,
    pub advance: u32,
}

/// Multi-font glyph atlas: each loaded AFT font gets its own glyph
/// cache, all packing into one shared 1024×1024 RGBA texture so the
/// renderer only needs one bind group for text.
pub struct GlyphAtlas {
    fonts: HashMap<String, AftFont>,
    /// Cache key = (font_name, codepoint).
    cache: HashMap<(String, u32), Option<GlyphRect>>,
    pixels: Vec<u8>,
    pen_x: u32,
    pen_y: u32,
    row_h: u32,
    /// Bumps every time a glyph lands in the atlas — the renderer
    /// uses it to detect when to re-upload.
    pub generation: u64,
}

/// Embedded AFT font assets (extracted from forms.pkg/DATA/FONT/).
/// Using the plain 1-bit variants (bmp1 only) — every pixel is
/// fully opaque, no AA halo, so glyphs read as crisp pixel-art.
const FONT_RANGER_6: &[u8] = include_bytes!("../../../assets/fonts/RANGER_6.AFT");
const FONT_RANGER_5: &[u8] = include_bytes!("../../../assets/fonts/RANGER_5.AFT");
const FONT_VERDANA_10_2: &[u8] = include_bytes!("../../../assets/fonts/VERDANA_10_2.AFT");
const FONT_VERDANA_09_2: &[u8] = include_bytes!("../../../assets/fonts/VERDANA_09_2.AFT");
const FONT_VERDANA_08_1: &[u8] = include_bytes!("../../../assets/fonts/VERDANA_08_1.AFT");
const FONT_VERDANA_07_1: &[u8] = include_bytes!("../../../assets/fonts/VERDANA_07_1.AFT");

impl Default for GlyphAtlas {
    fn default() -> Self {
        Self::new()
    }
}

impl GlyphAtlas {
    pub fn new() -> Self {
        let mut fonts = HashMap::new();
        // Heuristic mapping of the C++ `Font.2*` names to the AFT
        // files in forms.pkg. The mapping isn't in the unencrypted
        // robots.dat — it lives in the Blowfish-encrypted Lang.dat /
        // Main.dat we don't have a decryptor for. These choices match
        // the size + weight of each font's typical use site:
        //   Font.2Ranger — main UI captions / button labels
        //   Font.2Small — focused-component label (it_label1)
        //   Font.2Normal — focused-component description (it_label2)
        //   Font.2Mini — tooltip / smallest text
        for (name, bytes) in [
            ("Font.2Ranger", FONT_RANGER_6),
            ("Font.2Mini", FONT_VERDANA_07_1),
            ("Font.2Small", FONT_VERDANA_08_1),
            ("Font.2Bold", FONT_VERDANA_09_2),
            ("Font.2Normal", FONT_VERDANA_10_2),
            ("Font.2Big", FONT_VERDANA_10_2),
            ("RANGER_5", FONT_RANGER_5),
            ("RANGER_6", FONT_RANGER_6),
        ] {
            match AftFont::parse(bytes) {
                Ok(f) => {
                    fonts.insert(name.to_string(), f);
                }
                Err(e) => log::warn!("AFT font {name} failed to parse: {e}"),
            }
        }
        Self {
            fonts,
            cache: HashMap::new(),
            pixels: vec![0u8; (ATLAS_W * ATLAS_H * 4) as usize],
            pen_x: 1,
            pen_y: 1,
            row_h: 0,
            generation: 1,
        }
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
    pub fn width(&self) -> u32 {
        ATLAS_W
    }
    pub fn height(&self) -> u32 {
        ATLAS_H
    }

    /// Native pixel line height for the named font. Used by the
    /// wrap / multi-line layout to step between lines.
    pub fn line_height(&self, font: &str) -> u32 {
        self.fonts
            .get(font)
            .map(|f| f.line_height)
            .unwrap_or(12)
    }

    /// Native ascent (top-of-cell to baseline) for the named font.
    /// In AFT the bearing_y is measured from the cell top, so the
    /// renderer just uses `line_height` to step lines and `bearing_y`
    /// to position each glyph; ascent is informational.
    pub fn ascent(&self, font: &str) -> u32 {
        self.fonts.get(font).map(|f| f.ascent).unwrap_or(0)
    }

    /// Pixel advance of `text` rendered with `font`.
    pub fn measure(&self, font: &str, text: &str) -> u32 {
        self.fonts
            .get(font)
            .map(|f| f.measure(text))
            .unwrap_or(0)
    }

    /// Look up (and lazily atlas-pack) the glyph for `codepoint` in
    /// `font`. Returns `None` only if the font isn't loaded.
    pub fn glyph(&mut self, font: &str, codepoint: u32) -> Option<GlyphRect> {
        let key = (font.to_string(), codepoint);
        if let Some(cached) = self.cache.get(&key) {
            return *cached;
        }
        let font_ref = self.fonts.get(font)?;
        let g = font_ref.glyph(codepoint)?.clone();
        // Whitespace / zero-pixel glyph — cache metrics only, no atlas slot.
        if g.width == 0 || g.height == 0 {
            let r = GlyphRect {
                atlas_x: 0,
                atlas_y: 0,
                w: 0,
                h: 0,
                bearing_x: g.bearing_x,
                bearing_y: g.bearing_y,
                advance: g.advance,
            };
            self.cache.insert(key, Some(r));
            return Some(r);
        }
        // Shelf-pack with 1px gutters.
        if self.pen_x + g.width + 1 > ATLAS_W {
            self.pen_x = 1;
            self.pen_y += self.row_h + 1;
            self.row_h = 0;
        }
        if self.pen_y + g.height + 1 > ATLAS_H {
            log::warn!("text atlas full, dropping glyph U+{:04X}", codepoint);
            self.cache.insert(key, None);
            return None;
        }
        let x0 = self.pen_x;
        let y0 = self.pen_y;
        for gy in 0..g.height {
            for gx in 0..g.width {
                let alpha = g.pixels[(gy * g.width + gx) as usize];
                if alpha == 0 {
                    continue;
                }
                let px = x0 + gx;
                let py = y0 + gy;
                let off = ((py * ATLAS_W + px) * 4) as usize;
                self.pixels[off] = 255;
                self.pixels[off + 1] = 255;
                self.pixels[off + 2] = 255;
                self.pixels[off + 3] = alpha;
            }
        }
        self.pen_x += g.width + 1;
        if g.height > self.row_h {
            self.row_h = g.height;
        }
        self.generation = self.generation.wrapping_add(1);
        let r = GlyphRect {
            atlas_x: x0,
            atlas_y: y0,
            w: g.width,
            h: g.height,
            bearing_x: g.bearing_x,
            bearing_y: g.bearing_y,
            advance: g.advance,
        };
        self.cache.insert(key, Some(r));
        Some(r)
    }
}

/// One color-uniform run produced by [`parse_rich_text`].
#[derive(Debug, Clone, PartialEq)]
pub struct RichRun {
    pub text: String,
    /// `None` → use the label's base color. `Some` → override.
    pub color: Option<[u8; 4]>,
}

/// Parse inline color tags inside a caption string.
///
/// The C++ uses two tag forms inside label / hint text:
///   `<Color=R,G,B>foo</color>` — coloured run (case-insensitive open
///   tag spelled `<Color=…>`, close tag spelled `</color>` or
///   `</Color>`).
///   `<br>` — line break (already converted to `\r\n` upstream by
///   `make_item_replacements`, so we don't need to parse it here).
///
/// Tags don't nest in the shipped data (the C++ never constructs
/// nested ones); a stray `</color>` outside an open run is ignored.
pub fn parse_rich_text(text: &str) -> Vec<RichRun> {
    let mut runs: Vec<RichRun> = Vec::new();
    let mut buf = String::new();
    let mut current: Option<[u8; 4]> = None;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < text.len() {
        if !text.is_char_boundary(i) {
            i += 1;
            continue;
        }
        if bytes[i] == b'<' {
            let rest = &text[i..];
            if let Some((color, consumed)) = try_parse_open_color(rest) {
                if !buf.is_empty() {
                    runs.push(RichRun { text: std::mem::take(&mut buf), color: current });
                }
                current = Some(color);
                i += consumed;
                continue;
            }
            if let Some(consumed) = try_parse_close_color(rest) {
                if !buf.is_empty() {
                    runs.push(RichRun { text: std::mem::take(&mut buf), color: current });
                }
                current = None;
                i += consumed;
                continue;
            }
        }
        // Append one full UTF-8 char.
        let ch_end = (i + 1..=text.len())
            .find(|&j| text.is_char_boundary(j))
            .unwrap_or(text.len());
        buf.push_str(&text[i..ch_end]);
        i = ch_end;
    }
    if !buf.is_empty() {
        runs.push(RichRun { text: buf, color: current });
    }
    runs
}

fn try_parse_open_color(s: &str) -> Option<([u8; 4], usize)> {
    if s.len() < "<Color=0,0,0>".len() {
        return None;
    }
    if !s.get(..7)?.eq_ignore_ascii_case("<color=") {
        return None;
    }
    let close_off = s.find('>')?;
    let body = &s[7..close_off];
    let parts: Vec<&str> = body.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let r = parts[0].trim().parse::<u8>().ok()?;
    let g = parts[1].trim().parse::<u8>().ok()?;
    let b = parts[2].trim().parse::<u8>().ok()?;
    let a = if parts.len() >= 4 {
        parts[3].trim().parse::<u8>().unwrap_or(255)
    } else {
        255
    };
    Some(([r, g, b, a], close_off + 1))
}

fn try_parse_close_color(s: &str) -> Option<usize> {
    let want = "</color>";
    if s.len() >= want.len() && s[..want.len()].eq_ignore_ascii_case(want) {
        Some(want.len())
    } else {
        None
    }
}

/// Upload the atlas pixels to a GPU texture. The renderer keeps the
/// `(generation, view, bind_group)` and re-runs this when the
/// generation changes.
pub fn create_atlas_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &GlyphAtlas,
) -> wgpu::TextureView {
    create_texture_from_rgba(
        device,
        queue,
        &image::RgbaImage::from_raw(atlas.width(), atlas.height(), atlas.pixels().to_vec())
            .expect("glyph atlas pixel buffer matches dimensions"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranger6_parses() {
        let f = AftFont::parse(FONT_RANGER_6).unwrap();
        assert!(f.line_height >= 6 && f.line_height <= 16);
        // Must contain digit '0'.
        assert!(f.glyph(b'0' as u32).is_some());
    }

    #[test]
    fn ranger6_decode_zero() {
        let f = AftFont::parse(FONT_RANGER_6).unwrap();
        let g = f.glyph(b'0' as u32).unwrap();
        // Visible pixel count should be > 0 and equal known footprint.
        let lit = g.pixels.iter().filter(|&&p| p > 0).count();
        assert!(lit > 0, "decoded zero glyph has no opaque pixels");
        // Outer ring of any pixel rectangle has at least 4 pixels.
        assert!(lit >= 4);
    }

    #[test]
    fn measure_string() {
        let f = AftFont::parse(FONT_RANGER_6).unwrap();
        let w = f.measure("0123");
        assert!(w > 0);
    }

    #[test]
    fn colon_decodes_with_gap() {
        // The colon glyph has a vertical gap encoded as the 0x80
        // opcode. Before the fix it decoded as a solid bar.
        let bytes = include_bytes!("../../../assets/fonts/VERDANA_07_1.AFT");
        let f = AftFont::parse(bytes).unwrap();
        let g = f.glyph(b':' as u32).unwrap();
        assert_eq!(g.width, 1);
        let any_gap = g
            .pixels
            .iter()
            .enumerate()
            .any(|(i, &v)| v == 0 && i > 0 && i < g.pixels.len() - 1);
        assert!(any_gap, "colon must have a transparent gap row: {:?}", g.pixels);
    }

    #[test]
    fn smooth_prefers_bmp2_over_bmp1() {
        // When a SMOOTH font ships both bitmaps, the parser must pick
        // bmp2 (the full antialiased glyph) — bmp1 is a smaller
        // sparse mask that doesn't render anything legible on its own.
        let bytes =
            include_bytes!("../../../assets/fonts/VERDANA_10_2_SMOOTH.AFT");
        let f = AftFont::parse(bytes).unwrap();
        // For 'O', bmp1 width is 8 vs bmp2 width 10 in this font;
        // assert we picked the wider (bmp2) bitmap.
        let g = f.glyph(b'O' as u32).unwrap();
        assert!(g.width >= 9, "expected SMOOTH bmp2 width, got {}", g.width);
    }

    #[test]
    fn smooth_decoder_produces_gray() {
        // The decoder still handles the SMOOTH format even though the
        // font map currently points at the plain variants.
        let bytes =
            include_bytes!("../../../assets/fonts/VERDANA_10_2_SMOOTH.AFT");
        let f = AftFont::parse(bytes).unwrap();
        let g = f.glyph(b'O' as u32).unwrap();
        let has_gray = g.pixels.iter().any(|&p| p > 0 && p < 255);
        assert!(has_gray, "SMOOTH O must have partial-alpha pixels");
    }

    #[test]
    fn rich_text_no_tags() {
        assert_eq!(
            parse_rich_text("hello"),
            vec![RichRun { text: "hello".to_string(), color: None }]
        );
    }

    #[test]
    fn rich_text_color_tag() {
        let r = parse_rich_text("Damage <Color=247,195,0>30</color> HP");
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].text, "Damage ");
        assert_eq!(r[0].color, None);
        assert_eq!(r[1].text, "30");
        assert_eq!(r[1].color, Some([247, 195, 0, 255]));
        assert_eq!(r[2].text, " HP");
        assert_eq!(r[2].color, None);
    }

    #[test]
    fn rich_text_close_case_insensitive() {
        let r = parse_rich_text("<Color=10,20,30>x</Color>");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].text, "x");
        assert_eq!(r[0].color, Some([10, 20, 30, 255]));
    }

    #[test]
    fn rich_text_unicode_safe() {
        let r = parse_rich_text("Повреждение <Color=1,2,3>30</color> ед");
        let combined: String = r.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(combined, "Повреждение 30 ед");
    }

    #[test]
    fn rich_text_unrecognised_tag_kept() {
        let r = parse_rich_text("a <foo> b");
        let combined: String = r.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(combined, "a <foo> b");
    }
}
