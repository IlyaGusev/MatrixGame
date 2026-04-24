//! Faithful port of the SR2 bitmap-font text rendering.
//!
//! The original game ships fonts as `.AFT` bitmap files inside
//! `forms.pkg/DATA/FONT/`. Each font is a fixed-size, 1-bpp, RLE-
//! compressed glyph table. The Rangers DLL plugin (`m_RangersText`)
//! reads these and blits them through D3D with point sampling — there
//! is no scalable / antialiased rendering.
//!
//! We replicate that exactly: parse the AFT format, lay each glyph out
//! into a shared RGBA atlas (white pixels with alpha = pixel coverage),
//! and let `InterfaceRenderer` draw the same kind of textured quads it
//! draws for image atlases. A point sampler keeps the pixels crisp.
//!
//! ## .AFT format (reverse-engineered from forms.pkg)
//!
//! ```text
//! Header (32 bytes):
//!   0x00  u8[4]   "aft\0" magic
//!   0x04  u32     version (always 1)
//!   0x08  u32     glyph_count
//!   0x0C  u32     ascent (px from cell top down to baseline)
//!   0x10  u32     unknown (always 2)
//!   0x14  u32     line_height (px) — total cell height = ascent + descent
//!   0x18  u64     reserved (zero)
//!
//! Glyph entry (64 bytes per glyph, glyph_count entries follow header):
//!   0x00  u32     codepoint (UTF-32; ASCII or Cyrillic)
//!   0x04  u32     unknown
//!   0x08  u32     advance (px) — pen step including letter spacing
//!   0x0C  u64     reserved
//!   0x14  i32     bearing_y — signed offset from baseline to glyph TOP.
//!                  Negative = glyph is above baseline (the common case).
//!                  In screen coords (y down): glyph_top = baseline + bearing_y
//!   0x18  u32     bitmap width (== bitmap header width)
//!   0x1C  u32     bitmap height (== bitmap header height)
//!   0x20  u32     bitmap_offset (file-absolute)
//!   0x24  u32     bitmap_size  (bytes incl. per-glyph header)
//!   0x28  u8[24]  reserved
//!
//! Bitmap (bitmap_size bytes at bitmap_offset):
//!   0x00  u32     data_size (bytes, excluding this 16-byte header)
//!   0x04  u32     bitmap_width  (px)
//!   0x08  u32     bitmap_height (px)
//!   0x0C  u32     reserved
//!   0x10  u8[]    RLE: each byte is one opcode, processed row-major:
//!                   0x80 alone (high bit, count 0) → skip one full
//!                     row of `bitmap_width` transparent pixels. Used
//!                     to compactly encode the empty middle row of a
//!                     `:` and similar sparse glyphs.
//!                   any other byte with high bit set → emit
//!                     (byte & 0x7F) opaque pixels.
//!                   high bit clear → skip `byte` transparent pixels.
//!                 The walk ends when width × height pixel positions
//!                 have been visited.
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
    /// Extra inter-letter spacing added to every glyph's advance.
    /// The plain (non-`_SMOOTH`) AFT variants have `advance == bitmap
    /// width` for most glyphs, so consecutive letters touch. The
    /// shipped game uses the `_SMOOTH` variants which carry +1..+4 px
    /// of side-bearing baked into their wider advance — but their
    /// bitmap RLE is a different encoding the loader doesn't decode
    /// yet. As a stop-gap we add the missing letter spacing here so
    /// labels using plain VERDANA still read clearly.
    pub extra_advance: u32,
    /// Codepoint → metrics + raw bitmap bytes (RLE).
    glyphs: HashMap<u32, AftGlyph>,
}

#[derive(Clone)]
struct AftGlyph {
    width: u32,
    height: u32,
    bearing_y: i32,
    advance: u32,
    /// Decoded 8bpp alpha bitmap, width*height bytes.
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
            let advance = u32_at(bytes, o + 0x08)?;
            let bearing_y = i32_at(bytes, o + 0x14)?;
            let bmp_off = u32_at(bytes, o + 0x20)? as usize;
            let bmp_sz = u32_at(bytes, o + 0x24)? as usize;
            // Whitespace glyphs (e.g. space) have bmp_off == 0 and
            // carry only an advance value. Cache as zero-pixel glyphs.
            if bmp_off == 0 || bmp_sz < 16 || bmp_off + bmp_sz > bytes.len() {
                glyphs.insert(
                    code,
                    AftGlyph {
                        width: 0,
                        height: 0,
                        bearing_y,
                        advance,
                        pixels: Vec::new(),
                    },
                );
                continue;
            }
            let bw = u32_at(bytes, bmp_off + 4)?;
            let bh = u32_at(bytes, bmp_off + 8)?;
            let rle = &bytes[bmp_off + 16..bmp_off + bmp_sz];
            let pixels = decode_rle(rle, bw, bh);
            glyphs.insert(
                code,
                AftGlyph {
                    width: bw,
                    height: bh,
                    bearing_y,
                    advance,
                    pixels,
                },
            );
        }
        Ok(Self {
            line_height,
            ascent,
            extra_advance: 0,
            glyphs,
        })
    }

    fn glyph(&self, codepoint: u32) -> Option<&AftGlyph> {
        self.glyphs
            .get(&codepoint)
            .or_else(|| self.glyphs.get(&('?' as u32)))
    }

    /// Pixel advance of `text` at this font's native size, including
    /// any `extra_advance` letter-spacing nudge.
    pub fn measure(&self, text: &str) -> u32 {
        text.chars()
            .map(|c| {
                self.glyph(c as u32)
                    .map(|g| g.advance + self.extra_advance)
                    .unwrap_or(0)
            })
            .sum()
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

/// Decode the per-glyph RLE stream into an 8bpp alpha bitmap of
/// `w × h` pixels (255 = opaque, 0 = transparent). Opcodes are
/// row-major:
///   - `0x80` alone (high bit, count 0) → skip one full row
///     (`w` transparent pixels). Without this rule, sparse glyphs
///     like `:` and `;` collapse — they encode their inter-dot gap
///     with `0x80`, which a plain "high-bit = N opaque" decoder
///     would treat as a no-op.
///   - other high-bit bytes → emit `(byte & 0x7F)` opaque pixels.
///   - low-bit bytes → skip `byte` transparent pixels.
fn decode_rle(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let total = (w * h) as usize;
    let mut out = vec![0u8; total];
    let mut pos = 0usize;
    let row_w = w as usize;
    for &b in data {
        if b == 0x80 {
            // Skip a full row of transparent pixels.
            pos += row_w;
        } else if b & 0x80 != 0 {
            let n = (b & 0x7F) as usize;
            for _ in 0..n {
                if pos >= total {
                    return out;
                }
                out[pos] = 255;
                pos += 1;
            }
        } else {
            pos += b as usize;
        }
        if pos >= total {
            break;
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
const FONT_RANGER_6: &[u8] = include_bytes!("../../../assets/fonts/RANGER_6.AFT");
const FONT_RANGER_5: &[u8] = include_bytes!("../../../assets/fonts/RANGER_5.AFT");
const FONT_VERDANA_10_2: &[u8] = include_bytes!("../../../assets/fonts/VERDANA_10_2.AFT");
const FONT_VERDANA_09_2: &[u8] = include_bytes!("../../../assets/fonts/VERDANA_09_2.AFT");
const FONT_VERDANA_08_1: &[u8] = include_bytes!("../../../assets/fonts/VERDANA_08_1.AFT");
const FONT_VERDANA_07_1: &[u8] = include_bytes!("../../../assets/fonts/VERDANA_07_1.AFT");
#[allow(dead_code)] // kept for back-pocket if Font.2Mini ever needs the smallest size
const FONT_VERDANA_06_1: &[u8] = include_bytes!("../../../assets/fonts/VERDANA_06_1.AFT");

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
        // Each entry: (alias, AFT bytes, extra letter-spacing px).
        // RANGER fonts have a 1-px gap baked into their advance, so 0.
        // Plain VERDANA glyphs have advance == bitmap width (no gap),
        // so we add +1 px between each pair to approximate the
        // SMOOTH variants the original game uses.
        //
        // Sizes are bumped one notch over the obvious match (Small →
        // VERDANA_08, Normal → VERDANA_10) — based on user feedback
        // the lowest sizes look too small at the design-space scale
        // because the original ships SMOOTH variants with wider per-
        // glyph cells, and bumping size compensates for that until
        // the SMOOTH RLE decoder lands.
        for (name, bytes, extra) in [
            ("Font.2Ranger", FONT_RANGER_6, 0),
            ("Font.2Small", FONT_VERDANA_08_1, 1),
            ("Font.2Normal", FONT_VERDANA_10_2, 1),
            ("Font.2Mini", FONT_VERDANA_07_1, 1),
            ("RANGER_5", FONT_RANGER_5, 0),
            ("RANGER_6", FONT_RANGER_6, 0),
            // Optional aliases the data could reference.
            ("Font.2Big", FONT_VERDANA_10_2, 1),
            ("Font.2Bold", FONT_VERDANA_09_2, 1),
        ] {
            match AftFont::parse(bytes) {
                Ok(mut f) => {
                    f.extra_advance = extra;
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
        let extra = font_ref.extra_advance;
        let g = font_ref.glyph(codepoint)?.clone();
        // Whitespace / zero-pixel glyph — cache metrics only, no atlas slot.
        if g.width == 0 || g.height == 0 {
            let r = GlyphRect {
                atlas_x: 0,
                atlas_y: 0,
                w: 0,
                h: 0,
                bearing_y: g.bearing_y,
                advance: g.advance + extra,
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
            bearing_y: g.bearing_y,
            advance: g.advance + extra,
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
        let f = AftFont::parse(FONT_VERDANA_07_1).unwrap();
        let g = f.glyph(b':' as u32).unwrap();
        // Column-of-1 layout — count opaque pixels (top dot block +
        // bottom dot block) and verify there's at least one
        // transparent row in the middle.
        assert_eq!(g.width, 1);
        let any_gap = g
            .pixels
            .windows(1)
            .enumerate()
            .any(|(i, w)| w[0] == 0 && i > 0 && i < g.pixels.len() - 1);
        assert!(any_gap, "colon must have a transparent gap row: {:?}", g.pixels);
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
