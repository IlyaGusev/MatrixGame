//! Port of `CMatrixHint` (Interface/MatrixHint.{cpp,hpp}) — the
//! tooltip/hint system with 9-slice border chrome + inline resource
//! icons + positioning rules (`CorrectCoordinates`).
//!
//! ## What's ported (faithfully)
//!
//! * `TemplateLibrary::load` — reads `da/Templates` (MatrixHint.cpp:
//!   519-563): walks the BlockPar, concatenating `|`-continuation
//!   params with the same name into a single template string.
//! * `HintReplacer::from_storage` — reads both `da/Replaces`
//!   (MatrixGame.cpp:262) and `da/AllLabels/Replaces`
//!   (MatrixGame.cpp:266) into a single `[key] → value` map. Turret
//!   labels cached separately (MatrixHint + CInterface.cpp:4465-4509).
//! * `HintChromeLibrary` — reads `da/Hints/0` (9-slice rects + edge
//!   insets) and `da/Hints/Bitmaps` (name → PNG path for resource
//!   icons and the other inline hint images).
//! * `build_hint` — ports the Pass 1 layout loop at MatrixHint.cpp:
//!   42-149: walks `|`-separated directives, emits `HintPart`s with
//!   their `HEM_*` modifier, threads cx/cy/cw/ch the same way the
//!   original does.
//! * `apply_positions` — ports Pass 2 at MatrixHint.cpp:270-296:
//!   resolves the `-1000/-1001/-1002` sentinels (centre / centre-right
//!   / centre-left) once the total content width is known, shifts
//!   everything by `ots.left/top` so the first part lands inside the
//!   border.
//! * `[key]` substitution in `_TEXT:` / `_IF:` — ports the `Replace`
//!   helper at MatrixHint.cpp:486-514 (`[]` → baserepl; `[name]` →
//!   Replaces value; empty on miss). Iterates to a fixed cap so
//!   chained substitutions (`_TEXT:[titan_hint]` → text containing
//!   `[_titan_income]`) resolve to depth.
//! * `CorrectCoordinates` — ports CInterface.cpp:4356-4437 exactly:
//!   pins Main-panel hints to the panel's top-right (
//!   `m_MainX+554-width`, `m_MainY+28-height`), centres Base-panel
//!   history/counter hints above the element, then clamps to the
//!   screen with a `HINT_OTSTUP=5` margin.
//!
//! ## What's deferred
//!
//! * Sound (`_SOUNDIN:` / `_SOUNDOUT:`) — UI sound backend is stubbed.
//! * `_TEXTP:` modifiers (`L`/`B`/`C`/`CR`/`CL`/`COPY`/`T:N`) — we
//!   collapse these onto the default text layout. Resource-bar hints
//!   (Top panel) don't use them; turret hints use `L`/`B` which the
//!   original maps to `HEM_LAST_ON_LINE` / `HEM_BITMAP`.
//! * Custom `_BORDER:` override — we honour the border's own
//!   insets (TopDown / BottomDown / LeftDown / RightDown) and skip the
//!   per-template override (only used by the Begin/End templates we
//!   don't render yet).

use std::collections::HashMap;

use crate::matrix_lib::base::storage::Storage;

/// Milliseconds of continuous hover before a hint appears. The C++
/// doesn't have an explicit constant; 700 ms matches the feel of the
/// shipped game (user-confirmed approximate).
pub const HOVER_SHOW_MS: f32 = 700.0;

/// `PAR_TEMPLATES` (StringConstants.hpp:244).
pub const PAR_TEMPLATES: &str = "Templates";
/// `PAR_REPLACE` (StringConstants.hpp:245).
pub const PAR_REPLACE: &str = "Replaces";
/// `IF_LABELS_BLOCKPAR` (StringConstants.hpp:677).
pub const PAR_ALL_LABELS: &str = "AllLabels";
/// `PAR_SOURCE_HINTS` (StringConstants.hpp:336).
pub const PAR_SOURCE_HINTS: &str = "Hints";
/// `HINT_OTSTUP` margin used by `CorrectCoordinates` (CInterface.cpp:
/// 4404-4428). The original uses 5 px.
pub const HINT_OTSTUP: i32 = 5;

// ── Template / replacement plumbing ────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct TemplateLibrary {
    templates: HashMap<String, String>,
}

impl TemplateLibrary {
    pub fn load(storage: &Storage) -> Self {
        let mut templates: HashMap<String, String> = HashMap::new();
        let Some(rec) = storage.block_record("da", PAR_TEMPLATES) else {
            log::warn!("hint: no 'da/{PAR_TEMPLATES}' child — hints disabled");
            return Self { templates };
        };
        let Some(keys) = storage.get_buf(&rec, "0") else {
            return Self { templates };
        };
        let Some(values) = storage.get_buf(&rec, "1") else {
            return Self { templates };
        };
        let n = keys.arrays_count().min(values.arrays_count());
        let mut last_name: Option<String> = None;
        for i in 0..n {
            let name = keys.get_as_wstr(i);
            let value = values.get_as_wstr(i);
            let is_continuation =
                value.starts_with('|') && last_name.as_deref() == Some(name.as_str());
            if is_continuation {
                if let Some(entry) = templates.get_mut(&name) {
                    entry.push_str(&value);
                }
            } else {
                templates.entry(name.clone()).or_insert(value);
            }
            last_name = Some(name);
        }
        log::info!("hint: loaded {} templates", templates.len());
        Self { templates }
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.templates.get(name).map(String::as_str)
    }
}

#[derive(Debug, Default, Clone)]
pub struct HintReplacer {
    values: HashMap<String, String>,
    baserepl: String,
    turret_labels: [[String; 2]; 4],
}

impl HintReplacer {
    pub fn from_storage(storage: &Storage) -> Self {
        let mut values: HashMap<String, String> = HashMap::new();
        let top = storage.block_record("da", PAR_REPLACE);
        let nested = storage
            .block_record("da", PAR_ALL_LABELS)
            .and_then(|all| storage.block_record(&all, PAR_REPLACE));
        for rec_opt in [top.as_deref(), nested.as_deref()] {
            let Some(rec) = rec_opt else { continue };
            let Some(keys) = storage.get_buf(rec, "0") else { continue };
            let Some(vals) = storage.get_buf(rec, "1") else { continue };
            let n = keys.arrays_count().min(vals.arrays_count());
            for i in 0..n {
                values.entry(keys.get_as_wstr(i)).or_insert(vals.get_as_wstr(i));
            }
        }
        let mut turret_labels: [[String; 2]; 4] = Default::default();
        if let Some(all) = storage.block_record("da", PAR_ALL_LABELS) {
            if let Some(turrets) = storage.block_record(&all, "Turrets") {
                for i in 0..4 {
                    turret_labels[i][0] = storage
                        .block_param(&turrets, &format!("Tur{}_Name", i + 1))
                        .unwrap_or_default();
                    turret_labels[i][1] = storage
                        .block_param(&turrets, &format!("Tur{}_Range", i + 1))
                        .unwrap_or_default();
                }
            }
        }
        log::info!(
            "hint: loaded {} replacements (top={}, nested={})",
            values.len(),
            top.is_some(),
            nested.is_some()
        );
        Self {
            values,
            baserepl: String::new(),
            turret_labels,
        }
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.values.insert(key.into(), value.into());
    }
    pub fn set_baserepl(&mut self, name: impl Into<String>) {
        self.baserepl = name.into();
    }
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
    pub fn baserepl(&self) -> &str {
        &self.baserepl
    }
    pub fn turret_label(&self, idx: usize) -> (&str, &str) {
        let i = idx.min(3);
        (
            self.turret_labels[i][0].as_str(),
            self.turret_labels[i][1].as_str(),
        )
    }
}

// ── Hint chrome (9-slice border + bitmap aliases) ──────────────────

/// One slice of the 9-part border texture. Matches the SBordPart
/// struct at MatrixHint.cpp:151-159.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChromeSlice {
    /// Subrect inside the border atlas (texture pixels).
    pub tex_x: i32,
    pub tex_y: i32,
    pub w: i32,
    pub h: i32,
}

/// The 9-slice parts in MatrixHint.cpp ordering:
/// 0 LeftTop, 1 Top, 2 RightTop, 3 Left, 4 Center, 5 Right,
/// 6 LeftBottom, 7 Bottom, 8 RightBottom.
pub const SLICE_LT: usize = 0;
pub const SLICE_T: usize = 1;
pub const SLICE_RT: usize = 2;
pub const SLICE_L: usize = 3;
pub const SLICE_C: usize = 4;
pub const SLICE_R: usize = 5;
pub const SLICE_LB: usize = 6;
pub const SLICE_B: usize = 7;
pub const SLICE_RB: usize = 8;

#[derive(Debug, Default, Clone)]
pub struct ChromeBorder {
    /// Source atlas path (e.g. `Matrix/Textures/Hints/border0`).
    pub source_path: String,
    pub slices: [ChromeSlice; 9],
    pub top_down: i32,
    pub bottom_down: i32,
    pub left_down: i32,
    pub right_down: i32,
}

impl ChromeBorder {
    /// `ots = [left, top, right, bottom]` — the content-area insets
    /// derived from the border's `*_Down` values + slice sizes. Port
    /// of MatrixHint.cpp:222-264 "otstup" block (taken when the
    /// per-template `_BORDER:` override is absent, which is the
    /// common case).
    pub fn otstup(&self) -> [i32; 4] {
        let mut ots = [0i32; 4];
        let lt = self.slices[SLICE_LT];
        let rt = self.slices[SLICE_RT];
        let lb = self.slices[SLICE_LB];
        let rb = self.slices[SLICE_RB];
        let left = self.slices[SLICE_L];
        let top = self.slices[SLICE_T];
        let right = self.slices[SLICE_R];
        let bottom = self.slices[SLICE_B];
        if lt.w > 0 {
            ots[0] = lt.w - self.left_down;
            ots[1] = lt.h - self.top_down;
        }
        if rt.w > 0 {
            ots[2] = rt.w - self.right_down;
            ots[1] = rt.h - self.top_down;
        }
        if lb.w > 0 {
            ots[0] = lb.w - self.left_down;
            ots[3] = lb.h - self.bottom_down;
        }
        if rb.w > 0 {
            ots[2] = rb.w - self.right_down;
            ots[3] = rb.h - self.bottom_down;
        }
        if top.h > 0 {
            ots[1] = top.h - self.top_down;
        }
        if left.w > 0 {
            ots[0] = left.w - self.left_down;
        }
        if right.w > 0 {
            ots[2] = right.w - self.right_down;
        }
        if bottom.h > 0 {
            ots[3] = bottom.h - self.bottom_down;
        }
        ots
    }
}

/// Inline bitmap (resource icon, face portrait, etc.). `da/Hints/
/// Bitmaps/{name}` → the PNG path.
#[derive(Debug, Clone, Default)]
pub struct ChromeBitmap {
    pub path: String,
}

#[derive(Debug, Default, Clone)]
pub struct HintChromeLibrary {
    /// Border id → 9-slice border def. Border 0 is the default used
    /// by every resource / button hint (`0|_COLOR:…|_TEXT:…`).
    pub borders: HashMap<i32, ChromeBorder>,
    /// Alias → bitmap. `res_titan` / `res_energy` / `face_N` / …
    pub bitmaps: HashMap<String, ChromeBitmap>,
}

impl HintChromeLibrary {
    pub fn load(storage: &Storage) -> Self {
        let mut out = Self::default();
        let Some(hints_rec) = storage.block_record("da", PAR_SOURCE_HINTS) else {
            log::warn!("hint: no 'da/{PAR_SOURCE_HINTS}' block — chrome disabled");
            return out;
        };
        // Enumerate child records — each integer-named child is one
        // border-id (usually just `0`), `Bitmaps` holds the alias
        // table.
        let Some(names) = storage.get_buf(&hints_rec, "2") else { return out };
        let Some(recs) = storage.get_buf(&hints_rec, "3") else { return out };
        let n = names.arrays_count().min(recs.arrays_count());
        for i in 0..n {
            let name = names.get_as_wstr(i);
            let rec = recs.get_as_wstr(i);
            if name == "Bitmaps" {
                let Some(k) = storage.get_buf(&rec, "0") else { continue };
                let Some(v) = storage.get_buf(&rec, "1") else { continue };
                for j in 0..k.arrays_count().min(v.arrays_count()) {
                    out.bitmaps.insert(
                        k.get_as_wstr(j),
                        ChromeBitmap {
                            path: v.get_as_wstr(j),
                        },
                    );
                }
            } else if let Ok(id) = name.parse::<i32>() {
                out.borders.insert(id, parse_border(storage, &rec));
            }
        }
        log::info!(
            "hint: loaded {} borders + {} bitmap aliases",
            out.borders.len(),
            out.bitmaps.len()
        );
        out
    }
}

fn parse_border(stor: &Storage, rec: &str) -> ChromeBorder {
    let mut b = ChromeBorder::default();
    b.source_path = stor.block_param(rec, "Source").unwrap_or_default();
    let slice = |name: &str| -> ChromeSlice {
        let Some(raw) = stor.block_param(rec, name) else {
            return ChromeSlice::default();
        };
        let parts: Vec<i32> = raw.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        ChromeSlice {
            tex_x: parts.first().copied().unwrap_or(0),
            tex_y: parts.get(1).copied().unwrap_or(0),
            w: parts.get(2).copied().unwrap_or(0),
            h: parts.get(3).copied().unwrap_or(0),
        }
    };
    b.slices[SLICE_LT] = slice("LeftTop");
    b.slices[SLICE_T] = slice("Top");
    b.slices[SLICE_RT] = slice("RightTop");
    b.slices[SLICE_L] = slice("Left");
    b.slices[SLICE_C] = slice("Center");
    b.slices[SLICE_R] = slice("Right");
    b.slices[SLICE_LB] = slice("LeftBottom");
    b.slices[SLICE_B] = slice("Bottom");
    b.slices[SLICE_RB] = slice("RightBottom");
    let i = |k: &str| -> i32 {
        stor.block_param(rec, k)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    };
    b.top_down = i("TopDown");
    b.bottom_down = i("BottomDown");
    b.left_down = i("LeftDown");
    b.right_down = i("RightDown");
    b
}

// ── Parsed hint ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum HintPart {
    /// One run of text at `(x, y)`. Width is laid out from the AFT
    /// glyph advances; multi-line text already got split on `<br>`.
    Text {
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        text: String,
        font: String,
        color: [u8; 4],
    },
    /// Named inline image (resolves via `HintChromeLibrary::bitmaps`).
    Bitmap {
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        name: String,
    },
}

/// Fully-resolved hint ready for the renderer.
#[derive(Debug, Clone)]
pub struct Hint {
    pub parts: Vec<HintPart>,
    /// `cw + ots.left + ots.right` — full bitmap width including
    /// border (MatrixHint.cpp:302).
    pub total_w: i32,
    /// `ch + ots.top + ots.bottom` — full bitmap height.
    pub total_h: i32,
    /// Border id (usually 0). Resolved against `HintChromeLibrary`.
    pub border_id: i32,
    pub otstup: [i32; 4],
    pub screen_x: f32,
    pub screen_y: f32,
}

/// One quad to emit as part of the 9-slice border or a tile repeat.
/// Produced by [`Hint::slice_draws`]; the renderer just converts this
/// into two triangles sampling `tex_rect` from the border atlas.
#[derive(Debug, Clone, Copy)]
pub struct SliceDraw {
    /// Source subrect inside the border atlas (texture pixels).
    pub tex_x: i32,
    pub tex_y: i32,
    pub tex_w: i32,
    pub tex_h: i32,
    /// Destination rect in screen pixels, already scaled by the
    /// design→screen factor.
    pub screen_x: f32,
    pub screen_y: f32,
    pub screen_w: f32,
    pub screen_h: f32,
}

impl Hint {
    /// Port of the `bmpf.Copy` / `bmpf.Tile` calls at MatrixHint.cpp:
    /// 324-382: produce the 9-slice border quads for this hint.
    ///
    /// Corners (LT / RT / LB / RB) use `Copy` (1:1 stamp) in the
    /// original; edges (T / B / L / R) and the C centre use `Tile`,
    /// repeating the tile until the destination span is covered.
    /// We emit one `SliceDraw` per full tile step — wgpu can't
    /// `D3DTADDRESS_WRAP` arbitrary atlas subrects, so tiling has to
    /// be expressed as many clipped quads instead of one with wrap UVs.
    ///
    /// Returns an empty vec when the border is absent (e.g. during
    /// startup before `HintChromeLibrary::load` has run).
    pub fn slice_draws(&self, chrome: &HintChromeLibrary, scale: f32) -> Vec<SliceDraw> {
        let Some(border) = chrome.borders.get(&self.border_id) else {
            return Vec::new();
        };
        let fw = self.total_w as f32 * scale;
        let fh = self.total_h as f32 * scale;
        let x0 = self.screen_x;
        let y0 = self.screen_y;
        let s = &border.slices;

        let mut out: Vec<SliceDraw> = Vec::new();
        let push_copy = |out: &mut Vec<SliceDraw>, px: f32, py: f32, slice: &ChromeSlice| {
            if slice.w <= 0 || slice.h <= 0 {
                return;
            }
            out.push(SliceDraw {
                tex_x: slice.tex_x,
                tex_y: slice.tex_y,
                tex_w: slice.w,
                tex_h: slice.h,
                screen_x: px,
                screen_y: py,
                screen_w: slice.w as f32 * scale,
                screen_h: slice.h as f32 * scale,
            });
        };
        let push_tile =
            |out: &mut Vec<SliceDraw>,
             dst_x: f32,
             dst_y: f32,
             dst_w: f32,
             dst_h: f32,
             slice: &ChromeSlice| {
                if slice.w <= 0 || slice.h <= 0 || dst_w <= 0.0 || dst_h <= 0.0 {
                    return;
                }
                let tile_w = slice.w as f32 * scale;
                let tile_h = slice.h as f32 * scale;
                if tile_w <= 0.0 || tile_h <= 0.0 {
                    return;
                }
                let mut y = dst_y;
                let y_end = dst_y + dst_h;
                while y < y_end - 0.01 {
                    let row_h = tile_h.min(y_end - y);
                    let sh_src = ((row_h / scale).round() as i32).max(1).min(slice.h);
                    let mut x = dst_x;
                    let x_end = dst_x + dst_w;
                    while x < x_end - 0.01 {
                        let col_w = tile_w.min(x_end - x);
                        let sw_src = ((col_w / scale).round() as i32).max(1).min(slice.w);
                        out.push(SliceDraw {
                            tex_x: slice.tex_x,
                            tex_y: slice.tex_y,
                            tex_w: sw_src,
                            tex_h: sh_src,
                            screen_x: x,
                            screen_y: y,
                            screen_w: col_w,
                            screen_h: row_h,
                        });
                        x += tile_w;
                    }
                    y += tile_h;
                }
            };

        // Corners (MatrixHint.cpp:326-348 — four `bmpf.Copy` calls).
        let lt = &s[SLICE_LT];
        let rt = &s[SLICE_RT];
        let lb = &s[SLICE_LB];
        let rb = &s[SLICE_RB];
        push_copy(&mut out, x0, y0, lt);
        push_copy(&mut out, x0 + fw - rt.w as f32 * scale, y0, rt);
        push_copy(&mut out, x0, y0 + fh - lb.h as f32 * scale, lb);
        push_copy(
            &mut out,
            x0 + fw - rb.w as f32 * scale,
            y0 + fh - rb.h as f32 * scale,
            rb,
        );

        // Top / bottom edges (MatrixHint.cpp:350-360).
        let lt_w = lt.w as f32 * scale;
        let rt_w = rt.w as f32 * scale;
        let lb_w = lb.w as f32 * scale;
        let rb_w = rb.w as f32 * scale;
        let t_h = s[SLICE_T].h as f32 * scale;
        let b_h = s[SLICE_B].h as f32 * scale;
        let l_w = s[SLICE_L].w as f32 * scale;
        let r_w = s[SLICE_R].w as f32 * scale;
        push_tile(&mut out, x0 + lt_w, y0, fw - lt_w - rt_w, t_h, &s[SLICE_T]);
        push_tile(
            &mut out,
            x0 + lb_w,
            y0 + fh - b_h,
            fw - lb_w - rb_w,
            b_h,
            &s[SLICE_B],
        );

        // Left / right / centre (MatrixHint.cpp:362-380).
        let mid_y = y0 + t_h;
        let mid_h = fh - t_h - b_h;
        if mid_h > 0.0 {
            push_tile(&mut out, x0, mid_y, l_w, mid_h, &s[SLICE_L]);
            push_tile(&mut out, x0 + fw - r_w, mid_y, r_w, mid_h, &s[SLICE_R]);
            push_tile(
                &mut out,
                x0 + l_w,
                mid_y,
                fw - l_w - r_w,
                mid_h,
                &s[SLICE_C],
            );
        }
        out
    }

    /// Source atlas path for this hint's border chrome. Returns
    /// `None` if the border wasn't loaded.
    pub fn border_source_path<'a>(&self, chrome: &'a HintChromeLibrary) -> Option<&'a str> {
        chrome.borders.get(&self.border_id).map(|b| b.source_path.as_str())
    }
}

// ── Layout ──────────────────────────────────────────────────────────

/// Byte opcode tags for the modifier bundle passed to each part at
/// parse time. Maps directly onto the C++ `EHintElementModificator`
/// enum values that the layout loop at MatrixHint.cpp:52-149 reads.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum Hem {
    /// `HEM_BITMAP` — default: emit at `(cx, cy)` and advance cx by w.
    Bitmap,
    /// `HEM_LAST_ON_LINE` — emit, advance cx, then newline.
    LastOnLine,
    /// `HEM_CENTER` — emit centered; newline after.
    Center,
    /// `HEM_COORD` — absolute coord; next Bitmap uses stored coord.
    Coord { x: i32, y: i32 },
    /// `HEM_DOWN` — cy += N, no emit.
    Down { dy: i32 },
    /// `HEM_RIGHT` — cx += N, no emit.
    Right { dx: i32 },
    /// `HEM_TAB_*` — place at absolute X = n (w/o bumping cx start).
    Tab { x: i32 },
    /// `HEM_CENTER_RIGHT_*` / `HEM_CENTER_LEFT_*` — sentinels resolved
    /// in Pass 2 using total width.
    CenterRight { n: i32 },
    CenterLeft { n: i32 },
}

/// Parse one template into a laid-out `Hint`. Ports the two-pass
/// algorithm in MatrixHint.cpp:
///   Pass 1 (42-149): walk directives, emit `(HintPart, hem)` with
///     cx/cy tracking + collect parts' sentinel positions.
///   Pass 2 (270-296): resolve `-1000`/`-1001`/`-1002` sentinels,
///     shift everything by `(ots.left, ots.top)`, bump `ch` to fit
///     top+bottom slice heights.
pub fn build_hint(
    template: &str,
    border: &ChromeBorder,
    bitmaps: &HashMap<String, ChromeBitmap>,
    replacer: &HintReplacer,
    atlas: &mut super::text::GlyphAtlas,
    bitmap_sizer: impl Fn(&str) -> Option<(i32, i32)>,
) -> Option<(Vec<HintPart>, [i32; 4], i32, i32, i32, i32)> {
    let parts_iter: Vec<&str> = template.split('|').collect();
    if parts_iter.is_empty() {
        return None;
    }
    // First field is the border id (MatrixHint.cpp:579). Default 0.
    let border_id: i32 = parts_iter.first().and_then(|s| s.trim().parse().ok()).unwrap_or(0);

    // Layout state (mirrors MatrixHint.cpp:580-591).
    let mut font = "Font.2Normal".to_string();
    let mut color: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
    let mut modif: Hem = Hem::Bitmap;
    let mut h_width: i32 = 0;
    let mut _h_height: i32 = 0;
    let mut skip = false;

    // Raw (part, hem) stream — positions resolved in pass 2.
    struct Emitted {
        part: HintPart,
        hem: Hem,
    }
    let mut emitted: Vec<Emitted> = Vec::new();
    let mut cx: i32 = 0;
    let mut cy: i32 = 0;
    let mut cw: i32 = 0;
    let mut ch: i32 = 0;

    // `new_coord_f`/`new_coord` — when set by `HEM_COORD`, the next
    // bitmap is placed at the absolute coord instead of (cx, cy).
    let mut new_coord: Option<(i32, i32)> = None;

    for (idx, raw) in parts_iter.iter().enumerate() {
        if idx == 0 {
            continue; // border id
        }
        let directive = *raw;
        if directive.is_empty() {
            continue;
        }
        // _IF:/_ENDIF guard (MatrixHint.cpp:600-610).
        if directive == "_ENDIF" || directive.starts_with("_ENDIF") {
            skip = false;
            continue;
        }
        if let Some(rest) = directive.strip_prefix("_IF:") {
            let resolved = substitute(rest, replacer);
            skip = resolved.trim().is_empty();
            continue;
        }
        if skip {
            continue;
        }
        // Scalar state updates.
        if let Some(rest) = directive.strip_prefix("_FONT:") {
            font = rest.trim().to_string();
            continue;
        }
        if let Some(rest) = directive.strip_prefix("_COLOR:") {
            color = parse_argb(rest.trim());
            continue;
        }
        if let Some(rest) = directive.strip_prefix("_WIDTH:") {
            h_width = rest.trim().parse().unwrap_or(0);
            continue;
        }
        if let Some(rest) = directive.strip_prefix("_HEIGHT:") {
            _h_height = rest.trim().parse().unwrap_or(0);
            continue;
        }
        if directive.starts_with("_ALIGN:") || directive.starts_with("_SOUNDIN:")
            || directive.starts_with("_SOUNDOUT:") || directive.starts_with("_BORDER:")
        {
            continue; // scalar-ish; we don't use them for text hints.
        }
        // _POS:X:Y → set absolute coord for next bitmap.
        if let Some(rest) = directive.strip_prefix("_POS:") {
            let p: Vec<i32> = rest.split(':').filter_map(|s| s.parse().ok()).collect();
            if p.len() >= 2 {
                new_coord = Some((p[0], p[1]));
            }
            continue;
        }
        if let Some(rest) = directive.strip_prefix("_DOWN:") {
            let dy: i32 = rest.parse().unwrap_or(0);
            cy += dy;
            if cy > ch {
                ch = cy;
            }
            continue;
        }
        if let Some(rest) = directive.strip_prefix("_RIGHT:") {
            let dx: i32 = rest.parse().unwrap_or(0);
            cx += dx;
            if cx > cw {
                cw = cx;
            }
            continue;
        }
        // _MOD:M — update the modifier for the next bitmap-emitting
        // directive. `L`/`C`/`T:N`/`CR:N`/`CL:N`/`COPY`.
        if let Some(rest) = directive.strip_prefix("_MOD:") {
            modif = parse_mod(rest);
            continue;
        }
        if let Some(rest) = directive.strip_prefix("_TEXTP:") {
            modif = parse_mod(rest);
            continue;
        }
        // _BITMAP:name:modifier → resolve the alias to a PNG, look up
        // its size, emit with the directive-local modifier.
        if let Some(rest) = directive.strip_prefix("_BITMAP:") {
            let mut it = rest.split(':');
            let raw_name = it.next().unwrap_or("").trim().to_string();
            let resolved_name = substitute(&raw_name, replacer);
            let name_trim = resolved_name.trim().to_string();
            let m_spec = it.collect::<Vec<_>>().join(":");
            let part_mod = if m_spec.is_empty() {
                Hem::Bitmap
            } else {
                parse_mod(&m_spec)
            };
            if name_trim.is_empty() {
                continue;
            }
            let Some(bmp) = bitmaps.get(&name_trim) else {
                log::debug!("hint: unknown bitmap alias '{name_trim}'");
                continue;
            };
            let (bw, bh) = bitmap_sizer(&bmp.path).unwrap_or((16, 16));
            let (px, py) = pass1_position(
                part_mod,
                &mut cx,
                &mut cy,
                &mut cw,
                &mut ch,
                bw,
                bh,
                &mut new_coord,
            );
            emitted.push(Emitted {
                part: HintPart::Bitmap {
                    x: px,
                    y: py,
                    w: bw,
                    h: bh,
                    name: name_trim,
                },
                hem: part_mod,
            });
            continue;
        }
        // _TEXT:str — port of MatrixHint.cpp:693-731. `m_RangersText`
        // rasterises the ENTIRE text (with `<br>` → `\r\n`) into one
        // CBitmap whose `m_SizeX/m_SizeY` becomes the element's rect.
        // So we emit ONE `HintPart::Text` per `_TEXT:` directive — its
        // width is the longest wrapped line, its height is `N ×
        // line_h`. `modif` applies once (not once per line).
        if let Some(rest) = directive.strip_prefix("_TEXT:") {
            let resolved = substitute(rest, replacer).replace("<br>", "\n");
            // Split on explicit `\n` first, then greedy-wrap each
            // fragment at `h_width` when set (matches the `(w==0)?0:1`
            // wrap flag at MatrixHint.cpp:716).
            let mut lines: Vec<String> = Vec::new();
            for frag in resolved.split('\n') {
                if frag.is_empty() {
                    lines.push(String::new());
                    continue;
                }
                if h_width > 0 {
                    for sub in wrap_plain_text(atlas, &font, frag, h_width as u32) {
                        lines.push(sub);
                    }
                } else {
                    lines.push(frag.to_string());
                }
            }
            if lines.is_empty() {
                lines.push(String::new());
            }
            let line_h = atlas.line_height(&font) as i32;
            let max_w = lines
                .iter()
                .map(|s| measure_rich(atlas, &font, s) as i32)
                .max()
                .unwrap_or(0);
            let total_h = line_h * lines.len() as i32;
            let joined = lines.join("\n");
            let (px, py) = pass1_position(
                modif,
                &mut cx,
                &mut cy,
                &mut cw,
                &mut ch,
                max_w,
                total_h,
                &mut new_coord,
            );
            emitted.push(Emitted {
                part: HintPart::Text {
                    x: px,
                    y: py,
                    w: max_w,
                    h: total_h,
                    text: joined,
                    font: font.clone(),
                    color,
                },
                hem: modif,
            });
            continue;
        }
        // Unknown directive — ignore silently.
    }

    // Pass 2: resolve sentinels + apply padding shift. Matches
    // MatrixHint.cpp:270-296.
    let ots = border.otstup();
    // `ch` has to be large enough to fit top+bottom slice heights
    // minus the inset insets (MatrixHint.cpp:293-296).
    let min_ch = border.slices[SLICE_T].h + border.slices[SLICE_B].h - ots[1] - ots[3];
    if ch < min_ch {
        ch = min_ch.max(0);
    }
    let clw = cw;
    let clh = ch;
    let total_w = clw + ots[0] + ots[2];
    let total_h = clh + ots[1] + ots[3];

    let parts = emitted
        .into_iter()
        .map(|e| resolve_sentinel(e.part, e.hem, clw, ots[0], ots[1]))
        .collect();

    Some((parts, ots, total_w, total_h, border_id, h_width))
}

/// Pass 1: position a new part. `cx/cy/cw/ch` are borrowed-mut so we
/// advance them in place. Returns the part's `(x, y)` — which may be
/// a sentinel (`-1000`/`-1001`/`-1002`) when centre alignment can't
/// be resolved until total width is known.
fn pass1_position(
    hem: Hem,
    cx: &mut i32,
    cy: &mut i32,
    cw: &mut i32,
    ch: &mut i32,
    bx: i32,
    by: i32,
    new_coord: &mut Option<(i32, i32)>,
) -> (i32, i32) {
    match hem {
        Hem::Bitmap => {
            if let Some((px, py)) = new_coord.take() {
                (px, py)
            } else {
                let x = *cx;
                let y = *cy;
                *cx += bx;
                if *cx > *cw {
                    *cw = *cx;
                }
                if *cy + by > *ch {
                    *ch = *cy + by;
                }
                (x, y)
            }
        }
        Hem::LastOnLine => {
            let x = *cx;
            let y = *cy;
            *cx += bx;
            if *cx > *cw {
                *cw = *cx;
            }
            if *cy + by > *ch {
                *ch = *cy + by;
            }
            *cx = 0;
            *cy = *ch;
            (x, y)
        }
        Hem::Center => {
            let y = *cy;
            if bx > *cw {
                *cw = bx;
            }
            if *cy + by > *ch {
                *ch = *cy + by;
            }
            *cx = 0;
            *cy = *ch;
            (-1000, y) // sentinel — resolved in Pass 2
        }
        Hem::Tab { x: tabx } => {
            let y = *cy;
            *cx = tabx + bx;
            if *cx > *cw {
                *cw = *cx;
            }
            if *cy + by > *ch {
                *ch = *cy + by;
            }
            (tabx, y)
        }
        Hem::CenterRight { n } => {
            let y = *cy;
            if (n + bx) > (*cw / 2) {
                *cw = (n + bx) * 2;
            }
            *cx = *cw / 2 + n + bx;
            if *cy + by > *ch {
                *ch = *cy + by;
            }
            (-1001, y)
        }
        Hem::CenterLeft { n } => {
            let y = *cy;
            if (n + bx) > (*cw / 2) {
                *cw = (n + bx) * 2;
            }
            *cx = *cw / 2 - n;
            if *cy + by > *ch {
                *ch = *cy + by;
            }
            (-1002, y)
        }
        Hem::Coord { x, y } => (x, y),
        Hem::Down { .. } | Hem::Right { .. } => (0, 0),
    }
}

fn resolve_sentinel(part: HintPart, hem: Hem, cw: i32, ots_left: i32, ots_top: i32) -> HintPart {
    let (x0, y0, w0, h0) = match &part {
        HintPart::Text { x, y, w, h, .. } => (*x, *y, *w, *h),
        HintPart::Bitmap { x, y, w, h, .. } => (*x, *y, *w, *h),
    };
    let x = match x0 {
        -1000 => (cw - w0) / 2,
        -1001 => {
            let tv = match hem {
                Hem::CenterRight { n } => n,
                _ => 0,
            };
            cw / 2 + tv
        }
        -1002 => {
            let tv = match hem {
                Hem::CenterLeft { n } => n,
                _ => 0,
            };
            cw / 2 - tv - w0
        }
        v => v,
    };
    let x = x + ots_left;
    let y = y0 + ots_top;
    match part {
        HintPart::Text {
            text, font, color, ..
        } => HintPart::Text { x, y, w: w0, h: h0, text, font, color },
        HintPart::Bitmap { name, .. } => HintPart::Bitmap { x, y, w: w0, h: h0, name },
    }
}

fn parse_mod(spec: &str) -> Hem {
    let mut it = spec.split(':');
    let tag = it.next().unwrap_or("").trim();
    let arg: Option<i32> = it.next().and_then(|s| s.parse().ok());
    match tag {
        "C" => Hem::Center,
        "L" => Hem::LastOnLine,
        "CR" => Hem::CenterRight { n: arg.unwrap_or(0) },
        "CL" => Hem::CenterLeft { n: arg.unwrap_or(0) },
        "T" => Hem::Tab { x: arg.unwrap_or(0) },
        "B" | "" | "COPY" => Hem::Bitmap,
        _ => Hem::Bitmap,
    }
}

/// Visible-text width of `text` in `font`, with `<Color=...>` /
/// `</color>` tags stripped. The C++ pipes the full marked-up string
/// to `m_RangersText` which knows to skip tags when measuring; we
/// re-parse here so width accounting (wrap budgets, bitmap-after-text
/// placement) doesn't count tag bytes.
pub(super) fn measure_rich(
    atlas: &mut super::text::GlyphAtlas,
    font: &str,
    text: &str,
) -> u32 {
    super::text::parse_rich_text(text)
        .iter()
        .map(|run| atlas.measure(font, &run.text))
        .sum()
}

fn wrap_plain_text(
    atlas: &mut super::text::GlyphAtlas,
    font: &str,
    text: &str,
    wrap_px: u32,
) -> Vec<String> {
    if wrap_px == 0 {
        return vec![text.to_string()];
    }
    // Greedy wrap on space boundaries, but track an open `<Color=...>`
    // tag so each emitted line is self-contained — if we wrap inside
    // a coloured run, the next line re-opens the same colour and the
    // previous line gets its own `</color>`. Without this the second
    // half of a wrapped name would lose its colour (the close tag
    // landed on the last line, no open on the new one).
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut active_open: Option<String> = None;
    for word in text.split_inclusive(|c: char| c == ' ') {
        let tentative = format!("{current}{word}");
        if measure_rich(atlas, font, &tentative) <= wrap_px {
            current = tentative;
        } else {
            if !current.is_empty() {
                let mut s = current.trim_end().to_string();
                if active_open.is_some() {
                    s.push_str("</color>");
                }
                out.push(s);
            }
            let mut new_line = String::new();
            if let Some(t) = active_open.as_ref() {
                new_line.push_str(t);
            }
            new_line.push_str(word);
            current = new_line;
        }
        // Update the active-color tracker after committing the word.
        update_active_color(word, &mut active_open);
    }
    if !current.is_empty() {
        out.push(current.trim_end().to_string());
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Walk `frag` and update `active` to reflect the last open
/// `<Color=...>` tag (or `None` if a `</color>` closed it).
fn update_active_color(frag: &str, active: &mut Option<String>) {
    let bytes = frag.as_bytes();
    let mut i = 0;
    while i < frag.len() {
        if !frag.is_char_boundary(i) {
            i += 1;
            continue;
        }
        if bytes[i] == b'<' {
            let rest = &frag[i..];
            if rest.len() >= 7 && rest[..7].eq_ignore_ascii_case("<color=") {
                if let Some(end_off) = rest.find('>') {
                    *active = Some(rest[..=end_off].to_string());
                    i += end_off + 1;
                    continue;
                }
            }
            let close = "</color>";
            if rest.len() >= close.len() && rest[..close.len()].eq_ignore_ascii_case(close) {
                *active = None;
                i += close.len();
                continue;
            }
        }
        let next = (i + 1..=frag.len())
            .find(|&j| frag.is_char_boundary(j))
            .unwrap_or(frag.len());
        i = next;
    }
}

// ── Substitution ────────────────────────────────────────────────────

fn substitute(text: &str, replacer: &HintReplacer) -> String {
    let mut cur = text.to_string();
    for _ in 0..8 {
        let next = substitute_once(&cur, replacer);
        if next == cur {
            return next;
        }
        cur = next;
    }
    cur
}

fn substitute_once(text: &str, replacer: &HintReplacer) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(end_rel) = text[i + 1..].find(']') {
                let inner = &text[i + 1..i + 1 + end_rel];
                if inner.is_empty() {
                    out.push_str(replacer.baserepl());
                    i += 2 + end_rel;
                    continue;
                }
                if let Some(val) = replacer.get(inner) {
                    out.push_str(val);
                    i += 2 + end_rel;
                    continue;
                }
                // Unknown key → empty (matches `ParGetNE` at
                // MatrixHint.cpp:508).
                i += 2 + end_rel;
                continue;
            }
        }
        let ch = text[i..].chars().next().unwrap_or('\0');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn parse_argb(hex: &str) -> [u8; 4] {
    let hex = hex.trim_start_matches("0x").trim();
    let n = u32::from_str_radix(hex, 16).unwrap_or(0xFFFFFFFF);
    let a = ((n >> 24) & 0xFF) as u8;
    let r = ((n >> 16) & 0xFF) as u8;
    let g = ((n >> 8) & 0xFF) as u8;
    let b = (n & 0xFF) as u8;
    [r, g, b, a]
}

// ── Hover + display state ──────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct HintSystem {
    hovered: Option<HoverTarget>,
    timer_ms: f32,
    active_name: Option<String>,
    active: Option<Hint>,
}

#[derive(Debug, Clone)]
struct HoverTarget {
    panel_name: String,
    elem_name: String,
    template_name: String,
    offset_x: i32,
    offset_y: i32,
    elem_screen_x: f32,
    elem_screen_y: f32,
    elem_w_px: f32,
    elem_h_px: f32,
}

impl HintSystem {
    pub fn active(&self) -> Option<&Hint> {
        self.active.as_ref()
    }
    pub fn active_name(&self) -> Option<&str> {
        self.active_name.as_deref()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_hovered(
        &mut self,
        panel_name: Option<&str>,
        elem_name: Option<&str>,
        template_name: Option<&str>,
        offset_x: i32,
        offset_y: i32,
        elem_screen_x: f32,
        elem_screen_y: f32,
        elem_w_px: f32,
        elem_h_px: f32,
    ) {
        let target = match (panel_name, elem_name, template_name) {
            (Some(p), Some(e), Some(t)) if !t.is_empty() => Some(HoverTarget {
                panel_name: p.to_string(),
                elem_name: e.to_string(),
                template_name: t.to_string(),
                offset_x,
                offset_y,
                elem_screen_x,
                elem_screen_y,
                elem_w_px,
                elem_h_px,
            }),
            _ => None,
        };
        let same = matches!(
            (&self.hovered, &target),
            (Some(a), Some(b))
                if a.elem_name == b.elem_name && a.panel_name == b.panel_name
        );
        if !same {
            self.timer_ms = 0.0;
            self.active = None;
            self.active_name = None;
        }
        self.hovered = target;
    }

    pub fn clear(&mut self) {
        self.hovered = None;
        self.active = None;
        self.active_name = None;
        self.timer_ms = 0.0;
    }

    /// Per-frame tick. Builds the active hint once the timer passes
    /// the show threshold. `bitmap_sizer` resolves `(w, h)` for an
    /// inline `_BITMAP:` alias (uses the renderer's atlas cache).
    /// `panel_origins` gives `(Main.top_left, Base.top_left)` in
    /// screen px so `CorrectCoordinates` can pin hints to the right
    /// anchors.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        dt_ms: f32,
        templates: &TemplateLibrary,
        chrome: &HintChromeLibrary,
        replacer: &HintReplacer,
        atlas: &mut super::text::GlyphAtlas,
        bitmap_sizer: &dyn Fn(&str) -> Option<(i32, i32)>,
        screen_w: f32,
        screen_h: f32,
        scale: f32,
        main_origin: [f32; 2],
        base_origin: [f32; 2],
    ) {
        let Some(hov) = self.hovered.clone() else {
            return;
        };
        if self.active.is_some() {
            return;
        }
        self.timer_ms += dt_ms;
        if self.timer_ms < HOVER_SHOW_MS {
            return;
        }
        let Some(raw) = templates.get(&hov.template_name) else {
            log::debug!(
                "hint: template '{}' missing (elem '{}')",
                hov.template_name,
                hov.elem_name
            );
            self.active_name = Some(hov.elem_name.clone());
            return;
        };
        let mut local_repl = replacer.clone();
        local_repl.set_baserepl(&hov.elem_name);
        let border_default = ChromeBorder::default();
        let border = chrome.borders.get(&0).unwrap_or(&border_default);
        let Some((parts, ots, total_w_design, total_h_design, border_id, _wrap)) =
            build_hint(raw, border, &chrome.bitmaps, &local_repl, atlas, |path| {
                bitmap_sizer(path)
            })
        else {
            return;
        };
        if parts.is_empty() {
            self.active_name = Some(hov.elem_name.clone());
            return;
        }

        // Convert design px → screen px. The hint bitmap was laid out
        // in design space; the renderer scales it on draw. For
        // positioning we want to compute screen-space (x, y) in the
        // same space the other interface draws use.
        let total_w_px = total_w_design as f32 * scale;
        let total_h_px = total_h_design as f32 * scale;

        let elem_cx = hov.elem_screen_x + hov.elem_w_px * 0.5;
        let elem_bottom = hov.elem_screen_y + hov.elem_h_px;
        let desired_x = elem_cx + hov.offset_x as f32 * scale;
        let desired_y = elem_bottom + hov.offset_y as f32 * scale;
        // Port of `CorrectCoordinates` (CInterface.cpp:4356-4437).
        let Some((sx, sy)) = correct_coordinates(
            &hov.elem_name,
            desired_x,
            desired_y,
            total_w_px,
            total_h_px,
            screen_w,
            screen_h,
            scale,
            main_origin,
            base_origin,
        ) else {
            // "Hint doesn't fit on screen" — original aborts (return
            // false) in this case. Match that.
            self.active_name = Some(hov.elem_name.clone());
            return;
        };

        self.active = Some(Hint {
            parts,
            total_w: total_w_design,
            total_h: total_h_design,
            border_id,
            otstup: ots,
            screen_x: sx,
            screen_y: sy,
        });
        self.active_name = Some(hov.elem_name.clone());
    }
}

/// Port of `CIFaceList::CorrectCoordinates` at CInterface.cpp:
/// 4356-4437. Input / output are in screen pixels. The per-element
/// branches pin Main-panel hints to the panel's top-right corner
/// and centre Base-panel history buttons above the element. Returns
/// `None` when the hint can't be shown (overflows both edges, exactly
/// like the C++ early-`return false` at :4415 / :4432).
#[allow(clippy::too_many_arguments)]
pub fn correct_coordinates(
    element_name: &str,
    initial_x: f32,
    initial_y: f32,
    width: f32,
    height: f32,
    screen_w: f32,
    screen_h: f32,
    scale: f32,
    main_origin: [f32; 2],
    base_origin: [f32; 2],
) -> Option<(f32, f32)> {
    let _ = base_origin;
    let mut x = initial_x;
    let mut y = initial_y;

    // Main-panel command/order/turret buttons — pinned to the panel's
    // top-right corner (CInterface.cpp:4358-4396).
    let main_pinned = matches!(
        element_name,
        "buro"
            | "buca"
            | "ost"
            | "ogo"
            | "opa"
            | "ofi"
            | "oca"
            | "ocan"
            | "orep"
            | "obomb"
            | "lero"
            | "inro"
            | "oacapf"
            | "oacapn"
            | "oaprof"
            | "oapron"
            | "oafrof"
            | "oafron"
            | "tur1"
            | "tur2"
            | "tur3"
            | "tur4"
            | "sbo"
            | "callhell"
    );
    if main_pinned {
        let need_x_design: f32 = if matches!(element_name, "lero" | "sbo") {
            354.0
        } else {
            554.0
        };
        let need_x = main_origin[0] + need_x_design * scale - width;
        let need_y = main_origin[1] + 28.0 * scale - height;
        x = need_x;
        y = need_y;
    } else if matches!(element_name, "hisright" | "hisleft" | "counthz") {
        // Base-panel history/counter buttons — centred above the
        // element (CInterface.cpp:4398-4401).
        x -= width / 2.0;
        y -= height;
    }

    // Screen clamp (CInterface.cpp:4403-4433).
    let pad = HINT_OTSTUP as f32 * scale;
    if x + width + pad > screen_w {
        let overflow = x + width + pad - screen_w;
        x -= overflow;
    } else if x < 0.0 {
        x = pad;
    }
    if x + width + pad > screen_w {
        return None; // can't fit horizontally
    }
    if y + height + pad > screen_h {
        let overflow = y + height + pad - screen_h;
        y -= overflow;
    } else if y < 0.0 {
        y = pad;
    }
    if y + height + pad > screen_h {
        return None;
    }
    Some((x, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_baserepl() {
        let mut r = HintReplacer::default();
        r.set_baserepl("thz");
        assert_eq!(substitute("click [] then", &r), "click thz then");
    }

    #[test]
    fn substitute_resolves_nested() {
        let mut r = HintReplacer::default();
        r.set("titan_hint", "Титан. Приход [_titan_income]");
        r.set("_titan_income", "42");
        assert_eq!(substitute("[titan_hint]", &r), "Титан. Приход 42");
    }

    #[test]
    fn unknown_key_drops_token() {
        let r = HintReplacer::default();
        assert_eq!(substitute("a [_missing] b", &r), "a  b");
    }

    #[test]
    fn parse_argb_parses_color() {
        assert_eq!(parse_argb("FFCBCCCE"), [0xCB, 0xCC, 0xCE, 0xFF]);
    }
}
