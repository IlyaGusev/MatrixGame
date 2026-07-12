//! Port of `CIFaceElement` (Interface/CIFaceElement.{cpp,h}).
//!
//! Each interface panel (`CInterface`) owns a list of elements —
//! Static (image / label placements), Button (interactable), Image
//! (atlas references). Elements carry four per-state images
//! (`sNormal` / `sFocused` / `sPressed` / `sDisabled`) picked from a
//! shared texture atlas; the renderer draws the appropriate one
//! based on `m_CurState`.
//!
//! The C++ uses subclass polymorphism (`CIFaceButton` / `CIFaceStatic`
//! / `CIFaceImage` derive from `CIFaceElement`); we fold the kind
//! into an enum and keep one `IFaceElement` struct with a `kind`
//! discriminant. Functionally equivalent, cheaper per-element.

/// `IFaceElementState` (Interface/Interface.h:28-37). The C++ names
/// these `sNormal`/`sFocused`/`sPressed`/`sDisabled`/
/// `sPressedUnFocused` — we keep the same order since the state index
/// is encoded in animation configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ElementState {
    #[default]
    Normal = 0,
    Focused = 1,
    Pressed = 2,
    Disabled = 3,
    /// `IFACE_PRESSED_UNFOCUSED` — a latched check button the cursor
    /// has left. Only check-type buttons ever enter this state.
    PressedUnfocused = 4,
}

/// `MAX_STATES` (Interface/Interface.h:8).
pub const MAX_STATES: usize = 5;

/// Button sub-types decoded from the element's `type` param
/// (`IFaceElementType`, Interface/Interface.h:13-26). Only the values
/// the shipped data authors on Button elements are modelled; anything
/// else behaves as a plain push button.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ButtonType {
    /// `IFACE_PUSH_BUTTON` (2) — momentary.
    #[default]
    Push,
    /// `IFACE_CHECK_BUTTON` (4) — toggles; second click unlatches.
    Check,
    /// `IFACE_CHECK_BUTTON_SPECIAL` (7) — latches; only the group
    /// reset unlatches it (conf* presets).
    CheckSpecial,
    /// `IFACE_CHECK_PUSH_BUTTON` (8) — latches on release.
    CheckPush,
}

impl ButtonType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            4 => ButtonType::Check,
            7 => ButtonType::CheckSpecial,
            8 => ButtonType::CheckPush,
            _ => ButtonType::Push,
        }
    }

    /// True for the three check types that load `sPressedUnFocused`
    /// art and participate in `CheckGroupReset` (CInterface.cpp:469,
    /// CIFaceElement.cpp:87-89).
    pub fn is_check(self) -> bool {
        !matches!(self, ButtonType::Push)
    }
}

/// `SStateImage` in the C++ (Interface/Interface.h) — a sub-rect on a
/// texture atlas used for one state of the element. `tex_w` / `tex_h`
/// are the atlas dimensions at the time the rect was authored; the
/// renderer normalises `(x, y, w, h)` to [0,1] UVs at draw time.
#[derive(Debug, Clone, Default)]
pub struct StateImage {
    /// Pixel origin of the sub-rect on the atlas.
    pub x: f32,
    pub y: f32,
    /// Pixel size of the element's draw region (matches element
    /// `m_xSize/m_ySize`).
    pub w: f32,
    pub h: f32,
    /// Authoring-time atlas dimensions. 512×512 for `interface[1-3]`.
    pub tex_w: f32,
    pub tex_h: f32,
    /// Atlas texture path (e.g. `Matrix\\IFace\\interface2`). Lets the
    /// renderer cache textures by path.
    pub tex_path: String,
}

/// One caption attached to an [`IFaceElement`] for a particular state.
///
/// Port of one entry of the per-element `Labels` block (CInterface.cpp:
/// 660-797): each entry carries a `Params` (x,y,sme_x,sme_y,align_x,
/// align_y,perenos,clip_*) line, plus `Font`, `Color`, and `State`.
/// The text itself comes from `LabelsText/<panel>/<elem>_<state>`.
///
/// `state` lives on the label (not in an array slot) so a single state
/// can carry multiple labels — the C++ does this for cobuild / cocan /
/// the menu buttons, which render a black drop-shadow caption AT
/// `(x-1, y-1)` followed by the colored caption AT `(x, y)` for every
/// state (CInterface.cpp:774-792).
///
/// All offsets are 1024×768 design-space pixels; the renderer scales
/// to the surface size.
#[derive(Debug, Clone)]
pub struct ElementLabel {
    pub state: ElementState,
    pub text: String,
    /// `Params[0..2]` — x/y inside the element rect (pre-shadow).
    pub x: f32,
    pub y: f32,
    /// `Params[2..4]` — additional sme_x/sme_y offset (the C++ adds
    /// these unconditionally before alignment).
    pub sme_x: f32,
    pub sme_y: f32,
    /// 0 = left edge, 1 = centre, 2 = right edge of the element rect.
    pub align_x: i32,
    /// 0 = top edge, 1 = middle, 2 = bottom edge.
    pub align_y: i32,
    /// `Params[6]` — wordwrap when non-zero. Ignored for now (only
    /// it_label1/2 set it; we don't yet do multi-line layout).
    pub wrap: bool,
    /// Font family identifier from the `Font` param (e.g.
    /// `Font.2Ranger`, `Font.2Small`, `Font.2Normal`). Maps to a
    /// pixel size in `text.rs`.
    pub font: String,
    /// RGBA from the C++ ARGB packed `Color` param.
    pub color: [u8; 4],
}

/// `IFaceElementType` discriminants (Interface/Interface.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ElementKind {
    /// `CIFaceStatic` — a non-interactive image/label (uses `sNormal`).
    Static = 1,
    /// `CIFaceButton` — interactable, cycles Normal → Focused → Pressed.
    Button = 2,
    /// `CIFaceImage` — atlas reference; never rendered directly in the
    /// C++ (it's a source handle for other elements).
    Image = 3,
    /// Counter / progress (`CIFaceCounter`) — its digit display is
    /// ported as the digit-static toggling in
    /// `CInterface::refresh_base_visibility_with`.
    Counter = 4,
}

pub const IFEF_VISIBLE: u32 = 1 << 0;
pub const IFEF_CLEARRECT: u32 = 1 << 1;

/// Port of the data portion of `CIFaceElement` (Interface/CIFaceElement.h:
/// 33-98). The virtuals (`Render`, `OnMouseMove`, etc.) are dispatched
/// off `kind` by the renderer instead of by a vtable — keeps the
/// storage flat.
#[derive(Debug, Clone)]
pub struct IFaceElement {
    pub name: String,
    pub kind: ElementKind,
    /// `m_Type` sub-kind for Button elements, parsed from the `type`
    /// param (CInterface.cpp:237). Push for everything else.
    pub button_type: ButtonType,
    pub id: i32,
    pub group: i32,
    pub flags: u32,
    pub param1: f32,
    pub param2: f32,
    pub i_param: i32,
    /// Element position + size in 1024×768 design-space pixels, local
    /// to the parent panel's `xPos/yPos` origin.
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub size_x: f32,
    pub size_y: f32,
    /// Per-state images. MAX_STATES slots; only the ones actually
    /// authored in the config are populated.
    pub images: [Option<StateImage>; MAX_STATES],
    /// Per-element caption list — one entry per row of the element's
    /// `Labels` block. Multiple rows can share a `state` (e.g. a
    /// black drop-shadow row + the colored row for cobuild / cocan).
    /// Order is preserved: rows render in the order they appear so
    /// shadows draw before the foreground colour.
    pub labels: Vec<ElementLabel>,
    pub cur_state: ElementState,
    pub def_state: ElementState,
    /// Port of `SElementHint` (Interface/CIFaceElement.h:21-38). Empty
    /// string means "no hint on this element". The template resolves
    /// via `TemplateLibrary::get` at hover-show time.
    pub hint_template: String,
    /// `m_Hint.x` / `m_Hint.y` — design-space offset from the element's
    /// top-left where the hint box anchors. CInterface.cpp:233-234 /
    /// :540-541.
    pub hint_offset_x: i32,
    pub hint_offset_y: i32,
    /// Port of `CAnimation` (Interface/CAnimation.{cpp,h}) — optional
    /// frame cycler; when present, the renderer draws the current
    /// frame's atlas rect in place of the Normal state image.
    pub animation: Option<ElementAnimation>,
    /// `m_VisibleAlpha` (CIFaceElement) — per-element alpha override
    /// multiplied into the state tint; used by the weapon overheat
    /// overlays (alpha = heat·0.25, CInterface.cpp:1526-1535).
    pub visible_alpha: Option<f32>,
    /// Affordability outline colour (`CreateElementRamka`,
    /// CInterface.cpp:4198-4206) — NORMAL_RAMKA green / CRITICAL_RAMKA
    /// orange stamped onto constructor pylons every frame.
    pub ramka_color: Option<[f32; 4]>,
    /// `IF_CALLHELL_ID` cooldown drum (CIFaceElement.cpp:172-208):
    /// while DISABLED, the bottom `t` fraction of the icon shows the
    /// atlas image one icon-width to the right (the charged variant)
    /// rising as `BeforMaintenanceTimeT` goes 0→1.
    pub drum_t: Option<f32>,
}

/// `CAnimation` payload: the frame rects live in the same atlas as the
/// element's Normal image (CInterface.cpp:480-516); `period` ms per
/// frame.
#[derive(Debug, Clone)]
pub struct ElementAnimation {
    pub frames: Vec<StateImage>,
    pub period: i32,
    pub current: usize,
    pub time_pass: i32,
}

impl ElementAnimation {
    /// `CAnimation::LogicTakt` (CAnimation.cpp:41-53).
    pub fn logic_takt(&mut self, ms: i32) {
        self.time_pass += ms;
        if self.time_pass >= self.period {
            self.time_pass = 0;
            self.current += 1;
            if self.current >= self.frames.len() {
                self.current = 0;
            }
        }
    }

    pub fn current_frame(&self) -> Option<&StateImage> {
        self.frames.get(self.current)
    }
}

impl IFaceElement {
    pub fn visible(&self) -> bool {
        self.flags & IFEF_VISIBLE != 0
    }

    /// Port of `CInterface::CopyElements` (CInterface.cpp:2480-2496).
    /// Copies all 4 per-state images + `Param1` / `Param2` from `src`
    /// onto `dest` — the C++ also copies `m_Actions` but we dispatch
    /// by button name instead of via actions, so that part is inert.
    ///
    /// The destination keeps its own `pos`, `size`, `name`, `kind`.
    /// This is how the constructor panel swaps the currently-selected
    /// component icon onto the pylon elements.
    pub fn copy_images_from(&mut self, src: &IFaceElement) {
        for i in 0..MAX_STATES {
            self.images[i] = src.images[i].clone();
        }
        self.param1 = src.param1;
        self.param2 = src.param2;
    }

    pub fn set_visible(&mut self, v: bool) {
        if v {
            self.flags |= IFEF_VISIBLE;
        } else {
            self.flags &= !IFEF_VISIBLE;
        }
    }

    /// Iterator over captions for the current state — falling back to
    /// the `Normal`-state captions if no label was authored for the
    /// current state. Yields multiple labels in declaration order so
    /// drop-shadow rows draw before the foreground rows.
    pub fn current_labels(&self) -> impl Iterator<Item = &ElementLabel> {
        let cur = self.cur_state;
        let any_for_cur = self.labels.iter().any(|l| l.state == cur);
        let target = if any_for_cur {
            cur
        } else {
            ElementState::Normal
        };
        self.labels.iter().filter(move |l| l.state == target)
    }

    /// Sub-rect for the current state. Falls back to `Normal` when
    /// the hovered/pressed state has no authored image — mirrors the
    /// C++ `GetStateImage` which just reads the slot directly but
    /// most elements only ship `sNormal`.
    pub fn current_image(&self) -> Option<&StateImage> {
        // An attached CAnimation overrides the Normal image while the
        // element is in its default state (CIFaceElement render path —
        // the C++ draws m_Animation->GetCurrentFrame() instead).
        if self.cur_state == ElementState::Normal {
            if let Some(f) = self.animation.as_ref().and_then(|a| a.current_frame()) {
                return Some(f);
            }
        }
        let idx = self.cur_state as usize;
        if let Some(img) = self.images.get(idx).and_then(|x| x.as_ref()) {
            return Some(img);
        }
        self.images[ElementState::Normal as usize].as_ref()
    }

    /// Screen-space pixel rect given the parent panel's top-left
    /// origin + ui scale. Used for hit-testing + vertex generation.
    pub fn rect_in_panel(&self, panel_px: [f32; 2], scale: f32) -> [f32; 4] {
        [
            panel_px[0] + self.pos_x * scale,
            panel_px[1] + self.pos_y * scale,
            self.size_x * scale,
            self.size_y * scale,
        ]
    }

    /// Hit test against pixel `[sx, sy]` given the parent panel's
    /// top-left origin + ui scale. Matches `ElementCatch` semantics
    /// (Interface/CIFaceElement.cpp) — visible rect containment
    /// check. Per-pixel alpha lives in [`Self::hit_alpha`]; the C++
    /// applies it on hover only (CIFaceButton.cpp:127 vs :187).
    pub fn hit(&self, panel_px: [f32; 2], scale: f32, sx: f32, sy: f32) -> bool {
        if !self.visible() {
            return false;
        }
        let [x, y, w, h] = self.rect_in_panel(panel_px, scale);
        sx >= x && sy >= y && sx < x + w && sy < y + h
    }

    /// `ElementAlpha` (CIFaceElement.cpp:310-318): true when the
    /// current-state source pixel under the cursor is non-transparent.
    /// Unregistered atlases (runtime-baked portraits etc.) count as
    /// opaque.
    pub fn hit_alpha(&self, panel_px: [f32; 2], scale: f32, sx: f32, sy: f32) -> bool {
        let Some(img) = self.current_image() else {
            return true;
        };
        if img.tex_path.is_empty() || img.w <= 0.0 || img.h <= 0.0 {
            return true;
        }
        let [x, y, w, h] = self.rect_in_panel(panel_px, scale);
        if w <= 0.0 || h <= 0.0 || img.tex_w <= 0.0 || img.tex_h <= 0.0 {
            return true;
        }
        // Normalised UVs so a runtime atlas rescale keeps the lookup
        // aligned with what the renderer samples.
        let u = (img.x + (sx - x) / w * img.w) / img.tex_w;
        let v = (img.y + (sy - y) / h * img.h) / img.tex_h;
        atlas_alpha_at(&img.tex_path, u, v).unwrap_or(true)
    }
}

/// Atlas-path → alpha-channel registry backing `ElementAlpha`
/// hover hit-tests. Filled by the renderer at atlas decode time.
static ATLAS_ALPHA: std::sync::Mutex<
    Option<std::collections::HashMap<String, (u32, u32, Vec<u8>)>>,
> = std::sync::Mutex::new(None);

pub fn register_atlas_alpha(key: &str, w: u32, h: u32, alpha: Vec<u8>) {
    let mut g = ATLAS_ALPHA.lock().unwrap();
    g.get_or_insert_with(Default::default)
        .insert(key.to_string(), (w, h, alpha));
}

/// Alpha > 0 test at normalised atlas UV; `None` when the atlas isn't
/// registered or the coord is out of bounds.
fn atlas_alpha_at(tex_path: &str, u: f32, v: f32) -> Option<bool> {
    let key = tex_path.replace('\\', "/").to_lowercase();
    let g = ATLAS_ALPHA.lock().unwrap();
    let (w, h, alpha) = g.as_ref()?.get(&key)?;
    let x = (u * *w as f32) as i64;
    let y = (v * *h as f32) as i64;
    if x < 0 || y < 0 || x >= *w as i64 || y >= *h as i64 {
        return None;
    }
    Some(alpha[y as usize * *w as usize + x as usize] > 0)
}
