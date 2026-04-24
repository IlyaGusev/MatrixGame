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

/// `IFaceElementState` (Interface/Interface.h:36-45). The C++ names
/// these `sNormal`/`sFocused`/`sPressed`/`sDisabled` — we keep the
/// same order since the state index is encoded in animation configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ElementState {
    #[default]
    Normal = 0,
    Focused = 1,
    Pressed = 2,
    Disabled = 3,
}

pub const MAX_STATES: usize = 4;

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
    /// Counter / progress (`CIFaceCounter`) — stubbed for now.
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
    pub cur_state: ElementState,
    pub def_state: ElementState,
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

    /// Sub-rect for the current state. Falls back to `Normal` when
    /// the hovered/pressed state has no authored image — mirrors the
    /// C++ `GetStateImage` which just reads the slot directly but
    /// most elements only ship `sNormal`.
    pub fn current_image(&self) -> Option<&StateImage> {
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
    /// check, no alpha test (the C++ `ElementAlpha` path is deferred).
    pub fn hit(&self, panel_px: [f32; 2], scale: f32, sx: f32, sy: f32) -> bool {
        if !self.visible() {
            return false;
        }
        let [x, y, w, h] = self.rect_in_panel(panel_px, scale);
        sx >= x && sy >= y && sx < x + w && sy < y + h
    }
}
