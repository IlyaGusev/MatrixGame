//! Port of `CInterface` (Interface/CInterface.{cpp,h}).
//!
//! One panel — a named collection of `IFaceElement`s with a
//! screen-space origin. Ports the load path that reads `if/<Name>`
//! from `robots.dat` via CStorage's BlockPar-compatible accessors.
//! Mouse handling + focus-switching are here; rendering is handed
//! off to `renderer.rs`.

use crate::matrix_lib::base::storage::Storage;

use super::iface_element::{
    ElementKind, ElementState, IFaceElement, StateImage, IFEF_VISIBLE, MAX_STATES,
};

/// Design-space dimensions the C++ positions/sizes are authored in
/// (CInterface.cpp:177, :182). Every pixel value in the data is
/// relative to this; the renderer scales to the actual surface
/// dimensions at draw time.
pub const DESIGN_W: f32 = 1024.0;
pub const DESIGN_H: f32 = 768.0;

/// Port of `CInterface` — the data slice. The render/slide/focus
/// machinery lives in the parent `IFaceList` / `renderer.rs`.
#[derive(Debug, Clone)]
pub struct CInterface {
    /// `m_strName` — panel identifier (e.g. "Main", "MiniM", "Base").
    pub name: String,
    /// `m_nId` — user-defined integer id from the config.
    pub id: i32,
    /// `m_xPos` / `m_yPos` in screen-space pixels, resolved from
    /// design-space via the right/bottom anchor rule at
    /// CInterface.cpp:176-183:
    ///   if xPos != 0: xPos = screen_w - (1024 - xPos)
    ///   if yPos != 0: yPos = screen_h -  (768 - yPos)
    /// We keep the raw design-space values here and resolve once at
    /// render-time; resizing the window just re-runs the resolve.
    pub design_x: f32,
    pub design_y: f32,
    pub design_z: f32,
    pub on_top: bool,
    pub visible: bool,
    pub elements: Vec<IFaceElement>,
    /// Focused element index into `elements` (CIFaceList::m_FocusedElement
    /// in the original, but we cache it here when the focus is within
    /// this panel).
    pub focused: Option<usize>,
}

/// Subset of `CMatrixSide` state the Main panel's visibility dispatch
/// depends on. Ports the reads the C++ does at
/// CInterface.cpp:1215-1426 — `m_CurrSel`, the currently-active
/// object (for its kind + stack size), and a couple of flag queries.
#[derive(Debug, Clone, Copy)]
pub struct MainVisibilityCtx {
    pub curr_sel: crate::matrix_game::side::CurrSel,
    /// `m_ActiveObject->AsBuilding()->m_Kind` — the kind of the
    /// currently-selected building, or `None` if nothing is selected.
    pub building_kind: Option<crate::matrix_game::object_building::BuildingType>,
    /// `bld->m_BS.GetItemsCnt() == 0` — true iff the build stack is
    /// empty and the "description" / "res-per-minute" labels should
    /// be shown.
    pub building_stack_empty: bool,
    /// `bld->m_TurretsMax` — 1..4. Drives `podl1..podl4` visibility.
    pub building_turrets_max: i32,
    /// `bld->m_BS.GetItemsCnt()` — number of queued items. Drives
    /// stack-icon visibility (CInterface.cpp:1592).
    pub building_stack_items: i32,
}

impl CInterface {
    /// Port of `CInterface::Load(bp, name)` (CInterface.cpp:146 onward).
    /// Reads the named record under `if/` and walks its children
    /// extracting `Static` / `Button` / `Image` entries. Returns
    /// `None` if the panel isn't present.
    pub fn load(matrix_data: &Storage, name: &str) -> Option<Self> {
        let rec = matrix_data.block_record("if", name).or_else(|| {
            log::warn!("iface: block_record('if', {name}) returned None");
            None
        })?;
        let design_x = parse_f32(matrix_data, &rec, "xPos").unwrap_or(0.0);
        let design_y = parse_f32(matrix_data, &rec, "yPos").unwrap_or(0.0);
        let design_z = parse_f32(matrix_data, &rec, "zPos").unwrap_or(0.0);
        let id = parse_i32(matrix_data, &rec, "id").unwrap_or(0);
        let on_top = parse_i32(matrix_data, &rec, "OnTop").unwrap_or(0) != 0;

        // Walk children — Buttons / Statics / Images each with their
        // own rect + state-images blob.
        let mut elements = Vec::new();
        if let (Some(names), Some(recs)) = (
            matrix_data.get_buf(&rec, "2"),
            matrix_data.get_buf(&rec, "3"),
        ) {
            let n = names.arrays_count().min(recs.arrays_count());
            for i in 0..n {
                let kind_name = names.get_as_wstr(i);
                let child_rec = recs.get_as_wstr(i);
                let Some(kind) = (match kind_name.as_str() {
                    "Button" => Some(ElementKind::Button),
                    "Static" => Some(ElementKind::Static),
                    "Image"  => Some(ElementKind::Image),
                    _ => None,
                }) else {
                    continue;
                };
                if let Some(elem) = load_element(matrix_data, &child_rec, kind) {
                    elements.push(elem);
                }
            }
        }

        log::info!(
            "iface: loaded {} design=({},{}) id={} elements={}",
            name, design_x, design_y, id, elements.len()
        );
        Some(Self {
            name: name.to_string(),
            id,
            design_x,
            design_y,
            design_z,
            on_top,
            visible: true,
            elements,
            focused: None,
        })
    }

    /// Find a named element in this panel. Case-sensitive (matches
    /// the C++ `CWStr::operator==` comparison at CInterface.cpp:
    /// 1442, :1501, etc.).
    pub fn element_by_name(&self, name: &str) -> Option<&crate::matrix_game::interface::iface_element::IFaceElement> {
        self.elements.iter().find(|e| e.name == name)
    }

    /// Screen-space pixel rect of a named element. Computed from the
    /// panel's resolved_pos + the element's local (pos_x, pos_y,
    /// size_x, size_y) scaled by `scale`.
    pub fn element_rect(
        &self,
        name: &str,
        screen_w: f32,
        screen_h: f32,
    ) -> Option<[f32; 4]> {
        let e = self.element_by_name(name)?;
        let scale = (screen_h / DESIGN_H).max(0.1);
        let panel = self.resolved_pos(screen_w, screen_h, scale);
        Some(e.rect_in_panel(panel, scale))
    }

    /// Per-frame visibility refresh for `if/Main`. Ports the
    /// dispatch at CInterface.cpp:1214-1635 — hide every element by
    /// default, then selectively re-show based on `curr_sel` +
    /// building context. The element names match the
    /// `IF_*` constants in StringConstants.hpp.
    ///
    /// This port only covers the sub-set that fires for
    /// `CurrSel::BaseSelected` / `CurrSel::BuildingSelected` /
    /// `CurrSel::Nothing`. Robot-selection / arcade-mode /
    /// ordering-mode paths land with their owning subsystems.
    pub fn refresh_main_visibility(&mut self, ctx: &MainVisibilityCtx) {
        use crate::matrix_game::object_building::BuildingType;
        use crate::matrix_game::side::CurrSel;

        if self.name != "Main" {
            return;
        }

        // Step 1 — default EVERYTHING to hidden (CInterface.cpp:1437).
        for e in &mut self.elements {
            e.set_visible(false);
        }

        // Step 2 — unconditional panel backgrounds (CInterface.cpp:1442-1446).
        // The C++ sets these true *after* hiding by default, so they
        // always render regardless of selection.
        for e in &mut self.elements {
            match e.name.as_str() {
                "mp1" | "mp2" => e.set_visible(true),
                _ => {}
            }
        }

        let sel_something = ctx.curr_sel != CurrSel::Nothing;

        // Step 3 — labels visible whenever something is selected
        // (CInterface.cpp:1448-1507).
        if sel_something {
            for e in &mut self.elements {
                match e.name.as_str() {
                    "name" | "lives" => e.set_visible(true),
                    _ => {}
                }
            }
        }

        // Step 4 — building-specific art (CInterface.cpp:1576-1634).
        if matches!(ctx.curr_sel, CurrSel::BaseSelected | CurrSel::BuildingSelected) {
            let kind = ctx.building_kind;
            let empty = ctx.building_stack_empty;
            let has_items = ctx.building_stack_items > 0;
            for e in &mut self.elements {
                match e.name.as_str() {
                    "bopis" if empty => e.set_visible(true),
                    "mbres" if empty && kind == Some(BuildingType::Base) => e.set_visible(true),
                    "tfres" if empty && kind == Some(BuildingType::Titan) => e.set_visible(true),
                    "elfres" if empty && kind == Some(BuildingType::Electronic) => e.set_visible(true),
                    "enfres" if empty && kind == Some(BuildingType::Energy) => e.set_visible(true),
                    "pfres" if empty && kind == Some(BuildingType::Plasma) => e.set_visible(true),
                    // Per-kind platform / plant art.
                    "basepl" if kind == Some(BuildingType::Base) => e.set_visible(true),
                    "titpl"  if kind == Some(BuildingType::Titan) => e.set_visible(true),
                    "plaspl" if kind == Some(BuildingType::Plasma) => e.set_visible(true),
                    "elecpl" if kind == Some(BuildingType::Electronic) => e.set_visible(true),
                    "batpl"  if kind == Some(BuildingType::Energy) => e.set_visible(true),
                    "reppl"  if kind == Some(BuildingType::Repair) => e.set_visible(true),
                    // Build buttons.
                    "buro" if kind == Some(BuildingType::Base) => e.set_visible(true),
                    "buca" => e.set_visible(true),
                    "callhell" => e.set_visible(true),
                    "baseln" => e.set_visible(true),
                    "zagl1" => e.set_visible(true),
                    // Turret slot markers.
                    "podl1" if ctx.building_turrets_max == 1 => e.set_visible(true),
                    "podl2" if ctx.building_turrets_max == 2 => e.set_visible(true),
                    "podl3" if ctx.building_turrets_max == 3 => e.set_visible(true),
                    "podl4" if ctx.building_turrets_max == 4 => e.set_visible(true),
                    // Build-queue UI (CInterface.cpp:1592, :1679). The
                    // stack icons (`sticon`, `stother`) and progress
                    // track (`prog`) only show while an item is being
                    // produced.
                    "sticon" | "stother" if has_items => e.set_visible(true),
                    "prog" if has_items => e.set_visible(true),
                    _ => {}
                }
            }
        }
    }

    /// Resolve `(design_x, design_y)` to a top-left screen-space
    /// pixel anchor for the given surface size. Port of
    /// CInterface.cpp:176-183. If `design_{x,y}` are 0 the panel
    /// stays anchored to the top-left corner (the C++ leaves `m_{x,y}Pos`
    /// unchanged when the param is 0 — used by IF_TOP).
    pub fn resolved_pos(&self, screen_w: f32, screen_h: f32, scale: f32) -> [f32; 2] {
        let x = if self.design_x != 0.0 {
            screen_w - (DESIGN_W - self.design_x) * scale
        } else {
            0.0
        };
        let y = if self.design_y != 0.0 {
            screen_h - (DESIGN_H - self.design_y) * scale
        } else {
            0.0
        };
        [x, y]
    }
}

/// Port of the per-kind element-parse branches in `CInterface::Load`
/// (CInterface.cpp:222 onward — `Static` at :540+, `Button` at :236+,
/// `Image` at :630+).
fn load_element(
    stor: &Storage,
    rec: &str,
    kind: ElementKind,
) -> Option<IFaceElement> {
    let name = stor.block_param(rec, "Name").unwrap_or_default();
    let id = parse_i32(stor, rec, "id").unwrap_or(0);
    let group = parse_i32(stor, rec, "group").unwrap_or(0);
    let param1 = parse_f32(stor, rec, "Param1").unwrap_or(0.0);
    let param2 = parse_f32(stor, rec, "Param2").unwrap_or(0.0);
    let i_param = parse_i32(stor, rec, "PolNum").unwrap_or(0);
    let pos_x = parse_f32(stor, rec, "xPos").unwrap_or(0.0);
    let pos_y = parse_f32(stor, rec, "yPos").unwrap_or(0.0);
    let pos_z = parse_f32(stor, rec, "zPos").unwrap_or(0.0);
    let size_x = parse_f32(stor, rec, "xSize").unwrap_or(0.0);
    let size_y = parse_f32(stor, rec, "ySize").unwrap_or(0.0);
    let def_state_raw = parse_i32(stor, rec, "dState").unwrap_or(0);
    let def_state = match def_state_raw {
        1 => ElementState::Focused,
        2 => ElementState::Pressed,
        3 => ElementState::Disabled,
        _ => ElementState::Normal,
    };

    // sNormal / sFocused / sPressed / sDisabled — each is optional;
    // we parse the triple (path, X, Y, Width, Height) per CInterface.cpp
    // Static loading at :545-598 (sNormalX/sNormalY/sNormalWidth/
    // sNormalHeight + sNormal path).
    let mut images: [Option<StateImage>; MAX_STATES] = Default::default();
    for (idx, prefix) in [
        (ElementState::Normal as usize, "sNormal"),
        (ElementState::Focused as usize, "sFocused"),
        (ElementState::Pressed as usize, "sPressed"),
        (ElementState::Disabled as usize, "sDisabled"),
    ] {
        if let Some(img) = parse_state_image(stor, rec, prefix, size_x, size_y) {
            images[idx] = Some(img);
        }
    }

    // Skip elements that carry no image anywhere — likely data-only
    // records (ProgressBar anchors, counters) that the renderer
    // doesn't have a hook for yet.
    if images.iter().all(|i| i.is_none()) {
        return None;
    }

    Some(IFaceElement {
        name,
        kind,
        id,
        group,
        flags: IFEF_VISIBLE,
        param1,
        param2,
        i_param,
        pos_x,
        pos_y,
        pos_z,
        size_x,
        size_y,
        images,
        cur_state: def_state,
        def_state,
    })
}

fn parse_state_image(
    stor: &Storage,
    rec: &str,
    prefix: &str,
    size_x: f32,
    size_y: f32,
) -> Option<StateImage> {
    // `s<State>` is the atlas texture path; `s<State>X/Y/Width/Height`
    // are the sub-rect. A missing texture path means this state isn't
    // authored.
    let tex_path = stor.block_param(rec, prefix)?;
    if tex_path.is_empty() {
        return None;
    }
    let x = parse_f32(stor, rec, &format!("{prefix}X")).unwrap_or(0.0);
    let y = parse_f32(stor, rec, &format!("{prefix}Y")).unwrap_or(0.0);
    let tex_w = parse_f32(stor, rec, &format!("{prefix}Width")).unwrap_or(512.0);
    let tex_h = parse_f32(stor, rec, &format!("{prefix}Height")).unwrap_or(512.0);
    Some(StateImage {
        x,
        y,
        w: size_x,
        h: size_y,
        tex_w,
        tex_h,
        tex_path,
    })
}

fn parse_f32(stor: &Storage, rec: &str, key: &str) -> Option<f32> {
    stor.block_param(rec, key)
        .and_then(|s| s.trim().parse::<f32>().ok())
}

fn parse_i32(stor: &Storage, rec: &str, key: &str) -> Option<i32> {
    stor.block_param(rec, key)
        .and_then(|s| s.trim().parse::<i32>().ok())
}
