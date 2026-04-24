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
    /// Port of the `ConstPresent` / `ConstX` / `ConstY` / `ConstWidth` /
    /// `ConstHeight` top-level params (CInterface.cpp:204-214). When
    /// `ConstPresent != 0`, the panel reserves a sub-rect for the
    /// constructor's 3D-preview viewport; the rect is design-space
    /// pixels relative to the panel origin. None when `ConstPresent`
    /// is 0 or the panel doesn't host the constructor.
    pub const_rect: Option<[f32; 4]>,
}

/// Subset of game state the Base-panel constructor visibility refresh
/// reads. Ports the per-element conditions scattered across
/// CInterface.cpp:1799-2200 — armor weapon-slot caps gate pylon
/// visibility, history availability gates the prev/next buttons, and
/// resource affordability gates the build button.
#[derive(Debug, Clone, Copy)]
pub struct BaseVisibilityCtx<'a> {
    pub constructor_active: bool,
    pub build_count: i32,
    pub focused_price: Option<&'a crate::matrix_game::robot_units::UnitPrice>,
    pub summ_price: &'a crate::matrix_game::robot_units::UnitPrice,
    /// `g_MatrixMap->m_RobotWeaponMatrix[hull-1].common` —
    /// 0 hides pi1..pi4. CInterface.cpp:1817.
    pub armor_common_slots: i32,
    /// `…extra` — 0 hides pi5. CInterface.cpp:1818.
    pub armor_extra_slots: i32,
    /// `g_ConfigHistory->IsPrev()` — false → hisleft becomes
    /// IFACE_DISABLED. CInterface.cpp:1962.
    pub history_has_prev: bool,
    /// `g_ConfigHistory->IsNext()` — false → hisright disabled.
    pub history_has_next: bool,
    pub counter_up_enabled: bool,
    pub counter_down_enabled: bool,
    /// Side has enough resources for current preset and stack isn't
    /// full and robot count under cap. CInterface.cpp:1859.
    pub build_enabled: bool,
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
    /// Constructor panel is open — show the chassis/armor/head/weapon
    /// buttons + price readouts.
    pub constructor_active: bool,
    /// Turret-build mode active — show turret1..4 kind picker.
    pub turret_build_active: bool,
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

        // ConstPresent / ConstX / ConstY / ConstWidth / ConstHeight —
        // the constructor 3D-preview sub-rect (CInterface.cpp:204-214).
        let const_rect = if parse_i32(matrix_data, &rec, "ConstPresent").unwrap_or(0) != 0 {
            let cx = parse_f32(matrix_data, &rec, "ConstX").unwrap_or(0.0);
            let cy = parse_f32(matrix_data, &rec, "ConstY").unwrap_or(0.0);
            let cw = parse_f32(matrix_data, &rec, "ConstWidth").unwrap_or(0.0);
            let ch = parse_f32(matrix_data, &rec, "ConstHeight").unwrap_or(0.0);
            Some([cx, cy, cw, ch])
        } else {
            None
        };

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
                    "Image" => Some(ElementKind::Image),
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
            name,
            design_x,
            design_y,
            id,
            elements.len()
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
            const_rect,
        })
    }

    /// Find a named element in this panel. Case-sensitive (matches
    /// the C++ `CWStr::operator==` comparison at CInterface.cpp:
    /// 1442, :1501, etc.).
    pub fn element_by_name(
        &self,
        name: &str,
    ) -> Option<&crate::matrix_game::interface::iface_element::IFaceElement> {
        self.elements.iter().find(|e| e.name == name)
    }

    /// Look up a constructor-template element by `(type, kind)` —
    /// where `type` is 1/2/3/4 for Chassis/Weapon/Armor/Head (the
    /// `Param1` value) and `kind` is the specific sub-kind (`Param2`).
    ///
    /// Port of the `m_Chassis[]` / `m_Armor[]` / `m_Head[]` / `m_Weapon[]`
    /// lookup tables the C++ builds during `CInterface::Load` at
    /// CInterface.cpp:338-387. The C++ indexes those arrays by
    /// `Param2` (or `Param2 - 1` for chassis/armor); we walk the
    /// elements once on demand instead, which matches the same
    /// semantic (name-based element lookup is wrong because the
    /// template-button names don't correlate 1:1 with kinds —
    /// e.g. `chas1.Param2 = 3`, `weap1.Param2 = 7`).
    pub fn template_by_kind(
        &self,
        ty: i32,
        kind: i32,
    ) -> Option<&crate::matrix_game::interface::iface_element::IFaceElement> {
        self.elements.iter().find(|e| {
            (e.param1 as i32) == ty
                && (e.param2 as i32) == kind
                // Kind 0 is special — it resolves to `heade` / `weape`
                // (the "empty" templates). The numeric-kind templates
                // start at Param2 >= 1; we still allow kind==0 so the
                // lookup finds `heade` / `weape` correctly.
                && is_template_element_name(&e.name)
        })
    }

    /// Screen-space pixel rect of a named element. Computed from the
    /// panel's resolved_pos + the element's local (pos_x, pos_y,
    /// size_x, size_y) scaled by `scale`.
    pub fn element_rect(&self, name: &str, screen_w: f32, screen_h: f32) -> Option<[f32; 4]> {
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
        if matches!(
            ctx.curr_sel,
            CurrSel::BaseSelected | CurrSel::BuildingSelected
        ) {
            let kind = ctx.building_kind;
            let empty = ctx.building_stack_empty;
            let has_items = ctx.building_stack_items > 0;
            for e in &mut self.elements {
                match e.name.as_str() {
                    "bopis" if empty => e.set_visible(true),
                    "mbres" if empty && kind == Some(BuildingType::Base) => e.set_visible(true),
                    "tfres" if empty && kind == Some(BuildingType::Titan) => e.set_visible(true),
                    "elfres" if empty && kind == Some(BuildingType::Electronic) => {
                        e.set_visible(true)
                    }
                    "enfres" if empty && kind == Some(BuildingType::Energy) => e.set_visible(true),
                    "pfres" if empty && kind == Some(BuildingType::Plasma) => e.set_visible(true),
                    // Per-kind platform / plant art.
                    "basepl" if kind == Some(BuildingType::Base) => e.set_visible(true),
                    "titpl" if kind == Some(BuildingType::Titan) => e.set_visible(true),
                    "plaspl" if kind == Some(BuildingType::Plasma) => e.set_visible(true),
                    "elecpl" if kind == Some(BuildingType::Electronic) => e.set_visible(true),
                    "batpl" if kind == Some(BuildingType::Energy) => e.set_visible(true),
                    "reppl" if kind == Some(BuildingType::Repair) => e.set_visible(true),
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

        // Step 5 — constructor sub-panel visibility. When the player
        // opens the robot builder (`buro`), the Base panel's component
        // buttons become visible. Ports the `if (m_Active)` branches
        // scattered through CInterface.cpp:1215-1426.
        if ctx.constructor_active {
            for e in &mut self.elements {
                let n = e.name.as_str();
                let show = matches!(
                    n,
                    "conl" | "conr" | "conf"               // const left / right / foot frames
                    | "cocan" | "cobuild"                    // cancel + build
                    | "chas1" | "chas2" | "chas3" | "chas4" | "chas5"
                    | "hull1" | "hull2" | "hull3" | "hull4" | "hull5" | "hull6"
                    | "head1" | "head2" | "head3" | "head4" | "heade"
                    | "weap1" | "weap2" | "weap3" | "weap4" | "weap5"
                    | "weap6" | "weap7" | "weap8" | "weap9" | "weap10" | "weape"
                    | "pi1" | "pi2" | "pi3" | "pi4" | "pi5"
                    | "pihe" | "pihu" | "pich"
                    | "hisleft" | "hisright" | "counthz"
                    | "bup" | "bdown"
                    | "itch" | "itprice"                       // item chars + price
                    | "label1" | "label2" | "descrT" | "descrB"
                    | "weight" | "speed" | "struct" | "damage"
                    | "res_summ" | "res_unit"
                    | "titan" | "electr" | "energy" | "plasma"
                    | "titans" | "electrs" | "energys" | "plasmas"
                    | "warning" | "warning1"
                );
                if show {
                    e.set_visible(true);
                }
            }
        }

        // Step 6 — turret-build mode — show the turret-kind picker.
        // Ports `CInterface::BeginBuildTurret` + the m_Turrets[4] button
        // visibility at CInterface.cpp:3518-3542.
        if ctx.turret_build_active {
            for e in &mut self.elements {
                let n = e.name.as_str();
                if matches!(n, "tur1" | "tur2" | "tur3" | "tur4") {
                    e.set_visible(true);
                }
            }
        }
    }

    /// Per-frame visibility refresh for `if/Base`. Hides the overlapping
    /// chas/hull/head/weap template images (they live at stacked
    /// positions and are only laid out by the C++ popup overlay we
    /// haven't ported yet) while keeping pylons, backgrounds, stats,
    /// and the build/cancel buttons visible.
    pub fn refresh_base_visibility(&mut self, constructor_active: bool) {
        self.refresh_base_visibility_with(constructor_active, 1);
    }

    /// Same as [`refresh_base_visibility`] but additionally keeps only
    /// the digit-image static matching `build_count` visible (port of
    /// `CIFaceCounter::ManageButtons` + `GetImage`). The other digit
    /// statics — `zero`, `one`, ... `six` — are hidden.
    pub fn refresh_base_visibility_with(&mut self, constructor_active: bool, build_count: i32) {
        self.refresh_base_visibility_full(
            constructor_active,
            build_count,
            None,
            &crate::matrix_game::robot_units::UnitPrice::zero(),
        );
    }

    /// Full Base-panel visibility refresh including the per-resource
    /// icon toggles for the focused / summary price popups.
    ///
    /// `focused_price` is the price of the currently-focused component
    /// (or None when nothing is focused) — port of
    /// `CreateItemPrice` (CInterface.cpp:3146).
    ///
    /// `summ_price` is the live preview total ×`build_count` —
    /// port of `CreateSummPrice` (CInterface.cpp:3220).
    ///
    /// For each resource (titan / electr / energy / plasma) the
    /// matching template icon stays visible iff its slot has a non-
    /// zero value in either price. The C++ creates dynamic statics
    /// per non-zero resource; we just toggle the existing
    /// `titan`/`electr`/`energy`/`plasma` template-image elements
    /// that the panel data already holds.
    pub fn refresh_base_visibility_full(
        &mut self,
        constructor_active: bool,
        build_count: i32,
        focused_price: Option<&crate::matrix_game::robot_units::UnitPrice>,
        summ_price: &crate::matrix_game::robot_units::UnitPrice,
    ) {
        self.refresh_base_visibility_v2(BaseVisibilityCtx {
            constructor_active,
            build_count,
            focused_price,
            summ_price,
            armor_common_slots: 0,
            armor_extra_slots: 0,
            history_has_prev: false,
            history_has_next: false,
            counter_up_enabled: true,
            counter_down_enabled: true,
            build_enabled: true,
        });
    }

    /// Full per-element visibility refresh — port of the `IF_BASE`
    /// branch of `CInterface::LogicTakt` (CInterface.cpp:1799-2200).
    /// Each constructor element is conditionally shown based on
    /// armor weapon-slot caps, history availability, and resource
    /// affordability. The C++ also disables (rather than hides) some
    /// buttons via `SetState(IFACE_DISABLED)`; we mirror that on
    /// HISTORY_LEFT/RIGHT and CONST_BUILD.
    pub fn refresh_base_visibility_v2(&mut self, ctx: BaseVisibilityCtx) {
        use crate::matrix_game::robot_units::Resource;

        if self.name != "Base" {
            return;
        }
        if !ctx.constructor_active {
            for e in &mut self.elements {
                e.set_visible(false);
            }
            return;
        }
        let build_count = ctx.build_count;
        let focused_price = ctx.focused_price;
        let summ_price = ctx.summ_price;
        // Digit-image element names per CInterface.cpp:563-576.
        const DIGIT_NAMES: [&str; 7] = ["zero", "one", "two", "three", "four", "five", "six"];
        let active_digit = build_count.clamp(0, 6) as usize;

        // Resource-icon visibility — show the icon when EITHER the
        // summary-price OR the focused-price has a non-zero entry for
        // that resource (matches CreateSummPrice + CreateItemPrice
        // both rendering icons for non-zero rows).
        let res_visible = |r: Resource| -> bool {
            let summ_nz = summ_price.resources[r as usize] != 0;
            let focus_nz = focused_price
                .map(|p| p.resources[r as usize] != 0)
                .unwrap_or(false);
            summ_nz || focus_nz
        };
        let titan_v = res_visible(Resource::Titan);
        let electr_v = res_visible(Resource::Electronics);
        let energy_v = res_visible(Resource::Energy);
        let plasma_v = res_visible(Resource::Plasma);

        // Helper: detect template / popup-overlay element names that
        // should stay hidden until the popup-menu is invoked. The C++
        // stacks these at the same panel position and only one shows
        // at a time during a popup interaction. Until the popup is
        // ported we just hide them all.
        //
        // Pattern: a category prefix (chas/hull/head/weap) followed by
        // a digit suffix (chas1..5, hull1..6, head1..7, weap1..10) OR
        // the empty-slot variants `heade`/`weape`. Plain category
        // labels like `chasl`, `weapl`, `hulll`, `headl` are STATIC
        // text elements that remain visible.
        fn is_kind_template(n: &str) -> bool {
            for prefix in ["chas", "hull", "head", "weap"] {
                if let Some(suffix) = n.strip_prefix(prefix) {
                    // Digit suffix → numbered template button
                    if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                        return true;
                    }
                    // Empty-slot variants `heade` / `weape`
                    if suffix == "e" {
                        return true;
                    }
                    // Decorative `*Nst` (hull1st, weapon3st, etc.) —
                    // also `weaponNst` (note `weapon` not `weap`).
                    if suffix.ends_with("st")
                        && suffix
                            .trim_end_matches("st")
                            .chars()
                            .any(|c| c.is_ascii_digit())
                    {
                        return true;
                    }
                }
            }
            // `weaponNst` siblings (see panel dump — uses `weapon` not
            // `weap` prefix for the *st statics).
            if let Some(rest) = n.strip_prefix("weapon") {
                if rest.ends_with("st") {
                    return true;
                }
            }
            // Per-template info text statics: `iw1text`, `ihe2text`,
            // `ihu3text`, `ich4text`. The bare `iwtext`/`ihetext`/etc.
            // are headers — keep them visible.
            for prefix in ["iw", "ihe", "ihu", "ich"] {
                if let Some(suffix) = n.strip_prefix(prefix) {
                    if suffix.ends_with("text")
                        && suffix
                            .trim_end_matches("text")
                            .chars()
                            .any(|c| c.is_ascii_digit())
                    {
                        return true;
                    }
                }
            }
            false
        }

        for e in &mut self.elements {
            let n = e.name.as_str();
            let is_template = is_kind_template(n);
            let is_digit = DIGIT_NAMES.contains(&n);
            // CInterface.cpp:889-892 — resource icon template names.
            let res_v = match n {
                "titan" | "titans" => Some(titan_v),
                "electr" | "electrs" => Some(electr_v),
                "energy" | "energys" => Some(energy_v),
                "plasma" | "plasmas" => Some(plasma_v),
                _ => None,
            };

            // Weapon-pylon visibility per CInterface.cpp:1972-2090.
            // pi1..pi4 only visible when armor has common slots;
            // pi5 only when armor has extra slots.
            let pylon_v: Option<bool> = match n {
                "pi1" | "pi2" | "pi3" | "pi4" => Some(ctx.armor_common_slots > 0),
                "pi5" => Some(ctx.armor_extra_slots > 0),
                _ => None,
            };

            // History button enabled state per CInterface.cpp:1954-1967.
            // The C++ uses SetState(IFACE_DISABLED) and keeps the
            // button visible; we mirror via cur_state.
            match n {
                "hisleft" => {
                    if !ctx.history_has_prev {
                        e.cur_state = ElementState::Disabled;
                    } else if matches!(e.cur_state, ElementState::Disabled) {
                        e.cur_state = ElementState::Normal;
                    }
                }
                "hisright" => {
                    if !ctx.history_has_next {
                        e.cur_state = ElementState::Disabled;
                    } else if matches!(e.cur_state, ElementState::Disabled) {
                        e.cur_state = ElementState::Normal;
                    }
                }
                "bup" => {
                    if !ctx.counter_up_enabled {
                        e.cur_state = ElementState::Disabled;
                        e.def_state = ElementState::Disabled;
                    } else if matches!(e.cur_state, ElementState::Disabled) {
                        e.cur_state = ElementState::Normal;
                        e.def_state = ElementState::Normal;
                    }
                }
                "bdown" => {
                    if !ctx.counter_down_enabled {
                        e.cur_state = ElementState::Disabled;
                        e.def_state = ElementState::Disabled;
                    } else if matches!(e.cur_state, ElementState::Disabled) {
                        e.cur_state = ElementState::Normal;
                        e.def_state = ElementState::Normal;
                    }
                }
                "cobuild" => {
                    if !ctx.build_enabled {
                        e.cur_state = ElementState::Disabled;
                        e.def_state = ElementState::Disabled;
                    } else if matches!(e.cur_state, ElementState::Disabled) {
                        e.cur_state = ElementState::Normal;
                        e.def_state = ElementState::Normal;
                    }
                }
                _ => {}
            }

            if is_digit {
                let idx = DIGIT_NAMES.iter().position(|&d| d == n).unwrap_or(0);
                e.set_visible(idx == active_digit);
            } else if let Some(v) = pylon_v {
                e.set_visible(v);
            } else if let Some(v) = res_v {
                e.set_visible(v);
            } else {
                e.set_visible(!is_template);
            }
        }
    }

    /// Port of the `CInterface::CopyElements(src, dst)` calls in
    /// `CConstructor::SuperDjeans` (CConstructor.cpp:451-520). For each
    /// component slot (chassis, armor, head, 5 weapon pylons) read
    /// the currently-selected kind from `cfg` and swap the pylon
    /// element's images to match that template's images.
    ///
    /// Runs every frame while the constructor is active so preset
    /// loads / history restores / wrap-around cycles all reflect in
    /// the UI without bespoke call-site glue.
    pub fn apply_constructor_to_pylons(
        &mut self,
        cfg: &crate::matrix_game::robot_units::RobotConfig,
    ) {
        if self.name != "Base" {
            return;
        }
        // Faithful port of CConstructor.cpp:451/465/468/490 — each
        // pylon copy-target reads its source element via the
        // kind-indexed `m_Chassis[]` / `m_Armor[]` / `m_Head[]` /
        // `m_Weapon[]` tables (built at CInterface.cpp:338-387).
        // We resolve those indirectly through `template_by_kind` which
        // walks elements filtered by `Param1` (type) + `Param2` (kind).
        //
        // Param1 values from the Base panel config:
        //   1 = Chassis, 2 = Weapon, 3 = Armor, 4 = Head
        let find_template = |ty: i32, kind: i32| -> Option<String> {
            self.template_by_kind(ty, kind).map(|e| e.name.clone())
        };
        let chassis_src = find_template(1, cfg.chassis.kind.0);
        let hull_src = find_template(3, cfg.hull.unit.kind.0);
        // Head kind 0 resolves to the `heade` empty-head template
        // (which has Param2=0). Weapon kind 0 → `weape`.
        let head_src = find_template(4, cfg.head.kind.0);
        let weapon_srcs: Vec<(Option<String>, String)> = (0..5)
            .map(|i| {
                let slot_name = format!("pi{}", i + 1);
                let kind = cfg.weapon[i].kind.0;
                (find_template(2, kind), slot_name)
            })
            .collect();

        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let log_once = !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed);
        if log_once {
            log::info!(
                "apply_constructor_to_pylons: chassis(kind={})={:?} hull(kind={})={:?} head(kind={})={:?} weapons={:?}",
                cfg.chassis.kind.0, chassis_src,
                cfg.hull.unit.kind.0, hull_src,
                cfg.head.kind.0, head_src,
                weapon_srcs,
            );
        }
        self.copy_pair(chassis_src.as_deref(), "pich");
        self.copy_pair(hull_src.as_deref(), "pihu");
        self.copy_pair(head_src.as_deref(), "pihe");
        for (src, dst) in &weapon_srcs {
            self.copy_pair(src.as_deref(), dst.as_str());
        }
    }

    /// Port of `CInterface::CopyElements(src_name, dst_name)` by-name
    /// lookup. Finds both elements in `self.elements`; copies src's
    /// per-state images onto dst via `IFaceElement::copy_images_from`.
    fn copy_pair(&mut self, src_name: Option<&str>, dst_name: &str) {
        let Some(src_name) = src_name else {
            return;
        };
        let Some(src_idx) = self.elements.iter().position(|e| e.name == src_name) else {
            return;
        };
        let Some(dst_idx) = self.elements.iter().position(|e| e.name == dst_name) else {
            return;
        };
        if src_idx == dst_idx {
            return;
        }
        // Split borrows with `split_at_mut` so we can mutate dst while
        // immutably reading src.
        let (a, b) = if src_idx < dst_idx {
            let (left, right) = self.elements.split_at_mut(dst_idx);
            (&left[src_idx], &mut right[0])
        } else {
            let (left, right) = self.elements.split_at_mut(src_idx);
            (&right[0], &mut left[dst_idx])
        };
        b.copy_images_from(a);
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

/// Returns true for element names that match the constructor
/// template-button naming (`chas{N}`, `hull{N}`, `head{N}`/`heade`,
/// `weap{N}`/`weape`). Used by `template_by_kind` so a plain pylon
/// element like `pich` (which also has `Param1=1`) isn't returned as
/// a chassis template.
fn is_template_element_name(name: &str) -> bool {
    for prefix in ["chas", "hull", "head", "weap"] {
        if let Some(suffix) = name.strip_prefix(prefix) {
            if suffix == "e" || suffix.chars().all(|c| c.is_ascii_digit()) {
                return !suffix.is_empty();
            }
        }
    }
    false
}

/// Port of the per-kind element-parse branches in `CInterface::Load`
/// (CInterface.cpp:222 onward — `Static` at :540+, `Button` at :236+,
/// `Image` at :630+).
fn load_element(stor: &Storage, rec: &str, kind: ElementKind) -> Option<IFaceElement> {
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
