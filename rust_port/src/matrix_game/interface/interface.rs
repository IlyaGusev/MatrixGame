//! Port of `CInterface` (Interface/CInterface.{cpp,h}).
//!
//! One panel — a named collection of `IFaceElement`s with a
//! screen-space origin. Ports the load path that reads `if/<Name>`
//! from `robots.dat` via CStorage's BlockPar-compatible accessors.
//! Mouse handling + focus-switching are here; rendering is handed
//! off to `renderer.rs`.

use crate::matrix_lib::base::storage::Storage;

use super::iface_element::{
    ElementKind, ElementLabel, ElementState, IFaceElement, StateImage, IFEF_VISIBLE, MAX_STATES,
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
    pub focused_price: Option<&'a crate::matrix_game::interface::constructor::UnitPrice>,
    pub summ_price: &'a crate::matrix_game::interface::constructor::UnitPrice,
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
    /// True when the player has hit the side robot cap — drives `warn1`/
    /// `warnl` visibility. Port of CInterface.cpp:1829-1831.
    pub robot_limit_reached: bool,
    /// Currently-focused (type, kind) on a pylon — drives the right-side
    /// per-kind preview / info overlay (head{N}_st, ihu{N}text, etc.).
    /// `None` when no pylon is focused. Port of the
    /// `m_FocusedElement->m_Param1/Param2` reads at CInterface.cpp:2358-2406.
    pub focused_target: Option<crate::matrix_game::interface::constructor::FocusTarget>,
}

/// Subset of `CMatrixSide` state the Main panel's visibility dispatch
/// depends on. Ports the reads the C++ does at
/// CInterface.cpp:1215-1426 — `m_CurrSel`, the currently-active
/// object (for its kind + stack size), and a couple of flag queries.
#[derive(Debug, Clone)]
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
    /// Per-slot cannon kind across the build stack (`m_Top` →
    /// `m_NextStackItem` walk in C++). Index 0 is the head, 1..5 the
    /// queued tail items. `Some(N)` flags a turret of kind N (1..=4);
    /// `None` for an empty slot or a non-cannon item. Drives the
    /// CIFaceList::CreateStackIcon dispatch (CInterface.cpp:3956-4131):
    /// the head copies `tmd{N}` onto `sticon`; tails get a dynamic
    /// 25×25 small icon at `(225+(i-1)*31, 105)` from `tsm{N}`.
    pub building_stack_turret_kinds: [Option<i32>; MAX_STACK_ICONS],
    /// Per-slot robot-icon atlas keys. `Some(key)` flags a robot
    /// build-stack item whose 64×64 portrait has been baked by
    /// `RobotIconCache` and registered as a virtual atlas under `key`;
    /// the dynamic stack-icon element samples that atlas. `None` when
    /// the slot is empty / a turret / the icon hasn't been baked yet.
    /// Port of the robot branch of `CIFaceList::CreateStackIcon`
    /// (CInterface.cpp:3975-3982 + 4055-4117): the C++ sources from
    /// `m_MedTexture`/`m_SmallTexture` baked in
    /// `CMatrixRobotAI::CreateTextures` (MatrixRobot.cpp:5342-5380).
    pub building_stack_robot_atlas_keys: [Option<String>; MAX_STACK_ICONS],
}

/// `MAX_STACK_UNITS` (MatrixObjectBuilding.hpp:43) — 1 head + 5 tail.
pub const MAX_STACK_ICONS: usize = 6;
/// `STACK_ICON` (CInterface.h:72) — base id for dynamic stack-icon
/// elements. `STACK_ICON+(num-1)` identifies the icon for queue
/// position `num`. Used so per-frame regen can find/drop entries
/// from prior frames (`IS_STACK_ICON(x)` at CInterface.h:84 spans
/// `[STACK_ICON, STACK_ICON+9)`).
pub const STACK_ICON_BASE: i32 = 100;

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

        // Attach LabelsText/<panel>/<elem>_<state> captions to each
        // element. Port of the labels-walking loop in CInterface::Load
        // (CInterface.cpp:657-797). The C++ also stores per-state
        // position / alignment / colour parsed from `Params` blocks
        // nested per-element; for now we only carry the text +
        // a default colour because the constructor's button captions
        // all use the same centred-white layout.
        attach_labels(matrix_data, &rec, name, &mut elements);

        // Port of `CInterface::SortElementsByZ()` (CInterface.cpp:2439-2478),
        // called at the end of Load (CInterface.cpp:809). The C++ bubble-
        // sort swaps when `elem.zPos < next.zPos`, producing descending
        // zPos order — elements with higher zPos appear EARLIER in the
        // list, so they render first (back); lower-zPos elements render
        // last (front). Crucial for the Main panel: `mp1` (z=0.001) must
        // render behind `basepl` / `mbres` / `podl*` (z=1e-6) even though
        // the data authors them later.
        //
        // Stable sort preserves data-file order for elements with equal
        // zPos — the C++ bubble-sort has the same property.
        elements.sort_by(|a, b| {
            b.pos_z
                .partial_cmp(&a.pos_z)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let hinted: Vec<&str> = elements
            .iter()
            .filter(|e| !e.hint_template.is_empty())
            .map(|e| e.name.as_str())
            .collect();
        log::info!(
            "iface: loaded {} design=({},{}) id={} elements={} hinted={:?}",
            name,
            design_x,
            design_y,
            id,
            elements.len(),
            hinted
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

    /// Look up the localised popup label for a `(type, kind)` part
    /// (e.g. "Pulemyot" for weapon kind 1). The C++ stores these as
    /// `iw{N}text_sNormal` / `ihu{N}text_sNormal` /
    /// `ihe{N}text_sNormal` / `ich{N}text_sNormal` strings inside the
    /// `LabelsText/Base` block, then assigns them into per-popup
    /// `g_PopupChassis[]` / `g_PopupHull[]` / etc. arrays in
    /// `CInterface.cpp:715-772`. We reuse the already-attached
    /// `iw{N}text` / `ihu{N}text` / `ihe{N}text` / `ich{N}text` element
    /// labels — same data, no extra parse pass.
    pub fn popup_kind_label(&self, ty: i32, kind: i32) -> Option<&str> {
        if kind <= 0 {
            return None;
        }
        let prefix = match ty {
            1 => "ich",
            2 => "iw",
            3 => "ihu",
            4 => "ihe",
            _ => return None,
        };
        let elem_name = format!("{prefix}{kind}text");
        let e = self.elements.iter().find(|e| e.name == elem_name)?;
        e.labels
            .iter()
            .find(|l| matches!(l.state, ElementState::Normal))
            .map(|l| l.text.as_str())
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
            // `bld_tu` in CInterface.cpp:1224 — gates `buro` (1596),
            // `buca` (1600), and `callhell` (1607) so the turret-kind
            // picker can take over the same screen real estate.
            let bld_tu = ctx.turret_build_active;
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
                    // Build buttons — gated by `!bld_tu` (CInterface.cpp:
                    // 1596, 1600). In turret-build mode the picker
                    // (`tur1..4` + `ocan`) replaces them.
                    "buro" if !bld_tu && kind == Some(BuildingType::Base) => e.set_visible(true),
                    "buca" if !bld_tu => e.set_visible(true),
                    // Reinforcements ("call from hell") is visible for
                    // any selected building UNLESS we're in turret-
                    // build mode (CInterface.cpp:1607 — `&& !bld_tu`).
                    // State goes DISABLED while maintenance is off or
                    // the cooldown hasn't elapsed (CInterface.cpp:
                    // 1609-1613). We don't yet model maintenance, so
                    // it renders DISABLED always when shown.
                    "callhell" if !bld_tu => {
                        e.set_visible(true);
                        e.cur_state = ElementState::Disabled;
                        e.def_state = ElementState::Disabled;
                    }
                    "baseln" => e.set_visible(true),
                    "zagl1" => e.set_visible(true),
                    // Resource-income readout — shown while the build
                    // stack is empty. Separate keys for base vs factory
                    // (CInterface.cpp:1478, :1489).
                    "bresg" if empty && kind == Some(BuildingType::Base) => e.set_visible(true),
                    "fresg" if empty && kind.is_some() && kind != Some(BuildingType::Base) => {
                        e.set_visible(true)
                    }
                    // Turret slot markers.
                    "podl1" if ctx.building_turrets_max == 1 => e.set_visible(true),
                    "podl2" if ctx.building_turrets_max == 2 => e.set_visible(true),
                    "podl3" if ctx.building_turrets_max == 3 => e.set_visible(true),
                    "podl4" if ctx.building_turrets_max == 4 => e.set_visible(true),
                    // Build-queue stack icons (CInterface.cpp:1592).
                    // `prog` (IF_MAIN_PROG, "Программы") is the
                    // robot-orders programs panel and only shows when a
                    // group is selected (CInterface.cpp:1679); it's
                    // unrelated to the build stack.
                    "sticon" | "stother" if has_items => e.set_visible(true),
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
                    | "it_label1" | "it_label2" | "label1" | "label2" | "descrT" | "descrB"
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

        // Step 6 — turret-build mode — show the turret-kind picker
        // and the cancel button. Ports `CInterface::BeginBuildTurret`
        // (CInterface.cpp:3518-3542 for the tur1..4 buttons) and the
        // `bld_tu` else-branch at CInterface.cpp:1696 that flips
        // `IF_ORDER_CANCEL` (`ocan`) on while the player picks a
        // turret kind.
        if ctx.turret_build_active {
            for e in &mut self.elements {
                let n = e.name.as_str();
                if matches!(n, "tur1" | "tur2" | "tur3" | "tur4" | "ocan") {
                    e.set_visible(true);
                }
            }
        }

        // Step 7 — build-stack icons. Port of
        // `CIFaceList::CreateStackIcon` (CInterface.cpp:3956-4131).
        //
        // The original `CBuildStack::AddItem` calls `CreateStackIcon`
        // which spawns a new `CIFaceStatic` per queue position; the
        // head goes at (232, 55), 42×42 sourced from `tmd{N}`, the
        // tail items at (225+(num-2)*31, 105), 25×25 sourced from
        // `tsm{N}` (cannon branch CInterface.cpp:4018-4053).
        // `DeleteItem` removes them and shifts tail icons left when
        // the head completes (CInterface.cpp:4135-4170).
        //
        // We mirror the visible end state per frame: drop any
        // dynamically-spawned icons from the prior frame, then push a
        // fresh static for each turret slot. `sticon`/`stother`
        // remain unmodified — they're the authored frame plates the
        // dynamic icons sit inside, exactly like the C++ keeps
        // them under the dynamic STACK_ICON elements.
        //
        // Robot items source from a runtime-baked 64×64 texture
        // registered as a virtual atlas (`_robot_icon_<hash>`). Port
        // of the robot branch at CInterface.cpp:3975-3982 + 4067-4078;
        // the C++ samples the entire `m_MedTexture` (xTexPos=0,
        // yTexPos=0, full texture rect, fullsize=true) so we do the
        // same — the StateImage covers the whole 64×64 page, and the
        // element rect (42×42 / 25×25) handles the on-screen sizing.
        self.elements.retain(|e| !e.name.starts_with("_dynstack_"));
        for i in 0..MAX_STACK_ICONS {
            let (pos_x, pos_y, size_xy) = if i == 0 {
                (232.0, 55.0, 42.0)
            } else {
                (225.0 + (i as f32 - 1.0) * 31.0, 105.0, 25.0)
            };
            let images = if let Some(kind) = ctx.building_stack_turret_kinds[i] {
                let src_name = if i == 0 {
                    format!("tmd{}", kind)
                } else {
                    format!("tsm{}", kind)
                };
                let Some(src_idx) = self.elements.iter().position(|e| e.name == src_name) else {
                    continue;
                };
                self.elements[src_idx].images.clone()
            } else if let Some(key) = &ctx.building_stack_robot_atlas_keys[i] {
                let mut imgs: [Option<StateImage>; MAX_STATES] = Default::default();
                imgs[ElementState::Normal as usize] = Some(StateImage {
                    x: 0.0,
                    y: 0.0,
                    w: 64.0,
                    h: 64.0,
                    tex_w: 64.0,
                    tex_h: 64.0,
                    tex_path: key.clone(),
                });
                imgs
            } else {
                continue;
            };
            self.elements.push(IFaceElement {
                name: format!("_dynstack_{}", i + 1),
                kind: ElementKind::Static,
                id: STACK_ICON_BASE + i as i32,
                group: 0,
                flags: IFEF_VISIBLE,
                param1: 0.0,
                param2: 0.0,
                i_param: 0,
                pos_x,
                pos_y,
                pos_z: 0.0,
                size_x: size_xy,
                size_y: size_xy,
                images,
                labels: Vec::new(),
                cur_state: ElementState::Normal,
                def_state: ElementState::Normal,
                hint_template: String::new(),
                hint_offset_x: 0,
                hint_offset_y: 0,
            });
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
            &crate::matrix_game::interface::constructor::UnitPrice::zero(),
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
        focused_price: Option<&crate::matrix_game::interface::constructor::UnitPrice>,
        summ_price: &crate::matrix_game::interface::constructor::UnitPrice,
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
            robot_limit_reached: false,
            focused_target: None,
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
        use crate::matrix_game::config::Resource;

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

        // Decode (category_letter, kind_index) for the *_st preview and
        // i*N text statics. The C++ shows these on the right-hand side
        // of the constructor when the focused pylon's (type, kind)
        // matches.
        //
        //   c: chassis → MRT_CHASSIS  (chasNst, ichNtext, ichtext)
        //   h: armor   → MRT_ARMOR    (hullNst, ihuNtext, ihutext)
        //   d: head    → MRT_HEAD     (headNst, iheNtext, ihetext)
        //   w: weapon  → MRT_WEAPON   (weaponNst, iwNtext, iwtext)
        //
        // CInterface.cpp:2358-2406 enumerates each name + the matching
        // (Param1, Param2) gate. Numbered template buttons themselves
        // (`chasN`, `hullN`, `headN`, `weapN`, `heade`, `weape`) are
        // popup-only image sources — they never appear in the panel
        // outside the popup overlay, so we keep them hidden.
        fn template_target(n: &str) -> Option<(char, i32)> {
            // *Nst (chas1st, hull1st, head1st) — preview state image
            // tied to (type, kind).
            for (prefix, ch) in [("chas", 'c'), ("hull", 'h'), ("head", 'd')] {
                if let Some(suffix) = n.strip_prefix(prefix) {
                    if let Some(body) = suffix.strip_suffix("st") {
                        if !body.is_empty() && body.chars().all(|c| c.is_ascii_digit()) {
                            if let Ok(k) = body.parse::<i32>() {
                                return Some((ch, k));
                            }
                        }
                    }
                }
            }
            // weaponNst — note `weapon` (not `weap`), CInterface.cpp:
            // 2388-2406.
            if let Some(rest) = n.strip_prefix("weapon") {
                if let Some(num) = rest.strip_suffix("st") {
                    if let Ok(k) = num.parse::<i32>() {
                        return Some(('w', k));
                    }
                }
            }
            // ihe{N}text / iw{N}text / ihu{N}text / ich{N}text — info
            // text statics tied to (head/weapon/armor/chassis, kind N).
            for (prefix, ch) in [("ihe", 'd'), ("iw", 'w'), ("ihu", 'h'), ("ich", 'c')] {
                if let Some(suffix) = n.strip_prefix(prefix) {
                    if let Some(num) = suffix.strip_suffix("text") {
                        if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
                            if let Ok(k) = num.parse::<i32>() {
                                return Some((ch, k));
                            }
                        }
                    }
                }
            }
            None
        }

        // Bare info-text headers (`ihetext`/`iwtext`/`ihutext`/`ichtext`)
        // are visible while the matching category is focused. C++
        // condition at CInterface.cpp:2358-2406 (`pElement->m_strName ==
        // IF_BASE_IHE_TEXT && Param1 == MRT_HEAD`).
        fn info_header_for(n: &str) -> Option<char> {
            match n {
                "ihetext" => Some('d'),
                "iwtext" => Some('w'),
                "ihutext" => Some('h'),
                "ichtext" => Some('c'),
                _ => None,
            }
        }

        // Numbered popup-template buttons + the empty-slot variants —
        // never visible in the constructor outside the popup overlay,
        // which renders them via its own pass.
        fn is_popup_template(n: &str) -> bool {
            for prefix in ["chas", "hull", "head", "weap"] {
                if let Some(suffix) = n.strip_prefix(prefix) {
                    if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                        return true;
                    }
                    if suffix == "e" {
                        return true;
                    }
                }
            }
            false
        }

        // Map a focused (type, kind) into our (category_letter, kind)
        // tuple. `None` when no pylon is focused or the focused kind is
        // empty — `head{N}_st` etc. should NOT appear for the empty
        // slot in the original UI.
        use crate::matrix_game::object_robot::RobotUnitType;
        let focus_tk: Option<(char, i32)> = ctx.focused_target.and_then(|t| {
            if t.kind.is_empty() {
                return None;
            }
            let ch = match t.ty {
                RobotUnitType::Chassis => 'c',
                RobotUnitType::Armor => 'h',
                RobotUnitType::Head => 'd',
                RobotUnitType::Weapon => 'w',
                RobotUnitType::Empty => return None,
            };
            Some((ch, t.kind.0))
        });

        for e in &mut self.elements {
            let n = e.name.as_str();
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
            // pi1..pi4 each have their own threshold against
            // `m_RobotWeaponMatrix[hull-1].common`. pi5 needs `extra`.
            let pylon_v: Option<bool> = match n {
                "pi1" => Some(ctx.armor_common_slots > 0),
                "pi2" => Some(ctx.armor_common_slots > 1),
                "pi3" => Some(ctx.armor_common_slots > 2),
                "pi4" => Some(ctx.armor_common_slots > 3),
                "pi5" => Some(ctx.armor_extra_slots > 0),
                _ => None,
            };

            // CInterface.cpp:1829-1831 — `warn1` / `warnl` only when
            // the player has hit the side robot cap.
            let warn_v = match n {
                "warn1" | "warnl" | "warn" | "warning1" | "warning" => {
                    Some(ctx.robot_limit_reached)
                }
                _ => None,
            };

            // Right-side per-kind preview / info-text overlay. The C++
            // shows `headN_st` + `iheN_text` etc. for the focused
            // (type, kind) only — see CInterface.cpp:2358-2406.
            let template = template_target(n);
            let info_header = info_header_for(n);
            let template_v: Option<bool> = if let Some(tt) = template {
                Some(focus_tk == Some(tt))
            } else if let Some(cat) = info_header {
                Some(focus_tk.map(|(c, _)| c == cat).unwrap_or(false))
            } else if is_popup_template(n) {
                // Popup-only image sources — never visible directly.
                Some(false)
            } else {
                None
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
            } else if let Some(v) = warn_v {
                e.set_visible(v);
            } else if let Some(v) = template_v {
                e.set_visible(v);
            } else {
                e.set_visible(true);
            }
        }
    }

    /// Push the constructor's currently-focused-component label and
    /// description into the `it_label1` / `it_label2` dynamic-label
    /// elements + the live preview robot's stats (`struct` / `damage` /
    /// `rcname`) into their respective static labels, so the renderer's
    /// text pass picks them up.
    ///
    /// Port of CInterface.cpp:1869-1884 (it_label1/2),
    /// CInterface.cpp:2330-2354 (struct / damage), and
    /// CInterface.cpp:1822-1827 (rcname). Same per-frame-overwrite
    /// pattern as `apply_top_hud_text`.
    pub fn apply_focused_text(
        &mut self,
        label: &str,
        description: &str,
        structure: i32,
        damage: i32,
        robot_name: &str,
        summ_price: &crate::matrix_game::interface::constructor::UnitPrice,
        focused_price: Option<&crate::matrix_game::interface::constructor::UnitPrice>,
    ) {
        if self.name != "Base" {
            return;
        }
        use crate::matrix_game::config::Resource;
        let structure_text = structure.to_string();
        let damage_text = damage.to_string();
        let summ_titan = summ_price.resources[Resource::Titan as usize].to_string();
        let summ_elec = summ_price.resources[Resource::Electronics as usize].to_string();
        let summ_ener = summ_price.resources[Resource::Energy as usize].to_string();
        let summ_plas = summ_price.resources[Resource::Plasma as usize].to_string();
        // Per-item price (the focused-component cost, no count multiplier).
        // Port of `g_IFaceList->CreateItemPrice(price)` (CInterface.cpp:3146):
        // when nothing is focused or the focus has no cost, the
        // unit-panel statics stay empty.
        let unit_titan = focused_price
            .map(|p| p.resources[Resource::Titan as usize].to_string())
            .unwrap_or_default();
        let unit_elec = focused_price
            .map(|p| p.resources[Resource::Electronics as usize].to_string())
            .unwrap_or_default();
        let unit_ener = focused_price
            .map(|p| p.resources[Resource::Energy as usize].to_string())
            .unwrap_or_default();
        let unit_plas = focused_price
            .map(|p| p.resources[Resource::Plasma as usize].to_string())
            .unwrap_or_default();
        for e in &mut self.elements {
            let new_text: Option<&str> = match e.name.as_str() {
                "it_label1" => Some(label),
                "it_label2" => Some(description),
                "struct" => Some(structure_text.as_str()),
                "damage" => Some(damage_text.as_str()),
                "rcname" => Some(robot_name),
                _ => None,
            };
            let Some(new_text) = new_text else { continue };
            // The focused-part label / description are multi-line —
            // `<br>` substitution in `make_item_replacements` lands
            // `\r\n` inside the string, and the C++ uses `Word_wrap=1`
            // on these statics. Force wrap on so the renderer's
            // multi-line layout kicks in, regardless of what the panel
            // data shipped.
            let force_wrap = matches!(e.name.as_str(), "it_label1" | "it_label2");
            // C++ writes the dynamic caption directly onto
            // `m_StateImages[IFACE_NORMAL].m_Caption` (CInterface.cpp:
            // 1825/1872/2334/2353). If our panel data didn't ship a
            // `LabelsText/<panel>/<elem>_sNormal` entry, our
            // `attach_labels` skips the row and `e.labels` ends up
            // empty — meaning the per-frame caption update has nothing
            // to update. Seed a default Normal-state label here so the
            // text actually renders. Mirrors the C++ behaviour where
            // every state-image carries a (possibly empty) m_Caption
            // from construction.
            if e.labels.is_empty() {
                e.labels.push(ElementLabel {
                    state: ElementState::Normal,
                    text: new_text.to_string(),
                    x: 0.0,
                    y: 0.0,
                    sme_x: 0.0,
                    sme_y: 0.0,
                    align_x: if matches!(e.name.as_str(), "struct" | "damage") {
                        2 // right-aligned numeric readouts (CConstructor.cpp:2334-2353)
                    } else {
                        1 // centered text (rcname)
                    },
                    align_y: 1,
                    wrap: force_wrap,
                    font: "Font.2Small".to_string(),
                    color: [246, 192, 0, 255],
                });
            } else {
                for lbl in &mut e.labels {
                    lbl.text = new_text.to_string();
                    if force_wrap {
                        lbl.wrap = true;
                    }
                }
            }
        }
        // Helper price strings live with `apply_constructor_prices` now;
        // these are kept here just to silence the parameter-unused
        // warning when the price helpers were inlined.
        let _ = (
            summ_titan, summ_elec, summ_ener, summ_plas, unit_titan, unit_elec, unit_ener,
            unit_plas, summ_price, focused_price,
        );
    }

    /// Port of `CIFaceList::CreateSummPrice` (CInterface.cpp:3220-3297) +
    /// `CreateItemPrice` (CInterface.cpp:3146-3193). The C++ creates
    /// IFACE_DYNAMIC_STATIC elements at runtime for each non-zero
    /// resource, sourcing the icon from the named `titan` / `electr` /
    /// `energy` / `plasma` `CIFaceImage` template; the price text then
    /// gets layered onto the `res_summ` / `res_unit` panel via
    /// `SetStateText` with offset `m_titX+25` etc.
    ///
    /// We rebuild the dynamic icons + text labels each frame: dropping
    /// any `_dynprice_*` / `_dynsumm_*` from the previous frame, then
    /// pushing a fresh static for every non-zero entry. The icon image
    /// is borrowed from the `titan` / `electr` / `energy` / `plasma`
    /// IFaceImage element (the `FindImageByName` source); the text
    /// label is appended directly to the dynamic element.
    pub fn apply_constructor_prices(
        &mut self,
        constructor_active: bool,
        focused_price: Option<&crate::matrix_game::interface::constructor::UnitPrice>,
        summ_price: &crate::matrix_game::interface::constructor::UnitPrice,
    ) {
        if self.name != "Base" {
            return;
        }
        // Always wipe last frame's dynamic price entries — they're
        // re-emitted from scratch below if the constructor is active.
        self.elements
            .retain(|e| !e.name.starts_with("_dynsumm_") && !e.name.starts_with("_dynprice_"));
        if !constructor_active {
            return;
        }
        use crate::matrix_game::config::Resource;
        // Resolve the four icon templates up-front. Each is a CIFaceImage
        // element loaded by panel data; its `images[Normal]` carries the
        // atlas sub-rect we want to copy onto each dynamic static.
        // Precomputing avoids borrowing `self.elements` immutably during
        // the mutate-and-push pass below.
        let icons: [Option<(StateImage, f32, f32)>; 4] = {
            let resolve = |name: &str| -> Option<(StateImage, f32, f32)> {
                let e = self.elements.iter().find(|e| e.name == name)?;
                let img = e.images.first()?.as_ref()?.clone();
                // Image elements ship size via `Width` / `Height` (loaded
                // into StateImage.w/h); the element itself has size_x/y
                // == 0 since it's just a template handle.
                let mut w = if e.size_x > 0.0 { e.size_x } else { img.w };
                let mut h = if e.size_y > 0.0 { e.size_y } else { img.h };
                if w <= 0.0 {
                    w = 22.0;
                }
                if h <= 0.0 {
                    h = 22.0;
                }
                Some((img, w, h))
            };
            [
                resolve("titan"),
                resolve("electr"),
                resolve("energy"),
                resolve("plasma"),
            ]
        };
        static LOGGED_ICONS: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !LOGGED_ICONS.swap(true, std::sync::atomic::Ordering::Relaxed) {
            for (i, ico) in icons.iter().enumerate() {
                let label = ["titan", "electr", "energy", "plasma"][i];
                match ico {
                    Some((img, w, h)) => log::info!(
                        "price icon[{}] {} = atlas={:?} (x={},y={},w={},h={}) box={}x{}",
                        i, label, img.tex_path, img.x, img.y, img.w, img.h, w, h
                    ),
                    None => log::warn!("price icon[{}] {} = MISSING", i, label),
                }
            }
            log::info!(
                "summ_price = {:?}, focused_price = {:?}",
                summ_price.resources,
                focused_price.map(|p| p.resources),
            );
        }

        // Helper: push one icon + price text pair as a dynamic static.
        // Mirrors `CreateStaticFromImage` (CInterface.cpp:3170 etc.).
        let push = |elements: &mut Vec<IFaceElement>,
                    name_prefix: &str,
                    res_idx: usize,
                    icon: StateImage,
                    icon_w: f32,
                    icon_h: f32,
                    pos_x: f32,
                    pos_y: f32,
                    price: i32| {
            let mut images: [Option<StateImage>; MAX_STATES] = Default::default();
            images[ElementState::Normal as usize] = Some(icon);
            // Price text label sits to the RIGHT of the icon (the C++
            // uses `m_titX + 25 - elem.x` as `m_SmeX` to push the text
            // ~25 px past the icon). Anchor with align_y=1 (vertical
            // center): the renderer's cell_top = anchor_y - line_h/2,
            // so anchor_y must equal the icon's vertical mid-line for
            // the digits to land on the icon. Since size_y / 2 already
            // produces that anchor (align_y=1), label.y stays 0.
            let label_text = price.to_string();
            let label = ElementLabel {
                state: ElementState::Normal,
                text: label_text,
                x: icon_w + 4.0,
                y: 0.0,
                sme_x: 0.0,
                sme_y: 0.0,
                align_x: 0,
                align_y: 1,
                wrap: false,
                font: "Font.2Small".to_string(),
                color: [246, 192, 0, 255],
            };
            elements.push(IFaceElement {
                name: format!("{name_prefix}{res_idx}"),
                kind: ElementKind::Static,
                id: 0,
                group: 0,
                flags: IFEF_VISIBLE,
                param1: 0.0,
                param2: 0.0,
                i_param: 0,
                pos_x,
                pos_y,
                pos_z: 0.0,
                size_x: icon_w,
                size_y: icon_h,
                images,
                labels: vec![label],
                cur_state: ElementState::Normal,
                def_state: ElementState::Normal,
                hint_template: String::new(),
                hint_offset_x: 0,
                hint_offset_y: 0,
            });
        };

        // ── Summary price row (CInterface.cpp:3220-3297). Anchor
        // shifts left when fewer resources are present so the row
        // stays roughly centred under the build-button.
        let summ_count = (0..4)
            .filter(|i| summ_price.resources[*i] != 0)
            .count();
        let mut x = match summ_count {
            3 => 235.0,
            2 => 250.0,
            _ => 200.0,
        };
        let y = 352.0;
        for r in Resource::ALL {
            let v = summ_price.resources[r as usize];
            if v == 0 {
                continue;
            }
            let Some((img, w, h)) = icons[r as usize].clone() else {
                continue;
            };
            push(
                &mut self.elements,
                "_dynsumm_",
                r as usize,
                img,
                w,
                h,
                x,
                y,
                v,
            );
            // CInterface.cpp:3285 — advance by `s->m_xSize + 31`.
            x += w + 31.0;
        }

        // ── Per-item price row (CInterface.cpp:3146-3193). Always
        // anchored at (22, 243); icons advance by `s->m_xSize + 25`.
        if let Some(unit) = focused_price {
            let mut x = 22.0;
            let y = 243.0;
            for r in Resource::ALL {
                let v = unit.resources[r as usize];
                if v == 0 {
                    continue;
                }
                let Some((img, w, h)) = icons[r as usize].clone() else {
                    continue;
                };
                push(
                    &mut self.elements,
                    "_dynprice_",
                    r as usize,
                    img,
                    w,
                    h,
                    x,
                    y,
                    v,
                );
                x += w + 25.0;
            }
        }
    }

    /// Refresh the dynamic captions on the `Top` panel — the permanent
    /// top-of-screen HUD showing current resource pools + live/limit
    /// robot count. Ports the hint-replacement substitution path at
    /// CInterface.cpp:4439-4462 (`thz` / `enhz1` / `elhz` / `phz` /
    /// `rvhz`), but applied to the Top panel's always-visible
    /// `tit` / `elect` / `energ` / `plasm` / `rval` value labels.
    ///
    /// * `titan` / `elect` / `energy` / `plasma` — current
    ///   `m_Resources[]` values on the player side.
    /// * `(side_robots, max_side_robots)` — current live robot count
    ///   and the cap returned by `GetMaxSideRobots`.
    pub fn apply_top_hud_text(
        &mut self,
        titan: i32,
        elect: i32,
        energy: i32,
        plasma: i32,
        side_robots: i32,
        max_side_robots: i32,
    ) {
        if self.name != "Top" {
            return;
        }
        let rval_text = format!("{}/{}", side_robots, max_side_robots);
        for e in &mut self.elements {
            let new_text: Option<String> = match e.name.as_str() {
                "tit" => Some(titan.to_string()),
                "elect" => Some(elect.to_string()),
                "energ" => Some(energy.to_string()),
                "plasm" => Some(plasma.to_string()),
                "rval" => Some(rval_text.clone()),
                _ => None,
            };
            let Some(new_text) = new_text else { continue };
            for lbl in &mut e.labels {
                lbl.text = new_text.clone();
            }
        }
    }

    /// Refresh the dynamic captions on the `Main` panel when a base or
    /// factory is selected. Port of the per-element caption assignments
    /// at CInterface.cpp:1369-1499 — name / bopis / bresg get the
    /// localised strings from `AllLabels/Buildings`, lives gets the
    /// `"<hp>/<max>"` integer readout (CInterface.cpp:1503).
    pub fn apply_main_building_text(
        &mut self,
        kind: crate::matrix_game::object_building::BuildingType,
        hit_point: f32,
        hit_point_max: f32,
        income_per_minute: i32,
        labels: &crate::matrix_game::config::BuildingLabels,
    ) {
        use crate::matrix_game::object_building::BuildingType;
        if self.name != "Main" {
            return;
        }
        let (name, descr) = match kind {
            BuildingType::Base => (labels.base_name.as_str(), labels.base_descr.as_str()),
            BuildingType::Titan => (labels.titan_name.as_str(), labels.titan_descr.as_str()),
            BuildingType::Electronic => (
                labels.electronics_name.as_str(),
                labels.electronics_descr.as_str(),
            ),
            BuildingType::Energy => (labels.energy_name.as_str(), labels.energy_descr.as_str()),
            BuildingType::Plasma => (labels.plasma_name.as_str(), labels.plasma_descr.as_str()),
            BuildingType::Repair => ("", ""),
        };
        // Port of CInterface.cpp:1485-1486 — `suck.Replace(<resources>,
        // "<Color=247,195,0>N</Color>")`. We emit the same rich-text
        // markup so the renderer's inline `<Color>` tag parser colours
        // the number gold (247,195,0 = `g_Colors[DEF_NORMAL_LBL_COLOR]`).
        let bresg_text = if !labels.res_per.is_empty() {
            let gold = format!("<Color=247,195,0>{}</Color>", income_per_minute);
            labels.res_per.replace("<resources>", &gold)
        } else {
            String::new()
        };
        // Port of `CMatrixBuilding::GetHitPoint() { return m_HitPoint / 10; }`
        // (MatrixObjectBuilding.hpp:274-275). The stored `m_HitPoint` is
        // 10× the displayed value so the engine can track sub-unit
        // damage; the HUD always shows the tenths-rounded integer.
        let shown_hp = (hit_point / 10.0).round() as i32;
        let shown_max = (hit_point_max / 10.0).round() as i32;
        let lives_text = format!("{}/{}", shown_hp, shown_max);

        for e in &mut self.elements {
            let new_text: Option<&str> = match e.name.as_str() {
                "name" => Some(name),
                // `bopis` + `bresg` only show for a base; for factories
                // the matching `IF_FACTORY_RES_INC` branch would use
                // `fresg` (CInterface.cpp:1489). We leave those alone
                // since the factory description lives on the same
                // `bopis` element across all kinds.
                "bopis" => Some(descr),
                "bresg" if matches!(kind, BuildingType::Base) => Some(bresg_text.as_str()),
                "fresg" if !matches!(kind, BuildingType::Base) => Some(bresg_text.as_str()),
                "lives" => Some(lives_text.as_str()),
                _ => None,
            };
            let Some(new_text) = new_text else {
                continue;
            };
            for lbl in &mut e.labels {
                // Keep shadow rows in sync (same text, just different
                // colour / offset) so the drop-shadow under the name
                // and lives still tracks the caption.
                lbl.text = new_text.to_string();
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
        cfg: &crate::matrix_game::interface::constructor::RobotConfig,
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

        // Empty-pylon "?" badge. The C++ shows this character via the
        // heade/weape atlas region (a "?" graphic baked into the
        // sprite) — the atlas content lives in encrypted Lang.dat and
        // doesn't ship in our extracted data, so the copied image
        // arrives blank. Faithful workaround: when the pylon's kind is
        // empty, drop a single-glyph "?" label centred on the pylon
        // so users still get the empty-slot visual cue.
        let pylon_qmark: [(&str, bool); 8] = [
            ("pich", cfg.chassis.kind.is_empty()),
            ("pihu", cfg.hull.unit.kind.is_empty()),
            ("pihe", cfg.head.kind.is_empty()),
            ("pi1", cfg.weapon[0].kind.is_empty()),
            ("pi2", cfg.weapon[1].kind.is_empty()),
            ("pi3", cfg.weapon[2].kind.is_empty()),
            ("pi4", cfg.weapon[3].kind.is_empty()),
            ("pi5", cfg.weapon[4].kind.is_empty()),
        ];
        for (name, is_empty) in pylon_qmark {
            let Some(e) = self.elements.iter_mut().find(|e| e.name == name) else {
                continue;
            };
            // Remove any prior "?" we seeded last frame.
            e.labels.retain(|l| l.text != "?");
            if !is_empty {
                continue;
            }
            e.labels.push(ElementLabel {
                state: ElementState::Normal,
                text: "?".to_string(),
                x: 0.0,
                y: 0.0,
                sme_x: 0.0,
                sme_y: 0.0,
                align_x: 1, // centred
                align_y: 1,
                wrap: false,
                font: "Font.2Big".to_string(),
                color: [246, 192, 0, 255],
            });
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

    // `Image` kind uses a different parameter set (CInterface.cpp:636-646):
    // `TextureFile`, `TexPosX`, `TexPosY`, `TextureWidth`, `TextureHeight`,
    // `Width`, `Height` — no s<State>* keys. Translate that into a
    // single Normal-state `StateImage` so the element can be used as a
    // copy source for `FindImageByName` / `CreateStaticFromImage` ports
    // (`tmd1..4`, `tsm1..4`, etc.). The element itself isn't rendered
    // directly; it's a template the build-stack icon copies from.
    if matches!(kind, ElementKind::Image) && images.iter().all(|i| i.is_none()) {
        if let Some(img) = parse_image_element(stor, rec) {
            images[ElementState::Normal as usize] = Some(img);
        }
    }

    // Skip elements that carry no image anywhere — likely data-only
    // records (ProgressBar anchors, counters) that the renderer
    // doesn't have a hook for yet.
    if images.iter().all(|i| i.is_none()) {
        return None;
    }

    // Port of the per-element `Hint` param parse at CInterface.cpp:
    // 229-235 (Button) / :536-542 (Static). Comma-separated
    // `template,x,y` — missing or empty leaves the hint unset.
    let (hint_template, hint_offset_x, hint_offset_y) =
        parse_hint_param(stor, rec).unwrap_or_default();

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
        labels: Default::default(),
        cur_state: def_state,
        def_state,
        hint_template,
        hint_offset_x,
        hint_offset_y,
    })
}

/// Split a `Hint` param value into `(template, x, y)`. The C++ uses
/// `CWStr::GetStrPar(0, ",")` + `GetIntPar(1, ",")` + `GetIntPar(2, ",")`.
/// We replicate that with a simple comma split + trims. Returns `None`
/// when the param is missing or empty; returns `Some((String::new(),
/// 0, 0))` for malformed values so callers short-circuit on the empty
/// template name.
fn parse_hint_param(stor: &Storage, rec: &str) -> Option<(String, i32, i32)> {
    let raw = stor.block_param(rec, "Hint")?;
    if raw.is_empty() {
        return None;
    }
    let mut it = raw.split(',');
    let template = it.next().unwrap_or("").trim().to_string();
    let x = it
        .next()
        .and_then(|s| s.trim().parse::<i32>().ok())
        .unwrap_or(0);
    let y = it
        .next()
        .and_then(|s| s.trim().parse::<i32>().ok())
        .unwrap_or(0);
    if template.is_empty() {
        return None;
    }
    Some((template, x, y))
}

/// Faithful port of the labels-walking loop in CInterface::Load
/// (CInterface.cpp:657-797). Each element under `if/<panel>` has an
/// optional `Labels` sub-block whose children carry one
/// `Params`/`State`/`Font`/`Color` triple per per-state caption.
/// The text itself comes from `LabelsText/<panel>/<elem>_<state>`.
///
/// The C++ also adds a black DROP SHADOW for a small set of buttons —
/// `cobuild` (CONST_BUILD), `cocan` (CONST_CANCEL), `mm` (MAIN_MENU),
/// `inro` (ENTER_ROBOT), `lero` (LEAVE_ROBOT). For those, every state
/// gets two label rows: the shadow at `(x-1, y-1)` colour `0xFF000000`
/// and the colored row at `(x, y)` (CInterface.cpp:774-792). We
/// replicate that here so the rendered constructor button looks like
/// the original.
fn attach_labels(stor: &Storage, panel_rec: &str, panel_name: &str, elements: &mut [IFaceElement]) {
    let labels_text_rec = stor
        .block_record(panel_rec, "LabelsText")
        .and_then(|lt| stor.block_record(&lt, panel_name));

    // Walk the panel's element children to find each element's `Labels`
    // sub-block. Children live in cols "2" (names) / "3" (records).
    let Some(child_names) = stor.get_buf(panel_rec, "2") else {
        return;
    };
    let Some(child_recs) = stor.get_buf(panel_rec, "3") else {
        return;
    };
    let mut attached = 0_usize;
    for i in 0..child_names.arrays_count().min(child_recs.arrays_count()) {
        let kind = child_names.get_as_wstr(i);
        if kind != "Button" && kind != "Static" {
            continue;
        }
        let elem_rec = child_recs.get_as_wstr(i);
        let elem_name = stor.block_param(&elem_rec, "Name").unwrap_or_default();
        let Some(labels_rec) = stor.block_record(&elem_rec, "Labels") else {
            continue;
        };
        let Some(label_names) = stor.get_buf(&labels_rec, "2") else {
            continue;
        };
        let Some(label_recs) = stor.get_buf(&labels_rec, "3") else {
            continue;
        };
        // Find the destination IFaceElement (load_element may have
        // skipped elements with no images, so this lookup can fail).
        let Some(elem) = elements.iter_mut().find(|e| e.name == elem_name) else {
            continue;
        };
        elem.labels.clear();
        for k in 0..label_names.arrays_count().min(label_recs.arrays_count()) {
            let label_kind = label_names.get_as_wstr(k);
            // The C++ supports both StateStaticLabel and
            // StateDynamicLabel — both go through SetStateLabelParams,
            // they only differ in whether the text is present in
            // `LabelsText` (static) or set later via `SetCaption` /
            // `m_FocusedLabel` flow (dynamic). We attach the static
            // text now; dynamic captions get filled in by
            // `apply_focused_text` etc. each frame.
            if label_kind != "StateStaticLabel" && label_kind != "StateDynamicLabel" {
                continue;
            }
            let label_rec = label_recs.get_as_wstr(k);
            let params = stor.block_param(&label_rec, "Params").unwrap_or_default();
            let state_str = stor.block_param(&label_rec, "State").unwrap_or_default();
            let font = stor.block_param(&label_rec, "Font").unwrap_or_default();
            let color_str = stor.block_param(&label_rec, "Color").unwrap_or_default();
            let parts: Vec<i32> = params
                .split(',')
                .filter_map(|s| s.trim().parse::<i32>().ok())
                .collect();
            if parts.len() < 7 {
                continue;
            }
            let (x, y) = (parts[0] as f32, parts[1] as f32);
            let (sme_x, sme_y) = (parts[2] as f32, parts[3] as f32);
            let (align_x, align_y) = (parts[4], parts[5]);
            let wrap = parts[6] != 0;
            let state = match state_str.as_str() {
                "sNormal" => ElementState::Normal,
                "sFocused" => ElementState::Focused,
                "sPressed" => ElementState::Pressed,
                "sDisabled" => ElementState::Disabled,
                _ => continue,
            };
            // Decode `A,R,G,B` (the C++ packs aRGB via the same field
            // order at CInterface.cpp:691-696).
            let color = parse_argb(&color_str).unwrap_or([255, 255, 255, 255]);
            // Resolve the static text from `LabelsText/<panel>/<elem>_<state>`.
            // Dynamic labels start empty.
            let text = labels_text_rec
                .as_ref()
                .and_then(|rec| {
                    let key = format!("{elem_name}_{state_str}");
                    stor.block_param(rec, &key)
                })
                .unwrap_or_default();
            // Drop-shadow special case (CInterface.cpp:774-792). Black
            // shadow at (x-1, y-1) renders BEFORE the colored row.
            // Applies to the small set of buttons the C++ hardcodes.
            let with_shadow = matches!(
                elem_name.as_str(),
                "cobuild" | "cocan" | "mm" | "inro" | "lero"
            );
            if with_shadow && !text.is_empty() {
                elem.labels.push(ElementLabel {
                    state,
                    text: text.clone(),
                    x: x - 1.0,
                    y: y - 1.0,
                    sme_x,
                    sme_y,
                    align_x,
                    align_y,
                    wrap,
                    font: font.clone(),
                    color: [0, 0, 0, 255],
                });
            }
            elem.labels.push(ElementLabel {
                state,
                text,
                x,
                y,
                sme_x,
                sme_y,
                align_x,
                align_y,
                wrap,
                font,
                color,
            });
            attached += 1;
        }
    }
    log::info!(
        "iface labels: panel={panel_name} attached={attached} label rows"
    );
}

/// Decode the C++ aRGB packed `Color` param. Format: `A,R,G,B` ints
/// 0-255. Returns None on malformed input.
fn parse_argb(s: &str) -> Option<[u8; 4]> {
    let parts: Vec<u8> = s
        .split(',')
        .filter_map(|p| p.trim().parse::<u8>().ok())
        .collect();
    if parts.len() < 4 {
        return None;
    }
    Some([parts[1], parts[2], parts[3], parts[0]])
}

/// Port of the `Image` element parse branch at CInterface.cpp:635-650.
/// `Image` records carry `TextureFile`/`TexPosX`/`TexPosY`/`TextureWidth`/
/// `TextureHeight`/`Width`/`Height` instead of the `s<State>*` quartet
/// `Static`/`Button` use. A missing `TextureFile` skips the element.
fn parse_image_element(stor: &Storage, rec: &str) -> Option<StateImage> {
    let tex_path = stor.block_param(rec, "TextureFile")?;
    if tex_path.is_empty() {
        return None;
    }
    let tex_pos_x = parse_f32(stor, rec, "TexPosX").unwrap_or(0.0);
    let tex_pos_y = parse_f32(stor, rec, "TexPosY").unwrap_or(0.0);
    let tex_w = parse_f32(stor, rec, "TextureWidth").unwrap_or(512.0);
    let tex_h = parse_f32(stor, rec, "TextureHeight").unwrap_or(512.0);
    let w = parse_f32(stor, rec, "Width").unwrap_or(0.0);
    let h = parse_f32(stor, rec, "Height").unwrap_or(0.0);
    Some(StateImage {
        x: tex_pos_x,
        y: tex_pos_y,
        w,
        h,
        tex_w,
        tex_h,
        tex_path,
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
