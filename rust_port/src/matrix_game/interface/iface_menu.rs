//! Port of `CIFaceMenu` (Interface/CIFaceMenu.{cpp,h}) — the popup
//! menu shown when the user **right-clicks a constructor pylon**.
//!
//! In the C++ original (`CIFaceButton::OnMouseRBDown` lines 188-312),
//! pressing RMB on a pylon opens this popup at fixed offsets from the
//! Base panel origin. The popup lists each available kind of the
//! pylon's category as a row item; clicking an item commits the
//! selection via `SuperDjeans` and closes the popup.
//!
//! ## Differences from C++
//!
//! * Menu items render as **text labels** like the C++ (the catcher
//!   pass of CIFaceMenu.cpp:312-341 lives in renderer.rs, drawn with
//!   the AFT `Font.2Ranger` glyphs); `SMenuItemText.text` carries the
//!   localised captions parsed from the `iw/ihu/ihe/ich{N}text`
//!   element labels below.
//! * The C++ uses a separate `m_MenuGraphics` panel (`if/CIFaceMenu`)
//!   for the popup chrome (border / cursor). We render the popup
//!   inline using the constructor template buttons, drawn on top of
//!   the Base panel.

use crate::matrix_game::config::RobotUnitKind;
use crate::matrix_game::interface::constructor::RobotConfig;
use crate::matrix_game::object_robot::{RobotUnitType, MAX_WEAPON_CNT};

/// Identity of the pylon that opened the popup. Mirrors `EMenuParent`
/// (CIFaceMenu.h:59-69). The pilon index for weapons (0..=4) is
/// extracted from this so `SuperDjeans` knows which slot to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EMenuParent {
    PylonChassis,
    PylonHull,
    PylonHead,
    PylonWeapon(i32), // 0..=4
}

impl EMenuParent {
    pub fn unit_type(self) -> RobotUnitType {
        match self {
            EMenuParent::PylonChassis => RobotUnitType::Chassis,
            EMenuParent::PylonHull => RobotUnitType::Armor,
            EMenuParent::PylonHead => RobotUnitType::Head,
            EMenuParent::PylonWeapon(_) => RobotUnitType::Weapon,
        }
    }

    pub fn pilon(self) -> i32 {
        match self {
            EMenuParent::PylonWeapon(p) => p,
            _ => 0,
        }
    }
}

/// One selectable item in the popup. Port of `SMenuItemText`
/// (CIFaceMenu.h:70-77) — the C++ holds `text` (the localised row
/// caption) + a `color` (DEFAULT_LABELS_COLOR or NERES_LABELS_COLOR
/// flipped by the resource-affordability pass at
/// CIFaceButton.cpp:197-309).
#[derive(Debug, Clone)]
pub struct SMenuItemText {
    pub kind: RobotUnitKind,
    /// Localised row caption — one entry per popup row, pre-populated
    /// from `iw{N}text` / `ihu{N}text` / `ihe{N}text` / `ich{N}text`
    /// element captions (CInterface.cpp:715-772) or `AllLabels/Base/none`
    /// for the empty row (MatrixGame.cpp:521-523).
    pub text: String,
    /// Port of the per-item color flip at CIFaceButton.cpp:197-309.
    /// `true` = `DEFAULT_LABELS_COLOR`, `false` = `NERES_LABELS_COLOR`
    /// (set by `IsEnoughResourcesForThisPieceOfShit`). Defaults to
    /// `true`; the caller flips it after building the popup with the
    /// player side's bank.
    pub affordable: bool,
}

/// Active popup-menu state. Stored on `IFaceList::popup`. Mirrors the
/// `m_*` instance fields on the C++ `CIFaceMenu` class
/// (CIFaceMenu.h:81-121).
#[derive(Debug, Clone)]
pub struct CIFaceMenu {
    /// Port of `m_InterfaceParent` (CIFaceMenu.h:97).
    pub parent: EMenuParent,
    pub items: Vec<SMenuItemText>,
    /// Top-left of the popup in design-space coords *relative to the
    /// Base panel's resolved origin*. Mirrors the C++ `g_IFaceList->m_BaseX +
    /// offset_X` pattern at CIFaceButton.cpp:203, :220, etc.
    pub design_x: f32,
    pub design_y: f32,
    /// Item slot dimensions (design pixels). Each row spans `item_w`
    /// wide and `item_h` tall; rows stack vertically.
    pub item_w: f32,
    pub item_h: f32,
    /// Hovered item index, or `None`. Drives the highlight border
    /// and tells `commit()` which kind to select on click. Port of
    /// the cursor-tracking inside `CalcSelectedItem` (CIFaceMenu.cpp).
    pub hovered: Option<usize>,
    /// Port of `m_CurMenuPos` (CIFaceMenu.h:96) — index of the row
    /// matching the currently-equipped component, drawn with the
    /// "cursik" arrow indicator at CIFaceMenu.cpp:94-96.
    pub current_pos: Option<usize>,
    /// Port of `m_Caller` (CIFaceMenu.h:99) — name of the element
    /// (pylon button) that opened this popup.
    pub caller_name: String,
    pub saved_config: Option<RobotConfig>,
    pub previewed: Option<usize>,
}

impl CIFaceMenu {
    /// Build the chassis popup (5 kinds, no empty row). Mirrors the
    /// chassis branch of `CIFaceButton::OnMouseRBDown` at
    /// CIFaceButton.cpp:301-312 with text loaded from the per-kind
    /// `ich{N}text_sNormal` element captions
    /// (CInterface.cpp:763-772).
    pub fn for_chassis(base: Option<&super::CInterface>, none_label: &str) -> Self {
        let items = (1..=5)
            .map(|n| SMenuItemText {
                kind: RobotUnitKind(n),
                text: lookup_text(base, &format!("ich{n}text"))
                    .unwrap_or_else(|| format!("Chas{n}")),
                affordable: true,
            })
            .collect();
        Self::new_at(
            EMenuParent::PylonChassis,
            items,
            321.0,
            231.0,
            CHASSIS_MENU_WIDTH,
            none_label,
        )
    }

    /// Hull/armor popup (6 kinds, no empty). Order from
    /// CIFaceButton.cpp:283-300 (`[6, 1, 2, 3, 4, 5]`); text from
    /// the `ihu{N}text_sNormal` element captions
    /// (CInterface.cpp:737-748).
    pub fn for_hull(base: Option<&super::CInterface>, none_label: &str) -> Self {
        let order: [i32; 6] = [6, 1, 2, 3, 4, 5];
        let items = order
            .iter()
            .map(|&n| SMenuItemText {
                kind: RobotUnitKind(n),
                text: lookup_text(base, &format!("ihu{n}text"))
                    .unwrap_or_else(|| format!("Hull{n}")),
                affordable: true,
            })
            .collect();
        Self::new_at(
            EMenuParent::PylonHull,
            items,
            321.0,
            148.0,
            HULL_MENU_WIDTH,
            none_label,
        )
    }

    /// Head popup (4 kinds + empty `none` row at index 0). Per
    /// CIFaceButton.cpp:272-282 + the empty-row pre-assignment at
    /// MatrixGame.cpp:523.
    pub fn for_head(base: Option<&super::CInterface>, none_label: &str) -> Self {
        let mut items = vec![SMenuItemText {
            kind: RobotUnitKind::UNKNOWN,
            text: none_label.to_string(),
            affordable: true,
        }];
        for n in 1..=4 {
            items.push(SMenuItemText {
                kind: RobotUnitKind(n),
                text: lookup_text(base, &format!("ihe{n}text"))
                    .unwrap_or_else(|| format!("Head{n}")),
                affordable: true,
            });
        }
        Self::new_at(
            EMenuParent::PylonHead,
            items,
            315.0,
            76.0,
            HEAD_MENU_WIDTH,
            none_label,
        )
    }

    /// Common-weapon pylon popup (pylons 1..4). The C++ kind remapping
    /// at CIFaceButton.cpp:191-196 yields the kind sequence
    /// `[1, 2, 3, 4, 6, 8, 9, 10]` for 8 rows; index 0 is the `none`
    /// empty row (MatrixGame.cpp:522). Text comes from
    /// `iw{N}text_sNormal` (CInterface.cpp:715-736).
    pub fn for_weapon_normal(
        base: Option<&super::CInterface>,
        none_label: &str,
        pilon_idx: i32,
    ) -> Self {
        // (kind, label_idx) — label_idx is the source `iw{label_idx}text`.
        // The C++ remaps weapon-popup-row → weapon-kind so kind 5
        // (mortar) and 7 (bomb) skip to the extern popup.
        let rows: [(i32, i32); 8] = [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (6, 6),
            (8, 8),
            (9, 9),
            (10, 10),
        ];
        let mut items = vec![SMenuItemText {
            kind: RobotUnitKind::UNKNOWN,
            text: none_label.to_string(),
            affordable: true,
        }];
        for &(kind, lbl) in &rows {
            items.push(SMenuItemText {
                kind: RobotUnitKind(kind),
                text: lookup_text(base, &format!("iw{lbl}text"))
                    .unwrap_or_else(|| format!("Weap{kind}")),
                affordable: true,
            });
        }
        // Anchor coords mirror CIFaceButton.cpp:203/220/237/255.
        let (dx, dy) = match pilon_idx {
            0 => (242.0, 155.0),
            1 => (389.0, 155.0),
            2 => (242.0, 135.0),
            3 => (389.0, 135.0),
            _ => (242.0, 155.0),
        };
        Self::new_at(
            EMenuParent::PylonWeapon(pilon_idx),
            items,
            dx,
            dy,
            WEAPON_MENU_WIDTH,
            none_label,
        )
    }

    /// Extra-pylon popup (pylon 5). Mortar (kind 5) + Bomb (kind 7) +
    /// empty `none` row. Anchor from CIFaceButton.cpp:270; row text
    /// from `iw5text_sNormal` (mortar) and `iw7text_sNormal` (bomb)
    /// per CInterface.cpp:723-730.
    pub fn for_weapon_extern(base: Option<&super::CInterface>, none_label: &str) -> Self {
        let items = vec![
            SMenuItemText {
                kind: RobotUnitKind::UNKNOWN,
                text: none_label.to_string(),
                affordable: true,
            },
            SMenuItemText {
                kind: RobotUnitKind::WEAPON_MORTAR,
                text: lookup_text(base, "iw5text").unwrap_or_else(|| "Mortar".into()),
                affordable: true,
            },
            SMenuItemText {
                kind: RobotUnitKind::WEAPON_BOMB,
                text: lookup_text(base, "iw7text").unwrap_or_else(|| "Bomb".into()),
                affordable: true,
            },
        ];
        Self::new_at(
            EMenuParent::PylonWeapon(4),
            items,
            389.0,
            76.0,
            WEAPON_MENU_WIDTH,
            none_label,
        )
    }

    /// Common helper — assembles a popup at a fixed (design_x, design_y)
    /// with the menu width pre-shifted by `+CURSIK_WIDTH` per the C++
    /// `width += CURSIK_WIDTH` step (CIFaceMenu.cpp:91).
    fn new_at(
        parent: EMenuParent,
        items: Vec<SMenuItemText>,
        design_x: f32,
        design_y: f32,
        menu_width: f32,
        _none_label: &str,
    ) -> Self {
        Self {
            parent,
            items,
            design_x,
            design_y,
            // C++ stores `WIDTH` (the param to CreateMenu) in
            // CIFaceMenu.h. Total popup width is then
            // `WIDTH + CURSIK_WIDTH + LEFTLINE_WIDTH + RIGHTLINE_WIDTH`.
            // We carry the `menu_width` here and compute the total via
            // [`Self::total_w`].
            item_w: menu_width,
            item_h: UNIT_HEIGHT,
            hovered: None,
            current_pos: None,
            caller_name: String::new(),
            saved_config: None,
            previewed: None,
        }
    }

    /// Per-`CIFaceMenu.h` constants. `UNIT_HEIGHT` is the row height;
    /// the chrome wraps the items area with `TOPLINE_HEIGHT` above and
    /// `BOTTOMLINE_HEIGHT` below — the total popup height is
    /// `TOPLINE_HEIGHT + items*UNIT_HEIGHT + BOTTOMLINE_HEIGHT`
    /// (CIFaceMenu.cpp:88-92).
    /// Items area starts `ITEMS_TOP` (=11) below the popup top;
    /// catcher rect spans `(w-8) × UNIT_HEIGHT` per row
    /// (CIFaceMenu.cpp:316-319).
    pub const TOPLINE_HEIGHT: f32 = 18.0;
    pub const BOTTOMLINE_HEIGHT: f32 = 22.0;
    pub const ITEMS_TOP: f32 = 11.0;
    pub const LEFT_SPACE: f32 = 7.0;
    pub const CURSIK_WIDTH: f32 = 7.0;
    pub const LEFTLINE_WIDTH: f32 = 13.0;
    pub const RIGHTLINE_WIDTH: f32 = 18.0;
    pub const TOPLEFT_WIDTH: f32 = 13.0;
    pub const TOPRIGHT_WIDTH: f32 = 18.0;
    pub const BOTTOMLEFT_WIDTH: f32 = 14.0;
    pub const BOTTOMRIGHT_WIDTH: f32 = 13.0;
    pub const TOPLEFT_HEIGHT: f32 = 18.0;
    pub const BOTTOMLEFT_HEIGHT: f32 = 22.0;
    pub const BOTTOMRIGHT_HEIGHT: f32 = 21.0;
    pub const CATCHER_RIGHT_INSET: f32 = 8.0;

    /// Total popup width: `WIDTH + CURSIK_WIDTH + LEFTLINE_WIDTH +
    /// RIGHTLINE_WIDTH`. Port of CIFaceMenu.cpp:91-92.
    pub fn total_w(&self) -> f32 {
        self.item_w + Self::CURSIK_WIDTH + Self::LEFTLINE_WIDTH + Self::RIGHTLINE_WIDTH
    }

    /// Total popup height: `TOPLINE_HEIGHT + items*UNIT_HEIGHT +
    /// BOTTOMLINE_HEIGHT`. Port of CIFaceMenu.cpp:88-90.
    pub fn total_h(&self) -> f32 {
        Self::TOPLINE_HEIGHT + self.item_h * self.items.len() as f32 + Self::BOTTOMLINE_HEIGHT
    }

    /// Y offset (in design space) where the items area begins, relative
    /// to the popup top-left. Matches the C++ catcher base offset
    /// `y + 11 + UNIT_HEIGHT*i` (CIFaceMenu.cpp:316).
    pub fn items_top(&self) -> f32 {
        Self::ITEMS_TOP
    }

    /// Hit-test the popup at design-space cursor coords (relative to
    /// the Base panel origin). Returns the item index under the cursor.
    pub fn hit_test_design(&self, dx: f32, dy: f32) -> Option<usize> {
        if dx < self.design_x || dx >= self.design_x + self.total_w() {
            return None;
        }
        let items_y0 = self.design_y + self.items_top();
        if dy < items_y0 {
            return None;
        }
        let row = ((dy - items_y0) / self.item_h).floor() as i32;
        if row < 0 || (row as usize) >= self.items.len() {
            return None;
        }
        Some(row as usize)
    }

    /// Total popup rect in design-space (relative to Base panel
    /// origin), including chrome. Drives the renderer + click-outside.
    pub fn rect_design(&self) -> [f32; 4] {
        [self.design_x, self.design_y, self.total_w(), self.total_h()]
    }

    /// True when the design-space cursor is inside the popup's rect
    /// (used to decide whether a click outside should close it).
    pub fn contains_design(&self, dx: f32, dy: f32) -> bool {
        let r = self.rect_design();
        dx >= r[0] && dx < r[0] + r[2] && dy >= r[1] && dy < r[1] + r[3]
    }

    /// Port of the per-item affordability passes at CIFaceButton.cpp:
    /// 190-310. After building a popup, walk every non-empty item and
    /// flip `affordable` based on `RobotBuilder::is_enough_resources_for_pick`.
    /// Empty / "remove" rows (kind 0) stay `true` — they cost nothing
    /// to apply.
    pub fn refresh_affordability(
        &mut self,
        builder: &super::constructor::RobotBuilder,
        side_bank: &[i32; 4],
    ) {
        let ty = self.parent.unit_type();
        let pilon = self.parent.pilon();
        for item in &mut self.items {
            if item.kind.is_empty() {
                item.affordable = true;
                continue;
            }
            item.affordable = builder.is_enough_resources_for_pick(pilon, ty, item.kind, side_bank);
        }
    }

    /// Port of the cursik positioning at CIFaceMenu.cpp:94-96. Picks
    /// the row whose `kind` matches the currently-equipped component
    /// for this pylon — the C++ uses `GetIndexFromTK` to derive the
    /// index, then renders the cursik arrow at that row.
    pub fn refresh_current_pos(&mut self, builder: &super::constructor::RobotBuilder) {
        let cfg = builder.cfg();
        let equipped = match self.parent {
            EMenuParent::PylonChassis => cfg.chassis.kind,
            EMenuParent::PylonHull => cfg.hull.unit.kind,
            EMenuParent::PylonHead => cfg.head.kind,
            EMenuParent::PylonWeapon(p) if (p as usize) < MAX_WEAPON_CNT => {
                cfg.weapon[p as usize].kind
            }
            _ => return,
        };
        self.current_pos = self.items.iter().position(|item| item.kind == equipped);
    }

    /// Port of `CIFaceMenu::CreateMenu` (CIFaceMenu.cpp:62+) wrap-up:
    /// stores the pylon name that opened the popup so closing the menu
    /// (RemoteUnFocusElement guards) and click-routing can dispatch
    /// back to the right element.
    pub fn set_caller(&mut self, caller_name: impl Into<String>) {
        self.caller_name = caller_name.into();
    }

    pub fn set_saved_config(&mut self, cfg: RobotConfig) {
        self.saved_config = Some(cfg);
    }
}

#[allow(dead_code)]
fn chas_name(n: i32) -> &'static str {
    match n {
        1 => "chas1",
        2 => "chas2",
        3 => "chas3",
        4 => "chas4",
        5 => "chas5",
        _ => "chas1",
    }
}

#[allow(dead_code)]
fn hull_name(n: i32) -> &'static str {
    match n {
        1 => "hull1",
        2 => "hull2",
        3 => "hull3",
        4 => "hull4",
        5 => "hull5",
        6 => "hull6",
        _ => "hull1",
    }
}

#[allow(dead_code)]
fn head_name(n: i32) -> &'static str {
    match n {
        1 => "head1",
        2 => "head2",
        3 => "head3",
        4 => "head4",
        _ => "head1",
    }
}

#[allow(dead_code)]
fn weap_name(n: i32) -> &'static str {
    match n {
        1 => "weap1",
        2 => "weap2",
        3 => "weap3",
        4 => "weap4",
        5 => "weap5",
        6 => "weap6",
        7 => "weap7",
        8 => "weap8",
        9 => "weap9",
        10 => "weap10",
        _ => "weap1",
    }
}

/// Map a clicked pylon name to the popup that should open. Returns
/// `None` for non-pylon names. The Base panel + the localised "none"
/// label come from the caller (form_game) so we can stay free of
/// global state in the popup module.
pub fn popup_for_pylon(
    name: &str,
    base: Option<&super::CInterface>,
    none_label: &str,
) -> Option<CIFaceMenu> {
    match name {
        "pich" => Some(CIFaceMenu::for_chassis(base, none_label)),
        "pihu" => Some(CIFaceMenu::for_hull(base, none_label)),
        "pihe" => Some(CIFaceMenu::for_head(base, none_label)),
        "pi1" => Some(CIFaceMenu::for_weapon_normal(base, none_label, 0)),
        "pi2" => Some(CIFaceMenu::for_weapon_normal(base, none_label, 1)),
        "pi3" => Some(CIFaceMenu::for_weapon_normal(base, none_label, 2)),
        "pi4" => Some(CIFaceMenu::for_weapon_normal(base, none_label, 3)),
        "pi5" => Some(CIFaceMenu::for_weapon_extern(base, none_label)),
        _ => None,
    }
}

/// Per-CIFaceMenu.h constants accessible to the renderer + the popup
/// factory functions above.
pub const UNIT_HEIGHT: f32 = 19.0;

/// Russian-build menu widths (CIFaceMenu.h:47-50). The English-build
/// variant uses different numbers; we follow the Russian build since
/// our forms.pkg ships the Russian resource pack.
const WEAPON_MENU_WIDTH: f32 = 70.0;
const HULL_MENU_WIDTH: f32 = 90.0;
const HEAD_MENU_WIDTH: f32 = 100.0;
const CHASSIS_MENU_WIDTH: f32 = 90.0;

/// Look up a localised popup-row caption from a Base-panel element's
/// attached label. Mirrors the C++ catch-all that copies
/// `iw{N}text_sNormal` / `ihu{N}text_sNormal` / `ihe{N}text_sNormal` /
/// `ich{N}text_sNormal` text into the per-popup arrays
/// (CInterface.cpp:715-772). Returns `None` when the element is
/// missing or has no attached label — caller falls back to a stub.
fn lookup_text(base: Option<&super::CInterface>, elem_name: &str) -> Option<String> {
    use super::iface_element::ElementState;
    let base = base?;
    let elem = base.elements.iter().find(|e| e.name == elem_name)?;
    elem.labels
        .iter()
        .find(|l| matches!(l.state, ElementState::Normal))
        .map(|l| l.text.clone())
}

/// Decode the chosen menu kind for a `EMenuParent::PylonWeapon` —
/// caller may need the slot index too. Just a pass-through to keep
/// the call site clean.
#[allow(dead_code)]
pub fn weapon_target_pilon(parent: EMenuParent) -> i32 {
    match parent {
        EMenuParent::PylonWeapon(p) if (p as usize) < MAX_WEAPON_CNT => p,
        _ => 0,
    }
}
