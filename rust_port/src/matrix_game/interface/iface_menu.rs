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
//! * The C++ renders each menu item as a **text label** (e.g.
//!   "Machinegun", "Cannon", …) loaded from
//!   `Labels.AllLabels.Items.*` and drawn with the SR2 "Rangers"
//!   text-rasteriser. Our Rust port doesn't yet have a font/glyph
//!   renderer, so we use **icon items** instead — each row is the
//!   matching atlas sprite (`chas{N}` / `hull{N}` / `head{N}` / `weap{N}`)
//!   already loaded by the Base panel. Visually different from the
//!   original but functionally equivalent: hover highlights, click
//!   commits the kind. Text labels can land later when the font
//!   subsystem does.
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

/// One selectable item in the popup. The icon is the *element name*
/// in the loaded Base panel (e.g. `"chas2"`). The renderer copies
/// that element's images onto the popup-row hit slot for display.
#[derive(Debug, Clone, Copy)]
pub struct SMenuItemText {
    pub kind: RobotUnitKind,
    /// Element name on the Base panel whose images we render for
    /// this row (e.g. `"chas1"`, `"hull3"`, `"weap6"`).
    pub icon_element: &'static str,
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
    /// Build the chassis popup (5 kinds: Pneumatic..Antigravity).
    /// Mirrors the chassis branch of `CIFaceButton::OnMouseRBDown` at
    /// CIFaceButton.cpp:301-312 (anchor `+321, +231`).
    pub fn for_chassis() -> Self {
        let items = (1..=5)
            .map(|n| SMenuItemText {
                kind: RobotUnitKind(n),
                icon_element: chas_name(n),
                affordable: true,
            })
            .collect();
        Self {
            parent: EMenuParent::PylonChassis,
            items,
            design_x: 321.0,
            design_y: 231.0,
            // CIFaceMenu.h:50 — Russian build CHASSIS_MENU_WIDTH=90 +
            // chrome (CURSIK_WIDTH+LEFT_SPACE+LEFT/RIGHT line ≈ 18) =
            // ~108. UNIT_HEIGHT=19.
            item_w: 108.0,
            item_h: 19.0,
            hovered: None,
            current_pos: None,
            caller_name: String::new(),
            saved_config: None,
            previewed: None,
        }
    }

    /// Hull/armor popup (6 kinds, ordered by original C++ at
    /// CIFaceButton.cpp:283-300: index 1 → kind 6, index 2 → kind 1, … 6 → 5).
    /// We follow that exact ordering.
    pub fn for_hull() -> Self {
        let order: [i32; 6] = [6, 1, 2, 3, 4, 5];
        let items = order
            .iter()
            .map(|&n| SMenuItemText {
                kind: RobotUnitKind(n),
                icon_element: hull_name(n),
                affordable: true,
            })
            .collect();
        Self {
            parent: EMenuParent::PylonHull,
            items,
            design_x: 321.0,
            design_y: 148.0,
            // HULL_MENU_WIDTH=90 + chrome ≈ 108.
            item_w: 108.0,
            item_h: 19.0,
            hovered: None,
            current_pos: None,
            caller_name: String::new(),
            saved_config: None,
            previewed: None,
        }
    }

    /// Head popup (4 kinds + empty). The C++ menu_head_items includes
    /// the empty/none slot at index 0 (CIFaceButton.cpp:272-282).
    pub fn for_head() -> Self {
        let mut items = vec![SMenuItemText {
            kind: RobotUnitKind::UNKNOWN,
            icon_element: "heade",
            affordable: true,
        }];
        for n in 1..=4 {
            items.push(SMenuItemText {
                kind: RobotUnitKind(n),
                icon_element: head_name(n),
                affordable: true,
            });
        }
        Self {
            parent: EMenuParent::PylonHead,
            items,
            design_x: 315.0,
            design_y: 76.0,
            // HEAD_MENU_WIDTH=100 + chrome ≈ 118.
            item_w: 118.0,
            item_h: 19.0,
            hovered: None,
            current_pos: None,
            caller_name: String::new(),
            saved_config: None,
            previewed: None,
        }
    }

    /// Common-weapon pylon popup (pylons 1..4). The C++ skips kind 5
    /// (Mortar) and 7 (Bomb) — those are extra-pylon-only. The
    /// remapping at CIFaceButton.cpp:191-196 yields the kind sequence
    /// `[1, 2, 3, 4, 6, 8, 9, 10]` for 8 menu rows.
    pub fn for_weapon_normal(pilon_idx: i32) -> Self {
        let kinds: [i32; 8] = [1, 2, 3, 4, 6, 8, 9, 10];
        let mut items = vec![SMenuItemText {
            kind: RobotUnitKind::UNKNOWN,
            icon_element: "weape",
            affordable: true,
        }];
        for &k in &kinds {
            items.push(SMenuItemText {
                kind: RobotUnitKind(k),
                icon_element: weap_name(k),
                affordable: true,
            });
        }
        // Anchor coords mirror CIFaceButton.cpp:203/220/237/255 —
        // pylons 1+3 land at +242, pylons 2+4 at +389, vertical
        // varies. For simplicity we anchor uniformly above the pylon
        // here; precise positioning lands when the popup chrome ports.
        let (dx, dy) = match pilon_idx {
            0 => (242.0, 155.0),
            1 => (389.0, 155.0),
            2 => (242.0, 135.0),
            3 => (389.0, 135.0),
            _ => (242.0, 155.0),
        };
        Self {
            parent: EMenuParent::PylonWeapon(pilon_idx),
            items,
            design_x: dx,
            design_y: dy,
            // WEAPON_MENU_WIDTH=70 + chrome ≈ 88.
            item_w: 88.0,
            item_h: 19.0,
            hovered: None,
            current_pos: None,
            caller_name: String::new(),
            saved_config: None,
            previewed: None,
        }
    }

    /// Extra-pylon popup (pylon 5). Only Mortar / Bomb (+ empty). Anchor
    /// from CIFaceButton.cpp:270.
    pub fn for_weapon_extern() -> Self {
        let items = vec![
            SMenuItemText {
                kind: RobotUnitKind::UNKNOWN,
                icon_element: "weape",
                affordable: true,
            },
            SMenuItemText {
                kind: RobotUnitKind::WEAPON_MORTAR,
                icon_element: weap_name(5),
                affordable: true,
            },
            SMenuItemText {
                kind: RobotUnitKind::WEAPON_BOMB,
                icon_element: weap_name(7),
                affordable: true,
            },
        ];
        Self {
            parent: EMenuParent::PylonWeapon(4),
            items,
            design_x: 389.0,
            design_y: 76.0,
            item_w: 88.0,
            item_h: 19.0,
            hovered: None,
            current_pos: None,
            caller_name: String::new(),
            saved_config: None,
            previewed: None,
        }
    }

    /// Hit-test the popup at design-space cursor coords (relative to
    /// the Base panel origin). Returns the item index under the cursor.
    pub fn hit_test_design(&self, dx: f32, dy: f32) -> Option<usize> {
        if dx < self.design_x || dx >= self.design_x + self.item_w {
            return None;
        }
        if dy < self.design_y {
            return None;
        }
        let row = ((dy - self.design_y) / self.item_h).floor() as i32;
        if row < 0 || (row as usize) >= self.items.len() {
            return None;
        }
        Some(row as usize)
    }

    /// Total popup rect in design-space (relative to Base panel
    /// origin). Drives the renderer.
    pub fn rect_design(&self) -> [f32; 4] {
        [
            self.design_x,
            self.design_y,
            self.item_w,
            self.item_h * self.items.len() as f32,
        ]
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

fn head_name(n: i32) -> &'static str {
    match n {
        1 => "head1",
        2 => "head2",
        3 => "head3",
        4 => "head4",
        _ => "head1",
    }
}

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
/// `None` for non-pylon names.
pub fn popup_for_pylon(name: &str) -> Option<CIFaceMenu> {
    match name {
        "pich" => Some(CIFaceMenu::for_chassis()),
        "pihu" => Some(CIFaceMenu::for_hull()),
        "pihe" => Some(CIFaceMenu::for_head()),
        "pi1" => Some(CIFaceMenu::for_weapon_normal(0)),
        "pi2" => Some(CIFaceMenu::for_weapon_normal(1)),
        "pi3" => Some(CIFaceMenu::for_weapon_normal(2)),
        "pi4" => Some(CIFaceMenu::for_weapon_normal(3)),
        "pi5" => Some(CIFaceMenu::for_weapon_extern()),
        _ => None,
    }
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
