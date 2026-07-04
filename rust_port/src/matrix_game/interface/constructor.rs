//! Port of `CConstructor` + `CConstructorPanel`
//! (MatrixGame/src/Interface/CConstructor.{cpp,h}) — the factory-UI
//! state machine that drives robot configuration + build-stack
//! production.
//!
//! The C++ splits this across two classes: `CConstructor` owns the
//! live preview robot and the per-side build state; `CConstructorPanel`
//! owns the focused UI element + the preset history + price display.
//! We fold them into one `RobotBuilder` struct since both halves are
//! per-side and the separation in C++ is mostly historical
//! (CConstructor handles the D3D preview; the Rust preview runs
//! inside the existing robots renderer and doesn't need a sibling class).

use crate::matrix_game::config;
use crate::matrix_game::config::{RobotUnitKind, MAX_RESOURCES};
use crate::matrix_game::logic::Rnd;
use crate::matrix_game::map::default_weapon_matrix;
use crate::matrix_game::object_robot::{RobotUnitType, MAX_WEAPON_CNT, MR_MAXUNIT};
// ArmorUnit, RobotConfig, Unit, UnitPrice, WeaponUnit all live in the bottom
// half of this file (see the "Robot-composition types" section).

/// Port of `PRESETS` (CConstructor.h:169). The shipped build hardcodes
/// 1 active preset slot per side; `g_ConfigHistory` below carries the
/// ring of previously-built designs for the history nav buttons.
pub const PRESETS: usize = 1;

/// Decoded `(unit type, unit kind)` for the currently-focused base-
/// panel element. Returned by `focus_target_for` so callers can pull
/// the matching label/price/description without re-doing the dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FocusTarget {
    pub ty: RobotUnitType,
    pub kind: RobotUnitKind,
}

/// Rendered description labels for the currently-focused component —
/// the C++ fills these from `g_Config.m_Labels[]` / `m_Descriptions[]`
/// (MatrixConfig.cpp:932+) and they drive the `labelL/R` + `descrT/B`
/// statics on the Base panel. Tracked as plain strings; localization
/// lands when the label arrays land.
#[derive(Debug, Clone, Default)]
pub struct FocusedText {
    pub label: String,
    pub description: String,
}

/// Port of `CConstructor` + `CConstructorPanel`. One per
/// player/AI side that can build robots; only the PLAYER_SIDE copy
/// is rendered.
#[derive(Debug, Clone)]
pub struct RobotBuilder {
    /// Currently-edited preset index (0..PRESETS). Port of
    /// `m_CurrentConfig` (CConstructor.h:199).
    pub current_config: usize,

    /// Per-preset state. The C++ has `m_Configs[PRESETS]` on
    /// CConstructorPanel; we follow the same array layout so adding
    /// more presets is just bumping `PRESETS`.
    pub configs: [RobotConfig; PRESETS],

    /// The "live" robot being previewed by the 3D panel. Shadows the
    /// active `configs[current_config]` but kept as its own copy so
    /// mid-edit intermediate states (during `MakeItemReplacements`)
    /// don't clobber the persisted design. Ported from the C++
    /// `m_Head / m_Weapon[] / m_Armor / m_Chassis` instance fields
    /// on CConstructor (CConstructor.h:97-100).
    pub live_head: Unit,
    pub live_weapons: [WeaponUnit; MAX_WEAPON_CNT],
    pub live_armor: ArmorUnit,
    pub live_chassis: Unit,

    /// Active (0/1) — port of `m_Active` flag (CConstructor.h:202).
    /// True while the constructor panel is open; false hides the
    /// sub-panel elements. Set by `ActivateAndSelect`, cleared by
    /// `ResetGroupNClose`.
    pub active: bool,

    /// Port of `m_FocusedElement` (CConstructor.h:194). The C++ stores
    /// a `CIFaceElement*`; we store the element's stable name string —
    /// the pointer-equivalent identity, since `IFaceElement` lives in
    /// a `Vec` whose addresses aren't pointer-stable.
    pub focused_element: Option<String>,

    /// Cached label/description for the current focus. Updated on
    /// every focus change.
    pub focused_text: FocusedText,

    /// Per-resource cost of the focused component, or `None` when no
    /// component is focused. Port of
    /// `g_IFaceList->CreateItemPrice(price)` (CInterface.cpp:3146).
    /// The C++ instantiates dynamic statics for each non-zero
    /// resource icon; we just track the prices so the renderer can
    /// pick them up.
    pub focused_price: Option<UnitPrice>,
}

impl Default for RobotBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RobotBuilder {
    /// Port of the `CConstructorPanel` constructor (CConstructor.h:
    /// 223-239). Seeds the type-tag fields and starts with everything
    /// else zero/empty.
    pub fn new() -> Self {
        Self {
            current_config: 0,
            configs: [RobotConfig::new(); PRESETS],
            live_head: Unit {
                ty: RobotUnitType::Head,
                ..Unit::empty()
            },
            live_weapons: [WeaponUnit::empty(); MAX_WEAPON_CNT],
            live_armor: ArmorUnit {
                unit: Unit {
                    ty: RobotUnitType::Armor,
                    ..Unit::empty()
                },
                ..ArmorUnit::empty()
            },
            live_chassis: Unit {
                ty: RobotUnitType::Chassis,
                ..Unit::empty()
            },
            active: false,
            focused_element: None,
            focused_text: FocusedText::default(),
            focused_price: None,
        }
    }

    /// Port of `CIFaceList::ConstructorButtonsInit`
    /// (CInterface.cpp:4176-4188). Called once at panel-load time to
    /// seed the constructor with default kinds so the pylon icons
    /// show real components instead of "N/A" placeholders.
    ///
    /// Mirrors the C++ exactly:
    /// * chassis = Pneumatic
    /// * armor   = ARMOR_6
    /// * head    = empty (RUK_UNKNOWN)
    /// * weapon[0] = Machinegun (others stay empty)
    pub fn constructor_buttons_init(&mut self) {
        log::debug!("RobotBuilder::constructor_buttons_init starting");
        self.super_djeans(
            RobotUnitType::Chassis,
            RobotUnitKind::CHASSIS_PNEUMATIC,
            0,
            false,
        );
        self.super_djeans(RobotUnitType::Armor, RobotUnitKind::ARMOR_6, 0, false);
        self.super_djeans(RobotUnitType::Head, RobotUnitKind::UNKNOWN, 0, false);
        self.super_djeans(
            RobotUnitType::Weapon,
            RobotUnitKind::WEAPON_MACHINEGUN,
            0,
            false,
        );
        log::debug!(
            "RobotBuilder::constructor_buttons_init done: cfg.chassis={} hull={} head={} weapon[0]={} weapon[1]={} weapon[2]={} weapon[3]={} weapon[4]={}",
            self.cfg().chassis.kind.0,
            self.cfg().hull.unit.kind.0,
            self.cfg().head.kind.0,
            self.cfg().weapon[0].kind.0,
            self.cfg().weapon[1].kind.0,
            self.cfg().weapon[2].kind.0,
            self.cfg().weapon[3].kind.0,
            self.cfg().weapon[4].kind.0,
        );
    }

    /// Accessor for the live-edit preset. The C++ does `m_Configs[m_CurrentConfig]`
    /// inline everywhere; expose a method so the indexing arithmetic
    /// lives in one place.
    pub fn cfg(&self) -> &RobotConfig {
        &self.configs[self.current_config]
    }

    pub fn cfg_mut(&mut self) -> &mut RobotConfig {
        &mut self.configs[self.current_config]
    }

    /// Port of `CConstructorPanel::ActivateAndSelect` (CConstructor.cpp:
    /// 970-976). Opens the constructor + pauses the game.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Port of `CConstructorPanel::FocusElement` (CConstructor.cpp:
    /// 912-958). Dispatches on the panel-element name to figure out
    /// which (type, kind) is now hovered and refreshes the labels +
    /// price popup.
    ///
    /// The C++ keys off pylon names (`IF_BASE_PILON1..5`,
    /// `IF_BASE_PILON_HEAD/HULL/CHASSIS`); we accept the corresponding
    /// Rust element names emitted by the Base panel's static text /
    /// pylon images.
    pub fn focus_element(&mut self, element_name: &str) {
        let Some(FocusTarget { ty, kind }) = self.focus_target_for(element_name) else {
            return;
        };
        // CConstructor.cpp:914 — short-circuit when focus didn't change.
        if self.focused_element.as_deref() == Some(element_name) {
            return;
        }
        self.focused_element = Some(element_name.to_string());
        self.set_labels_and_price(ty, kind);
    }

    /// Port of `CConstructorPanel::UnFocusElement` (CConstructor.cpp:
    /// 960-968). Clears the focused state + cached label/description
    /// and tells the caller to drop the price popup. The C++ checks
    /// `m_FocusedElement == element` — we compare element names since
    /// that's our pointer-equivalent identity.
    pub fn unfocus_element(&mut self, element_name: &str) {
        if self.focused_element.as_deref() == Some(element_name) {
            self.focused_element = None;
            self.focused_text = FocusedText::default();
            self.focused_price = None;
        }
    }

    /// Decode the (type, kind) tuple for a Base-panel element name —
    /// port of the if/else dispatch in `CConstructorPanel::FocusElement`
    /// (CConstructor.cpp:912-958). Returns `None` for non-pylon and
    /// non-template element names.
    pub fn focus_target_for(&self, element_name: &str) -> Option<FocusTarget> {
        let cfg = self.cfg();
        match element_name {
            "pi1" => Some(FocusTarget {
                ty: RobotUnitType::Weapon,
                kind: cfg.weapon[0].kind,
            }),
            "pi2" => Some(FocusTarget {
                ty: RobotUnitType::Weapon,
                kind: cfg.weapon[1].kind,
            }),
            "pi3" => Some(FocusTarget {
                ty: RobotUnitType::Weapon,
                kind: cfg.weapon[2].kind,
            }),
            "pi4" => Some(FocusTarget {
                ty: RobotUnitType::Weapon,
                kind: cfg.weapon[3].kind,
            }),
            "pi5" => Some(FocusTarget {
                ty: RobotUnitType::Weapon,
                kind: cfg.weapon[4].kind,
            }),
            "pihe" => Some(FocusTarget {
                ty: RobotUnitType::Head,
                kind: cfg.head.kind,
            }),
            "pihu" => Some(FocusTarget {
                ty: RobotUnitType::Armor,
                kind: cfg.hull.unit.kind,
            }),
            "pich" => Some(FocusTarget {
                ty: RobotUnitType::Chassis,
                kind: cfg.chassis.kind,
            }),
            // Direct-component template buttons exposed in popup menus.
            n if n.starts_with("chas") => n
                .strip_prefix("chas")
                .and_then(|r| r.parse::<i32>().ok())
                .filter(|k| (1..=5).contains(k))
                .map(|k| FocusTarget {
                    ty: RobotUnitType::Chassis,
                    kind: RobotUnitKind(k),
                }),
            n if n.starts_with("hull") => n
                .strip_prefix("hull")
                .and_then(|r| r.parse::<i32>().ok())
                .filter(|k| (1..=6).contains(k))
                .map(|k| FocusTarget {
                    ty: RobotUnitType::Armor,
                    kind: RobotUnitKind(k),
                }),
            n if n.starts_with("head") && n != "heade" => n
                .strip_prefix("head")
                .and_then(|r| r.parse::<i32>().ok())
                .filter(|k| (1..=4).contains(k))
                .map(|k| FocusTarget {
                    ty: RobotUnitType::Head,
                    kind: RobotUnitKind(k),
                }),
            n if n.starts_with("weap") && n != "weape" => n
                .strip_prefix("weap")
                .and_then(|r| r.parse::<i32>().ok())
                .filter(|k| (1..=10).contains(k))
                .map(|k| FocusTarget {
                    ty: RobotUnitType::Weapon,
                    kind: RobotUnitKind(k),
                }),
            _ => None,
        }
    }

    /// Port of `CConstructorPanel::SetLabelsAndPrice` (CConstructor.cpp:
    /// 1193-1228). Pulls the localized label + description for the
    /// (type, kind) component, applies the template substitutions
    /// (`<Damage>`, `<Size>`, `<Pilons>`, `<WeaponSpeed>`, etc.) and
    /// stores the result in `focused_text` for the renderer to draw.
    ///
    /// Also pushes the per-component price into `focused_price` so the
    /// popup price card has fresh data — port of the
    /// `g_IFaceList->CreateItemPrice(&g_Config.m_Price[...])` call at
    /// CConstructor.cpp:1204/1210/1216/1222.
    pub fn set_labels_and_price(&mut self, ty: RobotUnitType, kind: RobotUnitKind) {
        if kind == RobotUnitKind::UNKNOWN {
            self.focused_text = FocusedText::default();
            self.focused_price = None;
            return;
        }
        let strings = config::global_strings();
        let label = strings.labels.get(ty, kind).to_string();
        let description = strings.descriptions.get(ty, kind).to_string();
        let mut text = FocusedText { label, description };
        make_item_replacements(&mut text, ty, kind);
        self.focused_text = text;
        // Cache the price for the popup card.
        let price = config::global().prices.get(ty, kind);
        self.focused_price = if price.is_zero() { None } else { Some(price) };
    }

    /// Port of `CConstructorPanel::ResetGroupNClose` (CConstructor.cpp:
    /// 986-994). Closes the panel; resumes the game.
    pub fn deactivate(&mut self) {
        self.active = false;
        self.focused_element = None;
        self.focused_text = FocusedText::default();
        self.focused_price = None;
    }

    pub fn refresh_current_focus(&mut self) {
        let Some(name) = self.focused_element.clone() else {
            return;
        };
        if let Some(FocusTarget { ty, kind }) = self.focus_target_for(&name) {
            self.set_labels_and_price(ty, kind);
        } else {
            self.focused_text = FocusedText::default();
            self.focused_price = None;
        }
    }

    /// Port of `CConstructor::ResetConstruction` (CConstructor.cpp:
    /// 786-796). Zeros the live preview state (chassis/armor/head/weapons)
    /// but leaves the persisted preset intact — matches the behaviour
    /// of the C++ where ResetConstruction clears the in-progress edit
    /// but m_Configs[] keeps the saved preset.
    pub fn reset_construction(&mut self) {
        self.live_head = Unit {
            ty: RobotUnitType::Head,
            ..Unit::empty()
        };
        self.live_weapons = [WeaponUnit::empty(); MAX_WEAPON_CNT];
        self.live_armor = ArmorUnit {
            unit: Unit {
                ty: RobotUnitType::Armor,
                ..Unit::empty()
            },
            ..ArmorUnit::empty()
        };
        self.live_chassis = Unit {
            ty: RobotUnitType::Chassis,
            ..Unit::empty()
        };
    }

    /// Port of `CConstructor::OperateUnit` (CConstructor.cpp:674-729).
    /// Low-level mutator: sets the matching live_ slot on the preview
    /// robot + cascades armor → weapon slot caps.
    pub fn operate_unit(&mut self, ty: RobotUnitType, kind: RobotUnitKind) {
        let prices = config::global().prices;
        let price = prices.get(ty, kind);

        match ty {
            RobotUnitType::Chassis => {
                self.live_chassis = Unit {
                    ty: RobotUnitType::Chassis,
                    kind,
                    price,
                };
            }
            RobotUnitType::Armor if !self.live_chassis.kind.is_empty() => {
                // Reset weapons when armor changes (CConstructor.cpp:682).
                self.live_weapons = [WeaponUnit::empty(); MAX_WEAPON_CNT];
                self.live_armor.unit = Unit {
                    ty: RobotUnitType::Armor,
                    kind,
                    price,
                };
                let matrix = default_weapon_matrix(kind);
                self.live_armor.max_common_weapon_cnt = matrix.common;
                self.live_armor.max_extra_weapon_cnt = matrix.extra;
            }
            RobotUnitType::Head if !self.live_armor.unit.kind.is_empty() => {
                self.live_head = Unit {
                    ty: RobotUnitType::Head,
                    kind,
                    price,
                };
            }
            RobotUnitType::Weapon if !self.live_armor.unit.kind.is_empty() => {
                let matrix = default_weapon_matrix(self.live_armor.unit.kind);
                let current = [
                    self.live_weapons[0].unit,
                    self.live_weapons[1].unit,
                    self.live_weapons[2].unit,
                    self.live_weapons[3].unit,
                    self.live_weapons[4].unit,
                ];
                if let Some(pos) = matrix.find_pylon_for(kind, &current) {
                    self.live_weapons[pos] = WeaponUnit {
                        pos: pos as i32 + 1,
                        unit: Unit {
                            ty: RobotUnitType::Weapon,
                            kind,
                            price,
                        },
                    };
                }
            }
            _ => {}
        }
    }

    /// Port of the weapon-pylon writes in `CConstructor::SuperDjeans`
    /// (CConstructor.cpp:469-528) / `Djeans007` (CConstructor.cpp:
    /// 582-636). Resolves the *physical* pylon for the UI pylon index
    /// and unconditionally overwrites it — kind 0 clears the slot, an
    /// occupied slot is replaced. This is deliberately different from
    /// `operate_unit`'s weapon path (`CheckWeaponLegality`), which
    /// only fills the first empty compatible slot.
    ///
    /// For `pilon != 4` the physical slot is the (pilon+1)-th common
    /// (non-extra) slot in the armor's weapon matrix ("fis_pilon");
    /// for `pilon == 4` it's the first extra slot.
    fn apply_weapon_to_live(&mut self, kind: RobotUnitKind, pilon: i32) {
        let hull_kind = self.cfg().hull.unit.kind;
        if hull_kind.is_empty() {
            return;
        }
        let matrix = default_weapon_matrix(hull_kind);
        let limit = matrix.cnt as usize;
        let phys = if pilon != 4 {
            // CConstructor.cpp:472-486 — count down common slots.
            let mut pilon_ost = pilon + 1;
            let mut t = 0usize;
            while t < limit && pilon_ost > 0 {
                if !matrix.is_extra_slot(t) {
                    pilon_ost -= 1;
                }
                t += 1;
            }
            if t == 0 {
                return;
            }
            t - 1
        } else {
            // CConstructor.cpp:500-511 — first extra slot. The C++
            // falls back to slot 0 when the armor has no extra slot,
            // but that path is unreachable (callers gate on
            // m_MaxExtraWeaponCnt); bail out instead.
            match (0..limit).find(|&t| matrix.is_extra_slot(t)) {
                Some(t) => t,
                None => return,
            }
        };
        if phys >= MAX_WEAPON_CNT {
            return;
        }
        self.live_weapons[phys] = WeaponUnit {
            pos: phys as i32 + 1,
            unit: Unit {
                ty: RobotUnitType::Weapon,
                kind,
                price: config::global().prices.get(RobotUnitType::Weapon, kind),
            },
        };
    }

    /// Refresh cached price / structure totals on the current preset
    /// from the **live preview** state. The C++ draws the pylon icons
    /// off `m_Configs[]` (user intent) but reads price/stats off
    /// `m_Robot` (the built preview — only populated for units whose
    /// dependencies are met). Mirror that: only recompute cost/struct,
    /// never clobber `cfg.{chassis,hull,head,weapon}` — those hold the
    /// user's intent and are written by `super_djeans` directly.
    fn sync_preview_totals(&mut self) {
        let cost = cost_of_live(
            &self.live_chassis,
            &self.live_armor,
            &self.live_head,
            &self.live_weapons,
        );
        let structure = structure_of_live(&self.live_chassis, &self.live_armor, &self.live_head);
        let damage = damage_of_live(&self.live_weapons, &self.live_head);
        let cfg = self.cfg_mut();
        cfg.tit_x = cost.titan();
        cfg.elec_x = cost.electronics();
        cfg.ener_x = cost.energy();
        cfg.plas_x = cost.plasma();
        cfg.structure = structure;
        cfg.damage = damage;
    }

    /// Port of `CConstructor::SuperDjeans` (CConstructor.cpp:441-574) —
    /// the top-level "the user clicked a chassis/armor/head/weapon
    /// button" handler. Delegates to `operate_unit` after applying the
    /// wrap-around / preserve-weapons / reset cascade logic that
    /// SuperDjeans wraps around the raw operate_unit call.
    ///
    /// `pilon` is the 0-based pylon index when `ty == Weapon`, else 0.
    /// `from_history` matches the C++ `ld_from_history` flag which
    /// skips the weapon-reset cascade during preset loading.
    pub fn super_djeans(
        &mut self,
        ty: RobotUnitType,
        kind: RobotUnitKind,
        pilon: i32,
        from_history: bool,
    ) {
        match ty {
            RobotUnitType::Head => {
                self.cfg_mut().head.kind = kind;
                self.operate_unit(RobotUnitType::Head, kind);
            }
            RobotUnitType::Armor => {
                let old_cfg = if !from_history {
                    Some(*self.cfg())
                } else {
                    None
                };
                self.cfg_mut().reset_weapons();
                self.cfg_mut().hull.unit.kind = kind;
                self.operate_unit(RobotUnitType::Armor, kind);
                // Restore compatible weapons from the old config
                // (CConstructor.cpp:537-560). Common weapons reattach
                // up to the new armor's cap; super weapons to the extra
                // slot.
                if let Some(old) = old_cfg {
                    let cc = self.live_armor.max_common_weapon_cnt;
                    let mut pos = 0;
                    for i in 0..MAX_WEAPON_CNT {
                        if pos >= cc as usize {
                            break;
                        }
                        let k = old.weapon[i].kind;
                        if !k.is_empty()
                            && k != RobotUnitKind::WEAPON_BOMB
                            && k != RobotUnitKind::WEAPON_MORTAR
                        {
                            self.super_djeans(RobotUnitType::Weapon, k, pos as i32, false);
                            pos += 1;
                        }
                    }
                    if self.live_armor.max_extra_weapon_cnt > 0 {
                        for i in 0..MAX_WEAPON_CNT {
                            let k = old.weapon[i].kind;
                            if k == RobotUnitKind::WEAPON_BOMB || k == RobotUnitKind::WEAPON_MORTAR
                            {
                                self.super_djeans(RobotUnitType::Weapon, k, 4, false);
                                break;
                            }
                        }
                    }
                    // Restore head (CConstructor.cpp:558).
                    self.super_djeans(RobotUnitType::Head, old.head.kind, 0, false);
                }
            }
            RobotUnitType::Chassis => {
                self.cfg_mut().chassis.kind = kind;
                self.operate_unit(RobotUnitType::Chassis, kind);
            }
            RobotUnitType::Weapon => {
                let p = pilon as usize;
                if p < MAX_WEAPON_CNT {
                    self.cfg_mut().weapon[p].kind = kind;
                }
                // CConstructor.cpp:472-528 — resolve the physical
                // pylon ("fis_pilon") and overwrite/clear it directly.
                self.apply_weapon_to_live(kind, pilon);
            }
            RobotUnitType::Empty => {}
        }
        self.sync_preview_totals();
        self.refresh_current_focus();
    }

    /// Port of `CConstructor::RemoteOperateUnit` (CConstructor.cpp:
    /// 362-438) — the pylon-button ON_UN_PRESS handler (wired at
    /// CInterface.cpp:262). Cycles the mounted component kind with
    /// wrap-around, then commits via `super_djeans`:
    /// * head:    0→1→2→3→4→0 (can cycle to empty)
    /// * armor:   1→…→6→1 (never empty)
    /// * chassis: 1→…→5→1 (never empty)
    /// * weapon (common pylon): 0→1→2→3→4→6→8→9→10→0 — kinds 4 and 6
    ///   jump by 2, skipping mortar (5) and bomb (7) which only fit
    ///   the extra pylon
    /// * weapon (extra pylon, `pi5`): 0→5(mortar)→7(bomb)→0
    pub fn remote_operate_unit(&mut self, element_name: &str) {
        let cfg = self.cfg();
        let (ty, kind, pilon) = match element_name {
            "pihe" => {
                let k = cfg.head.kind.0;
                let next = if k == 4 { 0 } else { k + 1 };
                (RobotUnitType::Head, next, 0)
            }
            "pihu" => {
                let k = cfg.hull.unit.kind.0;
                let next = if k == 6 { 1 } else { k + 1 };
                (RobotUnitType::Armor, next, 0)
            }
            "pich" => {
                let k = cfg.chassis.kind.0;
                let next = if k == 5 { 1 } else { k + 1 };
                (RobotUnitType::Chassis, next, 0)
            }
            "pi1" | "pi2" | "pi3" | "pi4" => {
                let pilon = match element_name {
                    "pi1" => 0,
                    "pi2" => 1,
                    "pi3" => 2,
                    _ => 3,
                };
                let k = cfg.weapon[pilon as usize].kind.0;
                let next = if k == 4 || k == 6 {
                    k + 2
                } else if k == 10 {
                    0
                } else {
                    k + 1
                };
                (RobotUnitType::Weapon, next, pilon)
            }
            "pi5" => {
                let next = match cfg.weapon[4].kind.0 {
                    0 => 5,
                    5 => 7,
                    _ => 0,
                };
                (RobotUnitType::Weapon, next, 4)
            }
            _ => return,
        };
        self.super_djeans(ty, RobotUnitKind(kind), pilon, false);
    }

    /// Port of `CConstructor::Djeans007` (CConstructor.cpp:576-671) —
    /// the recursive variant `SuperDjeans` calls during armor cascade
    /// to reattach compatible weapons. Differs from `super_djeans` in
    /// two ways:
    /// * it does NOT call `ResetWeapon` on armor change (the caller
    ///   already did);
    /// * it does NOT trigger nested armor cascades, so weapons placed
    ///   here just slot in without restoring an old config.
    ///
    /// We use it as the implementation for armor weapon-restoration so
    /// the recursion topology matches the C++. `super_djeans`'s armor
    /// branch in turn invokes `djeans007` for each restored weapon.
    pub fn djeans007(&mut self, ty: RobotUnitType, kind: RobotUnitKind, pilon: i32) {
        match ty {
            RobotUnitType::Head => {
                self.cfg_mut().head.kind = kind;
                self.operate_unit(RobotUnitType::Head, kind);
            }
            RobotUnitType::Chassis => {
                self.cfg_mut().chassis.kind = kind;
                self.operate_unit(RobotUnitType::Chassis, kind);
            }
            RobotUnitType::Armor => {
                // Mirror CConstructor.cpp:645-665. The C++ snapshots
                // the current weapons, calls OperateUnit (which clears
                // the weapon array as part of the armor cascade), then
                // re-attaches compatible weapons by recursing into
                // Djeans007 — *not* SuperDjeans. This is the popup-menu
                // "preview" path; SuperDjeans is the click-commit path.
                let tmp_weapon = self.cfg().weapon;
                self.cfg_mut().hull.unit.kind = kind;
                self.operate_unit(RobotUnitType::Armor, kind);
                let mut try_common = self.live_armor.max_common_weapon_cnt;
                let mut try_extra = self.live_armor.max_extra_weapon_cnt;
                for (i, w) in tmp_weapon.iter().enumerate() {
                    if try_common <= 0 {
                        break;
                    }
                    let k = w.kind;
                    if !k.is_empty()
                        && k != RobotUnitKind::WEAPON_BOMB
                        && k != RobotUnitKind::WEAPON_MORTAR
                    {
                        self.djeans007(RobotUnitType::Weapon, k, i as i32);
                        try_common -= 1;
                    }
                }
                for w in tmp_weapon.iter() {
                    if try_extra <= 0 {
                        break;
                    }
                    let k = w.kind;
                    if k == RobotUnitKind::WEAPON_BOMB || k == RobotUnitKind::WEAPON_MORTAR {
                        self.djeans007(RobotUnitType::Weapon, k, 4);
                        try_extra -= 1;
                    }
                }
            }
            RobotUnitType::Weapon => {
                let p = pilon as usize;
                if p < MAX_WEAPON_CNT {
                    self.cfg_mut().weapon[p].kind = kind;
                }
                // CConstructor.cpp:582-636 — same physical-pylon
                // overwrite as SuperDjeans.
                self.apply_weapon_to_live(kind, pilon);
            }
            RobotUnitType::Empty => {}
        }
        self.sync_preview_totals();
        self.refresh_current_focus();
    }

    /// Port of `CConstructor::CheckMaxUnits` (CConstructor.h:120) —
    /// returns true while the total populated units (chassis + armor +
    /// head + weapons) stays at or below `MR_MAXUNIT` (9). The C++
    /// never actually calls this (dead getter), but ports cleanly as
    /// an accessor for the same purpose.
    pub fn check_max_units(&self) -> bool {
        (self.cfg().unit_count() as usize) <= MR_MAXUNIT
    }

    /// Port of `CConstructor::m_nUnitCnt` (CConstructor.h:90) accessor.
    /// Recomputed from the persisted preset to mirror the running
    /// total `OperateCurrentConstruction` maintains at
    /// CConstructor.cpp:709-725.
    pub fn unit_count(&self) -> i32 {
        self.cfg().unit_count()
    }

    /// Port of `CConstructor::GetConstructionPrice` (CConstructor.cpp:
    /// 835-858). Sums the current live preview's component costs
    /// into a `UnitPrice`.
    pub fn construction_price(&self) -> UnitPrice {
        cost_of_live(
            &self.live_chassis,
            &self.live_armor,
            &self.live_head,
            &self.live_weapons,
        )
    }

    /// Port of `CConstructor::GetConstructionStructure` (CConstructor.cpp:
    /// 860-899).
    pub fn construction_structure(&self) -> i32 {
        structure_of_live(&self.live_chassis, &self.live_armor, &self.live_head)
    }

    /// Port of `GetConstructionDamage` (CConstructor.cpp:1574-1613).
    pub fn construction_damage(&self) -> i32 {
        damage_of_live(&self.live_weapons, &self.live_head)
    }

    /// Port of `GetConstructionName` (CConstructor.cpp:1540-1572).
    pub fn construction_name(&self) -> String {
        name_of_live(
            &self.live_chassis,
            &self.live_armor,
            &self.live_head,
            &self.live_weapons,
        )
    }

    /// Port of `CConstructorPanel::IsEnoughResourcesForThisPieceOfShit`
    /// (CConstructor.cpp:996-1095). Returns `false` when picking
    /// `(pilon, ty, kind)` for the current preset would leave the
    /// player short of any resource that the new item itself consumes.
    ///
    /// The C++ predicate is asymmetric: a resource gap matters only if
    /// the proposed item actually requires that resource (`item_res[i] != 0`).
    /// That way swapping in a chassis that costs only Titan won't be
    /// rejected because the player is low on Plasma.
    ///
    /// `side_bank` mirrors the four `ps->GetResourcesAmount(ERes(i))`
    /// reads at CConstructor.cpp:1089.
    pub fn is_enough_resources_for_pick(
        &self,
        pilon: i32,
        ty: RobotUnitType,
        kind: RobotUnitKind,
        side_bank: &[i32; 4],
    ) -> bool {
        let prices = config::global().prices;
        // CConstructor.cpp:1007 — start with current per-unit total.
        let mut total = self.construction_price();
        let mut minus = UnitPrice::zero();
        let mut plus = UnitPrice::zero();
        let mut item = UnitPrice::zero();

        let cfg = self.cfg();
        match ty {
            RobotUnitType::Head => {
                if !cfg.head.kind.is_empty() {
                    minus = prices.get(RobotUnitType::Head, cfg.head.kind);
                }
                plus = prices.get(RobotUnitType::Head, kind);
                item = plus;
            }
            RobotUnitType::Armor => {
                let new_matrix = default_weapon_matrix(kind);
                let mut try_common = new_matrix.common;
                let mut try_extra = new_matrix.extra;

                if !cfg.hull.unit.kind.is_empty() {
                    minus = prices.get(RobotUnitType::Armor, cfg.hull.unit.kind);
                    // CConstructor.cpp:1032-1038 — current armor's weapon
                    // costs subtract too (they were part of total).
                    for w in &cfg.weapon {
                        if !w.kind.is_empty() {
                            let p = prices.get(RobotUnitType::Weapon, w.kind);
                            for i in 0..4 {
                                minus.resources[i] += p.resources[i];
                            }
                        }
                    }
                }
                plus = prices.get(RobotUnitType::Armor, kind);
                item = plus;

                // CConstructor.cpp:1043-1050 — common weapons that fit
                // the new armor's matrix are kept; their cost rejoins
                // the running total.
                for w in &cfg.weapon {
                    if try_common <= 0 {
                        break;
                    }
                    if !w.kind.is_empty()
                        && w.kind != RobotUnitKind::WEAPON_BOMB
                        && w.kind != RobotUnitKind::WEAPON_MORTAR
                    {
                        let p = prices.get(RobotUnitType::Weapon, w.kind);
                        for i in 0..4 {
                            plus.resources[i] += p.resources[i];
                        }
                        try_common -= 1;
                    }
                }
                // CConstructor.cpp:1052-1059 — same for the extra slot.
                for w in &cfg.weapon {
                    if try_extra <= 0 {
                        break;
                    }
                    if w.kind == RobotUnitKind::WEAPON_BOMB
                        || w.kind == RobotUnitKind::WEAPON_MORTAR
                    {
                        let p = prices.get(RobotUnitType::Weapon, w.kind);
                        for i in 0..4 {
                            plus.resources[i] += p.resources[i];
                        }
                        try_extra -= 1;
                    }
                }
            }
            RobotUnitType::Chassis => {
                if !cfg.chassis.kind.is_empty() {
                    minus = prices.get(RobotUnitType::Chassis, cfg.chassis.kind);
                }
                plus = prices.get(RobotUnitType::Chassis, kind);
                item = plus;
            }
            RobotUnitType::Weapon => {
                let p = pilon as usize;
                if p < MAX_WEAPON_CNT && !cfg.weapon[p].kind.is_empty() {
                    minus = prices.get(RobotUnitType::Weapon, cfg.weapon[p].kind);
                }
                plus = prices.get(RobotUnitType::Weapon, kind);
                item = plus;
            }
            RobotUnitType::Empty => {}
        }

        // CConstructor.cpp:1075-1078.
        for i in 0..4 {
            total.resources[i] = total.resources[i] - minus.resources[i] + plus.resources[i];
        }
        // CConstructor.cpp:1087-1091 — only resources the new item
        // actually consumes act as gates.
        #[allow(clippy::needless_range_loop)]
        for i in 0..4 {
            if side_bank[i] < total.resources[i] && item.resources[i] != 0 {
                return false;
            }
        }
        true
    }

    /// Port of `CConstructor::OperateCurrentConstruction`
    /// (CConstructor.cpp:814-833) — repopulate the live preview from
    /// the persisted config. Used by `super_djeans` after a preset
    /// load and by the history navigation buttons.
    pub fn operate_current_construction(&mut self) {
        let cfg = *self.cfg();
        self.reset_construction();
        self.operate_unit(RobotUnitType::Chassis, cfg.chassis.kind);
        self.operate_unit(RobotUnitType::Armor, cfg.hull.unit.kind);
        for w in &cfg.weapon {
            if !w.kind.is_empty() {
                self.operate_unit(RobotUnitType::Weapon, w.kind);
            }
        }
        self.operate_unit(RobotUnitType::Head, cfg.head.kind);
        self.sync_preview_totals();
    }

    /// Load a config from history into the live preview + the current
    /// preset. Used by the hisleft/hisright buttons.
    pub fn apply_config(&mut self, cfg: RobotConfig) {
        *self.cfg_mut() = cfg;
        self.operate_current_construction();
    }

    /// Port of `CConstructor::BuildSpecialBot` (CConstructor.cpp:798-812).
    ///
    /// Loads a special-bot template into the live preview by walking
    /// each component through `operate_unit` (which handles armor →
    /// weapon-cap cascade etc.). Returns the produced `RobotConfig` +
    /// team so the caller can `Building::queue_robot` it; the C++
    /// directly calls `StackRobot(NULL, bot.m_Team)` at the end.
    pub fn build_special_bot(&mut self, bot: &SpecialBot) -> (RobotConfig, i32) {
        self.reset_construction();
        // CConstructor.cpp:802-810 — chassis, armor, every weapon
        // (incl. empty slots), then head.
        self.operate_unit(RobotUnitType::Chassis, bot.chassis.kind);
        self.operate_unit(RobotUnitType::Armor, bot.armor.unit.kind);
        for w in &bot.weapon {
            self.operate_unit(RobotUnitType::Weapon, w.unit.kind);
        }
        self.operate_unit(RobotUnitType::Head, bot.head.kind);
        // Mirror live-preview state into the persisted preset so the
        // returned config has the right component fields.
        let chassis = self.live_chassis;
        let armor = self.live_armor;
        let head = self.live_head;
        let weapons = self.live_weapons;
        let cfg = self.cfg_mut();
        cfg.chassis = chassis;
        cfg.hull = armor;
        cfg.head = head;
        for (i, w) in weapons.iter().enumerate() {
            cfg.weapon[i] = w.unit;
        }
        self.sync_preview_totals();
        (*self.cfg(), bot.team)
    }

    /// Port of `CConstructor::BuildRandomBot` (CConstructor.h:137-155).
    /// Same call pattern as `BuildSpecialBot` but using a random
    /// component combination from
    /// [`crate::matrix_game::interface::special_bot::build_random_bot`].
    /// Returns the produced config so the caller can stack it.
    pub fn build_random_bot(&mut self, rng: &mut Rnd) -> RobotConfig {
        let cfg = build_random_bot(rng);
        self.reset_construction();
        self.operate_unit(RobotUnitType::Chassis, cfg.chassis.kind);
        self.operate_unit(RobotUnitType::Armor, cfg.hull.unit.kind);
        for w in &cfg.weapon {
            if !w.is_empty() {
                self.operate_unit(RobotUnitType::Weapon, w.kind);
            }
        }
        self.operate_unit(RobotUnitType::Head, cfg.head.kind);
        *self.cfg_mut() = cfg;
        self.sync_preview_totals();
        cfg
    }
}

/// Helper: sum component costs from a live preview. Shared by
/// `RobotBuilder::construction_price` and the Build-button cost
/// preview on the Main panel.
pub fn cost_of_live(
    chassis: &Unit,
    armor: &ArmorUnit,
    head: &Unit,
    weapons: &[WeaponUnit; MAX_WEAPON_CNT],
) -> UnitPrice {
    let prices = config::global().prices;
    let mut out = UnitPrice::zero();
    if !chassis.is_empty() {
        out.add_from(prices.get(RobotUnitType::Chassis, chassis.kind));
    }
    if !armor.unit.is_empty() {
        out.add_from(prices.get(RobotUnitType::Armor, armor.unit.kind));
    }
    if !head.is_empty() {
        out.add_from(prices.get(RobotUnitType::Head, head.kind));
    }
    for w in weapons {
        if !w.unit.is_empty() {
            out.add_from(prices.get(RobotUnitType::Weapon, w.unit.kind));
        }
    }
    out
}

/// Component-price sum of a queued `RobotConfig` — port of
/// `CBuildStack::ReturnRobotResources`'s per-unit price walk
/// (MatrixObjectBuilding.cpp:1974-2008), used to refund cancelled
/// build-queue items.
pub fn cost_of_config(cfg: &RobotConfig) -> UnitPrice {
    let prices = config::global().prices;
    let mut out = UnitPrice::zero();
    if !cfg.chassis.is_empty() {
        out.add_from(prices.get(RobotUnitType::Chassis, cfg.chassis.kind));
    }
    if !cfg.hull.unit.is_empty() {
        out.add_from(prices.get(RobotUnitType::Armor, cfg.hull.unit.kind));
    }
    if !cfg.head.is_empty() {
        out.add_from(prices.get(RobotUnitType::Head, cfg.head.kind));
    }
    for w in &cfg.weapon {
        if !w.is_empty() {
            out.add_from(prices.get(RobotUnitType::Weapon, w.kind));
        }
    }
    out
}

/// Port of `CConstructorPanel::MakeItemReplacements` (CConstructor.cpp:
/// 1100-1191). Substitutes `<Damage>`, `<AddDamage>`, `<Size>`,
/// `<Pilons>`, `<AddPilons>`, `<WeaponSpeed>`, `<WeaponHeat>`,
/// `<ChassisSpeed>`, `<WeaponRadius>`, `<RadarRadius>`, `<Max>`,
/// `<LessHit>`, `<BombProtect>` template tags inside `text.label`
/// with computed values, each wrapped in the gold colour tag
/// (`<Color=247,195,0>` … `</color>`) the C++ uses for highlighted
/// numbers.
pub fn make_item_replacements(text: &mut FocusedText, ty: RobotUnitType, kind: RobotUnitKind) {
    if matches!(ty, RobotUnitType::Empty) || kind == RobotUnitKind::UNKNOWN {
        return;
    }
    // CConstructor.cpp:1104 — the source labels use literal `<br>`
    // separators that the renderer treats as a newline.
    text.label = text.label.replace("<br>", "\r\n");

    let cfg = config::global();
    let color_open = "<Color=247,195,0>";
    let color_close = "</color>";
    let render_int = |v: i32| -> String { format!("{}{}{}", color_open, v, color_close) };

    match ty {
        RobotUnitType::Weapon => {
            let wspeed = cfg.weapon_cooldown.for_kind(kind);
            let mut damage = cfg.robot_damages.for_kind(kind).damage;
            if wspeed > 0 {
                damage = (damage as f32 * 1000.0 / wspeed as f32).round() as i32;
            } else {
                damage = 0;
            }
            // Special-case auxiliary damage tags for ABLAZE / SHORTED,
            // and the BIGBOOM substitution for BOMB. Mirrors
            // CConstructor.cpp:1113-1124.
            let mut adamage = 0i32;
            use crate::matrix_game::effects::weapon::{
                WEAPON_ABLAZE, WEAPON_BIGBOOM, WEAPON_SHORTED,
            };
            if kind == RobotUnitKind::WEAPON_FLAMETHROWER {
                let dmg = cfg
                    .robot_damages
                    .table
                    .get(
                        crate::matrix_game::effects::weapon::weap_to_index(WEAPON_ABLAZE)
                            .unwrap_or(0),
                    )
                    .map(|d| d.damage)
                    .unwrap_or(0);
                // OBJECT_ROBOT_ABLAZE_PERIOD = 10 (MatrixMapStatic.hpp:94,
                // CConstructor.cpp:1117).
                adamage = (dmg as f32 * 1000.0 / 10.0).round() as i32;
            } else if kind == RobotUnitKind::WEAPON_ELECTRIC {
                let dmg = cfg
                    .robot_damages
                    .table
                    .get(
                        crate::matrix_game::effects::weapon::weap_to_index(WEAPON_SHORTED)
                            .unwrap_or(0),
                    )
                    .map(|d| d.damage)
                    .unwrap_or(0);
                // OBJECT_SHORTED_PERIOD = 50 (MatrixMapStatic.hpp:96,
                // CConstructor.cpp:1120).
                adamage = (dmg as f32 * 1000.0 / 50.0).round() as i32;
            } else if kind == RobotUnitKind::WEAPON_BOMB {
                damage = cfg
                    .robot_damages
                    .table
                    .get(
                        crate::matrix_game::effects::weapon::weap_to_index(WEAPON_BIGBOOM)
                            .unwrap_or(0),
                    )
                    .map(|d| d.damage)
                    .unwrap_or(0);
            }
            text.label = text
                .label
                .replace("<Damage>", &render_int(damage / 10))
                .replace("<AddDamage>", &render_int(adamage / 10));
        }
        RobotUnitType::Head => {
            // Mirrors CConstructor.cpp:1135-1170. Each `Chars/Heads/{X}`
            // param maps to a template tag; the value is wrapped in the
            // gold-colour markup. `sign` matches the C++:
            //   HIT_POINT_ADD → <Size>, sign 0.1
            //   COOL_DOWN     → <WeaponSpeed>, sign -1
            //   OVERHEAT      → <WeaponHeat>, sign -1
            //   CHASSIS_SPEED → <ChassisSpeed>, sign 1
            //   FIRE_DISTANCE → <WeaponRadius>, sign 1
            //   RADAR_DISTANCE→ <RadarRadius>, sign 1
            //   LIGHT_PROTECT → <Max>, sign 1
            //   AIM_PROTECT   → <LessHit>, sign 1
            //   BOMB_PROTECT  → <BombProtect>, sign 1
            let i = kind.as_index();
            let h = &cfg.head_chars;
            let put = |label: &mut String, tag: &str, v: f32| {
                let int_v = v.round() as i32;
                *label = label.replace(tag, &render_int(int_v));
            };
            if let Some(&v) = h.hit_point_add.get(i) {
                put(&mut text.label, "<Size>", v * 0.1);
            }
            if let Some(&v) = h.cool_down.get(i) {
                put(&mut text.label, "<WeaponSpeed>", -v);
            }
            if let Some(&v) = h.overheat.get(i) {
                put(&mut text.label, "<WeaponHeat>", -v);
            }
            if let Some(&v) = h.chassis_speed.get(i) {
                put(&mut text.label, "<ChassisSpeed>", v);
            }
            if let Some(&v) = h.fire_distance.get(i) {
                put(&mut text.label, "<WeaponRadius>", v);
            }
            if let Some(&v) = h.radar_distance.get(i) {
                put(&mut text.label, "<RadarRadius>", v);
            }
            if let Some(&v) = h.light_protect.get(i) {
                put(&mut text.label, "<Max>", v);
            }
            if let Some(&v) = h.aim_protect.get(i) {
                put(&mut text.label, "<LessHit>", v);
            }
            if let Some(&v) = h.bomb_protect.get(i) {
                put(&mut text.label, "<BombProtect>", v);
            }
        }
        RobotUnitType::Armor => {
            // Mirrors CConstructor.cpp:1171-1185.
            let idx = kind.as_index();
            let structure = cfg
                .item_chars
                .armor_structure
                .get(idx)
                .copied()
                .unwrap_or(0);
            let matrix = default_weapon_matrix(kind);
            let pilons = matrix.common;
            let ext_pilons = matrix.extra;
            text.label = text
                .label
                .replace("<Size>", &render_int(structure / 10))
                .replace("<Pilons>", &render_int(pilons))
                .replace("<AddPilons>", &render_int(ext_pilons));
        }
        RobotUnitType::Chassis => {
            // Mirrors CConstructor.cpp:1186-1190.
            let idx = kind.as_index();
            let structure = cfg
                .item_chars
                .chassis_structure
                .get(idx)
                .copied()
                .unwrap_or(0);
            text.label = text.label.replace("<Size>", &render_int(structure / 10));
        }
        RobotUnitType::Empty => {}
    }
}

/// Helper: total per-shot damage of the live config. Ports
/// `GetConstructionDamage` (CConstructor.cpp:1574-1613) — sums per-
/// weapon DPS adjusted by the head's COOL_DOWN bonus, then divides
/// by 10 for the display value.
pub fn damage_of_live(weapons: &[WeaponUnit; MAX_WEAPON_CNT], head: &Unit) -> i32 {
    let cfg = config::global();
    // CConstructor.cpp:1593 — COOL_DOWN is a percentage; divide by 100,
    // floor at -1.0 (anything below kills the firing rate).
    let head_idx = if head.is_empty() {
        None
    } else {
        Some(head.kind.as_index())
    };
    let cooldown_bonus = head_idx
        .and_then(|i| cfg.head_chars.cool_down.get(i).copied())
        .map(|v| (v / 100.0).max(-1.0))
        .unwrap_or(0.0);

    let mut damage = 0;
    for w in weapons {
        let kind = w.unit.kind;
        if kind.is_empty() || kind == RobotUnitKind::WEAPON_BOMB {
            // CConstructor.cpp:1602 — bomb is excluded from the running
            // damage total (it's a one-shot ordnance).
            continue;
        }
        let mut wspeed = cfg.weapon_cooldown.for_kind(kind) as f32;
        wspeed += wspeed * cooldown_bonus;
        if wspeed <= 0.0 {
            continue;
        }
        let single_damage = cfg.robot_damages.for_kind(kind).damage as f32;
        damage += (single_damage * 1000.0 / wspeed).round() as i32;
    }
    damage / 10
}

/// Helper: compose the robot's display name. Ports
/// `GetConstructionName` (CConstructor.cpp:1540-1572) — concatenates
/// the per-armor / per-chassis / per-head label fragments and appends
/// the total damage as a `-NNN` suffix.
pub fn name_of_live(
    chassis: &Unit,
    armor: &ArmorUnit,
    head: &Unit,
    weapons: &[WeaponUnit; MAX_WEAPON_CNT],
) -> String {
    let names = config::global_strings();
    let mut out = String::new();
    if !armor.unit.is_empty() {
        if let Some(s) = names.robot_names.hull.get(armor.unit.kind.as_index()) {
            out.push_str(s);
        }
    }
    if !chassis.is_empty() {
        if let Some(s) = names.robot_names.chassis.get(chassis.kind.as_index()) {
            out.push_str(s);
        }
    }
    if !head.is_empty() {
        if let Some(s) = names.robot_names.head.get(head.kind.as_index()) {
            out.push_str(s);
        }
    }
    let dmg = damage_of_live(weapons, head);
    if dmg != 0 {
        out.push('-');
        out.push_str(&dmg.to_string());
    }
    out
}

/// Helper: compute the robot's structural HP from the current live
/// preview. Mirrors `GetConstructionStructure` (CConstructor.cpp:
/// 860-899).
pub fn structure_of_live(chassis: &Unit, armor: &ArmorUnit, head: &Unit) -> i32 {
    // Divide-by-10 matches CConstructor.cpp:898 — the C++ displays
    // `structure / 10` as the UI value.
    raw_hitpoint_of_live(chassis, armor, head) / 10
}

/// Raw HP sum used by `CMatrixRobotAI::CalcRobotMass` +
/// `InitMaxHitpoint` at MatrixRobot.cpp:4150-4293. This is the value
/// stored on `Robot::hit_point_max` at spawn; the UI only shows it
/// divided by 10 (see [`structure_of_live`]).
pub fn raw_hitpoint_of_live(chassis: &Unit, armor: &ArmorUnit, head: &Unit) -> i32 {
    let chars = config::global().item_chars;
    let mut s = 0i32;
    if !chassis.is_empty() {
        let i = chassis
            .kind
            .as_index()
            .min(chars.chassis_structure.len() - 1);
        s += chars.chassis_structure[i];
    }
    if !armor.unit.is_empty() {
        let i = armor
            .unit
            .kind
            .as_index()
            .min(chars.armor_structure.len() - 1);
        s += chars.armor_structure[i];
    }
    if !head.is_empty() {
        let i = head.kind.as_index().min(chars.head_hp_add.len() - 1);
        s += chars.head_hp_add[i];
    }
    s
}

/// Translate a Base-panel element name (e.g. `"chas3"`, `"hull5"`,
/// `"weap7"`, `"pi2"`) into the corresponding (type, kind, pylon)
/// triple. Used by `form_game::dispatch_ui_click` to dispatch the
/// mountain of button ids.
///
/// Returns `None` for names that aren't constructor buttons.
pub fn parse_constructor_button(name: &str) -> Option<(RobotUnitType, RobotUnitKind, i32)> {
    if let Some(rest) = name.strip_prefix("chas") {
        let n: i32 = rest.parse().ok()?;
        if (1..=5).contains(&n) {
            return Some((RobotUnitType::Chassis, RobotUnitKind(n), 0));
        }
    }
    if let Some(rest) = name.strip_prefix("hull") {
        let n: i32 = rest.parse().ok()?;
        if (1..=6).contains(&n) {
            return Some((RobotUnitType::Armor, RobotUnitKind(n), 0));
        }
    }
    if let Some(rest) = name.strip_prefix("head") {
        // Exclude `headN_st` / `head{empty}` etc. by checking digit-only.
        if let Ok(n) = rest.parse::<i32>() {
            if (1..=4).contains(&n) {
                return Some((RobotUnitType::Head, RobotUnitKind(n), 0));
            }
        }
    }
    if name == "heade" {
        return Some((RobotUnitType::Head, RobotUnitKind::UNKNOWN, 0));
    }
    if let Some(rest) = name.strip_prefix("weap") {
        if let Ok(n) = rest.parse::<i32>() {
            if (1..=10).contains(&n) {
                return Some((RobotUnitType::Weapon, RobotUnitKind(n), 0));
            }
        }
    }
    if name == "weape" {
        return Some((RobotUnitType::Weapon, RobotUnitKind::UNKNOWN, 0));
    }
    // pi1..pi5 (pylon click — cycles the weapon in that pylon)
    if let Some(rest) = name.strip_prefix("pi") {
        if let Ok(n) = rest.parse::<i32>() {
            if (1..=5).contains(&n) {
                // Pylon click encodes as Weapon + unknown kind — the
                // caller uses the pylon index to cycle through the
                // matrix. We encode the pylon in the `pilon` slot
                // (1-indexed in the button name → 0-indexed here).
                return Some((RobotUnitType::Weapon, RobotUnitKind::UNKNOWN, n - 1));
            }
        }
    }
    None
}

// ───────────────────────────────────────────────────────────────────
// SSpecialBot — AI-preset robot templates.
// Port of the SSpecialBot block in CConstructor.{h,cpp}
// (CConstructor.h:60-84 + CConstructor.cpp:1281-1535). Lives in this
// file to mirror the C++ layout where SSpecialBot is declared inside
// the constructor header.
// ───────────────────────────────────────────────────────────────────

/// Port of `SSpecialBot` (CConstructor.h:60-84). Holds the parametric
/// description of one AI-preset robot.
#[derive(Debug, Clone, Copy)]
pub struct SpecialBot {
    pub head: Unit,
    pub weapon: [WeaponUnit; MAX_WEAPON_CNT],
    pub armor: ArmorUnit,
    pub chassis: Unit,
    pub team: i32,

    /// Total resource cost — sum of head/armor/chassis/weapon prices.
    pub resources: [i32; MAX_RESOURCES],

    /// `m_Pripor` — priority. Used by the AI to filter templates by
    /// "tier" (the param key from the AI/Robots block is the priority
    /// number).
    pub pripor: i32,
    /// `m_Strength` — sum of `g_Config.m_WeaponStrengthAI[kind]` for
    /// each weapon. Recomputed by [`SpecialBot::calc_strength`].
    pub strength: f32,
    pub have_bomb: bool,
    pub have_repair: bool,
}

impl Default for SpecialBot {
    fn default() -> Self {
        Self {
            head: Unit {
                ty: RobotUnitType::Head,
                ..Unit::empty()
            },
            weapon: [WeaponUnit::empty(); MAX_WEAPON_CNT],
            armor: ArmorUnit {
                unit: Unit {
                    ty: RobotUnitType::Armor,
                    ..Unit::empty()
                },
                ..ArmorUnit::empty()
            },
            chassis: Unit {
                ty: RobotUnitType::Chassis,
                ..Unit::empty()
            },
            team: 0,
            resources: [0; MAX_RESOURCES],
            pripor: 0,
            strength: 0.0,
            have_bomb: false,
            have_repair: false,
        }
    }
}

impl SpecialBot {
    /// Convert the special bot to a `RobotConfig` consumable by
    /// `Building::queue_robot`. Also fills the per-pylon position
    /// using the weapon matrix the way `SpecialBot::GetRobot`
    /// (CConstructor.cpp:1284-1357) does.
    pub fn to_robot_config(&self) -> RobotConfig {
        let mut cfg = RobotConfig::new();
        cfg.chassis = Unit {
            ty: RobotUnitType::Chassis,
            kind: self.chassis.kind,
            price: self.chassis.price,
        };
        cfg.hull = ArmorUnit {
            max_common_weapon_cnt: self.armor.max_common_weapon_cnt,
            max_extra_weapon_cnt: self.armor.max_extra_weapon_cnt,
            unit: Unit {
                ty: RobotUnitType::Armor,
                kind: self.armor.unit.kind,
                price: self.armor.unit.price,
            },
        };
        cfg.head = Unit {
            ty: RobotUnitType::Head,
            kind: self.head.kind,
            price: self.head.price,
        };

        // Re-place weapons against the destination armor's pylon
        // matrix — matches CConstructor.cpp:1303-1314 which calls
        // `CheckWeaponLegality` for each source weapon.
        let matrix = default_weapon_matrix(self.armor.unit.kind);
        let mut placed = [Unit::empty(); MAX_WEAPON_CNT];
        for w in &self.weapon {
            if w.unit.kind.is_empty() {
                continue;
            }
            if let Some(pos) = matrix.find_pylon_for(w.unit.kind, &placed) {
                placed[pos] = Unit {
                    ty: RobotUnitType::Weapon,
                    kind: w.unit.kind,
                    price: w.unit.price,
                };
            }
        }
        cfg.weapon = placed;
        cfg
    }

    /// Port of `SSpecialBot::CalcStrength` (CConstructor.cpp:1499-1508).
    pub fn calc_strength(&mut self) {
        let table = config::global().weapon_strength_ai;
        let mut s = 0.0f32;
        for w in &self.weapon {
            if w.unit.ty != RobotUnitType::Weapon {
                continue;
            }
            s += table.for_kind(w.unit.kind);
        }
        self.strength = s;
    }

    /// Port of `SSpecialBot::DifWeapon` (CConstructor.cpp:1510-1536).
    /// Returns the fraction of self's weapons whose kind also appears
    /// in `other`'s weapons. Symmetric: forwards to the larger-set
    /// side so the divisor is always the bigger weapon list.
    pub fn dif_weapon(&self, other: &SpecialBot) -> f32 {
        let cnt = |bot: &SpecialBot| -> i32 {
            bot.weapon
                .iter()
                .filter(|w| w.unit.ty == RobotUnitType::Weapon)
                .count() as i32
        };
        let t1 = cnt(self);
        let t2 = cnt(other);
        if t2 > t1 {
            return other.dif_weapon(self);
        }
        let mut total = 0.0f32;
        let mut matches_count = 0.0f32;
        for w in &self.weapon {
            if w.unit.ty != RobotUnitType::Weapon {
                continue;
            }
            total += 1.0;
            for ow in &other.weapon {
                if ow.unit.ty != RobotUnitType::Weapon {
                    continue;
                }
                if w.unit.kind == ow.unit.kind {
                    matches_count += 1.0;
                    break;
                }
            }
        }
        if total > 0.0 {
            matches_count / total
        } else {
            0.0
        }
    }
}

/// Container of loaded AI-preset templates. Mirrors the static
/// `m_AIRobotTypeList`/`m_AIRobotTypeCnt` pair on `SSpecialBot`
/// (CConstructor.cpp:1281-1282).
#[derive(Debug, Clone, Default)]
pub struct AIRobotCatalogue {
    pub bots: Vec<SpecialBot>,
}

impl AIRobotCatalogue {
    /// Port of `SSpecialBot::LoadAIRobotType` (CConstructor.cpp:
    /// 1361-1491). Loads templates from the supplied (key, value)
    /// pairs — each value is a comma-separated `Chassis,Armor,Weapons,Head`
    /// row. The key is the priority index (`m_Pripor`).
    ///
    /// Sorts by computed strength descending so the AI's top-choice
    /// lookup hits the toughest template first.
    pub fn load_from_pairs(pairs: &[(String, String)]) -> Self {
        let mut bots: Vec<SpecialBot> = Vec::new();
        for (key, value) in pairs {
            let pripor = key.trim().parse::<i32>().unwrap_or(0);
            if pripor < 1 {
                continue;
            }
            let parts: Vec<&str> = value.split(',').map(|s| s.trim()).collect();
            if parts.len() < 2 {
                continue;
            }
            let mut bot = SpecialBot {
                pripor,
                ..SpecialBot::default()
            };

            // Chassis (CConstructor.cpp:1402-1407).
            let chassis_kind = match parts[0] {
                "Pneumatic" | "P" => RobotUnitKind::CHASSIS_PNEUMATIC,
                "Whell" | "Wheel" | "W" => RobotUnitKind::CHASSIS_WHEEL,
                "Track" | "T" => RobotUnitKind::CHASSIS_TRACK,
                "Hovercraft" | "H" => RobotUnitKind::CHASSIS_HOVERCRAFT,
                "Antigravity" | "A" => RobotUnitKind::CHASSIS_ANTIGRAVITY,
                _ => continue,
            };
            bot.chassis.kind = chassis_kind;

            // Armor (CConstructor.cpp:1413-1419). The C++ encodes
            // armor names "1", "1S", "2", "2S", "3", "4S".
            let armor_kind = match parts[1] {
                "1" => RobotUnitKind::ARMOR_6,
                "1S" => RobotUnitKind::ARMOR_PASSIVE,
                "2" => RobotUnitKind::ARMOR_ACTIVE,
                "2S" => RobotUnitKind::ARMOR_FIREPROOF,
                "3" => RobotUnitKind::ARMOR_PLASMIC,
                "4S" => RobotUnitKind::ARMOR_NUCLEAR,
                _ => continue,
            };
            bot.armor.unit.kind = armor_kind;

            // Cache cost / weapon caps.
            let prices = config::global().prices;
            bot.chassis.price = prices.get(RobotUnitType::Chassis, chassis_kind);
            bot.armor.unit.price = prices.get(RobotUnitType::Armor, armor_kind);
            let mat = default_weapon_matrix(armor_kind);
            bot.armor.max_common_weapon_cnt = mat.common;
            bot.armor.max_extra_weapon_cnt = mat.extra;
            for j in 0..MAX_RESOURCES {
                bot.resources[j] += bot.chassis.price.resources[j];
                bot.resources[j] += bot.armor.unit.price.resources[j];
            }

            // Weapons (CConstructor.cpp:1426-1456). Each char in the
            // weapon string maps to a kind.
            if parts.len() >= 3 {
                let wstr = parts[2];
                let mut cnt_normal = 0;
                let mut cnt_extra = 0;
                for (u, ch) in wstr.chars().enumerate() {
                    if u >= MAX_WEAPON_CNT {
                        break;
                    }
                    let (kind, normal) = match ch {
                        'G' => (RobotUnitKind::WEAPON_MACHINEGUN, true),
                        'C' => (RobotUnitKind::WEAPON_CANNON, true),
                        'M' => (RobotUnitKind::WEAPON_MISSILE, true),
                        'F' => (RobotUnitKind::WEAPON_FLAMETHROWER, true),
                        'O' => (RobotUnitKind::WEAPON_MORTAR, false),
                        'L' => (RobotUnitKind::WEAPON_LASER, true),
                        'B' => {
                            bot.have_bomb = true;
                            (RobotUnitKind::WEAPON_BOMB, false)
                        }
                        'P' => (RobotUnitKind::WEAPON_PLASMA, true),
                        'E' => (RobotUnitKind::WEAPON_ELECTRIC, true),
                        'R' => {
                            bot.have_repair = true;
                            (RobotUnitKind::WEAPON_REPAIR, true)
                        }
                        _ => continue,
                    };
                    if normal {
                        cnt_normal += 1;
                    } else {
                        cnt_extra += 1;
                    }
                    bot.weapon[u] = WeaponUnit {
                        pos: u as i32,
                        unit: Unit {
                            ty: RobotUnitType::Weapon,
                            kind,
                            price: prices.get(RobotUnitType::Weapon, kind),
                        },
                    };
                    for j in 0..MAX_RESOURCES {
                        bot.resources[j] += bot.weapon[u].unit.price.resources[j];
                    }
                }
                if cnt_normal > bot.armor.max_common_weapon_cnt
                    || cnt_extra > bot.armor.max_extra_weapon_cnt
                {
                    log::warn!(
                        "special_bot: pripor {} weapon row exceeds armor caps (normal {} > {}, extra {} > {})",
                        pripor,
                        cnt_normal,
                        bot.armor.max_common_weapon_cnt,
                        cnt_extra,
                        bot.armor.max_extra_weapon_cnt
                    );
                }
            }

            // Head (CConstructor.cpp:1458-1471).
            if parts.len() >= 4 {
                let head_kind = match parts[3] {
                    "Strength" | "S" => RobotUnitKind::HEAD_BLOCKER,
                    "Dynamo" | "D" => RobotUnitKind::HEAD_DYNAMO,
                    "Locator" | "L" => RobotUnitKind::HEAD_LOCKATOR,
                    "Firewall" | "F" => RobotUnitKind::HEAD_FIREWALL,
                    _ => RobotUnitKind::UNKNOWN,
                };
                if !head_kind.is_empty() {
                    bot.head.kind = head_kind;
                    bot.head.price = prices.get(RobotUnitType::Head, head_kind);
                    for j in 0..MAX_RESOURCES {
                        bot.resources[j] += bot.head.price.resources[j];
                    }
                }
            }

            bot.calc_strength();
            bots.push(bot);
        }
        // CConstructor.cpp:1482-1490 — sort by strength descending
        // (bubble sort in the original; stable sort here is fine).
        bots.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { bots }
    }

    /// Load the AI catalogue from the root `AIRobotType` block — the
    /// `SSpecialBot::LoadAIRobotType(*g_MatrixData->BlockGet(L"AIRobotType"))`
    /// call at MatrixGame.cpp:412.
    pub fn from_matrix_data(stor: &crate::matrix_lib::base::storage::Storage) -> Self {
        let Some(robots_rec) = stor.block_record("da", "AIRobotType") else {
            return Self::default();
        };
        let (Some(keys), Some(values)) = (
            stor.get_buf(&robots_rec, "0"),
            stor.get_buf(&robots_rec, "1"),
        ) else {
            return Self::default();
        };
        let n = keys.arrays_count().min(values.arrays_count());
        let pairs: Vec<(String, String)> = (0..n)
            .map(|i| (keys.get_as_wstr(i), values.get_as_wstr(i)))
            .collect();
        Self::load_from_pairs(&pairs)
    }
}

/// Process-wide AI-robot catalogue. Mirrors the `SSpecialBot::m_AIRobotTypeList`
/// static array. Loaded once at startup, read by the AI sides.
static GLOBAL_AI_ROBOTS: std::sync::OnceLock<std::sync::RwLock<std::sync::Arc<AIRobotCatalogue>>> =
    std::sync::OnceLock::new();

pub fn global_ai_robots() -> std::sync::Arc<AIRobotCatalogue> {
    GLOBAL_AI_ROBOTS
        .get_or_init(|| std::sync::RwLock::new(std::sync::Arc::new(AIRobotCatalogue::default())))
        .read()
        .unwrap()
        .clone()
}

pub fn set_global_ai_robots(c: AIRobotCatalogue) {
    let slot = GLOBAL_AI_ROBOTS
        .get_or_init(|| std::sync::RwLock::new(std::sync::Arc::new(AIRobotCatalogue::default())));
    *slot.write().unwrap() = std::sync::Arc::new(c);
}

/// Port of `CConstructor::BuildRandomBot` (CConstructor.h:137-155).
/// Picks a random valid combination. Returns the produced `RobotConfig`.
pub fn build_random_bot(rng: &mut Rnd) -> RobotConfig {
    let mut cfg = RobotConfig::new();

    // Chassis (CConstructor.h:140).
    let rnd = rng.range(1, 5);
    cfg.chassis = Unit {
        ty: RobotUnitType::Chassis,
        kind: RobotUnitKind(rnd),
        price: config::global()
            .prices
            .get(RobotUnitType::Chassis, RobotUnitKind(rnd)),
    };

    // Armor (CConstructor.h:145).
    let rnd = rng.range(1, 6);
    cfg.hull.unit = Unit {
        ty: RobotUnitType::Armor,
        kind: RobotUnitKind(rnd),
        price: config::global()
            .prices
            .get(RobotUnitType::Armor, RobotUnitKind(rnd)),
    };
    let mat = default_weapon_matrix(cfg.hull.unit.kind);
    cfg.hull.max_common_weapon_cnt = mat.common;
    cfg.hull.max_extra_weapon_cnt = mat.extra;

    // Weapons (CConstructor.h:149-152). The C++ loops `nC <= rnd`
    // (one extra) and rolls 1..9 each iteration. We place via the
    // weapon matrix to keep slot legality.
    let rnd = rng.range(1, 5);
    let mut current = [Unit::empty(); MAX_WEAPON_CNT];
    for _ in 0..=rnd {
        let kind = RobotUnitKind(rng.range(1, 9));
        if let Some(pos) = mat.find_pylon_for(kind, &current) {
            current[pos] = Unit {
                ty: RobotUnitType::Weapon,
                kind,
                price: config::global().prices.get(RobotUnitType::Weapon, kind),
            };
        }
    }
    cfg.weapon = current;

    // Head (CConstructor.h:154).
    let rnd = rng.range(1, 7).min(4);
    cfg.head = Unit {
        ty: RobotUnitType::Head,
        kind: RobotUnitKind(rnd),
        price: config::global()
            .prices
            .get(RobotUnitType::Head, RobotUnitKind(rnd)),
    };

    cfg
}

/// Total cost helper — reads `bot.resources` directly (already cached
/// at load time). Wraps it in a `UnitPrice` for parity with the
/// constructor's pricing path.
pub fn special_bot_cost(bot: &SpecialBot) -> UnitPrice {
    UnitPrice {
        resources: bot.resources,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_construction_clears_live_state() {
        let mut b = RobotBuilder::new();
        b.operate_unit(RobotUnitType::Chassis, RobotUnitKind::CHASSIS_TRACK);
        b.operate_unit(RobotUnitType::Armor, RobotUnitKind::ARMOR_ACTIVE);
        assert!(!b.live_chassis.is_empty());
        b.reset_construction();
        assert!(b.live_chassis.is_empty());
        assert!(b.live_armor.unit.is_empty());
    }

    #[test]
    fn armor_requires_chassis_first() {
        let mut b = RobotBuilder::new();
        // Armor without chassis → ignored (CConstructor.cpp:681 guard).
        b.operate_unit(RobotUnitType::Armor, RobotUnitKind::ARMOR_PASSIVE);
        assert!(b.live_armor.unit.is_empty());

        b.operate_unit(RobotUnitType::Chassis, RobotUnitKind::CHASSIS_WHEEL);
        b.operate_unit(RobotUnitType::Armor, RobotUnitKind::ARMOR_PASSIVE);
        assert_eq!(b.live_armor.unit.kind, RobotUnitKind::ARMOR_PASSIVE);
    }

    #[test]
    fn head_requires_armor_first() {
        let mut b = RobotBuilder::new();
        b.operate_unit(RobotUnitType::Chassis, RobotUnitKind::CHASSIS_TRACK);
        b.operate_unit(RobotUnitType::Head, RobotUnitKind::HEAD_BLOCKER);
        assert!(b.live_head.is_empty());
        b.operate_unit(RobotUnitType::Armor, RobotUnitKind::ARMOR_ACTIVE);
        b.operate_unit(RobotUnitType::Head, RobotUnitKind::HEAD_BLOCKER);
        assert_eq!(b.live_head.kind, RobotUnitKind::HEAD_BLOCKER);
    }

    #[test]
    fn changing_armor_resets_weapons() {
        // The weapon matrix is normally populated from armor VOs at
        // startup; tests don't load VOs, so seed a synthetic 4-common
        // matrix for ARMOR_NUCLEAR and 2-common for ARMOR_PASSIVE so
        // weapon insertion / reset logic can exercise.
        let mut nuc = crate::matrix_game::map::WeaponMatrix::default();
        for _ in 0..4 {
            nuc.list[nuc.cnt as usize] = crate::matrix_game::map::WeaponMatrixSlot {
                id: 20 + nuc.cnt,
                // bits 0,1,2,3,5,7,8,9 — kinds 1,2,3,4,6,8,9,10
                // (machinegun..repair, excluding mortar=5 and bomb=7
                // which are extras; ACCESS_EXTRA_BIT_A=bit 4 / B=bit 6
                // must stay zero for a common slot).
                access_invert: 0b0000_0011_1010_1111,
            };
            nuc.cnt += 1;
            nuc.common += 1;
        }
        crate::matrix_game::map::set_weapon_matrix_for(RobotUnitKind::ARMOR_NUCLEAR, nuc);
        let mut pas = crate::matrix_game::map::WeaponMatrix::default();
        for _ in 0..2 {
            pas.list[pas.cnt as usize] = crate::matrix_game::map::WeaponMatrixSlot {
                id: 20 + pas.cnt,
                // bits 0,1,2,3,5,7,8,9 — kinds 1,2,3,4,6,8,9,10
                // (machinegun..repair, excluding mortar=5 and bomb=7
                // which are extras; ACCESS_EXTRA_BIT_A=bit 4 / B=bit 6
                // must stay zero for a common slot).
                access_invert: 0b0000_0011_1010_1111,
            };
            pas.cnt += 1;
            pas.common += 1;
        }
        crate::matrix_game::map::set_weapon_matrix_for(RobotUnitKind::ARMOR_PASSIVE, pas);

        let mut b = RobotBuilder::new();
        b.operate_unit(RobotUnitType::Chassis, RobotUnitKind::CHASSIS_TRACK);
        b.operate_unit(RobotUnitType::Armor, RobotUnitKind::ARMOR_NUCLEAR); // 4 common slots
        b.operate_unit(RobotUnitType::Weapon, RobotUnitKind::WEAPON_CANNON);
        assert!(!b.live_weapons[0].unit.is_empty());
        b.operate_unit(RobotUnitType::Armor, RobotUnitKind::ARMOR_PASSIVE);
        // Ports CConstructor.cpp:682 — armor change zero-fills weapons.
        assert!(b.live_weapons[0].unit.is_empty());
    }

    #[test]
    fn parse_button_names_cover_common_ids() {
        assert_eq!(
            parse_constructor_button("chas3"),
            Some((RobotUnitType::Chassis, RobotUnitKind(3), 0))
        );
        assert_eq!(
            parse_constructor_button("hull6"),
            Some((RobotUnitType::Armor, RobotUnitKind(6), 0))
        );
        assert_eq!(
            parse_constructor_button("head2"),
            Some((RobotUnitType::Head, RobotUnitKind(2), 0))
        );
        assert_eq!(
            parse_constructor_button("weap7"),
            Some((RobotUnitType::Weapon, RobotUnitKind(7), 0))
        );
        assert_eq!(
            parse_constructor_button("pi3"),
            Some((RobotUnitType::Weapon, RobotUnitKind::UNKNOWN, 2))
        );
        assert_eq!(parse_constructor_button("cobuild"), None);
    }

    #[test]
    fn random_bot_always_has_chassis_and_armor() {
        let mut rng = Rnd::new(42);
        for _ in 0..50 {
            let cfg = build_random_bot(&mut rng);
            assert!(!cfg.chassis.is_empty());
            assert!(!cfg.hull.unit.is_empty());
        }
    }

    #[test]
    fn special_bot_loads_track_2s_with_two_machineguns() {
        let pairs = vec![("1".to_string(), "T,2S,GG,S".to_string())];
        let cat = AIRobotCatalogue::load_from_pairs(&pairs);
        assert_eq!(cat.bots.len(), 1);
        let bot = &cat.bots[0];
        assert_eq!(bot.pripor, 1);
        assert_eq!(bot.chassis.kind, RobotUnitKind::CHASSIS_TRACK);
        assert_eq!(bot.armor.unit.kind, RobotUnitKind::ARMOR_FIREPROOF);
        assert_eq!(bot.head.kind, RobotUnitKind::HEAD_BLOCKER);
        assert_eq!(bot.weapon[0].unit.kind, RobotUnitKind::WEAPON_MACHINEGUN);
        assert_eq!(bot.weapon[1].unit.kind, RobotUnitKind::WEAPON_MACHINEGUN);
    }

    #[test]
    fn dif_weapon_returns_one_for_identical_kits() {
        let pairs = vec![
            ("1".to_string(), "T,2,GG,S".to_string()),
            ("2".to_string(), "T,2,GG,S".to_string()),
        ];
        let cat = AIRobotCatalogue::load_from_pairs(&pairs);
        assert_eq!(cat.bots[0].dif_weapon(&cat.bots[1]), 1.0);
    }
}

// ── Constructor preview render state ────────────────────────────────────
//
// Port of the constructor panel's 3D preview renderer. The C++
// `CConstructor::Render` (CConstructor.cpp:264-360) sets up its own
// D3DVIEWPORT9 sub-rect, a per-preview directional light, and draws
// the in-progress robot via `m_Robot->Draw()` with `SetInterfaceDraw(true)`.
//
// Our port reuses the existing `object_robot::RobotsRenderer` and emits
// a `PreviewTicket` here that the form-game render loop feeds to
// `RobotsRenderer::render_preview_full`. The viewport sub-rect is plumbed
// through as a scissor rect derived from the Base panel resolved origin +
// the design-space `preview_view` rect, and the per-preview directional
// light + ambient (CConstructor.cpp:283-306) are uploaded into the
// renderer's uniform buffer for the preview draw pass.

use crate::matrix_game::robot::ChassisKind;

/// A render ticket populated by the refresh step each frame. The actual
/// GPU draw is handled by `RobotsRenderer::render_preview`; emitting the
/// ticket here keeps the game logic independent of the GPU plumbing.
#[derive(Debug, Clone, Copy)]
pub struct PreviewTicket {
    pub chassis: ChassisKind,
    /// Design-space panel coords (pre-scale) of the viewport rect.
    pub design_rect: [f32; 4],
    /// Rotation angle (radians) for the slow turntable effect the original
    /// achieves via `m_Forward` rotation + D3DXMatrixLookAtLH.
    pub rotation_rad: f32,
}

#[derive(Debug, Clone, Default)]
pub struct BuilderPreview {
    last_chassis: Option<ChassisKind>,
}

impl BuilderPreview {
    pub fn new() -> Self {
        Self::default()
    }

    /// Per-frame tick. The C++ `CConstructor::BeforeRender`
    /// (CConstructor.cpp:251-262) has the rotation block commented out —
    /// the preview robot stays at a fixed `m_Forward = (1,0,0)` set by
    /// the constructor (CConstructor.cpp:44). Kept as a no-op so callers
    /// don't have to special-case the panel-active branch.
    pub fn tick(&mut self, _step_ms: f32) {}

    /// Build a draw ticket for the given preset. Returns `None` when the
    /// config's chassis isn't set (nothing to render yet).
    pub fn ticket(&mut self, cfg: &RobotConfig, design_rect: [f32; 4]) -> Option<PreviewTicket> {
        let chassis = chassis_from_cfg(cfg)?;
        self.last_chassis = Some(chassis);
        // Faithful port of CConstructor.cpp:44 — `m_Robot->m_Forward =
        // D3DXVECTOR3(1, 0, 0)`. The robot model's local +Y is its
        // "forward"; the C++ matrix-builder maps local +Y to the world
        // m_Forward direction, so to face +X (matching the camera at
        // (80,-30) looking at origin) the chassis world matrix is a
        // -π/2 rotation around Z. The renderer's `chassis_world` is a
        // simple Z-rotation by `rotation_rad`, so passing -π/2 here
        // reproduces the C++ orientation exactly.
        Some(PreviewTicket {
            chassis,
            design_rect,
            rotation_rad: -std::f32::consts::FRAC_PI_2,
        })
    }
}

fn chassis_from_cfg(cfg: &RobotConfig) -> Option<ChassisKind> {
    match cfg.chassis.kind {
        k if k == RobotUnitKind::CHASSIS_PNEUMATIC => Some(ChassisKind::Pneumatic),
        k if k == RobotUnitKind::CHASSIS_WHEEL => Some(ChassisKind::Wheel),
        k if k == RobotUnitKind::CHASSIS_TRACK => Some(ChassisKind::Track),
        k if k == RobotUnitKind::CHASSIS_HOVERCRAFT => Some(ChassisKind::Hovercraft),
        k if k == RobotUnitKind::CHASSIS_ANTIGRAVITY => Some(ChassisKind::AntiGravity),
        _ => None,
    }
}

// ── Robot-composition types (Interface/CConstructor.h) ──────────────────
//
// Port of `SPrice`, `SUnit`, `SArmorUnit`, `SWeaponUnit`, `SRobotConfig`
// from Interface/CConstructor.h:13-189. Plain data — no rendering, no
// globals.

use crate::matrix_game::config::Resource;

/// Port of `SPrice` (Interface/CConstructor.h:13-23). Per-component
/// resource cost vector. Zero for empty / unknown components.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitPrice {
    pub resources: [i32; MAX_RESOURCES],
}

impl UnitPrice {
    pub const fn zero() -> Self {
        Self {
            resources: [0; MAX_RESOURCES],
        }
    }

    pub fn titan(&self) -> i32 {
        self.resources[Resource::Titan as usize]
    }
    pub fn electronics(&self) -> i32 {
        self.resources[Resource::Electronics as usize]
    }
    pub fn energy(&self) -> i32 {
        self.resources[Resource::Energy as usize]
    }
    pub fn plasma(&self) -> i32 {
        self.resources[Resource::Plasma as usize]
    }

    pub fn add_from(&mut self, other: UnitPrice) {
        for i in 0..MAX_RESOURCES {
            self.resources[i] += other.resources[i];
        }
    }

    pub fn is_zero(&self) -> bool {
        self.resources.iter().all(|&r| r == 0)
    }
}

/// Port of `SUnit` (Interface/CConstructor.h:25-31). A single
/// (type, kind) component with its own cost cached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Unit {
    pub ty: RobotUnitType,
    pub kind: RobotUnitKind,
    pub price: UnitPrice,
}

impl Unit {
    pub const fn empty() -> Self {
        Self {
            ty: RobotUnitType::Empty,
            kind: RobotUnitKind::UNKNOWN,
            price: UnitPrice::zero(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.kind.is_empty() || self.ty == RobotUnitType::Empty
    }
}

/// Port of `SArmorUnit` (Interface/CConstructor.h:33-40). Adds the
/// per-armor weapon-slot caps to a plain `Unit`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArmorUnit {
    /// Common (non-extra) weapon slots exposed by this armor. Filled
    /// from `SRobotWeaponMatrix::common` at OperateUnit-MRT_ARMOR time
    /// (CConstructor.cpp:686).
    pub max_common_weapon_cnt: i32,
    /// Extra ("super") weapon slots — bomb/mortar go here. From
    /// `SRobotWeaponMatrix::extra`.
    pub max_extra_weapon_cnt: i32,
    pub unit: Unit,
}

impl ArmorUnit {
    pub const fn empty() -> Self {
        Self {
            max_common_weapon_cnt: 0,
            max_extra_weapon_cnt: 0,
            unit: Unit::empty(),
        }
    }
}

/// Port of `SWeaponUnit` (Interface/CConstructor.h:42-47). A weapon
/// slot with its pylon position within the armor's slot layout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WeaponUnit {
    pub pos: i32,
    pub unit: Unit,
}

impl WeaponUnit {
    pub const fn empty() -> Self {
        Self {
            pos: 0,
            unit: Unit::empty(),
        }
    }
}

/// Port of `SRobotConfig` (Interface/CConstructor.h:171-189). The
/// complete parametric description of one robot design — what the
/// constructor panel persists in `m_Configs[PRESETS]` and what the
/// build stack needs to produce a fully-configured robot on dequeue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RobotConfig {
    pub head: Unit,
    pub weapon: [Unit; MAX_WEAPON_CNT],
    pub chassis: Unit,
    pub hull: ArmorUnit,

    /// Cached running totals — kept for parity with C++'s
    /// `m_titX/m_elecX/m_enerX/m_plasX` fields updated by
    /// `SetLabelsAndPrice`.
    pub tit_x: i32,
    pub elec_x: i32,
    pub ener_x: i32,
    pub plas_x: i32,

    pub structure: i32,
    pub damage: i32,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl RobotConfig {
    /// Ports the ctor side-effects of `CConstructorPanel` that seed the
    /// type tags so `m_Hull.m_Unit.m_nType == MRT_ARMOR` etc. even
    /// before the player picks a kind (CConstructor.h:228-233).
    pub const fn new() -> Self {
        Self {
            head: Unit {
                ty: RobotUnitType::Head,
                kind: RobotUnitKind::UNKNOWN,
                price: UnitPrice::zero(),
            },
            weapon: [Unit {
                ty: RobotUnitType::Weapon,
                kind: RobotUnitKind::UNKNOWN,
                price: UnitPrice::zero(),
            }; MAX_WEAPON_CNT],
            chassis: Unit {
                ty: RobotUnitType::Chassis,
                kind: RobotUnitKind::UNKNOWN,
                price: UnitPrice::zero(),
            },
            hull: ArmorUnit {
                max_common_weapon_cnt: 0,
                max_extra_weapon_cnt: 0,
                unit: Unit {
                    ty: RobotUnitType::Armor,
                    kind: RobotUnitKind::UNKNOWN,
                    price: UnitPrice::zero(),
                },
            },
            tit_x: 0,
            elec_x: 0,
            ener_x: 0,
            plas_x: 0,
            structure: 0,
            damage: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// True once every required slot is populated (the build button
    /// gates on this — see the `m_nUnitCnt > 0` check at
    /// CConstructor.cpp:67 / :149).
    pub fn is_buildable(&self) -> bool {
        !self.chassis.is_empty() && !self.hull.unit.is_empty()
    }

    /// Port of `m_nUnitCnt` tally at CConstructor.cpp:709-725.
    pub fn unit_count(&self) -> i32 {
        let mut n = 0;
        if !self.chassis.is_empty() {
            n += 1;
        }
        if !self.hull.unit.is_empty() {
            n += 1;
        }
        if !self.head.is_empty() {
            n += 1;
        }
        for w in &self.weapon {
            if !w.is_empty() {
                n += 1;
            }
        }
        n
    }

    /// Port of `CConstructorPanel::ResetWeapon` (CConstructor.h:212).
    pub fn reset_weapons(&mut self) {
        for w in &mut self.weapon {
            *w = Unit {
                ty: RobotUnitType::Weapon,
                kind: RobotUnitKind::UNKNOWN,
                price: UnitPrice::zero(),
            };
        }
        self.damage = 0;
    }
}

#[cfg(test)]
mod robot_config_tests {
    use super::*;
    use crate::matrix_game::map::{default_weapon_matrix, WeaponMatrix};

    #[test]
    fn robot_config_is_buildable_needs_chassis_and_hull() {
        let mut c = RobotConfig::new();
        assert!(!c.is_buildable());
        c.chassis.kind = RobotUnitKind::CHASSIS_TRACK;
        assert!(!c.is_buildable());
        c.hull.unit.kind = RobotUnitKind::ARMOR_PASSIVE;
        assert!(c.is_buildable());
    }

    #[test]
    fn unit_count_matches_populated_slots() {
        let mut c = RobotConfig::new();
        assert_eq!(c.unit_count(), 0);
        c.chassis.kind = RobotUnitKind::CHASSIS_WHEEL;
        c.hull.unit.kind = RobotUnitKind::ARMOR_ACTIVE;
        c.head.kind = RobotUnitKind::HEAD_BLOCKER;
        c.weapon[0].kind = RobotUnitKind::WEAPON_MACHINEGUN;
        assert_eq!(c.unit_count(), 4);
    }

    /// Build a synthetic 2-common-+1-extra weapon matrix for tests.
    /// Mirrors the layout the C++ `RobotPreload` produces from a
    /// typical armor.vo, so the tests below stay independent of any
    /// real VO loading at test-time (which doesn't happen).
    fn fake_weapon_matrix(common: i32, extra: i32) -> WeaponMatrix {
        let mut m = WeaponMatrix::default();
        let mut next_id = 20;
        for _ in 0..common {
            m.list[m.cnt as usize] = crate::matrix_game::map::WeaponMatrixSlot {
                id: next_id,
                // Allow weapon kinds 1..=10 except the extra-only ones.
                // bits 0,1,2,3,5,7,8,9 — kinds 1,2,3,4,6,8,9,10
                // (machinegun..repair, excluding mortar=5 and bomb=7
                // which are extras; ACCESS_EXTRA_BIT_A=bit 4 / B=bit 6
                // must stay zero for a common slot).
                access_invert: 0b0000_0011_1010_1111,
            };
            m.cnt += 1;
            m.common += 1;
            next_id += 1;
        }
        for _ in 0..extra {
            m.list[m.cnt as usize] = crate::matrix_game::map::WeaponMatrixSlot {
                id: next_id,
                // Bit 4 (BOMB) and bit 6 (FLAMETHROWER) flag the slot
                // as extra per ACCESS_EXTRA_BIT_*.
                access_invert: (1 << 4) | (1 << 6),
            };
            m.cnt += 1;
            m.extra += 1;
            next_id += 1;
        }
        m
    }

    #[test]
    fn find_pylon_for_common_weapon_returns_first_empty() {
        crate::matrix_game::map::set_weapon_matrix_for(
            RobotUnitKind::ARMOR_ACTIVE,
            fake_weapon_matrix(2, 1),
        );
        let m: WeaponMatrix = default_weapon_matrix(RobotUnitKind::ARMOR_ACTIVE);
        let current = [Unit::empty(); MAX_WEAPON_CNT];
        assert_eq!(
            m.find_pylon_for(RobotUnitKind::WEAPON_CANNON, &current),
            Some(0)
        );
    }

    #[test]
    fn find_pylon_for_super_weapon_returns_extra_slot() {
        crate::matrix_game::map::set_weapon_matrix_for(
            RobotUnitKind::ARMOR_ACTIVE,
            fake_weapon_matrix(2, 1),
        );
        let m = default_weapon_matrix(RobotUnitKind::ARMOR_ACTIVE);
        let current = [Unit::empty(); MAX_WEAPON_CNT];
        assert_eq!(
            m.find_pylon_for(RobotUnitKind::WEAPON_BOMB, &current),
            Some(2)
        );
    }

    #[test]
    fn super_djeans_overwrites_occupied_pylon() {
        // ARMOR_PLASMIC chosen to stay clear of the kinds other tests
        // seed (NUCLEAR/PASSIVE/ACTIVE/FIREPROOF).
        crate::matrix_game::map::set_weapon_matrix_for(
            RobotUnitKind::ARMOR_PLASMIC,
            fake_weapon_matrix(2, 1),
        );
        let mut b = RobotBuilder::new();
        b.super_djeans(
            RobotUnitType::Chassis,
            RobotUnitKind::CHASSIS_TRACK,
            0,
            false,
        );
        b.super_djeans(RobotUnitType::Armor, RobotUnitKind::ARMOR_PLASMIC, 0, false);
        b.super_djeans(
            RobotUnitType::Weapon,
            RobotUnitKind::WEAPON_MACHINEGUN,
            0,
            false,
        );
        assert_eq!(
            b.live_weapons[0].unit.kind,
            RobotUnitKind::WEAPON_MACHINEGUN
        );
        // Replace the occupied pylon (CConstructor.cpp:490-492).
        b.super_djeans(
            RobotUnitType::Weapon,
            RobotUnitKind::WEAPON_CANNON,
            0,
            false,
        );
        assert_eq!(b.live_weapons[0].unit.kind, RobotUnitKind::WEAPON_CANNON);
        // Clear it via kind 0.
        b.super_djeans(RobotUnitType::Weapon, RobotUnitKind::UNKNOWN, 0, false);
        assert!(b.live_weapons[0].unit.is_empty());
        // Second common pylon maps to physical slot 1; extra weapon
        // (pilon 4) maps to the first extra slot (index 2 here).
        b.super_djeans(RobotUnitType::Weapon, RobotUnitKind::WEAPON_LASER, 1, false);
        assert_eq!(b.live_weapons[1].unit.kind, RobotUnitKind::WEAPON_LASER);
        b.super_djeans(RobotUnitType::Weapon, RobotUnitKind::WEAPON_BOMB, 4, false);
        assert_eq!(b.live_weapons[2].unit.kind, RobotUnitKind::WEAPON_BOMB);
    }

    #[test]
    fn remote_operate_unit_cycles_kinds() {
        crate::matrix_game::map::set_weapon_matrix_for(
            RobotUnitKind::ARMOR_PLASMIC,
            fake_weapon_matrix(2, 1),
        );
        let mut b = RobotBuilder::new();
        b.super_djeans(
            RobotUnitType::Chassis,
            RobotUnitKind::CHASSIS_TRACK,
            0,
            false,
        );
        b.super_djeans(RobotUnitType::Armor, RobotUnitKind::ARMOR_PLASMIC, 0, false);

        // Chassis 3 → 4; 5 wraps to 1.
        b.remote_operate_unit("pich");
        assert_eq!(b.cfg().chassis.kind.0, RobotUnitKind::CHASSIS_TRACK.0 + 1);
        b.cfg_mut().chassis.kind = RobotUnitKind(5);
        b.remote_operate_unit("pich");
        assert_eq!(b.cfg().chassis.kind.0, 1);

        // Head cycles through 0 (empty allowed).
        assert!(b.cfg().head.kind.is_empty());
        b.remote_operate_unit("pihe");
        assert_eq!(b.cfg().head.kind.0, 1);
        b.cfg_mut().head.kind = RobotUnitKind(4);
        b.remote_operate_unit("pihe");
        assert_eq!(b.cfg().head.kind.0, 0);

        // Common weapon pylon skips mortar (5) and bomb (7): 4→6, 6→8,
        // 10 wraps to 0, 0→1.
        b.super_djeans(RobotUnitType::Weapon, RobotUnitKind(4), 0, false);
        b.remote_operate_unit("pi1");
        assert_eq!(b.cfg().weapon[0].kind.0, 6);
        b.remote_operate_unit("pi1");
        assert_eq!(b.cfg().weapon[0].kind.0, 8);
        b.cfg_mut().weapon[0].kind = RobotUnitKind(10);
        b.remote_operate_unit("pi1");
        assert_eq!(b.cfg().weapon[0].kind.0, 0);
        b.remote_operate_unit("pi1");
        assert_eq!(b.cfg().weapon[0].kind.0, 1);

        // Extra pylon: 0→5→7→0.
        b.remote_operate_unit("pi5");
        assert_eq!(b.cfg().weapon[4].kind.0, 5);
        b.remote_operate_unit("pi5");
        assert_eq!(b.cfg().weapon[4].kind.0, 7);
        b.remote_operate_unit("pi5");
        assert_eq!(b.cfg().weapon[4].kind.0, 0);
    }

    #[test]
    fn weapon_matrix_is_extra_slot_flags_match() {
        // ARMOR_FIREPROOF chosen so other tests that seed ARMOR_NUCLEAR
        // / ARMOR_ACTIVE / ARMOR_PASSIVE for their own assertions don't
        // race with this one when cargo runs the test set in parallel.
        crate::matrix_game::map::set_weapon_matrix_for(
            RobotUnitKind::ARMOR_FIREPROOF,
            fake_weapon_matrix(4, 1),
        );
        let m = default_weapon_matrix(RobotUnitKind::ARMOR_FIREPROOF);
        assert!(!m.is_extra_slot(0));
        assert!(!m.is_extra_slot(3));
        assert!(m.is_extra_slot(4));
    }
}
