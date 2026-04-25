//! Port of `CIFaceList` (Interface/CInterface.h:269+ / CInterface.cpp).
//!
//! Global container of loaded `CInterface` panels + the focused
//! element tracker. The original maintains a doubly-linked list;
//! we use a `Vec` since iteration order is always front-to-back.
//!
//! Also carries the shared mouse-event plumbing: the game calls
//! `on_mouse_move` / `on_mouse_down` / `on_mouse_up` once per event
//! and this routes to the topmost panel whose element catches the
//! cursor (matches `CIFaceList::OnMouseMove` at CInterface.cpp:
//! ~3000+).

use crate::matrix_lib::base::storage::Storage;

use super::counter::CIFaceCounter;
use super::hint::{HintChromeLibrary, HintReplacer, HintSystem, TemplateLibrary};
use super::history::ConfigHistory;
use super::iface_element::ElementState;
use super::interface::{CInterface, DESIGN_H};
use crate::matrix_game::interface::constructor::RobotConfig;

// ── Turret-build placement state ────────────────────────────────────────
//
// Port of `PREORDER_BUILD_TURRET` (CInterface.h:57) + `BeginBuildTurret`
// (CInterface.h:379, CInterface.cpp:4650+) + the turret-placement
// rendering/click hooks at MatrixFormGame.cpp:1405, 1498-1512 and
// MatrixMap.cpp:1396, 1692.
//
// The C++ stores placement state globally on `CIFaceList::m_IfListFlags`
// plus a scratch turret instance on `CIFaceList`. We collect it into
// `TurretBuild` since only one turret-build session can be active at a
// time.

/// Port of `ECannonKind` — the 4 cannon variants the UI exposes
/// (RUK_TURRET_CANNON / GUN / LASER / ROCKET). The C++ stores this as
/// `1..=4` matching the UI element ids `turret1..turret4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TurretKind {
    Cannon = 1,
    Gun = 2,
    Laser = 3,
    Rocket = 4,
}

impl TurretKind {
    pub fn from_i32(n: i32) -> Option<Self> {
        Some(match n {
            1 => TurretKind::Cannon,
            2 => TurretKind::Gun,
            3 => TurretKind::Laser,
            4 => TurretKind::Rocket,
            _ => return None,
        })
    }

    pub fn to_index(self) -> usize {
        (self as u8 as usize).saturating_sub(1)
    }
}

/// The turret-build session state. `active` flips on when the player
/// clicks one of the `turret1..turret4` buttons or `buca` on a building;
/// the next map click either places the turret (on a valid building slot)
/// or cancels (anywhere else).
///
/// The placement preview is driven each frame by `update_cursor` (called
/// from the form_game cursor-move + render path): it snaps the cursor
/// to the nearest free turret slot on the parent building, smooths the
/// ghost cannon's angle toward the slot's rest angle, and flips the
/// validity tint. Mirrors `CMatrixSideUnit::LogicTakt` BUILDING_TURRET
/// branch (MatrixSide.cpp:528-585).
#[derive(Debug, Clone, Copy, Default)]
pub struct TurretBuild {
    pub active: bool,
    pub kind: Option<TurretKind>,
    /// The building that owns the turret placement — ports
    /// `g_IFaceList->m_BuildCa` / `m_CannonForBuild` (CInterface.h).
    pub parent: Option<crate::matrix_game::map_static::ObjectId>,
    /// Cursor hover position in world space — drives the placement preview.
    pub cursor_world: (f32, f32),
    /// Which turret slot on the parent building the cursor is over
    /// (0..turrets_max-1). `None` = cursor not over any free slot — the
    /// ghost falls back to the raw cursor position with the red tint.
    pub hovered_slot: Option<i32>,
    /// Ghost cannon's current angle in radians — interpolates toward
    /// the snapped slot's rest angle each tick using the C++ damping
    /// formula `dang * (1 - 0.99^ms)` (MatrixSide.cpp:578).
    pub ghost_angle: f32,
    /// Ghost cannon's current world-space XY (already snapped if a slot
    /// is hovered, else the raw terrain cursor).
    pub ghost_pos: glam::Vec2,
    /// Ghost cannon's pos-Z — taken from the parent base's `build_z`
    /// + the platform offset on `begin`, then held constant. The C++
    /// pulls this from the base's matrix every frame; the Rust port
    /// snaps once at session start since the base doesn't move.
    pub ghost_z: f32,
    /// `m_CanBuildFlag` (MatrixSide.cpp:555/558). True when the cursor
    /// is over a free slot AND the player still has resources for
    /// another turret on the parent.
    pub can_build: bool,
}

impl TurretBuild {
    pub fn new() -> Self {
        Self::default()
    }

    /// Port of the `bca` LMB-up branch (CInterface.cpp:3493-3499) —
    /// flips `PREORDER_BUILD_TURRET` so the kind picker is visible, but
    /// does NOT spawn the placement-preview ghost. The C++ defers the
    /// ghost cannon to `BeginBuildTurret(no)`, which only runs once the
    /// player picks a kind.
    pub fn open_picker(&mut self, parent: crate::matrix_game::map_static::ObjectId) {
        self.active = true;
        self.kind = None;
        self.parent = Some(parent);
        self.cursor_world = (0.0, 0.0);
        self.hovered_slot = None;
        self.ghost_angle = 0.0;
        self.ghost_pos = glam::Vec2::ZERO;
        self.ghost_z = 0.0;
        self.can_build = false;
    }

    /// Port of `CInterface::BeginBuildTurret(no)` (CInterface.cpp:4650+).
    /// Commits the kind — the per-frame placement preview becomes
    /// visible on the next tick.
    pub fn begin(&mut self, kind: TurretKind, parent: crate::matrix_game::map_static::ObjectId) {
        self.active = true;
        self.kind = Some(kind);
        self.parent = Some(parent);
        self.cursor_world = (0.0, 0.0);
        self.hovered_slot = None;
        self.ghost_angle = 0.0;
        self.ghost_pos = glam::Vec2::ZERO;
        self.ghost_z = 0.0;
        self.can_build = false;
    }

    pub fn cancel(&mut self) {
        *self = TurretBuild::default();
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

pub struct IFaceList {
    /// In front-to-back order — the FIRST panel receives events first
    /// (matches `LIST_ADD` prepend behaviour in the original).
    pub panels: Vec<CInterface>,
    /// `(panel_idx, element_idx)` for the currently-focused element,
    /// mirrors `CIFaceList::m_FocusedInterface` + `m_FocusedElement`.
    pub focused: Option<(usize, usize)>,
    /// `(panel_idx, element_idx)` for the currently-pressed-down LMB
    /// element. Cleared on LMB up.
    pub pressed: Option<(usize, usize)>,
    /// Same shape as `pressed` but tracks RMB. The C++
    /// `OnMouseRBDown` fires the action immediately at press-time;
    /// we still capture the press so RB up only fires when release
    /// is on the same element (matches the LMB down/up pattern).
    pub right_pressed: Option<(usize, usize)>,
    /// Active popup menu — port of `CIFaceMenu` / `g_PopupMenu`
    /// (Interface/CIFaceMenu.{cpp,h}). When `Some`, mouse events
    /// route to it before reaching the underlying panel; clicking
    /// outside the popup ramka closes it.
    pub popup: Option<super::iface_menu::CIFaceMenu>,
    /// Saved config to restore after canceling a popup hover-preview.
    pub popup_restore_pending: Option<RobotConfig>,
    /// Port of `g_ConfigHistory` (CHistory.cpp:11) — global history
    /// of committed robot configs cycled by the `hisleft` / `hisright`
    /// buttons.
    pub history: ConfigHistory,
    /// Port of `g_IFaceList->m_RCountControl` (CInterface.h:310) — the
    /// build-multiplier counter beside the `cobuild` button.
    pub r_count_control: CIFaceCounter,
    /// Turret-placement session — port of the
    /// `m_IfListFlags & PREORDER_BUILD_TURRET` flag plus
    /// `m_BuildCa` / `m_CannonForBuild` (CInterface.h:288, 308).
    pub turret_build: TurretBuild,
    /// Template snapshot from `robots.dat::Templates` (read once at
    /// `load_default_panels` and never mutated). Drives `CMatrixHint`
    /// template lookups via `HintSystem::update`.
    pub hint_templates: TemplateLibrary,
    /// 9-slice border + bitmap aliases (`Hints/0` + `Hints/Bitmaps`
    /// in `robots.dat`). Read once at init, consumed by the hint
    /// builder + the renderer.
    pub hint_chrome: HintChromeLibrary,
    /// Baseline + dynamic replacement values. `AddHintReplacements`
    /// (form_game.rs) mutates this each hover; the update tick reads
    /// it when building the hint.
    pub hint_replacer: HintReplacer,
    /// Hover timer + currently-shown hint. Port of
    /// `CIFaceList::m_CurrentHint` + `m_CurrentHintControlName`
    /// (CInterface.h:291-294) plus the per-button hover-show gate at
    /// CIFaceButton.cpp:128-145.
    pub hint_system: HintSystem,
}

/// Outcome of a button click. The C++ dispatches via action arrays
/// (`SAction m_Actions[MAX_ACTIONS]`); we return a simple enum the
/// caller can match on. Populated from the element's `Name` (the
/// same string `if/Main` uses as a button-id, e.g. "buro" / "buca").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Click {
    /// Left-mouse button release on a button element.
    /// Port of `CIFaceButton::OnMouseLBUp` (CIFaceButton.cpp:97-123).
    Button(String),
    /// Right-mouse button release on a button element.
    /// Port of `CIFaceButton::OnMouseRBDown` (CIFaceButton.cpp:183-321) —
    /// the C++ fires the popup-menu open on RBDown; we mirror the
    /// dispatch-on-RBUp pattern of LMB so a click is the press +
    /// release of the same element.
    RightButton(String),
    /// Left-button release on an item inside an active popup menu.
    /// Port of `CIFaceMenu::OnMenuItemPress` (CIFaceMenu.cpp:530+).
    PopupItem {
        parent: super::iface_menu::EMenuParent,
        kind: crate::matrix_game::config::RobotUnitKind,
    },
}

/// `(panel_name, element_name)` for the element that lost or gained
/// focus on a mouse-move event. `None` when no transition happened.
pub type FocusTransition = Option<(String, String)>;

impl IFaceList {
    pub fn new() -> Self {
        Self {
            panels: Vec::new(),
            focused: None,
            pressed: None,
            right_pressed: None,
            popup: None,
            popup_restore_pending: None,
            history: ConfigHistory::new(),
            r_count_control: CIFaceCounter::new(),
            turret_build: TurretBuild::new(),
            hint_templates: TemplateLibrary::default(),
            hint_chrome: HintChromeLibrary::default(),
            hint_replacer: HintReplacer::default(),
            hint_system: HintSystem::default(),
        }
    }

    /// Load the canonical set of panels from `robots.dat`. Ports the
    /// MatrixGame.cpp:474-510 sequence of `pInterface->Load(bpi, IF_*)`.
    /// Panels that fail to load (missing block) are skipped.
    ///
    /// For now show only `Main` — the other panels render at design
    /// positions that overlap with the minimap (MiniM / Radar occupy
    /// the bottom-left where our existing minimap renderer already
    /// owns the space) or need selection-aware visibility (Base /
    /// Hints). Flip them back on as the game-state plumbing catches up.
    pub fn load_default_panels(matrix_data: &Storage) -> Self {
        let mut list = Self::new();
        // Tooltip templates + baseline replacements — loaded once and
        // held on the list. Mirrors the init sequence at
        // MatrixGame.cpp:262-315 where `PAR_REPLACE` is primed out of
        // `Labels.Replaces` before the first hint fires.
        list.hint_templates = TemplateLibrary::load(matrix_data);
        list.hint_replacer = HintReplacer::from_storage(matrix_data);
        list.hint_chrome = HintChromeLibrary::load(matrix_data);
        // CIFaceMenu::LoadMenuGraphics (CIFaceMenu.cpp:53-59) loads the
        // chrome (corners / edges / cursor / cursik) from `if/PopupMenu`
        // into a static panel `m_MenuGraphics`. We push it into the same
        // list as the other panels and keep it invisible — the popup
        // renderer pulls element rects out of it on demand.
        for name in [
            "Top",
            "MiniM",
            "Radar",
            "Base",
            "Main",
            "Hints",
            "PopupMenu",
        ] {
            if let Some(mut p) = CInterface::load(matrix_data, name) {
                // `Top` holds the always-on resource + robot-limit HUD;
                // `Main` holds the selection-driven command panel. Both
                // are visible from start. Others are toggled in by their
                // respective selection / hint paths.
                p.visible = matches!(name, "Main" | "Top");
                list.panels.push(p);
            }
        }
        list
    }

    /// Hit-test the panel stack with a screen-space pixel and return
    /// the topmost (`front-of-vec`) element whose rect contains it.
    /// Ports `CIFaceList::OnMouseMove` (CInterface.cpp ~3000-3200) at
    /// the "find the element under the cursor" stage.
    pub fn hit_test(
        &self,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> Option<(usize, usize)> {
        let scale = (screen_h / DESIGN_H).max(0.1);
        for (pi, p) in self.panels.iter().enumerate() {
            if !p.visible {
                continue;
            }
            let panel_px = p.resolved_pos(screen_w, screen_h, scale);
            // Walk in reverse so the top-drawn element (last in vec)
            // wins when two overlap — mirrors the back-to-front
            // draw order + front-to-back hit-test order.
            for (ei, e) in p.elements.iter().enumerate().rev() {
                if e.hit(panel_px, scale, sx, sy) {
                    return Some((pi, ei));
                }
            }
        }
        None
    }

    /// Mouse-move event handler. Updates the `Focused` state of the
    /// element under the cursor, clearing the previous focus. Returns
    /// `(unfocused_name, focused_name)` describing any transition —
    /// this routes to `RobotBuilder::{focus,unfocus}_element` in the
    /// form-game glue (port of `RemoteFocusElement` / `RemoteUnFocusElement`
    /// at CConstructor.cpp:903-910).
    pub fn on_mouse_move(
        &mut self,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> (FocusTransition, FocusTransition) {
        // Track popup hover so the renderer can highlight the row
        // under the cursor (port of CIFaceMenu::CalcSelectedItem).
        if self.popup.is_some() {
            let hovered = self
                .screen_to_panel_design("Base", sx, sy, screen_w, screen_h)
                .and_then(|(dx, dy)| self.popup.as_ref().unwrap().hit_test_design(dx, dy));
            if let Some(popup) = self.popup.as_mut() {
                popup.hovered = hovered;
            }
            if hovered.is_some() {
                // An open popup suppresses hints — matches the
                // `POPUP_MENU_ACTIVE` gate in `CheckShowHintLogic`
                // (CInterface.cpp:4542-4549).
                self.hint_system.clear();
                return (None, None);
            }
        }
        let new_focus = self.hit_test(sx, sy, screen_w, screen_h);

        let mut prev_pair: FocusTransition = None;
        let mut new_pair: FocusTransition = None;

        // Clear the old focus — but only when the pressed button is
        // a different element. If a press is held and we drag onto
        // a different element, the C++ keeps the pressed element
        // 'Pressed' and no others are focused (CIFaceList::OnMouseMove
        // :3100-3150 roughly). We approximate: pressed element stays
        // pressed; other elements go Normal unless under the cursor.
        //
        // CIFaceStatic::OnMouseMove (CIFaceStatic.cpp:26-51) never
        // changes `m_CurState`; only CIFaceButton does. Gate every
        // transition on `ElementKind::Button` so static frames like
        // `conl`/`conr`/`conf` don't brighten on hover/press.
        if self.focused != new_focus {
            if let Some((pi, ei)) = self.focused {
                let pressed_self = self.pressed == Some((pi, ei));
                if let Some(p) = self.panels.get_mut(pi) {
                    if let Some(e) = p.elements.get_mut(ei) {
                        let is_button = matches!(e.kind, super::iface_element::ElementKind::Button);
                        if is_button && !pressed_self {
                            e.cur_state = e.def_state;
                        }
                        prev_pair = Some((p.name.clone(), e.name.clone()));
                    }
                }
            }
            if let Some((pi, ei)) = new_focus {
                if let Some(p) = self.panels.get_mut(pi) {
                    if let Some(e) = p.elements.get_mut(ei) {
                        let was_normal = matches!(e.cur_state, ElementState::Normal);
                        let is_button = matches!(e.kind, super::iface_element::ElementKind::Button);
                        if is_button {
                            if self.pressed != Some((pi, ei)) {
                                e.cur_state = ElementState::Focused;
                            } else {
                                e.cur_state = ElementState::Pressed;
                            }
                        }
                        new_pair = Some((p.name.clone(), e.name.clone()));
                        // CIFaceButton.cpp:147-169 — focus transition
                        // on a button fires `S_BENTER`.
                        if is_button && was_normal {
                            super::sound::play(super::sound::UiSound::BEnter);
                        }
                    }
                }
            }
            self.focused = new_focus;
        }
        // Refresh the hover target every move so the hint timer
        // resets when the cursor leaves the element. Mirrors
        // CIFaceButton.cpp:128-145 / CIFaceStatic.cpp path.
        self.refresh_hint_hover(screen_w, screen_h);
        (prev_pair, new_pair)
    }

    /// Push the currently-focused element's hint metadata into the
    /// hint timer, including the element's resolved screen rect so
    /// the hint renders at the right offset. No-op when the focus has
    /// no hint template.
    fn refresh_hint_hover(&mut self, screen_w: f32, screen_h: f32) {
        let clear = |h: &mut HintSystem| {
            h.set_hovered(None, None, None, 0, 0, 0.0, 0.0, 0.0, 0.0);
        };
        let Some((pi, ei)) = self.focused else {
            clear(&mut self.hint_system);
            return;
        };
        let scale = (screen_h / DESIGN_H).max(0.1);
        let Some(panel) = self.panels.get(pi) else {
            clear(&mut self.hint_system);
            return;
        };
        let Some(elem) = panel.elements.get(ei) else {
            clear(&mut self.hint_system);
            return;
        };
        if elem.hint_template.is_empty() {
            clear(&mut self.hint_system);
            return;
        }
        let panel_px = panel.resolved_pos(screen_w, screen_h, scale);
        let [ex, ey, w, h] = elem.rect_in_panel(panel_px, scale);
        self.hint_system.set_hovered(
            Some(&panel.name),
            Some(&elem.name),
            Some(&elem.hint_template),
            elem.hint_offset_x,
            elem.hint_offset_y,
            ex,
            ey,
            w,
            h,
        );
    }

    /// Advance the hint timer + rebuild the active hint when the
    /// threshold trips. Call once per frame from `form_game`;
    /// `bitmap_sizer` resolves inline `_BITMAP:` PNG sizes via the
    /// renderer's loaded atlas cache.
    pub fn update(
        &mut self,
        dt_ms: f32,
        screen_w: f32,
        screen_h: f32,
        atlas: &mut super::text::GlyphAtlas,
        bitmap_sizer: &dyn Fn(&str) -> Option<(i32, i32)>,
    ) {
        let scale = (screen_h / DESIGN_H).max(0.1);
        let main_origin = self
            .panel("Main")
            .map(|p| p.resolved_pos(screen_w, screen_h, scale))
            .unwrap_or([0.0, 0.0]);
        let base_origin = self
            .panel("Base")
            .map(|p| p.resolved_pos(screen_w, screen_h, scale))
            .unwrap_or([0.0, 0.0]);
        self.hint_system.update(
            dt_ms,
            &self.hint_templates,
            &self.hint_chrome,
            &self.hint_replacer,
            atlas,
            bitmap_sizer,
            screen_w,
            screen_h,
            scale,
            main_origin,
            base_origin,
        );
    }

    /// Translate a screen-space pixel to design-space coords relative
    /// to the named panel's resolved origin. Used by popup hit-test
    /// (popup design coords are relative to the Base panel).
    pub fn screen_to_panel_design(
        &self,
        panel_name: &str,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> Option<(f32, f32)> {
        let p = self.panel(panel_name)?;
        let scale = (screen_h / DESIGN_H).max(0.1);
        let origin = p.resolved_pos(screen_w, screen_h, scale);
        Some(((sx - origin[0]) / scale, (sy - origin[1]) / scale))
    }

    /// Mouse-button-down event. If an element is under the cursor,
    /// mark it Pressed and record the pressed-handle. Returns true
    /// if the event was consumed by the UI (caller can skip
    /// routing to the world picker).
    pub fn on_mouse_down(&mut self, sx: f32, sy: f32, screen_w: f32, screen_h: f32) -> bool {
        // Popup hit-test first — port of CInterface.cpp:1044-1051.
        // Click inside the popup ramka commits the hovered item;
        // click outside closes the popup (no commit).
        if self.popup.is_some() {
            let inside = self
                .screen_to_panel_design("Base", sx, sy, screen_w, screen_h)
                .map(|(dx, dy)| {
                    let popup = self.popup.as_ref().unwrap();
                    popup.contains_design(dx, dy)
                })
                .unwrap_or(false);
            if !inside {
                self.popup_restore_pending = self.popup.as_ref().and_then(|p| p.saved_config);
                self.popup = None;
                // Fall through to regular hit-test so the click can
                // also engage whatever was clicked outside.
            } else {
                // Treat as "captured by UI"; selection commit on
                // mouse-up via on_mouse_up's popup branch.
                return true;
            }
        }

        let hit = self.hit_test(sx, sy, screen_w, screen_h);
        match hit {
            Some((pi, ei)) => {
                let (panel_name, elem_name, is_button, was_disabled) = self
                    .panels
                    .get(pi)
                    .and_then(|p| {
                        p.elements.get(ei).map(|e| {
                            (
                                p.name.clone(),
                                e.name.clone(),
                                matches!(e.kind, super::iface_element::ElementKind::Button),
                                matches!(e.cur_state, ElementState::Disabled),
                            )
                        })
                    })
                    .unwrap_or_default();
                log::info!(
                    "iface: mouse_down ({:.0},{:.0}) hit panel='{}' elem='{}'",
                    sx,
                    sy,
                    panel_name,
                    elem_name
                );
                if was_disabled {
                    return true;
                }
                // Only CIFaceButton flips state on LBDown (CIFaceButton
                // .cpp:33-95). CIFaceStatic::OnMouseLBDown runs the
                // ON_PRESS action without touching `m_CurState`
                // (CIFaceStatic.cpp:53-57) — so `conl`/`conr`/`conf`
                // etc. must stay at `def_state` when clicked.
                if is_button {
                    if let Some(p) = self.panels.get_mut(pi) {
                        if let Some(e) = p.elements.get_mut(ei) {
                            e.cur_state = ElementState::Pressed;
                        }
                    }
                }
                self.pressed = Some((pi, ei));
                // CIFaceButton.cpp:33-95 — generic LBDown sound dispatch.
                // Disabled buttons don't fire (CIFaceButton.cpp:35-37
                // returns true before any sound).
                if is_button && !was_disabled {
                    super::sound::play(super::sound::for_push_button_down(&elem_name));
                }
                true
            }
            None => {
                log::info!("iface: mouse_down ({:.0},{:.0}) no hit", sx, sy);
                false
            }
        }
    }

    /// Mouse-button-up event. If release happens on the same element
    /// that was pressed, fire a `Click`. Ports
    /// `CIFaceButton::OnMouseLBUp` (Interface/CIFaceButton.cpp).
    pub fn on_mouse_up(&mut self, sx: f32, sy: f32, screen_w: f32, screen_h: f32) -> Option<Click> {
        // Popup-active path — port of CInterface.cpp:1049-1051's
        // `g_PopupMenu->OnMenuItemPress()` call. Returns
        // `PopupSelected` when an item is hovered at release-time;
        // dispatcher in form_game then calls SuperDjeans + closes.
        if let Some(popup) = self.popup.as_ref() {
            let hit = self
                .screen_to_panel_design("Base", sx, sy, screen_w, screen_h)
                .and_then(|(dx, dy)| popup.hit_test_design(dx, dy));
            self.pressed = None;
            if let Some(idx) = hit {
                if let Some(item) = popup.items.get(idx).copied() {
                    log::info!(
                        "popup: selected item idx={} kind={} parent={:?}",
                        idx,
                        item.kind.0,
                        popup.parent
                    );
                    return Some(Click::PopupItem {
                        parent: popup.parent,
                        kind: item.kind,
                    });
                }
            }
            // Click inside popup but not on any item — leave popup
            // open so the user can pick another item.
            return None;
        }

        let (pi, ei) = self.pressed.take()?;

        // Reset the pressed element's state (matching cursor position).
        let release_hit = self.hit_test(sx, sy, screen_w, screen_h);
        if let Some(p) = self.panels.get_mut(pi) {
            if let Some(e) = p.elements.get_mut(ei) {
                let is_button = matches!(e.kind, super::iface_element::ElementKind::Button);
                if matches!(e.cur_state, ElementState::Disabled)
                    || matches!(e.def_state, ElementState::Disabled)
                {
                    e.cur_state = ElementState::Disabled;
                    return None;
                }
                // Buttons cycle Pressed → Focused/Normal on release
                // (CIFaceButton.cpp:105-109). Statics leave state
                // untouched (CIFaceStatic.cpp:59-61 only fires
                // Action(ON_UN_PRESS)) so don't reassign.
                if is_button {
                    e.cur_state = if release_hit == Some((pi, ei)) {
                        ElementState::Focused
                    } else {
                        e.def_state
                    };
                }
                if release_hit == Some((pi, ei)) {
                    // Emit the click for buttons AND statics so the
                    // dispatcher can route Static ON_PRESS hooks like
                    // `basepl` (IF_BASE_PLANT → CIFaceList::JumpToBuilding,
                    // CInterface.cpp:585) + the resource-plant statics
                    // (titan/electr/energy/plasma).
                    return Some(Click::Button(e.name.clone()));
                }
            }
        }
        None
    }

    /// Right-mouse-button down. Captures the press so RBUp can compare
    /// release-on-same-element; returns true if a UI element caught
    /// the press (caller should suppress camera/move-order routing).
    /// Mirrors the early-return of `CIFaceList::OnMouseRBDown` in
    /// the C++ list dispatcher.
    ///
    /// Only consumes the event for elements that actually have an RBDown
    /// handler — i.e. the constructor pylons that open popup menus
    /// (CIFaceButton.cpp:183-321 returns early when no RB action is
    /// bound). Other buttons (turret picker, hud, etc.) let the press
    /// fall through so the camera can rotate-on-drag from over them.
    pub fn on_mouse_right_down(&mut self, sx: f32, sy: f32, screen_w: f32, screen_h: f32) -> bool {
        let hit = self.hit_test(sx, sy, screen_w, screen_h);
        let Some((pi, ei)) = hit else {
            return false;
        };
        let elem_name = self
            .panels
            .get(pi)
            .and_then(|p| p.elements.get(ei))
            .map(|e| e.name.as_str())
            .unwrap_or("");
        if !is_right_clickable(elem_name) {
            return false;
        }
        self.right_pressed = Some((pi, ei));
        true
    }

    /// Right-mouse-button up. Fires a `Click::RightButton` when
    /// release lands on the same element that was pressed.
    pub fn on_mouse_right_up(
        &mut self,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> Option<Click> {
        let (pi, ei) = self.right_pressed.take()?;
        let release_hit = self.hit_test(sx, sy, screen_w, screen_h);
        if release_hit != Some((pi, ei)) {
            return None;
        }
        let p = self.panels.get(pi)?;
        let e = p.elements.get(ei)?;
        if matches!(e.kind, super::iface_element::ElementKind::Button) {
            Some(Click::RightButton(e.name.clone()))
        } else {
            None
        }
    }

    /// Look up a panel by name (e.g. "Main"). Returns `None` if the
    /// panel wasn't loaded.
    pub fn panel(&self, name: &str) -> Option<&CInterface> {
        self.panels.iter().find(|p| p.name == name)
    }

    pub fn panel_mut(&mut self, name: &str) -> Option<&mut CInterface> {
        self.panels.iter_mut().find(|p| p.name == name)
    }
}

impl Default for IFaceList {
    fn default() -> Self {
        Self::new()
    }
}

/// Element-name allowlist for RMB consumption — kept in sync with the
/// names `popup_for_pylon` (interface/iface_menu.rs) recognises. The
/// constructor pylons are the only buttons that open popup menus on
/// RBDown; everything else lets RMB fall through to camera rotation.
fn is_right_clickable(name: &str) -> bool {
    matches!(
        name,
        "pich" | "pihu" | "pihe" | "pi1" | "pi2" | "pi3" | "pi4" | "pi5"
    )
}
