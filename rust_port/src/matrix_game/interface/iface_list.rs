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

use super::iface_element::ElementState;
use super::interface::{CInterface, DESIGN_H};

pub struct IFaceList {
    /// In front-to-back order — the FIRST panel receives events first
    /// (matches `LIST_ADD` prepend behaviour in the original).
    pub panels: Vec<CInterface>,
    /// `(panel_idx, element_idx)` for the currently-focused element,
    /// mirrors `CIFaceList::m_FocusedInterface` + `m_FocusedElement`.
    pub focused: Option<(usize, usize)>,
    /// `(panel_idx, element_idx)` for the currently-pressed-down
    /// element. Cleared on mouse up.
    pub pressed: Option<(usize, usize)>,
}

/// Outcome of a button click. The C++ dispatches via action arrays
/// (`SAction m_Actions[MAX_ACTIONS]`); we return a simple enum the
/// caller can match on. Populated from the element's `Name` (the
/// same string `if/Main` uses as a button-id, e.g. "buro" / "buca").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Click {
    /// Button element was pressed-and-released; `name` is the
    /// identifier from `if/Main` (e.g. "buro" for build-robot).
    Button(String),
}

impl IFaceList {
    pub fn new() -> Self {
        Self {
            panels: Vec::new(),
            focused: None,
            pressed: None,
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
        for name in ["Top", "MiniM", "Radar", "Base", "Main", "Hints"] {
            if let Some(mut p) = CInterface::load(matrix_data, name) {
                p.visible = name == "Main";
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
    /// element under the cursor, clearing the previous focus.
    pub fn on_mouse_move(&mut self, sx: f32, sy: f32, screen_w: f32, screen_h: f32) {
        let new_focus = self.hit_test(sx, sy, screen_w, screen_h);

        // Clear the old focus — but only when the pressed button is
        // a different element. If a press is held and we drag onto
        // a different element, the C++ keeps the pressed element
        // 'Pressed' and no others are focused (CIFaceList::OnMouseMove
        // :3100-3150 roughly). We approximate: pressed element stays
        // pressed; other elements go Normal unless under the cursor.
        if self.focused != new_focus {
            if let Some((pi, ei)) = self.focused {
                let pressed_self = self.pressed == Some((pi, ei));
                if let Some(p) = self.panels.get_mut(pi) {
                    if let Some(e) = p.elements.get_mut(ei) {
                        if !pressed_self {
                            e.cur_state = e.def_state;
                        }
                    }
                }
            }
            if let Some((pi, ei)) = new_focus {
                if let Some(p) = self.panels.get_mut(pi) {
                    if let Some(e) = p.elements.get_mut(ei) {
                        // Hovered: go Focused (unless this element is
                        // the currently-pressed one — in which case
                        // stay Pressed).
                        if self.pressed != Some((pi, ei)) {
                            e.cur_state = ElementState::Focused;
                        } else {
                            e.cur_state = ElementState::Pressed;
                        }
                    }
                }
            }
            self.focused = new_focus;
        }
    }

    /// Mouse-button-down event. If an element is under the cursor,
    /// mark it Pressed and record the pressed-handle. Returns true
    /// if the event was consumed by the UI (caller can skip
    /// routing to the world picker).
    pub fn on_mouse_down(&mut self, sx: f32, sy: f32, screen_w: f32, screen_h: f32) -> bool {
        let hit = self.hit_test(sx, sy, screen_w, screen_h);
        match hit {
            Some((pi, ei)) => {
                if let Some(p) = self.panels.get_mut(pi) {
                    if let Some(e) = p.elements.get_mut(ei) {
                        e.cur_state = ElementState::Pressed;
                    }
                }
                self.pressed = Some((pi, ei));
                true
            }
            None => false,
        }
    }

    /// Mouse-button-up event. If release happens on the same element
    /// that was pressed, fire a `Click`. Ports
    /// `CIFaceButton::OnMouseLBUp` (Interface/CIFaceButton.cpp).
    pub fn on_mouse_up(&mut self, sx: f32, sy: f32, screen_w: f32, screen_h: f32) -> Option<Click> {
        let (pi, ei) = self.pressed.take()?;

        // Reset the pressed element's state (matching cursor position).
        let release_hit = self.hit_test(sx, sy, screen_w, screen_h);
        if let Some(p) = self.panels.get_mut(pi) {
            if let Some(e) = p.elements.get_mut(ei) {
                e.cur_state = if release_hit == Some((pi, ei)) {
                    ElementState::Focused
                } else {
                    e.def_state
                };
                if release_hit == Some((pi, ei))
                    && matches!(e.kind, super::iface_element::ElementKind::Button)
                {
                    return Some(Click::Button(e.name.clone()));
                }
            }
        }
        None
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
