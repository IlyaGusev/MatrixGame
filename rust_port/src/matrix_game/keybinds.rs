//! User-remappable keyboard bindings.
//!
//! The HTML settings menu (index.html) writes overrides to localStorage
//! as `action=Code;action=Code` (W3C `KeyboardEvent.code` names, which
//! match winit's `KeyCode` Debug names) and fires a
//! `matrixgame-keys-changed` window event; we reload on that signal.
//!
//! Remapping is a translation layer in front of the hard-coded match
//! arms in form_game.rs: an incoming physical key bound to an action is
//! rewritten to that action's canonical key, and a canonical key whose
//! action was rebound elsewhere is swallowed. Camera and game bindings
//! are separate contexts because the defaults share keys (A/S/D).

use std::cell::RefCell;
use std::collections::HashMap;
use winit::keyboard::KeyCode;

#[derive(Clone, Copy, PartialEq)]
pub enum Ctx {
    Camera,
    Game,
}

/// (action id, canonical key, context). Must stay in sync with the
/// `KEY_ACTIONS` table in index.html. Fixed alternates (arrows, Space,
/// Home/End, digits) are not listed and always keep their default
/// meaning.
const ACTIONS: &[(&str, KeyCode, Ctx)] = &[
    ("cam_forward", KeyCode::KeyW, Ctx::Camera),
    ("cam_back", KeyCode::KeyS, Ctx::Camera),
    ("cam_left", KeyCode::KeyA, Ctx::Camera),
    ("cam_right", KeyCode::KeyD, Ctx::Camera),
    ("cam_rot_left", KeyCode::BracketLeft, Ctx::Camera),
    ("cam_rot_right", KeyCode::BracketRight, Ctx::Camera),
    ("cam_pitch_up", KeyCode::PageUp, Ctx::Camera),
    ("cam_pitch_down", KeyCode::PageDown, Ctx::Camera),
    ("cam_reset", KeyCode::Backslash, Ctx::Camera),
    ("stop", KeyCode::KeyS, Ctx::Game),
    ("move", KeyCode::KeyM, Ctx::Game),
    ("attack", KeyCode::KeyA, Ctx::Game),
    ("capture", KeyCode::KeyK, Ctx::Game),
    ("patrol", KeyCode::KeyP, Ctx::Game),
    ("repair", KeyCode::KeyR, Ctx::Game),
    ("explode", KeyCode::KeyE, Ctx::Game),
    ("auto_attack", KeyCode::KeyU, Ctx::Game),
    ("auto_capture", KeyCode::KeyC, Ctx::Game),
    ("auto_defend", KeyCode::KeyD, Ctx::Game),
    ("enter_robot", KeyCode::Enter, Ctx::Game),
    ("build_robot", KeyCode::KeyB, Ctx::Game),
    ("build_turret", KeyCode::KeyT, Ctx::Game),
    ("cancel_order", KeyCode::KeyX, Ctx::Game),
    ("minimap_zoom_in", KeyCode::Equal, Ctx::Game),
    ("minimap_zoom_out", KeyCode::Minus, Ctx::Game),
    ("robot_prev", KeyCode::Comma, Ctx::Game),
    ("robot_next", KeyCode::Period, Ctx::Game),
    ("pause", KeyCode::Pause, Ctx::Game),
];

thread_local! {
    static OVERRIDES: RefCell<HashMap<&'static str, String>> = RefCell::new(HashMap::new());
}

/// Translate a physical key for the given context. `Some(canonical)`
/// when the key is bound to an action (identity for untouched keys),
/// `None` when the key is a canonical default that was rebound away.
pub fn map(ctx: Ctx, code: KeyCode) -> Option<KeyCode> {
    OVERRIDES.with(|o| {
        let o = o.borrow();
        if o.is_empty() {
            return Some(code);
        }
        let code_s = format!("{:?}", code);
        for (id, canon, c) in ACTIONS {
            if *c != ctx {
                continue;
            }
            let canon_s = format!("{:?}", canon);
            if *o.get(id).unwrap_or(&canon_s) == code_s {
                return Some(*canon);
            }
        }
        for (id, canon, c) in ACTIONS {
            if *c == ctx && o.contains_key(id) && format!("{:?}", canon) == code_s {
                return None;
            }
        }
        Some(code)
    })
}

#[cfg(target_arch = "wasm32")]
const STORAGE_KEY: &str = "matrixgame.keys";

#[cfg(target_arch = "wasm32")]
fn reload_overrides() {
    let stored = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(STORAGE_KEY).ok().flatten())
        .unwrap_or_default();
    OVERRIDES.with(|o| {
        let mut o = o.borrow_mut();
        o.clear();
        for pair in stored.split(';') {
            let Some((id, code)) = pair.split_once('=') else {
                continue;
            };
            if let Some((known, ..)) = ACTIONS.iter().find(|(a, ..)| *a == id) {
                o.insert(known, code.to_string());
            }
        }
    });
}

/// Load persisted bindings and re-load whenever the settings menu
/// saves (it dispatches `matrixgame-keys-changed` on `window`).
#[cfg(target_arch = "wasm32")]
pub fn init() {
    use wasm_bindgen::prelude::*;
    reload_overrides();
    let Some(win) = web_sys::window() else {
        return;
    };
    let cb = Closure::wrap(Box::new(reload_overrides) as Box<dyn FnMut()>);
    let _ = win.add_event_listener_with_callback("matrixgame-keys-changed", cb.as_ref().unchecked_ref());
    cb.forget();
}

#[cfg(not(target_arch = "wasm32"))]
pub fn init() {}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(pairs: &[(&'static str, &str)]) {
        OVERRIDES.with(|o| {
            let mut o = o.borrow_mut();
            o.clear();
            for (id, code) in pairs {
                o.insert(id, code.to_string());
            }
        });
    }

    #[test]
    fn identity_without_overrides() {
        set(&[]);
        assert_eq!(map(Ctx::Game, KeyCode::KeyS), Some(KeyCode::KeyS));
        assert_eq!(map(Ctx::Camera, KeyCode::KeyQ), Some(KeyCode::KeyQ));
    }

    #[test]
    fn rebind_translates_and_shadows_canonical() {
        set(&[("stop", "KeyQ")]);
        assert_eq!(map(Ctx::Game, KeyCode::KeyQ), Some(KeyCode::KeyS));
        assert_eq!(map(Ctx::Game, KeyCode::KeyS), None);
        // Camera context untouched — S still moves the camera back.
        assert_eq!(map(Ctx::Camera, KeyCode::KeyS), Some(KeyCode::KeyS));
        set(&[]);
    }

    #[test]
    fn swap_within_context() {
        set(&[("cam_forward", "KeyS"), ("cam_back", "KeyW")]);
        assert_eq!(map(Ctx::Camera, KeyCode::KeyS), Some(KeyCode::KeyW));
        assert_eq!(map(Ctx::Camera, KeyCode::KeyW), Some(KeyCode::KeyS));
        set(&[]);
    }
}
