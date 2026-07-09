//! Port of the constructor / interface UI sound dispatch
//! (CSound::Play calls in Interface/CIFaceButton.cpp:33-167) plus the
//! by-name `CSound::Play(name, sl)` sites that have no `Objects`
//! access (selection voices, minimap zoom, hint SOUNDIN/SOUNDOUT).
//!
//! Events land in a thread-local queue the app loop drains into the
//! [`SoundMixer`](crate::matrix_game::sound::SoundMixer) once per
//! frame — same pattern as `Objects::pending_sounds`, just reachable
//! from UI code that only has `&self`.

use crate::matrix_game::sound::SoundLayer;
use std::cell::RefCell;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiSound {
    /// Port of `S_BCLICK` (MatrixSoundManager.hpp:44).
    BClick,
    /// Port of `S_BENTER` (MatrixSoundManager.hpp:45).
    BEnter,
    /// Port of `S_BUILD_CLICK` (MatrixSoundManager.hpp:50).
    BuildClick,
    /// Port of `S_CANCEL_CLICK` (MatrixSoundManager.hpp:51).
    CancelClick,
    /// Port of `S_PRESET_CLICK` (MatrixSoundManager.hpp:49).
    PresetClick,
}

impl UiSound {
    /// Sounds-block key (`CSound::Init`, MatrixSoundManager.cpp:84-91).
    fn key(self) -> &'static str {
        match self {
            UiSound::BClick => "bclick",
            UiSound::BEnter => "benter",
            UiSound::BuildClick => "build_click",
            UiSound::CancelClick => "cancel_click",
            UiSound::PresetClick => "preset_click",
        }
    }
}

thread_local! {
    static UI_SOUNDS: RefCell<Vec<(String, SoundLayer)>> = const { RefCell::new(Vec::new()) };
}

fn queue(key: &str, layer: SoundLayer) {
    UI_SOUNDS.with(|q| q.borrow_mut().push((key.to_string(), layer)));
}

/// Drain the queued UI sounds — called once per frame by the app
/// loop's `pump_sounds`.
pub fn drain() -> Vec<(String, SoundLayer)> {
    UI_SOUNDS.with(|q| std::mem::take(&mut *q.borrow_mut()))
}

/// Port of `CSound::Play(sound_id, SL_INTERFACE)`.
pub fn play(sound: UiSound) {
    queue(sound.key(), SoundLayer::Interface);
}

/// Fire a hint's `_SOUNDIN:`/`_SOUNDOUT:` sound by name (MatrixHint.hpp:
/// 134-158) — resolved as a Sounds-block key like every by-name play.
pub fn play_hint_sound(name: &str) {
    if !name.is_empty() {
        queue(name, SoundLayer::Interface);
    }
}

/// Fire an interface-layer sound by its Sounds-block key — the
/// `CSound::Play(S_*, SL_ALL)` sites outside the button dispatch
/// (minimap zoom, elevator field).
pub fn play_named(name: &str) {
    queue(name, SoundLayer::All);
}

/// `CSound::Play(S_*, SL_*)` by name with an explicit layer
/// (selection voices on SL_SELECTION, MatrixSide.cpp:980-1015).
pub fn play_named_layer(name: &str, layer: SoundLayer) {
    queue(name, layer);
}

/// Pick the right sound for an `LB-down on PUSH_BUTTON` event, matching
/// the C++ name-based dispatch at CIFaceButton.cpp:44-50.
pub fn for_push_button_down(name: &str) -> UiSound {
    match name {
        "cobuild" => UiSound::BuildClick,
        "cocan" => UiSound::CancelClick,
        _ => UiSound::BClick,
    }
}

/// Pick the right sound for `LB-down on CHECK_PUSH_BUTTON` (preset
/// toggle), CIFaceButton.cpp:79-83. The `conf*` family are the preset
/// slot toggles.
pub fn for_check_push_button_down(name: &str) -> UiSound {
    if name.contains("conf") {
        UiSound::PresetClick
    } else {
        UiSound::BClick
    }
}
