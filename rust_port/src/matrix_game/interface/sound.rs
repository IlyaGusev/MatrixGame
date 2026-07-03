//! Port of the constructor / interface UI sound dispatch
//! (CSound::Play calls in Interface/CIFaceButton.cpp:33-167).
//!
//! The C++ sound IDs are listed in MatrixSoundManager.hpp:44-52. The
//! relevant subset for the Base / constructor panel is:
//!
//! | C++ ID            | Trigger                                       |
//! |-------------------|-----------------------------------------------|
//! | `S_BCLICK`        | LB-down on a generic UI button                |
//! | `S_BENTER`        | Cursor enters a button (focus transition)     |
//! | `S_BUILD_CLICK`   | LB-down on `cobuild`                          |
//! | `S_CANCEL_CLICK`  | LB-down on `cocan`                            |
//! | `S_PRESET_CLICK`  | LB-down on a `conf*` preset toggle            |
//!
//! The Rust port doesn't yet have an audio backend (CROSSREF.md:50 —
//! "sound deferred"). This module owns the dispatch surface so every
//! call site that the C++ has is also wired here. `play()` is a stub
//! that logs the event; replacing it with a real backend (web-audio
//! on WASM, rodio on native) is a single-file change.

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

/// Port of `CSound::Play(sound_id, SL_INTERFACE)`. Currently a stub —
/// logs the event so we can verify dispatch wiring; once the audio
/// backend lands this becomes a real playback call.
pub fn play(sound: UiSound) {
    log::trace!("ui sound: {:?}", sound);
}

/// Fire a hint's `_SOUNDIN:`/`_SOUNDOUT:` sound by name (MatrixHint.hpp:
/// 134-158). Silent like the rest of the sound layer until a host backend
/// is attached, but the dispatch site is present.
pub fn play_hint_sound(name: &str) {
    if !name.is_empty() {
        log::trace!("hint sound: {name}");
    }
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
