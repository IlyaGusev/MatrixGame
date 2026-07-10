//! WebAudio implementation of [`SoundOutput`] (SOUND_PROPOSAL.md §3).
//!
//! Graph: `AudioBufferSourceNode → GainNode → StereoPannerNode → sfx
//! gain → master gain → destination`, one chain per voice. Samples
//! come from `assets/sounds.bundle` (see `examples/pack_sounds.rs`),
//! keyed by the SR2 resource path from the `Sounds` block; the bundle
//! is fetched once at startup and decoded lazily per path. Absent
//! bundle ⇒ `ready() == false` ⇒ the mixer drops every play — exactly
//! the standalone C++ build's silence.
//!
//! Autoplay policy: the AudioContext starts suspended until the first
//! pointer / key gesture resumes it; until then plays are dropped.
//!
//! Music (§6): `assets/music/playlist.txt` lists files (one per line,
//! relative to `assets/music/`); they loop shuffled through a
//! dedicated gain the `SetMusicVolume` hook fades over ~1 s.

use crate::gfx::bundle::AssetBundle;
use crate::matrix_game::sound::SoundOutput;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    AudioBuffer, AudioBufferSourceNode, AudioContext, AudioContextState, GainNode,
    StereoPannerNode,
};

// Bump ?sv= when the packed sample set changes so browsers refetch.
const SOUNDS_BUNDLE_URL: &str = "assets/sounds.bundle?sv=2";
const MUSIC_DIR: &str = "assets/music/";
const MUSIC_PLAYLIST_URL: &str = "assets/music/playlist.txt";
/// Soundtrack sits under the SFX by default.
const MUSIC_BASE_VOL: f32 = 0.5;
const STORAGE_KEY: &str = "matrixgame.sound";

enum BundleState {
    Pending,
    Absent,
    Ready(AssetBundle),
}

enum BufState {
    Loading,
    Missing,
    Ready(AudioBuffer),
}

struct Shared {
    enabled: Cell<bool>,
    bundle: RefCell<BundleState>,
    buffers: RefCell<HashMap<String, BufState>>,
    /// Next shuffled track (decoded, waiting for the pump to start it).
    music_buf: RefCell<Option<AudioBuffer>>,
    music_tracks: RefCell<Vec<String>>,
    music_loading: Cell<bool>,
    music_idx: Cell<usize>,
}

struct Voice {
    gain: GainNode,
    pan: StereoPannerNode,
    src: Option<AudioBufferSourceNode>,
    path: String,
    looped: bool,
    want_play: bool,
    started: bool,
    ended: Rc<Cell<bool>>,
    _onended: Option<Closure<dyn FnMut()>>,
}

pub struct WebOutput {
    ctx: Option<AudioContext>,
    sfx: Option<GainNode>,
    master: Option<GainNode>,
    music_gain: Option<GainNode>,
    shared: Rc<Shared>,
    voices: HashMap<u32, Voice>,
    next_voice: u32,
    music_src: Option<(AudioBufferSourceNode, Rc<Cell<bool>>, Option<Closure<dyn FnMut()>>)>,
    /// `m_TargetMusicVolume` — reapplied to fresh tracks.
    music_target: f32,
}

impl WebOutput {
    pub fn new() -> Self {
        let shared = Rc::new(Shared {
            enabled: Cell::new(load_enabled_pref()),
            bundle: RefCell::new(BundleState::Pending),
            buffers: RefCell::new(HashMap::new()),
            music_buf: RefCell::new(None),
            music_tracks: RefCell::new(Vec::new()),
            music_loading: Cell::new(false),
            music_idx: Cell::new(0),
        });

        let ctx = AudioContext::new().ok();
        let (master, sfx, music_gain) = if let Some(ctx) = &ctx {
            let master = ctx.create_gain().ok();
            let sfx = ctx.create_gain().ok();
            let music = ctx.create_gain().ok();
            if let (Some(m), Some(s), Some(mu)) = (&master, &sfx, &music) {
                let _ = m.connect_with_audio_node(&ctx.destination());
                let _ = s.connect_with_audio_node(m);
                let _ = mu.connect_with_audio_node(m);
                m.gain()
                    .set_value(if shared.enabled.get() { 1.0 } else { 0.0 });
                mu.gain().set_value(MUSIC_BASE_VOL);
            }
            (master, sfx, music)
        } else {
            log::warn!("audio: AudioContext creation failed — staying silent");
            (None, None, None)
        };

        if let Some(ctx) = &ctx {
            register_unlock_gestures(ctx.clone());
        }
        register_sound_button(shared.clone(), master.clone(), ctx.clone());

        // Probe the sample bundle; absence is the expected no-assets
        // case, not an error.
        {
            let shared = shared.clone();
            wasm_bindgen_futures::spawn_local(async move {
                match crate::gfx::loader::load_bytes(SOUNDS_BUNDLE_URL).await {
                    Ok(bytes) => match AssetBundle::from_bytes(&bytes) {
                        Ok(b) => {
                            log::info!("audio: sounds.bundle loaded ({} samples)", b.list_files().len());
                            *shared.bundle.borrow_mut() = BundleState::Ready(b);
                        }
                        Err(e) => {
                            log::warn!("audio: sounds.bundle unreadable: {e}");
                            *shared.bundle.borrow_mut() = BundleState::Absent;
                        }
                    },
                    Err(_) => {
                        log::info!("audio: no sounds.bundle — SFX silent (pack one with examples/pack_sounds.rs)");
                        *shared.bundle.borrow_mut() = BundleState::Absent;
                    }
                }
            });
        }
        // Music playlist probe.
        {
            let shared = shared.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Ok(bytes) = crate::gfx::loader::load_bytes(MUSIC_PLAYLIST_URL).await {
                    let mut tracks: Vec<String> = String::from_utf8_lossy(&bytes)
                        .lines()
                        .map(|l| l.trim().to_string())
                        .filter(|l| !l.is_empty() && !l.starts_with('#'))
                        .collect();
                    // Shuffle once (Fisher-Yates on Math.random).
                    for i in (1..tracks.len()).rev() {
                        let j = (js_sys::Math::random() * (i + 1) as f64) as usize;
                        tracks.swap(i, j.min(i));
                    }
                    if !tracks.is_empty() {
                        log::info!("audio: music playlist: {} tracks", tracks.len());
                    }
                    *shared.music_tracks.borrow_mut() = tracks;
                }
            });
        }

        Self {
            ctx,
            sfx,
            master,
            music_gain,
            shared,
            voices: HashMap::new(),
            next_voice: 1,
            music_src: None,
            music_target: 1.0,
        }
    }

    fn ctx_running(&self) -> bool {
        self.ctx
            .as_ref()
            .is_some_and(|c| c.state() == AudioContextState::Running)
    }

    /// Kick off (or look up) the decode of `path`'s sample.
    fn ensure_buffer(&self, path: &str) {
        let mut buffers = self.shared.buffers.borrow_mut();
        if buffers.contains_key(path) {
            return;
        }
        let bundle = self.shared.bundle.borrow();
        let BundleState::Ready(b) = &*bundle else {
            return;
        };
        let Some(bytes) = b.read_file(path) else {
            log::debug!("audio: no sample for {path}");
            buffers.insert(path.to_string(), BufState::Missing);
            return;
        };
        let Some(ctx) = &self.ctx else { return };
        buffers.insert(path.to_string(), BufState::Loading);
        let array = js_sys::Uint8Array::from(bytes);
        let Ok(promise) = ctx.decode_audio_data(&array.buffer()) else {
            buffers.insert(path.to_string(), BufState::Missing);
            return;
        };
        let shared = self.shared.clone();
        let path = path.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            let state = match wasm_bindgen_futures::JsFuture::from(promise).await {
                Ok(v) => match v.dyn_into::<AudioBuffer>() {
                    Ok(buf) => BufState::Ready(buf),
                    Err(_) => BufState::Missing,
                },
                Err(e) => {
                    log::warn!("audio: decode failed for {path}: {e:?}");
                    BufState::Missing
                }
            };
            shared.buffers.borrow_mut().insert(path, state);
        });
    }

    /// Attach the decoded buffer and start a voice that asked to play.
    fn try_start(&mut self, v: u32) {
        let Some(ctx) = self.ctx.clone() else { return };
        let Some(voice) = self.voices.get_mut(&v) else {
            return;
        };
        if voice.started || !voice.want_play {
            return;
        }
        let buffers = self.shared.buffers.borrow();
        match buffers.get(&voice.path) {
            Some(BufState::Ready(buf)) => {
                let Ok(src) = ctx.create_buffer_source() else {
                    voice.ended.set(true);
                    return;
                };
                src.set_buffer(Some(buf));
                src.set_loop(voice.looped);
                let _ = src.connect_with_audio_node(&voice.gain);
                if !voice.looped {
                    let ended = voice.ended.clone();
                    let cb = Closure::wrap(Box::new(move || ended.set(true)) as Box<dyn FnMut()>);
                    src.set_onended(Some(cb.as_ref().unchecked_ref()));
                    voice._onended = Some(cb);
                }
                let _ = src.start();
                voice.src = Some(src);
                voice.started = true;
            }
            Some(BufState::Missing) => voice.ended.set(true),
            Some(BufState::Loading) | None => {} // retried from takt()
        }
    }

    /// Advance the playlist: start a decoded track, or begin decoding
    /// the next one.
    fn pump_music(&mut self) {
        if !self.shared.enabled.get() || !self.ctx_running() {
            return;
        }
        if let Some((_, ended, _)) = &self.music_src {
            if !ended.get() {
                return;
            }
            self.music_src = None;
        }
        // A decoded track is waiting — start it.
        let buf = self.shared.music_buf.borrow_mut().take();
        if let Some(buf) = buf {
            let (Some(ctx), Some(gain)) = (&self.ctx, &self.music_gain) else {
                return;
            };
            let Ok(src) = ctx.create_buffer_source() else {
                return;
            };
            src.set_buffer(Some(&buf));
            let _ = src.connect_with_audio_node(gain);
            let ended = Rc::new(Cell::new(false));
            let e2 = ended.clone();
            let cb = Closure::wrap(Box::new(move || e2.set(true)) as Box<dyn FnMut()>);
            src.set_onended(Some(cb.as_ref().unchecked_ref()));
            let _ = src.start();
            self.music_src = Some((src, ended, Some(cb)));
            return;
        }
        // Otherwise queue the next decode.
        if self.music_loading() || self.shared.music_tracks.borrow().is_empty() {
            return;
        }
        let (track, idx) = {
            let tracks = self.shared.music_tracks.borrow();
            let idx = self.shared.music_idx.get() % tracks.len();
            (tracks[idx].clone(), idx)
        };
        self.shared.music_idx.set(idx + 1);
        self.shared.music_loading.set(true);
        let Some(ctx) = self.ctx.clone() else { return };
        let shared = self.shared.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let url = format!("{MUSIC_DIR}{track}");
            if let Ok(bytes) = crate::gfx::loader::load_bytes(&url).await {
                let array = js_sys::Uint8Array::from(bytes.as_slice());
                if let Ok(p) = ctx.decode_audio_data(&array.buffer()) {
                    if let Ok(v) = wasm_bindgen_futures::JsFuture::from(p).await {
                        if let Ok(buf) = v.dyn_into::<AudioBuffer>() {
                            *shared.music_buf.borrow_mut() = Some(buf);
                        }
                    }
                }
            } else {
                log::warn!("audio: music track fetch failed: {url}");
            }
            shared.music_loading.set(false);
        });
    }

    fn music_loading(&self) -> bool {
        self.shared.music_loading.get()
    }
}

impl SoundOutput for WebOutput {
    fn ready(&self) -> bool {
        self.shared.enabled.get()
            && self.ctx_running()
            && matches!(&*self.shared.bundle.borrow(), BundleState::Ready(_))
    }

    fn create(&mut self, path: &str, looped: bool) -> u32 {
        let (Some(ctx), Some(sfx)) = (&self.ctx, &self.sfx) else {
            return 0;
        };
        let (Ok(gain), Ok(pan)) = (ctx.create_gain(), ctx.create_stereo_panner()) else {
            return 0;
        };
        if gain.connect_with_audio_node(&pan).is_err() || pan.connect_with_audio_node(sfx).is_err()
        {
            return 0;
        }
        self.ensure_buffer(path);
        let v = self.next_voice;
        self.next_voice += 1;
        self.voices.insert(
            v,
            Voice {
                gain,
                pan,
                src: None,
                path: path.to_string(),
                looped,
                want_play: false,
                started: false,
                ended: Rc::new(Cell::new(false)),
                _onended: None,
            },
        );
        v
    }

    fn play(&mut self, voice: u32) {
        if let Some(v) = self.voices.get_mut(&voice) {
            v.want_play = true;
        }
        self.try_start(voice);
    }

    fn set_pan(&mut self, voice: u32, pan: f32) {
        if let Some(v) = self.voices.get(&voice) {
            v.pan.pan().set_value(pan.clamp(-1.0, 1.0));
        }
    }

    fn set_vol(&mut self, voice: u32, vol: f32) {
        if let Some(v) = self.voices.get(&voice) {
            v.gain.gain().set_value(vol.clamp(0.0, 1.0));
        }
    }

    fn is_playing(&self, voice: u32) -> bool {
        self.voices
            .get(&voice)
            .is_some_and(|v| !v.ended.get() && (v.want_play || v.started))
    }

    fn destroy(&mut self, voice: u32) {
        if let Some(v) = self.voices.remove(&voice) {
            if let Some(src) = &v.src {
                src.set_onended(None);
                let _ = src.stop();
                src.disconnect().ok();
            }
            v.gain.disconnect().ok();
            v.pan.disconnect().ok();
        }
    }

    fn set_music_volume(&mut self, vol: f32) {
        self.music_target = vol;
        let (Some(ctx), Some(gain)) = (&self.ctx, &self.music_gain) else {
            return;
        };
        // ~1 s linear fade — the C++ interpolates m_TargetMusicVolume.
        let g = gain.gain();
        let now = ctx.current_time();
        let _ = g.cancel_scheduled_values(now);
        let _ = g.set_value_at_time(g.value(), now);
        let _ = g.linear_ramp_to_value_at_time(vol * MUSIC_BASE_VOL, now + 1.0);
    }

    fn takt(&mut self) {
        // Voices whose decode finished after play() — start them now.
        let pending: Vec<u32> = self
            .voices
            .iter()
            .filter(|(_, v)| v.want_play && !v.started && !v.ended.get())
            .map(|(&k, _)| k)
            .collect();
        for v in pending {
            self.try_start(v);
        }
        self.pump_music();
    }
}

fn load_enabled_pref() -> bool {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(STORAGE_KEY).ok().flatten())
        .map(|v| v != "off")
        .unwrap_or(true)
}

fn save_enabled_pref(on: bool) {
    if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = s.set_item(STORAGE_KEY, if on { "on" } else { "off" });
    }
}

/// Resume the suspended AudioContext on the first user gesture
/// (browser autoplay policy). Listeners stay registered — repeat
/// resumes are no-ops.
fn register_unlock_gestures(ctx: AudioContext) {
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    for ev in ["pointerdown", "keydown"] {
        let ctx = ctx.clone();
        let cb = Closure::wrap(Box::new(move || {
            if ctx.state() == AudioContextState::Suspended {
                let _ = ctx.resume();
            }
        }) as Box<dyn FnMut()>);
        let _ = doc.add_event_listener_with_callback(ev, cb.as_ref().unchecked_ref());
        cb.forget();
    }
}

/// Wire the on-screen 🔊/🔇 toggle (`#sound-btn` in index.html):
/// flips the master gain, persists the choice, and doubles as an
/// unlock gesture.
fn register_sound_button(shared: Rc<Shared>, master: Option<GainNode>, ctx: Option<AudioContext>) {
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(btn) = doc.get_element_by_id("sound-btn") else {
        return;
    };
    let render = |el: &web_sys::Element, on: bool| {
        el.set_text_content(Some(if on { "🔊" } else { "🔇" }));
        let _ = el.set_attribute("title", if on { "Sound: on" } else { "Sound: off" });
    };
    render(&btn, shared.enabled.get());
    let btn2 = btn.clone();
    let cb = Closure::wrap(Box::new(move || {
        let on = !shared.enabled.get();
        shared.enabled.set(on);
        save_enabled_pref(on);
        render(&btn2, on);
        if let Some(m) = &master {
            m.gain().set_value(if on { 1.0 } else { 0.0 });
        }
        if on {
            if let Some(c) = &ctx {
                if c.state() == AudioContextState::Suspended {
                    let _ = c.resume();
                }
            }
        }
    }) as Box<dyn FnMut()>);
    let _ = btn.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref());
    cb.forget();
}
