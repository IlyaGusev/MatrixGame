//! Port of `CSound` / `CSoundArray` (MatrixSoundManager.{cpp,hpp}) —
//! the mixing policy: pan/volume from camera, slot eviction, layers,
//! positional dedup with ttl/fade, and looped-sound handles.
//!
//! The original renders nothing itself: every voice goes through the
//! host's function table (`g_RangersInterface->m_Sound*`). Here that
//! table is the [`SoundOutput`] trait — WebAudio on wasm
//! (`platform::audio_web`), a no-op on native. Game code never calls
//! the mixer directly; it queues [`SndEvent`]s on
//! `Objects::pending_sounds` (or the interface queue) and the app loop
//! drains them into [`SoundMixer::dispatch`] each frame.

use crate::matrix_game::config::SoundDefs;
use glam::Vec3;
use std::collections::HashMap;

/// `MAX_SOUNDS` (MatrixSoundManager.hpp:14) — 16 mixed voices.
pub const MAX_SOUNDS: usize = 16;
/// `SOUND_FULL_VOLUME_DIST` (hpp:10).
pub const SOUND_FULL_VOLUME_DIST: f32 = 200.0;
/// `SOUND_POS_DIVIDER` = 2·GLOBAL_SCALE (hpp:12).
pub const SOUND_POS_DIVIDER: f32 = 40.0;

const SOUND_ID_EMPTY: u32 = u32::MAX;

/// `ESoundLayer` (hpp:16-31). One voice per non-`All` layer.
///
/// NOTE: the shipped C++ never assigns `m_LayersI[sl].index`, so its
/// LayerOff / SEF_SKIP checks are inert — the layer system is dead
/// code in the original. We implement the *intended* semantics (a new
/// play on a layer stops the previous one) per the field comments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundLayer {
    All,
    Interface,
    ElevatorField,
    Selection,
    Hull,
    Chassis,
    Order,
}

const SL_COUNT: usize = 7;

impl SoundLayer {
    fn index(self) -> usize {
        match self {
            SoundLayer::All => 0,
            SoundLayer::Interface => 1,
            SoundLayer::ElevatorField => 2,
            SoundLayer::Selection => 3,
            SoundLayer::Hull => 4,
            SoundLayer::Chassis => 5,
            SoundLayer::Order => 6,
        }
    }
}

/// `ESoundInterruptFlag` (hpp:34-38).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interrupt {
    Interrupt,
    Skip,
}

/// Stands in for the `DWORD m_Sound` ids C++ callers keep across takts
/// (weapon hum, chassis loop, flyer vint). Callers can't hold mixer ids
/// through the event queue, so they key by owner instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SndHandle {
    Weapon(crate::matrix_game::effects::weapon::WeaponId),
    Chassis(crate::matrix_game::map_static::ObjectId),
    Flyer(crate::matrix_game::map_static::ObjectId),
    Elevator(u32),
}

/// One queued `CSound` call. Variants map 1:1 to the C++ entry points.
#[derive(Debug, Clone)]
pub enum SndEvent {
    /// `CSound::Play(snd, sl)` — non-positional one-shot;
    /// vol/pan randomized in `[vol0,vol1]`/`[pan0,pan1]`.
    Play { key: String, layer: SoundLayer },
    /// `CSound::Play(snd, pos, sl)` — positional immediate (no dedup).
    /// `handle: Some` mirrors the `m_Sound = Play(...)` id-capture so a
    /// later ChangePos / Stop can find the voice.
    PlayAt {
        key: String,
        pos: [f32; 3],
        layer: SoundLayer,
        handle: Option<SndHandle>,
    },
    /// `CSound::Play(id, snd, pos, sl)` — ambient retrigger: update
    /// pan/vol if the handle's voice still plays, else start anew.
    PlayHandle {
        handle: SndHandle,
        key: String,
        pos: [f32; 3],
        layer: SoundLayer,
    },
    /// `CSound::AddSound(snd, pos, sl, ifl)` — positional dedup: the
    /// same key already sounding in the same 2·GLOBAL_SCALE cell is
    /// retriggered / ttl-refreshed instead of layered.
    Add {
        key: String,
        pos: [f32; 3],
        layer: SoundLayer,
        ifl: Interrupt,
    },
    /// `CSound::AddSound(pos, attn, pan0, pan1, vol0, vol1, name)` —
    /// the by-name variant map ambient spawners use
    /// (EffectSpawnerSound, MatrixEffect.cpp:206-210): `path` is the
    /// SR2 resource path directly (bypasses the Sounds block), the
    /// mixing params come from the map data, `attn` pre-scaled ×0.002.
    /// Entries carry `snd == S_UNDEF` in the C++: never deduped,
    /// no ttl expiry, never looped.
    AddNamed {
        path: String,
        pos: [f32; 3],
        attn: f32,
        pan0: f32,
        pan1: f32,
        vol0: f32,
        vol1: f32,
    },
    /// `CSound::ChangePos(id, snd, pos)`.
    ChangePos {
        handle: SndHandle,
        key: String,
        pos: [f32; 3],
    },
    /// `CSound::StopPlay(id)`.
    Stop { handle: SndHandle },
    /// `CMatrixMap::SetMusicVolume` (MatrixMap.cpp:3583) — ducks the
    /// host soundtrack (Terron death). 1.0 restores.
    MusicVolume(f32),
}

/// The host voice table — `g_RangersInterface->m_Sound*` equivalents.
/// `voice == 0` is the invalid handle (`snd_create` failure).
pub trait SoundOutput {
    /// Assets present and autoplay unlocked. While false the mixer
    /// drops play requests (the original is equally silent when
    /// `g_RangersInterface == NULL`).
    fn ready(&self) -> bool;
    fn create(&mut self, path: &str, looped: bool) -> u32;
    fn play(&mut self, voice: u32);
    fn set_pan(&mut self, voice: u32, pan: f32);
    fn set_vol(&mut self, voice: u32, vol: f32);
    fn is_playing(&self, voice: u32) -> bool;
    fn destroy(&mut self, voice: u32);
    fn set_music_volume(&mut self, vol: f32);
    /// Per-frame pump (decode queue, music advance). Default no-op.
    fn takt(&mut self) {}
}

/// Silent output for native builds and tests.
pub struct NullOutput;

impl SoundOutput for NullOutput {
    fn ready(&self) -> bool {
        false
    }
    fn create(&mut self, _path: &str, _looped: bool) -> u32 {
        0
    }
    fn play(&mut self, _voice: u32) {}
    fn set_pan(&mut self, _voice: u32, _pan: f32) {}
    fn set_vol(&mut self, _voice: u32, _vol: f32) {}
    fn is_playing(&self, _voice: u32) -> bool {
        false
    }
    fn destroy(&mut self, _voice: u32) {}
    fn set_music_volume(&mut self, _vol: f32) {}
}

/// `SPlayedSound` (hpp:271-277).
#[derive(Clone, Copy)]
struct PlayedSound {
    id_internal: u32,
    id: u32,
    curvol: f32,
    curpan: f32,
}

/// `SLID` (hpp:263-269).
#[derive(Clone, Copy)]
struct Slid {
    index: i32,
    id: u32,
}

/// `CSoundArray::SSndData` (hpp:342-350) — params copied at add time
/// like the C++. `undef` marks by-name entries (`snd == S_UNDEF`):
/// they skip the dedup scan and the ttl bookkeeping.
struct SndData {
    key: String,
    undef: bool,
    id: u32,
    attn: f32,
    pan0: f32,
    pan1: f32,
    vol0: f32,
    vol1: f32,
    ttl: f32,
    fade: f32,
}

pub struct SoundMixer {
    defs: SoundDefs,
    out: Box<dyn SoundOutput>,
    slots: [PlayedSound; MAX_SOUNDS],
    layers: [Slid; SL_COUNT],
    /// `m_PosSounds` — Pos2Key cell → sounds living in that cell.
    pos_sounds: HashMap<u32, Vec<SndData>>,
    /// Owner handle → mixer sound id (the `m_Sound` members).
    handles: HashMap<SndHandle, u32>,
    last_id: u32,
    /// Camera focus (`GetFrustumCenter`) and right vector, refreshed
    /// by the app loop before draining events.
    focus: Vec3,
    right: Vec3,
    /// 100ms positional-update gate (`nextsoundtakt_1`).
    next_pos_takt: i64,
    /// 1s ended-voice cleanup gate (`nextsoundtakt`).
    next_clean_takt: i64,
    /// Non-sim RNG for RND(vol0,vol1) — the C++ uses the global rand,
    /// not the deterministic game rng.
    seed: u64,
}

impl SoundMixer {
    pub fn new(defs: SoundDefs, out: Box<dyn SoundOutput>) -> Self {
        Self {
            defs,
            out,
            slots: [PlayedSound {
                id_internal: 0,
                id: SOUND_ID_EMPTY,
                curvol: 0.0,
                curpan: 0.0,
            }; MAX_SOUNDS],
            layers: [Slid {
                index: -1,
                id: SOUND_ID_EMPTY,
            }; SL_COUNT],
            pos_sounds: HashMap::new(),
            handles: HashMap::new(),
            last_id: 0,
            focus: Vec3::ZERO,
            right: Vec3::X,
            next_pos_takt: 0,
            next_clean_takt: 0,
            seed: 0x9E3779B97F4A7C15,
        }
    }

    pub fn output_mut(&mut self) -> &mut dyn SoundOutput {
        &mut *self.out
    }

    pub fn set_listener(&mut self, focus: Vec3, right: Vec3) {
        self.focus = focus;
        self.right = right;
    }

    fn rnd(&mut self, a: f32, b: f32) -> f32 {
        self.seed = self
            .seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let t = (self.seed >> 33) as f32 / (1u64 << 31) as f32;
        a + (b - a) * t
    }

    fn rnd_index(&mut self, len: usize) -> usize {
        if len <= 1 {
            0
        } else {
            (self.rnd(0.0, len as f32 - 0.001) as usize).min(len - 1)
        }
    }

    /// `CSound::CalcPanVol` (MatrixSoundManager.cpp:764-787).
    fn calc_pan_vol(
        &self,
        pos: Vec3,
        attn: f32,
        pan0: f32,
        pan1: f32,
        vol0: f32,
        vol1: f32,
    ) -> (f32, f32) {
        let mut dir = pos - self.focus;
        let mut dist = dir.length();
        if dist != 0.0 {
            dir /= dist;
        }
        dist = (dist - SOUND_FULL_VOLUME_DIST).max(0.0) * attn;
        let dot = self.right.dot(dir);
        let pan = pan0 + (pan1 - pan0) * ((dot + 1.0) / 2.0);
        let k = (1.0 - dist).max(0.0);
        let vol = vol0 + (vol1 - vol0) * k;
        (pan, vol)
    }

    /// `Pos2Key` (cpp:899-931) — quantize to 2·GLOBAL_SCALE cells.
    fn pos2key(pos: Vec3) -> u32 {
        let mut x = (pos.x / SOUND_POS_DIVIDER).round() as i32;
        let mut y = (pos.y / SOUND_POS_DIVIDER).round() as i32;
        let mut z = (pos.z / SOUND_POS_DIVIDER).round() as i32;
        let mut key: u32 = 0;
        if x < 0 {
            x = -x;
            key |= 0x020000;
        }
        x = x.min(4095);
        if y < 0 {
            y = -y;
            key |= 0x02000000;
        }
        y = y.min(4095);
        z = z.clamp(0, 63);
        key |= (x as u32 & 31) | ((x as u32 & 4064) << 5);
        key |= ((y as u32 & 31) << 5) | ((y as u32 & 4064) << (8 + 5));
        key |= (z as u32) << 26;
        key
    }

    /// Inverse of `pos2key` (cpp:284-291) — cell center used for the
    /// periodic positional re-pan.
    fn key2pos(key: u32) -> Vec3 {
        let mut x = ((key & 0x1F) | ((key & 0x1FC00) >> 5)) as i32;
        if key & 0x020000 != 0 {
            x = -x;
        }
        let mut y = (((key & 0x3E0) >> 5) | ((key & 0x1FC0000) >> (8 + 5))) as i32;
        if key & 0x02000000 != 0 {
            y = -y;
        }
        let z = (key >> 26) as i32;
        Vec3::new(
            x as f32 * SOUND_POS_DIVIDER,
            y as f32 * SOUND_POS_DIVIDER,
            z as f32 * SOUND_POS_DIVIDER,
        )
    }

    // ── slot management (cpp:488-558) ───────────────────────────────

    fn stop_slot(&mut self, i: usize) {
        if self.slots[i].id_internal != 0 {
            self.out.destroy(self.slots[i].id_internal);
        }
        self.slots[i].id_internal = 0;
        self.slots[i].id = SOUND_ID_EMPTY;
    }

    fn find_sound_slot(&self, id: u32) -> Option<usize> {
        (0..MAX_SOUNDS).find(|&i| self.slots[i].id == id)
    }

    /// `FindSoundSlotPlayedOnly` (cpp:510-523).
    fn find_sound_slot_played(&mut self, id: u32) -> Option<usize> {
        let i = self.find_sound_slot(id)?;
        if self.out.is_playing(self.slots[i].id_internal) {
            Some(i)
        } else {
            self.stop_slot(i);
            None
        }
    }

    /// `FindSlotForSound` (cpp:525-558) — free / ended slot, else the
    /// quietest voice is evicted.
    fn find_slot_for_sound(&mut self) -> usize {
        let mut minv = 100.0f32;
        let mut deli = 0usize;
        for i in 0..MAX_SOUNDS {
            if self.slots[i].id_internal == 0 {
                return i;
            }
            if self.slots[i].id == SOUND_ID_EMPTY
                || !self.out.is_playing(self.slots[i].id_internal)
            {
                self.stop_slot(i);
                return i;
            }
            if self.slots[i].curvol < minv {
                minv = self.slots[i].curvol;
                deli = i;
            }
        }
        self.stop_slot(deli);
        deli
    }

    fn is_sound_play(&mut self, id: u32) -> bool {
        self.find_sound_slot_played(id).is_some()
    }

    /// `LayerOff` (cpp:376-390).
    fn layer_off(&mut self, sl: usize) {
        let slid = self.layers[sl];
        if slid.index >= 0 && (slid.index as usize) < MAX_SOUNDS {
            let idx = slid.index as usize;
            if slid.id == self.slots[idx].id {
                self.stop_slot(idx);
            }
        }
        self.layers[sl].index = -1;
    }

    fn layer_is_played(&mut self, sl: usize) -> bool {
        let slid = self.layers[sl];
        if slid.index >= 0 && (slid.index as usize) < MAX_SOUNDS {
            let idx = slid.index as usize;
            if self.slots[idx].id == slid.id {
                return self.out.is_playing(self.slots[idx].id_internal);
            }
        }
        false
    }

    /// `PlayInternal` (cpp:667-720).
    fn play_internal(
        &mut self,
        key: &str,
        vol: f32,
        pan: f32,
        layer: SoundLayer,
        interrupt: Interrupt,
    ) -> u32 {
        if !self.out.ready() || vol < 0.00001 {
            return SOUND_ID_EMPTY;
        }
        let Some(def) = self.defs.get(key) else {
            return SOUND_ID_EMPTY;
        };
        let (path, looped) = (def.path.clone(), def.looped);
        let newid = self.last_id;
        self.last_id += 1;

        let sl = layer.index();
        if layer != SoundLayer::All {
            if interrupt == Interrupt::Skip && self.layer_is_played(sl) {
                return SOUND_ID_EMPTY;
            }
            self.layer_off(sl);
            self.layers[sl].id = newid;
        }

        let si = self.find_slot_for_sound();
        let internal = self.out.create(&path, looped);
        if internal == 0 {
            return SOUND_ID_EMPTY;
        }
        self.slots[si] = PlayedSound {
            id_internal: internal,
            id: newid,
            curvol: vol,
            curpan: pan,
        };
        if layer != SoundLayer::All {
            self.layers[sl].index = si as i32;
        }
        self.out.set_pan(internal, pan);
        self.out.set_vol(internal, vol);
        self.out.play(internal);
        newid
    }

    /// `Play(snd, sl, interrupt)` (cpp:722-738).
    pub fn play(&mut self, key: &str, layer: SoundLayer, interrupt: Interrupt) -> u32 {
        let Some(def) = self.defs.get(key) else {
            return SOUND_ID_EMPTY;
        };
        let (v0, v1, p0, p1) = (def.vol0, def.vol1, def.pan0, def.pan1);
        let vol = self.rnd(v0, v1);
        let pan = self.rnd(p0, p1);
        self.play_internal(key, vol, pan, layer, interrupt)
    }

    /// `Play(snd, pos, sl, interrupt)` (cpp:740-762).
    pub fn play_at(&mut self, key: &str, pos: Vec3, layer: SoundLayer, interrupt: Interrupt) -> u32 {
        let Some(def) = self.defs.get(key) else {
            return SOUND_ID_EMPTY;
        };
        let (pan, vol) =
            self.calc_pan_vol(pos, def.attn, def.pan0, def.pan1, def.vol0, def.vol1);
        self.play_internal(key, vol, pan, layer, interrupt)
    }

    /// `Play(id, snd, pos, sl)` (cpp:789-827) — ambient retrigger.
    fn play_handle(&mut self, handle: SndHandle, key: &str, pos: Vec3, layer: SoundLayer) {
        let Some(def) = self.defs.get(key) else {
            return;
        };
        let (attn, p0, p1, v0, v1) = (def.attn, def.pan0, def.pan1, def.vol0, def.vol1);
        let (pan, vol) = self.calc_pan_vol(pos, attn, p0, p1, v0, v1);
        if let Some(&id) = self.handles.get(&handle) {
            if let Some(idx) = self.find_sound_slot_played(id) {
                let internal = self.slots[idx].id_internal;
                self.out.set_pan(internal, pan);
                self.out.set_vol(internal, vol);
                self.slots[idx].curpan = pan;
                self.slots[idx].curvol = vol;
                return;
            }
        }
        let id = self.play_internal(key, vol, pan, layer, Interrupt::Interrupt);
        if id != SOUND_ID_EMPTY {
            self.handles.insert(handle, id);
        } else {
            self.handles.remove(&handle);
        }
    }

    /// `ChangePos` (cpp:829-867) — update only, never restart.
    fn change_pos(&mut self, handle: SndHandle, key: &str, pos: Vec3) {
        let Some(&id) = self.handles.get(&handle) else {
            return;
        };
        let Some(def) = self.defs.get(key) else {
            return;
        };
        let (attn, p0, p1, v0, v1) = (def.attn, def.pan0, def.pan1, def.vol0, def.vol1);
        let (pan, vol) = self.calc_pan_vol(pos, attn, p0, p1, v0, v1);
        if let Some(idx) = self.find_sound_slot_played(id) {
            let internal = self.slots[idx].id_internal;
            self.out.set_pan(internal, pan);
            self.out.set_vol(internal, vol);
            self.slots[idx].curpan = pan;
            self.slots[idx].curvol = vol;
        } else {
            self.handles.remove(&handle);
        }
    }

    fn stop_handle(&mut self, handle: SndHandle) {
        if let Some(id) = self.handles.remove(&handle) {
            if let Some(idx) = self.find_sound_slot_played(id) {
                self.stop_slot(idx);
            }
        }
    }

    /// `AddSound(snd, pos, sl, ifl)` + `CSoundArray::AddSound`
    /// (cpp:934-950, 1178-1237).
    fn add_sound(&mut self, key: &str, pos: Vec3, layer: SoundLayer, ifl: Interrupt) {
        if !self.out.ready() {
            return;
        }
        let cell = Self::pos2key(pos);
        // Same key already in this cell → retrigger or ttl-refresh.
        let existing = self
            .pos_sounds
            .get(&cell)
            .and_then(|arr| arr.iter().position(|e| !e.undef && e.key == key))
            .map(|i| {
                let id = self.pos_sounds[&cell][i].id;
                (i, id)
            });
        if let Some((i, id)) = existing {
            match ifl {
                Interrupt::Interrupt => {
                    if let Some(idx) = self.find_sound_slot_played(id) {
                        self.stop_slot(idx);
                    }
                    self.pos_sounds.get_mut(&cell).unwrap().remove(i);
                }
                Interrupt::Skip => {
                    if self.is_sound_play(id) {
                        let Some(def) = self.defs.get(key) else {
                            return;
                        };
                        let (ttl, fade) = (def.ttl, def.fade);
                        let e = &mut self.pos_sounds.get_mut(&cell).unwrap()[i];
                        e.ttl = ttl;
                        e.fade = fade;
                        return;
                    }
                    self.pos_sounds.get_mut(&cell).unwrap().remove(i);
                }
            }
        }
        let id = self.play_at(key, pos, layer, ifl);
        if id == SOUND_ID_EMPTY {
            return;
        }
        let Some(def) = self.defs.get(key) else {
            return;
        };
        self.pos_sounds.entry(cell).or_default().push(SndData {
            key: key.to_string(),
            undef: false,
            id,
            attn: def.attn,
            pan0: def.pan0,
            pan1: def.pan1,
            vol0: def.vol0,
            vol1: def.vol1,
            ttl: def.ttl,
            fade: def.fade,
        });
    }

    /// `Play(pos, attn, pan0, pan1, vol0, vol1, name)` (cpp:587-629) +
    /// the appending `CSoundArray::AddSound` overload (hpp:357-370):
    /// no layer, never looped, `snd = S_UNDEF` entry semantics.
    #[allow(clippy::too_many_arguments)]
    fn add_sound_named(
        &mut self,
        path: &str,
        pos: Vec3,
        attn: f32,
        pan0: f32,
        pan1: f32,
        vol0: f32,
        vol1: f32,
    ) {
        if !self.out.ready() {
            return;
        }
        let (pan, vol) = self.calc_pan_vol(pos, attn, pan0, pan1, vol0, vol1);
        if vol < 0.00001 {
            return;
        }
        let newid = self.last_id;
        self.last_id += 1;
        let si = self.find_slot_for_sound();
        let internal = self.out.create(path, false);
        if internal == 0 {
            return;
        }
        self.slots[si] = PlayedSound {
            id_internal: internal,
            id: newid,
            curvol: vol,
            curpan: pan,
        };
        self.out.set_pan(internal, pan);
        self.out.set_vol(internal, vol);
        self.out.play(internal);
        self.pos_sounds
            .entry(Self::pos2key(pos))
            .or_default()
            .push(SndData {
                key: path.to_string(),
                undef: true,
                id: newid,
                attn,
                pan0,
                pan1,
                vol0,
                vol1,
                ttl: 1e30,
                fade: 1000.0,
            });
    }

    /// Drain one queued event into the mixer.
    pub fn dispatch(&mut self, ev: SndEvent) {
        match ev {
            SndEvent::Play { key, layer } => {
                self.play(&key, layer, Interrupt::Interrupt);
            }
            SndEvent::PlayAt {
                key,
                pos,
                layer,
                handle,
            } => {
                let id = self.play_at(&key, Vec3::from(pos), layer, Interrupt::Interrupt);
                if let Some(h) = handle {
                    if id != SOUND_ID_EMPTY {
                        self.handles.insert(h, id);
                    }
                }
            }
            SndEvent::PlayHandle {
                handle,
                key,
                pos,
                layer,
            } => self.play_handle(handle, &key, Vec3::from(pos), layer),
            SndEvent::Add {
                key,
                pos,
                layer,
                ifl,
            } => self.add_sound(&key, Vec3::from(pos), layer, ifl),
            SndEvent::AddNamed {
                path,
                pos,
                attn,
                pan0,
                pan1,
                vol0,
                vol1,
            } => self.add_sound_named(&path, Vec3::from(pos), attn, pan0, pan1, vol0, vol1),
            SndEvent::ChangePos { handle, key, pos } => {
                self.change_pos(handle, &key, Vec3::from(pos))
            }
            SndEvent::Stop { handle } => self.stop_handle(handle),
            SndEvent::MusicVolume(v) => self.out.set_music_volume(v),
        }
    }

    /// The `GameSound` order-voice queue (side_player.rs) — resolved
    /// to Sounds-block keys per MatrixSide.cpp:7960-8380, all SL_ORDER.
    pub fn dispatch_game_sound(&mut self, gs: crate::matrix_game::side_player::GameSound) {
        use crate::matrix_game::side_player::GameSound as G;
        let key: &str = match gs {
            G::OrderInProgress1 => "s_ord_inprogress1",
            G::OrderInProgress2 => "s_ord_inprogress2",
            G::OrderAccept => "s_ord_accept",
            G::OrderCapture => "s_ord_capture",
            G::OrderCapturePush => "s_ord_capture_push",
            G::OrderCaptureFuckOff => "s_ord_capoff",
            G::OrderAttack => "s_ord_attack",
            G::OrderRepair => "s_ord_repair",
            G::OrderAutoCapture => "s_orda_capture",
            G::OrderAutoAttack => "s_orda_attack",
            G::OrderAutoDefence => "s_orda_defence",
            G::ChassisMoveTo(kind) | G::ChassisPatrol(kind) => {
                let voices = self.defs.chassis_voices(kind).map(|(m, p)| {
                    if matches!(gs, G::ChassisMoveTo(_)) {
                        m.clone()
                    } else {
                        p.clone()
                    }
                });
                if let Some(v) = voices.filter(|v| !v.is_empty()) {
                    let pick = self.rnd_index(v.len());
                    self.play(&v[pick], SoundLayer::Order, Interrupt::Interrupt);
                }
                return;
            }
        };
        self.play(key, SoundLayer::Order, Interrupt::Interrupt);
    }

    /// `CSound::Takt` (cpp:298-374) + `CSoundArray::UpdateTimings` /
    /// `SetSoundPos` (cpp:1023-1176). `now_ms` is wall-clock.
    pub fn takt(&mut self, now_ms: i64) {
        // 100ms positional refresh.
        let delta = self.next_pos_takt - now_ms;
        if delta < 0 || delta > 100 {
            self.next_pos_takt = now_ms + 100;
            let ms = (if delta < 0 { 100 } else { delta.min(1000) }) as f32;
            let cells: Vec<u32> = self.pos_sounds.keys().copied().collect();
            for cell in cells {
                self.update_cell(cell, ms);
            }
            self.pos_sounds.retain(|_, arr| !arr.is_empty());
        }
        // 1s ended-voice cleanup.
        let delta = self.next_clean_takt - now_ms;
        if delta < 0 || delta > 1000 {
            self.next_clean_takt = now_ms + 1000;
            for i in 0..MAX_SOUNDS {
                if self.slots[i].id_internal != 0
                    && !self.out.is_playing(self.slots[i].id_internal)
                {
                    self.stop_slot(i);
                }
            }
        }
        self.out.takt();
    }

    /// One `m_PosSounds` cell: UpdateTimings(ms) then SetSoundPos.
    fn update_cell(&mut self, cell: u32, ms: f32) {
        let pos = Self::key2pos(cell);
        let mut arr = self.pos_sounds.remove(&cell).unwrap_or_default();
        arr.retain_mut(|e| {
            // UpdateTimings: ttl runs down, then fade counts `-ttl`→0.
            // S_UNDEF (by-name) entries skip the bookkeeping (cpp:1033).
            if !e.undef {
                match self.find_sound_slot_played(e.id) {
                    None => return false,
                    Some(idx) => {
                        if e.ttl < 0.0 {
                            if e.fade < 0.0 {
                                self.stop_slot(idx);
                                return false;
                            }
                            e.fade -= ms;
                        } else {
                            e.ttl -= ms;
                            if e.ttl < 0.0 {
                                e.ttl = -e.fade;
                            }
                        }
                    }
                }
            }
            // SetSoundPos: re-pan/vol at the cell center, fade ramp.
            let Some(idx) = self.find_sound_slot_played(e.id) else {
                return false;
            };
            let k = if !e.undef && e.ttl < 0.0 {
                -e.fade / e.ttl
            } else {
                1.0
            };
            let (pan, mut vol) = self.calc_pan_vol(pos, e.attn, e.pan0, e.pan1, e.vol0, e.vol1);
            vol *= k;
            if vol < 0.00001 {
                self.stop_slot(idx);
                return false;
            }
            let internal = self.slots[idx].id_internal;
            self.out.set_pan(internal, pan);
            self.out.set_vol(internal, vol);
            self.slots[idx].curpan = pan;
            self.slots[idx].curvol = vol;
            true
        });
        if !arr.is_empty() {
            self.pos_sounds.insert(cell, arr);
        }
    }

    /// `StopPlayAllSounds` (cpp:869-880).
    pub fn stop_all(&mut self) {
        for i in 0..MAX_SOUNDS {
            self.stop_slot(i);
        }
        self.pos_sounds.clear();
        self.handles.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_game::config::SoundDef;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct MockState {
        created: Vec<(String, bool)>,
        playing: std::collections::HashMap<u32, bool>,
        vols: std::collections::HashMap<u32, f32>,
        next: u32,
        music_vol: Option<f32>,
    }

    #[derive(Clone, Default)]
    struct MockOutput(Rc<RefCell<MockState>>);

    impl SoundOutput for MockOutput {
        fn ready(&self) -> bool {
            true
        }
        fn create(&mut self, path: &str, looped: bool) -> u32 {
            let mut st = self.0.borrow_mut();
            st.next += 1;
            let v = st.next;
            st.created.push((path.to_string(), looped));
            st.playing.insert(v, false);
            v
        }
        fn play(&mut self, v: u32) {
            self.0.borrow_mut().playing.insert(v, true);
        }
        fn set_pan(&mut self, _v: u32, _p: f32) {}
        fn set_vol(&mut self, v: u32, vol: f32) {
            self.0.borrow_mut().vols.insert(v, vol);
        }
        fn is_playing(&self, v: u32) -> bool {
            *self.0.borrow().playing.get(&v).unwrap_or(&false)
        }
        fn destroy(&mut self, v: u32) {
            self.0.borrow_mut().playing.remove(&v);
        }
        fn set_music_volume(&mut self, vol: f32) {
            self.0.borrow_mut().music_vol = Some(vol);
        }
    }

    fn defs_with(entries: &[(&str, SoundDef)]) -> SoundDefs {
        let mut defs = SoundDefs::default();
        for (k, d) in entries {
            defs.insert_for_test(k, d.clone());
        }
        defs
    }

    fn mixer() -> (SoundMixer, MockOutput) {
        let out = MockOutput::default();
        let mut d = SoundDef::default();
        d.path = "Sound.Test".into();
        let mut looped = SoundDef::default();
        looped.path = "Sound.Loop".into();
        looped.looped = true;
        let m = SoundMixer::new(
            defs_with(&[("test", d), ("loop", looped)]),
            Box::new(out.clone()),
        );
        (m, out)
    }

    #[test]
    fn add_sound_dedups_same_cell() {
        let (mut m, out) = mixer();
        m.set_listener(Vec3::ZERO, Vec3::X);
        let pos = Vec3::new(10.0, 10.0, 0.0);
        m.add_sound("test", pos, SoundLayer::All, Interrupt::Skip);
        m.add_sound("test", pos, SoundLayer::All, Interrupt::Skip);
        // SEF_SKIP + same key + same cell + still playing → one voice.
        assert_eq!(out.0.borrow().created.len(), 1);
        // A different cell plays its own instance.
        m.add_sound("test", pos + Vec3::new(500.0, 0.0, 0.0), SoundLayer::All, Interrupt::Skip);
        assert_eq!(out.0.borrow().created.len(), 2);
    }

    #[test]
    fn layer_replaces_previous_voice() {
        let (mut m, out) = mixer();
        let id1 = m.play("test", SoundLayer::Order, Interrupt::Interrupt);
        assert_ne!(id1, SOUND_ID_EMPTY);
        let v1 = out.0.borrow().next;
        m.play("test", SoundLayer::Order, Interrupt::Interrupt);
        // First order voice was stopped when the second started.
        assert!(!out.0.borrow().playing.contains_key(&v1));
        // SEF_SKIP with a live layer voice → dropped.
        let id3 = m.play("test", SoundLayer::Order, Interrupt::Skip);
        assert_eq!(id3, SOUND_ID_EMPTY);
        assert_eq!(out.0.borrow().created.len(), 2);
    }

    #[test]
    fn eviction_kills_quietest_at_capacity() {
        let (mut m, out) = mixer();
        m.set_listener(Vec3::ZERO, Vec3::X);
        for i in 0..MAX_SOUNDS {
            m.play_at(
                "test",
                Vec3::new(i as f32 * 4.0, 0.0, 0.0),
                SoundLayer::All,
                Interrupt::Interrupt,
            );
        }
        assert_eq!(out.0.borrow().created.len(), MAX_SOUNDS);
        // All 16 slots live; the 17th evicts one (the quietest).
        m.play_at("test", Vec3::new(1.0, 1.0, 0.0), SoundLayer::All, Interrupt::Interrupt);
        assert_eq!(out.0.borrow().created.len(), MAX_SOUNDS + 1);
        assert_eq!(out.0.borrow().playing.len(), MAX_SOUNDS);
    }

    #[test]
    fn handle_retrigger_and_stop() {
        let (mut m, out) = mixer();
        let h = SndHandle::Elevator(7);
        m.dispatch(SndEvent::PlayHandle {
            handle: h,
            key: "loop".into(),
            pos: [0.0; 3],
            layer: SoundLayer::All,
        });
        assert_eq!(out.0.borrow().created.len(), 1);
        assert!(out.0.borrow().created[0].1, "loop def creates looped voice");
        // Retrigger while playing → same voice, just re-panned.
        m.dispatch(SndEvent::PlayHandle {
            handle: h,
            key: "loop".into(),
            pos: [100.0, 0.0, 0.0],
            layer: SoundLayer::All,
        });
        assert_eq!(out.0.borrow().created.len(), 1);
        m.dispatch(SndEvent::Stop { handle: h });
        assert!(out.0.borrow().playing.is_empty());
        // ChangePos on a stopped handle is a no-op, not a restart.
        m.dispatch(SndEvent::ChangePos {
            handle: h,
            key: "loop".into(),
            pos: [0.0; 3],
        });
        assert_eq!(out.0.borrow().created.len(), 1);
    }

    #[test]
    fn music_volume_forwards_to_output() {
        let (mut m, out) = mixer();
        m.dispatch(SndEvent::MusicVolume(0.0));
        assert_eq!(out.0.borrow().music_vol, Some(0.0));
    }

    #[test]
    fn pos2key_roundtrip() {
        for p in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(400.0, 800.0, 40.0),
            Vec3::new(-400.0, 1200.0, 0.0),
            Vec3::new(2000.0, -2000.0, 120.0),
        ] {
            let key = SoundMixer::pos2key(p);
            let back = SoundMixer::key2pos(key);
            assert!(
                (back - p).length() <= SOUND_POS_DIVIDER,
                "{p:?} → {key:08x} → {back:?}"
            );
        }
    }

    #[test]
    fn calc_pan_vol_matches_cpp_rules() {
        let mut m = SoundMixer::new(SoundDefs::default(), Box::new(NullOutput));
        m.set_listener(Vec3::ZERO, Vec3::X);
        // Inside full-volume distance → vol1, pan at right → pan1.
        let (pan, vol) = m.calc_pan_vol(Vec3::new(100.0, 0.0, 0.0), 0.002, -1.0, 1.0, 0.0, 1.0);
        assert!((vol - 1.0).abs() < 1e-6);
        assert!((pan - 1.0).abs() < 1e-6);
        // 700 world units → dist 500 · 0.002 = 1.0 → k = 0 → vol0.
        let (_, vol) = m.calc_pan_vol(Vec3::new(700.0, 0.0, 0.0), 0.002, -1.0, 1.0, 0.25, 1.0);
        assert!((vol - 0.25).abs() < 1e-6);
        // attn = 0 → infinite radius, always vol1.
        let (_, vol) = m.calc_pan_vol(Vec3::new(99999.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.3, 0.9);
        assert!((vol - 0.9).abs() < 1e-6);
        // Left of the listener → pan0.
        let (pan, _) = m.calc_pan_vol(Vec3::new(-500.0, 0.0, 0.0), 0.002, -1.0, 1.0, 0.0, 1.0);
        assert!((pan + 1.0).abs() < 1e-6);
    }
}
