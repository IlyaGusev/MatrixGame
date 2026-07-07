# Sound & Music Proposal

Status quo, design, and implementation plan for making the Rust port audible.
Everything below is grounded in the C++ source (`MatrixSoundManager.{cpp,hpp}`)
and the current state of the port.

## 1. Where we are

### The original's architecture

The MatrixGame engine has **no audio engine of its own**. Every
`CSound::Play` / `AddSound` call funnels through `g_RangersInterface` — the
function-pointer table the SR2 host passes into the DLL
(`m_SoundCreate/m_SoundPlay/m_SoundPan/m_SoundVolume/m_SoundDestroy`).
In the standalone EXE build `g_RangersInterface == NULL` and every play call
is a no-op (`MatrixSoundManager.cpp:564/577`) — **the original standalone
build is silent too**.

Music even more so: the engine never plays a track. It only ducks/restores the
*host's* music volume (`CMatrixMap::SetMusicVolume`, `MatrixMap.cpp:3583`,
calling `g_RangersInterface->m_MusicVolumeGet/Set`) — e.g. the Terron death
mutes music. The SR2 host owns the soundtrack entirely.

### What the port has

- **Dispatch surface: complete.** Every C++ sound call site is wired with the
  canonical `Sounds`-block key:
  - world sounds → `Objects::pending_sounds` via `queue_snd[_at]`
    (weapon fire/hit keys, explosions, capture voices, base platform,
    "under attack", …), drained per-takt in `logic.rs` to `log::trace`;
  - UI sounds → `interface/sound.rs` (`play`, `play_named`,
    `play_hint_sound`) — logging stubs.
- **Per-sound config: available.** The `Sounds` block lives in
  `Data/robots.dat` (the same `Storage` the port already parses): per key
  `path`, `pan` (pan0,pan1), `vol` (vol0,vol1), `looped`, `ttl` (ttl,fade),
  `attn`. 118 entries carry a `path`.
- **Assets: NOT in our tree.** The `path` values are SR2 *resource paths*
  (`Sound.FormShipClose` style), resolved by the host from the SR2 main
  game's archives. None of the local pkgs (`robots.pkg`, `common.pkg`,
  `forms.pkg`, `mainmenu.pkg`, `russian.pkg`) contain any `.wav/.ogg` —
  verified by scanning the archives. The audio payloads must come from an
  SR2 installation.

## 2. Goals / non-goals

**Goals**

1. Faithful SFX playback on the WASM build (primary target) with the
   original mixing policy: attenuation by distance from the camera focus,
   stereo pan from the camera right vector, looped sounds, the positional
   dedup, and the layer system.
2. Asset pipeline consistent with the existing `pack_bundle` flow.
3. Graceful silence when assets are absent (exactly today's behaviour).

**Non-goals**

- Native (`rodio`) backend — nice-to-have, not scheduled (the native binary
  is a dev tool).
- Music *fidelity*: the original delegates music wholesale to SR2. We propose
  a minimal looped-track player (§6) since there is nothing to port.

## 3. Architecture

Keep the split the port already has: **game logic produces events, the host
consumes them**. No `web-sys` calls inside `matrix_game::*`.

```
matrix_game (logic)                    app / platform layer
──────────────────                     ─────────────────────
objs.queue_snd_at(key, pos)  ──────►   SoundBackend::dispatch(key, Some(pos))
interface/sound.rs play_*    ──────►   SoundBackend::dispatch(key, None)
                                        │
                                        ├── SoundDefs (from robots.dat "Sounds")
                                        │     path, vol0/1, pan0/1, attn, looped, ttl
                                        ├── SampleCache (decoded AudioBuffers)
                                        └── WebAudio graph:
                                            AudioBufferSourceNode → GainNode → StereoPannerNode → master Gain
```

New crate-level module `src/platform/audio.rs` (+ `audio_web.rs` behind
`#[cfg(target_arch = "wasm32")]`), consumed from `form_game.rs` in the same
place the queues are drained today.

### Mixing policy to port (`MatrixSoundManager.cpp`)

| Piece | C++ source | Rule |
|---|---|---|
| Pan/volume | `CalcPanVol` (:764-787) | `dist = max(0, |pos−camera.frustum_center| − 200)`; `k = max(0, 1 − dist·attn)`; `vol = lerp(vol0→vol1, k)`; `pan = lerp(pan0→pan1, (dot(camera.right, dir)+1)/2)` |
| Attenuation | `SureLoaded` (:439-452) | `attn = 0.002 · attn_par` (default 0.002, `0` ⇒ infinite radius) |
| Cull | `Play` (:603) | skip start if computed `vol < 0.00001` |
| Positional dedup | `AddSound` + `m_PosSounds` | world position quantized to `2·GLOBAL_SCALE` cells; the same key already playing in the same cell is *retriggered/updated*, not layered — this is what keeps 30 volcano turrets from clipping |
| Position updates | `CSound::Takt` (:300-328) | every 100 ms, re-`CalcPanVol` for live positional sounds; TTL/fade bookkeeping |
| Layers | `ESoundLayer` (SL_ORDER, SL_SELECTION, …) | one sound per layer; a new play on a layer stops the previous one (used for voice lines so orders don't overlap) |
| Looped | `looped` par + `ChangePos` | plasma/laser hum, base ambient: loop while the weapon fires, re-pan/re-vol on `ChangePos` |

The port's queues currently carry `(key, Option<pos>)`. Two small extensions
needed at the call sites (mechanical, the C++ reference is explicit at each):

1. **Layer tag** for the `CSound::Play(S_*, SL_*)` sites (order voices,
   selection voices) — add an enum field to the queued tuple.
2. **Looped-sound handles** for `ChangePos` semantics (weapon hum follows the
   barrel). Suggest: `queue_snd_loop(key, handle, pos)` where `handle` is the
   `WeaponId` — mirroring how `pending_light_follow` already keys follow-lights.

## 4. Assets

Two-tier source resolution, no licensing surprises:

1. **From an SR2 install (primary).** A new example
   `cargo run --example pack_sounds -- <path-to-SR2>` reads the SR2 resource
   archives, resolves every `path` in the `Sounds` block
   (`Sound.X.Y` → the host's sound resource tree), decodes to WAV/OGG and
   writes `assets/sounds.bundle` (same `AssetBundle` container as maps).
   The WASM loader fetches it lazily (`?sounds=<url>` or a fixed name probed
   at startup); absence ⇒ silent, no error.
2. **None (fallback).** Without the bundle everything behaves exactly as
   today — the dispatch drains to trace logs.

We do **not** commit audio payloads to the repo (SR2 assets are proprietary;
the repo's GPL covers code only — same reason `Data/` is untracked).

## 5. Implementation plan

Each step is independently shippable and verifiable.

1. **SoundDefs loader** — parse the `Sounds` block from `robots.dat` into a
   `HashMap<String, SoundDef>` at config-load time (next to the existing
   table loaders in `config.rs`). Unit test against real robots.dat values
   (e.g. `wplasma`, `expl_bb`).
   *~150 lines.*
2. **Backend trait + WebAudio impl** — `SoundBackend { dispatch, takt,
   set_master_volume }`; WebAudio graph per §3; sample cache keyed by path;
   decode via `decodeAudioData`. Autoplay policy: create/resume the
   `AudioContext` on the first user gesture (we already have pointer handlers).
   *~300 lines, all in `platform/`.*
3. **Wire the queues** — replace the trace-drain in `form_game.rs` with
   backend dispatch; `CalcPanVol` port + 100 ms position-update takt +
   quantized dedup map. Camera focus/right come from the existing `Camera`.
   *~200 lines.*
4. **Layers + voice lines** — add the `ESoundLayer` tag at the queue sites
   that use `SL_*` in C++ (order/selection/build voices), implement
   one-per-layer replacement.
   *~100 lines + call-site sweep.*
5. **Looped weapon hum** — `queue_snd_loop`/`snd_loop_kill` keyed by
   `WeaponId` (pattern copied from `pending_light_follow`), backing
   `ChangePos` semantics.
   *~120 lines.*
6. **`pack_sounds` example** — SR2 resource reader + bundle writer + loader
   probe in the WASM boot path.
   *Effort depends on the SR2 archive format; the reader may already exist in
   community tooling — investigate first.*

Suggested order: 1 → 2 → 3 ship "80% audible" (all one-shot world + UI SFX);
4-5 complete fidelity; 6 unblocks everything and can proceed in parallel.

## 6. Music (minimal, optional)

There is no music system to port — SR2 owned it. Proposal:

- `assets/music/*.ogg` (user-supplied, e.g. SR2's planetary-battle tracks or
  anything else) listed in a tiny manifest; the backend loops a
  shuffled playlist through a dedicated `GainNode`.
- Port the two hooks the engine *does* have: `SetMusicVolume(0)` /
  `RestoreMusicVolume` (Terron death at `MatrixObject.cpp:170`, dev console
  `music 0/1`) → fade the music gain over ~1 s (`m_TargetMusicVolume` is
  interpolated in the C++ too).
- No assets ⇒ no music, no error.

*~150 lines, independent of §5.*

## 7. Risks

- **Browser autoplay policies** — mitigated by gesture-gated `AudioContext`
  resume (step 2); until the first click the queue simply drops.
- **SR2 archive format for §4** — unknown effort; worst case we ship the
  backend with a "drop WAVs in a folder, run pack_sounds --dir" escape hatch.
- **Perf** — battles queue tens of sounds per second; the dedup map (§3) is
  what the original used to bound voice count (`MAX_SOUNDS = 16`,
  `FindSlotForSound` evicts the quietest). Port that eviction as-is.
