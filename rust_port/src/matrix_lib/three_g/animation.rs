//! Port of `CVectorObjectAnim` (VectorObject.{cpp,hpp}) — the
//! per-instance animation cursor that runs on top of a shared
//! `VoMesh`. In C++ each `SMatrixRobotUnit` owns one; the same
//! cursor will be allocated per robot + per unit in this port.
//!
//! Ported methods:
//!   * `takt(cms)` → `CVectorObjectAnim::Takt(cms)` at
//!     VectorObject.cpp:1863-1928 — advances `m_Time`, walks
//!     `m_Frame` forward as many ticks as needed, loops or clamps
//!     at the end, and syncs `m_VOFrame` (the geometry-frame index
//!     the renderer samples from `VoMesh::frames`).
//!   * `set_anim_by_name` → `SetAnimByName(name)` at
//!     VectorObject.hpp:524 — returns true on failure (matches the
//!     original's weird "true means not found" convention).
//!   * `first_frame` / `is_anim_end` — same naming as C++.

use std::sync::{Arc, OnceLock, RwLock};

use crate::matrix_lib::three_g::vector_object::VoMesh;

/// Global per-chassis VoMesh table, populated once at load time by
/// `RobotsRenderer::new`. Ports the role of `g_CacheHeap`-owned VO
/// singletons in the C++ engine — the AI layer (`robot.rs::logic_takt`)
/// needs to walk frame durations to drive animation without holding
/// a direct reference to the GPU-side renderer.
///
/// Indexed by `ChassisKind as usize`. `None` for kinds whose VO
/// failed to load.
static CHASSIS_VOS: OnceLock<RwLock<[Option<Arc<VoMesh>>; 5]>> = OnceLock::new();

fn chassis_slot() -> &'static RwLock<[Option<Arc<VoMesh>>; 5]> {
    CHASSIS_VOS.get_or_init(|| RwLock::new([const { None }; 5]))
}

pub fn set_chassis_vo(idx: usize, vo: Arc<VoMesh>) {
    if idx < 5 {
        chassis_slot().write().unwrap()[idx] = Some(vo);
    }
}

pub fn chassis_vo(idx: usize) -> Option<Arc<VoMesh>> {
    chassis_slot().read().unwrap().get(idx).cloned().flatten()
}

#[derive(Debug, Clone)]
pub struct AnimState {
    /// `m_Anim` — index into `VoMesh::animations`.
    pub anim: i32,
    /// `m_Frame` — animation-local frame cursor (0..frames_count).
    pub frame: i32,
    /// `m_VOFrame` — resolved VO frame index the renderer samples.
    pub vo_frame: usize,
    /// `m_Time` — accumulated animation-local ms.
    pub time_ms: i64,
    /// `m_TimeNext` — game-time threshold at which `frame` advances
    /// (used by `takt`, the normal constant-rate path).
    pub time_next_ms: i64,
    /// `m_AnimLooped` — 0 = play once and hold last frame, 1 = loop.
    pub looped: bool,
    /// `SMatrixRobotUnit::m_NextAnimTime` (MatrixObjectRobot.hpp) —
    /// game-time threshold at which `next_frame` advances (used by
    /// the speed-based per-frame path in `DoAnimation`). Separate
    /// from `time_next_ms` because the speed-based path uses
    /// `g_MatrixMap->GetTime()` as its clock, not the accumulated
    /// anim-local time.
    pub next_anim_time: f64,
}

impl Default for AnimState {
    fn default() -> Self {
        Self {
            anim: 0,
            frame: 0,
            vo_frame: 0,
            time_ms: 0,
            time_next_ms: 0,
            looped: true,
            next_anim_time: 0.0,
        }
    }
}

impl AnimState {
    /// Port of `FirstFrame` (VectorObject.hpp:552): reset frame cursor
    /// and set the next-advance threshold from anim 0's duration.
    pub fn first_frame(&mut self, vo: &VoMesh) {
        self.frame = 0;
        let (idx, time) = anim_frame(vo, self.anim, 0).unwrap_or((0, 0));
        self.vo_frame = idx;
        // Clamp to min 1ms — see takt() for why.
        self.time_next_ms = self.time_ms + (time as i64).max(1);
    }

    /// Port of `SetAnimByName(name)` (VectorObject.hpp:524). Returns
    /// `true` on failure (anim not found) to match the C++ convention.
    /// When the name resolves, the cursor resets to frame 0 and the
    /// loop flag is taken from `VoAnim::is_looped` (we don't carry
    /// that yet, so default to the passed `looped`).
    pub fn set_anim_by_name(&mut self, vo: &VoMesh, name: &str, looped: bool) -> bool {
        let Some(idx) = vo.animations.iter().position(|a| a.name == name) else {
            return true;
        };
        self.anim = idx as i32;
        self.looped = looped;
        self.first_frame(vo);
        false
    }

    /// Port of `IsAnimEnd` (VectorObject.hpp:553).
    pub fn is_anim_end(&self, vo: &VoMesh) -> bool {
        if self.looped { return false; }
        match vo.animations.get(self.anim as usize) {
            Some(a) => self.frame as usize == a.frames.len().saturating_sub(1),
            None => true,
        }
    }

    /// Port of `Takt(cms)` at VectorObject.cpp:1863-1928. Advances
    /// `time_ms` and walks `frame` forward as long as `time_ms >
    /// time_next_ms`. Returns `true` if `frame` changed so the caller
    /// can refresh a dirty flag. Non-looped anims delay 1000ms before
    /// stalling on the final frame, matching the C++.
    ///
    /// We clamp per-frame duration to min 1ms: the original engine
    /// assumes authored non-zero durations, but parsed data can carry
    /// 0 for default / unset frames, which would infinite-loop the
    /// `while` since `time_next_ms` wouldn't advance. Also caps total
    /// iterations at `2 * fcnt` as a defense-in-depth safeguard.
    pub fn takt(&mut self, vo: &VoMesh, cms: i32) -> bool {
        self.time_ms += cms as i64;
        let old_frame = self.frame;

        let fcnt = vo.animations.get(self.anim as usize)
            .map(|a| a.frames.len() as i32)
            .unwrap_or(0);
        if fcnt == 0 { return false; }

        let max_iters = (fcnt as i64) * 2 + 2;
        let mut iters = 0i64;
        while self.time_ms > self.time_next_ms && iters < max_iters {
            iters += 1;
            self.frame += 1;
            if self.looped {
                if self.frame >= fcnt { self.frame = 0; }
            } else if self.frame >= fcnt {
                self.time_next_ms += 1000;
                self.frame = fcnt - 1;
                break;
            }
            let (_, t) = anim_frame(vo, self.anim, self.frame).unwrap_or((0, 0));
            self.time_next_ms += (t as i64).max(1);
        }
        // If we hit the iteration cap (i.e. the data itself is
        // pathological — all-zero frame times or similar), forcibly
        // re-sync `time_next_ms` so we don't immediately re-enter
        // the loop next tick.
        if iters >= max_iters {
            self.time_next_ms = self.time_ms + 16;
        }

        if old_frame != self.frame {
            if let Some((idx, _)) = anim_frame(vo, self.anim, self.frame) {
                self.vo_frame = idx;
            }
            true
        } else {
            false
        }
    }
}

/// Helper: look up `(VoFrameRef.frame_index, time_ms)` for
/// `animations[anim].frames[frame]`. Out-of-range returns None.
fn anim_frame(vo: &VoMesh, anim: i32, frame: i32) -> Option<(usize, i32)> {
    let a = vo.animations.get(anim as usize)?;
    let f = a.frames.get(frame as usize)?;
    Some((f.frame_index, f.time_ms))
}

impl AnimState {
    /// Port of `CVectorObjectAnim::NextFrame` (VectorObject.cpp:1930-
    /// 1945). Advances `frame` by one (wrapping if looped, clamping
    /// otherwise), updates `vo_frame`, and returns the NEW current
    /// frame's duration — the caller uses that to compute how much
    /// to bump `next_anim_time` by.
    pub fn next_frame(&mut self, vo: &VoMesh) -> i32 {
        let fcnt = vo.animations.get(self.anim as usize)
            .map(|a| a.frames.len() as i32)
            .unwrap_or(0);
        if fcnt == 0 { return 0; }
        if self.looped {
            self.frame += 1;
            if self.frame >= fcnt { self.frame = 0; }
        } else if self.frame < fcnt - 1 {
            self.frame += 1;
        } else {
            // Last frame of a non-looped anim — return its
            // duration and don't advance.
            return anim_frame(vo, self.anim, self.frame).map(|(_, t)| t).unwrap_or(0);
        }
        let (idx, t) = anim_frame(vo, self.anim, self.frame).unwrap_or((0, 0));
        self.vo_frame = idx;
        t
    }
}
