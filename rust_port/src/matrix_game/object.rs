//! Decorative objects — palms, rocks, trees, grass, etc.
//!
//! Loads .vo meshes (matrix_lib/three_g/vector_object.rs) for each object type_id referenced by
//! the map, then draws all instances of each type as one instanced draw call.
//! Alpha-tested sampling handles foliage texture cutouts without z-ordering.
//!
//! Also hosts the `MapObject` game-object type — port of
//! `CMatrixMapObject` (MatrixObject.{cpp,hpp}). Rendering-side batching
//! (`ObjectsRenderer`, below) stays per-VO-type; the `MapObject` side
//! holds per-instance logical state (behaviour, TTLs, UID) and plugs
//! into the tick loop via the `MapStatic` trait.

use std::collections::{BTreeMap, HashMap};

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::matrix_lib::base::storage::Storage;
use crate::matrix_game::effects::point_light::PointLightSystem;
use crate::matrix_game::common::{unpack_rgb, FOG_END, FOG_START};
use crate::matrix_game::map::{GameMap, ObjectInstance, ObjectShadow};
use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectType, MR_ALL, MR_GRAPH, MR_MATRIX,
    MR_SHADOW_PROJ_GEOM, MR_SHADOW_PROJ_TEX, OBJECT_STATE_SPECIAL,
    OBJECT_STATE_TRACE_INVISIBLE,
};
use crate::matrix_game::common::{OTP_BEHAVIOUR, OTP_INVLOGIC};
use crate::matrix_game::rnd::Rnd;
use crate::matrix_lib::base::wstr;
use crate::matrix_lib::three_g::vector_object::{self, ShadowKind, VoAnimation, VoFrame, VoSurfaceMesh};
use crate::matrix_game::camera::Camera;
use crate::matrix_lib::three_g::texture::{
    create_solid_texture, create_texture_from_rgba, decode_texture_bytes,
};

// ── Game-object side of CMatrixMapObject ────────────────────────────────
//
// Scope B: the struct + `MapStatic` trait impl. The member list matches
// `MatrixObject.hpp:38-99` except fields tied to engine subsystems that
// haven't been ported yet (m_Graph/m_ShadowStencil/m_ShadowProj render
// state, the behaviour union's progress-bar pointers, spawner's
// m_SpawnRobotCore). Those arrive with their owning subsystem.

/// Ports `EBehFlag` (MatrixObject.hpp:10-23). Discriminants match the C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BehFlag {
    Static  = 0,
    Burn    = 1,
    Break   = 2,
    Anim    = 3,
    Sens    = 4,
    Spawner = 5,
    Terron  = 6,
    Portret = 7,
}

/// Port of `CMatrixMapObject` (MatrixObject.hpp:27-167). Holds the
/// per-instance logical state of one decorative / interactive map
/// object. Rendering is handled separately by [`ObjectsRenderer`].
pub struct MapObject {
    core: ObjectCore,
    rchange: u32,
    object_state: u32,
    ablaze_ttl: i32,
    shorted_ttl: i32,

    // Placement (MatrixObject.hpp:39-43)
    pub angle_z: f32,
    pub angle_x: f32,
    pub angle_y: f32,
    pub scale: f32,
    pub tex_bias: f32,

    /// `m_Type` — index into `g_MatrixMap->IdsGet(type)` string table.
    pub type_id: i32,
    pub uid: i32,

    /// Parsed behaviour — derived from the Ids behaviour-string parser
    /// when the side / Ids table is available. `Static` is the C++
    /// fallthrough when no known keyword matched (MatrixObject.cpp:988).
    pub beh_flag: BehFlag,

    // --- Union-backed per-behaviour state (MatrixObject.hpp:61-93) ---
    //
    // C++ overlays three struct shapes on the same bytes; Rust doesn't
    // have that without unsafe unions, so we keep the fields distinct.
    // Adds only what ported behaviours currently touch; more land with
    // their branches.

    /// `m_Photo` (BEHF_PORTRET): 0 = mask variant, 1 = back variant
    /// (MatrixObject.hpp:90). Toggled in `takt` on time-out.
    pub photo: i32,
    /// `m_PhotoTime` — ms until the next toggle. Counted down in takt.
    pub photo_time: i32,

    /// `m_PrevStateRobotsInRadius` (BEHF_SENS/SPAWNER/PORTRET, union
    /// with BEHF_PORTRET's first field in C++). -1 = uninitialised,
    /// 0 = idle, 1 = sensor triggered (robots nearby) — used to detect
    /// rising/falling edges for anim-switch + sound (MatrixObject.hpp:84).
    pub prev_state_robots_in_radius: i32,
    /// `m_SensRadius` — the detection radius parsed out of
    /// `"Sens,<radius>"` (MatrixObject.cpp:1080). Also reused by
    /// BEHF_SPAWNER for the "spawn if robot within" check.
    pub sens_radius: f32,
}

impl MapObject {
    /// Build a `MapObject` at construction time from a map-placed
    /// `ObjectInstance`. Ports the fragment of `CMatrixMapObject::Init`
    /// that wires up transform + type_id + default BehFlag
    /// (MatrixObject.cpp:36-62, :976-996). The `RChange(MR_Graph|…)`
    /// on init line (:982) is reproduced by the `m_RChange(0xffffffff)`
    /// default in [`ObjectCore`] + a redundant explicit set here to
    /// mirror the original literally.
    pub fn from_instance(inst: &ObjectInstance) -> Self {
        let mut core = ObjectCore {
            obj_type: ObjectType::MapObject,
            ..Default::default()
        };
        // Seed `m_Matrix._41..43` with the placement so a later `r_need`
        // call (not ported in scope B) can rebuild the full rotation
        // around it — matches MatrixObject.cpp:382 which reads the
        // translation out of the current matrix before rebuilding.
        core.matrix.w_axis.x = inst.x;
        core.matrix.w_axis.y = inst.y;
        core.matrix.w_axis.z = inst.z;
        // `m_Core->m_GeoCenter` is normally computed by `JoinToGroup`
        // (MatrixMapStatic.cpp:160-178) from CalcBounds. Until that
        // pipeline is ported, seed it from the placement so spatial
        // queries (`FindObjects`) have a usable point. `radius` stays 0
        // — the `radius * oscale` term in find_objects' distance test
        // vanishes, reducing to pure point-in-radius semantics, which
        // matches small-decoration expectations (palms, rocks).
        core.geo_center = glam::Vec3::new(inst.x, inst.y, inst.z);

        Self {
            core,
            rchange: MR_ALL,                  // m_RChange(0xffffffff)
            object_state: 0,
            ablaze_ttl: 0,
            shorted_ttl: 0,
            angle_z: inst.angle_z,
            angle_x: inst.angle_x,
            angle_y: inst.angle_y,
            scale: inst.scale.max(0.0001),
            tex_bias: -1.0,                   // MatrixObject.cpp:43
            type_id: inst.type_id as i32,
            uid: -1,                          // MatrixObject.cpp:61
            beh_flag: BehFlag::Static,        // MatrixObject.cpp:56
            photo: 0,
            photo_time: 0,
            prev_state_robots_in_radius: 0,
            sens_radius: 0.0,
        }
    }

    /// Drives this object's behaviour setup from the Ids row — port of
    /// `CMatrixMapObject::Init` (MatrixObject.cpp:976-1096). The return
    /// value mirrors the implicit "should the caller `AddLT()` this
    /// object?" answer — true when `m_BehFlag != BEHF_STATIC &&
    /// !BEHF_BURN`, since only those opt into logic-takts at init
    /// (Burn is lazily enrolled when damage is taken).
    ///
    /// `ids_row` is the `*`-delimited string from `m_Ids[m_Type]`.
    /// `on_before_win_inc` is a callback invoked when the row's
    /// behaviour field starts with `+` (MatrixObject.cpp:1020-1026) —
    /// the original bumps `g_MatrixMap->m_BeforeWinCount`. Callers that
    /// don't care can pass a no-op.
    pub fn apply_ids_row<F: FnMut()>(
        &mut self,
        ids_row: &str,
        rng: &mut Rnd,
        mut on_before_win_inc: F,
    ) -> bool {
        // OTP_INVLOGIC — trace invisibility bit
        // (MatrixObject.cpp:1006-1016). Only honored when the row has at
        // least that many `*`-parts, matching the `OTP_INVLOGIC < pcnt`
        // guard in the original.
        let pcnt = wstr::count_par(ids_row, "*");
        if OTP_INVLOGIC < pcnt {
            let invl = wstr::str_par(ids_row, OTP_INVLOGIC, "*");
            if invl == "1" {
                self.object_state |= OBJECT_STATE_TRACE_INVISIBLE;
            } else {
                self.object_state &= !OBJECT_STATE_TRACE_INVISIBLE;
            }
        }

        let mut beh = wstr::str_par(ids_row, OTP_BEHAVIOUR, "*").to_string();

        // '+' prefix → "special" object — death contributes to win
        // condition (MatrixObject.cpp:1020-1026).
        if beh.starts_with('+') {
            beh.remove(0);
            self.object_state |= OBJECT_STATE_SPECIAL;
            on_before_win_inc();
        }

        if wstr::compare_first(&beh, "Burn") {
            self.beh_flag = BehFlag::Burn;
            // `m_NextTime / m_BurnTimeTotal / m_BurnSkinVis` zero-init —
            // matches the defaults in our ctor. The `Tex`-variant burn
            // skin load (MatrixObject.cpp:1035-1042) depends on the skin
            // manager, still unported.
            false  // BEHF_BURN is not AddLT'd — it enrolls on damage.
        } else if wstr::compare_first(&beh, "Break") {
            // "Break,<something>,<hp>,Terron?"  (MatrixObject.cpp:1043-1058)
            let kind = wstr::str_par(&beh, 3, ",");
            if kind == "Terron" {
                self.beh_flag = BehFlag::Terron;
            } else {
                self.beh_flag = BehFlag::Break;
            }
            true
        } else if wstr::compare_first(&beh, "Anim") {
            // "Anim" or "AnimP" (Portret)
            // (MatrixObject.cpp:1060-1075). The 5th char (index 4) was
            // `temp[5]` in C++ — that 1-based indexing counts the `,`
            // after "Anim", so we look at char index 4 in Rust.
            if beh.chars().nth(4) == Some('P') {
                self.beh_flag = BehFlag::Portret;
                // MatrixObject.cpp:1066-1067 —
                //   m_PhotoTime = g_MatrixMap->Rnd(1000,2000);
                //   m_Photo = 0;
                self.photo_time = rng.range(1000, 2000);
                self.photo = 0;
                false  // BEHF_PORTRET doesn't AddLT (Takt handles it).
            } else {
                self.beh_flag = BehFlag::Anim;
                true
            }
        } else if wstr::compare_first(&beh, "Sens") {
            // "Sens,<radius>" (MatrixObject.cpp:1076-1083).
            self.beh_flag = BehFlag::Sens;
            // `m_SensRadius` is parsed from the 2nd ","-separated field
            // of the behaviour string (MatrixObject.cpp:1080).
            self.sens_radius = wstr::double_par(&beh, 1, ",") as f32;
            // -1 = uninitialised (first takt kicks the idle anim).
            self.prev_state_robots_in_radius = -1;
            // `SetAblazeTTL(101)` — the branch reuses the ablaze TTL
            // field as a detection-period timer.
            self.ablaze_ttl = 101;
            true
        } else if wstr::compare_first(&beh, "Spawn") {
            // "Spawn,<radius>" (MatrixObject.cpp:1084-1092).
            self.beh_flag = BehFlag::Spawner;
            self.sens_radius = wstr::double_par(&beh, 1, ",") as f32;
            self.prev_state_robots_in_radius = -1;
            self.ablaze_ttl = 101; // absolute timer in C++ uses GetTime();
                                   // the offset is what matters for the
                                   // first-tick delay.
            true
        } else {
            // Fallthrough: no recognised keyword → remains BEHF_STATIC.
            // Matches the default from the ctor (MatrixObject.cpp:56).
            false
        }
    }
}

impl MapStatic for MapObject {
    fn core(&self) -> &ObjectCore { &self.core }
    fn core_mut(&mut self) -> &mut ObjectCore { &mut self.core }
    fn rchange(&self) -> u32 { self.rchange }
    fn rchange_set(&mut self, b: u32) { self.rchange |= b; }
    fn rchange_clear(&mut self, b: u32) { self.rchange &= !b; }
    fn object_state(&self) -> u32 { self.object_state }
    fn object_state_set(&mut self, b: u32) { self.object_state |= b; }
    fn object_state_clear(&mut self, b: u32) { self.object_state &= !b; }
    fn ablaze_ttl(&self) -> i32 { self.ablaze_ttl }
    fn set_ablaze_ttl(&mut self, t: i32) { self.ablaze_ttl = t; }
    fn shorted_ttl(&self) -> i32 { self.shorted_ttl }
    fn set_shorted_ttl(&mut self, t: i32) { self.shorted_ttl = t; }

    /// Port of `CMatrixMapObject::RNeed` (MatrixObject.cpp:374-669). The
    /// full implementation rebuilds the world matrix, loads the
    /// `CVectorObjectAnim`, builds stencil/projected shadows, and
    /// re-renders the minimap patch. In scope B only the matrix branch
    /// is ported — the others require cache/skin/shadow subsystems that
    /// the Rust port doesn't yet have.
    fn r_need(&mut self, need: u32) {
        if need & self.rchange & MR_MATRIX != 0 {
            self.rchange &= !MR_MATRIX;

            // Preserve the current translation, rebuild rotation × scale.
            let pos = self.core.matrix.w_axis.truncate();

            let (sx, cx) = self.angle_x.sin_cos();
            let (sy, cy) = self.angle_y.sin_cos();
            let (sz, cz) = self.angle_z.sin_cos();
            let rx = Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx]);
            let ry = Mat3::from_cols_array(&[cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy]);
            let rz = Mat3::from_cols_array(&[cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0]);
            let rot = rx * ry * rz * self.scale;

            self.core.matrix = Mat4::from_cols(
                rot.x_axis.extend(0.0),
                rot.y_axis.extend(0.0),
                rot.z_axis.extend(0.0),
                pos.extend(1.0),
            );
            self.core.inv_matrix = self.core.matrix.inverse();

            // `JoinToGroup()` at MatrixObject.cpp:416 — needs the map-group
            // arena from `GameMap`; deferred with the group subsystem port.
        }

        // MR_GRAPH / MR_SHADOW_* / MR_MINIMAP branches all depend on
        // `LoadObject` / `CVOShadowStencil` / `CMatrixShadowProj` /
        // `CMinimap::RenderObjectToBackground`. Stubbed for scope B —
        // clear the dirty bits so the next `r_need` doesn't spin on them.
        if need & MR_GRAPH != 0 { self.rchange &= !MR_GRAPH; }
        if need & MR_SHADOW_PROJ_GEOM != 0 { self.rchange &= !MR_SHADOW_PROJ_GEOM; }
        if need & MR_SHADOW_PROJ_TEX != 0 { self.rchange &= !MR_SHADOW_PROJ_TEX; }
    }

    /// Port of `CMatrixMapObject::Takt` (MatrixObject.cpp:671-731). The
    /// graphic takt.
    fn takt(&mut self, cms: i32, rng: &mut Rnd, _objs: &mut crate::matrix_game::map_static::Objects) {
        if self.beh_flag == BehFlag::Portret {
            // MatrixObject.cpp:675-689 — photo toggle. `m_PhotoTime`
            // counts down; at 0 the `m_Photo` bit flips and the timer
            // reseeds with either a long (3000-5000ms) or short
            // (100-200ms) interval depending on the new state. Also
            // flags MR_Graph so the next `r_need` refreshes the
            // mask/back skin swap.
            self.photo_time -= cms;
            if self.photo_time < 0 {
                self.photo_time = if self.photo != 0 {
                    rng.range(3000, 5000)
                } else {
                    rng.range(100, 200)
                };
                self.photo ^= 1;
                self.rchange |= MR_GRAPH;
            }
        }
        // `m_Graph->Takt(cms)` — per-instance animation tick. Animation
        // currently runs once-per-VO-type in `ObjectsRenderer::takt`, not
        // per-instance. Revisit if a scenario needs per-instance phase.
    }

    /// Port of `CMatrixMapObject::LogicTakt` (MatrixObject.cpp:1229-1596).
    /// Massive switch on `m_BehFlag`. Currently handles BEHF_SENS (the
    /// sensor-radius detection path); other branches enroll via
    /// `apply_ids_row` but their bodies need subsystems (progress bars,
    /// effects, sound, robot spawning) that aren't ported.
    fn logic_takt(&mut self, ms: i32, _rng: &mut Rnd, objs: &mut crate::matrix_game::map_static::Objects) {
        use crate::matrix_game::common::TRACE_ROBOT;
        use crate::matrix_game::map_static::fit_to_mask as _fit_to_mask;
        let _ = _fit_to_mask;  // keep import used even when SENS disabled

        if self.beh_flag == BehFlag::Sens {
            // Port of the BEHF_SENS branch (MatrixObject.cpp:1485-1532).
            // `m_PrevStateRobotsInRadius < 0` → first tick: kick the
            // idle animation. The Rust port doesn't have per-instance
            // `m_Graph`, so the SetAnimById call is deferred; the
            // state field carries the transition.
            if self.prev_state_robots_in_radius < 0 {
                self.prev_state_robots_in_radius = 0;
                // C++: m_Graph->SetAnimById(behaviour.GetStrPar(1,",").GetIntPar(1,":"));
                // Deferred — per-instance anim not ported.
            }

            // Timer countdown. On expiry, re-arm with +107ms (MatrixObject.cpp:1499).
            self.ablaze_ttl -= ms;
            if self.ablaze_ttl < 0 {
                while self.ablaze_ttl < 0 {
                    self.ablaze_ttl += 107;
                }

                let pos = glam::Vec2::new(self.core.matrix.w_axis.x, self.core.matrix.w_axis.y);
                // Self-skip is implicit: our own slot is empty inside
                // `logic_takt` (take-the-box pattern in proceed_logic),
                // so find_objects cannot hit us. Passing `None` matches
                // the C++ which also doesn't skip self here.
                let robot_nearby = objs.any_object_in_radius(
                    pos, self.sens_radius, 1.0, TRACE_ROBOT, None,
                );

                if robot_nearby {
                    if self.prev_state_robots_in_radius == 0 {
                        self.prev_state_robots_in_radius = 1;
                        // SetAnimById(..., activate-clip) + CSound::AddSound —
                        // deferred.
                    }
                } else if self.prev_state_robots_in_radius != 0 {
                    self.prev_state_robots_in_radius = 0;
                    // SetAnimById(..., deactivate-clip) + CSound::AddSound —
                    // deferred.
                }
            }
            return;
        }

        // BEHF_STATIC: empty — subclass isn't on the logic-temp list.
        //
        // BEHF_BREAK / BEHF_ANIM / BEHF_TERRON / BEHF_SPAWNER: bodies
        // need progress bars / effects / sound / robot spawning.
        // Each branch lands with its owning subsystem.
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

/// Per-instance data: world transform expressed as 4 rows of a mat4, laid out
/// as sequential vertex attributes with step_mode=Instance.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceData {
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
    row3: [f32; 4],
    terrain_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    camera_pos: [f32; 4],
    time_ms: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaterialUniform {
    flags: [u32; 4],
    scroll: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowProjUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowTextureUniform {
    u_row: [f32; 4],
    v_row: [f32; 4],
    /// `.x != 0` ⇒ TF_ALPHATEST is set on the surface being rasterized.
    /// The shadow-gen fragment shader uses this to decide whether to
    /// discard based on the diffuse alpha (obj_shadow,
    /// MatrixSkinManager.cpp:188-201).
    alpha_test: [u32; 4],
}

struct MeshBatch {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    /// `(offset, count)` sub-range per VO frame inside `index_buffer`. Frames
    /// where this surface has no geometry get `(_, 0)`. Updated each time the
    /// shared animation state advances to a new VO frame.
    frame_ranges: Vec<(u32, u32)>,
    current_vo_frame: usize,
    instance_buffer: wgpu::Buffer,
    num_instances: u32,
    bind_group: wgpu::BindGroup,
    objects: Vec<ObjectInstance>,
    center: [f32; 2],
    /// Index into `ObjectsRenderer::anims` (one `AnimState` per VO type). All
    /// surfaces belonging to the same type share one state so they advance in
    /// lockstep — mirrors `CVectorObjectAnim` owning a single `m_VOFrame`.
    anim_slot: Option<usize>,
}

/// Per-VO-type animation runtime — ports CVectorObjectAnim's frame scheduler
/// (VectorObject.cpp:1863).
///
/// * `anim` is the current animation index (`m_Anim`, defaulting to 0).
/// * `frame_slot` is the step inside that animation's frame list (`m_Frame`).
/// * `vo_frame` is the resolved SVOKadr index (`m_VOFrame`) — the frame whose
///   surface partition should render this tick.
/// * `looped` comes from the sign of the first frame's time field
///   (VectorObject.hpp:390: `GetAnimLooped`).
struct AnimState {
    animations: Vec<VoAnimation>,
    anim: usize,
    frame_slot: usize,
    vo_frame: usize,
    time_ms: i32,
    time_next: i32,
    looped: bool,
}

impl AnimState {
    fn new(animations: Vec<VoAnimation>) -> Self {
        let mut s = Self {
            animations,
            anim: 0,
            frame_slot: 0,
            vo_frame: 0,
            time_ms: 0,
            time_next: 0,
            looped: true,
        };
        s.first_frame();
        s
    }

    fn current_anim_frame_count(&self) -> usize {
        self.animations
            .get(self.anim)
            .map(|a| a.frames.len())
            .unwrap_or(0)
    }

    fn anim_frame_time(&self, slot: usize) -> i32 {
        self.animations
            .get(self.anim)
            .and_then(|a| a.frames.get(slot))
            .map(|f| f.time_ms.abs())
            .unwrap_or(0)
    }

    fn anim_frame_index(&self, slot: usize) -> usize {
        self.animations
            .get(self.anim)
            .and_then(|a| a.frames.get(slot))
            .map(|f| f.frame_index)
            .unwrap_or(0)
    }

    /// Mirrors `CVectorObjectAnim::FirstFrame` (VectorObject.hpp:552).
    fn first_frame(&mut self) {
        self.frame_slot = 0;
        self.vo_frame = self.anim_frame_index(0);
        self.time_next = self.time_ms + self.anim_frame_time(0);
        // VectorObject.hpp:390 — loop flag follows the sign of the first
        // frame's time. Non-animated objects fall back to looping at rate 0.
        self.looped = self
            .animations
            .get(self.anim)
            .and_then(|a| a.frames.first())
            .map(|f| f.time_ms > 0)
            .unwrap_or(true);
    }

    /// Port of `CVectorObjectAnim::Takt(cms)`. Returns true when `vo_frame`
    /// changed (caller can use this to skip work when nothing moved).
    fn takt(&mut self, cms: i32) -> bool {
        self.time_ms = self.time_ms.saturating_add(cms);
        let fcnt = self.current_anim_frame_count();
        if fcnt == 0 {
            return false;
        }
        let old_frame = self.frame_slot;
        while self.time_ms > self.time_next {
            self.frame_slot += 1;
            if self.looped {
                if self.frame_slot >= fcnt {
                    self.frame_slot = 0;
                }
            } else if self.frame_slot >= fcnt {
                // Non-looped animation pinned to its last pose: advance the
                // deadline by an arbitrary 1s so we don't spin here next tick
                // (matches the C++ `m_TimeNext += 1000` stall).
                self.time_next = self.time_next.saturating_add(1000);
                self.frame_slot = fcnt - 1;
                break;
            }
            self.time_next = self
                .time_next
                .saturating_add(self.anim_frame_time(self.frame_slot));
        }
        if old_frame != self.frame_slot {
            self.vo_frame = self.anim_frame_index(self.frame_slot);
            true
        } else {
            false
        }
    }
}

struct ShadowBatch {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    bind_group: wgpu::BindGroup,
}

struct SurfaceShadowSource {
    indices: Vec<u32>,
    diffuse_view: wgpu::TextureView,
    /// Mirrors `TF_ALPHATEST` on the diffuse texture. `obj_shadow`
    /// (MatrixSkinManager.cpp:185-202) only enables alpha test for
    /// shadow rasterization when this is set; otherwise the shadow
    /// silhouette is opaque `D3DTA_TFACTOR`.
    alpha_test: bool,
}

/// Flattened per-surface view across all VO frames. `concat_indices` holds
/// every frame's indices back-to-back; `frame_ranges[f] = (offset, count)`
/// points at the sub-range for frame `f` inside that buffer.
struct PerSurfaceFrames {
    texture_ref: Option<String>,
    concat_indices: Vec<u32>,
    frame_ranges: Vec<(u32, u32)>,
}

/// Walks every frame's `surfaces` list, matching by `texture_ref`, and packs
/// per-frame triangle ranges into one `concat_indices` buffer per surface
/// slot. Surfaces that first appear in a later frame still register a slot
/// and earlier frames get `(_, 0)` placeholder ranges.
fn build_per_surface_frame_ranges(frames: &[VoFrame]) -> Vec<PerSurfaceFrames> {
    if frames.is_empty() {
        return Vec::new();
    }

    // Stable ordering: first time we see a surface (by texture_ref), assign
    // it a new slot. Mirrors the surfs/texs order the original enforces.
    let mut slot_by_key: Vec<Option<String>> = Vec::new();
    let slot_for = |key: &Option<String>, slots: &mut Vec<Option<String>>| -> usize {
        if let Some(idx) = slots.iter().position(|k| k == key) {
            idx
        } else {
            slots.push(key.clone());
            slots.len() - 1
        }
    };

    // Pass 1: collect the surface slots in a deterministic order.
    for frame in frames {
        for s in &frame.surfaces {
            slot_for(&s.texture_ref, &mut slot_by_key);
        }
    }

    let mut out: Vec<PerSurfaceFrames> = slot_by_key
        .into_iter()
        .map(|key| PerSurfaceFrames {
            texture_ref: key,
            concat_indices: Vec::new(),
            frame_ranges: vec![(0, 0); frames.len()],
        })
        .collect();

    // Pass 2: append frame indices per slot and record offsets.
    for (fi, frame) in frames.iter().enumerate() {
        for s in &frame.surfaces {
            let slot = out
                .iter()
                .position(|ps| ps.texture_ref == s.texture_ref)
                .expect("slot registered in pass 1");
            let ps = &mut out[slot];
            let offset = ps.concat_indices.len() as u32;
            ps.concat_indices.extend_from_slice(&s.indices);
            ps.frame_ranges[fi] = (offset, s.indices.len() as u32);
        }
    }

    // Drop surfaces that had no indices in any frame — they carry a slot but
    // no geometry (shouldn't happen for real data, defensive).
    out.retain(|ps| !ps.concat_indices.is_empty());
    out
}

// Silence unused-import warning when VoSurfaceMesh / VoAnimation / VoFrame
// happen to only be named through field types below.
#[allow(dead_code)]
fn _keep_vo_types_alive(_s: &VoSurfaceMesh, _a: &VoAnimation, _f: &VoFrame) {}

pub struct ObjectsRenderer {
    pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    batches: Vec<MeshBatch>,
    shadow_batches: Vec<ShadowBatch>,
    /// One animation runtime per VO type. `MeshBatch::anim_slot` references
    /// this vector so surfaces of the same type share one `vo_frame`.
    anims: Vec<AnimState>,
    uniform_buffer: wgpu::Buffer,
    shadow_uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    time_ms: f32,
    last_point_light_revision: u64,
}

impl ObjectsRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        stor: &Storage,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        if map.objects.is_empty() {
            return None;
        }

        let strings = stor.get_buf("strings", "String")?;

        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;
        let mut by_type: BTreeMap<u32, Vec<&ObjectInstance>> = BTreeMap::new();
        for obj in &map.objects {
            by_type.entry(obj.type_id).or_default().push(obj);
        }

        let [sr, sg, sb] = unpack_rgb(map.sky_color);
        let fog_color = [sr, sg, sb, 1.0];
        let [ar, ag, ab] = unpack_rgb(map.ambient_color_obj);
        let [lr, lg, lb] = unpack_rgb(map.light_main_color_obj);
        let ambient_color = [ar, ag, ab, 1.0];
        let light_color = [lr, lg, lb, 1.0];
        let light_dir = [
            map.light_main_dir[0],
            map.light_main_dir[1],
            map.light_main_dir[2],
            0.0,
        ];

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Objects UB"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                ambient_color,
                light_color,
                light_dir,
                camera_pos: [0.0, 0.0, 0.0, 1.0],
                time_ms: [0.0, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shadow_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Object Shadows UB"),
            contents: bytemuck::bytes_of(&ShadowProjUniform {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let objects_bgl = create_objects_bgl(device);
        let shadow_bgl = create_shadow_bgl(device);
        let shadow_gen_bgl = create_shadow_gen_bgl(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let pipeline = create_objects_pipeline(device, config, &objects_bgl);
        let shadow_pipeline = create_shadow_pipeline(device, config, &shadow_bgl);
        let shadow_gen_pipeline = create_shadow_texture_pipeline(device, &shadow_gen_bgl);

        let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
        let fallback_tex = create_solid_texture(device, queue, [200, 200, 200, 255]);
        let black_tex = create_solid_texture(device, queue, [0, 0, 0, 255]);
        let transparent_tex = create_solid_texture(device, queue, [0, 0, 0, 0]);

        let mut batches = Vec::new();
        let mut shadow_batches = Vec::new();
        let mut anims: Vec<AnimState> = Vec::new();
        let mut loaded_types = 0usize;
        let mut failed_types = 0usize;
        let mut animated_types = 0usize;

        for (type_id, instances) in &by_type {
            let id_str = if (*type_id as usize) < strings.arrays_count() {
                strings.get_as_wstr(*type_id as usize)
            } else {
                continue;
            };
            let Some(paths) = vector_object::resolve_paths(&id_str) else {
                failed_types += 1;
                continue;
            };
            let Some(vo_bytes) = read_texture(&paths.vo_path) else {
                log::debug!("objects: VO not found: {}", paths.vo_path);
                failed_types += 1;
                continue;
            };
            let mesh = match vector_object::parse_vo(&vo_bytes) {
                Ok(m) => m,
                Err(e) => {
                    log::warn!("objects: parse VO {} failed: {}", paths.vo_path, e);
                    failed_types += 1;
                    continue;
                }
            };

            let object_dir = paths
                .vo_path
                .rsplit_once('/')
                .map(|(dir, _)| format!("{dir}/"));
            let vertices: Vec<Vertex> = mesh
                .vertices
                .iter()
                .map(|v| Vertex {
                    position: v.position,
                    normal: v.normal,
                    uv: v.uv,
                })
                .collect();
            let inst_data: Vec<InstanceData> = instances
                .iter()
                .map(|obj| instance_matrix(obj, cx, cy, map, None))
                .collect();
            let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Objects Inst VB"),
                contents: bytemuck::cast_slice(&inst_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

            // Merge every frame's surface partition into one concat-indices
            // buffer per surface slot plus per-frame `(offset, count)` spans.
            // This lets us flip which triangle range is drawn each tick
            // without reallocating GPU buffers.
            let per_surface = build_per_surface_frame_ranges(&mesh.frames);
            let use_vo_surface_materials = per_surface.len() > 1;
            let mut shadow_surfaces = Vec::new();

            // One animation runtime per VO type. Single-frame meshes keep
            // `anim_slot = None` and render their sole frame with no tick
            // overhead (palms / rocks).
            let anim_slot = if mesh.frames.len() > 1 && !mesh.animations.is_empty() {
                animated_types += 1;
                let slot = anims.len();
                anims.push(AnimState::new(mesh.animations.clone()));
                Some(slot)
            } else {
                None
            };

            for surface in &per_surface {
                let surface_material = if use_vo_surface_materials {
                    surface.texture_ref.as_deref().map(|spec| {
                        vector_object::parse_material_spec_with_prefix(spec, object_dir.as_deref())
                    })
                } else {
                    None
                };
                let material = if use_vo_surface_materials {
                    vector_object::merge_materials(&paths.material, surface_material.as_ref())
                } else {
                    paths.material.clone()
                };

                let diffuse_view = resolve_texture(
                    material.diffuse.as_ref(),
                    device,
                    queue,
                    &mut tex_cache,
                    read_texture,
                )
                .unwrap_or_else(|| fallback_tex.clone());
                let gloss_view = resolve_texture(
                    material.gloss.as_ref(),
                    device,
                    queue,
                    &mut tex_cache,
                    read_texture,
                )
                .unwrap_or_else(|| black_tex.clone());
                let back_view = resolve_texture(
                    material.back.as_ref(),
                    device,
                    queue,
                    &mut tex_cache,
                    read_texture,
                )
                .unwrap_or_else(|| black_tex.clone());
                let mask_view = resolve_texture(
                    material.mask.as_ref(),
                    device,
                    queue,
                    &mut tex_cache,
                    read_texture,
                )
                .unwrap_or_else(|| transparent_tex.clone());

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Objects Mesh VB"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Objects Mesh IB"),
                    contents: bytemuck::cast_slice(&surface.concat_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                // Resolve the full alpha-test flag for this surface's diffuse
                // texture: start from the `?Trans` suffix parsed by
                // `MaterialSpec`, then let the sibling `.txt`'s `AlphaTest`
                // key override (Texture.cpp:113-136).
                let alpha_test = match material.diffuse.as_deref() {
                    Some(path) => vector_object::resolve_alpha_test_with_txt(
                        path,
                        material.alpha_test,
                        read_texture,
                    ),
                    None => false,
                };
                let mat_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Objects Material UB"),
                    contents: bytemuck::bytes_of(&MaterialUniform {
                        flags: [
                            material.gloss.is_some() as u32,
                            material.back.is_some() as u32,
                            material.mask.is_some() as u32,
                            alpha_test as u32,
                        ],
                        scroll: [material.scroll[0], material.scroll[1], 0.0, 0.0],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Objects BG"),
                    layout: &objects_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&diffuse_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&gloss_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&back_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&mask_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: mat_uniform.as_entire_binding(),
                        },
                    ],
                });

                batches.push(MeshBatch {
                    vertex_buffer,
                    index_buffer,
                    frame_ranges: surface.frame_ranges.clone(),
                    current_vo_frame: 0,
                    instance_buffer: instance_buffer.clone(),
                    num_instances: inst_data.len() as u32,
                    bind_group,
                    objects: instances.iter().map(|obj| (*obj).clone()).collect(),
                    center: [cx, cy],
                    anim_slot,
                });
                // Projected shadows still use frame-0 geometry — animated
                // shadow volumes would need per-frame rebuilds which the
                // original also recomputes only for ProjEx dynamic shadows.
                let frame0_range = surface.frame_ranges.first().copied().unwrap_or((0, 0));
                let frame0_indices = if frame0_range.1 > 0 {
                    let start = frame0_range.0 as usize;
                    let end = start + frame0_range.1 as usize;
                    surface.concat_indices[start..end].to_vec()
                } else {
                    Vec::new()
                };
                shadow_surfaces.push(SurfaceShadowSource {
                    indices: frame0_indices,
                    diffuse_view: diffuse_view.clone(),
                    alpha_test,
                });
            }

            if matches!(
                paths.shadow.kind,
                ShadowKind::ProjectedStatic | ShadowKind::ProjectedDynamic
            ) {
                for obj in instances
                    .iter()
                    .filter_map(|obj| obj.shadow.as_ref().map(|shadow| (&**obj, shadow)))
                {
                    if let Some(batch) = build_shadow_batch(
                        device,
                        queue,
                        &shadow_pipeline,
                        &shadow_gen_pipeline,
                        &shadow_bgl,
                        &shadow_gen_bgl,
                        &sampler,
                        &shadow_uniform_buffer,
                        &vertices,
                        &shadow_surfaces,
                        obj.0,
                        obj.1,
                        map,
                        paths.shadow.texture_size.max(32),
                    ) {
                        shadow_batches.push(batch);
                    }
                }
            }

            loaded_types += 1;
        }

        log::info!(
            "objects: {} mesh types loaded, {} skipped, {} total instances drawn, {} projected shadows",
            loaded_types,
            failed_types,
            batches.iter().map(|b| b.num_instances).sum::<u32>(),
            shadow_batches.len(),
        );

        if batches.is_empty() && shadow_batches.is_empty() {
            return None;
        }

        log::info!(
            "objects: {} animated types scheduled for tick advancement",
            animated_types
        );

        Some(Self {
            pipeline,
            shadow_pipeline,
            batches,
            shadow_batches,
            anims,
            uniform_buffer,
            shadow_uniform_buffer,
            fog_color,
            ambient_color,
            light_color,
            light_dir,
            time_ms: 0.0,
            last_point_light_revision: 0,
        })
    }

    pub fn takt(
        &mut self,
        dt_ms: f32,
        queue: &wgpu::Queue,
        map: &GameMap,
        point_lights: &PointLightSystem,
    ) {
        self.time_ms += dt_ms;
        if self.time_ms > 1_000_000.0 {
            self.time_ms -= 1_000_000.0;
        }

        // Advance each VO-type animation by the elapsed ms. Frame flips here
        // only mutate the per-type `vo_frame`; render() pulls that value per
        // batch to choose which index-buffer range to draw.
        let cms = dt_ms.round() as i32;
        if cms > 0 {
            for anim in &mut self.anims {
                anim.takt(cms);
            }
        }
        for batch in &mut self.batches {
            if let Some(slot) = batch.anim_slot {
                batch.current_vo_frame = self.anims[slot].vo_frame;
            }
        }

        let revision = point_lights.revision();
        if revision != self.last_point_light_revision {
            for batch in &mut self.batches {
                let [cx, cy] = batch.center;
                let inst_data: Vec<InstanceData> = batch
                    .objects
                    .iter()
                    .map(|obj| instance_matrix(obj, cx, cy, map, Some(point_lights)))
                    .collect();
                queue.write_buffer(&batch.instance_buffer, 0, bytemuck::cast_slice(&inst_data));
            }
            self.last_point_light_revision = revision;
        }
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
        view_proj: glam::Mat4,
    ) {
        let eye = camera.eye_pos();
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color: self.fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                ambient_color: self.ambient_color,
                light_color: self.light_color,
                light_dir: self.light_dir,
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                time_ms: [self.time_ms, 0.0, 0.0, 0.0],
            }),
        );
        queue.write_buffer(
            &self.shadow_uniform_buffer,
            0,
            bytemuck::bytes_of(&ShadowProjUniform {
                view_proj: view_proj.to_cols_array_2d(),
            }),
        );

        if !self.shadow_batches.is_empty() {
            pass.set_pipeline(&self.shadow_pipeline);
            for batch in &self.shadow_batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..batch.num_indices, 0, 0..1);
            }
        }

        if !self.batches.is_empty() {
            pass.set_pipeline(&self.pipeline);
            for batch in &self.batches {
                let (offset, count) = batch
                    .frame_ranges
                    .get(batch.current_vo_frame)
                    .copied()
                    .unwrap_or((0, 0));
                if count == 0 {
                    continue;
                }
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, batch.instance_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(offset..offset + count, 0, 0..batch.num_instances);
            }
        }
    }
}

fn create_objects_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Objects BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_shadow_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Object Shadows BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn create_shadow_gen_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Object Shadow Texture BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                // Fragment needs it too for the TF_ALPHATEST gate
                // (`u.alpha_test.x` in SHADOW_TEXTURE_SHADER).
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn create_objects_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Objects Shader"),
        source: wgpu::ShaderSource::Wgsl(OBJECTS_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Objects PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Objects Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<InstanceData>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 48,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 64,
                            shader_location: 7,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                },
            ],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_shadow_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Object Shadows Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Object Shadows PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Object Shadows Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ShadowVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            strip_index_format: Some(wgpu::IndexFormat::Uint32),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_shadow_texture_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Object Shadow Texture Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADOW_TEXTURE_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Object Shadow Texture PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Object Shadow Texture Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_shadow_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    _shadow_pipeline: &wgpu::RenderPipeline,
    shadow_gen_pipeline: &wgpu::RenderPipeline,
    shadow_bgl: &wgpu::BindGroupLayout,
    shadow_gen_bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    shadow_uniform_buffer: &wgpu::Buffer,
    vertices: &[Vertex],
    surfaces: &[SurfaceShadowSource],
    obj: &ObjectInstance,
    shadow: &ObjectShadow,
    map: &GameMap,
    texture_size: u32,
) -> Option<ShadowBatch> {
    if shadow.vertices.is_empty() || shadow.indices.len() < 3 || surfaces.is_empty() {
        return None;
    }

    let shadow_texture = build_shadow_texture(
        device,
        queue,
        shadow_gen_pipeline,
        shadow_gen_bgl,
        sampler,
        vertices,
        surfaces,
        obj,
        shadow,
        map,
        texture_size,
    )?;

    let shadow_vertices: Vec<ShadowVertex> = shadow
        .vertices
        .iter()
        .map(|v| ShadowVertex {
            position: [
                v.position[0] - map.world_width() * 0.5,
                v.position[1] - map.world_height() * 0.5,
                v.position[2],
            ],
            uv: v.uv,
        })
        .collect();
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Object Shadow VB"),
        contents: bytemuck::cast_slice(&shadow_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Object Shadow IB"),
        contents: bytemuck::cast_slice(&shadow.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Object Shadow BG"),
        layout: shadow_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadow_texture),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });

    Some(ShadowBatch {
        vertex_buffer,
        index_buffer,
        num_indices: shadow.indices.len() as u32,
        bind_group,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_shadow_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    vertices: &[Vertex],
    surfaces: &[SurfaceShadowSource],
    obj: &ObjectInstance,
    shadow: &ObjectShadow,
    map: &GameMap,
    texture_size: u32,
) -> Option<wgpu::TextureView> {
    let projection = shadow_texture_projection(obj, shadow, map)?;
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Object Shadow Source VB"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Object Shadow Texture"),
        size: wgpu::Extent3d {
            width: texture_size,
            height: texture_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&Default::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Object Shadow Texture Depth"),
        size: wgpu::Extent3d {
            width: texture_size,
            height: texture_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Object Shadow Encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Object Shadow Texture Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));

        for surface in surfaces {
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Object Shadow Source IB"),
                contents: bytemuck::cast_slice(&surface.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            // Per-surface UB so each surface carries its own TF_ALPHATEST
            // bit alongside the shared projection rows.
            let surface_uniform = ShadowTextureUniform {
                alpha_test: [surface.alpha_test as u32, 0, 0, 0],
                ..projection
            };
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Object Shadow Texture UB"),
                contents: bytemuck::bytes_of(&surface_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Object Shadow Texture BG"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&surface.diffuse_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..surface.indices.len() as u32, 0, 0..1);
        }
    }
    queue.submit([encoder.finish()]);

    Some(texture_view)
}

fn shadow_texture_projection(
    obj: &ObjectInstance,
    shadow: &ObjectShadow,
    map: &GameMap,
) -> Option<ShadowTextureUniform> {
    let rotation = object_rotation(obj);
    let inv_rotation = rotation.transpose();

    let mut light = inv_rotation
        * Vec3::new(
            map.light_main_dir[0],
            map.light_main_dir[1],
            map.light_main_dir[2],
        );
    if light.length_squared() < 1e-6 {
        return None;
    }
    light = light.normalize();

    let mut up = inv_rotation * Vec3::Z;
    if up.length_squared() < 1e-6 || up.normalize().dot(light).abs() > 0.98 {
        up = Vec3::Y;
    }
    let right = up.cross(light).normalize_or_zero();
    if right.length_squared() < 1e-6 {
        return None;
    }
    let up = light.cross(right).normalize_or_zero();
    if up.length_squared() < 1e-6 {
        return None;
    }

    let dim_x = shadow.dimensions[0].abs().max(0.001);
    let dim_y = shadow.dimensions[1].abs().max(0.001);
    let campos = Vec3::from_array(shadow.camera_pos);
    let u_row = Vec4::new(
        right.x / dim_x,
        right.y / dim_x,
        right.z / dim_x,
        -right.dot(campos) / dim_x,
    );
    let v_row = Vec4::new(
        up.x / dim_y,
        up.y / dim_y,
        up.z / dim_y,
        -up.dot(campos) / dim_y,
    );
    Some(ShadowTextureUniform {
        u_row: u_row.to_array(),
        v_row: v_row.to_array(),
        alpha_test: [0, 0, 0, 0],
    })
}

/// Build the same static-object transform order used by the original renderer:
/// Rx * Ry * Rz, then uniform scale, then translation into centered render space.
fn instance_matrix(
    obj: &ObjectInstance,
    cx: f32,
    cy: f32,
    map: &GameMap,
    point_lights: Option<&PointLightSystem>,
) -> InstanceData {
    let s = obj.scale.max(0.0001);
    let [terrain_r, terrain_g, terrain_b] =
        unpack_rgb(map.static_object_color_with_lighting(obj.x, obj.y, point_lights));
    let m = object_rotation(obj) * s;

    InstanceData {
        row0: [m.x_axis.x, m.y_axis.x, m.z_axis.x, obj.x - cx],
        row1: [m.x_axis.y, m.y_axis.y, m.z_axis.y, obj.y - cy],
        row2: [m.x_axis.z, m.y_axis.z, m.z_axis.z, obj.z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [terrain_r, terrain_g, terrain_b, 1.0],
    }
}

fn object_rotation(obj: &ObjectInstance) -> Mat3 {
    let (sx, cxr) = obj.angle_x.sin_cos();
    let (sy, cyr) = obj.angle_y.sin_cos();
    let (sz, cz) = obj.angle_z.sin_cos();
    let rx = Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, cxr, sx, 0.0, -sx, cxr]);
    let ry = Mat3::from_cols_array(&[cyr, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cyr]);
    let rz = Mat3::from_cols_array(&[cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0]);
    rx * ry * rz
}

fn resolve_texture(
    tex_path: Option<&String>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<wgpu::TextureView> {
    let path = tex_path?;
    if let Some(cached) = cache.get(path) {
        return Some(cached.clone());
    }
    let data = read_texture(path)?;
    let rgba = decode_texture_bytes(&data)?;
    let view = create_texture_from_rgba(device, queue, &rgba);
    cache.insert(path.clone(), view.clone());
    Some(view)
}

const OBJECTS_SHADER: &str = include_str!("../../shaders/object.wgsl");
const SHADOW_SHADER: &str = include_str!("../../shaders/object_shadow.wgsl");
const SHADOW_TEXTURE_SHADER: &str = include_str!("../../shaders/object_shadow_texture.wgsl");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_game::map::ObjectInstance;

    fn inst() -> ObjectInstance {
        ObjectInstance {
            x: 0.0, y: 0.0, z: 0.0,
            angle_z: 0.0, angle_x: 0.0, angle_y: 0.0,
            scale: 1.0, type_id: 0, shadow: None,
        }
    }

    #[test]
    fn empty_ids_row_leaves_static() {
        let mut o = MapObject::from_instance(&inst());
        let add_lt = o.apply_ids_row("", &mut Rnd::new(1), || {});
        assert_eq!(o.beh_flag, BehFlag::Static);
        assert!(!add_lt);
    }

    #[test]
    fn burn_keyword_sets_burn_but_does_not_addlt() {
        // BEHF_BURN is enrolled into the logic list lazily on damage
        // (MatrixObject.cpp:107-140), not at init.
        let row = "path*vo*tex******Burn,Tex,Burn01";
        let mut o = MapObject::from_instance(&inst());
        let add_lt = o.apply_ids_row(row, &mut Rnd::new(1), || {});
        assert_eq!(o.beh_flag, BehFlag::Burn);
        assert!(!add_lt);
    }

    #[test]
    fn break_variant_terron_is_distinguished() {
        let break_plain  = "path*vo*tex******Break,X,5000,Normal";
        let break_terron = "path*vo*tex******Break,X,5000,Terron";
        let mut a = MapObject::from_instance(&inst());
        let mut b = MapObject::from_instance(&inst());
        assert!(a.apply_ids_row(break_plain, &mut Rnd::new(1), || {}));
        assert!(b.apply_ids_row(break_terron, &mut Rnd::new(1), || {}));
        assert_eq!(a.beh_flag, BehFlag::Break);
        assert_eq!(b.beh_flag, BehFlag::Terron);
    }

    #[test]
    fn anim_vs_animp_split_on_fifth_char() {
        let anim_plain = "path*vo*tex******Anim";
        let animp      = "path*vo*tex******AnimP";
        let mut a = MapObject::from_instance(&inst());
        let mut p = MapObject::from_instance(&inst());
        let a_lt = a.apply_ids_row(anim_plain, &mut Rnd::new(1), || {});
        let p_lt = p.apply_ids_row(animp, &mut Rnd::new(1), || {});
        assert_eq!(a.beh_flag, BehFlag::Anim);
        assert!(a_lt, "Anim opts into AddLT");
        assert_eq!(p.beh_flag, BehFlag::Portret);
        assert!(!p_lt, "Portret is Takt-driven, not logic-temp");
    }

    #[test]
    fn sens_and_spawn_seed_ablaze_ttl_as_timer() {
        // The C++ repurposes `m_ObjectStateTTLAblaze` as the "next-tick"
        // deadline in these branches (MatrixObject.cpp:1082, :1090).
        let mut s = MapObject::from_instance(&inst());
        assert!(s.apply_ids_row("path*vo*tex******Sens,120.5", &mut Rnd::new(1), || {}));
        assert_eq!(s.beh_flag, BehFlag::Sens);
        assert_eq!(s.ablaze_ttl, 101);

        let mut sp = MapObject::from_instance(&inst());
        assert!(sp.apply_ids_row("path*vo*tex******Spawn,80.0", &mut Rnd::new(1), || {}));
        assert_eq!(sp.beh_flag, BehFlag::Spawner);
        assert_eq!(sp.ablaze_ttl, 101);
    }

    #[test]
    fn plus_prefix_marks_special_and_fires_callback() {
        let row = "path*vo*tex******+Break,X,5000,Normal";
        let mut o = MapObject::from_instance(&inst());
        let mut bumps = 0;
        let add_lt = o.apply_ids_row(row, &mut Rnd::new(1), || { bumps += 1; });
        assert!(add_lt);
        assert_eq!(o.beh_flag, BehFlag::Break);
        assert_ne!(o.object_state & OBJECT_STATE_SPECIAL, 0);
        assert_eq!(bumps, 1);
    }

    #[test]
    fn invlogic_sets_trace_invisible_when_one() {
        // 11 fields (10 stars) → index 10 = OTP_INVLOGIC = "1".
        let row = "path*vo*tex********1";
        let mut o = MapObject::from_instance(&inst());
        o.apply_ids_row(row, &mut Rnd::new(1), || {});
        assert_ne!(o.object_state & OBJECT_STATE_TRACE_INVISIBLE, 0);
    }

    #[test]
    fn portret_photo_toggle_cycles_through_long_and_short_timers() {
        // Portret object: apply_ids_row seeds photo_time with Rnd(1000,2000)
        // (MatrixObject.cpp:1066). Each `takt(cms)` counts it down; on
        // expiry (MatrixObject.cpp:675-689), the reseed reads the
        // *pre-XOR* photo value, then flips it. So photo=0 → short
        // (100-200ms) "looking" interval, photo=1 → long (3000-5000ms)
        // "displayed" interval.
        let mut rng = Rnd::new(42);
        let mut o = MapObject::from_instance(&inst());
        assert!(!o.apply_ids_row("path*vo*tex******AnimP", &mut rng, || {}));
        assert_eq!(o.beh_flag, BehFlag::Portret);
        assert!(o.photo_time >= 1000 && o.photo_time <= 2000);
        assert_eq!(o.photo, 0);

        // First expiry: photo was 0 → short reseed, then flip to 1.
        let seed_time = o.photo_time + 1;
        let mut empty_arena = crate::matrix_game::map_static::Objects::new();
        o.takt(seed_time, &mut rng, &mut empty_arena);
        assert_eq!(o.photo, 1, "photo flips on timer expiry");
        assert!(
            (100..=200).contains(&o.photo_time),
            "post-first-toggle timer must land in 100..=200, got {}",
            o.photo_time,
        );
        assert_ne!(o.rchange & MR_GRAPH, 0, "MR_Graph flagged for re-skin");

        // Second expiry: photo was 1 → long reseed, then flip to 0.
        o.rchange &= !MR_GRAPH;
        let next = o.photo_time + 1;
        o.takt(next, &mut rng, &mut empty_arena);
        assert_eq!(o.photo, 0);
        assert!(
            (3000..=5000).contains(&o.photo_time),
            "post-second-toggle timer must land in 3000..=5000, got {}",
            o.photo_time,
        );
    }

    #[test]
    fn sens_logic_takt_transitions_on_nearby_robot() {
        use crate::matrix_game::map_static::{MapStatic, ObjectType};
        use crate::matrix_game::rnd::Rnd;
        use crate::matrix_game::world::World;

        // Build a world with one SENS mapobject at the origin and a
        // "robot" (stub MapStatic with ObjectType::RobotAi) 30 units
        // away — inside the 50-unit sens radius.
        let mut world = World::with_seed(1);
        let mut sensor = MapObject::from_instance(&inst());
        sensor.apply_ids_row("path*vo*tex******Sens,50.0", &mut Rnd::new(1), || {});
        assert_eq!(sensor.beh_flag, BehFlag::Sens);
        assert_eq!(sensor.sens_radius, 50.0);
        assert_eq!(sensor.prev_state_robots_in_radius, -1);
        let sensor_id = world.objects.spawn(Box::new(sensor));
        world.objects.add_lt(sensor_id);

        let mut robot = MapObject::from_instance(&inst());
        robot.core_mut().obj_type = ObjectType::RobotAi;
        robot.core_mut().geo_center = glam::Vec3::new(30.0, 0.0, 0.0);
        let robot_id = world.objects.spawn(Box::new(robot));

        // Sensor's initial AblazeTTL is 101ms (seeded in apply_ids_row).
        // Drive one logic takt of 210ms to drain the timer (fires the
        // find_objects call).
        world.takt(210);

        // Inspect sensor state via downcast — MapObject is the concrete
        // type behind the trait object.
        let obj = world.objects.get(sensor_id).expect("sensor still live");
        let mapobj = unsafe {
            &*(obj as *const dyn MapStatic as *const MapObject)
        };
        assert_eq!(mapobj.prev_state_robots_in_radius, 1,
            "sensor detected the robot and transitioned to state=1");

        // Move robot outside the radius, takt again.
        let robot_slot_obj = world.objects.get_mut(robot_id).unwrap();
        robot_slot_obj.core_mut().geo_center = glam::Vec3::new(200.0, 0.0, 0.0);

        world.takt(210);

        let obj = world.objects.get(sensor_id).unwrap();
        let mapobj = unsafe {
            &*(obj as *const dyn MapStatic as *const MapObject)
        };
        assert_eq!(mapobj.prev_state_robots_in_radius, 0,
            "sensor falls back to idle after robot leaves the radius");
    }

    #[test]
    fn unknown_behaviour_keyword_falls_back_to_static() {
        let row = "path*vo*tex******Gibberish,foo,bar";
        let mut o = MapObject::from_instance(&inst());
        let add_lt = o.apply_ids_row(row, &mut Rnd::new(1), || {});
        assert_eq!(o.beh_flag, BehFlag::Static);
        assert!(!add_lt);
    }
}