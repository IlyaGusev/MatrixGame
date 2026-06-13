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

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{unpack_rgb, FOG_END, FOG_START};
use crate::matrix_game::common::{
    OBJECT_ABLAZE_BURNED_AT_MS, OTP_BEHAVIOUR, OTP_BIAS, OTP_INVLOGIC, PLAYER_SIDE,
};
use crate::matrix_game::effects::point_light::PointLightSystem;
use crate::matrix_game::effects::weapon::{
    is_fire_weapon, Weapon, WEAPON_ABLAZE, WEAPON_FLAMETHROWER, WEAPON_PLASMA,
};
use crate::matrix_game::logic::Rnd;
use crate::matrix_game::map::{GameMap, ObjectInstance, ObjectShadow};
use crate::matrix_game::map_static::{
    MapStatic, ObjectCore, ObjectId, ObjectType, Objects, MR_ALL, MR_GRAPH, MR_MATRIX,
    MR_SHADOW_PROJ_GEOM, MR_SHADOW_PROJ_TEX, MR_SHADOW_STENCIL, OBJECT_STATE_ABLAZE,
    OBJECT_STATE_BURNED, OBJECT_STATE_SHADOW_SPECIAL, OBJECT_STATE_SPECIAL,
    OBJECT_STATE_TRACE_INVISIBLE,
};
use crate::matrix_lib::base::storage::Storage;
use crate::matrix_lib::base::wstr;
use crate::matrix_lib::three_g::texture::{
    create_solid_texture, create_texture_from_rgba, decode_texture_bytes,
};
use crate::matrix_lib::three_g::texture::{
    has_trans_suffix, parse_scroll, resolve_texture_name, MaterialSpec,
};
use crate::matrix_lib::three_g::vector_object::{self, VoAnimation, VoFrame, VoSurfaceMesh};

// ── Object id-string parsing ─────────────────────────────────────────────
//
// Port of `CMatrixMapObject` id-string decoding (MatrixObject.cpp:429-472)
// using the OTP_* field layout from Common.hpp:176-191. The VO/material/
// shadow bundle lives with the map-object because the id-string is a map
// data field, not a VO library concept.

/// Resolve an object Id string (from `strings/String`, '*'-delimited) into a
/// VO file path and texture path. Returns `None` if empty.
///
/// Id string layout (MatrixObject.cpp:429-472, Common.hpp:176-191):
///   [0] OTP_PATH (e.g. `Matrix\Obj\palm\`)
///   [1] OTP_VO   (e.g. `palm00`)
///   [2] OTP_TEXTURE (e.g. `palm00?Trans`)
///   ... more fields
pub struct ResolvedObjectPaths {
    pub vo_path: String,
    pub material: MaterialSpec,
    pub shadow: ShadowSpec,
}

#[derive(Clone, Debug)]
pub struct ShadowSpec {
    pub kind: ShadowKind,
    pub texture_size: u32,
    pub cache_tag: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShadowKind {
    None,
    Stencil,
    ProjectedStatic,
    ProjectedDynamic,
}

pub fn resolve_paths(id_string: &str) -> Option<ResolvedObjectPaths> {
    let parts: Vec<&str> = id_string.split('*').collect();
    if parts.len() < 2 {
        return None;
    }
    let path = parts[0].replace('\\', "/");
    let vo_name = parts[1];
    if vo_name.is_empty() {
        return None;
    }
    let vo_path = format!("{}{}.vo", path, vo_name);

    let material = MaterialSpec {
        diffuse: parts
            .get(2)
            .and_then(|t| resolve_texture_name(t, Some(&path))),
        gloss: parts
            .get(3)
            .and_then(|t| resolve_texture_name(t, Some(&path))),
        back: parts
            .get(4)
            .and_then(|t| resolve_texture_name(t, Some(&path))),
        mask: parts
            .get(5)
            .and_then(|t| resolve_texture_name(t, Some(&path))),
        scroll: parts.get(6).map(|t| parse_scroll(t)).unwrap_or([0.0, 0.0]),
        alpha_test: parts.get(2).is_some_and(|t| has_trans_suffix(t)),
    };
    let shadow = parts
        .get(7)
        .map(|t| parse_shadow_spec(t))
        .unwrap_or(ShadowSpec {
            kind: ShadowKind::None,
            texture_size: 128,
            cache_tag: None,
        });

    Some(ResolvedObjectPaths {
        vo_path,
        material,
        shadow,
    })
}

fn parse_shadow_spec(raw: &str) -> ShadowSpec {
    let spec = raw.trim();
    if spec.is_empty() {
        return ShadowSpec {
            kind: ShadowKind::None,
            texture_size: 128,
            cache_tag: None,
        };
    }
    let mut parts = spec.split(',');
    let kind = match parts.next().unwrap_or("").trim() {
        "Stencil" => ShadowKind::Stencil,
        "Proj" => ShadowKind::ProjectedStatic,
        "ProjEx" => ShadowKind::ProjectedDynamic,
        _ => ShadowKind::None,
    };
    let texture_size = parts
        .next()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(128);
    let cache_tag = parts.next().and_then(|v| v.trim().parse::<u32>().ok());
    ShadowSpec {
        kind,
        texture_size,
        cache_tag,
    }
}

// ── Global object-Ids table (g_MatrixMap->m_Ids) ─────────────────────────
//
// `CMatrixMapObject::Init` re-reads `g_MatrixMap->IdsGet(m_Type)` at
// runtime (MatrixObject.cpp:985-1018) when a BREAK death / terron corpse
// swaps the object to its replacement type. The damage path only carries
// `&mut Objects`, so — like the C++ global map pointer — the rows are
// published process-globally at map load (`GameMap::from_cmap_bytes`).

static GLOBAL_IDS: std::sync::OnceLock<std::sync::RwLock<std::sync::Arc<Vec<String>>>> =
    std::sync::OnceLock::new();

fn global_ids_slot() -> &'static std::sync::RwLock<std::sync::Arc<Vec<String>>> {
    GLOBAL_IDS.get_or_init(|| std::sync::RwLock::new(std::sync::Arc::new(Vec::new())))
}

pub fn set_global_ids(rows: Vec<String>) {
    *global_ids_slot().write().unwrap() = std::sync::Arc::new(rows);
}

/// `g_MatrixMap->IdsGet(type)`. Out-of-range / negative types yield an
/// empty row (parses to BEHF_STATIC with no textures).
pub fn ids_get(type_id: i32) -> String {
    if type_id < 0 {
        return String::new();
    }
    global_ids_slot()
        .read()
        .unwrap()
        .get(type_id as usize)
        .cloned()
        .unwrap_or_default()
}

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
    Static = 0,
    Burn = 1,
    Break = 2,
    Anim = 3,
    Sens = 4,
    Spawner = 5,
    Terron = 6,
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

    /// `m_NextTime` (BEHF_BURN's first union field, MatrixObject.hpp:64).
    /// Game-time ms when the next ablaze fire-tick should fire. Set by
    /// `damage` when the object is kindled; consumed + advanced inside
    /// the IsAblaze branch of `logic_takt`.
    pub next_time: i32,
    /// `m_BurnTimeTotal` (MatrixObject.hpp:65). Running total of ms the
    /// object has been on fire. At 5000ms the OBJECT_STATE_BURNED bit is
    /// set and the skin swaps to the burnt variant.
    pub burn_time_total: i32,

    /// Own arena id, captured on the first `damage` call (the C++ uses
    /// `this`; the takt drivers don't pass an id). Needed by the terron
    /// death sequence to own its WEAPON_BIGBOOM blasts.
    self_id: Option<ObjectId>,

    /// `m_BreakHitPoint` (BEHF_BREAK/ANIM/TERRON union, MatrixObject.hpp:73).
    /// Current hit points — decremented by [`MapStatic::damage`] via the
    /// damage-table lookup; object transitions state / breaks when this
    /// reaches 0.
    pub break_hit_point: i32,
    /// `m_BreakHitPointMax` (MatrixObject.hpp:76). Starting hit points —
    /// stored for ratio display on the progress bar + full-heal on anim
    /// state transitions that specify `hp=0` (treated as "max",
    /// MatrixObject.cpp:311).
    pub break_hit_point_max: i32,
    /// `m_AnimState` (MatrixObject.hpp:74). Current BEHF_ANIM state id;
    /// indexes into the `Anim,<states>` table in the Ids behaviour
    /// string. -1 for non-BEHF_ANIM objects.
    pub anim_state: i32,
    /// Cached BEHAVIOUR field (post-`+` strip) from the Ids row. The
    /// C++ re-reads `g_MatrixMap->IdsGet(m_Type)` whenever `Damage` /
    /// `ApplyAnimState` needs the state table (MatrixObject.cpp:269);
    /// the world-scope Ids table isn't reachable from `damage` here,
    /// so `apply_ids_row` keeps a copy.
    pub behaviour: String,

    /// VO path of an explicitly-loaded graph — `InitAsBaseRuins`' direct
    /// `LoadObject(namev, …)` (MatrixObject.cpp:1137). `None` for normal
    /// objects whose VO derives from `type_id`. The render side keys its
    /// preloaded ruin batches off this path.
    pub ruin_graph: Option<String>,
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
            rchange: MR_ALL, // m_RChange(0xffffffff)
            object_state: 0,
            ablaze_ttl: 0,
            shorted_ttl: 0,
            angle_z: inst.angle_z,
            angle_x: inst.angle_x,
            angle_y: inst.angle_y,
            scale: inst.scale.max(0.0001),
            tex_bias: -1.0, // MatrixObject.cpp:43
            type_id: inst.type_id as i32,
            uid: -1,                   // MatrixObject.cpp:61
            beh_flag: BehFlag::Static, // MatrixObject.cpp:56
            self_id: None,
            photo: 0,
            photo_time: 0,
            prev_state_robots_in_radius: 0,
            sens_radius: 0.0,
            next_time: 0,
            burn_time_total: 0,
            break_hit_point: 0,
            break_hit_point_max: 0,
            anim_state: -1,
            behaviour: String::new(),
            ruin_graph: None,
        }
    }

    /// Port of `CMatrixMapObject::InitAsBaseRuins`
    /// (MatrixObject.cpp:1110-1141): a ruin spawned where a building
    /// finished its DIP explosion sequence. `angle` is the building's
    /// 0..3 quarter-turn (`m_AngleZ = angle * GRAD2RAD(90)`, :1120); `z`
    /// is `GetZ(pos)` (:1119). The C++ also picks SHADOW_PROJ_DYNAMIC /
    /// SHADOW_OFF and `m_BiasBuildings` — both render-side concerns the
    /// port resolves in the ruin batch, so they carry no field here.
    pub fn init_as_base_ruins(pos: glam::Vec2, z: f32, angle: i32, namev: &str) -> Self {
        let inst = ObjectInstance {
            x: pos.x,
            y: pos.y,
            z,
            angle_z: angle as f32 * std::f32::consts::FRAC_PI_2,
            angle_x: 0.0,
            angle_y: 0.0,
            scale: 1.0,
            type_id: 0,
            shadow: None,
        };
        let mut o = Self::from_instance(&inst);
        o.type_id = -1; // :1133
        o.ruin_graph = Some(namev.to_string());
        // :1135 — `m_RChange &= ~MR_Graph`: the graph is bound eagerly
        // (render-side ruin batch), not via the dirty-bit pipeline.
        o.rchange &= !MR_GRAPH;
        o.r_need(MR_MATRIX); // RNeed(MR_Matrix), :1140
        o
    }

    /// Runtime re-Init — port of `CMatrixMapObject::Init`
    /// (MatrixObject.cpp:929-1096). Swaps this object to type `ids`:
    /// BREAK death (:233) and the terron corpse (:1261) call this to
    /// become their broken-variant object. The graph / shadow releases
    /// (:960-979) are render-side; the MR_* dirty bits + the `type_id`
    /// change tell the instance sync to rebind the mesh.
    pub fn init(&mut self, ids: i32, objs: &mut Objects) {
        if self.type_id == ids {
            return; // :931
        }
        // :933-956 — MMFLAG_TERRON_ONMAP, progress-bar deletes and the
        // spawner-core release: all tied to unported subsystems.
        self.uid = -1; // :958
        self.type_id = ids;
        // :982 — RChange(MR_Graph|MR_ShadowStencil|MR_ShadowProjGeom|MR_ShadowProjTex)
        self.rchange |= MR_GRAPH | MR_SHADOW_STENCIL | MR_SHADOW_PROJ_GEOM | MR_SHADOW_PROJ_TEX;

        let row = ids_get(ids);
        self.tex_bias = wstr::double_par(&row, OTP_BIAS, "*") as f32; // :985
        self.beh_flag = BehFlag::Static; // :988

        // OTP_SHADOW parse (:991-1003) — shadow type resolves render-side
        // from the same row; no logic-side field.

        // :1005-1093 — shared with load-time init. A '+' prefix on a
        // runtime row would need m_BeforeWinCount plumbed out of
        // MapLogic; replacement rows never carry one, so the callback is
        // a no-op. The rng only seeds Portret's photo timer.
        let mut rng = Rnd::new((ids ^ 0x5bd1).max(1));
        let add_lt = self.apply_ids_row(&row, &mut rng, || {});
        if add_lt {
            if let Some(id) = self.self_id {
                objs.add_lt(id); // AddLT() inside the behaviour branches
            }
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
        self.behaviour = beh.clone();

        if wstr::compare_first(&beh, "Burn") {
            self.beh_flag = BehFlag::Burn;
            // `m_NextTime = 0; m_BurnTimeTotal = 0` (MatrixObject.cpp:
            // 1031-1032) — matters on a runtime re-`init` where the old
            // behaviour left them non-zero. The `Tex`-variant burn skin
            // load (:1035-1042) depends on the skin manager, unported.
            self.next_time = 0;
            self.burn_time_total = 0;
            false // BEHF_BURN is not AddLT'd — it enrolls on damage.
        } else if wstr::compare_first(&beh, "Break") {
            // "Break,<something>,<hp>,Terron?"  (MatrixObject.cpp:1043-1058)
            // Field 2 is the starting hit-point count.
            let hp = wstr::int_par(&beh, 2, ",");
            self.break_hit_point = hp;
            self.break_hit_point_max = hp;
            let kind = wstr::str_par(&beh, 3, ",");
            if kind == "Terron" {
                self.beh_flag = BehFlag::Terron;
            } else {
                self.beh_flag = BehFlag::Break;
            }
            true
        } else if wstr::compare_first(&beh, "Anim") {
            // "Anim,<states>" or Portret "Anim,P..."
            // (MatrixObject.cpp:1060-1075). The C++ checks `temp[5]`
            // — 0-based (CWStr.hpp:195), i.e. the char right after
            // the `,` that follows "Anim". A literal "AnimP" has no
            // index 5 and stays BEHF_ANIM.
            if beh.chars().nth(5) == Some('P') {
                self.beh_flag = BehFlag::Portret;
                // MatrixObject.cpp:1066-1067 —
                //   m_PhotoTime = g_MatrixMap->Rnd(1000,2000);
                //   m_Photo = 0;
                self.photo_time = rng.range(1000, 2000);
                self.photo = 0;
                false // BEHF_PORTRET doesn't AddLT (Takt handles it).
            } else {
                self.beh_flag = BehFlag::Anim;
                // MatrixObject.cpp:1072 — ApplyAnimState(0). The state
                // table lives inside the BEHAVIOUR field's 1st ","-sub:
                // "Anim,<states>". We parse it now so `break_hit_point`
                // is correctly seeded at spawn time.
                self.apply_anim_state(0, &beh);
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

    /// Port of `CMatrixMapObject::ApplyAnimState` (MatrixObject.cpp:295-318).
    /// `behaviour_field` is the BEHAVIOUR row (`"Anim,<states>"`); the
    /// state table lives in the 2nd `,`-separated sub, formatted as
    /// `<state_id>:<anim_id>:<hp>:<next_state>:...` entries joined by
    /// `#`. The C++ also re-reads the Ids table every call; we avoid
    /// that by threading the field explicitly.
    ///
    /// The `SetAnimById` call on `m_Graph` is deferred — per-instance
    /// anim state isn't ported. We track `anim_state` + `break_hit_point`
    /// faithfully so state transitions + damage continue to work.
    pub fn apply_anim_state(&mut self, state_id: i32, behaviour_field: &str) {
        self.anim_state = state_id;

        let state_table = wstr::str_par(behaviour_field, 1, ",");
        let cnt = wstr::count_par(state_table, "#");
        for i in 0..cnt {
            let entry = wstr::str_par(state_table, i, "#");
            if wstr::int_par(entry, 0, ":") == state_id {
                // C++: m_Graph->SetAnimById(...); deferred.
                let hp = wstr::int_par(entry, 2, ":");
                // MatrixObject.cpp:311-314 — a 0 hp value is treated as
                // "invincible" (sentinel 2_000_000_000). Preserves the
                // same magic constant.
                self.break_hit_point = if hp == 0 { 2_000_000_000 } else { hp };
                break;
            }
        }
    }

    /// Terron death countdown — port of the OBJECT_STATE_TERRON_EXPL
    /// branch of `CMatrixMapObject::LogicTakt` (MatrixObject.cpp:1235-1330).
    fn terron_expl_takt(
        &mut self,
        ms: i32,
        rng: &mut crate::matrix_game::logic::Rnd,
        objs: &mut crate::matrix_game::map_static::Objects,
    ) {
        use crate::matrix_game::map_static::{
            SpecialDeathKind, OBJECT_STATE_TERRON_EXPL1, OBJECT_STATE_TERRON_EXPL2,
        };

        self.ablaze_ttl -= ms;
        if self.ablaze_ttl < 0 {
            // MatrixObject.cpp:1244-1258 — special-win bookkeeping.
            // Unlike BREAK, SS_JUST_WIN fires regardless of the
            // remaining win counter (the count check is commented out
            // in the original).
            if self.object_state & OBJECT_STATE_SPECIAL != 0 {
                self.object_state &= !OBJECT_STATE_SPECIAL;
                objs.inc_side_stat(PLAYER_SIDE, |s| s.building_kill += 1);
                objs.pending_special_deaths
                    .push(SpecialDeathKind::Terron);
            }
            // `Init(temp.GetIntPar(1,L","))` — re-init as the corpse
            // type (MatrixObject.cpp:1260-1262).
            let new_type = wstr::int_par(&self.behaviour, 1, ",");
            self.init(new_type, objs);
        } else if self.ablaze_ttl < 100 && self.object_state & OBJECT_STATE_TERRON_EXPL2 == 0 {
            self.object_state |= OBJECT_STATE_TERRON_EXPL2;
            self.terron_bigboom(rng, objs);
        } else if self.ablaze_ttl < 1000 && self.object_state & OBJECT_STATE_TERRON_EXPL1 == 0 {
            self.object_state |= OBJECT_STATE_TERRON_EXPL1;
            self.terron_bigboom(rng, objs);
        } else {
            // Rolling pops every BUILDING_EXPLOSION_PERIOD (10ms),
            // 4-attempt surface pick around the geo-center, 4% Boom2
            // (MatrixObject.cpp:1292-1330). `next_time` (zeroed when
            // the death started) runs as a countdown accumulator.
            use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd};
            let origin = self.core.geo_center;
            let radius = self.core.radius.max(3.0);
            let mut vrng = crate::matrix_game::logic::Rnd::new(
                ((self.ablaze_ttl) ^ ((origin.x + origin.y) as i32)).max(1),
            );
            self.next_time -= ms;
            while self.next_time <= 0 {
                self.next_time += 10; // BUILDING_EXPLOSION_PERIOD
                let mut found = None;
                for _ in 0..4 {
                    let pos = origin
                        + glam::Vec3::new(
                            fsrnd(&mut vrng, radius),
                            fsrnd(&mut vrng, radius),
                            frnd(&mut vrng, 2.0 * radius),
                        );
                    let dir = (origin - pos).normalize_or_zero();
                    if dir == glam::Vec3::ZERO {
                        continue;
                    }
                    if let Some(t) = crate::matrix_game::map_trace::pick_sphere(
                        pos,
                        dir,
                        self.core.geo_center,
                        self.core.radius,
                    ) {
                        found = Some(pos + dir * (t + 2.0));
                        break;
                    }
                }
                let Some(pos) = found else { continue };
                let props = if frnd(&mut vrng, 1.0) < 0.04 {
                    &crate::matrix_game::effects::explosion::EXPLOSION_BUILDING_BOOM2
                } else {
                    &crate::matrix_game::effects::explosion::EXPLOSION_BUILDING_BOOM
                };
                objs.pending_explosions
                    .push(crate::matrix_game::map_static::ExplosionSpawn {
                        pos,
                        props,
                        fire: false,
                    });
            }
        }
    }

    /// One real WEAPON_BIGBOOM blast at the terron's geo-center
    /// (MatrixObject.cpp:1267-1273 / :1281-1287) — same create/fire/
    /// takt/release shape as the dying base's final blast.
    fn terron_bigboom(
        &mut self,
        rng: &mut crate::matrix_game::logic::Rnd,
        objs: &mut crate::matrix_game::map_static::Objects,
    ) {
        let (Some(self_id), Some(map)) = (self.self_id, crate::matrix_game::map::current_map())
        else {
            return;
        };
        use crate::matrix_game::effects::weapon::{
            weapon_takt, WeaponEffect, WeaponHandler, WEAPON_BIGBOOM,
        };
        let mut w = WeaponEffect::new(WEAPON_BIGBOOM, 0, WeaponHandler::None);
        w.set_owner(self_id, 0);
        w.modify(self.core.geo_center, glam::Vec3::Z, glam::Vec3::ZERO);
        let wid = objs.weapons.create(w);
        if let Some(we) = objs.weapons.get_mut(wid) {
            we.fire_begin(glam::Vec3::ZERO, Some(self_id));
        }
        weapon_takt(objs, wid, 1.0, map, rng);
        if let Some(we) = objs.weapons.get_mut(wid) {
            we.fire_end();
        }
        objs.weapons.release(wid);
    }
}

impl MapStatic for MapObject {
    fn core(&self) -> &ObjectCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut ObjectCore {
        &mut self.core
    }
    fn rchange(&self) -> u32 {
        self.rchange
    }
    fn rchange_set(&mut self, b: u32) {
        self.rchange |= b;
    }
    fn rchange_clear(&mut self, b: u32) {
        self.rchange &= !b;
    }
    fn object_state(&self) -> u32 {
        self.object_state
    }
    fn object_state_set(&mut self, b: u32) {
        self.object_state |= b;
    }
    fn object_state_clear(&mut self, b: u32) {
        self.object_state &= !b;
    }
    fn ablaze_ttl(&self) -> i32 {
        self.ablaze_ttl
    }
    fn set_ablaze_ttl(&mut self, t: i32) {
        self.ablaze_ttl = t;
    }
    fn shorted_ttl(&self) -> i32 {
        self.shorted_ttl
    }
    fn set_shorted_ttl(&mut self, t: i32) {
        self.shorted_ttl = t;
    }

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
            // C++ row-major `mx * my * mz` (MatrixObject.cpp:392) ==
            // column-major `rz * ry * rx`; scale is applied to all nine
            // rotation elements afterwards (uniform, so order-free).
            let rot = rz * ry * rx * self.scale;

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
        if need & MR_GRAPH != 0 {
            self.rchange &= !MR_GRAPH;
        }
        if need & MR_SHADOW_PROJ_GEOM != 0 {
            self.rchange &= !MR_SHADOW_PROJ_GEOM;
        }
        if need & MR_SHADOW_PROJ_TEX != 0 {
            self.rchange &= !MR_SHADOW_PROJ_TEX;
        }
    }

    /// Port of `CMatrixMapObject::Takt` (MatrixObject.cpp:671-731). The
    /// graphic takt.
    fn takt(
        &mut self,
        cms: i32,
        rng: &mut Rnd,
        _objs: &mut crate::matrix_game::map_static::Objects,
    ) {
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

    fn damage(
        &mut self,
        weap: Weapon,
        _pos: glam::Vec3,
        _dir: glam::Vec3,
        attacker_side: i32,
        _attacker: Option<ObjectId>,
        self_id: ObjectId,
        objs: &mut Objects,
    ) -> bool {
        // Port of `CMatrixMapObject::Damage` (MatrixObject.cpp:101-293).
        //
        // Only BEHF_BURN is fully implemented — that branch is
        // self-contained (just state + TTL + AddLT). BEHF_TERRON /
        // BEHF_BREAK / BEHF_ANIM depend on `g_Config.m_ObjectDamages`,
        // `CMatrixProgressBar`, per-instance anim, effects, and the
        // side/win-count state machine; the original bodies stay
        // commented below with line refs so the port can fill them in
        // when those subsystems land.

        self.self_id = Some(self_id);

        // MatrixObject.cpp:105 — special (win-target) objects can only
        // be hit by the player.
        if attacker_side != PLAYER_SIDE && self.object_state & OBJECT_STATE_SPECIAL != 0 {
            return false;
        }

        if self.beh_flag == BehFlag::Burn {
            if weap == WEAPON_ABLAZE {
                // MatrixObject.cpp:111 — CSound::AddSound(S_WEAPON_HIT_ABLAZE).
                // Deferred until CSound lands.
            } else if is_fire_weapon(weap) {
                // MatrixObject.cpp:115-141.
                if self.object_state & OBJECT_STATE_ABLAZE == 0 {
                    self.object_state |= OBJECT_STATE_ABLAZE;
                }

                // TTL bump per-weapon (MatrixObject.cpp:123-128).
                let bump = if weap == WEAPON_PLASMA {
                    200
                } else if weap == WEAPON_FLAMETHROWER {
                    100
                } else {
                    10_000
                };
                self.ablaze_ttl = self.ablaze_ttl.saturating_add(bump);

                // `m_NextTime = g_MatrixMap->GetTime()` — schedule the
                // next fire-tick immediately. We don't have access to
                // the game-time clock from here; inside `logic_takt`
                // the IsAblaze branch treats `next_time` as relative
                // ms-until-next-emit, so seeding to 0 is correct.
                self.next_time = 0;

                // Clamp TTL (MatrixObject.cpp:133-139).
                let cap = if self.object_state & OBJECT_STATE_BURNED != 0 {
                    1500
                } else {
                    15_000
                };
                if self.ablaze_ttl > cap {
                    self.ablaze_ttl = cap;
                }

                // MatrixObject.cpp:140 — enroll in the logic-temp list
                // so the IsAblaze branch in `logic_takt` ticks every
                // `LOGIC_TAKT_PERIOD_MS`.
                objs.add_lt(self_id);
            }
        }
        // Snapshot: `init` inside the BREAK death swaps `beh_flag` to the
        // replacement row's behaviour; the C++ else-if chain dispatches
        // on the pre-swap flag only, so the branches below must too.
        let beh0 = self.beh_flag;
        if beh0 == BehFlag::Break {
            // Port of BEHF_BREAK (MatrixObject.cpp:188-234). Uses the
            // damage-table lookup; decrements hp when the current
            // hp clears the per-weapon `mindamage` floor. Death drops
            // the SPECIAL flag and re-Inits the object to the
            // replacement type encoded in the BEHAVIOUR field.
            let entry = objs.object_damages.get(weap).unwrap_or_default();
            if self.break_hit_point > entry.mindamage {
                self.break_hit_point -= entry.damage;
            }
            // m_PB progress-bar instantiation (MatrixObject.cpp:197-200)
            // — deferred: CMatrixProgressBar unported.

            if self.break_hit_point <= 0 {
                if self.object_state & OBJECT_STATE_SPECIAL != 0 {
                    self.object_state &= !OBJECT_STATE_SPECIAL;
                    // MatrixObject.cpp:206-212 — `--m_BeforeWinCount`,
                    // player BUILDING_KILL credit, win check. The
                    // counter / side-status half runs in MapLogic's
                    // drain of `pending_special_deaths`.
                    objs.inc_side_stat(PLAYER_SIDE, |s| s.building_kill += 1);
                    objs.pending_special_deaths
                        .push(crate::matrix_game::map_static::SpecialDeathKind::Target);
                }
                // `CreateExplosion(geo, ExplosionObject, true)` when
                // BEHAVIOUR's 4th comma-field == "Explode"
                // (MatrixObject.cpp:224-227).
                if wstr::str_par(&self.behaviour, 3, ",") == "Explode" {
                    objs.pending_explosions
                        .push(crate::matrix_game::map_static::ExplosionSpawn {
                            pos: self.core.geo_center,
                            props: &crate::matrix_game::effects::explosion::EXPLOSION_OBJECT,
                            fire: true,
                        });
                }
                // `Init(temp.GetIntPar(1,L","))` — swap to the broken
                // variant. behaviour = "Break,<newtype>,<hp>,<kind>"
                // (MatrixObject.cpp:232-233).
                self.object_state &= !OBJECT_STATE_SHADOW_SPECIAL;
                let new_type = wstr::int_par(&self.behaviour, 1, ",");
                self.init(new_type, objs);
            }
        } else if beh0 == BehFlag::Anim {
            // Port of BEHF_ANIM (MatrixObject.cpp:235-289). Like BREAK
            // but on death transitions to a new anim state instead of
            // re-Init'ing. The `ApplyAnimState` call looks up the next
            // state from the BEHAVIOUR string's state-table.
            let entry = objs.object_damages.get(weap).unwrap_or_default();
            if self.break_hit_point > entry.mindamage {
                self.break_hit_point -= entry.damage;
            }
            if self.break_hit_point <= 0 {
                if self.object_state & OBJECT_STATE_SPECIAL != 0 {
                    self.object_state &= !OBJECT_STATE_SPECIAL;
                    // MatrixObject.cpp:253-260 — same win-target
                    // bookkeeping as the BREAK branch.
                    objs.inc_side_stat(PLAYER_SIDE, |s| s.building_kill += 1);
                    objs.pending_special_deaths
                        .push(crate::matrix_game::map_static::SpecialDeathKind::Target);
                }
                // Walk the state table to find the transition for the
                // current `anim_state`. Field index 3 is "next state"
                // in the `#`-joined `<id>:<anim>:<hp>:<next>:...` spec
                // (MatrixObject.cpp:269-286). The damage path uses
                // index 3 (vs index 4 in `IsAnimEnd` transitions);
                // `ApplyAnimState(newstate)` re-seeds HP from the new
                // state's entry. No transition → HP stays ≤ 0.
                let beh = self.behaviour.clone();
                let state_table = wstr::str_par(&beh, 1, ",");
                let cnt = wstr::count_par(state_table, "#");
                for i in 0..cnt {
                    let entry = wstr::str_par(state_table, i, "#");
                    if wstr::int_par(entry, 0, ":") == self.anim_state {
                        let newstate = wstr::int_par(entry, 3, ":");
                        if newstate >= 0 {
                            self.apply_anim_state(newstate, &beh);
                            break;
                        }
                    }
                }
            }
        }

        // BEHF_TERRON (MatrixObject.cpp:142-187). Pain animation /
        // sounds / progress bar / music-volume not ported; HP
        // depletion + the death flags are.
        if beh0 == BehFlag::Terron
            && self.object_state & crate::matrix_game::map_static::OBJECT_STATE_TERRON_EXPL == 0
        {
            if let Some(entry) = objs.object_damages.get(weap) {
                if self.break_hit_point > entry.mindamage {
                    self.break_hit_point -= entry.damage;
                }
            }
            if self.break_hit_point <= 0 {
                // MatrixObject.cpp:168-172 — MMFLAG_TERRON_DEAD only
                // when the terron is the special win target.
                if self.object_state & OBJECT_STATE_SPECIAL != 0 {
                    objs.terron_dead = true;
                }
                self.object_state |= crate::matrix_game::map_static::OBJECT_STATE_TERRON_EXPL;
                // Death animation runs for 5s of ablaze TTL
                // (MatrixObject.cpp:177).
                self.ablaze_ttl = 5000;
                self.next_time = 0;
            }
        }

        false
    }

    /// Port of `CMatrixMapObject::LogicTakt` (MatrixObject.cpp:1229-1596).
    /// Massive switch on `m_BehFlag`. Currently handles BEHF_SENS (the
    /// sensor-radius detection path) and the IsAblaze burn-out timer;
    /// other branches enroll via `apply_ids_row` but their bodies need
    /// subsystems (progress bars, effects, sound, robot spawning) that
    /// aren't ported.
    fn logic_takt(
        &mut self,
        ms: i32,
        _rng: &mut Rnd,
        objs: &mut crate::matrix_game::map_static::Objects,
    ) {
        use crate::matrix_game::common::TRACE_ROBOT;
        use crate::matrix_game::map_static::fit_to_mask as _fit_to_mask;
        let _ = _fit_to_mask; // keep import used even when SENS disabled

        // Terron death sequence (MatrixObject.cpp:1235-1330): once
        // OBJECT_STATE_TERRON_EXPL is set, the 5s ablaze TTL counts
        // down through three phases — rolling building-boom pops,
        // a real BIGBOOM at TTL<1000 and another at TTL<100 — and on
        // expiry resolves the special-win bookkeeping + body swap.
        if self.beh_flag == BehFlag::Terron
            && self.object_state & crate::matrix_game::map_static::OBJECT_STATE_TERRON_EXPL != 0
        {
            self.terron_expl_takt(ms, _rng, objs);
            return;
        }

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
                let robot_nearby =
                    objs.any_object_in_radius(pos, self.sens_radius, 1.0, TRACE_ROBOT, None);

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

        // IsAblaze branch — MatrixObject.cpp:1534-1609. Every
        // OBJECT_ABLAZE_PERIOD the C++ rolls `IRND(10000) < ablaze_ttl`
        // and, on success, picks a point on the mesh (4 attempts) and
        // spawns a flame + smoke (big variant while TTL > 3700), then
        // calls Damage(WEAPON_ABLAZE) — which for BEHF_BURN objects is
        // a hit sound only, so it has nothing to port. The TTL decay /
        // ABLAZE unmark already runs in `static_takt` (base-class takt,
        // MatrixMapStatic.cpp:108-121).
        if self.object_state & OBJECT_STATE_ABLAZE != 0 {
            self.burn_time_total = self.burn_time_total.saturating_add(ms);

            self.next_time = self.next_time.saturating_sub(ms);
            while self.next_time <= 0 {
                self.next_time += crate::matrix_game::common::OBJECT_ABLAZE_PERIOD_MS;
                use crate::matrix_game::effects::explosion::FireAnim;
                use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd, Smoke};
                use crate::matrix_game::effects::GameEffect;
                let origin = glam::Vec3::new(
                    self.core.matrix.w_axis.x,
                    self.core.matrix.w_axis.y,
                    self.core.matrix.w_axis.z,
                );
                // Visual-only randomness (the C++ Damage call this roll
                // gates is sound-only here) — local seeded stream.
                let mut vrng = crate::matrix_game::logic::Rnd::new(
                    ((self.burn_time_total as i32) ^ ((origin.x + origin.y) as i32)).max(1),
                );
                if (vrng.next() % 10000) as i32 >= self.ablaze_ttl {
                    continue;
                }
                let r = self.core.radius.max(3.0);
                // 4-attempt mesh pick (bounding-sphere stand-in).
                let mut found = None;
                for _ in 0..4 {
                    let p = origin
                        + glam::Vec3::new(
                            fsrnd(&mut vrng, r),
                            fsrnd(&mut vrng, r),
                            frnd(&mut vrng, r * 2.0),
                        );
                    let dir = (origin - p).normalize_or_zero();
                    if dir == glam::Vec3::ZERO {
                        continue;
                    }
                    if let Some(t) = crate::matrix_game::map_trace::pick_sphere(
                        p,
                        dir,
                        self.core.geo_center,
                        self.core.radius,
                    ) {
                        found = Some((p, dir, t));
                        break;
                    }
                }
                if let Some((p, dir, t)) = found {
                    if self.ablaze_ttl > 3700 {
                        let fp = p + dir * (t + 6.0);
                        objs.pending_effects.push(GameEffect::FireAnim(FireAnim::new(
                            fp,
                            20.0,
                            30.0 + frnd(&mut vrng, 20.0),
                            (self.ablaze_ttl as f32 - frnd(&mut vrng, 4000.0)).max(1.0),
                        )));
                        objs.pending_effects.push(GameEffect::Smoke(Smoke::new(
                            fp + glam::Vec3::new(0.0, 0.0, 10.0),
                            100.0,
                            1300.0,
                            20.0,
                            0x8030_3030,
                            false,
                            1.0 / 30.0,
                        )));
                    } else {
                        objs.pending_effects.push(GameEffect::Smoke(Smoke::new(
                            p + dir * (t + 2.0),
                            100.0,
                            1300.0,
                            20.0,
                            0x4030_3030,
                            false,
                            1.0 / 30.0,
                        )));
                    }
                }
            }

            // MatrixObject.cpp:1576-1609: 5s after ignition, flip to
            // BURNED. The original also swaps the skin & possibly the
            // VO (for "Burn,Type,..." variants); those depend on the
            // skin manager and VO loader per-instance which aren't
            // ported.
            if self.object_state & OBJECT_STATE_BURNED == 0
                && self.burn_time_total > OBJECT_ABLAZE_BURNED_AT_MS
            {
                self.object_state |= OBJECT_STATE_BURNED;
                // `rchange |= MR_GRAPH` in C++ — triggers a skin swap
                // on the next `r_need` pass.
                self.rchange |= MR_GRAPH;
            }
            // No early return: logic_takt continues so BEHF_STATIC
            // ablaze objects still get whatever shared behaviour runs
            // below. (Currently nothing; placeholder for future
            // additions.)
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
    /// Map's `m_ShadowColor` (DATA_SHADOWCOLOR / MatrixMap.cpp:1830) unpacked
    /// into linear `(r, g, b, a)`. Mirrors `D3DRS_TEXTUREFACTOR` for the
    /// projected-shadow stage in `DrawShadowsProjFast`.
    shadow_color: [f32; 4],
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
    /// Per-instance visibility — pre-registered break replacements start
    /// hidden; `sync_break_swaps` flips entries when the arena object's
    /// `type_id` diverges from this batch's type (the `Init` swap at
    /// MatrixObject.cpp:233).
    hidden: Vec<bool>,
    /// VO type this batch renders — match target for the swap sync.
    type_id: u32,
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
    /// Per-shadow owner: pos-key + the type whose mesh cast it. The swap
    /// sync hides the shadow once the arena object's type diverges (the
    /// C++ Init drops m_ShadowProj, MatrixObject.cpp:973-979).
    shadow_keys: Vec<((i32, i32), u32)>,
    shadow_visible: Vec<bool>,
    /// One animation runtime per VO type. `MeshBatch::anim_slot` references
    /// this vector so surfaces of the same type share one `vo_frame`.
    anims: Vec<AnimState>,
    uniform_buffer: wgpu::Buffer,
    shadow_uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    /// Cached `m_ShadowColor` from the map (DATA_SHADOWCOLOR), packed as
    /// 0xAARRGGBB. Forwarded each frame to the projected-shadow uniform.
    shadow_color: u32,
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

        // BREAK / terron replacement type from a row's BEHAVIOUR field —
        // "Break,<newtype>,<hp>,<kind>" (MatrixObject.cpp:233, :1261).
        let replacement_of = |type_id: u32| -> Option<u32> {
            if type_id as usize >= strings.arrays_count() {
                return None;
            }
            let row = strings.get_as_wstr(type_id as usize);
            let beh = wstr::str_par(&row, OTP_BEHAVIOUR, "*");
            let beh = beh.strip_prefix('+').unwrap_or(beh);
            if !wstr::compare_first(beh, "Break") {
                return None;
            }
            let r = wstr::int_par(beh, 1, ",");
            (r >= 0).then_some(r as u32)
        };

        // Each breakable instance is pre-registered (hidden) under every
        // type in its replacement chain so the death-time `Init` swap
        // (MatrixObject.cpp:233) is a pure visibility flip — no runtime
        // VO loading. `bool` = initially hidden.
        let mut by_type: BTreeMap<u32, Vec<(ObjectInstance, bool)>> = BTreeMap::new();
        for obj in &map.objects {
            by_type
                .entry(obj.type_id)
                .or_default()
                .push((obj.clone(), false));
            let mut seen = vec![obj.type_id];
            let mut t = obj.type_id;
            while let Some(r) = replacement_of(t) {
                if seen.contains(&r) {
                    break;
                }
                seen.push(r);
                let mut clone = obj.clone();
                clone.type_id = r;
                // The CMAP-baked silhouette belongs to the original mesh;
                // the replacement renders shadowless (the original drops
                // its m_ShadowProj in Init, MatrixObject.cpp:973-979).
                clone.shadow = None;
                by_type.entry(r).or_default().push((clone, true));
                t = r;
            }
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
                shadow_color: [0.0, 0.0, 0.0, 0.45],
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
        let mut shadow_keys: Vec<((i32, i32), u32)> = Vec::new();
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
            let Some(paths) = resolve_paths(&id_str) else {
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
            let type_objects: Vec<ObjectInstance> =
                instances.iter().map(|(obj, _)| obj.clone()).collect();
            let type_hidden: Vec<bool> = instances.iter().map(|(_, h)| *h).collect();
            let inst_data = batch_instance_data(&type_objects, &type_hidden, cx, cy, map, None);
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
                    objects: type_objects.clone(),
                    hidden: type_hidden.clone(),
                    type_id: *type_id,
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
                    .filter(|(_, hidden)| !hidden)
                    .filter_map(|(obj, _)| obj.shadow.as_ref().map(|shadow| (obj, shadow)))
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
                        shadow_keys.push((pos_key(obj.0.x, obj.0.y), *type_id));
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

        let shadow_visible = vec![true; shadow_batches.len()];
        Some(Self {
            pipeline,
            shadow_pipeline,
            batches,
            shadow_batches,
            shadow_keys,
            shadow_visible,
            anims,
            uniform_buffer,
            shadow_uniform_buffer,
            fog_color,
            ambient_color,
            light_color,
            light_dir,
            shadow_color: map.shadow_color,
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
                let inst_data = batch_instance_data(
                    &batch.objects,
                    &batch.hidden,
                    cx,
                    cy,
                    map,
                    Some(point_lights),
                );
                queue.write_buffer(&batch.instance_buffer, 0, bytemuck::cast_slice(&inst_data));
            }
            self.last_point_light_revision = revision;
        }
    }

    /// Flip batch-instance visibility to follow runtime `MapObject::init`
    /// type swaps (BREAK death MatrixObject.cpp:233, terron corpse :1261).
    /// Arena objects are matched to render instances by position; an
    /// instance shows only in the batch whose `type_id` matches the
    /// arena object's current type. Called once per frame after the
    /// logic takt (alongside the buildings sync).
    pub fn sync_break_swaps(
        &mut self,
        queue: &wgpu::Queue,
        objs: &Objects,
        map: &GameMap,
        point_lights: &PointLightSystem,
    ) {
        let mut live: HashMap<(i32, i32), i32> = HashMap::new();
        for id in objs.iter_live() {
            let Some(obj) = objs.get(id) else { continue };
            if !matches!(obj.core().obj_type, ObjectType::MapObject) {
                continue;
            }
            let mo = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
            if mo.ruin_graph.is_some() {
                continue; // ruins render via the buildings renderer
            }
            let w = obj.core().matrix.w_axis;
            live.insert(pos_key(w.x, w.y), mo.type_id);
        }
        if live.is_empty() {
            return;
        }
        for batch in &mut self.batches {
            let mut dirty = false;
            for (i, obj) in batch.objects.iter().enumerate() {
                let Some(&t) = live.get(&pos_key(obj.x, obj.y)) else {
                    continue;
                };
                let want_hidden = t != batch.type_id as i32;
                if batch.hidden[i] != want_hidden {
                    batch.hidden[i] = want_hidden;
                    dirty = true;
                }
            }
            if dirty {
                let [cx, cy] = batch.center;
                let inst_data = batch_instance_data(
                    &batch.objects,
                    &batch.hidden,
                    cx,
                    cy,
                    map,
                    Some(point_lights),
                );
                queue.write_buffer(&batch.instance_buffer, 0, bytemuck::cast_slice(&inst_data));
            }
        }
        for (i, (key, ty)) in self.shadow_keys.iter().enumerate() {
            if let Some(&t) = live.get(key) {
                self.shadow_visible[i] = t == *ty as i32;
            }
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
        let sc = self.shadow_color;
        let sa = ((sc >> 24) & 0xFF) as f32 / 255.0;
        let sr = ((sc >> 16) & 0xFF) as f32 / 255.0;
        let sg = ((sc >> 8) & 0xFF) as f32 / 255.0;
        let sb = (sc & 0xFF) as f32 / 255.0;
        queue.write_buffer(
            &self.shadow_uniform_buffer,
            0,
            bytemuck::bytes_of(&ShadowProjUniform {
                view_proj: view_proj.to_cols_array_2d(),
                shadow_color: [sr, sg, sb, sa],
            }),
        );

        if !self.shadow_batches.is_empty() {
            pass.set_pipeline(&self.shadow_pipeline);
            for (i, batch) in self.shadow_batches.iter().enumerate() {
                if !self.shadow_visible.get(i).copied().unwrap_or(true) {
                    continue;
                }
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
    // Faithful port of `CMatrixMapObject::SetupMatricesForShadowTextureCalc`
    // (MatrixObject.cpp:340-372). Static-shadow geometry stored in the CMAP
    // was baked offline against THIS function's projector — `LookAtLH(campos,
    // campos+light, camup)` with `_sx = +1/dim.x`, `_sy = +1/dim.y`, then
    // `OrthoLH(-1, 1, 1, 1000)` (note width = -1 → mProg flips X).
    //
    // Composing mView * mProg:
    //   clip.x = -2 * right·(p - campos) / dim_x
    //   clip.y = +2 * up·(p - campos) / dim_y
    //
    // Our WGSL bake computes `clip.x = 1 - 2*uv.x`, `clip.y = 2*uv.y - 1`,
    // so we want
    //   uv.x = +right·(p - campos)/dim_x + 0.5
    //   uv.y =   +up·(p - campos)/dim_y + 0.5
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
/// row-major `mx * my * mz` (MatrixObject.cpp:392) — column-major Rz * Ry * Rx —
/// then uniform scale, then translation into centered render space.
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

/// Degenerate transform for a hidden instance (pre-spawned break
/// replacement / already-swapped original) — all vertices collapse to one
/// point so nothing rasterizes.
fn hidden_instance() -> InstanceData {
    InstanceData {
        row0: [0.0; 4],
        row1: [0.0; 4],
        row2: [0.0; 4],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [0.0; 4],
    }
}

fn batch_instance_data(
    objects: &[ObjectInstance],
    hidden: &[bool],
    cx: f32,
    cy: f32,
    map: &GameMap,
    point_lights: Option<&PointLightSystem>,
) -> Vec<InstanceData> {
    objects
        .iter()
        .enumerate()
        .map(|(i, obj)| {
            if hidden.get(i).copied().unwrap_or(false) {
                hidden_instance()
            } else {
                instance_matrix(obj, cx, cy, map, point_lights)
            }
        })
        .collect()
}

/// Pos-derived identity shared between arena `MapObject`s and render
/// instances (decorations never move; same keying as the buildings sync).
fn pos_key(x: f32, y: f32) -> (i32, i32) {
    ((x * 10.0) as i32, (y * 10.0) as i32)
}

fn object_rotation(obj: &ObjectInstance) -> Mat3 {
    let (sx, cxr) = obj.angle_x.sin_cos();
    let (sy, cyr) = obj.angle_y.sin_cos();
    let (sz, cz) = obj.angle_z.sin_cos();
    let rx = Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, cxr, sx, 0.0, -sx, cxr]);
    let ry = Mat3::from_cols_array(&[cyr, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cyr]);
    let rz = Mat3::from_cols_array(&[cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0]);
    rz * ry * rx
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
            x: 0.0,
            y: 0.0,
            z: 0.0,
            angle_z: 0.0,
            angle_x: 0.0,
            angle_y: 0.0,
            scale: 1.0,
            type_id: 0,
            shadow: None,
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
        let break_plain = "path*vo*tex******Break,X,5000,Normal";
        let break_terron = "path*vo*tex******Break,X,5000,Terron";
        let mut a = MapObject::from_instance(&inst());
        let mut b = MapObject::from_instance(&inst());
        assert!(a.apply_ids_row(break_plain, &mut Rnd::new(1), || {}));
        assert!(b.apply_ids_row(break_terron, &mut Rnd::new(1), || {}));
        assert_eq!(a.beh_flag, BehFlag::Break);
        assert_eq!(b.beh_flag, BehFlag::Terron);
    }

    #[test]
    fn anim_vs_portret_split_on_index_5() {
        // C++ checks `temp[5] == 'P'` with 0-based indexing
        // (MatrixObject.cpp:1062, CWStr.hpp:195) — the char after the
        // `,` following "Anim". A literal "AnimP" is BEHF_ANIM; only
        // rows like "Anim,P..." are Portret.
        let anim_plain = "path*vo*tex******Anim,0:1:100";
        let animp = "path*vo*tex******AnimP";
        let portret = "path*vo*tex******Anim,P";
        let mut a = MapObject::from_instance(&inst());
        let mut ap = MapObject::from_instance(&inst());
        let mut p = MapObject::from_instance(&inst());
        let a_lt = a.apply_ids_row(anim_plain, &mut Rnd::new(1), || {});
        let ap_lt = ap.apply_ids_row(animp, &mut Rnd::new(1), || {});
        let p_lt = p.apply_ids_row(portret, &mut Rnd::new(1), || {});
        assert_eq!(a.beh_flag, BehFlag::Anim);
        assert!(a_lt, "Anim opts into AddLT");
        assert_eq!(
            ap.beh_flag,
            BehFlag::Anim,
            "literal AnimP is BEHF_ANIM in C++"
        );
        assert!(ap_lt);
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
        let add_lt = o.apply_ids_row(row, &mut Rnd::new(1), || {
            bumps += 1;
        });
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
        assert!(!o.apply_ids_row("path*vo*tex******Anim,P", &mut rng, || {}));
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
    fn damage_on_static_object_is_noop() {
        // Non-BEHF_BURN objects ignore damage (MatrixObject.cpp:107 branch
        // guard). Confirms the fall-through of the switch.
        use crate::matrix_game::effects::weapon::WEAPON_BIGBOOM;
        use crate::matrix_game::map_static::Objects;
        let mut objs = Objects::new();
        let id = objs.spawn(Box::new(MapObject::from_instance(&inst())));
        let dropped =
            objs.apply_damage(id, WEAPON_BIGBOOM, glam::Vec3::ZERO, glam::Vec3::Z, 2, None);
        assert!(!dropped);
        let o = objs.get(id).unwrap();
        assert_eq!(o.object_state() & OBJECT_STATE_ABLAZE, 0);
        assert!(!objs.in_lt(id));
    }

    #[test]
    fn damage_burn_marks_ablaze_caps_ttl_and_enrolls_in_logic_list() {
        use crate::matrix_game::effects::weapon::{
            WEAPON_BIGBOOM, WEAPON_FLAMETHROWER, WEAPON_PLASMA,
        };
        use crate::matrix_game::map_static::Objects;
        let mut rng = Rnd::new(1);

        // BEHF_BURN object via apply_ids_row.
        let mut o = MapObject::from_instance(&inst());
        assert!(!o.apply_ids_row("path*vo*tex******Burn,Tex,Burn01", &mut rng, || {}));
        assert_eq!(o.beh_flag, BehFlag::Burn);

        let mut objs = Objects::new();
        let id = objs.spawn(Box::new(o));

        // WEAPON_BIGBOOM (fire-type) kindles with +10_000 TTL.
        assert!(!objs.apply_damage(
            id,
            WEAPON_BIGBOOM,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None
        ));
        let o = objs.get(id).unwrap();
        assert_ne!(o.object_state() & OBJECT_STATE_ABLAZE, 0);
        assert_eq!(o.ablaze_ttl(), 10_000);
        assert!(objs.in_lt(id), "BEHF_BURN objects AddLT on first damage");

        // Another hit with WEAPON_FLAMETHROWER: +100 to TTL.
        assert!(!objs.apply_damage(
            id,
            WEAPON_FLAMETHROWER,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None
        ));
        assert_eq!(objs.get(id).unwrap().ablaze_ttl(), 10_100);

        // WEAPON_PLASMA: +200.
        assert!(!objs.apply_damage(
            id,
            WEAPON_PLASMA,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None
        ));
        // Unburned cap = 15_000; the stacked TTL is 10_300 < cap.
        assert_eq!(objs.get(id).unwrap().ablaze_ttl(), 10_300);

        // Big hits stack TTL up to the unburned cap.
        for _ in 0..5 {
            objs.apply_damage(
                id,
                WEAPON_BIGBOOM,
                glam::Vec3::ZERO,
                glam::Vec3::Z,
                PLAYER_SIDE,
                None,
            );
        }
        assert_eq!(objs.get(id).unwrap().ablaze_ttl(), 15_000);
    }

    /// Shared Ids table for the body-swap tests. Tests that set the
    /// global table must all use this exact content — they run in
    /// parallel within one process.
    fn seed_test_ids() {
        set_global_ids(vec![
            String::new(),
            "ruins*wreck*tex******Break,2,77,Normal".to_string(),
            "ruins*husk*tex******Burn,Tex,Burn01".to_string(),
            "ruins*corpse*tex".to_string(),
        ]);
    }

    #[test]
    fn break_death_re_inits_as_replacement_type() {
        // BEHF_BREAK object with 500hp. WEAPON_GUN deals 100 / floor 0.
        // Death calls `Init(temp.GetIntPar(1,L","))` (MatrixObject.cpp:
        // 233): type_id swaps to the replacement row and that row
        // re-seeds behaviour + hit points.
        use crate::matrix_game::config::WeaponDamage;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_GUN};
        use crate::matrix_game::map_static::Objects;

        seed_test_ids();

        let mut rng = Rnd::new(1);
        let mut o = MapObject::from_instance(&inst());
        assert!(o.apply_ids_row("path*vo*tex******Break,1,500,Normal", &mut rng, || {},));
        assert_eq!(o.beh_flag, BehFlag::Break);
        assert_eq!(o.break_hit_point, 500);
        assert_eq!(o.break_hit_point_max, 500);

        let mut objs = Objects::new();
        objs.object_damages.table[weap_to_index(WEAPON_GUN).unwrap()] = WeaponDamage {
            damage: 100,
            mindamage: 0,
            friend_damage: 0,
        };
        let id = objs.spawn(Box::new(o));

        // 500 → 400 → 300 → 200 → 100 → ≤0
        for expected in [400, 300, 200, 100] {
            objs.apply_damage(
                id,
                WEAPON_GUN,
                glam::Vec3::ZERO,
                glam::Vec3::Z,
                PLAYER_SIDE,
                None,
            );
            let obj = objs.get(id).unwrap();
            let mo = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
            assert_eq!(mo.break_hit_point, expected);
            assert_eq!(mo.type_id, 0, "no swap before death");
        }

        // 5th hit kills → Init(1): type swap, Ids row 1 re-seeds the
        // Break behaviour with 77hp, graph + shadows flagged dirty.
        objs.apply_damage(
            id,
            WEAPON_GUN,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        let obj = objs.get(id).unwrap();
        let mo = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
        assert_eq!(mo.type_id, 1);
        assert_eq!(mo.uid, -1);
        assert_eq!(mo.beh_flag, BehFlag::Break);
        assert_eq!(mo.behaviour, "Break,2,77,Normal");
        assert_eq!(mo.break_hit_point, 77, "Init re-seeds hp from new row");
        assert_eq!(mo.break_hit_point_max, 77);
        assert_eq!(
            mo.object_state() & OBJECT_STATE_BURNED,
            0,
            "BURNED stand-in retired — the swap carries the visuals"
        );
        assert_ne!(mo.rchange() & MR_GRAPH, 0);
        assert_ne!(mo.rchange() & MR_SHADOW_STENCIL, 0);

        // Chain: killing the wreck swaps again, to the Burn row.
        objs.apply_damage(
            id,
            WEAPON_GUN,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        let obj = objs.get(id).unwrap();
        let mo = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
        assert_eq!(mo.type_id, 2);
        assert_eq!(mo.beh_flag, BehFlag::Burn);
        assert_eq!(mo.behaviour, "Burn,Tex,Burn01");
    }

    #[test]
    fn terron_corpse_swaps_type_after_death_countdown() {
        // Terron death: 5s of OBJECT_STATE_TERRON_EXPL countdown, then
        // `Init(temp.GetIntPar(1,L","))` (MatrixObject.cpp:1260-1262).
        use crate::matrix_game::config::WeaponDamage;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_GUN};
        use crate::matrix_game::logic::MapLogic;
        use crate::matrix_game::map_static::OBJECT_STATE_TERRON_EXPL;

        seed_test_ids();

        let mut world = MapLogic::with_seed(1);
        let mut o = MapObject::from_instance(&inst());
        assert!(o.apply_ids_row("path*vo*tex******Break,3,100,Terron", &mut Rnd::new(1), || {},));
        assert_eq!(o.beh_flag, BehFlag::Terron);
        let id = world.objects.spawn(Box::new(o));
        world.objects.add_lt(id);
        world.objects.object_damages.table[weap_to_index(WEAPON_GUN).unwrap()] = WeaponDamage {
            damage: 100,
            mindamage: 0,
            friend_damage: 0,
        };

        world.objects.apply_damage(
            id,
            WEAPON_GUN,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        let obj = world.objects.get(id).unwrap();
        assert_ne!(obj.object_state() & OBJECT_STATE_TERRON_EXPL, 0);
        let mo = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
        assert_eq!(mo.type_id, 0, "corpse swap waits for the countdown");

        world.takt(5200);
        let obj = world.objects.get(id).unwrap();
        let mo = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
        assert_eq!(mo.type_id, 3, "corpse type from the BEHAVIOUR string");
        assert_eq!(mo.beh_flag, BehFlag::Static, "corpse row has no behaviour");
    }

    #[test]
    fn break_damage_respects_mindamage_floor() {
        // Hit-points above mindamage get decremented; below/equal they
        // don't — matches MatrixObject.cpp:191 (`if hp > mindamage`).
        use crate::matrix_game::config::WeaponDamage;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_GUN};
        use crate::matrix_game::map_static::Objects;

        let mut rng = Rnd::new(1);
        let mut o = MapObject::from_instance(&inst());
        o.apply_ids_row("path*vo*tex******Break,1,30,Normal", &mut rng, || {});

        let mut objs = Objects::new();
        // Big damage but a 50-point floor — won't go below 50.
        objs.object_damages.table[weap_to_index(WEAPON_GUN).unwrap()] = WeaponDamage {
            damage: 20,
            mindamage: 50,
            friend_damage: 0,
        };
        let id = objs.spawn(Box::new(o));

        // Starting hp 30 ≤ mindamage 50 → no decrement, no death.
        objs.apply_damage(
            id,
            WEAPON_GUN,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        let obj = objs.get(id).unwrap();
        let mo = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
        assert_eq!(mo.break_hit_point, 30, "mindamage floor prevents hp drop");
        assert_eq!(mo.object_state() & OBJECT_STATE_BURNED, 0);
    }

    #[test]
    fn apply_anim_state_parses_hp_from_state_table() {
        // "Anim,0:5:100#1:3:50#2:7:0" — apply state=0 → hp 100,
        // state=1 → hp 50, state=2 → hp=0 treated as invincible.
        let mut o = MapObject::from_instance(&inst());
        o.apply_anim_state(0, "Anim,0:5:100#1:3:50#2:7:0");
        assert_eq!(o.anim_state, 0);
        assert_eq!(o.break_hit_point, 100);

        o.apply_anim_state(1, "Anim,0:5:100#1:3:50#2:7:0");
        assert_eq!(o.anim_state, 1);
        assert_eq!(o.break_hit_point, 50);

        o.apply_anim_state(2, "Anim,0:5:100#1:3:50#2:7:0");
        assert_eq!(o.anim_state, 2);
        assert_eq!(
            o.break_hit_point, 2_000_000_000,
            "hp=0 → invincible sentinel"
        );

        // Missing state: state_id stays set, hp left as-is.
        let prev = o.break_hit_point;
        o.apply_anim_state(99, "Anim,0:5:100#1:3:50#2:7:0");
        assert_eq!(o.anim_state, 99);
        assert_eq!(o.break_hit_point, prev);
    }

    #[test]
    fn anim_death_transitions_to_next_state_and_reseeds_hp() {
        // BEHF_ANIM death (MatrixObject.cpp:269-286): look up the
        // current state's entry, take field 3 as the next state and
        // ApplyAnimState it (re-seeding HP). next=-1 → no transition,
        // HP stays ≤ 0.
        use crate::matrix_game::config::WeaponDamage;
        use crate::matrix_game::effects::weapon::{weap_to_index, WEAPON_GUN};
        use crate::matrix_game::map_static::Objects;

        let mut rng = Rnd::new(1);
        let mut o = MapObject::from_instance(&inst());
        assert!(o.apply_ids_row("path*vo*tex******Anim,0:1:100:1#1:2:50:-1", &mut rng, || {},));
        assert_eq!(o.beh_flag, BehFlag::Anim);
        assert_eq!(o.break_hit_point, 100);

        let mut objs = Objects::new();
        objs.object_damages.table[weap_to_index(WEAPON_GUN).unwrap()] = WeaponDamage {
            damage: 100,
            mindamage: 0,
            friend_damage: 0,
        };
        let id = objs.spawn(Box::new(o));

        // First kill: state 0 → 1, HP re-seeded from state 1's entry.
        objs.apply_damage(
            id,
            WEAPON_GUN,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        let mo = unsafe { &*(objs.get(id).unwrap() as *const dyn MapStatic as *const MapObject) };
        assert_eq!(mo.anim_state, 1);
        assert_eq!(mo.break_hit_point, 50);

        // Second kill: state 1 has next=-1 — no transition, HP ≤ 0.
        objs.apply_damage(
            id,
            WEAPON_GUN,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        let mo = unsafe { &*(objs.get(id).unwrap() as *const dyn MapStatic as *const MapObject) };
        assert_eq!(mo.anim_state, 1);
        assert!(mo.break_hit_point <= 0);
    }

    #[test]
    fn anim_behaviour_seeds_state_zero_at_spawn() {
        // BEHF_ANIM's Init path calls ApplyAnimState(0), so a fresh
        // Anim object should have anim_state=0 + hp from the state-0
        // entry (MatrixObject.cpp:1072).
        let mut rng = Rnd::new(1);
        let mut o = MapObject::from_instance(&inst());
        assert!(o.apply_ids_row("path*vo*tex******Anim,0:1:300#1:2:150", &mut rng, || {},));
        assert_eq!(o.beh_flag, BehFlag::Anim);
        assert_eq!(o.anim_state, 0);
        assert_eq!(o.break_hit_point, 300);
    }

    #[test]
    fn damage_ignored_when_attacker_is_not_player_for_special_object() {
        // MatrixObject.cpp:105 — SPECIAL objects are player-only targets.
        use crate::matrix_game::effects::weapon::WEAPON_BIGBOOM;
        use crate::matrix_game::map_static::Objects;
        let mut rng = Rnd::new(1);

        let mut o = MapObject::from_instance(&inst());
        o.apply_ids_row("path*vo*tex******+Burn,Tex,Burn01", &mut rng, || {});
        assert_eq!(o.beh_flag, BehFlag::Burn);
        assert_ne!(o.object_state & OBJECT_STATE_SPECIAL, 0);

        let mut objs = Objects::new();
        let id = objs.spawn(Box::new(o));
        // Enemy AI side 2 cannot hit a SPECIAL object.
        objs.apply_damage(id, WEAPON_BIGBOOM, glam::Vec3::ZERO, glam::Vec3::Z, 2, None);
        assert_eq!(
            objs.get(id).unwrap().object_state() & OBJECT_STATE_ABLAZE,
            0
        );
        // But player (side 1) can.
        objs.apply_damage(
            id,
            WEAPON_BIGBOOM,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );
        assert_ne!(
            objs.get(id).unwrap().object_state() & OBJECT_STATE_ABLAZE,
            0
        );
    }

    #[test]
    fn ablaze_logic_takt_flips_burned_flag_after_5s() {
        use crate::matrix_game::effects::weapon::WEAPON_BIGBOOM;
        use crate::matrix_game::logic::MapLogic;
        let mut world = MapLogic::with_seed(1);

        let mut o = MapObject::from_instance(&inst());
        o.apply_ids_row("path*vo*tex******Burn,Tex,Burn01", &mut Rnd::new(1), || {});
        let id = world.objects.spawn(Box::new(o));

        // Kindle: adds OBJECT_STATE_ABLAZE, enrolls in LT, TTL=10_000.
        world.objects.apply_damage(
            id,
            WEAPON_BIGBOOM,
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            PLAYER_SIDE,
            None,
        );

        // Drive 4s of game-time — still not burned.
        world.takt(4000);
        let o = world.objects.get(id).unwrap();
        assert_ne!(
            o.object_state() & OBJECT_STATE_ABLAZE,
            0,
            "still ablaze at 4s"
        );
        assert_eq!(
            o.object_state() & OBJECT_STATE_BURNED,
            0,
            "not yet burned at 4s"
        );

        // Drive another 1.5s to cross the 5000ms mark.
        world.takt(1500);
        let o = world.objects.get(id).unwrap();
        assert_ne!(
            o.object_state() & OBJECT_STATE_BURNED,
            0,
            "BURNED flag latched after 5s of ablaze accumulation"
        );
    }

    #[test]
    fn sens_logic_takt_transitions_on_nearby_robot() {
        use crate::matrix_game::logic::MapLogic;
        use crate::matrix_game::logic::Rnd;
        use crate::matrix_game::map_static::{MapStatic, ObjectType};

        // Build a world with one SENS mapobject at the origin and a
        // "robot" (stub MapStatic with ObjectType::RobotAi) 30 units
        // away — inside the 50-unit sens radius.
        let mut world = MapLogic::with_seed(1);
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
        let mapobj = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
        assert_eq!(
            mapobj.prev_state_robots_in_radius, 1,
            "sensor detected the robot and transitioned to state=1"
        );

        // Move robot outside the radius, takt again.
        let robot_slot_obj = world.objects.get_mut(robot_id).unwrap();
        robot_slot_obj.core_mut().geo_center = glam::Vec3::new(200.0, 0.0, 0.0);

        world.takt(210);

        let obj = world.objects.get(sensor_id).unwrap();
        let mapobj = unsafe { &*(obj as *const dyn MapStatic as *const MapObject) };
        assert_eq!(
            mapobj.prev_state_robots_in_radius, 0,
            "sensor falls back to idle after robot leaves the radius"
        );
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
