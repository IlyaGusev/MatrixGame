//! Map data + top-level draw orchestration — ports MatrixMap.cpp.
//!
//! The top half of the file is the map *data* (heightmap, groups,
//! properties) parsed from `.CMAP`. The bottom half is `MapRenderer`,
//! which owns the GPU resources and orchestrates the per-frame draw.
//!
//! Per-group terrain build code lives in `map_group.rs`; per-group
//! water alpha / shoreline in `water.rs`; atlas construction in
//! `map_prepare.rs`. Keep those split — this file is already big.

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{
    float2int, unpack_rgb, CELLFLAG_BRIDGE, CELLFLAG_FLAT, CELLFLAG_LAND, CELLFLAG_WATER, FOG_END,
    FOG_START, MAP_GROUP_SIZE, TEX_BOTTOM_SIZE,
};
use crate::matrix_game::effects::point_light::{PointLightRenderer, PointLightSystem};
use crate::matrix_game::map_group::{DrawBatch, Vertex};
use crate::matrix_game::map_prepare::build_tex_union_atlases;
use crate::matrix_game::ter_surface::{
    build_surface_overlays, GlossOverlayBatch, GlossResources, GlossVertex, OverlayBatch,
};
use crate::matrix_game::water::visible_groups_mask;
use crate::matrix_lib::base::storage::Storage;
use crate::matrix_lib::three_g::texture::*;

pub const GLOBAL_SCALE: f32 = 20.0;

// ── Scoped current-map pointer (the `g_MatrixMap` of this port) ────────
//
// Subsystems dispatched from `logic_takt` / `takt` via the
// `MapStatic` trait need a reference to the map (pathfinder, terrain
// Z lookup, spatial queries). Threading `&GameMap` through the
// trait signature would ripple through every test stub. Instead we
// stash a raw pointer in a static for the duration of the tick call,
// guarded by a scope RAII handle so it's cleared automatically.
// The pointer is only ever set while the caller's `&GameMap`
// borrow is live.

use std::cell::Cell;
use std::ptr;

// Thread-local, NOT a process global: parallel unit tests each run
// their own MapScope; a shared static would let one thread's drop
// (or re-enter) dangle another thread's pointer mid-takt.
thread_local! {
    static CURRENT_MAP: Cell<*const GameMap> = const { Cell::new(ptr::null()) };
    static CURRENT_ELAPSED_MS: Cell<i64> = const { Cell::new(0) };
}

/// Loaded map name (the CMAP path) — drives the "terron" easter-egg check
/// (MatrixLogic.cpp:2776). Set once at load.
static MAP_NAME: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
// BEHF_PORTRET map objects are hidden unless `MMFLAG_SHOWPORTRETS` (permanent,
// set when an in-position robot dies) or `MMFLAG_ROBOT_IN_POSITION` (per-frame,
// terron trigger held) is set (MatrixObject.cpp:889-892). Both off by default.
static PORTRETS_SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PORTRETS_IN_POS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
use std::sync::atomic::Ordering as PortretOrd;

pub fn set_map_name(name: &str) {
    *MAP_NAME.lock().unwrap() = name.to_string();
    // A fresh map load resets the reveal flags.
    PORTRETS_SHOWN.store(false, PortretOrd::Relaxed);
    PORTRETS_IN_POS.store(false, PortretOrd::Relaxed);
}

pub fn map_name() -> String {
    MAP_NAME.lock().unwrap().clone()
}

pub fn map_name_is_terron() -> bool {
    MAP_NAME.lock().unwrap().to_lowercase().contains("terron")
}

/// `ShowPortrets()` — permanently reveal (MatrixMap.cpp:3598-3601).
pub fn show_portrets() {
    PORTRETS_SHOWN.store(true, PortretOrd::Relaxed);
}

pub fn set_portrets_in_position(v: bool) {
    PORTRETS_IN_POS.store(v, PortretOrd::Relaxed);
}

pub fn portrets_visible() -> bool {
    PORTRETS_SHOWN.load(PortretOrd::Relaxed) || PORTRETS_IN_POS.load(PortretOrd::Relaxed)
}

/// RAII guard — sets `CURRENT_MAP` + `CURRENT_ELAPSED_MS` on
/// construction, clears on drop. Use:
/// `let _scope = MapScope::enter(&map, elapsed_ms); world.takt(ms);`.
pub struct MapScope;

impl MapScope {
    pub fn enter(map: &GameMap, elapsed_ms: i64) -> Self {
        CURRENT_MAP.with(|c| c.set(map as *const GameMap));
        CURRENT_ELAPSED_MS.with(|c| c.set(elapsed_ms));
        MapScope
    }
}
impl Drop for MapScope {
    fn drop(&mut self) {
        CURRENT_MAP.with(|c| c.set(ptr::null()));
        // Reset the clock too: mission setup runs outside any scope,
        // and a leftover end-of-mission time from the previous game
        // would seed spawn-time timers (e.g. cannon fire_next_think_time)
        // differently on every mission after the first. The C++
        // equivalent g_MatrixMap->GetTime() restarts at 0 per map.
        CURRENT_ELAPSED_MS.with(|c| c.set(0));
    }
}

/// Fetch the currently-scoped map. Returns `None` outside a takt
/// call. The lifetime is tied to the outstanding `MapScope` — do
/// not cache the reference past the tick.
///
/// Safety: caller must not hold the returned reference after the
/// outermost `MapScope` drops. Used only from inside takt dispatch
/// which always sits inside a scope.
pub fn current_map<'a>() -> Option<&'a GameMap> {
    let ptr = CURRENT_MAP.with(|c| c.get());
    if ptr.is_null() {
        None
    } else {
        unsafe { Some(&*ptr) }
    }
}

/// Game-time elapsed (ms) as registered by the active `MapScope`.
/// Ports `g_MatrixMap->GetTime()`.
pub fn current_elapsed_ms() -> i64 {
    CURRENT_ELAPSED_MS.with(|c| c.get())
}

thread_local! {
    static FRUSTUM_CENTER: Cell<Option<[f32; 2]>> = const { Cell::new(None) };
}

/// Camera frustum-center XY for effect-distance culling — set once per
/// frame by the app loop. `None` (headless sims/tests) disables culling.
pub fn set_frustum_center(xy: [f32; 2]) {
    FRUSTUM_CENTER.with(|c| c.set(Some(xy)));
}

thread_local! {
    static FULL_AUTO: Cell<bool> = const { Cell::new(false) };
}

/// `MMFLAG_FULLAUTO` (`g_MatrixMap->m_Flags`) — the player side is
/// AI-driven (demo mode / headless sim). ORed with the non-player-side
/// check in `CanBreakOrder`'s capture guard (MatrixRobot.hpp:387).
pub fn full_auto() -> bool {
    FULL_AUTO.with(|c| c.get())
}

pub fn set_full_auto(on: bool) {
    FULL_AUTO.with(|c| c.set(on));
}

/// `MAX_EFFECT_DISTANCE_SQ` gate (MatrixEffect.hpp:13, guards the
/// Create* effect factories at MatrixEffect.cpp:456 etc.).
pub fn effect_in_range(x: f32, y: f32) -> bool {
    const MAX_EFFECT_DISTANCE_SQ: f32 = 3000.0 * 3000.0;
    match FRUSTUM_CENTER.with(|c| c.get()) {
        Some([cx, cy]) => {
            let (dx, dy) = (x - cx, y - cy);
            dx * dx + dy * dy <= MAX_EFFECT_DISTANCE_SQ
        }
        None => true,
    }
}

/// Heightmap point from SCompilePoint (12 bytes, /Zp1 packed).
#[derive(Debug, Clone, Copy)]
pub struct CompilePoint {
    pub move_idx: i32,
    pub z: f32,
    pub b: u8,
    pub g: u8,
    pub r: u8,
    pub flags: u8,
}

/// Per-point surface normal (computed from surrounding heights).
#[derive(Debug, Clone, Copy)]
pub struct PointNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Per-fine-cell movement metadata. Ports `SMatrixMapMove`
/// (MatrixMap.hpp:139-168). The `stop` bitmask gates passability
/// per chassis kind: bit `1 << kind` set means "cannot cross this
/// cell" for that chassis kind. Zone/sphere/zubchik carry region
/// / sphere-corner / geometric-corner info the full AI / physics
/// code uses; the pathfinder only reads `stop`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoveCell {
    pub zone: i32,
    pub sphere: u32,
    pub zubchik: u32,
    pub stop: u32,
}

impl MoveCell {
    /// Bit-mask for a placement of `size` cells square by chassis
    /// kind `nsh`. Port of MatrixLogic.cpp:520 —
    /// `(1 << nsh) << (6 * (size - 1))`. `size` must be 1..=5.
    /// For a robot footprint (size = 4) the mask is at bits 18-22.
    #[inline]
    pub fn stop_mask(nsh: usize, size: i32) -> u32 {
        debug_assert!((1..=5).contains(&size));
        (1u32 << nsh) << (6 * (size - 1) as u32)
    }

    /// True when `size × size` footprint at this cell is blocked
    /// for chassis `nsh`. Matches `(m_Stop & mask) != 0` semantics
    /// from MatrixLogic.cpp:523, :877.
    #[inline]
    pub fn is_impassable_for(&self, nsh: usize, size: i32) -> bool {
        (self.stop & Self::stop_mask(nsh, size)) != 0
    }

    /// Port of `SMatrixMapMove::GetType(nsh)` (MatrixMap.hpp:149-166).
    /// Packs the 8 per-corner sphere/zubchik bits for chassis `nsh`
    /// into one byte:
    ///   - bits 0..3 = `m_Sphere` corner mask (sphere corners at LU/RU/RD/LD)
    ///   - bits 4..7 = `m_Zubchik` corner mask (triangular `zubchik` corners)
    ///
    /// Returns `0xff` when the size-1 cell is *passable* for `nsh`
    /// (`m_Stop` bit for size-1 is not set — shift=0). That sentinel is
    /// what `SphereRobotToAABBObstacleCollision` tests at :2791.
    #[inline]
    pub fn get_type(&self, nsh: usize) -> u8 {
        if (self.stop & (1u32 << nsh)) == 0 {
            return 0xff;
        }
        let mut rv: u8 = 0;
        if (self.sphere & (1u32 << nsh)) != 0 {
            rv |= 1;
        }
        if (self.sphere & ((1u32 << nsh) << 8)) != 0 {
            rv |= 2;
        }
        if (self.sphere & ((1u32 << nsh) << 16)) != 0 {
            rv |= 4;
        }
        if (self.sphere & ((1u32 << nsh) << 24)) != 0 {
            rv |= 8;
        }
        if (self.zubchik & (1u32 << nsh)) != 0 {
            rv |= 16;
        }
        if (self.zubchik & ((1u32 << nsh) << 8)) != 0 {
            rv |= 32;
        }
        if (self.zubchik & ((1u32 << nsh) << 16)) != 0 {
            rv |= 64;
        }
        if (self.zubchik & ((1u32 << nsh) << 24)) != 0 {
            rv |= 128;
        }
        rv
    }
}

/// Per-cell coefficients matching SMatrixMapUnit fields used by GetZ.
#[derive(Debug, Clone, Copy)]
pub struct MapUnit {
    pub flags: u8,
    pub a1: f32,
    pub b1: f32,
    pub c1: f32,
    pub a2: f32,
    pub b2: f32,
    pub c2: f32,
}

pub struct GameMap {
    pub size_x: usize,
    pub size_y: usize,
    pub camera_angle: f32,
    pub camera_pos: Option<[f32; 2]>,
    pub tex_union_dim: usize,
    pub water_color: u32,
    pub sky_color: u32,
    pub sky_name: String,
    pub sky_angle: f32,
    pub water_name: String,
    pub water_normal_len: f32,
    /// Per-side starting economy from the map `SideResInfo` property
    /// (MatrixMapPrepare.cpp:464-493): `(side_id, [res;4], force_up)`.
    pub side_res_info: Vec<(i32, [i32; 4], i32)>,
    /// Per-side AI tuning from the map `SideAIInfo` property
    /// (MatrixMapPrepare.cpp:420-458): `(side_name, [(key, value)])`
    /// with keys TBB/SK/DK/WRK/BK/TC.
    pub side_ai_info: Vec<(String, Vec<(String, String)>)>,
    /// `MaintenanceTime` map property → `m_MaintenancePRC`
    /// (MatrixMapPrepare.cpp:411-418): percentage scaling the
    /// reinforcement cooldown; 0 disables maintenance entirely.
    pub maintenance_prc: i32,
    /// Author-placed ambient effect spawners (RS_EFFECTS,
    /// MatrixMapPrepare.cpp:902-926): `(pos, spec)` where `spec` is
    /// the comma-separated `type,minper,maxper,…` string from the map
    /// IDS table.
    pub effect_spawners: Vec<([f32; 3], String)>,
    /// Camera terrain-follow clamp bounds from building floor heights
    /// (`m_GroundZBaseMiddle` = mean build_z, `m_GroundZBaseMax` = max;
    /// MatrixMapPrepare.cpp:623-690).
    pub ground_z_base_middle: f32,
    pub ground_z_base_max: f32,
    pub light_main_color: u32,
    pub light_main_color_obj: u32,
    pub light_main_dir: [f32; 3],
    pub ambient_color_obj: u32,
    pub ambient_color: u32,
    /// Map-defined `ShadowColor` (DATA_SHADOWCOLOR, MatrixMapPrepare.cpp:1204).
    /// Packed as 0xAARRGGBB and used by `DrawShadowsProjFast` as the
    /// `D3DRS_TEXTUREFACTOR` (MatrixMap.cpp:1830).
    pub shadow_color: u32,
    pub terrain2object_influence: f32,
    pub terrain2object_target_color: u32,
    /// Foamline tint — DATA_INSHOREWAVECOLOR, used by the inshore wave pass
    /// (MatrixMapPrepare.cpp:1183, MatrixWater.cpp:160).
    pub inshorewave_color: u32,
    /// Precomputed inshore wave spawn points, one list per map group in
    /// row-major order. Loaded from `inshores/{X,Y,NX,NY}` in the CMAP
    /// (MatrixMapPrepare.cpp:1386-1417). Groups without water get an empty
    /// vec. `pos` is uncentered world coords, `dir` points toward land.
    pub inshore_prespawns: Vec<Vec<InshorePreSpawn>>,
    pub macro_texture_path: Option<String>,
    pub macro_texture_size: i32,   // m_MacrotextureSize from "SIM" param
    pub points: Vec<CompilePoint>, // (size_x+1) * (size_y+1) points
    pub normals: Vec<PointNormal>, // same size, computed from heights
    pub units: Vec<MapUnit>,       // size_x * size_y cells, ports SMatrixMapUnit GetZ data
    pub objects: Vec<ObjectInstance>, // palms / rocks / decorative scenery placements
    /// Starting buildings from the `buildings/*` CMAP table
    /// (MatrixMapPrepare.cpp:621-694). Each entry carries the fields needed to
    /// construct a `CMatrixBuilding`; geometry resolution happens at render
    /// time via the building's `EBuildingType` → `Matrix\Building\bN.cvo`
    /// mapping (MatrixObjectBuilding.cpp:158-163).
    pub buildings: Vec<BuildingInstance>,
    /// Pre-placed decorative base ruins (`side == 255` records).
    pub ruins: Vec<RuinInstance>,
    /// All cannon records from the `cannons/*` CMAP table
    /// (MatrixMapPrepare.cpp:776-885). One entry per record regardless
    /// of `prop`: prop=0 standalone cannons, prop=1 factory cannons
    /// mounted on a base, prop=2 empty placement slots (skipped at
    /// spawn). The C++ creates a `CMatrixCannon` for every non-empty
    /// record; we mirror that in `MapLogic::spawn_buildings`.
    pub cannons: Vec<CannonInstance>,
    /// Initial robot placements from the `robots/*` CMAP table — one
    /// entry per `SPreRobot` parsed at MatrixMapPrepare.cpp:697-770.
    /// `MapLogic::spawn_robots` materialises these into live arena
    /// `Robot`s on world init, matching `StaticPrepare2`'s loop.
    pub robots: Vec<RobotInstance>,
    /// Per-group max LAND z, size = group_w * group_h.
    /// Ports GetGroupMaxZLand (MatrixMap.hpp:759): returns 0 for empty groups /
    /// groups whose max is negative. Used by the camera to keep the link-point
    /// above mountains (MatrixCamera.cpp:926 → GetZInterpolatedLand).
    pub group_max_z_land: Vec<f32>,
    pub group_w: usize,
    pub group_h: usize,
    /// Per-move-cell passability / terrain metadata. Flat row-major,
    /// dimensions `(size_x * 2) × (size_y * 2)`. Ports `m_Move` built
    /// in MatrixMapPrepare.cpp:1244-1290 — the fine-grained grid the
    /// robot AI pathfinder operates on.
    pub move_cells: Vec<MoveCell>,
    pub size_move_x: usize,
    pub size_move_y: usize,
    /// Global minimum heightmap z — the plane `CalcMapGroupVisibility`
    /// projects the frustum corner rays onto (MatrixVisiCalc.cpp:567).
    pub min_z: f32,
    /// Per-group axis-aligned vertical extents, row-major. Corresponds to
    /// CMatrixMapGroup::{m_minz, m_maxz} (MatrixMapGroup.hpp:103); fed into
    /// the 3D frustum-AABB test for the IsInFrustum fallback branch.
    pub group_bounds: Vec<GroupBounds>,
    /// `CMatrixMap::m_RN` — the road/zone/region network the robot AI
    /// navigates, loaded from the ver-27 `roads/Data` blob
    /// (MatrixMapPrepare.cpp:1599-1610). `None` when the map ships no
    /// roads block.
    /// `m_RN` (MatrixMap.hpp:298). `Mutex` (never contended — game
    /// logic is single-threaded) because the zone-path queries
    /// (`FindPathInZone`) mutate scratch fields inside the network
    /// while callers only hold `&GameMap`, and `GameMap` must stay
    /// `Sync` for the render-side closures.
    pub road_network: Option<std::sync::Mutex<crate::matrix_game::logic::road_network::RoadNetwork>>,
}

impl GameMap {
    /// Test-only: a flat all-land map of `size_x × size_y` cells at
    /// height `z`. Enough structure for `get_z` / `trace` / move-cell
    /// queries; everything render-related is defaulted.
    #[cfg(test)]
    pub fn test_flat(size_x: usize, size_y: usize, z: f32) -> Self {
        use crate::matrix_game::common::{CELLFLAG_FLAT, CELLFLAG_LAND};
        let points = vec![
            CompilePoint {
                move_idx: 0,
                z,
                b: 128,
                g: 128,
                r: 128,
                flags: 0,
            };
            (size_x + 1) * (size_y + 1)
        ];
        let normals = vec![
            PointNormal {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            };
            (size_x + 1) * (size_y + 1)
        ];
        let units = vec![
            MapUnit {
                flags: CELLFLAG_LAND | CELLFLAG_FLAT,
                a1: z,
                b1: 0.0,
                c1: 0.0,
                a2: 0.0,
                b2: 0.0,
                c2: 0.0,
            };
            size_x * size_y
        ];
        Self {
            size_x,
            size_y,
            camera_angle: 0.0,
            camera_pos: None,
            tex_union_dim: 0,
            water_color: 0,
            sky_color: 0,
            sky_name: String::new(),
            sky_angle: 0.0,
            water_name: String::new(),
            water_normal_len: 1.0,
            side_res_info: Vec::new(),
            side_ai_info: Vec::new(),
            maintenance_prc: 100,
            effect_spawners: Vec::new(),
            ground_z_base_middle: 0.0,
            ground_z_base_max: 0.0,
            light_main_color: 0,
            light_main_color_obj: 0,
            light_main_dir: [0.0, 0.0, -1.0],
            ambient_color_obj: 0,
            ambient_color: 0,
            shadow_color: 0,
            terrain2object_influence: 0.0,
            terrain2object_target_color: 0,
            inshorewave_color: 0,
            inshore_prespawns: Vec::new(),
            macro_texture_path: None,
            macro_texture_size: 0,
            points,
            normals,
            units,
            objects: Vec::new(),
            buildings: Vec::new(),
            ruins: Vec::new(),
            cannons: Vec::new(),
            robots: Vec::new(),
            group_max_z_land: Vec::new(),
            group_w: 0,
            group_h: 0,
            move_cells: vec![MoveCell::default(); size_x * 2 * size_y * 2],
            size_move_x: size_x * 2,
            size_move_y: size_y * 2,
            min_z: z,
            group_bounds: Vec::new(),
            road_network: None,
        }
    }
}

/// Ports CMatrixMapGroup's bounding-Z pair — used by the camera frustum
/// visibility check to decide whether a group's 3D footprint intersects the
/// view frustum. For empty / all-water groups the values are `(0, 0)`.
#[derive(Debug, Clone, Copy)]
pub struct GroupBounds {
    pub min_z: f32,
    pub max_z: f32,
}

/// Precomputed shoreline wave spawn point. Ports SPreInshorewave fields
/// (MatrixWater.hpp — see MatrixMapPrepare.cpp:1397-1414 for the load path).
#[derive(Debug, Clone, Copy)]
pub struct InshorePreSpawn {
    pub pos: [f32; 2],
    pub dir: [f32; 2],
}

/// A static object placement — ports the per-instance fields of DATA_OBJECTS
/// (MatrixMapPrepare.cpp:503-619). Position is uncentered world space (raw map
/// coords); renderer applies the map-center offset.
#[derive(Debug, Clone)]
pub struct ObjectInstance {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub angle_z: f32,
    pub angle_x: f32,
    pub angle_y: f32,
    pub scale: f32,
    pub type_id: u32,
    pub shadow: Option<ObjectShadow>,
}

#[derive(Debug, Clone)]
pub struct ObjectShadow {
    pub vertices: Vec<ObjectShadowVertex>,
    pub indices: Vec<u32>,
    pub camera_pos: [f32; 3],
    pub dimensions: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct ObjectShadowVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

/// A pre-placed decorative base ruin (`side == 255` building record,
/// MatrixMapPrepare.cpp:648-668): spawned as ruin-mesh `MapObject`s
/// rather than a live building.
#[derive(Debug, Clone, Copy)]
pub struct RuinInstance {
    pub x: f32,
    pub y: f32,
    pub build_z: f32,
    pub angle: i32,
    pub kind: i32,
}

/// A starting-building placement — ports the per-row fields of DATA_BUILDINGS
/// (MatrixMapPrepare.cpp:621-694). Position is uncentered world space; the
/// renderer applies the map-center offset like it does for objects. `build_z`
/// is the terrain floor height at `(x, y)` — `m_BuildZ` in
/// `CMatrixBuilding::OnLoad` (MatrixObjectBuilding.cpp:1029), which falls back
/// to the water-level trace when the cell is submerged.
#[derive(Debug, Clone)]
pub struct BuildingInstance {
    pub x: f32,
    pub y: f32,
    pub build_z: f32,
    /// Raw quantized angle from the CMAP (0..=3): 0°/90°/180°/270° around the
    /// Z axis. Ports the fixed rotation table in `CMatrixBuilding::RNeed`
    /// (MatrixObjectBuilding.cpp:120-137).
    pub angle: u8,
    /// 1..8 — team color. 255 is the "ruins" sentinel that
    /// MatrixMapPrepare.cpp:648-668 treats as a static ruin mesh instead of a
    /// live building; we skip those in this loader (the ruins branch would
    /// need `Matrix\Building\ruins\bN[p].vo` loading).
    pub side: u8,
    /// `EBuildingType` — 0=Base, 1=Titan, 2=Plasma, 3=Electronic, 4=Energy,
    /// 5=Repair (MatrixObjectBuilding.hpp:59-69).
    pub kind: u8,
    pub shadow_kind: u8,
    pub shadow_size: i32,
    /// Port of `m_TurretsPlacesCnt` (MatrixObjectBuilding.hpp:187) — the
    /// number of factory/turret slots assigned to this building by the
    /// cannon-placement pass in MatrixMapPrepare.cpp:776-897. Computed by
    /// scanning `cannons` records in the CMAP: each cannon with `prop=1`
    /// (factory cannon) or `prop=2` (place-only) attaches to the nearest
    /// building inside `MAX_DISTANCE_CANNON_BUILDING` and capped at
    /// `MAX_PLACES`. Drives the turret-UI slot-count visibility
    /// (`podl1..4` on the Main panel).
    pub turrets_places_cnt: i32,
    /// Per-slot data — port of `m_TurretsPlaces[MAX_PLACES]`
    /// (MatrixObjectBuilding.hpp:32-37). Each entry corresponds to one
    /// `cannons` record (prop=1 or 2) attached to this building by the
    /// nearest-building scan. `cannon_type` is `c->m_Num` for prop=1
    /// (a pre-mounted factory cannon) and -1 for prop=2 (an empty
    /// build slot).
    pub turret_places: Vec<TurretPlaceCmap>,
}

/// One CMAP `cannons/*` record. Mirrors the loop body at
/// MatrixMapPrepare.cpp:798-885 — every record allocates a
/// `CMatrixCannon` with these fields. `parent_building` resolves at
/// spawn time (the loader already does the nearest-base scan and
/// stores the building's index here for prop=1/2 attachments).
#[derive(Debug, Clone, Copy)]
pub struct CannonInstance {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub kind: u8, // `m_Num` — cannon variant (1..=4)
    pub side: u8,
    pub add_h: f32,
    /// 0 = standalone, 1 = factory cannon (mounted), 2 = empty slot
    /// (no cannon spawned, just records placement).
    pub prop: i32,
    /// Index into `GameMap::buildings` if attached, else None.
    pub parent_building: Option<usize>,
    /// Slot index on the parent (only meaningful for prop=1).
    pub parent_slot: i32,
}

/// One entry in `m_TurretsPlaces[]`. Coordinates are in MOVE-cell units
/// (`Float2Int(x / GLOBAL_SCALE_MOVE)`), matching the C++ at
/// MatrixMapPrepare.cpp:852-853 — slot equality + the snap test in
/// `IsInPlaces` work against these int coords.
#[derive(Debug, Clone, Copy)]
pub struct TurretPlaceCmap {
    /// Move-cell coords (x, y).
    pub coord: (i32, i32),
    /// Slot-default angle (cannon faces this direction unless rotated).
    pub angle: f32,
    /// -1 = empty (player can build here), 1..=4 = pre-mounted factory
    /// cannon of that kind.
    pub cannon_type: i32,
}

/// One initial robot placed by the map. Ports `SPreRobot`
/// (MatrixMapPrepare.cpp:697-770). The CMAP stores a parametric
/// description (chassis/armor/head/weapons) plus position, side,
/// shadow params, and a tactic group index; `MapLogic::spawn_robots`
/// later assembles a `RobotConfig` and live `Robot` for each entry,
/// mirroring `StaticPrepare2`'s loop at MatrixMapPrepare.cpp:1702-1736.
#[derive(Debug, Clone, Copy)]
pub struct RobotInstance {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Side ID — 1 = player, 2..=8 = AI (matches CMatrixSide convention).
    pub side: u8,
    pub shadow: u8,
    pub shadow_size: i32,
    /// Tactic group index (1..=3) for AI sides. -1 sentinel value when
    /// outside that range — `StaticPrepare2` calls `SetTeam(-1)` for
    /// "no preset team", deferred to `GroupNoTeamRobot`.
    pub group: i32,
    /// Chassis spawn angle in radians, from the third comma-separated
    /// field of the chassis token in `Units`. Used to seed
    /// `m_Forward = (-sin, cos)` (MatrixMapPrepare.cpp:1709-1710).
    pub angle: f32,
    /// Parsed chassis kind — preserves the raw `RUK_CHASSIS_*` value
    /// (1..=5) so `chassis_from_kind` can resolve the enum at spawn.
    pub chassis_kind: RobotUnitKind,
    pub armor_kind: RobotUnitKind,
    pub head_kind: RobotUnitKind,
    /// Sequential weapon kinds as parsed from `Units`. Indices here are
    /// "k-th weapon in the string"; the legality / pylon-pick
    /// re-shuffles them onto the armor's pylon matrix at spawn.
    pub weapons: [RobotUnitKind; crate::matrix_game::object_robot::MAX_WEAPON_CNT],
}

impl GameMap {
    /// Load a map from CMAP file data (raw bytes from pkg archive).
    pub fn from_cmap_bytes(cmap_data: &[u8]) -> Result<Self> {
        let stor = Storage::from_bytes(cmap_data).context("parsing CStorage from CMAP")?;

        let prop_names = stor
            .get_buf("properties", "Name")
            .context("missing properties/Name")?;
        let prop_values = stor
            .get_buf("properties", "Value")
            .context("missing properties/Value")?;

        let size_x = find_property_int(prop_names, prop_values, "SizeInUnitsX")
            .context("missing SizeInUnitsX")? as usize;
        let size_y = find_property_int(prop_names, prop_values, "SizeInUnitsY")
            .context("missing SizeInUnitsY")? as usize;

        let tex_union_dim =
            find_property_int(prop_names, prop_values, "TexUnionDim").unwrap_or(16) as usize;
        let camera_angle = find_property_float(prop_names, prop_values, "CamAngle").unwrap_or(0.0);
        let camera_pos = match (
            find_property_float(prop_names, prop_values, "CamPosX"),
            find_property_float(prop_names, prop_values, "CamPosY"),
        ) {
            (Ok(x), Ok(y)) => Some([x, y]),
            _ => None,
        };
        let water_color =
            find_property_int(prop_names, prop_values, "WaterColor").unwrap_or(0x003060) as u32;
        // DEF_SKY_COLOR (Common.hpp:27) = 0x1070FF
        let sky_color =
            find_property_int(prop_names, prop_values, "SkyColor").unwrap_or(0x1070FF) as u32;
        let sky_name = prop_names
            .find_as_wstr("SkyName")
            .map(|idx| prop_values.get_as_wstr(idx))
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "Default".to_string());
        // DATA_SKYANGLE is an offset in radians applied on top of the sky
        // config block's base angle (MatrixMapPrepare.cpp:1170-1174).
        let sky_angle = find_property_float(prop_names, prop_values, "SkyAngle").unwrap_or(0.0);
        let water_name = prop_names
            .find_as_wstr("WaterName")
            .map(|idx| prop_values.get_as_wstr(idx))
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "water_blue".to_string());
        // Per-side starting resources / base income multiplier.
        let side_res_info = prop_names
            .find_as_wstr("SideResInfo")
            .map(|idx| prop_values.get_as_wstr(idx))
            .map(|s| parse_side_res_info(&s))
            .unwrap_or_default();
        // Per-side AI tuning (MatrixMapPrepare.cpp:420-458) — entries
        // "SideName:KEY/VAL/KEY/VAL|...".
        let side_ai_info = prop_names
            .find_as_wstr("SideAIInfo")
            .map(|idx| prop_values.get_as_wstr(idx))
            .map(|s| parse_side_ai_info(&s))
            .unwrap_or_default();
        let maintenance_prc =
            find_property_int(prop_names, prop_values, "MaintenanceTime").unwrap_or(100);
        // Ambient effect spawners (RS_EFFECTS, MatrixMapPrepare.cpp:
        // 902-926): parallel f32 X/Y/Z arrays + i32 IDS index into the
        // map strings table, which holds the spawner spec string.
        let effect_spawners: Vec<([f32; 3], String)> = (|| {
            let fx = stor.get_buf("effects", "X")?;
            let fy = stor.get_buf("effects", "Y")?;
            let fz = stor.get_buf("effects", "Z")?;
            let ft = stor.get_buf("effects", "Type")?;
            let ids = stor.get_buf("strings", "String")?;
            let f32s = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let xs = f32s(fx.get_bytes(0));
            let ys = f32s(fy.get_bytes(0));
            let zs = f32s(fz.get_bytes(0));
            let ts: Vec<i32> = ft
                .get_bytes(0)
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let n = xs.len().min(ys.len()).min(zs.len()).min(ts.len());
            Some(
                (0..n)
                    .filter(|&i| ts[i] >= 0 && (ts[i] as usize) < ids.arrays_count())
                    .map(|i| ([xs[i], ys[i], zs[i]], ids.get_as_wstr(ts[i] as usize)))
                    .collect(),
            )
        })()
        .unwrap_or_default();
        let water_normal_len =
            find_property_float(prop_names, prop_values, "WaterNormLen").unwrap_or(1.0);
        let light_main_color =
            find_property_int(prop_names, prop_values, "LightMainColor").unwrap_or(0x989898) as u32;
        let light_main_color_obj = find_property_int(prop_names, prop_values, "LightMainColorObj")
            .unwrap_or(light_main_color as i32) as u32;
        let ambient_color_obj = find_property_int(prop_names, prop_values, "AmbientColorObj")
            .unwrap_or(0x808080) as u32;
        let ambient_color =
            find_property_int(prop_names, prop_values, "AmbientColor").unwrap_or(0x808080) as u32;
        // ShadowColor is uninitialized in the C++ default-ctor and only
        // assigned when the property is present (MatrixMapPrepare.cpp:1204).
        // Fall back to the alpha-only black that matches the previous
        // hardcoded `(0,0,0,0.45)` shader constant.
        let shadow_color = find_property_int(prop_names, prop_values, "ShadowColor")
            .unwrap_or(0x73000000u32 as i32) as u32;
        let mut terrain2object_influence =
            find_property_float(prop_names, prop_values, "Influence").unwrap_or(0.0);
        let inshorewave_color = find_property_int(prop_names, prop_values, "InshorewaveColor")
            .unwrap_or(0x008080) as u32;
        let terrain2object_target_color = if terrain2object_influence > 0.0 {
            0xFFFFFF
        } else if terrain2object_influence < 0.0 {
            terrain2object_influence = terrain2object_influence.abs();
            0x000000
        } else {
            0x000000
        };
        terrain2object_influence = terrain2object_influence.clamp(0.0, 1.0);

        // Light direction: RotX(angleX) * RotZ(angleZ) * (0, 0, -1)
        // Ports MatrixMapPrepare.cpp lines 1228-1231
        let angle_x =
            find_property_float(prop_names, prop_values, "LightMainAngleX").unwrap_or(0.61);
        let angle_z =
            find_property_float(prop_names, prop_values, "LightMainAngleZ").unwrap_or(-1.75);
        let (sx, cx_) = angle_x.sin_cos();
        let (sz, cz) = angle_z.sin_cos();
        // D3DXMatrixRotationX * D3DXMatrixRotationZ * (0,0,-1)
        // RotX: y' = y*cx - z*sx, z' = y*sx + z*cx
        // RotZ: x' = x*cz - y*sz, y' = x*sz + y*cz
        // Start with (0, 0, -1):
        // After RotX: (0, 0*cx - (-1)*sx, 0*sx + (-1)*cx) = (0, sx, -cx)
        // After RotZ: (0*cz - sx*sz, 0*sz + sx*cz, -cx) = (-sx*sz, sx*cz, -cx_)
        let light_main_dir = [-sx * sz, sx * cz, -cx_];

        // MacroTexture = "Matrix\Macrotexture\05?SIM80" → size = 80
        let mut macro_texture_path = None;
        let mut macro_texture_size = 1i32;
        if let Some(idx) = prop_names.find_as_wstr("MacroTexture") {
            let val = prop_values.get_as_wstr(idx);
            let path = val.split('?').next().unwrap_or("").replace('\\', "/");
            if !path.is_empty() {
                macro_texture_path = Some(path);
            }
            if let Some(sim_pos) = val.find("SIM") {
                if let Ok(size) = val[sim_pos + 3..].parse::<i32>() {
                    macro_texture_size = size;
                }
            }
        }

        log::info!(
            "map: size = {}x{} units, TexUnionDim={}",
            size_x,
            size_y,
            tex_union_dim
        );

        let points_buf = stor
            .get_buf("points", "Data")
            .context("missing points/Data")?;
        let points_data = points_buf.get_bytes(0);

        let expected_points = (size_x + 1) * (size_y + 1);
        let point_size = 12; // sizeof(SCompilePoint) under /Zp1
        if points_data.len() < expected_points * point_size {
            bail!(
                "points data too small: {} bytes, expected {} ({} points * {})",
                points_data.len(),
                expected_points * point_size,
                expected_points,
                point_size
            );
        }

        let mut points = Vec::with_capacity(expected_points);
        for i in 0..expected_points {
            let off = i * point_size;
            let move_idx = i32::from_le_bytes([
                points_data[off],
                points_data[off + 1],
                points_data[off + 2],
                points_data[off + 3],
            ]);
            let z = f32::from_le_bytes([
                points_data[off + 4],
                points_data[off + 5],
                points_data[off + 6],
                points_data[off + 7],
            ]);
            let b = points_data[off + 8];
            let g = points_data[off + 9];
            let r = points_data[off + 10];
            let flags = points_data[off + 11];

            points.push(CompilePoint {
                move_idx,
                z,
                b,
                g,
                r,
                flags,
            });
        }

        log::info!(
            "map: loaded {} heightmap points, z range: {:.1}..{:.1}",
            points.len(),
            points.iter().map(|p| p.z).fold(f32::INFINITY, f32::min),
            points.iter().map(|p| p.z).fold(f32::NEG_INFINITY, f32::max),
        );

        // Compute surface normals — ports PointCalcNormals (MatrixMapPrepare.cpp:20-105)
        let _w = size_x + 1;
        let normals = compute_normals(&points, size_x, size_y);
        let mut units = compute_units(&points, size_x, size_y);
        let (move_cells, size_move_x, size_move_y) =
            load_move_cells(&stor, &points, size_x, size_y);
        load_bridges(&stor, &mut points, &mut units, size_x, size_y);
        // Debug: count impassable-for-Track cells per size bucket.
        let mut track_blocked = [0usize; 5];
        let mut all_zero = 0usize;
        for c in &move_cells {
            if c.stop == 0 {
                all_zero += 1;
            }
            for size in 1..=5 {
                let mask = (1u32 << 2) << (6 * (size - 1));
                if c.stop & mask != 0 {
                    track_blocked[size - 1] += 1;
                }
            }
        }
        log::info!(
            "map: loaded move grid {}x{} ({} cells); Track-impassable per size [1,2,3,4,5]={:?}; m_Stop==0 cells={}",
            size_move_x, size_move_y, move_cells.len(), track_blocked, all_zero,
        );

        let objects = load_objects(&stor, &points, &normals, size_x, size_y);
        log::info!("map: loaded {} decorative objects", objects.len());

        // Publish the map's object-Ids table (g_MatrixMap->m_Ids) so the
        // runtime `MapObject::init` body swap can look up replacement
        // rows from the damage path (MatrixObject.cpp:985-1018).
        if let Some(strings) = stor.get_buf("strings", "String") {
            let rows: Vec<String> = (0..strings.arrays_count())
                .map(|i| strings.get_as_wstr(i))
                .collect();
            super::object::set_global_ids(rows);
        }

        // Ports MatrixMapPrepare.cpp:621-694 (RS_BUILDINGS step). We need the
        // per-cell units computed above so the floor-z fallback matches
        // `CMatrixBuilding::OnLoad`'s `GetZ` lookup.
        let mut ruins = Vec::new();
        let mut buildings = load_buildings(&stor, &units, size_x, size_y, &mut ruins);
        let cannons = load_cannons(&stor, &mut buildings);
        let robots = load_robots(&stor, &units, size_x, size_y);
        // Camera-follow clamp bounds (mean / max building floor z).
        let (ground_z_base_middle, ground_z_base_max) = if buildings.is_empty() {
            (0.0, 0.0)
        } else {
            let sum: f32 = buildings.iter().map(|b| b.build_z).sum();
            let max = buildings.iter().map(|b| b.build_z).fold(0.0f32, f32::max);
            // C++ divides by the TOTAL building-record count including
            // side==255 ruins (MatrixMapPrepare.cpp:643,691) — ruins are
            // split into their own vec here, so add them back.
            (sum / (buildings.len() + ruins.len()) as f32, max)
        };
        log::info!(
            "map: loaded {} starting buildings, {} cannon records, {} initial robots",
            buildings.len(),
            cannons.len(),
            robots.len(),
        );

        // No dilation: the stored per-group max is the max of THIS group's
        // land-cell corner z only. Dilation used to smear mountain height
        // into neighboring water groups, pushing the camera up while over
        // water and making the heavily-fogged far water read as sky color.
        let (group_max_z_land, group_w, group_h) =
            compute_group_max_z_land(&points, &units, size_x, size_y);

        let min_z = points.iter().map(|p| p.z).fold(f32::INFINITY, f32::min);
        let group_bounds = compute_group_bounds(&points, size_x, size_y, group_w, group_h);

        let road_network = wrap_road_network(load_road_network(&stor, size_move_x, size_move_y)?);

        let disable_inshore =
            find_property_int(prop_names, prop_values, "DisableInshore").unwrap_or(0) != 0;
        let inshore_prespawns =
            load_inshore_prespawns(&stor, group_w, group_h, disable_inshore);
        let total_inshores: usize = inshore_prespawns.iter().map(|v| v.len()).sum();
        log::info!(
            "map: loaded {} inshore wave spawn points across {} groups",
            total_inshores,
            inshore_prespawns.iter().filter(|v| !v.is_empty()).count()
        );

        Ok(Self {
            size_x,
            size_y,
            camera_angle,
            camera_pos,
            tex_union_dim,
            water_color,
            sky_color,
            sky_name,
            sky_angle,
            water_name,
            water_normal_len,
            side_res_info,
            side_ai_info,
            maintenance_prc,
            effect_spawners,
            ground_z_base_middle,
            ground_z_base_max,
            ruins,
            light_main_color,
            light_main_color_obj,
            light_main_dir,
            ambient_color_obj,
            ambient_color,
            shadow_color,
            terrain2object_influence,
            terrain2object_target_color,
            inshorewave_color,
            inshore_prespawns,
            macro_texture_path,
            macro_texture_size,
            points,
            normals,
            units,
            objects,
            buildings,
            cannons,
            robots,
            group_max_z_land,
            group_w,
            group_h,
            move_cells,
            size_move_x,
            size_move_y,
            min_z,
            group_bounds,
            road_network,
        })
    }

    /// Max land-corner z of the group containing world (wx, wy), clamped to 0.
    /// Ports GetGroupMaxZLand (MatrixMap.hpp:759). Returns 0 for out-of-bounds.
    pub fn group_max_z_land_at(&self, wx: f32, wy: f32) -> f32 {
        let gs = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE;
        let gx = (wx / gs).floor() as i32;
        let gy = (wy / gs).floor() as i32;
        if gx < 0 || gy < 0 || gx >= self.group_w as i32 || gy >= self.group_h as i32 {
            return 0.0;
        }
        self.group_max_z_land[gy as usize * self.group_w + gx as usize]
    }

    /// Bilinear-interpolated per-group max z in GROUP-CENTER coordinates.
    /// Ports the smooth-over-neighbors idea of GetZInterpolatedLand
    /// (MatrixMap.cpp:426-469) — a B-spline there, bilinear here. The returned
    /// value is clamped to `ceiling` so the camera never rises unreasonably
    /// high over tall terrain (original clamps per-group z to
    /// `[m_GroundZBaseMiddle, m_GroundZBaseMax]` from buildings).
    pub fn group_max_z_interpolated(&self, wx: f32, wy: f32, ceiling: f32) -> f32 {
        let gs = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE;
        // Group centers are at ((gx + 0.5) * gs, (gy + 0.5) * gs).
        let fx = wx / gs - 0.5;
        let fy = wy / gs - 0.5;
        let gx0 = fx.floor() as i32;
        let gy0 = fy.floor() as i32;
        let tx = fx - gx0 as f32;
        let ty = fy - gy0 as f32;

        // Clamp each group-z to [m_GroundZBaseMiddle, m_GroundZBaseMax]
        // (MatrixMap.cpp:445-446,455-456) when those bounds are valid.
        let (lo, hi) = (self.ground_z_base_middle, self.ground_z_base_max);
        let clamp_base = hi > lo;
        let sample = |gx: i32, gy: i32| -> f32 {
            if gx < 0 || gy < 0 || gx >= self.group_w as i32 || gy >= self.group_h as i32 {
                return 0.0;
            }
            let mut v = self.group_max_z_land[gy as usize * self.group_w + gx as usize];
            if clamp_base {
                v = v.clamp(lo, hi);
            }
            v.min(ceiling)
        };
        let a = sample(gx0, gy0);
        let b = sample(gx0 + 1, gy0);
        let c = sample(gx0, gy0 + 1);
        let d = sample(gx0 + 1, gy0 + 1);
        let ab = a + (b - a) * tx;
        let cd = c + (d - c) * tx;
        ab + (cd - ab) * ty
    }

    pub fn point(&self, x: usize, y: usize) -> &CompilePoint {
        &self.points[y * (self.size_x + 1) + x]
    }

    pub fn normal(&self, x: usize, y: usize) -> &PointNormal {
        &self.normals[y * (self.size_x + 1) + x]
    }

    pub fn unit(&self, x: usize, y: usize) -> &MapUnit {
        &self.units[y * self.size_x + x]
    }

    /// Bounds-checked `SMatrixMapUnit::m_Flags` accessor. Ports the
    /// `UnitGetTest(x, y)` helper at MatrixMap.hpp. Returns `None`
    /// for off-map / negative coords.
    pub fn unit_flags(&self, x: i32, y: i32) -> Option<u8> {
        if x < 0 || y < 0 {
            return None;
        }
        let (x, y) = (x as usize, y as usize);
        if x >= self.size_x || y >= self.size_y {
            return None;
        }
        Some(self.units[y * self.size_x + x].flags)
    }

    pub fn world_width(&self) -> f32 {
        self.size_x as f32 * GLOBAL_SCALE
    }

    pub fn world_height(&self) -> f32 {
        self.size_y as f32 * GLOBAL_SCALE
    }

    /// Move-cell width/height — `GLOBAL_SCALE_MOVE` in
    /// MatrixMap.hpp:18. Each map cell is sub-divided into `MOVE_CNT`
    /// (=2) move cells on each axis.
    pub const GLOBAL_SCALE_MOVE: f32 = GLOBAL_SCALE / 2.0;

    /// World coords → move-cell coords. Mirrors
    /// `Float2Int(x * INVERT(GLOBAL_SCALE_MOVE))` in `PlaceGet` /
    /// `MapPosCalc` / `GetLost` (MatrixLogic.cpp:468-470,
    /// MatrixRobot.hpp:355).
    pub fn world_to_move(&self, wx: f32, wy: f32) -> (i32, i32) {
        (
            float2int(wx / Self::GLOBAL_SCALE_MOVE),
            float2int(wy / Self::GLOBAL_SCALE_MOVE),
        )
    }

    /// Move-cell coords → world-space center of the cell.
    pub fn move_to_world(&self, mx: i32, my: i32) -> (f32, f32) {
        (
            (mx as f32 + 0.5) * Self::GLOBAL_SCALE_MOVE,
            (my as f32 + 0.5) * Self::GLOBAL_SCALE_MOVE,
        )
    }

    pub fn move_cell(&self, mx: i32, my: i32) -> Option<&MoveCell> {
        if mx < 0 || my < 0 {
            return None;
        }
        let (mx, my) = (mx as usize, my as usize);
        if mx >= self.size_move_x || my >= self.size_move_y {
            return None;
        }
        Some(&self.move_cells[my * self.size_move_x + mx])
    }

    /// Pathfinder passability check for a robot-sized footprint.
    /// Reads the single precomputed `size = ROBOT_MOVECELLS_PER_SIZE`
    /// bit — `(1 << chassis) << 18` — which the CMAP loader baked in
    /// (encodes the full 4×4 footprint test in one bit per cell per
    /// chassis kind). Ports `PlaceIsEmpty`'s wall test
    /// (MatrixLogic.cpp:877). `size` = 1..=5.
    pub fn is_passable_size(&self, mx: i32, my: i32, chassis_kind: usize, size: i32) -> bool {
        match self.move_cell(mx, my) {
            Some(c) => !c.is_impassable_for(chassis_kind, size),
            None => false,
        }
    }

    /// World point lies on a map cell (`UnitGetTest != NULL`).
    pub fn contains_point(&self, wx: f32, wy: f32) -> bool {
        let x = (wx / GLOBAL_SCALE).floor() as i32;
        let y = (wy / GLOBAL_SCALE).floor() as i32;
        x >= 0 && y >= 0 && x < self.size_x as i32 && y < self.size_y as i32
    }

    /// Ports CMatrixMap::GetZ for terrain/water boundary tests.
    pub fn get_z(&self, wx: f32, wy: f32) -> f32 {
        let sx = wx / GLOBAL_SCALE;
        let sy = wy / GLOBAL_SCALE;
        let x = sx.floor() as i32;
        let y = sy.floor() as i32;

        if x < 0 || y < 0 || x >= self.size_x as i32 || y >= self.size_y as i32 {
            return -1000.0;
        }

        let unit = self.unit(x as usize, y as usize);
        if unit.flags & CELLFLAG_BRIDGE == 0 && unit.flags & CELLFLAG_WATER != 0 {
            return -1000.0;
        }
        if unit.flags & CELLFLAG_FLAT != 0 {
            return unit.a1;
        }

        let local_x = wx - x as f32 * GLOBAL_SCALE;
        let local_y = wy - y as f32 * GLOBAL_SCALE;
        if local_y < local_x {
            unit.a1 * local_x + unit.b1 * local_y + unit.c1
        } else {
            unit.a2 * local_x + unit.b2 * local_y + unit.c2
        }
    }

    /// Average z of the 4 heightmap points under `(wx, wy)` — how the
    /// C++ derives a cannon's world Z every matrix rebuild
    /// (MatrixObjectCannon.cpp:263-273, `PointGetTest` × 4 + m_AddH).
    pub fn point_avg_z(&self, wx: f32, wy: f32) -> f32 {
        let px = crate::matrix_game::common::trunc_float(wx / GLOBAL_SCALE) as i32;
        let py = crate::matrix_game::common::trunc_float(wy / GLOBAL_SCALE) as i32;
        let stride = self.size_x as i32 + 1;
        let rows = self.size_y as i32 + 1;
        let mut cnt = 0;
        let mut rv = 0.0f32;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (x, y) = (px + dx, py + dy);
            if x >= 0 && y >= 0 && x < stride && y < rows {
                cnt += 1;
                rv += self.points[(y * stride + x) as usize].z;
            }
        }
        if cnt == 0 {
            0.0
        } else {
            rv / cnt as f32
        }
    }

    pub fn get_color(&self, wx: f32, wy: f32) -> u32 {
        self.get_color_with_lighting(wx, wy, None)
    }

    /// Port of `CMatrixMap::GetColor`, including optional point-light
    /// luminance contributions when a point-light system is present.
    pub fn get_color_with_lighting(
        &self,
        wx: f32,
        wy: f32,
        point_lights: Option<&PointLightSystem>,
    ) -> u32 {
        let sx = wx / GLOBAL_SCALE;
        let sy = wy / GLOBAL_SCALE;
        let x = sx.floor() as i32;
        let y = sy.floor() as i32;

        if x < 0 || y < 0 || x >= self.size_x as i32 || y >= self.size_y as i32 {
            return self.ambient_color_obj;
        }

        let unit = self.unit(x as usize, y as usize);
        if unit.flags & CELLFLAG_WATER != 0 && unit.flags & CELLFLAG_BRIDGE == 0 {
            return self.ambient_color_obj;
        }

        let kx = sx - x as f32;
        let ky = sy - y as f32;

        let c00 = self.point(x as usize, y as usize);
        let c10 = self.point(x as usize + 1, y as usize);
        let c01 = self.point(x as usize, y as usize + 1);
        let c11 = self.point(x as usize + 1, y as usize + 1);
        let l00 = point_lights
            .map(|lights| lights.point_lum(x as usize, y as usize, self.size_x))
            .unwrap_or([0, 0, 0]);
        let l10 = point_lights
            .map(|lights| lights.point_lum(x as usize + 1, y as usize, self.size_x))
            .unwrap_or([0, 0, 0]);
        let l01 = point_lights
            .map(|lights| lights.point_lum(x as usize, y as usize + 1, self.size_x))
            .unwrap_or([0, 0, 0]);
        let l11 = point_lights
            .map(|lights| lights.point_lum(x as usize + 1, y as usize + 1, self.size_x))
            .unwrap_or([0, 0, 0]);

        let sample =
            |c00: u8, c10: u8, c01: u8, c11: u8, l00: i32, l10: i32, l01: i32, l11: i32| -> u32 {
                // MatrixMap.cpp:
                //   Float2Int(LERPFLOAT(ky, LERPFLOAT(kx, c00, c10), LERPFLOAT(kx, c01, c11)))
                // using the per-point color plus runtime luminance.
                let top_left = c00 as i32 + l00;
                let top_right = c10 as i32 + l10;
                let bottom_left = c01 as i32 + l01;
                let bottom_right = c11 as i32 + l11;
                let top = top_left as f32 + (top_right - top_left) as f32 * kx;
                let bottom = bottom_left as f32 + (bottom_right - bottom_left) as f32 * kx;
                float2int(top + (bottom - top) * ky).clamp(0, 255) as u32
            };

        let r = sample(c00.r, c10.r, c01.r, c11.r, l00[0], l10[0], l01[0], l11[0]);
        let g = sample(c00.g, c10.g, c01.g, c11.g, l00[1], l10[1], l01[1], l11[1]);
        let b = sample(c00.b, c10.b, c01.b, c11.b, l00[2], l10[2], l01[2], l11[2]);

        (r << 16) | (g << 8) | b
    }

    /// Port of `CMatrixMap::GetNormal` (MatrixMap.cpp:636) — bilinearly
    /// interpolates point normals with two opt-in special cases:
    ///
    /// * Bridge cells derive their normal from finite differences of `GetZ`
    ///   (half a cell apart in each axis) so decals sit on the bridge plane
    ///   rather than the terrain below.
    /// * When `check_water` is true and any of the four surrounding heightmap
    ///   points dip below sea level, we return straight up so shore surfaces
    ///   flatten out instead of following submerged normals.
    pub fn get_normal(&self, wx: f32, wy: f32, check_water: bool) -> [f32; 3] {
        let scaled_x = wx / GLOBAL_SCALE;
        let scaled_y = wy / GLOBAL_SCALE;
        let x = scaled_x.floor() as i32;
        let y = scaled_y.floor() as i32;

        if x < 0 || y < 0 || x >= self.size_x as i32 || y >= self.size_y as i32 {
            return [0.0, 0.0, 1.0];
        }

        let unit = self.unit(x as usize, y as usize);
        if unit.flags & CELLFLAG_FLAT != 0 {
            return [0.0, 0.0, 1.0];
        }

        if unit.flags & CELLFLAG_BRIDGE != 0 {
            let half = GLOBAL_SCALE * 0.5;
            let nx = self.get_z(wx - half, wy) - self.get_z(wx + half, wy);
            let ny = self.get_z(wx, wy - half) - self.get_z(wx, wy + half);
            let nz = GLOBAL_SCALE;
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
            return [nx / len, ny / len, nz / len];
        }

        if unit.flags & CELLFLAG_WATER != 0 {
            return [0.0, 0.0, 1.0];
        }

        if check_water {
            let p00 = self.point(x as usize, y as usize);
            let p10 = self.point(x as usize + 1, y as usize);
            let p01 = self.point(x as usize, y as usize + 1);
            let p11 = self.point(x as usize + 1, y as usize + 1);
            if p00.z < 0.0 || p10.z < 0.0 || p01.z < 0.0 || p11.z < 0.0 {
                return [0.0, 0.0, 1.0];
            }
        }

        let kx = scaled_x - x as f32;
        let ky = scaled_y - y as f32;
        let n00 = self.normal(x as usize, y as usize);
        let n10 = self.normal(x as usize + 1, y as usize);
        let n01 = self.normal(x as usize, y as usize + 1);
        let n11 = self.normal(x as usize + 1, y as usize + 1);

        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let x0 = lerp(n00.x, n10.x, kx);
        let y0 = lerp(n00.y, n10.y, kx);
        let z0 = lerp(n00.z, n10.z, kx);
        let x1 = lerp(n01.x, n11.x, kx);
        let y1 = lerp(n01.y, n11.y, kx);
        let z1 = lerp(n01.z, n11.z, kx);
        let nx = lerp(x0, x1, ky);
        let ny = lerp(y0, y1, ky);
        let nz = lerp(z0, z1, ky);
        let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
        [nx / len, ny / len, nz / len]
    }

    pub fn static_object_color(&self, wx: f32, wy: f32) -> u32 {
        self.static_object_color_with_lighting(wx, wy, None)
    }

    pub fn static_object_color_with_lighting(
        &self,
        wx: f32,
        wy: f32,
        point_lights: Option<&PointLightSystem>,
    ) -> u32 {
        blend_color(
            self.get_color_with_lighting(wx, wy, point_lights),
            self.terrain2object_target_color,
            self.terrain2object_influence,
        )
    }
}

/// Ports PointCalcNormals (MatrixMapPrepare.cpp:20-105).
/// For each point, computes normal from cross products of 4 adjacent triangles.
fn compute_normals(points: &[CompilePoint], size_x: usize, size_y: usize) -> Vec<PointNormal> {
    let w = size_x + 1;
    let h = size_y + 1;
    let mut normals = vec![
        PointNormal {
            x: 0.0,
            y: 0.0,
            z: 1.0
        };
        w * h
    ];

    let get_z = |x: usize, y: usize| -> f32 { points[y * w + x].z };
    let get_flags = |x: usize, y: usize| -> u8 { points[y * w + x].flags };

    // Check if unit at (ux,uy) is land — unit flags are on point (ux,uy)
    let is_land = |ux: i32, uy: i32| -> bool {
        if ux < 0 || uy < 0 || ux >= size_x as i32 || uy >= size_y as i32 {
            return false;
        }
        get_flags(ux as usize, uy as usize) & CELLFLAG_LAND != 0
    };

    for py in 0..h {
        for px in 0..w {
            let ix = px as i32;
            let iy = py as i32;

            // Original: check unit adjacency for land before using neighbor points
            // p1=up (unit x,y-1), p2=right (unit x,y), p3=down (unit x,y), p4=left (unit x-1,y)
            let has_up = is_land(ix, iy - 1);
            let has_left = is_land(ix - 1, iy);
            let has_cur = is_land(ix, iy);

            let p0 = [
                px as f32 * GLOBAL_SCALE,
                py as f32 * GLOBAL_SCALE,
                get_z(px, py),
            ];

            // Vectors to neighbors (relative to p0)
            let v_up = if has_up && py > 0 {
                Some([0.0, -GLOBAL_SCALE, get_z(px, py - 1) - p0[2]])
            } else {
                None
            };

            let v_right = if has_cur && px < size_x {
                Some([GLOBAL_SCALE, 0.0, get_z(px + 1, py) - p0[2]])
            } else {
                None
            };

            let v_down = if has_cur && py < size_y {
                Some([0.0, GLOBAL_SCALE, get_z(px, py + 1) - p0[2]])
            } else {
                None
            };

            let v_left = if has_left && px > 0 {
                Some([-GLOBAL_SCALE, 0.0, get_z(px - 1, py) - p0[2]])
            } else {
                None
            };

            // Average cross products of adjacent triangle pairs
            let mut nx = 0.0f32;
            let mut ny = 0.0f32;
            let mut nz = 0.0f32;
            let mut cnt = 0;

            // cross(up, right)
            if let (Some(a), Some(b)) = (&v_up, &v_right) {
                let c = cross(a, b);
                let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if len > 0.0 {
                    nx += c[0] / len;
                    ny += c[1] / len;
                    nz += c[2] / len;
                    cnt += 1;
                }
            }
            // cross(right, down)
            if let (Some(a), Some(b)) = (&v_right, &v_down) {
                let c = cross(a, b);
                let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if len > 0.0 {
                    nx += c[0] / len;
                    ny += c[1] / len;
                    nz += c[2] / len;
                    cnt += 1;
                }
            }
            // cross(down, left)
            if let (Some(a), Some(b)) = (&v_down, &v_left) {
                let c = cross(a, b);
                let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if len > 0.0 {
                    nx += c[0] / len;
                    ny += c[1] / len;
                    nz += c[2] / len;
                    cnt += 1;
                }
            }
            // cross(left, up)
            if let (Some(a), Some(b)) = (&v_left, &v_up) {
                let c = cross(a, b);
                let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if len > 0.0 {
                    nx += c[0] / len;
                    ny += c[1] / len;
                    nz += c[2] / len;
                    cnt += 1;
                }
            }

            if cnt > 0 {
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                if len > 0.0 {
                    normals[py * w + px] = PointNormal {
                        x: nx / len,
                        y: ny / len,
                        z: nz / len,
                    };
                }
            }
        }
    }
    normals
}

/// Port of the `move/Data` buffer → `m_Move` expansion at
/// MatrixMapPrepare.cpp:1241-1290. Each `SCompileMoveCell` (4 ×
/// `SCompileMove`, 64 bytes) on the stored list holds the per-corner
/// move metadata for a map cell; we fan it out into the flat
/// `SizeMove.x × SizeMove.y` grid where `SizeMove = Size * 2`.
/// Returns `(move_cells, size_move_x, size_move_y)`.
fn load_move_cells(
    stor: &crate::matrix_lib::base::storage::Storage,
    points: &[CompilePoint],
    size_x: usize,
    size_y: usize,
) -> (Vec<MoveCell>, usize, usize) {
    let size_move_x = size_x * 2;
    let size_move_y = size_y * 2;
    let mut cells = vec![MoveCell::default(); size_move_x * size_move_y];

    let Some(buf) = stor.get_buf("move", "Data") else {
        log::warn!("map: move/Data block missing — pathfinder will treat every cell as impassable");
        return (cells, size_move_x, size_move_y);
    };
    let data = buf.get_bytes(0);

    // sizeof(SCompileMoveCell) = 4 × sizeof(SCompileMove) = 4 × 16 = 64
    const CELL_SIZE: usize = 64;
    let compiled_count = data.len() / CELL_SIZE;

    let read_move = |off: usize| -> (i32, u32, u32, u32) {
        let zone = i32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let stop = u32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        let sphere =
            u32::from_le_bytes([data[off + 8], data[off + 9], data[off + 10], data[off + 11]]);
        let zubchik = u32::from_le_bytes([
            data[off + 12],
            data[off + 13],
            data[off + 14],
            data[off + 15],
        ]);
        (zone, stop, sphere, zubchik)
    };

    let stride = size_x + 1;
    for y in 0..size_y {
        for x in 0..size_x {
            let mi = points[y * stride + x].move_idx;
            if mi < 0 || (mi as usize) >= compiled_count {
                continue;
            }
            let base_off = (mi as usize) * CELL_SIZE;
            // Four sub-cells c[0..3] at local positions:
            //   c[0] = (x*2,   y*2)
            //   c[1] = (x*2+1, y*2)
            //   c[2] = (x*2,   y*2+1)
            //   c[3] = (x*2+1, y*2+1)
            for (i, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].iter().enumerate() {
                let (zone, stop, sphere, zubchik) = read_move(base_off + i * 16);
                let mx = x * 2 + dx;
                let my = y * 2 + dy;
                if mx < size_move_x && my < size_move_y {
                    cells[my * size_move_x + mx] = MoveCell {
                        zone,
                        sphere,
                        zubchik,
                        stop,
                    };
                }
            }
        }
    }
    (cells, size_move_x, size_move_y)
}

/// Ports the `roads/Data` load step (MatrixMapPrepare.cpp:1599-1610):
/// the blob starts with a DWORD format version that must be 27, the
/// rest is the `CMatrixRoadNetwork::Load` stream; `InitPL` then builds
/// the place lookup grid over the move-cell grid.
fn load_road_network(
    stor: &crate::matrix_lib::base::storage::Storage,
    size_move_x: usize,
    size_move_y: usize,
) -> Result<Option<crate::matrix_game::logic::road_network::RoadNetwork>> {
    let Some(buf) = stor.get_buf("roads", "Data") else {
        return Ok(None);
    };
    if buf.arrays_count() == 0 {
        return Ok(None);
    }
    let data = buf.get_bytes(0);
    if data.is_empty() {
        return Ok(None);
    }
    if data.len() < 4 {
        bail!("roads/Data too small: {} bytes", data.len());
    }
    let ver = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if ver != 27 {
        // C++ ERROR_S "Please, recompile map with last editor..."
        bail!("roads/Data version {} unsupported (expected 27)", ver);
    }

    let mut rn = crate::matrix_game::logic::road_network::RoadNetwork::new();
    rn.load(&data[4..]).context("loading road network")?;
    rn.init_pl(size_move_x as i32, size_move_y as i32);
    log::info!(
        "map: road network: {} zones, {} crotches, {} roads, {} places, {} regions",
        rn.zones.len(),
        rn.crotches.len(),
        rn.roads.len(),
        rn.places.len(),
        rn.regions.len(),
    );
    Ok(Some(rn))
}

/// Convenience wrapper matching the `Option<Mutex<...>>` field shape.
fn wrap_road_network(
    rn: Option<crate::matrix_game::logic::road_network::RoadNetwork>,
) -> Option<std::sync::Mutex<crate::matrix_game::logic::road_network::RoadNetwork>> {
    rn.map(std::sync::Mutex::new)
}

fn compute_units(points: &[CompilePoint], size_x: usize, size_y: usize) -> Vec<MapUnit> {
    let stride = size_x + 1;
    let mut units = Vec::with_capacity(size_x * size_y);

    for y in 0..size_y {
        for x in 0..size_x {
            let p0 = points[y * stride + x];
            let p1 = points[y * stride + x + 1];
            let p3 = points[(y + 1) * stride + x];
            let p2 = points[(y + 1) * stride + x + 1];

            let flags = p0.flags;
            let flat = p0.z == p1.z && p0.z == p2.z && p0.z == p3.z;

            let (a1, b1, c1, a2, b2, c2) = if flat {
                (p0.z, 0.0, 0.0, p0.z, 0.0, 0.0)
            } else {
                (
                    (p1.z - p0.z) / GLOBAL_SCALE,
                    (p2.z - p1.z) / GLOBAL_SCALE,
                    p0.z,
                    (p2.z - p3.z) / GLOBAL_SCALE,
                    (p3.z - p0.z) / GLOBAL_SCALE,
                    p0.z,
                )
            };

            units.push(MapUnit {
                flags: if flat {
                    flags | CELLFLAG_FLAT
                } else {
                    flags & !CELLFLAG_FLAT
                },
                a1,
                b1,
                c1,
                a2,
                b2,
                c2,
            });
        }
    }

    units
}

/// Bridges pass (MatrixMapPrepare.cpp:1448-1532). Each `bridges/Data`
/// array is a CRect (left, top, right, bottom — 4×i32) followed by
/// `(right-left) * (bottom-top)` row-major deck heights. Every covered
/// unit gets `CELLFLAG_BRIDGE` and its plane coefficients rebuilt from
/// the deck z values (closed form of the `D3DXPlaneFromPoints` math,
/// same as `compute_units`); point z is overwritten so `GetZ` returns
/// deck height on bridge cells.
fn load_bridges(
    stor: &Storage,
    points: &mut [CompilePoint],
    units: &mut [MapUnit],
    size_x: usize,
    size_y: usize,
) {
    let Some(br) = stor.get_buf("bridges", "Data") else {
        return;
    };
    let stride = size_x + 1;
    for rec in 0..br.arrays_count() {
        let bytes = br.get_bytes(rec);
        if bytes.len() < 16 {
            continue;
        }
        let rd =
            |o: usize| i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        let (left, top, right, bottom) = (rd(0), rd(4), rd(8), rd(12));
        if left < 0 || top < 0 || right <= left || bottom <= top {
            continue;
        }
        let dx = (right - left) as usize;
        let dy = (bottom - top) as usize;
        let z = read_f32_array(&bytes[16..]);
        if right > size_x as i32 + 1 || bottom > size_y as i32 + 1 || z.len() < dx * dy {
            log::warn!("map: bridge record {rec} out of bounds; skipping");
            continue;
        }
        for y in top as usize..bottom as usize {
            for x in left as usize..right as usize {
                let zi = (y - top as usize) * dx + (x - left as usize);
                let z0 = z[zi];
                if x + 1 < right as usize && y + 1 < bottom as usize {
                    let u = &mut units[y * size_x + x];
                    u.flags |= CELLFLAG_BRIDGE;
                    let z1 = z[zi + 1];
                    let z2 = z[zi + dx + 1];
                    let z3 = z[zi + dx];
                    if z0 == z1 && z0 == z2 && z0 == z3 {
                        u.flags |= CELLFLAG_FLAT;
                        u.a1 = z0;
                    } else {
                        u.flags &= !CELLFLAG_FLAT;
                        u.a1 = (z1 - z0) / GLOBAL_SCALE;
                        u.b1 = (z2 - z1) / GLOBAL_SCALE;
                        u.c1 = z0;
                        u.a2 = (z2 - z3) / GLOBAL_SCALE;
                        u.b2 = (z3 - z0) / GLOBAL_SCALE;
                        u.c2 = z0;
                    }
                }
                points[y * stride + x].z = z0;
            }
        }
    }
    if br.arrays_count() > 0 {
        log::info!("map: applied {} bridge records", br.arrays_count());
    }
}

fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Per-group max land z, clamped to 0 (ports GetGroupMaxZLand + CMatrixMapGroup
/// max-z tracking; negative underwater groups report 0 like the original).
/// Collect each group's min/max heightmap z — CMatrixMapGroup::m_minz/m_maxz
/// in the original (MatrixMapGroup.cpp:254-346). Covers every corner of every
/// cell inside the group rectangle; empty groups (off-map edge fragments)
/// fall through as `(0, 0)`.
fn compute_group_bounds(
    points: &[CompilePoint],
    size_x: usize,
    size_y: usize,
    group_w: usize,
    group_h: usize,
) -> Vec<GroupBounds> {
    let gs = MAP_GROUP_SIZE as usize;
    let stride = size_x + 1;
    let mut out = Vec::with_capacity(group_w * group_h);
    for gy in 0..group_h {
        for gx in 0..group_w {
            let x0 = gx * gs;
            let y0 = gy * gs;
            let x1 = (x0 + gs).min(size_x);
            let y1 = (y0 + gs).min(size_y);
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for py in y0..=y1 {
                for px in x0..=x1 {
                    let z = points[py * stride + px].z;
                    if z < mn {
                        mn = z;
                    }
                    if z > mx {
                        mx = z;
                    }
                }
            }
            if !mn.is_finite() {
                mn = 0.0;
                mx = 0.0;
            }
            out.push(GroupBounds {
                min_z: mn,
                max_z: mx,
            });
        }
    }
    out
}

fn compute_group_max_z_land(
    points: &[CompilePoint],
    units: &[MapUnit],
    size_x: usize,
    size_y: usize,
) -> (Vec<f32>, usize, usize) {
    let gs = MAP_GROUP_SIZE as usize;
    let gw = size_x.div_ceil(gs);
    let gh = size_y.div_ceil(gs);
    let stride = size_x + 1;
    let mut out = vec![0.0f32; gw * gh];

    for gy in 0..gh {
        for gx in 0..gw {
            let x0 = gx * gs;
            let y0 = gy * gs;
            let x1 = (x0 + gs).min(size_x);
            let y1 = (y0 + gs).min(size_y);
            let mut mz = f32::NEG_INFINITY;
            for y in y0..y1 {
                for x in x0..x1 {
                    let u = units[y * size_x + x];
                    if u.flags & CELLFLAG_LAND == 0 {
                        continue;
                    }
                    for (dx, dy) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
                        let z = points[(y + dy) * stride + (x + dx)].z;
                        if z > mz {
                            mz = z;
                        }
                    }
                }
            }
            out[gy * gw + gx] = if mz.is_finite() { mz.max(0.0) } else { 0.0 };
        }
    }
    (out, gw, gh)
}

/// Parses DATA_OBJECTS columns from the CMAP (MatrixMapPrepare.cpp:513-576).
/// Each column is a flat buffer of N elements (one per object), accessed as row 0.
/// The `Height` column, when present, is an offset above interpolated terrain z at
/// the object's (x, y); otherwise `Z` is used directly.
/// Loads per-group shoreline wave spawn points from the CMAP's `inshores`
/// record (four parallel float columns `X`, `Y`, `NX`, `NY` — one array per
/// group, row-major). Ports MatrixMapPrepare.cpp:1386-1417 + the original
/// `CMatrixMapGroup::InitInshoreWaves` that stores (pos, dir) for later
/// wave spawning.
///
/// If any column is missing we return an empty list per group — the map
/// either has `DisableInshore` set or was built without inshore data.
fn load_inshore_prespawns(
    stor: &Storage,
    group_w: usize,
    group_h: usize,
    disable: bool,
) -> Vec<Vec<InshorePreSpawn>> {
    let total = group_w * group_h;
    let empty = vec![Vec::new(); total];
    // `DisableInshore` map property suppresses shoreline foam even when
    // the CMAP still carries baked spawn data (MatrixMapPrepare.cpp:1379-1387).
    if disable {
        return empty;
    }
    let (Some(bx), Some(by), Some(bnx), Some(bny)) = (
        stor.get_buf("inshores", "X"),
        stor.get_buf("inshores", "Y"),
        stor.get_buf("inshores", "NX"),
        stor.get_buf("inshores", "NY"),
    ) else {
        return empty;
    };
    if bx.arrays_count() < total {
        return empty;
    }

    let read_f32 = |b: &[u8], i: usize| -> f32 {
        f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
    };

    let mut out = Vec::with_capacity(total);
    for gi in 0..total {
        let xs = bx.get_bytes(gi);
        let ys = by.get_bytes(gi);
        let nxs = bnx.get_bytes(gi);
        let nys = bny.get_bytes(gi);
        let count = xs.len().min(ys.len()).min(nxs.len()).min(nys.len()) / 4;
        let mut list = Vec::with_capacity(count);
        for i in 0..count {
            list.push(InshorePreSpawn {
                pos: [read_f32(xs, i), read_f32(ys, i)],
                dir: [read_f32(nxs, i), read_f32(nys, i)],
            });
        }
        out.push(list);
    }
    out
}

fn load_objects(
    stor: &Storage,
    points: &[CompilePoint],
    normals: &[PointNormal],
    size_x: usize,
    size_y: usize,
) -> Vec<ObjectInstance> {
    let Some(col_x) = stor.get_buf("objects", "X") else {
        return Vec::new();
    };
    let Some(col_y) = stor.get_buf("objects", "Y") else {
        return Vec::new();
    };
    let Some(col_scale) = stor.get_buf("objects", "Scale") else {
        return Vec::new();
    };
    let Some(col_type) = stor.get_buf("objects", "Type") else {
        return Vec::new();
    };
    let Some(col_angle_z) = stor.get_buf("objects", "Angle") else {
        return Vec::new();
    };
    if col_x.arrays_count() == 0 {
        return Vec::new();
    }

    let xs = read_f32_array(col_x.get_bytes(0));
    let ys = read_f32_array(col_y.get_bytes(0));
    let n = xs.len().min(ys.len());
    if n == 0 {
        return Vec::new();
    }

    let scales = read_f32_array(col_scale.get_bytes(0));
    let angles_z = read_f32_array(col_angle_z.get_bytes(0));
    let types = read_u32_array(col_type.get_bytes(0));
    let angles_x = stor
        .get_buf("objects", "AngleX")
        .map(|b| read_f32_array(b.get_bytes(0)));
    let angles_y = stor
        .get_buf("objects", "AngleY")
        .map(|b| read_f32_array(b.get_bytes(0)));
    let heights = stor
        .get_buf("objects", "Height")
        .map(|b| read_f32_array(b.get_bytes(0)));
    let zs = stor
        .get_buf("objects", "Z")
        .map(|b| read_f32_array(b.get_bytes(0)));
    let shadows = stor.get_buf("objects", "Shadow");

    let w = size_x + 1;
    let sample_corners = |x: f32, y: f32| -> f32 {
        let px = (x / GLOBAL_SCALE) as i32;
        let py = (y / GLOBAL_SCALE) as i32;
        let mut sum = 0.0f32;
        let mut cnt = 0u32;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let cx = px + dx;
            let cy = py + dy;
            if cx < 0 || cy < 0 || cx > size_x as i32 || cy > size_y as i32 {
                continue;
            }
            sum += points[cy as usize * w + cx as usize].z;
            cnt += 1;
        }
        if cnt == 0 {
            0.0
        } else {
            sum / cnt as f32
        }
    };

    (0..n)
        .map(|i| {
            let x = xs[i];
            let y = ys[i];
            let z = match (&heights, &zs) {
                (Some(h), _) if i < h.len() => h[i] + sample_corners(x, y),
                (_, Some(zv)) if i < zv.len() => zv[i],
                _ => 0.0,
            };
            ObjectInstance {
                x,
                y,
                z,
                angle_z: angles_z.get(i).copied().unwrap_or(0.0),
                angle_x: angles_x
                    .as_ref()
                    .and_then(|a| a.get(i).copied())
                    .unwrap_or(0.0),
                angle_y: angles_y
                    .as_ref()
                    .and_then(|a| a.get(i).copied())
                    .unwrap_or(0.0),
                scale: scales.get(i).copied().unwrap_or(1.0),
                type_id: types.get(i).copied().unwrap_or(0),
                shadow: shadows
                    .and_then(|buf| parse_object_shadow(buf, i, points, normals, size_x, size_y)),
            }
        })
        .collect()
}

/// Ports the `RS_BUILDINGS` step of `CMatrixMap::ReloadDynamics`
/// (MatrixMapPrepare.cpp:621-694). We skip the `side==255` ruin-replacement
/// branch: those aren't live buildings, they spawn static ruin meshes via
/// `InitAsBaseRuins`, which the port doesn't wire up yet.
fn load_buildings(
    stor: &Storage,
    units: &[MapUnit],
    size_x: usize,
    size_y: usize,
    ruins: &mut Vec<RuinInstance>,
) -> Vec<BuildingInstance> {
    let (Some(cx), Some(cy), Some(cside), Some(ckind), Some(cshadow), Some(cshsz), Some(cang)) = (
        stor.get_buf("buildings", "X"),
        stor.get_buf("buildings", "Y"),
        stor.get_buf("buildings", "Side"),
        stor.get_buf("buildings", "Kind"),
        stor.get_buf("buildings", "Shadow"),
        stor.get_buf("buildings", "ShadowSize"),
        stor.get_buf("buildings", "Angle"),
    ) else {
        return Vec::new();
    };
    if cx.arrays_count() == 0 {
        return Vec::new();
    }

    let xs = read_f32_array(cx.get_bytes(0));
    let ys = read_f32_array(cy.get_bytes(0));
    let sides = cside.get_bytes(0);
    let kinds = ckind.get_bytes(0);
    let shadows = cshadow.get_bytes(0);
    let shszs = read_i32_array(cshsz.get_bytes(0));
    let angles = cang.get_bytes(0);

    let n = xs.len().min(ys.len()).min(sides.len());

    // Floor-z lookup matches `CMatrixMap::GetZ` (MatrixMap.cpp:1020-1062) as
    // pointed to from `CMatrixBuilding::OnLoad` (MatrixObjectBuilding.cpp:
    // 1029). Below WATER_LEVEL the original traces down from z=1000 until it
    // hits the geometry; we defer that to the caller or just clamp to 0.
    let get_z = |wx: f32, wy: f32| -> f32 {
        let sx = wx / GLOBAL_SCALE;
        let sy = wy / GLOBAL_SCALE;
        let ix = sx.floor() as i32;
        let iy = sy.floor() as i32;
        if ix < 0 || iy < 0 || ix >= size_x as i32 || iy >= size_y as i32 {
            return 0.0;
        }
        let u = units[iy as usize * size_x + ix as usize];
        if u.flags & crate::matrix_game::common::CELLFLAG_FLAT != 0 {
            return u.a1;
        }
        let lx = wx - ix as f32 * GLOBAL_SCALE;
        let ly = wy - iy as f32 * GLOBAL_SCALE;
        if ly < lx {
            u.a1 * lx + u.b1 * ly + u.c1
        } else {
            u.a2 * lx + u.b2 * ly + u.c2
        }
    };

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let side = *sides.get(i).unwrap_or(&0);
        let x = xs[i];
        let y = ys[i];
        let build_z = get_z(x, y).max(0.0);
        if side == 255 {
            // Decorative pre-placed ruin (MatrixMapPrepare.cpp:648-668):
            // spawned as ruin-mesh objects, not a live building.
            ruins.push(RuinInstance {
                x,
                y,
                build_z,
                angle: *angles.get(i).unwrap_or(&0) as i32,
                kind: *kinds.get(i).unwrap_or(&0) as i32,
            });
            continue;
        }
        out.push(BuildingInstance {
            x,
            y,
            build_z,
            angle: *angles.get(i).unwrap_or(&0),
            side,
            kind: *kinds.get(i).unwrap_or(&0),
            shadow_kind: *shadows.get(i).unwrap_or(&0),
            shadow_size: shszs.get(i).copied().unwrap_or(128),
            turrets_places_cnt: 0,
            turret_places: Vec::new(),
        });
    }

    // Port of MatrixMapPrepare.cpp:776-897 — scan `cannons` records and
    // attach each cannon (prop=1 or 2) to the nearest building within
    // `MAX_DISTANCE_CANNON_BUILDING` (500 world units / `GLOBAL_SCALE`).
    // Each attachment increments that building's `m_TurretsPlacesCnt`,
    // which later clamps `m_TurretsMax = min(4, cnt)` so podl1..4 show
    // the right slot count.
    out
}

/// Port of MatrixMapPrepare.cpp:776-885 — read every CMAP `cannons/*`
/// record, attach factory + place-only entries to the nearest base, and
/// return the full record list. Mutates `buildings` to record per-base
/// turret-slot tables and per-base `turrets_have` for pre-mounted
/// factory cannons.
fn load_cannons(stor: &Storage, buildings: &mut [BuildingInstance]) -> Vec<CannonInstance> {
    let Some(cxb) = stor.get_buf("cannons", "X") else {
        return Vec::new();
    };
    let Some(cyb) = stor.get_buf("cannons", "Y") else {
        return Vec::new();
    };
    let cx_arr = read_f32_array(cxb.get_bytes(0));
    let cy_arr = read_f32_array(cyb.get_bytes(0));
    let n = cx_arr.len().min(cy_arr.len());
    if n == 0 {
        return Vec::new();
    }
    let sides: Vec<u8> = stor
        .get_buf("cannons", "Side")
        .map(|b| b.get_bytes(0).to_vec())
        .unwrap_or_else(|| vec![0; n]);
    let kinds: Vec<u8> = stor
        .get_buf("cannons", "Kind")
        .map(|b| b.get_bytes(0).to_vec())
        .unwrap_or_else(|| vec![1; n]);
    let angles: Vec<f32> = stor
        .get_buf("cannons", "Angle")
        .map(|b| read_f32_array(b.get_bytes(0)))
        .unwrap_or_else(|| vec![0.0; n]);
    let addh: Vec<f32> = stor
        .get_buf("cannons", "AddH")
        .map(|b| read_f32_array(b.get_bytes(0)))
        .unwrap_or_else(|| vec![0.0; n]);
    // `Prop` is optional (MatrixMapPrepare.cpp:794) — when absent every
    // record is treated as standalone (prop=0).
    let props: Vec<i32> = stor
        .get_buf("cannons", "Prop")
        .map(|b| read_i32_array(b.get_bytes(0)))
        .unwrap_or_else(|| vec![0; n]);

    const MAX_PLACES: i32 = 4;
    const MAX_DIST: f32 = 500.0;
    const MAX_DIST_SQ: f32 = MAX_DIST * MAX_DIST;

    let mut out: Vec<CannonInstance> = Vec::with_capacity(n);
    for i in 0..n {
        let cxv = cx_arr[i];
        let cyv = cy_arr[i];
        let mut prop = props.get(i).copied().unwrap_or(0);
        let kind = kinds.get(i).copied().unwrap_or(1);
        let side = sides.get(i).copied().unwrap_or(0);
        let angle = angles.get(i).copied().unwrap_or(0.0);
        let add_h = addh.get(i).copied().unwrap_or(0.0);

        let mut parent_building: Option<usize> = None;
        let mut parent_slot: i32 = -1;
        let mut owning_side = side;

        if prop == 1 || prop == 2 {
            // Nearest-base scan — port of MatrixMapPrepare.cpp:826-845.
            let mut best: Option<usize> = None;
            let mut best_d2 = f32::INFINITY;
            for (bi, b) in buildings.iter().enumerate() {
                if b.turrets_places_cnt >= MAX_PLACES {
                    continue;
                }
                let dx = cxv - b.x;
                let dy = cyv - b.y;
                let d2 = dx * dx + dy * dy;
                if d2 < best_d2 && d2 < MAX_DIST_SQ {
                    best_d2 = d2;
                    best = Some(bi);
                }
            }
            if let Some(bi) = best {
                let bld = &mut buildings[bi];
                let coord_x = float2int(cxv / GameMap::GLOBAL_SCALE_MOVE);
                let coord_y = float2int(cyv / GameMap::GLOBAL_SCALE_MOVE);
                let cannon_type = if prop == 1 { kind as i32 } else { -1 };
                bld.turret_places.push(TurretPlaceCmap {
                    coord: (coord_x, coord_y),
                    angle,
                    cannon_type,
                });
                parent_slot = bld.turrets_places_cnt;
                bld.turrets_places_cnt += 1;
                parent_building = Some(bi);
                // Inherit the parent base's side (MatrixMapPrepare.cpp:850).
                owning_side = bld.side;
            } else if prop == 1 {
                // No building in range — demote to an empty slot
                // (MatrixMapPrepare.cpp:842-845).
                prop = 2;
            }
        }

        // Skip prop=2 records — empty placement slots and orphaned
        // factory cannons spawn no Cannon entity (MatrixMapPrepare.cpp:
        // 863-873).
        if prop == 2 {
            continue;
        }

        out.push(CannonInstance {
            x: cxv,
            y: cyv,
            angle,
            kind,
            side: owning_side,
            add_h,
            prop,
            parent_building,
            parent_slot,
        });
    }
    out
}

/// Port of `RS_ROBOTS` step (MatrixMapPrepare.cpp:697-770) — read the
/// `robots/{X,Y,Side,Shadow,ShadowSize,Group,Units}` columns and parse
/// each row's `Units` wstr ("|"-separated tokens of "type,kind[,angle]")
/// into a [`RobotInstance`]. The chassis token's optional third field
/// is the spawn angle in radians.
///
/// `MRT_*` numeric values come from `ERobotUnitType`
/// (MatrixObjectRobot.hpp:47-55):
///   1=Chassis, 2=Weapon, 3=Armor, 4=Head.
fn load_robots(
    stor: &Storage,
    units: &[MapUnit],
    size_x: usize,
    size_y: usize,
) -> Vec<RobotInstance> {
    let Some(cx) = stor.get_buf("robots", "X") else {
        return Vec::new();
    };
    let Some(cy) = stor.get_buf("robots", "Y") else {
        return Vec::new();
    };
    let Some(cside) = stor.get_buf("robots", "Side") else {
        return Vec::new();
    };
    let Some(cshadow) = stor.get_buf("robots", "Shadow") else {
        return Vec::new();
    };
    let Some(cshsz) = stor.get_buf("robots", "ShadowSize") else {
        return Vec::new();
    };
    let Some(cgroup) = stor.get_buf("robots", "Group") else {
        return Vec::new();
    };
    let Some(cunits) = stor.get_buf("robots", "Units") else {
        return Vec::new();
    };

    if cx.arrays_count() == 0 {
        return Vec::new();
    }

    let xs = read_f32_array(cx.get_bytes(0));
    let ys = read_f32_array(cy.get_bytes(0));
    let sides = cside.get_bytes(0);
    let shadows = cshadow.get_bytes(0);
    let shszs = read_i32_array(cshsz.get_bytes(0));
    let groups = read_i32_array(cgroup.get_bytes(0));

    let n = xs.len().min(ys.len());
    if n == 0 {
        return Vec::new();
    }

    // Mirror of `CMatrixMap::GetZ` (MatrixMap.cpp:1020-1062) — same
    // floor-z resolver used by `load_buildings`. Returns 0 below water
    // (the C++ traces from z=1000 down for submerged cells; we clamp).
    let get_z = |wx: f32, wy: f32| -> f32 {
        let sx = wx / GLOBAL_SCALE;
        let sy = wy / GLOBAL_SCALE;
        let ix = sx.floor() as i32;
        let iy = sy.floor() as i32;
        if ix < 0 || iy < 0 || ix >= size_x as i32 || iy >= size_y as i32 {
            return 0.0;
        }
        let u = units[iy as usize * size_x + ix as usize];
        if u.flags & crate::matrix_game::common::CELLFLAG_FLAT != 0 {
            return u.a1;
        }
        let lx = wx - ix as f32 * GLOBAL_SCALE;
        let ly = wy - iy as f32 * GLOBAL_SCALE;
        if ly < lx {
            u.a1 * lx + u.b1 * ly + u.c1
        } else {
            u.a2 * lx + u.b2 * ly + u.c2
        }
    };

    const MRT_CHASSIS: i32 = 1;
    const MRT_WEAPON: i32 = 2;
    const MRT_ARMOR: i32 = 3;
    const MRT_HEAD: i32 = 4;
    const W_CNT: usize = crate::matrix_game::object_robot::MAX_WEAPON_CNT;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let units_str = cunits.get_as_wstr(i);

        let mut chassis_kind = RobotUnitKind(0);
        let mut armor_kind = RobotUnitKind(0);
        let mut head_kind = RobotUnitKind(0);
        let mut weapons = [RobotUnitKind(0); W_CNT];
        let mut wi = 0usize;
        let mut angle = 0.0f32;

        for token in units_str.split('|') {
            if token.is_empty() {
                continue;
            }
            let mut parts = token.split(',');
            let Some(type_s) = parts.next() else { continue };
            let Some(kind_s) = parts.next() else { continue };
            let ty = type_s.trim().parse::<i32>().unwrap_or(0);
            let kind = kind_s.trim().parse::<i32>().unwrap_or(0);
            match ty {
                MRT_CHASSIS => {
                    chassis_kind = RobotUnitKind(kind);
                    if let Some(angle_s) = parts.next() {
                        angle = angle_s.trim().parse::<f32>().unwrap_or(0.0);
                    }
                }
                MRT_ARMOR => armor_kind = RobotUnitKind(kind),
                MRT_HEAD => head_kind = RobotUnitKind(kind),
                MRT_WEAPON => {
                    if wi < W_CNT {
                        weapons[wi] = RobotUnitKind(kind);
                        wi += 1;
                    }
                }
                _ => {}
            }
        }

        let x = xs[i];
        let y = ys[i];
        out.push(RobotInstance {
            x,
            y,
            z: get_z(x, y),
            side: *sides.get(i).unwrap_or(&0),
            shadow: *shadows.get(i).unwrap_or(&0),
            shadow_size: shszs.get(i).copied().unwrap_or(128),
            group: groups.get(i).copied().unwrap_or(-1),
            angle,
            chassis_kind,
            armor_kind,
            head_kind,
            weapons,
        });
    }
    out
}

fn read_i32_array(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn parse_object_shadow(
    buf: &crate::matrix_lib::base::storage::DataBuf,
    index: usize,
    points: &[CompilePoint],
    normals: &[PointNormal],
    size_x: usize,
    size_y: usize,
) -> Option<ObjectShadow> {
    if index >= buf.arrays_count() {
        return None;
    }
    let bytes = buf.get_bytes(index);
    if bytes.len() < 4 {
        return None;
    }

    let mut off = 0usize;
    let group_count = read_i32_le(bytes, &mut off)? as usize;
    let group_bytes = group_count.checked_mul(4)?;
    if off + group_bytes > bytes.len() {
        return None;
    }
    off += group_bytes;

    let _min_x = read_i32_le(bytes, &mut off)?;
    let _min_y = read_i32_le(bytes, &mut off)?;
    let vert_count = read_i32_le(bytes, &mut off)? as usize;
    let index_byte_size = read_i32_le(bytes, &mut off)? as usize;
    let index_count = index_byte_size / 2;

    let vertex_bytes = vert_count.checked_mul(12)?;
    let index_bytes = index_byte_size;
    if off + vertex_bytes + index_bytes + 20 > bytes.len() {
        return None;
    }

    let mut vertices = Vec::with_capacity(vert_count);
    for _ in 0..vert_count {
        let px = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as usize;
        let py = u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]) as usize;
        let tu = f32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]);
        let tv = f32::from_le_bytes([
            bytes[off + 8],
            bytes[off + 9],
            bytes[off + 10],
            bytes[off + 11],
        ]);
        off += 12;

        if px > size_x || py > size_y {
            return None;
        }
        let p = points[py * (size_x + 1) + px];
        let n = normals[py * (size_x + 1) + px];
        vertices.push(ObjectShadowVertex {
            position: [
                px as f32 * GLOBAL_SCALE + n.x * 0.1,
                py as f32 * GLOBAL_SCALE + n.y * 0.1,
                p.z + n.z * 0.1,
            ],
            uv: [tu, tv],
        });
    }

    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        indices.push(u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32);
        off += 2;
    }

    let camera_pos = [
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]),
        f32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]),
        f32::from_le_bytes([
            bytes[off + 8],
            bytes[off + 9],
            bytes[off + 10],
            bytes[off + 11],
        ]),
    ];
    off += 12;
    let dimensions = [
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]),
        f32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]),
    ];

    Some(ObjectShadow {
        vertices,
        indices,
        camera_pos,
        dimensions,
    })
}

fn read_i32_le(bytes: &[u8], off: &mut usize) -> Option<i32> {
    if *off + 4 > bytes.len() {
        return None;
    }
    let value = i32::from_le_bytes([
        bytes[*off],
        bytes[*off + 1],
        bytes[*off + 2],
        bytes[*off + 3],
    ]);
    *off += 4;
    Some(value)
}

fn read_f32_array(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32_array(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn blend_color(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |ca: u32, cb: u32| -> u32 {
        ((ca as f32) + ((cb as f32) - (ca as f32)) * t)
            .round()
            .clamp(0.0, 255.0) as u32
    };
    let ar = (a >> 16) & 0xFF;
    let ag = (a >> 8) & 0xFF;
    let ab = a & 0xFF;
    let br = (b >> 16) & 0xFF;
    let bg = (b >> 8) & 0xFF;
    let bb = b & 0xFF;
    (lerp(ar, br) << 16) | (lerp(ag, bg) << 8) | lerp(ab, bb)
}

/// Parse the `SideResInfo` property: `|`-separated per-side entries,
/// each `id,res0,res1,res2,res3[,forceUp]` (MatrixMapPrepare.cpp:471-493).
/// Parse the `SideAIInfo` map property (MatrixMapPrepare.cpp:420-458):
/// `Name:KEY/VAL/KEY/VAL|Name2:...` — side names resolve to ids via the
/// `da/Side` data block at apply time.
fn parse_side_ai_info(s: &str) -> Vec<(String, Vec<(String, String)>)> {
    let mut out = Vec::new();
    for entry in s.split('|') {
        let Some((name, kvs)) = entry.split_once(':') else {
            continue;
        };
        let parts: Vec<&str> = kvs.split('/').collect();
        let pairs: Vec<(String, String)> = parts
            .chunks_exact(2)
            .map(|c| (c[0].trim().to_string(), c[1].trim().to_string()))
            .collect();
        out.push((name.trim().to_string(), pairs));
    }
    out
}

fn parse_side_res_info(s: &str) -> Vec<(i32, [i32; 4], i32)> {
    let mut out = Vec::new();
    for entry in s.split('|') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let f: Vec<i32> = entry.split(',').map(|p| p.trim().parse().unwrap_or(0)).collect();
        if f.len() < 5 {
            continue;
        }
        let force_up = if f.len() > 5 { f[5] } else { 100 };
        out.push((f[0], [f[1], f[2], f[3], f[4]], force_up));
    }
    out
}

fn find_property_int(
    names: &crate::matrix_lib::base::storage::DataBuf,
    values: &crate::matrix_lib::base::storage::DataBuf,
    key: &str,
) -> Result<i32> {
    let idx = names
        .find_as_wstr(key)
        .with_context(|| format!("property '{key}' not found"))?;
    let val_str = values.get_as_wstr(idx);
    val_str
        .trim()
        .parse::<i32>()
        .with_context(|| format!("property '{key}' not a valid int: '{val_str}'"))
}

fn find_property_float(
    names: &crate::matrix_lib::base::storage::DataBuf,
    values: &crate::matrix_lib::base::storage::DataBuf,
    key: &str,
) -> Result<f32> {
    let idx = names
        .find_as_wstr(key)
        .with_context(|| format!("property '{key}' not found"))?;
    let val_str = values.get_as_wstr(idx);
    val_str
        .trim()
        .parse::<f32>()
        .with_context(|| format!("property '{key}' not a valid float: '{val_str}'"))
}
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4], // x = fog_start, y = fog_end
}

/// Uniforms for the gloss pass. Carries view matrix so the vertex shader can
/// transform normals to camera space (matching D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GlossUniforms {
    view_proj: [[f32; 4]; 4],
    normal_mat: [[f32; 4]; 4],
    /// Camera view matrix — the fragment shader needs camera-space
    /// POSITION to build the true reflection vector R = 2(N·E)N − E
    /// (D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR).
    view: [[f32; 4]; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
}

pub struct MapRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Depth-only pipeline for the `-1` texture sentinel. See
    /// MatrixMapGroup.cpp:593-606.
    depth_only_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    gloss_pipeline: wgpu::RenderPipeline,
    batches: Vec<DrawBatch>,
    depth_only_batches: Vec<DrawBatch>,
    overlay_batches: Vec<OverlayBatch>,
    gloss_batches: Vec<GlossOverlayBatch>,
    sky: Sky,
    clear_color: wgpu::Color,
    fog_color: [f32; 4],
    objects: Option<super::object::ObjectsRenderer>,
    buildings: Option<super::object_building::BuildingsRenderer>,
    robots: Option<super::object_robot::RobotsRenderer>,
    cannons: Option<super::object_cannon::CannonsRenderer>,
    slot_markers: Option<super::slot_marker::SlotMarkerRenderer>,
    point_lights: PointLightRenderer,
    water: Option<super::water::Water>,
    uniform_buffer: wgpu::Buffer,
    gloss_uniform_buffer: wgpu::Buffer,
    depth_texture: wgpu::TextureView,
    last_point_light_revision: u64,
    /// Stencil shadow volume accumulator + darken pass — the
    /// `CMatrixMap::DrawShadows` composition (MatrixMap.cpp:1865).
    stencil_shadows: super::shadow::StencilShadowRenderer,
}

impl MapRenderer {
    /// Accessor for the chassis renderer — external passes (like the
    /// constructor 3D preview) use `robots.render_preview` directly.
    pub fn robots(&self) -> Option<&super::object_robot::RobotsRenderer> {
        self.robots.as_ref()
    }

    /// Shared depth buffer view — used by external effect renderers
    /// (e.g. the selection ring) that want to depth-test against the
    /// terrain drawn in the same frame without re-attaching a
    /// separate depth texture.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_texture
    }
}

impl MapRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        stor: &Storage,
        matrix_data: &Storage,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Self {
        let tex_union_dim = map.tex_union_dim;
        let atlas_size = TEX_BOTTOM_SIZE * tex_union_dim;
        let ts_inv = 1.0 / atlas_size as f64;

        // BuildTexUnions (MatrixMapPrepare.cpp:108)
        let atlas_views = build_tex_union_atlases(device, queue, stor, tex_union_dim, read_texture);
        log::info!(
            "built {} texture union atlases ({}x{})",
            atlas_views.len(),
            atlas_size,
            atlas_size
        );

        // BuildBottom (MatrixMapGroup.cpp:231) — see `map_group::build_bottom_cpu_batches`.
        let batches_by_tex =
            crate::matrix_game::map_group::build_bottom_cpu_batches(stor, map, ts_inv);
        log::info!("parsed {} group batches", batches_by_tex.len());

        let [sr, sg, sb] = unpack_rgb(map.sky_color);
        let fog_color = [sr, sg, sb, 1.0];

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Uniforms"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let surface_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let sampler_wrap_v = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        // Load macrotexture. When the map has no `MacroTexture` property set
        // (training.cmap leaves it blank) we must not paint a gray/semi-opaque
        // placeholder — the terrain shader blends `macro.rgb*macro.a +
        // atlas.rgb*(1-macro.a)`, so any non-zero macro alpha washes the
        // ground toward the placeholder color. Alpha=0 makes the stage a
        // no-op, mirroring the original's behavior of simply not binding a
        // macrotexture in that case.
        let macro_view = if let Some(data) = map
            .macro_texture_path
            .as_deref()
            .and_then(read_texture)
            .or_else(|| read_texture("macrotexture"))
        {
            if let Some(rgba) = decode_texture_bytes(&data) {
                log::info!(
                    "terrain: loaded macrotexture ({}x{})",
                    rgba.width(),
                    rgba.height()
                );
                create_texture_from_rgba_mipped(device, queue, &rgba, 6)
            } else {
                log::warn!("terrain: macrotexture decode failed; disabling macro stage");
                create_solid_texture(device, queue, [0, 0, 0, 0])
            }
        } else {
            log::info!("terrain: no macrotexture for this map; disabling macro stage");
            create_solid_texture(device, queue, [0, 0, 0, 0])
        };

        let macro_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain BGL"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let transparent = create_solid_texture(device, queue, [0, 0, 0, 0]);

        let mut batches = Vec::new();
        let mut depth_only_batches = Vec::new();
        let mut total_tris = 0u32;
        // Some batches bind the atlas and write color; the -1 sentinel batch
        // just needs geometry — no atlas sample. Reuse the first available
        // atlas view for the sentinel bind group so the layout is satisfied.
        let sentinel_atlas = atlas_views.first();
        for (tex_idx, (verts, idxs, point_coords)) in &batches_by_tex {
            if idxs.is_empty() {
                continue;
            }
            let is_sentinel = *tex_idx == u32::MAX;
            let tex_view = if is_sentinel {
                match sentinel_atlas {
                    Some(v) => v,
                    None => {
                        log::warn!("terrain: -1 batch with no atlases available; skipping");
                        continue;
                    }
                }
            } else {
                match atlas_views.get(*tex_idx as usize) {
                    Some(v) => v,
                    None => {
                        log::warn!(
                            "terrain: batch texture index {} out of range (have {} atlases); skipping",
                            tex_idx,
                            atlas_views.len()
                        );
                        continue;
                    }
                }
            };
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(idxs),
                usage: wgpu::BufferUsages::INDEX,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(tex_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&macro_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&macro_sampler),
                    },
                ],
            });
            total_tris += idxs.len() as u32 / 3;
            let batch = DrawBatch {
                bind_group: bg,
                vertex_buffer: vb,
                index_buffer: ib,
                num_indices: idxs.len() as u32,
                index_format: wgpu::IndexFormat::Uint32,
                cpu_vertices: Some(verts.clone()),
                point_coords: Some(point_coords.clone()),
            };
            if is_sentinel {
                depth_only_batches.push(batch);
            } else {
                batches.push(batch);
            }
        }
        log::info!(
            "terrain bottom: {} draw batches, {} triangles",
            batches.len(),
            total_tris
        );

        // Fill cells that have NO bottom geometry at all (authored pits
        // under bases / factories / ruin sites) with flat vertex-colored
        // quads. The C++ leaves these pixels undrawn and relies on never
        // clearing the color target in release (MatrixFormGame.cpp:149),
        // so they self-heal with the previous frame's contents; our WebGL
        // swapchain has no stable previous frame, and the gaps showed as
        // sky-colored "unpainted" patches wherever a building mesh or its
        // projected shadow didn't fully cover the hole. Water cells are
        // excluded (the water passes own them); depth-only sentinel cells
        // count as covered (their carve is intentional).
        {
            let cx = map.world_width() * 0.5;
            let cy = map.world_height() * 0.5;
            // A cell counts as covered when the bottom triangles blanket
            // (nearly) all of a 4x4 sample grid inside it — exact enough
            // for the cell-quad geometry and robust against the down-cell
            // -normal*0.5 nudges and cliff-face triangles.
            let sx = map.size_x;
            let mut samples = vec![0u16; sx * map.size_y];
            // Per covered cell: the atlas tile (page + uv rect) drawn there,
            // so filler cells can borrow the nearest neighbor's tile and
            // read as continued ground instead of a flat color patch.
            let mut cell_tile: Vec<Option<(u32, [f32; 2], [f32; 2])>> =
                vec![None; sx * map.size_y];
            // The -1 sentinel batch (depth-only carve) draws no color, so
            // its cells still need visible fill — don't count it.
            for (&tex, (verts, idxs, _)) in
                batches_by_tex.iter().filter(|(&tex, _)| tex != u32::MAX)
            {
                for t in idxs.chunks_exact(3) {
                    let p: Vec<[f32; 2]> = t
                        .iter()
                        .map(|&i| {
                            let v = &verts[i as usize].position;
                            [v[0] + cx, v[1] + cy]
                        })
                        .collect();
                    let (x0, x1) = (
                        p.iter().map(|q| q[0]).fold(f32::MAX, f32::min),
                        p.iter().map(|q| q[0]).fold(f32::MIN, f32::max),
                    );
                    let (y0, y1) = (
                        p.iter().map(|q| q[1]).fold(f32::MAX, f32::min),
                        p.iter().map(|q| q[1]).fold(f32::MIN, f32::max),
                    );
                    let step = GLOBAL_SCALE / 4.0;
                    let gx0 = ((x0 / step).floor() as i32).max(0);
                    let gx1 = ((x1 / step).ceil() as i32).min((sx * 4) as i32 - 1);
                    let gy0 = ((y0 / step).floor() as i32).max(0);
                    let gy1 = ((y1 / step).ceil() as i32).min((map.size_y * 4) as i32 - 1);
                    let edge = |a: &[f32; 2], b: &[f32; 2], px: f32, py: f32| {
                        (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
                    };
                    for gy in gy0..=gy1 {
                        for gx in gx0..=gx1 {
                            let px = (gx as f32 + 0.5) * step;
                            let py = (gy as f32 + 0.5) * step;
                            let e0 = edge(&p[0], &p[1], px, py);
                            let e1 = edge(&p[1], &p[2], px, py);
                            let e2 = edge(&p[2], &p[0], px, py);
                            if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0)
                                || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)
                            {
                                let cell = (gy / 4) as usize * sx + (gx / 4) as usize;
                                let bit = ((gy % 4) * 4 + (gx % 4)) as u16;
                                samples[cell] |= 1 << bit;
                            }
                        }
                    }
                    // Record / expand this cell's tile uv rect (the cell's
                    // two triangles together span the full tile).
                    let ccx = ((p[0][0] + p[1][0] + p[2][0]) / 3.0 / GLOBAL_SCALE) as i32;
                    let ccy = ((p[0][1] + p[1][1] + p[2][1]) / 3.0 / GLOBAL_SCALE) as i32;
                    if ccx >= 0 && ccy >= 0 && (ccx as usize) < sx && (ccy as usize) < map.size_y
                    {
                        let slot = &mut cell_tile[ccy as usize * sx + ccx as usize];
                        let mut u0 = [f32::MAX; 2];
                        let mut u1 = [f32::MIN; 2];
                        for &i in t {
                            let uv = verts[i as usize].uv;
                            u0[0] = u0[0].min(uv[0]);
                            u0[1] = u0[1].min(uv[1]);
                            u1[0] = u1[0].max(uv[0]);
                            u1[1] = u1[1].max(uv[1]);
                        }
                        match slot {
                            Some((stex, s0, s1)) if *stex == tex => {
                                s0[0] = s0[0].min(u0[0]);
                                s0[1] = s0[1].min(u0[1]);
                                s1[0] = s1[0].max(u1[0]);
                                s1[1] = s1[1].max(u1[1]);
                            }
                            Some(_) => {}
                            None => *slot = Some((tex, u0, u1)),
                        }
                    }
                }
            }
            let cell_covered: Vec<bool> = samples
                .iter()
                .map(|&m| m.count_ones() >= 14)
                .collect();

            // Compiled-mesh z per heightmap point, from vertices actually
            // referenced by textured triangles (cliff faces stack several
            // verts per point — keep the top edge). Filler corners snap to
            // these so the quads seam with the real mesh.
            let mut real_z: std::collections::HashMap<(usize, usize), f32> =
                std::collections::HashMap::new();
            for (verts, idxs, coords) in batches_by_tex
                .iter()
                .filter(|(&tex, _)| tex != u32::MAX)
                .map(|(_, b)| b)
            {
                for &i in idxs {
                    let z = verts[i as usize].position[2];
                    real_z
                        .entry(coords[i as usize])
                        .and_modify(|m| *m = m.max(z))
                        .or_insert(z);
                }
            }

            // Lowest surface-overlay z per cell (center-sampled). Hole cells
            // can keep heightfield mounds the compiled mesh omits; whether
            // the filler may follow one depends on what's above: under an
            // overlay it must stay below the overlay plane (training hides a
            // z≈2.3 mound under its flat base platform — following it pokes
            // dirt patches through), while an uncovered mound is scenery
            // (Armageddon's snow mesa under the landing pad).
            let mut overlay_floor: Vec<Option<f32>> = vec![None; sx * map.size_y];
            for (name, vert_size) in [("surfacesM", 32usize), ("surfaces", 24)] {
                let Some(buf) = stor.get_buf(name, "Data") else {
                    continue;
                };
                for i in 0..buf.arrays_count() {
                    let raw = buf.get_bytes(i);
                    if raw.len() < 32 {
                        continue;
                    }
                    let rd_u32 =
                        |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
                    let rd_f32 =
                        |o: usize| f32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
                    let vcnt = rd_u32(12) as usize;
                    let idxsz = rd_u32(16) as usize;
                    let disp_x = rd_f32(24);
                    let disp_y = rd_f32(28);
                    let vend = 32 + vcnt * vert_size;
                    if vend + idxsz > raw.len() {
                        continue;
                    }
                    let pt = |vi: usize| {
                        [
                            rd_f32(32 + vi * vert_size) + disp_x,
                            rd_f32(32 + vi * vert_size + 4) + disp_y,
                            rd_f32(32 + vi * vert_size + 8),
                        ]
                    };
                    let idx: Vec<usize> = (0..idxsz / 2)
                        .map(|k| {
                            u16::from_le_bytes(
                                raw[vend + k * 2..vend + k * 2 + 2].try_into().unwrap(),
                            ) as usize
                        })
                        .collect();
                    for w in idx.windows(3) {
                        if w[0] == w[1]
                            || w[1] == w[2]
                            || w[0] == w[2]
                            || w.iter().any(|&v| v >= vcnt)
                        {
                            continue;
                        }
                        let (a, b, c) = (pt(w[0]), pt(w[1]), pt(w[2]));
                        let tri_min_z = a[2].min(b[2]).min(c[2]);
                        let x0 = (a[0].min(b[0]).min(c[0]) / GLOBAL_SCALE).floor() as i32;
                        let x1 = (a[0].max(b[0]).max(c[0]) / GLOBAL_SCALE).ceil() as i32;
                        let y0 = (a[1].min(b[1]).min(c[1]) / GLOBAL_SCALE).floor() as i32;
                        let y1 = (a[1].max(b[1]).max(c[1]) / GLOBAL_SCALE).ceil() as i32;
                        for gy in y0.max(0)..y1.min(map.size_y as i32) {
                            for gx in x0.max(0)..x1.min(sx as i32) {
                                let px = (gx as f32 + 0.5) * GLOBAL_SCALE;
                                let py = (gy as f32 + 0.5) * GLOBAL_SCALE;
                                let e = |p: &[f32; 3], q: &[f32; 3]| {
                                    (q[0] - p[0]) * (py - p[1]) - (q[1] - p[1]) * (px - p[0])
                                };
                                let (e0, e1, e2) = (e(&a, &b), e(&b, &c), e(&c, &a));
                                if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0)
                                    || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)
                                {
                                    let slot = &mut overlay_floor[gy as usize * sx + gx as usize];
                                    *slot =
                                        Some(slot.map_or(tri_min_z, |m: f32| m.min(tri_min_z)));
                                }
                            }
                        }
                    }
                }
            }

            #[derive(Default)]
            struct FillSet {
                verts: Vec<Vertex>,
                idxs: Vec<u32>,
                coords: Vec<(usize, usize)>,
            }
            let macro_step = 1.0 / map.macro_texture_size.max(1) as f32;
            let mut per_tex: std::collections::HashMap<u32, FillSet> =
                std::collections::HashMap::new();
            let mut fallback = FillSet::default();
            let mut filler_cells = 0usize;
            for y in 0..map.size_y {
                for x in 0..map.size_x {
                    if cell_covered[y * sx + x] {
                        continue;
                    }
                    if map.unit(x, y).flags & crate::matrix_game::common::CELLFLAG_WATER != 0 {
                        continue;
                    }
                    // Cells inside a base's 3x3 footprint are the robot
                    // construction shaft: the platform sinks 66 units and
                    // rises through the opening (sub_unit_offset), so a
                    // ground-level filler would read as a dirt lid the pod
                    // pokes through. Sink those to shaft-floor depth with
                    // the dark pit tone instead.
                    let wx_c = x as f32 * GLOBAL_SCALE + GLOBAL_SCALE * 0.5;
                    let wy_c = y as f32 * GLOBAL_SCALE + GLOBAL_SCALE * 0.5;
                    let base_pit = map.buildings.iter().any(|b| {
                        b.kind == 0 && (b.x - wx_c).abs() <= 35.0 && (b.y - wy_c).abs() <= 35.0
                    });
                    if base_pit {
                        let base = fallback.verts.len() as u32;
                        for (dx, dy) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
                            let p = map.point(x + dx, y + dy);
                            fallback.verts.push(Vertex {
                                position: [
                                    (x + dx) as f32 * GLOBAL_SCALE - cx,
                                    (y + dy) as f32 * GLOBAL_SCALE - cy,
                                    p.z - 70.0,
                                ],
                                color: [
                                    p.r as f32 / 255.0,
                                    p.g as f32 / 255.0,
                                    p.b as f32 / 255.0,
                                    1.0,
                                ],
                                uv: [0.5, 0.5],
                                macro_uv: [0.0, 0.0],
                            });
                            fallback.coords.push((x + dx, y + dy));
                        }
                        fallback
                            .idxs
                            .extend([base, base + 1, base + 2, base + 2, base + 1, base + 3]);
                        filler_cells += 1;
                        continue;
                    }
                    // Nearest covered cell's tile, expanding square rings.
                    let mut tile: Option<(u32, [f32; 2], [f32; 2])> = None;
                    'ring: for r in 1i32..=10 {
                        for ny in (y as i32 - r)..=(y as i32 + r) {
                            for nx in (x as i32 - r)..=(x as i32 + r) {
                                if (ny - y as i32).abs() != r && (nx - x as i32).abs() != r {
                                    continue;
                                }
                                if nx < 0
                                    || ny < 0
                                    || nx as usize >= sx
                                    || ny as usize >= map.size_y
                                {
                                    continue;
                                }
                                if let Some(t) = cell_tile[ny as usize * sx + nx as usize] {
                                    tile = Some(t);
                                    break 'ring;
                                }
                            }
                        }
                    }
                    let set = match tile {
                        Some((tex, _, _)) => per_tex.entry(tex).or_default(),
                        None => &mut fallback,
                    };
                    let (uv0, uv1) = match tile {
                        Some((_, a, b)) => (a, b),
                        None => ([0.5, 0.5], [0.5, 0.5]),
                    };
                    let base = set.verts.len() as u32;
                    // Corner z: seam to the real mesh where it has a vertex;
                    // elsewhere follow the heightfield, but never rise past
                    // the covering overlay (see overlay_floor above).
                    const CORNERS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
                    let seam: [Option<f32>; 4] =
                        CORNERS.map(|(dx, dy)| real_z.get(&(x + dx, y + dy)).copied());
                    let cap = overlay_floor[y * sx + x].map(|z| z - 0.5);
                    for (k, &(dx, dy)) in CORNERS.iter().enumerate() {
                        let p = map.point(x + dx, y + dy);
                        let z = seam[k].unwrap_or_else(|| match cap {
                            Some(c) => p.z.min(c),
                            None => p.z,
                        });
                        set.verts.push(Vertex {
                            position: [
                                (x + dx) as f32 * GLOBAL_SCALE - cx,
                                (y + dy) as f32 * GLOBAL_SCALE - cy,
                                z,
                            ],
                            color: [
                                p.r as f32 / 255.0,
                                p.g as f32 / 255.0,
                                p.b as f32 / 255.0,
                                1.0,
                            ],
                            uv: [
                                if dx == 0 { uv0[0] } else { uv1[0] },
                                if dy == 0 { uv0[1] } else { uv1[1] },
                            ],
                            macro_uv: [
                                macro_step * (x + dx) as f32,
                                macro_step * (y + dy) as f32,
                            ],
                        });
                        set.coords.push((x + dx, y + dy));
                    }
                    set.idxs
                        .extend([base, base + 1, base + 2, base + 2, base + 1, base + 3]);
                    filler_cells += 1;
                }
            }
            if filler_cells > 0 {
                log::info!("terrain bottom: {} pit-filler cells", filler_cells);
            }
            // Fallback tone for cells with no textured neighbor in range:
            // vertex_color * 0.44 reads as shadowed pit interior.
            let dark = create_solid_texture(device, queue, [112, 112, 112, 255]);
            let mut fill_batches: Vec<(Option<u32>, FillSet)> = per_tex
                .into_iter()
                .map(|(tex, set)| (Some(tex), set))
                .collect();
            if !fallback.idxs.is_empty() {
                fill_batches.push((None, fallback));
            }
            for (tex, set) in fill_batches {
                let tex_view: &wgpu::TextureView =
                    match tex.and_then(|t| atlas_views.get(t as usize)) {
                        Some(v) => v,
                        None => &dark,
                    };
                let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Bottom Filler VB"),
                    contents: bytemuck::cast_slice(&set.verts),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
                let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Bottom Filler IB"),
                    contents: bytemuck::cast_slice(&set.idxs),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bottom Filler BG"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&macro_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&macro_sampler),
                        },
                    ],
                });
                batches.push(DrawBatch {
                    bind_group: bg,
                    vertex_buffer: vb,
                    index_buffer: ib,
                    num_indices: set.idxs.len() as u32,
                    index_format: wgpu::IndexFormat::Uint32,
                    cpu_vertices: Some(set.verts),
                    point_coords: Some(set.coords),
                });
            }
        }

        // Reflection texture for the gloss pass — the per-sky `Reflection`
        // param from the Sky block (MatrixMapPrepare.cpp:1137), defaulting
        // to TEXTURE_PATH_REFLECTION (StringConstants.hpp:124) when absent.
        // Falls back to a warm highlight if the asset isn't found.
        let reflection_path = matrix_data
            .block_record("da", "Sky")
            .and_then(|sky_root| matrix_data.block_record(&sky_root, &map.sky_name))
            .and_then(|sky_rec| matrix_data.block_param(&sky_rec, "Reflection"))
            .map(|p| p.trim().replace('\\', "/"))
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| "Matrix/Textures/reflection".to_string());
        let reflection_view = if let Some(rgba) =
            read_texture(&reflection_path).and_then(|data| decode_texture_bytes(&data))
        {
            create_texture_from_rgba_mipped(device, queue, &rgba, 6)
        } else {
            create_solid_texture(device, queue, [230, 220, 200, 255])
        };
        let reflection_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Gloss Reflection Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let gloss_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gloss Uniforms"),
            contents: bytemuck::bytes_of(&GlossUniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                normal_mat: glam::Mat4::IDENTITY.to_cols_array_2d(),
                view: glam::Mat4::IDENTITY.to_cols_array_2d(),
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let gloss_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gloss BGL"),
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
            ],
        });

        // Surface overlays (MatrixTerSurface.cpp)
        let overlays = build_surface_overlays(
            device,
            queue,
            &uniform_buffer,
            &bgl,
            &surface_sampler,
            &sampler_wrap_v,
            &transparent,
            &macro_sampler,
            stor,
            map,
            read_texture,
            Some(GlossResources {
                bgl: &gloss_bgl,
                uniform_buffer: &gloss_uniform_buffer,
                reflection_view: &reflection_view,
                reflection_sampler: &reflection_sampler,
            }),
        );
        let overlay_batches = overlays.base;
        let gloss_batches = overlays.gloss;

        // Decorative objects (palms / rocks / etc.)
        let objects =
            super::object::ObjectsRenderer::new(device, queue, config, map, stor, read_texture);
        // Starting buildings from the CMAP `buildings/*` table.
        let buildings = super::object_building::BuildingsRenderer::new(
            device,
            queue,
            config,
            map,
            read_texture,
        );
        // Robot chassis meshes — built once from `Matrix/Robot/ChassisN.vo`
        // (MatrixObjectRobot.cpp:352). Instances are synced per-frame
        // from the arena.
        let robots =
            super::object_robot::RobotsRenderer::new(device, queue, config, map, read_texture);
        // Cannons / turrets — MatrixObjectCannon.cpp:188-249 loads
        // Basis.vo + Turret{N}.vo + Shaft{N}.vo on first `RNeed`.
        let cannons =
            super::object_cannon::CannonsRenderer::new(device, queue, config, map, read_texture);
        // SPOT_TURRET landscape decals (MatrixObjectBuilding.cpp:1640).
        let slot_markers =
            super::slot_marker::SlotMarkerRenderer::new(device, queue, config, read_texture);

        // Water (MatrixWater.cpp)
        let water =
            super::water::Water::new(device, queue, config, map, stor, matrix_data, read_texture);
        let point_lights = PointLightRenderer::new(device, config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bottom Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main_opaque"),
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
                format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Depth-only pipeline for groups whose geometry carries a -1 texture
        // sentinel. Ports MatrixMapGroup.cpp:593-606: draws the triangles with
        // color writes disabled so they only populate the depth buffer —
        // occluding water / distant geometry without contributing any color.
        let depth_only_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bottom Depth-Only Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main_opaque"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
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
                format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Overlay Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main_alpha"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: -1,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Gloss pipeline — ports TerSurfGlossMW (MatrixRenderPipeline.cpp:1460).
        // Runs as a second additive pass on top of the base overlay, weighted by
        // atlas alpha so unused texels don't leak into the highlight.
        let gloss_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gloss Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(GLOSS_SHADER.into()),
        });
        let gloss_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gloss Overlay Layout"),
                bind_group_layouts: &[&gloss_bgl],
                immediate_size: 0,
            });
        let gloss_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gloss Overlay Pipeline"),
            layout: Some(&gloss_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &gloss_shader,
                entry_point: Some("vs_main"),
                buffers: &[GlossVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gloss_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let depth_texture = create_depth_texture(device, config);
        let stencil_shadows = super::shadow::StencilShadowRenderer::new(device, config);

        let sky = Sky::new(
            device,
            queue,
            config,
            map.sky_color,
            map.water_color,
            &map.sky_name,
            map.sky_angle,
            matrix_data,
            read_texture,
        );
        let clear_color = wgpu::Color {
            r: sr as f64,
            g: sg as f64,
            b: sb as f64,
            a: 1.0,
        };

        Self {
            pipeline,
            depth_only_pipeline,
            overlay_pipeline,
            gloss_pipeline,
            batches,
            depth_only_batches,
            overlay_batches,
            gloss_batches,
            sky,
            clear_color,
            fog_color,
            objects,
            point_lights,
            water,
            uniform_buffer,
            gloss_uniform_buffer,
            depth_texture,
            stencil_shadows,
            last_point_light_revision: 0,
            buildings,
            robots,
            cannons,
            slot_markers,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) {
        self.depth_texture = create_depth_texture(device, config);
    }

    pub fn takt(
        &mut self,
        dt_ms: f32,
        map: &GameMap,
        point_lights: &PointLightSystem,
        camera: &Camera,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let revision = point_lights.revision();
        if revision != self.last_point_light_revision {
            // Retint only the vertices inside the relit window and
            // upload just that byte range per batch — a battle's
            // muzzle-flash churn used to rewrite + re-upload the whole
            // map's vertex buffers on every relight.
            let [rx0, ry0, rx1, ry1] = point_lights.changed_rect();
            for batch in &mut self.batches {
                let (Some(cpu_vertices), Some(point_coords)) =
                    (batch.cpu_vertices.as_mut(), batch.point_coords.as_ref())
                else {
                    continue;
                };
                let mut lo = usize::MAX;
                let mut hi = 0usize;
                for (i, (vertex, &(px, py))) in
                    cpu_vertices.iter_mut().zip(point_coords.iter()).enumerate()
                {
                    let (xi, yi) = (px as i32, py as i32);
                    if xi < rx0 || xi > rx1 || yi < ry0 || yi > ry1 {
                        continue;
                    }
                    let point = map.point(px, py);
                    let lum = point_lights.point_lum(px, py, map.size_x);
                    vertex.color[0] = ((point.r as i32 + lum[0]).clamp(0, 255) as f32) / 255.0;
                    vertex.color[1] = ((point.g as i32 + lum[1]).clamp(0, 255) as f32) / 255.0;
                    vertex.color[2] = ((point.b as i32 + lum[2]).clamp(0, 255) as f32) / 255.0;
                    lo = lo.min(i);
                    hi = hi.max(i);
                }
                if lo <= hi && lo != usize::MAX {
                    let vsize = std::mem::size_of::<Vertex>();
                    queue.write_buffer(
                        &batch.vertex_buffer,
                        (lo * vsize) as u64,
                        bytemuck::cast_slice(&cpu_vertices[lo..=hi]),
                    );
                }
            }
            self.last_point_light_revision = revision;
        }

        if let Some(water) = &mut self.water {
            water.takt(dt_ms, device, queue, camera, map);
        }
        if let Some(objects) = &mut self.objects {
            objects.takt(dt_ms, queue, map, point_lights);
        }
        if let Some(buildings) = &mut self.buildings {
            buildings.takt(dt_ms, queue, map, point_lights);
        }
        if let Some(robots) = &mut self.robots {
            robots.takt(dt_ms);
        }
        if let Some(cannons) = &mut self.cannons {
            cannons.takt(dt_ms);
        }
        self.sky.takt(dt_ms);
        self.point_lights.sync(device, map, point_lights);
    }

    /// Push the current per-frame `CMatrixBuilding::m_BaseFloor` /
    /// sub-unit matrix state from the live arena into the
    /// `BuildingsRenderer` instance buffers. Ports
    /// MatrixObjectBuilding.cpp:836-852. Called each frame after
    /// `World::takt` so the platform-rise animation is visible.
    pub fn sync_building_animation(
        &mut self,
        queue: &wgpu::Queue,
        objs: &super::map_static::Objects,
        map: &GameMap,
        point_lights: &PointLightSystem,
    ) {
        if let Some(buildings) = &mut self.buildings {
            buildings.sync_building_animation(queue, objs, map, point_lights);
        }
        // Decorative-object body swaps (BREAK death / terron corpse) ride
        // the same once-per-frame sync point.
        if let Some(objects) = &mut self.objects {
            objects.sync_break_swaps(queue, objs, map, point_lights);
        }
    }

    /// Rebuild per-robot instance transforms for the chassis
    /// renderer + advance per-robot animation cursors. Called each
    /// frame after `MapLogic::takt` so freshly spawned robots show
    /// up and tracks keep animating. `cms` is the per-frame delta
    /// used for the anim step.
    #[allow(clippy::too_many_arguments)]
    pub fn sync_robots(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        objs: &mut super::map_static::Objects,
        map: &GameMap,
        point_lights: &PointLightSystem,
        cms: i32,
        ghost_cannon: Option<super::object_cannon::GhostCannon>,
        slot_markers: &[super::slot_marker::SlotMarker],
        camera: &Camera,
    ) {
        let visible = visible_groups_mask(camera, map);
        if let Some(objects) = &mut self.objects {
            objects.sync_visibility(queue, &visible);
        }
        self.stencil_shadows.begin_frame();
        if let Some(objects) = &self.objects {
            objects.push_stencil_shadows(&mut self.stencil_shadows, &visible);
        }
        if let Some(buildings) = &mut self.buildings {
            buildings.push_stencil_shadows(&mut self.stencil_shadows);
        }
        if let Some(robots) = &mut self.robots {
            robots.sync_robots(
                device,
                queue,
                objs,
                map,
                point_lights,
                cms,
                &visible,
                &mut self.stencil_shadows,
            );
        }
        if let Some(cannons) = &mut self.cannons {
            cannons.sync_cannons(
                device,
                queue,
                objs,
                map,
                ghost_cannon,
                &[],
                &visible,
                &mut self.stencil_shadows,
            );
        }
        if let Some(sm) = &mut self.slot_markers {
            sm.sync(queue, map, slot_markers);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        _device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        queue: &wgpu::Queue,
        camera: &Camera,
        view_proj: glam::Mat4,
        view_mat: glam::Mat4,
        map: &GameMap,
    ) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color: self.fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
        );

        // Gloss pass needs a camera-space normal matrix matching
        // D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR: inverse-transpose of the
        // rendering view matrix. `view_mat` is the Z-up look-at used by the
        // water renderer for the same purpose.
        if !self.gloss_batches.is_empty() {
            let normal_mat = view_mat.inverse().transpose();
            queue.write_buffer(
                &self.gloss_uniform_buffer,
                0,
                bytemuck::bytes_of(&GlossUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    normal_mat: normal_mat.to_cols_array_2d(),
                    view: view_mat.to_cols_array_2d(),
                    fog_color: self.fog_color,
                    fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                }),
            );
        }

        self.stencil_shadows
            .upload(_device, queue, view_proj, map.shadow_color);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Terrain Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Diagnostic stage gate for map_render_bench: MG_SKIP=objects,water,...
        let skip_env = std::env::var("MG_SKIP").unwrap_or_default();
        let skip = |name: &str| skip_env.split(',').any(|s| s == name);

        // Sky gradient (ports DrawSky, MatrixMap.cpp:2020). Drawn before landscape
        // so terrain/water overwrite it where geometry exists.
        self.sky.render(queue, &mut pass, camera);

        // Bottom geometry (opaque)
        pass.set_pipeline(&self.pipeline);
        for batch in self.batches.iter().take(if skip("bottom") { 0 } else { usize::MAX }) {
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
            pass.draw_indexed(0..batch.num_indices, 0, 0..1);
        }

        // Depth-only pass for groups whose -1 texture sentinel signals
        // "carve a hole but occlude everything behind" (MatrixMapGroup.cpp:
        // 593-606). Populating the depth buffer here keeps later water /
        // object draws from showing through these triangles.
        if !self.depth_only_batches.is_empty() {
            pass.set_pipeline(&self.depth_only_pipeline);
            for batch in &self.depth_only_batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
                pass.draw_indexed(0..batch.num_indices, 0, 0..1);
            }
        }

        // Per-group visibility mask, matches CMatrixMap::m_VisibleGroups
        // after CalcMapGroupVisibility (MatrixVisiCalc.cpp:534-779). Surfaces
        // the C++ attached to a specific group via `g->AddSurface(this)` only
        // draw when at least one of their owning groups is visible. The same
        // mask culls decorative objects, their projected shadows and the
        // per-group water passes below.
        let visible = visible_groups_mask(camera, map);
        let overlay_visible = |groups: &[u32]| -> bool {
            if groups.is_empty() {
                // Surfaces with no group association (e.g. turret-place
                // foundation pads baked into the terrain) must still be
                // drawn — they render unconditionally. The port doesn't
                // always populate per-surface group indices the way the
                // C++ group-marking pass does, so gating them off hides
                // legitimate terrain overlays.
                return true;
            }
            groups
                .iter()
                .any(|&gi| visible.get(gi as usize).copied().unwrap_or(false))
        };

        // Surface overlays (alpha blended, triangle strips)
        if !self.overlay_batches.is_empty() && !skip("overlays") {
            pass.set_pipeline(&self.overlay_pipeline);
            for batch in &self.overlay_batches {
                if !overlay_visible(&batch.groups) {
                    continue;
                }
                let draw = &batch.draw;
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), draw.index_format);
                pass.draw_indexed(0..draw.num_indices, 0, 0..1);
            }
        }

        // Gloss pass: adds gloss*reflection weighted by atlas alpha on top of
        // the already-composited overlay (ports the stage 5 ADD(TEMP, CURRENT)
        // step of TerSurfGlossMW).
        if !self.gloss_batches.is_empty() && !skip("gloss") {
            pass.set_pipeline(&self.gloss_pipeline);
            for batch in &self.gloss_batches {
                if !overlay_visible(&batch.groups) {
                    continue;
                }
                let draw = &batch.draw;
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..draw.num_indices, 0, 0..1);
            }
        }

        // Decorative objects (their shadows now land in the stencil
        // phase after water, like the C++ DrawShadows).
        if let Some(objects) = &self.objects {
            if !skip("objects") {
                objects.render(queue, &mut pass, camera, view_proj);
            }
        }
        // Starting buildings — drawn after objects so their shadow projection
        // (when we add it) can overlay object silhouettes the way the original
        // does (MatrixMap.cpp DrawLandscape ordering).
        if let Some(robots) = &self.robots {
            if !skip("robots") {
                robots.render(queue, &mut pass, camera, view_proj);
            }
        }
        if let Some(buildings) = &self.buildings {
            if !skip("buildings") {
                buildings.render(queue, &mut pass, camera, view_proj);
            }
        }
        if let Some(cannons) = &self.cannons {
            if !skip("cannons") {
                cannons.render(queue, &mut pass, camera, view_proj);
            }
        }
        if let Some(sm) = &self.slot_markers {
            sm.render(queue, &mut pass, view_proj);
        }

        // Visible additive point-light pass on terrain-conforming geometry.
        self.point_lights.render(queue, &mut pass, view_proj);

        if let Some(water) = &mut self.water {
            if !skip("water") {
                water.render(_device, &mut pass, queue, camera, view_proj, view_mat, &visible);
            }
        }

        // `DrawShadows` (MatrixMap.cpp:2285): stencil-count the shadow
        // volumes, mark the proj silhouettes (colour writes off), then
        // darken everything at stencil >= 1 with one fullscreen quad.
        if !skip("shadows") {
            self.stencil_shadows.render_volumes(&mut pass);
            let mut marked = self.stencil_shadows.has_volumes();
            if let Some(objects) = &self.objects {
                marked |= objects.render_shadows_stencil(&mut pass, &visible);
            }
            if marked {
                self.stencil_shadows.render_darken(&mut pass);
            }
        }
    }

    /// Ports `CMinimap::RenderBackground` (MatrixMinimap.cpp:855-1199).
    ///
    /// Renders the landscape orthographically from above into `color_view` /
    /// `depth_view`. Matches the original pass set it calls via
    /// `DrawLandscape(true)` + `DrawLandscapeSurfaces(true)` + water alpha:
    /// opaque bottom → depth-only → surfaces → gloss → per-group water alpha.
    /// Sky, objects, buildings, point lights are intentionally skipped — the
    /// original sets `MMFLAG_DISABLE_DRAW_OBJECT_LIGHTS` for the bake and
    /// stamps buildings via `RenderObjectToBackground` separately.
    #[allow(clippy::too_many_arguments)]
    pub fn bake_minimap(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        view_proj: glam::Mat4,
        clear_color: wgpu::Color,
    ) {
        // Repoint both uniform buffers at the orthographic VP. `normal_mat`
        // for gloss is left at identity since the ortho top-down view has no
        // meaningful camera-space reflection direction — matches the flat
        // look of RenderBackground's pass.
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color: self.fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            }),
        );
        if !self.gloss_batches.is_empty() {
            queue.write_buffer(
                &self.gloss_uniform_buffer,
                0,
                bytemuck::bytes_of(&GlossUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    normal_mat: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    view: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    fog_color: self.fog_color,
                    fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                }),
            );
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Minimap Bake"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Bottom (opaque terrain).
        pass.set_pipeline(&self.pipeline);
        for batch in &self.batches {
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
            pass.draw_indexed(0..batch.num_indices, 0, 0..1);
        }

        // Depth-only sentinel batches (MatrixMapGroup.cpp:593-606).
        if !self.depth_only_batches.is_empty() {
            pass.set_pipeline(&self.depth_only_pipeline);
            for batch in &self.depth_only_batches {
                pass.set_bind_group(0, &batch.bind_group, &[]);
                pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
                pass.set_index_buffer(batch.index_buffer.slice(..), batch.index_format);
                pass.draw_indexed(0..batch.num_indices, 0, 0..1);
            }
        }

        // Surface overlays — no visibility culling, the whole map is in view.
        if !self.overlay_batches.is_empty() {
            pass.set_pipeline(&self.overlay_pipeline);
            for batch in &self.overlay_batches {
                let draw = &batch.draw;
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), draw.index_format);
                pass.draw_indexed(0..draw.num_indices, 0, 0..1);
            }
        }

        if !self.gloss_batches.is_empty() {
            pass.set_pipeline(&self.gloss_pipeline);
            for batch in &self.gloss_batches {
                let draw = &batch.draw;
                pass.set_bind_group(0, &draw.bind_group, &[]);
                pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                pass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..draw.num_indices, 0, 0..1);
            }
        }

        // Water solid + fill + alpha — fills the non-terrain area inside the
        // fsz×fsz ortho footprint with proper animated water, matching the
        // double loop in RenderBackground (MatrixMinimap.cpp:1055-1112).
        if let Some(water) = &mut self.water {
            water.bake_minimap_all(device, queue, &mut pass, view_proj);
        }
    }
}

/// Terrain shader — ports TerBotM (MatrixRenderPipeline.cpp:1198-1213).
const SHADER: &str = include_str!("../../shaders/terrain.wgsl");
/// Gloss overlay shader — ports TerSurfGlossMW (MatrixRenderPipeline.cpp:1460).
/// Runs as an additive pass on top of the already-composited base overlay: we
/// add `gloss.rgb * reflection.rgb` and pipe `atlas.a` through as the source
/// alpha so the hardware blend does `final += gloss*refl*atlas.a`, matching
/// the stage 5 `ADD(TEMP, CURRENT)` with SrcAlpha/InvSrcAlpha blending in the
/// original single-pass pipeline.
const GLOSS_SHADER: &str = include_str!("../../shaders/terrain_gloss.wgsl");

// ════════════════════════════════════════════════════════════════════════
// CMatrixMap::DrawSky — ports MatrixMap.cpp:2020-2189.
//
// Two parts:
//   * Skybox pass: four textured walls laid out as vertical strips of a
//     single panoramic texture (Fore / Rite / Back / Left), drawn with a
//     rotation-only view + shallow perspective (`CalcSkyMatrix`).
//   * Gradient pass: screen-space fade along the horizon line computed
//     from camera direction (MatrixVisiCalc.cpp:609-626). The top color
//     fades from transparent to the sky color so the gradient blends
//     into the skybox instead of clipping against it.
//
// `m_SkyAngle += m_SkyDeltaAngle * step` advance in CMatrixMap::Takt
// (MatrixMap.cpp:2495).
// ════════════════════════════════════════════════════════════════════════

/// MAX_VIEW_DISTANCE from MatrixCamera.cpp:13.
const MAX_VIEW_DISTANCE: f32 = 4000.0;
/// SH1 = g_ScreenY * 0.270416... (MatrixMap.cpp:2144).
const SH1_FRAC: f32 = 0.270_416_68;
/// SH2 = g_ScreenY * 0.07 (MatrixMap.cpp:2145).
const SH2_FRAC: f32 = 0.07;

/// Matches CAM_HFOV from MatrixCamera.hpp:51 — used only by the skybox
/// perspective so skybox walls project onto the frustum exactly once per face.
const SKY_HFOV: f32 = std::f32::consts::PI / 3.0;

/// Sky face names in the order the Rust port's skybox geometry expects them:
/// Fore (+Y), Rite (+X), Back (-Y), Left (-X). Matches the C++ SKY_FORE /
/// SKY_RITE / SKY_BACK / SKY_LEFT enum ordering (MatrixMap.cpp:2071-2098)
/// and the `Sky` block key names in `cfg/robots/data.txt`.
const FACE_CONFIG_KEYS: [&str; 4] = ["Fore", "Right", "Back", "Left"];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GradientVertex {
    position: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BoxVertex {
    xy: [f32; 2],
    bottom_w: f32,
    u: f32,
    v0: f32,
    v1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SkyboxUniforms {
    sky_view_proj: [[f32; 4]; 4],
    cut_dn: [f32; 4],
}

pub struct Sky {
    gradient_pipeline: wgpu::RenderPipeline,
    gradient_vertex_buffer: wgpu::Buffer,
    sky_color: [f32; 3],
    skybox: Option<Skybox>,
}

struct Skybox {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    base_angle: f32,
    delta_angle: f32,
    current_angle: f32,
}

impl Sky {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        sky_color_rgba: u32,
        _water_color_rgba: u32,
        sky_name: &str,
        sky_angle: f32,
        matrix_data: &Storage,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Self {
        let sky_color = unpack_rgb(sky_color_rgba);

        let gradient_pipeline = build_gradient_pipeline(device, config);
        let gradient_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sky Gradient VB"),
            contents: bytemuck::cast_slice(
                &[GradientVertex {
                    position: [0.0, 0.0],
                    color: [0.0; 4],
                }; 10],
            ),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let sky_cfg = resolve_sky_config(matrix_data, sky_name).or_else(|| {
            log::warn!(
                "sky: no config for '{}' in robots.dat; skybox disabled",
                sky_name
            );
            None
        });
        let skybox = sky_cfg
            .and_then(|cfg| load_skybox(device, queue, config, &cfg, sky_angle, read_texture));

        Self {
            gradient_pipeline,
            gradient_vertex_buffer,
            sky_color,
            skybox,
        }
    }

    /// Advance the skybox rotation. Mirrors `m_SkyAngle += m_SkyDeltaAngle *
    /// step` in `CMatrixMap::Takt` (MatrixMap.cpp:2495).
    pub fn takt(&mut self, dt_ms: f32) {
        if let Some(skybox) = &mut self.skybox {
            skybox.current_angle += skybox.delta_angle * dt_ms;
        }
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
    ) {
        let has_skybox = self.skybox.is_some();

        if let Some(skybox) = &self.skybox {
            skybox.render(queue, pass, camera);
        }

        self.render_gradient(queue, pass, camera, has_skybox);
    }

    fn render_gradient<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
        has_skybox: bool,
    ) {
        let sh_frac = compute_sky_height_frac(camera);
        let bot_ndc = 1.0 - 2.0 * sh_frac;
        let mid_ndc = 1.0 - 2.0 * (sh_frac - SH2_FRAC);
        let mut top_ndc = 1.0 - 2.0 * (sh_frac - SH1_FRAC);
        if !has_skybox && top_ndc < 1.0 {
            top_ndc = 1.0;
        }

        let [sr, sg, sb] = self.sky_color;
        let top_color = if has_skybox {
            [sr, sg, sb, 0.0]
        } else {
            [0.0, 0.0, 0.0, 0.0]
        };
        let sky_opaque = [sr, sg, sb, 1.0];
        // Below the horizon everything the geometry doesn't cover must read
        // as fog. The original leaves those pixels holding the previous
        // frame (no color clear in release, MatrixFormGame.cpp:149) and its
        // linear fog converges distant geometry to m_SkyColor — so the fog
        // color is the faithful fill. This used to be `water_color`, which
        // is 0xFFFFFF on ARMAGEDD and turned every coverage gap (base pits,
        // hole cells) into a glaring white "unpainted tile".
        let water_opaque = sky_opaque;
        let water_top = bot_ndc.clamp(-1.0, 1.0);

        let verts = [
            GradientVertex {
                position: [-1.0, top_ndc],
                color: top_color,
            },
            GradientVertex {
                position: [1.0, top_ndc],
                color: top_color,
            },
            GradientVertex {
                position: [-1.0, mid_ndc],
                color: sky_opaque,
            },
            GradientVertex {
                position: [1.0, mid_ndc],
                color: sky_opaque,
            },
            GradientVertex {
                position: [-1.0, bot_ndc],
                color: sky_opaque,
            },
            GradientVertex {
                position: [1.0, bot_ndc],
                color: sky_opaque,
            },
            GradientVertex {
                position: [-1.0, water_top],
                color: water_opaque,
            },
            GradientVertex {
                position: [1.0, water_top],
                color: water_opaque,
            },
            GradientVertex {
                position: [-1.0, -1.0],
                color: water_opaque,
            },
            GradientVertex {
                position: [1.0, -1.0],
                color: water_opaque,
            },
        ];
        queue.write_buffer(
            &self.gradient_vertex_buffer,
            0,
            bytemuck::cast_slice(&verts),
        );

        pass.set_pipeline(&self.gradient_pipeline);
        pass.set_vertex_buffer(0, self.gradient_vertex_buffer.slice(..));
        pass.draw(0..6, 0..1);
        pass.draw(6..10, 0..1);
    }
}

impl Skybox {
    fn render<'a>(&'a self, queue: &wgpu::Queue, pass: &mut wgpu::RenderPass<'a>, camera: &Camera) {
        let sky_view_proj = build_sky_view_proj(camera, self.base_angle + self.current_angle);
        let cut_dn = compute_cut_dn(camera.eye_pos().z);
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&SkyboxUniforms {
                sky_view_proj: sky_view_proj.to_cols_array_2d(),
                cut_dn: [cut_dn, 0.0, 0.0, 0.0],
            }),
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..24, 0..1);
    }
}

fn build_gradient_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sky Gradient Shader"),
        source: wgpu::ShaderSource::Wgsl(GRADIENT_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Sky Gradient Layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Sky Gradient Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GradientVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: 8,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x4,
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
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn load_skybox(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    cfg: &SkyConfig,
    sky_angle: f32,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<Skybox> {
    let texture_path = &cfg.faces[0].texture;
    let data = read_texture(texture_path)?;
    let rgba = decode_texture_bytes(&data)?;
    let tex_w = rgba.width() as f32;
    let tex_h = rgba.height() as f32;
    let texture_view = if rgba.width() >= 4 && rgba.height() >= 4 {
        create_texture_from_rgba_mipped(device, queue, &rgba, 6)
    } else {
        create_texture_from_rgba(device, queue, &rgba)
    };
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Skybox Sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Skybox Shader"),
        source: wgpu::ShaderSource::Wgsl(SKYBOX_SHADER.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Skybox BGL"),
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
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Skybox Layout"),
        bind_group_layouts: &[&bgl],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Skybox Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<BoxVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: 8,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32,
                    },
                    wgpu::VertexAttribute {
                        offset: 16,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32,
                    },
                    wgpu::VertexAttribute {
                        offset: 20,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32,
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
                blend: None,
                write_mask: wgpu::ColorWrites::COLOR,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let verts = build_skybox_vertices(cfg, tex_w, tex_h);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Skybox VB"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Skybox UB"),
        contents: bytemuck::bytes_of(&SkyboxUniforms {
            sky_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            cut_dn: [0.5, 0.0, 0.0, 0.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Skybox BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    log::info!(
        "skybox: loaded '{}' (angle = {:.3} + {:.3} rad, drift = {:.6} rad/ms)",
        texture_path,
        cfg.base_angle,
        sky_angle,
        cfg.delta_angle
    );
    Some(Skybox {
        pipeline,
        vertex_buffer,
        uniform_buffer,
        bind_group,
        base_angle: cfg.base_angle + sky_angle,
        delta_angle: cfg.delta_angle,
        current_angle: 0.0,
    })
}

struct SkyConfig {
    base_angle: f32,
    delta_angle: f32,
    faces: [SkyFaceConfig; 4],
}

struct SkyFaceConfig {
    texture: String,
    uv: [f32; 4],
}

fn resolve_sky_config(stor: &Storage, sky_name: &str) -> Option<SkyConfig> {
    let sky_root = stor.block_record("da", "Sky")?;
    let sky_rec = stor.block_record(&sky_root, sky_name)?;

    let parse_deg = |s: Option<String>| -> f32 {
        s.and_then(|v| v.parse::<f32>().ok())
            .map(|d| d.to_radians())
            .unwrap_or(0.0)
    };
    let base_angle = parse_deg(stor.block_param(&sky_rec, "Angle"));
    let delta_angle = parse_deg(stor.block_param(&sky_rec, "DeltaAngle"));

    let faces = std::array::from_fn(|i| {
        let raw = stor
            .block_param(&sky_rec, FACE_CONFIG_KEYS[i])
            .unwrap_or_default();
        parse_face(&raw)
    });

    Some(SkyConfig {
        base_angle,
        delta_angle,
        faces,
    })
}

fn parse_face(raw: &str) -> SkyFaceConfig {
    let mut parts = raw.splitn(5, ',');
    let path = parts.next().unwrap_or("").trim().replace('\\', "/");
    let mut uv = [0.0_f32, 0.0, 1.0, 1.0];
    for slot in &mut uv {
        if let Some(p) = parts.next() {
            if let Ok(v) = p.trim().parse::<f32>() {
                *slot = v;
            }
        }
    }
    SkyFaceConfig { texture: path, uv }
}

fn build_skybox_vertices(cfg: &SkyConfig, tex_w: f32, tex_h: f32) -> Vec<BoxVertex> {
    let positions: [[[f32; 2]; 4]; 4] = [
        [[-1.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]],
        [[1.0, -1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, -1.0]],
        [[-1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]],
    ];

    let mut verts = Vec::with_capacity(24);
    for (i, p) in positions.iter().enumerate() {
        let [u0, v0, u1, v1] = cfg.faces[i].uv;
        let nu0 = u0 / tex_w;
        let nu1 = u1 / tex_w;
        let nv0 = v0 / tex_h;
        let nv1 = v1 / tex_h;
        push_face(&mut verts, p[0], p[1], p[2], p[3], nu0, nu1, nv0, nv1);
    }

    verts
}

#[allow(clippy::too_many_arguments)]
fn push_face(
    out: &mut Vec<BoxVertex>,
    xy_tl: [f32; 2],
    xy_tr: [f32; 2],
    xy_bl: [f32; 2],
    xy_br: [f32; 2],
    u_left: f32,
    u_right: f32,
    v0: f32,
    v1: f32,
) {
    let tl = BoxVertex {
        xy: xy_tl,
        bottom_w: 0.0,
        u: u_left,
        v0,
        v1,
    };
    let tr = BoxVertex {
        xy: xy_tr,
        bottom_w: 0.0,
        u: u_right,
        v0,
        v1,
    };
    let bl = BoxVertex {
        xy: xy_bl,
        bottom_w: 1.0,
        u: u_left,
        v0,
        v1,
    };
    let br = BoxVertex {
        xy: xy_br,
        bottom_w: 1.0,
        u: u_right,
        v0,
        v1,
    };
    out.push(tl);
    out.push(bl);
    out.push(tr);
    out.push(tr);
    out.push(bl);
    out.push(br);
}

fn compute_cut_dn(eye_z: f32) -> f32 {
    (200.0 + eye_z) * 0.5 / MAX_VIEW_DISTANCE + 0.5
}

fn build_sky_view_proj(camera: &Camera, sky_angle: f32) -> glam::Mat4 {
    let fwd = camera.forward();
    let y_up_fwd = glam::Vec3::new(fwd.x, fwd.z, -fwd.y);
    let view_rot = glam::Mat4::look_at_lh(glam::Vec3::ZERO, y_up_fwd, glam::Vec3::Y);

    let yaw = glam::Mat4::from_rotation_z(sky_angle);
    let z_to_y = glam::Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);

    let proj = glam::Mat4::perspective_lh(SKY_HFOV, camera.aspect, 0.01, 3.0);

    proj * view_rot * z_to_y * yaw
}

/// Ports the m_SkyHeight computation from MatrixVisiCalc.cpp:609-626.
/// Returns the horizon line as a fraction of screen height (0=top, 1=bottom).
fn compute_sky_height_frac(camera: &Camera) -> f32 {
    let dir = camera.forward();
    let mut proj = glam::Vec3::new(dir.x, dir.y, 0.0);
    let len = proj.length();
    if len < 0.0001 {
        return -1.0;
    }
    proj /= len;
    let eye = camera.eye_pos();
    let mut target = eye + proj * MAX_VIEW_DISTANCE;
    target.z -= eye.z * 1.5;

    let clip = camera.view_proj() * glam::Vec4::new(target.x, target.y, target.z, 1.0);
    if clip.w.abs() < 0.0001 {
        return -1.0;
    }
    let ndc_y = clip.y / clip.w;
    (1.0 - ndc_y) * 0.5
}

const GRADIENT_SHADER: &str = include_str!("../../shaders/sky_gradient.wgsl");
const SKYBOX_SHADER: &str = include_str!("../../shaders/sky_skybox.wgsl");

// ── SRobotWeaponMatrix ──────────────────────────────────────────────────
//
// Port of `SRobotWeaponMatrix` (MatrixMap.hpp:170-180) — per-armor pylon
// layout loaded from the chassis VO at map-load time (MatrixMap.cpp:
// 236-334). Each armor kind (1..=ROBOT_ARMOR_CNT) carries a list of
// pylon slots; each slot encodes which weapon categories it accepts
// via `access_invert` (bit N set = weapon index N is blocked, because
// the C++ stores it inverted).
//
// Until the VO loader is wired for reading this off the meshes,
// `default_weapon_matrix` provides a hand-rolled table matching shipped
// gameplay: 4 common slots + 1 extra (bomb/mortar) per armor.

use crate::matrix_game::config::RobotUnitKind;
use crate::matrix_game::interface::constructor::Unit;
use crate::matrix_game::object_robot::MAX_WEAPON_CNT;

pub const ROBOT_WEAPONS_PER_ROBOT_CNT: usize = 16;

#[derive(Debug, Clone, Copy, Default)]
pub struct WeaponMatrixSlot {
    pub id: i32,
    pub access_invert: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct WeaponMatrix {
    /// Total slots on this armor. Slot indices `< common` are
    /// "common" (MG/Cannon/Missile/Laser/Plasma/Electric/Repair);
    /// the rest are "extra" (Bomb/Mortar/Flamethrower).
    pub cnt: i32,
    pub common: i32,
    pub extra: i32,
    pub list: [WeaponMatrixSlot; ROBOT_WEAPONS_PER_ROBOT_CNT],
}

impl Default for WeaponMatrix {
    fn default() -> Self {
        Self {
            cnt: 0,
            common: 0,
            extra: 0,
            list: [WeaponMatrixSlot::default(); ROBOT_WEAPONS_PER_ROBOT_CNT],
        }
    }
}

/// Extra-slot bit — bit 4 (Flamethrower) or bit 6 (Bomb/Mortar) set
/// in `access_invert` marks a slot as "extra" (CConstructor.cpp:477).
pub const ACCESS_EXTRA_BIT_A: u32 = 1 << 4;
pub const ACCESS_EXTRA_BIT_B: u32 = 1 << 6;

impl WeaponMatrix {
    /// Port of the bit-test at CConstructor.cpp:477-480 / :503-504.
    pub fn is_extra_slot(&self, i: usize) -> bool {
        let a = self.list[i].access_invert;
        (a & ACCESS_EXTRA_BIT_A) != 0 || (a & ACCESS_EXTRA_BIT_B) != 0
    }

    /// Port of `CConstructor::CheckWeaponLegality` (CConstructor.cpp:
    /// 731-761) — find the first empty pylon on this armor that
    /// accepts `weapon_kind`.
    pub fn find_pylon_for(
        &self,
        weapon_kind: RobotUnitKind,
        current: &[Unit; MAX_WEAPON_CNT],
    ) -> Option<usize> {
        let is_super = weapon_kind == RobotUnitKind::WEAPON_MORTAR
            || weapon_kind == RobotUnitKind::WEAPON_BOMB;
        let limit = (self.cnt as usize).min(ROBOT_WEAPONS_PER_ROBOT_CNT);
        if !is_super {
            for (t, unit) in current.iter().enumerate().take(limit) {
                if !self.is_extra_slot(t) && unit.is_empty() {
                    return Some(t);
                }
            }
            None
        } else {
            (0..limit).find(|&t| self.is_extra_slot(t))
        }
    }
}

/// Total armor kinds we cache weapon matrices for (matches
/// ROBOT_ARMOR_CNT = 6).
const ROBOT_ARMOR_CNT_LOCAL: usize = 6;

/// Process-wide armor → WeaponMatrix table. Populated at startup by
/// `set_weapon_matrix_for` after the robot VOs load (see
/// `RobotsRenderer::new`); falls back to an empty matrix when not yet
/// populated. Mirrors `g_MatrixMap->m_RobotWeaponMatrix[]` (MatrixMap.
/// hpp:179).
static GLOBAL_WEAPON_MATRICES: std::sync::OnceLock<
    std::sync::RwLock<[WeaponMatrix; ROBOT_ARMOR_CNT_LOCAL]>,
> = std::sync::OnceLock::new();

fn weapon_matrices_slot() -> &'static std::sync::RwLock<[WeaponMatrix; ROBOT_ARMOR_CNT_LOCAL]> {
    GLOBAL_WEAPON_MATRICES
        .get_or_init(|| std::sync::RwLock::new([WeaponMatrix::default(); ROBOT_ARMOR_CNT_LOCAL]))
}

/// Faithful port of `CMatrixMap::RobotPreload`'s armor-VO weapon-slot
/// construction (MatrixMap.cpp:270-326). Walks every bone with `id >=
/// 20` whose name starts with `"W"`, parses the comma-separated slot
/// spec, and fills out a `WeaponMatrix` mirroring
/// `m_RobotWeaponMatrix[i]`.
///
/// The bone name format is `"W,n[,n...][,I]"` where each `n` is a
/// 1-based weapon-kind index (RUK_WEAPON_*) the slot accepts and `I`
/// flips the X-mirror flag (bit 31 of `access_invert`).
pub fn weapon_matrix_from_vo(
    vo: &crate::matrix_lib::three_g::vector_object::VoMesh,
) -> WeaponMatrix {
    let mut out = WeaponMatrix::default();
    for m in &vo.matrices {
        if m.id < 20 {
            continue;
        }
        let parts: Vec<&str> = m.name.split(',').map(|s| s.trim()).collect();
        if parts.is_empty() || parts[0] != "W" {
            continue;
        }
        if (out.cnt as usize) >= ROBOT_WEAPONS_PER_ROBOT_CNT {
            break;
        }
        let slot_idx = out.cnt as usize;
        out.list[slot_idx] = WeaponMatrixSlot {
            id: m.id as i32,
            access_invert: 0,
        };
        out.cnt += 1;
        // First slot-kind token decides whether this is a common or
        // extra slot (MatrixMap.cpp:298-304). The C++ checks ONLY the
        // first kind token because access_invert is still 0; subsequent
        // tokens just OR in more accepted kinds.
        let mut first_kind_token = true;
        for tok in &parts[1..] {
            if *tok == "I" {
                out.list[slot_idx].access_invert |= 1u32 << 31;
                continue;
            }
            let Ok(kind) = tok.parse::<i32>() else {
                continue;
            };
            if first_kind_token {
                first_kind_token = false;
                if kind == RobotUnitKind::WEAPON_BOMB.0 || kind == RobotUnitKind::WEAPON_MORTAR.0 {
                    out.extra += 1;
                } else {
                    out.common += 1;
                }
            }
            if (1..=32).contains(&kind) {
                out.list[slot_idx].access_invert |= 1u32 << (kind - 1);
            }
        }
    }
    // MatrixMap.cpp:314-325 — sort slots by bone id ascending.
    let n = out.cnt as usize;
    if n > 1 {
        out.list[..n].sort_by_key(|s| s.id);
    }
    out
}

/// Replace the cached weapon matrix for `kind` (1..=ROBOT_ARMOR_CNT).
/// Called from the robot-VO loader once per armor.
pub fn set_weapon_matrix_for(kind: RobotUnitKind, m: WeaponMatrix) {
    let idx = match kind.0 {
        n if (1..=ROBOT_ARMOR_CNT_LOCAL as i32).contains(&n) => (n - 1) as usize,
        _ => return,
    };
    if let Ok(mut slot) = weapon_matrices_slot().write() {
        slot[idx] = m;
    }
}

/// Read the cached weapon matrix for `armor_kind`. Returns an empty
/// matrix if the kind is out of range or hasn't been populated yet.
/// Faithful port of `g_MatrixMap->m_RobotWeaponMatrix[kind-1]` reads
/// scattered through CConstructor.cpp.
pub fn default_weapon_matrix(armor_kind: RobotUnitKind) -> WeaponMatrix {
    let idx = match armor_kind.0 {
        n if (1..=ROBOT_ARMOR_CNT_LOCAL as i32).contains(&n) => (n - 1) as usize,
        _ => return WeaponMatrix::default(),
    };
    weapon_matrices_slot()
        .read()
        .ok()
        .map(|slot| slot[idx])
        .unwrap_or_default()
}

// ── GetSideColor / GetSideColorMM (MatrixMap.cpp:1014-1015) ─────────────

/// Port of `CMatrixMap::GetSideColor` (MatrixMap.cpp:1014, MatrixMap.hpp:738).
/// Returns the diffuse RGB components (0..1) the C++ loads from
/// `Side/{id}=<name,r,g,b,Minimap,rMM,gMM,bMM>` entries of `robots.dat`.
///
/// Shipped values (confirmed by dumping `Side` from robots.dat):
///   0 — Neutral = (128,128,128)
///   1 — Player  = (227,158,31)   Yellow
///   2 — AI Red  = (142,0,0)
///   3 — AI Blue = (0,0,150)
///   4 — AI Green= (0,150,0)
pub fn side_color_rgb(side: i32) -> [f32; 3] {
    let c = match side {
        1 => [227u8, 158, 31],
        2 => [142, 0, 0],
        3 => [0, 0, 150],
        4 => [0, 150, 0],
        _ => [128, 128, 128],
    };
    [
        c[0] as f32 / 255.0,
        c[1] as f32 / 255.0,
        c[2] as f32 / 255.0,
    ]
}

/// `GetSideColor` as a packed ARGB dword (alpha 0) — the form the
/// capture paint (`m_TrueColor.m_Color`) consumes.
pub fn side_color_u32(side: i32) -> u32 {
    let c: [u8; 3] = match side {
        1 => [227, 158, 31],
        2 => [142, 0, 0],
        3 => [0, 0, 150],
        4 => [0, 150, 0],
        _ => [128, 128, 128],
    };
    ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32
}

/// Port of `CMatrixMap::GetSideColorMM` (MatrixMap.cpp:1015,
/// MatrixMap.hpp:745) — saturated minimap variant of the side colour.
pub fn side_color_minimap_rgb(side: i32) -> [f32; 3] {
    let c = match side {
        1 => [255u8, 255, 0],
        2 => [255, 0, 0],
        3 => [0, 0, 255],
        4 => [0, 255, 0],
        _ => [128, 128, 128],
    };
    [
        c[0] as f32 / 255.0,
        c[1] as f32 / 255.0,
        c[2] as f32 / 255.0,
    ]
}
