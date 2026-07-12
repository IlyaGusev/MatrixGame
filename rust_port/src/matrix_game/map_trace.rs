//! Port of `MatrixMapTrace.cpp` — the landscape / object ray-cast side
//! of `CMatrixMap::Trace`. This file is the Rust mirror of the C++
//! tracing module; the object-scan branch is implemented as
//! `Objects::pick_object` in map_static.rs and re-exported here.
//!
//! Local pathfinding (`CMatrixMap::FindLocalPath` +
//! `OptimizeMovePath`) used to live here; it now lives in `logic.rs`
//! where its owning class `CMatrixMapLogic` does in the original.

// Re-export the pathfinding primitives that used to live here, so the
// robot code can keep importing `map_trace::{MovePt, MovePath, ...}`
// while the definitive home is `logic.rs`. The re-exports document the
// physical location move without churning every call site.
pub use crate::matrix_game::logic::{
    find_local_path, footprint_passable, optimize_move_path, path_total_length,
    waypoint_to_world, Blocker, LocalPathResult, MovePath, MovePt, MoveRect,
    ROBOT_FOOTPRINT_HALF,
};

/// Object-scan branch of `CMatrixMap::Trace` (MatrixMapTrace.cpp —
/// linear scan variant). Currently lives as `Objects::pick_object` on
/// `map_static.rs::Objects` so callers can `objects.pick_object(...)`
/// ergonomically; the method doc cross-references this file. Ray
/// intersection test + nearest-hit book-keeping all happen there. The
/// re-export below exposes the same function as a free helper so the
/// mirror point with MatrixMapTrace.cpp is addressable by path.
pub fn pick_object(
    objects: &crate::matrix_game::map_static::Objects,
    origin: glam::Vec3,
    dir: glam::Vec3,
    mask: u32,
    skip: Option<crate::matrix_game::map_static::ObjectId>,
) -> Option<(crate::matrix_game::map_static::ObjectId, f32)> {
    objects.pick_object(origin, dir, mask, skip)
}

use crate::matrix_game::common::{TRACE_ANYOBJECT, TRACE_LANDSCAPE, TRACE_WATER, WATER_LEVEL};
use crate::matrix_game::map::{GameMap, GLOBAL_SCALE};
use crate::matrix_game::map_static::{ObjectId, Objects};
use glam::Vec3;

/// Result of [`trace`] — the `TRACE_STOP_*` sentinels of
/// `CMatrixMap::Trace` (MatrixMap.hpp:44-48) made type-safe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceStop {
    None,
    Landscape,
    Water,
    Object(ObjectId),
}

impl TraceStop {
    /// `IS_TRACE_STOP_OBJECT(o)` (MatrixMap.hpp:48).
    pub fn object(self) -> Option<ObjectId> {
        match self {
            TraceStop::Object(id) => Some(id),
            _ => None,
        }
    }
}

/// Möller–Trumbore with backface culling — mirrors the C++
/// `IntersectTriangle` (Math3D.cpp:25-59), whose `det < 0.0001f` guard
/// culls back faces. Terrain triangles are wound face-up, so a ray that
/// starts below the surface and travels up-and-out (e.g. a turret whose
/// geo-center sits inside base terrain) passes through instead of being
/// blocked — without this cull such turrets never acquire a target and
/// never fire. Returns ray parameter `t ≥ 0`.
fn intersect_triangle(orig: Vec3, dir: Vec3, p0: Vec3, p1: Vec3, p2: Vec3) -> Option<f32> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let pv = dir.cross(e2);
    let det = e1.dot(pv);
    if det < 0.0001 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = orig - p0;
    let u = tv.dot(pv) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qv = tv.cross(e1);
    let v = dir.dot(qv) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(qv) * inv;
    if t < 0.0 {
        None
    } else {
        Some(t)
    }
}

/// Test the two heightmap triangles of cell `(x, y)` exactly as the
/// per-cell loop at MatrixMapTrace.cpp:255-282 does:
/// `p0=(x,y+1)  p1=(x,y)  p2=(x+1,y+1)  p3=(x+1,y)`; first triangle
/// `(p0,p1,p2)`, second `(p1,p3,p2)`.
fn cell_trace(map: &GameMap, x: i32, y: i32, start: Vec3, dir: Vec3) -> Option<f32> {
    if x < 0 || y < 0 || x >= map.size_x as i32 || y >= map.size_y as i32 {
        return None;
    }
    let (xu, yu) = (x as usize, y as usize);
    let z00 = map.point(xu, yu).z;
    let z10 = map.point(xu + 1, yu).z;
    let z01 = map.point(xu, yu + 1).z;
    let z11 = map.point(xu + 1, yu + 1).z;
    let fx = x as f32 * GLOBAL_SCALE;
    let fy = y as f32 * GLOBAL_SCALE;
    let p0 = Vec3::new(fx, fy + GLOBAL_SCALE, z01);
    let p1 = Vec3::new(fx, fy, z00);
    let p2 = Vec3::new(fx + GLOBAL_SCALE, fy + GLOBAL_SCALE, z11);
    let p3 = Vec3::new(fx + GLOBAL_SCALE, fy, z10);
    intersect_triangle(start, dir, p0, p1, p2)
        .or_else(|| intersect_triangle(start, dir, p1, p3, p2))
}

/// First landscape intersection along `start → start + dir*max_t`,
/// walking heightmap cells in ray order (2D grid traversal). The C++
/// achieves the same with a group-level + cell-level interleaved DDA
/// (MatrixMapTrace.cpp:30-402 + 591-758); cells are visited near-to-far
/// so the first triangle hit is the nearest.
fn trace_landscape(map: &GameMap, start: Vec3, dir: Vec3, max_t: f32) -> Option<f32> {
    let end = start + dir * max_t;
    let inv = 1.0 / GLOBAL_SCALE;
    let mut cx = (start.x * inv).floor() as i32;
    let mut cy = (start.y * inv).floor() as i32;
    let ex = (end.x * inv).floor() as i32;
    let ey = (end.y * inv).floor() as i32;

    let step_x = if dir.x > 0.0 { 1 } else { -1 };
    let step_y = if dir.y > 0.0 { 1 } else { -1 };

    // Parametric distance to the next cell border on each axis.
    let next_border = |c: i32, step: i32| -> f32 {
        (if step > 0 { (c + 1) as f32 } else { c as f32 }) * GLOBAL_SCALE
    };
    let mut t_max_x = if dir.x.abs() > 1.0e-10 {
        (next_border(cx, step_x) - start.x) / dir.x
    } else {
        f32::INFINITY
    };
    let mut t_max_y = if dir.y.abs() > 1.0e-10 {
        (next_border(cy, step_y) - start.y) / dir.y
    } else {
        f32::INFINITY
    };
    let t_delta_x = if dir.x.abs() > 1.0e-10 {
        GLOBAL_SCALE / dir.x.abs()
    } else {
        f32::INFINITY
    };
    let t_delta_y = if dir.y.abs() > 1.0e-10 {
        GLOBAL_SCALE / dir.y.abs()
    } else {
        f32::INFINITY
    };

    // Hard bound on visited cells — segment can cross at most
    // |Δcx| + |Δcy| + 1 cells.
    let mut remaining = (ex - cx).abs() + (ey - cy).abs() + 1;
    loop {
        if let Some(t) = cell_trace(map, cx, cy, start, dir) {
            if t <= max_t {
                return Some(t);
            }
            // Hit beyond the segment — nothing nearer can follow.
            return None;
        }
        remaining -= 1;
        if remaining <= 0 {
            return None;
        }
        if t_max_x < t_max_y {
            cx += step_x;
            t_max_x += t_delta_x;
        } else {
            cy += step_y;
            t_max_y += t_delta_y;
        }
    }
}

/// Port of `CMatrixMap::Trace` (MatrixMapTrace.cpp:404-803) — segment
/// trace against objects, landscape and the water plane.
///
/// Returns the stop kind and the hit position (`*result`). On
/// `TraceStop::None` the position is `end`, matching the C++ tail.
///
/// Differences from the original, both documented at their source:
/// * objects are tested with the sphere fallback of
///   [`crate::matrix_game::map_static::MapStatic::pick`] until per-mesh
///   `Pick` lands — the C++ only does that under `TRACE_OBJECTSPHERE`;
/// * no per-group object buckets — the arena is scanned linearly
///   (same contract as `Objects::find_objects`).
pub fn trace(
    map: &GameMap,
    objs: &Objects,
    start: Vec3,
    end: Vec3,
    mask: u32,
    skip: Option<ObjectId>,
) -> (TraceStop, Vec3) {
    let d = end - start;
    let len = d.length();
    if len <= 1.0e-10 {
        return (TraceStop::None, end);
    }
    let dir = d / len;

    // Objects — nearest `t` within the segment.
    let mut best_obj: Option<(ObjectId, f32)> = None;
    if mask & TRACE_ANYOBJECT != 0 {
        if let Some((id, t)) = objs.pick_object_within(start, dir, len, mask, skip) {
            if t <= len {
                best_obj = Some((id, t));
            }
        }
    }

    // Landscape — first heightmap-triangle crossing.
    let land_t = if mask & TRACE_LANDSCAPE != 0 {
        trace_landscape(map, start, dir, len)
    } else {
        None
    };

    match (best_obj, land_t) {
        (Some((id, to)), Some(tl)) if to <= tl => {
            return (TraceStop::Object(id), start + dir * to);
        }
        (Some((id, to)), None) => {
            return (TraceStop::Object(id), start + dir * to);
        }
        (_, Some(tl)) => {
            let out = start + dir * tl;
            if out.z > WATER_LEVEL {
                return (TraceStop::Landscape, out);
            }
            // Terrain hit below sea level. The C++ skips the landscape
            // return here and lets the water-plane test fire — but that
            // test is only correct over a real water cell. Over dry
            // sub-sea-level ground (e.g. the basins the below-water
            // turrets sit in) there is no water, so the shot must hit
            // the ground itself, not a phantom surface. Otherwise fall
            // through to the water block, which returns the shallower
            // surface point.
            if !(mask & TRACE_WATER != 0 && water_cell_at(map, out.x, out.y)) {
                return (TraceStop::Landscape, out);
            }
        }
        (None, None) => {}
    }

    // Water plane (MatrixMapTrace.cpp:789-800) — only where a water cell
    // actually sits under the crossing point. The global z=-2 plane
    // would otherwise swallow every shot fired toward or across any
    // below-sea-level dry ground (below-water turrets, deep pits),
    // which is the "below-water / laser turrets don't fire" bug.
    if mask & TRACE_WATER != 0 {
        let minz = start.z.min(end.z);
        let maxz = start.z.max(end.z);
        if minz < WATER_LEVEL && maxz >= WATER_LEVEL && dir.z.abs() > 1.0e-10 {
            let k = (WATER_LEVEL - start.z) / dir.z;
            let out = start + dir * k;
            if water_cell_at(map, out.x, out.y) {
                return (TraceStop::Water, Vec3::new(out.x, out.y, WATER_LEVEL));
            }
        }
    }

    (TraceStop::None, end)
}

/// True if the map cell under world `(wx, wy)` is open water (not a
/// bridge deck). Ports the `IsWater() && !IsBridge()` guard `GetZ` uses
/// (MatrixMap.cpp:621) so the trace's water plane only stops shots where
/// there is genuinely water to splash into.
fn water_cell_at(map: &GameMap, wx: f32, wy: f32) -> bool {
    use crate::matrix_game::common::{CELLFLAG_BRIDGE, CELLFLAG_WATER};
    let x = (wx / GLOBAL_SCALE).floor() as i32;
    let y = (wy / GLOBAL_SCALE).floor() as i32;
    map.unit_flags(x, y)
        .map(|f| f & CELLFLAG_WATER != 0 && f & CELLFLAG_BRIDGE == 0)
        .unwrap_or(false)
}

/// Ray → bounding-sphere distance; stands in for the per-mesh
/// `Pick`/`PickFull` of the effect-placement loops (burn flames,
/// building DIP explosions) until per-mesh picking lands. `dir` must
/// be normalized. Returns the nearest non-negative hit `t`.
pub fn pick_sphere(orig: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = orig - center;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t = if -b - sq >= 0.0 { -b - sq } else { -b + sq };
    (t >= 0.0).then_some(t)
}
