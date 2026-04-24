//! Port of `MatrixMapTrace.cpp` — the landscape / object ray-cast side
//! of `CMatrixMap::Trace`. This file is the Rust mirror of the C++
//! tracing module; the `CMatrixMap::Trace` object-scan branch is
//! slated for import from `map_static.rs::pick_object` (FS_DIFF).
//!
//! Local pathfinding (`CMatrixMap::FindLocalPath` +
//! `OptimizeMovePath`) used to live here; it now lives in `logic.rs`
//! where its owning class `CMatrixMapLogic` does in the original.

// Re-export the pathfinding primitives that used to live here, so the
// robot code can keep importing `map_trace::{MovePt, MovePath, ...}`
// while the definitive home is `logic.rs`. The re-exports document the
// physical location move without churning every call site.
pub use crate::matrix_game::logic::{
    find_path, footprint_passable, optimize_path, path_total_length, waypoint_to_world, Blocker,
    MovePath, MovePt, ROBOT_FOOTPRINT_HALF,
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
