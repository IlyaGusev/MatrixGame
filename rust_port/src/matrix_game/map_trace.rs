//! Port of `MatrixMapTrace.{cpp,hpp}` — the A* / line-of-sight
//! pathfinder on the fine-grained move grid plus path-smoothing.
//!
//! Entry points mirror the original:
//!   * `find_path` → `CMatrixMap::FindLocalPath` (8-way A* over the
//!     move cells). The C++ also accepts a "zone hint" chain from
//!     `ZonePathCalc` which we defer — the regions network isn't
//!     ported yet. For small maps this has no effect on the
//!     resulting path.
//!   * `optimize_path` → `CMatrixMap::OptimizeMovePath` (drops
//!     collinear midpoints with a line-of-sight check).
//!
//! Waypoint semantics match the original: each path cell
//! `(mx, my)` is the **upper-left corner of the robot's 4×4 move-cell
//! footprint** (ROBOT_MOVECELLS_PER_SIZE = 4, see MatrixMap.hpp:25).
//! The cell center in world coords is `((mx + 2) * GLOBAL_SCALE_MOVE,
//! (my + 2) * GLOBAL_SCALE_MOVE)` — the `+ 2` shifts from upper-left
//! corner to footprint center (MatrixRobot.cpp:1713-1716).
//!
//! `PlaceIsEmpty` / `PlaceFindNear` live on `CMatrixMapLogic` in C++
//! and correspondingly on `crate::matrix_game::logic` in Rust —
//! they need the live arena for robot-proximity checks.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::matrix_game::logic::ROBOT_MOVECELLS_PER_SIZE;
use crate::matrix_game::map::GameMap;

/// Half of the footprint — how far the robot's center is from its
/// upper-left corner in move cells.
pub const ROBOT_FOOTPRINT_HALF: i32 = ROBOT_MOVECELLS_PER_SIZE / 2;

/// A single move-grid waypoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MovePt {
    pub x: i32,
    pub y: i32,
}

impl MovePt {
    pub fn new(x: i32, y: i32) -> Self { Self { x, y } }
}

/// Port of `m_MovePath[]` contents — a contiguous list of move-cell
/// waypoints. Usage mirrors the C++: walk `cur..cnt-1`, each pair
/// `(cur, cur+1)` is the current segment being driven.
#[derive(Debug, Default, Clone)]
pub struct MovePath {
    pub pts: Vec<MovePt>,
    pub cur: usize,
    /// Total length in world units (MatrixRobot.cpp:1686-1688).
    pub total_len: f32,
    /// How far the robot has traveled along the path so far.
    pub followed_len: f32,
}

impl MovePath {
    pub fn clear(&mut self) { *self = Self::default(); }

    pub fn is_active(&self) -> bool {
        !self.pts.is_empty() && self.cur + 1 < self.pts.len()
    }

    pub fn current_segment(&self) -> Option<(MovePt, MovePt)> {
        if self.cur + 1 >= self.pts.len() { return None; }
        Some((self.pts[self.cur], self.pts[self.cur + 1]))
    }
}

/// Passability predicate for a robot-sized footprint anchored at
/// `(mx, my)`. Port of `CMatrixMapLogic::IsAbsenceWall(chassis, 4,
/// mx, my)` (MatrixLogic.cpp:513-523): reads the single
/// precomputed size-4 bit at that cell (`(1 << chassis) << 18`),
/// which the map's CMAP loader already folded in for us. This is
/// much cheaper than iterating the 4×4 footprint — the compiled
/// data already encodes the full-footprint test.
pub fn footprint_passable(map: &GameMap, mx: i32, my: i32, chassis_kind: usize) -> bool {
    crate::matrix_game::logic::is_absence_wall(
        map, chassis_kind, ROBOT_MOVECELLS_PER_SIZE, mx, my,
    )
}

#[derive(Copy, Clone, PartialEq)]
struct Node {
    f: f32,
    g: f32,
    x: i32,
    y: i32,
}
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, o: &Self) -> Ordering {
        // Min-heap by f.
        o.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}

/// A* on the move grid with 8-way connectivity. Returns a path
/// inclusive of `start` and `goal` as move-cell upper-left corners.
/// The robot's 4×4 footprint must be passable at every cell on the
/// path (see `footprint_passable`).
///
/// `blockers` is a list of move-cell positions (upper-left corners
/// of footprints) that A* must treat as impassable for the
/// duration of this search — port of the `other_des` /
/// `other_path_list` arguments the C++ `FindLocalPath` takes at
/// MatrixRobot.cpp:1658-1664. Used to route around other robots'
/// current positions and destinations so the initial path doesn't
/// drive straight into another robot.
///
/// Port of `CMatrixMap::FindLocalPath` (MatrixMapTrace.cpp / the
/// pathfinder in the road-network). We skip the zone constraints;
/// those are efficiency hints in the original, not correctness.
pub fn find_path(
    map: &GameMap,
    start: MovePt,
    goal: MovePt,
    chassis_kind: usize,
    blockers: &[MovePt],
) -> Option<Vec<MovePt>> {
    let sx = map.size_move_x as i32;
    let sy = map.size_move_y as i32;

    let in_bounds = |p: MovePt| {
        p.x >= 0
            && p.y >= 0
            && p.x + ROBOT_MOVECELLS_PER_SIZE <= sx
            && p.y + ROBOT_MOVECELLS_PER_SIZE <= sy
    };
    if !in_bounds(start) || !in_bounds(goal) { return None; }
    if !footprint_passable(map, goal.x, goal.y, chassis_kind) { return None; }

    // Expand each blocker into its footprint cells. Two footprints
    // overlap iff their upper-left corners are within
    // ROBOT_MOVECELLS_PER_SIZE on each axis — the standard AABB
    // test. We store the set as a bitmap for O(1) lookup.
    let w = sx as usize;
    let h = sy as usize;
    let mut blocked = vec![false; w * h];
    for b in blockers {
        if *b == start { continue; } // don't block our own start
        for dy in -(ROBOT_MOVECELLS_PER_SIZE - 1)..=(ROBOT_MOVECELLS_PER_SIZE - 1) {
            for dx in -(ROBOT_MOVECELLS_PER_SIZE - 1)..=(ROBOT_MOVECELLS_PER_SIZE - 1) {
                let bx = b.x + dx;
                let by = b.y + dy;
                if bx >= 0 && by >= 0 && bx < sx && by < sy {
                    blocked[(by as usize) * w + (bx as usize)] = true;
                }
            }
        }
    }
    let is_blocked = |p: MovePt| -> bool {
        if p.x < 0 || p.y < 0 || p.x >= sx || p.y >= sy { return false; }
        blocked[(p.y as usize) * w + (p.x as usize)]
    };

    // Flat index into (sx+1)-wide scratch arrays. We index cells by
    // their upper-left corner so the reachable cell set is
    // sx * sy = map.size_move_x * map.size_move_y.
    let idx = |p: MovePt| -> usize { (p.y as usize) * w + (p.x as usize) };

    // g-score (best known) + parent (for path reconstruction).
    let mut g = vec![f32::INFINITY; w * h];
    let mut parent = vec![(-1i32, -1i32); w * h];
    let mut closed = vec![false; w * h];

    let h_cost = |p: MovePt| -> f32 {
        let dx = (goal.x - p.x).abs() as f32;
        let dy = (goal.y - p.y).abs() as f32;
        // Octile distance — admissible for 8-way uniform-step grid.
        let (a, b) = if dx < dy { (dx, dy) } else { (dy, dx) };
        (b - a) + 1.41421356 * a
    };

    let mut open = BinaryHeap::new();
    g[idx(start)] = 0.0;
    open.push(Node { f: h_cost(start), g: 0.0, x: start.x, y: start.y });

    // 8-way: 4 ortho (cost 1) + 4 diag (cost √2). For diagonals we
    // also require both ortho neighbors be passable so the robot's
    // footprint doesn't corner-clip.
    const MOVES: [(i32, i32, f32); 8] = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.41421356), (1, -1, 1.41421356),
        (-1, 1, 1.41421356), (-1, -1, 1.41421356),
    ];

    while let Some(Node { g: gu, x, y, .. }) = open.pop() {
        let u = MovePt::new(x, y);
        if u == goal {
            // Reconstruct.
            let mut out = vec![goal];
            let mut cur = goal;
            while cur != start {
                let (px, py) = parent[idx(cur)];
                if px < 0 { return None; }
                cur = MovePt::new(px, py);
                out.push(cur);
            }
            out.reverse();
            return Some(out);
        }
        let iu = idx(u);
        if closed[iu] { continue; }
        closed[iu] = true;

        for (dx, dy, step) in MOVES {
            let v = MovePt::new(u.x + dx, u.y + dy);
            if !in_bounds(v) { continue; }
            if !footprint_passable(map, v.x, v.y, chassis_kind) { continue; }
            if is_blocked(v) { continue; }
            // For diagonals, both ortho neighbors must also pass —
            // blocks corner-clipping across a diagonal wall.
            if dx != 0 && dy != 0 {
                if !footprint_passable(map, u.x + dx, u.y, chassis_kind) { continue; }
                if !footprint_passable(map, u.x, u.y + dy, chassis_kind) { continue; }
            }

            let iv = idx(v);
            if closed[iv] { continue; }
            let new_g = gu + step;
            if new_g + 1e-5 < g[iv] {
                g[iv] = new_g;
                parent[iv] = (u.x, u.y);
                open.push(Node { f: new_g + h_cost(v), g: new_g, x: v.x, y: v.y });
            }
        }
    }
    None
}

/// Port of `CMatrixMap::OptimizeMovePath` (MatrixRobot.cpp:1681
/// caller, actual impl in MatrixMapTrace.cpp). Walks the raw A*
/// path and drops intermediate waypoints whose entire line-of-sight
/// segment to the latest kept waypoint is passable — yielding a
/// shorter path of diagonal straight runs.
///
/// `blockers` is the same list handed to `find_path`. The optimizer
/// must not smooth through dynamic blockers or the resulting path
/// will drive straight into another robot that A* had carefully
/// routed around.
pub fn optimize_path(
    map: &GameMap,
    path: &[MovePt],
    chassis_kind: usize,
    blockers: &[MovePt],
) -> Vec<MovePt> {
    if path.len() <= 2 { return path.to_vec(); }
    let mut out = Vec::with_capacity(path.len());
    out.push(path[0]);
    let mut anchor = 0usize;
    let mut i = 1usize;
    while i < path.len() {
        // Try to extend the anchor→path[i] segment. If the segment
        // past `i` still has line-of-sight from anchor, keep going.
        if i + 1 < path.len()
            && line_of_sight(map, path[anchor], path[i + 1], chassis_kind, blockers)
        {
            i += 1;
        } else {
            out.push(path[i]);
            anchor = i;
            i += 1;
        }
    }
    out
}

fn line_of_sight(
    map: &GameMap,
    a: MovePt,
    b: MovePt,
    chassis_kind: usize,
    blockers: &[MovePt],
) -> bool {
    // Bresenham-ish footprint sweep. Each cell on the line must have
    // a passable footprint (terrain/walls) AND not overlap any
    // dynamic blocker's ROBOT_MOVECELLS_PER_SIZE footprint.
    let blocker_hit = |p: MovePt| -> bool {
        for b in blockers {
            if (p.x - b.x).abs() < ROBOT_MOVECELLS_PER_SIZE
                && (p.y - b.y).abs() < ROBOT_MOVECELLS_PER_SIZE
            {
                return true;
            }
        }
        false
    };

    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let steps = dx.abs().max(dy.abs());
    if steps == 0 { return footprint_passable(map, a.x, a.y, chassis_kind); }
    let fx = dx as f32 / steps as f32;
    let fy = dy as f32 / steps as f32;
    for s in 0..=steps {
        let x = (a.x as f32 + fx * s as f32).round() as i32;
        let y = (a.y as f32 + fy * s as f32).round() as i32;
        if !footprint_passable(map, x, y, chassis_kind) { return false; }
        if blocker_hit(MovePt::new(x, y)) { return false; }
    }
    true
}

/// Compute the total world-space length of a sequence of waypoints,
/// matching MatrixRobot.cpp:1686-1688.
pub fn path_total_length(pts: &[MovePt]) -> f32 {
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let mut total = 0.0;
    for w in pts.windows(2) {
        let dx = (w[1].x - w[0].x) as f32;
        let dy = (w[1].y - w[0].y) as f32;
        total += gs * (dx * dx + dy * dy).sqrt();
    }
    total
}

/// Convert a waypoint's upper-left corner to the world-space center
/// of the 4×4 footprint.
pub fn waypoint_to_world(p: MovePt) -> (f32, f32) {
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    (
        (p.x as f32 + ROBOT_FOOTPRINT_HALF as f32) * gs,
        (p.y as f32 + ROBOT_FOOTPRINT_HALF as f32) * gs,
    )
}
