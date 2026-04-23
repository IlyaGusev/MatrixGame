//! Port of `MatrixMapTrace.{cpp,hpp}` — the A* / line-of-sight
//! pathfinder on the fine-grained move grid plus path-smoothing.
//!
//! Entry points mirror the original:
//!   * `find_path` → `CMatrixMap::FindLocalPath` (8-way A* over the
//!     move cells).
//!   * `optimize_path` → `CMatrixMap::OptimizeMovePath` (drops
//!     collinear midpoints with a line-of-sight check).
//!
//! ## Zone-path hint — deferred by design
//!
//! The C++ `FindLocalPath` takes a `zonepath[]` array computed by
//! `CMatrixRobotAI::ZonePathCalc` (MatrixRobot.cpp:1578-1605) that
//! comes from a precomputed road network (`CMatrixRoadNetwork`,
//! 2706 lines at `Logic/MatrixRoadNetwork.cpp`). Its effect inside
//! `FindLocalPath` (MatrixLogic.cpp:1245-1259) is to **restrict the
//! search rectangle** to the union of the listed zones' bboxes. It
//! does not alter which path is chosen when a path exists — A* run
//! over the full map yields the same (or a shorter) route, only
//! slower on very large maps.
//!
//! Porting the zone subsystem requires (a) deserialising the `rn`
//! block from each CMAP (MatrixMapPrepare.cpp:1608 —
//! `m_RN.Load(rnb, ver)`), (b) porting `CMatrixRoadNetwork`'s zone /
//! crotch / group graph, and (c) wiring `CMatrixSide` team / group
//! logic that feeds `FindPathInZone`. None of those side
//! dependencies are in the port yet, and the only observable effect
//! is search speed on large maps. Left as a targeted follow-up.
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
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Port of the `(other_path_list[i], other_des[i])` tuple fed to
/// `CMatrixMap::FindLocalPath` (MatrixRobot.cpp:1630-1643). Each
/// blocker is another live robot (or cannon) whose future footprint
/// should raise the cost of routing through those cells:
///   - `pos`: where the robot currently stands — `path_list[0]` in
///     C++. Weight 30 (MatrixLogic.cpp:1289-1300).
///   - `dest`: where the robot is heading. Weight 200
///     (MatrixLogic.cpp:1278-1287).
///
/// The C++ version originally also walked the *remaining* path and
/// stamped `SetWeightFromTo` along it, but that loop is commented
/// out in the shipped binary (MatrixLogic.cpp:1273-1276), so we
/// faithfully omit it. Blockers influence `find_path` *cost* only —
/// A* can still route through them when no detour exists.
#[derive(Debug, Clone, Copy)]
pub struct Blocker {
    /// Current standing cell (upper-left corner of footprint). `None`
    /// for stationary objects where only the final pos is known.
    pub pos: Option<MovePt>,
    /// Destination cell (upper-left corner of footprint).
    pub dest: MovePt,
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
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn is_active(&self) -> bool {
        !self.pts.is_empty() && self.cur + 1 < self.pts.len()
    }

    pub fn current_segment(&self) -> Option<(MovePt, MovePt)> {
        if self.cur + 1 >= self.pts.len() {
            return None;
        }
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
    crate::matrix_game::logic::is_absence_wall(map, chassis_kind, ROBOT_MOVECELLS_PER_SIZE, mx, my)
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
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// A* on the move grid with 8-way connectivity. Returns a path
/// inclusive of `start` and `goal` as move-cell upper-left corners.
/// The robot's 4×4 footprint must be passable at every cell on the
/// path (see `footprint_passable`).
///
/// `blockers` is a list of `(pos, dest)` cells from other live
/// robots / cannons. Port of the `other_des` / `other_path_list`
/// arguments to `CMatrixMap::FindLocalPath` (MatrixRobot.cpp:1658-
/// 1664 + MatrixLogic.cpp:1217-1301). Each blocker raises the
/// per-cell traversal weight inside a footprint-sized window:
///   - `dest` → weight 200 (line 1285),
///   - `pos`  → weight 30  (line 1297).
///
/// Everything else has a base weight of 5. Rust rescales to 1.0 /
/// 6.0 / 40.0 so the octile heuristic stays admissible at step=1.
///
/// Port of `CMatrixMap::FindLocalPath` (MatrixLogic.cpp:1217). The
/// zone-constraint argument (`zonepath`) is omitted — the regions
/// network isn't ported yet and the C++ uses it purely as an
/// efficiency hint (restricts the search rectangle); correctness
/// is preserved by letting A* see the full map.
pub fn find_path(
    map: &GameMap,
    start: MovePt,
    goal: MovePt,
    chassis_kind: usize,
    blockers: &[Blocker],
) -> Option<Vec<MovePt>> {
    let sx = map.size_move_x as i32;
    let sy = map.size_move_y as i32;

    let in_bounds = |p: MovePt| {
        p.x >= 0
            && p.y >= 0
            && p.x + ROBOT_MOVECELLS_PER_SIZE <= sx
            && p.y + ROBOT_MOVECELLS_PER_SIZE <= sy
    };
    if !in_bounds(start) || !in_bounds(goal) {
        return None;
    }
    if !footprint_passable(map, goal.x, goal.y, chassis_kind) {
        return None;
    }

    // Per-cell traversal weight grid. Base = 1.0; each blocker stamps
    // a footprint window around `pos` (weight 6.0) and `dest`
    // (weight 40.0). `max` between old and new matches C++ `if(w<200)
    // w=200` / `if(w<30) w=30` semantics (MatrixLogic.cpp:1285, 1297).
    let w = sx as usize;
    let h = sy as usize;
    let mut weight = vec![1.0_f32; w * h];
    const W_POS: f32 = 6.0; // C++ 30 / 5
    const W_DEST: f32 = 40.0; // C++ 200 / 5
    let stamp = |grid: &mut [f32], c: MovePt, new_w: f32| {
        // Footprint window = `[c.x-(S-1) .. c.x+S) × [c.y-(S-1) ..
        // c.y+S)` — same `other_size[i]=4` window the C++ uses at
        // :1278-1287 and :1290-1299.
        for dy in -(ROBOT_MOVECELLS_PER_SIZE - 1)..ROBOT_MOVECELLS_PER_SIZE {
            for dx in -(ROBOT_MOVECELLS_PER_SIZE - 1)..ROBOT_MOVECELLS_PER_SIZE {
                let bx = c.x + dx;
                let by = c.y + dy;
                if bx >= 0 && by >= 0 && bx < sx && by < sy {
                    let i = (by as usize) * w + (bx as usize);
                    if grid[i] < new_w {
                        grid[i] = new_w;
                    }
                }
            }
        }
    };
    for b in blockers {
        // Dest first (higher weight) then pos — stamp order matches
        // C++ and the `max` semantics make it order-insensitive.
        stamp(&mut weight, b.dest, W_DEST);
        if let Some(p) = b.pos {
            stamp(&mut weight, p, W_POS);
        }
    }
    // Never penalise our own start cell — the C++ never does because
    // path_list[0] of the *current* robot was never added to its own
    // blocker list.
    weight[(start.y as usize) * w + (start.x as usize)] = 1.0;

    let idx = |p: MovePt| -> usize { (p.y as usize) * w + (p.x as usize) };

    let mut g = vec![f32::INFINITY; w * h];
    let mut parent = vec![(-1i32, -1i32); w * h];
    let mut closed = vec![false; w * h];

    let h_cost = |p: MovePt| -> f32 {
        let dx = (goal.x - p.x).abs() as f32;
        let dy = (goal.y - p.y).abs() as f32;
        // Octile distance — admissible for 8-way grid with min weight
        // 1.0. Safe overestimate for weighted cells (conservative).
        let (a, b) = if dx < dy { (dx, dy) } else { (dy, dx) };
        (b - a) + std::f32::consts::SQRT_2 * a
    };

    let mut open = BinaryHeap::new();
    g[idx(start)] = 0.0;
    open.push(Node {
        f: h_cost(start),
        g: 0.0,
        x: start.x,
        y: start.y,
    });

    const D: f32 = std::f32::consts::SQRT_2;
    const MOVES: [(i32, i32, f32); 8] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, D),
        (1, -1, D),
        (-1, 1, D),
        (-1, -1, D),
    ];

    while let Some(Node { g: gu, x, y, .. }) = open.pop() {
        let u = MovePt::new(x, y);
        if u == goal {
            let mut out = vec![goal];
            let mut cur = goal;
            while cur != start {
                let (px, py) = parent[idx(cur)];
                if px < 0 {
                    return None;
                }
                cur = MovePt::new(px, py);
                out.push(cur);
            }
            out.reverse();
            return Some(out);
        }
        let iu = idx(u);
        if closed[iu] {
            continue;
        }
        closed[iu] = true;

        for (dx, dy, step) in MOVES {
            let v = MovePt::new(u.x + dx, u.y + dy);
            if !in_bounds(v) {
                continue;
            }
            if !footprint_passable(map, v.x, v.y, chassis_kind) {
                continue;
            }
            if dx != 0 && dy != 0 {
                if !footprint_passable(map, u.x + dx, u.y, chassis_kind) {
                    continue;
                }
                if !footprint_passable(map, u.x, u.y + dy, chassis_kind) {
                    continue;
                }
            }

            let iv = idx(v);
            if closed[iv] {
                continue;
            }
            // Step cost = base direction cost × enter-cell weight —
            // matches C++ `smm->m_Find = smm2->m_Find + smm->m_Weight`
            // where the step itself is free and the entered cell's
            // weight dominates (MatrixLogic.cpp:1373).
            let new_g = gu + step * weight[iv];
            if new_g + 1e-5 < g[iv] {
                g[iv] = new_g;
                parent[iv] = (u.x, u.y);
                open.push(Node {
                    f: new_g + h_cost(v),
                    g: new_g,
                    x: v.x,
                    y: v.y,
                });
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
/// **No blocker awareness** — matches C++ where `OptimizeMovePath`
/// takes only `(chassis, size, cnt, path)` and never consults the
/// dynamic-blocker list. Blockers affected A* cost; the optimizer
/// collapses the resulting path on pure terrain passability.
pub fn optimize_path(map: &GameMap, path: &[MovePt], chassis_kind: usize) -> Vec<MovePt> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let mut out = Vec::with_capacity(path.len());
    out.push(path[0]);
    let mut anchor = 0usize;
    let mut i = 1usize;
    while i < path.len() {
        if i + 1 < path.len() && line_of_sight(map, path[anchor], path[i + 1], chassis_kind) {
            i += 1;
        } else {
            out.push(path[i]);
            anchor = i;
            i += 1;
        }
    }
    out
}

fn line_of_sight(map: &GameMap, a: MovePt, b: MovePt, chassis_kind: usize) -> bool {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let steps = dx.abs().max(dy.abs());
    if steps == 0 {
        return footprint_passable(map, a.x, a.y, chassis_kind);
    }
    let fx = dx as f32 / steps as f32;
    let fy = dy as f32 / steps as f32;
    for s in 0..=steps {
        let x = (a.x as f32 + fx * s as f32).round() as i32;
        let y = (a.y as f32 + fy * s as f32).round() as i32;
        if !footprint_passable(map, x, y, chassis_kind) {
            return false;
        }
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
