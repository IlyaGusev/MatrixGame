//! Port of the ENEMY-AI slice of `MatrixSide.cpp` — the non-player
//! branch of `CMatrixSideUnit::LogicTakt`:
//!
//! * `TaktHL` (MatrixSide.cpp:2450) — 100ms strategic layer: per-region
//!   statistics, danger map, war-side pick, per-team action variants
//!   (`BestAction`/`CompareAction`), team regrouping/merging, and the
//!   `BuildRobot` / `BuildCannon` production drivers.
//! * `TaktTL` (MatrixSide.cpp:4218) — 10ms team layer: per-logic-group
//!   order upkeep, `WarTL` / `RepairTL`, and place assignment
//!   (`AssignPlace`).
//! * The shared helpers: `Regroup`, `GroupNoTeamRobot`, `ClacSpawnTeam`,
//!   `CalcStrength`, `PlaceInRegion`, `SortRobotList`, `CalcRegionPath`,
//!   `CanMoveNoEnemy`, `FindNearRegionWithUTR`, `BuildRobotMinStrange`.
//!
//! The dead `Logic/MatrixTactics.cpp` class family (CMatrixTactics /
//! CMatrixRule / CMatrixState) is NOT ported — every call site in the
//! original is commented out; this file is the entire live enemy AI.
//!
//! Deliberate deviations from the C++ (each marked at the site):
//! * `m_Team[u].m_Action.m_Type=mlat_Capture` assignment-typos inside
//!   conditions (MatrixSide.cpp:3242/3249) are ported as comparisons.
//! * `m_Team[u]` out-of-bounds read at MatrixSide.cpp:3881 is ported as
//!   the intended `m_Team[k]`.
//! * `ProduceRobot`'s spawn-team region uses (x, y), not the C++
//!   (x, x) typo (MatrixRobot.cpp:2205).

use crate::matrix_game::common::float2int;
use crate::matrix_game::logic::{
    building_ref, cannon_ref, get_map_pos, get_region, get_world_pos, is_live_unit, place_list,
    place_list_grow, robot_mut, robot_ref, MapLogic, ROBOT_MOVECELLS_PER_SIZE,
};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{MapStatic, ObjectId, ObjectType};
use crate::matrix_game::object_building::BuildingType;
use crate::matrix_game::robot::OrderType;
use crate::matrix_game::side::{
    LogicAction, LogicActionType, LogicRegion, Side, SideStatus, Stat, MAX_LOGIC_GROUP,
    REGION_PATH_MAX_CNT,
};
use crate::matrix_game::side_player::{
    can_change_place, chassis_bit, mark_occupied_places, place_get, place_set_data,
    prepare_break_order, zero_places,
};

/// Read-only snapshot of the region graph (near lists + centers) so the
/// AI's many BFS passes don't hold the road-network mutex across calls
/// into helpers that lock it themselves.
struct RegionGraph {
    cnt: usize,
    /// Per region: `(near_region, near_move_mask)`.
    near: Vec<Vec<(i32, u8)>>,
    centers: Vec<(i32, i32)>,
}

impl RegionGraph {
    fn snapshot(map: &GameMap) -> Option<RegionGraph> {
        let rn = map.road_network.as_ref()?.lock().unwrap();
        let cnt = rn.regions.len();
        let mut near = Vec::with_capacity(cnt);
        let mut centers = Vec::with_capacity(cnt);
        for r in &rn.regions {
            near.push(
                r.near
                    .iter()
                    .zip(r.near_move.iter())
                    .map(|(&n, &m)| (n, m))
                    .collect(),
            );
            centers.push((r.center.x, r.center.y));
        }
        Some(RegionGraph { cnt, near, centers })
    }

    /// `IsNerestRegion(r1, r2)` — adjacency test.
    fn is_near(&self, r1: i32, r2: i32) -> bool {
        if r1 < 0 || r2 < 0 || r1 as usize >= self.cnt {
            return false;
        }
        self.near[r1 as usize].iter().any(|&(n, _)| n == r2)
    }
}

/// Place's region (`GetPlacePtr(no)->m_Region`).
fn place_region(map: &GameMap, no: i32) -> i32 {
    if no < 0 {
        return -1;
    }
    let Some(rn_lock) = map.road_network.as_ref() else {
        return -1;
    };
    let rn = rn_lock.lock().unwrap();
    rn.places.get(no as usize).map(|p| p.region).unwrap_or(-1)
}

/// `GetDesRegion(robot)` (MatrixSide.cpp:9472-9478) — region of the
/// robot's assigned place, -1 when unplaced.
fn des_region(map: &GameMap, place: i32) -> i32 {
    place_region(map, place)
}

/// `IsToPlace(robot, place)` (MatrixSide.cpp:9389-9408).
fn is_to_place(map: &GameMap, r: &crate::matrix_game::robot::Robot, place: i32) -> bool {
    let Some((_, _, pos, _)) = place_get(map, place) else {
        return false;
    };
    if let Some(tp) = r.move_to_coords() {
        if r.orders.has(OrderType::CaptureFactory) {
            return false;
        }
        if pos == tp {
            return true;
        }
        if let Some(rt) = r.return_coords() {
            if pos == rt {
                return true;
            }
        }
        false
    } else {
        if let Some(rt) = r.return_coords() {
            if pos == rt {
                return true;
            }
        }
        (r.map_x, r.map_y) == pos
    }
}

fn resize_stats(side: &mut Side, cnt: usize) {
    if side.region_stats.len() < cnt {
        side.region_stats.resize(cnt, LogicRegion::default());
    }
    if side.region_index.len() < cnt {
        side.region_index.resize(cnt, 0);
    }
}

impl MapLogic {
    /// Non-player branch of `CMatrixSideUnit::LogicTakt`
    /// (MatrixSide.cpp:420-600) — `TaktHL` + `TaktTL` + CalcMaxSpeed +
    /// the per-takt robot census.
    pub fn ai_side_logic_takt(&mut self, map: &GameMap, sid: i32) {
        self.drain_ai_spawn_teams(map);

        let Some(idx) = self.other_sides.iter().position(|s| s.id == sid) else {
            return;
        };
        if self.other_sides[idx].status == SideStatus::None {
            return;
        }
        let mut side = std::mem::take(&mut self.other_sides[idx]);

        let t = self.elapsed_ms as i32;
        side.set_stat(Stat::Time, t);

        self.reteam_stuck_robots(map, &mut side);
        self.takt_hl(map, &mut side);
        self.takt_tl(map, &mut side);
        self.ai_calc_max_speed(&side);

        // Robot census tail (MatrixSide.cpp:503-600).
        side.robots_cnt = 0;
        for tm in side.teams.iter_mut() {
            tm.robot_cnt = 0;
        }
        for id in self.objects.iter_units() {
            let Some(r) = robot_ref(&self.objects, id) else {
                continue;
            };
            if !r.is_live() || r.side != sid {
                continue;
            }
            side.robots_cnt += 1;
            let team = r.team;
            if team >= 0 && (team as usize) < side.teams.len() {
                side.teams[team as usize].robot_cnt += 1;
            }
        }

        self.other_sides[idx] = side;
    }

    /// `CalcStrength` (MatrixSide.cpp:1810-1842) — cached side strength.
    pub(crate) fn calc_side_strength(&self, sid: i32, resources: &[i32], strength_mul: f32) -> f32 {
        let mut c_base = 0i32;
        let mut c_building = 0i32;
        let mut s_cannon = 0.0f32;
        let mut s_robot = 0.0f32;
        for id in self.objects.iter_units() {
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            if obj.side() != sid {
                continue;
            }
            match obj.core().obj_type {
                ObjectType::Building => {
                    if let Some(b) = building_ref(&self.objects, id) {
                        if b.is_live() {
                            if b.is_base() {
                                c_base += 1;
                            } else {
                                c_building += 1;
                            }
                        }
                    }
                }
                ObjectType::Cannon => {
                    if is_live_unit(&self.objects, id) {
                        if let Some(c) = cannon_ref(&self.objects, id) {
                            s_cannon += c.get_strength();
                        }
                    }
                }
                ObjectType::RobotAi => {
                    if let Some(r) = robot_ref(&self.objects, id) {
                        if r.is_live() {
                            s_robot += r.strength;
                        }
                    }
                }
                _ => {}
            }
        }
        let mut res = 0i32;
        for &r in resources.iter().take(4) {
            res += r.min(1000);
        }
        res /= 4;
        (5.0 * c_base as f32
            + c_building as f32
            + s_cannon / 2000.0
            + s_robot / 1000.0
            + res as f32 / 100.0)
            * strength_mul
    }

    /// `Regroup` (MatrixSide.cpp:1844-2011) — split logic groups whose
    /// robots drifted >400 apart, group stray robots, merge groups
    /// within 300.
    fn regroup(&mut self, side: &mut Side) {
        let sid = side.id;
        for lg in side.logic_groups.iter_mut() {
            lg.robots_cnt = 0;
        }

        struct R {
            id: ObjectId,
            team: i32,
            group: i32,
            pos: (f32, f32),
        }
        let mut rl: Vec<R> = Vec::new();
        for id in self.objects.iter_units() {
            let Some(r) = robot_ref(&self.objects, id) else {
                continue;
            };
            if r.side != sid {
                continue;
            }
            if r.group_logic >= 0 && (r.group_logic as usize) < MAX_LOGIC_GROUP {
                side.logic_groups[r.group_logic as usize].robots_cnt += 1;
                side.logic_groups[r.group_logic as usize].team = r.team;
            }
            if r.team >= 0 {
                rl.push(R {
                    id,
                    team: r.team,
                    group: r.group_logic,
                    pos: (r.pos_x, r.pos_y),
                });
            }
        }
        let rlcnt = rl.len();

        // Subgroup flood by ≤400 world distance within (team, group).
        let mut subl = vec![-1i32; rlcnt];
        let mut subcnt = 0i32;
        for u in 0..rlcnt {
            if subl[u] >= 0 {
                continue;
            }
            let mut queue = vec![u];
            let mut sme = 0usize;
            subcnt += 1;
            subl[u] = subcnt - 1;
            while sme < queue.len() {
                let cur = queue[sme];
                let v1 = rl[cur].pos;
                for i in (u + 1)..rlcnt {
                    if subl[i] >= 0 || cur == i {
                        continue;
                    }
                    if rl[cur].team != rl[i].team || rl[cur].group != rl[i].group {
                        continue;
                    }
                    let v2 = rl[i].pos;
                    if (v1.0 - v2.0).powi(2) + (v1.1 - v2.1).powi(2) > 400.0f32 * 400.0 {
                        continue;
                    }
                    queue.push(i);
                    subl[i] = subcnt - 1;
                }
                sme += 1;
            }
        }

        // Split each logic group into one group per subgroup.
        'outer: for u in 0..MAX_LOGIC_GROUP {
            if side.logic_groups[u].robots_cnt < 0 {
                continue;
            }
            let mut tl: Vec<i32> = Vec::new();
            for i in 0..rlcnt {
                if rl[i].group != u as i32 {
                    continue;
                }
                if !tl.contains(&subl[i]) {
                    tl.push(subl[i]);
                }
            }
            for i in 1..tl.len() {
                let Some(k) = side.logic_groups.iter().position(|g| g.robots_cnt <= 0) else {
                    break 'outer;
                };
                // Seed the new group from the first subgroup's group.
                let Some(t0) = (0..rlcnt)
                    .find(|&t| rl[t].group == u as i32 && subl[t] == tl[0])
                else {
                    break 'outer;
                };
                side.logic_groups[k] = side.logic_groups[rl[t0].group as usize].clone();
                side.logic_groups[k].robots_cnt = 0;
                for t in 0..rlcnt {
                    if rl[t].group != u as i32 || subl[t] != tl[i] {
                        continue;
                    }
                    // C++ increments the old group's count here
                    // (MatrixSide.cpp:1938) — kept; TaktTL recounts.
                    side.logic_groups[rl[t].group as usize].robots_cnt += 1;
                    rl[t].group = k as i32;
                    if let Some(r) = robot_mut(&mut self.objects, rl[t].id) {
                        r.group_logic = k as i32;
                    }
                    side.logic_groups[k].robots_cnt += 1;
                }
            }
        }

        // Fresh group for ungrouped robots.
        for t in 0..rlcnt {
            if rl[t].group >= 0 && (rl[t].group as usize) < MAX_LOGIC_GROUP {
                continue;
            }
            let Some(i) = side.logic_groups.iter().position(|g| g.robots_cnt <= 0) else {
                break;
            };
            side.logic_groups[i] = Default::default();
            side.logic_groups[i].team = rl[t].team;
            side.logic_groups[i].robots_cnt = 1;
            side.logic_groups[i].action.ty = LogicActionType::None;
            rl[t].group = i as i32;
            if let Some(r) = robot_mut(&mut self.objects, rl[t].id) {
                r.group_logic = i as i32;
            }
        }

        // Merge same-team groups within 300 world units.
        for u in 0..MAX_LOGIC_GROUP {
            if side.logic_groups[u].robots_cnt <= 0 || side.logic_groups[u].team < 0 {
                continue;
            }
            for t in (u + 1)..MAX_LOGIC_GROUP {
                if side.logic_groups[t].robots_cnt <= 0 || side.logic_groups[t].team < 0 {
                    continue;
                }
                if side.logic_groups[u].team != side.logic_groups[t].team {
                    continue;
                }
                let close = rl.iter().any(|a| {
                    a.group == u as i32
                        && rl.iter().any(|b| {
                            b.group == t as i32
                                && (a.pos.0 - b.pos.0).powi(2) + (a.pos.1 - b.pos.1).powi(2)
                                    < 300.0f32 * 300.0
                        })
                });
                if !close {
                    continue;
                }
                if side.logic_groups[u].action.ty == LogicActionType::None {
                    side.logic_groups.swap(u, t);
                }
                let add = side.logic_groups[t].robots_cnt;
                side.logic_groups[u].robots_cnt += add;
                for a in rl.iter_mut().filter(|a| a.group == t as i32) {
                    a.group = u as i32;
                    if let Some(r) = robot_mut(&mut self.objects, a.id) {
                        r.group_logic = u as i32;
                    }
                }
            }
        }
    }

    /// `GroupNoTeamRobot` (MatrixSide.cpp:2257-2341) — pre-placed
    /// team-less robots each get a Defence logic group at their region.
    /// (The C++ clustering loop checks `rl[i]` instead of `rl[u]` and
    /// therefore never clusters — actual behavior is one group per
    /// robot, which we reproduce.)
    pub(crate) fn group_no_team_robot(&mut self, map: &GameMap, sid: i32) {
        if sid == self.player_side.id {
            return;
        }
        let Some(idx) = self.other_sides.iter().position(|s| s.id == sid) else {
            return;
        };
        let mut side = std::mem::take(&mut self.other_sides[idx]);
        for lg in side.logic_groups.iter_mut() {
            lg.robots_cnt = 0;
        }
        let mut strays: Vec<ObjectId> = Vec::new();
        for id in self.objects.iter_units() {
            let Some(r) = robot_ref(&self.objects, id) else {
                continue;
            };
            if r.side != sid {
                continue;
            }
            if r.team >= 0 {
                if r.group_logic >= 0 && (r.group_logic as usize) < MAX_LOGIC_GROUP {
                    side.logic_groups[r.group_logic as usize].robots_cnt += 1;
                    side.logic_groups[r.group_logic as usize].team = r.team;
                }
            } else {
                strays.push(id);
            }
        }
        for rid in strays {
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.group_logic = -1;
            }
            let Some(g) = side.logic_groups.iter().position(|g| g.robots_cnt <= 0) else {
                break;
            };
            let (cx, cy) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (r.pos_x, r.pos_y)
            };
            side.logic_groups[g] = Default::default();
            side.logic_groups[g].team = -1;
            side.logic_groups[g].robots_cnt = 1;
            side.logic_groups[g].action.ty = LogicActionType::Defence;
            let gsm = GameMap::GLOBAL_SCALE_MOVE;
            let region = get_region(map, float2int(cx / gsm), float2int(cy / gsm));
            side.logic_groups[g].action.region = region;
            if region < 0 {
                log::warn!("group_no_team_robot: robot stands in prohibited area");
            }
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.group_logic = g as i32;
            }
        }
        self.other_sides[idx] = side;
    }

    /// `ClacSpawnTeam` (MatrixSide.cpp:2027-2088) — pick the team for a
    /// freshly spawned robot: an empty team if any, else the team of
    /// nearby friendlies (widening tolerance passes), else team 0.
    pub(crate) fn clac_spawn_team(
        &mut self,
        map: &GameMap,
        side: &mut Side,
        region: i32,
        nsh: i32,
    ) -> i32 {
        for i in 0..side.team_cnt as usize {
            if side.teams[i].robot_cnt <= 0 {
                side.clear_team(i);
                return i as i32;
            }
        }
        let Some(g) = RegionGraph::snapshot(map) else {
            return 0;
        };
        if region < 0 || region as usize >= g.cnt {
            return 0;
        }
        resize_stats(side, g.cnt);

        for ct in 0..=2 {
            for r in side.region_stats.iter_mut() {
                r.data = 0;
            }
            let mut queue: Vec<i32> = vec![region];
            side.region_stats[region as usize].data = 1;
            let mut sme = 0usize;
            let mut teamfind = -1i32;
            while sme < queue.len() {
                let cur = queue[sme] as usize;
                for &(u, mv) in &g.near[cur] {
                    let ui = u as usize;
                    if side.region_stats[ui].data != 0 {
                        continue;
                    }
                    if mv & (1u8 << nsh) != 0 {
                        continue;
                    }
                    if ct == 0 && side.region_stats[ui].enemy_robot_cnt > 0 {
                        continue;
                    }
                    if side.region_stats[ui].our_robot_cnt > 0 {
                        for id in self.objects.iter_units() {
                            let Some(r) = robot_ref(&self.objects, id) else {
                                continue;
                            };
                            if !r.is_live() || r.side != side.id || r.team < 0 {
                                continue;
                            }
                            if ct == 2 || r.env.enemy_cnt() <= 0 {
                                let t = r.team as usize;
                                if t < side.teams.len()
                                    && (teamfind < 0
                                        || side.teams[t].robot_cnt
                                            < side.teams[teamfind as usize].robot_cnt)
                                {
                                    teamfind = r.team;
                                }
                            }
                        }
                    }
                    queue.push(u);
                    side.region_stats[ui].data = 1;
                }
                sme += 1;
            }
            if teamfind >= 0 {
                return teamfind;
            }
        }
        0
    }

    /// Assign teams to freshly produced AI robots (the
    /// `side->ClacSpawnTeam` call in `CMatrixRobotAI::ProduceRobot`,
    /// MatrixRobot.cpp:2204-2205).
    fn drain_ai_spawn_teams(&mut self, map: &GameMap) {
        if self.objects.pending_ai_spawn.is_empty() {
            return;
        }
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let rids: Vec<ObjectId> = self.objects.pending_ai_spawn.drain(..).collect();
        for rid in rids {
            let Some((sid, base, nsh)) = robot_ref(&self.objects, rid)
                .map(|r| (r.side, r.base, r.chassis.kind_index() as i32))
            else {
                continue;
            };
            if sid == self.player_side.id {
                continue;
            }
            let region = base
                .and_then(|b| building_ref(&self.objects, b))
                .map(|b| get_region(map, float2int(b.pos.x / gsm), float2int(b.pos.y / gsm)))
                .unwrap_or(-1);
            let Some(idx) = self.other_sides.iter().position(|s| s.id == sid) else {
                continue;
            };
            let mut side = std::mem::take(&mut self.other_sides[idx]);
            let team = self.clac_spawn_team(map, &mut side, region, nsh);
            self.other_sides[idx] = side;
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.set_team(team);
            }
        }
    }

    /// `ZonePathCalc`'s can't-get-there fallback (MatrixRobot.cpp:
    /// 1601-1604): reassign team + drop the logic group.
    fn reteam_stuck_robots(&mut self, map: &GameMap, side: &mut Side) {
        let stuck: Vec<ObjectId> = self
            .objects
            .iter_units()
            .filter(|&id| {
                robot_ref(&self.objects, id)
                    .map(|r| r.side == side.id && r.zone_path_fail_reteam)
                    .unwrap_or(false)
            })
            .collect();
        for rid in stuck {
            let (region, nsh) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (
                    get_region(map, r.map_x, r.map_y),
                    r.chassis.kind_index() as i32,
                )
            };
            let team = self.clac_spawn_team(map, side, region, nsh);
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.zone_path_fail_reteam = false;
                r.set_team(team);
                r.group_logic = -1;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // TaktHL
    // ─────────────────────────────────────────────────────────────────

    /// `TaktHL` (MatrixSide.cpp:2450-4014).
    fn takt_hl(&mut self, map: &GameMap, side: &mut Side) {
        let Some(g) = RegionGraph::snapshot(map) else {
            return;
        };
        if g.cnt == 0 {
            return;
        }
        let sid = side.id;
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let now = self.elapsed_ms as i32;

        self.regroup(side);
        resize_stats(side, g.cnt);

        // Once per 100ms (MatrixSide.cpp:2486).
        if side.last_takt_hl != 0 && now - side.last_takt_hl < 100 {
            return;
        }
        side.last_takt_hl = now;

        side.strength = self.calc_side_strength(sid, &side.resources, side.strength_mul);
        self.escape_from_bomb(map, sid);

        // ── War-side pick (2502-2556) ──────────────────────────────
        if side.next_war_side_calc_time < now {
            // (id, active, strength) for every other side incl. player.
            let mut others: Vec<(i32, bool, f32)> = vec![(
                self.player_side.id,
                self.player_side.status != SideStatus::None,
                self.player_side.strength,
            )];
            for s in &self.other_sides {
                if s.id == 0 {
                    continue; // the placeholder left by mem::take
                }
                others.push((s.id, s.status != SideStatus::None, s.strength));
            }
            if side.war_side < 0 || self.rng.range(0, 2) != 0 {
                // Weakest side below half our strength, else strongest.
                side.war_side = -1;
                let mut mst = f32::MAX;
                if side.strength > 0.0 {
                    for &(id, active, st) in &others {
                        if id == sid || !active || st <= 0.0 {
                            continue;
                        }
                        if st < mst && st < side.strength * 0.5 {
                            mst = st;
                            side.war_side = id;
                        }
                    }
                }
                if side.war_side < 0 {
                    let mut mst = f32::MIN;
                    for &(id, active, st) in &others {
                        if id == sid || !active {
                            continue;
                        }
                        if st > mst {
                            mst = st;
                            side.war_side = id;
                        }
                    }
                }
            } else {
                // Random target (2539-2554); the candidate pool also
                // contains self like the C++ m_Side[] walk.
                let mut all = others.clone();
                all.push((sid, true, side.strength));
                let mut tries = 10;
                side.war_side = -1;
                while tries > 0 {
                    let i = self.rng.range(0, all.len() as i32 - 1) as usize;
                    if all[i].0 != sid && all[i].1 {
                        side.war_side = all[i].0;
                        break;
                    }
                    tries -= 1;
                }
            }
            side.next_war_side_calc_time = now + 60000;
        }

        // ── Reset per-takt stats (2558-2617) ───────────────────────
        side.robots_cnt = 0;
        for lg in side.logic_groups.iter_mut() {
            lg.strength = 0.0;
        }
        let team_cnt = side.team_cnt.clamp(1, side.teams.len() as i32) as usize;
        for tm in side.teams.iter_mut().take(team_cnt) {
            tm.robot_cnt = 0;
            tm.strength = 0.0;
            tm.group_cnt = 0;
            tm.stay = true;
            tm.war = false;
            tm.center_mass = (0, 0);
            tm.radius_mass = 0;
            tm.rect = (1000000000, 1000000000, 0, 0);
            tm.center = (0, 0);
            tm.radius = 0;
            tm.action_list.clear();
            tm.region_near_danger = -1;
            tm.region_far_danger = -1;
            tm.region_near_enemy = -1;
            tm.region_near_retreat = -1;
            tm.region_near_forward = -1;
            tm.region_nearest_base = -1;
            tm.action_prev = tm.action.clone();
            tm.region_next = -1;
            tm.robot_in_des_region = false;
            tm.move_mask = 0;
            tm.region_list.clear();
        }
        for r in side.region_stats.iter_mut() {
            *r = LogicRegion {
                enemy_robot_dist: -1,
                enemy_building_dist: -1,
                our_base_dist: -1,
                ..LogicRegion::default()
            };
        }

        // ── Census (2619-2724) ─────────────────────────────────────
        // (C++ also counts ourbasecnt here but never reads it.)
        for id in self.objects.iter_units() {
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            match obj.core().obj_type {
                ObjectType::Building => {
                    let Some(b) = building_ref(&self.objects, id) else {
                        continue;
                    };
                    let i = get_region(map, (b.pos.x / gsm) as i32, (b.pos.y / gsm) as i32);
                    if i < 0 {
                        continue;
                    }
                    let st = &mut side.region_stats[i as usize];
                    let bside = b.side;
                    if bside == 0 {
                        st.neutral_building_cnt += 1;
                    } else if bside != sid {
                        st.enemy_building_cnt += 1;
                        if bside == side.war_side {
                            st.war_enemy_building_cnt += 1;
                        }
                    } else {
                        st.our_building_cnt += 1;
                    }
                    if b.is_base() {
                        if bside == 0 {
                            st.neutral_base_cnt += 1;
                        } else if bside != sid {
                            st.enemy_base_cnt += 1;
                            if bside == side.war_side {
                                st.war_enemy_base_cnt += 1;
                            }
                        } else {
                            st.our_base_cnt += 1;
                        }
                    }
                }
                ObjectType::RobotAi => {
                    let Some(r) = robot_ref(&self.objects, id) else {
                        continue;
                    };
                    if !r.is_live() {
                        continue;
                    }
                    let tp = ((r.pos_x / gsm) as i32, (r.pos_y / gsm) as i32);
                    let i = get_region(map, tp.0, tp.1);
                    if i >= 0 {
                        if r.side == 0 {
                            // neutral robots don't exist, but keep parity
                        } else if r.side != sid {
                            let st = &mut side.region_stats[i as usize];
                            st.enemy_robot_cnt += 1;
                            if r.side == side.war_side {
                                st.war_enemy_robot_cnt += 1;
                            }
                            st.danger += r.strength * side.danger_mul;
                        } else {
                            side.region_stats[i as usize].our_robot_cnt += 1;
                        }
                    }
                    if r.side == sid {
                        side.robots_cnt += 1;
                        let team = r.team;
                        if team >= 0 && (team as usize) < team_cnt {
                            let tm = &mut side.teams[team as usize];
                            tm.robot_cnt += 1;
                            tm.strength += r.strength;
                            tm.center_mass.0 += tp.0;
                            tm.center_mass.1 += tp.1;
                            tm.rect.0 = tm.rect.0.min(tp.0);
                            tm.rect.1 = tm.rect.1.min(tp.1);
                            tm.rect.2 = tm.rect.2.max(tp.0);
                            tm.rect.3 = tm.rect.3.max(tp.1);
                            tm.move_mask |= chassis_bit(r);
                            if i >= 0 && i == tm.action.region {
                                tm.robot_in_des_region = true;
                            }
                            if i >= 0 {
                                match tm.region_list.iter_mut().find(|(rg, _)| *rg == i) {
                                    Some((_, c)) => *c += 1,
                                    None => tm.region_list.push((i, 1)),
                                }
                            }
                            if !tm.war {
                                tm.war = r.env.enemy_cnt() > 0;
                            }
                            if r.orders.has(OrderType::MoveTo)
                                || r.orders.has(OrderType::MoveReturn)
                            {
                                tm.stay = false;
                            }
                        }
                        if r.group_logic >= 0 && (r.group_logic as usize) < MAX_LOGIC_GROUP {
                            side.logic_groups[r.group_logic as usize].strength += r.strength;
                        }
                    }
                }
                ObjectType::Cannon => {
                    let Some(c) = cannon_ref(&self.objects, id) else {
                        continue;
                    };
                    let tp = ((c.pos.x / gsm) as i32, (c.pos.y / gsm) as i32);
                    let i = get_region(map, tp.0, tp.1);
                    if i < 0 {
                        continue;
                    }
                    let strength = c.get_strength();
                    let st = &mut side.region_stats[i as usize];
                    if c.side == 0 {
                        st.neutral_cannon_cnt += 1;
                        st.danger_add += strength * side.danger_mul;
                    } else if c.side != sid {
                        st.enemy_cannon_cnt += 1;
                        if c.side == side.war_side {
                            st.war_enemy_cannon_cnt += 1;
                        }
                        st.danger_add += strength * side.danger_mul;
                    } else {
                        st.our_cannon_cnt += 1;
                    }
                }
                _ => {}
            }
        }

        // ── Grow danger into neighbours (2727-2769) ────────────────
        for i in 0..g.cnt {
            if side.region_stats[i].danger <= 0.0 {
                continue;
            }
            for r in side.region_stats.iter_mut() {
                r.data = 0;
            }
            let d = side.region_stats[i].danger;
            for &(t, _) in &g.near[i] {
                let tu = t as usize;
                if side.region_stats[tu].data > 0 {
                    continue;
                }
                side.region_stats[tu].danger_add += d;
                side.region_stats[tu].data = 1;
            }
        }
        for r in side.region_stats.iter_mut() {
            r.danger += r.danger_add;
        }

        // ── Per-team aggregates (2771-2976) ────────────────────────
        for i in 0..team_cnt {
            if side.teams[i].robot_cnt <= 0 {
                let tm = &mut side.teams[i];
                tm.region_mass = -1;
                tm.region_mass_prev = -1;
                tm.brave = 0;
                tm.brave_strange_cancel = 0.0;
                tm.war = false;
                continue;
            }
            if side.teams[i].war {
                side.teams[i].regroup_only_after_war = false;
            }
            side.teams[i].region_list.sort_by(|a, b| b.1.cmp(&a.1));
            {
                let tm = &mut side.teams[i];
                tm.center_mass.0 /= tm.robot_cnt;
                tm.center_mass.1 /= tm.robot_cnt;
                tm.center = ((tm.rect.0 + tm.rect.2) / 2, (tm.rect.1 + tm.rect.3) / 2);
            }
            if side.teams[i].region_list.is_empty() {
                continue;
            }

            // Current region `cr` (2802-2812).
            let mut cr = -1i32;
            {
                let tm = &side.teams[i];
                let k = tm.region_list[0].1;
                if tm.action.ty != LogicActionType::None {
                    for &(rg, c) in &tm.region_list {
                        if c != k {
                            break;
                        }
                        if tm.action.region == rg {
                            cr = tm.action.region;
                            break;
                        }
                    }
                }
                if cr < 0 {
                    cr = tm.region_list[0].0;
                }
            }

            // If destination is adjacent and full, treat as arrived
            // (2816-2851).
            let (act_ty, act_region) = (side.teams[i].action.ty, side.teams[i].action.region);
            if act_ty != LogicActionType::None
                && act_region >= 0
                && act_region != cr
                && g.is_near(cr, act_region)
            {
                let region_place: Vec<i32> = {
                    let rn = map.road_network.as_ref().unwrap().lock().unwrap();
                    rn.regions
                        .get(act_region as usize)
                        .map(|r| r.place.clone())
                        .unwrap_or_default()
                };
                zero_places(map, &region_place);
                // Robots of team i whose place isn't in the target region.
                let mut outsiders: Vec<(ObjectId, u8)> = Vec::new();
                for id in self.objects.iter_units() {
                    let Some(r) = robot_ref(&self.objects, id) else {
                        continue;
                    };
                    if r.is_live() && r.side == sid && r.team == i as i32 {
                        let pr = place_region(map, r.env.place);
                        if r.env.place < 0 || pr != act_region {
                            outsiders.push((id, chassis_bit(r)));
                        }
                    }
                }
                mark_occupied_places(map, &self.objects);
                if !outsiders.is_empty() {
                    let mut any_can_place = false;
                    'rl: for &(_, cbit) in &outsiders {
                        for &pi in &region_place {
                            let Some((data, mv, _, _)) = place_get(map, pi) else {
                                continue;
                            };
                            if data != 0 || mv & cbit != 0 {
                                continue;
                            }
                            any_can_place = true;
                            break 'rl;
                        }
                    }
                    if !any_can_place {
                        cr = act_region;
                    }
                }
            }
            if side.teams[i].region_mass != cr {
                side.teams[i].region_mass_prev = side.teams[i].region_mass;
                side.teams[i].region_mass = cr;
            }

            // Next region on the path (2857-2931).
            if side.teams[i].action.ty != LogicActionType::None {
                let path = side.teams[i].action.region_path.clone();
                let rm = side.teams[i].region_mass;
                let mut on_path = false;
                for u in 0..path.len() {
                    if path[u] == rm {
                        if u + 1 < path.len() {
                            side.teams[i].region_next = path[u + 1];
                        }
                        on_path = true;
                        break;
                    }
                }
                if path.len() >= 2 && !on_path && rm >= 0 {
                    for r in side.region_stats.iter_mut() {
                        r.data = 0;
                    }
                    for (u, &pr) in path.iter().enumerate() {
                        side.region_stats[pr as usize].data = (-((u as i32) + 1)) as u32;
                    }
                    let mut queue: Vec<i32> = vec![rm];
                    let mut sme = 0usize;
                    let mut k = -1i32;
                    let mut level = 1u32;
                    side.region_stats[rm as usize].data = level;
                    level += 1;
                    let mut next = queue.len();
                    while sme < next {
                        for &(u, _) in &g.near[queue[sme] as usize] {
                            let ud = side.region_stats[u as usize].data as i32;
                            if ud > 0 {
                                continue;
                            }
                            if k >= 0 {
                                if (side.region_stats[k as usize].data as i32) < ud {
                                    k = u;
                                }
                            } else if side.region_stats[u as usize].data == 0 {
                                queue.push(u);
                                side.region_stats[u as usize].data = level;
                            } else {
                                k = u;
                            }
                        }
                        sme += 1;
                        if sme >= next {
                            next = queue.len();
                            // C++ `if(k) break;` — breaks for k==-1 too.
                            if k != 0 {
                                break;
                            }
                            level += 1;
                        }
                    }
                    if k >= 0 {
                        let mut level = side.region_stats[k as usize].data;
                        loop {
                            let mut p = -1i32;
                            for &(u, _) in &g.near[k as usize] {
                                let ud = side.region_stats[u as usize].data as i32;
                                if ud <= 0 {
                                    continue;
                                }
                                if (ud as u32) < level {
                                    p = u;
                                    break;
                                }
                            }
                            if p < 0 {
                                break;
                            }
                            if side.region_stats[p as usize].data <= 1 {
                                side.teams[i].region_next = k;
                                break;
                            }
                            k = p;
                            level = side.region_stats[k as usize].data;
                        }
                    }
                }
            }

            // Near enemy / danger regions (2933-2975).
            let rm = side.teams[i].region_mass;
            if rm >= 0 {
                if side.region_stats[rm as usize].enemy_robot_cnt > 0 {
                    side.teams[i].region_near_enemy = rm;
                } else {
                    for &(t, _) in &g.near[rm as usize] {
                        if side.region_stats[t as usize].enemy_robot_cnt > 0 {
                            side.teams[i].region_near_enemy = t;
                            break;
                        }
                    }
                }
                let mut md = 0.0f32;
                let mut md2 = 0.0f32;
                if side.region_stats[rm as usize].danger > 0.0 {
                    side.teams[i].region_near_danger = rm;
                    side.teams[i].region_far_danger = rm;
                    md = side.region_stats[rm as usize].danger;
                    md2 = md;
                }
                for &(t, _) in &g.near[rm as usize] {
                    let dt = side.region_stats[t as usize].danger;
                    if dt > md {
                        md = dt;
                        side.teams[i].region_near_danger = t;
                    }
                    if dt > md2 {
                        md2 = dt;
                        side.teams[i].region_far_danger = t;
                    }
                    for &(p, _) in &g.near[t as usize] {
                        let dp = side.region_stats[p as usize].danger;
                        if dp > md2 {
                            md2 = dp;
                            side.teams[i].region_far_danger = p;
                        }
                    }
                }
            }
        }

        // Team radii (2977-2994).
        for id in self.objects.iter_units() {
            let Some(r) = robot_ref(&self.objects, id) else {
                continue;
            };
            if !r.is_live() || r.side != sid {
                continue;
            }
            let tp = ((r.pos_x / gsm) as i32, (r.pos_y / gsm) as i32);
            let team = r.team;
            if team >= 0 && (team as usize) < team_cnt {
                let tm = &mut side.teams[team as usize];
                let d1 = (tm.center_mass.0 - tp.0).pow(2) + (tm.center_mass.1 - tp.1).pow(2);
                let d2 = (tm.center.0 - tp.0).pow(2) + (tm.center.1 - tp.1).pow(2);
                tm.radius_mass = tm.radius_mass.max(d1);
                tm.radius = tm.radius.max(d2);
            }
        }
        for tm in side.teams.iter_mut().take(team_cnt) {
            tm.radius_mass = (tm.radius_mass as f64).sqrt() as i32;
            tm.radius = (tm.radius as f64).sqrt() as i32;
        }

        // Group counts + wait-union (2996-3026).
        for i in 0..team_cnt {
            side.teams[i].group_cnt = side
                .logic_groups
                .iter()
                .filter(|lg| lg.robots_cnt > 0 && lg.team == i as i32)
                .count() as i32;
        }
        for i in 0..team_cnt {
            let (group_cnt, robot_cnt) = (side.teams[i].group_cnt, side.teams[i].robot_cnt);
            if group_cnt <= 1 || robot_cnt <= 1 {
                side.teams[i].wait_union = false;
                continue;
            }
            let mut groupms: Option<usize> = None;
            for (u, lg) in side.logic_groups.iter().enumerate() {
                if lg.robots_cnt <= 0 || lg.team != i as i32 {
                    continue;
                }
                if groupms
                    .map(|m| lg.strength > side.logic_groups[m].strength)
                    .unwrap_or(true)
                {
                    groupms = Some(u);
                }
            }
            let Some(gm) = groupms else {
                side.teams[i].wait_union = false;
                continue;
            };
            if now - side.teams[i].wait_union_last < 5000 {
                continue;
            }
            let ratio = if side.teams[i].strength > 0.0 {
                side.logic_groups[gm].strength / side.teams[i].strength
            } else {
                1.0
            };
            side.teams[i].wait_union = !side.teams[i].stay && ratio <= 0.8;
            side.teams[i].wait_union_last = now;
        }

        // ── Distance waves (3034-3119) ─────────────────────────────
        let wave = |stats: &mut Vec<LogicRegion>,
                    seed: &dyn Fn(&LogicRegion) -> bool,
                    get: &dyn Fn(&LogicRegion) -> i32,
                    set: &dyn Fn(&mut LogicRegion, i32)| {
            let mut queue: Vec<i32> = Vec::new();
            let mut dist = 0i32;
            for (i, st) in stats.iter_mut().enumerate() {
                if seed(st) {
                    queue.push(i as i32);
                    set(st, dist);
                }
            }
            let mut sme = 0usize;
            let mut next = queue.len();
            dist += 1;
            while sme < queue.len() {
                for &(u, _) in &g.near[queue[sme] as usize] {
                    if get(&stats[u as usize]) >= 0 {
                        continue;
                    }
                    queue.push(u);
                    set(&mut stats[u as usize], dist);
                }
                sme += 1;
                if sme >= next {
                    next = queue.len();
                    dist += 1;
                }
            }
        };
        wave(
            &mut side.region_stats,
            &|s| s.enemy_robot_cnt > 0,
            &|s| s.enemy_robot_dist,
            &|s, d| s.enemy_robot_dist = d,
        );
        wave(
            &mut side.region_stats,
            &|s| s.enemy_building_cnt > 0,
            &|s| s.enemy_building_dist,
            &|s, d| s.enemy_building_dist = d,
        );
        wave(
            &mut side.region_stats,
            &|s| s.our_base_cnt > 0,
            &|s| s.our_base_dist,
            &|s, d| s.our_base_dist = d,
        );

        // Retreat region (3121-3141).
        for i in 0..team_cnt {
            let tm = &side.teams[i];
            if tm.robot_cnt <= 0 || tm.region_mass < 0 || tm.region_near_danger < 0 {
                continue;
            }
            let rm = tm.region_mass;
            let strength = tm.strength;
            let mut md = f32::MAX;
            let mut retreat = side.teams[i].region_near_retreat;
            for &(t, _) in &g.near[rm as usize] {
                let st = &side.region_stats[t as usize];
                if (strength * 0.4).max(1.0) >= st.danger {
                    if st.danger < md {
                        md = st.danger;
                        retreat = t;
                    } else if st.danger == md
                        && retreat >= 0
                        && st.our_base_dist
                            < side.region_stats[retreat as usize].our_base_dist
                    {
                        md = st.danger;
                        retreat = t;
                    }
                }
            }
            side.teams[i].region_near_retreat = retreat;
        }

        // Nearest own base (3144-3160).
        let our_bases: Vec<i32> = self
            .objects
            .iter_units()
            .filter_map(|id| building_ref(&self.objects, id))
            .filter(|b| b.is_base() && b.side == sid)
            .map(|b| get_region(map, (b.pos.x / gsm) as i32, (b.pos.y / gsm) as i32))
            .filter(|&u| u >= 0)
            .collect();
        for i in 0..team_cnt {
            let rm = side.teams[i].region_mass;
            if rm < 0 {
                continue;
            }
            let mut best: Option<(i32, i32)> = None;
            for &u in &our_bases {
                let c1 = g.centers[rm as usize];
                let c2 = g.centers[u as usize];
                let t = (c1.0 - c2.0).pow(2) + (c1.1 - c2.1).pow(2);
                if best.map(|(_, bd)| t < bd).unwrap_or(true) {
                    best = Some((u, t));
                }
            }
            if let Some((u, _)) = best {
                side.teams[i].region_nearest_base = u;
            }
        }

        // ── Production (3162-3166) ─────────────────────────────────
        self.build_robot(map, side);
        self.build_cannon(map, side);

        // ── Bravery (3169-3197) ────────────────────────────────────
        let max_robots = self.compute_max_side_robots(sid);
        for i in 0..team_cnt {
            if side.teams[i].robot_cnt <= 0 {
                continue;
            }
            if side.teams[i].brave != 0
                && now - side.teams[i].brave > 10000
                && side.teams[i].strength < side.teams[i].brave_strange_cancel
            {
                side.teams[i].brave = 0;
            }
            if side.teams[i].brave == 0 {
                let mut bravecnt = max_robots;
                bravecnt = float2int(side.brave_mul * bravecnt as f32).min(bravecnt);
                if bravecnt < 1 {
                    bravecnt = 1;
                }
                if side.teams[i].robot_cnt >= bravecnt {
                    side.teams[i].brave = now;
                    side.teams[i].brave_strange_cancel = side.teams[i].strength * 0.3;
                    if side.teams[i].action.ty == LogicActionType::Retreat {
                        side.teams[i].action.ty = LogicActionType::None;
                        side.teams[i].action_time = now;
                    }
                }
            }
        }

        // ── Current-task validity (3204-3326) ──────────────────────
        let mut i = 0usize;
        while i < team_cnt {
            if side.teams[i].robot_cnt <= 0 {
                i += 1;
                continue;
            }
            side.teams[i].l_ok = true;
            match side.teams[i].action.ty {
                LogicActionType::None => {
                    side.teams[i].l_ok = false;
                }
                LogicActionType::Defence => {
                    if side.teams[i].war {
                        i += 1;
                        continue;
                    }
                    let tm = &side.teams[i];
                    if tm.brave != 0 && !tm.wait_union {
                        side.teams[i].l_ok = false;
                        i += 1;
                        continue;
                    }
                    if tm.region_far_danger < 0
                        && now - tm.action_time > 10000
                        && !tm.wait_union
                    {
                        side.teams[i].l_ok = false;
                        i += 1;
                        continue;
                    }
                    if tm.region_far_danger >= 0
                        && now - tm.action_time > 1000
                        && side.region_stats[tm.region_far_danger as usize].danger < tm.strength
                        && !tm.wait_union
                    {
                        side.teams[i].l_ok = false;
                        i += 1;
                        continue;
                    }
                    let ar = tm.action.region;
                    if tm.robot_in_des_region
                        && ar >= 0
                        && (side.region_stats[ar as usize].neutral_building_cnt > 0
                            || side.region_stats[ar as usize].enemy_building_cnt > 0)
                    {
                        side.teams[i].action.ty = LogicActionType::Capture;
                        side.teams[i].action_time = now;
                        i += 1;
                        continue;
                    }
                }
                LogicActionType::Attack => {
                    side.teams[i].l_ok = side.teams[i].war;
                }
                LogicActionType::Forward => {
                    if side.teams[i].wait_union {
                        side.teams[i].l_ok = false;
                        i += 1;
                        continue;
                    }
                    let rn = side.teams[i].region_next;
                    if rn >= 0
                        && side.region_stats[rn as usize].danger * 0.6 > side.teams[i].strength
                    {
                        side.teams[i].l_ok = false;
                        i += 1;
                        continue;
                    }
                    side.teams[i].l_ok =
                        side.teams[i].region_mass != side.teams[i].action.region;
                    if side.teams[i].l_ok && side.teams[i].war {
                        side.teams[i].l_ok = false;
                    }

                    // Passing a capturable building → capture here
                    // (3239-3260). The C++ conditions contain
                    // `m_Action.m_Type=mlat_Capture` assignment typos;
                    // ported as the intended comparisons.
                    let rm = side.teams[i].region_mass;
                    if side.teams[i].l_ok
                        && rm >= 0
                        && (side.region_stats[rm as usize].neutral_building_cnt > 0
                            || side.region_stats[rm as usize].enemy_building_cnt > 0)
                    {
                        let mut claimed = false;
                        for u in 0..team_cnt {
                            if u == i {
                                continue;
                            }
                            let tu = &side.teams[u];
                            if matches!(
                                tu.action.ty,
                                LogicActionType::Capture | LogicActionType::Forward
                            ) && tu.action.region == rm
                                && (tu.action.region == tu.region_mass
                                    || g.is_near(tu.action.region, tu.region_mass))
                            {
                                claimed = true;
                                break;
                            }
                        }
                        if !claimed {
                            for u in 0..team_cnt {
                                if u == i {
                                    continue;
                                }
                                if matches!(
                                    side.teams[u].action.ty,
                                    LogicActionType::Capture | LogicActionType::Forward
                                ) && side.teams[u].action.region == rm
                                {
                                    side.teams[u].action.ty = LogicActionType::None;
                                    side.teams[u].action_time = now;
                                }
                            }
                            side.teams[i].action.ty = LogicActionType::Capture;
                            side.teams[i].action.region = rm;
                            side.teams[i].action_time = now;
                        }
                    }

                    // A capturable building in the destination while
                    // some robot already arrived → switch to capture
                    // (3263-3277).
                    let ar = side.teams[i].action.region;
                    if side.teams[i].l_ok
                        && ar >= 0
                        && (side.region_stats[ar as usize].neutral_building_cnt > 0
                            || side.region_stats[ar as usize].enemy_building_cnt > 0)
                    {
                        let mut arrived = false;
                        for id in self.objects.iter_units() {
                            let Some(r) = robot_ref(&self.objects, id) else {
                                continue;
                            };
                            if r.is_live() && r.side == sid && r.team == i as i32 {
                                let rr = get_region(map, r.map_x, r.map_y);
                                if rr == ar {
                                    arrived = true;
                                    break;
                                }
                            }
                        }
                        if arrived {
                            side.teams[i].action.ty = LogicActionType::Capture;
                            side.teams[i].action_time = now;
                            // C++ `i--; continue;` — re-evaluate team i.
                            continue;
                        }
                    }
                }
                LogicActionType::Retreat => {
                    side.teams[i].l_ok = !side.teams[i].war;
                    if side.teams[i].l_ok
                        && side.teams[i].action.region == side.teams[i].region_mass
                    {
                        let fd = side.teams[i].region_far_danger;
                        if fd >= 0 {
                            let fd_danger = side.region_stats[fd as usize].danger;
                            if side.teams[i].strength * 0.9 > fd_danger
                                && !side.teams[i].wait_union
                            {
                                side.teams[i].action.ty = LogicActionType::Forward;
                                side.teams[i].action.region = fd;
                                side.teams[i].action_time = now;
                                i += 1;
                                continue;
                            } else if side.teams[i].strength > fd_danger * 0.6 {
                                side.teams[i].action.ty = LogicActionType::Defence;
                                side.teams[i].action_time = now;
                                i += 1;
                                continue;
                            }
                        }
                        side.teams[i].l_ok = false;
                    }
                }
                LogicActionType::Capture => {
                    let rn = side.teams[i].region_next;
                    if rn >= 0
                        && side.region_stats[rn as usize].danger * 0.6 > side.teams[i].strength
                    {
                        side.teams[i].l_ok = false;
                        i += 1;
                        continue;
                    }
                    let ar = side.teams[i].action.region;
                    side.teams[i].l_ok = ar >= 0
                        && (side.region_stats[ar as usize].neutral_building_cnt > 0
                            || side.region_stats[ar as usize].enemy_building_cnt > 0);
                    if side.teams[i].l_ok && side.teams[i].war {
                        side.teams[i].l_ok = false;
                    }
                }
                LogicActionType::Intercept => {
                    side.teams[i].l_ok = false;
                }
            }
            i += 1;
        }

        // ── Forward variants (3329-3425) ───────────────────────────
        for i in 0..team_cnt {
            if side.teams[i].l_ok
                || side.teams[i].robot_cnt <= 0
                || side.teams[i].wait_union
                || side.teams[i].region_mass < 0
            {
                continue;
            }
            for r in side.region_stats.iter_mut() {
                r.data = 0;
            }
            let rm = side.teams[i].region_mass;
            let mut queue: Vec<i32> = vec![rm];
            side.region_stats[rm as usize].data = 1;
            let mut sme = 0usize;
            let mut next = queue.len();
            let mut dist = 1i32;
            while sme < queue.len() {
                let cur = queue[sme] as usize;
                for near_i in 0..g.near[cur].len() {
                    let (u, mv) = g.near[cur][near_i];
                    let uu = u as usize;
                    if side.region_stats[uu].data != 0 {
                        continue;
                    }
                    if mv & side.teams[i].move_mask != 0 {
                        continue;
                    }
                    if side.teams[i].brave == 0
                        && side.teams[i].strength < side.region_stats[uu].danger * 0.7
                    {
                        continue;
                    }
                    side.region_stats[uu].data = (1 + dist) as u32;
                    queue.push(u);

                    // Teams must not converge on one region (3360-3370).
                    let mut conflict = false;
                    for k in 0..team_cnt {
                        if k == i || side.teams[k].robot_cnt <= 0 {
                            continue;
                        }
                        if side.teams[k].l_ok {
                            if side.teams[k].action.ty != LogicActionType::None
                                && side.teams[k].action.region == u
                            {
                                conflict = true;
                                break;
                            }
                        } else if !side.teams[k].action_list.is_empty()
                            && side.teams[k].action_list[0].ty != LogicActionType::None
                            && side.teams[k].action_list[0].region == u
                        {
                            conflict = true;
                            break;
                        }
                    }
                    if conflict {
                        continue;
                    }

                    let st = &side.region_stats[uu];
                    let take = st.war_enemy_building_cnt > 0
                        || (st.enemy_building_cnt > 0
                            && dist <= 1
                            && side.teams[i].strength * 0.33 > st.danger)
                        || st.neutral_building_cnt > 0
                        || st.war_enemy_robot_cnt > 0;
                    if take {
                        let mut ac = LogicAction {
                            ty: LogicActionType::Forward,
                            region: u,
                            region_path: Vec::new(),
                        };
                        let mm_team = side.teams[i].move_mask;
                        Self::calc_region_path(&g, side, &mut ac, u, mm_team);
                        side.teams[i].action_list.push(ac);
                        // LiveAction (4210-4213).
                        if side.teams[i].action_list.len() >= 16 {
                            Self::best_action(side, i);
                        }
                    }
                }
                sme += 1;
                if sme >= next {
                    next = queue.len();
                    dist += 1;
                }
            }
            Self::best_action(side, i);
        }

        // Defence fallback variant (3427-3440).
        for i in 0..team_cnt {
            let tm = &side.teams[i];
            if tm.robot_cnt <= 0
                || tm.l_ok
                || !tm.action_list.is_empty()
                || tm.region_mass < 0
                || tm.war
            {
                continue;
            }
            let rm = tm.region_mass;
            side.teams[i].action_list.push(LogicAction {
                ty: LogicActionType::Defence,
                region: rm,
                region_path: vec![rm],
            });
        }

        // Other action variants (3473-3585).
        for i in 0..team_cnt {
            if side.teams[i].l_ok || side.teams[i].robot_cnt <= 0 {
                continue;
            }
            let rm = side.teams[i].region_mass;
            if side.teams[i].war && side.teams[i].action_list.len() < 16 {
                side.teams[i].action_list.push(LogicAction {
                    ty: LogicActionType::Attack,
                    region: rm,
                    region_path: vec![rm],
                });
            }
            if side.teams[i].war
                && side.teams[i].region_near_enemy >= 0
                && side.teams[i].action_list.len() < 16
            {
                let ne = side.teams[i].region_near_enemy;
                side.teams[i].action_list.push(LogicAction {
                    ty: LogicActionType::Attack,
                    region: ne,
                    region_path: vec![rm, ne],
                });
            }
            if side.teams[i].war && side.teams[i].action_list.len() < 16 {
                side.teams[i].action_list.push(LogicAction {
                    ty: LogicActionType::Defence,
                    region: rm,
                    region_path: vec![rm],
                });
            }
            if side.teams[i].war
                && side.teams[i].region_near_danger > 0
                && side.region_stats[side.teams[i].region_near_danger as usize].danger
                    > side.teams[i].strength
                && side.teams[i].action_list.len() < 16
            {
                side.teams[i].action_list.push(LogicAction {
                    ty: LogicActionType::Defence,
                    region: rm,
                    region_path: vec![rm],
                });
            }
            if !side.teams[i].war
                && side.teams[i].brave == 0
                && side.teams[i].region_near_danger > 0
                && side.region_stats[side.teams[i].region_near_danger as usize].danger * 0.8
                    > side.teams[i].strength
                && side.teams[i].region_near_retreat >= 0
            {
                let nr = side.teams[i].region_near_retreat;
                side.teams[i].action_list.push(LogicAction {
                    ty: LogicActionType::Retreat,
                    region: nr,
                    region_path: vec![rm, nr],
                });
            }
            if !side.teams[i].war && side.teams[i].wait_union {
                side.teams[i].action_list.push(LogicAction {
                    ty: LogicActionType::Defence,
                    region: rm,
                    region_path: vec![rm],
                });
            }
            if !side.teams[i].war
                && rm >= 0
                && (side.region_stats[rm as usize].neutral_building_cnt > 0
                    || side.region_stats[rm as usize].enemy_building_cnt > 0)
                && side.teams[i].action_list.len() < 16
            {
                // Skip when another team already captures here.
                let mut taken = false;
                for u in 0..team_cnt {
                    if u == i || !side.teams[u].l_ok || side.teams[u].robot_cnt <= 0 {
                        continue;
                    }
                    if side.teams[u].action.ty == LogicActionType::Capture
                        && side.teams[u].action.region == rm
                    {
                        taken = true;
                        break;
                    }
                }
                if !taken {
                    side.teams[i].action_list.push(LogicAction {
                        ty: LogicActionType::Capture,
                        region: rm,
                        region_path: vec![rm],
                    });
                }
            }
        }

        // ── Pick the best action (3588-3630) ───────────────────────
        for i in 0..team_cnt {
            if side.teams[i].l_ok {
                continue;
            }
            Self::best_action(side, i);
            if !side.teams[i].action_list.is_empty() {
                side.teams[i].action = side.teams[i].action_list[0].clone();
                side.teams[i].action_time = now;
                side.teams[i].road_path.clear_fast();
                if side.teams[i].action.region_path.len() > 1 {
                    if let Some(rn_lock) = map.road_network.as_ref() {
                        let mut rn = rn_lock.lock().unwrap();
                        let path = side.teams[i].action.region_path.clone();
                        let mm = side.teams[i].move_mask;
                        rn.find_path_from_region_path(mm, &path, &mut side.teams[i].road_path);
                    }
                }
                // Robots snapshot the team route for ZonePathCalc
                // (MatrixRobot.cpp:1590-1591 reads it live in C++).
                self.distribute_team_road_path(side, i);
            } else {
                side.teams[i].action.ty = LogicActionType::None;
                side.teams[i].action_time = now;
            }
        }

        // ── Team reorganisations, one per takt (3632-3936) ─────────
        let mut changeok = false;

        // Retreating team pulls in reinforcements (3636-3714).
        for i in 0..team_cnt {
            if changeok {
                break;
            }
            let tm = &side.teams[i];
            if tm.l_ok
                || tm.robot_cnt <= 0
                || tm.action.ty != LogicActionType::Retreat
                || tm.region_far_danger < 0
            {
                continue;
            }
            // Candidate groups sorted by region-path distance.
            let mut cands: Vec<(usize, i32)> = Vec::new();
            for u in 0..MAX_LOGIC_GROUP {
                let lg = &side.logic_groups[u];
                if lg.robots_cnt <= 0
                    || lg.team == i as i32
                    || lg.war
                    || lg.action.ty == LogicActionType::Attack
                {
                    continue;
                }
                let Some(first) = self
                    .group_robots_of(sid, u)
                    .first()
                    .copied()
                else {
                    continue;
                };
                let rr = {
                    let r = robot_ref(&self.objects, first).unwrap();
                    get_region(map, r.map_x, r.map_y)
                };
                let gid = if rr != side.teams[i].action.region {
                    let mut path = [0i32; 8];
                    let cnt = {
                        let Some(rn_lock) = map.road_network.as_ref() else {
                            continue;
                        };
                        let mut rn = rn_lock.lock().unwrap();
                        rn.find_path_in_region_run(
                            side.teams[i].move_mask,
                            rr,
                            side.teams[i].action.region,
                            Some(&mut path),
                            8,
                            false,
                        ) as i32
                    };
                    if cnt <= 0 {
                        continue;
                    }
                    let mut blocked = false;
                    for (t, &pr) in path.iter().enumerate().take(cnt as usize) {
                        if side.region_stats[pr as usize].enemy_robot_cnt > 0 {
                            blocked = true;
                            break;
                        }
                        if t >= 4 && side.region_stats[pr as usize].danger > 0.0 {
                            blocked = true;
                            break;
                        }
                    }
                    if blocked {
                        continue;
                    }
                    cnt
                } else {
                    0
                };
                if gid >= 5 {
                    continue; // too far
                }
                cands.push((u, gid));
            }
            cands.sort_by_key(|&(_, d)| d);
            let fd_danger = side.region_stats[side.teams[i].region_far_danger as usize].danger;
            for (u, _) in cands {
                for rid in self.group_robots_of(sid, u) {
                    let strength = robot_ref(&self.objects, rid).unwrap().strength;
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.group_logic = -1;
                        r.set_team(i as i32);
                        r.group_road_path = None;
                    }
                    side.teams[i].strength += strength;
                }
                if side.teams[i].strength >= fd_danger {
                    break;
                }
            }
            changeok = true;
            side.last_team_change = now;
            break;
        }

        // Fighting team low on strength merges an idle neighbour team
        // (3717-3752).
        if !changeok && now - side.last_team_change > 3000 {
            'outer2: for i in 0..team_cnt {
                let tm = &side.teams[i];
                if tm.robot_cnt <= 0 || !tm.war || tm.region_mass < 0 || tm.region_far_danger < 0
                {
                    continue;
                }
                if side.region_stats[tm.region_far_danger as usize].danger
                    < tm.strength * 0.7
                {
                    continue;
                }
                for u in 0..team_cnt {
                    if i == u {
                        continue;
                    }
                    let tu = &side.teams[u];
                    if tu.robot_cnt <= 0
                        || tu.war
                        || tu.region_mass < 0
                        || tu.action.ty == LogicActionType::Capture
                    {
                        continue;
                    }
                    if side.teams[i].region_mass != tu.region_mass
                        && !g.is_near(side.teams[i].region_mass, tu.region_mass)
                    {
                        continue;
                    }
                    self.move_team_robots(sid, u as i32, i as i32);
                    changeok = true;
                    side.last_team_change = now;
                    break 'outer2;
                }
            }
        }

        // Merge two threatened idle teams (3755-3795).
        if !changeok && now - side.last_team_change > 3000 {
            'outer3: for i in 0..team_cnt {
                let tm = &side.teams[i];
                if tm.robot_cnt <= 0
                    || tm.war
                    || tm.region_far_danger < 0
                    || tm.region_mass < 0
                    || tm.action.ty == LogicActionType::Capture
                {
                    continue;
                }
                if side.region_stats[tm.region_far_danger as usize].danger < tm.strength {
                    continue;
                }
                for u in 0..team_cnt {
                    if u == i {
                        continue;
                    }
                    let tu = &side.teams[u];
                    if tu.robot_cnt <= 0
                        || tu.war
                        || tu.region_far_danger < 0
                        || tu.region_mass < 0
                        || tu.action.ty == LogicActionType::Capture
                    {
                        continue;
                    }
                    if side.region_stats[tu.region_far_danger as usize].danger < tu.strength {
                        continue;
                    }
                    if side.teams[i].region_mass != tu.region_mass
                        && !g.is_near(side.teams[i].region_mass, tu.region_mass)
                    {
                        continue;
                    }
                    self.move_team_robots(sid, u as i32, i as i32);
                    side.teams[i].action.ty = LogicActionType::None;
                    side.teams[i].action_time = now;
                    changeok = true;
                    side.last_team_change = now;
                    break 'outer3;
                }
            }
        }

        // Refill an empty team from the biggest safe team (3798-3855).
        if !changeok && now - side.last_team_change > 3000 {
            for i in 0..team_cnt {
                if side.teams[i].robot_cnt > 0 {
                    continue;
                }
                let mut k: i32 = -1;
                for u in 0..team_cnt {
                    let tu = &side.teams[u];
                    if tu.robot_cnt < 2 {
                        continue;
                    }
                    if !matches!(
                        tu.action.ty,
                        LogicActionType::Forward | LogicActionType::Capture
                    ) {
                        continue;
                    }
                    if tu.region_far_danger >= 0 || tu.regroup_only_after_war {
                        continue;
                    }
                    if k < 0 || tu.robot_cnt > side.teams[k as usize].robot_cnt {
                        k = u as i32;
                    }
                }
                if k < 0 {
                    break;
                }
                side.clear_team(i);
                if side.teams[k as usize].group_cnt == 1 {
                    // Split by robots.
                    let mut u = side.teams[k as usize].robot_cnt / 2;
                    for rid in self.side_robots_of(sid) {
                        if u == 0 {
                            break;
                        }
                        let team = robot_ref(&self.objects, rid).unwrap().team;
                        if team != k {
                            continue;
                        }
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.set_team(i as i32);
                            r.group_logic = -1;
                            r.group_road_path = None;
                        }
                        u -= 1;
                    }
                } else {
                    // Split by groups.
                    let mut u = side.teams[k as usize].group_cnt / 2;
                    for t in 0..MAX_LOGIC_GROUP {
                        if u == 0 {
                            break;
                        }
                        if side.logic_groups[t].robots_cnt <= 0
                            || side.logic_groups[t].team != k
                        {
                            continue;
                        }
                        for rid in self.group_robots_of(sid, t) {
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                r.set_team(i as i32);
                                r.group_logic = -1;
                                r.group_road_path = None;
                            }
                        }
                        u -= 1;
                    }
                }
                changeok = true;
                side.last_team_change = now;
                break;
            }
        }

        // Balance adjacent teams with uneven robot counts (3858-3902).
        if !changeok && now - side.last_team_change > 3000 {
            for i in 0..team_cnt {
                let tm = &side.teams[i];
                if tm.robot_cnt <= 0
                    || !matches!(
                        tm.action.ty,
                        LogicActionType::Forward | LogicActionType::Capture
                    )
                    || tm.region_mass < 0
                    || tm.region_far_danger >= 0
                    || tm.regroup_only_after_war
                {
                    continue;
                }
                let mut k: i32 = -1;
                for u in 0..team_cnt {
                    if i == u {
                        continue;
                    }
                    let tu = &side.teams[u];
                    if tu.robot_cnt <= 0
                        || tu.region_mass < 0
                        || !matches!(
                            tu.action.ty,
                            LogicActionType::Forward | LogicActionType::Capture
                        )
                        || tu.region_far_danger >= 0
                        || tu.regroup_only_after_war
                    {
                        continue;
                    }
                    if tu.region_mass != tm.region_mass
                        && !g.is_near(tu.region_mass, tm.region_mass)
                    {
                        continue;
                    }
                    if k < 0 || tu.robot_cnt > side.teams[k as usize].robot_cnt {
                        k = u as i32;
                    }
                }
                if k < 0 {
                    continue;
                }
                // C++ reads m_Team[u] (out of bounds) — intended k.
                let mut u = side.teams[k as usize].robot_cnt - side.teams[i].robot_cnt;
                if u < 2 {
                    continue;
                }
                u /= 2;
                for rid in self.side_robots_of(sid) {
                    if u == 0 {
                        break;
                    }
                    let team = robot_ref(&self.objects, rid).unwrap().team;
                    if team != k {
                        continue;
                    }
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.set_team(i as i32);
                        r.group_logic = -1;
                        r.group_road_path = None;
                    }
                    u -= 1;
                }
                side.teams[i].action.ty = LogicActionType::None;
                side.teams[i].action_time = now;
                changeok = true;
                side.last_team_change = now;
                break;
            }
        }

        // Merge adjacent defending teams (3905-3936).
        if !changeok && now - side.last_team_change > 3000 {
            for i in 0..team_cnt {
                let tm = &side.teams[i];
                if tm.robot_cnt <= 0
                    || tm.wait_union
                    || tm.action.ty != LogicActionType::Defence
                    || tm.region_mass < 0
                {
                    continue;
                }
                for u in 0..team_cnt {
                    if i == u {
                        continue;
                    }
                    let tu = &side.teams[u];
                    if tu.robot_cnt <= 0
                        || tu.wait_union
                        || tu.action.ty != LogicActionType::Defence
                        || tu.region_mass < 0
                        || tu.regroup_only_after_war
                    {
                        continue;
                    }
                    let ok = side.teams[i].action.region == tu.action.region
                        || g.is_near(side.teams[i].action.region, tu.action.region)
                        || self.can_move_no_enemy(
                            &g,
                            side,
                            side.teams[i].move_mask | side.teams[u].move_mask,
                            side.teams[i].action.region,
                            side.teams[u].action.region,
                        );
                    if !ok {
                        continue;
                    }
                    side.teams[u].robot_cnt = 0;
                    side.teams[i].regroup_only_after_war = true;
                    self.move_team_robots(sid, u as i32, i as i32);
                }
            }
        }
    }

    /// Move every robot of `from` team to `to` (the repeated
    /// SetTeam/SetGroupLogic(-1) walk in the reorganisation blocks).
    fn move_team_robots(&mut self, sid: i32, from: i32, to: i32) {
        for rid in self.side_robots_of(sid) {
            let team = robot_ref(&self.objects, rid).unwrap().team;
            if team != from {
                continue;
            }
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.set_team(to);
                r.group_logic = -1;
                r.group_road_path = None;
            }
        }
    }

    /// Give the team's robots a snapshot of the team road route so
    /// `ZonePathCalc` can constrain pathing to it.
    fn distribute_team_road_path(&mut self, side: &Side, team: usize) {
        let arc = if side.teams[team].road_path.list_cnt > 0 {
            Some(std::sync::Arc::new(side.teams[team].road_path.clone()))
        } else {
            None
        };
        for rid in self.side_robots_of(side.id) {
            let t = robot_ref(&self.objects, rid).unwrap().team;
            if t != team as i32 {
                continue;
            }
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.group_road_path = arc.clone();
            }
        }
    }

    /// `FindNearRegionWithUTR` (MatrixSide.cpp:4016-4068). Flags:
    /// 1-our 2-neutral 4-enemy 8-base 16-building 32-robot 64-cannon.
    fn find_near_region_with_utr(
        g: &RegionGraph,
        side: &mut Side,
        from: i32,
        exclude: &[i32],
        flags: u32,
    ) -> i32 {
        if from < 0 || from as usize >= g.cnt {
            return -1;
        }
        for r in side.region_stats.iter_mut() {
            r.data = 0;
        }
        let mut level = 1u32;
        side.region_stats[from as usize].data = level;
        let mut queue: Vec<i32> = vec![from];
        let mut sme = 0usize;
        let mut next = queue.len();
        level += 1;
        while sme < queue.len() {
            for &(u, _) in &g.near[queue[sme] as usize] {
                let uu = u as usize;
                if side.region_stats[uu].data > 0 {
                    continue;
                }
                if !exclude.contains(&u) {
                    let st = &side.region_stats[uu];
                    if flags & 1 != 0 {
                        if flags & 8 != 0 && st.our_base_cnt != 0 {
                            return u;
                        } else if flags & 16 != 0 && st.our_building_cnt != 0 {
                            return u;
                        } else if flags & 32 != 0 && st.our_robot_cnt != 0 {
                            return u;
                        } else if flags & 64 != 0 && st.our_cannon_cnt != 0 {
                            return u;
                        }
                    }
                    if flags & 2 != 0 {
                        if flags & 8 != 0 && st.neutral_base_cnt != 0 {
                            return u;
                        } else if flags & 16 != 0 && st.neutral_building_cnt != 0 {
                            return u;
                        } else if flags & 64 != 0 && st.neutral_cannon_cnt != 0 {
                            return u;
                        }
                    }
                    if flags & 4 != 0 {
                        if flags & 8 != 0 && st.enemy_base_cnt != 0 {
                            return u;
                        } else if flags & 16 != 0 && st.enemy_building_cnt != 0 {
                            return u;
                        } else if flags & 32 != 0 && st.enemy_robot_cnt != 0 {
                            return u;
                        } else if flags & 64 != 0 && st.enemy_cannon_cnt != 0 {
                            return u;
                        }
                    }
                }
                queue.push(u);
                side.region_stats[uu].data = level;
            }
            sme += 1;
            if sme >= next {
                next = queue.len();
                level += 1;
            }
        }
        -1
    }

    /// `CompareAction` (MatrixSide.cpp:4070-4194): negative when `a2`
    /// is better than `a1`.
    fn compare_action(side: &Side, team: usize, a1: &LogicAction, a2: &LogicAction) -> i32 {
        let (a1, a2, scale) = if a1.ty > a2.ty {
            (a2, a1, -1)
        } else {
            (a1, a2, 1)
        };
        let tm = &side.teams[team];
        let danger = |r: i32| -> f32 {
            if r >= 0 {
                side.region_stats[r as usize].danger
            } else {
                0.0
            }
        };
        use LogicActionType as T;
        let v = match (a1.ty, a2.ty) {
            (T::Defence, T::Defence) => {
                if danger(a1.region) != danger(a2.region) {
                    if danger(a1.region) < danger(a2.region) {
                        1
                    } else {
                        -1
                    }
                } else {
                    0
                }
            }
            (T::Forward, T::Forward) => {
                if a1.region_path.len() != a2.region_path.len() {
                    if a1.region_path.len() > a2.region_path.len() {
                        -1
                    } else {
                        1
                    }
                } else {
                    let e1 = side.region_stats[a1.region as usize].enemy_building_cnt;
                    let e2 = side.region_stats[a2.region as usize].enemy_building_cnt;
                    if e1 != e2 {
                        if e1 < e2 {
                            -1
                        } else {
                            1
                        }
                    } else {
                        let n1 = side.region_stats[a1.region as usize].neutral_building_cnt;
                        let n2 = side.region_stats[a2.region as usize].neutral_building_cnt;
                        if n1 != n2 {
                            if n1 < n2 {
                                -1
                            } else {
                                1
                            }
                        } else {
                            0
                        }
                    }
                }
            }
            (T::Attack, T::Attack)
            | (T::Retreat, T::Retreat)
            | (T::Capture, T::Capture)
            | (T::Intercept, T::Intercept) => 0,
            (T::Defence, T::Attack) => {
                if !tm.war && tm.wait_union {
                    1
                } else if tm.region_near_danger >= 0
                    && danger(tm.region_near_danger) > tm.strength
                {
                    1
                } else {
                    -1
                }
            }
            (T::Defence, T::Forward) => {
                if !tm.war && tm.wait_union {
                    1
                } else if tm.war {
                    -1
                } else if danger(a2.region) > 0.0 {
                    if tm.region_near_danger >= 0 && danger(tm.region_near_danger) > tm.strength
                    {
                        -1
                    } else {
                        1
                    }
                } else if tm.region_near_danger < 0
                    || danger(tm.region_near_danger) * 0.5 > tm.strength
                {
                    1
                } else {
                    -1
                }
            }
            (T::Defence, T::Retreat) => {
                if tm.war {
                    -1
                } else if tm.region_near_danger >= 0
                    && danger(tm.region_near_danger) * 0.5 > tm.strength
                {
                    1
                } else {
                    -1
                }
            }
            (T::Defence, T::Capture) => {
                if !tm.war && tm.wait_union {
                    1
                } else if tm.war {
                    -1
                } else if danger(a2.region) > 0.0 {
                    if tm.region_near_danger >= 0 && danger(tm.region_near_danger) > tm.strength
                    {
                        1
                    } else {
                        -1
                    }
                } else if tm.region_near_danger >= 0
                    && danger(tm.region_near_danger) * 0.5 > tm.strength
                {
                    1
                } else {
                    -1
                }
            }
            (T::Defence, T::Intercept) => -1,
            (T::Attack, T::Forward) => 1,
            (T::Attack, T::Retreat) | (T::Forward, T::Retreat) => {
                if tm.region_near_danger >= 0
                    && tm.strength >= danger(tm.region_near_danger) * 0.8
                {
                    1
                } else {
                    -1
                }
            }
            (T::Attack, T::Capture) => 1,
            (T::Attack, T::Intercept) => 1,
            (T::Forward, T::Capture) => -1,
            (T::Forward, T::Intercept) => 1,
            (T::Retreat, T::Capture) => 1,
            (T::Retreat, T::Intercept) => 1,
            (T::Capture, T::Intercept) => 1,
            _ => 0,
        };
        v * scale
    }

    /// `BestAction` (MatrixSide.cpp:4196-4208) — reduce the variant
    /// list to the single best entry.
    fn best_action(side: &mut Side, team: usize) {
        if side.teams[team].action_list.len() <= 1 {
            return;
        }
        let list = side.teams[team].action_list.clone();
        let mut k = 0usize;
        for u in 1..list.len() {
            if Self::compare_action(side, team, &list[k], &list[u]) < 0 {
                k = u;
            }
        }
        let best = list[k].clone();
        side.teams[team].action_list.clear();
        side.teams[team].action_list.push(best);
    }

    /// `CalcRegionPath` (MatrixSide.cpp:9550-9586) — extract the region
    /// path from the wave levels stored in `region_stats[].data`.
    fn calc_region_path(
        g: &RegionGraph,
        side: &mut Side,
        ac: &mut LogicAction,
        rend_in: i32,
        mm: u8,
    ) {
        let mut rend = rend_in;
        let mut rev: Vec<i32> = vec![rend];
        let mut level = side.region_stats[rend as usize].data;
        loop {
            let mut i = -1i32;
            for &(u, mv) in &g.near[rend as usize] {
                let d = side.region_stats[u as usize].data;
                if d == 0 || d >= level {
                    continue;
                }
                if mv & mm != 0 {
                    continue;
                }
                i = u;
                break;
            }
            if i < 0 {
                log::warn!("calc_region_path: dead end (C++ ERROR_E)");
                break;
            }
            if rev.len() >= REGION_PATH_MAX_CNT {
                break;
            }
            rev.push(i);
            rend = i;
            level = side.region_stats[rend as usize].data;
            if level <= 1 {
                break;
            }
        }
        rev.reverse();
        ac.region_path = rev;
    }

    /// `CanMoveNoEnemy` (MatrixSide.cpp:9588-9664) — an enemy-free
    /// route from r1 to r2 exists and is at most 1.3× the direct one.
    fn can_move_no_enemy(&self, g: &RegionGraph, side: &mut Side, mm: u8, r1: i32, r2: i32) -> bool {
        if r1 < 0 || r2 < 0 {
            return false;
        }
        let bfs = |side: &mut Side, skip_enemy: bool| -> Option<i32> {
            for r in side.region_stats.iter_mut() {
                r.data = 0;
            }
            let mut dist = 1i32;
            let mut queue: Vec<i32> = vec![r1];
            side.region_stats[r1 as usize].data = dist as u32;
            dist += 1;
            let mut sme = 0usize;
            let mut next = queue.len();
            while sme < queue.len() {
                for &(u, mv) in &g.near[queue[sme] as usize] {
                    if side.region_stats[u as usize].data != 0 {
                        continue;
                    }
                    if mv & mm != 0 {
                        continue;
                    }
                    if u == r2 {
                        return Some(dist);
                    }
                    if skip_enemy
                        && (side.region_stats[u as usize].enemy_robot_cnt > 0
                            || side.region_stats[u as usize].enemy_cannon_cnt > 0)
                    {
                        continue;
                    }
                    queue.push(u);
                    side.region_stats[u as usize].data = dist as u32;
                }
                sme += 1;
                if sme >= next {
                    next = queue.len();
                    dist += 1;
                }
            }
            None
        };
        let Some(dist) = bfs(side, false) else {
            return false;
        };
        let Some(dist2) = bfs(side, true) else {
            return false;
        };
        dist2 <= float2int(1.3 * dist as f32)
    }

    // ─────────────────────────────────────────────────────────────────
    // TaktTL
    // ─────────────────────────────────────────────────────────────────

    /// `TaktTL` (MatrixSide.cpp:4218-4631).
    fn takt_tl(&mut self, map: &GameMap, side: &mut Side) {
        let now = self.elapsed_ms as i32;
        let sid = side.id;
        if side.last_takt_tl != 0 && now - side.last_takt_tl < 10 {
            return;
        }
        side.last_takt_tl = now;

        // Underfire refresh, 500ms (4234-4287).
        if side.last_takt_underfire == 0 || now - side.last_takt_underfire > 500 {
            side.last_takt_underfire = now;
            self.underfire_calc(map, sid);
        }

        // No-break release + place desync clear (4290-4304).
        for rid in self.side_robots_of(sid) {
            let (no_break, can_break, place, capturing, to_place) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (
                    r.env.order_no_break,
                    r.can_break_order(),
                    r.env.place,
                    r.get_capture_factory().is_some(),
                    is_to_place(map, r, r.env.place),
                )
            };
            if no_break && can_break {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.env.order_no_break = false;
                    r.env.place = -1;
                }
            } else if place >= 0 && can_break && !capturing && !to_place {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.env.place = -1;
                }
            }
        }

        for gi in 0..MAX_LOGIC_GROUP {
            if side.logic_groups[gi].robots_cnt <= 0 {
                continue;
            }
            if side.logic_groups[gi].war {
                self.war_tl(map, side, gi);
            } else {
                self.repair_tl(map, side, gi);
            }
        }

        for gi in 0..MAX_LOGIC_GROUP {
            if side.logic_groups[gi].robots_cnt <= 0 {
                continue;
            }
            let grp = self.group_robots_of(sid, gi);
            if grp.is_empty() {
                continue;
            }
            let team = robot_ref(&self.objects, grp[0]).unwrap().team;
            let team_ok = team >= 0 && (team as usize) < side.teams.len();
            side.logic_groups[gi].robots_cnt = grp.len() as i32;

            let cmp_order = |side: &Side, team: i32, gi: usize| -> bool {
                if team < 0 {
                    return true;
                }
                let lg = &side.logic_groups[gi];
                let tm = &side.teams[team as usize];
                lg.action.ty == tm.action.ty && lg.action.region == tm.action.region
            };

            // ── Order validity (4338-4472) ─────────────────────────
            let mut orderok = true;
            loop {
                match side.logic_groups[gi].action.ty {
                    LogicActionType::None => {
                        if team_ok {
                            side.logic_groups[gi].action =
                                side.teams[team as usize].action.clone();
                            if side.logic_groups[gi].action.ty != LogicActionType::None {
                                orderok = false;
                                continue;
                            }
                        }
                    }
                    LogicActionType::Capture => {
                        if !cmp_order(side, team, gi) {
                            side.logic_groups[gi].action =
                                side.teams[team as usize].action.clone();
                            orderok = false;
                            continue;
                        }
                        let ar = side.logic_groups[gi].action.region;
                        if ar < 0
                            || (side.region_stats[ar as usize].neutral_building_cnt <= 0
                                && side.region_stats[ar as usize].enemy_building_cnt <= 0)
                        {
                            // Nothing to capture (4347-4352).
                            if ar
                                != side
                                    .teams
                                    .get(team.max(0) as usize)
                                    .map(|t| t.region_mass)
                                    .unwrap_or(-1)
                            {
                                side.logic_groups[gi].action.ty = LogicActionType::Forward;
                            } else {
                                side.logic_groups[gi].action.ty = LogicActionType::None;
                            }
                            orderok = false;
                            break;
                        }
                        // Any robot of the whole team capturing?
                        let mut team_capturing = false;
                        for rid in self.side_robots_of(sid) {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            if r.team == team && r.orders.has(OrderType::CaptureFactory) {
                                team_capturing = true;
                                break;
                            }
                        }
                        orderok = team_capturing;
                        if orderok {
                            for &rid in &grp {
                                let r = robot_ref(&self.objects, rid).unwrap();
                                if r.orders.has(OrderType::CaptureFactory) {
                                    continue;
                                }
                                if !self.place_in_region(map, rid, r.env.place, ar)
                                    && can_change_place(now, r)
                                {
                                    orderok = false;
                                    break;
                                }
                            }
                        } else {
                            // All robots busy → success (4377-4378).
                            let mut all_busy = true;
                            for &rid in &grp {
                                if robot_ref(&self.objects, rid).unwrap().can_break_order() {
                                    all_busy = false;
                                    break;
                                }
                            }
                            if all_busy {
                                orderok = true;
                            }
                            if !orderok {
                                let mut all_in_region = true;
                                for &rid in &grp {
                                    let place =
                                        robot_ref(&self.objects, rid).unwrap().env.place;
                                    if !self.place_in_region(map, rid, place, ar) {
                                        all_in_region = false;
                                        break;
                                    }
                                }
                                if all_in_region {
                                    for rid in self.side_robots_of(sid) {
                                        let cf = robot_ref(&self.objects, rid)
                                            .unwrap()
                                            .get_capture_factory();
                                        if let Some(cf) = cf {
                                            let rr = get_map_pos(&self.objects, cf)
                                                .map(|p| get_region(map, p.0, p.1))
                                                .unwrap_or(-1);
                                            if rr == ar {
                                                orderok = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    LogicActionType::Defence => {
                        if !side.logic_groups[gi].war {
                            let mut any_enemy = false;
                            for &rid in &grp {
                                if robot_ref(&self.objects, rid).unwrap().env.enemy_cnt() > 0 {
                                    any_enemy = true;
                                    break;
                                }
                            }
                            if any_enemy {
                                orderok = false;
                            }
                            if orderok {
                                if team_ok && !cmp_order(side, team, gi) {
                                    side.logic_groups[gi].action =
                                        side.teams[team as usize].action.clone();
                                    orderok = false;
                                    continue;
                                }
                                let ar = side.logic_groups[gi].action.region;
                                for &rid in &grp {
                                    let r = robot_ref(&self.objects, rid).unwrap();
                                    if can_change_place(now, r)
                                        && des_region(map, r.env.place) != ar
                                    {
                                        orderok = false;
                                        break;
                                    }
                                }
                            }
                        } else {
                            let mut any_enemy = false;
                            for &rid in &grp {
                                if robot_ref(&self.objects, rid).unwrap().env.enemy_cnt() > 0 {
                                    any_enemy = true;
                                    break;
                                }
                            }
                            if !any_enemy {
                                orderok = false;
                            }
                        }
                    }
                    LogicActionType::Attack => {
                        let mut u = 0;
                        if !side.logic_groups[gi].war {
                            orderok = false;
                        } else {
                            let ar = side.logic_groups[gi].action.region;
                            for &rid in &grp {
                                if !orderok {
                                    break;
                                }
                                let r = robot_ref(&self.objects, rid).unwrap();
                                if r.env.enemy_cnt() > 0 {
                                    u += 1;
                                } else if can_change_place(now, r)
                                    && des_region(map, r.env.place) != ar
                                {
                                    orderok = false;
                                    break;
                                }
                            }
                        }
                        if u == 0 && orderok {
                            // Team-mates fighting nearby → rush to help
                            // (4436-4449).
                            for rid in self.side_robots_of(sid) {
                                let r = robot_ref(&self.objects, rid).unwrap();
                                if r.team != team || r.group_logic == gi as i32 {
                                    continue;
                                }
                                if let Some(t) = r.env.target {
                                    let rr = get_map_pos(&self.objects, t)
                                        .map(|p| get_region(map, p.0, p.1))
                                        .unwrap_or(-1);
                                    side.logic_groups[gi].action.region = rr;
                                    u = grp.len();
                                    orderok = false;
                                    break;
                                }
                            }
                        }
                        if u == 0
                            && team_ok
                            && side.teams[team as usize].action.ty != LogicActionType::Attack
                            && !cmp_order(side, team, gi)
                        {
                            side.logic_groups[gi].action =
                                side.teams[team as usize].action.clone();
                            orderok = false;
                            continue;
                        }
                    }
                    LogicActionType::Forward | LogicActionType::Retreat => {
                        if team_ok && !cmp_order(side, team, gi) {
                            side.logic_groups[gi].action =
                                side.teams[team as usize].action.clone();
                            orderok = false;
                            continue;
                        }
                        let ar = side.logic_groups[gi].action.region;
                        for &rid in &grp {
                            if !orderok {
                                break;
                            }
                            let r = robot_ref(&self.objects, rid).unwrap();
                            if can_change_place(now, r) && des_region(map, r.env.place) != ar {
                                orderok = false;
                                break;
                            }
                        }
                    }
                    LogicActionType::Intercept => {}
                }
                break;
            }
            if orderok {
                continue;
            }

            // Attack freshly assigned → old places invalid (4475-4479).
            if side.logic_groups[gi].action.ty == LogicActionType::Attack
                && !side.logic_groups[gi].war
            {
                for &rid in &grp {
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.env.place = -1;
                    }
                }
            }
            side.logic_groups[gi].war = false;

            // ── Execute the new order (4483-4585) ──────────────────
            match side.logic_groups[gi].action.ty {
                LogicActionType::Defence => {
                    let mut any_enemy = false;
                    for &rid in &grp {
                        if robot_ref(&self.objects, rid).unwrap().env.enemy_cnt() > 0 {
                            any_enemy = true;
                            break;
                        }
                    }
                    if any_enemy {
                        side.logic_groups[gi].war = true;
                    } else {
                        let region = side.logic_groups[gi].action.region;
                        self.assign_place_group(map, side, gi, region);
                        self.ai_move_group_to_places(map, &grp);
                    }
                }
                LogicActionType::Attack => {
                    side.logic_groups[gi].war = true;
                }
                LogicActionType::Forward | LogicActionType::Retreat => {
                    let region = side.logic_groups[gi].action.region;
                    self.assign_place_group(map, side, gi, region);
                    self.ai_move_group_to_places(map, &grp);
                }
                LogicActionType::Capture => {
                    let region = side.logic_groups[gi].action.region;
                    self.assign_place_group(map, side, gi, region);

                    // Distribute capture targets (4540-4575).
                    let mut targets: Vec<ObjectId> = Vec::new();
                    for id in self.objects.iter_units() {
                        let Some(b) = building_ref(&self.objects, id) else {
                            continue;
                        };
                        if !b.is_live() || b.side == sid || !b.can_be_captured() {
                            continue;
                        }
                        let r = get_map_pos(&self.objects, id)
                            .map(|p| get_region(map, p.0, p.1))
                            .unwrap_or(-1);
                        if r == region {
                            targets.push(id);
                        }
                    }
                    for tid in targets {
                        // Skip if anyone on our side already captures it.
                        let mut claimed = false;
                        for rid in self.side_robots_of(sid) {
                            if robot_ref(&self.objects, rid)
                                .unwrap()
                                .orders
                                .find_order(OrderType::CaptureFactory, tid)
                            {
                                claimed = true;
                                break;
                            }
                        }
                        if claimed {
                            continue;
                        }
                        let twp = get_world_pos(&self.objects, tid).unwrap_or_default();
                        let mut best: Option<(f32, ObjectId)> = None;
                        for &rid in &grp {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            if r.orders.has(OrderType::CaptureFactory) {
                                continue;
                            }
                            let Some(wp) = get_world_pos(&self.objects, rid) else {
                                continue;
                            };
                            let d = (wp - twp).length_squared();
                            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                                best = Some((d, rid));
                            }
                        }
                        if let Some((_, rid)) = best {
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                if prepare_break_order(r) {
                                    r.capture_factory(tid);
                                }
                            }
                        }
                    }
                    // Everyone else takes their place (4577-4584).
                    for &rid in &grp {
                        let (has_cf, place) = {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            (r.orders.has(OrderType::CaptureFactory), r.env.place)
                        };
                        if has_cf {
                            continue;
                        }
                        if let Some((_, _, pos, _)) = place_get(map, place) {
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                if prepare_break_order(r) {
                                    r.move_to_high(pos.0, pos.1);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// The repeated `PrepareBreakOrder + MoveToHigh(place)` loop.
    fn ai_move_group_to_places(&mut self, map: &GameMap, grp: &[ObjectId]) {
        for &rid in grp {
            let place = robot_ref(&self.objects, rid).unwrap().env.place;
            let Some((_, _, pos, _)) = place_get(map, place) else {
                continue;
            };
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                if prepare_break_order(r) {
                    r.move_to_high(pos.0, pos.1);
                }
            }
        }
    }

    /// `WarTL` (MatrixSide.cpp:4633-5215).
    fn war_tl(&mut self, map: &GameMap, side: &mut Side, group: usize) {
        let now = self.elapsed_ms as i32;
        let sid = side.id;
        let grp = self.group_robots_of(sid, group);
        if grp.is_empty() {
            return;
        }
        let mut mm = 0u8;
        for &rid in &grp {
            mm |= chassis_bit(robot_ref(&self.objects, rid).unwrap());
        }
        let gsm = GameMap::GLOBAL_SCALE_MOVE;

        // ── Target selection (4656-4723) ───────────────────────────
        for &rid in &grp {
            {
                let r = robot_mut(&mut self.objects, rid).unwrap();
                if r.env.target_attack == Some(rid) {
                    r.env.target_attack = None;
                }
            }
            let need = robot_ref(&self.objects, rid)
                .unwrap()
                .env
                .target_attack
                .is_none();
            if need {
                if let Some(found) = self.pick_enemy_for(map, sid, rid, None) {
                    let is_cannon = is_live_unit(&self.objects, found)
                        && cannon_ref(&self.objects, found).is_some();
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.env.target_attack = Some(found);
                        // New target is a cannon → reposition (4718-4720).
                        if is_cannon {
                            r.env.place = -1;
                        }
                    }
                }
            }
        }

        // ── Movement validation (4726-4780) ────────────────────────
        let region = side.logic_groups[group].action.region;
        let mut rlokmove = vec![true; grp.len()];
        let mut moveok = true;
        for (i, &rid) in grp.iter().enumerate() {
            let (can_break, ta, place) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (r.can_break_order(), r.env.target_attack, r.env.place)
            };
            if !can_break {
                continue;
            }
            if ta.is_none() {
                // No target → head for the destination region (4731-4744).
                if des_region(map, place) != region {
                    let in_region = self.place_in_region(map, rid, place, region);
                    let can_change = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        can_change_place(now, r)
                    };
                    if !in_region && can_change {
                        self.assign_place_robot(map, rid, region);
                        let place = robot_ref(&self.objects, rid).unwrap().env.place;
                        if let Some((_, _, pos, _)) = place_get(map, place) {
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                if prepare_break_order(r) {
                                    r.move_to_high(pos.0, pos.1);
                                }
                            }
                        }
                    }
                }
                continue;
            }
            let ta = ta.unwrap();
            if place < 0 {
                let r = robot_ref(&self.objects, rid).unwrap();
                if can_change_place(now, r) {
                    rlokmove[i] = false;
                    moveok = false;
                }
                continue;
            }
            let Some(tv) = get_world_pos(&self.objects, ta) else {
                continue;
            };
            match place_get(map, place) {
                None => {
                    let can = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        can_change_place(now, r)
                    };
                    if can {
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.env.place = -1;
                        }
                        rlokmove[i] = false;
                        moveok = false;
                    }
                }
                Some((_, _, pos, _)) => {
                    let half = gsm * ROBOT_MOVECELLS_PER_SIZE as f32 / 2.0;
                    let pc = glam::Vec2::new(gsm * pos.0 as f32 + half, gsm * pos.1 as f32 + half);
                    let r = robot_ref(&self.objects, rid).unwrap();
                    let reach = r.max_fire_dist - half;
                    if (pc - tv).length_squared() > reach * reach && can_change_place(now, r) {
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.env.place = -1;
                        }
                        rlokmove[i] = false;
                        moveok = false;
                    }
                }
            }
        }

        // ── Re-place lost robots (4782-5036) ───────────────────────
        if !moveok {
            let mut tp_sum = (0i64, 0i64);
            let mut f = 0i64;
            for &rid in &grp {
                let ta = robot_ref(&self.objects, rid).unwrap().env.target_attack;
                let Some(ta) = ta else { continue };
                let Some(tp) = get_map_pos(&self.objects, ta) else {
                    continue;
                };
                tp_sum.0 += tp.0 as i64;
                tp_sum.1 += tp.1 as i64;
                f += 1;
            }
            if f <= 0 {
                return;
            }
            let tp_avg = ((tp_sum.0 / f) as i32, (tp_sum.1 / f) as i32);
            let mut center = tp_avg;
            let mut best = i64::MAX;
            for &rid in &grp {
                let ta = robot_ref(&self.objects, rid).unwrap().env.target_attack;
                let Some(ta) = ta else { continue };
                let Some(tp2) = get_map_pos(&self.objects, ta) else {
                    continue;
                };
                let f2 = ((tp_avg.0 - tp2.0) as i64).pow(2) + ((tp_avg.1 - tp2.1) as i64).pow(2);
                if f2 < best {
                    best = f2;
                    center = tp2;
                }
            }
            let mut radius = 0i32;
            let mut radiusrobot = 0i32;
            for &rid in &grp {
                let r = robot_ref(&self.objects, rid).unwrap();
                let Some(ta) = r.env.target_attack else {
                    continue;
                };
                let Some(tp2) = get_map_pos(&self.objects, ta) else {
                    continue;
                };
                radiusrobot = radiusrobot.max(float2int(r.max_fire_dist / gsm));
                let d = (((center.0 - tp2.0) as f32).powi(2)
                    + ((center.1 - tp2.1) as f32).powi(2))
                .sqrt();
                radius = radius
                    .max(float2int(d + r.max_fire_dist / gsm + ROBOT_MOVECELLS_PER_SIZE as f32));
            }

            let r0_pos = {
                let r0 = robot_ref(&self.objects, grp[0]).unwrap();
                (r0.map_x, r0.map_y)
            };
            let mut cplr = true;
            let mut list: Vec<i32> = Vec::new();
            let (st, _) = place_list(map, mm, r0_pos, center, radius, false, &mut list);
            if st == 0 {
                for &rid in &grp {
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.env.place_not_found = now;
                    }
                }
            } else {
                zero_places(map, &list);
                mark_occupied_places(map, &self.objects);

                let mut i = 0usize;
                while i < grp.len() {
                    let rid = grp[i];
                    if rlokmove[i] {
                        i += 1;
                        continue;
                    }
                    let (have_bomb, ta) = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        (r.have_bomb(&self.objects), r.env.target_attack)
                    };
                    let Some(ta) = ta else {
                        i += 1;
                        continue;
                    };

                    // Target geometry (4890-4899).
                    let ta_type = self.objects.get(ta).map(|o| o.core().obj_type);
                    let (tvx, tvy, enemy_fire_dist) = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        match ta_type {
                            Some(ObjectType::RobotAi) => {
                                let t = robot_ref(&self.objects, ta).unwrap();
                                (
                                    t.pos_x - r.pos_x,
                                    t.pos_y - r.pos_y,
                                    float2int(t.max_fire_dist),
                                )
                            }
                            Some(ObjectType::Cannon) => {
                                let c = cannon_ref(&self.objects, ta).unwrap();
                                (
                                    c.pos.x - r.pos_x,
                                    c.pos.y - r.pos_y,
                                    (c.fire_radius(&self.objects) + gsm) as i32,
                                )
                            }
                            _ => {
                                i += 1;
                                continue;
                            }
                        }
                    };
                    let tsize2 = tvx * tvx + tvy * tvy;
                    let tsize2o = if tsize2 != 0.0 { 1.0 / tsize2 } else { 0.0 };

                    let (r_pos, r_maxfd, r_minfd, r_cbit) = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        (
                            glam::Vec2::new(r.pos_x, r.pos_y),
                            r.max_fire_dist,
                            r.min_fire_dist,
                            chassis_bit(r),
                        )
                    };
                    let Some(des) = crate::matrix_game::logic::point_of_aim(map, &self.objects, ta)
                    else {
                        i += 1;
                        continue;
                    };

                    let mut placebest: i32 = -1;
                    let mut s_f1 = 0.0f32;
                    let mut s_underfire = 0i32;
                    let mut s_close = false;

                    for &iplace in &list {
                        let Some((data, mv, pos, underfire0)) = place_get(map, iplace) else {
                            continue;
                        };
                        if data != 0 || mv & r_cbit != 0 {
                            continue;
                        }
                        {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            if r.env.is_bad_place(iplace) {
                                continue;
                            }
                        }
                        let half = gsm * 4.0 / 2.0;
                        let pcx = gsm * pos.0 as f32 + half;
                        let pcy = gsm * pos.1 as f32 + half;
                        let pvx = pcx - r_pos.x;
                        let pvy = pcy - r_pos.y;
                        let k = (pvx * tvx + pvy * tvy) * tsize2o;
                        // (4922-4930): cannons → no rear cutoff; robots →
                        // k>0.95; (buildings can't be env targets here).
                        if !have_bomb && !matches!(ta_type, Some(ObjectType::Cannon)) && k > 0.95
                        {
                            continue;
                        }
                        let m = (-pvx * tvy + pvy * tvx) * tsize2o;
                        let distfrom2 = (-m * tvy).powi(2) + (m * tvx).powi(2);
                        // Faithful to the C++ (tvx-pcx)²+(tvy-pcx)² quirk
                        // (MatrixSide.cpp:4935).
                        let distplace2 = (tvx - pcx).powi(2) + (tvy - pcx).powi(2);
                        let cannon_outrange = matches!(ta_type, Some(ObjectType::Cannon))
                            && (r_maxfd - gsm) > enemy_fire_dist as f32;
                        if placebest < 0 || cannon_outrange {
                            let lim = 0.95 * r_maxfd - gsm * ROBOT_MOVECELLS_PER_SIZE as f32 / 2.0;
                            if distplace2 > lim * lim {
                                continue;
                            }
                        } else {
                            let lim = 0.95 * r_minfd - gsm * ROBOT_MOVECELLS_PER_SIZE as f32 / 2.0;
                            if distplace2 > lim * lim {
                                continue;
                            }
                        }
                        if !have_bomb && matches!(ta_type, Some(ObjectType::RobotAi)) {
                            let lim = (200.0 + 100.0f32).powi(2);
                            if distfrom2 > lim {
                                continue;
                            }
                        }
                        let mut underfire = underfire0 as i32;
                        if distplace2 <= (enemy_fire_dist as f32).powi(2) {
                            underfire += 1;
                        }
                        let from = glam::Vec3::new(pcx, pcy, map.get_z(pcx, pcy) + 20.0);
                        let (res, _) = crate::matrix_game::map_trace::trace(
                            map,
                            &self.objects,
                            from,
                            des,
                            crate::matrix_game::common::TRACE_OBJECT
                                | crate::matrix_game::common::TRACE_NONOBJECT
                                | crate::matrix_game::common::TRACE_OBJECTSPHERE
                                | crate::matrix_game::common::TRACE_SKIP_INVISIBLE,
                            Some(rid),
                        );
                        use crate::matrix_game::map_trace::TraceStop;
                        let close = matches!(res, TraceStop::Water | TraceStop::Landscape)
                            || matches!(res, TraceStop::Object(h) if self
                                .objects
                                .get(h)
                                .map(|o| matches!(o.core().obj_type, ObjectType::MapObject))
                                .unwrap_or(false));

                        if placebest >= 0 {
                            if have_bomb {
                                if distplace2 > s_f1 {
                                    continue;
                                }
                            } else if close != s_close {
                                if close {
                                    continue;
                                }
                            } else if underfire == 0 && s_underfire != 0 {
                                // prefer un-shelled places
                            } else if underfire != 0 && s_underfire == 0 {
                                continue;
                            } else if underfire != 0 {
                                if underfire > s_underfire {
                                    continue;
                                }
                                if distplace2 < s_f1 {
                                    continue;
                                }
                            } else if distplace2 > s_f1 {
                                continue;
                            }
                        }
                        s_close = close;
                        s_f1 = distplace2;
                        s_underfire = underfire;
                        placebest = iplace;
                    }

                    if placebest >= 0 {
                        cplr = false;
                        place_set_data(map, placebest, 1);
                        let pos = place_get(map, placebest).unwrap().2;
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.env.place = placebest;
                            if prepare_break_order(r) {
                                r.move_to_high(pos.0, pos.1);
                            }
                        }
                        i += 1;
                    } else if cplr {
                        cplr = false;
                        let (st2, _) =
                            place_list(map, mm, r0_pos, center, radiusrobot, false, &mut list);
                        if st2 == 0 {
                            for &rid2 in &grp {
                                if let Some(r) = robot_mut(&mut self.objects, rid2) {
                                    r.env.place_not_found = now;
                                }
                            }
                            break;
                        }
                        zero_places(map, &list);
                        mark_occupied_places(map, &self.objects);
                        i = 0; // C++ `i=-1; continue;`
                    } else {
                        // Not found (5003-5033).
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.env.place_not_found = now;
                        }
                        let mut chosen: Option<i32> = None;
                        for &pi in &list {
                            let Some((data, mv, _, _)) = place_get(map, pi) else {
                                continue;
                            };
                            if data != 0 || mv & r_cbit != 0 {
                                continue;
                            }
                            chosen = Some(pi);
                            break;
                        }
                        if let Some(pi) = chosen {
                            place_set_data(map, pi, 1);
                            let pos = place_get(map, pi).unwrap().2;
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                r.env.place = pi;
                                if prepare_break_order(r) {
                                    r.move_to_high(pos.0, pos.1);
                                }
                            }
                            i += 1;
                        } else {
                            if place_list_grow(map, mm, &mut list, grp.len() as i32) <= 0 {
                                i += 1;
                                continue;
                            }
                            zero_places(map, &list);
                            mark_occupied_places(map, &self.objects);
                            i += 1;
                        }
                    }
                }
            }
        }

        // ── Fire correction (5038-5214) ────────────────────────────
        self.fire_correction_for(map, sid, &grp, true);
    }

    /// `RepairTL` (MatrixSide.cpp:5217-5267).
    fn repair_tl(&mut self, map: &GameMap, side: &Side, group: usize) {
        let sid = side.id;
        let grp = self.group_robots_of(sid, group);
        if grp.is_empty() {
            return;
        }
        for &rid in &grp {
            let (repair_dist, my_wp) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (
                    r.repair_dist,
                    get_world_pos(&self.objects, rid).unwrap_or_default(),
                )
            };
            if repair_dist <= 0.0 {
                continue;
            }
            let keep = {
                let r = robot_ref(&self.objects, rid).unwrap();
                r.env
                    .target
                    .filter(|&t| {
                        self.objects
                            .get(t)
                            .map(|o| o.is_live() && o.need_repair())
                            .unwrap_or(false)
                    })
                    .and_then(|t| get_world_pos(&self.objects, t))
                    .map(|tp| (tp - my_wp).length_squared() < repair_dist * repair_dist)
                    .unwrap_or(false)
            };
            if keep {
                continue;
            }
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.env.target = None;
            }
            let mut new_target = None;
            for oid in self.objects.iter_units() {
                if oid == rid {
                    continue;
                }
                let Some(obj) = self.objects.get(oid) else {
                    continue;
                };
                if !obj.is_live() || obj.side() != sid || !obj.need_repair() {
                    continue;
                }
                let Some(tp) = get_world_pos(&self.objects, oid) else {
                    continue;
                };
                if (tp - my_wp).length_squared() < repair_dist * repair_dist {
                    new_target = Some(oid);
                    break;
                }
            }
            if let Some(nt) = new_target {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.env.target = Some(nt);
                }
            }
        }
        // Fire the repair beams (5260-5266).
        for &rid in &grp {
            let t = robot_ref(&self.objects, rid).unwrap().env.target;
            let Some(t) = t else { continue };
            let Some(des) = crate::matrix_game::logic::point_of_aim(map, &self.objects, t) else {
                continue;
            };
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.fire(des, 2);
            }
        }
    }

    /// `AssignPlace(group, region)` (MatrixSide.cpp:5303-5455) — line
    /// the group up in `region`, strongest robots to the enemy-facing
    /// edge.
    fn assign_place_group(&mut self, map: &GameMap, side: &mut Side, group: usize, region: i32) {
        if region < 0 {
            return;
        }
        let sid = side.id;
        let now = self.elapsed_ms as i32;
        let Some(g) = RegionGraph::snapshot(map) else {
            return;
        };
        // Clear all places of the region, mark those taken by others.
        let (region_place, region_place_all) = {
            let Some(rn_lock) = map.road_network.as_ref() else {
                return;
            };
            let rn = rn_lock.lock().unwrap();
            let Some(reg) = rn.regions.get(region as usize) else {
                return;
            };
            (reg.place.clone(), reg.place_all.clone())
        };
        zero_places(map, &region_place_all);

        let mut grp: Vec<ObjectId> = Vec::new();
        let mut mm = 0u8;
        for id in self.objects.iter_units() {
            if let Some(r) = robot_ref(&self.objects, id) {
                if !r.is_live() {
                    continue;
                }
                if r.side == sid && r.group_logic == group as i32 {
                    grp.push(id);
                    mm |= chassis_bit(r);
                } else {
                    place_set_data(map, r.env.place, 1);
                }
            } else if is_live_unit(&self.objects, id) {
                if let Some(c) = cannon_ref(&self.objects, id) {
                    place_set_data(map, c.place, 1);
                }
            }
        }
        if grp.is_empty() {
            return;
        }

        self.sort_robot_list(&mut grp);

        // Enemy direction vector (5338-5356).
        let cr = {
            let r = robot_ref(&self.objects, grp[0]).unwrap();
            get_region(map, r.map_x, r.map_y)
        };
        let r_near = Self::find_near_region_with_utr(&g, side, cr, &[], 4 + 32 + 64);
        let team = side.logic_groups[group].team;
        let (tp, tp2) = if r_near >= 0 && r_near != cr {
            (g.centers[r_near as usize], g.centers[cr.max(0) as usize])
        } else if team >= 0
            && (team as usize) < side.teams.len()
            && side.teams[team as usize].region_mass_prev != r_near
            && side.teams[team as usize].region_mass_prev >= 0
            && cr >= 0
        {
            (
                g.centers[cr as usize],
                g.centers[side.teams[team as usize].region_mass_prev as usize],
            )
        } else {
            ((0, 0), (1, 1))
        };
        let mut venemy = glam::Vec2::new((tp.0 - tp2.0) as f32, (tp.1 - tp2.1) as f32);
        let len = venemy.length();
        if len > 0.0 {
            venemy /= len;
        }
        let vcenter = glam::Vec2::new(
            g.centers[region as usize].0 as f32,
            g.centers[region as usize].1 as f32,
        );

        // Free-place list sorted by projection toward the enemy.
        let mut list: Vec<i32> = Vec::new();
        for &pi in &region_place {
            if let Some((data, _, _, _)) = place_get(map, pi) {
                if data == 0 {
                    list.push(pi);
                }
            }
        }
        let sort_by_projection = |map: &GameMap, list: &mut Vec<i32>| {
            let mut keyed: Vec<(f32, i32)> = list
                .iter()
                .map(|&pi| {
                    let pos = place_get(map, pi).map(|p| p.2).unwrap_or((0, 0));
                    let pr = venemy.x * (pos.0 as f32 - vcenter.x)
                        + venemy.y * (pos.1 as f32 - vcenter.y);
                    (pr, pi)
                })
                .collect();
            keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            *list = keyed.into_iter().map(|(_, pi)| pi).collect();
        };
        sort_by_projection(map, &mut list);

        // Assign (5384-5426).
        let mut t = 0usize;
        while t < grp.len() {
            let rid = grp[t];
            let cbit = chassis_bit(robot_ref(&self.objects, rid).unwrap());
            let mut chosen: Option<i32> = None;
            for &pi in &list {
                let Some((data, mv, _, _)) = place_get(map, pi) else {
                    continue;
                };
                if data != 0 || mv & cbit != 0 {
                    continue;
                }
                chosen = Some(pi);
                break;
            }
            match chosen {
                Some(pi) => {
                    place_set_data(map, pi, 1);
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.env.place = pi;
                    }
                    t += 1;
                }
                None => {
                    for &rid2 in &grp {
                        if let Some(r) = robot_mut(&mut self.objects, rid2) {
                            r.env.place_not_found = now;
                        }
                    }
                    if place_list_grow(map, mm, &mut list, grp.len() as i32) <= 0 {
                        t += 1;
                        continue;
                    }
                    sort_by_projection(map, &mut list);
                    zero_places(map, &list);
                    for id in self.objects.iter_units() {
                        if let Some(r) = robot_ref(&self.objects, id) {
                            if !r.is_live() {
                                continue;
                            }
                            if r.side != sid || r.group_logic != group as i32 {
                                place_set_data(map, r.env.place, 1);
                            }
                        } else if is_live_unit(&self.objects, id) {
                            if let Some(c) = cannon_ref(&self.objects, id) {
                                place_set_data(map, c.place, 1);
                            }
                        }
                    }
                    t = 0; // C++ `t=-1; continue;`
                }
            }
        }
    }

    /// `SortRobotList` (MatrixSide.cpp:5457-5537) — strength ascending,
    /// then interleave: every 2nd a bomber, every 3rd a repairer.
    fn sort_robot_list(&self, rl: &mut Vec<ObjectId>) {
        if rl.len() <= 1 {
            return;
        }
        rl.sort_by(|&a, &b| {
            let sa = robot_ref(&self.objects, a).map(|r| r.strength).unwrap_or(0.0);
            let sb = robot_ref(&self.objects, b).map(|r| r.strength).unwrap_or(0.0);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut rln: Vec<ObjectId> = Vec::new();
        let mut rlr: Vec<ObjectId> = Vec::new();
        let mut rlb: Vec<ObjectId> = Vec::new();
        for &rid in rl.iter() {
            let r = robot_ref(&self.objects, rid).unwrap();
            if r.have_bomb(&self.objects) {
                rlb.push(rid);
            } else if r.have_repair != 0 {
                rlr.push(rid);
            } else {
                rln.push(rid);
            }
        }
        rl.clear();
        let (mut s_normal, mut s_bomb, mut s_repair) = (0usize, 0usize, 0usize);
        let (mut i_bomb, mut i_repair) = (0i32, 0i32);
        while s_normal < rln.len() || s_repair < rlr.len() || s_bomb < rlb.len() {
            if i_bomb >= 1 && s_bomb < rlb.len() {
                rl.push(rlb[s_bomb]);
                s_bomb += 1;
                i_bomb = 0;
                i_repair = 0;
            } else if i_repair >= 2 && s_repair < rlr.len() {
                rl.push(rlr[s_repair]);
                s_repair += 1;
                i_repair = 0;
            } else if s_normal < rln.len() {
                rl.push(rln[s_normal]);
                s_normal += 1;
                i_bomb += 1;
                i_repair += 1;
            } else {
                i_bomb += 1;
                i_repair += 1;
            }
        }
    }

    /// `PlaceInRegion` (MatrixSide.cpp:5539-5577).
    fn place_in_region(&self, map: &GameMap, rid: ObjectId, place: i32, region: i32) -> bool {
        if place < 0 || region < 0 {
            return false;
        }
        if place_region(map, place) == region {
            return true;
        }
        let (region_place, region_place_all) = {
            let Some(rn_lock) = map.road_network.as_ref() else {
                return false;
            };
            let rn = rn_lock.lock().unwrap();
            let Some(reg) = rn.regions.get(region as usize) else {
                return false;
            };
            (reg.place.clone(), reg.place_all.clone())
        };
        zero_places(map, &region_place);
        crate::matrix_game::side_player::mark_occupied_places_skip(
            map,
            &self.objects,
            Some(rid),
        );
        let cbit = chassis_bit(robot_ref(&self.objects, rid).unwrap());
        for &pi in &region_place {
            let Some((data, mv, _, _)) = place_get(map, pi) else {
                continue;
            };
            if data != 0 || mv & cbit != 0 {
                continue;
            }
            return false; // a free spot inside the region exists
        }
        // Near-region places count as inside (5573-5575).
        region_place_all
            .get(region_place.len()..)
            .map(|s| s.contains(&place))
            .unwrap_or(false)
    }

    // ─────────────────────────────────────────────────────────────────
    // Production
    // ─────────────────────────────────────────────────────────────────

    /// `BuildRobotMinStrange` (MatrixSide.cpp:5582-5607).
    fn build_robot_min_strange(&self, map: &GameMap, sid: i32, base_region: i32) -> f32 {
        let mut minstrange = 0.0f32;
        let near = |r: i32| -> bool {
            if r == base_region {
                return true;
            }
            if r < 0 || base_region < 0 {
                return false;
            }
            let Some(rn_lock) = map.road_network.as_ref() else {
                return false;
            };
            let rn = rn_lock.lock().unwrap();
            rn.is_nerest_region(r, base_region)
        };
        for id in self.objects.iter_units() {
            if let Some(r) = robot_ref(&self.objects, id) {
                if !r.is_live() {
                    continue;
                }
                let i = get_region(map, r.map_x, r.map_y);
                if near(i) {
                    if r.side == sid {
                        minstrange -= r.strength;
                    } else {
                        minstrange += r.strength;
                    }
                }
            } else if is_live_unit(&self.objects, id) {
                if let Some(c) = cannon_ref(&self.objects, id) {
                    if c.side == sid {
                        minstrange -= c.get_strength();
                    }
                }
            }
        }
        (minstrange * 0.7).max(0.0)
    }

    /// `BuildRobot` (MatrixSide.cpp:5609-5912).
    fn build_robot(&mut self, map: &GameMap, side: &mut Side) {
        use crate::matrix_game::interface::constructor::global_ai_robots;
        let sid = side.id;
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let now = self.elapsed_ms as i32;
        let cat = global_ai_robots();
        if cat.bots.is_empty() {
            return;
        }

        let mut basecnt = 0i32;
        let mut wr = [0i32; 4];
        let mut cnt = 0i32;
        let mut base: Option<ObjectId> = None;
        let mut base_region = -1i32;
        let mut minstrange = 0.0f32;

        for id in self.objects.iter_units() {
            if let Some(b) = building_ref(&self.objects, id) {
                if b.side != sid || !b.is_live() {
                    continue;
                }
                if b.is_base() {
                    basecnt += 1;
                    cnt += b.build_stack.items() as i32;
                    let i = get_region(map, (b.pos.x / gsm) as i32, (b.pos.y / gsm) as i32);
                    match base {
                        None => {
                            base = Some(id);
                            base_region = i;
                            minstrange = self.build_robot_min_strange(map, sid, i);
                        }
                        Some(_) => {
                            let istr = self.build_robot_min_strange(map, sid, i);
                            if istr != minstrange {
                                if istr < minstrange {
                                    base = Some(id);
                                    base_region = i;
                                    minstrange = istr;
                                }
                            } else if i >= 0 && base_region >= 0 {
                                let di = side.region_stats[i as usize].enemy_robot_dist;
                                let dk = side.region_stats[base_region as usize].enemy_robot_dist;
                                if di != dk {
                                    if dk < 0 || (di >= 0 && di < dk) {
                                        base = Some(id);
                                        base_region = i;
                                        minstrange = istr;
                                    }
                                } else {
                                    let bi =
                                        side.region_stats[i as usize].enemy_building_dist;
                                    let bk = side.region_stats[base_region as usize]
                                        .enemy_building_dist;
                                    if bi != bk && (bk < 0 || (bi >= 0 && bi < bk)) {
                                        base = Some(id);
                                        base_region = i;
                                        minstrange = istr;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    match b.kind {
                        BuildingType::Titan => wr[0] += 1,
                        BuildingType::Electronic => wr[1] += 1,
                        BuildingType::Energy => wr[2] += 1,
                        BuildingType::Plasma => wr[3] += 1,
                        _ => {}
                    }
                }
            } else if let Some(r) = robot_ref(&self.objects, id) {
                if r.side == sid && r.is_live() && r.team >= 0 {
                    cnt += 1;
                }
            }
        }
        if cnt >= self.compute_max_side_robots(sid) {
            return;
        }
        let Some(base_id) = base else { return };
        let queue_len = building_ref(&self.objects, base_id)
            .map(|b| b.build_stack.items())
            .unwrap_or(0);
        if queue_len > 0 {
            return; // a robot is already queued (5676)
        }

        // How long we may wait for resources (5679-5683).
        let mut waittime = 10000i32;
        if base_region >= 0 {
            let st = &side.region_stats[base_region as usize];
            if st.enemy_robot_dist >= 0 {
                waittime = 10000 * (st.enemy_robot_dist - 1).max(0);
            } else if st.enemy_building_dist >= 0 {
                waittime = 10000 * (st.enemy_building_dist - 1).max(0);
            }
        }
        waittime = float2int(side.wait_res_mul * waittime.min(40000) as f32);

        // Wait time until the planned robot is affordable (5686-5696).
        let timings = crate::matrix_game::config::global().timings;
        let timing_arr = [
            timings.resource_titan,
            timings.resource_electronics,
            timings.resource_energy,
            timings.resource_plasma,
        ];
        const RESOURCES_INCOME: i32 = 10;
        const RESOURCES_INCOME_BASE: i32 = 3;
        let mut waitend = -1i32;
        if side.wait_res_for_build_robot >= 0 {
            let bot = &cat.bots[side.wait_res_for_build_robot as usize];
            waitend = 0;
            for r in 0..4 {
                let k = bot.resources[r] - side.resources[r];
                if k > 0 {
                    if wr[r] <= 0 && basecnt <= 0 {
                        waitend = 1000000000;
                        break;
                    }
                    let income = wr[r] * RESOURCES_INCOME
                        + RESOURCES_INCOME_BASE * side.base_res_force / 100 * basecnt;
                    if income > 0 && timing_arr[r] > 0 {
                        waitend =
                            waitend.max(float2int((k * timing_arr[r]) as f32 / income as f32));
                    }
                }
            }
        }

        // Projected resources after waiting (5699-5709).
        let mut mr = 0i32;
        for r in 0..4 {
            mr += side.resources[r].min(2000);
        }
        mr = float2int((mr / 4) as f32 * 0.6);
        let mut wrr = [0i32; 4];
        for r in 0..4 {
            let per_tick = RESOURCES_INCOME * wr[r]
                + RESOURCES_INCOME_BASE * side.base_res_force / 100 * basecnt;
            let ticks = if timing_arr[r] > 0 {
                waittime / timing_arr[r]
            } else {
                0
            };
            wrr[r] = per_tick * ticks + side.resources[r];
        }

        // Candidate lists (5712-5763).
        let mut nobomb = [side.build_robot_last, side.build_robot_last2, side.build_robot_last3]
            .iter()
            .any(|&l| l >= 0 && cat.bots[l as usize].have_bomb);
        if !nobomb && now - side.time_last_bomb < side.time_next_bomb {
            nobomb = true;
        }
        let norepair = [side.build_robot_last, side.build_robot_last2, side.build_robot_last3]
            .iter()
            .any(|&l| l >= 0 && cat.bots[l as usize].have_repair);

        let too_similar = |i: usize| -> bool {
            if i as i32 == side.wait_res_for_build_robot {
                return false;
            }
            if side.build_robot_last >= 0
                && cat.bots[i].dif_weapon(&cat.bots[side.build_robot_last as usize]) > 0.6
            {
                return true;
            }
            if side.build_robot_last2 >= 0
                && cat.bots[i].dif_weapon(&cat.bots[side.build_robot_last2 as usize]) > 0.8
            {
                return true;
            }
            false
        };

        let mut list: Vec<usize> = Vec::new();
        let mut lwait: Vec<usize> = Vec::new();
        for (i, bot) in cat.bots.iter().enumerate() {
            if nobomb && bot.have_bomb {
                continue;
            }
            if norepair && bot.have_repair {
                continue;
            }
            if !norepair && !bot.have_repair {
                continue;
            }
            if bot.strength < minstrange {
                continue;
            }
            let affordable = (0..4).all(|r| side.resources[r] >= bot.resources[r]);
            if affordable {
                if !too_similar(i) {
                    list.push(i);
                }
            } else if (0..4).all(|r| wrr[r] >= bot.resources[r]) && !too_similar(i) {
                lwait.push(i);
            }
        }
        if list.is_empty() && lwait.is_empty() {
            return;
        }

        // The awaited robot became affordable → build it (5766-5786).
        if side.wait_res_for_build_robot >= 0 {
            if let Some(&i) = list
                .iter()
                .find(|&&i| i as i32 == side.wait_res_for_build_robot)
            {
                if queue_len < 6 {
                    self.ai_queue_robot(side, base_id, &cat.bots[i]);
                    if cat.bots[i].have_bomb {
                        side.time_last_bomb = now;
                    }
                    side.build_robot_last3 = side.build_robot_last2;
                    side.build_robot_last2 = side.build_robot_last;
                    side.build_robot_last = i as i32;
                }
                side.wait_res_for_build_robot = -1;
                return;
            }
        }

        // Cancel a too-distant wait (5789-5792).
        if waitend >= 0 && waitend > waittime {
            side.wait_res_for_build_robot = -1;
        }
        if side.wait_res_for_build_robot >= 0 {
            return;
        }

        // Choose to wait for a stronger robot or build now (5804-5909).
        let scarce_cost = |i: usize| -> i32 {
            (0..4)
                .filter(|&r| side.resources[r] < mr)
                .map(|r| cat.bots[i].resources[r])
                .sum()
        };
        let refine = |cand: &mut Vec<usize>, cat: &crate::matrix_game::interface::constructor::AIRobotCatalogue, side: &Side| {
            // Keep only the top-strength tier (within 0.7×).
            let mut n = cand.len();
            for i in 1..cand.len() {
                if cat.bots[cand[i]].strength < 0.7 * cat.bots[cand[0]].strength {
                    n = i;
                    break;
                }
            }
            cand.truncate(n);
            if (0..4).any(|r| side.resources[r] < mr) {
                cand.sort_by_key(|&i| scarce_cost(i));
                let ik = scarce_cost(cand[0]);
                let ik = ik + ik / 10;
                let mut n = cand.len();
                for i in 1..cand.len() {
                    if scarce_cost(cand[i]) > ik {
                        n = i;
                        break;
                    }
                }
                cand.truncate(n);
            }
        };
        let pick_by_pripor = |cand: &[usize], rng: &mut crate::matrix_game::logic::Rnd| -> usize {
            let total: i32 = cand.iter().map(|&i| cat.bots[i].pripor).sum();
            let mut k = rng.range(0, (total - 1).max(0));
            for &i in cand {
                k -= cat.bots[i].pripor;
                if k < 0 {
                    return i;
                }
            }
            *cand.last().unwrap()
        };

        if !list.is_empty()
            && !lwait.is_empty()
            && cat.bots[lwait[0]].strength * 0.6 > cat.bots[list[0]].strength
        {
            refine(&mut lwait, &cat, side);
            let i = pick_by_pripor(&lwait, &mut self.rng);
            side.wait_res_for_build_robot = i as i32;
        } else if !list.is_empty() {
            refine(&mut list, &cat, side);
            let i = pick_by_pripor(&list, &mut self.rng);
            if queue_len < 6 {
                self.ai_queue_robot(side, base_id, &cat.bots[i]);
                if cat.bots[i].have_bomb {
                    side.time_last_bomb = now;
                }
                side.build_robot_last3 = side.build_robot_last2;
                side.build_robot_last2 = side.build_robot_last;
                side.build_robot_last = i as i32;
            }
        }
    }

    /// Deduct resources + queue — the `m_Constructor->BuildSpecialBot`
    /// call (MatrixSide.cpp:5770-5772 / 5896-5899).
    fn ai_queue_robot(
        &mut self,
        side: &mut Side,
        base_id: ObjectId,
        bot: &crate::matrix_game::interface::constructor::SpecialBot,
    ) {
        for r in 0..4 {
            side.resources[r] = (side.resources[r] - bot.resources[r]).max(0);
        }
        let cfg = bot.to_robot_config();
        if let Some(b) = crate::matrix_game::logic::building_mut(&mut self.objects, base_id) {
            b.queue_robot(cfg);
        }
    }

    /// `BuildCannon` (MatrixSide.cpp:5914-6028).
    fn build_cannon(&mut self, map: &GameMap, side: &mut Side) {
        let sid = side.id;
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let maxrobot = self.compute_max_side_robots(sid);

        let mut curcannoncnt = 0;
        for id in self.objects.iter_units() {
            if is_live_unit(&self.objects, id) {
                if let Some(c) = cannon_ref(&self.objects, id) {
                    if c.side == sid {
                        curcannoncnt += 1;
                    }
                }
            }
        }
        if side.robots_cnt < maxrobot && curcannoncnt >= side.robots_cnt {
            return;
        }

        // Building nearest the front with a free slot (5931-5952).
        let mut building: Option<ObjectId> = None;
        let mut mdist = 0i32;
        for id in self.objects.iter_units() {
            let Some(b) = building_ref(&self.objects, id) else {
                continue;
            };
            if b.side != sid || !b.is_live() {
                continue;
            }
            if b.turrets_have >= b.turrets_max {
                continue;
            }
            let r = get_region(map, (b.pos.x / gsm) as i32, (b.pos.y / gsm) as i32);
            if r < 0 {
                continue;
            }
            if self.build_robot_min_strange(map, sid, r) > 0.0 {
                continue;
            }
            let dist = side.region_stats[r as usize].enemy_robot_dist;
            if building.is_some() && dist >= mdist {
                continue;
            }
            mdist = dist;
            building = Some(id);
        }
        let Some(bid) = building else { return };

        // Turret-type histogram: standing + queued (5954-5975).
        let mut ct = [0i32; 4];
        for id in self.objects.iter_units() {
            if !is_live_unit(&self.objects, id) {
                continue;
            }
            let Some(c) = cannon_ref(&self.objects, id) else {
                continue;
            };
            if c.parent == Some(bid) && (1..=4).contains(&c.kind) {
                ct[(c.kind - 1) as usize] += 1;
            }
        }
        {
            let Some(b) = building_ref(&self.objects, bid) else {
                return;
            };
            for item in b.build_stack.iter() {
                if let crate::matrix_game::object_building::PendingKind::Turret {
                    turret_kind,
                    ..
                } = item.kind
                {
                    if (1..=4).contains(&turret_kind) {
                        ct[(turret_kind - 1) as usize] += 1;
                    }
                }
            }
        }
        let vmin = *ct.iter().min().unwrap();
        let mut curtype;
        loop {
            curtype = self.rng.range(0, 3);
            if ct[curtype as usize] == vmin {
                break;
            }
        }

        let cost = crate::matrix_game::config::global()
            .turrets
            .cost_of(curtype + 1);
        for r in 0..4 {
            if cost.resources[r] > side.resources[r] {
                return; // not enough resources (5984-5985)
            }
        }
        {
            let Some(b) = building_ref(&self.objects, bid) else {
                return;
            };
            if b.build_stack.items() > 0 {
                return; // already building something (5987)
            }
        }

        // Random free slot (5989-5995).
        let free_slots: Vec<i32> = {
            let Some(b) = building_ref(&self.objects, bid) else {
                return;
            };
            b.turret_places
                .iter()
                .enumerate()
                .filter(|(_, p)| p.cannon_type < 0)
                .map(|(i, _)| i as i32)
                .collect()
        };
        if free_slots.is_empty() {
            return;
        }
        let slot = free_slots[self.rng.range(0, free_slots.len() as i32 - 1) as usize];

        let queued = crate::matrix_game::logic::building_mut(&mut self.objects, bid)
            .map(|b| b.queue_turret_slot(slot, curtype + 1))
            .unwrap_or(false);
        if queued {
            for r in 0..4 {
                side.resources[r] = (side.resources[r] - cost.resources[r]).max(0);
            }
        }
    }

    /// The AI branch of `CalcMaxSpeed` (MatrixSide.cpp:2343-2448) —
    /// keyed off `m_LogicGroup` instead of the player groups.
    fn ai_calc_max_speed(&mut self, side: &Side) {
        use crate::matrix_game::logic::COLLIDE_BOT_R;
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let half = gsm * ROBOT_MOVECELLS_PER_SIZE as f32 / 2.0;
        let sid = side.id;

        for &rid in &self.side_robots_of(sid) {
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.reset_group_speed();
            }
        }

        for i in 0..MAX_LOGIC_GROUP {
            if side.logic_groups[i].robots_cnt <= 0 {
                continue;
            }
            let grp = self.group_robots_of(sid, i);
            if grp.len() <= 1 {
                continue;
            }
            let mut cx = 0.0f32;
            let mut cy = 0.0f32;
            let mut dx = 0.0f32;
            let mut dy = 0.0f32;
            struct Entry {
                rid: ObjectId,
                pos: glam::Vec2,
                pr: f32,
                returning: bool,
                moving: bool,
                enemies: bool,
            }
            let mut rl: Vec<Entry> = Vec::with_capacity(grp.len());
            for &rid in &grp {
                let r = robot_ref(&self.objects, rid).unwrap();
                let pos = glam::Vec2::new(r.pos_x, r.pos_y);
                cx += pos.x;
                cy += pos.y;
                let (dest, returning, moving) = if let Some(tp) = r.return_coords() {
                    (
                        glam::Vec2::new(gsm * tp.0 as f32 + half, gsm * tp.1 as f32 + half),
                        true,
                        false,
                    )
                } else if let Some(tp) = r.move_to_coords() {
                    (
                        glam::Vec2::new(gsm * tp.0 as f32 + half, gsm * tp.1 as f32 + half),
                        false,
                        true,
                    )
                } else {
                    (pos, false, false)
                };
                dx += dest.x;
                dy += dest.y;
                rl.push(Entry {
                    rid,
                    pos,
                    pr: 0.0,
                    returning,
                    moving,
                    enemies: r.env.enemy_cnt() > 0,
                });
            }
            let d = 1.0 / rl.len() as f32;
            cx *= d;
            cy *= d;
            dx = dx * d - cx;
            dy = dy * d - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 100.0 {
                continue;
            }
            dx /= dist;
            dy /= dist;
            let mut maxpr = f32::MIN;
            for e in rl.iter_mut() {
                e.pr = dx * (e.pos.x - cx) + dy * (e.pos.y - cy);
                maxpr = maxpr.max(e.pr);
            }
            rl.sort_by(|a, b| a.pr.partial_cmp(&b.pr).unwrap_or(std::cmp::Ordering::Equal));
            let mut u = 0usize;
            while u + 1 < rl.len() {
                if rl[u + 1].pr - rl[u].pr > COLLIDE_BOT_R * 7.0 {
                    break;
                }
                u += 1;
            }
            let minpr = rl[u].pr;
            let maxpr = maxpr - minpr;

            for e in &rl {
                if e.returning || !e.moving || e.enemies {
                    continue;
                }
                let pr = e.pr - minpr;
                if pr < COLLIDE_BOT_R * 10.0 {
                    continue;
                }
                let k = 0.6 + 0.4 * (maxpr - pr) / (maxpr - COLLIDE_BOT_R * 10.0);
                if let Some(r) = robot_mut(&mut self.objects, e.rid) {
                    r.scale_group_speed(k);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_game::interface::constructor::{
        set_global_ai_robots, AIRobotCatalogue, SpecialBot,
    };
    use crate::matrix_game::map::BuildingInstance;
    use crate::matrix_game::object_building::{Building, BuildingType};
    use crate::matrix_game::road_network::{Place, Point, Region, RoadNetwork};
    use crate::matrix_game::robot::{ChassisKind, Robot, RobotState};

    fn spawn_robot(game: &mut MapLogic, side: i32, x: f32, y: f32) -> ObjectId {
        let mut r = Robot::new(glam::Vec3::new(x, y, 0.0), side, ChassisKind::Track);
        r.state = RobotState::Idle;
        r.map_x = (x / GameMap::GLOBAL_SCALE_MOVE) as i32;
        r.map_y = (y / GameMap::GLOBAL_SCALE_MOVE) as i32;
        r.strength = 10.0;
        let id = game.objects.spawn(Box::new(r));
        if let Some(r) = robot_mut(&mut game.objects, id) {
            r.self_id = Some(id);
        }
        game.objects.add_lt(id);
        id
    }

    fn spawn_building(w: &mut MapLogic, side: i32, kind: BuildingType, x: f32, y: f32) -> ObjectId {
        let inst = BuildingInstance {
            x,
            y,
            build_z: 0.0,
            angle: 0,
            side: side as u8,
            kind: kind as u8,
            turrets_places_cnt: 4,
            shadow_kind: 0,
            shadow_size: 128,
            turret_places: Vec::new(),
        };
        let b = Building::from_instance(&inst);
        let id = w.objects.spawn(Box::new(b));
        if let Some(obj) = w.objects.get_mut(id) {
            let b_mut: &mut Building =
                unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
            b_mut.self_id = Some(id);
        }
        w.objects.add_lt(id);
        id
    }

    /// 64×64 flat map with a synthetic two-region road network:
    /// region 0 around cell (8,8), region 1 around (44,8), adjacent.
    fn map_with_regions() -> GameMap {
        let mut map = GameMap::test_flat(64, 64, 0.0);
        let mut rn = RoadNetwork::new();
        let mk = |x: i32, y: i32, region: i32| -> Place {
            Place {
                pos: Point::new(x, y),
                region,
                ..Place::default()
            }
        };
        rn.places = vec![
            mk(4, 4, 0),
            mk(4, 10, 0),
            mk(10, 4, 0),
            mk(10, 10, 0),
            mk(40, 4, 1),
            mk(40, 10, 1),
            mk(46, 4, 1),
            mk(46, 10, 1),
        ];
        // Chain the places so wave searches can traverse them.
        for i in 0..rn.places.len() {
            let mut near = Vec::new();
            if i > 0 {
                near.push(i as i32 - 1);
            }
            if i + 1 < rn.places.len() {
                near.push(i as i32 + 1);
            }
            rn.places[i].near_move = vec![0; near.len()];
            rn.places[i].near = near;
        }
        let mut r0 = Region::default();
        r0.center = Point::new(8, 8);
        r0.place = vec![0, 1, 2, 3];
        r0.place_all = vec![0, 1, 2, 3, 4];
        r0.near = vec![1];
        r0.near_move = vec![0];
        let mut r1 = Region::default();
        r1.center = Point::new(44, 8);
        r1.place = vec![4, 5, 6, 7];
        r1.place_all = vec![4, 5, 6, 7, 3];
        r1.near = vec![0];
        r1.near_move = vec![0];
        rn.regions = vec![r0, r1];
        rn.init_pl(128, 128);
        map.road_network = Some(std::sync::Mutex::new(rn));
        map
    }

    fn repair_bot_catalogue() -> AIRobotCatalogue {
        // First-build pass requires a repair template (BuildRobot
        // skips non-repair bots until one was built recently).
        let mut bot = SpecialBot {
            pripor: 1,
            have_repair: true,
            strength: 5.0,
            ..SpecialBot::default()
        };
        bot.chassis.kind = crate::matrix_game::config::RobotUnitKind(3);
        bot.armor.unit.kind = crate::matrix_game::config::RobotUnitKind(1);
        AIRobotCatalogue { bots: vec![bot] }
    }

    #[test]
    fn clac_spawn_team_prefers_empty_team() {
        let map = map_with_regions();
        let mut game = MapLogic::with_seed(3);
        let mut side = Side::new(2);
        side.teams[0].robot_cnt = 2;
        side.teams[1].robot_cnt = 0;
        side.teams[2].robot_cnt = 1;
        let t = game.clac_spawn_team(&map, &mut side, 0, 0);
        assert_eq!(t, 1);
        // clear_team reset the picked team.
        assert_eq!(side.teams[1].action.ty, LogicActionType::None);
    }

    #[test]
    fn ai_side_builds_robot_when_affordable() {
        set_global_ai_robots(repair_bot_catalogue());
        let map = map_with_regions();
        let mut game = MapLogic::with_seed(5);
        let base = spawn_building(&mut game, 2, BuildingType::Base, 80.0, 80.0);
        game.ensure_sides_from_objects();

        let idx = game.other_sides.iter().position(|s| s.id == 2).unwrap();
        let mut side = std::mem::take(&mut game.other_sides[idx]);
        resize_stats(&mut side, 2);
        game.build_robot(&map, &mut side);
        game.other_sides[idx] = side;

        let queued = building_ref(&game.objects, base)
            .map(|b| b.build_stack.robots_cnt())
            .unwrap_or(0);
        assert_eq!(queued, 1, "AI should queue a robot at its base");
    }

    #[test]
    fn ai_takt_assigns_teams_and_actions() {
        set_global_ai_robots(repair_bot_catalogue());
        let map = map_with_regions();
        let mut game = MapLogic::with_seed(9);
        spawn_building(&mut game, 2, BuildingType::Base, 80.0, 80.0);
        let r1 = spawn_robot(&mut game, 2, 80.0, 80.0);
        let r2 = spawn_robot(&mut game, 2, 90.0, 90.0);
        game.ensure_sides_from_objects();
        assert_eq!(
            game.side_by_id(2).map(|s| s.status),
            Some(SideStatus::Active)
        );

        game.elapsed_ms = 10;
        game.ai_side_logic_takt(&map, 2);
        game.elapsed_ms = 200;
        game.ai_side_logic_takt(&map, 2);

        // Regroup put both robots into a logic group.
        let g1 = robot_ref(&game.objects, r1).unwrap().group_logic;
        let g2 = robot_ref(&game.objects, r2).unwrap().group_logic;
        assert!(g1 >= 0, "robot 1 must join a logic group");
        assert_eq!(g1, g2, "robots within 300 merge into one group");

        let side = game.side_by_id(2).unwrap();
        assert_eq!(side.teams[0].robot_cnt, 2, "census counts both robots");
        assert_ne!(
            side.teams[0].action.ty,
            LogicActionType::None,
            "TaktHL must assign a team action"
        );
        // TaktTL placed the robots somewhere in the region.
        let p1 = robot_ref(&game.objects, r1).unwrap().env.place;
        let p2 = robot_ref(&game.objects, r2).unwrap().env.place;
        assert!(p1 >= 0 && p2 >= 0, "robots must receive places");
        assert_ne!(p1, p2, "distinct places");
    }
}
