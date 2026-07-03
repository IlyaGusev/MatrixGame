//! Port of the PLAYER-logic slice of `MatrixSide.cpp` —
//! `CMatrixSideUnit::TaktPL` / `FirePL` / `RepairPL` / `WarPL`, the
//! `PGOrder*` command family, the `PG*` placement helpers, and the
//! file-scope inline helpers at MatrixSide.cpp:9353-9548.
//!
//! The enemy-AI sides (`TaktHL` / `TaktTL` / team logic) are out of
//! scope by project decision, so everything here operates on
//! `MapLogic::player_side` directly — which also keeps the borrow
//! splits trivial (`&mut self.player_side` + `&self.objects` are
//! disjoint fields).
//!
//! Locking discipline: the road network sits behind a `Mutex` on
//! `GameMap`. Helpers that need a single place lookup lock/unlock
//! internally; no caller holds the guard across a call into
//! `place_list` / `find_near_place` (std mutexes don't re-enter).

use crate::matrix_game::common::float2int;
use crate::matrix_game::logic::{
    self, building_ref, cannon_ref, get_map_pos, get_region, get_world_pos, is_live_unit,
    place_list, place_list_grow, point_of_aim, robot_mut, robot_ref, MapLogic,
};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{MapStatic, ObjectId, ObjectType, Objects};
use crate::matrix_game::map_trace::{trace, TraceStop};
use crate::matrix_game::object_cannon::CannonState;
use crate::matrix_game::logic::ROBOT_MOVECELLS_PER_SIZE;
use crate::matrix_game::robot::{OrderType, Robot};
use crate::matrix_game::side::{LogicRegion, PlayerOrder, MAX_LOGIC_GROUP, REGION_PATH_MAX_CNT};

/// Game sounds the player-order layer requests. Queued on
/// [`MapLogic::sound_queue`]; the audio backend drains it (stage 7 —
/// until then the queue is the verified dispatch surface).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameSound {
    /// `S_ORDER_INPROGRESS1/2` — picked by `IRND(2)` at call site.
    OrderInProgress1,
    OrderInProgress2,
    /// `S_ORDER_ACCEPT`.
    OrderAccept,
    /// `S_ORDER_CAPTURE` / `S_ORDER_CAPTURE_PUSH` / fuck-off variant.
    OrderCapture,
    OrderCapturePush,
    OrderCaptureFuckOff,
    OrderAttack,
    OrderRepair,
    OrderAutoCapture,
    OrderAutoAttack,
    OrderAutoDefence,
    /// `ChassisSounds/<kind>` MoveTo / Patrol voice lines — resolved
    /// by the audio layer from the per-chassis config block.
    ChassisMoveTo(i32),
    ChassisPatrol(i32),
}

// ── File-scope inline helpers (MatrixSide.cpp:9353-9548) ───────────────

/// `PrepareBreakOrder` (MatrixSide.cpp:9353-9356): stamps
/// `env.order_no_break = !CanBreakOrder()` and returns whether the
/// order may be (re)issued.
pub fn prepare_break_order(r: &mut Robot) -> bool {
    let can = r.can_break_order();
    r.env.order_no_break = !can;
    can
}

/// `CanChangePlace` (MatrixSide.cpp:55) — 2s cooldown after a failed
/// place search.
pub fn can_change_place(now: i32, r: &Robot) -> bool {
    now - r.env.place_not_found > 2000
}

/// Road-network place position (upper-left cell), or `None`.
pub fn place_pos(map: &GameMap, no: i32) -> Option<(i32, i32)> {
    if no < 0 {
        return None;
    }
    let rn = map.road_network.as_ref()?.lock().unwrap();
    let p = rn.places.get(no as usize)?;
    Some((p.pos.x, p.pos.y))
}

/// `PLPlacePos` (MatrixSide.cpp:9541-9548).
pub fn pl_place_pos(map: &GameMap, r: &Robot) -> (i32, i32) {
    if r.env.place >= 0 {
        if let Some(p) = place_pos(map, r.env.place) {
            return p;
        }
    }
    if r.env.place_add.0 >= 0 {
        return r.env.place_add;
    }
    (-1, -1)
}

/// `PLIsToPlace` (MatrixSide.cpp:9519-9539).
pub fn pl_is_to_place(map: &GameMap, r: &Robot) -> bool {
    let ptp = pl_place_pos(map, r);
    if ptp.0 < 0 {
        return false;
    }
    if let Some(tp) = r.move_to_coords() {
        if r.orders.has(OrderType::CaptureFactory) {
            return false;
        }
        if ptp == tp {
            return true;
        }
        if let Some(rt) = r.return_coords() {
            if ptp == rt {
                return true;
            }
        }
        false
    } else {
        if let Some(rt) = r.return_coords() {
            if ptp == rt {
                return true;
            }
        }
        (r.map_x, r.map_y) == ptp
    }
}

/// `CMatrixRobotAI::PLIsInPlace` (MatrixRobot.hpp:528-542).
pub fn pl_is_in_place(map: &GameMap, r: &Robot) -> bool {
    if r.orders.has(OrderType::MoveTo) || r.orders.has(OrderType::MoveReturn) {
        return false;
    }
    let ptp = pl_place_pos(map, r);
    if ptp.0 < 0 {
        return false;
    }
    (r.map_x, r.map_y) == ptp
}

/// `IsInPlace(robot)` (MatrixSide.cpp:9408-9422) — standing exactly on
/// `env.place` with no MOVE_TO in flight.
pub fn is_in_place(map: &GameMap, r: &Robot) -> bool {
    let Some(p) = place_pos(map, r.env.place) else {
        return false;
    };
    if r.orders.has(OrderType::MoveTo) {
        return false;
    }
    (r.map_x, r.map_y) == p
}

/// Place `m_Data` write (occupancy scratch).
fn place_set_data(map: &GameMap, no: i32, v: u32) {
    if no < 0 {
        return;
    }
    if let Some(rn) = map.road_network.as_ref() {
        let mut rn = rn.lock().unwrap();
        if let Some(p) = rn.places.get_mut(no as usize) {
            p.data = v;
        }
    }
}

fn place_get(map: &GameMap, no: i32) -> Option<(u32, u8, (i32, i32), u8)> {
    if no < 0 {
        return None;
    }
    let rn = map.road_network.as_ref()?.lock().unwrap();
    let p = rn.places.get(no as usize)?;
    Some((p.data, p.move_mask, (p.pos.x, p.pos.y), p.underfire))
}

/// `ObjPlaceData(obj, 1)` for every live unit (the "mark occupied
/// places" loop repeated all over WarPL / RepairPL / PGAssignPlace).
/// Robots claim `env.place`; live cannons claim their `place` (set
/// from `FindInPL` at build/load, MatrixSide.cpp:9436-9461).
fn mark_occupied_places(map: &GameMap, objs: &Objects) {
    mark_occupied_places_skip(map, objs, None)
}

/// `mark_occupied_places` with the `obj != robot` exemption the C++
/// per-robot AssignPlace uses (MatrixSide.cpp:5284-5288).
fn mark_occupied_places_skip(map: &GameMap, objs: &Objects, skip: Option<ObjectId>) {
    for id in objs.iter_live() {
        if Some(id) == skip || !is_live_unit(objs, id) {
            continue;
        }
        if let Some(r) = robot_ref(objs, id) {
            place_set_data(map, r.env.place, 1);
        } else if let Some(c) = crate::matrix_game::logic::cannon_ref(objs, id) {
            place_set_data(map, c.place, 1);
        }
    }
}

/// Zero `m_Data` for every place in `list`.
fn zero_places(map: &GameMap, list: &[i32]) {
    if let Some(rn) = map.road_network.as_ref() {
        let mut rn = rn.lock().unwrap();
        for &i in list {
            if let Some(p) = rn.places.get_mut(i as usize) {
                p.data = 0;
            }
        }
    }
}

/// Chassis mask bit — `1 << (m_Unit[0].m_Kind - 1)`.
fn chassis_bit(r: &Robot) -> u8 {
    1u8 << r.chassis.kind_index()
}

impl MapLogic {
    fn queue_sound(&mut self, s: GameSound) {
        self.sound_queue.push(s);
    }

    /// `IRND(2)`-style in-progress order acknowledgement
    /// (MatrixSide.cpp:7960-7964 pattern).
    fn sound_order_in_progress(&mut self) {
        let s = if self.rng.range(0, 1) == 0 {
            GameSound::OrderInProgress1
        } else {
            GameSound::OrderInProgress2
        };
        self.queue_sound(s);
    }

    /// Live player robots of logic group `no` (helper for the many
    /// `GetFirstLogic … GetGroupLogic()==no` walks).
    fn group_robots(&self, no: usize) -> Vec<ObjectId> {
        let side_id = self.player_side.id;
        self.objects
            .iter_live()
            .filter(|&id| {
                robot_ref(&self.objects, id)
                    .map(|r| r.is_live() && r.side == side_id && r.group_logic == no as i32)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// All live player robots.
    fn side_robots(&self) -> Vec<ObjectId> {
        let side_id = self.player_side.id;
        self.objects
            .iter_live()
            .filter(|&id| {
                robot_ref(&self.objects, id)
                    .map(|r| r.is_live() && r.side == side_id)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Port of the underfire recalculation at MatrixSide.cpp:6054-6107
    /// — every place's `m_Underfire` counts enemy units whose max
    /// (and, doubled, min) fire range covers it.
    fn underfire_calc(&mut self, map: &GameMap) {
        let Some(rn_lock) = map.road_network.as_ref() else {
            return;
        };
        let side_id = self.player_side.id;

        // Collect enemy fire envelopes first to keep the rn lock scope
        // clean of arena reads.
        struct Envelope {
            tp: (i32, i32),
            firedist: i32,
            firedist2: i32,
        }
        let mut envs: Vec<Envelope> = Vec::new();
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        for id in self.objects.iter_live() {
            if !is_live_unit(&self.objects, id) {
                continue;
            }
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            if obj.side() == side_id {
                continue;
            }
            let Some(tp) = get_map_pos(&self.objects, id) else {
                continue;
            };
            let (fd, fd2) = match obj.core().obj_type {
                ObjectType::RobotAi => {
                    let r = robot_ref(&self.objects, id).unwrap();
                    (
                        float2int(r.max_fire_dist + gsm),
                        float2int(r.min_fire_dist + gsm),
                    )
                }
                ObjectType::Cannon => {
                    let c = cannon_ref(&self.objects, id).unwrap();
                    let f = float2int(c.fire_radius(&self.objects) + gsm);
                    (f, f)
                }
                _ => continue,
            };
            envs.push(Envelope {
                tp,
                firedist: fd / gsm as i32,
                firedist2: fd2 / gsm as i32,
            });
        }

        let mut rn = rn_lock.lock().unwrap();
        for p in rn.places.iter_mut() {
            p.underfire = 0;
        }
        for e in envs {
            let rect = crate::matrix_game::road_network::Rect {
                left: e.tp.0 - e.firedist,
                top: e.tp.1 - e.firedist,
                right: e.tp.0 + ROBOT_MOVECELLS_PER_SIZE + e.firedist,
                bottom: e.tp.1 + ROBOT_MOVECELLS_PER_SIZE + e.firedist,
            };
            let cx = e.tp.0 + (ROBOT_MOVECELLS_PER_SIZE >> 1);
            let cy = e.tp.1 + (ROBOT_MOVECELLS_PER_SIZE >> 1);
            let fd_sq = (e.firedist as i64) * (e.firedist as i64);
            let fd2_sq = (e.firedist2 as i64) * (e.firedist2 as i64);
            let plr = rn.correct_rect_pl(&rect);
            for y in plr.top..plr.bottom {
                for x in plr.left..plr.right {
                    let plist = rn.pl_list[(x + y * rn.pl_size_x) as usize];
                    for u in 0..plist.cnt {
                        let pi = (plist.sme + u) as usize;
                        let place = &mut rn.places[pi];
                        let pcx = place.pos.x + ROBOT_MOVECELLS_PER_SIZE / 2;
                        let pcy = place.pos.y + ROBOT_MOVECELLS_PER_SIZE / 2;
                        let d = ((cx - pcx) as i64).pow(2) + ((cy - pcy) as i64).pow(2);
                        if fd_sq >= d {
                            place.underfire += 1;
                        }
                        if fd2_sq >= d {
                            place.underfire += 1;
                        }
                    }
                }
            }
        }
    }

    /// Port of `CMatrixSideUnit::TaktPL` (MatrixSide.cpp:6033-6829).
    /// `onlygroup < 0` services every group.
    /// Port of `CMatrixSideUnit::EscapeFromBomb` (MatrixSide.cpp:2090-2255).
    /// Player-side automation: auto-ordered robots retreat from nearby
    /// bomb-carrying robots (friendly or enemy) so they aren't caught in
    /// the blast; robots covering a friendly bomb-carrier stay put.
    fn escape_from_bomb(&mut self, map: &GameMap) {
        use crate::matrix_game::common::float2int;
        use crate::matrix_game::road_network::Point;
        use crate::matrix_game::side::PlayerOrder;

        const ESCAPE_RADIUS: f32 = 250.0 * 1.2;
        let esc_r2 = ESCAPE_RADIUS * ESCAPE_RADIUS;
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let escape_dist = float2int(ESCAPE_RADIUS / gsm) + 4;
        let escape_dist2 = (escape_dist as i64) * (escape_dist as i64);
        let my_id = self.player_side.id;
        let auto_cap = PlayerOrder::AutoCapture as u8;
        let is_manual = |gl: i32, ps: &crate::matrix_game::side::Side| -> bool {
            gl >= 0 && (ps.player_groups[gl as usize].order as u8) < auto_cap
        };

        // Pass 1: bomb robots (any side) with enemies nearby.
        // (id, world_pos, side, min_enemy_dist2, map_cell)
        let mut rb: Vec<(ObjectId, glam::Vec2, i32, f32, (i32, i32))> = Vec::new();
        for id in self.objects.iter_live() {
            let Some(r) = robot_ref(&self.objects, id) else {
                continue;
            };
            if !r.is_live() || !r.have_bomb(&self.objects) || r.env.enemy_cnt() <= 0 {
                continue;
            }
            let Some(mp) = get_world_pos(&self.objects, id) else {
                continue;
            };
            let mut mde = 1e20f32;
            for e in &r.env.enemies {
                if let Some(ep) = get_world_pos(&self.objects, e.enemy) {
                    mde = mde.min((ep - mp).length_squared());
                }
            }
            rb.push((id, mp, r.side(), mde, (r.map_x, r.map_y)));
        }
        if rb.is_empty() {
            return;
        }

        let mine: Vec<ObjectId> = self
            .objects
            .iter_live()
            .filter(|&id| {
                robot_ref(&self.objects, id)
                    .map(|r| r.is_live() && r.side() == my_id)
                    .unwrap_or(false)
            })
            .collect();

        // Pass 2: the covering robot nearest an ENEMY bomb-carrier is exempt.
        let mut skip_normal: Option<ObjectId> = None;
        let mut skip_withbomb: Option<ObjectId> = None;
        let (mut sn_dist, mut sw_dist) = (0.0f32, 0.0f32);
        for &id in &mine {
            let (gl, has_bomb) = {
                let r = robot_ref(&self.objects, id).unwrap();
                (r.group_logic, r.have_bomb(&self.objects))
            };
            if is_manual(gl, &self.player_side) {
                continue;
            }
            let Some(rp) = get_world_pos(&self.objects, id) else {
                continue;
            };
            let mut mdist = 1e20f32;
            for &(bid, bp, bside, _, _) in &rb {
                if bid == id || bside == my_id {
                    continue;
                }
                let d = (rp - bp).length_squared();
                if d < esc_r2 {
                    mdist = mdist.min(d);
                }
            }
            if mdist > 1e19 {
                continue;
            }
            if !has_bomb {
                if skip_normal.is_none() || mdist < sn_dist {
                    skip_normal = Some(id);
                    sn_dist = mdist;
                }
            } else if skip_withbomb.is_none() || mdist < sw_dist {
                skip_withbomb = Some(id);
                sw_dist = mdist;
            }
        }
        if skip_withbomb.is_some() {
            skip_normal = None;
        }

        // Pass 3: retreat via a place-graph wave to a spot far from every
        // bomb robot. One road-network lock guards the whole pass.
        let Some(rn_lock) = map.road_network.as_ref() else {
            return;
        };
        let mut rn = rn_lock.lock().unwrap();
        for &id in &mine {
            if Some(id) == skip_normal || Some(id) == skip_withbomb {
                continue;
            }
            let (gl, has_return, mm, mapx, mapy, mde, rp) = {
                let r = robot_ref(&self.objects, id).unwrap();
                let Some(rp) = get_world_pos(&self.objects, id) else {
                    continue;
                };
                let mut mde = 1e20f32;
                for e in &r.env.enemies {
                    if let Some(ep) = get_world_pos(&self.objects, e.enemy) {
                        mde = mde.min((ep - rp).length_squared());
                    }
                }
                (
                    r.group_logic,
                    r.return_coords().is_some(),
                    chassis_bit(r),
                    r.map_x,
                    r.map_y,
                    mde,
                    rp,
                )
            };
            if is_manual(gl, &self.player_side) || has_return {
                continue;
            }
            // Near a bomb robot we don't cover?
            let mut near = false;
            for &(bid, bp, bside, min_de, _) in &rb {
                if bid == id {
                    continue;
                }
                if bside == my_id && mde < min_de {
                    continue; // covering the friendly bomb — hold position
                }
                if (rp - bp).length_squared() < esc_r2 {
                    near = true;
                    break;
                }
            }
            if !near {
                continue;
            }
            let start = crate::matrix_game::logic::find_near_place_impl(&rn, map, mm, (mapx, mapy));
            if start < 0 {
                continue;
            }

            // Mark places already claimed by other units (data|2) so the
            // wave doesn't send everyone to the same spot.
            let mut touched: Vec<i32> = Vec::new();
            for oid in self.objects.iter_live() {
                if oid == id {
                    continue;
                }
                if let Some(orr) = robot_ref(&self.objects, oid) {
                    if !orr.is_live() {
                        continue;
                    }
                    if let Some((tx, ty)) = orr.move_to_coords() {
                        // C++ marks via FindPlace (MatrixLogic.cpp:2127-
                        // 2142) whose bounds check reads `mappos.x <
                        // (tp.y+4)` — a typo that makes the lookup fail
                        // for most cells. Kept bit-faithful so escaping
                        // robots pick the same retreat spots as the
                        // original (a correct find_in_pl excludes more
                        // candidates and retreats farther).
                        let np = {
                            let zone = map.move_cell(tx, ty).map(|c| c.zone).unwrap_or(-1);
                            if zone < 0 {
                                -1
                            } else {
                                rn.zones
                                    .get(zone as usize)
                                    .and_then(|z| {
                                        z.place.iter().copied().find(|&pl| {
                                            rn.places
                                                .get(pl as usize)
                                                .map(|p| {
                                                    tx >= p.pos.x
                                                        && tx < (p.pos.y + 4)
                                                        && ty >= p.pos.y
                                                        && ty < (p.pos.y + 4)
                                                })
                                                .unwrap_or(false)
                                        })
                                    })
                                    .unwrap_or(-1)
                            }
                        };
                        if np >= 0 {
                            rn.places[np as usize].data |= 2;
                            touched.push(np);
                        }
                    }
                    let ep = orr.env.place;
                    if ep >= 0 && (ep as usize) < rn.places.len() {
                        rn.places[ep as usize].data |= 2;
                        touched.push(ep);
                    }
                } else if let Some(c) = crate::matrix_game::logic::cannon_ref(&self.objects, oid) {
                    let np = rn.find_in_pl(&Point {
                        x: float2int(c.pos.x / gsm),
                        y: float2int(c.pos.y / gsm),
                    });
                    if np >= 0 {
                        rn.places[np as usize].data |= 2;
                        touched.push(np);
                    }
                }
            }

            // Breadth-first wave from `start` over unblocked near-links.
            let sme_start = touched.len();
            rn.places[start as usize].data |= 1;
            touched.push(start);
            let mut sme = sme_start;
            let mut chosen: Option<(i32, i32)> = None;
            while sme < touched.len() {
                let rp_i = touched[sme];
                sme += 1;
                let place_pos = rn.places[rp_i as usize].pos;
                if rn.places[rp_i as usize].data & 2 == 0 {
                    let far = rb.iter().all(|&(bid, _, _, _, (bx, by))| {
                        if bid == id {
                            return true;
                        }
                        let dx = (place_pos.x - bx) as i64;
                        let dy = (place_pos.y - by) as i64;
                        dx * dx + dy * dy >= escape_dist2
                    });
                    if far {
                        chosen = Some((place_pos.x, place_pos.y));
                        break;
                    }
                }
                let near_cnt = rn.places[rp_i as usize].near.len();
                for k in 0..near_cnt {
                    let np = rn.places[rp_i as usize].near[k];
                    if np < 0 || (np as usize) >= rn.places.len() {
                        continue;
                    }
                    if rn.places[np as usize].data & 1 != 0 {
                        continue;
                    }
                    if rn.places[rp_i as usize].near_move[k] & mm != 0 {
                        continue; // link blocked for this chassis
                    }
                    rn.places[np as usize].data |= 1;
                    touched.push(np);
                }
            }
            for &t in &touched {
                rn.places[t as usize].data = 0;
            }

            if let Some((tx, ty)) = chosen {
                if let Some(r) = robot_mut(&mut self.objects, id) {
                    if let Some((rx, ry)) = r.move_to_coords() {
                        r.move_return(rx, ry);
                    } else {
                        r.move_return(r.map_x, r.map_y);
                    }
                    r.move_to(tx, ty);
                }
            }
        }
    }

    pub fn takt_pl(&mut self, map: &GameMap, onlygroup: i32) {
        let now = self.elapsed_ms as i32;

        // Once per 100ms (MatrixSide.cpp:6046).
        if self.player_side.last_takt_pl != 0 && now - self.player_side.last_takt_pl < 100 {
            return;
        }
        self.player_side.last_takt_pl = now;

        // Player robots dodge bomb blasts (MatrixSide.cpp:6051). CalcStrength
        // is refreshed per-frame in GatherInfo instead.
        self.escape_from_bomb(map);

        if self.player_side.last_takt_underfire == 0
            || now - self.player_side.last_takt_underfire > 500
        {
            self.player_side.last_takt_underfire = now;
            self.underfire_calc(map);
        }

        let side_id = self.player_side.id;

        // Robot-count census + no-break release (MatrixSide.cpp:6110-6127).
        for pg in self.player_side.player_groups.iter_mut() {
            pg.robot_cnt = 0;
        }
        let robots = self.side_robots();
        for &rid in &robots {
            let (can_break, group_logic, no_break) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (r.can_break_order(), r.group_logic, r.env.order_no_break)
            };
            if no_break && can_break {
                let clear_place = group_logic < 0
                    || self.player_side.player_groups[group_logic as usize].order
                        != PlayerOrder::MoveTo;
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.env.order_no_break = false;
                    if clear_place {
                        r.env.place = -1;
                        r.env.place_add = (-1, -1);
                    }
                }
            }
            if (0..MAX_LOGIC_GROUP as i32).contains(&group_logic) {
                self.player_side.player_groups[group_logic as usize].robot_cnt += 1;
            }
        }

        // Validate group targets (MatrixSide.cpp:6130-6159).
        for i in 0..MAX_LOGIC_GROUP {
            if self.player_side.player_groups[i].robot_cnt <= 0 {
                continue;
            }
            let Some(obj) = self.player_side.player_groups[i].obj else {
                continue;
            };
            let alive = self
                .objects
                .get(obj)
                .map(|o| o.is_live())
                .unwrap_or(false);
            if !alive {
                self.player_side.player_groups[i].obj = None;
                continue;
            }
            // Близко к цели атаки → заносим во врагов.
            let order = self.player_side.player_groups[i].order;
            if matches!(
                order,
                PlayerOrder::Attack | PlayerOrder::AutoAttack | PlayerOrder::AutoDefence
            ) {
                if let Some(tp) = get_map_pos(&self.objects, obj) {
                    for &rid in &self.group_robots(i) {
                        let Some(r) = robot_ref(&self.objects, rid) else {
                            continue;
                        };
                        let rp = (r.map_x, r.map_y);
                        let d2 = ((tp.0 - rp.0) as i64).pow(2) + ((tp.1 - rp.1) as i64).pow(2);
                        if d2 < 30 * 30 && !r.env.search_enemy(obj) {
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                r.env.add_to_list(obj);
                            }
                        }
                    }
                }
            }
        }

        // Combat sub-takts (MatrixSide.cpp:6161-6166).
        for i in 0..MAX_LOGIC_GROUP {
            if self.player_side.player_groups[i].robot_cnt <= 0 {
                continue;
            }
            if self.player_side.player_groups[i].order == PlayerOrder::Repair {
                self.repair_pl(map, i);
            } else if self.player_side.player_groups[i].war {
                self.war_pl(map, i);
            } else if !self.fire_pl(map, i) {
                self.repair_pl(map, i);
            }
        }

        // Order-success checks (MatrixSide.cpp:6169-6562).
        let mut orderok = [true; MAX_LOGIC_GROUP];
        for i in 0..MAX_LOGIC_GROUP {
            if onlygroup >= 0 && i as i32 != onlygroup {
                continue;
            }
            if self.player_side.player_groups[i].robot_cnt <= 0 {
                continue;
            }
            orderok[i] = true;
            let prevwar = self.player_side.player_groups[i].war;
            self.player_side.player_groups[i].war = false;
            let order = self.player_side.player_groups[i].order;
            let grp = self.group_robots(i);

            match order {
                PlayerOrder::Stop => {
                    for &rid in &grp {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        if !pl_is_to_place(map, r) {
                            orderok[i] = false;
                            break;
                        }
                    }
                }
                PlayerOrder::MoveTo => {
                    for &rid in &grp {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        if !pl_is_to_place(map, r) && can_change_place(now, r) {
                            orderok[i] = false;
                            break;
                        }
                    }
                }
                PlayerOrder::Patrol => {
                    let mut t = true;
                    for &rid in &grp {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        if r.env.enemy_cnt() > 0 {
                            orderok[i] = true;
                            self.player_side.player_groups[i].war = true;
                            if !prevwar {
                                self.pg_place_clear(i);
                            }
                            t = false;
                            break;
                        }
                        if !pl_is_to_place(map, r) && can_change_place(now, r) {
                            orderok[i] = false;
                            break;
                        }
                        if !pl_is_in_place(map, r) {
                            t = false;
                        }
                    }
                    if t {
                        orderok[i] = false;
                        let pr = self.player_side.player_groups[i].patrol_return;
                        self.player_side.player_groups[i].patrol_return = !pr;
                    }
                }
                PlayerOrder::Repair => {
                    let need = self.player_side.player_groups[i]
                        .obj
                        .and_then(|o| self.objects.get(o))
                        .map(|o| o.need_repair())
                        .unwrap_or(false);
                    if !need {
                        self.pg_order_stop(map, i);
                    }
                }
                PlayerOrder::Capture => {
                    if !self.takt_capture_check(map, i, prevwar, &mut orderok, now, false) {
                        continue;
                    }
                }
                PlayerOrder::Attack => {
                    for &rid in &grp {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        let target = self.player_side.player_groups[i].obj;
                        if let Some(t) = target {
                            if r.env.search_enemy(t) {
                                orderok[i] = true;
                                self.player_side.player_groups[i].war = true;
                                if !prevwar {
                                    self.pg_place_clear(i);
                                }
                                break;
                            }
                        } else if r.env.enemy_cnt() > 0 {
                            orderok[i] = true;
                            self.player_side.player_groups[i].war = true;
                            if !prevwar {
                                self.pg_place_clear(i);
                            }
                            break;
                        }
                        if !pl_is_to_place(map, r) && can_change_place(now, r) {
                            orderok[i] = false;
                            break;
                        }
                    }
                    if !self.player_side.player_groups[i].war && prevwar {
                        orderok[i] = false;
                        continue;
                    }
                    if !self.player_side.player_groups[i].war && orderok[i] {
                        // Has the target moved away from the places?
                        let tp = self.player_side.player_groups[i]
                            .obj
                            .and_then(|o| get_map_pos(&self.objects, o))
                            .unwrap_or(self.player_side.player_groups[i].to);
                        let mut near = false;
                        for &rid in &grp {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            let pp = pl_place_pos(map, r);
                            let d2 = ((tp.0 - pp.0) as i64).pow(2) + ((tp.1 - pp.1) as i64).pow(2);
                            if d2 < 15 * 15 {
                                near = true;
                                break;
                            }
                        }
                        if !near {
                            orderok[i] = false;
                            continue;
                        }
                    }
                }
                PlayerOrder::Bomb => {
                    let mut t = 0;
                    for &rid in &grp {
                        let (has_bomb, in_place, place_d2, near_obj) = {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            let pp = pl_place_pos(map, r);
                            let pd2 = if pp.0 >= 0 {
                                ((pp.0 - r.map_x) as i64).pow(2) + ((pp.1 - r.map_y) as i64).pow(2)
                            } else {
                                i64::MAX
                            };
                            let near_obj = self.player_side.player_groups[i]
                                .obj
                                .filter(|&o| {
                                    self.objects.get(o).map(|x| x.is_live()).unwrap_or(false)
                                })
                                .and_then(|o| get_world_pos(&self.objects, o))
                                .map(|wp| {
                                    let rp =
                                        get_world_pos(&self.objects, rid).unwrap_or_default();
                                    (wp - rp).length_squared() < 150.0 * 150.0
                                })
                                .unwrap_or(false);
                            (
                                r.have_bomb(&self.objects),
                                pl_is_in_place(map, r),
                                pd2,
                                near_obj,
                            )
                        };
                        if has_bomb {
                            t += 1;
                            if in_place || place_d2 < 2 * 2 || near_obj {
                                self.robot_big_boom(map, rid);
                                t -= 1;
                            }
                        }
                        let r = robot_ref(&self.objects, rid).unwrap();
                        if !pl_is_to_place(map, r) && can_change_place(now, r) {
                            orderok[i] = false;
                        }
                    }
                    if t <= 0 {
                        self.pg_order_stop(map, i);
                        continue;
                    }
                    if orderok[i] {
                        let tp = self.player_side.player_groups[i]
                            .obj
                            .and_then(|o| get_map_pos(&self.objects, o))
                            .unwrap_or(self.player_side.player_groups[i].to);
                        let mut near = false;
                        for &rid in &grp {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            let pp = pl_place_pos(map, r);
                            let d2 = ((tp.0 - pp.0) as i64).pow(2) + ((tp.1 - pp.1) as i64).pow(2);
                            if d2 < 10 * 10 {
                                near = true;
                                break;
                            }
                        }
                        if !near {
                            orderok[i] = false;
                            continue;
                        }
                    }
                }
                PlayerOrder::AutoCapture => {
                    let mut strange = 0.0f32;
                    for &rid in &grp {
                        strange += robot_ref(&self.objects, rid).unwrap().strength;
                    }
                    for &rid in &grp {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        if r.env.enemy_cnt() > 0 {
                            orderok[i] = true;
                            if strange >= 1.0 {
                                self.player_side.player_groups[i].war = true;
                            }
                            if !prevwar {
                                self.pg_place_clear(i);
                            }
                            break;
                        }
                    }
                    if self.player_side.player_groups[i].war {
                        continue;
                    }
                    let obj_ok = self.player_side.player_groups[i]
                        .obj
                        .and_then(|o| building_ref(&self.objects, o))
                        .map(|b| b.is_live() && b.side != side_id)
                        .unwrap_or(false);
                    if !obj_ok {
                        self.player_side.player_groups[i].obj = None;
                        orderok[i] = false;
                        continue;
                    }
                    if !self.takt_capture_check(map, i, prevwar, &mut orderok, now, true) {
                        continue;
                    }
                }
                PlayerOrder::AutoAttack => {
                    for &rid in &grp {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        if r.env.enemy_cnt() > 0 {
                            orderok[i] = true;
                            self.player_side.player_groups[i].war = true;
                            if !prevwar {
                                self.pg_place_clear(i);
                            }
                            break;
                        }
                        if !pl_is_to_place(map, r) && can_change_place(now, r) {
                            orderok[i] = false;
                            break;
                        }
                    }
                    if self.player_side.player_groups[i].war {
                        continue;
                    }
                    let alive = self.player_side.player_groups[i]
                        .obj
                        .and_then(|o| self.objects.get(o))
                        .map(|o| o.is_live())
                        .unwrap_or(false);
                    if !alive {
                        self.player_side.player_groups[i].obj = None;
                        orderok[i] = false;
                        continue;
                    }
                    if orderok[i] {
                        let tp = self.player_side.player_groups[i]
                            .obj
                            .and_then(|o| get_map_pos(&self.objects, o))
                            .unwrap_or(self.player_side.player_groups[i].to);
                        let mut near = false;
                        for &rid in &grp {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            let pp = pl_place_pos(map, r);
                            let d2 = ((tp.0 - pp.0) as i64).pow(2) + ((tp.1 - pp.1) as i64).pow(2);
                            if d2 < 15 * 15 {
                                near = true;
                                break;
                            }
                        }
                        if !near {
                            orderok[i] = false;
                            continue;
                        }
                    }
                }
                PlayerOrder::AutoDefence => {
                    let mut last_robot = None;
                    for &rid in &grp {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        last_robot = Some(rid);
                        if r.env.enemy_cnt() > 0 {
                            orderok[i] = true;
                            self.player_side.player_groups[i].war = true;
                            if !prevwar {
                                self.pg_place_clear(i);
                            }
                            break;
                        }
                        if !pl_is_to_place(map, r) {
                            orderok[i] = false;
                            break;
                        }
                    }
                    if self.player_side.player_groups[i].war {
                        continue;
                    }
                    let robot_target_ok = self.player_side.player_groups[i]
                        .obj
                        .and_then(|o| robot_ref(&self.objects, o))
                        .map(|r| r.is_live())
                        .unwrap_or(false);
                    if !robot_target_ok || self.player_side.player_groups[i].region < 0 {
                        self.player_side.player_groups[i].obj = None;
                        orderok[i] = false;
                        continue;
                    }
                    if orderok[i] {
                        // Is the target still heading into our region?
                        let tobj = self.player_side.player_groups[i].obj.unwrap();
                        let tp = robot_ref(&self.objects, tobj)
                            .and_then(|r| r.move_to_coords())
                            .or_else(|| get_map_pos(&self.objects, tobj))
                            .unwrap_or((-1, -1));
                        let reg = get_region(map, tp.0, tp.1);
                        if self.player_side.player_groups[i].region != reg {
                            let can = last_robot
                                .and_then(|rid| robot_ref(&self.objects, rid))
                                .map(|r| can_change_place(now, r))
                                .unwrap_or(true);
                            if can {
                                self.player_side.player_groups[i].obj = None;
                                orderok[i] = false;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Apply orders (MatrixSide.cpp:6565-6803).
        for i in 0..MAX_LOGIC_GROUP {
            if onlygroup >= 0 && i as i32 != onlygroup {
                continue;
            }
            if self.player_side.player_groups[i].robot_cnt <= 0 {
                continue;
            }
            if orderok[i] {
                continue;
            }
            let order = self.player_side.player_groups[i].order;
            match order {
                PlayerOrder::Stop | PlayerOrder::MoveTo => {
                    self.move_group_to_places(map, i);
                    if self.player_side.player_groups[i].show_place {
                        self.pg_show_place(map, i);
                    }
                }
                PlayerOrder::Patrol => {
                    let target = if !self.player_side.player_groups[i].patrol_return {
                        self.player_side.player_groups[i].from
                    } else {
                        self.player_side.player_groups[i].to
                    };
                    self.pg_assign_place(map, i, target);
                    self.move_group_to_places(map, i);
                    if self.player_side.player_groups[i].show_place {
                        self.pg_show_place(map, i);
                    }
                }
                PlayerOrder::Capture | PlayerOrder::AutoCapture => {
                    if order == PlayerOrder::AutoCapture {
                        if self.player_side.player_groups[i].obj.is_none() {
                            self.pg_find_capture_factory(map, i);
                        }
                        if self.player_side.player_groups[i].obj.is_none() {
                            self.pg_order_stop(map, i);
                            continue;
                        }
                    }
                    self.apply_capture_order(map, i);
                    if order == PlayerOrder::Capture
                        && self.player_side.player_groups[i].show_place
                    {
                        self.pg_show_place(map, i);
                    }
                }
                PlayerOrder::Attack => {
                    let center = self.player_side.player_groups[i]
                        .obj
                        .and_then(|o| get_map_pos(&self.objects, o))
                        .unwrap_or(self.player_side.player_groups[i].to);
                    self.pg_assign_place_player(map, i, center);
                    self.move_group_to_places(map, i);
                    if self.player_side.player_groups[i].show_place {
                        self.pg_show_place(map, i);
                    }
                }
                PlayerOrder::Bomb => {
                    let center = self.player_side.player_groups[i]
                        .obj
                        .and_then(|o| get_map_pos(&self.objects, o))
                        .unwrap_or(self.player_side.player_groups[i].to);
                    self.pg_assign_place_player(map, i, center);
                    self.move_group_to_places(map, i);
                    if self.player_side.player_groups[i].show_place {
                        self.pg_show_place(map, i);
                    }
                }
                PlayerOrder::AutoAttack => {
                    if self.player_side.player_groups[i].obj.is_none() {
                        self.pg_find_attack_target(map, i);
                    }
                    if self.player_side.player_groups[i].obj.is_none() {
                        continue;
                    }
                    let center = self.player_side.player_groups[i]
                        .obj
                        .and_then(|o| get_map_pos(&self.objects, o))
                        .unwrap_or(self.player_side.player_groups[i].to);
                    self.pg_assign_place_player(map, i, center);
                    self.move_group_to_places(map, i);
                }
                PlayerOrder::AutoDefence => {
                    if self.player_side.player_groups[i].obj.is_none() {
                        self.pg_find_defence_target(map, i);
                    }
                    if self.player_side.player_groups[i].obj.is_none()
                        || self.player_side.player_groups[i].region < 0
                    {
                        continue;
                    }
                    let center = {
                        let reg = self.player_side.player_groups[i].region;
                        map.road_network
                            .as_ref()
                            .map(|rn| {
                                let rn = rn.lock().unwrap();
                                let r = rn.get_region(reg);
                                (r.center.x, r.center.y)
                            })
                            .unwrap_or((-1, -1))
                    };
                    if center.0 >= 0 {
                        self.pg_assign_place(map, i, center);
                    }
                    self.move_group_to_places(map, i);
                }
                PlayerOrder::Repair => {}
            }
        }
    }

    /// The repeated "walk group, MoveToHigh everyone to their place"
    /// loop (e.g. MatrixSide.cpp:6571-6580).
    fn move_group_to_places(&mut self, map: &GameMap, no: usize) {
        for &rid in &self.group_robots(no) {
            let (tp, disable_manual) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (pl_place_pos(map, r), r.is_disable_manual())
            };
            if disable_manual || tp.0 < 0 {
                continue;
            }
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                if prepare_break_order(r) {
                    r.move_to_high(tp.0, tp.1);
                }
            }
        }
    }

    /// Shared Capture / AutoCapture order-ok logic
    /// (MatrixSide.cpp:6240-6317 and 6442-6476). Returns false when
    /// the caller should `continue` to the next group.
    fn takt_capture_check(
        &mut self,
        map: &GameMap,
        i: usize,
        prevwar: bool,
        orderok: &mut [bool; MAX_LOGIC_GROUP],
        now: i32,
        auto: bool,
    ) -> bool {
        let side_id = self.player_side.id;
        let target = self.player_side.player_groups[i].obj;

        // Is anyone (side-wide) capturing the target? Is the target
        // still capturable?
        let mut someone_capturing = false;
        let mut capturable = false;
        for id in self.objects.iter_live() {
            if let Some(r) = robot_ref(&self.objects, id) {
                if r.is_live() && r.side == side_id {
                    if let Some(cf) = r.get_capture_factory() {
                        if Some(cf) == target {
                            someone_capturing = true;
                        }
                    }
                }
            }
            if Some(id) == target {
                if let Some(b) = building_ref(&self.objects, id) {
                    if b.is_live() && b.side != side_id {
                        capturable = true;
                    }
                }
            }
        }
        if !capturable {
            if auto {
                orderok[i] = false;
            } else {
                self.pg_order_stop(map, i);
            }
            return false;
        }
        orderok[i] = someone_capturing;

        // Base with turrets → fight the turrets first
        // (MatrixSide.cpp:6260-6301; the non-auto branch only).
        if !auto {
            let base_with_turrets = target
                .and_then(|t| building_ref(&self.objects, t))
                .map(|b| b.is_base() && b.turrets_have > 0)
                .unwrap_or(false);
            if base_with_turrets {
                let grp = self.group_robots(i);
                let mut engaged = false;
                'outer: for id in self.objects.iter_live() {
                    let Some(c) = cannon_ref(&self.objects, id) else {
                        continue;
                    };
                    if !matches!(c.state, CannonState::Dip | CannonState::UnderConstruction)
                        && c.parent == target
                    {
                        for &rid in &grp {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            if r.env.search_enemy(id) {
                                engaged = true;
                                break 'outer;
                            }
                        }
                    }
                }
                if engaged {
                    for &rid in &grp {
                        let (cf, can) = {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            (r.get_capture_factory(), r.can_break_order())
                        };
                        if cf.is_some() && can {
                            self.robot_break_all_orders(rid);
                        }
                    }
                    self.player_side.player_groups[i].war = true;
                    if self.enable_capture_fuckoff_sound {
                        self.queue_sound(GameSound::OrderCaptureFuckOff);
                        self.enable_capture_fuckoff_sound = false;
                    }
                    if !prevwar {
                        self.pg_place_clear(i);
                    }
                    orderok[i] = true;
                    return false;
                }
            }
        }

        if orderok[i] {
            // Everyone has a valid place? (MatrixSide.cpp:6302-6317)
            let grp = self.group_robots(i);
            for &rid in &grp {
                let r = robot_ref(&self.objects, rid).unwrap();
                if r.env.place < 0 && r.env.place_add.0 < 0 {
                    if can_change_place(now, r) {
                        orderok[i] = false;
                    }
                    break;
                }
            }
        }
        true
    }

    /// The Capture / AutoCapture order-apply body
    /// (MatrixSide.cpp:6611-6661 / 6701-6758): pick the closest
    /// breakable robot to run the capture, send the rest to places
    /// ringing the building.
    fn apply_capture_order(&mut self, map: &GameMap, i: usize) {
        let side_id = self.player_side.id;
        let Some(target) = self.player_side.player_groups[i].obj else {
            return;
        };

        // Already-capturing robot anywhere on the side?
        let mut capturer: Option<ObjectId> = None;
        for id in self.objects.iter_live() {
            let Some(r) = robot_ref(&self.objects, id) else {
                continue;
            };
            if r.is_live() && r.side == side_id && !r.is_disable_manual() {
                if r.get_capture_factory() == Some(target) {
                    capturer = Some(id);
                    break;
                }
            }
        }
        if capturer.is_none() {
            // Closest breakable group member.
            let Some(ttp) = get_map_pos(&self.objects, target) else {
                return;
            };
            let mut best = i64::MAX;
            for &rid in &self.group_robots(i) {
                let r = robot_ref(&self.objects, rid).unwrap();
                if r.is_disable_manual() || !r.can_break_order() {
                    continue;
                }
                let d2 = ((ttp.0 - r.map_x) as i64).pow(2) + ((ttp.1 - r.map_y) as i64).pow(2);
                if d2 < best {
                    best = d2;
                    capturer = Some(rid);
                }
            }
            if let Some(rid) = capturer {
                let ok = robot_mut(&mut self.objects, rid)
                    .map(prepare_break_order)
                    .unwrap_or(false);
                if ok {
                    self.sound_capture(i);
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.capture_factory(target);
                    }
                }
            }
        }
        if let Some(c) = get_map_pos(&self.objects, target) {
            self.pg_assign_place(map, i, c);
        }
        for &rid in &self.group_robots(i) {
            if Some(rid) == capturer {
                continue;
            }
            let (tp, disable_manual) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (pl_place_pos(map, r), r.is_disable_manual())
            };
            if disable_manual || tp.0 < 0 {
                continue;
            }
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                if prepare_break_order(r) {
                    r.move_to_high(tp.0, tp.1);
                }
            }
        }
    }

    /// `SoundCapture` (MatrixSide.cpp:8013-8031).
    fn sound_capture(&mut self, pg: usize) {
        if self.sound_order_capture_enabled {
            if self.player_side.player_groups[pg].order == PlayerOrder::Capture {
                self.sound_order_in_progress();
            } else {
                self.queue_sound(GameSound::OrderCapture);
            }
        }
        self.sound_order_capture_enabled = false;
    }

    /// `BigBoom` trigger through the arena (borrow helper — the
    /// take-the-box pattern of `proceed_logic`).
    fn robot_big_boom(&mut self, map: &GameMap, rid: ObjectId) {
        let Some(mut boxed) = self.objects.take_obj(rid) else {
            return;
        };
        if matches!(boxed.core().obj_type, ObjectType::RobotAi) {
            let r =
                unsafe { &mut *(boxed.as_mut() as *mut dyn MapStatic as *mut Robot) };
            r.big_boom(map, &mut self.objects, &mut self.rng);
        }
        self.objects.put_obj(rid, boxed);
    }

    /// `BreakAllOrders` through the arena (the captured-factory
    /// release inside needs `&mut Objects`).
    pub fn robot_break_all_orders(&mut self, rid: ObjectId) {
        let Some(mut boxed) = self.objects.take_obj(rid) else {
            return;
        };
        if matches!(boxed.core().obj_type, ObjectType::RobotAi) {
            let r =
                unsafe { &mut *(boxed.as_mut() as *mut dyn MapStatic as *mut Robot) };
            r.break_all_orders(&mut self.objects);
        }
        self.objects.put_obj(rid, boxed);
    }

    // ── Target acquisition shared by FirePL / WarPL ─────────────────

    /// The "nearest open enemy, else nearest covered enemy" scan
    /// (MatrixSide.cpp:7027-7086 / 7297-7358). Returns the chosen
    /// enemy, or `None`.
    fn pick_enemy_for(&self, map: &GameMap, rid: ObjectId, prefer: Option<ObjectId>) -> Option<ObjectId> {
        let side_id = self.player_side.id;
        let me = robot_ref(&self.objects, rid)?;
        let my_pos = get_world_pos(&self.objects, rid)?;
        let from = me.core().geo_center;

        // Player-designated target first (WarPL: SearchEnemy(m_Obj)).
        if let Some(p) = prefer {
            if p != rid && me.env.search_enemy(p) {
                return Some(p);
            }
        }

        let mut mindist = f32::MAX;
        let mut found: Option<ObjectId> = None;
        // Nearest enemy not covered by a friendly.
        for e in &me.env.enemies {
            let eid = e.enemy;
            if !is_live_unit(&self.objects, eid) || eid == rid {
                continue;
            }
            let Some(epos) = get_world_pos(&self.objects, eid) else {
                continue;
            };
            let cd = (epos - my_pos).length_squared();
            if cd >= mindist {
                continue;
            }
            let Some(des) = point_of_aim(map, &self.objects, eid) else {
                continue;
            };
            let dist = (des - from).length();
            if dist <= 0.0 {
                continue;
            }
            let dir = (des - from) / dist;
            if !self.friendly_in_fire_line(rid, eid, from, dir, dist) {
                mindist = cd;
                found = Some(eid);
            }
        }
        if found.is_none() {
            // Nearest covered enemy.
            for e in &me.env.enemies {
                let eid = e.enemy;
                if !is_live_unit(&self.objects, eid) || eid == rid {
                    continue;
                }
                let Some(epos) = get_world_pos(&self.objects, eid) else {
                    continue;
                };
                let cd = (epos - my_pos).length_squared();
                if cd < mindist {
                    mindist = cd;
                    found = Some(eid);
                }
            }
        }
        let _ = side_id;
        found
    }

    /// The `IsIntersectSphere` friendly-block scan
    /// (MatrixSide.cpp:7048-7058 / 7114-7132): true when one of our
    /// own live units sits within 25 units of the fire line.
    fn friendly_in_fire_line(
        &self,
        rid: ObjectId,
        target: ObjectId,
        from: glam::Vec3,
        dir: glam::Vec3,
        dist: f32,
    ) -> bool {
        let side_id = self.player_side.id;
        for oid in self.objects.iter_live() {
            if oid == rid || oid == target {
                continue;
            }
            if !is_live_unit(&self.objects, oid) {
                continue;
            }
            let Some(obj) = self.objects.get(oid) else {
                continue;
            };
            if obj.side() != side_id {
                continue;
            }
            let p = match point_of_aim_core(obj) {
                Some(p) => p,
                None => continue,
            };
            if let Some(t) = ray_sphere(p, 25.0, from, dir) {
                if t >= 0.0 && t < dist {
                    return true;
                }
            }
        }
        false
    }

    /// The shared "fire-line check + aim-protect twist + Fire/StopFire"
    /// tail of FirePL (MatrixSide.cpp:7093-7229) and WarPL
    /// (7682-7854). `war` enables the WarPL-only stuck-recovery rules.
    fn fire_correction(&mut self, map: &GameMap, group: usize, war: bool) -> bool {
        let mut rv = false;
        let now = self.elapsed_ms as i32;
        let side_id = self.player_side.id;

        for &rid in &self.group_robots(group) {
            let Some(target) = ({
                let r = robot_ref(&self.objects, rid).unwrap();
                let t = r.env.target_attack;
                // FirePL additionally requires the target to still be
                // in the enemy list (MatrixSide.cpp:7094).
                if war {
                    t
                } else {
                    t.filter(|&t| r.env.search_enemy(t))
                }
            }) else {
                continue;
            };

            let target_alive = self
                .objects
                .get(target)
                .map(|o| o.is_live())
                .unwrap_or(false);
            if !target_alive {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.stop_fire();
                }
                continue;
            }

            let mut des = point_of_aim(map, &self.objects, target).unwrap();
            let (from, have_repair, max_fd, my_pos) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (
                    r.core().geo_center,
                    r.have_repair,
                    r.max_fire_dist,
                    glam::Vec2::new(r.pos_x, r.pos_y),
                )
            };
            let dist = (des - from).length();
            let mut fireline = have_repair != 2;

            if fireline && dist > 0.0 {
                let dir = (des - from) / dist;
                if self.friendly_in_fire_line(rid, target, from, dir, dist) {
                    fireline = false;
                }
            }

            // m_TargetLast bookkeeping (FirePL:7139; WarPL:7734 also
            // stamps m_TargetChange).
            {
                let r = robot_mut(&mut self.objects, rid).unwrap();
                if r.env.target_attack != r.env.target_last {
                    r.env.target_last = r.env.target_attack;
                    if war {
                        r.env.target_change = now;
                    }
                }
            }

            if fireline {
                fireline = (des - from).length_squared() <= max_fd * max_fd;
                if fireline {
                    let (res, _) = trace(
                        map,
                        &self.objects,
                        from,
                        des,
                        crate::matrix_game::common::TRACE_OBJECT
                            | crate::matrix_game::common::TRACE_NONOBJECT,
                        Some(rid),
                    );
                    fireline = !(matches!(res, TraceStop::Water | TraceStop::Landscape)
                        || matches!(res, TraceStop::Object(h) if self
                            .objects
                            .get(h)
                            .map(|o| matches!(o.core().obj_type, ObjectType::MapObject))
                            .unwrap_or(false)));
                }
            }

            if fireline {
                rv = true;
                // Firewall-head aim protection (MatrixSide.cpp:7170-7191).
                let target_aim_protect = robot_ref(&self.objects, target)
                    .map(|t| t.aim_protect)
                    .unwrap_or(0.0);
                if target_aim_protect > 0.0 {
                    let rnd1 = self.rng.range(1, 100) as f32;
                    let flip = self.rng.range(0, 9) < 5;
                    let r = robot_mut(&mut self.objects, rid).unwrap();
                    let to_rad = std::f32::consts::PI / 180.0;
                    if r.env.target != r.env.target_attack
                        || r.env.target_angle.abs() <= 1.3 * to_rad
                    {
                        r.env.target_angle =
                            (8.0 * to_rad).min(rnd1 / 100.0 * 16.0 * target_aim_protect * to_rad);
                        if flip {
                            r.env.target_angle = -r.env.target_angle;
                        }
                    } else {
                        r.env.target_angle *= 0.75;
                    }
                    if r.env.target_angle != 0.0 {
                        let vx = des.x - r.pos_x;
                        let vy = des.y - r.pos_y;
                        let (sa, ca) = r.env.target_angle.sin_cos();
                        des.x = (ca * vx + sa * vy) + r.pos_x;
                        des.y = (-sa * vx + ca * vy) + r.pos_y;
                    }
                }

                let in_place = {
                    let r = robot_ref(&self.objects, rid).unwrap();
                    is_in_place(map, r)
                };
                let r = robot_mut(&mut self.objects, rid).unwrap();
                r.env.target = r.env.target_attack;
                if war {
                    r.env.last_fire = now;
                }
                r.fire(des, 0);

                if war {
                    // Stuck recovery (MatrixSide.cpp:7791-7807).
                    if in_place {
                        if r.env.enemy_cnt() > 1
                            && now - r.env.target_change > 4000
                            && now - r.env.last_hit_target > 4000
                        {
                            r.env.target_attack = None;
                        }
                        if r.env.enemy_cnt() == 1
                            && now - r.env.target_change > 4000
                            && now - r.env.last_hit_target > 4000
                        {
                            let p = r.env.place;
                            r.env.add_bad_place(p);
                            r.env.place = -1;
                        }
                        if now - r.env.target_change > 2000 && now - r.env.last_hit_target > 10000
                        {
                            let p = r.env.place;
                            r.env.add_bad_place(p);
                            r.env.place = -1;
                        }
                    } else {
                        r.env.target_change = now;
                    }
                }
            } else {
                // Repair-target search (MatrixSide.cpp:7196-7220 /
                // 7809-7833).
                let (have_rep, repair_dist, target_change_repair) = {
                    let r = robot_ref(&self.objects, rid).unwrap();
                    (r.have_repair, r.repair_dist, r.env.target_change_repair)
                };
                if have_rep != 0 && now - target_change_repair > 1000 {
                    let cur_ok = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        r.env
                            .target
                            .filter(|&t| {
                                // Both FirePL and WarPL gate on IsLiveUnit —
                                // buildings are never auto-repair targets
                                // (MatrixSide.cpp:7199,7812).
                                let live = is_live_unit(&self.objects, t);
                                live && self
                                    .objects
                                    .get(t)
                                    .map(|o| o.side() == side_id && o.need_repair())
                                    .unwrap_or(false)
                            })
                            .and_then(|t| get_world_pos(&self.objects, t))
                            .map(|tp| (tp - my_pos).length_squared() <= repair_dist * repair_dist)
                            .unwrap_or(false)
                    };
                    if !cur_ok {
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.env.target = None;
                        }
                        // Find a new repairable.
                        let mut new_target = None;
                        for oid in self.objects.iter_live() {
                            if oid == rid {
                                continue;
                            }
                            if !is_live_unit(&self.objects, oid) {
                                continue;
                            }
                            let Some(obj) = self.objects.get(oid) else {
                                continue;
                            };
                            if obj.side() != side_id || !obj.need_repair() {
                                continue;
                            }
                            let Some(tp) = get_world_pos(&self.objects, oid) else {
                                continue;
                            };
                            if (tp - my_pos).length_squared() < repair_dist * repair_dist {
                                new_target = Some(oid);
                                break;
                            }
                        }
                        if let Some(nt) = new_target {
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                r.env.target = Some(nt);
                                r.env.target_change_repair = now;
                            }
                        }
                    }
                }

                let (tt, rep_target) = {
                    let r = robot_ref(&self.objects, rid).unwrap();
                    (r.env.target_type(), r.env.target)
                };
                if tt == 2 {
                    let des = rep_target
                        .and_then(|t| point_of_aim(map, &self.objects, t))
                        .unwrap_or_default();
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.fire(des, 2);
                    }
                } else if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.stop_fire();
                }

                if war {
                    // Covered-too-long recovery (MatrixSide.cpp:7838-7851).
                    let in_place = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        is_in_place(map, r)
                    };
                    let r = robot_mut(&mut self.objects, rid).unwrap();
                    if in_place {
                        if r.env.enemy_cnt() > 1
                            && now - r.env.target_change > 4000
                            && now - r.env.last_fire > 4000
                        {
                            r.env.target_attack = None;
                        }
                        if now - r.env.target_change > 4000 && now - r.env.last_fire > 6000 {
                            let can = now - r.env.place_not_found > 2000;
                            if can {
                                let p = r.env.place;
                                r.env.add_bad_place(p);
                                r.env.place = -1;
                            }
                        }
                    } else {
                        r.env.target_change = now;
                    }
                }
            }
        }
        rv
    }

    /// Port of `CMatrixSideUnit::FirePL` (MatrixSide.cpp:7002-7231).
    fn fire_pl(&mut self, map: &GameMap, group: usize) -> bool {
        let grp = self.group_robots(group);
        if grp.is_empty() {
            return false;
        }

        // Acquire/validate targets (MatrixSide.cpp:7021-7087).
        for &rid in &grp {
            let drop_target = {
                let r = robot_ref(&self.objects, rid).unwrap();
                if let Some(t) = r.env.target_attack {
                    if r.env.search_enemy(t) {
                        let far = get_world_pos(&self.objects, t)
                            .zip(get_world_pos(&self.objects, rid))
                            .map(|(a, b)| {
                                (a - b).length_squared() > r.max_fire_dist * r.max_fire_dist
                            })
                            .unwrap_or(true);
                        far
                    } else {
                        false
                    }
                } else {
                    false
                }
            };
            if drop_target {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.env.target_attack = None;
                }
            }
            let need_new = {
                let r = robot_ref(&self.objects, rid).unwrap();
                !r.env
                    .target_attack
                    .map(|t| r.env.search_enemy(t))
                    .unwrap_or(false)
            };
            if need_new {
                if let Some(found) = self.pick_enemy_for(map, rid, None) {
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.env.target_attack = Some(found);
                    }
                }
            }
        }

        self.fire_correction(map, group, false)
    }

    /// Port of `CMatrixSideUnit::RepairPL` (MatrixSide.cpp:6831-7000).
    fn repair_pl(&mut self, map: &GameMap, group: usize) {
        let now = self.elapsed_ms as i32;
        let side_id = self.player_side.id;
        let grp = self.group_robots(group);
        if grp.is_empty() {
            return;
        }
        let mut mm = 0u8;
        for &rid in &grp {
            mm |= chassis_bit(robot_ref(&self.objects, rid).unwrap());
        }
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let half = gsm * ROBOT_MOVECELLS_PER_SIZE as f32 / 2.0;

        let target = self.player_side.player_groups[group].obj.filter(|&o| {
            self.player_side.player_groups[group].order == PlayerOrder::Repair
                && self.objects.get(o).map(|x| x.is_live()).unwrap_or(false)
        });

        if let Some(tobj) = target {
            // Point everyone's repair beam at the ordered target
            // (MatrixSide.cpp:6860-6863).
            for &rid in &grp {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    if Some(rid) == Some(tobj) {
                        r.env.target = None;
                    } else {
                        r.env.target = Some(tobj);
                    }
                }
            }

            // Does everyone's repairer reach? (6866-6879)
            let twp = get_world_pos(&self.objects, tobj).unwrap_or_default();
            let mut rlok: Vec<bool> = Vec::with_capacity(grp.len());
            let mut ok = true;
            for &rid in &grp {
                let r = robot_ref(&self.objects, rid).unwrap();
                let mut this_ok = true;
                if r.repair_dist > 0.0 {
                    let tp = pl_place_pos(map, r);
                    let v = glam::Vec2::new(
                        gsm * tp.0 as f32 + half,
                        gsm * tp.1 as f32 + half,
                    );
                    if (twp - v).length_squared() > r.repair_dist * r.repair_dist
                        && now - r.env.place_not_found > 2000
                    {
                        this_ok = false;
                        ok = false;
                    }
                }
                rlok.push(this_ok);
            }

            // Re-place anyone out of range (6882-6963).
            if !ok {
                let (r0_pos, r0_repair) = {
                    let r0 = robot_ref(&self.objects, grp[0]).unwrap();
                    ((r0.map_x, r0.map_y), r0.repair_dist)
                };
                let ttp = get_map_pos(&self.objects, tobj).unwrap_or((0, 0));
                let mut list: Vec<i32> = Vec::new();
                let (st, _) = place_list(
                    map,
                    mm,
                    r0_pos,
                    ttp,
                    float2int(r0_repair / gsm),
                    false,
                    &mut list,
                );
                if st == 0 {
                    for &rid in &grp {
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.env.place_not_found = now;
                        }
                    }
                    return;
                }
                zero_places(map, &list);
                mark_occupied_places(map, &self.objects);

                for (i, &rid) in grp.iter().enumerate() {
                    if rlok[i] {
                        continue;
                    }
                    // First free place whose center reaches the target.
                    let repair_dist = robot_ref(&self.objects, rid).unwrap().repair_dist;
                    let mut chosen: Option<i32> = None;
                    for &pi in &list {
                        let Some((data, _mv, pos, _)) = place_get(map, pi) else {
                            continue;
                        };
                        if data != 0 {
                            continue;
                        }
                        let v = glam::Vec2::new(
                            gsm * pos.0 as f32 + half,
                            gsm * pos.1 as f32 + half,
                        );
                        if (v - twp).length_squared() < repair_dist * repair_dist {
                            chosen = Some(pi);
                            break;
                        }
                    }
                    match chosen {
                        Some(pi) => {
                            place_set_data(map, pi, 1);
                            let pos = place_get(map, pi).unwrap().2;
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                r.env.place = pi;
                                r.env.place_add = (-1, -1);
                                if prepare_break_order(r) {
                                    r.move_to_high(pos.0, pos.1);
                                }
                            }
                        }
                        None => {
                            // Robots don't fit — degrade exactly like
                            // the C++ tail (6914-6953): grow + first
                            // free place regardless of reach.
                            for &rid2 in &grp[i..] {
                                if let Some(r) = robot_mut(&mut self.objects, rid2) {
                                    r.env.place = -1;
                                    r.env.place_add = (-1, -1);
                                    r.env.place_not_found = now;
                                }
                            }
                            let mut list2 = list.clone();
                            for &rid2 in &grp[i..] {
                                let mut free: Option<i32> = None;
                                loop {
                                    for &pi in &list2 {
                                        let Some((data, _, _, _)) = place_get(map, pi) else {
                                            continue;
                                        };
                                        if data == 0 {
                                            free = Some(pi);
                                            break;
                                        }
                                    }
                                    if free.is_some() {
                                        break;
                                    }
                                    if place_list_grow(map, mm, &mut list2, grp.len() as i32) <= 0
                                    {
                                        break;
                                    }
                                    zero_places(map, &list2);
                                    mark_occupied_places(map, &self.objects);
                                }
                                if let Some(pi) = free {
                                    place_set_data(map, pi, 1);
                                    let pos = place_get(map, pi).unwrap().2;
                                    if let Some(r) = robot_mut(&mut self.objects, rid2) {
                                        r.env.place = pi;
                                        r.env.place_add = (-1, -1);
                                        if prepare_break_order(r) {
                                            r.move_to_high(pos.0, pos.1);
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        } else {
            // No ordered target — each repairer finds someone nearby
            // (MatrixSide.cpp:6966-6989).
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
                for oid in self.objects.iter_live() {
                    if oid == rid {
                        continue;
                    }
                    let Some(obj) = self.objects.get(oid) else {
                        continue;
                    };
                    if !obj.is_live() || obj.side() != side_id || !obj.need_repair() {
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
        }

        // Fire the beams (MatrixSide.cpp:6993-6999).
        for &rid in &grp {
            let t = robot_ref(&self.objects, rid).unwrap().env.target;
            let Some(t) = t else { continue };
            let Some(des) = point_of_aim(map, &self.objects, t) else {
                continue;
            };
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.fire(des, 2);
            }
        }
    }
}

/// `PointOfAim` reading a borrowed object directly (no map Z lookup
/// for buildings — those callers use [`logic::point_of_aim`]).
fn point_of_aim_core(obj: &dyn MapStatic) -> Option<glam::Vec3> {
    let mut p = obj.core().geo_center;
    match obj.core().obj_type {
        ObjectType::RobotAi | ObjectType::Cannon => p.z += 5.0,
        ObjectType::Building => p.z += 20.0,
        _ => return None,
    }
    Some(p)
}

/// `IsIntersectSphere` (MatrixLib Math3D) — ray/sphere, returns `t`.
fn ray_sphere(center: glam::Vec3, radius: f32, from: glam::Vec3, dir: glam::Vec3) -> Option<f32> {
    let oc = from - center;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    Some(t)
}

impl MapLogic {
    /// Port of `CMatrixSideUnit::WarPL` (MatrixSide.cpp:7233-7855).
    fn war_pl(&mut self, map: &GameMap, group: usize) {
        let now = self.elapsed_ms as i32;
        let grp = self.group_robots(group);
        if grp.is_empty() {
            return;
        }
        let mut mm = 0u8;
        for &rid in &grp {
            mm |= chassis_bit(robot_ref(&self.objects, rid).unwrap());
        }
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let order = self.player_side.player_groups[group].order;
        let pg_obj = self.player_side.player_groups[group].obj;

        // ── Target selection (MatrixSide.cpp:7256-7360) ────────────
        for &rid in &grp {
            // Attack order: lock onto the designated object.
            if order == PlayerOrder::Attack {
                if let Some(t) = pg_obj {
                    let valid = t != rid
                        && self.objects.get(t).map(|o| o.is_live()).unwrap_or(false);
                    if valid {
                        let lock = {
                            let r = robot_ref(&self.objects, rid).unwrap();
                            r.env.target_attack != Some(t) && r.env.search_enemy(t)
                        };
                        if lock {
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                r.env.target_attack = Some(t);
                            }
                        }
                    }
                }
            }
            // Capture order: prefer the target base's turrets.
            if order == PlayerOrder::Capture {
                if let Some(t) = pg_obj {
                    let t_alive = self.objects.get(t).map(|o| o.is_live()).unwrap_or(false);
                    let cur_is_other = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        r.env
                            .target_attack
                            .filter(|&ta| {
                                self.objects.get(ta).map(|o| o.is_live()).unwrap_or(false)
                            })
                            .map(|ta| {
                                cannon_ref(&self.objects, ta)
                                    .map(|c| c.parent != Some(t))
                                    .unwrap_or(true)
                            })
                            .unwrap_or(false)
                    };
                    if t_alive && cur_is_other {
                        let my_wp = get_world_pos(&self.objects, rid).unwrap_or_default();
                        let mut best: Option<(f32, ObjectId)> = None;
                        let enemies: Vec<ObjectId> = robot_ref(&self.objects, rid)
                            .unwrap()
                            .env
                            .enemies
                            .iter()
                            .map(|e| e.enemy)
                            .collect();
                        for eid in enemies {
                            let is_t_cannon = cannon_ref(&self.objects, eid)
                                .map(|c| {
                                    !matches!(
                                        c.state,
                                        CannonState::Dip | CannonState::UnderConstruction
                                    ) && c.parent == Some(t)
                                })
                                .unwrap_or(false);
                            if !is_t_cannon {
                                continue;
                            }
                            let Some(ep) = get_world_pos(&self.objects, eid) else {
                                continue;
                            };
                            let cd = (ep - my_wp).length_squared();
                            if best.map(|(bd, _)| cd < bd).unwrap_or(true) {
                                best = Some((cd, eid));
                            }
                        }
                        if let Some((_, eid)) = best {
                            if let Some(r) = robot_mut(&mut self.objects, rid) {
                                r.env.target_attack = Some(eid);
                                r.env.place = -1;
                            }
                        }
                    }
                }
            }
            // No target yet → standard pick.
            let need = robot_ref(&self.objects, rid)
                .unwrap()
                .env
                .target_attack
                .is_none();
            if need {
                if let Some(found) = self.pick_enemy_for(map, rid, pg_obj.filter(|&o| o != rid)) {
                    let reposition = cannon_ref(&self.objects, found)
                        .map(|c| {
                            !matches!(c.state, CannonState::Dip | CannonState::UnderConstruction)
                        })
                        .unwrap_or(false)
                        || building_ref(&self.objects, found)
                            .map(|b| b.is_live())
                            .unwrap_or(false);
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.env.target_attack = Some(found);
                        if reposition {
                            r.env.place = -1;
                        }
                    }
                }
            }
        }

        // ── Movement validation (MatrixSide.cpp:7362-7417) ─────────
        let mut rlokmove = vec![true; grp.len()];
        let mut moveok = true;
        for (i, &rid) in grp.iter().enumerate() {
            let r = robot_ref(&self.objects, rid).unwrap();
            if !r.can_break_order() {
                continue;
            }
            let Some(ta) = r.env.target_attack else {
                continue;
            };
            if r.env.place < 0 {
                if can_change_place(now, r) {
                    rlokmove[i] = false;
                    moveok = false;
                }
                continue;
            }
            let Some(tv) = get_world_pos(&self.objects, ta) else {
                continue;
            };
            match place_get(map, r.env.place) {
                None => {
                    if can_change_place(now, r) {
                        rlokmove[i] = false;
                        moveok = false;
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.env.place = -1;
                        }
                    }
                }
                Some((_, _, pos, _)) => {
                    let half = gsm * ROBOT_MOVECELLS_PER_SIZE as f32 / 2.0;
                    let pc = glam::Vec2::new(gsm * pos.0 as f32 + half, gsm * pos.1 as f32 + half);
                    let dist2 = (pc - tv).length_squared();
                    let reach = r.max_fire_dist - half;
                    if dist2 > reach * reach && can_change_place(now, r) {
                        rlokmove[i] = false;
                        moveok = false;
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.env.place = -1;
                        }
                    }
                }
            }
        }

        // ── Re-place robots that lost their slot (7420-7680) ───────
        if !moveok {
            // Group target centroid → anchor target (the target
            // closest to the centroid).
            let mut tp_sum = (0i64, 0i64);
            let mut f = 0i64;
            for &rid in &grp {
                let r = robot_ref(&self.objects, rid).unwrap();
                let Some(ta) = r.env.target_attack else {
                    continue;
                };
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
                let r = robot_ref(&self.objects, rid).unwrap();
                let Some(ta) = r.env.target_attack else {
                    continue;
                };
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
                let d = (((center.0 - tp2.0) as f32).powi(2) + ((center.1 - tp2.1) as f32).powi(2))
                    .sqrt();
                radius =
                    radius.max(float2int(d + r.max_fire_dist / gsm + ROBOT_MOVECELLS_PER_SIZE as f32));
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

                    // Target geometry (MatrixSide.cpp:7525-7542).
                    let (tvx, tvy, enemy_fire_dist) = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        let me_pos = glam::Vec2::new(r.pos_x, r.pos_y);
                        match self.objects.get(ta).map(|o| o.core().obj_type) {
                            Some(ObjectType::RobotAi) => {
                                let t = robot_ref(&self.objects, ta).unwrap();
                                (
                                    t.pos_x - me_pos.x,
                                    t.pos_y - me_pos.y,
                                    float2int(t.max_fire_dist),
                                )
                            }
                            Some(ObjectType::Cannon) => {
                                let c = cannon_ref(&self.objects, ta).unwrap();
                                (
                                    c.pos.x - me_pos.x,
                                    c.pos.y - me_pos.y,
                                    (c.fire_radius(&self.objects) + gsm) as i32,
                                )
                            }
                            Some(ObjectType::Building) => {
                                let b = building_ref(&self.objects, ta).unwrap();
                                (b.pos.x - me_pos.x, b.pos.y - me_pos.y, 150)
                            }
                            _ => {
                                i += 1;
                                continue;
                            }
                        }
                    };
                    let tsize2 = tvx * tvx + tvy * tvy;
                    let tsize2o = if tsize2 != 0.0 { 1.0 / tsize2 } else { 0.0 };

                    let ta_type = self.objects.get(ta).map(|o| o.core().obj_type);
                    let (r_pos, r_maxfd, r_minfd, r_chassis_bit) = {
                        let r = robot_ref(&self.objects, rid).unwrap();
                        (
                            glam::Vec2::new(r.pos_x, r.pos_y),
                            r.max_fire_dist,
                            r.min_fire_dist,
                            chassis_bit(r),
                        )
                    };
                    let des = point_of_aim(map, &self.objects, ta).unwrap();

                    let mut placebest: i32 = -1;
                    let mut s_f1 = 0.0f32;
                    let mut s_underfire = 0i32;
                    let mut s_close = false;

                    for &iplace in &list {
                        let Some((data, mv, pos, underfire0)) = place_get(map, iplace) else {
                            continue;
                        };
                        if data != 0 {
                            continue;
                        }
                        if mv & r_chassis_bit != 0 {
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
                        if !have_bomb {
                            match ta_type {
                                Some(ObjectType::Cannon) => {}
                                Some(ObjectType::Building) => {
                                    if k > 1.2 {
                                        continue;
                                    }
                                }
                                _ => {
                                    if k > 0.95 {
                                        continue;
                                    }
                                }
                            }
                        }
                        let m = (-pvx * tvy + pvy * tvx) * tsize2o;
                        let distfrom2 = (-m * tvy).powi(2) + (m * tvx).powi(2);
                        // NOTE: faithful to the C++ at 7576 which uses
                        // (tvx - pcx)² + (tvy - pcx)² — including its
                        // pcx-for-pvy quirk.
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
                        let (res, _) = trace(
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
                        // Last resort: first free movable place, else grow.
                        if let Some(r) = robot_mut(&mut self.objects, rid) {
                            r.env.place_not_found = now;
                        }
                        let mut chosen: Option<i32> = None;
                        for &pi in &list {
                            let Some((data, mv, _, _)) = place_get(map, pi) else {
                                continue;
                            };
                            if data != 0 || mv & r_chassis_bit != 0 {
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
                            // C++'s for-loop increment advances to the
                            // next robot after a successful grow
                            // (MatrixSide.cpp:7666-7675).
                            i += 1;
                        }
                    }
                }
            }
        }

        // ── Fire correction (7682-7854) ────────────────────────────
        self.fire_correction(map, group, true);
    }

    /// Port of `SelGroupToLogicGroup` (MatrixSide.cpp:7857-7895) —
    /// allocate a free logic-group slot and assign every live robot
    /// of the current selection group into it.
    pub fn sel_group_to_logic_group(&mut self) -> usize {
        let no = self.alloc_logic_group();
        let members: Vec<ObjectId> = self
            .player_side
            .cur_group()
            .map(|g| g.objects.iter().map(|o| o.object).collect())
            .unwrap_or_default();
        for rid in members {
            let live = robot_ref(&self.objects, rid)
                .map(|r| r.is_live())
                .unwrap_or(false);
            if live {
                self.player_side.player_groups[no].robot_cnt += 1;
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.group_logic = no as i32;
                }
            }
        }
        no
    }

    /// Port of `RobotToLogicGroup` (MatrixSide.cpp:7897-7928).
    pub fn robot_to_logic_group(&mut self, robot: ObjectId) -> usize {
        let no = self.alloc_logic_group();
        self.player_side.player_groups[no].robot_cnt += 1;
        if let Some(r) = robot_mut(&mut self.objects, robot) {
            r.group_logic = no as i32;
        }
        no
    }

    /// Shared census + free-slot scan of the two functions above.
    fn alloc_logic_group(&mut self) -> usize {
        for pg in self.player_side.player_groups.iter_mut() {
            pg.robot_cnt = 0;
        }
        for &rid in &self.side_robots() {
            let gl = robot_ref(&self.objects, rid).unwrap().group_logic;
            if (0..MAX_LOGIC_GROUP as i32).contains(&gl) {
                self.player_side.player_groups[gl as usize].robot_cnt += 1;
            }
        }
        let no = self
            .player_side
            .player_groups
            .iter()
            .position(|pg| pg.robot_cnt <= 0)
            .unwrap_or(MAX_LOGIC_GROUP - 1);
        let pg = &mut self.player_side.player_groups[no];
        pg.order = PlayerOrder::Stop;
        pg.obj = None;
        pg.war = false;
        pg.road_path.clear();
        self.distribute_road_path(no);
        no
    }

    // ── PGOrder family (MatrixSide.cpp:7930-8427) ───────────────────

    /// `PGOrderStop` (MatrixSide.cpp:7930-7951).
    pub fn pg_order_stop(&mut self, map: &GameMap, no: usize) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        let to = self.pg_calc_center_group(no);
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::Stop;
            pg.to = to;
            pg.obj = None;
            pg.show_place = true;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, None);
        let pc = self.pg_calc_place_center(map, no);
        let d2 = ((to.0 - pc.0) as i64).pow(2) + ((to.1 - pc.1) as i64).pow(2);
        if d2 > 10 * 10 {
            self.pg_assign_place_player(map, no, to);
        } else {
            self.pg_show_place(map, no);
        }
        self.player_side.last_takt_pl = -1000;
        self.takt_pl(map, -1);
        self.pg_show_place(map, no);
    }

    /// `PGOrderMoveTo` (MatrixSide.cpp:7953-8011).
    pub fn pg_order_move_to(&mut self, map: &GameMap, no: usize, tp: (i32, i32)) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        if self.player_side.player_groups[no].order == PlayerOrder::MoveTo {
            self.sound_order_in_progress();
        } else {
            let chassis = self
                .player_side
                .get_cur_sel_object()
                .and_then(|o| robot_ref(&self.objects, o))
                .filter(|r| r.is_live())
                .map(|r| r.chassis.kind_index() as i32 + 1);
            if let Some(ch) = chassis {
                self.queue_sound(GameSound::ChassisMoveTo(ch));
            }
        }
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::MoveTo;
            pg.to = tp;
            pg.obj = None;
            pg.show_place = true;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, None);
        self.pg_assign_place_player(map, no, tp);
        self.player_side.last_takt_pl = -1000;
        self.takt_pl(map, -1);
        self.pg_show_place(map, no);
    }

    /// `PGOrderCapture` (MatrixSide.cpp:8033-8097).
    pub fn pg_order_capture(&mut self, map: &GameMap, no: usize, building: ObjectId) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        self.enable_capture_fuckoff_sound = true;
        self.sound_order_capture_enabled = true;
        if self.player_side.player_groups[no].order == PlayerOrder::Capture {
            self.sound_order_in_progress();
        } else {
            self.queue_sound(GameSound::OrderCapturePush);
        }
        let to = get_map_pos(&self.objects, building).unwrap_or((-1, -1));
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::Capture;
            pg.to = to;
            pg.obj = Some(building);
            pg.show_place = true;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, Some(building));

        // Cancel mismatched captures (MatrixSide.cpp:8070-8092).
        for &rid in &self.side_robots() {
            let (gl, b2, can) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (r.group_logic, r.get_capture_factory(), r.can_break_order())
            };
            let mismatched = if gl == no as i32 {
                b2.map(|b| b != building).unwrap_or(false)
            } else {
                b2 == Some(building)
            };
            if mismatched && can {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.stop_capture();
                    r.env.place = -1;
                }
            }
        }
        self.player_side.last_takt_pl = -1000;
        self.takt_pl(map, no as i32);
        self.pg_show_place(map, no);
    }

    /// `PGOrderAttack` (MatrixSide.cpp:8099-8173).
    pub fn pg_order_attack(&mut self, map: &GameMap, no: usize, tp: (i32, i32), target: Option<ObjectId>) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        if !self.sound_order_attack_disable {
            if self.player_side.player_groups[no].order == PlayerOrder::Attack {
                self.sound_order_in_progress();
            } else {
                self.queue_sound(GameSound::OrderAttack);
            }
        }
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::Attack;
            pg.to = tp;
            pg.obj = target;
            pg.show_place = true;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, target);

        for &rid in &self.group_robots(no) {
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.env.place = -1;
                r.env.place_add = (-1, -1);
            }
        }

        if let Some(t) = target {
            // Rally point clear of the base mouth (MatrixSide.cpp:8141-8148).
            if let Some(b) = building_ref(&self.objects, t) {
                let angle = b.angle;
                let pg = &mut self.player_side.player_groups[no];
                match angle {
                    0 => pg.to.1 += 20,
                    1 => pg.to.0 -= 20,
                    2 => pg.to.1 -= 20,
                    3 => pg.to.0 += 20,
                    _ => {}
                }
            }
            for &rid in &self.group_robots(no) {
                let lock = robot_ref(&self.objects, rid)
                    .map(|r| r.env.search_enemy(t))
                    .unwrap_or(false);
                if lock {
                    if let Some(r) = robot_mut(&mut self.objects, rid) {
                        r.env.target = Some(t);
                    }
                }
            }
        }
        self.player_side.last_takt_pl = -1000;
        self.takt_pl(map, -1);
        self.pg_show_place(map, no);
    }

    /// `PGOrderPatrol` (MatrixSide.cpp:8175-8223).
    pub fn pg_order_patrol(&mut self, map: &GameMap, no: usize, tp: (i32, i32)) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        if self.player_side.player_groups[no].order == PlayerOrder::Patrol {
            self.sound_order_in_progress();
        } else {
            let chassis = self
                .player_side
                .get_cur_sel_object()
                .and_then(|o| robot_ref(&self.objects, o))
                .filter(|r| r.is_live())
                .map(|r| r.chassis.kind_index() as i32 + 1);
            if let Some(ch) = chassis {
                self.queue_sound(GameSound::ChassisPatrol(ch));
            }
        }
        let from = self.pg_calc_place_center(map, no);
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::Patrol;
            pg.to = tp;
            pg.from = from;
            pg.obj = None;
            pg.patrol_return = false;
            pg.show_place = true;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, None);
        self.pg_assign_place_player(map, no, tp);
        self.player_side.last_takt_pl = -1000;
        self.takt_pl(map, -1);
        self.pg_show_place(map, no);
    }

    /// `PGOrderRepair` (MatrixSide.cpp:8225-8261).
    pub fn pg_order_repair(&mut self, map: &GameMap, no: usize, target: ObjectId) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        if self.player_side.player_groups[no].order == PlayerOrder::Repair {
            self.sound_order_in_progress();
        } else if self.rng.range(0, 1) == 0 {
            self.queue_sound(GameSound::OrderAccept);
        } else {
            self.queue_sound(GameSound::OrderRepair);
        }
        let to = get_map_pos(&self.objects, target).unwrap_or((-1, -1));
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::Repair;
            pg.to = to;
            pg.from = (-1, -1);
            pg.obj = Some(target);
            pg.show_place = true;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, Some(target));
        self.player_side.last_takt_pl = -1000;
        self.takt_pl(map, -1);
        self.pg_show_place(map, no);
    }

    /// `PGOrderBomb` (MatrixSide.cpp:8263-8289).
    pub fn pg_order_bomb(&mut self, map: &GameMap, no: usize, tp: (i32, i32), target: Option<ObjectId>) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::Bomb;
            pg.to = tp;
            pg.obj = target;
            pg.show_place = true;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, target);
        for &rid in &self.group_robots(no) {
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.env.place = -1;
                r.env.place_add = (-1, -1);
            }
        }
        self.player_side.last_takt_pl = -1000;
        self.takt_pl(map, -1);
        self.pg_show_place(map, no);
    }

    /// `PGOrderAutoCapture` (MatrixSide.cpp:8291-8317).
    pub fn pg_order_auto_capture(&mut self, map: &GameMap, no: usize) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        if self.player_side.player_groups[no].order == PlayerOrder::AutoCapture {
            self.sound_order_in_progress();
        } else {
            self.queue_sound(GameSound::OrderAutoCapture);
        }
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::AutoCapture;
            pg.to = (-1, -1);
            pg.from = (-1, -1);
            pg.obj = None;
            pg.show_place = false;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, None);
        let _ = map;
    }

    /// `PGOrderAutoAttack` (MatrixSide.cpp:8319-8363).
    pub fn pg_order_auto_attack(&mut self, map: &GameMap, no: usize) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        if self.player_side.player_groups[no].order == PlayerOrder::AutoAttack {
            self.sound_order_in_progress();
        } else {
            self.queue_sound(GameSound::OrderAutoAttack);
        }
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::AutoAttack;
            pg.to = (-1, -1);
            pg.from = (-1, -1);
            pg.obj = None;
            pg.show_place = false;
            pg.region = -1;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, None);
        for &rid in &self.side_robots() {
            let (gl, cf, can) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (r.group_logic, r.get_capture_factory(), r.can_break_order())
            };
            if gl == no as i32 && cf.is_some() && can {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.stop_capture();
                    r.env.place = -1;
                    r.env.place_not_found = -10000;
                }
            }
        }
        let _ = map;
    }

    /// `PGOrderAutoDefence` (MatrixSide.cpp:8365-8427).
    pub fn pg_order_auto_defence(&mut self, map: &GameMap, no: usize) {
        if self.player_side.player_groups[no].robot_cnt <= 0 {
            return;
        }
        if self.player_side.player_groups[no].order == PlayerOrder::AutoDefence {
            self.sound_order_in_progress();
        } else {
            self.queue_sound(GameSound::OrderAutoDefence);
        }
        let mut tp = (0i64, 0i64);
        let mut cnt = 0i64;
        for &rid in &self.side_robots() {
            let (gl, mx, my, cf, can) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                (
                    r.group_logic,
                    r.map_x,
                    r.map_y,
                    r.get_capture_factory(),
                    r.can_break_order(),
                )
            };
            if gl != no as i32 {
                continue;
            }
            tp.0 += mx as i64;
            tp.1 += my as i64;
            cnt += 1;
            if cf.is_some() && can {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.stop_capture();
                    r.env.place = -1;
                    r.env.place_not_found = -10000;
                }
            }
        }
        if cnt == 0 {
            return;
        }
        let region = get_region(map, (tp.0 / cnt) as i32, (tp.1 / cnt) as i32);
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.order = PlayerOrder::AutoDefence;
            pg.to = (-1, -1);
            pg.from = (-1, -1);
            pg.obj = None;
            pg.show_place = false;
            pg.region = region;
            pg.road_path.clear();
        }
        self.distribute_road_path(no);
        self.pg_remove_all_passive(no, None);
        if region >= 0 {
            let center = map
                .road_network
                .as_ref()
                .map(|rn| {
                    let rn = rn.lock().unwrap();
                    let r = rn.get_region(region);
                    (r.center.x, r.center.y)
                })
                .unwrap_or((-1, -1));
            if center.0 >= 0 {
                self.pg_assign_place(map, no, center);
            }
        }
        self.move_group_to_places(map, no);
    }

    /// `PGRemoveAllPassive` (MatrixSide.cpp:8429-8452) — purge
    /// non-combat targets (buildings + own units) from the group's
    /// envs, except `skip`.
    fn pg_remove_all_passive(&mut self, no: usize, skip: Option<ObjectId>) {
        let side_id = self.player_side.id;
        for &rid in &self.group_robots(no) {
            // Resolve passivity with arena reads first.
            let (target_passive, passive_enemies) = {
                let r = robot_ref(&self.objects, rid).unwrap();
                let is_passive = |id: ObjectId| -> bool {
                    let Some(obj) = self.objects.get(id) else {
                        return false;
                    };
                    matches!(obj.core().obj_type, ObjectType::Building)
                        || (obj.is_live() && obj.side() == side_id)
                };
                let tp = r
                    .env
                    .target
                    .map(|t| Some(t) != skip && is_passive(t))
                    .unwrap_or(false);
                let pe: Vec<ObjectId> = r
                    .env
                    .enemies
                    .iter()
                    .map(|e| e.enemy)
                    .filter(|&e| Some(e) != skip && is_passive(e))
                    .collect();
                (tp, pe)
            };
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                if target_passive {
                    r.env.target = None;
                }
                for e in passive_enemies {
                    r.env.remove_from_list(e);
                }
            }
        }
    }

    /// `PGSetPlace` (MatrixSide.cpp:8454-8474).
    fn pg_set_place(&mut self, map: &GameMap, rid: ObjectId, p: (i32, i32)) {
        let mut found: Option<i32> = None;
        if let Some(rn_lock) = map.road_network.as_ref() {
            let rn = rn_lock.lock().unwrap();
            let rect = crate::matrix_game::road_network::Rect {
                left: p.0 - 4,
                top: p.1 - 4,
                right: p.0 + 4,
                bottom: p.1 + 4,
            };
            let plr = rn.correct_rect_pl(&rect);
            'outer: for y in plr.top..plr.bottom {
                for x in plr.left..plr.right {
                    let plist = rn.pl_list[(x + y * rn.pl_size_x) as usize];
                    for u in 0..plist.cnt {
                        let pi = plist.sme + u;
                        let pos = rn.places[pi as usize].pos;
                        if !(p.0 >= pos.x + 4 || p.0 + 4 <= pos.x)
                            && !(p.1 >= pos.y + 4 || p.1 + 4 <= pos.y)
                        {
                            found = Some(pi);
                            break 'outer;
                        }
                    }
                }
            }
        }
        if let Some(r) = robot_mut(&mut self.objects, rid) {
            match found {
                Some(pi) => {
                    r.env.place = pi;
                    r.env.place_add = (-1, -1);
                }
                None => {
                    r.env.place = -1;
                    r.env.place_add = p;
                }
            }
        }
    }

    /// `PGCalcCenterGroup` (MatrixSide.cpp:8476-8495).
    fn pg_calc_center_group(&self, no: usize) -> (i32, i32) {
        let mut tp = (0i64, 0i64);
        let mut cnt = 0i64;
        for &rid in &self.group_robots(no) {
            let r = robot_ref(&self.objects, rid).unwrap();
            tp.0 += r.map_x as i64;
            tp.1 += r.map_y as i64;
            cnt += 1;
        }
        if cnt == 0 {
            return (0, 0);
        }
        ((tp.0 / cnt) as i32, (tp.1 / cnt) as i32)
    }

    /// `PGPlaceClear` (MatrixSide.cpp:8497-8511).
    fn pg_place_clear(&mut self, no: usize) {
        for &rid in &self.group_robots(no) {
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.env.place = -1;
                r.env.place_add = (-1, -1);
            }
        }
    }

    /// `PGCalcPlaceCenter` (MatrixSide.cpp:8513-8536).
    fn pg_calc_place_center(&self, map: &GameMap, no: usize) -> (i32, i32) {
        let mut tp = (0i64, 0i64);
        let mut cnt = 0i64;
        for &rid in &self.group_robots(no) {
            let r = robot_ref(&self.objects, rid).unwrap();
            let p = pl_place_pos(map, r);
            if p.0 >= 0 {
                tp.0 += p.0 as i64;
                tp.1 += p.1 as i64;
                cnt += 1;
            }
        }
        if cnt == 0 {
            return self.pg_calc_center_group(no);
        }
        ((tp.0 / cnt) as i32, (tp.1 / cnt) as i32)
    }

    /// `PGShowPlace` (MatrixSide.cpp:8538-8568) — replace the move-to
    /// pings with one per group robot.
    fn pg_show_place(&mut self, map: &GameMap, no: usize) {
        self.moveto_clear_pending = true;
        self.moveto_pings.clear();
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let half = gsm * ROBOT_MOVECELLS_PER_SIZE as f32 / 2.0;
        for &rid in &self.group_robots(no) {
            let r = robot_ref(&self.objects, rid).unwrap();
            if let Some(cf) = r.get_capture_factory() {
                if let Some(v) = get_world_pos(&self.objects, cf) {
                    self.moveto_pings
                        .push(glam::Vec3::new(v.x, v.y, map.get_z(v.x, v.y) + 2.0));
                }
            } else {
                let tp = pl_place_pos(map, r);
                if tp.0 >= 0 {
                    let x = gsm * tp.0 as f32 + half;
                    let y = gsm * tp.1 as f32 + half;
                    self.moveto_pings
                        .push(glam::Vec3::new(x, y, map.get_z(x, y)));
                }
            }
        }
        self.player_side.player_groups[no].show_place = false;
    }

    /// `PGAssignPlace` (MatrixSide.cpp:8570-8692).
    fn pg_assign_place(&mut self, map: &GameMap, no: usize, center_in: (i32, i32)) {
        let now = self.elapsed_ms as i32;
        let grp = self.group_robots(no);
        if grp.is_empty() {
            return;
        }
        let mut mm = 0u8;
        for &rid in &grp {
            mm |= chassis_bit(robot_ref(&self.objects, rid).unwrap());
        }

        // Nudge center onto a standable cell (MatrixSide.cpp:8595-8598).
        let mut center = center_in;
        {
            let chassis0 = robot_ref(&self.objects, grp[0]).unwrap().chassis.kind_index();
            if let Some((x, y)) = logic::place_find_near_with_blockers(
                map, chassis0, 4, center.0, center.1, &[],
            ) {
                center = (x, y);
            }
        }

        let mut list: Vec<i32> = Vec::new();
        let first = logic::find_near_place(map, mm, center);
        if first < 0 {
            for &rid in &grp {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.env.place_not_found = now;
                }
            }
            return;
        }
        list.push(first);
        if grp.len() > 1 && place_list_grow(map, mm, &mut list, grp.len() as i32 - 1) <= 0 {
            for &rid in &grp {
                if let Some(r) = robot_mut(&mut self.objects, rid) {
                    r.env.place_not_found = now;
                }
            }
            return;
        }
        zero_places(map, &list);
        mark_occupied_places(map, &self.objects);

        let mut i = 0usize;
        while i < grp.len() {
            let rid = grp[i];
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
                        r.env.place_add = (-1, -1);
                    }
                    i += 1;
                }
                None => {
                    if place_list_grow(map, mm, &mut list, grp.len() as i32) <= 0 {
                        for &rid2 in &grp[i..] {
                            if let Some(r) = robot_mut(&mut self.objects, rid2) {
                                r.env.place_not_found = now;
                            }
                        }
                        return;
                    }
                    zero_places(map, &list);
                    mark_occupied_places(map, &self.objects);
                    // retry same robot
                }
            }
        }
    }

    /// `PGAssignPlacePlayer` (MatrixSide.cpp:8694-8757).
    fn pg_assign_place_player(&mut self, map: &GameMap, no: usize, center: (i32, i32)) {
        let side_id = self.player_side.id;

        // Blockers: out-of-group robots' claims + cannon places.
        let mut other: Vec<(i32, i32)> = Vec::new();
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        for oid in self.objects.iter_live() {
            let Some(obj) = self.objects.get(oid) else {
                continue;
            };
            match obj.core().obj_type {
                ObjectType::RobotAi => {
                    let r = robot_ref(&self.objects, oid).unwrap();
                    if !r.is_live() {
                        continue;
                    }
                    if r.side == side_id && r.group_logic == no as i32 {
                        continue;
                    }
                    if r.env.place >= 0 {
                        if let Some(p) = place_pos(map, r.env.place) {
                            other.push(p);
                        }
                    } else if r.env.place_add.0 >= 0 {
                        other.push(r.env.place_add);
                    }
                }
                ObjectType::Cannon => {
                    let c = cannon_ref(&self.objects, oid).unwrap();
                    if !matches!(c.state, CannonState::Dip) {
                        other.push((
                            float2int(c.pos.x / gsm) - ROBOT_MOVECELLS_PER_SIZE / 2,
                            float2int(c.pos.y / gsm) - ROBOT_MOVECELLS_PER_SIZE / 2,
                        ));
                    }
                }
                _ => {}
            }
        }

        for &rid in &self.group_robots(no) {
            let chassis = robot_ref(&self.objects, rid).unwrap().chassis.kind_index();
            let (mx, my) = logic::place_find_near_with_blockers(
                map, chassis, 4, center.0, center.1, &other,
            )
            .unwrap_or(center);
            self.pg_set_place(map, rid, (mx, my));
            let r = robot_ref(&self.objects, rid).unwrap();
            if r.env.place >= 0 {
                if let Some(p) = place_pos(map, r.env.place) {
                    other.push(p);
                }
            } else if r.env.place_add.0 >= 0 {
                other.push(r.env.place_add);
            }
        }
    }
}

impl MapLogic {
    /// `PGCalcStat` (MatrixSide.cpp:8759-8890) — per-region force
    /// census + one-ring danger spread.
    fn pg_calc_stat(&mut self, map: &GameMap) {
        let side_id = self.player_side.id;
        let region_cnt = map
            .road_network
            .as_ref()
            .map(|rn| rn.lock().unwrap().regions.len())
            .unwrap_or(0);
        if region_cnt == 0 {
            return;
        }
        let stats = &mut self.player_side.region_stats;
        stats.clear();
        stats.resize(region_cnt, LogicRegion::default());
        self.player_side.region_index.clear();
        self.player_side
            .region_index
            .resize(region_cnt, 0);

        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        for oid in self.objects.iter_live() {
            let Some(obj) = self.objects.get(oid) else {
                continue;
            };
            match obj.core().obj_type {
                ObjectType::Building => {
                    let b = building_ref(&self.objects, oid).unwrap();
                    let i = get_region(
                        map,
                        (b.pos.x / gsm) as i32,
                        (b.pos.y / gsm) as i32,
                    );
                    if i < 0 {
                        continue;
                    }
                    let st = &mut self.player_side.region_stats[i as usize];
                    if b.side == 0 {
                        st.neutral_building_cnt += 1;
                    } else if b.side != side_id {
                        st.enemy_building_cnt += 1;
                    } else {
                        st.our_building_cnt += 1;
                    }
                    if b.is_base() {
                        if b.side == 0 {
                            st.neutral_base_cnt += 1;
                        } else if b.side != side_id {
                            st.enemy_base_cnt += 1;
                        } else {
                            st.our_base_cnt += 1;
                        }
                    }
                }
                ObjectType::RobotAi => {
                    let r = robot_ref(&self.objects, oid).unwrap();
                    if !r.is_live() {
                        continue;
                    }
                    let i = get_region(
                        map,
                        (r.pos_x / gsm) as i32,
                        (r.pos_y / gsm) as i32,
                    );
                    if i < 0 {
                        continue;
                    }
                    let st = &mut self.player_side.region_stats[i as usize];
                    if r.side == 0 {
                    } else if r.side != side_id {
                        st.enemy_robot_cnt += 1;
                        st.danger += r.strength;
                    } else {
                        st.our_robot_cnt += 1;
                    }
                }
                ObjectType::Cannon => {
                    let c = cannon_ref(&self.objects, oid).unwrap();
                    let i = get_region(
                        map,
                        (c.pos.x / gsm) as i32,
                        (c.pos.y / gsm) as i32,
                    );
                    if i < 0 {
                        continue;
                    }
                    let strength = c.get_strength();
                    let st = &mut self.player_side.region_stats[i as usize];
                    if c.side == 0 {
                        st.neutral_cannon_cnt += 1;
                        st.danger_add += strength;
                    } else if c.side != side_id {
                        st.enemy_cannon_cnt += 1;
                        st.danger_add += strength;
                    } else {
                        st.our_cannon_cnt += 1;
                    }
                }
                _ => {}
            }
        }

        // One-ring danger spread (MatrixSide.cpp:8847-8889).
        let near: Vec<Vec<i32>> = {
            let rn = map.road_network.as_ref().unwrap().lock().unwrap();
            rn.regions.iter().map(|r| r.near.clone()).collect()
        };
        for i in 0..region_cnt {
            if self.player_side.region_stats[i].danger <= 0.0 {
                continue;
            }
            for st in self.player_side.region_stats.iter_mut() {
                st.data = 0;
            }
            let danger = self.player_side.region_stats[i].danger;
            for &t in &near[i] {
                let st = &mut self.player_side.region_stats[t as usize];
                if st.data > 0 {
                    continue;
                }
                st.danger_add += danger;
                st.data = 1;
            }
        }
        for st in self.player_side.region_stats.iter_mut() {
            st.danger += st.danger_add;
        }
    }

    /// Group chassis mask + total strength + mass region
    /// (shared head of the PGFind* searches).
    fn group_mass(&self, map: &GameMap, no: usize) -> (u8, f32, i32) {
        let mut mm = 0u8;
        let mut strength = 0.0f32;
        let mut regionmass = get_region(map, self.pg_calc_center_group(no).0, self.pg_calc_center_group(no).1);
        for &rid in &self.group_robots(no) {
            let r = robot_ref(&self.objects, rid).unwrap();
            mm |= chassis_bit(r);
            strength += r.strength;
            if regionmass < 0 {
                regionmass = get_region(map, r.map_x, r.map_y);
            }
        }
        (mm, strength, regionmass)
    }

    /// Region adjacency snapshot (near + near_move) for wave searches.
    fn region_graph(&self, map: &GameMap) -> Vec<(Vec<i32>, Vec<u8>)> {
        map.road_network
            .as_ref()
            .map(|rn| {
                let rn = rn.lock().unwrap();
                rn.regions
                    .iter()
                    .map(|r| (r.near.clone(), r.near_move.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `PGCalcRegionPath` (MatrixSide.cpp:9298-9347) — backtrack the
    /// wave levels in `region_stats[..].data` from `rend` to the
    /// start, writing the start→end region list into the group.
    fn pg_calc_region_path(&mut self, map: &GameMap, no: usize, rend_in: i32, mm: u8) {
        let graph = self.region_graph(map);
        let mut rev: Vec<i32> = vec![rend_in];
        let mut rend = rend_in;
        let mut level = self.player_side.region_stats[rend as usize].data;
        loop {
            let mut next = -1i32;
            let (near, near_move) = &graph[rend as usize];
            for (t, &u) in near.iter().enumerate() {
                let d = self.player_side.region_stats[u as usize].data;
                if d == 0 || d >= level {
                    continue;
                }
                if near_move[t] & mm != 0 {
                    continue;
                }
                next = u;
                break;
            }
            if next < 0 {
                break; // C++ ERROR_E — degrade gracefully
            }
            if rev.len() >= REGION_PATH_MAX_CNT {
                break;
            }
            rev.push(next);
            rend = next;
            level = self.player_side.region_stats[rend as usize].data;
            if level <= 1 {
                break;
            }
        }
        rev.reverse();
        self.player_side.player_groups[no].region_path = rev;
    }

    /// Build the road route for the freshly computed region path
    /// (the `FindPathFromRegionPath` follow-up every PGFind* does).
    fn pg_build_road_path(&mut self, map: &GameMap, no: usize, mm: u8) {
        {
            let pg = &mut self.player_side.player_groups[no];
            pg.road_path.clear_fast();
            if pg.region_path.len() > 1 {
                if let Some(rn_lock) = map.road_network.as_ref() {
                    let mut rn = rn_lock.lock().unwrap();
                    let rlist = pg.region_path.clone();
                    rn.find_path_from_region_path(mm, &rlist, &mut pg.road_path);
                }
            }
        }
        self.distribute_road_path(no);
    }

    /// Hand every group member a snapshot of the group route — the
    /// arena-handle stand-in for `side->m_PlayerGroup[gl].m_RoadPath`
    /// reads inside `CMatrixRobotAI::ZonePathCalc`.
    fn distribute_road_path(&mut self, no: usize) {
        let snapshot = {
            let pg = &self.player_side.player_groups[no];
            if pg.road_path.list_cnt > 0 {
                Some(std::sync::Arc::new(pg.road_path.clone()))
            } else {
                None
            }
        };
        for &rid in &self.group_robots(no) {
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.group_road_path = snapshot.clone();
            }
        }
    }

    /// `PGFindCaptureFactory` (MatrixSide.cpp:8892-9002).
    fn pg_find_capture_factory(&mut self, map: &GameMap, no: usize) {
        self.pg_calc_stat(map);
        let side_id = self.player_side.id;
        let (mm, strength, regionmass) = self.group_mass(map, no);
        if regionmass < 0 {
            return;
        }

        let building_in_region = |s: &Self, reg: i32| -> Option<ObjectId> {
            for oid in s.objects.iter_live() {
                let Some(b) = building_ref(&s.objects, oid) else {
                    continue;
                };
                if b.is_live() && b.side != side_id {
                    if let Some(tp) = get_map_pos(&s.objects, oid) {
                        if get_region(map, tp.0, tp.1) == reg {
                            return Some(oid);
                        }
                    }
                }
            }
            None
        };

        // Current region first (if not too dangerous).
        if strength >= self.player_side.region_stats[regionmass as usize].danger {
            if let Some(b) = building_in_region(self, regionmass) {
                let pg = &mut self.player_side.player_groups[no];
                pg.obj = Some(b);
                pg.region_path.clear();
                pg.road_path.clear_fast();
                // C++ robots read the group route live (MatrixRobot.cpp:1587);
                // our per-robot snapshots must be refreshed on clear too.
                self.distribute_road_path(no);
                return;
            }
        }

        let graph = self.region_graph(map);
        let region_cnt = graph.len();

        // type 0: safe regions only; type 1: any.
        for ty in 0..=1 {
            for st in self.player_side.region_stats.iter_mut() {
                st.data = 0;
            }
            let mut index: Vec<i32> = vec![regionmass];
            self.player_side.region_stats[regionmass as usize].data = 1;
            let mut sme = 0usize;
            let mut next = index.len();
            let mut dist = 1u32;
            while sme < index.len() {
                let cur = index[sme] as usize;
                let (near, near_move) = &graph[cur];
                for (t, &u) in near.iter().enumerate() {
                    if self.player_side.region_stats[u as usize].data != 0 {
                        continue;
                    }
                    if near_move[t] & mm != 0 {
                        continue;
                    }
                    if ty == 0
                        && strength < self.player_side.region_stats[u as usize].danger
                    {
                        continue;
                    }
                    self.player_side.region_stats[u as usize].data = 1 + dist;
                    if index.len() >= region_cnt * 4 {
                        break;
                    }
                    index.push(u);

                    // Groups must not converge on one region
                    // (MatrixSide.cpp:8963-8974).
                    let mut taken = false;
                    for k in 0..MAX_LOGIC_GROUP {
                        if k == no {
                            continue;
                        }
                        let pgk = &self.player_side.player_groups[k];
                        if pgk.robot_cnt <= 0
                            || !matches!(
                                pgk.order,
                                PlayerOrder::Capture | PlayerOrder::AutoCapture
                            )
                        {
                            continue;
                        }
                        let Some(o) = pgk.obj else { continue };
                        let is_live_b = building_ref(&self.objects, o)
                            .map(|b| b.is_live())
                            .unwrap_or(false);
                        if !is_live_b {
                            continue;
                        }
                        if get_map_pos(&self.objects, o)
                            .map(|tp| get_region(map, tp.0, tp.1) == u)
                            .unwrap_or(false)
                        {
                            taken = true;
                            break;
                        }
                    }
                    if taken {
                        continue;
                    }

                    let st = &self.player_side.region_stats[u as usize];
                    if st.enemy_building_cnt > 0 || st.neutral_building_cnt > 0 {
                        if let Some(b) = building_in_region(self, u) {
                            self.player_side.player_groups[no].obj = Some(b);
                            self.pg_calc_region_path(map, no, u, mm);
                            self.pg_build_road_path(map, no, mm);
                            return;
                        }
                    }
                }
                sme += 1;
                if sme >= next {
                    next = index.len();
                    dist += 1;
                }
            }
        }
    }

    /// `PGFindAttackTarget` (MatrixSide.cpp:9004-9161).
    fn pg_find_attack_target(&mut self, map: &GameMap, no: usize) {
        self.pg_calc_stat(map);
        let side_id = self.player_side.id;
        let (mm, _strength, regionmass) = self.group_mass(map, no);
        if regionmass < 0 {
            return;
        }

        let unit_in_region = |s: &Self, reg: i32| -> Option<ObjectId> {
            for oid in s.objects.iter_live() {
                if !is_live_unit(&s.objects, oid) {
                    continue;
                }
                let Some(obj) = s.objects.get(oid) else {
                    continue;
                };
                if obj.side() == side_id {
                    continue;
                }
                if let Some(tp) = get_map_pos(&s.objects, oid) {
                    if get_region(map, tp.0, tp.1) == reg {
                        return Some(oid);
                    }
                }
            }
            None
        };

        if let Some(t) = unit_in_region(self, regionmass) {
            let pg = &mut self.player_side.player_groups[no];
            pg.obj = Some(t);
            pg.region_path.clear();
            pg.road_path.clear_fast();
            self.distribute_road_path(no);
            return;
        }

        let graph = self.region_graph(map);
        let region_cnt = graph.len();

        for withourgroup in 0..=1 {
            for st in self.player_side.region_stats.iter_mut() {
                st.data = 0;
            }
            let mut index: Vec<i32> = vec![regionmass];
            self.player_side.region_stats[regionmass as usize].data = 1;
            let mut sme = 0usize;
            let mut next = index.len();
            let mut dist = 1u32;
            while sme < index.len() {
                let cur = index[sme] as usize;
                let (near, near_move) = &graph[cur];
                for (t, &u) in near.iter().enumerate() {
                    if self.player_side.region_stats[u as usize].data != 0 {
                        continue;
                    }
                    if near_move[t] & mm != 0 {
                        continue;
                    }
                    self.player_side.region_stats[u as usize].data = 1 + dist;
                    if index.len() >= region_cnt * 4 {
                        break;
                    }
                    index.push(u);

                    if withourgroup == 0 {
                        let mut taken = false;
                        for k in 0..MAX_LOGIC_GROUP {
                            if k == no {
                                continue;
                            }
                            let pgk = &self.player_side.player_groups[k];
                            if pgk.robot_cnt <= 0
                                || !matches!(
                                    pgk.order,
                                    PlayerOrder::Attack | PlayerOrder::AutoAttack
                                )
                            {
                                continue;
                            }
                            let Some(o) = pgk.obj else { continue };
                            if !is_live_unit(&self.objects, o) {
                                continue;
                            }
                            if get_map_pos(&self.objects, o)
                                .map(|tp| get_region(map, tp.0, tp.1) == u)
                                .unwrap_or(false)
                            {
                                taken = true;
                                break;
                            }
                        }
                        if taken {
                            continue;
                        }
                    }

                    let st = &self.player_side.region_stats[u as usize];
                    if st.enemy_robot_cnt > 0
                        || st.enemy_cannon_cnt > 0
                        || st.neutral_cannon_cnt > 0
                    {
                        if let Some(t2) = unit_in_region(self, u) {
                            self.player_side.player_groups[no].obj = Some(t2);
                            self.pg_calc_region_path(map, no, u, mm);
                            self.pg_build_road_path(map, no, mm);
                            return;
                        }
                    }
                }
                sme += 1;
                if sme >= next {
                    next = index.len();
                    dist += 1;
                }
            }
        }

        // No units anywhere — march on an enemy base
        // (MatrixSide.cpp:9113-9160).
        for st in self.player_side.region_stats.iter_mut() {
            st.data = 0;
        }
        let mut index: Vec<i32> = vec![regionmass];
        self.player_side.region_stats[regionmass as usize].data = 1;
        let mut sme = 0usize;
        let mut next = index.len();
        let mut dist = 1u32;
        while sme < index.len() {
            let cur = index[sme] as usize;
            let (near, near_move) = &graph[cur];
            for (t, &u) in near.iter().enumerate() {
                if self.player_side.region_stats[u as usize].data != 0 {
                    continue;
                }
                if near_move[t] & mm != 0 {
                    continue;
                }
                self.player_side.region_stats[u as usize].data = 1 + dist;
                if index.len() >= region_cnt * 4 {
                    break;
                }
                index.push(u);
                if self.player_side.region_stats[u as usize].enemy_base_cnt > 0 {
                    let mut found: Option<ObjectId> = None;
                    for oid in self.objects.iter_live() {
                        let Some(b) = building_ref(&self.objects, oid) else {
                            continue;
                        };
                        if b.is_live() && b.is_base() && b.side != side_id {
                            if let Some(tp) = get_map_pos(&self.objects, oid) {
                                if get_region(map, tp.0, tp.1) == u {
                                    found = Some(oid);
                                    break;
                                }
                            }
                        }
                    }
                    if let Some(b) = found {
                        self.player_side.player_groups[no].obj = Some(b);
                        self.pg_calc_region_path(map, no, u, mm);
                        self.pg_build_road_path(map, no, mm);
                        return;
                    }
                }
            }
            sme += 1;
            if sme >= next {
                next = index.len();
                dist += 1;
            }
        }
    }

    /// `PGFindDefenceTarget` (MatrixSide.cpp:9163-9296).
    fn pg_find_defence_target(&mut self, map: &GameMap, no: usize) {
        self.pg_calc_stat(map);
        let side_id = self.player_side.id;
        let (mm, _strength, regionmass) = self.group_mass(map, no);
        if regionmass < 0 {
            return;
        }

        // Current region first.
        let mut local_target = None;
        for oid in self.objects.iter_live() {
            if !is_live_unit(&self.objects, oid) {
                continue;
            }
            let Some(obj) = self.objects.get(oid) else {
                continue;
            };
            if obj.side() == side_id {
                continue;
            }
            if get_map_pos(&self.objects, oid)
                .map(|tp| get_region(map, tp.0, tp.1) == regionmass)
                .unwrap_or(false)
            {
                local_target = Some(oid);
                break;
            }
        }
        if let Some(oid) = local_target {
            let pg = &mut self.player_side.player_groups[no];
            pg.obj = Some(oid);
            pg.region = regionmass;
            pg.region_path.clear();
            pg.road_path.clear_fast();
            self.distribute_road_path(no);
            return;
        }

        let graph = self.region_graph(map);
        let region_cnt = graph.len();

        // type 0/1: seed from our bases; 2/3: from the group region.
        // Even types respect other groups' claims.
        for ty in 0..=3 {
            for st in self.player_side.region_stats.iter_mut() {
                st.data = 0;
            }
            let mut index: Vec<i32> = Vec::new();
            if ty <= 1 {
                for u in 0..region_cnt {
                    if self.player_side.region_stats[u].our_base_cnt > 0 {
                        index.push(u as i32);
                        self.player_side.region_stats[u].data = 1;
                    }
                }
            } else {
                index.push(regionmass);
                self.player_side.region_stats[regionmass as usize].data = 1;
            }
            let mut sme = 0usize;
            let mut next = index.len();
            let mut dist = 1u32;
            while sme < index.len() {
                let cur = index[sme] as usize;
                let (near, near_move) = &graph[cur];
                for (t, &u) in near.iter().enumerate() {
                    if self.player_side.region_stats[u as usize].data != 0 {
                        continue;
                    }
                    if near_move[t] & mm != 0 {
                        continue;
                    }
                    if self.player_side.region_stats[u as usize].enemy_building_cnt > 0 {
                        continue;
                    }
                    if self.player_side.region_stats[u as usize].enemy_cannon_cnt > 0 {
                        continue;
                    }
                    self.player_side.region_stats[u as usize].data = 1 + dist;
                    if index.len() >= region_cnt * 4 {
                        break;
                    }
                    index.push(u);

                    if ty == 0 || ty == 2 {
                        let mut taken = false;
                        for k in 0..MAX_LOGIC_GROUP {
                            if k == no {
                                continue;
                            }
                            let pgk = &self.player_side.player_groups[k];
                            if pgk.robot_cnt <= 0
                                || !matches!(
                                    pgk.order,
                                    PlayerOrder::Attack | PlayerOrder::AutoAttack
                                )
                            {
                                continue;
                            }
                            let Some(o) = pgk.obj else { continue };
                            if !is_live_unit(&self.objects, o) {
                                continue;
                            }
                            if get_map_pos(&self.objects, o)
                                .map(|tp| get_region(map, tp.0, tp.1) == u)
                                .unwrap_or(false)
                            {
                                taken = true;
                                break;
                            }
                        }
                        if taken {
                            continue;
                        }
                    }

                    let st = &self.player_side.region_stats[u as usize];
                    if st.our_building_cnt > 0 || st.our_cannon_cnt > 0 {
                        // Enemy robot heading into region u?
                        let mut incoming: Option<ObjectId> = None;
                        for oid in self.objects.iter_live() {
                            let Some(r) = robot_ref(&self.objects, oid) else {
                                continue;
                            };
                            if !r.is_live() || r.side == side_id {
                                continue;
                            }
                            let tp = r
                                .move_to_coords()
                                .unwrap_or((r.map_x, r.map_y));
                            let reg = get_region(map, tp.0, tp.1);
                            if reg >= 0 && reg == u {
                                incoming = Some(oid);
                                break;
                            }
                        }
                        if let Some(oid) = incoming {
                            let pg = &mut self.player_side.player_groups[no];
                            pg.region = u;
                            pg.obj = Some(oid);
                            self.pg_calc_region_path(map, no, u, mm);
                            self.pg_build_road_path(map, no, mm);
                            return;
                        }
                    }
                }
                sme += 1;
                if sme >= next {
                    next = index.len();
                    dist += 1;
                }
            }
        }
    }
}

impl MapLogic {
    /// Port of `CMatrixSideUnit::CalcMaxSpeed` (MatrixSide.cpp:
    /// 2343-2447) — robots ahead of their group's march axis slow
    /// down so the formation stays together.
    pub fn calc_max_speed(&mut self, _map: &GameMap) {
        use crate::matrix_game::logic::COLLIDE_BOT_R;
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let half = gsm * ROBOT_MOVECELLS_PER_SIZE as f32 / 2.0;

        // m_GroupSpeed reset to max speed (MatrixSide.cpp:2354-2360).
        for &rid in &self.side_robots() {
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.reset_group_speed();
            }
        }

        for i in 0..MAX_LOGIC_GROUP {
            if self.player_side.player_groups[i].robot_cnt <= 0 {
                continue;
            }
            let grp = self.group_robots(i);
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
            // First straggler gap > 7R splits the pack (2429-2432).
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
                    continue; // stragglers keep full speed
                }
                let k = 0.6 + 0.4 * (maxpr - pr) / (maxpr - COLLIDE_BOT_R * 10.0);
                if let Some(r) = robot_mut(&mut self.objects, e.rid) {
                    r.scale_group_speed(k);
                }
            }
        }
    }

    /// Per-10ms player-side logic slice — the player branch of
    /// `CMatrixSideUnit::LogicTakt` (MatrixSide.cpp:420-540):
    /// PumpGroups, dead-selection cleanup, TaktPL, CalcMaxSpeed,
    /// STAT_TIME upkeep.
    pub fn side_logic_takt(&mut self, map: &GameMap) {
        use crate::matrix_game::side::{SideStatus, Stat};

        if self.player_side.status != SideStatus::None {
            let t = self.elapsed_ms as i32;
            self.player_side.set_stat(Stat::Time, t);
        }

        // Stale-member sweep — the C++ removes objects from groups in
        // each death path; the arena-handle equivalent prunes here.
        {
            let objects = &self.objects;
            self.player_side.cur_sel_group.prune_dead(|id| {
                objects
                    .get(id)
                    .filter(|o| o.is_live())
                    .map(|o| o.core().obj_type)
            });
            for g in self.player_side.groups.iter_mut() {
                g.prune_dead(|id| {
                    objects
                        .get(id)
                        .filter(|o| o.is_live())
                        .map(|o| o.core().obj_type)
                });
            }
        }

        // Drop dead handles from the UI selection set (the C++
        // RemoveFromSelection happens inside ReleaseMe).
        {
            let objects = &self.objects;
            self.player_side
                .selected
                .retain(|&id| objects.get(id).map(|o| o.is_live()).unwrap_or(false));
            if let Some(a) = self.player_side.active_object {
                if objects.get(a).map(|o| !o.is_live()).unwrap_or(true) {
                    self.player_side.active_object = self.player_side.selected.first().copied();
                }
            }
        }

        self.player_side.pump_groups();

        // `if((!GetCurGroup() || !cnt) && m_CurrSel in {GROUP,ROBOT,FLYER})
        //      Select(NOTHING, NULL);` (MatrixSide.cpp:490-493)
        let group_empty = self
            .player_side
            .cur_group()
            .map(|g| g.objects_cnt() == 0)
            .unwrap_or(true);
        if group_empty
            && matches!(
                self.player_side.curr_sel,
                crate::matrix_game::side::CurrSel::RobotsSelected
            )
        {
            self.select_nothing();
        }

        self.takt_pl(map, -1);
        self.calc_max_speed(map);
    }
}

impl MapLogic {
    /// `Select(NOTHING, NULL)` (MatrixSide.cpp:1038-1042) — clear the
    /// active selection + groups.
    pub fn select_nothing(&mut self) {
        self.selected_group_unselect();
        self.player_side.clear();
    }

    /// `SelectedGroupUnselect` (MatrixSide.cpp:1374-1390).
    pub fn selected_group_unselect(&mut self) {
        let members: Vec<ObjectId> = self
            .player_side
            .cur_group()
            .map(|g| g.objects.iter().map(|o| o.object).collect())
            .unwrap_or_default();
        for id in members {
            if let Some(r) = robot_mut(&mut self.objects, id) {
                r.unselect();
            }
        }
        self.player_side.current_group = None;
    }

    /// `SelectedGroupBreakOrders` (MatrixSide.cpp:1412-1422).
    pub fn selected_group_break_orders(&mut self) {
        let members: Vec<ObjectId> = self
            .player_side
            .cur_group()
            .map(|g| g.objects.iter().map(|o| o.object).collect())
            .unwrap_or_default();
        for id in members {
            let is_robot = robot_ref(&self.objects, id).is_some();
            if is_robot {
                self.robot_break_all_orders(id);
            }
        }
    }

    /// `Reselect` (MatrixSide.cpp:1045-1068) — sync every robot's
    /// group-selection flag with current-group membership.
    pub fn reselect(&mut self) {
        let members: Vec<ObjectId> = self
            .player_side
            .cur_group()
            .map(|g| g.objects.iter().map(|o| o.object).collect())
            .unwrap_or_default();
        let all: Vec<ObjectId> = self
            .objects
            .iter_live()
            .filter(|&id| robot_ref(&self.objects, id).is_some())
            .collect();
        for id in all {
            let selected = members.contains(&id);
            if let Some(r) = robot_mut(&mut self.objects, id) {
                if selected {
                    r.select_by_group();
                } else {
                    r.unselect();
                }
            }
        }
    }

    /// `CreateGroupFromCurrent()` (MatrixSide.cpp:1454-1479) — turn
    /// the scratch selection group into the new current group.
    pub fn create_group_from_current(&mut self) {
        let members = std::mem::take(&mut self.player_side.cur_sel_group.objects);
        self.player_side.cur_sel_group.remove_all();
        let mut ng = crate::matrix_game::side::Group::default();
        for m in &members {
            // Remove from any saved group first (LIST walk at 1462-1469).
            let ty = self
                .objects
                .get(m.object)
                .map(|o| o.core().obj_type)
                .unwrap_or(ObjectType::Empty);
            for g in self.player_side.groups.iter_mut() {
                g.remove_object(m.object, ty);
            }
            ng.add_object(m.object, -4, ty);
        }
        self.player_side.groups.push(ng);
        self.player_side.current_group = Some(self.player_side.groups.len() - 1);
        self.reselect();
    }

    /// `CreateGroupFromCurrent(CMatrixMapStatic*)` (MatrixSide.cpp:
    /// 1480-1499) — single-object variant.
    pub fn create_group_from_object(&mut self, obj: ObjectId) {
        self.player_side.cur_sel_group.remove_all();
        let ty = self
            .objects
            .get(obj)
            .map(|o| o.core().obj_type)
            .unwrap_or(ObjectType::Empty);
        let mut ng = crate::matrix_game::side::Group::default();
        for g in self.player_side.groups.iter_mut() {
            g.remove_object(obj, ty);
        }
        ng.add_object(obj, -4, ty);
        self.player_side.groups.push(ng);
        self.player_side.current_group = Some(self.player_side.groups.len() - 1);
        self.reselect();
    }

    /// `AddToCurrentGroup` (MatrixSide.cpp:1501-1517) — fold the
    /// scratch selection into the current group (9-robot cap).
    pub fn add_to_current_group(&mut self) {
        if self.player_side.current_group.is_none() {
            return;
        }
        let members = std::mem::take(&mut self.player_side.cur_sel_group.objects);
        self.player_side.cur_sel_group.remove_all();
        for m in &members {
            let ty = self
                .objects
                .get(m.object)
                .map(|o| o.core().obj_type)
                .unwrap_or(ObjectType::Empty);
            if let Some(g) = self.player_side.cur_group_mut() {
                if !g.find_object(m.object) && g.objects_cnt() < 9 {
                    g.add_object(m.object, -4, ty);
                }
            }
        }
        self.reselect();
    }

    /// `RemoveObjectFromSelectedGroup` (MatrixSide.cpp:1557-1580).
    pub fn remove_object_from_selected_group(&mut self, o: ObjectId) {
        let ty = self
            .objects
            .get(o)
            .map(|x| x.core().obj_type)
            .unwrap_or(ObjectType::Empty);
        if let Some(r) = robot_mut(&mut self.objects, o) {
            r.unselect();
        }
        if let Some(g) = self.player_side.cur_group_mut() {
            g.remove_object(o, ty);
        }
        self.player_side.cur_sel_group.remove_object(o, ty);
        let i = self.player_side.cur_sel_num;
        self.player_side.cur_sel_num = i.max(0);
        if let Some(g) = self.player_side.cur_group() {
            if self.player_side.cur_sel_num >= g.objects_cnt() {
                self.player_side.cur_sel_num = (g.objects_cnt() - 1).max(0);
            }
        }
        self.reselect();
    }

    /// `ShowOrderState` (MatrixSide.cpp:1070-1104) — returns the
    /// (auto_capture, auto_attack, auto_defence) flags for the UI's
    /// AUTO_* button states.
    pub fn show_order_state(&self) -> (bool, bool, bool) {
        let mut auto = (false, false, false);
        let Some(g) = self.player_side.cur_group() else {
            return auto;
        };
        for m in &g.objects {
            let Some(r) = robot_ref(&self.objects, m.object) else {
                continue;
            };
            if !r.is_live() || r.group_logic < 0 {
                continue;
            }
            match self
                .player_side
                .player_groups
                .get(r.group_logic as usize)
                .map(|g| g.order)
            {
                Some(PlayerOrder::AutoCapture) => auto.0 = true,
                Some(PlayerOrder::AutoAttack) => auto.1 = true,
                Some(PlayerOrder::AutoDefence) => auto.2 = true,
                _ => {}
            }
        }
        auto
    }

    /// Rebuild the group machinery from the UI-side `selected` set —
    /// the bridge between the existing click/marquee handlers (which
    /// maintain `Side::selected`) and the C++ CMatrixGroup flow
    /// (cur-sel-group → CreateGroupFromCurrent → Reselect).
    pub fn sync_group_from_selection(&mut self) {
        let side_id = self.player_side.id;
        let robots: Vec<ObjectId> = self
            .player_side
            .selected
            .iter()
            .copied()
            .filter(|&id| {
                robot_ref(&self.objects, id)
                    .map(|r| r.is_live() && r.side == side_id)
                    .unwrap_or(false)
            })
            .take(9)
            .collect();
        if robots.is_empty() {
            self.selected_group_unselect();
            return;
        }
        self.player_side.cur_sel_group.remove_all();
        for id in robots {
            self.player_side
                .cur_sel_group
                .add_object(id, -4, ObjectType::RobotAi);
        }
        self.create_group_from_current();
        self.player_side.cur_sel_num = 0;
    }
}

impl MapLogic {
    /// Mirror the current group back into the UI-side `selected` set
    /// (active object = first member).
    fn mirror_selection_from_group(&mut self) {
        let members: Vec<ObjectId> = self
            .player_side
            .cur_group()
            .map(|g| g.objects.iter().map(|o| o.object).collect())
            .unwrap_or_default();
        let primary = members.first().copied();
        self.player_side.select_replace(
            members,
            primary,
            crate::matrix_game::side::CurrSel::RobotsSelected,
        );
        // `Select(GROUP, ...)` resets the personal-panel cursor to the
        // first member (SetCurSelNum(0), MatrixSide.cpp:1020/1032).
        self.player_side.cur_sel_num = 0;
    }

    /// Port of `CMatrixSideUnit::OnLButtonDouble` (MatrixSide.cpp:
    /// 750-784) — double-click on an own robot selects every own
    /// robot within FRIENDLY_SEARCH_RADIUS.
    pub fn on_left_double_click(
        &mut self,
        camera: &crate::matrix_game::camera::Camera,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> bool {
        use crate::matrix_game::common::TRACE_ANYOBJECT;
        use crate::matrix_game::side::FRIENDLY_SEARCH_RADIUS;

        let (origin, dir) = camera.screen_to_world_ray(sx, sy, screen_w, screen_h);
        let Some((hit, _)) = self
            .objects
            .pick_object(origin, dir, TRACE_ANYOBJECT, None)
        else {
            return false;
        };
        let side_id = self.player_side.id;
        let Some(center) = robot_ref(&self.objects, hit)
            .filter(|r| r.is_live() && r.side == side_id)
            .map(|r| r.core().geo_center)
        else {
            return false;
        };

        if self.player_side.current_group.is_some() {
            self.selected_group_unselect();
            self.player_side.cur_sel_group.remove_all();
        }

        let nearby: Vec<ObjectId> = self
            .objects
            .iter_live()
            .filter(|&id| {
                robot_ref(&self.objects, id)
                    .map(|r| {
                        r.is_live()
                            && r.side == side_id
                            && (r.core().geo_center - center).length_squared()
                                <= FRIENDLY_SEARCH_RADIUS * FRIENDLY_SEARCH_RADIUS
                    })
                    .unwrap_or(false)
            })
            .collect();
        for id in nearby {
            self.player_side
                .cur_sel_group
                .add_object(id, -4, ObjectType::RobotAi);
        }
        self.create_group_from_current();
        self.mirror_selection_from_group();
        true
    }
}

impl MapLogic {
    /// Port of `CMatrixSideUnit::OrderFlyer` (MatrixSide.cpp:385-397)
    /// + the FO_GIVE_BOT robot creation inside `ApplyOrder`
    /// (MatrixFlyer.cpp:1196-1250): spawn the transport, build the
    /// supply robot from the GiveBot entry, hook it underneath, and
    /// give it an attack-move at the drop place.
    pub fn order_flyer(&mut self, map: &GameMap, to: glam::Vec2, ang: f32, place: i32, botpar_i: usize) {
        use crate::matrix_game::flyer::Flyer;
        use crate::matrix_game::interface::constructor::{ArmorUnit, RobotConfig, Unit};
        use crate::matrix_game::config::RobotUnitKind;
        use crate::matrix_game::robot::Robot;

        // `MAX_ALWAYS_DRAW_OBJ` cap (MatrixSide.cpp:388) — at most 16
        // live supply flyers at once.
        const MAX_ALWAYS_DRAW_OBJ: usize = 16;
        let live_flyers = self
            .objects
            .iter_live()
            .filter(|&id| {
                self.objects
                    .get(id)
                    .map(|o| {
                        matches!(
                            o.core().obj_type,
                            crate::matrix_game::map_static::ObjectType::Flyer
                        )
                    })
                    .unwrap_or(false)
            })
            .count();
        if live_flyers >= MAX_ALWAYS_DRAW_OBJ {
            return;
        }

        let cfg = crate::matrix_game::config::give_bot_config();
        let Some(entry) = cfg.bots.get(botpar_i).cloned() else {
            return;
        };
        let dist = cfg.distance + (self.rng.float01() as f32 * 2.0 - 1.0) * 150.0;

        let side_id = self.player_side.id;
        let mut flyer = Flyer::new_give_bot(
            map,
            to,
            side_id,
            ang,
            dist,
            cfg.height,
            cfg.land_height,
            cfg.flyer_kind,
            cfg.hitpoint,
        );
        let spawn_pos = flyer.pos;
        flyer.carry.robot_mass_factor = 1.0; // quick_connect = true

        let fid = self.objects.spawn(Box::new(flyer));
        if let Some(f) = crate::matrix_game::logic::flyer_mut(&mut self.objects, fid) {
            f.self_id = Some(fid);
        }
        self.objects.add_lt(fid);

        // SSpecialBot::GetRobot equivalent — config from kinds.
        let mut rc = RobotConfig::new();
        rc.chassis = Unit {
            ty: crate::matrix_game::object_robot::RobotUnitType::Chassis,
            kind: RobotUnitKind(entry.chassis),
            price: Default::default(),
        };
        rc.hull = ArmorUnit {
            max_common_weapon_cnt: 0,
            max_extra_weapon_cnt: 0,
            unit: Unit {
                ty: crate::matrix_game::object_robot::RobotUnitType::Armor,
                kind: RobotUnitKind(entry.armor),
                price: Default::default(),
            },
        };
        if entry.head > 0 {
            rc.head = Unit {
                ty: crate::matrix_game::object_robot::RobotUnitType::Head,
                kind: RobotUnitKind(entry.head),
                price: Default::default(),
            };
        }
        for (i, &w) in entry.weapons.iter().take(rc.weapon.len()).enumerate() {
            rc.weapon[i] = Unit {
                ty: crate::matrix_game::object_robot::RobotUnitType::Weapon,
                kind: RobotUnitKind(w),
                price: Default::default(),
            };
        }
        let chassis = crate::matrix_game::object_building::chassis_from_kind(rc.chassis.kind)
            .unwrap_or(crate::matrix_game::robot::ChassisKind::Track);
        let mut robot = Robot::new(spawn_pos, side_id, chassis);
        robot.config = rc;
        robot.pos_z = spawn_pos.z;
        robot.strength = entry.strength;
        if entry.hitpoint > 0.0 {
            robot.hit_point_max = entry.hitpoint;
            robot.hit_point = entry.hitpoint;
        }
        robot.carry_attach(fid);
        let rid = self.objects.spawn(Box::new(robot));
        if let Some(r) = robot_mut(&mut self.objects, rid) {
            r.self_id = Some(rid);
        }
        self.objects.add_lt(rid);
        if let Some(f) = crate::matrix_game::logic::flyer_mut(&mut self.objects, fid) {
            f.carrying = Some(rid);
        }

        // PGOrderAttack at the drop place, sound-muted
        // (MatrixFlyer.cpp:1243-1246).
        if let Some(pp) = place_pos(map, place) {
            self.sound_order_attack_disable = true;
            let no = self.robot_to_logic_group(rid);
            self.pg_order_attack(map, no, pp, None);
            self.sound_order_attack_disable = false;
        }
        if let Some(r) = robot_mut(&mut self.objects, rid) {
            r.env.place = place;
        }
    }

    /// Port of `CMatrixBuilding::Maintenance` (MatrixObjectBuilding.
    /// cpp:1265-1352) — the supply-drop dispatcher for `building`.
    pub fn building_maintenance(&mut self, map: &GameMap, building: ObjectId) {
        let Some(b) = building_ref(&self.objects, building) else {
            return;
        };
        if b.side == 0 || self.maintenance_disabled() || self.maintenance_time > 0 {
            return;
        }
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let mut cx = float2int(b.pos.x / gsm);
        let mut cy = float2int(b.pos.y / gsm);
        if b.is_base() {
            match b.angle & 3 {
                0 => cy += 6,
                1 => cx -= 7,
                2 => cy -= 7,
                _ => cx += 6,
            }
        }
        // C++ PlaceFindNear adjusts cx,cy by reference before PlaceList
        // floods from them (MatrixObjectBuilding.cpp:1285-1292).
        if let Some((nx, ny)) = logic::place_find_near_with_blockers(map, 0, 4, cx, cy, &[]) {
            cx = nx;
            cy = ny;
        }

        let mut list: Vec<i32> = Vec::new();
        let (ret, _) = place_list(map, 0x1f, (cx, cy), (cx, cy), 100, false, &mut list);
        if ret == 0 {
            return;
        }
        self.init_maintenance_time();
        // S_MAINTENANCE (MatrixObjectBuilding.cpp:1295).
        self.objects.queue_snd("s_maintenance");

        // Mark occupied (MatrixObjectBuilding.cpp:1298-1310).
        zero_places(map, &list);
        mark_occupied_places(map, &self.objects);

        let angle = (self.rng.float01() as f32 * 2.0 - 1.0) * std::f32::consts::PI;
        let cfg = crate::matrix_game::config::give_bot_config();
        if cfg.bots.is_empty() {
            return;
        }

        let mut score = 0i32;
        let mut pli = 0usize;
        while score < 100 {
            if pli >= list.len() {
                break;
            }
            let place = list[pli];
            pli += 1;
            let Some((data, mv, pos, _)) = place_get(map, place) else {
                continue;
            };
            if data != 0 || mv & 0x1f != 0 {
                continue;
            }

            let mut botpar_i = 0usize;
            let mut sc = 0i32;
            for _ in 0..10 {
                botpar_i = self.rng.range(0, cfg.bots.len() as i32 - 1) as usize;
                sc = cfg.bots[botpar_i].score;
                if score + sc > 130 {
                    continue;
                }
                break;
            }
            score += sc;
            if sc > 130 {
                break;
            }

            let to = glam::Vec2::new(
                pos.0 as f32 * gsm + ROBOT_MOVECELLS_PER_SIZE as f32 * gsm / 2.0,
                pos.1 as f32 * gsm + ROBOT_MOVECELLS_PER_SIZE as f32 * gsm / 2.0,
            );
            self.order_flyer(map, to, angle, place, botpar_i);
        }
    }
}

impl MapLogic {
    fn screen_move_cell(
        &self,
        camera: &crate::matrix_game::camera::Camera,
        map: &GameMap,
        sx: f32,
        sy: f32,
        w: f32,
        h: f32,
    ) -> Option<(i32, i32)> {
        let (wx, wy) = logic::screen_to_terrain_xy(camera, map, sx, sy, w, h)?;
        let (cmx, cmy) = map.world_to_move(wx, wy);
        Some((
            cmx - ROBOT_MOVECELLS_PER_SIZE / 2,
            cmy - ROBOT_MOVECELLS_PER_SIZE / 2,
        ))
    }

    fn screen_pick(
        &self,
        camera: &crate::matrix_game::camera::Camera,
        sx: f32,
        sy: f32,
        w: f32,
        h: f32,
    ) -> Option<ObjectId> {
        use crate::matrix_game::common::TRACE_ANYOBJECT;
        let (origin, dir) = camera.screen_to_world_ray(sx, sy, w, h);
        self.objects
            .pick_object(origin, dir, TRACE_ANYOBJECT, None)
            .map(|(id, _)| id)
    }

    /// PREORDER_CAPTURE click (MatrixSide.cpp:715-721). Returns false
    /// when the click wasn't a valid capture target (order stays armed).
    pub fn order_capture_at_screen(
        &mut self,
        camera: &crate::matrix_game::camera::Camera,
        sx: f32,
        sy: f32,
        w: f32,
        h: f32,
        map: &GameMap,
    ) -> bool {
        let Some(id) = self.screen_pick(camera, sx, sy, w, h) else {
            return false;
        };
        let valid = building_ref(&self.objects, id)
            .map(|b| b.is_live() && b.side != self.player_side.id)
            .unwrap_or(false);
        if !valid {
            return false;
        }
        let no = self.sel_group_to_logic_group();
        self.pg_order_capture(map, no, id);
        true
    }

    /// PREORDER_PATROL click (MatrixSide.cpp:723-726).
    pub fn order_patrol_at_screen(
        &mut self,
        camera: &crate::matrix_game::camera::Camera,
        sx: f32,
        sy: f32,
        w: f32,
        h: f32,
        map: &GameMap,
    ) {
        let Some(tp) = self.screen_move_cell(camera, map, sx, sy, w, h) else {
            return;
        };
        let no = self.sel_group_to_logic_group();
        self.pg_order_patrol(map, no, tp);
    }

    /// PREORDER_BOMB click (MatrixSide.cpp:727-737).
    pub fn order_bomb_at_screen(
        &mut self,
        camera: &crate::matrix_game::camera::Camera,
        sx: f32,
        sy: f32,
        w: f32,
        h: f32,
        map: &GameMap,
    ) {
        let target = self
            .screen_pick(camera, sx, sy, w, h)
            .filter(|&id| crate::matrix_game::logic::is_live_or_special(&self.objects, id));
        match target {
            Some(id) => {
                let Some(tp) = get_map_pos(&self.objects, id) else {
                    return;
                };
                let no = self.sel_group_to_logic_group();
                self.pg_order_bomb(map, no, tp, Some(id));
            }
            None => {
                let Some(tp) = self.screen_move_cell(camera, map, sx, sy, w, h) else {
                    return;
                };
                let no = self.sel_group_to_logic_group();
                self.pg_order_bomb(map, no, tp, None);
            }
        }
    }

    /// PREORDER_REPAIR click (MatrixSide.cpp:738-745). Returns false
    /// for invalid targets (order stays armed).
    pub fn order_repair_at_screen(
        &mut self,
        camera: &crate::matrix_game::camera::Camera,
        sx: f32,
        sy: f32,
        w: f32,
        h: f32,
        map: &GameMap,
    ) -> bool {
        let Some(id) = self.screen_pick(camera, sx, sy, w, h) else {
            return false;
        };
        let valid = self
            .objects
            .get(id)
            .map(|o| o.is_live() && o.side() == self.player_side.id)
            .unwrap_or(false);
        if !valid {
            return false;
        }
        let no = self.sel_group_to_logic_group();
        self.pg_order_repair(map, no, id);
        true
    }
}

impl MapLogic {
    /// Move order at a world position (the minimap right-click path,
    /// MatrixSide.cpp:821-847).
    pub fn order_move_to_world(&mut self, map: &GameMap, w: glam::Vec2) {
        use crate::matrix_game::side::CurrSel;
        if !matches!(self.player_side.curr_sel, CurrSel::RobotsSelected) {
            return;
        }
        let (cmx, cmy) = map.world_to_move(w.x, w.y);
        let tp = (
            cmx - ROBOT_MOVECELLS_PER_SIZE / 2,
            cmy - ROBOT_MOVECELLS_PER_SIZE / 2,
        );
        let no = self.sel_group_to_logic_group();
        self.pg_order_move_to(map, no, tp);
    }

    /// Port of `CMatrixSideUnit::AssignPlace(robot, region)`
    /// (MatrixSide.cpp:5270-5301) — pick the first free, standable
    /// place in `region` for this robot (keeping the current place if
    /// it's already in the region).
    fn assign_place_robot(&mut self, map: &GameMap, rid: ObjectId, region: i32) {
        if region < 0 {
            return;
        }
        let (cur_place, cbit) = match robot_ref(&self.objects, rid) {
            Some(r) => (r.env.place, chassis_bit(r)),
            None => return,
        };
        let place_all: Vec<i32> = {
            let Some(rn_lock) = map.road_network.as_ref() else {
                return;
            };
            let rn = rn_lock.lock().unwrap();
            let Some(reg) = rn.regions.get(region as usize) else {
                return;
            };
            if cur_place >= 0 && reg.place_all.contains(&cur_place) {
                return;
            }
            reg.place_all.clone()
        };
        zero_places(map, &place_all);
        mark_occupied_places_skip(map, &self.objects, Some(rid));
        for &pi in &place_all {
            let Some((data, mv, _, _)) = place_get(map, pi) else {
                continue;
            };
            if data != 0 || mv & cbit != 0 {
                continue;
            }
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.env.place = pi;
            }
            place_set_data(map, pi, 1);
            break;
        }
    }

    /// RobotSpawn rally (MatrixRobot.cpp:2216-2221): each freshly
    /// produced player robot gets a place in the base's region and a
    /// sound-muted attack-move to it (or to the base itself when no
    /// place was found).
    pub fn drain_spawn_rallies(&mut self, map: &GameMap) {
        if self.objects.pending_spawn_rallies.is_empty() {
            return;
        }
        let gsm = GameMap::GLOBAL_SCALE_MOVE;
        let rids: Vec<ObjectId> = self.objects.pending_spawn_rallies.drain(..).collect();
        for rid in rids {
            let Some(base_id) = robot_ref(&self.objects, rid).and_then(|r| r.base) else {
                continue;
            };
            let Some(bpos) = building_ref(&self.objects, base_id).map(|b| b.pos) else {
                continue;
            };
            let bp = ((bpos.x / gsm) as i32, (bpos.y / gsm) as i32);
            let region = get_region(map, bp.0, bp.1);
            self.assign_place_robot(map, rid, region);
            let place = robot_ref(&self.objects, rid).map(|r| r.env.place).unwrap_or(-1);
            let tp = place_get(map, place).map(|p| p.2).unwrap_or(bp);
            self.sound_order_attack_disable = true;
            let no = self.robot_to_logic_group(rid);
            self.pg_order_attack(map, no, tp, None);
            self.sound_order_attack_disable = false;
        }
    }

    /// World-position move cell with the footprint offset — the
    /// `CalcMinimap2World`-substituted coordinates in the C++
    /// pre-order dispatch (MatrixSide.cpp:672-680).
    fn world_move_cell(&self, map: &GameMap, w: glam::Vec2) -> (i32, i32) {
        let (cmx, cmy) = map.world_to_move(w.x, w.y);
        (
            cmx - ROBOT_MOVECELLS_PER_SIZE / 2,
            cmy - ROBOT_MOVECELLS_PER_SIZE / 2,
        )
    }

    /// PREORDER_FIRE at a minimap world position — always attack-move
    /// to landscape (the minimap substitutes TRACE_STOP_LANDSCAPE).
    pub fn order_attack_world(&mut self, map: &GameMap, w: glam::Vec2) {
        let tp = self.world_move_cell(map, w);
        let no = self.sel_group_to_logic_group();
        self.pg_order_attack(map, no, tp, None);
    }

    /// PREORDER_PATROL at a minimap world position.
    pub fn order_patrol_world(&mut self, map: &GameMap, w: glam::Vec2) {
        let tp = self.world_move_cell(map, w);
        let no = self.sel_group_to_logic_group();
        self.pg_order_patrol(map, no, tp);
    }

    /// PREORDER_BOMB at a minimap world position.
    pub fn order_bomb_world(&mut self, map: &GameMap, w: glam::Vec2) {
        let tp = self.world_move_cell(map, w);
        let no = self.sel_group_to_logic_group();
        self.pg_order_bomb(map, no, tp, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_game::robot::{ChassisKind, Robot, RobotState};
    use crate::matrix_game::side::{CurrSel, PlayerOrder, SideStatus};

    fn spawn_robot(game: &mut MapLogic, side: i32) -> ObjectId {
        let mut r = Robot::new(glam::Vec3::new(100.0, 100.0, 0.0), side, ChassisKind::Track);
        r.state = RobotState::Idle;
        let id = game.objects.spawn(Box::new(r));
        if let Some(r) = robot_mut(&mut game.objects, id) {
            r.self_id = Some(id);
        }
        game.objects.add_lt(id);
        id
    }

    #[test]
    fn sync_group_builds_current_group_and_flags() {
        let mut game = MapLogic::with_seed(7);
        let a = spawn_robot(&mut game, 1);
        let b = spawn_robot(&mut game, 1);
        game.player_side
            .select_replace(vec![a, b], Some(a), CurrSel::RobotsSelected);
        game.sync_group_from_selection();
        let g = game.player_side.cur_group().expect("group");
        assert_eq!(g.objects_cnt(), 2);
        assert!(g.find_object(a) && g.find_object(b));
        assert!(robot_ref(&game.objects, a).unwrap().is_selected());
        assert!(robot_ref(&game.objects, b).unwrap().is_selected());
    }

    #[test]
    fn alloc_logic_group_reuses_empty_slots() {
        let mut game = MapLogic::with_seed(7);
        let a = spawn_robot(&mut game, 1);
        let no = game.robot_to_logic_group(a);
        assert_eq!(robot_ref(&game.objects, a).unwrap().group_logic, no as i32);
        assert_eq!(game.player_side.player_groups[no].order, PlayerOrder::Stop);
        // Re-allocation moves the robot to a fresh slot (the census
        // still counts it in the old one — same as the C++).
        let no2 = game.robot_to_logic_group(a);
        assert_ne!(no, no2);
        assert_eq!(robot_ref(&game.objects, a).unwrap().group_logic, no2 as i32);
        assert_eq!(game.player_side.player_groups[no2].robot_cnt, 1);
    }

    #[test]
    fn group_unselect_clears_flags() {
        let mut game = MapLogic::with_seed(7);
        let a = spawn_robot(&mut game, 1);
        game.player_side
            .select_replace(vec![a], Some(a), CurrSel::RobotsSelected);
        game.sync_group_from_selection();
        assert!(robot_ref(&game.objects, a).unwrap().is_selected());
        game.select_nothing();
        assert!(!robot_ref(&game.objects, a).unwrap().is_selected());
        assert!(game.player_side.cur_group().is_none());
        assert!(game.player_side.selected.is_empty());
    }

    #[test]
    fn check_status_declares_win_when_last_side_standing() {
        let mut game = MapLogic::with_seed(7);
        let _own = spawn_robot(&mut game, 1);
        let enemy = spawn_robot(&mut game, 2);
        game.ensure_sides_from_objects();
        assert_eq!(game.other_sides.len(), 1);

        // Both alive — no result.
        game.elapsed_ms = 5000;
        game.check_status();
        assert_eq!(game.game_over, None);

        // Kill the enemy robot.
        if let Some(r) = robot_mut(&mut game.objects, enemy) {
            r.state = RobotState::Dip;
        }
        game.elapsed_ms = 12000;
        // Same pass: enemy → JustDead → consumed to None, leaving the
        // player as the only active side → win armed (the C++ Takt
        // does all three in one CheckStatus too).
        game.check_status();
        assert_eq!(game.side_by_id(2).map(|s| s.status), Some(SideStatus::None));
        assert!(game.win_flag);
        game.elapsed_ms = 20000;
        game.check_status(); // dialog entry
        assert_eq!(game.game_over, Some(true));
    }

    #[test]
    fn break_all_orders_clears_pool_and_env() {
        let mut game = MapLogic::with_seed(7);
        let a = spawn_robot(&mut game, 1);
        let b = spawn_robot(&mut game, 2);
        if let Some(r) = robot_mut(&mut game.objects, a) {
            r.move_to(10, 10);
            r.env.target = Some(b);
            r.env.target_attack = Some(b);
        }
        game.robot_break_all_orders(a);
        let r = robot_ref(&game.objects, a).unwrap();
        assert!(r.orders.is_empty());
        assert_eq!(r.env.target, None);
        assert_eq!(r.env.target_attack, None);
    }
}
