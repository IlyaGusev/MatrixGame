//! Port of `Logic/MatrixEnvironment.{h,cpp}` — `CEnemy` + `CInfo`,
//! the per-robot sensing state (`CMatrixRobotAI::m_Environment`).
//!
//! The C++ keeps enemies in an intrusive doubly-linked list
//! (`m_FirstEnemy`..`m_LastEnemy`, LIST_ADD appends at the tail); a
//! `Vec` in push order preserves the identical traversal. Raw
//! `CMatrixMapStatic*` pointers become `ObjectId` handles — a freed
//! object simply stops resolving, but the *list bookkeeping* still
//! mirrors the original (entries are removed by the same
//! `RemoveFromList` call sites, not garbage-collected implicitly).

use crate::matrix_game::map_static::ObjectId;

/// `ENEMY_UNDEF` / `ENEMY_ANY` (MatrixRule.h). `ClassifyEnemy` always
/// assigns `ENEMY_ANY` (MatrixEnvironment.cpp:26-29).
pub const ENEMY_UNDEF: u32 = 0;
pub const ENEMY_ANY: u32 = 1;

/// Port of `CEnemy` (MatrixEnvironment.h:10).
#[derive(Clone, Copy, Debug)]
pub struct Enemy {
    pub enemy: ObjectId,
    pub kind: u32,
    /// `m_DelSlowly` — RemoveFromListSlowly strike counter; the entry
    /// is dropped on the third strike (MatrixEnvironment.cpp:132-151).
    pub del_slowly: i32,
}

/// Port of `CInfo` (MatrixEnvironment.h:33). All time fields are
/// `GetTime()` milliseconds like the C++.
#[derive(Clone, Debug)]
pub struct Info {
    pub target: Option<ObjectId>,
    pub target_attack: Option<ObjectId>,
    /// `m_TargetLast` — detects target changes.
    pub target_last: Option<ObjectId>,
    /// `m_TargetChange` — time the target last changed.
    pub target_change: i32,
    /// `m_TargetChangeRepair` — time the repair target last changed.
    pub target_change_repair: i32,
    pub target_angle: f32,

    /// `m_Place` — road-network place the robot stands in / walks to.
    pub place: i32,
    /// `m_PlaceAdd` — player-assigned fallback coordinate;
    /// `CPoint(-1,-1)` = none. Kept as raw ints because call sites
    /// compare `m_PlaceAdd.x >= 0`.
    pub place_add: (i32, i32),
    /// `m_PlaceNotFound` — time of the last failed place search.
    pub place_not_found: i32,
    /// `m_OrderNoBreak` — an unbreakable order is in flight.
    pub order_no_break: bool,
    pub last_fire: i32,
    pub last_hit_target: i32,
    pub last_hit_enemy: i32,
    pub last_hit_friendly: i32,

    /// `m_BadPlace[8]` ring (MatrixEnvironment.cpp:208-227).
    bad_places: Vec<i32>,
    /// `m_BadCoord[16]` ring.
    bad_coords: Vec<(i32, i32)>,
    /// `m_Ignore[16]` + `m_IgnoreTime[16]` — targets ignored for 5 s.
    ignore: Vec<(Option<ObjectId>, i32)>,

    pub enemies: Vec<Enemy>,
}

const BAD_PLACES_MAX: usize = 8;
const BAD_COORDS_MAX: usize = 16;
const IGNORE_MAX: usize = 16;
/// Ignore entries expire after 5000 ms (MatrixEnvironment.cpp:257,281).
const IGNORE_TTL: i32 = 5000;

impl Default for Info {
    fn default() -> Self {
        Self::new()
    }
}

impl Info {
    pub fn new() -> Self {
        Self {
            target: None,
            target_attack: None,
            target_last: None,
            target_change: 0,
            target_change_repair: 0,
            target_angle: 0.0,
            place: -1,
            place_add: (-1, -1),
            place_not_found: -1_000_000,
            order_no_break: false,
            last_fire: 0,
            last_hit_target: 0,
            last_hit_enemy: 0,
            last_hit_friendly: 0,
            bad_places: Vec::new(),
            bad_coords: Vec::new(),
            ignore: Vec::new(),
            enemies: Vec::new(),
        }
    }

    /// `CInfo::TargetType` (MatrixEnvironment.h:73): 0 = no target,
    /// 1 = target is in the enemy list, 2 = friendly/other target.
    pub fn target_type(&self) -> i32 {
        match self.target {
            None => 0,
            Some(t) if self.search_enemy(t) => 1,
            Some(_) => 2,
        }
    }

    pub fn enemy_cnt(&self) -> i32 {
        self.enemies.len() as i32
    }

    /// `CInfo::SearchEnemy` — true if `ms` is in the enemy list.
    pub fn search_enemy(&self, ms: ObjectId) -> bool {
        self.enemies.iter().any(|e| e.enemy == ms)
    }

    /// `CInfo::AddToList` (MatrixEnvironment.cpp:153). The C++ does
    /// NOT dedup here — callers guard with SearchEnemy first.
    pub fn add_to_list(&mut self, ms: ObjectId) {
        self.del_ignore(ms);
        self.enemies.push(Enemy {
            enemy: ms,
            kind: ENEMY_ANY,
            del_slowly: 0,
        });
    }

    /// `CInfo::AddToListSlowly` (MatrixEnvironment.cpp:164) — resets
    /// the slow-delete counter if already present, else appends.
    pub fn add_to_list_slowly(&mut self, ms: ObjectId) {
        self.del_ignore(ms);
        if let Some(e) = self.enemies.iter_mut().find(|e| e.enemy == ms) {
            e.del_slowly = 0;
            return;
        }
        self.enemies.push(Enemy {
            enemy: ms,
            kind: ENEMY_ANY,
            del_slowly: 0,
        });
    }

    /// `CInfo::RemoveFromList(CMatrixMapStatic*)`.
    pub fn remove_from_list(&mut self, ms: ObjectId) {
        if self.target == Some(ms) {
            self.target = None;
        }
        if self.target_attack == Some(ms) {
            self.target_attack = None;
        }
        self.enemies.retain(|e| e.enemy != ms);
    }

    /// `CInfo::RemoveFromListSlowly` — third strike removes.
    pub fn remove_from_list_slowly(&mut self, ms: ObjectId) {
        let Some(pos) = self.enemies.iter().position(|e| e.enemy == ms) else {
            if self.target == Some(ms) {
                self.target = None;
            }
            if self.target_attack == Some(ms) {
                self.target_attack = None;
            }
            return;
        };
        self.enemies[pos].del_slowly += 1;
        if self.enemies[pos].del_slowly >= 3 {
            if self.target == Some(ms) {
                self.target = None;
            }
            if self.target_attack == Some(ms) {
                self.target_attack = None;
            }
            self.enemies.remove(pos);
        }
    }

    /// `CInfo::RemoveAllBuilding` — `is_building(id)` resolves the
    /// object type at the call site (the env itself has no arena
    /// access).
    pub fn remove_all_building(
        &mut self,
        skip: Option<ObjectId>,
        mut is_building: impl FnMut(ObjectId) -> bool,
    ) {
        if let Some(t) = self.target {
            if Some(t) != skip && is_building(t) {
                self.target = None;
            }
        }
        if let Some(t) = self.target_attack {
            if Some(t) != skip && is_building(t) {
                self.target_attack = None;
            }
        }
        self.enemies
            .retain(|e| Some(e.enemy) == skip || !is_building(e.enemy));
    }

    /// `CInfo::RemoveAllSlowely` — drops every entry with a pending
    /// slow-delete strike.
    pub fn remove_all_slowly(&mut self) {
        let target = &mut self.target;
        let target_attack = &mut self.target_attack;
        self.enemies.retain(|e| {
            if e.del_slowly != 0 {
                if *target == Some(e.enemy) {
                    *target = None;
                }
                if *target_attack == Some(e.enemy) {
                    *target_attack = None;
                }
                false
            } else {
                true
            }
        });
    }

    /// `CInfo::Reset`.
    pub fn reset(&mut self) {
        self.target = None;
        self.target_attack = None;
    }

    /// `CInfo::Clear` — drop the whole enemy list.
    pub fn clear(&mut self) {
        self.enemies.clear();
    }

    /// `CInfo::AddBadPlace` — 8-entry FIFO ring.
    pub fn add_bad_place(&mut self, place: i32) {
        if self.is_bad_place(place) {
            return;
        }
        if self.bad_places.len() >= BAD_PLACES_MAX {
            self.bad_places.remove(0);
        }
        self.bad_places.push(place);
    }

    pub fn is_bad_place(&self, place: i32) -> bool {
        self.bad_places.contains(&place)
    }

    /// `CInfo::AddBadCoord` — 16-entry FIFO ring.
    pub fn add_bad_coord(&mut self, coord: (i32, i32)) {
        if self.is_bad_coord(coord) {
            return;
        }
        if self.bad_coords.len() >= BAD_COORDS_MAX {
            self.bad_coords.remove(0);
        }
        self.bad_coords.push(coord);
    }

    pub fn is_bad_coord(&self, coord: (i32, i32)) -> bool {
        self.bad_coords.contains(&coord)
    }

    /// Reset the step-aside bad-coord ring (`m_BadCoordCnt=0`) once a
    /// robot stops colliding (MatrixRobot.cpp:377).
    pub fn clear_bad_coords(&mut self) {
        self.bad_coords.clear();
    }

    /// The accumulated bad-coord ring (`m_BadCoord` walk in
    /// `PlaceFindNearReturn`, MatrixLogic.cpp:766-773).
    pub fn bad_coords(&self) -> &[(i32, i32)] {
        &self.bad_coords
    }

    /// `CInfo::AddIgnore` — refresh if present, else reuse an expired
    /// slot, else append (16 cap with FIFO shift).
    pub fn add_ignore(&mut self, ms: ObjectId, now: i32) {
        let mut empty = None;
        for (i, (id, t)) in self.ignore.iter().enumerate() {
            if *id == Some(ms) {
                self.ignore[i].1 = now;
                return;
            }
            if empty.is_none() && (id.is_none() || now - *t >= IGNORE_TTL) {
                empty = Some(i);
            }
        }
        if let Some(i) = empty {
            self.ignore[i] = (Some(ms), now);
        } else if self.ignore.len() < IGNORE_MAX {
            self.ignore.push((Some(ms), now));
        } else {
            self.ignore.remove(0);
            self.ignore.push((Some(ms), now));
        }
    }

    pub fn is_ignore(&self, ms: ObjectId, now: i32) -> bool {
        self.ignore
            .iter()
            .any(|(id, t)| *id == Some(ms) && now - *t < IGNORE_TTL)
    }

    pub fn del_ignore(&mut self, ms: ObjectId) {
        for (id, _) in self.ignore.iter_mut() {
            if *id == Some(ms) {
                *id = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u32) -> ObjectId {
        ObjectId::synthetic(n)
    }

    #[test]
    fn slow_remove_takes_three_strikes() {
        let mut info = Info::new();
        info.add_to_list(id(1));
        info.target = Some(id(1));
        info.remove_from_list_slowly(id(1));
        info.remove_from_list_slowly(id(1));
        assert!(info.search_enemy(id(1)));
        assert_eq!(info.target, Some(id(1)));
        info.remove_from_list_slowly(id(1));
        assert!(!info.search_enemy(id(1)));
        assert_eq!(info.target, None);
    }

    #[test]
    fn add_slowly_resets_strikes() {
        let mut info = Info::new();
        info.add_to_list(id(1));
        info.remove_from_list_slowly(id(1));
        info.add_to_list_slowly(id(1));
        assert_eq!(info.enemies.len(), 1);
        assert_eq!(info.enemies[0].del_slowly, 0);
    }

    #[test]
    fn ignore_expires_after_ttl() {
        let mut info = Info::new();
        info.add_ignore(id(7), 1000);
        assert!(info.is_ignore(id(7), 2000));
        assert!(!info.is_ignore(id(7), 6001));
    }

    #[test]
    fn bad_place_ring_caps_at_8() {
        let mut info = Info::new();
        for p in 0..10 {
            info.add_bad_place(p);
        }
        assert!(!info.is_bad_place(0));
        assert!(!info.is_bad_place(1));
        assert!(info.is_bad_place(2));
        assert!(info.is_bad_place(9));
    }
}
