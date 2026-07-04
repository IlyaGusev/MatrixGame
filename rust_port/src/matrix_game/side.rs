//! Partial port of `CMatrixSide` (MatrixSide.{cpp,hpp}).
//!
//! CMatrixSide in the original is ~thousands of lines covering
//! selection state, logical-group rosters, resource accounts, AI
//! planners, strategy orders, stats, and arcaded-robot handover.
//! This file holds the per-side data: selection state, resources,
//! stats, player groups, and the enemy-AI structures (logic groups,
//! teams, region stats) driven by `side_ai.rs`.
//!
//! What's here:
//! * `CurrSel` enum — mirrors the C++ per-side "what's currently
//!   selected" enum (MatrixSide.hpp).
//! * `Side` struct — minimal bookkeeping: `id`, `active_object`,
//!   `curr_sel`.
//!
//! Resources / stats / kill counters / logical groups etc. land when
//! their call sites need them.

use crate::matrix_game::config::{Resource, MAX_RESOURCES};
use crate::matrix_game::map_static::{ObjectId, ObjectType};
use crate::matrix_game::road_network::RoadRoute;

/// `MAX_ROBOTS` (MatrixSide.hpp:14).
pub const MAX_ROBOTS: usize = 60;
/// `MAX_LOGIC_GROUP` (MatrixSide.hpp:17).
pub const MAX_LOGIC_GROUP: usize = MAX_ROBOTS + 1;
/// `REGION_PATH_MAX_CNT` (MatrixSide.hpp:171).
pub const REGION_PATH_MAX_CNT: usize = 32;
/// `FRIENDLY_SEARCH_RADIUS` (MatrixSide.hpp:30).
pub const FRIENDLY_SEARCH_RADIUS: f32 = 400.0;
/// `MAX_TEAM_CNT` (MatrixSide.hpp:19).
pub const MAX_TEAM_CNT: usize = 3;

/// Port of `EMatrixLogicActionType` (MatrixSide.hpp:155-166).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum LogicActionType {
    #[default]
    None,
    Defence,
    Attack,
    Forward,
    Retreat,
    Capture,
    Intercept,
}

/// Port of `SMatrixLogicAction` (MatrixSide.hpp:173-178).
#[derive(Debug, Clone, Default)]
pub struct LogicAction {
    pub ty: LogicActionType,
    pub region: i32,
    /// `m_RegionPath[REGION_PATH_MAX_CNT]` + `m_RegionPathCnt`.
    pub region_path: Vec<i32>,
}

/// Port of `SMatrixLogicGroup` (MatrixSide.hpp:180-198). The C++ packs
/// robots-cnt + war into `m_Bits`; plain fields here.
#[derive(Debug, Clone, Default)]
pub struct LogicGroup {
    pub robots_cnt: i32,
    pub war: bool,
    pub action: LogicAction,
    pub team: i32,
    pub strength: f32,
}

/// Port of `SMatrixTeam` (MatrixSide.hpp:247-325). The ctor zeroes the
/// struct and sets `m_TargetRegion=-1` (MatrixSide.cpp:194-198).
#[derive(Debug, Clone)]
pub struct SideTeam {
    /// `Move()` — chassis-kind bitmask of the team's robots.
    pub move_mask: u8,
    pub war: bool,
    pub stay: bool,
    pub wait_union: bool,
    pub robot_in_des_region: bool,
    pub regroup_only_after_war: bool,
    pub l_ok: bool,
    pub robot_cnt: i32,
    pub group_cnt: i32,
    pub wait_union_last: i32,
    pub strength: f32,
    pub target_region: i32,
    pub center_mass: (i32, i32),
    pub radius_mass: i32,
    /// `m_Rect` (left, top, right, bottom).
    pub rect: (i32, i32, i32, i32),
    pub center: (i32, i32),
    pub radius: i32,
    pub region_mass_prev: i32,
    pub region_mass: i32,
    pub region_near_danger: i32,
    pub region_far_danger: i32,
    pub region_near_enemy: i32,
    pub region_near_retreat: i32,
    pub region_near_forward: i32,
    pub region_nearest_base: i32,
    /// `m_Brave` — timestamp when bravery kicked in (0 = not brave).
    pub brave: i32,
    pub brave_strange_cancel: f32,
    /// `m_ActionList[16]` + `m_ActionCnt`.
    pub action_list: Vec<LogicAction>,
    pub action: LogicAction,
    pub action_prev: LogicAction,
    pub action_time: i32,
    pub region_next: i32,
    pub road_path: RoadRoute,
    /// `m_RegionList`/`m_RegionListRobots` + cnt — (region, robots).
    pub region_list: Vec<(i32, i32)>,
}

impl Default for SideTeam {
    fn default() -> Self {
        Self {
            move_mask: 0,
            war: false,
            stay: false,
            wait_union: false,
            robot_in_des_region: false,
            regroup_only_after_war: false,
            l_ok: false,
            robot_cnt: 0,
            group_cnt: 0,
            wait_union_last: 0,
            strength: 0.0,
            target_region: -1,
            center_mass: (0, 0),
            radius_mass: 0,
            rect: (0, 0, 0, 0),
            center: (0, 0),
            radius: 0,
            region_mass_prev: 0,
            region_mass: 0,
            region_near_danger: 0,
            region_far_danger: 0,
            region_near_enemy: 0,
            region_near_retreat: 0,
            region_near_forward: 0,
            region_nearest_base: 0,
            brave: 0,
            brave_strange_cancel: 0.0,
            action_list: Vec::new(),
            action: LogicAction::default(),
            action_prev: LogicAction::default(),
            action_time: 0,
            region_next: -1,
            road_path: RoadRoute::default(),
            region_list: Vec::new(),
        }
    }
}

/// Port of `ESideStatus` (MatrixSide.hpp:103-112).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SideStatus {
    /// Side absent.
    None,
    /// Active side.
    #[default]
    Active,
    /// Just died — CheckStatus consumes this and flips it to `None`.
    JustDead,
    /// Just won — valid only for the player side.
    JustWin,
}

/// Port of `EStat` (MatrixSide.hpp:54-64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stat {
    RobotBuild = 0,
    RobotKill = 1,
    TurretBuild = 2,
    TurretKill = 3,
    BuildingKill = 4,
    Time = 5,
}
pub const MAX_STATISTICS: usize = 6;

/// Port of `EMatrixPlayerOrder` (MatrixSide.hpp:200-214).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlayerOrder {
    #[default]
    Stop,
    MoveTo,
    Capture,
    Attack,
    Patrol,
    Repair,
    Bomb,
    AutoCapture,
    AutoAttack,
    AutoDefence,
}

/// Port of `SMatrixPlayerGroup` (MatrixSide.hpp:216-245). The C++
/// packs order + 3 flags into `m_Bits`; plain fields here.
#[derive(Debug, Clone, Default)]
pub struct PlayerGroup {
    pub order: PlayerOrder,
    /// Bit 31 — group has engaged in combat.
    pub war: bool,
    /// Bit 30 — PGShowPlace should run after the next assignment.
    pub show_place: bool,
    /// Bit 29 — patrol leg direction.
    pub patrol_return: bool,
    pub robot_cnt: i32,
    pub from: (i32, i32),
    pub to: (i32, i32),
    pub obj: Option<ObjectId>,
    pub region: i32,
    /// `m_RegionPath[REGION_PATH_MAX_CNT]` + cnt — start→end order.
    pub region_path: Vec<i32>,
    /// `m_RoadPath` — owned route storage (the C++ news a
    /// `CMatrixRoadRoute` per group).
    pub road_path: RoadRoute,
}

/// Port of `CMatrixGroupObject` payload (MatrixAIGroup.h:30) — one
/// `(object, team)` membership entry.
#[derive(Debug, Clone, Copy)]
pub struct GroupObject {
    pub object: ObjectId,
    pub team: i32,
}

/// Port of `CMatrixGroup` (MatrixAIGroup.h:55) — an ordered roster of
/// selected objects with cached per-type counts.
#[derive(Debug, Clone, Default)]
pub struct Group {
    pub objects: Vec<GroupObject>,
    pub robots_cnt: i32,
    pub flyers_cnt: i32,
    pub buildings_cnt: i32,
    pub team: i32,
    pub id: i32,
}

impl Group {
    pub fn objects_cnt(&self) -> i32 {
        self.objects.len() as i32
    }

    /// `CMatrixGroup::AddObject` (MatrixAIGroup.cpp). The C++ reads
    /// the object's type for the counters; the caller passes it since
    /// the group has no arena access. Duplicates are skipped like the
    /// original (FindObject guard at every call site that matters).
    pub fn add_object(&mut self, object: ObjectId, team: i32, ty: ObjectType) {
        if self.find_object(object) {
            return;
        }
        self.objects.push(GroupObject { object, team });
        match ty {
            ObjectType::RobotAi => self.robots_cnt += 1,
            ObjectType::Flyer => self.flyers_cnt += 1,
            ObjectType::Building => self.buildings_cnt += 1,
            _ => {}
        }
    }

    /// `CMatrixGroup::RemoveObject(CMatrixMapStatic*)`. Needs the type
    /// to decrement the right counter; callers resolve it.
    pub fn remove_object(&mut self, object: ObjectId, ty: ObjectType) {
        let Some(pos) = self.objects.iter().position(|o| o.object == object) else {
            return;
        };
        self.objects.remove(pos);
        match ty {
            ObjectType::RobotAi => self.robots_cnt -= 1,
            ObjectType::Flyer => self.flyers_cnt -= 1,
            ObjectType::Building => self.buildings_cnt -= 1,
            _ => {}
        }
    }

    /// Drop members whose id no longer resolves (dead objects) —
    /// the C++ relies on explicit RemoveObject calls from death
    /// paths; this is the arena-handle equivalent sweep.
    pub fn prune_dead(&mut self, mut alive: impl FnMut(ObjectId) -> Option<ObjectType>) {
        let mut i = 0;
        while i < self.objects.len() {
            match alive(self.objects[i].object) {
                Some(_) => i += 1,
                None => {
                    // Type unknown once dead; recount below.
                    self.objects.remove(i);
                }
            }
        }
        self.robots_cnt = 0;
        self.flyers_cnt = 0;
        self.buildings_cnt = 0;
        for o in &self.objects {
            match alive(o.object) {
                Some(ObjectType::RobotAi) => self.robots_cnt += 1,
                Some(ObjectType::Flyer) => self.flyers_cnt += 1,
                Some(ObjectType::Building) => self.buildings_cnt += 1,
                _ => {}
            }
        }
    }

    /// `CMatrixGroup::RemoveAll`.
    pub fn remove_all(&mut self) {
        self.objects.clear();
        self.robots_cnt = 0;
        self.flyers_cnt = 0;
        self.buildings_cnt = 0;
    }

    /// `CMatrixGroup::FindObject`.
    pub fn find_object(&self, object: ObjectId) -> bool {
        self.objects.iter().any(|o| o.object == object)
    }

    /// `CMatrixGroup::GetObjectByN`.
    pub fn get_object_by_n(&self, num: i32) -> Option<ObjectId> {
        self.objects.get(num.max(0) as usize).map(|o| o.object)
    }
}

/// Port of the hard-coded 9000 cap inside `CMatrixSideUnit::AddResourceAmount`
/// (MatrixSide.hpp:438-443). Every `AddResourceAmount` call clamps the
/// final value to this ceiling; the HUD also displays `9000` when the
/// pool saturates.
pub const RESOURCE_CAP: i32 = 9000;

// `side_color_rgb` and `side_color_minimap_rgb` — port of
// `CMatrixMap::GetSideColor` / `GetSideColorMM` (MatrixMap.cpp:1014-1015,
// MatrixMap.hpp:738/745) — live in `matrix_game::map` alongside their
// owning class. Re-exported here for the existing call sites.
pub use crate::matrix_game::map::{side_color_minimap_rgb, side_color_rgb};

/// Port of `ESelectedType` (MatrixSide.hpp). Tracks what the player's
/// side is currently pointing at — the C++ interface panel dispatches
/// on this to decide which menu to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CurrSel {
    #[default]
    Nothing,
    /// A base (BUILDING_BASE) is selected — drives the
    /// "build robots / turrets" panel.
    BaseSelected,
    /// A non-base building (factory) is selected — drives the
    /// "produce resource / build turrets" panel.
    BuildingSelected,
    /// One or more robots are selected — drives unit-command panel.
    RobotsSelected,
    /// A cannon / turret is selected.
    CannonSelected,
    FlyerSelected,
}

/// Port of the selection-state slice of `CMatrixSide`. One instance
/// per side (id 0 = neutral/world, 1 = player, 2..=8 = AI factions).
///
/// The C++ stores multi-selection in a separate `CMultiSelection`
/// class and tags each object with `IsSelected()`. We collapse both
/// onto this struct: `selected` is the multi-set, `active_object` is
/// the "primary" (last picked / panel focus), and `curr_sel` keeps
/// the enum that drives the interface panel dispatcher.
#[derive(Debug, Clone, Default)]
pub struct Side {
    /// `m_Id` — side index (MatrixSide.hpp). `1 = PLAYER_SIDE`.
    pub id: i32,
    /// `m_ActiveObject` — primary selection. Drives the interface
    /// panel (`CurrSel`) and the main selection ring. The C++ stores
    /// a raw `CMatrixMapStatic *`; we use `ObjectId` so a freed
    /// object reads as a stale handle.
    pub active_object: Option<ObjectId>,
    /// `m_CurrSel` — enum above.
    pub curr_sel: CurrSel,
    /// Multi-selection set — port of `CMultiSelection::m_Sel`
    /// (MatrixMultiSelection.cpp). Order is insertion order so
    /// callers that iterate (e.g. move-order dispatch) keep a stable
    /// traversal. Always contains `active_object` when Some.
    pub selected: Vec<ObjectId>,

    /// `m_Resources[MAX_RESOURCES]` (MatrixSide.hpp:393) — per-side
    /// resource bank. Indexed by `Resource as usize`; starts empty
    /// and is topped up by factory-building `AddResourceAmount` calls
    /// (MatrixObjectBuilding.cpp:614 etc.) or drained by unit builds
    /// (CConstructor.cpp:239-242).
    pub resources: [i32; MAX_RESOURCES],

    /// `m_RobotsCnt` (MatrixSide.hpp:404) — cached count of live
    /// robots owned by this side. Refreshed each logic takt by
    /// `MapLogic::refresh_robots_cnt`. The C++ increments/decrements
    /// it from robot spawn / death call sites; we recompute once per
    /// tick which is simpler and equivalent for UI display.
    pub robots_cnt: i32,

    /// `m_BaseResForce` (MatrixSide.hpp:392) — the base-resource
    /// "force-up" multiplier in percent. Default 100 = normal income.
    /// Used by `GetResourceForceUp` at MatrixSide.cpp:346 etc.
    pub base_res_force: i32,

    /// Port of the `CConstructorPanel` slice of `CMatrixSide`. Held
    /// inline since each side has its own robot configurator in the
    /// C++ (`m_ConstructPanel`). Filled lazily — the actual
    /// `RobotBuilder` sub-struct lives in `interface::constructor`.
    /// `None` for neutral / AI sides that don't need the player UI.
    pub builder: Option<crate::matrix_game::interface::constructor::RobotBuilder>,

    /// `m_SideStatus` (MatrixSide.hpp:371).
    pub status: SideStatus,
    /// `m_Statistic[MAX_STATISTICS]` (MatrixSide.hpp:396).
    pub statistic: [i32; MAX_STATISTICS],
    /// `m_PlayerGroup[MAX_LOGIC_GROUP]` (MatrixSide.hpp:415).
    pub player_groups: Vec<PlayerGroup>,
    /// `m_CurSelGroup` — the scratch group marquee / shift-click
    /// selection accumulates into before `CreateGroupFromCurrent`.
    pub cur_sel_group: Group,
    /// `m_FirstGroup..m_LastGroup` — saved groups. `PumpGroups`
    /// deletes everything except the current one each takt, so this
    /// rarely holds more than one entry.
    pub groups: Vec<Group>,
    /// `m_CurrentGroup` — index into `groups`; `None` = no selection.
    pub current_group: Option<usize>,
    /// `m_CurSelNum` (MatrixSide.hpp:394) — primary-selection index
    /// within the current group.
    pub cur_sel_num: i32,
    /// `m_LastTaktHL` — TaktPL 100ms gate (MatrixSide.cpp:6046; the
    /// player takt reuses the HL timestamp). PGOrder* force a re-run
    /// by setting it to -1000.
    pub last_takt_pl: i32,
    /// `m_LastTaktUnderfire` — 500ms underfire-recalc gate
    /// (MatrixSide.cpp:6054).
    pub last_takt_underfire: i32,
    /// `m_Region` (SMatrixLogicRegion per road-network region) —
    /// PGCalcStat scratch, lazily sized.
    pub region_stats: Vec<LogicRegion>,
    /// `m_RegionIndex` — wave-search scratch parallel to regions.
    pub region_index: Vec<i32>,

    /// `MMFLAG_SOUND_BASE_SEL_ENABLED` (MatrixSide.cpp:1009-1012) —
    /// the very first base selection (game-start auto-select) is
    /// silent; the flag arms afterwards.
    pub base_sel_sound_enabled: bool,
    /// Cycles the S_SELECTION_1..7 voice pick (C++ uses Rnd(0,6),
    /// MatrixSide.cpp:978-993).
    pub sel_voice_seed: u32,

    // ── Enemy-AI state (MatrixSide.cpp ctor 156-233) ────────────────
    /// `m_TimeNextBomb` — min interval between bomb-robot builds;
    /// map `SideAIInfo` TBB overrides with `GetTime() + val`.
    pub time_next_bomb: i32,
    /// `m_TimeLastBomb`.
    pub time_last_bomb: i32,
    /// `m_StrengthMul` (SK).
    pub strength_mul: f32,
    /// `m_BraveMul` (BK).
    pub brave_mul: f32,
    /// `m_DangerMul` (DK).
    pub danger_mul: f32,
    /// `m_WaitResMul` (WRK).
    pub wait_res_mul: f32,
    /// `m_TeamCnt` (TC), 1..=3.
    pub team_cnt: i32,
    /// `m_Strength` — cached side strength from `CalcStrength`.
    pub strength: f32,
    /// `m_WarSide` — the side id we currently focus attacks on.
    pub war_side: i32,
    /// `m_NextWarSideCalcTime`.
    pub next_war_side_calc_time: i32,
    /// `m_LastTaktHL` (AI sides; the player reuses `last_takt_pl`).
    pub last_takt_hl: i32,
    /// `m_LastTaktTL`.
    pub last_takt_tl: i32,
    /// `m_LastTeamChange`.
    pub last_team_change: i32,
    /// `m_WaitResForBuildRobot` — AI-catalogue index we're saving up
    /// for, -1 = none.
    pub wait_res_for_build_robot: i32,
    /// `m_BuildRobotLast/2/3` — last three built template indices.
    pub build_robot_last: i32,
    pub build_robot_last2: i32,
    pub build_robot_last3: i32,
    /// `m_LogicGroup[MAX_LOGIC_GROUP]` — AI logic groups (the player
    /// uses `player_groups` instead).
    pub logic_groups: Vec<LogicGroup>,
    /// `m_Team[MAX_TEAM_CNT]`.
    pub teams: Vec<SideTeam>,
}

/// Port of `SMatrixLogicRegion` (MatrixSide.hpp:327-355).
#[derive(Debug, Clone, Copy, Default)]
pub struct LogicRegion {
    pub war_enemy_robot_cnt: i32,
    pub war_enemy_cannon_cnt: i32,
    pub war_enemy_building_cnt: i32,
    pub war_enemy_base_cnt: i32,
    pub enemy_robot_cnt: i32,
    pub enemy_cannon_cnt: i32,
    pub enemy_building_cnt: i32,
    pub enemy_base_cnt: i32,
    pub neutral_cannon_cnt: i32,
    pub neutral_building_cnt: i32,
    pub neutral_base_cnt: i32,
    pub our_robot_cnt: i32,
    pub our_cannon_cnt: i32,
    pub our_building_cnt: i32,
    pub our_base_cnt: i32,
    pub enemy_robot_dist: i32,
    pub enemy_building_dist: i32,
    pub our_base_dist: i32,
    pub danger: f32,
    pub danger_add: f32,
    pub data: u32,
}

impl Side {
    pub fn new(id: i32) -> Self {
        Self {
            id,
            active_object: None,
            curr_sel: CurrSel::Nothing,
            selected: Vec::new(),
            // C++ ctor default (MatrixSide.cpp:215-218); maps override
            // via `SideResInfo` in `apply_side_resources`.
            resources: [300, 300, 300, 300],
            robots_cnt: 0,
            base_res_force: 100,
            builder: Some(crate::matrix_game::interface::constructor::RobotBuilder::new()),
            // `SS_NONE` until StaticPrepare2 activates base/robot
            // owners (MatrixMapPrepare.cpp:1690-1697, 1713-1719) —
            // see `ensure_sides_from_objects`.
            status: SideStatus::None,
            statistic: [0; MAX_STATISTICS],
            player_groups: std::iter::repeat_with(PlayerGroup::default)
                .take(MAX_LOGIC_GROUP)
                .collect(),
            cur_sel_group: Group::default(),
            groups: Vec::new(),
            current_group: None,
            cur_sel_num: 0,
            last_takt_pl: 0,
            last_takt_underfire: 0,
            region_stats: Vec::new(),
            region_index: Vec::new(),
            base_sel_sound_enabled: false,
            sel_voice_seed: 0,
            time_next_bomb: 60000,
            time_last_bomb: 0,
            strength_mul: 1.0,
            brave_mul: 0.5,
            danger_mul: 1.0,
            wait_res_mul: 1.0,
            team_cnt: 3,
            strength: 0.0,
            war_side: -1,
            next_war_side_calc_time: 0,
            last_takt_hl: 0,
            last_takt_tl: 0,
            last_team_change: 0,
            wait_res_for_build_robot: -1,
            build_robot_last: -1,
            build_robot_last2: -1,
            build_robot_last3: -1,
            logic_groups: std::iter::repeat_with(LogicGroup::default)
                .take(MAX_LOGIC_GROUP)
                .collect(),
            teams: std::iter::repeat_with(SideTeam::default)
                .take(MAX_TEAM_CNT)
                .collect(),
        }
    }

    /// `ClearTeam` (MatrixSide.cpp:2013-2025).
    pub fn clear_team(&mut self, team: usize) {
        let t = &mut self.teams[team];
        t.target_region = -1;
        t.action.ty = LogicActionType::None;
        t.action.region_path.clear();
        t.war = false;
        t.brave = 0;
        t.brave_strange_cancel = 0.0;
        t.regroup_only_after_war = false;
        t.wait_union = false;
        t.wait_union_last = 0;
    }

    /// `GetStatValue` / `SetStatValue` / `IncStatValue`
    /// (MatrixSide.hpp:428-430).
    pub fn get_stat(&self, stat: Stat) -> i32 {
        self.statistic[stat as usize]
    }
    pub fn set_stat(&mut self, stat: Stat, v: i32) {
        self.statistic[stat as usize] = v;
    }
    pub fn inc_stat(&mut self, stat: Stat) {
        self.statistic[stat as usize] += 1;
    }

    /// `GetCurGroup()` (MatrixSide.hpp:482).
    pub fn cur_group(&self) -> Option<&Group> {
        self.current_group.and_then(|i| self.groups.get(i))
    }
    pub fn cur_group_mut(&mut self) -> Option<&mut Group> {
        self.current_group.and_then(move |i| self.groups.get_mut(i))
    }

    /// `GetCurSelObject` (MatrixSide.cpp:1438) — the
    /// `m_CurSelNum`-th member of the current group.
    pub fn get_cur_sel_object(&self) -> Option<ObjectId> {
        self.cur_group()?.get_object_by_n(self.cur_sel_num)
    }

    /// `PumpGroups` (MatrixSide.cpp:1518) — drop every saved group
    /// except the current one.
    pub fn pump_groups(&mut self) {
        match self.current_group {
            Some(cur) => {
                if cur != 0 {
                    self.groups.swap(0, cur);
                    self.current_group = Some(0);
                }
                self.groups.truncate(1);
            }
            None => self.groups.clear(),
        }
    }

    /// Port of `CMatrixSideUnit::AddResourceAmount` (MatrixSide.hpp:
    /// 438-443). Caps at 9000 per resource (hard-coded in the original
    /// setter); floors at 0 so decrements can't go negative.
    pub fn add_resource_amount(&mut self, res: Resource, amount: i32) {
        let i = res as usize;
        let v = self.resources[i].saturating_add(amount);
        self.resources[i] = v.clamp(0, RESOURCE_CAP);
    }

    /// Port of `CMatrixSideUnit::GetResourceForceUp` (MatrixSide.hpp:442).
    pub fn get_resource_force_up(&self) -> i32 {
        self.base_res_force
    }

    /// Port of `CMatrixSideUnit::SetResourceForceUp` (MatrixSide.hpp:441).
    pub fn set_resource_force_up(&mut self, fu: i32) {
        self.base_res_force = fu;
    }

    /// Port of `CMatrixSideUnit::GetSideRobots` (MatrixSide.hpp:436) —
    /// cached live-robot count. Refreshed each tick.
    pub fn get_side_robots(&self) -> i32 {
        self.robots_cnt
    }

    /// Port of `CMatrixSideUnit::GetResourceAmount` (MatrixSide.cpp).
    pub fn get_resource_amount(&self, res: Resource) -> i32 {
        self.resources[res as usize]
    }

    /// True if the side can afford the cost (all four resources are
    /// ≥ the requested amount).
    pub fn can_afford(&self, cost: &crate::matrix_game::interface::constructor::UnitPrice) -> bool {
        for r in Resource::ALL {
            if self.resources[r as usize] < cost.resources[r as usize] {
                return false;
            }
        }
        true
    }

    /// Replace the selection with a single object. Matches
    /// `CMatrixSide::SelectObject` (MatrixSide.cpp): drops any prior
    /// multi-selection and sets the new object as both active and
    /// sole selected. `curr_sel` is assigned by the caller from the
    /// object type.
    pub fn select_single(&mut self, id: ObjectId, curr_sel: CurrSel) {
        self.selected.clear();
        self.selected.push(id);
        self.active_object = Some(id);
        self.curr_sel = curr_sel;
        self.selection_sound(curr_sel);
    }

    /// Selection voices — port of the CSound::Play calls in
    /// `CMatrixSideUnit::Select` (MatrixSide.cpp:976-1016). Player
    /// side only; robots get one of seven voices, bases/factories a
    /// fixed one. The first base select (game-start auto-select) is
    /// suppressed via [`Self::base_sel_sound_enabled`].
    fn selection_sound(&mut self, curr_sel: CurrSel) {
        use crate::matrix_game::interface::sound::play_named;
        if self.id != crate::matrix_game::common::PLAYER_SIDE {
            return;
        }
        match curr_sel {
            CurrSel::RobotsSelected => {
                self.sel_voice_seed = self.sel_voice_seed.wrapping_add(1);
                play_named(&format!("s_selection_{}", self.sel_voice_seed % 7 + 1));
            }
            CurrSel::BaseSelected => {
                if self.base_sel_sound_enabled {
                    play_named("s_base_sel");
                }
                self.base_sel_sound_enabled = true;
            }
            CurrSel::BuildingSelected => play_named("s_building_sel"),
            _ => {}
        }
    }

    /// Backward-compat alias for `select_single`. Ports
    /// `CMatrixSide::SelectObject`.
    pub fn select(&mut self, id: ObjectId, curr_sel: CurrSel) {
        self.select_single(id, curr_sel);
    }

    /// Shift-click: add `id` if absent, remove it if present
    /// (ports `CMultiSelection::Add` + `Remove` at
    /// MatrixSide.cpp:1584-1598). `curr_sel` is applied when the
    /// toggle results in adding *and* the active focus moves to
    /// this object.
    pub fn select_toggle(&mut self, id: ObjectId, curr_sel: CurrSel) {
        if let Some(pos) = self.selected.iter().position(|&x| x == id) {
            self.selected.swap_remove(pos);
            if self.active_object == Some(id) {
                self.active_object = self.selected.last().copied();
                // When the toggled-off was primary, fall back to the
                // last-added remaining selection. Clear enum if none.
                if self.selected.is_empty() {
                    self.curr_sel = CurrSel::Nothing;
                }
            }
        } else {
            self.selected.push(id);
            self.active_object = Some(id);
            self.curr_sel = curr_sel;
            self.selection_sound(curr_sel);
        }
    }

    /// Replace the selection with a whole set at once (marquee
    /// drag-box). `primary` becomes the active object; `curr_sel` is
    /// keyed off it. Empty input clears. Ports the end-of-drag fold
    /// at `CMultiSelection::End` (MatrixMultiSelection.cpp).
    pub fn select_replace(
        &mut self,
        ids: Vec<ObjectId>,
        primary: Option<ObjectId>,
        curr_sel: CurrSel,
    ) {
        self.selected = ids;
        self.active_object = primary;
        self.curr_sel = if primary.is_some() {
            curr_sel
        } else {
            CurrSel::Nothing
        };
        if primary.is_some() {
            self.selection_sound(curr_sel);
        }
    }

    /// Clear the selection — C++ `CMatrixSide::UnSelect` equivalent.
    pub fn clear(&mut self) {
        self.active_object = None;
        self.curr_sel = CurrSel::Nothing;
        self.selected.clear();
    }

    pub fn is_selected(&self, id: ObjectId) -> bool {
        self.selected.contains(&id)
    }
}
