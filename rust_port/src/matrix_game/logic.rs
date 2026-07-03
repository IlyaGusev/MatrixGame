//! Port of `MatrixLogic.{cpp,hpp}` — `CMatrixMapLogic`, the
//! logic-layer subclass of `CMatrixMap`. Owns the Objects arena
//! (indirectly), the shared RNG, the takt-decomposition driver,
//! per-side state, and the cell-level place / wall helpers used by
//! the robot AI.
//!
//! Also the module root for `Logic/*.cpp` ports — `ai_group.rs`
//! (Logic/MatrixAIGroup.cpp) and future siblings declare here.

pub mod ai_group;
pub mod environment;

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{PLAYER_SIDE, TRACE_ANYOBJECT};
use crate::matrix_game::config::Resource;
use crate::matrix_game::config::{
    self, BuildingDamages, BuildingLabels, ChassisChars, GlobalConfig, HeadCharsTable,
    ItemCharsTable, ItemDescriptions, ItemLabels, ObjectDamages, PriceTable, RobotDamages,
    RobotNameParts, StringTables, Timings, TurretProps, WeaponCooldown, WeaponStrengthAI,
};
use crate::matrix_game::map::{GameMap, MoveCell};
use crate::matrix_game::map_static::{MapStatic, ObjectId, ObjectType, Objects};
use crate::matrix_game::object::MapObject;
use crate::matrix_game::object_building::{Building, BuildingType};
use crate::matrix_game::side::{CurrSel, Side};
use crate::matrix_lib::base::storage::Storage;

/// Per-building income constants — port of
/// MatrixObjectBuilding.hpp:22-23.
pub const RESOURCES_INCOME: i32 = 10;
pub const RESOURCES_INCOME_BASE: i32 = 3;

/// Port of ROBOTS_BY_BASE / ROBOTS_BY_MAIN (MatrixSide.hpp:24-26).
/// `ROBOT_BY_FACTORY` is defined in the C++ but the `GetMaxSideRobots`
/// formula uses `factories * 1` directly (the multiplier is commented
/// out at MatrixSide.cpp:1666), so we inline the `* 1` and don't carry
/// the constant.
pub const ROBOTS_BY_BASE: i32 = 3;
pub const ROBOTS_BY_MAIN: i32 = 4;

/// `LOGIC_TAKT_PERIOD` from `MatrixLogic.hpp:13`.
pub const LOGIC_TAKT_PERIOD_MS: i32 = 10;

/// Port of `ROBOT_MOVECELLS_PER_SIZE` (MatrixMap.hpp:25). The robot
/// footprint in move cells is a square this wide on each side. The
/// `SMatrixMapMove::m_Stop` bitmask stores per-size passability at
/// bits `(1 << chassis) << (6 * (size-1))`; for a robot we always
/// use `size = ROBOT_MOVECELLS_PER_SIZE`, hence the shift = 18.
pub const ROBOT_MOVECELLS_PER_SIZE: i32 = 4;

/// Port of `COLLIDE_BOT_R` (MatrixRobot.hpp:26). Radius of the
/// robot's collision sphere — used by `PlaceIsEmpty` to exclude
/// candidate cells that sit too close to another robot or to
/// another robot's destination.
pub const COLLIDE_BOT_R: f32 = 18.0;

/// Port of the Takt driver + per-session arena / RNG / player-side
/// aggregation. In C++ all these live on `CMatrixMapLogic`; in
/// Rust we keep the map (`GameMap`) and the logic state in
/// different structs because the map is built at load time and
/// shared immutably across renderer / physics / AI paths, whereas
/// the logic state mutates every tick.
pub struct MapLogic {
    pub objects: Objects,
    /// Live gameplay effects (projectiles, blasts, flame streams) —
    /// the `CMatrixEffect` list on `CMatrixMap`, gameplay subset.
    pub effects: Vec<crate::matrix_game::effects::GameEffect>,
    /// The shared RNG — matches `CMatrixMapLogic::m_Rnd`
    /// (MatrixLogic.hpp:75).
    pub rng: Rnd,
    /// Number of LOGIC_TAKT_PERIOD portions elapsed since construction.
    pub tick: u64,
    /// Total game-time in ms. Ports `g_MatrixMap->GetTime()` return.
    pub elapsed_ms: i64,
    /// Port of `g_MatrixMap->GetPlayerSide()`. Full per-side state
    /// lands with `CMatrixSide`.
    pub player_side: Side,
    /// Non-player sides (`m_Side[]` minus the player) — created by
    /// [`MapLogic::ensure_sides_from_objects`] from the side ids the
    /// map spawns. Neutral (side 0) owns no Side entry, as in C++.
    pub other_sides: Vec<Side>,
    /// `m_GatherInfoLast` (MatrixLogic.cpp:51) — 100ms env-refresh gate.
    pub gather_info_last: i64,
    /// `m_PrevTimeCheckStatus` (MatrixMap.hpp:421) — 1s win/lose gate.
    pub prev_time_check_status: i64,
    /// `m_BeforeWinLooseDialogCount` (MatrixMap.hpp:422).
    pub before_win_loose_dialog_count: i32,
    /// `m_BeforeWinCount` (MatrixMap.hpp:309) — outstanding special
    /// map objects that must break before victory can trigger.
    pub before_win_count: i32,
    /// `MMFLAG_WIN` + EnterDialogMode(WIN/LOOSE) outcome —
    /// `Some(true)` = player won, `Some(false)` = lost. The UI layer
    /// reads this to show the end dialog.
    pub game_over: Option<bool>,
    /// One-shot dialog request — stands in for the direct
    /// `EnterDialogMode(TEMPLATE_DIALOG_WIN/LOOSE)` call at
    /// MatrixLogic.cpp:2900-2906 (logic can't reach the UI layer
    /// here). The form-game loop takes it each frame.
    pub pending_win_loose_dialog: Option<bool>,
    /// Queued gameplay sounds (CSound::Play call sites). The audio
    /// backend drains this each frame; until it lands the queue is
    /// the verified dispatch surface.
    pub sound_queue: Vec<crate::matrix_game::side_player::GameSound>,
    /// `MMFLAG_SOUND_ORDER_CAPTURE_ENABLED` (set by PGOrderCapture,
    /// consumed by SoundCapture).
    pub sound_order_capture_enabled: bool,
    /// `MMFLAG_ENABLE_CAPTURE_FUCKOFF_SOUND`.
    pub enable_capture_fuckoff_sound: bool,
    /// `MMFLAG_SOUND_ORDER_ATTACK_DISABLE`.
    pub sound_order_attack_disable: bool,
    /// Move-to ping world positions queued by `PGShowPlace`; the
    /// renderer drains them. `moveto_clear_pending` mirrors
    /// `CMatrixEffect::DeleteAllMoveto` preceding the new batch.
    pub moveto_pings: Vec<glam::Vec3>,
    pub moveto_clear_pending: bool,
    /// `MMFLAG_WIN` (set when victory is detected, read by the
    /// delayed dialog entry).
    pub win_flag: bool,
    /// `MMFLAG_SPECIAL_BROKEN` (MatrixMap.hpp:232) — all special
    /// win-target objects destroyed; exempts the player from the
    /// JUST_DEAD scan (MatrixLogic.cpp:2866).
    pub special_broken: bool,
    /// `MMFLAG_SHOWPORTRETS` / `MMFLAG_ROBOT_IN_POSITION` — terron easter
    /// egg (MatrixLogic.cpp:2775-2803, MatrixObject.cpp:889-892).
    pub show_portrets: bool,
    pub robot_in_position: bool,
    /// `m_MaintenanceTime` (MatrixMap.hpp:434) — countdown until the
    /// next supply drop may be called.
    pub maintenance_time: i32,
    /// `m_MaintenancePRC` (MatrixMap.hpp:435) — map-level maintenance
    /// scale percent; 0 disables.
    pub maintenance_prc: i32,
}

impl Default for MapLogic {
    fn default() -> Self {
        Self::new()
    }
}

impl MapLogic {
    pub fn new() -> Self {
        Self::with_seed(1)
    }

    /// Deterministic-seed constructor — tests and replay use this
    /// so the generator state is reproducible.
    pub fn with_seed(seed: i32) -> Self {
        Self {
            objects: Objects::new(),
            effects: Vec::new(),
            rng: Rnd::new(seed),
            tick: 0,
            elapsed_ms: 0,
            player_side: Side::new(PLAYER_SIDE),
            other_sides: Vec::new(),
            gather_info_last: 0,
            prev_time_check_status: 0,
            before_win_loose_dialog_count: 0,
            before_win_count: 0,
            game_over: None,
            pending_win_loose_dialog: None,
            sound_queue: Vec::new(),
            sound_order_capture_enabled: false,
            enable_capture_fuckoff_sound: false,
            sound_order_attack_disable: false,
            moveto_pings: Vec::new(),
            moveto_clear_pending: false,
            win_flag: false,
            special_broken: false,
            show_portrets: false,
            robot_in_position: false,
            maintenance_time: 0,
            maintenance_prc: 100,
        }
    }

    /// `MaintenanceDisabled` / `BeforeMaintenanceTime` /
    /// `InitMaintenanceTime` (MatrixMap.hpp:468-475).
    pub fn maintenance_disabled(&self) -> bool {
        self.maintenance_prc == 0
    }
    pub fn init_maintenance_time(&mut self) {
        let cfg = crate::matrix_game::config::global();
        self.maintenance_time = (cfg.difficulty.k_time_before_maintenance
            * cfg.timings.maintenance_period as f32
            * self.maintenance_prc as f32
            / 100.0) as i32;
    }

    /// `GetSideById` (MatrixMap.cpp). Side 0 (neutral) has no entry.
    pub fn side_by_id(&self, id: i32) -> Option<&Side> {
        if id == self.player_side.id {
            return Some(&self.player_side);
        }
        self.other_sides.iter().find(|s| s.id == id)
    }

    pub fn side_by_id_mut(&mut self, id: i32) -> Option<&mut Side> {
        if id == self.player_side.id {
            return Some(&mut self.player_side);
        }
        self.other_sides.iter_mut().find(|s| s.id == id)
    }

    /// Create a `Side` entry for every non-neutral side id present in
    /// the spawned objects — the C++ builds `m_Side[]` from the map
    /// data at load (MatrixMapPrepare.cpp). Call after spawning.
    pub fn ensure_sides_from_objects(&mut self) {
        let mut ids: Vec<i32> = Vec::new();
        for id in self.objects.iter_live() {
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            let side = obj.side();
            if side > 0 && side != self.player_side.id && !ids.contains(&side) {
                ids.push(side);
            }
        }
        for sid in ids {
            if self.side_by_id(sid).is_none() {
                self.other_sides.push(Side::new(sid));
            }
        }
    }

    /// Seed each side's starting resources + base income multiplier from
    /// the map `SideResInfo` property (MatrixMapPrepare.cpp:464-493).
    /// Call once after sides exist; overrides the pre-map defaults.
    pub fn apply_side_resources(&mut self, map: &GameMap) {
        for &(id, res, fu) in &map.side_res_info {
            if let Some(su) = self.side_by_id_mut(id) {
                for (k, &amt) in res.iter().enumerate() {
                    // Plain assignment — C++ SetResourceAmount does not
                    // clamp (MatrixSide.hpp:440); only AddResourceAmount does.
                    su.resources[k] = amt;
                }
                su.set_resource_force_up(fu);
            }
        }
    }

    /// Port of the logic-takt portion of `CMatrixMapLogic::Takt`
    /// (`MatrixLogic.cpp:2722-2734`). Full 10ms slices, then the
    /// remainder — matching the trailing `if (portions) ProceedLogic(portions);`.
    pub fn takt(&mut self, step_ms: i32) {
        if step_ms <= 0 {
            return;
        }
        // Maintenance countdown (MatrixLogic.cpp:2655-2663).
        if self.maintenance_time > 0 {
            self.maintenance_time = (self.maintenance_time - step_ms).max(0);
        }
        // GatherInfo every >100ms (MatrixLogic.cpp:2713-2718), before
        // the logic portions like the C++ Takt.
        if self.elapsed_ms - self.gather_info_last > 100 {
            self.gather_info_last = self.elapsed_ms;
            if let Some(map) = crate::matrix_game::map::current_map() {
                self.gather_info(map, 0);
                self.gather_info(map, 1);
            }
        }
        let full = step_ms / LOGIC_TAKT_PERIOD_MS;
        for _ in 0..full {
            self.objects
                .proceed_logic(LOGIC_TAKT_PERIOD_MS, &mut self.rng);
            self.tick += 1;
            // `m_Side[i].LogicTakt(LOGIC_TAKT_PERIOD)` — player side
            // only (enemy-AI sides excluded by scope),
            // MatrixLogic.cpp:2737-2745.
            if let Some(map) = crate::matrix_game::map::current_map() {
                self.side_logic_takt(map);
            }
        }
        let rem = step_ms - full * LOGIC_TAKT_PERIOD_MS;
        if rem > 0 {
            self.objects.proceed_logic(rem, &mut self.rng);
        }
        // Special win-target deaths queued by map objects this takt
        // (MatrixObject.cpp:203-212 / :249-260 / :1244-1258).
        if !self.objects.pending_special_deaths.is_empty() {
            use crate::matrix_game::map_static::SpecialDeathKind;
            use crate::matrix_game::side::SideStatus;
            let deaths: Vec<_> = self.objects.pending_special_deaths.drain(..).collect();
            for d in deaths {
                self.before_win_count -= 1;
                match d {
                    SpecialDeathKind::Target => {
                        if self.before_win_count <= 0 {
                            self.special_broken = true;
                            self.player_side.status = SideStatus::JustWin;
                        }
                    }
                    SpecialDeathKind::Terron => {
                        // SS_ACTIVE → SS_JUST_WIN + MMFLAG_WIN,
                        // unconditional (MatrixObject.cpp:1251-1258).
                        self.win_flag = true;
                        self.player_side.status = SideStatus::JustWin;
                    }
                }
            }
        }
        // Win/lose CheckStatus once per second (MatrixLogic.cpp:2773+).
        self.check_status();
        // Notices addressed to robots that died before draining them.
        self.objects.pending_hit_notices.clear();
        // Gameplay effects tick on the frame clock — the effect-list
        // walk in `CMatrixMap::Takt`. Needs the scoped map for traces;
        // outside a `MapScope` (unit tests of unrelated systems) the
        // projectiles simply hold still.
        if let Some(map) = crate::matrix_game::map::current_map() {
            crate::matrix_game::effects::effects_takt(
                &mut self.effects,
                step_ms as f32,
                map,
                &mut self.objects,
                &mut self.rng,
            );
            // BEHF_SPAWNER robots queued by object logic_takt this frame.
            self.drain_spawner_bots(map);
        }
        // Port of the per-building `m_ResourcePeriod` tick at
        // MatrixObjectBuilding.cpp:605-667. The C++ advances the
        // counter from inside each building's `LogicTakt`; we run it
        // once per outer slice so the per-building state ports 1:1
        // without threading the player Side through `proceed_logic`.
        self.accrue_resources(step_ms);
        self.refresh_side_robots();
        self.sync_side_stats();
        // The standalone build plays no audio (g_RangersInterface is
        // NULL in the original) — drain the queued sound events each
        // frame so the dispatch surface can't grow unboundedly. A
        // future host backend consumes them here instead.
        self.sound_queue.clear();
        self.elapsed_ms += step_ms as i64;
    }

    /// Port of the side-status / win-lose check in
    /// `CMatrixMapLogic::Takt` (MatrixLogic.cpp:2773-2955). Runs once
    /// per ~2s (the C++ `m_PrevTimeCheckStatus = GetTime() + 1001`
    /// quirk is kept). Dialog entry becomes `game_over`.
    pub(crate) fn check_status(&mut self) {
        use crate::matrix_game::side::{SideStatus, Stat};

        if self.elapsed_ms - self.prev_time_check_status <= 1001 {
            return;
        }
        self.prev_time_check_status = self.elapsed_ms + 1001;

        // Which sides still own a live robot or a live base?
        let mut ids: Vec<i32> = vec![self.player_side.id];
        ids.extend(self.other_sides.iter().map(|s| s.id));
        let mut alive: Vec<bool> = vec![false; ids.len()];
        let mut was_none: Vec<bool> = vec![false; ids.len()];
        for (k, &sid) in ids.iter().enumerate() {
            if self.side_by_id(sid).map(|s| s.status) == Some(SideStatus::None) {
                was_none[k] = true;
                alive[k] = true;
            }
        }
        for id in self.objects.iter_live() {
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            let side = obj.side();
            let Some(k) = ids.iter().position(|&s| s == side) else {
                continue;
            };
            if alive[k] {
                continue;
            }
            let lives = match obj.core().obj_type {
                ObjectType::RobotAi => obj.is_live(),
                ObjectType::Building => {
                    obj.is_live()
                        && building_ref(&self.objects, id)
                            .map(|b| b.is_base())
                            .unwrap_or(false)
                }
                _ => false,
            };
            if lives {
                alive[k] = true;
            }
        }

        // Freshly dead sides: JUST_DEAD + neutralize their buildings
        // (MatrixLogic.cpp:2860-2886).
        for (k, &sid) in ids.iter().enumerate() {
            if alive[k] {
                continue;
            }
            // MatrixLogic.cpp:2864-2867 — a player who already broke
            // the terron / all special targets can't lose to attrition.
            if sid == self.player_side.id && (self.special_broken || self.objects.terron_dead) {
                continue;
            }
            if let Some(s) = self.side_by_id_mut(sid) {
                s.status = SideStatus::JustDead;
            }
            let buildings: Vec<ObjectId> = self
                .objects
                .iter_live()
                .filter(|&id| {
                    building_ref(&self.objects, id)
                        .map(|b| b.is_live() && b.side == sid)
                        .unwrap_or(false)
                })
                .collect();
            for bid in buildings {
                if let Some(b) = building_mut(&mut self.objects, bid) {
                    b.set_neutral();
                }
            }
        }

        // Win/lose bookkeeping (MatrixLogic.cpp:2891-2954).
        if self.game_over.is_some() {
            return;
        }
        if self.before_win_loose_dialog_count > 1 {
            self.before_win_loose_dialog_count -= 1;
        } else if self.before_win_loose_dialog_count == 1 {
            let win = self.win_flag;
            if win {
                let t = self.player_side.get_stat(Stat::Time);
                self.player_side.set_stat(Stat::Time, -t);
            }
            self.game_over = Some(win);
            self.pending_win_loose_dialog = Some(win);
        } else if self.player_side.status == SideStatus::JustWin {
            self.player_side.status = SideStatus::Active;
            self.before_win_loose_dialog_count = 1;
            self.win_flag = true;
        } else if self.player_side.status == SideStatus::JustDead {
            for (k, &sid) in ids.iter().enumerate() {
                if alive[k] && !was_none[k] {
                    if let Some(s) = self.side_by_id_mut(sid) {
                        let t = s.get_stat(Stat::Time);
                        s.set_stat(Stat::Time, -t);
                    }
                }
            }
            self.player_side.status = SideStatus::None;
            self.before_win_loose_dialog_count = 1;
            self.win_flag = false;
        } else {
            let mut acnt = 0;
            for &sid in &ids {
                let st = self.side_by_id(sid).map(|s| s.status);
                if st == Some(SideStatus::JustDead) {
                    if let Some(s) = self.side_by_id_mut(sid) {
                        s.status = SideStatus::None;
                    }
                }
                if self.side_by_id(sid).map(|s| s.status) == Some(SideStatus::Active) {
                    acnt += 1;
                }
            }
            if acnt == 1
                && self.before_win_count <= 0
                && self.player_side.status == SideStatus::Active
                && !self.other_sides.is_empty()
            {
                // Only one active side left — the player. Win.
                self.before_win_loose_dialog_count = 1;
                self.win_flag = true;
            }
        }

        self.terron_egg_check();
    }

    /// The "terron" map easter egg (MatrixLogic.cpp:2775-2803): if exactly
    /// one full-HP bomb-carrying player robot stands on the trigger cell
    /// (and it's the only robot within 160), reveal the hidden portrets.
    fn terron_egg_check(&mut self) {
        use crate::matrix_game::common::{PLAYER_SIDE, TRACE_ROBOT};
        use crate::matrix_game::map_static::Control;

        if crate::matrix_game::map::map_name_is_terron() {
            let center = glam::Vec2::new(3833.8, 2298.1);
            // Egg1: score robots within radius 50; mark the one that keeps
            // the running count at exactly 1.
            let mut ids50 = Vec::new();
            self.objects
                .find_objects(center, 50.0, 1.0, TRACE_ROBOT, None, |_, id| {
                    ids50.push(id);
                    Control::Continue
                });
            let mut egg1 = 0i32;
            let mut mark: Option<ObjectId> = None;
            for id in ids50 {
                let Some(r) = robot_ref(&self.objects, id) else {
                    continue;
                };
                if !r.is_live() {
                    continue;
                }
                egg1 += 1;
                if r.side() != PLAYER_SIDE {
                    egg1 += 100;
                }
                let bad = !r.have_bomb(&self.objects) || (r.hit_point_max - r.hit_point).abs() > 1.0;
                if bad && !r.in_position {
                    egg1 += 100;
                }
                if egg1 == 1 {
                    mark = Some(id);
                }
            }
            if let Some(mid) = mark {
                if let Some(r) = robot_mut(&mut self.objects, mid) {
                    r.in_position = true;
                }
            }
            let mut in_pos = false;
            if egg1 == 1 {
                let mut egg2 = 0i32;
                let mut ids160 = Vec::new();
                self.objects
                    .find_objects(center, 160.0, 1.0, TRACE_ROBOT, None, |_, id| {
                        ids160.push(id);
                        Control::Continue
                    });
                for id in ids160 {
                    if robot_ref(&self.objects, id).map(|r| r.is_live()).unwrap_or(false) {
                        egg2 += 1;
                    }
                }
                in_pos = egg2 == 1;
            }
            self.robot_in_position = in_pos;
            if !in_pos {
                // "oblom" — nobody in position: clear every robot's flag.
                let all: Vec<_> = self.objects.iter_live().collect();
                for id in all {
                    if let Some(r) = robot_mut(&mut self.objects, id) {
                        r.in_position = false;
                    }
                }
            }
        } else {
            self.robot_in_position = false;
        }
        crate::matrix_game::map::set_portrets_in_position(self.robot_in_position);
    }

    /// Port of `CMatrixMapLogic::GatherInfo` (MatrixLogic.cpp:115-134)
    /// + `CMatrixRobotAI::GatherInfo` (MatrixRobot.cpp:4333-4552).
    /// `gtype` 0 = line-of-sight scan, 1 = radio exchange within the
    /// same logic group / team+region.
    pub fn gather_info(&mut self, map: &GameMap, gtype: i32) {
        use crate::matrix_game::common::float2int;
        use crate::matrix_game::object_cannon::CannonState;
        use crate::matrix_game::robot::BARREL_TO_SHOT_ANGLE;

        let now = self.elapsed_ms as i32;
        let gs = GameMap::GLOBAL_SCALE_MOVE;

        let robot_ids: Vec<ObjectId> = self
            .objects
            .iter_live()
            .filter(|&id| {
                robot_ref(&self.objects, id)
                    .map(|r| r.is_live())
                    .unwrap_or(false)
            })
            .collect();

        enum Act {
            Add(ObjectId),
            AddIgnore(ObjectId),
            RemoveSlowly(ObjectId),
            AddSlowly(ObjectId),
        }

        for &rid in &robot_ids {
            // CalcStrength at the top of GatherInfo (MatrixRobot.cpp:4337).
            let strength = robot_ref(&self.objects, rid)
                .map(|r| r.calc_strength(&self.objects))
                .unwrap_or(0.0);
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.strength = strength;
            }

            let Some(me) = robot_ref(&self.objects, rid) else {
                continue;
            };
            let my_side = me.side;
            let my_pos = glam::Vec2::new(me.pos_x, me.pos_y);
            let my_map = (me.map_x, me.map_y);
            let my_geo = me.core().geo_center;
            let my_maxfd = me.max_fire_dist;
            let my_chassis = me.chassis.kind_index() as u8;
            let my_group_logic = me.group_logic;
            let my_team = me.team;

            let mut acts: Vec<Act> = Vec::new();

            if gtype == 0 {
                let mut place_buf: Vec<i32> = Vec::new();
                for oid in self.objects.iter_live() {
                    if oid == rid {
                        continue;
                    }
                    let Some(obj) = self.objects.get(oid) else {
                        continue;
                    };
                    match obj.core().obj_type {
                        ObjectType::RobotAi => {
                            let other = robot_ref(&self.objects, oid).unwrap();
                            if other.side == my_side {
                                continue;
                            }
                            let d = glam::Vec2::new(other.pos_x, other.pos_y) - my_pos;
                            let dist_enemy = d.length_squared();
                            let t1 = other.max_fire_dist.max(my_maxfd) * 1.1;
                            if dist_enemy <= t1 * t1 && other.is_live() {
                                if is_logic_visible(map, &self.objects, rid, oid, 0.0) {
                                    let me_env = &robot_ref(&self.objects, rid).unwrap().env;
                                    if !me_env.search_enemy(oid) && !me_env.is_ignore(oid, now) {
                                        let p_to = (other.map_x, other.map_y);
                                        let (st, dist) = place_list(
                                            map,
                                            my_chassis,
                                            my_map,
                                            p_to,
                                            float2int(my_maxfd / gs),
                                            false,
                                            &mut place_buf,
                                        );
                                        let d2 = {
                                            let dx = (my_map.0 - p_to.0) as i64;
                                            let dy = (my_map.1 - p_to.1) as i64;
                                            dx * dx + dy * dy
                                        };
                                        if st != 0 && ((dist / 4) as i64).pow(2) < d2 {
                                            acts.push(Act::Add(oid));
                                        } else {
                                            acts.push(Act::AddIgnore(oid));
                                        }
                                    }
                                }
                            } else {
                                let t2 = other.max_fire_dist.max(my_maxfd) * 1.4;
                                if dist_enemy > t2 * t2 {
                                    acts.push(Act::RemoveSlowly(oid));
                                }
                            }
                        }
                        ObjectType::Cannon => {
                            let cannon = cannon_ref(&self.objects, oid).unwrap();
                            if cannon.side == my_side
                                || matches!(cannon.state, CannonState::UnderConstruction)
                            {
                                continue;
                            }
                            let cgeo = cannon.core().geo_center;
                            let dvec = cgeo - glam::Vec3::new(my_pos.x, my_pos.y, 0.0);
                            let dist_enemy = dvec.length_squared();
                            let fire_r = cannon.fire_radius(&self.objects);
                            let t1 = (fire_r * 1.01).max(my_maxfd * 1.1);
                            if dist_enemy <= t1 * t1 && !matches!(cannon.state, CannonState::Dip) {
                                if is_logic_visible(map, &self.objects, rid, oid, 0.0) {
                                    let me_env = &robot_ref(&self.objects, rid).unwrap().env;
                                    if !me_env.search_enemy(oid) && !me_env.is_ignore(oid, now) {
                                        let p_to = (
                                            float2int(cannon.pos.x / gs),
                                            float2int(cannon.pos.y / gs),
                                        );
                                        let d2 = {
                                            let dx = (my_map.0 - p_to.0) as i64;
                                            let dy = (my_map.1 - p_to.1) as i64;
                                            dx * dx + dy * dy
                                        };
                                        let d = (d2 as f32).sqrt();
                                        let z = (cgeo.z - my_geo.z).abs();
                                        if z / (d * gs) >= BARREL_TO_SHOT_ANGLE.tan() {
                                            acts.push(Act::AddIgnore(oid));
                                        } else {
                                            let (st, dist) = place_list(
                                                map,
                                                my_chassis,
                                                my_map,
                                                p_to,
                                                float2int(my_maxfd / gs),
                                                false,
                                                &mut place_buf,
                                            );
                                            if st != 0 && ((dist / 4) as i64).pow(2) < d2 {
                                                acts.push(Act::Add(oid));
                                            } else {
                                                acts.push(Act::AddIgnore(oid));
                                            }
                                        }
                                    }
                                }
                            } else {
                                let t2 = fire_r.max(my_maxfd) * 1.5;
                                if cannon.target != Some(rid) && dist_enemy > t2 * t2 {
                                    acts.push(Act::RemoveSlowly(oid));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            } else if gtype == 1 {
                // Radio (MatrixRobot.cpp:4481-4520): merge enemies of
                // same-group robots, or same-team robots in my region.
                let my_region = get_region(map, my_map.0, my_map.1);
                for oid in self.objects.iter_live() {
                    if oid == rid {
                        continue;
                    }
                    let Some(other) = robot_ref(&self.objects, oid) else {
                        continue;
                    };
                    if other.side != my_side {
                        continue;
                    }
                    let same_group = other.group_logic == my_group_logic;
                    let same_team_region = !same_group
                        && other.team == my_team
                        && get_region(map, other.map_x, other.map_y) == my_region;
                    if !(same_group || same_team_region) {
                        continue;
                    }
                    for e in &other.env.enemies {
                        let eside = self
                            .objects
                            .get(e.enemy)
                            .map(|o| o.side())
                            .unwrap_or(my_side);
                        if eside != my_side && e.del_slowly == 0 {
                            acts.push(Act::AddSlowly(e.enemy));
                        }
                    }
                }
            }

            if let Some(r) = robot_mut(&mut self.objects, rid) {
                for a in acts {
                    match a {
                        Act::Add(id) => {
                            if !r.env.search_enemy(id) {
                                r.env.add_to_list(id);
                            }
                        }
                        Act::AddIgnore(id) => r.env.add_ignore(id, now),
                        Act::RemoveSlowly(id) => r.env.remove_from_list_slowly(id),
                        Act::AddSlowly(id) => r.env.add_to_list_slowly(id),
                    }
                }
            }
        }
    }

    /// Port of `CMatrixMap::Takt`'s `SortEndGraphicTakt` call
    /// (MatrixMap.cpp:2501 → MatrixMapStatic.cpp:755-765).
    pub fn graphic_takt(&mut self, step_ms: i32) {
        if step_ms <= 0 {
            return;
        }
        self.objects.graphic_takt(step_ms, &mut self.rng);
    }

    /// Load damage / chassis tables from `robots.dat` into
    /// `g_Config` + per-object-arena caches. Ports the loader
    /// portions of `CMatrixConfig::LoadConfig` invoked at startup.
    pub fn load_config(&mut self, matrix_data: &Storage) {
        self.objects.object_damages =
            ObjectDamages::from_matrix_data(matrix_data).unwrap_or_default();
        self.objects.building_damages =
            BuildingDamages::from_matrix_data(matrix_data).unwrap_or_default();
        let chassis = ChassisChars::from_matrix_data(matrix_data).unwrap_or_default();
        let prices = PriceTable::from_matrix_data(matrix_data).unwrap_or_default();
        let item_chars = ItemCharsTable::from_matrix_data(matrix_data).unwrap_or_default();
        let timings = Timings::from_matrix_data(matrix_data).unwrap_or_default();
        let turrets = TurretProps::from_matrix_data(matrix_data).unwrap_or_default();
        let robot_damages = RobotDamages::from_matrix_data(matrix_data).unwrap_or_default();
        let cannon_damages =
            config::CannonDamages::from_matrix_data(matrix_data).unwrap_or_default();
        let flyer_damages = config::FlyerDamages::from_matrix_data(matrix_data).unwrap_or_default();
        let weapon_radius = config::WeaponRadius::from_matrix_data(matrix_data).unwrap_or_default();
        let weapon_cooldown = WeaponCooldown::from_matrix_data(matrix_data).unwrap_or_default();
        let weapon_strength_ai =
            WeaponStrengthAI::from_matrix_data(matrix_data).unwrap_or_default();
        let overheat = config::Overheat::from_matrix_data(matrix_data).unwrap_or_default();
        let head_chars = HeadCharsTable::from_matrix_data(matrix_data).unwrap_or_default();
        let difficulty = config::Difficulty::from_matrix_data(matrix_data);
        log::info!(
            "config: loaded prices+chars+timings (unit_robot_ms={}, base_hp={})",
            timings.unit_robot,
            item_chars.chassis_structure.iter().sum::<i32>(),
        );
        config::set_global(GlobalConfig {
            chassis,
            prices,
            item_chars,
            timings,
            turrets,
            robot_damages,
            cannon_damages,
            flyer_damages,
            weapon_radius,
            weapon_cooldown,
            weapon_strength_ai,
            overheat,
            head_chars,
            difficulty,
        });
        // String-heavy tables — labels, descriptions, robot-name parts.
        let labels = ItemLabels::from_matrix_data(matrix_data).unwrap_or_default();
        let descriptions = ItemDescriptions::from_matrix_data(matrix_data).unwrap_or_default();
        let robot_names = RobotNameParts::from_matrix_data(matrix_data).unwrap_or_default();
        let buildings = BuildingLabels::from_matrix_data(matrix_data).unwrap_or_default();
        // `AllLabels/Base/none` — the empty-slot row text the C++ pre-
        // assigns to g_PopupHead[0] / g_PopupWeaponNormal[0] /
        // g_PopupWeaponExtern[0] (MatrixGame.cpp:521-523).
        let popup_none = matrix_data
            .block_record("da", "AllLabels")
            .and_then(|labels_rec| matrix_data.block_record(&labels_rec, "Base"))
            .and_then(|base_rec| matrix_data.block_param(&base_rec, "none"))
            .unwrap_or_default();
        log::info!(
            "config: loaded labels (chassis[0]={:?}, robot_names[hull1]={:?}, base_name={:?}, popup_none={:?})",
            labels.chassis.first().map(|s| s.as_str()).unwrap_or(""),
            robot_names.hull.first().map(|s| s.as_str()).unwrap_or(""),
            buildings.base_name,
            popup_none,
        );
        config::set_give_bot_config(
            config::GiveBotConfig::from_matrix_data(matrix_data).unwrap_or_default(),
        );
        config::set_robot_spawn_config(
            config::RobotSpawnConfig::from_matrix_data(matrix_data).unwrap_or_default(),
        );
        let turrets_labels =
            config::TurretLabels::from_matrix_data(matrix_data).unwrap_or_default();
        config::set_global_strings(StringTables {
            labels,
            descriptions,
            robot_names,
            buildings,
            turrets: turrets_labels,
            popup_none,
        });
        // AI robot catalogue (CConstructor.cpp:1361 / SSpecialBot::LoadAIRobotType).
        let ai_robots =
            crate::matrix_game::interface::constructor::AIRobotCatalogue::from_matrix_data(
                matrix_data,
            );
        log::info!("config: loaded {} AI robot templates", ai_robots.bots.len());
        crate::matrix_game::interface::constructor::set_global_ai_robots(ai_robots);
    }

    /// Populate the arena with one [`MapObject`] per decorative
    /// object placed on the map. Ports `CMatrixMap::LoadObjects`'s
    /// `new CMatrixMapObject() + Init(type)` pattern.
    pub fn spawn_map_objects(
        &mut self,
        map: &GameMap,
        map_stor: &Storage,
    ) -> (Vec<ObjectId>, SpawnStats) {
        let mut ids = Vec::with_capacity(map.objects.len());
        let mut stats = SpawnStats::default();

        let strings = map_stor.get_buf("strings", "String");
        let ids_count = strings.map(|s| s.arrays_count()).unwrap_or(0);

        for inst in &map.objects {
            let mut obj = MapObject::from_instance(inst);
            let ids_row = if (inst.type_id as usize) < ids_count {
                strings
                    .map(|s| s.get_as_wstr(inst.type_id as usize))
                    .unwrap_or_default()
            } else {
                String::new()
            };

            let add_lt = if ids_row.is_empty() {
                false
            } else {
                obj.apply_ids_row(&ids_row, &mut self.rng, || {
                    stats.special_win_target += 1;
                })
            };

            stats.bump(obj.beh_flag);

            let id = self.objects.spawn(Box::new(obj));
            if add_lt {
                self.objects.add_lt(id);
            }
            ids.push(id);
        }
        // `++m_BeforeWinCount` per '+'-prefixed behaviour row
        // (MatrixObject.cpp:1020-1026).
        self.before_win_count += stats.special_win_target as i32;
        (ids, stats)
    }

    /// Populate the arena with one [`Building`] per starting base /
    /// turret. Ports `CMatrixMap::LoadBuildings` + `Building::OnLoad`.
    /// Spawn pre-placed decorative base ruins (`side == 255` records) as
    /// ruin-mesh `MapObject`s (MatrixMapPrepare.cpp:648-668). Mirrors the
    /// runtime building→ruin swap in `Building::logic_takt`.
    pub fn spawn_ruins(&mut self, map: &GameMap) {
        use crate::matrix_game::object::MapObject;
        for r in &map.ruins {
            let pos = glam::Vec2::new(r.x, r.y);
            let namet = format!("Matrix/Building/ruins/b{}", r.kind);
            self.objects.spawn(Box::new(MapObject::init_as_base_ruins(
                pos,
                r.build_z,
                r.angle,
                &format!("{namet}.vo"),
            )));
            if r.kind != 0 {
                self.objects.spawn(Box::new(MapObject::init_as_base_ruins(
                    pos,
                    r.build_z,
                    r.angle,
                    &format!("{namet}p.vo"),
                )));
            }
        }
    }

    /// Drain BEHF_SPAWNER requests: build each robot from the `RobotSpawn`
    /// catalogue, spawn it, and order it to attack the nearest player robot
    /// (MatrixObject.cpp:1383-1465). Runs at MapLogic level because the
    /// object layer can't reach the side/order code.
    pub fn drain_spawner_bots(&mut self, map: &GameMap) {
        use crate::matrix_game::interface::constructor::{ArmorUnit, RobotConfig, Unit};
        use crate::matrix_game::map::default_weapon_matrix;
        use crate::matrix_game::object_building::chassis_from_kind;
        use crate::matrix_game::object_robot::{RobotUnitType, MAX_WEAPON_CNT};
        use crate::matrix_game::robot::Robot;
        use crate::matrix_game::config::RobotUnitKind;
        use crate::matrix_game::common::float2int;

        let reqs: Vec<_> = self.objects.pending_spawner_bots.drain(..).collect();
        if reqs.is_empty() {
            return;
        }
        let cat = crate::matrix_game::config::robot_spawn_config();
        for req in reqs {
            let Some(bot) = cat.choose(req.number, req.pick) else {
                continue;
            };
            let Some(chassis) = chassis_from_kind(RobotUnitKind(bot.chassis)) else {
                continue;
            };
            let mut cfg = RobotConfig::new();
            cfg.chassis = Unit {
                ty: RobotUnitType::Chassis,
                kind: RobotUnitKind(bot.chassis),
                price: Default::default(),
            };
            cfg.hull = ArmorUnit {
                max_common_weapon_cnt: 0,
                max_extra_weapon_cnt: 0,
                unit: Unit {
                    ty: RobotUnitType::Armor,
                    kind: RobotUnitKind(bot.armor),
                    price: Default::default(),
                },
            };
            cfg.head = Unit {
                ty: RobotUnitType::Head,
                kind: RobotUnitKind(bot.head),
                price: Default::default(),
            };
            let matrix = default_weapon_matrix(RobotUnitKind(bot.armor));
            let mut placed = [Unit {
                ty: RobotUnitType::Weapon,
                kind: RobotUnitKind(0),
                price: Default::default(),
            }; MAX_WEAPON_CNT];
            for &w in &bot.weapons {
                if w == 0 {
                    continue;
                }
                if let Some(pos) = matrix.find_pylon_for(RobotUnitKind(w), &placed) {
                    placed[pos] = Unit {
                        ty: RobotUnitType::Weapon,
                        kind: RobotUnitKind(w),
                        price: Default::default(),
                    };
                }
            }
            cfg.weapon = placed;

            let spawn_pos = glam::Vec3::new(req.pos.x, req.pos.y, map.get_z(req.pos.x, req.pos.y));
            let mut robot = Robot::new(spawn_pos, bot.side, chassis);
            robot.config = cfg;
            robot.pos_z = spawn_pos.z;
            if bot.hitpoint > 0.0 {
                robot.hit_point_max = bot.hitpoint;
                robot.hit_point = bot.hitpoint;
            }
            let rid = self.objects.spawn(Box::new(robot));
            if let Some(r) = robot_mut(&mut self.objects, rid) {
                r.self_id = Some(rid);
            }
            self.objects.add_lt(rid);

            // Attack the nearest live player robot (FindOnlyPlayerRobotsTgt),
            // else advance on the spawn point.
            let mut target: Option<ObjectId> = None;
            let mut best = f32::MAX;
            for id in self.objects.iter_live() {
                if id == rid {
                    continue;
                }
                let Some(o) = self.objects.get(id) else { continue };
                if o.side() != PLAYER_SIDE
                    || !matches!(o.core().obj_type, ObjectType::RobotAi)
                    || !o.is_live()
                {
                    continue;
                }
                let d = (o.core().geo_center.truncate() - req.pos.truncate()).length_squared();
                // FindObjects bounds the search to the spawner's sensor
                // radius (MatrixObject.cpp:1442); no target → advance on
                // the spawn point.
                if d <= req.sens_radius * req.sens_radius && d < best {
                    best = d;
                    target = Some(id);
                }
            }
            let tp_world = target
                .and_then(|id| self.objects.get(id).map(|o| o.core().geo_center.truncate()))
                .unwrap_or_else(|| req.pos.truncate());
            // C++ dispatches the order on the spawned robot's OWN side
            // (`GetSideById(r->GetSide())->PGOrderAttack(...)`,
            // MatrixObject.cpp:1452-1455). Only the player side's logic
            // groups are modelled here, so route player-side bots through
            // it and leave enemy-side bots to the (unported) enemy AI —
            // routing them through player_side would clobber real player
            // command groups.
            if bot.side == self.player_side.id {
                let gsm = GameMap::GLOBAL_SCALE_MOVE;
                let tp = (float2int(tp_world.x / gsm), float2int(tp_world.y / gsm));
                self.sound_order_attack_disable = true;
                let no = self.robot_to_logic_group(rid);
                self.pg_order_attack(map, no, tp, target);
                self.sound_order_attack_disable = false;
            }
        }
    }

    pub fn spawn_buildings(&mut self, map: &GameMap) -> Vec<ObjectId> {
        let mut ids = Vec::with_capacity(map.buildings.len());
        for inst in &map.buildings {
            let mut b = Building::from_instance(inst);
            let kind_idx = b.kind as usize;
            let hp = self
                .objects
                .building_damages
                .hitpoint
                .get(kind_idx)
                .copied()
                .unwrap_or(0);
            if hp > 0 {
                b.init_max_hitpoint(hp as f32);
            }
            // Port of MatrixMapPrepare.cpp:891 —
            //   cb->m_TurretsMax = min(cb->m_TurretsMax, cb->m_TurretsPlacesCnt);
            // clamps the per-kind `EBuildingTurrets` cap (always 4) to the
            // number of cannon slots actually defined in the CMAP for
            // this building. Without this, every base shows `podl4` even
            // when the map only places 2 or 3 turret slots.
            b.turrets_max = b.turrets_max.min(inst.turrets_places_cnt);
            let id = self.objects.spawn(Box::new(b));
            if let Some(obj) = self.objects.get_mut(id) {
                let b_mut: &mut Building =
                    unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
                b_mut.self_id = Some(id);
            }
            self.objects.add_lt(id);
            ids.push(id);
        }
        // Port of MatrixMapPrepare.cpp:798-885's per-record loop: every
        // CMAP `cannons/*` record (prop=0 standalone or prop=1 mounted)
        // gets a live `CMatrixCannon`. The map loader already did the
        // nearest-base scan and stashed the parent index + slot on
        // each `CannonInstance`.
        let mut spawned_cannons = 0;
        for c in &map.cannons {
            // Resolve parent building's z / id (if any).
            let (parent_id, build_z) = if let Some(bi) = c.parent_building {
                let bid = ids.get(bi).copied();
                let bz = map.buildings.get(bi).map(|b| b.build_z).unwrap_or(0.0);
                (bid, bz)
            } else {
                // Standalone: anchor to terrain height at (x, y).
                let z = map.get_z(c.x, c.y).max(0.0);
                (None, z)
            };
            let cannon = crate::matrix_game::object_cannon::Cannon::new(
                glam::Vec2::new(c.x, c.y),
                build_z + c.add_h,
                c.angle,
                c.side as i32,
                c.kind as i32,
                parent_id,
                c.parent_slot,
            );
            let cid = self.objects.spawn(Box::new(cannon));
            if let Some(obj) = self.objects.get_mut(cid) {
                let c: &mut crate::matrix_game::object_cannon::Cannon = unsafe {
                    &mut *(obj as *mut dyn MapStatic
                        as *mut crate::matrix_game::object_cannon::Cannon)
                };
                c.self_id = Some(cid);
            }
            self.objects.add_lt(cid);
            spawned_cannons += 1;
        }
        log::info!(
            "spawn: {} buildings, {} cannons (from {} records)",
            ids.len(),
            spawned_cannons,
            map.cannons.len()
        );
        ids
    }

    /// Populate the arena with one [`Robot`] per initial-robot record
    /// in the CMAP. Ports the robot-spawn loop of `CMatrixMap::
    /// StaticPrepare2` (MatrixMapPrepare.cpp:1702-1736):
    ///   - `SSpecialBot::GetRobot` (CConstructor.cpp:1284-1357) builds
    ///     the live robot with chassis/armor/weapons/head inserted —
    ///     mirrored here by assembling a `RobotConfig` (with
    ///     `find_pylon_for` placing each weapon on the armor's pylon
    ///     matrix exactly like `CheckWeaponLegality`) and setting
    ///     `robot.config`.
    ///   - `m_Forward = (-sin(angle), cos(angle))` — the chassis token's
    ///     third comma field provides the angle.
    ///   - For non-player sides: `SetTeam(group-1)` if the CMAP group
    ///     index is 1..=3, else `-1` (deferred-team path that
    ///     `GroupNoTeamRobot` later resolves).
    ///
    /// Returns the spawned robots' arena IDs.
    pub fn spawn_robots(&mut self, map: &GameMap) -> Vec<ObjectId> {
        use crate::matrix_game::interface::constructor::{
            name_of_live, raw_hitpoint_of_live, ArmorUnit, RobotConfig, Unit,
        };
        use crate::matrix_game::map::default_weapon_matrix;
        use crate::matrix_game::object_building::{chassis_from_kind, robot_weapons_from_cfg};
        use crate::matrix_game::object_robot::{RobotUnitType, MAX_WEAPON_CNT};
        use crate::matrix_game::robot::Robot;

        let mut ids = Vec::with_capacity(map.robots.len());
        for inst in &map.robots {
            // Resolve chassis enum. If the CMAP rec carries an unknown
            // chassis kind we skip — the C++ `GetRobot` returns NULL on
            // bad sides / config and the caller skips, but in practice
            // every shipped map sets all four units.
            let Some(chassis) = chassis_from_kind(inst.chassis_kind) else {
                log::warn!(
                    "spawn_robots: skipping robot at ({:.1}, {:.1}) — invalid chassis kind {}",
                    inst.x,
                    inst.y,
                    inst.chassis_kind.0,
                );
                continue;
            };

            // Assemble RobotConfig — port of `SSpecialBot::GetRobot`'s
            // unit-insert sequence (CConstructor.cpp:1296-1325). The
            // weapon-matrix re-placement matches the
            // `CheckWeaponLegality` loop at :1303-1314.
            let mut cfg = RobotConfig::new();
            cfg.chassis = Unit {
                ty: RobotUnitType::Chassis,
                kind: inst.chassis_kind,
                price: Default::default(),
            };
            cfg.hull = ArmorUnit {
                max_common_weapon_cnt: 0,
                max_extra_weapon_cnt: 0,
                unit: Unit {
                    ty: RobotUnitType::Armor,
                    kind: inst.armor_kind,
                    price: Default::default(),
                },
            };
            cfg.head = Unit {
                ty: RobotUnitType::Head,
                kind: inst.head_kind,
                price: Default::default(),
            };
            // Weapons: legality-place on the armor's pylon matrix.
            let matrix = default_weapon_matrix(inst.armor_kind);
            let mut placed = [Unit {
                ty: RobotUnitType::Weapon,
                kind: crate::matrix_game::config::RobotUnitKind(0),
                price: Default::default(),
            }; MAX_WEAPON_CNT];
            for w in &inst.weapons {
                if w.0 == 0 {
                    continue;
                }
                if let Some(pos) = matrix.find_pylon_for(*w, &placed) {
                    placed[pos] = Unit {
                        ty: RobotUnitType::Weapon,
                        kind: *w,
                        price: Default::default(),
                    };
                }
            }
            cfg.weapon = placed;

            // Build the live robot. Pos.z = floor z (set by
            // `load_robots` via GetZ), matching MatrixMapPrepare.cpp:761.
            let pos = glam::Vec3::new(inst.x, inst.y, inst.z);
            let mut robot = Robot::new(pos, inst.side as i32, chassis);

            // m_Forward = (-sin(angle), cos(angle)) — :1709-1710.
            robot.forward = glam::Vec2::new(-inst.angle.sin(), inst.angle.cos());
            robot.hull_forward = robot.forward;

            robot.config = cfg;

            // HP from chassis+armor+head (port of CalcRobotMass +
            // InitMaxHitpoint at MatrixRobot.cpp:4150-4293).
            let hp = raw_hitpoint_of_live(&cfg.chassis, &cfg.hull, &cfg.head) as f32;
            if hp > 0.0 {
                robot.hit_point_max = hp;
                robot.hit_point = hp;
            }
            // m_Name — same naming helper used by the construct-from-base
            // path in `BuildStack::pop` (CConstructor.cpp:218).
            robot.name = name_of_live(
                &cfg.chassis,
                &cfg.hull,
                &cfg.head,
                &robot_weapons_from_cfg(&cfg),
            );

            // Team assignment (MatrixMapPrepare.cpp:1724-1732). Player
            // side gets a `PGOrderStop` in the C++; we skip that here
            // because the side-logic group machinery isn't wired yet —
            // the player's initial robots will sit idle until selected.
            if inst.side as i32 != PLAYER_SIDE {
                if (1..=3).contains(&inst.group) {
                    robot.set_team(inst.group - 1);
                } else {
                    // C++ `SetTeam(-1)` — clamp() in our setter floors
                    // to 0 instead of preserving the deferred sentinel.
                    // GroupNoTeamRobot (the C++ post-pass that resolves
                    // -1 into a real team) is not ported yet, so default
                    // to team 0 for now.
                    robot.set_team(0);
                }
            }

            let id = self.objects.spawn(Box::new(robot));
            if let Some(obj) = self.objects.get_mut(id) {
                let r: &mut crate::matrix_game::robot::Robot = unsafe {
                    &mut *(obj as *mut dyn MapStatic as *mut crate::matrix_game::robot::Robot)
                };
                r.self_id = Some(id);
            }
            self.objects.add_lt(id);
            ids.push(id);
        }
        // Per-side breakdown so we can verify the CMAP side byte was
        // parsed correctly (player=1 yellow, AI=2/3/4 red/blue/green).
        let mut by_side = [0i32; 9];
        for inst in &map.robots {
            let s = inst.side as usize;
            if s < by_side.len() {
                by_side[s] += 1;
            }
        }
        log::info!(
            "spawn: {} initial robots (side1={} side2={} side3={} side4={} other={})",
            ids.len(),
            by_side[1],
            by_side[2],
            by_side[3],
            by_side[4],
            by_side[0] + by_side[5] + by_side[6] + by_side[7] + by_side[8],
        );
        ids
    }

    /// Pick the `CurrSel` enum for a given object id — port of the
    /// `switch(ms->GetObjectType())` in `CMatrixSideUnit::SelectObject`
    /// (MatrixSide.cpp).
    fn curr_sel_for(&self, id: ObjectId) -> CurrSel {
        match self.objects.get(id).map(|o| o.core().obj_type) {
            Some(ObjectType::Building) => {
                let is_base = self
                    .objects
                    .get(id)
                    .and_then(|o| {
                        let p = o as *const dyn MapStatic
                            as *const crate::matrix_game::object_building::Building;
                        unsafe { p.as_ref() }.map(|b| {
                            b.kind == crate::matrix_game::object_building::BuildingType::Base
                        })
                    })
                    .unwrap_or(false);
                if is_base {
                    CurrSel::BaseSelected
                } else {
                    CurrSel::BuildingSelected
                }
            }
            Some(ObjectType::RobotAi) => CurrSel::RobotsSelected,
            Some(ObjectType::Cannon) => CurrSel::CannonSelected,
            Some(ObjectType::Flyer) => CurrSel::FlyerSelected,
            _ => CurrSel::Nothing,
        }
    }

    /// Port of the single-click selection entry path — routes
    /// `CMatrixMap::Pick` → `CMatrixSide::SelectObject` +
    /// `CMultiSelection::Add/Remove` (MatrixFormGame.cpp:530-642,
    /// MatrixSide.cpp:1584-1598). `shift` = true matches the C++
    /// shift-modifier branch that toggles the hit in the multi-set
    /// instead of replacing it.
    pub fn click_at_screen(
        &mut self,
        camera: &Camera,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
        shift: bool,
    ) -> Option<ObjectId> {
        let (origin, dir) = camera.screen_to_world_ray(sx, sy, screen_w, screen_h);
        // Cannons are excluded from the click pick — the original
        // doesn't allow selecting them; clicks pass through to the
        // parent base / terrain (MatrixSide.cpp's SelectObject only
        // routes ROBOT / BUILDING / FLYER, not CANNON).
        let mask = TRACE_ANYOBJECT & !crate::matrix_game::common::TRACE_CANNON;
        let hit = self.objects.pick_object(origin, dir, mask, None);
        // `SideSelectionCallBack` (MatrixSide.cpp:1711-1714) rejects
        // anything not owned by the player — enemy / neutral objects
        // are unselectable; the click falls through to "clear".
        let hit = hit.filter(|&(id, _)| self.object_side(id) == self.player_side.id);
        match hit {
            Some((id, _t)) => {
                let sel = self.curr_sel_for(id);
                // Multi-select is only valid on own-side robots — the
                // C++ callback rejects other types (MatrixSide.cpp's
                // SideSelectionCallBack filters on `IsLiveRobot() &&
                // GetSide()==PLAYER_SIDE`). Ctrl+click on a building
                // etc. still does single-select like the C++.
                let own_robot =
                    sel == CurrSel::RobotsSelected && self.object_side(id) == self.player_side.id;
                if shift && own_robot {
                    self.player_side.select_toggle(id, sel);
                } else {
                    self.player_side.select_single(id, sel);
                }
                Some(id)
            }
            None => {
                if !shift {
                    self.player_side.clear();
                }
                None
            }
        }
    }

    /// Back-compat alias for call sites that pre-date shift support.
    /// Equivalent to `click_at_screen(..., shift = false)`.
    pub fn select_at_screen(
        &mut self,
        camera: &Camera,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> Option<ObjectId> {
        self.click_at_screen(camera, sx, sy, screen_w, screen_h, false)
    }

    /// Right-click move order — ports the order-dispatch path at
    /// `CMatrixSideUnit::OnRButtonDown` + `PGOrderMoveTo` +
    /// `PGAssignPlacePlayer` (MatrixSide.cpp:847, 7953-8010, 8694-
    /// 8757). Ray-casts the cursor to the terrain, then runs the
    /// formation-placement spiral search per selected robot so each
    /// ends up at a unique cell around the click point rather than
    /// piling onto the same destination.
    ///
    /// Returns the list of per-robot assigned world-space destinations
    /// (centered on the 4×4 footprint), so the caller can spawn one
    /// move-order ping per slot — matching `PGShowPlace`
    /// (MatrixSide.cpp:8538-8568) which creates a `CMatrixEffectMoveto`
    /// per robot at its own assigned place.
    pub fn order_move_to_at(
        &mut self,
        camera: &Camera,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
        map: &GameMap,
    ) -> Vec<(f32, f32)> {
        let Some((wx, wy)) = screen_to_terrain_xy(camera, map, sx, sy, screen_w, screen_h) else {
            return Vec::new();
        };
        // World XY → move-cell, then the `-ROBOT_MOVECELLS_PER_SIZE/2`
        // footprint offset of MatrixSide.cpp:847.
        let (cmx, cmy) = map.world_to_move(wx, wy);
        let tp = (
            cmx - ROBOT_MOVECELLS_PER_SIZE / 2,
            cmy - ROBOT_MOVECELLS_PER_SIZE / 2,
        );
        let no = self.sel_group_to_logic_group();
        self.pg_order_move_to(map, no, tp);
        // Pings flow through `moveto_pings` (PGShowPlace) now.
        Vec::new()
    }

    /// Port of `CMatrixSideUnit::OnRButtonDown` (MatrixSide.cpp:
    /// 799-863) — the no-preorder right-click dispatch: capture an
    /// enemy building, attack an enemy unit, or move.
    pub fn on_right_click(
        &mut self,
        camera: &Camera,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
        map: &GameMap,
    ) {
        use crate::matrix_game::side::CurrSel;
        if !matches!(self.player_side.curr_sel, CurrSel::RobotsSelected) {
            return;
        }
        if self
            .player_side
            .cur_group()
            .map(|g| g.objects_cnt() == 0)
            .unwrap_or(true)
        {
            return;
        }

        let (origin, dir) = camera.screen_to_world_ray(sx, sy, screen_w, screen_h);
        let hit = self.objects.pick_object(origin, dir, TRACE_ANYOBJECT, None);

        if let Some((id, _)) = hit {
            let (side, is_building, is_unit_live) = {
                let Some(obj) = self.objects.get(id) else {
                    return;
                };
                (
                    obj.side(),
                    matches!(obj.core().obj_type, ObjectType::Building) && obj.is_live(),
                    is_live_unit(&self.objects, id),
                )
            };
            if is_building && side != self.player_side.id {
                // Capture (MatrixSide.cpp:837-839).
                let no = self.sel_group_to_logic_group();
                self.pg_order_capture(map, no, id);
                return;
            }
            if is_unit_live && side != self.player_side.id {
                // Attack (MatrixSide.cpp:841-843).
                if let Some(tp) = get_map_pos(&self.objects, id) {
                    let no = self.sel_group_to_logic_group();
                    self.pg_order_attack(map, no, tp, Some(id));
                }
                return;
            }
            // Own / neutral object under cursor → move next to it
            // (the final `else` arm includes IS_TRACE_STOP_OBJECT).
        }
        self.order_move_to_at(camera, sx, sy, screen_w, screen_h, map);
    }

    /// Right-click attack/repair order — the order-issuing slice of
    /// `CMatrixSideUnit::OnRButtonDown` → `PGOrderAttack` /
    /// `CMatrixRobotAI::Fire` (MatrixSide.cpp:5146/5195). When the
    /// cursor ray hits an object, every selected robot gets a FIRE
    /// order at it: enemies get weapon type 0; own damaged units get
    /// the repair type 2 when the robot mounts a repairer. The group
    /// attack-tactics (move into range, re-target) live in the side AI
    /// and aren't ported yet — out-of-range robots simply hold fire.
    ///
    /// Returns the target when an order was issued.
    pub fn order_fire_at_screen(
        &mut self,
        camera: &Camera,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
        map: &GameMap,
    ) -> Option<ObjectId> {
        // PREORDER_FIRE click (MatrixSide.cpp:702-713): live/special
        // object → PGOrderAttack at it; terrain → attack-move there.
        let (origin, dir) = camera.screen_to_world_ray(sx, sy, screen_w, screen_h);
        let hit = self.objects.pick_object(origin, dir, TRACE_ANYOBJECT, None);
        match hit {
            Some((id, _)) if self.objects.get(id).map(|o| o.is_live()).unwrap_or(false) => {
                let tp = get_map_pos(&self.objects, id)?;
                let no = self.sel_group_to_logic_group();
                self.pg_order_attack(map, no, tp, Some(id));
                Some(id)
            }
            _ => {
                let (wx, wy) = screen_to_terrain_xy(camera, map, sx, sy, screen_w, screen_h)?;
                let (cmx, cmy) = map.world_to_move(wx, wy);
                let tp = (
                    cmx - ROBOT_MOVECELLS_PER_SIZE / 2,
                    cmy - ROBOT_MOVECELLS_PER_SIZE / 2,
                );
                let no = self.sel_group_to_logic_group();
                self.pg_order_attack(map, no, tp, None);
                None
            }
        }
    }

    /// `PGOrderStop` for the selection (the IF_ORDER_STOP button,
    /// CInterface.cpp:3472 → MatrixSide.cpp:7930).
    pub fn order_stop(&mut self, map: &GameMap) {
        let no = self.sel_group_to_logic_group();
        self.pg_order_stop(map, no);
    }

    /// Return the active-selection id if it's still a live object.
    pub fn active_object(&self) -> Option<ObjectId> {
        let id = self.player_side.active_object?;
        if self.objects.is_valid(id) {
            Some(id)
        } else {
            None
        }
    }

    /// Side id of `obj`, or `0` (neutral) if not resolvable.
    pub(crate) fn object_side(&self, id: ObjectId) -> i32 {
        self.objects.get(id).map(|o| o.side()).unwrap_or(0)
    }

    /// Port of `CMatrixSideUnit::GetMaxSideRobots` (MatrixSide.cpp:
    /// 1653-1667). Walks the live-object list, counts this side's
    /// bases + factories, and returns
    /// `bases*ROBOTS_BY_BASE + (bases>0 ? ROBOTS_BY_MAIN : 0) + factories`.
    pub fn compute_max_side_robots(&self, side_id: i32) -> i32 {
        let (bases, factories) = self.count_bases_factories(side_id);
        bases * ROBOTS_BY_BASE + if bases > 0 { ROBOTS_BY_MAIN } else { 0 } + factories
    }

    /// Live-robot count for `side_id` — walks the arena and counts
    /// robots whose `side` matches. The C++ keeps `m_RobotsCnt`
    /// incremented / decremented at spawn / death; we recompute once
    /// per tick which is equivalent for the HUD.
    pub fn compute_side_robots(&self, side_id: i32) -> i32 {
        let mut n = 0;
        for id in self.objects.iter_live() {
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
                continue;
            }
            if obj.side() == side_id {
                n += 1;
            }
        }
        n
    }

    /// Port of `CMatrixSideUnit::GetResourceIncome` (MatrixSide.cpp:
    /// 307-350). Returns `(base_income, factory_income)` in units per
    /// tick-period — the HUD shows their sum.
    ///
    /// `fu` is the force-up multiplier (percent); factories' income is
    /// constant (`RESOURCES_INCOME`), bases' income scales with `fu`.
    pub fn compute_resource_income(&self, side_id: i32, resource: Resource) -> (i32, i32) {
        let target = match resource {
            Resource::Titan => BuildingType::Titan,
            Resource::Electronics => BuildingType::Electronic,
            Resource::Energy => BuildingType::Energy,
            Resource::Plasma => BuildingType::Plasma,
        };
        let (mut bases, mut factories) = (0, 0);
        for id in self.objects.iter_live() {
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            if !matches!(obj.core().obj_type, ObjectType::Building) {
                continue;
            }
            if obj.side() != side_id {
                continue;
            }
            // SAFETY: ObjectType::Building slots only hold `Building`.
            let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
            if matches!(
                b.state,
                crate::matrix_game::object_building::BaseState::Dip
                    | crate::matrix_game::object_building::BaseState::DipExploded
            ) {
                continue;
            }
            if b.kind == BuildingType::Base {
                bases += 1;
            } else if b.kind == target {
                factories += 1;
            }
        }
        let fu = self.side_force_up(side_id);
        let fa_i = factories * RESOURCES_INCOME;
        let base_i = bases * RESOURCES_INCOME_BASE * fu / 100;
        (base_i, fa_i)
    }

    /// Force-up lookup keyed by side id. Only the player side is
    /// modelled for now; other sides fall back to `100` (unmodified).
    fn side_force_up(&self, side_id: i32) -> i32 {
        if side_id == self.player_side.id {
            self.player_side.get_resource_force_up()
        } else {
            100
        }
    }

    /// Tally bases and non-base buildings owned by `side_id`. Shared
    /// helper between [`compute_max_side_robots`] and the income
    /// accrual pass.
    fn count_bases_factories(&self, side_id: i32) -> (i32, i32) {
        let (mut bases, mut factories) = (0, 0);
        for id in self.objects.iter_live() {
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            if !matches!(obj.core().obj_type, ObjectType::Building) {
                continue;
            }
            if obj.side() != side_id {
                continue;
            }
            let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
            if matches!(
                b.state,
                crate::matrix_game::object_building::BaseState::Dip
                    | crate::matrix_game::object_building::BaseState::DipExploded
            ) {
                continue;
            }
            if b.kind == BuildingType::Base {
                bases += 1;
            } else {
                factories += 1;
            }
        }
        (bases, factories)
    }

    /// Port of the `m_ResourcePeriod` tick block in
    /// `CMatrixBuilding::LogicTakt` (MatrixObjectBuilding.cpp:605-667).
    ///
    /// Walks every live building, advances its per-instance
    /// `resource_period` by `cms`, and on threshold crossings emits
    /// income to the owning side. Ports the per-kind branches:
    ///
    /// * Factories (TITAN / ELECTRONIC / ENERGY / PLASMA) emit
    ///   `RESOURCES_INCOME` of the matching resource.
    /// * BASE emits `RESOURCES_INCOME_BASE * fu / 100` of **all four**
    ///   resources.
    ///
    /// Only the player side has a full `Side` struct today; income for
    /// other sides is skipped until `CMatrixSideUnit` proper lands. The
    /// per-building timer still advances (so when AI sides are wired,
    /// their buildings won't all emit a first payout on the same tick).
    pub fn accrue_resources(&mut self, cms: i32) {
        let timings = config::global().timings;
        if cms <= 0 {
            return;
        }
        // Collect building ids first so we can take &mut on Objects /
        // Side without re-entering the arena iterator.
        let ids: Vec<ObjectId> = self
            .objects
            .iter_live()
            .filter(|&id| {
                self.objects
                    .get(id)
                    .map(|o| matches!(o.core().obj_type, ObjectType::Building))
                    .unwrap_or(false)
            })
            .collect();

        for id in ids {
            // Per-kind threshold + payout kind; side captured for the
            // credit below. Every owned building generates income for its
            // side (MatrixObjectBuilding.cpp:597-668), not just the player.
            let (bside, bkind, threshold): (i32, BuildingType, i32);
            let bpos;
            {
                let Some(obj) = self.objects.get_mut(id) else {
                    continue;
                };
                // SAFETY: filtered to ObjectType::Building above.
                let b: &mut Building =
                    unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
                bpos = glam::Vec3::new(b.pos.x, b.pos.y, b.build_z);
                if matches!(
                    b.state,
                    crate::matrix_game::object_building::BaseState::Dip
                        | crate::matrix_game::object_building::BaseState::DipExploded
                ) {
                    continue;
                }
                if b.side == 0 {
                    continue; // neutral / unclaimed — C++ guards on `m_Side`.
                }
                b.resource_period += cms;
                let th = match b.kind {
                    BuildingType::Titan => timings.resource_titan,
                    BuildingType::Electronic => timings.resource_electronics,
                    BuildingType::Energy => timings.resource_energy,
                    BuildingType::Plasma => timings.resource_plasma,
                    BuildingType::Base => timings.resource_base,
                    BuildingType::Repair => continue,
                };
                if th <= 0 || b.resource_period < th {
                    continue;
                }
                b.resource_period = 0;
                bside = b.side;
                bkind = b.kind;
                threshold = th;
            }
            let _ = threshold;
            let is_player = bside == PLAYER_SIDE;
            let Some(su) = self.side_by_id_mut(bside) else {
                continue;
            };
            // Floating "+N" income label for the player's buildings
            // (MatrixObjectBuilding.cpp:355-389). Icon char + amount digits:
            // t/p/e/b factories = "10"; base = "a<amount>".
            let mut score: Option<String> = None;
            match bkind {
                BuildingType::Titan => {
                    su.add_resource_amount(Resource::Titan, RESOURCES_INCOME);
                    if is_player {
                        score = Some("t10".into());
                    }
                }
                BuildingType::Electronic => {
                    su.add_resource_amount(Resource::Electronics, RESOURCES_INCOME);
                    if is_player {
                        score = Some("e10".into());
                    }
                }
                BuildingType::Energy => {
                    su.add_resource_amount(Resource::Energy, RESOURCES_INCOME);
                    if is_player {
                        score = Some("b10".into());
                    }
                }
                BuildingType::Plasma => {
                    su.add_resource_amount(Resource::Plasma, RESOURCES_INCOME);
                    if is_player {
                        score = Some("p10".into());
                    }
                }
                BuildingType::Base => {
                    let ra = RESOURCES_INCOME_BASE * su.get_resource_force_up() / 100;
                    for r in Resource::ALL {
                        su.add_resource_amount(r, ra);
                    }
                    if is_player {
                        score = Some(format!("a{ra}"));
                    }
                }
                BuildingType::Repair => {}
            }
            if let Some(s) = score {
                let pos = glam::Vec3::new(bpos.x, bpos.y, bpos.z + 40.0);
                self.objects.pending_effects.push(
                    crate::matrix_game::effects::GameEffect::Score(
                        crate::matrix_game::effects::billboard_fx::ScoreFx::new(&s, pos),
                    ),
                );
            }
        }
    }

    /// Refresh `m_RobotsCnt` on the player side from the live arena.
    /// Called from [`takt`] each logic slice so the HUD robot counter
    /// is always in sync.
    pub fn refresh_side_robots(&mut self) {
        self.player_side.robots_cnt = self.compute_side_robots(self.player_side.id);
    }

    /// Mirror the kill/build counters accumulated in the object arena
    /// (`Objects::side_stats`) into each side's stat array. The C++
    /// increments `m_Statistic` on the side directly
    /// (`IncStatValue(STAT_*)`); the object code here can't reach the
    /// sides, so the arena accumulates and this copies every takt.
    fn sync_side_stats(&mut self) {
        use crate::matrix_game::side::Stat;
        let ids: Vec<i32> = std::iter::once(self.player_side.id)
            .chain(self.other_sides.iter().map(|s| s.id))
            .collect();
        for sid in ids {
            let Some(&stats) = self.objects.side_stats.get(sid as usize) else {
                continue;
            };
            if let Some(s) = self.side_by_id_mut(sid) {
                s.set_stat(Stat::RobotBuild, stats.robot_build);
                s.set_stat(Stat::RobotKill, stats.robot_kill);
                s.set_stat(Stat::TurretBuild, stats.turret_build);
                s.set_stat(Stat::TurretKill, stats.turret_kill);
                s.set_stat(Stat::BuildingKill, stats.building_kill);
            }
        }
    }
}

/// Intersect the view ray at `(sx, sy)` with the terrain heightmap.
/// Returns the world-space XY hit or `None` when the ray misses (e.g.
/// pointing above the horizon).
///
/// Approximates the C++ `CMatrixCamera::GetCursorOnMap` / CursorHit
/// logic (MatrixCamera.cpp) — a plane-intersection fallback followed
/// by two refinement iterations that re-sample `get_z` under the
/// current XY estimate. Converges in practice for reasonable slopes.
pub fn screen_to_terrain_xy(
    camera: &Camera,
    map: &GameMap,
    sx: f32,
    sy: f32,
    screen_w: f32,
    screen_h: f32,
) -> Option<(f32, f32)> {
    let (origin, dir) = camera.screen_to_world_ray(sx, sy, screen_w, screen_h);
    if dir.z >= -1.0e-4 {
        return None;
    } // pointing up / parallel

    // First hit at z=0.
    let mut t = -origin.z / dir.z;
    let mut x = origin.x + t * dir.x;
    let mut y = origin.y + t * dir.y;
    // Refine against actual terrain height twice — the typical map
    // max slope means this converges to ~pixel accuracy.
    for _ in 0..2 {
        let z = map.get_z(x, y);
        t = (z - origin.z) / dir.z;
        x = origin.x + t * dir.x;
        y = origin.y + t * dir.y;
    }
    // Clamp to map bounds.
    let max_x = map.world_width();
    let max_y = map.world_height();
    if x < 0.0 || y < 0.0 || x >= max_x || y >= max_y {
        return None;
    }
    Some((x, y))
}

// ── Cell / wall helpers (MatrixLogic.cpp:440-903) ───────────────────
//
// The original class methods hang off `CMatrixMapLogic` because they
// read `m_Move` (owned by the map base class) and the live-object
// list (owned by the logic layer). In Rust we split the two into
// `&GameMap` + `&Objects` parameters so any caller can invoke them
// without threading a `&MapLogic` through every order / tick path.

/// Port of `CMatrixMapLogic::IsAbsenceWall` (MatrixLogic.cpp:513-536).
/// Returns true when the precomputed size-N bit for `chassis_kind`
/// is CLEAR at `(mx, my)` — i.e. no wall blocks a placement of
/// `size` cells square here.
pub fn is_absence_wall(map: &GameMap, chassis_kind: usize, size: i32, mx: i32, my: i32) -> bool {
    if mx < 0 || (mx + size) as usize > map.size_move_x {
        return false;
    }
    if my < 0 || (my + size) as usize > map.size_move_y {
        return false;
    }
    map.is_passable_size(mx, my, chassis_kind, size)
}

/// Port of `CMatrixMapLogic::PlaceIsEmpty` (MatrixLogic.cpp:864-903).
/// Combines the size-aware wall check with a scan of all live
/// robots in `objs`: rejects if any non-DIP robot's center lies
/// within `2 * COLLIDE_BOT_R` of the candidate's center, or if
/// any other robot has a destination / return target there.
///
/// `skip` optionally excludes a specific object id (typically the
/// robot doing the query) from the collision check.
pub fn place_is_empty(
    map: &GameMap,
    objs: &Objects,
    chassis_kind: usize,
    size: i32,
    mx: i32,
    my: i32,
    skip: Option<ObjectId>,
) -> bool {
    if !is_absence_wall(map, chassis_kind, size, mx, my) {
        return false;
    }

    let kof = GameMap::GLOBAL_SCALE_MOVE * (ROBOT_MOVECELLS_PER_SIZE as f32) / 2.0;
    let cx = GameMap::GLOBAL_SCALE_MOVE * mx as f32 + kof;
    let cy = GameMap::GLOBAL_SCALE_MOVE * my as f32 + kof;
    let r2 = (COLLIDE_BOT_R * 2.0).powi(2);

    use crate::matrix_game::robot::Robot;

    for id in objs.iter_live() {
        if skip == Some(id) {
            continue;
        }
        let Some(obj) = objs.get(id) else { continue };
        if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
            continue;
        }
        let r: &Robot = unsafe { &*(obj as *const dyn MapStatic as *const Robot) };
        // DIP corpses don't block placement (PlaceIsEmpty, MatrixLogic.cpp:886).
        if !r.is_live() {
            continue;
        }

        let dx = cx - r.pos_x;
        let dy = cy - r.pos_y;
        if dx * dx + dy * dy < r2 {
            return false;
        }

        // GetMoveToCoords — check robot's current MOVE_TO destination.
        if let Some(pt) = r.move_to_coords() {
            let tx = GameMap::GLOBAL_SCALE_MOVE * pt.0 as f32 + kof;
            let ty = GameMap::GLOBAL_SCALE_MOVE * pt.1 as f32 + kof;
            if (cx - tx).powi(2) + (cy - ty).powi(2) < r2 {
                return false;
            }
        }
        // GetReturnCoords — check robot's MOVE_RETURN anchor if any.
        if let Some(pt) = r.return_coords() {
            let tx = GameMap::GLOBAL_SCALE_MOVE * pt.0 as f32 + kof;
            let ty = GameMap::GLOBAL_SCALE_MOVE * pt.1 as f32 + kof;
            if (cx - tx).powi(2) + (cy - ty).powi(2) < r2 {
                return false;
            }
        }
    }
    true
}

/// Collect the occupancy blockers every `PlaceFindNear`-family call
/// scans from the live-object list (MatrixLogic.cpp:713-758): for
/// each non-DIP robot (except `skip`) its MOVE_RETURN anchor, plus
/// either its MOVE_TO destination or its standing map cell; for each
/// live cannon its claimed place cell. All entries use footprint
/// size 4.
///
/// The C++ reads the cannon's road-network place
/// (`m_RN.GetPlace(cannon->m_Place)->m_Pos`); until the cannon side
/// carries a place index we derive the same cell from its position
/// like `ZoneMoveCalc` does at MatrixRobot.cpp:1651.
pub fn collect_place_blockers(objs: &Objects, skip: Option<ObjectId>) -> Vec<(i32, i32)> {
    use crate::matrix_game::common::float2int;
    use crate::matrix_game::object_cannon::{Cannon, CannonState};
    use crate::matrix_game::robot::Robot;

    let mut out: Vec<(i32, i32)> = Vec::new();
    for id in objs.iter_live() {
        if skip == Some(id) {
            continue;
        }
        let Some(obj) = objs.get(id) else { continue };
        match obj.core().obj_type {
            ObjectType::RobotAi => {
                let r: &Robot = unsafe { &*(obj as *const dyn MapStatic as *const Robot) };
                if !r.is_live() {
                    continue;
                }
                if let Some(tp) = r.return_coords() {
                    out.push(tp);
                }
                if let Some(tp) = r.move_to_coords() {
                    out.push(tp);
                } else {
                    out.push((r.map_x, r.map_y));
                }
            }
            ObjectType::Cannon => {
                let c: &Cannon = unsafe { &*(obj as *const dyn MapStatic as *const Cannon) };
                if matches!(c.state, CannonState::Dip) {
                    continue;
                }
                out.push((
                    float2int(c.pos.x / GameMap::GLOBAL_SCALE_MOVE) - ROBOT_MOVECELLS_PER_SIZE / 2,
                    float2int(c.pos.y / GameMap::GLOBAL_SCALE_MOVE) - ROBOT_MOVECELLS_PER_SIZE / 2,
                ));
            }
            _ => {}
        }
    }
    out
}

/// Port of `CMatrixMapLogic::PlaceFindNearReturn` (MatrixLogic.cpp:
/// 760-826): the step-aside variant of `place_find_near`. The blocker
/// list additionally carries the stepping robot's accumulated bad
/// coords, and *every* live robot's current cell + return coords —
/// no skip, so the robot's own spot blocks and the spiral must yield
/// a genuinely different cell. (The original pushes the return coords
/// twice — a harmless quirk we don't replicate.)
pub fn place_find_near_return(
    map: &GameMap,
    objs: &Objects,
    chassis_kind: usize,
    size: i32,
    mx: i32,
    my: i32,
    bad_coords: &[(i32, i32)],
) -> Option<(i32, i32)> {
    use crate::matrix_game::common::float2int;
    use crate::matrix_game::object_cannon::{Cannon, CannonState};
    use crate::matrix_game::robot::{Robot, RobotState};

    let mut blockers: Vec<(i32, i32)> = bad_coords.to_vec();
    for id in objs.iter_live() {
        let Some(obj) = objs.get(id) else { continue };
        match obj.core().obj_type {
            ObjectType::RobotAi => {
                let r: &Robot = unsafe { &*(obj as *const dyn MapStatic as *const Robot) };
                if matches!(r.state, RobotState::Dip) {
                    continue;
                }
                blockers.push((r.map_x, r.map_y));
                if let Some(tp) = r.return_coords() {
                    blockers.push(tp);
                }
                if let Some(tp) = r.move_to_coords() {
                    blockers.push(tp);
                } else {
                    blockers.push((r.map_x, r.map_y));
                }
            }
            ObjectType::Cannon => {
                let c: &Cannon = unsafe { &*(obj as *const dyn MapStatic as *const Cannon) };
                if matches!(c.state, CannonState::Dip) {
                    continue;
                }
                blockers.push((
                    float2int(c.pos.x / GameMap::GLOBAL_SCALE_MOVE) - ROBOT_MOVECELLS_PER_SIZE / 2,
                    float2int(c.pos.y / GameMap::GLOBAL_SCALE_MOVE) - ROBOT_MOVECELLS_PER_SIZE / 2,
                ));
            }
            _ => {}
        }
    }
    place_find_near_with_blockers(map, chassis_kind, size, mx, my, &blockers)
}

/// Port of `CMatrixMapLogic::PlaceFindNear(nsh, size, mx, my, skip)`
/// (MatrixLogic.cpp:713-758): collects the live robots' / cannons'
/// claimed cells, then spirals outward from `(mx, my)` via the
/// blocker variant. Returns the nearest valid cell.
pub fn place_find_near(
    map: &GameMap,
    objs: &Objects,
    chassis_kind: usize,
    size: i32,
    mx: i32,
    my: i32,
    skip: Option<ObjectId>,
) -> Option<(i32, i32)> {
    let blockers = collect_place_blockers(objs, skip);
    place_find_near_with_blockers(map, chassis_kind, size, mx, my, &blockers)
}

/// Port of `CMatrixMapLogic::PlaceFindNear` with blocker list
/// (MatrixLogic.cpp:572-627). Spirals outward from `(mx, my)` and
/// returns the first cell whose `size × size` footprint passes
/// `is_absence_wall` AND doesn't overlap any `(bx, by)` blocker
/// footprint. Each blocker is the upper-left corner of another
/// robot's claimed `size × size` placement.
///
/// AABB-overlap matches the C++ predicate at :582, :595, etc.:
///   `!(bx + size <= tx || bx >= tx + size)
///    && !(by + size <= ty || by >= ty + size)`
///
/// Unlike [`place_find_near`] this variant ignores the live-robot
/// arena — callers pass the relevant blockers directly, which lets
/// the group-formation loop include in-progress same-group
/// placements that aren't yet written back to the arena.
pub fn place_find_near_with_blockers(
    map: &GameMap,
    chassis_kind: usize,
    size: i32,
    mx: i32,
    my: i32,
    blockers: &[(i32, i32)],
) -> Option<(i32, i32)> {
    let overlap = |tx: i32, ty: i32| -> bool {
        for &(bx, by) in blockers {
            // Equivalent to the C++ AABB predicate at :582 — two `size`-
            // sized squares overlap iff neither axis separates them.
            let sep_x = bx + size <= tx || bx >= tx + size;
            let sep_y = by + size <= ty || by >= ty + size;
            if !sep_x && !sep_y {
                return true;
            }
        }
        false
    };
    // Start cell — C++ tests it directly before spiraling.
    if is_absence_wall(map, chassis_kind, size, mx, my) && !overlap(mx, my) {
        return Some((mx, my));
    }
    // Spiral shell-by-shell. Each shell of radius `i+1` visits the
    // top/bottom rows first (length `2(i+1)+1`), then the left/right
    // columns minus corners (length `2i+1`). Matches the interleaved
    // loops at MatrixLogic.cpp:590-623.
    let limit = map.size_move_x.max(map.size_move_y) as i32;
    let mut i = 0i32;
    while i < limit {
        // Top / bottom rows.
        for u in 0..(i + 1) * 2 + 1 {
            for (tx, ty) in [
                (mx - (i + 1) + u, my - (i + 1)), // top row
                (mx - (i + 1) + u, my + (i + 1)), // bottom row
            ] {
                if is_absence_wall(map, chassis_kind, size, tx, ty) && !overlap(tx, ty) {
                    return Some((tx, ty));
                }
            }
        }
        // Left / right columns (corners already covered above).
        for u in 0..i * 2 + 1 {
            for (tx, ty) in [
                (mx - (i + 1), my - i + u), // left col
                (mx + (i + 1), my - i + u), // right col
            ] {
                if is_absence_wall(map, chassis_kind, size, tx, ty) && !overlap(tx, ty) {
                    return Some((tx, ty));
                }
            }
        }
        i += 1;
    }
    None
}

/// Per-BehFlag spawn counts collected during `spawn_map_objects`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpawnStats {
    pub r#static: u32,
    pub burn: u32,
    pub r#break: u32,
    pub anim: u32,
    pub sens: u32,
    pub spawner: u32,
    pub terron: u32,
    pub portret: u32,
    pub special_win_target: u32,
}

impl SpawnStats {
    fn bump(&mut self, flag: crate::matrix_game::object::BehFlag) {
        use crate::matrix_game::object::BehFlag::*;
        match flag {
            Static => self.r#static += 1,
            Burn => self.burn += 1,
            Break => self.r#break += 1,
            Anim => self.anim += 1,
            Sens => self.sens += 1,
            Spawner => self.spawner += 1,
            Terron => self.terron += 1,
            Portret => self.portret += 1,
        }
    }

    pub fn total(&self) -> u32 {
        self.r#static
            + self.burn
            + self.r#break
            + self.anim
            + self.sens
            + self.spawner
            + self.terron
            + self.portret
    }
}

// ── Rnd: CMatrixMapLogic's Park–Miller MINSTD LCG ───────────────────────
//
// Port of the RNG owned by `CMatrixMapLogic` (MatrixLogic.cpp:84-113).
// Park–Miller MINSTD LCG with a 32-bit state — the generator the
// original game uses for all deterministic-world decisions (object
// animation timers, spawn jitter, tactical noise, etc.). The C++ seeds
// the generator from `rand()` once at construction and burns one output
// with `Rnd(0,1)` to mix the seed in (MatrixLogic.cpp:49). `Rnd::new`
// reproduces that contract for bit-for-bit parity.

/// Recurrence constants from `CMatrixMapLogic::Rnd` (MatrixLogic.cpp:88).
/// `m_Rnd = 16807 * (m_Rnd % 127773) - 2836 * (m_Rnd / 127773)` —
/// the classic Schrage-factored MINSTD step. Output is `m_Rnd - 1` so
/// the stream starts at 0 instead of 1.
const MINSTD_A: i32 = 16_807;
const MINSTD_Q: i32 = 127_773; // 2^31-1 / A
const MINSTD_R: i32 = 2_836; // 2^31-1 % A
const MINSTD_M_MINUS_1: i32 = 2_147_483_647; // 2^31 - 1

pub struct Rnd {
    /// Matches `m_Rnd` in CMatrixMapLogic (MatrixLogic.hpp:75).
    /// Must stay strictly positive; the step reseeds to +(2^31-1) when
    /// it ever lands at or below zero, so the state never gets stuck.
    state: i32,
}

impl Rnd {
    /// Construct with an explicit seed — matches setting `m_Rnd = seed`
    /// before the constructor's `Rnd(0,1)` mix-in.
    pub fn new(seed: i32) -> Self {
        let mut r = Self { state: seed };
        let _ = r.range(0, 1);
        r
    }

    /// Seed from the platform clock — what the original effectively
    /// does via `m_Rnd=rand()`. Kept separate so tests can construct
    /// deterministic streams via [`Rnd::new`].
    pub fn from_clock() -> Self {
        let now = crate::platform::now_secs();
        let bits = now.to_bits() as u64;
        let seed = ((bits ^ (bits >> 32)) as u32 & 0x7FFF_FFFF) as i32;
        Self::new(if seed == 0 { 1 } else { seed })
    }

    /// Raw `CMatrixMapLogic::Rnd()` — one step of the generator.
    /// Result in `[0, 2^31-2]` (matches the C++ `return m_Rnd-1`).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> i32 {
        self.state = MINSTD_A.wrapping_mul(self.state % MINSTD_Q)
            - MINSTD_R.wrapping_mul(self.state / MINSTD_Q);
        if self.state <= 0 {
            self.state = self.state.wrapping_add(MINSTD_M_MINUS_1);
        }
        self.state - 1
    }

    /// `Rnd(zmin, zmax)` — inclusive range on both ends. Mirrors the
    /// C++ semantics where `zmin > zmax` swaps the endpoints
    /// (MatrixLogic.cpp:100-106).
    pub fn range(&mut self, zmin: i32, zmax: i32) -> i32 {
        if zmin <= zmax {
            zmin + (self.next() % (zmax - zmin + 1))
        } else {
            zmax + (self.next() % (zmin - zmax + 1))
        }
    }

    /// `RndFloat()` — uniform `[0, 1]`.
    pub fn float01(&mut self) -> f64 {
        self.next() as f64 / (MINSTD_M_MINUS_1 as f64 - 2.0)
    }

    /// `RndFloat(zmin, zmax)` — uniform float in `[zmin, zmax)`.
    pub fn float_range(&mut self, zmin: f64, zmax: f64) -> f64 {
        zmin + self.float01() * (zmax - zmin)
    }
}

#[cfg(test)]
mod rnd_tests {
    use super::*;

    #[test]
    fn stream_is_deterministic_for_a_fixed_seed() {
        let mut a = Rnd::new(12345);
        let mut b = Rnd::new(12345);
        for _ in 0..32 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn range_respects_bounds_and_allows_swapped_endpoints() {
        let mut r = Rnd::new(42);
        for _ in 0..200 {
            let v = r.range(-5, 10);
            assert!((-5..=10).contains(&v));
        }
        for _ in 0..200 {
            let v = r.range(10, -5);
            assert!((-5..=10).contains(&v));
        }
    }

    #[test]
    fn float01_is_in_unit_interval() {
        let mut r = Rnd::new(7);
        for _ in 0..1000 {
            let f = r.float01();
            assert!((0.0..=1.0).contains(&f));
        }
    }

    #[test]
    fn first_draws_match_hand_simulation() {
        let mut r = Rnd::new(1);
        let v0 = r.next();
        let v1 = r.next();
        let step = |s: &mut i32| {
            *s = MINSTD_A.wrapping_mul(*s % MINSTD_Q) - MINSTD_R.wrapping_mul(*s / MINSTD_Q);
            if *s <= 0 {
                *s = s.wrapping_add(MINSTD_M_MINUS_1);
            }
        };
        let mut expect = 1i32;
        step(&mut expect);
        let _ = expect;
        step(&mut expect);
        assert_eq!(v0, expect - 1);
        step(&mut expect);
        assert_eq!(v1, expect - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_game::map_static::{MapStatic, ObjectCore, ObjectType};
    use std::cell::RefCell;
    use std::rc::Rc;

    struct Counter {
        core: ObjectCore,
        rchange: u32,
        state: u32,
        ablaze: i32,
        shorted: i32,
        calls: Rc<RefCell<Vec<i32>>>,
    }
    impl MapStatic for Counter {
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
            self.state
        }
        fn object_state_set(&mut self, b: u32) {
            self.state |= b;
        }
        fn object_state_clear(&mut self, b: u32) {
            self.state &= !b;
        }
        fn ablaze_ttl(&self) -> i32 {
            self.ablaze
        }
        fn set_ablaze_ttl(&mut self, t: i32) {
            self.ablaze = t;
        }
        fn shorted_ttl(&self) -> i32 {
            self.shorted
        }
        fn set_shorted_ttl(&mut self, t: i32) {
            self.shorted = t;
        }
        fn r_need(&mut self, _: u32) {}
        fn takt(&mut self, _: i32, _: &mut Rnd, _: &mut crate::matrix_game::map_static::Objects) {}
        fn logic_takt(
            &mut self,
            cms: i32,
            _: &mut Rnd,
            _: &mut crate::matrix_game::map_static::Objects,
        ) {
            self.calls.borrow_mut().push(cms);
        }
    }

    #[test]
    fn takt_decomposes_step_into_full_portions_plus_remainder() {
        let mut w = MapLogic::new();
        let calls = Rc::new(RefCell::new(Vec::new()));
        let id = w.objects.spawn(Box::new(Counter {
            core: ObjectCore {
                obj_type: ObjectType::MapObject,
                ..Default::default()
            },
            rchange: 0,
            state: 0,
            ablaze: 0,
            shorted: 0,
            calls: calls.clone(),
        }));
        w.objects.add_lt(id);

        w.takt(25);
        // 25 = 2 * 10 + 5  →  [10, 10, 5]
        assert_eq!(calls.borrow().clone(), vec![10, 10, 5]);
        assert_eq!(w.tick, 2);
        assert_eq!(w.elapsed_ms, 25);
    }

    #[test]
    fn takt_with_exact_multiple_has_no_remainder_call() {
        let mut w = MapLogic::new();
        let calls = Rc::new(RefCell::new(Vec::new()));
        let id = w.objects.spawn(Box::new(Counter {
            core: ObjectCore {
                obj_type: ObjectType::MapObject,
                ..Default::default()
            },
            rchange: 0,
            state: 0,
            ablaze: 0,
            shorted: 0,
            calls: calls.clone(),
        }));
        w.objects.add_lt(id);

        w.takt(30);
        assert_eq!(calls.borrow().clone(), vec![10, 10, 10]);
        assert_eq!(w.tick, 3);
    }

    #[test]
    fn takt_with_sub_period_step_uses_only_remainder() {
        let mut w = MapLogic::new();
        let calls = Rc::new(RefCell::new(Vec::new()));
        let id = w.objects.spawn(Box::new(Counter {
            core: ObjectCore {
                obj_type: ObjectType::MapObject,
                ..Default::default()
            },
            rchange: 0,
            state: 0,
            ablaze: 0,
            shorted: 0,
            calls: calls.clone(),
        }));
        w.objects.add_lt(id);

        w.takt(4);
        assert_eq!(calls.borrow().clone(), vec![4]);
        assert_eq!(w.tick, 0);
        assert_eq!(w.elapsed_ms, 4);
    }

    #[test]
    fn takt_with_nonpositive_step_is_noop() {
        let mut w = MapLogic::new();
        w.takt(0);
        w.takt(-5);
        assert_eq!(w.tick, 0);
        assert_eq!(w.elapsed_ms, 0);
    }

    #[test]
    fn last_special_target_death_sets_just_win() {
        use crate::matrix_game::map_static::SpecialDeathKind;
        use crate::matrix_game::side::SideStatus;
        let mut w = MapLogic::new();
        w.before_win_count = 2;
        w.objects
            .pending_special_deaths
            .push(SpecialDeathKind::Target);
        w.takt(10);
        // One of two targets down — no win yet.
        assert_eq!(w.before_win_count, 1);
        assert!(!w.special_broken);
        assert_ne!(w.player_side.status, SideStatus::JustWin);

        w.objects
            .pending_special_deaths
            .push(SpecialDeathKind::Target);
        w.takt(10);
        assert_eq!(w.before_win_count, 0);
        assert!(w.special_broken);
        // check_status (gated to ~1s) hasn't consumed JUST_WIN yet at
        // 20ms elapsed, so the status must still be visible here.
        assert_eq!(w.player_side.status, SideStatus::JustWin);
    }

    #[test]
    fn terron_death_wins_regardless_of_remaining_targets() {
        use crate::matrix_game::map_static::SpecialDeathKind;
        use crate::matrix_game::side::SideStatus;
        let mut w = MapLogic::new();
        w.before_win_count = 5;
        w.objects
            .pending_special_deaths
            .push(SpecialDeathKind::Terron);
        w.takt(10);
        assert_eq!(w.before_win_count, 4);
        assert!(w.win_flag);
        assert_eq!(w.player_side.status, SideStatus::JustWin);
    }

    // ── Economy / robot-limit tests ───────────────────────────────
    //
    // These seed a fresh `MapLogic` with hand-rolled `Building`
    // instances so the ports of GetMaxSideRobots / GetResourceIncome /
    // the per-building resource tick can be exercised without the full
    // map-load path.
    use crate::matrix_game::map::BuildingInstance;
    use crate::matrix_game::object_building::{Building, BuildingType};

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

    #[test]
    fn max_side_robots_counts_bases_and_factories() {
        let mut w = MapLogic::new();
        // Empty side: no robots allowed.
        assert_eq!(w.compute_max_side_robots(PLAYER_SIDE), 0);

        // One base alone: 3 + 4 = 7.
        spawn_building(&mut w, PLAYER_SIDE, BuildingType::Base, 0.0, 0.0);
        assert_eq!(w.compute_max_side_robots(PLAYER_SIDE), 7);

        // Add a factory: + 1 = 8.
        spawn_building(&mut w, PLAYER_SIDE, BuildingType::Titan, 50.0, 0.0);
        assert_eq!(w.compute_max_side_robots(PLAYER_SIDE), 8);

        // Add a second base: 2*3 + 4 + 1 = 11.
        spawn_building(&mut w, PLAYER_SIDE, BuildingType::Base, 100.0, 0.0);
        assert_eq!(w.compute_max_side_robots(PLAYER_SIDE), 11);

        // Neutral-owned building doesn't count.
        spawn_building(&mut w, 0, BuildingType::Plasma, 200.0, 0.0);
        assert_eq!(w.compute_max_side_robots(PLAYER_SIDE), 11);
    }

    #[test]
    fn resource_income_matches_cpp_formula() {
        let mut w = MapLogic::new();
        w.player_side.set_resource_force_up(100);
        spawn_building(&mut w, PLAYER_SIDE, BuildingType::Base, 0.0, 0.0);
        spawn_building(&mut w, PLAYER_SIDE, BuildingType::Titan, 50.0, 0.0);
        spawn_building(&mut w, PLAYER_SIDE, BuildingType::Titan, 100.0, 0.0);

        // One base (3 @ 100%) + two titan factories (10 each) =
        //   base_i = 1*3*100/100 = 3
        //   fa_i   = 2*10 = 20
        let (base_i, fa_i) = w.compute_resource_income(PLAYER_SIDE, Resource::Titan);
        assert_eq!(base_i, 3);
        assert_eq!(fa_i, 20);

        // Plasma factory missing → only base income.
        let (base_i, fa_i) = w.compute_resource_income(PLAYER_SIDE, Resource::Plasma);
        assert_eq!(base_i, 3);
        assert_eq!(fa_i, 0);

        // Force-up doubles the base contribution (factories are flat).
        w.player_side.set_resource_force_up(200);
        let (base_i, fa_i) = w.compute_resource_income(PLAYER_SIDE, Resource::Titan);
        assert_eq!(base_i, 6);
        assert_eq!(fa_i, 20);
    }

    #[test]
    fn accrue_resources_emits_on_threshold() {
        // Preseed a lightweight Timings — in tests we don't run
        // `load_config`, so the globals default to 0 and nothing fires.
        // Inject directly via `config::set_global` using values close
        // to the shipping robots.dat (titan=10_000 ms, base=15_000 ms).
        let mut g = config::global();
        g.timings.resource_titan = 10_000;
        g.timings.resource_electronics = 10_000;
        g.timings.resource_energy = 10_000;
        g.timings.resource_plasma = 10_000;
        g.timings.resource_base = 15_000;
        config::set_global(g);

        let mut w = MapLogic::new();
        w.player_side.resources = [0; crate::matrix_game::config::MAX_RESOURCES];
        w.player_side.set_resource_force_up(100);
        spawn_building(&mut w, PLAYER_SIDE, BuildingType::Titan, 0.0, 0.0);
        spawn_building(&mut w, PLAYER_SIDE, BuildingType::Base, 50.0, 0.0);

        // Under threshold: nothing credited.
        w.accrue_resources(5_000);
        assert_eq!(
            w.player_side.resources,
            [0; crate::matrix_game::config::MAX_RESOURCES]
        );

        // Past titan threshold only (10_000 < 15_000): +10 titan,
        // 0 elsewhere.
        w.accrue_resources(6_000); // total 11_000 for titan / base
        assert_eq!(
            w.player_side.resources[Resource::Titan as usize],
            RESOURCES_INCOME
        );
        assert_eq!(w.player_side.resources[Resource::Plasma as usize], 0);

        // Past base threshold too: base emits +3 to all four.
        w.accrue_resources(5_000); // base now at 16_000
        assert_eq!(
            w.player_side.resources[Resource::Titan as usize],
            RESOURCES_INCOME + RESOURCES_INCOME_BASE
        );
        for r in [Resource::Electronics, Resource::Energy, Resource::Plasma] {
            assert_eq!(
                w.player_side.resources[r as usize], RESOURCES_INCOME_BASE,
                "base income missing for {r:?}"
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Local A* pathfinding — ports `CMatrixMapLogic::FindLocalPath` and
// `OptimizeMovePath` from MatrixLogic.cpp:1217-1400+.
//
// `CMatrixMap::FindLocalPath` is the top-level symbol in the C++; the
// implementation lives alongside the logic RNG and `IsAbsenceWall`
// passability predicate in `MatrixLogic.cpp`, hence owned here.
//
// `MatrixMapTrace.cpp` is reserved for the landscape / object ray-cast
// side of `CMatrixMap::Trace`, which is a separate concern (picking,
// weapon hit-tests) implemented in a different file in the original.
//
// Waypoint semantics: each path cell `(mx, my)` is the upper-left
// corner of the robot's 4×4 move-cell footprint; the cell center in
// world coords is `((mx + 2) * GLOBAL_SCALE_MOVE, (my + 2) * GLOBAL_SCALE_MOVE)`.
// ════════════════════════════════════════════════════════════════════════

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
    /// for stationary objects where only the final pos is known —
    /// C++ `other_path_cnt[i] == 0`.
    pub pos: Option<MovePt>,
    /// Destination cell (upper-left corner of footprint).
    pub dest: MovePt,
    /// Footprint side in move cells — C++ `other_size[i]` (always 4
    /// for robots/cannons today).
    pub size: i32,
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

/// Axis-aligned rect over move cells with exclusive right/bottom,
/// mirroring how `FindLocalPath` treats `Base::CRect` after the
/// clamps at MatrixLogic.cpp:1251-1259.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// Result of [`find_local_path`]. The C++ stamps `m_Weight` into the
/// persistent move grid and `OptimizeMovePath` reads those stamps
/// right after `FindLocalPath` returns (MatrixRobot.cpp:1658-1681);
/// we thread the same data through this struct instead of mutating
/// the map. Cells outside the search rect read 0 — same as freshly
/// `HAllocClear`ed grid memory the C++ starts from.
pub struct LocalPathResult {
    pub path: Vec<MovePt>,
    pub weight: Vec<i32>,
}

/// Stamp one blocker window into the weight grid. Port of the two
/// loops at MatrixLogic.cpp:1278-1300, including the original's
/// row-advance arithmetic `smm += m_SizeMove.x - (ey - sy)` (note:
/// `ey - sy`, not `ex - sx` — at clamped map borders the C++ pointer
/// walk skews rows; we reproduce it via the same linear-index math).
fn stamp_blocker_window(
    weight: &mut [i32],
    stride: usize,
    sx_map: i32,
    sy_map: i32,
    c: MovePt,
    bsize: i32,
    wval: i32,
) {
    let sx = 0.max(c.x - (bsize - 1));
    let sy = 0.max(c.y - (bsize - 1));
    let ex = sx_map.min(c.x + bsize);
    let ey = sy_map.min(c.y + bsize);
    if sy >= ey {
        return;
    }
    let mut idx = sy as usize * stride + sx as usize;
    for _y in sy..ey {
        for _x in sx..ex {
            if idx >= weight.len() {
                return;
            }
            if weight[idx] < wval {
                weight[idx] = wval;
            }
            idx += 1;
        }
        // C++ for-loop increment: `smm += m_SizeMove.x - (ey - sy)`.
        // The window height (≤ 2*size-1) is always smaller than the
        // grid stride, so this cannot underflow.
        idx += stride - (ey - sy) as usize;
    }
}

/// Port of `CMatrixMapLogic::FindLocalPath` (MatrixLogic.cpp:1217-
/// 1435) — the 4-way cost-wave search the original uses for all
/// robot movement. Faithful details preserved:
///
///   - The search window is the union of the first
///     `min(zone_path.len()-1, 6) + 1` zone bounding rects, expanded
///     by the start cell and a `size` margin (:1245-1259). When a
///     zone rect is unavailable (`zone_rect_of` returns `None`) the
///     window falls back to the whole map.
///   - Cell weights: base 5, other-robot standing cell 30, other
///     robot/cannon destination 200 (:1267-1300).
///   - Neighbors of the *start* cell skip the impassability mask
///     (`sme != 0` at :1349) so a robot can escape a blocked cell.
///   - Exact-target match wins immediately; otherwise, when more
///     zones remain beyond the window, the cheapest cell belonging
///     to the window's last zone is tracked as `findbest`
///     (:1354-1360).
///   - The wave keeps expanding 16 levels past the first hit to
///     improve `findbest` (:1379-1386).
///   - Backtrace walks descending `m_Find` with a `lastangle`
///     direction preference (:1391-1427) and caps the path at
///     `MatrixPathMoveMax` cells.
///
/// `zone_path` must contain at least the current zone (the C++
/// asserts `zonepathcnt >= 1` and callers pass `&m_ZoneCur, 1` when
/// no zone path exists — MatrixRobot.cpp:1662).
#[allow(clippy::too_many_arguments)]
pub fn find_local_path(
    map: &GameMap,
    nsh: usize,
    size: i32,
    mx: i32,
    my: i32,
    zone_path: &[i32],
    zone_rect_of: &dyn Fn(i32) -> Option<MoveRect>,
    dx: i32,
    dy: i32,
    others: &[Blocker],
) -> LocalPathResult {
    const MATRIX_PATH_MOVE_MAX: usize = 256; // MatrixMap.hpp:22

    let sx_map = map.size_move_x as i32;
    let sy_map = map.size_move_y as i32;
    let stride = map.size_move_x;
    let mut weight = vec![0i32; stride * map.size_move_y];
    let fail = |weight: Vec<i32>| LocalPathResult {
        path: Vec::new(),
        weight,
    };

    debug_assert!(!zone_path.is_empty());
    if zone_path.is_empty() || map.move_cell(mx, my).is_none() {
        return fail(weight);
    }

    // Search window — union of the first zoneskipcnt+1 zone rects
    // (MatrixLogic.cpp:1245-1259).
    let zoneskipcnt = ((zone_path.len() as i32 - 1).min(6)) as usize;
    let full_map = MoveRect {
        left: 0,
        top: 0,
        right: sx_map,
        bottom: sy_map,
    };
    let mut re = zone_rect_of(zone_path[zoneskipcnt]).unwrap_or(full_map);
    for &z in &zone_path[..zoneskipcnt] {
        let r = zone_rect_of(z).unwrap_or(full_map);
        re.left = re.left.min(r.left);
        re.top = re.top.min(r.top);
        re.right = re.right.max(r.right);
        re.bottom = re.bottom.max(r.bottom);
    }
    re.left = re.left.min(mx);
    re.top = re.top.min(my);
    re.right = re.right.max(mx + 1);
    re.bottom = re.bottom.max(my + 1);
    re.left = (re.left - size).max(0);
    re.top = (re.top - size).max(0);
    re.right = (re.right + size).min(sx_map - size);
    re.bottom = (re.bottom + size).min(sy_map - size);
    if re.left >= re.right || re.top >= re.bottom {
        return fail(weight);
    }

    // Reset weight=5 inside the window (:1263-1269); `find` starts -1.
    let mut find = vec![-1i32; stride * map.size_move_y];
    for y in re.top..re.bottom {
        let row = y as usize * stride;
        for x in re.left..re.right {
            weight[row + x as usize] = 5;
        }
    }

    // Blocker stamps (:1271-1301): destination 200, standing cell 30.
    for b in others {
        stamp_blocker_window(&mut weight, stride, sx_map, sy_map, b.dest, b.size, 200);
        if let Some(p) = b.pos {
            stamp_blocker_window(&mut weight, stride, sx_map, sy_map, p, b.size, 30);
        }
    }

    let idx = |x: i32, y: i32| -> usize { y as usize * stride + x as usize };
    let nsh_mask = MoveCell::stop_mask(nsh, size);
    let stop_at = |x: i32, y: i32| -> bool {
        map.move_cell(x, y)
            .map(|c| (c.stop & nsh_mask) != 0)
            .unwrap_or(true)
    };
    let zone_at = |x: i32, y: i32| -> i32 { map.move_cell(x, y).map(|c| c.zone).unwrap_or(-1) };

    // Cost wave (:1316-1387).
    let maxmax = stride * map.size_move_y;
    let mut queue: Vec<MovePt> = Vec::with_capacity(1024);
    queue.push(MovePt::new(mx, my));
    find[idx(mx, my)] = 0;
    let mut sme = 0usize;
    let mut cnt = 1usize;
    let mut next = cnt;
    let mut findok = 0i32;
    let mut findbest = -1i32;
    let mut tpfind = MovePt::new(mx, my);

    'wave: while sme < cnt {
        let cur = queue[sme];
        let cur_find = find[idx(cur.x, cur.y)];
        for u in 0..4 {
            let tp = match u {
                0 => {
                    if cur.x - 1 < re.left {
                        continue;
                    }
                    MovePt::new(cur.x - 1, cur.y)
                }
                1 => {
                    if cur.y - 1 < re.top {
                        continue;
                    }
                    MovePt::new(cur.x, cur.y - 1)
                }
                2 => {
                    if cur.x + 1 >= re.right {
                        continue;
                    }
                    MovePt::new(cur.x + 1, cur.y)
                }
                _ => {
                    if cur.y + 1 >= re.bottom {
                        continue;
                    }
                    MovePt::new(cur.x, cur.y + 1)
                }
            };
            // Neighbors of the start cell skip the stop mask (:1349).
            if sme != 0 && stop_at(tp.x, tp.y) {
                continue;
            }
            let ti = idx(tp.x, tp.y);
            let new_cost = cur_find + weight[ti];
            if find[ti] >= 0 && new_cost >= find[ti] {
                continue;
            }

            if findok == 0 && tp.x == dx && tp.y == dy {
                tpfind = tp;
                findok = 1;
            } else if (findok == 0 || findbest >= 0)
                && zoneskipcnt + 1 < zone_path.len()
                && zone_at(tp.x, tp.y) == zone_path[zoneskipcnt]
            {
                if findbest >= 0 && new_cost >= findbest {
                    continue;
                }
                findbest = new_cost;
                tpfind = tp;
                findok = 1;
            }

            if cnt >= maxmax {
                break 'wave;
            }
            find[ti] = new_cost;
            queue.push(tp);
            cnt += 1;
        }
        sme += 1;
        if sme >= next {
            next = cnt;
            if findok != 0 {
                findok += 1;
                if findok > 16 {
                    break;
                }
            }
        }
    }

    if findok == 0 {
        return fail(weight);
    }

    // Backtrace (:1391-1430) — descending `find`, preferring the last
    // direction taken.
    let mut out: Vec<MovePt> = Vec::with_capacity(64);
    out.push(tpfind);
    let mut lastangle = 0i32;
    let mut cur = tpfind;
    while find[idx(cur.x, cur.y)] != 0 {
        let mut best: Option<(MovePt, i32, i32)> = None;
        for u in 0..4 {
            let a = (u + lastangle) & 3;
            let tp = match a {
                0 => {
                    if cur.x - 1 < re.left {
                        continue;
                    }
                    MovePt::new(cur.x - 1, cur.y)
                }
                1 => {
                    if cur.y - 1 < re.top {
                        continue;
                    }
                    MovePt::new(cur.x, cur.y - 1)
                }
                2 => {
                    if cur.x + 1 >= re.right {
                        continue;
                    }
                    MovePt::new(cur.x + 1, cur.y)
                }
                _ => {
                    if cur.y + 1 >= re.bottom {
                        continue;
                    }
                    MovePt::new(cur.x, cur.y + 1)
                }
            };
            let f2 = find[idx(tp.x, tp.y)];
            if f2 < 0 {
                continue;
            }
            if let Some((_, _, bf)) = best {
                if bf <= f2 {
                    continue;
                }
            }
            best = Some((tp, a, f2));
        }
        let Some((tp, a, _)) = best else {
            // C++ ERROR_E — should be unreachable on a consistent wave.
            return fail(weight);
        };
        cur = tp;
        lastangle = a;
        if out.len() >= MATRIX_PATH_MOVE_MAX {
            // C++ ERROR_E on path overflow.
            return fail(weight);
        }
        out.push(cur);
    }
    out.reverse();

    LocalPathResult { path: out, weight }
}

/// Port of `CMatrixMapLogic::CanOptimize` (MatrixLogic.cpp:1865-
/// 1918): Bresenham walk from `(x1,y1)` to `(x2,y2)` over move
/// cells. Each visited cell (the start cell is pre-incremented past,
/// like the C++) must pass the size-aware stop mask AND its whole
/// `size × size` footprint window must carry blocker weight < 40 —
/// i.e. shortcuts may cross "standing" stamps (30) but not
/// "destination" stamps (200).
fn can_optimize(
    map: &GameMap,
    nsh: usize,
    size: i32,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    weight: &[i32],
) -> bool {
    let stride = map.size_move_x;
    let nsh_mask = MoveCell::stop_mask(nsh, size);
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x2 >= x1 { 1i32 } else { -1i32 };
    let sy = if y2 >= y1 { 1i32 } else { -1i32 };

    let mut x = x1;
    let mut y = y1;
    let window_blocked = |x: i32, y: i32| -> bool {
        for wy in 0..size {
            for wx in 0..size {
                let i = (y + wy) as usize * stride + (x + wx) as usize;
                if weight.get(i).copied().unwrap_or(0) >= 40 {
                    return true;
                }
            }
        }
        false
    };
    let stop = |x: i32, y: i32| -> bool {
        map.move_cell(x, y)
            .map(|c| (c.stop & nsh_mask) != 0)
            .unwrap_or(true)
    };

    if dy <= dx {
        let mut d = (dy << 1) - dx;
        let d1 = dy << 1;
        let d2 = (dy - dx) << 1;
        x += sx;
        for _ in 1..=dx {
            if d > 0 {
                d += d2;
                y += sy;
            } else {
                d += d1;
            }
            if stop(x, y) || window_blocked(x, y) {
                return false;
            }
            x += sx;
        }
    } else {
        let mut d = (dx << 1) - dy;
        let d1 = dx << 1;
        let d2 = (dx - dy) << 1;
        y += sy;
        for _ in 1..=dy {
            if d > 0 {
                d += d2;
                x += sx;
            } else {
                d += d1;
            }
            if stop(x, y) || window_blocked(x, y) {
                return false;
            }
            y += sy;
        }
    }
    true
}

/// Port of `OptimizeMovePath_Delete` (MatrixLogic.cpp:1856-1863):
/// removes the points strictly between `from` and `to`.
fn optimize_move_path_delete(path: &mut Vec<MovePt>, from: i32, to: i32) {
    if from >= to || from < 0 || to as usize >= path.len() {
        return;
    }
    path.drain((from + 1) as usize..to as usize);
}

/// Port of `CMatrixMapLogic::OptimizeMovePath` (MatrixLogic.cpp:1920-
/// 2010) + `OptimizeMovePathSimple` (:2012-2034). Finds every corner
/// (direction change) of the raw wave path, grows straight intervals
/// outward from each corner while `CanOptimize` holds, deletes the
/// in-between points, then runs the simple forward shortcutter.
///
/// `weight` is the stamp grid returned by [`find_local_path`] — the
/// C++ reads the same `m_Weight` cells it just stamped.
pub fn optimize_move_path(
    map: &GameMap,
    nsh: usize,
    size: i32,
    path: &mut Vec<MovePt>,
    weight: &[i32],
) {
    if path.len() <= 2 {
        return;
    }
    let cnt = path.len();

    struct Os {
        ibegin: i32,
        iend: i32,
        loopagain: bool,
    }
    let mut os: Vec<Os> = Vec::with_capacity(cnt + 2);

    // Corner detection (:1932-1958) — interval [0,0], every direction
    // change, and [cnt-1, cnt-1].
    let mut prevangle = (path[1].x - path[0].x, path[1].y - path[0].y);
    os.push(Os {
        ibegin: 0,
        iend: 0,
        loopagain: true,
    });
    for i in 2..cnt {
        let curangle = (path[i].x - path[i - 1].x, path[i].y - path[i - 1].y);
        if curangle != prevangle {
            prevangle = curangle;
            os.push(Os {
                ibegin: i as i32 - 1,
                iend: i as i32 - 1,
                loopagain: true,
            });
        }
    }
    os.push(Os {
        ibegin: cnt as i32 - 1,
        iend: cnt as i32 - 1,
        loopagain: true,
    });
    let oscnt = os.len();

    // Grow corners in both directions (:1966-1989).
    let mut loopagain = true;
    while loopagain {
        loopagain = false;
        for i in 0..oscnt {
            if !os[i].loopagain {
                continue;
            }
            os[i].loopagain = false;

            if (i == 0 && os[i].ibegin > 0) || (i != 0 && os[i].ibegin > os[i - 1].iend) {
                let a = path[(os[i].ibegin - 1) as usize];
                let b = path[os[i].iend as usize];
                if can_optimize(map, nsh, size, a.x, a.y, b.x, b.y, weight) {
                    loopagain = true;
                    os[i].loopagain = true;
                    os[i].ibegin -= 1;
                }
            }
            if (i == oscnt - 1 && (os[i].iend as usize) < cnt - 1)
                || (i != oscnt - 1 && os[i].iend < os[i + 1].ibegin)
            {
                let a = path[os[i].ibegin as usize];
                let b = path[(os[i].iend + 1) as usize];
                if can_optimize(map, nsh, size, a.x, a.y, b.x, b.y, weight) {
                    loopagain = true;
                    os[i].loopagain = true;
                    os[i].iend += 1;
                }
            }
        }
    }

    // Delete the covered spans, back to front (:1997-2003).
    for i in (1..oscnt).rev() {
        optimize_move_path_delete(path, os[i].ibegin, os[i].iend);
        optimize_move_path_delete(path, os[i - 1].iend, os[i].ibegin);
    }
    optimize_move_path_delete(path, os[0].ibegin, os[0].iend);

    optimize_move_path_simple(map, nsh, size, path, weight);
}

/// Port of `CMatrixMapLogic::OptimizeMovePathSimple` (MatrixLogic.cpp:
/// 2012-2034) — forward scan collapsing runs of waypoints reachable
/// in a straight line, with the two distance escape hatches that let
/// short blocked hops be skipped anyway.
fn optimize_move_path_simple(
    map: &GameMap,
    nsh: usize,
    size: i32,
    path: &mut Vec<MovePt>,
    weight: &[i32],
) {
    let mut from = 0usize;
    while path.len() >= 2 && from <= path.len() - 2 {
        let mut to = from + 2;
        while to < path.len() {
            let a = path[from];
            let b = path[to];
            if !can_optimize(map, nsh, size, a.x, a.y, b.x, b.y, weight) {
                let p = path[to - 1];
                let d_prev = (p.x - b.x).pow(2) + (p.y - b.y).pow(2);
                let d_from = (a.x - b.x).pow(2) + (a.y - b.y).pow(2);
                if d_prev >= 16 {
                    break;
                } else if d_from >= 64 {
                    break;
                }
            }
            to += 1;
        }
        to -= 1;
        if to - from > 1 {
            path.drain(from + 1..to);
            from += 1;
        } else {
            from = to;
        }
    }
}

/// Port of `CMatrixMapLogic::ZoneFindNear` (MatrixLogic.cpp:440-463):
/// the zone of the cell at `(mx, my)` when valid and passable for
/// chassis `nsh` (or `nsh < 0`), otherwise the zone with the nearest
/// center. Returns -1 when neither resolves (the C++ ERROR_Es).
pub fn zone_find_near(map: &GameMap, nsh: i32, mx: i32, my: i32) -> i32 {
    let Some(c) = map.move_cell(mx, my) else {
        return -1;
    };
    if c.zone >= 0 && (nsh < 0 || (c.stop & (1u32 << nsh)) == 0) {
        return c.zone;
    }
    let Some(rn) = map.road_network.as_ref() else {
        return -1;
    };
    let rn = rn.lock().unwrap();
    let mut mind = i64::MAX;
    let mut minz = -1i32;
    for (i, z) in rn.zones.iter().enumerate() {
        let dx = (mx - z.center.x) as i64;
        let dy = (my - z.center.y) as i64;
        let d = dx * dx + dy * dy;
        if d < mind {
            mind = d;
            minz = i as i32;
        }
    }
    minz
}

// ── Typed arena accessors ───────────────────────────────────────────────
// The `AsRobot()` / `AsCannon()` / `AsBuilding()` downcasts of
// CMatrixMapStatic. Type is checked via `core().obj_type` before the
// pointer cast, so these are safe at the same level as the C++ pattern.

pub fn robot_ref(objs: &Objects, id: ObjectId) -> Option<&crate::matrix_game::robot::Robot> {
    let obj = objs.get(id)?;
    if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
        return None;
    }
    Some(unsafe { &*(obj as *const dyn MapStatic as *const crate::matrix_game::robot::Robot) })
}

pub fn robot_mut(
    objs: &mut Objects,
    id: ObjectId,
) -> Option<&mut crate::matrix_game::robot::Robot> {
    let obj = objs.get_mut(id)?;
    if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
        return None;
    }
    Some(unsafe { &mut *(obj as *mut dyn MapStatic as *mut crate::matrix_game::robot::Robot) })
}

pub fn cannon_ref(
    objs: &Objects,
    id: ObjectId,
) -> Option<&crate::matrix_game::object_cannon::Cannon> {
    let obj = objs.get(id)?;
    if !matches!(obj.core().obj_type, ObjectType::Cannon) {
        return None;
    }
    Some(unsafe {
        &*(obj as *const dyn MapStatic as *const crate::matrix_game::object_cannon::Cannon)
    })
}

pub fn cannon_mut(
    objs: &mut Objects,
    id: ObjectId,
) -> Option<&mut crate::matrix_game::object_cannon::Cannon> {
    let obj = objs.get_mut(id)?;
    if !matches!(obj.core().obj_type, ObjectType::Cannon) {
        return None;
    }
    Some(unsafe {
        &mut *(obj as *mut dyn MapStatic as *mut crate::matrix_game::object_cannon::Cannon)
    })
}

pub fn flyer_ref(objs: &Objects, id: ObjectId) -> Option<&crate::matrix_game::flyer::Flyer> {
    let obj = objs.get(id)?;
    if !matches!(obj.core().obj_type, ObjectType::Flyer) {
        return None;
    }
    Some(unsafe { &*(obj as *const dyn MapStatic as *const crate::matrix_game::flyer::Flyer) })
}

pub fn flyer_mut(
    objs: &mut Objects,
    id: ObjectId,
) -> Option<&mut crate::matrix_game::flyer::Flyer> {
    let obj = objs.get_mut(id)?;
    if !matches!(obj.core().obj_type, ObjectType::Flyer) {
        return None;
    }
    Some(unsafe { &mut *(obj as *mut dyn MapStatic as *mut crate::matrix_game::flyer::Flyer) })
}

/// True when `(x, y)` lies on a BASE building's 3×3 terrain-cell
/// footprint — the cells a base marks as `m_Base`
/// (MatrixObjectBuilding.cpp:1065-1077). Used to suppress explosion
/// craters on base pads (`!g->IsBaseOn()`, MatrixEffect.cpp:491-499).
pub fn is_on_base(objs: &Objects, x: f32, y: f32) -> bool {
    use crate::matrix_game::common::float2int;
    use crate::matrix_game::object_building::{Building, BuildingType};
    let gs = crate::matrix_game::map::GLOBAL_SCALE;
    let cx = float2int(x / gs);
    let cy = float2int(y / gs);
    for id in objs.iter_live() {
        let Some(obj) = objs.get(id) else { continue };
        if !matches!(obj.core().obj_type, ObjectType::Building) {
            continue;
        }
        let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
        if b.kind != BuildingType::Base {
            continue;
        }
        let px = float2int((b.pos.x - gs / 2.0) / gs) - 1;
        let py = float2int((b.pos.y - gs / 2.0) / gs) - 1;
        if cx >= px && cx <= px + 2 && cy >= py && cy <= py + 2 {
            return true;
        }
    }
    false
}

pub fn building_ref(
    objs: &Objects,
    id: ObjectId,
) -> Option<&crate::matrix_game::object_building::Building> {
    let obj = objs.get(id)?;
    if !matches!(obj.core().obj_type, ObjectType::Building) {
        return None;
    }
    Some(unsafe {
        &*(obj as *const dyn MapStatic as *const crate::matrix_game::object_building::Building)
    })
}

pub fn building_mut(
    objs: &mut Objects,
    id: ObjectId,
) -> Option<&mut crate::matrix_game::object_building::Building> {
    let obj = objs.get_mut(id)?;
    if !matches!(obj.core().obj_type, ObjectType::Building) {
        return None;
    }
    Some(unsafe {
        &mut *(obj as *mut dyn MapStatic as *mut crate::matrix_game::object_building::Building)
    })
}

/// Port of `IsLiveUnit` (MatrixSide.cpp:9358-9363): live robot, or a
/// cannon that is neither DIP nor under construction. Anything else
/// (buildings included) is not a "unit".
pub fn is_live_unit(objs: &Objects, id: ObjectId) -> bool {
    use crate::matrix_game::object_cannon::CannonState;
    let Some(obj) = objs.get(id) else {
        return false;
    };
    match obj.core().obj_type {
        ObjectType::RobotAi => obj.is_live(),
        ObjectType::Cannon => cannon_ref(objs, id)
            .map(|c| !matches!(c.state, CannonState::Dip | CannonState::UnderConstruction))
            .unwrap_or(false),
        _ => false,
    }
}

/// Port of `GetMapPos` (MatrixSide.cpp:9365-9376) — the object's
/// upper-left move-cell coordinate.
pub fn get_map_pos(objs: &Objects, id: ObjectId) -> Option<(i32, i32)> {
    let obj = objs.get(id)?;
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    match obj.core().obj_type {
        ObjectType::RobotAi => robot_ref(objs, id).map(|r| (r.map_x, r.map_y)),
        ObjectType::Cannon => {
            cannon_ref(objs, id).map(|c| ((c.pos.x / gs) as i32, (c.pos.y / gs) as i32))
        }
        ObjectType::Building => {
            building_ref(objs, id).map(|b| ((b.pos.x / gs) as i32, (b.pos.y / gs) as i32))
        }
        _ => {
            let gc = obj.core().geo_center;
            Some(((gc.x / gs) as i32, (gc.y / gs) as i32))
        }
    }
}

/// Port of `GetWorldPos` (MatrixSide.cpp:9378-9386).
pub fn get_world_pos(objs: &Objects, id: ObjectId) -> Option<glam::Vec2> {
    let obj = objs.get(id)?;
    match obj.core().obj_type {
        ObjectType::RobotAi => robot_ref(objs, id).map(|r| glam::Vec2::new(r.pos_x, r.pos_y)),
        ObjectType::Cannon => cannon_ref(objs, id).map(|c| glam::Vec2::new(c.pos.x, c.pos.y)),
        ObjectType::Building => building_ref(objs, id).map(|b| glam::Vec2::new(b.pos.x, b.pos.y)),
        _ => {
            let gc = obj.core().geo_center;
            Some(glam::Vec2::new(gc.x, gc.y))
        }
    }
}

/// Port of `PointOfAim` (MatrixSide.cpp:9494-9517).
pub fn point_of_aim(map: &GameMap, objs: &Objects, id: ObjectId) -> Option<glam::Vec3> {
    let obj = objs.get(id)?;
    let mut p = obj.core().geo_center;
    match obj.core().obj_type {
        ObjectType::RobotAi | ObjectType::Cannon => p.z += 5.0,
        ObjectType::Building => p.z = map.get_z(p.x, p.y) + 20.0,
        _ => {}
    }
    Some(p)
}

/// Port of `CMatrixMapLogic::GetRegion` (MatrixLogic.hpp:186-211):
/// move-cell zone → zone's region, falling back to the nearest
/// region by center distance.
pub fn get_region(map: &GameMap, x: i32, y: i32) -> i32 {
    let Some(rn_lock) = map.road_network.as_ref() else {
        return -1;
    };
    let rn = rn_lock.lock().unwrap();
    if let Some(c) = map.move_cell(x, y) {
        if c.zone >= 0 && (c.zone as usize) < rn.zones.len() {
            let reg = rn.zones[c.zone as usize].region;
            if reg >= 0 {
                return reg;
            }
        }
    }
    rn.find_nerest_region(&crate::matrix_game::road_network::Point { x, y })
}

/// Port of `CMatrixMapLogic::IsLogicVisible` (MatrixLogic.cpp:
/// 3188-3227) — double trace with a building re-test, optionally
/// repeated from 50 units higher.
pub fn is_logic_visible(
    map: &GameMap,
    objs: &Objects,
    ofrom: ObjectId,
    oto: ObjectId,
    second_z: f32,
) -> bool {
    use crate::matrix_game::common::{
        TRACE_BUILDING, TRACE_CANNON, TRACE_FLYER, TRACE_NONOBJECT, TRACE_OBJECT,
        TRACE_OBJECTSPHERE, TRACE_ROBOT, TRACE_SKIP_INVISIBLE,
    };
    use crate::matrix_game::map_trace::{trace, TraceStop};

    let Some(from_obj) = objs.get(ofrom) else {
        return false;
    };
    let Some(to_obj) = objs.get(oto) else {
        return false;
    };
    let mut vstart = from_obj.core().geo_center;
    let vend = if matches!(to_obj.core().obj_type, ObjectType::Cannon) {
        let c = cannon_ref(objs, oto).unwrap();
        glam::Vec3::new(c.pos.x, c.pos.y, map.get_z(c.pos.x, c.pos.y) + 20.0)
    } else {
        to_obj.core().geo_center
    };

    for pass in 0..2 {
        if pass == 1 {
            if second_z == 0.0 {
                return false;
            }
            vstart.z += 50.0;
        }
        let (mut res, _) = trace(
            map,
            objs,
            vstart,
            vend,
            TRACE_ANYOBJECT | TRACE_NONOBJECT | TRACE_OBJECTSPHERE | TRACE_SKIP_INVISIBLE,
            Some(ofrom),
        );
        if let TraceStop::Object(hit) = res {
            let hit_is_building = objs
                .get(hit)
                .map(|o| matches!(o.core().obj_type, ObjectType::Building))
                .unwrap_or(false);
            if hit_is_building {
                let (res2, _) = trace(
                    map,
                    objs,
                    vstart,
                    vend,
                    TRACE_BUILDING | TRACE_SKIP_INVISIBLE,
                    Some(ofrom),
                );
                let blocked = matches!(res2, TraceStop::Object(h2) if objs
                    .get(h2)
                    .map(|o| matches!(o.core().obj_type, ObjectType::Building))
                    .unwrap_or(false));
                if blocked {
                    continue; // C++ `break` out of the while(true) → next pass
                }
                let (res3, _) = trace(
                    map,
                    objs,
                    vstart,
                    vend,
                    (TRACE_OBJECT | TRACE_ROBOT | TRACE_CANNON | TRACE_FLYER)
                        | TRACE_NONOBJECT
                        | TRACE_OBJECTSPHERE
                        | TRACE_SKIP_INVISIBLE,
                    Some(ofrom),
                );
                res = res3;
            }
        }
        if res.object() == Some(oto) {
            return true;
        }
    }
    false
}

/// Port of `CMatrixMapLogic::FindNearPlace` (MatrixLogic.cpp:2077-2125)
/// — BFS over zones from the cell at `mappos` until a zone with
/// places is found; returns the place nearest to `mappos`, or -1.
pub fn find_near_place(map: &GameMap, mm: u8, mappos: (i32, i32)) -> i32 {
    let Some(rn) = map.road_network.as_ref() else {
        return -1;
    };
    let rn = rn.lock().unwrap();
    find_near_place_impl(&rn, map, mm, mappos)
}

pub(crate) fn find_near_place_impl(
    rn: &crate::matrix_game::road_network::RoadNetwork,
    map: &GameMap,
    mm: u8,
    mappos: (i32, i32),
) -> i32 {
    let Some(c) = map.move_cell(mappos.0, mappos.1) else {
        return -1;
    };
    let zone = c.zone;
    if zone < 0 || zone as usize >= rn.zones.len() {
        return -1;
    }
    let d2 = |p: &crate::matrix_game::road_network::Point| -> i64 {
        let dx = (mappos.0 - p.x) as i64;
        let dy = (mappos.1 - p.y) as i64;
        dx * dx + dy * dy
    };
    let mut seen = vec![false; rn.zones.len()];
    let mut index: Vec<i32> = vec![zone];
    seen[zone as usize] = true;
    let mut sme = 0usize;
    while sme < index.len() {
        let z = &rn.zones[index[sme] as usize];
        if !z.place.is_empty() {
            let mut pl = z.place[0];
            let mut md = d2(&rn.places[pl as usize].pos);
            for &p in &z.place[1..] {
                let pd = d2(&rn.places[p as usize].pos);
                if pd < md {
                    md = pd;
                    pl = p;
                }
            }
            return pl;
        }
        for (i, &nz) in z.near_zone.iter().enumerate() {
            if seen[nz as usize] {
                continue;
            }
            if z.near_zone_move[i] & mm != 0 {
                continue;
            }
            index.push(nz);
            seen[nz as usize] = true;
        }
        sme += 1;
    }
    -1
}

/// Port of `CMatrixMapLogic::PlaceList` (MatrixLogic.cpp:2144-2362).
/// Fills `list` with reachable places around `to` (radius in move
/// cells) for chassis mask `mm`, starting from `from`. Returns
/// `(status, outdist)`: status 0 = unreachable, 1 = reachable via
/// the simple zone wave, 2 = via the cluster fallback. `outdist`
/// mirrors the C++ `dist*16` wave estimate.
pub fn place_list(
    map: &GameMap,
    mm: u8,
    from: (i32, i32),
    to: (i32, i32),
    radius: i32,
    farpath: bool,
    list: &mut Vec<i32>,
) -> (i32, i32) {
    list.clear();
    let Some(rn_lock) = map.road_network.as_ref() else {
        return (0, 0);
    };
    let rn = rn_lock.lock().unwrap();
    let pcnt = rn.places.len();
    if pcnt == 0 {
        return (0, 0);
    }

    let dist2 = |a: (i32, i32), b: &crate::matrix_game::road_network::Point| -> i64 {
        let dx = (a.0 - b.x) as i64;
        let dy = (a.1 - b.y) as i64;
        dx * dx + dy * dy
    };

    let fromplace = find_near_place_impl(&rn, map, mm, from);
    if fromplace < 0 {
        return (0, 0);
    }
    let toplace = find_near_place_impl(&rn, map, mm, to);

    let mut outdist;
    // `m_ZoneDataZero` equivalent, indexed by place.
    let mut data = vec![0u32; pcnt];
    let mut index: Vec<i32> = Vec::new();

    if toplace >= 0 {
        let tm = rn.places[toplace as usize].move_mask;
        if (tm & 0x1f) == 0x1f || (tm & mm) == 0 {
            let mut findend = -1i32;
            index.push(toplace);
            data[toplace as usize] = 1;
            list.push(toplace);
            if fromplace == toplace {
                findend = 0;
            }

            // Collect places within the radius (MatrixLogic.cpp:2172).
            let mut sme = 0usize;
            while sme < index.len() {
                let place = &rn.places[index[sme] as usize];
                for (i, &np) in place.near.iter().enumerate() {
                    if data[np as usize] != 0 {
                        continue;
                    }
                    if place.near_move[i] & mm != 0 {
                        continue;
                    }
                    if dist2(to, &rn.places[np as usize].pos) > (radius as i64) * (radius as i64) {
                        continue;
                    }
                    if fromplace == np {
                        findend = 0;
                    }
                    index.push(np);
                    data[np as usize] = 1;
                    list.push(np);
                }
                sme += 1;
            }

            // Can we reach the start point? (MatrixLogic.cpp:2194).
            let mut sme = 0usize;
            let mut dist = 0i32;
            let mut next = index.len();
            let mut reached = findend >= 0;
            if !reached {
                'wave: while sme < index.len() {
                    let place_idx = index[sme] as usize;
                    let near_cnt = rn.places[place_idx].near.len();
                    for i in 0..near_cnt {
                        let np = rn.places[place_idx].near[i];
                        if data[np as usize] != 0 {
                            continue;
                        }
                        if rn.places[place_idx].near_move[i] & mm != 0 {
                            continue;
                        }
                        if np == fromplace {
                            reached = true;
                            break 'wave;
                        }
                        if index.len() >= pcnt {
                            break;
                        }
                        index.push(np);
                        data[np as usize] = 1;
                    }
                    sme += 1;
                    if sme >= next {
                        next = index.len();
                        dist += 1;
                    }
                }
            }
            outdist = dist * 16;
            for &i in &index {
                data[i as usize] = 0;
            }
            if reached {
                let fd2 = {
                    let dx = (from.0 - to.0) as i64;
                    let dy = (from.1 - to.1) as i64;
                    dx * dx + dy * dy
                };
                if farpath || (dist as i64 * 4) * (dist as i64 * 4) <= fd2 {
                    return (1, outdist);
                }
            }
        }
    }

    // Fallback: rect-scan the place grid + clusterize
    // (MatrixLogic.cpp:2229-2361).
    list.clear();
    index.clear();

    let rc = crate::matrix_game::road_network::Rect {
        left: to.0 - radius - ROBOT_MOVECELLS_PER_SIZE,
        top: to.1 - radius - ROBOT_MOVECELLS_PER_SIZE,
        right: to.0 + radius + ROBOT_MOVECELLS_PER_SIZE,
        bottom: to.1 + radius + ROBOT_MOVECELLS_PER_SIZE,
    };
    let plr = rn.correct_rect_pl(&rc);
    for y in plr.top..plr.bottom {
        for x in plr.left..plr.right {
            let plist = rn.pl_list[(x + y * rn.pl_size_x) as usize];
            for u in 0..plist.cnt {
                let ip = plist.sme + u;
                let place = &rn.places[ip as usize];
                if place.move_mask & mm != 0 {
                    continue;
                }
                if dist2(to, &place.pos) > (radius as i64) * (radius as i64) {
                    continue;
                }
                index.push(ip);
                data[ip as usize] = (list.len() as u32 + 1) | 0x8000_0000;
                list.push(ip);
            }
        }
    }

    // Clusterize (MatrixLogic.cpp:2260-2308).
    let mut findend = -1i64;
    let oldcnt = index.len();
    let mut clcnt = 0u32;
    for u in 0..list.len() {
        if data[list[u] as usize] & 0x7fff_0000 != 0 {
            continue;
        }
        clcnt += 1;

        let mut sme = index.len();
        index.push(list[u]);
        data[list[u] as usize] |= clcnt << 16;

        while sme < index.len() {
            let place_idx = index[sme] as usize;
            let near_cnt = rn.places[place_idx].near.len();
            for i in 0..near_cnt {
                let np = rn.places[place_idx].near[i] as usize;
                if data[np] & 0x7fff_0000 != 0 {
                    continue;
                }
                if rn.places[place_idx].near_move[i] & mm != 0 {
                    continue;
                }
                if index.len() >= pcnt * 2 {
                    break;
                }
                if data[np] & 0x8000_0000 != 0 {
                    index.push(np as i32);
                    data[np] |= clcnt << 16;
                    if fromplace == np as i32 {
                        findend = (data[np] & 0x7fff_0000) as i64;
                    }
                } else {
                    let mut x = data[place_idx];
                    if x & 0x8000_0000 != 0 {
                        x = 1;
                    } else {
                        x = (x & 0xffff) + 1;
                    }
                    if x < 4 {
                        index.push(np as i32);
                        data[np] = x | (clcnt << 16);
                    }
                }
            }
            sme += 1;
        }
    }
    for &i in &index[oldcnt..] {
        if data[i as usize] & 0x8000_0000 == 0 {
            data[i as usize] = 0;
        }
    }
    index.truncate(oldcnt);

    // Wave from the clusters back to the start place
    // (MatrixLogic.cpp:2310-2344).
    let mut sme = 0usize;
    let mut dist = 0i32;
    let mut next = index.len();
    let mut u: i64 = 0;
    if findend >= 0 {
        u = findend;
    } else {
        'wave2: while sme < index.len() {
            if index[sme] == fromplace {
                u = (data[index[sme] as usize] & 0x7fff_0000) as i64;
                break;
            }
            let place_idx = index[sme] as usize;
            let near_cnt = rn.places[place_idx].near.len();
            for i in 0..near_cnt {
                let np = rn.places[place_idx].near[i] as usize;
                if data[np] & 0x7fff_0000 != 0 {
                    continue;
                }
                if rn.places[place_idx].near_move[i] & mm != 0 {
                    continue;
                }
                if np as i32 == fromplace {
                    u = (data[place_idx] & 0x7fff_0000) as i64;
                    break 'wave2;
                }
                if index.len() >= pcnt * 2 {
                    break;
                }
                index.push(np as i32);
                data[np] = data[place_idx] & 0x7fff_0000;
            }
            if u != 0 {
                break;
            }
            sme += 1;
            if sme >= next {
                next = index.len();
                dist += 1;
            }
        }
    }
    if u == 0 {
        list.clear();
        return (0, 0);
    }
    outdist = dist * 16;

    // Final list: places of the winning cluster (MatrixLogic.cpp:2348).
    let winners: Vec<i32> = index[..oldcnt.min(index.len())]
        .iter()
        .copied()
        .filter(|&i| (data[i as usize] & 0x7fff_0000) as i64 == u)
        .collect();
    list.clear();
    list.extend(winners);
    (2, outdist)
}

/// Port of `CMatrixMapLogic::PlaceListGrow` (MatrixLogic.cpp:
/// 2364-2405) — BFS-extend `list` by up to `growcnt` reachable
/// places. Returns the number added.
pub fn place_list_grow(map: &GameMap, mm: u8, list: &mut Vec<i32>, growcnt: i32) -> i32 {
    let Some(rn_lock) = map.road_network.as_ref() else {
        return 0;
    };
    let rn = rn_lock.lock().unwrap();
    let pcnt = rn.places.len();
    if pcnt == 0 {
        return 0;
    }

    let mut addcnt = 0i32;
    let mut data = vec![false; pcnt];
    let mut index: Vec<i32> = Vec::with_capacity(list.len());
    for &p in list.iter() {
        index.push(p);
        data[p as usize] = true;
    }

    let mut sme = 0usize;
    'outer: while sme < index.len() {
        let place_idx = index[sme] as usize;
        let near_cnt = rn.places[place_idx].near.len();
        for i in 0..near_cnt {
            let np = rn.places[place_idx].near[i];
            if data[np as usize] {
                continue;
            }
            if rn.places[place_idx].near_move[i] & mm != 0 {
                continue;
            }
            index.push(np);
            data[np as usize] = true;
            list.push(np);
            addcnt += 1;
            if addcnt >= growcnt {
                break 'outer;
            }
        }
        sme += 1;
    }
    addcnt
}

/// Wrapper over `CMatrixMapLogic::FindPathInZone` (MatrixLogic.cpp:
/// 1446+, ported in [`crate::matrix_game::road_network`]) returning
/// the zone path as a Vec. Empty when no road network is loaded or no
/// path exists.
pub fn find_path_in_zone(
    map: &GameMap,
    nsh: usize,
    zstart: i32,
    zend: i32,
    route: Option<(&crate::matrix_game::road_network::RoadRoute, usize)>,
) -> Vec<i32> {
    let Some(rn) = map.road_network.as_ref() else {
        return Vec::new();
    };
    let mut rn = rn.lock().unwrap();
    let mut path = vec![0i32; rn.zones.len().max(1)];
    let cnt = rn.find_path_in_zone(nsh, zstart, zend, route, &mut path);
    path.truncate(cnt);
    path
}

/// Bounding rect of a road-network zone as a [`MoveRect`] — the
/// `m_RN.m_Zone[...].m_Rect` lookups `FindLocalPath` makes at
/// MatrixLogic.cpp:1247-1250. `None` when the network or zone is
/// missing (callers fall back to the whole map).
pub fn zone_move_rect(map: &GameMap, zone: i32) -> Option<MoveRect> {
    let rn = map.road_network.as_ref()?.lock().unwrap();
    let z = rn.zones.get(usize::try_from(zone).ok()?)?;
    Some(MoveRect {
        left: z.rect.left,
        top: z.rect.top,
        right: z.rect.right,
        bottom: z.rect.bottom,
    })
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
