//! Port of `MatrixLogic.{cpp,hpp}` — `CMatrixMapLogic`, the
//! logic-layer subclass of `CMatrixMap`. Owns the Objects arena
//! (indirectly), the shared RNG, the takt-decomposition driver,
//! per-side state, and the cell-level place / wall helpers used by
//! the robot AI.
//!
//! Also the module root for `Logic/*.cpp` ports — `ai_group.rs`
//! (Logic/MatrixAIGroup.cpp) and future siblings declare here.

pub mod ai_group;

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{PLAYER_SIDE, TRACE_ANYOBJECT};
use crate::matrix_game::config::{
    self, BuildingDamages, BuildingLabels, ChassisChars, GlobalConfig, HeadCharsTable,
    ItemCharsTable, ItemDescriptions, ItemLabels, ObjectDamages, PriceTable, RobotDamages,
    RobotNameParts, StringTables, Timings, TurretProps, WeaponCooldown, WeaponStrengthAI,
};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{MapStatic, ObjectId, ObjectType, Objects};
use crate::matrix_game::object::MapObject;
use crate::matrix_game::object_building::{Building, BuildingType};
use crate::matrix_game::config::Resource;
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
            rng: Rnd::new(seed),
            tick: 0,
            elapsed_ms: 0,
            player_side: Side::new(PLAYER_SIDE),
        }
    }

    /// Port of the logic-takt portion of `CMatrixMapLogic::Takt`
    /// (`MatrixLogic.cpp:2722-2734`). Full 10ms slices, then the
    /// remainder — matching the trailing `if (portions) ProceedLogic(portions);`.
    pub fn takt(&mut self, step_ms: i32) {
        if step_ms <= 0 {
            return;
        }
        let full = step_ms / LOGIC_TAKT_PERIOD_MS;
        for _ in 0..full {
            self.objects
                .proceed_logic(LOGIC_TAKT_PERIOD_MS, &mut self.rng);
            self.tick += 1;
        }
        let rem = step_ms - full * LOGIC_TAKT_PERIOD_MS;
        if rem > 0 {
            self.objects.proceed_logic(rem, &mut self.rng);
        }
        // Port of the per-building `m_ResourcePeriod` tick at
        // MatrixObjectBuilding.cpp:605-667. The C++ advances the
        // counter from inside each building's `LogicTakt`; we run it
        // once per outer slice so the per-building state ports 1:1
        // without threading the player Side through `proceed_logic`.
        self.accrue_resources(step_ms);
        self.refresh_side_robots();
        self.elapsed_ms += step_ms as i64;
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
        let weapon_cooldown = WeaponCooldown::from_matrix_data(matrix_data).unwrap_or_default();
        let weapon_strength_ai =
            WeaponStrengthAI::from_matrix_data(matrix_data).unwrap_or_default();
        let head_chars = HeadCharsTable::from_matrix_data(matrix_data).unwrap_or_default();
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
            weapon_cooldown,
            weapon_strength_ai,
            head_chars,
        });
        // String-heavy tables — labels, descriptions, robot-name parts.
        let labels = ItemLabels::from_matrix_data(matrix_data).unwrap_or_default();
        let descriptions = ItemDescriptions::from_matrix_data(matrix_data).unwrap_or_default();
        let robot_names = RobotNameParts::from_matrix_data(matrix_data).unwrap_or_default();
        let buildings = BuildingLabels::from_matrix_data(matrix_data).unwrap_or_default();
        log::info!(
            "config: loaded labels (chassis[0]={:?}, robot_names[hull1]={:?}, base_name={:?})",
            labels.chassis.first().map(|s| s.as_str()).unwrap_or(""),
            robot_names.hull.first().map(|s| s.as_str()).unwrap_or(""),
            buildings.base_name,
        );
        config::set_global_strings(StringTables {
            labels,
            descriptions,
            robot_names,
            buildings,
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
        (ids, stats)
    }

    /// Populate the arena with one [`Building`] per starting base /
    /// turret. Ports `CMatrixMap::LoadBuildings` + `Building::OnLoad`.
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
        let hit = self.objects.pick_object(origin, dir, TRACE_ANYOBJECT, None);
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
        // World XY → move-cell upper-left corner of the robot's 4×4
        // footprint, matching the conversion at MatrixRobot.cpp:4625 /
        // MatrixSide.cpp:816-817.
        let (cmx, cmy) = map.world_to_move(wx, wy);
        let center_mx = cmx - ROBOT_MOVECELLS_PER_SIZE / 2;
        let center_my = cmy - ROBOT_MOVECELLS_PER_SIZE / 2;

        // Seed the blocker list with every out-of-group robot's
        // claimed place. The C++ at MatrixSide.cpp:8700-8727 walks
        // `GetFirstLogic` and adds entries where `GetGroupLogic != no`
        // — we approximate "not in our group" as "not in the current
        // selection" since per-side logical groups aren't ported yet.
        let selected: Vec<ObjectId> = self.player_side.selected.clone();
        let mut blockers: Vec<(i32, i32)> = Vec::new();
        for id in self.objects.iter_live() {
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
                continue;
            }
            if selected.contains(&id) {
                continue;
            }
            let other: &crate::matrix_game::robot::Robot = unsafe {
                &*(obj as *const dyn MapStatic as *const crate::matrix_game::robot::Robot)
            };
            if let Some((px, py)) = other.place_add {
                blockers.push((px, py));
            }
        }

        // Spiral-search a unique cell per in-group robot; each newly
        // assigned slot joins the blocker list before the next robot
        // starts searching. Matches the second loop in
        // `PGAssignPlacePlayer` (MatrixSide.cpp:8729-8756) — that loop
        // also stamps the result into the robot's env via
        // `PGSetPlace` + feeds it back into `other_des`.
        let mut out: Vec<(f32, f32)> = Vec::new();
        for id in selected {
            let Some(obj) = self.objects.get_mut(id) else {
                continue;
            };
            if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
                continue;
            }
            let r: &mut crate::matrix_game::robot::Robot = unsafe {
                &mut *(obj as *mut dyn MapStatic as *mut crate::matrix_game::robot::Robot)
            };
            if r.side != self.player_side.id {
                continue;
            }
            let chassis = r.chassis as usize;
            let (mx, my) = place_find_near_with_blockers(
                map,
                chassis,
                ROBOT_MOVECELLS_PER_SIZE,
                center_mx,
                center_my,
                &blockers,
            )
            .unwrap_or((center_mx, center_my));
            // `PGSetPlace` → `m_PlaceAdd = (mx, my)` (MatrixSide.cpp:8473).
            r.place_add = Some((mx, my));
            r.move_to(mx, my);
            blockers.push((mx, my));
            // World-space ping position = footprint center.
            let gs = crate::matrix_game::map::GameMap::GLOBAL_SCALE_MOVE;
            let half = ROBOT_MOVECELLS_PER_SIZE as f32 * 0.5;
            out.push(((mx as f32 + half) * gs, (my as f32 + half) * gs));
        }
        out
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
            let b: &Building =
                unsafe { &*(obj as *const dyn MapStatic as *const Building) };
            if matches!(b.state, crate::matrix_game::object_building::BaseState::Dip
                | crate::matrix_game::object_building::BaseState::DipExploded)
            {
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
            let b: &Building =
                unsafe { &*(obj as *const dyn MapStatic as *const Building) };
            if matches!(b.state, crate::matrix_game::object_building::BaseState::Dip
                | crate::matrix_game::object_building::BaseState::DipExploded)
            {
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

        let player_id = self.player_side.id;
        let fu = self.player_side.get_resource_force_up();
        for id in ids {
            let Some(obj) = self.objects.get_mut(id) else {
                continue;
            };
            // SAFETY: filtered to ObjectType::Building above.
            let b: &mut Building =
                unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
            if matches!(b.state, crate::matrix_game::object_building::BaseState::Dip
                | crate::matrix_game::object_building::BaseState::DipExploded)
            {
                continue;
            }
            if b.side == 0 {
                continue; // neutral / unclaimed — C++ guards on `m_Side`.
            }
            b.resource_period += cms;

            // Per-kind threshold lookup.
            let (threshold, payout): (i32, Payout) = match b.kind {
                BuildingType::Titan => {
                    (timings.resource_titan, Payout::Single(Resource::Titan, RESOURCES_INCOME))
                }
                BuildingType::Electronic => (
                    timings.resource_electronics,
                    Payout::Single(Resource::Electronics, RESOURCES_INCOME),
                ),
                BuildingType::Energy => (
                    timings.resource_energy,
                    Payout::Single(Resource::Energy, RESOURCES_INCOME),
                ),
                BuildingType::Plasma => {
                    (timings.resource_plasma, Payout::Single(Resource::Plasma, RESOURCES_INCOME))
                }
                BuildingType::Base => {
                    let ra = RESOURCES_INCOME_BASE * fu / 100;
                    (timings.resource_base, Payout::All(ra))
                }
                BuildingType::Repair => continue,
            };
            if threshold <= 0 || b.resource_period < threshold {
                continue;
            }
            b.resource_period = 0;
            if b.side != player_id {
                // Non-player sides have no Side struct yet; timer still
                // resets so future per-side accrual starts in phase.
                continue;
            }
            match payout {
                Payout::Single(res, amt) => self.player_side.add_resource_amount(res, amt),
                Payout::All(amt) => {
                    for r in Resource::ALL {
                        self.player_side.add_resource_amount(r, amt);
                    }
                }
            }
        }
    }

    /// Refresh `m_RobotsCnt` on the player side from the live arena.
    /// Called from [`takt`] each logic slice so the HUD robot counter
    /// is always in sync.
    pub fn refresh_side_robots(&mut self) {
        self.player_side.robots_cnt = self.compute_side_robots(self.player_side.id);
    }
}

/// Shape of one per-kind income emission. Internal to `accrue_resources`.
enum Payout {
    /// Factory — single resource type.
    Single(Resource, i32),
    /// Base — same amount to all four resources.
    All(i32),
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

/// Port of `CMatrixMapLogic::PlaceFindNear(nsh, size, mx, my)` —
/// the commented-out simpler variant from MatrixLogic.cpp:540-570
/// (the full variant at :713-758 adds other-robot-destination
/// avoidance; we defer that). Spirals outward looking for a cell
/// that passes `place_is_empty` for `(chassis_kind, size)`.
/// Returns the nearest valid cell as `(mx, my)`.
#[allow(clippy::too_many_arguments)]
pub fn place_find_near(
    map: &GameMap,
    objs: &Objects,
    chassis_kind: usize,
    size: i32,
    mx: i32,
    my: i32,
    radius: i32,
    skip: Option<ObjectId>,
) -> Option<(i32, i32)> {
    if place_is_empty(map, objs, chassis_kind, size, mx, my, skip) {
        return Some((mx, my));
    }
    for r in 1..=radius {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx.abs() != r && dy.abs() != r {
                    continue;
                } // ring only
                let nx = mx + dx;
                let ny = my + dy;
                if place_is_empty(map, objs, chassis_kind, size, nx, ny, skip) {
                    return Some((nx, ny));
                }
            }
        }
    }
    None
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

    // ── Economy / robot-limit tests ───────────────────────────────
    //
    // These seed a fresh `MapLogic` with hand-rolled `Building`
    // instances so the ports of GetMaxSideRobots / GetResourceIncome /
    // the per-building resource tick can be exercised without the full
    // map-load path.
    use crate::matrix_game::map::BuildingInstance;
    use crate::matrix_game::object_building::{Building, BuildingType};

    fn spawn_building(
        w: &mut MapLogic,
        side: i32,
        kind: BuildingType,
        x: f32,
        y: f32,
    ) -> ObjectId {
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
        assert_eq!(w.player_side.resources, [0; crate::matrix_game::config::MAX_RESOURCES]);

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
                w.player_side.resources[r as usize],
                RESOURCES_INCOME_BASE,
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

use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
