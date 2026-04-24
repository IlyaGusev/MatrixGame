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
use crate::matrix_game::object_building::Building;
use crate::matrix_game::rnd::Rnd;
use crate::matrix_game::side::{CurrSel, Side};
use crate::matrix_lib::base::storage::Storage;

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

    /// Port of the end-of-marquee fold at `CMultiSelection::End`
    /// (MatrixMultiSelection.cpp). Projects every live own-side robot
    /// to screen coords and keeps the ones whose projection lands
    /// inside the axis-aligned rect `[rect_min, rect_max]`. The
    /// result is committed as the new selection with the first hit
    /// as `active_object`.
    pub fn marquee_select(
        &mut self,
        camera: &Camera,
        rect_min: [f32; 2],
        rect_max: [f32; 2],
        screen_w: f32,
        screen_h: f32,
        shift: bool,
    ) -> usize {
        let vp = camera.view_proj();
        let map_cx = camera.map_cx();
        let map_cy = camera.map_cy();
        let mut hits: Vec<ObjectId> = if shift {
            self.player_side.selected.clone()
        } else {
            Vec::new()
        };

        for id in self.objects.iter_live() {
            let Some(obj) = self.objects.get(id) else {
                continue;
            };
            if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
                continue;
            }
            if self.object_side(id) != self.player_side.id {
                continue;
            }
            let c = obj.core().geo_center;
            // View-proj expects centered world (see camera::view_proj
            // + selection shader — map_center subtracted everywhere).
            let clip = vp * glam::Vec4::new(c.x - map_cx, c.y - map_cy, c.z, 1.0);
            if clip.w <= 0.0 {
                continue;
            } // behind camera
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            // NDC → screen (Y flip, same as screen_to_world_ray).
            let sx = (ndc_x * 0.5 + 0.5) * screen_w;
            let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * screen_h;
            if sx >= rect_min[0]
                && sx <= rect_max[0]
                && sy >= rect_min[1]
                && sy <= rect_max[1]
                && !hits.contains(&id)
            {
                hits.push(id);
            }
        }
        let n = hits.len();
        let primary = hits.last().copied();
        self.player_side
            .select_replace(hits, primary, CurrSel::RobotsSelected);
        n
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
    fn object_side(&self, id: ObjectId) -> i32 {
        self.objects.get(id).map(|o| o.side()).unwrap_or(0)
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
}
