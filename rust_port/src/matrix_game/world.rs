//! Top-level game state: the [`Objects`] arena + the logic/graphic tick
//! driver.
//!
//! Ports the `CMatrixMapLogic::Takt` decomposition in
//! `MatrixLogic.cpp:2720-2766`: step is broken into full
//! `LOGIC_TAKT_PERIOD` (10ms) portions + a remainder, each dispatched
//! through `CMatrixMapStatic::ProceedLogic`. Side logic and the graphic
//! takt (`CMatrixMap::Takt`) aren't part of scope A/B — they land when
//! sides + the full map takt arrive.

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{PLAYER_SIDE, TRACE_ANYOBJECT};
use crate::matrix_game::config::{BuildingDamages, ObjectDamages};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{MapStatic, ObjectId, ObjectType, Objects};
use crate::matrix_game::object::MapObject;
use crate::matrix_game::object_building::Building;
use crate::matrix_game::rnd::Rnd;
use crate::matrix_game::side::{CurrSel, Side};
use crate::matrix_lib::base::storage::Storage;

/// `LOGIC_TAKT_PERIOD` from `MatrixLogic.hpp:13`.
pub const LOGIC_TAKT_PERIOD_MS: i32 = 10;

pub struct World {
    pub objects: Objects,
    /// The shared RNG — matches `CMatrixMapLogic::m_Rnd`
    /// (MatrixLogic.hpp:75). Threaded into every `takt`/`logic_takt` so
    /// subclasses (and us here) can branch deterministically.
    pub rng: Rnd,
    /// Number of LOGIC_TAKT_PERIOD portions elapsed since construction.
    /// `u64` so a 10ms tick doesn't wrap in any reasonable session.
    pub tick: u64,
    /// Total game-time in ms since the first `takt` call. Mirrors the
    /// C++ `GetTime()` return type (milliseconds on an int clock).
    pub elapsed_ms: i64,
    /// Port of `g_MatrixMap->GetPlayerSide()` (MatrixMap.hpp). The
    /// player's slice of the per-side state — selection, active
    /// object, etc. Full per-side AI / resource / stats table lands
    /// with the full `CMatrixSide` port.
    pub player_side: Side,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        Self::with_seed(1)
    }

    /// Deterministic-seed constructor — tests and replay paths use this
    /// so the generator state is reproducible. Matches the C++ ctor up
    /// to the `rand()` seeding step (which is nondeterministic).
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
    /// (`MatrixLogic.cpp:2722-2734`). Full 10ms slices run first, then
    /// the remainder in a single final call — matching the original's
    /// trailing `if (portions) ProceedLogic(portions);`.
    pub fn takt(&mut self, step_ms: i32) {
        if step_ms <= 0 {
            return;
        }
        let full = step_ms / LOGIC_TAKT_PERIOD_MS;
        for _ in 0..full {
            self.objects.proceed_logic(LOGIC_TAKT_PERIOD_MS, &mut self.rng);
            self.tick += 1;
        }
        let rem = step_ms - full * LOGIC_TAKT_PERIOD_MS;
        if rem > 0 {
            self.objects.proceed_logic(rem, &mut self.rng);
        }
        self.elapsed_ms += step_ms as i64;
    }

    /// Port of `CMatrixMap::Takt`'s `SortEndGraphicTakt` call
    /// (MatrixMap.cpp:2501 → MatrixMapStatic.cpp:755-765). Per-frame
    /// per-object graphic takt; paired with (and strictly *after*) the
    /// logic takt. Only the `Takt` dispatch portion is ported here —
    /// the sky-angle / effects / sound / minimap / skin-manager takts
    /// from `CMatrixMap::Takt` live elsewhere in the Rust port (already
    /// driven from `form_game.rs`).
    pub fn graphic_takt(&mut self, step_ms: i32) {
        if step_ms <= 0 {
            return;
        }
        self.objects.graphic_takt(step_ms, &mut self.rng);
    }

    /// Load `g_Config.m_ObjectDamages` from `robots.dat`
    /// (MatrixConfig.cpp:591-607). Missing / malformed data falls back
    /// to the zero-initialized table (same semantics as the C++
    /// `memset` at :593). Safe to call multiple times; subsequent
    /// calls overwrite.
    pub fn load_config(&mut self, matrix_data: &Storage) {
        self.objects.object_damages =
            ObjectDamages::from_matrix_data(matrix_data).unwrap_or_default();
        self.objects.building_damages =
            BuildingDamages::from_matrix_data(matrix_data).unwrap_or_default();
    }

    /// Populate the arena with one [`MapObject`] per decorative object
    /// placed on the map. Ports the `new CMatrixMapObject()` + `Init(type)`
    /// pattern in `CMatrixMap::LoadObjects`.
    ///
    /// `map_stor` is the map's `STRG` storage so each instance can look
    /// up its Ids row and drive `apply_ids_row` — the behaviour-keyword
    /// switch that decides `BehFlag` and whether to `AddLT`.
    ///
    /// Returns `(spawned_ids, stats)` where `stats` summarises how many
    /// objects landed in each `BehFlag` bucket. Useful for logging at
    /// init; the original prints similar counts via `DM(...)` calls.
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
            // `g_MatrixMap->IdsGet(m_Type)` — the row; may be absent if
            // `m_Type` is `m_IdsCnt-1` (the mapname slot — mapobjects
            // don't land there) or if strings isn't present.
            let ids_row = if (inst.type_id as usize) < ids_count {
                strings.map(|s| s.get_as_wstr(inst.type_id as usize)).unwrap_or_default()
            } else {
                String::new()
            };

            let add_lt = if ids_row.is_empty() {
                // No row → fallthrough to BEHF_STATIC (the from_instance
                // default). `apply_ids_row` on "" would do the same but
                // skipping avoids the unnecessary parse pass.
                false
            } else {
                obj.apply_ids_row(&ids_row, &mut self.rng, || {
                    // `g_MatrixMap->m_BeforeWinCount` bump on '+' prefix
                    // (MatrixObject.cpp:1023). The counter itself isn't
                    // ported yet — it drives the "you must destroy all
                    // special objects to win" UI, a separate scope.
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

    /// Populate the arena with one [`Building`] per starting base / turret
    /// placed on the map. Ports the `new CMatrixBuilding()` + `OnLoad()`
    /// path that `CMatrixMap::LoadBuildings` runs during init.
    ///
    /// Buildings opt into the logic-temp list immediately so their
    /// state machine (currently a stub) ticks every LOGIC_TAKT_PERIOD.
    /// That matches C++ `CMatrixBuilding::OnLoad` which calls `AddLT()`
    /// at MatrixObjectBuilding.cpp:1088.
    ///
    /// Returns the spawned IDs so callers can look them up by side /
    /// kind without a subsequent arena scan.
    /// Port of the selection-entry code path triggered on left-click
    /// (MatrixFormGame.cpp:530-642 → CMatrixMap pick → CMatrixSide
    /// SelectObject). Given a screen pixel under the cursor, casts a
    /// world-space ray via the camera, picks the nearest building /
    /// unit, and stores it on `player_side`.
    ///
    /// Robots / flyers / cannons aren't in the arena yet, so `mask`
    /// defaults to `TRACE_ANYOBJECT`: buildings match today, other
    /// subclasses join the search when they land.
    ///
    /// Returns `Some(id)` if the click hit an object, `None` to
    /// mirror the C++ behaviour of "click on empty ground → clear
    /// selection".
    pub fn select_at_screen(
        &mut self,
        camera: &Camera,
        sx: f32,
        sy: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> Option<ObjectId> {
        let (origin, dir) = camera.screen_to_world_ray(sx, sy, screen_w, screen_h);
        let hit = self.objects.pick_object(origin, dir, TRACE_ANYOBJECT, None);
        match hit {
            Some((id, _t)) => {
                let sel = match self.objects.get(id).map(|o| o.core().obj_type) {
                    Some(ObjectType::Building) => {
                        // Base vs factory panel (MatrixSide.cpp's
                        // SelectObject branch — base kind = BaseSelected,
                        // other kinds = BuildingSelected).
                        let is_base = self.objects.get(id)
                            .and_then(|o| {
                                let p = o as *const dyn MapStatic
                                    as *const crate::matrix_game::object_building::Building;
                                unsafe { p.as_ref() }
                                    .map(|b| b.kind == crate::matrix_game::object_building::BuildingType::Base)
                            })
                            .unwrap_or(false);
                        if is_base { CurrSel::BaseSelected } else { CurrSel::BuildingSelected }
                    }
                    Some(ObjectType::RobotAi)   => CurrSel::RobotsSelected,
                    Some(ObjectType::Cannon)    => CurrSel::CannonSelected,
                    Some(ObjectType::Flyer)     => CurrSel::FlyerSelected,
                    _ => CurrSel::Nothing,
                };
                self.player_side.select(id, sel);
                Some(id)
            }
            None => {
                self.player_side.clear();
                None
            }
        }
    }

    /// Return the active-selection id if it's still a live object in
    /// the arena — tombstones (remove() bumped the generation) return
    /// `None`, matching the C++ `m_ActiveObject->m_Object == NULL`
    /// stale-pointer guard.
    pub fn active_object(&self) -> Option<ObjectId> {
        let id = self.player_side.active_object?;
        if self.objects.is_valid(id) { Some(id) } else { None }
    }

    pub fn spawn_buildings(&mut self, map: &GameMap) -> Vec<ObjectId> {
        let mut ids = Vec::with_capacity(map.buildings.len());
        for inst in &map.buildings {
            let mut b = Building::from_instance(inst);
            // Seed max HP from the loaded `Weapons/Damages/Building/HITPOINT`
            // table. Indexed by `EBuildingType`. Falls back to 0 when
            // robots.dat hasn't been loaded — tests can bypass with
            // explicit `init_max_hitpoint` calls.
            let kind_idx = b.kind as usize;
            let hp = self.objects.building_damages
                .hitpoint
                .get(kind_idx)
                .copied()
                .unwrap_or(0);
            if hp > 0 {
                b.init_max_hitpoint(hp as f32);
            }
            let id = self.objects.spawn(Box::new(b));
            // Back-fill the building's own id so BuildStack can hand
            // it to freshly-produced robots for their spawn animation.
            if let Some(obj) = self.objects.get_mut(id) {
                let b_mut: &mut Building = unsafe {
                    &mut *(obj as *mut dyn crate::matrix_game::map_static::MapStatic as *mut Building)
                };
                b_mut.self_id = Some(id);
            }
            self.objects.add_lt(id);
            ids.push(id);
        }
        ids
    }
}

/// Per-BehFlag spawn counts. `MapObject::apply_ids_row` drives the
/// dispatch, so aggregating here is the cheapest way to surface "what
/// actually landed in the arena" for a given map.
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
    /// Count of rows whose behaviour field started with '+' (special
    /// win-target objects, see `apply_ids_row`).
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
        self.r#static + self.burn + self.r#break + self.anim
            + self.sens + self.spawner + self.terron + self.portret
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
        fn core(&self) -> &ObjectCore { &self.core }
        fn core_mut(&mut self) -> &mut ObjectCore { &mut self.core }
        fn rchange(&self) -> u32 { self.rchange }
        fn rchange_set(&mut self, b: u32) { self.rchange |= b; }
        fn rchange_clear(&mut self, b: u32) { self.rchange &= !b; }
        fn object_state(&self) -> u32 { self.state }
        fn object_state_set(&mut self, b: u32) { self.state |= b; }
        fn object_state_clear(&mut self, b: u32) { self.state &= !b; }
        fn ablaze_ttl(&self) -> i32 { self.ablaze }
        fn set_ablaze_ttl(&mut self, t: i32) { self.ablaze = t; }
        fn shorted_ttl(&self) -> i32 { self.shorted }
        fn set_shorted_ttl(&mut self, t: i32) { self.shorted = t; }
        fn r_need(&mut self, _: u32) {}
        fn takt(&mut self, _: i32, _: &mut Rnd, _: &mut crate::matrix_game::map_static::Objects) {}
        fn logic_takt(&mut self, cms: i32, _: &mut Rnd, _: &mut crate::matrix_game::map_static::Objects) {
            self.calls.borrow_mut().push(cms);
        }
    }

    #[test]
    fn takt_decomposes_step_into_full_portions_plus_remainder() {
        let mut w = World::new();
        let calls = Rc::new(RefCell::new(Vec::new()));
        let id = w.objects.spawn(Box::new(Counter {
            core: ObjectCore { obj_type: ObjectType::MapObject, ..Default::default() },
            rchange: 0, state: 0, ablaze: 0, shorted: 0,
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
        let mut w = World::new();
        let calls = Rc::new(RefCell::new(Vec::new()));
        let id = w.objects.spawn(Box::new(Counter {
            core: ObjectCore { obj_type: ObjectType::MapObject, ..Default::default() },
            rchange: 0, state: 0, ablaze: 0, shorted: 0,
            calls: calls.clone(),
        }));
        w.objects.add_lt(id);

        w.takt(30);
        assert_eq!(calls.borrow().clone(), vec![10, 10, 10]);
        assert_eq!(w.tick, 3);
    }

    #[test]
    fn takt_with_sub_period_step_uses_only_remainder() {
        let mut w = World::new();
        let calls = Rc::new(RefCell::new(Vec::new()));
        let id = w.objects.spawn(Box::new(Counter {
            core: ObjectCore { obj_type: ObjectType::MapObject, ..Default::default() },
            rchange: 0, state: 0, ablaze: 0, shorted: 0,
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
        let mut w = World::new();
        w.takt(0);
        w.takt(-5);
        assert_eq!(w.tick, 0);
        assert_eq!(w.elapsed_ms, 0);
    }
}
