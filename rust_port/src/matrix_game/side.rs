//! Partial port of `CMatrixSide` (MatrixSide.{cpp,hpp}).
//!
//! CMatrixSide in the original is ~thousands of lines covering
//! selection state, logical-group rosters, resource accounts, AI
//! planners, strategy orders, stats, and arcaded-robot handover.
//! This file ports only the selection fields the interface /
//! building-selection flow reads in the C++, leaving the logic/AI /
//! resources / stats for later.
//!
//! What's here:
//! * `CurrSel` enum — mirrors the C++ per-side "what's currently
//!   selected" enum (MatrixSide.hpp).
//! * `Side` struct — minimal bookkeeping: `id`, `active_object`,
//!   `curr_sel`.
//!
//! Resources / stats / kill counters / logical groups etc. land when
//! their call sites need them.

use crate::matrix_game::map_static::ObjectId;
use crate::matrix_game::robot_units::{Resource, MAX_RESOURCES};

/// Port of the hard-coded 9000 cap inside `CMatrixSideUnit::AddResourceAmount`
/// (MatrixSide.hpp:438-443). Every `AddResourceAmount` call clamps the
/// final value to this ceiling; the HUD also displays `9000` when the
/// pool saturates.
pub const RESOURCE_CAP: i32 = 9000;

/// Port of `CMatrixMap::GetSideColor` (MatrixMap.cpp:1014, MatrixMap.hpp:738).
/// Returns the diffuse RGB components (0..1) the C++ loads from
/// `Side/{id}=<name,r,g,b,Minimap,rMM,gMM,bMM>` entries of `robots.dat`.
///
/// The shipped values (confirmed by dumping `Side` from robots.dat):
///   0 — Neutral = (128,128,128)
///   1 — Player  = (227,158,31)   Yellow
///   2 — AI Red  = (142,0,0)
///   3 — AI Blue = (0,0,150)
///   4 — AI Green= (0,150,0)
///
/// Out-of-range ids fall back to the neutral grey (matches the C++
/// `if (id == 0) return m_NeutralSideColor` branch when the subscript
/// lookup below would overrun).
pub fn side_color_rgb(side: i32) -> [f32; 3] {
    let c = match side {
        1 => [227u8, 158, 31],
        2 => [142, 0, 0],
        3 => [0, 0, 150],
        4 => [0, 150, 0],
        _ => [128, 128, 128],
    };
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]
}

/// Port of `CMatrixMap::GetSideColorMM` (MatrixMap.cpp:1015,
/// MatrixMap.hpp:745) — saturated minimap variant of the side colour.
/// Same mapping as `side_color_rgb` except the factions use pure R/G/B
/// so they read cleanly on the minimap backdrop.
pub fn side_color_minimap_rgb(side: i32) -> [f32; 3] {
    let c = match side {
        1 => [255u8, 255, 0],
        2 => [255, 0, 0],
        3 => [0, 0, 255],
        4 => [0, 255, 0],
        _ => [128, 128, 128],
    };
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]
}

/// Port of `ESelectedType` (MatrixSide.hpp). Tracks what the player's
/// side is currently pointing at — the C++ interface panel dispatches
/// on this to decide which menu to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CurrSel {
    /// Nothing selected.
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
    /// A flyer is selected.
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
}

impl Side {
    pub fn new(id: i32) -> Self {
        Self {
            id,
            active_object: None,
            curr_sel: CurrSel::Nothing,
            selected: Vec::new(),
            // Each side starts with a plausible resource pool — the
            // C++ seeds this from map `StartResources` which isn't
            // parsed yet. Give every side enough to build a basic
            // robot so the constructor UI is exercisable until the
            // map-side seeding lands.
            resources: [500, 500, 500, 500],
            robots_cnt: 0,
            base_res_force: 100,
            builder: Some(crate::matrix_game::interface::constructor::RobotBuilder::new()),
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
    pub fn can_afford(&self, cost: &crate::matrix_game::robot_units::UnitPrice) -> bool {
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
    }

    /// Clear the selection — C++ `CMatrixSide::UnSelect` equivalent.
    pub fn clear(&mut self) {
        self.active_object = None;
        self.curr_sel = CurrSel::Nothing;
        self.selected.clear();
    }

    /// Is `id` currently in the multi-selection?
    pub fn is_selected(&self, id: ObjectId) -> bool {
        self.selected.contains(&id)
    }
}
