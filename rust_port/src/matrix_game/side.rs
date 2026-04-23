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
}

impl Side {
    pub fn new(id: i32) -> Self {
        Self {
            id,
            active_object: None,
            curr_sel: CurrSel::Nothing,
            selected: Vec::new(),
        }
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
        self.curr_sel = if primary.is_some() { curr_sel } else { CurrSel::Nothing };
    }

    /// Clear the selection — C++ `CMatrixSide::UnSelect` equivalent.
    pub fn clear(&mut self) {
        self.active_object = None;
        self.curr_sel = CurrSel::Nothing;
        self.selected.clear();
    }

    /// Is `id` currently in the multi-selection?
    pub fn is_selected(&self, id: ObjectId) -> bool {
        self.selected.iter().any(|&x| x == id)
    }
}
