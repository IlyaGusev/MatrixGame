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
#[derive(Debug, Clone, Copy, Default)]
pub struct Side {
    /// `m_Id` — side index (MatrixSide.hpp). `1 = PLAYER_SIDE`.
    pub id: i32,
    /// `m_ActiveObject` — what this side currently has selected. The
    /// C++ stores a raw `CMatrixMapStatic *`; we use `ObjectId` so a
    /// freed object reads as a stale handle and `active_object_live`
    /// returns false (matches the C++ `m_ActiveObject->m_Object`
    /// tombstone check).
    pub active_object: Option<ObjectId>,
    /// `m_CurrSel` — enum above.
    pub curr_sel: CurrSel,
}

impl Side {
    pub fn new(id: i32) -> Self {
        Self { id, active_object: None, curr_sel: CurrSel::Nothing }
    }

    /// Set the selection. Matches `CMatrixSide::SelectObject`
    /// (MatrixSide.cpp); picks `CurrSel` from the object type so
    /// callers don't have to think about the enum overlay.
    pub fn select(&mut self, id: ObjectId, curr_sel: CurrSel) {
        self.active_object = Some(id);
        self.curr_sel = curr_sel;
    }

    /// Clear the selection — C++ `CMatrixSide::UnSelect` equivalent.
    pub fn clear(&mut self) {
        self.active_object = None;
        self.curr_sel = CurrSel::Nothing;
    }
}
