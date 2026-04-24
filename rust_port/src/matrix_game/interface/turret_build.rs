//! Port of the turret-placement UI state. Backs
//! `PREORDER_BUILD_TURRET` (CInterface.h:57) + `BeginBuildTurret`
//! (CInterface.h:379, CInterface.cpp:4650+) + the turret-placement
//! rendering/click hooks at MatrixFormGame.cpp:1405, 1498-1512 and
//! MatrixMap.cpp:1396, 1692.
//!
//! The C++ stores placement state globally on `CIFaceList::m_IfListFlags`
//! plus a scratch turret instance on `CIFaceList`. We collect it into
//! this struct since only one turret-build session can be active at a time;
//! the global list stays cleaner without another side-effect flag.

/// Port of `ECannonKind` — the 4 cannon variants the UI exposes
/// (RUK_TURRET_CANNON / GUN / LASER / ROCKET). The C++ stores this
/// as `1..=4` matching the UI element ids `turret1..turret4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TurretKind {
    Cannon = 1,
    Gun = 2,
    Laser = 3,
    Rocket = 4,
}

impl TurretKind {
    pub fn from_i32(n: i32) -> Option<Self> {
        Some(match n {
            1 => TurretKind::Cannon,
            2 => TurretKind::Gun,
            3 => TurretKind::Laser,
            4 => TurretKind::Rocket,
            _ => return None,
        })
    }

    pub fn to_index(self) -> usize {
        (self as u8 as usize).saturating_sub(1)
    }
}

/// The turret-build session state. `active` flips on when the player
/// clicks one of the `turret1..turret4` buttons or `buca` on a
/// building; the next map click either places the turret (on a valid
/// building slot) or cancels (anywhere else).
#[derive(Debug, Clone, Copy, Default)]
pub struct TurretBuild {
    pub active: bool,
    pub kind: Option<TurretKind>,
    /// The building that owns the turret placement — ports
    /// `g_IFaceList->m_BuildCa` / `m_CannonForBuild` (CInterface.h).
    /// World-space position + angle of the cannon is derived from
    /// this building + the slot the user hovers.
    pub parent: Option<crate::matrix_game::map_static::ObjectId>,
    /// Cursor hover position in world space (x, y) — drives the
    /// placement preview. Updated on mouse-move while `active`.
    pub cursor_world: (f32, f32),
    /// Which turret slot on the parent building the cursor is over
    /// (1..=turrets_max) — 0 means none.
    pub hovered_slot: i32,
}

impl TurretBuild {
    pub fn new() -> Self {
        Self::default()
    }

    /// Port of `CInterface::BeginBuildTurret(no)` (CInterface.cpp:4650+).
    /// Enters placement mode for the given kind; `parent` is the base
    /// the turret is attached to.
    pub fn begin(&mut self, kind: TurretKind, parent: crate::matrix_game::map_static::ObjectId) {
        self.active = true;
        self.kind = Some(kind);
        self.parent = Some(parent);
        self.cursor_world = (0.0, 0.0);
        self.hovered_slot = 0;
    }

    /// Port of the "click anywhere / esc" cancel path.
    pub fn cancel(&mut self) {
        *self = TurretBuild::default();
    }

    /// True iff a turret is currently being placed.
    pub fn is_active(&self) -> bool {
        self.active
    }
}
