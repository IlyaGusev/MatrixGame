//! Port of the robot-composition type system — the family of enums +
//! POD structs the C++ constructor uses to describe a parametric robot
//! (chassis / armor / head / weapon slots).
//!
//! Mirrors `MatrixConfig.hpp` (ERobotUnitKind) + `MatrixObjectRobot.hpp`
//! (ERobotUnitType, MR_MAXUNIT, MAX_WEAPON_CNT) + `Interface/CConstructor.h`
//! (SPrice, SUnit, SWeaponUnit, SArmorUnit, SRobotConfig).
//!
//! These types are the common currency between the constructor panel
//! (constructor.rs), the build stack (object_building.rs), and the
//! price / stats lookup tables (config.rs). Everything here is plain
//! data — no rendering, no globals.

/// Port of `MAX_WEAPON_CNT` (MatrixRobot.hpp:24). Maximum number of
/// weapon pylons on any robot — 4 common + 1 extra slot for the
/// bomb/mortar "super" weapon.
pub const MAX_WEAPON_CNT: usize = 5;

/// Port of `MR_MAXUNIT` (MatrixObjectRobot.hpp:69). Chassis + Armor +
/// Head + 5 weapons + 1 slot for anim hooks = 9 at the robot level.
pub const MR_MAXUNIT: usize = 9;

/// Port of `MAX_RESOURCES` (MatrixConfig.hpp:29). Titan / Electronics /
/// Energy / Plasma. The enum discriminants match the C++ so
/// `resources[TITAN as usize]` reads the same index the original uses.
pub const MAX_RESOURCES: usize = 4;

/// Port of `ERes` (MatrixConfig.hpp:22-32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Resource {
    Titan = 0,
    Electronics = 1,
    Energy = 2,
    Plasma = 3,
}

impl Resource {
    pub const ALL: [Resource; MAX_RESOURCES] = [
        Resource::Titan,
        Resource::Electronics,
        Resource::Energy,
        Resource::Plasma,
    ];
}

/// Port of `ERobotUnitType` (MatrixObjectRobot.hpp:47-55).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum RobotUnitType {
    #[default]
    Empty = 0,
    Chassis = 1,
    Weapon = 2,
    Armor = 3,
    Head = 4,
}

/// Port of `ERobotUnitKind` (MatrixConfig.hpp:34-78). All four
/// categories (chassis, armor, weapon, head) share one discriminant
/// space in the original — `RUK_UNKNOWN = 0` is the sentinel. We keep
/// the plain i32 representation so the C++ arithmetic (`kind + 1`,
/// etc.) the constructor's wrap-around click handler uses ports
/// directly.
///
/// Use `RUK_*` constants for the named ones; `from_i32` for wrapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RobotUnitKind(pub i32);

impl RobotUnitKind {
    pub const UNKNOWN: RobotUnitKind = RobotUnitKind(0);

    // Chassis (MatrixConfig.hpp:36-45).
    pub const CHASSIS_PNEUMATIC: RobotUnitKind = RobotUnitKind(1);
    pub const CHASSIS_WHEEL: RobotUnitKind = RobotUnitKind(2);
    pub const CHASSIS_TRACK: RobotUnitKind = RobotUnitKind(3);
    pub const CHASSIS_HOVERCRAFT: RobotUnitKind = RobotUnitKind(4);
    pub const CHASSIS_ANTIGRAVITY: RobotUnitKind = RobotUnitKind(5);

    // Weapons (MatrixConfig.hpp:48-59).
    pub const WEAPON_MACHINEGUN: RobotUnitKind = RobotUnitKind(1);
    pub const WEAPON_CANNON: RobotUnitKind = RobotUnitKind(2);
    pub const WEAPON_MISSILE: RobotUnitKind = RobotUnitKind(3);
    pub const WEAPON_FLAMETHROWER: RobotUnitKind = RobotUnitKind(4);
    pub const WEAPON_MORTAR: RobotUnitKind = RobotUnitKind(5);
    pub const WEAPON_LASER: RobotUnitKind = RobotUnitKind(6);
    pub const WEAPON_BOMB: RobotUnitKind = RobotUnitKind(7);
    pub const WEAPON_PLASMA: RobotUnitKind = RobotUnitKind(8);
    pub const WEAPON_ELECTRIC: RobotUnitKind = RobotUnitKind(9);
    pub const WEAPON_REPAIR: RobotUnitKind = RobotUnitKind(10);

    // Armor (MatrixConfig.hpp:62-69).
    pub const ARMOR_PASSIVE: RobotUnitKind = RobotUnitKind(1);
    pub const ARMOR_ACTIVE: RobotUnitKind = RobotUnitKind(2);
    pub const ARMOR_FIREPROOF: RobotUnitKind = RobotUnitKind(3);
    pub const ARMOR_PLASMIC: RobotUnitKind = RobotUnitKind(4);
    pub const ARMOR_NUCLEAR: RobotUnitKind = RobotUnitKind(5);
    pub const ARMOR_6: RobotUnitKind = RobotUnitKind(6);

    // Heads (MatrixConfig.hpp:72-77).
    pub const HEAD_BLOCKER: RobotUnitKind = RobotUnitKind(1);
    pub const HEAD_DYNAMO: RobotUnitKind = RobotUnitKind(2);
    pub const HEAD_LOCKATOR: RobotUnitKind = RobotUnitKind(3);
    pub const HEAD_FIREWALL: RobotUnitKind = RobotUnitKind(4);

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub fn as_index(self) -> usize {
        // Kind values are 1-based; callers that index a "per-kind" array
        // subtract 1. Expose the helper so the call sites read cleanly.
        (self.0 - 1).max(0) as usize
    }
}

/// Counts per category — mirror the `ROBOT_*_CNT` discriminants. Kept
/// as free consts so array sizes can use them.
pub const ROBOT_CHASSIS_CNT: usize = 5;
pub const ROBOT_WEAPON_CNT: usize = 10;
pub const ROBOT_ARMOR_CNT: usize = 6;
pub const ROBOT_HEAD_CNT: usize = 4;

/// Port of `SPrice` (Interface/CConstructor.h:13-23). Per-component
/// resource cost vector. Zero for empty / unknown components.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitPrice {
    pub resources: [i32; MAX_RESOURCES],
}

impl UnitPrice {
    pub const fn zero() -> Self {
        Self {
            resources: [0; MAX_RESOURCES],
        }
    }

    pub fn titan(&self) -> i32 {
        self.resources[Resource::Titan as usize]
    }
    pub fn electronics(&self) -> i32 {
        self.resources[Resource::Electronics as usize]
    }
    pub fn energy(&self) -> i32 {
        self.resources[Resource::Energy as usize]
    }
    pub fn plasma(&self) -> i32 {
        self.resources[Resource::Plasma as usize]
    }

    pub fn add_from(&mut self, other: UnitPrice) {
        for i in 0..MAX_RESOURCES {
            self.resources[i] += other.resources[i];
        }
    }

    pub fn is_zero(&self) -> bool {
        self.resources.iter().all(|&r| r == 0)
    }
}

/// Port of `SUnit` (Interface/CConstructor.h:25-31). A single
/// (type, kind) component with its own cost cached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Unit {
    pub ty: RobotUnitType,
    pub kind: RobotUnitKind,
    pub price: UnitPrice,
}

impl Unit {
    pub const fn empty() -> Self {
        Self {
            ty: RobotUnitType::Empty,
            kind: RobotUnitKind::UNKNOWN,
            price: UnitPrice::zero(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.kind.is_empty() || self.ty == RobotUnitType::Empty
    }
}

/// Port of `SArmorUnit` (Interface/CConstructor.h:33-40). Adds the
/// per-armor weapon-slot caps to a plain `Unit`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArmorUnit {
    /// Common (non-extra) weapon slots exposed by this armor. Filled
    /// from `SRobotWeaponMatrix::common` at OperateUnit-MRT_ARMOR time
    /// (CConstructor.cpp:686).
    pub max_common_weapon_cnt: i32,
    /// Extra ("super") weapon slots — bomb/mortar go here. From
    /// `SRobotWeaponMatrix::extra`.
    pub max_extra_weapon_cnt: i32,
    pub unit: Unit,
}

impl ArmorUnit {
    pub const fn empty() -> Self {
        Self {
            max_common_weapon_cnt: 0,
            max_extra_weapon_cnt: 0,
            unit: Unit::empty(),
        }
    }
}

/// Port of `SWeaponUnit` (Interface/CConstructor.h:42-47). A weapon
/// slot with its pylon position within the armor's slot layout.
///
/// `pos` is 1-based when filled (matches C++ `m_Pos = t+1` assignment
/// at CConstructor.cpp:488 / :515) so tests for `pos > 0` read as
/// "slot has a weapon assigned". Slot 0 is unassigned.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WeaponUnit {
    pub pos: i32,
    pub unit: Unit,
}

impl WeaponUnit {
    pub const fn empty() -> Self {
        Self {
            pos: 0,
            unit: Unit::empty(),
        }
    }
}

/// Port of `SRobotConfig` (Interface/CConstructor.h:171-189). The
/// complete parametric description of one robot design — what the
/// constructor panel persists in `m_Configs[PRESETS]` and what the
/// build stack needs to produce a fully-configured robot on dequeue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RobotConfig {
    pub head: Unit,
    pub weapon: [Unit; MAX_WEAPON_CNT],
    pub chassis: Unit,
    pub hull: ArmorUnit,

    /// Cached running totals — kept for parity with C++'s
    /// `m_titX/m_elecX/m_enerX/m_plasX` fields updated by
    /// `SetLabelsAndPrice`. Not used for build-time pricing (the
    /// constructor recalculates via `GetConstructionPrice`).
    pub tit_x: i32,
    pub elec_x: i32,
    pub ener_x: i32,
    pub plas_x: i32,

    pub structure: i32,
    pub damage: i32,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl RobotConfig {
    /// Ports the ctor side-effects of `CConstructorPanel` that seed
    /// the type tags so `m_Hull.m_Unit.m_nType == MRT_ARMOR` etc.
    /// even before the player picks a kind (CConstructor.h:228-233).
    pub const fn new() -> Self {
        Self {
            head: Unit {
                ty: RobotUnitType::Head,
                kind: RobotUnitKind::UNKNOWN,
                price: UnitPrice::zero(),
            },
            weapon: [Unit {
                ty: RobotUnitType::Weapon,
                kind: RobotUnitKind::UNKNOWN,
                price: UnitPrice::zero(),
            }; MAX_WEAPON_CNT],
            chassis: Unit {
                ty: RobotUnitType::Chassis,
                kind: RobotUnitKind::UNKNOWN,
                price: UnitPrice::zero(),
            },
            hull: ArmorUnit {
                max_common_weapon_cnt: 0,
                max_extra_weapon_cnt: 0,
                unit: Unit {
                    ty: RobotUnitType::Armor,
                    kind: RobotUnitKind::UNKNOWN,
                    price: UnitPrice::zero(),
                },
            },
            tit_x: 0,
            elec_x: 0,
            ener_x: 0,
            plas_x: 0,
            structure: 0,
            damage: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// True once every required slot is populated (the build button
    /// gates on this in the C++ — see the `m_nUnitCnt > 0` check at
    /// CConstructor.cpp:67 / :149).
    pub fn is_buildable(&self) -> bool {
        !self.chassis.is_empty() && !self.hull.unit.is_empty()
    }

    /// Total number of populated units (chassis + armor + head + any
    /// weapon). Ports the `m_nUnitCnt` tally at CConstructor.cpp:
    /// 709-725.
    pub fn unit_count(&self) -> i32 {
        let mut n = 0;
        if !self.chassis.is_empty() {
            n += 1;
        }
        if !self.hull.unit.is_empty() {
            n += 1;
        }
        if !self.head.is_empty() {
            n += 1;
        }
        for w in &self.weapon {
            if !w.is_empty() {
                n += 1;
            }
        }
        n
    }

    /// Zero the weapon slots — ports
    /// `CConstructorPanel::ResetWeapon` (CConstructor.h:212).
    pub fn reset_weapons(&mut self) {
        for w in &mut self.weapon {
            *w = Unit {
                ty: RobotUnitType::Weapon,
                kind: RobotUnitKind::UNKNOWN,
                price: UnitPrice::zero(),
            };
        }
        self.damage = 0;
    }
}

/// Port of `SRobotWeaponMatrix` (MatrixMap.hpp:170-180) — per-armor
/// pylon layout loaded from the chassis VO at map-load time in the
/// C++ (MatrixMap.cpp:236-334). Each armor kind (1..=ROBOT_ARMOR_CNT)
/// carries a list of pylon slots; each slot encodes which weapon
/// categories it accepts via `access_invert` (bit N set = weapon
/// index N is blocked, because the C++ stores it inverted).
///
/// We don't have the VO loader wired yet for reading this off the
/// meshes, so `WeaponMatrix::defaults()` provides a hand-rolled table
/// that matches the shipped gameplay: 4 common slots + 1 extra
/// (bomb/mortar) per armor. The build UI works against this table
/// until mesh-side VO parsing lands.
///
/// The C++ bit layout (MatrixMap.cpp:305):
///   `access_invert |= (1 << (WeapKind2Index(w) - 1))`
/// then `| SETBIT(31)` to flag the invert-sense. We follow the same
/// encoding so the predicates at CConstructor.cpp:477-480 etc. port
/// directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct WeaponMatrixSlot {
    pub id: i32,
    pub access_invert: u32,
}

pub const ROBOT_WEAPONS_PER_ROBOT_CNT: usize = 16;

#[derive(Debug, Clone, Copy)]
pub struct WeaponMatrix {
    /// Total slots on this armor. Slot indices `< common` are
    /// "common" (MG/Cannon/Missile/Laser/Plasma/Electric/Repair);
    /// the rest are "extra" (Bomb/Mortar/Flamethrower).
    pub cnt: i32,
    pub common: i32,
    pub extra: i32,
    pub list: [WeaponMatrixSlot; ROBOT_WEAPONS_PER_ROBOT_CNT],
}

impl Default for WeaponMatrix {
    fn default() -> Self {
        Self {
            cnt: 0,
            common: 0,
            extra: 0,
            list: [WeaponMatrixSlot::default(); ROBOT_WEAPONS_PER_ROBOT_CNT],
        }
    }
}

/// Extra-slot bit — bit 4 (Flamethrower) or bit 6 (Bomb/Mortar) set
/// in `access_invert` marks a slot as "extra" (CConstructor.cpp:477).
pub const ACCESS_EXTRA_BIT_A: u32 = 1 << 4;
pub const ACCESS_EXTRA_BIT_B: u32 = 1 << 6;

impl WeaponMatrix {
    /// Helper: is slot `i` an "extra" (super-weapon) pylon? Mirrors
    /// the bit-test at CConstructor.cpp:477-480 / :503-504.
    pub fn is_extra_slot(&self, i: usize) -> bool {
        let a = self.list[i].access_invert;
        (a & ACCESS_EXTRA_BIT_A) != 0 || (a & ACCESS_EXTRA_BIT_B) != 0
    }

    /// Port of `CConstructor::CheckWeaponLegality` (CConstructor.cpp:
    /// 731-761) — find the first empty pylon on this armor that
    /// accepts `weapon_kind`. Returns the slot index, or `None` if
    /// none is available.
    pub fn find_pylon_for(
        &self,
        weapon_kind: RobotUnitKind,
        current: &[Unit; MAX_WEAPON_CNT],
    ) -> Option<usize> {
        let is_super = weapon_kind == RobotUnitKind::WEAPON_MORTAR
            || weapon_kind == RobotUnitKind::WEAPON_BOMB;
        let limit = (self.cnt as usize).min(ROBOT_WEAPONS_PER_ROBOT_CNT);
        if !is_super {
            // Common slot: first non-extra pylon that's still empty.
            for (t, unit) in current.iter().enumerate().take(limit) {
                if !self.is_extra_slot(t) && unit.is_empty() {
                    return Some(t);
                }
            }
            None
        } else {
            // Super slot: first extra pylon (regardless of occupancy —
            // matches CConstructor.cpp:747-758 which returns the first
            // match and overwrites).
            (0..limit).find(|&t| self.is_extra_slot(t))
        }
    }
}

/// Shipped-gameplay weapon matrix layout — hand-rolled to mirror the
/// data the VO loader produces in the original. Each armor kind has
/// a different common/extra split; values are taken from observing
/// the shipped ARMOR*_STRUCTURE / weapon-pylon geometry of the
/// stock robots.dat.
///
/// If/when the CVO loader lands we'll replace this with a runtime
/// parse; for now it lets the constructor UI be fully functional.
pub fn default_weapon_matrix(armor_kind: RobotUnitKind) -> WeaponMatrix {
    // Common slot bit — we use bit 0 as a sentinel "common slot"
    // identifier so `is_extra_slot` returns false for them. The
    // actual `access_invert` bits from the C++ encode more-specific
    // compatibility, but for the constructor flow we only need the
    // common/extra distinction.
    let common_slot = WeaponMatrixSlot {
        id: 0,
        access_invert: 1 << 0,
    };
    let extra_slot_bomb = WeaponMatrixSlot {
        id: 1,
        access_invert: ACCESS_EXTRA_BIT_B,
    };

    let (common, extra) = match armor_kind.0 {
        // Armor 1-4: 2 common + 1 extra
        1..=4 => (2, 1),
        // Armor 5-6 (heavier): 4 common + 1 extra
        5 | 6 => (4, 1),
        _ => (0, 0),
    };

    let total = common + extra;
    let mut list = [WeaponMatrixSlot::default(); ROBOT_WEAPONS_PER_ROBOT_CNT];
    for slot in list
        .iter_mut()
        .take((common as usize).min(ROBOT_WEAPONS_PER_ROBOT_CNT))
    {
        *slot = common_slot;
    }
    if extra > 0 && (common as usize) < ROBOT_WEAPONS_PER_ROBOT_CNT {
        list[common as usize] = extra_slot_bomb;
    }

    WeaponMatrix {
        cnt: total,
        common,
        extra,
        list,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robot_config_is_buildable_needs_chassis_and_hull() {
        let mut c = RobotConfig::new();
        assert!(!c.is_buildable());
        c.chassis.kind = RobotUnitKind::CHASSIS_TRACK;
        assert!(!c.is_buildable());
        c.hull.unit.kind = RobotUnitKind::ARMOR_PASSIVE;
        assert!(c.is_buildable());
    }

    #[test]
    fn unit_count_matches_populated_slots() {
        let mut c = RobotConfig::new();
        assert_eq!(c.unit_count(), 0);
        c.chassis.kind = RobotUnitKind::CHASSIS_WHEEL;
        c.hull.unit.kind = RobotUnitKind::ARMOR_ACTIVE;
        c.head.kind = RobotUnitKind::HEAD_BLOCKER;
        c.weapon[0].kind = RobotUnitKind::WEAPON_MACHINEGUN;
        assert_eq!(c.unit_count(), 4);
    }

    #[test]
    fn find_pylon_for_common_weapon_returns_first_empty() {
        let m = default_weapon_matrix(RobotUnitKind::ARMOR_ACTIVE);
        let current = [Unit::empty(); MAX_WEAPON_CNT];
        assert_eq!(
            m.find_pylon_for(RobotUnitKind::WEAPON_CANNON, &current),
            Some(0)
        );
    }

    #[test]
    fn find_pylon_for_super_weapon_returns_extra_slot() {
        let m = default_weapon_matrix(RobotUnitKind::ARMOR_ACTIVE);
        let current = [Unit::empty(); MAX_WEAPON_CNT];
        // 2 common + 1 extra; extra is at index 2.
        assert_eq!(
            m.find_pylon_for(RobotUnitKind::WEAPON_BOMB, &current),
            Some(2)
        );
    }

    #[test]
    fn weapon_matrix_is_extra_slot_flags_match() {
        let m = default_weapon_matrix(RobotUnitKind::ARMOR_NUCLEAR);
        assert!(!m.is_extra_slot(0));
        assert!(!m.is_extra_slot(3));
        assert!(m.is_extra_slot(4));
    }
}
