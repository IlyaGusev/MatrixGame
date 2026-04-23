//! Port of `EWeapon` from `MatrixEffectWeapon.hpp:34-62`.
//!
//! The C++ enum carries sparse, partly-magical discriminants (200, 70,
//! 10000000, …) because a few callers reuse them as damage magnitudes
//! or sort keys. We preserve those values for parity with saved-game
//! blobs and config files. Only the discriminants are ported — the
//! `CMatrixEffectWeapon` class itself (effect lifecycle + damage
//! propagation) lands when the effects subsystem arrives.
//!
//! Named `effects/weapon.rs` to mirror the C++ file layout
//! (`Effects/MatrixEffectWeapon.hpp`).

pub type Weapon = u32;

pub const WEAPON_NONE: Weapon = 0;
pub const WEAPON_VOLCANO: Weapon = 70;
pub const WEAPON_PLASMA: Weapon = 200;
pub const WEAPON_HOMING_MISSILE: Weapon = 1000;
pub const WEAPON_BOMB: Weapon = 2500;
pub const WEAPON_FLAMETHROWER: Weapon = 60;
pub const WEAPON_BIGBOOM: Weapon = 10_000;
pub const WEAPON_LIGHTENING: Weapon = 99;
pub const WEAPON_LASER: Weapon = 98;
pub const WEAPON_GUN: Weapon = 598;
pub const WEAPON_REPAIR: Weapon = 57;

pub const WEAPON_CANNON0: Weapon = 300;
pub const WEAPON_CANNON1: Weapon = 998;
pub const WEAPON_CANNON2: Weapon = 97;
pub const WEAPON_CANNON3: Weapon = 1002;

pub const WEAPON_ABLAZE: Weapon = 10_000_000;
pub const WEAPON_SHORTED: Weapon = 10_000_001;
pub const WEAPON_DEBRIS: Weapon = 10_000_002;

pub const WEAPON_INSTANT_DEATH: Weapon = 0x7FFF_FFFE;

/// Ports the fire-weapon predicate used by `CMatrixMapObject::Damage`
/// (MatrixObject.cpp:115) — the C++ spells out the OR chain literally.
pub fn is_fire_weapon(w: Weapon) -> bool {
    matches!(
        w,
        WEAPON_BIGBOOM | WEAPON_HOMING_MISSILE | WEAPON_BOMB | WEAPON_PLASMA | WEAPON_FLAMETHROWER
    )
}

/// `WEAPON_COUNT` from the EWeapon enum — length of the damage
/// lookup tables. Indices are dense `0..17`, produced by
/// [`weap_to_index`].
pub const WEAPON_COUNT: usize = 17;

/// Dense 0..16 index used by `g_Config.m_ObjectDamages[idx]`.
/// Ports `Weap2Index` (MatrixEffectWeapon.hpp:108-128). Returns `None`
/// for weapons without a damage-table slot (e.g. `WEAPON_NONE`,
/// `WEAPON_INSTANT_DEATH`).
pub fn weap_to_index(w: Weapon) -> Option<usize> {
    Some(match w {
        WEAPON_PLASMA => 0,
        WEAPON_VOLCANO => 1,
        WEAPON_HOMING_MISSILE => 2,
        WEAPON_BOMB => 3,
        WEAPON_FLAMETHROWER => 4,
        WEAPON_BIGBOOM => 5,
        WEAPON_LIGHTENING => 6,
        WEAPON_LASER => 7,
        WEAPON_GUN => 8,
        WEAPON_ABLAZE => 9,
        WEAPON_SHORTED => 10,
        WEAPON_DEBRIS => 11,
        WEAPON_REPAIR => 12,
        WEAPON_CANNON0 => 13,
        WEAPON_CANNON1 => 14,
        WEAPON_CANNON2 => 15,
        WEAPON_CANNON3 => 16,
        _ => return None,
    })
}

/// `Weap2Index`'s string-keyed inverse, used by `MatrixConfig.cpp` to
/// parse the `Weapons/Damages/Object` block (MatrixEffectWeapon.hpp:64-84).
/// Returns the same dense index as [`weap_to_index`] applied to the
/// matching weapon discriminant.
pub fn weap_name_to_index(name: &str) -> Option<usize> {
    Some(match name {
        "WEAPON_PLASMA" => 0,
        "WEAPON_VOLCANO" => 1,
        "WEAPON_HOMING_MISSILE" => 2,
        "WEAPON_BOMB" => 3,
        "WEAPON_FLAMETHROWER" => 4,
        "WEAPON_BIGBOOM" => 5,
        "WEAPON_LIGHTENING" => 6,
        "WEAPON_LASER" => 7,
        "WEAPON_GUN" => 8,
        "WEAPON_ABLAZE" => 9,
        "WEAPON_SHORTED" => 10,
        "WEAPON_DEBRIS" => 11,
        "WEAPON_REPAIR" => 12,
        "WEAPON_CANNON0" => 13,
        "WEAPON_CANNON1" => 14,
        "WEAPON_CANNON2" => 15,
        "WEAPON_CANNON3" => 16,
        _ => return None,
    })
}
