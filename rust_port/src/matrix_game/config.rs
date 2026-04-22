//! Partial port of `CMatrixConfig` (MatrixConfig.{cpp,hpp}).
//!
//! The original config struct covers gamma/keybinds/sound volumes as
//! well as the weapon-damage lookup tables. This file only ports the
//! weapon-damage portion of that — specifically the per-weapon damage
//! table used by `CMatrixMapObject::Damage` (MatrixObject.cpp:145 etc).
//! Other parts of `g_Config` land with their call sites.
//!
//! The tables are loaded from `robots.dat` via the CStorage-backed
//! BlockPar accessors on [`Storage`]. Tests can construct the structs
//! directly without touching the filesystem.

use crate::matrix_game::effects::weapon::{weap_name_to_index, WEAPON_COUNT};
use crate::matrix_lib::base::storage::Storage;
use crate::matrix_lib::base::wstr;

/// Port of `SWeaponDamage` (MatrixConfig.hpp around line 574). The C++
/// carries `damage`, `mindamage`, plus `friend_damage` for the
/// building/cannon variants — applied when the attacker is on the
/// same side as the target (friendly-fire gets its own number so
/// explosions can hurt your own base less than the enemy's).
/// `mindamage` is a floor — the victim only subtracts `damage` when
/// its current hit-points exceed `mindamage`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WeaponDamage {
    pub damage: i32,
    pub mindamage: i32,
    /// Present on building / cannon damage tables
    /// (MatrixConfig.cpp:502, :532). Object damages leave this at 0 —
    /// MapObjects don't distinguish friendly fire.
    pub friend_damage: i32,
}

/// Per-category lookup indexed by `weap_to_index(weap)`. Ports
/// `m_ObjectDamages[WEAPON_COUNT]` for now; sibling arrays (robot /
/// cannon / building / flyer) will land when their subclasses need
/// them.
#[derive(Debug, Clone, Copy)]
pub struct ObjectDamages {
    pub table: [WeaponDamage; WEAPON_COUNT],
}

impl Default for ObjectDamages {
    fn default() -> Self {
        // `memset(&m_ObjectDamages, 0, sizeof(m_ObjectDamages))`
        // at MatrixConfig.cpp:593 — zero-fill before reading.
        Self { table: [WeaponDamage::default(); WEAPON_COUNT] }
    }
}

impl ObjectDamages {
    /// Load the `Weapons/Damages/Object` block from `robots.dat`.
    /// Each param is `WEAPON_<NAME> = <damage>[,<mindamage>]`
    /// (MatrixConfig.cpp:591-607).
    ///
    /// Returns `None` if the block isn't present (caller should fall
    /// back to `Default::default()` — same as the C++ zero-fill).
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        // The BlockPar root record is "da" (StoreBlockPar auto-names:
        // see form_game.rs for precedent with "Config").
        let weapons_rec = stor.block_record("da", "Weapons")?;
        let damages_rec = stor.block_record(&weapons_rec, "Damages")?;
        let object_rec  = stor.block_record(&damages_rec, "Object")?;

        // Iterate the param keys in the "Object" record by reading the
        // Storage columns directly — we need both names ("0") and
        // values ("1") enumerated in lock-step (block_param indexes by
        // key but would cost us a scan for each known weapon name).
        let keys = stor.get_buf(&object_rec, "0")?;
        let values = stor.get_buf(&object_rec, "1")?;
        let n = keys.arrays_count().min(values.arrays_count());

        let mut out = Self::default();
        for i in 0..n {
            let name = keys.get_as_wstr(i);
            let Some(idx) = weap_name_to_index(&name) else {
                // Unknown key — same as `WeapName2Index` returning -1;
                // the C++ guards with `if (idx >=0)` at MatrixConfig.cpp:600.
                continue;
            };
            let val = values.get_as_wstr(i);
            let damage = wstr::int_par(&val, 0, ",");
            let nn = wstr::count_par(&val, ",");
            let mindamage = if nn > 1 { wstr::int_par(&val, 1, ",") } else { 0 };
            // The Object block never carries friend_damage — the C++
            // loader at MatrixConfig.cpp:591-607 only reads two fields.
            out.table[idx] = WeaponDamage { damage, mindamage, friend_damage: 0 };
        }
        Some(out)
    }

    /// Look up the entry for a weapon discriminant. Non-damage-table
    /// weapons (e.g. `WEAPON_NONE`) return `None`.
    pub fn get(
        &self,
        weap: crate::matrix_game::effects::weapon::Weapon,
    ) -> Option<WeaponDamage> {
        let idx = crate::matrix_game::effects::weapon::weap_to_index(weap)?;
        Some(self.table[idx])
    }
}

/// Building-specific damages + per-kind HITPOINT caps. Ports the
/// `Weapons/Damages/Building` block (MatrixConfig.cpp:507-535). The
/// `HITPOINT` key is an extra `,`-list of per-`EBuildingType` hp
/// ceilings — 6 entries for Base/Titan/Plasma/Electronic/Energy/Repair.
#[derive(Debug, Clone, Copy)]
pub struct BuildingDamages {
    pub table: [WeaponDamage; WEAPON_COUNT],
    /// `HITPOINT` per building type. `hitpoint[kind as usize]` yields
    /// the max HP to seed via `Building::init_max_hitpoint` at spawn.
    pub hitpoint: [i32; 6],
}

impl Default for BuildingDamages {
    fn default() -> Self {
        Self {
            table: [WeaponDamage::default(); WEAPON_COUNT],
            hitpoint: [0; 6],
        }
    }
}

impl BuildingDamages {
    /// Load `Weapons/Damages/Building` (MatrixConfig.cpp:507-535). Keys
    /// ending in `HITPOINT` are the per-type hp list; other keys are
    /// weapon names identical to the Object table.
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let weapons_rec  = stor.block_record("da", "Weapons")?;
        let damages_rec  = stor.block_record(&weapons_rec, "Damages")?;
        let building_rec = stor.block_record(&damages_rec, "Building")?;

        let keys = stor.get_buf(&building_rec, "0")?;
        let values = stor.get_buf(&building_rec, "1")?;
        let n = keys.arrays_count().min(values.arrays_count());

        let mut out = Self::default();
        for i in 0..n {
            let name = keys.get_as_wstr(i);
            let val = values.get_as_wstr(i);
            if name == "HITPOINT" {
                let cnt = wstr::count_par(&val, ",").min(out.hitpoint.len());
                for j in 0..cnt {
                    out.hitpoint[j] = wstr::int_par(&val, j, ",");
                }
                continue;
            }
            let Some(idx) = weap_name_to_index(&name) else { continue };
            let nn = wstr::count_par(&val, ",");
            let damage = wstr::int_par(&val, 0, ",");
            let mindamage = if nn > 1 { wstr::int_par(&val, 1, ",") } else { 0 };
            // MatrixConfig.cpp:532 — if friend_damage column absent,
            // it defaults to `damage`, not 0.
            let friend_damage =
                if nn > 2 { wstr::int_par(&val, 2, ",") } else { damage };
            out.table[idx] = WeaponDamage { damage, mindamage, friend_damage };
        }
        Some(out)
    }

    pub fn get(
        &self,
        weap: crate::matrix_game::effects::weapon::Weapon,
    ) -> Option<WeaponDamage> {
        let idx = crate::matrix_game::effects::weapon::weap_to_index(weap)?;
        Some(self.table[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_game::effects::weapon::{WEAPON_BIGBOOM, WEAPON_NONE, WEAPON_PLASMA};

    #[test]
    fn default_is_zero_filled() {
        let d = ObjectDamages::default();
        for e in d.table.iter() {
            assert_eq!(*e, WeaponDamage::default());
        }
    }

    #[test]
    fn get_maps_through_weap_to_index() {
        let mut d = ObjectDamages::default();
        d.table[5] = WeaponDamage { damage: 400, mindamage: 100, friend_damage: 0 }; // BIGBOOM
        assert_eq!(
            d.get(WEAPON_BIGBOOM),
            Some(WeaponDamage { damage: 400, mindamage: 100, friend_damage: 0 }),
        );
        d.table[0] = WeaponDamage { damage: 80, mindamage: 40, friend_damage: 0 };   // PLASMA
        assert_eq!(
            d.get(WEAPON_PLASMA),
            Some(WeaponDamage { damage: 80, mindamage: 40, friend_damage: 0 }),
        );
        // `WEAPON_NONE` has no table slot.
        assert_eq!(d.get(WEAPON_NONE), None);
    }
}
