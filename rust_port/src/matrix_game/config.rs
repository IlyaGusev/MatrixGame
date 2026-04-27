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
        Self {
            table: [WeaponDamage::default(); WEAPON_COUNT],
        }
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
        let object_rec = stor.block_record(&damages_rec, "Object")?;

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
            let mindamage = if nn > 1 {
                wstr::int_par(&val, 1, ",")
            } else {
                0
            };
            // The Object block never carries friend_damage — the C++
            // loader at MatrixConfig.cpp:591-607 only reads two fields.
            out.table[idx] = WeaponDamage {
                damage,
                mindamage,
                friend_damage: 0,
            };
        }
        Some(out)
    }

    /// Look up the entry for a weapon discriminant. Non-damage-table
    /// weapons (e.g. `WEAPON_NONE`) return `None`.
    pub fn get(&self, weap: crate::matrix_game::effects::weapon::Weapon) -> Option<WeaponDamage> {
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
        let weapons_rec = stor.block_record("da", "Weapons")?;
        let damages_rec = stor.block_record(&weapons_rec, "Damages")?;
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
            let Some(idx) = weap_name_to_index(&name) else {
                continue;
            };
            let nn = wstr::count_par(&val, ",");
            let damage = wstr::int_par(&val, 0, ",");
            let mindamage = if nn > 1 {
                wstr::int_par(&val, 1, ",")
            } else {
                0
            };
            // MatrixConfig.cpp:532 — if friend_damage column absent,
            // it defaults to `damage`, not 0.
            let friend_damage = if nn > 2 {
                wstr::int_par(&val, 2, ",")
            } else {
                damage
            };
            out.table[idx] = WeaponDamage {
                damage,
                mindamage,
                friend_damage,
            };
        }
        Some(out)
    }

    pub fn get(&self, weap: crate::matrix_game::effects::weapon::Weapon) -> Option<WeaponDamage> {
        let idx = crate::matrix_game::effects::weapon::weap_to_index(weap)?;
        Some(self.table[idx])
    }
}

/// Port of the `g_Config.m_ItemChars[CHASSIS{N}_MOVE_SPEED]` rows at
/// MatrixConfig.cpp:874,881,888,895,902. The original stores per-chassis
/// rotation speed, slope/water corrections, structure hp, etc. — only
/// move speed is ported here because that's all `ROBOT_BASE_MOVEOUT`
/// needs. The others land when their call sites land.
///
/// Indexed by `ChassisKind as usize` (0..=4).
#[derive(Debug, Clone, Copy, Default)]
pub struct ChassisChars {
    pub move_speed: [f32; 5],
    pub rotation_speed: [f32; 5],
    /// `m_ItemChars[CHASSIS{N}_MOVE_WATER_CORR]` (MatrixConfig.cpp:875,...).
    /// Multiplier applied to `m_Speed` when `ROBOT_FLAG_ONWATER` is set
    /// (MatrixRobot.cpp:2420). Typical values: 0 for land-only chassis,
    /// 0.5–1.0 for amphibious.
    pub water_corr: [f32; 5],
    /// `m_ItemChars[CHASSIS{N}_MOVE_SLOPE_CORR_UP]` (MatrixConfig.cpp:877,...).
    /// Slope (sin θ) at which uphill speed multiplier reaches zero. Values
    /// below this lerp linearly from 1.0 at slope=0 to 0.0 at slope=this
    /// (MatrixRobot.cpp:2432-2438).
    pub slope_corr_up: [f32; 5],
    /// `m_ItemChars[CHASSIS{N}_MOVE_SLOPE_CORR_DOWN]` (MatrixConfig.cpp:876,...).
    /// Target multiplier reached when going fully downhill (slope = -1).
    /// LERP from 1.0 at slope=0 to this at slope=-1 (MatrixRobot.cpp:2441).
    pub slope_corr_down: [f32; 5],
}

impl ChassisChars {
    /// Load `Chars/Chassis/CHASSIS{N}_MOVE_SPEED` +
    /// `CHASSIS{N}_ROTATION_SPEED` from `robots.dat`
    /// (MatrixConfig.cpp:870-905). Any missing key falls back to 0.0
    /// — the C++ `BlockGet` for a missing param returns a default
    /// `0.0` double.
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let chars_rec = stor.block_record("da", "Chars")?;
        let chassis_rec = stor.block_record(&chars_rec, "Chassis")?;
        let keys = stor.get_buf(&chassis_rec, "0")?;
        let values = stor.get_buf(&chassis_rec, "1")?;
        let n = keys.arrays_count().min(values.arrays_count());

        let mut out = Self::default();
        for i in 0..n {
            let name = keys.get_as_wstr(i);
            let val = values.get_as_wstr(i);
            let Some(rest) = name.strip_prefix("CHASSIS") else {
                continue;
            };
            let (digit, field) = match rest.split_once('_') {
                Some(v) => v,
                None => continue,
            };
            let Ok(n) = digit.parse::<usize>() else {
                continue;
            };
            if !(1..=5).contains(&n) {
                continue;
            }
            let v: f32 = val.parse().unwrap_or(0.0);
            match field {
                "MOVE_SPEED" => out.move_speed[n - 1] = v,
                "ROTATION_SPEED" => out.rotation_speed[n - 1] = v,
                "MOVE_WATER_CORR" => out.water_corr[n - 1] = v,
                "MOVE_SLOPE_CORR_UP" => out.slope_corr_up[n - 1] = v,
                "MOVE_SLOPE_CORR_DN" => out.slope_corr_down[n - 1] = v,
                _ => {}
            }
        }
        Some(out)
    }
}

// ── Component prices + item chars + timings (robot constructor) ────
//
// These three tables mirror the `g_Config.m_Price[]`, `m_ItemChars[]`,
// and `m_Timings[]` flat arrays the C++ CConstructor reads at
// price / stats / build-time calculation sites (CConstructor.cpp:
// 835-898). We keep the flat layout so the indexing arithmetic in
// `GetConstructionPrice` / `GetConstructionStructure` ports directly.

use crate::matrix_game::interface::constructor::UnitPrice;
use crate::matrix_game::object_robot::RobotUnitType;
// Resource, RobotUnitKind, MAX_RESOURCES, and the ROBOT_*_CNT counts all live
// in this file (see bottom half). No external import needed here.

/// Resource indices packed per-component. Order matches
/// MatrixConfig.cpp's `HEAD1_TITAN..PRICE_LAST` enum slab, which groups
/// 4 resource entries per component in the order Titan / Electronics /
/// Energy / Plasma (see enum `EPrice` at MatrixConfig.hpp:237-366).
///
/// Layout:
///   `[head][4×HEAD_COUNT] [armor][4×ARMOR_COUNT] [weapon][4×WEAPON_COUNT] [chassis][4×CHASSIS_COUNT]`
///
/// Indexed via the `price_index_*` helpers below so the call sites
/// stay readable.
#[derive(Debug, Clone, Copy)]
pub struct PriceTable {
    pub heads: [UnitPrice; ROBOT_HEAD_CNT],
    pub armors: [UnitPrice; ROBOT_ARMOR_CNT],
    pub weapons: [UnitPrice; ROBOT_WEAPON_CNT],
    pub chassis: [UnitPrice; ROBOT_CHASSIS_CNT],
}

impl Default for PriceTable {
    fn default() -> Self {
        Self {
            heads: [UnitPrice::zero(); ROBOT_HEAD_CNT],
            armors: [UnitPrice::zero(); ROBOT_ARMOR_CNT],
            weapons: [UnitPrice::zero(); ROBOT_WEAPON_CNT],
            chassis: [UnitPrice::zero(); ROBOT_CHASSIS_CNT],
        }
    }
}

impl PriceTable {
    /// Look up the cost of one (type, kind) component. Matches the
    /// C++ memcpy-from-flat-array at CConstructor.cpp:840-852.
    pub fn get(&self, ty: RobotUnitType, kind: RobotUnitKind) -> UnitPrice {
        if kind.is_empty() {
            return UnitPrice::zero();
        }
        let idx = kind.as_index();
        match ty {
            RobotUnitType::Head => self.heads.get(idx).copied().unwrap_or_default(),
            RobotUnitType::Armor => self.armors.get(idx).copied().unwrap_or_default(),
            RobotUnitType::Weapon => self.weapons.get(idx).copied().unwrap_or_default(),
            RobotUnitType::Chassis => self.chassis.get(idx).copied().unwrap_or_default(),
            RobotUnitType::Empty => UnitPrice::zero(),
        }
    }

    /// Load `robots.dat`'s `Source/Price/*` block — the flat
    /// `HEAD{N}_{RES}` / `ARMOR{N}_{RES}` / `WEAPON{N}_{RES}` /
    /// `CHASSIS{N}_{RES}` keys at MatrixConfig.cpp:713-802.
    ///
    /// The C++ loader hardcodes every key; we iterate the block keys
    /// and match by prefix so the table keeps filling in if new
    /// components are added to the data without changing code.
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let source_rec = stor.block_record("da", "Source")?;
        let price_rec = stor.block_record(&source_rec, "Price")?;
        let keys = stor.get_buf(&price_rec, "0")?;
        let values = stor.get_buf(&price_rec, "1")?;
        let n = keys.arrays_count().min(values.arrays_count());

        let mut out = Self::default();
        for i in 0..n {
            let name = keys.get_as_wstr(i);
            let val = values.get_as_wstr(i);
            let Ok(v) = val.trim().parse::<i32>() else {
                continue;
            };
            let Some((prefix, num, res)) = parse_component_key(&name) else {
                continue;
            };
            // CBlockPar lookups in C++ are case-insensitive; the data
            // file may author keys as `head1_titan` / `Head1_Titan` /
            // `HEAD1_TITAN`. Compare without case so all variants match.
            let res_upper = res.to_ascii_uppercase();
            let res_idx = match res_upper.as_str() {
                "TITAN" => Resource::Titan as usize,
                "ELECTRONICS" => Resource::Electronics as usize,
                "ENERGY" => Resource::Energy as usize,
                "PLASM" | "PLASMA" => Resource::Plasma as usize,
                _ => continue,
            };
            let slot = (num - 1) as usize;
            let prefix_upper = prefix.to_ascii_uppercase();
            match prefix_upper.as_str() {
                "HEAD" if slot < ROBOT_HEAD_CNT => out.heads[slot].resources[res_idx] = v,
                "ARMOR" if slot < ROBOT_ARMOR_CNT => out.armors[slot].resources[res_idx] = v,
                "WEAPON" if slot < ROBOT_WEAPON_CNT => out.weapons[slot].resources[res_idx] = v,
                "CHASSIS" if slot < ROBOT_CHASSIS_CNT => out.chassis[slot].resources[res_idx] = v,
                _ => {}
            }
        }
        Some(out)
    }
}

/// Parse a key like `"HEAD2_TITAN"` → `("HEAD", 2, "TITAN")`.
fn parse_component_key(key: &str) -> Option<(&str, i32, &str)> {
    // Find the last `_` — everything left of it is `{PREFIX}{N}`, right
    // is the resource name.
    let (left, res) = key.rsplit_once('_')?;
    // Split letters / digits — the prefix is the maximal leading
    // alphabetic run, the rest is the number.
    let split = left.find(|c: char| c.is_ascii_digit())?;
    let (prefix, num_str) = left.split_at(split);
    let num: i32 = num_str.parse().ok()?;
    Some((prefix, num, res))
}

/// Port of the `SUnitChars` slices the constructor uses for per-
/// component stats. Mirrors the flat layout at MatrixConfig.hpp:
/// 436-502 (`EChars`). Only the rows `GetConstructionStructure`
/// needs are modelled here — ARMOR{N}_STRUCTURE (2 floats per entry)
/// + CHASSIS{N}_STRUCTURE (6 floats per entry, we only use the first).
#[derive(Debug, Clone, Copy)]
pub struct ItemCharsTable {
    pub armor_structure: [i32; ROBOT_ARMOR_CNT],
    pub armor_rotation_speed: [f32; ROBOT_ARMOR_CNT],
    pub chassis_structure: [i32; ROBOT_CHASSIS_CNT],
    /// `HIT_POINT_ADD` per head — from `Chars/Heads/{X}/HIT_POINT_ADD`.
    /// Indexed by `(kind as i32 - 1)` matching RUK_HEAD_*.
    pub head_hp_add: [i32; ROBOT_HEAD_CNT],
}

impl Default for ItemCharsTable {
    fn default() -> Self {
        Self {
            armor_structure: [0; ROBOT_ARMOR_CNT],
            armor_rotation_speed: [0.0; ROBOT_ARMOR_CNT],
            chassis_structure: [0; ROBOT_CHASSIS_CNT],
            head_hp_add: [0; ROBOT_HEAD_CNT],
        }
    }
}

impl ItemCharsTable {
    /// Load `Chars/*` — ARMOR{N}_STRUCTURE / CHASSIS{N}_STRUCTURE from
    /// the top-level `Chars/Armor` and `Chars/Chassis` sub-blocks, plus
    /// the per-head HIT_POINT_ADD bonus from `Chars/Heads/{S|D|L|F}`
    /// (CConstructor.cpp:877-894).
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let mut out = Self::default();
        let chars_rec = stor.block_record("da", "Chars")?;

        if let Some(armor_rec) = stor.block_record(&chars_rec, "Armor") {
            if let (Some(keys), Some(values)) =
                (stor.get_buf(&armor_rec, "0"), stor.get_buf(&armor_rec, "1"))
            {
                let n = keys.arrays_count().min(values.arrays_count());
                for i in 0..n {
                    let name = keys.get_as_wstr(i);
                    let val = values.get_as_wstr(i);
                    let Some(rest) = name.strip_prefix("ARMOR") else {
                        continue;
                    };
                    let Some((digit, field)) = rest.split_once('_') else {
                        continue;
                    };
                    let Ok(n) = digit.parse::<usize>() else {
                        continue;
                    };
                    if !(1..=ROBOT_ARMOR_CNT).contains(&n) {
                        continue;
                    }
                    match field {
                        "STRUCTURE" => {
                            if let Ok(v) = val.trim().parse::<i32>() {
                                out.armor_structure[n - 1] = v;
                            } else if let Ok(v) = val.trim().parse::<f32>() {
                                out.armor_structure[n - 1] = v as i32;
                            }
                        }
                        "ROTATION_SPEED" => {
                            if let Ok(v) = val.trim().parse::<f32>() {
                                out.armor_rotation_speed[n - 1] = v;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Chassis block — only STRUCTURE needed for now; ChassisChars
        // already handles the movement fields.
        if let Some(chassis_rec) = stor.block_record(&chars_rec, "Chassis") {
            if let (Some(keys), Some(values)) = (
                stor.get_buf(&chassis_rec, "0"),
                stor.get_buf(&chassis_rec, "1"),
            ) {
                let n = keys.arrays_count().min(values.arrays_count());
                for i in 0..n {
                    let name = keys.get_as_wstr(i);
                    let val = values.get_as_wstr(i);
                    let Some(rest) = name.strip_prefix("CHASSIS") else {
                        continue;
                    };
                    let Some((digit, field)) = rest.split_once('_') else {
                        continue;
                    };
                    let Ok(n) = digit.parse::<usize>() else {
                        continue;
                    };
                    if !(1..=ROBOT_CHASSIS_CNT).contains(&n) || field != "STRUCTURE" {
                        continue;
                    }
                    if let Ok(v) = val.trim().parse::<i32>() {
                        out.chassis_structure[n - 1] = v;
                    } else if let Ok(v) = val.trim().parse::<f32>() {
                        out.chassis_structure[n - 1] = v as i32;
                    }
                }
            }
        }

        // Heads/{S,D,L,F}/HIT_POINT_ADD — one sub-block per head kind
        // keyed by a 1-char ID (CConstructor.cpp:880-883).
        if let Some(heads_rec) = stor.block_record(&chars_rec, "Heads") {
            for (idx, ch) in ["S", "D", "L", "F"].iter().enumerate() {
                if let Some(sub) = stor.block_record(&heads_rec, ch) {
                    if let Some(s) = stor.block_param(&sub, "HIT_POINT_ADD") {
                        if let Ok(v) = s.trim().parse::<i32>() {
                            out.head_hp_add[idx] = v;
                        }
                    }
                }
            }
        }

        Some(out)
    }
}

/// Port of `g_Config.m_Timings[]` (MatrixConfig.cpp:649-658). All
/// values in ms; `0` means "not loaded" and the caller should fall
/// back to a reasonable default.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub resource_titan: i32,
    pub resource_electronics: i32,
    pub resource_energy: i32,
    pub resource_plasma: i32,
    pub resource_base: i32,
    pub unit_robot: i32,
    pub unit_flyer: i32,
    pub unit_turret: i32,
    pub maintenance_period: i32,
}

impl Timings {
    /// Port of the three-block `BlockGet(Timings)->BlockGet(...)->ParGet(...)`
    /// sequence in `CMatrixConfig::LoadConfig` (MatrixConfig.cpp:641-658).
    /// Note the nesting: param keys live under `Timings/Resources/*` and
    /// `Timings/Units/*`, not directly under `Timings`.
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let timings_rec = stor.block_record("da", "Timings")?;
        let resources_rec = stor.block_record(&timings_rec, "Resources")?;
        let units_rec = stor.block_record(&timings_rec, "Units")?;
        let read_at = |rec: &str, key: &str| -> i32 {
            stor.block_param(rec, key)
                .and_then(|s| s.trim().parse::<i32>().ok())
                .unwrap_or(0)
        };
        Some(Self {
            // Resources sub-block (MatrixConfig.cpp:648-653).
            resource_titan: read_at(&resources_rec, "RESOURCE_TITAN"),
            resource_electronics: read_at(&resources_rec, "RESOURCE_ELECTRONICS"),
            resource_energy: read_at(&resources_rec, "RESOURCE_ENERGY"),
            resource_plasma: read_at(&resources_rec, "RESOURCE_PLASMA"),
            resource_base: read_at(&resources_rec, "RESOURCE_BASE"),
            // Units sub-block (MatrixConfig.cpp:655-658).
            unit_robot: read_at(&units_rec, "UNIT_ROBOT"),
            unit_flyer: read_at(&units_rec, "UNIT_FLYER"),
            unit_turret: read_at(&units_rec, "UNIT_TURRET"),
            // Top-level Timings param (MatrixConfig.cpp:644).
            maintenance_period: read_at(&timings_rec, "MAINTENANCE"),
        })
    }
}

/// Port of `SCannonProps` (MatrixConfig.hpp:504-515) — per-turret
/// cost + stats. Indexed 0..3 for the four `RUK_TURRET_*` kinds.
#[derive(Debug, Clone, Copy, Default)]
pub struct CannonProps {
    pub resources: [i32; MAX_RESOURCES],
    pub strength: f32,
    pub hitpoint: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TurretProps {
    pub cannons: [CannonProps; 4],
}

impl TurretProps {
    /// Load `Models/Cannons/{0..3}` — mirrors the C++ loader at
    /// MatrixConfig.cpp:1052-1079, which walks
    /// `g_MatrixData->BlockGet("Models")->BlockGet("Cannons")` and
    /// reads each child's `Titan` / `Energy` / `Plasm` (sic) /
    /// `Electronics` / `Strength` / `Hitpoint`.
    ///
    /// Children are enumerated (not named `Cannon{N}`). Resource keys
    /// are capitalized, not uppercase. Missing block or params → zero.
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let mut out = TurretProps::default();
        let models_rec = stor.block_record("da", "Models")?;
        let cannons_rec = stor.block_record(&models_rec, "Cannons")?;
        let (names_buf, recs_buf) = (
            stor.get_buf(&cannons_rec, "2")?,
            stor.get_buf(&cannons_rec, "3")?,
        );
        let n = names_buf.arrays_count().min(recs_buf.arrays_count()).min(4);
        for i in 0..n {
            let rec = recs_buf.get_as_wstr(i);
            let read_i = |k: &str| -> i32 {
                stor.block_param(&rec, k)
                    .and_then(|s| s.trim().parse::<i32>().ok())
                    .unwrap_or(0)
            };
            let read_f = |k: &str| -> f32 {
                stor.block_param(&rec, k)
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.0)
            };
            // Resource key ordering is the C++ `Resource` enum:
            // TITAN=0, ELECTRONICS=1, ENERGY=2, PLASMA=3.
            // The data file uses `Plasm` (with no trailing `a`) —
            // preserve the typo to match the shipped assets.
            out.cannons[i] = CannonProps {
                resources: [
                    read_i("Titan"),
                    read_i("Electronics"),
                    read_i("Energy"),
                    read_i("Plasm"),
                ],
                strength: read_f("Strength"),
                hitpoint: read_f("Hitpoint"),
            };
        }
        Some(out)
    }

    pub fn cost_of(&self, kind: i32) -> UnitPrice {
        let i = ((kind - 1).max(0) as usize).min(3);
        UnitPrice {
            resources: self.cannons[i].resources,
        }
    }
}

/// Per-weapon damage table ported from `g_Config.m_RobotDamages[]`
/// (MatrixConfig.cpp:568-587). Indexed by `WeapKind2Index(kind)` —
/// see [`weap_kind_to_index`]. Weapons that aren't damage-table
/// entries (e.g. `RUK_UNKNOWN`) read as zero.
#[derive(Debug, Clone, Copy)]
pub struct RobotDamages {
    pub table: [WeaponDamage; crate::matrix_game::effects::weapon::WEAPON_COUNT],
}

impl Default for RobotDamages {
    fn default() -> Self {
        Self {
            table: [WeaponDamage::default(); crate::matrix_game::effects::weapon::WEAPON_COUNT],
        }
    }
}

impl RobotDamages {
    /// Load `Weapons/Damages/Robot` (MatrixConfig.cpp:568-587). Same
    /// shape as `ObjectDamages::from_matrix_data` — keys are
    /// `WEAPON_<NAME>` and values are `damage[,mindamage[,friend_damage]]`.
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let weapons_rec = stor.block_record("da", "Weapons")?;
        let damages_rec = stor.block_record(&weapons_rec, "Damages")?;
        let robot_rec = stor.block_record(&damages_rec, "Robot")?;
        let keys = stor.get_buf(&robot_rec, "0")?;
        let values = stor.get_buf(&robot_rec, "1")?;
        let n = keys.arrays_count().min(values.arrays_count());
        let mut out = Self::default();
        for i in 0..n {
            let name = keys.get_as_wstr(i);
            let Some(idx) = weap_name_to_index(&name) else {
                continue;
            };
            let val = values.get_as_wstr(i);
            let nn = wstr::count_par(&val, ",");
            let damage = wstr::int_par(&val, 0, ",");
            let mindamage = if nn > 1 {
                wstr::int_par(&val, 1, ",")
            } else {
                0
            };
            // MatrixConfig.cpp:584 — falls back to `damage` when absent.
            let friend_damage = if nn > 2 {
                wstr::int_par(&val, 2, ",")
            } else {
                damage
            };
            out.table[idx] = WeaponDamage {
                damage,
                mindamage,
                friend_damage,
            };
        }
        Some(out)
    }

    /// Look up damage for a robot-side weapon kind. `kind` is the
    /// `RUK_WEAPON_*` discriminant (1..=10). Returns 0 for unknown.
    pub fn for_kind(&self, kind: RobotUnitKind) -> WeaponDamage {
        if let Some(idx) = weap_kind_to_index(kind) {
            self.table[idx]
        } else {
            WeaponDamage::default()
        }
    }
}

/// Per-weapon cooldown table ported from `g_Config.m_WeaponCooldown[]`
/// (MatrixConfig.cpp:625-639). Values in milliseconds — minimum delay
/// between shots for that weapon kind.
#[derive(Debug, Clone, Copy)]
pub struct WeaponCooldown {
    pub table: [i32; crate::matrix_game::effects::weapon::WEAPON_COUNT],
}

impl Default for WeaponCooldown {
    fn default() -> Self {
        Self {
            table: [0; crate::matrix_game::effects::weapon::WEAPON_COUNT],
        }
    }
}

impl WeaponCooldown {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let weapons_rec = stor.block_record("da", "Weapons")?;
        let cooldown_rec = stor.block_record(&weapons_rec, "Cooldown")?;
        let keys = stor.get_buf(&cooldown_rec, "0")?;
        let values = stor.get_buf(&cooldown_rec, "1")?;
        let n = keys.arrays_count().min(values.arrays_count());
        let mut out = Self::default();
        for i in 0..n {
            let name = keys.get_as_wstr(i);
            let Some(idx) = weap_name_to_index(&name) else {
                continue;
            };
            let val = values.get_as_wstr(i);
            out.table[idx] = val.trim().parse::<i32>().unwrap_or(0);
        }
        Some(out)
    }

    pub fn for_kind(&self, kind: RobotUnitKind) -> i32 {
        if let Some(idx) = weap_kind_to_index(kind) {
            self.table[idx]
        } else {
            0
        }
    }
}

/// Port of `WeapKind2Index` (MatrixConfig.hpp:544). Maps the robot-
/// constructor weapon kind enum (RUK_WEAPON_*) into the dense damage-
/// table index. Returns `None` for kinds that don't carry a damage row
/// (RUK_UNKNOWN, etc.).
pub fn weap_kind_to_index(kind: RobotUnitKind) -> Option<usize> {
    Some(match kind.0 {
        // RUK_WEAPON_PLASMA = 8 → 0
        8 => 0,
        // RUK_WEAPON_MACHINEGUN = 1 → 1
        1 => 1,
        // RUK_WEAPON_MISSILE = 3 → 2
        3 => 2,
        // RUK_WEAPON_MORTAR = 5 → 3
        5 => 3,
        // RUK_WEAPON_FLAMETHROWER = 4 → 4
        4 => 4,
        // RUK_WEAPON_BOMB = 7 → 5
        7 => 5,
        // RUK_WEAPON_ELECTRIC = 9 → 6
        9 => 6,
        // RUK_WEAPON_LASER = 6 → 7
        6 => 7,
        // RUK_WEAPON_CANNON = 2 → 8
        2 => 8,
        // RUK_WEAPON_REPAIR = 10 → 12
        10 => 12,
        _ => return None,
    })
}

/// Per-weapon AI-strength weights ported from `g_Config.m_WeaponStrengthAI`
/// (MatrixConfig.cpp:923-934). Indexed by `RUK_WEAPON_*` (1-based).
/// `[0]` is unused. Drives `SSpecialBot::CalcStrength` (CConstructor.cpp:1499).
#[derive(Debug, Clone, Copy, Default)]
pub struct WeaponStrengthAI {
    pub table: [f32; 11],
}

impl WeaponStrengthAI {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let ai_rec = stor.block_record("da", "AIStrange")?;
        let weapon_rec = stor.block_record(&ai_rec, "Weapon")?;
        let read = |k: &str| -> f32 {
            stor.block_param(&weapon_rec, k)
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0)
        };
        let mut out = Self::default();
        out.table[RobotUnitKind::WEAPON_MACHINEGUN.0 as usize] = read("Machinegun");
        out.table[RobotUnitKind::WEAPON_CANNON.0 as usize] = read("Cannon");
        out.table[RobotUnitKind::WEAPON_MISSILE.0 as usize] = read("Missile");
        out.table[RobotUnitKind::WEAPON_FLAMETHROWER.0 as usize] = read("Flamethrower");
        out.table[RobotUnitKind::WEAPON_MORTAR.0 as usize] = read("Mortar");
        out.table[RobotUnitKind::WEAPON_LASER.0 as usize] = read("Laser");
        out.table[RobotUnitKind::WEAPON_BOMB.0 as usize] = read("Bomb");
        out.table[RobotUnitKind::WEAPON_PLASMA.0 as usize] = read("Plasma");
        out.table[RobotUnitKind::WEAPON_ELECTRIC.0 as usize] = read("Electric");
        out.table[RobotUnitKind::WEAPON_REPAIR.0 as usize] = read("Repair");
        Some(out)
    }

    pub fn for_kind(&self, kind: RobotUnitKind) -> f32 {
        let i = kind.0 as usize;
        if i < self.table.len() {
            self.table[i]
        } else {
            0.0
        }
    }
}

/// Per-head modifiers loaded from `Chars/Heads/{S,D,L,F}` blocks
/// (MatrixConfig.cpp + CConstructor.cpp:1138-1169 reads them via
/// MakeItemReplacements). Values are raw doubles read from the params;
/// CConstructor scales them via the `<…>` template substitutions
/// (e.g. HIT_POINT_ADD ×0.1, COOL_DOWN ×-1, OVERHEAT ×-1).
///
/// Indexed by `kind.as_index()` matching RUK_HEAD_*.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeadCharsTable {
    /// `HIT_POINT_ADD` raw value (already loaded into ItemCharsTable.head_hp_add
    /// for structure math; mirrored here for label-replacement).
    pub hit_point_add: [f32; ROBOT_HEAD_CNT],
    /// `COOL_DOWN` raw value (percentage). The C++ scales by /100
    /// before applying.
    pub cool_down: [f32; ROBOT_HEAD_CNT],
    /// `OVERHEAT` raw value (percentage).
    pub overheat: [f32; ROBOT_HEAD_CNT],
    /// `CHASSIS_SPEED` raw value (percentage).
    pub chassis_speed: [f32; ROBOT_HEAD_CNT],
    /// `FIRE_DISTANCE` raw value.
    pub fire_distance: [f32; ROBOT_HEAD_CNT],
    /// `RADAR_DISTANCE` raw value.
    pub radar_distance: [f32; ROBOT_HEAD_CNT],
    /// `LIGHT_PROTECT` (rendered as `<Max>`).
    pub light_protect: [f32; ROBOT_HEAD_CNT],
    /// `AIM_PROTECT` (rendered as `<LessHit>`).
    pub aim_protect: [f32; ROBOT_HEAD_CNT],
    /// `BOMB_PROTECT` (rendered as `<BombProtect>`).
    pub bomb_protect: [f32; ROBOT_HEAD_CNT],
}

impl HeadCharsTable {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let chars_rec = stor.block_record("da", "Chars")?;
        let heads_rec = stor.block_record(&chars_rec, "Heads")?;
        let mut out = Self::default();
        // CConstructor.cpp:880-883 maps head kind → block letter.
        for (idx, ch) in ["S", "D", "L", "F"].iter().enumerate() {
            let Some(sub) = stor.block_record(&heads_rec, ch) else {
                continue;
            };
            let read = |k: &str| -> f32 {
                stor.block_param(&sub, k)
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.0)
            };
            out.hit_point_add[idx] = read("HIT_POINT_ADD");
            out.cool_down[idx] = read("COOL_DOWN");
            out.overheat[idx] = read("OVERHEAT");
            out.chassis_speed[idx] = read("CHASSIS_SPEED");
            out.fire_distance[idx] = read("FIRE_DISTANCE");
            out.radar_distance[idx] = read("RADAR_DISTANCE");
            out.light_protect[idx] = read("LIGHT_PROTECT");
            out.aim_protect[idx] = read("AIM_PROTECT");
            out.bomb_protect[idx] = read("BOMB_PROTECT");
        }
        Some(out)
    }
}

/// Process-wide config — ports `g_Config` (MatrixConfig.hpp). Only the
/// subsets ported so far are exposed here. `World::load_config` seeds
/// it; reading before that yields all-default tables.
///
/// Global singleton rather than threaded everywhere because `g_Config`
/// is read from `CMatrixRobotAI::logic_takt` + sibling tick paths that
/// can't easily carry an extra param through the `MapStatic` trait.
static GLOBAL_CONFIG: std::sync::OnceLock<std::sync::RwLock<GlobalConfig>> =
    std::sync::OnceLock::new();

#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalConfig {
    pub chassis: ChassisChars,
    pub prices: PriceTable,
    pub item_chars: ItemCharsTable,
    pub timings: Timings,
    pub turrets: TurretProps,
    pub robot_damages: RobotDamages,
    pub weapon_cooldown: WeaponCooldown,
    pub weapon_strength_ai: WeaponStrengthAI,
    pub head_chars: HeadCharsTable,
}

pub fn global() -> GlobalConfig {
    *GLOBAL_CONFIG
        .get_or_init(|| std::sync::RwLock::new(GlobalConfig::default()))
        .read()
        .unwrap()
}

pub fn set_global(cfg: GlobalConfig) {
    let slot = GLOBAL_CONFIG.get_or_init(|| std::sync::RwLock::new(GlobalConfig::default()));
    *slot.write().unwrap() = cfg;
}

// ── Labels / Descriptions / RobotNames ─────────────────────────────
// String-heavy tables ported separately so `GlobalConfig` stays Copy.
// Mirrors `g_Config.m_Labels[]` / `m_Descriptions[]` (MatrixConfig.cpp:
// 938-1008) and the `AllLabels/RobotNames` block CConstructor.cpp:1544
// reads for `GetConstructionName`.

/// Port of `g_Config.m_Labels[]` (MatrixConfig.cpp:941-972). One label
/// string per item kind keyed via [`label_index_*`] helpers. Loaded
/// from `da/AllLabels/Items` block.
#[derive(Debug, Clone, Default)]
pub struct ItemLabels {
    /// W{1..10}_CHAR — weapons.
    pub weapon: [String; ROBOT_WEAPON_CNT],
    /// HU{1..6}_CHAR — armors.
    pub armor: [String; ROBOT_ARMOR_CNT],
    /// HE{1..4}_CHAR — heads.
    pub head: [String; ROBOT_HEAD_CNT],
    /// CH{1..5}_CHAR — chassis.
    pub chassis: [String; ROBOT_CHASSIS_CNT],
}

impl ItemLabels {
    /// Load the `AllLabels/Items` block — keys like `w1_char`,
    /// `hu3_char`, `he2_char`, `ch4_char`. Missing keys read as empty
    /// string.
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        Self::from_block(stor, "Items")
    }

    fn from_block(stor: &Storage, suffix: &str) -> Option<Self> {
        let labels_rec = stor.block_record("da", "AllLabels")?;
        let items_rec = stor.block_record(&labels_rec, suffix)?;
        let mut out = Self::default();
        for i in 0..ROBOT_WEAPON_CNT {
            let key = format!("w{}_char", i + 1);
            let key_alt = format!("w{}_descr", i + 1);
            if let Some(s) = stor
                .block_param(&items_rec, &key)
                .or_else(|| stor.block_param(&items_rec, &key_alt))
            {
                out.weapon[i] = s;
            }
        }
        for i in 0..ROBOT_ARMOR_CNT {
            let key = format!("hu{}_char", i + 1);
            let key_alt = format!("hu{}_descr", i + 1);
            if let Some(s) = stor
                .block_param(&items_rec, &key)
                .or_else(|| stor.block_param(&items_rec, &key_alt))
            {
                out.armor[i] = s;
            }
        }
        for i in 0..ROBOT_HEAD_CNT {
            let key = format!("he{}_char", i + 1);
            let key_alt = format!("he{}_descr", i + 1);
            if let Some(s) = stor
                .block_param(&items_rec, &key)
                .or_else(|| stor.block_param(&items_rec, &key_alt))
            {
                out.head[i] = s;
            }
        }
        for i in 0..ROBOT_CHASSIS_CNT {
            let key = format!("ch{}_char", i + 1);
            let key_alt = format!("ch{}_descr", i + 1);
            if let Some(s) = stor
                .block_param(&items_rec, &key)
                .or_else(|| stor.block_param(&items_rec, &key_alt))
            {
                out.chassis[i] = s;
            }
        }
        Some(out)
    }

    /// Look up the label for a (type, kind) pair. Returns empty for
    /// out-of-range or unknown.
    pub fn get(&self, ty: RobotUnitType, kind: RobotUnitKind) -> &str {
        if kind.is_empty() {
            return "";
        }
        let idx = kind.as_index();
        match ty {
            RobotUnitType::Weapon => self.weapon.get(idx).map(|s| s.as_str()).unwrap_or(""),
            RobotUnitType::Armor => self.armor.get(idx).map(|s| s.as_str()).unwrap_or(""),
            RobotUnitType::Head => self.head.get(idx).map(|s| s.as_str()).unwrap_or(""),
            RobotUnitType::Chassis => self.chassis.get(idx).map(|s| s.as_str()).unwrap_or(""),
            RobotUnitType::Empty => "",
        }
    }
}

/// Port of `g_Config.m_Descriptions[]` (MatrixConfig.cpp:977-1008).
/// Same shape as `ItemLabels` but loaded from the `Descriptions` sub-
/// block.
#[derive(Debug, Clone, Default)]
pub struct ItemDescriptions {
    pub inner: ItemLabels,
}

impl ItemDescriptions {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        Some(Self {
            inner: ItemLabels::from_block(stor, "Descriptions")?,
        })
    }

    pub fn get(&self, ty: RobotUnitType, kind: RobotUnitKind) -> &str {
        self.inner.get(ty, kind)
    }
}

/// Port of the `AllLabels/RobotNames` block CConstructor.cpp:1544 reads
/// for `GetConstructionName`. Each kind has its own `Hull{N}Key`,
/// `Chas{N}Key`, `Head{N}Key` string fragment that the constructor
/// concatenates to make the robot's display name.
#[derive(Debug, Clone, Default)]
pub struct RobotNameParts {
    pub hull: [String; ROBOT_ARMOR_CNT],
    pub chassis: [String; ROBOT_CHASSIS_CNT],
    pub head: [String; ROBOT_HEAD_CNT],
}

impl RobotNameParts {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let labels_rec = stor.block_record("da", "AllLabels")?;
        let names_rec = stor.block_record(&labels_rec, "RobotNames")?;
        let mut out = Self::default();
        for i in 0..ROBOT_ARMOR_CNT {
            let key = format!("Hull{}Key", i + 1);
            if let Some(s) = stor.block_param(&names_rec, &key) {
                out.hull[i] = s;
            }
        }
        for i in 0..ROBOT_CHASSIS_CNT {
            let key = format!("Chas{}Key", i + 1);
            if let Some(s) = stor.block_param(&names_rec, &key) {
                out.chassis[i] = s;
            }
        }
        for i in 0..ROBOT_HEAD_CNT {
            let key = format!("Head{}Key", i + 1);
            if let Some(s) = stor.block_param(&names_rec, &key) {
                out.head[i] = s;
            }
        }
        Some(out)
    }
}

/// Port of `g_MatrixData->BlockGet(IF_LABELS_BLOCKPAR)->BlockGet(L"Buildings")`
/// — the localised per-kind name/description strings the Main panel's
/// `name` + `bopis` + `bresg` captions read each frame. Keys:
/// `Base_Name`, `Base_Descr`, `Titan_Name`, `Titan_Descr`,
/// `Electronics_Name`, `Electronics_Descr`, `Energy_Name`,
/// `Energy_Descr`, `Plasma_Name`, `Plasma_Descr`, `ResPer` (the
/// `"<resources> ресурсов в минуту"` income template,
/// CInterface.cpp:1485).
#[derive(Debug, Clone, Default)]
pub struct BuildingLabels {
    pub base_name: String,
    pub base_descr: String,
    pub titan_name: String,
    pub titan_descr: String,
    pub electronics_name: String,
    pub electronics_descr: String,
    pub energy_name: String,
    pub energy_descr: String,
    pub plasma_name: String,
    pub plasma_descr: String,
    /// `ResPer` — income-per-minute template, e.g.
    /// `"<resources> ресурсов в минуту"`. The `<resources>` placeholder
    /// is replaced with a gold-coloured rich-text run containing the
    /// income count (CInterface.cpp:1485-1486).
    pub res_per: String,
}

impl BuildingLabels {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let labels_rec = stor.block_record("da", "AllLabels")?;
        let b_rec = stor.block_record(&labels_rec, "Buildings")?;
        let get = |key: &str| -> String { stor.block_param(&b_rec, key).unwrap_or_default() };
        Some(Self {
            base_name: get("Base_Name"),
            base_descr: get("Base_Descr"),
            titan_name: get("Titan_Name"),
            titan_descr: get("Titan_Descr"),
            electronics_name: get("Electronics_Name"),
            electronics_descr: get("Electronics_Descr"),
            energy_name: get("Energy_Name"),
            energy_descr: get("Energy_Descr"),
            plasma_name: get("Plasma_Name"),
            plasma_descr: get("Plasma_Descr"),
            res_per: get("ResPer"),
        })
    }
}

/// Bundle of the string-heavy tables. Held behind a separate static so
/// `GlobalConfig` can stay `Copy`.
#[derive(Debug, Clone, Default)]
pub struct StringTables {
    pub labels: ItemLabels,
    pub descriptions: ItemDescriptions,
    pub robot_names: RobotNameParts,
    pub buildings: BuildingLabels,
}

static GLOBAL_STRINGS: std::sync::OnceLock<std::sync::RwLock<std::sync::Arc<StringTables>>> =
    std::sync::OnceLock::new();

pub fn global_strings() -> std::sync::Arc<StringTables> {
    GLOBAL_STRINGS
        .get_or_init(|| std::sync::RwLock::new(std::sync::Arc::new(StringTables::default())))
        .read()
        .unwrap()
        .clone()
}

pub fn set_global_strings(t: StringTables) {
    let slot = GLOBAL_STRINGS
        .get_or_init(|| std::sync::RwLock::new(std::sync::Arc::new(StringTables::default())));
    *slot.write().unwrap() = std::sync::Arc::new(t);
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
        d.table[5] = WeaponDamage {
            damage: 400,
            mindamage: 100,
            friend_damage: 0,
        }; // BIGBOOM
        assert_eq!(
            d.get(WEAPON_BIGBOOM),
            Some(WeaponDamage {
                damage: 400,
                mindamage: 100,
                friend_damage: 0
            }),
        );
        d.table[0] = WeaponDamage {
            damage: 80,
            mindamage: 40,
            friend_damage: 0,
        }; // PLASMA
        assert_eq!(
            d.get(WEAPON_PLASMA),
            Some(WeaponDamage {
                damage: 80,
                mindamage: 40,
                friend_damage: 0
            }),
        );
        // `WEAPON_NONE` has no table slot.
        assert_eq!(d.get(WEAPON_NONE), None);
    }
}

// ── Resource / robot-unit-kind enums (MatrixConfig.hpp) ─────────────────

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

/// Port of `ERobotUnitKind` (MatrixConfig.hpp:34-78). All four categories
/// (chassis, armor, weapon, head) share one discriminant space; `UNKNOWN=0`
/// is the sentinel. We keep the plain i32 representation so the C++
/// arithmetic (`kind + 1`, etc.) in the constructor's wrap-around click
/// handler ports directly.
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
        (self.0 - 1).max(0) as usize
    }
}

/// Counts per category — mirror the `ROBOT_*_CNT` discriminants.
pub const ROBOT_CHASSIS_CNT: usize = 5;
pub const ROBOT_WEAPON_CNT: usize = 10;
pub const ROBOT_ARMOR_CNT: usize = 6;
pub const ROBOT_HEAD_CNT: usize = 4;

// ── Build-time accessors (ports from MatrixConfig.cpp) ──────────────────

const UNIT_ROBOT_BUILD_TIME_MS: i32 = 5000;

/// Per-kind robot build time in ms, resolved from `g_Config.m_Timings`
/// with a fall-back to `UNIT_ROBOT_BUILD_TIME_MS` when the config
/// hasn't been loaded yet.
pub fn robot_build_time_ms() -> i32 {
    let t = global().timings.unit_robot;
    if t > 0 {
        t
    } else {
        UNIT_ROBOT_BUILD_TIME_MS
    }
}

/// Turret build time (UNIT_TURRET in MatrixConfig.cpp:658).
pub fn turret_build_time_ms() -> i32 {
    let t = global().timings.unit_turret;
    if t > 0 {
        t
    } else {
        UNIT_ROBOT_BUILD_TIME_MS
    }
}
