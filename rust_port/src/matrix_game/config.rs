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

use crate::matrix_game::effects::weapon::{weap_name_to_index, weapon_for_index, WEAPON_COUNT};
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

    /// Load `robots.dat`'s `Price/{Head,Armor,Weapon,Chassis}` blocks
    /// — the flat `HEAD{N}_{RES}` / `ARMOR{N}_{RES}` / etc. keys read
    /// at MatrixConfig.cpp:711-833. The C++ structure is:
    ///
    ///   Price/
    ///     Head/    HEAD1_TITAN, HEAD1_ELECTRONICS, …
    ///     Armor/   ARMOR1_TITAN, …
    ///     Weapon/  WEAPON1_TITAN, …
    ///     Chassis/ CHASSIS1_TITAN, …
    ///
    /// Each sub-block is its own BlockPar. We walk all four and parse
    /// every `{PREFIX}{N}_{RES}` key the same way the C++ does (case-
    /// insensitive `ParGet`).
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let price_rec = stor.block_record("da", "Price")?;
        let mut out = Self::default();
        for sub in ["Head", "Armor", "Weapon", "Chassis"] {
            let Some(sub_rec) = stor.block_record(&price_rec, sub) else {
                continue;
            };
            let Some(keys) = stor.get_buf(&sub_rec, "0") else {
                continue;
            };
            let Some(values) = stor.get_buf(&sub_rec, "1") else {
                continue;
            };
            let n = keys.arrays_count().min(values.arrays_count());
            for i in 0..n {
                let name = keys.get_as_wstr(i);
                let val = values.get_as_wstr(i);
                let Ok(v) = val.trim().parse::<i32>() else {
                    continue;
                };
                let Some((prefix, num, res)) = parse_component_key(&name) else {
                    continue;
                };
                // CBlockPar lookups in C++ are case-insensitive; data
                // may author keys as `head1_titan` / `Head1_Titan` /
                // `HEAD1_TITAN`. Compare uppercased.
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
                    "CHASSIS" if slot < ROBOT_CHASSIS_CNT => {
                        out.chassis[slot].resources[res_idx] = v
                    }
                    _ => {}
                }
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
    /// `m_CaptureTimeErase` / `m_CaptureTimePaint` / `m_CaptureTimeRolback`
    /// (MatrixConfig.cpp:660-663, loaded from `Timings/Capture/*`).
    /// Defaults 750 / 500 / 1500 (MatrixConfig.cpp:322-324).
    pub capture_time_erase: i32,
    pub capture_time_paint: i32,
    pub capture_time_rollback: i32,
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
            // Capture sub-block (MatrixConfig.cpp:660-663).
            capture_time_erase: stor
                .block_record(&timings_rec, "Capture")
                .map(|rec| read_at(&rec, "ERASE"))
                .filter(|v| *v > 0)
                .unwrap_or(750),
            capture_time_paint: stor
                .block_record(&timings_rec, "Capture")
                .map(|rec| read_at(&rec, "PAINT"))
                .filter(|v| *v > 0)
                .unwrap_or(500),
            capture_time_rollback: stor
                .block_record(&timings_rec, "Capture")
                .map(|rec| read_at(&rec, "ROLLBACK"))
                .filter(|v| *v > 0)
                .unwrap_or(1500),
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
    /// `max_top_angle` / `max_bottom_angle` — barrel elevation clamps,
    /// radians (loaded as degrees, MatrixConfig.cpp:1060-1061).
    pub max_top_angle: f32,
    pub max_bottom_angle: f32,
    /// `weapon` — the `EWeapon` this turret fires (WEAPON_CANNON0..3).
    pub weapon: crate::matrix_game::effects::weapon::Weapon,
    /// `max_da` — per-takt rotation step in radians; 0 = smooth
    /// exponential aiming (MatrixConfig.hpp:509).
    pub max_da: f32,
    /// `seek_radius` — target-acquisition radius.
    pub seek_radius: f32,
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
            let grad2rad = |deg: f32| deg * std::f32::consts::PI / 180.0;
            // `WeapName2Weap` on the `Weapon` key (MatrixConfig.cpp:1064).
            let weapon = stor
                .block_param(&rec, "Weapon")
                .and_then(|s| weap_name_to_index(s.trim()))
                .map(weapon_for_index)
                .unwrap_or(crate::matrix_game::effects::weapon::WEAPON_NONE);
            out.cannons[i] = CannonProps {
                resources: [
                    read_i("Titan"),
                    read_i("Electronics"),
                    read_i("Energy"),
                    read_i("Plasm"),
                ],
                strength: read_f("Strength"),
                hitpoint: read_f("Hitpoint"),
                max_top_angle: grad2rad(read_f("MaxTopAngle")),
                max_bottom_angle: grad2rad(read_f("MaxBottomAngle")),
                weapon,
                max_da: grad2rad(read_f("Rotation")),
                seek_radius: read_f("Radius"),
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

/// Shared loader for the `Weapons/Damages/<block>` tables that follow
/// the same `damage[,mindamage[,friend_damage]]` shape (MatrixConfig.cpp
/// :487-587). `friend_damage` falls back to `damage` when the third
/// column is absent — true for the Cannon / Building / Flyer / Robot
/// tables (the Object table never reads it; it keeps its own loader).
fn load_damage_table(stor: &Storage, block: &str) -> Option<[WeaponDamage; WEAPON_COUNT]> {
    let weapons_rec = stor.block_record("da", "Weapons")?;
    let damages_rec = stor.block_record(&weapons_rec, "Damages")?;
    let rec = stor.block_record(&damages_rec, block)?;
    let keys = stor.get_buf(&rec, "0")?;
    let values = stor.get_buf(&rec, "1")?;
    let n = keys.arrays_count().min(values.arrays_count());
    let mut table = [WeaponDamage::default(); WEAPON_COUNT];
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
        let friend_damage = if nn > 2 {
            wstr::int_par(&val, 2, ",")
        } else {
            damage
        };
        table[idx] = WeaponDamage {
            damage,
            mindamage,
            friend_damage,
        };
    }
    Some(table)
}

/// Port of `g_Config.m_CannonDamages[]` (MatrixConfig.cpp:487-504) —
/// damage received by turrets, indexed by `weap_to_index`.
#[derive(Debug, Clone, Copy)]
pub struct CannonDamages {
    pub table: [WeaponDamage; WEAPON_COUNT],
}

impl Default for CannonDamages {
    fn default() -> Self {
        Self {
            table: [WeaponDamage::default(); WEAPON_COUNT],
        }
    }
}

impl CannonDamages {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        Some(Self {
            table: load_damage_table(stor, "Cannon")?,
        })
    }

    pub fn get(&self, weap: crate::matrix_game::effects::weapon::Weapon) -> Option<WeaponDamage> {
        let idx = crate::matrix_game::effects::weapon::weap_to_index(weap)?;
        Some(self.table[idx])
    }
}

/// Port of `g_Config.m_FlyerDamages[]` (MatrixConfig.cpp:538-566).
/// Flyers themselves aren't ported yet, but the table is loaded so the
/// damage subsystem is complete.
#[derive(Debug, Clone, Copy)]
pub struct FlyerDamages {
    pub table: [WeaponDamage; WEAPON_COUNT],
}

impl Default for FlyerDamages {
    fn default() -> Self {
        Self {
            table: [WeaponDamage::default(); WEAPON_COUNT],
        }
    }
}

impl FlyerDamages {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        Some(Self {
            table: load_damage_table(stor, "Flyer")?,
        })
    }

    pub fn get(&self, weap: crate::matrix_game::effects::weapon::Weapon) -> Option<WeaponDamage> {
        let idx = crate::matrix_game::effects::weapon::weap_to_index(weap)?;
        Some(self.table[idx])
    }
}

/// Port of `g_Config.m_WeaponRadius[]` (MatrixConfig.cpp:609-623) —
/// per-weapon range, loaded from `Weapons/Radius`. This is the
/// `m_WeaponDist` a `CMatrixEffectWeapon` is born with.
#[derive(Debug, Clone, Copy)]
pub struct WeaponRadius {
    pub table: [f32; WEAPON_COUNT],
}

impl Default for WeaponRadius {
    fn default() -> Self {
        Self {
            table: [0.0; WEAPON_COUNT],
        }
    }
}

impl WeaponRadius {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let weapons_rec = stor.block_record("da", "Weapons")?;
        let radius_rec = stor.block_record(&weapons_rec, "Radius")?;
        let keys = stor.get_buf(&radius_rec, "0")?;
        let values = stor.get_buf(&radius_rec, "1")?;
        let n = keys.arrays_count().min(values.arrays_count());
        let mut out = Self::default();
        for i in 0..n {
            let name = keys.get_as_wstr(i);
            let Some(idx) = weap_name_to_index(&name) else {
                continue;
            };
            let val = values.get_as_wstr(i);
            out.table[idx] = val.trim().parse::<f32>().unwrap_or(0.0);
        }
        Some(out)
    }

    /// `g_Config.m_WeaponRadius[Weap2Index(w)]` — 0.0 for weapons with
    /// no table slot.
    pub fn get(&self, weap: crate::matrix_game::effects::weapon::Weapon) -> f32 {
        crate::matrix_game::effects::weapon::weap_to_index(weap)
            .map(|i| self.table[i])
            .unwrap_or(0.0)
    }
}

/// One weapon class's overheat numbers out of `g_Config.m_Overheat[]`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OverheatParams {
    /// `WEAPON_*_HEAT` — heat added per shot (`SBotWeapon::m_HeatMod`).
    pub heat_mod: i32,
    /// `WCP_*` — cooling period in ms (`m_CoolDownPeriod` threshold).
    pub cool_period: i32,
    /// `WEAPON_*_COOL` — heat removed per elapsed period
    /// (`SBotWeapon::m_CoolDownMod`).
    pub cool_mod: i32,
}

/// Port of `g_Config.m_Overheat[OVERHEAT_LAST]` (MatrixConfig.hpp:637,
/// loaded at MatrixConfig.cpp:666-706). Only the 8 robot weapon classes
/// have entries; BIGBOOM and REPAIR get hardcoded 10/10 heat/cool mods
/// in `RobotWeaponInit` and no cooling period.
#[derive(Debug, Clone, Copy, Default)]
pub struct Overheat {
    pub volcano: OverheatParams,
    pub plasma: OverheatParams,
    pub laser: OverheatParams,
    pub homing_missile: OverheatParams,
    pub flamethrower: OverheatParams,
    pub bomb: OverheatParams,
    pub gun: OverheatParams,
    pub lightening: OverheatParams,
}

impl Overheat {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let weapons_rec = stor.block_record("da", "Weapons")?;
        let rec = stor.block_record(&weapons_rec, "Overheat")?;
        let read = |k: &str| -> i32 {
            stor.block_param(&rec, k)
                .and_then(|s| s.trim().parse::<i32>().ok())
                .unwrap_or(0)
        };
        let load = |name: &str| -> OverheatParams {
            OverheatParams {
                heat_mod: read(&format!("WEAPON_{name}_HEAT")),
                cool_period: read(&format!("WCP_{name}")),
                cool_mod: read(&format!("WEAPON_{name}_COOL")),
            }
        };
        Some(Self {
            volcano: load("VOLCANO"),
            plasma: load("PLASMA"),
            laser: load("LASER"),
            homing_missile: load("HOMING_MISSILE"),
            flamethrower: load("FLAMETHROWER"),
            bomb: load("BOMB"),
            gun: load("GUN"),
            lightening: load("LIGHTENING"),
        })
    }

    /// Entry for a weapon discriminant — covers exactly the classes the
    /// robot fire-control switch reads (MatrixRobot.cpp:1510-1535).
    /// `None` for BIGBOOM / REPAIR / cannon weapons.
    pub fn for_weapon(
        &self,
        weap: crate::matrix_game::effects::weapon::Weapon,
    ) -> Option<OverheatParams> {
        use crate::matrix_game::effects::weapon as w;
        Some(match weap {
            w::WEAPON_VOLCANO => self.volcano,
            w::WEAPON_PLASMA => self.plasma,
            w::WEAPON_LASER => self.laser,
            w::WEAPON_HOMING_MISSILE => self.homing_missile,
            w::WEAPON_FLAMETHROWER => self.flamethrower,
            w::WEAPON_BOMB => self.bomb,
            w::WEAPON_GUN => self.gun,
            w::WEAPON_LIGHTENING => self.lightening,
            _ => return None,
        })
    }
}

/// Port of `SDifficulty` (MatrixMap.hpp:182-187) + the loader in
/// `CMatrixMap::CMatrixMap` (MatrixMap.cpp:116-145): defaults are 1.0;
/// `Replaces/_difficulty` names a key in the `Difficulty` block whose
/// value is `"<dmg>,<maint>,<ff>"` percentages applied as
/// `1.0 + v/100`, clamped to ≥ 0.01.
#[derive(Debug, Clone, Copy)]
pub struct Difficulty {
    pub k_damage_enemy_to_player: f32,
    pub k_time_before_maintenance: f32,
    pub k_friendly_fire: f32,
}

impl Default for Difficulty {
    fn default() -> Self {
        Self {
            k_damage_enemy_to_player: 1.0,
            k_time_before_maintenance: 1.0,
            k_friendly_fire: 1.0,
        }
    }
}

impl Difficulty {
    pub fn from_matrix_data(stor: &Storage) -> Self {
        let mut out = Self::default();
        let Some(repl_rec) = stor.block_record("da", "Replaces") else {
            return out;
        };
        let Some(level) = stor.block_param(&repl_rec, "_difficulty") else {
            return out;
        };
        if level.is_empty() {
            return out;
        }
        let Some(dif_rec) = stor.block_record("da", "Difficulty") else {
            return out;
        };
        let Some(t2) = stor.block_param(&dif_rec, &level) else {
            return out;
        };
        if t2.is_empty() {
            return out;
        }
        let par = |i: usize| -> f32 { 1.0 + wstr::double_par(&t2, i, ",") as f32 / 100.0 };
        out.k_damage_enemy_to_player = par(0).max(0.01);
        out.k_time_before_maintenance = par(1).max(0.01);
        out.k_friendly_fire = par(2).max(0.01);
        out
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
    pub cannon_damages: CannonDamages,
    pub flyer_damages: FlyerDamages,
    pub weapon_radius: WeaponRadius,
    pub weapon_cooldown: WeaponCooldown,
    pub weapon_strength_ai: WeaponStrengthAI,
    pub overheat: Overheat,
    pub head_chars: HeadCharsTable,
    /// Per-map difficulty coefficients — `g_MatrixMap->m_Difficulty` in
    /// the C++; kept here because it loads from the same robots.dat.
    pub difficulty: Difficulty,
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
        // The data authors many of these with a trailing tab as a
        // visual delimiter (e.g. `Hull1Key = "б\t"`). The C++
        // concatenates them verbatim and the Rangers font treats the
        // tab as zero-width, but our AFT renderer skips unknown
        // glyphs without any fallback, so a trailing tab leaves a
        // visible gap. Strip the trailing whitespace so the rendered
        // robot name stays compact.
        let trim = |s: String| -> String { s.trim_end().to_string() };
        for i in 0..ROBOT_ARMOR_CNT {
            let key = format!("Hull{}Key", i + 1);
            if let Some(s) = stor.block_param(&names_rec, &key) {
                out.hull[i] = trim(s);
            }
        }
        for i in 0..ROBOT_CHASSIS_CNT {
            let key = format!("Chas{}Key", i + 1);
            if let Some(s) = stor.block_param(&names_rec, &key) {
                out.chassis[i] = trim(s);
            }
        }
        for i in 0..ROBOT_HEAD_CNT {
            let key = format!("Head{}Key", i + 1);
            if let Some(s) = stor.block_param(&names_rec, &key) {
                out.head[i] = trim(s);
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

/// `Labels/Turrets` — Tur{1..4}_Name / Tur{1..4}_Range, consumed by
/// the turret-button hover hints (CInterface.cpp:4463-4520).
#[derive(Debug, Clone, Default)]
pub struct TurretLabels {
    pub name: [String; 4],
    pub range: [String; 4],
}

impl TurretLabels {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let labels_rec = stor.block_record("da", "Labels")?;
        let rec = stor.block_record(&labels_rec, "Turrets")?;
        let mut out = Self::default();
        for i in 0..4 {
            if let Some(s) = stor.block_param(&rec, &format!("Tur{}_Name", i + 1)) {
                out.name[i] = s;
            }
            if let Some(s) = stor.block_param(&rec, &format!("Tur{}_Range", i + 1)) {
                out.range[i] = s;
            }
        }
        Some(out)
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
    pub turrets: TurretLabels,
    /// Localised "none" word — port of `AllLabels/Base/none` loaded at
    /// startup (MatrixGame.cpp:521-523) and assigned to
    /// `g_PopupHead[0].text` / `g_PopupWeaponNormal[0].text` /
    /// `g_PopupWeaponExtern[0].text` as the empty-slot popup row.
    pub popup_none: String,
}

/// One supply-drop robot loadout — a child block of
/// `FlyerOrders/GiveBot` whose NAME is the score cost
/// (MatrixObjectBuilding.cpp:1334, MatrixFlyer.cpp:1196-1219).
#[derive(Debug, Clone, Default)]
pub struct GiveBotEntry {
    pub score: i32,
    pub chassis: i32,
    pub armor: i32,
    pub head: i32,
    pub weapons: Vec<i32>,
    pub strength: f32,
    pub hitpoint: f32,
}

/// Port of the `FlyerOrders/GiveBot` block (PAR_SOURCE_FLYER_ORDERS,
/// MatrixFlyer.cpp:1162-1250).
#[derive(Debug, Clone, Default)]
pub struct GiveBotConfig {
    pub distance: f32,
    pub height: f32,
    pub land_height: f32,
    pub flyer_kind: i32,
    pub hitpoint: f32,
    pub bots: Vec<GiveBotEntry>,
}

impl GiveBotConfig {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let fo = stor.block_record("da", "FlyerOrders")?;
        let gb = stor.block_record(&fo, "GiveBot")?;
        let pf = |k: &str| -> f32 {
            stor.block_param(&gb, k)
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0)
        };
        let mut out = Self {
            distance: pf("Distance"),
            height: pf("Height"),
            land_height: pf("LandHeight"),
            flyer_kind: pf("Flyer") as i32,
            hitpoint: pf("Hitpoint"),
            bots: Vec::new(),
        };
        // Child blocks: names (column "2") are the score costs,
        // records (column "3") hold the loadout params.
        let names = stor.get_buf(&gb, "2")?;
        let records = stor.get_buf(&gb, "3")?;
        for i in 0..names.arrays_count() {
            let score = names.get_as_wstr(i).trim().parse::<i32>().unwrap_or(0);
            let rec = records.get_as_wstr(i);
            let pi = |k: &str| -> i32 {
                stor.block_param(&rec, k)
                    .and_then(|s| s.trim().parse::<i32>().ok())
                    .unwrap_or(0)
            };
            let pe = |k: &str| -> f32 {
                stor.block_param(&rec, k)
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.0)
            };
            // BotWeapon appears multiple times — walk the raw key/value
            // columns in lock-step.
            let mut weapons = Vec::new();
            if let (Some(keys), Some(vals)) = (stor.get_buf(&rec, "0"), stor.get_buf(&rec, "1")) {
                for k in 0..keys.arrays_count() {
                    if keys.get_as_wstr(k).eq_ignore_ascii_case("BotWeapon") {
                        if let Ok(w) = vals.get_as_wstr(k).trim().parse::<i32>() {
                            weapons.push(w);
                        }
                    }
                }
            }
            out.bots.push(GiveBotEntry {
                score,
                chassis: pi("BotChassis"),
                armor: pi("BotArmor"),
                head: pi("BotHead"),
                weapons,
                strength: pe("BotStrength"),
                hitpoint: pe("BotHitpoint"),
            });
        }
        Some(out)
    }
}

/// One entry of the `RobotSpawn` catalogue — a BEHF_SPAWNER bot loadout
/// (MatrixObject.cpp:1383-1408). `number` is the child-block name.
#[derive(Debug, Clone)]
pub struct RobotSpawnBot {
    pub number: i32,
    pub chassis: i32,
    pub armor: i32,
    pub head: i32,
    pub weapons: Vec<i32>,
    pub side: i32,
    pub hitpoint: f32,
}

/// Port of `g_MatrixData->BlockGet("RobotSpawn")` — the spawner-object
/// bot catalogue keyed by robot number.
#[derive(Debug, Clone, Default)]
pub struct RobotSpawnConfig {
    pub bots: Vec<RobotSpawnBot>,
}

impl RobotSpawnConfig {
    pub fn from_matrix_data(stor: &Storage) -> Option<Self> {
        let rs = stor.block_record("da", "RobotSpawn")?;
        let mut out = Self { bots: Vec::new() };
        for (name, rec) in stor.block_records(&rs) {
            let number = name.trim().parse::<i32>().unwrap_or(-1);
            let pi = |k: &str| -> i32 {
                stor.block_param(&rec, k)
                    .and_then(|s| s.trim().parse::<i32>().ok())
                    .unwrap_or(0)
            };
            let weapons: Vec<i32> = stor
                .block_params(&rec, "BotWeapon")
                .iter()
                .filter_map(|s| s.trim().parse::<i32>().ok())
                .collect();
            out.bots.push(RobotSpawnBot {
                number,
                chassis: pi("BotChassis"),
                armor: pi("BotArmor"),
                head: pi("BotHead"),
                weapons,
                side: pi("BotSide"),
                hitpoint: stor
                    .block_param(&rec, "BotHitpoint")
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.0),
            });
        }
        Some(out)
    }

    /// Pick the bot with `number`, else a pseudo-random one (mirrors
    /// `BlockGetNE(number)` with the `BlockGet(Rnd(...))` fallback,
    /// MatrixObject.cpp:1387-1391). `pick` selects the fallback index.
    pub fn choose(&self, number: i32, pick: usize) -> Option<&RobotSpawnBot> {
        if let Some(b) = self.bots.iter().find(|b| b.number == number) {
            return Some(b);
        }
        if self.bots.is_empty() {
            return None;
        }
        self.bots.get(pick % self.bots.len())
    }
}

static ROBOT_SPAWN_CONFIG: std::sync::OnceLock<std::sync::RwLock<std::sync::Arc<RobotSpawnConfig>>> =
    std::sync::OnceLock::new();

pub fn robot_spawn_config() -> std::sync::Arc<RobotSpawnConfig> {
    ROBOT_SPAWN_CONFIG
        .get_or_init(|| std::sync::RwLock::new(std::sync::Arc::new(RobotSpawnConfig::default())))
        .read()
        .unwrap()
        .clone()
}

pub fn set_robot_spawn_config(c: RobotSpawnConfig) {
    let slot = ROBOT_SPAWN_CONFIG
        .get_or_init(|| std::sync::RwLock::new(std::sync::Arc::new(RobotSpawnConfig::default())));
    *slot.write().unwrap() = std::sync::Arc::new(c);
}

static GIVE_BOT_CONFIG: std::sync::OnceLock<std::sync::RwLock<std::sync::Arc<GiveBotConfig>>> =
    std::sync::OnceLock::new();

pub fn give_bot_config() -> std::sync::Arc<GiveBotConfig> {
    GIVE_BOT_CONFIG
        .get_or_init(|| std::sync::RwLock::new(std::sync::Arc::new(GiveBotConfig::default())))
        .read()
        .unwrap()
        .clone()
}

pub fn set_give_bot_config(c: GiveBotConfig) {
    let slot = GIVE_BOT_CONFIG
        .get_or_init(|| std::sync::RwLock::new(std::sync::Arc::new(GiveBotConfig::default())));
    *slot.write().unwrap() = std::sync::Arc::new(c);
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

// ── Sounds block (MatrixSoundManager.cpp SureLoaded) ─────────────────

/// One `Sounds/<key>` entry from robots.dat — the parameters
/// `CSound::SureLoaded` (MatrixSoundManager.cpp:392-460) reads lazily
/// per sound. Defaults match the C++ pre-init values exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct SoundDef {
    /// SR2 resource path (`Sound.X`) — the asset id the host resolves.
    pub path: String,
    pub vol0: f32,
    pub vol1: f32,
    pub pan0: f32,
    pub pan1: f32,
    /// Already scaled: `0.002 * attn_par`; `0` ⇒ infinite radius.
    pub attn: f32,
    pub looped: bool,
    /// Valid only for looped positional sounds (ms).
    pub ttl: f32,
    pub fade: f32,
}

impl Default for SoundDef {
    fn default() -> Self {
        Self {
            path: String::new(),
            vol0: 1.0,
            vol1: 1.0,
            pan0: 0.0,
            pan1: 0.0,
            attn: 0.002,
            looped: false,
            ttl: 1e30,
            fade: 1000.0,
        }
    }
}

/// All `Sounds/*` entries, plus the `ChassisSounds` order-voice table.
/// Missing keys resolve to the `dummy` entry (`Sound.Mute`, vol 0) —
/// the `bps->BlockGetNE(L"dummy")` fallback in SureLoaded, which is
/// how the original silences sounds absent from robots.dat.
#[derive(Debug, Clone, Default)]
pub struct SoundDefs {
    map: std::collections::HashMap<String, SoundDef>,
    /// `Chars/ChassisSounds/<kind 1..5>` → (MoveTo keys, Patrol keys)
    /// (MatrixSide.cpp:7975/8196).
    chassis_voices: std::collections::HashMap<i32, (Vec<String>, Vec<String>)>,
}

impl SoundDefs {
    pub fn from_matrix_data(stor: &Storage) -> Self {
        let mut map = std::collections::HashMap::new();
        if let Some(snds) = stor.block_record("da", "Sounds") {
            for (name, rec) in stor.block_records(&snds) {
                let mut def = SoundDef::default();
                if let Some(p) = stor.block_param(&rec, "path") {
                    def.path = p;
                }
                if let Some(v) = stor.block_param(&rec, "pan") {
                    def.pan0 = wstr::double_par(&v, 0, ",") as f32;
                    def.pan1 = wstr::double_par(&v, 1, ",") as f32;
                }
                if let Some(v) = stor.block_param(&rec, "vol") {
                    def.vol0 = wstr::double_par(&v, 0, ",") as f32;
                    def.vol1 = wstr::double_par(&v, 1, ",") as f32;
                }
                if let Some(v) = stor.block_param(&rec, "looped") {
                    def.looped = wstr::int_par(&v, 0, ",") != 0;
                }
                if let Some(v) = stor.block_param(&rec, "ttl") {
                    def.ttl = wstr::double_par(&v, 0, ",") as f32;
                    def.fade = wstr::double_par(&v, 1, ",") as f32;
                }
                if let Some(v) = stor.block_param(&rec, "attn") {
                    def.attn = 0.002 * wstr::double_par(&v, 0, ",") as f32;
                }
                map.insert(name.to_lowercase(), def);
            }
        }
        let mut chassis_voices = std::collections::HashMap::new();
        if let Some(chars) = stor.block_record("da", "Chars") {
            if let Some(cs) = stor.block_record(&chars, "ChassisSounds") {
                for (name, rec) in stor.block_records(&cs) {
                    if let Ok(kind) = name.parse::<i32>() {
                        chassis_voices.insert(
                            kind,
                            (
                                stor.block_params(&rec, "MoveTo"),
                                stor.block_params(&rec, "Patrol"),
                            ),
                        );
                    }
                }
            }
        }
        Self {
            map,
            chassis_voices,
        }
    }

    /// Resolve a Sounds-block key; falls back to `dummy` like the C++.
    /// Lookup is case-insensitive (CBlockPar compares with _wcsicmp).
    pub fn get(&self, key: &str) -> Option<&SoundDef> {
        self.map
            .get(&key.to_lowercase())
            .or_else(|| self.map.get("dummy"))
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate all `(key, def)` pairs — used by the asset packer.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &SoundDef)> {
        self.map.iter()
    }

    /// Order-voice keys for a chassis kind: `(MoveTo, Patrol)`.
    pub fn chassis_voices(&self, kind: i32) -> Option<&(Vec<String>, Vec<String>)> {
        self.chassis_voices.get(&kind)
    }

    /// Test-only entry point — mixer tests build tiny def sets
    /// without a robots.dat on disk.
    #[cfg(test)]
    pub fn insert_for_test(&mut self, key: &str, def: SoundDef) {
        self.map.insert(key.to_lowercase(), def);
    }
}

#[cfg(test)]
mod sound_tests {
    use super::*;

    fn load() -> Option<SoundDefs> {
        let data = std::fs::read("../Data/robots.dat").ok()?;
        let stor = Storage::from_bytes(&data).ok()?;
        Some(SoundDefs::from_matrix_data(&stor))
    }

    #[test]
    fn sounds_block_parses_real_values() {
        let Some(defs) = load() else {
            eprintln!("robots.dat not present; skipping");
            return;
        };
        let w = defs.get("wplasma").unwrap();
        assert_eq!(w.path, "Sound.WeapPlasma");
        assert_eq!((w.vol0, w.vol1), (0.0, 0.5));
        assert_eq!((w.pan0, w.pan1), (-0.3, 0.3));
        assert!((w.attn - 0.001).abs() < 1e-6); // 0.002 * 0.5
        assert!(!w.looped);

        let e = defs.get("expl_bb").unwrap();
        assert_eq!(e.path, "Sound.ExpBuildS");
        assert_eq!((e.vol0, e.vol1), (0.4, 1.0));
        assert!((e.attn - 0.002).abs() < 1e-6);

        // expl_bb4 has attn=0 ⇒ infinite radius sentinel.
        assert_eq!(defs.get("expl_bb4").unwrap().attn, 0.0);

        // Looped + ttl pair.
        let a = defs.get("whablaze").unwrap();
        assert!(a.looped);
        assert_eq!((a.ttl, a.fade), (200.0, 100.0));

        // Missing key falls back to the silent dummy.
        let d = defs.get("no_such_sound").unwrap();
        assert_eq!(d.path, "Sound.Mute");
        assert_eq!((d.vol0, d.vol1), (0.0, 0.0));

        // Chassis order voices.
        let (move_to, patrol) = defs.chassis_voices(3).unwrap();
        assert_eq!(move_to, &["s_ord_mtc3_1", "s_ord_mtc3_2"]);
        assert_eq!(patrol, &["s_ord_mtc3_1", "s_ord_mtc3_2"]);
    }
}
