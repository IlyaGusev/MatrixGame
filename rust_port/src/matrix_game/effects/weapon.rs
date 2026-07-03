//! Port of `MatrixEffectWeapon.{cpp,hpp}` — the `EWeapon` enum plus
//! `CMatrixEffectWeapon`, the per-weapon firing state machine that
//! owns cooldown timing, hit-scan damage delivery and projectile
//! spawning.
//!
//! The C++ enum carries sparse, partly-magical discriminants (200, 70,
//! 10000000, …) because a few callers reuse them as damage magnitudes
//! or sort keys. We preserve those values for parity with saved-game
//! blobs and config files.
//!
//! Ownership: the C++ class is ref-counted — the owning robot/cannon
//! holds one ref, every in-flight projectile holds another, and
//! `WeaponHit(FEHF_LASTHIT)` drops the projectile's ref. Rust mirrors
//! that with the [`Weapons`] slab: `create` returns a handle at ref 1,
//! projectiles `add_ref` on spawn and the effect driver releases on
//! effect death (instead of inside `weapon_hit` — same net counts, no
//! double-release hazard).
//!
//! Purely visual sub-effects (CLaser / CVolcano billboards, konus
//! muzzle flashes, gun-fire sprites, sounds) are not ported — only the
//! gameplay-observable behavior is.

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

/// Inverse of [`weap_to_index`] — dense index back to the weapon
/// discriminant (used by `WeapName2Weap`-style config lookups).
pub fn weapon_for_index(idx: usize) -> Weapon {
    const TABLE: [Weapon; WEAPON_COUNT] = [
        WEAPON_PLASMA,
        WEAPON_VOLCANO,
        WEAPON_HOMING_MISSILE,
        WEAPON_BOMB,
        WEAPON_FLAMETHROWER,
        WEAPON_BIGBOOM,
        WEAPON_LIGHTENING,
        WEAPON_LASER,
        WEAPON_GUN,
        WEAPON_ABLAZE,
        WEAPON_SHORTED,
        WEAPON_DEBRIS,
        WEAPON_REPAIR,
        WEAPON_CANNON0,
        WEAPON_CANNON1,
        WEAPON_CANNON2,
        WEAPON_CANNON3,
    ];
    TABLE.get(idx).copied().unwrap_or(WEAPON_NONE)
}

// ── CMatrixEffectWeapon port ─────────────────────────────────────────

use crate::matrix_game::common::{TRACE_ALL, TRACE_BUILDING, TRACE_CANNON, TRACE_ROBOT};
use crate::matrix_game::config;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{MapStatic, ObjectId, ObjectType, Objects};
use crate::matrix_game::map_trace::{trace, TraceStop};
use glam::Vec3;

/// `WEAPON_MAX_HEAT` (MatrixEffectWeapon.hpp:31).
pub const WEAPON_MAX_HEAT: i32 = 1000;
/// `ARCADEBOT_WEAPON_COEFF` / `DEFBOT_WEAPON_COEFF`
/// (MatrixEffectWeapon.hpp:18-19).
pub const ARCADEBOT_WEAPON_COEFF: f32 = 1.2;
pub const DEFBOT_WEAPON_COEFF: f32 = 1.0;

/// `FEHF_*` fire-end-handler flags (MatrixEffect.hpp:132-133).
pub const FEHF_LASTHIT: u32 = 1 << 0;
pub const FEHF_DAMAGE_ROBOT: u32 = 1 << 1;

/// Who receives the `FIRE_END_HANDLER` callback — the C++ passes a
/// function pointer + user data (`robot_weapon_hit` with the robot, or
/// `CMatrixCannon::FireHandler` with the cannon core).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeaponHandler {
    None,
    Robot(ObjectId),
    Cannon(ObjectId),
}

/// Handle into the [`Weapons`] slab — generational like `ObjectId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeaponId {
    index: u32,
    generation: u32,
}

impl WeaponId {
    /// Synthetic key for the follow-light registry — non-weapon light
    /// owners (fire debris embers) key with the high bit set, a space
    /// real slab handles never reach.
    pub fn synthetic(n: u32) -> Self {
        Self {
            index: n | 0x8000_0000,
            generation: u32::MAX,
        }
    }
}

/// Canonical Sounds-block key for a weapon's FIRE sound (`m_SoundType`,
/// MatrixEffectWeapon.cpp:90-137 → MatrixSoundManager.cpp:130-144).
pub fn fire_sound_key(w: Weapon) -> &'static str {
    match w {
        WEAPON_PLASMA => "wplasma",
        WEAPON_VOLCANO => "wvolcano",
        WEAPON_HOMING_MISSILE => "wmissile",
        WEAPON_BOMB => "wbomb",
        WEAPON_FLAMETHROWER => "wflame",
        WEAPON_BIGBOOM => "wbigboom",
        WEAPON_LIGHTENING => "wlightening",
        WEAPON_LASER => "wlaser",
        WEAPON_GUN => "wgun",
        WEAPON_REPAIR => "wrepair",
        WEAPON_CANNON0 => "wcannon0",
        WEAPON_CANNON1 => "wcannon1",
        WEAPON_CANNON2 => "wcannon2",
        WEAPON_CANNON3 => "wcannon3",
        _ => "",
    }
}

/// Canonical key for a weapon's HIT sound (`SoundHit`,
/// MatrixEffectWeapon.cpp:846 → MatrixSoundManager.cpp:146-163).
pub fn hit_sound_key(w: Weapon) -> &'static str {
    match w {
        WEAPON_PLASMA => "whplasma",
        WEAPON_VOLCANO => "whvolcano",
        WEAPON_HOMING_MISSILE => "whmissile",
        WEAPON_BOMB => "whbomb",
        WEAPON_FLAMETHROWER => "whflame",
        WEAPON_BIGBOOM => "whbigboom",
        WEAPON_LIGHTENING => "whlightening",
        WEAPON_LASER => "whlaser",
        WEAPON_GUN => "whgun",
        WEAPON_REPAIR => "whrepair",
        WEAPON_ABLAZE => "whablaze",
        WEAPON_SHORTED => "whshorted",
        WEAPON_DEBRIS => "whdebris",
        WEAPON_CANNON0 => "whcannon0",
        WEAPON_CANNON1 => "whcannon1",
        WEAPON_CANNON2 => "whcannon2",
        WEAPON_CANNON3 => "whcannon3",
        _ => "",
    }
}

struct WSlot {
    generation: u32,
    /// `m_Ref` (MatrixEffectWeapon.hpp:216).
    refs: i32,
    /// `None` while checked out by `weapon_takt` or after drop.
    w: Option<WeaponEffect>,
    /// Slot is live (distinguishes checked-out from freed).
    live: bool,
}

/// Slab of live weapon effects. Lives on [`Objects`] so damage /
/// projectile paths can reach it through the one `&mut Objects` they
/// already carry.
#[derive(Default)]
pub struct Weapons {
    slots: Vec<WSlot>,
    free: Vec<u32>,
}

impl Weapons {
    pub fn create(&mut self, w: WeaponEffect) -> WeaponId {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.w = Some(w);
            slot.refs = 1;
            slot.live = true;
            WeaponId {
                index,
                generation: slot.generation,
            }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(WSlot {
                generation: 1,
                refs: 1,
                w: Some(w),
                live: true,
            });
            WeaponId {
                index,
                generation: 1,
            }
        }
    }

    fn slot(&self, id: WeaponId) -> Option<&WSlot> {
        self.slots
            .get(id.index as usize)
            .filter(|s| s.generation == id.generation && s.live)
    }

    fn slot_mut(&mut self, id: WeaponId) -> Option<&mut WSlot> {
        self.slots
            .get_mut(id.index as usize)
            .filter(|s| s.generation == id.generation && s.live)
    }

    pub fn get(&self, id: WeaponId) -> Option<&WeaponEffect> {
        self.slot(id).and_then(|s| s.w.as_ref())
    }

    pub fn get_mut(&mut self, id: WeaponId) -> Option<&mut WeaponEffect> {
        self.slot_mut(id).and_then(|s| s.w.as_mut())
    }

    /// `m_Ref++` — a projectile spawned with this weapon as callback.
    pub fn add_ref(&mut self, id: WeaponId) {
        if let Some(s) = self.slot_mut(id) {
            s.refs += 1;
        }
    }

    /// `CMatrixEffectWeapon::Release` (MatrixEffectWeapon.cpp:189) —
    /// drop one ref; free the slot at zero. If the entry is checked
    /// out (take-the-box), `put_back` finishes the free.
    pub fn release(&mut self, id: WeaponId) {
        let Some(s) = self.slot_mut(id) else { return };
        s.refs -= 1;
        if s.refs <= 0 && s.w.is_some() {
            self.free_slot(id);
        }
    }

    fn free_slot(&mut self, id: WeaponId) {
        let slot = &mut self.slots[id.index as usize];
        slot.w = None;
        slot.live = false;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(id.index);
    }

    fn take(&mut self, id: WeaponId) -> Option<WeaponEffect> {
        self.slot_mut(id).and_then(|s| s.w.take())
    }

    /// Iterate live weapon effects (for the visual draw pass).
    pub fn iter(&self) -> impl Iterator<Item = &WeaponEffect> {
        self.slots
            .iter()
            .filter(|s| s.live)
            .filter_map(|s| s.w.as_ref())
    }

    fn put_back(&mut self, id: WeaponId, w: WeaponEffect) {
        let Some(s) = self
            .slots
            .get_mut(id.index as usize)
            .filter(|s| s.generation == id.generation && s.live)
        else {
            return;
        };
        if s.refs <= 0 {
            // Released while checked out — finish the free.
            self.free_slot(id);
        } else {
            s.w = Some(w);
        }
    }
}

// `m_Flags` bits local to the weapon effect (WEAPFLAGS_*,
// MatrixEffect.hpp). Sound bits are not ported.
const WEAPFLAGS_FIRE: u32 = 1 << 0;
const WEAPFLAGS_FIREWAS: u32 = 1 << 1;
const WEAPFLAGS_HITWAS: u32 = 1 << 2;

/// One puff the flamethrower weapon queued for its flame effect.
#[derive(Debug, Clone, Copy)]
pub struct FlamePuffSpawn {
    pub pos: Vec3,
    pub dir: Vec3,
    pub speed: Vec3,
}

/// Port of `CMatrixEffectWeapon`.
pub struct WeaponEffect {
    pub weapon_type: Weapon,
    /// `m_WeaponDist` — base range from `g_Config.m_WeaponRadius`.
    weapon_dist: f32,
    /// `m_WeaponCoefficient` — 1.0, or 1.2 for the player's arcaded bot.
    coefficient: f32,
    pub pos: Vec3,
    pub dir: Vec3,
    pub speed: Vec3,
    /// `m_Time` — goes negative after each shot by `m_CoolDown`.
    time: f32,
    cool_down: f32,
    pub skip: Option<ObjectId>,
    fire_count: i32,
    flags: u32,
    owner: Option<ObjectId>,
    /// `m_SideStorage` — owner's side captured at SetOwner so kills
    /// still credit the right side after the owner dies.
    side_storage: i32,
    pub handler: WeaponHandler,

    // Flamethrower link (replaces the C++ `m_Effect` pointer — see
    // module doc).
    pub pending_puffs: Vec<FlamePuffSpawn>,
    pub flame_alive: bool,
    /// Set at FireEnd for the flamethrower — the flame effect marks
    /// its newest puff as a burst boundary (`Break()`,
    /// MatrixEffectWeapon.cpp:810-814).
    pub flame_break_pending: bool,

    // Repair beam state (replaces the C++ `m_Repair` effect pointer;
    // the seek logic from MatrixEffectRepair.cpp:177-250 runs inline
    // on the weapon's own clock).
    repair_target: Option<ObjectId>,
    repair_time: f32,
    repair_next_seek: f32,

    // ── Weapon-owned visuals ────────────────────────────────────────
    /// CLaser / lightening beam endpoints (refreshed per takt while
    /// firing; the C++ updates them in `Modify`).
    pub beam: Option<(Vec3, Vec3)>,
    /// CVolcano muzzle visual active (drawn from pos/dir).
    pub volcano_on: bool,
    /// Plasma muzzle-cone birth time + angle — the cone is weapon-owned
    /// so `Modify(pos, dir)` re-aims it while the barrel moves
    /// (MatrixEffectWeapon.cpp:236-241); drawn live like the volcano.
    plasma_muzzle: Option<(i64, f32)>,
    /// CMatrixEffectLightening owned by the electric weapon.
    pub bolt: Option<crate::matrix_game::effects::lightening::Lightening>,
    /// Traveling repair-beam glow billboards: (t, dt) pairs
    /// (MatrixEffectRepair.cpp:420-436).
    pub repair_bb: Vec<(f32, f32)>,
    repair_next_spawn: f32,
    /// `m_OffTargetAmp` — growing seek wobble amplitude, capped at
    /// seek_radius·0.2 (MatrixEffectRepair.cpp:164-173).
    repair_off_amp: f32,
    /// `m_ChangeTime` countdown for the beam-shape morph when a
    /// patient is acquired (REPAIR_CHANGE_TIME 500 ms).
    repair_change: f32,
}

impl WeaponEffect {
    /// Ctor (MatrixEffectWeapon.cpp:67-88): range from
    /// `m_WeaponRadius`, cooldown from the explicit parameter, the
    /// weapon discriminant value, or `m_WeaponCooldown` (in priority
    /// order — config wins when positive).
    pub fn new(weapon_type: Weapon, cooldown: i32, handler: WeaponHandler) -> Self {
        let cfg = config::global();
        let widx = weap_to_index(weapon_type);
        let weapon_dist = widx.map(|i| cfg.weapon_radius.table[i]).unwrap_or(0.0);
        let mut cool_down = if cooldown != 0 {
            cooldown as f32
        } else {
            weapon_type as f32
        };
        if let Some(i) = widx {
            if cfg.weapon_cooldown.table[i] > 0 {
                cool_down = cfg.weapon_cooldown.table[i] as f32;
            }
        }
        Self {
            weapon_type,
            weapon_dist,
            coefficient: DEFBOT_WEAPON_COEFF,
            pos: Vec3::ZERO,
            dir: Vec3::Z,
            speed: Vec3::ZERO,
            time: 0.0,
            cool_down,
            skip: None,
            fire_count: 0,
            flags: 0,
            owner: None,
            side_storage: 0,
            handler: WeaponHandler::None,
            pending_puffs: Vec::new(),
            flame_alive: false,
            flame_break_pending: false,
            repair_target: None,
            repair_time: 0.0,
            repair_next_seek: 0.0,
            beam: None,
            volcano_on: false,
            plasma_muzzle: None,
            bolt: None,
            repair_bb: Vec::new(),
            repair_next_spawn: 0.0,
            repair_off_amp: 0.0,
            repair_change: 0.0,
        }
        .with_handler(handler)
    }

    fn with_handler(mut self, handler: WeaponHandler) -> Self {
        self.handler = handler;
        self
    }

    /// `SetOwner` — keeps the owner handle + its side for kill credit.
    pub fn set_owner(&mut self, owner: ObjectId, side: i32) {
        self.owner = Some(owner);
        self.side_storage = side;
    }

    pub fn owner(&self) -> Option<ObjectId> {
        self.owner
    }
    pub fn side_storage(&self) -> i32 {
        self.side_storage
    }
    pub fn weapon_type(&self) -> Weapon {
        self.weapon_type
    }
    /// `GetWeaponDist` — range with the arcade coefficient applied.
    pub fn weapon_dist(&self) -> f32 {
        self.weapon_dist * self.coefficient
    }
    pub fn set_default_coefficient(&mut self) {
        self.coefficient = DEFBOT_WEAPON_COEFF;
    }
    pub fn set_arcade_coefficient(&mut self) {
        self.coefficient = ARCADEBOT_WEAPON_COEFF;
    }
    /// `ModifyCoolDown(addk)` / `ModifyDist(addk)` — head bonuses.
    pub fn modify_cool_down(&mut self, addk: f32) {
        self.cool_down += self.cool_down * addk;
    }
    pub fn modify_dist(&mut self, addk: f32) {
        self.weapon_dist += self.weapon_dist * addk;
    }

    pub fn is_fire(&self) -> bool {
        self.flags & WEAPFLAGS_FIRE != 0
    }
    /// `IsFireWas` — read-and-clear.
    pub fn is_fire_was(&mut self) -> bool {
        let r = self.flags & WEAPFLAGS_FIREWAS != 0;
        self.flags &= !WEAPFLAGS_FIREWAS;
        r
    }
    /// `IsHitWas` — read-and-clear.
    pub fn is_hit_was(&mut self) -> bool {
        let r = self.flags & WEAPFLAGS_HITWAS != 0;
        self.flags &= !WEAPFLAGS_HITWAS;
        r
    }
    pub fn reset_fire_count(&mut self) {
        self.fire_count = 0;
    }
    pub fn fire_count(&self) -> i32 {
        self.fire_count
    }

    /// `FireBegin` (MatrixEffectWeapon.hpp:277-286).
    pub fn fire_begin(&mut self, speed: Vec3, skip: Option<ObjectId>) {
        if self.is_fire() {
            return;
        }
        self.speed = speed;
        self.flags |= WEAPFLAGS_FIRE;
        self.flags &= !(WEAPFLAGS_FIREWAS | WEAPFLAGS_HITWAS);
        self.skip = skip;
    }

    /// `FireEnd` (MatrixEffectWeapon.cpp:749). The flamethrower's
    /// `Break()` happens on the flame effect's side: it watches
    /// `is_fire()` each takt. The repair beam just drops its target.
    pub fn fire_end(&mut self) {
        if !self.is_fire() {
            return;
        }
        self.flags &= !WEAPFLAGS_FIRE;
        if self.weapon_type == WEAPON_FLAMETHROWER {
            self.flame_break_pending = true;
        }
        if self.weapon_type == WEAPON_REPAIR {
            // The C++ destroys the repair effect here; the next burst
            // creates a fresh one whose first takt seeks immediately.
            self.repair_target = None;
            self.repair_next_seek = self.repair_time;
            self.repair_bb.clear();
        }
        // FireEnd tears the beam visuals down (laser / volcano /
        // lightening sub-effects, MatrixEffectWeapon.cpp:763-795).
        self.beam = None;
        self.volcano_on = false;
        self.bolt = None;
    }

    /// `Modify(pos, dir, speed)` — visual sub-effect repositioning is
    /// not ported; position/direction updates are.
    pub fn modify(&mut self, pos: Vec3, dir: Vec3, speed: Vec3) {
        self.pos = pos;
        self.dir = dir;
        self.speed = speed;
    }

    /// `Takt` (MatrixEffectWeapon.cpp:203-227) — the cooldown loop.
    /// Called via [`weapon_takt`] which handles the registry checkout.
    fn takt(
        &mut self,
        self_id: WeaponId,
        step: f32,
        map: &GameMap,
        objs: &mut Objects,
        rng: &mut crate::matrix_game::logic::Rnd,
    ) {
        if self.time < 0.0 {
            self.time += step;
        }
        if self.time > 0.0 && !self.is_fire() {
            self.time = 0.0;
        }
        if self.weapon_type == WEAPON_REPAIR {
            self.repair_time += step;
            // Seek wobble amplitude grows to seek_radius·0.2
            // (MatrixEffectRepair.cpp:164-173); the shape-morph
            // countdown runs whenever active (:141-158).
            let lim = self.weapon_dist() * 0.2;
            self.repair_off_amp = (self.repair_off_amp + 0.001 * step).min(lim);
            self.repair_change = (self.repair_change - step).max(0.0);
        }
        if self.is_fire() && self.cool_down > 0.0 {
            while self.time >= 0.0 {
                self.fire(self_id, map, objs, rng);
                self.flags |= WEAPFLAGS_FIREWAS;
                self.time -= self.cool_down;
            }
        }
        self.visual_takt(step, map, objs, rng);
    }

    /// Beam / bolt endpoint tracking — the trace the C++ runs in
    /// `Modify` for LASER / CANNON2 / LIGHTENING while firing
    /// (MatrixEffectWeapon.cpp:251-269), plus the repair traveling
    /// glints.
    fn visual_takt(
        &mut self,
        step: f32,
        map: &GameMap,
        objs: &Objects,
        rng: &mut crate::matrix_game::logic::Rnd,
    ) {
        if self.is_fire()
            && matches!(
                self.weapon_type,
                WEAPON_LASER | WEAPON_CANNON2 | WEAPON_LIGHTENING
            )
        {
            let (_, hitpos) = trace(
                map,
                objs,
                self.pos,
                self.pos + self.dir * self.weapon_dist(),
                TRACE_ALL,
                self.skip,
            );
            if self.weapon_type == WEAPON_LIGHTENING {
                match &mut self.bolt {
                    Some(b) => {
                        b.set_pos(self.pos, hitpos);
                        b.takt(step, rng);
                    }
                    None => {
                        self.bolt = Some(crate::matrix_game::effects::lightening::Lightening::new(
                            self.pos,
                            hitpos,
                            1_000_000.0,
                            3.0,
                            crate::matrix_game::effects::lightening::LIGHTENING_WIDTH,
                            0xFFFF_FFFF,
                        ));
                    }
                }
            } else {
                self.beam = Some((self.pos, hitpos));
            }
        }
        if self.is_fire() && self.weapon_type == WEAPON_REPAIR {
            // REPAIR_SPAWN_PERIOD = 13ms; speed 0.006 + FRND(0.03).
            while self.repair_time > self.repair_next_spawn {
                self.repair_next_spawn = self.repair_time + 13.0;
                if self.repair_bb.len() < 50 {
                    let dt = 0.006 + crate::matrix_game::effects::smoke_and_fire::frnd(rng, 0.03);
                    self.repair_bb.push((0.0, dt));
                }
            }
            for bb in &mut self.repair_bb {
                bb.0 += bb.1 * step * 0.06; // dt per ~16ms frame in the C++
            }
            self.repair_bb.retain(|bb| bb.0 <= 1.0);
        }
    }

    /// `Fire` (MatrixEffectWeapon.cpp:281-747), gameplay branches only.
    fn fire(
        &mut self,
        self_id: WeaponId,
        map: &GameMap,
        objs: &mut Objects,
        rng: &mut crate::matrix_game::logic::Rnd,
    ) {
        use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd};
        use crate::matrix_game::effects::{
            big_boom::BigBoom, fire_plasma::FirePlasma, flame::Flame, moving_object::MoKind,
            moving_object::MovingObject, GameEffect,
        };

        self.fire_count += 1;
        let dist = self.weapon_dist();

        // Per-shot fire sound (`m_SoundType` at the muzzle,
        // MatrixEffectWeapon.cpp:291-294).
        let snd = fire_sound_key(self.weapon_type);
        if !snd.is_empty() {
            objs.queue_snd_at(snd, self.pos);
        }

        let now = crate::matrix_game::map::current_elapsed_ms();
        let muzzle_ang = std::f32::consts::TAU / 4096.0 * ((now & 4095) as f32);
        match self.weapon_type {
            WEAPON_PLASMA => {
                // Muzzle cone (MatrixEffectWeapon.cpp:305-306) —
                // weapon-owned so it re-aims with the barrel
                // (Modify, :236-241); drawn in `draw()`.
                self.plasma_muzzle = Some((now, muzzle_ang));
                objs.weapons.add_ref(self_id);
                objs.pending_effects
                    .push(GameEffect::FirePlasma(FirePlasma::new(
                        self.pos,
                        self.pos + self.dir * dist,
                        0.5,
                        TRACE_ALL,
                        self.skip,
                        self_id,
                    )));
            }
            WEAPON_VOLCANO => {
                // CVolcano muzzle visual stays live while firing
                // (MatrixEffectWeapon.cpp:315-327); draw() renders it.
                self.volcano_on = true;
                // Hit-scan (MatrixEffectWeapon.cpp:311-366).
                let (s, hitpos) = trace(
                    map,
                    objs,
                    self.pos,
                    self.pos + self.dir * dist,
                    TRACE_ALL,
                    self.skip,
                );
                if s == TraceStop::None {
                    return;
                }
                let mut s = s;
                if let Some(id) = s.object() {
                    let dead = objs.apply_damage(
                        id,
                        self.weapon_type,
                        hitpos,
                        self.dir,
                        self.side_storage,
                        self.owner,
                    );
                    self.flags |= WEAPFLAGS_HITWAS;
                    if dead {
                        s = TraceStop::None;
                    }
                } else if s == TraceStop::Water {
                    // Water splash konus (MatrixEffectWeapon.cpp:348).
                    objs.queue_snd_at("splash", hitpos);
                    objs.pending_effects.push(GameEffect::Konus(
                        crate::matrix_game::effects::konus::Konus::new_splash(
                            hitpos,
                            Vec3::Z,
                            10.0,
                            5.0,
                            fsrnd(rng, std::f32::consts::PI),
                            1000.0,
                            true,
                            crate::matrix_lib::three_g::billboard::TexRef::Path(
                                crate::matrix_game::effects::effects_renderer::TEXTURE_PATH_SPLASH,
                            ),
                        ),
                    ));
                } else {
                    // Landscape spark konus pair (MatrixEffectWeapon.cpp:355-357).
                    let n = map.get_normal(hitpos.x, hitpos.y, false);
                    let splash = Vec3::new(n[0], n[1], n[2]);
                    for (r, h, tex) in [
                        (5.0, 10.0, crate::matrix_game::effects::effects_renderer::TEXTURE_PATH_GUN_BULLETS1),
                        (5.0, 5.0, crate::matrix_game::effects::effects_renderer::TEXTURE_PATH_GUN_BULLETS2),
                    ] {
                        objs.pending_effects.push(GameEffect::Konus(
                            crate::matrix_game::effects::konus::Konus::new(
                                hitpos,
                                splash,
                                r,
                                h,
                                fsrnd(rng, std::f32::consts::PI),
                                300.0,
                                true,
                                crate::matrix_lib::three_g::billboard::TexRef::Path(tex),
                            ),
                        ));
                    }
                }
                // 10% chance tracer line (MatrixEffectWeapon.cpp:359-361).
                if frnd(rng, 1.0) < 0.1 {
                    use crate::matrix_game::effects::billboard_fx::BillboardLineFx;
                    objs.pending_effects.push(GameEffect::BillboardLine(BillboardLineFx::new(
                        self.pos,
                        hitpos,
                        0.5,
                        0x80FF_FFFF,
                        0x0000_0000,
                        100.0,
                        crate::matrix_lib::three_g::billboard::TexRef::Bbt(
                            crate::matrix_lib::three_g::billboard::BBT_TRASSA,
                        ),
                    )));
                }
                dispatch_fire_end_handler(objs, self.handler, s, hitpos, FEHF_LASTHIT);
            }
            WEAPON_HOMING_MISSILE | WEAPON_CANNON3 => {
                objs.weapons.add_ref(self_id);
                objs.pending_effects
                    .push(GameEffect::MovingObject(MovingObject::new(
                        self.pos,
                        self.pos + self.dir * 1500.0,
                        self.dir,
                        self.side_storage,
                        TRACE_ALL,
                        self.skip,
                        self_id,
                        MoKind::new_missile(),
                    )));
            }
            WEAPON_BOMB => {
                // Mortar lob (MatrixEffectWeapon.cpp:392-487). m_Speed
                // carries the target POSITION for the bomb (set by the
                // ROT_FIRE order processing, MatrixRobot.cpp:1185).
                let mut len = (self.pos - self.speed).length();
                if len < 1.0e-6 {
                    len = 1.0e-6;
                }
                let dir = (self.pos - self.speed) / len;
                let len = len.min(dist);
                let target = self.pos - dir * len;
                let mid = (self.pos + target) * 0.5;

                let (pts, vel_scale): (Vec<Vec3>, f32) = if self.dir.z < 0.0 {
                    // Flyer bombomet branch.
                    (
                        vec![
                            self.pos,
                            self.pos + self.dir * 70.0 - Vec3::new(0.0, 0.0, 25.0),
                            self.pos + self.dir * 110.0 - Vec3::new(0.0, 0.0, 100.0),
                            self.pos + self.dir * 130.0 - Vec3::new(0.0, 0.0, 300.0),
                        ],
                        0.0006,
                    )
                } else {
                    (
                        vec![
                            self.pos,
                            mid + dir * len * 0.15 + Vec3::new(0.0, 0.0, len * 0.5),
                            mid - dir * len * 0.15 + Vec3::new(0.0, 0.0, len * 0.5),
                            target,
                            // "more beautiful trajectory"
                            target - dir * len * 0.35 - Vec3::new(0.0, 0.0, len * 0.5),
                        ],
                        0.0005,
                    )
                };
                objs.weapons.add_ref(self_id);
                objs.pending_effects
                    .push(GameEffect::MovingObject(MovingObject::new(
                        self.pos,
                        target,
                        self.dir,
                        self.side_storage,
                        TRACE_ALL,
                        self.skip,
                        self_id,
                        MoKind::new_bomb(&pts, vel_scale),
                    )));
            }
            WEAPON_GUN | WEAPON_CANNON1 => {
                self.muzzle_flash(objs, rng, 50.0, 30.0);
                objs.weapons.add_ref(self_id);
                objs.pending_effects
                    .push(GameEffect::MovingObject(MovingObject::new(
                        self.pos,
                        self.speed,
                        self.dir * 22.0,
                        self.side_storage,
                        TRACE_ALL,
                        self.skip,
                        self_id,
                        MoKind::new_gun(dist),
                    )));
            }
            WEAPON_CANNON0 => {
                self.muzzle_flash(objs, rng, 35.0, 25.0);
                objs.weapons.add_ref(self_id);
                objs.pending_effects
                    .push(GameEffect::MovingObject(MovingObject::new(
                        self.pos,
                        self.speed,
                        self.dir * 27.0,
                        self.side_storage,
                        TRACE_ALL,
                        self.skip,
                        self_id,
                        MoKind::new_gun_cannon(dist),
                    )));
            }
            WEAPON_LIGHTENING => {
                let (s, hitpos) = trace(
                    map,
                    objs,
                    self.pos,
                    self.pos + self.dir * dist,
                    TRACE_ALL,
                    self.skip,
                );
                let mut s = s;
                if let Some(id) = s.object() {
                    let dead = objs.apply_damage(
                        id,
                        self.weapon_type,
                        hitpos,
                        self.dir,
                        self.side_storage,
                        self.owner,
                    );
                    self.flags |= WEAPFLAGS_HITWAS;
                    if dead {
                        s = TraceStop::None;
                    } else {
                        // Short-out arc to the ground next to the hit
                        // (MatrixEffectWeapon.cpp:556-559).
                        let px = hitpos.x + fsrnd(rng, 20.0);
                        let py = hitpos.y + fsrnd(rng, 20.0);
                        let pz = map.get_z(px, py);
                        objs.pending_effects
                            .push(crate::matrix_game::effects::GameEffect::Lightening(
                                crate::matrix_game::effects::lightening::Lightening::new_shorted(
                                    hitpos,
                                    glam::Vec3::new(px, py, pz),
                                    frnd(rng, 500.0) + 400.0,
                                    0xFFFF_FFFF,
                                ),
                            ));
                    }
                } else if s == TraceStop::Landscape {
                    // SPOT_PLASMA_HIT scorch (MatrixEffectWeapon.cpp:565).
                    objs.pending_spots.push(
                        crate::matrix_game::effects::landscape_spot::SpotSpawn {
                            pos: glam::Vec2::new(hitpos.x, hitpos.y),
                            angle: fsrnd(rng, std::f32::consts::PI),
                            scale: frnd(rng, 1.0) + 0.5,
                            kind: crate::matrix_game::effects::landscape_spot::SpotKind::PlasmaHit,
                        },
                    );
                }
                dispatch_fire_end_handler(objs, self.handler, s, hitpos, FEHF_LASTHIT);
            }
            WEAPON_LASER | WEAPON_CANNON2 => {
                let (s, hitpos) = trace(
                    map,
                    objs,
                    self.pos,
                    self.pos + self.dir * dist,
                    TRACE_ALL,
                    self.skip,
                );
                let mut s = s;
                if let Some(id) = s.object() {
                    let dead = objs.apply_damage(
                        id,
                        self.weapon_type,
                        hitpos,
                        self.dir,
                        self.side_storage,
                        self.owner,
                    );
                    self.flags |= WEAPFLAGS_HITWAS;
                    if dead {
                        s = TraceStop::None;
                    }
                    objs.pending_explosions
                        .push(crate::matrix_game::map_static::ExplosionSpawn {
                            pos: hitpos,
                            props: &crate::matrix_game::effects::explosion::EXPLOSION_LASER_HIT,
                            fire: false,
                        });
                    // The C++ calls the handler twice for object hits
                    // (MatrixEffectWeapon.cpp:613 + :639) — preserved.
                    dispatch_fire_end_handler(objs, self.handler, s, hitpos, FEHF_LASTHIT);
                } else if s == TraceStop::Landscape {
                    objs.pending_spots.push(
                        crate::matrix_game::effects::landscape_spot::SpotSpawn {
                            pos: glam::Vec2::new(hitpos.x, hitpos.y),
                            angle: crate::matrix_game::effects::smoke_and_fire::fsrnd(
                                rng,
                                std::f32::consts::PI,
                            ),
                            scale: crate::matrix_game::effects::smoke_and_fire::frnd(rng, 1.0)
                                + 0.5,
                            kind: crate::matrix_game::effects::landscape_spot::SpotKind::PlasmaHit,
                        },
                    );
                    objs.pending_explosions
                        .push(crate::matrix_game::map_static::ExplosionSpawn {
                            pos: hitpos,
                            props: &crate::matrix_game::effects::explosion::EXPLOSION_LASER_HIT,
                            fire: false,
                        });
                } else if s == TraceStop::Water {
                    objs.pending_effects.push(GameEffect::Smoke(
                        crate::matrix_game::effects::smoke_and_fire::Smoke::new(
                            hitpos,
                            100.0,
                            1400.0,
                            10.0,
                            // C++ passes 0x00FFFFFF — alpha 0, the steam
                            // plume is intentionally invisible
                            // (MatrixEffectWeapon.cpp:625).
                            0x00FF_FFFF,
                            false,
                            0.04,
                        ),
                    ));
                }
                dispatch_fire_end_handler(objs, self.handler, s, hitpos, FEHF_LASTHIT);
            }
            WEAPON_FLAMETHROWER => {
                if !self.flame_alive {
                    let ttl = dist * 8.333;
                    objs.weapons.add_ref(self_id);
                    objs.pending_effects.push(GameEffect::Flame(Flame::new(
                        ttl, TRACE_ALL, self.skip, self_id,
                    )));
                    self.flame_alive = true;
                }
                self.pending_puffs.push(FlamePuffSpawn {
                    pos: self.pos,
                    dir: self.dir,
                    speed: self.speed,
                });
            }
            WEAPON_BIGBOOM => {
                // Damage ring — skip=None intentionally (MatrixEffectWeapon.cpp:663
                // passes NULL): the blast hurts its own owner too.
                objs.weapons.add_ref(self_id);
                objs.pending_effects.push(GameEffect::BigBoom(BigBoom::new(
                    self.pos, dist, 300.0, TRACE_ALL, None, self_id,
                )));
                // Two purely-visual rings at the raw weapon dist
                // (MatrixEffectWeapon.cpp:664-665). The 0xFFAFAF40 on
                // the third call is the LIGHT argument (dead — light=0
                // path); the shell itself is always white
                // (MatrixEffectBigBoom.cpp:31). tracetype 0 → no damage.
                let mut ring = BigBoom::new(self.pos, self.weapon_dist, 350.0, 0, None, self_id);
                ring.deals_damage = false;
                objs.pending_effects.push(GameEffect::BigBoom(ring));
                let mut ring = BigBoom::new(self.pos, self.weapon_dist, 400.0, 0, None, self_id);
                ring.deals_damage = false;
                objs.pending_effects.push(GameEffect::BigBoom(ring));
                // Central detonation flash (MatrixEffectWeapon.cpp:666).
                objs.pending_explosions
                    .push(crate::matrix_game::map_static::ExplosionSpawn {
                        pos: self.pos,
                        props: &crate::matrix_game::effects::explosion::EXPLOSION_BIG_BOOM,
                        fire: true,
                    });
            }
            WEAPON_REPAIR => {
                self.repair_fire(map, objs, rng);
            }
            _ => {}
        }
    }

    /// WEAPON_REPAIR branch of `Fire` + the seek loop of
    /// `CMatrixEffectRepair::Takt` (MatrixEffectRepair.cpp:177-260),
    /// folded onto the weapon's own clock (the standalone repair
    /// effect is purely visual otherwise).
    fn repair_fire(
        &mut self,
        _map: &GameMap,
        objs: &mut Objects,
        _rng: &mut crate::matrix_game::logic::Rnd,
    ) {
        use crate::matrix_game::map_static::Control;

        let seek_radius = self.weapon_dist();
        if self.repair_time >= self.repair_next_seek {
            // REPAIR_SEEK_TARGET_PERIOD = 100 (MatrixEffectRepair.cpp:14).
            self.repair_next_seek = self.repair_time + 100.0;

            // FindPatient (MatrixEffectRepair.cpp:96-129): same-side
            // objects that NeedRepair, robots preferred, nearest to the
            // search center.
            let side_of_owner = self.side_storage;
            let center3 = self.pos + self.dir * seek_radius * 0.5;
            let wpos = self.pos;
            let wdist2 = seek_radius * seek_radius;
            let mut tgt: Option<(ObjectId, f32, bool)> = None;
            objs.find_objects(
                glam::Vec2::new(center3.x, center3.y),
                seek_radius * 0.5,
                1.0,
                TRACE_ROBOT | TRACE_BUILDING | TRACE_CANNON,
                self.skip,
                |_fpos, id| {
                    let Some(o) = objs.get(id) else {
                        return Control::Continue;
                    };
                    if !o.need_repair() || o.side() != side_of_owner {
                        return Control::Continue;
                    }
                    let gc = o.core().geo_center;
                    if (wpos - gc).length_squared() > wdist2 {
                        return Control::Continue;
                    }
                    // C++ FindPatient ranks by distance from the search
                    // CENTER (the fpos FindObjects passes through,
                    // MatrixMap.cpp:2716) — our find_objects hands the
                    // callback the candidate's own center, so use the
                    // seek center directly (MatrixEffectRepair.cpp:105).
                    let dist = (center3 - gc).length_squared();
                    let is_robot = matches!(o.core().obj_type, ObjectType::RobotAi);
                    match tgt {
                        Some((_, best, best_robot)) => {
                            if dist < best && (is_robot || !best_robot) {
                                tgt = Some((id, dist, is_robot));
                            }
                        }
                        None => tgt = Some((id, dist, is_robot)),
                    }
                    Control::Continue
                },
            );
            let new_target = tgt.map(|(id, _, _)| id);
            if new_target != self.repair_target && new_target.is_some() {
                // ERF_FOUND_TARGET — 500 ms morph from the straight
                // seek spline onto the wrap-around one (:141-147).
                self.repair_change = 500.0;
            }
            self.repair_target = new_target;
        }
        // Drop dead targets (the `m_Target->m_Object == NULL` check).
        if let Some(t) = self.repair_target {
            if !objs.is_valid(t) {
                self.repair_target = None;
            }
        }

        if let Some(t) = self.repair_target {
            let gc = objs.get(t).map(|o| o.core().geo_center).unwrap_or(self.pos);
            dispatch_fire_end_handler(objs, self.handler, TraceStop::Object(t), gc, FEHF_LASTHIT);
            self.flags |= WEAPFLAGS_HITWAS;
            objs.apply_damage(
                t,
                self.weapon_type,
                gc,
                self.dir,
                self.side_storage,
                self.owner,
            );
        } else {
            dispatch_fire_end_handler(objs, self.handler, TraceStop::None, self.pos, FEHF_LASTHIT);
        }
    }
}

impl WeaponEffect {
    /// Gun muzzle flash — random GUNFIRE billboard line + (deferred)
    /// point light (MatrixEffectWeapon.cpp:509-527 / :720-738).
    fn muzzle_flash(
        &self,
        objs: &mut Objects,
        rng: &mut crate::matrix_game::logic::Rnd,
        len: f32,
        width: f32,
    ) {
        use crate::matrix_game::effects::billboard_fx::BillboardLineFx;
        use crate::matrix_game::effects::GameEffect;
        use crate::matrix_lib::three_g::billboard as bb;
        let f = (crate::matrix_game::effects::smoke_and_fire::frnd(rng, 3.0)) as usize;
        let tex = [bb::BBT_GUNFIRE1, bb::BBT_GUNFIRE2, bb::BBT_GUNFIRE3][f.min(2)];
        objs.pending_effects
            .push(GameEffect::BillboardLine(BillboardLineFx::new(
                self.pos,
                self.pos + self.dir * len,
                width,
                0xFFFF_FFFF,
                // LIC lerps ALL channels to 0 — muzzle streaks dim to
                // black as they fade (MatrixEffectWeapon.cpp:523-527),
                // not stay hot white.
                0x0000_0000,
                1000.0,
                bb::TexRef::Bbt(tex),
            )));
        // Muzzle point-light flash — gun r=60 (MatrixEffectWeapon.cpp:
        // 514), turret cannons r=40 (:725). Alpha must be opaque
        // (0xFF303030): the light-mesh shader multiplies by color.a,
        // so alpha 0 drew nothing.
        let r = if self.weapon_type() == WEAPON_CANNON0
            || self.weapon_type() == WEAPON_CANNON1
            || self.weapon_type() == WEAPON_CANNON2
            || self.weapon_type() == WEAPON_CANNON3
        {
            40.0
        } else {
            60.0
        };
        objs.pending_point_lights
            .push(crate::matrix_game::map_static::PendingLight {
                pos: [self.pos.x, self.pos.y, self.pos.z],
                r1: r,
                r2: r,
                c1: 0xFF30_3030,
                c2: 0,
                ttl: 1000.0,
                t1: 1000.0,
            });
    }

    /// Weapon-owned visuals — CLaser / CVolcano / lightening bolt /
    /// repair beam (MatrixEffectWeapon.cpp:850-904 + Repair).
    pub fn draw(
        &self,
        q: &mut crate::matrix_lib::three_g::billboard::BillboardQueue,
        objs: &Objects,
        rng: &mut crate::matrix_game::logic::Rnd,
        paused: bool,
    ) {
        use crate::matrix_game::effects::smoke_and_fire::{frnd, fsrnd};
        use crate::matrix_lib::three_g::billboard as bb;
        if let Some((p0, p1)) = self.beam {
            // CLaser::Draw — flickering alpha.
            let a = if paused {
                240
            } else {
                (frnd(rng, 128.0) + 128.0) as u32
            };
            let color = (a << 24) | 0x00FF_FFFF;
            q.line(p0, p1, 10.0, color, bb::TexRef::Bbt(bb::BBT_LASER));
            q.billboard(p0, 5.0, 0.0, color, bb::TexRef::Bbt(bb::BBT_LASEREND));
        }
        if let Some((born, ang)) = self.plasma_muzzle {
            // Plasma muzzle cone, re-aimed to the live pos/dir each
            // frame (Konus::Modify, MatrixEffectWeapon.cpp:236-241).
            let now = crate::matrix_game::map::current_elapsed_ms();
            let age = (now - born) as f32;
            if age >= 300.0 {
                // Cone expired — clear so the borrow below is free.
            } else {
                let mut k = crate::matrix_game::effects::konus::Konus::new(
                    self.pos,
                    self.dir,
                    10.0,
                    10.0,
                    ang,
                    300.0,
                    true,
                    bb::TexRef::Path(
                        crate::matrix_game::effects::effects_renderer::TEXTURE_PATH_GUN_FIRE,
                    ),
                );
                let _ = k.takt(age);
                k.draw(q);
            }
        }
        if self.volcano_on && self.is_fire() {
            // CVolcano::Draw — cone + one of three GUNFIRE lines.
            let now = crate::matrix_game::map::current_elapsed_ms();
            let ang = std::f32::consts::TAU / 4096.0 * ((now & 4095) as f32);
            let mut k = crate::matrix_game::effects::konus::Konus::new(
                self.pos,
                self.dir,
                5.0,
                5.0,
                ang,
                300.0,
                true,
                bb::TexRef::Path(
                    crate::matrix_game::effects::effects_renderer::TEXTURE_PATH_GUN_FIRE,
                ),
            );
            let _ = k.takt(0.0);
            k.draw(q);
            let idx = if paused { 0 } else { (frnd(rng, 3.0)) as usize };
            let tex = [bb::BBT_GUNFIRE1, bb::BBT_GUNFIRE2, bb::BBT_GUNFIRE3][idx.min(2)];
            q.line(
                self.pos,
                self.pos + self.dir * 15.0,
                10.0,
                0xFFFF_FFFF,
                bb::TexRef::Bbt(tex),
            );
        }
        if let Some(bolt) = &self.bolt {
            bolt.draw(q);
        }
        if self.is_fire() && self.weapon_type == WEAPON_REPAIR && !self.repair_bb.is_empty() {
            // Repair beam — the C++ 10-phase wobble sine bank
            // (MatrixEffectRepair.cpp:270-281).
            let tm = self.repair_time;
            let sins: [f32; 10] = [
                (tm * 0.005 + 0.9).sin(),
                (tm * 0.007 + 0.2).sin(),
                (tm * 0.0073 + 0.3).sin(),
                (tm * 0.0013).sin(),
                (tm * 0.0022 + 0.2).sin(),
                (tm * 0.0017 + 0.8).sin(),
                (tm * 0.0028 + 0.3).sin(),
                (tm * 0.0051 + 0.9).sin(),
                (tm * 0.003 + 0.01).sin(),
                (tm * 0.001 + 0.93).sin(),
            ];
            let seek = self.weapon_dist();
            let amp = self.repair_off_amp;
            // Straight seek spline with growing wobble (p4, :286-303).
            let p4 = [
                self.pos,
                self.pos + self.dir * (seek / 3.0),
                self.pos
                    + self.dir * (seek * 2.0 / 3.0)
                    + Vec3::new(
                        0.4 * amp * sins[3],
                        0.4 * amp * sins[7],
                        0.4 * amp * sins[9],
                    ),
                self.pos + self.dir * seek + Vec3::new(amp * sins[1], amp * sins[4], amp * sins[8]),
            ];
            let straight = crate::matrix_lib::three_g::math3d::Trajectory::new(&p4);
            // Wrap-around spline (p9, :305-357): out to the patient,
            // around it (front·1.4 → side → back → −side → front·1.4,
            // each wobbled by disp = len·0.01), and back to the muzzle.
            let wrap = self.repair_target.and_then(|t| objs.get(t)).map(|o| {
                let gc = o.core().geo_center;
                let radius = o.core().radius;
                let len = (self.pos - gc).length();
                let disp = len * 0.01;
                let p1 = self.pos + self.dir * (len / 3.0);
                let td = (p1 - gc).normalize_or_zero();
                let side = td.cross(Vec3::Z).normalize_or_zero();
                let w = |a: usize, b: usize, c: usize| {
                    Vec3::new(disp * sins[a], disp * sins[b], disp * sins[c])
                };
                let p9 = [
                    self.pos,
                    p1,
                    gc + td * radius * 1.4 + w(0, 1, 2),
                    gc + side * radius + w(2, 3, 4),
                    gc - td * radius + w(4, 5, 6),
                    gc - side * radius + w(6, 7, 8),
                    gc + td * radius * 1.4 + w(3, 6, 9),
                    p1,
                    self.pos,
                ];
                p9
            });
            // 500 ms found-morph between the two shapes (:359-395);
            // losing the patient snaps back to the seek spline (the
            // C++ also morphs out — its target core outlives the loss,
            // ours doesn't, so the out-morph is skipped).
            let spline = match wrap {
                Some(p9) => {
                    let k = (self.repair_change / 500.0).clamp(0.0, 1.0);
                    if k > 0.0 {
                        let mut blended = [Vec3::ZERO; 9];
                        for (i, b) in blended.iter_mut().enumerate() {
                            let t = i as f32 / 8.0;
                            let s = straight.calc_point(t);
                            *b = s + (p9[i] - s) * (1.0 - k);
                        }
                        crate::matrix_lib::three_g::math3d::Trajectory::new(&blended)
                    } else {
                        crate::matrix_lib::three_g::math3d::Trajectory::new(&p9)
                    }
                }
                None => straight,
            };
            // The C++ draws ONLY the traveling glints — there is no
            // persistent solid beam (MatrixEffectRepair.cpp:60-67);
            // glint tails span t-0.06 (:388-395).
            for (t, _) in &self.repair_bb {
                let a = ((t / 0.2).clamp(0.0, 1.0) * 255.0) as u32;
                let p0 = spline.calc_point((t - 0.06).max(0.0));
                let p1 = spline.calc_point(*t);
                q.line(
                    p0,
                    p1,
                    4.0,
                    (a << 24) | 0x00FF_FFFF,
                    bb::TexRef::Bbt(bb::BBT_REPGLOWEND),
                );
            }
            // Muzzle glow (SWeaponRepairData, MatrixRobot.cpp:55-84) —
            // a small pulsing sprite at the emitter.
            let pulse = 4.0 + (tm * 0.01).sin().abs() * 2.0;
            q.billboard(
                self.pos,
                pulse,
                0.0,
                0xB0FF_FFFF,
                bb::TexRef::Bbt(bb::BBT_REPGLOWEND),
            );
            let _ = fsrnd;
        }
    }
}

/// Drive one weapon's `Takt` with the registry checkout dance. The
/// weapon may damage other objects / spawn projectiles while it runs,
/// so its own slot is temporarily empty (mirrors take-the-box on the
/// object arena).
pub fn weapon_takt(
    objs: &mut Objects,
    id: WeaponId,
    step: f32,
    map: &GameMap,
    rng: &mut crate::matrix_game::logic::Rnd,
) {
    let Some(mut w) = objs.weapons.take(id) else {
        return;
    };
    w.takt(id, step, map, objs, rng);
    objs.weapons.put_back(id, w);
}

/// Port of `CMatrixEffectWeapon::WeaponHit` (MatrixEffectWeapon.cpp:
/// 14-64) — the projectile / area-effect hit callback. `dist2` is
/// `CMatrixEffect::m_Dist2`, the squared distance from the shot origin
/// (the range check for ballistic weapons).
///
/// Weapon-ref release on `FEHF_LASTHIT` is NOT done here — the effect
/// driver releases when the effect dies (see module doc).
pub fn weapon_hit(
    objs: &mut Objects,
    weapon: WeaponId,
    hiti: TraceStop,
    pos: Vec3,
    flags: u32,
    dist2: f32,
) {
    let Some(w) = objs.weapons.get(weapon) else {
        return;
    };
    let wtype = w.weapon_type;
    let wdist = w.weapon_dist();
    let wdir = w.dir;
    let wpos = w.pos;
    let side = w.side_storage;
    let owner = w.owner;
    let handler = w.handler;

    let mut give_damage = true;
    if matches!(
        wtype,
        WEAPON_HOMING_MISSILE | WEAPON_GUN | WEAPON_CANNON0 | WEAPON_CANNON1 | WEAPON_CANNON3
    ) && wdist * wdist < dist2
    {
        give_damage = false;
    }

    let mut odead = false;
    let mut flags_add = 0;
    let mut hiti = hiti;

    if give_damage {
        if let Some(id) = hiti.object() {
            let dir = if wtype == WEAPON_BIGBOOM {
                (pos - wpos).normalize_or_zero()
            } else {
                wdir
            };
            odead = objs.apply_damage(id, wtype, pos, dir, side, owner);
            if !odead {
                let is_robot = objs
                    .get(id)
                    .map(|o| matches!(o.core().obj_type, ObjectType::RobotAi))
                    .unwrap_or(false);
                if is_robot {
                    flags_add = FEHF_DAMAGE_ROBOT;
                }
            }
        }
    }
    // Water splash (MatrixEffectWeapon.cpp:50-53).
    if hiti == TraceStop::Water && wtype != WEAPON_FLAMETHROWER && wtype != WEAPON_BIGBOOM {
        let ang = (pos.x + pos.y).fract() * std::f32::consts::PI;
        objs.queue_snd_at("splash", pos);
        objs.pending_effects
            .push(crate::matrix_game::effects::GameEffect::Konus(
                crate::matrix_game::effects::konus::Konus::new_splash(
                    pos,
                    Vec3::Z,
                    10.0,
                    5.0,
                    ang,
                    1000.0,
                    true,
                    crate::matrix_lib::three_g::billboard::TexRef::Path(
                        crate::matrix_game::effects::effects_renderer::TEXTURE_PATH_SPLASH,
                    ),
                ),
            ));
    }

    if odead {
        hiti = TraceStop::None;
    }
    dispatch_fire_end_handler(objs, handler, hiti, pos, flags | flags_add);
}

/// Dispatch the `FIRE_END_HANDLER`. Robot side ports `robot_weapon_hit`
/// → `CMatrixRobotAI::HitTo` (MatrixRobot.cpp:4011-4050); cannon side
/// ports `CMatrixCannon::FireHandler` (MatrixObjectCannon.cpp:29-49,
/// minus the ref-protection dance which the slab makes unnecessary).
pub fn dispatch_fire_end_handler(
    objs: &mut Objects,
    handler: WeaponHandler,
    hiti: TraceStop,
    pos: Vec3,
    flags: u32,
) {
    match handler {
        WeaponHandler::None => {}
        WeaponHandler::Robot(id) => {
            // HitTo reads the victim's id/side/type — fetch before the
            // mutable borrow of the shooter.
            let hit_info = hiti
                .object()
                .and_then(|tid| objs.get(tid).map(|o| (tid, o.side(), o.core().obj_type)));
            if objs.get(id).is_none() {
                // The shooter's box is checked out (its own weapon takt
                // fired a hitscan) — queue the notice; the robot drains
                // it right after `weapons_logic_takt`.
                if objs.is_checked_out(id) {
                    objs.pending_hit_notices.push((id, hit_info, pos));
                }
                return;
            }
            // hit_to mutates the victim's env too — use the
            // take-the-box pattern.
            let Some(mut boxed) = objs.take_obj(id) else {
                return;
            };
            if matches!(boxed.core().obj_type, ObjectType::RobotAi) {
                let r: &mut crate::matrix_game::robot::Robot = unsafe {
                    &mut *(boxed.as_mut() as *mut dyn MapStatic
                        as *mut crate::matrix_game::robot::Robot)
                };
                // `m_CurrState != ROBOT_DIP` gate (robot_weapon_hit,
                // MatrixRobot.cpp:4011-4023) — dead shooters don't
                // update their env from in-flight shots landing.
                if r.state != crate::matrix_game::robot::RobotState::Dip {
                    r.hit_to(hit_info, pos, objs);
                }
            }
            objs.put_obj(id, boxed);
        }
        WeaponHandler::Cannon(id) => {
            if flags & FEHF_DAMAGE_ROBOT == 0 {
                return;
            }
            let Some(obj) = objs.get_mut(id) else { return };
            if !matches!(obj.core().obj_type, ObjectType::Cannon) {
                return;
            }
            let c: &mut crate::matrix_game::object_cannon::Cannon = unsafe {
                &mut *(obj as *mut dyn MapStatic as *mut crate::matrix_game::object_cannon::Cannon)
            };
            // "попадание! обновим тайминг косой стрельбы"
            c.time_from_fire = crate::matrix_game::object_cannon::CANNON_TIME_FROM_FIRE;
        }
    }
}
