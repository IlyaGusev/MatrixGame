//! Port of `CMatrixMapStatic` (MatrixMapStatic.{cpp,hpp}).
//!
//! C++ keeps every live map object in a single intrusive linked list rooted
//! at `m_FirstLogicTemp` and walked by `ProceedLogic`. Objects opt in with
//! `AddLT()` (only when their subclass has per-takt work — `BEHF_STATIC`
//! palms never AddLT, for example). `ProceedLogic` snapshots the next
//! pointer before dispatching so a callee may remove itself from the list
//! during its own takt without breaking the walk
//! (`MatrixMapStatic.cpp:349`).
//!
//! Rust equivalent: a generational-index arena. `ObjectId` lookups returning
//! `None` reproduce the `SObjectCore::m_Object == NULL` tombstone that the
//! C++ effects use to detect owner death. The logic-temp list is stored as
//! `Option<ObjectId>` prev/next fields on each slot.
//!
//! Scope A: base class + tick driver. Concrete subclasses land later; the
//! `MapStatic` trait here is what they implement.
//!
//! Scope B adds `CMatrixMapObject` (see `object.rs`) — the first live
//! subclass wired through `ProceedLogic`.

use glam::{Mat4, Vec2, Vec3};

use crate::matrix_game::common::{
    TRACE_BUILDING, TRACE_CANNON, TRACE_FLYER, TRACE_OBJECT, TRACE_ROBOT, TRACE_SKIP_INVISIBLE,
};
use crate::matrix_game::config::{BuildingDamages, ObjectDamages};
use crate::matrix_game::logic::Rnd;

// ── Resource-change bits (MR_*) (MatrixMapStatic.hpp:17-25) ──────────────
//
// Subclasses set these via `rchange_set` to mark which derived resources
// need recomputing; `RNeed(mask)` rebuilds the intersection of requested +
// dirty bits and clears them. Default at construction is `ALL` (matches
// `m_RChange(0xffffffff)`, MatrixMapStatic.hpp:346).

pub const MR_GRAPH: u32 = 1 << 0;
pub const MR_MATRIX: u32 = 1 << 1;
pub const MR_POS: u32 = 1 << 2;
pub const MR_ROTATE: u32 = 1 << 3;
pub const MR_SHADOW_STENCIL: u32 = 1 << 4;
pub const MR_SHADOW_PROJ_GEOM: u32 = 1 << 6;
pub const MR_SHADOW_PROJ_TEX: u32 = 1 << 7;
pub const MR_MINIMAP: u32 = 1 << 8;
pub const MR_ALL: u32 = 0xFFFF_FFFF;

// ── Object state bits (MatrixMapStatic.hpp:46-85) ────────────────────────
//
// The low bits are shared across all subclasses; bits 10..22 are
// subclass-overlayed (ROBOT_FLAG_*, BUILDING_*, cannon, mesh). We expose
// the overlays as `#[allow(dead_code)]` constants so subclass ports can
// reference them by their original names.

pub const OBJECT_STATE_ABLAZE: u32 = 1 << 0;
pub const OBJECT_STATE_SHORTED: u32 = 1 << 1;
pub const OBJECT_STATE_INVISIBLE: u32 = 1 << 2;
pub const OBJECT_STATE_INTERFACE: u32 = 1 << 3;
pub const OBJECT_STATE_INVULNERABLE: u32 = 1 << 3;
pub const OBJECT_STATE_SHADOW_SPECIAL: u32 = 1 << 4;
pub const OBJECT_STATE_TRACE_INVISIBLE: u32 = 1 << 5;
pub const OBJECT_STATE_DIP: u32 = 1 << 6;

// Mesh-only (CMatrixMapObject).
#[allow(dead_code)]
pub const OBJECT_STATE_BURNED: u32 = 1 << 10;
#[allow(dead_code)]
pub const OBJECT_STATE_EXPLOSIVE: u32 = 1 << 11;
#[allow(dead_code)]
pub const OBJECT_STATE_NORMALIZENORMALS: u32 = 1 << 12;
#[allow(dead_code)]
pub const OBJECT_STATE_SPECIAL: u32 = 1 << 13;
#[allow(dead_code)]
pub const OBJECT_STATE_TERRON_EXPL: u32 = 1 << 14;
#[allow(dead_code)]
pub const OBJECT_STATE_TERRON_EXPL1: u32 = 1 << 15;
#[allow(dead_code)]
pub const OBJECT_STATE_TERRON_EXPL2: u32 = 1 << 16;

// Robot-only (`ROBOT_FLAG_*` — MatrixMapStatic.hpp:62-77).
/// `ROBOT_FLAG_ONWATER` (`SETBIT(14)`, MatrixMapStatic.hpp:76). Set by
/// `Z_From_Pos` (MatrixObjectRobot.cpp:282) whenever the terrain height
/// at the robot's XY is below `WATER_LEVEL`. Read by `Seek`
/// (MatrixRobot.cpp:2420) to apply the chassis `m_SpeedWaterCorr`.
pub const ROBOT_FLAG_ONWATER: u32 = 1 << 14;
/// `ROBOT_FLAG_COLLISION` — set by LowLevelMove when a collision
/// correction kicked in this tick (MatrixRobot.cpp:2623).
#[allow(dead_code)]
pub const ROBOT_FLAG_COLLISION: u32 = 1 << 15;

// Building-only (overlays bits 10..=12 from MatrixMapStatic.hpp:88-90).
/// `BUILDING_NEW_INCOME` — a resource tick just fired; Takt emits a
/// billboard-score effect on the next frame (MatrixObjectBuilding.cpp:355).
#[allow(dead_code)]
pub const OBJECT_STATE_BUILDING_NEW_INCOME: u32 = 1 << 10;
/// `BUILDING_SPAWNBOT` — base door is open and a robot is being
/// delivered; prevents `Close()` mid-spawn (MatrixObjectBuilding.hpp:269-270).
#[allow(dead_code)]
pub const OBJECT_STATE_BUILDING_SPAWNBOT: u32 = 1 << 11;
/// `BUILDING_CAPTURE_IN_PROGRESS` — a robot of another side is
/// capturing the base (MatrixObjectBuilding.hpp:191).
#[allow(dead_code)]
pub const OBJECT_STATE_BUILDING_CAPTURE_IN_PROGRESS: u32 = 1 << 12;

/// `EObjectType` (MatrixMapStatic.hpp:29-39). Discriminants match the C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ObjectType {
    Empty = 0,
    MapObject = 2,
    RobotAi = 3,
    Building = 4,
    Cannon = 5,
    Flyer = 6,
}

/// Ports `SObjectCore` (MatrixMapStatic.hpp:133-199). Ref-counting is
/// dropped: the `ObjectId → Option<&Slot>` lookup reproduces the
/// `m_Object == NULL` tombstone (see module doc). `m_GeoCenter` /
/// `m_Radius` / `m_TerainColor` are populated by subclass `join_to_group`
/// (not implemented in scope A).
#[derive(Debug, Clone)]
pub struct ObjectCore {
    pub matrix: Mat4,         // m_Matrix
    pub inv_matrix: Mat4,     // m_IMatrix
    pub radius: f32,          // m_Radius
    pub geo_center: Vec3,     // m_GeoCenter
    pub obj_type: ObjectType, // m_Type
    pub terrain_color: u32,   // m_TerainColor  (0xFFFFFFFF default)
}

impl Default for ObjectCore {
    fn default() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
            inv_matrix: Mat4::IDENTITY,
            radius: 0.0,
            geo_center: Vec3::ZERO,
            obj_type: ObjectType::Empty,
            terrain_color: 0xFFFF_FFFF,
        }
    }
}

/// The `CMatrixMapStatic` contract. Subclasses implement this trait; the
/// [`Objects`] arena stores `Box<dyn MapStatic>`.
///
/// Scope A: only `takt` + `logic_takt` + `r_need` are hit by the tick
/// loop. The remaining virtuals (`pick`, `draw`, `calc_bounds`, …) are
/// here as default-noop methods so concrete subclasses can implement
/// them incrementally — matching the `= 0` pure virtuals at
/// `MatrixMapStatic.hpp:459-478`.
pub trait MapStatic {
    // --- Core accessors (inline getters on CMatrixMapStatic) ---
    fn core(&self) -> &ObjectCore;
    fn core_mut(&mut self) -> &mut ObjectCore;
    fn rchange(&self) -> u32;
    fn rchange_set(&mut self, bits: u32); // RChange(zn)  — OR-in
    fn rchange_clear(&mut self, bits: u32); // RNoNeed(zn)  — AND-NOT
    fn object_state(&self) -> u32;
    fn object_state_set(&mut self, bits: u32);
    fn object_state_clear(&mut self, bits: u32);

    // --- Ablaze / Shorted TTLs (MatrixMapStatic.hpp:254-263) ---
    fn ablaze_ttl(&self) -> i32;
    fn set_ablaze_ttl(&mut self, ttl: i32);
    fn shorted_ttl(&self) -> i32;
    fn set_shorted_ttl(&mut self, ttl: i32);

    // --- Pure virtuals (MatrixMapStatic.hpp:457-478) ---
    //
    // `takt` is the per-frame graphic takt; `logic_takt` is the fixed-step
    // 10ms takt. `r_need` rebuilds requested resources and clears the
    // matching MR_* bits in `m_RChange`. The `rng` is `g_MatrixMap->Rnd`
    // — a shared stream every subclass may advance; `objs` is the arena
    // with the current object's slot temporarily empty (see
    // [`Objects::proceed_logic`] / [`Objects::graphic_takt`]). Both are
    // passed explicitly so Takt/LogicTakt aren't hiding a thread-local
    // global.
    fn r_need(&mut self, need: u32);
    fn takt(&mut self, cms: i32, rng: &mut Rnd, objs: &mut Objects);
    fn logic_takt(&mut self, cms: i32, rng: &mut Rnd, objs: &mut Objects);

    // Default-noop virtuals so subclasses can opt in as they're ported.
    fn before_draw(&mut self) {}
    fn free_dynamic_resources(&mut self) {}
    fn side(&self) -> i32 {
        -1
    }
    fn need_repair(&self) -> bool {
        false
    }

    /// Port of `virtual bool Pick(orig, dir, float *outt)`
    /// (MatrixMapStatic.hpp:462). Ray / object intersection. The
    /// default is a sphere test against `core().geo_center` +
    /// `core().radius` — a conservative approximation for subclasses
    /// that haven't ported their mesh-level pick. Subclasses (Building,
    /// Robot) can override with a tighter bounds test once per-instance
    /// mesh data is available.
    ///
    /// Returns the ray parameter `t` (distance along `dir` from
    /// `origin`) for the nearest hit, or `None` if the ray misses.
    /// `dir` must be normalized for the returned `t` to be a distance.
    fn pick(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        ray_sphere_pick(self.core().geo_center, self.core().radius, origin, dir)
    }

    /// Port of `virtual bool Damage(EWeapon, pos, dir, attacker_side,
    /// attacker)` (MatrixMapStatic.hpp:464). Returns true iff the
    /// damage caused *this* object to be removed from play (C++
    /// `Damage` returns `true` only when `Init` reset mid-call, per
    /// comment at MatrixObject.cpp:1572). Default: ignore.
    ///
    /// `objs` is the arena sans this object's slot — same contract as
    /// `takt`/`logic_takt`. Callees use it to enroll themselves into
    /// the logic-temp list (AddLT) or spawn effects.
    #[allow(clippy::too_many_arguments)]
    fn damage(
        &mut self,
        _weap: crate::matrix_game::effects::weapon::Weapon,
        _pos: Vec3,
        _dir: Vec3,
        _attacker_side: i32,
        _attacker: Option<ObjectId>,
        _self_id: ObjectId,
        _objs: &mut Objects,
    ) -> bool {
        false
    }
}

// ── Helper predicates on &dyn MapStatic (matches `IsRobot()` etc.) ─────

#[allow(dead_code)]
pub fn is_robot(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::RobotAi)
}

#[allow(dead_code)]
pub fn is_building(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::Building)
}

#[allow(dead_code)]
pub fn is_cannon(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::Cannon)
}

#[allow(dead_code)]
pub fn is_flyer(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::Flyer)
}

#[allow(dead_code)]
pub fn is_map_object(obj: &dyn MapStatic) -> bool {
    matches!(obj.core().obj_type, ObjectType::MapObject)
}

/// Port of `CMatrixMapStatic::FitToMask` (MatrixMap.hpp:73-81). Returns
/// true when this object's type (and, for robots/cannons/buildings, its
/// live state) satisfies a `TRACE_*` mask.
///
/// The C++ guards on `IsLiveRobot` / `IsLiveCannon` / `IsLiveBuilding` —
/// those check the per-subclass state machine (`CurrState != ROBOT_DIP`
/// etc.). Those subclasses aren't ported yet, so we treat every
/// robot/cannon/building as "live" (the mask check alone decides
/// inclusion). When the state-machines land, gate this on an
/// `is_live()` trait method.
///
/// `TRACE_SKIP_INVISIBLE` additionally excludes objects with the
/// `OBJECT_STATE_TRACE_INVISIBLE` bit set — ported here because
/// `FindObjects` consumers set the flag in the mask.
pub fn fit_to_mask(obj: &dyn MapStatic, mask: u32) -> bool {
    // `TRACE_SKIP_INVISIBLE`: objects opting out of trace visibility
    // (MatrixMapStatic.hpp:51, set by OTP_INVLOGIC="1") are filtered.
    if mask & TRACE_SKIP_INVISIBLE != 0 && obj.object_state() & OBJECT_STATE_TRACE_INVISIBLE != 0 {
        return false;
    }
    match obj.core().obj_type {
        ObjectType::RobotAi => mask & TRACE_ROBOT != 0,
        ObjectType::Cannon => mask & TRACE_CANNON != 0,
        ObjectType::Building => mask & TRACE_BUILDING != 0,
        ObjectType::MapObject => mask & TRACE_OBJECT != 0,
        ObjectType::Flyer => mask & TRACE_FLYER != 0,
        ObjectType::Empty => false,
    }
}

/// Return from a `find_objects` callback — matches the bool contract in
/// `ENUM_OBJECTS2D` (MatrixMap.hpp:84) where `true` means "keep
/// searching", `false` means "stop".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Continue,
    Break,
}

/// Ray-sphere intersection. Returns the nearest positive `t` (ray
/// parameter, equal to distance when `dir` is a unit vector) at which
/// `origin + t*dir` intersects the sphere centered at `center` with
/// radius `radius`. `None` if the ray misses or both roots are behind
/// the origin. Used as the default `MapStatic::pick` body for
/// subclasses that don't override with a per-mesh test.
pub fn ray_sphere_pick(center: Vec3, radius: f32, origin: Vec3, dir: Vec3) -> Option<f32> {
    if radius <= 0.0 {
        return None;
    }
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    // Two roots: -b - sqrt_d (entering) and -b + sqrt_d (exiting). We
    // want the nearest positive t — the entering hit unless the ray
    // origin is already inside the sphere.
    let t0 = -b - sqrt_d;
    let t1 = -b + sqrt_d;
    if t0 >= 0.0 {
        Some(t0)
    } else if t1 >= 0.0 {
        Some(t1)
    } else {
        None
    }
}

// ── Arena + logic-temp list ─────────────────────────────────────────────

/// Stable handle into [`Objects`]. A generational index so reused slots
/// don't silently alias a freed id — equivalent to checking
/// `core->m_Object != NULL` in the C++ ref-counted world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId {
    index: u32,
    generation: u32,
}

struct Slot {
    generation: u32,
    obj: Option<Box<dyn MapStatic>>,
    in_lt: bool,
    prev_lt: Option<ObjectId>,
    next_lt: Option<ObjectId>,
}

/// Arena of all map-static objects + the intrusive logic-temp list.
/// Ports the `static` members + `m_FirstLogicTemp/m_NextLogicTemp` linkage
/// from `CMatrixMapStatic` (MatrixMapStatic.hpp:265-268, .cpp:60-61).
pub struct Objects {
    slots: Vec<Slot>,
    free: Vec<u32>,
    first_lt: Option<ObjectId>,
    last_lt: Option<ObjectId>,

    /// Ports `g_MatrixMap->m_NextLogicObject` — the snapshot of the next
    /// list node taken before each `static_takt` dispatch, so a callee
    /// can `del_lt(self)` without breaking the walk
    /// (MatrixMapStatic.cpp:349).
    next_logic_object: Option<ObjectId>,

    /// Ports the `ms != g_MatrixMap->GetPlayerSide()->GetArcadedObject()`
    /// guard in `ProceedLogic` (MatrixMapStatic.cpp:350). Filled in when
    /// sides land; `None` means the guard is inactive.
    pub arcaded_object: Option<ObjectId>,

    /// Port of `g_Config.m_ObjectDamages` (MatrixConfig.hpp:574). The
    /// per-weapon damage table used by MapObject's Damage branches.
    /// Lives on `Objects` because it's a shared world-level resource
    /// that damage callees need from inside the take-the-box pattern.
    pub object_damages: ObjectDamages,
    /// Port of `g_Config.m_BuildingDamages` + `m_BuildingHitPoints`
    /// (MatrixConfig.cpp:507-535). Same shape as `object_damages` plus
    /// a per-EBuildingType max-HP vector.
    pub building_damages: BuildingDamages,
}

impl Default for Objects {
    fn default() -> Self {
        Self::new()
    }
}

impl Objects {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            first_lt: None,
            last_lt: None,
            next_logic_object: None,
            arcaded_object: None,
            object_damages: ObjectDamages::default(),
            building_damages: BuildingDamages::default(),
        }
    }

    /// Insert `obj` into the arena and return its handle. The ctor
    /// equivalent for `CMatrixMapStatic`-derived objects: after this
    /// returns, the subclass may call `add_lt` to opt into logic takts.
    pub fn spawn(&mut self, obj: Box<dyn MapStatic>) -> ObjectId {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.obj = Some(obj);
            slot.in_lt = false;
            slot.prev_lt = None;
            slot.next_lt = None;
            ObjectId {
                index,
                generation: slot.generation,
            }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 1,
                obj: Some(obj),
                in_lt: false,
                prev_lt: None,
                next_lt: None,
            });
            ObjectId {
                index,
                generation: 1,
            }
        }
    }

    /// Destroy the object at `id`. Ports `~CMatrixMapStatic`
    /// (MatrixMapStatic.cpp:87-104): removes from logic-temp list and
    /// releases the storage. Subsequent lookups return `None`
    /// (the tombstone).
    pub fn remove(&mut self, id: ObjectId) {
        if !self.is_valid(id) {
            return;
        }
        self.del_lt(id);
        let slot = &mut self.slots[id.index as usize];
        slot.obj = None;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(id.index);
    }

    pub fn is_valid(&self, id: ObjectId) -> bool {
        self.slots
            .get(id.index as usize)
            .map(|s| s.generation == id.generation && s.obj.is_some())
            .unwrap_or(false)
    }

    /// Like [`is_valid`] but returns true even when the slot's box is
    /// temporarily checked out (take-the-box pattern in
    /// `proceed_logic`/`graphic_takt`/`apply_damage`). List-membership
    /// ops (`add_lt`/`del_lt`/`in_lt`) must use this so a takt body
    /// can self-`add_lt` via the passed `&mut Objects` — matching the
    /// C++ where `this->AddLT()` works from inside one of its own
    /// methods. External read access (`get`/`get_mut`) keeps the
    /// stricter check to preserve the "object not accessible" tombstone.
    fn gen_matches(&self, id: ObjectId) -> bool {
        self.slots
            .get(id.index as usize)
            .map(|s| s.generation == id.generation)
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read-only view of an object. `None` is the `m_Object == NULL`
    /// tombstone — callers must handle it.
    pub fn get(&self, id: ObjectId) -> Option<&dyn MapStatic> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.obj.as_deref()
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Option<&mut (dyn MapStatic + 'static)> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.obj.as_deref_mut()
    }

    /// Visit each live object in the arena.
    pub fn for_each_live(&self, mut f: impl FnMut(ObjectId, &dyn MapStatic)) {
        for (i, slot) in self.slots.iter().enumerate() {
            let Some(obj) = slot.obj.as_deref() else {
                continue;
            };
            let id = ObjectId {
                index: i as u32,
                generation: slot.generation,
            };
            f(id, obj);
        }
    }

    /// Ports `InLT` (MatrixMapStatic.hpp:440). Works even while the
    /// object's box is checked out by a takt driver — the list
    /// membership flag lives on the slot, not in the box.
    pub fn in_lt(&self, id: ObjectId) -> bool {
        self.slots
            .get(id.index as usize)
            .map(|s| s.generation == id.generation && s.in_lt)
            .unwrap_or(false)
    }

    /// Ports `AddLT` (MatrixMapStatic.hpp:441). Idempotent. Uses
    /// `gen_matches` (not `is_valid`) so self-enroll during a takt
    /// body works even though the slot's box is currently checked out.
    pub fn add_lt(&mut self, id: ObjectId) {
        if !self.gen_matches(id) || self.in_lt(id) {
            return;
        }
        // LIST_ADD(this, m_FirstLogicTemp, m_LastLogicTemp,
        //          m_PrevLogicTemp, m_NextLogicTemp)
        let old_last = self.last_lt;
        {
            let slot = &mut self.slots[id.index as usize];
            slot.in_lt = true;
            slot.prev_lt = old_last;
            slot.next_lt = None;
        }
        if let Some(prev) = old_last {
            self.slots[prev.index as usize].next_lt = Some(id);
        } else {
            self.first_lt = Some(id);
        }
        self.last_lt = Some(id);
    }

    /// Ports `DelLT` (MatrixMapStatic.hpp:442). Idempotent.
    pub fn del_lt(&mut self, id: ObjectId) {
        if !self.in_lt(id) {
            return;
        }
        let (prev, next) = {
            let slot = &mut self.slots[id.index as usize];
            let pn = (slot.prev_lt, slot.next_lt);
            slot.in_lt = false;
            slot.prev_lt = None;
            slot.next_lt = None;
            pn
        };
        // If we're removing the snapshotted next cursor mid-walk, advance
        // it so ProceedLogic still lands on a live node.
        if self.next_logic_object == Some(id) {
            self.next_logic_object = next;
        }
        if let Some(p) = prev {
            self.slots[p.index as usize].next_lt = next;
        } else {
            self.first_lt = patch_head(self.first_lt, id, next);
        }
        if let Some(n) = next {
            self.slots[n.index as usize].prev_lt = prev;
        } else {
            self.last_lt = patch_head(self.last_lt, id, prev);
        }
    }

    /// Port of `CMatrixMapStatic::ProceedLogic` (MatrixMapStatic.cpp:338-362).
    /// Walks the logic-temp list, snapshotting the next link before each
    /// dispatch so a callee may remove itself (or the current tail) from
    /// the list without breaking iteration.
    ///
    /// Dispatch uses the "take the box out" pattern: we `take()` the
    /// object's `Box<dyn MapStatic>` out of its slot, invoke
    /// [`static_takt`] with `&mut Objects` (the arena sans that slot),
    /// then return the box. This lets takt bodies issue arena queries
    /// like [`Objects::find_objects`] without re-entrant borrow errors.
    /// Tombstone lookups of the taken id return `None` for the duration
    /// of the call — consistent with the C++ `m_Object == NULL` check.
    pub fn proceed_logic(&mut self, takts: i32, rng: &mut Rnd) {
        let mut cursor = self.first_lt;
        while let Some(id) = cursor {
            // Snapshot *before* dispatch, same line as the original:
            //   g_MatrixMap->m_NextLogicObject = ms->m_NextLogicTemp;
            let snapshotted_next = self.slots[id.index as usize].next_lt;
            self.next_logic_object = snapshotted_next;

            if Some(id) != self.arcaded_object {
                static_takt(self, id, takts, rng);
            }

            cursor = self.next_logic_object;
        }
        self.next_logic_object = None;
    }

    /// Used by tests + future sides code. Iterates the logic-temp list
    /// without invoking tiks — snapshot-safe (mid-iteration mutation via
    /// `del_lt` is undefined for this method; use `proceed_logic` for
    /// that contract).
    pub fn iter_logic(&self) -> LogicIter<'_> {
        LogicIter {
            objs: self,
            cursor: self.first_lt,
        }
    }

    /// All live objects (arena order). Cheap `O(slots)`; skips freed
    /// slots and generation mismatches.
    pub fn iter_live(&self) -> impl Iterator<Item = ObjectId> + '_ {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.obj.as_ref().map(|_| ObjectId {
                index: i as u32,
                generation: s.generation,
            })
        })
    }

    /// Port of `CMatrixMap::FindObjects` (MatrixMap.cpp:2617-2814). Enumerates
    /// live objects overlapping a circle of radius `radius` at `pos`, filtered
    /// by `mask` ([`fit_to_mask`]). The `oscale` factor scales the candidate
    /// object's own radius for the distance test — the C++ uses it to accept
    /// "near-misses" or tighter-fit checks depending on caller intent.
    ///
    /// `skip` excludes one object (typically `this` in the caller's context).
    ///
    /// `visit(center, id) -> Control::Continue|Break` receives each hit in
    /// arena order; the callback's matching-bool return mirrors the C++
    /// `callback` return type at MatrixMap.hpp:84.
    ///
    /// Differences from the original:
    /// * No spatial group index yet — we linear-scan all live objects.
    ///   For atoll object counts this is trivial; when a group-indexed
    ///   variant lands it will share this contract.
    /// * `m_IntersectFlagFindObjects` re-visit guard isn't needed in the
    ///   linear path (each slot visited exactly once).
    /// * Flyer's `GetCarryingRobot` promotion (MatrixMap.cpp:2699-2710)
    ///   is deferred until the flyer subclass lands.
    pub fn find_objects(
        &self,
        pos: Vec2,
        radius: f32,
        oscale: f32,
        mask: u32,
        skip: Option<ObjectId>,
        mut visit: impl FnMut(Vec2, ObjectId) -> Control,
    ) -> bool {
        let mut hit = false;
        for (i, slot) in self.slots.iter().enumerate() {
            let obj = match slot.obj.as_deref() {
                Some(o) => o,
                None => continue,
            };
            let id = ObjectId {
                index: i as u32,
                generation: slot.generation,
            };
            if Some(id) == skip {
                continue;
            }
            if !fit_to_mask(obj, mask) {
                continue;
            }
            let center3 = obj.core().geo_center;
            let center = Vec2::new(center3.x, center3.y);
            // MatrixMap.cpp:2704 — `dist = length(center.xy - pos) - radius * oscale`,
            // hit when `dist < radius` (the candidate's own radius is
            // absorbed into the per-object test in FindObjectAny, so
            // here we use `obj.radius * oscale` as the subtract term).
            let dist = (center - pos).length() - obj.core().radius * oscale;
            if dist >= radius {
                continue;
            }
            hit = true;
            if visit(center, id) == Control::Break {
                return hit;
            }
        }
        hit
    }

    /// Predicate-only variant — matches the C++ `FindObjects(..., NULL, 0)`
    /// pattern used by BEHF_SENS at MatrixObject.cpp:1501 ("is anything
    /// in the radius?"). Stops as soon as a hit is found.
    pub fn any_object_in_radius(
        &self,
        pos: Vec2,
        radius: f32,
        oscale: f32,
        mask: u32,
        skip: Option<ObjectId>,
    ) -> bool {
        self.find_objects(pos, radius, oscale, mask, skip, |_, _| Control::Break)
    }

    /// Arena ray-cast. Scans live objects matching `mask` and returns
    /// the nearest hit `(id, t)` — `t` measured along `dir` from
    /// `origin`. Ports the "iterate objects, call Pick, keep nearest"
    /// fragment that CMatrixMap::Trace runs on the `TRACE_*OBJECT`
    /// bits (MatrixMapTrace.cpp — linear scan variant; group-indexed
    /// path arrives with the spatial index port).
    ///
    /// `skip` excludes one id (typically the caller's own). `dir`
    /// should be unit-length for the returned `t` to be a distance.
    pub fn pick_object(
        &self,
        origin: Vec3,
        dir: Vec3,
        mask: u32,
        skip: Option<ObjectId>,
    ) -> Option<(ObjectId, f32)> {
        let mut best: Option<(ObjectId, f32)> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            let obj = match slot.obj.as_deref() {
                Some(o) => o,
                None => continue,
            };
            let id = ObjectId {
                index: i as u32,
                generation: slot.generation,
            };
            if Some(id) == skip || !fit_to_mask(obj, mask) {
                continue;
            }
            if let Some(t) = obj.pick(origin, dir) {
                if t < 0.0 {
                    continue;
                }
                match best {
                    Some((_, bt)) if bt <= t => {}
                    _ => best = Some((id, t)),
                }
            }
        }
        best
    }

    /// External entry point for `CMatrixMapStatic::Damage(...)`. Uses the
    /// take-the-box pattern so `damage()` receives `&mut Objects` as the
    /// arena-sans-self, allowing the callee to `add_lt` itself (the
    /// BEHF_BURN enrollment) or query other objects.
    ///
    /// Returns whatever the subclass `damage` returned — `true` means
    /// "this object was reset / removed mid-call" (the only case where
    /// the original returns true; see MatrixObject.cpp:1572).
    pub fn apply_damage(
        &mut self,
        target: ObjectId,
        weap: crate::matrix_game::effects::weapon::Weapon,
        pos: Vec3,
        dir: Vec3,
        attacker_side: i32,
        attacker: Option<ObjectId>,
    ) -> bool {
        let mut boxed = match self.slots.get_mut(target.index as usize).and_then(|s| {
            if s.generation == target.generation {
                s.obj.take()
            } else {
                None
            }
        }) {
            Some(b) => b,
            None => return false,
        };
        let result = boxed.damage(weap, pos, dir, attacker_side, attacker, target, self);
        let slot = &mut self.slots[target.index as usize];
        if slot.generation == target.generation && slot.obj.is_none() {
            slot.obj = Some(boxed);
        }
        result
    }

    /// Ports the per-object pass inside `CMatrixMapStatic::SortEndGraphicTakt`
    /// (MatrixMapStatic.cpp:755-765). Calls `takt(cms, rng)` on every
    /// live object. Visibility culling (the `objects_left/objects_rite`
    /// sorted window) is deferred until the vis-list arrives; for now
    /// we dispatch to all live slots — Takt bodies are cheap no-ops for
    /// BEHF_STATIC objects and the atoll-size object count is trivial.
    ///
    /// Unlike `proceed_logic`, this does NOT snapshot a `next` cursor:
    /// the C++ graphic takt walks a fixed index range filled during
    /// sort, so in-loop mutations (via `AddObject`/`DIP`) don't affect
    /// the current pass. Our `iter_live` is similarly snapshotted —
    /// `Vec` indices collected up-front, so spawning mid-takt is safe.
    pub fn graphic_takt(&mut self, cms: i32, rng: &mut Rnd) {
        // Collect IDs first so mutations inside `takt` (e.g. `add_lt`,
        // spawn) can't skip or revisit the current pass.
        let ids: Vec<ObjectId> = self.iter_live().collect();
        for id in ids {
            // Take-the-box pattern (see proceed_logic): the dispatched
            // object gets `&mut Objects` with its own slot temporarily
            // empty, so callees can freely query/mutate the arena
            // without aliasing.
            let mut boxed = match self.slots[id.index as usize].obj.take() {
                Some(b) => b,
                None => continue,
            };
            boxed.takt(cms, rng, self);
            // Only re-install if the slot wasn't overwritten. A callee
            // could have despawned this id (bumping generation + adding
            // to free list); in that case we drop the box.
            let slot = &mut self.slots[id.index as usize];
            if slot.generation == id.generation && slot.obj.is_none() {
                slot.obj = Some(boxed);
            }
        }
    }
}

/// If the head/tail pointer was pointing at the removed node, repoint it
/// at the replacement. Otherwise leave it alone — the removed node was
/// somewhere mid-list and the caller already patched its neighbours.
fn patch_head(
    current: Option<ObjectId>,
    removed: ObjectId,
    replacement: Option<ObjectId>,
) -> Option<ObjectId> {
    if current == Some(removed) {
        replacement
    } else {
        current
    }
}

pub struct LogicIter<'a> {
    objs: &'a Objects,
    cursor: Option<ObjectId>,
}

impl<'a> Iterator for LogicIter<'a> {
    type Item = ObjectId;
    fn next(&mut self) -> Option<ObjectId> {
        let id = self.cursor?;
        self.cursor = self.objs.slots[id.index as usize].next_lt;
        Some(id)
    }
}

/// Port of `CMatrixMapStatic::StaticTakt` (MatrixMapStatic.cpp:107-143).
/// Decrements ablaze/shorted TTLs then delegates to the subclass'
/// `logic_takt`.
///
/// The `SwitchAnimation(ANIMATION_STAY)` side effect on SHORTED→clear
/// (MatrixMapStatic.cpp:133) is deferred: it sits inside
/// `if (IsRobot())`, and no robot subclass exists yet in this port.
fn static_takt(objs: &mut Objects, id: ObjectId, ms: i32, rng: &mut Rnd) {
    // Take-the-box pattern — see `proceed_logic`. This also handles the
    // case where `id` points at a freed slot (returns early, same as
    // the C++ tombstone branch).
    let mut boxed = match objs.slots.get_mut(id.index as usize).and_then(|s| {
        if s.generation == id.generation {
            s.obj.take()
        } else {
            None
        }
    }) {
        Some(b) => b,
        None => return,
    };

    if boxed.object_state() & OBJECT_STATE_ABLAZE != 0 {
        let ttl = (boxed.ablaze_ttl() - ms).max(0);
        boxed.set_ablaze_ttl(ttl);
        if ttl == 0 {
            boxed.object_state_clear(OBJECT_STATE_ABLAZE);
        }
    }
    if boxed.object_state() & OBJECT_STATE_SHORTED != 0 {
        let ttl = (boxed.shorted_ttl() - ms).max(0);
        boxed.set_shorted_ttl(ttl);
        if ttl == 0 {
            boxed.object_state_clear(OBJECT_STATE_SHORTED);
            // Robot-specific:
            //   if (IsRobot()) AsRobot()->SwitchAnimation(ANIMATION_STAY);
            // Lands with the robot subclass port.
        }
    }
    boxed.logic_takt(ms, rng, objs);

    // Re-install unless the callee despawned us.
    let slot = &mut objs.slots[id.index as usize];
    if slot.generation == id.generation && slot.obj.is_none() {
        slot.obj = Some(boxed);
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Counting stub that also lets each invocation optionally mutate the
    /// arena via a closure. The closure captures `&mut Objects`, which
    /// would collide with the &mut borrow `static_takt` holds on the
    /// target object — we work around that by parking the closure on a
    /// shared handle (`Rc<RefCell<...>>`) and only draining it from the
    /// outside after a `proceed_logic` call.
    struct Stub {
        core: ObjectCore,
        rchange: u32,
        state: u32,
        ablaze: i32,
        shorted: i32,
        log: Rc<RefCell<Vec<(&'static str, i32)>>>,
        name: &'static str,
    }

    impl Stub {
        fn new(name: &'static str, log: Rc<RefCell<Vec<(&'static str, i32)>>>) -> Self {
            Self {
                core: ObjectCore {
                    obj_type: ObjectType::MapObject,
                    ..Default::default()
                },
                rchange: MR_ALL,
                state: 0,
                ablaze: 0,
                shorted: 0,
                log,
                name,
            }
        }
    }

    impl MapStatic for Stub {
        fn core(&self) -> &ObjectCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut ObjectCore {
            &mut self.core
        }
        fn rchange(&self) -> u32 {
            self.rchange
        }
        fn rchange_set(&mut self, b: u32) {
            self.rchange |= b;
        }
        fn rchange_clear(&mut self, b: u32) {
            self.rchange &= !b;
        }
        fn object_state(&self) -> u32 {
            self.state
        }
        fn object_state_set(&mut self, b: u32) {
            self.state |= b;
        }
        fn object_state_clear(&mut self, b: u32) {
            self.state &= !b;
        }
        fn ablaze_ttl(&self) -> i32 {
            self.ablaze
        }
        fn set_ablaze_ttl(&mut self, t: i32) {
            self.ablaze = t;
        }
        fn shorted_ttl(&self) -> i32 {
            self.shorted
        }
        fn set_shorted_ttl(&mut self, t: i32) {
            self.shorted = t;
        }
        fn r_need(&mut self, _need: u32) {}
        fn takt(&mut self, _cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {}
        fn logic_takt(&mut self, cms: i32, _rng: &mut Rnd, _objs: &mut Objects) {
            self.log.borrow_mut().push((self.name, cms));
        }
    }

    fn mk_stub(
        name: &'static str,
        log: Rc<RefCell<Vec<(&'static str, i32)>>>,
    ) -> Box<dyn MapStatic> {
        Box::new(Stub::new(name, log))
    }

    #[test]
    fn add_lt_is_idempotent_and_del_lt_on_absent_is_noop() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        assert!(!objs.in_lt(a));
        objs.del_lt(a); // noop
        objs.add_lt(a);
        objs.add_lt(a); // idempotent
        assert!(objs.in_lt(a));
        assert_eq!(objs.iter_logic().collect::<Vec<_>>(), vec![a]);
        objs.del_lt(a);
        objs.del_lt(a); // idempotent
        assert!(!objs.in_lt(a));
        assert!(objs.iter_logic().next().is_none());
    }

    #[test]
    fn proceed_logic_visits_each_object_once() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        let b = objs.spawn(mk_stub("b", log.clone()));
        let c = objs.spawn(mk_stub("c", log.clone()));
        objs.add_lt(a);
        objs.add_lt(b);
        objs.add_lt(c);

        objs.proceed_logic(10, &mut Rnd::new(1));
        assert_eq!(log.borrow().clone(), vec![("a", 10), ("b", 10), ("c", 10)]);
    }

    #[test]
    fn fit_to_mask_covers_every_object_type() {
        use crate::matrix_game::common::{
            TRACE_BUILDING, TRACE_CANNON, TRACE_FLYER, TRACE_OBJECT, TRACE_ROBOT,
        };
        fn stub(kind: ObjectType, state: u32) -> Box<dyn MapStatic> {
            let log = Rc::new(RefCell::new(Vec::new()));
            let mut s = Stub::new("s", log);
            s.core.obj_type = kind;
            s.state = state;
            Box::new(s)
        }
        let rob = stub(ObjectType::RobotAi, 0);
        let cnn = stub(ObjectType::Cannon, 0);
        let bld = stub(ObjectType::Building, 0);
        let obj = stub(ObjectType::MapObject, 0);
        let fly = stub(ObjectType::Flyer, 0);
        let emp = stub(ObjectType::Empty, 0);
        assert!(fit_to_mask(&*rob, TRACE_ROBOT));
        assert!(!fit_to_mask(&*rob, TRACE_OBJECT));
        assert!(fit_to_mask(&*cnn, TRACE_CANNON));
        assert!(fit_to_mask(&*bld, TRACE_BUILDING));
        assert!(fit_to_mask(&*obj, TRACE_OBJECT));
        assert!(fit_to_mask(&*fly, TRACE_FLYER));
        assert!(!fit_to_mask(&*emp, u32::MAX)); // Empty always fails.
                                                // OR'd mask — robot also passes against a TRACE_ANYOBJECT-style mask.
        assert!(fit_to_mask(&*rob, TRACE_ROBOT | TRACE_OBJECT));
    }

    #[test]
    fn fit_to_mask_respects_skip_invisible() {
        use crate::matrix_game::common::{TRACE_OBJECT, TRACE_SKIP_INVISIBLE};
        let visible = Stub::new("v", Rc::new(RefCell::new(Vec::new())));
        let mut invis = Stub::new("i", Rc::new(RefCell::new(Vec::new())));
        invis.state = OBJECT_STATE_TRACE_INVISIBLE;
        // Matching type, skip-invisible set: invisible object is excluded.
        assert!(fit_to_mask(&visible, TRACE_OBJECT | TRACE_SKIP_INVISIBLE));
        assert!(!fit_to_mask(&invis, TRACE_OBJECT | TRACE_SKIP_INVISIBLE));
        // Without skip-invisible flag, invisible still passes type filter.
        assert!(fit_to_mask(&invis, TRACE_OBJECT));
    }

    #[test]
    fn find_objects_returns_all_hits_in_radius_and_honors_skip() {
        use crate::matrix_game::common::TRACE_OBJECT;
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let mut make = |x: f32, y: f32| -> ObjectId {
            let mut s = Stub::new("x", log.clone());
            s.core.obj_type = ObjectType::MapObject;
            s.core.geo_center = glam::Vec3::new(x, y, 0.0);
            objs.spawn(Box::new(s))
        };
        let a = make(0.0, 0.0);
        let b = make(5.0, 0.0);
        let c = make(20.0, 0.0); // outside radius 10
        let _d = make(3.0, 4.0); // distance 5, inside

        let mut hits = Vec::new();
        let got = objs.find_objects(glam::Vec2::ZERO, 10.0, 1.0, TRACE_OBJECT, None, |_, id| {
            hits.push(id);
            Control::Continue
        });
        assert!(got, "at least one hit");
        // `a`, `b`, `d` expected; `c` excluded.
        assert!(hits.contains(&a));
        assert!(hits.contains(&b));
        assert!(!hits.contains(&c));

        // Skip `a` → it's excluded from the enumeration.
        hits.clear();
        objs.find_objects(
            glam::Vec2::ZERO,
            10.0,
            1.0,
            TRACE_OBJECT,
            Some(a),
            |_, id| {
                hits.push(id);
                Control::Continue
            },
        );
        assert!(!hits.contains(&a));
    }

    #[test]
    fn find_objects_stops_on_break() {
        use crate::matrix_game::common::TRACE_OBJECT;
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        for _ in 0..5 {
            let mut s = Stub::new("x", log.clone());
            s.core.obj_type = ObjectType::MapObject;
            s.core.geo_center = glam::Vec3::ZERO;
            objs.spawn(Box::new(s));
        }
        let mut visited = 0;
        let got = objs.find_objects(glam::Vec2::ZERO, 10.0, 1.0, TRACE_OBJECT, None, |_, _| {
            visited += 1;
            Control::Break
        });
        assert!(got);
        assert_eq!(visited, 1, "Break stops after the first hit");

        // `any_object_in_radius` shortcut — semantically the same.
        assert!(objs.any_object_in_radius(glam::Vec2::ZERO, 10.0, 1.0, TRACE_OBJECT, None));
    }

    #[test]
    fn find_objects_filters_by_type_mask() {
        use crate::matrix_game::common::{TRACE_OBJECT, TRACE_ROBOT};
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let mut mapobj = Stub::new("o", log.clone());
        mapobj.core.obj_type = ObjectType::MapObject;
        mapobj.core.geo_center = glam::Vec3::ZERO;
        objs.spawn(Box::new(mapobj));
        // Asking for robots: no match.
        assert!(!objs.any_object_in_radius(glam::Vec2::ZERO, 10.0, 1.0, TRACE_ROBOT, None,));
        // Asking for objects: hit.
        assert!(objs.any_object_in_radius(glam::Vec2::ZERO, 10.0, 1.0, TRACE_OBJECT, None,));
    }

    #[test]
    fn ray_sphere_pick_hits_front_face_when_outside() {
        // Sphere at (10, 0, 0) radius 2. Ray from origin down +X.
        // Front face at x=8 → t=8.
        let c = Vec3::new(10.0, 0.0, 0.0);
        let o = Vec3::ZERO;
        let d = Vec3::X;
        let t = ray_sphere_pick(c, 2.0, o, d).unwrap();
        assert!((t - 8.0).abs() < 1e-5);
    }

    #[test]
    fn ray_sphere_pick_misses_off_axis() {
        let c = Vec3::new(10.0, 5.0, 0.0);
        let o = Vec3::ZERO;
        let d = Vec3::X;
        // Sphere centered 5 units off the ray axis with radius 2 — misses.
        assert!(ray_sphere_pick(c, 2.0, o, d).is_none());
    }

    #[test]
    fn ray_sphere_pick_returns_exit_when_inside_sphere() {
        let c = Vec3::ZERO;
        let o = Vec3::ZERO;
        let d = Vec3::X;
        // Origin at center → -b - sqrt_d = -radius, -b + sqrt_d = +radius.
        // Nearest positive t = +radius.
        let t = ray_sphere_pick(c, 3.0, o, d).unwrap();
        assert!((t - 3.0).abs() < 1e-5);
    }

    #[test]
    fn ray_sphere_pick_rejects_zero_radius() {
        assert!(ray_sphere_pick(Vec3::ZERO, 0.0, Vec3::new(-10.0, 0.0, 0.0), Vec3::X).is_none());
    }

    #[test]
    fn pick_object_returns_nearest_hit_matching_mask() {
        use crate::matrix_game::common::{TRACE_BUILDING, TRACE_OBJECT, TRACE_ROBOT};
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let mut mk = |x: f32, kind: ObjectType, r: f32| -> ObjectId {
            let mut s = Stub::new("x", log.clone());
            s.core.obj_type = kind;
            s.core.geo_center = Vec3::new(x, 0.0, 0.0);
            s.core.radius = r;
            objs.spawn(Box::new(s))
        };
        let _near = mk(5.0, ObjectType::MapObject, 1.0); // t=4
        let mid = mk(10.0, ObjectType::Building, 1.0); // t=9
        let _far = mk(20.0, ObjectType::Building, 1.0); // t=19

        // TRACE_OBJECT: should hit the MapObject at t≈4.
        let (_, t) = objs
            .pick_object(Vec3::ZERO, Vec3::X, TRACE_OBJECT, None)
            .unwrap();
        assert!((t - 4.0).abs() < 1e-5);

        // TRACE_BUILDING: should hit the NEAREST building (mid, t≈9)
        // not the farther one.
        let (hit_id, t) = objs
            .pick_object(Vec3::ZERO, Vec3::X, TRACE_BUILDING, None)
            .unwrap();
        assert_eq!(hit_id, mid);
        assert!((t - 9.0).abs() < 1e-5);

        // TRACE_ROBOT: nothing matches.
        assert!(objs
            .pick_object(Vec3::ZERO, Vec3::X, TRACE_ROBOT, None)
            .is_none());
    }

    #[test]
    fn graphic_takt_visits_every_live_object_ignoring_logic_temp_membership() {
        // Ports the SortEndGraphicTakt contract (MatrixMapStatic.cpp:755):
        // every visible static object gets Takt, regardless of whether
        // it opted into the logic-temp list. The Rust stub below tracks
        // takt calls — we add `a` and `c` to logic-temp but leave `b`
        // off, and assert all three get their `takt` called.
        type TaktLog = Rc<RefCell<Vec<(&'static str, &'static str, i32)>>>;
        struct TaktCounter {
            core: ObjectCore,
            rchange: u32,
            state: u32,
            ablaze: i32,
            shorted: i32,
            name: &'static str,
            log: TaktLog,
        }
        impl MapStatic for TaktCounter {
            fn core(&self) -> &ObjectCore {
                &self.core
            }
            fn core_mut(&mut self) -> &mut ObjectCore {
                &mut self.core
            }
            fn rchange(&self) -> u32 {
                self.rchange
            }
            fn rchange_set(&mut self, b: u32) {
                self.rchange |= b;
            }
            fn rchange_clear(&mut self, b: u32) {
                self.rchange &= !b;
            }
            fn object_state(&self) -> u32 {
                self.state
            }
            fn object_state_set(&mut self, b: u32) {
                self.state |= b;
            }
            fn object_state_clear(&mut self, b: u32) {
                self.state &= !b;
            }
            fn ablaze_ttl(&self) -> i32 {
                self.ablaze
            }
            fn set_ablaze_ttl(&mut self, t: i32) {
                self.ablaze = t;
            }
            fn shorted_ttl(&self) -> i32 {
                self.shorted
            }
            fn set_shorted_ttl(&mut self, t: i32) {
                self.shorted = t;
            }
            fn r_need(&mut self, _: u32) {}
            fn takt(&mut self, cms: i32, _: &mut Rnd, _: &mut Objects) {
                self.log.borrow_mut().push(("takt", self.name, cms));
            }
            fn logic_takt(&mut self, cms: i32, _: &mut Rnd, _: &mut Objects) {
                self.log.borrow_mut().push(("logic", self.name, cms));
            }
        }
        fn mk(name: &'static str, log: TaktLog) -> Box<dyn MapStatic> {
            Box::new(TaktCounter {
                core: ObjectCore {
                    obj_type: ObjectType::MapObject,
                    ..Default::default()
                },
                rchange: 0,
                state: 0,
                ablaze: 0,
                shorted: 0,
                name,
                log,
            })
        }

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk("a", log.clone()));
        let _b = objs.spawn(mk("b", log.clone()));
        let c = objs.spawn(mk("c", log.clone()));
        objs.add_lt(a);
        objs.add_lt(c); // `b` stays off the logic list.

        let mut rng = Rnd::new(1);
        objs.graphic_takt(33, &mut rng);
        // All three reached by graphic_takt.
        let got: Vec<_> = log.borrow().iter().cloned().collect();
        assert_eq!(
            got,
            vec![("takt", "a", 33), ("takt", "b", 33), ("takt", "c", 33)]
        );

        log.borrow_mut().clear();
        objs.proceed_logic(10, &mut rng);
        let got: Vec<_> = log.borrow().iter().cloned().collect();
        // Only the logic-temp members get logic_takt.
        assert_eq!(got, vec![("logic", "a", 10), ("logic", "c", 10)]);
    }

    #[test]
    fn graphic_takt_with_nonpositive_step_in_world_is_noop() {
        // World::graphic_takt guards on step<=0; Objects::graphic_takt
        // here accepts any value but we never invoke it negatively in
        // production. Sanity-check that Objects::graphic_takt with 0
        // still iterates (ports the visible-takt contract — zero-delta
        // frame hitches shouldn't drop takts; it's the caller's job).
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let _a = objs.spawn(mk_stub("a", log.clone()));
        let mut rng = Rnd::new(1);
        objs.graphic_takt(0, &mut rng);
        // mk_stub's `takt` is a noop; this just verifies we don't crash.
    }

    #[test]
    fn arcaded_object_is_skipped() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        let b = objs.spawn(mk_stub("b", log.clone()));
        objs.add_lt(a);
        objs.add_lt(b);
        objs.arcaded_object = Some(b);

        objs.proceed_logic(5, &mut Rnd::new(1));
        assert_eq!(log.borrow().clone(), vec![("a", 5)]);
    }

    #[test]
    fn ablaze_ttl_clears_on_boundary() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        objs.get_mut(a)
            .unwrap()
            .object_state_set(OBJECT_STATE_ABLAZE);
        objs.get_mut(a).unwrap().set_ablaze_ttl(10);
        objs.add_lt(a);

        objs.proceed_logic(10, &mut Rnd::new(1));
        let o = objs.get(a).unwrap();
        assert_eq!(o.ablaze_ttl(), 0);
        assert_eq!(o.object_state() & OBJECT_STATE_ABLAZE, 0);
    }

    #[test]
    fn ablaze_ttl_does_not_clear_early() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        let a = objs.spawn(mk_stub("a", log.clone()));
        objs.get_mut(a)
            .unwrap()
            .object_state_set(OBJECT_STATE_ABLAZE);
        objs.get_mut(a).unwrap().set_ablaze_ttl(15);
        objs.add_lt(a);

        objs.proceed_logic(10, &mut Rnd::new(1));
        let o = objs.get(a).unwrap();
        assert_eq!(o.ablaze_ttl(), 5);
        assert_ne!(o.object_state() & OBJECT_STATE_ABLAZE, 0);
    }

    /// Exercises the `m_NextLogicObject` snapshot (MatrixMapStatic.cpp:349).
    /// A stub removes its *successor* mid-walk; `proceed_logic` must still
    /// land on a live node for the next iteration.
    #[test]
    fn del_lt_of_next_during_takt_does_not_crash_walk() {
        struct Remover {
            core: ObjectCore,
            rchange: u32,
            state: u32,
            ablaze: i32,
            shorted: i32,
            target: ObjectId,
        }
        impl MapStatic for Remover {
            fn core(&self) -> &ObjectCore {
                &self.core
            }
            fn core_mut(&mut self) -> &mut ObjectCore {
                &mut self.core
            }
            fn rchange(&self) -> u32 {
                self.rchange
            }
            fn rchange_set(&mut self, b: u32) {
                self.rchange |= b;
            }
            fn rchange_clear(&mut self, b: u32) {
                self.rchange &= !b;
            }
            fn object_state(&self) -> u32 {
                self.state
            }
            fn object_state_set(&mut self, b: u32) {
                self.state |= b;
            }
            fn object_state_clear(&mut self, b: u32) {
                self.state &= !b;
            }
            fn ablaze_ttl(&self) -> i32 {
                self.ablaze
            }
            fn set_ablaze_ttl(&mut self, t: i32) {
                self.ablaze = t;
            }
            fn shorted_ttl(&self) -> i32 {
                self.shorted
            }
            fn set_shorted_ttl(&mut self, t: i32) {
                self.shorted = t;
            }
            fn r_need(&mut self, _n: u32) {}
            fn takt(&mut self, _c: i32, _r: &mut Rnd, _o: &mut Objects) {}
            fn logic_takt(&mut self, _c: i32, _r: &mut Rnd, objs: &mut Objects) {
                // Natural with the take-the-box pattern: `objs` is the
                // arena sans this slot, so mutating a different slot's
                // list membership is a plain safe call.
                objs.del_lt(self.target);
            }
        }

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();

        let a_id = objs.spawn(Box::new(Remover {
            core: ObjectCore {
                obj_type: ObjectType::MapObject,
                ..Default::default()
            },
            rchange: 0,
            state: 0,
            ablaze: 0,
            shorted: 0,
            target: ObjectId {
                index: 999,
                generation: 0,
            }, // patched below
        }));
        let b_id = objs.spawn(mk_stub("b", log.clone()));
        let c_id = objs.spawn(mk_stub("c", log.clone()));
        // Patch the remover's target now that `b_id` is known.
        {
            let obj = objs.get_mut(a_id).unwrap();
            let rem = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Remover) };
            rem.target = b_id;
        }
        objs.add_lt(a_id);
        objs.add_lt(b_id);
        objs.add_lt(c_id);

        objs.proceed_logic(10, &mut Rnd::new(1));
        // `b` should NOT be ticked (the remover nuked it mid-walk before
        // the cursor reached it); `c` should still tick because the
        // snapshot re-pointed to it.
        assert_eq!(log.borrow().clone(), vec![("c", 10)]);
        assert!(!objs.in_lt(b_id));
    }

    /// Ports the other half of the `m_NextLogicObject` contract: objects
    /// added *during* a walk are NOT visited this call
    /// (MatrixMapStatic.cpp:349 snapshots the next link before dispatch).
    #[test]
    fn add_lt_during_takt_visits_on_next_call_only() {
        struct Adder {
            core: ObjectCore,
            rchange: u32,
            state: u32,
            ablaze: i32,
            shorted: i32,
            new_id: ObjectId,
        }
        impl MapStatic for Adder {
            fn core(&self) -> &ObjectCore {
                &self.core
            }
            fn core_mut(&mut self) -> &mut ObjectCore {
                &mut self.core
            }
            fn rchange(&self) -> u32 {
                self.rchange
            }
            fn rchange_set(&mut self, b: u32) {
                self.rchange |= b;
            }
            fn rchange_clear(&mut self, b: u32) {
                self.rchange &= !b;
            }
            fn object_state(&self) -> u32 {
                self.state
            }
            fn object_state_set(&mut self, b: u32) {
                self.state |= b;
            }
            fn object_state_clear(&mut self, b: u32) {
                self.state &= !b;
            }
            fn ablaze_ttl(&self) -> i32 {
                self.ablaze
            }
            fn set_ablaze_ttl(&mut self, t: i32) {
                self.ablaze = t;
            }
            fn shorted_ttl(&self) -> i32 {
                self.shorted
            }
            fn set_shorted_ttl(&mut self, t: i32) {
                self.shorted = t;
            }
            fn r_need(&mut self, _n: u32) {}
            fn takt(&mut self, _c: i32, _r: &mut Rnd, _o: &mut Objects) {}
            fn logic_takt(&mut self, _c: i32, _r: &mut Rnd, objs: &mut Objects) {
                objs.add_lt(self.new_id);
            }
        }

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut objs = Objects::new();
        // Pre-create the "future" object but keep it out of the list.
        let future_id = objs.spawn(mk_stub("future", log.clone()));
        let adder_id = objs.spawn(Box::new(Adder {
            core: ObjectCore {
                obj_type: ObjectType::MapObject,
                ..Default::default()
            },
            rchange: 0,
            state: 0,
            ablaze: 0,
            shorted: 0,
            new_id: future_id,
        }));
        objs.add_lt(adder_id);

        objs.proceed_logic(10, &mut Rnd::new(1)); // 1st call: only `adder` runs.
        assert!(
            log.borrow().is_empty(),
            "future tick ran in 1st walk: {:?}",
            log.borrow()
        );
        assert!(objs.in_lt(future_id));

        objs.proceed_logic(7, &mut Rnd::new(1)); // 2nd call: `adder` + `future`.
        assert_eq!(log.borrow().clone(), vec![("future", 7)]);
    }
}
