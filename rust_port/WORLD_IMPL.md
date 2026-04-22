# World / Game-Object Tick Loop — Implementation Plan

Port of `CMatrixMapStatic` base class and the logic/graphic takt driver
(`MatrixLogic.cpp:2680-2770`, `MatrixMapStatic.{cpp,hpp}`). Scope A from
the earlier proposal: base class + world/tick loop only, no concrete
subclasses wired to logic yet.

Layout follows `CROSSREF.md`: new rows added there in the same PR.

## Files to add

| Rust file                          | Mirrors                                  |
|------------------------------------|------------------------------------------|
| `matrix_game/map_static.rs`        | `MatrixMapStatic.{cpp,hpp}`              |
| (modify) `matrix_game/world.rs`    | `MatrixMap.cpp` top-level takt driver    |
| (modify) `matrix_game/form_game.rs`| `MatrixFormGame.cpp` tick entry          |
| (modify) `matrix_game/mod.rs`      | add `pub mod map_static;`                |
| (modify) `src/CROSSREF.md`         | add row for `map_static.rs`              |

No files are deleted. Nothing in `object.rs` / `object_building.rs` /
`units.rs` is rewired in this pass — they keep their render-only roles
until their concrete subclasses land.

## Data model

### `SObjectCore` → `ObjectCore`

`MatrixMapStatic.hpp:133-199`. Fields in order:

```rust
pub struct ObjectCore {
    pub matrix: glam::Mat4,       // m_Matrix
    pub inv_matrix: glam::Mat4,   // m_IMatrix
    pub radius: f32,              // m_Radius
    pub geo_center: glam::Vec3,   // m_GeoCenter
    pub obj_type: ObjectType,     // m_Type
    pub terrain_color: u32,       // m_TerainColor (0xFFFFFFFF default)
}
```

Ref-counting is dropped. C++ uses it so effects (`CMatrixEffectWeapon`,
`CMatrixEffectRepair`) can hold a `SObjectCore*` after the owning
object is destroyed and read `m_Object == NULL` as a tombstone. In
Rust we'll represent the same contract with `ObjectId` handles into
the arena (below) — a looked-up handle returning `None` from the
arena is the direct equivalent of `core->m_Object == NULL`. The
`Rc`-less design keeps lifetimes tractable and matches the existing
port's pattern (`PointLightSystem`, `ObjectInstance`).

### `EObjectType` → `ObjectType`

Straight enum with the same discriminants
(`MatrixMapStatic.hpp:29-39`):

```rust
#[repr(u32)]
pub enum ObjectType {
    Empty    = 0,
    MapObject = 2,
    RobotAi   = 3,
    Building  = 4,
    Cannon    = 5,
    Flyer     = 6,
}
```

### Resource-change bitmask → `RChange` bitflags

`MR_Graph..MR_MiniMap` (`MatrixMapStatic.hpp:17-25`). `bitflags!`
macro; default `!0` (all bits set) matching `m_RChange(0xffffffff)`
in the ctor (`MatrixMapStatic.hpp:346`).

### Object-state flags → `ObjectState` bitflags

`OBJECT_STATE_ABLAZE`..`OBJECT_STATE_DIP` plus the per-subclass
overlay bits. Subclass-specific bits (ROBOT_FLAG_*, BUILDING_*, …)
are listed but left as `#[allow(dead_code)]` constants — they'll be
used when the subclasses land.

## Object arena and logic-temp list

The C++ uses intrusive doubly-linked lists of raw pointers:
`m_FirstLogicTemp` / `m_NextLogicTemp`. Objects can add/remove
themselves during iteration — `ProceedLogic` snapshots the next
pointer before calling `StaticTakt` (`MatrixMapStatic.cpp:349`).

Rust equivalent in `world.rs`:

```rust
pub struct ObjectId(pub u32);  // generational index

struct Slot {
    gen: u32,
    obj: Option<Box<dyn MapStatic>>,
    in_logic_temp: bool,
    next_logic_temp: Option<ObjectId>,
    prev_logic_temp: Option<ObjectId>,
}

pub struct Objects {
    slots: Vec<Slot>,
    free: Vec<u32>,
    first_logic_temp: Option<ObjectId>,
    last_logic_temp: Option<ObjectId>,
}
```

Public ops (all mirror `CMatrixMapStatic` statics):

- `spawn(Box<dyn MapStatic>) -> ObjectId` — replaces `new` + ctor.
- `remove(id)` — `~CMatrixMapStatic`: calls `DelLT`, drops the box.
- `add_lt(id)` / `del_lt(id)` — `AddLT` / `DelLT`
  (`MatrixMapStatic.hpp:441-442`). Idempotent like the C++ `InLT`
  guard.
- `proceed_logic(&mut self, takts: i32)` — ports
  `ProceedLogic` (`MatrixMapStatic.cpp:338-362`):
  1. Walk `first_logic_temp`.
  2. Before dispatch, stash `next = slot.next_logic_temp` so the
     callee can `del_lt` itself without breaking the walk. (The
     `g_MatrixMap->m_NextLogicObject` global in the original
     becomes a field on `Objects` — needed because the
     trait-object `static_takt` reborrows `Objects`.)
  3. Skip when `id == arcaded_object` (mirrors
     `MatrixMapStatic.cpp:350`; `arcaded_object` is `None` for now —
     the player-side plumbing lands later).
  4. Call `static_takt(self, id, takts)`.

The "stash next" step is the critical correctness invariant — losing
it means an object that removes itself mid-tick stops the whole walk.
The trait method takes `&mut Objects` + `ObjectId` (not `&mut self`)
so the callee can also spawn / despawn siblings.

## `MapStatic` trait — the `CMatrixMapStatic` contract

```rust
pub trait MapStatic {
    fn core(&self) -> &ObjectCore;
    fn core_mut(&mut self) -> &mut ObjectCore;
    fn rchange(&self) -> RChange;
    fn rchange_mut(&mut self) -> &mut RChange;
    fn object_state(&self) -> ObjectState;
    fn object_state_mut(&mut self) -> &mut ObjectState;

    fn ablaze_ttl(&self) -> i32;
    fn set_ablaze_ttl(&mut self, ttl: i32);
    fn shorted_ttl(&self) -> i32;
    fn set_shorted_ttl(&mut self, ttl: i32);

    // Pure virtuals (MatrixMapStatic.hpp:457-478)
    fn r_need(&mut self, need: RChange);
    fn takt(&mut self, cms: i32);
    fn logic_takt(&mut self, cms: i32);

    // Left as `unimplemented!()` default methods on the trait for
    // the Scope-A pass — none of them are called by ProceedLogic or
    // the world loop, so stubs are safe. They light up as subclasses
    // land.
    fn pick(&self, _orig: glam::Vec3, _dir: glam::Vec3) -> Option<f32> { None }
    fn before_draw(&mut self) {}
    fn draw(&mut self) {}
    fn free_dynamic_resources(&mut self) {}
    fn calc_bounds(&self) -> Option<(glam::Vec3, glam::Vec3)> { None }
    fn side(&self) -> i32 { -1 }
    fn need_repair(&self) -> bool { false }
}
```

The `__forceinline bool IsRobot()` etc. helpers become free
functions against `&dyn MapStatic` that read `core().obj_type`.

## `static_takt` — ablaze / shorted TTLs

Free function on `Objects`, not a trait method, so it can read + mutate
the slot without gymnastics:

```rust
fn static_takt(objs: &mut Objects, id: ObjectId, ms: i32) {
    let Some(obj) = objs.get_mut(id) else { return };
    // Port of MatrixMapStatic.cpp:107-143.
    if obj.object_state().contains(ObjectState::ABLAZE) {
        let ttl = (obj.ablaze_ttl() - ms).max(0);
        obj.set_ablaze_ttl(ttl);
        if ttl == 0 {
            *obj.object_state_mut() &= !ObjectState::ABLAZE;
        }
    }
    if obj.object_state().contains(ObjectState::SHORTED) {
        let ttl = (obj.shorted_ttl() - ms).max(0);
        obj.set_shorted_ttl(ttl);
        if ttl == 0 {
            *obj.object_state_mut() &= !ObjectState::SHORTED;
            // Robot-specific `SwitchAnimation(ANIMATION_STAY)` deferred
            // until the robot subclass exists; matches the
            // `if (IsRobot())` gate at MatrixMapStatic.cpp:133.
        }
    }
    obj.logic_takt(ms);
}
```

## World top-level tick

`MatrixLogic.cpp:2720-2766` decomposes `step_ms` into `n` full
`LOGIC_TAKT_PERIOD=10ms` portions plus a remainder. Each portion
calls `ProceedLogic(10)`; the remainder calls `ProceedLogic(rem)`.

```rust
pub const LOGIC_TAKT_PERIOD_MS: i32 = 10;

pub struct World {
    pub objects: Objects,
    pub tick: u64,       // elapsed LOGIC_TAKT_PERIODs — keep u64 for stats
    pub elapsed_ms: i64, // replaces the old f32 `elapsed`; ints match C++
}

impl World {
    pub fn takt(&mut self, step_ms: i32) {
        let full = step_ms / LOGIC_TAKT_PERIOD_MS;
        for _ in 0..full {
            self.objects.proceed_logic(LOGIC_TAKT_PERIOD_MS);
            self.tick += 1;
        }
        let rem = step_ms - full * LOGIC_TAKT_PERIOD_MS;
        if rem > 0 {
            self.objects.proceed_logic(rem);
        }
        self.elapsed_ms += step_ms as i64;
    }
}
```

Dropped vs the existing `World`:

- `pub elapsed: f32` — nothing reads it; the old `update(dt)`
  signature is replaced by `takt(step_ms)` to match the C++ units.
  Callers multiply by 1000 at the boundary (they already do — see
  `state.camera.takt(dt * 1000.0)`).
- `fn update(dt: f32)` — replaced by `takt`. Single call site in
  `form_game.rs`.

## `form_game.rs` wiring

One call-site change in the redraw branch (currently
`state.game.update(dt)`). New line:

```rust
let step_ms = (dt * 1000.0).round() as i32;
state.game.takt(step_ms);
```

Left alone:

- camera / minimap / terrain `takt` calls — they already receive
  `ms` and don't depend on World.
- map / renderer wiring — rendering is downstream of the logic takt
  in the original too (`CMatrixMap::Takt(step)` runs after
  `ProceedLogic`, `MatrixLogic.cpp:2761`), so ordering stays:
  `world.takt` → `camera.takt` → `terrain.takt` → render.

## Verification plan

1. **Compile (native + wasm)** — `cargo check` and `wasm-pack build
   --dev`.  Baseline: both succeed before; both must succeed after.
2. **Unit tests** under `tests/` or `#[cfg(test)]` in `map_static.rs`
   / `world.rs`, covering:
   - `Objects::add_lt` is idempotent; `del_lt` on a non-present id
     is a no-op (matches `InLT` guard,
     `MatrixMapStatic.hpp:440-442`).
   - `proceed_logic` visits every logic-temp object exactly once
     per call.
   - An object that calls `del_lt(self)` inside `logic_takt` does
     not break the walk — the snapshot-next invariant.
   - An object that `add_lt`s a *new* sibling inside `logic_takt`
     does **not** visit the new sibling this call (matches
     `m_NextLogicObject` = snapshotted-old-next,
     `MatrixMapStatic.cpp:349`).
   - `static_takt` clears ABLAZE when TTL hits 0 exactly, not
     earlier (boundary test with `ttl=10, ms=10`).
   - `World::takt(25)` with `LOGIC_TAKT_PERIOD_MS=10` runs
     `proceed_logic(10)` twice plus `proceed_logic(5)` once (use a
     counting dummy object to assert).
3. **Visual regression** — start the native binary and the
   browser build, confirm the frame still renders. Nothing in the
   scene uses `World::takt`'s object list yet, so output must be
   bit-identical to the pre-change baseline. If anything moves on
   screen, something wired through accidentally.

## Non-goals (deferred)

- Concrete subclasses: `CMatrixMapObject`, `CMatrixRobotAI`,
  `CMatrixBuilding`, `CMatrixCannon`, `CMatrixFlyer`. Listed in
  `CROSSREF.md` "Not yet ported" — unchanged.
- `CMatrixMapLogic`: pathfinding, zones, `GatherInfo`,
  `m_TaktNext`-driven side logic.
- Graphic takt (`CMatrixMap::Takt`): tile transitions, interface
  takts, arcaded-robot interpolation.
- Effects tick: `CMatrixEffect` base + all subclasses.
- Visibility lists (`m_FirstVisNew` / `m_FirstVisOld`): the
  `WillDraw` TODO in the original (`MatrixMapStatic.hpp:281`) is
  inherited verbatim.
- Ref-counted `SObjectCore`: replaced by `ObjectId` lookup as
  described above. If a future effect needs the tombstone semantics,
  revisit.

## Risk / open questions

- **Trait object storage**: `Box<dyn MapStatic>` behind an arena
  slot forces `'static` bounds on all subclass data. Should be fine
  — C++ subclasses already own their data and borrow nothing
  compile-time — but flag this if a subclass later wants to hold a
  `&GameMap`. The original solves that with the `g_MatrixMap`
  global; we can too via a thread-local or explicit
  `&mut WorldContext` passed into `takt`.
- **`arcaded_object` skip**: the player-side `GetArcadedObject()`
  check in `ProceedLogic` is left as an `Option<ObjectId>` field
  on `Objects`, defaulted to `None`. Filled in when sides land.
- **Interpolation of ints vs floats**: the old `World::elapsed: f32`
  drifted under long runs; moving to `i64 elapsed_ms` matches the
  C++ `GetTime()` contract. No observable difference in the first
  few hours; the caller-facing signature change (dt→step_ms) is the
  only visible effect.
