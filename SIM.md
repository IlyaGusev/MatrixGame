# SIM.md — headless battle simulation & measurement playbook

How to run the game logic without a window/GPU to reproduce gameplay
bugs, measure balance/behavior, and profile CPU cost. Everything here
runs on the REAL map data and REAL config (`Data/robots.pkg` +
`Data/robots.dat`) — no mocks, so numbers transfer to the shipped game.

## Core recipe (the harness every probe uses)

```rust
use matrixgame_rs::matrix_game::logic::MapLogic;
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg")?)?;
let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP")?;   // any .CMAP; list_files() to browse
let map = GameMap::from_cmap_bytes(&cmap)?;
let matrix_data = Storage::from_bytes(&std::fs::read("../Data/robots.dat")?)?;

let mut game = MapLogic::new();          // or MapLogic::with_seed(n) for determinism
game.load_config(&matrix_data);          // g_Config tables + AI robot catalogue + side names
game.spawn_buildings(&map);              // optional: the map's authored bases/factories
game.spawn_robots(&map);                 // optional: the map's pre-placed robots
game.ensure_sides_from_objects();        // build Side entries, set SS_ACTIVE
game.apply_side_resources(&map);         // SideResInfo/SideAIInfo + GroupNoTeamRobot
game.accrue_resources(100_000);          // load-time income fast-forward (parity)

// The sim loop. MapScope is MANDATORY per iteration — logic reaches the
// map through the current_map() TLS pointer (ports g_MatrixMap).
for _ in 0..(4 * 60 * 20) {              // 4 sim-minutes at 50ms frames
    let _scope = MapScope::enter(&map, game.elapsed_ms);
    game.takt(50);                        // any step; logic slices it into 10ms takts
}
```

Key facts:
- `game.takt(step_ms)` advances `elapsed_ms` itself and runs everything:
  robot orders/movement, weapons, effects, production, win/lose, and the
  per-side AI (TaktHL/TaktTL) for every non-player side.
- Player-side robots only act on orders; AI sides play autonomously.
  For a "battle in a jar" give robots to sides 1 vs 2 and let side 2's
  AI attack, or drive player robots with `pg_order_*`.
- Rendering never runs; anything visual-only (billboards, meshes) still
  ACCUMULATES in `objs.pending_effects` → `game.effects`, so effect
  counts/kinds are observable.
- Step size: 50ms ≈ a 20 FPS frame. Results are step-size-insensitive
  because logic always advances in fixed 10ms slices internally.

## Spawning synthetic actors

Robots (see `tests/test_ai.rs`, `examples/laser_turret_probe.rs`):

```rust
let mut r = Robot::new(glam::Vec3::new(x, y, z), side, ChassisKind::Track);
r.state = RobotState::Idle;              // = ROBOT_SUCCESSFULLY_BUILD
let id = game.objects.spawn(Box::new(r));
robot_mut(&mut game.objects, id).unwrap().self_id = Some(id);
game.objects.add_lt(id);                 // join the logic list
```

Buildings: `Building::from_instance(&BuildingInstance { x, y, side, kind, .. })`
(same spawn + `self_id` + `add_lt` dance; see `logic.rs` tests).
Cannons: `Cannon::new(pos2, z, angle, side, kind /*1..=4*/, parent, slot)`,
then set `hit_point`/`hit_point_max` (or `begin_construction`).
Give robots weapons via `r.config` (a `RobotConfig`) if the default
chassis-only bot isn't enough. Find a passable spawn cell with
`logic::is_absence_wall(&map, 2, 4, mx, my)`; world = cell × `GLOBAL_SCALE_MOVE`.
Remove actors between scenario rounds with `game.objects.remove(id)`.

## Observation points (what to assert / print)

- Robots: `robot_ref(&game.objects, id)` → `hit_point`, `env.target/
  target_attack/enemies`, `orders`, `team`, `group_logic`, `map_x/y`.
- Sides: `game.side_by_id(sid)` → `teams[i].action.ty`, `robots_cnt`,
  `strength`, `war_side`, `logic_groups`.
- Turrets: cast via `object_cannon::Cannon` (`c.target`, `c.kind`) and
  poll `c.weapons.iter().filter_map(|&w| game.objects.weapons.get(w))`
  → `w.is_fire()` for is-it-shooting.
- Effects: `game.effects` (match on `GameEffect::MovingObject` etc.).
- Damage: snapshot `hit_point` before/after N sim-seconds.

Aggregate over time, don't eyeball single ticks: e.g. the turret study
sampled every ~200ms for 6 sim-minutes into per-turret counters
(aim ticks / fire ticks / alive ticks) and compared kinds — that's what
exposed "aims 19% of the time but fires only 12% of it" for one kind,
which then decomposed into gate counters (below).

## Drilling into a decision (temporary gate counters)

When a ratio looks wrong, instrument the decision point directly:
drop `static GATE: [AtomicU64; N]` counters into the branch being
studied (one per bail-out reason), bump them in each arm, print from
the probe after the sim. This turned "fires only 12% of aim time" into
"1864 out-of-range vs 216 fires vs 0 trace-fails" in one run — i.e. the
turret was tracking targets beyond weapon range (authentic behavior),
not failing its aim trace. REMOVE the counters after the diagnosis;
they're scaffolding, not product code.

## CPU profiling

**CRITICAL fidelity rule learned the hard way**: a battle bench without
`spawn_map_objects` flatters everything ~10× — decorative objects share
the object arena and (before the typed-index fix) poisoned every scan
and trace. Always spawn them:
`game.spawn_map_objects(&map, &Storage::from_bytes(&cmap)?)`.

Native (headless, release — the honest way to time logic):

```rust
let t0 = std::time::Instant::now();
game.takt(50);
let ms = t0.elapsed().as_secs_f64() * 1000.0;   // accumulate avg/worst
```

Run with `cargo run --release --example <probe>`. Reference point
(2026-07, ATOLL + map objects, 26-robot missile brawl, after the
typed-index + gather-reorder fixes): full logic = **~4.4ms avg wasm /
~1.3ms steady** per 50ms frame.

### wasm CPU (node) — profiling the actual wasm build locally

Native numbers can mislead (different allocator, no JIT warmup): the
`wasm-bench` feature exposes the same heavy battle as
`bench_battle(pkg_bytes, dat_bytes)` and adds per-phase timers inside
`MapLogic::takt` (`TAKT_PHASE_US`: gather / proceed / sides / fx / rest):

```
wasm-pack build --target nodejs --out-dir pkg-bench --features wasm-bench
node bench_node.mjs
```

This is what isolated the three battle-FPS killers of 2026-07: (1) all
object traces/scans walking every decorative arena slot → typed
`unit_ids`/`mapobject_ids` indices in `Objects`; (2) `gather_info`
running its visibility traces before the already-known-enemy check →
reordered (behavior-identical); (3) per-frame Vec collections feeding
`calc_proj` → precomputed per-chassis.

In-browser (the real deal, since wasm shares one thread): the app logs
once per second to the console (F12) and the on-screen FPS text:

```
fps: 29.6 (...) | per-frame ms: takt=11.10 gfx=0.03 ui=0.00 syncR=5.00 sync=1.30 render=7.70 | robots=25 effects=390 bb=513
```

- `takt`  — pure game simulation (AI, movement, weapons, effects),
- `gfx`   — per-object graphic takt,
- `ui`    — interface relayout / text,
- `syncR` — robot instance + projected-shadow prep,
- `sync`  — camera/terrain/minimap/other prep,
- `render`— billboard/mesh queue build + all pass encoding + submit.
Implemented as `platform::now_secs()` spans in `form_game.rs`
(`perf_acc`), drained with the FPS log line.

## Where the existing harnesses live

- `tests/test_ai.rs` — end-to-end AI acceptance: real map, 4 sim-min,
  asserts AI sides build robots, form teams, pick actions. Slow (~30s),
  skips gracefully when `Data/` is absent.
- `examples/rocket_battle_probe.rs` — full-map AI battle + per-turret
  aim/fire statistics + takt timing.
- `examples/rocket_turret_probe.rs` — single turret vs robot matrix
  (distance/height/motion scans, per-position fired/damage table).
- `examples/laser_turret_probe.rs` — beam-weapon variant of the above.
- `examples/attack_order_sim.rs`, `collision_sim.rs` — older
  order/movement sims, same pattern.
- `src/matrix_game/side_ai.rs` `#[cfg(test)]` — synthetic 2-region road
  network fixture (`map_with_regions`) for fast AI unit tests when the
  real map is overkill.

Probes are kept in-tree on purpose (repo convention — dozens of
`probe_*.rs` / `*_sim.rs` examples): a reproduced bug should leave its
reproducer behind.
