# SIM.md — headless simulation & measurement playbook

How to run the game logic without a window/GPU: reproduce gameplay bugs,
measure balance/behavior, profile CPU cost. Everything runs on the REAL
map data and REAL config (`../Data/robots.pkg` + `../Data/robots.dat`)
— no mocks, so numbers transfer to the shipped game.

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
- Rendering never runs; anything visual-only still ACCUMULATES in
  `objs.pending_effects` → `game.effects`, so effect counts/kinds are
  observable.
- Step size: 50ms ≈ a 20 FPS frame. Results are step-size-insensitive
  because logic always advances in fixed 10ms slices internally.
- Full-mission extras (what `game_sim` adds on top): install weapon-mount
  matrices from `ARMOR*.VO` (or AI-built robots come out weaponless),
  `spawn_ruins`, `init_effect_spawners`, `spawn_map_objects`, plus
  `graphic_takt(50)` and `drain_frame_queues` per frame so headless runs
  don't leak.

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
Give robots weapons via `r.config` (a `RobotConfig`). Find a passable
spawn cell with `logic::is_absence_wall(&map, 2, 4, mx, my)`;
world = cell × `GLOBAL_SCALE_MOVE`. Remove actors between rounds with
`game.objects.remove(id)`.

## Observation points (what to assert / print)

- Robots: `robot_ref(&game.objects, id)` → `hit_point`, `env.target/
  target_attack/enemies`, `orders`, `team`, `group_logic`, `map_x/y`.
- Sides: `game.side_by_id(sid)` → `teams[i].action.ty`, `robots_cnt`,
  `strength`, `war_side`, `logic_groups`.
- Turrets: cast via `object_cannon::Cannon` (`c.target`, `c.kind`) and
  poll `c.weapons` → `w.is_fire()` for is-it-shooting.
- Effects: `game.effects` (match on `GameEffect::MovingObject` etc.).
- Damage: snapshot `hit_point` before/after N sim-seconds.

Aggregate over time, don't eyeball single ticks: the turret study
sampled every ~200ms for 6 sim-minutes into per-turret counters, which
exposed "aims 19% of the time but fires only 12% of it" for one kind.

When a ratio looks wrong, instrument the decision point directly: drop
temporary `static GATE: [AtomicU64; N]` counters into the branch (one
per bail-out reason), print after the sim. That turned the above into
"1864 out-of-range vs 216 fires vs 0 trace-fails" — authentic behavior,
not a broken aim trace. Remove the counters after diagnosis.

## CPU profiling

**Fidelity rule learned the hard way**: a battle bench without
`spawn_map_objects` flatters everything ~10× — decorative objects share
the object arena. Always spawn them:
`game.spawn_map_objects(&map, &Storage::from_bytes(&cmap)?)`.

Native: wrap `game.takt(50)` in `std::time::Instant` and accumulate
avg/worst; run with `cargo run --release --example <probe>`. Reference
(2026-07, ATOLL + map objects, 26-robot missile brawl, post
typed-index + gather-reorder fixes): **~4.4ms avg wasm / ~1.3ms steady**
per 50ms frame.

wasm (node) — native numbers can mislead (allocator, JIT warmup). The
`wasm-bench` feature exposes the same battle as `bench_battle(pkg,
dat)` with per-phase timers inside `MapLogic::takt` (`TAKT_PHASE_US`:
gather / proceed / sides / fx / rest):

```
wasm-pack build --target nodejs --out-dir pkg-bench --features wasm-bench
node bench_node.mjs
```

This isolated the three battle-FPS killers of 2026-07: (1) traces/scans
walking every decorative arena slot → typed `unit_ids`/`mapobject_ids`
indices; (2) `gather_info` tracing before the known-enemy check →
reordered; (3) per-frame Vec collections in `calc_proj` → precomputed
per-chassis.

In-browser: the app logs once per second (console + on-screen FPS text):

```
fps: 29.6 (...) | per-frame ms: takt=11.10 gfx=0.03 ui=0.00 syncR=5.00 sync=1.30 render=7.70 | robots=25 effects=390 bb=513
```

takt = pure simulation, gfx = per-object graphic takt, ui = interface,
syncR = robot instance + shadow prep, sync = camera/terrain/minimap
prep, render = queue build + pass encoding + submit. Implemented as
`platform::now_secs()` spans in `form_game.rs` (`perf_acc`).

## game_sim — the autonomous bug-catcher (`examples/game_sim.rs`)

Runs complete missions on real maps with all sides playing, sweeps the
world for invariant violations every game second, and reports anomalies
with a repro command line. Exit code = number of runs with findings.

```bash
cargo run --example game_sim                                  # ATOLL, seed 1, scripted player, 15 min
cargo run --example game_sim -- --map dubna --seed 3 --minutes 30
cargo run --example game_sim -- --all-maps --runs 2 --minutes 10
cargo run --example game_sim -- --player ai --minutes 40      # AI plays the player side too
cargo run --example game_sim -- --check-determinism
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--map NAME` | ATOLL | Short name (`dubna`) or full path; repeatable |
| `--all-maps` | off | Every `.CMAP` in robots.pkg |
| `--seed N` | 1 | Game RNG + player-driver seed |
| `--runs N` | 1 | Seeds `seed..seed+N` per map |
| `--minutes M` | 15 | Simulated game minutes per run |
| `--player ai\|script\|idle` | script | Player-side driver (below) |
| `--check-determinism` | off | Run each config twice, compare 10s state hashes |
| `--verbose` | off | t=0 + per-minute side/resource/effects lines |

Player drivers:
- **script** — a scripted "human": queues the strongest affordable
  template on the base and every ~12-24s groups up to 9 idle robots and
  issues an order through the same selection + `pg_order_*` path the UI
  uses (mix of auto-capture/attack/defence/move). Exercises
  `side_player.rs`.
- **ai** — parks the human side on id 100 so side 1 becomes a regular
  enemy-AI side: symmetric N-way AI battle ending via
  LAST-SIDE-STANDING. Exercises `side_ai.rs` for every side.
- **idle** — player side does nothing; AI-vs-AI plus a passive victim.

Invariants (checked once per game second; first 5 of each kind print
details, the rest count; anomalies don't stop a run):

| Kind | Meaning |
|------|---------|
| `nan` | Non-finite robot position/velocity/HP |
| `hp-range` | Live object HP ≤ 0, HP > max, or max ≤ 0 |
| `oob` | Robot outside map bounds + 16-cell margin |
| `robot-leak` / `fx-leak` | > 200 live robots / > 8000 live effects |
| `neg-res` | Active side with a negative resource |
| `stall` | AI side with ≥ 2 live robots, none moved > 2 units in 3 min; dumps full AI state (team actions, logic groups, per-robot orders/place/target/env/path). Scripted/idle player exempt |
| `no-fire-standoff` | Hostile pair within 60 units for 60s+ with neither holding a FIRE order |
| `nondet` | State-hash divergence between two identical runs |
| PANIC | Caught per run, reported with location + repro; sweep continues |

Outcomes: `WIN`/`LOSE` (real `check_status` path fired for the player),
`LAST-SIDE-STANDING [ids]` (≤ 1 side owns a base or robot for 30s),
`TIMEOUT` (fine for short smoke runs).

Reproducing: every run is deterministic given (map, seed, player mode)
— rerun the printed repro line with `--verbose`, drop `--minutes` to
just past the anomaly, add prints or a robot micro-trace (see the tail
of `examples/ai_stall_probe.rs` for a ready-made pattern).

Extending: add invariants in `check_invariants` (route through
`push_anomaly` — dedup/reporting come free); add player behaviors in
`script_order`.

Debug env vars:
- `MG_TRACE_CAPTURE=1` — prints every event that destroys/replaces a
  capture-marked MoveTo (the recurring deadlock bug class).
- `MG_TRACE_STALL=1` — prints zone-path and local-path failures with a
  no-blockers retry verdict.

## game_sim baseline (2026-07-11)

Full sweep: all 84 maps × 8 min, scripted player — 0 panics, 0 state
corruption, 35 legitimate LOSE outcomes (survival maps where the player
spawns without a base, e.g. SANATORY). Determinism verified on ATOLL,
DUSKBATTLE, KRATER, REACTOR (script + ai, including back-to-back runs
in one process).

Bugs found and fixed (see robot.rs / map.rs):

- **Capture-order deadlocks** — three mechanisms producing side-wide AI
  freezes on 10+ maps: watchdog GetLost destroyed capture-approach
  MoveTos without saving the destination; a stale pre-capture
  MoveReturn yanked robots away mid-capture; `can_break_order` missed
  the C++ `MMFLAG_FULLAUTO` OR-term. Fixed by refreshing return points
  with the live capture destination, clearing stale MoveReturns in
  `capture_factory()`, re-plotting via the Empty phase when both paths
  are gone (documented deviation — the C++ strands forever), and
  porting `map::set_full_auto`.
- **Cross-mission time leak** — `MapScope::drop` didn't reset
  `CURRENT_ELAPSED_MS`, so a second mission in one process seeded
  spawn-time timers from the previous mission's end time (also affects
  in-app restarts). Found by TRIDENT stalling only as run 2.
- **Zombie carried robots** — robot death didn't detach the transport
  flyer (C++ does at MatrixRobot.cpp:1302-1310/5158-5161); a robot
  killed mid-delivery was later "dropped", overwriting DIP with
  Falling: live robot at negative HP. Found as `hp-range` on SUMMER4_2E
  seed 2.
- **Upper-path wedge / no-fire standoff** (ATOLL; repro: atoll seed 3
  script) — three cooperating fixes: (1) `link_blocked` in
  road_network.rs skips zone links narrower than the robot footprint
  (`near_zone_connect_size` < 4 — authored data the C++ never reads;
  documented deviation); (2) a MoveTo whose local path fails for 3s
  with no robot blockers poisons its place and reteams
  (`local_path_escape`, robot.rs); (3) `gather_info` always adds an
  enemy inside fire range with clear LOS instead of AddIgnoring it.
  Also guarded `find_path_in_zone` against zone -1 (was an index panic;
  C++ is UB there).

Residual known standstills (verified to match the original C++, not
port bugs): capturer starved of approach cells by group-mates parked
around the factory; base-capture approach jammed against the base wall;
passive TaktCaptureCandidate capture of a factory with no pathable
approach (the port busy-retries where the C++ freezes silently).
Note: a side reported "defence-idle" (all teams Defence, war down,
robots holding places) is the AI's designed idle posture, not a stall —
e.g. both sides on TERRON.

## Where the existing harnesses live

- `examples/game_sim.rs` — the autonomous full-game simulator above.
- `tests/test_ai.rs` — end-to-end AI acceptance: real map, 4 sim-min,
  asserts AI sides build robots, form teams, pick actions. Slow (~30s),
  skips gracefully when `Data/` is absent.
- `examples/ai_stall_probe.rs` — AI deadlock deep-dive (order state,
  passability grid around a stuck robot, per-takt movement trace).
- `examples/heavy_battle_bench.rs` — takt CPU cost.
- `examples/rocket_battle_probe.rs` — full-map AI battle + per-turret
  aim/fire statistics + takt timing.
- `examples/rocket_turret_probe.rs` / `laser_turret_probe.rs` — single
  turret vs robot matrix (distance/height/motion scans).
- `examples/attack_order_sim.rs`, `collision_sim.rs` — older
  order/movement sims, same pattern.
- `src/matrix_game/side_ai.rs` `#[cfg(test)]` — synthetic 2-region road
  network fixture (`map_with_regions`) for fast AI unit tests.

Probes are kept in-tree on purpose (repo convention — dozens of
`probe_*.rs` / `*_sim.rs` examples): a reproduced bug should leave its
reproducer behind.
