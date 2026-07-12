# Game Simulation Harness (`examples/game_sim.rs`)

Autonomous headless full-game simulator — the environment for catching
logic bugs without a display. It runs complete missions on real maps
with all sides playing, sweeps the world for invariant violations every
game second, and reports anomalies with a repro command line.

## Quick start

```bash
cd rust_port
cargo run --example game_sim                                  # ATOLL, seed 1, scripted player, 15 min
cargo run --example game_sim -- --map dubna --seed 3 --minutes 30
cargo run --example game_sim -- --all-maps --runs 2 --minutes 10
cargo run --example game_sim -- --player ai --minutes 40      # AI plays the player side too
cargo run --example game_sim -- --check-determinism
```

Requires `../Data/robots.pkg` and `../Data/robots.dat`. Exit code =
number of runs with findings (0 = clean), capped at 100.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--map NAME` | ATOLL | Short name (`dubna`) or full path (`MATRIX/MAP/DUBNA.CMAP`); repeatable |
| `--all-maps` | off | Every `.CMAP` in robots.pkg |
| `--seed N` | 1 | Game RNG + player-driver seed |
| `--runs N` | 1 | Seeds `seed..seed+N` per map |
| `--minutes M` | 15 | Simulated game minutes per run |
| `--player ai\|script\|idle` | script | Player-side driver (below) |
| `--check-determinism` | off | Run each config twice, compare 10s state hashes |
| `--verbose` | off | Initial (t=0) + per-minute side/resource/effects progress lines |

## Player drivers

- **script** — a scripted "human": queues the strongest affordable
  template on the player base (paying like `commit_and_queue_robot`),
  and every ~12-24s groups up to 9 idle robots and issues an order
  through the same selection + `pg_order_*` path the UI uses
  (auto-capture 35%, auto-attack 25%, auto-defence 15%, random move
  15%, attack at enemy 10%). Exercises `side_player.rs` order paths.
- **ai** — parks the human side on id 100 so `ensure_sides_from_objects`
  registers side 1 as a regular enemy-AI side: a symmetric N-way AI
  battle. Full games end via LAST-SIDE-STANDING. Exercises `side_ai.rs`
  for every side.
- **idle** — player side does nothing (base sits there); pure
  enemy-AI-vs-enemy-AI plus a passive victim.

## Invariants checked (once per game second)

| Kind | Meaning |
|------|---------|
| `nan` | Non-finite robot position/velocity/HP |
| `hp-range` | Live object HP ≤ 0, HP > max, or max ≤ 0 (robots, cannons, buildings) |
| `oob` | Robot outside map bounds + 16-cell margin |
| `robot-leak` | > 200 live robots |
| `fx-leak` | > 8000 live effects |
| `neg-res` | Active side with a negative resource amount |
| `stall` | AI-driven side with ≥ 2 live robots where none moved > 2 units in 3 min (AI deadlock detector, from `ai_stall_probe`). A scripted/idle player side is exempt — standing guard is legitimate. On detection the side's full AI state is dumped: team actions, logic groups, and each frozen robot's orders/place/target/env/nearest-enemy/path state |
| `no-fire-standoff` | Hostile robot pair within 60 units for 60s+ with neither holding a FIRE order (the "wedged columns ignoring each other" screenshot) |
| `nondet` | `--check-determinism` found a state-hash divergence between two identical runs |
| PANIC | Any panic — caught per run, reported with location + repro command; a sweep continues with the next run |

First 5 of each kind print details; the rest just count. Anomalies do
not stop a run.

## Outcomes

- `WIN` / `LOSE` — the real `check_status` → `pending_win_loose_dialog`
  path fired for the player side.
- `LAST-SIDE-STANDING [ids]` — ≤ 1 side still owns a base or robot for
  30s (how ai-player games end).
- `TIMEOUT` — `--minutes` elapsed, nothing conclusive; fine for short
  smoke runs.

## Reproducing a finding

Every run is deterministic given (map, seed, player mode): rerun with
the printed repro line and add `--verbose`. To debug closely, drop the
time to just past the anomaly timestamp and add prints / a robot
micro-trace (see the tail of `examples/ai_stall_probe.rs` for a
ready-made pattern: order state, passability grid around a stuck robot,
per-takt movement trace).

## How it works / extending

Mission init is the standard headless recipe (same as
`ai_stall_probe`): load CMAP + `robots.dat`, install weapon-mount
matrices from `ARMOR*.VO` (or AI-built robots come out weaponless),
then `spawn_buildings/spawn_ruins/spawn_robots/ensure_sides_from_objects/
apply_side_resources/init_effect_spawners/accrue_resources/
spawn_map_objects`. Each 50ms frame runs inside `MapScope::enter`, calls
`takt(50)` + `graphic_takt(50)`, sets the frustum center to the robot
centroid (effect culling), and drains the pending queues the renderer
normally consumes (`drain_frame_queues`) so headless runs don't leak.

Add a new invariant in `check_invariants` (or a new sweep alongside
`check_stalls`) and route violations through `push_anomaly` — dedup and
reporting come for free. Add new player behaviors in `script_order`.

## Baseline status (2026-07-11)

Full sweep: all 84 maps × 8 min, scripted player — 0 panics, 0 state
corruption (nan/hp/oob/leaks), 35 legitimate LOSE outcomes (several
maps are survival scenarios where the player spawns without a base,
e.g. SANATORY). Determinism verified on ATOLL, DUSKBATTLE, KRATER,
REACTOR (script + ai modes, including back-to-back runs in one
process).

Bugs found by this environment and fixed (see robot.rs / map.rs):

- **Capture-order deadlocks** (three related mechanisms, all producing
  side-wide AI freezes on 10+ maps): the stuck-watchdog GetLost sidestep
  destroyed capture-approach MoveTos without saving the destination; a
  stale pre-capture MoveReturn defeated the step-aside save-guard and
  later yanked robots away mid-capture (ending up "Capturing" from
  across the map, permanently claiming the factory); and
  `can_break_order` was missing the C++ `MMFLAG_FULLAUTO` OR-term.
  Fixes: watchdog/step-aside refresh the return point with the live
  capture destination, `capture_factory()` clears stale MoveReturns,
  CaptureMoving re-plots via the Empty phase when both approach and
  return are gone (documented deviation — the C++ strands forever),
  `map::set_full_auto` ports MMFLAG_FULLAUTO.
- **Cross-mission time leak** — `MapScope::drop` didn't reset
  `CURRENT_ELAPSED_MS`, so any second mission in one process seeded
  spawn-time timers (cannon `fire_next_think_time` etc.) from the
  previous mission's end time, diverging entire battles (also affects
  in-app mission restarts). Found by TRIDENT stalling only as run 2.
- **Zombie carried robots** — the robot death path didn't detach the
  transport flyer (C++ does at MatrixRobot.cpp:1302-1310/5158-5161),
  so a robot killed mid-delivery was later "dropped" by its flyer,
  overwriting DIP with Falling: a live robot at negative HP. Found as
  an `hp-range` anomaly on SUMMER4_2E seed 2 (script mode).
- **Upper-path wedge / no-fire standoff** (user screenshot on ATOLL;
  `stall` repro: atoll seed 3 script) — three cooperating fixes:
  (1) zone pathfinding routed size-4 robots through links narrower
  than the robot footprint (`near_zone_connect_size` < 4 — authored
  map data the C++ loads but never reads), so columns marched into a
  gap `FindLocalPath` can never walk and froze; `link_blocked` in
  road_network.rs now skips narrow links (documented deviation).
  (2) A MoveTo whose local path keeps failing for 3s — impassable even
  with no robot blockers — now poisons its place and reteams
  (`local_path_escape`, robot.rs) instead of grinding forever like the
  C++. (3) GatherInfo AddIgnored enemies at point-blank range when
  PlaceList couldn't reach them through the blocked pass; an enemy
  inside fire range with clear LOS is now always added (gather_info,
  logic.rs). Also guarded `find_path_in_zone` against zone -1 (was an
  index panic; C++ is UB/ERROR_E there).
  Debug aid: `MG_TRACE_STALL=1` prints zone-path and local-path
  failures with a no-blockers retry verdict.

Residual known standstills (verified to match the original C++
behavior, not port bugs): capturer starved of approach cells by
group-mates parked on every place near the factory; base-capture
approach jammed against the base wall from a bad angle; passive
TaktCaptureCandidate capture of a factory with no pathable approach
(the port busy-retries where the C++ freezes silently).

Debug aid: `MG_TRACE_CAPTURE=1` prints every event that destroys or
replaces a capture-marked MoveTo (the recurring deadlock bug class).
Note: a side reported as "defence-idle" (all teams Defence, war down,
robots holding places) is the AI's designed idle posture, not a stall
— e.g. both sides on TERRON.

Related existing probes: `ai_stall_probe` (AI deadlock deep-dive),
`heavy_battle_bench` (takt CPU cost), `attack_order_sim`,
`collision_sim`, `rocket_battle_probe`, and `tests/test_ai.rs` (CI-style
smoke assertion).
