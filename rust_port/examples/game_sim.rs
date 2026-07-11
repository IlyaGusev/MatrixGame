//! Autonomous full-game simulation harness — the bug-catching
//! environment. Runs complete headless missions on real maps: enemy
//! AI sides play normally, the player side is driven by a scripted
//! "human" (builds robots at the base, issues capture/attack/move
//! orders through the same entry points the UI uses) or by the enemy
//! AI itself. Every second the world is swept for invariant
//! violations (NaN state, HP out of range, out-of-map objects,
//! effect/robot leaks, negative resources, stalled sides); panics are
//! caught and reported with a repro command line.
//!
//! Examples:
//!   cargo run --example game_sim                       # ATOLL, seed 1, scripted player
//!   cargo run --example game_sim -- --map dubna --seed 3 --minutes 30
//!   cargo run --example game_sim -- --all-maps --runs 2 --minutes 10
//!   cargo run --example game_sim -- --player ai        # AI plays the player side too
//!   cargo run --example game_sim -- --check-determinism
//!
//! Exit code: number of failed runs (0 = everything clean).

use matrixgame_rs::matrix_game::config::Resource;
use matrixgame_rs::matrix_game::logic::{
    building_mut, building_ref, cannon_ref, is_absence_wall, robot_ref, MapLogic,
};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::{MapStatic, ObjectType};
use matrixgame_rs::matrix_game::side::{CurrSel, Side, SideStatus};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::panic::AssertUnwindSafe;
use std::sync::Mutex;

const TAKT_MS: i32 = 50;

#[derive(Clone, Copy, PartialEq)]
enum PlayerMode {
    Script,
    Ai,
    Idle,
}

#[derive(Clone)]
struct Opts {
    minutes: f64,
    player: PlayerMode,
    verbose: bool,
}

struct Anomaly {
    t_ms: i64,
    kind: &'static str,
    detail: String,
}

struct RunResult {
    outcome: String,
    end_ms: i64,
    anomalies: Vec<Anomaly>,
    hash_trace: Vec<(i64, u64)>,
    panic: Option<String>,
}

static LAST_PANIC: Mutex<Option<String>> = Mutex::new(None);

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Error)
        .init();

    let mut maps: Vec<String> = Vec::new();
    let mut all_maps = false;
    let mut seed = 1i32;
    let mut runs = 1i32;
    let mut check_det = false;
    let mut opts = Opts {
        minutes: 15.0,
        player: PlayerMode::Script,
        verbose: false,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_else(|| {
                eprintln!("missing value for {}", args[*i - 1]);
                std::process::exit(2);
            })
        };
        match args[i].as_str() {
            "--map" => maps.push(next(&mut i)),
            "--all-maps" => all_maps = true,
            "--seed" => seed = next(&mut i).parse().unwrap(),
            "--runs" => runs = next(&mut i).parse().unwrap(),
            "--minutes" => opts.minutes = next(&mut i).parse().unwrap(),
            "--player" => {
                opts.player = match next(&mut i).as_str() {
                    "ai" => PlayerMode::Ai,
                    "script" => PlayerMode::Script,
                    "idle" => PlayerMode::Idle,
                    other => {
                        eprintln!("unknown player mode {other}");
                        std::process::exit(2);
                    }
                }
            }
            "--check-determinism" => check_det = true,
            "--verbose" => opts.verbose = true,
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let dat = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();
    install_weapon_matrices(&pkg);

    if all_maps {
        let mut v: Vec<String> = pkg
            .list_files()
            .into_iter()
            .filter(|f| f.ends_with(".CMAP"))
            .map(|s| s.to_string())
            .collect();
        v.sort();
        maps = v;
    } else if maps.is_empty() {
        maps.push("ATOLL".into());
    }

    // Report panics through LAST_PANIC but keep the default printout.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_default();
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic>".into());
        *LAST_PANIC.lock().unwrap() = Some(format!("{msg} at {loc}"));
        default_hook(info);
    }));

    let mut failed = 0u32;
    let total = maps.len() as i32 * runs;
    let mut done = 0;
    for map_name in &maps {
        let full = resolve_map(map_name);
        for r in 0..runs {
            let s = seed + r;
            done += 1;
            println!(
                "\n=== run {done}/{total}: map={full} seed={s} player={} minutes={} ===",
                mode_name(opts.player),
                opts.minutes
            );
            let res = run_guarded(&pkg, &dat, &full, s, &opts);
            let mut bad = report(&res, &full, s, &opts);
            if check_det && res.panic.is_none() {
                let res2 = run_guarded(&pkg, &dat, &full, s, &opts);
                if let Some(div) = first_divergence(&res.hash_trace, &res2.hash_trace) {
                    println!("ANOMALY nondet: first state-hash divergence at t={}s", div / 1000);
                    bad = true;
                } else {
                    println!("determinism: OK ({} checkpoints)", res.hash_trace.len());
                }
            }
            if bad {
                failed += 1;
            }
        }
    }
    println!("\n==== {failed}/{total} runs with findings ====");
    std::process::exit(failed.min(100) as i32);
}

fn mode_name(m: PlayerMode) -> &'static str {
    match m {
        PlayerMode::Script => "script",
        PlayerMode::Ai => "ai",
        PlayerMode::Idle => "idle",
    }
}

fn resolve_map(name: &str) -> String {
    if name.contains('/') {
        name.to_string()
    } else {
        let mut n = name.to_uppercase();
        if !n.ends_with(".CMAP") {
            n.push_str(".CMAP");
        }
        format!("MATRIX/MAP/{n}")
    }
}

/// Weapon mount matrices normally load in RobotsRenderer::new —
/// headless must populate them or every built robot comes out
/// weaponless (same workaround as ai_stall_probe).
fn install_weapon_matrices(pkg: &PkgArchive) {
    for idx in 1..=6usize {
        if let Ok(data) = pkg.read_file(&format!("MATRIX/ROBOT/ARMOR{idx}.VO")) {
            let vo = matrixgame_rs::matrix_lib::three_g::vector_object::parse_vo(&data).unwrap();
            let m = matrixgame_rs::matrix_game::map::weapon_matrix_from_vo(&vo);
            matrixgame_rs::matrix_game::map::set_weapon_matrix_for(
                matrixgame_rs::matrix_game::config::RobotUnitKind(idx as i32),
                m,
            );
        }
    }
}

fn run_guarded(pkg: &PkgArchive, dat: &Storage, map_path: &str, seed: i32, opts: &Opts) -> RunResult {
    *LAST_PANIC.lock().unwrap() = None;
    match std::panic::catch_unwind(AssertUnwindSafe(|| run_sim(pkg, dat, map_path, seed, opts))) {
        Ok(r) => r,
        Err(_) => RunResult {
            outcome: "PANIC".into(),
            end_ms: -1,
            anomalies: Vec::new(),
            hash_trace: Vec::new(),
            panic: LAST_PANIC.lock().unwrap().take().or(Some("<unknown>".into())),
        },
    }
}

fn report(res: &RunResult, map: &str, seed: i32, opts: &Opts) -> bool {
    if let Some(p) = &res.panic {
        println!("PANIC: {p}");
        println!(
            "repro: cargo run --example game_sim -- --map {map} --seed {seed} --player {} --minutes {}",
            mode_name(opts.player),
            opts.minutes
        );
        return true;
    }
    let mut by_kind: HashMap<&str, usize> = HashMap::new();
    for a in &res.anomalies {
        *by_kind.entry(a.kind).or_default() += 1;
    }
    println!(
        "outcome: {} at t={}s, anomalies: {:?}",
        res.outcome,
        res.end_ms / 1000,
        by_kind
    );
    !res.anomalies.is_empty()
}

fn first_divergence(a: &[(i64, u64)], b: &[(i64, u64)]) -> Option<i64> {
    for (x, y) in a.iter().zip(b.iter()) {
        if x != y {
            return Some(x.0);
        }
    }
    if a.len() != b.len() {
        return Some(a.last().map(|x| x.0).unwrap_or(0));
    }
    None
}

/// Deterministic LCG for the scripted player, independent of the game
/// RNG so the driver's choices never perturb game-internal streams.
struct SimRng(u64);
impl SimRng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
}

fn run_sim(pkg: &PkgArchive, dat: &Storage, map_path: &str, seed: i32, opts: &Opts) -> RunResult {
    let cmap = pkg.read_file(map_path).unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();

    let mut game = MapLogic::with_seed(seed);
    game.load_config(dat);
    if opts.player == PlayerMode::Ai {
        // Park the human side on an unused id so ensure_sides picks up
        // side 1 as a regular enemy-AI side.
        game.player_side = Side::new(100);
    }
    game.spawn_buildings(&map);
    game.spawn_ruins(&map);
    game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.init_effect_spawners(&map);
    game.accrue_resources(100_000);
    let stor = Storage::from_bytes(&cmap).unwrap();
    game.spawn_map_objects(&map, &stor);

    if opts.verbose {
        print_progress(&game);
    }

    let mut driver = SimRng(seed as u64 ^ 0x9e3779b97f4a7c15);
    let mut next_build_ms = 15_000i64;
    let mut next_order_ms = 20_000i64;

    let mut anomalies: Vec<Anomaly> = Vec::new();
    let mut kind_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut hash_trace: Vec<(i64, u64)> = Vec::new();
    // Per-robot (id, last pos, last time it moved) for stall detection.
    let mut last_move: HashMap<String, (glam::Vec2, i64)> = HashMap::new();
    let mut stall_reported: HashMap<i32, i64> = HashMap::new();
    let mut lone_since: Option<i64> = None;
    let mut outcome = String::from("TIMEOUT");

    let steps = (opts.minutes * 60.0 * 1000.0 / TAKT_MS as f64) as i64;
    for step in 0..steps {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            // Point effect culling at the action instead of a real camera.
            matrixgame_rs::matrix_game::map::set_frustum_center(battle_centroid(&game, &map));
            game.takt(TAKT_MS);
            game.graphic_takt(TAKT_MS);

            match opts.player {
                PlayerMode::Script => {
                    if game.elapsed_ms >= next_build_ms {
                        next_build_ms = game.elapsed_ms + 15_000 + driver.below(10_000) as i64;
                        script_build(&mut game);
                    }
                    if game.elapsed_ms >= next_order_ms {
                        next_order_ms = game.elapsed_ms + 12_000 + driver.below(12_000) as i64;
                        script_order(&mut game, &map, &mut driver);
                    }
                }
                PlayerMode::Ai | PlayerMode::Idle => {}
            }
        }

        drain_frame_queues(&mut game);

        if let Some(win) = game.pending_win_loose_dialog.take() {
            outcome = if win { "WIN".into() } else { "LOSE".into() };
            break;
        }

        // Once per game second: invariants + stall + progress.
        if step % 20 != 19 {
            continue;
        }
        check_invariants(&game, &map, &mut anomalies, &mut kind_counts);
        check_stalls(
            &game,
            opts,
            &mut last_move,
            &mut stall_reported,
            &mut anomalies,
            &mut kind_counts,
        );
        if game.elapsed_ms % 10_000 < 1000 {
            hash_trace.push((game.elapsed_ms, state_hash(&game)));
        }
        if opts.verbose && game.elapsed_ms % 60_000 < 1000 {
            print_progress(&game);
        }

        // AI-vs-AI has no player win path: stop when one force remains.
        let alive = sides_with_forces(&game);
        if alive.len() <= 1 {
            if let Some(t0) = lone_since {
                if game.elapsed_ms - t0 > 30_000 {
                    outcome = format!("LAST-SIDE-STANDING {alive:?}");
                    break;
                }
            } else {
                lone_since = Some(game.elapsed_ms);
            }
        } else {
            lone_since = None;
        }
    }

    for a in anomalies.iter().take(40) {
        println!("ANOMALY {} t={}s: {}", a.kind, a.t_ms / 1000, a.detail);
    }
    if anomalies.len() > 40 {
        println!("... {} more anomalies suppressed", anomalies.len() - 40);
    }
    RunResult {
        outcome,
        end_ms: game.elapsed_ms,
        anomalies,
        hash_trace,
        panic: None,
    }
}

/// Mirror of the app loop's per-frame queue drains (form_game.rs)
/// minus the renderer consumers — without this the pending vecs grow
/// forever in headless runs.
fn drain_frame_queues(game: &mut MapLogic) {
    game.objects.pending_sounds.clear();
    game.sound_queue.clear();
    game.objects.weapons.freed.clear();
    game.objects.pending_spots.clear();
    game.objects.pending_point_lights.clear();
    game.objects.pending_light_follow.clear();
    game.objects.pending_light_kill.clear();
    let _ = matrixgame_rs::matrix_game::interface::sound::drain();
}

fn battle_centroid(game: &MapLogic, map: &GameMap) -> [f32; 2] {
    let mut n = 0f32;
    let mut cx = 0f32;
    let mut cy = 0f32;
    for id in game.objects.iter_live() {
        if let Some(r) = robot_ref(&game.objects, id) {
            cx += r.pos_x;
            cy += r.pos_y;
            n += 1.0;
        }
    }
    if n > 0.0 {
        [cx / n, cy / n]
    } else {
        let gs = GameMap::GLOBAL_SCALE_MOVE;
        [
            map.size_move_x as f32 * gs * 0.5,
            map.size_move_y as f32 * gs * 0.5,
        ]
    }
}

/// Sides that still own a live base or robot.
fn sides_with_forces(game: &MapLogic) -> Vec<i32> {
    let mut v: Vec<i32> = Vec::new();
    for id in game.objects.iter_live() {
        let Some(obj) = game.objects.get(id) else { continue };
        let side = obj.side();
        if side <= 0 || v.contains(&side) {
            continue;
        }
        let owns = match obj.core().obj_type {
            ObjectType::RobotAi => obj.is_live(),
            ObjectType::Building => building_ref(&game.objects, id)
                .map(|b| b.is_live() && b.is_base())
                .unwrap_or(false),
            _ => false,
        };
        if owns {
            v.push(side);
        }
    }
    v.sort();
    v
}

// ---------------------------------------------------------------- player script

/// Queue the strongest affordable template on the player base, paying
/// for it the way commit_and_queue_robot does.
fn script_build(game: &mut MapLogic) {
    let cat = matrixgame_rs::matrix_game::interface::constructor::global_ai_robots();
    if cat.bots.is_empty() {
        return;
    }
    let base_id = game.objects.iter_live().find(|&id| {
        building_ref(&game.objects, id)
            .map(|b| b.is_live() && b.is_base() && b.side == game.player_side.id)
            .unwrap_or(false)
    });
    let Some(base_id) = base_id else { return };
    let queue_len = building_ref(&game.objects, base_id)
        .map(|b| b.build_stack.items())
        .unwrap_or(0);
    if queue_len >= 3 {
        return;
    }
    let bank: Vec<i32> = (0..4).map(|r| game.player_side.resources[r]).collect();
    // Catalogue is sorted strongest-first.
    let pick = cat
        .bots
        .iter()
        .find(|b| (0..4).all(|r| bank[r] >= b.resources[r]));
    let Some(bot) = pick else { return };
    let cfg = bot.to_robot_config();
    let queued = building_mut(&mut game.objects, base_id)
        .map(|b| b.queue_robot(cfg))
        .unwrap_or(false);
    if queued {
        for r in 0..4 {
            game.player_side
                .add_resource_amount(Resource::ALL[r], -bot.resources[r]);
        }
    }
}

/// Group up to 9 idle player robots and give them an order through
/// the same selection + pg_order_* path the UI uses.
fn script_order(game: &mut MapLogic, map: &GameMap, rng: &mut SimRng) {
    let pid = game.player_side.id;
    let idle: Vec<_> = game
        .objects
        .iter_live()
        .filter(|&id| {
            robot_ref(&game.objects, id)
                .map(|r| r.is_live() && r.side == pid && r.orders.is_empty())
                .unwrap_or(false)
        })
        .collect();
    if idle.is_empty() {
        return;
    }
    let take = (1 + rng.below(9) as usize).min(idle.len());

    game.create_group_from_object(idle[0]);
    for &id in idle.iter().skip(1).take(take - 1) {
        let ty = game
            .objects
            .get(id)
            .map(|o| o.core().obj_type)
            .unwrap_or(ObjectType::Empty);
        game.player_side.cur_sel_group.add_object(id, -4, ty);
    }
    if take > 1 {
        game.add_to_current_group();
    }
    game.player_side.curr_sel = CurrSel::RobotsSelected;

    match rng.below(100) {
        0..=34 => {
            let no = game.sel_group_to_logic_group();
            game.pg_order_auto_capture(map, no);
        }
        35..=59 => {
            let no = game.sel_group_to_logic_group();
            game.pg_order_auto_attack(map, no);
        }
        60..=74 => {
            let no = game.sel_group_to_logic_group();
            game.pg_order_auto_defence(map, no);
        }
        75..=89 => {
            if let Some(w) = random_passable_point(game, map, rng, idle[0]) {
                game.order_move_to_world(map, w);
            }
        }
        _ => {
            if let Some(w) = random_enemy_pos(game, rng) {
                game.order_attack_world(map, w);
            }
        }
    }
}

fn random_passable_point(
    game: &MapLogic,
    map: &GameMap,
    rng: &mut SimRng,
    rid: matrixgame_rs::matrix_game::map_static::ObjectId,
) -> Option<glam::Vec2> {
    let kind = robot_ref(&game.objects, rid).map(|r| r.chassis.kind_index())?;
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    for _ in 0..60 {
        let mx = 10 + rng.below(map.size_move_x.saturating_sub(20) as u32) as i32;
        let my = 10 + rng.below(map.size_move_y.saturating_sub(20) as u32) as i32;
        if is_absence_wall(map, kind, 4, mx, my) {
            return Some(glam::Vec2::new(mx as f32 * gs, my as f32 * gs));
        }
    }
    None
}

fn random_enemy_pos(game: &MapLogic, rng: &mut SimRng) -> Option<glam::Vec2> {
    let pid = game.player_side.id;
    let mut targets: Vec<glam::Vec2> = Vec::new();
    for id in game.objects.iter_live() {
        if let Some(r) = robot_ref(&game.objects, id) {
            if r.is_live() && r.side != pid && r.side > 0 {
                targets.push(glam::Vec2::new(r.pos_x, r.pos_y));
            }
        } else if let Some(b) = building_ref(&game.objects, id) {
            if b.is_live() && b.side != pid && b.side > 0 {
                targets.push(glam::Vec2::new(b.pos.x, b.pos.y));
            }
        }
    }
    if targets.is_empty() {
        None
    } else {
        Some(targets[rng.below(targets.len() as u32) as usize])
    }
}

// ---------------------------------------------------------------- invariants

fn push_anomaly(
    anomalies: &mut Vec<Anomaly>,
    counts: &mut HashMap<&'static str, usize>,
    t_ms: i64,
    kind: &'static str,
    detail: String,
) {
    let c = counts.entry(kind).or_default();
    *c += 1;
    // Keep the first few of each kind; the rest only bump the counter.
    if *c <= 5 {
        anomalies.push(Anomaly { t_ms, kind, detail });
    }
}

fn check_invariants(
    game: &MapLogic,
    map: &GameMap,
    anomalies: &mut Vec<Anomaly>,
    counts: &mut HashMap<&'static str, usize>,
) {
    let t = game.elapsed_ms;
    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let (w, h) = (map.size_move_x as f32 * gs, map.size_move_y as f32 * gs);
    let margin = 16.0 * gs;
    let mut robots = 0usize;

    for id in game.objects.iter_live() {
        if let Some(r) = robot_ref(&game.objects, id) {
            if !r.is_live() {
                continue;
            }
            robots += 1;
            let finite = r.pos_x.is_finite()
                && r.pos_y.is_finite()
                && r.velocity.x.is_finite()
                && r.velocity.y.is_finite()
                && r.hit_point.is_finite();
            if !finite {
                push_anomaly(anomalies, counts, t, "nan", format!(
                    "robot {id:?} side={} pos=({},{}) vel=({},{}) hp={}",
                    r.side, r.pos_x, r.pos_y, r.velocity.x, r.velocity.y, r.hit_point
                ));
                continue;
            }
            if r.hit_point > r.hit_point_max * 1.001 || r.hit_point <= 0.0 || r.hit_point_max <= 0.0 {
                push_anomaly(anomalies, counts, t, "hp-range", format!(
                    "robot {id:?} side={} hp={}/{} (live)",
                    r.side, r.hit_point, r.hit_point_max
                ));
            }
            if r.pos_x < -margin || r.pos_y < -margin || r.pos_x > w + margin || r.pos_y > h + margin
            {
                push_anomaly(anomalies, counts, t, "oob", format!(
                    "robot {id:?} side={} pos=({:.0},{:.0}) map=({:.0}x{:.0})",
                    r.side, r.pos_x, r.pos_y, w, h
                ));
            }
        } else if let Some(c) = cannon_ref(&game.objects, id) {
            if c.is_live() && (!c.hit_point.is_finite() || c.hit_point > c.hit_point_max * 1.001) {
                push_anomaly(anomalies, counts, t, "hp-range", format!(
                    "cannon {id:?} side={} hp={}/{}",
                    c.side, c.hit_point, c.hit_point_max
                ));
            }
        } else if let Some(b) = building_ref(&game.objects, id) {
            if b.is_live() && (!b.hit_point.is_finite() || b.hit_point > b.hit_point_max * 1.001) {
                push_anomaly(anomalies, counts, t, "hp-range", format!(
                    "building {id:?} side={} hp={}/{}",
                    b.side, b.hit_point, b.hit_point_max
                ));
            }
        }
    }

    if robots > 200 {
        push_anomaly(anomalies, counts, t, "robot-leak", format!("{robots} live robots"));
    }
    if game.effects.len() > 8000 {
        push_anomaly(anomalies, counts, t, "fx-leak", format!("{} effects", game.effects.len()));
    }

    let mut sides: Vec<&Side> = vec![&game.player_side];
    sides.extend(game.other_sides.iter());
    for s in sides {
        if s.status == SideStatus::None {
            continue;
        }
        for r in 0..4 {
            if s.resources[r] < 0 {
                push_anomaly(anomalies, counts, t, "neg-res", format!(
                    "side {} resources={:?}",
                    s.id, s.resources
                ));
                break;
            }
        }
    }
}

fn check_stalls(
    game: &MapLogic,
    opts: &Opts,
    last_move: &mut HashMap<String, (glam::Vec2, i64)>,
    stall_reported: &mut HashMap<i32, i64>,
    anomalies: &mut Vec<Anomaly>,
    counts: &mut HashMap<&'static str, usize>,
) {
    let now = game.elapsed_ms;
    let mut per_side: HashMap<i32, (usize, usize)> = HashMap::new(); // (robots, moved recently)
    for id in game.objects.iter_live() {
        let Some(r) = robot_ref(&game.objects, id) else { continue };
        if !r.is_live() {
            continue;
        }
        let pos = glam::Vec2::new(r.pos_x, r.pos_y);
        let e = last_move.entry(format!("{id:?}")).or_insert((pos, now));
        if (pos - e.0).length() > 2.0 {
            *e = (pos, now);
        }
        let s = per_side.entry(r.side).or_insert((0, 0));
        s.0 += 1;
        if now - e.1 < 180_000 {
            s.1 += 1;
        }
    }
    for (&side, &(robots, moving)) in &per_side {
        // Only the enemy AI promises motion; a scripted/idle player
        // standing guard (auto-defence) is legitimate.
        if opts.player != PlayerMode::Ai && side == game.player_side.id {
            continue;
        }
        if robots >= 2
            && moving == 0
            && now - stall_reported.get(&side).copied().unwrap_or(i64::MIN / 2) > 300_000
        {
            stall_reported.insert(side, now);
            push_anomaly(anomalies, counts, now, "stall", format!(
                "side {side}: {robots} live robots, none moved for 3min"
            ));
            dump_stalled_side(game, side);
        }
    }
}

/// AI-state dump for a stalled side (condensed ai_stall_probe format):
/// team actions, live logic groups, and each frozen robot's orders.
fn dump_stalled_side(game: &MapLogic, side: i32) {
    if let Some(su) = game.side_by_id(side) {
        for (ti, t) in su.teams.iter().enumerate() {
            if t.robot_cnt == 0 {
                continue;
            }
            println!(
                "  team {ti}: robots={} action={:?}/r{} path={:?} target_region={} war={} stay={} wait_union={} region_mass={}",
                t.robot_cnt, t.action.ty, t.action.region, t.action.region_path,
                t.target_region, t.war, t.stay, t.wait_union, t.region_mass,
            );
        }
        for (gi, lg) in su.logic_groups.iter().enumerate() {
            if lg.robots_cnt <= 0 {
                continue;
            }
            println!(
                "  group {gi}: team={} robots={} strength={:.0} war={} action={:?}/r{}",
                lg.team, lg.robots_cnt, lg.strength, lg.war, lg.action.ty, lg.action.region,
            );
        }
    }
    for id in game.objects.iter_live() {
        let Some(r) = robot_ref(&game.objects, id) else { continue };
        if r.side != side || !r.is_live() {
            continue;
        }
        let orders: Vec<String> = r
            .orders
            .iter()
            .map(|o| format!("{:?}/{:?}", o.ty, o.phase))
            .collect();
        println!(
            "  robot {id:?} pos=({:.0},{:.0}) state={:?} team={} group={} orders={orders:?} place={} target={:?}",
            r.pos_x, r.pos_y, r.state, r.team, r.group_logic, r.env.place, r.env.target,
        );
    }
}

fn state_hash(game: &MapLogic) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    game.elapsed_ms.hash(&mut h);
    game.effects.len().hash(&mut h);
    for id in game.objects.iter_live() {
        if let Some(r) = robot_ref(&game.objects, id) {
            r.side.hash(&mut h);
            r.pos_x.to_bits().hash(&mut h);
            r.pos_y.to_bits().hash(&mut h);
            r.hit_point.to_bits().hash(&mut h);
            (r.orders.iter().count() as u32).hash(&mut h);
        } else if let Some(c) = cannon_ref(&game.objects, id) {
            c.side.hash(&mut h);
            c.hit_point.to_bits().hash(&mut h);
        } else if let Some(b) = building_ref(&game.objects, id) {
            b.side.hash(&mut h);
            b.hit_point.to_bits().hash(&mut h);
        }
    }
    let mut sides: Vec<&Side> = vec![&game.player_side];
    sides.extend(game.other_sides.iter());
    for s in sides {
        s.id.hash(&mut h);
        s.resources.hash(&mut h);
        s.robots_cnt.hash(&mut h);
    }
    h.finish()
}

fn print_progress(game: &MapLogic) {
    let mut per_side: HashMap<i32, (usize, usize, usize)> = HashMap::new(); // robots, cannons, buildings
    for id in game.objects.iter_live() {
        let Some(obj) = game.objects.get(id) else { continue };
        if !obj.is_live() {
            continue;
        }
        let e = per_side.entry(obj.side()).or_default();
        match obj.core().obj_type {
            ObjectType::RobotAi => e.0 += 1,
            ObjectType::Cannon => e.1 += 1,
            ObjectType::Building => e.2 += 1,
            _ => {}
        }
    }
    let mut rows: Vec<_> = per_side.into_iter().collect();
    rows.sort();
    let res: Vec<(i32, [i32; 4])> = std::iter::once(&game.player_side)
        .chain(game.other_sides.iter())
        .map(|s| (s.id, s.resources))
        .collect();
    println!(
        "t={:>5}s sides(r/c/b)={:?} res={:?} fx={}",
        game.elapsed_ms / 1000,
        rows,
        res,
        game.effects.len()
    );
}
