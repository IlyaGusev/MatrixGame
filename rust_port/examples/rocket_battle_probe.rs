//! Observe turret behavior in a real AI battle on ATOLL: per-cannon
//! aim/fire ratios over simulated minutes.
//!
//!   cargo run --example rocket_battle_probe

use matrixgame_rs::matrix_game::logic::MapLogic;
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::MapStatic;
use matrixgame_rs::matrix_game::object_cannon::Cannon;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use std::collections::HashMap;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Error)
        .init();
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let matrix_data =
        Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();

    let mut game = MapLogic::new();
    game.load_config(&matrix_data);
    {
        use matrixgame_rs::matrix_game::effects::weapon::*;
        let t = matrixgame_rs::matrix_game::config::global().weapon_radius.table;
        for (name, w) in [
            ("CANNON0(mg)", WEAPON_CANNON0),
            ("CANNON1(gun)", WEAPON_CANNON1),
            ("CANNON2(laser)", WEAPON_CANNON2),
            ("CANNON3(rocket)", WEAPON_CANNON3),
        ] {
            let i = weap_to_index(w);
            println!("{name}: radius={}", i.map(|i| t[i]).unwrap_or(-1.0));
        }
    }
    game.spawn_buildings(&map);
    game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.accrue_resources(100_000);

    // (kind, side) -> [aim_ticks, fire_ticks, alive_ticks]
    let mut stats: HashMap<(i32, String), [u64; 3]> = HashMap::new();
    let mut worst_ms = 0.0f64;
    let mut sum_ms = 0.0f64;
    let mut frames = 0u64;
    for step in 0..(6 * 60 * 20) {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        let t0 = std::time::Instant::now();
        game.takt(50);
        let el = t0.elapsed().as_secs_f64() * 1000.0;
        if step > 2400 {
            // battle phase only (after 2 sim-minutes)
            worst_ms = worst_ms.max(el);
            sum_ms += el;
            frames += 1;
        }
        if step % 4 != 0 {
            continue;
        }
        let ids: Vec<_> = game.objects.iter_live().collect();
        for id in ids {
            let Some(obj) = game.objects.get(id) else { continue };
            if !matches!(
                obj.core().obj_type,
                matrixgame_rs::matrix_game::map_static::ObjectType::Cannon
            ) {
                continue;
            }
            let c: &Cannon = unsafe { &*(obj as *const dyn MapStatic as *const Cannon) };
            if !c.is_live() {
                continue;
            }
            let key = (c.kind, format!("{:?}", id));
            let e = stats.entry(key).or_insert([0, 0, 0]);
            e[2] += 1;
            if c.target.is_some() {
                e[0] += 1;
            }
            let firing = c
                .weapons
                .iter()
                .filter_map(|&w| game.objects.weapons.get(w))
                .any(|w| w.is_fire());
            if firing {
                e[1] += 1;
            }
        }
    }
    let mut per_kind: HashMap<i32, [u64; 3]> = HashMap::new();
    for ((kind, _), v) in &stats {
        let e = per_kind.entry(*kind).or_insert([0, 0, 0]);
        for i in 0..3 {
            e[i] += v[i];
        }
    }
    println!(
        "takt(50ms) cost during battle: avg={:.2}ms worst={:.2}ms over {} frames",
        sum_ms / frames.max(1) as f64,
        worst_ms,
        frames
    );
    for (kind, v) in &per_kind {
        println!(
            "kind={kind} aim_ticks={} fire_ticks={} alive_ticks={} aim%={:.0} fire-when-aim%={:.0}",
            v[0],
            v[1],
            v[2],
            100.0 * v[0] as f64 / v[2].max(1) as f64,
            100.0 * v[1] as f64 / v[0].max(1) as f64,
        );
    }
}
