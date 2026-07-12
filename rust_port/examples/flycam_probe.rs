//! Headless check of the FLYCAM autopilot (SAutoFlyData port):
//! damage feeds war pairs, the camera flies toward the fight and
//! stays finite. Run: `cargo run --example flycam_probe`

use matrixgame_rs::matrix_game::camera::{AutoFlyData, Camera};
use matrixgame_rs::matrix_game::effects::weapon::WEAPON_LASER;
use matrixgame_rs::matrix_game::logic::MapLogic;
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::robot::{ChassisKind, Robot, RobotState};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg")?)?;
    let map = GameMap::from_cmap_bytes(&pkg.read_file("MATRIX/MAP/ATOLL.CMAP")?)?;
    let matrix_data = Storage::from_bytes(&std::fs::read("../Data/robots.dat")?)?;

    let mut game = MapLogic::with_seed(7);
    game.load_config(&matrix_data);

    // First dry spot with dry neighbors (ATOLL is mostly water).
    let mut land = None;
    'scan: for gy in (4..map.size_y - 4).step_by(2) {
        for gx in (4..map.size_x - 4).step_by(2) {
            let (x, y) = (gx as f32 * 20.0, gy as f32 * 20.0);
            if map.get_z(x, y) > 0.0 && map.get_z(x + 60.0, y) > 0.0 {
                land = Some((x, y));
                break 'scan;
            }
        }
    }
    let (fx, fy) = land.expect("no land found");

    let spawn = |game: &mut MapLogic, x: f32, y: f32, side: i32| {
        let z = map.get_z(x, y);
        let mut r = Robot::new(glam::Vec3::new(x, y, z), side, ChassisKind::Track);
        r.state = RobotState::Idle;
        game.objects.spawn(Box::new(r))
    };
    let a = spawn(&mut game, fx, fy, 1);
    let b = spawn(&mut game, fx + 60.0, fy, 2);

    game.objects.fly_cam = true;

    let mut cam = Camera::new(16.0 / 9.0);
    cam.set_map(map.size_x as f32 * 20.0, map.size_y as f32 * 20.0);
    // Park the camera far from the fight.
    cam.set_xy_strategy([1500.0, 1500.0]);
    cam.takt(50.0);
    let start = cam.eye_pos_world();

    let mut afd = AutoFlyData::new(&cam);
    let mut pair_events = 0usize;
    for i in 0..600 {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
        }
        // Poke both robots so the Damage path emits war pairs.
        if i % 10 == 0 {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.objects.apply_damage(
                a,
                WEAPON_LASER,
                glam::Vec3::new(fx, fy, 10.0),
                glam::Vec3::X,
                2,
                Some(b),
            );
        }
        let pairs: Vec<_> = game.objects.pending_war_pairs.drain(..).collect();
        pair_events += pairs.len();
        for (t, at) in pairs {
            afd.add_war_pair(t, at, &game.objects);
        }
        afd.takt(
            50.0,
            &mut cam,
            &map,
            &game.objects,
            &mut game.rng,
            game.elapsed_ms as i32,
            false,
        );
        let e = cam.eye_pos_world();
        assert!(e.is_finite(), "camera went non-finite at tick {i}: {e}");
        if i % 100 == 0 {
            let f = cam.forward();
            println!(
                "tick {i:3}: eye {e}, forward.z {:.3} (cos pitch {:.3})",
                f.z, -f.z
            );
        }
    }

    let end = cam.eye_pos_world();
    let fight = glam::Vec3::new(fx + 30.0, fy, map.get_z(fx + 30.0, fy));
    let d_start = (start - fight).truncate().length();
    let d_end = (end - fight).truncate().length();
    println!("war-pair events: {pair_events}");
    println!("start {start} ({d_start:.0} from fight)");
    println!("end   {end} ({d_end:.0} from fight)");
    assert!(pair_events > 0, "damage never queued a war pair");
    assert!(
        d_end < d_start * 0.5,
        "camera did not fly toward the fight: {d_start:.0} -> {d_end:.0}"
    );
    println!("OK: fly-cam chased the war pair");
    Ok(())
}
