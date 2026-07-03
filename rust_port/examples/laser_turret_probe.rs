//! Headless probe for the laser turret (cannon kind 3, WEAPON_CANNON2)
//! on the real map + real config: logs fire state, beam endpoints and
//! shots over time.

use matrixgame_rs::matrix_game::logic::{robot_mut, robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::map_static::MapStatic;
use matrixgame_rs::matrix_game::object_cannon::Cannon;
use matrixgame_rs::matrix_game::robot::{ChassisKind, Robot, RobotState};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Warn)
        .init();
    let data = std::fs::read("../Data/robots.pkg").expect("robots.pkg");
    let pkg = PkgArchive::from_bytes(data).unwrap();
    let cmap_name = pkg
        .list_files()
        .into_iter()
        .find(|f| f.ends_with(".CMAP"))
        .expect("no CMAP")
        .to_string();
    let cmap = pkg.read_file(&cmap_name).unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let dat = std::fs::read("../Data/robots.dat").expect("robots.dat");
    let matrix_data = Storage::from_bytes(&dat).unwrap();

    let mut game = MapLogic::with_seed(1234);
    game.load_config(&matrix_data);

    let gs = GameMap::GLOBAL_SCALE_MOVE;
    let mut spot = None;
    'outer: for my in (20..map.size_move_y as i32 - 20).step_by(4) {
        for mx in (20..map.size_move_x as i32 - 20).step_by(4) {
            if matrixgame_rs::matrix_game::logic::is_absence_wall(&map, 2, 4, mx, my)
                && matrixgame_rs::matrix_game::logic::is_absence_wall(&map, 2, 4, mx + 14, my)
            {
                spot = Some((mx, my));
                break 'outer;
            }
        }
    }
    let (mx, my) = spot.expect("no passable spot");
    let (tx, ty) = (mx as f32 * gs, my as f32 * gs);
    let tz = map.get_z(tx, ty);

    // Laser turret (kind 3), player side.
    let mut c = Cannon::new(glam::Vec2::new(tx, ty), tz, 0.0, 1, 3, None, 0);
    let props = matrixgame_rs::matrix_game::config::global().turrets.cannons[2];
    println!(
        "laser props: weapon={} seek_radius={} max_da={:.4}",
        props.weapon, props.seek_radius, props.max_da
    );
    c.hit_point = 1000.0;
    c.hit_point_max = 1000.0;
    let cid = game.objects.spawn(Box::new(c));
    if let Some(obj) = game.objects.get_mut(cid) {
        let cm: &mut Cannon = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Cannon) };
        cm.self_id = Some(cid);
    }
    game.objects.add_lt(cid);

    // Enemy robot 14 cells away, idle.
    let (rx, ry) = ((mx + 14) as f32 * gs, my as f32 * gs);
    let rz = map.get_z(rx, ry);
    let mut r = Robot::new(glam::Vec3::new(rx, ry, rz), 2, ChassisKind::Track);
    r.state = RobotState::Idle;
    r.hit_point = 100_000.0;
    r.hit_point_max = 100_000.0;
    let rid = game.objects.spawn(Box::new(r));
    if let Some(rm) = robot_mut(&mut game.objects, rid) {
        rm.self_id = Some(rid);
    }
    game.objects.add_lt(rid);
    game.ensure_sides_from_objects();

    let dist = ((tx - rx).powi(2) + (ty - ry).powi(2)).sqrt();
    println!("turret at ({tx:.0},{ty:.0},{tz:.0}), robot at ({rx:.0},{ry:.0},{rz:.0}), dist={dist:.0}");

    let mut prev_fire = false;
    let mut prev_beam = false;
    let mut last_hp = 100_000.0f32;
    for step in 0..12000i64 {
        let _scope = MapScope::enter(&map, game.elapsed_ms);
        game.takt(10);

        let mut fire = false;
        let mut beam = None;
        let mut muzzle = glam::Vec3::ZERO;
        for w in game.objects.weapons.iter() {
            fire |= w.is_fire();
            if w.beam.is_some() {
                beam = w.beam;
            }
            muzzle = w.pos;
        }
        if fire != prev_fire || beam.is_some() != prev_beam {
            println!(
                "t={:>5}ms fire={fire} beam={beam:?} muzzle={muzzle:?}",
                game.elapsed_ms
            );
            prev_fire = fire;
            prev_beam = beam.is_some();
        }
        if step % 1000 == 0 {
            let hp = robot_ref(&game.objects, rid).map(|r| r.hit_point).unwrap_or(-1.0);
            println!(
                "t={:>5}ms robot_hp={hp:.0} (delta {:+.0}) fire={fire}",
                game.elapsed_ms,
                hp - last_hp
            );
            last_hp = hp;
        }
    }
}
