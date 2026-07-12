//! Headless probe for the rocket turret (cannon kind 4)
//! on the real map + real config: logs fire state, rocket flight and
//! shots over time.
//!
//!   cargo run --example rocket_turret_probe

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
    let mut c = Cannon::new(glam::Vec2::new(tx, ty), tz, 0.0, 1, 4, None, 0);
    let props = matrixgame_rs::matrix_game::config::global().turrets.cannons[3];
    println!(
        "rocket props: weapon={} seek_radius={} max_da={:.4}",
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

    // Base building right behind the turret (base-defense layout).
    {
        use matrixgame_rs::matrix_game::map::BuildingInstance;
        use matrixgame_rs::matrix_game::object_building::Building;
        let inst = BuildingInstance {
            x: (mx - 8) as f32 * gs,
            y: my as f32 * gs,
            build_z: tz,
            angle: 0,
            side: 1,
            kind: 0,
            turrets_places_cnt: 4,
            shadow_kind: 0,
            shadow_size: 128,
            turret_places: Vec::new(),
        };
        let b = Building::from_instance(&inst);
        let bid = game.objects.spawn(Box::new(b));
        if let Some(obj) = game.objects.get_mut(bid) {
            let bm: &mut Building =
                unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
            bm.self_id = Some(bid);
        }
        game.objects.add_lt(bid);
    }

    // Scan: spawn the enemy robot at varying offsets, run 4s, report fire.
    for (dx, dy) in [
        (4i32, 0i32), (6, 0), (8, 0), (10, 0), (14, 0), (20, 0), (26, 0),
        (32, 0), (40, 0), (48, 0), (56, 0), (14, 14), (20, 20), (30, 30),
        (40, 40), (8, 20), (0, 30), (0, 50),
    ] {
        let (rx, ry) = ((mx + dx) as f32 * gs, (my + dy) as f32 * gs);
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

        let mut fired = false;
        let hp0 = 100_000.0f32;
        for _ in 0..400i64 {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(10);
            for w in game.objects.weapons.iter() {
                fired |= w.is_fire();
            }
        }
        let hp = robot_ref(&game.objects, rid).map(|r| r.hit_point).unwrap_or(-1.0);
        let dist = (((dx * dx + dy * dy) as f32).sqrt()) * gs;
        println!(
            "offset=({dx:>2},{dy:>2}) dist={dist:>5.0} fired={fired} dmg={:.0}",
            hp0 - hp
        );
        // remove robot for next round
        if let Some(rm) = robot_mut(&mut game.objects, rid) {
            rm.hit_point = 0.0;
            rm.state = RobotState::Dip;
        }
        game.objects.remove(rid);
    }
}
