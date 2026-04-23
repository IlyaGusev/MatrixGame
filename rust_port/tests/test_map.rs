use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;

#[test]
fn load_map_from_pkg() {
    let _ = env_logger::builder().is_test(true).try_init();

    let pkg_data = std::fs::read("../Data/robots.pkg").expect("read robots.pkg");
    let pkg = PkgArchive::from_bytes(pkg_data).expect("parse pkg");

    // Pick a small map
    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").expect("read CMAP");
    println!("CMAP size: {} bytes", cmap_data.len());

    let map = GameMap::from_cmap_bytes(&cmap_data).expect("parse map");
    println!("Map: {}x{} units", map.size_x, map.size_y);
    println!(
        "World size: {:.0}x{:.0}",
        map.world_width(),
        map.world_height()
    );

    // Sample some points
    let mid_x = map.size_x / 2;
    let mid_y = map.size_y / 2;
    let center = map.point(mid_x, mid_y);
    println!(
        "Center point ({mid_x},{mid_y}): z={:.2}, rgb=({},{},{}), flags=0x{:02x}",
        center.z, center.r, center.g, center.b, center.flags
    );

    assert!(map.size_x > 0);
    assert!(map.size_y > 0);
    assert_eq!(map.points.len(), (map.size_x + 1) * (map.size_y + 1));
}
