//! Fix-2 probe: dissect the ATOLL zone path that wedges the seed-3
//! robots (pos (161,264) → des (258,260), pneumatic chassis 0).
//! Prints the zone path, then for each consecutive link: the
//! near_zone_move mask and whether a size-4 cell-level path exists
//! between the two zone centers.

use matrixgame_rs::matrix_game::logic::{find_local_path, zone_find_near, zone_move_rect};
use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;

fn main() {
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();

    let nsh = 0usize; // pneumatic
    let (sx, sy) = (161, 264);
    let (dx, dy) = (258, 260);
    let zstart = zone_find_near(&map, nsh as i32, sx, sy);
    let zend = zone_find_near(&map, nsh as i32, dx, dy);
    println!("zstart={zstart} zend={zend}");

    // (path, per-zone (center, move_mask, near ids+masks)) — collected
    // under the lock, released before find_local_path re-locks it.
    let (path, zinfo) = {
        let rn_lock = map.road_network.as_ref().unwrap();
        let mut rn = rn_lock.lock().unwrap();
        let mut path = vec![0i32; rn.zones.len()];
        let cnt = rn.find_path_in_zone(nsh, zstart, zend, None, &mut path);
        path.truncate(cnt);
        let zinfo: Vec<_> = rn
            .zones
            .iter()
            .map(|z| {
                (
                    z.center,
                    z.move_mask,
                    z.near_zone.clone(),
                    z.near_zone_move.clone(),
                    z.near_zone_connect_size.clone(),
                )
            })
            .collect();
        (path, zinfo)
    };
    println!("zone path ({}): {path:?}", path.len());

    for w in path.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (ca, mv_a, near, near_mv, near_cs) = &zinfo[a as usize];
        let (cb, mv_b, _, _, _) = &zinfo[b as usize];
        let link = near.iter().position(|&z| z == b);
        let (mask, csize) = match link {
            Some(i) => (near_mv[i], near_cs[i]),
            None => {
                println!("  {a}->{b}: NOT A NEIGHBOR?!");
                continue;
            }
        };
        let (ca, cb) = (*ca, *cb);
        let (mv_a, mv_b) = (*mv_a, *mv_b);
        let window = [a, b];
        let res = find_local_path(
            &map,
            nsh,
            4,
            ca.x,
            ca.y,
            &window,
            &|z| zone_move_rect(&map, z),
            cb.x,
            cb.y,
            &[],
        );
        println!(
            "  {a}({},{}) -> {b}({},{}): near_move={mask:#04x} connect_size={csize} zmask=({mv_a:#04x},{mv_b:#04x}) size4_path={}pts{}",
            ca.x, ca.y, cb.x, cb.y,
            res.path.len(),
            if res.path.is_empty() { "  <-- IMPASSABLE" } else { "" },
        );
    }
}
