use matrixgame_rs::matrix_game::common::{CELLFLAG_BRIDGE, CELLFLAG_FLAT};
use matrixgame_rs::matrix_game::map::GameMap;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

const CELLFLAG_LAND: u8 = 1 << 0;
const CELLFLAG_WATER: u8 = 1 << 1;
const CELLFLAG_DOWN: u8 = 1 << 5;

fn main() {
    let data = std::fs::read("../Data/robots.pkg").expect("robots.pkg");
    let pkg = PkgArchive::from_bytes(data).unwrap();

    let cmap_name = pkg
        .list_files()
        .into_iter()
        .find(|f| f.ends_with(".CMAP"))
        .expect("no CMAP");
    println!("loading {}", cmap_name);
    let cmap = pkg.read_file(cmap_name).unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();

    println!("map {}x{}", map.size_x, map.size_y);
    println!("objects: {}", map.objects.len());

    use matrixgame_rs::matrix_lib::three_g::vector_object;
    let stor = Storage::from_bytes(&cmap).unwrap();
    let strings = stor.get_buf("strings", "String").unwrap();
    let seen: std::collections::BTreeSet<u32> = map.objects.iter().map(|o| o.type_id).collect();
    let mut parsed_ok = 0;
    let mut load_fail = 0;
    let mut parse_fail = 0;
    for t in &seen {
        if (*t as usize) >= strings.arrays_count() {
            continue;
        }
        let id = strings.get_as_wstr(*t as usize);
        let Some(paths) = matrixgame_rs::matrix_game::object::resolve_paths(&id) else {
            continue;
        };
        let key = paths.vo_path.to_uppercase();
        let Ok(data) = pkg.read_file(&key) else {
            println!("  MISS {} -> {}", t, paths.vo_path);
            load_fail += 1;
            continue;
        };
        match vector_object::parse_vo(&data) {
            Ok(m) => {
                let tri_count: usize = m.surfaces.iter().map(|s| s.indices.len() / 3).sum();
                let texture_refs: Vec<_> = m
                    .surfaces
                    .iter()
                    .filter_map(|s| s.texture_ref.as_deref())
                    .collect();
                println!(
                    "  OK   {:>3} -> {} ({} verts, {} tris, tex={:?})",
                    t,
                    paths.vo_path,
                    m.vertices.len(),
                    tri_count,
                    texture_refs
                );
                parsed_ok += 1;
            }
            Err(e) => {
                println!("  ERR  {} -> {}: {}", t, paths.vo_path, e);
                parse_fail += 1;
            }
        }
    }
    println!(
        "\nvo summary: {} parsed, {} not-found, {} parse-failed (of {} unique types)",
        parsed_ok,
        load_fail,
        parse_fail,
        seen.len()
    );

    // Find shore cells: CELLFLAG_LAND adjacent to CELLFLAG_WATER.
    let mut shore_count = 0;
    let mut down_count = 0;
    let mut land_count = 0;
    let mut water_count = 0;
    let mut min_z_shore = f32::INFINITY;
    let mut max_z_shore = f32::NEG_INFINITY;

    for y in 0..map.size_y {
        for x in 0..map.size_x {
            let u = map.unit(x, y);
            if u.flags & CELLFLAG_LAND != 0 {
                land_count += 1;
            }
            if u.flags & CELLFLAG_WATER != 0 {
                water_count += 1;
            }
            if u.flags & CELLFLAG_DOWN != 0 {
                down_count += 1;
            }

            if u.flags & CELLFLAG_LAND == 0 {
                continue;
            }
            // Check neighbors for water-only
            let has_water_neighbor =
                [(1i32, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|&(dx, dy)| {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= map.size_x as i32 || ny >= map.size_y as i32 {
                            return false;
                        }
                        let nu = map.unit(nx as usize, ny as usize);
                        nu.flags & CELLFLAG_LAND == 0 && nu.flags & CELLFLAG_WATER != 0
                    });
            if !has_water_neighbor {
                continue;
            }
            shore_count += 1;

            // Min z at this cell's corners
            let zs = [
                map.point(x, y).z,
                map.point(x + 1, y).z,
                map.point(x, y + 1).z,
                map.point(x + 1, y + 1).z,
            ];
            let min_z = zs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_z = zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            min_z_shore = min_z_shore.min(min_z);
            max_z_shore = max_z_shore.max(max_z);

            if shore_count <= 20 {
                println!(
                    "shore cell ({},{}) flags={:#04x} z corners = {:?} DOWN={}",
                    x,
                    y,
                    u.flags,
                    zs,
                    (u.flags & CELLFLAG_DOWN) != 0
                );
            }
        }
    }

    println!(
        "total cells: land={} water={} down={} shore={}",
        land_count, water_count, down_count, shore_count
    );
    println!("shore z range: {:.2} .. {:.2}", min_z_shore, max_z_shore);

    // Also scan all point z values
    let mut min_all = f32::INFINITY;
    let mut max_all = f32::NEG_INFINITY;
    for p in &map.points {
        min_all = min_all.min(p.z);
        max_all = max_all.max(p.z);
    }
    println!("all points z range: {:.2} .. {:.2}", min_all, max_all);

    // Check DOWN-flagged cells' z values
    let mut down_z_min = f32::INFINITY;
    let mut down_z_max = f32::NEG_INFINITY;
    for y in 0..map.size_y {
        for x in 0..map.size_x {
            let u = map.unit(x, y);
            if u.flags & CELLFLAG_DOWN == 0 {
                continue;
            }
            let z = map.point(x, y).z;
            down_z_min = down_z_min.min(z);
            down_z_max = down_z_max.max(z);
        }
    }
    println!("DOWN cells z range: {:.2} .. {:.2}", down_z_min, down_z_max);

    // Find cells at the actual water line (land cells whose z straddles -2).
    let water_level: f32 = -2.0;
    let mut waterline_cells = Vec::new();
    for y in 0..map.size_y {
        for x in 0..map.size_x {
            let u = map.unit(x, y);
            if u.flags & CELLFLAG_LAND == 0 {
                continue;
            }
            let zs = [
                map.point(x, y).z,
                map.point(x + 1, y).z,
                map.point(x, y + 1).z,
                map.point(x + 1, y + 1).z,
            ];
            let min_z = zs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_z = zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if min_z < water_level && max_z > water_level {
                waterline_cells.push((x, y, u.flags, min_z, max_z));
            }
        }
    }
    println!(
        "\ncells straddling water line (-2): {}",
        waterline_cells.len()
    );
    for &(x, y, f, mn, mx) in waterline_cells.iter().take(10) {
        println!(
            "  ({},{}) flags={:#04x} z {:.2}..{:.2} DOWN={}",
            x,
            y,
            f,
            mn,
            mx,
            f & CELLFLAG_DOWN != 0
        );
    }

    // Also: cells where z corners are entirely above water, adjacent to cells where
    // z corners are entirely below water (no smooth transition).
    let mut abrupt = 0;
    for y in 0..map.size_y {
        for x in 0..map.size_x {
            let u = map.unit(x, y);
            if u.flags & CELLFLAG_LAND == 0 {
                continue;
            }
            let min_z = [
                map.point(x, y).z,
                map.point(x + 1, y).z,
                map.point(x, y + 1).z,
                map.point(x + 1, y + 1).z,
            ]
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
            if min_z <= water_level {
                continue;
            }
            // This cell is entirely above water. Check neighbors.
            for (dx, dy) in [(1i32, 0), (-1, 0), (0, 1), (0, -1)].iter() {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= map.size_x as i32 || ny >= map.size_y as i32 {
                    continue;
                }
                let nu = map.unit(nx as usize, ny as usize);
                if nu.flags & CELLFLAG_WATER == 0 {
                    continue;
                }
                // Neighbor is water-only. This is an abrupt land→water-only edge above water.
                if abrupt < 10 {
                    println!(
                        "abrupt edge land({},{})z={:.2} -> water-only({},{})",
                        x, y, min_z, nx, ny
                    );
                }
                abrupt += 1;
            }
        }
    }
    println!("abrupt above-water land→water-only edges: {}", abrupt);

    let _ = (CELLFLAG_BRIDGE, CELLFLAG_FLAT);
}
