use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;

fn rd_u32(d: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    v
}
fn rd_i32(d: &[u8], o: &mut usize) -> i32 {
    let v = i32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    v
}
fn rd_f32(d: &[u8], o: &mut usize) -> f32 {
    let v = f32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    v
}
fn rd_u16(d: &[u8], o: &mut usize) -> u16 {
    let v = u16::from_le_bytes([d[*o], d[*o + 1]]);
    *o += 2;
    v
}

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();
    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();
    let strings = stor.get_buf("strings", "String").unwrap();
    let srfm = stor.get_buf("surfacesM", "Data").unwrap();

    let mut bad = 0usize;
    for i in 0..srfm.arrays_count() {
        let raw = srfm.get_bytes(i);
        if raw.len() < 32 {
            continue;
        }
        let mut off = 0;
        let ids = rd_i32(raw, &mut off);
        let index = rd_i32(raw, &mut off);
        let _color = rd_u32(raw, &mut off);
        let vcnt = rd_u32(raw, &mut off) as usize;
        let idxsz = rd_u32(raw, &mut off) as usize;
        let _grpsc = rd_u32(raw, &mut off) as usize;
        let _disp_x = rd_f32(raw, &mut off);
        let _disp_y = rd_f32(raw, &mut off);

        let tex_path = if ids >= 0 && (ids as usize) < strings.arrays_count() {
            strings.get_as_wstr(ids as usize)
        } else {
            "<bad-id>".to_string()
        };

        let needed = off + vcnt * 32 + idxsz;
        if needed > raw.len() {
            println!(
                "TRUNC surf {i}: {tex_path}, index={index}, vcnt={vcnt}, idxsz={idxsz}, raw={}",
                raw.len()
            );
            bad += 1;
            continue;
        }

        off += vcnt * 32;
        let idx_count = idxsz / 2;
        let mut max_idx = 0u16;
        for _ in 0..idx_count {
            let idx = rd_u16(raw, &mut off);
            if idx != 0xFFFF {
                max_idx = max_idx.max(idx);
            }
        }

        if max_idx as usize >= vcnt {
            println!("BAD surf {i}: {tex_path}, draw_index={index}, vcnt={vcnt}, max_idx={max_idx}, idxsz={idxsz}");
            bad += 1;
            if bad > 20 {
                break;
            }
        }
    }

    println!("bad surfaces: {bad}");
}
