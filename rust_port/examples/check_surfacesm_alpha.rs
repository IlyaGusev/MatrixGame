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

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();
    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();
    let strings = stor.get_buf("strings", "String").unwrap();
    let srfm = stor.get_buf("surfacesM", "Data").unwrap();

    for i in 0..srfm.arrays_count() {
        let raw = srfm.get_bytes(i);
        if raw.len() < 12 {
            continue;
        }
        let mut off = 0;
        let ids = rd_i32(raw, &mut off);
        let index = rd_i32(raw, &mut off);
        let color_dw = rd_u32(raw, &mut off);
        let alpha = (color_dw >> 24) & 0xFF;
        let rgb = color_dw & 0x00FF_FFFF;
        if alpha != 0xFF || rgb != 0x00FF_FFFF {
            let tex_path = if ids >= 0 && (ids as usize) < strings.arrays_count() {
                strings.get_as_wstr(ids as usize)
            } else {
                "<bad-id>".to_string()
            };
            println!("argb={color_dw:08X} alpha={} index={} tex={}", alpha, index, tex_path);
        }
    }
}
