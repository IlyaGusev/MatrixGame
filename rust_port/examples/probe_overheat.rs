//! Diagnostic: dump the Main panel's weapon-slot / overheat elements
//! ("wsl", "ovhe", "manbg", weapon-toggle dynamics).
use matrixgame_rs::matrix_game::interface::CInterface;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let dat = std::fs::read("../Data/robots.dat").expect("read robots.dat");
    let stor = Storage::from_bytes(&dat).unwrap();
    let p = CInterface::load(&stor, "Main").expect("Main panel load");
    for e in &p.elements {
        if ["wsl", "ovhe", "manbg", "inro", "lero"].contains(&e.name.as_str()) {
            let img0 = e.images.first().and_then(|x| x.as_ref());
            let img_info = img0
                .map(|i| format!("img({},{},{},{}) path={}", i.x, i.y, i.w, i.h, i.tex_path))
                .unwrap_or_else(|| "no img".to_string());
            println!(
                "{:8} kind={:?} pos=({}, {}, z={}) size=({}, {}) param1={} param2={} {}",
                e.name, e.kind, e.pos_x, e.pos_y, e.pos_z, e.size_x, e.size_y,
                e.param1, e.param2, img_info,
            );
        }
    }
}
