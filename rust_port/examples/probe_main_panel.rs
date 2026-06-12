//! Diagnostic: dump the Main panel's elements, focusing on the
//! `gram`/`name`/`lives`/`mp1`/`mp2` ones we need for the robot
//! selection panel.

use matrixgame_rs::matrix_game::interface::CInterface;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let dat = std::fs::read("../Data/robots.dat").expect("read robots.dat");
    let stor = Storage::from_bytes(&dat).unwrap();
    let p = CInterface::load(&stor, "Main").expect("Main panel load");
    println!(
        "Main: design=({}, {}) elements={}",
        p.design_x,
        p.design_y,
        p.elements.len()
    );
    let watch = [
        "gram", "name", "lives", "mp1", "mp2", "prog", "ovhe", "inro", "lero",
    ];
    for e in &p.elements {
        if watch.contains(&e.name.as_str()) {
            let img0 = e.images.first().and_then(|x| x.as_ref());
            let img_info = img0
                .map(|i| {
                    format!(
                        "img({},{},{},{}) tex_w={},tex_h={} path={}",
                        i.x, i.y, i.w, i.h, i.tex_w, i.tex_h, i.tex_path
                    )
                })
                .unwrap_or_else(|| "no img".to_string());
            let labels: Vec<String> = e
                .labels
                .iter()
                .map(|l| format!("[{:?} \"{}\"]", l.state, l.text))
                .collect();
            println!(
                "  {:8} kind={:?} pos=({}, {}, z={}) size=({}, {}) {} labels={:?}",
                e.name, e.kind, e.pos_x, e.pos_y, e.pos_z, e.size_x, e.size_y, img_info, labels,
            );
        }
    }
}
