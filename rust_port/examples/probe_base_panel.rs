//! Diagnostic: print each template button's `Param1` (type) and
//! `Param2` (kind) as stored in the Base panel config — the C++
//! `CConstructor::RemoteOperateUnit` reads the kind off Param2, so
//! chas{N} doesn't necessarily map to kind=N.

use matrixgame_rs::matrix_game::interface::CInterface;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let dat = std::fs::read("../Data/robots.dat").expect("read robots.dat");
    let stor = Storage::from_bytes(&dat).unwrap();
    let p = CInterface::load(&stor, "Base").expect("Base panel load");

    let prefixes = ["chas", "hull", "head", "weap"];
    for pre in &prefixes {
        println!("== {} ==", pre);
        for e in &p.elements {
            if !e.name.starts_with(pre) {
                continue;
            }
            let suf = &e.name[pre.len()..];
            // only numeric suffix or "e"
            if !(suf == "e" || suf.chars().all(|c| c.is_ascii_digit())) {
                continue;
            }
            println!(
                "  {:8} Param1={:3.0} Param2={:3.0} pos=({},{}) size=({},{})",
                e.name, e.param1, e.param2, e.pos_x, e.pos_y, e.size_x, e.size_y
            );
        }
    }
}
