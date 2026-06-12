//! Dump the Billboards block from robots.dat (BBT texture table).
use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let bytes = std::fs::read("../Data/robots.dat")
        .or_else(|_| std::fs::read("/home/yallen/MatrixGameDev/Data/robots.dat"))
        .expect("robots.dat");
    let stor = Storage::from_bytes(&bytes).expect("parse");
    let bb = stor.block_record("da", "Billboards").expect("Billboards");
    if let Some(ts) = stor.block_param(&bb, "TexSort") {
        println!("TexSort = {ts}");
    }
    let tex = stor.block_record(&bb, "Textures").expect("Textures");
    if let (Some(k), Some(v)) = (stor.get_buf(&tex, "0"), stor.get_buf(&tex, "1")) {
        for i in 0..k.arrays_count() {
            println!("{} = {}", k.get_as_wstr(i), v.get_as_wstr(i));
        }
    }
}
