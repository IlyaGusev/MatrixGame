use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();
    if let Some(rec) = stor.block_record("da", "Templates") {
        for val in stor.block_params(&rec, "Menu").into_iter().take(3) {
            println!("{val}");
        }
        for val in stor.block_params(&rec, "Win").into_iter().take(2) {
            println!("WIN: {val}");
        }
    }
}
