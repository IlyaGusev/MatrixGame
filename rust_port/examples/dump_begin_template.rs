use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();
    if let Some(rec) = stor.block_record("da", "Templates") {
        for (i, val) in stor.block_params(&rec, "Begin").into_iter().enumerate() {
            println!("--- Templates/Begin[{i}] ---\n{val}\n");
        }
    }
}
