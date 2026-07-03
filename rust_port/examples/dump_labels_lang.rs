use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();
    if let Some(rec) = stor.block_record("da", "Replaces") {
        for key in ["continue", "exit", "helpb", "cancel"] {
            println!("Replaces/{key} = {:?}", stor.block_param(&rec, key));
        }
    }
}
