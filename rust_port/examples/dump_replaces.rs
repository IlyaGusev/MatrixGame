use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();
    for blk in ["Replaces"] {
        if let Some(rec) = stor.block_record("da", blk) {
            for key in ["_begin", "_win", "_loose", "_planet", "_race", "_difficulty"] {
                println!("da/{blk}/{key} = {:?}", stor.block_param(&rec, key));
            }
        }
    }
    // Labels/Replaces (IF_LABELS_BLOCKPAR)
    if let Some(labels) = stor.block_record("da", "Labels") {
        if let Some(rec) = stor.block_record(&labels, "Replaces") {
            for key in ["_begin", "_win", "_loose", "_planet", "_race"] {
                println!("da/Labels/Replaces/{key} = {:?}", stor.block_param(&rec, key));
            }
        } else { println!("no Labels/Replaces"); }
    }
}
