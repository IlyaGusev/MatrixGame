use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let s = Storage::from_bytes(&data).unwrap();
    let models = s.block_record("da", "Models").unwrap();
    let deb = s.block_record(&models, "Debris").unwrap();
    let k = s.get_buf(&deb, "0").unwrap();
    let v = s.get_buf(&deb, "1").unwrap();
    for i in 0..v.arrays_count() {
        println!("{:?} -> {}", k.get_as_wstr(i), v.get_as_wstr(i));
    }
}
