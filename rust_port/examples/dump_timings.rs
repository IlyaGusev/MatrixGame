use matrixgame_rs::matrix_game::config::Timings;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();
    println!("{:?}", Timings::from_matrix_data(&stor));
}
