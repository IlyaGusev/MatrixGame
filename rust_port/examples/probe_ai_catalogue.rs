fn main() {
    let dat = std::fs::read("../Data/robots.dat").unwrap();
    let stor = matrixgame_rs::matrix_lib::base::storage::Storage::from_bytes(&dat).unwrap();
    for name in ["AIRobotType", "AI", "Side", "Config"] {
        println!("{}: {:?}", name, stor.block_record("da", name).is_some());
    }
    let cat = matrixgame_rs::matrix_game::interface::constructor::AIRobotCatalogue::from_matrix_data(&stor);
    println!("catalogue bots: {}", cat.bots.len());
}
