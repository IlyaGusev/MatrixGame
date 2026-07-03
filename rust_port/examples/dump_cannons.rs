use matrixgame_rs::matrix_game::logic::MapLogic;
use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let dat = std::fs::read("../Data/robots.dat").expect("robots.dat");
    let s = Storage::from_bytes(&dat).unwrap();
    let mut game = MapLogic::with_seed(1);
    game.load_config(&s);
    let cfg = matrixgame_rs::matrix_game::config::global();
    for (i, p) in cfg.turrets.cannons.iter().enumerate() {
        println!(
            "kind {}: weapon={} seek={} hp={} da={:.4}",
            i + 1,
            p.weapon,
            p.seek_radius,
            p.hitpoint,
            p.max_da
        );
    }
}
