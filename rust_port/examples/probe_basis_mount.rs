use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::three_g::vector_object::parse_vo;
fn main() {
    let data = std::fs::read("../Data/robots.pkg")
        .or_else(|_| std::fs::read("/home/yallen/MatrixGameDev/Data/robots.pkg"))
        .unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();
    for p in ["Basis", "Turret1", "Turret2", "Turret3", "Turret4"] {
        let b = pkg.read_file(&format!("Matrix\\Cannon\\{p}.vo")).unwrap();
        let mesh = parse_vo(&b).unwrap();
        match mesh.matrix_by_id(20, 0) {
            Some(m) => println!("{p}: mount = ({:.3}, {:.3}, {:.3})", m[12], m[13], m[14]),
            None => println!("{p}: no matrix 20"),
        }
    }
}
