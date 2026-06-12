//! Count fire matrices (id 50..=59) per cannon shaft — feeds the
//! `FIRE_BARRELS` table in object_cannon.rs.
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::three_g::vector_object::parse_vo;

fn main() {
    let data = std::fs::read("../Data/robots.pkg")
        .or_else(|_| std::fs::read("/home/yallen/MatrixGameDev/Data/robots.pkg"))
        .expect("robots.pkg");
    let pkg = PkgArchive::from_bytes(data).expect("parse pkg");
    for kind in 1..=4 {
        let path = format!("Matrix\\Cannon\\Shaft{kind}.vo");
        match pkg.read_file(&path) {
            Ok(bytes) => match parse_vo(&bytes) {
                Ok(mesh) => {
                    let ids: Vec<u32> = mesh
                        .matrices
                        .iter()
                        .map(|m| m.id)
                        .filter(|id| (50..=59).contains(id))
                        .collect();
                    println!("{path}: fire matrices {ids:?}");
                    for m in &mesh.matrices {
                        if (50..=59).contains(&m.id) {
                            if let Some(mat) = mesh.matrix_by_id(m.id, 0) {
                                println!(
                                    "  id {} pos = ({:.2}, {:.2}, {:.2})",
                                    m.id, mat[12], mat[13], mat[14]
                                );
                            }
                        }
                    }
                }
                Err(e) => println!("{path}: vo parse failed {e:?}"),
            },
            Err(e) => println!("{path}: read failed {e:?}"),
        }
    }
}
