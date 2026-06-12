//! AABB of Basis+Turret+Shaft per cannon kind — feeds the mesh-derived
//! geo_center / radius (`JoinToGroup`, MatrixMapStatic.cpp:160-178).
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::three_g::vector_object::parse_vo;

fn main() {
    let data = std::fs::read("../Data/robots.pkg")
        .or_else(|_| std::fs::read("/home/yallen/MatrixGameDev/Data/robots.pkg"))
        .unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();
    let load = |p: &str| {
        let b = pkg.read_file(&format!("Matrix\\Cannon\\{p}.vo")).unwrap();
        parse_vo(&b).unwrap()
    };
    let basis = load("Basis");
    let bm = basis.matrix_by_id(20, 0).map(|m| [m[12], m[13], m[14]]).unwrap();
    for kind in 1..=4 {
        let turret = load(&format!("Turret{kind}"));
        let tm = turret.matrix_by_id(20, 0).map(|m| [m[12], m[13], m[14]]).unwrap();
        let shaft = load(&format!("Shaft{kind}"));
        let mut mn = [f32::MAX; 3];
        let mut mx = [f32::MIN; 3];
        let mut acc = |verts: &[_], off: [f32; 3]| {
            for v in verts {
                let p: [f32; 3] = vert_pos(v);
                for i in 0..3 {
                    mn[i] = mn[i].min(p[i] + off[i]);
                    mx[i] = mx[i].max(p[i] + off[i]);
                }
            }
        };
        acc(&basis.vertices, [0.0; 3]);
        acc(&turret.vertices, bm);
        acc(&shaft.vertices, [bm[0] + tm[0], bm[1] + tm[1], bm[2] + tm[2]]);
        let c = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
        let d = ((mx[0] - mn[0]).powi(2) + (mx[1] - mn[1]).powi(2) + (mx[2] - mn[2]).powi(2)).sqrt();
        println!(
            "kind {kind}: aabb ({:.1},{:.1},{:.1})..({:.1},{:.1},{:.1}) center=({:.2},{:.2},{:.2}) radius={:.2}",
            mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], c[0], c[1], c[2], d * 0.5
        );
    }
}

fn vert_pos(v: &matrixgame_rs::matrix_lib::three_g::vector_object::VoVertex) -> [f32; 3] {
    v.position
}
