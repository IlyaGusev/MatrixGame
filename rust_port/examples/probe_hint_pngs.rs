//! Verify `Matrix/Textures/Hints/*.png` are resolvable in one of the
//! shipped pkgs. The hint renderer fails silently when the atlas
//! isn't found, so we need to prove the path/casing works.
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;

fn main() {
    let targets = [
        "Matrix/Textures/Hints/border0.png",
        "Matrix\\Textures\\Hints\\icon_titan.png",
        "Matrix/Textures/Hints/icon_energy.png",
        "MATRIX/TEXTURES/HINTS/BORDER0.PNG",
        "MATRIX/TEXTURES/HINTS/ICON_TITAN.PNG",
    ];
    for pkg_name in ["robots.pkg", "common.pkg", "forms.pkg", "mainmenu.pkg", "russian.pkg"] {
        let Ok(data) = std::fs::read(format!("../Data/{pkg_name}")) else {
            println!("skip {pkg_name}");
            continue;
        };
        let pkg = PkgArchive::from_bytes(data).expect("pkg parse");
        println!("== {pkg_name} ==");
        for t in &targets {
            match pkg.read_file(t) {
                Ok(b) => println!("  OK  {t} -> {} bytes", b.len()),
                Err(_) => {}
            }
        }
    }
}
