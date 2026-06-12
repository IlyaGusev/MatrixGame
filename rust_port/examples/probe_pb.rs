use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
fn main() {
    let data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();
    let b = pkg.read_file("Matrix\\Textures\\pb.dds").unwrap();
    let (h, w) = (
        u32::from_le_bytes(b[12..16].try_into().unwrap()),
        u32::from_le_bytes(b[16..20].try_into().unwrap()),
    );
    println!("pb.dds {}x{} ({} bytes)", w, h, b.len());
}
