use matrixgame_rs::matrix_lib::three_g::texture;

fn main() {
    let cases = [
        r"D:\rangers\Matrix\Robot\Chassis3",
        r".\Matrix\Robot\Chassis1",
        r"Matrix/Robot/Armor1",
        r"D:\rangers\Matrix\Robot\Chassis3?Alpha",
    ];
    for raw in cases {
        let resolved = texture::resolve_texture_name(raw, Some("Matrix/Robot/"));
        println!("{:50} -> {:?}", raw, resolved);
    }
}
