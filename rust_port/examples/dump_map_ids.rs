use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(data).unwrap();
    let mut names: Vec<String> = pkg.list_files().into_iter().filter(|f| f.ends_with(".CMAP")).map(|s| s.to_string()).collect();
    names.sort();
    for cmap_name in names.iter().take(3) {
        println!("=== {cmap_name} ===");
        let cmap = pkg.read_file(cmap_name).unwrap();
        let stor = Storage::from_bytes(&cmap).unwrap();
        if let Some(buf) = stor.get_buf("strings", "String") {
            for i in 0..buf.arrays_count().min(20) {
                println!("  ids[{i}] = {:?}", buf.get_as_wstr(i));
            }
        } else {
            println!("  no strings buf");
        }
    }
}
