use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();
    if let Some(snds) = stor.block_record("da", "Sounds") {
        let recs = stor.block_records(&snds);
        println!("{} sound blocks", recs.len());
        for (name, rec) in recs.iter().take(6) {
            println!("  {name}: path={:?} vol={:?}", stor.block_param(rec, "path"), stor.block_param(rec, "vol"));
        }
        for (name, rec) in recs.iter() {
            if name == "bclick" || name == "s_maintenance" || name == "border_attack" {
                println!("  {name}: path={:?}", stor.block_param(rec, "path"));
            }
        }
    } else { println!("no Sounds block"); }
    let pkg = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg).unwrap();
    let wavs: Vec<_> = pkg.list_files().into_iter().filter(|f| f.to_lowercase().ends_with(".wav")).collect();
    println!("{} wavs in pkg, first: {:?}", wavs.len(), wavs.first());
}
