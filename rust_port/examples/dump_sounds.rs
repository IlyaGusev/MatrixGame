use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();
    if let Some(snds) = stor.block_record("da", "Sounds") {
        let recs = stor.block_records(&snds);
        println!("{} sound blocks", recs.len());
        for (name, rec) in recs.iter() {
            println!(
                "  {name}: path={:?} vol={:?} pan={:?} attn={:?} looped={:?} ttl={:?}",
                stor.block_param(rec, "path"),
                stor.block_param(rec, "vol"),
                stor.block_param(rec, "pan"),
                stor.block_param(rec, "attn"),
                stor.block_param(rec, "looped"),
                stor.block_param(rec, "ttl"),
            );
        }
    } else {
        println!("no Sounds block");
    }
    if let Some(chars) = stor.block_record("da", "Chars") {
        if let Some(cs) = stor.block_record(&chars, "ChassisSounds") {
            println!("ChassisSounds:");
            for (name, rec) in stor.block_records(&cs) {
                println!(
                    "  {name}: MoveTo={:?} Patrol={:?}",
                    stor.block_params(&rec, "MoveTo"),
                    stor.block_params(&rec, "Patrol"),
                );
            }
        } else {
            println!("no ChassisSounds block");
        }
    }
    let pkg = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg).unwrap();
    let wavs: Vec<_> = pkg
        .list_files()
        .into_iter()
        .filter(|f| f.to_lowercase().ends_with(".wav"))
        .collect();
    println!("{} wavs in pkg, first: {:?}", wavs.len(), wavs.first());
}
