//! Dump CStorage structure from a CMAP to understand texture references.
use matrixgame_rs::assets::pkg_reader::PkgArchive;
use matrixgame_rs::assets::storage::Storage;

fn main() {
    let pkg_data = std::fs::read("../Data/robots.pkg").unwrap();
    let pkg = PkgArchive::from_bytes(pkg_data).unwrap();

    let cmap_data = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let stor = Storage::from_bytes(&cmap_data).unwrap();

    // Print all records and items
    stor.dump_structure();

    // Print string table contents
    if let Some(strings) = stor.get_buf("strings", "String") {
        println!(
            "\n=== String table ({} entries) ===",
            strings.arrays_count()
        );
        for i in 0..strings.arrays_count().min(30) {
            println!("  [{}] {}", i, strings.get_as_wstr(i));
        }
        if strings.arrays_count() > 30 {
            println!("  ... {} more", strings.arrays_count() - 30);
        }
    }

    // Print properties
    if let (Some(names), Some(values)) = (
        stor.get_buf("properties", "Name"),
        stor.get_buf("properties", "Value"),
    ) {
        println!("\n=== Properties ({}) ===", names.arrays_count());
        for i in 0..names.arrays_count() {
            println!("  {} = {}", names.get_as_wstr(i), values.get_as_wstr(i));
        }
    }

    // Check surface-related tables
    for table in &["surfaces", "surfacesM", "texunions", "bottom"] {
        if let Some(buf) = stor.get_buf(table, "Data") {
            println!("\n=== {}/Data: {} arrays ===", table, buf.arrays_count());
            for i in 0..buf.arrays_count().min(3) {
                let bytes = buf.get_bytes(i);
                println!(
                    "  [{}] {} bytes, first 32: {:?}",
                    i,
                    bytes.len(),
                    &bytes[..bytes.len().min(32)]
                );
            }
        }
    }
}
