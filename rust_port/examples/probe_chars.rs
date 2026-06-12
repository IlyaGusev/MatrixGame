//! Verify ItemChars.{chassis,armor}_structure + head_hp_add populate
//! correctly and check Turret HP loading.
use matrixgame_rs::matrix_game::config::{ItemCharsTable, TurretProps};
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let bytes = std::fs::read("../Data/robots.dat").expect("robots.dat");
    let stor = Storage::from_bytes(&bytes).expect("storage parse");

    let chars = ItemCharsTable::from_matrix_data(&stor).expect("ItemChars");
    println!("chassis_structure = {:?}", chars.chassis_structure);
    println!("armor_structure = {:?}", chars.armor_structure);
    println!("head_hp_add = {:?}", chars.head_hp_add);

    let turrets = TurretProps::from_matrix_data(&stor).expect("TurretProps");
    for (i, c) in turrets.cannons.iter().enumerate() {
        println!(
            "Turret {} -> hitpoint={} strength={} res={:?}",
            i + 1,
            c.hitpoint,
            c.strength,
            c.resources
        );
    }

    // Expected (from MatrixConfig.cpp:4160-4192 + shipped robots.dat):
    // a full pnuematic+passive+blocker robot = ~2000-2500 HP, scaled /10 in UI = ~200-250.

    // Raw Chars/Chassis / Chars/Armor dumps.
    let chars_rec = stor.block_record("da", "Chars").unwrap();
    for sub in ["Chassis", "Armor"] {
        let rec = stor.block_record(&chars_rec, sub).unwrap();
        let k = stor.get_buf(&rec, "0").unwrap();
        let v = stor.get_buf(&rec, "1").unwrap();
        println!("\n== Chars/{sub}  ({} entries) ==", k.arrays_count());
        for i in 0..k.arrays_count() {
            let name = k.get_as_wstr(i);
            if name.ends_with("_STRUCTURE") || name.ends_with("_ROTATION_SPEED") {
                println!("  {name} = {}", v.get_as_wstr(i));
            }
        }
    }
}
