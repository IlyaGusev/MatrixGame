use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    let data = std::fs::read("../Data/robots.dat").unwrap();
    let stor = Storage::from_bytes(&data).unwrap();

    let Some(cam_rec) = stor.block_record("da", "Camera") else {
        println!("no Camera block in robots.dat");
        return;
    };

    println!("=== Camera (top-level) ===");
    for key in [
        "CamBaseAngleZ",
        "CamMoveSpeed",
        "CamInRobotForward0",
        "CamInRobotForward1",
    ] {
        println!("  {} = {:?}", key, stor.block_param(&cam_rec, key).as_deref());
    }

    for sub in ["Strategy", "InRobot"] {
        println!("=== Camera/{} ===", sub);
        let Some(rec) = stor.block_record(&cam_rec, sub) else {
            println!("  (missing)");
            continue;
        };
        for key in [
            "CamRotSpeedX",
            "CamRotSpeedZ",
            "CamMouseWheelStep",
            "CamRotAngleMin",
            "CamRotAngleMax",
            "CamDistMin",
            "CamDistMax",
            "CamAngleParam",
            "CamHeight",
        ] {
            println!("  {} = {:?}", key, stor.block_param(&rec, key).as_deref());
        }
    }
}
