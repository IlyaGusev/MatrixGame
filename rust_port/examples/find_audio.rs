use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
fn main() {
    for p in ["../Data/robots.pkg", "../Data/common.pkg", "../Data/mainmenu.pkg", "../Data/russian.pkg", "../Data/forms.pkg"] {
        let Ok(data) = std::fs::read(p) else { println!("{p}: unreadable"); continue };
        let Ok(pkg) = PkgArchive::from_bytes(data) else { println!("{p}: parse fail"); continue };
        let audio: Vec<_> = pkg.list_files().into_iter().filter(|f| {
            let l = f.to_lowercase();
            l.ends_with(".wav") || l.ends_with(".ogg") || l.ends_with(".mp3") || l.contains("sound")
        }).collect();
        println!("{p}: {} audio-ish files, e.g. {:?}", audio.len(), audio.iter().take(3).collect::<Vec<_>>());
    }
}
