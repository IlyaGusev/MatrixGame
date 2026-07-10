//! Pack audio files into `assets/sounds.bundle` for the WebAudio
//! backend (SOUND_PROPOSAL.md §4). SR2 assets are proprietary, so the
//! payloads come from the user's SR2 installation:
//!
//!   cargo run --example pack_sounds -- <Sound.pkg | dir> [out.bundle]
//!
//! **Pkg mode** (arg ends in .pkg): reads SR2's `DATA/Sound.pkg` and
//! resolves each `Sound.X` resource id through the hand-derived
//! [`PKG_MAP`] table (the id→file tree lives in the host's encrypted
//! Main.dat; the SOUND/ROBOTS/* names map 1:1 so the table is short).
//!
//! **Folder mode**: for every `Sounds/<key>` entry in robots.dat with
//! a `path` (`Sound.WeapPlasma` style), the folder is probed for
//!   `<dir>/<key>.{wav,ogg,mp3}`          e.g. wplasma.ogg
//!   `<dir>/<path>.{wav,ogg,mp3}`         e.g. Sound.WeapPlasma.ogg
//!   `<dir>/<path with . as />.{wav,ogg,mp3}`  e.g. Sound/WeapPlasma.ogg
//! (case-insensitive on the stem).
//!
//! Matches are stored under the SR2 resource path — the key the
//! runtime sample cache uses. Missing entries are reported and stay
//! silent in-game (the spoken voice lines are NOT in Sound.pkg; they
//! ship with the SR2 host's speech resources).

use matrixgame_rs::gfx::bundle::AssetBundle;
use matrixgame_rs::matrix_game::config::SoundDefs;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use std::collections::HashMap;
use std::path::Path;

const EXTS: [&str; 3] = ["wav", "ogg", "mp3"];

/// `Sound.X` resource id → file inside SR2's DATA/Sound.pkg. Derived
/// by matching the id names against the SOUND/ROBOTS tree (the
/// authoritative mapping lives in the host's encrypted Main.dat).
/// Speech ids (order voices, base-captured lines, win/lose) have no
/// Sound.pkg file and are absent here.
const PKG_MAP: [(&str, &str); 88] = [
    ("Sound.WeapPlasma", "SOUND/ROBOTS/WEAPON/PLASMA.WAV"),
    ("Sound.WeapVolcano", "SOUND/ROBOTS/WEAPON/VOLCANO.WAV"),
    ("Sound.WeapMissile", "SOUND/ROBOTS/WEAPON/MISSILE.WAV"),
    ("Sound.WeapGun", "SOUND/ROBOTS/WEAPON/GUN.WAV"),
    ("Sound.WeapLaser", "SOUND/ROBOTS/WEAPON/LASER.WAV"),
    ("Sound.WeapLight", "SOUND/ROBOTS/WEAPON/LIGHTENING.WAV"),
    ("Sound.WeapRem", "SOUND/ROBOTS/WEAPON/REM.WAV"),
    ("Sound.Flame", "SOUND/ROBOTS/WEAPON/FIRE.WAV"),
    ("Sound.Gorit", "SOUND/ROBOTS/WEAPON/ABLAZE.WAV"),
    ("Sound.Zamknulo", "SOUND/ROBOTS/WEAPON/SHORTED.WAV"),
    ("Sound.Splash", "SOUND/ROBOTS/WORLD/SPLASH.WAV"),
    ("Sound.Rhit", "SOUND/ROBOTS/HIT/RHIT.WAV"),
    ("Sound.expl1", "SOUND/EXPL/EXPL1.WAV"),
    ("Sound.ExpMiss", "SOUND/ROBOTS/EXP/EXPMISS.WAV"),
    ("Sound.ExpRobot", "SOUND/ROBOTS/EXP/ROBOT.WAV"),
    ("Sound.ExpObj", "SOUND/ROBOTS/EXP/OBJECT.WAV"),
    ("Sound.ExpBuild", "SOUND/ROBOTS/EXP/BUILDING.WAV"),
    ("Sound.ExpBuildL", "SOUND/ROBOTS/EXP/BUILDING2.WAV"),
    ("Sound.ExpBuildS", "SOUND/ROBOTS/EXP/SMALLBUILD.WAV"),
    ("Sound.ExpBuildS2", "SOUND/ROBOTS/EXP/SMALLBUILD2.WAV"),
    ("Sound.BaseDoorOpen", "SOUND/ROBOTS/BASE/BASE_DOOR_OPEN.WAV"),
    ("Sound.BaseDoorClose", "SOUND/ROBOTS/BASE/BASE_DOOR_CLOSE.WAV"),
    ("Sound.BasePlatformUp", "SOUND/ROBOTS/BASE/BASE_PLATFORM_UP.WAV"),
    ("Sound.BasePlatformDown", "SOUND/ROBOTS/BASE/BASE_PLATFORM_DOWN.WAV"),
    ("Sound.BasePlatformUpStop", "SOUND/ROBOTS/BASE/BASE_PLATFORM_UP_STOP.WAV"),
    ("Sound.BaseAmbient", "SOUND/ROBOTS/BASE/BASE_AMB.WAV"),
    ("Sound.ChasPne", "SOUND/ROBOTS/CHASSIS/PNEUMATIC.WAV"),
    ("Sound.ChasWheel", "SOUND/ROBOTS/CHASSIS/WHEEL.WAV"),
    ("Sound.ChasTrack", "SOUND/ROBOTS/CHASSIS/TRACK.WAV"),
    ("Sound.ChasHover", "SOUND/ROBOTS/CHASSIS/HOVERCRAFT.WAV"),
    ("Sound.ChasAnti", "SOUND/ROBOTS/CHASSIS/ANTIGRAVITY.WAV"),
    ("Sound.HullUniv", "SOUND/ROBOTS/HULL/HULL.WAV"),
    ("Sound.Fall", "SOUND/ROBOTS/FALL.WAV"),
    ("Sound.Mute", "SOUND/ROBOTS/MUTE.WAV"),
    ("Sound.Nini", "SOUND/ROBOTS/NINI.WAV"),
    ("Sound.Resurs", "SOUND/ROBOTS/RESURS.WAV"),
    ("Sound.TPain1", "SOUND/ROBOTS/TERRON/PAIN1.WAV"),
    ("Sound.TPain2", "SOUND/ROBOTS/TERRON/PAIN2.WAV"),
    ("Sound.TPain3", "SOUND/ROBOTS/TERRON/PAIN3.WAV"),
    ("Sound.TerronDead", "SOUND/ROBOTS/TERRON/TERRONDEAD.WAV"),
    ("Sound.Stand", "SOUND/ROBOTS/TURELL/STAND.WAV"),
    ("Sound.VertStart", "SOUND/ROBOTS/VERTOLET/START.WAV"),
    ("Sound.VertLoop", "SOUND/ROBOTS/VERTOLET/LOOP.WAV"),
    ("Sound.DOpen", "SOUND/ROBOTS/MAP/DOOROPEN.WAV"),
    ("Sound.DClose", "SOUND/ROBOTS/MAP/DOORCLOSE.WAV"),
    ("Sound.ButtonClick", "SOUND/ROBOTS/CLICKS/CLICK.WAV"),
    ("Sound.Rclick1", "SOUND/ROBOTS/VOICES/ROBOTS/CLICK.WAV"),
    ("Sound.Rclick2", "SOUND/ROBOTS/VOICES/ROBOTS/CLICK2.WAV"),
    ("Sound.Plus", "SOUND/ROBOTS/CLICKS/PLUS.WAV"),
    ("Sound.Minus", "SOUND/ROBOTS/CLICKS/MINUS.WAV"),
    ("Sound.MapWater", "SOUND/ROBOTS/MAP/WATER.WAV"),
    // ── Speech (voicesRus.pkg — the localized announcer/robot voices) ──
    ("Sound.GetBase", "SOUND/ROBOTS/VOICES/BASE/BAZAPROTIVNIKA.WAV"),
    ("Sound.UnderAttack0", "SOUND/ROBOTS/VOICES/BASE/NANASNAPALI.WAV"),
    ("Sound.UnderAttack1", "SOUND/ROBOTS/VOICES/BASE/NASATAKUUT.WAV"),
    ("Sound.GetOurBase", "SOUND/ROBOTS/VOICES/BASE/NASHABAZA.WAV"),
    ("Sound.KillOurBase", "SOUND/ROBOTS/VOICES/BASE/NASHABAZAUNICH.WAV"),
    ("Sound.KillOurFactory", "SOUND/ROBOTS/VOICES/BASE/NASHZAVODUNICH.WAV"),
    ("Sound.GetOurFactory", "SOUND/ROBOTS/VOICES/BASE/NASHZAVODZAHV.WAV"),
    ("Sound.KillOurBuilding", "SOUND/ROBOTS/VOICES/BASE/VRAGUNICHTOJILNASHE.WAV"),
    ("Sound.GetFactory", "SOUND/ROBOTS/VOICES/BASE/ZAHVACHENZAVODPRO.WAV"),
    ("Sound.HelpApp", "SOUND/ROBOTS/VOICES/HELPAPPROACHING.WAV"),
    ("Sound.Help", "SOUND/ROBOTS/VOICES/MAINTAINCE.WAV"),
    ("Sound.RWin", "SOUND/ROBOTS/VOICES/WIN.WAV"),
    ("Sound.RLoose", "SOUND/ROBOTS/VOICES/LOOSE.WAV"),
    ("Sound.Armor", "SOUND/ROBOTS/VOICES/ROBOTS/ARMOR.WAV"),
    ("Sound.Attack", "SOUND/ROBOTS/VOICES/ROBOTS/ATTACK.WAV"),
    ("Sound.Capture", "SOUND/ROBOTS/VOICES/ROBOTS/CAPTURE.WAV"),
    ("Sound.ArmorProg", "SOUND/ROBOTS/VOICES/ROBOTS/PROGARMOR.WAV"),
    ("Sound.AttackProg", "SOUND/ROBOTS/VOICES/ROBOTS/PROGATTACK.WAV"),
    ("Sound.CaptureProg", "SOUND/ROBOTS/VOICES/ROBOTS/PROGCAPTURE.WAV"),
    ("Sound.Patrul", "SOUND/ROBOTS/VOICES/ROBOTS/GO.WAV"),
    ("Sound.RReady", "SOUND/ROBOTS/VOICES/ROBOTS/ROBOTDONE.WAV"),
    ("Sound.RReadyA", "SOUND/ROBOTS/VOICES/ROBOTS/ROBOTDONE1.WAV"),
    ("Sound.BReady", "SOUND/ROBOTS/VOICES/ROBOTS/ROBOTGOTOV.WAV"),
    ("Sound.GoPneumatic0", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/PNEUMATIC.WAV"),
    ("Sound.GoPneumatic1", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/PNEUMATIC2.WAV"),
    ("Sound.GoWheel0", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/WHEEL.WAV"),
    ("Sound.GoWheel1", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/WHEEL2.WAV"),
    ("Sound.GoTrack0", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/TRACK.WAV"),
    ("Sound.GoTrack1", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/TRACK2.WAV"),
    ("Sound.GoHovercraft0", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/HOVERCRAFT.WAV"),
    ("Sound.GoHovercraft1", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/HOVERCRAFT2.WAV"),
    ("Sound.GoAntigravi0", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/ANTIGRAVITY.WAV"),
    ("Sound.GoAntigravi1", "SOUND/ROBOTS/VOICES/ROBOTS/CHASSIS/ANTIGRAVITY2.WAV"),
    // E/H/L/R turret-built voices (t_build_0..3).
    ("Sound.ETurel", "SOUND/ROBOTS/VOICES/TURELS/EASYWEAPON.WAV"),
    ("Sound.HTurel", "SOUND/ROBOTS/VOICES/TURELS/HARDWEAPON.WAV"),
    ("Sound.LTurel", "SOUND/ROBOTS/VOICES/TURELS/LAZERWEAPON.WAV"),
    ("Sound.RTurel", "SOUND/ROBOTS/VOICES/TURELS/ROCKETWEAPON.WAV"),
];

/// Ambience ids beyond the fixed table — AMB_LOW1..4 for MapAmb1..4,
/// FANTASY/BOTTLE for the rest (best-guess; all are quiet loops).
fn pkg_map_extra(path: &str) -> Option<&'static str> {
    Some(match path {
        "Sound.MapAmb1" => "SOUND/ROBOTS/MAP/AMB_LOW1.WAV",
        "Sound.MapAmb2" => "SOUND/ROBOTS/MAP/AMB_LOW2.WAV",
        "Sound.MapAmb3" => "SOUND/ROBOTS/MAP/AMB_LOW3.WAV",
        "Sound.MapAmb4" => "SOUND/ROBOTS/MAP/AMB_LOW4.WAV",
        "Sound.MapAmb5" => "SOUND/ROBOTS/MAP/FANTASY1.WAV",
        "Sound.MapAmb6" => "SOUND/ROBOTS/MAP/FANTASY2.WAV",
        "Sound.MapAmb7" => "SOUND/ROBOTS/MAP/BOTTLE.WAV",
        "Sound.HangarOpen" => "SOUND/HANGAROPEN.WAV",
        "Sound.FormShipOpen" => "SOUND/FORMSHIPOPEN.WAV",
        "Sound.FormShipClose" => "SOUND/FORMSHIPCLOSE.WAV",
        "Sound.FormPlanetNone1" => "SOUND/PLANET/NONE/1.WAV",
        "Sound.FormPlanetNone3" => "SOUND/PLANET/NONE/3.WAV",
        _ => return None,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "usage: cargo run --example pack_sounds -- <Sound.pkg ...|dir> [out.bundle]"
        );
        std::process::exit(2);
    }
    let (pkg_paths, rest): (Vec<&String>, Vec<&String>) =
        args.iter().partition(|a| a.to_lowercase().ends_with(".pkg"));
    let dir = rest.first().filter(|a| !a.ends_with(".bundle")).cloned();
    let out = args
        .iter()
        .find(|a| a.ends_with(".bundle"))
        .cloned()
        .unwrap_or_else(|| "assets/sounds.bundle".into());

    let dat = std::fs::read("../Data/robots.dat").expect("../Data/robots.dat");
    let stor = Storage::from_bytes(&dat).expect("parse robots.dat");
    let defs = SoundDefs::from_matrix_data(&stor);

    let pkgs: Vec<PkgArchive> = pkg_paths
        .iter()
        .map(|p| {
            let data = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
            PkgArchive::from_bytes(data).unwrap_or_else(|e| panic!("parse {p}: {e}"))
        })
        .collect();
    for (p, pkg) in pkg_paths.iter().zip(&pkgs) {
        println!("{p}: {} files", pkg.list_files().len());
    }

    // Folder mode: index the folder once (lowercased relative stem → path).
    let mut files: HashMap<String, std::path::PathBuf> = HashMap::new();
    if let Some(dir) = &dir {
        index_dir(Path::new(dir), Path::new(dir), &mut files);
        println!("{} audio files under {dir}", files.len());
    }

    // (key, SR2 resource path) worklist: every Sounds-block entry plus
    // the map ambience the EST_SOUND spawners play by name
    // (Sound.MapWater etc. — resolved by the host in the original,
    // absent from the Sounds block).
    let mut wanted: Vec<(String, String)> = defs
        .iter()
        .filter(|(_, d)| !d.path.is_empty())
        .map(|(k, d)| (k.clone(), d.path.clone()))
        .collect();
    wanted.sort();
    // Ambient names go last so paths shared with Sounds-block keys
    // resolve through the key probe first.
    for name in ambient_spawner_names() {
        wanted.push((name.clone(), name));
    }

    let mut bundle = AssetBundle::new();
    let mut packed: HashMap<String, String> = HashMap::new(); // path → source file
    let mut missing: Vec<String> = Vec::new();
    for (key, path) in wanted {
        if packed.contains_key(&path) {
            continue;
        }
        if !pkgs.is_empty() {
            let file = PKG_MAP
                .iter()
                .find(|(id, _)| *id == path)
                .map(|(_, f)| *f)
                .or_else(|| pkg_map_extra(&path));
            match file.and_then(|f| pkgs.iter().find_map(|pkg| pkg.read_file(f).ok())) {
                Some(bytes) => {
                    bundle.add(&path, bytes);
                    packed.insert(path.clone(), file.unwrap().to_string());
                }
                None => missing.push(format!("{key} ({path})")),
            }
            continue;
        }
        let candidates = [
            key.to_lowercase(),
            path.to_lowercase(),
            path.to_lowercase().replace('.', "/"),
        ];
        let hit = candidates.iter().find_map(|c| files.get(c));
        match hit {
            Some(f) => {
                let bytes = std::fs::read(f).expect("read audio file");
                bundle.add(&path, bytes);
                packed.insert(path.clone(), f.display().to_string());
            }
            None => missing.push(format!("{key} ({path})")),
        }
    }

    missing.sort();
    for m in &missing {
        println!("  missing: {m}");
    }
    println!(
        "packed {} samples, {} sound keys unresolved (they stay silent)",
        packed.len(),
        missing.len()
    );
    let bytes = bundle.to_zstd_bytes(19);
    std::fs::write(&out, &bytes).expect("write bundle");
    println!("wrote {out} ({} KiB)", bytes.len() / 1024);
}

/// Distinct sound names authored on EST_SOUND effect spawners across
/// all maps in robots.pkg (`3,min,max,vol0,vol1,pan0,pan1,attn,name`).
fn ambient_spawner_names() -> Vec<String> {
    let mut out = std::collections::BTreeSet::new();
    let Ok(data) = std::fs::read("../Data/robots.pkg") else {
        return Vec::new();
    };
    let Ok(pkg) = PkgArchive::from_bytes(data) else {
        return Vec::new();
    };
    for f in pkg.list_files() {
        if !f.ends_with(".CMAP") {
            continue;
        }
        let Ok(cmap) = pkg.read_file(&f) else { continue };
        let Ok(map) = matrixgame_rs::matrix_game::map::GameMap::from_cmap_bytes(&cmap) else {
            continue;
        };
        for (_, spec) in &map.effect_spawners {
            let parts: Vec<&str> = spec.split(',').map(|s| s.trim()).collect();
            if parts.first() == Some(&"3") {
                if let Some(name) = parts.get(8).filter(|n| !n.is_empty()) {
                    out.insert(name.to_string());
                }
            }
        }
    }
    println!("{} ambient spawner sounds across maps: {:?}", out.len(), out);
    out.into_iter().collect()
}

/// Recursive scan; keys are lowercased extension-less paths relative
/// to the root (`sound/weapplasma`), plus the bare stem (`weapplasma`)
/// so flat probes hit files in subfolders too.
fn index_dir(root: &Path, dir: &Path, out: &mut HashMap<String, std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            index_dir(root, &p, out);
            continue;
        }
        let Some(ext) = p.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        if !EXTS.contains(&ext.to_lowercase().as_str()) {
            continue;
        }
        if let Ok(rel) = p.strip_prefix(root) {
            let rel_stem = rel
                .with_extension("")
                .to_string_lossy()
                .replace('\\', "/")
                .to_lowercase();
            out.entry(rel_stem.clone()).or_insert_with(|| p.clone());
            if let Some(stem) = Path::new(&rel_stem).file_name().and_then(|s| s.to_str()) {
                out.entry(stem.to_string()).or_insert_with(|| p.clone());
            }
        }
    }
}
