//! Pack audio files into `assets/sounds.bundle` for the WebAudio
//! backend (SOUND_PROPOSAL.md §4). SR2 assets are proprietary, so the
//! payloads come from a user-supplied folder — the "drop WAVs in a
//! folder" escape hatch:
//!
//!   cargo run --example pack_sounds -- <dir> [out.bundle]
//!
//! For every `Sounds/<key>` entry in robots.dat with a `path`
//! (`Sound.WeapPlasma` style), the folder is probed for
//!   `<dir>/<key>.{wav,ogg,mp3}`          e.g. wplasma.ogg
//!   `<dir>/<path>.{wav,ogg,mp3}`         e.g. Sound.WeapPlasma.ogg
//!   `<dir>/<path with . as />.{wav,ogg,mp3}`  e.g. Sound/WeapPlasma.ogg
//! (case-insensitive on the stem). Matches are stored under the SR2
//! resource path — the key the runtime sample cache uses. Missing
//! entries are reported and stay silent in-game.

use matrixgame_rs::gfx::bundle::AssetBundle;
use matrixgame_rs::matrix_game::config::SoundDefs;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use std::collections::HashMap;
use std::path::Path;

const EXTS: [&str; 3] = ["wav", "ogg", "mp3"];

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: cargo run --example pack_sounds -- <dir-with-audio-files> [out.bundle]");
        std::process::exit(2);
    };
    let out = args.next().unwrap_or_else(|| "assets/sounds.bundle".into());

    let dat = std::fs::read("../Data/robots.dat").expect("../Data/robots.dat");
    let stor = Storage::from_bytes(&dat).expect("parse robots.dat");
    let defs = SoundDefs::from_matrix_data(&stor);

    // Index the folder once: lowercased relative stem → path.
    let mut files: HashMap<String, std::path::PathBuf> = HashMap::new();
    index_dir(Path::new(&dir), Path::new(&dir), &mut files);
    println!("{} audio files under {dir}", files.len());

    let mut bundle = AssetBundle::new();
    let mut packed: HashMap<String, String> = HashMap::new(); // path → source file
    let mut missing: Vec<String> = Vec::new();
    let mut keys: Vec<_> = defs.iter().collect();
    keys.sort_by(|a, b| a.0.cmp(b.0));
    for (key, def) in keys {
        if def.path.is_empty() {
            continue;
        }
        if packed.contains_key(&def.path) {
            continue;
        }
        let candidates = [
            key.to_lowercase(),
            def.path.to_lowercase(),
            def.path.to_lowercase().replace('.', "/"),
        ];
        let hit = candidates.iter().find_map(|c| files.get(c));
        match hit {
            Some(f) => {
                let bytes = std::fs::read(f).expect("read audio file");
                bundle.add(&def.path, bytes);
                packed.insert(def.path.clone(), f.display().to_string());
            }
            None => missing.push(format!("{key} ({})", def.path)),
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
