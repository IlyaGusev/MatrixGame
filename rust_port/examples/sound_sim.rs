//! Headless sound-truth probe: run the real atoll battle logic and
//! drain the sound queues through the mixer with a listener parked at
//! the player start base — print every voice that actually starts
//! (key, volume, distance) so mis-attenuated keys stand out.

use matrixgame_rs::matrix_game::config::SoundDefs;
use matrixgame_rs::matrix_game::logic::MapLogic;
use matrixgame_rs::matrix_game::map::{GameMap, MapScope};
use matrixgame_rs::matrix_game::sound::{Interrupt, SoundMixer, SoundOutput};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Default)]
struct LogState {
    next: u32,
    playing: HashMap<u32, (String, bool)>,
    vols: HashMap<u32, f32>,
    started: Vec<(String, f32)>, // (path, vol at start)
    live_looped: usize,
}

#[derive(Clone, Default)]
struct LogOutput(Rc<RefCell<LogState>>);

impl SoundOutput for LogOutput {
    fn ready(&self) -> bool {
        true
    }
    fn create(&mut self, path: &str, looped: bool) -> u32 {
        let mut st = self.0.borrow_mut();
        st.next += 1;
        let v = st.next;
        st.playing.insert(v, (path.to_string(), looped));
        v
    }
    fn play(&mut self, v: u32) {
        let mut st = self.0.borrow_mut();
        let vol = *st.vols.get(&v).unwrap_or(&0.0);
        if let Some((p, _)) = st.playing.get(&v) {
            let p = p.clone();
            st.started.push((p, vol));
        }
    }
    fn set_pan(&mut self, _v: u32, _p: f32) {}
    fn set_vol(&mut self, v: u32, vol: f32) {
        self.0.borrow_mut().vols.insert(v, vol);
    }
    fn is_playing(&self, v: u32) -> bool {
        self.0.borrow().playing.contains_key(&v)
    }
    fn destroy(&mut self, v: u32) {
        self.0.borrow_mut().playing.remove(&v);
    }
    fn set_music_volume(&mut self, _vol: f32) {}
}

fn main() {
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let dat = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();

    let mut game = MapLogic::with_seed(7);
    game.load_config(&dat);
    game.spawn_buildings(&map);
    game.spawn_ruins(&map);
    game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.init_effect_spawners(&map);
    game.accrue_resources(100_000);
    let stor = Storage::from_bytes(&cmap).unwrap();
    game.spawn_map_objects(&map, &stor);

    // Listener at the player base, camera-ish 260 units up.
    let base = game
        .objects
        .iter_live()
        .find_map(|id| {
            matrixgame_rs::matrix_game::logic::building_ref(&game.objects, id).and_then(|b| {
                (b.side == matrixgame_rs::matrix_game::common::PLAYER_SIDE
                    && matches!(b.kind, matrixgame_rs::matrix_game::object_building::BuildingType::Base))
                .then_some(glam::Vec3::new(b.pos.x, b.pos.y, 0.0))
            })
        })
        .unwrap_or(glam::Vec3::new(1000.0, 1000.0, 0.0));
    let focus = base + glam::Vec3::new(0.0, -150.0, 220.0);
    println!("listener at {focus:?}");

    let out = LogOutput::default();
    let mut mixer = SoundMixer::new(SoundDefs::from_matrix_data(&dat), Box::new(out.clone()));
    mixer.set_listener(focus, glam::Vec3::X);

    // 4 minutes of battle at 50ms takts.
    let mut agg: HashMap<String, (usize, f32)> = HashMap::new();
    for step in 0..(4 * 60 * 20) {
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
        }
        for id in std::mem::take(&mut game.objects.weapons.freed) {
            mixer.dispatch(matrixgame_rs::matrix_game::sound::SndEvent::Stop {
                handle: matrixgame_rs::matrix_game::sound::SndHandle::Weapon(id),
            });
        }
        for ev in game.objects.pending_sounds.drain(..) {
            mixer.dispatch(ev);
        }
        for gs in game.sound_queue.drain(..) {
            mixer.dispatch_game_sound(gs);
        }
        for (key, layer) in matrixgame_rs::matrix_game::interface::sound::drain() {
            mixer.play(&key, layer, Interrupt::Interrupt);
        }
        mixer.takt(game.elapsed_ms);
        for (path, vol) in out.0.borrow_mut().started.drain(..) {
            let e = agg.entry(path).or_insert((0, 0.0));
            e.0 += 1;
            e.1 = e.1.max(vol);
        }
        if step % 1200 == 1199 {
            let st = out.0.borrow();
            let looped_live = st.playing.values().filter(|(_, l)| *l).count();
            println!("t={}s: {} live voices ({} looped)", (step + 1) / 20, st.playing.len(), looped_live);
        }
    }
    let mut rows: Vec<_> = agg.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    println!("\nstarted voices over 4 min (count, max start vol):");
    for (path, (n, maxv)) in rows {
        println!("  {n:5}x maxvol={maxv:.3}  {path}");
    }
}
