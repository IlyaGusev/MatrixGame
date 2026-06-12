//! Port of `MatrixGame/src/Effects/`.
//! One Rust module per original `MatrixEffect*.cpp`. The gameplay
//! carriers (weapon, moving_object, fire_plasma, flame, big_boom) are
//! ported for mechanics; purely visual effects land with the
//! rendering work.

pub mod big_boom;
pub mod billboard_fx;
pub mod effects_renderer;
pub mod explosion;
pub mod fire_plasma;
pub mod flame;
pub mod konus;
pub mod landscape_spot;
pub mod lightening;
pub mod move_to;
pub mod moving_object;
pub mod point_light;
pub mod selection;
pub mod smoke_and_fire;
pub mod dust;
pub mod elevator_field;
pub mod weapon;
pub mod zahvat;

use crate::matrix_game::logic::Rnd;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::Objects;

/// The closed set of gameplay effects the map ticks each frame —
/// stands in for the `CMatrixEffect` intrusive list on `CMatrixMap`
/// (visual-only effects aren't routed through here).
pub enum GameEffect {
    MovingObject(moving_object::MovingObject),
    FirePlasma(fire_plasma::FirePlasma),
    Flame(flame::Flame),
    BigBoom(big_boom::BigBoom),
    Konus(konus::Konus),
    Smoke(smoke_and_fire::Smoke),
    Fire(smoke_and_fire::Fire),
    BillboardLine(billboard_fx::BillboardLineFx),
    Lightening(lightening::Lightening),
    Explosion(explosion::Explosion),
    FireAnim(explosion::FireAnim),
    Dust(dust::Dust),
}

impl GameEffect {
    /// Dispatch one `Takt`. Returns false at end of life.
    fn takt(&mut self, step: f32, map: &GameMap, objs: &mut Objects, rng: &mut Rnd) -> bool {
        match self {
            GameEffect::MovingObject(e) => e.takt(step, map, objs),
            GameEffect::FirePlasma(e) => e.takt(step, map, objs),
            GameEffect::Flame(e) => e.takt(step, map, objs),
            GameEffect::BigBoom(e) => e.takt(step, map, objs),
            GameEffect::Konus(e) => e.takt(step),
            GameEffect::Smoke(e) => e.takt(step, rng),
            GameEffect::Fire(e) => e.takt(step, rng),
            GameEffect::BillboardLine(e) => e.takt(step),
            GameEffect::Lightening(e) => e.takt(step, rng),
            GameEffect::Explosion(e) => e.takt(step, map, rng),
            GameEffect::FireAnim(e) => e.takt(step),
            GameEffect::Dust(e) => e.takt(step, map),
        }
    }

    /// The weapon ref a gameplay effect holds (spawned with
    /// `Weapons::add_ref`). Released by the driver on death — the
    /// Rust stand-in for `WeaponHit`'s `FEHF_LASTHIT → Release()`.
    fn weapon(&self) -> Option<weapon::WeaponId> {
        match self {
            GameEffect::MovingObject(e) => Some(e.weapon),
            GameEffect::FirePlasma(e) => Some(e.weapon),
            GameEffect::Flame(e) => Some(e.weapon),
            GameEffect::BigBoom(e) => Some(e.weapon),
            _ => None,
        }
    }

    /// Per-frame draw — fills the billboard + mesh queues.
    pub fn draw(
        &self,
        q: &mut crate::matrix_lib::three_g::billboard::BillboardQueue,
        meshes: &mut explosion::MeshQueue,
    ) {
        match self {
            GameEffect::MovingObject(e) => e.draw(meshes),
            GameEffect::FirePlasma(e) => e.draw(q),
            GameEffect::Flame(e) => e.draw(q),
            GameEffect::BigBoom(e) => e.draw(q),
            GameEffect::Konus(e) => e.draw(q),
            GameEffect::Smoke(e) => e.draw(q),
            GameEffect::Fire(e) => e.draw(q),
            GameEffect::BillboardLine(e) => e.draw(q),
            GameEffect::Lightening(e) => e.draw(q),
            GameEffect::Explosion(e) => e.draw(q, meshes),
            GameEffect::FireAnim(e) => e.draw(q),
            GameEffect::Dust(e) => e.draw(q),
        }
    }
}

/// Per-frame driver — the effect-list walk of `CMatrixMap::Takt`.
/// Effects spawned during the walk (queued on `objs.pending_effects`)
/// join the list afterwards and get their first takt next frame, same
/// as the C++ list insertion at the head.
pub fn effects_takt(
    effects: &mut Vec<GameEffect>,
    step: f32,
    map: &GameMap,
    objs: &mut Objects,
    rng: &mut Rnd,
) {
    // Deferred CreateExplosion calls (queued by damage paths that
    // carry no map/RNG): build the explosion + the optional FireAnim /
    // smoke pair + the voronka spot (MatrixEffect.cpp:452-501).
    let spawns: Vec<_> = objs.pending_explosions.drain(..).collect();
    for sp in spawns {
        let e = explosion::Explosion::new(sp.pos, sp.props, map, rng, &objs.debris_types);
        let (voronka, scale) = e.voronka();
        objs.pending_effects.push(GameEffect::Explosion(e));
        if sp.fire {
            let z = map.get_z(sp.pos.x, sp.pos.y);
            objs.pending_effects
                .push(GameEffect::FireAnim(explosion::FireAnim::new(
                    glam::Vec3::new(sp.pos.x, sp.pos.y, z),
                    35.0,
                    60.0,
                    2000.0,
                )));
            objs.pending_effects
                .push(GameEffect::Smoke(explosion::explosion_fire_smoke(sp.pos)));
        }
        if voronka != explosion::SpotType::None {
            objs.pending_spots.push(landscape_spot::SpotSpawn {
                pos: glam::Vec2::new(sp.pos.x, sp.pos.y),
                angle: (rng.float01() as f32 * 2.0 - 1.0) * std::f32::consts::PI,
                scale,
                kind: match voronka {
                    explosion::SpotType::PlasmaHit => landscape_spot::SpotKind::PlasmaHit,
                    _ => landscape_spot::SpotKind::Voronka,
                },
            });
        }
    }

    effects.append(&mut objs.pending_effects);
    let mut list = std::mem::take(effects);
    let mut survivors: Vec<GameEffect> = Vec::with_capacity(list.len());
    for mut e in list.drain(..) {
        if e.takt(step, map, objs, rng) {
            survivors.push(e);
        } else if let Some(w) = e.weapon() {
            objs.weapons.release(w);
        }
    }
    *effects = survivors;
    effects.append(&mut objs.pending_effects);
    enforce_limits(effects, objs);
}

/// Per-type count caps with delete-oldest semantics — the
/// `MAX_EFFECTS_*` + `DeleteFirst` machinery of CMatrixEffect
/// (MatrixEffect.hpp:186-190, AddEffect list bookkeeping).
fn enforce_limits(effects: &mut Vec<GameEffect>, objs: &mut Objects) {
    const MAX_EXPLOSIONS: usize = 50; // MAX_EFFECTS_EXPLOSIONS
    const MAX_FIREANIM: usize = 50; // MAX_EFFECTS_FIREANIM
    const MAX_SMOKEFIRES: usize = 100; // MAX_EFFECTS_SMOKEFIRES

    let mut explosions = 0usize;
    let mut fireanims = 0usize;
    let mut smokefires = 0usize;
    for e in effects.iter() {
        match e {
            GameEffect::Explosion(_) => explosions += 1,
            GameEffect::FireAnim(_) => fireanims += 1,
            GameEffect::Smoke(_) | GameEffect::Fire(_) => smokefires += 1,
            _ => {}
        }
    }
    let mut kill_explosions = explosions.saturating_sub(MAX_EXPLOSIONS);
    let mut kill_fireanims = fireanims.saturating_sub(MAX_FIREANIM);
    let mut kill_smokefires = smokefires.saturating_sub(MAX_SMOKEFIRES);
    if kill_explosions == 0 && kill_fireanims == 0 && kill_smokefires == 0 {
        return;
    }
    let list = std::mem::take(effects);
    for e in list {
        let drop_it = match &e {
            GameEffect::Explosion(_) if kill_explosions > 0 => {
                kill_explosions -= 1;
                true
            }
            GameEffect::FireAnim(_) if kill_fireanims > 0 => {
                kill_fireanims -= 1;
                true
            }
            GameEffect::Smoke(_) | GameEffect::Fire(_) if kill_smokefires > 0 => {
                kill_smokefires -= 1;
                true
            }
            _ => false,
        };
        if drop_it {
            if let Some(w) = e.weapon() {
                objs.weapons.release(w);
            }
        } else {
            effects.push(e);
        }
    }
}
