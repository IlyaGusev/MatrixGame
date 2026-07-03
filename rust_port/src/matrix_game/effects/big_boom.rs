//! Port of `MatrixEffectBigBoom.cpp` — the expanding blast sphere of
//! the robot self-destruct bomb: the end-of-TTL damage sweep plus the
//! shrinking additive geosphere shell with its rotating 3D texture
//! transform. The per-frame robot knock-tilt (`BoomEnumNaklon` →
//! `ApplyNaklon`) needs the robot tilt system and is not ported.

use crate::matrix_game::common::{TRACE_ANYOBJECT, TRACE_LANDSCAPE, TRACE_WATER, WATER_LEVEL};
use crate::matrix_game::effects::weapon::{weapon_hit, WeaponId};
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{Control, ObjectId, Objects};
use crate::matrix_game::map_trace::TraceStop;
use glam::Vec3;

pub struct BigBoom {
    center: Vec3,
    radius: f32,
    ttl: f32,
    ttl0: f32,
    hitmask: u32,
    skip: Option<ObjectId>,
    pub weapon: WeaponId,
    /// `m_k1..3` — FSRND(3) texture-spin rates. Seeded from the center
    /// coords (visual-only randomness; keeps the world RNG stream
    /// untouched).
    k_rot: [f32; 3],
    /// Optional shell color override (the two visual-only booms pass
    /// 0xFFAFAF40); default white.
    pub color_rgb: u32,
    /// false for the pure-visual companion booms (no handler sweep).
    pub deals_damage: bool,
}

impl BigBoom {
    pub fn new(
        center: Vec3,
        radius: f32,
        ttl: f32,
        hitmask: u32,
        skip: Option<ObjectId>,
        weapon: WeaponId,
    ) -> Self {
        let h = |k: f32| ((center.x * 0.717 + center.y * k).fract() * 2.0 - 1.0) * 3.0;
        Self {
            center,
            radius,
            ttl,
            ttl0: ttl,
            hitmask,
            skip,
            weapon,
            k_rot: [h(0.31), h(0.53), h(0.77)],
            color_rgb: 0x00FF_FFFF,
            deals_damage: true,
        }
    }

    /// `CMatrixEffectBigBoom::Takt` (MatrixEffectBigBoom.cpp:294-376).
    /// Returns false when the blast delivered its hits and died.
    pub fn takt(&mut self, step: f32, map: &GameMap, objs: &mut Objects) -> bool {
        self.ttl -= step;
        if self.ttl >= 0.0 {
            return true;
        }
        if !self.deals_damage {
            return false;
        }

        if self.hitmask & TRACE_WATER != 0 && (self.center.z - self.radius) <= WATER_LEVEL {
            weapon_hit(
                objs,
                self.weapon,
                TraceStop::Water,
                Vec3::new(self.center.x, self.center.y, WATER_LEVEL),
                0,
                0.0,
            );
        }
        if self.hitmask & TRACE_LANDSCAPE != 0 {
            let z = map.get_z(self.center.x, self.center.y);
            if (self.center.z - self.radius) <= z {
                weapon_hit(
                    objs,
                    self.weapon,
                    TraceStop::Landscape,
                    Vec3::new(self.center.x, self.center.y, z),
                    0,
                    0.0,
                );
            }
        }
        if self.hitmask & TRACE_ANYOBJECT != 0 {
            // BoomEnum (MatrixEffectBigBoom.cpp:271-278): hit point is
            // the point of the victim's sphere facing the blast.
            struct Victim {
                id: ObjectId,
                center: Vec3,
                radius: f32,
            }
            let mut victims: Vec<Victim> = Vec::new();
            objs.find_objects_3d(
                self.center,
                self.radius,
                1.0,
                self.hitmask,
                self.skip,
                |_, id| {
                    if let Some(o) = objs.get(id) {
                        victims.push(Victim {
                            id,
                            center: o.core().geo_center,
                            radius: o.core().radius,
                        });
                    }
                    Control::Continue
                },
            );
            for v in victims {
                let anorm = (self.center - v.center).normalize_or_zero();
                weapon_hit(
                    objs,
                    self.weapon,
                    TraceStop::Object(v.id),
                    v.center + anorm * v.radius,
                    0,
                    0.0,
                );
            }
        }
        false
    }

    /// The geosphere shell (MatrixEffectBigBoom.cpp:85-158 generation,
    /// :228-372 draw): radius shrinks `R → 0`, alpha `k*512` clamped,
    /// 3D texcoords = vertex position run through the spinning
    /// `m_TMat`, additive blend.
    pub fn draw(&self, q: &mut crate::matrix_lib::three_g::billboard::BillboardQueue) {
        use crate::matrix_lib::three_g::billboard::{QueuedTri, TexRef};
        let k = (self.ttl / self.ttl0).clamp(0.0, 1.0);
        if k <= 0.0 {
            return;
        }
        let a = ((k * 512.0) as u32).min(255);
        let color = (a << 24) | self.color_rgb;
        let scale = self.radius - self.radius * k;
        let rot = glam::Quat::from_euler(
            glam::EulerRot::ZXY,
            k * self.k_rot[0],
            k * self.k_rot[1],
            k * self.k_rot[2],
        );
        let tscale = k * 2.0;
        for tri in geosphere().chunks_exact(3) {
            let mut v = [Vec3::ZERO; 3];
            let mut uv = [[0.0f32; 2]; 3];
            for i in 0..3 {
                v[i] = self.center + tri[i] * scale;
                // Raw rotated coords ×k·2 with WRAP addressing — the
                // spinning noise tiles up to ~4× across the shell
                // (MatrixEffectBigBoom.cpp:253-258); the *0.5+0.5
                // remap squeezed one stretched clamped copy.
                let t = rot * tri[i] * tscale;
                uv[i] = [t.x, t.y];
            }
            q.tris.push(QueuedTri {
                v,
                uv,
                color,
                tex: TexRef::Path(
                    crate::matrix_game::effects::effects_renderer::TEXTURE_PATH_BIGBOOM,
                ),
                additive: true,
            });
        }
    }
}

/// Unit geosphere triangles — the C++ builds a 5-point bipyramid and
/// subdivides 3 times with normalized midpoints + centers
/// (MatrixEffectBigBoom.cpp:85-158). Cached after the first build.
fn geosphere() -> &'static [Vec3] {
    use std::sync::OnceLock;
    static SPHERE: OnceLock<Vec<Vec3>> = OnceLock::new();
    SPHERE.get_or_init(|| {
        let p = [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.866_025_4, 0.0),
            Vec3::new(-0.5, -0.866_025_4, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let mut tris: Vec<[Vec3; 3]> = vec![
            [p[0], p[1], p[2]],
            [p[0], p[2], p[3]],
            [p[0], p[3], p[1]],
            [p[4], p[2], p[1]],
            [p[4], p[3], p[2]],
            [p[4], p[1], p[3]],
        ];
        for _ in 0..3 {
            let mut next = Vec::with_capacity(tris.len() * 4);
            for t in &tris {
                let m01 = ((t[0] + t[1]) * 0.5).normalize();
                let m12 = ((t[1] + t[2]) * 0.5).normalize();
                let m20 = ((t[2] + t[0]) * 0.5).normalize();
                next.push([t[0], m01, m20]);
                next.push([m01, t[1], m12]);
                next.push([m20, m12, t[2]]);
                next.push([m01, m12, m20]);
            }
            tris = next;
        }
        tris.into_iter().flatten().collect()
    })
}
