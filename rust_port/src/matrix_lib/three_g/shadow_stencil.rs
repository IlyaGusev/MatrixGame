//! Port of `CVOShadowStencil` (MatrixLib/3G/src/ShadowStencil.cpp) —
//! stencil shadow volume builder.
//!
//! `Build` walks the mesh's silhouette-edge adjacency (`VoMesh::edges`,
//! the on-disk `frames/edges` chunk) with the light direction compressed
//! to the same `SVONormal` integer form as the stored face normals. An
//! edge is on the silhouette when the two adjacent faces' integer dot
//! products against the light have different sign bits. Each silhouette
//! edge extrudes a quad from its two vertices to `vertex + light*len`.
//!
//! Geometry is cached per animation frame keyed on (compressed light,
//! len) exactly like the C++ (`SSSFrameData::m_light/m_len`, rebuild
//! skipped when the light matches and len moved < 1.0). The GPU upload
//! lives with the renderer (`matrix_game/shadow.rs`), matching the
//! original's split between `Build` and `DX_Prepare`/`Render`.

use glam::Vec3;

use super::vector_object::VoMesh;

/// Per-frame cached volume geometry (`SSSFrameData`).
#[derive(Default, Clone)]
struct FrameData {
    /// Vertex pairs: even = mesh vertex, odd = vertex + light*len.
    verts: Vec<[f32; 3]>,
    inds: Vec<u16>,
    len: f32,
    /// Compressed light (`SVONormal.all`) this data was built for.
    light: [u8; 4],
    built: bool,
}

pub struct ShadowStencil {
    frames: Vec<FrameData>,
    /// `m_FrameFor` — the frame whose geometry `geometry()` exposes.
    current: usize,
}

/// Compress a direction into `SVONormal` form (ShadowStencil.cpp:103-118):
/// sign-bit mask (bit0=x, bit1=y, bit2=z, from the raw float sign bits)
/// plus round-to-nearest-even `|v|*255` magnitudes.
fn compress_light(v: Vec3) -> [u8; 4] {
    let n = v.normalize_or_zero();
    let s = ((n.x.to_bits() >> 31) | ((n.y.to_bits() >> 30) & 2) | ((n.z.to_bits() >> 29) & 4))
        as u8;
    [
        (n.x.abs() * 255.0).round_ties_even() as u8,
        (n.y.abs() * 255.0).round_ties_even() as u8,
        (n.z.abs() * 255.0).round_ties_even() as u8,
        s,
    ]
}

/// Integer dot of two `SVONormal`s: sum of per-axis |a|*|b| each negated
/// when the axes' sign bits differ (ShadowStencil.cpp:205-224).
fn signed_dot(n: &[u8; 4], l: &[u8; 4]) -> i32 {
    let sign = n[3] ^ l[3];
    let mut dot = 0i32;
    for a in 0..3 {
        let v = n[a] as i32 * l[a] as i32;
        dot += if sign >> a & 1 != 0 { -v } else { v };
    }
    dot
}

impl ShadowStencil {
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            current: 0,
        }
    }

    /// Port of `CVOShadowStencil::Build` (ShadowStencil.cpp:65).
    ///
    /// `v_light` is the light direction in object space (callers
    /// transform the map light by the unit's inverse world matrix);
    /// `len` the extrusion length before the internal
    /// `+ radius*2 + geo_center.z` adjustment; `invert` flips the
    /// winding for mirrored units.
    pub fn build(&mut self, vo: &VoMesh, frame: usize, v_light: Vec3, len: f32, invert: bool) {
        if vo.edges.is_empty() {
            return;
        }
        let mut dirty = false;
        if self.frames.len() != vo.frames.len() {
            self.frames = vec![FrameData::default(); vo.frames.len()];
            dirty = true;
        }

        let light = compress_light(v_light);
        let k = &vo.frames[frame];
        let len = len + k.radius * 2.0 + k.geo_center[2];

        // Frame/light/len cache check (ShadowStencil.cpp:125-141): keep
        // the frame's cached geometry when the mesh didn't change and
        // the light/len still match.
        let fd = &self.frames[frame];
        self.current = frame;
        if !dirty && fd.built && light == fd.light && (len - fd.len).abs() < 1.0 {
            return;
        }

        let lenv = v_light * len;
        let fd = &mut self.frames[frame];
        fd.len = len;
        fd.light = light;
        fd.built = true;
        fd.verts.clear();
        fd.inds.clear();

        // Local vertex dedup (the C++ `verts` _alloca scratch).
        let mut seen: std::collections::HashMap<u32, u16> = std::collections::HashMap::new();

        for e in &vo.edges[k.edge_start..k.edge_start + k.edge_cnt] {
            let dot0 = signed_dot(&e.n0, &light);
            let dot1 = signed_dot(&e.n1, &light);
            // Silhouette iff the sign bits differ (0 counts as positive).
            if (dot0 ^ dot1) >= 0 {
                continue;
            }

            let mut emit = |vi: u32| -> u16 {
                *seen.entry(vi).or_insert_with(|| {
                    let idx = (fd.verts.len() / 2) as u16;
                    let p = Vec3::from(vo.vertices[vi as usize].position);
                    fd.verts.push(p.into());
                    fd.verts.push((p + lenv).into());
                    idx
                })
            };
            let vi0 = emit(e.v0);
            let vi1 = emit(e.v1);

            // Two triangles per silhouette edge; winding by which face
            // is lit (ShadowStencil.cpp:282-308).
            if (dot0 >= 0) ^ invert {
                fd.inds.extend_from_slice(&[
                    vi1 * 2,
                    vi0 * 2,
                    vi1 * 2 + 1,
                    vi0 * 2,
                    vi0 * 2 + 1,
                    vi1 * 2 + 1,
                ]);
            } else {
                fd.inds.extend_from_slice(&[
                    vi1 * 2 + 1,
                    vi0 * 2 + 1,
                    vi1 * 2,
                    vi0 * 2 + 1,
                    vi0 * 2,
                    vi1 * 2,
                ]);
            }
        }
    }

    /// `IsReady` — geometry exists for the current frame.
    pub fn is_ready(&self) -> bool {
        self.frames.get(self.current).is_some_and(|f| f.built)
    }

    /// Current frame's volume mesh, or `None` when nothing was built.
    pub fn geometry(&self) -> Option<(&[[f32; 3]], &[u16])> {
        let fd = self.frames.get(self.current)?;
        if !fd.built || fd.inds.is_empty() {
            return None;
        }
        Some((&fd.verts, &fd.inds))
    }
}

impl Default for ShadowStencil {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_lib::three_g::vector_object::{VoEdge, VoFrame, VoMesh, VoVertex};

    /// SVONormal for an axis-aligned direction.
    fn axis_normal(x: f32, y: f32, z: f32) -> [u8; 4] {
        compress_light(Vec3::new(x, y, z))
    }

    fn vertex(p: [f32; 3]) -> VoVertex {
        VoVertex {
            position: p,
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        }
    }

    /// One edge between +Z-facing and -Z-facing tris: silhouette for a
    /// horizontal light along +X, not for a vertical light.
    fn test_mesh() -> VoMesh {
        VoMesh {
            vertices: vec![vertex([0.0, 0.0, 1.0]), vertex([0.0, 1.0, 1.0])],
            surfaces: Vec::new(),
            frames: vec![VoFrame {
                bounds_min: [0.0; 3],
                bounds_max: [0.0; 3],
                geo_center: [0.0; 3],
                radius: 1.0,
                surfaces: Vec::new(),
                edge_start: 0,
                edge_cnt: 1,
            }],
            animations: Vec::new(),
            matrices: Vec::new(),
            all_matrices: Vec::new(),
            lights: Vec::new(),
            edges: vec![VoEdge {
                v0: 0,
                v1: 1,
                n0: axis_normal(1.0, 0.0, 0.5),
                n1: axis_normal(-1.0, 0.0, 0.5),
            }],
        }
    }

    #[test]
    fn silhouette_edge_extrudes_quad() {
        let vo = test_mesh();
        let mut ss = ShadowStencil::new();
        // Light along -X: n0·l < 0, n1·l > 0 → silhouette.
        ss.build(&vo, 0, Vec3::new(-1.0, 0.0, 0.0), 10.0, false);
        let (verts, inds) = ss.geometry().expect("geometry");
        assert_eq!(verts.len(), 4); // 2 verts × (base, extruded)
        assert_eq!(inds.len(), 6); // 2 triangles
        // len = 10 + radius*2 + geo_center.z = 12; extruded = v + light*12.
        assert_eq!(verts[1][0], -12.0);
        assert_eq!(verts[1][2], 1.0);
    }

    #[test]
    fn no_silhouette_when_both_faces_lit() {
        let vo = test_mesh();
        let mut ss = ShadowStencil::new();
        // Light along -Z: both faces have positive dot (both n have z=+0.5).
        ss.build(&vo, 0, Vec3::new(0.0, 0.0, -1.0), 10.0, false);
        assert!(ss.geometry().is_none());
    }

    #[test]
    fn cache_skips_rebuild_for_same_light_and_len() {
        let vo = test_mesh();
        let mut ss = ShadowStencil::new();
        ss.build(&vo, 0, Vec3::new(-1.0, 0.0, 0.0), 10.0, false);
        let before = ss.geometry().expect("geometry").0.as_ptr();
        // len delta below the 1.0 threshold — cache must hold.
        ss.build(&vo, 0, Vec3::new(-1.0, 0.0, 0.0), 10.5, false);
        assert_eq!(before, ss.geometry().expect("geometry").0.as_ptr());
    }

    #[test]
    fn invert_flips_winding() {
        let vo = test_mesh();
        let mut ss = ShadowStencil::new();
        ss.build(&vo, 0, Vec3::new(-1.0, 0.0, 0.0), 10.0, false);
        let normal: Vec<u16> = ss.geometry().unwrap().1.to_vec();
        let mut ss2 = ShadowStencil::new();
        ss2.build(&vo, 0, Vec3::new(-1.0, 0.0, 0.0), 10.0, true);
        let inverted: Vec<u16> = ss2.geometry().unwrap().1.to_vec();
        assert_ne!(normal, inverted);
    }
}
