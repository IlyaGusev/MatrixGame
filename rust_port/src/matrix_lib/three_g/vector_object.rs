//! `.vo` mesh loader — ports VectorObject.cpp:130-386.
//!
//! Each .vo file is a CStorage archive with these relevant buffers:
//!   verts/data     : one array of SVOVertex = pos[3]f32 + normal[3]f32 + uv[2]f32 (32 bytes)
//!   idxs/data      : one array of u16 triangle indices
//!   surfs/texs     : wide-char per-surface texture references
//!   unions/data    : surface/triangle ranges as SVOUnion (24 bytes)
//!   frames/data    : SVOKadr[] (64 bytes each) — per-frame bounds + union range
//!   frames/unions  : u32 indices into unions/data, sliced by each frame's [UnionStart, UnionStart+UnionCnt)
//!   anims/{name,id,disp,cnt,frames}: named animations pointing at per-anim sequences
//!     of (frame_index, duration_ms) pairs (SVOFrameIndex[] in anims/frames)
//!
//! We parse all frames and animations. `VoMesh::surfaces` stays as frame 0's
//! surface list so existing consumers keep working; `VoMesh::frames[n].surfaces`
//! exposes the same data per frame.

use anyhow::{Context, Result};

use crate::matrix_lib::base::storage::Storage;

pub struct VoMesh {
    pub vertices: Vec<VoVertex>,
    /// Frame 0's surfaces. Convenience alias for `frames[0].surfaces` so
    /// non-animated consumers don't need to touch the frame list.
    pub surfaces: Vec<VoSurfaceMesh>,
    /// All frames — at least one. Each frame has its own surface/triangle
    /// partition computed from `frames/unions` + `unions/data`.
    pub frames: Vec<VoFrame>,
    /// Named animations, each a sequence of (frame index, duration ms).
    /// Empty for non-animated objects.
    pub animations: Vec<VoAnimation>,
    /// Named bone matrices. Ports `SVOMatrix[]` (VectorObject.hpp:227) +
    /// `SVOMatrix::m_MatrixStart` indirection into `m_AllMatrixs` at
    /// VectorObject.cpp:188-215. Used by `CMatrixRobot::Draw` for
    /// weapon/head attach points on the parent armor/chassis graph
    /// (MatrixObjectRobot.cpp:460-575).
    pub matrices: Vec<VoMatrix>,
    /// Flat `D3DXMATRIX[]` bone pose table. Each bone's per-frame
    /// matrix lives at `matrices[i].start + frame` into this array.
    /// Stored in D3D row-major layout (same 16 floats as on disk).
    pub all_matrices: Vec<[f32; 16]>,
    /// On-mesh emissive lights (`$`-named bones, `InitLights`,
    /// VectorObject.cpp:1769-1858): billboard lights whose colour animates
    /// over the interval table.
    pub lights: Vec<VoLight>,
}

/// One `$`-bone light: a billboard at bone `matid` whose colour LERPs
/// through `intervals` (sorted by time) over `period` ms.
#[derive(Clone, Debug)]
pub struct VoLight {
    pub matid: u32,
    pub radius: f32,
    pub period: i32,
    /// `(time1, time2, c1, c2)` — colour LICs `c1→c2` over `[time1,time2]`.
    pub intervals: Vec<(i32, i32, u32, u32)>,
}

impl VoLight {
    /// Colour at animation time `t_ms` (LIC within the active interval).
    pub fn color_at(&self, t_ms: i32) -> u32 {
        if self.intervals.is_empty() {
            return 0xFFFF_FFFF;
        }
        let t = if self.period > 0 { t_ms.rem_euclid(self.period) } else { 0 };
        for &(t1, t2, c1, c2) in &self.intervals {
            if t >= t1 && t <= t2 {
                let span = (t2 - t1).max(1) as f32;
                let k = ((t - t1) as f32 / span).clamp(0.0, 1.0);
                let ch = |sh: u32| {
                    let a = ((c1 >> sh) & 0xFF) as f32;
                    let b = ((c2 >> sh) & 0xFF) as f32;
                    (a + (b - a) * k) as u32
                };
                return (ch(24) << 24) | (ch(16) << 16) | (ch(8) << 8) | ch(0);
            }
        }
        self.intervals[0].2
    }
}

/// Parse the `$`-named light bones from the VO property table
/// (`InitLights`, VectorObject.cpp:1769-1858).
fn parse_vo_lights(stor: &Storage, matrices: &[VoMatrix]) -> Vec<VoLight> {
    use crate::matrix_lib::base::wstr;
    let (Some(pn), Some(pv)) = (
        stor.get_buf("properties", "name"),
        stor.get_buf("properties", "value"),
    ) else {
        return Vec::new();
    };
    let mut lights = Vec::new();
    for m in matrices.iter().filter(|m| m.name.starts_with('$')) {
        let Some(idx) = pn.find_as_wstr(&m.name) else {
            continue;
        };
        let info = pv.get_as_wstr(idx);
        let radius = wstr::double_par(&info, 0, ",") as f32;
        let cnt = wstr::count_par(&info, ",");
        let mut pts: Vec<(i32, u32)> = Vec::new();
        for pa in 1..cnt {
            let par = wstr::str_par(&info, pa, ":");
            let _ = par;
            let field = wstr::str_par(&info, pa, ",");
            let time = wstr::int_par(field, 0, ":");
            let hex = wstr::str_par(field, 1, ":");
            let color = 0xFF00_0000 | u32::from_str_radix(hex.trim(), 16).unwrap_or(0);
            pts.push((time, color));
        }
        pts.sort_by_key(|p| p.0);
        let mut intervals = Vec::new();
        let period;
        if pts.is_empty() {
            intervals.push((0, 0, 0xFFFF_FFFF, 0xFFFF_FFFF));
            period = 0;
        } else if pts.len() == 1 {
            intervals.push((pts[0].0, pts[0].0, pts[0].1, pts[0].1));
            period = 0;
        } else {
            period = pts.last().unwrap().0;
            for w in pts.windows(2) {
                intervals.push((w[0].0, w[1].0, w[0].1, w[1].1));
            }
        }
        lights.push(VoLight {
            matid: m.id,
            radius,
            period,
            intervals,
        });
    }
    lights
}

/// Port of `SVOMatrix` (VectorObject.hpp:227). `start` is the offset
/// into `VoMesh::all_matrices` where this bone's frame list begins.
#[derive(Clone, Debug)]
pub struct VoMatrix {
    pub name: String,
    pub id: u32,
    pub start: u32,
}

impl VoMesh {
    /// Port of `CVectorObject::GetMatrixById` (VectorObject.cpp:459-470).
    /// Returns the 16-float row-major D3DXMATRIX for the named bone at
    /// the given frame, or `None` when no bone with that id exists.
    pub fn matrix_by_id(&self, id: u32, frame: usize) -> Option<[f32; 16]> {
        let m = self.matrices.iter().find(|m| m.id == id)?;
        self.all_matrices.get(m.start as usize + frame).copied()
    }

    /// Iterate all bone matrices at `frame` with their `(name, id, mat16)`.
    /// Used by the map loader to build the weapon-slot table on armor VOs
    /// (MatrixMap.cpp:270-326) — armor bones with id >= 20 and name
    /// starting with `W` declare weapon attach points.
    pub fn iter_matrices_at(
        &self,
        frame: usize,
    ) -> impl Iterator<Item = (&str, u32, [f32; 16])> + '_ {
        self.matrices.iter().filter_map(move |m| {
            self.all_matrices
                .get(m.start as usize + frame)
                .copied()
                .map(|mat| (m.name.as_str(), m.id, mat))
        })
    }
}

/// Per-frame geometry state. Mirrors SVOKadr (VectorObject.hpp:214) plus the
/// materialized surface partition for that frame.
#[derive(Clone, Debug)]
pub struct VoFrame {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub geo_center: [f32; 3],
    pub radius: f32,
    pub surfaces: Vec<VoSurfaceMesh>,
}

/// Ports SVOAnimation (VectorObject.hpp:237). `frames` resolves the raw
/// `anims/frames` SVOFrameIndex[] slice selected by this animation's
/// `FramesStart`/`FramesCnt`.
#[derive(Clone, Debug)]
pub struct VoAnimation {
    pub name: String,
    pub id: u32,
    pub frames: Vec<VoFrameRef>,
}

/// Ports SVOFrameIndex (VectorObject.hpp:252). `time_ms` is the duration the
/// pose from `frame_index` should hold before advancing.
#[derive(Clone, Copy, Debug)]
pub struct VoFrameRef {
    pub frame_index: usize,
    pub time_ms: i32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VoVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct VoSurfaceMesh {
    pub indices: Vec<u32>,
    pub texture_ref: Option<String>,
}

use crate::matrix_lib::three_g::texture::parse_scroll;
pub use crate::matrix_lib::three_g::texture::{
    has_trans_suffix, merge_materials, parse_material_spec, parse_material_spec_with_prefix,
    resolve_alpha_test_with_txt, MaterialSpec,
};

pub fn parse_vo(data: &[u8]) -> Result<VoMesh> {
    let stor = Storage::from_bytes(data).context("parsing VO as CStorage")?;

    let verts_buf = stor
        .get_buf("verts", "data")
        .context("missing verts/data")?;
    let idxs_buf = stor.get_buf("idxs", "data").context("missing idxs/data")?;
    if verts_buf.arrays_count() == 0 || idxs_buf.arrays_count() == 0 {
        anyhow::bail!("VO has empty verts or idxs");
    }

    let vb = verts_buf.get_bytes(0);
    let vcount = vb.len() / 32;
    let mut vertices = Vec::with_capacity(vcount);
    for i in 0..vcount {
        let off = i * 32;
        let f = |o: usize| {
            f32::from_le_bytes([
                vb[off + o],
                vb[off + o + 1],
                vb[off + o + 2],
                vb[off + o + 3],
            ])
        };
        vertices.push(VoVertex {
            position: [f(0), f(4), f(8)],
            normal: [f(12), f(16), f(20)],
            uv: [f(24), f(28)],
        });
    }

    let ib = idxs_buf.get_bytes(0);
    let icount = ib.len() / 2;
    let mut indices = Vec::with_capacity(icount);
    for i in 0..icount {
        let off = i * 2;
        indices.push(u16::from_le_bytes([ib[off], ib[off + 1]]));
    }

    let frames = parse_frames(&stor, &indices).context("parsing VO frames")?;
    let animations = parse_animations(&stor);
    let (matrices, all_matrices) = parse_matrices(&stor);
    let surfaces = frames
        .first()
        .map(|f| f.surfaces.clone())
        .unwrap_or_default();

    let lights = parse_vo_lights(&stor, &matrices);
    Ok(VoMesh {
        vertices,
        surfaces,
        frames,
        animations,
        matrices,
        all_matrices,
        lights,
    })
}

/// Port of the `matrices/{name,id,disp,data}` buffer reader at
/// VectorObject.cpp:187-215. `name` is WCHAR, `id` + `disp` are u32,
/// `data` is a flat D3DXMATRIX[] array (16 f32 per matrix, row-major).
fn parse_matrices(stor: &Storage) -> (Vec<VoMatrix>, Vec<[f32; 16]>) {
    let Some(names) = stor.get_buf("matrices", "name") else {
        return (Vec::new(), Vec::new());
    };
    let Some(ids) = stor.get_buf("matrices", "id") else {
        return (Vec::new(), Vec::new());
    };
    let Some(disps) = stor.get_buf("matrices", "disp") else {
        return (Vec::new(), Vec::new());
    };
    let Some(data) = stor.get_buf("matrices", "data") else {
        return (Vec::new(), Vec::new());
    };
    let n = names.arrays_count();
    let mut matrices = Vec::with_capacity(n);
    let id_bytes = ids.get_bytes(0);
    let disp_bytes = disps.get_bytes(0);
    let read_u32 = |buf: &[u8], i: usize| -> u32 {
        let o = i * 4;
        u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
    };
    for i in 0..n {
        let name = names.get_as_wstr(i);
        // Strip trailing NUL from wchar string (C++ wcslen semantics).
        let name = name.trim_end_matches('\0').to_string();
        let id = read_u32(id_bytes, i);
        let start = read_u32(disp_bytes, i);
        matrices.push(VoMatrix { name, id, start });
    }
    let raw = data.get_bytes(0);
    let count = raw.len() / 64;
    let mut all = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 64;
        let mut m = [0.0f32; 16];
        for (j, slot) in m.iter_mut().enumerate() {
            let o = off + j * 4;
            *slot = f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
        }
        all.push(m);
    }
    (matrices, all)
}

/// One sub-mesh declared inside a `.cvo` (CVectorObjectGroup) file. Ports
/// `CVectorObjectGroupUnit` (VectorObject.hpp:~1900) — the fields we populate
/// are a subset: the model path, a material built from the block's
/// `Texture*` params, and the optional id/link bookkeeping the original uses
/// for frame-matrix hierarchies.
///
/// `link` follows `CVectorObjectGroup::Load` (VectorObject.cpp:2491, 2527-2549):
/// a raw `Link=<name_or_id>,<matrix_id>` string split by comma. We keep the
/// raw text so the consumer can resolve it after all units are loaded (same
/// two-pass order as the original).
#[derive(Clone, Debug)]
pub struct CvoUnit {
    /// Sub-block name (`BlockGetName(i)` in the C++). Empty for anonymous
    /// `{ ... }` blocks — also seen in shipped `b0.cvo`.
    pub name: String,
    pub id: Option<i32>,
    pub model_path: String,
    pub material: MaterialSpec,
    pub link_raw: Option<String>,
}

/// Parsed contents of a `.cvo` file. Ports `CVectorObjectGroup` in
/// VectorObject.cpp:2415-2553: one `CvoUnit` per top-level block, in source
/// order. Caller is expected to load each `model_path` via `parse_vo`.
#[derive(Clone, Debug, Default)]
pub struct CvoGroup {
    pub units: Vec<CvoUnit>,
}

/// Ports `CVectorObjectGroup::Load` (VectorObject.cpp:2415-2553). We mirror
/// the `CacheReplaceFileNameAndExt` path-stitching (CWStr.cpp): `Model`,
/// `Texture`, `TextureGloss`, `TextureMask`, `TextureBack` values are
/// resolved relative to the `.cvo` file's directory. The `?Trans` suffix is
/// honored exactly like `parse_material_spec` does for regular VO specs.
///
/// `cvo_path` is the archive path of the .cvo so we can derive the directory
/// for asset siblings. `bytes` is the raw file contents.
pub fn parse_cvo(cvo_path: &str, bytes: &[u8]) -> CvoGroup {
    let bp = crate::matrix_lib::base::blockpar::BlockPar::parse_bytes(bytes);
    let dir = cvo_path
        .rsplit_once('/')
        .map(|(d, _)| format!("{d}/"))
        .unwrap_or_default();

    let mut units = Vec::new();
    for entry in bp.entries() {
        let crate::matrix_lib::base::blockpar::Entry::Block { name, block, .. } = entry else {
            continue;
        };

        // Model=<rel path>  — drops any `?...` suffix before path stitching
        // (VectorObject.cpp:2466-2469).
        let model_raw = block.par_get_ne("Model").unwrap_or("").trim();
        let model_clean = model_raw.split('?').next().unwrap_or("").trim();
        if model_clean.is_empty() {
            continue;
        }
        let model_path = join_cvo_sibling(&dir, model_clean);

        let diffuse = block
            .par_get_ne("Texture")
            .and_then(|v| resolve_cvo_texture(v, &dir));
        let gloss = block
            .par_get_ne("TextureGloss")
            .and_then(|v| resolve_cvo_texture(v, &dir));
        let mask = block
            .par_get_ne("TextureMask")
            .and_then(|v| resolve_cvo_texture(v, &dir));
        // `CVectorObjectGroup::Load` (VectorObject.cpp:2493) explicitly clears
        // TextureBack when TextureMask is absent, so do the same here.
        let back = if mask.is_some() {
            block
                .par_get_ne("TextureBack")
                .and_then(|v| resolve_cvo_texture(v, &dir))
        } else {
            None
        };
        let scroll = block
            .par_get_ne("TextureBackScroll")
            .map(parse_scroll)
            .unwrap_or([0.0, 0.0]);
        let alpha_test = block.par_get_ne("Texture").is_some_and(has_trans_suffix);

        let id = block
            .par_get_ne("Id")
            .and_then(|v| v.trim().parse::<i32>().ok());
        let link_raw = block.par_get_ne("Link").map(|s| s.to_string());

        units.push(CvoUnit {
            name: name.clone(),
            id,
            model_path,
            material: MaterialSpec {
                diffuse,
                gloss,
                back,
                mask,
                scroll,
                alpha_test,
            },
            link_raw,
        });
    }

    CvoGroup { units }
}

fn join_cvo_sibling(dir: &str, name: &str) -> String {
    // Normalize backslashes and collapse `./` prefix like the original's
    // `CacheReplaceFileNameAndExt` does implicitly (CWStr.cpp).
    let n = name.trim().trim_start_matches(".\\").replace('\\', "/");
    if n.contains('/') {
        n
    } else {
        format!("{dir}{n}")
    }
}

fn resolve_cvo_texture(raw: &str, dir: &str) -> Option<String> {
    // CVO texture paths come without an extension (e.g. `b0_1`, `platform`).
    // Strip any `?Trans`/other suffix, prefix with the CVO directory; the
    // caller's asset-read closure handles extension resolution (.dds/.png)
    // the same way it does for regular object textures.
    let clean = raw.trim().trim_end_matches('\0');
    if clean.is_empty() || clean == "." {
        return None;
    }
    let base = clean.split('?').next().unwrap_or("").trim();
    if base.is_empty() {
        return None;
    }
    Some(join_cvo_sibling(dir, base))
}

/// SVOKadr size on disk — `/Zp1` keeps the struct 64 bytes (6 floats bounds +
/// 1 float radius + 6 i32 counters; VectorObject.hpp:214).
const SVO_KADR_SIZE: usize = 64;
/// SVOUnion size on disk — 6 i32 under `/Zp1` (VectorObject.hpp:200).
const SVO_UNION_SIZE: usize = 24;
/// SVOFrameIndex size — two i32 (VectorObject.hpp:252).
const SVO_FRAME_INDEX_SIZE: usize = 8;

/// Ports VectorObject.cpp:260-344. The original unconditionally reads
/// `frames/data`, `frames/unions`, and `unions/data` (VectorObject.cpp:263-280,
/// 335-342) — a missing or undersized buffer there was a hard failure, not a
/// point where the engine synthesized a stand-in pose. We mirror that: if the
/// metadata isn't usable, the VO is rejected.
fn parse_frames(stor: &Storage, indices: &[u16]) -> Result<Vec<VoFrame>> {
    let mut texture_refs: Vec<Option<String>> = Vec::new();
    if let Some(buf) = stor.get_buf("surfs", "texs") {
        for i in 0..buf.arrays_count() {
            let s = buf.get_as_wstr(i);
            let s = s.trim_end_matches('\0').trim().to_string();
            texture_refs.push(if s.is_empty() { None } else { Some(s) });
        }
    }
    let surface_count = texture_refs.len().max(1);

    let frame_data = stor
        .get_buf("frames", "data")
        .map(|b| b.get_bytes(0))
        .context("missing frames/data")?;
    let frame_unions = stor
        .get_buf("frames", "unions")
        .map(|b| b.get_bytes(0))
        .context("missing frames/unions")?;
    let unions_data = stor
        .get_buf("unions", "data")
        .map(|b| b.get_bytes(0))
        .context("missing unions/data")?;
    if frame_data.len() < SVO_KADR_SIZE {
        anyhow::bail!(
            "frames/data too small: {} bytes (need >= {})",
            frame_data.len(),
            SVO_KADR_SIZE
        );
    }
    if frame_unions.len() < 4 {
        anyhow::bail!("frames/unions too small: {} bytes", frame_unions.len());
    }
    if unions_data.len() < SVO_UNION_SIZE {
        anyhow::bail!(
            "unions/data too small: {} bytes (need >= {})",
            unions_data.len(),
            SVO_UNION_SIZE
        );
    }

    let frame_count = frame_data.len() / SVO_KADR_SIZE;
    let mut frames = Vec::with_capacity(frame_count);
    for fi in 0..frame_count {
        let off = fi * SVO_KADR_SIZE;
        let bounds_min = [
            read_f32(frame_data, off),
            read_f32(frame_data, off + 4),
            read_f32(frame_data, off + 8),
        ];
        let bounds_max = [
            read_f32(frame_data, off + 12),
            read_f32(frame_data, off + 16),
            read_f32(frame_data, off + 20),
        ];
        let geo_center = [
            read_f32(frame_data, off + 24),
            read_f32(frame_data, off + 28),
            read_f32(frame_data, off + 32),
        ];
        let radius = read_f32(frame_data, off + 36);
        let union_start = read_i32(frame_data, off + 40).max(0) as usize;
        let union_count = read_i32(frame_data, off + 44).max(0) as usize;

        let mut surfs: Vec<VoSurfaceMesh> = (0..surface_count)
            .map(|i| VoSurfaceMesh {
                indices: Vec::new(),
                texture_ref: texture_refs.get(i).cloned().unwrap_or(None),
            })
            .collect();

        for i in 0..union_count {
            let idx_off = (union_start + i) * 4;
            if idx_off + 4 > frame_unions.len() {
                break;
            }
            let union_index = read_u32(frame_unions, idx_off) as usize;
            let union_off = union_index * SVO_UNION_SIZE;
            if union_off + SVO_UNION_SIZE > unions_data.len() {
                continue;
            }
            let surface = read_i32(unions_data, union_off).max(0) as usize;
            let base = read_i32(unions_data, union_off + 4);
            let ibase = read_i32(unions_data, union_off + 8);
            let tri_cnt = read_i32(unions_data, union_off + 16).max(0) as usize;
            let tri_start = read_i32(unions_data, union_off + 20).max(0) as usize;

            let dst_idx = surface.min(surfs.len().saturating_sub(1));
            let dst = &mut surfs[dst_idx];
            // SVOUnion's m_IBase (VectorObject.hpp:200-212): when negative,
            // the draw call starts at -m_IBase instead of m_TriStart*3
            // (VectorObject.cpp:1366-1371).
            let start = if ibase < 0 {
                (-ibase) as usize
            } else {
                tri_start * 3
            };
            let end = start + tri_cnt * 3;
            if end > indices.len() {
                continue;
            }
            dst.indices.extend(
                indices[start..end]
                    .iter()
                    .map(|&idx| (idx as i32 + base) as u32),
            );
        }

        let surfaces: Vec<VoSurfaceMesh> = surfs
            .into_iter()
            .filter(|s| !s.indices.is_empty())
            .collect();

        frames.push(VoFrame {
            bounds_min,
            bounds_max,
            geo_center,
            radius,
            surfaces,
        });
    }

    if frames.is_empty() {
        anyhow::bail!("frames/data parsed to zero frames");
    }
    if frames.iter().all(|f| f.surfaces.is_empty()) {
        anyhow::bail!("no frame has any populated surface");
    }

    Ok(frames)
}

fn parse_animations(stor: &Storage) -> Vec<VoAnimation> {
    let Some(name_buf) = stor.get_buf("anims", "name") else {
        return Vec::new();
    };
    let Some(id_buf) = stor.get_buf("anims", "id") else {
        return Vec::new();
    };
    let Some(disp_buf) = stor.get_buf("anims", "disp") else {
        return Vec::new();
    };
    let Some(cnt_buf) = stor.get_buf("anims", "cnt") else {
        return Vec::new();
    };
    let Some(frames_buf) = stor.get_buf("anims", "frames") else {
        return Vec::new();
    };

    let count = name_buf.arrays_count();
    if count == 0
        || id_buf.arrays_count() == 0
        || disp_buf.arrays_count() == 0
        || cnt_buf.arrays_count() == 0
        || frames_buf.arrays_count() == 0
    {
        return Vec::new();
    }

    let ids = id_buf.get_bytes(0);
    let disps = disp_buf.get_bytes(0);
    let cnts = cnt_buf.get_bytes(0);
    let frames_blob = frames_buf.get_bytes(0);

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let name = name_buf
            .get_as_wstr(i)
            .trim_end_matches('\0')
            .trim()
            .to_string();
        let id = read_u32(ids, i * 4);
        let frames_start = read_u32(disps, i * 4) as usize;
        let frames_cnt = read_u32(cnts, i * 4) as usize;

        let mut frames = Vec::with_capacity(frames_cnt);
        for f in 0..frames_cnt {
            let off = (frames_start + f) * SVO_FRAME_INDEX_SIZE;
            if off + SVO_FRAME_INDEX_SIZE > frames_blob.len() {
                break;
            }
            let frame_index = read_i32(frames_blob, off).max(0) as usize;
            let time_ms = read_i32(frames_blob, off + 4);
            frames.push(VoFrameRef {
                frame_index,
                time_ms,
            });
        }

        out.push(VoAnimation { name, id, frames });
    }
    out
}

fn read_i32(bytes: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_f32(bytes: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}
// ====================================================================
// Port of `CVectorObjectAnim` (VectorObject.{cpp,hpp}) — the
// per-instance animation cursor that runs on top of a shared `VoMesh`.
// In C++ each `SMatrixRobotUnit` owns one; the same cursor is
// allocated per robot + per unit in this port.
//
// Ported methods:
//   * `takt(cms)` → `CVectorObjectAnim::Takt(cms)` at
//     VectorObject.cpp:1863-1928 — advances `m_Time`, walks
//     `m_Frame` forward as many ticks as needed, loops or clamps
//     at the end, and syncs `m_VOFrame` (the geometry-frame index
//     the renderer samples from `VoMesh::frames`).
//   * `set_anim_by_name` → `SetAnimByName(name)` at
//     VectorObject.hpp:524 — returns true on failure (matches the
//     original's weird "true means not found" convention).
//   * `first_frame` / `is_anim_end` — same naming as C++.
// ====================================================================

use std::sync::{Arc, OnceLock, RwLock};

/// Global per-chassis VoMesh table, populated once at load time by
/// `RobotsRenderer::new`. Ports the role of `g_CacheHeap`-owned VO
/// singletons in the C++ engine — the AI layer (`robot.rs::logic_takt`)
/// needs to walk frame durations to drive animation without holding
/// a direct reference to the GPU-side renderer.
///
/// Indexed by `ChassisKind::kind_index()` (RUK kind − 1). `None` for
/// kinds whose VO failed to load.
static CHASSIS_VOS: OnceLock<RwLock<[Option<Arc<VoMesh>>; 5]>> = OnceLock::new();

fn chassis_slot() -> &'static RwLock<[Option<Arc<VoMesh>>; 5]> {
    CHASSIS_VOS.get_or_init(|| RwLock::new([const { None }; 5]))
}

pub fn set_chassis_vo(idx: usize, vo: Arc<VoMesh>) {
    if idx < 5 {
        chassis_slot().write().unwrap()[idx] = Some(vo);
    }
}

pub fn chassis_vo(idx: usize) -> Option<Arc<VoMesh>> {
    chassis_slot().read().unwrap().get(idx).cloned().flatten()
}

/// Local-space AABBs for pick tests — the C++ picks against the VO
/// mesh (`CVectorObject::Pick`); a per-kind OBB from the real frame-0
/// bounds is the port's stand-in, replacing the bounding-sphere
/// approximation that made turret fire lines clip their own building
/// and robot shots explode on ruin "air". Registered by the renderers
/// at VO load; absent entries fall back to the sphere.
static BUILDING_BOUNDS: OnceLock<RwLock<std::collections::HashMap<u8, Vec<([f32; 3], [f32; 3])>>>> =
    OnceLock::new();
static MAPOBJ_BOUNDS: OnceLock<RwLock<std::collections::HashMap<i32, ([f32; 3], [f32; 3])>>> =
    OnceLock::new();

fn building_bounds_slot(
) -> &'static RwLock<std::collections::HashMap<u8, Vec<([f32; 3], [f32; 3])>>> {
    BUILDING_BOUNDS.get_or_init(Default::default)
}
fn mapobj_bounds_slot() -> &'static RwLock<std::collections::HashMap<i32, ([f32; 3], [f32; 3])>> {
    MAPOBJ_BOUNDS.get_or_init(Default::default)
}

/// Register a building kind's sub-VO AABBs. Kept per sub-VO — a
/// UNION box would wall off the flat spawn platform (its pad spans
/// the local +y half at z≈0 while the body towers at −y; the union
/// is a phantom 76-unit wall over the platform mouth).
pub fn set_building_boxes(kind: u8, boxes: Vec<([f32; 3], [f32; 3])>) {
    building_bounds_slot().write().unwrap().insert(kind, boxes);
}

/// Nearest ray hit across the building kind's sub-VO boxes (local
/// space). `None` when no boxes are registered OR the ray misses.
/// Callers distinguish the two via [`has_building_boxes`].
pub fn pick_building_boxes(kind: u8, origin: [f32; 3], dir: [f32; 3]) -> Option<f32> {
    let g = building_bounds_slot().read().unwrap();
    let boxes = g.get(&kind)?;
    boxes
        .iter()
        .filter_map(|(mn, mx)| ray_aabb(origin, dir, *mn, *mx))
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

pub fn has_building_boxes(kind: u8) -> bool {
    building_bounds_slot().read().unwrap().contains_key(&kind)
}

pub fn set_mapobj_bounds(type_id: i32, min: [f32; 3], max: [f32; 3]) {
    mapobj_bounds_slot()
        .write()
        .unwrap()
        .insert(type_id, (min, max));
}
pub fn mapobj_bounds(type_id: i32) -> Option<([f32; 3], [f32; 3])> {
    mapobj_bounds_slot().read().unwrap().get(&type_id).copied()
}

/// Ruin MapObjects carry a graph path instead of a type id
/// (`init_as_base_ruins`) — separate string-keyed registry.
static RUIN_BOUNDS: OnceLock<RwLock<std::collections::HashMap<String, ([f32; 3], [f32; 3])>>> =
    OnceLock::new();

fn ruin_bounds_slot() -> &'static RwLock<std::collections::HashMap<String, ([f32; 3], [f32; 3])>>
{
    RUIN_BOUNDS.get_or_init(Default::default)
}

pub fn merge_ruin_bounds(name: &str, min: [f32; 3], max: [f32; 3]) {
    let mut g = ruin_bounds_slot().write().unwrap();
    let e = g.entry(name.to_string()).or_insert((min, max));
    for i in 0..3 {
        e.0[i] = e.0[i].min(min[i]);
        e.1[i] = e.1[i].max(max[i]);
    }
}
pub fn ruin_bounds(name: &str) -> Option<([f32; 3], [f32; 3])> {
    ruin_bounds_slot().read().unwrap().get(name).copied()
}

/// Ray / AABB slab test. Returns the ray parameter of the nearest
/// non-negative hit.
pub fn ray_aabb(origin: [f32; 3], dir: [f32; 3], min: [f32; 3], max: [f32; 3]) -> Option<f32> {
    let mut t0 = 0.0f32;
    let mut t1 = f32::INFINITY;
    for i in 0..3 {
        if dir[i].abs() < 1e-10 {
            if origin[i] < min[i] || origin[i] > max[i] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / dir[i];
        let (mut ta, mut tb) = ((min[i] - origin[i]) * inv, (max[i] - origin[i]) * inv);
        if ta > tb {
            std::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if t0 > t1 {
            return None;
        }
    }
    Some(t0)
}

/// Same pattern for the cannon shaft VOs (`Matrix/Cannon/Shaft{1-4}.vo`),
/// populated by `CannonsRenderer::new`. The cannon game object needs
/// frame durations to drive the fire/idle shaft animation
/// (MatrixObjectCannon.cpp:448-460, 824-851). Indexed by kind − 1.
static CANNON_SHAFT_VOS: OnceLock<RwLock<[Option<Arc<VoMesh>>; 4]>> = OnceLock::new();

fn cannon_shaft_slot() -> &'static RwLock<[Option<Arc<VoMesh>>; 4]> {
    CANNON_SHAFT_VOS.get_or_init(|| RwLock::new([const { None }; 4]))
}

pub fn set_cannon_shaft_vo(idx: usize, vo: Arc<VoMesh>) {
    if idx < 4 {
        cannon_shaft_slot().write().unwrap()[idx] = Some(vo);
    }
}

pub fn cannon_shaft_vo(idx: usize) -> Option<Arc<VoMesh>> {
    cannon_shaft_slot()
        .read()
        .unwrap()
        .get(idx)
        .cloned()
        .flatten()
}

#[derive(Debug, Clone)]
pub struct AnimState {
    /// `m_Anim` — index into `VoMesh::animations`.
    pub anim: i32,
    /// `m_Frame` — animation-local frame cursor (0..frames_count).
    pub frame: i32,
    /// `m_VOFrame` — resolved VO frame index the renderer samples.
    pub vo_frame: usize,
    /// `m_Time` — accumulated animation-local ms.
    pub time_ms: i64,
    /// `m_TimeNext` — game-time threshold at which `frame` advances
    /// (used by `takt`, the normal constant-rate path).
    pub time_next_ms: i64,
    /// `m_AnimLooped` — 0 = play once and hold last frame, 1 = loop.
    pub looped: bool,
    /// `SMatrixRobotUnit::m_NextAnimTime` (MatrixObjectRobot.hpp) —
    /// game-time threshold at which `next_frame` advances (used by
    /// the speed-based per-frame path in `DoAnimation`). Separate
    /// from `time_next_ms` because the speed-based path uses
    /// `g_MatrixMap->GetTime()` as its clock, not the accumulated
    /// anim-local time.
    pub next_anim_time: f64,
}

impl Default for AnimState {
    fn default() -> Self {
        Self {
            anim: 0,
            frame: 0,
            vo_frame: 0,
            time_ms: 0,
            time_next_ms: 0,
            looped: true,
            next_anim_time: 0.0,
        }
    }
}

impl AnimState {
    /// Port of `FirstFrame` (VectorObject.hpp:552): reset frame cursor
    /// and set the next-advance threshold from anim 0's duration.
    pub fn first_frame(&mut self, vo: &VoMesh) {
        self.frame = 0;
        let (idx, time) = anim_frame(vo, self.anim, 0).unwrap_or((0, 0));
        self.vo_frame = idx;
        // `GetAnimFrameTime` = abs(time) (VectorObject.hpp:389) — the
        // sign encodes loopedness, not a negative duration. Clamp to
        // min 1ms — see takt() for why.
        self.time_next_ms = self.time_ms + (time as i64).abs().max(1);
    }

    /// Port of `SetAnimByName(name)` (VectorObject.hpp:524). Returns
    /// `true` on failure (anim not found) to match the C++ convention.
    /// When the name resolves, the cursor resets to frame 0 and the
    /// loop flag is taken from `VoAnim::is_looped` (we don't carry
    /// that yet, so default to the passed `looped`).
    pub fn set_anim_by_name(&mut self, vo: &VoMesh, name: &str, looped: bool) -> bool {
        let Some(idx) = vo.animations.iter().position(|a| a.name == name) else {
            return true;
        };
        self.anim = idx as i32;
        self.looped = looped;
        self.first_frame(vo);
        false
    }

    /// Port of `SetAnimByName(name)` single-arg overload (VectorObject.
    /// hpp:524): loopedness comes from the data (`SetLoopStatus` —
    /// the sign of the new anim's first frame duration, hpp:390).
    pub fn set_anim_by_name_data_looped(&mut self, vo: &VoMesh, name: &str) -> bool {
        let Some(idx) = vo.animations.iter().position(|a| a.name == name) else {
            return true;
        };
        self.anim = idx as i32;
        self.looped = anim_data_looped(vo, self.anim);
        self.first_frame(vo);
        false
    }

    /// Port of `SetAnimByNameNoBegin(name)` (VectorObject.hpp:527):
    /// like `set_anim_by_name` but doesn't restart when the anim is
    /// already current. NB: the C++ calls `SetLoopStatus()` BEFORE
    /// switching `m_Anim`, so the loop flag is read from the PREVIOUS
    /// anim's data — kept verbatim (benign in practice since idle
    /// anims are looped).
    pub fn set_anim_by_name_no_begin(&mut self, vo: &VoMesh, name: &str) -> bool {
        let Some(idx) = vo.animations.iter().position(|a| a.name == name) else {
            return true;
        };
        self.looped = anim_data_looped(vo, self.anim);
        if self.anim != idx as i32 {
            self.anim = idx as i32;
            self.first_frame(vo);
        }
        false
    }

    /// Port of `SetAnimLooped` (VectorObject.hpp:530).
    pub fn set_anim_looped(&mut self, looped: bool) {
        self.looped = looped;
    }

    /// Port of `IsAnimEnd` (VectorObject.hpp:553).
    pub fn is_anim_end(&self, vo: &VoMesh) -> bool {
        if self.looped {
            return false;
        }
        match vo.animations.get(self.anim as usize) {
            Some(a) => self.frame as usize == a.frames.len().saturating_sub(1),
            None => true,
        }
    }

    /// Port of `Takt(cms)` at VectorObject.cpp:1863-1928. Advances
    /// `time_ms` and walks `frame` forward as long as `time_ms >
    /// time_next_ms`. Returns `true` if `frame` changed so the caller
    /// can refresh a dirty flag. Non-looped anims delay 1000ms before
    /// stalling on the final frame, matching the C++.
    ///
    /// We clamp per-frame duration to min 1ms: the original engine
    /// assumes authored non-zero durations, but parsed data can carry
    /// 0 for default / unset frames, which would infinite-loop the
    /// `while` since `time_next_ms` wouldn't advance. Also caps total
    /// iterations at `2 * fcnt` as a defense-in-depth safeguard.
    pub fn takt(&mut self, vo: &VoMesh, cms: i32) -> bool {
        self.time_ms += cms as i64;
        let old_frame = self.frame;

        let fcnt = vo
            .animations
            .get(self.anim as usize)
            .map(|a| a.frames.len() as i32)
            .unwrap_or(0);
        if fcnt == 0 {
            return false;
        }

        let max_iters = (fcnt as i64) * 2 + 2;
        let mut iters = 0i64;
        while self.time_ms > self.time_next_ms && iters < max_iters {
            iters += 1;
            self.frame += 1;
            if self.looped {
                if self.frame >= fcnt {
                    self.frame = 0;
                }
            } else if self.frame >= fcnt {
                self.time_next_ms += 1000;
                self.frame = fcnt - 1;
                break;
            }
            let (_, t) = anim_frame(vo, self.anim, self.frame).unwrap_or((0, 0));
            // abs(): negative authored durations encode loopedness
            // (GetAnimFrameTime, VectorObject.hpp:389).
            self.time_next_ms += (t as i64).abs().max(1);
        }
        // If we hit the iteration cap (i.e. the data itself is
        // pathological — all-zero frame times or similar), forcibly
        // re-sync `time_next_ms` so we don't immediately re-enter
        // the loop next tick.
        if iters >= max_iters {
            self.time_next_ms = self.time_ms + 16;
        }

        if old_frame != self.frame {
            if let Some((idx, _)) = anim_frame(vo, self.anim, self.frame) {
                self.vo_frame = idx;
            }
            true
        } else {
            false
        }
    }
}

/// Helper: look up `(VoFrameRef.frame_index, time_ms)` for
/// `animations[anim].frames[frame]`. Out-of-range returns None.
fn anim_frame(vo: &VoMesh, anim: i32, frame: i32) -> Option<(usize, i32)> {
    let a = vo.animations.get(anim as usize)?;
    let f = a.frames.get(frame as usize)?;
    Some((f.frame_index, f.time_ms))
}

/// Port of `GetAnimLooped` (VectorObject.hpp:390): the sign of an
/// anim's first authored frame duration encodes loopedness in the
/// data — positive means looped.
fn anim_data_looped(vo: &VoMesh, anim: i32) -> bool {
    anim_frame(vo, anim, 0).map(|(_, t)| t > 0).unwrap_or(false)
}

impl AnimState {
    /// Port of `CVectorObjectAnim::NextFrame` (VectorObject.cpp:1930-
    /// 1945). Advances `frame` by one (wrapping if looped, clamping
    /// otherwise), updates `vo_frame`, and returns the NEW current
    /// frame's duration — the caller uses that to compute how much
    /// to bump `next_anim_time` by.
    pub fn next_frame(&mut self, vo: &VoMesh) -> i32 {
        let fcnt = vo
            .animations
            .get(self.anim as usize)
            .map(|a| a.frames.len() as i32)
            .unwrap_or(0);
        if fcnt == 0 {
            return 0;
        }
        if self.looped {
            self.frame += 1;
            if self.frame >= fcnt {
                self.frame = 0;
            }
        } else if self.frame < fcnt - 1 {
            self.frame += 1;
        } else {
            // Last frame of a non-looped anim — return its
            // duration and don't advance. abs(): the sign encodes
            // loopedness (VectorObject.hpp:389).
            return anim_frame(vo, self.anim, self.frame)
                .map(|(_, t)| t.abs())
                .unwrap_or(0);
        }
        let (idx, t) = anim_frame(vo, self.anim, self.frame).unwrap_or((0, 0));
        self.vo_frame = idx;
        t.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wstr(s: &str) -> Vec<u8> {
        let mut v: Vec<u8> = s.encode_utf16().flat_map(|c| c.to_le_bytes()).collect();
        v.extend_from_slice(&[0, 0]);
        v
    }

    /// Single-array CDataBuf: 12-byte header, data, one 12-byte table entry.
    fn databuf(element_size: u32, data: &[u8], count: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&(12 + data.len() as u32).to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&element_size.to_le_bytes());
        v.extend_from_slice(data);
        v.extend_from_slice(&12u32.to_le_bytes());
        v.extend_from_slice(&count.to_le_bytes());
        v.extend_from_slice(&count.to_le_bytes());
        v
    }

    fn strg(records: &[(&str, &[(&str, Vec<u8>)])]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&0x47525453u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes()); // version 0 = uncompressed
        v.extend_from_slice(&(records.len() as u32).to_le_bytes());
        for (rec, items) in records {
            v.extend_from_slice(&wstr(rec));
            v.extend_from_slice(&(items.len() as u32).to_le_bytes());
            for (item, data) in *items {
                v.extend_from_slice(&wstr(item));
                v.extend_from_slice(&0u32.to_le_bytes());
                v.extend_from_slice(&(data.len() as u32).to_le_bytes());
                v.extend_from_slice(data);
            }
        }
        v
    }

    #[test]
    fn negative_ibase_overrides_tri_start() {
        let verts = vec![0u8; 3 * 32];
        let idx: Vec<u8> = [0u16, 1, 2, 2, 1, 0]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();

        // SVOKadr: bounds/radius zero, UnionStart=0 (@40), UnionCnt=1 (@44).
        let mut kadr = vec![0u8; SVO_KADR_SIZE];
        kadr[44..48].copy_from_slice(&1u32.to_le_bytes());
        let frame_unions = 0u32.to_le_bytes().to_vec();

        // SVOUnion: surface=0, base=0, m_IBase=-3, vercnt=3, tricnt=1,
        // tristart=999 (out of range — must be ignored when m_IBase < 0).
        let mut un = Vec::new();
        for v in [0i32, 0, -3, 3, 1, 999] {
            un.extend_from_slice(&v.to_le_bytes());
        }

        let raw = strg(&[
            ("verts", &[("data", databuf(32, &verts, 3))]),
            ("idxs", &[("data", databuf(2, &idx, 6))]),
            (
                "frames",
                &[
                    ("data", databuf(SVO_KADR_SIZE as u32, &kadr, 1)),
                    ("unions", databuf(4, &frame_unions, 1)),
                ],
            ),
            (
                "unions",
                &[("data", databuf(SVO_UNION_SIZE as u32, &un, 1))],
            ),
        ]);

        let vo = parse_vo(&raw).expect("parse_vo");
        assert_eq!(vo.frames[0].surfaces[0].indices, vec![2, 1, 0]);
    }
}
