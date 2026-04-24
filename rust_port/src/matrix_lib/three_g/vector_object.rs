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

pub use crate::matrix_lib::three_g::texture::{
    has_trans_suffix, merge_materials, parse_material_spec, parse_material_spec_with_prefix,
    resolve_alpha_test_with_txt, MaterialSpec,
};
use crate::matrix_lib::three_g::texture::parse_scroll;

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

    Ok(VoMesh {
        vertices,
        surfaces,
        frames,
        animations,
        matrices,
        all_matrices,
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
            let tri_cnt = read_i32(unions_data, union_off + 16).max(0) as usize;
            let tri_start = read_i32(unions_data, union_off + 20).max(0) as usize;

            let dst_idx = surface.min(surfs.len().saturating_sub(1));
            let dst = &mut surfs[dst_idx];
            let start = tri_start * 3;
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
/// Indexed by `ChassisKind as usize`. `None` for kinds whose VO
/// failed to load.
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
        // Clamp to min 1ms — see takt() for why.
        self.time_next_ms = self.time_ms + (time as i64).max(1);
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
            self.time_next_ms += (t as i64).max(1);
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
            // duration and don't advance.
            return anim_frame(vo, self.anim, self.frame)
                .map(|(_, t)| t)
                .unwrap_or(0);
        }
        let (idx, t) = anim_frame(vo, self.anim, self.frame).unwrap_or((0, 0));
        self.vo_frame = idx;
        t
    }
}
