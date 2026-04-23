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

#[derive(Clone, Debug, Default)]
pub struct MaterialSpec {
    pub diffuse: Option<String>,
    pub gloss: Option<String>,
    pub back: Option<String>,
    pub mask: Option<String>,
    pub scroll: [f32; 2],
    /// True when the diffuse texture's `ParseFlags` call would set
    /// `TF_ALPHATEST` (Texture.cpp:96-152). We set it eagerly from the
    /// `?Trans` suffix (Texture.cpp:108); the sibling `.txt` `AlphaTest`
    /// override (Texture.cpp:131-136) is applied later via
    /// `resolve_alpha_test_with_txt`, which needs asset I/O.
    pub alpha_test: bool,
}

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
    let surfaces = frames
        .first()
        .map(|f| f.surfaces.clone())
        .unwrap_or_default();

    Ok(VoMesh {
        vertices,
        surfaces,
        frames,
        animations,
    })
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

/// Resolve an object Id string (from `strings/String`, '*'-delimited) into a
/// VO file path and texture path. Returns `None` if empty.
///
/// Id string layout (MatrixObject.cpp:429-472, Common.hpp:176-191):
///   [0] OTP_PATH (e.g. `Matrix\Obj\palm\`)
///   [1] OTP_VO   (e.g. `palm00`)
///   [2] OTP_TEXTURE (e.g. `palm00?Trans`)
///   ... more fields
pub struct ResolvedObjectPaths {
    pub vo_path: String,
    pub material: MaterialSpec,
    pub shadow: ShadowSpec,
}

#[derive(Clone, Debug)]
pub struct ShadowSpec {
    pub kind: ShadowKind,
    pub texture_size: u32,
    pub cache_tag: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShadowKind {
    None,
    Stencil,
    ProjectedStatic,
    ProjectedDynamic,
}

pub fn resolve_paths(id_string: &str) -> Option<ResolvedObjectPaths> {
    let parts: Vec<&str> = id_string.split('*').collect();
    if parts.len() < 2 {
        return None;
    }
    let path = parts[0].replace('\\', "/");
    let vo_name = parts[1];
    if vo_name.is_empty() {
        return None;
    }
    let vo_path = format!("{}{}.vo", path, vo_name);

    let material = MaterialSpec {
        diffuse: parts
            .get(2)
            .and_then(|t| resolve_texture_name(t, Some(&path))),
        gloss: parts
            .get(3)
            .and_then(|t| resolve_texture_name(t, Some(&path))),
        back: parts
            .get(4)
            .and_then(|t| resolve_texture_name(t, Some(&path))),
        mask: parts
            .get(5)
            .and_then(|t| resolve_texture_name(t, Some(&path))),
        scroll: parts.get(6).map(|t| parse_scroll(t)).unwrap_or([0.0, 0.0]),
        alpha_test: parts.get(2).is_some_and(|t| has_trans_suffix(t)),
    };
    let shadow = parts
        .get(7)
        .map(|t| parse_shadow_spec(t))
        .unwrap_or(ShadowSpec {
            kind: ShadowKind::None,
            texture_size: 128,
            cache_tag: None,
        });

    Some(ResolvedObjectPaths {
        vo_path,
        material,
        shadow,
    })
}

pub fn parse_material_spec(spec: &str) -> MaterialSpec {
    parse_material_spec_with_prefix(spec, None)
}

pub fn parse_material_spec_with_prefix(spec: &str, prefix: Option<&str>) -> MaterialSpec {
    let parts: Vec<&str> = spec.split('*').collect();
    MaterialSpec {
        diffuse: parts.first().and_then(|t| resolve_texture_name(t, prefix)),
        gloss: parts.get(1).and_then(|t| resolve_texture_name(t, prefix)),
        back: parts.get(2).and_then(|t| resolve_texture_name(t, prefix)),
        mask: parts.get(3).and_then(|t| resolve_texture_name(t, prefix)),
        scroll: parts.get(4).map(|t| parse_scroll(t)).unwrap_or([0.0, 0.0]),
        alpha_test: parts.first().is_some_and(|t| has_trans_suffix(t)),
    }
}

pub fn merge_materials(base: &MaterialSpec, overlay: Option<&MaterialSpec>) -> MaterialSpec {
    let Some(overlay) = overlay else {
        return base.clone();
    };
    // alpha_test tracks the diffuse that actually wins the merge.
    let (diffuse, alpha_test) = if overlay.diffuse.is_some() {
        (overlay.diffuse.clone(), overlay.alpha_test)
    } else {
        (base.diffuse.clone(), base.alpha_test)
    };
    MaterialSpec {
        diffuse,
        gloss: overlay.gloss.clone().or_else(|| base.gloss.clone()),
        back: overlay.back.clone().or_else(|| base.back.clone()),
        mask: overlay.mask.clone().or_else(|| base.mask.clone()),
        scroll: if overlay.scroll != [0.0, 0.0] {
            overlay.scroll
        } else {
            base.scroll
        },
        alpha_test,
    }
}

/// Ports the `?Trans` flag detection inside `CBaseTexture::ParseFlags`
/// (Texture.cpp:102-111). The raw texture spec is `path?Opt1?Opt2?...`; any
/// `Trans` option sets `TF_ALPHATEST`. Other options (`Alpha`, `Compressed`)
/// don't affect alpha-test.
///
/// The comparison in the original is `tstr == L"Trans"` via `CWStr::Equal`
/// (CWStr.cpp:678-684), which is a memcmp — i.e. case-sensitive. We mirror
/// that: `?trans` (lowercase) would NOT set the flag in the real engine, so
/// it mustn't here either.
fn has_trans_suffix(raw: &str) -> bool {
    let raw = raw.trim().trim_end_matches('\0');
    let mut parts = raw.split('?');
    parts.next(); // the actual path — skip
    parts.any(|opt| opt.trim() == "Trans")
}

/// Ports the sibling `.txt` override pass of `CBaseTexture::ParseFlags`
/// (Texture.cpp:113-136). The original reads `<basename>.txt` next to each
/// texture (via `CacheReplaceFileExt`) and, if present, applies
/// `AlphaTest = 0` / `AlphaTest = <anything else>` to the flag chosen from
/// the `?Trans` suffix.
///
/// The original explicitly skips `pinguin.txt` and `robotarget.txt`
/// (Texture.cpp:121-124): those files carry unrelated block content that
/// would spuriously set the alpha-test flag. We mirror that guard.
///
/// `read_file` is the port's asset-lookup callback (the same one that
/// resolves textures / VO files). It can return `None` when the sibling
/// `.txt` doesn't exist, which is the common case.
pub fn resolve_alpha_test_with_txt(
    diffuse_path: &str,
    suffix_flag: bool,
    read_file: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> bool {
    let txt_path = replace_extension(diffuse_path, "txt");
    // Substring match to mirror the original's `tstr.Find(L"pinguin.txt") >= 0`
    // guard (Texture.cpp:121-124). `CWStr::Find` (CWStr.cpp:489-506) is a
    // case-sensitive byte scan, so match the casing of the excluded files
    // verbatim.
    if txt_path.contains("pinguin.txt") || txt_path.contains("robotarget.txt") {
        return suffix_flag;
    }
    let Some(bytes) = read_file(&txt_path) else {
        return suffix_flag;
    };
    let bp = crate::matrix_lib::base::blockpar::BlockPar::parse_bytes(&bytes);
    // The C++ branch gates on `!tstr.IsEmpty()` (Texture.cpp:132), so both an
    // absent key and an empty value leave the suffix flag alone. Only a
    // non-empty value overrides: "0" clears, anything else sets.
    match bp.par_get_ne("AlphaTest") {
        Some(v) => {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                suffix_flag
            } else {
                trimmed != "0"
            }
        }
        None => suffix_flag,
    }
}

fn replace_extension(path: &str, new_ext: &str) -> String {
    match path.rsplit_once('.') {
        Some((stem, _)) => format!("{stem}.{new_ext}"),
        None => format!("{path}.{new_ext}"),
    }
}

fn resolve_texture_name(raw: &str, prefix: Option<&str>) -> Option<String> {
    let name = raw.trim().trim_end_matches('\0');
    if name.is_empty() || name == "." {
        return None;
    }
    let name = name.split('?').next().unwrap_or("").trim();
    if name.is_empty() {
        return None;
    }
    let normalized = name.trim_start_matches(".\\").replace('\\', "/");
    match prefix {
        Some(prefix) if !normalized.contains('/') => Some(format!("{prefix}{normalized}")),
        _ => Some(normalized),
    }
}

fn parse_scroll(raw: &str) -> [f32; 2] {
    let mut it = raw.split(',');
    let u = it
        .next()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(0.0);
    let v = it
        .next()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(0.0);
    [u, v]
}

fn parse_shadow_spec(raw: &str) -> ShadowSpec {
    let spec = raw.trim();
    if spec.is_empty() {
        return ShadowSpec {
            kind: ShadowKind::None,
            texture_size: 128,
            cache_tag: None,
        };
    }
    let mut parts = spec.split(',');
    let kind = match parts.next().unwrap_or("").trim() {
        "Stencil" => ShadowKind::Stencil,
        "Proj" => ShadowKind::ProjectedStatic,
        "ProjEx" => ShadowKind::ProjectedDynamic,
        _ => ShadowKind::None,
    };
    let texture_size = parts
        .next()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(128);
    let cache_tag = parts.next().and_then(|v| v.trim().parse::<u32>().ok());
    ShadowSpec {
        kind,
        texture_size,
        cache_tag,
    }
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
