//! `.vo` mesh loader — ports VectorObject.cpp:130-386 for static meshes.
//!
//! Each .vo file is a CStorage archive with these relevant buffers:
//!   verts/data    : one array of SVOVertex = pos[3]f32 + normal[3]f32 + uv[2]f32 (32 bytes)
//!   idxs/data     : one array of u16 triangle indices
//!   surfs/texs    : wide-char per-surface texture references
//!   unions/data   : frame-independent surface/triangle ranges
//!   frames/data   : frame metadata, including the union range for frame 0
//!
//! Animations (anims/*, frames/*, matrices/*) are ignored; we use frame 0 only.

use anyhow::{Context, Result};

use crate::assets::storage::Storage;

pub struct VoMesh {
    pub vertices: Vec<VoVertex>,
    pub surfaces: Vec<VoSurfaceMesh>,
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

    let surfaces = parse_surfaces(&stor, &indices);

    Ok(VoMesh { vertices, surfaces })
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
    }
}

pub fn merge_materials(base: &MaterialSpec, overlay: Option<&MaterialSpec>) -> MaterialSpec {
    let Some(overlay) = overlay else {
        return base.clone();
    };
    MaterialSpec {
        diffuse: overlay.diffuse.clone().or_else(|| base.diffuse.clone()),
        gloss: overlay.gloss.clone().or_else(|| base.gloss.clone()),
        back: overlay.back.clone().or_else(|| base.back.clone()),
        mask: overlay.mask.clone().or_else(|| base.mask.clone()),
        scroll: if overlay.scroll != [0.0, 0.0] {
            overlay.scroll
        } else {
            base.scroll
        },
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
    if normalized.contains('/') || prefix.is_none() {
        Some(normalized)
    } else {
        Some(format!("{}{}", prefix.unwrap(), normalized))
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

fn parse_surfaces(stor: &Storage, indices: &[u16]) -> Vec<VoSurfaceMesh> {
    let surfaces_buf = stor.get_buf("surfs", "texs");
    let frame_buf = stor.get_buf("frames", "data");
    let frame_unions_buf = stor.get_buf("frames", "unions");
    let unions_buf = stor.get_buf("unions", "data");

    let mut texture_refs = Vec::new();
    if let Some(buf) = surfaces_buf {
        for i in 0..buf.arrays_count() {
            let s = buf.get_as_wstr(i);
            let s = s.trim_end_matches('\0').trim().to_string();
            texture_refs.push(if s.is_empty() { None } else { Some(s) });
        }
    }

    let Some(frame_data) = frame_buf.map(|b| b.get_bytes(0)) else {
        return vec![VoSurfaceMesh {
            indices: indices.iter().map(|&i| i as u32).collect(),
            texture_ref: texture_refs.into_iter().next().flatten(),
        }];
    };
    let Some(frame_unions) = frame_unions_buf.map(|b| b.get_bytes(0)) else {
        return vec![VoSurfaceMesh {
            indices: indices.iter().map(|&i| i as u32).collect(),
            texture_ref: texture_refs.into_iter().next().flatten(),
        }];
    };
    let Some(unions_data) = unions_buf.map(|b| b.get_bytes(0)) else {
        return vec![VoSurfaceMesh {
            indices: indices.iter().map(|&i| i as u32).collect(),
            texture_ref: texture_refs.into_iter().next().flatten(),
        }];
    };
    if frame_data.len() < 48 || frame_unions.len() < 4 || unions_data.len() < 24 {
        return vec![VoSurfaceMesh {
            indices: indices.iter().map(|&i| i as u32).collect(),
            texture_ref: texture_refs.into_iter().next().flatten(),
        }];
    }

    let union_start = read_i32(frame_data, 40).max(0) as usize;
    let union_count = read_i32(frame_data, 44).max(0) as usize;
    let surface_count = texture_refs.len().max(1);
    let mut out: Vec<VoSurfaceMesh> = (0..surface_count)
        .map(|i| VoSurfaceMesh {
            indices: Vec::new(),
            texture_ref: texture_refs.get(i).cloned().unwrap_or(None),
        })
        .collect();

    for i in 0..union_count {
        let union_index_off = (union_start + i) * 4;
        if union_index_off + 4 > frame_unions.len() {
            break;
        }
        let union_index = u32::from_le_bytes([
            frame_unions[union_index_off],
            frame_unions[union_index_off + 1],
            frame_unions[union_index_off + 2],
            frame_unions[union_index_off + 3],
        ]) as usize;
        let union_off = union_index * 24;
        if union_off + 24 > unions_data.len() {
            continue;
        }

        let surface = read_i32(unions_data, union_off).max(0) as usize;
        let base = read_i32(unions_data, union_off + 4);
        let tri_cnt = read_i32(unions_data, union_off + 16).max(0) as usize;
        let tri_start = read_i32(unions_data, union_off + 20).max(0) as usize;
        let dst_index = if surface < out.len() { surface } else { 0 };
        let dst = &mut out[dst_index];
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

    if out.iter().all(|s| s.indices.is_empty()) {
        return vec![VoSurfaceMesh {
            indices: indices.iter().map(|&i| i as u32).collect(),
            texture_ref: texture_refs.into_iter().next().flatten(),
        }];
    }

    out.into_iter().filter(|s| !s.indices.is_empty()).collect()
}

fn read_i32(bytes: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}
