//! `.vo` mesh loader — ports VectorObject.cpp:130-386 for static meshes.
//!
//! Each .vo file is a CStorage archive with these relevant buffers:
//!   verts/data  : one array of SVOVertex = pos[3]f32 + normal[3]f32 + uv[2]f32 (32 bytes)
//!   idxs/data   : one array of u16 triangle indices
//!   surfs/texs  : wide-char texture reference (".\path\name")
//!
//! Animations (anims/*, frames/*, matrices/*) are ignored; we use frame 0 only.

use anyhow::{Context, Result};

use crate::assets::storage::Storage;

pub struct VoMesh {
    pub vertices: Vec<VoVertex>,
    pub indices: Vec<u16>,
    /// Texture reference as it appears in the file — typically ".\Matrix\Obj\<dir>\<name>"
    /// (no extension, backslashes). Caller resolves to pkg path.
    pub texture_ref: Option<String>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VoVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

pub fn parse_vo(data: &[u8]) -> Result<VoMesh> {
    let stor = Storage::from_bytes(data).context("parsing VO as CStorage")?;

    let verts_buf = stor.get_buf("verts", "data").context("missing verts/data")?;
    let idxs_buf = stor.get_buf("idxs", "data").context("missing idxs/data")?;
    if verts_buf.arrays_count() == 0 || idxs_buf.arrays_count() == 0 {
        anyhow::bail!("VO has empty verts or idxs");
    }

    let vb = verts_buf.get_bytes(0);
    let vcount = vb.len() / 32;
    let mut vertices = Vec::with_capacity(vcount);
    for i in 0..vcount {
        let off = i * 32;
        let f = |o: usize| f32::from_le_bytes([vb[off + o], vb[off + o + 1], vb[off + o + 2], vb[off + o + 3]]);
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

    let texture_ref = stor.get_buf("surfs", "texs").and_then(|b| {
        if b.arrays_count() > 0 {
            let s = b.get_as_wstr(0);
            let s = s.trim_end_matches('\0').trim().to_string();
            if s.is_empty() { None } else { Some(s) }
        } else { None }
    });

    Ok(VoMesh { vertices, indices, texture_ref })
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
    pub texture_path: Option<String>,
}

pub fn resolve_paths(id_string: &str) -> Option<ResolvedObjectPaths> {
    let parts: Vec<&str> = id_string.split('*').collect();
    if parts.len() < 2 { return None; }
    let path = parts[0].replace('\\', "/");
    let vo_name = parts[1];
    if vo_name.is_empty() { return None; }
    let vo_path = format!("{}{}.vo", path, vo_name);

    let tex_path = parts.get(2).and_then(|t| {
        let name = t.split('?').next().unwrap_or("");
        if name.is_empty() { None } else { Some(format!("{}{}", path, name)) }
    });

    Some(ResolvedObjectPaths { vo_path, texture_path: tex_path })
}
