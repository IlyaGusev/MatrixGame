//! Terrain surface overlays — ports CTerSurface::LoadM + TerSurfMW / TerSurfGlossMW
//! (MatrixTerSurface.cpp + MatrixRenderPipeline.cpp).

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::terrain::{DrawBatch, Vertex};
use crate::matrix_lib::base::storage::Storage;
use crate::matrix_game::common::{rd_f32, rd_i32, rd_u16, rd_u32};
use crate::matrix_game::map::GameMap;
use crate::matrix_lib::three_g::texture::{
    create_solid_texture, create_texture_from_rgba_mipped, decode_texture_bytes,
};

/// Per-vertex data for the gloss pass. Carries view-space normal inputs and
/// the atlas UV so the fragment shader can weight the additive contribution by
/// the base atlas alpha (matching the original single-pass gloss pipeline).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GlossVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl GlossVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub struct GlossBatch {
    pub bind_group: wgpu::BindGroup,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

/// A surface draw batch plus the list of `CMatrixMapGroup` indices the
/// original attached it to via `g->AddSurface(this)`
/// (MatrixTerSurface.cpp:187-193). Empty `groups` means the surface had
/// no `grpsc` list and should be drawn unconditionally.
pub struct OverlayBatch {
    pub draw: DrawBatch,
    pub groups: Vec<u32>,
}

pub struct GlossOverlayBatch {
    pub draw: GlossBatch,
    pub groups: Vec<u32>,
}

pub struct SurfaceOverlays {
    pub base: Vec<OverlayBatch>,
    pub gloss: Vec<GlossOverlayBatch>,
}

pub struct GlossResources<'a> {
    pub bgl: &'a wgpu::BindGroupLayout,
    pub uniform_buffer: &'a wgpu::Buffer,
    pub reflection_view: &'a wgpu::TextureView,
    pub reflection_sampler: &'a wgpu::Sampler,
}

#[allow(clippy::too_many_arguments)]
pub fn build_surface_overlays(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    uniform_buffer: &wgpu::Buffer,
    bgl: &wgpu::BindGroupLayout,
    sampler_clamp: &wgpu::Sampler,
    sampler_wrap_v: &wgpu::Sampler,
    macro_view: &wgpu::TextureView,
    macro_sampler: &wgpu::Sampler,
    stor: &Storage,
    map: &GameMap,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    gloss: Option<GlossResources<'_>>,
) -> SurfaceOverlays {
    // The original processes both buffers: `surfacesM` entries via
    // `CTerSurface::LoadM` (32-byte vertices with macro UVs) and `surfaces`
    // entries via `CTerSurface::Load` (24-byte vertices, no macro UVs)
    // (MatrixMapPrepare.cpp:1544-1579). Maps typically only populate one of
    // the two — atoll uses `surfacesM`, training uses plain `surfaces`.
    // Collect `(buffer, has_macro_uv)` pairs and feed them through the same
    // assembly loop; macro UVs default to (0, 0) when the vertex doesn't
    // carry them, matching the C++'s SURFF_MACRO-less render path.
    let srfm = stor
        .get_buf("surfacesM", "Data")
        .filter(|b| b.arrays_count() > 0);
    let srf = stor
        .get_buf("surfaces", "Data")
        .filter(|b| b.arrays_count() > 0);
    if srfm.is_none() && srf.is_none() {
        return SurfaceOverlays {
            base: vec![],
            gloss: vec![],
        };
    }
    let sources: Vec<(&crate::matrix_lib::base::storage::DataBuf, bool)> = srfm
        .map(|b| (b, true))
        .into_iter()
        .chain(srf.map(|b| (b, false)))
        .collect();
    let Some(strings) = stor.get_buf("strings", "String") else {
        return SurfaceOverlays {
            base: vec![],
            gloss: vec![],
        };
    };

    let cx = map.world_width() * 0.5;
    let cy = map.world_height() * 0.5;

    let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
    let mut gloss_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
    let white = create_solid_texture(device, queue, [255, 255, 255, 255]);

    struct SurfData {
        index: i32,
        tex_path: String,
        gloss_path: Option<String>,
        wrap_y: bool,
        verts: Vec<Vertex>,
        indices: Vec<u16>,
        gloss_verts: Vec<GlossVertex>,
        groups: Vec<u32>,
    }
    let mut surfaces: Vec<SurfData> = Vec::new();

    for (srfm, has_macro_uv) in &sources {
        let vert_size = if *has_macro_uv { 32 } else { 24 };
        for i in 0..srfm.arrays_count() {
        let raw = srfm.get_bytes(i);
        if raw.len() < 32 {
            continue;
        }
        let mut off = 0;

        let ids = rd_i32(raw, &mut off);
        let index = rd_i32(raw, &mut off);
        let color_dw = rd_u32(raw, &mut off);
        let vcnt = rd_u32(raw, &mut off) as usize;
        let idxsz = rd_u32(raw, &mut off) as usize;
        let grpsc = rd_u32(raw, &mut off) as usize;
        let disp_x = rd_f32(raw, &mut off);
        let disp_y = rd_f32(raw, &mut off);

        let ids_str = if ids >= 0 && (ids as usize) < strings.arrays_count() {
            strings.get_as_wstr(ids as usize)
        } else {
            continue;
        };

        let (tex_path, params) = ids_str
            .split_once('?')
            .map(|(t, p)| (t.to_string(), p.to_string()))
            .unwrap_or((ids_str.clone(), String::new()));

        let tex_path = tex_path.replace('\\', "/");

        let gloss_path = gloss.as_ref().and_then(|_| {
            parse_gloss_param(&params).and_then(|name| resolve_gloss_path(&tex_path, &name))
        });

        let r = ((color_dw >> 16) & 0xFF) as f32 / 255.0;
        let g = ((color_dw >> 8) & 0xFF) as f32 / 255.0;
        let b = (color_dw & 0xFF) as f32 / 255.0;
        let a = ((color_dw >> 24) & 0xFF) as f32 / 255.0;

        let needed = off + vcnt * vert_size + idxsz + grpsc * 4;
        if needed > raw.len() {
            continue;
        }

        let mut verts = Vec::with_capacity(vcnt);
        let mut gloss_verts = if gloss_path.is_some() {
            Vec::with_capacity(vcnt)
        } else {
            Vec::new()
        };
        let mut wrap_y = false;
        for _ in 0..vcnt {
            let px = rd_f32(raw, &mut off);
            let py = rd_f32(raw, &mut off);
            let pz = rd_f32(raw, &mut off);
            let vcol = rd_u32(raw, &mut off);
            let tu = rd_f32(raw, &mut off);
            let tv = rd_f32(raw, &mut off);
            // Plain (non-M) surfaces don't carry macro UVs in their vertex
            // stream — `CTerSurface::Load` uses `STerSurfVertex` (24 bytes)
            // versus `STerSurfVertexM` (32 bytes) in `LoadM`. Skip reading
            // the tum/tvm pair and emit (0, 0) instead; the draw path's
            // macrotexture is disabled on maps that use plain surfaces
            // (training has MacroTexture=""), so the value is irrelevant.
            let (_tum, _tvm) = if *has_macro_uv {
                (rd_f32(raw, &mut off), rd_f32(raw, &mut off))
            } else {
                (0.0, 0.0)
            };

            let vr = ((vcol >> 16) & 0xFF) as f32 / 255.0;
            let vg = ((vcol >> 8) & 0xFF) as f32 / 255.0;
            let vb = (vcol & 0xFF) as f32 / 255.0;
            let va = ((vcol >> 24) & 0xFF) as f32 / 255.0;
            if !(0.0..=1.0).contains(&tv) {
                wrap_y = true;
            }

            let world_x = px + disp_x - cx;
            let world_y = py + disp_y - cy;
            let world_z = pz + 0.05;

            verts.push(Vertex {
                position: [world_x, world_y, world_z],
                color: [vr * r, vg * g, vb * b, va * a],
                uv: [tu, tv],
                macro_uv: [_tum, _tvm],
            });

            if gloss_path.is_some() {
                let n = map.get_normal(px + disp_x, py + disp_y, false);
                gloss_verts.push(GlossVertex {
                    position: [world_x, world_y, world_z],
                    normal: n,
                    uv: [tu, tv],
                });
            }
        }

        let idx_count = idxsz / 2;
        let mut strip = Vec::with_capacity(idx_count);
        for _ in 0..idx_count {
            if off + 2 > raw.len() {
                break;
            }
            strip.push(rd_u16(raw, &mut off));
        }

        // Trailing `grpsc` DWORDs: indices into the linear group array
        // (`m_Group[idx]`, MatrixTerSurface.cpp:187-193 / 274-280). Each
        // index matches `gy * group_w + gx` on our side — same ordering
        // as `visible_groups_mask`.
        let mut groups = Vec::with_capacity(grpsc);
        for _ in 0..grpsc {
            if off + 4 > raw.len() {
                break;
            }
            groups.push(rd_u32(raw, &mut off));
        }

        if !tex_cache.contains_key(&tex_path) {
            if let Some(data) = read_texture(&tex_path) {
                if let Some(rgba) = decode_texture_bytes(&data) {
                    tex_cache.insert(
                        tex_path.clone(),
                        create_texture_from_rgba_mipped(device, queue, &rgba, 6),
                    );
                }
            }
        }

        if let Some(gp) = gloss_path.as_ref() {
            if !gloss_cache.contains_key(gp) {
                if let Some(data) = read_texture(gp) {
                    if let Some(rgba) = decode_texture_bytes(&data) {
                        gloss_cache.insert(
                            gp.clone(),
                            create_texture_from_rgba_mipped(device, queue, &rgba, 6),
                        );
                    }
                }
            }
        }

        surfaces.push(SurfData {
            index,
            tex_path,
            gloss_path,
            wrap_y,
            verts,
            indices: strip,
            gloss_verts,
            groups,
        });
        }
    }

    surfaces.sort_by_key(|s| s.index);

    let mut base = Vec::new();
    let mut gloss_batches = Vec::new();
    let mut overlay_tris = 0u32;
    let mut gloss_tris = 0u32;
    let mut skipped_gloss = 0usize;

    for surf in &surfaces {
        if surf.indices.len() < 3 {
            continue;
        }

        let tex_view = tex_cache.get(surf.tex_path.as_str()).unwrap_or(&white);
        let sampler = if surf.wrap_y {
            sampler_wrap_v
        } else {
            sampler_clamp
        };

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&surf.verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&surf.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(macro_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(macro_sampler),
                },
            ],
        });

        overlay_tris += surf.indices.len().saturating_sub(2) as u32;
        base.push(OverlayBatch {
            draw: DrawBatch {
                bind_group: bg,
                vertex_buffer: vb,
                index_buffer: ib,
                num_indices: surf.indices.len() as u32,
                index_format: wgpu::IndexFormat::Uint16,
                cpu_vertices: None,
                point_coords: None,
            },
            groups: surf.groups.clone(),
        });

        // Gloss pass: adds gloss * reflection weighted by atlas alpha.
        if let (Some(gloss_res), Some(gloss_path)) = (gloss.as_ref(), surf.gloss_path.as_ref()) {
            let Some(gloss_view) = gloss_cache.get(gloss_path.as_str()) else {
                skipped_gloss += 1;
                continue;
            };
            let g_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&surf.gloss_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let g_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&surf.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let g_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: gloss_res.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gloss_res.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(tex_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(gloss_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(gloss_res.reflection_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(gloss_res.reflection_sampler),
                    },
                ],
            });
            gloss_tris += surf.indices.len().saturating_sub(2) as u32;
            gloss_batches.push(GlossOverlayBatch {
                draw: GlossBatch {
                    bind_group: g_bg,
                    vertex_buffer: g_vb,
                    index_buffer: g_ib,
                    num_indices: surf.indices.len() as u32,
                },
                groups: surf.groups.clone(),
            });
        }
    }

    log::info!(
        "terrain overlays: {} base ({} tris), {} gloss ({} tris, {} skipped), {} base tex, {} gloss tex",
        base.len(),
        overlay_tris,
        gloss_batches.len(),
        gloss_tris,
        skipped_gloss,
        tex_cache.len(),
        gloss_cache.len()
    );
    SurfaceOverlays {
        base,
        gloss: gloss_batches,
    }
}

/// Parse `gloss=name` from a surface-id parameter string like `gloss=floor05_gloss`.
/// Returns `None` when the gloss parameter is absent or empty (matching the
/// original's `parv.IsEmpty()` skip — only `surfaces` / `Load()` guards against
/// empty names; `LoadM` accepts empty names, so we do the same via `Some("")`
/// handling below at the call site).
fn parse_gloss_param(params: &str) -> Option<String> {
    for chunk in params.split('?') {
        let (k, v) = chunk.split_once('=')?;
        if k.eq_ignore_ascii_case("gloss") && !v.is_empty() {
            return Some(v.to_string());
        }
    }
    None
}

/// Replace the filename of `tex_path` with `name` (matching
/// `CacheReplaceFileNameAndExt`). The extension is left off so the texture
/// reader can try `.DDS` / `.PNG` in turn.
fn resolve_gloss_path(tex_path: &str, name: &str) -> Option<String> {
    let slash = tex_path.rfind('/');
    let dir = slash.map(|i| &tex_path[..i]).unwrap_or("");
    if dir.is_empty() {
        Some(name.to_string())
    } else {
        Some(format!("{}/{}", dir, name))
    }
}
