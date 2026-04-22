//! Port of MatrixMapGroup.cpp — per-group terrain geometry build code.
//!
//! **Intentional divergence from the original.** The original C++ keeps
//! one `CMatrixMapGroup` instance per map tile, each owning a slice of
//! `BigVB_bottom` / `BigIB_bottom` and emitting its own draw call. The
//! Rust port instead walks every group at init and merges triangles
//! across groups into one vertex/index buffer per texture, so rendering
//! becomes one draw per texture globally — much friendlier to wgpu /
//! WebGL. No `CMatrixMapGroup` struct exists at runtime here.
//!
//! This file holds the build code (BuildBottom-equivalent) plus the
//! shared bottom vertex format. The call site that iterates groups and
//! submits draws lives in `matrix_game/map.rs` on the `MapRenderer`
//! struct.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::matrix_lib::base::storage::Storage;
use crate::matrix_game::common::{rd_u16, rd_u32, CELLFLAG_BRIDGE, CELLFLAG_DOWN, MAP_GROUP_SIZE};
use crate::matrix_game::map::{GameMap, GLOBAL_SCALE};

/// Bottom vertex — ports `SMatrixMapVertexBottom` from MatrixMapGroup.cpp.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub macro_uv: [f32; 2],
}

impl Vertex {
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
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// A single bottom-landscape draw call. Ports one `g->m_PreparedVB` entry in
/// MatrixMapGroup.cpp; `cpu_vertices`/`point_coords` are kept around so
/// point-light tinting can patch vertex colors without rebuilding the mesh.
pub struct DrawBatch {
    pub bind_group: wgpu::BindGroup,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub index_format: wgpu::IndexFormat,
    pub cpu_vertices: Option<Vec<Vertex>>,
    pub point_coords: Option<Vec<(usize, usize)>>,
}

/// CPU-side output of the BuildBottom pass, keyed by atlas texture index.
/// Each entry holds the merged vertex buffer, index buffer (indices rebased
/// onto the merged vertex space), and per-vertex `(px, py)` back-references
/// used later to re-tint vertex colors when point-light state changes.
pub type BottomCpuBatches = HashMap<u32, (Vec<Vertex>, Vec<u32>, Vec<(usize, usize)>)>;

/// Port of CMatrixMapGroup::BuildBottom — scans every group in `groups/Data`,
/// decodes its SCompileBottomVert vertices and index streams, and merges
/// triangles across all groups that share a texture into a single per-texture
/// batch (see the module-level divergence note). The result is CPU-side only;
/// the caller uploads each entry into GPU buffers and builds bind groups.
pub fn build_bottom_cpu_batches(
    stor: &Storage,
    map: &GameMap,
    ts_inv: f64,
) -> BottomCpuBatches {
    let groups_buf = stor.get_buf("groups", "Data");
    let cx = map.world_width() * 0.5;
    let cy = map.world_height() * 0.5;

    let mut batches_by_tex: BottomCpuBatches = HashMap::new();

    let Some(grp) = groups_buf else {
        return batches_by_tex;
    };

    for gi in 0..grp.arrays_count() {
        let raw = grp.get_bytes(gi);
        if raw.len() < 4 {
            continue;
        }
        let mut off = 0;

        let gx = rd_u16(raw, &mut off) as i32;
        let gy = rd_u16(raw, &mut off) as i32;
        let _w = MAP_GROUP_SIZE.min(map.size_x as i32 - gx);
        let _h = MAP_GROUP_SIZE.min(map.size_y as i32 - gy);

        let geom_count = rd_u32(raw, &mut off);

        struct BottomGeom {
            texture: u32,
            idx_offset: usize,
            idx_count: usize,
        }
        let mut geoms = Vec::new();
        let mut all_indices: Vec<u16> = Vec::new();

        for _ in 0..geom_count {
            if off + 8 > raw.len() {
                break;
            }
            let texture = rd_u32(raw, &mut off);
            let idx_bytes = rd_u32(raw, &mut off) as usize;
            let idx_start = all_indices.len();
            let idx_count = idx_bytes / 2;
            for _ in 0..idx_count {
                if off + 2 > raw.len() {
                    break;
                }
                all_indices.push(rd_u16(raw, &mut off));
            }
            geoms.push(BottomGeom {
                texture,
                idx_offset: idx_start,
                idx_count,
            });
        }

        if off + 4 > raw.len() {
            continue;
        }
        let vert_bytes = rd_u32(raw, &mut off) as usize;
        let vert_count = vert_bytes / 8;

        let macro_step = 1.0 / map.macro_texture_size as f32;

        let mut vertices = Vec::with_capacity(vert_count);
        let mut point_coords = Vec::with_capacity(vert_count);
        for _ in 0..vert_count {
            if off + 8 > raw.len() {
                break;
            }
            let vx = rd_u16(raw, &mut off) as i32;
            let vy = rd_u16(raw, &mut off) as i32;
            let tx = rd_u16(raw, &mut off);
            let ty = rd_u16(raw, &mut off);

            let pt = map.point(vx as usize, vy as usize);
            let world_x = vx as f32 * GLOBAL_SCALE - cx;
            let world_y = vy as f32 * GLOBAL_SCALE - cy;

            let r = pt.r as f32 / 255.0;
            let g = pt.g as f32 / 255.0;
            let b = pt.b as f32 / 255.0;

            // Half-texel offset — intentional divergence from
            // MatrixMapGroup.cpp:341-342, which samples the corner
            // (`ts_inv * v->tx`). The original atlas ships as DXT1, so each
            // 4×4 block is already averaged and bilinear sampling at the
            // corner of a tile smudges across the DXT boundary without
            // visible seams. Our atlas is uploaded as uncompressed
            // Rgba8UnormSrgb (see `texture::create_texture_from_rgba_mipped`
            // — DXT1 encoding is not done), so the `+0.5` centers the sample
            // inside the texel and stops bilinear from reaching into the
            // neighbouring tile. Remove only if/when the atlas is switched
            // to a BC format that gives the same natural blur.
            let u = ts_inv * (tx as f64 + 0.5);
            let v = ts_inv * (ty as f64 + 0.5);
            let mu = macro_step * vx as f32;
            let mv = macro_step * vy as f32;

            // Down-cell handling (MatrixMapGroup.cpp:314-335)
            let mut pos_x = world_x;
            let mut pos_y = world_y;
            let mut pos_z = pt.z;

            let vxi = vx as usize;
            let vyi = vy as usize;
            let down = vxi >= 1
                && vyi >= 1
                && vxi < map.size_x
                && vyi < map.size_y
                && (map.point(vxi, vyi).flags & CELLFLAG_DOWN) != 0
                && (map.point(vxi - 1, vyi).flags & CELLFLAG_DOWN) != 0
                && (map.point(vxi - 1, vyi - 1).flags & CELLFLAG_DOWN) != 0
                && (map.point(vxi, vyi - 1).flags & CELLFLAG_DOWN) != 0;

            if down {
                let n = map.normal(vxi, vyi);
                pos_x -= n.x * 0.5;
                pos_y -= n.y * 0.5;
                pos_z -= n.z * 0.5;
            }

            vertices.push(Vertex {
                position: [pos_x, pos_y, pos_z],
                color: [r, g, b, 1.0],
                uv: [u as f32, v as f32],
                macro_uv: [mu, mv],
            });
            point_coords.push((vxi, vyi));
        }

        for geom in &geoms {
            let (verts, idxs, coords) = batches_by_tex.entry(geom.texture).or_default();
            let base = verts.len() as u32;
            verts.extend_from_slice(&vertices);
            coords.extend_from_slice(&point_coords);
            for &index in all_indices
                .iter()
                .skip(geom.idx_offset)
                .take(geom.idx_count)
            {
                idxs.push(base + index as u32);
            }
        }
    }

    batches_by_tex
}

// ---- BuildWater (MatrixMapGroup.cpp:366-452) ----
//
// The original builds a 64×64 alpha texture per water-bearing group that
// fades the terrain hole into the ocean (255 = fully water, 0 = fully
// land). The Rust port pre-scans groups at load time and uploads the
// alphas as an atlas (see `water::Water::new`). These helpers are the
// BuildWater-analog.

/// Depth ranges for the per-group shoreline alpha — see BuildWater.
const WATER_ALPHA_UP_LEVEL: f32 = -1.0;
const WATER_ALPHA_DOWN_LEVEL: f32 = -20.1;

/// Sample the heightmap at a world-space (wx, wy). Bridge cells are
/// bilinearly interpolated from their four corner points; everything
/// else falls through to `GameMap::get_z`. Ports the water-side height
/// sampling in the original BuildWater pass.
pub fn sample_height_for_water(map: &GameMap, wx: f32, wy: f32) -> f32 {
    let scaledx = wx / GLOBAL_SCALE;
    let scaledy = wy / GLOBAL_SCALE;
    let ix = scaledx as i32;
    let iy = scaledy as i32;

    if ix >= 0 && iy >= 0 && ix < map.size_x as i32 && iy < map.size_y as i32 {
        let unit = map.unit(ix as usize, iy as usize);
        if unit.flags & CELLFLAG_BRIDGE != 0 {
            let kx = scaledx - ix as f32;
            let ky = scaledy - iy as f32;

            let z0 = map.point(ix as usize, iy as usize).z;
            let z1 = map.point(ix as usize + 1, iy as usize).z;
            let z2 = map.point(ix as usize, iy as usize + 1).z;
            let z3 = map.point(ix as usize + 1, iy as usize + 1).z;

            let top = z0 + (z1 - z0) * kx;
            let bottom = z2 + (z3 - z2) * kx;
            return top + (bottom - top) * ky;
        }
    }

    map.get_z(wx, wy)
}

/// Build one `alpha_size × alpha_size` shoreline alpha image for the
/// group whose top-left world corner is `(group_x0, group_y0)`. Each
/// texel's alpha is a linear ramp from `DOWN_LEVEL` (fully water, 255)
/// to `UP_LEVEL` (fully land, 0). Ports the inner loop of BuildWater
/// in MatrixMapGroup.cpp:366-452.
pub fn build_water_alpha_image(
    map: &GameMap,
    group_x0: f32,
    group_y0: f32,
    alpha_size: usize,
) -> image::GrayImage {
    let alpha_step = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE / alpha_size as f32;
    let mut img = image::GrayImage::new(alpha_size as u32, alpha_size as u32);

    for j in 0..alpha_size {
        for i in 0..alpha_size {
            let wx = (i as f32 + 0.5) * alpha_step + group_x0;
            let wy = (j as f32 + 0.5) * alpha_step + group_y0;
            let wz = sample_height_for_water(map, wx, wy);

            let zz: u8 = if wz < WATER_ALPHA_DOWN_LEVEL {
                255
            } else if wz > WATER_ALPHA_UP_LEVEL {
                0
            } else {
                (255.0
                    - ((wz - WATER_ALPHA_DOWN_LEVEL)
                        / (WATER_ALPHA_UP_LEVEL - WATER_ALPHA_DOWN_LEVEL)
                        * 255.0)) as u8
            };

            img.put_pixel(i as u32, j as u32, image::Luma([zz]));
        }
    }

    img
}

/// Return the `(gx, gy)` group coordinates of every group whose terrain
/// dips below sea level (min-z < 0). These are the groups that need a
/// shoreline alpha texture. Ports the group-selection part of BuildWater.
pub fn water_bearing_groups(stor: &Storage, map: &GameMap) -> Vec<(i32, i32)> {
    let Some(groups_buf) = stor.get_buf("groups", "Data") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for gi in 0..groups_buf.arrays_count() {
        let raw = groups_buf.get_bytes(gi);
        if raw.len() < 4 {
            continue;
        }
        let gx = u16::from_le_bytes([raw[0], raw[1]]) as i32;
        let gy = u16::from_le_bytes([raw[2], raw[3]]) as i32;
        let w = MAP_GROUP_SIZE.min(map.size_x as i32 - gx);
        let h = MAP_GROUP_SIZE.min(map.size_y as i32 - gy);
        let mut min_z = f32::INFINITY;
        for py in 0..=h {
            for px in 0..=w {
                min_z = min_z.min(map.point((gx + px) as usize, (gy + py) as usize).z);
            }
        }
        if min_z < 0.0 {
            out.push((gx, gy));
        }
    }
    out
}
