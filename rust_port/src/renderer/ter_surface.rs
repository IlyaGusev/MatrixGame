//! Terrain surface overlays — ports CTerSurface::LoadM from MatrixTerSurface.cpp.

use std::collections::HashMap;
use wgpu::util::DeviceExt;

use super::terrain::{DrawBatch, Vertex};
use crate::assets::storage::Storage;
use crate::game::common::{rd_f32, rd_i32, rd_u16, rd_u32};
use crate::game::map::GameMap;
use crate::renderer::texture::{
    create_solid_texture, create_texture_from_rgba_mipped, decode_texture_bytes,
};

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
) -> Vec<DrawBatch> {
    let srfm = match stor.get_buf("surfacesM", "Data") {
        Some(b) if b.arrays_count() > 0 => b,
        _ => return vec![],
    };
    let strings = match stor.get_buf("strings", "String") {
        Some(b) => b,
        _ => return vec![],
    };

    let cx = map.world_width() * 0.5;
    let cy = map.world_height() * 0.5;

    let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
    let white = create_solid_texture(device, queue, [255, 255, 255, 255]);

    #[allow(dead_code)]
    struct SurfData {
        index: i32,
        tex_path: String,
        wrap_y: bool,
        color: [f32; 4],
        verts: Vec<Vertex>,
        indices: Vec<u16>,
    }
    let mut surfaces: Vec<SurfData> = Vec::new();

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
        let _grpsc = rd_u32(raw, &mut off) as usize;
        let disp_x = rd_f32(raw, &mut off);
        let disp_y = rd_f32(raw, &mut off);

        let tex_path = if ids >= 0 && (ids as usize) < strings.arrays_count() {
            strings
                .get_as_wstr(ids as usize)
                .split('?')
                .next()
                .unwrap_or("")
                .replace('\\', "/")
        } else {
            continue;
        };

        let r = ((color_dw >> 16) & 0xFF) as f32 / 255.0;
        let g = ((color_dw >> 8) & 0xFF) as f32 / 255.0;
        let b = (color_dw & 0xFF) as f32 / 255.0;
        let a = ((color_dw >> 24) & 0xFF) as f32 / 255.0;

        let needed = off + vcnt * 32 + idxsz;
        if needed > raw.len() {
            continue;
        }

        let mut verts = Vec::with_capacity(vcnt);
        let mut wrap_y = false;
        for _ in 0..vcnt {
            let px = rd_f32(raw, &mut off);
            let py = rd_f32(raw, &mut off);
            let pz = rd_f32(raw, &mut off);
            let vcol = rd_u32(raw, &mut off);
            let tu = rd_f32(raw, &mut off);
            let tv = rd_f32(raw, &mut off);
            let _tum = rd_f32(raw, &mut off);
            let _tvm = rd_f32(raw, &mut off);

            let vr = ((vcol >> 16) & 0xFF) as f32 / 255.0;
            let vg = ((vcol >> 8) & 0xFF) as f32 / 255.0;
            let vb = (vcol & 0xFF) as f32 / 255.0;
            let va = ((vcol >> 24) & 0xFF) as f32 / 255.0;
            if tv < 0.0 || tv > 1.0 {
                wrap_y = true;
            }

            verts.push(Vertex {
                position: [px + disp_x - cx, py + disp_y - cy, pz + 0.05],
                color: [vr * r, vg * g, vb * b, va * a],
                uv: [tu, tv],
                macro_uv: [_tum, _tvm],
            });
        }

        let idx_count = idxsz / 2;
        let mut strip = Vec::with_capacity(idx_count);
        for _ in 0..idx_count {
            if off + 2 > raw.len() {
                break;
            }
            strip.push(rd_u16(raw, &mut off));
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

        surfaces.push(SurfData {
            index,
            tex_path,
            wrap_y,
            color: [r, g, b, a],
            verts,
            indices: strip,
        });
    }

    surfaces.sort_by_key(|s| s.index);

    let mut overlay_batches = Vec::new();
    let mut overlay_tris = 0u32;

    for surf in &surfaces {
        if surf.indices.len() < 3 {
            continue;
        }

        let tex_view = tex_cache.get(surf.tex_path.as_str()).unwrap_or(&white);

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
                    resource: wgpu::BindingResource::Sampler(if surf.wrap_y {
                        sampler_wrap_v
                    } else {
                        sampler_clamp
                    }),
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

        let prim_count = surf.indices.len().saturating_sub(2) as u32;
        overlay_tris += prim_count;
        overlay_batches.push(DrawBatch {
            bind_group: bg,
            vertex_buffer: vb,
            index_buffer: ib,
            num_indices: surf.indices.len() as u32,
            index_format: wgpu::IndexFormat::Uint16,
            cpu_vertices: None,
            point_coords: None,
        });
    }

    log::info!(
        "terrain overlays: {} batches, {} triangles, {} textures",
        overlay_batches.len(),
        overlay_tris,
        tex_cache.len()
    );
    overlay_batches
}
