//! wgpu back end for the effect primitives — the Rust stand-in for
//! `CBillboard::SortEndDraw` (MatrixLib/3G/CBillboard.cpp:20-284) plus
//! the BBT texture table loaded by `CMatrixEffect::InitEffects`
//! (MatrixEffect.cpp:386-429).
//!
//! Two pipelines mirror the two D3D blend setups: alpha
//! (SRCALPHA/INVSRCALPHA, the depth-sorted "sortable" bucket) and
//! additive (ONE/ONE, the "intense" bucket). Both depth-test against
//! the main pass with depth writes off and no culling.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::matrix_lib::base::storage::Storage;
use crate::matrix_lib::three_g::billboard::{
    expand_billboard, expand_line, BillboardQueue, TexRef, BBT_LAST, BBT_NAMES,
};
use crate::matrix_lib::three_g::texture::{
    create_texture_from_rgba, decode_texture_bytes, resolve_texture_name,
};
use wgpu::util::DeviceExt;

/// Standalone effect textures referenced by path (TEXTURE_PATH_* in
/// StringConstants.hpp). Loaded up front like the BBT set.
pub const TEXTURE_PATH_GUN_FIRE: &str = "Matrix/Textures/Effects/gun_fire";
pub const TEXTURE_PATH_SPLASH: &str = "Matrix/Textures/Effects/splash";
pub const TEXTURE_PATH_BIGBOOM: &str = "Matrix/Textures/Effects/bigboom";
pub const TEXTURE_PATH_GUN_BULLETS1: &str = "Matrix/Textures/Effects/gun_bullets1";
pub const TEXTURE_PATH_GUN_BULLETS2: &str = "Matrix/Textures/Effects/gun_bullets2";
/// `TEXTURE_PATH_KEELWATER` (StringConstants.hpp:137).
pub const TEXTURE_PATH_KEELWATER: &str = "Matrix/Textures/Billboard/keelwater";

const PATH_TEXTURES: [&str; 9] = [
    crate::matrix_game::effects::move_to::TEXTURE_PATH_MOVETO,
    TEXTURE_PATH_GUN_FIRE,
    TEXTURE_PATH_SPLASH,
    TEXTURE_PATH_BIGBOOM,
    TEXTURE_PATH_GUN_BULLETS1,
    TEXTURE_PATH_GUN_BULLETS2,
    TEXTURE_PATH_KEELWATER,
    crate::matrix_game::effects::zahvat::TEXTURE_PATH_ZAHVAT,
    crate::matrix_game::effects::dust::TEXTURE_PATH_DUST,
];

/// One BBT table entry — atlas rect in the sortable texture (alpha
/// bucket) or a standalone additive texture.
#[derive(Clone, Copy)]
struct BbtEntry {
    intense: bool,
    /// uv rect (tu0, tv0, tu1, tv1) — full `(0,0,1,1)` for intense.
    uv: [f32; 4],
    /// Index into `bind_groups`; intense entries get their own
    /// texture, sorted entries all share slot 0 (the atlas).
    tex_slot: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 4],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

/// ARGB u32 → linear RGBA f32 (the original modulates the raw sRGB
/// value; the existing pipelines in this port do the same).
fn argb_to_f32(c: u32) -> [f32; 4] {
    [
        ((c >> 16) & 0xFF) as f32 / 255.0,
        ((c >> 8) & 0xFF) as f32 / 255.0,
        (c & 0xFF) as f32 / 255.0,
        ((c >> 24) & 0xFF) as f32 / 255.0,
    ]
}

/// A draw batch: consecutive vertex range sharing texture + blend.
struct Batch {
    tex_slot: usize,
    additive: bool,
    first: u32,
    count: u32,
}

pub struct EffectsRenderer {
    pipeline_alpha: wgpu::RenderPipeline,
    pipeline_add: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    bind_groups: Vec<wgpu::BindGroup>,
    bbt: [BbtEntry; BBT_LAST],
    path_slots: HashMap<&'static str, usize>,
    vb: wgpu::Buffer,
    vb_cap: usize,
    batches: Vec<Batch>,
    vertices: Vec<Vertex>,
    spot_vb: wgpu::Buffer,
    spot_vb_cap: usize,
    spot_batches: Vec<Batch>,
    spot_vertices: Vec<Vertex>,
}

impl EffectsRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        depth_format: wgpu::TextureFormat,
        matrix_data: &Storage,
        read_file: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("billboard shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/billboard.wgsl").into(),
            ),
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx uniforms bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx texture bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx pipeline layout"),
            bind_group_layouts: &[&uniform_bgl, &tex_bgl],
            immediate_size: 0,
        });

        let make_pipeline = |blend: wgpu::BlendState, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32x2],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None, // D3DCULL_NONE
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: false, // D3DRS_ZWRITEENABLE = FALSE
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                cache: None,
                multiview_mask: None,
            })
        };

        // SRCALPHA / INVSRCALPHA.
        let pipeline_alpha = make_pipeline(wgpu::BlendState::ALPHA_BLENDING, "fx alpha");
        // ONE / ONE — D3D additive uses src color directly (alpha
        // pre-applied by the MODULATE), so blend with src factor One
        // on (color * alpha) computed in the shader... the original
        // modulates color by texture and the alpha-blend ONE/ONE adds
        // `src.rgb` — replicate with PremultipliedAdd semantics:
        let pipeline_add = make_pipeline(
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            "fx additive",
        );

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx uniforms bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fx sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let mut bind_groups: Vec<wgpu::BindGroup> = Vec::new();
        let mut make_bg = |view: &wgpu::TextureView| -> usize {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx tex bg"),
                layout: &tex_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            bind_groups.push(bg);
            bind_groups.len() - 1
        };

        let load_view = |path: &str| -> Option<(wgpu::TextureView, u32, u32)> {
            let bytes = read_file(path)?;
            let img = decode_texture_bytes(&bytes)?;
            let (w, h) = (img.width(), img.height());
            Some((create_texture_from_rgba(device, queue, &img), w, h))
        };

        // White 1×1 fallback in slot reserved later via map default.
        let white = image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]));
        let white_view = create_texture_from_rgba(device, queue, &white);
        let white_slot = make_bg(&white_view);

        // ── BBT table (InitEffects port) ────────────────────────────
        let mut bbt = [BbtEntry {
            intense: true,
            uv: [0.0, 0.0, 1.0, 1.0],
            tex_slot: white_slot,
        }; BBT_LAST];

        let bb_rec = matrix_data.block_record("da", "Billboards");
        if let Some(bb_rec) = bb_rec {
            // The sortable atlas (`TexSort` param).
            let (sort_slot, sort_w, sort_h) = matrix_data
                .block_param(&bb_rec, "TexSort")
                .and_then(|p| load_view(&p.replace('\\', "/")))
                .map(|(v, w, h)| (make_bg(&v), w as f32, h as f32))
                .unwrap_or((white_slot, 1.0, 1.0));

            if let Some(tex_rec) = matrix_data.block_record(&bb_rec, "Textures") {
                for (id, name) in BBT_NAMES {
                    let Some(par) = matrix_data.block_param(&tex_rec, name) else {
                        continue;
                    };
                    let parts: Vec<&str> = par.split(',').collect();
                    if parts.first().map(|s| s.trim()) == Some("0") {
                        // Sorted atlas rect: 0,x0,y0,sx,sy.
                        let n = |i: usize| -> f32 {
                            parts
                                .get(i)
                                .and_then(|s| s.trim().parse::<f32>().ok())
                                .unwrap_or(0.0)
                        };
                        let (x0, y0, sx, sy) = (n(1), n(2), n(3), n(4));
                        bbt[id] = BbtEntry {
                            intense: false,
                            uv: [
                                x0 / sort_w,
                                y0 / sort_h,
                                (x0 + sx) / sort_w,
                                (y0 + sy) / sort_h,
                            ],
                            tex_slot: sort_slot,
                        };
                    } else if let Some(path) = parts.get(1) {
                        let slot = load_view(&path.trim().replace('\\', "/"))
                            .map(|(v, _, _)| make_bg(&v))
                            .unwrap_or(white_slot);
                        bbt[id] = BbtEntry {
                            intense: true,
                            uv: [0.0, 0.0, 1.0, 1.0],
                            tex_slot: slot,
                        };
                    }
                }
            }
        } else {
            log::warn!("effects: robots.dat has no Billboards block — effect textures disabled");
        }

        // Standalone path textures (alpha-blended unless used by an
        // intense primitive — the konus `intense` ctor flag decides
        // the bucket per primitive, not per texture).
        let mut path_slots: HashMap<&'static str, usize> = HashMap::new();
        for p in PATH_TEXTURES {
            let slot = load_view(p)
                .map(|(v, _, _)| make_bg(&v))
                .unwrap_or(white_slot);
            path_slots.insert(p, slot);
        }
        let mut path_slots_extra: Vec<(&'static str, usize)> = Vec::new();

        let vb_cap = 4096;
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx vb"),
            size: (vb_cap * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spot_vb_cap = 4096;
        let spot_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx spot vb"),
            size: (spot_vb_cap * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Spot textures load lazily through `path_slots` — pre-warm
        // the three LandSpots maps.
        for p in [
            "Matrix/Textures/LandSpots/varonka",
            "Matrix/Textures/LandSpots/spot_hit",
            "Matrix/Textures/LandSpots/spot",
            "Matrix/Textures/LandSpots/sole",
            "Matrix/Textures/LandSpots/sole_w",
            "Matrix/Textures/LandSpots/sole_p",
        ] {
            let slot = load_view(p)
                .map(|(v, _, _)| make_bg(&v))
                .unwrap_or(white_slot);
            path_slots_extra.push((p, slot));
        }

        let mut path_slots = path_slots;
        for (p, slot) in path_slots_extra {
            path_slots.insert(p, slot);
        }

        Self {
            pipeline_alpha,
            pipeline_add,
            uniform_buf,
            uniform_bg,
            bind_groups,
            bbt,
            path_slots,
            vb,
            vb_cap,
            batches: Vec::new(),
            vertices: Vec::new(),
            spot_vb,
            spot_vb_cap,
            spot_batches: Vec::new(),
            spot_vertices: Vec::new(),
        }
    }

    fn resolve(&self, tex: TexRef) -> (usize, [f32; 4], bool) {
        match tex {
            TexRef::Bbt(i) => {
                let e = self.bbt[i.min(BBT_LAST - 1)];
                (e.tex_slot, e.uv, e.intense)
            }
            TexRef::Path(p) => {
                let slot = self.path_slots.get(p).copied().unwrap_or(0);
                (slot, [0.0, 0.0, 1.0, 1.0], false)
            }
        }
    }

    /// Expand the queue into vertices + batches and upload. Mirrors
    /// `SortEndDraw`: alpha bucket sorted back-to-front by view depth,
    /// additive bucket after, batched by texture.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: &BillboardQueue,
        view_proj: Mat4,
        view: Mat4,
        cam_pos: Vec3,
        cam_right: Vec3,
        cam_up: Vec3,
        map_center: glam::Vec2,
    ) {
        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
            }),
        );

        self.vertices.clear();
        self.batches.clear();
        let mc = Vec3::new(map_center.x, map_center.y, 0.0);

        // Item = (sort_z, additive, tex_slot, corners, uv rect or tri).
        enum Prim {
            Quad([Vec3; 4], [f32; 4]),
            Tri([Vec3; 3], [[f32; 2]; 3]),
        }
        struct Item {
            z: f32,
            additive: bool,
            tex_slot: usize,
            color: u32,
            prim: Prim,
        }
        let mut items: Vec<Item> =
            Vec::with_capacity(q.billboards.len() + q.lines.len() + q.tris.len());

        let view_z = |p: Vec3| -> f32 {
            // Depth along view forward (the C++ sorts by view-space z).
            (view * p.extend(1.0)).z
        };

        for b in &q.billboards {
            let (slot, uv, intense) = self.resolve(b.tex);
            let v = expand_billboard(b, cam_right, cam_up).map(|p| p - mc);
            items.push(Item {
                z: view_z(b.pos - mc),
                additive: intense,
                tex_slot: slot,
                color: b.color,
                prim: Prim::Quad(v, uv),
            });
        }
        for l in &q.lines {
            let (slot, uv, intense) = self.resolve(l.tex);
            // Lines are always in the intense path in the C++ (their
            // textures are all standalone additive ones).
            let v = expand_line(l, cam_pos).map(|p| p - mc);
            items.push(Item {
                z: view_z((l.p0 + l.p1) * 0.5 - mc),
                additive: intense || matches!(l.tex, TexRef::Bbt(_)),
                tex_slot: slot,
                color: l.color,
                prim: Prim::Quad(v, uv),
            });
        }
        for t in &q.tris {
            let (slot, _, _) = self.resolve(t.tex);
            items.push(Item {
                z: view_z(t.v[0] - mc),
                additive: t.additive,
                tex_slot: slot,
                color: t.color,
                prim: Prim::Tri([t.v[0] - mc, t.v[1] - mc, t.v[2] - mc], t.uv),
            });
        }

        // Alpha bucket back-to-front; additive after, grouped by
        // texture for batching.
        items.sort_by(|a, b| {
            a.additive.cmp(&b.additive).then_with(|| {
                if a.additive {
                    a.tex_slot.cmp(&b.tex_slot)
                } else {
                    b.z.partial_cmp(&a.z).unwrap_or(std::cmp::Ordering::Equal)
                }
            })
        });

        for it in &items {
            let col = argb_to_f32(it.color);
            let first = self.vertices.len() as u32;
            match &it.prim {
                Prim::Quad(v, uv) => {
                    // Corner order (-1,-1)(-1,1)(1,-1)(1,1) →
                    // uv (u0,v1)(u0,v0)(u1,v1)(u1,v0) as in the C++.
                    let vtx = |p: Vec3, u: f32, vv: f32| Vertex {
                        pos: p.to_array(),
                        color: col,
                        uv: [u, vv],
                    };
                    let [u0, v0, u1, v1] = *uv;
                    let quad = [
                        vtx(v[0], u0, v1),
                        vtx(v[1], u0, v0),
                        vtx(v[2], u1, v1),
                        vtx(v[3], u1, v0),
                    ];
                    self.vertices
                        .extend_from_slice(&[quad[0], quad[1], quad[2], quad[2], quad[1], quad[3]]);
                }
                Prim::Tri(v, uv) => {
                    for k in 0..3 {
                        self.vertices.push(Vertex {
                            pos: v[k].to_array(),
                            color: col,
                            uv: uv[k],
                        });
                    }
                }
            }
            let count = self.vertices.len() as u32 - first;
            match self.batches.last_mut() {
                Some(b) if b.tex_slot == it.tex_slot && b.additive == it.additive => {
                    b.count += count;
                }
                _ => self.batches.push(Batch {
                    tex_slot: it.tex_slot,
                    additive: it.additive,
                    first,
                    count,
                }),
            }
        }

        if self.vertices.len() > self.vb_cap {
            self.vb_cap = self.vertices.len().next_power_of_two();
            self.vb = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx vb"),
                size: (self.vb_cap * std::mem::size_of::<Vertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !self.vertices.is_empty() {
            queue.write_buffer(&self.vb, 0, bytemuck::cast_slice(&self.vertices));
        }
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.vertices.is_empty() {
            return;
        }
        pass.set_vertex_buffer(0, self.vb.slice(..));
        pass.set_bind_group(0, &self.uniform_bg, &[]);
        let mut current_add: Option<bool> = None;
        for b in &self.batches {
            if current_add != Some(b.additive) {
                pass.set_pipeline(if b.additive {
                    &self.pipeline_add
                } else {
                    &self.pipeline_alpha
                });
                current_add = Some(b.additive);
            }
            pass.set_bind_group(1, &self.bind_groups[b.tex_slot], &[]);
            pass.draw(b.first..b.first + b.count, 0..1);
        }
    }
}

// ── Landscape-spot rendering (shares the billboard shader) ──────────

impl EffectsRenderer {
    /// Upload + record the landscape-spot decals. Drawn right after
    /// the terrain (the C++ slots `CMatrixEffectLandscapeSpot::DrawAll`
    /// between the surfaces and the water, MatrixMap.cpp:2244).
    pub fn upload_spots(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spots: &crate::matrix_game::effects::landscape_spot::LandscapeSpots,
        map_center: glam::Vec2,
    ) {
        self.spot_vertices.clear();
        self.spot_batches.clear();
        for s in &spots.spots {
            let (slot, _, _) = self.resolve(s.kind.texture());
            let col = argb_to_f32(s.color);
            let first = self.spot_vertices.len() as u32;
            for &i in &s.indices {
                let v = &s.verts[i as usize];
                self.spot_vertices.push(Vertex {
                    pos: [v.pos.x - map_center.x, v.pos.y - map_center.y, v.pos.z],
                    color: col,
                    uv: v.uv,
                });
            }
            let count = self.spot_vertices.len() as u32 - first;
            match self.spot_batches.last_mut() {
                Some(b) if b.tex_slot == slot => b.count += count,
                _ => self.spot_batches.push(Batch {
                    tex_slot: slot,
                    additive: false,
                    first,
                    count,
                }),
            }
        }
        if self.spot_vertices.len() > self.spot_vb_cap {
            self.spot_vb_cap = self.spot_vertices.len().next_power_of_two();
            self.spot_vb = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx spot vb"),
                size: (self.spot_vb_cap * std::mem::size_of::<Vertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !self.spot_vertices.is_empty() {
            queue.write_buffer(&self.spot_vb, 0, bytemuck::cast_slice(&self.spot_vertices));
        }
    }

    pub fn render_spots<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.spot_vertices.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline_alpha);
        pass.set_vertex_buffer(0, self.spot_vb.slice(..));
        pass.set_bind_group(0, &self.uniform_bg, &[]);
        for b in &self.spot_batches {
            pass.set_bind_group(1, &self.bind_groups[b.tex_slot], &[]);
            pass.draw(b.first..b.first + b.count, 0..1);
        }
    }
}

// ── Effect mesh renderer (projectiles + debris pieces) ───────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MeshVertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MeshUniform {
    rows: [[f32; 4]; 4],
    color: [f32; 4],
    _pad: [f32; 44], // pad to the 256-byte dynamic-offset alignment
}

// Dynamic uniform offsets must be multiples of
// `min_uniform_buffer_offset_alignment` (256 on every backend we
// target) — a wrong stride panics in `set_bind_group` at runtime.
const _: () = assert!(std::mem::size_of::<MeshUniform>() == 256);

struct GpuMesh {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    index_count: u32,
    tex_slot: usize,
}

/// Draws the `roket` / `mina` / `bullet` VO meshes and the
/// `Models/Debris` catalog pieces — the `CVectorObject::Draw` calls of
/// `CMatrixEffectMovingObject` and the explosion debris.
pub struct EffectMeshRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    bind_groups: Vec<wgpu::BindGroup>,
    draw_ubo: wgpu::Buffer,
    draw_bg: wgpu::BindGroup,
    draw_cap: usize,
    bullets: [Option<GpuMesh>; 3], // rocket / mina / bullet
    debris: Vec<GpuMesh>,
    debris_types: Vec<i32>,
    queued: Vec<(usize, bool, MeshUniform)>, // (mesh slot, is_debris, uniform)
}

impl EffectMeshRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        depth_format: wgpu::TextureFormat,
        matrix_data: &Storage,
        read_file: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx mesh shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/effect_mesh.wgsl").into(),
            ),
        });
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx mesh vp bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx mesh draw bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<MeshUniform>() as u64
                    ),
                },
                count: None,
            }],
        });
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx mesh tex bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx mesh layout"),
            bind_group_layouts: &[&uniform_bgl, &draw_bgl, &tex_bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fx mesh"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            cache: None,
            multiview_mask: None,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx mesh vp"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx mesh vp bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let draw_cap = 1024;
        let draw_ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx mesh draws"),
            size: (draw_cap * std::mem::size_of::<MeshUniform>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx mesh draws bg"),
            layout: &draw_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &draw_ubo,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<MeshUniform>() as u64),
                }),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fx mesh sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut bind_groups = Vec::new();
        let mut make_bg = |view: &wgpu::TextureView| -> usize {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fx mesh tex"),
                layout: &tex_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            bind_groups.push(bg);
            bind_groups.len() - 1
        };
        let white = image::RgbaImage::from_pixel(1, 1, image::Rgba([200, 200, 200, 255]));
        let white_view = create_texture_from_rgba(device, queue, &white);
        let white_slot = make_bg(&white_view);

        let mut load_mesh = |path: &str| -> Option<GpuMesh> {
            use crate::matrix_lib::three_g::vector_object::parse_vo;
            let bytes = read_file(path)?;
            let mesh = parse_vo(&bytes).ok()?;
            // Single texture per effect mesh (the bullets/debris VOs
            // are single-surface); take the first surface's texture.
            let tex_slot = mesh
                .surfaces
                .first()
                .and_then(|s| s.texture_ref.as_ref())
                .and_then(|t| resolve_texture_name(t, None))
                .and_then(|t: String| {
                    let p = t.replace('\\', "/");
                    let bytes = read_file(&p)?;
                    let img = decode_texture_bytes(&bytes)?;
                    Some(make_bg(&create_texture_from_rgba(device, queue, &img)))
                })
                .unwrap_or(white_slot);
            let mut verts: Vec<MeshVertex> = mesh
                .vertices
                .iter()
                .map(|v| MeshVertex {
                    pos: v.position,
                    uv: v.uv,
                })
                .collect();
            if verts.is_empty() {
                verts.push(MeshVertex {
                    pos: [0.0; 3],
                    uv: [0.0; 2],
                });
            }
            let indices: Vec<u32> = mesh
                .surfaces
                .iter()
                .flat_map(|s| s.indices.iter().copied())
                .collect();
            if indices.is_empty() {
                return None;
            }
            Some(GpuMesh {
                vb: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("fx mesh vb"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                ib: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("fx mesh ib"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
                index_count: indices.len() as u32,
                tex_slot,
            })
        };

        let bullets = [
            load_mesh("Matrix/Bullets/roket.vo"),
            load_mesh("Matrix/Bullets/mina.vo"),
            load_mesh("Matrix/Bullets/bullet.vo"),
        ];
        // Debris catalog — `Models/Debris` (MatrixEffect.cpp:302-318).
        // The par NAME's first comma-field is the deb_type explosions
        // partition the catalog by ("0,0" → gaika, "1,0" → mysor).
        let mut debris = Vec::new();
        let mut debris_types = Vec::new();
        if let Some(models_rec) = matrix_data.block_record("da", "Models") {
            if let Some(deb_rec) = matrix_data.block_record(&models_rec, "Debris") {
                if let (Some(k), Some(v)) = (
                    matrix_data.get_buf(&deb_rec, "0"),
                    matrix_data.get_buf(&deb_rec, "1"),
                ) {
                    for i in 0..v.arrays_count() {
                        let path = v.get_as_wstr(i).replace('\\', "/");
                        if let Some(m) = load_mesh(&path) {
                            let name = k.get_as_wstr(i);
                            debris.push(m);
                            debris_types.push(
                                name.split(',')
                                    .next()
                                    .and_then(|s| s.trim().parse::<i32>().ok())
                                    .unwrap_or(0),
                            );
                        }
                    }
                }
            }
        }
        log::info!(
            "effects: loaded {} debris meshes, bullets: {:?}",
            debris.len(),
            [
                bullets[0].is_some(),
                bullets[1].is_some(),
                bullets[2].is_some()
            ]
        );

        Self {
            pipeline,
            uniform_buf,
            uniform_bg,
            bind_groups,
            draw_ubo,
            draw_bg,
            draw_cap,
            bullets,
            debris,
            debris_types,
            queued: Vec::new(),
        }
    }

    pub fn debris_count(&self) -> usize {
        self.debris.len().max(1)
    }

    pub fn debris_types(&self) -> &[i32] {
        &self.debris_types
    }

    pub fn upload(
        &mut self,
        queue: &wgpu::Queue,
        meshes: &crate::matrix_game::effects::explosion::MeshQueue,
        view_proj: Mat4,
        map_center: glam::Vec2,
    ) {
        use crate::matrix_game::effects::explosion::MeshKind;
        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
            }),
        );
        self.queued.clear();
        for d in meshes.draws.iter().take(self.draw_cap) {
            let (slot, is_debris) = match d.kind {
                MeshKind::Rocket => (0usize, false),
                MeshKind::Mina => (1, false),
                MeshKind::Bullet => (2, false),
                MeshKind::Debris(i) => {
                    if self.debris.is_empty() {
                        continue;
                    }
                    (i % self.debris.len(), true)
                }
            };
            if !is_debris && self.bullets[slot].is_none() {
                continue;
            }
            let rows = [
                [d.rows[0].x, d.rows[0].y, d.rows[0].z, 0.0],
                [d.rows[1].x, d.rows[1].y, d.rows[1].z, 0.0],
                [d.rows[2].x, d.rows[2].y, d.rows[2].z, 0.0],
                [
                    d.rows[3].x - map_center.x,
                    d.rows[3].y - map_center.y,
                    d.rows[3].z,
                    1.0,
                ],
            ];
            self.queued.push((
                slot,
                is_debris,
                MeshUniform {
                    rows,
                    color: argb_to_f32(d.color),
                    _pad: [0.0; 44],
                },
            ));
        }
        if !self.queued.is_empty() {
            let data: Vec<MeshUniform> = self.queued.iter().map(|(_, _, u)| *u).collect();
            queue.write_buffer(&self.draw_ubo, 0, bytemuck::cast_slice(&data));
        }
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.queued.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniform_bg, &[]);
        let stride = std::mem::size_of::<MeshUniform>() as u32;
        for (i, (slot, is_debris, _)) in self.queued.iter().enumerate() {
            let mesh = if *is_debris {
                &self.debris[*slot]
            } else {
                self.bullets[*slot].as_ref().unwrap()
            };
            pass.set_bind_group(1, &self.draw_bg, &[i as u32 * stride]);
            pass.set_bind_group(2, &self.bind_groups[mesh.tex_slot], &[]);
            pass.set_vertex_buffer(0, mesh.vb.slice(..));
            pass.set_index_buffer(mesh.ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }
}
