//! UI quad renderer for `CInterface` panels.
//!
//! Multi-atlas: a panel's elements often reference several texture
//! atlases (e.g. `if/Main` uses `interface1` / `interface2` /
//! `interface3` / `base_1` / `base_4` / `text_1`). We bucket
//! elements by their atlas path, emit one draw per bucket with the
//! corresponding bind group, and share a single vertex buffer
//! across all of them.
//!
//! Port of the render path in `CIFaceElement::Render` (Interface/
//! CIFaceElement.cpp) — the D3D pipeline there just draws textured
//! quads with the per-state atlas sub-rect, same idea.

use std::borrow::Cow;
use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::matrix_lib::three_g::texture::{create_texture_from_rgba, decode_texture_bytes};

use super::hint::{Hint, HintChromeLibrary, HintPart};
use super::iface_element::ElementState;
use super::interface::{CInterface, DESIGN_H};
use super::text::{create_atlas_texture, parse_rich_text, GlyphAtlas, RichRun, TEXT_ATLAS_KEY};


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    tint: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    screen_size: [f32; 4],
}

/// One loaded atlas — keeps the texture view + bind group alive for
/// the lifetime of the renderer, plus the pixel dimensions needed to
/// normalise arbitrary (x, y, w, h) subrects into 0..=1 UVs (the 9-
/// slice hint chrome + inline resource icons need this).
struct Atlas {
    _tex: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

/// One draw-bucket per atlas per frame — a contiguous vertex range
/// that all samples from the same atlas.
struct DrawGroup {
    atlas_key: String,
    start: u32,
    count: u32,
}

pub struct InterfaceRenderer {
    pipeline: wgpu::RenderPipeline,
    /// group 0 — per-frame uniforms. Kept alive because the pipeline
    /// layout references it even though we don't consult it post-init.
    #[allow(dead_code)]
    uniform_bgl: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    /// group 1 — atlas texture + sampler (swapped per draw).
    atlas_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    text_sampler: wgpu::Sampler,

    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    atlases: HashMap<String, Atlas>,
    draw_groups: Vec<DrawGroup>,
    num_verts: u32,
    /// Glyph cache + GPU upload tracking. Re-uploaded whenever a new
    /// glyph is rasterised (atlas.generation changes).
    glyph_atlas: GlyphAtlas,
    glyph_atlas_generation: u64,
}

impl InterfaceRenderer {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Interface Uniform BGL"),
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
        let atlas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Interface Atlas BGL"),
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Interface Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Interface PL"),
            bind_group_layouts: &[&uniform_bgl, &atlas_bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Interface Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Interface UB"),
            contents: bytemuck::bytes_of(&Uniforms {
                screen_size: [1.0, 1.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Interface Uniform BG"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        // Linear sampler for the text atlas. The panel scale is
        // typically non-integer (~1.4 at 1080p from the 1024×768
        // design space), and nearest-neighbour sampling at that scale
        // gives 1-px native stems an inconsistent 1-or-2 output-pixel
        // width depending on where each glyph's left edge lands.
        // Linear blends the texels evenly so every stem renders at a
        // uniform width — slightly softened, but visually consistent.
        let text_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let vertex_capacity = 6 * 1024; // plenty of quads.
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Interface VB"),
            size: (vertex_capacity * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_bgl,
            uniform_buffer,
            uniform_bg,
            atlas_bgl,
            sampler,
            text_sampler,
            vertex_buffer,
            vertex_capacity,
            atlases: HashMap::new(),
            draw_groups: Vec::new(),
            num_verts: 0,
            glyph_atlas: GlyphAtlas::new(),
            glyph_atlas_generation: 0,
        }
    }

    /// Re-upload the glyph atlas to the GPU if new glyphs were
    /// rasterised since the last upload. Called from `upload_inner`
    /// after laying out all text quads, before the renderer flushes
    /// the vertex buffer.
    fn sync_glyph_atlas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.glyph_atlas_generation == self.glyph_atlas.generation {
            // No new glyphs — keep existing bind group.
            return;
        }
        let view = create_atlas_texture(device, queue, &self.glyph_atlas);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Text Atlas BG"),
            layout: &self.atlas_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.text_sampler),
                },
            ],
        });
        self.atlases.insert(
            TEXT_ATLAS_KEY.to_string(),
            Atlas {
                _tex: view,
                bind_group,
                width: self.glyph_atlas.width(),
                height: self.glyph_atlas.height(),
            },
        );
        self.glyph_atlas_generation = self.glyph_atlas.generation;
    }

    /// Load (or re-use) an atlas texture at `atlas_path`, keyed by the
    /// normalised lowercase/forward-slash form so it survives case /
    /// slash differences between the C++ data, the pkg, and the bundle.
    pub fn ensure_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        read: &dyn Fn(&str) -> Option<Vec<u8>>,
        atlas_path: &str,
    ) {
        let key = normalise_atlas_key(atlas_path);
        if self.atlases.contains_key(&key) {
            return;
        }
        // Resolve bytes across all case/slash/extension variants.
        let fwd = atlas_path.replace('\\', "/");
        let with_iface = fwd
            .replace("/IFace/", "/Iface/")
            .replace("IFace/", "Iface/");
        let with_lower = fwd
            .replace("/IFace/", "/iface/")
            .replace("IFace/", "iface/");
        let upper = fwd.to_uppercase();
        let mut candidates: Vec<String> = Vec::new();
        for base in [&fwd, &with_iface, &with_lower, &upper] {
            candidates.push(base.clone());
            candidates.push(format!("{base}.png"));
            candidates.push(format!("{base}.PNG"));
        }
        candidates.dedup();
        let mut bytes: Option<Vec<u8>> = None;
        for c in &candidates {
            if let Some(b) = read(c) {
                bytes = Some(b);
                break;
            }
        }
        let Some(bytes) = bytes else {
            log::warn!("interface: atlas {} not found in pkg/bundle", atlas_path);
            return;
        };
        let Some(rgba) = decode_texture_bytes(&bytes) else {
            log::warn!("interface: atlas {} failed to decode", atlas_path);
            return;
        };
        let view = create_texture_from_rgba(device, queue, &rgba);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Interface Atlas BG"),
            layout: &self.atlas_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        log::info!(
            "interface: atlas {} ({} bytes) bound as {}",
            atlas_path,
            bytes.len(),
            key
        );
        self.atlases.insert(
            key,
            Atlas {
                _tex: view,
                bind_group,
                width: rgba.width(),
                height: rgba.height(),
            },
        );
    }

    /// Preload the hint chrome atlases. Called at init right after
    /// `preload_for_panels`. Mirrors `CMatrixHint::PreloadBitmaps`
    /// (MatrixHint.cpp:441-459) — loads the border0 PNG + every alias
    /// in `Hints/Bitmaps` (resource icons, face portraits, etc.).
    pub fn preload_hint_chrome(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        read: &dyn Fn(&str) -> Option<Vec<u8>>,
        chrome: &HintChromeLibrary,
    ) {
        for border in chrome.borders.values() {
            if !border.source_path.is_empty() {
                self.ensure_atlas(device, queue, read, &border.source_path);
            }
        }
        for bmp in chrome.bitmaps.values() {
            if !bmp.path.is_empty() {
                self.ensure_atlas(device, queue, read, &bmp.path);
            }
        }
    }

    /// `(width, height)` of the PNG bound under `path`, or `None` if
    /// not loaded. Used by the hint builder to resolve inline
    /// `_BITMAP:` sizes (the AFT glyph layout needs to know how much
    /// horizontal space each resource icon consumes).
    pub fn atlas_size(&self, path: &str) -> Option<(u32, u32)> {
        let key = normalise_atlas_key(path);
        self.atlases.get(&key).map(|a| (a.width, a.height))
    }

    pub fn glyph_atlas_mut(&mut self) -> &mut GlyphAtlas {
        &mut self.glyph_atlas
    }

    /// Preload every atlas referenced by the given panels. Called once
    /// at init so `upload` never has to lazily load.
    pub fn preload_for_panels<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        read: &dyn Fn(&str) -> Option<Vec<u8>>,
        panels: impl IntoIterator<Item = &'a CInterface>,
    ) {
        let mut paths: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut panel_names: Vec<String> = Vec::new();
        for p in panels {
            panel_names.push(format!("{}({})", p.name, p.elements.len()));
            for e in &p.elements {
                for img in e.images.iter().flatten() {
                    if !img.tex_path.is_empty() {
                        paths.insert(img.tex_path.clone());
                    }
                }
            }
        }
        log::info!(
            "interface: preload_for_panels — panels={:?}, {} unique tex paths: {:?}",
            panel_names,
            paths.len(),
            paths
        );
        for path in paths {
            self.ensure_atlas(device, queue, read, &path);
        }
    }

    /// Rebuild the vertex stream + draw-groups from the current panel
    /// snapshot. Designed to run every frame.
    ///
    /// `popup` is the active constructor popup (or None) — when
    /// present, additional rows are appended on top of the regular
    /// panel draws, taking each item's icon images from the matching
    /// `chas{N}` / `hull{N}` / `head{N}` / `weap{N}` element on the
    /// Base panel.
    pub fn upload_with_popup(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        panels: &[&CInterface],
        popup: Option<&super::iface_menu::CIFaceMenu>,
        screen_w: f32,
        screen_h: f32,
    ) {
        self.upload_inner(device, queue, panels, popup, None, None, screen_w, screen_h);
    }

    /// Full-fat upload that also renders a hovering tooltip on top.
    /// Port of the `CMatrixHint::DrawAll` pass at MatrixHint.cpp:
    /// 830-862 — stacked last so the hint's chrome + text/icons
    /// always sit above panels + popup.
    pub fn upload_with_popup_and_hint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        panels: &[&CInterface],
        popup: Option<&super::iface_menu::CIFaceMenu>,
        hint: Option<&Hint>,
        chrome: Option<&HintChromeLibrary>,
        screen_w: f32,
        screen_h: f32,
    ) {
        self.upload_inner(device, queue, panels, popup, hint, chrome, screen_w, screen_h);
    }

    /// Compatibility shim — same as upload_with_popup with no popup.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        panels: &[&CInterface],
        screen_w: f32,
        screen_h: f32,
    ) {
        self.upload_inner(device, queue, panels, None, None, None, screen_w, screen_h);
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_inner(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        panels: &[&CInterface],
        popup: Option<&super::iface_menu::CIFaceMenu>,
        hint: Option<&Hint>,
        chrome: Option<&HintChromeLibrary>,
        screen_w: f32,
        screen_h: f32,
    ) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                screen_size: [screen_w, screen_h, 0.0, 0.0],
            }),
        );

        let scale = (screen_h / DESIGN_H).max(0.1);
        // Draw order must match the element iteration order inside
        // each panel — that's the configuration-authored back-to-front
        // order the C++ relies on (e.g. `conl`/`conr` backgrounds
        // appear before `pich`/pylons, so pylon icons draw on top).
        // Previous BTreeMap-bucketing-by-atlas broke this: `base_3`
        // (conl) sorted after `base_2` (pylon atlas) and overwrote the
        // pylon icons. Instead we stream quads in element order and
        // coalesce consecutive same-atlas quads into one draw.
        let mut all_verts: Vec<Vertex> = Vec::new();
        self.draw_groups.clear();
        let mut current_key: Option<String> = None;
        let mut current_start: u32 = 0;
        let open_run = |key: &str,
                        all_verts: &Vec<Vertex>,
                        current_key: &mut Option<String>,
                        current_start: &mut u32,
                        draw_groups: &mut Vec<DrawGroup>| {
            if current_key.as_deref() != Some(key) {
                if let Some(k) = current_key.take() {
                    let end = all_verts.len() as u32;
                    if end > *current_start {
                        draw_groups.push(DrawGroup {
                            atlas_key: k,
                            start: *current_start,
                            count: end - *current_start,
                        });
                    }
                }
                *current_key = Some(key.to_string());
                *current_start = all_verts.len() as u32;
            }
        };
        let mut per_panel_counts: Vec<(String, u32)> = Vec::new();
        // Per-panel tally of elements whose atlas *failed to load*
        // (or wasn't preloaded) so the element silently skips drawing.
        let mut per_panel_missing: std::collections::BTreeMap<String, u32> =
            std::collections::BTreeMap::new();
        for panel in panels {
            if !panel.visible {
                continue;
            }
            let [px, py] = panel.resolved_pos(screen_w, screen_h, scale);
            let mut n_visible = 0u32;
            for elem in &panel.elements {
                if !elem.visible() {
                    continue;
                }
                let Some(img) = elem.current_image() else {
                    continue;
                };
                let key = normalise_atlas_key(&img.tex_path);
                if !self.atlases.contains_key(&key) {
                    *per_panel_missing
                        .entry(format!("{}::{}", panel.name, key))
                        .or_insert(0) += 1;
                    continue;
                }
                n_visible += 1;
                let [x, y, w, h] = elem.rect_in_panel([px, py], scale);
                // Half-pixel UV inset. The atlas textures aren't edge-
                // padded: atlas col 511 (just outside mp1's sub-rect)
                // and col 175 (just outside mp2's) are fully
                // transparent. Linear filtering at UV=img.x/tex_w or
                // (img.x+img.w)/tex_w lands exactly on the sub-rect
                // edge, so the sampler blends 50/50 with the neighbour
                // column — making mp1's rightmost ~1 px and mp2's
                // leftmost ~1 px render at half alpha. Stacked at the
                // mp1/mp2 screen junction this shows the map through a
                // 1–2 px seam. Inset by 0.5 px so the filter stays
                // inside the sub-rect.
                let u0 = (img.x + 0.5) / img.tex_w;
                let v0 = (img.y + 0.5) / img.tex_h;
                let u1 = (img.x + img.w - 0.5) / img.tex_w;
                let v1 = (img.y + img.h - 0.5) / img.tex_h;
                let tint = match elem.cur_state {
                    ElementState::Focused => [1.0, 1.0, 1.0, 1.0],
                    ElementState::Pressed => [0.8, 0.8, 0.8, 1.0],
                    ElementState::Disabled => [0.5, 0.5, 0.5, 0.8],
                    ElementState::Normal => [1.0, 1.0, 1.0, 1.0],
                };
                open_run(
                    &key,
                    &all_verts,
                    &mut current_key,
                    &mut current_start,
                    &mut self.draw_groups,
                );
                all_verts.extend_from_slice(&[
                    Vertex {
                        pos: [x, y],
                        uv: [u0, v0],
                        tint,
                    },
                    Vertex {
                        pos: [x + w, y],
                        uv: [u1, v0],
                        tint,
                    },
                    Vertex {
                        pos: [x, y + h],
                        uv: [u0, v1],
                        tint,
                    },
                    Vertex {
                        pos: [x + w, y],
                        uv: [u1, v0],
                        tint,
                    },
                    Vertex {
                        pos: [x + w, y + h],
                        uv: [u1, v1],
                        tint,
                    },
                    Vertex {
                        pos: [x, y + h],
                        uv: [u0, v1],
                        tint,
                    },
                ]);
            }
            per_panel_counts.push((panel.name.clone(), n_visible));
        }

        // ── Popup overlay ─────────────────────────────────────────
        // Port of CIFaceMenu::Render. Draws (in back-to-front order):
        //   1. ramka (9-slice border from the PopupMenu panel),
        //   2. selector bar over the hovered row,
        //   3. row icons (per-row sprites borrowed from Base panel),
        //   4. cursik arrow at `current_pos`.
        if let Some(popup) = popup {
            let popup_panel = panels.iter().find(|p| p.name == "PopupMenu");
            let base_panel = panels.iter().find(|p| p.name == "Base");
            if let Some(base_panel) = base_panel {
                let [bx, by] = base_panel.resolved_pos(screen_w, screen_h, scale);
                // Origin of the popup rect, screen pixels.
                let ox = bx + popup.design_x * scale;
                let oy = by + popup.design_y * scale;
                let total_w = popup.item_w * scale;
                let total_h = popup.item_h * popup.items.len() as f32 * scale;

                // Helper to emit one textured quad — borrows an atlas
                // sub-rect from an arbitrary panel element. Used by the
                // chrome (PopupMenu source) and the icons (Base source).
                // Streams into the same vertex buffer as regular panels,
                // so popup quads always stack on top.
                let atlases = &self.atlases;
                let emit_textured = |all_verts: &mut Vec<Vertex>,
                                     current_key: &mut Option<String>,
                                     current_start: &mut u32,
                                     draw_groups: &mut Vec<DrawGroup>,
                                     x: f32,
                                     y: f32,
                                     w: f32,
                                     h: f32,
                                     img: &super::iface_element::StateImage,
                                     tint: [f32; 4]| {
                    let key = normalise_atlas_key(&img.tex_path);
                    if !atlases.contains_key(&key) {
                        return;
                    }
                    let u0 = img.x / img.tex_w;
                    let v0 = img.y / img.tex_h;
                    let u1 = (img.x + img.w) / img.tex_w;
                    let v1 = (img.y + img.h) / img.tex_h;
                    if current_key.as_deref() != Some(&key) {
                        if let Some(k) = current_key.take() {
                            let end = all_verts.len() as u32;
                            if end > *current_start {
                                draw_groups.push(DrawGroup {
                                    atlas_key: k,
                                    start: *current_start,
                                    count: end - *current_start,
                                });
                            }
                        }
                        *current_key = Some(key.clone());
                        *current_start = all_verts.len() as u32;
                    }
                    all_verts.extend_from_slice(&[
                        Vertex {
                            pos: [x, y],
                            uv: [u0, v0],
                            tint,
                        },
                        Vertex {
                            pos: [x + w, y],
                            uv: [u1, v0],
                            tint,
                        },
                        Vertex {
                            pos: [x, y + h],
                            uv: [u0, v1],
                            tint,
                        },
                        Vertex {
                            pos: [x + w, y],
                            uv: [u1, v0],
                            tint,
                        },
                        Vertex {
                            pos: [x + w, y + h],
                            uv: [u1, v1],
                            tint,
                        },
                        Vertex {
                            pos: [x, y + h],
                            uv: [u0, v1],
                            tint,
                        },
                    ]);
                };

                // (1) Chrome 9-slice — port of CIFaceMenu::CreateMenu's
                // ramka assembly (CIFaceMenu.cpp:124-183) using the
                // loaded `if/PopupMenu` source elements: topleft /
                // topright / bottomleft / bottomright corners, and
                // topline / bottomline / leftline / rightline edges
                // stretched to fill. The C++ bakes these into a single
                // texture; we draw them as 9 quads each frame.
                let pop_img = |name: &str| -> Option<&super::iface_element::StateImage> {
                    popup_panel?
                        .elements
                        .iter()
                        .find(|e| e.name == name)
                        .and_then(|e| e.images.first()?.as_ref())
                };
                let opaque = [1.0, 1.0, 1.0, 1.0];
                if let Some(tl) = pop_img("topleft") {
                    let cw = tl.w * scale;
                    let ch = tl.h * scale;
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox,
                        oy,
                        cw,
                        ch,
                        tl,
                        opaque,
                    );
                }
                if let Some(tr) = pop_img("topright") {
                    let cw = tr.w * scale;
                    let ch = tr.h * scale;
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox + total_w - cw,
                        oy,
                        cw,
                        ch,
                        tr,
                        opaque,
                    );
                }
                if let Some(bl) = pop_img("bottomleft") {
                    let cw = bl.w * scale;
                    let ch = bl.h * scale;
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox,
                        oy + total_h - ch,
                        cw,
                        ch,
                        bl,
                        opaque,
                    );
                }
                if let Some(br) = pop_img("bottomright") {
                    let cw = br.w * scale;
                    let ch = br.h * scale;
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox + total_w - cw,
                        oy + total_h - ch,
                        cw,
                        ch,
                        br,
                        opaque,
                    );
                }
                // Edges — `topline` / `bottomline` are 1px wide source
                // sprites stretched horizontally between the corners;
                // `leftline` / `rightline` are 1px tall stretched
                // vertically.
                let corner_h = pop_img("topleft").map(|i| i.h * scale).unwrap_or(0.0);
                let corner_h_b = pop_img("bottomleft").map(|i| i.h * scale).unwrap_or(0.0);
                let corner_w = pop_img("topleft").map(|i| i.w * scale).unwrap_or(0.0);
                let corner_w_r = pop_img("topright").map(|i| i.w * scale).unwrap_or(0.0);
                if let Some(tline) = pop_img("topline") {
                    let h = tline.h * scale;
                    let span = (total_w - corner_w - corner_w_r).max(0.0);
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox + corner_w,
                        oy,
                        span,
                        h,
                        tline,
                        opaque,
                    );
                }
                if let Some(bline) = pop_img("bottomline") {
                    let h = bline.h * scale;
                    let span = (total_w - corner_w - corner_w_r).max(0.0);
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox + corner_w,
                        oy + total_h - h,
                        span,
                        h,
                        bline,
                        opaque,
                    );
                }
                if let Some(lline) = pop_img("leftline") {
                    let w = lline.w * scale;
                    let span = (total_h - corner_h - corner_h_b).max(0.0);
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox,
                        oy + corner_h,
                        w,
                        span,
                        lline,
                        opaque,
                    );
                }
                if let Some(rline) = pop_img("rightline") {
                    let w = rline.w * scale;
                    let span = (total_h - corner_h - corner_h_b).max(0.0);
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox + total_w - w,
                        oy + corner_h,
                        w,
                        span,
                        rline,
                        opaque,
                    );
                }

                // (2) Selector bar — port of the `m_Selector` element
                // (CIFaceMenu.cpp:268-297). Drawn over the hovered row
                // when any. Sourced from the `sel` element on PopupMenu;
                // falls back to a simple tint if the asset is missing.
                if let Some(hi) = popup.hovered {
                    let row_y = oy + hi as f32 * popup.item_h * scale;
                    let row_h = popup.item_h * scale;
                    if let Some(sel) = pop_img("sel") {
                        emit_textured(
                            &mut all_verts,
                            &mut current_key,
                            &mut current_start,
                            &mut self.draw_groups,
                            ox + corner_w,
                            row_y,
                            (total_w - corner_w - corner_w_r).max(0.0),
                            row_h,
                            sel,
                            [1.0, 1.0, 1.0, 0.7],
                        );
                    }
                }

                // (3) Row icons — resolved via the Base panel's
                // `template_by_kind` so the element whose `Param2`
                // matches the item's `kind` is used. The template-
                // button names (chas1, weap1, …) DON'T line up with
                // kinds 1:1 (e.g. `weap7` is the MACHINEGUN template,
                // not `weap1`) — see CInterface.cpp:338-387 for how the
                // C++ builds the same lookup.
                let popup_ty = popup.parent.unit_type() as i32;
                for (i, item) in popup.items.iter().enumerate() {
                    let Some(src) = base_panel.template_by_kind(popup_ty, item.kind.0) else {
                        continue;
                    };
                    let Some(img) = src.images.first().and_then(|x| x.as_ref()) else {
                        continue;
                    };
                    let x = ox;
                    let y = oy + i as f32 * popup.item_h * scale;
                    let w = popup.item_w * scale;
                    let h = popup.item_h * scale;
                    // Hovered row gets the brightened tint; others
                    // dimmed slightly so the active selection pops.
                    // Unaffordable items render in the C++
                    // `NERES_LABELS_COLOR` reddish-grey (CIFaceButton.cpp
                    // :198 etc.) — we approximate via a red-shifted dim.
                    let tint = if !item.affordable {
                        if popup.hovered == Some(i) {
                            [0.85, 0.35, 0.35, 1.0]
                        } else {
                            [0.55, 0.20, 0.20, 1.0]
                        }
                    } else if popup.hovered == Some(i) {
                        [1.0, 1.0, 1.0, 1.0]
                    } else {
                        [0.7, 0.7, 0.7, 1.0]
                    };
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        x,
                        y,
                        w,
                        h,
                        img,
                        tint,
                    );
                }

                // (4) Cursik — the arrow indicator at the row matching
                // the currently-equipped item. Port of `m_CursikImage` /
                // `cursik_hpos` placement at CIFaceMenu.cpp:94-96, 217.
                if let Some(pos) = popup.current_pos {
                    if let Some(cursik) = pop_img("cursik") {
                        let cw = cursik.w * scale;
                        let ch = cursik.h * scale;
                        let row_y = oy + pos as f32 * popup.item_h * scale;
                        // C++ places it at LEFT_SPACE + 1 (8px). Centred
                        // vertically in the row.
                        let cx = ox + 8.0 * scale;
                        let cy = row_y + (popup.item_h * scale - ch) * 0.5;
                        emit_textured(
                            &mut all_verts,
                            &mut current_key,
                            &mut current_start,
                            &mut self.draw_groups,
                            cx,
                            cy,
                            cw,
                            ch,
                            cursik,
                            opaque,
                        );
                    }
                }
            }
        }

        // ── Text pass ─────────────────────────────────────────────
        // After all image + popup quads, lay out AFT glyphs for every
        // visible element's `current_labels()` rows. Each glyph emits
        // one textured quad sourced from the shared glyph atlas; the
        // existing pipeline picks them up via the TEXT_ATLAS_KEY bind
        // group (uploaded at the end via `sync_glyph_atlas`).
        //
        // AFT fonts are FIXED PIXEL SIZE — they were rasterised at the
        // 1024×768 design-space resolution and don't carry vector data.
        // We scale them as a single unit by `scale`, with point
        // sampling on the text atlas to keep the pixels crisp.
        for panel in panels {
            if !panel.visible {
                continue;
            }
            let [px, py] = panel.resolved_pos(screen_w, screen_h, scale);
            for elem in &panel.elements {
                if !elem.visible() {
                    continue;
                }
                for label in elem.current_labels() {
                    if label.text.is_empty() {
                        continue;
                    }
                    // Anchor is computed in DESIGN-SPACE px first, then
                    // multiplied by `scale` for the screen. Matches the
                    // C++ which also computes alignment on the design-
                    // space rect (m_SX/m_SY).
                    let base_x = match label.align_x {
                        1 => elem.size_x * 0.5,
                        2 => elem.size_x,
                        _ => 0.0,
                    };
                    let base_y = match label.align_y {
                        1 => elem.size_y * 0.5,
                        2 => elem.size_y,
                        _ => 0.0,
                    };
                    let design_anchor_x =
                        elem.pos_x + base_x + label.x + label.sme_x;
                    let design_anchor_y =
                        elem.pos_y + base_y + label.y + label.sme_y;
                    let anchor_x = px + design_anchor_x * scale;
                    let anchor_y = py + design_anchor_y * scale;
                    // Parse inline `<Color=R,G,B>...</color>` tags so
                    // numeric values inserted by `make_item_replacements`
                    // (CConstructor.cpp:1100-1191) render in their
                    // override colour. Drop-shadow rows already pass
                    // black as `label.color`; we don't want the gold
                    // `<Color>` tag to override the shadow, so for any
                    // black-shadow row we drop the override colours.
                    let mut runs = parse_rich_text(&label.text);
                    let is_shadow = label.color == [0, 0, 0, 255];
                    if is_shadow {
                        for r in &mut runs {
                            r.color = None;
                        }
                    }
                    if label.wrap {
                        // Available text width from the label's anchor
                        // out to the element's right edge. `label.x`
                        // is the text origin inside the element;
                        // `label.sme_x` is an extra offset that's
                        // applied at render time too, so it shrinks
                        // the wrap budget by the same amount.
                        let wrap_width =
                            (elem.size_x - label.x - label.sme_x).max(0.0) * scale;
                        emit_wrapped_text(
                            &mut all_verts,
                            &mut current_key,
                            &mut current_start,
                            &mut self.draw_groups,
                            &mut self.glyph_atlas,
                            &label.font,
                            &runs,
                            scale,
                            anchor_x,
                            anchor_y,
                            wrap_width,
                            label.align_x,
                            label.align_y,
                            label.color,
                        );
                    } else {
                        emit_rich_line(
                            &mut all_verts,
                            &mut current_key,
                            &mut current_start,
                            &mut self.draw_groups,
                            &mut self.glyph_atlas,
                            &label.font,
                            &runs,
                            scale,
                            anchor_x,
                            anchor_y,
                            label.align_x,
                            label.align_y,
                            label.color,
                        );
                    }
                }
            }
        }

        // Tooltip overlay — drawn last so it always sits on top.
        // Port of `CMatrixHint::DrawAll` (MatrixHint.cpp:830-862).
        // Layout + 9-slice geometry lives in `hint.rs`; `emit_hint`
        // here is a dumb vertex emitter.
        if let (Some(h), Some(c)) = (hint, chrome) {
            emit_hint(
                &mut all_verts,
                &mut current_key,
                &mut current_start,
                &mut self.draw_groups,
                &mut self.glyph_atlas,
                &self.atlases,
                c,
                h,
                scale,
            );
        }

        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        // Log once on change. We check if the count for any panel
        // differs from the last log.
        static LAST_BASE_COUNT: std::sync::atomic::AtomicU32 =
            std::sync::atomic::AtomicU32::new(u32::MAX);
        let base_count = per_panel_counts
            .iter()
            .find(|(n, _)| n == "Base")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        if LAST_BASE_COUNT.swap(base_count, std::sync::atomic::Ordering::Relaxed) != base_count {
            log::info!(
                "interface upload: screen={}x{} scale={:.2} panels={:?} atlases={:?} missing={:?}",
                screen_w,
                screen_h,
                scale,
                per_panel_counts,
                self.atlases.keys().collect::<Vec<_>>(),
                per_panel_missing,
            );
            LOGGED.store(1, std::sync::atomic::Ordering::Relaxed);
        }

        // Flush the trailing run that was still open when we finished
        // iterating panel + popup quads.
        if let Some(k) = current_key.take() {
            let end = all_verts.len() as u32;
            if end > current_start {
                self.draw_groups.push(DrawGroup {
                    atlas_key: k,
                    start: current_start,
                    count: end - current_start,
                });
            }
        }

        if all_verts.len() > self.vertex_capacity {
            // Grow the vertex buffer to hold every emitted quad —
            // glyph quads can easily push past the original 6 KB
            // capacity once a few captions render. Re-create the
            // buffer at the next power of two and re-upload the
            // current frame's data.
            let new_cap = all_verts.len().next_power_of_two();
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Interface VB (grown)"),
                size: (new_cap * std::mem::size_of::<Vertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            log::info!(
                "interface: VB grown to {} verts (was {})",
                new_cap,
                self.vertex_capacity
            );
            self.vertex_capacity = new_cap;
        }
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&all_verts));
        self.num_verts = all_verts.len() as u32;

        // Re-upload the glyph atlas if any new glyphs landed in it
        // during this frame's text pass. Must happen AFTER text quads
        // are streamed but BEFORE the render pass binds the atlas BG.
        self.sync_glyph_atlas(device, queue);
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.num_verts == 0 || self.draw_groups.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniform_bg, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        for g in &self.draw_groups {
            let Some(atlas) = self.atlases.get(&g.atlas_key) else {
                continue;
            };
            pass.set_bind_group(1, &atlas.bind_group, &[]);
            pass.draw(g.start..g.start + g.count, 0..1);
        }
    }
}

/// Case/slash-normalised key for an atlas path. All variants
/// (`Matrix\\IFace\\interface2`, `Matrix/Iface/interface2`,
/// `MATRIX/IFACE/INTERFACE2`) collapse to the same string.
fn normalise_atlas_key(path: &str) -> String {
    path.replace('\\', "/").to_lowercase()
}

/// Emit one hint's draw commands. Visual logic (9-slice layout,
/// part positioning) already lives in `hint.rs`; this function is a
/// dumb vertex emitter that:
///   1. Opens a run keyed by the border atlas path, streams
///      `hint.slice_draws(chrome, scale)`.
///   2. For each `HintPart`: text parts go through the glyph atlas
///      (`emit_rich_line`), bitmap parts open a run keyed by the
///      bitmap's PNG path and stream one textured quad.
fn emit_hint(
    all_verts: &mut Vec<Vertex>,
    current_key: &mut Option<String>,
    current_start: &mut u32,
    draw_groups: &mut Vec<DrawGroup>,
    glyph_atlas: &mut GlyphAtlas,
    atlases: &HashMap<String, Atlas>,
    chrome: &HintChromeLibrary,
    hint: &Hint,
    scale: f32,
) {
    // ── 9-slice chrome (layout produced by `Hint::slice_draws`) ─────
    if let Some(path) = hint.border_source_path(chrome) {
        let atlas_key = normalise_atlas_key(path);
        if let Some(atlas) = atlases.get(&atlas_key) {
            let tw = atlas.width as f32;
            let th = atlas.height as f32;
            open_run(&atlas_key, current_key, current_start, all_verts, draw_groups);
            for d in hint.slice_draws(chrome, scale) {
                let u0 = d.tex_x as f32 / tw;
                let v0 = d.tex_y as f32 / th;
                let u1 = (d.tex_x + d.tex_w) as f32 / tw;
                let v1 = (d.tex_y + d.tex_h) as f32 / th;
                let (x, y, w, h) = (d.screen_x, d.screen_y, d.screen_w, d.screen_h);
                all_verts.extend_from_slice(&[
                    Vertex { pos: [x, y], uv: [u0, v0], tint: [1.0; 4] },
                    Vertex { pos: [x + w, y], uv: [u1, v0], tint: [1.0; 4] },
                    Vertex { pos: [x, y + h], uv: [u0, v1], tint: [1.0; 4] },
                    Vertex { pos: [x + w, y], uv: [u1, v0], tint: [1.0; 4] },
                    Vertex { pos: [x + w, y + h], uv: [u1, v1], tint: [1.0; 4] },
                    Vertex { pos: [x, y + h], uv: [u0, v1], tint: [1.0; 4] },
                ]);
            }
        }
    }

    // ── Parts (text + inline bitmaps) ───────────────────────────────
    for part in &hint.parts {
        match part {
            HintPart::Text { x, y, text, font, color, .. } => {
                if text.is_empty() {
                    continue;
                }
                // One `_TEXT:` can span multiple rendered lines
                // (explicit `<br>` or wrap). The hint layout already
                // picked the per-line widths; here we render each
                // `\n`-separated sub-line at its own vertical stride.
                let line_stride_px =
                    (glyph_atlas.line_height(font) as f32 + 1.0) * scale;
                let base_x = hint.screen_x + *x as f32 * scale;
                let base_y = hint.screen_y + *y as f32 * scale;
                for (line_idx, line) in text.split('\n').enumerate() {
                    if line.is_empty() {
                        continue;
                    }
                    let runs = parse_rich_text(line);
                    emit_rich_line(
                        all_verts,
                        current_key,
                        current_start,
                        draw_groups,
                        glyph_atlas,
                        font,
                        &runs,
                        scale,
                        base_x,
                        base_y + line_idx as f32 * line_stride_px,
                        0,
                        0,
                        *color,
                    );
                }
            }
            HintPart::Bitmap { x, y, w, h, name } => {
                let Some(bmp) = chrome.bitmaps.get(name) else {
                    continue;
                };
                let key = normalise_atlas_key(&bmp.path);
                if !atlases.contains_key(&key) {
                    continue;
                }
                open_run(&key, current_key, current_start, all_verts, draw_groups);
                let px = hint.screen_x + *x as f32 * scale;
                let py = hint.screen_y + *y as f32 * scale;
                let pw = *w as f32 * scale;
                let ph = *h as f32 * scale;
                all_verts.extend_from_slice(&[
                    Vertex { pos: [px, py], uv: [0.0, 0.0], tint: [1.0; 4] },
                    Vertex { pos: [px + pw, py], uv: [1.0, 0.0], tint: [1.0; 4] },
                    Vertex { pos: [px, py + ph], uv: [0.0, 1.0], tint: [1.0; 4] },
                    Vertex { pos: [px + pw, py], uv: [1.0, 0.0], tint: [1.0; 4] },
                    Vertex { pos: [px + pw, py + ph], uv: [1.0, 1.0], tint: [1.0; 4] },
                    Vertex { pos: [px, py + ph], uv: [0.0, 1.0], tint: [1.0; 4] },
                ]);
            }
        }
    }
}

/// Open (or extend) a draw run keyed by `key`. Closes the previous
/// run if the key changes. Called before streaming any vertices that
/// sample a particular atlas.
fn open_run(
    key: &str,
    current_key: &mut Option<String>,
    current_start: &mut u32,
    all_verts: &Vec<Vertex>,
    draw_groups: &mut Vec<DrawGroup>,
) {
    if current_key.as_deref() == Some(key) {
        return;
    }
    if let Some(k) = current_key.take() {
        let end = all_verts.len() as u32;
        if end > *current_start {
            draw_groups.push(DrawGroup {
                atlas_key: k,
                start: *current_start,
                count: end - *current_start,
            });
        }
    }
    *current_key = Some(key.to_string());
    *current_start = all_verts.len() as u32;
}

/// Lay out a sequence of [`RichRun`]s at `(anchor_x, anchor_y)` in
/// screen pixels using the AFT bitmap glyphs of `font`, scaled by
/// `scale`. Each run uses its own colour override (or `base_color`
/// when none). One textured quad per non-empty glyph.
///
/// `align_x`: 0=left, 1=centre, 2=right of the WHOLE line.
/// `align_y`: 0=top of cell, 1=cell middle, 2=bottom of cell.
#[allow(clippy::too_many_arguments)]
fn emit_rich_line(
    all_verts: &mut Vec<Vertex>,
    current_key: &mut Option<String>,
    current_start: &mut u32,
    draw_groups: &mut Vec<DrawGroup>,
    glyph_atlas: &mut GlyphAtlas,
    font: &str,
    runs: &[RichRun],
    scale: f32,
    anchor_x: f32,
    anchor_y: f32,
    align_x: i32,
    align_y: i32,
    base_color: [u8; 4],
) {
    let key = TEXT_ATLAS_KEY;
    let atlas_w = glyph_atlas.width() as f32;
    let atlas_h = glyph_atlas.height() as f32;
    // Total advance is the sum across all runs.
    let total_w_native: u32 = runs
        .iter()
        .map(|r| glyph_atlas.measure(font, &r.text))
        .sum();
    let line_h_native = glyph_atlas.line_height(font) as f32;
    let ascent_native = glyph_atlas.ascent(font) as f32;
    let total_w = total_w_native as f32 * scale;
    let line_h = line_h_native * scale;

    let start_x = match align_x {
        1 => anchor_x - total_w * 0.5,
        2 => anchor_x - total_w,
        _ => anchor_x,
    };
    let cell_top = match align_y {
        1 => anchor_y - line_h * 0.5,
        2 => anchor_y - line_h,
        _ => anchor_y,
    };
    // Baseline sits `ascent` pixels below the cell top. AFT bearing_y
    // is signed offset from baseline to glyph TOP (negative = above
    // baseline), so the on-screen top row of the glyph quad is
    // baseline + bearing_y * scale.
    let baseline_y = cell_top + ascent_native * scale;

    // Open / extend a TEXT_ATLAS_KEY draw run.
    if current_key.as_deref() != Some(key) {
        if let Some(k) = current_key.take() {
            let end = all_verts.len() as u32;
            if end > *current_start {
                draw_groups.push(DrawGroup {
                    atlas_key: k,
                    start: *current_start,
                    count: end - *current_start,
                });
            }
        }
        *current_key = Some(key.to_string());
        *current_start = all_verts.len() as u32;
    }

    let mut pen_x_native = 0.0_f32;
    for run in runs {
        let color = run.color.unwrap_or(base_color);
        let tint = [
            color[0] as f32 / 255.0,
            color[1] as f32 / 255.0,
            color[2] as f32 / 255.0,
            color[3] as f32 / 255.0,
        ];
        for c in run.text.chars() {
            let Some(g) = glyph_atlas.glyph(font, c as u32) else {
                continue;
            };
            let advance = g.advance as f32;
            if g.w == 0 || g.h == 0 {
                pen_x_native += advance;
                continue;
            }
            let x = start_x + (pen_x_native + g.bearing_x as f32) * scale;
            let y = baseline_y + g.bearing_y as f32 * scale;
            let w = g.w as f32 * scale;
            let h = g.h as f32 * scale;
            let u0 = g.atlas_x as f32 / atlas_w;
            let v0 = g.atlas_y as f32 / atlas_h;
            let u1 = (g.atlas_x + g.w) as f32 / atlas_w;
            let v1 = (g.atlas_y + g.h) as f32 / atlas_h;
            all_verts.extend_from_slice(&[
                Vertex { pos: [x, y], uv: [u0, v0], tint },
                Vertex { pos: [x + w, y], uv: [u1, v0], tint },
                Vertex { pos: [x, y + h], uv: [u0, v1], tint },
                Vertex { pos: [x + w, y], uv: [u1, v0], tint },
                Vertex { pos: [x + w, y + h], uv: [u1, v1], tint },
                Vertex { pos: [x, y + h], uv: [u0, v1], tint },
            ]);
            pen_x_native += advance;
        }
    }
}

/// Multi-line layout — port of the `m_WordWrap` (perenos) branch of
/// the original `m_RangersText` call. Tokenises rich runs into
/// (word, color) tokens, greedy-packs them into lines, and emits each
/// line via `emit_rich_line` so colour overrides + alignment match
/// the non-wrapped path exactly.
///
/// Source-side newlines (`\r\n`, from upstream `<br>` substitution) are
/// honoured as forced line breaks.
#[allow(clippy::too_many_arguments)]
fn emit_wrapped_text(
    all_verts: &mut Vec<Vertex>,
    current_key: &mut Option<String>,
    current_start: &mut u32,
    draw_groups: &mut Vec<DrawGroup>,
    glyph_atlas: &mut GlyphAtlas,
    font: &str,
    runs: &[RichRun],
    scale: f32,
    anchor_x: f32,
    anchor_y: f32,
    wrap_width: f32,
    align_x: i32,
    align_y: i32,
    base_color: [u8; 4],
) {
    let wrap_native = if scale > 0.0 { wrap_width / scale } else { wrap_width };
    let lines = wrap_rich_lines(glyph_atlas, font, runs, wrap_native as u32);
    if lines.is_empty() {
        return;
    }
    // 1 px of leading on top of (ascent + descent) so descenders on
    // one row don't kiss ascenders on the next.
    let line_h = (glyph_atlas.line_height(font) as f32 + 1.0) * scale;
    let block_h = line_h * lines.len() as f32;
    let block_top = match align_y {
        1 => anchor_y - block_h * 0.5,
        2 => anchor_y - block_h,
        _ => anchor_y,
    };
    for (i, line) in lines.iter().enumerate() {
        let line_anchor_y = block_top + i as f32 * line_h;
        emit_rich_line(
            all_verts,
            current_key,
            current_start,
            draw_groups,
            glyph_atlas,
            font,
            line,
            scale,
            anchor_x,
            line_anchor_y,
            align_x,
            0,
            base_color,
        );
    }
}

/// Greedy word-wrap on AFT native pixel widths, preserving per-run
/// colours. Each output line is a fresh `Vec<RichRun>` whose runs
/// alternate colours as the input does.
///
/// Tokenisation rule: split each input run on whitespace; whitespace
/// itself becomes a separate run carrying the same colour (so a wrap
/// at the space position discards it cleanly). Forced breaks come
/// from `\r\n` (or `\n`) characters in the input.
fn wrap_rich_lines(
    atlas: &mut GlyphAtlas,
    font: &str,
    runs: &[RichRun],
    wrap_width_px: u32,
) -> Vec<Vec<RichRun>> {
    // Tokenise the rich stream into (word_or_space, color, hard_break).
    enum Tok {
        Word { text: String, color: Option<[u8; 4]> },
        Space { color: Option<[u8; 4]> },
        Break,
    }
    let mut toks: Vec<Tok> = Vec::new();
    for run in runs {
        let mut buf = String::new();
        for c in run.text.chars() {
            if c == '\n' {
                if !buf.is_empty() {
                    toks.push(Tok::Word { text: std::mem::take(&mut buf), color: run.color });
                }
                toks.push(Tok::Break);
            } else if c == '\r' {
                if !buf.is_empty() {
                    toks.push(Tok::Word { text: std::mem::take(&mut buf), color: run.color });
                }
                // Skip CR; LF will trigger Break.
            } else if c.is_whitespace() {
                if !buf.is_empty() {
                    toks.push(Tok::Word { text: std::mem::take(&mut buf), color: run.color });
                }
                toks.push(Tok::Space { color: run.color });
            } else {
                buf.push(c);
            }
        }
        if !buf.is_empty() {
            toks.push(Tok::Word { text: buf, color: run.color });
        }
    }

    let space_w = atlas.measure(font, " ");
    let mut lines: Vec<Vec<RichRun>> = Vec::new();
    let mut current: Vec<RichRun> = Vec::new();
    let mut current_w: u32 = 0;

    let push_line = |current: &mut Vec<RichRun>, current_w: &mut u32, lines: &mut Vec<Vec<RichRun>>| {
        // Trim trailing spaces from the line being closed.
        while let Some(last) = current.last() {
            if last.text.chars().all(|c| c == ' ') {
                current.pop();
            } else {
                break;
            }
        }
        if !current.is_empty() {
            lines.push(std::mem::take(current));
        } else {
            // Preserve empty lines from explicit breaks.
            lines.push(Vec::new());
        }
        *current_w = 0;
    };

    for tok in toks {
        match tok {
            Tok::Break => push_line(&mut current, &mut current_w, &mut lines),
            Tok::Space { color } => {
                if current.is_empty() {
                    continue; // skip leading spaces
                }
                if wrap_width_px > 0 && current_w + space_w > wrap_width_px {
                    push_line(&mut current, &mut current_w, &mut lines);
                } else {
                    current.push(RichRun { text: " ".to_string(), color });
                    current_w += space_w;
                }
            }
            Tok::Word { text, color } => {
                let word_w = atlas.measure(font, &text);
                if wrap_width_px > 0 && word_w > wrap_width_px && current.is_empty() {
                    // Char-break the oversize word.
                    let chunks = split_oversize_word(atlas, font, &text, wrap_width_px);
                    let mut iter = chunks.into_iter().peekable();
                    while let Some(chunk) = iter.next() {
                        let cw = atlas.measure(font, &chunk);
                        current.push(RichRun { text: chunk, color });
                        current_w += cw;
                        if iter.peek().is_some() {
                            push_line(&mut current, &mut current_w, &mut lines);
                        }
                    }
                } else if wrap_width_px > 0 && current_w + word_w > wrap_width_px {
                    push_line(&mut current, &mut current_w, &mut lines);
                    current.push(RichRun { text, color });
                    current_w = word_w;
                } else {
                    current.push(RichRun { text, color });
                    current_w += word_w;
                }
            }
        }
    }
    if !current.is_empty() {
        // Trim trailing spaces.
        while let Some(last) = current.last() {
            if last.text.chars().all(|c| c == ' ') {
                current.pop();
            } else {
                break;
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
    }
    lines
}

/// Char-by-char split a single word that exceeds the wrap width.
fn split_oversize_word(
    atlas: &mut GlyphAtlas,
    font: &str,
    word: &str,
    wrap_width_px: u32,
) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_w: u32 = 0;
    for c in word.chars() {
        let g = atlas.glyph(font, c as u32);
        let advance = g.map(|g| g.advance).unwrap_or(0);
        if !current.is_empty() && current_w + advance > wrap_width_px {
            chunks.push(std::mem::take(&mut current));
            current_w = 0;
        }
        current.push(c);
        current_w += advance;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

const SHADER: &str = include_str!("../../../shaders/interface.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    fn r(text: &str) -> Vec<RichRun> {
        vec![RichRun { text: text.to_string(), color: None }]
    }

    fn line_text(line: &[RichRun]) -> String {
        line.iter().map(|r| r.text.as_str()).collect()
    }

    #[test]
    fn wrap_short_text_is_one_line() {
        let mut a = GlyphAtlas::new();
        let lines = wrap_rich_lines(&mut a, "Font.2Ranger", &r("Hi"), 1000);
        assert_eq!(lines.len(), 1);
        assert_eq!(line_text(&lines[0]), "Hi");
    }

    #[test]
    fn wrap_breaks_on_word_boundary() {
        let mut a = GlyphAtlas::new();
        let lines = wrap_rich_lines(&mut a, "Font.2Ranger", &r("alpha beta gamma delta"), 60);
        assert!(lines.len() >= 2);
        let joined: String = lines.iter().map(|l| line_text(l)).collect::<Vec<_>>().join(" ");
        assert_eq!(joined.split_whitespace().count(), 4);
    }

    #[test]
    fn wrap_explicit_break() {
        let mut a = GlyphAtlas::new();
        let lines = wrap_rich_lines(&mut a, "Font.2Ranger", &r("first\r\nsecond"), 1000);
        assert_eq!(lines.len(), 2);
        assert_eq!(line_text(&lines[0]), "first");
        assert_eq!(line_text(&lines[1]), "second");
    }

    #[test]
    fn wrap_preserves_color_runs() {
        let mut a = GlyphAtlas::new();
        let runs = vec![
            RichRun { text: "Damage ".to_string(), color: None },
            RichRun { text: "30".to_string(), color: Some([247, 195, 0, 255]) },
            RichRun { text: " HP".to_string(), color: None },
        ];
        let lines = wrap_rich_lines(&mut a, "Font.2Ranger", &runs, 1000);
        assert_eq!(lines.len(), 1);
        // The gold "30" run should still carry its color override.
        let gold_count = lines[0]
            .iter()
            .filter(|r| r.color == Some([247, 195, 0, 255]))
            .count();
        assert!(gold_count > 0, "color override lost during wrap: {:?}", lines[0]);
    }

    #[test]
    fn wrap_oversize_word_is_char_split() {
        let mut a = GlyphAtlas::new();
        let lines = wrap_rich_lines(&mut a, "Font.2Ranger", &r("supercalifragilistic"), 30);
        assert!(lines.len() > 1);
        let joined: String = lines.iter().map(|l| line_text(l)).collect();
        assert_eq!(joined, "supercalifragilistic");
    }

    #[test]
    fn wrap_zero_width_keeps_input() {
        let mut a = GlyphAtlas::new();
        let lines = wrap_rich_lines(&mut a, "Font.2Ranger", &r("anything"), 0);
        assert_eq!(lines.len(), 1);
        assert_eq!(line_text(&lines[0]), "anything");
    }
}
