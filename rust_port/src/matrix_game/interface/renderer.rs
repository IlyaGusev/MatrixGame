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

use super::iface_element::ElementState;
use super::interface::{CInterface, DESIGN_H};

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
/// the lifetime of the renderer.
struct Atlas {
    _tex: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
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

    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    atlases: HashMap<String, Atlas>,
    draw_groups: Vec<DrawGroup>,
    num_verts: u32,
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
            vertex_buffer,
            vertex_capacity,
            atlases: HashMap::new(),
            draw_groups: Vec::new(),
            num_verts: 0,
        }
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
            },
        );
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
        queue: &wgpu::Queue,
        panels: &[&CInterface],
        popup: Option<&super::face_menu::CIFaceMenu>,
        screen_w: f32,
        screen_h: f32,
    ) {
        self.upload_inner(queue, panels, popup, screen_w, screen_h);
    }

    /// Compatibility shim — same as upload_with_popup with no popup.
    pub fn upload(
        &mut self,
        queue: &wgpu::Queue,
        panels: &[&CInterface],
        screen_w: f32,
        screen_h: f32,
    ) {
        self.upload_inner(queue, panels, None, screen_w, screen_h);
    }

    fn upload_inner(
        &mut self,
        queue: &wgpu::Queue,
        panels: &[&CInterface],
        popup: Option<&super::face_menu::CIFaceMenu>,
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
                let u0 = img.x / img.tex_w;
                let v0 = img.y / img.tex_h;
                let u1 = (img.x + img.w) / img.tex_w;
                let v1 = (img.y + img.h) / img.tex_h;
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
                        .and_then(|e| e.images.get(0)?.as_ref())
                };
                let opaque = [1.0, 1.0, 1.0, 1.0];
                if let Some(tl) = pop_img("topleft") {
                    let cw = tl.w * scale;
                    let ch = tl.h * scale;
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups, ox, oy, cw, ch, tl, opaque);
                }
                if let Some(tr) = pop_img("topright") {
                    let cw = tr.w * scale;
                    let ch = tr.h * scale;
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups, ox + total_w - cw, oy, cw, ch, tr, opaque);
                }
                if let Some(bl) = pop_img("bottomleft") {
                    let cw = bl.w * scale;
                    let ch = bl.h * scale;
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups, ox, oy + total_h - ch, cw, ch, bl, opaque);
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
                        &mut self.draw_groups, ox + corner_w, oy, span, h, tline, opaque);
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
                        &mut self.draw_groups, ox, oy + corner_h, w, span, lline, opaque);
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
                    let Some(img) = src.images.get(0).and_then(|x| x.as_ref()) else {
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
                        &mut self.draw_groups, x, y, w, h, img, tint);
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
                        &mut self.draw_groups, cx, cy, cw, ch, cursik, opaque);
                    }
                }
            }
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
            self.vertex_capacity = all_verts.len().next_power_of_two();
            log::warn!(
                "interface: VB overflow, clamping to {} verts",
                self.vertex_capacity
            );
            all_verts.truncate(self.vertex_capacity);
        }
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&all_verts));
        self.num_verts = all_verts.len() as u32;
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

const SHADER: &str = include_str!("../../../shaders/interface.wgsl");
