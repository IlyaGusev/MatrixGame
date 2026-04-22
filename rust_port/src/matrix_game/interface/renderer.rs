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
use std::collections::{BTreeMap, HashMap};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::matrix_lib::three_g::texture::{create_texture_from_rgba, decode_texture_bytes};

use super::interface::{CInterface, DESIGN_H};
use super::iface_element::ElementState;

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
        let with_iface = fwd.replace("/IFace/", "/Iface/").replace("IFace/", "Iface/");
        let with_lower = fwd.replace("/IFace/", "/iface/").replace("IFace/", "iface/");
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
        log::info!("interface: atlas {} ({} bytes) bound as {}", atlas_path, bytes.len(), key);
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
        for p in panels {
            for e in &p.elements {
                for img in e.images.iter().flatten() {
                    if !img.tex_path.is_empty() {
                        paths.insert(img.tex_path.clone());
                    }
                }
            }
        }
        for path in paths {
            self.ensure_atlas(device, queue, read, &path);
        }
    }

    /// Rebuild the vertex stream + draw-groups from the current panel
    /// snapshot. Designed to run every frame.
    pub fn upload(
        &mut self,
        queue: &wgpu::Queue,
        panels: &[&CInterface],
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
        // Bucket by atlas key. BTreeMap so iteration order is
        // deterministic — a HashMap re-ordered the draw calls each
        // frame, which made alpha-blending non-commutative pixels
        // flicker (visible as "buttons blinking").
        let mut buckets: BTreeMap<String, Vec<Vertex>> = BTreeMap::new();
        for panel in panels {
            if !panel.visible {
                continue;
            }
            let [px, py] = panel.resolved_pos(screen_w, screen_h, scale);
            for elem in &panel.elements {
                if !elem.visible() {
                    continue;
                }
                let Some(img) = elem.current_image() else { continue };
                let key = normalise_atlas_key(&img.tex_path);
                if !self.atlases.contains_key(&key) {
                    continue;
                }
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
                let v = [
                    Vertex { pos: [x, y], uv: [u0, v0], tint },
                    Vertex { pos: [x + w, y], uv: [u1, v0], tint },
                    Vertex { pos: [x, y + h], uv: [u0, v1], tint },
                    Vertex { pos: [x + w, y], uv: [u1, v0], tint },
                    Vertex { pos: [x + w, y + h], uv: [u1, v1], tint },
                    Vertex { pos: [x, y + h], uv: [u0, v1], tint },
                ];
                buckets.entry(key).or_default().extend_from_slice(&v);
            }
        }

        // Concat buckets into one VB, emit a DrawGroup per atlas.
        let mut all_verts: Vec<Vertex> = Vec::new();
        self.draw_groups.clear();
        for (key, verts) in buckets {
            let start = all_verts.len() as u32;
            all_verts.extend_from_slice(&verts);
            self.draw_groups.push(DrawGroup {
                atlas_key: key,
                start,
                count: verts.len() as u32,
            });
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
            let Some(atlas) = self.atlases.get(&g.atlas_key) else { continue };
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
