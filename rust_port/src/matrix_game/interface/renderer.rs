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

/// Atlas key for the 1×1 solid-white pixel texture, used to draw
/// flat-colour quads via the standard textured pipeline (the tint
/// channel multiplies with the texel; an all-white texel passes the
/// tint colour through unchanged). Created in `InterfaceRenderer::new`.
pub const SOLID_ATLAS_KEY: &str = "__solid__";

/// Atlas key for the dynamically-baked popup texture. Re-baked each
/// frame the popup is open so size changes (different popups have
/// different items counts and widths) are picked up automatically.
/// Mirrors the C++ `m_RamTex` (CIFaceMenu.h:90, baked in
/// CIFaceMenu::CreateMenu).
pub const POPUP_BAKED_KEY: &str = "__popup_baked__";

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
///
/// `cpu_pixels` holds the decoded RGBA bytes alongside the GPU view —
/// needed for popup texture baking (`bake_popup`), which has to read
/// individual chrome-sprite pixels and composite them on the CPU
/// exactly as `CIFaceMenu::CreateMenu` does (CIFaceMenu.cpp:117-226).
struct Atlas {
    _tex: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    cpu_pixels: Option<image::RgbaImage>,
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
    /// First draw-group index of the "above the constructor preview"
    /// layer (dialog hints, Hints buttons, popup, hover hint). The C++
    /// renders CConstructor::Render mid-interface, after the panel
    /// backdrop but below popups/hints (CInterface.cpp:929-933);
    /// `render_base` / `render_overlay` split here so the 3D preview
    /// pass can slot between them.
    overlay_start: usize,
    num_verts: u32,
    /// Glyph cache + GPU upload tracking. Re-uploaded whenever a new
    /// glyph is rasterised (atlas.generation changes).
    glyph_atlas: GlyphAtlas,
    glyph_atlas_generation: u64,
}

impl InterfaceRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
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
        // Nearest sampler for the text atlas. Matches the original
        // D3D9 path which sets `D3DTEXF_POINT` for UI sampling
        // (Interface/CIFaceElement.cpp:154-156). At non-integer panel
        // scales (~1.4× at 1080p) this gives slightly inconsistent
        // 1-or-2 output-pixel stem widths, but each pixel renders at
        // full coverage instead of bilinear's fractional-alpha halo
        // — text reads bright/crisp instead of soft/dim.
        let text_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
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

        // Solid-white 1×1 atlas for flat-colour quads (popup background
        // fill etc.). Texel = (1,1,1,1); the tint then sets the colour
        // outright since the shader multiplies tint × texel.
        let solid_atlases = {
            let pixels = [255u8, 255, 255, 255];
            let view = create_texture_from_rgba(
                device,
                queue,
                &image::RgbaImage::from_raw(1, 1, pixels.to_vec()).expect("1x1 rgba"),
            );
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Solid Atlas BG"),
                layout: &atlas_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            let mut m: HashMap<String, Atlas> = HashMap::new();
            m.insert(
                SOLID_ATLAS_KEY.to_string(),
                Atlas {
                    _tex: view,
                    bind_group,
                    width: 1,
                    height: 1,
                    cpu_pixels: None,
                },
            );
            m
        };

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
            atlases: solid_atlases,
            draw_groups: Vec::new(),
            overlay_start: 0,
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
                cpu_pixels: None,
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
            candidates.push(format!("{base}.dds"));
            candidates.push(format!("{base}.DDS"));
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
        log::debug!(
            "interface: atlas {} ({} bytes) bound as {}",
            atlas_path,
            bytes.len(),
            key
        );
        let (w, h) = (rgba.width(), rgba.height());
        self.atlases.insert(
            key,
            Atlas {
                _tex: view,
                bind_group,
                width: w,
                height: h,
                cpu_pixels: Some(rgba),
            },
        );
    }

    /// Register a runtime-rendered texture as an atlas under `key`.
    /// Used by `RobotIconCache` to wire offscreen-baked robot portraits
    /// into the same atlas-keyed draw path as the static UI atlases —
    /// the corresponding `IFaceElement::StateImage::tex_path` carries
    /// `key` so the existing emit loop picks it up unchanged.
    pub fn register_dynamic_atlas(
        &mut self,
        device: &wgpu::Device,
        key: &str,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Interface Dynamic Atlas BG"),
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
        self.atlases.insert(
            key.to_string(),
            Atlas {
                _tex: view,
                bind_group,
                width,
                height,
                cpu_pixels: None,
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

    /// Faithful port of `CIFaceMenu::CreateMenu` (CIFaceMenu.cpp:62-226)
    /// — composes the popup chrome (corners + edges + tiled SEL +
    /// SELMOUSE + SELRIGHTMOUSE) into a single 512×512 RGBA buffer and
    /// uploads it under `__popup_baked__`. The renderer then draws ONE
    /// quad sourcing from this baked texture, matching the C++
    /// `m_Ramka` composition pixel-for-pixel.
    ///
    /// Returns the popup width and height in design pixels (the C++
    /// `w` and `h` locals at CIFaceMenu.cpp:86-92), or `None` when the
    /// chrome atlas isn't loaded yet.
    ///
    /// `width` is the per-popup width param the C++ receives in
    /// `CreateMenu(parent, elements, width, x, y, ...)` — it does NOT
    /// include the chrome side margins. Total popup width is
    /// `width + CURSIK_WIDTH + LEFTLINE_WIDTH + RIGHTLINE_WIDTH`.
    pub fn bake_popup(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        popup_panel: &CInterface,
        width: i32,
        elements: i32,
    ) -> Option<(u32, u32)> {
        // CIFaceMenu.h constants.
        const TOPLEFT_WIDTH: i32 = 13;
        const TOPLEFT_HEIGHT: i32 = 18;
        const TOPRIGHT_WIDTH: i32 = 18;
        const BOTTOMLEFT_HEIGHT: i32 = 22;
        const BOTTOMRIGHT_WIDTH: i32 = 13;
        const BOTTOMRIGHT_HEIGHT: i32 = 21;
        const TOPLINE_HEIGHT: i32 = 18;
        const BOTTOMLINE_HEIGHT: i32 = 22;
        const RIGHTLINE_WIDTH: i32 = 18;
        const LEFTLINE_WIDTH: i32 = 13;
        const CURSIK_WIDTH: i32 = 7;
        const UNIT_HEIGHT: i32 = 19;
        const MOD: i32 = 8;

        // CIFaceMenu.cpp:88-92.
        let mut h = TOPLINE_HEIGHT + BOTTOMLINE_HEIGHT;
        let h_clean = elements * UNIT_HEIGHT;
        h += h_clean;
        let width = width + CURSIK_WIDTH;
        let w = width + RIGHTLINE_WIDTH + LEFTLINE_WIDTH;

        let tex_w = 512u32;
        let tex_h = 512u32;
        let mut buf = image::RgbaImage::from_pixel(tex_w, tex_h, image::Rgba([0, 0, 0, 0]));

        // Resolve a chrome sprite by element name → (atlas pixels, src_x,
        // src_y, sprite_w, sprite_h). The C++ reads `els->m_Image`,
        // `m_xTexPos`, `m_yTexPos`, `m_Width`, `m_Height` for each
        // entry of `m_FirstImage` (CIFaceMenu.cpp:122-225).
        let get_sprite = |name: &str| -> Option<(&image::RgbaImage, i32, i32, i32, i32)> {
            let elem = popup_panel.elements.iter().find(|e| e.name == name)?;
            let img = elem.images.first()?.as_ref()?;
            let key = normalise_atlas_key(&img.tex_path);
            let atlas = self.atlases.get(&key)?;
            let cpu = atlas.cpu_pixels.as_ref()?;
            // For Image-kind entries, sprite size lives in img.w / img.h
            // (parsed from `Width`/`Height` in parse_image_element).
            // The element's `size_x/size_y` is 0 so we ignore it.
            Some((cpu, img.x as i32, img.y as i32, img.w as i32, img.h as i32))
        };

        // Faithful port of `CBitmap::MergeWithAlpha` — alpha-blend a
        // (src_w × src_h) region from src at (sx, sy) onto buf at (dx,
        // dy). Mirrors MatrixLib/Bitmap/CBitmap.cpp's blend formula.
        fn merge(
            buf: &mut image::RgbaImage,
            src: &image::RgbaImage,
            sx: i32,
            sy: i32,
            sw: i32,
            sh: i32,
            dx: i32,
            dy: i32,
        ) {
            let bw = buf.width() as i32;
            let bh = buf.height() as i32;
            let sxw = src.width() as i32;
            let syh = src.height() as i32;
            for py in 0..sh {
                for px in 0..sw {
                    let dx_p = dx + px;
                    let dy_p = dy + py;
                    let sx_p = sx + px;
                    let sy_p = sy + py;
                    if dx_p < 0 || dy_p < 0 || dx_p >= bw || dy_p >= bh {
                        continue;
                    }
                    if sx_p < 0 || sy_p < 0 || sx_p >= sxw || sy_p >= syh {
                        continue;
                    }
                    let s = src.get_pixel(sx_p as u32, sy_p as u32).0;
                    let d = buf.get_pixel(dx_p as u32, dy_p as u32).0;
                    let alpha = s[3] as u16;
                    let inv = 255 - alpha;
                    let blended = [
                        ((d[0] as u16 * inv + s[0] as u16 * alpha) / 255) as u8,
                        ((d[1] as u16 * inv + s[1] as u16 * alpha) / 255) as u8,
                        ((d[2] as u16 * inv + s[2] as u16 * alpha) / 255) as u8,
                        // Alpha: dest_alpha + src_alpha*(1-dest_alpha).
                        // Equivalent for the porter-duff over operator
                        // when premultiplied alpha is straight.
                        (((d[3] as u16) * inv / 255) + alpha as u16).min(255) as u8,
                    ];
                    buf.put_pixel(dx_p as u32, dy_p as u32, image::Rgba(blended));
                }
            }
        }

        // Faithful port of CIFaceMenu.cpp:122-225. The IF_POPUP_*
        // sprite names match the original macro string-constants in
        // StringConstants.hpp.
        if let Some((src, sx, sy, sw, sh)) = get_sprite("topleft") {
            // CIFaceMenu.cpp:129 — drawn at (1, 0).
            merge(&mut buf, src, sx, sy, sw, sh, 1, 0);
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("topright") {
            // CIFaceMenu.cpp:136 — drawn at (w - TOPRIGHT_WIDTH + 1, 0).
            merge(&mut buf, src, sx, sy, sw, sh, w - TOPRIGHT_WIDTH + 1, 0);
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("bottomleft") {
            // CIFaceMenu.cpp:143 — drawn at (0, h - BOTTOMLEFT_HEIGHT - mod).
            merge(
                &mut buf,
                src,
                sx,
                sy,
                sw,
                sh,
                0,
                h - BOTTOMLEFT_HEIGHT - MOD,
            );
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("bottomright") {
            // CIFaceMenu.cpp:150 — drawn at (w - BOTTOMRIGHT_WIDTH - 4,
            // h - BOTTOMRIGHT_HEIGHT - 1 - mod).
            merge(
                &mut buf,
                src,
                sx,
                sy,
                sw,
                sh,
                w - BOTTOMRIGHT_WIDTH - 4,
                h - BOTTOMRIGHT_HEIGHT - 1 - MOD,
            );
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("leftline") {
            // CIFaceMenu.cpp:156-158 — tiled at (1, TOPLEFT_HEIGHT + i)
            // for i in 0..h_clean - mod.
            for i in 0..(h_clean - MOD) {
                merge(&mut buf, src, sx, sy, sw, sh, 1, TOPLEFT_HEIGHT + i);
            }
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("rightline") {
            // CIFaceMenu.cpp:164-166 — tiled at (w - RIGHTLINE_WIDTH + 1,
            // TOPRIGHT_HEIGHT + i).
            for i in 0..(h_clean - MOD) {
                merge(
                    &mut buf,
                    src,
                    sx,
                    sy,
                    sw,
                    sh,
                    w - RIGHTLINE_WIDTH + 1,
                    TOPLINE_HEIGHT + i,
                );
            }
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("topline") {
            // CIFaceMenu.cpp:172-174 — tiled at (TOPLEFT_WIDTH + i + 1, 0)
            // for i in 0..width.
            for i in 0..width {
                merge(&mut buf, src, sx, sy, sw, sh, TOPLEFT_WIDTH + i + 1, 0);
            }
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("bottomline") {
            // CIFaceMenu.cpp:180-182 — tiled at (BOTTOMLEFT_WIDTH + i,
            // h - BOTTOMLEFT_HEIGHT - mod).
            for i in 0..width {
                merge(
                    &mut buf,
                    src,
                    sx,
                    sy,
                    sw,
                    sh,
                    14 + i, // BOTTOMLEFT_WIDTH = 14 in CIFaceMenu.h
                    h - BOTTOMLEFT_HEIGHT - MOD,
                );
            }
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("sel") {
            // CIFaceMenu.cpp:188-192 — tiled SEL pixel-by-pixel for
            // every row at (j, 11 + UNIT_HEIGHT*i) for j in 5..w-9.
            for i in 0..elements {
                for j in 5..(w - 9) {
                    merge(&mut buf, src, sx, sy, sw, sh, j, 11 + UNIT_HEIGHT * i);
                }
            }
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("selright") {
            // CIFaceMenu.cpp:198-200 — drawn at (w - 12, 11 +
            // UNIT_HEIGHT*i) per row.
            for i in 0..elements {
                merge(&mut buf, src, sx, sy, sw, sh, w - 12, 11 + UNIT_HEIGHT * i);
            }
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("selmouse") {
            // CIFaceMenu.cpp:207-209 — tiled at (w + j, 0) for j 5..w-9.
            // This populates the SECOND copy of the popup (right of the
            // first one in the texture) used by the SELECTOR overlay.
            for j in 5..(w - 9) {
                merge(&mut buf, src, sx, sy, sw, sh, w + j, 0);
            }
        }
        if let Some((src, sx, sy, sw, sh)) = get_sprite("selrightmouse") {
            // CIFaceMenu.cpp:215 — drawn at (w*2 - 12, 0).
            merge(&mut buf, src, sx, sy, sw, sh, w * 2 - 12, 0);
        }

        // Upload the baked buffer as the popup texture.
        let view = create_texture_from_rgba(device, queue, &buf);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Popup Baked Atlas BG"),
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
        self.atlases.insert(
            POPUP_BAKED_KEY.to_string(),
            Atlas {
                _tex: view,
                bind_group,
                width: tex_w,
                height: tex_h,
                cpu_pixels: None,
            },
        );

        Some((w as u32, h as u32))
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
        log::debug!(
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
        self.upload_inner(
            device,
            queue,
            panels,
            popup,
            None,
            &[],
            None,
            screen_w,
            screen_h,
        );
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
        dialog_hints: &[&Hint],
        chrome: Option<&HintChromeLibrary>,
        screen_w: f32,
        screen_h: f32,
    ) {
        self.upload_inner(
            device,
            queue,
            panels,
            popup,
            hint,
            dialog_hints,
            chrome,
            screen_w,
            screen_h,
        );
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
        self.upload_inner(
            device,
            queue,
            panels,
            None,
            None,
            &[],
            None,
            screen_w,
            screen_h,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_inner(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        panels: &[&CInterface],
        popup: Option<&super::iface_menu::CIFaceMenu>,
        hint: Option<&Hint>,
        dialog_hints: &[&Hint],
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

        // Two passes: 0 = every panel except `Hints`, 1 = the dialog-
        // mode hint widgets followed by the `Hints` button panel. The
        // C++ draws hints AFTER the whole interface (MatrixMap.cpp:
        // 2472-2474) and lets the buttons show through alpha holes
        // that `CMatrixHint::Build` punches into the hint bitmap with
        // raw `bmpf.Copy` of the bhole/exit anchor images
        // (MatrixHint.cpp:397-403). A quad streamer can't erase
        // already-emitted pixels, so we reach the same visual stack by
        // drawing the buttons after the dialog hints instead.
        for pass in 0..2u32 {
            if pass == 1 {
                // Everything emitted so far draws BELOW the constructor
                // 3D preview; dialogs / hint buttons / popup / hover
                // hint draw ABOVE it. Flush the open run so the split
                // lands on a group boundary.
                if let Some(k) = current_key.take() {
                    let end = all_verts.len() as u32;
                    if end > current_start {
                        self.draw_groups.push(DrawGroup {
                            atlas_key: k,
                            start: current_start,
                            count: end - current_start,
                        });
                    }
                    current_start = all_verts.len() as u32;
                }
                self.overlay_start = self.draw_groups.len();
                if let Some(c) = chrome {
                    for h in dialog_hints {
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
                }
            }
            for panel in panels {
                if (pass == 0) == (panel.name == "Hints") {
                    continue;
                }
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
                        // Latched check buttons draw their authored
                        // `sPressedUnFocused` art untinted.
                        ElementState::PressedUnfocused => [1.0, 1.0, 1.0, 1.0],
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
        }

        // ── Popup overlay ─────────────────────────────────────────
        // Faithful port of CIFaceMenu::Render (CIFaceMenu.cpp:62-369):
        //   - `m_Ramka` is a single static drawing the BAKED popup
        //     texture composed in CreateMenu (chrome + tiled SEL).
        //   - `m_Selector` is a row-tall static drawing from offset
        //     (w, 0) of the SAME baked texture (the SELMOUSE-tiled
        //     copy populated at CIFaceMenu.cpp:203-216).
        //   - Per-row text catchers render the row labels.
        //   - `m_Cursor` is a Cursik static drawn at the equipped row.
        //
        // We rebake every frame the popup is open so size changes pick
        // up automatically. Cost is ~ items*width pixels of CPU bitmap
        // composition + a 512×512 GPU upload — cheap for a popup that
        // only renders during the constructor's right-click interaction.
        if let Some(popup) = popup {
            // Bake the popup texture into POPUP_BAKED_KEY. The bake call
            // needs the PopupMenu chrome panel for sprite lookups; bail
            // out cleanly if it isn't present (e.g. if the panel data
            // didn't load).
            let popup_baked_size =
                if let Some(popup_panel) = panels.iter().find(|p| p.name == "PopupMenu").copied() {
                    self.bake_popup(
                        device,
                        queue,
                        popup_panel,
                        // C++ `width` is the per-popup item-width param
                        // (CIFaceMenu.h: WEAPON_MENU_WIDTH etc.). We carry
                        // it on `popup.item_w`.
                        popup.item_w as i32,
                        popup.items.len() as i32,
                    )
                } else {
                    None
                };
            let popup_panel = panels.iter().find(|p| p.name == "PopupMenu");
            let base_panel = panels.iter().find(|p| p.name == "Base");
            if let Some(base_panel) = base_panel {
                let [bx, by] = base_panel.resolved_pos(screen_w, screen_h, scale);
                // Origin of the popup rect, screen pixels — top-left of
                // the chrome (NOT the items area). Items sit inset by
                // `popup.items_top()` from the top edge.
                let ox = bx + popup.design_x * scale;
                let oy = by + popup.design_y * scale;
                let total_w = popup.total_w() * scale;
                let total_h = popup.total_h() * scale;
                let items_y0 = oy + popup.items_top() * scale;
                let _items_h = popup.item_h * popup.items.len() as f32 * scale;

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
                    // Half-pixel inset on the UVs so linear filtering
                    // doesn't blend in the colour of neighbour atlas
                    // pixels — important for tiny 1×1 source rects (the
                    // popup interior fill samples a single pixel of the
                    // chrome line and must read it as a solid colour
                    // without bleed from the next pixel).
                    let u0 = (img.x + 0.5) / img.tex_w;
                    let v0 = (img.y + 0.5) / img.tex_h;
                    let u1 = (img.x + img.w - 0.5).max(img.x + 0.5) / img.tex_w;
                    let v1 = (img.y + img.h - 0.5).max(img.y + 0.5) / img.tex_h;
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

                // (1) m_Ramka — render the baked popup texture as ONE
                // quad sourcing from (0, 0) to (popup_w, popup_h)
                // inside POPUP_BAKED_KEY. CIFaceMenu.cpp:248-262.
                let opaque = [1.0, 1.0, 1.0, 1.0];
                if let Some((bake_w, bake_h)) = popup_baked_size {
                    let img = super::iface_element::StateImage {
                        x: 0.0,
                        y: 0.0,
                        w: bake_w as f32,
                        h: bake_h as f32,
                        tex_w: 512.0,
                        tex_h: 512.0,
                        tex_path: POPUP_BAKED_KEY.to_string(),
                    };
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox,
                        oy,
                        total_w,
                        total_h,
                        &img,
                        opaque,
                    );
                }

                use crate::matrix_game::interface::iface_menu::CIFaceMenu;

                // (2) m_Selector overlay — port of CIFaceMenu.cpp:
                // 268-297. The C++ selector samples the baked texture
                // starting at (w, 0) (i.e. the SECOND copy of the
                // popup populated by SELMOUSE+SELRIGHTMOUSE) sized
                // w × 18, repositioned to the hovered row's y. Drawn
                // BEFORE the row text so labels render on top.
                if let (Some(hi), Some((bake_w, _bake_h))) = (popup.hovered, popup_baked_size) {
                    let row_y = oy + (CIFaceMenu::ITEMS_TOP + popup.item_h * hi as f32) * scale;
                    let img = super::iface_element::StateImage {
                        x: bake_w as f32,
                        y: 0.0,
                        w: bake_w as f32,
                        h: 18.0,
                        tex_w: 512.0,
                        tex_h: 512.0,
                        tex_path: POPUP_BAKED_KEY.to_string(),
                    };
                    emit_textured(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        ox,
                        row_y,
                        total_w,
                        18.0 * scale,
                        &img,
                        opaque,
                    );
                }

                // (3) Row labels — faithful port of CIFaceMenu::CreateMenu
                // catcher pass at CIFaceMenu.cpp:312-341. Drawn AFTER the
                // selector so labels stay readable on the highlighted
                // row.
                for (i, item) in popup.items.iter().enumerate() {
                    let row_y = items_y0 + i as f32 * popup.item_h * scale;
                    let row_h = popup.item_h * scale;
                    // C++ NERES_LABELS_COLOR (red-grey) when unaffordable;
                    // DEFAULT_LABELS_COLOR (gold) otherwise.
                    let color = if !item.affordable {
                        [255, 67, 25, 255]
                    } else {
                        [246, 192, 0, 255]
                    };
                    let runs = vec![RichRun {
                        text: item.text.clone(),
                        color: None,
                    }];
                    // C++ catcher text origin: `LEFT_SPACE+6 = 13` from
                    // catcher left, `-3` from catcher top with align_y=1
                    // (CIFaceMenu.cpp:338).
                    let text_x = ox + 13.0 * scale;
                    let text_y = row_y + (row_h * 0.5) - 3.0 * scale;
                    emit_rich_line(
                        &mut all_verts,
                        &mut current_key,
                        &mut current_start,
                        &mut self.draw_groups,
                        &mut self.glyph_atlas,
                        "Font.2Ranger",
                        &runs,
                        scale,
                        text_x,
                        text_y,
                        0,
                        1,
                        color,
                    );
                }

                // (5) Cursik — the arrow indicator at the row matching
                // the currently-equipped item. Port of `m_CursikImage` /
                // `cursik_hpos` placement at CIFaceMenu.cpp:94-96, 217:
                // x = LEFT_SPACE (7), y = idx*UNIT_HEIGHT + TOPLINE_HEIGHT
                // (relative to popup top).
                if let Some(pos) = popup.current_pos {
                    if let Some(cursik) = pop_img("cursik") {
                        let cw = cursik.w * scale;
                        let ch = cursik.h * scale;
                        // CIFaceMenu.cpp:96: cursik_hpos = idx*19 + 18.
                        let cursik_hpos = pos as f32 * popup.item_h + 18.0;
                        let cx = ox + 7.0 * scale;
                        let cy = oy + cursik_hpos * scale - ch * 0.5;
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
        //
        // When the popup is open, the C++ renders the popup on its own
        // m_MenuGraphics interface that's drawn LAST and covers
        // anything underneath (CIFaceMenu.cpp:225-369). Our text pass
        // runs after the popup quads but iterates ALL panel labels —
        // including chasl/hulll/headl/weapl etc. that visually fall
        // INSIDE the popup rect and would leak on top of the baked
        // chrome. Compute the popup rect here in screen pixels and
        // skip any label whose anchor lands inside it.
        let popup_rect: Option<[f32; 4]> = popup.and_then(|popup| {
            let base_panel = panels.iter().find(|p| p.name == "Base")?;
            let [bx, by] = base_panel.resolved_pos(screen_w, screen_h, scale);
            let ox = bx + popup.design_x * scale;
            let oy = by + popup.design_y * scale;
            let w = popup.total_w() * scale;
            let h = popup.total_h() * scale;
            Some([ox, oy, w, h])
        });
        for panel in panels {
            if !panel.visible {
                continue;
            }
            let [px, py] = panel.resolved_pos(screen_w, screen_h, scale);
            // Don't text-draw the PopupMenu chrome elements — the
            // popup-overlay block above already laid out the row
            // labels via `emit_rich_line`. Letting the chrome's own
            // panel labels (e.g. Russian-localised headers from the
            // PopupMenu data) draw through here would double-render.
            if panel.name == "PopupMenu" {
                continue;
            }
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
                    let design_anchor_x = elem.pos_x + base_x + label.x + label.sme_x;
                    let design_anchor_y = elem.pos_y + base_y + label.y + label.sme_y;
                    let anchor_x = px + design_anchor_x * scale;
                    let anchor_y = py + design_anchor_y * scale;
                    // Skip labels whose anchor falls inside the popup
                    // rect — the C++ achieves the same effect by
                    // drawing the popup on its own panel that sits on
                    // top of the constructor-base panel.
                    if let Some([rx, ry, rw, rh]) = popup_rect {
                        if anchor_x >= rx
                            && anchor_x < rx + rw
                            && anchor_y >= ry
                            && anchor_y < ry + rh
                        {
                            continue;
                        }
                    }
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
                        let wrap_width = (elem.size_x - label.x - label.sme_x).max(0.0) * scale;
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
            log::debug!(
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
            log::debug!(
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
        self.render_groups(pass, 0, self.draw_groups.len());
    }

    /// Panels only — everything below the constructor 3D preview.
    pub fn render_base<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        self.render_groups(pass, 0, self.overlay_start.min(self.draw_groups.len()));
    }

    /// Dialog hints / hint buttons / popup / hover hint — everything
    /// above the constructor 3D preview.
    pub fn render_overlay<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        self.render_groups(pass, self.overlay_start.min(self.draw_groups.len()), self.draw_groups.len());
    }

    fn render_groups<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, from: usize, to: usize) {
        if self.num_verts == 0 || from >= to {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniform_bg, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        for g in &self.draw_groups[from..to] {
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
            open_run(
                &atlas_key,
                current_key,
                current_start,
                all_verts,
                draw_groups,
            );
            for d in hint.slice_draws(chrome, scale) {
                let u0 = d.tex_x as f32 / tw;
                let v0 = d.tex_y as f32 / th;
                let u1 = (d.tex_x + d.tex_w) as f32 / tw;
                let v1 = (d.tex_y + d.tex_h) as f32 / th;
                let (x, y, w, h) = (d.screen_x, d.screen_y, d.screen_w, d.screen_h);
                all_verts.extend_from_slice(&[
                    Vertex {
                        pos: [x, y],
                        uv: [u0, v0],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [x + w, y],
                        uv: [u1, v0],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [x, y + h],
                        uv: [u0, v1],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [x + w, y],
                        uv: [u1, v0],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [x + w, y + h],
                        uv: [u1, v1],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [x, y + h],
                        uv: [u0, v1],
                        tint: [1.0; 4],
                    },
                ]);
            }
        }
    }

    // ── Parts (text + inline bitmaps) ───────────────────────────────
    for part in &hint.parts {
        match part {
            HintPart::Text {
                x,
                y,
                h,
                text,
                font,
                color,
                ..
            } => {
                if text.is_empty() {
                    continue;
                }
                // One `_TEXT:` can span multiple rendered lines
                // (explicit `<br>` or wrap). The hint layout already
                // picked the per-line widths; here we render each
                // `\n`-separated sub-line at its own vertical stride.
                let line_stride_px = (glyph_atlas.line_height(font) as f32 + 1.0) * scale;
                let line_h = glyph_atlas.line_height(font) as f32;
                let line_count = text.split('\n').count().max(1) as f32;
                // Natural visible height of the text block — matches the
                // renderer's per-line stride, with no trailing leading.
                let natural_h_design = line_h + (line_h + 1.0) * (line_count - 1.0);
                // When the layout reserved more vertical space than the
                // text actually needs (e.g. `_HEIGHT:23` on the
                // BuildTurret price row sets h=23 around a 13-px-tall
                // text), vertically centre the content inside that
                // band — port of the Rangers rasterizer's behaviour
                // for `m_RangersText(..., h, ..., aligny=0, ...)`.
                let extra_h = (*h as f32 - natural_h_design).max(0.0);
                let v_offset = (extra_h * 0.5) * scale;
                let base_x = hint.screen_x + *x as f32 * scale;
                let base_y = hint.screen_y + *y as f32 * scale + v_offset;
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
                    Vertex {
                        pos: [px, py],
                        uv: [0.0, 0.0],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [px + pw, py],
                        uv: [1.0, 0.0],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [px, py + ph],
                        uv: [0.0, 1.0],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [px + pw, py],
                        uv: [1.0, 0.0],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [px + pw, py + ph],
                        uv: [1.0, 1.0],
                        tint: [1.0; 4],
                    },
                    Vertex {
                        pos: [px, py + ph],
                        uv: [0.0, 1.0],
                        tint: [1.0; 4],
                    },
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
    // Glyphs render at native AFT pixel size (no `* scale`),
    // matching the original game's 1:1 D3DTEXF_POINT path. The
    // `scale` parameter is still consumed by the caller layout
    // (anchor_x / anchor_y come in at scaled panel coords) but
    // glyph dimensions and inner offsets stay in native pixels.
    let _ = scale;
    let key = TEXT_ATLAS_KEY;
    let atlas_w = glyph_atlas.width() as f32;
    let atlas_h = glyph_atlas.height() as f32;
    let total_w: f32 = runs
        .iter()
        .map(|r| glyph_atlas.measure(font, &r.text) as f32)
        .sum();
    let line_h = glyph_atlas.line_height(font) as f32;
    let ascent = glyph_atlas.ascent(font) as f32;

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
    let baseline_y = cell_top + ascent;

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
            // Glyph origins snap to integer pixels (caller anchors
            // come in at sub-pixel scaled coords). Native size means
            // 1:1 texel-to-pixel mapping at any UI scale, matching
            // the original D3D9 `D3DTEXF_POINT` path.
            let x = (start_x + pen_x_native + g.bearing_x as f32).round();
            let y = (baseline_y + g.bearing_y as f32).round();
            let w = g.w as f32;
            let h = g.h as f32;
            let u0 = g.atlas_x as f32 / atlas_w;
            let v0 = g.atlas_y as f32 / atlas_h;
            let u1 = (g.atlas_x + g.w) as f32 / atlas_w;
            let v1 = (g.atlas_y + g.h) as f32 / atlas_h;
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
    // Text renders at native pixel size in `emit_rich_line`, so
    // wrap on the unscaled wrap width and step lines by the native
    // line height.
    let _ = scale;
    let lines = wrap_rich_lines(glyph_atlas, font, runs, wrap_width as u32);
    if lines.is_empty() {
        return;
    }
    // 1 px of leading on top of (ascent + descent) so descenders on
    // one row don't kiss ascenders on the next.
    let line_h = glyph_atlas.line_height(font) as f32 + 1.0;
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
        Word {
            text: String,
            color: Option<[u8; 4]>,
        },
        Space {
            color: Option<[u8; 4]>,
        },
        Break,
    }
    let mut toks: Vec<Tok> = Vec::new();
    for run in runs {
        let mut buf = String::new();
        for c in run.text.chars() {
            if c == '\n' {
                if !buf.is_empty() {
                    toks.push(Tok::Word {
                        text: std::mem::take(&mut buf),
                        color: run.color,
                    });
                }
                toks.push(Tok::Break);
            } else if c == '\r' {
                if !buf.is_empty() {
                    toks.push(Tok::Word {
                        text: std::mem::take(&mut buf),
                        color: run.color,
                    });
                }
                // Skip CR; LF will trigger Break.
            } else if c.is_whitespace() {
                if !buf.is_empty() {
                    toks.push(Tok::Word {
                        text: std::mem::take(&mut buf),
                        color: run.color,
                    });
                }
                toks.push(Tok::Space { color: run.color });
            } else {
                buf.push(c);
            }
        }
        if !buf.is_empty() {
            toks.push(Tok::Word {
                text: buf,
                color: run.color,
            });
        }
    }

    let space_w = atlas.measure(font, " ");
    let mut lines: Vec<Vec<RichRun>> = Vec::new();
    let mut current: Vec<RichRun> = Vec::new();
    let mut current_w: u32 = 0;

    let push_line =
        |current: &mut Vec<RichRun>, current_w: &mut u32, lines: &mut Vec<Vec<RichRun>>| {
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
                    current.push(RichRun {
                        text: " ".to_string(),
                        color,
                    });
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
        vec![RichRun {
            text: text.to_string(),
            color: None,
        }]
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
        let joined: String = lines
            .iter()
            .map(|l| line_text(l))
            .collect::<Vec<_>>()
            .join(" ");
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
            RichRun {
                text: "Damage ".to_string(),
                color: None,
            },
            RichRun {
                text: "30".to_string(),
                color: Some([247, 195, 0, 255]),
            },
            RichRun {
                text: " HP".to_string(),
                color: None,
            },
        ];
        let lines = wrap_rich_lines(&mut a, "Font.2Ranger", &runs, 1000);
        assert_eq!(lines.len(), 1);
        // The gold "30" run should still carry its color override.
        let gold_count = lines[0]
            .iter()
            .filter(|r| r.color == Some([247, 195, 0, 255]))
            .count();
        assert!(
            gold_count > 0,
            "color override lost during wrap: {:?}",
            lines[0]
        );
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
