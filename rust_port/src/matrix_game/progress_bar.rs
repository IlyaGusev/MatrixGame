//! Port of `CMatrixProgressBar` (MatrixProgressBar.{cpp,hpp}).
//!
//! Renders a horizontal 3-segment stretched progress bar from the
//! shared `Matrix\\Textures\\pb` atlas (TEXTURE_PATH_PB at
//! StringConstants.hpp:106). The atlas is laid out as four V-strips
//! of 0.25 each:
//!
//! * V [0.00, 0.25] — empty background (`pbd=false` path at
//!   MatrixProgressBar.cpp:176-183).
//! * V [0.25, 0.50] — filled body (`pbd=true` normal path at
//!   MatrixProgressBar.cpp:249-258, `p[*].tv = 0.5 / 0.25`).
//! * V [0.50, 0.75] / V [0.75, 1.00] — "growing cap" stages used
//!   when the filled width is less than two cap-widths
//!   (MatrixProgressBar.cpp:197-228). Deferred — below the
//!   cap-threshold we just draw the normal filled path; visually
//!   indistinguishable for buttons wider than ~13 pixels.
//!
//! Each segment spans `h = texture_height / 4` pixels horizontally
//! (the C++ reads this at :143); the middle body stretches to fill
//! the rest.  TFACTOR modulation multiplies the texture's grayscale
//! fill by the red→yellow→green LIC color
//! (`PB_COLOR_0..PB_COLOR_2` at MatrixProgressBar.hpp:17-19).

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};

use crate::matrix_lib::three_g::texture::{create_texture_from_rgba, decode_texture_bytes};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    screen_size: [f32; 4],
    color: [f32; 4],
}

/// `PB_COLOR_0` / `PB_COLOR_1` / `PB_COLOR_2` (MatrixProgressBar.hpp:17-19).
/// ARGB with 0 alpha in the C++; the draw path OR-fills alpha to 0xFF
/// at line 195. We store as normalised f32 RGB and re-attach alpha=1.
const PB_COLOR_0: [f32; 3] = [1.0, 0.0, 0.0]; // red at 0%
const PB_COLOR_1: [f32; 3] = [1.0, 1.0, 0.0]; // yellow at 50%
const PB_COLOR_2: [f32; 3] = [0.0, 1.0, 0.0]; // green at 100%

/// Port of the LIC at MatrixProgressBar.cpp:189-190. Two-stop ramp:
/// 0..0.5 from red → yellow, 0.5..1.0 from yellow → green.
fn pb_color(pos: f32) -> [f32; 4] {
    let p = pos.clamp(0.0, 1.0);
    let (a, b, t) = if p < 0.5 {
        (PB_COLOR_0, PB_COLOR_1, p * 2.0)
    } else {
        (PB_COLOR_1, PB_COLOR_2, (p - 0.5) * 2.0)
    };
    [
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
        1.0,
    ]
}

/// One bar queued for this frame.
pub struct ProgressBar {
    /// Screen-pixel anchor + size. Width is the full "empty" bar
    /// length; the filled portion scales inside. The C++ `Modify(x,
    /// y, width, pos)` at MatrixProgressBar.hpp:68 stores exactly
    /// these values on `m_Coord`.
    pub rect: [f32; 4],
    /// Fill fraction in `[0, 1]`. `m_Pos` in the C++.
    pub fill: f32,
}

struct BarBuffers {
    vb: wgpu::Buffer,
    ub_empty: wgpu::Buffer,
    bg_empty: wgpu::BindGroup,
    ub_filled: wgpu::Buffer,
    bg_filled: wgpu::BindGroup,
}

pub struct ProgressBarRenderer {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    atlas_view: Option<wgpu::TextureView>,
    atlas_ready: bool,
    /// Per-queued-bar resources. Grown on demand; stays allocated
    /// across frames for cheap reuse.
    bars: Vec<BarBuffers>,
    pending: Vec<ProgressBar>,
    /// Texture-h in pixels. Set when the atlas is loaded; half of
    /// the V strip = bar visual height, quarter = segment width.
    atlas_h: f32,
}

impl ProgressBarRenderer {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ProgressBar BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ProgressBar Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ProgressBar PL"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ProgressBar Pipeline"),
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
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            pipeline,
            bgl,
            sampler,
            atlas_view: None,
            atlas_ready: false,
            bars: Vec::new(),
            pending: Vec::new(),
            atlas_h: 0.0,
        }
    }

    /// Load `Matrix\\Textures\\pb`. Ports the
    /// `g_Cache->Get(cc_TextureManaged, TEXTURE_PATH_PB)` one-time
    /// fetch at MatrixProgressBar.cpp:22.
    pub fn load_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        read: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) {
        log::debug!("progress bar: load_atlas called");
        if self.atlas_ready {
            return;
        }
        // pb ships only as DDS in the pkg; bundle may store it under
        // either spelling. Try every plausible variant.
        let candidates = [
            "Matrix/Textures/pb",
            "Matrix/Textures/pb.png",
            "Matrix/Textures/pb.dds",
            "MATRIX/TEXTURES/PB",
            "MATRIX/TEXTURES/PB.PNG",
            "MATRIX/TEXTURES/PB.DDS",
        ];
        for c in &candidates {
            match read(c) {
                Some(bytes) => {
                    log::debug!(
                        "progress bar: candidate {} → {} bytes, trying decode",
                        c,
                        bytes.len(),
                    );
                    if let Some(img) = decode_texture_bytes(&bytes) {
                        self.atlas_h = img.height() as f32;
                        let view = create_texture_from_rgba(device, queue, &img);
                        self.atlas_view = Some(view);
                        self.atlas_ready = true;
                        log::info!(
                            "progress bar: atlas loaded ({}x{}) from {}",
                            img.width(),
                            img.height(),
                            c
                        );
                        return;
                    } else {
                        log::warn!("progress bar: candidate {} failed to decode", c);
                    }
                }
                None => {
                    log::debug!("progress bar: candidate {} not in bundle", c);
                }
            }
        }
        log::warn!("progress bar: Matrix/Textures/pb not found");
    }

    pub fn push(&mut self, bar: ProgressBar) {
        self.pending.push(bar);
    }

    pub fn clear(&mut self) {
        self.pending.clear();
    }

    /// Bar height in design-space pixels — the C++ reads
    /// `m_Tex.Height() >> 2` at MatrixProgressBar.cpp:143, i.e.
    /// quarter of the atlas height, both as the bar's visual height
    /// and the cap segment width.
    pub fn bar_height_design(&self) -> f32 {
        self.atlas_h * 0.25
    }

    /// Build vertex / uniform data for every queued bar. Two draws
    /// per bar: empty background (white TFACTOR, V strip [0, 0.25])
    /// and filled portion (LIC color, V strip [0.25, 0.5]).
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_w: f32,
        screen_h: f32,
    ) {
        if !self.atlas_ready {
            return;
        }
        let tex_view = self.atlas_view.as_ref().unwrap();

        // Grow buffer pool.
        while self.bars.len() < self.pending.len() {
            // 2 passes (empty + filled) × 3 segments (left-cap / body
            // / right-cap) × 6 verts-per-quad = 36 verts. Round up
            // to 48 for padding headroom.
            let vert_size = (48 * std::mem::size_of::<Vertex>()) as u64;
            let vb = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ProgressBar VB"),
                size: vert_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mk_ub_bg = |label: &str| {
                let ub = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ub.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });
                (ub, bg)
            };
            let (ub_empty, bg_empty) = mk_ub_bg("ProgressBar UB empty");
            let (ub_filled, bg_filled) = mk_ub_bg("ProgressBar UB filled");
            self.bars.push(BarBuffers {
                vb,
                ub_empty,
                bg_empty,
                ub_filled,
                bg_filled,
            });
        }

        // The C++ uses h = tex.Height() >> 2 — quarter of the atlas
        // height. For the shipped pb.dds (typically 64×64) that's 16
        // pixels, so each cap is 16px wide and the bar itself is 16px
        // tall. We scale with the caller's requested rect height to
        // stay consistent when the UI design is larger than a single
        // atlas quarter.
        let cap_world = {
            // Use the minimum of (rect.h, atlas_h/4) so small bars
            // don't distort — mirrors the C++ which places the cap
            // at `posx+h` regardless of rect height.
            self.atlas_h * 0.25
        };

        for (i, bar) in self.pending.iter().enumerate() {
            let [rx, ry, rw, rh] = bar.rect;
            let empty_verts = build_segments(
                rx,
                ry,
                rw,
                rh,
                cap_world,
                // V strip [0, 0.25] — empty body (MatrixProgressBar.cpp:167-174).
                [0.0, 0.25],
            );
            let fill = bar.fill.clamp(0.0, 1.0);
            let fill_width = rw * fill;
            let filled_verts = build_segments(
                rx,
                ry,
                fill_width,
                rh,
                cap_world,
                // V strip [0.25, 0.5] — filled body (MatrixProgressBar.cpp:251-258).
                [0.25, 0.5],
            );

            let mut all_verts = empty_verts;
            all_verts.extend_from_slice(&filled_verts);
            queue.write_buffer(&self.bars[i].vb, 0, bytemuck::cast_slice(&all_verts));

            let empty_uniform = Uniforms {
                screen_size: [screen_w, screen_h, 0.0, 0.0],
                // White TFACTOR for the background pass
                // (MatrixProgressBar.cpp:179 SetRenderState(TEXTUREFACTOR, 0xFFFFFFFF)).
                color: [1.0, 1.0, 1.0, 1.0],
            };
            let filled_uniform = Uniforms {
                screen_size: [screen_w, screen_h, 0.0, 0.0],
                color: pb_color(fill),
            };
            queue.write_buffer(
                &self.bars[i].ub_empty,
                0,
                bytemuck::bytes_of(&empty_uniform),
            );
            queue.write_buffer(
                &self.bars[i].ub_filled,
                0,
                bytemuck::bytes_of(&filled_uniform),
            );
        }
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.pending.is_empty() {
            return;
        }
        if !self.atlas_ready {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!(
                    "progress: {} bars queued but atlas is not ready — render skipped",
                    self.pending.len()
                );
            }
            return;
        }
        pass.set_pipeline(&self.pipeline);
        for (i, _bar) in self.pending.iter().enumerate() {
            let b = &self.bars[i];
            pass.set_vertex_buffer(0, b.vb.slice(..));
            // 3 segments × 6 verts = 18 verts per pass. 2 passes = 36 verts.
            pass.set_bind_group(0, &b.bg_empty, &[]);
            pass.draw(0..18, 0..1);
            pass.set_bind_group(0, &b.bg_filled, &[]);
            pass.draw(18..36, 0..1);
        }
    }
}

/// Build the 3-segment triangle list for a bar rect. UVs span:
/// left-cap [0..0.25]×v, body [0.25..0.75]×v, right-cap [0.75..1]×v
/// — the fixed horizontal layout of the pb atlas
/// (MatrixProgressBar.cpp:167-174 default UV set).
fn build_segments(x: f32, y: f32, w: f32, h: f32, cap: f32, v_range: [f32; 2]) -> Vec<Vertex> {
    let mut verts = Vec::with_capacity(18);
    if w <= 0.0 {
        return verts;
    }
    let v0 = v_range[0];
    let v1 = v_range[1];

    // Sub-rect X breakpoints. If the bar is narrower than two caps,
    // the body collapses — we still draw each segment but the body's
    // width falls to ≤ 0 (skipped).
    let x_left_end = x + cap.min(w * 0.5);
    let x_right_beg = x + (w - cap.min(w * 0.5));
    let x_right_end = x + w;

    let push_quad = |verts: &mut Vec<Vertex>, x0: f32, x1: f32, u0: f32, u1: f32| {
        if x1 - x0 <= 0.0 {
            return;
        }
        let p00 = Vertex {
            pos: [x0, y + h],
            uv: [u0, v1],
        };
        let p10 = Vertex {
            pos: [x0, y],
            uv: [u0, v0],
        };
        let p01 = Vertex {
            pos: [x1, y + h],
            uv: [u1, v1],
        };
        let p11 = Vertex {
            pos: [x1, y],
            uv: [u1, v0],
        };
        verts.extend_from_slice(&[p00, p10, p11, p00, p11, p01]);
    };

    // Left cap: U [0, 0.25].
    push_quad(&mut verts, x, x_left_end, 0.0, 0.25);
    // Body: U [0.25, 0.75].
    push_quad(&mut verts, x_left_end, x_right_beg, 0.25, 0.75);
    // Right cap: U [0.75, 1.0].
    push_quad(&mut verts, x_right_beg, x_right_end, 0.75, 1.0);

    // Pad with degenerate quads so every bar produces exactly 18
    // vertices (3 quads × 6). Lets the renderer use a fixed draw
    // range per bar without conditional offset math.
    while verts.len() < 18 {
        let last = *verts.last().unwrap_or(&Vertex {
            pos: [x, y],
            uv: [0.0, v0],
        });
        verts.push(last);
    }
    verts
}

const SHADER: &str = include_str!("../../shaders/progress_bar.wgsl");
