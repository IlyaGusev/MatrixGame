//! Port of `CMultiSelection::Draw` (MatrixMultiSelection.cpp:104-202) —
//! the screen-space selection rectangle the user drags to multi-select
//! their own robots.
//!
//! The original draws three concentric line-strips (as
//! `DrawPrimitiveUP(D3DPT_LINESTRIP, ...)` with `D3DFVF_XYZRHW`):
//!
//!   1. Exact rect at `(LT, RB)`  — `MS_RAMKA_COLOR  = 0xFF00FF00`
//!      (opaque green).
//!   2. Outer rect inset by `(-1, -1)` and `(+1, +1)` — i.e. 1px outside
//!      the exact rect — `MS_RAMKA_COLOR2 = 0x80000000` (50% black).
//!   3. Inner rect inset by `(+1, +1)` and `(-1, -1)` relative to the
//!      exact rect — i.e. 1px inside — same `MS_RAMKA_COLOR2`.
//!
//! On release, `End()` flips `MS_FLAG_DIP` and records `m_TimeBeforeDip`;
//! the rect then fades out over `MS_DIP_TIME = 50ms`
//! (MatrixMultiSelection.cpp:109-122) by LIC-ing the alpha to zero.
//!
//! Not ported (deferred):
//! * `DrawPass1/Pass2/PassEnd` (:40-101) — stencil-masked wireframe pass
//!   that re-renders objects inside the rect as wireframe outlines. That
//!   requires enabling `POLYGON_MODE_LINE` (unavailable on WebGL2) and
//!   re-issuing every object draw call with a different fill mode.
//!   The marquee rect itself is fully functional without it.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};

/// MS_RAMKA_COLOR  (MatrixMultiSelection.cpp:18) — ARGB 0xFF00FF00,
/// bright opaque green. The exact-rect line strip color.
const MS_RAMKA_COLOR_RGBA:  [f32; 4] = [0.0, 1.0, 0.0, 1.0];
/// MS_RAMKA_COLOR2 (MatrixMultiSelection.cpp:19) — ARGB 0x80000000,
/// 50%-alpha black. The ±1px line-strip color — the two dark strips
/// make the bright green line read as a beveled edge.
const MS_RAMKA_COLOR2_RGBA: [f32; 4] = [0.0, 0.0, 0.0, 128.0 / 255.0];
/// Port of `MS_DIP_TIME` (MatrixMultiSelection.hpp:15) — ms the rect
/// fades for after release.
const MS_DIP_TIME_MS: f32 = 50.0;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MarqueeVertex {
    /// Clip-space XY in `[-1, 1]`.
    pos: [f32; 2],
    /// Per-vertex RGBA — not premultiplied. Alpha blend state handles
    /// the `SRC_ALPHA, ONE_MINUS_SRC_ALPHA` combine.
    color: [f32; 4],
}

/// Three concentric line strips = 3 × 5 verts (closed loops).
/// Unchanged between frames, rebuilt on `set_rect`.
const MAX_VERTS: usize = 15;

pub struct MarqueeRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    /// Number of verts uploaded this frame (0, 5, 10, or 15 — three
    /// draws of 5 verts each, drawn as one LineStrip batch split into
    /// three primitive-restart-free sub-ranges via `pass.draw`).
    vertex_count: u32,
    /// When `Some`, we're in the DIP-fade countdown: `release_time_ms`
    /// snapshots `elapsed_ms` at release; the alpha LIC-s from full
    /// to zero over `MS_DIP_TIME_MS` before the renderer hides itself.
    dip_release_ms: Option<f32>,
}

impl MarqueeRenderer {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Marquee Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Marquee PL"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Marquee Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MarqueeVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as u64,
                            shader_location: 1,
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
                    // MatrixMultiSelection.cpp:160 sets `D3DRS_DESTBLEND =
                    // D3DBLEND_INVSRCALPHA` (and the default SRCBLEND is
                    // D3DBLEND_SRCALPHA) — classic `SRC_ALPHA,
                    // ONE_MINUS_SRC_ALPHA` alpha-over blend.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                // C++: D3DPT_LINESTRIP. The three rects are rendered as
                // three separate line strips (one per DrawPrimitiveUP).
                // wgpu doesn't support primitive restart in WebGL2, so
                // we issue three `pass.draw(5 verts)` calls each frame.
                topology: wgpu::PrimitiveTopology::LineStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Marquee VB"),
            size: (MAX_VERTS * std::mem::size_of::<MarqueeVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { pipeline, vertex_buffer, vertex_count: 0, dip_release_ms: None }
    }

    /// Push a live marquee rect (the user is still dragging). Pixel
    /// coords, top-left origin, axis-aligned. `rect = [x0, y0, x1, y1]`
    /// with no ordering constraint — the C++ `Update` normalises via
    /// the XOR swap at :70-71 so `(LT, RB)` always maps to the min/max.
    pub fn set_rect(
        &mut self,
        queue: &wgpu::Queue,
        rect: [f32; 4],
        screen_w: f32,
        screen_h: f32,
    ) {
        self.dip_release_ms = None;
        self.upload_rect(queue, rect, screen_w, screen_h, 1.0);
    }

    /// Called at LMB release — freezes the last rect and starts the
    /// 50ms alpha LIC that matches the C++ DIP fade
    /// (MatrixMultiSelection.cpp:278-279 + :109-123).
    pub fn begin_dip_fade(&mut self, elapsed_ms: f32) {
        if self.vertex_count > 0 {
            self.dip_release_ms = Some(elapsed_ms);
        }
    }

    /// Advance the DIP fade — call once per frame. Hides the rect
    /// when the fade timer expires.
    pub fn takt(
        &mut self,
        queue: &wgpu::Queue,
        elapsed_ms: f32,
        last_rect: Option<[f32; 4]>,
        screen_w: f32,
        screen_h: f32,
    ) {
        let Some(t0) = self.dip_release_ms else { return };
        let cdt = (elapsed_ms - t0).max(0.0);
        if cdt > MS_DIP_TIME_MS {
            self.clear();
            return;
        }
        // C++ `LIC(color, color & 0x00FFFFFF, k)` — linearly fades the
        // color toward the same RGB with alpha=0. Equivalent to
        // alpha *= (1 - k).
        let k = cdt / MS_DIP_TIME_MS;
        let alpha_scale = 1.0 - k;
        if let Some(r) = last_rect {
            self.upload_rect(queue, r, screen_w, screen_h, alpha_scale);
        }
    }

    /// Hide the rect (zero vertex count → `render` is a no-op).
    pub fn clear(&mut self) {
        self.vertex_count = 0;
        self.dip_release_ms = None;
    }

    /// Is the fade currently playing? Used by the caller to keep the
    /// frozen rect coords alive long enough for `takt` to drive them.
    pub fn is_fading(&self) -> bool {
        self.dip_release_ms.is_some()
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.vertex_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        // C++: three separate DrawPrimitiveUP(LINESTRIP, 4 prims, 5
        // verts) calls. LineStrip topology treats each `draw` range as
        // an independent strip (the primitive-restart / breaking logic
        // in wgpu is otherwise implicit — subsequent draws re-start
        // the strip).
        pass.draw(0..5, 0..1);   // exact rect    — green
        pass.draw(5..10, 0..1);  // outer (+1 px) — 50% black
        pass.draw(10..15, 0..1); // inner (−1 px) — 50% black
    }

    /// Build the 15-vertex buffer and write it. `alpha_scale` multiplies
    /// every strip's alpha — used by the DIP fade.
    fn upload_rect(
        &mut self,
        queue: &wgpu::Queue,
        rect: [f32; 4],
        screen_w: f32,
        screen_h: f32,
        alpha_scale: f32,
    ) {
        if screen_w <= 1.0 || screen_h <= 1.0 {
            self.vertex_count = 0;
            return;
        }
        let [mut x0, mut y0, mut x1, mut y1] = rect;
        // MatrixMultiSelection.cpp:70-71 normalises the rect so
        // `(LT, RB)` always bound the min/max corners — same intent.
        if x0 > x1 { std::mem::swap(&mut x0, &mut x1); }
        if y0 > y1 { std::mem::swap(&mut y0, &mut y1); }

        let sw = screen_w;
        let sh = screen_h;
        let verts = [
            // Strip 1: exact rect — MS_RAMKA_COLOR (green).
            strip_vertex(x0, y0, sw, sh, MS_RAMKA_COLOR_RGBA, alpha_scale),
            strip_vertex(x1, y0, sw, sh, MS_RAMKA_COLOR_RGBA, alpha_scale),
            strip_vertex(x1, y1, sw, sh, MS_RAMKA_COLOR_RGBA, alpha_scale),
            strip_vertex(x0, y1, sw, sh, MS_RAMKA_COLOR_RGBA, alpha_scale),
            strip_vertex(x0, y0, sw, sh, MS_RAMKA_COLOR_RGBA, alpha_scale),
            // Strip 2: outer rect (+1 px on every side) — MS_RAMKA_COLOR2.
            // Post-shifts from the C++ :168-178 applied to v2[0..3]:
            //   TL −= (1, 1), TR += (1, -1), BR += (1, 1), BL += (-1, 1).
            strip_vertex(x0 - 1.0, y0 - 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
            strip_vertex(x1 + 1.0, y0 - 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
            strip_vertex(x1 + 1.0, y1 + 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
            strip_vertex(x0 - 1.0, y1 + 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
            strip_vertex(x0 - 1.0, y0 - 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
            // Strip 3: inner rect (−1 px on every side) — MS_RAMKA_COLOR2.
            // Post-shifts from :185-196: outer shifts + (2, 2) / (-2, 2) /
            // (-2, -2) / (2, -2) land each corner 1 px inside the exact rect.
            strip_vertex(x0 + 1.0, y0 + 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
            strip_vertex(x1 - 1.0, y0 + 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
            strip_vertex(x1 - 1.0, y1 - 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
            strip_vertex(x0 + 1.0, y1 - 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
            strip_vertex(x0 + 1.0, y0 + 1.0, sw, sh, MS_RAMKA_COLOR2_RGBA, alpha_scale),
        ];
        self.vertex_count = MAX_VERTS as u32;
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&verts));
    }
}

fn strip_vertex(px: f32, py: f32, sw: f32, sh: f32, rgba: [f32; 4], a: f32) -> MarqueeVertex {
    MarqueeVertex {
        pos: [2.0 * px / sw - 1.0, 1.0 - 2.0 * py / sh],
        color: [rgba[0], rgba[1], rgba[2], rgba[3] * a],
    }
}

const SHADER: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(pos, 0.0, 1.0);
    o.color = color;
    return o;
}

@fragment
fn fs_main(v: VOut) -> @location(0) vec4<f32> {
    return v.color;
}
"#;
