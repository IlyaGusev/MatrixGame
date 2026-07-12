//! Off-screen render-to-texture cache for build-stack robot icons.
//!
//! Port of the runtime-rendered preview path used by
//! `CIFaceList::CreateStackIcon`'s robot branch (CInterface.cpp:3975-
//! 3982). The C++ keeps a per-robot 64×64 "medium" texture
//! (`m_MedTexture`) baked once in `CMatrixRobotAI::CreateTextures`
//! (MatrixRobot.cpp:5342-5380). It composes the chassis+armor+head+
//! weapons stack via `RenderToTexture` (MatrixMapStatic.cpp:382+) using
//! the same per-unit graph the live world draw uses, then samples the
//! result whenever a stack icon needs a portrait.
//!
//! We mirror that: each unique `RobotConfig` queued for build gets a
//! 64×64 wgpu texture rendered via the existing
//! `RobotsRenderer::render_preview_full` path. The cache key is a hash
//! of the kind discriminants so identical configs share a texture.
//! The texture is registered with the `InterfaceRenderer` under a
//! `_robot_icon_<hash>` atlas key; the build-stack `IFaceElement`'s
//! `StateImage::tex_path` carries that key, so the existing
//! atlas-driven UI emit loop draws the icon with no additional
//! plumbing.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::matrix_game::interface::constructor::RobotConfig;
use crate::matrix_game::object_robot::{IconCamera, RobotsRenderer};
use crate::matrix_game::robot::ChassisKind;

use super::renderer::InterfaceRenderer;

/// Render size of the baked medium icon. The C++ uses 64 for
/// `m_MedTexture` (MatrixRobot.cpp:5347, 5365). Used for the build-stack
/// 25×25 / 42×42 portraits and the Main-panel 47×36 group icons.
///
/// Not rendered directly: `RenderToTexture` (MatrixMapStatic.cpp:648-652)
/// renders once at `RENDSZ` (256) and halves down to 64 with
/// `Make2xSmaller` + `sharpen_run(…, 16)` after each halving. The
/// [`MipFilter`] passes replicate that chain on the GPU.
const ICON_SIZE: u32 = 64;
/// Render size of the baked big portrait — port of `m_BigTexture`
/// (MatrixRobot.cpp:5347 — 256×256). The Main panel personal icon is
/// 114×114, the upscale to that size needs the higher source resolution.
const BIG_ICON_SIZE: u32 = 256;

pub struct RobotIconCache {
    /// Keyed by `(config_hash, size)` so med-vs-big share the same hash
    /// space without clobbering each other.
    entries: HashMap<(u64, u32), IconEntry>,
    /// Downsample+sharpen pipelines, created on the first 64px bake.
    filter: Option<MipFilter>,
}

struct IconEntry {
    atlas_key: String,
    /// Owned GPU texture kept alive while the entry exists. The bound
    /// `TextureView` lives inside the `InterfaceRenderer::atlases` map.
    _texture: wgpu::Texture,
}

impl RobotIconCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            filter: None,
        }
    }

    /// Get or bake the icon for `cfg`; returns the atlas key the UI
    /// element should reference, or `None` if rendering failed (e.g.
    /// chassis VO not loaded yet — happens on the first frames before
    /// asset loading completes).
    pub fn ensure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        robots: &RobotsRenderer,
        iface_renderer: &mut InterfaceRenderer,
        cfg: &RobotConfig,
    ) -> Option<String> {
        self.ensure_sized(
            device,
            queue,
            format,
            robots,
            iface_renderer,
            cfg,
            ICON_SIZE,
        )
    }

    /// Big variant — port of `m_BigTexture` (MatrixRobot.cpp:5347).
    /// Used by `CIFaceList::CreatePersonal` (CInterface.cpp:3805) for the
    /// 114×114 personal portrait on the Main panel.
    pub fn ensure_big(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        robots: &RobotsRenderer,
        iface_renderer: &mut InterfaceRenderer,
        cfg: &RobotConfig,
    ) -> Option<String> {
        self.ensure_sized(
            device,
            queue,
            format,
            robots,
            iface_renderer,
            cfg,
            BIG_ICON_SIZE,
        )
    }

    fn ensure_sized(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        robots: &RobotsRenderer,
        iface_renderer: &mut InterfaceRenderer,
        cfg: &RobotConfig,
        size: u32,
    ) -> Option<String> {
        let key = (config_hash(cfg), size);
        if let Some(e) = self.entries.get(&key) {
            return Some(e.atlas_key.clone());
        }
        if size == ICON_SIZE && self.filter.is_none() {
            self.filter = Some(MipFilter::new(device, format));
        }
        let texture = render_to_texture(device, queue, format, robots, cfg, size, self.filter.as_ref())?;
        let atlas_key = format!("_robot_icon_{}_{:016x}", size, key.0);
        let view = texture.create_view(&Default::default());
        iface_renderer.register_dynamic_atlas(device, &atlas_key, view, size, size);
        self.entries.insert(
            key,
            IconEntry {
                atlas_key: atlas_key.clone(),
                _texture: texture,
            },
        );
        Some(atlas_key)
    }
}

impl Default for RobotIconCache {
    fn default() -> Self {
        Self::new()
    }
}

fn config_hash(cfg: &RobotConfig) -> u64 {
    let mut h = DefaultHasher::new();
    cfg.chassis.kind.0.hash(&mut h);
    cfg.hull.unit.kind.0.hash(&mut h);
    cfg.head.kind.0.hash(&mut h);
    for w in &cfg.weapon {
        w.kind.0.hash(&mut h);
    }
    h.finish()
}

/// GPU pipelines for the RenderToTexture mip chain: `fs_downsample` =
/// `CBitmap::Make2xSmaller`, `fs_sharpen` = `sharpen_run(…, 16)`. Exact
/// integer math per `matrix_lib/bitmap/sharpen.rs` (the CPU reference).
struct MipFilter {
    bgl: wgpu::BindGroupLayout,
    downsample: wgpu::RenderPipeline,
    sharpen: wgpu::RenderPipeline,
}

impl MipFilter {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Icon Sharpen Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/icon_sharpen.wgsl").into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Icon Sharpen BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Icon Sharpen Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            bgl,
            downsample: make("fs_downsample"),
            sharpen: make("fs_sharpen"),
        }
    }

    /// One filter pass: full-target triangle reading `src`.
    fn pass(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
    ) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Icon Sharpen BG"),
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Icon Filter Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn render_to_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    robots: &RobotsRenderer,
    cfg: &RobotConfig,
    size: u32,
    filter: Option<&MipFilter>,
) -> Option<wgpu::Texture> {
    use crate::matrix_game::object_building::chassis_from_kind;

    let make_color_tex = |label: &str, sz: u32| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: sz,
                height: sz,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };

    // The C++ renders once at RENDSZ=256 whatever the requested texture
    // sizes; 64 is derived from that render by the mip chain below.
    let render_size = if size == ICON_SIZE { BIG_ICON_SIZE } else { size };
    let texture = make_color_tex("Robot Icon", render_size);
    let color_view = texture.create_view(&Default::default());

    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Robot Icon Depth"),
        size: wgpu::Extent3d {
            width: render_size,
            height: render_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());

    let chassis = chassis_from_kind(cfg.chassis.kind).unwrap_or(ChassisKind::Track);
    let kind_or_none = |k: i32| if k >= 1 { Some(k) } else { None };
    let weapon_kinds = [
        kind_or_none(cfg.weapon[0].kind.0),
        kind_or_none(cfg.weapon[1].kind.0),
        kind_or_none(cfg.weapon[2].kind.0),
        kind_or_none(cfg.weapon[3].kind.0),
        kind_or_none(cfg.weapon[4].kind.0),
    ];

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Robot Icon Encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Robot Icon Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: wgpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        robots.render_preview_full(
            queue,
            &mut pass,
            chassis,
            kind_or_none(cfg.hull.unit.kind.0),
            kind_or_none(cfg.head.kind.0),
            &weapon_kinds,
            0.0,
            [0, 0, render_size, render_size],
            render_size,
            render_size,
            Some(IconCamera::CPP_DEFAULTS),
            None,
        );
    }

    // Med-icon mip chain (MatrixMapStatic.cpp:648-652): 256 → half →
    // sharpen → 128 → half → sharpen → 64.
    let texture = if size == ICON_SIZE {
        let filter = filter?;
        let t128a = make_color_tex("Robot Icon 128a", 128);
        let t128b = make_color_tex("Robot Icon 128b", 128);
        let t64a = make_color_tex("Robot Icon 64a", 64);
        let t64b = make_color_tex("Robot Icon 64b", 64);
        let v128a = t128a.create_view(&Default::default());
        let v128b = t128b.create_view(&Default::default());
        let v64a = t64a.create_view(&Default::default());
        let v64b = t64b.create_view(&Default::default());
        filter.pass(device, &mut encoder, &filter.downsample, &color_view, &v128a);
        filter.pass(device, &mut encoder, &filter.sharpen, &v128a, &v128b);
        filter.pass(device, &mut encoder, &filter.downsample, &v128b, &v64a);
        filter.pass(device, &mut encoder, &filter.sharpen, &v64a, &v64b);
        t64b
    } else {
        texture
    };

    queue.submit(std::iter::once(encoder.finish()));
    Some(texture)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::matrix_lib::bitmap::sharpen::{make_2x_smaller, sharpen_run};

    /// The GPU mip chain must be byte-identical to the CPU reference
    /// port of Make2xSmaller + sharpen_run(lv=16).
    #[test]
    fn gpu_filter_matches_cpu_reference() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: None,
            }))
        else {
            eprintln!("no adapter available; skipping GPU filter test");
            return;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");

        const SZ: u32 = 64;
        // Deterministic pseudo-random source image (LCG).
        let mut seed = 0x1234_5678u32;
        let mut src_img = image::RgbaImage::new(SZ, SZ);
        for p in src_img.pixels_mut() {
            for c in 0..4 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                p.0[c] = (seed >> 24) as u8;
            }
        }

        let format = wgpu::TextureFormat::Rgba8Unorm;
        let filter = MipFilter::new(&device, format);

        let mk_tex = |sz: u32, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: sz,
                    height: sz,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let src_tex = mk_tex(
            SZ,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let half_a = mk_tex(
            SZ / 2,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let half_b = mk_tex(
            SZ / 2,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        queue.write_texture(
            src_tex.as_image_copy(),
            src_img.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SZ * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: SZ,
                height: SZ,
                depth_or_array_layers: 1,
            },
        );

        let v_src = src_tex.create_view(&Default::default());
        let v_a = half_a.create_view(&Default::default());
        let v_b = half_b.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        filter.pass(&device, &mut encoder, &filter.downsample, &v_src, &v_a);
        filter.pass(&device, &mut encoder, &filter.sharpen, &v_a, &v_b);

        // Read back half_b (rows padded to COPY_BYTES_PER_ROW_ALIGNMENT).
        let hsz = SZ / 2;
        let padded_row = (hsz * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded_row * hsz) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            half_b.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: hsz,
                height: hsz,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range();

        let expected = sharpen_run(&make_2x_smaller(&src_img), 16);
        for y in 0..hsz {
            let row = &data[(y * padded_row) as usize..][..(hsz * 4) as usize];
            for x in 0..hsz {
                let got = &row[(x * 4) as usize..][..4];
                let want = expected.get_pixel(x, y).0;
                assert_eq!(got, want, "pixel ({x},{y})");
            }
        }
    }
}
