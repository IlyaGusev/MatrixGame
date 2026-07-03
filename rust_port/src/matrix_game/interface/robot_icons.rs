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
const ICON_SIZE: u32 = 64;
/// Render size of the baked big portrait — port of `m_BigTexture`
/// (MatrixRobot.cpp:5347 — 256×256). The Main panel personal icon is
/// 114×114, the upscale to that size needs the higher source resolution.
const BIG_ICON_SIZE: u32 = 256;

pub struct RobotIconCache {
    /// Keyed by `(config_hash, size)` so med-vs-big share the same hash
    /// space without clobbering each other.
    entries: HashMap<(u64, u32), IconEntry>,
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
        let texture = render_to_texture(device, queue, format, robots, cfg, size)?;
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

fn render_to_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    robots: &RobotsRenderer,
    cfg: &RobotConfig,
    size: u32,
) -> Option<wgpu::Texture> {
    use crate::matrix_game::object_building::chassis_from_kind;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Robot Icon"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let color_view = texture.create_view(&Default::default());

    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Robot Icon Depth"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
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
                stencil_ops: None,
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
            [0, 0, size, size],
            size,
            size,
            Some(IconCamera::CPP_DEFAULTS),
            None,
        );
    }
    queue.submit(std::iter::once(encoder.finish()));
    Some(texture)
}
