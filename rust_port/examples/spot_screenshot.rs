//! Headless screenshot of landscape burn decals — renders the terrain
//! plus a cluster of voronka/constant spots into an offscreen texture
//! (lavapipe works, no display needed) and saves a PNG so decal bugs
//! can be seen without a browser session.
//!
//!   cargo run --example spot_screenshot -- [x] [y]
//!
//! Defaults to a shore point on atoll. Output: scratch spot_shot.png.

use glam::{Vec2, Vec3};
use matrixgame_rs::matrix_game::camera::Camera;
use matrixgame_rs::matrix_game::effects::landscape_spot::{LandscapeSpots, SpotKind, SpotSpawn};
use matrixgame_rs::matrix_game::map::{GameMap, MapRenderer};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;

fn main() {
    env_logger::init();
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let cmap = pkg.read_file("MATRIX/MAP/ATOLL.CMAP").unwrap();
    let stor = Storage::from_bytes(&cmap).unwrap();
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    let dat = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();
    matrixgame_rs::matrix_game::map::set_map_name("MATRIX/MAP/ATOLL.CMAP");

    let cx: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1030.0);
    let cy: f32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(560.0);

    // ── wgpu offscreen device ───────────────────────────────────────
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        // Old lavapipe advertises debug_utils but can't load its entry
        // points — keep instance flags empty.
        flags: wgpu::InstanceFlags::empty(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no adapter");
    println!("adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .expect("no device");

    const W: u32 = 1024;
    const H: u32 = 768;
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: W,
        height: H,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    let tex_reader = |path: &str| -> Option<Vec<u8>> {
        let key = path.to_uppercase();
        for candidate in [&key, &format!("{key}.PNG"), &format!("{key}.DDS")] {
            if let Ok(data) = pkg.read_file(candidate) {
                return Some(data);
            }
        }
        None
    };

    let mut camera = Camera::new(W as f32 / H as f32);
    camera.apply_camera_config(&dat);
    camera.set_map(map.world_width(), map.world_height());
    camera.set_aspect(W as f32, H as f32);
    camera.init_strategy_angle(map.camera_angle);
    camera.set_xy_strategy([cx, cy]);
    {
        let m = std::sync::Arc::new(map);
        let m2 = m.clone();
        camera.set_terrain_sampler(Box::new(move |x, y| {
            m.group_max_z_interpolated(x, y, f32::MAX)
        }));
        camera.set_ground_sampler(Box::new(move |x, y| m2.get_z(x, y)));
    }
    // Re-parse the map (camera samplers consumed the Arc).
    let map = GameMap::from_cmap_bytes(&cmap).unwrap();
    camera.takt(10_000.0); // settle interpolation

    let mut terrain = MapRenderer::new(&device, &queue, &config, &map, &stor, &dat, &tex_reader);

    let mut effects = matrixgame_rs::matrix_game::effects::effects_renderer::EffectsRenderer::new(
        &device,
        &queue,
        &config,
        wgpu::TextureFormat::Depth32Float,
        &dat,
        &tex_reader,
    );

    // ── spawn decals around the focus (crater + burn marks) ────────
    let mut spots = LandscapeSpots::default();
    let mut spawn = |kind: SpotKind, x: f32, y: f32, scale: f32, angle: f32| {
        spots.spawn(
            &map,
            &SpotSpawn {
                pos: Vec2::new(x, y),
                angle,
                scale,
                kind,
            },
        );
    };
    // Carpet the area with craters + burn marks (battlefield-like),
    // including sloped/warped cells.
    for iy in -3..=3 {
        for ix in -3..=3 {
            let (fx, fy) = (cx + ix as f32 * 55.0, cy + iy as f32 * 55.0);
            let a = (ix * 7 + iy * 3) as f32 * 0.6;
            if (ix + iy) % 2 == 0 {
                spawn(SpotKind::Voronka, fx, fy, 4.0 + (ix as f32).abs() * 0.8, a);
            } else {
                spawn(SpotKind::Constant, fx, fy, 3.0, a);
            }
        }
    }
    println!(
        "spawned {} spots ({} verts / {} idx in first)",
        spots.spots.len(),
        spots.spots.first().map(|s| s.verts.len()).unwrap_or(0),
        spots.spots.first().map(|s| s.indices.len()).unwrap_or(0),
    );
    // Freeze fade at full visibility.
    for s in spots.spots.iter_mut() {
        s.takt(1200.0); // past the voronka fade-in ramp
    }

    // ── render ──────────────────────────────────────────────────────
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shot color"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color.create_view(&Default::default());

    let vp = camera.view_proj();
    let vm = camera.view_matrix();
    let mc = Vec2::new(camera.map_cx(), camera.map_cy());

    let mut encoder = device.create_command_encoder(&Default::default());
    terrain.render(&device, &mut encoder, &color_view, &queue, &camera, vp, vm, &map);

    // Spots — same pass shape as form_game's effects pass.
    effects.upload(
        &device,
        &queue,
        &matrixgame_rs::matrix_lib::three_g::billboard::BillboardQueue::default(),
        vp,
        vm,
        camera.eye_pos_world(),
        camera.camera_right_world(),
        camera.camera_up_world(),
        mc,
    );
    effects.upload_spots(&device, &queue, &spots, mc);
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("spots"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: terrain.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        effects.render_spots(&mut pass);
    }

    // ── read back ───────────────────────────────────────────────────
    let bpr = (W * 4 + 255) & !255;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (bpr * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = slice.get_mapped_range();
    let mut img = image::RgbaImage::new(W, H);
    for y in 0..H {
        let row = &data[(y * bpr) as usize..(y * bpr + W * 4) as usize];
        for x in 0..W {
            let i = (x * 4) as usize;
            img.put_pixel(x, y, image::Rgba([row[i], row[i + 1], row[i + 2], 255]));
        }
    }
    let out = std::env::var("SHOT_OUT").unwrap_or_else(|_| "spot_shot.png".into());
    img.save(&out).unwrap();
    println!("wrote {out}");
}
