//! Headless per-map render cost probe (lavapipe): builds the full
//! MapRenderer for each map, camera at the mission start position, and
//! times encode+submit+wait per frame. Relative differences across maps
//! expose map-dependent render hot spots.
//!
//!   cargo run --example map_render_bench -- [frames] [map...]

use matrixgame_rs::matrix_game::camera::Camera;
use matrixgame_rs::matrix_game::effects::point_light::PointLightSystem;
use matrixgame_rs::matrix_game::logic::MapLogic;
use matrixgame_rs::matrix_game::map::{GameMap, MapRenderer, MapScope};
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use std::time::Instant;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .init();
    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let dat = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let frames: usize = args
        .first()
        .and_then(|s| s.parse().ok())
        .map(|n| {
            args.remove(0);
            n
        })
        .unwrap_or(30);
    let maps: Vec<String> = if args.is_empty() {
        ["ATOLL", "CROSSFIRE", "ISLANDS", "ASYLUM", "VIRUS", "ARMAGEDD"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        args
    };

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        flags: wgpu::InstanceFlags::empty(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no adapter");
    println!("adapter: {:?}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .expect("no device");

    const W: u32 = 1280;
    const H: u32 = 720;
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

    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench color"),
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

    for name in &maps {
        let path = format!("MATRIX/MAP/{}.CMAP", name.to_uppercase());
        let cmap = pkg.read_file(&path).unwrap();
        let stor = Storage::from_bytes(&cmap).unwrap();
        let map = GameMap::from_cmap_bytes(&cmap).unwrap();
        matrixgame_rs::matrix_game::map::set_map_name(&path);

        let tex_reader = |p: &str| -> Option<Vec<u8>> {
            let key = p.to_uppercase();
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
        if let Ok(cam) = std::env::var("MG_CAM") {
            let mut it = cam.split(',').filter_map(|v| v.trim().parse::<f32>().ok());
            let (x, y) = (it.next().unwrap_or(0.0), it.next().unwrap_or(0.0));
            camera.set_xy_strategy([x, y]);
        } else if let Some(pos) = map.camera_pos {
            camera.set_xy_strategy(pos);
        } else {
            camera.set_xy_strategy([map.world_width() * 0.5, map.world_height() * 0.5]);
        }
        {
            let m = std::sync::Arc::new(GameMap::from_cmap_bytes(&cmap).unwrap());
            let m2 = m.clone();
            camera.set_terrain_sampler(Box::new(move |x, y| {
                m.group_max_z_interpolated(x, y, f32::MAX)
            }));
            camera.set_ground_sampler(Box::new(move |x, y| m2.get_z(x, y)));
        }
        camera.takt(10_000.0);

        let t_build = Instant::now();
        let mut terrain = MapRenderer::new(&device, &queue, &config, &map, &stor, &dat, &tex_reader);
        let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

        let point_lights = PointLightSystem::new(&map);

        // Live arena with the map's authored robots / cannons / buildings
        // so the robot + cannon sync/render paths carry realistic load.
        let mut game = MapLogic::with_seed(1);
        game.load_config(&dat);
        game.spawn_buildings(&map);
        game.spawn_ruins(&map);
        game.spawn_robots(&map);
        game.ensure_sides_from_objects();
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            game.takt(50);
        }

        // Warmup frame (pipeline compiles etc.)
        for _ in 0..3 {
            render_once(
                &device, &queue, &color_view, &mut terrain, &camera, &map, &point_lights,
                &mut game,
            );
        }

        let mut sum = 0f64;
        let mut worst = 0f64;
        for _ in 0..frames {
            let t0 = Instant::now();
            render_once(
                &device, &queue, &color_view, &mut terrain, &camera, &map, &point_lights,
                &mut game,
            );
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            sum += ms;
            worst = worst.max(ms);
        }
        println!(
            "{name:10} frame avg={:.2}ms worst={:.2}ms (build {build_ms:.0}ms)",
            sum / frames as f64,
            worst
        );

        if std::env::var("MG_SHOT").is_ok() {
            save_png(&device, &queue, &color, W, H, &format!("shot_{name}.png"));
        }
    }
}

fn save_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color: &wgpu::Texture,
    w: u32,
    h: u32,
    path: &str,
) {
    let bpr = (w * 4 + 255) & !255;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (bpr * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: color,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = slice.get_mapped_range();
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        let row = &data[(y * bpr) as usize..(y * bpr + w * 4) as usize];
        for x in 0..w {
            let p = &row[(x * 4) as usize..(x * 4 + 4) as usize];
            img.put_pixel(x, y, image::Rgba([p[0], p[1], p[2], 255]));
        }
    }
    img.save(path).unwrap();
    println!("saved {path}");
}

#[allow(clippy::too_many_arguments)]
fn render_once(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_view: &wgpu::TextureView,
    terrain: &mut MapRenderer,
    camera: &Camera,
    map: &GameMap,
    point_lights: &PointLightSystem,
    game: &mut MapLogic,
) {
    terrain.takt(16.6, map, point_lights, camera, device, queue);
    {
        let _scope = MapScope::enter(map, game.elapsed_ms);
        terrain.sync_robots(
            device,
            queue,
            &mut game.objects,
            map,
            point_lights,
            17,
            None,
            &[],
            camera,
        );
    }
    let vp = camera.view_proj();
    let vm = camera.view_matrix();
    let mut encoder = device.create_command_encoder(&Default::default());
    terrain.render(device, &mut encoder, color_view, queue, camera, vp, vm, map);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}
