//! Headless per-map render cost probe (lavapipe): builds the full
//! MapRenderer for each map, camera at the mission start position, and
//! times encode+submit+wait per frame. Relative differences across maps
//! expose map-dependent render hot spots.
//!
//!   cargo run --example map_render_bench -- [frames] [map...]
//!
//! Defaults: 30 frames over a fixed 6-map spread. Env knobs:
//! MG_CAM=x,y,dist,angz,angx (camera override), MG_SHOT=1 (save a PNG
//! per map), MG_SPAWNTEST=1 (queue a robot build), MG_STEP=ms (logic
//! step), MG_NOSCOPE=1 (replicate the missing-MapScope sync bug),
//! MG_PICK=px,py[,z] (unproject a screen pixel).

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

        // MG_SPAWNTEST=1: queue a robot on the player base and capture
        // frames of the spawn sequence — headless repro of "robot can't
        // leave the pod".
        if std::env::var("MG_SPAWNTEST").is_ok() {
            use matrixgame_rs::matrix_game::config::RobotUnitKind;
            use matrixgame_rs::matrix_game::interface::constructor::{RobotConfig, Unit};
            use matrixgame_rs::matrix_game::logic::{building_mut, building_ref};
            use matrixgame_rs::matrix_game::map_static::MapStatic;
            use matrixgame_rs::matrix_game::object_robot::RobotUnitType;
            let base_id = game
                .objects
                .iter_live()
                .find(|&id| {
                    building_ref(&game.objects, id)
                        .map(|b| b.is_live() && b.is_base() && b.side == game.player_side.id)
                        .unwrap_or(false)
                })
                .expect("no player base");
            let base_pos = building_ref(&game.objects, base_id).map(|b| b.pos).unwrap();
            camera.set_xy_strategy([base_pos.x, base_pos.y]);
            camera.takt(10_000.0);
            let mut cfg = RobotConfig::new();
            cfg.chassis = Unit {
                ty: RobotUnitType::Chassis,
                kind: RobotUnitKind(1),
                price: Default::default(),
            };
            cfg.hull.unit = Unit {
                ty: RobotUnitType::Armor,
                kind: RobotUnitKind(6),
                price: Default::default(),
            };
            cfg.head = Unit {
                ty: RobotUnitType::Head,
                kind: RobotUnitKind(0),
                price: Default::default(),
            };
            for i in 0..4 {
                cfg.weapon[i] = Unit {
                    ty: RobotUnitType::Weapon,
                    kind: RobotUnitKind(3),
                    price: Default::default(),
                };
            }
            building_mut(&mut game.objects, base_id)
                .map(|b| b.queue_robot(cfg))
                .unwrap();
            let step_ms: i32 = std::env::var("MG_STEP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50);
            let mut shot_no = 0;
            for step in 0..(16_000 / step_ms * 1) {
                {
                    let _scope = MapScope::enter(&map, game.elapsed_ms);
                    matrixgame_rs::matrix_game::map::set_frustum_center([base_pos.x, base_pos.y]);
                    game.takt(step_ms);
                    game.graphic_takt(step_ms);
                }
                game.objects.pending_sounds.clear();
                game.sound_queue.clear();
                game.objects.weapons.freed.clear();
                game.objects.pending_spots.clear();
                game.objects.pending_point_lights.clear();
                game.objects.pending_light_follow.clear();
                game.objects.pending_light_kill.clear();
                let _ = matrixgame_rs::matrix_game::interface::sound::drain();
                {
                    // MG_NOSCOPE=1 replicates form_game's bug: syncs ran
                    // outside the MapScope, so current_elapsed_ms() == 0.
                    let _scope = if std::env::var("MG_NOSCOPE").is_err() {
                        Some(MapScope::enter(&map, game.elapsed_ms))
                    } else {
                        None
                    };
                    terrain.takt(step_ms as f32, &map, &point_lights, &camera, &device, &queue);
                    terrain.sync_building_animation(
                        &queue,
                        &game.objects,
                        &map,
                        &point_lights,
                    );
                    terrain.sync_robots(
                        &device,
                        &queue,
                        &mut game.objects,
                        &map,
                        &point_lights,
                        step_ms,
                        None,
                        &[],
                        &camera,
                    );
                    let vp = camera.view_proj();
                    let vm = camera.view_matrix();
                    let mut encoder = device.create_command_encoder(&Default::default());
                    terrain.render(
                        &device,
                        &mut encoder,
                        &color_view,
                        &queue,
                        &camera,
                        vp,
                        vm,
                        &map,
                    );
                    queue.submit([encoder.finish()]);
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                }
                // At t=8s: select the robot and order it 300 units north
                // through the real pg_order path (the user's scenario).
                if (step * step_ms) <= 8000 && ((step + 1) * step_ms) > 8000 {
                    let rid = game.objects.iter_live().find(|&id| {
                        matrixgame_rs::matrix_game::logic::robot_ref(&game.objects, id)
                            .map(|r| r.side == game.player_side.id)
                            .unwrap_or(false)
                    });
                    if let Some(rid) = rid {
                        let _scope = MapScope::enter(&map, game.elapsed_ms);
                        let no = game.robot_to_logic_group(rid);
                        let (tx, ty) = {
                            let r = matrixgame_rs::matrix_game::logic::robot_ref(&game.objects, rid)
                                .unwrap();
                            (
                                (r.pos_x / GameMap::GLOBAL_SCALE_MOVE) as i32,
                                (r.pos_y / GameMap::GLOBAL_SCALE_MOVE) as i32 - 30,
                            )
                        };
                        game.pg_order_move_to(&map, no, (tx, ty));
                        println!("t={}ms ORDER move to ({tx},{ty})", game.elapsed_ms);
                    }
                }
                if (step * step_ms) % 1000 < step_ms {
                    let mut seen = false;
                    for id in game.objects.iter_live().collect::<Vec<_>>() {
                        if let Some(r) =
                            matrixgame_rs::matrix_game::logic::robot_ref(&game.objects, id)
                        {
                            if r.side == game.player_side.id {
                                seen = true;
                                println!(
                                    "t={}ms robot pos=({:.0},{:.0},{:.1}) hp={:.0} state={:?}",
                                    game.elapsed_ms, r.pos_x, r.pos_y, r.pos_z, r.hit_point, r.state
                                );
                            }
                        }
                    }
                    if !seen {
                        println!("t={}ms NO PLAYER ROBOT ALIVE", game.elapsed_ms);
                    }
                }
                if (step * step_ms) % 2000 < step_ms {
                    shot_no += 1;
                    save_png(
                        &device,
                        &queue,
                        &color,
                        W,
                        H,
                        &format!("spawn_{name}_{shot_no:02}.png"),
                    );
                }
            }
        }

        // MG_PICK=px,py[,z]: unproject the screen pixel onto the z plane
        // (default derived by scanning plane heights) and print world pos.
        if let Ok(pick) = std::env::var("MG_PICK") {
            let mut it = pick.split(',').filter_map(|v| v.trim().parse::<f32>().ok());
            let (px, py) = (it.next().unwrap_or(0.0), it.next().unwrap_or(0.0));
            let zp = it.next().unwrap_or(92.0);
            let inv = camera.view_proj().inverse();
            let ndc_x = px / W as f32 * 2.0 - 1.0;
            let ndc_y = 1.0 - py / H as f32 * 2.0;
            let p0 = inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 0.0));
            let p1 = inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 0.9));
            let dir = (p1 - p0).normalize();
            let t = (zp - p0.z) / dir.z;
            let hit = p0 + dir * t;
            println!(
                "MG_PICK ({px},{py}) z={zp}: world ({:.1}, {:.1}) cell ({}, {})",
                hit.x + map.world_width() * 0.5,
                hit.y + map.world_height() * 0.5,
                ((hit.x + map.world_width() * 0.5) / 20.0) as i32,
                ((hit.y + map.world_height() * 0.5) / 20.0) as i32,
            );
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
