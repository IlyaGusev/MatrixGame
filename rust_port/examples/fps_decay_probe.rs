//! Long-run FPS decay probe: full game (AI on every side) + full render
//! loop on one map for N sim-minutes, timing each frame phase and
//! dumping accumulation counters every 30 sim-seconds. Repro harness
//! for "FPS collapses after ~10 minutes of play".
//!
//!   cargo run --example fps_decay_probe -- [map] [minutes] [render_every_steps]

use matrixgame_rs::matrix_game::camera::Camera;
use matrixgame_rs::matrix_game::effects::effects_renderer::{EffectMeshRenderer, EffectsRenderer};
use matrixgame_rs::matrix_game::effects::explosion::MeshQueue;
use matrixgame_rs::matrix_game::effects::landscape_spot::LandscapeSpots;
use matrixgame_rs::matrix_game::effects::point_light::PointLightSystem;
use matrixgame_rs::matrix_game::logic::{robot_ref, MapLogic};
use matrixgame_rs::matrix_game::map::{GameMap, MapRenderer, MapScope};
use matrixgame_rs::matrix_game::map_static::ObjectType;
use matrixgame_rs::matrix_game::side::Side;
use matrixgame_rs::matrix_lib::base::pack::PkgArchive;
use matrixgame_rs::matrix_lib::base::storage::Storage;
use matrixgame_rs::matrix_lib::three_g::billboard::BillboardQueue;
use std::collections::HashMap;
use std::time::Instant;

const TAKT_MS: i32 = 50;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Error)
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let map_name = args.first().map(|s| s.as_str()).unwrap_or("DESERT1_3E");
    let minutes: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(14.0);
    let render_every: i64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);

    let pkg = PkgArchive::from_bytes(std::fs::read("../Data/robots.pkg").unwrap()).unwrap();
    let dat = Storage::from_bytes(&std::fs::read("../Data/robots.dat").unwrap()).unwrap();
    let path = format!("MATRIX/MAP/{}.CMAP", map_name.to_uppercase());
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
        label: Some("probe color"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color.create_view(&Default::default());

    let mut camera = Camera::new(W as f32 / H as f32);
    camera.apply_camera_config(&dat);
    camera.set_map(map.world_width(), map.world_height());
    camera.set_aspect(W as f32, H as f32);
    camera.init_strategy_angle(map.camera_angle);
    if let Some(pos) = map.camera_pos {
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

    let mut terrain = MapRenderer::new(&device, &queue, &config, &map, &stor, &dat, &tex_reader);
    let mut point_lights = PointLightSystem::new(&map);
    let mut effects_renderer = EffectsRenderer::new(
        &device,
        &queue,
        &config,
        matrixgame_rs::matrix_lib::three_g::texture::DEPTH_FORMAT,
        &dat,
        &tex_reader,
    );
    let mut effect_meshes = EffectMeshRenderer::new(
        &device,
        &queue,
        &config,
        matrixgame_rs::matrix_lib::three_g::texture::DEPTH_FORMAT,
        &dat,
        &tex_reader,
    );
    let mut spots = LandscapeSpots::default();
    let mut bb_queue = BillboardQueue::default();
    let mut mesh_queue = MeshQueue::default();
    let mut light_follow_map: HashMap<
        matrixgame_rs::matrix_game::effects::weapon::WeaponId,
        matrixgame_rs::matrix_game::effects::point_light::PointLightId,
    > = HashMap::new();

    // Full-auto game: every side AI-driven so the battle runs the whole probe.
    let mut game = MapLogic::with_seed(1);
    game.load_config(&dat);
    matrixgame_rs::matrix_game::map::set_full_auto(true);
    game.player_side = Side::new(100);
    game.spawn_buildings(&map);
    game.spawn_ruins(&map);
    game.spawn_robots(&map);
    game.ensure_sides_from_objects();
    game.apply_side_resources(&map);
    game.init_effect_spawners(&map);
    game.accrue_resources(100_000);
    game.spawn_map_objects(&map, &stor);
    game.objects.debris_catalog_len = effect_meshes.debris_count();
    game.objects.debris_types = effect_meshes.debris_types().to_vec();

    let steps = (minutes * 60.0 * 1000.0 / TAKT_MS as f64) as i64;
    // Phase accumulators: [takt, graphic, drains, terrain.takt, sync_bld,
    // sync_robots, bb_build, upload, encode+submit+wait]
    let mut acc = [0f64; 9];
    let mut frames = 0u32;
    let mut worst_render = 0f64;
    let mut outcome: Option<String> = None;

    for step in 0..steps {
        // Camera chases the battle so effect culling matches real play.
        let centroid = battle_centroid(&game);
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            matrixgame_rs::matrix_game::map::set_frustum_center(centroid);
            camera.set_xy_strategy(centroid);
            camera.takt(TAKT_MS as f32);

            let t0 = Instant::now();
            game.takt(TAKT_MS);
            let t1 = Instant::now();
            game.graphic_takt(TAKT_MS);
            let t2 = Instant::now();
            acc[0] += (t1 - t0).as_secs_f64();
            acc[1] += (t2 - t1).as_secs_f64();

            // form_game drains: spots, point lights, follow lights, kills.
            let t3 = Instant::now();
            spots.takt(TAKT_MS as f32);
            let pending: Vec<_> = game.objects.pending_spots.drain(..).collect();
            for sp in pending {
                spots.spawn(&map, &sp);
            }
            let pending_lights: Vec<_> = game.objects.pending_point_lights.drain(..).collect();
            for pl in pending_lights {
                point_lights.add_transient_light_anim(
                    &map, pl.pos, pl.r1, pl.r2, pl.c1, pl.c2, pl.ttl, pl.t1,
                );
            }
            let follows: Vec<_> = game.objects.pending_light_follow.drain(..).collect();
            for (key, pos, radius, color) in follows {
                if let Some(&id) = light_follow_map.get(&key) {
                    point_lights.set_pos(&map, id, pos);
                    point_lights.set_radius(&map, id, radius);
                    point_lights.set_color(&map, id, color);
                } else {
                    let id = point_lights.add_light(&map, pos, radius, color);
                    light_follow_map.insert(key, id);
                }
            }
            let kills: Vec<_> = game.objects.pending_light_kill.drain(..).collect();
            for key in kills {
                if let Some(id) = light_follow_map.remove(&key) {
                    point_lights.remove_light(&map, id);
                }
            }
            point_lights.takt(&map, TAKT_MS as f32);
            point_lights.flush_throttled(&map, game.elapsed_ms as f64, 50.0);
            acc[2] += t3.elapsed().as_secs_f64();
        }
        game.objects.pending_sounds.clear();
        game.sound_queue.clear();
        game.objects.weapons.freed.clear();
        let _ = matrixgame_rs::matrix_game::interface::sound::drain();
        if let Some(win) = game.pending_win_loose_dialog.take() {
            if outcome.is_none() {
                outcome = Some(format!("{} at t={}s", if win { "WIN" } else { "LOSE" }, game.elapsed_ms / 1000));
            }
        }

        if step % render_every != 0 {
            continue;
        }
        frames += 1;
        let tr0 = Instant::now();
        {
            let _scope = MapScope::enter(&map, game.elapsed_ms);
            let t = Instant::now();
            terrain.takt(
                TAKT_MS as f32 * render_every as f32,
                &map,
                &point_lights,
                &camera,
                &device,
                &queue,
            );
            acc[3] += t.elapsed().as_secs_f64();
            let t = Instant::now();
            terrain.sync_building_animation(&queue, &game.objects, &map, &point_lights);
            acc[4] += t.elapsed().as_secs_f64();
            let t = Instant::now();
            terrain.sync_robots(
                &device,
                &queue,
                &mut game.objects,
                &map,
                &point_lights,
                TAKT_MS * render_every as i32,
                None,
                &[],
                &camera,
            );
            acc[5] += t.elapsed().as_secs_f64();
        }

        let vp = camera.view_proj();
        let vm = camera.view_matrix();
        let cr = camera.camera_right_world();
        let cu = camera.camera_up_world();
        let mc = glam::Vec2::new(map.world_width() * 0.5, map.world_height() * 0.5);

        // Build the effect-primitive queues like form_game's render block.
        let t = Instant::now();
        bb_queue.clear();
        if let Some(r) = terrain.robots() {
            r.emit_light_billboards(&mut bb_queue);
        }
        mesh_queue.draws.clear();
        for e in &game.effects {
            e.draw(&mut bb_queue, &mut mesh_queue);
        }
        {
            let objs = &game.objects;
            let rng = &mut game.rng;
            for w in objs.weapons.iter() {
                w.draw(&mut bb_queue, objs, rng, false);
            }
            for id in objs.iter_units() {
                let Some(o) = objs.get(id) else { continue };
                if o.core().obj_type == ObjectType::RobotAi {
                    let r: &matrixgame_rs::matrix_game::robot::Robot = unsafe {
                        &*(o as *const dyn matrixgame_rs::matrix_game::map_static::MapStatic
                            as *const matrixgame_rs::matrix_game::robot::Robot)
                    };
                    if r.state == matrixgame_rs::matrix_game::robot::RobotState::Dip {
                        r.draw_dip(&mut bb_queue);
                    }
                }
            }
        }
        acc[6] += t.elapsed().as_secs_f64();
        let bb_count = bb_queue.billboards.len() + bb_queue.lines.len() + bb_queue.tris.len();
        let mesh_count = mesh_queue.draws.len();

        let t = Instant::now();
        let inv_view = vm.inverse();
        let cam_pos = glam::Vec3::new(
            inv_view.w_axis.x + mc.x,
            inv_view.w_axis.y + mc.y,
            inv_view.w_axis.z,
        );
        effects_renderer.upload(&device, &queue, &bb_queue, vp, vm, cam_pos, cr, cu, mc);
        effects_renderer.upload_spots(&device, &queue, &spots, mc);
        effect_meshes.upload(&queue, &mesh_queue, vp, mc);
        acc[7] += t.elapsed().as_secs_f64();

        let t = Instant::now();
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
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Effects Pass"),
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
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            effects_renderer.render_spots(&mut pass);
            effect_meshes.render(&mut pass);
            effects_renderer.render(&mut pass);
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        acc[8] += t.elapsed().as_secs_f64();
        worst_render = worst_render.max(tr0.elapsed().as_secs_f64());

        // 30-sim-second report.
        if (step / render_every) % (30_000 / (TAKT_MS * render_every as i32)) as i64 == 0 {
            let n = frames as f64;
            let mut fx_kinds = [0usize; 14];
            for e in &game.effects {
                fx_kinds[e.kind_index()] += 1;
            }
            let mut robots = 0;
            let mut cannons = 0;
            let mut live = 0;
            for id in game.objects.iter_live() {
                live += 1;
                if robot_ref(&game.objects, id).is_some() {
                    robots += 1;
                }
                if let Some(o) = game.objects.get(id) {
                    if o.core().obj_type == ObjectType::Cannon {
                        cannons += 1;
                    }
                }
            }
            println!(
                "t={:>4}s frame={:5.1}ms worst={:6.1} [takt={:4.1} gfx={:4.1} drain={:4.1} ttakt={:4.1} sbld={:4.1} srob={:4.1} bb={:4.1} up={:4.1} enc={:5.1}] fx={:3} bb={:4} mesh={:3} spots={:3} lights={:3} follow={:3} weap={:3} live={:3} rob={:2} can={:2} kinds={:?}",
                game.elapsed_ms / 1000,
                (acc[3..].iter().sum::<f64>()) / n * 1000.0,
                worst_render * 1000.0,
                acc[0] / (step + 1) as f64 * 1000.0,
                acc[1] / (step + 1) as f64 * 1000.0,
                acc[2] / (step + 1) as f64 * 1000.0,
                acc[3] / n * 1000.0,
                acc[4] / n * 1000.0,
                acc[5] / n * 1000.0,
                acc[6] / n * 1000.0,
                acc[7] / n * 1000.0,
                acc[8] / n * 1000.0,
                game.effects.len(),
                bb_count,
                mesh_count,
                spots.spots.len(),
                point_lights.lights().len(),
                light_follow_map.len(),
                game.objects.weapons.iter().count(),
                live,
                robots,
                cannons,
                fx_kinds,
            );
            acc = [0.0; 9];
            frames = 0;
            worst_render = 0.0;
        }
    }
    if let Some(o) = outcome {
        println!("game outcome: {o}");
    }
}

fn battle_centroid(game: &MapLogic) -> [f32; 2] {
    let mut n = 0f32;
    let mut cx = 0f32;
    let mut cy = 0f32;
    for id in game.objects.iter_live() {
        if let Some(r) = robot_ref(&game.objects, id) {
            cx += r.pos_x;
            cy += r.pos_y;
            n += 1.0;
        }
    }
    if n > 0.0 {
        [cx / n, cy / n]
    } else {
        [0.0, 0.0]
    }
}
