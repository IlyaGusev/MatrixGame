use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use crate::matrix_game::effects::point_light::PointLightSystem;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::world::World;
use crate::matrix_game::camera::Camera;
use crate::gfx::context::GfxContext;
use crate::matrix_game::minimap::Minimap;
use crate::matrix_game::map::MapRenderer;
struct AppState {
    window: Arc<Window>,
    gfx: GfxContext,
    map: Arc<GameMap>,
    point_lights: PointLightSystem,
    terrain: MapRenderer,
    minimap: Minimap,
    camera: Camera,
    game: World,
    last_time: f64,
    cursor: [f32; 2],
    minimap_dragging: bool,
}

pub struct App {
    state: Rc<RefCell<Option<AppState>>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.borrow().is_some() {
            return;
        }

        let attrs = WindowAttributes::default().with_title("MatrixGame");

        #[cfg(target_arch = "wasm32")]
        let attrs = {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("game-canvas")
                .unwrap()
                .unchecked_into::<web_sys::HtmlCanvasElement>();
            attrs.with_canvas(Some(canvas))
        };

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        // Native: init synchronously
        #[cfg(not(target_arch = "wasm32"))]
        {
            let gfx = pollster::block_on(GfxContext::new(window.clone()));

            let (map, stor, matrix_data, pkg) = load_map();

            let map = Arc::new(map);
            let mut camera = Camera::new(gfx.config.width as f32 / gfx.config.height as f32);
            camera.apply_camera_config(&matrix_data);
            camera.set_map(map.world_width(), map.world_height());
            camera.set_aspect(gfx.config.width as f32, gfx.config.height as f32);
            camera.init_strategy_angle(map.camera_angle);
            if let Some(pos) = map.camera_pos {
                camera.set_xy_strategy(pos);
            }
            camera.set_terrain_sampler({
                let m = map.clone();
                Box::new(move |x, y| m.group_max_z_interpolated(x, y, f32::MAX))
            });
            camera.set_ground_sampler({
                let m = map.clone();
                Box::new(move |x, y| m.get_z(x, y))
            });

            let tex_reader = |path: &str| -> Option<Vec<u8>> {
                let key = path.to_uppercase();
                for candidate in [&key, &format!("{key}.PNG"), &format!("{key}.DDS")] {
                    if let Ok(data) = pkg.read_file(candidate) {
                        return Some(data);
                    }
                }
                None
            };
            let terrain = MapRenderer::new(
                &gfx.device,
                &gfx.queue,
                &gfx.config,
                &map,
                &stor,
                &matrix_data,
                &tex_reader,
            );
            let point_lights = PointLightSystem::new(&map);
            let mut minimap = Minimap::new(
                &gfx.device,
                &gfx.queue,
                gfx.config.format,
                &map,
                &matrix_data,
                &tex_reader,
            );
            minimap.set_angle(-map.camera_angle);

            let mut game = World::new();
            game.load_config(&matrix_data);
            let (_ids, stats) = game.spawn_map_objects(&map, &stor);
            log::info!(
                "world: spawned {} map objects (static={}, burn={}, break={}, anim={}, sens={}, spawner={}, terron={}, portret={}, special={})",
                stats.total(), stats.r#static, stats.burn, stats.r#break, stats.anim,
                stats.sens, stats.spawner, stats.terron, stats.portret, stats.special_win_target,
            );
            log::info!(
                "world: {} map objects enrolled in logic-temp list at init",
                game.objects.iter_logic().count(),
            );

            *self.state.borrow_mut() = Some(AppState {
                window,
                gfx,
                map,
                point_lights,
                terrain,
                minimap,
                camera,
                game,
                last_time: crate::platform::now_secs(),
                cursor: [-1.0, -1.0],
                minimap_dragging: false,
            });
        }

        // WASM: init asynchronously
        #[cfg(target_arch = "wasm32")]
        {
            let state_slot = self.state.clone();
            let win = window;
            wasm_bindgen_futures::spawn_local(async move {
                let mut gfx = GfxContext::new(win.clone()).await;

                let (map, stor, matrix_data, bundle) = load_map_async().await;
                let map = Arc::new(map);

                let mut camera = Camera::new(gfx.config.width as f32 / gfx.config.height as f32);
                camera.apply_camera_config(&matrix_data);
                camera.set_map(map.world_width(), map.world_height());
                camera.set_aspect(gfx.config.width as f32, gfx.config.height as f32);
                camera.init_strategy_angle(map.camera_angle);
                if let Some(pos) = map.camera_pos {
                    camera.set_xy_strategy(pos);
                }
                camera.set_terrain_sampler({
                    let m = map.clone();
                    Box::new(move |x, y| m.group_max_z_interpolated(x, y, f32::MAX))
                });
                camera.set_ground_sampler({
                    let m = map.clone();
                    Box::new(move |x, y| m.get_z(x, y))
                });

                let tex_reader =
                    |path: &str| -> Option<Vec<u8>> { bundle.read_file(path).map(|s| s.to_vec()) };
                let mut terrain = MapRenderer::new(
                    &gfx.device,
                    &gfx.queue,
                    &gfx.config,
                    &map,
                    &stor,
                    &matrix_data,
                    &tex_reader,
                );
                let point_lights = PointLightSystem::new(&map);
                let mut minimap = Minimap::new(
                    &gfx.device,
                    &gfx.queue,
                    gfx.config.format,
                    &map,
                    &matrix_data,
                    &tex_reader,
                );
                minimap.set_angle(-map.camera_angle);

                // Force resize to match actual canvas dimensions
                let size = win.inner_size();
                log::info!(
                    "wasm init: window inner_size = {}x{}",
                    size.width,
                    size.height
                );
                log::info!(
                    "wasm init: surface config = {}x{}",
                    gfx.config.width,
                    gfx.config.height
                );
                if size.width > 0 && size.height > 0 {
                    gfx.resize(size.width, size.height);
                    terrain.resize(&gfx.device, &gfx.config);
                    // set_aspect mirrors the *surface* size so cursor (which
                    // we rescale from winit coords on the fly) and edge-pan
                    // thresholds share the same coord system.
                    camera.set_aspect(
                        gfx.config.width as f32,
                        gfx.config.height as f32,
                    );
                }

                let mut game = World::new();
                let (_ids, stats) = game.spawn_map_objects(&map, &stor);
                log::info!(
                    "world: spawned {} map objects (static={}, burn={}, break={}, anim={}, sens={}, spawner={}, terron={}, portret={}, special={})",
                    stats.total(), stats.r#static, stats.burn, stats.r#break, stats.anim,
                    stats.sens, stats.spawner, stats.terron, stats.portret, stats.special_win_target,
                );
                log::info!(
                    "world: {} map objects enrolled in logic-temp list at init",
                    game.objects.iter_logic().count(),
                );

                *state_slot.borrow_mut() = Some(AppState {
                    window: win.clone(),
                    gfx,
                    map,
                    point_lights,
                    terrain,
                    minimap,
                    camera,
                    game,
                    last_time: crate::platform::now_secs(),
                    cursor: [-1.0, -1.0],
                    minimap_dragging: false,
                });
                win.request_redraw();
                hide_loading_overlay();
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let mut state_ref = self.state.borrow_mut();
        let state = match state_ref.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    state.gfx.resize(size.width, size.height);
                    state.terrain.resize(&state.gfx.device, &state.gfx.config);
                    state.camera.set_aspect(
                        state.gfx.config.width as f32,
                        state.gfx.config.height as f32,
                    );
                }
            }

            // ── Mouse input (MatrixFormGame.cpp:530-642) ──
            // Middle or right button toggles MouseCam mode (rotate-on-drag).
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                use winit::event::{ElementState, MouseButton};
                if matches!(button, MouseButton::Middle | MouseButton::Right) {
                    let pressed = btn_state == ElementState::Pressed;
                    let [x, y] = state.cursor;
                    state.camera.on_rotate_button(pressed, x, y);
                } else if button == MouseButton::Left {
                    use crate::matrix_game::minimap::MinimapClick;
                    let [cx, cy] = state.cursor;
                    match btn_state {
                        ElementState::Pressed => {
                            match state.minimap.click(cx, cy) {
                                MinimapClick::BeginDrag(tgt) => {
                                    state.camera.set_xy_strategy(tgt);
                                    state.minimap_dragging = true;
                                }
                                MinimapClick::ZoomIn
                                | MinimapClick::ZoomOut
                                | MinimapClick::None => {
                                    state.minimap_dragging = false;
                                }
                            }
                        }
                        ElementState::Released => {
                            state.minimap_dragging = false;
                        }
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                // winit reports cursor in its own "physical" coord system
                // (CSS × DPR). When the surface is clamped to
                // `max_texture_dimension_2d` (2048 on WebGL2) but the viewport
                // is larger, surface coords are a *subset* of winit coords —
                // rendering and hit-tests use the smaller surface coords, so
                // rescale cursor to the surface coord system here.
                let win_size = state.window.inner_size();
                let sx = if win_size.width > 0 {
                    state.gfx.config.width as f64 / win_size.width as f64
                } else {
                    1.0
                };
                let sy = if win_size.height > 0 {
                    state.gfx.config.height as f64 / win_size.height as f64
                } else {
                    1.0
                };
                let cx = (position.x * sx) as f32;
                let cy = (position.y * sy) as f32;
                state.cursor = [cx, cy];
                if state.minimap_dragging {
                    if let Some(tgt) = state.minimap.click_to_world(cx, cy) {
                        state.camera.set_xy_strategy(tgt);
                    }
                }
                state.camera.on_mouse_move(cx, cy);
            }

            WindowEvent::CursorLeft { .. } => {
                state.cursor = [-1.0, -1.0];
                state.camera.on_cursor_left();
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Each wheel notch = one ZoomInStep/OutStep call.
                use winit::event::MouseScrollDelta;
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                };
                state.camera.on_mouse_wheel(notches);
            }

            // ── Keyboard (MatrixFormGame.cpp:247-282) ──
            WindowEvent::KeyboardInput { event, .. } => {
                use crate::matrix_game::camera::KeyAction;
                use winit::keyboard::{KeyCode, PhysicalKey};
                let pressed = event.state == winit::event::ElementState::Pressed;
                let action =
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                            Some(KeyAction::MoveForward)
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown)
                        | PhysicalKey::Code(KeyCode::KeyS) => Some(KeyAction::MoveBack),
                        PhysicalKey::Code(KeyCode::ArrowLeft)
                        | PhysicalKey::Code(KeyCode::KeyA) => Some(KeyAction::MoveLeft),
                        PhysicalKey::Code(KeyCode::ArrowRight)
                        | PhysicalKey::Code(KeyCode::KeyD) => Some(KeyAction::MoveRight),
                        // Yaw (KA_ROTATE_LEFT/RIGHT). Original: Home/End + `[`/`]`.
                        PhysicalKey::Code(KeyCode::Home)
                        | PhysicalKey::Code(KeyCode::BracketLeft) => Some(KeyAction::RotLeft),
                        PhysicalKey::Code(KeyCode::End)
                        | PhysicalKey::Code(KeyCode::BracketRight) => Some(KeyAction::RotRight),
                        // Pitch (KA_ROTATE_UP/DOWN). Original: PageUp/PageDown.
                        PhysicalKey::Code(KeyCode::PageUp) => Some(KeyAction::RotUp),
                        PhysicalKey::Code(KeyCode::PageDown) => Some(KeyAction::RotDown),
                        // Reset angles (KA_CAM_SETDEFAULT). Original: `\`.
                        PhysicalKey::Code(KeyCode::Backslash) => Some(KeyAction::ResetAngles),
                        _ => None,
                    };
                if let Some(a) = action {
                    state.camera.on_key(a, pressed);
                }
            }

            WindowEvent::RedrawRequested => {
                let now = crate::platform::now_secs();
                let dt = (now - state.last_time) as f32;
                state.last_time = now;
                // Logic takt first (ports `CMatrixMapLogic::Takt`, which
                // runs `ProceedLogic` before the graphic takt starts —
                // MatrixLogic.cpp:2722-2761). Then per-object graphic
                // takt (SortEndGraphicTakt, MatrixMapStatic.cpp:755-765).
                // Camera / minimap / terrain takts mirror
                // `CMatrixMap::Takt`'s remaining subsystem dispatches.
                let step_ms = (dt * 1000.0).round() as i32;
                state.game.takt(step_ms);
                state.game.graphic_takt(step_ms);
                state.camera.takt(dt * 1000.0); // camera update (ms)
                state.minimap.takt(dt * 1000.0);
                state.terrain.takt(
                    dt * 1000.0,
                    &state.map,
                    &state.point_lights,
                    &state.camera,
                    &state.gfx.device,
                    &state.gfx.queue,
                ); // water animation + dynamic object tint updates

                // Bake the minimap background the first time — ports
                // CMinimap::RenderBackground (called once at map load). Must
                // be its own submission so `queue.write_buffer` doesn't get
                // clobbered by the main render's VP write.
                state.minimap.bake_background(
                    &state.gfx.device,
                    &state.gfx.queue,
                    &mut state.terrain,
                    &state.map,
                );
                match state.gfx.begin_frame() {
                    Ok((output, view, mut encoder)) => {
                        let vp = state.camera.view_proj();
                        let vm = state.camera.view_matrix();
                        state.terrain.render(
                            &state.gfx.device,
                            &mut encoder,
                            &view,
                            &state.gfx.queue,
                            &state.camera,
                            vp,
                            vm,
                            &state.map,
                        );
                        {
                            let mut pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Minimap Pass"),
                                    color_attachments: &[Some(
                                        wgpu::RenderPassColorAttachment {
                                            view: &view,
                                            resolve_target: None,
                                            depth_slice: None,
                                            ops: wgpu::Operations {
                                                load: wgpu::LoadOp::Load,
                                                store: wgpu::StoreOp::Store,
                                            },
                                        },
                                    )],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                });
                            state.minimap.render(
                                &state.gfx.queue,
                                &mut pass,
                                state.gfx.config.width as f32,
                                state.gfx.config.height as f32,
                                &state.map,
                                &state.camera,
                            );
                        }
                        state.gfx.end_frame(output, encoder);
                    }
                    Err(wgpu::SurfaceError::Lost) => {
                        let size = state.window.inner_size();
                        state.gfx.resize(size.width, size.height);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => log::warn!("surface error: {e:?}"),
                }

                state.window.request_redraw();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.borrow().as_ref() {
            state.window.request_redraw();
        }
    }
}

/// Load map — native reads from pkg, returns map + map storage + global
/// matrix data storage (robots.dat) + pkg for texture loading.
///
/// Matches the original startup contract (MatrixGame.cpp:240-257, 383-399):
/// the caller can request a specific map via the first CLI argument; if
/// absent, we fall back to `Config/Map` in the global data. Global data
/// itself is a required dependency (MatrixGame.cpp:240-257) — the engine
/// dereferences `g_MatrixData->BlockGet(...)` without NULL guards all over
/// Init, so missing `robots.dat` is a fatal startup error.
#[cfg(not(target_arch = "wasm32"))]
fn load_map() -> (
    GameMap,
    crate::matrix_lib::base::storage::Storage,
    crate::matrix_lib::base::storage::Storage,
    crate::matrix_lib::base::pack::PkgArchive,
) {
    use crate::matrix_lib::base::pack::PkgArchive;
    use crate::matrix_lib::base::storage::Storage;

    let candidates = [
        "../Data/robots.pkg",
        "Data/robots.pkg",
        "../../Data/robots.pkg",
    ];
    let pkg_path = candidates
        .iter()
        .find(|c| std::path::Path::new(c).exists())
        .unwrap_or(&"../Data/robots.pkg");

    log::info!("loading pkg: {}", pkg_path);
    let pkg_data = std::fs::read(pkg_path).expect("failed to read robots.pkg");
    let pkg = PkgArchive::from_bytes(pkg_data).expect("failed to parse pkg");

    // robots.dat is required, not optional — the original game crashes with
    // a null deref if it's absent (see MatrixGame.cpp:240-257, 371-373).
    let dat_candidates = [
        "../Data/robots.dat",
        "Data/robots.dat",
        "../../Data/robots.dat",
    ];
    let dat_path = dat_candidates
        .iter()
        .find(|c| std::path::Path::new(c).exists())
        .expect(
            "robots.dat is required; place it next to robots.pkg (e.g. ../Data/robots.dat)",
        );
    log::info!("loading matrix data: {}", dat_path);
    let dat_bytes = std::fs::read(dat_path).expect("failed to read robots.dat");
    let matrix_data =
        Storage::from_bytes(&dat_bytes).expect("failed to parse robots.dat CStorage");

    // Pick the map the same way the original does (MatrixGame.cpp:383-394):
    // CLI arg wins; otherwise read `Config/Map` from the global data.
    let cli_map = std::env::args().nth(1);
    let mapname = match cli_map.as_deref() {
        Some(arg) if !arg.is_empty() => resolve_map_name(arg),
        _ => {
            let config_rec = matrix_data
                .block_record("da", "Config")
                .expect("robots.dat has no Config block");
            let map_param = matrix_data
                .block_param(&config_rec, "Map")
                .filter(|s| !s.is_empty())
                .expect(
                    "no map requested and Config/Map is missing — pass a map name as the first CLI arg",
                );
            resolve_map_name(&map_param)
        }
    };
    log::info!("loading map: {}", mapname);

    let cmap_data = pkg
        .read_file(&mapname)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", mapname, e));
    let stor = Storage::from_bytes(&cmap_data).expect("failed to parse map CStorage");
    let map = GameMap::from_cmap_bytes(&cmap_data).expect("failed to parse CMAP");

    (map, stor, matrix_data, pkg)
}

/// Mirror the path-building in `MatrixGame::Init` (MatrixGame.cpp:385-394):
/// a bare name like `Atoll` becomes `Matrix\Map\Atoll.CMAP`; anything with
/// a path separator is used verbatim. We normalize to forward slashes,
/// uppercase for the pkg's case-folded lookup, and append `.CMAP` when the
/// caller didn't supply it.
#[cfg(not(target_arch = "wasm32"))]
fn resolve_map_name(requested: &str) -> String {
    let trimmed = requested.trim();
    let has_sep = trimmed.contains('/') || trimmed.contains('\\');
    let normalized = trimmed.replace('\\', "/");
    let with_prefix = if has_sep {
        normalized
    } else {
        format!("MATRIX/MAP/{normalized}")
    };
    let with_ext = if with_prefix.to_uppercase().ends_with(".CMAP") {
        with_prefix
    } else {
        format!("{with_prefix}.CMAP")
    };
    with_ext.to_uppercase()
}

/// WASM counterpart of `load_map`. The bundle URL is taken from the page's
/// `?bundle=<url>` query parameter so the same build can serve different
/// maps without rewiring JS — this mirrors the original's configurable map
/// path (MatrixGame.cpp:383-394) inside the bundle-delivery model.
/// `assets/atoll.bundle` is kept as the default for backwards compatibility.
#[cfg(target_arch = "wasm32")]
async fn load_map_async() -> (
    GameMap,
    crate::matrix_lib::base::storage::Storage,
    crate::matrix_lib::base::storage::Storage,
    crate::gfx::bundle::AssetBundle,
) {
    use crate::gfx::bundle::AssetBundle;
    use crate::matrix_lib::base::storage::Storage;

    let bundle_url = bundle_url_from_query().unwrap_or_else(|| "assets/atoll.bundle".to_string());
    log::info!("loading bundle: {}", bundle_url);
    let bundle_data = crate::gfx::loader::load_bytes(&bundle_url)
        .await
        .unwrap_or_else(|_| panic!("failed to fetch asset bundle: {}", bundle_url));
    let bundle = AssetBundle::from_bytes(&bundle_data).expect("failed to parse bundle");

    let cmap_data = bundle
        .read_file("map.cmap")
        .expect("no map.cmap in bundle")
        .to_vec();
    let stor = Storage::from_bytes(&cmap_data).expect("failed to parse CStorage");
    let map = GameMap::from_cmap_bytes(&cmap_data).expect("failed to parse CMAP");

    // robots.dat is required here too — see the native loader comment.
    let dat_bytes = bundle.read_file("robots.dat").expect(
        "bundle must contain robots.dat; rebuild the bundle with examples/pack_bundle.rs",
    );
    let matrix_data =
        Storage::from_bytes(dat_bytes).expect("failed to parse bundled robots.dat CStorage");
    log::info!("loaded matrix data from bundle (robots.dat)");

    (map, stor, matrix_data, bundle)
}

/// Parse the current page URL for a `?bundle=<url>` parameter. Returns None
/// when the parameter isn't set or the URL APIs aren't available.
#[cfg(target_arch = "wasm32")]
fn hide_loading_overlay() {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(el) = doc.get_element_by_id("loading") {
            let _ = el.set_attribute("class", "fade");
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn bundle_url_from_query() -> Option<String> {
    let location = web_sys::window()?.location();
    let search = location.search().ok()?;
    if search.is_empty() {
        return None;
    }
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    params.get("bundle").filter(|s| !s.is_empty())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App {
        state: Rc::new(RefCell::new(None)),
    };
    event_loop.run_app(&mut app).expect("event loop error");
}

#[cfg(target_arch = "wasm32")]
pub fn run() {
    use winit::platform::web::EventLoopExtWebSys;
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let app = App {
        state: Rc::new(RefCell::new(None)),
    };
    event_loop.spawn_app(app);
}
