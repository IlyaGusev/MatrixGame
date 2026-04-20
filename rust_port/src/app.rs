use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use crate::effects::point_light::PointLightSystem;
use crate::game::map::GameMap;
use crate::game::world::World;
use crate::renderer::camera::Camera;
use crate::renderer::context::GfxContext;
use crate::renderer::terrain::TerrainRenderer;
struct AppState {
    window: Arc<Window>,
    gfx: GfxContext,
    map: Arc<GameMap>,
    point_lights: PointLightSystem,
    terrain: TerrainRenderer,
    camera: Camera,
    game: World,
    last_time: f64,
    cursor: [f32; 2],
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
                let pkg = pkg.as_ref()?;
                let key = path.to_uppercase();
                for candidate in [&key, &format!("{key}.PNG"), &format!("{key}.DDS")] {
                    if let Ok(data) = pkg.read_file(candidate) {
                        return Some(data);
                    }
                }
                None
            };
            let terrain = TerrainRenderer::new(
                &gfx.device,
                &gfx.queue,
                &gfx.config,
                &map,
                &stor,
                matrix_data.as_ref(),
                &tex_reader,
            );
            let point_lights = PointLightSystem::new(&map);

            *self.state.borrow_mut() = Some(AppState {
                window,
                gfx,
                map,
                point_lights,
                terrain,
                camera,
                game: World::new(),
                last_time: crate::platform::now_secs(),
                cursor: [-1.0, -1.0],
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
                let mut terrain = TerrainRenderer::new(
                    &gfx.device,
                    &gfx.queue,
                    &gfx.config,
                    &map,
                    &stor,
                    matrix_data.as_ref(),
                    &tex_reader,
                );
                let point_lights = PointLightSystem::new(&map);

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
                    camera.set_aspect(size.width as f32, size.height as f32);
                }

                *state_slot.borrow_mut() = Some(AppState {
                    window: win.clone(),
                    gfx,
                    map,
                    point_lights,
                    terrain,
                    camera,
                    game: World::new(),
                    last_time: crate::platform::now_secs(),
                    cursor: [-1.0, -1.0],
                });
                win.request_redraw();
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
                    state
                        .camera
                        .set_aspect(size.width as f32, size.height as f32);
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
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                state.cursor = [position.x as f32, position.y as f32];
                state
                    .camera
                    .on_mouse_move(position.x as f32, position.y as f32);
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
                use crate::renderer::camera::KeyAction;
                use winit::keyboard::{KeyCode, PhysicalKey};
                let pressed = event.state == winit::event::ElementState::Pressed;
                let action =
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                            Some(KeyAction::MoveBack)
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown)
                        | PhysicalKey::Code(KeyCode::KeyS) => Some(KeyAction::MoveForward),
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
                state.game.update(dt);
                state.camera.takt(dt * 1000.0); // camera update (ms)
                state.terrain.takt(
                    dt * 1000.0,
                    &state.map,
                    &state.point_lights,
                    &state.camera,
                    &state.gfx.device,
                    &state.gfx.queue,
                ); // water animation + dynamic object tint updates

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
#[cfg(not(target_arch = "wasm32"))]
fn load_map() -> (
    GameMap,
    crate::assets::storage::Storage,
    Option<crate::assets::storage::Storage>,
    Option<crate::assets::pkg_reader::PkgArchive>,
) {
    use crate::assets::pkg_reader::PkgArchive;
    use crate::assets::storage::Storage;

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

    let files = pkg.list_files();
    let mut cmaps: Vec<_> = files.iter().filter(|f| f.ends_with(".CMAP")).collect();
    cmaps.sort();
    let map_name = cmaps.first().expect("no CMAP files found in pkg");
    log::info!("loading map: {}", map_name);

    let cmap_data = pkg.read_file(map_name).expect("failed to read CMAP");
    let stor = Storage::from_bytes(&cmap_data).expect("failed to parse CStorage");
    let map = GameMap::from_cmap_bytes(&cmap_data).expect("failed to parse CMAP");

    // Load robots.dat alongside the pkg. This is the serialized global
    // BlockPar tree (Sky/Water/Side/...). Optional — missing file just
    // disables the skybox/water config lookups.
    let dat_candidates = [
        "../Data/robots.dat",
        "Data/robots.dat",
        "../../Data/robots.dat",
    ];
    let matrix_data = dat_candidates
        .iter()
        .find(|c| std::path::Path::new(c).exists())
        .and_then(|path| match std::fs::read(path) {
            Ok(bytes) => match Storage::from_bytes(&bytes) {
                Ok(s) => {
                    log::info!("loaded matrix data: {}", path);
                    Some(s)
                }
                Err(e) => {
                    log::warn!("failed to parse {}: {}", path, e);
                    None
                }
            },
            Err(e) => {
                log::warn!("failed to read {}: {}", path, e);
                None
            }
        });

    (map, stor, matrix_data, Some(pkg))
}

#[cfg(target_arch = "wasm32")]
async fn load_map_async() -> (
    GameMap,
    crate::assets::storage::Storage,
    Option<crate::assets::storage::Storage>,
    crate::assets::bundle::AssetBundle,
) {
    use crate::assets::bundle::AssetBundle;
    use crate::assets::storage::Storage;

    let bundle_data = crate::assets::loader::load_bytes("assets/atoll.bundle")
        .await
        .expect("failed to fetch asset bundle");
    let bundle = AssetBundle::from_bytes(&bundle_data).expect("failed to parse bundle");

    let cmap_data = bundle
        .read_file("map.cmap")
        .expect("no map.cmap in bundle")
        .to_vec();
    let stor = Storage::from_bytes(&cmap_data).expect("failed to parse CStorage");
    let map = GameMap::from_cmap_bytes(&cmap_data).expect("failed to parse CMAP");

    let matrix_data =
        bundle
            .read_file("robots.dat")
            .and_then(|bytes| match Storage::from_bytes(bytes) {
                Ok(s) => {
                    log::info!("loaded matrix data from bundle (robots.dat)");
                    Some(s)
                }
                Err(e) => {
                    log::warn!("failed to parse bundled robots.dat: {}", e);
                    None
                }
            });

    (map, stor, matrix_data, bundle)
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
