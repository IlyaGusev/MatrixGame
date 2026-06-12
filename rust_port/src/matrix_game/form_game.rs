use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use crate::gfx::context::GfxContext;
use crate::matrix_game::camera::Camera;
use crate::matrix_game::effects::point_light::PointLightSystem;
use crate::matrix_game::logic::MapLogic;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map::MapRenderer;
use crate::matrix_game::minimap::Minimap;
struct AppState {
    window: Arc<Window>,
    gfx: GfxContext,
    map: Arc<GameMap>,
    point_lights: PointLightSystem,
    terrain: MapRenderer,
    minimap: Minimap,
    camera: Camera,
    game: MapLogic,
    last_time: f64,
    cursor: [f32; 2],
    minimap_dragging: bool,
    /// Shift-key modifier state — tracked so left-click can toggle
    /// the multi-selection (`CMultiSelection::Add/Remove` at
    /// MatrixSide.cpp:1584-1598) vs. replace it.
    shift_down: bool,
    /// Left-button press anchor (screen coords) while the button is
    /// held. `Some` → either a pending click or an in-progress
    /// marquee-drag; `None` → button released. `CMultiSelection::Begin`
    /// in the C++ stores the same state (MatrixMultiSelection.cpp).
    lmb_anchor: Option<[f32; 2]>,
    /// Previous click (time_ms, x, y) for double-click detection —
    /// drives `OnLButtonDouble` (MatrixFormGame.cpp WM_LBUTTONDBLCLK).
    last_click: Option<(i64, f32, f32)>,
    /// Whether the current LMB press was consumed by UI / minimap
    /// (if so, release mustn't issue a world click or marquee).
    lmb_consumed_by_ui: bool,
    /// Last rect the marquee rendered / was releasing. Needed across
    /// frames because the DIP fade keeps drawing for 50ms after the
    /// user lifts LMB and has to resample alpha off the same coords.
    marquee_last_rect: Option<[f32; 4]>,
    /// Selection-ring renderer — ports `CMatrixEffectSelection`
    /// (MatrixEffectSelection.cpp). Drawn over the terrain after the
    /// object pass; green ring on the ground around the selected
    /// object.
    selection_ring: crate::matrix_game::effects::selection::SelectionRingRenderer,
    /// Screen-space marquee rectangle drawn while the user holds
    /// left-button and drags on empty terrain. Rebuilt each frame
    /// from `lmb_anchor` + current cursor; hidden otherwise.
    marquee: crate::matrix_game::multi_selection::MarqueeRenderer,
    /// Move-order ground-ping effect — port of `CMatrixEffectMoveto`
    /// (Effects/MatrixEffectMoveTo.cpp). Spawned at the terrain hit
    /// point when a right-click issues a move order while robots are
    /// selected; lifetime 400ms.
    move_to: crate::matrix_game::effects::move_to::MoveToRenderer,
    /// Loaded UI panels (IF_MAIN / IF_BASE / etc.) + focus state.
    /// Ported from `CIFaceList` (Interface/CInterface.h:269+).
    iface_list: crate::matrix_game::interface::IFaceList,
    /// 2D textured-quad renderer for the UI panels. Drawn last so
    /// the HUD sits on top of world + minimap.
    iface_renderer: crate::matrix_game::interface::InterfaceRenderer,
    /// Horizontal 3-segment progress bars — port of
    /// `CMatrixProgressBar`. Queued each frame from
    /// `refresh_progress_bars`; drawn after the interface pass.
    progress_bars: crate::matrix_game::progress_bar::ProgressBarRenderer,
    /// Robot-constructor 3D preview state — port of
    /// `CConstructor::Render`'s viewport setup (CConstructor.cpp:
    /// 264-360). Emits a chassis draw-ticket per frame while the
    /// constructor panel is active.
    builder_preview: crate::matrix_game::interface::constructor::BuilderPreview,
    /// Cached robot-icon textures for the build-queue UI. One 64×64
    /// per unique `RobotConfig`, lazily baked the first time the queue
    /// surfaces a config. Port of `m_MedTexture` (MatrixRobot.cpp:
    /// 5342-5380); see `interface::robot_icons` for the bake path.
    robot_icons: crate::matrix_game::interface::RobotIconCache,
    /// Fullscreen translucent-black overlay drawn when `is_paused` —
    /// port of MatrixMap.cpp:2430-2454.
    pause_overlay: crate::matrix_game::pause_overlay::PauseOverlay,
    /// Mirrors `g_MatrixMap->IsPaused()` (MatrixMap.hpp:640). Constructor
    /// open / close toggles this; logic takt is gated on it.
    is_paused: bool,
    /// Effect-primitive renderer (billboards / beams / cones / shells)
    /// — the `CBillboard::SortEndDraw` + BBT texture table port.
    effects_renderer: crate::matrix_game::effects::effects_renderer::EffectsRenderer,
    /// Projectile + debris mesh renderer.
    effect_meshes: crate::matrix_game::effects::effects_renderer::EffectMeshRenderer,
    /// Landscape decals (craters / scorch marks).
    spots: crate::matrix_game::effects::landscape_spot::LandscapeSpots,
    /// Per-frame primitive queues (reused allocations).
    bb_queue: crate::matrix_lib::three_g::billboard::BillboardQueue,
    mesh_queue: crate::matrix_game::effects::explosion::MeshQueue,
    /// FPS counter — wall-clock time of the last log emit and number of
    /// frames since. Drained once a second from `RedrawRequested`.
    fps_last_log: f64,
    fps_frames: u32,
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

            let mut game = MapLogic::new();
            game.load_config(&matrix_data);
            let building_ids = game.spawn_buildings(&map);
            log::info!("world: spawned {} buildings", building_ids.len());
            let robot_ids = game.spawn_robots(&map);
            log::info!("world: spawned {} initial robots", robot_ids.len());
            game.ensure_sides_from_objects();
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

            let selection_ring = crate::matrix_game::effects::selection::SelectionRingRenderer::new(
                &gfx.device,
                &gfx.config,
            );
            let marquee =
                crate::matrix_game::multi_selection::MarqueeRenderer::new(&gfx.device, &gfx.config);
            let move_to =
                crate::matrix_game::effects::move_to::MoveToRenderer::new(&gfx.device, &gfx.config);

            let iface_list =
                crate::matrix_game::interface::IFaceList::load_default_panels(&matrix_data);
            log::info!("iface: loaded {} panels", iface_list.panels.len());
            let mut iface_renderer = crate::matrix_game::interface::InterfaceRenderer::new(
                &gfx.device,
                &gfx.queue,
                &gfx.config,
            );
            // Preload every atlas referenced by any element of any
            // loaded panel. `if/Main` alone pulls from interface1/2/3,
            // base_1, base_4, text_1; other panels add base_2/3/5/6.
            let pkg_ref = &pkg;
            let read = |p: &str| -> Option<Vec<u8>> {
                let key = p.replace('\\', "/").to_uppercase();
                for candidate in [&key, &format!("{key}.PNG")] {
                    if let Ok(data) = pkg_ref.read_file(candidate) {
                        return Some(data);
                    }
                }
                None
            };
            iface_renderer.preload_for_panels(
                &gfx.device,
                &gfx.queue,
                &read,
                iface_list.panels.iter(),
            );
            // Port of `CMatrixHint::PreloadBitmaps` (MatrixHint.cpp:
            // 441-459) — load the 9-slice border PNG + every alias in
            // `Hints/Bitmaps` (res_titan / res_energy / face_N / …).
            iface_renderer.preload_hint_chrome(
                &gfx.device,
                &gfx.queue,
                &read,
                &iface_list.hint_chrome,
            );
            let mut progress_bars = crate::matrix_game::progress_bar::ProgressBarRenderer::new(
                &gfx.device,
                &gfx.config,
            );
            progress_bars.load_atlas(&gfx.device, &gfx.queue, &read);

            // Port of `CIFaceList::ConstructorButtonsInit` (MatrixGame.cpp:517).
            // Seeds the constructor with default Pneumatic chassis +
            // ARMOR_6 + Machinegun so pylons show real components on
            // first open instead of "N/A" placeholders.
            if let Some(b) = game.player_side.builder.as_mut() {
                b.constructor_buttons_init();
            }

            let pause_overlay =
                crate::matrix_game::pause_overlay::PauseOverlay::new(&gfx.device, &gfx.config);
            let effects_renderer =
                crate::matrix_game::effects::effects_renderer::EffectsRenderer::new(
                    &gfx.device,
                    &gfx.queue,
                    &gfx.config,
                    wgpu::TextureFormat::Depth32Float,
                    &matrix_data,
                    &tex_reader,
                );
            let effect_meshes =
                crate::matrix_game::effects::effects_renderer::EffectMeshRenderer::new(
                    &gfx.device,
                    &gfx.queue,
                    &gfx.config,
                    wgpu::TextureFormat::Depth32Float,
                    &matrix_data,
                    &tex_reader,
                );
            game.objects.debris_catalog_len = effect_meshes.debris_count();
            game.objects.debris_types = effect_meshes.debris_types().to_vec();
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
                shift_down: false,
                lmb_anchor: None,
                last_click: None,
                lmb_consumed_by_ui: false,
                marquee_last_rect: None,
                selection_ring,
                marquee,
                move_to,
                iface_list,
                iface_renderer,
                progress_bars,
                builder_preview: crate::matrix_game::interface::constructor::BuilderPreview::new(),
                robot_icons: crate::matrix_game::interface::RobotIconCache::new(),
                pause_overlay,
                effects_renderer,
                effect_meshes,
                spots: Default::default(),
                bb_queue: Default::default(),
                mesh_queue: Default::default(),
                is_paused: false,
                fps_last_log: crate::platform::now_secs(),
                fps_frames: 0,
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
                log::debug!(
                    "wasm init: window inner_size = {}x{}",
                    size.width,
                    size.height
                );
                log::debug!(
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
                    camera.set_aspect(gfx.config.width as f32, gfx.config.height as f32);
                }

                let mut game = MapLogic::new();
                game.load_config(&matrix_data);
                let building_ids = game.spawn_buildings(&map);
                log::info!("world: spawned {} buildings", building_ids.len());
                let robot_ids = game.spawn_robots(&map);
                log::info!("world: spawned {} initial robots", robot_ids.len());
                game.ensure_sides_from_objects();
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

                let selection_ring =
                    crate::matrix_game::effects::selection::SelectionRingRenderer::new(
                        &gfx.device,
                        &gfx.config,
                    );
                let marquee = crate::matrix_game::multi_selection::MarqueeRenderer::new(
                    &gfx.device,
                    &gfx.config,
                );
                let move_to = crate::matrix_game::effects::move_to::MoveToRenderer::new(
                    &gfx.device,
                    &gfx.config,
                );

                let iface_list =
                    crate::matrix_game::interface::IFaceList::load_default_panels(&matrix_data);
                log::info!("iface: loaded {} panels", iface_list.panels.len());
                let mut iface_renderer = crate::matrix_game::interface::InterfaceRenderer::new(
                    &gfx.device,
                    &gfx.queue,
                    &gfx.config,
                );
                let read = |p: &str| -> Option<Vec<u8>> { bundle.read_file(p).map(|b| b.to_vec()) };
                iface_renderer.preload_for_panels(
                    &gfx.device,
                    &gfx.queue,
                    &read,
                    iface_list.panels.iter(),
                );
                iface_renderer.preload_hint_chrome(
                    &gfx.device,
                    &gfx.queue,
                    &read,
                    &iface_list.hint_chrome,
                );
                let mut progress_bars = crate::matrix_game::progress_bar::ProgressBarRenderer::new(
                    &gfx.device,
                    &gfx.config,
                );
                progress_bars.load_atlas(&gfx.device, &gfx.queue, &read);

                // Port of `CIFaceList::ConstructorButtonsInit`
                // (MatrixGame.cpp:517).
                if let Some(b) = game.player_side.builder.as_mut() {
                    b.constructor_buttons_init();
                }

                let pause_overlay =
                    crate::matrix_game::pause_overlay::PauseOverlay::new(&gfx.device, &gfx.config);
                let effects_renderer =
                    crate::matrix_game::effects::effects_renderer::EffectsRenderer::new(
                        &gfx.device,
                        &gfx.queue,
                        &gfx.config,
                        wgpu::TextureFormat::Depth32Float,
                        &matrix_data,
                        &read,
                    );
                let effect_meshes =
                    crate::matrix_game::effects::effects_renderer::EffectMeshRenderer::new(
                        &gfx.device,
                        &gfx.queue,
                        &gfx.config,
                        wgpu::TextureFormat::Depth32Float,
                        &matrix_data,
                        &read,
                    );
                game.objects.debris_catalog_len = effect_meshes.debris_count();
            game.objects.debris_types = effect_meshes.debris_types().to_vec();
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
                    shift_down: false,
                    lmb_anchor: None,
                    last_click: None,
                    lmb_consumed_by_ui: false,
                    marquee_last_rect: None,
                    selection_ring,
                    marquee,
                    move_to,
                    iface_list,
                    iface_renderer,
                    progress_bars,
                    builder_preview:
                        crate::matrix_game::interface::constructor::BuilderPreview::new(),
                    robot_icons: crate::matrix_game::interface::RobotIconCache::new(),
                    pause_overlay,
                    effects_renderer,
                    effect_meshes,
                    spots: Default::default(),
                    bb_queue: Default::default(),
                    mesh_queue: Default::default(),
                    is_paused: false,
                    fps_last_log: crate::platform::now_secs(),
                    fps_frames: 0,
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
            // Middle button toggles MouseCam mode (rotate-on-drag,
            // MatrixFormGame.cpp:631-642 — VK_MBUTTON only). Right-click
            // issues move orders to the selected robots
            // (C++: `CMatrixSideUnit::OnRButtonDown`).
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                use winit::event::{ElementState, MouseButton};
                let [cx, cy] = state.cursor;
                let w = state.gfx.config.width as f32;
                let h = state.gfx.config.height as f32;
                if button == MouseButton::Middle {
                    let pressed = btn_state == ElementState::Pressed;
                    state.camera.on_rotate_button(pressed, cx, cy);
                } else if button == MouseButton::Right {
                    // RMB → UI first (CIFaceButton::OnMouseRBDown opens
                    // the constructor popup menu when a pylon catches
                    // the press). Move-orders only run if no UI element
                    // claims the event.
                    let ui_consumed_rmb = match btn_state {
                        ElementState::Pressed => state.iface_list.on_mouse_right_down(cx, cy, w, h),
                        ElementState::Released => state
                            .iface_list
                            .on_mouse_right_up(cx, cy, w, h)
                            .map(|click| {
                                log::debug!("iface: right-clicked {:?}", click);
                                dispatch_ui_right_click(state, &click);
                            })
                            .is_some(),
                    };
                    // World move-orders are gated on the cursor not being
                    // over any interface element or the minimap — port of
                    // the `g_IFaceList->m_InFocus == UNKNOWN` guard at
                    // MatrixFormGame.cpp:756-760.
                    let over_ui = state.iface_list.hit_test(cx, cy, w, h).is_some()
                        || state.minimap.click_to_world(cx, cy).is_some();
                    // Right-click backs out of an armed order without
                    // issuing anything (mirrors ResetOrderingMode).
                    if !ui_consumed_rmb
                        && btn_state == ElementState::Pressed
                        && state.iface_list.pre_order.take().is_some()
                    {
                        log::debug!("order: cancelled by right-click");
                    } else if !ui_consumed_rmb
                        && btn_state == ElementState::Pressed
                        && !state.game.player_side.selected.is_empty()
                        && state.minimap.click_to_world(cx, cy).is_some()
                    {
                        // Right-click on the minimap → move order at the
                        // minimap world position + red ping
                        // (MatrixSide.cpp:821-830).
                        if let Some(tgt) = state.minimap.click_to_world(cx, cy) {
                            state.game.order_move_to_world(
                                &state.map,
                                glam::Vec2::new(tgt[0], tgt[1]),
                            );
                            state
                                .minimap
                                .add_event(tgt[0], tgt[1], 0xffff0000, 0xffff0000);
                        }
                    } else if !ui_consumed_rmb
                        && !over_ui
                        && btn_state == ElementState::Pressed
                        && !state.game.player_side.selected.is_empty()
                    {
                        // Full OnRButtonDown dispatch (MatrixSide.cpp:
                        // 799-863): enemy building → capture, enemy
                        // unit → attack, else move.
                        state
                            .game
                            .on_right_click(&state.camera, cx, cy, w, h, &state.map);
                    }
                } else if button == MouseButton::Left {
                    use crate::matrix_game::minimap::MinimapClick;
                    match btn_state {
                        ElementState::Pressed => {
                            // UI first dibs (MatrixFormGame.cpp:748-755).
                            if state.iface_list.on_mouse_down(cx, cy, w, h) {
                                if let Some(cfg) = state.iface_list.popup_restore_pending.take() {
                                    if let Some(b) = state.game.player_side.builder.as_mut() {
                                        b.apply_config(cfg);
                                    }
                                    reset_build_counter(state);
                                }
                                state.minimap_dragging = false;
                                state.lmb_anchor = None;
                                state.lmb_consumed_by_ui = true;
                            } else {
                                if let Some(cfg) = state.iface_list.popup_restore_pending.take() {
                                    if let Some(b) = state.game.player_side.builder.as_mut() {
                                        b.apply_config(cfg);
                                    }
                                    reset_build_counter(state);
                                }
                                match state.minimap.click(cx, cy) {
                                    MinimapClick::BeginDrag(tgt) => {
                                        state.camera.set_xy_strategy(tgt);
                                        state.minimap_dragging = true;
                                        state.lmb_anchor = None;
                                        state.lmb_consumed_by_ui = true;
                                    }
                                    MinimapClick::ZoomIn | MinimapClick::ZoomOut => {
                                        state.minimap_dragging = false;
                                        state.lmb_anchor = None;
                                        state.lmb_consumed_by_ui = true;
                                    }
                                    MinimapClick::None => {
                                        state.minimap_dragging = false;
                                        // Record the press anchor. The
                                        // click-vs-marquee decision is
                                        // deferred until release, matching
                                        // `CMultiSelection::Begin` +
                                        // `End` semantics (MatrixFormGame.
                                        // cpp:763-770, 664-670).
                                        //
                                        // Pause guard mirrors
                                        // `CMultiSelection::Begin` at
                                        // MatrixMultiSelection.cpp:34
                                        // returning NULL when the map is
                                        // paused — so while the constructor
                                        // is open (Pause(true) inside
                                        // CConstructorPanel::ActivateAndSelect,
                                        // CConstructor.cpp:975) world
                                        // clicks neither click-select
                                        // nor marquee, leaving the
                                        // constructor open until the
                                        // user presses `cocan`.
                                        if !state.is_paused {
                                            state.lmb_anchor = Some([cx, cy]);
                                            state.lmb_consumed_by_ui = false;
                                        } else {
                                            state.lmb_anchor = None;
                                            state.lmb_consumed_by_ui = true;
                                        }
                                    }
                                }
                            }
                        }
                        ElementState::Released => {
                            state.minimap_dragging = false;
                            if let Some(click) = state.iface_list.on_mouse_up(cx, cy, w, h) {
                                log::debug!("iface: clicked {:?}", click);
                                dispatch_ui_click(state, &click);
                            } else if state.iface_list.turret_build.is_active()
                                && state.iface_list.turret_build.kind.is_some()
                            {
                                // Turret placement click — only fires
                                // once a kind has been committed via
                                // tur1..4. Picker-only state (kind
                                // None) leaves world clicks alone so
                                // the user can still interact with
                                // the rest of the UI / world.
                                state.lmb_anchor = None;
                                try_place_turret(state, cx, cy, w, h);
                            } else if state.iface_list.pre_order.is_some()
                                && !state.lmb_consumed_by_ui
                            {
                                // Armed robot order — this click is its
                                // target (the PREORDER_* execution at
                                // MatrixSide.cpp:702-727).
                                state.lmb_anchor = None;
                                execute_pre_order(state, cx, cy, w, h);
                            } else if let Some([ax, ay]) = state.lmb_anchor.take() {
                                if !state.lmb_consumed_by_ui {
                                    // Drag distance — anything ≤ 4 px is a
                                    // click, otherwise a marquee rect.
                                    let dx = (cx - ax).abs();
                                    let dy = (cy - ay).abs();
                                    const DRAG_PX: f32 = 4.0;
                                    if dx <= DRAG_PX && dy <= DRAG_PX {
                                        // Double-click → select all own
                                        // robots in radius (OnLButtonDouble).
                                        let now_ms = state.game.elapsed_ms;
                                        let dbl = state
                                            .last_click
                                            .map(|(t, px, py)| {
                                                now_ms - t < 350
                                                    && (px - cx).abs() <= DRAG_PX
                                                    && (py - cy).abs() <= DRAG_PX
                                            })
                                            .unwrap_or(false);
                                        state.last_click = Some((now_ms, cx, cy));
                                        if dbl
                                            && state.game.on_left_double_click(
                                                &state.camera,
                                                cx,
                                                cy,
                                                w,
                                                h,
                                            )
                                        {
                                            state.lmb_consumed_by_ui = false;
                                            state.lmb_anchor = None;
                                            return;
                                        }
                                        let hit = state.game.click_at_screen(
                                            &state.camera,
                                            cx,
                                            cy,
                                            w,
                                            h,
                                            state.shift_down,
                                        );
                                        match hit {
                                            Some(id) => log::debug!(
                                                "selection: hit object {:?}, curr_sel={:?}, selected={}",
                                                id, state.game.player_side.curr_sel,
                                                state.game.player_side.selected.len(),
                                            ),
                                            None => log::debug!(
                                                "selection: cleared (selected={})",
                                                state.game.player_side.selected.len(),
                                            ),
                                        }
                                        // Mirror into the CMatrixGroup
                                        // machinery the PGOrder layer
                                        // dispatches on.
                                        state.game.sync_group_from_selection();
                                    } else {
                                        let rmin = [ax.min(cx), ay.min(cy)];
                                        let rmax = [ax.max(cx), ay.max(cy)];
                                        let n = crate::matrix_game::multi_selection::marquee_select(
                                            &mut state.game,
                                            &state.camera,
                                            rmin,
                                            rmax,
                                            w,
                                            h,
                                            state.shift_down,
                                        );
                                        log::debug!("marquee: selected {} robot(s)", n,);
                                        state.game.sync_group_from_selection();
                                        // Start the 50ms DIP fade — port
                                        // of `CMultiSelection::End` at
                                        // MatrixMultiSelection.cpp:278-279.
                                        state.marquee.begin_dip_fade(state.game.elapsed_ms as f32);
                                    }
                                }
                            }
                            state.lmb_consumed_by_ui = false;
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
                // Live marquee rect — update while LMB is held and
                // the drag has exceeded the click threshold. Ports the
                // `CMultiSelection::Update` call at MatrixFormGame.cpp:
                // 564 that re-evaluates the rect each mouse move.
                if let Some([ax, ay]) = state.lmb_anchor {
                    if !state.lmb_consumed_by_ui && state.iface_list.pre_order.is_none() {
                        const DRAG_PX: f32 = 4.0;
                        if (cx - ax).abs() > DRAG_PX || (cy - ay).abs() > DRAG_PX {
                            let rect = [ax, ay, cx, cy];
                            state.marquee_last_rect = Some(rect);
                            let w = state.gfx.config.width as f32;
                            let h = state.gfx.config.height as f32;
                            state.marquee.set_rect(&state.gfx.queue, rect, w, h);
                        }
                    }
                }
                // Interface hover-state tracking — port of
                // `CIFaceList::OnMouseMove` (Interface/CInterface.cpp).
                {
                    let w = state.gfx.config.width as f32;
                    let h = state.gfx.config.height as f32;
                    let (unfocused, focused) = state.iface_list.on_mouse_move(cx, cy, w, h);
                    let mut preview_changed = false;
                    if let Some(b) = state.game.player_side.builder.as_mut() {
                        preview_changed = preview_popup_hover(b, state.iface_list.popup.as_mut());
                    }
                    if preview_changed {
                        reset_build_counter(state);
                    }
                    // Route Base-panel focus changes into the
                    // constructor — port of CConstructor.cpp:903-958
                    // (`RemoteFocusElement` / `RemoteUnFocusElement`).
                    //
                    // While the popup is open the C++ skips OnMouseMove
                    // for buttons (CInterface.cpp:979) and additionally
                    // RemoteUnFocusElement guards on `POPUP_MENU_ACTIVE`
                    // (CConstructor.cpp:907-910). Net effect: pylon
                    // hover changes are frozen until the popup closes.
                    let popup_active = state.iface_list.popup.is_some();
                    if !popup_active {
                        if let Some(b) = state.game.player_side.builder.as_mut() {
                            if let Some((panel, elem)) = unfocused {
                                if panel == "Base" {
                                    b.unfocus_element(&elem);
                                }
                            }
                            if let Some((panel, elem)) = focused {
                                if panel == "Base" {
                                    b.focus_element(&elem);
                                }
                            }
                        }
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
                        // Shift — tracked for click-to-toggle in the
                        // multi-selection path. Matches the C++ shift
                        // modifier branch in `CMultiSelection::Add`
                        // (MatrixSide.cpp:1584-1598).
                        PhysicalKey::Code(KeyCode::ShiftLeft)
                        | PhysicalKey::Code(KeyCode::ShiftRight) => {
                            state.shift_down = pressed;
                            None
                        }
                        _ => None,
                    };
                if let Some(a) = action {
                    state.camera.on_key(a, pressed);
                }
            }

            WindowEvent::RedrawRequested => {
                let now = crate::platform::now_secs();
                // Clamp each takt to 100 ms like the original render loop
                // (3g.cpp:425-433, `tt1 = min(100, cur_takt)`) — keeps a
                // tab-switch / debugger pause from fast-forwarding logic.
                let dt = ((now - state.last_time) as f32).min(0.1);
                state.last_time = now;

                state.fps_frames += 1;
                let fps_elapsed = now - state.fps_last_log;
                if fps_elapsed >= 1.0 {
                    log::info!(
                        "fps: {:.1} ({} frames / {:.2}s)",
                        state.fps_frames as f64 / fps_elapsed,
                        state.fps_frames,
                        fps_elapsed,
                    );
                    state.fps_last_log = now;
                    state.fps_frames = 0;
                }
                // Logic takt first (ports `CMatrixMapLogic::Takt`, which
                // runs `ProceedLogic` before the graphic takt starts —
                // MatrixLogic.cpp:2722-2761). Then per-object graphic
                // takt (SortEndGraphicTakt, MatrixMapStatic.cpp:755-765).
                // Camera / minimap / terrain takts mirror
                // `CMatrixMap::Takt`'s remaining subsystem dispatches.
                let step_ms = (dt * 1000.0).round() as i32;
                {
                    // Scope the map pointer for the duration of
                    // the logic + graphic takt so dispatched
                    // `logic_takt` / `takt` methods can call
                    // `current_map()` (ports `g_MatrixMap`).
                    let _scope =
                        crate::matrix_game::map::MapScope::enter(&state.map, state.game.elapsed_ms);
                    // Port of `if (IsPaused()) return;` guarding
                    // `CMatrixMapLogic::Takt` (MatrixLogic.cpp:2607).
                    // Graphic takt still runs so animation cursors keep
                    // ticking — matches the C++ where rendering and
                    // effect timers continue while logic is frozen.
                    if !state.is_paused {
                        state.game.takt(step_ms);
                    }
                    state.game.graphic_takt(step_ms);
                    // Landscape decals: age + build geometry for new
                    // spawns (CMatrixEffectLandscapeSpot list).
                    if !state.is_paused {
                        state.spots.takt(step_ms as f32);
                    }
                    let pending: Vec<_> = state.game.objects.pending_spots.drain(..).collect();
                    for sp in pending {
                        state.spots.spawn(&state.map, &sp);
                    }
                }

                // Reconcile + animate the selection-ring effect with
                // `player_side.active_object`. Ports the C++
                // CMatrixEffectSelection lifecycle (create on Select,
                // destroy on UnSelect, follow the object's geo-center
                // each frame, advance dot animation per takt).
                sync_selection_ring(state, step_ms as f32);

                // Advance the marquee's DIP fade (ports the per-frame
                // `CMultiSelection::Draw` fade block at
                // MatrixMultiSelection.cpp:109-123). When the rect is
                // not in DIP, this is a no-op.
                if state.marquee.is_fading() {
                    let w = state.gfx.config.width as f32;
                    let h = state.gfx.config.height as f32;
                    state.marquee.takt(
                        &state.gfx.queue,
                        state.game.elapsed_ms as f32,
                        state.marquee_last_rect,
                        w,
                        h,
                    );
                }

                // Drain the move-order pings PGShowPlace queued — the
                // CreateMoveto / DeleteAllMoveto pair (MatrixSide.cpp:
                // 8542-8562).
                if state.game.moveto_clear_pending {
                    state.game.moveto_clear_pending = false;
                    state.move_to.clear();
                }
                for p in std::mem::take(&mut state.game.moveto_pings) {
                    state.move_to.spawn(p);
                }

                // Per-frame move-order ping animation advance — port
                // of `CMatrixEffectMoveto::Takt` (MatrixEffectMoveTo.cpp:93).
                if state.move_to.is_active() {
                    state.move_to.takt(step_ms as f32);
                }

                // Robot mesh rendering is in place via
                // `state.terrain.sync_robots`, so the per-robot point
                // lights that used to stand in for visibility are no
                // longer needed. Keeping that helper alive cost ~40 FPS
                // because every moving robot bumped the point-light
                // revision, which forced `terrain.takt` to rewrite +
                // re-upload the entire terrain vertex-colour buffer
                // each frame. Left out unless we add real point lights
                // (e.g. weapon flashes) that justify the GPU sync cost.

                // Per-frame interface visibility dispatch — ports the
                // `CInterface::LogicTakt` branch at
                // CInterface.cpp:1214-1635. Only `if/Main` for now.
                refresh_interface_visibility(state);

                // Advance the constructor 3D preview turntable + emit
                // a preview draw-ticket while the constructor panel is
                // open. Stand-in for CConstructor.cpp:251-262 +
                // :264-360 (Render).
                tick_builder_preview(state, dt * 1000.0);

                // Drain a pending popup-close focus clear — when the
                // popup was dismissed by clicking outside it, the
                // focused pylon's right-side preview + price label
                // need to drop along with the popup. The C++ achieves
                // this naturally via the next OnMouseMove pass under
                // !POPUP_MENU_ACTIVE; we clear explicitly here so
                // there's no one-frame stale state.
                if state.iface_list.popup_focus_clear_pending {
                    state.iface_list.popup_focus_clear_pending = false;
                    if let Some(b) = state.game.player_side.builder.as_mut() {
                        if let Some(name) = b.focused_element.clone() {
                            b.unfocus_element(&name);
                        }
                    }
                }

                // Tooltip timer + dynamic hint text refresh. Port of
                // `CIFaceList::OnMouseMove` hint build pass at
                // CIFaceButton.cpp:134-145 combined with
                // `AddHintReplacements` (CInterface.cpp:4439-4540).
                refresh_hint_replacements(state);
                {
                    let w = state.gfx.config.width as f32;
                    let h = state.gfx.config.height as f32;
                    // Snapshot each loaded hint-bitmap's pixel
                    // dimensions up front so the hint layout engine
                    // can size inline resource icons without re-
                    // borrowing `iface_renderer` (which we need as
                    // &mut for the glyph atlas). The snapshot is
                    // cheap — there are ~30 alias entries.
                    let mut sizes: std::collections::HashMap<String, (i32, i32)> =
                        std::collections::HashMap::new();
                    for bmp in state.iface_list.hint_chrome.bitmaps.values() {
                        if let Some((w, h)) = state.iface_renderer.atlas_size(&bmp.path) {
                            sizes.insert(bmp.path.clone(), (w as i32, h as i32));
                        }
                    }
                    let sizer = |path: &str| sizes.get(path).copied();
                    state.iface_list.update(
                        dt * 1000.0,
                        w,
                        h,
                        state.iface_renderer.glyph_atlas_mut(),
                        &sizer,
                    );
                }

                state.camera.takt(dt * 1000.0); // camera update (ms)

                // Progress bars — panel clones (build stack / building
                // HP) + the floating HP bars over objects. Runs AFTER
                // the camera takt so the world→screen projection uses
                // the same camera state the frame renders with.
                refresh_progress_bars(state);

                state.minimap.takt(dt * 1000.0);
                state.terrain.takt(
                    dt * 1000.0,
                    &state.map,
                    &state.point_lights,
                    &state.camera,
                    &state.gfx.device,
                    &state.gfx.queue,
                ); // water animation + dynamic object tint updates

                // Push BASE per-sub-unit animation (platform rise,
                // door slide) from the live `Building::base_floor_progress`
                // into the BuildingsRenderer instance buffers. Ports
                // MatrixObjectBuilding.cpp:836-852. Must run AFTER
                // `state.game.takt` advances the base's state machine.
                state.terrain.sync_building_animation(
                    &state.gfx.queue,
                    &state.game.objects,
                    &state.map,
                    &state.point_lights,
                );

                // Per-frame turret-placement preview update — port of
                // `CMatrixSideUnit::LogicTakt`'s BUILDING_TURRET branch
                // (MatrixSide.cpp:528-585). Snaps the cursor to the
                // nearest free turret slot on the parent base, smooth-
                // rotates the ghost cannon to the slot's rest angle,
                // and flips the validity tint. Also collects free-slot
                // markers (port of `CreatePlacesShow`,
                // MatrixObjectBuilding.cpp:1617) — visible whenever the
                // picker is open so the player can see where to click.
                let (ghost, markers) = update_turret_build(state, step_ms);

                // Rebuild chassis instance buffers from the live arena
                // so newly-spawned robots show up (stand-in for
                // `CMatrixRobotAI::RNeed`'s per-robot matrix update —
                // MatrixObjectRobot.cpp:359-480).
                state.terrain.sync_robots(
                    &state.gfx.device,
                    &state.gfx.queue,
                    &mut state.game.objects,
                    &state.map,
                    &state.point_lights,
                    step_ms,
                    ghost,
                    &markers,
                );

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
                            // Selection-ring pass — share the terrain's
                            // depth buffer so the dots are occluded by
                            // buildings / objects that sit in front of
                            // them. Upload first with the current view
                            // basis so the billboards face the camera.
                            let cr = state.camera.camera_right_world();
                            let cu = state.camera.camera_up_world();
                            let mc = glam::Vec2::new(
                                state.map.world_width() * 0.5,
                                state.map.world_height() * 0.5,
                            );
                            state
                                .selection_ring
                                .upload(&state.gfx.queue, vp, cr, cu, mc);
                            state
                                .move_to
                                .upload(&state.gfx.queue, &state.map, vp, cr, cu, mc);
                            // ── Effect primitives (DrawEffects port) ──
                            state.bb_queue.clear();
                            state.mesh_queue.draws.clear();
                            for e in &state.game.effects {
                                e.draw(&mut state.bb_queue, &mut state.mesh_queue);
                            }
                            {
                                let objs = &state.game.objects;
                                let rng = &mut state.game.rng;
                                for w in objs.weapons.iter() {
                                    w.draw(&mut state.bb_queue, objs, rng, state.is_paused);
                                }
                                // DIP wreck smoke (the pieces are the
                                // robot's own part meshes, drawn by the
                                // robots renderer).
                                for id in objs.iter_live() {
                                    let Some(o) = objs.get(id) else { continue };
                                    match o.core().obj_type {
                                        crate::matrix_game::map_static::ObjectType::RobotAi => {
                                            let r: &crate::matrix_game::robot::Robot = unsafe {
                                                &*(o as *const dyn crate::matrix_game::map_static::MapStatic
                                                    as *const crate::matrix_game::robot::Robot)
                                            };
                                            if r.state == crate::matrix_game::robot::RobotState::Dip {
                                                r.draw_dip(&mut state.bb_queue);
                                            }
                                        }
                                        crate::matrix_game::map_static::ObjectType::Cannon => {
                                            let c: &crate::matrix_game::object_cannon::Cannon = unsafe {
                                                &*(o as *const dyn crate::matrix_game::map_static::MapStatic
                                                    as *const crate::matrix_game::object_cannon::Cannon)
                                            };
                                            if c.state == crate::matrix_game::object_cannon::CannonState::Dip {
                                                for u in &c.dip_units {
                                                    u.smoke.draw(&mut state.bb_queue);
                                                }
                                            }
                                        }
                                        crate::matrix_game::map_static::ObjectType::Building => {
                                            let b: &crate::matrix_game::object_building::Building = unsafe {
                                                &*(o as *const dyn crate::matrix_game::map_static::MapStatic
                                                    as *const crate::matrix_game::object_building::Building)
                                            };
                                            // Capture-progress ring
                                            // (CMatrixEffectZahvat::Draw).
                                            if let Some(z) = b.zahvat.as_ref() {
                                                z.draw(&mut state.bb_queue);
                                            }
                                        }
                                        crate::matrix_game::map_static::ObjectType::Flyer => {
                                            let f: &crate::matrix_game::flyer::Flyer = unsafe {
                                                &*(o as *const dyn crate::matrix_game::map_static::MapStatic
                                                    as *const crate::matrix_game::flyer::Flyer)
                                            };
                                            // Tractor-beam spirals
                                            // (CMatrixEffectElevatorField::Draw).
                                            if let Some(e) = f.carry.elevator.as_ref() {
                                                e.draw(&mut state.bb_queue);
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            let inv_view = vm.inverse();
                            let cam_pos = glam::Vec3::new(
                                inv_view.w_axis.x + mc.x,
                                inv_view.w_axis.y + mc.y,
                                inv_view.w_axis.z,
                            );
                            state.effects_renderer.upload(
                                &state.gfx.device,
                                &state.gfx.queue,
                                &state.bb_queue,
                                vp,
                                vm,
                                cam_pos,
                                cr,
                                cu,
                                mc,
                            );
                            state.effects_renderer.upload_spots(
                                &state.gfx.device,
                                &state.gfx.queue,
                                &state.spots,
                                mc,
                            );
                            state
                                .effect_meshes
                                .upload(&state.gfx.queue, &state.mesh_queue, vp, mc);
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Selection Ring Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: state.terrain.depth_view(),
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Load,
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                            // Landscape decals first (the C++ draws
                            // them right after the terrain surfaces).
                            state.effects_renderer.render_spots(&mut pass);
                            // Projectile / debris meshes (depth-write).
                            state.effect_meshes.render(&mut pass);
                            state.selection_ring.render(&mut pass);
                            // Move-order ping — same color/depth target
                            // as the selection ring (billboards are
                            // additively blended + depth-tested against
                            // terrain so they occlude behind geometry).
                            state.move_to.render(&mut pass);
                            // Effect billboards (alpha bucket sorted,
                            // additive after) — `DrawEffects`.
                            state.effects_renderer.render(&mut pass);
                            // Marquee shares the overlay color target +
                            // depth attachment with the selection ring;
                            // render inline to avoid a separate pass.
                            state.marquee.render(&mut pass);
                        }
                        {
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Minimap Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
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
                                &state.game.objects,
                            );
                        }
                        // Pause overlay — fullscreen translucent black
                        // tint when the game is paused. Sits between
                        // world+minimap and the UI so the HUD stays
                        // bright while the playfield dims. Port of
                        // MatrixMap.cpp:2430-2454.
                        let pause_visible = state.is_paused
                            || state
                                .game
                                .player_side
                                .builder
                                .as_ref()
                                .map(|b| b.active)
                                .unwrap_or(false);
                        if pause_visible {
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Pause Overlay Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                            state.pause_overlay.draw(&mut pass);
                        }
                        // Interface (HUD) pass — draws on top of
                        // world + minimap. Ports
                        // `CIFaceList::Render` iteration.
                        {
                            // Pause hint — port of MatrixLogic.cpp:2607-2614:
                            //   m_PauseHint = CMatrixHint::Build(TEMPLATE_PAUSE);
                            //   m_PauseHint->Show(14, 62);
                            // Built each frame while paused (cheap layout).
                            // Falls through to the regular hover hint when
                            // unpaused or when the "Pause" template is
                            // missing from `da/Templates`. Build BEFORE
                            // borrowing `iface_list.panels` because the
                            // builder needs `&mut iface_renderer`.
                            let pause_hint_local: Option<crate::matrix_game::interface::Hint> =
                                if pause_visible {
                                    build_pause_hint(state)
                                } else {
                                    None
                                };
                            let panels: Vec<&crate::matrix_game::interface::CInterface> =
                                state.iface_list.panels.iter().collect();
                            let hint_to_show = pause_hint_local
                                .as_ref()
                                .or_else(|| state.iface_list.hint_system.active());
                            state.iface_renderer.upload_with_popup_and_hint(
                                &state.gfx.device,
                                &state.gfx.queue,
                                &panels,
                                state.iface_list.popup.as_ref(),
                                hint_to_show,
                                Some(&state.iface_list.hint_chrome),
                                state.gfx.config.width as f32,
                                state.gfx.config.height as f32,
                            );
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Interface Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                            state.iface_renderer.render(&mut pass);
                        }
                        // Constructor 3D preview — one chassis drawn
                        // into the sub-viewport on the Base panel.
                        // Ports CConstructor::Render's viewport setup
                        // (CConstructor.cpp:264-360). Runs after the
                        // UI pass so the panel backdrop is in place,
                        // but before the progress-bar pass so the bar
                        // can overlay the preview if they happen to
                        // collide.
                        let q_opt = builder_preview_query(state);
                        let robots_ok = state.terrain.robots().is_some();
                        {
                            static LOGGED: std::sync::atomic::AtomicBool =
                                std::sync::atomic::AtomicBool::new(false);
                            static LAST_STATE: std::sync::atomic::AtomicU8 =
                                std::sync::atomic::AtomicU8::new(0xFF);
                            let s = (q_opt.is_some() as u8) | ((robots_ok as u8) << 1);
                            if LAST_STATE.swap(s, std::sync::atomic::Ordering::Relaxed) != s
                                || !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
                            {
                                log::debug!(
                                    "preview: query={} robots_renderer={}",
                                    if q_opt.is_some() { "Some" } else { "None" },
                                    if robots_ok { "Some" } else { "None" },
                                );
                            }
                        }
                        if let Some(q) = q_opt {
                            let chassis = q.chassis;
                            let angle_rad = q.angle_rad;
                            let design_rect = q.design_rect;
                            let armor_kind = q.armor_kind;
                            let head_kind = q.head_kind;
                            let weapon_kinds = q.weapon_kinds;
                            if let Some(robots) = state.terrain.robots() {
                                // Design-space → pixel rect: the Base
                                // panel resolves its top-left via
                                // CInterface::resolved_pos; design
                                // coords are Y-down from that origin.
                                let surface_w = state.gfx.config.width;
                                let surface_h = state.gfx.config.height;
                                let scale = (surface_h as f32
                                    / crate::matrix_game::interface::interface::DESIGN_H)
                                    .max(0.1);
                                let panel = state
                                    .iface_list
                                    .panel("Base")
                                    .map(|p| {
                                        p.resolved_pos(surface_w as f32, surface_h as f32, scale)
                                    })
                                    .unwrap_or([0.0, 0.0]);
                                let sx = (panel[0] + design_rect[0] * scale).max(0.0) as u32;
                                let sy = (panel[1] + design_rect[1] * scale).max(0.0) as u32;
                                let sw = (design_rect[2] * scale) as u32;
                                let sh = (design_rect[3] * scale) as u32;

                                let depth_view = state.terrain.depth_view();
                                let mut pass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("Constructor Preview Pass"),
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
                                        depth_stencil_attachment: Some(
                                            wgpu::RenderPassDepthStencilAttachment {
                                                view: depth_view,
                                                depth_ops: Some(wgpu::Operations {
                                                    // Clear just the preview
                                                    // region via scissor — but
                                                    // we can't scissor clears,
                                                    // so load the existing
                                                    // depth. The robot draw
                                                    // will depth-test against
                                                    // whatever was there;
                                                    // since the preview region
                                                    // is over UI (sky depth),
                                                    // the chassis effectively
                                                    // writes freely.
                                                    load: wgpu::LoadOp::Clear(1.0),
                                                    store: wgpu::StoreOp::Store,
                                                }),
                                                stencil_ops: None,
                                            },
                                        ),
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                        multiview_mask: None,
                                    });
                                robots.render_preview_full(
                                    &state.gfx.queue,
                                    &mut pass,
                                    chassis,
                                    armor_kind,
                                    head_kind,
                                    &weapon_kinds,
                                    angle_rad,
                                    [sx, sy, sw, sh],
                                    surface_w,
                                    surface_h,
                                    None,
                                );
                            }
                        }
                        // Progress-bar overlay pass — on top of the UI.
                        {
                            state.progress_bars.upload(
                                &state.gfx.device,
                                &state.gfx.queue,
                                state.gfx.config.width as f32,
                                state.gfx.config.height as f32,
                                Some(&state.camera),
                            );
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("ProgressBar Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            });
                            state.progress_bars.render(&mut pass);
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

// Load map — native reads from pkg, returns map + map storage + global
// matrix data storage (robots.dat) + pkg for texture loading.
//
// Matches the original startup contract (MatrixGame.cpp:240-257, 383-399):
// the caller can request a specific map via the first CLI argument; if
// absent, we fall back to `Config/Map` in the global data. Global data
// itself is a required dependency (MatrixGame.cpp:240-257) — the engine
// dereferences `g_MatrixData->BlockGet(...)` without NULL guards all over
// Init, so missing `robots.dat` is a fatal startup error. See `load_map`
// (native) and `load_map_async` (wasm) further below.

/// Port of `CMatrixSideUnit::PlayerAction` + the follow-on
/// `CBuildStack::AddItem` call. Dispatches the button identified by
/// its `Name` to the right game-state change. Currently handles
/// `buro` (build robot) → push a default-chassis robot onto the
/// selected base's build stack. The C++ opens the full
/// `m_ConstructPanel` for chassis/armor/weapon selection first; the
/// constructor UI isn't ported, so we skip straight to AddItem with
/// a default chassis.
fn dispatch_ui_click(state: &mut AppState, click: &crate::matrix_game::interface::Click) {
    use crate::matrix_game::config::RobotUnitKind;
    use crate::matrix_game::interface::constructor::parse_constructor_button;
    use crate::matrix_game::interface::iface_list::TurretKind;
    use crate::matrix_game::interface::Click;
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::{Building, BuildingType};
    use crate::matrix_game::object_robot::RobotUnitType;
    use crate::matrix_game::robot::ChassisKind;
    use crate::matrix_game::side::CurrSel;

    let name = match click {
        Click::Button(n) => n.as_str(),
        // Right-button clicks are dispatched separately
        // (see `dispatch_ui_right_click`). Returning early here keeps
        // the LMB handler's match arms uncluttered.
        Click::RightButton(_) => return,
        // Popup-menu item commit. Port of `CIFaceMenu::OnMenuItemPress`
        // (CIFaceMenu.cpp:530+) — calls SuperDjeans with the chosen
        // (type, kind, pilon) then closes the popup.
        Click::PopupItem { parent, kind } => {
            use crate::matrix_game::interface::iface_menu::EMenuParent;
            let ty = parent.unit_type();
            let pilon = parent.pilon();
            // Pylon-empty kind (RUK_UNKNOWN) is a valid selection for
            // `heade` / `weape` ("clear this slot"); pass through.
            if let Some(b) = state.game.player_side.builder.as_mut() {
                b.super_djeans(ty, *kind, pilon, false);
                log::debug!(
                    "popup commit: type={:?} kind={} pilon={} cfg.chassis={} hull={} weap0={}",
                    ty,
                    kind.0,
                    pilon,
                    b.cfg().chassis.kind.0,
                    b.cfg().hull.unit.kind.0,
                    b.cfg().weapon[0].kind.0,
                );
            }
            reset_build_counter(state);
            state.iface_list.popup = None;
            state.iface_list.popup_restore_pending = None;
            state.iface_list.popup_focus_clear_pending = true;
            let _ = EMenuParent::PylonChassis; // keep import live
            return;
        }
    };

    // ── Static `ON_PRESS` handlers (CInterface.cpp:577-586) ───────
    // In the C++, `basepl` / `titpl` / `elecpl` / `energpl` / `plaspl`
    // are CIFaceStatic elements bound to `CIFaceList::JumpToBuilding`
    // (CInterface.cpp:4552-4559). The callback centres the strategy
    // camera on the currently-selected building's geo center.
    if name == "basepl" {
        if matches!(
            state.game.player_side.curr_sel,
            CurrSel::BuildingSelected | CurrSel::BaseSelected
        ) {
            if let Some(id) = state.game.active_object() {
                if let Some(obj) = state.game.objects.get(id) {
                    let p = obj.core().geo_center;
                    state.camera.set_xy_strategy([p.x, p.y]);
                    log::debug!(
                        "basepl: center camera on building at ({:.1},{:.1})",
                        p.x,
                        p.y
                    );
                }
            }
        }
        return;
    }

    // ── Top-level menu buttons ────────────────────────────────────
    match name {
        "buro" => {
            // Port of MatrixFormGame.cpp:1385-1389 + CConstructor.cpp:
            // 970-975 — reset the build-multiplier counter, validate it
            // against the side's resources, then open the constructor.
            if state.game.player_side.curr_sel != CurrSel::BaseSelected {
                log::debug!("buro: no base selected, ignoring");
                return;
            }
            state.iface_list.r_count_control.reset();
            let ctx = build_counter_ctx(state);
            state.iface_list.r_count_control.check_up(ctx);
            if let Some(b) = state.game.player_side.builder.as_mut() {
                b.activate();
            }
            // Pause the game while the constructor is open — port of
            // `g_MatrixMap->Pause(true)` at CConstructor.cpp:975.
            state.is_paused = true;
            log::debug!("buro: opened robot constructor");
            // Silence "unused import" on paths we only need in other
            // arms of this match.
            let _ = (BuildingType::Base, ChassisKind::Track);
            let _: Option<&dyn MapStatic> = None;
            let _ = ObjectType::Building;
            let _: fn(&mut Building) = |_b: &mut Building| {};
            return;
        }
        "buca" => {
            // Port of CInterface.cpp:3493-3499 — IF_BUILD_CA opens the
            // turret-kind picker on the Main panel. The C++ flips
            // ORDERING_MODE + PREORDER_BUILD_TURRET; the actual kind is
            // not picked yet — that comes when the user clicks one of
            // the `tur1..tur4` buttons. Visibility for any selected
            // building (base OR factory) since `buca` is shown for
            // BUILDING_SELECTED / BASE_SELECTED at CInterface.cpp:1600.
            if !matches!(
                state.game.player_side.curr_sel,
                CurrSel::BaseSelected | CurrSel::BuildingSelected
            ) {
                log::debug!("buca: no building selected, ignoring");
                return;
            }
            let Some(id) = state.game.active_object() else {
                return;
            };
            // Open the kind picker only — the placement-preview ghost
            // is held off until the player commits a kind via tur1..4.
            // Mirrors the C++ at CInterface.cpp:3493-3499 where the
            // PREORDER_BUILD_TURRET flag flips before BeginBuildTurret
            // (and thus before `m_CannonForBuild.m_Cannon` is created).
            state.iface_list.turret_build.open_picker(id);
            log::debug!("buca: entered turret-build mode (kind picker)");
            return;
        }
        "cocan" => {
            // Port of CConstructorPanel::ResetGroupNClose
            // (CConstructor.cpp:986-994). Despite the name, this only
            // clears `m_Active` + closes the popup — it does NOT reset
            // the live construction state. The next `buro` reopens with
            // the same chassis/armor/head/weapons the user had.
            if let Some(b) = state.game.player_side.builder.as_mut() {
                b.deactivate();
            }
            // Resume the game — port of `g_MatrixMap->Pause(false)` at
            // CConstructor.cpp:992.
            state.is_paused = false;
            state.iface_list.r_count_control.reset();
            state.iface_list.popup = None;
            state.iface_list.popup_restore_pending = None;
            state.iface_list.turret_build.cancel();
            log::debug!("cocan: constructor closed");
            return;
        }
        "cobuild" => {
            commit_and_queue_robot(state);
            return;
        }
        "hisleft" => {
            if let Some(cfg) = state.iface_list.history.prev() {
                if let Some(b) = state.game.player_side.builder.as_mut() {
                    b.apply_config(cfg);
                    log::debug!(
                        "hisleft: loaded preset cursor={}",
                        state.iface_list.history.cursor
                    );
                }
                reset_build_counter(state);
            }
            return;
        }
        "hisright" => {
            if let Some(cfg) = state.iface_list.history.next() {
                if let Some(b) = state.game.player_side.builder.as_mut() {
                    b.apply_config(cfg);
                    log::debug!(
                        "hisright: loaded preset cursor={}",
                        state.iface_list.history.cursor
                    );
                }
                reset_build_counter(state);
            }
            return;
        }
        "bup" => {
            let ctx = build_counter_ctx(state);
            state.iface_list.r_count_control.up(ctx);
            return;
        }
        "bdown" => {
            let ctx = build_counter_ctx(state);
            state.iface_list.r_count_control.down(ctx);
            return;
        }
        "tur1" | "tur2" | "tur3" | "tur4" => {
            // Turret kind picker — names match the `tur{N}` buttons on
            // the Main panel (StringConstants.hpp IF_MAIN_TURRET*).
            let n: i32 = name.trim_start_matches("tur").parse().unwrap_or(1);
            let Some(kind) = TurretKind::from_i32(n) else {
                return;
            };
            let parent = state.iface_list.turret_build.parent.or_else(|| {
                if state.game.player_side.curr_sel == CurrSel::BaseSelected {
                    state.game.active_object()
                } else {
                    None
                }
            });
            if let Some(parent) = parent {
                state.iface_list.turret_build.begin(kind, parent);
                log::debug!("tur{}: selected kind={:?}", n, kind);
            }
            return;
        }
        "ocan" => {
            // Port of CInterface.cpp:3465-3470 — IF_ORDER_CANCEL clears
            // any in-flight turret placement (`m_CannonForBuild.Delete`
            // + `m_CurrentAction = NOTHING_SPECIAL`) and resets the
            // ordering mode, robot pre-orders included.
            state.iface_list.turret_build.cancel();
            state.iface_list.pre_order = None;
            log::debug!("ocan: cancelled ordering mode");
            return;
        }
        // Robot order buttons (IF_ORDER_*, CInterface.cpp:3447-3479).
        // `ofi` / `ogo` arm a pre-order executed by the next map click;
        // `ost` acts immediately.
        "ofi" => {
            state.iface_list.pre_order =
                Some(crate::matrix_game::interface::iface_list::PreOrder::Fire);
            log::debug!("order: attack armed — click a target or the ground");
            return;
        }
        "ogo" => {
            state.iface_list.pre_order =
                Some(crate::matrix_game::interface::iface_list::PreOrder::Move);
            log::debug!("order: move armed — click a destination");
            return;
        }
        "ost" => {
            state.iface_list.pre_order = None;
            state.game.order_stop(&state.map);
            log::debug!("order: stop issued");
            return;
        }
        "oca" => {
            state.iface_list.pre_order =
                Some(crate::matrix_game::interface::iface_list::PreOrder::Capture);
            log::debug!("order: capture armed — click an enemy building");
            return;
        }
        "opa" => {
            state.iface_list.pre_order =
                Some(crate::matrix_game::interface::iface_list::PreOrder::Patrol);
            log::debug!("order: patrol armed — click a destination");
            return;
        }
        "obomb" => {
            state.iface_list.pre_order =
                Some(crate::matrix_game::interface::iface_list::PreOrder::Bomb);
            log::debug!("order: bomb armed — click a target");
            return;
        }
        "orep" => {
            state.iface_list.pre_order =
                Some(crate::matrix_game::interface::iface_list::PreOrder::Repair);
            log::debug!("order: repair armed — click a damaged friendly");
            return;
        }
        // Auto-order toggles act immediately (CInterface.cpp:3480-3499).
        "oacapn" | "oacapf" => {
            let no = state.game.sel_group_to_logic_group();
            state.game.pg_order_auto_capture(&state.map, no);
            return;
        }
        "oafrn" | "oafrf" | "oafron" | "oafrof" => {
            let no = state.game.sel_group_to_logic_group();
            state.game.pg_order_auto_attack(&state.map, no);
            return;
        }
        "oafcn" | "oafcf" => {
            let no = state.game.sel_group_to_logic_group();
            state.game.pg_order_auto_defence(&state.map, no);
            return;
        }
        // IF_BUILD_REPAIR — the base "call maintenance" button
        // (CInterface.cpp:3501-3508).
        "bure" => {
            if let Some(b) = state.game.active_object() {
                state.game.building_maintenance(&state.map, b);
                log::debug!("maintenance: requested");
            }
            return;
        }
        _ => {}
    }

    // ── Constructor pylon buttons (LMB) ──────────────────────────
    // Port of the pylon ON_UN_PRESS wiring at CInterface.cpp:262 —
    // left-click fires CConstructor::RemoteOperateUnit, which cycles
    // the mounted component kind with wrap-around.
    if matches!(
        name,
        "pich" | "pihu" | "pihe" | "pi1" | "pi2" | "pi3" | "pi4" | "pi5"
    ) {
        if state.game.player_side.curr_sel != CurrSel::BaseSelected {
            return;
        }
        let active = state
            .game
            .player_side
            .builder
            .as_ref()
            .map(|b| b.active)
            .unwrap_or(false);
        if active {
            if let Some(b) = state.game.player_side.builder.as_mut() {
                b.remote_operate_unit(name);
            }
            reset_build_counter(state);
        }
        return;
    }

    // ── Direct component buttons (from the popup overlay) ─────────
    // chas1..5 / hull1..6 / head1..7 / weap1..10 — in the C++ these
    // are only clickable when CInterface opens a popup with the items
    // laid out at computed positions. The popup mechanic isn't ported
    // yet; as long as the data positions them stacked at the default
    // template slot the direct-click path can't reliably pick a
    // specific kind. We still wire the dispatch so that *if* a popup
    // mechanic lands later, the handler is already in place.
    if let Some((ty, kind, pilon)) = parse_constructor_button(name) {
        if state.game.player_side.curr_sel != CurrSel::BaseSelected {
            return;
        }
        if let Some(b) = state.game.player_side.builder.as_mut() {
            b.super_djeans(ty, kind, pilon, false);
            let p = b.construction_price();
            log::debug!(
                "constructor: {} {:?}/{} → titan={} elec={} energy={} plasma={} structure={}",
                name,
                ty,
                kind.0,
                p.titan(),
                p.electronics(),
                p.energy(),
                p.plasma(),
                b.construction_structure(),
            );
        }
        reset_build_counter(state);
        return;
    }

    log::debug!("dispatch_ui_click: unhandled click {:?}", click);
    let _ = RobotUnitType::Chassis;
    let _ = RobotUnitKind::UNKNOWN;
}

/// Dispatch a UI right-click. Currently routes pylon right-clicks to
/// the constructor popup menu (port of `CIFaceButton::OnMouseRBDown`
/// at CIFaceButton.cpp:183-321).
/// Execute the armed robot order at a world click — port of the
/// `PREORDER_FIRE` / `PREORDER_MOVE` branches of the side's
/// OnLButtonDown (MatrixSide.cpp:684-727). The attack order hits the
/// object under the cursor when there is one (any side — friendly
/// fire is the player's prerogative, like `PGOrderAttack(.., pObject)`),
/// otherwise the bare terrain point.
fn execute_pre_order(state: &mut AppState, cx: f32, cy: f32, w: f32, h: f32) {
    use crate::matrix_game::interface::iface_list::PreOrder;

    let Some(po) = state.iface_list.pre_order.take() else {
        return;
    };
    match po {
        PreOrder::Move => {
            state
                .game
                .order_move_to_at(&state.camera, cx, cy, w, h, &state.map);
        }
        PreOrder::Fire => {
            let hit = state
                .game
                .order_fire_at_screen(&state.camera, cx, cy, w, h, &state.map);
            log::debug!("order: attack dispatched (target: {:?})", hit);
        }
        PreOrder::Capture => {
            if !state
                .game
                .order_capture_at_screen(&state.camera, cx, cy, w, h, &state.map)
            {
                // Invalid target keeps the order armed (MatrixSide.cpp:715).
                state.iface_list.pre_order = Some(PreOrder::Capture);
            }
        }
        PreOrder::Patrol => {
            state
                .game
                .order_patrol_at_screen(&state.camera, cx, cy, w, h, &state.map);
        }
        PreOrder::Bomb => {
            state
                .game
                .order_bomb_at_screen(&state.camera, cx, cy, w, h, &state.map);
        }
        PreOrder::Repair => {
            if !state
                .game
                .order_repair_at_screen(&state.camera, cx, cy, w, h, &state.map)
            {
                state.iface_list.pre_order = Some(PreOrder::Repair);
            }
        }
    }
}

fn dispatch_ui_right_click(state: &mut AppState, click: &crate::matrix_game::interface::Click) {
    use crate::matrix_game::interface::iface_menu::popup_for_pylon;
    use crate::matrix_game::interface::Click;
    use crate::matrix_game::side::CurrSel;

    let name = match click {
        Click::RightButton(n) => n.as_str(),
        _ => return,
    };

    // The popup only opens while the constructor is active and the
    // selection is on a base. Mirrors the early-return guards at
    // CIFaceButton.cpp:188-189.
    let active = state
        .game
        .player_side
        .builder
        .as_ref()
        .map(|b| b.active)
        .unwrap_or(false);
    if !active || state.game.player_side.curr_sel != CurrSel::BaseSelected {
        return;
    }

    let strings = crate::matrix_game::config::global_strings();
    let base_panel = state.iface_list.panel("Base");
    if let Some(mut popup) = popup_for_pylon(name, base_panel, &strings.popup_none) {
        // CIFaceMenu::CreateMenu (CIFaceMenu.cpp:62-100) — record the
        // caller pylon, locate the cursik index for the equipped item.
        popup.set_caller(name);
        if let Some(b) = state.game.player_side.builder.as_ref() {
            popup.set_saved_config(*b.cfg());
            popup.refresh_current_pos(b);
            // CIFaceButton.cpp:190-310 — colour each row by affordability
            // before showing the menu.
            let mut bank = [0i32; 4];
            for r in crate::matrix_game::config::Resource::ALL {
                bank[r as usize] = state.game.player_side.get_resource_amount(r);
            }
            popup.refresh_affordability(b, &bank);
        }
        log::debug!("popup: opened for pylon {}", name);
        state.iface_list.popup_restore_pending = None;
        state.iface_list.popup = Some(popup);
    }
}

/// Returns `true` when the hover changed the live preview config so
/// the caller can reset the build counter (Djeans007 resets
/// `m_RCountControl` at CConstructor.cpp:578).
fn preview_popup_hover(
    builder: &mut crate::matrix_game::interface::constructor::RobotBuilder,
    popup: Option<&mut crate::matrix_game::interface::iface_menu::CIFaceMenu>,
) -> bool {
    let Some(popup) = popup else {
        return false;
    };
    if popup.previewed == popup.hovered {
        return false;
    }
    popup.previewed = popup.hovered;
    let Some(idx) = popup.hovered else {
        // Cursor left the popup rows — restore the saved preview
        // and re-apply the originating pylon's focused label/price
        // for the equipped component, so the Base-panel left card
        // keeps showing the equipped item rather than going blank.
        // The C++ leaves `m_FocusedElement` pointed at the pylon
        // for the entire popup lifetime (CIFaceMenu.cpp:383 commented
        // out), and `Djeans007` only fires on row hover.
        if let Some(saved) = popup.saved_config {
            builder.apply_config(saved);
        }
        builder.refresh_current_focus();
        return true;
    };
    let Some(item) = popup.items.get(idx).cloned() else {
        return false;
    };
    let ty = popup.parent.unit_type();
    builder.djeans007(ty, item.kind, popup.parent.pilon());
    // Port of the per-hover `RemoteFocusElement` fire in the C++
    // popup loop (CIFaceList::OnMouseMove + CConstructor.cpp:912-958).
    // Updates the focused label/description/price card on the Base
    // panel so hovering a popup row gives the same readouts as
    // hovering the equivalent template button would.
    builder.set_labels_and_price(ty, item.kind);
    true
}

/// Seed dynamic `[key]` replacement values on `IFaceList::hint_replacer`
/// based on the hovered element name. Port of
/// `CIFaceList::AddHintReplacements` (CInterface.cpp:4439-4540) — we
/// only populate values for the element the pointer currently owns so
/// the map stays lean and the per-frame cost is O(1).
///
/// The original mutates the global `PAR_REPLACE` block in place;
/// `HintReplacer` is the Rust equivalent. Values that depend on live
/// state (resource income, robot counts) are refreshed every frame so
/// a long-held hover shows up-to-date numbers.
fn refresh_hint_replacements(state: &mut AppState) {
    use crate::matrix_game::config::Resource;

    // `HintSystem::update` early-returns when nothing is hovered, so
    // we can skip the full refresh when the focused element has no
    // hint. Cheaper than rebuilding the income query every frame.
    let Some((pi, ei)) = state.iface_list.focused else {
        return;
    };
    let Some(panel) = state.iface_list.panels.get(pi) else {
        return;
    };
    let Some(elem) = panel.elements.get(ei) else {
        return;
    };
    if elem.hint_template.is_empty() {
        return;
    }
    let elem_name = elem.name.clone();
    let side_id = state.game.player_side.id;
    let repl = &mut state.iface_list.hint_replacer;
    match elem_name.as_str() {
        "thz" => {
            let (base_i, fa_i) = state.game.compute_resource_income(side_id, Resource::Titan);
            repl.set("_titan_income", (base_i + fa_i).to_string());
        }
        "enhz1" | "enhz2" => {
            let (base_i, fa_i) = state
                .game
                .compute_resource_income(side_id, Resource::Energy);
            repl.set("_energy_income", (base_i + fa_i).to_string());
        }
        "elhz" => {
            let (base_i, fa_i) = state
                .game
                .compute_resource_income(side_id, Resource::Electronics);
            repl.set("_electronics_income", (base_i + fa_i).to_string());
        }
        "phz" => {
            let (base_i, fa_i) = state
                .game
                .compute_resource_income(side_id, Resource::Plasma);
            repl.set("_plasma_income", (base_i + fa_i).to_string());
        }
        "rvhz" => {
            let total = state.game.player_side.robots_cnt;
            let max = state.game.compute_max_side_robots(side_id);
            repl.set("_total_robots", total.to_string());
            repl.set("_max_robots", max.to_string());
        }
        "tur1" | "tur2" | "tur3" | "tur4" => {
            // Port of CInterface.cpp:4463-4519 turret hint replacements.
            // One call per turret slot; the template `BuildTurret`
            // references `_turret_name`, `_turret_range`,
            // `_turret_structure`, `_turret_damage`, `_turret_res1..4`.
            let idx = match elem_name.as_str() {
                "tur1" => 0,
                "tur2" => 1,
                "tur3" => 2,
                "tur4" => 3,
                _ => unreachable!(),
            };
            let (name_label, range_label) = state.iface_list.hint_replacer.turret_label(idx);
            let name_label = name_label.to_string();
            let range_label = range_label.to_string();
            let cfg = crate::matrix_game::config::global();
            let cannon = cfg.turrets.cannons[idx];
            // Structure: hitpoint / 10 (matches CInterface.cpp:4467).
            let structure = (cannon.hitpoint / 10.0) as i32;
            // Damage per second — shots/sec × per-shot damage, then
            // /10 to match the UI's display scale. Mirrors
            // CInterface.cpp:4464 `damage = (1 / (cooldown/1000)) *
            // m_RobotDamages[…].damage`.
            use crate::matrix_game::effects::weapon::{
                weap_to_index, WEAPON_CANNON0, WEAPON_CANNON1, WEAPON_CANNON2, WEAPON_CANNON3,
            };
            let cannon_weap = [
                WEAPON_CANNON0,
                WEAPON_CANNON1,
                WEAPON_CANNON2,
                WEAPON_CANNON3,
            ][idx];
            let cannon_idx = weap_to_index(cannon_weap).unwrap_or(0);
            let cooldown_ms = cfg.weapon_cooldown.table[cannon_idx];
            let per_shot_damage = cfg.robot_damages.table[cannon_idx].damage;
            let dps = if cooldown_ms > 0 {
                ((1000.0 / cooldown_ms as f32) * per_shot_damage as f32) as i32
            } else {
                0
            };
            let dmg10 = dps / 10;
            // Turret 1 + 4 render their damage as "X+X" in the hint
            // (CInterface.cpp:4468 / :4510) — both burst-weapons in
            // the shipped game; the other two use the plain value.
            let damage_str = if matches!(idx, 0 | 3) {
                format!("{dmg10}+{dmg10}")
            } else {
                dmg10.to_string()
            };
            let repl = &mut state.iface_list.hint_replacer;
            repl.set("_turret_name", name_label);
            repl.set("_turret_range", range_label);
            repl.set("_turret_structure", structure.to_string());
            repl.set("_turret_damage", damage_str);
            for i in 0..Resource::ALL.len() {
                let v = cannon.resources[i];
                let key = format!("_turret_res{}", i + 1);
                if v > 0 {
                    repl.set(key, v.to_string());
                } else {
                    repl.set(key, String::new());
                }
            }
        }
        // IF_CALL_FROM_HELL — maintenance availability + countdown
        // (CInterface.cpp:4520-4539).
        "callhell" | "bure" => {
            let repl = &mut state.iface_list.hint_replacer;
            repl.set("_ch_cant", String::new());
            repl.set("_ch_can", String::new());
            repl.set("_ch_time_min", String::new());
            repl.set("_ch_time_sec", String::new());
            if state.game.maintenance_disabled() {
                state
                    .iface_list
                    .hint_replacer
                    .set("_ch_cant", "1".to_string());
            } else if state.game.maintenance_time > 0 {
                let ms = state.game.maintenance_time;
                let minutes = ms / 60000;
                let seconds = ms / 1000 - minutes * 60;
                let repl = &mut state.iface_list.hint_replacer;
                repl.set("_ch_time_min", minutes.max(0).to_string());
                repl.set("_ch_time_sec", seconds.max(0).to_string());
            } else {
                state
                    .iface_list
                    .hint_replacer
                    .set("_ch_can", "1".to_string());
            }
        }
        _ => {}
    }
}

/// Advance the constructor preview turntable while the panel is open.
fn tick_builder_preview(state: &mut AppState, step_ms: f32) {
    let active = state
        .game
        .player_side
        .builder
        .as_ref()
        .map(|b| b.active)
        .unwrap_or(false);
    if !active {
        return;
    }
    state.builder_preview.tick(step_ms);
}

/// Per-frame snapshot of what the constructor preview wants drawn.
/// Mirrors the multi-unit `m_Robot->Draw()` walk at
/// MatrixObjectRobot.cpp:319-356 — chassis is required, armor / head /
/// weapons stack above when populated.
pub struct BuilderPreviewQuery {
    pub chassis: crate::matrix_game::robot::ChassisKind,
    pub armor_kind: Option<i32>,
    pub head_kind: Option<i32>,
    pub weapon_kinds: [Option<i32>; 5],
    pub angle_rad: f32,
    pub design_rect: [f32; 4],
}

/// Query helper for the render pass — returns the preview chassis +
/// armor/head/weapon kinds + turntable angle + design-space viewport
/// rect when the panel is open and the live config has a chassis
/// selected. Returns `None` when nothing should be drawn (panel
/// closed or chassis empty).
fn builder_preview_query(state: &mut AppState) -> Option<BuilderPreviewQuery> {
    let active = state
        .game
        .player_side
        .builder
        .as_ref()
        .map(|b| b.active)
        .unwrap_or(false);
    if !active {
        return None;
    }
    // Preview viewport rect — port of `SetRenderProps` reading the
    // `Const{X,Y,Width,Height}` panel-level params at
    // CInterface.cpp:208-213. The C++ adds these to the panel's
    // resolved screen origin to get the absolute viewport rect; we
    // do the equivalent in `render_preview_full` (panel.resolved_pos
    // + design_rect * scale).
    let design_rect = state
        .iface_list
        .panel("Base")
        .and_then(|p| p.const_rect)
        .unwrap_or([426.0, 56.0, 221.0, 314.0]);
    let cfg = *state.game.player_side.builder.as_ref()?.cfg();
    let ticket = state.builder_preview.ticket(&cfg, design_rect)?;
    let kind_or_none = |k: i32| if k >= 1 { Some(k) } else { None };
    Some(BuilderPreviewQuery {
        chassis: ticket.chassis,
        armor_kind: kind_or_none(cfg.hull.unit.kind.0),
        head_kind: kind_or_none(cfg.head.kind.0),
        weapon_kinds: [
            kind_or_none(cfg.weapon[0].kind.0),
            kind_or_none(cfg.weapon[1].kind.0),
            kind_or_none(cfg.weapon[2].kind.0),
            kind_or_none(cfg.weapon[3].kind.0),
            kind_or_none(cfg.weapon[4].kind.0),
        ],
        angle_rad: ticket.rotation_rad,
        design_rect: ticket.design_rect,
    })
}

/// Per-frame placement-preview tick — port of
/// `CMatrixSideUnit::LogicTakt`'s BUILDING_TURRET branch
/// (MatrixSide.cpp:528-585). Returns a `GhostCannon` to render when a
/// turret-build session is active; `None` otherwise.
///
/// Steps (all mirrored from the C++):
///   1. Ray-cast the cursor onto the terrain plane at the parent base's
///      `build_z`. (The C++ traces against landscape+water; intersecting
///      a flat plane is good enough since slots are at base elevation.)
///   2. Walk the parent's `turret_places[]` and find the nearest free
///      slot inside `SNAP_DIST` (4 move-cells in C++ → 40 world units).
///   3. Set `can_build` = `turrets_have < turrets_max` AND a slot was
///      hovered AND the player has the required resources.
///   4. Snap ghost pos to the slot if hovered, otherwise to the raw
///      cursor; smooth-rotate ghost angle toward the slot's rest angle
///      using the C++ damping `dang * (1 - 0.99^ms)`.
fn update_turret_build(
    state: &mut AppState,
    step_ms: i32,
) -> (
    Option<crate::matrix_game::object_cannon::GhostCannon>,
    Vec<crate::matrix_game::slot_marker::SlotMarker>,
) {
    use crate::matrix_game::config::Resource;
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::Building;
    use crate::matrix_game::object_cannon::GhostCannon;
    use crate::matrix_game::slot_marker::SlotMarker;

    if !state.iface_list.turret_build.is_active() {
        return (None, Vec::new());
    }
    let Some(parent_id) = state.iface_list.turret_build.parent else {
        return (None, Vec::new());
    };

    // Snapshot the parent's slot table so the cursor scan doesn't fight
    // the borrow on `state.iface_list.turret_build`.
    let (parent_pos, parent_z, slots, turrets_have, turrets_max, parent_side) = {
        let Some(obj) = state.game.objects.get(parent_id) else {
            return (None, Vec::new());
        };
        if !matches!(obj.core().obj_type, ObjectType::Building) {
            return (None, Vec::new());
        }
        let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
        (
            b.pos,
            b.build_z,
            b.turret_places.clone(),
            b.turrets_have,
            b.turrets_max,
            b.side,
        )
    };

    // Refuse builds on enemy bases up front — same gate as the click
    // path (CConstructor.cpp:225-227).
    if parent_side != state.game.player_side.id {
        return (None, Vec::new());
    }

    // Slot markers — port of `CreatePlacesShow`
    // (MatrixObjectBuilding.cpp:1617). Visible whenever the picker
    // is open (regardless of whether a kind has been committed) so
    // the player can see where they may build before clicking.
    // Spot radius — port of MatrixEffectLandscapeSpot.cpp:233-236:
    //   xc = SPOT_SIZE * scalex * cos(angle)
    // SPOT_SIZE = 5.0 (MatrixEffectLandscapeSpot.hpp:22), and the
    // scale passed at MatrixObjectBuilding.cpp:1640 is 6.0. So the
    // quad's half-extent is 30 world units.
    const SPOT_SIZE: f32 = 5.0;
    const SPOT_TURRET_SCALE: f32 = 6.0;
    let radius = SPOT_SIZE * SPOT_TURRET_SCALE;
    let markers: Vec<SlotMarker> = slots
        .iter()
        .filter(|p| p.cannon_type < 0)
        .map(|p| {
            // Sample terrain at the four corners + center; lift the
            // marker above the highest of them so corners don't sink
            // below uneven terrain (which clips the decal because it's
            // a single flat quad rather than a true terrain-conforming
            // decal).
            let cx = p.world.x;
            let cy = p.world.y;
            let r = radius;
            let z_max = state
                .map
                .get_z(cx, cy)
                .max(state.map.get_z(cx + r, cy + r))
                .max(state.map.get_z(cx - r, cy + r))
                .max(state.map.get_z(cx + r, cy - r))
                .max(state.map.get_z(cx - r, cy - r))
                .max(parent_z);
            SlotMarker {
                pos: p.world,
                pos_z: z_max + 1.0,
                radius,
            }
        })
        .collect();

    // No kind chosen yet → markers only, no ghost.
    let Some(kind) = state.iface_list.turret_build.kind.map(|k| k as i32) else {
        return (None, markers);
    };

    // Ray-cast cursor onto z=parent_z plane.
    let [cx, cy] = state.cursor;
    let w = state.gfx.config.width as f32;
    let h = state.gfx.config.height as f32;
    let (origin, dir) = state.camera.screen_to_world_ray(cx, cy, w, h);
    let cursor_world = if dir.z.abs() > 1e-6 {
        let t = (parent_z - origin.z) / dir.z;
        if t > 0.0 {
            glam::Vec2::new(origin.x + dir.x * t, origin.y + dir.y * t)
        } else {
            parent_pos
        }
    } else {
        parent_pos
    };

    // Snap to nearest free slot — C++ uses `rr < 4` in move-cell space
    // (MatrixSide.cpp:1645). 4 move-cells × GLOBAL_SCALE_MOVE = 40
    // world units.
    const SNAP_DIST: f32 = 4.0 * crate::matrix_game::map::GameMap::GLOBAL_SCALE_MOVE;
    const SNAP_DIST_SQ: f32 = SNAP_DIST * SNAP_DIST;
    let mut hovered: Option<(usize, f32)> = None;
    for (i, p) in slots.iter().enumerate() {
        if p.cannon_type > 0 {
            continue; // occupied
        }
        let d2 = (p.world - cursor_world).length_squared();
        if d2 < SNAP_DIST_SQ && hovered.map_or(true, |(_, best)| d2 < best) {
            hovered = Some((i, d2));
        }
    }

    // Resource check — port of `IsEnoughResources` (CSide.cpp:?).
    let cost = crate::matrix_game::config::global().turrets.cost_of(kind);
    let resources_ok = Resource::ALL
        .iter()
        .all(|r| state.game.player_side.get_resource_amount(*r) >= cost.resources[*r as usize]);

    let can_build = hovered.is_some() && turrets_have < turrets_max && resources_ok;

    let tb = &mut state.iface_list.turret_build;
    tb.cursor_world = (cursor_world.x, cursor_world.y);
    tb.hovered_slot = hovered.map(|(i, _)| i as i32);
    tb.can_build = can_build;
    // Ghost Z: keep at base_z + small platform offset so the cannon sits
    // on top of the base mesh. The C++ pulls this off `m_BuildZ` directly.
    tb.ghost_z = parent_z + 8.0;

    let (target_pos, target_angle) = if let Some((i, _)) = hovered {
        let p = &slots[i];
        (p.world, p.angle)
    } else {
        (cursor_world, tb.ghost_angle)
    };
    tb.ghost_pos = target_pos;

    // Smooth-rotate to the slot's rest angle. Port of MatrixSide.cpp:578:
    //   m_Angle += dang * (1 - 0.99^ms)
    // where `dang` is the shortest-arc angle from current to target.
    let dang = shortest_arc(tb.ghost_angle, target_angle);
    let damping = 1.0 - (0.99_f64).powf(step_ms.max(0) as f64) as f32;
    tb.ghost_angle += dang * damping;

    let ghost = GhostCannon {
        kind,
        pos: tb.ghost_pos,
        pos_z: tb.ghost_z,
        angle: tb.ghost_angle,
        can_build: tb.can_build,
        side: state.game.player_side.id,
    };
    (Some(ghost), markers)
}

/// Shortest signed angle from `a` to `b` in radians, in the range
/// `(-π, π]`. Port of the C++ `AngleDist` helper used at
/// MatrixSide.cpp:576.
fn shortest_arc(a: f32, b: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let mut d = (b - a) % two_pi;
    if d > std::f32::consts::PI {
        d -= two_pi;
    } else if d < -std::f32::consts::PI {
        d += two_pi;
    }
    d
}

/// Port of the "click during PREORDER_BUILD_TURRET" path
/// (MatrixFormGame.cpp:1498-1512). If the click lands on the parent
/// base's ring, queue the turret + deduct cost; otherwise cancel.
fn try_place_turret(state: &mut AppState, _cx: f32, _cy: f32, _w: f32, _h: f32) {
    use crate::matrix_game::config::Resource;
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::Building;

    let Some(parent_id) = state.iface_list.turret_build.parent else {
        state.iface_list.turret_build.cancel();
        return;
    };
    let Some(kind) = state.iface_list.turret_build.kind else {
        state.iface_list.turret_build.cancel();
        return;
    };

    // The per-frame `update_turret_build` already snapped the cursor to
    // a slot (or determined no slot is hovered). Honour its decision —
    // the C++ click path also reads `m_CannonForBuild.m_CanBuildFlag`
    // set by the LogicTakt branch (MatrixSide.cpp:626).
    if !state.iface_list.turret_build.can_build {
        log::debug!("turret: click without valid slot — cancelling placement");
        state.iface_list.turret_build.cancel();
        return;
    }
    let Some(slot_idx) = state.iface_list.turret_build.hovered_slot else {
        state.iface_list.turret_build.cancel();
        return;
    };
    let ghost_pos = state.iface_list.turret_build.ghost_pos;
    let ghost_z = state.iface_list.turret_build.ghost_z;
    let ghost_angle = state.iface_list.turret_build.ghost_angle;

    let turret_cost = crate::matrix_game::config::global()
        .turrets
        .cost_of(kind as i32);

    let placement_ok = {
        let Some(obj) = state.game.objects.get_mut(parent_id) else {
            state.iface_list.turret_build.cancel();
            return;
        };
        if !matches!(obj.core().obj_type, ObjectType::Building) {
            state.iface_list.turret_build.cancel();
            return;
        }
        let b: &mut Building = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
        if b.side != state.game.player_side.id {
            state.iface_list.turret_build.cancel();
            return;
        }
        // Reserve the slot — flips m_CannonType to the kind so subsequent
        // hovers in the same session won't snap to it.
        b.queue_turret_slot(slot_idx, kind as i32)
    };
    if !placement_ok {
        log::debug!("turret: slot {} no longer free — cancelling", slot_idx);
        state.iface_list.turret_build.cancel();
        return;
    }

    // Spawn the Cannon object immediately — the build-stack timer
    // still runs for the cost/progress UI, but the C++ mounts the
    // cannon on the building as soon as BeginBuildTurret commits.
    // HP starts at 0 + state UNDER_CONSTRUCTION + invulnerable; the
    // build-stack tick ramps HP up over `turret_build_time_ms`.
    let mut cannon = crate::matrix_game::object_cannon::Cannon::new(
        ghost_pos,
        ghost_z,
        ghost_angle,
        state.game.player_side.id,
        kind as i32,
        Some(parent_id),
        slot_idx,
    );
    cannon.begin_construction();
    let id = state.game.objects.spawn(Box::new(cannon));
    if let Some(obj) = state.game.objects.get_mut(id) {
        let c: &mut crate::matrix_game::object_cannon::Cannon = unsafe {
            &mut *(obj as *mut dyn crate::matrix_game::map_static::MapStatic
                as *mut crate::matrix_game::object_cannon::Cannon)
        };
        c.self_id = Some(id);
    }
    state.game.objects.add_lt(id);

    for r in Resource::ALL {
        state
            .game
            .player_side
            .add_resource_amount(r, -turret_cost.resources[r as usize]);
    }
    log::debug!(
        "turret: placed {:?} on slot {} as object {:?}",
        kind,
        slot_idx,
        id,
    );
    state.iface_list.turret_build.cancel();
}

/// Build the `CheckUpCtx` (CCounter.cpp:66-99 inputs) for the
/// build-multiplier counter. Reads the player side's per-resource pool
/// and the live preview's per-unit cost. The remaining fields are
/// computed by scanning the live object list so the counter and build
/// button honor robot-cap and base-stack limits like the C++.
fn build_counter_ctx(state: &AppState) -> crate::matrix_game::interface::counter::CheckUpCtx {
    use crate::matrix_game::config::Resource;
    use crate::matrix_game::interface::constructor::UnitPrice;
    use crate::matrix_game::interface::counter::CheckUpCtx;
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::{Building, BuildingType};
    use crate::matrix_game::robot::Robot;
    let per_unit_price = state
        .game
        .player_side
        .builder
        .as_ref()
        .map(|b| b.construction_price())
        .unwrap_or_else(UnitPrice::zero);
    let mut side_resources = [0i32; 4];
    for r in Resource::ALL {
        side_resources[r as usize] = state.game.player_side.get_resource_amount(r);
    }
    let mut side_robots = 0;
    let mut robots_in_stack = 0;
    let mut bases = 0;
    let mut factories = 0;
    state
        .game
        .objects
        .for_each_live(|_, obj| match obj.core().obj_type {
            ObjectType::RobotAi => {
                let r: &Robot = unsafe { &*(obj as *const dyn MapStatic as *const Robot) };
                if r.side == state.game.player_side.id {
                    side_robots += 1;
                }
            }
            ObjectType::Building => {
                let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
                if b.side == state.game.player_side.id {
                    if b.kind == BuildingType::Base {
                        bases += 1;
                        robots_in_stack += b.build_stack.items() as i32;
                    } else {
                        factories += 1;
                    }
                }
            }
            _ => {}
        });
    let active_base_stack_items = state
        .game
        .active_object()
        .and_then(|id| state.game.objects.get(id))
        .filter(|o| matches!(o.core().obj_type, ObjectType::Building))
        .and_then(|obj| {
            let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };
            (b.kind == BuildingType::Base).then_some(b.build_stack.items() as i32)
        });
    CheckUpCtx {
        per_unit_price,
        side_resources,
        side_robots,
        robots_in_stack,
        max_side_robots: bases * 3 + if bases == 0 { 0 } else { 4 } + factories,
        active_base_stack_items,
    }
}

/// Port of the `m_RCountControl->Reset()` at the top of every
/// `SuperDjeans` / `Djeans007` (CConstructor.cpp:443,578) plus the
/// trailing `m_RCountControl->CheckUp()` — every component change
/// resets the build multiplier to 1 and revalidates it against the
/// side's resources/caps. Called on each dispatch path that mutates
/// the constructor's configuration.
fn reset_build_counter(state: &mut AppState) {
    state.iface_list.r_count_control.reset();
    let ctx = build_counter_ctx(state);
    state.iface_list.r_count_control.check_up(ctx);
}

/// Port of `CConstructor::RemoteBuild` (CConstructor.cpp:223-250).
fn commit_and_queue_robot(state: &mut AppState) {
    use crate::matrix_game::config::Resource;
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::{Building, BuildingType};
    use crate::matrix_game::side::CurrSel;

    if state.game.player_side.curr_sel != CurrSel::BaseSelected {
        log::debug!("cobuild: no base selected");
        return;
    }
    let Some(id) = state.game.active_object() else {
        return;
    };

    // CConstructor.cpp:225-227 — player-side guard.
    // No upfront affordability check: the C++ relies on
    // `CIFaceCounter::CheckUp` to cap the counter to what the player
    // can afford / fit in the base stack. The cobuild button is also
    // disabled when `build_enabled` is false (see refresh visibility).
    let (cfg, price, count) = {
        let Some(b) = state.game.player_side.builder.as_ref() else {
            return;
        };
        (
            *b.cfg(),
            b.construction_price(),
            state.iface_list.r_count_control.counter().max(1),
        )
    };

    let Some(obj) = state.game.objects.get_mut(id) else {
        return;
    };
    if !matches!(obj.core().obj_type, ObjectType::Building) {
        return;
    }
    let b: &mut Building = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
    if b.kind != BuildingType::Base {
        return;
    }
    // Port of `CConstructor::RemoteBuild`'s guard at CConstructor.cpp:
    // 225-227 — `m_Base->m_Side != PLAYER_SIDE` returns early, so you
    // can't queue robots on enemy / neutral bases even if the panel is
    // somehow open.
    if b.side != state.game.player_side.id {
        log::debug!(
            "cobuild: refused — base side {} != player side {}",
            b.side,
            state.game.player_side.id
        );
        return;
    }
    // CConstructor.cpp:233-235 — push `counter` robots onto the base
    // stack (StackRobot drops the request if the stack is full).
    let mut queued = 0;
    for _ in 0..count {
        if !b.queue_robot(cfg) {
            break;
        }
        queued += 1;
    }
    // CConstructor.cpp:237-242 — deduct price × counter. NOTE: the C++
    // deducts even if some StackRobot calls were dropped by a full
    // stack; CIFaceCounter::CheckUp is supposed to have prevented that
    // case ahead of time. We mirror the C++ deduction faithfully.
    for r in Resource::ALL {
        let spent = price.resources[r as usize] * count;
        state.game.player_side.add_resource_amount(r, -spent);
    }
    // CConstructor.cpp:231 — push to global config history.
    state.iface_list.history.add(cfg);
    // CConstructor.cpp:244-246 — close the panel via ResetGroupNClose.
    // That call only clears `m_Active`; it does NOT reset the live
    // construction state. `RemoteBuild` itself stacks robots with
    // `StackRobot` (CConstructor.cpp:233-235) which never resets either,
    // so the user's design persists across builds.
    if let Some(b) = state.game.player_side.builder.as_mut() {
        b.deactivate();
    }
    state.is_paused = false;
    // CConstructor.cpp:247-248 — reset + revalidate the counter.
    state.iface_list.r_count_control.reset();
    let ctx = build_counter_ctx(state);
    state.iface_list.r_count_control.check_up(ctx);
    log::debug!(
        "cobuild: queued {} (chas={},hull={},head={}); left T/E/En/P={}/{}/{}/{}",
        queued,
        cfg.chassis.kind.0,
        cfg.hull.unit.kind.0,
        cfg.head.kind.0,
        state.game.player_side.get_resource_amount(Resource::Titan),
        state
            .game
            .player_side
            .get_resource_amount(Resource::Electronics),
        state.game.player_side.get_resource_amount(Resource::Energy),
        state.game.player_side.get_resource_amount(Resource::Plasma),
    );
}

/// Queue progress bars for the active building's build stack. Ports
/// the `m_BS.m_PB.Modify(...) + CreateClone(PBC_CLONE1, x, y, 87)`
/// dispatch at MatrixObjectBuilding.cpp:1681-1689 — a 87-pixel-wide
/// bar drawn over the `if/Main` panel coords that CInterface resolves
/// at `(m_xPos+283, m_yPos+71)`.  We route through
/// `CInterface::element_rect("prog")` so the bar lands exactly where
/// the Static the C++ draws underneath it lives.
fn refresh_progress_bars(state: &mut AppState) {
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::Building;
    use crate::matrix_game::progress_bar::ProgressBar;
    use crate::matrix_game::side::CurrSel;

    state.progress_bars.clear();

    // ── Floating HP bars over robots / turrets / buildings ──────────
    {
        use crate::matrix_game::common::{TRACE_ALL, TRACE_LANDSCAPE};
        use crate::matrix_game::interface::interface::DESIGN_H;
        use crate::matrix_game::map_trace::{trace, TraceStop};

        let w = state.gfx.config.width as f32;
        let h = state.gfx.config.height as f32;
        let scale = (h / DESIGN_H).max(0.1);
        let bar_h = {
            let d = state.progress_bars.bar_height_design();
            if d > 0.0 {
                d
            } else {
                8.0
            }
        };

        // Arm the bar of the object under the cursor for 1s — the
        // per-frame mouse trace at MatrixMap.cpp:1150-1178.
        let [mx, my] = state.cursor;
        if mx >= 0.0 {
            let (o, d) = state.camera.screen_to_world_ray(mx, my, w, h);
            let (stop, _) = trace(
                &state.map,
                &state.game.objects,
                o,
                o + d * 10_000.0,
                TRACE_ALL,
                state.game.objects.arcaded_object,
            );
            if let TraceStop::Object(id) = stop {
                if let Some(obj) = state.game.objects.get_mut(id) {
                    obj.show_hitpoint();
                }
            }
        }

        // Collect every armed bar (`BeforeDraw` PB blocks), gated by a
        // landscape LOS trace from the camera like the originals.
        let eye = state.camera.eye_pos_world();
        let ids: Vec<_> = state.game.objects.iter_live().collect();
        for id in ids {
            let Some(obj) = state.game.objects.get(id) else {
                continue;
            };
            let Some(bar) = obj.hitpoint_bar(&state.map) else {
                continue;
            };
            let (stop, _) = trace(
                &state.map,
                &state.game.objects,
                eye,
                bar.anchor,
                TRACE_LANDSCAPE,
                None,
            );
            if stop != TraceStop::None {
                continue;
            }
            // Projection happens at upload time with the render
            // camera; only the screen-pixel offsets are fixed here.
            state
                .progress_bars
                .push_world(crate::matrix_game::progress_bar::WorldBar {
                    anchor: bar.anchor,
                    x_off: bar.x_off * scale,
                    y_off: bar.y_off * scale,
                    width: bar.width * scale,
                    height: bar_h * scale,
                    fill: bar.fill,
                });
        }
    }

    // Robot-selection panel clones (CInterface.cpp:1556-1565): the
    // active robot's HP bar at (+68, +179) width 68 — same slot the
    // building HP clone uses — plus, with more than one robot
    // selected, a small 46px bar under each group icon
    // (`CreateProgressBarClone(icon_x, icon_y+36, 46, PBC_CLONE1)`).
    if matches!(state.game.player_side.curr_sel, CurrSel::RobotsSelected) {
        use crate::matrix_game::interface::interface::DESIGN_H;
        use crate::matrix_game::robot::Robot;

        let w = state.gfx.config.width as f32;
        let h = state.gfx.config.height as f32;
        let scale = (h / DESIGN_H).max(0.1);
        let bar_h = {
            let d = state.progress_bars.bar_height_design();
            if d > 0.0 {
                d
            } else {
                8.0
            }
        };
        if let Some(main) = state.iface_list.panel("Main") {
            let [panel_x, panel_y] = main.resolved_pos(w, h, scale);
            let ids = state.game.player_side.selected.clone();
            let mut fills: Vec<f32> = Vec::with_capacity(ids.len());
            let mut sel_idx = 0usize;
            for id in &ids {
                let Some(obj) = state.game.objects.get(*id) else {
                    continue;
                };
                if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
                    continue;
                }
                let r: &Robot = unsafe { &*(obj as *const dyn MapStatic as *const Robot) };
                if state.game.player_side.active_object == Some(*id) {
                    sel_idx = fills.len();
                }
                fills.push((r.hit_point / r.hit_point_max.max(1.0)).clamp(0.0, 1.0));
            }
            // Group-icon grid constants — must match the `_dyngroup_`
            // layout in interface.rs (CInterface.cpp:3678-3697).
            const GROUP_X0: f32 = 225.0;
            const GROUP_Y0: f32 = 49.0;
            const GROUP_DX: f32 = 48.0;
            const GROUP_DY: f32 = 49.0;
            for (i, fill) in fills.iter().enumerate().take(9) {
                if i == sel_idx {
                    state.progress_bars.push(ProgressBar {
                        rect: [
                            panel_x + 68.0 * scale,
                            panel_y + 179.0 * scale,
                            68.0 * scale,
                            bar_h * scale,
                        ],
                        fill: *fill,
                    });
                }
                if fills.len() > 1 {
                    let col = (i % 3) as f32;
                    let row = (i / 3) as f32;
                    state.progress_bars.push(ProgressBar {
                        rect: [
                            panel_x + (GROUP_X0 + col * GROUP_DX) * scale,
                            panel_y + (GROUP_Y0 + row * GROUP_DY + 36.0) * scale,
                            46.0 * scale,
                            bar_h * scale,
                        ],
                        fill: *fill,
                    });
                }
            }
        }
    }

    // Only fires for a selected building — matches the C++ dispatch
    // at CInterface.cpp:1576-1578 (HP-bar clone) and
    // MatrixObjectBuilding.cpp:1685-1689 (build-progress clone).
    if !matches!(
        state.game.player_side.curr_sel,
        CurrSel::BaseSelected | CurrSel::BuildingSelected
    ) {
        return;
    }
    let Some(id) = state.game.active_object() else {
        return;
    };
    let Some(obj) = state.game.objects.get(id) else {
        return;
    };
    if !matches!(obj.core().obj_type, ObjectType::Building) {
        return;
    }
    let b: &Building = unsafe { &*(obj as *const dyn MapStatic as *const Building) };

    let w = state.gfx.config.width as f32;
    let h = state.gfx.config.height as f32;
    let Some(main) = state.iface_list.panel("Main") else {
        log::warn!("progress: no Main panel");
        return;
    };

    use crate::matrix_game::interface::interface::DESIGN_H;
    let scale = (h / DESIGN_H).max(0.1);
    let [panel_x, panel_y] = main.resolved_pos(w, h, scale);
    let bar_h_design = {
        let d = state.progress_bars.bar_height_design();
        if d > 0.0 {
            d
        } else {
            16.0
        }
    };

    // Port of CInterface.cpp:1578 — `bld->CreateProgressBarClone(
    // m_xPos+68, m_yPos+179, 68, PBC_CLONE2)`. HP bar, always visible
    // when a building is selected, width 68 at (+68, +179).
    const HP_OFFSET_X: f32 = 68.0;
    const HP_OFFSET_Y: f32 = 179.0;
    const HP_WIDTH: f32 = 68.0;
    let hp_fill = if b.hit_point_max > 0.0 {
        (b.hit_point / b.hit_point_max).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let hp_rect = [
        panel_x + HP_OFFSET_X * scale,
        panel_y + HP_OFFSET_Y * scale,
        HP_WIDTH * scale,
        bar_h_design * scale,
    ];
    state.progress_bars.push(ProgressBar {
        rect: hp_rect,
        fill: hp_fill,
    });

    // Port of MatrixObjectBuilding.cpp:1676-1689 — build-queue
    // progress bar at (+283, +71), width 87, PBC_CLONE1. Only when an
    // item is queued for construction.
    if !b.build_stack.is_empty() {
        const PB_OFFSET_X: f32 = 283.0;
        const PB_OFFSET_Y: f32 = 71.0;
        const PB_WIDTH: f32 = 87.0;
        let rect = [
            panel_x + PB_OFFSET_X * scale,
            panel_y + PB_OFFSET_Y * scale,
            PB_WIDTH * scale,
            bar_h_design * scale,
        ];

        // One-shot log so we can verify the rect + fill on first build.
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log::debug!(
                "progress: pushing bar rect=({:.0},{:.0},{:.0},{:.0}) fill={:.2}",
                rect[0],
                rect[1],
                rect[2],
                rect[3],
                b.build_stack.progress(),
            );
        }

        state.progress_bars.push(ProgressBar {
            rect,
            fill: b.build_stack.progress(),
        });
    }
}

/// Port of `CInterface::LogicTakt`'s per-frame visibility dispatch
/// for the `if/Main` panel (CInterface.cpp:1214-1635). Reads
/// `player_side.curr_sel` + the currently-active building's kind /
/// stack state to decide which `if/Main` elements should show this
/// frame. Other panels don't have their dispatch ported yet; they
/// stay hidden.
fn refresh_interface_visibility(state: &mut AppState) {
    use crate::matrix_game::interface::MainVisibilityCtx;
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::{Building, BuildingType};
    use crate::matrix_game::side::CurrSel;

    let curr_sel = state.game.player_side.curr_sel;
    let player_side_id = state.game.player_side.id;
    // Pull building context when the selection is a Building. Also
    // carry the selected building's `m_Side` so the Base-panel build
    // button can gate on "is this our base?" — port of the
    // `m_Base->m_Side == PLAYER_SIDE` guard at CConstructor.cpp:225-227.
    use crate::matrix_game::interface::constructor::RobotConfig;
    use crate::matrix_game::interface::MAX_STACK_ICONS;

    // First pass — read-only snapshot of the build stack: per-slot
    // turret kinds and per-slot robot configs. We can't borrow
    // `state.game.objects` and `state.robot_icons`/`state.iface_renderer`
    // mutably at the same time, so the icon-bake pass runs after the
    // borrow is released.
    let (
        kind,
        stack_empty,
        stack_items,
        turrets_max,
        hit_point,
        hit_point_max,
        active_side,
        stack_kinds,
        stack_robot_cfgs,
    ): (
        Option<BuildingType>,
        bool,
        i32,
        i32,
        f32,
        f32,
        i32,
        [Option<i32>; MAX_STACK_ICONS],
        [Option<RobotConfig>; MAX_STACK_ICONS],
    ) = match curr_sel {
        CurrSel::BaseSelected | CurrSel::BuildingSelected => {
            use crate::matrix_game::object_building::PendingKind;
            let active = state.game.active_object();
            active
                .and_then(|id| state.game.objects.get(id))
                .filter(|o| matches!(o.core().obj_type, ObjectType::Building))
                .map(|o| {
                    let b: &Building = unsafe { &*(o as *const dyn MapStatic as *const Building) };
                    let n = b.build_stack.items() as i32;
                    let mut kinds: [Option<i32>; MAX_STACK_ICONS] = [None; MAX_STACK_ICONS];
                    let mut cfgs: [Option<RobotConfig>; MAX_STACK_ICONS] = [None; MAX_STACK_ICONS];
                    for (slot, item) in b
                        .build_stack
                        .list()
                        .iter()
                        .take(MAX_STACK_ICONS)
                        .enumerate()
                    {
                        match item.kind {
                            PendingKind::Turret { turret_kind, .. } => {
                                kinds[slot] = Some(turret_kind)
                            }
                            PendingKind::Robot(cfg) => cfgs[slot] = Some(cfg),
                        }
                    }
                    (
                        Some(b.kind),
                        n == 0,
                        n,
                        b.turrets_max,
                        b.hit_point,
                        b.hit_point_max,
                        b.side,
                        kinds,
                        cfgs,
                    )
                })
                .unwrap_or((
                    None,
                    true,
                    0,
                    0,
                    0.0,
                    0.0,
                    0,
                    [None; MAX_STACK_ICONS],
                    [None; MAX_STACK_ICONS],
                ))
        }
        _ => (
            None,
            true,
            0,
            0,
            0.0,
            0.0,
            0,
            [None; MAX_STACK_ICONS],
            [None; MAX_STACK_ICONS],
        ),
    };

    // Bake (or look up cached) robot icons for any robot configs
    // queued in the build stack. Each new config triggers a separate
    // off-screen submit, after which the texture is reused for free.
    let mut stack_robot_keys: [Option<String>; MAX_STACK_ICONS] = Default::default();
    if let Some(robots) = state.terrain.robots() {
        let format = state.gfx.config.format;
        for (slot, cfg_opt) in stack_robot_cfgs.iter().enumerate() {
            if let Some(cfg) = cfg_opt {
                stack_robot_keys[slot] = state.robot_icons.ensure(
                    &state.gfx.device,
                    &state.gfx.queue,
                    format,
                    robots,
                    &mut state.iface_renderer,
                    cfg,
                );
            }
        }
    }
    let active_is_player_owned = active_side == player_side_id;

    // Port of `CMatrixSideUnit::GetIncomePerTime(kind, 60000)`
    // (MatrixSide.cpp:352-377). The original's per-ms scaling is
    // commented out, so this returns the flat per-tick rate:
    // `RESOURCES_INCOME_BASE * fu / 100` for a base (3 @ fu=100),
    // `RESOURCES_INCOME` for a factory (10).
    const RESOURCES_INCOME: i32 = 10;
    const RESOURCES_INCOME_BASE: i32 = 3;
    let force_up = 100; // default m_BaseResForce (MatrixSide.hpp:441).
    let income_per_minute = match kind {
        Some(BuildingType::Base) => RESOURCES_INCOME_BASE * force_up / 100,
        Some(_) => RESOURCES_INCOME,
        None => 0,
    };

    let constructor_active = state
        .game
        .player_side
        .builder
        .as_ref()
        .map(|b| b.active)
        .unwrap_or(false);
    let turret_build_active = state.iface_list.turret_build.is_active();
    // `kind.is_some()` ↔ player has clicked tur1..tur4 and we're now in
    // the placement preview (m_CurrentAction == BUILDING_TURRET in C++).
    // CInterface.cpp:1710 hides the picker buttons in that sub-mode.
    let turret_kind_committed = state.iface_list.turret_build.kind.is_some();

    // Per-kind picker DISABLED + buca DISABLED — port of
    // CInterface.cpp:1602-1605 (`buca`) and 1713-1738 (`tur1..4`):
    // a button is DISABLED when the building has no free turret slot
    // OR the player can't afford the cannon. `buca` is also disabled
    // when the build stack is full.
    //
    // Also collect per-slot installed `cannon_type` so the Main panel
    // can overlay `bt{N}` icons on the `podl{N}` strip — port of
    // `CIFaceList::CreateDynamicTurrets` (CInterface.cpp:4572-4621).
    let mut turret_disabled = [false; 4];
    let mut buca_disabled = false;
    let mut installed_turret_kinds: [Option<i32>; 4] = [None; 4];
    if matches!(curr_sel, CurrSel::BaseSelected | CurrSel::BuildingSelected) {
        if let Some(active_id) = state.game.active_object() {
            if let Some(o) = state.game.objects.get(active_id) {
                if matches!(o.core().obj_type, ObjectType::Building) {
                    let b: &Building = unsafe { &*(o as *const dyn MapStatic as *const Building) };
                    let has_free_slot = b.turret_places.iter().any(|p| p.cannon_type < 0);
                    let stack_full = b.build_stack.is_full();
                    let cfg = crate::matrix_game::config::global();
                    let side = &state.game.player_side;
                    let mut affordable = [false; 4];
                    for k in 0..4 {
                        let cost = cfg.turrets.cost_of((k + 1) as i32);
                        affordable[k] = crate::matrix_game::config::Resource::ALL
                            .iter()
                            .all(|r| side.get_resource_amount(*r) >= cost.resources[*r as usize]);
                        turret_disabled[k] = !has_free_slot || !affordable[k];
                    }
                    let any_affordable = affordable.iter().any(|a| *a);
                    buca_disabled = !has_free_slot || stack_full || !any_affordable;
                    for (i, p) in b.turret_places.iter().take(4).enumerate() {
                        if p.cannon_type > 0 {
                            installed_turret_kinds[i] = Some(p.cannon_type);
                        }
                    }
                }
            }
        }
    }

    // Selected-robot panel snapshot. Port of the work_group walk at
    // CInterface.cpp:1227-1311 + CreatePersonal/CreateGroupIcons inputs
    // — for each selected robot (own side or enemy / single or group),
    // collect the cached `RobotConfig` and HP readout, plus the
    // `m_CurrSelNum` index pointing at the active member.
    let mut robot_panel: Option<crate::matrix_game::interface::RobotPanelCtx> = None;
    if matches!(curr_sel, CurrSel::RobotsSelected) {
        use crate::matrix_game::interface::{RobotEntry, RobotPanelCtx};
        use crate::matrix_game::map_static::ObjectId as OId;
        use crate::matrix_game::robot::Robot;

        // Snapshot the selected ids + their cfg/HP/name. Read the
        // robot pointer through MapStatic and downcast — the same
        // cast pattern the marquee/selection code uses elsewhere.
        let primary = state.game.player_side.active_object;
        let ids: Vec<OId> = state.game.player_side.selected.clone();
        let mut snapshot: Vec<(
            OId,
            crate::matrix_game::interface::constructor::RobotConfig,
            String,
            i32,
            i32,
        )> = Vec::with_capacity(ids.len());
        for id in &ids {
            let Some(obj) = state.game.objects.get(*id) else {
                continue;
            };
            if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
                continue;
            }
            let r: &Robot = unsafe { &*(obj as *const dyn MapStatic as *const Robot) };
            // C++ stores HP as a float ranging 0..max — UI displays
            // integers (CInterface.cpp:1503: `Float2Int(lives)`).
            let hp = r.hit_point.round().max(0.0) as i32;
            let max_hp = r.hit_point_max.round().max(0.0) as i32;
            snapshot.push((*id, r.config, r.name.clone(), hp, max_hp));
        }
        if !snapshot.is_empty() {
            // Bake the medium icons (group cells) ahead of time. Each
            // unique config maps to a single 64×64 atlas key.
            let format = state.gfx.config.format;
            let mut group: Vec<RobotEntry> = Vec::with_capacity(snapshot.len());
            if let Some(robots) = state.terrain.robots() {
                for (_id, cfg, _name, _hp, _max) in &snapshot {
                    let key = state.robot_icons.ensure(
                        &state.gfx.device,
                        &state.gfx.queue,
                        format,
                        robots,
                        &mut state.iface_renderer,
                        cfg,
                    );
                    group.push(RobotEntry { atlas_key: key });
                }
            } else {
                for _ in &snapshot {
                    group.push(RobotEntry { atlas_key: None });
                }
            }
            // Selected index: position of `active_object` within the
            // snapshot. `m_CurrSelNum` in C++ — the C++ panel uses it
            // to draw the ramka over the active slot and to pick the
            // robot whose big-portrait + name + HP appear on the left
            // of the panel.
            let selected_index = primary
                .and_then(|p| snapshot.iter().position(|(id, _, _, _, _)| *id == p))
                .unwrap_or(0);
            let (_id, cur_cfg, cur_name, cur_hp, cur_max) = &snapshot[selected_index];
            // Bake / look up the big portrait for the active robot.
            let personal_atlas_key = state.terrain.robots().and_then(|robots| {
                state.robot_icons.ensure_big(
                    &state.gfx.device,
                    &state.gfx.queue,
                    format,
                    robots,
                    &mut state.iface_renderer,
                    cur_cfg,
                )
            });
            robot_panel = Some(RobotPanelCtx {
                group,
                selected_index,
                personal_atlas_key,
                robot_name: cur_name.clone(),
                robot_hp: *cur_hp,
                robot_max_hp: *cur_max,
            });
        }
    }

    // An armed order only makes sense while robots are selected —
    // selection loss (death / deselect) backs out of ordering mode.
    if state.iface_list.pre_order.is_some() && !matches!(curr_sel, CurrSel::RobotsSelected) {
        state.iface_list.pre_order = None;
    }
    let ctx = MainVisibilityCtx {
        curr_sel,
        building_kind: kind,
        building_stack_empty: stack_empty,
        building_stack_items: stack_items,
        building_turrets_max: turrets_max,
        constructor_active,
        ordering: state.iface_list.pre_order.is_some(),
        turret_build_active,
        turret_kind_committed,
        turret_disabled,
        buca_disabled,
        installed_turret_kinds,
        building_stack_turret_kinds: stack_kinds,
        building_stack_robot_atlas_keys: stack_robot_keys,
        robot_panel: robot_panel.clone(),
    };
    if let Some(p) = state.iface_list.panel_mut("Main") {
        p.refresh_main_visibility(&ctx);
        // Push per-kind captions + HP readout into the dynamic labels
        // — ports CInterface.cpp:1369-1499 (name / bopis / bresg / lives).
        if let Some(k) = kind {
            let strings = crate::matrix_game::config::global_strings();
            p.apply_main_building_text(
                k,
                hit_point,
                hit_point_max,
                income_per_minute,
                &strings.buildings,
            );
        }
        // Robot-selection captions — port of CInterface.cpp:1369-1404.
        if let Some(rp) = robot_panel.as_ref() {
            p.apply_main_robot_text(&rp.robot_name, rp.robot_hp, rp.robot_max_hp);
        }
    }
    // Top-of-screen permanent HUD — resource pools + robot count. Port
    // of the CInterface::AddHintReplacements-driven `thz/enhz1/elhz/
    // phz/rvhz` substitution at CInterface.cpp:4444-4462, applied to
    // the Top panel's always-visible value labels.
    {
        use crate::matrix_game::config::Resource;
        let side = &state.game.player_side;
        let titan = side.get_resource_amount(Resource::Titan);
        let elect = side.get_resource_amount(Resource::Electronics);
        let energy = side.get_resource_amount(Resource::Energy);
        let plasma = side.get_resource_amount(Resource::Plasma);
        let robots = side.get_side_robots();
        let max_robots = state.game.compute_max_side_robots(side.id);
        if let Some(p) = state.iface_list.panel_mut("Top") {
            p.apply_top_hud_text(titan, elect, energy, plasma, robots, max_robots);
        }
    }
    // Snapshot the live preset so we can feed it to `apply_constructor_to_pylons`
    // without holding a borrow across the panel_mut call.
    let live_cfg = state.game.player_side.builder.as_ref().map(|b| *b.cfg());
    let hist_prev = state.iface_list.history.is_prev();
    let hist_next = state.iface_list.history.is_next();
    let build_count = state.iface_list.r_count_control.counter();
    let counter_state = state.iface_list.r_count_control.clone();
    let counter_ctx = build_counter_ctx(state);
    let (focused_price, summ_price, armor_common, armor_extra, build_enabled, focused_target) =
        state
            .game
            .player_side
            .builder
            .as_ref()
            .map(|b| {
                // Armor weapon-slot caps from the live preview's matrix
                // (port of g_MatrixMap->m_RobotWeaponMatrix[hull-1] reads).
                let common = b.live_armor.max_common_weapon_cnt;
                let extra = b.live_armor.max_extra_weapon_cnt;
                // Port of `g_IFaceList->CreateSummPrice(m_Counter)`
                // (CInterface.cpp:3220 / CCounter.cpp:42-50). The C++
                // multiplies per-unit price by the counter at panel-refresh
                // time; we do the same here.
                let mult = build_count.max(1);
                let mut total_cost = b.construction_price();
                for r in 0..total_cost.resources.len() {
                    total_cost.resources[r] = total_cost.resources[r].saturating_mul(mult);
                }
                // Build button enabled: stack not full + side can afford
                // the live preview cost. Ports CInterface.cpp:1850-1867.
                let mut enough = true;
                for r in crate::matrix_game::config::Resource::ALL {
                    if state.game.player_side.get_resource_amount(r)
                        < total_cost.resources[r as usize]
                    {
                        enough = false;
                        break;
                    }
                }
                let buildable_base = matches!(kind, Some(BuildingType::Base));
                let under_cap = counter_ctx.side_robots + counter_ctx.robots_in_stack
                    < counter_ctx.max_side_robots;
                // Port of `CConstructor::RemoteBuild`'s player-side guard
                // at CConstructor.cpp:225-227 — enemy / neutral bases can
                // be selected for inspection but the build button must not
                // fire on them.
                enough = enough && buildable_base && under_cap && active_is_player_owned;
                // Decode the focused pylon → (type, kind) so the visibility
                // refresh knows which `head{N}_st` / `iw{N}text` etc. to
                // expose.
                let focused = b
                    .focused_element
                    .as_deref()
                    .and_then(|n| b.focus_target_for(n));
                (b.focused_price, total_cost, common, extra, enough, focused)
            })
            .unwrap_or((
                None,
                crate::matrix_game::interface::constructor::UnitPrice::zero(),
                0,
                0,
                true,
                None,
            ));
    // Robot-limit warning gate — port of CInterface.cpp:1830-1831
    // (`ps->GetRobotsCnt()+ps->GetRobotsInStack() >= ps->GetMaxSideRobots()`).
    let robot_limit_reached =
        counter_ctx.side_robots + counter_ctx.robots_in_stack >= counter_ctx.max_side_robots;
    // Port of the C++ `m_VisibleAlpha = IS_VISIBLEA` gate at
    // CInterface.cpp:1797 — the Base panel is *visible* whenever a
    // base/building is the active selection; whether individual
    // constructor elements inside render is gated on
    // `constructor_active` via the per-element refresh below.
    let base_panel_visible =
        matches!(curr_sel, CurrSel::BaseSelected | CurrSel::BuildingSelected) && kind.is_some();
    if let Some(p) = state.iface_list.panel_mut("Base") {
        let was_visible = p.visible;
        p.visible = base_panel_visible;
        p.refresh_base_visibility_v2(
            crate::matrix_game::interface::interface::BaseVisibilityCtx {
                constructor_active,
                build_count,
                focused_price: focused_price.as_ref(),
                summ_price: &summ_price,
                armor_common_slots: armor_common,
                armor_extra_slots: armor_extra,
                history_has_prev: hist_prev,
                history_has_next: hist_next,
                counter_up_enabled: counter_state.button_up_enabled,
                counter_down_enabled: counter_state.button_down_enabled,
                build_enabled,
                robot_limit_reached,
                focused_target,
            },
        );
        if constructor_active {
            if let Some(cfg) = live_cfg.as_ref() {
                p.apply_constructor_to_pylons(cfg);
            }
            // Push the focused-component label / description into the
            // `it_label1` / `it_label2` statics + the live preview's
            // structure / damage / robot-name into their respective
            // statics. Matches CInterface.cpp:1822-1827 (rcname),
            // 1869-1884 (it_label1/2), and 2330-2354 (struct/damage).
            if let Some(b) = state.game.player_side.builder.as_ref() {
                let structure = b.construction_structure();
                // C++ damage path zeroes the value when there are no
                // weapons (CInterface.cpp:2347-2350); construction_damage
                // already sums only populated weapon slots, so the same
                // outcome falls out.
                let damage = b.construction_damage();
                let robot_name = b.construction_name();
                p.apply_focused_text(
                    &b.focused_text.label,
                    &b.focused_text.description,
                    structure,
                    damage,
                    &robot_name,
                    &summ_price,
                    b.focused_price.as_ref(),
                );
            }
        }
        // Per-frame port of CIFaceList::CreateSummPrice / CreateItemPrice
        // (CInterface.cpp:3146-3297). Wipes the prior frame's dynamic
        // price icons + price-text labels, then re-emits them for every
        // non-zero resource on the current focused / summary price.
        let focused_price_owned = state
            .game
            .player_side
            .builder
            .as_ref()
            .and_then(|b| b.focused_price);
        let side_res = state.game.player_side.resources;
        p.apply_constructor_prices_res(
            constructor_active,
            focused_price_owned.as_ref(),
            &summ_price,
            &side_res,
        );
        if was_visible != p.visible {
            let n_vis = p.elements.iter().filter(|e| e.visible()).count();
            let n_total = p.elements.len();
            let visible_names: Vec<&str> = p
                .elements
                .iter()
                .filter(|e| e.visible())
                .map(|e| e.name.as_str())
                .take(20)
                .collect();
            log::debug!(
                "iface: Base panel visibility → {} (constructor_active={}, {}/{} elements visible, first: {:?})",
                p.visible,
                constructor_active,
                n_vis,
                n_total,
                visible_names,
            );
        }
    }
}

/// Keep the selection-ring effects in sync with
/// `player_side.selected`. Called once per frame after the logic
/// takt advances. Ports the effect-lifecycle hooks
/// `CMatrixBuilding::Select` / `UnSelect` +
/// `CMatrixEffectSelection::SetPos` (MatrixObjectBuilding.cpp:1460-1525,
/// MatrixEffectSelection.hpp:61-65) as a single reconciler, driven
/// off the whole multi-select set so every selected object gets its
/// own ring.
fn sync_selection_ring(state: &mut AppState, step_ms: f32) {
    use crate::matrix_game::map_static::{ObjectId, ObjectType};
    use crate::matrix_game::object_building::{selection_placement, Building, BuildingType};

    let color = side_selection_color(state.game.player_side.id);
    let mut desired: Vec<(ObjectId, glam::Vec3, f32, u32)> = Vec::new();
    let ids: Vec<ObjectId> = state.game.player_side.selected.clone();
    for id in ids {
        if !state.game.objects.is_valid(id) {
            continue;
        }
        let placement = state.game.objects.get(id).map(|obj| {
            let ot = obj.core().obj_type;
            if ot == ObjectType::Building {
                let b: &Building = unsafe {
                    &*(obj as *const dyn crate::matrix_game::map_static::MapStatic
                        as *const Building)
                };
                let (c, r) = selection_placement(b.pos, b.build_z, b.angle, b.kind);
                let z = state.map.get_z(c.x, c.y).max(c.z);
                let _ = BuildingType::Base; // keep import live
                (glam::Vec3::new(c.x, c.y, z), r)
            } else {
                (obj.core().geo_center, obj.core().radius.max(12.0))
            }
        });
        if let Some((c, r)) = placement {
            desired.push((id, c, r, color));
        }
    }
    state.selection_ring.set_selections(&desired);
    state
        .selection_ring
        .takt(step_ms, |x, y| state.map.get_z(x, y));
}

/// Pick a highlight color for the selection ring by side. The C++
/// defaults every selection to `SEL_COLOR_DEFAULT` (green) regardless
/// of side — the enemy-ring color change is a runtime tint handled
/// elsewhere (the `SetColor(SEL_COLOR_TMP)` calls at MatrixSide.cpp:
/// 1750-1753 are commented out in the shipping code). We keep a
/// per-side tint as a placeholder for when the full side integration
/// lands; for the player case we match the original exactly.
fn side_selection_color(side: i32) -> u32 {
    use crate::matrix_game::effects::selection::SEL_COLOR_DEFAULT;
    match side {
        1 => SEL_COLOR_DEFAULT, // player — green
        2 => 0xFFFF_3333,       // enemy red (placeholder)
        3 => 0xFFFF_AA00,       // enemy orange (placeholder)
        4 => 0xFFFF_FF33,       // enemy yellow (placeholder)
        _ => 0xFFFF_FFFF,       // neutral / default — white
    }
}

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
        .expect("robots.dat is required; place it next to robots.pkg (e.g. ../Data/robots.dat)");
    log::info!("loading matrix data: {}", dat_path);
    let dat_bytes = std::fs::read(dat_path).expect("failed to read robots.dat");
    let matrix_data = Storage::from_bytes(&dat_bytes).expect("failed to parse robots.dat CStorage");

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
    // Cache-bust: append the same `?v=N` that index.html uses for
    // the JS import so the browser refetches when the UI textures
    // in the bundle change.
    let bundle_url = if bundle_url.contains('?') {
        bundle_url
    } else {
        // Bump this whenever `pack_bundle.rs` changes the set of
        // packed keys, so the browser refetches instead of serving
        // a stale cached response.
        format!("{bundle_url}?bv=8")
    };
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
    let dat_bytes = bundle
        .read_file("robots.dat")
        .expect("bundle must contain robots.dat; rebuild the bundle with examples/pack_bundle.rs");
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

/// Construct the "Pause" hint widget — port of MatrixLogic.cpp:2611:
///
///     m_PauseHint = CMatrixHint::Build(TEMPLATE_PAUSE);
///     m_PauseHint->Show(14, 62);
///
/// The hint goes through the standard hint render path so it picks up
/// the engine's chrome (border, background) and font choices baked
/// into the `Pause` template in `da/Templates`. Returns `None` when
/// the template is missing — older / stripped data files that don't
/// ship `Pause` just won't show a label, only the dim tint.
fn build_pause_hint(state: &mut AppState) -> Option<crate::matrix_game::interface::Hint> {
    use crate::matrix_game::interface::hint::{build_hint, ChromeBorder};

    let raw = state.iface_list.hint_templates.get("Pause")?;
    let raw = raw.to_string();
    let chrome = &state.iface_list.hint_chrome;
    let border_default = ChromeBorder::default();
    let border = chrome.borders.get(&0).unwrap_or(&border_default);

    let screen_w = state.gfx.config.width as f32;
    let screen_h = state.gfx.config.height as f32;
    let scale = (screen_h / crate::matrix_game::interface::interface::DESIGN_H).max(0.1);

    // Snapshot bitmap sizes for the hint layout — same pattern as the
    // hover-hint update path (see the `iface_list.update` call
    // elsewhere in this file).
    let mut sizes: std::collections::HashMap<String, (i32, i32)> = std::collections::HashMap::new();
    for bmp in chrome.bitmaps.values() {
        if let Some((w, h)) = state.iface_renderer.atlas_size(&bmp.path) {
            sizes.insert(bmp.path.clone(), (w as i32, h as i32));
        }
    }
    let sizer = |path: &str| sizes.get(path).copied();

    let replacer = state.iface_list.hint_replacer.clone();
    let atlas = state.iface_renderer.glyph_atlas_mut();
    let (mut parts, otstup, total_w, total_h, border_id, _wrap) =
        build_hint(&raw, border, &chrome.bitmaps, &replacer, atlas, sizer)?;
    if parts.is_empty() {
        return None;
    }

    // Center text horizontally within the hint's content area. The
    // original C++ Rangers rasterizer centers text inside its bounding
    // box by default (`alignx = 1` at MatrixHint.cpp:590); our
    // hint renderer renders `HintPart::Text` left-aligned at `x`, and
    // when the template uses `_WIDTH:N` the part's `w` may be wider
    // than the actual glyph run. Re-measure and shift each text part
    // so its glyph run sits centered within the hint's content area.
    let content_w = total_w - otstup[0] - otstup[2];
    let atlas = state.iface_renderer.glyph_atlas_mut();
    for part in parts.iter_mut() {
        if let crate::matrix_game::interface::hint::HintPart::Text { x, text, font, .. } = part {
            // Use the longest line for measurement when text is
            // multi-line (e.g. `<br>` separator). That preserves the
            // visual block centering the C++ produces.
            let measured: i32 = text
                .split('\n')
                .map(|line| atlas.measure(font, line) as i32)
                .max()
                .unwrap_or(0);
            *x = otstup[0] + (content_w - measured) / 2;
        }
    }

    // C++ position from MatrixLogic.cpp:2612 — `Show(14, 62)` is in
    // design-space pixels (1024×768). Scale to current screen.
    let screen_x = 14.0 * scale;
    let screen_y = 62.0 * scale;
    // Clamp so it doesn't fall off-screen on tiny windows.
    let screen_x = screen_x
        .min(screen_w - total_w as f32 * scale - 2.0)
        .max(0.0);
    let screen_y = screen_y
        .min(screen_h - total_h as f32 * scale - 2.0)
        .max(0.0);

    Some(crate::matrix_game::interface::Hint {
        parts,
        total_w,
        total_h,
        border_id,
        otstup,
        screen_x,
        screen_y,
    })
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
