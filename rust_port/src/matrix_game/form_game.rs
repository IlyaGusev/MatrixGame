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
    /// Whether the current LMB press was consumed by UI / minimap
    /// (if so, release mustn't issue a world click or marquee).
    lmb_consumed_by_ui: bool,
    /// Last rect the marquee rendered / was releasing. Needed across
    /// frames because the DIP fade keeps drawing for 50ms after the
    /// user lifts LMB and has to resample alpha off the same coords.
    marquee_last_rect: Option<[f32; 4]>,
    /// Map of currently-tracked arena robots → their visibility
    /// point-light id. Rebuilt per frame in `sync_robot_lights` so
    /// freshly-spawned robots light up and despawned ones go dark.
    /// Substitutes for the full mesh / billboard renderer (the
    /// CMatrixRobotAI draw path) until that lands.
    robot_lights: std::collections::HashMap<
        crate::matrix_game::map_static::ObjectId,
        crate::matrix_game::effects::point_light::PointLightId,
    >,
    /// Selection-ring renderer — ports `CMatrixEffectSelection`
    /// (MatrixEffectSelection.cpp). Drawn over the terrain after the
    /// object pass; green ring on the ground around the selected
    /// object.
    selection_ring: crate::matrix_game::effects::selection::SelectionRingRenderer,
    /// Screen-space marquee rectangle drawn while the user holds
    /// left-button and drags on empty terrain. Rebuilt each frame
    /// from `lmb_anchor` + current cursor; hidden otherwise.
    marquee: crate::matrix_game::effects::marquee::MarqueeRenderer,
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
            let marquee = crate::matrix_game::effects::marquee::MarqueeRenderer::new(
                &gfx.device,
                &gfx.config,
            );

            let iface_list =
                crate::matrix_game::interface::IFaceList::load_default_panels(&matrix_data);
            log::info!("iface: loaded {} panels", iface_list.panels.len());
            let mut iface_renderer =
                crate::matrix_game::interface::InterfaceRenderer::new(&gfx.device, &gfx.config);
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
            let mut progress_bars = crate::matrix_game::progress_bar::ProgressBarRenderer::new(
                &gfx.device,
                &gfx.config,
            );
            progress_bars.load_atlas(&gfx.device, &gfx.queue, &read);

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
                lmb_consumed_by_ui: false,
                marquee_last_rect: None,
                robot_lights: std::collections::HashMap::new(),
                selection_ring,
                marquee,
                iface_list,
                iface_renderer,
                progress_bars,
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
                    camera.set_aspect(gfx.config.width as f32, gfx.config.height as f32);
                }

                let mut game = MapLogic::new();
                game.load_config(&matrix_data);
                let building_ids = game.spawn_buildings(&map);
                log::info!("world: spawned {} buildings", building_ids.len());
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
                let marquee = crate::matrix_game::effects::marquee::MarqueeRenderer::new(
                    &gfx.device,
                    &gfx.config,
                );

                let iface_list =
                    crate::matrix_game::interface::IFaceList::load_default_panels(&matrix_data);
                log::info!("iface: loaded {} panels", iface_list.panels.len());
                let mut iface_renderer =
                    crate::matrix_game::interface::InterfaceRenderer::new(&gfx.device, &gfx.config);
                let read = |p: &str| -> Option<Vec<u8>> { bundle.read_file(p).map(|b| b.to_vec()) };
                iface_renderer.preload_for_panels(
                    &gfx.device,
                    &gfx.queue,
                    &read,
                    iface_list.panels.iter(),
                );
                let mut progress_bars = crate::matrix_game::progress_bar::ProgressBarRenderer::new(
                    &gfx.device,
                    &gfx.config,
                );
                progress_bars.load_atlas(&gfx.device, &gfx.queue, &read);

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
                    lmb_consumed_by_ui: false,
                    marquee_last_rect: None,
                    robot_lights: std::collections::HashMap::new(),
                    selection_ring,
                    marquee,
                    iface_list,
                    iface_renderer,
                    progress_bars,
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
            // Right-click down also issues move orders to the selected
            // robots (C++: `CMatrixSideUnit::OnRButtonDown`, dispatched
            // alongside the camera-rotate state toggle).
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                use winit::event::{ElementState, MouseButton};
                let [cx, cy] = state.cursor;
                let w = state.gfx.config.width as f32;
                let h = state.gfx.config.height as f32;
                if matches!(button, MouseButton::Middle | MouseButton::Right) {
                    let pressed = btn_state == ElementState::Pressed;
                    state.camera.on_rotate_button(pressed, cx, cy);
                    // Right-click down → issue move orders to all own
                    // robots currently selected. MatrixFormGame.cpp:758
                    // fires the order unconditionally at RBDOWN; camera
                    // rotate only shows visibly if the user actually
                    // drags, so both coexist.
                    if button == MouseButton::Right
                        && pressed
                        && !state.game.player_side.selected.is_empty()
                    {
                        let n =
                            state
                                .game
                                .order_move_to_at(&state.camera, cx, cy, w, h, &state.map);
                        if n > 0 {
                            log::info!("move order: issued to {} robot(s)", n);
                        }
                    }
                } else if button == MouseButton::Left {
                    use crate::matrix_game::minimap::MinimapClick;
                    match btn_state {
                        ElementState::Pressed => {
                            // UI first dibs (MatrixFormGame.cpp:748-755).
                            if state.iface_list.on_mouse_down(cx, cy, w, h) {
                                state.minimap_dragging = false;
                                state.lmb_anchor = None;
                                state.lmb_consumed_by_ui = true;
                            } else {
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
                                        state.lmb_anchor = Some([cx, cy]);
                                        state.lmb_consumed_by_ui = false;
                                    }
                                }
                            }
                        }
                        ElementState::Released => {
                            state.minimap_dragging = false;
                            if let Some(click) = state.iface_list.on_mouse_up(cx, cy, w, h) {
                                log::info!("iface: clicked {:?}", click);
                                dispatch_ui_click(state, &click);
                            } else if let Some([ax, ay]) = state.lmb_anchor.take() {
                                if !state.lmb_consumed_by_ui {
                                    // Drag distance — anything ≤ 4 px is a
                                    // click, otherwise a marquee rect.
                                    let dx = (cx - ax).abs();
                                    let dy = (cy - ay).abs();
                                    const DRAG_PX: f32 = 4.0;
                                    if dx <= DRAG_PX && dy <= DRAG_PX {
                                        let hit = state.game.click_at_screen(
                                            &state.camera,
                                            cx,
                                            cy,
                                            w,
                                            h,
                                            state.shift_down,
                                        );
                                        match hit {
                                            Some(id) => log::info!(
                                                "selection: hit object {:?}, curr_sel={:?}, selected={}",
                                                id, state.game.player_side.curr_sel,
                                                state.game.player_side.selected.len(),
                                            ),
                                            None => log::info!(
                                                "selection: cleared (selected={})",
                                                state.game.player_side.selected.len(),
                                            ),
                                        }
                                    } else {
                                        let rmin = [ax.min(cx), ay.min(cy)];
                                        let rmax = [ax.max(cx), ay.max(cy)];
                                        let n = state.game.marquee_select(
                                            &state.camera,
                                            rmin,
                                            rmax,
                                            w,
                                            h,
                                            state.shift_down,
                                        );
                                        log::info!("marquee: selected {} robot(s)", n,);
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
                    if !state.lmb_consumed_by_ui {
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
                    state.iface_list.on_mouse_move(cx, cy, w, h);
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
                let dt = (now - state.last_time) as f32;
                state.last_time = now;
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
                    state.game.takt(step_ms);
                    state.game.graphic_takt(step_ms);
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

                // Per-frame: reconcile point lights for each live
                // robot so the build-factory spawns are visible even
                // without a robot renderer. Stand-in for the C++
                // `CMatrixRobotAI::Draw` path until mesh rendering lands.
                sync_robot_lights(state);

                // Per-frame interface visibility dispatch — ports the
                // `CInterface::LogicTakt` branch at
                // CInterface.cpp:1214-1635. Only `if/Main` for now.
                refresh_interface_visibility(state);

                // Queue the build-stack progress bar on top of the
                // `prog` element. Ports
                // `m_BS.m_PB.Modify(m_Timer / UNIT_ROBOT) +
                // CreateClone(PBC_CLONE1, x, y, 87)` at
                // MatrixObjectBuilding.cpp:1681-1689.
                refresh_progress_bars(state);

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

                // Rebuild chassis instance buffers from the live arena
                // so newly-spawned robots show up (stand-in for
                // `CMatrixRobotAI::RNeed`'s per-robot matrix update —
                // MatrixObjectRobot.cpp:359-480).
                state.terrain.sync_robots(
                    &state.gfx.queue,
                    &mut state.game.objects,
                    &state.map,
                    &state.point_lights,
                    step_ms,
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
                            state.selection_ring.render(&mut pass);
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
                            );
                        }
                        // Interface (HUD) pass — draws on top of
                        // world + minimap. Ports
                        // `CIFaceList::Render` iteration.
                        {
                            let panels: Vec<&crate::matrix_game::interface::CInterface> =
                                state.iface_list.panels.iter().collect();
                            state.iface_renderer.upload(
                                &state.gfx.queue,
                                &panels,
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
                        // Progress-bar overlay pass — on top of the UI.
                        {
                            state.progress_bars.upload(
                                &state.gfx.device,
                                &state.gfx.queue,
                                state.gfx.config.width as f32,
                                state.gfx.config.height as f32,
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

/// Keep per-robot visibility point lights in sync with the arena.
/// Every frame: spawn a light for each new `ObjectType::RobotAi`,
/// drop lights whose target id is no longer valid. Per-side color
/// so enemy and friendly spawns read apart at a glance.
fn sync_robot_lights(state: &mut AppState) {
    use crate::matrix_game::map_static::ObjectType;

    // Build a set of current live robot ids.
    let mut live_robots: Vec<crate::matrix_game::map_static::ObjectId> = Vec::new();
    for id in state.game.objects.iter_live() {
        if let Some(obj) = state.game.objects.get(id) {
            if matches!(obj.core().obj_type, ObjectType::RobotAi) {
                live_robots.push(id);
            }
        }
    }

    // Add lights for new robots + update positions for existing ones
    // so the light follows the robot rising out of the silo.
    for id in &live_robots {
        let Some(obj) = state.game.objects.get(*id) else {
            continue;
        };
        let pos = obj.core().geo_center;
        let lit_pos = [pos.x, pos.y, pos.z + 6.0];
        if let Some(light_id) = state.robot_lights.get(id).copied() {
            state.point_lights.set_pos(&state.map, light_id, lit_pos);
        } else {
            // Bright yellow for player, red for enemies.
            let color = if obj.side() == crate::matrix_game::common::PLAYER_SIDE {
                0xFFFF66
            } else {
                0xFF3333
            };
            let light_id = state
                .point_lights
                .add_light(&state.map, lit_pos, 25.0, color);
            state.robot_lights.insert(*id, light_id);
        }
    }

    // Remove lights for dead robots (tombstone-aware via `is_valid`).
    let dead: Vec<_> = state
        .robot_lights
        .iter()
        .filter_map(|(id, light_id)| {
            if state.game.objects.is_valid(*id) {
                None
            } else {
                Some((*id, *light_id))
            }
        })
        .collect();
    for (id, light_id) in dead {
        state.point_lights.remove_light(&state.map, light_id);
        state.robot_lights.remove(&id);
    }
}

/// Port of `CMatrixSideUnit::PlayerAction` + the follow-on
/// `CBuildStack::AddItem` call. Dispatches the button identified by
/// its `Name` to the right game-state change. Currently handles
/// `buro` (build robot) → push a default-chassis robot onto the
/// selected base's build stack. The C++ opens the full
/// `m_ConstructPanel` for chassis/armor/weapon selection first; the
/// constructor UI isn't ported, so we skip straight to AddItem with
/// a default chassis.
fn dispatch_ui_click(state: &mut AppState, click: &crate::matrix_game::interface::Click) {
    use crate::matrix_game::interface::Click;
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::{Building, BuildingType};
    use crate::matrix_game::robot::ChassisKind;
    use crate::matrix_game::side::CurrSel;

    match click {
        Click::Button(name) if name == "buro" => {
            if state.game.player_side.curr_sel != CurrSel::BaseSelected {
                log::info!("buro: no base selected, ignoring");
                return;
            }
            let Some(id) = state.game.active_object() else {
                return;
            };
            // Downcast the active MapStatic to Building and queue a
            // robot. `ChassisKind::Track` is a reasonable default
            // (the C++ defaults vary by constructor state; Track is
            // the cheapest one).
            let Some(obj) = state.game.objects.get_mut(id) else {
                return;
            };
            if !matches!(obj.core().obj_type, ObjectType::Building) {
                return;
            }
            let b: &mut Building = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
            if b.kind != BuildingType::Base {
                log::info!("buro: can only build from a base, got {:?}", b.kind);
                return;
            }
            if b.queue_robot(ChassisKind::Track) {
                log::info!(
                    "buro: queued robot on base at ({:.0},{:.0}); stack now has {} items",
                    b.pos.x,
                    b.pos.y,
                    b.build_stack.items(),
                );
            } else {
                log::info!("buro: stack full, click rejected");
            }
        }
        _ => {}
    }
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

    // Only fires for a selected building with queued items — matches
    // the C++ guard at MatrixObjectBuilding.cpp:1685-1689.
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
    if b.build_stack.is_empty() {
        return;
    }

    let w = state.gfx.config.width as f32;
    let h = state.gfx.config.height as f32;
    let Some(main) = state.iface_list.panel("Main") else {
        log::warn!("progress: no Main panel");
        return;
    };

    // Port of MatrixObjectBuilding.cpp:1676-1689:
    //   float x = g_IFaceList->GetMainX() + 283;
    //   float y = g_IFaceList->GetMainY() + 71;
    //   ...
    //   m_PB.CreateClone(PBC_CLONE1, x, y, 87);
    //
    // `GetMainX()/GetMainY()` return the IF_MAIN panel's resolved
    // top-left in screen pixels; the +283/+71 offsets and 87-wide
    // clone are design-space pixels that the C++ uses at its fixed
    // 1024×768 resolution. We scale them by `screen_h / 768` the
    // same way CInterface scales its own panels.
    const PB_OFFSET_X: f32 = 283.0;
    const PB_OFFSET_Y: f32 = 71.0;
    const PB_WIDTH: f32 = 87.0;
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
    let rect = [
        panel_x + PB_OFFSET_X * scale,
        panel_y + PB_OFFSET_Y * scale,
        PB_WIDTH * scale,
        bar_h_design * scale,
    ];

    // One-shot log so we can verify the rect + fill on first build.
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::info!(
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
    // Pull building context when the selection is a Building.
    let (kind, stack_empty, stack_items, turrets_max) = match curr_sel {
        CurrSel::BaseSelected | CurrSel::BuildingSelected => {
            let active = state.game.active_object();
            active
                .and_then(|id| state.game.objects.get(id))
                .filter(|o| matches!(o.core().obj_type, ObjectType::Building))
                .map(|o| {
                    let b: &Building = unsafe { &*(o as *const dyn MapStatic as *const Building) };
                    let n = b.build_stack.items() as i32;
                    (Some(b.kind), n == 0, n, b.turrets_max)
                })
                .unwrap_or((None::<BuildingType>, true, 0, 0))
        }
        _ => (None, true, 0, 0),
    };

    let ctx = MainVisibilityCtx {
        curr_sel,
        building_kind: kind,
        building_stack_empty: stack_empty,
        building_stack_items: stack_items,
        building_turrets_max: turrets_max,
    };
    if let Some(p) = state.iface_list.panel_mut("Main") {
        p.refresh_main_visibility(&ctx);
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
        format!("{bundle_url}?bv=3")
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
