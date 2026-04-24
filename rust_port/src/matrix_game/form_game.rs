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
            let marquee = crate::matrix_game::multi_selection::MarqueeRenderer::new(
                &gfx.device,
                &gfx.config,
            );
            let move_to =
                crate::matrix_game::effects::move_to::MoveToRenderer::new(&gfx.device, &gfx.config);

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
                move_to,
                iface_list,
                iface_renderer,
                progress_bars,
                builder_preview:
                    crate::matrix_game::interface::constructor::BuilderPreview::new(),
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
                let mut iface_renderer =
                    crate::matrix_game::interface::InterfaceRenderer::new(&gfx.device, &gfx.config);
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
                    move_to,
                    iface_list,
                    iface_renderer,
                    progress_bars,
                    builder_preview:
                        crate::matrix_game::interface::constructor::BuilderPreview::new(),
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
                    // RMB → UI first (CIFaceButton::OnMouseRBDown opens
                    // the constructor popup menu when a pylon catches
                    // the press). Camera-rotate + move-orders only run
                    // if no UI element claims the event.
                    let ui_consumed_rmb = if button == MouseButton::Right {
                        match btn_state {
                            ElementState::Pressed => {
                                state.iface_list.on_mouse_right_down(cx, cy, w, h)
                            }
                            ElementState::Released => state
                                .iface_list
                                .on_mouse_right_up(cx, cy, w, h)
                                .map(|click| {
                                    log::info!("iface: right-clicked {:?}", click);
                                    dispatch_ui_right_click(state, &click);
                                })
                                .is_some(),
                        }
                    } else {
                        false
                    };
                    if !ui_consumed_rmb {
                        state.camera.on_rotate_button(pressed, cx, cy);
                        if button == MouseButton::Right
                            && pressed
                            && !state.game.player_side.selected.is_empty()
                        {
                            let slots = state.game.order_move_to_at(
                                &state.camera,
                                cx,
                                cy,
                                w,
                                h,
                                &state.map,
                            );
                            if !slots.is_empty() {
                                log::info!("move order: issued to {} robot(s)", slots.len());
                                for (wx, wy) in slots {
                                    let wz = state.map.get_z(wx, wy);
                                    state.move_to.spawn(glam::Vec3::new(wx, wy, wz));
                                }
                            }
                        }
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
                                }
                                state.minimap_dragging = false;
                                state.lmb_anchor = None;
                                state.lmb_consumed_by_ui = true;
                            } else {
                                if let Some(cfg) = state.iface_list.popup_restore_pending.take() {
                                    if let Some(b) = state.game.player_side.builder.as_mut() {
                                        b.apply_config(cfg);
                                    }
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
                            } else if state.iface_list.turret_build.is_active() {
                                // Turret placement click — land the
                                // turret on the parent base if the
                                // click landed on one of its slots.
                                // Ports MatrixFormGame.cpp:1498-1512's
                                // PREORDER_BUILD_TURRET branch.
                                state.lmb_anchor = None;
                                try_place_turret(state, cx, cy, w, h);
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
                                        let n = crate::matrix_game::multi_selection::marquee_select(
                                            &mut state.game,
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
                    let (unfocused, focused) = state.iface_list.on_mouse_move(cx, cy, w, h);
                    if let Some(b) = state.game.player_side.builder.as_mut() {
                        preview_popup_hover(b, state.iface_list.popup.as_mut());
                    }
                    // Route Base-panel focus changes into the
                    // constructor — port of CConstructor.cpp:903-958
                    // (`RemoteFocusElement` / `RemoteUnFocusElement`).
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

                // Per-frame move-order ping animation advance — port
                // of `CMatrixEffectMoveto::Takt` (MatrixEffectMoveTo.cpp:93).
                if state.move_to.is_active() {
                    state.move_to.takt(step_ms as f32);
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

                // Advance the constructor 3D preview turntable + emit
                // a preview draw-ticket while the constructor panel is
                // open. Stand-in for CConstructor.cpp:251-262 +
                // :264-360 (Render).
                tick_builder_preview(state, dt * 1000.0);

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
                            state
                                .move_to
                                .upload(&state.gfx.queue, &state.map, vp, cr, cu, mc);
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
                            // Move-order ping — same color/depth target
                            // as the selection ring (billboards are
                            // additively blended + depth-tested against
                            // terrain so they occlude behind geometry).
                            state.move_to.render(&mut pass);
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
                            state.iface_renderer.upload_with_popup_and_hint(
                                &state.gfx.device,
                                &state.gfx.queue,
                                &panels,
                                state.iface_list.popup.as_ref(),
                                state.iface_list.hint_system.active(),
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
                                log::info!(
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
    use crate::matrix_game::interface::constructor::parse_constructor_button;
    use crate::matrix_game::interface::iface_list::TurretKind;
    use crate::matrix_game::interface::Click;
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::{Building, BuildingType};
    use crate::matrix_game::robot::ChassisKind;
    use crate::matrix_game::config::RobotUnitKind;
    use crate::matrix_game::object_robot::RobotUnitType;
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
                log::info!(
                    "popup commit: type={:?} kind={} pilon={} cfg.chassis={} hull={} weap0={}",
                    ty,
                    kind.0,
                    pilon,
                    b.cfg().chassis.kind.0,
                    b.cfg().hull.unit.kind.0,
                    b.cfg().weapon[0].kind.0,
                );
            }
            state.iface_list.popup = None;
            state.iface_list.popup_restore_pending = None;
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
                    log::info!("basepl: center camera on building at ({:.1},{:.1})", p.x, p.y);
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
                log::info!("buro: no base selected, ignoring");
                return;
            }
            state.iface_list.r_count_control.reset();
            let ctx = build_counter_ctx(state);
            state.iface_list.r_count_control.check_up(ctx);
            if let Some(b) = state.game.player_side.builder.as_mut() {
                b.activate();
            }
            log::info!("buro: opened robot constructor");
            // Silence "unused import" on paths we only need in other
            // arms of this match.
            let _ = (BuildingType::Base, ChassisKind::Track);
            let _: Option<&dyn MapStatic> = None;
            let _ = ObjectType::Building;
            let _: fn(&mut Building) = |_b: &mut Building| {};
            return;
        }
        "buca" => {
            if state.game.player_side.curr_sel != CurrSel::BaseSelected {
                log::info!("buca: no base selected, ignoring");
                return;
            }
            let Some(id) = state.game.active_object() else {
                return;
            };
            state.iface_list.turret_build.begin(TurretKind::Cannon, id);
            log::info!("buca: entered turret-build mode (default Cannon)");
            return;
        }
        "cocan" => {
            // Port of CConstructor.cpp:986-994 — close panel, reset the
            // counter, kill any open popup. The C++ also unpauses the
            // world; the Rust port doesn't pause yet so that step is a
            // no-op until the pause/resume plumbing lands.
            if let Some(b) = state.game.player_side.builder.as_mut() {
                b.deactivate();
                b.reset_construction();
            }
            state.iface_list.r_count_control.reset();
            state.iface_list.popup = None;
            state.iface_list.popup_restore_pending = None;
            state.iface_list.turret_build.cancel();
            log::info!("cocan: constructor closed");
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
                    log::info!(
                        "hisleft: loaded preset cursor={}",
                        state.iface_list.history.cursor
                    );
                }
            }
            return;
        }
        "hisright" => {
            if let Some(cfg) = state.iface_list.history.next() {
                if let Some(b) = state.game.player_side.builder.as_mut() {
                    b.apply_config(cfg);
                    log::info!(
                        "hisright: loaded preset cursor={}",
                        state.iface_list.history.cursor
                    );
                }
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
                log::info!("tur{}: selected kind={:?}", n, kind);
            }
            return;
        }
        _ => {}
    }

    // ── Constructor pylon buttons (LMB) ──────────────────────────
    // The C++ does NOT cycle on left-click — pylons fire only on
    // RBDown (which opens the popup, see dispatch_ui_right_click).
    // Left-click is a no-op so we eat the event here for parity.
    if matches!(
        name,
        "pich" | "pihu" | "pihe" | "pi1" | "pi2" | "pi3" | "pi4" | "pi5"
    ) {
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
            log::info!(
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
        return;
    }

    log::debug!("dispatch_ui_click: unhandled click {:?}", click);
    let _ = RobotUnitType::Chassis;
    let _ = RobotUnitKind::UNKNOWN;
}

/// Dispatch a UI right-click. Currently routes pylon right-clicks to
/// the constructor popup menu (port of `CIFaceButton::OnMouseRBDown`
/// at CIFaceButton.cpp:183-321).
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

    if let Some(mut popup) = popup_for_pylon(name) {
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
        log::info!("popup: opened for pylon {}", name);
        state.iface_list.popup_restore_pending = None;
        state.iface_list.popup = Some(popup);
    }
}

fn preview_popup_hover(
    builder: &mut crate::matrix_game::interface::constructor::RobotBuilder,
    popup: Option<&mut crate::matrix_game::interface::iface_menu::CIFaceMenu>,
) {
    let Some(popup) = popup else {
        return;
    };
    if popup.previewed == popup.hovered {
        return;
    }
    popup.previewed = popup.hovered;
    let Some(idx) = popup.hovered else {
        // Cursor left the popup rows — restore the saved preview
        // and clear the focused label/price so the Base-panel card
        // stops showing stale row-preview text.
        if let Some(saved) = popup.saved_config {
            builder.apply_config(saved);
        }
        builder.clear_focused_card();
        return;
    };
    let Some(item) = popup.items.get(idx).copied() else {
        return;
    };
    let ty = popup.parent.unit_type();
    builder.djeans007(ty, item.kind, popup.parent.pilon());
    // Port of the per-hover `RemoteFocusElement` fire in the C++
    // popup loop (CIFaceList::OnMouseMove + CConstructor.cpp:912-958).
    // Updates the focused label/description/price card on the Base
    // panel so hovering a popup row gives the same readouts as
    // hovering the equivalent template button would.
    builder.set_labels_and_price(ty, item.kind);
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
            let (base_i, fa_i) = state.game.compute_resource_income(side_id, Resource::Energy);
            repl.set("_energy_income", (base_i + fa_i).to_string());
        }
        "elhz" => {
            let (base_i, fa_i) =
                state.game.compute_resource_income(side_id, Resource::Electronics);
            repl.set("_electronics_income", (base_i + fa_i).to_string());
        }
        "phz" => {
            let (base_i, fa_i) = state.game.compute_resource_income(side_id, Resource::Plasma);
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
            let (name_label, range_label) =
                state.iface_list.hint_replacer.turret_label(idx);
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
        _ => {
            // Other hinted elements (call-from-hell, combat-mode, …)
            // require plumbing we haven't ported yet. The base
            // template still renders — unresolved `[keys]` fall
            // through as empty strings per `HintReplacer::get` → `None`.
        }
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

/// Port of the "click during PREORDER_BUILD_TURRET" path
/// (MatrixFormGame.cpp:1498-1512). If the click lands on the parent
/// base's ring, queue the turret + deduct cost; otherwise cancel.
fn try_place_turret(state: &mut AppState, cx: f32, cy: f32, w: f32, h: f32) {
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::Building;
    use crate::matrix_game::config::Resource;

    let Some(parent_id) = state.iface_list.turret_build.parent else {
        state.iface_list.turret_build.cancel();
        return;
    };
    let Some(kind) = state.iface_list.turret_build.kind else {
        state.iface_list.turret_build.cancel();
        return;
    };
    // Validate the click hit the parent base.
    let hit = {
        let (origin, dir) = state.camera.screen_to_world_ray(cx, cy, w, h);
        state.game.objects.pick_object(
            origin,
            dir,
            crate::matrix_game::common::TRACE_ANYOBJECT,
            None,
        )
    };
    let landed_on_parent = matches!(hit, Some((id, _)) if id == parent_id);
    if !landed_on_parent {
        log::info!("turret: click missed parent base — cancelling placement");
        state.iface_list.turret_build.cancel();
        return;
    }

    // Cost check.
    let turret_cost = crate::matrix_game::config::global()
        .turrets
        .cost_of(kind as i32);
    for r in Resource::ALL {
        if state.game.player_side.get_resource_amount(r) < turret_cost.resources[r as usize] {
            log::info!(
                "turret: insufficient {:?}: need {}, have {}",
                r,
                turret_cost.resources[r as usize],
                state.game.player_side.get_resource_amount(r),
            );
            state.iface_list.turret_build.cancel();
            return;
        }
    }

    // Queue on the parent building + snapshot placement for the
    // cannon spawn below.
    let placement = {
        let Some(obj) = state.game.objects.get_mut(parent_id) else {
            state.iface_list.turret_build.cancel();
            return;
        };
        if !matches!(obj.core().obj_type, ObjectType::Building) {
            state.iface_list.turret_build.cancel();
            return;
        }
        let b: &mut Building = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Building) };
        // Same player-side guard as robot builds (CConstructor.cpp:225-227) —
        // enemy bases can be clicked but not built on.
        if b.side != state.game.player_side.id {
            log::info!(
                "turret: refused — base is side {}, not player side {}",
                b.side,
                state.game.player_side.id
            );
            state.iface_list.turret_build.cancel();
            return;
        }
        if !b.queue_turret(kind as i32) {
            log::info!("turret: all turret slots full on base");
            state.iface_list.turret_build.cancel();
            return;
        }
        let slot = (b.turrets_have - 1).max(0);
        let ang = (b.angle & 3) as f32 * std::f32::consts::FRAC_PI_2;
        // Turret slot offset (4 slots around the base, roughly 40
        // units from centre). Port of the per-slot position the C++
        // reads off the base's `Turret{N}` named matrices on the
        // building mesh (MatrixObjectBuilding.cpp::m_Turrets init).
        // We use a fixed cross pattern until the VO matrix-name
        // lookup lands — positionally close enough for the display.
        let (dx, dy) = match slot {
            0 => (30.0, 30.0),
            1 => (-30.0, 30.0),
            2 => (30.0, -30.0),
            _ => (-30.0, -30.0),
        };
        let (s, c) = ang.sin_cos();
        let off_x = c * dx - s * dy;
        let off_y = s * dx + c * dy;
        (
            glam::Vec2::new(b.pos.x + off_x, b.pos.y + off_y),
            b.build_z + 8.0,
            ang,
            slot,
            b.turrets_max,
            b.turrets_have,
        )
    };

    // Spawn the Cannon object immediately — the build-stack timer
    // still runs for the cost/progress UI, but the C++ mounts the
    // cannon on the building as soon as BeginBuildTurret commits
    // so we match that.
    let cannon = crate::matrix_game::object_cannon::Cannon::new(
        placement.0,
        placement.1,
        placement.2,
        state.game.player_side.id,
        kind as i32,
        parent_id,
        placement.3,
    );
    let id = state.game.objects.spawn(Box::new(cannon));
    state.game.objects.add_lt(id);

    for r in Resource::ALL {
        state
            .game
            .player_side
            .add_resource_amount(r, -turret_cost.resources[r as usize]);
    }
    log::info!(
        "turret: placed {:?} on base (slot {}/{}) as object {:?}",
        kind,
        placement.5,
        placement.4,
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
    use crate::matrix_game::interface::counter::CheckUpCtx;
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::{Building, BuildingType};
    use crate::matrix_game::robot::Robot;
    use crate::matrix_game::config::Resource;
    use crate::matrix_game::interface::constructor::UnitPrice;
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

/// Port of `CConstructor::RemoteBuild` (CConstructor.cpp:223-250).
fn commit_and_queue_robot(state: &mut AppState) {
    use crate::matrix_game::map_static::{MapStatic, ObjectType};
    use crate::matrix_game::object_building::{Building, BuildingType};
    use crate::matrix_game::config::Resource;
    use crate::matrix_game::side::CurrSel;

    if state.game.player_side.curr_sel != CurrSel::BaseSelected {
        log::info!("cobuild: no base selected");
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
        log::info!(
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
    // CConstructor.cpp:244-246 — close the panel.
    if let Some(b) = state.game.player_side.builder.as_mut() {
        b.deactivate();
        b.reset_construction();
    }
    // CConstructor.cpp:247-248 — reset + revalidate the counter.
    state.iface_list.r_count_control.reset();
    let ctx = build_counter_ctx(state);
    state.iface_list.r_count_control.check_up(ctx);
    log::info!(
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
    let (kind, stack_empty, stack_items, turrets_max, hit_point, hit_point_max, active_side) =
        match curr_sel {
            CurrSel::BaseSelected | CurrSel::BuildingSelected => {
                let active = state.game.active_object();
                active
                    .and_then(|id| state.game.objects.get(id))
                    .filter(|o| matches!(o.core().obj_type, ObjectType::Building))
                    .map(|o| {
                        let b: &Building =
                            unsafe { &*(o as *const dyn MapStatic as *const Building) };
                        let n = b.build_stack.items() as i32;
                        (
                            Some(b.kind),
                            n == 0,
                            n,
                            b.turrets_max,
                            b.hit_point,
                            b.hit_point_max,
                            b.side,
                        )
                    })
                    .unwrap_or((None::<BuildingType>, true, 0, 0, 0.0, 0.0, 0))
            }
            _ => (None, true, 0, 0, 0.0, 0.0, 0),
        };
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
    let ctx = MainVisibilityCtx {
        curr_sel,
        building_kind: kind,
        building_stack_empty: stack_empty,
        building_stack_items: stack_items,
        building_turrets_max: turrets_max,
        constructor_active,
        turret_build_active,
    };
    if let Some(p) = state.iface_list.panel_mut("Main") {
        p.refresh_main_visibility(&ctx);
        // Push per-kind captions + HP readout into the dynamic labels
        // — ports CInterface.cpp:1369-1499 (name / bopis / bresg / lives).
        if let Some(k) = kind {
            let strings = crate::matrix_game::config::global_strings();
            p.apply_main_building_text(k, hit_point, hit_point_max, income_per_minute, &strings.buildings);
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
    let (focused_price, summ_price, armor_common, armor_extra, build_enabled) = state
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
                if state.game.player_side.get_resource_amount(r) < total_cost.resources[r as usize]
                {
                    enough = false;
                    break;
                }
            }
            let buildable_base = matches!(kind, Some(BuildingType::Base));
            let under_cap =
                counter_ctx.side_robots + counter_ctx.robots_in_stack < counter_ctx.max_side_robots;
            // Port of `CConstructor::RemoteBuild`'s player-side guard
            // at CConstructor.cpp:225-227 — enemy / neutral bases can
            // be selected for inspection but the build button must not
            // fire on them.
            enough = enough && buildable_base && under_cap && active_is_player_owned;
            (b.focused_price, total_cost, common, extra, enough)
        })
        .unwrap_or((
            None,
            crate::matrix_game::interface::constructor::UnitPrice::zero(),
            0,
            0,
            true,
        ));
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
            },
        );
        if constructor_active {
            if let Some(cfg) = live_cfg.as_ref() {
                p.apply_constructor_to_pylons(cfg);
            }
            // Push the focused-component label / description into the
            // `it_label1` / `it_label2` statics so the text pass picks
            // them up. Matches CInterface.cpp:1869-1884.
            if let Some(b) = state.game.player_side.builder.as_ref() {
                p.apply_focused_text(&b.focused_text.label, &b.focused_text.description);
            }
        }
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
            log::info!(
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
        format!("{bundle_url}?bv=4")
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
