pub mod app;
pub mod assets;
pub mod game;
pub mod platform;
pub mod renderer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("=== MatrixGame WASM v2 (bundle textures) ===");
    app::run();
}
