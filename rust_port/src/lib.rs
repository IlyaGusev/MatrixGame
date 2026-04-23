pub mod gfx;
pub mod matrix_game;
pub mod matrix_lib;
pub mod platform;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("=== MatrixGame WASM v74 (anim+move debug) ===");
    matrix_game::form_game::run();
}
