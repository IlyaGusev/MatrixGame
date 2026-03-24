#[cfg(not(target_arch = "wasm32"))]
mod native;

#[cfg(target_arch = "wasm32")]
mod web;

/// Returns the current time in seconds (monotonic-ish).
#[cfg(not(target_arch = "wasm32"))]
pub fn now_secs() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
pub fn now_secs() -> f64 {
    web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now()
        / 1000.0
}
