//! Cross-platform graphics plumbing (wgpu device + surface, WASM asset
//! bundle, file loader). No C++ counterpart — this layer exists because
//! the original relied on the Windows/DirectX platform layer which we
//! replace with wgpu and a WASM shim.

pub mod bundle;
pub mod context;
pub mod loader;
