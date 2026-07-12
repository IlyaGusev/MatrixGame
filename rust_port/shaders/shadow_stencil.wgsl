// Stencil shadow volumes + darken quad — port of the render half of
// CVOShadowStencil::Render (ShadowStencil.cpp:412) and the
// CMatrixMap::DrawShadows composition (MatrixMap.cpp:1865-2000).
//
// Volumes draw with color writes off; only the stencil INCR/DECR
// (front/back) matters. The darken pass then covers the screen where
// stencil >= 1 with `m_ShadowColor` modulated by the 0xC0C0C0C0
// texture factor (computed CPU-side into u.darken_color) using
// src-alpha blending.

struct Uniforms {
    view_proj: mat4x4<f32>,
    darken_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_volume(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return u.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_volume() -> @location(0) vec4<f32> {
    // Write mask is empty — the value is irrelevant (the C++ blends
    // ZERO/ONE for the same effect, MatrixMap.cpp:1887-1889).
    return vec4<f32>(0.0);
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_darken() -> @location(0) vec4<f32> {
    return u.darken_color;
}
