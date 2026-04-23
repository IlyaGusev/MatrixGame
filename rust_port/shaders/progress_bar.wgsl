// Port of `CMatrixProgressBar::Draw` (MatrixProgressBar.cpp:141-281).
// Two passes per bar:
//   1. Empty background — 3-segment stretch, UV row [0, 0.25], white
//      TFACTOR.
//   2. Filled portion — 3-segment stretch at variable width, UV row
//      [0.25, 0.5] (normal) or [0.5, 0.75] / [0.75, 1.0] (growing
//      cap when width < 2h), modulated by a red→yellow→green LIC
//      color matching `PB_COLOR_0..2`.
//
// The C++ D3DTOP_MODULATE combines texture color × TFACTOR; in WGSL
// we multiply `textureSample().rgb * u.color.rgb` directly.

struct Uniforms {
    screen_size: vec4<f32>,  // .xy
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;

struct VsIn {
    @location(0) pos: vec2<f32>,  // screen pixels
    @location(1) uv:  vec2<f32>,  // 0..1 atlas UV
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let nx = in.pos.x / u.screen_size.x * 2.0 - 1.0;
    let ny = 1.0 - in.pos.y / u.screen_size.y * 2.0;
    var o: VsOut;
    o.clip = vec4<f32>(nx, ny, 0.0, 1.0);
    o.uv = in.uv;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas, atlas_sampler, in.uv);
    // D3DTOP_MODULATE (tex * TFACTOR) + D3DTOP_MODULATE alpha.
    return vec4<f32>(tex.rgb * u.color.rgb, tex.a * u.color.a);
}
