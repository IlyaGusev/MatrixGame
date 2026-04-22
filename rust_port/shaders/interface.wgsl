// 2D textured quad pipeline for CIFaceElement rendering
// (Interface/CIFaceElement.cpp Render). Positions come through in
// screen-space pixel coordinates; the vertex shader converts to NDC
// using a screen-size uniform.

struct Uniforms {
    // .x = screen_w, .y = screen_h
    screen_size: vec4<f32>,
};

// Group 0: shared per-frame uniforms.
@group(0) @binding(0) var<uniform> u: Uniforms;

// Group 1: the atlas bound for the current draw batch. Re-bound
// between batches so one draw call per atlas covers its elements.
@group(1) @binding(0) var atlas: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct VsIn {
    @location(0) pos: vec2<f32>,     // screen pixels, top-left origin
    @location(1) uv:  vec2<f32>,
    @location(2) tint: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let nx = in.pos.x / u.screen_size.x * 2.0 - 1.0;
    let ny = 1.0 - in.pos.y / u.screen_size.y * 2.0;
    var out: VsOut;
    out.clip = vec4<f32>(nx, ny, 0.0, 1.0);
    out.uv = in.uv;
    out.tint = in.tint;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(atlas, atlas_sampler, in.uv);
    return c * in.tint;
}
