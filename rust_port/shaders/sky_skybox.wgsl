struct U {
    sky_view_proj: mat4x4<f32>,
    // .x = cut_dn. Ports MatrixMap.cpp:2064 — recomputed each frame from
    // camera altitude so the skybox's horizon row tracks the view.
    cut_dn: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_main(
    @location(0) xy: vec2<f32>,
    @location(1) bottom_w: f32,
    @location(2) u_coord: f32,
    @location(3) v0: f32,
    @location(4) v1: f32,
) -> VOut {
    var out: VOut;
    let cut = u.cut_dn.x;
    // Top corners at z=1.0, bottom corners at geo_dn = 1 - 2*cut_dn
    // (MatrixMap.cpp:2067). bottom_w is 0 or 1, interpolating between them.
    let z = 1.0 - 2.0 * cut * bottom_w;
    let pos = vec3<f32>(xy, z);
    // Bottom V scales with cut_dn too (MatrixMap.cpp:2072): top = v0,
    // bottom = v0 + (v1 - v0) * cut_dn.
    let v = v0 + (v1 - v0) * cut * bottom_w;
    let clip = u.sky_view_proj * vec4<f32>(pos, 1.0);
    // Pin to far plane so the depth test never rejects the skybox.
    out.clip_pos = vec4<f32>(clip.xy, clip.w, clip.w);
    out.uv = vec2<f32>(u_coord, v);
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(t, s, in.uv).rgb, 1.0);
}
