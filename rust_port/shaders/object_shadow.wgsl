struct U {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_shadow: texture_2d<f32>;
@group(0) @binding(2) var s_shadow: sampler;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    out.clip_position = u.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let shadow = textureSample(t_shadow, s_shadow, in.uv).a;
    if (shadow <= 0.01) { discard; }
    return vec4<f32>(0.0, 0.0, 0.0, shadow * 0.45);
}
