struct U {
    u_row: vec4<f32>,
    v_row: vec4<f32>,
    // .x != 0 ⇒ TF_ALPHATEST on the surface being baked. When not set, the
    // original passes D3DTA_TFACTOR through the alpha stage, which produces
    // a fully opaque silhouette regardless of the diffuse's alpha channel
    // (MatrixSkinManager.cpp:195-201).
    alpha_test: vec4<u32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var s_diffuse: sampler;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs_main(in: VIn) -> VOut {
    let p = vec4<f32>(in.position, 1.0);
    let uv = vec2<f32>(dot(u.u_row, p) + 0.5, dot(u.v_row, p) + 0.5);
    var out: VOut;
    out.clip_position = vec4<f32>(1.0 - uv.x * 2.0, uv.y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_diffuse, s_diffuse, in.uv);
    // Discard only for TF_ALPHATEST diffuses, and with the engine default
    // alpha ref of 8/255 (3g.cpp:662-664). Non-alpha-test surfaces render
    // as a solid silhouette — the shadow density is carried by alpha=1.0
    // so the projected shadow pass picks the full footprint.
    if (u.alpha_test.x != 0u) {
        if (tex.a < 8.0 / 255.0) { discard; }
        return vec4<f32>(1.0, 1.0, 1.0, tex.a);
    }
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
