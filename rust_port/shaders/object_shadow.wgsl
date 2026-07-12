struct U {
    view_proj: mat4x4<f32>,
    shadow_color: vec4<f32>,
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
    // DrawShadowsProjFast (MatrixMap.cpp:1830-1834):
    //   colour = D3DTA_TFACTOR              -> u.shadow_color.rgb
    //   alpha  = MODULATE(TEXTURE, TFACTOR) -> silhouette.a * u.shadow_color.a
    let shadow = textureSample(t_shadow, s_shadow, in.uv).a;
    let a = shadow * u.shadow_color.a;
    if (a <= 0.01) { discard; }
    return vec4<f32>(u.shadow_color.rgb, a);
}

// Stencil-mark variant — the DrawShadows composition (MatrixMap.cpp:
// 1917-1961) draws proj silhouettes with color writes off, alpha test
// `texture.a >= 8/255` (D3DRS_ALPHAREF 0x8 / GREATEREQUAL, alpha op
// SELECT TEXTURE) and D3DSTENCILOP_REPLACE ref 1; the shared darken
// quad then applies the color wherever stencil >= 1.
@fragment fn fs_stencil(in: VOut) -> @location(0) vec4<f32> {
    if (textureSample(t_shadow, s_shadow, in.uv).a < 8.0 / 255.0) { discard; }
    return vec4<f32>(0.0);
}
