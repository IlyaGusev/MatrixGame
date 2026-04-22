struct GU {
    view_proj: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: GU;
@group(0) @binding(1) var t_atlas: texture_2d<f32>;
@group(0) @binding(2) var s_atlas: sampler;
@group(0) @binding(3) var t_gloss: texture_2d<f32>;
@group(0) @binding(4) var t_refl: texture_2d<f32>;
@group(0) @binding(5) var s_refl: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) cam_normal: vec3<f32>,
    @location(2) view_dist: f32,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VOut {
    var out: VOut;
    let clip = u.view_proj * vec4<f32>(position, 1.0);
    out.clip_pos = clip;
    out.uv = uv;
    out.cam_normal = (u.normal_mat * vec4<f32>(normal, 0.0)).xyz;
    out.view_dist = clip.w;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let atlas = textureSample(t_atlas, s_atlas, in.uv);
    // Gloss texture uses the same UVs as the atlas (same clamp/wrap rules),
    // so we reuse the atlas sampler rather than carving out a second binding.
    let gloss = textureSample(t_gloss, s_atlas, in.uv);
    // Reflection UV from camera-space normal (sphere map, matches water mirror
    // sampling in water.rs).
    let refl_uv = normalize(in.cam_normal).xy * 0.5 + 0.5;
    let refl = textureSample(t_refl, s_refl, refl_uv);
    let spec = gloss.rgb * refl.rgb;
    // Fog attenuates specular the same way it attenuates the diffuse base.
    let f = clamp((u.fog_params.y - in.view_dist) / (u.fog_params.y - u.fog_params.x), 0.0, 1.0);
    let spec_fogged = spec * f;
    return vec4<f32>(spec_fogged, atlas.a);
}
