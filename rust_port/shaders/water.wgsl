struct U {
    view_proj: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    water_color: vec4<f32>,
    light_color: vec4<f32>,
    light_dir: vec4<f32>,
    params: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_water: texture_2d<f32>;
@group(0) @binding(2) var t_mirror: texture_2d<f32>;
@group(0) @binding(3) var s: sampler;
@group(0) @binding(4) var t_alpha: texture_2d<f32>;
@group(0) @binding(5) var s_alpha: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) water_uv: vec2<f32>,
    @location(1) cam_normal: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) alpha_uv: vec2<f32>,
    @location(4) view_dist: f32,
};

@vertex fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) water_uv: vec2<f32>,
    @location(3) alpha_uv: vec2<f32>,
) -> VOut {
    var out: VOut;
    let clip = u.view_proj * vec4<f32>(position, 1.0);
    out.clip_pos = clip;
    out.water_uv = water_uv;
    out.alpha_uv = alpha_uv;

    out.cam_normal = (u.normal_mat * vec4<f32>(normal, 0.0)).xyz;
    out.world_normal = normal;
    out.view_dist = clip.w;
    return out;
}

// Solid (ocean) pass vertex shader — writes clip-space z = w so post-divide
// NDC z = 1.0 (far plane). Combined with LessEqual depth testing, this
// paints a solid-water backdrop only on pixels where no terrain drew depth:
//   * truly off-map cells (depth buffer still cleared to 1.0)
//   * water "holes" inside on-map has_group meshes where the map compiler
//     skipped triangles over specific water cells (seen around Atoll's
//     coastal atolls — without this fill the alpha water pass blends
//     against the clear color and the sky shows through).
// Fog still reads `view_dist` from clip.w so atmospheric depth matches the
// alpha pass (MatrixWater.cpp's single `m_Water->Draw(m)` reuses one mesh
// for both passes).
@vertex fn vs_main_solid(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) water_uv: vec2<f32>,
    @location(3) alpha_uv: vec2<f32>,
) -> VOut {
    var out: VOut;
    let clip = u.view_proj * vec4<f32>(position, 1.0);
    out.clip_pos = vec4<f32>(clip.xy, clip.w, clip.w);
    out.water_uv = water_uv;
    out.alpha_uv = alpha_uv;

    out.cam_normal = (u.normal_mat * vec4<f32>(normal, 0.0)).xyz;
    out.world_normal = normal;
    out.view_dist = clip.w;
    return out;
}

@fragment fn fs_main(f: VOut) -> @location(0) vec4<f32> {
    let n = normalize(f.world_normal);
    let ndotl = max(dot(n, -u.light_dir.xyz), 0.0);
    let diffuse = u.water_color.rgb + u.light_color.rgb * ndotl;

    let water = textureSample(t_water, s, f.water_uv);
    let stage1 = clamp(water.rgb * diffuse * 2.0, vec3<f32>(0.0), vec3<f32>(1.0));

    let mirror_uv = f.cam_normal.xy * 0.5 + 0.5;
    let mirror = textureSample(t_mirror, s, mirror_uv);
    let final_rgb = mirror.rgb * mirror.a + stage1 * (1.0 - mirror.a);

    let fog_f = clamp((u.fog_params.y - f.view_dist) / (u.fog_params.y - u.fog_params.x), 0.0, 1.0);
    let fogged = mix(u.fog_color.rgb, final_rgb, fog_f);

    let alpha = textureSample(t_alpha, s_alpha, f.alpha_uv).r;

    return vec4<f32>(fogged, alpha);
}
