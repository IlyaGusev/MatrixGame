struct U {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    ambient_color: vec4<f32>,
    light_color: vec4<f32>,
    light_dir: vec4<f32>,
    camera_pos: vec4<f32>,
    time_ms: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var t_gloss: texture_2d<f32>;
@group(0) @binding(3) var t_back: texture_2d<f32>;
@group(0) @binding(4) var t_mask: texture_2d<f32>;
@group(0) @binding(5) var s_diffuse: sampler;
struct M {
    flags: vec4<u32>,
    scroll: vec4<f32>,
};
@group(0) @binding(6) var<uniform> m: M;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
    @location(7) terrain_color: vec4<f32>,
};
struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) view_dist: f32,
    @location(3) terrain_color: vec3<f32>,
    @location(4) world_pos: vec3<f32>,
};

@vertex fn vs_main(in: VIn) -> VOut {
    let p = vec4<f32>(in.position, 1.0);
    let world = vec4<f32>(dot(in.m0, p), dot(in.m1, p), dot(in.m2, p), dot(in.m3, p));
    let clip = u.view_proj * world;
    var out: VOut;
    out.clip_position = clip;
    out.uv = in.uv;
    let n = vec3<f32>(dot(in.m0.xyz, in.normal), dot(in.m1.xyz, in.normal), dot(in.m2.xyz, in.normal));
    out.normal = normalize(n);
    out.view_dist = clip.w;
    out.terrain_color = in.terrain_color.rgb;
    out.world_pos = world.xyz;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_diffuse, s_diffuse, in.uv);
    if (m.flags.w != 0u && tex.a < 8.0 / 255.0) { discard; }

    let n = normalize(in.normal);
    let l = normalize(-u.light_dir.xyz);
    let ndotl = max(dot(n, l), 0.0);
    let lighting = clamp(u.ambient_color.rgb + u.light_color.rgb * ndotl, vec3<f32>(0.0), vec3<f32>(1.0));
    var rgb = tex.rgb * in.terrain_color * lighting;
    let scroll_uv = in.uv + m.scroll.xy * u.time_ms.x;

    if (m.flags.z != 0u) {
        let mask = textureSample(t_mask, s_diffuse, scroll_uv);
        let back = textureSample(t_back, s_diffuse, scroll_uv);
        let back_rgb = back.rgb * in.terrain_color * lighting;
        let blend = max(mask.a, max(mask.r, max(mask.g, mask.b)));
        rgb = mix(rgb, back_rgb, clamp(blend, 0.0, 1.0));
    }

    if (m.flags.x != 0u) {
        let gloss = textureSample(t_gloss, s_diffuse, in.uv).rgb;
        let view_dir = normalize(u.camera_pos.xyz - in.world_pos);
        let fresnel = pow(1.0 - max(dot(view_dir, n), 0.0), 4.0);
        let reflection = mix(u.fog_color.rgb, vec3<f32>(1.0), fresnel);
        rgb += gloss * reflection;
    }

    let fog_f = clamp((u.fog_params.y - in.view_dist) / (u.fog_params.y - u.fog_params.x), 0.0, 1.0);
    rgb = mix(u.fog_color.rgb, rgb, fog_f);
    return vec4<f32>(rgb, 1.0);
}
