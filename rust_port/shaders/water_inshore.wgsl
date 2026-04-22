struct U {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var t_foam: texture_2d<f32>;
@group(0) @binding(2) var s_foam: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_dist: f32,
};

@vertex fn vs_main(
    @location(0) quad_pos: vec3<f32>,
    @location(1) quad_uv: vec2<f32>,
    @location(2) x_axis: vec2<f32>,
    @location(3) y_axis: vec2<f32>,
    @location(4) translation: vec3<f32>,
    @location(5) color: vec4<f32>,
) -> VOut {
    // Rebuild the per-wave world transform from the packed axis vectors.
    // Matches CVectorObjectAnim/MatrixWater.cpp:148-154 row-major layout:
    //   X row = dir.xy * scale
    //   Y row = (-dir.y, dir.x) * scale
    //   Z row = (0,0,1)
    //   Translation = (xx, yy, WATER_LEVEL)
    let world_xy = quad_pos.x * x_axis + quad_pos.y * y_axis;
    let world = vec3<f32>(world_xy.x + translation.x, world_xy.y + translation.y, translation.z);
    let clip = u.view_proj * vec4<f32>(world, 1.0);
    var out: VOut;
    out.clip_pos = clip;
    out.uv = quad_uv;
    out.color = color;
    out.view_dist = clip.w;
    return out;
}

@fragment fn fs_main(f: VOut) -> @location(0) vec4<f32> {
    // SetColor/Alpha MODULATE TEXTURE × TFACTOR (MatrixWater.cpp:109-110):
    // final = tex.rgb * color.rgb, final.a = tex.a * color.a.
    let tex = textureSample(t_foam, s_foam, f.uv);
    let rgb = tex.rgb * f.color.rgb;
    let alpha = tex.a * f.color.a;
    // Fog applies to the pre-alpha color in the original fixed-function pass.
    let fog_f = clamp((u.fog_params.y - f.view_dist) / (u.fog_params.y - u.fog_params.x), 0.0, 1.0);
    let fogged = mix(u.fog_color.rgb, rgb, fog_f);
    return vec4<f32>(fogged, alpha);
}
