struct Uniforms {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var t_atlas: texture_2d<f32>;
@group(0) @binding(2) var s_atlas: sampler;
@group(0) @binding(3) var t_macro: texture_2d<f32>;
@group(0) @binding(4) var s_macro: sampler;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) macro_uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) macro_uv: vec2<f32>,
    @location(3) view_dist: f32,
};

@vertex fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    let clip = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.clip_position = clip;
    out.color = in.color;
    out.uv = in.uv;
    out.macro_uv = in.macro_uv;
    out.view_dist = clip.w;
    return out;
}

// Linear fog (D3DFOG_LINEAR): factor=1 keeps original color, factor=0 replaces with fog_color.
// Use clip-space w (== view-space distance along forward axis) as the fog distance,
// matching D3DFOG_TABLE + WFOG semantics.
fn apply_fog(color: vec3<f32>, clip_w: f32) -> vec3<f32> {
    let f = clamp((uniforms.fog_params.y - clip_w) / (uniforms.fog_params.y - uniforms.fog_params.x), 0.0, 1.0);
    return mix(uniforms.fog_color.rgb, color, f);
}

fn shade_terrain(in: VOut) -> vec4<f32> {
    let atlas = textureSample(t_atlas, s_atlas, in.uv);
    let macro_tex = textureSample(t_macro, s_macro, in.macro_uv);
    let blended = macro_tex.rgb * macro_tex.a + atlas.rgb * (1.0 - macro_tex.a);
    return vec4<f32>(blended * in.color.rgb, atlas.a);
}

@fragment fn fs_main_opaque(in: VOut) -> @location(0) vec4<f32> {
    let shaded = shade_terrain(in);
    let fogged = apply_fog(shaded.rgb, in.view_dist);
    return vec4<f32>(fogged, 1.0);
}

@fragment fn fs_main_alpha(in: VOut) -> @location(0) vec4<f32> {
    let shaded = shade_terrain(in);
    let fogged = apply_fog(shaded.rgb, in.view_dist);
    return vec4<f32>(fogged, shaded.a);
}
