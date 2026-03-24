struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) uv       : vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0)       uv            : vec2<f32>,
};

struct Uniforms {
    view_proj : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(1) @binding(0) var t_diffuse : texture_2d<f32>;
@group(1) @binding(1) var s_diffuse : sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
