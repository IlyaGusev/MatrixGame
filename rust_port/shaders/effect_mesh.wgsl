// Effect mesh shader — projectiles (roket/mina/bullet) and explosion
// debris pieces. Row-vector world matrix + color tint (the D3D
// TEXTUREFACTOR fade on debris).

struct Uniforms { view_proj: mat4x4<f32> };
struct DrawData { rows: mat4x4<f32>, color: vec4<f32> };

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var<uniform> d: DrawData;
@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    // Row-vector convention: world = x*row0 + y*row1 + z*row2 + row3.
    let world = d.rows[0].xyz * pos.x + d.rows[1].xyz * pos.y + d.rows[2].xyz * pos.z
        + d.rows[3].xyz;
    var out: VsOut;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let t = textureSample(tex, samp, in.uv);
    return vec4<f32>(t.rgb * d.color.rgb, 1.0);
}
