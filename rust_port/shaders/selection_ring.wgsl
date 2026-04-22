// Port of `CMatrixEffectSelection::Draw` (MatrixEffectSelection.cpp:87-95).
// Each ring "dot" in the original is a CBillboard — a camera-facing
// textured quad. We skip the texture and draw a soft circular disc in
// the fragment shader; the visible silhouette is the same (round dots
// around a ring).

struct Uniforms {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,   // xyz = world-space camera right
    cam_up:    vec4<f32>,   // xyz = world-space camera up
    color:     vec4<f32>,   // a = alpha
    size:      vec4<f32>,   // x = dot world-space half-size (SEL_SIZE)
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    // Per-vertex: corner offset (−1 or +1 per axis).
    @location(0) corner: vec2<f32>,
    // Per-instance: world-space dot center.
    @location(1) center: vec3<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    // Camera-facing billboard: expand the quad along camera right/up
    // in world space so every dot always faces the viewer. Matches the
    // CBillboard visual.
    let half = u.size.x;
    let world = in.center
              + u.cam_right.xyz * (in.corner.x * half)
              + u.cam_up.xyz    * (in.corner.y * half);
    var out: VsOut;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    out.uv = in.corner;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Soft circular disc. `d` is radial distance in the corner space
    // (0 at center, 1 at the quad edge). Discard outside the unit
    // circle + gentle feather for anti-aliasing.
    let d = length(in.uv);
    if d > 1.0 { discard; }
    let alpha = u.color.a * smoothstep(1.0, 0.7, d);
    return vec4<f32>(u.color.rgb, alpha);
}
