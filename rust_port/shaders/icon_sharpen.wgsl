// Robot-icon mip filter — GPU mirror of matrix_lib/bitmap/sharpen.rs
// (port of sharpen.cpp / CBitmap::Make2xSmaller). RenderToTexture
// (MatrixMapStatic.cpp:648-652) halves the 256px robot render twice,
// running sharpen_run(lv=16) after each halving, to bake the 64px
// medium icon. Both passes are exact integer math on u8 texels, so
// textureLoad + round(x*255) keeps the result byte-identical to the
// CPU reference.

@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle.
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn texel(p: vec2<i32>) -> vec4<u32> {
    let dims = vec2<i32>(textureDimensions(src));
    let c = clamp(p, vec2<i32>(0), dims - 1);
    return vec4<u32>(round(textureLoad(src, c, 0) * 255.0));
}

// CBitmap::Make2xSmaller: per channel (p00+p01+p10+p11) >> 2.
@fragment
fn fs_downsample(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let d = vec2<i32>(pos.xy) * 2;
    let s = texel(d) + texel(d + vec2<i32>(1, 0)) + texel(d + vec2<i32>(0, 1))
        + texel(d + vec2<i32>(1, 1));
    return vec4<f32>(s >> vec4<u32>(2u)) / 255.0;
}

// sharpen_run(lv=16): out = clamp((c*384 - s8*16) >> 8, 0, 255) with
// clamp-to-edge neighbours; the 1px border zeroes alpha (do_conv path).
@fragment
fn fs_sharpen(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(src));
    let p = vec2<i32>(pos.xy);
    var s8 = vec4<i32>(0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx != 0 || dy != 0) {
                s8 += vec4<i32>(texel(p + vec2<i32>(dx, dy)));
            }
        }
    }
    let c = vec4<i32>(texel(p));
    var v = clamp((c * 384 - s8 * 16) >> vec4<u32>(8u), vec4<i32>(0), vec4<i32>(255));
    if (p.x == 0 || p.y == 0 || p.x == dims.x - 1 || p.y == dims.y - 1) {
        v.w = 0;
    }
    return vec4<f32>(v) / 255.0;
}
