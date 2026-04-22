struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex fn vs_main(@location(0) pos: vec2<f32>, @location(1) col: vec4<f32>) -> VOut {
    var out: VOut;
    out.clip_pos = vec4<f32>(pos, 0.0, 1.0);
    out.color = col;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return in.color;
}
