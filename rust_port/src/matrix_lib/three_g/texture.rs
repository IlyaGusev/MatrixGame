//! GPU texture utilities — DDS decoding, texture creation, mipmaps,
//! plus `CBaseTexture::ParseFlags` material/alpha-test parsing
//! (Texture.cpp:96-152).

use wgpu::util::DeviceExt;

#[derive(Clone, Debug, Default)]
pub struct MaterialSpec {
    pub diffuse: Option<String>,
    pub gloss: Option<String>,
    pub back: Option<String>,
    pub mask: Option<String>,
    pub scroll: [f32; 2],
    /// True when the diffuse texture's `ParseFlags` call would set
    /// `TF_ALPHATEST` (Texture.cpp:96-152). Set eagerly from the
    /// `?Trans` suffix (Texture.cpp:108); the sibling `.txt` `AlphaTest`
    /// override (Texture.cpp:131-136) is applied later via
    /// `resolve_alpha_test_with_txt`, which needs asset I/O.
    pub alpha_test: bool,
}

pub fn parse_material_spec(spec: &str) -> MaterialSpec {
    parse_material_spec_with_prefix(spec, None)
}

pub fn parse_material_spec_with_prefix(spec: &str, prefix: Option<&str>) -> MaterialSpec {
    let parts: Vec<&str> = spec.split('*').collect();
    MaterialSpec {
        diffuse: parts.first().and_then(|t| resolve_texture_name(t, prefix)),
        gloss: parts.get(1).and_then(|t| resolve_texture_name(t, prefix)),
        back: parts.get(2).and_then(|t| resolve_texture_name(t, prefix)),
        mask: parts.get(3).and_then(|t| resolve_texture_name(t, prefix)),
        scroll: parts.get(4).map(|t| parse_scroll(t)).unwrap_or([0.0, 0.0]),
        alpha_test: parts.first().is_some_and(|t| has_trans_suffix(t)),
    }
}

pub fn merge_materials(base: &MaterialSpec, overlay: Option<&MaterialSpec>) -> MaterialSpec {
    let Some(overlay) = overlay else {
        return base.clone();
    };
    let (diffuse, alpha_test) = if overlay.diffuse.is_some() {
        (overlay.diffuse.clone(), overlay.alpha_test)
    } else {
        (base.diffuse.clone(), base.alpha_test)
    };
    MaterialSpec {
        diffuse,
        gloss: overlay.gloss.clone().or_else(|| base.gloss.clone()),
        back: overlay.back.clone().or_else(|| base.back.clone()),
        mask: overlay.mask.clone().or_else(|| base.mask.clone()),
        scroll: if overlay.scroll != [0.0, 0.0] {
            overlay.scroll
        } else {
            base.scroll
        },
        alpha_test,
    }
}

/// Ports the `?Trans` flag detection inside `CBaseTexture::ParseFlags`
/// (Texture.cpp:102-111). The raw texture spec is `path?Opt1?Opt2?...`; any
/// `Trans` option sets `TF_ALPHATEST`. Case-sensitive match mirrors the
/// original `CWStr::Equal` memcmp.
pub fn has_trans_suffix(raw: &str) -> bool {
    let raw = raw.trim().trim_end_matches('\0');
    let mut parts = raw.split('?');
    parts.next();
    parts.any(|opt| opt.trim() == "Trans")
}

/// Ports the sibling `.txt` override pass of `CBaseTexture::ParseFlags`
/// (Texture.cpp:113-136). Skips `pinguin.txt` / `robotarget.txt` because
/// those carry unrelated block content (Texture.cpp:121-124).
pub fn resolve_alpha_test_with_txt(
    diffuse_path: &str,
    suffix_flag: bool,
    read_file: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> bool {
    let txt_path = replace_extension(diffuse_path, "txt");
    if txt_path.contains("pinguin.txt") || txt_path.contains("robotarget.txt") {
        return suffix_flag;
    }
    let Some(bytes) = read_file(&txt_path) else {
        return suffix_flag;
    };
    let bp = crate::matrix_lib::base::blockpar::BlockPar::parse_bytes(&bytes);
    match bp.par_get_ne("AlphaTest") {
        Some(v) => {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                suffix_flag
            } else {
                trimmed != "0"
            }
        }
        None => suffix_flag,
    }
}

fn replace_extension(path: &str, new_ext: &str) -> String {
    match path.rsplit_once('.') {
        Some((stem, _)) => format!("{stem}.{new_ext}"),
        None => format!("{path}.{new_ext}"),
    }
}

pub fn resolve_texture_name(raw: &str, prefix: Option<&str>) -> Option<String> {
    let name = raw.trim().trim_end_matches('\0');
    if name.is_empty() || name == "." {
        return None;
    }
    let name = name.split('?').next().unwrap_or("").trim();
    if name.is_empty() {
        return None;
    }
    let normalized = name.trim_start_matches(".\\").replace('\\', "/");
    match prefix {
        Some(prefix) if !normalized.contains('/') => Some(format!("{prefix}{normalized}")),
        _ => Some(normalized),
    }
}

pub fn parse_scroll(raw: &str) -> [f32; 2] {
    let mut it = raw.split(',');
    let u = it
        .next()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(0.0);
    let v = it
        .next()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(0.0);
    [u, v]
}

/// Decode DDS (DXT1/DXT3/DXT5) or standard image formats.
pub fn decode_texture_bytes(data: &[u8]) -> Option<image::RgbaImage> {
    if data.len() > 128 && &data[0..4] == b"DDS " {
        return decode_dds(data);
    }
    image::load_from_memory(data).ok().map(|img| img.to_rgba8())
}

fn decode_dds(data: &[u8]) -> Option<image::RgbaImage> {
    if data.len() < 128 {
        return None;
    }
    let height = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let width = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    let fourcc = &data[84..88];
    let pixel_data = &data[128..];

    let is_dxt3 = fourcc == b"DXT3";
    let is_dxt5 = fourcc == b"DXT5";
    let block_bytes = if fourcc == b"DXT1" {
        8usize
    } else if is_dxt3 || is_dxt5 {
        16
    } else {
        return None;
    };

    let bw = width.div_ceil(4) as usize;
    let bh = height.div_ceil(4) as usize;
    let mut img = image::RgbaImage::new(width, height);

    for by in 0..bh {
        for bx in 0..bw {
            let block_start = (by * bw + bx) * block_bytes;
            if block_start + block_bytes > pixel_data.len() {
                break;
            }

            let color_off = if block_bytes == 16 { 8 } else { 0 };
            let mut colors = decode_dxt_color_block(
                &pixel_data[block_start + color_off..block_start + color_off + 8],
                fourcc == b"DXT1",
            );

            if is_dxt3 {
                let alpha_block = &pixel_data[block_start..block_start + 8];
                for py in 0..4usize {
                    let row_bits =
                        u16::from_le_bytes([alpha_block[py * 2], alpha_block[py * 2 + 1]]);
                    for px in 0..4usize {
                        let a4 = ((row_bits >> (px * 4)) & 0xF) as u8;
                        colors[py * 4 + px][3] = (a4 << 4) | a4;
                    }
                }
            }
            if is_dxt5 {
                let a0 = pixel_data[block_start] as u16;
                let a1 = pixel_data[block_start + 1] as u16;
                let mut alpha_lut = [0u8; 8];
                alpha_lut[0] = a0 as u8;
                alpha_lut[1] = a1 as u8;
                if a0 > a1 {
                    for i in 2..8u16 {
                        alpha_lut[i as usize] = ((a0 * (8 - i) + a1 * (i - 1)) / 7) as u8;
                    }
                } else {
                    for i in 2..6u16 {
                        alpha_lut[i as usize] = ((a0 * (6 - i) + a1 * (i - 1)) / 5) as u8;
                    }
                    alpha_lut[6] = 0;
                    alpha_lut[7] = 255;
                }
                let bits: u64 = pixel_data[block_start + 2] as u64
                    | (pixel_data[block_start + 3] as u64) << 8
                    | (pixel_data[block_start + 4] as u64) << 16
                    | (pixel_data[block_start + 5] as u64) << 24
                    | (pixel_data[block_start + 6] as u64) << 32
                    | (pixel_data[block_start + 7] as u64) << 40;
                for (p, color) in colors.iter_mut().enumerate() {
                    let idx = ((bits >> (p * 3)) & 7) as usize;
                    color[3] = alpha_lut[idx];
                }
            }

            for py in 0..4u32 {
                for px in 0..4u32 {
                    let x = bx as u32 * 4 + px;
                    let y = by as u32 * 4 + py;
                    if x < width && y < height {
                        img.put_pixel(x, y, image::Rgba(colors[py as usize * 4 + px as usize]));
                    }
                }
            }
        }
    }
    Some(img)
}

fn decode_dxt_color_block(block: &[u8], allow_dxt1_transparency: bool) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let expand5 = |v: u16| -> u8 {
        let x = (v & 0x1F) as u8;
        (x << 3) | (x >> 2)
    };
    let expand6 = |v: u16| -> u8 {
        let x = (v & 0x3F) as u8;
        (x << 2) | (x >> 4)
    };

    let col0 = [expand5(c0 >> 11), expand6(c0 >> 5), expand5(c0), 255];
    let col1 = [expand5(c1 >> 11), expand6(c1 >> 5), expand5(c1), 255];

    let (col2, col3) = if !allow_dxt1_transparency || c0 > c1 {
        (
            [
                ((2 * col0[0] as u16 + col1[0] as u16) / 3) as u8,
                ((2 * col0[1] as u16 + col1[1] as u16) / 3) as u8,
                ((2 * col0[2] as u16 + col1[2] as u16) / 3) as u8,
                255,
            ],
            [
                ((col0[0] as u16 + 2 * col1[0] as u16) / 3) as u8,
                ((col0[1] as u16 + 2 * col1[1] as u16) / 3) as u8,
                ((col0[2] as u16 + 2 * col1[2] as u16) / 3) as u8,
                255,
            ],
        )
    } else {
        let avg = [
            ((col0[0] as u16 + col1[0] as u16) / 2) as u8,
            ((col0[1] as u16 + col1[1] as u16) / 2) as u8,
            ((col0[2] as u16 + col1[2] as u16) / 2) as u8,
            255,
        ];
        let transparent = [avg[0], avg[1], avg[2], 0];
        (avg, transparent)
    };

    let palette = [col0, col1, col2, col3];
    let mut result = [[0u8; 4]; 16];
    for row in 0..4 {
        let bits = block[4 + row] as u32;
        for col in 0..4 {
            result[row * 4 + col] = palette[((bits >> (col * 2)) & 3) as usize];
        }
    }
    result
}

pub fn create_solid_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color: [u8; 4],
) -> wgpu::TextureView {
    let pixels: Vec<u8> = color.repeat(4);
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("solid"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &pixels,
    );
    texture.create_view(&Default::default())
}

pub fn create_texture_from_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &image::RgbaImage,
) -> wgpu::TextureView {
    create_texture_from_rgba_mipped(device, queue, img, 1)
}

/// Ports CTextureManaged::LoadFromBitmap(bm, D3DFMT_DXT1, levels).
pub fn create_texture_from_rgba_mipped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &image::RgbaImage,
    levels: u32,
) -> wgpu::TextureView {
    let w = img.width();
    let h = img.height();
    let max_levels = (w.max(h) as f32).log2().floor() as u32 + 1;
    let mip_count = if levels <= 1 {
        1
    } else {
        levels.min(max_levels)
    };

    let mut all_data: Vec<u8> = Vec::new();
    all_data.extend_from_slice(img.as_raw());

    let mut current = img.clone();
    for _level in 1..mip_count {
        let mw = (current.width() / 2).max(1);
        let mh = (current.height() / 2).max(1);
        let mut mip = image::RgbaImage::new(mw, mh);
        for y in 0..mh {
            for x in 0..mw {
                let sx = x * 2;
                let sy = y * 2;
                let p00 = current
                    .get_pixel(sx.min(current.width() - 1), sy.min(current.height() - 1))
                    .0;
                let p10 = current
                    .get_pixel(
                        (sx + 1).min(current.width() - 1),
                        sy.min(current.height() - 1),
                    )
                    .0;
                let p01 = current
                    .get_pixel(
                        sx.min(current.width() - 1),
                        (sy + 1).min(current.height() - 1),
                    )
                    .0;
                let p11 = current
                    .get_pixel(
                        (sx + 1).min(current.width() - 1),
                        (sy + 1).min(current.height() - 1),
                    )
                    .0;
                let a00 = p00[3] as f32 / 255.0;
                let a10 = p10[3] as f32 / 255.0;
                let a01 = p01[3] as f32 / 255.0;
                let a11 = p11[3] as f32 / 255.0;
                let out_a = (p00[3] as u16 + p10[3] as u16 + p01[3] as u16 + p11[3] as u16) as f32
                    / (4.0 * 255.0);
                let premul_r = ((p00[0] as f32 * a00)
                    + (p10[0] as f32 * a10)
                    + (p01[0] as f32 * a01)
                    + (p11[0] as f32 * a11))
                    * 0.25;
                let premul_g = ((p00[1] as f32 * a00)
                    + (p10[1] as f32 * a10)
                    + (p01[1] as f32 * a01)
                    + (p11[1] as f32 * a11))
                    * 0.25;
                let premul_b = ((p00[2] as f32 * a00)
                    + (p10[2] as f32 * a10)
                    + (p01[2] as f32 * a01)
                    + (p11[2] as f32 * a11))
                    * 0.25;
                let (out_r, out_g, out_b) = if out_a > 1e-6 {
                    (
                        (premul_r / out_a).clamp(0.0, 255.0) as u8,
                        (premul_g / out_a).clamp(0.0, 255.0) as u8,
                        (premul_b / out_a).clamp(0.0, 255.0) as u8,
                    )
                } else {
                    (0, 0, 0)
                };
                mip.put_pixel(
                    x,
                    y,
                    image::Rgba([
                        out_r,
                        out_g,
                        out_b,
                        (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
                    ]),
                );
            }
        }
        all_data.extend_from_slice(mip.as_raw());
        current = mip;
    }

    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::MipMajor,
        &all_data,
    );
    texture.create_view(&Default::default())
}

pub fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&Default::default())
}
