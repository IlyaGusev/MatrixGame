//! Map loader — reads terrain heightmap and properties from .CMAP files.

use anyhow::{bail, Context, Result};

use crate::assets::storage::Storage;

pub const GLOBAL_SCALE: f32 = 20.0;

/// Heightmap point from SCompilePoint (12 bytes, /Zp1 packed).
#[derive(Debug, Clone, Copy)]
pub struct CompilePoint {
    pub move_idx: i32,
    pub z: f32,
    pub b: u8,
    pub g: u8,
    pub r: u8,
    pub flags: u8,
}

/// Loaded map data.
pub struct GameMap {
    pub size_x: usize,
    pub size_y: usize,
    pub tex_union_dim: usize,
    pub water_color: u32,
    pub light_main_color: u32,
    pub light_main_dir: [f32; 3],
    pub macro_texture_path: Option<String>,
    pub macro_texture_size: i32,  // m_MacrotextureSize from "SIM" param
    pub points: Vec<CompilePoint>, // (size_x+1) * (size_y+1) points
}

impl GameMap {
    /// Load a map from CMAP file data (raw bytes from pkg archive).
    pub fn from_cmap_bytes(cmap_data: &[u8]) -> Result<Self> {
        let stor = Storage::from_bytes(cmap_data).context("parsing CStorage from CMAP")?;

        // Read properties
        let prop_names = stor
            .get_buf("properties", "Name")
            .context("missing properties/Name")?;
        let prop_values = stor
            .get_buf("properties", "Value")
            .context("missing properties/Value")?;

        let size_x = find_property_int(prop_names, prop_values, "SizeInUnitsX")
            .context("missing SizeInUnitsX")? as usize;
        let size_y = find_property_int(prop_names, prop_values, "SizeInUnitsY")
            .context("missing SizeInUnitsY")? as usize;

        let tex_union_dim = find_property_int(prop_names, prop_values, "TexUnionDim")
            .unwrap_or(16) as usize;
        let water_color = find_property_int(prop_names, prop_values, "WaterColor")
            .unwrap_or(0x003060) as u32;
        let light_main_color = find_property_int(prop_names, prop_values, "LightMainColor")
            .unwrap_or(0x989898) as u32;

        // Light direction: RotX(angleX) * RotZ(angleZ) * (0, 0, -1)
        // Ports MatrixMapPrepare.cpp lines 1228-1231
        let angle_x = find_property_float(prop_names, prop_values, "LightMainAngleX")
            .unwrap_or(0.61);
        let angle_z = find_property_float(prop_names, prop_values, "LightMainAngleZ")
            .unwrap_or(-1.75);
        let (sx, cx_) = angle_x.sin_cos();
        let (sz, cz) = angle_z.sin_cos();
        // D3DXMatrixRotationX * D3DXMatrixRotationZ * (0,0,-1)
        // RotX: y' = y*cx - z*sx, z' = y*sx + z*cx
        // RotZ: x' = x*cz - y*sz, y' = x*sz + y*cz
        // Start with (0, 0, -1):
        // After RotX: (0, 0*cx - (-1)*sx, 0*sx + (-1)*cx) = (0, sx, -cx)
        // After RotZ: (0*cz - sx*sz, 0*sz + sx*cz, -cx) = (-sx*sz, sx*cz, -cx_)
        let light_main_dir = [-sx * sz, sx * cz, -cx_];

        // MacroTexture = "Matrix\Macrotexture\05?SIM80" → size = 80
        let mut macro_texture_path = None;
        let mut macro_texture_size = 1i32;
        if let Some(idx) = prop_names.find_as_wstr("MacroTexture") {
            let val = prop_values.get_as_wstr(idx);
            let path = val.split('?').next().unwrap_or("").replace('\\', "/");
            if !path.is_empty() {
                macro_texture_path = Some(path);
            }
            if let Some(sim_pos) = val.find("SIM") {
                if let Ok(size) = val[sim_pos + 3..].parse::<i32>() {
                    macro_texture_size = size;
                }
            }
        }

        log::info!("map: size = {}x{} units, TexUnionDim={}", size_x, size_y, tex_union_dim);

        // Read heightmap points
        let points_buf = stor
            .get_buf("points", "Data")
            .context("missing points/Data")?;
        let points_data = points_buf.get_bytes(0);

        let expected_points = (size_x + 1) * (size_y + 1);
        let point_size = 12; // sizeof(SCompilePoint) under /Zp1
        if points_data.len() < expected_points * point_size {
            bail!(
                "points data too small: {} bytes, expected {} ({} points * {})",
                points_data.len(),
                expected_points * point_size,
                expected_points,
                point_size
            );
        }

        let mut points = Vec::with_capacity(expected_points);
        for i in 0..expected_points {
            let off = i * point_size;
            let move_idx = i32::from_le_bytes([
                points_data[off],
                points_data[off + 1],
                points_data[off + 2],
                points_data[off + 3],
            ]);
            let z = f32::from_le_bytes([
                points_data[off + 4],
                points_data[off + 5],
                points_data[off + 6],
                points_data[off + 7],
            ]);
            let b = points_data[off + 8];
            let g = points_data[off + 9];
            let r = points_data[off + 10];
            let flags = points_data[off + 11];

            points.push(CompilePoint {
                move_idx,
                z,
                b,
                g,
                r,
                flags,
            });
        }

        log::info!(
            "map: loaded {} heightmap points, z range: {:.1}..{:.1}",
            points.len(),
            points.iter().map(|p| p.z).fold(f32::INFINITY, f32::min),
            points.iter().map(|p| p.z).fold(f32::NEG_INFINITY, f32::max),
        );

        Ok(Self {
            size_x,
            size_y,
            tex_union_dim,
            water_color,
            light_main_color,
            light_main_dir,
            macro_texture_path,
            macro_texture_size,
            points,
        })
    }

    /// Get a heightmap point at grid coordinates.
    pub fn point(&self, x: usize, y: usize) -> &CompilePoint {
        &self.points[y * (self.size_x + 1) + x]
    }

    /// Map width in world units.
    pub fn world_width(&self) -> f32 {
        self.size_x as f32 * GLOBAL_SCALE
    }

    /// Map height in world units.
    pub fn world_height(&self) -> f32 {
        self.size_y as f32 * GLOBAL_SCALE
    }
}

fn find_property_int(
    names: &crate::assets::storage::DataBuf,
    values: &crate::assets::storage::DataBuf,
    key: &str,
) -> Result<i32> {
    let idx = names
        .find_as_wstr(key)
        .with_context(|| format!("property '{key}' not found"))?;
    let val_str = values.get_as_wstr(idx);
    val_str
        .trim()
        .parse::<i32>()
        .with_context(|| format!("property '{key}' not a valid int: '{val_str}'"))
}

fn find_property_float(
    names: &crate::assets::storage::DataBuf,
    values: &crate::assets::storage::DataBuf,
    key: &str,
) -> Result<f32> {
    let idx = names
        .find_as_wstr(key)
        .with_context(|| format!("property '{key}' not found"))?;
    let val_str = values.get_as_wstr(idx);
    val_str
        .trim()
        .parse::<f32>()
        .with_context(|| format!("property '{key}' not a valid float: '{val_str}'"))
}
