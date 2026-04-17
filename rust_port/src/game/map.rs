//! Map loader — reads terrain heightmap and properties from .CMAP files.

use anyhow::{bail, Context, Result};

use crate::assets::storage::Storage;
use crate::game::common::{CELLFLAG_BRIDGE, CELLFLAG_FLAT, CELLFLAG_WATER};

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

/// Per-point surface normal (computed from surrounding heights).
#[derive(Debug, Clone, Copy)]
pub struct PointNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Per-cell coefficients matching SMatrixMapUnit fields used by GetZ.
#[derive(Debug, Clone, Copy)]
pub struct MapUnit {
    pub flags: u8,
    pub a1: f32,
    pub b1: f32,
    pub c1: f32,
    pub a2: f32,
    pub b2: f32,
    pub c2: f32,
}

/// Loaded map data.
pub struct GameMap {
    pub size_x: usize,
    pub size_y: usize,
    pub tex_union_dim: usize,
    pub water_color: u32,
    pub sky_color: u32,
    pub water_name: String,
    pub water_normal_len: f32,
    pub light_main_color: u32,
    pub light_main_dir: [f32; 3],
    pub macro_texture_path: Option<String>,
    pub macro_texture_size: i32,  // m_MacrotextureSize from "SIM" param
    pub points: Vec<CompilePoint>, // (size_x+1) * (size_y+1) points
    pub normals: Vec<PointNormal>,  // same size, computed from heights
    pub units: Vec<MapUnit>,        // size_x * size_y cells, ports SMatrixMapUnit GetZ data
    pub objects: Vec<ObjectInstance>, // palms / rocks / decorative scenery placements
}

/// A static object placement — ports the per-instance fields of DATA_OBJECTS
/// (MatrixMapPrepare.cpp:503-619). Position is uncentered world space (raw map
/// coords); renderer applies the map-center offset.
#[derive(Debug, Clone, Copy)]
pub struct ObjectInstance {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub angle_z: f32,
    pub angle_x: f32,
    pub angle_y: f32,
    pub scale: f32,
    pub type_id: u32,
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
        // DEF_SKY_COLOR (Common.hpp:27) = 0x1070FF
        let sky_color = find_property_int(prop_names, prop_values, "SkyColor")
            .unwrap_or(0x1070FF) as u32;
        let water_name = prop_names
            .find_as_wstr("WaterName")
            .map(|idx| prop_values.get_as_wstr(idx))
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "water_blue".to_string());
        let water_normal_len = find_property_float(prop_names, prop_values, "WaterNormLen")
            .unwrap_or(1.0);
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

        // Compute surface normals — ports PointCalcNormals (MatrixMapPrepare.cpp:20-105)
        let _w = size_x + 1;
        let normals = compute_normals(&points, size_x, size_y);
        let units = compute_units(&points, size_x, size_y);

        let objects = load_objects(&stor, &points, size_x, size_y);
        log::info!("map: loaded {} decorative objects", objects.len());

        Ok(Self {
            size_x,
            size_y,
            tex_union_dim,
            water_color,
            sky_color,
            water_name,
            water_normal_len,
            light_main_color,
            light_main_dir,
            macro_texture_path,
            macro_texture_size,
            points,
            normals,
            units,
            objects,
        })
    }

    /// Get a heightmap point at grid coordinates.
    pub fn point(&self, x: usize, y: usize) -> &CompilePoint {
        &self.points[y * (self.size_x + 1) + x]
    }

    pub fn normal(&self, x: usize, y: usize) -> &PointNormal {
        &self.normals[y * (self.size_x + 1) + x]
    }

    pub fn unit(&self, x: usize, y: usize) -> &MapUnit {
        &self.units[y * self.size_x + x]
    }

    pub fn world_width(&self) -> f32 {
        self.size_x as f32 * GLOBAL_SCALE
    }

    pub fn world_height(&self) -> f32 {
        self.size_y as f32 * GLOBAL_SCALE
    }

    /// Ports CMatrixMap::GetZ for terrain/water boundary tests.
    pub fn get_z(&self, wx: f32, wy: f32) -> f32 {
        let sx = wx / GLOBAL_SCALE;
        let sy = wy / GLOBAL_SCALE;
        let x = sx.floor() as i32;
        let y = sy.floor() as i32;

        if x < 0 || y < 0 || x >= self.size_x as i32 || y >= self.size_y as i32 {
            return -1000.0;
        }

        let unit = self.unit(x as usize, y as usize);
        if unit.flags & CELLFLAG_BRIDGE == 0 && unit.flags & CELLFLAG_WATER != 0 {
            return -1000.0;
        }
        if unit.flags & CELLFLAG_FLAT != 0 {
            return unit.a1;
        }

        let local_x = wx - x as f32 * GLOBAL_SCALE;
        let local_y = wy - y as f32 * GLOBAL_SCALE;
        if local_y < local_x {
            unit.a1 * local_x + unit.b1 * local_y + unit.c1
        } else {
            unit.a2 * local_x + unit.b2 * local_y + unit.c2
        }
    }
}

const CELLFLAG_LAND: u8 = 1 << 0;
const CELLFLAG_DOWN: u8 = 1 << 5;

/// Ports PointCalcNormals (MatrixMapPrepare.cpp:20-105).
/// For each point, computes normal from cross products of 4 adjacent triangles.
fn compute_normals(points: &[CompilePoint], size_x: usize, size_y: usize) -> Vec<PointNormal> {
    let w = size_x + 1;
    let h = size_y + 1;
    let mut normals = vec![PointNormal { x: 0.0, y: 0.0, z: 1.0 }; w * h];

    let get_z = |x: usize, y: usize| -> f32 { points[y * w + x].z };
    let get_flags = |x: usize, y: usize| -> u8 { points[y * w + x].flags };

    // Check if unit at (ux,uy) is land — unit flags are on point (ux,uy)
    let is_land = |ux: i32, uy: i32| -> bool {
        if ux < 0 || uy < 0 || ux >= size_x as i32 || uy >= size_y as i32 { return false; }
        get_flags(ux as usize, uy as usize) & CELLFLAG_LAND != 0
    };

    for py in 0..h {
        for px in 0..w {
            let ix = px as i32;
            let iy = py as i32;

            // Original: check unit adjacency for land before using neighbor points
            // p1=up (unit x,y-1), p2=right (unit x,y), p3=down (unit x,y), p4=left (unit x-1,y)
            let has_up = is_land(ix, iy - 1);
            let has_left = is_land(ix - 1, iy);
            let has_cur = is_land(ix, iy);

            let p0 = [px as f32 * GLOBAL_SCALE, py as f32 * GLOBAL_SCALE, get_z(px, py)];

            // Vectors to neighbors (relative to p0)
            let v_up = if has_up && py > 0 {
                Some([0.0, -GLOBAL_SCALE, get_z(px, py - 1) - p0[2]])
            } else { None };

            let v_right = if has_cur && px < size_x {
                Some([GLOBAL_SCALE, 0.0, get_z(px + 1, py) - p0[2]])
            } else { None };

            let v_down = if has_cur && py < size_y {
                Some([0.0, GLOBAL_SCALE, get_z(px, py + 1) - p0[2]])
            } else { None };

            let v_left = if has_left && px > 0 {
                Some([-GLOBAL_SCALE, 0.0, get_z(px - 1, py) - p0[2]])
            } else { None };

            // Average cross products of adjacent triangle pairs
            let mut nx = 0.0f32;
            let mut ny = 0.0f32;
            let mut nz = 0.0f32;
            let mut cnt = 0;

            // cross(up, right)
            if let (Some(a), Some(b)) = (&v_up, &v_right) {
                let c = cross(a, b);
                let len = (c[0]*c[0] + c[1]*c[1] + c[2]*c[2]).sqrt();
                if len > 0.0 { nx += c[0]/len; ny += c[1]/len; nz += c[2]/len; cnt += 1; }
            }
            // cross(right, down)
            if let (Some(a), Some(b)) = (&v_right, &v_down) {
                let c = cross(a, b);
                let len = (c[0]*c[0] + c[1]*c[1] + c[2]*c[2]).sqrt();
                if len > 0.0 { nx += c[0]/len; ny += c[1]/len; nz += c[2]/len; cnt += 1; }
            }
            // cross(down, left)
            if let (Some(a), Some(b)) = (&v_down, &v_left) {
                let c = cross(a, b);
                let len = (c[0]*c[0] + c[1]*c[1] + c[2]*c[2]).sqrt();
                if len > 0.0 { nx += c[0]/len; ny += c[1]/len; nz += c[2]/len; cnt += 1; }
            }
            // cross(left, up)
            if let (Some(a), Some(b)) = (&v_left, &v_up) {
                let c = cross(a, b);
                let len = (c[0]*c[0] + c[1]*c[1] + c[2]*c[2]).sqrt();
                if len > 0.0 { nx += c[0]/len; ny += c[1]/len; nz += c[2]/len; cnt += 1; }
            }

            if cnt > 0 {
                let len = (nx*nx + ny*ny + nz*nz).sqrt();
                if len > 0.0 {
                    normals[py * w + px] = PointNormal { x: nx/len, y: ny/len, z: nz/len };
                }
            }
        }
    }
    normals
}

fn compute_units(points: &[CompilePoint], size_x: usize, size_y: usize) -> Vec<MapUnit> {
    let stride = size_x + 1;
    let mut units = Vec::with_capacity(size_x * size_y);

    for y in 0..size_y {
        for x in 0..size_x {
            let p0 = points[y * stride + x];
            let p1 = points[y * stride + x + 1];
            let p3 = points[(y + 1) * stride + x];
            let p2 = points[(y + 1) * stride + x + 1];

            let flags = p0.flags;
            let flat = p0.z == p1.z && p0.z == p2.z && p0.z == p3.z;

            let (a1, b1, c1, a2, b2, c2) = if flat {
                (p0.z, 0.0, 0.0, p0.z, 0.0, 0.0)
            } else {
                (
                    (p1.z - p0.z) / GLOBAL_SCALE,
                    (p2.z - p1.z) / GLOBAL_SCALE,
                    p0.z,
                    (p2.z - p3.z) / GLOBAL_SCALE,
                    (p3.z - p0.z) / GLOBAL_SCALE,
                    p0.z,
                )
            };

            units.push(MapUnit {
                flags: if flat { flags | CELLFLAG_FLAT } else { flags & !CELLFLAG_FLAT },
                a1,
                b1,
                c1,
                a2,
                b2,
                c2,
            });
        }
    }

    units
}

fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Parses DATA_OBJECTS columns from the CMAP (MatrixMapPrepare.cpp:513-576).
/// Each column is a flat buffer of N elements (one per object), accessed as row 0.
/// The `Height` column, when present, is an offset above interpolated terrain z at
/// the object's (x, y); otherwise `Z` is used directly.
fn load_objects(
    stor: &Storage,
    points: &[CompilePoint],
    size_x: usize,
    size_y: usize,
) -> Vec<ObjectInstance> {
    let Some(col_x) = stor.get_buf("objects", "X") else { return Vec::new(); };
    let Some(col_y) = stor.get_buf("objects", "Y") else { return Vec::new(); };
    let Some(col_scale) = stor.get_buf("objects", "Scale") else { return Vec::new(); };
    let Some(col_type) = stor.get_buf("objects", "Type") else { return Vec::new(); };
    let Some(col_angle_z) = stor.get_buf("objects", "Angle") else { return Vec::new(); };
    if col_x.arrays_count() == 0 { return Vec::new(); }

    let xs = read_f32_array(col_x.get_bytes(0));
    let ys = read_f32_array(col_y.get_bytes(0));
    let n = xs.len().min(ys.len());
    if n == 0 { return Vec::new(); }

    let scales = read_f32_array(col_scale.get_bytes(0));
    let angles_z = read_f32_array(col_angle_z.get_bytes(0));
    let types = read_u32_array(col_type.get_bytes(0));
    let angles_x = stor.get_buf("objects", "AngleX").map(|b| read_f32_array(b.get_bytes(0)));
    let angles_y = stor.get_buf("objects", "AngleY").map(|b| read_f32_array(b.get_bytes(0)));
    let heights = stor.get_buf("objects", "Height").map(|b| read_f32_array(b.get_bytes(0)));
    let zs = stor.get_buf("objects", "Z").map(|b| read_f32_array(b.get_bytes(0)));

    let w = size_x + 1;
    let sample_corners = |x: f32, y: f32| -> f32 {
        let px = (x / GLOBAL_SCALE) as i32;
        let py = (y / GLOBAL_SCALE) as i32;
        let mut sum = 0.0f32;
        let mut cnt = 0u32;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let cx = px + dx;
            let cy = py + dy;
            if cx < 0 || cy < 0 || cx > size_x as i32 || cy > size_y as i32 { continue; }
            sum += points[cy as usize * w + cx as usize].z;
            cnt += 1;
        }
        if cnt == 0 { 0.0 } else { sum / cnt as f32 }
    };

    (0..n).map(|i| {
        let x = xs[i];
        let y = ys[i];
        let z = match (&heights, &zs) {
            (Some(h), _) if i < h.len() => h[i] + sample_corners(x, y),
            (_, Some(zv)) if i < zv.len() => zv[i],
            _ => 0.0,
        };
        ObjectInstance {
            x,
            y,
            z,
            angle_z: angles_z.get(i).copied().unwrap_or(0.0),
            angle_x: angles_x.as_ref().and_then(|a| a.get(i).copied()).unwrap_or(0.0),
            angle_y: angles_y.as_ref().and_then(|a| a.get(i).copied()).unwrap_or(0.0),
            scale: scales.get(i).copied().unwrap_or(1.0),
            type_id: types.get(i).copied().unwrap_or(0),
        }
    }).collect()
}

fn read_f32_array(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_u32_array(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
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
