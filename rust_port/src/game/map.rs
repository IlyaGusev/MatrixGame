//! Map loader — reads terrain heightmap and properties from .CMAP files.

use anyhow::{bail, Context, Result};

use crate::assets::storage::Storage;
use crate::effects::point_light::PointLightSystem;
use crate::game::common::{
    CELLFLAG_BRIDGE, CELLFLAG_FLAT, CELLFLAG_LAND, CELLFLAG_WATER, MAP_GROUP_SIZE,
};

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
    pub camera_angle: f32,
    pub camera_pos: Option<[f32; 2]>,
    pub tex_union_dim: usize,
    pub water_color: u32,
    pub sky_color: u32,
    pub sky_name: String,
    pub sky_angle: f32,
    pub water_name: String,
    pub water_normal_len: f32,
    pub light_main_color: u32,
    pub light_main_color_obj: u32,
    pub light_main_dir: [f32; 3],
    pub ambient_color_obj: u32,
    pub ambient_color: u32,
    pub terrain2object_influence: f32,
    pub terrain2object_target_color: u32,
    /// Foamline tint — DATA_INSHOREWAVECOLOR, used by the inshore wave pass
    /// (MatrixMapPrepare.cpp:1183, MatrixWater.cpp:160).
    pub inshorewave_color: u32,
    /// Precomputed inshore wave spawn points, one list per map group in
    /// row-major order. Loaded from `inshores/{X,Y,NX,NY}` in the CMAP
    /// (MatrixMapPrepare.cpp:1386-1417). Groups without water get an empty
    /// vec. `pos` is uncentered world coords, `dir` points toward land.
    pub inshore_prespawns: Vec<Vec<InshorePreSpawn>>,
    pub macro_texture_path: Option<String>,
    pub macro_texture_size: i32,   // m_MacrotextureSize from "SIM" param
    pub points: Vec<CompilePoint>, // (size_x+1) * (size_y+1) points
    pub normals: Vec<PointNormal>, // same size, computed from heights
    pub units: Vec<MapUnit>,       // size_x * size_y cells, ports SMatrixMapUnit GetZ data
    pub objects: Vec<ObjectInstance>, // palms / rocks / decorative scenery placements
    /// Per-group max LAND z, size = group_w * group_h.
    /// Ports GetGroupMaxZLand (MatrixMap.hpp:759): returns 0 for empty groups /
    /// groups whose max is negative. Used by the camera to keep the link-point
    /// above mountains (MatrixCamera.cpp:926 → GetZInterpolatedLand).
    pub group_max_z_land: Vec<f32>,
    pub group_w: usize,
    pub group_h: usize,
}

/// Precomputed shoreline wave spawn point. Ports SPreInshorewave fields
/// (MatrixWater.hpp — see MatrixMapPrepare.cpp:1397-1414 for the load path).
#[derive(Debug, Clone, Copy)]
pub struct InshorePreSpawn {
    pub pos: [f32; 2],
    pub dir: [f32; 2],
}

/// A static object placement — ports the per-instance fields of DATA_OBJECTS
/// (MatrixMapPrepare.cpp:503-619). Position is uncentered world space (raw map
/// coords); renderer applies the map-center offset.
#[derive(Debug, Clone)]
pub struct ObjectInstance {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub angle_z: f32,
    pub angle_x: f32,
    pub angle_y: f32,
    pub scale: f32,
    pub type_id: u32,
    pub shadow: Option<ObjectShadow>,
}

#[derive(Debug, Clone)]
pub struct ObjectShadow {
    pub vertices: Vec<ObjectShadowVertex>,
    pub indices: Vec<u32>,
    pub camera_pos: [f32; 3],
    pub dimensions: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct ObjectShadowVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
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

        let tex_union_dim =
            find_property_int(prop_names, prop_values, "TexUnionDim").unwrap_or(16) as usize;
        let camera_angle = find_property_float(prop_names, prop_values, "CamAngle").unwrap_or(0.0);
        let camera_pos = match (
            find_property_float(prop_names, prop_values, "CamPosX"),
            find_property_float(prop_names, prop_values, "CamPosY"),
        ) {
            (Ok(x), Ok(y)) => Some([x, y]),
            _ => None,
        };
        let water_color =
            find_property_int(prop_names, prop_values, "WaterColor").unwrap_or(0x003060) as u32;
        // DEF_SKY_COLOR (Common.hpp:27) = 0x1070FF
        let sky_color =
            find_property_int(prop_names, prop_values, "SkyColor").unwrap_or(0x1070FF) as u32;
        let sky_name = prop_names
            .find_as_wstr("SkyName")
            .map(|idx| prop_values.get_as_wstr(idx))
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "Default".to_string());
        // DATA_SKYANGLE is an offset in radians applied on top of the sky
        // config block's base angle (MatrixMapPrepare.cpp:1170-1174).
        let sky_angle =
            find_property_float(prop_names, prop_values, "SkyAngle").unwrap_or(0.0);
        let water_name = prop_names
            .find_as_wstr("WaterName")
            .map(|idx| prop_values.get_as_wstr(idx))
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "water_blue".to_string());
        let water_normal_len =
            find_property_float(prop_names, prop_values, "WaterNormLen").unwrap_or(1.0);
        let light_main_color =
            find_property_int(prop_names, prop_values, "LightMainColor").unwrap_or(0x989898) as u32;
        let light_main_color_obj = find_property_int(prop_names, prop_values, "LightMainColorObj")
            .unwrap_or(light_main_color as i32) as u32;
        let ambient_color_obj = find_property_int(prop_names, prop_values, "AmbientColorObj")
            .unwrap_or(0x808080) as u32;
        let ambient_color =
            find_property_int(prop_names, prop_values, "AmbientColor").unwrap_or(0x808080) as u32;
        let mut terrain2object_influence =
            find_property_float(prop_names, prop_values, "Influence").unwrap_or(0.0);
        let inshorewave_color = find_property_int(prop_names, prop_values, "InshorewaveColor")
            .unwrap_or(0x008080) as u32;
        let terrain2object_target_color = if terrain2object_influence > 0.0 {
            0xFFFFFF
        } else if terrain2object_influence < 0.0 {
            terrain2object_influence = terrain2object_influence.abs();
            0x000000
        } else {
            0x000000
        };
        terrain2object_influence = terrain2object_influence.clamp(0.0, 1.0);

        // Light direction: RotX(angleX) * RotZ(angleZ) * (0, 0, -1)
        // Ports MatrixMapPrepare.cpp lines 1228-1231
        let angle_x =
            find_property_float(prop_names, prop_values, "LightMainAngleX").unwrap_or(0.61);
        let angle_z =
            find_property_float(prop_names, prop_values, "LightMainAngleZ").unwrap_or(-1.75);
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

        log::info!(
            "map: size = {}x{} units, TexUnionDim={}",
            size_x,
            size_y,
            tex_union_dim
        );

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

        let objects = load_objects(&stor, &points, &normals, size_x, size_y);
        log::info!("map: loaded {} decorative objects", objects.len());

        // No dilation: the stored per-group max is the max of THIS group's
        // land-cell corner z only. Dilation used to smear mountain height
        // into neighboring water groups, pushing the camera up while over
        // water and making the heavily-fogged far water read as sky color.
        let (group_max_z_land, group_w, group_h) =
            compute_group_max_z_land(&points, &units, size_x, size_y);

        let inshore_prespawns = load_inshore_prespawns(&stor, group_w, group_h);
        let total_inshores: usize = inshore_prespawns.iter().map(|v| v.len()).sum();
        log::info!(
            "map: loaded {} inshore wave spawn points across {} groups",
            total_inshores,
            inshore_prespawns.iter().filter(|v| !v.is_empty()).count()
        );

        Ok(Self {
            size_x,
            size_y,
            camera_angle,
            camera_pos,
            tex_union_dim,
            water_color,
            sky_color,
            sky_name,
            sky_angle,
            water_name,
            water_normal_len,
            light_main_color,
            light_main_color_obj,
            light_main_dir,
            ambient_color_obj,
            ambient_color,
            terrain2object_influence,
            terrain2object_target_color,
            inshorewave_color,
            inshore_prespawns,
            macro_texture_path,
            macro_texture_size,
            points,
            normals,
            units,
            objects,
            group_max_z_land,
            group_w,
            group_h,
        })
    }

    /// Max land-corner z of the group containing world (wx, wy), clamped to 0.
    /// Ports GetGroupMaxZLand (MatrixMap.hpp:759). Returns 0 for out-of-bounds.
    pub fn group_max_z_land_at(&self, wx: f32, wy: f32) -> f32 {
        let gs = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE;
        let gx = (wx / gs).floor() as i32;
        let gy = (wy / gs).floor() as i32;
        if gx < 0 || gy < 0 || gx >= self.group_w as i32 || gy >= self.group_h as i32 {
            return 0.0;
        }
        self.group_max_z_land[gy as usize * self.group_w + gx as usize]
    }

    /// Bilinear-interpolated per-group max z in GROUP-CENTER coordinates.
    /// Ports the smooth-over-neighbors idea of GetZInterpolatedLand
    /// (MatrixMap.cpp:426-469) — a B-spline there, bilinear here. The returned
    /// value is clamped to `ceiling` so the camera never rises unreasonably
    /// high over tall terrain (original clamps per-group z to
    /// `[m_GroundZBaseMiddle, m_GroundZBaseMax]` from buildings).
    pub fn group_max_z_interpolated(&self, wx: f32, wy: f32, ceiling: f32) -> f32 {
        let gs = MAP_GROUP_SIZE as f32 * GLOBAL_SCALE;
        // Group centers are at ((gx + 0.5) * gs, (gy + 0.5) * gs).
        let fx = wx / gs - 0.5;
        let fy = wy / gs - 0.5;
        let gx0 = fx.floor() as i32;
        let gy0 = fy.floor() as i32;
        let tx = fx - gx0 as f32;
        let ty = fy - gy0 as f32;

        let sample = |gx: i32, gy: i32| -> f32 {
            if gx < 0 || gy < 0 || gx >= self.group_w as i32 || gy >= self.group_h as i32 {
                return 0.0;
            }
            let v = self.group_max_z_land[gy as usize * self.group_w + gx as usize];
            v.min(ceiling)
        };
        let a = sample(gx0, gy0);
        let b = sample(gx0 + 1, gy0);
        let c = sample(gx0, gy0 + 1);
        let d = sample(gx0 + 1, gy0 + 1);
        let ab = a + (b - a) * tx;
        let cd = c + (d - c) * tx;
        ab + (cd - ab) * ty
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

    pub fn get_color(&self, wx: f32, wy: f32) -> u32 {
        self.get_color_with_lighting(wx, wy, None)
    }

    /// Port of `CMatrixMap::GetColor`, including optional point-light
    /// luminance contributions when a point-light system is present.
    pub fn get_color_with_lighting(
        &self,
        wx: f32,
        wy: f32,
        point_lights: Option<&PointLightSystem>,
    ) -> u32 {
        let sx = wx / GLOBAL_SCALE;
        let sy = wy / GLOBAL_SCALE;
        let x = sx.floor() as i32;
        let y = sy.floor() as i32;

        if x < 0 || y < 0 || x >= self.size_x as i32 || y >= self.size_y as i32 {
            return self.ambient_color_obj;
        }

        let unit = self.unit(x as usize, y as usize);
        if unit.flags & CELLFLAG_WATER != 0 && unit.flags & CELLFLAG_BRIDGE == 0 {
            return self.ambient_color_obj;
        }

        let kx = sx - x as f32;
        let ky = sy - y as f32;

        let c00 = self.point(x as usize, y as usize);
        let c10 = self.point(x as usize + 1, y as usize);
        let c01 = self.point(x as usize, y as usize + 1);
        let c11 = self.point(x as usize + 1, y as usize + 1);
        let l00 = point_lights
            .map(|lights| lights.point_lum(x as usize, y as usize, self.size_x))
            .unwrap_or([0, 0, 0]);
        let l10 = point_lights
            .map(|lights| lights.point_lum(x as usize + 1, y as usize, self.size_x))
            .unwrap_or([0, 0, 0]);
        let l01 = point_lights
            .map(|lights| lights.point_lum(x as usize, y as usize + 1, self.size_x))
            .unwrap_or([0, 0, 0]);
        let l11 = point_lights
            .map(|lights| lights.point_lum(x as usize + 1, y as usize + 1, self.size_x))
            .unwrap_or([0, 0, 0]);

        let sample =
            |c00: u8, c10: u8, c01: u8, c11: u8, l00: i32, l10: i32, l01: i32, l11: i32| -> u32 {
                // MatrixMap.cpp:
                //   Float2Int(LERPFLOAT(ky, LERPFLOAT(kx, c00, c10), LERPFLOAT(kx, c01, c11)))
                // using the per-point color plus runtime luminance.
                let top_left = c00 as i32 + l00;
                let top_right = c10 as i32 + l10;
                let bottom_left = c01 as i32 + l01;
                let bottom_right = c11 as i32 + l11;
                let top = top_left as f32 + (top_right - top_left) as f32 * kx;
                let bottom = bottom_left as f32 + (bottom_right - bottom_left) as f32 * kx;
                float2int(top + (bottom - top) * ky).clamp(0, 255) as u32
            };

        let r = sample(c00.r, c10.r, c01.r, c11.r, l00[0], l10[0], l01[0], l11[0]);
        let g = sample(c00.g, c10.g, c01.g, c11.g, l00[1], l10[1], l01[1], l11[1]);
        let b = sample(c00.b, c10.b, c01.b, c11.b, l00[2], l10[2], l01[2], l11[2]);

        (r << 16) | (g << 8) | b
    }

    /// Port of `CMatrixMap::GetNormal` (MatrixMap.cpp:636) — bilinearly
    /// interpolates point normals with two opt-in special cases:
    ///
    /// * Bridge cells derive their normal from finite differences of `GetZ`
    ///   (half a cell apart in each axis) so decals sit on the bridge plane
    ///   rather than the terrain below.
    /// * When `check_water` is true and any of the four surrounding heightmap
    ///   points dip below sea level, we return straight up so shore surfaces
    ///   flatten out instead of following submerged normals.
    pub fn get_normal(&self, wx: f32, wy: f32, check_water: bool) -> [f32; 3] {
        let scaled_x = wx / GLOBAL_SCALE;
        let scaled_y = wy / GLOBAL_SCALE;
        let x = scaled_x.floor() as i32;
        let y = scaled_y.floor() as i32;

        if x < 0 || y < 0 || x >= self.size_x as i32 || y >= self.size_y as i32 {
            return [0.0, 0.0, 1.0];
        }

        let unit = self.unit(x as usize, y as usize);
        if unit.flags & CELLFLAG_FLAT != 0 {
            return [0.0, 0.0, 1.0];
        }

        if unit.flags & CELLFLAG_BRIDGE != 0 {
            let half = GLOBAL_SCALE * 0.5;
            let nx = self.get_z(wx - half, wy) - self.get_z(wx + half, wy);
            let ny = self.get_z(wx, wy - half) - self.get_z(wx, wy + half);
            let nz = GLOBAL_SCALE;
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
            return [nx / len, ny / len, nz / len];
        }

        if unit.flags & CELLFLAG_WATER != 0 {
            return [0.0, 0.0, 1.0];
        }

        if check_water {
            let p00 = self.point(x as usize, y as usize);
            let p10 = self.point(x as usize + 1, y as usize);
            let p01 = self.point(x as usize, y as usize + 1);
            let p11 = self.point(x as usize + 1, y as usize + 1);
            if p00.z < 0.0 || p10.z < 0.0 || p01.z < 0.0 || p11.z < 0.0 {
                return [0.0, 0.0, 1.0];
            }
        }

        let kx = scaled_x - x as f32;
        let ky = scaled_y - y as f32;
        let n00 = self.normal(x as usize, y as usize);
        let n10 = self.normal(x as usize + 1, y as usize);
        let n01 = self.normal(x as usize, y as usize + 1);
        let n11 = self.normal(x as usize + 1, y as usize + 1);

        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let x0 = lerp(n00.x, n10.x, kx);
        let y0 = lerp(n00.y, n10.y, kx);
        let z0 = lerp(n00.z, n10.z, kx);
        let x1 = lerp(n01.x, n11.x, kx);
        let y1 = lerp(n01.y, n11.y, kx);
        let z1 = lerp(n01.z, n11.z, kx);
        let nx = lerp(x0, x1, ky);
        let ny = lerp(y0, y1, ky);
        let nz = lerp(z0, z1, ky);
        let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
        [nx / len, ny / len, nz / len]
    }

    pub fn static_object_color(&self, wx: f32, wy: f32) -> u32 {
        self.static_object_color_with_lighting(wx, wy, None)
    }

    pub fn static_object_color_with_lighting(
        &self,
        wx: f32,
        wy: f32,
        point_lights: Option<&PointLightSystem>,
    ) -> u32 {
        blend_color(
            self.get_color_with_lighting(wx, wy, point_lights),
            self.terrain2object_target_color,
            self.terrain2object_influence,
        )
    }
}

const CELLFLAG_DOWN: u8 = 1 << 5;

/// Ports PointCalcNormals (MatrixMapPrepare.cpp:20-105).
/// For each point, computes normal from cross products of 4 adjacent triangles.
fn compute_normals(points: &[CompilePoint], size_x: usize, size_y: usize) -> Vec<PointNormal> {
    let w = size_x + 1;
    let h = size_y + 1;
    let mut normals = vec![
        PointNormal {
            x: 0.0,
            y: 0.0,
            z: 1.0
        };
        w * h
    ];

    let get_z = |x: usize, y: usize| -> f32 { points[y * w + x].z };
    let get_flags = |x: usize, y: usize| -> u8 { points[y * w + x].flags };

    // Check if unit at (ux,uy) is land — unit flags are on point (ux,uy)
    let is_land = |ux: i32, uy: i32| -> bool {
        if ux < 0 || uy < 0 || ux >= size_x as i32 || uy >= size_y as i32 {
            return false;
        }
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

            let p0 = [
                px as f32 * GLOBAL_SCALE,
                py as f32 * GLOBAL_SCALE,
                get_z(px, py),
            ];

            // Vectors to neighbors (relative to p0)
            let v_up = if has_up && py > 0 {
                Some([0.0, -GLOBAL_SCALE, get_z(px, py - 1) - p0[2]])
            } else {
                None
            };

            let v_right = if has_cur && px < size_x {
                Some([GLOBAL_SCALE, 0.0, get_z(px + 1, py) - p0[2]])
            } else {
                None
            };

            let v_down = if has_cur && py < size_y {
                Some([0.0, GLOBAL_SCALE, get_z(px, py + 1) - p0[2]])
            } else {
                None
            };

            let v_left = if has_left && px > 0 {
                Some([-GLOBAL_SCALE, 0.0, get_z(px - 1, py) - p0[2]])
            } else {
                None
            };

            // Average cross products of adjacent triangle pairs
            let mut nx = 0.0f32;
            let mut ny = 0.0f32;
            let mut nz = 0.0f32;
            let mut cnt = 0;

            // cross(up, right)
            if let (Some(a), Some(b)) = (&v_up, &v_right) {
                let c = cross(a, b);
                let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if len > 0.0 {
                    nx += c[0] / len;
                    ny += c[1] / len;
                    nz += c[2] / len;
                    cnt += 1;
                }
            }
            // cross(right, down)
            if let (Some(a), Some(b)) = (&v_right, &v_down) {
                let c = cross(a, b);
                let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if len > 0.0 {
                    nx += c[0] / len;
                    ny += c[1] / len;
                    nz += c[2] / len;
                    cnt += 1;
                }
            }
            // cross(down, left)
            if let (Some(a), Some(b)) = (&v_down, &v_left) {
                let c = cross(a, b);
                let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if len > 0.0 {
                    nx += c[0] / len;
                    ny += c[1] / len;
                    nz += c[2] / len;
                    cnt += 1;
                }
            }
            // cross(left, up)
            if let (Some(a), Some(b)) = (&v_left, &v_up) {
                let c = cross(a, b);
                let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if len > 0.0 {
                    nx += c[0] / len;
                    ny += c[1] / len;
                    nz += c[2] / len;
                    cnt += 1;
                }
            }

            if cnt > 0 {
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                if len > 0.0 {
                    normals[py * w + px] = PointNormal {
                        x: nx / len,
                        y: ny / len,
                        z: nz / len,
                    };
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
                flags: if flat {
                    flags | CELLFLAG_FLAT
                } else {
                    flags & !CELLFLAG_FLAT
                },
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
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Per-group max land z, clamped to 0 (ports GetGroupMaxZLand + CMatrixMapGroup
/// max-z tracking; negative underwater groups report 0 like the original).
fn compute_group_max_z_land(
    points: &[CompilePoint],
    units: &[MapUnit],
    size_x: usize,
    size_y: usize,
) -> (Vec<f32>, usize, usize) {
    let gs = MAP_GROUP_SIZE as usize;
    let gw = (size_x + gs - 1) / gs;
    let gh = (size_y + gs - 1) / gs;
    let stride = size_x + 1;
    let mut out = vec![0.0f32; gw * gh];

    for gy in 0..gh {
        for gx in 0..gw {
            let x0 = gx * gs;
            let y0 = gy * gs;
            let x1 = (x0 + gs).min(size_x);
            let y1 = (y0 + gs).min(size_y);
            let mut mz = f32::NEG_INFINITY;
            for y in y0..y1 {
                for x in x0..x1 {
                    let u = units[y * size_x + x];
                    if u.flags & CELLFLAG_LAND == 0 {
                        continue;
                    }
                    for (dx, dy) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
                        let z = points[(y + dy) * stride + (x + dx)].z;
                        if z > mz {
                            mz = z;
                        }
                    }
                }
            }
            out[gy * gw + gx] = if mz.is_finite() { mz.max(0.0) } else { 0.0 };
        }
    }
    (out, gw, gh)
}

/// Morphological max-dilation: each cell becomes the max of itself and its
/// neighbors within `radius` cells. Used to expand per-group max-z so a single
/// lookup covers the camera's eye offset.
fn dilate_max(src: &[f32], w: usize, h: usize, radius: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for gy in 0..h {
        for gx in 0..w {
            let mut m = 0.0f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = gx as i32 + dx;
                    let ny = gy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let v = src[ny as usize * w + nx as usize];
                    if v > m {
                        m = v;
                    }
                }
            }
            out[gy * w + gx] = m;
        }
    }
    out
}

/// Parses DATA_OBJECTS columns from the CMAP (MatrixMapPrepare.cpp:513-576).
/// Each column is a flat buffer of N elements (one per object), accessed as row 0.
/// The `Height` column, when present, is an offset above interpolated terrain z at
/// the object's (x, y); otherwise `Z` is used directly.
/// Loads per-group shoreline wave spawn points from the CMAP's `inshores`
/// record (four parallel float columns `X`, `Y`, `NX`, `NY` — one array per
/// group, row-major). Ports MatrixMapPrepare.cpp:1386-1417 + the original
/// `CMatrixMapGroup::InitInshoreWaves` that stores (pos, dir) for later
/// wave spawning.
///
/// If any column is missing we return an empty list per group — the map
/// either has `DisableInshore` set or was built without inshore data.
fn load_inshore_prespawns(
    stor: &Storage,
    group_w: usize,
    group_h: usize,
) -> Vec<Vec<InshorePreSpawn>> {
    let total = group_w * group_h;
    let empty = vec![Vec::new(); total];
    let (Some(bx), Some(by), Some(bnx), Some(bny)) = (
        stor.get_buf("inshores", "X"),
        stor.get_buf("inshores", "Y"),
        stor.get_buf("inshores", "NX"),
        stor.get_buf("inshores", "NY"),
    ) else {
        return empty;
    };
    if bx.arrays_count() < total {
        return empty;
    }

    let read_f32 = |b: &[u8], i: usize| -> f32 {
        f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
    };

    let mut out = Vec::with_capacity(total);
    for gi in 0..total {
        let xs = bx.get_bytes(gi);
        let ys = by.get_bytes(gi);
        let nxs = bnx.get_bytes(gi);
        let nys = bny.get_bytes(gi);
        let count = xs.len().min(ys.len()).min(nxs.len()).min(nys.len()) / 4;
        let mut list = Vec::with_capacity(count);
        for i in 0..count {
            list.push(InshorePreSpawn {
                pos: [read_f32(xs, i), read_f32(ys, i)],
                dir: [read_f32(nxs, i), read_f32(nys, i)],
            });
        }
        out.push(list);
    }
    out
}

fn load_objects(
    stor: &Storage,
    points: &[CompilePoint],
    normals: &[PointNormal],
    size_x: usize,
    size_y: usize,
) -> Vec<ObjectInstance> {
    let Some(col_x) = stor.get_buf("objects", "X") else {
        return Vec::new();
    };
    let Some(col_y) = stor.get_buf("objects", "Y") else {
        return Vec::new();
    };
    let Some(col_scale) = stor.get_buf("objects", "Scale") else {
        return Vec::new();
    };
    let Some(col_type) = stor.get_buf("objects", "Type") else {
        return Vec::new();
    };
    let Some(col_angle_z) = stor.get_buf("objects", "Angle") else {
        return Vec::new();
    };
    if col_x.arrays_count() == 0 {
        return Vec::new();
    }

    let xs = read_f32_array(col_x.get_bytes(0));
    let ys = read_f32_array(col_y.get_bytes(0));
    let n = xs.len().min(ys.len());
    if n == 0 {
        return Vec::new();
    }

    let scales = read_f32_array(col_scale.get_bytes(0));
    let angles_z = read_f32_array(col_angle_z.get_bytes(0));
    let types = read_u32_array(col_type.get_bytes(0));
    let angles_x = stor
        .get_buf("objects", "AngleX")
        .map(|b| read_f32_array(b.get_bytes(0)));
    let angles_y = stor
        .get_buf("objects", "AngleY")
        .map(|b| read_f32_array(b.get_bytes(0)));
    let heights = stor
        .get_buf("objects", "Height")
        .map(|b| read_f32_array(b.get_bytes(0)));
    let zs = stor
        .get_buf("objects", "Z")
        .map(|b| read_f32_array(b.get_bytes(0)));
    let shadows = stor.get_buf("objects", "Shadow");

    let w = size_x + 1;
    let sample_corners = |x: f32, y: f32| -> f32 {
        let px = (x / GLOBAL_SCALE) as i32;
        let py = (y / GLOBAL_SCALE) as i32;
        let mut sum = 0.0f32;
        let mut cnt = 0u32;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let cx = px + dx;
            let cy = py + dy;
            if cx < 0 || cy < 0 || cx > size_x as i32 || cy > size_y as i32 {
                continue;
            }
            sum += points[cy as usize * w + cx as usize].z;
            cnt += 1;
        }
        if cnt == 0 {
            0.0
        } else {
            sum / cnt as f32
        }
    };

    (0..n)
        .map(|i| {
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
                angle_x: angles_x
                    .as_ref()
                    .and_then(|a| a.get(i).copied())
                    .unwrap_or(0.0),
                angle_y: angles_y
                    .as_ref()
                    .and_then(|a| a.get(i).copied())
                    .unwrap_or(0.0),
                scale: scales.get(i).copied().unwrap_or(1.0),
                type_id: types.get(i).copied().unwrap_or(0),
                shadow: shadows
                    .and_then(|buf| parse_object_shadow(buf, i, points, normals, size_x, size_y)),
            }
        })
        .collect()
}

fn parse_object_shadow(
    buf: &crate::assets::storage::DataBuf,
    index: usize,
    points: &[CompilePoint],
    normals: &[PointNormal],
    size_x: usize,
    size_y: usize,
) -> Option<ObjectShadow> {
    if index >= buf.arrays_count() {
        return None;
    }
    let bytes = buf.get_bytes(index);
    if bytes.len() < 4 {
        return None;
    }

    let mut off = 0usize;
    let group_count = read_i32_le(bytes, &mut off)? as usize;
    let group_bytes = group_count.checked_mul(4)?;
    if off + group_bytes > bytes.len() {
        return None;
    }
    off += group_bytes;

    let _min_x = read_i32_le(bytes, &mut off)?;
    let _min_y = read_i32_le(bytes, &mut off)?;
    let vert_count = read_i32_le(bytes, &mut off)? as usize;
    let index_byte_size = read_i32_le(bytes, &mut off)? as usize;
    let index_count = index_byte_size / 2;

    let vertex_bytes = vert_count.checked_mul(12)?;
    let index_bytes = index_byte_size;
    if off + vertex_bytes + index_bytes + 20 > bytes.len() {
        return None;
    }

    let mut vertices = Vec::with_capacity(vert_count);
    for _ in 0..vert_count {
        let px = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as usize;
        let py = u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]) as usize;
        let tu = f32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]);
        let tv = f32::from_le_bytes([
            bytes[off + 8],
            bytes[off + 9],
            bytes[off + 10],
            bytes[off + 11],
        ]);
        off += 12;

        if px > size_x || py > size_y {
            return None;
        }
        let p = points[py * (size_x + 1) + px];
        let n = normals[py * (size_x + 1) + px];
        vertices.push(ObjectShadowVertex {
            position: [
                px as f32 * GLOBAL_SCALE + n.x * 0.1,
                py as f32 * GLOBAL_SCALE + n.y * 0.1,
                p.z + n.z * 0.1,
            ],
            uv: [tu, tv],
        });
    }

    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        indices.push(u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32);
        off += 2;
    }

    let camera_pos = [
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]),
        f32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]),
        f32::from_le_bytes([
            bytes[off + 8],
            bytes[off + 9],
            bytes[off + 10],
            bytes[off + 11],
        ]),
    ];
    off += 12;
    let dimensions = [
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]),
        f32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]),
    ];

    Some(ObjectShadow {
        vertices,
        indices,
        camera_pos,
        dimensions,
    })
}

fn read_i32_le(bytes: &[u8], off: &mut usize) -> Option<i32> {
    if *off + 4 > bytes.len() {
        return None;
    }
    let value = i32::from_le_bytes([
        bytes[*off],
        bytes[*off + 1],
        bytes[*off + 2],
        bytes[*off + 3],
    ]);
    *off += 4;
    Some(value)
}

fn read_f32_array(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32_array(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn blend_color(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |ca: u32, cb: u32| -> u32 {
        ((ca as f32) + ((cb as f32) - (ca as f32)) * t)
            .round()
            .clamp(0.0, 255.0) as u32
    };
    let ar = (a >> 16) & 0xFF;
    let ag = (a >> 8) & 0xFF;
    let ab = a & 0xFF;
    let br = (b >> 16) & 0xFF;
    let bg = (b >> 8) & 0xFF;
    let bb = b & 0xFF;
    (lerp(ar, br) << 16) | (lerp(ag, bg) << 8) | lerp(ab, bb)
}

fn float2int(x: f32) -> i32 {
    // The original uses x87 `fistp` via Float2Int. Rust has no direct stable
    // equivalent, so use nearest-integer rounding for the current viewer port.
    x.round() as i32
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
