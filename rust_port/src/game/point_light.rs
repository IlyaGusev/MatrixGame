use crate::game::map::{GameMap, GLOBAL_SCALE};

const WATER_LEVEL: f32 = -2.0;
const POINTLIGHT_ALTITUDE: f32 = 0.7;

pub type PointLightId = u64;

#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    pub id: PointLightId,
    pub pos: [f32; 3],
    pub radius: f32,
    pub color: u32,
}

pub struct PointLightSystem {
    lights: Vec<PointLight>,
    point_lums: Vec<[i32; 3]>,
    next_id: PointLightId,
    revision: u64,
}

impl PointLightSystem {
    pub fn new(map: &GameMap) -> Self {
        Self {
            lights: Vec::new(),
            point_lums: vec![[0, 0, 0]; map.points.len()],
            next_id: 1,
            revision: 0,
        }
    }

    pub fn add_light(
        &mut self,
        map: &GameMap,
        pos: [f32; 3],
        radius: f32,
        color: u32,
    ) -> PointLightId {
        let id = self.next_id;
        self.next_id += 1;
        self.lights.push(PointLight {
            id,
            pos,
            radius: radius.max(0.001),
            color,
        });
        self.recompute(map);
        id
    }

    pub fn remove_light(&mut self, map: &GameMap, id: PointLightId) -> bool {
        let old_len = self.lights.len();
        self.lights.retain(|light| light.id != id);
        let removed = self.lights.len() != old_len;
        if removed {
            self.recompute(map);
        }
        removed
    }

    pub fn set_pos(&mut self, map: &GameMap, id: PointLightId, pos: [f32; 3]) -> bool {
        let Some(light) = self.lights.iter_mut().find(|light| light.id == id) else {
            return false;
        };
        light.pos = pos;
        self.recompute(map);
        true
    }

    pub fn set_radius(&mut self, map: &GameMap, id: PointLightId, radius: f32) -> bool {
        let Some(light) = self.lights.iter_mut().find(|light| light.id == id) else {
            return false;
        };
        light.radius = radius.max(0.001);
        self.recompute(map);
        true
    }

    pub fn set_color(&mut self, map: &GameMap, id: PointLightId, color: u32) -> bool {
        let Some(light) = self.lights.iter_mut().find(|light| light.id == id) else {
            return false;
        };
        light.color = color;
        self.recompute(map);
        true
    }

    pub fn lights(&self) -> &[PointLight] {
        &self.lights
    }

    pub fn point_lum(&self, x: usize, y: usize, size_x: usize) -> [i32; 3] {
        self.point_lums[y * (size_x + 1) + x]
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn recompute(&mut self, map: &GameMap) {
        if self.point_lums.len() != map.points.len() {
            self.point_lums.resize(map.points.len(), [0, 0, 0]);
        }
        for lum in &mut self.point_lums {
            *lum = [0, 0, 0];
        }

        let stride = map.size_x + 1;
        for light in &self.lights {
            let radius_inv = 1.0 / light.radius.max(0.001);
            let left = float2int((light.pos[0] - light.radius) / GLOBAL_SCALE) - 1;
            let top = float2int((light.pos[1] - light.radius) / GLOBAL_SCALE) - 1;
            let right = 1 + float2int((light.pos[0] + light.radius) / GLOBAL_SCALE);
            let bottom = 1 + float2int((light.pos[1] + light.radius) / GLOBAL_SCALE);

            for y in top..=bottom {
                for x in left..=right {
                    if x < 0 || y < 0 || x > map.size_x as i32 || y > map.size_y as i32 {
                        continue;
                    }

                    let idx = y as usize * stride + x as usize;
                    let point = map.point(x as usize, y as usize);
                    let point_z = (point.z + POINTLIGHT_ALTITUDE).max(WATER_LEVEL);

                    let dx = (light.pos[0] - x as f32 * GLOBAL_SCALE) * radius_inv;
                    let dy = (light.pos[1] - y as f32 * GLOBAL_SCALE) * radius_inv;
                    let dz = (light.pos[2] - point_z) * radius_inv;
                    let lum = 1.0 - (dx * dx + dy * dy + dz * dz);
                    if lum <= 0.0 {
                        continue;
                    }

                    self.point_lums[idx][0] += float2int(lum * ((light.color >> 16) & 0xFF) as f32);
                    self.point_lums[idx][1] += float2int(lum * ((light.color >> 8) & 0xFF) as f32);
                    self.point_lums[idx][2] += float2int(lum * (light.color & 0xFF) as f32);
                }
            }
        }

        self.revision = self.revision.wrapping_add(1);
    }
}

fn float2int(x: f32) -> i32 {
    x.round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::common::CELLFLAG_LAND;
    use crate::game::map::{CompilePoint, GameMap, MapUnit, ObjectInstance, PointNormal};

    fn test_map() -> GameMap {
        GameMap {
            size_x: 1,
            size_y: 1,
            camera_angle: 0.0,
            camera_pos: None,
            tex_union_dim: 16,
            water_color: 0,
            sky_color: 0,
            water_name: String::new(),
            water_normal_len: 1.0,
            light_main_color: 0,
            light_main_color_obj: 0,
            light_main_dir: [0.0, 0.0, -1.0],
            ambient_color_obj: 0,
            ambient_color: 0,
            terrain2object_influence: 0.0,
            terrain2object_target_color: 0,
            macro_texture_path: None,
            macro_texture_size: 1,
            points: vec![
                CompilePoint { move_idx: 0, z: 0.0, b: 16, g: 32, r: 64, flags: CELLFLAG_LAND },
                CompilePoint { move_idx: 0, z: 0.0, b: 16, g: 32, r: 64, flags: CELLFLAG_LAND },
                CompilePoint { move_idx: 0, z: 0.0, b: 16, g: 32, r: 64, flags: CELLFLAG_LAND },
                CompilePoint { move_idx: 0, z: 0.0, b: 16, g: 32, r: 64, flags: CELLFLAG_LAND },
            ],
            normals: vec![PointNormal { x: 0.0, y: 0.0, z: 1.0 }; 4],
            units: vec![MapUnit {
                flags: CELLFLAG_LAND,
                a1: 0.0,
                b1: 0.0,
                c1: 0.0,
                a2: 0.0,
                b2: 0.0,
                c2: 0.0,
            }],
            objects: Vec::<ObjectInstance>::new(),
            group_max_z_land: vec![0.0],
            group_w: 1,
            group_h: 1,
        }
    }

    #[test]
    fn point_light_changes_sampled_color() {
        let map = test_map();
        let base = map.get_color(10.0, 10.0);

        let mut lights = PointLightSystem::new(&map);
        lights.add_light(&map, [10.0, 10.0, 8.0], 30.0, 0x00604020);

        let lit = map.get_color_with_lighting(10.0, 10.0, Some(&lights));
        assert!(lit != base, "expected point light to affect sampled color");
    }
}
