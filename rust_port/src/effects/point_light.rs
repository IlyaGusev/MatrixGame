use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RenderVertex {
    position: [f32; 3],
    alpha: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    color: [f32; 4],
}

#[derive(Clone, Copy)]
struct TempVertex {
    position: [f32; 3],
    outside: bool,
    alpha: f32,
    index: i32,
}

struct LightBatch {
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    bind_group: wgpu::BindGroup,
    color: [f32; 4],
}

pub struct PointLightRenderer {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    batches: Vec<LightBatch>,
    last_revision: u64,
}

impl PointLightRenderer {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PointLight BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PointLight Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PointLight Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PointLight Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RenderVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            batches: Vec::new(),
            last_revision: 0,
        }
    }

    pub fn sync(
        &mut self,
        device: &wgpu::Device,
        map: &GameMap,
        point_lights: &PointLightSystem,
    ) {
        let revision = point_lights.revision();
        if revision == self.last_revision {
            return;
        }

        self.batches.clear();
        for light in point_lights.lights() {
            let Some((vertices, indices)) = build_light_mesh(map, light) else {
                continue;
            };
            let color = color_to_rgba(light.color);
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PointLight UB"),
                contents: bytemuck::bytes_of(&Uniforms {
                    view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    color,
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PointLight BG"),
                layout: &self.bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PointLight VB"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PointLight IB"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            self.batches.push(LightBatch {
                uniform_buffer,
                vertex_buffer: vb,
                index_buffer: ib,
                num_indices: indices.len() as u32,
                bind_group,
                color,
            });
        }

        self.last_revision = revision;
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        view_proj: glam::Mat4,
    ) {
        if self.batches.is_empty() {
            return;
        }

        pass.set_pipeline(&self.pipeline);
        for batch in &self.batches {
            queue.write_buffer(&batch.uniform_buffer, 0, bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                color: batch.color,
            }));
            pass.set_bind_group(0, &batch.bind_group, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.slice(..));
            pass.set_index_buffer(batch.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..batch.num_indices, 0, 0..1);
        }
    }
}

fn build_light_mesh(map: &GameMap, light: &PointLight) -> Option<(Vec<RenderVertex>, Vec<u32>)> {
    let left = float2int((light.pos[0] - light.radius) / GLOBAL_SCALE) - 1;
    let top = float2int((light.pos[1] - light.radius) / GLOBAL_SCALE) - 1;
    let right = 1 + float2int((light.pos[0] + light.radius) / GLOBAL_SCALE);
    let bottom = 1 + float2int((light.pos[1] + light.radius) / GLOBAL_SCALE);
    let width = (right - left + 1).max(0) as usize;
    let height = (bottom - top + 1).max(0) as usize;
    if width < 2 || height < 2 {
        return None;
    }

    let radius_inv = 1.0 / light.radius.max(0.001);
    let mut temp = vec![
        TempVertex {
            position: [0.0, 0.0, WATER_LEVEL],
            outside: true,
            alpha: 0.0,
            index: -1,
        };
        width * height
    ];

    for gy in 0..height {
        for gx in 0..width {
            let x = left + gx as i32;
            let y = top + gy as i32;
            let idx = gy * width + gx;
            let world_x = x as f32 * GLOBAL_SCALE;
            let world_y = y as f32 * GLOBAL_SCALE;
            let world_z = if x >= 0 && y >= 0 && x <= map.size_x as i32 && y <= map.size_y as i32 {
                (map.point(x as usize, y as usize).z + POINTLIGHT_ALTITUDE).max(WATER_LEVEL)
            } else {
                WATER_LEVEL
            };

            let dx = (light.pos[0] - world_x) * radius_inv;
            let dy = (light.pos[1] - world_y) * radius_inv;
            let dz = (light.pos[2] - world_z) * radius_inv;
            let lum = (1.0 - (dx * dx + dy * dy + dz * dz)).max(0.0);

            temp[idx] = TempVertex {
                position: [
                    world_x - map.world_width() * 0.5,
                    world_y - map.world_height() * 0.5,
                    world_z,
                ],
                outside: lum <= 0.0,
                alpha: lum.clamp(0.0, 1.0),
                index: -1,
            };
        }
    }

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for gy in 0..(height - 1) {
        for gx in 0..(width - 1) {
            let i0 = gy * width + gx;
            let i1 = gy * width + gx + 1;
            let i2 = (gy + 1) * width + gx;
            let i3 = (gy + 1) * width + gx + 1;
            if temp[i0].outside && temp[i1].outside && temp[i2].outside && temp[i3].outside {
                continue;
            }

            let v0 = ensure_vertex(&mut temp, &mut vertices, i2);
            let v1 = ensure_vertex(&mut temp, &mut vertices, i0);
            let v2 = ensure_vertex(&mut temp, &mut vertices, i3);
            let v3 = ensure_vertex(&mut temp, &mut vertices, i1);

            indices.extend_from_slice(&[v0, v1, v2, v1, v3, v2]);
        }
    }

    if indices.is_empty() {
        None
    } else {
        Some((vertices, indices))
    }
}

fn ensure_vertex(temp: &mut [TempVertex], vertices: &mut Vec<RenderVertex>, idx: usize) -> u32 {
    if temp[idx].index >= 0 {
        return temp[idx].index as u32;
    }
    let out_idx = vertices.len() as u32;
    let v = temp[idx];
    vertices.push(RenderVertex {
        position: v.position,
        alpha: v.alpha,
    });
    temp[idx].index = out_idx as i32;
    out_idx
}

fn color_to_rgba(color: u32) -> [f32; 4] {
    let a = ((color >> 24) & 0xFF) as f32 / 255.0;
    let r = ((color >> 16) & 0xFF) as f32 / 255.0;
    let g = ((color >> 8) & 0xFF) as f32 / 255.0;
    let b = (color & 0xFF) as f32 / 255.0;
    [r, g, b, a]
}

fn float2int(x: f32) -> i32 {
    x.round() as i32
}

const SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) alpha: f32,
};
struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) alpha: f32,
};

@vertex fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip_pos = u.view_proj * vec4<f32>(v.position, 1.0);
    out.alpha = v.alpha * u.color.a;
    return out;
}

@fragment fn fs_main(v: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(u.color.rgb, v.alpha);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::common::CELLFLAG_LAND;
    use crate::game::map::{CompilePoint, MapUnit, ObjectInstance, PointNormal};

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
