//! Minimal chassis-only renderer for `CMatrixRobotAI`.
//!
//! Ports the chassis branch of `CMatrixRobot::RNeed` at
//! MatrixObjectRobot.cpp:299-357: each robot's `m_Unit[MRT_CHASSIS]`
//! calls `LoadObject(Matrix\\Robot\\ChassisN.vo)` where N is the
//! `ERobotUnitKind`. Armor / weapons / head are deferred — they
//! compose additional sub-VOs the original anchors to bones on the
//! chassis, and a faithful port needs the per-frame anchor table
//! from `CVectorObjectAnim::m_Animation`.
//!
//! Instance buffers are rebuilt each frame from the live arena so
//! robots that get spawned mid-session immediately render.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{unpack_rgb, FOG_END, FOG_START};
use crate::matrix_game::effects::point_light::PointLightSystem;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{MapStatic, ObjectType, Objects};
use crate::matrix_game::robot::{Animation, ChassisKind, Robot};
use crate::matrix_lib::three_g::texture::{
    create_solid_texture, create_texture_from_rgba, decode_texture_bytes,
};
use crate::matrix_lib::three_g::vector_object::{self, MaterialSpec};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceData {
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
    row3: [f32; 4],
    terrain_color: [f32; 4],
    /// Unused for robots (no sub-unit animation) — kept only because
    /// we reuse the buildings shader, which reads this attribute.
    unit_offset: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    fog_color: [f32; 4],
    fog_params: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    camera_pos: [f32; 4],
    time_ms: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaterialUniform {
    flags: [u32; 4],
    scroll: [f32; 4],
}

/// One surface draw for one VO frame. VOs have per-frame
/// triangulation (different unions per frame), so each animation
/// frame needs its own index buffer and bind group.
struct SurfaceGpu {
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    bind_group: wgpu::BindGroup,
}

/// One VO-frame's surface list.
struct FrameGpu {
    surfaces: Vec<SurfaceGpu>,
}

/// Per-chassis GPU resources. Vertices are shared across all frames
/// (animation pulls different subsets of the same vertex buffer).
struct ChassisGpu {
    /// `Arc` shared with the global `animation::CHASSIS_VOS` table
    /// so the AI layer (`robot::logic_takt`) can advance animation
    /// without touching the renderer.
    vo_mesh: std::sync::Arc<vector_object::VoMesh>,
    vertex_buffer: wgpu::Buffer,
    frames: Vec<FrameGpu>,
}

/// Per-robot draw ticket, rebuilt from scratch each frame in
/// `sync_robots`. Each ticket emits one indexed draw per chassis
/// surface of the robot's current VO frame.
struct RobotDraw {
    chassis: ChassisKind,
    vo_frame: usize,
    instance_offset: u32,
}

pub struct RobotsRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Indexed by `ChassisKind as usize`; None for kinds whose VO
    /// failed to load.
    chassis: Vec<Option<ChassisGpu>>,
    /// Per-frame instance buffer: one `InstanceData` per live robot.
    /// Written contiguously in `sync_robots`; robots read at offset
    /// `RobotDraw::instance_offset`.
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    draws: Vec<RobotDraw>,
    uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    time_ms: f32,
    /// World-center offset applied to every instance so local-
    /// frame vertices share the terrain renderer's origin
    /// (matches the pattern `BuildingsRenderer` uses).
    center: [f32; 2],
}

const MAX_LIVE_ROBOTS: u32 = 128;

impl RobotsRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        map: &GameMap,
        read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Option<Self> {
        let cx = map.world_width() * 0.5;
        let cy = map.world_height() * 0.5;

        let [sr, sg, sb] = unpack_rgb(map.sky_color);
        let fog_color = [sr, sg, sb, 1.0];
        let [ar, ag, ab] = unpack_rgb(map.ambient_color_obj);
        let [lr, lg, lb] = unpack_rgb(map.light_main_color_obj);
        let ambient_color = [ar, ag, ab, 1.0];
        let light_color = [lr, lg, lb, 1.0];
        let light_dir = [
            map.light_main_dir[0],
            map.light_main_dir[1],
            map.light_main_dir[2],
            0.0,
        ];

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Robots UB"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                ambient_color,
                light_color,
                light_dir,
                camera_pos: [0.0, 0.0, 0.0, 1.0],
                time_ms: [0.0, 0.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = create_bgl(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });
        let pipeline = create_pipeline(device, config, &bgl);

        let mut tex_cache: HashMap<String, wgpu::TextureView> = HashMap::new();
        let fallback_tex = create_solid_texture(device, queue, [200, 200, 200, 255]);
        let black_tex = create_solid_texture(device, queue, [0, 0, 0, 255]);
        let transparent_tex = create_solid_texture(device, queue, [0, 0, 0, 0]);

        let chassis_list = [
            ChassisKind::Pneumatic,
            ChassisKind::Wheel,
            ChassisKind::Track,
            ChassisKind::AntiGravity,
            ChassisKind::Hovercraft,
        ];

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Robots Inst VB"),
            size: (MAX_LIVE_ROBOTS as u64) * std::mem::size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut chassis: Vec<Option<ChassisGpu>> = (0..chassis_list.len()).map(|_| None).collect();
        let mut total_surfaces = 0usize;
        for &ck in &chassis_list {
            let n = chassis_kind_index(ck) as u32 + 1;
            let vo_path = format!("Matrix/Robot/Chassis{}.vo", n);
            let Some(vo_bytes) = read_texture(&vo_path) else {
                log::warn!("robots: VO not found: {}", vo_path);
                continue;
            };
            let vo_mesh = match vector_object::parse_vo(&vo_bytes) {
                Ok(m) => m,
                Err(e) => {
                    log::warn!("robots: parse {} failed: {}", vo_path, e);
                    continue;
                }
            };

            let vertices: Vec<Vertex> = vo_mesh
                .vertices
                .iter()
                .map(|v| Vertex {
                    position: v.position,
                    normal: v.normal,
                    uv: v.uv,
                })
                .collect();
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Robots Mesh VB"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let vo_dir = vo_path.rsplit_once('/').map(|(d, _)| format!("{d}/"));
            let top_diffuse = format!("Matrix/Robot/Chassis{}", n);
            let top_gloss = format!("Matrix/Robot/Chassis{}_gloss", n);

            let mut frames: Vec<FrameGpu> = Vec::with_capacity(vo_mesh.frames.len());
            for frame in &vo_mesh.frames {
                let mut surfaces = Vec::with_capacity(frame.surfaces.len());
                for surf in &frame.surfaces {
                    if surf.indices.is_empty() {
                        continue;
                    }
                    let mut material: MaterialSpec = MaterialSpec {
                        diffuse: Some(top_diffuse.clone()),
                        gloss: Some(top_gloss.clone()),
                        ..Default::default()
                    };
                    if let Some(spec) = surf.texture_ref.as_deref() {
                        let surface_mat =
                            vector_object::parse_material_spec_with_prefix(spec, vo_dir.as_deref());
                        material = vector_object::merge_materials(&surface_mat, Some(&material));
                    }

                    let (diffuse_view, alpha_test) =
                        resolve_diffuse(&material, device, queue, &mut tex_cache, read_texture)
                            .unwrap_or_else(|| (fallback_tex.clone(), false));
                    let gloss_view = resolve_texture(
                        material.gloss.as_ref(),
                        device,
                        queue,
                        &mut tex_cache,
                        read_texture,
                    )
                    .unwrap_or_else(|| black_tex.clone());
                    let back_view = resolve_texture(
                        material.back.as_ref(),
                        device,
                        queue,
                        &mut tex_cache,
                        read_texture,
                    )
                    .unwrap_or_else(|| black_tex.clone());
                    let mask_view = resolve_texture(
                        material.mask.as_ref(),
                        device,
                        queue,
                        &mut tex_cache,
                        read_texture,
                    )
                    .unwrap_or_else(|| transparent_tex.clone());

                    let index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Robots Mesh IB"),
                            contents: bytemuck::cast_slice(&surf.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                    let mat_uniform =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Robots Material UB"),
                            contents: bytemuck::bytes_of(&MaterialUniform {
                                flags: [
                                    material.gloss.is_some() as u32,
                                    material.back.is_some() as u32,
                                    material.mask.is_some() as u32,
                                    alpha_test as u32,
                                ],
                                scroll: [material.scroll[0], material.scroll[1], 0.0, 0.0],
                            }),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Robots BG"),
                        layout: &bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&diffuse_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&gloss_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&back_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&mask_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: mat_uniform.as_entire_binding(),
                            },
                        ],
                    });
                    surfaces.push(SurfaceGpu {
                        index_buffer,
                        num_indices: surf.indices.len() as u32,
                        bind_group,
                    });
                }
                total_surfaces += surfaces.len();
                frames.push(FrameGpu { surfaces });
            }

            log::info!(
                "robots: chassis{} frames={} anims: {:?}",
                n,
                vo_mesh.frames.len(),
                vo_mesh
                    .animations
                    .iter()
                    .map(|a| (a.name.clone(), a.frames.len()))
                    .collect::<Vec<_>>(),
            );
            let vo_mesh = std::sync::Arc::new(vo_mesh);
            crate::matrix_lib::three_g::animation::set_chassis_vo(
                chassis_kind_index(ck),
                vo_mesh.clone(),
            );
            chassis[chassis_kind_index(ck)] = Some(ChassisGpu {
                vo_mesh,
                vertex_buffer,
                frames,
            });
        }

        log::info!(
            "robots: {} chassis loaded ({} frame-surface slots)",
            chassis.iter().filter(|c| c.is_some()).count(),
            total_surfaces,
        );
        if chassis.iter().all(|c| c.is_none()) {
            return None;
        }

        Some(Self {
            pipeline,
            chassis,
            instance_buffer,
            instance_capacity: MAX_LIVE_ROBOTS,
            draws: Vec::new(),
            uniform_buffer,
            fog_color,
            ambient_color,
            light_color,
            light_dir,
            time_ms: 0.0,
            center: [cx, cy],
        })
    }

    pub fn takt(&mut self, dt_ms: f32) {
        self.time_ms += dt_ms;
        if self.time_ms > 1_000_000.0 {
            self.time_ms -= 1_000_000.0;
        }
    }

    /// Walk live `Robot`s, advance each chassis's animation cursor,
    /// build per-robot draw tickets, and upload the instance buffer.
    /// Port of `CMatrixRobot::Takt`'s animation step + the matrix
    /// build in `RNeed(MR_Matrix)`.
    ///
    /// `cms` is the logic-takt delta (in ms) to advance the anim
    /// cursor by. Passed in from `form_game.rs` once per frame.
    pub fn sync_robots(
        &mut self,
        queue: &wgpu::Queue,
        objs: &mut Objects,
        map: &GameMap,
        point_lights: &PointLightSystem,
        cms: i32,
    ) {
        let [cx, cy] = self.center;
        self.draws.clear();

        let mut instance_data: Vec<InstanceData> = Vec::with_capacity(16);
        let mut offset: u32 = 0;

        // Snapshot the live ids first so we can mutate robots as we
        // iterate (need &mut for anim advance).
        let ids: Vec<_> = objs.iter_live().collect();
        for id in ids {
            let Some(obj) = objs.get_mut(id) else {
                continue;
            };
            if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
                continue;
            }
            let robot: &mut Robot = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Robot) };
            let ck_idx = chassis_kind_index(robot.chassis);
            let Some(chassis_gpu) = self.chassis.get(ck_idx).and_then(|o| o.as_ref()) else {
                continue;
            };
            // Port of `CMatrixRobot::DoAnimation` at
            // MatrixObjectRobot.cpp:756-880 — chassis-unit branch.
            // The chassis cursor advances differently by state and
            // chassis kind; in particular, for Track/Wheel the
            // STAY / BEGINMOVE / ENDMOVE states DO NOT advance the
            // cursor at all (falls through past every branch to the
            // `endanim` label), leaving the chassis frame frozen.
            let vo_mesh = chassis_gpu.vo_mesh.as_ref();
            let now_ms = crate::matrix_game::map::current_elapsed_ms() as f64;
            do_chassis_animation(robot, vo_mesh, now_ms, cms);
            // BeginMove → Move chaining (MatrixObjectRobot.cpp:1414).
            if robot.animation == Animation::BeginMove && robot.chassis_anim.is_anim_end(vo_mesh) {
                robot.animation = Animation::Move;
                robot.chassis_anim.set_anim_by_name(vo_mesh, "Move", true);
            }
            let vo_frame = robot
                .chassis_anim
                .vo_frame
                .min(chassis_gpu.frames.len().saturating_sub(1));

            if offset >= self.instance_capacity {
                break;
            }
            instance_data.push(robot_instance(robot, cx, cy, map, Some(point_lights)));
            self.draws.push(RobotDraw {
                chassis: robot.chassis,
                vo_frame,
                instance_offset: offset,
            });
            offset += 1;
        }

        if !instance_data.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instance_data),
            );
        }
    }

    pub fn render<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
        view_proj: glam::Mat4,
    ) {
        let eye = camera.eye_pos();
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color: self.fog_color,
                fog_params: [FOG_START, FOG_END, 0.0, 0.0],
                ambient_color: self.ambient_color,
                light_color: self.light_color,
                light_dir: self.light_dir,
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                time_ms: [self.time_ms, 0.0, 0.0, 0.0],
            }),
        );

        pass.set_pipeline(&self.pipeline);
        for draw in &self.draws {
            let ck_idx = chassis_kind_index(draw.chassis);
            let Some(chassis_gpu) = self.chassis.get(ck_idx).and_then(|o| o.as_ref()) else {
                continue;
            };
            let Some(frame) = chassis_gpu.frames.get(draw.vo_frame) else {
                continue;
            };
            pass.set_vertex_buffer(0, chassis_gpu.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            for surface in &frame.surfaces {
                pass.set_bind_group(0, &surface.bind_group, &[]);
                pass.set_index_buffer(surface.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                // `first_instance = instance_offset` routes this draw
                // to the robot's slot in the shared instance buffer.
                pass.draw_indexed(
                    0..surface.num_indices,
                    0,
                    draw.instance_offset..(draw.instance_offset + 1),
                );
            }
        }
    }
}

/// Port of the chassis-unit branch of `CMatrixRobot::DoAnimation`
/// (MatrixObjectRobot.cpp:778-880). Decides whether to advance the
/// chassis animation cursor this tick and how:
///
///   * STAY/BEGINMOVE/ENDMOVE(+back variants) + Hover/Pneu/Antigrav
///     → normal constant-rate `Takt(cms)` (line 791).
///   * ROTATE (any chassis) → speed-based advance scaled by
///     `k = ANIMSPEED / m_RotSpeed` clamped to 3 (lines 802-840).
///   * MOVE/BEGINMOVE/ENDMOVE(+back variants) (Track/Wheel/Pneu)
///     → speed-based advance scaled by `k = ANIMSPEED / m_Speed`
///     clamped to 3 (lines 842-880).
///   * Anything else (e.g. Track/Wheel in STAY) → no advance. The
///     cursor stays put, so the tracks are motionless when the
///     robot is stopped.
///
/// ROTATE handling is stubbed — full rotation lands with the Seek
/// rotation branch.
fn do_chassis_animation(robot: &mut Robot, vo: &vector_object::VoMesh, now_ms: f64, cms: i32) {
    use crate::matrix_game::robot::ChassisKind::*;
    let anim = robot.animation;

    let is_stay_like = matches!(
        anim,
        Animation::Stay
            | Animation::BeginMove
            | Animation::EndMove
            | Animation::BeginMoveBack
            | Animation::EndMoveBack
    );
    if is_stay_like && matches!(robot.chassis, Hovercraft | Pneumatic | AntiGravity) {
        robot.chassis_anim.takt(vo, cms);
        return;
    }

    let is_move_like = matches!(
        anim,
        Animation::Move
            | Animation::BeginMove
            | Animation::EndMove
            | Animation::MoveBack
            | Animation::BeginMoveBack
            | Animation::EndMoveBack
    );
    if is_move_like {
        // MatrixObjectRobot.hpp:59-61.
        const ANIMSPEED_TRACK: f32 = 0.20;
        const ANIMSPEED_WHEEL: f32 = 0.36;
        const ANIMSPEED_PNEU: f32 = 0.155;
        let animspeed = match robot.chassis {
            Track => ANIMSPEED_TRACK,
            Wheel => ANIMSPEED_WHEEL,
            Pneumatic => ANIMSPEED_PNEU,
            _ => return, // Hover/Antigrav aren't in the MOVE branch here.
        };
        // Avoid div-by-zero when robot is stationary; the C++
        // divides directly and relies on m_Speed being non-zero
        // during MOVE states. If it reaches zero we clamp `k` to
        // its upper limit (3), which freezes the animation.
        let speed = robot.speed.abs().max(1e-3);
        let k = (animspeed / speed).min(3.0);
        while now_ms > robot.chassis_anim.next_anim_time {
            let frame_time = robot.chassis_anim.next_frame(vo) as f32;
            let add = k * frame_time;
            robot.chassis_anim.next_anim_time += add.max(0.1) as f64;
        }
        return;
    }

    // No match → no advance. Track/Wheel in STAY lands here, which
    // matches the C++ behaviour at lines 778-801: the STAY block
    // only advances for Hover/Pneu/Antigrav, and everything else
    // falls through to `endanim` without touching the cursor.
    let _ = (cms, now_ms);
}

fn chassis_kind_index(c: ChassisKind) -> usize {
    match c {
        ChassisKind::Pneumatic => 0,
        ChassisKind::Wheel => 1,
        ChassisKind::Track => 2,
        ChassisKind::AntiGravity => 3,
        ChassisKind::Hovercraft => 4,
    }
}

/// Build one instance transform for a live `Robot`. Port of the
/// side/forward/up → `m_Core->m_Matrix._11.._34` column assignment
/// at MatrixObjectRobot.cpp:443-458 (the branch taken for the normal
/// land / water / move-out states).
///
/// The original D3D row-major assignment:
///   `_11/_12/_13` = side world xyz  (row 1)
///   `_21/_22/_23` = forward world xyz (row 2)
///   `_31/_32/_33` = up world xyz    (row 3)
///   `_41/_42/_43` = pos world xyz   (row 4)
///
/// In D3D row-major, a local point `(1,0,0)` ends up at row 1 = side;
/// i.e. local X maps to world side, local Y to world forward, local Z
/// to world up. Our shader is column-major glam and reads the instance
/// as three basis *columns*: `row0.xyz = x-axis-to-world`,
/// `row1.xyz = y-axis-to-world`, `row2.xyz = z-axis-to-world`. Under
/// that convention `row_i.x = side.x (=_11)`, `row_i.y = side.y (=_12)`,
/// etc. So the instance rows end up literally identical to the D3D
/// row-major layout — we just copy `_11/_12/_13/_14` into `row0` and
/// so on, with the fourth column carrying the world translation.
///
/// Up is world-Z (terrain-normal slope fitting lands later — needs
/// `CMatrixMap::GetNormal`). Side = cross(forward, up) ports the
/// `D3DXVec3Cross(&side, &m_Forward, &up)` at MatrixObjectRobot.cpp:
/// 450 + :451 Normalize.
fn robot_instance(
    r: &Robot,
    cx: f32,
    cy: f32,
    map: &GameMap,
    point_lights: Option<&PointLightSystem>,
) -> InstanceData {
    let [terrain_r, terrain_g, terrain_b] =
        unpack_rgb(map.static_object_color_with_lighting(r.pos_x, r.pos_y, point_lights));

    // forward in XY plane, normalized. pos_x/y is the robot center.
    let f = {
        let v = r.forward;
        let l = v.length();
        if l > 1e-6 {
            v / l
        } else {
            glam::Vec2::new(0.0, 1.0)
        }
    };
    let forward = glam::Vec3::new(f.x, f.y, 0.0);
    let up = glam::Vec3::new(0.0, 0.0, 1.0);
    // side = cross(forward, up) — MatrixObjectRobot.cpp:450.
    let side = forward.cross(up).normalize_or_zero();
    // tmp_forward = cross(up, side) — :452. In the planar case with
    // slope up = (0,0,1), tmp_forward equals forward; we compute it
    // faithfully so the slope port drops in unchanged.
    let fwd_out = up.cross(side).normalize_or_zero();

    InstanceData {
        row0: [side.x, fwd_out.x, up.x, r.pos_x - cx],
        row1: [side.y, fwd_out.y, up.y, r.pos_y - cy],
        row2: [side.z, fwd_out.z, up.z, r.pos_z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [terrain_r, terrain_g, terrain_b, 1.0],
        unit_offset: [0.0, 0.0, 0.0, 0.0],
    }
}

fn resolve_diffuse(
    material: &MaterialSpec,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<(wgpu::TextureView, bool)> {
    let path = material.diffuse.as_ref()?;
    let view = resolve_texture(Some(path), device, queue, cache, read_texture)?;
    let alpha_test =
        vector_object::resolve_alpha_test_with_txt(path, material.alpha_test, read_texture);
    Some((view, alpha_test))
}

fn resolve_texture(
    tex_path: Option<&String>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut HashMap<String, wgpu::TextureView>,
    read_texture: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Option<wgpu::TextureView> {
    let path = tex_path?;
    if let Some(v) = cache.get(path) {
        return Some(v.clone());
    }
    let data = read_texture(path)?;
    let rgba = decode_texture_bytes(&data)?;
    let view = create_texture_from_rgba(device, queue, &rgba);
    cache.insert(path.clone(), view.clone());
    Some(view)
}

fn create_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Robots BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Robots Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Robots PL"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Robots Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
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
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<InstanceData>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 48,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 64,
                            shader_location: 7,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 80,
                            shader_location: 8,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                },
            ],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
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
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

const SHADER: &str = include_str!("../../shaders/object_building.wgsl");
