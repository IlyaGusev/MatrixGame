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
    /// Per-side tint — same role as in buildings' `InstanceData`.
    /// Robots also carry `m_Side` in the C++ and get the same
    /// `GetSideColorTexture(m_Side)` treatment on team-marker
    /// surfaces (the shipped VOs flag the pilot-cabin trim etc.);
    /// we reuse the whole-mesh reduced-strength tint.
    side_color: [f32; 4],
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
    /// Per-`RUK_ARMOR_*` armor mesh (slot `kind-1`). Used by the
    /// constructor preview renderer; not read by the world pass.
    armor: Vec<Option<ChassisGpu>>,
    /// Per-`RUK_HEAD_*` head mesh.
    head: Vec<Option<ChassisGpu>>,
    /// Per-`RUK_WEAPON_*` weapon mesh.
    weapon: Vec<Option<ChassisGpu>>,
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
    /// Dedicated 1-element instance buffer for the constructor-panel
    /// preview robot (CConstructor.cpp:264-360). Keeps the world
    /// instance buffer untouched so the main pass is unaffected.
    preview_instance_buffer: wgpu::Buffer,
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
        // 8 slots: chassis(1) + armor(1) + head(1) + weapons(5). Each
        // part gets its own world matrix derived from the bone chain
        // in `render_preview_full`.
        const PREVIEW_INSTANCE_SLOTS: u64 = 8;
        let preview_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Robots Preview Inst VB"),
            size: PREVIEW_INSTANCE_SLOTS * std::mem::size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Helper: load N meshes from a path template and return the
        // per-slot GPU resources. Used both for chassis and for the
        // constructor's armor / head / weapon overlays.
        let load_part_meshes = |kind_count: usize,
                                path_prefix: &str,
                                tex_cache: &mut HashMap<String, wgpu::TextureView>|
         -> (Vec<Option<ChassisGpu>>, usize) {
            let mut out: Vec<Option<ChassisGpu>> = (0..kind_count).map(|_| None).collect();
            let mut surf_count = 0usize;
            for n in 1..=kind_count {
                let vo_path = format!("Matrix/Robot/{}{}.vo", path_prefix, n);
                let Some(vo_bytes) = read_texture(&vo_path) else {
                    continue;
                };
                let vo_mesh = match vector_object::parse_vo(&vo_bytes) {
                    Ok(m) => m,
                    Err(_) => continue,
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
                    label: Some("Robots Part VB"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let vo_dir = vo_path.rsplit_once('/').map(|(d, _)| format!("{d}/"));
                let top_diffuse = format!("Matrix/Robot/{}{}", path_prefix, n);
                let top_gloss = format!("Matrix/Robot/{}{}_gloss", path_prefix, n);

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
                            let surface_mat = vector_object::parse_material_spec_with_prefix(
                                spec,
                                vo_dir.as_deref(),
                            );
                            material =
                                vector_object::merge_materials(&surface_mat, Some(&material));
                        }
                        let (diffuse_view, alpha_test) =
                            resolve_diffuse(&material, device, queue, tex_cache, read_texture)
                                .unwrap_or_else(|| (fallback_tex.clone(), false));
                        let gloss_view = resolve_texture(
                            material.gloss.as_ref(),
                            device,
                            queue,
                            tex_cache,
                            read_texture,
                        )
                        .unwrap_or_else(|| black_tex.clone());
                        let back_view = resolve_texture(
                            material.back.as_ref(),
                            device,
                            queue,
                            tex_cache,
                            read_texture,
                        )
                        .unwrap_or_else(|| black_tex.clone());
                        let mask_view = resolve_texture(
                            material.mask.as_ref(),
                            device,
                            queue,
                            tex_cache,
                            read_texture,
                        )
                        .unwrap_or_else(|| transparent_tex.clone());
                        let index_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Robots Part IB"),
                                contents: bytemuck::cast_slice(&surf.indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });
                        let mat_uniform =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Robots Part Material UB"),
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
                            label: Some("Robots Part BG"),
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
                    surf_count += surfaces.len();
                    frames.push(FrameGpu { surfaces });
                }
                let vo_mesh = std::sync::Arc::new(vo_mesh);
                out[n - 1] = Some(ChassisGpu {
                    vo_mesh,
                    vertex_buffer,
                    frames,
                });
            }
            (out, surf_count)
        };

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
            crate::matrix_lib::three_g::vector_object::set_chassis_vo(
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

        // Load constructor-preview overlays — armor (6 kinds), head
        // (4 kinds), weapon (10 kinds). Counts mirror MatrixConfig.hpp's
        // ROBOT_*_CNT constants. Missing meshes (older asset bundles
        // pre-dating the pack_bundle update) leave those slots None.
        let (armor, armor_surfs) = load_part_meshes(6, "Armor", &mut tex_cache);
        let (head, head_surfs) = load_part_meshes(4, "Head", &mut tex_cache);
        let (weapon, weapon_surfs) = load_part_meshes(10, "Weapon", &mut tex_cache);
        log::info!(
            "robots: preview parts loaded — armor={}/{}, head={}/{}, weapon={}/{} ({} surfaces)",
            armor.iter().filter(|c| c.is_some()).count(),
            6,
            head.iter().filter(|c| c.is_some()).count(),
            4,
            weapon.iter().filter(|c| c.is_some()).count(),
            10,
            armor_surfs + head_surfs + weapon_surfs,
        );

        Some(Self {
            pipeline,
            chassis,
            armor,
            head,
            weapon,
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
            preview_instance_buffer,
        })
    }

    /// Port of `CConstructor::Render` (CConstructor.cpp:264-360) — draws
    /// one robot chassis into the sub-viewport defined by
    /// `scissor_px = [x, y, w, h]` in screen pixels. Uses the same
    /// pipeline as the world pass but a preview-specific view-proj
    /// (eye at (80, -30, 5), look at origin) and a constructor-
    /// specific directional light.
    ///
    /// `angle_rad` turntables the chassis around its Z axis. The
    /// caller (form_game::tick_builder_preview) increments this per
    /// frame.
    #[allow(clippy::too_many_arguments)]
    pub fn render_preview<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        chassis: ChassisKind,
        angle_rad: f32,
        scissor_px: [u32; 4],
        surface_w: u32,
        surface_h: u32,
    ) {
        // Backwards-compat shim — chassis-only preview, used by
        // callers that haven't yet been updated to pass armor / head /
        // weapon kinds. Internally just calls render_preview_full.
        self.render_preview_full(
            queue, pass, chassis, None, None, &[None; 5], angle_rad, scissor_px, surface_w,
            surface_h,
        );
    }

    /// Port of `CConstructor::Render` plus the multi-unit `m_Robot->Draw()`
    /// loop at MatrixObjectRobot.cpp:299-357 — for each populated
    /// (chassis / armor / head / weapon) component, queue a draw using
    /// the matching VO mesh.
    ///
    /// Components stack at the chassis origin. The C++ uses bone-anchor
    /// matrices stored in the chassis VO (`m_Unit[i].m_LinkMatrix`),
    /// which the current vector_object parser doesn't extract — until
    /// that landing, all overlays render at the chassis origin.
    /// Visually this is "all parts present, slightly mis-stacked";
    /// once the bone-anchor extension lands the per-unit transform
    /// drops in here.
    ///
    /// `armor_kind` / `head_kind` / `weapon_kinds` are the 1-based
    /// `RUK_*` discriminants; pass `None` / 0 to skip a slot.
    #[allow(clippy::too_many_arguments)]
    pub fn render_preview_full<'a>(
        &'a self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        chassis: ChassisKind,
        armor_kind: Option<i32>,
        head_kind: Option<i32>,
        weapon_kinds: &[Option<i32>; 5],
        angle_rad: f32,
        scissor_px: [u32; 4],
        surface_w: u32,
        surface_h: u32,
    ) {
        let ck_idx = chassis_kind_index(chassis);
        let chassis_gpu = match self.chassis.get(ck_idx).and_then(|o| o.as_ref()) {
            Some(g) => g,
            None => {
                static ONCE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::warn!(
                        "preview: chassis {:?} (idx {}) has no GPU mesh — chassis.len()={}, loaded={:?}",
                        chassis,
                        ck_idx,
                        self.chassis.len(),
                        self.chassis
                            .iter()
                            .enumerate()
                            .map(|(i, o)| (i, o.is_some()))
                            .collect::<Vec<_>>(),
                    );
                }
                return;
            }
        };
        // Preview always uses VO frame 0 (the at-rest pose — the
        // original sets `m_CurrState = ROBOT_EMBRYO` which freezes the
        // animation cursor; see CConstructor.cpp:45).
        let Some(_frame) = chassis_gpu.frames.first() else {
            static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!("preview: chassis {:?} GPU mesh has zero frames", chassis);
            }
            return;
        };
        {
            static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::info!(
                    "preview: drawing chassis={:?} armor={:?} head={:?} weapons={:?} scissor={:?} surface={}x{}",
                    chassis, armor_kind, head_kind, weapon_kinds, scissor_px, surface_w, surface_h,
                );
            }
        }

        // Preview view-proj (CConstructor.cpp:318-324). Eye at
        // (80, -30, h+5), looking at (0, 0, h) where h is the chassis
        // height above ground. FOV = π/4, aspect = panel rect, near 1
        // / far 300.
        //
        // `h` is the armor-mount Z on the chassis bone graph — port
        // of `CMatrixRobot::GetChassisHeight` (MatrixObjectRobot.cpp:
        // 271-275 = `m_Unit[0].m_Graph->GetMatrixById(1)->_43`). A
        // hardcoded `spawn_z_offset` value doesn't match the per-VO
        // bone offset and produces a camera aim elevated above the
        // robot body.
        let height = chassis_gpu
            .vo_mesh
            .matrix_by_id(1, 0)
            .map(|m| m[14]) // D3DXMATRIX _43 = row 3 col 2 = flat[14]
            .unwrap_or_else(|| chassis.spawn_z_offset());
        let _ = chassis; // keep the kind in scope for logging downstream
        let scissor_aspect = if scissor_px[3] > 0 {
            scissor_px[2] as f32 / scissor_px[3] as f32
        } else {
            1.0
        };
        let eye = glam::Vec3::new(80.0, -30.0, height + 5.0);
        let target = glam::Vec3::new(0.0, 0.0, height);
        let up = glam::Vec3::new(0.0, 0.0, 1.0);
        let view = glam::Mat4::look_at_lh(eye, target, up);
        let proj =
            glam::Mat4::perspective_lh(std::f32::consts::FRAC_PI_4, scissor_aspect, 1.0, 300.0);
        // D3D and wgpu both use Y+ up / Z in [0,1] for clip space,
        // and glam's `look_at_lh` + `perspective_lh` produces that
        // clip orientation natively. No additional flip needed —
        // the previous code applied a spurious Y-flip that rendered
        // the robot upside-down.
        let view_proj = proj * view;

        // Upload preview uniforms — constructor-specific light from
        // CConstructor.cpp:283-303.
        let preview_light_dir = [-0.82242596, 0.56887215, 0.0, 0.0];
        let preview_ambient = [0.5, 0.5, 0.5, 1.0];
        let preview_light = [1.0, 1.0, 1.0, 1.0];
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                fog_color: [0.0, 0.0, 0.0, 0.0],
                fog_params: [10000.0, 10001.0, 0.0, 0.0],
                ambient_color: preview_ambient,
                light_color: preview_light,
                light_dir: preview_light_dir,
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                time_ms: [self.time_ms, 0.0, 0.0, 0.0],
            }),
        );

        // ── Build the per-part world matrix chain ─────────────────
        //
        // Faithful port of `CMatrixRobot::Draw`'s matrix-graph walk at
        // MatrixObjectRobot.cpp:480-575 together with the weapon-slot
        // table built at MatrixMap.cpp:270-326 and the slot assignment
        // at MatrixObjectRobot.cpp:252-268.
        //
        // All internal math runs in D3D row-major (the on-disk format
        // of SVOMatrix). The shader reads the transpose (m0..m3 are
        // the rows of a matrix used with column-vector multiply = the
        // columns of a D3D row-major matrix), so `pack` transposes on
        // upload.

        fn flat_to_rows(flat: [f32; 16]) -> [[f32; 4]; 4] {
            [
                [flat[0], flat[1], flat[2], flat[3]],
                [flat[4], flat[5], flat[6], flat[7]],
                [flat[8], flat[9], flat[10], flat[11]],
                [flat[12], flat[13], flat[14], flat[15]],
            ]
        }
        fn row_translate(m: &[[f32; 4]; 4]) -> [f32; 3] {
            // D3D row-major: translation lives in row 3, columns 0-2
            // (_41/_42/_43).
            [m[3][0], m[3][1], m[3][2]]
        }
        fn matmul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
            // Row-major multiply — matches D3D's `v_out = v_in * A * B`
            // convention so `child.m_Matrix = local * parent`.
            let mut r = [[0.0f32; 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    r[i][j] = a[i][0] * b[0][j]
                        + a[i][1] * b[1][j]
                        + a[i][2] * b[2][j]
                        + a[i][3] * b[3][j];
                }
            }
            r
        }

        // D3DXMatrixRotationZ (row-major) used for the turntable spin.
        // The C++ preview itself uses an identity world matrix
        // (CConstructor.cpp:320); the 0.2 rad/sec spin here is the
        // commented-out rotation at CConstructor.cpp:296-300 which the
        // Rust port opted to keep as a visual touch.
        let chassis_world = {
            let (s, c) = angle_rad.sin_cos();
            [
                [c, s, 0.0, 0.0],
                [-s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        };

        // Chassis mount = chassis.bone(1).translate — port of
        // MatrixObjectRobot.cpp:490-492 (`tm = m_Unit[0].m_Graph->GetMatrixById(1);
        //   p = *(D3DXVECTOR3 *)&tm->_41;`).
        const IDENTITY: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let chassis_bone1 = chassis_gpu
            .vo_mesh
            .matrix_by_id(1, 0)
            .map(flat_to_rows)
            .unwrap_or(IDENTITY);
        let mut p = row_translate(&chassis_bone1);

        // Armor branch (MatrixObjectRobot.cpp:525-545):
        //   D3DXMatrixIdentity(&m);
        //   th = D3DXVec3TransformNormal(m_HullForward, m_Core->m_IMatrix);
        //   m._21 = th.x;   m._22 = th.y;
        //   m._11 = th.y;   m._12 = -th.x;
        //   goto calc;  // m._41..43 = p, m_Unit[i].m_Matrix = m * core;
        //
        // For the preview the C++ Draw path runs against a robot whose
        // `m_Core->m_Matrix = identity` and `m_HullForward = (0,1,0)`
        // (the default in `RNeed(MR_Matrix)` before any animation step);
        // that yields `th = (0,1,0)` and `m` becomes the identity
        // rotation. Our chassis_world carries the turntable spin
        // additionally so armor/head/weapons rotate with it.
        let armor_rot = IDENTITY; // th=(0,1,0) → m._11=1, m._22=1, off-diag=0
        let armor_world_opt = armor_kind.and_then(|k| {
            if k < 1 {
                return None;
            }
            let armor_gpu = self.armor.get((k - 1) as usize).and_then(|o| o.as_ref())?;
            // m = armor_rot with translation column (_41..43) = p.
            let mut m = armor_rot;
            m[3][0] = p[0];
            m[3][1] = p[1];
            m[3][2] = p[2];
            Some((armor_gpu, matmul(&m, &chassis_world)))
        });

        // After the `calc:` label the C++ advances `p` by the unit's
        // own bone 1 transformed by the child's rotation (Matrix-
        // ObjectRobot.cpp:571-574):
        //   p.x += tm->_41 * m._11 + tm->_42 * m._21;
        //   p.y += tm->_41 * m._12 + tm->_42 * m._22;
        //   p.z += tm->_43;
        // where `tm = m_Unit[i].m_Graph->GetMatrixById(1)` (the unit's
        // local mount point) and `m` is the child's rotation.
        if let Some((armor_gpu, _)) = armor_world_opt.as_ref() {
            if let Some(ab) = armor_gpu.vo_mesh.matrix_by_id(1, 0) {
                let ab = flat_to_rows(ab);
                let tlx = ab[3][0];
                let tly = ab[3][1];
                let tlz = ab[3][2];
                p[0] += tlx * armor_rot[0][0] + tly * armor_rot[1][0];
                p[1] += tlx * armor_rot[0][1] + tly * armor_rot[1][1];
                p[2] += tlz;
            }
        }

        // Head branch (MatrixObjectRobot.cpp:547-559): same shape as
        // armor but with the accumulated `p`.
        let head_rot = IDENTITY;
        let head_gpu_opt = head_kind.and_then(|k| {
            if k < 1 {
                return None;
            }
            self.head.get((k - 1) as usize).and_then(|o| o.as_ref())
        });
        let head_world_opt = head_gpu_opt.map(|_| {
            let mut m = head_rot;
            m[3][0] = p[0];
            m[3][1] = p[1];
            m[3][2] = p[2];
            matmul(&m, &chassis_world)
        });

        // ── Weapon slot table (port of MatrixMap.cpp:270-326) ─────
        //
        // For each armor VO, collect matrices with id >= 20 whose name
        // encodes `"W, kind1, kind2, ..., [I]"`. Each bone becomes a
        // weapon slot; the `access_invert` bitmask records compatible
        // weapon kinds (`1 << (kind-1)`) plus `SETBIT(31)` when the
        // slot name ends in `,I` (invert bit → mirror-flip on X axis
        // for the attached weapon). Slots are then sorted by bone id.
        #[derive(Clone, Copy, Debug)]
        struct WeaponSlot {
            id: u32,
            access_invert: u32,
        }
        fn build_weapon_slots(vo: &vector_object::VoMesh) -> Vec<WeaponSlot> {
            let mut slots: Vec<WeaponSlot> = Vec::new();
            for m in &vo.matrices {
                if m.id < 20 {
                    continue;
                }
                // Parse comma-separated name: first token must be "W".
                let parts: Vec<&str> = m.name.split(',').map(|s| s.trim()).collect();
                if parts.is_empty() || parts[0] != "W" {
                    continue;
                }
                let mut access_invert: u32 = 0;
                for tok in &parts[1..] {
                    if *tok == "I" {
                        access_invert |= 1u32 << 31;
                    } else if let Ok(kind) = tok.parse::<i32>() {
                        if (1..=32).contains(&kind) {
                            access_invert |= 1u32 << (kind - 1);
                        }
                    }
                }
                slots.push(WeaponSlot {
                    id: m.id,
                    access_invert,
                });
            }
            slots.sort_by_key(|s| s.id);
            slots
        }
        let weapon_slots: Vec<WeaponSlot> = armor_world_opt
            .as_ref()
            .map(|(armor_gpu, _)| build_weapon_slots(&armor_gpu.vo_mesh))
            .unwrap_or_default();

        // Slot assignment (port of MatrixObjectRobot.cpp:254-268).
        // For each weapon in insertion order, grab the first available
        // slot whose `access_invert` bit for this weapon kind is set.
        // Records both the bone id and the invert flag for mirroring.
        let mut weapon_assignments: [Option<(u32, bool)>; 5] = [None; 5];
        let mut slot_used = vec![false; weapon_slots.len()];
        for (pilon, wk) in weapon_kinds.iter().enumerate().take(5) {
            let Some(kind) = wk else {
                continue;
            };
            if *kind < 1 {
                continue;
            }
            let bit = 1u32 << (*kind - 1);
            for (t, s) in weapon_slots.iter().enumerate() {
                if slot_used[t] {
                    continue;
                }
                if (s.access_invert & bit) == 0 {
                    continue;
                }
                slot_used[t] = true;
                let invert = (s.access_invert & (1u32 << 31)) != 0;
                weapon_assignments[pilon] = Some((s.id, invert));
                break;
            }
        }

        // Compute per-weapon world matrices. Port of
        // MatrixObjectRobot.cpp:504-522:
        //   tm = m_Unit[narmor].m_Graph->GetMatrixById(m_Unit[i].m_LinkMatrix);
        //   m_Unit[i].m_Matrix = (*tm) * m_Unit[narmor].m_Matrix;
        //   if(m_Unit[i].m_Invert) negate X-axis row (_11.._13).
        let mut weapon_worlds: [Option<[[f32; 4]; 4]>; 5] = [None; 5];
        if let Some((armor_gpu, armor_world)) = armor_world_opt.as_ref() {
            for (pilon, assign) in weapon_assignments.iter().enumerate() {
                let Some((slot_id, invert)) = assign else {
                    continue;
                };
                let Some(slot) = armor_gpu.vo_mesh.matrix_by_id(*slot_id, 0) else {
                    continue;
                };
                let mut wm = matmul(&flat_to_rows(slot), armor_world);
                if *invert {
                    // MatrixObjectRobot.cpp:515-521 — flip X basis (row 0).
                    wm[0][0] = -wm[0][0];
                    wm[0][1] = -wm[0][1];
                    wm[0][2] = -wm[0][2];
                }
                weapon_worlds[pilon] = Some(wm);
            }
        }

        // `pack` transposes D3D row-major → the shader's m0..m3 layout
        // (each shader row = D3D column). Matches the convention used
        // by the existing `robot_instance` at line ~1230.
        fn pack(m: &[[f32; 4]; 4]) -> InstanceData {
            InstanceData {
                row0: [m[0][0], m[1][0], m[2][0], m[3][0]],
                row1: [m[0][1], m[1][1], m[2][1], m[3][1]],
                row2: [m[0][2], m[1][2], m[2][2], m[3][2]],
                row3: [m[0][3], m[1][3], m[2][3], m[3][3]],
                terrain_color: [1.0, 1.0, 1.0, 1.0],
                unit_offset: [0.0, 0.0, 0.0, 0.0],
                // Constructor preview: no side tint. The C++'s
                // `CConstructor::Render` draws the preview robot with
                // the default white texture-factor, not via
                // `GetSideColorTexture` (CConstructor.cpp:340-360).
                side_color: [1.0, 1.0, 1.0, 1.0],
            }
        }
        // Upload all 8 instance slots. Slot assignment:
        //   0 — chassis, 1 — armor, 2 — head, 3..7 — weapons[0..4]
        let identity_packed = pack(&IDENTITY);
        let mut instances = [identity_packed; 8];
        instances[0] = pack(&chassis_world);
        if let Some((_, m)) = armor_world_opt.as_ref() {
            instances[1] = pack(m);
        }
        if let Some(m) = head_world_opt.as_ref() {
            instances[2] = pack(m);
        }
        for (i, m) in weapon_worlds.iter().enumerate() {
            if let Some(m) = m {
                instances[3 + i] = pack(m);
            }
        }
        queue.write_buffer(
            &self.preview_instance_buffer,
            0,
            bytemuck::cast_slice(&instances),
        );

        // Clamp scissor to surface bounds — wgpu panics if the rect
        // extends past the attachment.
        let sx = scissor_px[0].min(surface_w.saturating_sub(1));
        let sy = scissor_px[1].min(surface_h.saturating_sub(1));
        let sw = scissor_px[2].min(surface_w.saturating_sub(sx));
        let sh = scissor_px[3].min(surface_h.saturating_sub(sy));
        if sw == 0 || sh == 0 {
            return;
        }
        // Viewport MUST match the scissor — the projection matrix
        // above was built with `scissor_aspect` so NDC (-1..1) is
        // meant to map to the preview sub-rect. Without a viewport
        // set, wgpu maps NDC across the whole surface and the robot
        // renders at screen-centre, outside the scissor — completely
        // clipped out. Matches the D3DVIEWPORT9 restore at
        // CConstructor.cpp:272-276.
        pass.set_viewport(sx as f32, sy as f32, sw as f32, sh as f32, 0.0, 1.0);
        pass.set_scissor_rect(sx, sy, sw, sh);
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(1, self.preview_instance_buffer.slice(..));

        // Draw each part with its own instance slot (0..7). Order
        // matches CConstructor.cpp:71-85 `UnitInsert(0, ...)` —
        // chassis, armor, weapons (under armor), head.
        let mut parts: Vec<(&ChassisGpu, u32)> = Vec::with_capacity(8);
        parts.push((chassis_gpu, 0));
        if let Some((armor_gpu, _)) = armor_world_opt {
            parts.push((armor_gpu, 1));
        }
        for (pilon, wk) in weapon_kinds.iter().enumerate().take(5) {
            if let Some(k) = wk {
                if *k >= 1 {
                    if let Some(gpu) = self.weapon.get((*k - 1) as usize).and_then(|o| o.as_ref()) {
                        // Only draw if the bone chain produced a real
                        // world matrix (otherwise we'd draw at origin).
                        if weapon_worlds[pilon].is_some() {
                            parts.push((gpu, 3 + pilon as u32));
                        }
                    }
                }
            }
        }
        if let Some(head_gpu) = head_gpu_opt {
            parts.push((head_gpu, 2));
        }

        for (gpu, instance_idx) in parts {
            let Some(frame) = gpu.frames.first() else {
                continue;
            };
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
            for surface in &frame.surfaces {
                pass.set_bind_group(0, &surface.bind_group, &[]);
                pass.set_index_buffer(surface.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..surface.num_indices, 0, instance_idx..instance_idx + 1);
            }
        }

        // Restore viewport + scissor to full surface so downstream
        // passes aren't affected.
        pass.set_viewport(0.0, 0.0, surface_w as f32, surface_h as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, surface_w, surface_h);
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

    let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(r.side);
    InstanceData {
        row0: [side.x, fwd_out.x, up.x, r.pos_x - cx],
        row1: [side.y, fwd_out.y, up.y, r.pos_y - cy],
        row2: [side.z, fwd_out.z, up.z, r.pos_z],
        row3: [0.0, 0.0, 0.0, 1.0],
        terrain_color: [terrain_r, terrain_g, terrain_b, 1.0],
        unit_offset: [0.0, 0.0, 0.0, 0.0],
        side_color: [sr, sg, sb, 1.0],
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
                        wgpu::VertexAttribute {
                            offset: 96,
                            shader_location: 9,
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

// ── Robot unit type / slot counts (MatrixObjectRobot.hpp) ───────────────

/// Port of `MAX_WEAPON_CNT` (MatrixRobot.hpp:24). Maximum number of
/// weapon pylons on any robot — 4 common + 1 extra slot for the
/// bomb/mortar "super" weapon.
pub const MAX_WEAPON_CNT: usize = 5;

/// Port of `MR_MAXUNIT` (MatrixObjectRobot.hpp:69). Chassis + Armor +
/// Head + 5 weapons + 1 slot for anim hooks = 9 at the robot level.
pub const MR_MAXUNIT: usize = 9;

/// Port of `ERobotUnitType` (MatrixObjectRobot.hpp:47-55).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum RobotUnitType {
    #[default]
    Empty = 0,
    Chassis = 1,
    Weapon = 2,
    Armor = 3,
    Head = 4,
}
