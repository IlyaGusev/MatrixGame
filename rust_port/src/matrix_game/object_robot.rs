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

/// Identifies which mesh slot supplies geometry for a `PartDraw`.
/// Frame 0 is used for armor/head/weapons (port of CConstructor.cpp
/// preview path; the C++ steady-state pose for these parts is the
/// at-rest frame even on animated chassis).
#[derive(Clone, Copy, Debug)]
enum PartKind {
    /// Chassis with active animation cursor frame.
    Chassis(ChassisKind, usize),
    /// `kind - 1` index into `RobotsRenderer::armor`.
    Armor(usize),
    /// `kind - 1` index into `RobotsRenderer::head`.
    Head(usize),
    /// `kind - 1` index into `RobotsRenderer::weapon`.
    Weapon(usize),
}

/// Per-part draw ticket, rebuilt from scratch each frame in
/// `sync_robots`. Each robot emits 1 chassis draw plus optional
/// armor/head/weapon draws based on its `RobotConfig`.
struct PartDraw {
    kind: PartKind,
    instance_offset: u32,
}

const IDENTITY_MAT: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Reinterpret a 16-float D3D row-major matrix as a 4x4 row-array.
fn flat_to_rows(flat: [f32; 16]) -> [[f32; 4]; 4] {
    [
        [flat[0], flat[1], flat[2], flat[3]],
        [flat[4], flat[5], flat[6], flat[7]],
        [flat[8], flat[9], flat[10], flat[11]],
        [flat[12], flat[13], flat[14], flat[15]],
    ]
}

/// Translation column of a D3D row-major matrix (`_41/_42/_43`).
fn row_translate(m: &[[f32; 4]; 4]) -> [f32; 3] {
    [m[3][0], m[3][1], m[3][2]]
}

/// 4x4 row-major multiply matching D3D's `v_out = v_in * A * B`
/// convention so `child.m_Matrix = local * parent`.
fn matmul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
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

/// `pack` transposes D3D row-major → the shader's m0..m3 layout
/// (each shader row = D3D column). Caller supplies the per-instance
/// terrain/side tints — preview uses (1,1,1,1) for both, world uses
/// the per-robot lighting + side color.
fn pack_part_instance(
    m: &[[f32; 4]; 4],
    terrain_color: [f32; 4],
    side_color: [f32; 4],
) -> InstanceData {
    InstanceData {
        row0: [m[0][0], m[1][0], m[2][0], m[3][0]],
        row1: [m[0][1], m[1][1], m[2][1], m[3][1]],
        row2: [m[0][2], m[1][2], m[2][2], m[3][2]],
        row3: [m[0][3], m[1][3], m[2][3], m[3][3]],
        terrain_color,
        unit_offset: [0.0, 0.0, 0.0, 0.0],
        side_color,
    }
}

/// Resolved world matrices for a robot's secondary parts. Each entry
/// carries the `(kind - 1)` index into the matching GPU mesh table
/// alongside the D3D row-major world matrix for the part.
struct PartChain {
    armor: Option<(usize, [[f32; 4]; 4])>,
    head: Option<(usize, [[f32; 4]; 4])>,
    weapons: [Option<(usize, [[f32; 4]; 4])>; 5],
}

#[derive(Clone, Copy, Debug)]
struct VoWeaponSlot {
    id: u32,
    access_invert: u32,
}

/// Port of MatrixMap.cpp:270-326. Walks an armor VO's bones with
/// `id >= 20` whose name starts with `"W,"` and packs the comma-
/// separated kind list (plus optional `,I` invert flag) into a sorted
/// slot table.
fn build_vo_weapon_slots(vo: &vector_object::VoMesh) -> Vec<VoWeaponSlot> {
    let mut slots: Vec<VoWeaponSlot> = Vec::new();
    for m in &vo.matrices {
        if m.id < 20 {
            continue;
        }
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
        slots.push(VoWeaponSlot {
            id: m.id,
            access_invert,
        });
    }
    slots.sort_by_key(|s| s.id);
    slots
}

/// Port of `CMatrixRobot::Draw`'s matrix-graph walk
/// (MatrixObjectRobot.cpp:480-575) + the weapon-slot assignment at
/// MatrixObjectRobot.cpp:252-268. Returns the world matrices for
/// armor / head / each pilon weapon, expressed in D3D row-major and
/// rooted at `chassis_world`.
fn compute_part_chain(
    chassis_world: &[[f32; 4]; 4],
    chassis_gpu: &ChassisGpu,
    armor_gpus: &[Option<ChassisGpu>],
    armor_kind: Option<i32>,
    head_kind: Option<i32>,
    weapon_kinds: &[Option<i32>; 5],
) -> PartChain {
    // Chassis bone-1 mount (MatrixObjectRobot.cpp:490-492).
    let chassis_bone1 = chassis_gpu
        .vo_mesh
        .matrix_by_id(1, 0)
        .map(flat_to_rows)
        .unwrap_or(IDENTITY_MAT);
    let mut p = row_translate(&chassis_bone1);

    // Armor branch (MatrixObjectRobot.cpp:525-545). The C++ rotation
    // term reduces to identity in the steady state (HullForward = local
    // Y axis), so we use IDENTITY directly — the chassis_world supplies
    // the actual robot orientation.
    let armor_rot = IDENTITY_MAT;
    let armor_idx = armor_kind.and_then(|k| if k >= 1 { Some((k - 1) as usize) } else { None });
    let armor_world_opt = armor_idx.and_then(|idx| {
        let armor_gpu = armor_gpus.get(idx).and_then(|o| o.as_ref())?;
        let mut m = armor_rot;
        m[3][0] = p[0];
        m[3][1] = p[1];
        m[3][2] = p[2];
        Some((idx, armor_gpu, matmul(&m, chassis_world)))
    });

    // Advance `p` by the armor's own bone-1 mount
    // (MatrixObjectRobot.cpp:571-574).
    if let Some((_, armor_gpu, _)) = armor_world_opt.as_ref() {
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

    // Head branch (MatrixObjectRobot.cpp:547-559).
    let head_idx = head_kind.and_then(|k| if k >= 1 { Some((k - 1) as usize) } else { None });
    let head_world_opt = head_idx.map(|idx| {
        let mut m = IDENTITY_MAT;
        m[3][0] = p[0];
        m[3][1] = p[1];
        m[3][2] = p[2];
        (idx, matmul(&m, chassis_world))
    });

    // Weapon slot assignment + per-weapon worlds.
    let weapon_slots: Vec<VoWeaponSlot> = armor_world_opt
        .as_ref()
        .map(|(_, armor_gpu, _)| build_vo_weapon_slots(&armor_gpu.vo_mesh))
        .unwrap_or_default();

    let mut weapon_assignments: [Option<(u32, bool, usize)>; 5] = [None; 5];
    let mut slot_used = vec![false; weapon_slots.len()];
    for (pilon, wk) in weapon_kinds.iter().enumerate().take(5) {
        let Some(kind) = wk else { continue };
        if *kind < 1 {
            continue;
        }
        let weapon_idx = (*kind - 1) as usize;
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
            weapon_assignments[pilon] = Some((s.id, invert, weapon_idx));
            break;
        }
    }

    let mut weapons: [Option<(usize, [[f32; 4]; 4])>; 5] = [None; 5];
    if let Some((_, armor_gpu, armor_world)) = armor_world_opt.as_ref() {
        for (pilon, assign) in weapon_assignments.iter().enumerate() {
            let Some((slot_id, invert, weapon_idx)) = assign else {
                continue;
            };
            let Some(slot) = armor_gpu.vo_mesh.matrix_by_id(*slot_id, 0) else {
                continue;
            };
            let mut wm = matmul(&flat_to_rows(slot), armor_world);
            if *invert {
                wm[0][0] = -wm[0][0];
                wm[0][1] = -wm[0][1];
                wm[0][2] = -wm[0][2];
            }
            weapons[pilon] = Some((*weapon_idx, wm));
        }
    }

    PartChain {
        armor: armor_world_opt.map(|(idx, _, m)| (idx, m)),
        head: head_world_opt,
        weapons,
    }
}

/// Camera override for icon-bake calls into `render_preview_full`.
/// Port of the `anglez` / `anglex` / `fov` parameters of
/// `CMatrixMapStatic::RenderToTexture` (MatrixMapStatic.hpp:488).
/// `CPP_DEFAULTS` matches the C++ default arguments used by
/// `CMatrixRobotAI::CreateTextures` for the build-stack icon bake.
#[derive(Debug, Clone, Copy)]
pub struct IconCamera {
    /// Field-of-view in radians. C++ default `GRAD2RAD(60)` = π/3.
    pub fov: f32,
    /// Azimuth around Z. C++ default `GRAD2RAD(30)` = π/6.
    pub anglez: f32,
    /// Elevation. C++ default `GRAD2RAD(30)` = π/6.
    pub anglex: f32,
}

impl IconCamera {
    pub const CPP_DEFAULTS: Self = Self {
        fov: std::f32::consts::FRAC_PI_3,
        anglez: std::f32::consts::FRAC_PI_6,
        anglex: std::f32::consts::FRAC_PI_6,
    };
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
    /// Per-frame instance buffer: one `InstanceData` per live part
    /// (chassis + armor + head + per-weapon). Written contiguously in
    /// `sync_robots`; each `PartDraw::instance_offset` points into here.
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    draws: Vec<PartDraw>,
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
/// Up to 8 part instances per robot (1 chassis + 1 armor + 1 head +
/// 5 weapons). Sized for the worst case so a fully-equipped fleet
/// can render without a capacity break.
const MAX_LIVE_INSTANCES: u32 = MAX_LIVE_ROBOTS * 8;

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
            size: (MAX_LIVE_INSTANCES as u64) * std::mem::size_of::<InstanceData>() as u64,
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
        // Populate the global weapon-matrix table from the just-loaded
        // armor VOs. Faithful port of CMatrixMap::RobotPreload's
        // weapon-slot construction (MatrixMap.cpp:270-326): each
        // armor's bones with id >= 20 + name "W,..." declare the
        // weapon attach points.
        for (idx, slot) in armor.iter().enumerate() {
            if let Some(g) = slot {
                let m = crate::matrix_game::map::weapon_matrix_from_vo(&g.vo_mesh);
                crate::matrix_game::map::set_weapon_matrix_for(
                    crate::matrix_game::config::RobotUnitKind((idx + 1) as i32),
                    m,
                );
            }
        }
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
            instance_capacity: MAX_LIVE_INSTANCES,
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
            surface_h, None,
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
        icon_camera: Option<IconCamera>,
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
        let scissor_aspect = if scissor_px[3] > 0 {
            scissor_px[2] as f32 / scissor_px[3] as f32
        } else {
            1.0
        };
        let (eye, target, up, fov, far_z) = if let Some(icon) = icon_camera {
            // Faithful port of `CMatrixMapStatic::RenderToTexture`'s
            // camera setup (MatrixMapStatic.cpp:382-561). The C++
            // builds a per-config camera distance from a "robot
            // adjusted radius" (`ra`) that combines chassis height
            // with per-chassis and per-armor offsets, then places the
            // eye on a 30°/30° azimuth/elevation arc and looks at a
            // point shifted ~3 units sideways from origin.
            //
            // Tables come straight from the C++ switch statements at
            // MatrixMapStatic.cpp:487-547.
            let mut h = height;
            let mut ra = height;
            match chassis {
                ChassisKind::AntiGravity => {
                    h -= 5.0;
                    ra -= 5.5;
                }
                ChassisKind::Hovercraft => {
                    h -= 7.0;
                }
                ChassisKind::Pneumatic => {
                    h -= 9.0;
                    ra -= 1.0;
                }
                ChassisKind::Track => {
                    h -= 5.0;
                }
                ChassisKind::Wheel => {
                    h -= 8.0;
                    ra += 3.5;
                }
            }
            // Armor — RUK_ARMOR_* numeric values per
            // `matrix_game::config` (1=PASSIVE, 2=ACTIVE, 3=FIREPROOF,
            // 4=PLASMIC, 5=NUCLEAR, 6=ARMOR_6).
            match armor_kind {
                Some(2) => {
                    h += 9.0;
                    ra += 5.5;
                }
                Some(3) => {
                    h += 5.5;
                    ra += 3.0;
                }
                Some(5) => {
                    h += 13.5;
                    ra += 6.0;
                }
                Some(1) => {
                    h += 7.0;
                    ra += 2.5;
                }
                Some(4) => {
                    h += 10.0;
                    ra += 5.5;
                }
                Some(6) => {
                    h += 7.5;
                    ra += 6.5;
                }
                _ => {}
            }
            // `cdist = ra * 0.8 / tan(fov/2)` (MatrixMapStatic.cpp:549).
            let cdist = ra * 0.8 / (icon.fov * 0.5).tan();
            let (sin_z, cos_z) = icon.anglez.sin_cos();
            let (sin_x, _) = icon.anglex.sin_cos();
            let eye = glam::Vec3::new(cdist * sin_z, cdist * cos_z, sin_x * cdist + h);
            // `right = normalize(eye × Z_up)` then target is offset
            // sideways by 3 units in -right (MatrixMapStatic.cpp:554-561).
            let right = eye.cross(glam::Vec3::Z).normalize_or_zero();
            let target = glam::Vec3::new(-right.x * 3.0, -right.y * 3.0, h);
            (eye, target, glam::Vec3::Z, icon.fov, 500.0)
        } else {
            // Constructor preview camera — eye at (80, -30, h+5),
            // target at (0, 0, h), FOV π/4 (CConstructor.cpp:318-324).
            let eye = glam::Vec3::new(80.0, -30.0, height + 5.0);
            let target = glam::Vec3::new(0.0, 0.0, height);
            (
                eye,
                target,
                glam::Vec3::Z,
                std::f32::consts::FRAC_PI_4,
                300.0,
            )
        };
        let _ = chassis; // keep the kind in scope for logging downstream
        let view = glam::Mat4::look_at_lh(eye, target, up);
        let proj = glam::Mat4::perspective_lh(fov, scissor_aspect, 1.0, far_z);
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
        // MatrixObjectRobot.cpp:480-575 — chassis_world is the
        // turntable spin (`CConstructor::Render` runs the preview on
        // an identity-rooted robot), and the bone chain rooted at it
        // produces armor / head / weapon world matrices.

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

        let chain = compute_part_chain(
            &chassis_world,
            chassis_gpu,
            &self.armor,
            armor_kind,
            head_kind,
            weapon_kinds,
        );

        // Upload all 8 instance slots. Slot assignment:
        //   0 — chassis, 1 — armor, 2 — head, 3..7 — weapons[0..4].
        // Constructor preview tints are flat white (CConstructor.cpp:
        // 340-360 doesn't apply terrain or side colour).
        let pack = |m: &[[f32; 4]; 4]| {
            pack_part_instance(m, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        };
        let mut instances = [pack(&IDENTITY_MAT); 8];
        instances[0] = pack(&chassis_world);
        if let Some((_, m)) = chain.armor.as_ref() {
            instances[1] = pack(m);
        }
        if let Some((_, m)) = chain.head.as_ref() {
            instances[2] = pack(m);
        }
        for (i, w) in chain.weapons.iter().enumerate() {
            if let Some((_, m)) = w {
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
        if let Some((idx, _)) = chain.armor.as_ref() {
            if let Some(gpu) = self.armor.get(*idx).and_then(|o| o.as_ref()) {
                parts.push((gpu, 1));
            }
        }
        for (pilon, w) in chain.weapons.iter().enumerate() {
            if let Some((idx, _)) = w {
                if let Some(gpu) = self.weapon.get(*idx).and_then(|o| o.as_ref()) {
                    parts.push((gpu, 3 + pilon as u32));
                }
            }
        }
        if let Some((idx, _)) = chain.head.as_ref() {
            if let Some(gpu) = self.head.get(*idx).and_then(|o| o.as_ref()) {
                parts.push((gpu, 2));
            }
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

            // Per-robot lighting / side tint — same values for every
            // part of this robot.
            let [terrain_r, terrain_g, terrain_b] = unpack_rgb(
                map.static_object_color_with_lighting(
                    robot.pos_x,
                    robot.pos_y,
                    Some(point_lights),
                ),
            );
            let terrain_color = [terrain_r, terrain_g, terrain_b, 1.0];
            let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(robot.side);
            let side_color = [sr, sg, sb, 1.0];

            // Robot world matrix in D3D row-major form (the same
            // basis `robot_instance` builds, but laid out row-by-row
            // before transpose-pack so the bone chain helper can
            // multiply it as a parent matrix).
            let chassis_world = robot_world_d3d_rowmajor(robot, cx, cy);

            // Compute armor / head / weapon worlds.
            let armor_kind_i = robot.config.hull.unit.kind.0;
            let head_kind_i = robot.config.head.kind.0;
            let weapon_kinds: [Option<i32>; 5] = [
                non_zero_kind(robot.config.weapon[0].kind.0),
                non_zero_kind(robot.config.weapon[1].kind.0),
                non_zero_kind(robot.config.weapon[2].kind.0),
                non_zero_kind(robot.config.weapon[3].kind.0),
                non_zero_kind(robot.config.weapon[4].kind.0),
            ];
            let chain = compute_part_chain(
                &chassis_world,
                chassis_gpu,
                &self.armor,
                non_zero_kind(armor_kind_i),
                non_zero_kind(head_kind_i),
                &weapon_kinds,
            );

            // Push chassis instance + part instances. Each gets its
            // own slot in the shared instance buffer with a matching
            // PartDraw entry.
            let push_instance = |instance_data: &mut Vec<InstanceData>,
                                 draws: &mut Vec<PartDraw>,
                                 offset: &mut u32,
                                 cap: u32,
                                 m: &[[f32; 4]; 4],
                                 kind: PartKind|
             -> bool {
                if *offset >= cap {
                    return false;
                }
                instance_data.push(pack_part_instance(m, terrain_color, side_color));
                draws.push(PartDraw {
                    kind,
                    instance_offset: *offset,
                });
                *offset += 1;
                true
            };

            if !push_instance(
                &mut instance_data,
                &mut self.draws,
                &mut offset,
                self.instance_capacity,
                &chassis_world,
                PartKind::Chassis(robot.chassis, vo_frame),
            ) {
                break;
            }
            if let Some((idx, m)) = chain.armor.as_ref() {
                push_instance(
                    &mut instance_data,
                    &mut self.draws,
                    &mut offset,
                    self.instance_capacity,
                    m,
                    PartKind::Armor(*idx),
                );
            }
            if let Some((idx, m)) = chain.head.as_ref() {
                push_instance(
                    &mut instance_data,
                    &mut self.draws,
                    &mut offset,
                    self.instance_capacity,
                    m,
                    PartKind::Head(*idx),
                );
            }
            for w in chain.weapons.iter() {
                if let Some((idx, m)) = w {
                    push_instance(
                        &mut instance_data,
                        &mut self.draws,
                        &mut offset,
                        self.instance_capacity,
                        m,
                        PartKind::Weapon(*idx),
                    );
                }
            }
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
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        for draw in &self.draws {
            let (gpu, vo_frame) = match draw.kind {
                PartKind::Chassis(chassis, frame) => {
                    let Some(g) =
                        self.chassis.get(chassis_kind_index(chassis)).and_then(|o| o.as_ref())
                    else {
                        continue;
                    };
                    (g, frame)
                }
                PartKind::Armor(idx) => {
                    let Some(g) = self.armor.get(idx).and_then(|o| o.as_ref()) else {
                        continue;
                    };
                    (g, 0)
                }
                PartKind::Head(idx) => {
                    let Some(g) = self.head.get(idx).and_then(|o| o.as_ref()) else {
                        continue;
                    };
                    (g, 0)
                }
                PartKind::Weapon(idx) => {
                    let Some(g) = self.weapon.get(idx).and_then(|o| o.as_ref()) else {
                        continue;
                    };
                    (g, 0)
                }
            };
            let Some(frame) = gpu.frames.get(vo_frame) else {
                continue;
            };
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
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
    // Mirror MatrixConfig.hpp:39-43 — RUK_CHASSIS_PNEUMATIC=1,
    // WHEEL=2, TRACK=3, HOVERCRAFT=4, ANTIGRAVITY=5. The .vo file
    // index is `kind - 1` (Chassis1.vo = Pneumatic, Chassis5.vo =
    // Antigravity), so Hovercraft must come BEFORE AntiGravity here.
    match c {
        ChassisKind::Pneumatic => 0,
        ChassisKind::Wheel => 1,
        ChassisKind::Track => 2,
        ChassisKind::Hovercraft => 3,
        ChassisKind::AntiGravity => 4,
    }
}

/// Returns `Some(k)` for `k >= 1`, else `None`. Convenience for
/// `RobotConfig` weapon kinds where 0 means "empty pylon".
fn non_zero_kind(k: i32) -> Option<i32> {
    if k >= 1 {
        Some(k)
    } else {
        None
    }
}

/// Robot's chassis world matrix in D3D row-major form — the same
/// basis the C++ assembles at MatrixObjectRobot.cpp:443-458 for the
/// normal land / water / move-out states. Used as the parent matrix
/// for `compute_part_chain` and consumed by `pack_part_instance`,
/// which transposes to the shader's column-vector m0..m3 layout.
///
/// D3D row-major assignment (`_11/_12/_13` = side; `_21/_22/_23` =
/// forward; `_31/_32/_33` = up; `_41/_42/_43` = pos). Up is world-Z
/// (slope-fit lands later via `CMatrixMap::GetNormal`); side =
/// `cross(forward, up)`.
fn robot_world_d3d_rowmajor(r: &Robot, cx: f32, cy: f32) -> [[f32; 4]; 4] {
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
    let side = forward.cross(up).normalize_or_zero();
    let fwd_out = up.cross(side).normalize_or_zero();
    [
        [side.x, side.y, side.z, 0.0],
        [fwd_out.x, fwd_out.y, fwd_out.z, 0.0],
        [up.x, up.y, up.z, 0.0],
        [r.pos_x - cx, r.pos_y - cy, r.pos_z, 1.0],
    ]
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
