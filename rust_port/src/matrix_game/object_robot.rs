//! Renderer for `CMatrixRobotAI` — ports `CMatrixRobot::RNeed`
//! (MatrixObjectRobot.cpp:299-357): the chassis `m_Unit[MRT_CHASSIS]`
//! VO (`Matrix\\Robot\\ChassisN.vo` by `ERobotUnitKind`) plus the
//! armor / head / weapon sub-VOs anchored to the chassis matrix
//! per each robot's `RobotConfig`.
//!
//! Instance buffers are rebuilt each frame from the live arena so
//! robots that get spawned mid-session immediately render.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::matrix_game::camera::Camera;
use crate::matrix_game::common::{unpack_rgb, FOG_END, FOG_START};
use crate::matrix_game::effects::landscape_spot::{SpotKind, SpotSpawn};
use crate::matrix_game::effects::point_light::PointLightSystem;
use crate::matrix_game::map::GameMap;
use crate::matrix_game::map_static::{MapStatic, ObjectId, ObjectType, Objects};
use crate::matrix_game::robot::{Animation, ChassisKind, Robot};
use crate::matrix_game::shadow::StencilShadowRenderer;
use crate::matrix_lib::three_g::shadow_stencil::ShadowStencil;
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
///
/// Two bind groups per surface — `bind_group` references the world
/// uniform buffer; `bind_group_preview` references the constructor-
/// preview uniform buffer. The two paths share textures + the
/// per-material UB, so duplicating the bind group only doubles the
/// uniform-binding overhead. Necessary because wgpu's
/// `queue.write_buffer` schedules writes for the next submit and
/// multiple writes to the same buffer coalesce — without a separate
/// preview UB, the world chassis would read preview camera/lighting.
struct SurfaceGpu {
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    bind_group: wgpu::BindGroup,
    bind_group_preview: wgpu::BindGroup,
    /// Enemy variant of `bind_group` — same material/uniform/sampler,
    /// but with the diffuse view swapped to `<top_diffuse>_e` (the
    /// `_e`-suffixed texture the C++ probes for at
    /// MatrixObjectRobot.cpp:335-348). Only populated when the `_e`
    /// variant exists for this surface's diffuse path; render falls
    /// back to `bind_group` otherwise. Used for `robot.side != PLAYER_SIDE`.
    bind_group_enemy: Option<wgpu::BindGroup>,
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
#[derive(Clone, Copy, Debug)]
enum PartKind {
    /// Flyer hull / rotor by EFlyerKind index.
    FlyerBody(usize),
    FlyerVint(usize),
    Chassis(ChassisKind),
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
///
/// `invert` reflects the weapon-slot mirror flag (bit 31 of
/// `WeaponMatrixSlot::access_invert`). The renderer dispatches
/// invert=true draws through `pipeline_inverted` so the cull mode
/// flips to compensate for the flipped X-basis (port of
/// MatrixObjectRobot.cpp:1052-1056 — `D3DCULL_CW`/`CCW` toggle).
#[derive(Clone, Copy)]
struct PartDraw {
    kind: PartKind,
    instance_offset: u32,
    /// Consecutive instances sharing this mesh + frame — sync_robots
    /// sorts and merges draws so one call covers the whole run.
    instance_count: u32,
    vo_frame: usize,
    invert: bool,
    /// `true` when the source robot's side is non-player. Selects
    /// `SurfaceGpu::bind_group_enemy` (the `_e`-textured variant) at
    /// render time when present; falls back to `bind_group` otherwise.
    is_enemy: bool,
}

impl PartDraw {
    /// Merge key: draws with equal keys share mesh, frame, pipeline
    /// and bind-group choice, so contiguous ones batch into a single
    /// instanced draw.
    fn sort_key(&self) -> (u8, usize, usize, bool, bool) {
        let (k, idx) = match self.kind {
            PartKind::Chassis(c) => (0u8, chassis_kind_index(c)),
            PartKind::Armor(i) => (1, i),
            PartKind::Head(i) => (2, i),
            PartKind::Weapon(i) => (3, i),
            PartKind::FlyerBody(i) => (4, i),
            PartKind::FlyerVint(i) => (5, i),
        };
        (k, idx, self.vo_frame, self.invert, self.is_enemy)
    }
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
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
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
/// alongside the D3D row-major world matrix. Weapon entries also
/// carry the slot's invert flag — the renderer flips its cull mode
/// for invert=true draws.
struct PartChain {
    armor: Option<(usize, [[f32; 4]; 4])>,
    head: Option<(usize, [[f32; 4]; 4])>,
    weapons: [Option<WeaponPart>; 5],
}

#[derive(Clone, Copy)]
struct WeaponPart {
    idx: usize,
    world: [[f32; 4]; 4],
    invert: bool,
}

/// Port of `CMatrixRobot::RNeed(MR_Matrix)`'s bone-chain walk
/// (MatrixObjectRobot.cpp:480-575) plus `WeaponInsert`'s slot pick
/// (MatrixObjectRobot.cpp:118-138). Returns world matrices for
/// armor / head / each pilon weapon, expressed in D3D row-major and
/// rooted at `chassis_world`.
///
/// `hull_forward` is the world-space hull direction (m_HullForward).
/// When it diverges from the chassis forward axis the armor/head
/// rotate around Z to match — port of MatrixObjectRobot.cpp:530-536.
///
/// Slot assignment uses the global `default_weapon_matrix` (port of
/// `g_MatrixMap->m_RobotWeaponMatrix[hullno]`) populated at startup by
/// `RobotsRenderer::new`. Mirrors `WeaponInsert`'s preference for
/// `slot[pilon-1]` with a fallback scan for the (pilon)-th compatible
/// slot. The X-basis flip on invert-flagged slots matches
/// MatrixObjectRobot.cpp:515-521; the matching D3DCULL flip lives on
/// the renderer (the inverted pipeline at `pipeline_inverted`).
fn compute_part_chain(
    chassis_world: &[[f32; 4]; 4],
    chassis_gpu: &ChassisGpu,
    chassis_frame: usize,
    armor_gpus: &[Option<ChassisGpu>],
    armor_kind: Option<i32>,
    head_kind: Option<i32>,
    weapon_kinds: &[Option<i32>; 5],
    hull_forward: glam::Vec2,
) -> PartChain {
    // Chassis bone-1 mount at the chassis's live animation frame — the
    // C++ reads GetMatrixById at m_VOFrame (MatrixObjectRobot.cpp:490-492,
    // VectorObject.hpp:539) so armor/head/weapons ride the animated hull
    // (pneumatic bob, idle hover sway) instead of the rest pose.
    let chassis_bone1 = chassis_gpu
        .vo_mesh
        .matrix_by_id(1, chassis_frame)
        .map(flat_to_rows)
        .unwrap_or(IDENTITY_MAT);
    let mut p = row_translate(&chassis_bone1);

    // Armor / head local rotation (MatrixObjectRobot.cpp:529-536):
    //   D3DXMatrixIdentity(&m);
    //   D3DXVec3TransformNormal(&th, &m_HullForward, &m_Core->m_IMatrix);
    //   m._21 = th.x;  m._22 = th.y;
    //   m._11 = th.y;  m._12 = -th.x;
    //
    // `D3DXVec3TransformNormal(out, vec, R^-1)` rotates `vec` into the
    // chassis-local frame. For the rotation matrix we built (rows = side,
    // forward, up), R^-1 = R^T, so:
    //   th.x = dot(hull_forward, side)
    //   th.y = dot(hull_forward, forward)
    // (z component is zero for a planar hull direction).
    let side = glam::Vec3::new(
        chassis_world[0][0],
        chassis_world[0][1],
        chassis_world[0][2],
    );
    let forward = glam::Vec3::new(
        chassis_world[1][0],
        chassis_world[1][1],
        chassis_world[1][2],
    );
    let hf3 = glam::Vec3::new(hull_forward.x, hull_forward.y, 0.0);
    // Normalize so the rotation block stays orthonormal even if the
    // caller passes a non-unit hull_forward. The C++ keeps m_HullForward
    // unit-length; we defend in depth.
    let hf3 = if hf3.length_squared() > 1e-12 {
        hf3.normalize()
    } else {
        // Fallback: hull aligns with chassis forward → identity rotation.
        forward
    };
    let th_x = hf3.dot(side);
    let th_y = hf3.dot(forward);
    let hull_rot: [[f32; 4]; 4] = [
        [th_y, -th_x, 0.0, 0.0],
        [th_x, th_y, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Armor branch (MatrixObjectRobot.cpp:525-545).
    let armor_idx = armor_kind.and_then(|k| if k >= 1 { Some((k - 1) as usize) } else { None });
    let armor_world_opt = armor_idx.and_then(|idx| {
        let armor_gpu = armor_gpus.get(idx).and_then(|o| o.as_ref())?;
        let mut m = hull_rot;
        m[3][0] = p[0];
        m[3][1] = p[1];
        m[3][2] = p[2];
        Some((idx, armor_gpu, matmul(&m, chassis_world)))
    });

    // Advance `p` by the armor's own bone-1 mount
    // (MatrixObjectRobot.cpp:571-574). The C++ uses `m._11/._12/._21/._22`
    // — i.e. the hull rotation block — to project the bone-1 local
    // translation through the same rotation we just applied to armor.
    if let Some((_, armor_gpu, _)) = armor_world_opt.as_ref() {
        if let Some(ab) = armor_gpu.vo_mesh.matrix_by_id(1, 0) {
            let ab = flat_to_rows(ab);
            let tlx = ab[3][0];
            let tly = ab[3][1];
            let tlz = ab[3][2];
            p[0] += tlx * hull_rot[0][0] + tly * hull_rot[1][0];
            p[1] += tlx * hull_rot[0][1] + tly * hull_rot[1][1];
            p[2] += tlz;
        }
    }

    // Head branch (MatrixObjectRobot.cpp:547-559) — same hull-rotation
    // shape as armor, anchored at the post-armor `p`.
    let head_idx = head_kind.and_then(|k| if k >= 1 { Some((k - 1) as usize) } else { None });
    let head_world_opt = head_idx.map(|idx| {
        let mut m = hull_rot;
        m[3][0] = p[0];
        m[3][1] = p[1];
        m[3][2] = p[2];
        (idx, matmul(&m, chassis_world))
    });

    // ── Weapon slot assignment (port of WeaponInsert,
    // MatrixObjectRobot.cpp:118-138) + per-weapon world matrices
    // (MatrixObjectRobot.cpp:504-522). ───────────────────────────
    let mut weapons: [Option<WeaponPart>; 5] = [None; 5];
    if let (Some((_, armor_gpu, armor_world)), Some(armor_kind_v)) =
        (armor_world_opt.as_ref(), armor_kind)
    {
        let matrix = crate::matrix_game::map::default_weapon_matrix(
            crate::matrix_game::config::RobotUnitKind(armor_kind_v),
        );
        let weapon_num = matrix.cnt as usize;
        let slots = &matrix.list[..weapon_num.min(matrix.list.len())];

        for (pilon, wk) in weapon_kinds.iter().enumerate().take(5) {
            let Some(kind) = wk else { continue };
            if *kind < 1 {
                continue;
            }
            let weapon_idx = (*kind - 1) as usize;
            let bit = 1u32 << (*kind - 1);
            // C++ pilon is 1-based (`m_Weapon[nC].m_Pos = nC + 1`,
            // CConstructor.cpp:515) — matches `pilon_idx + 1` here.
            let pilon_1 = (pilon + 1) as i32;

            // WeaponInsert step 1: try preferred slot at index pilon-1
            // (`m_RobotWeaponMatrix[...].list[pilon-1]`).
            let mut fis_pilon_1 = pilon_1;
            let preferred = slots.get((pilon_1 - 1) as usize);
            let preferred_accepts = preferred
                .map(|s| (s.access_invert & bit) != 0)
                .unwrap_or(false);
            if !preferred_accepts {
                // WeaponInsert step 2: scan for the `pilon`-th compatible
                // slot (1-based). MatrixObjectRobot.cpp:126-132 — note
                // this scan does NOT track "used" slots; weapons at
                // distinct pilons can map to the same slot in pathological
                // mixed-type armors. The C++ deliberately accepts that.
                let mut pilon_ost = pilon_1;
                let mut t: i32 = 0;
                while (t as usize) < weapon_num && pilon_ost > 0 {
                    if (slots[t as usize].access_invert & bit) != 0 {
                        pilon_ost -= 1;
                    }
                    t += 1;
                }
                fis_pilon_1 = t;
            }

            let Some(slot) = slots.get((fis_pilon_1 - 1).max(0) as usize) else {
                continue;
            };
            let Some(bone) = armor_gpu.vo_mesh.matrix_by_id(slot.id as u32, 0) else {
                continue;
            };
            let mut wm = matmul(&flat_to_rows(bone), armor_world);
            // MatrixObjectRobot.cpp:515-521 — flip X basis (row 0)
            // when the slot is invert-flagged.
            let invert = (slot.access_invert & (1u32 << 31)) != 0;
            if invert {
                wm[0][0] = -wm[0][0];
                wm[0][1] = -wm[0][1];
                wm[0][2] = -wm[0][2];
            }
            weapons[pilon] = Some(WeaponPart {
                idx: weapon_idx,
                world: wm,
                invert,
            });
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
    /// Mirror of `pipeline` with the cull mode flipped (`Front` instead
    /// of `Back`) for invert-flagged weapon slots. Port of
    /// MatrixObjectRobot.cpp:1052-1056 — `D3DCULL_CW`/`CCW` toggle
    /// around inverted weapon draws so the X-mirror in
    /// `compute_part_chain` doesn't leave back-faces showing.
    pipeline_inverted: wgpu::RenderPipeline,
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
    /// Flyer hull + rotor meshes per EFlyerKind (SPEED/ATTACK/
    /// TRANSPORT/BOMB) and the rotor attach matrix from the body VO.
    flyer_bodies: Vec<Option<ChassisGpu>>,
    flyer_vints: Vec<Option<ChassisGpu>>,
    flyer_vint_attach: Vec<glam::Mat4>,
    /// Per-frame instance buffer: one `InstanceData` per live part
    /// (chassis + armor + head + per-weapon). Written contiguously in
    /// `sync_robots`; each `PartDraw::instance_offset` points into here.
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    draws: Vec<PartDraw>,
    uniform_buffer: wgpu::Buffer,
    /// Mirror of `uniform_buffer`, populated only by
    /// `render_preview_full`. See `SurfaceGpu::bind_group_preview`.
    preview_uniform_buffer: wgpu::Buffer,
    fog_color: [f32; 4],
    ambient_color: [f32; 4],
    light_color: [f32; 4],
    light_dir: [f32; 4],
    time_ms: f32,
    /// On-mesh emissive light billboards (`DrawLights`) collected each
    /// `sync_robots` and flushed into the effects billboard queue.
    light_bbs: Vec<(glam::Vec3, f32, u32)>,
    /// World-center offset applied to every instance so local-
    /// frame vertices share the terrain renderer's origin
    /// (matches the pattern `BuildingsRenderer` uses).
    center: [f32; 2],
    /// Dedicated 1-element instance buffer for the constructor-panel
    /// preview robot (CConstructor.cpp:264-360). Keeps the world
    /// instance buffer untouched so the main pass is unaffected.
    preview_instance_buffer: wgpu::Buffer,
    /// Per-unit stencil shadow volume builders keyed by
    /// `(ObjectId, unit ordinal)` — the `m_Unit[i].m_ShadowStencil`
    /// instances of MatrixObjectRobot.cpp:650-695 / MatrixFlyer.cpp:
    /// 657-686. Robots and flyers always cast SHADOW_STENCIL in the
    /// original (constructor default, MatrixObjectRobot.cpp:36).
    /// Stale entries are evicted at the end of every `sync_robots`.
    stencil_shadows: HashMap<(ObjectId, u8), ShadowStencil>,
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

        let uniform_init = Uniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            fog_color,
            fog_params: [FOG_START, FOG_END, 0.0, 0.0],
            ambient_color,
            light_color,
            light_dir,
            camera_pos: [0.0, 0.0, 0.0, 1.0],
            time_ms: [0.0, 0.0, 0.0, 0.0],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Robots UB"),
            contents: bytemuck::bytes_of(&uniform_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let preview_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Robots Preview UB"),
            contents: bytemuck::bytes_of(&uniform_init),
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
        let pipeline = create_pipeline(device, config, &bgl, false);
        let pipeline_inverted = create_pipeline(device, config, &bgl, true);

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
                                path_fn: &dyn Fn(usize) -> (String, String),
                                tex_cache: &mut HashMap<String, wgpu::TextureView>|
         -> (Vec<Option<ChassisGpu>>, usize) {
            let mut out: Vec<Option<ChassisGpu>> = (0..kind_count).map(|_| None).collect();
            let mut surf_count = 0usize;
            let mut enemy_surf_count = 0usize;
            for n in 1..=kind_count {
                let (vo_path, tex_base) = path_fn(n);
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
                let top_diffuse = tex_base.clone();
                let top_gloss = format!("{}_gloss", tex_base);

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
                        // Probe `<diffuse>_e` for the enemy-tint variant
                        // (MatrixObjectRobot.cpp:335-348). The C++ swap
                        // is anchored on the kind-level path; we
                        // generalise by probing whatever the surface
                        // actually resolved to, so VO-declared skins
                        // that happen to point at the same kind path
                        // also get the swap.
                        let enemy_diffuse_view = material.diffuse.as_deref().and_then(|d| {
                            resolve_texture(
                                Some(&format!("{}_e", d)),
                                device,
                                queue,
                                tex_cache,
                                read_texture,
                            )
                        });
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
                        let make_bg = |ub: &wgpu::Buffer,
                                       diffuse: &wgpu::TextureView,
                                       label: &'static str| {
                            device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some(label),
                                layout: &bgl,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: ub.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::TextureView(diffuse),
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
                            })
                        };
                        let bind_group = make_bg(&uniform_buffer, &diffuse_view, "Robots Part BG");
                        let bind_group_preview = make_bg(
                            &preview_uniform_buffer,
                            &diffuse_view,
                            "Robots Part BG (preview)",
                        );
                        let bind_group_enemy = enemy_diffuse_view
                            .as_ref()
                            .map(|d| make_bg(&uniform_buffer, d, "Robots Part BG (enemy)"));
                        if bind_group_enemy.is_some() {
                            enemy_surf_count += 1;
                        }
                        surfaces.push(SurfaceGpu {
                            index_buffer,
                            num_indices: surf.indices.len() as u32,
                            bind_group,
                            bind_group_preview,
                            bind_group_enemy,
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
            log::info!(
                "robots: part set loaded {}/{} surfaces with `_e` enemy variant",
                enemy_surf_count,
                surf_count,
            );
            (out, surf_count)
        };

        let mut chassis: Vec<Option<ChassisGpu>> = (0..chassis_list.len()).map(|_| None).collect();
        let mut total_surfaces = 0usize;
        let mut chassis_enemy_surf = 0usize;
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
            for frame in vo_mesh.frames.iter() {
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
                    // Probe `<diffuse>_e` (MatrixObjectRobot.cpp:335-348).
                    let enemy_diffuse_view = material.diffuse.as_deref().and_then(|d| {
                        resolve_texture(
                            Some(&format!("{}_e", d)),
                            device,
                            queue,
                            &mut tex_cache,
                            read_texture,
                        )
                    });

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
                    let make_bg =
                        |ub: &wgpu::Buffer, diffuse: &wgpu::TextureView, label: &'static str| {
                            device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some(label),
                                layout: &bgl,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: ub.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::TextureView(diffuse),
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
                            })
                        };
                    let bind_group = make_bg(&uniform_buffer, &diffuse_view, "Robots BG");
                    let bind_group_preview = make_bg(
                        &preview_uniform_buffer,
                        &diffuse_view,
                        "Robots BG (preview)",
                    );
                    let bind_group_enemy = enemy_diffuse_view
                        .as_ref()
                        .map(|d| make_bg(&uniform_buffer, d, "Robots BG (enemy)"));
                    if bind_group_enemy.is_some() {
                        chassis_enemy_surf += 1;
                    }
                    surfaces.push(SurfaceGpu {
                        index_buffer,
                        num_indices: surf.indices.len() as u32,
                        bind_group,
                        bind_group_preview,
                        bind_group_enemy,
                    });
                }
                total_surfaces += surfaces.len();
                frames.push(FrameGpu { surfaces });
            }

            log::debug!(
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
            "robots: {} chassis loaded ({} frame-surface slots, {} with `_e` enemy variant)",
            chassis.iter().filter(|c| c.is_some()).count(),
            total_surfaces,
            chassis_enemy_surf,
        );
        if chassis.iter().all(|c| c.is_none()) {
            return None;
        }

        // Load constructor-preview overlays — armor (6 kinds), head
        // (4 kinds), weapon (10 kinds). Counts mirror MatrixConfig.hpp's
        // ROBOT_*_CNT constants. Missing meshes (older asset bundles
        // pre-dating the pack_bundle update) leave those slots None.
        let robot_path = |prefix: &'static str| {
            move |n: usize| {
                (
                    format!("Matrix/Robot/{}{}.vo", prefix, n),
                    format!("Matrix/Robot/{}{}", prefix, n),
                )
            }
        };
        let (armor, armor_surfs) = load_part_meshes(6, &robot_path("Armor"), &mut tex_cache);
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
        let (head, head_surfs) = load_part_meshes(4, &robot_path("Head"), &mut tex_cache);
        let (weapon, weapon_surfs) = load_part_meshes(10, &robot_path("Weapon"), &mut tex_cache);

        // Flyer hulls + rotors (Models/Flyers — kind order SPEED,
        // ATTACK, TRANSPORT, BOMB; model paths fixed in the shipped
        // robots.dat: {SH,AH,TH,BH}.VO + Screw_*.VO).
        const FLYER_TAGS: [&str; 4] = ["SH", "AH", "TH", "BH"];
        let flyer_body_path = |n: usize| {
            let t = FLYER_TAGS[n - 1];
            (
                format!("Matrix/Flyer/{}.VO", t),
                format!("Matrix/Flyer/{}", t.to_lowercase()),
            )
        };
        let flyer_vint_path = |n: usize| {
            let t = FLYER_TAGS[n - 1];
            (
                format!("Matrix/Flyer/Screw_{}.VO", t),
                // The rotor's diffuse is `TextureVint` —
                // `Matrix\Flyer\screw{kind}` (SCREWSH.DDS etc.).
                format!("Matrix/Flyer/screw{}", t.to_lowercase()),
            )
        };
        let (flyer_bodies, _fb_surfs) = load_part_meshes(4, &flyer_body_path, &mut tex_cache);
        let (flyer_vints, _fv_surfs) = load_part_meshes(4, &flyer_vint_path, &mut tex_cache);
        // Rotor attach point: exporter matrix id 1 of the body VO
        // (the Vint block's `Matrix` param; 1 in the shipped data).
        let flyer_vint_attach: Vec<glam::Mat4> = flyer_bodies
            .iter()
            .map(|b| {
                b.as_ref()
                    .and_then(|g| g.vo_mesh.matrix_by_id(1, 0))
                    .map(|m| glam::Mat4::from_cols_array(&m))
                    .unwrap_or(glam::Mat4::IDENTITY)
            })
            .collect();
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
            pipeline_inverted,
            preview_uniform_buffer,
            chassis,
            armor,
            head,
            weapon,
            flyer_bodies,
            flyer_vints,
            flyer_vint_attach,
            instance_buffer,
            instance_capacity: MAX_LIVE_INSTANCES,
            draws: Vec::new(),
            uniform_buffer,
            fog_color,
            ambient_color,
            light_color,
            light_dir,
            time_ms: 0.0,
            light_bbs: Vec::new(),
            center: [cx, cy],
            preview_instance_buffer,
            stencil_shadows: HashMap::new(),
        })
    }

    /// Flush this frame's on-mesh light billboards into the shared effects
    /// billboard queue (`DrawLights` → `SortIntense`).
    pub fn emit_light_billboards(
        &self,
        q: &mut crate::matrix_lib::three_g::billboard::BillboardQueue,
    ) {
        use crate::matrix_lib::three_g::billboard::{TexRef, BBT_POINTLIGHT};
        for &(pos, radius, color) in &self.light_bbs {
            q.billboard(pos, radius, 0.0, color, TexRef::Bbt(BBT_POINTLIGHT));
        }
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
            surface_h, None, None,
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
        fx: Option<&'a crate::matrix_game::effects::effects_renderer::EffectsRenderer>,
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
                log::debug!(
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
        // Preview uniforms go to the dedicated `preview_uniform_buffer`
        // so the world's `uniform_buffer` (written by `sync_robots`)
        // isn't overwritten — wgpu coalesces queue.write_buffer calls
        // for the same buffer at submit time, which would otherwise
        // make world chassis read preview camera/lighting and render
        // off-screen.
        queue.write_buffer(
            &self.preview_uniform_buffer,
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

        // Constructor preview: hull_forward aligns with chassis
        // forward (CConstructor.cpp:192). chassis_world here is just
        // the turntable spin around Z, so rotating the hull_forward by
        // that same spin keeps the armor identity-aligned within the
        // chassis frame — the rotation cancels out.
        let preview_hull_forward = glam::Vec2::new(chassis_world[1][0], chassis_world[1][1]);
        let chain = compute_part_chain(
            &chassis_world,
            chassis_gpu,
            0, // preview is the rest pose
            &self.armor,
            armor_kind,
            head_kind,
            weapon_kinds,
            preview_hull_forward,
        );

        // Upload all 8 instance slots. Slot assignment:
        //   0 — chassis, 1 — armor, 2 — head, 3..7 — weapons[0..4].
        // Constructor preview tints are flat white (CConstructor.cpp:
        // 340-360 doesn't apply terrain or side colour).
        let pack =
            |m: &[[f32; 4]; 4]| pack_part_instance(m, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]);
        let mut instances = [pack(&IDENTITY_MAT); 8];
        instances[0] = pack(&chassis_world);
        if let Some((_, m)) = chain.armor.as_ref() {
            instances[1] = pack(m);
        }
        if let Some((_, m)) = chain.head.as_ref() {
            instances[2] = pack(m);
        }
        for (i, w) in chain.weapons.iter().enumerate() {
            if let Some(wp) = w {
                instances[3 + i] = pack(&wp.world);
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
        // chassis, armor, weapons (under armor), head. `invert` flips
        // the cull-mode pipeline for mirrored slots (port of
        // MatrixObjectRobot.cpp:1052-1056).
        let mut parts: Vec<(&ChassisGpu, u32, bool)> = Vec::with_capacity(8);
        parts.push((chassis_gpu, 0, false));
        if let Some((idx, _)) = chain.armor.as_ref() {
            if let Some(gpu) = self.armor.get(*idx).and_then(|o| o.as_ref()) {
                parts.push((gpu, 1, false));
            }
        }
        for (pilon, w) in chain.weapons.iter().enumerate() {
            if let Some(wp) = w {
                if let Some(gpu) = self.weapon.get(wp.idx).and_then(|o| o.as_ref()) {
                    parts.push((gpu, 3 + pilon as u32, wp.invert));
                }
            }
        }
        if let Some((idx, _)) = chain.head.as_ref() {
            if let Some(gpu) = self.head.get(*idx).and_then(|o| o.as_ref()) {
                parts.push((gpu, 2, false));
            }
        }

        let mut current_pipeline_inverted = false;
        pass.set_pipeline(&self.pipeline);
        for (gpu, instance_idx, invert) in parts {
            let Some(frame) = gpu.frames.first() else {
                continue;
            };
            if invert != current_pipeline_inverted {
                pass.set_pipeline(if invert {
                    &self.pipeline_inverted
                } else {
                    &self.pipeline
                });
                current_pipeline_inverted = invert;
            }
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
            for surface in &frame.surfaces {
                // Preview path uses the dedicated preview-uniform bind
                // group so the world's `uniform_buffer` (set by
                // `sync_robots`) isn't shadowed at submit time.
                pass.set_bind_group(0, &surface.bind_group_preview, &[]);
                pass.set_index_buffer(surface.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..surface.num_indices, 0, instance_idx..instance_idx + 1);
            }
        }

        // Interface-draw effects (CMatrixRobot::Draw's IsInterfaceDraw
        // branch, MatrixObjectRobot.cpp:1135-1171): antigrav jet
        // streams + repair-gun glow, additive, inside the preview
        // viewport.
        if let Some(fx) = fx {
            use crate::matrix_lib::three_g::billboard::{
                expand_billboard, expand_line, QueuedBillboard, QueuedLine, TexRef,
                BBT_FIRESTREAM1, BBT_REPGLOW, BBT_REPGLOWEND,
            };
            let cam_pos = eye;
            let inv = view.inverse();
            let cam_right = inv.x_axis.truncate().normalize_or_zero();
            let cam_up = inv.y_axis.truncate().normalize_or_zero();
            // The random flicker of the C++ (IRND texture pick, FRND
            // alpha) driven off the renderer clock.
            let flick = (self.time_ms / 60.0) as usize;
            let mut quads: Vec<([glam::Vec3; 4], TexRef, u32)> = Vec::new();

            if matches!(chassis, ChassisKind::AntiGravity) {
                // FireStream jets at chassis bones 10/11
                // (MatrixObjectRobot.cpp:91-95, :578-599).
                let world = glam::Mat4::from_cols_array_2d(&chassis_world);
                for (k, mid) in [10u32, 11u32].into_iter().enumerate() {
                    let Some(m) = chassis_gpu.vo_mesh.matrix_by_id(mid, 0) else {
                        continue;
                    };
                    let pos = world.transform_point3(glam::Vec3::new(m[12], m[13], m[14]));
                    let dir = world
                        .transform_vector3(-glam::Vec3::new(m[4], m[5], m[6]))
                        .normalize_or_zero();
                    let idx = (flick + k * 3) % 5;
                    let l = QueuedLine {
                        p0: pos,
                        p1: pos + dir * 10.0, // m_StreamLen
                        width: 5.0,           // FIRE_STREAM_WIDTH
                        color: 0xFFFF_FFFF,
                        tex: TexRef::Bbt(BBT_FIRESTREAM1 + idx),
                    };
                    quads.push((expand_line(&l, cam_pos), l.tex, l.color));
                }
            }

            // Repair-gun glow — SWeaponRepairData (MatrixRobot.cpp:
            // 24-73): glow line between weapon bones 1/2 + an end
            // sprite on each, nudged 3 units toward the camera,
            // alpha FRND(128)+128.
            const REPAIR_IDX: usize = 9; // RUK_WEAPON_REPAIR - 1
            for (pilon, w) in chain.weapons.iter().enumerate() {
                let Some(wp) = w else { continue };
                if wp.idx != REPAIR_IDX {
                    continue;
                }
                let Some(g) = self.weapon.get(wp.idx).and_then(|o| o.as_ref()) else {
                    continue;
                };
                let ww = glam::Mat4::from_cols_array_2d(&wp.world);
                let (Some(m1), Some(m2)) = (
                    g.vo_mesh.matrix_by_id(1, 0),
                    g.vo_mesh.matrix_by_id(2, 0),
                ) else {
                    continue;
                };
                let mut p0 = ww.transform_point3(glam::Vec3::new(m1[12], m1[13], m1[14]));
                let mut p1 = ww.transform_point3(glam::Vec3::new(m2[12], m2[13], m2[14]));
                p0 += (cam_pos - p0).normalize_or_zero() * 3.0;
                p1 += (cam_pos - p1).normalize_or_zero() * 3.0;
                let a = 128 + ((flick.wrapping_mul(97).wrapping_add(pilon * 31)) % 128) as u32;
                let color = (a << 24) | 0x00FF_FFFF;
                let l = QueuedLine {
                    p0,
                    p1,
                    width: 10.0,
                    color,
                    tex: TexRef::Bbt(BBT_REPGLOW),
                };
                quads.push((expand_line(&l, cam_pos), l.tex, l.color));
                for (p, size) in [(p0, 5.0f32), (p1, 10.0)] {
                    let b = QueuedBillboard {
                        pos: p,
                        scale: size,
                        rot: glam::Vec2::new(1.0, 0.0),
                        disp: glam::Vec2::ZERO,
                        color,
                        tex: TexRef::Bbt(BBT_REPGLOWEND),
                    };
                    quads.push((expand_billboard(&b, cam_right, cam_up), b.tex, b.color));
                }
            }

            fx.draw_preview_quads(queue, pass, &quads, view_proj);
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
    /// `visible` is the per-frame map-group mask; off-screen robots keep
    /// their full logic sync (animation advance, weapon-mount write-back)
    /// but skip draw and shadow emission, like the C++ visible-object
    /// draw lists.
    pub fn sync_robots(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        objs: &mut Objects,
        map: &GameMap,
        point_lights: &PointLightSystem,
        cms: i32,
        visible: &[bool],
        stencil: &mut StencilShadowRenderer,
    ) {
        let _ = device;
        let [cx, cy] = self.center;
        self.draws.clear();

        let light_world = Vec3::new(
            map.light_main_dir[0],
            map.light_main_dir[1],
            map.light_main_dir[2],
        );
        // Track which ids cast shadows this frame so dead objects'
        // ShadowStencil builders can be evicted at the end.
        let mut alive_shadow_ids: std::collections::HashSet<ObjectId> =
            std::collections::HashSet::new();
        // Borrow-friendly handle for the per-unit stencil builders (the
        // loop also borrows self.chassis / self.armor / ... immutably).
        let mut stencil_map = std::mem::take(&mut self.stencil_shadows);
        let mut push_unit_stencil = |id: ObjectId,
                                     unit: u8,
                                     vo: &vector_object::VoMesh,
                                     frame: usize,
                                     world_rows: &[[f32; 4]; 4],
                                     len: f32,
                                     invert: bool| {
            if vo.edges.is_empty() {
                return;
            }
            let world = Mat4::from_cols_array_2d(world_rows);
            // `D3DXVec3TransformNormal(light, m_IMatrix)` — the light in
            // unit-local space (MatrixObjectRobot.cpp:679-680).
            let light_local = world.inverse().transform_vector3(light_world);
            let ss = stencil_map.entry((id, unit)).or_default();
            ss.build(vo, frame, light_local, len, invert);
            if let Some((verts, inds)) = ss.geometry() {
                stencil.push_volume(world, verts, inds);
            }
        };

        let mut instance_data: Vec<InstanceData> = Vec::with_capacity(16);
        let mut offset: u32 = 0;

        // Snapshot the live ids first so we can mutate robots as we
        // iterate (need &mut for anim advance).
        let ids: Vec<_> = objs.iter_live().collect();
        // Pneumatic footprint spawns collected here (can't push to `objs`
        // while a robot is checked out) and drained after the loop.
        let mut pneumatic_decals: Vec<SpotSpawn> = Vec::new();
        // On-mesh emissive light billboards (DrawLights), rebuilt each frame.
        let mut light_bbs_local: Vec<(glam::Vec3, f32, u32)> = Vec::new();
        for id in ids {
            let Some(obj) = objs.get_mut(id) else {
                continue;
            };
            // Flyer hull + spinning rotor (CMatrixFlyer::Draw,
            // MatrixFlyer.cpp:1880-2020 — body/vint unit draws).
            if matches!(obj.core().obj_type, ObjectType::Flyer) {
                let fl: &crate::matrix_game::flyer::Flyer = unsafe {
                    &*(obj as *const dyn MapStatic as *const crate::matrix_game::flyer::Flyer)
                };
                let kidx = fl.kind.clamp(0, 3) as usize;
                if self.flyer_bodies.get(kidx).map(|o| o.is_none()).unwrap_or(true) {
                    continue;
                }
                // Generous radius: flyers hover high above the group AABBs.
                if !crate::matrix_game::water::groups_visible_around(
                    visible, map, fl.pos.x, fl.pos.y, 100.0,
                ) {
                    continue;
                }
                let (s, c) = fl.angle.sin_cos();
                let forward = glam::Vec3::new(-s, c, 0.0);
                let up = glam::Vec3::Z;
                let side = forward.cross(up).normalize_or_zero();
                let body: [[f32; 4]; 4] = [
                    [side.x, side.y, side.z, 0.0],
                    [forward.x, forward.y, forward.z, 0.0],
                    [up.x, up.y, up.z, 0.0],
                    [fl.pos.x - cx, fl.pos.y - cy, fl.pos.z, 1.0],
                ];
                let terrain_color = [1.0, 1.0, 1.0, 1.0];
                let side_color = {
                    let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(fl.side);
                    [sr, sg, sb, 1.0]
                };
                if offset < self.instance_capacity {
                    instance_data.push(pack_part_instance(&body, terrain_color, side_color));
                    self.draws.push(PartDraw {
                        kind: PartKind::FlyerBody(kidx),
                        instance_offset: offset,
                        instance_count: 1,
                        vo_frame: 0,
                        invert: false,
                        is_enemy: fl.side != crate::matrix_game::common::PLAYER_SIDE,
                    });
                    offset += 1;
                    // Flyer stencil shadow (MatrixFlyer.cpp:657-686):
                    // len = pos.z - m_GroundZBase, per-unit light.
                    alive_shadow_ids.insert(id);
                    if let Some(g) = self.flyer_bodies.get(kidx).and_then(|o| o.as_ref()) {
                        push_unit_stencil(
                            id,
                            0,
                            g.vo_mesh.as_ref(),
                            0,
                            &body,
                            fl.pos.z - crate::matrix_game::common::GROUND_Z_BASE,
                            false,
                        );
                    }
                }
                if self.flyer_vints.get(kidx).map(|o| o.is_some()).unwrap_or(false)
                    && offset < self.instance_capacity
                {
                    let attach = self.flyer_vint_attach[kidx].to_cols_array_2d();
                    let (vs, vc) = fl.vint_angle.sin_cos();
                    let spin: [[f32; 4]; 4] = [
                        [vc, vs, 0.0, 0.0],
                        [-vs, vc, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ];
                    let local = matmul(&spin, &attach);
                    let world = matmul(&local, &body);
                    instance_data.push(pack_part_instance(&world, terrain_color, side_color));
                    self.draws.push(PartDraw {
                        kind: PartKind::FlyerVint(kidx),
                        instance_offset: offset,
                        instance_count: 1,
                        vo_frame: 0,
                        invert: false,
                        is_enemy: fl.side != crate::matrix_game::common::PLAYER_SIDE,
                    });
                    offset += 1;
                    if let Some(g) = self.flyer_vints.get(kidx).and_then(|o| o.as_ref()) {
                        push_unit_stencil(
                            id,
                            1,
                            g.vo_mesh.as_ref(),
                            0,
                            &world,
                            fl.pos.z - crate::matrix_game::common::GROUND_Z_BASE,
                            false,
                        );
                    }
                }
                continue;
            }
            if !matches!(obj.core().obj_type, ObjectType::RobotAi) {
                continue;
            }
            let robot: &mut Robot = unsafe { &mut *(obj as *mut dyn MapStatic as *mut Robot) };
            let on_screen = crate::matrix_game::water::groups_visible_around(
                visible,
                map,
                robot.pos_x,
                robot.pos_y,
                40.0,
            );
            // DIP robots render their own part meshes tumbling at the
            // wreck-piece transforms (CMatrixRobot::Draw DIP branch,
            // MatrixObjectRobot.cpp:1045-1061). Texture factor is the
            // fixed 0xFF808080 gray; no shadows.
            if robot.state == crate::matrix_game::robot::RobotState::Dip {
                if !on_screen {
                    continue;
                }
                use crate::matrix_game::robot::DipPart;
                let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(robot.side);
                let side_color = [sr, sg, sb, 1.0];
                let gray = [0.502, 0.502, 0.502, 1.0];
                let is_enemy = robot.side != crate::matrix_game::common::PLAYER_SIDE;
                let t = crate::matrix_game::map::current_elapsed_ms() as f32;
                for u in &robot.dip_units {
                    if u.ttl <= 0.0 || offset >= self.instance_capacity {
                        continue;
                    }
                    let resolved = match u.part {
                        DipPart::Chassis => self
                            .chassis
                            .get(chassis_kind_index(robot.chassis))
                            .and_then(|o| o.as_ref())
                            .map(|g| {
                                (
                                    PartKind::Chassis(robot.chassis),
                                    robot
                                        .chassis_anim
                                        .vo_frame
                                        .min(g.frames.len().saturating_sub(1)),
                                )
                            }),
                        DipPart::Head => non_zero_kind(robot.config.head.kind.0)
                            .map(|k| (k - 1) as usize)
                            .and_then(|idx| {
                                self.head.get(idx).and_then(|o| o.as_ref()).map(|g| {
                                    (
                                        PartKind::Head(idx),
                                        robot
                                            .head_anim
                                            .vo_frame
                                            .min(g.frames.len().saturating_sub(1)),
                                    )
                                })
                            }),
                        DipPart::Weapon(p) => non_zero_kind(robot.config.weapon[p].kind.0)
                            .map(|k| (k - 1) as usize)
                            .and_then(|idx| {
                                self.weapon.get(idx).and_then(|o| o.as_ref()).map(|g| {
                                    (
                                        PartKind::Weapon(idx),
                                        robot.weapon_anims[p]
                                            .vo_frame
                                            .min(g.frames.len().saturating_sub(1)),
                                    )
                                })
                            }),
                    };
                    let Some((kind, vo_frame)) = resolved else {
                        continue;
                    };
                    // D3DXMatrixRotationYawPitchRoll(dy·t, dp·t, dr·t);
                    // the C++ composes a geo-center pivot but then
                    // overwrites the translation with m_Pos
                    // (MatrixRobot.cpp:261-270), so the net world is
                    // plain R + pos. `yaw` keeps the stay-chassis
                    // orientation; `freeze_t` stops the spin clock for
                    // landed pieces.
                    let te = u.freeze_t.unwrap_or(t);
                    let rot = glam::Quat::from_rotation_z(u.yaw)
                        * glam::Quat::from_euler(
                            glam::EulerRot::YXZ,
                            u.dy * te,
                            u.dp * te,
                            u.dr * te,
                        );
                    let rx = rot * Vec3::X;
                    let ry = rot * Vec3::Y;
                    let rz = rot * Vec3::Z;
                    let m: [[f32; 4]; 4] = [
                        [rx.x, rx.y, rx.z, 0.0],
                        [ry.x, ry.y, ry.z, 0.0],
                        [rz.x, rz.y, rz.z, 0.0],
                        [u.pos.x - cx, u.pos.y - cy, u.pos.z, 1.0],
                    ];
                    instance_data.push(pack_part_instance(&m, gray, side_color));
                    self.draws.push(PartDraw {
                        kind,
                        instance_offset: offset,
                        instance_count: 1,
                        vo_frame,
                        invert: u.invert,
                        is_enemy,
                    });
                    offset += 1;
                }
                continue;
            }
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
            // Pneumatic walkers plant their feet (no slide) + leave prints.
            if robot.chassis == ChassisKind::Pneumatic {
                link_pneumatic(robot, vo_mesh, map, &mut pneumatic_decals);
            }
            // BeginMove → Move chaining (MatrixObjectRobot.cpp:1414).
            if robot.animation == Animation::BeginMove && robot.chassis_anim.is_anim_end(vo_mesh) {
                robot.animation = Animation::Move;
                robot.chassis_anim.set_anim_by_name(vo_mesh, "Move", true);
            }
            let vo_frame = robot
                .chassis_anim
                .vo_frame
                .min(chassis_gpu.frames.len().saturating_sub(1));

            // Non-chassis Takt (MatrixObjectRobot.cpp:759-776). For each
            // populated part: advance its anim cursor, and when the
            // current anim ends fall back to "Idle" (matches
            // `m_Graph->SetAnimByName(ANIMATION_NAME_IDLE)` at line 771).
            // Chassis VOs ship "Idle" / "Stay" / "Move" / "Rotate" anims;
            // the non-chassis parts ship a single idle loop that
            // advances at the constant Takt rate.
            tick_part_anim(
                &mut robot.armor_anim,
                self.armor
                    .get(
                        non_zero_kind(robot.config.hull.unit.kind.0)
                            .map(|k| (k - 1) as usize)
                            .unwrap_or(usize::MAX),
                    )
                    .and_then(|o| o.as_ref()),
                cms,
            );
            tick_part_anim(
                &mut robot.head_anim,
                self.head
                    .get(
                        non_zero_kind(robot.config.head.kind.0)
                            .map(|k| (k - 1) as usize)
                            .unwrap_or(usize::MAX),
                    )
                    .and_then(|o| o.as_ref()),
                cms,
            );
            for (pilon, weap_anim) in robot.weapon_anims.iter_mut().enumerate() {
                tick_part_anim(
                    weap_anim,
                    self.weapon
                        .get(
                            non_zero_kind(robot.config.weapon[pilon].kind.0)
                                .map(|k| (k - 1) as usize)
                                .unwrap_or(usize::MAX),
                        )
                        .and_then(|o| o.as_ref()),
                    cms,
                );
            }

            // Per-robot lighting / side tint — same values for every
            // part of this robot.
            let [terrain_r, terrain_g, terrain_b] = unpack_rgb(
                map.static_object_color_with_lighting(robot.pos_x, robot.pos_y, Some(point_lights)),
            );
            let terrain_color = [terrain_r, terrain_g, terrain_b, 1.0];
            let [sr, sg, sb] = crate::matrix_game::side::side_color_rgb(robot.side);
            let side_color = [sr, sg, sb, 1.0];

            // Robot world matrix in D3D row-major form (the same
            // basis `robot_instance` builds, but laid out row-by-row
            // before transpose-pack so the bone chain helper can
            // multiply it as a parent matrix).
            let chassis_world = robot_world_d3d_rowmajor(robot, cx, cy);

            // On-mesh emissive lights (`DrawLights`, MatrixObjectRobot.cpp:
            // 1081-1090). Chassis lights collected here; armor/head/weapon
            // parts add theirs below once the part chain is built.
            collect_part_lights(
                &chassis_world,
                chassis_gpu.vo_mesh.as_ref(),
                vo_frame,
                cx,
                cy,
                self.time_ms,
                &mut light_bbs_local,
            );

            alive_shadow_ids.insert(id);

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
                robot.chassis_anim.vo_frame,
                &self.armor,
                non_zero_kind(armor_kind_i),
                non_zero_kind(head_kind_i),
                &weapon_kinds,
                robot.hull_forward,
            );

            // Armor / head / weapon on-mesh lights (same DrawLights pass as
            // the chassis; each part uses its own world matrix + anim frame).
            if let Some((aidx, aworld)) = chain.armor {
                if let Some(g) = self.armor.get(aidx).and_then(|o| o.as_ref()) {
                    collect_part_lights(
                        &aworld,
                        g.vo_mesh.as_ref(),
                        robot.armor_anim.vo_frame,
                        cx,
                        cy,
                        self.time_ms,
                        &mut light_bbs_local,
                    );
                }
            }
            if let Some((hidx, hworld)) = chain.head {
                if let Some(g) = self.head.get(hidx).and_then(|o| o.as_ref()) {
                    collect_part_lights(
                        &hworld,
                        g.vo_mesh.as_ref(),
                        robot.head_anim.vo_frame,
                        cx,
                        cy,
                        self.time_ms,
                        &mut light_bbs_local,
                    );
                }
            }
            for (pilon, w) in chain.weapons.iter().enumerate() {
                if let Some(wp) = w {
                    if let Some(g) = self.weapon.get(wp.idx).and_then(|o| o.as_ref()) {
                        collect_part_lights(
                            &wp.world,
                            g.vo_mesh.as_ref(),
                            robot.weapon_anims[pilon].vo_frame,
                            cx,
                            cy,
                            self.time_ms,
                            &mut light_bbs_local,
                        );
                    }
                }
            }

            // Write the weapon muzzle transforms back to the game
            // object — the renderer is the only place the real part
            // chain exists. Combine the weapon mesh's barrel bone
            // (matrix id 1) with the pylon world like the C++ fire
            // control does (MatrixRobot.cpp:1390-1394). `wp.world` is
            // in centered render space — add the map center back.
            {
                let mut mounts: [Option<crate::matrix_game::robot::WeaponMount>; 5] = [None; 5];
                let mut glows: [Option<(glam::Vec3, glam::Vec3)>; 5] = [None; 5];
                for (pilon, w) in chain.weapons.iter().enumerate() {
                    let Some(wp) = w else { continue };
                    let bone1 = self
                        .weapon
                        .get(wp.idx)
                        .and_then(|o| o.as_ref())
                        .and_then(|g| g.vo_mesh.matrix_by_id(1, 0))
                        .map(flat_to_rows)
                        .unwrap_or(IDENTITY_MAT);
                    let m = matmul(&bone1, &wp.world);
                    let pos = glam::Vec3::new(m[3][0] + cx, m[3][1] + cy, m[3][2]);
                    let dir = glam::Vec3::new(m[1][0], m[1][1], m[1][2]).normalize_or_zero();
                    mounts[pilon] = Some(crate::matrix_game::robot::WeaponMount { pos, dir });
                    // Repair-gun glow anchors — weapon bones 1/2 world
                    // positions (SWeaponRepairData::Update,
                    // MatrixRobot.cpp:63-73).
                    if wp.idx == 9 {
                        // RUK_WEAPON_REPAIR - 1
                        if let Some(g) = self.weapon.get(wp.idx).and_then(|o| o.as_ref()) {
                            if let (Some(m1), Some(m2)) = (
                                g.vo_mesh.matrix_by_id(1, 0),
                                g.vo_mesh.matrix_by_id(2, 0),
                            ) {
                                let ww = glam::Mat4::from_cols_array_2d(&wp.world);
                                let off = glam::Vec3::new(cx, cy, 0.0);
                                let p0 = ww
                                    .transform_point3(glam::Vec3::new(m1[12], m1[13], m1[14]))
                                    + off;
                                let p1 = ww
                                    .transform_point3(glam::Vec3::new(m2[12], m2[13], m2[14]))
                                    + off;
                                glows[pilon] = Some((p0, p1));
                            }
                        }
                    }
                }
                robot.weapon_mounts = mounts;
                robot.repair_glows = glows;

                // Part origins for the death scatter (`m_Unit[i].m_Pos`
                // seeds at MatrixRobot.cpp:2056-2107 read the unit
                // world matrices; only the renderer has them).
                let uncenter = |m: &[[f32; 4]; 4]| {
                    glam::Vec3::new(m[3][0] + cx, m[3][1] + cy, m[3][2])
                };
                let mut weapons: [Option<(glam::Vec3, bool)>; 5] = [None; 5];
                for (pilon, w) in chain.weapons.iter().enumerate() {
                    if let Some(wp) = w {
                        weapons[pilon] = Some((uncenter(&wp.world), wp.invert));
                    }
                }
                robot.part_snapshot = crate::matrix_game::robot::DipPartSnapshot {
                    chassis: Some(uncenter(&chassis_world)),
                    head: chain.head.as_ref().map(|(_, m)| uncenter(m)),
                    weapons,
                };
            }

            // core.geo_center and core.radius are logic-owned
            // (ChassisKind::mesh_radius + pos_z + 9). This per-frame
            // RENDER sync used to write both from the render-space
            // mesh AABB, fighting the logic's writes every tick —
            // the oscillating target z kept turret pitch from ever
            // locking, and the radius differed between browser and
            // headless sims. The renderer must not touch the core.

            // Resolved per-part vo_frame indices, clamped to each
            // mesh's frame count.
            let armor_frame = chain.armor.as_ref().and_then(|(idx, _)| {
                self.armor.get(*idx).and_then(|o| o.as_ref()).map(|g| {
                    robot
                        .armor_anim
                        .vo_frame
                        .min(g.frames.len().saturating_sub(1))
                })
            });
            let head_frame = chain.head.as_ref().and_then(|(idx, _)| {
                self.head.get(*idx).and_then(|o| o.as_ref()).map(|g| {
                    robot
                        .head_anim
                        .vo_frame
                        .min(g.frames.len().saturating_sub(1))
                })
            });

            // Push chassis instance + part instances. Each gets its
            // own slot in the shared instance buffer with a matching
            // PartDraw entry.
            let is_enemy = robot.side != crate::matrix_game::common::PLAYER_SIDE;
            let push_instance = |instance_data: &mut Vec<InstanceData>,
                                 draws: &mut Vec<PartDraw>,
                                 offset: &mut u32,
                                 cap: u32,
                                 m: &[[f32; 4]; 4],
                                 kind: PartKind,
                                 vo_frame: usize,
                                 invert: bool|
             -> bool {
                if !on_screen {
                    // Culled: report success so callers proceed normally.
                    return true;
                }
                if *offset >= cap {
                    return false;
                }
                instance_data.push(pack_part_instance(m, terrain_color, side_color));
                draws.push(PartDraw {
                    kind,
                    instance_offset: *offset,
                    instance_count: 1,
                    vo_frame,
                    invert,
                    is_enemy,
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
                PartKind::Chassis(robot.chassis),
                vo_frame,
                false,
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
                    armor_frame.unwrap_or(0),
                    false,
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
                    head_frame.unwrap_or(0),
                    false,
                );
            }
            for (pilon, w) in chain.weapons.iter().enumerate() {
                if let Some(wp) = w {
                    let wf = self
                        .weapon
                        .get(wp.idx)
                        .and_then(|o| o.as_ref())
                        .map(|g| {
                            robot.weapon_anims[pilon]
                                .vo_frame
                                .min(g.frames.len().saturating_sub(1))
                        })
                        .unwrap_or(0);
                    push_instance(
                        &mut instance_data,
                        &mut self.draws,
                        &mut offset,
                        self.instance_capacity,
                        &wp.world,
                        PartKind::Weapon(wp.idx),
                        wf,
                        wp.invert,
                    );
                }
            }

            // Per-unit stencil shadow volumes (MatrixObjectRobot.cpp:
            // 650-695): light in unit-local space, len = radius*2 +
            // unit world z - ground. Robots rising out of a base use
            // the deeper base floor (`IsNearBase()` in the C++ — here
            // the spawn/capture states are the near-base cases).
            if on_screen {
                let ground = if matches!(
                    robot.state,
                    crate::matrix_game::robot::RobotState::InSpawn
                        | crate::matrix_game::robot::RobotState::BaseCapture
                ) {
                    crate::matrix_game::common::GROUND_Z_BASE
                } else {
                    crate::matrix_game::common::GROUND_Z
                };
                let r2 = robot.core().radius * 2.0;
                push_unit_stencil(
                    id,
                    0,
                    chassis_gpu.vo_mesh.as_ref(),
                    vo_frame,
                    &chassis_world,
                    r2 + chassis_world[3][2] - ground,
                    false,
                );
                if let Some((idx, m)) = chain.armor.as_ref() {
                    if let Some(g) = self.armor.get(*idx).and_then(|o| o.as_ref()) {
                        push_unit_stencil(
                            id,
                            1,
                            g.vo_mesh.as_ref(),
                            armor_frame.unwrap_or(0),
                            m,
                            r2 + m[3][2] - ground,
                            false,
                        );
                    }
                }
                if let Some((idx, m)) = chain.head.as_ref() {
                    if let Some(g) = self.head.get(*idx).and_then(|o| o.as_ref()) {
                        push_unit_stencil(
                            id,
                            2,
                            g.vo_mesh.as_ref(),
                            head_frame.unwrap_or(0),
                            m,
                            r2 + m[3][2] - ground,
                            false,
                        );
                    }
                }
                for (pilon, w) in chain.weapons.iter().enumerate() {
                    if let Some(wp) = w {
                        if let Some(g) = self.weapon.get(wp.idx).and_then(|o| o.as_ref()) {
                            let wf = robot.weapon_anims[pilon]
                                .vo_frame
                                .min(g.frames.len().saturating_sub(1));
                            push_unit_stencil(
                                id,
                                3 + pilon as u8,
                                g.vo_mesh.as_ref(),
                                wf,
                                &wp.world,
                                r2 + wp.world[3][2] - ground,
                                wp.invert,
                            );
                        }
                    }
                }
            }
        }
        objs.pending_spots.append(&mut pneumatic_decals);
        self.light_bbs = light_bbs_local;

        // Sort draws by mesh/frame and merge contiguous same-key runs
        // into instanced draws — the render pass then binds each mesh
        // once and issues one draw per run instead of one per part.
        // Instance data is permuted to match the new draw order.
        if !self.draws.is_empty() {
            let mut order: Vec<usize> = (0..self.draws.len()).collect();
            order.sort_by_key(|&i| self.draws[i].sort_key());
            let mut sorted_draws: Vec<PartDraw> = Vec::with_capacity(self.draws.len());
            let mut sorted_instances: Vec<InstanceData> =
                Vec::with_capacity(instance_data.len());
            for &i in &order {
                let mut d = self.draws[i];
                sorted_instances.push(instance_data[d.instance_offset as usize]);
                d.instance_offset = (sorted_instances.len() - 1) as u32;
                d.instance_count = 1;
                match sorted_draws.last_mut() {
                    Some(prev)
                        if prev.sort_key() == d.sort_key()
                            && prev.instance_offset + prev.instance_count
                                == d.instance_offset =>
                    {
                        prev.instance_count += 1;
                    }
                    _ => sorted_draws.push(d),
                }
            }
            self.draws = sorted_draws;
            instance_data = sorted_instances;
        }

        if !instance_data.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instance_data),
            );
        }

        // Drop ShadowStencil builders for objects that died this frame.
        stencil_map.retain(|(id, _), _| alive_shadow_ids.contains(id));
        self.stencil_shadows = stencil_map;
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
        let mut current_pipeline_inverted = false;
        // (mesh-ordinal, mesh-idx) of the bound vertex buffer — draws
        // are sorted, so same-mesh runs bind it once.
        let mut bound_mesh: Option<(u8, usize)> = None;
        for draw in &self.draws {
            let gpu = match draw.kind {
                PartKind::FlyerBody(idx) => {
                    let Some(g) = self.flyer_bodies.get(idx).and_then(|o| o.as_ref()) else {
                        continue;
                    };
                    g
                }
                PartKind::FlyerVint(idx) => {
                    let Some(g) = self.flyer_vints.get(idx).and_then(|o| o.as_ref()) else {
                        continue;
                    };
                    g
                }
                PartKind::Chassis(chassis) => {
                    let Some(g) = self
                        .chassis
                        .get(chassis_kind_index(chassis))
                        .and_then(|o| o.as_ref())
                    else {
                        continue;
                    };
                    g
                }
                PartKind::Armor(idx) => {
                    let Some(g) = self.armor.get(idx).and_then(|o| o.as_ref()) else {
                        continue;
                    };
                    g
                }
                PartKind::Head(idx) => {
                    let Some(g) = self.head.get(idx).and_then(|o| o.as_ref()) else {
                        continue;
                    };
                    g
                }
                PartKind::Weapon(idx) => {
                    let Some(g) = self.weapon.get(idx).and_then(|o| o.as_ref()) else {
                        continue;
                    };
                    g
                }
            };
            let Some(frame) = gpu.frames.get(draw.vo_frame) else {
                continue;
            };
            // Switch to the X-mirror pipeline for invert-flagged parts —
            // port of MatrixObjectRobot.cpp:1052-1056 (`D3DCULL_CW`/`CCW`
            // toggle around the inverted weapon's draw).
            if draw.invert != current_pipeline_inverted {
                pass.set_pipeline(if draw.invert {
                    &self.pipeline_inverted
                } else {
                    &self.pipeline
                });
                current_pipeline_inverted = draw.invert;
            }
            let mesh_key = {
                let (k, i, _, _, _) = draw.sort_key();
                (k, i)
            };
            if bound_mesh != Some(mesh_key) {
                pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                bound_mesh = Some(mesh_key);
            }
            for surface in &frame.surfaces {
                // For non-player robots, prefer the `_e`-textured
                // bind group (port of MatrixObjectRobot.cpp:335-348);
                // fall back to default if this surface has no enemy
                // variant (e.g. weapon kinds that ship without a
                // `_e` diffuse).
                let bg = if draw.is_enemy {
                    surface
                        .bind_group_enemy
                        .as_ref()
                        .unwrap_or(&surface.bind_group)
                } else {
                    &surface.bind_group
                };
                pass.set_bind_group(0, bg, &[]);
                pass.set_index_buffer(surface.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                // `first_instance = instance_offset` routes this draw
                // to the robot's slot in the shared instance buffer.
                pass.draw_indexed(
                    0..surface.num_indices,
                    0,
                    draw.instance_offset..(draw.instance_offset + draw.instance_count),
                );
            }
        }
    }
}

/// Port of MatrixObjectRobot.cpp:759-776 — the non-chassis Takt path.
/// Advances the part's animation cursor at the constant rate, and
/// resets to the "Idle" animation on anim-end (mirroring the C++
/// `if (m_Graph->IsAnimEnd()) SetAnimByName(ANIMATION_NAME_IDLE);`).
fn tick_part_anim(state: &mut vector_object::AnimState, gpu: Option<&ChassisGpu>, cms: i32) {
    let Some(gpu) = gpu else {
        return;
    };
    let vo = gpu.vo_mesh.as_ref();
    state.takt(vo, cms);
    if state.is_anim_end(vo) {
        // The C++ `ANIMATION_NAME_IDLE` resolves to the first
        // animation declared by the VO ("Idle" by convention). Try
        // that name first; if the VO doesn't ship it, fall back to
        // re-arming the current animation so the cursor doesn't stay
        // stuck on the last frame.
        if state.set_anim_by_name(vo, "Idle", true) {
            state.first_frame(vo);
        }
    }
}

/// Per-chassis animation speed constants — port of the
/// `ANIMSPEED_CHAISIS_*` macros at MatrixObjectRobot.hpp:59-61.
const ANIMSPEED_TRACK: f32 = 0.20;
const ANIMSPEED_WHEEL: f32 = 0.36;
const ANIMSPEED_PNEU: f32 = 0.155;

/// Port of the chassis-unit branch of `CMatrixRobot::DoAnimation`
/// (MatrixObjectRobot.cpp:778-880). Decides whether to advance the
/// chassis animation cursor this tick and how:
///
///   * STAY/BEGINMOVE/ENDMOVE(+back variants) + Hover/Pneu/Antigrav
///     → normal constant-rate `Takt(cms)` (line 791).
///   * ROTATE → speed-based advance scaled by `k = ANIMSPEED /
///     m_RotSpeed` clamped to 3 for Track/Wheel/Pneu, default k=1
///     otherwise (lines 802-840). After advancing, switches to STAY
///     (line 838).
///   * MOVE/BEGINMOVE/ENDMOVE(+back variants) → speed-based advance
///     scaled by `k = ANIMSPEED / m_Speed` clamped to 3 for
///     Track/Wheel/Pneu, default k=1 otherwise (lines 842-880).
///   * Anything else (e.g. Track/Wheel in STAY) → no advance. The
///     cursor stays put, so the tracks are motionless when the
///     robot is stopped.
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

    if matches!(anim, Animation::Rotate) {
        // ROTATE branch (MatrixObjectRobot.cpp:802-840). The C++ scales
        // the cursor advance by `k = ANIMSPEED / m_RotSpeed`, so a faster
        // rotation cycles the chassis animation faster. Default k=1 —
        // only Track/Wheel/Pneumatic override (:809-822), Hover/AntiGrav
        // still advance.
        let animspeed = match robot.chassis {
            Track => Some(ANIMSPEED_TRACK),
            Wheel => Some(ANIMSPEED_WHEEL),
            Pneumatic => Some(ANIMSPEED_PNEU),
            _ => None,
        };
        let k = if let Some(animspeed) = animspeed {
            let rot_speed = crate::matrix_game::config::global()
                .chassis
                .rotation_speed
                .get(chassis_kind_index(robot.chassis))
                .copied()
                .unwrap_or(0.0)
                .abs()
                .max(1e-3);
            (animspeed / rot_speed).min(3.0)
        } else {
            1.0
        };
        while now_ms > robot.chassis_anim.next_anim_time {
            let frame_time = robot.chassis_anim.next_frame(vo) as f32;
            let add = k * frame_time;
            robot.chassis_anim.next_anim_time += add.max(0.1) as f64;
        }
        // C++ flips back to STAY immediately at line 838 — the seek
        // logic re-issues ROTATE on the next tick if rotation is still
        // in progress. We mirror that here so the chassis animation
        // doesn't get stuck in ROTATE.
        robot.animation = Animation::Stay;
        robot.chassis_anim.set_anim_by_name(vo, "Stay", true);
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
        // Default k=1 — only Track/Wheel/Pneumatic scale by m_Speed
        // (MatrixObjectRobot.cpp:849-865); Hover/AntiGrav still advance.
        let animspeed = match robot.chassis {
            Track => Some(ANIMSPEED_TRACK),
            Wheel => Some(ANIMSPEED_WHEEL),
            Pneumatic => Some(ANIMSPEED_PNEU),
            _ => None,
        };
        let k = if let Some(animspeed) = animspeed {
            // Avoid div-by-zero when robot is stationary; the C++
            // divides directly and relies on m_Speed being non-zero
            // during MOVE states. If it reaches zero we clamp `k` to
            // its upper limit (3), which freezes the animation.
            let speed = robot.speed.abs().max(1e-3);
            (animspeed / speed).min(3.0)
        } else {
            1.0
        };
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
    // `m_Kind - 1` per MatrixConfig.hpp:39-43 — the .vo file index
    // too (Chassis4.vo = Hovercraft, Chassis5.vo = Antigravity).
    c.kind_index()
}

/// Collect a part's `$`-bone light billboards (`DrawLights`). `world` is the
/// part's centred world matrix; the map centre is added back to reach the
/// effects billboard-queue's (uncentred) world space.
fn collect_part_lights(
    world: &[[f32; 4]; 4],
    vo: &vector_object::VoMesh,
    frame: usize,
    cx: f32,
    cy: f32,
    time_ms: f32,
    out: &mut Vec<(glam::Vec3, f32, u32)>,
) {
    for light in &vo.lights {
        if let Some(bm) = vo.matrix_by_id(light.matid, frame) {
            let (lx, ly, lz) = (bm[12], bm[13], bm[14]);
            let wx = lx * world[0][0] + ly * world[1][0] + lz * world[2][0] + world[3][0] + cx;
            let wy = lx * world[0][1] + ly * world[1][1] + lz * world[2][1] + world[3][1] + cy;
            let wz = lx * world[0][2] + ly * world[1][2] + lz * world[2][2] + world[3][2];
            out.push((
                glam::Vec3::new(wx, wy, wz),
                light.radius,
                light.color_at(time_ms as i32),
            ));
        }
    }
}

// ── Pneumatic (walker) foot-linking ─────────────────────────────────────
// Port of `SPneumaticData` + `BuildPneumaticData` / `LinkPneumatic`
// (MatrixObjectRobot.cpp:1508-1821). The chassis animation plants a foot
// each cycle; the body is nudged so the planted foot stays put (no slide),
// and a `SPOT_SOLE_PNEUMATIC` footprint is stamped when a new foot lands.

#[derive(Clone, Copy, Default)]
struct PneumaticFrame {
    foot: (f32, f32),
    other_foot: (f32, f32),
    newlink: bool,
}

struct PneumaticTable {
    fcnt: usize,
    /// `[0..fcnt)` = Move (forward), `[fcnt..2*fcnt)` = MoveBack.
    frames: Vec<PneumaticFrame>,
}

static PNEUMATIC_TABLE: std::sync::OnceLock<Option<PneumaticTable>> = std::sync::OnceLock::new();

/// `BuildPneumaticData` — analyse the Move/MoveBack anims' left/right foot
/// bones (ids 2 and 3) to find, per frame, which foot is planted.
fn build_pneumatic_table(vo: &vector_object::VoMesh) -> Option<PneumaticTable> {
    let bone = |frame_idx: usize, id: u32| -> Option<(f32, f32)> {
        // Row-major D3DXMATRIX: _41 = [12], _42 = [13].
        vo.matrix_by_id(id, frame_idx).map(|m| (m[12], m[13]))
    };
    let move_anim = vo
        .animations
        .iter()
        .find(|a| a.name.eq_ignore_ascii_case("Move"))?;
    let fcnt = move_anim.frames.len();
    if fcnt == 0 {
        return None;
    }
    let mut frames = vec![PneumaticFrame::default(); fcnt * 2];

    // Forward pass: the planted foot at frame i is the one furthest ahead
    // (max bone _42); `newlink` marks where the planted foot switches.
    let fill = |frames: &mut [PneumaticFrame],
                anim: &vector_object::VoAnimation,
                base: usize,
                forward: bool|
     -> Option<()> {
        let (mut leftf, mut ritef) = (0i32, 0i32);
        let (mut maxl, mut maxr) = if forward { (-1e30f32, -1e30f32) } else { (1e30f32, 1e30f32) };
        for i in 0..fcnt {
            let fr = anim.frames.get(i)?.frame_index;
            let (ml, mr) = (bone(fr, 2)?, bone(fr, 3)?);
            if (forward && ml.1 > maxl) || (!forward && ml.1 < maxl) {
                maxl = ml.1;
                leftf = i as i32;
            }
            if (forward && mr.1 > maxr) || (!forward && mr.1 < maxr) {
                maxr = mr.1;
                ritef = i as i32;
            }
        }
        let mut prev = if leftf < ritef { 3 } else { 2 };
        for i in 0..fcnt {
            let fr = anim.frames[i].frame_index;
            let (ml, mr) = (bone(fr, 2)?, bone(fr, 3)?);
            let newv = if i as i32 == leftf {
                2
            } else if i as i32 == ritef {
                3
            } else {
                prev
            };
            let e = &mut frames[base + i];
            if newv == 2 {
                e.newlink = prev != newv;
                prev = newv;
                if e.newlink {
                    e.other_foot = ml;
                    e.foot = mr;
                } else {
                    e.foot = ml;
                }
            } else {
                e.newlink = prev != 3;
                prev = newv;
                if e.newlink {
                    e.other_foot = mr;
                    e.foot = ml;
                } else {
                    e.foot = mr;
                }
            }
        }
        Some(())
    };

    fill(&mut frames, move_anim, 0, true)?;
    if let Some(back) = vo
        .animations
        .iter()
        .find(|a| a.name.eq_ignore_ascii_case("MoveBack"))
    {
        if back.frames.len() == fcnt {
            let _ = fill(&mut frames, back, fcnt, false);
        }
    }
    Some(PneumaticTable { fcnt, frames })
}

fn pneumatic_table(vo: &vector_object::VoMesh) -> Option<&'static PneumaticTable> {
    PNEUMATIC_TABLE
        .get_or_init(|| build_pneumatic_table(vo))
        .as_ref()
}

/// `LinkPneumatic` — stamp footprints on new plants and nudge the body so
/// the planted foot doesn't slide. Uses a flat up-vector (decals/plants sit
/// on near-level ground); the correction is clamped so a mis-built table
/// can never teleport the robot.
fn link_pneumatic(robot: &mut Robot, vo: &vector_object::VoMesh, map: &GameMap, decals: &mut Vec<SpotSpawn>) {
    use crate::matrix_game::common::{trunc_float, CELLFLAG_BRIDGE, CELLFLAG_LAND};
    use crate::matrix_game::map::GLOBAL_SCALE;

    let is_move = robot.animation == Animation::Move;
    let is_back = robot.animation == Animation::MoveBack;
    if !is_move && !is_back {
        robot.link_prev_frame = -1;
        return;
    }
    let Some(table) = pneumatic_table(vo) else {
        return;
    };
    let g_fcnt = table.fcnt as i32;
    if g_fcnt <= 0 {
        return;
    }
    let vof_add = if is_back { g_fcnt } else { 0 };
    let vof = robot.chassis_anim.frame.clamp(0, g_fcnt - 1);

    let fwd = robot.forward.normalize_or_zero();
    let fwd3 = glam::Vec3::new(fwd.x, fwd.y, 0.0);
    let side = fwd3.cross(glam::Vec3::Z).normalize_or_zero();
    let tmp_fwd = glam::Vec3::Z.cross(side).normalize_or_zero();
    let xf = |lp: (f32, f32), px: f32, py: f32| -> (f32, f32) {
        (
            lp.0 * side.x + lp.1 * tmp_fwd.x + px,
            lp.0 * side.y + lp.1 * tmp_fwd.y + py,
        )
    };

    if robot.link_prev_frame < 0 {
        // FirstLinkPneumatic — anchor the planted foot.
        let f = table.frames[(vof + vof_add) as usize].foot;
        let (wx, wy) = xf(f, robot.pos_x, robot.pos_y);
        robot.link_pos = glam::Vec2::new(wx, wy);
        robot.link_prev_frame = vof;
        return;
    }

    let mut guard = 0;
    while robot.link_prev_frame != vof && guard < g_fcnt {
        guard += 1;
        let e = table.frames[(robot.link_prev_frame + vof_add) as usize];
        if e.newlink {
            let (wx, wy) = xf(e.other_foot, robot.pos_x, robot.pos_y);
            robot.link_pos = glam::Vec2::new(wx, wy);
            let cx = trunc_float(wx / GLOBAL_SCALE) as i32;
            let cy = trunc_float(wy / GLOBAL_SCALE) as i32;
            if let Some(fl) = map.unit_flags(cx, cy) {
                if fl & CELLFLAG_LAND != 0 && fl & CELLFLAG_BRIDGE == 0 {
                    decals.push(SpotSpawn {
                        pos: glam::Vec2::new(wx, wy),
                        angle: (-fwd.x).atan2(fwd.y),
                        scale: 1.0,
                        kind: SpotKind::SolePneumatic,
                    });
                }
            }
        }
        robot.link_prev_frame += 1;
        if robot.link_prev_frame >= g_fcnt {
            robot.link_prev_frame = 0;
        }
    }

    // Body correction: shift so the current planted foot lands on link_pos.
    if robot.state != crate::matrix_game::robot::RobotState::Carrying && !robot.is_colliding() {
        let f = table.frames[(vof + vof_add) as usize].foot;
        let (cx, cy) = xf(f, robot.pos_x, robot.pos_y);
        let (dx, dy) = (cx - robot.link_pos.x, cy - robot.link_pos.y);
        let lim = 2.0 * GLOBAL_SCALE;
        if dx * dx + dy * dy < lim * lim {
            robot.pos_x -= dx;
            robot.pos_y -= dy;
        }
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

/// Uncentered world matrix for the robot (raw map coordinates). Mirrors
/// `robot_world_d3d_rowmajor` but skips the renderer-specific center
/// offset — hover-jet effect anchors use raw map points.
pub(crate) fn robot_world_uncentered(r: &Robot) -> Mat4 {
    let f = {
        let v = r.forward;
        let l = v.length();
        if l > 1e-6 {
            v / l
        } else {
            glam::Vec2::new(0.0, 1.0)
        }
    };
    let forward = Vec3::new(f.x, f.y, 0.0);
    let up = Vec3::new(0.0, 0.0, 1.0);
    let side = forward.cross(up).normalize_or_zero();
    let fwd_out = up.cross(side).normalize_or_zero();
    let rot = Mat3::from_cols(side, fwd_out, up);
    Mat4::from_translation(Vec3::new(r.pos_x, r.pos_y, r.pos_z)) * Mat4::from_mat3(rot)
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
    cull_inverted: bool,
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
    let label = if cull_inverted {
        "Robots Pipeline (cull invert)"
    } else {
        "Robots Pipeline"
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
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
            // D3D's default `D3DCULL_CCW` culls CCW-wound triangles in
            // screen space — i.e. CW is front-facing. Mirror that with
            // `front_face: Cw, cull_mode: Back`. The inverted pipeline
            // flips to `cull_mode: Front` so X-mirrored weapons (whose
            // winding is reversed by the basis flip in
            // `compute_part_chain`) render their front faces. Port of
            // MatrixObjectRobot.cpp:1052-1056.
            front_face: wgpu::FrontFace::Cw,
            cull_mode: Some(if cull_inverted {
                wgpu::Face::Front
            } else {
                wgpu::Face::Back
            }),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: crate::matrix_lib::three_g::texture::DEPTH_FORMAT,
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
