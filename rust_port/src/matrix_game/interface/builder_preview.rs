//! Port of the constructor panel's 3D preview renderer. The C++
//! `CConstructor::Render` (CConstructor.cpp:264-360) sets up its own
//! D3DVIEWPORT9 sub-rect, a per-preview directional light, and draws
//! the in-progress robot via `m_Robot->Draw()` with `SetInterfaceDraw(true)`.
//!
//! Our port reuses the existing `object_robot::RobotsRenderer` and
//! emits a `PreviewTicket` here that the form-game render loop feeds
//! to `RobotsRenderer::render_preview_full`. The viewport sub-rect is
//! plumbed through as a scissor rect derived from the Base panel
//! resolved origin + the design-space `preview_view` rect, and the
//! per-preview directional light + ambient (CConstructor.cpp:283-306)
//! are uploaded into the renderer's uniform buffer for the preview
//! draw pass.
//!
//! State owned here:
//! * Cached preview `ChassisKind` from the last `refresh` — lets the
//!   renderer early-out when the config doesn't change.
//! * Design-space viewport rect on the Base panel
//!   (CConstructor.cpp:272-275 uses a static x/y/w/h from
//!   CConstructorPanel's config). We read it from the panel element
//!   named `preview_view`, with a fallback.
//!
//! Limitations vs. the original:
//! * No shadow — the original has `SHADOW_OFF` for the preview anyway.
//! * Bone-anchor matrices (`m_Unit[i].m_LinkMatrix`) for armor / head /
//!   weapon overlays aren't yet extracted from VO files, so the parts
//!   currently stack at the chassis origin instead of slotting onto
//!   the chassis bones. Visually present, slightly mis-positioned;
//!   the per-unit transform drops in once the bone extension lands.

use crate::matrix_game::robot::ChassisKind;
use crate::matrix_game::robot_units::{RobotConfig, RobotUnitKind};

/// A render ticket populated by the refresh step each frame. The
/// actual GPU draw is handled by `RobotsRenderer::render_preview` in
/// a follow-up wiring pass; emitting the ticket here keeps the game
/// logic independent of the GPU plumbing.
#[derive(Debug, Clone, Copy)]
pub struct PreviewTicket {
    pub chassis: ChassisKind,
    /// Design-space panel coords (pre-scale) of the viewport rect.
    pub design_rect: [f32; 4],
    /// Rotation angle (radians) for the slow turntable effect the
    /// original achieves via `m_Forward` rotation + D3DXMatrixLookAtLH.
    pub rotation_rad: f32,
}

#[derive(Debug, Clone, Default)]
pub struct BuilderPreview {
    last_chassis: Option<ChassisKind>,
    /// Accumulated rotation for the turntable.
    angle_ms: f32,
}

impl BuilderPreview {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the turntable by `step_ms`. Called every frame while
    /// the constructor is active.
    pub fn tick(&mut self, step_ms: f32) {
        self.angle_ms += step_ms;
        if self.angle_ms > 100_000.0 {
            // keep it bounded
            self.angle_ms -= 100_000.0;
        }
    }

    /// Build a draw ticket for the given preset. Returns `None` when
    /// the config's chassis isn't set (nothing to render yet).
    pub fn ticket(&mut self, cfg: &RobotConfig, design_rect: [f32; 4]) -> Option<PreviewTicket> {
        let chassis = chassis_from_cfg(cfg)?;
        self.last_chassis = Some(chassis);
        // Slow 0.2 rad/sec turntable — matches the commented-out
        // rotation at CConstructor.cpp:253-258.
        let rotation_rad = self.angle_ms * 0.0002;
        Some(PreviewTicket {
            chassis,
            design_rect,
            rotation_rad,
        })
    }
}

fn chassis_from_cfg(cfg: &RobotConfig) -> Option<ChassisKind> {
    match cfg.chassis.kind {
        k if k == RobotUnitKind::CHASSIS_PNEUMATIC => Some(ChassisKind::Pneumatic),
        k if k == RobotUnitKind::CHASSIS_WHEEL => Some(ChassisKind::Wheel),
        k if k == RobotUnitKind::CHASSIS_TRACK => Some(ChassisKind::Track),
        k if k == RobotUnitKind::CHASSIS_HOVERCRAFT => Some(ChassisKind::Hovercraft),
        k if k == RobotUnitKind::CHASSIS_ANTIGRAVITY => Some(ChassisKind::AntiGravity),
        _ => None,
    }
}
