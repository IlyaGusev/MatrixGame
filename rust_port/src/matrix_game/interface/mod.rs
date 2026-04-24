//! Port of `MatrixGame/src/Interface/` — `CIFaceElement` / `CIFaceButton`
//! / `CInterface` / `CIFaceList`, plus the 2D UI renderer. The
//! animation / counter / hint / popup-menu subclasses land
//! incrementally.

pub mod builder_preview;
pub mod constructor;
pub mod counter;
pub mod history;
pub mod iface_element;
pub mod iface_list;
pub mod iface_menu;
#[allow(clippy::module_inception)]
pub mod interface;
pub mod renderer;
pub mod sound;
pub mod text;
pub mod turret_build;

pub use constructor::{FocusTarget, RobotBuilder};
pub use counter::{CIFaceCounter, CheckUpCtx};
pub use history::ConfigHistory;
pub use iface_element::{ElementKind, ElementState, IFaceElement, StateImage, MAX_STATES};
pub use iface_list::{Click, IFaceList};
pub use interface::{CInterface, MainVisibilityCtx};
pub use renderer::InterfaceRenderer;
pub use turret_build::TurretBuild;
