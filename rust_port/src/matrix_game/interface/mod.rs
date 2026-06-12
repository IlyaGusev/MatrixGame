//! Port of `MatrixGame/src/Interface/` — `CIFaceElement` / `CIFaceButton`
//! / `CInterface` / `CIFaceList`, plus the 2D UI renderer. The
//! animation / counter / hint / popup-menu subclasses land
//! incrementally.

pub mod constructor;
pub mod counter;
pub mod hint;
pub mod history;
pub mod iface_button;
pub mod iface_element;
pub mod iface_image;
pub mod iface_list;
pub mod iface_menu;
pub mod iface_static;
#[allow(clippy::module_inception)]
pub mod interface;
pub mod renderer;
pub mod sound;
pub mod text;

pub use constructor::{FocusTarget, RobotBuilder};
pub use counter::{CIFaceCounter, CheckUpCtx};
pub use hint::{Hint, HintReplacer, HintSystem, TemplateLibrary};
pub use history::ConfigHistory;
pub use iface_element::{ElementKind, ElementState, IFaceElement, StateImage, MAX_STATES};
pub use iface_list::{Click, IFaceList, TurretBuild};
pub mod robot_icons;
pub use interface::{CInterface, MainVisibilityCtx, RobotEntry, RobotPanelCtx, MAX_STACK_ICONS};
pub use renderer::InterfaceRenderer;
pub use robot_icons::RobotIconCache;
