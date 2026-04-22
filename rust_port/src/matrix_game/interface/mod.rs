//! Port of `MatrixGame/src/Interface/` — `CIFaceElement` / `CIFaceButton`
//! / `CInterface` / `CIFaceList`, plus the 2D UI renderer. The
//! animation / counter / hint / popup-menu subclasses land
//! incrementally.

pub mod iface_element;
pub mod iface_list;
pub mod interface;
pub mod renderer;

pub use iface_element::{ElementKind, ElementState, IFaceElement, StateImage, MAX_STATES};
pub use iface_list::{Click, IFaceList};
pub use interface::{CInterface, MainVisibilityCtx};
pub use renderer::InterfaceRenderer;
