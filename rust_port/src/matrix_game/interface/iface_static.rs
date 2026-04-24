//! Port of `CIFaceStatic` (Interface/CIFaceStatic.{cpp,h}) — the
//! non-interactable subclass of `CIFaceElement`. The C++ `CIFaceStatic`
//! overrides only `Render`; its data is a bare per-state image.
//!
//! In this port the per-element logic currently routes through the
//! enum-dispatched [`super::iface_element::IFaceElement`] with
//! `kind = ElementKind::Static`. This file is the canonical home for
//! any future static-only code.
