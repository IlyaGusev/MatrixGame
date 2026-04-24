//! Port of `CIFaceImage` (Interface/CIFaceImage.{cpp,h}) — the atlas-
//! reference subclass of `CIFaceElement`. `CIFaceImage` in the C++ is
//! a non-renderable handle used as a source for other elements'
//! per-state sub-rects.
//!
//! In this port the per-element logic currently routes through the
//! enum-dispatched [`super::iface_element::IFaceElement`] with
//! `kind = ElementKind::Image`. This file is the canonical home for
//! any future image-only code.
