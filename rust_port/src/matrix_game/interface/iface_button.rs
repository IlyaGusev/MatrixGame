//! Port of `CIFaceButton` (Interface/CIFaceButton.{cpp,h}) — the
//! interactable subclass of `CIFaceElement`. The full C++ class carries
//! its own `OnMouseLBDown/Up` / `OnMouseRBDown/Up` overrides and the
//! `PGOrder*` popup-menu launchers (MatrixFormGame.cpp click handlers
//! delegate to them).
//!
//! In this port the per-element logic currently routes through the
//! enum-dispatched [`super::iface_element::IFaceElement`] with
//! `kind = ElementKind::Button`; button-only state is folded onto the
//! shared element struct. This file is the canonical home for any
//! future button-only code (e.g. the RBDown pylon-popup decision tree
//! — see the `iface_menu` pylon_for_* helpers, which port parts of
//! `CIFaceButton::OnMouseRBDown` and logically belong here).
