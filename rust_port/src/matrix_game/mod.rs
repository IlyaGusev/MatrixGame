//! Port of `MatrixGame/src/` — game-specific code (camera, map, water, ...).
//!
//! File names are intentionally kept close to the original C++ files
//! (e.g. `camera.rs` <- `MatrixCamera.cpp`). See CROSSREF.md at the
//! project root for the full mapping.

pub mod camera;
pub mod common;
pub mod config;
pub mod effects;
pub mod form_game;
pub mod interface;
pub mod logic;
pub mod map;
pub mod map_group;
pub mod map_prepare;
pub mod map_static;
pub mod map_trace;
pub mod minimap;
pub mod multi_selection;
pub mod object;
pub mod object_building;
pub mod object_cannon;
pub mod object_robot;
pub mod particles;
pub mod progress_bar;
pub mod render_pipeline;
pub mod robot;
pub mod side;
pub mod ter_surface;
pub mod water;
