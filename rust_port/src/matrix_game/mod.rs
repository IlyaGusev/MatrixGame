//! Port of `MatrixGame/src/` — game-specific code (camera, map, water, ...).
//!
//! File names are intentionally kept close to the original C++ files
//! (e.g. `camera.rs` <- `MatrixCamera.cpp`). See CROSSREF.md at the
//! project root for the full mapping.

pub mod camera;
pub mod common;
pub mod effects;
pub mod form_game;
pub mod interface;
pub mod logic;
pub mod map;
pub mod map_group;
pub mod map_prepare;
pub mod map_static;
pub mod minimap;
pub mod object;
pub mod object_building;
pub mod particles;
pub mod render_pipeline;
pub mod rnd;
pub mod sky;
pub mod ter_surface;
pub mod units;
pub mod water;
pub mod world;
