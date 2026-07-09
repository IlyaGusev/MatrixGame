//! Port of `MatrixGame/src/` — game-specific code (camera, map, water, ...).
//!
//! File names are intentionally kept close to the original C++ files
//! (e.g. `camera.rs` <- `MatrixCamera.cpp`). See CROSSREF.md at the
//! project root for the full mapping.

pub mod camera;
#[cfg(test)]
mod combat_tests;
pub mod common;
pub mod config;
pub mod cursor;
pub mod effects;
pub mod flyer;
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
pub mod pause_overlay;
pub mod progress_bar;
pub mod render_pipeline;
pub mod road_network;
pub mod robot;
pub mod shadow;
pub mod side;
pub mod side_ai;
pub mod side_player;
pub mod slot_marker;
pub mod sound;
pub mod ter_surface;
pub mod water;
