# Rust port file-placement diff — status after refactor pass

Scope: existing Rust items under `rust_port/src/` that directly port a C++
class, struct, or function from a different original file than their Rust file
implies.

## Addressed

Structural / file-level moves completed:

- `MatrixLib/Bitmap` relocated from `matrix_lib/base/bitmap.rs` to
  `matrix_lib/bitmap/mod.rs`, matching the original `MatrixLib/Bitmap/`
  directory.
- `three_g/animation.rs` folded into `three_g/vector_object.rs` —
  `CVectorObjectAnim` lives inside `VectorObject.cpp` / `.hpp` in the C++.
- `CBaseTexture::ParseFlags` helpers (`parse_material_spec`,
  `parse_material_spec_with_prefix`, `resolve_alpha_test_with_txt`,
  `has_trans_suffix`, `merge_materials`, `MaterialSpec`) moved into
  `three_g/texture.rs`; `vector_object.rs` re-exports them for compat.
- `resolve_paths` + `ResolvedObjectPaths` / `ShadowSpec` / `ShadowKind`
  + `parse_shadow_spec` moved from `three_g/vector_object.rs` to
  `matrix_game/object.rs` (MatrixObject.cpp id-string parsing).
- `matrix_game/rnd.rs` folded into `logic.rs` — `Rnd` is
  `CMatrixMapLogic::Rnd` / `m_Rnd` from `MatrixLogic.cpp`.
- `matrix_game/orders.rs` folded into `robot.rs` — `SOrder` +
  `m_OrdersList` are `CMatrixRobotAI` members.
- `matrix_game/sky.rs` folded into `map.rs` — `DrawSky` +
  `m_SkyAngle` advance are `CMatrixMap` code; no separate
  `MatrixSky.cpp` in the original.
- `interface/builder_preview.rs` folded into `interface/constructor.rs`
  — constructor-preview turntable is `CConstructor::Render` state.
- `interface/turret_build.rs` folded into `interface/iface_list.rs` —
  turret placement state ports `CInterface::BeginBuildTurret` +
  `CIFaceList::m_BuildCa` flags.
- `matrix_game/robot_units.rs` scattered:
  - `Resource` / `MAX_RESOURCES` / `RobotUnitKind` / `ROBOT_*_CNT`
    → `matrix_game/config.rs` (MatrixConfig.hpp).
  - `RobotUnitType` / `MAX_WEAPON_CNT` / `MR_MAXUNIT`
    → `matrix_game/object_robot.rs` (MatrixObjectRobot.hpp).
  - `UnitPrice` / `Unit` / `ArmorUnit` / `WeaponUnit` / `RobotConfig`
    → `matrix_game/interface/constructor.rs` (Interface/CConstructor.h).
  - `WeaponMatrix` / `WeaponMatrixSlot` / `default_weapon_matrix`
    / `ROBOT_WEAPONS_PER_ROBOT_CNT` / `ACCESS_EXTRA_BIT_*`
    → `matrix_game/map.rs` (MatrixMap.hpp).
  All call sites updated to import from the new locations.
- `matrix_game/multi_selection.rs` created. `MarqueeRenderer` moved
  from `effects/marquee.rs`; `marquee_select` free function moved from
  `logic.rs`. `CMultiSelection::Add/Remove/End` (shift-click +
  marquee-end) remain on `Side::select_toggle/select_replace`; their
  docstrings cross-reference `MatrixMultiSelection.cpp`.
- `matrix_game/map_trace.rs` repurposed to mirror `MatrixMapTrace.cpp`.
  The A* pathfinder (`Blocker`, `MovePath`, `footprint_passable`,
  `find_path`, `optimize_path`, `path_total_length`, `waypoint_to_world`,
  `ROBOT_FOOTPRINT_HALF`) moved into `logic.rs` (MatrixLogic.cpp), and
  re-exported from `map_trace.rs` so `robot.rs` keeps its existing
  import path. `pick_object` (object-scan branch of
  `CMatrixMap::Trace`) exposed as a free function in `map_trace.rs`.
- Per-subclass interface mirror files added: `iface_button.rs`,
  `iface_static.rs`, `iface_image.rs` each mirror
  `CIFaceButton.{cpp,h}` / `CIFaceStatic.{cpp,h}` / `CIFaceImage.{cpp,h}`.
  Current port uses enum dispatch on `IFaceElement.kind`; new files
  carry the doc anchor so per-subclass code lands in the right place
  as methods are un-folded.
- `side_color_rgb` / `side_color_minimap_rgb` moved from `side.rs` to
  `map.rs` (port of `CMatrixMap::GetSideColor` / `GetSideColorMM`);
  re-exported from `side.rs`.
- `robot_build_time_ms` / `turret_build_time_ms` moved from
  `object_building.rs` to `config.rs` (`g_Config.m_Timings` accessors);
  re-exported from `object_building.rs`.

Native build (`cargo check`, `cargo build`) and WASM-target lib build
(`cargo check --lib --target wasm32-unknown-unknown`) both pass after
every change. No functional regressions — every move preserved the
public surface either by re-export or by an identical free-function
signature.

## Deferred — still mis-placed

These need a focused follow-up pass. The items are grouped by the
reason they're harder than a straight file move.

### Threading state across files

Moving these requires rethreading `&mut` access to fields that live on
`GameMap` / `MapLogic` / `Objects` / `AppState`. The ergonomic call
sites in the current layout would have to change shape.

- `logic.rs` — `curr_sel_for`, `click_at_screen`, `order_move_to_at`,
  `compute_max_side_robots`, `compute_resource_income` → `side.rs`
  (these are `CMatrixSideUnit` methods in C++).
- `logic.rs` — `accrue_resources` → `object_building.rs`
  (`CMatrixBuilding::LogicTakt` resource-timer branch).
- `logic.rs` — `screen_to_terrain_xy` → `camera.rs`
  (`CMatrixCamera::GetCursorOnMap`).
- `map.rs` — `compute_normals`, `load_move_cells`,
  `load_inshore_prespawns`, `load_objects`, `load_buildings`
  → `map_prepare.rs` (`MatrixMapPrepare.cpp`).
- `map.rs` — `MoveCell::stop_mask` / `is_impassable_for` → `logic.rs`
  (`CMatrixMapLogic::IsAbsenceWall` bit math).
- `map.rs` — `sync_building_animation` → `object_building.rs`.
- `map.rs` — `bake_minimap` → `minimap.rs` (`CMinimap::RenderBackground`).
- `map.rs` — `MapRenderer` render orchestration → split so
  render-pipeline state lives in `render_pipeline.rs`.
- `map_static.rs` — `find_objects`, `any_object_in_radius` → `map.rs`.
- `object_building.rs` — `BuildStack::tick_timer` robot-construction
  branch + `pick_balanced_team` → `interface/constructor.rs`.
- `object_robot.rs` — `render_preview`, `render_preview_full`
  → `interface/constructor.rs`.
- `robot.rs` — `Animation`, `switch_animation`, `z_from_pos`
  → `object_robot.rs` (graphical `CMatrixRobot` functions).

### Minimap parsing helpers

- `minimap.rs` — `Button`, `parse_minim_button`, `parse_mmp_static`
  → `interface/` element parsers.
- `minimap.rs` — `parse_side_colors_mm` → `map.rs`.

### Water / visibility split

- `water.rs` — `bake_minimap_all` → `minimap.rs`.
- `water.rs` — `collect_visible_solid_tiles`, `check_candidate`,
  and `camera.rs::frustum_bounds_on_plane_zup` → a new `visi_calc.rs`
  mirror of `MatrixVisiCalc.cpp` (or fold into `map.rs`).
- `water.rs` — `InshoreSystem::takt` → `map_group.rs`
  (`CMatrixMapGroup::GraphicTakt` shoreline advancement).

### UI dispatcher split

- `hint.rs` — `correct_coordinates` → `interface.rs`/`iface_list.rs`.
- `iface_menu.rs` — pylon popup builders (`for_chassis` / `for_hull`
  / `for_head` / `for_weapon_normal` / `for_weapon_extern` /
  `popup_for_pylon`) → `iface_button.rs` (decision tree belongs to
  `CIFaceButton::OnMouseRBDown`).

### `form_game.rs` dispatcher split

`form_game.rs` still contains 12 dispatcher/query functions that
should move to the classes whose methods they port:
`dispatch_ui_click`, `dispatch_ui_right_click`, `preview_popup_hover`,
`refresh_hint_replacements`, `tick_builder_preview`,
`BuilderPreviewQuery`, `builder_preview_query`, `try_place_turret`,
`build_counter_ctx`, `commit_and_queue_robot`, `refresh_progress_bars`,
`refresh_interface_visibility`, `sync_selection_ring`. These pull in
state from `side.rs`, `interface/*.rs`, `object_building.rs`, and
`effects/selection.rs`; splitting them is a coordinated refactor, not
a series of individual moves.
