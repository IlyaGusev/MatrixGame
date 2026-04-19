# Rust Port vs C++ Original

This note compares the Rust port in `rust_port/` against the original C++
codebase in `MatrixGame/src/`.

Scope:

- only features that already exist in Rust are considered
- missing gameplay systems are treated as out of scope, not as porting bugs
- the goal is to describe what is faithful, what is approximate, and what is
  currently outside the implemented overlap

## High-Level Summary

The Rust code is still best described as a viewer/rendering port, not a full
game port.

Implemented overlap exists in these areas:

- `.pkg` archive reading
- `STRG` / `CStorage` parsing
- map loading
- terrain height / color sampling
- terrain bottom rendering
- terrain surface overlays
- water rendering
- sky rendering
- decorative static object loading and rendering
- projected static object shadows
- point-light luminance system and visible additive pass
- strategy camera controls

Within that overlap, the port is mixed:

- binary asset loading is fairly close to the original
- terrain atlas construction and terrain geometry are structurally close
- water and decorative objects are implemented as real systems, not stubs
- but several rendering behaviors are still practical approximations rather
  than faithful reproductions of the original engine

So the Rust code should not be described as a faithful port of the implemented
viewer slice. It is a functional reinterpretation with some subsystems close to
the original and some clearly simplified.

## Implemented Areas That Are Broadly Faithful

### Asset formats

Rust faithfully handles the key binary formats used by the implemented viewer:

- `rust_port/src/assets/pkg_reader.rs`
- `rust_port/src/assets/storage.rs`

This is close to the original expectations for:

- `.pkg` archive traversal
- `STRG` record parsing
- ZL02/ZL03 decompression
- `CDataBuf`-style item access

This remains one of the strongest parts of the port.

### Terrain map loading and core height sampling

Rust map parsing and terrain queries are centered in:

- `rust_port/src/game/map.rs`

This overlaps with:

- `MatrixGame/src/MatrixMap.cpp`
- `MatrixGame/src/MatrixMapPrepare.cpp`

In particular, `GameMap::get_z()` is structurally close to `CMatrixMap::GetZ()`:

- same cell lookup idea
- same split-triangle decision
- same flat-cell fast path
- same water rejection behavior

That makes terrain hit/height behavior reasonably faithful inside the viewer
scope.

### Texture-union atlas construction

Rust atlas construction:

- `rust_port/src/game/map_prepare.rs`

Original:

- `MatrixGame/src/MatrixMapPrepare.cpp`, `BuildTexUnions`

Rust keeps the key structure:

- base-tile copy
- overlay composition using masks
- alpha-overlay fallback path
- empty-slot edge extension
- atlas upload with mipmaps

This is one of the more faithful rendering-preparation ports.

### Terrain bottom rendering

Rust terrain-bottom rendering:

- `rust_port/src/renderer/terrain.rs`

Original overlap:

- `MatrixGame/src/MatrixMapGroup.cpp`
- `MatrixGame/src/MatrixMap.cpp`

The Rust code mirrors the same broad workflow:

- parse group data
- batch by texture union
- build world positions
- compute atlas UVs
- apply macrotexture blending
- handle down-cell offseting

The exact render path is different, but the geometry/data flow is close.

### Water is a real renderer, not a stub

Rust water:

- `rust_port/src/renderer/water.rs`

Original overlap:

- `MatrixGame/src/MatrixMapGroup.cpp`
- `MatrixGame/src/MatrixWater.cpp`
- `MatrixGame/src/MatrixVisiCalc.cpp`

Rust includes real implementations for:

- per-group shoreline alpha masks
- animated water lattice deformation
- visible ocean tile expansion outside terrain groups
- frustum-projected water coverage
- water preset texture selection

That is substantial viewer functionality, not a placeholder.

### Decorative static objects are real loaded assets

Rust object loading and rendering:

- `rust_port/src/game/vo_loader.rs`
- `rust_port/src/renderer/objects.rs`

Original overlap:

- `MatrixGame/src/MatrixMapPrepare.cpp`
- `MatrixGame/src/MatrixMapStatic.cpp`
- `MatrixGame/src/MatrixObject.cpp`

Rust loads actual `.vo` meshes with real per-surface material specs (diffuse,
gloss, back, mask, scroll) and renders placed map objects using real map/object
data. Projected static shadows are now a real pass as well (separate
`ShadowBatch` list with its own pipeline), so shadows are no longer absent from
the viewer slice. That keeps the viewer much closer to the original than a
scene made from stand-in meshes would be.

### Point-light system is a real subsystem

Rust:

- `rust_port/src/effects/point_light.rs`

Original overlap:

- `MatrixGame/src/MatrixEffect*.cpp`
- point-color additive pipeline used by `CMatrixMap::GetColor`

Rust implements a standalone `PointLightSystem` that accumulates per-map-point
`[r, g, b]` luminance contributions from every active light, with the same
quadratic falloff shape as the original and a revision counter that lets
terrain and object renderers refresh vertex colors only when the set of lights
changes. It also drives a separate additive visible-light pass on
terrain-conforming geometry. Point lighting is therefore a real system, not an
optional tint.

## Important Differences

## 1. Rust is still a viewer subset, not a gameplay-equivalent port

Rust implements only the rendering/viewer slice plus enough world state to
display maps:

- `rust_port/src/game/world.rs`
- `rust_port/src/game/ai.rs`
- `rust_port/src/renderer/units.rs`
- `rust_port/src/renderer/particles.rs`

Compared with the original C++ game, this is the main architectural gap.

This is not a correctness bug by itself, but it means parity claims must stay
within the viewer/rendering slice.

## 2. Camera terrain-height behavior is still simplified

Rust camera:

- `rust_port/src/renderer/camera.rs`

Rust terrain-height helper:

- `rust_port/src/game/map.rs`, `group_max_z_interpolated()`

Original:

- `MatrixGame/src/MatrixCamera.cpp`
- `MatrixGame/src/MatrixMap.cpp`, `GetZInterpolatedLand()`

The original camera terrain-height path uses smoother land-height sampling.
Rust replaces that with a group-based interpolation / approximation.

Result:

- same purpose
- similar feel
- different smoothing curve
- different coast / ridge behavior

This is not a faithful port of the original terrain-following camera behavior.

## 3. `GetColor()` is only partially faithful

Rust:

- `GameMap::get_color` / `get_color_with_lighting` in
  `rust_port/src/game/map.rs`
- `PointLightSystem::point_lum` in `rust_port/src/effects/point_light.rs`

Original:

- `CMatrixMap::GetColor` in `MatrixGame/src/MatrixMap.cpp`

Rust samples terrain vertex color and now layers the real per-point luminance
contribution from `PointLightSystem` on top — both for terrain vertex colors
and for the terrain color fed into object tinting. That covers the
point-color/luminance half of the original pipeline.

What is still not faithful:

- the original four-corner weighted sample inside a cell
- any extra per-pixel lighting terms and normal-based shading done by the
  original fixed-function state beyond the vertex color path
- some of the clamping / saturation behavior of the original integer pipeline

So sampled map color is closer to the original than before, but it is still
not a direct reproduction of `CMatrixMap::GetColor`.

## 4. Surface overlay material behavior (gloss pass ported)

Rust:

- `rust_port/src/renderer/ter_surface.rs`
- gloss pipeline + WGSL shader in `rust_port/src/renderer/terrain.rs`

Original:

- `MatrixGame/src/MatrixTerSurface.cpp`, `CTerSurface::LoadM`
- `MatrixGame/src/MatrixRenderPipeline.cpp`, `TerSurfMW` / `TerSurfGlossMW`

Rust loads and renders overlay geometry, sorts it by draw index, binds the
referenced textures, and now also handles the gloss material branch:

- `?gloss=<name>` parameters are parsed from each surface-id string; the
  sibling gloss texture is resolved in the same folder as the base texture
  and loaded alongside the atlas texture
- `Matrix/Textures/reflection` is loaded once and reused as the environment
  map
- per-vertex terrain normals are sampled via `GameMap::get_normal` (port of
  `CMatrixMap::GetNormal`, including the bridge / flat / water fast paths)
- a second additive pass draws `gloss.rgb * reflection.rgb` weighted by
  atlas alpha, matching the stage 5 `ADD(TEMP, CURRENT)` step of
  `TerSurfGlossMW` when the single-pass blend is decomposed with `SrcAlpha`
  as the source factor
- reflection UV comes from the camera-space vertex normal using the same
  sphere-map mapping as `water.rs` (`cam_normal.xy * 0.5 + 0.5`)

The bundle packer (`examples/pack_bundle.rs`) follows by packing both the
gloss sibling textures and the reflection texture so the WASM build finds
them.

What remains approximate:

- the sphere-map approximation of `D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR` is
  a plausible stand-in, not a pixel-identical replay of the fixed-function
  texture coordinate generator
- non-macro surfaces (`surfaces/Data`) are still not loaded (the shipped
  atoll map has zero, but other maps may use them via `CTerSurface::Load`)
- the gloss "2" variants (`TerSurfGloss2*`, toggled by certain GPU tiers in
  `MatrixRenderPipeline.cpp`) are not implemented
- non-white per-surface `m_Color` (`TFACTOR`) still applies as a straight
  multiply on the vertex color rather than the two-stage order used by the
  original (not user-visible on atoll — all 756 surfaces ship white — but
  still a divergence if a map uses tinted surfaces)

## 5. Static object loading is present, but object rendering is approximate

Rust:

- `rust_port/src/game/map.rs`
- `rust_port/src/game/vo_loader.rs`
- `rust_port/src/renderer/objects.rs`

Original:

- `MatrixGame/src/MatrixMapPrepare.cpp`
- `MatrixGame/src/MatrixMapStatic.cpp`
- `MatrixGame/src/MatrixObject.cpp`

Rust object placement is reasonably close, object Z placement follows the same
broad "average surrounding corner heights" idea when height data is present,
and `.vo` material specs (diffuse, gloss, back, mask, scroll) are loaded and
passed through to shaders per surface. Projected static shadows are also
ported — `ShadowKind::ProjectedStatic` batches render into a generated shadow
texture and are projected onto terrain overlays as a separate pass.

What is still not faithful:

- the old fixed-function material pipeline is re-implemented as a WGSL
  shader, so gloss / back-layer / mask blending is structurally present but
  not pixel-identical to the original `D3DTSS_*` state
- only projected-static / projected-dynamic shadow kinds are covered; other
  shadow variants (stencil volume, object-owned stencil caster, etc.) are not
  ported
- per-object render-state quirks (custom blend, z-bias, two-sidedness) handled
  by the original `CMatrixObject` code path are approximated, not replicated
- dynamic objects, skeletal animation, and runtime object behavior remain out
  of scope

So object loading is real and shadows are no longer absent, but object
rendering fidelity still trails the original material/state pipeline.

## 6. Water is structurally close, but shoreline sampling still diverges

Rust:

- `rust_port/src/renderer/water.rs`

Original:

- `MatrixGame/src/MatrixMapGroup.cpp`
- `MatrixGame/src/MatrixWater.cpp`
- `MatrixGame/src/MatrixVisiCalc.cpp`

Rust builds shoreline alpha from sampled terrain height. Bridge cells now use
the original's bilinear four-corner sample via `sample_height_for_water`
(matching `CELLFLAG_BRIDGE` handling in the C++ build), so bridges are no
longer an outlier.

What remains approximate:

- non-bridge cells still use `GameMap::get_z` rather than the exact
  per-corner accumulation the original uses along `CELLFLAG_INSHORE` boundaries
- the shared 17×17 wave lattice phase/amplitude and the WaterAlpha_t3 stage
  layout are reproduced, but the mirror / reflection pass is a shader
  approximation rather than a literal replay of the fixed-function stages
- visible ocean tile selection matches the original's frustum-projected
  footprint idea but the tile-culling and sort order are not bit-identical

So shoreline behavior is closer than it was, and the broad water architecture
(per-group alpha masks, animated lattice, ocean tiles, preset textures) is in
place — but it is still not a faithful replay of every edge case in the
original water pipeline.

## 7. Sky rendering still only covers the no-skybox style

Rust:

- `rust_port/src/renderer/sky.rs`

Original:

- `MatrixGame/src/MatrixMap.cpp`

Rust implements the no-skybox style and uses a water-colored lower band to hide
gaps between shoreline alpha water and the background.

This is practical for the viewer, but it is not the full original sky system.
The original has more branching and a broader sky setup than the Rust viewer
currently represents.

So the Rust sky is acceptable as a viewer approximation, but not a faithful
replacement for the full original behavior.

## 8. Camera scope is intentionally narrower

Rust camera comments explicitly say that only strategy mode is implemented:

- `rust_port/src/renderer/camera.rs`

Missing from Rust:

- arcade mode
- fly-cam / other camera modes tied to full game behavior

The original `CMatrixCamera` covers a wider behavioral surface. Rust ports only
the strategy-viewer slice.

## Not Faithful vs Reasonably Faithful

### Reasonably faithful

- `.pkg` reading
- `STRG` / `CStorage` parsing
- ZL02/ZL03 decompression
- map point/unit loading
- terrain `GetZ`-style height sampling
- `GetNormal`-style bilinear normal sampling
- texture-union atlas construction
- terrain bottom geometry generation
- terrain surface overlay base pass + additive gloss/reflection pass
- water as a real renderer (including bridge shoreline sampling)
- decorative `.vo` mesh loading and per-surface material spec parsing
- projected static / dynamic object shadow pass
- `PointLightSystem` luminance accumulation and revision-gated refresh

### Not faithful

- terrain color sampling via `get_color()` — per-corner weighting and
  fixed-function saturation behavior are still approximated
- camera terrain-height interpolation vs `GetZInterpolatedLand()`
- non-macro `surfaces/Data` overlay path (not used by the atoll map), the
  `TerSurfGloss2*` GPU tier, and non-white surface TFACTOR application order
- parts of static object shading/material behavior (gloss/back/mask blend,
  per-object render-state quirks)
- non-projected shadow kinds (stencil volumes, object-owned casters)
- non-bridge shoreline edge cases (inshore corner accumulation, ocean tile
  culling/sort order)
- sphere-map approximation of `TCI_CAMERASPACEREFLECTIONVECTOR` used by both
  terrain surface gloss and water mirror passes
- the full sky system
- anything outside the strategy/viewer slice

## Bottom Line

If the comparison is limited to the functionality already implemented in Rust,
then:

- the port is strongest in binary data loading and terrain preparation
- the port is solid as a map/viewer renderer
- the port is weakest where the original depended on old fixed-function
  material behavior, detailed camera smoothing, or special-case rendering logic
- it is not accurate to call the Rust code a faithful port of the implemented
  viewer/rendering slice
- it is more accurate to call it a functional reinterpretation of that slice,
  with some subsystems close to the original and some visibly simplified
