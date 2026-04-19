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

Rust does load actual `.vo` meshes and render placed map objects using real
map/object data. That keeps the viewer much closer to the original than a
scene made from stand-in meshes would be.

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

Original:

- `CMatrixMap::GetColor` in `MatrixGame/src/MatrixMap.cpp`

The original color sampling uses more of the original point-color / luminance
pipeline. Rust samples terrain color and now optionally layers point-light
influence on top, but it is still not a direct reproduction of the original
terrain-color logic.

Effect:

- terrain-derived object tinting can differ
- sampled map color can differ
- any downstream behavior using sampled terrain color is only approximate

This should still be considered not faithful.

## 4. Surface overlays exist, but material behavior is simplified

Rust:

- `rust_port/src/renderer/ter_surface.rs`

Original:

- `MatrixGame/src/MatrixTerSurface.cpp`, `CTerSurface::LoadM`

Rust does load and render overlay geometry, sort it by draw index, and bind the
referenced textures. But the original fixed-function material behavior is not
fully reproduced.

What remains simplified:

- original normal-based overlay lighting behavior
- gloss/material interactions from the old render pipeline
- exact fixed-function texture-stage behavior

So overlays are present, but not faithfully reproduced.

## 5. Static object loading is present, but object rendering is approximate

Rust:

- `rust_port/src/game/map.rs`
- `rust_port/src/game/vo_loader.rs`
- `rust_port/src/renderer/objects.rs`

Original:

- `MatrixGame/src/MatrixMapPrepare.cpp`
- `MatrixGame/src/MatrixMapStatic.cpp`
- `MatrixGame/src/MatrixObject.cpp`

Rust object placement is reasonably close, and object Z placement follows the
same broad "average surrounding corner heights" idea when height data is
present.

But several original behaviors are not ported faithfully:

- original shadow-projection / stencil systems are absent
- the old material pipeline is approximated with modern shader logic
- object gloss/back/mask behavior is only approximate visually
- original object-specific render state and shadow variants are missing

So object loading is real, but object rendering is still only partially
faithful.

## 6. Water is structurally close, but not exact in shoreline edge cases

Rust:

- `rust_port/src/renderer/water.rs`

Original:

- `MatrixGame/src/MatrixMapGroup.cpp`
- `MatrixGame/src/MatrixWater.cpp`
- `MatrixGame/src/MatrixVisiCalc.cpp`

The strongest difference inside the implemented overlap is still shoreline /
height sampling behavior.

Rust builds shoreline alpha from sampled terrain height, but some original edge
cases are not reproduced exactly, especially around special-case terrain/water
transitions such as bridges and related boundary situations.

That means:

- the broad architecture is close
- the visible result is plausible
- but exact shoreline behavior is not fully faithful

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
- texture-union atlas construction
- terrain bottom geometry generation
- water as a real renderer
- decorative `.vo` mesh loading

### Not faithful

- terrain color sampling via `get_color()`
- camera terrain-height interpolation vs `GetZInterpolatedLand()`
- terrain surface overlay material behavior
- parts of static object shading/material behavior
- exact shoreline alpha behavior in edge cases
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
