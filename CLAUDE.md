# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MatrixGame is the Space Rangers 2 Planetary Battles Engine — a DirectX 9.0 based 3D game engine written in C++ targeting Windows (x86). Licensed under GPLv2+.

## Build System (Original C++)

CMake 3.13+, requires MSVC (Visual Studio 2010/2012+) and DirectX 9 SDK.

```bash
cmake -B build -G "Visual Studio 16 2019" -A Win32
cmake --build build --config Release
cmake --install build --config Release
```

### CMake Options

- `MATRIXGAME_BUILD_DLL=ON` (default): Builds as DLL for SR2 engine integration
- `MATRIXGAME_BUILD_DLL=OFF`: Builds standalone EXE (`EXE_VERSION` defined)
- `MATRIXGAME_CHEATS=ON`: Enables cheat mode (`CHEATS_ON` defined)

### Compiler Settings

- Both configs use `/Zp1` (1-byte struct packing) and `/Gr` (__fastcall convention)
- Release: `/O2 /Ob2 /Oi /Ot /Oy /MT /Zp1`
- Debug: `/Od /RTCc /RTC1 /MTd /Zp1`

## Build System (Rust Port)

```bash
cd rust_port

# Native
cargo build
cargo run  # needs display + Data/robots.pkg

# WASM
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --out-dir pkg

# Pack assets for WASM (extracts textures from robots.pkg); one bundle per map,
# named by slug (lowercase, non-alnum → _). Pack every map the menu lists with:
#   for m in $(cargo run --example pkg_ls -- ../Data/robots.pkg 2>/dev/null \
#     | grep -i '^MATRIX/MAP/.*CMAP$'); do cargo run --example pack_bundle -- "$m"; done
cargo run --example pack_bundle

# Main-menu art for index.html (decodes .GI from mainmenu.pkg + map previews)
cargo run --example menu_assets

# Map names/descriptions/briefings from Lang.dat (menu + in-game Begin/Win/Loose dialogs)
python3 ../tools/lang_dat.py ../Data/Lang.dat --maps assets/menu/maps.txt

# Serve locally
python3 -m http.server 8081
# open http://localhost:8081
```

### Rebuild and Rerun

When asked to "rebuild and rerun":

1. `wasm-pack build --dev --target web --out-dir pkg` (~2s incremental vs ~46s release)
2. Bump `?v=N` → `?v=N+1` in `index.html` import line
3. Ensure a server is on 8081 (a long-lived one is usually already running; `http.server` serves files fresh from disk, so no restart is needed): `ss -tln | grep -q :8081 || nohup python3 -m http.server 8081 > /dev/null 2>&1 &`

When asked to just "rebuild", skip step 3. Use `--dev` by default; only use release (`wasm-pack build --target web --out-dir pkg`) when explicitly requested.

ALWAYS build with opt-level 3: `Cargo.toml` sets `[profile.dev] opt-level = 3` so `--dev` builds are O3 too — never remove that override or build unoptimized.

## Rust Port File Structure (mirrors original C++)

The layout mirrors the original tree; `rust_port/CROSSREF.md` is the
authoritative per-file mapping — when adding a Rust file, place it where
that table predicts and extend the table in the same change.

```
rust_port/src/
├── lib.rs / main.rs   ← WASM / native entry points (no C++ analogue)
├── gfx/               ← wgpu device/surface, asset bundle, file loading
│                        (replaces the DirectX/Windows platform layer)
├── platform/          ← time + WebAudio/native audio glue
├── matrix_lib/        ← MatrixLib/
│   ├── base/          ←   Base/   (blockpar, pack, storage, wstr)
│   ├── bitmap/        ←   Bitmap/
│   └── three_g/       ←   3G/     (billboard, math3d, texture, vector_object)
└── matrix_game/       ← MatrixGame/src/  (one file per Matrix*.cpp:
    │                    camera, config, form_game, logic, map*, minimap,
    │                    object*, robot, side*, sound, water, …)
    ├── effects/       ←   Effects/    (one file per MatrixEffect*.cpp)
    ├── interface/     ←   Interface/  (one file per C*.cpp interface class)
    └── logic/         ←   Logic/      (environment, road_network)
```

WGSL shaders live in `rust_port/shaders/` (the original uses D3D9
fixed-function state; see the table at the end of CROSSREF.md).

## Original C++ Architecture

### Key Data Formats

- **`.pkg`**: Archive format with ZL02/ZL03 compressed files. Header: 4-byte root offset → SFolderRec → SFileRec[]. Record size = 158 bytes under /Zp1.
- **`.CMAP`**: Map files using CStorage (STRG) binary format. Contains heightmap points, texture unions, groups, surfaces, properties.
- **STRG format**: Magic `0x47525453`, version (0=raw, 1=ZL03 compressed), record count, then records. Each record: WStr name, item count, items. Each item: WStr name, u32 type, u32 size, data (CDataBuf).
- **ZL03 format**: `ZL03` magic, i32 block_count, then blocks of (u32 compressed_size, zlib data). NOT a single zlib stream.
- **CDataBuf**: Header (alloc_table_disp, arrays_count, element_type_size), data, then allocation table entries (disp, count, allocated_count).
- **SCompilePoint**: 12 bytes under /Zp1: i32 move, f32 z, u8 b, u8 g, u8 r, u8 flags.
- **SCompileBottomVert**: 8 bytes: u16 x, u16 y, u16 tx, u16 ty.

### Coordinate System

The original uses **X right, Y forward, Z up** (D3D left-handed). The Rust port stores all vertex data in original Z-up coords. A single `z_to_y` conversion matrix in `camera.view_proj()` converts to wgpu's Y-up clip space. **Never scatter Y↔Z swaps throughout the code.**

### Terrain Rendering Pipeline

1. **BuildTexUnions** (MatrixMapPrepare.cpp:108): Builds 1024×1024 texture atlas from 64×64 tiles. Each tile: base PNG + alpha-masked overlays from `bottom/Data` + `bitmaps/Bitmap`. Edge extension for empty slots. Uploaded with 6 mip levels.
2. **BuildBottom** (MatrixMapGroup.cpp:231): Per-group geometry from `groups/Data`. SCompileBottomVert vertices → world positions with atlas UVs and macrotexture UVs. Down-cell vertices offset by `-normal * 0.5`.
3. **TerBotM** (MatrixRenderPipeline.cpp:1198): 3-stage fixed-function: SELECT(atlas) → BLENDTEXTUREALPHA(macrotexture) → MODULATE(vertex_color).
4. **Surface overlays** (MatrixTerSurface.cpp): Triangle strip geometry with per-surface textures. Alpha blended, Z-write off, sorted by draw index.

### Water Rendering Pipeline

1. **BuildWater** (MatrixMapGroup.cpp:366): Per-group 64×64 alpha texture from terrain depth. `up_level=-1.0`, `down_level=-20.1`.
2. **CMatrixWater** (MatrixWater.cpp): Shared 17×17 mesh, sine wave animation per cell (`h[i] = r * sin(angle + phase)`), normal computation from height gradients.
3. **DrawWater** (MatrixMap.cpp:1706): Two passes — alpha pass (per-group with depth alpha texture) + solid pass (opaque water for tiles outside map, computed from camera frustum footprint).
4. **WaterAlpha_t3** (MatrixRenderPipeline.cpp:98): Stage 0 alpha from depth texture, stage 1 MODULATE2X(water_tex, WaterColor), stage 2 BLENDTEXTUREALPHA(mirror_tex using camera-space normal UVs). Lighting: `D3DRS_AMBIENT=WaterColor` + directional light.

### Camera (MatrixCamera.cpp)

- Strategy mode: spherical orbit around link point. `angle_z` (yaw), `angle_x` (pitch), `dist` (distance).
- View matrix: `translate(-lp) * rotZ(-angle_z) * rotX(angle_x)` with distance in rotX, then negate Y/Z columns.
- Mouse: right-drag rotates (yaw speed 0.01 rad/px, pitch 0.0025 rad/px). Wheel zooms (±0.225 per step).
- Smooth interpolation: `mul = 1 - 0.995^ms`.
- FOV: 60° horizontal.

### Critical Implementation Details

- **D3D uses row-major matrices**, glam uses column-major. Multiplication order reverses: D3D `A * B * C` (row-major, v on left) = glam `C * B * A` (column-major, v on right).
- **MergeByMask** (CBitmap.cpp:1280): mask=0 shows overlay, mask=255 shows background. This is **inverted** from typical alpha blending.
- **DXT3 alpha**: Stored in first 8 bytes of each 16-byte block as 4-bit-per-pixel explicit alpha. Must be decoded separately from the color block.
- **Texture atlas seams**: Original hides them via DXT1 compression (4×4 block blending) + 6 mip levels. Without DXT1, need macrotexture overlay and proper mipmaps.
- **Water normal transform**: D3D `TCI_CAMERASPACENORMAL` uses `inverse_transpose(world * view)`. The original world matrix scales by `water_scale=12.5`, which flattens normals. Must replicate this scaling.
- **`queue.write_buffer` on WebGL**: May not reliably update vertex buffers for the current frame. Shader-based animation (via time uniform) is more reliable than CPU buffer updates.
- **WebGL texture limits**: Max dimension 2048px. Canvas and surface config must be clamped. Mipmap uploads via `create_texture_with_data` with `MipMajor` ordering.

## Testing

The original C++ has no tests. The Rust port has:

- `cargo test --lib` — unit tests (~247) embedded in the modules
- `rust_port/tests/` — integration tests against real `Data/` files
- `rust_port/examples/` — headless probes/sims (`game_sim.rs` is the
  autonomous battle harness; see `rust_port/SIM.md`)

Rendering is still validated visually against original game screenshots.
