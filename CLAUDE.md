# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MatrixGame is the Space Rangers 2 Planetary Battles Engine — a DirectX 9.0 based 3D game engine written in C++ targeting Windows (x86). Licensed under GPLv2+.

## Build System

CMake 3.13+, requires MSVC (Visual Studio 2010/2012+) and DirectX 9 SDK.

```bash
# Configure (from repo root)
cmake -B build -G "Visual Studio 16 2019" -A Win32

# Build
cmake --build build --config Release
cmake --build build --config Debug

# Install binaries to bin/
cmake --install build --config Release
```

### CMake Options

- `MATRIXGAME_BUILD_DLL=ON` (default): Builds as DLL for SR2 engine integration
- `MATRIXGAME_BUILD_DLL=OFF`: Builds standalone EXE (`EXE_VERSION` defined)
- `MATRIXGAME_CHEATS=ON`: Enables cheat mode (`CHEATS_ON` defined)

### Output

- Release: `bin/MatrixGame.dll` or `.exe`
- Debug: `bin/Debug/MatrixGame.dll` + `.pdb`

## Architecture

### Module Layout

- **MatrixGame/src/** — Main game engine (~50K LOC)
  - Root: Core classes (CMatrixMapLogic, CRenderPipeline, CMatrixSide, CMatrixRobot, CMatrixBuilding, CMatrixCannon)
  - `Effects/` — Particle system (explosions, flames, plasma, billboards)
  - `Interface/` — UI system (CIFaceElement hierarchy, menus, HUD, CConstructor)
  - `Logic/` — AI (CMatrixAIGroup, CMatrixTactics), pathfinding, environment, game rules
- **MatrixLib/** — Engine utility library
  - `Base/` — Memory management (CHeap), file I/O, strings (CStr/CWStr), config parser (CBlockPar), CRC32
  - `3G/` — DirectX 9 abstraction: rendering, camera, shadows, stencils, vertex/index buffers (BigVB/BigIB)
  - `Bitmap/` — Image operations (includes x86 ASM-optimized sharpening from VirtualDub)
  - `DebugMsg/` — Debug output utilities
  - `FilePNG/` — PNG loading via LibPNG
- **ThirdParty/** — ZLib, LibPNG, LibJPEG
- **Extras/MaxExp/** — 3ds Max exporter plugin (.dle), separate CMake target

### Key Patterns

- **Global singletons**: `g_MatrixHeap`, `g_MatrixData`, `g_MatrixMap`, `g_IFaceList`, `g_Render`
- **Entry point**: `MatrixGameInit()` in `MatrixGame.cpp`
- **Config**: Text-based key=value files parsed by `CBlockPar`
- **String constants**: Centralized in `StringConstants.hpp`
- **Unicode**: Wide-character strings (CWStr) used throughout
- **x86-specific**: 32-bit target with MASM assembly in bitmap processing

### Library Dependencies (linked in CMakeLists.txt)

MatrixGame links: MatrixLib, DebugMsg, FilePNG, ZLIB, winmm, DirectX 9 libs

### Compiler Settings

- Release: `/O2 /Ob2 /Oi /Ot /Oy /MT /Zp1` (1-byte struct packing, static CRT, full optimization)
- Debug: `/Od /RTCc /RTC1 /MTd /Zp1` (runtime checks, static debug CRT)
- Both configs use `/Zp1` (1-byte packing) and `/Gr` (__fastcall convention)

## No Test Suite

There is no automated test infrastructure. Validation is done through in-game integration testing.
