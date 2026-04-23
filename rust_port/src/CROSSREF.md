# Rust ↔ Original C++ File Cross-Reference

Mapping between Rust port modules and original Space Rangers 2 files.
When adding a new Rust file, place it where this table predicts — if
there is no row yet, extend the table in the same PR.

Layout mirrors the original tree:

    MatrixLib/Base/      <-> matrix_lib/base/
    MatrixLib/3G/        <-> matrix_lib/three_g/
    MatrixGame/src/      <-> matrix_game/
    MatrixGame/src/Effects/    <-> matrix_game/effects/
    MatrixGame/src/Interface/  <-> matrix_game/interface/
    MatrixGame/src/Logic/      <-> matrix_game/logic/

`gfx/` and `platform/` have no C++ counterpart — they hold wgpu and
WASM/native glue that replaces the original Windows/DirectX platform
layer.

## matrix_lib/base/ (MatrixLib/Base/)

| Rust file                         | Original                        |
|-----------------------------------|---------------------------------|
| matrix_lib/base/bitmap.rs         | MatrixLib/Bitmap/src/CBitmap.cpp|
| matrix_lib/base/blockpar.rs       | MatrixLib/Base/src/CBlockPar.cpp|
| matrix_lib/base/pack.rs           | MatrixLib/Base/src/Pack.cpp     |
| matrix_lib/base/storage.rs        | MatrixLib/Base/src/CStorage.cpp |
| matrix_lib/base/wstr.rs           | MatrixLib/Base/src/CWStr.cpp (field parsers only: `GetStrPar`, `GetIntPar`, `GetDoublePar`, `GetCountPar`, `CompareFirst`) |

Not yet ported: CBuf, CDWORDMap, CException, CFile, CHeap, CList,
CMain, CRC32, CReminder, CStr, Mem, Registry, Tracer. (CWStr partial —
only the read-only field-parsing helpers are in `wstr.rs`.)

## matrix_lib/three_g/ (MatrixLib/3G/)

| Rust file                               | Original                           |
|-----------------------------------------|------------------------------------|
| matrix_lib/three_g/texture.rs           | MatrixLib/3G/src/Texture.cpp       |
| matrix_lib/three_g/vector_object.rs     | MatrixLib/3G/src/VectorObject.cpp  |

Not yet ported: 3g, BigIB, BigVB, Cache, CBillboard, DeviceState,
Form, Helper, Math3D, ShadowProj, ShadowStencil.

## matrix_game/ (MatrixGame/src/)

| Rust file                     | Original                          |
|-------------------------------|-----------------------------------|
| matrix_game/camera.rs         | MatrixCamera.cpp                  |
| matrix_game/common.rs         | Common.hpp                        |
| matrix_game/config.rs         | MatrixConfig.cpp (damage tables only; gamma/keybind/sound deferred) |
| matrix_game/form_game.rs      | MatrixFormGame.cpp (+ MatrixGame.cpp entry glue) |
| matrix_game/map.rs            | MatrixMap.cpp (map data + `MapRenderer` draw orchestration)|
| matrix_game/map_group.rs      | MatrixMapGroup.cpp (BuildBottom + BuildWater — merged across groups by texture; see header) |
| matrix_game/map_prepare.rs    | MatrixMapPrepare.cpp              |
| matrix_game/map_static.rs     | MatrixMapStatic.{cpp,hpp} (base class + ProceedLogic driver + Objects arena) |
| matrix_game/minimap.rs        | MatrixMinimap.cpp                 |
| matrix_game/object.rs         | MatrixObject.cpp (OBJECT_TYPE_MAPOBJECT — decorative rendering + `MapObject` game-object side)|
| matrix_game/object_building.rs| MatrixObjectBuilding.cpp          |
| matrix_game/object_robot.rs   | MatrixObjectRobot.cpp (chassis-only RNeed + per-frame instance sync) |
| matrix_game/particles.rs      | (stub — will split across Effects/) |
| matrix_game/progress_bar.rs   | MatrixProgressBar.cpp (3-segment bar + LIC color, atlas-backed) |
| matrix_game/render_pipeline.rs| MatrixRenderPipeline.cpp          |
| matrix_game/rnd.rs            | MatrixLogic.cpp `CMatrixMapLogic::Rnd` (Park–Miller LCG) |
| matrix_game/side.rs           | MatrixSide.cpp (selection fields only — `m_ActiveObject`, `m_CurrSel`; resources/AI/stats deferred) |
| matrix_game/sky.rs            | DrawSky in MatrixMap.cpp + skybox parts |
| matrix_game/ter_surface.rs    | MatrixTerSurface.cpp              |
| matrix_game/units.rs          | (stub — will become MatrixRobot.cpp etc.) |
| matrix_game/water.rs          | MatrixWater.cpp + BuildWater in MatrixMapGroup.cpp + WaterAlpha_t3 in MatrixRenderPipeline.cpp — will split in Stage 2 part 2 |
| matrix_game/world.rs          | MatrixLogic.cpp `CMatrixMapLogic::Takt` (logic-takt decomposition only; sides/pathfinding deferred) |

Not yet ported: MatrixConfig, MatrixCursor, MatrixDebugInfo,
MatrixFlyer, MatrixInstantDraw, MatrixLoadProgress, MatrixLogic,
MatrixMapTexture, MatrixMapTrace, MatrixMultiSelection,
MatrixObjectCannon, MatrixObjectRobot, MatrixProgressBar, MatrixRobot,
MatrixSampleStateManager, MatrixShadowManager, MatrixSide,
MatrixSkinManager, MatrixSoundManager, MatrixTransition,
MatrixVisiCalc, DevConsole, StringConstants.

## matrix_game/effects/ (MatrixGame/src/Effects/)

| Rust file                               | Original                        |
|-----------------------------------------|---------------------------------|
| matrix_game/effects/point_light.rs      | MatrixEffectPointLight.cpp      |
| matrix_game/effects/selection.rs        | MatrixEffectSelection.cpp (animated dot billboards; CBillboard draw-queue + BBT_SELDOT texture deferred — own pipeline + radial-alpha shader stand in) |
| matrix_game/effects/weapon.rs           | MatrixEffectWeapon.hpp (EWeapon enum constants only; effect class deferred) |

Not yet ported: MatrixEffect (base), BigBoom, Billboard, Dust,
ElevatorField, Explosion, FirePlasma, Flame, Konus, LandscapeSpot,
Lightening, MoveTo, MovingObject, Path, Repair, Selection, Shleif,
SmokeAndFire, Weapon, Zahvat.

## matrix_game/interface/ (MatrixGame/src/Interface/)

| Rust file                               | Original                                           |
|-----------------------------------------|----------------------------------------------------|
| matrix_game/interface/iface_element.rs  | CIFaceElement.{cpp,h} (data portion + hit-test)    |
| matrix_game/interface/iface_list.rs     | CInterface.h::CIFaceList (panel container + events)|
| matrix_game/interface/interface.rs      | CInterface.{cpp,h} (panel struct + `if/<Name>` loader) |
| matrix_game/interface/renderer.rs       | (no C++ peer — wgpu 2D quad pipeline for the HUD)  |

Not yet ported: CAnimation → animation.rs, CConstructor → constructor.rs,
CCounter → counter.rs, CHistory → history.rs, CIFaceButton → iface_button.rs
(folded into iface_element.rs as a kind), CIFaceImage → iface_image.rs,
CIFaceMenu → iface_menu.rs, CIFaceStatic → iface_static.rs (folded too),
MatrixHint → hint.rs.

## matrix_game/logic/ (MatrixGame/src/Logic/)

| Rust file                               | Original                        |
|-----------------------------------------|---------------------------------|
| matrix_game/logic/ai_group.rs           | MatrixAIGroup.cpp (stub)        |

Not yet ported: MatrixEnvironment, MatrixLogicSlot, MatrixRoadNetwork,
MatrixRule, MatrixState, MatrixTactics.

## Top-level files

| Rust file   | Original                     |
|-------------|------------------------------|
| lib.rs      | WASM entry (no C++ analogue) |
| main.rs     | native entry (no C++ analogue) |

## gfx/ and platform/ (no C++ counterpart)

| Rust file        | Purpose                                     |
|------------------|---------------------------------------------|
| gfx/bundle.rs    | WASM asset bundle (packaged textures/maps)  |
| gfx/context.rs   | wgpu device + surface setup                 |
| gfx/loader.rs    | Platform-split file loading                 |
| platform/native.rs | Native-specific time / fs                 |
| platform/web.rs  | WASM-specific time / fs                     |

## shaders/ (WGSL pulled out of the fused renderer files)

The original uses DirectX 9 fixed-function state, not custom shaders.
We need WGSL for wgpu; keeping it next to the Rust that binds it —
not embedded as raw-string constants.

| Shader file                          | Bound by                         |
|--------------------------------------|----------------------------------|
| shaders/minimap.wgsl                 | matrix_game/minimap.rs           |
| shaders/object.wgsl                  | matrix_game/object.rs            |
| shaders/object_building.wgsl         | matrix_game/object_building.rs   |
| shaders/object_shadow.wgsl           | matrix_game/object.rs            |
| shaders/object_shadow_texture.wgsl   | matrix_game/object.rs            |
| shaders/sky_gradient.wgsl            | matrix_game/sky.rs               |
| shaders/sky_skybox.wgsl              | matrix_game/sky.rs               |
| shaders/terrain.wgsl                 | matrix_game/map.rs (MapRenderer) |
| shaders/terrain_gloss.wgsl           | matrix_game/map.rs (MapRenderer) |
| shaders/water.wgsl                   | matrix_game/water.rs             |
| shaders/water_inshore.wgsl           | matrix_game/water.rs             |
