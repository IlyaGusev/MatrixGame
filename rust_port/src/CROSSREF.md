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
| matrix_lib/three_g/billboard.rs         | MatrixLib/3G/src/CBillboard.cpp (queue + quad/line expansion; GPU flush in effects_renderer.rs) |
| matrix_lib/three_g/math3d.rs            | MatrixLib/3G/src/Math3D.cpp (CTrajectory + VecToMatrixX/Y) |
| matrix_lib/three_g/texture.rs           | MatrixLib/3G/src/Texture.cpp       |
| matrix_lib/three_g/vector_object.rs     | MatrixLib/3G/src/VectorObject.cpp  |

Not yet ported: 3g, BigIB, BigVB, Cache, CBillboard, DeviceState,
Form, Helper, ShadowProj, ShadowStencil. (Math3D partial — CTrajectory.)

## matrix_game/ (MatrixGame/src/)

| Rust file                     | Original                          |
|-------------------------------|-----------------------------------|
| matrix_game/camera.rs         | MatrixCamera.cpp                  |
| matrix_game/common.rs         | Common.hpp                        |
| matrix_game/config.rs         | MatrixConfig.cpp (damage/radius/cooldown/overheat tables, cannon props, difficulty; gamma/keybind/sound deferred) |
| matrix_game/form_game.rs      | MatrixFormGame.cpp (+ MatrixGame.cpp entry glue) |
| matrix_game/map.rs            | MatrixMap.cpp (map data + `MapRenderer` draw orchestration)|
| matrix_game/map_group.rs      | MatrixMapGroup.cpp (BuildBottom + BuildWater — merged across groups by texture; see header) |
| matrix_game/map_prepare.rs    | MatrixMapPrepare.cpp              |
| matrix_game/map_static.rs     | MatrixMapStatic.{cpp,hpp} (base class + ProceedLogic driver + Objects arena) |
| matrix_game/minimap.rs        | MatrixMinimap.cpp                 |
| matrix_game/object.rs         | MatrixObject.cpp (OBJECT_TYPE_MAPOBJECT — decorative rendering + `MapObject` game-object side)|
| matrix_game/object_building.rs| MatrixObjectBuilding.cpp          |
| matrix_game/combat_tests.rs   | (test-only — weapon/damage behavior tests) |
| matrix_game/object_robot.rs   | MatrixObjectRobot.cpp (chassis-only RNeed + per-frame instance sync) |
| matrix_game/logic.rs          | MatrixLogic.cpp (CMatrixMapLogic: Takt driver, Place*/IsAbsenceWall helpers; also module root for Logic/ subsystems) |
| matrix_game/map_trace.rs      | MatrixMapTrace.cpp (CMatrixMap::Trace hitscan + FindLocalPath A* + OptimizeMovePath) |
| matrix_game/orders.rs         | MatrixRobot.hpp SOrder/OrderType/OrderPhase + AllocPlaceForOrderOnTop pool |
| matrix_game/particles.rs      | (stub — will split across Effects/) |
| matrix_game/progress_bar.rs   | MatrixProgressBar.cpp (3-segment bar + LIC color, atlas-backed) |
| matrix_game/render_pipeline.rs| MatrixRenderPipeline.cpp          |
| matrix_game/rnd.rs            | MatrixLogic.cpp `CMatrixMapLogic::Rnd` (Park–Miller LCG) |
| matrix_game/side.rs           | MatrixSide.cpp (selection fields only — `m_ActiveObject`, `m_CurrSel`; resources/AI/stats deferred) |
| matrix_game/sky.rs            | DrawSky in MatrixMap.cpp + skybox parts |
| matrix_game/ter_surface.rs    | MatrixTerSurface.cpp              |
| matrix_game/robot.rs          | MatrixRobot.cpp (CMatrixRobotAI — spawn flow + move-out + MoveTo + GetLost + SBotWeapon fire control / heat / Damage / ablaze-shorted DOT) |
| matrix_game/water.rs          | MatrixWater.cpp + BuildWater in MatrixMapGroup.cpp + WaterAlpha_t3 in MatrixRenderPipeline.cpp — will split in Stage 2 part 2 |

Not yet ported: MatrixConfig, MatrixCursor, MatrixDebugInfo,
MatrixFlyer, MatrixInstantDraw, MatrixLoadProgress,
MatrixMapTexture, MatrixMultiSelection,
MatrixObjectCannon, MatrixSampleStateManager, MatrixShadowManager,
MatrixSkinManager, MatrixSoundManager, MatrixTransition,
MatrixVisiCalc, DevConsole, StringConstants.

## matrix_game/effects/ (MatrixGame/src/Effects/)

| Rust file                               | Original                        |
|-----------------------------------------|---------------------------------|
| matrix_game/effects/point_light.rs      | MatrixEffectPointLight.cpp      |
| matrix_game/effects/selection.rs        | MatrixEffectSelection.cpp (animated dot billboards; CBillboard draw-queue + BBT_SELDOT texture deferred — own pipeline + radial-alpha shader stand in) |
| matrix_game/effects/big_boom.rs         | MatrixEffectBigBoom.cpp (blast sweep + geosphere shell) |
| matrix_game/effects/billboard_fx.rs     | MatrixEffectBillboard.cpp (TTL'd billboard-line effects) |
| matrix_game/effects/effects_renderer.rs | (no direct peer — wgpu flush for CBillboard::SortEndDraw + the BBT table of MatrixEffect.cpp InitEffects + CVectorObject draws for bullets/debris) |
| matrix_game/effects/explosion.rs        | MatrixEffectExplosion.cpp (all presets, sparks/intense/fire/mesh debris + FireAnim) |
| matrix_game/effects/fire_plasma.rs      | MatrixEffectFirePlasma.cpp (bolt movement + hit + sprites) |
| matrix_game/effects/flame.rs            | MatrixEffectFlame.cpp (puffs: damage sweep + 10-billboard chains) |
| matrix_game/effects/konus.rs            | MatrixEffectKonus.cpp (cone + splash variants) |
| matrix_game/effects/landscape_spot.rs   | MatrixEffectLandscapeSpot.cpp (voronka / plasma-hit / constant decals) |
| matrix_game/effects/lightening.rs       | MatrixEffectLightening.cpp (bolt + shorted arcs) |
| matrix_game/effects/moving_object.rs    | MatrixEffectMovingObject.cpp (gun/cannon/missile/bomb takts + trails/tracers/meshes) |
| matrix_game/effects/smoke_and_fire.rs   | MatrixEffectSmokeAndFire.cpp (smoke + fire puff emitters) |
| matrix_game/effects/weapon.rs           | MatrixEffectWeapon.{cpp,hpp} (EWeapon + CMatrixEffectWeapon + WeaponHit + CLaser/CVolcano/repair-beam visuals) |

Not yet ported: MatrixEffect (base class itself — its list/limits are
the GameEffect enum + effects_takt), Dust, ElevatorField, Path,
Zahvat. (Shleif's AddSmoke/AddFire map to standalone Smoke/Fire
spawns; the repair beam visuals live on the weapon.)

## matrix_game/interface/ (MatrixGame/src/Interface/)

| Rust file                               | Original                                           |
|-----------------------------------------|----------------------------------------------------|
| matrix_game/interface/builder_preview.rs| CConstructor::Render (CConstructor.cpp:264-360) — preview viewport + directional light slice |
| matrix_game/interface/constructor.rs    | CConstructor.{cpp,h} (+ folded CConstructorPanel)  |
| matrix_game/interface/counter.rs        | CCounter.{cpp,h}                                   |
| matrix_game/interface/history.rs        | CHistory.{cpp,h}                                   |
| matrix_game/interface/iface_element.rs  | CIFaceElement.{cpp,h} (folds CIFaceButton / CIFaceImage / CIFaceStatic) |
| matrix_game/interface/iface_list.rs     | CInterface.h::CIFaceList (panel container + events)|
| matrix_game/interface/iface_menu.rs     | CIFaceMenu.{cpp,h}                                 |
| matrix_game/interface/interface.rs      | CInterface.{cpp,h} (panel struct + `if/<Name>` loader) |
| matrix_game/interface/renderer.rs       | (no C++ peer — wgpu 2D quad pipeline for the HUD)  |
| matrix_game/interface/sound.rs          | (no direct peer — dispatch wrapper for CSound::Play UI calls; backend deferred) |
| matrix_game/interface/turret_build.rs   | CInterface::BeginBuildTurret slice (CInterface.cpp:4650+ + placement state) |

Not yet ported: CAnimation → animation.rs, MatrixHint → hint.rs.

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
| shaders/billboard.wgsl               | matrix_game/effects/effects_renderer.rs |
| shaders/effect_mesh.wgsl             | matrix_game/effects/effects_renderer.rs |
