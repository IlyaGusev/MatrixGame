# Rust Port Bug Audit — 2026-06-09

Full comparison of `rust_port/` (HEAD `17f1a7f`, ~47.4k lines, builds clean with
`cargo build`) against the original C++ in `MatrixGame/src` and `MatrixLib/`.
Scope: behavioral bugs in code that IS ported. Features documented as deferred
in `rust_port/src/CROSSREF.md` / `DIFF.md` are excluded.

Severity: **H** = gameplay/visual breaker, **M** = clearly wrong behavior,
**L** = fidelity deviation visible in edge cases.

> **Fix status (2026-06-10):** all 7 high and all 33 medium bugs below are
> FIXED in the working tree (uncommitted): R1-R6, M1-M3, W1-W3, C1-C5,
> O1-O5, I1-I11, L1-L3, A1-A3, P1, plus the Phase 0 `float2int`/`trunc_float`
> helpers in `common.rs`. Verified: `cargo build`, `cargo test --lib`
> (156 passed), `cargo check --lib --target wasm32-unknown-unknown`.
> Low-severity items (Phase 4/5) remain open. Note: `examples/check_shore.rs`
> was already broken at HEAD (references removed `vector_object::resolve_paths`)
> and is untouched.

---

## 1. Robots / AI / pathfinding

| # | Sev | Where | Bug |
|---|-----|-------|-----|
| R1 | **H** | `robot.rs:505,956,963-978,687-711,636-646,892`, `logic.rs:643` | Hovercraft/AntiGravity chassis index swap: `chassis as usize` (AntiGravity=3, Hovercraft=4) used as config/nsh index, but C++ `m_Unit[0].m_Kind-1` maps Hovercraft→3, AntiGravity→4 (`MatrixConfig.hpp:42-43`, `MatrixLogic.cpp:513-523`, `MatrixRobot.cpp:4159-4191`). Both chassis get each other's speeds and passability masks. Use the RUK mapping `object_robot.rs::chassis_kind_index` already implements. |
| R2 | M | `map.rs:791-795` (via `robot.rs:671,635`, `logic.rs:594`) | `world_to_move` floors; C++ `Float2Int` rounds to nearest (`Math3D.hpp:307`). Off-by-one in path start cells, get-lost destinations, order placement. |
| R3 | M | `logic.rs:1913-1931` | `optimize_path` ignores blocker weights; C++ `CanOptimize` rejects shortcuts through weight≥40 cells (`MatrixLogic.cpp:1893-1916`) — paths straighten through other robots' claimed destinations. |
| R4 | M | `robot.rs:1027-1064` | `robot_to_object_collision` skips cannons; C++ also collides vs `OBJECT_TYPE_CANNON`, radius 18+20, unhalved push-out (`MatrixRobot.cpp:3036-3066`). Robots drive through turrets. |
| R5 | M | `robot.rs:725-758` | Pathfinding blockers omit cannons; C++ `ZoneMoveCalc` adds each live cannon as weight-200 blocker (`MatrixRobot.cpp:1646-1652`). |
| R6 | M | `object_robot.rs:2094-2111, 2053-2058` | `do_chassis_animation` returns for Hover/AntiGravity in MOVE/ROTATE branches; C++ runs all chassis with default k=1 and ROTATE always ends `SwitchAnimation(STAY)` (`MatrixObjectRobot.cpp:802-880`). |
| R7 | L/M | `robot.rs:636-662` | `get_lost` spiral clipped to radius 4 and suppresses the order on failure; C++ searches map-wide and always issues MoveTo (`MatrixRobot.cpp:5228-5234`). Crowded spawns never disperse. |
| R8 | L/M | `robot.rs:686-718` | `dispatch_move_to` spiral-snaps start/goal and `stop_moving()` on failure; C++ paths to the ordered dest directly, uses `PlaceGet` +1/+1 nudge for start, and `FindLocalPath` tolerates blocked-start neighbors (`MatrixRobot.cpp:1036-1110`, `MatrixLogic.cpp:484-493,1352`). Robots in blocked regions permanently ignore orders. |
| R9 | L | `robot.rs:599-606` | `rotate_hull` missing 3× speed cap branch for turns ≥160° (`MatrixRobot.cpp:2283-2287`). |
| R10 | L | `robot.rs:1118-1120` | `rnd_float01` = `(rnd & 0x7fff)/32768`; C++ `RndFloat` = `Rnd()/2147483645.0`. Breaks RNG bit-parity. Use `rng.float01()`. |
| R11 | L | `robot.rs:401-405` | `switch_animation(Rotate)` doesn't set the Rotate anim cursor nor the `m_Speed != 0` early-out (`MatrixObjectRobot.cpp:1375,1494-1500`). Latent until Rotate is wired. |
| R12 | L | `robot.rs:1624-1655` | InSpawn missing the 10 s `m_TimeWithBase` watchdog that closes the base and kills the robot (`MatrixRobot.cpp:766-776`). |
| R13 | L | `object_robot.rs:2059-2067` | ROTATE anim k from config rotation speed; C++ `m_RotSpeed` is always 0 → k always clamps to 3. Hardcode k=3. |

## 2. Map core

| # | Sev | Where | Bug |
|---|-----|-------|-----|
| M1 | **H** | `map.rs:437-703` | Bridges pass entirely missing: `bridges/Data` never read, so no `SetBridge`, no z rewrite, no plane-coeff rebuild (`MatrixMapPrepare.cpp:1448-1532`). Bridge decks render sunken; all `CELLFLAG_BRIDGE` branches dead. |
| M2 | M | `map.rs:1700-1721` | Orphaned prop=1 cannon record (no parent building) spawns as live cannon; C++ converts prop=1→2 and skips (`MatrixMapPrepare.cpp:842-873`). |
| M3 | M | `map.rs:1687-1688` | Turret-slot move-cell coords use `.floor()`; C++ `Float2Int` rounds (slot-snap parity). |
| M4 | L | `map.rs:880-882` | `get_color_with_lighting` exempts bridge cells from the water early-out; C++ returns ambient for any water cell (`MatrixMap.cpp:382-383`). |
| M5 | L | `map.rs:1129-1138` | `compute_normals` cnt==0 fallback is (0,0,1); C++ copies up-neighbor's or left-neighbor's normal first (`MatrixMapPrepare.cpp:95-99`). |
| M6 | L | `map.rs:1227-1247` | FLAT flag recomputed from corner-z equality; C++ takes it verbatim from compiled flags (`MatrixMapPrepare.cpp:1284`). |
| M7 | L | `map.rs:834-836,872-874,937-938` | Cell index `.floor()` vs C++ `TruncFloat` (toward zero) — differs for coords in (-GLOBAL_SCALE, 0). |
| M8 | L | `map.rs:2033-2046` | `find_property_int` parses i32; C++ `GetDword` — color properties ≥ 0x80000000 silently fall back to defaults. Parse as u32/i64. |
| M9 | L | `map.rs:1368-1408` | `DisableInshore` property ignored (`MatrixMapPrepare.cpp:1379-1387`). |
| M10 | L | `map.rs:1552-1571,1781-1800` | Local `get_z` in `load_buildings`/`load_robots` omits the water→-1000 branch; buildings also lose the negative-z shoreline handling (`MatrixMap.cpp:619-622`, `MatrixObjectBuilding.cpp:1029-1036`). |
| M11 | L | `map.rs:3745` | `ROBOT_WEAPONS_PER_ROBOT_CNT = 16`; C++ = 10 (`MatrixMap.hpp:24`). |
| M12 | L | `map.rs:3312-3321` | No-skybox sky gradient: port draws alpha-blended transparent→sky; C++ draws opaque black→skycolor with blending off (`MatrixMap.cpp:2140-2177`). (Found independently by two agents.) |

## 3. Water / terrain surfaces / sky

| # | Sev | Where | Bug |
|---|-----|-------|-----|
| W1 | **H** | `water.rs:794-804` | Wave takt advances `angle += steps`; C++ `FillVB(k)` with k starting at 1 advances N+1 — port animates water at half speed (`MatrixWater.cpp:294-308`). |
| W2 | M | `shaders/water.wgsl:75-77` | Water normal is normalized before lighting; original FFP leaves normals unnormalized and the 12.5× world-scale inverse-transpose shrinks them ~12×, making directional light ~12× weaker. Drop `normalize`, scale by `water_normal_len / water_scale`. |
| W3 | M | `map.rs:2372-2378` | Gloss reflection texture hardcoded to `Matrix/Textures/reflection`; C++ reads per-sky `Reflection` param (mars sky uses `reflection_red`) (`MatrixMapPrepare.cpp:1137`). |
| W4 | L | `shaders/water.wgsl:82` | Mirror UV remapped `*0.5+0.5` to texture center; original passes raw camera-space normal xy with WRAP (window around texture corner). |
| W5 | L | `ter_surface.rs:236-241` + `shaders/terrain.wgsl:45-50` | Surface `m_Color` applied after macro blend and its alpha ignored; C++ `TerSurfM` modulates tex×TFACTOR (rgb+a) *before* BLENDTEXTUREALPHA (`MatrixRenderPipeline.cpp:1799-1808`). |
| W6 | L | `ter_surface.rs:64-68`, `map.rs:2955-2958` | Surfaces with empty `groups` drawn unconditionally; C++ never draws unregistered surfaces. |

## 4. Camera / game loop / input

| # | Sev | Where | Bug |
|---|-----|-------|-----|
| C1 | **H** | `camera.rs:177-179,262` | `zoom()` steps by raw `CamMouseWheelStep` (0.05 in shipped robots.dat); C++ multiplies by 4.5 (`MatrixCamera.hpp:277,285`) — zoom 4.5× too slow in every configured run. |
| C2 | M | `camera.rs:465-472,623-632` | Ground clamp applied only in `eye_pos()`, not `view_matrix()`; C++ clamps the view matrix itself (`MatrixCamera.cpp:748-755`). Render camera clips into terrain; frustum dirs inconsistent when clamp engages. |
| C3 | M | `form_game.rs:501-525` | Right mouse button rotates the camera (treated like middle); C++ rotates on MMB only, RMB is orders only (`MatrixFormGame.cpp:631-642,758-760`). |
| C4 | M | `form_game.rs:524-545` | RMB world orders not blocked by UI hover; C++ gates on `m_InFocus == UNKNOWN` (`MatrixFormGame.cpp:756-760`). Right-click on HUD issues move orders. |
| C5 | M | `form_game.rs:817-839` | Frame step uncapped; original clamps each takt to 100 ms + smooths (`3g.cpp:425-433`). Tab-switch on WASM fast-forwards logic by thousands of takts. |
| C6 | L | `form_game.rs:137-139,305-307` | Missing RS_CAMPOS else-branch: CamPos-less maps should place camera at player base (offset −100·sin/+100·cos) and auto-select it (`MatrixMapPrepare.cpp:940-982`). |
| C7 | L | `form_game.rs:680-757` | During mouse-cam rotate, minimap drag / marquee / UI hover still run; C++ early-returns after `RotateByMouse` (`MatrixFormGame.cpp:530-551`). |
| C8 | L | `form_game.rs:775-814`, `camera.rs:293-321` | Key/pan/shift state never cleared on focus loss — alt-tab with key held pans forever. Clear on `Focused(false)`/`CursorLeft`. |
| C9 | L | `camera.rs:567-621` | `frustum_bounds_on_plane_zup` omits horizon-case ±one-group top-edge widening (`MatrixVisiCalc.cpp:586-590`). |

## 5. Map objects / buildings / cannons

| # | Sev | Where | Bug |
|---|-----|-------|-----|
| O1 | **H** | `object.rs:372` | "AnimP" detection checks char index 4; C++ checks index 5 (0-based) (`MatrixObject.cpp:1062`) — Portret rows misparsed as BEHF_ANIM; the unit test enshrines the inversion. |
| O2 | **H** | `object_building.rs:258-262` | `BuildStack::tick_timer` requires `parent_state == Closed` before producing anything; C++ gates only robots on BASE_CLOSED — and non-base factories never reach Closed in the port, so queued turrets never finish (`MatrixObjectBuilding.cpp:1690,1776`). |
| O3 | M | `object.rs:504,2291` | Object rotation composed `rx*ry*rz`; C++ row-major `mx*my*mz` ≡ glam `rz*ry*rx`. Multi-angle decorations oriented wrong. |
| O4 | M | `object.rs:698` | BEHF_ANIM death sets HP=2e9 (invincible); C++ transitions to the `#`-table's next state via `ApplyAnimState`, reseeding HP (`MatrixObject.cpp:269-286`). |
| O5 | M | `object_building.rs:357-386` | `pick_balanced_team` returns least-populated team; C++ unconditionally overwrites with team 0 (`CConstructor.cpp:121`). |
| O6 | L | `object_building.rs:719` | `base_floor_progress` seeded 0.0 instead of 0.2 (spawn close animation lost); duplicate dead `base_floor` field. |
| O7 | L | `object_building.rs:954-974` | `build_stack.tick_timer` runs during DIP/neutral; C++ only ticks when alive and sided (`MatrixObjectBuilding.cpp:601-605`). |
| O8 | L | `object_building.rs:1106` | `damage` triggers HP overlay every hit; C++ shows it on hover-trace only (`MatrixMap.cpp:1154`). |
| O9 | L | `object_cannon.rs:211-216,823-843` | Cannon z = caller's `pos_z`; C++ averages the 4 surrounding heightmap points + `m_AddH` (`MatrixObjectCannon.cpp:266-275`). |
| O10 | L | `object_cannon.rs:830-834` | Idle cannons drawn untinted; C++ uses terrain lighting color as TFACTOR (`MatrixObjectCannon.cpp:551`). |

## 6. Interface / constructor

| # | Sev | Where | Bug |
|---|-----|-------|-----|
| I1 | **H** | `interface/constructor.rs:523-532,597-603` | `SuperDjeans(WEAPON,…)` routes through first-empty-slot insertion; C++ unconditionally overwrites/clears the resolved physical pylon (`CConstructor.cpp:469-498`). Replacing/clearing a weapon on an occupied pylon doesn't update. |
| I2 | M | `interface/constructor.rs:939-952` | `<AddDamage>` periods hardcoded 500 ms; C++ ABLAZE=10, SHORTED=50 — tooltip aux damage 50×/10× too small. |
| I3 | M | `form_game.rs:1595-1602` | Pylon LMB eaten as no-op; C++ `RemoteOperateUnit` cycles component kind with wrap (`CInterface.cpp:262`, `CConstructor.cpp:379-438`). |
| I4 | M | `interface/constructor.rs:464-537`, `form_game.rs:1409,1618` | Build-multiplier counter (`m_RCountControl`) not reset on component change (`CConstructor.cpp:443,578`). |
| I5 | M | `interface/hint.rs:1302-1305` | Hint anchored at element center/bottom; C++ anchors top-left (`CIFaceButton.cpp:138-139`). All non-pinned hints shifted. |
| I6 | M | `interface/hint.rs:702-705` | `_MOD:` treated as sticky modifier; C++ emits an immediate zero-size element (forces line break / cursor move) (`MatrixHint.cpp:662-666`). |
| I7 | M | `interface/iface_list.rs:626-627` | Click on popup chrome (not a row) leaves popup open; C++ cancels + restores config (`CIFaceMenu.cpp:386-395`). |
| I8 | M | `interface/iface_list.rs:344-416` | Mouse-move while popup open still fires focus/`SetLabelsAndPrice`; C++ blocks all non-static OnMouseMove during POPUP_MENU_ACTIVE (`CInterface.cpp:979`). |
| I9 | M | `interface/iface_list.rs:392-398` | Focus transition overrides Disabled buttons to Focused for a frame; C++ only NORMAL→FOCUSED (`CIFaceButton.cpp:147-152`). |
| I10 | M | `interface/interface.rs:1877-1895`, `iface_element.rs:18-28` | Element `type` param unparsed; no `IFACE_PRESSED_UNFOCUSED` state — CHECK/CHECK_PUSH buttons can't latch, `sPressedUnFocused*` art never loaded, no `CheckGroupReset`. |
| I11 | M | `interface/interface.rs:1469-1635` | No resource-shortage feedback: missing DYNAMIC_WARNING icons + red price text when unaffordable (`CInterface.cpp:3284-3288,2108-2288`). |
| I12 | L | `interface/hint.rs:68` | `HINT_OTSTUP = 5`; C++ = 2. |
| I13 | L | `interface/hint.rs:1408-1426` | Overflowing hints shifted on-screen; C++ suppresses them entirely (`CInterface.cpp:4404-4433`). |
| I14 | L | `interface/iface_menu.rs:514-517` | Popup widths use Russian-build constants; this tree compiles `_ENGLISH_BUILD` (95/60/45/60). Confirm against shipped assets. |
| I15 | L | `interface/renderer.rs:1236-1238` | Popup cursik vertically centered; C++ places top edge at `y + hpos` (`CIFaceMenu.cpp:352-353`). |
| I16 | L | `interface/counter.rs:144-148` | Disables both counter buttons when unaffordable; C++ (typo) leaves Down enabled. Decide + document. |
| I17 | L | `interface/constructor.rs:1610-1632` | Random-bot head roll `range(1,7).min(4)` skews kind 4 to 3/7; C++ `Rnd(1,7)` unclamped. |
| I18 | L | `interface/hint.rs:844-855` | Pass-2 height rounding to center-tile multiple not ported (`MatrixHint.cpp:308-313`). |
| I19 | L | `interface/renderer.rs:931-935` | State tints (×0.8 / ×0.5) applied on top of authored state art; C++ never tints. |
| I20 | L | `interface/interface.rs:1208-1212` | `warn` gated on robot-cap; in C++ `warn` is the per-resource warning template, `warn1`/`warnl` are the cap warnings. |
| I21 | L | `interface/interface.rs:1306-1308` | `refresh_base_visibility_v2` defaults unmatched elements VISIBLE; C++ IF_BASE hides all and shows explicitly (e.g. `res_unit` only when a component is focused). |
| I22 | L | `interface/constructor.rs:1137-1155` | `.min(len - 1)` underflows when config table empty → panic. Use `.get().unwrap_or(0)`. |
| I23 | L | `interface/iface_list.rs:310-327` vs `renderer.rs:894` | Hit-test prioritizes first panel; renderer draws last panel on top — overlapping panels' top one loses hit-test. |
| I24 | L | `interface/hint.rs:646-649` | `_IF:` truth test trims; C++ `IsEmpty()` doesn't — whitespace-only value differs. |
| I25 | L | `interface/sound.rs:55-61` | `contains("conf")` inverts the C++ `Find(L"conf")` truthiness quirk (≠0 includes −1). Decide + document. |
| I26 | L | `interface/constructor.rs:551-608` | `djeans007` (hover preview) writes the persisted config; C++ only mutates the live preview — dropping the popup without restore commits the hover. |
| I27 | L | `interface/iface_element.rs:186-203` | State label/image falls back to Normal; C++ renders the state's own (possibly empty) data. |
| I28 | L | `interface/renderer.rs:1561-1598` | Hint/popup text at native AFT size while chrome scales by `screen_h/768` — text overflows box at scale ≠ 1. |

## 7. MatrixLib (parsers, formats, textures)

| # | Sev | Where | Bug |
|---|-----|-------|-----|
| L1 | M | `matrix_lib/base/blockpar.rs:144-151` | `par_get_ne` stops at the first index hit; C++ skips same-named blocks and returns the first *par* (`CBlockPar.cpp:412-432`). |
| L2 | M | `matrix_lib/three_g/vector_object.rs:455-476` | SVOUnion `m_IBase` never read; C++ uses `StartIndex = -m_IBase` when negative (`VectorObject.cpp:1366-1371`) — wrong triangle ranges for optimized meshes. |
| L3 | M | `matrix_lib/base/pack.rs:166-178` | Type validation before the `m_Free` check — one free record with garbage type aborts the whole archive (`Pack.cpp:203-223`). |
| L4 | L | `blockpar.rs:62-64,145,156,168` | Lookups case-insensitive; C++ is case-sensitive. |
| L5 | L | `blockpar.rs:326-330,349-357` | Values trimmed both ends; C++ keeps leading spaces (affects `AlphaTest = 0` flag reset in `Texture.cpp:134`). |
| L6 | L | `blockpar.rs:434-449` | Content after inline `{}` / closing `}` on same line discarded; C++ continues parsing the line (`CBlockPar.cpp:1046-1068`). |
| L7 | L | `blockpar.rs:422-437` | `Name { Value }` parsed as closed one-line block; C++ only supports empty `{}` inline. |
| L8 | L | `blockpar.rs:199-200` | Non-BOM text decoded UTF-8; C++ uses CP_ACP (CP1251 for Russian data). |
| L9 | L | `wstr.rs:77-103` | `int_par`/`double_par` parse numeric prefix + exponents; C++ scrapes digits anywhere, no exponents (`CWStr.cpp:253-313`). (Shipped data unaffected per config audit.) |
| L10 | L | `storage.rs:111-118` | `find_as_wstr` case-insensitive; C++ exact memcmp. |
| L11 | L | `bitmap/mod.rs:48-62` | `merge_by_mask` forces alpha 255; C++ processes all BytePP channels. |
| L12 | L | `bitmap/mod.rs:84-95` | `merge_with_alpha` truncates (C++ rounds) and forces alpha 255 (C++ `oalpha + (255-oalpha)*A`). |
| L13 | L | `three_g/texture.rs:279-288` | DXT1 3-color code 3 decoded as avg color; D3DX emits `[0,0,0,0]`. |
| L14 | L | `three_g/texture.rs:386-425` | Mips alpha-weighted; D3DX_FILTER_BOX averages channels independently. |

## 8. Aux rendering (minimap, shadows, effects)

| # | Sev | Where | Bug |
|---|-----|-------|-----|
| A1 | M | `minimap.rs:1402-1404` | Flash cycle uses TAU/100 ms; C++ `x*π` → 200 ms. Halve the frequency. |
| A2 | M | `multi_selection.rs:404-456` | Marquee select unlimited; C++ caps at 9 and collapses <9 px² drags to single-select (`MatrixMultiSelection.cpp:211,257,270`). |
| A3 | M | `slot_marker.rs:331` | `render()` overwrites uniform with green `[0.4,1,0.4,0.7]` each frame; C++ SPOT_TURRET is `0x80FFFFFF` translucent white (already correctly init'd at :167). |
| A4 | L | `shadow.rs:506` | Ground-shadow quads split along i01–i10 diagonal; C++ strip splits along i00–i11 (point_light.rs gets it right). |
| A5 | L | `minimap.rs:922-932` | Zoom buttons compound from `tgt_scale`; C++ from animated current scale. |
| A6 | L | `minimap.rs:1270-1293` | Camera frustum projected onto z=WATER_LEVEL with disp+clamp; C++ projects onto z=0, no disp, viewport clips (`MatrixMinimap.cpp:794-841`). |
| A7 | L | `effects/selection.rs:77` | Dot count truncates; C++ `Float2Int` rounds. |
| A8 | L | `effects/selection.rs:521` | Alpha-0 override to 0.9 breaks the 200 ms fade-in (first-frame black dots). Remove. |
| A9 | L | `pause_overlay.rs:19` | Dim alpha 0.4 vs C++ 0x14/255 ≈ 0.078 — intentional crank, needs sign-off or revert. |
| A10 | L | `minimap.rs:1192-1215` | Under-attack marker flash (radius×2, white, 128 ms gate) not ported and not DIFF-listed (`MatrixMinimap.cpp:692-737`). |
| A11 | L | `shadow.rs:441` | `tu == 1.0` counts inside; C++ `tu >= 1.0` outside. |

## 9. Platform / config

| # | Sev | Where | Bug |
|---|-----|-------|-----|
| P1 | M | `platform/mod.rs:8-14` | Native `now_secs()` uses wall-clock `SystemTime` — NTP step ⇒ negative or giant takt. Use `Instant` against process-start epoch (WASM path already monotonic). |

`common.rs`, `config.rs` (damage tables, enums, key strings — cross-checked
against shipped `robots.dat`), `side.rs`, `gfx/` were audited and came back
clean. `render_pipeline.rs` holds only an unused vertex layout.

Also noted: the "Rust Port File Structure" section in `CLAUDE.md` is stale —
it describes the pre-refactor `src/game`/`src/renderer` layout; the live map
is `rust_port/src/CROSSREF.md`.

---

# Step-by-step fix plan

Each step ends with a verification gate. No automated test suite exists, so
gates are `cargo build` + targeted visual/behavioral checks against the
original-game reference, plus unit tests where the logic is pure (parsers,
math) — add tests when fixing those.

## Phase 0 — shared parity helper (unblocks several fixes)

1. Add `float2int(f: f32) -> i32` (round-half-to-even is x87 `fistp` default;
   in practice `.round()` matches for game data) and `trunc_float` helpers in
   `common.rs`, with unit tests vs known C++ values.
   → verify: `cargo test`.
2. Apply at all sites: R2 (`world_to_move`), M3 (turret slots), M7
   (`get_z`/`get_color`/`get_normal` cell index → truncate), A7 (selection dot
   count), L12 (`merge_with_alpha` rounding).
   → verify: build + spot-check robot path starts and turret slot snapping.

## Phase 1 — high-severity gameplay breakers

3. **R1** chassis index swap: route every `chassis as usize` config/nsh lookup
   through the RUK mapping (reuse `object_robot.rs::chassis_kind_index`).
   → verify: Hovercraft crosses water per CHASSIS4 chars; AntiGravity uses
   CHASSIS5 speeds (compare `chassis_max_speed` values against robots.dat).
4. **O2** turret production: gate `tick_timer`'s early return on
   `PendingKind::Robot` only.
   → verify: queue a turret on a non-base factory; it completes and goes Idle.
5. **O1** AnimP parse: check char index 5; fix the wrong test
   `anim_vs_animp_split_on_fifth_char` to assert the C++ behavior.
   → verify: `cargo test`; Portret map objects animate per-state.
6. **C1** zoom step ×4.5 (keep raw config value stored).
   → verify: wheel zoom traverses the [0.25,4.0] range in the same number of
   notches as the original (~17 with step 0.225).
7. **W1** water angle advance `steps + 1` when `steps > 0`.
   → verify: wave period visually matches original footage (half-speed bug
   disappears).
8. **I1** SuperDjeans weapon path: resolve physical pylon from the weapon
   matrix and overwrite/clear `weapon[fis_pilon-1]` directly.
   → verify: in constructor, replacing and clearing a weapon on an occupied
   pylon updates price/damage/name/preview.
9. **M1** bridges pass: port the `bridges/Data` loop (SetBridge, z rewrite,
   plane coefficient rebuild) after `compute_units`.
   → verify: load a bridge map; deck renders at deck height, units on bridge
   report bridge z in `get_z`.

## Phase 2 — medium gameplay/logic correctness

10. **R4 + R5** cannons in collision and pathfinding blockers.
    → verify: a robot ordered through a turret detours around it.
11. **R3** `optimize_path` rejects shortcuts through weight≥40 cells (thread
    the blocker grid in).
    → verify: two robots ordered to adjacent cells don't straighten through
    each other's destinations.
12. **R6** chassis animation: k=1 fallback for Hover/AntiGravity; ROTATE
    always exits to Stay. **R13** hardcode rotate k=3.
    → verify: moving hovercraft animates; rotation anim returns to stay.
13. **R7 + R8** get_lost map-wide spiral + always issue order;
    dispatch_move_to keeps ordered dest, PlaceGet nudge, blocked-start
    tolerance instead of `stop_moving()`.
    → verify: spawn a crowd — robots disperse; robot adjacent to blocked
    region accepts move orders.
14. **O3** rotation order `rz*ry*rx`; **O4** ANIM death-state transition via
    `apply_anim_state`; **O5** team always 0.
    → verify: tilted decorations match original screenshots; multi-stage
    destructible objects take staged damage.
15. **C2–C5** camera/input: clamp inside `view_matrix()`; MMB-only rotate;
    RMB blocked over UI; cap takt at 100 ms.
    → verify: low-pitch camera doesn't clip terrain; RMB on HUD does nothing;
    tab-away/back on WASM doesn't fast-forward.
16. **P1** monotonic native clock.
    → verify: build + run native.
17. **L1–L3** parser fixes (par_get_ne skip-blocks, vo `m_IBase`, pack free-rec
    order) with unit tests for each.
    → verify: `cargo test`; all shipped maps/models still load.

## Phase 3 — medium UI correctness

18. **I2** AddDamage periods 10/50; **I4** reset build counter on component
    change; **I3** port `RemoteOperateUnit` LMB cycling.
    → verify: tooltip damage numbers match original; counter resets to 1 on
    any change; pylon LMB cycles kinds with wrap.
19. **I7 + I8** popup modality (chrome-click cancels+restores; suppress
    focus/hover while open); **I26** keep hover preview out of persisted
    config.
    → verify: open popup, hover rows, click chrome — config restored exactly.
20. **I5 + I6** hint anchor top-left and `_MOD:` immediate element; **I12**
    OTSTUP=2; **I13** suppress overflowing hints.
    → verify: hint positions/layouts match original screenshots.
21. **I9 + I10** element state machine: parse `type`, add PRESSED_UNFOCUSED +
    latch transitions + `CheckGroupReset`; no Disabled→Focused flicker.
    → verify: auto-order toggles latch; preset buttons group-unlatch.
22. **I11** resource shortage warnings + red price text.
    → verify: with empty resources, prices render red and warn icons show.
23. **A1–A3** minimap flash period π, marquee cap 9 + tiny-drag collapse,
    slot-marker white.
    → verify: visual check vs original.

## Phase 4 — low-severity fidelity batch

24. Map: M2, M4–M11 (skip orphan cannons, water color/z branches, normals
    fallback, FLAT from flags, GetDword props, DisableInshore, weapons-cnt
    10).
25. Water/sky: W2–W6, M12 (water lighting magnitude, per-sky reflection,
    mirror UV, surface color order+alpha, empty-groups skip, opaque no-skybox
    gradient).
26. Camera/input: C6–C9 (base camera placement+auto-select, early-return
    during mouse-cam, clear keys on focus loss, horizon widening).
27. Robots: R9–R12 (160° hull cap, RndFloat parity, Rotate anim arm, 10 s
    base watchdog).
28. Buildings/cannons: O6–O10 (base_floor 0.2, tick gating, hover-only HP
    bar, cannon terrain z + idle tint).
29. UI leftovers: I14–I25, I27, I28 (decide+document the two intentional
    divergences I16/I25; fix the I22 panic with `.get()`).
30. Lib/aux: L4–L14, A4–A8, A10, A11; revisit A9 (pause alpha) with the user.
    → verify after each sub-batch: `cargo build`, `cargo test`, visual pass
    on a reference map (native + `wasm-pack build --dev`).

## Phase 5 — closeout

31. Update `CLAUDE.md`'s stale file-structure section to point at
    `CROSSREF.md`.
32. Append the deliberate-divergence decisions (I14, I16, I25, A9) to
    `DIFF.md` so future audits don't re-flag them.
33. Full visual comparison run vs original screenshots (terrain, water,
    bridges, constructor, hints, minimap) on native and WASM.
