# DIFF — C++ Original vs Rust Port (UI / Interface focus)

Scope: everything under `MatrixGame/src/Interface/` plus the UI-adjacent
top-level files (`MatrixCursor`, `MatrixMinimap`, `MatrixDebugInfo`,
`MatrixLoadProgress`, `MatrixProgressBar`, `MatrixTransition`,
`MatrixMultiSelection`, `DevConsole`) vs their Rust counterparts in
`rust_port/src/matrix_game/interface/` and `rust_port/src/matrix_game/`.

Cross-reference baseline: `rust_port/src/CROSSREF.md`.

Statuses:
- **FULL** — behavior mirrors C++ within acceptable tolerance.
- **PARTIAL** — core data/logic present, pieces of behavior missing.
- **STUB** — file exists but functionality largely deferred.
- **MISSING** — no Rust analogue.
- **CUSTOM** — Rust-only (no C++ peer by design, e.g. wgpu glue).

---

## 1. File-structure diff

### C++ files with no Rust analogue

| C++ file | LOC | Current Rust home | Status |
|----------|----:|-------------------|--------|
| `Interface/CAnimation.cpp` / `.h` | 150 | — | MISSING |
| `Interface/MatrixHint.cpp` / `.hpp` | 1033 | — | MISSING |
| `MatrixCursor.cpp` / `.hpp` | ~320 | — | MISSING |
| `MatrixDebugInfo.cpp` / `.hpp` | ~200 | — | MISSING |
| `MatrixLoadProgress.cpp` / `.hpp` | ~70 | — | MISSING |
| `MatrixTransition.cpp` / `.hpp` | ~250 | — | MISSING |
| `MatrixMultiSelection.cpp` / `.hpp` | ~360 | (logic partially in `form_game.rs`, no renderer) | MISSING (visual) |
| `DevConsole.cpp` / `.hpp` | ~420 | — | MISSING |

### Rust files with no C++ peer (intentional glue)

| Rust file | Purpose | Notes |
|-----------|---------|-------|
| `interface/renderer.rs` (1473 LOC) | wgpu 2D HUD quad pipeline | Replaces D3D9 fixed-function UI blending |
| `interface/text.rs` (627 LOC) | Glyph / label rasterization | Replaces `ID3DXFont` usage |
| `interface/sound.rs` (61 LOC) | UI sound dispatch wrapper | Backend deferred (see §18) |
| `interface/iface_list.rs` | Hoists `CInterface.h::CIFaceList` panel container | OK by design |
| `interface/builder_preview.rs` | Carved from `CConstructor::Render` preview slice | Stub (see §10) |
| `interface/turret_build.rs` | Carved from `CInterface::BeginBuildTurret` | Partial (see §11) |

### Action points (file-structure)

- **A-FS-1** Create `interface/animation.rs` (port of `CAnimation.{cpp,h}`) and wire
  it into `iface_element.rs` as the element's optional animation source.
- **A-FS-2** Create `interface/hint.rs` (port of `MatrixHint.{cpp,hpp}`) and add a
  hint template field + hover timer to `iface_element.rs`.
- **A-FS-3** Create `matrix_game/cursor.rs` (port of `MatrixCursor`) or formally
  document the decision to rely on the browser/native cursor.
- **A-FS-4** Create `matrix_game/debug_info.rs` (port of `MatrixDebugInfo`) —
  at minimum FPS + object count overlay.
- **A-FS-5** Create `matrix_game/load_progress.rs` or replace with a WASM-level
  asset-streaming progress UI.
- **A-FS-6** Create `matrix_game/transition.rs` (fade in/out) if campaign flow
  is in scope; otherwise document as skipped.
- **A-FS-7** Add a visual renderer to the existing drag-select code (new
  `matrix_game/multi_selection.rs`), since `MatrixMultiSelection` rendering is
  absent even though some selection logic lives in `form_game.rs`.
- **A-FS-8** Create `matrix_game/dev_console.rs` or replace with a native-only
  egui/debug shell; decide once and document.
- **A-FS-9** Update `CROSSREF.md` as each of A-FS-1..A-FS-8 lands so the index
  stays authoritative.

---

## 2. `CIFaceElement` (base UI element) — PARTIAL

- C++ `Interface/CIFaceElement.{cpp,h}` (~560 LOC)
- Rust `interface/iface_element.rs` (218 LOC)

Rust folds `CIFaceButton` / `CIFaceImage` / `CIFaceStatic` into an
`ElementKind` enum instead of OOP inheritance. The *data* is roughly
equivalent; the *behavior* is what's missing.

Gaps:
- `CAnimation* m_Animation` — no per-element animation pointer.
- `SElementHint m_Hint` — no hover tooltip template + timer.
- `SAction m_Actions[MAX_ACTIONS]` — Rust dispatches by button name
  rather than an action array; OK, but action **params** (string/int
  payloads attached per state) are lost.
- `m_VisibleAlpha`, `ElementAlpha`, per-pixel alpha hit test — Rust
  uses plain AABB hit tests; pixels that are transparent in the atlas
  are still clickable.
- `SetClearRect()` / `HasClearRect()` — no Z-buffer clear optimization
  when rendering the HUD over the world.
- `ElementGeomInit(..., bool full_size)` + `RecalcPos` path — Rust
  re-builds geometry in the renderer every frame; acceptable, but any
  C++ behavior that relied on stable per-element quads (e.g.,
  animation-frame reposition) needs re-thinking.
- `BeforeRender()` — no per-element pre-render hook.
- State-change sounds — C++ plays focus/press sounds on state
  transition; Rust has no wiring (see §18 sound).

Action points:
- **A-EL-1** Add `animation: Option<Animation>` field after A-FS-1.
- **A-EL-2** Add `hint: Option<HintTemplate>` + hover timer after A-FS-2.
- **A-EL-3** Replace name-based dispatch with a first-class action
  table so per-button state payloads (e.g., which `djeans_id` a pylon
  button argues to `OperateUnit`) survive.
- **A-EL-4** Implement per-pixel alpha hit-test for buttons that
  overlap atlas regions with holes (matches `CIFaceButton::Hit`).
- **A-EL-5** Hook state-change sounds through `interface/sound.rs`.

---

## 3. `CIFaceButton` — PARTIAL (A-BT-1 / A-BT-3 landed)

- C++ `Interface/CIFaceButton.{cpp,h}` (~340 LOC)
- Rust folded into `iface_element.rs::ElementKind::Button`

Gaps:
- **RMB pylon popup** — `CIFaceButton::OnMouseRBDown` at
  `CIFaceButton.cpp:188-312` walks the constructor's current config
  and builds a `CIFaceMenu` with the alternatives. Rust opens a popup
  only via `iface_list.rs` (see §5) and misses the hull-vs-weapon
  branching logic at `CIFaceButton.cpp:283-300` / `:301-312`.
- **Hint show/hide on hover** — entire block at
  `CIFaceButton.cpp:109-180` has no Rust equivalent.
- **Animated buttons** — no `m_Animation` render path.
- **`SuperDjeans` binding** at RMB time — C++ wires the menu callback
  to `CConstructor::SuperDjeans`; Rust's popup path calls a different
  entry (`IFaceList::popup_restore_pending`), which means previews on
  hover aren't restored the same way on cancel.

Action points:
- **A-BT-1** ✅ Done — `popup_for_pylon` in `iface_menu.rs` covers
  `pich` / `pihu` / `pihe` / `pi1..pi5`; RMB dispatch lives in
  `form_game::dispatch_ui_right_click`.
- **A-BT-2** ✅ Done — hover hint timer + show/hide lives on
  `IFaceList::hint_system`; timer resets on focus change and clears on
  popup open. See `iface_list::refresh_hint_hover`.
- **A-BT-3** ✅ Done — `CIFaceMenu::saved_config` +
  `popup_restore_pending` + `preview_popup_hover` implement
  preview-on-hover / restore-on-cancel.

---

## 4. `CIFaceStatic` / `CIFaceImage` — PARTIAL

- C++ `Interface/CIFaceStatic.{cpp,h}` (~100 LOC), `Interface/CIFaceImage.{cpp,h}` (~50 LOC)
- Rust folded into `iface_element.rs`

Gaps:
- Static elements have the same missing **hint** path as buttons.
- `CIFaceImage` is a pure metadata holder in C++ that other elements
  copy UV rects from (used by `CConstructor` to build pylon icons).
  Rust inlines `StateImage` per element; missing the shared registry
  means runtime cloning paths (`CInterface::CreateStaticFromImage`)
  have to re-resolve images by name every time.

Action points:
- **A-ST-1** Share A-FS-2 hint plumbing with statics.
- **A-ST-2** Add a small `image_library: HashMap<String, ImageRef>` to
  `CInterface` if we port `CreateStaticFromImage` (needed for a few
  dynamic element paths — see §6).

---

## 5. `CIFaceMenu` (RMB popup menu) — PARTIAL

- C++ `Interface/CIFaceMenu.{cpp,h}` (632 LOC)
- Rust `interface/iface_menu.rs` (431 LOC)

What's present: menu open/close, item hit-testing, item selection
callbacks, selector position tracking.

Gaps:
- **Item text labels** — `CreateMenu` takes `SMenuItemText* labels`
  (font + color per item) at `CIFaceMenu.cpp:62`. Rust renders only
  icon elements; menu item names from `LabelsText/<panel>/…` aren't
  drawn.
- **Chrome panel `IF_POPUP_MENU`** — `LoadMenuGraphics` loads a
  dedicated panel for border / cursor arrow / selector highlight
  (`m_MenuGraphics`, `m_CurMenuPos`, `m_Selector`). Rust inlines a
  template element and skips the decorative border and cursor arrow.
- **Preview-on-hover + restore-on-cancel** — `m_RobotConfig` saves
  the previous robot config when the menu opens; if the user cancels,
  C++ restores it. Rust has `popup_restore_pending` but the wiring is
  incomplete (confirmed by missing pieces in §3).

Action points:
- **A-MN-1** Integrate `text.rs` into menu rendering and parse
  `LabelsText/<panel>/<item_name>_<state>` from the interface config.
- **A-MN-2** Port `LoadMenuGraphics` and render the popup frame /
  selector highlight / cursor arrow elements.
- **A-MN-3** Finish save/restore of the constructor config around
  menu lifetime so hover previews match the C++ feel.

---

## 6. `CInterface` (panel container) — PARTIAL

- C++ `Interface/CInterface.{cpp,h}` (4734 LOC + 397 LOC)
- Rust `interface/interface.rs` (1177 LOC)

The static panel loader + basic visibility refresh is in Rust. The
*dynamic* UI construction layer is ~600 LOC of C++ that has no Rust
equivalent.

Major missing blocks (all in `CInterface.cpp`):

| Missing C++ method | Purpose | Effort |
|--------------------|---------|-------:|
| `CreateWeaponDynamicStatics` / `DeleteWeaponDynamicStatics` | Per-robot weapon icons on selection HUD | L |
| `CreateItemPrice` / `DeleteItemPrice` | Per-item price tag on pylon | M |
| `CreateSummPrice` / `DeleteSummPrice` | Total build cost label | M |
| `CreateGroupSelection` / `DeleteGroupSelection` | Unit-group selector widgets | M |
| `CreateGroupIcons` / `DeleteGroupIcons` | Team group markers | S |
| `CreatePersonal` / `DeletePersonal` | Team-color picker | M |
| `CreateStackIcon` / `DeleteStackIcon` / `MoveStackIcons` | Build-queue icons | M |
| `CreateOrdersGlow` | Order-preview highlight | S |
| `CreateElementRamka` | Highlight border drawing around selected elements (uses `CRITICAL_RAMKA` / `NORMAL_RAMKA` colors) | S |
| `CreateDynamicTurrets` / `DeleteDynamicTurrets` | Turret slot widgets in base panel | M |
| `CreateHintButton` / `HideHintButtons` / `DisableMainMenuButton` / `EnableMainMenuButton` / `PressHintButton` / `HintButtonId2Name` | Dialog button management (confirm / cancel / exit) | M |
| `AddHintReplacements` / `CheckShowHintLogic` | Hint text substitutions | M (depends on A-FS-2) |
| `SlideFocusedInterfaceLeft` / `SlideFocusedInterfaceRight` / `ReCalcElementsPos` / `BeginSlide` / `SlideStep` | Panel slide-in/out animation | M |
| `ConstructorButtonsInit` / `WeaponPilonsInit` | Pylon template setup | M |
| `LiveRobot` / `EnterRobot` | Arcade-mode transitions | L (out of scope?) |
| `JumpToBuilding` / `JumpToRobot` | Camera jumps on double-click | S |
| `BeginBuildTurret` | Turret placement mode (partially in `turret_build.rs` — see §11) | S |

Config-loading gaps (from `if/<Name>` parse):
- Animation frames block (`frames_cnt`, `period`, `frames/`) — ignored.
- Per-element `Hint` reference — ignored.
- Per-element `Actions` array with payload params — ignored.

Action points:
- **A-IF-1** Implement the dynamic-creation methods in the order listed
  (prices → stack icons → group/personal → orders glow → ramka →
  dynamic turrets → hint buttons → slide animations). Each is
  independent; bundle by HUD screen to keep PRs bounded.
- **A-IF-2** Extend the `if/<Name>` loader to parse animation, hint,
  and action blocks once A-FS-1 / A-FS-2 land.
- **A-IF-3** Decide whether `LiveRobot`/`EnterRobot` arcade transitions
  are in scope; if not, annotate explicitly in `CROSSREF.md`.

---

## 7. `CIFaceList` (global UI container) — PARTIAL

- C++ — spread across `CInterface.h` (class) and `CInterface.cpp`
  (methods, roughly `:2787+`).
- Rust `interface/iface_list.rs` (469 LOC).

Core routing (mouse move / LBDown / LBUp / RBDown / hit test) is
ported. Everything in §6's "Missing C++ method" table lives here in
C++; `CInterface` wraps per-panel versions while `CIFaceList` holds
the orchestration. Fix §6 and most of `CIFaceList` falls into place.

Additional gaps:
- `ShowInterface` / `BeforeRender` / `Render` / `LogicTakt` —
  orchestration is deferred to the game loop; OK, but the per-frame
  `LogicTakt` that drives slide animations and hint timers needs a
  concrete home (suggest: a single `update(dt)` on `IFaceList`).
- Rust tracks focused element by `(panel_idx, element_idx)`; C++ uses
  pointers. Fine, as long as panel/element deletion paths invalidate
  the tuple (double-check when A-IF-1 dynamic deletions land).

Action points:
- **A-LS-1** Add `IFaceList::update(dt_ms)` and route animation + hint
  timers + slide stepping through it.
- **A-LS-2** Audit focused-tuple invalidation once dynamic
  create/delete paths exist.

---

## 8. `CCounter` (build multiplier) — PARTIAL

- C++ `Interface/CCounter.{cpp,h}` (~140 LOC)
- Rust `interface/counter.rs` (292 LOC)

Gaps:
- `MulRes` / `DivRes` — C++ internal helpers that multiply the
  displayed per-item prices by the counter value. Rust returns the
  multiplier and expects the caller to apply it; behaviorally
  equivalent **if** §6's price-display porting respects it. Flag for
  integration testing once A-IF-1 prices land.
- Direct `SetState(IFACE_DISABLED)` on `m_ButtonUp` / `m_ButtonDown`
  is deferred to the visibility-refresh pass. That means a
  click-to-disable feels one frame late vs. C++. Minor but real.

Action points:
- **A-CT-1** When A-IF-1 price widgets ship, verify the counter's
  multiplier is applied identically to C++'s `MulRes`/`DivRes` path.
- **A-CT-2** (optional) Apply button enable/disable immediately on
  `Inc`/`Dec` rather than waiting for visibility refresh.

---

## 9. `CHistory` (config history) — FULL

- C++ `Interface/CHistory.{cpp,h}` (107 + 27 LOC)
- Rust `interface/history.rs` (150 LOC)

`AddConfig` / `PrevConfig` / `NextConfig` / `IsPrev` / `IsNext` all
match. No action items.

---

## 10. `CConstructor` (robot constructor) — PARTIAL

- C++ `Interface/CConstructor.{cpp,h}` (1600+ LOC)
- Rust `interface/constructor.rs` (1761 LOC) + `interface/builder_preview.rs` (95 LOC)

Ported: `OperateUnit`, `SuperDjeans`, `Djeans007`, `ProduceRobot` /
`StackRobot` (as `operate_current_construction`),
`GetConstructionPrice`, `CheckMaxUnits`, random / special bot
helpers, item-description text replacements.

Gaps:
- **3D preview viewport** — `CConstructor::Render` /
  `BeforeRender` (`CConstructor.cpp:264-360`) sets up a scissor
  viewport, a directional light, and renders the robot instance.
  `builder_preview.rs` has the viewport rect but not the robot
  rendering or light. Depends on the 3D pipeline being reachable
  from UI, which today it isn't.
- `SetSide` — team-color application to the preview robot: missing.
- `SetBase` — association with the parent production building:
  missing (needed so that the preview inherits the building's
  location for `JumpToBuilding`).
- `SetRenderProps` — viewport configuration: partial (rect only).
- `CConstructorPanel` nested class (labels / prices / button state) —
  folded into `RobotBuilder`, mostly present but integration with
  §6's `CreateItemPrice` / `CreateSummPrice` is pending.
- `RemoteOperateUnit` / `RemoteBuild` — `__stdcall` callback wrappers
  used by button dispatch. Rust routes through name dispatch;
  behavioral parity is OK unless a caller reflects on the C++
  callback identity (none found).

Action points:
- **A-CN-1** Complete `builder_preview.rs`: live robot instance +
  directional light + shadow + team coloring.
- **A-CN-2** Wire `SetBase` so `JumpToBuilding` (A-IF-1) can target
  the constructor's owning building.
- **A-CN-3** Once §6 prices land, drop the placeholder price math in
  `RobotBuilder` and route through the real price widgets.

---

## 11. Turret placement — PARTIAL

- C++ `CInterface.cpp:4650+` (`BeginBuildTurret`) + `m_IfListFlags`
  `PREORDER_BUILD_TURRET` bit + `m_BuildCa` pointer.
- Rust `interface/turret_build.rs` (86 LOC).

Ported: mode enter / cancel / is-active query.

Gaps:
- Live turret preview at cursor during placement.
- Slot validation against `building->m_TurretsMax` and CMAP cannon
  placement coordinates.
- Hover-slot tracking (`hovered_slot` field exists but isn't updated).
- Final placement callback that actually spawns the turret.

Action points:
- **A-TR-1** Render a preview mesh (or at minimum a footprint quad) at
  the hovered cannon slot.
- **A-TR-2** Compute hovered slot from the cursor ray vs building's
  cannon placements.
- **A-TR-3** Fire the turret-spawn callback on LMB commit, then
  `cancel()`.

---

## 12. `MatrixHint` (tooltip system) — PARTIAL (A-HN-1..4 landed, text-only)

- C++ `Interface/MatrixHint.{cpp,hpp}` (862 + 171 LOC).
- Rust — no file.

This is a heavy subsystem and touches every button + static:
`CIFaceButton::OnMouseMove`, `CIFaceStatic::OnMouseMove`,
`CInterface::AddHintReplacements`, `CInterface::CheckShowHintLogic`,
the global linked list of active hints, and sound-in/sound-out hooks.

Missing behaviors:
- `CMatrixHint::Build(template_name, replacements)` — assembles a hint
  bitmap from a template with `HEM_*` layout elements.
- `PreloadBitmaps` — caches hint bitmaps at load time.
- `DrawAll` / `DrawNow` / `ClearAll`.
- `SElementHint m_Hint` on `CIFaceElement` (template name + hover
  timer).
- Hover-to-show / leave-to-hide from button and static mouse handlers.
- `SoundIn` / `SoundOut` playback (depends on §18).

Action points:
- **A-HN-1** ✅ Done — `interface/hint.rs` has `TemplateLibrary`,
  `HintReplacer`, `HintSystem`, and `build_text` (pipe-parser supporting
  `_FONT:` / `_COLOR:` / `_TEXT:` / `_IF:` / `_ENDIF` / `[key]`
  substitution / `<br>` newlines).
- **A-HN-2** ✅ Done — `IFaceElement` gained `hint_template`,
  `hint_offset_x`, `hint_offset_y`; `IFaceList::update(dt_ms)` drives
  the hover timer and builds / tears down the active hint.
- **A-HN-3** ✅ Done — `TemplateLibrary::load` reads the
  `Templates` block; `HintReplacer::from_storage` reads `Replaces`;
  per-element `Hint` param is parsed in `load_element`.
- **A-HN-4** ✅ Partial — `form_game::refresh_hint_replacements`
  ports the resource-income + robot-count cases (`thz`, `enhz1/2`,
  `elhz`, `phz`, `rvhz`). The turret-button + call-from-hell cases are
  still deferred (depend on turret/maintenance plumbing).
- **Remaining gap** — the 9-slice bitmap chrome + per-element sound
  in/out is intentionally skipped (DIFF §18).

---

## 13. `CAnimation` (animated UI frames) — MISSING

- C++ `Interface/CAnimation.{cpp,h}` (100 + 50 LOC).
- Rust — no file.

Missing:
- `CAnimation::LogicTakt(ms)` — advance current frame.
- `LoadNextFrame(SFrame*)` — load frame into buffer.
- `GetCurrentFrame` — used by the element render path.
- `RecalcPos` — reposition frames on parent move.
- The `m_Animation` field on `CIFaceElement` and its render/hit paths.

Used by: buttons and labels whose `if/<Name>_Element` config has a
`frames_cnt` / `period` block (constructor pylons in particular).

Action points:
- **A-AN-1** Port to `interface/animation.rs` and integrate with
  `iface_element.rs` (field + render path).
- **A-AN-2** Extend the `if/<Name>` loader to parse animation frames.
- **A-AN-3** Drive `LogicTakt` from `IFaceList::update(dt_ms)`
  (A-LS-1).

---

## 14. `MatrixCursor` (custom cursor) — MISSING

- C++ `MatrixCursor.{cpp,hpp}` (~290 + 30 LOC).
- Rust — none.

Gaps: `Select(name)` / `Draw` / `Takt(ms)` / `SetPos` / `SetVisible`
/ frame UV calc.

Action points:
- **A-CU-1** Decide: port to Rust, or accept browser/native cursor
  and document the decision. If port: new `interface/cursor.rs` or
  `matrix_game/cursor.rs`; wire frame advance into
  `IFaceList::update(dt_ms)`.

---

## 15. `MatrixMinimap` — PARTIAL

- C++ `MatrixMinimap.{cpp,hpp}` (1389 LOC).
- Rust `matrix_game/minimap.rs`.

Ported (per CROSSREF): background bake, world↔map transforms with
pan/zoom, optional rotation, timed event overlays + off-screen arrows,
building markers, camera frustum projection.

Gaps:
- `DrawRadar` in-robot view (arcade mode) — not ported, probably
  out of scope.
- Disk-cached background PNG — irrelevant for WASM.

Action points:
- **A-MM-1** If arcade mode is ever in scope, revisit `DrawRadar`;
  otherwise mark skipped in `CROSSREF.md`.

---

## 16. `MatrixProgressBar` — PARTIAL

- C++ `MatrixProgressBar.{cpp,hpp}` (413 LOC).
- Rust `matrix_game/progress_bar.rs`.

Ported: 3-segment atlas bar, LIC color interpolation, position/size,
per-frame queuing.

Gaps:
- `CreateClone` / `KillClone` / `ClonePresent` / `DrawClones` —
  off-screen health indicator bars.

Action points:
- **A-PB-1** Port clone API and hook it into the off-screen indicator
  path (once §6 group icons / minimap off-screen arrows are aligned).

---

## 17. `MatrixMultiSelection` (drag-select) — MISSING (visual)

- C++ `MatrixMultiSelection.{cpp,hpp}` (~360 LOC).
- Rust — selection logic partially in `form_game.rs`, no visual rect.

Missing: `Begin` / `Update` / `End` callback flow,
`DrawAll`/`DrawPass1`/`DrawPass2`/`DrawPassEnd` passes, dip animation
(`MS_DIP_TIME`, `MS_FLAG_DIP`).

Action points:
- **A-MS-1** Add the drag-rect renderer (2D quad with border) in a
  new `matrix_game/multi_selection.rs`, hooked into the existing
  `form_game.rs` selection state.
- **A-MS-2** Port the dip animation timers.

---

## 18. UI sound dispatch — STUB

- C++ wired via `MatrixSoundManager.hpp` across buttons, statics,
  hints.
- Rust `interface/sound.rs` (61 LOC) — dispatch wrapper only.

Missing: every actual playback call (focus sound, press sound, hint
in/out, menu open/close).

Action points:
- **A-SD-1** Land a backend first (WebAudio / rodio). Separate
  prerequisite — track under audio subsystem, not UI.
- **A-SD-2** Once backend exists, attach playback to:
  state transitions (A-EL-5), hint in/out (A-HN-1), popup menu
  open/close (§5), counter inc/dec.

---

## 19. `MatrixDebugInfo` — MISSING

- C++ `MatrixDebugInfo.{cpp,hpp}` (~200 LOC). `T(key, value, ttl)`,
  `Draw` / `Takt`, `DI_*` flags for FPS / memory / visible objects /
  target coords / side info / sounds / frustum center.
- Rust — none.

Action points:
- **A-DB-1** Minimal debug overlay: FPS + visible object count +
  camera target coords. Gated by the existing `MATRIXGAME_CHEATS`
  equivalent (a Rust feature flag).

---

## 20. `MatrixLoadProgress` — MISSING

- C++ `MatrixLoadProgress.{cpp,hpp}` (~70 LOC). `SetCurLP`,
  `InitCurLP`, `SetCurLPPos`.
- Rust — none (WASM streams assets; native bundles them).

Action points:
- **A-LP-1** Decide: port the C++ progress UI, or ship a WASM
  asset-streaming progress bar driven by `fetch` progress. Document.

---

## 21. `MatrixTransition` — MISSING

- C++ `MatrixTransition.{cpp,hpp}` (~250 LOC). Fullscreen alpha-fade
  quad + `Takt(ms)` timing.
- Rust — none.

Action points:
- **A-TN-1** Only needed when campaign / map transitions ship. If in
  scope, port as `matrix_game/transition.rs` — it's straightforward
  (one fullscreen quad + timed alpha).

---

## 22. `DevConsole` — MISSING

- C++ `DevConsole.{cpp,hpp}` (~420 LOC). Command-input field,
  cursor blink, static command table, `SetActive`, `Keyboard(scan)`,
  `ShowHelp`.
- Rust — none.

Action points:
- **A-DC-1** Either port the full console or replace with a Rust
  debug shell (native-only; egui overlay). Pick one and document.

---

## Prioritized punch list

### Tier 1 — Missing / blocking core gameplay parity
1. **A-HN-1 .. A-HN-4** — Hint/tooltip system (`MatrixHint`). Every
   UI button expects a tooltip; its absence is user-visible.
2. **A-AN-1 .. A-AN-3** — `CAnimation` for animated UI elements
   (constructor pylons in particular).
3. **A-IF-1** — Dynamic UI element creation (weapon icons, price
   labels, group selection, stack icons, orders glow, ramka, dynamic
   turrets, hint buttons, slide animations). ~600 LOC of C++ with no
   Rust analogue; break down by HUD screen.
4. **A-MN-1 .. A-MN-3** — `CIFaceMenu` text labels + chrome frame +
   hover preview restore. User-visible in every RMB pylon menu.

### Tier 2 — User-visible polish
5. **A-BT-1 .. A-BT-3** — RMB button pylon popup branching + hover
   hint wiring.
6. **A-TR-1 .. A-TR-3** — Turret placement preview + slot tracking
   + spawn commit.
7. **A-CN-1 .. A-CN-3** — Constructor 3D preview (depends on 3D
   pipeline reachability from UI).
8. **A-PB-1** — Progress-bar clones (off-screen health indicators).
9. **A-MS-1 .. A-MS-2** — Drag-select visual rectangle + dip
   animation.

### Tier 3 — Debug / infra / optional
10. **A-FS-1 .. A-FS-9** — File-structure scaffolding + `CROSSREF.md`
    updates (many of these are prerequisites for Tier 1/2 items).
11. **A-LS-1 .. A-LS-2** — `IFaceList::update(dt_ms)` orchestration +
    focus-tuple invalidation audit.
12. **A-EL-1 .. A-EL-5** — Element-level hint / animation / action
    table / per-pixel hit-test / state-change sound hooks.
13. **A-ST-1 .. A-ST-2** — Shared image library for dynamic clones.
14. **A-CT-1 .. A-CT-2** — Counter / price integration parity.
15. **A-DB-1** — Debug overlay (FPS, object count).
16. **A-CU-1** — Custom cursor decision.
17. **A-LP-1** — Loading progress decision.
18. **A-TN-1** — Transition fade decision.
19. **A-DC-1** — Dev console decision.
20. **A-SD-1 .. A-SD-2** — UI sound backend (depends on audio
    subsystem landing separately).
21. **A-MM-1** — `DrawRadar` decision (arcade mode).
