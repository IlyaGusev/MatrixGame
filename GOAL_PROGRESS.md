# Goal: complete all rust-port mechanics

Consolidated work list from the 2026-06-12 six-subsystem audit (C++ vs working
tree). The original scope excluded FPS/manual-control (arcade) mode and the
enemy-AI decision layers; both have since been ported (enemy AI in commit
37a0d1c, arcade mode in Stage 9 below), so the port now has full mechanic
parity. Robot *sensing* (CInfo env) and player auto-orders ARE in scope.

Status legend: [ ] todo, [~] in progress, [x] done+verified (build+tests).
Update this file as work lands. NOTHING below is done until marked [x].

## Stage 1 — player-side order layer (MatrixSide.cpp player logic)
- [x] 1a. CInfo/CEnemy env port (`logic/environment.rs`); robot fields
        `env`, `group_logic`; fire-dist/repair/bomb accessors
- [x] 1b. RN-place helpers: `find_near_place`, `place_list`,
        `place_list_grow` (MatrixLogic.cpp:2077-2405) + zone scratch buf
- [x] 1c. `GatherInfo(0/1)` (MatrixRobot.cpp:4333) + `IsLogicVisible`
        (MatrixLogic.cpp:3188); drive from logic takt each 100ms
        (MatrixLogic.cpp:2713-2718)
- [x] 1d. Side: `SMatrixPlayerGroup` array, statistics, `ESideStatus`,
        sel-groups (CMatrixGroup → Vec), cur-sel-group, cur_sel_num
- [x] 1e. TaktPL + FirePL + RepairPL + WarPL + PL helpers (PLIsToPlace,
        PLPlacePos, PrepareBreakOrder, point_of_aim, underfire calc)
- [x] 1f. PGOrder{Stop,MoveTo,Capture,Attack,Patrol,Repair,Bomb,
        AutoCapture,AutoAttack,AutoDefence} + PGAssignPlace[Player] +
        PGSetPlace/PGPlaceClear/PGCalc*/PGShowPlace/PGFind*/
        PGCalcRegionPath/PGRemoveAllPassive + SelGroupToLogicGroup +
        RobotToLogicGroup
- [x] 1g. Input wiring complete: OnRButtonDown dispatch, all 6 preorder
        buttons + autos (6a), double-click select-all, minimap orders (6i)
- [x] 1h. Group management: CreateGroupFromCurrent, AddToCurrentGroup,
        PumpGroups, Reselect, SelectedGroupUnselect/BreakOrders,
        RemoveObjectFromSelectedGroup, GetCurSelObject, ShowOrderState
- [x] 1i. Robot: CanBreakOrder (capture never breaks), IsDisableManual,
        ROP_GETING_LOST phase, BreakAllOrders semantics
- [x] 1j. Win/lose: CheckStatus from Takt (MatrixLogic.cpp:2773-2820),
        side status SS_*, stat counters (ROBOT_BUILD etc.)
- [x] 1k. Robot Damage→env wiring: LastHitTarget/Enemy/Friendly,
        GetLost-on-hit, minimap flash

## Stage 2 — robot lifecycle gaps
- [x] 2a. ReleaseMe cascade on death: BreakAllOrders, weapon FireEnd
        release, selection kill, deregister from groups/selection/UI
- [x] 2b. Selection rings: SGROUP flags ported; ring lifecycle already
        covered by the per-frame sync_selection_ring reconciler
- [x] 2c. Order queries: FindOrder, RemoveOrder(pos), HaveBomb
- [x] 2d. ApplyNaklon — N/A: defined but never called in the C++ (dead code)
- [x] 2e. InSpawn 10s base-watchdog ported (time_with_base)

## Stage 3 — buildings/cannons gaps
- [x] 3a. Maintenance + supply drops (building_maintenance/order_flyer in side_player.rs; UI trigger button lands with 6a)
- [x] 3b. Capture zahvat visual state (14-point ring) + Zahvat effect
- [x] 3c. Building selection rings — covered by sync_selection_ring reconciler
- [x] 3d. Turret slot visuals — already ported (slot_marker.rs)
- [x] 3e. Building ReleaseMe: cannon unbind, capture-candidate clear
- [x] 3f. Cannon selection N/A (no CANNON in C++ ESelection); bar clones → 6g

## Stage 4 — flyer (delivery only; manual mode excluded)
- [x] 4a. CMatrixFlyer port: FO_GIVE_BOT delivery state machine, takt,
        SetTarget placement, body/engine units + rotor anim
- [x] 4b. ElevatorField beam effect + robot carry attach
- [x] 4c. Robot delivery: base OrderFlyer → flyer spawn → carry robot →
        drop at target (ROBOT_FALLING + impact dust)

## Stage 5 — effects gaps
- [x] 5a. Shleif — done-by-design: AddSmoke/AddFire map to standalone
        Smoke/Fire spawns; missile/bomb emit_trail + ablaze flame
        tongues already fire at the C++ call sites (verified)
- [x] 5b. Zahvat spinner (with 3b)
- [x] 5c. Dust (robot movement/landing) + spawn sites
- [x] 5d. ElevatorField (with 4b)
- [x] 5e. Repair beam — already ported (weapon.rs repair_bb glints + repair target seek); verified
- [x] 5f. Shorted arcs: robot/cannon DOT arcs existed; added weapon-hit arc + plasma spot (MatrixEffectWeapon.cpp:556-565)
- [x] 5g. Per-type caps: explosions/fireanim 50, smoke+fire 100 with
        delete-oldest (enforce_limits in effects/mod.rs); spots/pointlights
        have their own renderer budgets

## Stage 6 — UI mechanics
- [x] 6a. Order buttons wired: ost/ogo/ofi/oca/opa/obomb/orep preorders,
        auto toggles (oacap*/oafr*/oafc*), bure maintenance, ocan cancel.
        Ordering-glow highlight = minor polish, not ported (no gameplay)
- [x] 6b. CAnimation: ElementAnimation parsed from Animation/Frames
        config, ticked in iface update, drawn via current_image
- [x] 6c. Cursor — done-by-design: the C++ only ever selects
        CURSOR_ARROW (scoping audit); the OS arrow cursor is equivalent
- [x] 6d. Verified complete in working tree (group icon grid + per-icon
        HP bars + personal portrait + ramka + name/HP text)
- [x] 6e. Verified complete (popup bake renders item.text; affordability
        recolor via refresh_affordability)
- [x] 6f. False positive: C++ CCounter has no rolling animation; counter
        logic (limits/ManageButtons/MulRes) already ported
- [x] 6g. Verified complete (robot CLONE1 per-group-icon 46px bars +
        CLONE2 68px selected bar + building bars all present)
- [x] 6h. Turret cases were already complete; added callhell/bure
        maintenance countdown replacements (_ch_cant/_ch_can/_ch_time_*)
- [x] 6i. Drag/click/zoom verified complete; added minimap right-click
        move order + red ping (MatrixSide.cpp:821-830)
- [x] 6j. Done-by-design: index.html spinner + fade covers the load
        bar; native logs progress
- [x] 6k. Done-by-design: CSS canvas fade replaces the D3D render-
        target transition (device-capture not applicable to wgpu)
- [x] 6l. N/A: dev/cheat overlay (CHEATS_ON), not a game mechanic;
        port has FPS counter + console logging

## Stage 7 — sound
- [x] 7a-c. FAITHFULLY SILENT: every CSound::Play in the original is
        gated on `g_RangersInterface` (the SR2 host); the standalone
        EXE build the port replicates has it NULL → no audio at all.
        The dispatch surfaces (MapLogic::sound_queue GameSound events +
        interface UiSound) are ported and verified so a host backend
        can be attached later, exactly like the original DLL mode.

## Stage 8 — docs/cleanup
- [x] 8a. CROSSREF.md updated (side_player, road_network, flyer,
        logic/environment, effects/{zahvat,dust,elevator_field,move_to})
- [x] 8b. D3D9-infra files marked N/A-by-design in CROSSREF
- [x] 8c. cargo build clean (0 warnings), 188 lib tests pass, wasm
        check green, all examples build (fixed pre-broken check_shore);
        new tests: env rings/strikes, group sync/unselect, logic-group
        alloc, win/lose CheckStatus, BreakAllOrders

## Audit false-positives verified OK (no action)
- Cannon rendering/aiming/fire (object_cannon.rs + renderers complete)
- Constructor prices/total cost labels (committed earlier)
- Robot selection panel portrait/group icons (committed earlier)
- MoveTo path-dots: C++ Path effect has no spawn sites; MoveTo ping ported
- Minimap core (bake/zoom/markers/arrows/frustum/events all FULL)

## Stage 9 — arcade / first-person (manual-control) mode  [2026-07-07]

Previously excluded ("FPS/manual-control mode"), now ported in full so
the port has feature parity with the original's arcade mode. Verified:
`cargo build` clean, 221 lib tests (6 new arcade tests), wasm check
green, and an end-to-end headless-browser run on SPHERE.CMAP (enter →
drive → steer → fire → leave, no panics; screenshots captured).

- [x] 9a. Core state: `Objects::arcaded_object` setter +
        `Objects::arcade_input` (WASD/fire/cursor snapshot). Robot
        `max_speed_boost` (SPEED_BOOST 1.1) + `is_arcaded` cache.
- [x] 9b. `MapLogic::set_arcaded_object` (MatrixSide.cpp:1290-1355):
        reject non-player, chassis engine-loop sound, ×1.1 speed +
        ×1.2 weapon coeff, SelectArcade flags, BreakAllOrders on enter;
        hand back to AI (muted PGOrderAttack) + undo boosts on exit.
- [x] 9c. `MapLogic::arcade_takt`: dead-robot handover release +
        OnForward/OnBackward manual move orders. Arcaded object takes a
        single whole-frame StaticTakt (MatrixLogic.cpp:2749) after
        proceed_logic skips it.
- [x] 9d. Robot LogicTakt arcade branches (MatrixRobot.cpp:843-1038):
        hull tracks cursor trace, A/D steer flags → RotateRobot ±90°,
        W/S LowLevelMove bypassing pathfinding, LMB → Fire(cursor) /
        release → StopFire; hull-servo sound hysteresis; GetLost no-op;
        weapon range-cutoff skipped for the player bot.
- [x] 9e. Camera CAMERA_INROBOT mode (MatrixCamera.cpp:895-1058):
        per-mode angle/dist params, CalcLinkPoint chase-follow (speed-
        lerped forward offset, yaw locked to robot heading), mode-change
        easing `1-0.995^ms`, pan/edge-scroll disabled in-robot.
- [x] 9f. Input wiring (MatrixFormGame.cpp): held-key + LMB snapshots,
        Enter/Space enter-from-selection & leave, Esc leaves, E self-
        destruct, mouse-cam drag steers, marquee/right-orders gated off.
- [x] 9g. UI: `inro`/`lero`/`sbo` visibility (CInterface.cpp:1510-1544),
        Main-panel ±196px arcade slide, arcade crosshair cursor,
        minimap recenters on the arcaded object.
