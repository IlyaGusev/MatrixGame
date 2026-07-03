# Visual/pipeline sweep (2026-07-03, final) — the previously-unaudited code

## FIXED — pipeline
- VO animation frame times: negative durations now abs()'d at all 3
  cursor sites (GetAnimFrameTime = abs, VectorObject.hpp:389) — 26
  shipped anims (chassis3 track roll, passage gates, decorative
  idles) played hundreds of times too fast.
- Terrain gloss: true CAMERASPACEREFLECTIONVECTOR R=2(N·E)N−E (view
  matrix added to the gloss uniforms) — was sampling by camera-space
  normal.
- Label word-wrap uses the FULL element width (m_boundX = m_xSize,
  CInterface.cpp:775/4674) — label.x/sme_x only offset the blit.
- Pressed UI element Reset()s on drag-off (CInterface.cpp:979-984);
  the press is dropped so drag-off-then-release can't fire the click
  or desync check groups.
- Water directional diffuse uses LightMainColorObj (device light 0,
  MatrixGame.cpp:676) not the terrain-bake LightMainColor.
- Dust puffs grow only halfway to r2 (`*k*0.5`,
  MatrixEffectBillboard.cpp:79).

## FIXED — effect visuals
- Intense explosion fireballs: SetScale is absolute (1→15) — were 4×
  too large on every preset with intense_cnt.
- Elevator carry beam: helices rebuilt as Catmull-Rom THROUGH the 6
  points (Trajectory/Init2) + the ~1 rad/s field spin — was a
  near-straight non-rotating Bezier.
- Shocked turrets spawn electric arcs each 50 ms DOT
  (MatrixObjectCannon.cpp:993), same as robots.
- Missile/bomb trail smoke+fire hang static (speed=0 at all 5 C++
  call sites) instead of rising/spinning.
- Gun/cannon muzzle streaks and shell tracers fade to BLACK (LIC to
  0) rather than staying hot white.
- Blast flame (FireAnim) rebuilt: vertical line billboard 35×60,
  grow over first 10% of TTL / shrink over the rest at constant
  alpha, 100 ms frames from a pseudo-random start.
- Muzzle ground glow visible (alpha was 0 → shader drew nothing) and
  turret flashes use radius 40 (guns 60).
- Flame: head billboard fades linearly (k1) with only tail bills k1²;
  FLAMELINE width/alpha from the OLDER endpoint; `m_Break` ported —
  no flame streak across firing bursts.
- BigBoom third shell is white (0xFFAFAF40 was the dead light arg).
- Explosion debris: dies off-map and on base footprints; water hits
  splash (CreateKonusSplash); fire debris winds down over 1000 ms
  instead of popping; fire emitters rise at 0.02 not 0.04.
- Repair beam: solid 12-segment beam removed (C++ draws glints only);
  glint tails t−0.06.
- Selection ring anchors at robot body z+3 (was geo_center ≈ +9).
- Laser-on-water steam is invisible like the original (alpha 0).
- Explosion flash light at max(pos.z, ground+10) — mid-air blasts
  flash in place.
- Plasma muzzle cone is weapon-owned and re-aims with the barrel
  (Konus::Modify semantics).
- Income billboards rise from the building top (GeoCenter().z + 40).
- Robot DIP debris landing on base footprints dies silently.

## Second pass on the "approximations" — now FIXED too
- BigBoom shell: standalone effect textures get a WRAP sampler
  (D3D9's device default) and the shell samples raw rotated ×k·2
  coords — tiled spinning noise like the original. The BBT atlas
  keeps clamp (sub-rect UVs must not bleed).
- Plasma bolt renders its BBT_POINTLIGHT glow sprite (r 20,
  0x80202030, MatrixEffectFirePlasma.cpp:29).
- Minimap zi/zo grey out at the scale bounds (disabled tint).
- Hover keelwater is now a flat FORWARD-ORIENTED ground quad growing
  over 3000 ms (KeelwaterFx via queued tris) — was a fixed
  camera-facing line.
- `</color>` restores the enclosing color via a tag stack; an
  oversize word arriving mid-line breaks the line first and
  char-splits instead of overflowing the element.

## Third pass — the last two visual items, now FIXED
- Repair beam fully shaped: the 10-phase wobble sine bank, the growing
  off-target amplitude (0.001·t capped at seek·0.2), the straight
  4-point seek spline, the 9-point wrap-around spline (out → front·1.4
  → side → back → −side → front·1.4 → back to the muzzle, each point
  wobbled by len·0.01), the 500 ms found-morph between them, and a
  pulsing muzzle glow. Only the lost-target OUT-morph is snapped (the
  C++ target core outlives the loss; ours doesn't).
- Fire-debris embers cast their moving terrain glow (r 20, 0x22222202)
  via a synthetic follow-light key space, killed when the emitter
  winds down.

## Remaining non-bugs (complete list, none silent)
- Kept-better-than-C++ on purpose: `$light` hex parse uses the
  artist's value (the C++ GetHexUnsigned folds trailing spaces into
  garbage — armor1/2.vo fade color); BlockPar trims values (the C++
  preserves leading spaces — load-bearing for our VO/texture
  consumers, untriggerable by shipped data along with `}`-same-line
  parsing); UI element animation overlay model (no panel Animation
  blocks exist in shipped data).
- Absent subsystems by user decision (GOAL_PROGRESS.md): enemy-AI
  decision layers, FPS/arcade mode.
- Audio: VERIFIED UNIMPLEMENTABLE from this repository. All five
  shipped archives (robots/common/mainmenu/russian/forms.pkg) contain
  ZERO audio files (checked via examples/find_audio.rs); the Sounds
  block's `path` values ("Sound.ButtonClick", "Sound.Help", …) are
  SR2 HOST resource identifiers — the original DLL plays audio via
  `g_RangersInterface->m_SoundCreate/…` callbacks into the Space
  Rangers 2 host process, whose audio assets are not in this repo.
  The port's fully-wired silent dispatch queue (118 sound blocks
  parsed, every CSound call site routed) is the maximum achievable
  fidelity, same as the host-supplied `_begin` mission text.

# Bug hunt round (2026-07-03) — user-reported bugs + 4-agent sweep

All fixes below verified: cargo build, 207/207 lib tests, wasm32 check,
attack_order_sim, collision_sim (track+hover crowd runs, 0 violations),
check_maintenance.

## FIXED — user-reported
- **Helicopter reinforcements had no initial cooldown**: C++ seeds
  m_MaintenancePRC from the `MaintenanceTime` map property and calls
  InitMaintenanceTime() at load (MatrixMapPrepare.cpp:411-418); the port
  never did either. `GameMap.maintenance_prc` + seeding in
  apply_side_resources (verified 500000 ms at load, ticking). Also the
  drop-point `PlaceFindNear` snap was discarded (cx,cy now updated,
  MatrixObjectBuilding.cpp:1285) and drop spots could land under live
  turrets (cannons now carry an RN place, see below).
- **Robots sometimes go through static objects** — root cause: MapPosCalc
  = PlaceGet (MatrixLogic.cpp:465-495) clamps to bounds and NUDGES the
  footprint anchor off a stop-blocked cell to (x+1,y)/(x,y+1)/(x+1,y+1);
  the port truncated unconditionally, so a robot hugging an obstacle fed
  a blocked start cell into the pathfinder, whose first-step stop
  exemption legitimised path segments through the obstacle footprint
  (then the shared inside-cell quirk let it traverse). Nudge ported into
  map_pos_calc. Related: map_pos_calc now runs at spawn for all initial
  + factory robots (MatrixMapPrepare.cpp:1721) so gather_info/radio
  regions don't read cell (0,0).

## FIXED — high severity (agent sweep)
- Cannon side-flip + cannon death now scrub every robot env
  (MatrixObjectCannon.cpp:888-897, 1600-1610): no more shooting your
  own freshly captured turrets; war groups retarget off dead turrets.
- BuildStack DeleteItem/ClearStack teardown ported
  (MatrixObjectBuilding.cpp:1844-1919): capture/neutralize/death now
  refund queued items, silently delete queued ghost turrets, free the
  slot + turrets_have (bases no longer become permanently uncapturable);
  build-queue icons (_dynstack_N) are clickable to cancel with refund;
  building death kills queued ghosts.
- Building capturer flag released on every capture-order drop
  (SOrder::Release, MatrixRobot.hpp:219-229) via OrderList.released
  queue → recapture no longer permanently blocked.
- Stale group road route: the 3 same-region auto-retarget branches
  (PGFind{CaptureFactory,AttackTarget,DefenceTarget}) redistribute the
  per-robot route snapshots after clear_fast.
- Turret RN places (m_Place via FindInPL) + marked in every occupancy
  pass — robots no longer assigned places under turrets.
- seek() now returns the real rotate out-param (was: always true) —
  pneumatic ROBOT_FLAG_COLLISION unstuck; foot-link body correction live.
- LMB on empty terrain / enemy keeps the current selection
  (MatrixFormGame.cpp:674-741); empty marquee keeps selection.
- Armed pre-orders (move/fire/patrol/bomb) execute via minimap LMB
  (MatrixSide.cpp:672-680).
- is_special() ported; attack paths use IsLive()||IsSpecial()
  (MatrixSide.cpp:704/729/841) — special win-target objects attackable,
  scenery no longer a valid fire/bomb target.
- RobotSpawn rally: fresh player robots get AssignPlace + muted
  PGOrderAttack near the base (MatrixRobot.cpp:2216-2221) — no more
  pile-up at the base door.

## FIXED — medium/low (agent sweep)
- get_lost: >70° rotate early-out, self-exempt PlaceFindNear,
  unconditional MoveTo (MatrixRobot.cpp:5201-5236).
- Shift+marquee toggles off already-selected robots; marquee hit-test
  now projected bounding-circle vs rect (approximates C++ InRect vertex
  test — partially-boxed robots select).
- RMB cancels turret placement (drops ghost); misclick keeps placement
  armed; slot snap radius 20 wu (rr<4 cells², MatrixSide.cpp:1645).
- Double-click select-all resets cur_sel_num; WarPL advances to next
  robot after place_list_grow.
- Build button disabled when stack full (MAX_STACK_UNITS); deduction
  charges only actually-queued robots; robot cap counts only queued
  ROBOTS (not turrets) and excludes DIP robots.
- Repair beam ranks patients by distance from the seek center
  (find_objects passes candidate center — now uses center3) + robot
  preference works.
- Start-of-game: player base auto-selected + camera fallback to base
  (RS_CAMPOS, MatrixMapPrepare.cpp:951-985); "Begin" mission dialog
  shown at load when the template exists; buildings fast-forwarded
  (LogicTakt(100000)) for the initial income tick.
- Pause now freezes graphic takt / water / walk anims (C++ pause returns
  before CMatrixMap::Takt) — no more effect-burst on unpause.
- Sides start SS_NONE, activated only for base/robot owners; default
  resources 300 (C++ ctor); first status check at t≈0 (prev=-1500);
  terron death also sets terron_dead (attrition-death exemption).
- Side LogicTakt paced on the m_TaktNext accumulator (100/sec at any
  FPS); cannon fire think seeded now+100ms; carrying robots get the
  separation push (MatrixRobot.cpp:678-691); heat_mod Float2Int
  rounding; fire-end handler gated on !DIP shooter; mortar keeps the
  3D barrel axis (steep-drop branch reachable); per-projectile water
  impacts (missile explodes, shells nothing, bomb shallow-water crater
  +fire); supply-flyer dispatch capped at 16 (MAX_ALWAYS_DRAW_OBJ).

## Round 2 of the sweep (same day) — deferred items resolved + new subsystems
- RobotSpawn now sets ROBOT_FLAG_DISABLE_MANUAL (MatrixRobot.cpp:2187),
  cleared at move-out completion (:802) — spawn-pad robots reject
  manual orders like the original.
- AllocPlaceForOrderOnTop base-close (MatrixRobot.cpp:4558-4563):
  RESOLVED BY TRACE, not ported — with the flag lifecycle above, C++
  can never reach that branch (base attached ⇒ DISABLE_MANUAL set;
  the aborted-capture case is handled in BreakAllOrders, which the
  port already mirrors including MustDie-on-ride).
- EscapeFromBomb move-to marking now replicates the C++ FindPlace
  bounds-check typo (MatrixLogic.cpp:2138) — bit-faithful retreat
  spot selection.
- RS_EFFECTS ambient effect spawners PORTED (MatrixMapPrepare.cpp:
  902-926 + CEffectSpawner, MatrixEffect.cpp:105-190): smoke/fire
  specs parsed from map IDS strings, ticked from MapLogic::takt
  (4915 spawners across the 84 shipped maps; sound/lightening kinds
  skipped — silent layer / paired subsystem).
- Cannon m_UnderAttackTime ported (decay + player-hit arming,
  MatrixObjectCannon.cpp:900-904/1407-1417) — ready for the sound
  layer, consistent with buildings.
- Dead "H" hotkey arm removed (all C++ KEY_H sites are cheat
  sequences; the port's arm dispatched a nonexistent button).

## Hints audit (dedicated agent) — fixed
- _ALIGN honored: alignx tracked per _TEXT part; _ALIGN:0:0 templates
  (every shipped tooltip) render left-aligned; only alignx>=1 centers.
  (Round-6's "center everything" had it backwards.)
- Turret hint DPS: Float2Int(1000/cooldown) FIRST then × damage
  (CInterface.cpp:4464) — turrets 1/4 showed 13+13/24+24 instead of
  12+12/25+25.
- Click kills the hovered hint (CIFaceButton.cpp:97-103) so dynamic
  values (costs, countdowns, toggles) rebuild fresh.
- Popup menu no longer suppresses STATIC hints (CIFaceStatic.cpp:26-40)
  — Top-panel resource tooltips work with a popup open.
- _BITMAP with unresolved alias emits a null 0×0 element consuming its
  layout modifier + COPY anchor (MatrixHint.cpp:671-691).
- Template `|`-continuations append regardless of interleaving
  (MatrixHint.cpp:519-563).

## Final sweep (effects/flyers/map-objects + UI/minimap/camera/loaders) — fixed
- Flyer altitude: was hugging exact terrain triangles and clipping
  through structures; now samples a smoothed per-group envelope of
  terrain max + static building/object tops (the static slice of
  GetZInterpolatedObjRobots, MatrixMap.cpp:512-546; live robots are
  covered by FLYER_ALT_MIN). Lazy grid on Objects, bilinear sample.
- BEHF_BURN `Burn,<repl>,Type` rows: after 5 s aflame the object now
  re-Inits to the burnt replacement type and demotes to STATIC —
  permanently inert (MatrixObject.cpp:1576-1595). ("Tex" char-skin
  ramp remains render-side.)
- BEHF_SPAWNER pacing: idle detection poll fixed at 107 ms
  (MatrixObject.cpp:1358); behaviour par 1 is the post-trigger re-arm
  delay (was misused as the poll period, with a bogus hardcoded 3 s
  cooldown).
- Popup hover-preview: leaving the rows keeps the last hovered
  component visible until commit/cancel like the C++ (Djeans007 fires
  on hover only; restore happens in ResetMenu).
- Minimap: zoom buttons compound from the currently animating scale
  (MatrixMinimap.cpp:1344-1354); event pings draw UNDER object blips
  (:638-763).
- Storage::block_params now case-insensitive like block_param
  (repeated-key enumeration could silently miss mixed-case entries).

## CInterface per-frame state engine — coverage completed (exhaustive
## audit of CInterface.cpp:156-4621), all findings fixed
- Robot HP readout now /10 (GetHitPoint, MatrixObjectRobot.hpp:243) —
  was showing 1150/1150 instead of 115/115.
- SINGLE_MODE panel ported: `wsl` weapon plate + per-weapon icons at
  the m_DWeaponX/Y grid + overheat overlays (alpha = heat·0.25, via a
  new per-element visible_alpha) for a lone selected robot; group grid
  + ramka suppressed in single mode (CInterface.cpp:1510-1574,
  3013-3114).
- Group-icon clicks wired (CInterface.cpp:1061-1080): switch active
  member / promote to single / shift-remove from group.
- JumpToBuilding wired for all five plant portraits (titpl/plaspl/
  elecpl/batpl/basepl, CInterface.cpp:577-586).
- Group ramka frames now behind EVERY occupied slot (never an
  active-slot marker), hidden in single mode (CInterface.cpp:1546-1571).
- Constructor pylon affordability outlines ported (NORMAL_RAMKA green /
  CRITICAL_RAMKA orange, CInterface.cpp:1885-2097) via per-element
  ramka_color + flat-colour border strips in the renderer.
- callhell greys out during the maintenance cooldown
  (CInterface.cpp:1609-1613).
- Base income caption uses the side's real force-up multiplier
  (GetIncomePerTime, MatrixSide.cpp:352-376).
- Order-glow aggregated across the whole selection with the C++
  cross-suppression rules (move vs capture/bomb/repair; bomb/repair
  glow needs a bomber/repairer) — several can glow at once
  (CInterface.cpp:1316-1354, 1663-1677).
- Turret-kind picker gated on a building/base selection
  (CInterface.cpp:1708-1741).
- Not fixed by design: minimap zi/zo disabled ART at scale bounds
  (behaviorally identical — clamped; custom minimap button render has
  no disabled art variant).

## Verified faithful / no action (final sweep)
- Income "+N" billboards confirmed wired via accrue_resources (agent
  finding was a false positive).
- Hints: layout ops, colors, substitution, positioning, hover delay
  (700 ms deliberate), dynamic replacements.
- Projectiles: gun/cannon/missile/bomb flight math fully faithful
  (incl. the C++ no-op speed clamp); DOT periods; BigBoom; Flame;
  Terron; flyer flight + FO_GIVE_BOT sequence; BEHF_BREAK/SENS.
- Constructor core, camera strategy mode, minimap transforms,
  STRG/ZL03/CDataBuf loaders — line-verified faithful.
- Deliberate improvements kept (safer than C++): popup-cancel restores
  the full config (C++ drops weapons after a pylon gap); counter
  down-button disable (C++ typo); build-button affordability ×counter
  + charge-only-queued; flyer death releases the carried robot (C++
  leaves a dangling pointer); elevator-field activation dt-normalized.
- Not applicable to shipped data: BlockPar leading-space values and
  `}`-same-line parsing (the text parser only loads VO/texture
  property tables from the pkg); enemy-side spawner bots idle (enemy
  AI out of scope — routing them through player groups would corrupt
  real command groups); BEHF_ANIM idle anim-state chains (map-object
  clip clock lives render-side; damage transitions ARE ported).

# Fresh C++↔Rust parity audit (2026-07-01) — in-scope gaps to fix

## DONE (build clean + 207 tests green) — 23 fixes
- weapon.rs: H1 volcano visuals (muzzle+splash+spark konus+trassa); M1 bigboom 3 rings+central explosion
- object_cannon.rs: H2 ReleaseMe frees turret slot/turrets_have; lightning stops fire-loop anim
- interface.rs+form_game.rs: H3 order buttons opa/oca/obomb/orep/prog visible (bomber/repairer gated);
  callhell→maintenance dispatch+enabled; mm→menu dispatch
- robot.rs: M3 bad-coord ring clear on no-collide; M4 rotate-hull 3x for ≥160°; sole track/wheel decals
- object_robot.rs: M5 sub-part mounts use live chassis frame (not frame 0)
- logic.rs: M10 all sides accrue building income (per-side force_up); M11 place_is_empty skips DIP
- side.rs+map.rs+logic.rs: M9 SideResInfo starting resources+force_up seeded at load
- map.rs: DisableInshore property gate
- minimap.rs: M12 flyer blips drawn
- effects: shorted-arc glow-cap suppressed; selection ring size fixed 20; flame smoke trail; flameline;
  SpotKind SoleTrack/SoleWheel

## DONE — round 2 (build+tests green)
- side_player.rs: FirePL/WarPL repair target uses is_live_unit (no building repair)
- object_building.rs: build-stack frozen on DIP/unowned building
- moving_object.rs: bomb early-trail smoke 300 (vs missile 400)
- elevator_field.rs: fixed 0.1 tracer tail + correct dt spread
- **Dynamic point lights** (point_light.rs transient system + Objects.pending_point_lights queue +
  form_game drain/takt): all 10 explosion presets carry light_radius1/2 + light_color1/2 and
  spawn an animated flash; gun/cannon muzzle flashes spawn a fading light. Terrain lights up around
  blasts. (Follow-lights for plasma bolt / flame / growing bigboom still deferred — need per-effect
  light-handle management threaded into effect takts.)

## DONE — round 3 (build+tests green)
- effects/mod.rs + logic.rs: voronka craters suppressed on base 3×3 footprints (is_on_base)
- landscape_spot.rs: plasma-hit decal per-axis slope compression (LSFLAG_SCALE_BY_NORMAL)
- map.rs: camera terrain-follow clamp to [ground_z_base_middle, ground_z_base_max] from building floors

## DONE — round 4 (build+tests green)
- map.rs + logic.rs: pre-placed base ruins (side==255) spawn as ruin-mesh MapObjects (spawn_ruins)
- terrain.wgsl: surface overlays modulate alpha by m_Color.a (non-white surfaces no longer too opaque)

## DONE — round 5 (build+tests green)
- Follow-lights: plasma bolt + flame cast a moving terrain glow via keyed
  pending_light_follow/kill queues drained into a light-handle map in the app loop.
  Point-light feature now complete (explosions + muzzle + plasma + flame).

## DONE — round 6 (build+wasm+tests green)
- Storage: added block_params (repeated-key) + block_records (child-block) enumeration
- config.rs: RobotSpawnConfig catalogue (mirrors GiveBotConfig) + global Arc accessor, loaded in load_config
- object.rs: BEHF_SPAWNER state machine (detect player robot in sens_radius → queue spawn on a timer)
- logic.rs: drain_spawner_bots — builds robot from catalogue, spawns, PGOrderAttack nearest player robot;
  Objects.pending_spawner_bots queue. Enemy-wave spawner objects now function.

- hint.rs: hover-hint text now centred (alignx=1) like the dialog path

## DONE — round 7 (build+wasm+tests green)
- Auto-order ON/OFF button state: iface shows _ON/_OFF variant from show_order_state(); dispatch
  toggles correctly (ON cancels via PGOrderStop). Completes H3 interactive order UI (minus osel glow).
- Income "+N" billboards: ScoreFx glyph-billboard effect (digit/icon layout + rise/fade) + billboard_disp
  in-plane displacement + trigger in accrue_resources for player buildings (t/e/b/p10, a<amount>).

## DONE — round 8 (build+wasm+tests green)
- robot.rs: drowning death (MustDie when non-hover chassis sinks below WATER_LEVEL-100)
- map.rs: empty-group (grpsc==0) surfaces no longer drawn (matches C++)
- hint.rs: off-screen hints shown clipped (not dropped); _DOWN/_RIGHT/_POS clear the COPY anchor
- multi_selection.rs + form_game.rs: marquee selects a lone player building; live re-selection
  highlights boxed units during a (non-shift) drag

## DONE — round 9 (build+wasm+tests green)
- side_player.rs: EscapeFromBomb fully ported (3-pass: bomb-robot scan, covering-robot exemption,
  place-graph wave to a safe spot > escape_dist from every bomb robot → MoveReturn+MoveTo). Single
  road-network lock, deadlock-free (move_to/find_near_place_impl/find_in_pl take no re-lock).

## DONE — round 10 (build+wasm+tests green)
- interface.rs + form_game.rs: order-glow — the osel icon (id ORDERS_GLOW_ID+0..5) lights for the
  active group's order (stop/move/patrol/fire/capture/bomb-repair). H3 order UI now fully complete.
- form_game.rs: gameplay cursor — context-sensitive OS cursor (crosshair when an order/turret
  placement is armed, move in the screen-edge scroll band, arrow otherwise), approximating
  CURSOR_CROSS/STAR/ARROW consistent with the port's OS-cursor substitution.

## DONE — round 11 (build+wasm+tests green)
- Pneumatic foot-linking (object_robot.rs + robot.rs): BuildPneumaticData analyses the Move/MoveBack
  anims' left/right foot bones (ids 2/3) into a per-frame plant table; LinkPneumatic stamps
  SPOT_SOLE_PNEUMATIC footprints on each new plant and nudges the body so the planted foot doesn't
  slide. Uses a flat up-vector; the body correction is clamped (<2·GLOBAL_SCALE) so a mis-built table
  can't teleport the robot, and falls back to the old behaviour if the table can't be built.

## DONE — round 11b
- terrain.wgsl: surface overlays now tint only the base texture by m_Color (macrotexture term
  left untinted), while base terrain keeps full-blend vertex lighting.

## DONE — round 12 (build+wasm+tests green) — final cosmetic tail
- hint.rs: h_delta height rounding (center slice tiles a whole number of times).
- hint.rs + sound.rs: _SOUNDIN/_SOUNDOUT parsed → SoundIn on show / SoundOut on hide (dispatch
  surface present; silent like the rest of the sound layer).
- terron easter egg (map.rs globals + logic.rs terron scan + robot.rs in-position flag +
  object.rs portret render gate): BEHF_PORTRET map objects are hidden until a lone full-HP
  bomb-carrying player robot holds the trigger cell on the "terron" map (revealed permanently
  when it dies). Zero effect on other maps.
- VO DrawLights (vector_object.rs + object_robot.rs): `$`-bone emissive lights parsed from the VO
  property table (radius + colour intervals) and drawn as animated Pointlight billboards at EVERY
  part's light bones — chassis, armor, head, and all weapons (via collect_part_lights over the
  full part chain).

## DONE — round 13 (build+wasm+tests+3 sims green) — sound-dispatch wiring complete
All 32 missing CSound::Play/AddSound dispatch sites from the sound audit are now wired
(playback stays impossible — 0 audio files ship in any .pkg; keys log via the silent layer):
- Infra: `Objects::pending_sounds` + `queue_snd`/`queue_snd_at`, drained in MapLogic::takt;
  `interface::sound::play_named` for UI-layer keys.
- weapon.rs: fire_sound_key/hit_sound_key maps; fire sound at muzzle; hit sound in apply_damage.
- object_building.rs: under-attack voices s_side_attacked_1..3 (throttled), death voices
  (s_base_dead/s_fa_dead/s_building_dead), platform/door stops, rolling DIP expl_bb,
  base-final expl_bb4, ruin-swap expl_bb3, robot build-end r_build_e/_alt 50/50
  (MatrixObjectBuilding.cpp:1693-1695), turret build t_build_0..3 (:1780-1786).
- object.rs: terron pain s_terron_pain1..4 + s_terron_killed (MatrixObject.cpp:156-175),
  expl_bb4 at both bigbooms (:1266/1280), rolling expl_bb pops on the 100-500ms timer
  (:1293-1298, new next_snd_time field); SENS activate/deactivate config sounds (BEHAVIOUR
  par-1 fields 4/5, :1509-1528); spawner open/spawn/close config sounds (fields 8/9/10,
  :1368/:1428/:1468 — all three fire at the collapsed one-tick spawn).
- robot.rs: base capture s_eb_cap/s_pb_cap (MatrixRobotAI.cpp:1354-1366), factory capture
  s_ef_cap/s_pf_cap at CaptureStatus::Done (MatrixObjectBuilding.cpp:1052-1063),
  landing s_upal (MatrixRobot.cpp:604).
- side.rs: selection voices s_selection_1..7 + s_base_sel (first-select suppression via new
  base_sel_sound_enabled = MMFLAG_SOUND_BASE_SEL_ENABLED) + s_building_sel on all three
  selection commit paths (MatrixSide.cpp:976-1016).
- side_player.rs/logic.rs: s_maintenance after init_maintenance_time (MatrixObjectBuilding
  .cpp:1295); s_maintenance_on when the countdown hits 0 (MatrixLogic.cpp:2655-2663).
- form_game.rs: t_build_s on turret placement commit (MatrixSide.cpp:659).
- object_cannon.rs: cannon under-attack voices (MatrixObjectCannon.cpp:1407-1417).
- minimap.rs: map_plus/map_minus on zoom buttons (MatrixMinimap.cpp:1347/1353).
- iface_list.rs LIVE BUG FIX: CHECK_PUSH button-down now routes through
  for_check_push_button_down so `conf*` presets fire S_PRESET_CLICK (was dead code).
- explosion.rs/effects/mod.rs: ExplosionProps gained the preset `sound` key
  (expl_norm/expl_missile/expl_rh/expl_lh/expl_bb2/expl_rb/expl_rbs/expl_bigboom/expl_obj;
  BuildingBoom = S_NONE), played at the pending_explosions drain (MatrixEffect.cpp:481).
- elevator_field.rs: ef_start in ctor / ef_end in Drop (MatrixEffectElevatorField.cpp:26/41).
- splash: "splash" queued at all four KonusSplash spawn sites (MatrixEffect.cpp:803).
Not wired BY SCOPE: hull-turn voices (PlayHullSound, MatrixRobot.cpp:2297-2306) — gated on
`this == GetArcadedObject()`, i.e. arcade/FPS manual mode, which is user-excluded.

## DONE — round 14 (build+wasm+tests+3 sims green) — resolved-list cleanup round
Re-verified every item in the historical HIGH/MEDIUM/LOW lists below against the code and
fixed the stragglers:
- M13 [form_game.rs] `srb` (IF_SHOWROBOTS_BUTT) now dispatched: cyan 0xFF00FFFF minimap ping
  on every player robot (CMinimap::ShowPlayerBots, MatrixMinimap.cpp:1381-1390).
- JumpToRobot [form_game.rs] `_dynpers_` portrait click jumps the camera to the active robot
  (CIFaceList::JumpToRobot, CInterface.cpp:4562-4570).
- ElementAlpha [iface_element.rs/iface_list.rs/renderer.rs] per-pixel alpha hover hit-test:
  atlas alpha channels registered at decode; `hit_test_hover` skips transparent pixels
  (CIFaceElement.cpp:310-318; hover-only like the C++ — clicks stay rect-catch,
  CIFaceButton.cpp:127 vs :187).
- MAX_EFFECT_DISTANCE_SQ [map.rs/effects/mod.rs/form_game.rs] 3000-unit effect-creation cull
  at the explosion drain, frustum center fed per-frame from the strategy link point
  (MatrixEffect.hpp:13, MatrixEffect.cpp:456). Headless sims (no camera) skip the cull.
- Spot eviction [landscape_spot.rs] priority-aware: oldest lowest-priority spot evicted at the
  100 cap (plasma hit 0 < constant/sole 100 < voronka 500; MatrixEffectLandscapeSpot.cpp:611-617,
  MatrixMap.hpp:689-703).
- Minimap frustum [minimap.rs] projected on the z=0 plane like the C++ (MatrixMinimap.cpp:798-807),
  not WATER_LEVEL.
Verified already-fixed (this round's re-check): H1 volcano visuals, H2 turret-slot release,
H3 order buttons, M1 bigboom 3 rings + central blast, M2 voronka base suppression (is_on_base),
M3-M5, M6 pneumatic link + soles, M7 spawner, M8 EscapeFromBomb, M9-M12, M14-M18, camera base-z
clamp (group_max_z_interpolated), sky + sky-height (compute_sky_height_frac), edge-pan, marquee,
base ruins, ambient effect spawners, income "+N" billboards, drowning MustDie, KonusSplash Takt(0),
hint off-screen clip + COPY anchor, auto-order ON/OFF state.
Verified no-op in the original: `buhe` build-flyer button — its PREORDER_BUILD_FLYER flag is set
(CInterface.cpp:3492) but read nowhere in the C++; flyer building was cut from the shipping game.

## REMAINING — nothing to port
  - bigboom's own ring point light: the original's `CMatrixEffectBigBoom` guards it with `if(light!=0)`,
    and every `CreateBigBoom` caller (the 3 in WEAPON_BIGBOOM) passes `light = 0` — so in the shipping
    game this code never executes. Faithfully omitted (a no-op branch); the blast flash is the separate
    ExplosionBigBoom the weapon already spawns.
  - projectile trail/tracer frustum gating: the C++ `IsInFrustum` checks before drawing trails are a
    CPU-side draw-skip micro-optimization; the GPU clips out-of-frustum geometry identically. No
    behavioural difference — nothing to port.
  - `buhe` build-flyer button: sets PREORDER_BUILD_FLYER (CInterface.cpp:3492) which no code reads —
    flyer building was cut from the shipping game. Faithfully absent.
  - hull-turn voices (PlayHullSound): gated on GetArcadedObject() — arcade/FPS mode, user-excluded.
  (An earlier mid-audit snapshot of open GAMEPLAY/VISUAL/MAP/CAMERA/UI/LOW items stood here; every
  item in it has since been fixed — see rounds 1-14 — so the list was removed.)

Scope excludes FPS/manual mode and enemy-AI decision layers. Everything else must be faithful.
Severity: H breaker, M clearly wrong, L edge/fidelity.

# Historical finding lists (2026-07-01 audit) — ALL RESOLVED
Every item below was fixed in the DONE rounds above (or verified as a C++ no-op / user-excluded);
kept verbatim as the audit record. See round 14 for the final re-verification.

## HIGH (all fixed)
- H1 [weapon.rs] WEAPON_VOLCANO fires with NO visuals: `volcano_on` never set true (weapon.rs:400/497); missing impact water-splash konus, 2× landscape spark konus, 10% trassa line. C++ MatrixEffectWeapon.cpp:311-366.
- H2 [object_cannon.rs] Cannon death never frees turret slot / decrements turrets_have (ReleaseMe not ported). C++ MatrixObjectCannon.cpp:1543-1615. Rebuild permanently blocked; base-capture protection wrong.
- H3 [interface/interface.rs:557-565] Robot-group panel shows only ost/ogo/ofi; opa/oca/obomb/orep + 6 auto-order buttons never made visible. Dispatch+keyboard exist. C++ CInterface.cpp:1637-1691. [VERIFY]

## MEDIUM (all fixed)
- M1 [weapon.rs:903-910] WEAPON_BIGBOOM: only 1 of 3 rings, no central explosion. C++ :657-668.
- M2 [effects/mod.rs:134] Voronka crater not suppressed on base tiles (need group IsBaseOn). C++ MatrixEffect.cpp:491-499.
- M3 [robot.rs:2919] Bad-coord ring never cleared when collisions stop (missing else reset). C++ MatrixRobot.cpp:376-377.
- M4 [robot.rs:1214] RotateHull missing 3× fast-turn for angle ≥160°. C++ :2282-2286.
- M5 [object_robot.rs:266/329/402] Sub-part mounts read bone at frame 0; must use live vo_frame. C++ MatrixObjectRobot.cpp:490/513/570.
- M6 [object_robot.rs] Pneumatic foot-linking (BuildPneumaticData/LinkPneumatic) absent: legs slide, no sole decals. C++ :1508-1821, MatrixRobot.cpp:350-351. BIG.
- M7 [object.rs logic_takt] BEHF_SPAWNER map objects inert (no 4-state spawn machine). C++ MatrixObject.cpp:1346-1483.
- M8 [side_player.rs] EscapeFromBomb missing (player-side, mislabeled AI). C++ MatrixSide.cpp:2090-2255,6051.
- M9 [side.rs/map.rs] DATA_SIDERESINFO start resources + force-up multiplier not seeded. C++ MatrixMapPrepare.cpp:464-493.
- M10 [logic.rs:1497] Enemy/AI-side buildings generate no resources (player-only gate). C++ MatrixObjectBuilding.cpp:597-668.
- M11 [logic.rs:1645] place_is_empty counts DIP robots as blockers (missing state!=Dip). C++ MatrixLogic.cpp:886.
- M12 [minimap.rs:1242] Flyer blips never drawn. C++ MatrixMinimap.cpp:696-699.
- M13 [minimap.rs] ShowPlayerBots minimap button not ported. C++ MatrixMinimap.cpp:1381-1390, CInterface.cpp:323.
- M14 [hint.rs:1454-1511] Text centering (alignx=1) not applied to hover hints. C++ MatrixHint.cpp:590,715.
- M15 [iface_menu/form_game] preset push-buttons dispatch wrong sound (for_check_push_button_down unused); GROUP_SELECTOR_ID LogicTakt reposition missing.
- M16 [interface.rs/form_game] Order-glow (osel/ORDERS_GLOW_ID) not ported. C++ CInterface.cpp:1663-1677.
- M17 [form_game/interface.rs:522] Maintenance button callhell unclickable + hard-disabled; maintenance mis-wired to bure. C++ CInterface.cpp:3505-3508,1607-1613,3501-3504.
- M18 [form_game] Main-menu HUD button (mm) click not dispatched. C++ CInterface.cpp:3407-3408.

## LOW (all fixed or verified no-op)
- L-hint: sound dispatch names dropped; h_delta height rounding; off-screen drop vs clip; COPY anchor survives DOWN/RIGHT/POS. (700ms hover delay: user-confirmed, LEAVE)
- L-cannon: LIGHTENING hit doesn't stop barrel fire-loop anim (object_cannon.rs:1002).
- L-building: income "+N" score billboards (BUILDING_NEW_INCOME) not spawned; build stack ticks on DIP building; base map-cell m_Base ownership.
- L-weapon/fx: bomb early-trail smoke size 400 vs 300 (moving_object.rs:136); projectile trail/tracer frustum gating; muzzle/bolt point lights; splash angle randomness; plasma muzzle konus texture.
- L-effects: MAX_EFFECT_DISTANCE_SQ (3000²) culling absent; landscape-spot eviction not priority-aware.
- L-robot: VO DrawLights emissive; drowning MustDie; GetLost>70° early-out; add-order base-close; BASE_CAPTURE fire/cooldown.
- L-side: FirePL idle-repair is_live vs is_live_unit (repairs buildings); side name; side roster from objects.
- L-minimap: building/robot alert pings (yellow/green) not wired; frustum plane WATER_LEVEL vs z=0; corner clamp vs clip.
- L-flyer: altitude terrain-only Z vs GetZInterpolatedObjRobots.
- L-logic: maintenance-on sound; terron easter egg; final-win extra guards; takt time-advance phase lag.
- L-ui: buhe build-flyer button absent; auto-order ON/OFF toggle collapses to on (latent behind H3).

## Verified FAITHFUL (no action): water, road-network runtime, effect base/billboard/VO, most robot/cannon/building/side-order logic, counter, history, constructor prices, most minimap.
