# Goal: port enemy AI to Rust (MatrixSide.cpp AI slice)

Source: MatrixGame/src/MatrixSide.cpp. The Logic/MatrixTactics/Rule/State
classes are DEAD CODE in C++ (only commented-out call sites) — the real AI is:
LogicTakt (non-player branch) → TaktHL (100ms) + TaktTL (10ms), plus
BuildRobot/BuildCannon, CalcStrength, Regroup, EscapeFromBomb(shared, already
ported for player), team logic (WarTL/RepairTL/AssignPlace/SortRobotList/
PlaceInRegion/CalcRegionPath/CanMoveNoEnemy/FindNearRegionWithUTR/
CompareAction/BestAction/LiveAction), GroupNoTeamRobot (map prepare),
ClacSpawnTeam (robot spawn/path-fail).

C++ line map (MatrixSide.cpp): CalcStrength 1810, Regroup 1844, ClearTeam 2013,
ClacSpawnTeam 2027, EscapeFromBomb 2090, GroupNoTeamRobot 2257, CalcMaxSpeed
2343 (AI branch missing in Rust), TaktHL 2450-4014, FindNearRegionWithUTR 4016,
CompareAction 4070, BestAction 4196, LiveAction 4210, TaktTL 4218, WarTL 4633,
RepairTL 5217, AssignPlace(robot) 5270, AssignPlace(group) 5303, SortRobotList
5457, PlaceInRegion 5539, BuildRobotMinStrange 5582, BuildRobot 5609,
BuildCannon 5914, CalcRegionPath 9550, CanMoveNoEnemy 9588.
GetMaxSideRobots 1653 (= compute_max_side_robots in logic.rs:1746 DONE).
SideAIInfo map options: MatrixMapPrepare.cpp:410-461 (TBB/SK/DK/WRK/BK/TC via
da/Side name→id block). ProduceRobot team assign: MatrixRobot.cpp:2204-2205
(ClacSpawnTeam). ZonePathCalc fail→reteam: MatrixRobot.cpp:1601-1604. Team
road path used in path calc: MatrixRobot.cpp:1590-1591.

Existing Rust infrastructure (verified):
- side.rs: Side struct (player fields), LogicRegion (needs war counters +
  dist fields added), Group, PlayerGroup.
- side_player.rs: escape_from_bomb (player-hardwired — generalize),
  underfire_calc (generalize by side id), war_pl/repair_pl (models for
  WarTL/RepairTL), assign_place_robot (=AssignPlace(robot)) player-hardwired,
  pg_calc_region_path, group_robots/side_robots helpers (player-hardwired).
- constructor.rs: SpecialBot + AIRobotCatalogue + global_ai_robots() loaded at
  startup (logic.rs:1032). dif_weapon, calc_strength, to_robot_config done.
- road_network.rs: full port (regions w/ near/near_move/center/place/place_all,
  places w/ move_mask/data/underfire/region, find_path_in_region_run,
  find_path_from_region_path, is_nerest_region, find_in_pl).
- logic.rs: get_region(map,x,y), is_live_unit, point_of_aim, place_list,
  place_list_grow, find_near_place, compute_max_side_robots, Rnd.range
  (inclusive), gather_info (ALL sides — verified robots of any side sense).
- object_building.rs: build_stack (items, queue_robot, queue_turret_slot,
  turrets_have/max, turret_places), production tick at ~line 300 sets team=0
  (needs AI ClacSpawnTeam hook via objects.pending list).
- robot.rs: move_to/move_return/move_to_high/capture_factory/fire/stop_fire,
  have_bomb/have_repair, max_fire_dist/min_fire_dist/repair_dist fields,
  team/group_logic, group_road_path: Option<Arc<RoadRoute>> (used by
  zone_path_calc; PG layer sets it at side_player.rs:3588 — mirror for teams),
  calc_strength. env: Info (place, place_add, target_attack, target,
  target_last, target_change, target_change_repair, target_angle, last_fire,
  last_hit_target, order_no_break, bad places, place_not_found).
- object_cannon.rs: get_strength, fire_radius.
- logic.rs takt loop line 384-390: only player side_logic_takt — add AI sides.

## Work list
- [x] 1. side.rs: LogicActionType/LogicAction/LogicGroup/SideTeam structs;
        Side AI fields (time_next_bomb=60000 rel→abs at load, time_last_bomb=0,
        strength_mul=1, brave_mul=0.5, danger_mul=1, wait_res_mul=1, team_cnt=3,
        strength, war_side=-1, last_takt_hl/tl=0, last_team_change=0,
        next_war_side_calc_time=0, wait_res_for_build_robot=-1,
        build_robot_last/2/3=-1, logic_groups, teams); extend LogicRegion.
- [x] 2. map.rs: parse SideAIInfo property (name:kv/kv|...) → side_ai_info;
        logic.rs apply via da/Side name→id (storage block) in apply_side_resources.
- [x] 3. side_ai.rs: full port of the AI functions (impl MapLogic, side taken
        out via mem::take pattern; sub-fns take &mut Side).
- [x] 4. Wiring: logic.rs takt loop → ai_side_logic_takt per other_side
        (maintenance for AI handled by C++ only for m_Id==PLAYER_SIDE in auto
        mode — skip); GroupNoTeamRobot after map load; production team assign
        (objects.pending_ai_spawn drained in MapLogic takt → clac_spawn_team);
        zone-path-fail reteam (robot pushes to objects.pending_ai_reteam);
        team road path set on robots when TaktTL issues orders; AI calc_max_speed.
- [x] 5. cargo build + cargo test pass. (native build ok, 236 tests pass)
- [x] 6. wasm build (release ~35s + pack_bundle unchanged) + bump index.html ?v=.
        (wasm-pack dev build OK; index.html v=121→122)

## Verification status
COMPLETE:
- native build clean, no warnings; all unit tests pass (210 lib tests,
  incl. 3 new AI tests: ai_side_builds_robot_when_affordable,
  clac_spawn_team_prefers_empty_team, ai_takt_assigns_teams_and_actions).
- NEW real-map integration test tests/test_ai.rs (enemy_ai_plays_on_real_map):
  loads robots.pkg ATOLL.CMAP + robots.dat, runs 4 simulated minutes of the
  full logic loop → all 3 AI sides build robots (11 total), form teams with
  Forward/Defence/Attack actions, and pick war targets. PASSES.
- Found+fixed pre-existing latent bug via that test: AIRobotCatalogue read
  block "da/AI/Robots" but robots.dat stores the templates in the root
  "AIRobotType" block (MatrixGame.cpp:412) → catalogue was empty (159
  templates now load).
- wasm-pack dev build OK; index.html bumped to ?v=298; server on 8081.

## Key decisions
- side_ai.rs uses mem::take(&mut self.other_sides[idx]) at ai_side_logic_takt
  entry, passes &mut Side down; player-side funcs untouched.
- underfire_calc + escape_from_bomb generalized with side id param
  (escape_from_bomb(map) → escape_from_bomb_side(map, sid); manual-order guard
  only applies when sid == player).
- BuildRobot deducts side resources directly (C++ parity), then
  base.queue_robot(bot.to_robot_config()); team assigned at production time
  via objects.pending_ai_team (Vec<(ObjectId, ObjectId)> robot+base) drained
  by MapLogic::drain_ai_spawn_teams → clac_spawn_team.
- BuildCannon picks random free slot from turret_places (slot cannon_type<0),
  queue_turret_slot(slot, kind+1... kind index parity: C++ m_Num=curtype+1,
  Rust turret_kind 1..=4 — verify mapping when writing).
- Robot team roadpath: TaktTL move orders set r.group_road_path =
  Some(Arc(team.road_path.clone())) mirroring side_player.rs:3588.
