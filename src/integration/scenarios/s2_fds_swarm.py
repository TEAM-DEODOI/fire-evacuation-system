"""Scenario S2 -- Drone swarm guided by FDS-truth RiskMap (D-025 / D-030).

Same Scene and N :class:`PersonAgent` instances as
:mod:`src.integration.scenarios.s1_fixed_sign`, but a small swarm of
:class:`~src.integration.drone_swarm.DroneAgent` searches the building
and shepherds survivors. Drones do **not** know occupant positions a
priori (D-030 user spec, 2026-05-14): they spawn at the 3 exits, patrol
via frontier-based coverage, and only acquire a target when one falls
within their ``sense_range_m``. The PersonAgent walks toward whichever
GUIDING drone is currently in its ``follow_range_m``.

The planner for the drone is :class:`EvacuationPlanner` consuming the
FDS-truth :class:`StaticRiskMap`. S3 reuses everything but swaps in a
model RiskMap for the planner.

**S1 vs S2 -> H6** ("dynamic guidance reduces FED >= 30 %"):
both scenarios share the same building and the same starting positions
for a given seed. The only difference is *who tells the agent where to
go*: a static greedy-nearest sign (S1) vs an actively searching drone
swarm (S2). Any divergence in the :class:`ScenarioMetrics` row is
attributable to the guidance system, not to physics or demographics.

**M4-mid status (2026-05-14):**

* Drone bodies are LOGICAL only -- the swarm advances kinematically via
  fluid-mask veto, identical to :class:`PersonAgent`. PyBullet Crazyflie
  bodies + visual rendering are M4-full work.
* Search + guide finite-state machine: each drone is SEARCHING or
  GUIDING; transitions on `sense_range`-based detection.
* PersonAgent follows the nearest GUIDING drone in `follow_range_m`.
  When no drone is in sight the agent stays put (no fixed-sign fallback).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.integration.drone_swarm import DroneSwarm, DroneSwarmConfig, DroneAgentConfig
from src.integration.metrics import ScenarioMetrics
from src.integration.recorder import SimulationRecorder
from src.integration.scene import Scene, SceneConfig
from src.integration.person_agent import PersonStatus
from src.integration.scenarios._common import (
    BUILDING_URDF,
    PLACEHOLDER_URDF,
    building_urdf_path,
    exit_positions,
    load_truth_co_field,
    load_truth_risk_map,
    spawn_agents,
)
from src.path_planning.building_graph import (
    load_default_fluid_mask,
    load_interior_mask,
)
from src.path_planning.edge_weights import EdgeWeightConfig
from src.path_planning.planners import EvacuationPlanner
from src.shared.constants import CELL_SIZE_M, REPLAN_PERIOD_S


# ─── Run loop ─────────────────────────────────────────────────────────────
def run(
    fire_scenario_id: str,
    fds_dir: Path,
    n_persons: int = 20,
    n_drones: int = 3,        # accepted for forward compat; M4-mini has no bodies
    seed: int = 0,
    t_end_s: float = 300.0,
    dt_s: float = 1.0,
    replan_period_s: float = REPLAN_PERIOD_S,
    t_start_s: float = 0.0,
    *,
    recorder: Optional[SimulationRecorder] = None,
) -> ScenarioMetrics:
    """Execute one S2 run.

    Per-tick logic for each ALIVE agent:

    1. If ``t_now - last_replan_t >= replan_period_s`` (or no path yet):
       call ``planner.replan(agent.position, truth_rm, t=t_now)`` and
       reset ``wp_idx = 1`` (planner prepends the start position as
       index 0).
    2. Look up current target waypoint ``paths[agent_id][wp_idx]``;
       skip the agent if path is empty.
    3. ``step_toward`` the current waypoint.
    4. If within 1.0 m of the current waypoint, advance ``wp_idx``.
    5. Accumulate ``dt_s`` to ``exposure_s[agent_id]`` if local
       ``truth_rm.query`` > 0.5.
    6. Mark arrived if within ``arrival_tolerance_m`` of any exit.

    Args:
        fire_scenario_id: FDS scenario label, recorded in the returned
            row.
        fds_dir: Path to the FDS scenario directory. Loaded via
            :func:`load_truth_risk_map` (cached) or zero-risk fallback.
        n_persons: Maximum number of PersonAgents to spawn. Capped at
            10 (number of room nodes). D-025 default 20.
        n_drones: Accepted for forward compatibility with the M4-full
            DroneSwarm class. M4-mini does not spawn drone bodies.
        seed: RNG seed for the start-position scatter.
        t_end_s: Maximum simulation time (s).
        dt_s: Outer-loop step (s).
        replan_period_s: Cadence at which each agent's path is
            recomputed via the planner. Default ``REPLAN_PERIOD_S``
            (= 30 s, D-014).

    Returns:
        :class:`ScenarioMetrics` with ``scenario_id="S2_fds_swarm"``.

    Raises:
        FileNotFoundError: If ``assets/placeholder_building.urdf`` is
            missing (run ``python -m src.integration.urdf_builder``).
    """
    urdf = building_urdf_path()
    if not urdf.exists():
        raise FileNotFoundError(
            f"No building URDF on disk. Tried {BUILDING_URDF} (real STL) "
            f"and {PLACEHOLDER_URDF} (fallback). "
            f"Generate one: python -m src.integration.urdf_builder"
        )

    cfg = SceneConfig(
        connection_mode="DIRECT",
        building_urdf=urdf,
        dt_s=dt_s,
        draw_origin_axes=False,
    )
    scene = Scene.create(cfg)
    try:
        fluid_mask = load_default_fluid_mask()
        interior_mask = load_interior_mask()
        truth_rm = load_truth_risk_map(fds_dir, verbose=True)
        truth_co = load_truth_co_field(fds_dir, verbose=True)
        exits_xyz = exit_positions()
        agents = spawn_agents(
            scene, n_persons, seed,
            fluid_mask=fluid_mask,
            interior_mask=interior_mask,
        )
        n_actual = len(agents)

        # Planner: risk_threshold=0.95 -- only near-saturated risk closes
        # an edge so the planner does not pessimistically refuse paths in
        # smoke-filled corridors.
        # D-040 (2026-05-14): pure risk-based edge weighting.
        # weight = base_cost · length + risk_scale · edge_risk
        # Uses ONLY planner_rm (each scenario's own model). No CO field,
        # no proxy. The planner minimises cumulative risk along the path.
        planner = EvacuationPlanner(
            _building_graph(),
            config=EdgeWeightConfig(
                base_cost=1.0,
                risk_scale=10.0,
                risk_threshold=1.0,  # no hard cutoff
                n_samples=5,
            ),
            heuristic="euclidean",
            fallback_to_last=True,
        )

        # Drone swarm: one drone per exit (D-030 default).
        swarm = DroneSwarm(
            config=DroneSwarmConfig(
                n_drones=int(n_drones),
                drone=DroneAgentConfig(
                    replan_period_s=float(replan_period_s),
                ),
            ),
        )
        # Spawn at the exits (cycling so n_drones != 3 still works).
        spawn_seeds = [
            exits_xyz[i % len(exits_xyz)].copy() for i in range(int(n_drones))
        ]
        swarm.spawn(spawn_xyzs=spawn_seeds, interior_mask=interior_mask)

        arrived: Dict[str, float] = {}
        died_at: Dict[str, float] = {}
        exposure_s: Dict[str, float] = {a.agent_id: 0.0 for a in agents}

        # D-034: t_sim (sim wall clock, 0..duration) vs t_fire (fire's
        # absolute time, t_sim + t_start_s). Risk + CO queries use
        # t_fire; per-agent timing uses t_sim.
        duration_s = max(0.0, float(t_end_s) - float(t_start_s))
        max_steps = int(math.ceil(duration_s / dt_s)) + 5
        DANGER_THRESHOLD = 0.5

        for _ in range(max_steps):
            t_sim = float(scene.t)
            if t_sim >= duration_s:
                break
            t_fire = t_sim + float(t_start_s)

            # ── 1. Advance the swarm (search / acquire / guide).
            #      D-040: risk-only planner — no co_field passed.
            swarm.update(
                t=t_fire,
                dt_s=float(dt_s),
                persons=agents,
                planner_rm=truth_rm,
                planner=planner,
                interior_mask=interior_mask,
                exits=exits_xyz,
            )

            # ── 2. Each ALIVE person scans for a visible GUIDING drone
            #      and walks toward it via self path planning (D-032).
            #      Fallback (D-037, 2026-05-14): when no drone is in
            #      sight, behave like S1 — head to the risk-blind
            #      nearest exit. Occupant never just stands still.
            for agent in agents:
                if agent.status != PersonStatus.ALIVE:
                    continue
                target_drone = agent.scan_for_drones(swarm.drones)
                if target_drone is not None:
                    waypoint = np.array([
                        target_drone.position[0],
                        target_drone.position[1],
                        agent.position[2],
                    ], dtype=np.float64)
                else:
                    # S1-equivalent fallback: nearest exit by XY Euclidean.
                    waypoint = min(
                        exits_xyz,
                        key=lambda ex, _p=agent.position: float(
                            np.linalg.norm(np.asarray(ex)[:2] - _p[:2])
                        ),
                    )
                    waypoint = np.array([
                        waypoint[0], waypoint[1], agent.position[2],
                    ], dtype=np.float64)
                tgt_cell = (
                    int(waypoint[0] / CELL_SIZE_M),
                    int(waypoint[1] / CELL_SIZE_M),
                )
                if (
                    not agent.nav_path
                    or agent.last_path_target_cell != tgt_cell
                    or (t_sim - agent.last_path_replan_t)
                    >= agent.config.path_replan_period_s
                ):
                    agent.replan_path_to(waypoint, fluid_mask, t=t_sim)
                agent.step_along_path(scene.client, dt_s, fluid_mask)

                # ── 3. Truth-sampled danger + CO (fire clock) ───────
                danger = float(truth_rm.query(agent.position, t=t_fire))
                co_ppm = float(truth_co.query(agent.position, t=t_fire))
                if danger > DANGER_THRESHOLD:
                    exposure_s[agent.agent_id] += float(dt_s)

                # ── 4. FED accumulation + DEAD transition ───────────
                died = agent.accumulate_exposure(
                    experienced_co_ppm=co_ppm,
                    experienced_danger=danger,
                    dt_s=float(dt_s),
                )
                if died:
                    died_at[agent.agent_id] = t_sim
                    continue

                # ── 5. Arrival vs any exit (XY) ─────────────────────
                for ex in exits_xyz:
                    if (
                        float(np.linalg.norm(agent.position[:2] - ex[:2]))
                        <= agent.config.arrival_tolerance_m
                    ):
                        arrived[agent.agent_id] = t_sim
                        agent.mark_evacuated()
                        break

            # Optional recording — stamp with t_fire so renderer reads
            # the correct risk-map slice.
            if recorder is not None:
                recorder.record(
                    t=t_fire,
                    agents=agents,
                    risk_map=truth_rm,
                    arrived=set(arrived.keys()),
                    drones=swarm.drones,
                    agent_extras_fn=lambda a, _es=exposure_s: {
                        "exposure_s": _es.get(a.agent_id, 0.0),
                        "cumulative_fed": float(a.cumulative_fed),
                        "following_drone_id": a.following_drone_id,
                    },
                )
            scene.step()
            if len(arrived) + len(died_at) == n_actual:
                break

        # ── Aggregate ─────────────────────────────────────────────
        success_rate = (len(arrived) / n_actual) if n_actual else 0.0
        if arrived:
            mean_evac_time = float(np.mean(list(arrived.values())))
        else:
            mean_evac_time = float("nan")
        mean_exposure = (
            float(np.mean(list(exposure_s.values()))) if exposure_s else 0.0
        )
        casualty_rate = (len(died_at) / n_actual) if n_actual else 0.0
        feds = np.asarray([a.cumulative_fed for a in agents], dtype=np.float64)
        mean_fed = float(feds.mean()) if feds.size else 0.0
        max_fed = float(feds.max()) if feds.size else 0.0
        p90_fed = float(np.percentile(feds, 90)) if feds.size else 0.0

        return ScenarioMetrics(
            scenario_id="S2_fds_swarm",
            fire_scenario_id=fire_scenario_id,
            seed=seed,
            n_persons=n_actual,
            evacuation_success_rate=success_rate,
            mean_evacuation_time_s=mean_evac_time,
            danger_zone_exposure_time_s=mean_exposure,
            casualty_rate=casualty_rate,
            cumulative_fed=mean_fed,
            max_cumulative_fed=max_fed,
            p90_cumulative_fed=p90_fed,
        )
    finally:
        scene.close()


# ─── Local graph cache (avoids re-importing on every call) ────────────────
_GRAPH_CACHE = None


def _building_graph():
    """Memoised access to the canonical building graph.

    :class:`EvacuationPlanner` reads the graph topology once at
    construction. Cache the result so back-to-back ``run()`` calls
    (typical in EXP-PATH-001 sweeps) skip the NetworkX rebuild.
    """
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        from src.path_planning.building_graph import build_graph
        _GRAPH_CACHE = build_graph()
    return _GRAPH_CACHE


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("s2_fds_swarm.py self-test (M4-mini)")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. Zero-risk fallback smoke ────────────────────────────────
    print("\n[1] run() smoke: 5 agents, 60 s, no FDS (fallback to zero risk)")
    res_zero = run(
        fire_scenario_id="m4_mini_smoke",
        fds_dir=Path("data/raw/__no_such_scenario__"),
        n_persons=5,
        seed=0,
        t_end_s=60.0,
        dt_s=1.0,
    )
    print(f"  {res_zero.summary_line()}")
    if res_zero.scenario_id != "S2_fds_swarm":
        errors.append(f"scenario_id wrong: {res_zero.scenario_id}")
    if res_zero.n_persons != 5:
        errors.append(f"n_persons {res_zero.n_persons} != 5")
    if res_zero.danger_zone_exposure_time_s != 0.0:
        errors.append(
            f"zero-risk exposure should be 0, got {res_zero.danger_zone_exposure_time_s}"
        )
    if res_zero.casualty_rate != 0.0:
        errors.append(f"casualty != 0: {res_zero.casualty_rate}")
    if res_zero.cumulative_fed != 0.0:
        errors.append(f"fed != 0: {res_zero.cumulative_fed}")

    # ── 2. Real FDS run + back-to-back S1 vs S2 numeric comparison ────
    real_fds = Path("data/raw/sim_1500kw_2m2_T05")
    if real_fds.exists():
        print(
            "\n[2] S1 vs S2 on real FDS (sim_1500kw_2m2_T05), "
            "5 agents, seed=0, t_end=200 s"
        )
        # S2 first
        res_s2 = run(
            fire_scenario_id="sim_1500kw_2m2_T05",
            fds_dir=real_fds,
            n_persons=5,
            seed=0,
            t_end_s=200.0,
            dt_s=1.0,
            replan_period_s=30.0,
        )
        print(f"  S2: {res_s2.summary_line()}")

        from src.integration.scenarios.s1_fixed_sign import run as s1_run
        res_s1 = s1_run(
            fire_scenario_id="sim_1500kw_2m2_T05",
            fds_dir=real_fds,
            n_persons=5,
            seed=0,
            t_end_s=200.0,
            dt_s=1.0,
        )
        print(f"  S1: {res_s1.summary_line()}")

        # Diff summary
        d_evac = res_s2.evacuation_success_rate - res_s1.evacuation_success_rate
        d_exposure = (
            res_s2.danger_zone_exposure_time_s
            - res_s1.danger_zone_exposure_time_s
        )
        print(
            f"\n  diff (S2 - S1):  evac={d_evac*100:+.1f}%p  "
            f"exposure={d_exposure:+.2f}s"
        )
        if not math.isnan(res_s1.mean_evacuation_time_s) and not math.isnan(
            res_s2.mean_evacuation_time_s
        ):
            d_t = res_s2.mean_evacuation_time_s - res_s1.mean_evacuation_time_s
            print(f"                  mean_t_evac={d_t:+.2f}s")
        if res_s2.scenario_id != "S2_fds_swarm":
            errors.append("S2 scenario_id wrong")
        if res_s1.scenario_id != "S1_fixed_sign":
            errors.append("S1 scenario_id wrong")
        # Sanity: same population size
        if res_s1.n_persons != res_s2.n_persons:
            errors.append(
                f"S1/S2 n_persons differ: {res_s1.n_persons} vs {res_s2.n_persons}"
            )
    else:
        print(f"\n[2] SKIP: real FDS scenario {real_fds} not on disk")

    # ── 3. CSV round-trip ─────────────────────────────────────────
    print("\n[3] ScenarioMetrics.to_dict round-trip")
    d = res_zero.to_dict()
    if d["scenario_id"] != "S2_fds_swarm":
        errors.append("to_dict lost scenario_id")
    print(f"  keys = {sorted(d)}")

    # ── Verdict ────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: s2_fds_swarm M4-mini validated")
