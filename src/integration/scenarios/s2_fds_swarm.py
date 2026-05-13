"""Scenario S2 -- Drone swarm guided by FDS-truth RiskMap (D-025).

Same Scene and N :class:`PersonAgent` instances as
:mod:`src.integration.scenarios.s1_fixed_sign`, but the waypoint source
is :class:`~src.path_planning.planners.EvacuationPlanner` consuming the
same FDS-derived :class:`StaticRiskMap`. Drones replan every 30 s
(D-014) and waypoints are fed agent-by-agent.

**S1 vs S2 -> H6** ("dynamic guidance reduces FED >= 30 %"):
both scenarios share the same building, the same FDS truth RiskMap,
and the same starting positions for a given seed. The only difference
is *which waypoints the agents follow*: nearest-exit greedy (S1) vs
risk-weighted A* with periodic replans (S2). Any divergence in the
:class:`ScenarioMetrics` row is attributable to the guidance system,
not to physics, demographics or risk perception.

**M4-mini status (2026-05-14):**

* Skips the actual ``DroneSwarm`` body (Crazyflie URDFs, Boids/APF
  coordination) - that is M4-full work.
* Implements the *guidance logic* the drone swarm will eventually
  provide: per-agent weighted-A* paths from
  :class:`~src.path_planning.planners.EvacuationPlanner`, refreshed
  every ``replan_period_s`` (D-014: 30 s).
* No casualty / FED metrics (M2-full prerequisite, same as S1).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.integration.metrics import ScenarioMetrics
from src.integration.recorder import SimulationRecorder
from src.integration.scene import Scene, SceneConfig
from src.integration.scenarios._common import (
    BUILDING_URDF,
    PLACEHOLDER_URDF,
    building_urdf_path,
    exit_positions,
    load_truth_risk_map,
    spawn_agents,
)
from src.path_planning.building_graph import load_default_fluid_mask
from src.path_planning.edge_weights import EdgeWeightConfig
from src.path_planning.planners import EvacuationPlanner
from src.shared.constants import REPLAN_PERIOD_S


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
    del n_drones  # M4-mini: no bodies; accepted for forward compat.

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
        truth_rm = load_truth_risk_map(fds_dir, verbose=True)
        exits_xyz = exit_positions()
        agents = spawn_agents(scene, n_persons, seed, fluid_mask=fluid_mask)
        n_actual = len(agents)

        # Planner: risk_threshold=0.95 means only near-saturated risk
        # closes an edge -- generous so the M4-mini doesn't accidentally
        # turn into a topological no-path scenario.
        planner = EvacuationPlanner(
            _building_graph(),
            config=EdgeWeightConfig(
                base_cost=1.0,
                risk_scale=10.0,
                risk_threshold=0.95,
                n_samples=5,
            ),
            heuristic="euclidean",
            fallback_to_last=True,
        )

        # Per-agent path state
        paths: Dict[str, List[np.ndarray]] = {}
        wp_idx: Dict[str, int] = {}
        last_replan_t: Dict[str, float] = {a.agent_id: -math.inf for a in agents}

        arrived: Dict[str, float] = {}
        exposure_s: Dict[str, float] = {a.agent_id: 0.0 for a in agents}

        # Initial plan for every agent at t=0
        for a in agents:
            initial_path = planner.plan(a.position, truth_rm, t=0.0)
            paths[a.agent_id] = initial_path
            wp_idx[a.agent_id] = 1 if len(initial_path) > 1 else 0
            last_replan_t[a.agent_id] = 0.0

        max_steps = int(math.ceil(t_end_s / dt_s)) + 5
        WP_TOLERANCE_M = 1.0
        DANGER_THRESHOLD = 0.5

        for _ in range(max_steps):
            t_now = float(scene.t)
            if t_now >= t_end_s:
                break

            for agent in agents:
                if agent.agent_id in arrived:
                    continue

                # ── Replan if interval elapsed ──────────────────────
                if (t_now - last_replan_t[agent.agent_id]) >= replan_period_s:
                    new_path = planner.replan(agent.position, truth_rm, t=t_now)
                    if new_path:
                        paths[agent.agent_id] = new_path
                        # Skip index 0 (= agent's current position).
                        wp_idx[agent.agent_id] = 1 if len(new_path) > 1 else 0
                    last_replan_t[agent.agent_id] = t_now

                # ── Step toward current waypoint ────────────────────
                path = paths[agent.agent_id]
                idx = wp_idx[agent.agent_id]
                if path and idx < len(path):
                    wp = path[idx]
                    agent.step_toward(
                        scene.client,
                        wp,
                        dt_s,
                        fluid_mask=fluid_mask,
                    )
                    # Advance to next waypoint if reached.
                    if (
                        float(np.linalg.norm(agent.position[:2] - wp[:2]))
                        <= WP_TOLERANCE_M
                    ):
                        wp_idx[agent.agent_id] = idx + 1

                # ── Exposure (truth) ────────────────────────────────
                danger = float(truth_rm.query(agent.position, t=t_now))
                if danger > DANGER_THRESHOLD:
                    exposure_s[agent.agent_id] += float(dt_s)

                # ── Arrival vs any exit (XY) ────────────────────────
                for ex in exits_xyz:
                    if (
                        float(np.linalg.norm(agent.position[:2] - ex[:2]))
                        <= agent.config.arrival_tolerance_m
                    ):
                        arrived[agent.agent_id] = t_now
                        break

            # Optional recording (decoupled, no-op if recorder is None).
            if recorder is not None:
                recorder.record(
                    t=t_now,
                    agents=agents,
                    risk_map=truth_rm,
                    arrived=set(arrived.keys()),
                    agent_extras_fn=lambda a, _es=exposure_s: {
                        "exposure_s": _es.get(a.agent_id, 0.0),
                    },
                )
            scene.step()
            if len(arrived) == n_actual:
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

        return ScenarioMetrics(
            scenario_id="S2_fds_swarm",
            fire_scenario_id=fire_scenario_id,
            seed=seed,
            n_persons=n_actual,
            evacuation_success_rate=success_rate,
            mean_evacuation_time_s=mean_evac_time,
            danger_zone_exposure_time_s=mean_exposure,
            casualty_rate=0.0,   # M2-full prerequisite
            cumulative_fed=0.0,  # M3-full prerequisite (CO grid)
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
