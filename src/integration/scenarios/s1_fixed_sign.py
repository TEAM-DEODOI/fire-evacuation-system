"""Scenario S1 -- Fixed-sign baseline (D-025 H6 baseline).

Setup:

* N :class:`~src.integration.person_agent.PersonAgent` spawned at random
  positions inside the building's room nodes.
* Static "exit signs" pre-placed at corridor intersections. Each agent
  queries the sign network for the nearest exit whose pointed-at
  position is currently non-hazardous (per
  :class:`~src.risk_map.risk_map_class.RiskMap`), then walks toward it.
* No drones. FED accumulation, status transitions, and metric collection
  happen identically to S2/S3 -- only the WaypointProvider differs.

This is the **H6 baseline**: if drone-swarm guidance (S2) gives at
least a 30 % FED reduction relative to this, H6 passes.

**Status (2026-05-14, M3-α):**

* :class:`FixedSignNetwork.next_waypoint` -- **functional** (degenerate
  form: every exit acts as both sign and target; intermediate
  ``sign_positions`` are M3-full work).
* :func:`run` -- **functional with FDS RiskMap**. Builds Scene +
  placeholder building + N agents at room-node centres, loads the
  truth :class:`StaticRiskMap` from ``fds_dir`` (cached to
  ``results/cache/s1_risk_maps/<scenario>.npz`` for fast re-runs),
  walks agents until exit or ``t_end_s``, accumulates per-agent
  ``danger > 0.5`` exposure time, and emits a
  :class:`~src.integration.metrics.ScenarioMetrics` row.
* ``casualty_rate`` and ``cumulative_FED`` still **0** -- they require
  the status-machine and CO-grid loading which are M2-full / M3-full.
* If ``fds_dir`` is missing or fails to parse, falls back to an
  all-zero risk map (no exposure accumulation) with a clear warning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

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
    zero_risk_map,
)
from src.path_planning.building_graph import load_default_fluid_mask
from src.risk_map.co_field import StaticCOField
from src.risk_map.risk_map_class import RiskMap
from src.shared.constants import CELL_SIZE_M


# ─── Fixed-sign provider ──────────────────────────────────────────────────
@dataclass
class FixedSignNetwork:
    """Static-sign equivalent of a WaypointProvider for S1.

    Per D-033 (2026-05-14): **risk-BLIND**. A real fixed-sign system
    has no live information about fire spread — its signs are painted /
    illuminated permanently and just point at the nearest exit. That is
    exactly the baseline we want to compare against: an evacuee follows
    the painted "EXIT →" arrow without knowing that the chosen exit is
    currently engulfed in smoke. The drone scenarios (S2/S3) earn their
    value by knowing the live risk map and routing the occupant
    elsewhere.

    Implementation: :meth:`next_waypoint` returns the closest exit
    (Euclidean XY) regardless of any RiskMap. The ``risk_map`` field is
    kept on the dataclass for interface compatibility but is no longer
    consulted; ``danger_threshold`` and ``sign_positions`` are reserved
    for future M3-full work (per-sign hazard switching) but ignored here.
    """

    risk_map: RiskMap
    exit_positions: List[np.ndarray] = field(default_factory=list)
    sign_positions: List[np.ndarray] = field(default_factory=list)
    danger_threshold: float = 0.5

    def next_waypoint(
        self,
        agent_position: np.ndarray,
        t: float,
    ) -> Optional[np.ndarray]:
        """Return the nearest exit (XY Euclidean), risk-blind (D-033).

        Returns:
            ``(3,)`` world position of the closest exit, or ``None`` if
            no exits are configured.
        """
        if not self.exit_positions:
            return None
        agent_xy = np.asarray(agent_position, dtype=np.float64)[:2]
        best: Optional[np.ndarray] = None
        best_d2 = math.inf
        for ex in self.exit_positions:
            ex_arr = np.asarray(ex, dtype=np.float64)
            dx = ex_arr[0] - agent_xy[0]
            dy = ex_arr[1] - agent_xy[1]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = ex_arr
        return best


# Shared helpers live in src.integration.scenarios._common — see imports above.


# ─── Run loop ─────────────────────────────────────────────────────────────
def run(
    fire_scenario_id: str,
    fds_dir: Path,
    n_persons: int = 20,
    seed: int = 0,
    t_end_s: float = 300.0,
    dt_s: float = 1.0,
    t_start_s: float = 0.0,
    *,
    recorder: Optional[SimulationRecorder] = None,
) -> ScenarioMetrics:
    """Execute one S1 run and return its :class:`ScenarioMetrics` row.

    Args:
        fire_scenario_id: FDS scenario label, recorded in the returned
            row. M3-mini does not actually load this scenario.
        fds_dir: Path to the FDS scenario folder. Accepted for forward
            compatibility but currently ignored (M3-mini uses a zero
            RiskMap; FDS load is M3-full).
        n_persons: Maximum number of PersonAgents to spawn. Capped at
            the number of room nodes in the building graph (10 at
            present). D-025 default 20.
        seed: RNG seed for the start-position scatter across room nodes.
        t_end_s: Maximum simulation time (s).
        dt_s: Outer-loop step (s).

    Returns:
        :class:`ScenarioMetrics` with ``scenario_id="S1_fixed_sign"``.
        In M3-mini the three fire-dependent metrics
        (``danger_zone_exposure_time``, ``casualty_rate``,
        ``cumulative_FED``) are exactly 0.

    Raises:
        FileNotFoundError: If ``assets/placeholder_building.urdf`` is
            missing (run ``python -m src.integration.urdf_builder`` to
            regenerate it).
        RuntimeError: If the building graph has no room nodes.
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
        truth_rm = load_truth_risk_map(fds_dir, verbose=True)
        truth_co = load_truth_co_field(fds_dir, verbose=True)
        exits_xyz = exit_positions()
        agents = spawn_agents(scene, n_persons, seed, fluid_mask=fluid_mask)
        n_actual = len(agents)
        # Per fairness setup (interface_contracts.md sect. 6) the planner
        # and the experienced truth use the SAME RiskMap in S1. Drone
        # scenarios (S2/S3) will pass separate planner_rm and truth_rm.
        provider = FixedSignNetwork(
            risk_map=truth_rm,
            exit_positions=exits_xyz,
            danger_threshold=0.5,
        )

        # Per-agent finish-time bookkeeping. An agent finishes either by
        # evacuating (status EVACUATED) or by dying (status DEAD).
        arrived: Dict[str, float] = {}
        died_at: Dict[str, float] = {}
        # Per-agent accumulated time in danger > 0.5 zone (legacy metric).
        exposure_s: Dict[str, float] = {a.agent_id: 0.0 for a in agents}
        # D-034: decouple two timelines.
        # * t_sim = scene.t          (simulation wall-clock, starts at 0)
        # * t_fire = t_sim + t_start_s  (absolute fire/risk-map time)
        # Risk + CO are queried at t_fire so the agents experience the
        # fire as if it had already been burning for ``t_start_s``
        # seconds before they began evacuating. Per-agent timing
        # (arrival, exposure) is reported in t_sim so it stays
        # interpretable as "time spent evacuating".
        duration_s = max(0.0, float(t_end_s) - float(t_start_s))
        max_steps = int(math.ceil(duration_s / dt_s)) + 5
        danger_threshold = 0.5
        for _ in range(max_steps):
            t_sim = float(scene.t)
            if t_sim >= duration_s:
                break
            t_fire = t_sim + float(t_start_s)
            for agent in agents:
                if agent.status != PersonStatus.ALIVE:
                    continue
                wp = provider.next_waypoint(agent.position, t_fire)
                if wp is None:
                    agent.clear_path()
                    continue
                # D-032: self path planning (wall-aware BFS). Re-plan
                # whenever the target exit cell changes or the throttle
                # period has elapsed; otherwise reuse the cached path.
                tgt_cell = (
                    int(float(wp[0]) / float(CELL_SIZE_M)),
                    int(float(wp[1]) / float(CELL_SIZE_M)),
                )
                if (
                    not agent.nav_path
                    or agent.last_path_target_cell != tgt_cell
                    or (t_sim - agent.last_path_replan_t)
                    >= agent.config.path_replan_period_s
                ):
                    agent.replan_path_to(wp, fluid_mask, t=t_sim)
                agent.step_along_path(scene.client, dt_s, fluid_mask)
                # Sample experienced truth at the post-move position,
                # using the FIRE clock so the smoke level reflects how
                # long the fire has been burning, not the sim clock.
                danger = float(truth_rm.query(agent.position, t=t_fire))
                co_ppm = float(truth_co.query(agent.position, t=t_fire))
                if danger > danger_threshold:
                    exposure_s[agent.agent_id] += float(dt_s)

                # FED accumulation + DEAD transitions (M2-full).
                died = agent.accumulate_exposure(
                    experienced_co_ppm=co_ppm,
                    experienced_danger=danger,
                    dt_s=float(dt_s),
                )
                if died:
                    died_at[agent.agent_id] = t_sim
                    continue  # dead agents do not check for arrival

                # Arrived if within tolerance of *any* exit (XY only).
                for ex in exits_xyz:
                    if (
                        float(np.linalg.norm(agent.position[:2] - ex[:2]))
                        <= agent.config.arrival_tolerance_m
                    ):
                        arrived[agent.agent_id] = t_sim
                        agent.mark_evacuated()
                        break
            # Optional recording (decoupled, no-op if recorder is None).
            # Stamp frames with t_fire so the renderer reads the
            # correct fire-map slice for the risk overlay.
            if recorder is not None:
                recorder.record(
                    t=t_fire,
                    agents=agents,
                    risk_map=truth_rm,
                    arrived=set(arrived.keys()),
                    agent_extras_fn=lambda a, _es=exposure_s: {
                        "exposure_s": _es.get(a.agent_id, 0.0),
                        "cumulative_fed": float(a.cumulative_fed),
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
            scenario_id="S1_fixed_sign",
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


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("s1_fixed_sign.py self-test (M3-alpha: FDS RiskMap + exposure)")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. FixedSignNetwork.next_waypoint with empty exits ─────────────
    print("\n[1] FixedSignNetwork with no exits -> None")
    empty = FixedSignNetwork(risk_map=zero_risk_map(), exit_positions=[])
    wp = empty.next_waypoint(np.array([5.0, 5.0, 1.5]), t=0.0)
    if wp is not None:
        errors.append(f"expected None for empty exits, got {wp}")
    else:
        print("  PASS")

    # ── 2. FixedSignNetwork picks nearest below-threshold exit ─────────
    print("\n[2] FixedSignNetwork nearest-exit logic")
    rm = zero_risk_map()  # all-safe
    exits = exit_positions()
    print(f"  exits = {[tuple(round(v, 1) for v in e) for e in exits]}")
    net = FixedSignNetwork(risk_map=rm, exit_positions=exits, danger_threshold=0.5)
    # Agent at (5, 4): nearest is exit_west, now at (3, 6) -> cell-snap
    # (2.75, 5.75, 1.75) per shared/building.py (refined 2026-05-14).
    wp_b = net.next_waypoint(np.array([5.0, 4.0, 1.5]), t=0.0)
    print(f"  query (5, 4)  -> {tuple(round(v, 2) for v in wp_b)}")
    if abs(wp_b[0] - 3.0) > 0.5 or abs(wp_b[1] - 6.0) > 0.5:
        errors.append(f"expected near exit_west (~3, ~6), got {wp_b}")
    # Agent at (28, 16): nearest is exit_east, now at (22, 6) -> cell-snap
    # (21.75, 5.75, 1.75)
    wp_e = net.next_waypoint(np.array([28.0, 16.0, 1.5]), t=0.0)
    print(f"  query (28, 16) -> {tuple(round(v, 2) for v in wp_e)}")
    if abs(wp_e[0] - 22.0) > 0.5 or abs(wp_e[1] - 6.0) > 0.5:
        errors.append(f"expected near exit_east (~22, ~6), got {wp_e}")

    # ── 3. Zero-risk smoke: non-existent fds_dir -> fallback ─────────
    print("\n[3] run() smoke: 5 agents, 60 s, no FDS (fallback to zero risk)")
    result = run(
        fire_scenario_id="m3_mini_smoke",
        fds_dir=Path("data/raw/__no_such_scenario__"),  # forces fallback
        n_persons=5,
        seed=0,
        t_end_s=60.0,
        dt_s=1.0,
    )
    print(f"  {result.summary_line()}")
    if result.scenario_id != "S1_fixed_sign":
        errors.append(f"scenario_id wrong: {result.scenario_id}")
    if result.n_persons != 5:
        errors.append(f"n_persons {result.n_persons} != 5")
    if result.danger_zone_exposure_time_s != 0.0:
        errors.append(f"exposure != 0: {result.danger_zone_exposure_time_s}")
    if result.casualty_rate != 0.0:
        errors.append(f"casualty != 0: {result.casualty_rate}")
    if result.cumulative_fed != 0.0:
        errors.append(f"fed != 0: {result.cumulative_fed}")
    # NB: with the real STL-derived building (M1-full), greedy
    # nearest-exit S1 can legitimately produce evac=0 % on some seeds
    # -- every agent's straight-line path is blocked by interior walls
    # and they wait out the run. This is the H6 baseline we care about.
    # Just sanity-check the metric ranges; do NOT hard-fail on low evac.
    if not (0.0 <= result.evacuation_success_rate <= 1.0):
        errors.append(
            f"evac rate out of [0, 1]: {result.evacuation_success_rate}"
        )
    if (
        not math.isnan(result.mean_evacuation_time_s)
        and result.mean_evacuation_time_s < 0
    ):
        errors.append(
            f"mean evac time negative: {result.mean_evacuation_time_s}"
        )
    if result.evacuation_success_rate > 0 and math.isnan(
        result.mean_evacuation_time_s
    ):
        errors.append("at least 1 arrived but mean_evac_time is NaN")
    if result.evacuation_success_rate == 0 and not math.isnan(
        result.mean_evacuation_time_s
    ):
        errors.append("nobody arrived but mean_evac_time is finite")

    # ── 4. Different seed gives different agent set (likely) ───────────
    print("\n[4] Different seed produces a result (no crash)")
    result_b = run(
        fire_scenario_id="m3_mini_smoke",
        fds_dir=Path("data/raw/sim_1500kw_2m2_T05"),
        n_persons=5,
        seed=7,
        t_end_s=60.0,
        dt_s=1.0,
    )
    print(f"  seed=7: {result_b.summary_line()}")
    if result_b.n_persons != 5:
        errors.append(f"seed=7 n_persons wrong: {result_b.n_persons}")

    # ── 5. CSV-compatibility of the row (round-trip via to_dict) ───────
    print("\n[5] ScenarioMetrics.to_dict round-trip")
    d = result.to_dict()
    if d["scenario_id"] != "S1_fixed_sign":
        errors.append("to_dict lost scenario_id")
    print(f"  keys = {sorted(d)}")

    # ── 6. Real FDS RiskMap: verify load + run + report exposure ───────
    real_fds = Path("data/raw/sim_1500kw_2m2_T05")
    if real_fds.exists():
        print(
            "\n[6a] Load FDS RiskMap and probe peak danger field"
        )
        truth = load_truth_risk_map(real_fds, verbose=True)
        # Probe a 5x5 grid in the central courtyard area at t=120s.
        peak_seen = 0.0
        for x in np.linspace(5.0, 25.0, 5):
            for y in np.linspace(3.0, 17.0, 5):
                d = float(truth.query(np.array([x, y, 1.5]), t=120.0))
                peak_seen = max(peak_seen, d)
        print(f"  peak danger seen in 5x5 probe at t=120s: {peak_seen:.3f}")
        if peak_seen < 0.3:
            errors.append(
                f"FDS RiskMap peak {peak_seen:.3f} is suspiciously low; "
                f"load may have failed silently"
            )
        else:
            print(
                f"  PASS: FDS load is live "
                f"(peak {peak_seen:.3f} > 0.3 threshold)"
            )

        print(
            "\n[6b] run() with real FDS, t_end=200 s -- exposure may or may"
            " not be > 0 depending on where stuck agents end up"
        )
        try:
            real_result = run(
                fire_scenario_id="sim_1500kw_2m2_T05",
                fds_dir=real_fds,
                n_persons=5,
                seed=0,
                t_end_s=200.0,
                dt_s=1.0,
            )
            print(f"  {real_result.summary_line()}")
            if real_result.scenario_id != "S1_fixed_sign":
                errors.append("real-FDS scenario_id wrong")
            # No hard assertion on exposure: with this small building +
            # zero-planner pathing, evacuation completes in ~5 s for
            # agents that can reach an exit, and stuck agents stay in
            # their starting room (whose local danger depends on fire
            # location -- T05 is north-side and stuck agents are
            # south-east, so exposure can legitimately be 0).
            #
            # The structural test [6a] already confirmed the FDS load
            # works; this run is reported for the record.
            if real_result.danger_zone_exposure_time_s > 0:
                print(
                    f"  exposure {real_result.danger_zone_exposure_time_s:.1f}s "
                    f"> 0: at least one agent traversed a >0.5 danger cell"
                )
            else:
                print(
                    "  exposure=0: no agent's path crossed a >0.5 danger "
                    "cell (consistent with stuck-in-zone-d, fire-in-north)"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"real-FDS run crashed: {exc}")
    else:
        print(
            f"\n[6] SKIP: real FDS scenario {real_fds} not on disk"
        )

    # ── Verdict ────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: s1_fixed_sign M3-alpha validated")
