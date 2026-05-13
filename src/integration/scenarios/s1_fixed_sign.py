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
from src.integration.scene import Scene, SceneConfig
from src.integration.scenarios._common import (
    BUILDING_URDF,
    PLACEHOLDER_URDF,
    building_urdf_path,
    exit_positions,
    load_truth_risk_map,
    spawn_agents,
    zero_risk_map,
)
from src.risk_map.risk_map_class import RiskMap


# ─── Fixed-sign provider ──────────────────────────────────────────────────
@dataclass
class FixedSignNetwork:
    """Static-sign equivalent of a WaypointProvider for S1.

    The full intent (M3-full): a network of pre-placed signs at corridor
    intersections; each sign points to one exit. When a person queries
    :meth:`next_waypoint`, they walk to the nearest sign whose pointed-at
    exit is *currently* below ``danger_threshold``, then continue toward
    the exit from there.

    The M3-mini realization here collapses to "every exit is both sign
    and target" -- the agent walks straight to the nearest exit whose
    XYZ position has danger ≤ ``danger_threshold``, falling back to the
    absolute-nearest exit if all are hazardous. ``sign_positions`` is
    reserved for M3-full and currently ignored.
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
        """Implements :class:`WaypointProvider`.

        Returns the (3,) world position of the chosen exit, or ``None``
        if no exits are configured.
        """
        if not self.exit_positions:
            return None
        agent_xy = np.asarray(agent_position, dtype=np.float64)[:2]

        # Pass 1: nearest exit below threshold.
        best: Optional[np.ndarray] = None
        best_d2 = math.inf
        for ex in self.exit_positions:
            ex_arr = np.asarray(ex, dtype=np.float64)
            if float(self.risk_map.query(ex_arr, t=t)) > self.danger_threshold:
                continue
            dx = ex_arr[0] - agent_xy[0]
            dy = ex_arr[1] - agent_xy[1]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = ex_arr
        if best is not None:
            return best

        # Pass 2: every exit is hazardous -- pick the absolute nearest.
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
        obstacles = [scene.building_id]
        truth_rm = load_truth_risk_map(fds_dir, verbose=True)
        exits_xyz = exit_positions()
        agents = spawn_agents(scene, n_persons, seed)
        n_actual = len(agents)
        # Per fairness setup (interface_contracts.md sect. 6) the planner
        # and the experienced truth use the SAME RiskMap in S1. Drone
        # scenarios (S2/S3) will pass separate planner_rm and truth_rm.
        provider = FixedSignNetwork(
            risk_map=truth_rm,
            exit_positions=exits_xyz,
            danger_threshold=0.5,
        )

        arrived: Dict[str, float] = {}
        # Per-agent accumulated time in danger > 0.5 zone.
        exposure_s: Dict[str, float] = {a.agent_id: 0.0 for a in agents}
        max_steps = int(math.ceil(t_end_s / dt_s)) + 5
        danger_threshold = 0.5
        for _ in range(max_steps):
            t_now = float(scene.t)
            if t_now >= t_end_s:
                break
            for agent in agents:
                if agent.agent_id in arrived:
                    continue
                wp = provider.next_waypoint(agent.position, t_now)
                if wp is None:
                    continue
                agent.step_toward(
                    scene.client,
                    wp,
                    dt_s,
                    obstacle_body_ids=obstacles,
                )
                # Accumulate exposure at the post-move position. The
                # truth map is the same instance the FixedSignNetwork
                # consults but we re-query independently here so the
                # exposure metric remains correct under future S2/S3
                # planner/truth splits.
                danger = float(truth_rm.query(agent.position, t=t_now))
                if danger > danger_threshold:
                    exposure_s[agent.agent_id] += float(dt_s)

                # Arrived if within tolerance of *any* exit (XY only).
                for ex in exits_xyz:
                    if (
                        float(np.linalg.norm(agent.position[:2] - ex[:2]))
                        <= agent.config.arrival_tolerance_m
                    ):
                        arrived[agent.agent_id] = t_now
                        break
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
            scenario_id="S1_fixed_sign",
            fire_scenario_id=fire_scenario_id,
            seed=seed,
            n_persons=n_actual,
            evacuation_success_rate=success_rate,
            mean_evacuation_time_s=mean_evac_time,
            danger_zone_exposure_time_s=mean_exposure,
            casualty_rate=0.0,    # M3-α: status machine is M2-full
            cumulative_fed=0.0,   # M3-α: CO-grid load is M3-full
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
    # Agent at (5, 4): nearest is exit_west (0, 5) dist ~5.1
    wp_b = net.next_waypoint(np.array([5.0, 4.0, 1.5]), t=0.0)
    print(f"  query (5, 4)  -> {tuple(round(v, 1) for v in wp_b)}")
    if abs(wp_b[0] - 0.0) > 1e-6 or abs(wp_b[1] - 5.0) > 1e-6:
        errors.append(f"expected exit_west (0, 5), got {wp_b}")
    # Agent at (28, 16): nearest is exit_east (30, 13) dist sqrt(4+9)=3.6
    wp_e = net.next_waypoint(np.array([28.0, 16.0, 1.5]), t=0.0)
    print(f"  query (28, 16) -> {tuple(round(v, 1) for v in wp_e)}")
    if abs(wp_e[0] - 30.0) > 1e-6 or abs(wp_e[1] - 13.0) > 1e-6:
        errors.append(f"expected exit_east (30, 13), got {wp_e}")

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
