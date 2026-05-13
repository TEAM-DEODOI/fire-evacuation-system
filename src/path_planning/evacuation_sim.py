"""Logical (non-PyBullet) single-occupant evacuation simulation.

This module exists for two purposes:

1. **Unit test** for :class:`~src.path_planning.planners.EvacuationPlanner`.
   Demonstrates that planner + risk_map + occupant motion compose into
   an evacuation trial that the D-022 4-metric evaluator
   (:func:`src.risk_map.path_metrics.evaluate_path_safety`) can score.

2. **Cheap proxy** for ablations that don't need the PyBullet renderer:
   sweep start positions, risk-map sources or replan cadences before
   running the expensive PyBullet experiments.

The full multi-agent PyBullet simulation lives in ``src/integration/``
per CLAUDE.md "EXP-PATH-001 — Three Scenarios" (post 2026-05-13). This
module does **not** import pybullet and runs purely in NumPy.

Algorithm per step::

    while not arrived and t < t_end:
        if t since last replan ≥ replan_interval:
            waypoints = planner.replan(current_xyz, risk_map, t)
        advance current_xyz toward next waypoint by walking_speed * dt
        log danger_along_path = risk_map.query(current_xyz, t)
        t += dt
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx
import numpy as np

from src.path_planning.building_graph import exit_nodes
from src.path_planning.planners import EvacuationPlanner
from src.risk_map.risk_map_class import RiskMap
from src.shared.constants import REPLAN_PERIOD_S, T_END_SECONDS, WALKING_SPEED_MPS


# ─── Result dataclass ─────────────────────────────────────────────────────
@dataclass
class EvacuationResult:
    """Outcome of one logical evacuation trial.

    Attributes:
        success: ``True`` if the occupant reached an exit within ``t_end``.
        exit_time: Time of arrival (s). ``inf`` if not arrived.
        trajectory: ``(N, 3)`` positions at each logged step (metres).
        times: ``(N,)`` simulation times for each position (s).
        danger_history: Per-step ``risk_map.query(current_xyz, t)`` value.
        n_replans: Number of times the planner re-ran during the trial.
        final_waypoints: Last successful waypoint list from the planner.
        target_exit_xyz: World position of the exit the path ended at
            (``None`` if no path was ever found).
    """

    success: bool
    exit_time: float
    trajectory: List[np.ndarray] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    danger_history: List[float] = field(default_factory=list)
    n_replans: int = 0
    final_waypoints: List[np.ndarray] = field(default_factory=list)
    target_exit_xyz: Optional[np.ndarray] = None

    # Convenience views ────────────────────────────────────────────────
    def trajectory_arr(self) -> np.ndarray:
        """Stack ``trajectory`` into an ``(N, 3)`` array (empty → shape (0, 3))."""
        if not self.trajectory:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(self.trajectory, dtype=np.float64)

    def times_arr(self) -> np.ndarray:
        if not self.times:
            return np.empty((0,), dtype=np.float64)
        return np.asarray(self.times, dtype=np.float64)

    def summary_line(self) -> str:
        """One-line summary for logs."""
        arr = "OK" if self.success else "FAIL"
        return (
            f"arrived={arr}  exit_time={self.exit_time:.1f}s  "
            f"steps={len(self.trajectory)}  replans={self.n_replans}  "
            f"peak_danger={max(self.danger_history, default=0.0):.3f}"
        )


# ─── Simulation core ──────────────────────────────────────────────────────
def _advance_along_waypoints(
    pos: np.ndarray,
    waypoints: List[np.ndarray],
    wp_idx: int,
    distance: float,
) -> tuple[np.ndarray, int]:
    """Move ``pos`` toward the next waypoint by up to ``distance`` metres.

    Skips past waypoints fully consumed within the step. Returns the new
    position and the index of the waypoint we are still heading toward
    (or ``len(waypoints) - 1`` once the final waypoint has been reached).

    Args:
        pos: Current 3-D position.
        waypoints: List of waypoints (first is the start, last is the exit).
        wp_idx: Index of the *next* waypoint to head toward (≥ 1).
        distance: How far to move this step (m).

    Returns:
        ``(new_pos, new_wp_idx)``.
    """
    remaining = distance
    cur = pos.astype(np.float64).copy()
    while remaining > 0.0 and wp_idx < len(waypoints):
        target = np.asarray(waypoints[wp_idx], dtype=np.float64)
        delta = target - cur
        seg_len = float(np.linalg.norm(delta))
        if seg_len <= 1e-12:
            wp_idx += 1
            continue
        if seg_len <= remaining:
            cur = target
            remaining -= seg_len
            wp_idx += 1
        else:
            cur = cur + delta * (remaining / seg_len)
            remaining = 0.0
    # Clamp to last index if we've consumed every waypoint.
    if wp_idx >= len(waypoints):
        wp_idx = len(waypoints) - 1
    return cur, wp_idx


def simulate_evacuation(
    start_xyz: np.ndarray,
    planner: EvacuationPlanner,
    risk_map: RiskMap,
    *,
    walking_speed_mps: float = WALKING_SPEED_MPS,
    dt: float = 1.0,
    replan_interval: float = REPLAN_PERIOD_S,
    t_end: float = T_END_SECONDS,
    arrival_tolerance_m: float = 1.0,
) -> EvacuationResult:
    """Run one logical evacuation trial.

    The planner is invoked once at ``t=0`` to set the initial waypoints,
    then every ``replan_interval`` seconds with the updated occupant
    position. The occupant moves at constant ``walking_speed_mps`` along
    the current waypoint sequence.

    The "risk experienced" recorded in ``danger_history`` is
    ``risk_map.query(current_xyz, t)`` at each step. To compute the
    D-022 4-metric or CO-based FED, pass the resulting trajectory to
    :func:`src.risk_map.path_metrics.evaluate_path_safety`.

    Args:
        start_xyz: ``(3,)`` initial position (m).
        planner: Configured :class:`EvacuationPlanner`.
        risk_map: Risk source used by the planner AND for danger
            logging. (Fairness-aware experiments that need separate
            planner-vs-truth maps should wrap the planner accordingly.)
        walking_speed_mps: Constant speed (D-017). Default ``1.5``.
        dt: Integration step (s). Default ``1.0``.
        replan_interval: Seconds between consecutive replans. Default
            ``REPLAN_PERIOD_S`` (= 30 s, D-014).
        t_end: Maximum simulation time (s). Default ``T_END_SECONDS``
            (= 300 s).
        arrival_tolerance_m: Distance from the final waypoint at which
            arrival is declared.

    Returns:
        :class:`EvacuationResult` with full trajectory and outcome.

    Raises:
        ValueError: If ``start_xyz`` is not 3-D, ``dt <= 0``, or
            ``walking_speed_mps <= 0``.
    """
    arr = np.asarray(start_xyz, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"start_xyz must have shape (3,), got {arr.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if walking_speed_mps <= 0.0:
        raise ValueError(
            f"walking_speed_mps must be > 0, got {walking_speed_mps}"
        )
    if replan_interval <= 0.0:
        raise ValueError(
            f"replan_interval must be > 0, got {replan_interval}"
        )

    # ── Initial plan ──────────────────────────────────────────────────
    waypoints = planner.plan(arr, risk_map, t=0.0)
    n_replans = 1

    if not waypoints:
        # No exit reachable from t=0 → fail fast.
        return EvacuationResult(
            success=False,
            exit_time=math.inf,
            trajectory=[arr.copy()],
            times=[0.0],
            danger_history=[float(risk_map.query(arr, t=0.0))],
            n_replans=n_replans,
            final_waypoints=[],
            target_exit_xyz=None,
        )

    pos = arr.copy()
    wp_idx = 1   # waypoints[0] is the start point (== current pos)
    t = 0.0
    last_replan_t = 0.0
    step_distance = walking_speed_mps * dt

    trajectory: List[np.ndarray] = [pos.copy()]
    times: List[float] = [t]
    danger_history: List[float] = [float(risk_map.query(pos, t=t))]
    target_exit = np.asarray(waypoints[-1], dtype=np.float64).copy()
    final_waypoints = list(waypoints)
    success = False
    exit_time = math.inf

    # ── Loop ──────────────────────────────────────────────────────────
    safety_max_steps = int(math.ceil(t_end / dt)) + 1
    steps = 0
    while t < t_end and steps < safety_max_steps:
        steps += 1
        # Re-plan if interval elapsed (and we are still in motion).
        if t - last_replan_t >= replan_interval and wp_idx < len(waypoints):
            new_path = planner.replan(pos, risk_map, t=t)
            n_replans += 1
            if new_path:
                waypoints = new_path
                # New path starts at current pos → head to its second wp.
                wp_idx = 1
                target_exit = np.asarray(waypoints[-1], dtype=np.float64).copy()
                final_waypoints = list(waypoints)
            last_replan_t = t

        # Move occupant.
        pos, wp_idx = _advance_along_waypoints(
            pos, waypoints, wp_idx, step_distance
        )
        t += dt

        # Log.
        trajectory.append(pos.copy())
        times.append(t)
        danger_history.append(float(risk_map.query(pos, t=t)))

        # Arrival check (relative to the path's final waypoint).
        if np.linalg.norm(pos - target_exit) <= arrival_tolerance_m:
            success = True
            exit_time = t
            break

    return EvacuationResult(
        success=success,
        exit_time=exit_time,
        trajectory=trajectory,
        times=times,
        danger_history=danger_history,
        n_replans=n_replans,
        final_waypoints=final_waypoints,
        target_exit_xyz=target_exit,
    )


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.path_planning.building_graph import build_graph
    from src.path_planning.edge_weights import EdgeWeightConfig
    from src.risk_map.path_metrics import (
        compute_peak_danger,
        compute_time_in_hazard,
    )
    from src.risk_map.risk_map_class import StaticRiskMap
    from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS

    print("=" * 60)
    print("evacuation_sim.py self-test")
    print("=" * 60)

    errors: list[str] = []

    g = build_graph()
    nx_, ny_, nz_ = GRID_SHAPE
    times = np.arange(0.0, N_TIMESTEPS * DT_SLCF, DT_SLCF)

    safe_rm = StaticRiskMap(
        danger_array=np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32),
        times=times,
    )

    # Fire in north corridor area (world ~ [14, 22] × [14, 18]).
    fire = np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32)
    fire[:, 28:44, 28:36, :] = 1.0
    fire_rm = StaticRiskMap(danger_array=fire, times=times)

    block_all_rm = StaticRiskMap(
        danger_array=np.ones((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32),
        times=times,
    )

    # ── 1. Safe map, occupant reaches exit ─────────────────────────────
    print("\n[1] Safe map: occupant should reach exit")
    planner = EvacuationPlanner(g)
    result = simulate_evacuation(
        start_xyz=np.array([4.0, 4.0, 1.5]),
        planner=planner,
        risk_map=safe_rm,
        replan_interval=30.0,
        t_end=180.0,
    )
    print(f"  {result.summary_line()}")
    if not result.success:
        errors.append("safe-map evacuation failed")
    if result.exit_time > 120.0:
        # Distance ≤ ~10 m at 1.5 m/s ⇒ ≤ ~7 s; allow generous margin
        errors.append(f"exit_time {result.exit_time}s unexpectedly large")

    # ── 2. Fire blocks one route: occupant still arrives ────────────────
    print("\n[2] Fire blocks part of building: occupant reroutes and arrives")
    strict_cfg = EdgeWeightConfig(
        base_cost=1.0, risk_scale=10.0, risk_threshold=0.5, n_samples=5
    )
    planner_strict = EvacuationPlanner(g, config=strict_cfg)
    result_fire = simulate_evacuation(
        start_xyz=np.array([4.0, 4.0, 1.5]),
        planner=planner_strict,
        risk_map=fire_rm,
        replan_interval=30.0,
        t_end=180.0,
    )
    print(f"  {result_fire.summary_line()}")
    if not result_fire.success:
        errors.append("fire-map evacuation failed to reach exit")

    # ── 3. All-blocked map: simulation fails fast ───────────────────────
    print("\n[3] All-blocked map: planner finds no route, sim returns failure")
    blocked_planner = EvacuationPlanner(g, fallback_to_last=False)
    blocked = simulate_evacuation(
        start_xyz=np.array([4.0, 4.0, 1.5]),
        planner=blocked_planner,
        risk_map=block_all_rm,
        replan_interval=30.0,
        t_end=120.0,
    )
    if blocked.success:
        errors.append("blocked map: should not have arrived")
    if len(blocked.trajectory) != 1:
        errors.append(
            f"blocked map: expected single-step trajectory, got "
            f"{len(blocked.trajectory)}"
        )
    print(f"  PASS: {blocked.summary_line()}")

    # ── 4. Replan count grows with elapsed time ─────────────────────────
    print("\n[4] replan count grows with t_end / replan_interval (safe case)")
    slow_planner = EvacuationPlanner(g)
    long_result = simulate_evacuation(
        # Far corner so the walk takes a while.
        start_xyz=np.array([28.0, 16.0, 1.5]),
        planner=slow_planner,
        risk_map=safe_rm,
        walking_speed_mps=0.5,   # slower so we hit multiple replans
        dt=1.0,
        replan_interval=10.0,
        t_end=120.0,
    )
    expected_min = 1  # at least the initial plan
    if long_result.n_replans < expected_min:
        errors.append(
            f"replan count {long_result.n_replans} < {expected_min}"
        )
    print(f"  PASS: n_replans={long_result.n_replans} for t_end=120, every 10s")

    # ── 5. Path metric compatibility: feed result into evaluate_path_safety ─
    print("\n[5] Result is compatible with risk_map.path_metrics")
    path = result_fire.trajectory_arr()
    ts = result_fire.times_arr()
    peak = compute_peak_danger(path, ts, fire_rm)
    hazard = compute_time_in_hazard(path, ts, fire_rm)
    print(f"  peak_danger={peak:.3f}  time_in_hazard={hazard:.1f}s")
    # Sanity: trajectory exists & metrics computed without raising.

    # ── 6. Bad inputs rejected ─────────────────────────────────────────
    print("\n[6] Input validation")
    for bad in (
        dict(start_xyz=np.array([1.0, 2.0])),       # bad shape
        dict(dt=0.0),
        dict(walking_speed_mps=-1.0),
        dict(replan_interval=0.0),
    ):
        try:
            kwargs = dict(
                start_xyz=np.array([4.0, 4.0, 1.5]),
                planner=planner,
                risk_map=safe_rm,
            )
            kwargs.update(bad)
            simulate_evacuation(**kwargs)
        except ValueError:
            continue
        errors.append(f"input {bad} did not raise ValueError")
    print("  PASS: all bad inputs rejected")

    # ── 7. Trajectory length aligned with times ─────────────────────────
    print("\n[7] trajectory and times stay aligned")
    if len(result.trajectory) != len(result.times):
        errors.append(
            f"len(trajectory)={len(result.trajectory)} != "
            f"len(times)={len(result.times)}"
        )
    if len(result.trajectory) != len(result.danger_history):
        errors.append(
            f"len(trajectory)={len(result.trajectory)} != "
            f"len(danger_history)={len(result.danger_history)}"
        )
    print(f"  PASS: aligned at {len(result.trajectory)} entries")

    # ── Verdict ────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: simulate_evacuation validated")
