"""Predictive evacuation planner -- static lookahead offset (D-048, S3 only).

Standard :class:`~src.path_planning.planners.EvacuationPlanner` queries
every cell at the current wall-clock time ``t_now``: A* sees the
*present* risk field even though the Tier 1 GNN actually predicts
``t_now .. t_now + 50 s`` (6 frames at 10 s). This planner exploits
that lookahead via the simplest possible mechanism --- a constant
**lookahead offset** ``delta_s`` applied at the RiskMap query call::

    plan(start_xyz, risk_map, t=t_now)
                  ↓
        super().plan(start_xyz, risk_map, t=t_now + delta_s)
                  ↓
        compute_edge_weights queries risk_map at the SHIFTED time
                  ↓
        A* on a graph that reflects the future risk field

**Why S3 only**

S1 has no planner. S2 sees the FDS truth, where querying "now" or
"30 s later" returns ground-truth values in both cases --- the offset
just trades a different oracle slice. S3 is the only scenario whose
RiskMap genuinely encodes *future* state (the GNN's 6-frame forecast),
so the offset is where the GNN's lookahead pays for itself.

**Behaviour at boundaries**

The shifted time may fall outside the RiskMap's valid range:

* :class:`~src.tier1.tier1_risk_map.Tier1RiskMap` clamps OOB times to
  the nearest predicted frame (D-038), so we get the freshest or
  furthest forecast frame rather than the "OOB → 1.0" safety value.
* :class:`StaticRiskMap` (the fallback when Tier 1 artefacts are
  missing) interpolates within its 0..300 s coverage and snaps at the
  edges, so the offset is similarly safe.

A non-positive ``delta_s`` reduces this planner to the parent class
(no-op offset).
"""
from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np

from src.path_planning.edge_weights import EdgeWeightConfig
from src.path_planning.planners import EvacuationPlanner
from src.risk_map.risk_map_class import RiskMap


class PredictiveEvacuationPlanner(EvacuationPlanner):
    """Static-offset predictive variant of :class:`EvacuationPlanner`.

    Drop-in replacement: same ``plan`` / ``replan`` signature, same
    return shape, identical internal A*. The only change is a constant
    ``delta_s`` added to the time passed into the parent's edge-weight
    computation, so every cell is evaluated at ``t_now + delta_s``
    instead of ``t_now``.

    Args:
        graph: Cell-grid from
            :func:`src.path_planning.building_graph.build_graph`.
        config: Edge-weight hyperparameters. Defaults to
            :class:`EdgeWeightConfig()`.
        heuristic: ``"euclidean"`` or ``"manhattan"``. See parent.
        max_path_length: Safety cap on returned waypoints.
        fallback_to_last: When ``replan`` cannot find a route, return
            the last successful path.
        delta_s: Lookahead offset (s) applied to the RiskMap query.
            Default ``30.0`` -- 3 frames forward into the GNN's 6-frame
            forecast, roughly the time a person at 0.5 m/s covers in a
            mid-length evacuation. Set to ``0`` to reduce to the parent
            class.
    """

    def __init__(
        self,
        graph: nx.Graph,
        config: EdgeWeightConfig | None = None,
        heuristic: str = "euclidean",
        max_path_length: int = 500,
        fallback_to_last: bool = True,
        *,
        delta_s: float = 30.0,
    ) -> None:
        super().__init__(
            graph,
            config=config,
            heuristic=heuristic,
            max_path_length=max_path_length,
            fallback_to_last=fallback_to_last,
        )
        if delta_s < 0:
            raise ValueError(
                f"delta_s must be >= 0, got {delta_s}"
            )
        self.delta_s: float = float(delta_s)

    # ─── Public API ────────────────────────────────────────────────────
    def plan(
        self,
        start_xyz: np.ndarray,
        risk_map: RiskMap,
        t: float = 0.0,
        *,
        co_field: Optional[object] = None,
    ) -> List[np.ndarray]:
        """Plan with the RiskMap queried at ``t + delta_s``.

        Args:
            start_xyz: ``(3,)`` world position in metres.
            risk_map: :class:`RiskMap` supporting per-call ``t``.
            t: Current wall-clock time (s). The planner queries the
                RiskMap at ``t + self.delta_s``.
            co_field: Optional CO field, forwarded to the parent.

        Returns:
            Waypoints (list of ``(3,)`` arrays).
        """
        return super().plan(
            start_xyz,
            risk_map,
            t=float(t) + self.delta_s,
            co_field=co_field,
        )

    # Note: ``replan`` is inherited from EvacuationPlanner and calls
    # ``self.plan(...)``, so the offset is applied automatically there
    # too. No override needed.


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    from src.path_planning.building_graph import build_graph, snap_to_graph
    from src.risk_map.risk_map_class import StaticRiskMap
    from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS

    print("=" * 60)
    print("predictive_planner.py self-test (D-048 / Plan A)")
    print("=" * 60)

    errors: list[str] = []

    g = build_graph()
    nx_, ny_, nz_ = GRID_SHAPE
    times = np.arange(0.0, N_TIMESTEPS * DT_SLCF, DT_SLCF)

    # ── 1. delta_s validation ──────────────────────────────────────────
    print("\n[1] delta_s rejects negative values")
    try:
        PredictiveEvacuationPlanner(g, delta_s=-5.0)
    except ValueError:
        print("  PASS: -5.0 -> ValueError")
    else:
        errors.append("delta_s=-5.0 did not raise")

    # ── 2. Safe map: any delta gives a valid path ──────────────────────
    print("\n[2] safe map: planner returns a path to an exit")
    safe_rm = StaticRiskMap(
        danger_array=np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32),
        times=times,
    )
    planner = PredictiveEvacuationPlanner(g, delta_s=30.0)
    start = np.array([4.0, 4.0, 1.5])
    path = planner.plan(start, safe_rm, t=0.0)
    if not path:
        errors.append("safe-map predictive plan returned []")
    else:
        end_node = snap_to_graph(path[-1], g)
        if not g.nodes[end_node].get("is_exit"):
            errors.append("final waypoint is not an exit")
        print(f"  PASS: {len(path)} waypoints, ends at {end_node}")

    # ── 3. delta_s=0 reduces to standard planner ───────────────────────
    print("\n[3] delta_s=0 reproduces the parent EvacuationPlanner path")
    danger = np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32)
    danger[:, :, :, :] = 0.3      # mild uniform risk -- planner navigates normally
    barrier_rm = StaticRiskMap(danger_array=danger, times=times)

    from src.path_planning.planners import EvacuationPlanner as PlainPlanner
    cfg_for_3 = EdgeWeightConfig(
        base_cost=1.0, risk_scale=10.0, risk_threshold=1.0,
    )
    plain = PlainPlanner(g, config=cfg_for_3)
    pred_zero = PredictiveEvacuationPlanner(g, config=cfg_for_3, delta_s=0.0)
    start_east = np.array([22.0, 10.0, 1.5])
    plain_path = plain.plan(start_east, barrier_rm, t=0.0)
    zero_path  = pred_zero.plan(start_east, barrier_rm, t=0.0)
    plain_ids = [snap_to_graph(p, g) for p in plain_path]
    zero_ids  = [snap_to_graph(p, g) for p in zero_path]
    if plain_ids != zero_ids:
        errors.append(
            f"delta_s=0 path differs from plain "
            f"(plain={len(plain_ids)}, zero={len(zero_ids)})"
        )
    else:
        print(f"  PASS: identical paths ({len(plain_ids)} nodes)")

    # ── 4. Future-only fire: delta_s shifts the planner's view ─────────
    # Risk = 0 at t=0..50s. From t=60s onward a barrier spans x=15..18 m.
    # delta_s=0 sees no barrier (t=0 query, all zero).
    # delta_s=60 sees the barrier and must re-route.
    # Risk = 0 at t=0..50s everywhere. From t=60s onward risk = 1.0
    # everywhere -> every edge becomes impassable under the strict
    # threshold and the planner returns []. The delta=0 planner sees
    # only the t=0 zeros and plans normally. The delta=60 planner sees
    # only the future ones and bails. That asymmetry is the simplest
    # proof the lookahead offset is actually shifting the query time.
    print("\n[4] future-spike risk: delta=60 sees barrier, delta=0 does not")
    fut = np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32)
    fut[6:, :, :, :] = 1.0     # t >= 60 s -> everything dangerous
    fut_rm = StaticRiskMap(danger_array=fut, times=times)

    strict_cfg = EdgeWeightConfig(
        base_cost=1.0, risk_scale=10.0, risk_threshold=0.5,
    )
    p_now    = PredictiveEvacuationPlanner(g, config=strict_cfg, delta_s=0.0)
    p_future = PredictiveEvacuationPlanner(g, config=strict_cfg, delta_s=60.0)
    p_now_path    = p_now   .plan(start_east, fut_rm, t=0.0)
    p_future_path = p_future.plan(start_east, fut_rm, t=0.0)
    print(
        f"  delta=0  : {len(p_now_path)} waypoints (expect > 0)\n"
        f"  delta=60 : {len(p_future_path)} waypoints (expect 0)"
    )
    if not p_now_path:
        errors.append("delta=0 plan was empty at t=0 -- bad fixture")
    if p_future_path:
        errors.append(
            f"delta=60 plan returned {len(p_future_path)} waypoints "
            f"but every cell should be impassable at t=60s"
        )
    if p_now_path and not p_future_path:
        print("  PASS: lookahead offset is shifting the query time")

    # ── 5. replan() inherits the offset ────────────────────────────────
    print("\n[5] replan() also queries at t + delta_s")
    pr = PredictiveEvacuationPlanner(g, delta_s=30.0)
    first = pr.plan(start, safe_rm, t=0.0)
    again = pr.replan(start, safe_rm, t=0.0)
    if [snap_to_graph(p, g) for p in first] != [snap_to_graph(p, g) for p in again]:
        errors.append("replan returned a different path with same inputs")
    else:
        print(f"  PASS: replan reproduced the original path ({len(first)} nodes)")

    # ── Verdict ────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nPASS: PredictiveEvacuationPlanner validated (Plan A)")
