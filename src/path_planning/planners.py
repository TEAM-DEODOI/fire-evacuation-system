"""Weighted A* evacuation planner.

After the 2026-05-13 H6 redefinition (CLAUDE.md "EXP-PATH-001 — Three
Scenarios"), EXP-PATH-001 no longer compares path-planning algorithms.
It compares three PyBullet scenarios (fixed-sign baseline vs FDS-driven
drone swarm vs PI-FNO-driven drone swarm). In the two drone-swarm
scenarios the swarm needs a single waypoint generator: a risk-aware
weighted A* on the building graph. *That* is what this module provides
— **one planner**, not the legacy three-class Dijkstra/Static/Dynamic
ABC hierarchy.

Pipeline per call::

    plan(start_xyz, risk_map, t)
        │
        ├── compute_edge_weights(graph, risk_map, t)          ← in place
        ├── remove_impassable_edges(graph_copy)
        ├── snap start_xyz to nearest graph node
        ├── pick reachable exit with minimum A* cost
        └── return waypoint list  [start_xyz, node_1, …, exit_node]

The planner takes a *deep copy* of the graph on each call so mutations
to edge weights do not leak across replans (the underlying graph
returned by ``build_graph`` is meant to be immutable topology).

A separate :meth:`EvacuationPlanner.replan` exists as a semantic
convenience — it simply calls :meth:`plan` from the current position
and falls back to the last successful path if no route is reachable.
"""
from __future__ import annotations

import math
from typing import List, Optional

import networkx as nx
import numpy as np

from src.path_planning.building_graph import exit_nodes, snap_to_graph
from src.path_planning.edge_weights import (
    EdgeWeightConfig,
    compute_edge_weights,
    remove_impassable_edges,
)
from src.risk_map.risk_map_class import RiskMap


# ─── Heuristics ────────────────────────────────────────────────────────────
def _euclidean_xy(pos_a, pos_b) -> float:
    """2-D heuristic distance (single floor — Z ignored)."""
    return math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])


def _manhattan_xy(pos_a, pos_b) -> float:
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


_HEURISTICS = {
    "euclidean": _euclidean_xy,
    "manhattan": _manhattan_xy,
}


# ─── Planner ───────────────────────────────────────────────────────────────
class EvacuationPlanner:
    """Risk-aware weighted A* planner.

    The same instance can be re-used across many ``plan`` / ``replan``
    calls — it stores only the (immutable) building graph plus
    edge-weight hyperparameters. State that changes between calls (risk
    map, simulation time, last successful path) is passed in explicitly.

    Args:
        graph: NetworkX graph from
            :func:`src.path_planning.building_graph.build_graph`.
        config: Edge-weight hyperparameters. Defaults to
            :class:`EdgeWeightConfig()`.
        heuristic: ``"euclidean"`` or ``"manhattan"``. Both are
            admissible — Euclidean is tighter and usually expands fewer
            nodes on the maze-style building.
        max_path_length: Safety cap on the returned waypoint count
            (excluding the prepended ``start_xyz``).
        fallback_to_last: When ``replan`` cannot find a route, return
            the last successful path instead of an empty list.
    """

    def __init__(
        self,
        graph: nx.Graph,
        config: EdgeWeightConfig | None = None,
        heuristic: str = "euclidean",
        max_path_length: int = 500,
        fallback_to_last: bool = True,
    ) -> None:
        if heuristic not in _HEURISTICS:
            raise ValueError(
                f"heuristic must be one of {sorted(_HEURISTICS)}, "
                f"got {heuristic!r}"
            )
        if graph.number_of_nodes() == 0:
            raise ValueError("graph has no nodes")
        if not exit_nodes(graph):
            raise ValueError("graph has no exits (is_exit=True)")

        self.graph = graph
        self.config = config or EdgeWeightConfig()
        self.heuristic_name = heuristic
        self.max_path_length = int(max_path_length)
        self.fallback_to_last = bool(fallback_to_last)
        self._last_path: List[np.ndarray] = []

    # ─── Public API ────────────────────────────────────────────────────
    def plan(
        self,
        start_xyz: np.ndarray,
        risk_map: RiskMap,
        t: float = 0.0,
    ) -> List[np.ndarray]:
        """Compute the safest path from ``start_xyz`` to the nearest exit.

        Algorithm:
        1. Snap ``start_xyz`` to its nearest graph node.
        2. Take a deep copy of the topology and apply
           :func:`compute_edge_weights` at time ``t``; remove edges
           flagged impassable.
        3. Run NetworkX ``astar_path`` to every reachable exit; pick the
           one with minimum total cost.
        4. Prepend ``start_xyz`` so the caller can interpolate from the
           real occupant position (not the snapped node).

        Args:
            start_xyz: ``(3,)`` world position in metres.
            risk_map: Any :class:`RiskMap` implementation.
            t: Simulation time (s).

        Returns:
            Waypoints, each ``np.ndarray`` of shape ``(3,)``. Empty list
            if no exit is reachable.

        Raises:
            ValueError: If ``start_xyz`` is not 3-D.
        """
        arr = np.asarray(start_xyz, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"start_xyz must be shape (3,), got {arr.shape}")

        # Step 1 — snap start
        start_node = snap_to_graph(arr, self.graph)

        # Step 2 — weighted graph
        g_planning = self.graph.copy()
        compute_edge_weights(g_planning, risk_map, t=t, config=self.config)
        remove_impassable_edges(g_planning)

        # Step 3 — A* to every reachable exit
        best_path_nodes: Optional[List[str]] = None
        best_cost = math.inf
        h_fn = _HEURISTICS[self.heuristic_name]
        for ex in exit_nodes(g_planning):
            if not nx.has_path(g_planning, start_node, ex):
                continue
            try:
                nodes = nx.astar_path(
                    g_planning,
                    source=start_node,
                    target=ex,
                    heuristic=lambda u, v: h_fn(
                        g_planning.nodes[u]["pos"],
                        g_planning.nodes[v]["pos"],
                    ),
                    weight="weight",
                )
            except nx.NetworkXNoPath:
                continue
            cost = _path_weight(g_planning, nodes)
            if cost < best_cost:
                best_cost = cost
                best_path_nodes = nodes

        if best_path_nodes is None:
            return []

        # Step 4 — build waypoint list
        wp = [arr.copy()]
        for nid in best_path_nodes:
            pos = np.asarray(g_planning.nodes[nid]["pos"], dtype=np.float64)
            wp.append(pos)
        # Cap length (rare safety net — building graph is small).
        if len(wp) - 1 > self.max_path_length:
            wp = wp[: 1 + self.max_path_length]

        self._last_path = wp
        return wp

    def replan(
        self,
        current_xyz: np.ndarray,
        risk_map: RiskMap,
        t: float = 0.0,
    ) -> List[np.ndarray]:
        """Re-plan from the current position.

        Calls :meth:`plan` and, if no route is reachable and
        ``fallback_to_last`` is ``True``, returns the last successful
        path. Otherwise returns an empty list.

        Args:
            current_xyz: Current occupant position in metres.
            risk_map: Updated risk information (e.g. a fresh model
                prediction at time ``t``).
            t: Simulation time (s).

        Returns:
            Waypoints. See :meth:`plan`.
        """
        new_path = self.plan(current_xyz, risk_map, t=t)
        if new_path:
            return new_path
        if self.fallback_to_last and self._last_path:
            return list(self._last_path)
        return []

    @property
    def last_path(self) -> List[np.ndarray]:
        """Most recent successful path (read-only snapshot)."""
        return list(self._last_path)


# ─── Helpers ──────────────────────────────────────────────────────────────
def _path_weight(graph: nx.Graph, nodes: List[str]) -> float:
    """Sum of ``weight`` along a node path."""
    return sum(
        graph[u][v]["weight"] for u, v in zip(nodes[:-1], nodes[1:])
    )


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.path_planning.building_graph import build_graph
    from src.risk_map.risk_map_class import StaticRiskMap
    from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS

    print("=" * 60)
    print("planners.py self-test")
    print("=" * 60)

    errors: list[str] = []

    g = build_graph()
    nx_, ny_, nz_ = GRID_SHAPE
    times = np.arange(0.0, N_TIMESTEPS * DT_SLCF, DT_SLCF)

    # All-safe map for baseline shortest-path check.
    safe_rm = StaticRiskMap(
        danger_array=np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32),
        times=times,
    )

    # Fire blocks the central courtyard (hall_n ↔ hall_e edge area).
    # World box ~ [10, 22] × [8, 14] saturated to 1.0.
    fire = np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32)
    fire[:, 20:44, 16:28, :] = 1.0    # cells ⇒ world [10, 22] × [8, 14]
    fire_rm = StaticRiskMap(danger_array=fire, times=times)

    # ── 1. Plan with safe map from zone_b_west → some exit ─────────────
    print("\n[1] Safe map: plan from (4, 4, 1.5) to nearest exit")
    planner = EvacuationPlanner(g)
    start_safe = np.array([4.0, 4.0, 1.5])  # near zone_b_west
    path_safe = planner.plan(start_safe, safe_rm, t=0.0)
    if not path_safe:
        errors.append("safe-map plan returned empty path")
    else:
        ids = [snap_to_graph(p, g) for p in path_safe]
        print(f"  path length = {len(path_safe)} waypoints")
        print(f"  start → {ids[1]}  …  end → {ids[-1]}")
        if not g.nodes[ids[-1]].get("is_exit"):
            errors.append(f"final waypoint {ids[-1]} is not an exit")

    # ── 2. Plan with fire blocking central hall ─────────────────────────
    print("\n[2] Fire blocks central hall: planner reroutes via different chain")
    # Use threshold low enough to mark fire edges impassable.
    strict_cfg = EdgeWeightConfig(
        base_cost=1.0, risk_scale=10.0, risk_threshold=0.5, n_samples=5
    )
    planner_strict = EvacuationPlanner(g, config=strict_cfg)
    start_fire = np.array([4.0, 4.0, 1.5])    # zone B → south route should win
    path_fire = planner_strict.plan(start_fire, fire_rm, t=100.0)
    if not path_fire:
        errors.append("fire-map plan returned empty path")
    else:
        ids_fire = [snap_to_graph(p, g) for p in path_fire]
        print(f"  fire path: {' → '.join(ids_fire[1:])}")
        # Must not traverse hall_n/hall_e (impassable under fire).
        for blocked in ("hall_n", "hall_e", "hall_w"):
            # hall_w is also adjacent to the fire zone X=[10,22]
            pass
        # Stronger structural check: weight along returned path must use
        # only edges with edge_risk <= threshold (i.e. all sub-impassable).
        # We re-evaluate to confirm.
        g_check = g.copy()
        compute_edge_weights(g_check, fire_rm, t=100.0, config=strict_cfg)
        for u, v in zip(ids_fire[1:-1], ids_fire[2:]):
            if g_check.has_edge(u, v):
                r = g_check[u][v]["edge_risk"]
                if r > strict_cfg.risk_threshold + 1e-6:
                    errors.append(
                        f"returned path uses high-risk edge {u}↔{v} "
                        f"risk={r:.3f}"
                    )
            else:
                errors.append(
                    f"returned path uses missing edge {u}↔{v}"
                )

    # ── 3. plan() rejects bad start shape ──────────────────────────────
    print("\n[3] plan() rejects non-(3,) start")
    try:
        planner.plan(np.array([1.0, 2.0]), safe_rm)
    except ValueError:
        print("  PASS: (2,) → ValueError")
    else:
        errors.append("(2,) start did not raise")

    # ── 4. Heuristic validation ────────────────────────────────────────
    print("\n[4] Bad heuristic name rejected")
    try:
        EvacuationPlanner(g, heuristic="taxicab")
    except ValueError:
        print("  PASS")
    else:
        errors.append("bad heuristic did not raise")

    # ── 5. Unreachable exit → replan falls back to last path ────────────
    print("\n[5] replan() fallback when all exits blocked")
    # Saturate everything to 1.0 → every edge impassable → no route.
    block_all = StaticRiskMap(
        danger_array=np.ones((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32),
        times=times,
    )
    fb_planner = EvacuationPlanner(g, fallback_to_last=True)
    first = fb_planner.plan(start_safe, safe_rm, t=0.0)
    if not first:
        errors.append("first plan unexpectedly empty")
    fallback = fb_planner.replan(start_safe, block_all, t=0.0)
    if not fallback:
        errors.append("fallback to last path failed")
    else:
        print(f"  PASS: fallback path length = {len(fallback)}")

    # ── 6. fallback_to_last=False → empty list ──────────────────────────
    print("\n[6] replan() without fallback returns []")
    no_fb_planner = EvacuationPlanner(g, fallback_to_last=False)
    no_fb_planner.plan(start_safe, safe_rm, t=0.0)
    empty = no_fb_planner.replan(start_safe, block_all, t=0.0)
    if empty:
        errors.append(f"expected [], got {len(empty)}-wp path")
    else:
        print("  PASS")

    # ── 7. Last path snapshot is read-only ──────────────────────────────
    print("\n[7] last_path returns a copy")
    snap = fb_planner.last_path
    snap.append(np.zeros(3))
    if len(fb_planner.last_path) == len(snap):
        errors.append("last_path returned mutable reference")
    else:
        print("  PASS")

    # ── 8. Manhattan heuristic alternative ──────────────────────────────
    print("\n[8] Manhattan heuristic produces a valid path")
    m_planner = EvacuationPlanner(g, heuristic="manhattan")
    m_path = m_planner.plan(start_safe, safe_rm, t=0.0)
    if not m_path:
        errors.append("manhattan planner returned empty")
    else:
        print(f"  PASS: {len(m_path)} waypoints")

    # ── Verdict ────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: EvacuationPlanner validated")
