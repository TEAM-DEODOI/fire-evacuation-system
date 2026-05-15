"""39-detector navigation graph + planner for S3 drone routing (D-047).

Replaces the cell-grid (60×40×6) A* graph with a coarse-grained
**39-node graph** built on the canonical detector positions
(``src.tier1.detector_positions.ALL_DETECTORS``). Each node carries
a world XY position; edges are computed via **BFS over the fluid
mask** so the connection between two detectors is the true geodesic
distance through navigable cells — wall-crossing edges are dropped
automatically.

Why coarse?  Drone path planning needs only enough detail to choose
between corridors / rooms — exact cell-level path-following is the
person's job (D-032 self-BFS). With 39 nodes the A* is essentially
instantaneous and the node danger values come **straight from the
Tier 1 GNN forward** (one value per node, no node→cell projection
needed).

Public API:

* :func:`build_node_graph` — construct the graph (once per process).
* :class:`NodeGraphPlanner` — drop-in replacement for
  :class:`src.path_planning.planners.EvacuationPlanner` (same
  ``plan / replan`` signature so ``DroneSwarm.update`` works unchanged).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from src.path_planning.building_graph import _ACTIVE_EXIT_NAMES, load_default_fluid_mask
from src.shared.constants import CELL_SIZE_M
from src.tier1.detector_positions import ALL_DETECTORS


# ─── Detector exit id → building.py exit name mapping ─────────────────────
# ALL_DETECTORS uses ``exit_1_west`` / ``exit_2_north`` / ``exit_3_east``;
# shared.building uses ``exit_west`` / ``exit_north`` / ``exit_east``.
# D-041 keeps only ``exit_west`` active, so we filter via this map.
_DETECTOR_TO_BUILDING_EXIT: Dict[str, str] = {
    "exit_1_west":  "exit_west",
    "exit_2_north": "exit_north",
    "exit_3_east":  "exit_east",
}


# ─── Geodesic distance over fluid mask ────────────────────────────────────
def _snap_to_fluid_cell(
    xy_m: np.ndarray,
    layer: np.ndarray,
    *,
    max_search_cells: int = 8,
) -> Optional[Tuple[int, int]]:
    """Snap a world-XY position to the nearest fluid cell on ``layer``.

    Some detectors sit slightly outside the fluid region at z=3 (e.g.
    ``zone_c_mid_4``) — instead of dropping them, we project to the
    closest navigable cell within a small search radius. If no fluid
    cell exists within ``max_search_cells`` Chebyshev distance, returns
    None so the caller can drop the detector entirely.
    """
    nx_, ny_ = layer.shape
    ix0 = int(xy_m[0] / CELL_SIZE_M)
    iy0 = int(xy_m[1] / CELL_SIZE_M)
    ix0 = max(0, min(nx_ - 1, ix0))
    iy0 = max(0, min(ny_ - 1, iy0))
    if bool(layer[ix0, iy0]):
        return (ix0, iy0)
    # Spiral outward up to max_search_cells.
    for r in range(1, int(max_search_cells) + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                ix = ix0 + dx
                iy = iy0 + dy
                if not (0 <= ix < nx_ and 0 <= iy < ny_):
                    continue
                if bool(layer[ix, iy]):
                    return (ix, iy)
    return None


def _bfs_distances(
    start: Tuple[int, int],
    layer: np.ndarray,
) -> np.ndarray:
    """4-connected BFS over fluid mask layer. Returns ``(nx, ny)`` int32
    array of cell-step distances; unreachable cells are ``-1``.
    """
    nx_, ny_ = layer.shape
    dist = np.full((nx_, ny_), -1, dtype=np.int32)
    sx, sy = start
    if not bool(layer[sx, sy]):
        return dist
    dist[sx, sy] = 0
    q: deque = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        d0 = int(dist[x, y])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx2 = x + dx
            ny2 = y + dy
            if not (0 <= nx2 < nx_ and 0 <= ny2 < ny_):
                continue
            if dist[nx2, ny2] != -1:
                continue
            if not bool(layer[nx2, ny2]):
                continue
            dist[nx2, ny2] = d0 + 1
            q.append((nx2, ny2))
    return dist


# ─── Graph builder ───────────────────────────────────────────────────────
def build_node_graph(
    *,
    detectors: Optional[Sequence] = None,
    fluid_mask: Optional[np.ndarray] = None,
    k_neighbors: int = 6,
    z_layer: int = 3,
    max_edge_length_m: float = 30.0,
    detour_ratio_max: float = 1.6,
) -> nx.Graph:
    """Build the 39-detector navigation graph (D-047, BFS variant).

    Connectivity is derived from **BFS over the fluid mask** at
    ``z_layer``: for every detector we snap to the nearest fluid cell,
    then run a 4-connected BFS to compute the geodesic distance to all
    other detectors' snap cells. Pairs whose geodesic distance fits
    within ``max_edge_length_m`` and whose detour ratio (geodesic /
    Euclidean) stays below ``detour_ratio_max`` become edges; the edge
    ``length`` is the geodesic distance.

    Each node carries:
      * ``pos``        : ``(x, y, z)`` world metres (taken from the
                         DetectorPosition).
      * ``node_type``  : ``"room" | "corridor" | "exit"``.
      * ``is_exit``    : True iff this detector maps to a currently
                         **active** building exit (D-041 west-only by
                         default).
      * ``detector_id``: the original detector name.
      * ``snap_cell``  : ``(ix, iy)`` fluid cell the detector snapped
                         to. ``None`` if the detector is hopelessly
                         outside the fluid region.

    Each edge carries:
      * ``length`` : geodesic distance in metres (BFS hops × CELL_SIZE_M).

    Args:
        detectors: List of DetectorPosition objects. Defaults to
            ALL_DETECTORS.
        fluid_mask: ``(60, 40, 6)`` boolean. Loaded from disk if None.
        k_neighbors: Each node keeps an edge to up to this many of its
            geodesically-nearest peers, on top of any pair that meets
            the ``detour_ratio_max`` test. Default 6 keeps the graph
            dense enough that the planner has alternatives without
            becoming a near-complete graph.
        z_layer: Z-slice used for BFS (3 = breathing zone, 1.75 m).
        max_edge_length_m: Hard cap on edge geodesic length (m).
        detour_ratio_max: Skip edges whose geodesic / Euclidean ratio
            exceeds this — those routes go "around something" and are
            redundant with edges through intermediate detectors.

    Returns:
        ``networkx.Graph`` with one node per detector.
    """
    if detectors is None:
        detectors = ALL_DETECTORS
    if fluid_mask is None:
        fluid_mask = load_default_fluid_mask()

    nx_, ny_, nz_ = fluid_mask.shape
    if not (0 <= z_layer < nz_):
        raise ValueError(f"z_layer {z_layer} outside fluid mask {fluid_mask.shape}")
    # Use the **union** of fluid cells across breathing-zone z-layers
    # (z=2..z=4 by default, i.e. 1.0 m..2.5 m). Some door openings are
    # only "fluid" at z=2 (door-handle height) and some passages only at
    # z=4 (transom), so taking the OR gives us a connected navigable
    # space at human-walking height. Detectors all still snap to a
    # single layer-merged 2-D grid.
    z_lo = max(0, z_layer - 1)
    z_hi = min(nz_ - 1, z_layer + 1)
    layer = np.any(fluid_mask[:, :, z_lo:z_hi + 1], axis=2)

    g = nx.Graph()

    # 1. Add nodes with snap-cell.
    snap_cells: Dict[str, Optional[Tuple[int, int]]] = {}
    for d in detectors:
        building_name = _DETECTOR_TO_BUILDING_EXIT.get(d.detector_id)
        is_active_exit = (
            building_name is not None
            and building_name in _ACTIVE_EXIT_NAMES
        )
        xy = np.asarray(d.position[:2], dtype=np.float64)
        cell = _snap_to_fluid_cell(xy, layer, max_search_cells=8)
        snap_cells[d.detector_id] = cell
        g.add_node(
            d.detector_id,
            pos=tuple(d.position),
            node_type=d.node_type,
            is_exit=is_active_exit,
            detector_id=d.detector_id,
            snap_cell=cell,
        )

    detector_list = list(detectors)
    positions_xy = np.asarray(
        [d.position[:2] for d in detector_list], dtype=np.float64,
    )
    n = len(detector_list)

    # 2. BFS sweep — one BFS per detector with a valid snap cell.
    bfs_dist: Dict[str, Optional[np.ndarray]] = {}
    for d in detector_list:
        cell = snap_cells[d.detector_id]
        if cell is None:
            bfs_dist[d.detector_id] = None
        else:
            bfs_dist[d.detector_id] = _bfs_distances(cell, layer)

    # 3. Build edges from pairwise geodesic distances.
    eucl_dists = np.linalg.norm(
        positions_xy[:, None, :] - positions_xy[None, :, :], axis=2,
    )
    geo_dist_m = np.full((n, n), np.inf, dtype=np.float64)
    for i, d_i in enumerate(detector_list):
        dist_grid = bfs_dist[d_i.detector_id]
        if dist_grid is None:
            continue
        for j, d_j in enumerate(detector_list):
            if i == j:
                continue
            cell_j = snap_cells[d_j.detector_id]
            if cell_j is None:
                continue
            d_steps = int(dist_grid[cell_j[0], cell_j[1]])
            if d_steps < 0:
                continue    # unreachable
            geo_dist_m[i, j] = float(d_steps) * float(CELL_SIZE_M)

    # 4. Add edges that pass the cap + detour-ratio test, plus the
    # k geodesically-nearest peers as a safety net so every node has
    # at least k neighbours (until distances run out).
    for i, d_i in enumerate(detector_list):
        # All pairs within the cap + detour test.
        for j in range(n):
            if j == i:
                continue
            g_ij = geo_dist_m[i, j]
            if not np.isfinite(g_ij):
                continue
            if g_ij > float(max_edge_length_m):
                continue
            e_ij = float(eucl_dists[i, j])
            # Avoid divide-by-zero for coincident detectors.
            ratio = g_ij / max(e_ij, 0.25)
            if ratio > float(detour_ratio_max):
                continue
            d_j = detector_list[j]
            if not g.has_edge(d_i.detector_id, d_j.detector_id):
                g.add_edge(
                    d_i.detector_id,
                    d_j.detector_id,
                    length=g_ij,
                )
        # k nearest geodesic neighbours (regardless of detour ratio).
        if int(k_neighbors) > 0:
            order = np.argsort(geo_dist_m[i])
            taken = 0
            for j in order:
                j = int(j)
                if j == i:
                    continue
                g_ij = geo_dist_m[i, j]
                if not np.isfinite(g_ij):
                    continue
                if g_ij > float(max_edge_length_m):
                    break    # sorted, so all subsequent are too far
                d_j = detector_list[j]
                if not g.has_edge(d_i.detector_id, d_j.detector_id):
                    g.add_edge(
                        d_i.detector_id,
                        d_j.detector_id,
                        length=g_ij,
                    )
                taken += 1
                if taken >= int(k_neighbors):
                    break

    # 5. Sanity — every node must be reachable from at least one exit.
    exits = [nid for nid, a in g.nodes(data=True) if a.get("is_exit")]
    if not exits:
        return g    # no active exits — caller responsible for handling
    main_component = max(nx.connected_components(g), key=len)
    isolated = [
        nid for nid in g.nodes
        if nid not in main_component and not g.nodes[nid].get("is_exit")
    ]
    if isolated:
        print(
            f"[node_graph] WARN {len(isolated)} node(s) disconnected from "
            f"the main component: {isolated[:6]}..."
        )
    return g


# ─── Planner config ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class NodeGraphPlannerConfig:
    """Edge-weight + heuristic hyperparameters.

    weight(u→v) = base_cost · length + risk_scale · 0.5 · (danger[u] + danger[v])

    Defaults mirror the cell-grid :class:`EdgeWeightConfig` so the two
    planners produce comparable trade-offs.
    """
    base_cost: float = 1.0
    risk_scale: float = 10.0


# ─── Planner ─────────────────────────────────────────────────────────────
class NodeGraphPlanner:
    """A* over the 39-detector graph — drop-in for EvacuationPlanner.

    Plan / replan signature is identical so :class:`DroneSwarm` can use
    either planner without code changes.

    Args:
        graph: Output of :func:`build_node_graph`.
        config: Edge-weight hyperparameters.
        fallback_to_last: If a replan finds no route, return the last
            successful path rather than ``[]``.
    """

    def __init__(
        self,
        graph: nx.Graph,
        config: Optional[NodeGraphPlannerConfig] = None,
        fallback_to_last: bool = True,
    ) -> None:
        self.graph: nx.Graph = graph
        self.config: NodeGraphPlannerConfig = config or NodeGraphPlannerConfig()
        self.fallback_to_last: bool = fallback_to_last
        self._last_path: List[np.ndarray] = []
        # Precompute the active exit node ids.
        self._exit_ids: List[str] = [
            nid for nid, a in graph.nodes(data=True) if a.get("is_exit")
        ]
        # Precompute node positions for fast nearest-node lookup.
        self._node_positions_xy: Dict[str, np.ndarray] = {
            nid: np.asarray(a["pos"], dtype=np.float64)[:2]
            for nid, a in graph.nodes(data=True)
        }

    # ── Public API ──────────────────────────────────────────────────
    def plan(
        self,
        start_xyz: np.ndarray,
        risk_map,
        t: float = 0.0,
        *,
        co_field=None,
    ) -> List[np.ndarray]:
        """Snap → weight → A* → list of (3,) world waypoints.

        The first waypoint is ``start_xyz`` itself (so the caller can
        interpolate from the exact drone position). The rest are node
        centres in path order, ending at the chosen exit.
        """
        del co_field    # accepted for interface compat; weighting is risk-only
        arr = np.asarray(start_xyz, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"start_xyz must be (3,), got {arr.shape}")

        if not self._exit_ids:
            return []

        # 1. Snap start to nearest node (XY).
        start_node = self._nearest_node(arr[:2])

        # 2. Query danger at each node — Tier 1 GNN goes through the
        # ``Tier1RiskMap`` adapter which returns per-node values at the
        # nearest detector, so querying at node positions is exact.
        node_danger: Dict[str, float] = {}
        for nid, pos_xy in self._node_positions_xy.items():
            pos3 = np.array(
                [pos_xy[0], pos_xy[1], float(self.graph.nodes[nid]["pos"][2])],
                dtype=np.float64,
            )
            node_danger[nid] = float(risk_map.query(pos3, t=t))

        # 3. Apply weights on a working copy.
        g_planning = self.graph.copy()
        for u, v, attrs in g_planning.edges(data=True):
            length = float(attrs.get("length", 1.0))
            risk = 0.5 * (node_danger[u] + node_danger[v])
            attrs["weight"] = (
                self.config.base_cost * length
                + self.config.risk_scale * risk
            )

        # 4. A* to every active exit, keep the cheapest.
        best_path: Optional[List[str]] = None
        best_cost = math.inf
        for ex_id in self._exit_ids:
            if not nx.has_path(g_planning, start_node, ex_id):
                continue
            try:
                nodes_path = nx.astar_path(
                    g_planning,
                    source=start_node,
                    target=ex_id,
                    heuristic=lambda u, v: float(
                        np.linalg.norm(
                            self._node_positions_xy[u]
                            - self._node_positions_xy[v]
                        )
                    ),
                    weight="weight",
                )
            except nx.NetworkXNoPath:
                continue
            cost = sum(
                g_planning[u][v]["weight"]
                for u, v in zip(nodes_path[:-1], nodes_path[1:])
            )
            if cost < best_cost:
                best_cost = cost
                best_path = nodes_path

        if best_path is None:
            return []

        # 5. Materialise the path as (3,) waypoints, prepending the
        # exact drone position so the first sub-leg interpolates
        # smoothly from where the drone actually is.
        wp: List[np.ndarray] = [arr.copy()]
        for nid in best_path:
            wp.append(np.asarray(
                self.graph.nodes[nid]["pos"], dtype=np.float64,
            ).reshape(3))
        self._last_path = wp
        return wp

    def replan(
        self,
        current_xyz: np.ndarray,
        risk_map,
        t: float = 0.0,
        *,
        co_field=None,
    ) -> List[np.ndarray]:
        new_path = self.plan(current_xyz, risk_map, t=t, co_field=co_field)
        if not new_path and self.fallback_to_last:
            return self._last_path
        return new_path

    # ── Helpers ─────────────────────────────────────────────────────
    def _nearest_node(self, xy: np.ndarray) -> str:
        best_id: Optional[str] = None
        best_d2 = math.inf
        for nid, pos in self._node_positions_xy.items():
            dx = pos[0] - xy[0]
            dy = pos[1] - xy[1]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_id = nid
        assert best_id is not None    # graph has at least one node by construction
        return best_id


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("node_graph.py self-test (D-047)")
    print("=" * 60)

    errors: list[str] = []

    # 1. Build graph with default detectors
    print("\n[1] build_node_graph defaults")
    g = build_node_graph()
    print(
        f"  nodes={g.number_of_nodes()}  edges={g.number_of_edges()}  "
        f"components={nx.number_connected_components(g)}"
    )
    if g.number_of_nodes() != 39:
        errors.append(f"expected 39 nodes, got {g.number_of_nodes()}")
    exits = [nid for nid, a in g.nodes(data=True) if a.get("is_exit")]
    print(f"  active exits = {exits}")
    if exits != ["exit_1_west"]:
        errors.append(
            f"D-041 expects west-only, got {exits}"
        )

    # 2. Edge attributes
    print("\n[2] edge attributes")
    sample_edge = next(iter(g.edges(data=True)))
    print(f"  sample {sample_edge[0]} ↔ {sample_edge[1]}: {sample_edge[2]}")
    if "length" not in sample_edge[2]:
        errors.append("edge missing 'length' attr")

    # 3. Visibility — west exit reachable from far-side rooms?
    print("\n[3] reachability from a representative node to west exit")
    west = "exit_1_west"
    if west in g:
        ok = sum(
            1 for nid in g.nodes
            if nid != west and nx.has_path(g, nid, west)
        )
        print(f"  {ok} / {g.number_of_nodes() - 1} nodes reach west exit")
        if ok < g.number_of_nodes() // 2:
            errors.append(
                f"only {ok} nodes reach west exit — graph fragmented"
            )

    # 4. Planner round-trip with a zero-risk synthetic map.
    print("\n[4] NodeGraphPlanner with zero-risk synthetic map")
    from src.integration.scenarios._common import zero_risk_map
    rm = zero_risk_map()
    planner = NodeGraphPlanner(g)
    start = np.array([22.0, 5.75, 1.75])     # near east exit / detector
    path = planner.plan(start, rm, t=0.0)
    print(f"  path len = {len(path)}")
    if len(path) < 2:
        errors.append("planner returned an empty / single-waypoint path")
    else:
        end = np.asarray(path[-1])
        west_pos = np.asarray(g.nodes[west]["pos"])
        if float(np.linalg.norm(end[:2] - west_pos[:2])) > 0.5:
            errors.append(
                f"path does not end at west exit: end={end}, west={west_pos}"
            )
        else:
            print(f"  PASS path ends at west exit (len {len(path)} wp)")

    # 5. Planner with a synthetic risk that makes the direct corridor expensive.
    print("\n[5] Planner re-routes around a high-risk node")

    class _BlockMap:
        """Synthetic RiskMap: very high danger at one specific node."""

        def __init__(self, target_id: str):
            self.target = target_id
            self.target_pos = np.asarray(g.nodes[target_id]["pos"])

        def query(self, xyz, t=None):
            xyz_arr = np.asarray(xyz)
            if xyz_arr.ndim == 1:
                d = float(np.linalg.norm(xyz_arr[:2] - self.target_pos[:2]))
                return 0.99 if d < 1.0 else 0.0
            return np.array(
                [0.99 if float(np.linalg.norm(p[:2] - self.target_pos[:2])) < 1.0 else 0.0
                 for p in xyz_arr],
                dtype=np.float32,
            )

    # Find a node that lies on the unblocked greedy path to drop a "fire"
    # on, and verify the re-route differs. The blocked node must be a
    # **true intermediate** (not start, not exit) so the planner has
    # somewhere to detour to.
    # Use a longer-haul start that forces at least one intermediate.
    long_start = np.array([27.0, 15.0, 1.75])     # far north-east area
    direct_path = planner.plan(long_start, rm, t=0.0)
    # direct_path = [start_xyz, node_1, node_2, ..., exit]; first wp is
    # the literal start vector, so node ids start at index 1.
    if len(direct_path) >= 4:
        block_wp = direct_path[len(direct_path) // 2]
        # block_node here is a position; map back to id, but skip the
        # exit and the nearest-to-start so we don't trivially block
        # them.
        block_id = None
        west_pos_xy = np.asarray(g.nodes[west]["pos"])[:2]
        for nid, pos in planner._node_positions_xy.items():
            if nid == west:
                continue
            if float(np.linalg.norm(pos - block_wp[:2])) < 0.1:
                block_id = nid
                break
        if block_id is not None:
            blocked_planner = NodeGraphPlanner(g)
            blocked_path = blocked_planner.plan(
                long_start, _BlockMap(block_id), t=0.0,
            )
            same = (
                len(blocked_path) == len(direct_path)
                and all(
                    float(np.linalg.norm(a - b)) < 0.1
                    for a, b in zip(blocked_path, direct_path)
                )
            )
            print(
                f"  blocked node {block_id!r}: "
                f"direct_len={len(direct_path)}, blocked_len={len(blocked_path)}, "
                f"identical={same}"
            )
            if same:
                errors.append("planner did not re-route around blocked node")
        else:
            print("  (skip: could not identify block_id)")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: node_graph validated")
