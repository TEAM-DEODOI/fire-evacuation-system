"""Building graph adapter for path planning.

The canonical building topology lives in :mod:`src.shared.building` —
a 19-node L-shaped NetworkX graph with 3 exits matching
``docs/interface_contracts.md`` §5 (target 16–20 nodes). This module is
a *thin adapter*: it re-exports that graph in the raw ``nx.Graph`` form
the path-planning algorithms consume, plus a ``snap_to_graph`` helper
that maps an arbitrary world coordinate to the nearest graph node.

Node attributes carried on the returned graph::

    pos         : tuple[float, float, float]  — world metres
    node_type   : "room" | "corridor" | "intersection" | "exit"
    is_exit     : bool
    has_detector: bool
    node        : BuildingNode dataclass (back-pointer)

Edge attributes::

    length: float  — Euclidean distance in metres
    width : float  — corridor width (for bottleneck modelling, default 1.5 m)

Compatibility note: the original ``build_graph(obstacle_mask, exits)``
skeleton presumed a 14 400-cell graph. That design conflicted with
``interface_contracts.md`` §5 and duplicated work already done in
``src.shared.building``. The new signature keeps the same name but drops
the cell-level intent — ``obstacle_mask`` and ``exits`` are accepted for
future flexibility (e.g., custom layouts) and ignored when ``None``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from src.shared.building import BuildingGraph, build_default_graph


def build_graph(
    obstacle_mask: Optional[np.ndarray] = None,
    exits: Optional[List[Tuple[float, float, float]]] = None,
) -> nx.Graph:
    """Return the canonical building graph as a raw :class:`nx.Graph`.

    Args:
        obstacle_mask: Reserved for future use (custom obstacles). When
            supplied, currently raises :class:`NotImplementedError` to
            surface the design departure rather than silently ignoring it.
        exits: Reserved for future use (extra exits). Same behaviour as
            ``obstacle_mask`` when non-``None``.

    Returns:
        Undirected :class:`nx.Graph` whose nodes are string IDs (e.g.
        ``"hall_n"``) carrying ``pos``, ``node_type``, ``is_exit``,
        ``has_detector`` attributes. Edges carry ``length`` and ``width``.

    Raises:
        NotImplementedError: If a non-``None`` ``obstacle_mask`` or
            ``exits`` is passed — the L-shape layout is fixed by
            ``shared.building`` and overriding requires a deliberate
            schema change (see ``docs/decisions.md`` D-016).
    """
    if obstacle_mask is not None:
        raise NotImplementedError(
            "Custom obstacle masks not supported — building topology is "
            "fixed in src.shared.building.build_default_graph(). Modify "
            "_DEFAULT_NODES/_DEFAULT_EDGES there if a layout change is needed."
        )
    if exits is not None:
        raise NotImplementedError(
            "Custom exit lists not supported — exits are defined in "
            "src.shared.building._DEFAULT_NODES with is_exit=True."
        )
    return build_default_graph().graph


def add_exit_nodes(
    graph: nx.Graph,
    exits: List[Tuple[float, float, float]],
) -> nx.Graph:
    """Mark exits in an existing graph (idempotent).

    For the canonical building layout this is essentially a no-op because
    :func:`build_graph` already tags the three exits. The function exists
    to honour the skeleton contract and to allow injecting exits in
    custom layouts: each ``(x, y, z)`` is snapped to the nearest existing
    graph node, which then has ``is_exit=True`` set.

    Args:
        graph: NetworkX graph from :func:`build_graph`.
        exits: World coordinates ``[(x, y, z), ...]``.

    Returns:
        Same graph object (mutated in place).
    """
    if not exits:
        return graph
    for xyz in exits:
        nid = snap_to_graph(np.asarray(xyz, dtype=np.float64), graph)
        graph.nodes[nid]["is_exit"] = True
        bn = graph.nodes[nid].get("node")
        if bn is not None:
            # BuildingNode is frozen — replace it with an exit-tagged copy.
            from dataclasses import replace as _replace
            graph.nodes[nid]["node"] = _replace(bn, is_exit=True)
    return graph


def snap_to_graph(xyz: np.ndarray, graph: nx.Graph) -> str:
    """Return the ID of the graph node closest to ``xyz`` (XY distance).

    Single floor → Z is ignored, matching
    :meth:`src.shared.building.BuildingGraph.nearest_node` semantics.

    Args:
        xyz: ``(3,)`` world coordinate in metres.
        graph: NetworkX graph with ``pos`` node attributes.

    Returns:
        Node ID of the closest node.

    Raises:
        ValueError: If ``xyz`` is not 3-D or ``graph`` is empty.
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"xyz must have shape (3,), got {arr.shape}")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph is empty — no nodes to snap to")

    best_id: Optional[str] = None
    best_d2 = float("inf")
    for nid, attrs in graph.nodes(data=True):
        pos = attrs.get("pos")
        if pos is None:
            continue
        dx = pos[0] - arr[0]
        dy = pos[1] - arr[1]
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_id = nid
    if best_id is None:
        raise ValueError("graph nodes are missing 'pos' attributes")
    return best_id


def exit_nodes(graph: nx.Graph) -> List[str]:
    """Return all node IDs where ``is_exit=True``."""
    return [nid for nid, a in graph.nodes(data=True) if a.get("is_exit")]


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("building_graph.py self-test")
    print("=" * 60)

    errors: List[str] = []

    # ── 1. build_graph returns connected nx.Graph ──────────────────────
    print("\n[1] build_graph() returns connected nx.Graph")
    g = build_graph()
    if not isinstance(g, nx.Graph):
        errors.append(f"build_graph returned {type(g).__name__}, not nx.Graph")
    if not nx.is_connected(g):
        errors.append("graph not connected")
    print(f"  nodes={g.number_of_nodes()} edges={g.number_of_edges()}")
    if not (16 <= g.number_of_nodes() <= 20):
        errors.append(f"node count {g.number_of_nodes()} outside [16, 20]")

    # ── 2. Node attributes present ──────────────────────────────────────
    print("\n[2] Required node attributes present")
    for nid, a in g.nodes(data=True):
        for key in ("pos", "node_type", "is_exit"):
            if key not in a:
                errors.append(f"node {nid} missing attribute {key!r}")
                break

    # ── 3. Three exits ─────────────────────────────────────────────────
    print("\n[3] Exactly 3 exits")
    ex = exit_nodes(g)
    print(f"  exits = {ex}")
    if len(ex) != 3:
        errors.append(f"exit count {len(ex)} != 3")

    # ── 4. Edge attributes ─────────────────────────────────────────────
    print("\n[4] Every edge has positive length")
    bad = [
        (u, v) for u, v, a in g.edges(data=True)
        if not a.get("length", 0.0) > 0.0
    ]
    if bad:
        errors.append(f"{len(bad)} edges with non-positive length: {bad[:3]}")

    # ── 5. snap_to_graph ───────────────────────────────────────────────
    print("\n[5] snap_to_graph centre vs corner")
    nid_centre = snap_to_graph(np.array([15.0, 10.0, 1.5]), g)
    nid_corner = snap_to_graph(np.array([0.0, 0.0, 1.5]), g)
    print(f"  (15, 10) → {nid_centre}")
    print(f"  ( 0,  0) → {nid_corner}")
    if nid_centre not in {"hall_n", "hall_s", "hall_e", "hall_w"}:
        errors.append(f"centre snap → unexpected {nid_centre}")
    if nid_corner not in {"exit_west", "zone_b_west"}:
        errors.append(f"corner snap → unexpected {nid_corner}")

    # ── 6. snap_to_graph shape validation ───────────────────────────────
    print("\n[6] snap_to_graph rejects bad shapes")
    try:
        snap_to_graph(np.array([1.0, 2.0]), g)
    except ValueError:
        print("  PASS: (2,) → ValueError")
    else:
        errors.append("(2,) input did not raise ValueError")

    # ── 7. add_exit_nodes idempotency ──────────────────────────────────
    print("\n[7] add_exit_nodes is idempotent for existing exits")
    before = set(exit_nodes(g))
    add_exit_nodes(g, [(0.0, 5.0, 1.5)])  # exit_west position
    after = set(exit_nodes(g))
    if before != after:
        errors.append(
            f"add_exit_nodes mutated exit set: {before} -> {after}"
        )
    else:
        print("  PASS")

    # ── 8. build_graph rejects custom mask/exits ────────────────────────
    print("\n[8] build_graph rejects unsupported overrides")
    for kw in ({"obstacle_mask": np.zeros((60, 40, 6), dtype=bool)},
               {"exits": [(1.0, 1.0, 1.5)]}):
        try:
            build_graph(**kw)
        except NotImplementedError:
            pass
        else:
            errors.append(f"build_graph did not reject {list(kw)[0]}")
    print("  PASS")

    # ── Verdict ────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: building_graph adapter validated")
