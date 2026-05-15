"""Compare a length-only BFS path against the risk-aware A* path
between drone spawn 2 and exit 0 (s_029, t=120s).

If both paths visit the same cells, the graph topology forces those
cells — no alternative corridor exists in the cell-grid graph.
If they differ, the planner is leaving free detours on the table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.integration.scenarios._common import load_truth_risk_map
from src.path_planning.building_graph import build_graph, load_default_fluid_mask
from src.path_planning.edge_weights import EdgeWeightConfig
from src.path_planning.planners import EvacuationPlanner
from src.shared.constants import CELL_SIZE_M


def path_to_cells(path):
    return [
        (int(p[0] / CELL_SIZE_M), int(p[1] / CELL_SIZE_M))
        for p in np.asarray(path, dtype=np.float64)
    ]


def main() -> int:
    fds_dir = Path("data/raw/s_029")
    truth_rm = load_truth_risk_map(fds_dir, verbose=False)
    fluid_mask = load_default_fluid_mask()
    graph = build_graph(fluid_mask=fluid_mask)

    spawn = np.array([22.0, 6.0, 1.5])
    exit_xyz = np.array([2.75, 5.75, 1.75])

    # 1. Length-only path (risk_scale=0): planner should pick the shortest geometric path
    planner_len = EvacuationPlanner(
        graph,
        config=EdgeWeightConfig(base_cost=1.0, risk_scale=0.0, risk_threshold=1.0),
        heuristic="euclidean",
        fallback_to_last=True,
    )
    # 2. Risk-aware path (risk_scale=1000)
    planner_risk = EvacuationPlanner(
        graph,
        config=EdgeWeightConfig(base_cost=1.0, risk_scale=1000.0, risk_threshold=1.0),
        heuristic="euclidean",
        fallback_to_last=True,
    )

    path_len = planner_len.plan(spawn, truth_rm, t=120.0)
    path_risk = planner_risk.plan(spawn, truth_rm, t=120.0)
    cells_len = path_to_cells(path_len)
    cells_risk = path_to_cells(path_risk)

    risks_len = np.asarray(truth_rm.query(np.asarray(path_len), t=120.0))
    risks_risk = np.asarray(truth_rm.query(np.asarray(path_risk), t=120.0))

    print(f"length-only: {len(path_len)} wp, risk mean={risks_len.mean():.3f} max={risks_len.max():.3f}")
    print(f"risk-aware:  {len(path_risk)} wp, risk mean={risks_risk.mean():.3f} max={risks_risk.max():.3f}")
    same = cells_len == cells_risk
    print(f"same cell sequence: {same}")
    if not same:
        common = sum(1 for a, b in zip(cells_len, cells_risk) if a == b)
        print(f"  common cells: {common}/{max(len(cells_len), len(cells_risk))}")
        # Diff in first 40 cells
        for k in range(min(40, max(len(cells_len), len(cells_risk)))):
            a = cells_len[k] if k < len(cells_len) else None
            b = cells_risk[k] if k < len(cells_risk) else None
            tag = "" if a == b else "  <-- differ"
            print(f"  wp[{k:>2d}]  len={a}  risk={b}{tag}")

    # ── BFS over "safe" cells only (risk < 0.05 at t=120s) ─────────────
    import networkx as nx
    # Build a sub-graph keeping only nodes with risk < 0.05.
    safe_nodes = []
    for n, attrs in graph.nodes(data=True):
        pos = np.asarray(attrs["pos"])
        r = float(truth_rm.query(pos, t=120.0))
        if r < 0.05:
            safe_nodes.append(n)
    sub = graph.subgraph(safe_nodes).copy()
    print(f"\nSafe subgraph: {len(safe_nodes)}/{graph.number_of_nodes()} nodes (risk < 0.05)")

    # Find the nearest safe node to spawn and to exit and check connectivity.
    spawn_cell = (int(spawn[0] / CELL_SIZE_M), int(spawn[1] / CELL_SIZE_M), 3)
    exit_cell  = (int(exit_xyz[0] / CELL_SIZE_M),
                  int(exit_xyz[1] / CELL_SIZE_M), 3)
    print(f"spawn cell: {spawn_cell}  in safe subgraph: {spawn_cell in sub.nodes}")
    print(f"exit cell:  {exit_cell}   in safe subgraph: {exit_cell in sub.nodes}")

    # Treat reachability between connected components.
    def comp_of(g, n):
        if n not in g.nodes:
            return None
        for cc in nx.connected_components(g):
            if n in cc:
                return cc
        return None

    spawn_comp = comp_of(sub, spawn_cell)
    exit_comp = comp_of(sub, exit_cell)
    print(f"spawn safe component size: {len(spawn_comp) if spawn_comp else 0}")
    print(f"exit  safe component size: {len(exit_comp)  if exit_comp  else 0}")
    if spawn_comp and exit_comp:
        if spawn_comp is exit_comp or (
            isinstance(spawn_comp, set) and isinstance(exit_comp, set)
            and spawn_comp == exit_comp
        ):
            print("  -> spawn and exit ARE in the same safe component")
            try:
                p = nx.shortest_path(sub, spawn_cell, exit_cell)
                print(f"  safe-only path: {len(p)} cells")
            except nx.NetworkXNoPath:
                print("  (no safe path despite same component? unexpected)")
        else:
            print("  -> spawn and exit are NOT in the same safe component")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
