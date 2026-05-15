"""Diagnose why S2's drone plan crosses high-risk cells.

Loads s_029 truth RiskMap and runs the actual EvacuationPlanner used by
S2 (same EdgeWeightConfig) from each canonical drone spawn position to
each exit. For each plan we print:

  - path length (# cells, m)
  - cell-by-cell risk along the path
  - max / mean / above-threshold counts

If drone path crosses cells with risk >> 0 while neighbouring lower-risk
cells exist, the planner is failing despite the risk-aware weighting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.integration.scenarios._common import (
    load_truth_risk_map, exit_positions, drone_spawn_positions,
)
from src.path_planning.building_graph import build_graph, load_default_fluid_mask
from src.path_planning.edge_weights import EdgeWeightConfig
from src.path_planning.planners import EvacuationPlanner
from src.shared.constants import CELL_SIZE_M


def main() -> int:
    fds_dir = Path("data/raw/s_029")
    truth_rm = load_truth_risk_map(fds_dir, verbose=False)
    fluid_mask = load_default_fluid_mask()
    exits = exit_positions()
    drone_spawns = drone_spawn_positions()

    # Diagnostic: risk_scale extreme (1000) to test whether the path
    # actually has a detour, or the suspect cells are the only route.
    cfg = EdgeWeightConfig(
        base_cost=1.0,
        risk_scale=1000.0,
        risk_threshold=1.0,
        n_samples=5,
    )

    graph = build_graph(fluid_mask=fluid_mask)
    planner = EvacuationPlanner(
        graph, config=cfg, heuristic="euclidean", fallback_to_last=True,
    )

    # ── 1. Risk distribution across the navigable layer (z=3, t=120s) ──
    print("=" * 70)
    print("s_029 truth RM at t=120s -- risk distribution over fluid layer z=3")
    print("=" * 70)
    layer = fluid_mask[:, :, 3]
    cells = [(i, j, 3) for i in range(layer.shape[0])
             for j in range(layer.shape[1]) if layer[i, j]]
    pts = np.array([[0.25 + CELL_SIZE_M * i,
                     0.25 + CELL_SIZE_M * j,
                     0.25 + CELL_SIZE_M * 3] for (i, j, _k) in cells],
                   dtype=np.float64)
    risks = np.asarray(truth_rm.query(pts, t=120.0), dtype=np.float64)
    print(f"  fluid cells:     {len(cells)}")
    print(f"  risk min/mean/max:  {risks.min():.3f} / {risks.mean():.3f} / {risks.max():.3f}")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q:>2d}:            {float(np.percentile(risks, q)):.3f}")
    high = (risks > 0.5).sum()
    print(f"  cells with risk > 0.5:  {high}/{len(cells)} ({100.0*high/len(cells):.1f}%)")

    # ── 2. Plan from each drone spawn -> each exit, dump risk along path ─
    print("\n" + "=" * 70)
    print("Drone plans (truth RM, t=120s)")
    print("=" * 70)
    for spawn_i, spawn in enumerate(drone_spawns):
        for exit_j, exit_xyz in enumerate(exits):
            path = planner.plan(spawn, truth_rm, t=120.0)
            if not path:
                print(f"  spawn {spawn_i} -> exit {exit_j}: NO PATH")
                continue
            path_arr = np.asarray(path, dtype=np.float64)
            # last waypoint should be near exit
            d_to_exit = float(np.linalg.norm(path_arr[-1, :2] - exit_xyz[:2]))
            cell_risks = np.asarray(
                truth_rm.query(path_arr, t=120.0), dtype=np.float64
            )
            total_len = 0.0
            for k in range(len(path_arr) - 1):
                total_len += float(np.linalg.norm(path_arr[k + 1, :2] - path_arr[k, :2]))
            print(
                f"\n  spawn {spawn_i} {tuple(np.round(spawn, 2))} -> "
                f"exit {exit_j} {tuple(np.round(exit_xyz, 2))}"
            )
            print(
                f"    waypoints: {len(path)}  length: {total_len:.1f} m  "
                f"end_to_exit_dist: {d_to_exit:.2f} m"
            )
            print(
                f"    path risk:  min {cell_risks.min():.3f}  "
                f"mean {cell_risks.mean():.3f}  max {cell_risks.max():.3f}"
            )
            hi_cells = np.where(cell_risks > 0.3)[0]
            if len(hi_cells):
                print(f"    cells with risk > 0.3 along path: {len(hi_cells)}/{len(path)}")
                for idx in hi_cells[:8]:
                    p = path_arr[idx]
                    print(f"      wp[{idx:>3d}]  xyz=({p[0]:5.2f},{p[1]:5.2f},{p[2]:4.2f})  "
                          f"risk={cell_risks[idx]:.3f}")

    # ── 3. Compute weight at one specific suspicious edge ───────────────
    print("\n" + "=" * 70)
    print("Sample edge weights (a high-risk cell vs a low-risk neighbour)")
    print("=" * 70)
    # Find the highest-risk fluid cell and one of its low-risk neighbours.
    hi_idx = int(np.argmax(risks))
    hi_cell = cells[hi_idx]
    hi_pos = pts[hi_idx]
    print(f"  highest-risk cell {hi_cell} at xyz=({hi_pos[0]:.2f},{hi_pos[1]:.2f},{hi_pos[2]:.2f})  risk={risks[hi_idx]:.3f}")
    # Examine all 8 neighbours.
    neighbours = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            nb = (hi_cell[0] + di, hi_cell[1] + dj, 3)
            if nb in graph.nodes:
                nb_pos = np.asarray(graph.nodes[nb]["pos"], dtype=np.float64)
                nb_risk = float(truth_rm.query(nb_pos, t=120.0))
                edge_risk = 0.5 * (risks[hi_idx] + nb_risk)
                length = float(np.linalg.norm(nb_pos[:2] - hi_pos[:2]))
                weight = cfg.base_cost * length + cfg.risk_scale * edge_risk
                neighbours.append((nb, nb_risk, edge_risk, length, weight))
    neighbours.sort(key=lambda r: r[1])
    for nb, nb_risk, edge_risk, length, weight in neighbours:
        print(
            f"    nb {nb}  risk={nb_risk:.3f}  edge_risk={edge_risk:.3f}  "
            f"length={length:.3f}  weight={weight:.3f}"
        )

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
