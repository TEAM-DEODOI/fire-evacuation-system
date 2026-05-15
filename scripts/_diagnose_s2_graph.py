"""Show whether the wp 12-16 cells in s_029 path have low-risk detours
available in the cell-grid graph.

For each suspicious cell along the drone path we:
  1. List its graph neighbours (8-connected) that ARE in the graph
  2. Report their risk at t=120s
  3. Try a y-strip scan: across the full Y axis at the same X, mark
     which cells are in the graph and which are not, plus their risk.

If a low-risk alternate corridor exists at some y offset, the graph
nodes should be present there; if the whole y-column is one narrow row
of fluid cells, no detour is possible — the planner is right.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.integration.scenarios._common import load_truth_risk_map
from src.path_planning.building_graph import build_graph, load_default_fluid_mask
from src.shared.constants import CELL_SIZE_M


def main() -> int:
    fds_dir = Path("data/raw/s_029")
    truth_rm = load_truth_risk_map(fds_dir, verbose=False)
    fluid_mask = load_default_fluid_mask()
    graph = build_graph(fluid_mask=fluid_mask)
    z_layer = 3

    # The cells the drone path crosses at risk > 0.3 (from earlier diagnostic).
    suspects = [
        (33, 11, z_layer),   # 16.75 -- actually (16.25, 5.75) -> (32, 11)
        (32, 11, z_layer),   # (16.25, 5.75)  wp[12]
        (31, 11, z_layer),   # (15.75, 5.75)  wp[13]
        (30, 11, z_layer),   # (15.25, 5.75)  wp[14]
        (29, 10, z_layer),   # (14.75, 5.25)  wp[15]
        (28,  9, z_layer),   # (14.25, 4.75)  wp[16]
    ]

    print("=" * 70)
    print("Suspect cells (cell_idx, world_xyz, risk, in_graph?, neighbours)")
    print("=" * 70)
    for cell in suspects:
        i, j, k = cell
        wx = 0.25 + CELL_SIZE_M * i
        wy = 0.25 + CELL_SIZE_M * j
        wz = 0.25 + CELL_SIZE_M * k
        in_g = cell in graph.nodes
        risk = float(truth_rm.query(np.array([wx, wy, wz]), t=120.0))
        print(f"\n  cell {cell}  xyz=({wx:.2f},{wy:.2f},{wz:.2f})  risk={risk:.3f}  in_graph={in_g}")
        if in_g:
            nbrs = sorted(graph.neighbors(cell))
            print(f"    {len(nbrs)} graph neighbours:")
            for nb in nbrs:
                ni, nj, nk = nb
                wnx = 0.25 + CELL_SIZE_M * ni
                wny = 0.25 + CELL_SIZE_M * nj
                nrisk = float(truth_rm.query(
                    np.array([wnx, wny, wz]), t=120.0
                ))
                print(f"      {nb}  xyz=({wnx:.2f},{wny:.2f})  risk={nrisk:.3f}")

    # Y-strip scan: for each X column of a suspect, print every j cell
    # along Y from j=2 .. j=20 (covers the building width).
    print("\n" + "=" * 70)
    print("Y-strip scan at the suspect X columns (▮ = in graph, · = no fluid)")
    print("=" * 70)
    for x_col in (44, 40, 38, 36, 34, 32, 31, 30, 29, 20, 10, 5):
        print(f"\n  X column i={x_col}  (world x = {0.25 + CELL_SIZE_M * x_col:.2f} m)")
        for j in range(2, 22):
            cell = (x_col, j, z_layer)
            wnx = 0.25 + CELL_SIZE_M * x_col
            wny = 0.25 + CELL_SIZE_M * j
            in_g = cell in graph.nodes
            marker = "GRAPH" if in_g else " wall"
            risk_str = ""
            if in_g:
                r = float(truth_rm.query(
                    np.array([wnx, wny, 0.25 + CELL_SIZE_M * z_layer]),
                    t=120.0,
                ))
                risk_str = f"  risk={r:.3f}"
            print(f"    j={j:>2d}  y={wny:5.2f}  {marker}{risk_str}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
