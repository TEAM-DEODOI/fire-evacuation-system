"""Inspect truth CO field for s_029 and verify FED-aware planning works.

Goal: confirm whether (a) the truth CO field has non-trivial values on
the S2-suspect cells, and (b) the FED-aware planner config actually
chooses a different path than the risk-only planner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.integration.scenarios._common import (
    load_truth_risk_map, load_truth_co_field,
)
from src.path_planning.building_graph import build_graph, load_default_fluid_mask
from src.path_planning.edge_weights import EdgeWeightConfig, compute_edge_weights
from src.path_planning.planners import EvacuationPlanner
from src.shared.constants import CELL_SIZE_M


def path_cells(path):
    return [
        (int(p[0] / CELL_SIZE_M), int(p[1] / CELL_SIZE_M))
        for p in np.asarray(path, dtype=np.float64)
    ]


def main() -> int:
    fds_dir = Path("data/raw/s_029")
    truth_rm = load_truth_risk_map(fds_dir, verbose=False)
    truth_co = load_truth_co_field(fds_dir, verbose=False)
    fluid_mask = load_default_fluid_mask()
    graph = build_graph(fluid_mask=fluid_mask)

    # ── 1. CO field distribution at t=120s on the breathing-zone layer ──
    layer = fluid_mask[:, :, 3]
    cells = [(i, j, 3) for i in range(layer.shape[0])
             for j in range(layer.shape[1]) if layer[i, j]]
    pts = np.array([[0.25 + CELL_SIZE_M * i,
                     0.25 + CELL_SIZE_M * j,
                     0.25 + CELL_SIZE_M * 3] for (i, j, _k) in cells],
                   dtype=np.float64)
    co_vals = np.asarray(truth_co.query(pts, t=120.0), dtype=np.float64)
    risks = np.asarray(truth_rm.query(pts, t=120.0), dtype=np.float64)
    print("=" * 70)
    print("Truth CO ppm distribution over fluid layer z=3 at t=120s")
    print("=" * 70)
    print(f"  cells:                 {len(cells)}")
    print(f"  CO min/mean/max:       {co_vals.min():.1f} / {co_vals.mean():.1f} / {co_vals.max():.1f} ppm")
    for q in (50, 75, 90, 95, 99):
        print(f"  CO p{q:>2d}:               {float(np.percentile(co_vals, q)):.1f} ppm")
    print(f"  cells with CO > 100 ppm: {(co_vals > 100).sum()}/{len(cells)}")
    print(f"  cells with CO > 50 ppm:  {(co_vals > 50).sum()}/{len(cells)}")

    # Specific suspect cells on the S2 path
    suspects = [(32, 11, 3), (31, 11, 3), (30, 11, 3), (29, 10, 3), (28, 9, 3)]
    print("\n  Suspect cells (S2 path wp 12-16) CO + risk:")
    for cell in suspects:
        i, j, k = cell
        wx = 0.25 + CELL_SIZE_M * i
        wy = 0.25 + CELL_SIZE_M * j
        wz = 0.25 + CELL_SIZE_M * k
        pos = np.array([wx, wy, wz])
        co = float(truth_co.query(pos, t=120.0))
        r = float(truth_rm.query(pos, t=120.0))
        print(f"    {cell}  xyz=({wx:.2f},{wy:.2f})  CO={co:7.1f} ppm  risk={r:.3f}")

    # ── 2. Compute edge weights with current S2 config -- see how big the FED term is ─
    cfg_fed = EdgeWeightConfig(
        base_cost=0.001, fed_scale=1_000_000.0, walking_speed_mps=1.2,
        risk_scale=0.0, risk_threshold=1.0,
    )
    print("\n" + "=" * 70)
    print("Edge-weight breakdown on S2-suspect edges at t=120s")
    print("=" * 70)
    compute_edge_weights(graph, truth_rm, t=120.0, config=cfg_fed, co_field=truth_co)
    suspect_edges = [
        ((32, 11, 3), (31, 11, 3)),
        ((31, 11, 3), (30, 11, 3)),
        ((30, 11, 3), (29, 10, 3)),
    ]
    for u, v in suspect_edges:
        if u in graph and v in graph[u]:
            a = graph[u][v]
            print(f"  edge {u} -> {v}:")
            print(f"    length={a['length']:.3f}  edge_risk={a['edge_risk']:.3f}  edge_fed={a['edge_fed']:.4f}  weight={a['weight']:.3f}")
            comp_len = cfg_fed.base_cost * a['length']
            comp_fed = cfg_fed.fed_scale * a['edge_fed']
            comp_risk = cfg_fed.risk_scale * a['edge_risk']
            print(f"    contributions:  length={comp_len:.3f}  fed_term={comp_fed:.3f}  risk_term={comp_risk:.3f}")

    # ── 3. Compare actual paths under FED-mode vs risk-only ────────────
    spawn = np.array([22.0, 6.0, 1.5])

    planner_fed = EvacuationPlanner(
        graph, config=cfg_fed, heuristic="euclidean", fallback_to_last=True,
    )
    planner_risk = EvacuationPlanner(
        graph,
        config=EdgeWeightConfig(base_cost=1.0, risk_scale=10.0, risk_threshold=1.0),
        heuristic="euclidean", fallback_to_last=True,
    )

    p_fed = planner_fed.plan(spawn, truth_rm, t=120.0, co_field=truth_co)
    p_risk = planner_risk.plan(spawn, truth_rm, t=120.0)

    arr_fed = np.asarray(p_fed)
    arr_risk = np.asarray(p_risk)
    fed_truth_co = np.asarray(truth_co.query(arr_fed, t=120.0))
    risk_truth_co = np.asarray(truth_co.query(arr_risk, t=120.0))
    fed_truth_risk = np.asarray(truth_rm.query(arr_fed, t=120.0))
    risk_truth_risk = np.asarray(truth_rm.query(arr_risk, t=120.0))

    print("\n" + "=" * 70)
    print("Path comparison: FED-mode planner vs risk-only planner")
    print("=" * 70)
    print(f"  FED-mode path:  {len(p_fed)} wp  CO sum={fed_truth_co.sum():.1f}  CO max={fed_truth_co.max():.1f}  risk sum={fed_truth_risk.sum():.2f}")
    print(f"  risk-only path: {len(p_risk)} wp  CO sum={risk_truth_co.sum():.1f}  CO max={risk_truth_co.max():.1f}  risk sum={risk_truth_risk.sum():.2f}")
    same_cells = path_cells(p_fed) == path_cells(p_risk)
    print(f"  SAME path: {same_cells}")
    if not same_cells:
        cells_fed = path_cells(p_fed)
        cells_risk = path_cells(p_risk)
        for k in range(min(40, max(len(cells_fed), len(cells_risk)))):
            a = cells_fed[k] if k < len(cells_fed) else None
            b = cells_risk[k] if k < len(cells_risk) else None
            tag = "" if a == b else "  <-- differ"
            print(f"    wp[{k:>2d}]  fed={a}  risk={b}{tag}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
