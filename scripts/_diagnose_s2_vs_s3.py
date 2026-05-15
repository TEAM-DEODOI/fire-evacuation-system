"""Side-by-side comparison of S2 vs S3 paths for spawn 2 -> exit 0 in s_029.

Builds a Tier1RiskMap (v6, S3's planner_rm) and a truth StaticRiskMap
(S2's planner_rm). Runs the SAME planner on both. For each path:
  - lists the (i, j) cell sequence
  - shows BOTH the truth-RM and model-RM risk at each cell
  - computes cumulative risk under each RM

Goal: see whether S3's path exists in the graph (it must — same graph)
and whether the truth RM assigns it a HIGHER cumulative risk than the
S2 path. If yes: A* is optimal under truth RM and the issue is the
truth RM's spatial distribution, not a bug. If no: A* is failing to
find the optimal path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from src.integration.scenarios._common import load_truth_risk_map
from src.integration.scenarios.s3_fno_swarm import (
    _load_tier1_artefacts, _build_tier1_rm,
)
from src.path_planning.building_graph import build_graph, load_default_fluid_mask
from src.path_planning.edge_weights import EdgeWeightConfig
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
    fluid_mask = load_default_fluid_mask()
    graph = build_graph(fluid_mask=fluid_mask)

    # Build v6 GNN forward at t0=120s (same as S3 would at t_fire=120).
    art = _load_tier1_artefacts(
        "s_029", ckpt_path=Path("checkpoints/tier1_gnn_v6/best.pt"),
        verbose=False,
    )
    if art is None:
        print("FAIL: could not load v6 artefacts")
        return 1
    model_rm = _build_tier1_rm(art, t0_s=120.0, fluid_mask=fluid_mask, verbose=False)
    if model_rm is None:
        print("FAIL: could not build model RM")
        return 1

    spawn = np.array([22.0, 6.0, 1.5])
    exit_xyz = np.array([2.75, 5.75, 1.75])

    cfg_s2 = EdgeWeightConfig(  # S2 currently
        base_cost=1.0, risk_scale=50.0, risk_threshold=1.0,
    )
    cfg_s3 = EdgeWeightConfig(  # S3 currently
        base_cost=1.0, risk_scale=10.0, risk_threshold=1.0,
    )
    planner_truth = EvacuationPlanner(
        graph, config=cfg_s2, heuristic="euclidean", fallback_to_last=True,
    )
    planner_model = EvacuationPlanner(
        graph, config=cfg_s3, heuristic="euclidean", fallback_to_last=True,
    )

    # S2 path: planner consults truth RM
    path_s2 = planner_truth.plan(spawn, truth_rm, t=120.0)
    # S3 path: planner consults v6 model RM
    path_s3 = planner_model.plan(spawn, model_rm, t=120.0)

    cells_s2 = path_cells(path_s2)
    cells_s3 = path_cells(path_s3)

    arr_s2 = np.asarray(path_s2, dtype=np.float64)
    arr_s3 = np.asarray(path_s3, dtype=np.float64)
    s2_truth_risks = np.asarray(truth_rm.query(arr_s2, t=120.0), dtype=np.float64)
    s2_model_risks = np.asarray(model_rm.query(arr_s2, t=120.0), dtype=np.float64)
    s3_truth_risks = np.asarray(truth_rm.query(arr_s3, t=120.0), dtype=np.float64)
    s3_model_risks = np.asarray(model_rm.query(arr_s3, t=120.0), dtype=np.float64)

    def edge_risk_sum(risks):
        # Match compute_edge_weights: edge_risk = 0.5*(r[u]+r[v]).
        return float(0.5 * (risks[:-1] + risks[1:]).sum())

    print("=" * 70)
    print("S2 path (planner: truth RM, risk_scale=50)")
    print("=" * 70)
    print(f"  waypoints: {len(path_s2)}")
    print(f"  truth-RM risks along path:  mean {s2_truth_risks.mean():.3f}  max {s2_truth_risks.max():.3f}  sum {s2_truth_risks.sum():.2f}")
    print(f"  model-RM risks along path:  mean {s2_model_risks.mean():.3f}  max {s2_model_risks.max():.3f}  sum {s2_model_risks.sum():.2f}")
    print(f"  truth-RM cumulative EDGE risk: {edge_risk_sum(s2_truth_risks):.2f}")

    print("\n" + "=" * 70)
    print("S3 path (planner: v6 model RM, risk_scale=10)")
    print("=" * 70)
    print(f"  waypoints: {len(path_s3)}")
    print(f"  truth-RM risks along path:  mean {s3_truth_risks.mean():.3f}  max {s3_truth_risks.max():.3f}  sum {s3_truth_risks.sum():.2f}")
    print(f"  model-RM risks along path:  mean {s3_model_risks.mean():.3f}  max {s3_model_risks.max():.3f}  sum {s3_model_risks.sum():.2f}")
    print(f"  truth-RM cumulative EDGE risk: {edge_risk_sum(s3_truth_risks):.2f}")

    # ─── Decisive check: compute S2 weight (length+risk_scale·edge_risk)
    #     for BOTH paths under the SAME (truth RM, risk_scale=50) config.
    #     If the S3 path's total weight is LOWER than S2's, A* failed.
    #     If it's HIGHER, S2's choice is genuinely optimal under truth RM.
    def total_weight(path, risks, base_cost=1.0, risk_scale=50.0):
        arr = np.asarray(path, dtype=np.float64)
        seg_lens = np.linalg.norm(arr[1:, :2] - arr[:-1, :2], axis=1)
        edge_r = 0.5 * (risks[:-1] + risks[1:])
        return float((base_cost * seg_lens + risk_scale * edge_r).sum())

    w_s2 = total_weight(path_s2, s2_truth_risks)
    w_s3_under_truth = total_weight(path_s3, s3_truth_risks)
    print("\n" + "=" * 70)
    print("Optimality check under truth-RM with risk_scale=50:")
    print("=" * 70)
    print(f"  total weight of S2 path:                 {w_s2:.2f}")
    print(f"  total weight of S3 path (re-scored):     {w_s3_under_truth:.2f}")
    if w_s3_under_truth < w_s2 - 1e-6:
        print("  >>> A* FAILED: S3's path has lower weight under truth RM")
    else:
        print("  >>> A* optimal: S3's path is MORE expensive under truth RM")
        print(f"      delta = {w_s3_under_truth - w_s2:.2f}")

    # Print cell-by-cell comparison
    print("\n" + "=" * 70)
    print("Cell-by-cell  ( i, j )  truth_risk   model_risk")
    print("=" * 70)
    print("\nS2 path:")
    for k, (c, tr, mr) in enumerate(zip(cells_s2, s2_truth_risks, s2_model_risks)):
        tag = "  *" if tr > 0.3 else ""
        print(f"  wp[{k:>2d}]  {c}  truth={tr:.3f}  model={mr:.3f}{tag}")

    print("\nS3 path:")
    for k, (c, tr, mr) in enumerate(zip(cells_s3, s3_truth_risks, s3_model_risks)):
        tag = "  *" if tr > 0.3 else ""
        print(f"  wp[{k:>2d}]  {c}  truth={tr:.3f}  model={mr:.3f}{tag}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
