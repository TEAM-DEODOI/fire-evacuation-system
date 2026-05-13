"""3-way Ensemble — Tier 1 GNN + Sparse ConvLSTM + Sparse FNO.

각 시나리오에 대해 3 모델 forward → 가중 평균.
w_t1 + w_conv + w_fno = 1.0 제약 + grid search.

Sweep:
  w_t1 ∈ {0.0, 0.2, 0.4, 0.5, 0.6}
  w_conv (Tier 2 ConvLSTM 부분) / w_fno (Tier 2 FNO 부분):
    각 w_t1 마다 w_t2 = 1 - w_t1 을 conv:fno 비율 5종 (0/0, 0.25/0.75, 0.5/0.5, 0.75/0.25, 1/0) 으로 분할

산출물: figures/current/09_ensemble_3way/grid_search.png + best 조합
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency
from evaluate_t_locations import load_mask, load_model
from evaluate_sparse_fno import load_sparse_fno
from train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator
from evaluate_ensemble import (
    precompute_node_to_cell_weights, gnn_node_pred_to_cell_danger,
    tier1_forward, tier2_forward, iou_rmse_fnr,
)
from src.shared.constants import DT_SLCF


def eval_scenario_3way(
    scen_dir, gnn_model, conv_model, fno_model,
    adj, mask, sparse_ind, knn_idx, knn_w,
    t_start, t0_seconds, weight_combos,
    seq_dir, device,
) -> List[Dict]:
    name = scen_dir.name
    # Each tier forward once
    t1_node = tier1_forward(name, seq_dir, gnn_model, adj, t_start, device)
    t1_cell = gnn_node_pred_to_cell_danger(t1_node, knn_idx, knn_w)
    t2_conv, truth = tier2_forward(scen_dir, conv_model, mask, sparse_ind,
                                    t0_seconds, device, arch="conv_lstm")
    t2_fno, _ = tier2_forward(scen_dir, fno_model, mask, sparse_ind,
                               t0_seconds, device, arch="fno")
    iou_t1 = iou_rmse_fnr(t1_cell[5:6], truth[5:6], mask)["iou"]
    iou_conv = iou_rmse_fnr(t2_conv[5:6], truth[5:6], mask)["iou"]
    iou_fno = iou_rmse_fnr(t2_fno[5:6], truth[5:6], mask)["iou"]
    print(f"  [{name}] T1={iou_t1:.3f}  Conv={iou_conv:.3f}  FNO={iou_fno:.3f}")
    results = []
    for w_t1, w_conv, w_fno in weight_combos:
        ens = w_t1 * t1_cell + w_conv * t2_conv + w_fno * t2_fno
        m6 = iou_rmse_fnr(ens[5:6], truth[5:6], mask)
        results.append({
            "name": name, "w_t1": w_t1, "w_conv": w_conv, "w_fno": w_fno,
            "iou_step6": m6["iou"], "fnr_step6": m6["fnr"],
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn-ckpt", type=Path,
                        default=Path("checkpoints/tier1_gnn_v3/best.pt"))
    parser.add_argument("--conv-ckpt", type=Path,
                        default=Path("checkpoints/conv_lstm_sparse_v3/best.pt"))
    parser.add_argument("--fno-ckpt", type=Path,
                        default=Path("checkpoints/fno_sparse_v3/best.pt"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--seq-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path, default=Path("configs/building.yaml"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/current/10_ensemble_3way"))
    parser.add_argument("--out-csv", type=Path,
                        default=Path("results/exp_ensemble_3way/comparison.csv"))
    parser.add_argument("--t0", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--geodesic-projection", action="store_true")
    args = parser.parse_args()

    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[setup] loading 3 models...")

    # Tier 1 GNN
    t1_ckpt = torch.load(args.gnn_ckpt, weights_only=False, map_location=device)
    cfg = t1_ckpt.get("config", {})
    gnn_model = SimpleFireGNN(
        in_feat=5, hidden=cfg.get("hidden", 32),
        n_graph_layers=cfg.get("n_graph_layers", 2), T_out=cfg.get("T_out", 6),
    )
    gnn_model.load_state_dict(t1_ckpt["model"])
    gnn_model.to(device).eval()
    adj = build_knn_adjacency(k=cfg.get("knn_k", 4))

    # Tier 2 Sparse ConvLSTM (5-ch)
    conv_model = load_model(args.conv_ckpt, device, "conv_lstm")
    # Tier 2 Sparse FNO (6-ch)
    fno_model = load_sparse_fno(args.fno_ckpt, device)

    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)

    proj_name = "geodesic" if args.geodesic_projection else "Euclidean"
    print(f"[setup] precomputing {proj_name} cell-node IDW weights...")
    node_positions = [d.position for d in ALL_DETECTORS]
    knn_idx, knn_w = precompute_node_to_cell_weights(
        node_positions, k=3, sigma=5.0,
        mask=mask, use_geodesic=args.geodesic_projection,
    )

    # Weight grid: w_t1 + w_conv + w_fno = 1
    weight_combos = []
    for w_t1 in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]:
        w_rem = 1.0 - w_t1
        for r in [0.0, 0.25, 0.5, 0.75, 1.0]:  # conv:fno 비율
            w_conv = w_rem * r
            w_fno = w_rem * (1 - r)
            weight_combos.append((round(w_t1, 2), round(w_conv, 3), round(w_fno, 3)))
    print(f"[setup] {len(weight_combos)} weight combos")

    t_start = int(args.t0 // DT_SLCF) - 6
    scens = sorted(d for d in args.raw_root.glob("sim_*_T*") if d.is_dir())
    all_results = []
    for scen in scens:
        try:
            rs = eval_scenario_3way(
                scen, gnn_model, conv_model, fno_model,
                adj, mask, sparse_ind, knn_idx, knn_w,
                t_start, args.t0, weight_combos, args.seq_dir, device,
            )
            all_results.extend(rs)
        except Exception as e:
            print(f"[skip] {scen.name}: {e}")

    # Aggregate
    print(f"\n[agg] {len(weight_combos)} combos × 13 scenarios:")
    print(f"  {'w_t1':>6} {'w_conv':>7} {'w_fno':>7} {'mIoU':>7} {'mFNR':>7} {'H5':>4}")
    agg = []
    for combo in weight_combos:
        subset = [r for r in all_results
                   if (r["w_t1"], r["w_conv"], r["w_fno"]) == combo]
        ious = [r["iou_step6"] for r in subset]
        fnrs = [r["fnr_step6"] for r in subset]
        n_pass = sum(1 for i in ious if i >= 0.7)
        agg.append({"w_t1": combo[0], "w_conv": combo[1], "w_fno": combo[2],
                    "mean_iou": float(np.mean(ious)),
                    "mean_fnr": float(np.mean(fnrs)),
                    "h5_pass": n_pass})

    # Print sorted by IoU
    print("\n[best 10 by IoU]")
    print(f"  {'w_t1':>6} {'w_conv':>7} {'w_fno':>7} {'mIoU':>7} {'mFNR':>7} {'H5':>4}")
    for a in sorted(agg, key=lambda x: x["mean_iou"], reverse=True)[:10]:
        print(f"  {a['w_t1']:>6.2f} {a['w_conv']:>7.3f} {a['w_fno']:>7.3f} "
              f"{a['mean_iou']:>7.3f} {a['mean_fnr']*100:>6.1f}% {a['h5_pass']:>3d}")

    best_iou = max(agg, key=lambda x: x["mean_iou"])
    best_fnr = min(agg, key=lambda x: x["mean_fnr"])
    print(f"\n[best by IoU]  w=({best_iou['w_t1']}, {best_iou['w_conv']:.2f}, "
          f"{best_iou['w_fno']:.2f}) -> IoU={best_iou['mean_iou']:.3f}, "
          f"FNR={best_iou['mean_fnr']*100:.1f}%, H5={best_iou['h5_pass']}/13")
    print(f"[best by FNR]  w=({best_fnr['w_t1']}, {best_fnr['w_conv']:.2f}, "
          f"{best_fnr['w_fno']:.2f}) -> IoU={best_fnr['mean_iou']:.3f}, "
          f"FNR={best_fnr['mean_fnr']*100:.1f}%, H5={best_fnr['h5_pass']}/13")

    # Save
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["w_t1", "w_conv", "w_fno", "mean_iou", "mean_fnr", "h5_pass"])
        for a in agg:
            w.writerow([a["w_t1"], a["w_conv"], a["w_fno"],
                        a["mean_iou"], a["mean_fnr"], a["h5_pass"]])

    # Heatmap: w_t1 (x) vs conv:fno ratio (y) per metric
    w_t1_vals = sorted(set(a["w_t1"] for a in agg))
    ratios = sorted(set(round(a["w_conv"] / (a["w_conv"] + a["w_fno"]), 2)
                        if (a["w_conv"] + a["w_fno"]) > 1e-9 else 0.5
                        for a in agg))
    iou_grid = np.zeros((len(ratios), len(w_t1_vals)))
    fnr_grid = np.zeros((len(ratios), len(w_t1_vals)))
    for a in agg:
        denom = a["w_conv"] + a["w_fno"]
        ratio = round(a["w_conv"] / denom, 2) if denom > 1e-9 else 0.5
        i = ratios.index(ratio)
        j = w_t1_vals.index(a["w_t1"])
        iou_grid[i, j] = a["mean_iou"]
        fnr_grid[i, j] = a["mean_fnr"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axes[0].imshow(iou_grid, cmap="RdYlGn", vmin=0.4, vmax=0.65,
                          aspect="auto", origin="lower")
    axes[0].set_xticks(range(len(w_t1_vals)))
    axes[0].set_xticklabels([f"{w:.1f}" for w in w_t1_vals])
    axes[0].set_yticks(range(len(ratios)))
    axes[0].set_yticklabels([f"{r:.2f}" for r in ratios])
    axes[0].set_xlabel("w_tier1 (GNN)")
    axes[0].set_ylabel("ConvLSTM : (ConvLSTM+FNO) ratio")
    axes[0].set_title("Mean IoU (13 OOD)")
    plt.colorbar(im0, ax=axes[0])
    for i in range(len(ratios)):
        for j in range(len(w_t1_vals)):
            axes[0].text(j, i, f"{iou_grid[i, j]:.2f}", ha="center", va="center",
                          fontsize=8, color="black")

    im1 = axes[1].imshow(fnr_grid * 100, cmap="RdYlGn_r", vmin=2, vmax=20,
                          aspect="auto", origin="lower")
    axes[1].set_xticks(range(len(w_t1_vals)))
    axes[1].set_xticklabels([f"{w:.1f}" for w in w_t1_vals])
    axes[1].set_yticks(range(len(ratios)))
    axes[1].set_yticklabels([f"{r:.2f}" for r in ratios])
    axes[1].set_xlabel("w_tier1 (GNN)")
    axes[1].set_ylabel("ConvLSTM : (ConvLSTM+FNO) ratio")
    axes[1].set_title("Mean FNR (%)")
    plt.colorbar(im1, ax=axes[1])
    for i in range(len(ratios)):
        for j in range(len(w_t1_vals)):
            axes[1].text(j, i, f"{fnr_grid[i, j]*100:.0f}", ha="center", va="center",
                          fontsize=8, color="black")
    fig.suptitle("3-way Ensemble grid search (Tier 1 GNN + Sparse ConvLSTM + Sparse FNO)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out_figures / "grid_search.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {args.out_figures / 'grid_search.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
