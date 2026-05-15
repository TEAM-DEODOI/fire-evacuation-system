"""Visualize v3 vs v4 Tier 1 GNN comparison — per-scenario IoU bar chart
+ s_029 per-node rollout side-by-side.

Outputs:
    figures/current/13_gnn_v4/per_scenario_compare.png  — 33 train + 13 OOD bars
    figures/current/13_gnn_v4/s029_rollout.png          — s_029 6-step truth/v3/v4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_dataset import Tier1FireDataset
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency
from visualize_tier1_predictions import (
    load_mask_wall_segments, plot_one_panel, iou_fnr,
)


def load_gnn(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    m = SimpleFireGNN(in_feat=5, hidden=cfg.get("hidden", 32),
                      n_graph_layers=cfg.get("n_graph_layers", 2),
                      T_out=cfg.get("T_out", 6))
    m.load_state_dict(ckpt["model"]); m.eval()
    return m, build_knn_adjacency(k=cfg.get("knn_k", 4))


def iou_at_t6(model, adj, names: List[str], seq_dir: Path) -> List[Tuple[str, float, float]]:
    rows = []
    for n in names:
        ds = Tier1FireDataset(seq_dir, [n], T_in=6, T_out=6)
        for i, (_, t) in enumerate(ds.pairs):
            if t == 6:
                x, y = ds[i]
                with torch.no_grad():
                    yp = model(x.unsqueeze(0), adj).squeeze(0).numpy()
                p, t_arr = yp[:, 5] >= 0.5, y.numpy()[:, 5] >= 0.5
                tp = (p & t_arr).sum()
                fp = (p & ~t_arr).sum()
                fn = (~p & t_arr).sum()
                rows.append((n, float(tp / (tp + fp + fn + 1e-9)),
                              float(fn / (fn + tp + 1e-9))))
                break
    return rows


def plot_per_scenario(v3_rows, v4_rows, train_names, out_path):
    v3d = {r[0]: r for r in v3_rows}
    v4d = {r[0]: r for r in v4_rows}
    # Two-panel layout: train (left) + OOD (right), within each sorted by v3 IoU
    train_ids = sorted(train_names, key=lambda n: v3d[n][1])
    ood_ids   = sorted([r[0] for r in v3_rows if r[0].startswith("sim_")],
                         key=lambda n: v3d[n][1])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    for ax, ids, title in [(axes[0], train_ids, "33 Train scenarios (sorted by v3 IoU)"),
                            (axes[1], ood_ids,  "13 OOD scenarios (sorted by v3 IoU)")]:
        n = len(ids)
        x = np.arange(n)
        v3_vals = [v3d[i][1] for i in ids]
        v4_vals = [v4d[i][1] for i in ids]
        w = 0.4
        ax.bar(x - w/2, v3_vals, w, label="v3 (MSE, paper-main)",
                color="tab:gray", edgecolor="black", linewidth=0.4)
        ax.bar(x + w/2, v4_vals, w, label="v4 (focal+asym, small-fire boost)",
                color="tab:blue", edgecolor="black", linewidth=0.4)
        ax.axhline(0.70, color="red", lw=0.7, ls="--", label="H5 >= 0.70")
        # Annotate large gains
        for i, (v3v, v4v) in enumerate(zip(v3_vals, v4_vals)):
            d = v4v - v3v
            if abs(d) >= 0.05:
                ax.annotate(f"{d:+.2f}",
                             xy=(i, max(v3v, v4v) + 0.02),
                             ha="center", fontsize=7,
                             color="tab:green" if d > 0 else "tab:red")
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("IoU @ +60s")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Tier 1 GNN — v3 (MSE) vs v4 (focal+asymmetric BCE + small-fire boost)\n"
        f"Train mean: 0.820 -> 0.876 (+0.056)   |   "
        f"OOD mean: 0.889 -> 0.901 (+0.012, H5 pass 12->13/13)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_s029_rollout(m3, adj3, m4, adj4, seq_dir, dataset_h5, out_path):
    """s_029 per-node prediction: truth / v3 / v4 across 6 lookahead steps."""
    wall = load_mask_wall_segments(Path(dataset_h5))
    ds = Tier1FireDataset(Path(seq_dir), ["s_029"], T_in=6, T_out=6)
    target = None
    for i, (_, t) in enumerate(ds.pairs):
        if t == 6:
            target = i; break
    x, y = ds[target]
    truth = y.numpy().T  # (T_out=6, N=39)
    with torch.no_grad():
        v3_pred = m3(x.unsqueeze(0), adj3).squeeze(0).numpy().T   # (6, N)
        v4_pred = m4(x.unsqueeze(0), adj4).squeeze(0).numpy().T   # (6, N)

    iou3 = iou_fnr(v3_pred[5], truth[5])["iou"]
    iou4 = iou_fnr(v4_pred[5], truth[5])["iou"]

    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    for col in range(6):
        t_label = f"t = {120 + (col + 1) * 10:.0f}s"
        plot_one_panel(axes[0, col], wall, truth[col],   title=t_label)
        plot_one_panel(axes[1, col], wall, v3_pred[col], title="")
        plot_one_panel(axes[2, col], wall, v4_pred[col], title="")

    axes[0, 0].set_ylabel("FDS truth", fontsize=13, rotation=90, labelpad=12,
                          fontweight="bold")
    axes[1, 0].set_ylabel(f"v3 (MSE)\nIoU {iou3:.3f}", fontsize=12,
                          rotation=90, labelpad=12, fontweight="bold",
                          color="tab:gray")
    axes[2, 0].set_ylabel(f"v4 (focal+asym)\nIoU {iou4:.3f}", fontsize=12,
                          rotation=90, labelpad=12, fontweight="bold",
                          color="darkred")

    cbar_ax = fig.add_axes([0.92, 0.20, 0.013, 0.6])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, cax=cbar_ax, label="danger in [0, 1]")

    fig.suptitle(
        f"s_029 (sim_500kw_1m2_H03)  --  worst-case small fire  --  "
        f"v3 IoU {iou3:.3f}  ->  v4 IoU {iou4:.3f}  (dIoU {iou4-iou3:+.3f})\n"
        f"v4 = focal (gamma=2.0) + asymmetric BCE (fn_weight=2.5) + small-fire boost x2.0",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 0.94])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-ckpt", type=Path,
                        default=Path("checkpoints/tier1_gnn_v3/best.pt"))
    parser.add_argument("--v4-ckpt", type=Path,
                        default=Path("checkpoints/tier1_gnn_v4/best.pt"))
    parser.add_argument("--seq-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--dataset-h5", type=Path,
                        default=Path("data/processed/dataset.h5"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/current/13_gnn_v4"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    m3, adj3 = load_gnn(args.v3_ckpt)
    m4, adj4 = load_gnn(args.v4_ckpt)

    with open("data/raw/scenario_config.json", encoding="utf-8") as f:
        scens = json.load(f)["scenarios"]
    train_names = [s["id"] for s in scens if s["split"] == "train"]
    ood_names = sorted([p.stem for p in args.seq_dir.glob("sim_*.npz")])
    all_names = train_names + ood_names

    v3_rows = iou_at_t6(m3, adj3, all_names, args.seq_dir)
    v4_rows = iou_at_t6(m4, adj4, all_names, args.seq_dir)

    plot_per_scenario(v3_rows, v4_rows, train_names,
                       args.out_dir / "per_scenario_compare.png")
    print(f"[plot] {args.out_dir / 'per_scenario_compare.png'}")

    plot_s029_rollout(m3, adj3, m4, adj4,
                       args.seq_dir, args.dataset_h5,
                       args.out_dir / "s029_rollout.png")
    print(f"[plot] {args.out_dir / 's029_rollout.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
