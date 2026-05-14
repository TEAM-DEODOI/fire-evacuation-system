"""Tier 1 GNN alone — per-node 60s prediction (baseline visualization).

Visualizes the "Tier 1 GNN alone | IoU 0.904 (per-node) | FNR 4.6% | 13/13 H5"
row of the hypothesis table.

Format matches `visualize_60s_*model.py` (FDS truth row on top, model row
below, 6 columns for t₀+10s..t₀+60s), but at PER-NODE resolution (39 nodes
colored by danger over the floor-plan wall outline).

Output: figures/current/04_tier1_gnn/<scenario>_alone_t<NN>.png
        figures/current/04_tier1_gnn/alone_headline.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_dataset import Tier1FireDataset, default_splits
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency
from visualize_tier1_predictions import (
    iou_fnr, load_mask_wall_segments, load_model, plot_one_panel,
)

DT_SLCF = 10.0
N_NODES = len(ALL_DETECTORS)
LOOKAHEAD_STEPS = 6


def plot_alone_2x6(
    scen_name: str,
    truth_seq: np.ndarray,    # (T_out, N)
    pred_seq: np.ndarray,     # (T_out, N)
    wall_segments,
    t0_seconds: float,
    iou: float,
    fnr: float,
    out_path: Path,
) -> None:
    """2 rows (FDS truth / GNN prediction) × 6 cols (t₀+10s .. t₀+60s).

    Matches the standard 60s-comparison grid format but at per-node level.
    """
    n_cols = truth_seq.shape[0]
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 5.5))

    for col in range(n_cols):
        t_label = f"t = {t0_seconds + (col + 1) * DT_SLCF:.0f} s"
        plot_one_panel(axes[0, col], wall_segments, truth_seq[col],
                       title=t_label)
        plot_one_panel(axes[1, col], wall_segments, pred_seq[col],
                       title="")

    axes[0, 0].set_ylabel("FDS truth",       fontsize=13,
                          rotation=90, labelpad=12, fontweight="bold")
    axes[1, 0].set_ylabel("Tier 1 GNN",      fontsize=13,
                          rotation=90, labelpad=12, fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                                norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, cax=cbar_ax, label="danger ∈ [0, 1]")

    fig.suptitle(
        f"Tier 1 GNN alone — per-node 60 s risk forecast  |  {scen_name}\n"
        f"t₀ = {t0_seconds:.0f} s, 39 sensors (D-024 v3.3), 12 K params  "
        f"|  IoU = {iou:.3f}  |  FNR = {fnr*100:.1f}%  "
        f"|  baseline of L4f row",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 0.92])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=Path("checkpoints/tier1_gnn_v3/best.pt"))
    parser.add_argument("--sequence-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/processed/dataset.h5"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/current/04_tier1_gnn"))
    parser.add_argument("--scenarios", type=str, nargs="+",
                        default=["sim_1500kw_2m2_T05",
                                 "sim_500kw_1m2_T01",
                                 "sim_1000kw_1m2_T03"])
    parser.add_argument("--headline-scenario", type=str,
                        default="sim_1500kw_2m2_T05")
    parser.add_argument("--t0", type=float, default=120.0,
                        help="t₀ in seconds (input ends at t₀, forecast t₀+10..t₀+60)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading model {args.ckpt}")
    model, ckpt_meta = load_model(args.ckpt)
    print(f"        val_iou={ckpt_meta.get('val_iou')}  epoch={ckpt_meta.get('epoch')}")
    adj = build_knn_adjacency(k=ckpt_meta.get("config", {}).get("knn_k", 4))

    print(f"[setup] loading mask wall segments")
    wall_segments = load_mask_wall_segments(args.dataset)

    # t_start: sliding window start (input frames [t_start .. t_start+5])
    # so the output window starts at frame (t_start+6) = t₀/10.
    t_start = int(args.t0 // DT_SLCF) - 6

    # Cache for headline
    headline_payload = None

    for scen_name in args.scenarios:
        ds = Tier1FireDataset(args.sequence_dir, [scen_name],
                               T_in=6, T_out=LOOKAHEAD_STEPS)
        idx = None
        for i, (s_idx, t) in enumerate(ds.pairs):
            if t == t_start:
                idx = i; break
        if idx is None:
            print(f"[skip] {scen_name}: no pair at t_start={t_start}")
            continue

        x, y_truth = ds[idx]
        with torch.no_grad():
            y_pred = model(x.unsqueeze(0), adj).squeeze(0).numpy()
        y_truth = y_truth.numpy()
        truth_seq = y_truth.T   # (T_out, N)
        pred_seq = y_pred.T

        # Metric at +60s (step 6 = index 5)
        m6 = iou_fnr(pred_seq[5], truth_seq[5])
        iou6 = m6["iou"]; fnr6 = m6["fnr"]
        print(f"  {scen_name:30s}  IoU={iou6:.3f}  FNR={fnr6*100:.1f}%")

        out_path = args.out_dir / f"{scen_name}_alone_t{t_start:02d}.png"
        plot_alone_2x6(scen_name, truth_seq, pred_seq, wall_segments,
                        args.t0, iou6, fnr6, out_path)
        print(f"  -> {out_path}")

        if scen_name == args.headline_scenario:
            headline_payload = (scen_name, truth_seq, pred_seq, iou6, fnr6)

    # Headline (alone version)
    if headline_payload is not None:
        scen_name, truth_seq, pred_seq, iou6, fnr6 = headline_payload
        head_path = args.out_dir / "alone_headline.png"
        plot_alone_2x6(scen_name, truth_seq, pred_seq, wall_segments,
                        args.t0, iou6, fnr6, head_path)
        print(f"[headline] -> {head_path}")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
