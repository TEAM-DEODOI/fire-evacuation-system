"""L1 → L4h evaluation layer summary — single-figure paper headline.

Visualizes the IoU @ +60s on the 13 OOD scenarios across all evaluation
layers, from the L2 oracle upper bound (full SLCF + ConvLSTM, IoU 0.92)
down to the L4 sparse-interpolation valley (~0.20) and back up through
L4f/L4g/L4h to the deployable plateau.

Highlights:
* Upper-bound L2 (ConvLSTM full SLCF)        — what is theoretically achievable
* Lower-end L4 (sparse interpolation)         — what naive sparse inputs give
* L4f Tier 1 GNN (per-node, binary)           — Paper Figure 1 contribution
* L4h Learned Decoder (cell-level, β fn=2.5)  — Paper Figure 2 contribution
* L4g 3-way hand-crafted ensemble (D-026)     — baseline against L4h
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/current/02_l1_l4_layers"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Layer rows. Format: (label, iou, group, highlight)
    # group ∈ {oracle, baseline_full, l3, l4_naive, l4_retrain, l4_gnn, l4_ensemble}
    rows = [
        ("L2  ConvLSTM full SLCF (oracle, mid-fire)",     0.920, "oracle",     "upper"),
        ("L2  FNO PI full SLCF",                           0.890, "oracle",     ""),
        ("L1  ConvLSTM teacher-forced single-step",        0.890, "oracle",     ""),
        ("L1  FNO PI teacher-forced",                      0.840, "oracle",     ""),
        ("L2  FNO no-PI full SLCF",                        0.820, "oracle",     ""),
        # ─── Paper main contributions ───
        ("L4f  Tier 1 GNN per-node binary  ★",             0.904, "headline",   "main"),
        ("L4h  Learned Decoder (β, fn=2.5)  ★",            0.733, "headline",   "main"),
        ("L4h  Learned Decoder (γ, fn=4.0, safety)",       0.718, "headline",   ""),
        # ─── Hand-crafted ensembles ───
        ("L4g  3-way Balanced (geodesic, D-026 baseline)", 0.618, "ensemble",   "baseline"),
        ("L4g  2-way GNN+FNO (geodesic best)",             0.569, "ensemble",   ""),
        # ─── Detector trigger / sparse models alone ───
        ("L3   Detector-triggered ConvLSTM",               0.530, "l3",         ""),
        ("L4e  Sparse-ConvLSTM v3 + re-sparsify",          0.581, "l4_retrain", ""),
        ("L4e' Sparse-FNO v3 + sensor indicator",          0.525, "l4_retrain", ""),
        # ─── Naive sparse interpolation valley ───
        ("L4d  Sparse + FNO no-PI + geodesic IDW (39)",    0.430, "l4_naive",   ""),
        ("L4a  Sparse + nearest interp (16 sensors)",      0.280, "l4_naive",   ""),
        ("L4d  Sparse + geodesic IDW + ConvLSTM (39)",     0.210, "l4_naive",   "valley"),
        ("L4b  Sparse + linear interp",                    0.190, "l4_naive",   ""),
        ("L4c  Sparse + cubic interp",                     0.190, "l4_naive",   ""),
        ("L4e  Sparse-ConvLSTM naive chaining (no fix)",   0.180, "l4_naive",   ""),
    ]

    # Sort by IoU descending
    rows = sorted(rows, key=lambda r: -r[1])

    labels = [r[0] for r in rows]
    ious   = [r[1] for r in rows]
    groups = [r[2] for r in rows]
    highlights = [r[3] for r in rows]

    color_by_group = {
        "oracle":     "tab:blue",
        "headline":   "darkred",
        "ensemble":   "tab:gray",
        "l3":         "tab:orange",
        "l4_retrain": "tab:purple",
        "l4_naive":   "tab:brown",
    }
    colors = [color_by_group[g] for g in groups]
    edgewidths = [2.0 if h == "main" else (1.4 if h in ("upper", "baseline", "valley") else 0.6)
                    for h in highlights]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, ious, color=colors, edgecolor="black",
                    linewidth=edgewidths, alpha=0.92)

    # Reference lines
    ax.axvline(0.92, color="tab:blue", ls=":", lw=0.8, alpha=0.7,
                label="L2 oracle upper bound (0.92)")
    ax.axvline(0.70, color="green", ls="--", lw=0.8, label="H5 ≥ 0.70")

    # Y-axis: layer labels
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean IoU @ t₀ + 60 s (13 OOD scenarios)", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.3, axis="x")

    # Value annotations
    for i, v in enumerate(ious):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    # Star markers next to paper main rows
    for i, h in enumerate(highlights):
        if h == "main":
            ax.text(-0.015, i, "★", va="center", ha="right",
                     color="darkred", fontsize=14)

    # Legend by group
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="tab:blue",   edgecolor="black", label="Oracle (L1/L2 full SLCF)"),
        Patch(facecolor="darkred",    edgecolor="black",
                label="L4f/L4h paper main contributions ★"),
        Patch(facecolor="tab:gray",   edgecolor="black",
                label="L4g hand-crafted ensemble (D-026)"),
        Patch(facecolor="tab:orange", edgecolor="black", label="L3 detector-triggered"),
        Patch(facecolor="tab:purple", edgecolor="black",
                label="L4e/L4e' sparse retrain (D-025 re-sparsify)"),
        Patch(facecolor="tab:brown",  edgecolor="black", label="L4a-d sparse interpolation"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    ax.set_title(
        "Evaluation-layer summary — IoU @ +60 s across L1–L4h\n"
        "★ rows = paper main contributions (L4f per-node 0.904, L4h cell-level 0.733)",
        fontsize=13,
    )

    fig.tight_layout()
    out = args.out_dir / "model_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")

    # ── Second figure: contribution staircase ──────────────────────────
    fig2, ax2 = plt.subplots(1, 1, figsize=(11, 5.5))
    # Staircase rows: L1 oracle, L4 valley, L4f, L4g, L4h
    stairs = [
        ("L2 oracle\n(ConvLSTM full SLCF)",        0.920, "tab:blue"),
        ("L4d sparse IDW\n+ ConvLSTM",              0.210, "tab:brown"),
        ("L4e' Sparse FNO\n(+ re-sparsify, 6-ch)",  0.525, "tab:purple"),
        ("L4g 3-way Balanced\n(geodesic D-026)",   0.618, "tab:gray"),
        ("L4h Learned\nDecoder fn=2.5  ★",         0.733, "darkred"),
        ("L4f Tier 1 GNN\nper-node binary  ★",     0.904, "darkred"),
    ]
    x = np.arange(len(stairs))
    iou_vals = [s[1] for s in stairs]
    cols     = [s[2] for s in stairs]
    ax2.bar(x, iou_vals, color=cols, edgecolor="black", linewidth=1.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[0] for s in stairs], fontsize=10)
    ax2.set_ylabel("Mean IoU @ +60 s")
    ax2.axhline(0.92, color="tab:blue", ls=":", lw=0.8, alpha=0.7,
                  label="Oracle 0.92")
    ax2.axhline(0.70, color="green", ls="--", lw=0.8, label="H5 ≥ 0.70")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, 1.0)
    ax2.set_title(
        "Closing the deployment gap — how each layer recovers the L2 → L4 valley\n"
        "Oracle (left) → naive sparse (-0.71) → retrain (+0.32) → hand ensemble (+0.09)\n"
        "→ learned decoder (+0.12) → per-node GNN (+0.17)",
        fontsize=12,
    )
    for xi, v in zip(x, iou_vals):
        ax2.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    fig2.tight_layout()
    out2 = args.out_dir / "staircase.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plot] {out2}")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
