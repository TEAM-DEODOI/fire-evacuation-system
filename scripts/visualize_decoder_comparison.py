"""Visualize the learned decoder vs hand-crafted ensemble (Option 2 results).

Three figures:
1. frontier.png            — IoU vs FNR Pareto frontier across fn_weight + baseline
2. per_scenario.png        — 13 OOD bar chart (hand-crafted vs fn=2.5 vs fn=4.0)
3. training_overlay.png    — training curves overlay (4 fn variants)

Inputs:
- results/exp_decoder_ensemble_fn{10,25,40}/per_scenario.csv  (sweep variants)
- results/exp_decoder_ensemble/per_scenario.csv               (= fn=2.5 default)
- results/exp_ensemble_3way_geodesic/comparison.csv           (hand-crafted baseline)

Output: figures/current/11_decoder_ensemble/
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─── Load helpers ─────────────────────────────────────────────────────────
def load_per_scenario(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "scenario": r["scenario"],
                "iou": float(r["iou_step6"]),
                "fnr": float(r["fnr_step6"]),
                "rmse": float(r["rmse_step6"]),
            })
    return rows


def load_baseline_balanced(grid_csv: Path) -> Dict:
    """Pick the balanced (0.5, 0.25, 0.25) row from the 3-way grid search."""
    with grid_csv.open() as f:
        for r in csv.DictReader(f):
            if (abs(float(r["w_t1"]) - 0.5) < 1e-6
                and abs(float(r["w_conv"]) - 0.25) < 1e-6
                and abs(float(r["w_fno"]) - 0.25) < 1e-6):
                return {
                    "iou": float(r["mean_iou"]),
                    "fnr": float(r["mean_fnr"]),
                    "h5": int(r["h5_pass"]),
                }
    raise ValueError("Balanced (0.5, 0.25, 0.25) row not found")


def summarize(rows: List[Dict]) -> Dict:
    ious = [r["iou"] for r in rows]
    fnrs = [r["fnr"] for r in rows]
    return {
        "mean_iou": float(np.mean(ious)),
        "mean_fnr": float(np.mean(fnrs)),
        "n_h5": sum(1 for v in ious if v >= 0.7),
        "n_h4": sum(1 for v in fnrs if v < 0.10),
        "ious": ious, "fnrs": fnrs,
        "names": [r["scenario"] for r in rows],
    }


# ─── Plot 1: IoU vs FNR frontier ──────────────────────────────────────────
def plot_frontier(
    decoder_summaries: Dict[str, Dict],     # {fn_weight_label: summary}
    baseline: Dict,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5))

    # Baseline (hand-crafted 3-way Balanced)
    ax.scatter(baseline["fnr"] * 100, baseline["iou"], s=240, marker="X",
                color="tab:gray", edgecolors="black", linewidths=1.4, zorder=4,
                label=f"Hand-crafted 3-way Balanced (D-026)\n  IoU={baseline['iou']:.3f}, "
                       f"FNR={baseline['fnr']*100:.1f}%, H5={baseline['h5']}/13")

    # Decoder variants
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(decoder_summaries)))
    fn_order = sorted(decoder_summaries.keys(), key=lambda k: float(k.split("=")[1]))
    for color, label in zip(cmap, fn_order):
        s = decoder_summaries[label]
        marker = "*" if "2.5" in label else "o"
        size = 480 if "2.5" in label else 200
        ax.scatter(s["mean_fnr"] * 100, s["mean_iou"], s=size, marker=marker,
                    color=color, edgecolors="black", linewidths=1.2, zorder=5,
                    label=f"Learned Decoder {label}\n  IoU={s['mean_iou']:.3f}, "
                           f"FNR={s['mean_fnr']*100:.1f}%, H5={s['n_h5']}/13, H4={s['n_h4']}/13")

    # Reference lines
    ax.axhline(0.70, color="green", lw=0.8, ls=":", alpha=0.7, label="H5 ≥ 0.70")
    ax.axvline(10.0, color="red", lw=0.8, ls=":", alpha=0.7, label="H4 < 10%")

    # Pareto: connect dots in fn_weight order
    xs = [decoder_summaries[k]["mean_fnr"] * 100 for k in fn_order]
    ys = [decoder_summaries[k]["mean_iou"] for k in fn_order]
    ax.plot(xs, ys, "k--", lw=0.8, alpha=0.45, zorder=2)

    ax.set_xlabel("Mean FNR % @ +60s (lower = safer)", fontsize=12)
    ax.set_ylabel("Mean IoU @ +60s (higher = more accurate)", fontsize=12)
    ax.set_title(
        "L4h Learned Decoder vs L4g hand-crafted ensemble\n"
        "Pareto frontier across asymmetric-BCE fn_weight",
        fontsize=13,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_xlim(left=0)
    ax.set_ylim(0.55, 0.78)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Plot 2: per-scenario bar chart ───────────────────────────────────────
def plot_per_scenario(
    baseline_rows: List[Dict],
    decoder_rows_fn25: List[Dict],
    decoder_rows_fn40: List[Dict],
    out_path: Path,
) -> None:
    """Side-by-side IoU + FNR bars for 13 OOD."""
    names = [r["scenario"] for r in baseline_rows]
    # short labels
    labels = [n.replace("sim_", "").replace("kw", "k") for n in names]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    width = 0.27
    x = np.arange(len(names))

    # IoU subplot
    iou_b  = [r["iou"] for r in baseline_rows]
    iou_25 = [r["iou"] for r in decoder_rows_fn25]
    iou_40 = [r["iou"] for r in decoder_rows_fn40]
    axes[0].bar(x - width, iou_b,  width, label="Hand-crafted Balanced (D-026)",
                 color="tab:gray", edgecolor="black", linewidth=0.5)
    axes[0].bar(x,         iou_25, width, label="Learned Decoder fn=2.5 ★",
                 color="tab:blue", edgecolor="black", linewidth=0.5)
    axes[0].bar(x + width, iou_40, width, label="Learned Decoder fn=4.0",
                 color="tab:cyan", edgecolor="black", linewidth=0.5)
    axes[0].axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    axes[0].set_ylabel("IoU @ +60s")
    axes[0].set_title("Per-scenario IoU — Hand-crafted vs Learned (13 OOD)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_ylim(0, 1)

    # FNR subplot
    fnr_b  = [r["fnr"] * 100 for r in baseline_rows]
    fnr_25 = [r["fnr"] * 100 for r in decoder_rows_fn25]
    fnr_40 = [r["fnr"] * 100 for r in decoder_rows_fn40]
    axes[1].bar(x - width, fnr_b,  width, label="Hand-crafted Balanced",
                 color="tab:gray", edgecolor="black", linewidth=0.5)
    axes[1].bar(x,         fnr_25, width, label="Learned Decoder fn=2.5 ★",
                 color="tab:orange", edgecolor="black", linewidth=0.5)
    axes[1].bar(x + width, fnr_40, width, label="Learned Decoder fn=4.0",
                 color="tab:red", edgecolor="black", linewidth=0.5)
    axes[1].axhline(10.0, color="red", lw=0.8, ls="--", label="H4 < 10%")
    axes[1].set_ylabel("FNR % @ +60s")
    axes[1].set_title("Per-scenario FNR — lower is safer")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Cell-level ensemble — per-scenario breakdown (13 OOD)",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Plot 3: Training curves overlay ──────────────────────────────────────
def plot_training_overlay(
    summaries_paths: Dict[str, Path],
    out_path: Path,
) -> None:
    """Re-train? No — we already have per-epoch logs in stdout. Instead, we
    use the per-scenario CSV (final eval only) to show the agg metric per
    variant. The training_curves.png files are already saved per variant;
    this is a one-glance comparison via final-eval bars.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    fn_labels = sorted(summaries_paths.keys(), key=lambda k: float(k.split("=")[1]))
    fn_vals = [float(k.split("=")[1]) for k in fn_labels]

    ious = []
    fnrs = []
    h5s = []
    h4s = []
    for k in fn_labels:
        rows = load_per_scenario(summaries_paths[k])
        s = summarize(rows)
        ious.append(s["mean_iou"])
        fnrs.append(s["mean_fnr"] * 100)
        h5s.append(s["n_h5"])
        h4s.append(s["n_h4"])

    axes[0].plot(fn_vals, ious, "o-", lw=2, color="tab:blue", markersize=10)
    axes[0].axhline(0.618, color="tab:gray", ls="--", lw=0.8, label="hand-crafted 0.618")
    axes[0].axhline(0.70, color="green", ls=":", lw=0.8, label="H5 ≥ 0.70")
    axes[0].set_xlabel("Asymmetric BCE fn_weight")
    axes[0].set_ylabel("Mean IoU @ +60s")
    axes[0].set_title("IoU vs fn_weight")
    axes[0].grid(alpha=0.3); axes[0].legend()
    for x, y in zip(fn_vals, ious):
        axes[0].text(x, y + 0.005, f"{y:.3f}", ha="center", fontsize=9)

    axes[1].plot(fn_vals, fnrs, "s-", lw=2, color="tab:red", markersize=10)
    axes[1].axhline(5.1, color="tab:gray", ls="--", lw=0.8, label="hand-crafted 5.1%")
    axes[1].axhline(10.0, color="red", ls=":", lw=0.8, label="H4 < 10%")
    axes[1].set_xlabel("Asymmetric BCE fn_weight")
    axes[1].set_ylabel("Mean FNR % @ +60s")
    axes[1].set_title("FNR vs fn_weight (lower=safer)")
    axes[1].grid(alpha=0.3); axes[1].legend()
    for x, y in zip(fn_vals, fnrs):
        axes[1].text(x, y + 0.3, f"{y:.1f}%", ha="center", fontsize=9)

    axes[2].plot(fn_vals, h5s, "^-", lw=2, color="tab:green",
                  markersize=10, label="H5 pass (IoU ≥ 0.70)")
    axes[2].plot(fn_vals, h4s, "v-", lw=2, color="tab:orange",
                  markersize=10, label="H4 pass (FNR < 10%)")
    axes[2].axhline(5, color="tab:gray", ls="--", lw=0.8, label="baseline H5 = 5")
    axes[2].set_xlabel("Asymmetric BCE fn_weight")
    axes[2].set_ylabel("# scenarios passing")
    axes[2].set_title("Hypothesis pass counts (max 13)")
    axes[2].grid(alpha=0.3); axes[2].legend()
    axes[2].set_ylim(0, 13.5)
    for x, y in zip(fn_vals, h5s):
        axes[2].text(x, y + 0.3, f"{y}", ha="center", fontsize=9)
    for x, y in zip(fn_vals, h4s):
        axes[2].text(x, y - 0.7, f"{y}", ha="center", fontsize=9, color="darkorange")

    fig.suptitle(
        "Learned Decoder — fn_weight sweep summary (13 OOD)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path,
                        default=Path("results/exp_ensemble_3way_geodesic/comparison.csv"),
                        help="Hand-crafted 3-way grid search CSV")
    parser.add_argument("--decoder-fn10", type=Path,
                        default=Path("results/exp_decoder_ensemble_fn10/per_scenario.csv"))
    parser.add_argument("--decoder-fn25", type=Path,
                        default=Path("results/exp_decoder_ensemble_fn25/per_scenario.csv"))
    parser.add_argument("--decoder-fn40", type=Path,
                        default=Path("results/exp_decoder_ensemble_fn40/per_scenario.csv"))
    parser.add_argument("--baseline-per-scenario-rows", type=Path,
                        default=Path("results/exp_ensemble_3way_geodesic/per_scenario.csv"),
                        help="Optional: per-scenario CSV for hand-crafted. "
                             "If missing, we recompute using the 3-way grid weights.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/current/11_decoder_ensemble"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: frontier ────────────────────────────────────────────────
    baseline = load_baseline_balanced(args.baseline_csv)
    decoder_summaries = {}
    for label, path in [("fn=1.0", args.decoder_fn10),
                         ("fn=2.5", args.decoder_fn25),
                         ("fn=4.0", args.decoder_fn40)]:
        if path.exists():
            decoder_summaries[label] = summarize(load_per_scenario(path))
    print(f"[frontier] hand-crafted baseline: IoU={baseline['iou']:.3f}, "
          f"FNR={baseline['fnr']*100:.1f}%, H5={baseline['h5']}/13")
    for label, s in decoder_summaries.items():
        print(f"           Decoder {label}:  IoU={s['mean_iou']:.3f}, "
              f"FNR={s['mean_fnr']*100:.1f}%, H5={s['n_h5']}/13, H4={s['n_h4']}/13")

    plot_frontier(decoder_summaries, baseline, args.out_dir / "frontier.png")
    print(f"[plot] {args.out_dir / 'frontier.png'}")

    # ── Plot 2: per-scenario ───────────────────────────────────────────
    # Need baseline per-scenario rows. If missing, re-run a small forward
    # is overkill — instead we can fall back to computing the balanced
    # average from the decoder data cache. For now require the file.
    if not args.baseline_per_scenario_rows.exists():
        print(f"[warn] baseline per-scenario CSV missing at "
              f"{args.baseline_per_scenario_rows}")
        print("       Skipping per_scenario.png — regenerate baseline with a "
              "per-scenario dump option, or use decoder_data cache.")
    else:
        baseline_rows = load_per_scenario(args.baseline_per_scenario_rows)
        decoder_rows_25 = load_per_scenario(args.decoder_fn25)
        decoder_rows_40 = load_per_scenario(args.decoder_fn40)
        # Align on scenario name
        bn = {r["scenario"]: r for r in baseline_rows}
        b_aligned = [bn[r["scenario"]] for r in decoder_rows_25
                     if r["scenario"] in bn]
        d25_aligned = [r for r in decoder_rows_25 if r["scenario"] in bn]
        d40_aligned = [r for r in decoder_rows_40
                       if r["scenario"] in bn]
        plot_per_scenario(b_aligned, d25_aligned, d40_aligned,
                           args.out_dir / "per_scenario.png")
        print(f"[plot] {args.out_dir / 'per_scenario.png'}")

    # ── Plot 3: fn_weight sweep summary ────────────────────────────────
    sweep_paths = {}
    for label, path in [("fn=1.0", args.decoder_fn10),
                         ("fn=2.5", args.decoder_fn25),
                         ("fn=4.0", args.decoder_fn40)]:
        if path.exists():
            sweep_paths[label] = path
    plot_training_overlay(sweep_paths, args.out_dir / "fn_sweep_summary.png")
    print(f"[plot] {args.out_dir / 'fn_sweep_summary.png'}")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
