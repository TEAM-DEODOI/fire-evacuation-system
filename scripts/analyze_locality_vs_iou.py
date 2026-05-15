"""Analyze model performance vs fire-footprint locality across 13 OOD scenarios.

Hypothesis (user-reported): cell-level surrogates predict poorly when fire
is highly localised. We quantify the relationship.

Locality metric: fire_footprint = fraction of fluid cells with truth danger
≥ 0.5 at t₀+60s (smaller = more localised).

Models compared:
    ① Sparse-ConvLSTM v3 단독         (no ensemble)
    ② Sparse-FNO v3 단독              (no ensemble)
    ③ Hand-crafted 3-way Balanced     (D-026 weighted avg)
    ④ Decoder fn=2.5 ★                (D-028 paper main)
    ⑤ Decoder fn=4.0                  (D-028 safety variant)
    α Tier 1 GNN (per-node)           (D-029 reference)
    GNN cell-projected (geodesic IDW) (single component of ③-⑤)

Outputs:
    figures/current/12_locality_analysis/scatter.png      — 7 models × scatter
    figures/current/12_locality_analysis/boxplots.png     — top-3 vs mid vs bottom-3 footprint
    figures/current/12_locality_analysis/per_scenario.png — bar chart sorted by footprint
    results/exp_locality_analysis/correlations.csv
    results/exp_locality_analysis/by_regime.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.tier1.tier1_dataset import Tier1FireDataset
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency


def iou_step(pred, truth, mask, thr=0.5):
    f = (mask > 0.5)
    fm = np.broadcast_to(f, pred.shape).reshape(-1)
    p = (pred >= thr).reshape(-1)[fm]
    t = (truth >= thr).reshape(-1)[fm]
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    return float(tp / (tp + fp + fn + 1e-9))


def fnr_step(pred, truth, mask, thr=0.5):
    f = (mask > 0.5)
    fm = np.broadcast_to(f, pred.shape).reshape(-1)
    p = (pred >= thr).reshape(-1)[fm]
    t = (truth >= thr).reshape(-1)[fm]
    fn, tp = (~p & t).sum(), (p & t).sum()
    return float(fn / (fn + tp + 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("results/decoder_data"))
    parser.add_argument("--seq-dir",  type=Path, default=Path("results/detector_sequences"))
    parser.add_argument("--gnn-ckpt", type=Path,
                        default=Path("checkpoints/tier1_gnn_v3/best.pt"))
    parser.add_argument("--decoder-fn25-csv", type=Path,
                        default=Path("results/exp_decoder_ensemble_fn25/per_scenario.csv"))
    parser.add_argument("--decoder-fn40-csv", type=Path,
                        default=Path("results/exp_decoder_ensemble_fn40/per_scenario.csv"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/current/12_locality_analysis"))
    parser.add_argument("--out-results", type=Path,
                        default=Path("results/exp_locality_analysis"))
    args = parser.parse_args()
    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_results.mkdir(parents=True, exist_ok=True)

    # ── 1. Compute footprint + cell-level IoUs (sparse_conv, sparse_fno, gnn_cell,
    #      hand-crafted) from cached decoder data ─────────────────────────
    names, fps, fps_mean = [], [], []
    iou_sc, iou_sf, iou_gc, iou_hand = [], [], [], []
    fnr_sc, fnr_sf, fnr_gc, fnr_hand = [], [], [], []
    for p in sorted(args.data_dir.glob("sim_*.npz")):
        d = np.load(p, allow_pickle=True)
        truth, mask = d["truth"], d["mask"]
        fluid = mask > 0.5
        fp6 = float((truth[5][fluid] >= 0.5).mean())
        fpm = float(np.mean([(truth[s][fluid] >= 0.5).mean() for s in range(6)]))
        names.append(p.stem); fps.append(fp6); fps_mean.append(fpm)

        sc, sf, gc = d["sparse_conv"], d["sparse_fno"], d["gnn_cell"]
        hand = 0.5 * gc + 0.25 * sc + 0.25 * sf
        # Step 6 (final lookahead) IoU
        iou_sc.append(iou_step(sc[5:6], truth[5:6], mask))
        iou_sf.append(iou_step(sf[5:6], truth[5:6], mask))
        iou_gc.append(iou_step(gc[5:6], truth[5:6], mask))
        iou_hand.append(iou_step(hand[5:6], truth[5:6], mask))
        fnr_sc.append(fnr_step(sc[5:6], truth[5:6], mask))
        fnr_sf.append(fnr_step(sf[5:6], truth[5:6], mask))
        fnr_gc.append(fnr_step(gc[5:6], truth[5:6], mask))
        fnr_hand.append(fnr_step(hand[5:6], truth[5:6], mask))

    # ── 2. Tier 1 GNN per-node IoU (from sequences) ────────────────────────
    gnn_ckpt = torch.load(args.gnn_ckpt, weights_only=False, map_location="cpu")
    cfg = gnn_ckpt["config"]
    gnn = SimpleFireGNN(in_feat=5, hidden=cfg.get("hidden", 32),
                          n_graph_layers=cfg.get("n_graph_layers", 2),
                          T_out=cfg.get("T_out", 6))
    gnn.load_state_dict(gnn_ckpt["model"]); gnn.eval()
    adj = build_knn_adjacency(k=cfg.get("knn_k", 4))

    iou_gnn_node = []
    for name in names:
        ds = Tier1FireDataset(args.seq_dir, [name], T_in=6, T_out=6)
        target = None
        for i, (_, t) in enumerate(ds.pairs):
            if t == 6:  # t0 = 120s
                target = i; break
        x, y = ds[target]
        with torch.no_grad():
            y_pred = gnn(x.unsqueeze(0), adj).squeeze(0).numpy()
        p, t = y_pred[:, 5] >= 0.5, y.numpy()[:, 5] >= 0.5
        tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
        iou_gnn_node.append(float(tp / (tp + fp + fn + 1e-9)))

    # ── 3. Load decoder fn=2.5/4.0 IoU from CSV ─────────────────────────
    def load_csv(path):
        d = {}
        if not path.exists(): return d
        with path.open() as f:
            for r in csv.DictReader(f):
                d[r.get("scenario", r.get("name"))] = float(r["iou_step6"])
        return d
    iou_dec25 = [load_csv(args.decoder_fn25_csv).get(n, np.nan) for n in names]
    iou_dec40 = [load_csv(args.decoder_fn40_csv).get(n, np.nan) for n in names]

    # ── 4. Assemble + report correlations ───────────────────────────────
    fps_arr = np.array(fps)
    models = [
        ("① Sparse-ConvLSTM 단독",          np.array(iou_sc)),
        ("② Sparse-FNO 단독",               np.array(iou_sf)),
        ("GNN cell-projected (geo IDW)",     np.array(iou_gc)),
        ("③ Hand-crafted Balanced",         np.array(iou_hand)),
        ("④ Decoder fn=2.5 ★",              np.array(iou_dec25)),
        ("⑤ Decoder fn=4.0",                np.array(iou_dec40)),
        ("α Tier 1 GNN (per-node)",          np.array(iou_gnn_node)),
    ]
    print(f"{'Model':35s}  {'Pearson r':>10s}  {'mean IoU':>10s}  {'IoU std':>9s}  {'min IoU':>9s}")
    print("-" * 80)
    corrs = []
    for label, ious in models:
        valid = ~np.isnan(ious)
        if valid.sum() < 3:
            print(f"{label:35s}  (insufficient data)")
            continue
        r = float(np.corrcoef(fps_arr[valid], ious[valid])[0, 1])
        corrs.append((label, r, float(ious[valid].mean()),
                       float(ious[valid].std()), float(ious[valid].min())))
        print(f"{label:35s}  {r:>+10.3f}  {ious[valid].mean():>10.3f}  "
              f"{ious[valid].std():>9.3f}  {ious[valid].min():>9.3f}")

    # CSV: correlations
    with (args.out_results / "correlations.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "pearson_r", "mean_iou", "std_iou", "min_iou"])
        for r in corrs:
            w.writerow(r)

    # ── 5. Regime split (small / medium / large footprint) ───────────────
    quantiles = np.quantile(fps_arr, [1/3, 2/3])
    regimes = np.where(fps_arr < quantiles[0], "small",
                       np.where(fps_arr < quantiles[1], "medium", "large"))
    print(f"\nRegime split: small < {quantiles[0]*100:.1f}% ≤ medium < {quantiles[1]*100:.1f}% ≤ large")
    print(f"{'Model':35s}  {'small (n=4-5)':>14s}  {'medium':>10s}  {'large':>10s}")
    by_regime = []
    for label, ious in models:
        valid = ~np.isnan(ious)
        rs = {}
        for reg in ["small", "medium", "large"]:
            mask = valid & (regimes == reg)
            rs[reg] = float(ious[mask].mean()) if mask.any() else np.nan
        by_regime.append((label, rs["small"], rs["medium"], rs["large"]))
        print(f"{label:35s}  {rs['small']:>14.3f}  {rs['medium']:>10.3f}  {rs['large']:>10.3f}")
    with (args.out_results / "by_regime.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "small_footprint", "medium_footprint", "large_footprint"])
        for r in by_regime: w.writerow(r)

    # ── 6. Visualizations ────────────────────────────────────────────────
    # Scatter — 7 models, one panel
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, (label, ious) in enumerate(models):
        ax = axes[i]
        valid = ~np.isnan(ious)
        ax.scatter(fps_arr[valid] * 100, ious[valid], s=60, alpha=0.7,
                    edgecolor="black", linewidth=0.5)
        # Pearson
        if valid.sum() >= 3:
            r = float(np.corrcoef(fps_arr[valid], ious[valid])[0, 1])
            # Best fit
            z = np.polyfit(fps_arr[valid] * 100, ious[valid], 1)
            xs = np.linspace(fps_arr.min() * 100, fps_arr.max() * 100, 50)
            ax.plot(xs, z[0] * xs + z[1], "--", color="tab:red", alpha=0.6,
                    label=f"r = {r:+.3f}")
            ax.legend(loc="lower right", fontsize=9)
        ax.axhline(0.70, color="green", lw=0.6, ls=":", alpha=0.7)
        ax.set_xlabel("Fire footprint @ +60s (% of fluid cells)", fontsize=9)
        ax.set_ylabel("IoU @ +60s", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
    # Hide the 8th panel
    axes[7].axis("off")
    fig.suptitle(
        "Model robustness to localised fires — IoU @ +60s vs fire footprint (13 OOD)\n"
        "강한 양의 상관 = 국소 화재에서 IoU 떨어짐",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out_figures / "scatter.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {args.out_figures / 'scatter.png'}")

    # Boxplot by regime
    fig, ax = plt.subplots(1, 1, figsize=(13, 6.5))
    n_models = len(models)
    positions = np.arange(n_models) * 4
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, 3))
    width = 0.9
    for j, reg in enumerate(["small", "medium", "large"]):
        vals = []
        for label, ious in models:
            valid = ~np.isnan(ious) & (regimes == reg)
            vals.append(ious[valid] if valid.any() else np.array([np.nan]))
        bplot = ax.boxplot(vals, positions=positions + j - 1,
                            widths=width, patch_artist=True,
                            boxprops=dict(facecolor=colors[j]),
                            medianprops=dict(color="black", lw=1.5),
                            flierprops=dict(marker="o", markersize=4))
    ax.axhline(0.70, color="red", lw=0.7, ls="--", label="H5 ≥ 0.70")
    ax.set_xticks(positions)
    ax.set_xticklabels([m[0] for m in models], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("IoU @ +60s")
    ax.set_title(
        "IoU by fire-footprint regime (small / medium / large, n≈4-5 each)\n"
        f"small < {quantiles[0]*100:.1f}%   |   "
        f"medium < {quantiles[1]*100:.1f}%   |   large",
        fontsize=11,
    )
    # custom legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[i], label=label)
               for i, label in enumerate(["small footprint", "medium footprint", "large footprint"])]
    handles.append(plt.Line2D([], [], color="red", ls="--", label="H5 ≥ 0.70"))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(args.out_figures / "boxplots.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {args.out_figures / 'boxplots.png'}")

    # Per-scenario bar chart sorted by footprint
    order = np.argsort(fps_arr)
    n_scen = len(names)
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    width = 0.115
    x = np.arange(n_scen)
    for i, (label, ious) in enumerate(models):
        valid_mask = ~np.isnan(ious[order])
        bars_x = (x - 3*width + i*width)
        ax.bar(bars_x[valid_mask], ious[order][valid_mask], width=width,
                label=label, edgecolor="black", linewidth=0.3)
    ax.axhline(0.70, color="red", lw=0.7, ls="--", label="H5 ≥ 0.70")
    # Footprint annotation on x-axis
    labels = [f"{names[i].replace('sim_','').replace('kw','k')}\n({fps_arr[i]*100:.1f}%)"
              for i in order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("IoU @ +60s")
    ax.set_title(
        "Per-scenario IoU (sorted by fire footprint, increasing →)\n"
        "왼쪽 = 가장 국소적 화재 / 오른쪽 = 가장 넓은 화재",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(args.out_figures / "per_scenario.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {args.out_figures / 'per_scenario.png'}")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
