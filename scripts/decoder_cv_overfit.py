"""5-fold cross-validation on the 33 train scenarios — check whether the
decoder is overfitting (33 scenarios is a small training set).

The OOD-only number from train_ensemble_decoder.py (IoU 0.733 / FNR 11.5%
at fn=2.5) is the test result; this script measures the **train-vs-
held-out generalization gap** *within* the 33-scenario training pool.

For each of 5 folds:
- Hold out 6-7 train scenarios as "fold-val".
- Train a fresh decoder on the remaining ~26 train scenarios.
- Evaluate on the fold-val + the unchanged 13-OOD set.
- Record gap (= OOD IoU − fold-val IoU); a small gap means the decoder
  generalises uniformly.

Output:
  results/exp_decoder_cv/cv_summary.csv
  figures/current/14_decoder_cv/cv_bars.png

Note: this is "scenario-level" CV, not cell-level CV. Hold-out scenarios
share the building geometry; the model may still overfit room-specific
features, but at least we measure across-scenario robustness.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.tier1.ensemble_decoder import (
    DecoderDataset, PerCellEnsembleDecoder, asymmetric_bce_loss,
    decoder_forward_grid,
)


def iou_fnr(pred, truth, mask, threshold=0.5) -> Dict[str, float]:
    fluid = (mask > 0.5)
    fm = np.broadcast_to(fluid, pred.shape).reshape(-1)
    p = (pred >= threshold).reshape(-1)[fm]
    t = (truth >= threshold).reshape(-1)[fm]
    tp = float(np.sum(p & t)); fp = float(np.sum(p & ~t)); fn = float(np.sum(~p & t))
    return {
        "iou": tp / (tp + fp + fn + 1e-9),
        "fnr": fn / (fn + tp + 1e-9),
    }


def eval_paths(decoder, paths, device) -> Dict:
    rows = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        pred = decoder_forward_grid(
            decoder, data["gnn_cell"], data["sparse_conv"], data["sparse_fno"],
            data["mask"], device=device,
        )
        m6 = iou_fnr(pred[5:6], data["truth"][5:6], data["mask"])
        rows.append({"scenario": path.stem, **m6})
    ious = [r["iou"] for r in rows]; fnrs = [r["fnr"] for r in rows]
    return {
        "rows": rows,
        "mean_iou": float(np.mean(ious)),
        "mean_fnr": float(np.mean(fnrs)),
        "n_h5": sum(1 for v in ious if v >= 0.7),
    }


def train_decoder(
    train_paths: List[Path], device, fn_weight: float = 2.5,
    hidden: int = 32, n_layers: int = 2,
    epochs: int = 30, lr: float = 1e-3, batch_size: int = 2048,
):
    train_ds = DecoderDataset(train_paths, threshold=0.5, fluid_only=True)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                         num_workers=0)
    decoder = PerCellEnsembleDecoder(hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    for ep in range(epochs):
        decoder.train()
        for x, target in loader:
            x = x.to(device); target = target.to(device)
            pred = decoder(x)
            loss = asymmetric_bce_loss(pred, target[:, 0:1], fn_weight=fn_weight)
            opt.zero_grad(); loss.backward(); opt.step()
    return decoder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("results/decoder_data"))
    parser.add_argument("--out-results", type=Path,
                        default=Path("results/exp_decoder_cv"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/current/14_decoder_cv"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fn-weight", type=float, default=2.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.out_results.mkdir(parents=True, exist_ok=True)
    args.out_figures.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    all_paths = sorted(args.data_dir.glob("*.npz"))
    train_paths = sorted([p for p in all_paths
                            if p.stem.startswith("s_") and not p.stem.startswith("sim_")])
    ood_paths   = sorted([p for p in all_paths if p.stem.startswith("sim_")])
    print(f"[data] {len(train_paths)} train + {len(ood_paths)} OOD")

    # ── 5-fold split of 33 train scenarios ─────────────────────────────
    indices = np.arange(len(train_paths))
    np.random.shuffle(indices)
    folds = np.array_split(indices, args.n_folds)
    print(f"[folds] {[len(f) for f in folds]} held-out scenarios per fold")

    fold_results = []
    for fold_idx, val_idxs in enumerate(folds):
        val_paths = [train_paths[i] for i in val_idxs]
        tr_paths = [train_paths[i] for i in indices if i not in set(val_idxs)]
        print(f"\n[fold {fold_idx}] train={len(tr_paths)}  fold-val={len(val_paths)}")
        t0 = time.time()
        decoder = train_decoder(
            tr_paths, device, fn_weight=args.fn_weight,
            epochs=args.epochs,
        )
        decoder.eval()
        train_eval  = eval_paths(decoder, tr_paths,  device)
        val_eval    = eval_paths(decoder, val_paths, device)
        ood_eval    = eval_paths(decoder, ood_paths, device)
        elapsed = time.time() - t0

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(tr_paths),
            "n_val": len(val_paths),
            "train_iou":   train_eval["mean_iou"],
            "fold_val_iou": val_eval["mean_iou"],
            "ood_iou":     ood_eval["mean_iou"],
            "train_fnr":   train_eval["mean_fnr"],
            "fold_val_fnr": val_eval["mean_fnr"],
            "ood_fnr":     ood_eval["mean_fnr"],
            "ood_h5":      ood_eval["n_h5"],
            "gap_train_val":     train_eval["mean_iou"] - val_eval["mean_iou"],
            "gap_train_ood":     train_eval["mean_iou"] - ood_eval["mean_iou"],
            "elapsed_s":   elapsed,
        })
        print(f"  train IoU  = {train_eval['mean_iou']:.3f}  FNR = {train_eval['mean_fnr']*100:5.1f}%")
        print(f"  fold-val   = {val_eval['mean_iou']:.3f}  FNR = {val_eval['mean_fnr']*100:5.1f}%   gap = {fold_results[-1]['gap_train_val']:+.3f}")
        print(f"  OOD  (13)  = {ood_eval['mean_iou']:.3f}  FNR = {ood_eval['mean_fnr']*100:5.1f}%   gap = {fold_results[-1]['gap_train_ood']:+.3f}   H5 = {ood_eval['n_h5']}/13")
        print(f"  elapsed {elapsed:.1f}s")

    # ── Aggregate ──────────────────────────────────────────────────────
    print("\n[summary]")
    print(f"  {'fold':>4}  {'train':>7}  {'foldval':>8}  {'ood':>7}  "
          f"{'gap_tv':>7}  {'gap_to':>7}")
    for r in fold_results:
        print(f"  {r['fold']:>4}  {r['train_iou']:>7.3f}  {r['fold_val_iou']:>8.3f}  "
              f"{r['ood_iou']:>7.3f}  {r['gap_train_val']:>+7.3f}  {r['gap_train_ood']:>+7.3f}")

    keys = ["train_iou", "fold_val_iou", "ood_iou",
             "gap_train_val", "gap_train_ood"]
    means = {k: float(np.mean([r[k] for r in fold_results])) for k in keys}
    stds  = {k: float(np.std([r[k] for r in fold_results])) for k in keys}
    print(f"  {'mean':>4}  {means['train_iou']:>7.3f}  {means['fold_val_iou']:>8.3f}  "
          f"{means['ood_iou']:>7.3f}  {means['gap_train_val']:>+7.3f}  {means['gap_train_ood']:>+7.3f}")
    print(f"  {'std':>4}  {stds['train_iou']:>7.3f}  {stds['fold_val_iou']:>8.3f}  "
          f"{stds['ood_iou']:>7.3f}  {stds['gap_train_val']:>7.3f}  {stds['gap_train_ood']:>7.3f}")

    # ── CSV ────────────────────────────────────────────────────────────
    csv_path = args.out_results / "cv_summary.csv"
    cols = ["fold","n_train","n_val","train_iou","fold_val_iou","ood_iou",
            "train_fnr","fold_val_fnr","ood_fnr","ood_h5",
            "gap_train_val","gap_train_ood","elapsed_s"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in fold_results:
            w.writerow([r[c] for c in cols])
    print(f"\n[csv]  {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    folds_x = np.arange(len(fold_results))
    width = 0.25
    ax.bar(folds_x - width, [r["train_iou"]    for r in fold_results],
            width, label="train", color="tab:blue", edgecolor="black")
    ax.bar(folds_x,         [r["fold_val_iou"] for r in fold_results],
            width, label="fold-val (held-out 7 train)",
            color="tab:orange", edgecolor="black")
    ax.bar(folds_x + width, [r["ood_iou"]      for r in fold_results],
            width, label="13 OOD",
            color="darkred", edgecolor="black")
    ax.axhline(0.733, color="tab:gray", ls="--", lw=0.8,
                label="full-train OOD baseline 0.733")
    ax.axhline(0.70, color="green", ls=":", lw=0.8, label="H5 ≥ 0.70")
    ax.set_xticks(folds_x); ax.set_xticklabels([f"Fold {i}" for i in folds_x])
    ax.set_ylabel("Mean IoU @ +60s")
    ax.set_title(
        f"5-fold CV on 33 train scenarios — checks decoder overfit\n"
        f"fn_weight={args.fn_weight}, {args.epochs} epochs, each fold trained from scratch"
    )
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    ax.set_ylim(0, 1.0)
    for i, r in enumerate(fold_results):
        ax.text(i - width, r["train_iou"] + 0.01,
                 f"{r['train_iou']:.3f}", ha="center", fontsize=8)
        ax.text(i,         r["fold_val_iou"] + 0.01,
                 f"{r['fold_val_iou']:.3f}", ha="center", fontsize=8)
        ax.text(i + width, r["ood_iou"] + 0.01,
                 f"{r['ood_iou']:.3f}", ha="center", fontsize=8)
    out = args.out_figures / "cv_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")

    print(f"\n[PASS]  mean train-vs-OOD gap = {means['gap_train_ood']:+.3f}  "
          f"(std {stds['gap_train_ood']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
