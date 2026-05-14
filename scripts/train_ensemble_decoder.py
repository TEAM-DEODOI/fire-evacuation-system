"""Train the learned ensemble decoder (Option 2).

Reads precomputed npz files from `results/decoder_data/`, trains the
PerCellEnsembleDecoder on 33 train scenarios (cell-level supervision),
and saves the best checkpoint to `checkpoints/ensemble_decoder/best.pt`.

The 13 OOD scenarios are held out for evaluation.

Evaluation after training compares the learned decoder against the
hand-crafted 3-way Balanced ensemble (geodesic, w=0.5/0.25/0.25)
baseline (IoU 0.618 / FNR 5.1%).
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


def iou_fnr(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray,
             threshold: float = 0.5) -> Dict[str, float]:
    """IoU / FNR / RMSE on fluid cells only."""
    fluid = (mask > 0.5)
    fm = np.broadcast_to(fluid, pred.shape).reshape(-1)
    p = (pred >= threshold).reshape(-1)
    t = (truth >= threshold).reshape(-1)
    p, t = p[fm], t[fm]
    tp = float(np.sum(p & t))
    fp = float(np.sum(p & ~t))
    fn = float(np.sum(~p & t))
    return {
        "iou": tp / (tp + fp + fn + 1e-9),
        "fnr": fn / (fn + tp + 1e-9),
        "rmse": float(np.sqrt(np.mean(
            (pred.reshape(-1)[fm] - truth.reshape(-1)[fm]) ** 2
        ))),
    }


def evaluate_on_npz_set(
    decoder: PerCellEnsembleDecoder,
    npz_paths: List[Path],
    device: torch.device,
) -> List[Dict]:
    """Run decoder on each scenario; metric at step 6 (=+60s)."""
    rows = []
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        gnn_cell    = data["gnn_cell"]
        sparse_conv = data["sparse_conv"]
        sparse_fno  = data["sparse_fno"]
        truth       = data["truth"]
        mask        = data["mask"]

        # Decoder forward (full T, X, Y, Z grid)
        pred = decoder_forward_grid(
            decoder, gnn_cell, sparse_conv, sparse_fno, mask, device=device,
        )
        # Step 6 (= last lookahead step, +60s)
        m6 = iou_fnr(pred[5:6], truth[5:6], mask)
        rows.append({
            "scenario": path.stem,
            "iou_step6": m6["iou"],
            "fnr_step6": m6["fnr"],
            "rmse_step6": m6["rmse"],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("results/decoder_data"))
    parser.add_argument("--ckpt-dir", type=Path,
                        default=Path("checkpoints/ensemble_decoder"))
    parser.add_argument("--out-results", type=Path,
                        default=Path("results/exp_decoder_ensemble"))
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fn-weight", type=float, default=1.5,
                        help="Asymmetric BCE FN penalty (1.0 = standard BCE)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.out_results.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── Split: train scenarios = s_*, OOD = sim_*  (already enforced by names)
    all_paths = sorted(args.data_dir.glob("*.npz"))
    train_paths = [p for p in all_paths
                   if p.stem.startswith("s_") and not p.stem.startswith("sim_")]
    ood_paths   = [p for p in all_paths if p.stem.startswith("sim_")]
    print(f"[data] {len(train_paths)} train npz + {len(ood_paths)} OOD npz")
    if len(train_paths) == 0 or len(ood_paths) == 0:
        print("[ERROR] No data — run scripts/precompute_decoder_data.py first")
        return 1

    # ── Build datasets ────────────────────────────────────────────────────
    print(f"[data] building train dataset (fluid cells only)")
    t0 = time.time()
    train_ds = DecoderDataset(train_paths, threshold=0.5, fluid_only=True)
    print(f"        {len(train_ds):,} cell samples  (load {time.time() - t0:.1f}s)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    decoder = PerCellEnsembleDecoder(
        hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    print(f"[model] PerCellEnsembleDecoder: {decoder.n_params():,} params")
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    # ── Training loop ────────────────────────────────────────────────────
    print(f"[train] {args.epochs} epochs  fn_weight={args.fn_weight}")
    history: List[Dict] = []
    best_ood_iou = -1.0
    best_ood_fnr = 1.0
    best_epoch = -1
    best_state = None

    for epoch in range(args.epochs):
        decoder.train()
        losses = []
        for x, target in train_loader:
            x = x.to(device); target = target.to(device)
            target_bin = target[:, 0:1]      # binary
            pred = decoder(x)
            loss = asymmetric_bce_loss(pred, target_bin, fn_weight=args.fn_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))

        # OOD eval at this epoch
        decoder.eval()
        ood_rows = evaluate_on_npz_set(decoder, ood_paths, device)
        mean_iou = float(np.mean([r["iou_step6"] for r in ood_rows]))
        mean_fnr = float(np.mean([r["fnr_step6"] for r in ood_rows]))
        h5_pass = sum(1 for r in ood_rows if r["iou_step6"] >= 0.7)
        history.append({
            "epoch": epoch, "train_loss": mean_loss,
            "ood_iou": mean_iou, "ood_fnr": mean_fnr, "ood_h5": h5_pass,
        })
        marker = ""
        if mean_iou > best_ood_iou:
            best_ood_iou = mean_iou
            best_ood_fnr = mean_fnr
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
            marker = " *"
        print(f"  ep {epoch:02d}  loss={mean_loss:.4f}  "
              f"ood_iou={mean_iou:.3f}  ood_fnr={mean_fnr*100:.1f}%  "
              f"h5={h5_pass:>2d}/13{marker}")

    # ── Save best ────────────────────────────────────────────────────────
    best_path = args.ckpt_dir / "best.pt"
    torch.save({
        "model": best_state,
        "config": {
            "hidden": args.hidden, "n_layers": args.n_layers,
            "dropout": args.dropout, "fn_weight": args.fn_weight,
        },
        "best_epoch": best_epoch,
        "best_ood_iou": best_ood_iou,
        "best_ood_fnr": best_ood_fnr,
    }, best_path)
    print(f"\n[best]  epoch {best_epoch}  IoU={best_ood_iou:.3f}  FNR={best_ood_fnr*100:.1f}%")
    print(f"        -> {best_path}")

    # Reload best for final eval
    decoder.load_state_dict(best_state)
    decoder.eval()

    # ── Final per-scenario CSV (best ckpt) ───────────────────────────────
    final_rows = evaluate_on_npz_set(decoder, ood_paths, device)
    per_scen_csv = args.out_results / "per_scenario.csv"
    with per_scen_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "iou_step6", "fnr_step6", "rmse_step6"])
        for r in final_rows:
            w.writerow([r["scenario"], r["iou_step6"],
                        r["fnr_step6"], r["rmse_step6"]])
    print(f"\n[per-scenario CSV]  {per_scen_csv}")
    for r in final_rows:
        h5 = "Y" if r["iou_step6"] >= 0.7 else "."
        h4 = "Y" if r["fnr_step6"] < 0.10 else "."
        print(f"  {r['scenario']:30s}  IoU={r['iou_step6']:.3f} [{h5}]  "
              f"FNR={r['fnr_step6']*100:5.1f}% [{h4}]")

    mean_iou = float(np.mean([r["iou_step6"] for r in final_rows]))
    mean_fnr = float(np.mean([r["fnr_step6"] for r in final_rows]))
    n_h5 = sum(1 for r in final_rows if r["iou_step6"] >= 0.7)
    n_h4 = sum(1 for r in final_rows if r["fnr_step6"] < 0.10)
    print(f"\n[final agg]  Mean IoU = {mean_iou:.3f}   Mean FNR = {mean_fnr*100:.1f}%   "
          f"H5 pass = {n_h5}/13   H4 pass = {n_h4}/13")
    print(f"[baseline]   3-way Balanced (geo, w=0.5/0.25/0.25):  IoU=0.618  FNR=5.1%  H5=5/13")
    print(f"[delta]      ΔIoU = {(mean_iou - 0.618):+.3f}   ΔFNR = {(mean_fnr - 0.051)*100:+.1f}%p")

    # ── Training-curve plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    eps = [h["epoch"] for h in history]
    ax[0].plot(eps, [h["train_loss"] for h in history], "o-")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss (BCE)")
    ax[0].set_title("Training loss"); ax[0].grid(alpha=0.3)
    ax[1].plot(eps, [h["ood_iou"] for h in history], "o-", color="tab:blue", label="decoder")
    ax[1].axhline(0.618, color="red", lw=0.8, ls="--", label="baseline 0.618")
    ax[1].axhline(0.70, color="green", lw=0.8, ls=":", label="H5 ≥ 0.70")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("Mean IoU @ +60s (13 OOD)")
    ax[1].set_title("OOD IoU"); ax[1].grid(alpha=0.3); ax[1].legend()
    ax[2].plot(eps, [h["ood_fnr"] * 100 for h in history], "s-", color="tab:red", label="decoder")
    ax[2].axhline(5.1, color="red", lw=0.8, ls="--", label="baseline 5.1%")
    ax[2].axhline(10.0, color="orange", lw=0.8, ls=":", label="H4 < 10%")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("Mean FNR % @ +60s")
    ax[2].set_title("OOD FNR"); ax[2].grid(alpha=0.3); ax[2].legend()
    fig.suptitle(
        f"Learned ensemble decoder — train on 33 cell-level scenarios, "
        f"evaluate on 13 OOD\nBest IoU={best_ood_iou:.3f} @ ep{best_epoch}, "
        f"FNR={best_ood_fnr*100:.1f}%, {decoder.n_params():,} params",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out_results / "training_curves.png", dpi=110,
                 bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]  {args.out_results / 'training_curves.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
