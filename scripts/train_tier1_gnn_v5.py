"""Train Tier 1 GNN v5 — v4 + Tversky loss + wider architecture.

Building on v4 (focal+asymmetric BCE + small-fire boost):
    + Tversky loss term (IoU-aligned, lambda_tversky=0.5)
    + hidden 32 → 48 (params 12K → ~22K, still small)
    + graph layers 2 → 3 (deeper message passing)

Tversky variant of Dice: TI = TP / (TP + α·FN + β·FP), with α=0.7, β=0.3
weights FN more heavily (consistent with asymmetric BCE's safety bias).

The v5 ckpt is saved separately at checkpoints/tier1_gnn_v5/ so v3 and v4
remain untouched for ablation comparison.
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.tier1.tier1_dataset import Tier1FireDataset, default_splits
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency
from train_tier1_gnn_v4 import (
    focal_asymmetric_bce, compute_scenario_weights, evaluate,
)


# ─── Tversky loss ─────────────────────────────────────────────────────────
def soft_tversky_loss(
    pred: torch.Tensor,      # (B, N, T) ∈ [0, 1]
    target: torch.Tensor,    # (B, N, T) ∈ [0, 1]
    alpha: float = 0.7,      # FN weight
    beta: float = 0.3,       # FP weight
    eps: float = 1e-7,
) -> torch.Tensor:
    """1 - Tversky index. α > β biases against false negatives (safety-friendly,
    matches asymmetric BCE).

    Tversky index = TP / (TP + α·FN + β·FP). At α=β=0.5 this is Dice;
    at α=β=1.0 this is IoU.

    Computed on the continuous outputs (soft) for differentiability.
    """
    y = (target >= 0.5).float()
    tp = (pred * y).sum()
    fn = ((1 - pred) * y).sum()
    fp = (pred * (1 - y)).sum()
    ti = tp / (tp + alpha * fn + beta * fp + eps)
    return 1.0 - ti


def combined_loss(
    pred: torch.Tensor, target: torch.Tensor,
    gamma: float = 2.0, fn_weight: float = 2.5,
    lambda_tversky: float = 0.5,
    tversky_alpha: float = 0.7, tversky_beta: float = 0.3,
) -> torch.Tensor:
    bce = focal_asymmetric_bce(pred, target, gamma=gamma, fn_weight=fn_weight)
    tv  = soft_tversky_loss(pred, target,
                              alpha=tversky_alpha, beta=tversky_beta)
    return bce + lambda_tversky * tv


def plot_loss_curve(history: List[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_iou = [h.get("val_iou", np.nan) for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, train_loss, lw=1.5)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("combined loss")
    axes[0].set_title("Training loss (v5)"); axes[0].grid(alpha=0.3)
    if not all(np.isnan(val_iou)):
        axes[1].plot(epochs, val_iou, color="tab:green", lw=1.5)
        axes[1].axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Val IoU @ 0.5")
        axes[1].set_title("Validation IoU"); axes[1].legend()
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/tier1_gnn_v5"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    # ── Architecture ↑ ─────────────────────────────────────────────────────
    parser.add_argument("--hidden", type=int, default=48)         # was 32
    parser.add_argument("--n-graph-layers", type=int, default=3)  # was 2
    # ────────────────────────────────────────────────────────────────────────
    parser.add_argument("--T-in", type=int, default=6)
    parser.add_argument("--T-out", type=int, default=6)
    parser.add_argument("--knn-k", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--fn-weight", type=float, default=2.5)
    parser.add_argument("--lambda-tversky", type=float, default=0.5)
    parser.add_argument("--tversky-alpha", type=float, default=0.7)
    parser.add_argument("--tversky-beta", type=float, default=0.3)
    parser.add_argument("--small-fire-boost", type=float, default=2.0)
    parser.add_argument("--small-fire-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device}")
    print(f"[setup] focal γ={args.gamma}, fn_weight={args.fn_weight}, "
          f"λ_tversky={args.lambda_tversky} (α={args.tversky_alpha}, β={args.tversky_beta})")
    print(f"[setup] arch: hidden={args.hidden}, graph_layers={args.n_graph_layers}")
    print(f"[setup] small-fire boost {args.small_fire_boost}× "
          f"(threshold {args.small_fire_threshold})")

    # 1) Data
    train_names, val_names, test_names = default_splits()
    train_ds = Tier1FireDataset(args.sequence_dir, train_names, args.T_in, args.T_out)
    val_ds   = Tier1FireDataset(args.sequence_dir, val_names,   args.T_in, args.T_out)
    test_ds  = Tier1FireDataset(args.sequence_dir, test_names,  args.T_in, args.T_out)
    print(f"[setup] train={len(train_ds)} pairs, val={len(val_ds)}, test={len(test_ds)}")

    if abs(args.small_fire_boost - 1.0) > 1e-6:
        weights = compute_scenario_weights(
            train_ds, boost=args.small_fire_boost,
            threshold=args.small_fire_threshold,
        )
        n_boosted = int((weights > 1.0).sum())
        print(f"[setup] {n_boosted}/{len(train_ds)} pairs boosted")
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_ds), replacement=True,
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2) Graph
    adj = build_knn_adjacency(k=args.knn_k).to(device)

    # 3) Model (wider + deeper)
    model = SimpleFireGNN(
        in_feat=5, hidden=args.hidden,
        n_graph_layers=args.n_graph_layers, T_out=args.T_out,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] model parameters: {n_params:,}  (v3/v4 = 12,006)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4) Train loop
    history = []
    best_val_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_dl:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            pred = model(x, adj)
            loss = combined_loss(pred, y,
                                   gamma=args.gamma, fn_weight=args.fn_weight,
                                   lambda_tversky=args.lambda_tversky,
                                   tversky_alpha=args.tversky_alpha,
                                   tversky_beta=args.tversky_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(losses))
        rec = {"epoch": epoch, "train_loss": train_loss}

        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            val = evaluate(model, val_dl, adj, device)
            rec["val_iou"] = val["iou"]; rec["val_fnr"] = val["fnr"]; rec["val_mse"] = val["mse"]
            marker = ""
            if val["iou"] > best_val_iou:
                best_val_iou = val["iou"]
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch, "val_iou": val["iou"],
                    "config": vars(args),
                }, args.output / "best.pt")
                marker = " *"
            print(f"  ep {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
                  f"val_iou={val['iou']:.3f}  val_fnr={val['fnr']*100:.1f}%"
                  f"  lr={scheduler.get_last_lr()[0]:.2e}{marker}")
        history.append(rec)

    # 5) Test
    print(f"\n[test] running on {len(test_names)} OOD scenarios using best.pt")
    ckpt = torch.load(args.output / "best.pt", weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_res = evaluate(model, test_dl, adj, device)
    print(f"  test IoU: {test_res['iou']:.3f}  "
          f"(H5 ≥ 0.70: {'PASS' if test_res['iou'] >= 0.70 else 'FAIL'})")
    print(f"  test FNR: {test_res['fnr']*100:.1f}%  "
          f"(H4 < 10%: {'PASS' if test_res['fnr'] < 0.10 else 'FAIL'})")

    torch.save({
        "model": model.state_dict(), "epoch": args.epochs,
        "test_iou": test_res["iou"], "test_fnr": test_res["fnr"],
        "config": vars(args),
    }, args.output / "final.pt")
    plot_loss_curve(history, args.output / "loss_curve.png")
    np.savetxt(args.output / "history.csv",
                [[h["epoch"], h["train_loss"], h.get("val_iou", -1),
                  h.get("val_fnr", -1)] for h in history],
                fmt="%.6f", delimiter=",",
                header="epoch,train_loss,val_iou,val_fnr")

    print(f"\n[PASS]")
    print(f"  best.pt @ epoch {ckpt['epoch']} (val_iou {best_val_iou:.3f})")
    print(f"  params: {n_params:,}  |  v3/v4 = 12,006")
    print(f"  v5 artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
