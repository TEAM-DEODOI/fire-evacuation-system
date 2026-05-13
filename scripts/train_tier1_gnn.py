"""Train Tier 1 GNN — binary detector signal → future node danger.

Usage:
    python scripts/train_tier1_gnn.py --epochs 100 --batch-size 8
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
from torch.utils.data import DataLoader

from src.tier1.tier1_dataset import Tier1FireDataset, default_splits
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency


def evaluate(
    model: SimpleFireGNN,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute mean MSE + IoU@0.5 over a dataset."""
    model.eval()
    mse_sum, n_pairs = 0.0, 0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x, adj)
            mse_sum += nn.functional.mse_loss(pred, y, reduction="sum").item()
            n_pairs += y.numel()
            p_bin = (pred >= 0.5)
            t_bin = (y >= 0.5)
            tp += (p_bin & t_bin).sum().item()
            fp += (p_bin & (~t_bin)).sum().item()
            fn += ((~p_bin) & t_bin).sum().item()
            tn += ((~p_bin) & (~t_bin)).sum().item()
    iou = tp / (tp + fp + fn + 1e-9)
    fnr = fn / (fn + tp + 1e-9)
    return {
        "mse": mse_sum / max(n_pairs, 1),
        "iou": iou,
        "fnr": fnr,
        "n_pairs": n_pairs // (27 * 6),
    }


def plot_loss_curve(history: List[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_mse = [h["train_mse"] for h in history]
    val_iou = [h.get("val_iou", np.nan) for h in history]
    val_mse = [h.get("val_mse", np.nan) for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, train_mse, label="train MSE", lw=1.5)
    if not all(np.isnan(val_mse)):
        axes[0].plot(epochs, val_mse, label="val MSE", lw=1.5, ls="--")
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE"); axes[0].set_title("Training loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
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
                        default=Path("checkpoints/tier1_gnn"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-graph-layers", type=int, default=2)
    parser.add_argument("--T-in", type=int, default=6)
    parser.add_argument("--T-out", type=int, default=6)
    parser.add_argument("--knn-k", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device}")

    # 1) Data
    train_names, val_names, test_names = default_splits()
    train_ds = Tier1FireDataset(args.sequence_dir, train_names, args.T_in, args.T_out)
    val_ds   = Tier1FireDataset(args.sequence_dir, val_names,   args.T_in, args.T_out)
    test_ds  = Tier1FireDataset(args.sequence_dir, test_names,  args.T_in, args.T_out)
    print(f"[setup] train={len(train_ds)} pairs, val={len(val_ds)}, test={len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # 2) Graph adjacency (shared, fixed)
    adj = build_knn_adjacency(k=args.knn_k).to(device)
    print(f"[setup] adj nonzero: {(adj > 0).sum().item()} / {adj.numel()}")

    # 3) Model
    model = SimpleFireGNN(
        in_feat=5, hidden=args.hidden,
        n_graph_layers=args.n_graph_layers, T_out=args.T_out,
    ).to(device)
    print(f"[setup] model parameters: {model.count_parameters():,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4) Training loop
    history = []
    best_val_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for x, y in train_dl:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            pred = model(x, adj)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()
        train_mse = float(np.mean(epoch_losses))
        rec = {"epoch": epoch, "train_mse": train_mse}

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            val = evaluate(model, val_dl, adj, device)
            rec["val_mse"] = val["mse"]
            rec["val_iou"] = val["iou"]
            rec["val_fnr"] = val["fnr"]
            if val["iou"] > best_val_iou:
                best_val_iou = val["iou"]
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch, "val_iou": val["iou"], "val_mse": val["mse"],
                    "config": vars(args),
                }, args.output / "best.pt")
            print(f"  ep {epoch:3d}/{args.epochs}  train_mse={train_mse:.5f}  "
                  f"val_mse={val['mse']:.5f}  val_iou={val['iou']:.3f}  "
                  f"val_fnr={val['fnr']*100:.1f}%  lr={scheduler.get_last_lr()[0]:.2e}")
        else:
            print(f"  ep {epoch:3d}/{args.epochs}  train_mse={train_mse:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
        history.append(rec)

    # 5) Final test
    print(f"\n[test] running on {len(test_names)} OOD scenarios")
    ckpt = torch.load(args.output / "best.pt", weights_only=False,
                       map_location=device)
    model.load_state_dict(ckpt["model"])
    test_res = evaluate(model, test_dl, adj, device)
    print(f"  test mse: {test_res['mse']:.5f}")
    print(f"  test IoU: {test_res['iou']:.3f}  (H5 target ≥ 0.70: "
          f"{'PASS' if test_res['iou'] >= 0.70 else 'FAIL'})")
    print(f"  test FNR: {test_res['fnr']*100:.1f}%  (H4 target < 10%: "
          f"{'PASS' if test_res['fnr'] < 0.10 else 'FAIL'})")

    # 6) Save final ckpt + curves
    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs,
        "test_iou": test_res["iou"], "test_fnr": test_res["fnr"],
        "config": vars(args),
    }, args.output / "final.pt")
    plot_loss_curve(history, args.output / "loss_curve.png")
    np.savetxt(args.output / "history.csv",
                [[h["epoch"], h["train_mse"], h.get("val_mse", -1),
                  h.get("val_iou", -1)] for h in history],
                fmt="%.6f", delimiter=",",
                header="epoch,train_mse,val_mse,val_iou")

    print(f"\n[PASS]")
    print(f"  best.pt @ epoch {ckpt['epoch']} (val_iou {best_val_iou:.3f})")
    print(f"  artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
