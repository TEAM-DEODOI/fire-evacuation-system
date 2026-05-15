"""Train Tier 1 GNN v4 — focal + asymmetric BCE for small-fire robustness.

Motivation:
    v3 (MSE loss) struggles on small-fire scenarios (500kW × 1m²):
      - s_029  (sim_500kw_1m2_H03)  IoU 0.565  ← user-reported
      - s_021  (sim_500kw_1m2_008)  IoU 0.450
      - s_022  (sim_500kw_1m2_009)  IoU 0.474
      Train mean IoU @ t0=120s = 0.820, std 0.130.
    Worst 10 train scenarios are all 500kW small fires.
    Root cause: MSE collapses on class-imbalanced node targets (most nodes
    safe → MSE small with "all 0" prediction).

Loss design:
    L = mean( focal · (fn_weight · y · -log(p) + (1-y) · -log(1-p)) )
        where  focal = (1 - pt)^gamma,  pt = p if y≥0.5 else 1-p
    - focal (gamma=2.0): downweight easy-correct nodes, focus on hard mismatches
    - asymmetric (fn_weight=2.5): false negatives 2.5× more expensive than FP

Optional: per-scenario re-weighting based on truth dangerous-node fraction
(--small-fire-boost). Larger weight to scenarios with rare positives.

The v4 checkpoint is saved separately at checkpoints/tier1_gnn_v4/ so the
v3 paper-main result (OOD IoU 0.904) is preserved untouched.
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


# ─── Loss ─────────────────────────────────────────────────────────────────
def focal_asymmetric_bce(
    pred: torch.Tensor,    # (B, N, T_out) ∈ [0, 1]
    target: torch.Tensor,  # (B, N, T_out) ∈ [0, 1] (continuous danger, threshold 0.5)
    gamma: float = 2.0,
    fn_weight: float = 2.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Focal + asymmetric BCE on node danger predictions.

    target ≥ 0.5 → positive (danger), else negative (safe).

    Args:
        pred:    sigmoid output ∈ [0, 1].
        target:  continuous truth in [0, 1].
        gamma:   focal exponent. 0 = standard BCE, 2 = strong focus.
        fn_weight: FN penalty multiplier. 1 = symmetric, >1 = favor recall.
    """
    p = pred.clamp(eps, 1 - eps)
    y = (target >= 0.5).float()
    pt = torch.where(y == 1, p, 1 - p)
    focal = (1 - pt) ** gamma
    fn_term = -fn_weight * y * torch.log(p)
    fp_term = -(1 - y) * torch.log(1 - p)
    return (focal * (fn_term + fp_term)).mean()


# ─── Sample weighting (optional) ──────────────────────────────────────────
def compute_scenario_weights(
    ds: Tier1FireDataset, boost: float = 2.0, threshold: float = 0.3,
) -> np.ndarray:
    """Per-pair weight: scenarios with truth dangerous-node fraction below
    `threshold` at the target window's last step receive `boost`× weight.

    Returns:
        (n_pairs,) numpy array of weights for WeightedRandomSampler.
    """
    weights = np.ones(len(ds), dtype=np.float32)
    for i, (s_idx, t) in enumerate(ds.pairs):
        scen = ds.scenarios[s_idx]
        # target window's last step danger
        end_idx = t + ds.T_in + ds.T_out - 1
        end_idx = min(end_idx, scen["danger"].shape[0] - 1)
        frac_pos = float((scen["danger"][end_idx] >= 0.5).mean())
        if frac_pos < threshold:
            weights[i] = boost
    return weights


# ─── Evaluation ───────────────────────────────────────────────────────────
def evaluate(
    model: SimpleFireGNN,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
) -> dict:
    model.eval()
    mse_sum, n_elem = 0.0, 0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x, adj)
            mse_sum += nn.functional.mse_loss(pred, y, reduction="sum").item()
            n_elem += y.numel()
            p_bin = (pred >= 0.5); t_bin = (y >= 0.5)
            tp += (p_bin & t_bin).sum().item()
            fp += (p_bin & (~t_bin)).sum().item()
            fn += ((~p_bin) & t_bin).sum().item()
            tn += ((~p_bin) & (~t_bin)).sum().item()
    return {
        "mse": mse_sum / max(n_elem, 1),
        "iou": tp / (tp + fp + fn + 1e-9),
        "fnr": fn / (fn + tp + 1e-9),
    }


def plot_loss_curve(history: List[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_iou = [h.get("val_iou", np.nan) for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, train_loss, lw=1.5)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Focal+asym BCE loss")
    axes[0].set_title("Training loss (v4)"); axes[0].grid(alpha=0.3)
    if not all(np.isnan(val_iou)):
        axes[1].plot(epochs, val_iou, color="tab:green", lw=1.5)
        axes[1].axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Val IoU @ 0.5")
        axes[1].set_title("Validation IoU"); axes[1].legend()
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/tier1_gnn_v4"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-graph-layers", type=int, default=2)
    parser.add_argument("--T-in", type=int, default=6)
    parser.add_argument("--T-out", type=int, default=6)
    parser.add_argument("--knn-k", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="focal exponent (0 = standard BCE, 2 = strong focus)")
    parser.add_argument("--fn-weight", type=float, default=2.5,
                        help="FN penalty multiplier (1 = symmetric)")
    parser.add_argument("--small-fire-boost", type=float, default=2.0,
                        help="WeightedRandomSampler boost for small-fire pairs. "
                             "1.0 = uniform.")
    parser.add_argument("--small-fire-threshold", type=float, default=0.3,
                        help="frac_positive_nodes < threshold ⇒ small-fire boost.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device}")
    print(f"[setup] focal γ={args.gamma}, fn_weight={args.fn_weight}, "
          f"small-fire boost={args.small_fire_boost}× (threshold {args.small_fire_threshold})")

    # 1) Data
    train_names, val_names, test_names = default_splits()
    train_ds = Tier1FireDataset(args.sequence_dir, train_names, args.T_in, args.T_out)
    val_ds   = Tier1FireDataset(args.sequence_dir, val_names,   args.T_in, args.T_out)
    test_ds  = Tier1FireDataset(args.sequence_dir, test_names,  args.T_in, args.T_out)
    print(f"[setup] train={len(train_ds)} pairs, val={len(val_ds)}, test={len(test_ds)}")

    # 1a) Optional small-fire oversampling
    if abs(args.small_fire_boost - 1.0) > 1e-6:
        weights = compute_scenario_weights(
            train_ds, boost=args.small_fire_boost,
            threshold=args.small_fire_threshold,
        )
        n_boosted = int((weights > 1.0).sum())
        print(f"[setup] {n_boosted}/{len(train_ds)} pairs boosted (fraction "
              f"{n_boosted/len(train_ds)*100:.1f}%)")
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_ds), replacement=True,
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2) Graph adjacency
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
            loss = focal_asymmetric_bce(pred, y,
                                         gamma=args.gamma,
                                         fn_weight=args.fn_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        rec = {"epoch": epoch, "train_loss": train_loss}

        # Validate every 5 epochs (+ first + last)
        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            val = evaluate(model, val_dl, adj, device)
            rec["val_mse"] = val["mse"]
            rec["val_iou"] = val["iou"]
            rec["val_fnr"] = val["fnr"]
            marker = ""
            if val["iou"] > best_val_iou:
                best_val_iou = val["iou"]
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch, "val_iou": val["iou"], "val_mse": val["mse"],
                    "config": vars(args),
                }, args.output / "best.pt")
                marker = " *"
            print(f"  ep {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
                  f"val_iou={val['iou']:.3f}  val_fnr={val['fnr']*100:.1f}%"
                  f"  lr={scheduler.get_last_lr()[0]:.2e}{marker}")
        history.append(rec)

    # 5) Final test (on best.pt)
    print(f"\n[test] running on {len(test_names)} OOD scenarios using best.pt")
    ckpt = torch.load(args.output / "best.pt", weights_only=False,
                       map_location=device)
    model.load_state_dict(ckpt["model"])
    test_res = evaluate(model, test_dl, adj, device)
    print(f"  test IoU: {test_res['iou']:.3f}  "
          f"(H5 ≥ 0.70: {'PASS' if test_res['iou'] >= 0.70 else 'FAIL'})")
    print(f"  test FNR: {test_res['fnr']*100:.1f}%  "
          f"(H4 < 10%: {'PASS' if test_res['fnr'] < 0.10 else 'FAIL'})")

    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs,
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
    print(f"  v4 artifacts: {args.output}")
    print(f"  v3 (paper main) preserved at: checkpoints/tier1_gnn_v3/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
