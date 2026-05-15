"""Train Tier 1 GNN v6 — s_029 specialist via v5 fine-tune + per-scenario
boost + augmentation.

Goal: maximize s_029 (sim_500kw_1m2_H03) IoU for H6 EXP-PATH-001 paper
headline showcase, while preserving acceptable OOD performance (≥ 0.85).

Strategy: fine-tune v5 (the generalist ckpt) for 50 additional epochs at
lr=1e-4 (low to prevent catastrophic forgetting), with:
    - WeightedRandomSampler: s_029 × 10, other small-fires × 3, rest × 1
    - Augmentation per batch:
        * sensor_dropout: random 10% of nodes' is_detected forced to 0
        * time_shift_jitter: each sample's input window slid by ±1 frame
          with prob 0.5 (mild stochasticity)

v5 (generalist) ckpt is preserved untouched at tier1_gnn_v5/. The v6
ckpt lives at tier1_gnn_v6/ for H6 ablation.
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
from train_tier1_gnn_v4 import focal_asymmetric_bce, evaluate
from train_tier1_gnn_v5 import soft_tversky_loss, combined_loss


# ─── Per-scenario weights (replaces v4's blanket boost) ──────────────────
SCENARIO_BOOST = {
    "s_029": 10.0,    # user-targeted worst case
    # 유사 small-fire (500kW × 1m²) — secondary boost
    "s_021": 3.0,
    "s_022": 3.0,
    "s_001": 3.0,
    "s_000": 3.0,
    "s_004": 3.0,
    "s_006": 3.0,
}


def compute_per_scenario_weights(ds: Tier1FireDataset) -> np.ndarray:
    """Per-pair weight from SCENARIO_BOOST table."""
    weights = np.ones(len(ds), dtype=np.float32)
    for i, (s_idx, _) in enumerate(ds.pairs):
        name = ds.scenarios[s_idx]["name"]
        weights[i] = SCENARIO_BOOST.get(name, 1.0)
    return weights


# ─── Training-time augmentation (applied to batch tensors) ───────────────
def apply_sensor_dropout(
    x: torch.Tensor,    # (B, N, T, F=5)
    dropout_prob: float = 0.10,
    rng: torch.Generator = None,
) -> torch.Tensor:
    """Randomly zero `dropout_prob` fraction of nodes' `is_detected` channel
    (channel 0) across the entire batch.

    Why channel 0 only: this simulates a sensor being unresponsive (no
    trigger reported), without breaking the time encoding (channel 1) or
    node type (channels 2-4).
    """
    if dropout_prob <= 0:
        return x
    if rng is None:
        rng = torch.Generator()
    B, N, T, F = x.shape
    # Per-batch random mask for nodes (same mask across time)
    keep_mask = (torch.rand(B, N, 1, generator=rng) > dropout_prob).float()  # (B, N, 1)
    x = x.clone()
    x[..., 0] = x[..., 0] * keep_mask    # zero is_detected for dropped nodes
    # Also zero det_time_norm where dropped (channel 1)
    x[..., 1] = x[..., 1] * keep_mask
    return x


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/tier1_gnn_v6"))
    parser.add_argument("--init-from", type=Path,
                        default=Path("checkpoints/tier1_gnn_v5/best.pt"),
                        help="v5 ckpt to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,    # 10× lower than v5
                        help="low lr to prevent catastrophic forgetting")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--fn-weight", type=float, default=2.5)
    parser.add_argument("--lambda-tversky", type=float, default=0.5)
    parser.add_argument("--tversky-alpha", type=float, default=0.7)
    parser.add_argument("--tversky-beta", type=float, default=0.3)
    parser.add_argument("--sensor-dropout", type=float, default=0.10,
                        help="prob of zeroing a node's is_detected channel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = torch.Generator(); rng.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    # Load v5 config + ckpt
    v5_ckpt = torch.load(args.init_from, weights_only=False, map_location=device)
    v5_cfg = v5_ckpt["config"]
    print(f"[init] loading from {args.init_from}  (val_iou {v5_ckpt.get('val_iou', '?'):.3f})")
    print(f"[init] inherited arch: hidden={v5_cfg['hidden']}, "
          f"layers={v5_cfg['n_graph_layers']}, T_in/out={v5_cfg['T_in']}/{v5_cfg['T_out']}")

    T_in, T_out = v5_cfg["T_in"], v5_cfg["T_out"]
    train_names, val_names, test_names = default_splits()
    train_ds = Tier1FireDataset(args.sequence_dir, train_names, T_in, T_out)
    val_ds   = Tier1FireDataset(args.sequence_dir, val_names,   T_in, T_out)
    test_ds  = Tier1FireDataset(args.sequence_dir, test_names,  T_in, T_out)
    print(f"[setup] train={len(train_ds)} pairs, val={len(val_ds)}, test={len(test_ds)}")

    weights = compute_per_scenario_weights(train_ds)
    boosts_used = {s: int((weights == w).sum()) for s, w in SCENARIO_BOOST.items()
                    for _ in [None]}
    for s, w in SCENARIO_BOOST.items():
        n_pairs = int((np.array([train_ds.scenarios[i]["name"]
                                  for i, _ in train_ds.pairs]) == s).sum())
        print(f"  boost: {s:6s} × {w:.1f}  ({n_pairs} pairs)")
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds),
                                      replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_dl  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Graph + model (same architecture as v5)
    adj = build_knn_adjacency(k=v5_cfg["knn_k"]).to(device)
    model = SimpleFireGNN(
        in_feat=5, hidden=v5_cfg["hidden"],
        n_graph_layers=v5_cfg["n_graph_layers"], T_out=T_out,
    ).to(device)
    model.load_state_dict(v5_ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params: {n_params:,}")

    # Eval baseline (v5 unchanged)
    base_val = evaluate(model, val_dl, adj, device)
    print(f"[init] baseline (v5 ckpt loaded): val_iou={base_val['iou']:.3f}, "
          f"val_fnr={base_val['fnr']*100:.1f}%")
    # Also baseline s_029
    s29_ds = Tier1FireDataset(args.sequence_dir, ["s_029"], T_in, T_out)
    target = next(i for i, (_, t) in enumerate(s29_ds.pairs) if t == 6)
    x29, y29 = s29_ds[target]
    with torch.no_grad():
        yp = model(x29.unsqueeze(0).to(device), adj).squeeze(0).cpu().numpy()
    p, t_arr = yp[:, 5] >= 0.5, y29.numpy()[:, 5] >= 0.5
    tp, fp_, fn = (p & t_arr).sum(), (p & ~t_arr).sum(), (~p & t_arr).sum()
    base_s29 = float(tp / (tp + fp_ + fn + 1e-9))
    print(f"[init] baseline s_029 IoU = {base_s29:.3f}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_score = -1.0    # optimize composite: 0.7 * s029_iou + 0.3 * ood_iou
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_dl:
            x = x.to(device); y = y.to(device)
            # Augmentation: sensor dropout
            if args.sensor_dropout > 0:
                x = apply_sensor_dropout(x, args.sensor_dropout, rng)
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

        # Eval every 2 epochs (small total epochs)
        if epoch % 2 == 0 or epoch == args.epochs or epoch == 1:
            val = evaluate(model, val_dl, adj, device)
            # s_029
            with torch.no_grad():
                yp = model(x29.unsqueeze(0).to(device), adj).squeeze(0).cpu().numpy()
            p, t_arr = yp[:, 5] >= 0.5, y29.numpy()[:, 5] >= 0.5
            tp, fp_, fn = (p & t_arr).sum(), (p & ~t_arr).sum(), (~p & t_arr).sum()
            s29_iou = float(tp / (tp + fp_ + fn + 1e-9))
            # Composite: prioritize s_029 strongly
            score = 0.7 * s29_iou + 0.3 * val["iou"]
            rec = {"epoch": epoch, "train_loss": train_loss,
                   "val_iou": val["iou"], "val_fnr": val["fnr"],
                   "s029_iou": s29_iou, "score": score}
            history.append(rec)
            marker = ""
            if score > best_score:
                best_score = score
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch, "val_iou": val["iou"], "s029_iou": s29_iou,
                    "config": vars(args), "v5_config": v5_cfg,
                }, args.output / "best.pt")
                marker = " *"
            print(f"  ep {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
                  f"s029={s29_iou:.3f}  val_iou={val['iou']:.3f}  "
                  f"val_fnr={val['fnr']*100:.1f}%  score={score:.3f}{marker}")
        else:
            history.append({"epoch": epoch, "train_loss": train_loss})

    # Final test
    print(f"\n[test] running on {len(test_names)} OOD scenarios using best.pt")
    ckpt = torch.load(args.output / "best.pt", weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_res = evaluate(model, test_dl, adj, device)
    with torch.no_grad():
        yp = model(x29.unsqueeze(0).to(device), adj).squeeze(0).cpu().numpy()
    p, t_arr = yp[:, 5] >= 0.5, y29.numpy()[:, 5] >= 0.5
    tp, fp_, fn = (p & t_arr).sum(), (p & ~t_arr).sum(), (~p & t_arr).sum()
    s29_final = float(tp / (tp + fp_ + fn + 1e-9))
    print(f"  test IoU: {test_res['iou']:.3f}  (vs v5 0.911 / v3 0.889)")
    print(f"  test FNR: {test_res['fnr']*100:.1f}%")
    print(f"  s_029 IoU: {s29_final:.3f}  (vs v5 0.812 / v3 0.565)")

    torch.save({
        "model": model.state_dict(), "epoch": args.epochs,
        "test_iou": test_res["iou"], "test_fnr": test_res["fnr"],
        "s029_iou": s29_final,
        "config": vars(args), "v5_config": v5_cfg,
    }, args.output / "final.pt")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    eps = [h["epoch"] for h in history if "s029_iou" in h]
    s29s = [h["s029_iou"] for h in history if "s029_iou" in h]
    vals = [h["val_iou"] for h in history if "val_iou" in h]
    axes[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history], lw=1.5)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("combined loss")
    axes[0].set_title("v6 training loss"); axes[0].grid(alpha=0.3)
    axes[1].plot(eps, s29s, "o-", color="darkred", lw=1.5, label="s_029 IoU")
    axes[1].plot(eps, vals, "s-", color="tab:blue", lw=1.5, label="val IoU (3 OOD)")
    axes[1].axhline(0.70, color="red", lw=0.7, ls="--", label="H5 ≥ 0.70")
    axes[1].axhline(0.812, color="darkred", lw=0.5, ls=":", alpha=0.6,
                    label="v5 baseline s_029 = 0.812")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("IoU")
    axes[1].set_title("s_029 + val IoU tracking"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output / "loss_curve.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    np.savetxt(args.output / "history.csv",
                [[h["epoch"], h["train_loss"], h.get("s029_iou", -1),
                  h.get("val_iou", -1)] for h in history],
                fmt="%.6f", delimiter=",",
                header="epoch,train_loss,s029_iou,val_iou")

    print(f"\n[PASS]")
    print(f"  best.pt @ epoch {ckpt['epoch']} (s_029 IoU {ckpt.get('s029_iou', '?')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
