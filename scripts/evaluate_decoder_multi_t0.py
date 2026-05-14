"""Multi-t0 robustness check for the learned ensemble decoder.

The current `checkpoints/ensemble_decoder/best.pt` was trained on t0=120s
samples only. H6 path-planning will call the risk map at many t values
(every 30s during evacuation), so we need to verify the decoder behaves
sensibly across t0 ∈ {60, 90, 120, 150, 180, 210}s.

For each t0 value:
- Re-forward the 3 surrogate models on the 13 OOD scenarios at this t0.
- Run the decoder; measure IoU/FNR @ +60s (last step of the 6-step rollout).
- Aggregate across scenarios and produce a t0-curve.

If IoU drops sharply at non-120s t0 values, we will retrain the decoder
on multi-t0 data.

Output:
  figures/current/12_decoder_multi_t0/iou_fnr_vs_t0.png
  figures/current/12_decoder_multi_t0/per_scenario_t0.png
  results/exp_decoder_multi_t0/by_t0.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import build_input_tensor, normalize_scenario
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import DT_SLCF, N_TIMESTEPS
from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency
from src.tier1.ensemble_decoder import (
    PerCellEnsembleDecoder, decoder_forward_grid,
)

from evaluate_t_locations import load_mask, load_model
from evaluate_sparse_fno import (
    load_sparse_fno, build_sparse_6ch_input, autoregress_sparse_fno,
)
from evaluate_ensemble import (
    gnn_node_pred_to_cell_danger, precompute_node_to_cell_weights,
    tier1_forward,
)
from train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator
from visualize_60s_5model import (
    autoregress_sparse_input, sparsify_initial_input,
)

LOOKAHEAD_STEPS = 6
T_IN = 6


def iou_fnr(pred, truth, mask, threshold=0.5):
    fluid = (mask > 0.5)
    fm = np.broadcast_to(fluid, pred.shape).reshape(-1)
    p = (pred >= threshold).reshape(-1)[fm]
    t = (truth >= threshold).reshape(-1)[fm]
    tp = float(np.sum(p & t)); fp = float(np.sum(p & ~t)); fn = float(np.sum(~p & t))
    return {
        "iou": tp / (tp + fp + fn + 1e-9),
        "fnr": fn / (fn + tp + 1e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--seq-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path,
                        default=Path("configs/building.yaml"))
    parser.add_argument("--decoder-ckpt", type=Path,
                        default=Path("checkpoints/ensemble_decoder/best.pt"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/current/12_decoder_multi_t0"))
    parser.add_argument("--out-results", type=Path,
                        default=Path("results/exp_decoder_multi_t0"))
    parser.add_argument("--t0-list", type=float, nargs="+",
                        default=[60.0, 90.0, 120.0, 150.0, 180.0, 210.0])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--knn-sigma", type=float, default=5.0)
    args = parser.parse_args()

    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_results.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load models ──────────────────────────────────────────────────────
    print("[setup] loading 3 surrogates + decoder")
    sparse_conv = load_model(Path("checkpoints/conv_lstm_sparse_v3/best.pt"),
                              device, "conv_lstm")
    sparse_fno = load_sparse_fno(Path("checkpoints/fno_sparse_v3/best.pt"), device)
    gnn_ckpt = torch.load(Path("checkpoints/tier1_gnn_v3/best.pt"),
                           weights_only=False, map_location=device)
    cfg = gnn_ckpt.get("config", {})
    gnn_model = SimpleFireGNN(
        in_feat=5, hidden=cfg.get("hidden", 32),
        n_graph_layers=cfg.get("n_graph_layers", 2),
        T_out=cfg.get("T_out", 6),
    )
    gnn_model.load_state_dict(gnn_ckpt["model"])
    gnn_model.to(device).eval()
    adj = build_knn_adjacency(k=cfg.get("knn_k", 4))

    dec_ckpt = torch.load(args.decoder_ckpt, weights_only=False, map_location=device)
    dec_cfg = dec_ckpt["config"]
    decoder = PerCellEnsembleDecoder(
        hidden=dec_cfg["hidden"], n_layers=dec_cfg["n_layers"],
        dropout=dec_cfg.get("dropout", 0.0),
    )
    decoder.load_state_dict(dec_ckpt["model"])
    decoder.to(device).eval()
    print(f"        decoder ckpt: trained at t0=120s "
          f"(best epoch {dec_ckpt.get('best_epoch')}, IoU {dec_ckpt.get('best_ood_iou'):.3f})")

    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)

    print("[setup] precomputing geodesic k-NN cell-to-node weights")
    node_positions = [d.position for d in ALL_DETECTORS]
    knn_idx, knn_w = precompute_node_to_cell_weights(
        node_positions, k=args.knn_k, sigma=args.knn_sigma,
        mask=mask, use_geodesic=True,
    )

    # ── 13 OOD scenarios ────────────────────────────────────────────────
    ood_scens = sorted([d.name for d in args.raw_root.glob("sim_*") if d.is_dir()])
    print(f"[setup] {len(ood_scens)} OOD scenarios, t0 ∈ {args.t0_list}")

    # ── Loop ────────────────────────────────────────────────────────────
    rows: List[Dict] = []
    for sname in ood_scens:
        sdir = args.raw_root / sname
        slices = extract_slices(sdir)
        norm = normalize_scenario(slices)
        inp = build_input_tensor(norm, mask, times=slices["times"])
        inp_6ch = build_sparse_6ch_input(slices, mask, sparse_ind)
        truth_danger = compute_total_danger(
            slices["temperature"], slices["visibility"], slices["co"],
        ).astype(np.float32)

        for t0 in args.t0_list:
            t0_idx = int(t0 // DT_SLCF)
            if t0_idx < T_IN:
                continue           # not enough history
            if t0_idx + LOOKAHEAD_STEPS >= N_TIMESTEPS:
                continue
            t_start = t0_idx - T_IN
            times_arr = np.array([t0 + (s + 1) * DT_SLCF
                                    for s in range(LOOKAHEAD_STEPS)])
            truth_window = truth_danger[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

            try:
                # Sparse ConvLSTM
                init_sparse = sparsify_initial_input(inp[t0_idx], sparse_ind)
                preds_norm = autoregress_sparse_input(
                    sparse_conv, init_sparse, sparse_ind, t0, device,
                )
                sparse_conv_danger = prediction_to_danger(preds_norm, times_arr)

                # Sparse FNO
                preds_norm = autoregress_sparse_fno(
                    sparse_fno, inp_6ch[t0_idx], sparse_ind, t0, device,
                    resparsify=True,
                )
                sparse_fno_danger = prediction_to_danger(preds_norm, times_arr)

                # GNN cell-projected
                t1_node = tier1_forward(sname, args.seq_dir, gnn_model, adj,
                                          t_start, device)
                gnn_cell = gnn_node_pred_to_cell_danger(t1_node, knn_idx, knn_w)

                # Decoder
                decoded = decoder_forward_grid(
                    decoder, gnn_cell, sparse_conv_danger, sparse_fno_danger,
                    mask, device=device,
                )

                m6 = iou_fnr(decoded[5:6], truth_window[5:6], mask)
                rows.append({
                    "scenario": sname, "t0": t0,
                    "iou": m6["iou"], "fnr": m6["fnr"],
                })
                print(f"  {sname:30s}  t0={t0:5.0f}s  IoU={m6['iou']:.3f}  FNR={m6['fnr']*100:5.1f}%")
            except Exception as e:
                print(f"  [FAIL] {sname} t0={t0}: {e}")

    # ── Aggregate per t0 ─────────────────────────────────────────────────
    by_t0: Dict[float, Dict] = {}
    for t0 in args.t0_list:
        ts = [r for r in rows if r["t0"] == t0]
        if not ts:
            continue
        ious = [r["iou"] for r in ts]; fnrs = [r["fnr"] for r in ts]
        by_t0[t0] = {
            "n": len(ts),
            "mean_iou": float(np.mean(ious)),
            "mean_fnr": float(np.mean(fnrs)),
            "n_h5": sum(1 for v in ious if v >= 0.7),
            "n_h4": sum(1 for v in fnrs if v < 0.10),
        }

    print(f"\n[agg]")
    print(f"  {'t0':>6}  {'N':>3}  {'mean IoU':>9}  {'mean FNR':>9}  {'H5':>4}  {'H4':>4}")
    for t0, s in sorted(by_t0.items()):
        print(f"  {t0:6.0f}  {s['n']:>3d}  {s['mean_iou']:>9.3f}  "
              f"{s['mean_fnr']*100:>8.1f}%  {s['n_h5']:>3d}/{s['n']:<2d}  {s['n_h4']:>3d}/{s['n']:<2d}")

    # ── CSV ─────────────────────────────────────────────────────────────
    csv_path = args.out_results / "by_t0.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "t0", "iou_step6", "fnr_step6"])
        for r in rows:
            w.writerow([r["scenario"], r["t0"], r["iou"], r["fnr"]])
    print(f"\n[csv]  {csv_path}")

    # ── Plot 1: t0 vs mean IoU + mean FNR ───────────────────────────────
    t0s = sorted(by_t0.keys())
    if t0s:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot(t0s, [by_t0[t]["mean_iou"] for t in t0s], "o-",
                    lw=2, markersize=10, color="tab:blue")
        ax[0].axhline(0.733, color="tab:gray", ls="--", lw=0.8,
                       label="t0=120s baseline 0.733")
        ax[0].axhline(0.70, color="green", ls=":", lw=0.8, label="H5 ≥ 0.70")
        ax[0].set_xlabel("t0 (s)"); ax[0].set_ylabel("Mean IoU @ +60s")
        ax[0].set_title("Decoder mean IoU vs t0 (13 OOD)")
        ax[0].grid(alpha=0.3); ax[0].legend()
        for t in t0s:
            ax[0].text(t, by_t0[t]["mean_iou"] + 0.005,
                        f"{by_t0[t]['mean_iou']:.3f}", ha="center", fontsize=9)

        ax[1].plot(t0s, [by_t0[t]["mean_fnr"] * 100 for t in t0s], "s-",
                    lw=2, markersize=10, color="tab:red")
        ax[1].axhline(11.5, color="tab:gray", ls="--", lw=0.8,
                       label="t0=120s baseline 11.5%")
        ax[1].axhline(10.0, color="red", ls=":", lw=0.8, label="H4 < 10%")
        ax[1].set_xlabel("t0 (s)"); ax[1].set_ylabel("Mean FNR % @ +60s")
        ax[1].set_title("Decoder mean FNR vs t0")
        ax[1].grid(alpha=0.3); ax[1].legend()
        for t in t0s:
            ax[1].text(t, by_t0[t]["mean_fnr"] * 100 + 0.5,
                        f"{by_t0[t]['mean_fnr']*100:.1f}%", ha="center", fontsize=9)

        fig.suptitle(
            "Decoder multi-t0 robustness — trained on t0=120s, evaluated across t0",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out1 = args.out_figures / "iou_fnr_vs_t0.png"
        fig.savefig(out1, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out1}")

    # ── Plot 2: per-scenario curves ─────────────────────────────────────
    scens = sorted(set(r["scenario"] for r in rows))
    if t0s and scens:
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
        cmap = plt.cm.tab20(np.linspace(0, 1, len(scens)))
        for color, scen in zip(cmap, scens):
            xs = [r["t0"] for r in rows if r["scenario"] == scen]
            ys = [r["iou"] for r in rows if r["scenario"] == scen]
            pairs = sorted(zip(xs, ys))
            xs, ys = zip(*pairs)
            ax.plot(xs, ys, "o-", color=color, lw=1.2, alpha=0.8,
                     label=scen.replace("sim_", ""))
        ax.axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
        ax.set_xlabel("t0 (s)"); ax.set_ylabel("Decoder IoU @ +60s")
        ax.set_title("Per-scenario decoder IoU vs t0 (decoder trained at t0=120s)")
        ax.legend(loc="lower right", fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        out2 = args.out_figures / "per_scenario_t0.png"
        fig.tight_layout()
        fig.savefig(out2, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out2}")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
