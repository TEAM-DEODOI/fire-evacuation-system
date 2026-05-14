"""H1 verification — inference time of every surrogate in the stack.

Hypothesis H1: surrogate inference is ≥ 1000× faster than FDS.

Baseline (from CLAUDE.md): FDS ~23 minutes per scenario (300 s sim, 6 z
layers).  That's the wall-clock cost of producing the (31, 60, 40, 6)
ground-truth danger grid.

We measure CPU latency for:
  - SimpleFireGNN (Tier 1, 12 K params)            single forward → (39, 6)
  - Sparse-ConvLSTM v3 (Tier 2, ~349 K params)      6-step rollout
  - Sparse-FNO v3 (Tier 2, ~1.79 M params)          6-step rollout
  - PerCellEnsembleDecoder (L4h, 1.4 K params)      full-grid forward
  - **Full L4h pipeline** (GNN + 2 sparse + decoder, end-to-end)

Numbers go into:
  - figures/current/13_h1_speed/inference_latency.png
  - results/exp_h1_speed/latency.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.shared.constants import DT_SLCF
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
FDS_SECONDS = 23 * 60   # CLAUDE.md baseline


def time_callable(fn, *, warmup: int = 5, n: int = 30) -> dict:
    """Run fn n times, return mean/std/min in ms after warmup."""
    for _ in range(warmup):
        fn()
    times_ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms":  float(arr.std()),
        "min_ms":  float(arr.min()),
        "max_ms":  float(arr.max()),
        "n_runs":  n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path,
                        default=Path("configs/building.yaml"))
    parser.add_argument("--seq-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/current/13_h1_speed"))
    parser.add_argument("--out-results", type=Path,
                        default=Path("results/exp_h1_speed"))
    parser.add_argument("--scenario", type=str, default="sim_1500kw_2m2_T05")
    parser.add_argument("--t0", type=float, default=120.0)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_results.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[setup] loading models  (device={device})")
    # GNN
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

    sparse_conv = load_model(Path("checkpoints/conv_lstm_sparse_v3/best.pt"),
                              device, "conv_lstm")
    sparse_fno = load_sparse_fno(Path("checkpoints/fno_sparse_v3/best.pt"),
                                  device)

    decoder_ckpt = torch.load(Path("checkpoints/ensemble_decoder/best.pt"),
                                weights_only=False, map_location=device)
    decoder = PerCellEnsembleDecoder(
        hidden=decoder_ckpt["config"]["hidden"],
        n_layers=decoder_ckpt["config"]["n_layers"],
        dropout=decoder_ckpt["config"].get("dropout", 0.0),
    )
    decoder.load_state_dict(decoder_ckpt["model"])
    decoder.to(device).eval()

    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)
    node_positions = [d.position for d in ALL_DETECTORS]
    knn_idx, knn_w = precompute_node_to_cell_weights(
        node_positions, k=3, sigma=5.0, mask=mask, use_geodesic=True,
    )

    # ── Build cached inputs for the chosen scenario ─────────────────────
    print(f"[setup] preparing inputs for {args.scenario} at t0={args.t0}")
    from src.data_pipeline.fds_extractor import extract_slices
    from src.data_pipeline.normalize import (
        build_input_tensor, normalize_scenario,
    )
    sdir = Path("data/raw") / args.scenario
    slices = extract_slices(sdir)
    norm = normalize_scenario(slices)
    inp = build_input_tensor(norm, mask, times=slices["times"])
    inp_6ch = build_sparse_6ch_input(slices, mask, sparse_ind)
    t0_idx = int(args.t0 // DT_SLCF)
    init_sparse = sparsify_initial_input(inp[t0_idx], sparse_ind)
    init_fno = inp_6ch[t0_idx]

    # GNN input (binary detector history) — already in seq_dir
    from src.tier1.tier1_dataset import Tier1FireDataset
    ds = Tier1FireDataset(args.seq_dir, [args.scenario], T_in=6, T_out=6)
    target_pair = None
    for i, (s_idx, t_) in enumerate(ds.pairs):
        if t_ == t0_idx - 6:
            target_pair = i; break
    gnn_x, _ = ds[target_pair]
    gnn_x = gnn_x.unsqueeze(0)

    # ── Timed callables ────────────────────────────────────────────────
    results = []
    # 1) GNN
    print("\n[time] SimpleFireGNN (12K params, 6-step)")
    def f_gnn():
        with torch.no_grad():
            _ = gnn_model(gnn_x, adj)
    t = time_callable(f_gnn, warmup=args.n_warmup, n=args.n_runs)
    results.append(("Tier 1 GNN (single forward)", 12_006, t))

    # 2) Sparse-ConvLSTM 6-step rollout
    print("[time] Sparse-ConvLSTM v3 (6-step rollout)")
    def f_conv():
        _ = autoregress_sparse_input(
            sparse_conv, init_sparse, sparse_ind, args.t0, device,
        )
    t = time_callable(f_conv, warmup=2, n=10)
    results.append(("Sparse-ConvLSTM v3 (6-step rollout)", 349_000, t))

    # 3) Sparse-FNO 6-step rollout
    print("[time] Sparse-FNO v3 (6-step rollout)")
    def f_fno():
        _ = autoregress_sparse_fno(
            sparse_fno, init_fno, sparse_ind, args.t0, device, resparsify=True,
        )
    t = time_callable(f_fno, warmup=2, n=10)
    results.append(("Sparse-FNO v3 (6-step rollout)", 1_790_000, t))

    # 4) Decoder full-grid forward
    print("[time] Learned Decoder (full-grid forward)")
    gnn_cell_pre = np.random.rand(6, 60, 40, 6).astype(np.float32)
    sparse_conv_pre = np.random.rand(6, 60, 40, 6).astype(np.float32)
    sparse_fno_pre = np.random.rand(6, 60, 40, 6).astype(np.float32)
    def f_dec():
        _ = decoder_forward_grid(
            decoder, gnn_cell_pre, sparse_conv_pre, sparse_fno_pre, mask,
            device=device,
        )
    t = time_callable(f_dec, warmup=args.n_warmup, n=args.n_runs)
    results.append(("Learned Decoder (full grid)", 1_377, t))

    # 5) Full L4h pipeline (everything except FDS extract — that is cached)
    print("[time] Full L4h pipeline (GNN + 2 sparse + decoder)")
    def f_pipe():
        with torch.no_grad():
            t1 = gnn_model(gnn_x, adj).squeeze(0).numpy().T   # (T_out, N)
        gnn_cell = gnn_node_pred_to_cell_danger(t1, knn_idx, knn_w)
        pc = autoregress_sparse_input(
            sparse_conv, init_sparse, sparse_ind, args.t0, device,
        )
        from src.risk_map.converter import prediction_to_danger
        times_arr = np.array([args.t0 + (s + 1) * DT_SLCF
                                for s in range(LOOKAHEAD_STEPS)])
        sc_d = prediction_to_danger(pc, times_arr)
        pf = autoregress_sparse_fno(
            sparse_fno, init_fno, sparse_ind, args.t0, device, resparsify=True,
        )
        sf_d = prediction_to_danger(pf, times_arr)
        _ = decoder_forward_grid(
            decoder, gnn_cell, sc_d, sf_d, mask, device=device,
        )
    t = time_callable(f_pipe, warmup=2, n=10)
    pipeline_total = 12_006 + 349_000 + 1_790_000 + 1_377
    results.append(("Full L4h pipeline (end-to-end)", pipeline_total, t))

    # ── Report ─────────────────────────────────────────────────────────
    print(f"\n[results]  (FDS baseline: {FDS_SECONDS} s = {FDS_SECONDS*1000} ms)")
    print(f"  {'Module':45s} {'#params':>10s} {'mean (ms)':>11s} "
          f"{'std':>7s} {'min':>7s}    speedup")
    for name, params, t in results:
        speedup = (FDS_SECONDS * 1000) / max(t["mean_ms"], 1e-6)
        print(f"  {name:45s} {params:>10,d} {t['mean_ms']:>10.2f} "
              f"{t['std_ms']:>7.2f} {t['min_ms']:>7.2f}   {speedup:>8.0f}x")

    # ── CSV ────────────────────────────────────────────────────────────
    csv_path = args.out_results / "latency.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "n_params", "mean_ms", "std_ms",
                     "min_ms", "max_ms", "n_runs", "speedup_vs_fds"])
        for name, params, t in results:
            speedup = (FDS_SECONDS * 1000) / max(t["mean_ms"], 1e-6)
            w.writerow([name, params, t["mean_ms"], t["std_ms"],
                         t["min_ms"], t["max_ms"], t["n_runs"], speedup])
    print(f"\n[csv] {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────
    names = [r[0].replace(" (", "\n(") for r in results]
    means = [r[2]["mean_ms"] for r in results]
    stds  = [r[2]["std_ms"]  for r in results]
    speedups = [(FDS_SECONDS * 1000) / m for m in means]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "darkred"]
    axes[0].bar(range(len(names)), means, yerr=stds, color=colors,
                 edgecolor="black")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=9, rotation=0)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_yscale("log")
    axes[0].set_title("Inference latency per module (log scale)")
    axes[0].grid(alpha=0.3, axis="y", which="both")
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m, f"{m:.1f}±{s:.1f}", ha="center",
                      va="bottom", fontsize=8)

    axes[1].bar(range(len(names)), speedups, color=colors, edgecolor="black")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=9, rotation=0)
    axes[1].set_ylabel("Speedup vs FDS (×)")
    axes[1].set_yscale("log")
    axes[1].axhline(1000, color="red", lw=0.8, ls="--", label="H1 ≥ 1000×")
    axes[1].set_title("Speedup over FDS (23 min/scenario)")
    axes[1].grid(alpha=0.3, axis="y", which="both")
    axes[1].legend()
    for i, s in enumerate(speedups):
        axes[1].text(i, s, f"{s:.0f}x", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"H1 hypothesis — surrogate inference latency on CPU  "
        f"({args.scenario}, t0={args.t0:.0f}s)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = args.out_figures / "inference_latency.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
