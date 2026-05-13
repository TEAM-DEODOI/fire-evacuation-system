"""6-row 60s autoregress 비교 figure — FDS + 4 full-input + 2 sparse-retrain.

5-row 의 확장: Sparse FNO (6-channel + re-sparsify) row 추가.

산출물: figures/current/05_future_prediction/<scenario>_grid_6model_t0_<NNN>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import (
    build_input_tensor, normalize_scenario,
)
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS, T_END_SECONDS
from evaluate_t_locations import load_model, load_mask
from evaluate_sparse_fno import (
    load_sparse_fno, build_sparse_6ch_input, autoregress_sparse_fno,
)
from train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator
from visualize_60s_5model import (
    autoregress_full_input, autoregress_sparse_input, sparsify_initial_input,
)

Z_IDX = 3
LOOKAHEAD_STEPS = 6


def plot_6row(truth_danger, preds_by_model, scenario_name, t0_seconds, out_path):
    n_cols = truth_danger.shape[0]
    rows = ["FDS truth"] + list(preds_by_model.keys())
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))

    for col in range(n_cols):
        t_label = f"t = {t0_seconds + (col + 1) * DT_SLCF:.0f} s"
        im = axes[0, col].imshow(
            truth_danger[col, :, :, Z_IDX].T,
            origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
            extent=[0, 30, 0, 20], aspect="equal",
        )
        axes[0, col].set_title(t_label, fontsize=10)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        for r, name in enumerate(preds_by_model.keys(), start=1):
            axes[r, col].imshow(
                preds_by_model[name][col, :, :, Z_IDX].T,
                origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
                extent=[0, 30, 0, 20], aspect="equal",
            )
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])

    for r, label in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=12,
                                fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="danger ∈ [0, 1]")

    fig.suptitle(
        f"60 s future risk forecast — 6-model comparison  |  {scenario_name}\n"
        f"t0 = {t0_seconds:.0f} s, z = 1.75 m, autoregressive rollout (6 × 10 s)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 0.93])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path, default=Path("configs/building.yaml"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/current/05_future_prediction"))
    parser.add_argument("--scenarios", type=str, nargs="+",
                        default=["sim_1500kw_2m2_T05", "sim_500kw_1m2_T01",
                                  "sim_1000kw_1m2_T03"])
    parser.add_argument("--t0", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[setup] loading models...")

    # Full-input 모델 4개 + Sparse-retrain 모델 2개 (ConvLSTM, FNO)
    full_models = {
        "ConvLSTM":  load_model(Path("checkpoints/conv_lstm/best.pt"),  device, "conv_lstm"),
        "FNO no-PI": load_model(Path("checkpoints/fno_no_pi/best.pt"),  device, "fno"),
        "FNO PI":    load_model(Path("checkpoints/fno_pi/best.pt"),     device, "fno"),
    }
    sparse_conv = load_model(Path("checkpoints/conv_lstm_sparse_v3/best.pt"),
                              device, "conv_lstm")
    sparse_fno = load_sparse_fno(Path("checkpoints/fno_sparse_v3/best.pt"), device)

    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)

    for sname in args.scenarios:
        sdir = args.raw_root / sname
        if not sdir.is_dir():
            continue
        print(f"[scen] {sname}")
        slices = extract_slices(sdir)
        norm = normalize_scenario(slices)
        inp = build_input_tensor(norm, mask, times=slices["times"])     # (31, 5, ...)
        inp_6ch = build_sparse_6ch_input(slices, mask, sparse_ind)      # (31, 6, ...)
        truth_danger = compute_total_danger(
            slices["temperature"], slices["visibility"], slices["co"],
        ).astype(np.float32)
        t0_idx = int(args.t0 // DT_SLCF)
        if t0_idx + LOOKAHEAD_STEPS >= N_TIMESTEPS:
            continue
        truth_window = truth_danger[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

        preds: Dict[str, np.ndarray] = {}
        # Full-input 모델 3개
        for name, model in full_models.items():
            preds_norm = autoregress_full_input(model, inp[t0_idx], args.t0, device)
            times_arr = np.array([args.t0 + (s + 1) * DT_SLCF for s in range(LOOKAHEAD_STEPS)])
            preds[name] = prediction_to_danger(preds_norm, times_arr)
        # Sparse ConvLSTM (5-ch + re-sparsify)
        init_sparse = sparsify_initial_input(inp[t0_idx], sparse_ind)
        preds_norm = autoregress_sparse_input(
            sparse_conv, init_sparse, sparse_ind, args.t0, device,
        )
        times_arr = np.array([args.t0 + (s + 1) * DT_SLCF for s in range(LOOKAHEAD_STEPS)])
        preds["Sparse-ConvLSTM"] = prediction_to_danger(preds_norm, times_arr)
        # Sparse FNO (6-ch + re-sparsify)
        preds_norm = autoregress_sparse_fno(
            sparse_fno, inp_6ch[t0_idx], sparse_ind, args.t0, device,
            resparsify=True,
        )
        preds["Sparse-FNO (6-ch)"] = prediction_to_danger(preds_norm, times_arr)

        out_path = args.out_dir / f"{sname}_grid_6model_t0_{int(args.t0):03d}.png"
        plot_6row(truth_window, preds, sname, args.t0, out_path)
        print(f"  -> {out_path}")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
