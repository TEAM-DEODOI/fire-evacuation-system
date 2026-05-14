"""5-row 60s autoregress 비교 figure — FDS + 4 모델.

기존 `visualize_60s_prediction.py` (4-row: truth + ConvLSTM + FNO no-PI + FNO PI)
에 Sparse-retrain ConvLSTM (L4e) row 추가. 같은 시나리오, 같은 t₀.

산출물: figures/current/05_future_prediction/<scenario>_grid_5model_t0_<NNN>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import (
    build_input_tensor, build_target_tensor, normalize_scenario,
)
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS, T_END_SECONDS
from evaluate_t_locations import load_model, load_mask
from train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator

Z_IDX = 3   # z ≈ 1.75 m
LOOKAHEAD_STEPS = 6


def autoregress_full_input(model, initial_input, t0_seconds, device,
                            n_steps=LOOKAHEAD_STEPS):
    """Standard autoregress (모든 cell 자유롭게 chaining)."""
    state = initial_input.copy()
    mask_ch = state[3].copy()
    preds = np.zeros((n_steps, 3, *GRID_SHAPE), dtype=np.float32)
    with torch.no_grad():
        for step in range(n_steps):
            x = torch.from_numpy(state).unsqueeze(0).to(device)
            y_pred = np.clip(model(x).cpu().numpy()[0], 0.0, 1.0)
            preds[step] = y_pred
            t_next = t0_seconds + (step + 1) * DT_SLCF
            state = np.zeros_like(state)
            state[:3] = y_pred
            state[3] = mask_ch
            state[4] = np.full_like(mask_ch, t_next / T_END_SECONDS)
    return preds


def autoregress_sparse_input(model, initial_sparse_input, sparse_ind,
                              t0_seconds, device, n_steps=LOOKAHEAD_STEPS):
    """Sparse-input autoregress — sensor 위치 외 T/V/CO 를 0 으로 강제."""
    state = initial_sparse_input.copy()
    mask_ch = state[3].copy()
    not_sensor = ~sparse_ind
    preds = np.zeros((n_steps, 3, *GRID_SHAPE), dtype=np.float32)
    with torch.no_grad():
        for step in range(n_steps):
            x = torch.from_numpy(state).unsqueeze(0).to(device)
            y_pred = np.clip(model(x).cpu().numpy()[0], 0.0, 1.0)
            preds[step] = y_pred
            t_next = t0_seconds + (step + 1) * DT_SLCF
            state = np.zeros_like(state)
            state[:3] = y_pred
            # Re-sparsify — training 분포와 일치 (autoregress drift 방지)
            for c in range(3):
                state[c][not_sensor] = 0.0
            state[3] = mask_ch
            state[4] = np.full_like(mask_ch, t_next / T_END_SECONDS)
    return preds


def sparsify_initial_input(inp_t0, sparse_ind):
    """t₀ 시점 input 의 T/V/CO 채널을 sensor 위치만 남김."""
    out = inp_t0.copy()
    not_sensor = ~sparse_ind
    for c in range(3):
        out[c][not_sensor] = 0.0
    return out


def plot_5_row_grid(
    truth_danger: np.ndarray,            # (n_steps, X, Y, Z)
    preds_by_model: Dict[str, np.ndarray], # {name: (n_steps, X, Y, Z)}
    scenario_name: str,
    t0_seconds: float,
    out_path: Path,
):
    n_cols = truth_danger.shape[0]
    rows = ["FDS truth"] + list(preds_by_model.keys())
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.6 * n_rows))

    for col in range(n_cols):
        t_label = f"t = {t0_seconds + (col + 1) * DT_SLCF:.0f} s"
        # truth row
        im = axes[0, col].imshow(
            truth_danger[col, :, :, Z_IDX].T,
            origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
            extent=[0, 30, 0, 20], aspect="equal",
        )
        axes[0, col].set_title(t_label, fontsize=10)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        # model rows
        for r, name in enumerate(preds_by_model.keys(), start=1):
            axes[r, col].imshow(
                preds_by_model[name][col, :, :, Z_IDX].T,
                origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
                extent=[0, 30, 0, 20], aspect="equal",
            )
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])

    for r, label in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=13, rotation=90, labelpad=12,
                                fontweight="bold")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="danger ∈ [0, 1]")

    fig.suptitle(
        f"60 s future risk forecast — 5-model comparison  |  {scenario_name}\n"
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
    print(f"[setup] device={device}")
    print(f"[setup] loading 4 models...")

    # 4 모델 (full-input + sparse-retrain)
    models = {
        "ConvLSTM":             (load_model(Path("checkpoints/conv_lstm/best.pt"),  device, "conv_lstm"), "full"),
        "FNO no-PI":            (load_model(Path("checkpoints/fno_no_pi/best.pt"),  device, "fno"),       "full"),
        "FNO PI":               (load_model(Path("checkpoints/fno_pi/best.pt"),     device, "fno"),       "full"),
        "Sparse-ConvLSTM (v3)": (load_model(Path("checkpoints/conv_lstm_sparse_v3/best.pt"), device, "conv_lstm"), "sparse"),
    }
    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)
    print(f"[setup] {int(np.sum(sparse_ind))} sparse cells / {np.prod(GRID_SHAPE)}")

    for sname in args.scenarios:
        sdir = args.raw_root / sname
        if not sdir.is_dir():
            print(f"[skip] {sname}: not a directory")
            continue
        print(f"[scen] {sname}")

        slices = extract_slices(sdir)
        norm = normalize_scenario(slices)
        inp = build_input_tensor(norm, mask, times=slices["times"])  # (31, 5, ...)

        truth_danger = compute_total_danger(
            slices["temperature"], slices["visibility"], slices["co"],
        ).astype(np.float32)

        t0_idx = int(args.t0 // DT_SLCF)
        if t0_idx + LOOKAHEAD_STEPS >= N_TIMESTEPS:
            print(f"  skip: t0 too late")
            continue
        truth_window = truth_danger[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

        # Model rollouts
        preds_by_model: Dict[str, np.ndarray] = {}
        for name, (model, kind) in models.items():
            if kind == "sparse":
                # Sparsify t0 input
                init_sparse = sparsify_initial_input(inp[t0_idx], sparse_ind)
                preds_norm = autoregress_sparse_input(
                    model, init_sparse, sparse_ind, args.t0, device,
                )
            else:
                preds_norm = autoregress_full_input(
                    model, inp[t0_idx], args.t0, device,
                )
            times_arr = np.array([args.t0 + (s + 1) * DT_SLCF
                                    for s in range(LOOKAHEAD_STEPS)])
            preds_by_model[name] = prediction_to_danger(preds_norm, times_arr)

        out_path = args.out_dir / f"{sname}_grid_5model_t0_{int(args.t0):03d}.png"
        plot_5_row_grid(truth_window, preds_by_model, sname, args.t0, out_path)
        print(f"  -> {out_path}")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
