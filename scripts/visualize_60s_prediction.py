"""60초 미래 위험 예측 지도 시각화 (헤드라인 데모).

진짜 "future prediction" — 관측 시점 t₀ 에서 autoregress 로 6 step 진행해서
t₀ + 60s 까지의 risk map 을 모델 출력만으로 생성. teacher-forcing 없음.

기존 ``evaluate_t_locations.py`` 의 risk_animation.gif 는 매 시점에서 1-step
prediction 의 모자이크였음 (모델이 매 step 마다 FDS truth 를 input 으로 받음).
본 스크립트는 **순수 model rollout** 만으로 60s 미래 예측을 생성한다.

산출물 (`figures/future_prediction/`)
------------------------------------
* ``<scenario>_grid.png`` — 4 row × 6 col 그리드:
    Row 1 : FDS truth at t = t₀+10, +20, ..., +60s
    Row 2 : ConvLSTM autoregress 1..6 step
    Row 3 : FNO no-PI autoregress 1..6 step
    Row 4 : FNO PI autoregress 1..6 step
* ``<scenario>_animation.gif`` — t₀ 가 sliding (0, 30, 60, 120, 180s) 하면서
    각 시점에서의 60s lookahead 영상.
* ``summary_all_locations.png`` — 5 위치 × t₀=0 의 60초 미래만 모음.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# scripts/ 를 import path 에 추가 (load_model / load_mask 재사용)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import (
    build_input_tensor, build_target_tensor, normalize_scenario,
)
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS, T_END_SECONDS
from evaluate_t_locations import load_model, load_mask

Z_IDX = 3  # z ≈ 1.75 m (호흡고도)
LOOKAHEAD_STEPS = 6   # 60 s


# ─── Autoregress 핵심 ──────────────────────────────────────────────────────
def autoregress_60s(
    model: torch.nn.Module,
    initial_input: np.ndarray,   # (5, X, Y, Z) — 5-channel input at t₀
    t0_seconds: float,
    device: torch.device,
    n_steps: int = LOOKAHEAD_STEPS,
) -> np.ndarray:
    """t₀ 에서 시작해 n_steps autoregress. (n_steps, 3, X, Y, Z) 반환."""
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


# ─── Grid figure: 모델 × 시점 ──────────────────────────────────────────────
def plot_60s_grid(
    truth_danger: np.ndarray,    # (n_steps, X, Y, Z)
    model_preds_danger: Dict[str, np.ndarray],  # name -> (n_steps, X, Y, Z)
    scenario_name: str,
    t0_seconds: float,
    out_path: Path,
) -> None:
    """4 row × 6 col (truth + 3 모델) × (t₀+10, ..., t₀+60s)."""
    rows = ["FDS truth"] + list(model_preds_danger.keys())
    n_rows = len(rows)
    n_cols = truth_danger.shape[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for col in range(n_cols):
        t_label = f"t = {t0_seconds + (col + 1) * DT_SLCF:.0f} s"
        # truth
        im = axes[0, col].imshow(
            truth_danger[col, :, :, Z_IDX].T,
            origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
            extent=[0, 30, 0, 20], aspect="equal",
        )
        axes[0, col].set_title(t_label, fontsize=10)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        # models
        for r, name in enumerate(model_preds_danger.keys(), start=1):
            axes[r, col].imshow(
                model_preds_danger[name][col, :, :, Z_IDX].T,
                origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
                extent=[0, 30, 0, 20], aspect="equal",
            )
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])

    # row labels
    for r, label in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=10)

    # single shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="danger ∈ [0, 1]")

    fig.suptitle(
        f"60 s 미래 위험 예측  |  {scenario_name}  |  관측 시점 t₀ = {t0_seconds:.0f} s  "
        f"(z = 1.75 m, autoregressive rollout)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Sliding-t₀ animation ──────────────────────────────────────────────────
def plot_sliding_animation(
    sliding_truth: List[np.ndarray],     # list of (6, X, Y, Z) per t₀
    sliding_models: List[Dict[str, np.ndarray]],  # list of {name: (6, X, Y, Z)}
    t0_list: List[float],
    scenario_name: str,
    out_path: Path,
) -> None:
    """t₀ 가 슬라이딩하면서 t₀+60s 의 truth vs 모델별 예측 비교 (4 panel × N t₀ frame)."""
    if not sliding_truth:
        return
    model_names = list(sliding_models[0].keys())
    fig, axes = plt.subplots(1, len(model_names) + 1, figsize=(4.5 * (len(model_names) + 1), 4))
    panels = []
    panels.append(axes[0].imshow(
        sliding_truth[0][-1, :, :, Z_IDX].T,
        origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
        extent=[0, 30, 0, 20], aspect="equal",
    ))
    axes[0].set_title("FDS truth", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(panels[0], ax=axes[0], fraction=0.04)
    for i, name in enumerate(model_names, start=1):
        panels.append(axes[i].imshow(
            sliding_models[0][name][-1, :, :, Z_IDX].T,
            origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
            extent=[0, 30, 0, 20], aspect="equal",
        ))
        axes[i].set_title(name, fontsize=11)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        plt.colorbar(panels[i], ax=axes[i], fraction=0.04)
    title = fig.suptitle(
        f"{scenario_name}  |  t₀ = {t0_list[0]:.0f}s  →  60s lookahead",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    writer = PillowWriter(fps=1)
    with writer.saving(fig, str(out_path), dpi=85):
        for f_idx, t0 in enumerate(t0_list):
            panels[0].set_data(sliding_truth[f_idx][-1, :, :, Z_IDX].T)
            for i, name in enumerate(model_names, start=1):
                panels[i].set_data(sliding_models[f_idx][name][-1, :, :, Z_IDX].T)
            title.set_text(
                f"{scenario_name}  |  t₀ = {t0:.0f}s  →  prediction at t = {t0+60:.0f}s"
            )
            writer.grab_frame()
    plt.close(fig)


# ─── Per-step error curve ──────────────────────────────────────────────────
def plot_error_curves(
    truth_danger: np.ndarray,
    model_preds_danger: Dict[str, np.ndarray],
    mask: np.ndarray,
    scenario_name: str,
    out_path: Path,
) -> None:
    fluid = (mask > 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, preds in model_preds_danger.items():
        rmse = []
        iou = []
        for step in range(preds.shape[0]):
            t_img = truth_danger[step]
            p_img = preds[step]
            # RMSE on fluid cells
            diff = (p_img - t_img)[fluid]
            rmse.append(float(np.sqrt(np.mean(diff ** 2))))
            # IoU @ 0.5
            tp = float(np.sum((p_img[fluid] >= 0.5) & (t_img[fluid] >= 0.5)))
            fp = float(np.sum((p_img[fluid] >= 0.5) & (t_img[fluid] < 0.5)))
            fn = float(np.sum((p_img[fluid] < 0.5) & (t_img[fluid] >= 0.5)))
            iou.append(tp / (tp + fp + fn + 1e-9))
        steps = np.arange(1, preds.shape[0] + 1)
        axes[0].plot(steps, rmse, "o-", label=name, lw=1.6)
        axes[1].plot(steps, iou, "o-", label=name, lw=1.6)
    axes[0].set_xlabel("autoregress step (10 s each)")
    axes[0].set_ylabel("danger RMSE on fluid cells")
    axes[0].set_title("Per-step error (lower = better)")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("autoregress step (10 s each)")
    axes[1].set_ylabel("Risk IoU @ 0.5")
    axes[1].set_title("Per-step IoU (higher = better)")
    axes[1].axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(f"60 s rollout error compounding  |  {scenario_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Main pipeline ────────────────────────────────────────────────────────
def evaluate_one_scenario(
    fds_dir: Path,
    models: Dict[str, torch.nn.Module],
    mask: np.ndarray,
    device: torch.device,
    t0_list: List[float],
    out_dir: Path,
) -> None:
    scenario_name = fds_dir.name
    print(f"[scen] {scenario_name}")

    # Extract truth + build (input, target)
    slices = extract_slices(fds_dir)
    norm = normalize_scenario(slices)
    inp = build_input_tensor(norm, mask, times=slices["times"])  # (31, 5, ...)
    tgt = build_target_tensor(norm)                              # (31, 3, ...)

    # Truth danger from raw slices
    truth_danger_full = compute_total_danger(
        slices["temperature"], slices["visibility"], slices["co"],
    ).astype(np.float32)  # (31, X, Y, Z)

    sliding_truth: List[np.ndarray] = []
    sliding_models: List[Dict[str, np.ndarray]] = []

    # 1) Grid figure for t₀ = 0
    for t0_seconds in t0_list:
        t0_idx = int(t0_seconds // DT_SLCF)
        if t0_idx + LOOKAHEAD_STEPS >= N_TIMESTEPS:
            print(f"  skip t0={t0_seconds}s (would exceed sim end)")
            continue

        # Truth windows
        truth_window = truth_danger_full[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

        # Model rollouts
        preds_danger_by_model: Dict[str, np.ndarray] = {}
        for name, model in models.items():
            initial = inp[t0_idx]
            preds_norm = autoregress_60s(model, initial, t0_seconds, device)
            times_arr = np.array([t0_seconds + (s + 1) * DT_SLCF for s in range(LOOKAHEAD_STEPS)])
            preds_danger_by_model[name] = prediction_to_danger(preds_norm, times_arr)

        sliding_truth.append(truth_window)
        sliding_models.append(preds_danger_by_model)

        # Grid figure (모든 t₀ 에서)
        grid_path = out_dir / f"{scenario_name}_grid_t0_{int(t0_seconds):03d}.png"
        plot_60s_grid(truth_window, preds_danger_by_model, scenario_name, t0_seconds, grid_path)
        print(f"  → {grid_path.name}")
        err_path = out_dir / f"{scenario_name}_error_curve_t0_{int(t0_seconds):03d}.png"
        plot_error_curves(truth_window, preds_danger_by_model, mask, scenario_name, err_path)
        print(f"  → {err_path.name}")

    # 2) Sliding animation (multiple t₀ frames)
    if len(sliding_truth) >= 2:
        anim_path = out_dir / f"{scenario_name}_sliding_animation.gif"
        plot_sliding_animation(
            sliding_truth, sliding_models,
            [t for t in t0_list if int(t // DT_SLCF) + LOOKAHEAD_STEPS < N_TIMESTEPS],
            scenario_name, anim_path,
        )
        print(f"  → {anim_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--output", type=Path, default=Path("figures/future_prediction"))
    parser.add_argument(
        "--scenarios", type=str, nargs="+",
        default=[
            "sim_1500kw_2m2_T05",  # 가장 잘 예측되는 케이스
            "sim_500kw_1m2_T01",   # 가장 어려운 케이스
            "sim_1000kw_1m2_T03",  # 중간 난이도
        ],
        help="대상 시나리오 이름 (sim_*_T* 형식)",
    )
    parser.add_argument(
        "--t0-list", type=float, nargs="+",
        default=[0.0, 30.0, 60.0, 120.0, 180.0],
        help="autoregress 시작 시점 (초)",
    )
    parser.add_argument(
        "--ckpt-conv-lstm", type=Path,
        default=Path("checkpoints/conv_lstm/best.pt"),
    )
    parser.add_argument(
        "--ckpt-fno-no-pi", type=Path,
        default=Path("checkpoints/fno_no_pi/best.pt"),
    )
    parser.add_argument(
        "--ckpt-fno-pi", type=Path,
        default=Path("checkpoints/fno_pi/best.pt"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[setup] device={device}")
    print(f"[setup] loading 3 models...")
    models = {
        "ConvLSTM":  load_model(args.ckpt_conv_lstm,  device, model_type="conv_lstm"),
        "FNO no-PI": load_model(args.ckpt_fno_no_pi,  device, model_type="fno"),
        "FNO PI":    load_model(args.ckpt_fno_pi,     device, model_type="fno"),
    }
    mask = load_mask(args.dataset)

    for sname in args.scenarios:
        sdir = args.raw_root / sname
        if not sdir.is_dir():
            print(f"[skip] {sname} not a directory under {args.raw_root}")
            continue
        evaluate_one_scenario(sdir, models, mask, device, args.t0_list, args.output)

    print(f"\n[PASS] outputs at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
