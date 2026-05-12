"""ConvLSTM 학습 결과 종합 평가.

수행 항목:
1. 학습 곡선 (per-epoch train loss)
2. 단일 스텝 예측 정확도:
   - 정규화 도메인 MSE / RelL2 (가설 H2 ≤ 15% 기준)
   - 물리 단위 RMSE (°C, m, ppm)
3. 자가회귀 (autoregression) 6-step rollout 오차 누적 (L-011 검증)
4. Inference 속도 측정 (가설 H1 ≥ 1000× vs FDS)
5. z=1.75 m 단면 예측 vs 정답 시각화

출력: ``figures/eval_convlstm/`` 디렉토리.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.conv_lstm_3d import FireConvLSTM
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS, T_END_SECONDS
from src.shared.normalization import (
    denormalize_co,
    denormalize_temperature,
    denormalize_visibility,
)


# ── Model loading ──────────────────────────────────────────────────────────
def load_model(ckpt: Path, device: torch.device) -> FireConvLSTM:
    # weights_only=False — own-trained checkpoint (PyTorch 2.6+ default change)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        # FNO-style checkpoint dict with metadata
        state = state["model"]
    # _metadata round-trip fix (PyTorch 2.6+ serialization quirk)
    if "_metadata" in state:
        try:
            state._metadata = state.pop("_metadata")
        except Exception:
            state.pop("_metadata", None)
    model = FireConvLSTM(
        in_channels=5, out_channels=3, hidden_dim=32,
        kernel_size=(3, 3, 3), num_layers=2,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ── Metrics ────────────────────────────────────────────────────────────────
def rel_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> float:
    """Relative L2 norm — ‖pred-target‖₂ / ‖target‖₂."""
    num = float(np.sqrt(np.mean((pred - target) ** 2)))
    den = float(np.sqrt(np.mean(target ** 2))) + eps
    return num / den


def rmse_per_channel_normalized(
    pred: np.ndarray, target: np.ndarray,
) -> Tuple[float, float, float]:
    """RMSE per channel in normalised [0, 1] domain. ``pred/target``: (C=3, X, Y, Z)."""
    return (
        float(np.sqrt(np.mean((pred[0] - target[0]) ** 2))),
        float(np.sqrt(np.mean((pred[1] - target[1]) ** 2))),
        float(np.sqrt(np.mean((pred[2] - target[2]) ** 2))),
    )


def rmse_per_channel_physical(
    pred: np.ndarray, target: np.ndarray,
) -> Tuple[float, float, float]:
    """RMSE in physical units (°C, m, ppm)."""
    T_p = denormalize_temperature(pred[0])
    T_t = denormalize_temperature(target[0])
    V_p = denormalize_visibility(pred[1])
    V_t = denormalize_visibility(target[1])
    CO_p = denormalize_co(pred[2])
    CO_t = denormalize_co(target[2])
    return (
        float(np.sqrt(np.mean((T_p - T_t) ** 2))),
        float(np.sqrt(np.mean((V_p - V_t) ** 2))),
        float(np.sqrt(np.mean((CO_p - CO_t) ** 2))),
    )


# ── Single-step evaluation ─────────────────────────────────────────────────
def evaluate_single_step(
    model: FireConvLSTM,
    h5: h5py.File,
    scenario_keys: List[str],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """For each scenario, compute mean single-step prediction error across all pairs."""
    out: Dict[str, Dict[str, float]] = {}
    for key in scenario_keys:
        inp = np.asarray(h5[key]["input"])    # (31, 5, X, Y, Z)
        tgt = np.asarray(h5[key]["target"])   # (31, 3, X, Y, Z)
        # pairs: input[t] → target[t+1]
        n_pairs = inp.shape[0] - 1
        rel = []
        rmse_n_T, rmse_n_V, rmse_n_CO = [], [], []
        rmse_p_T, rmse_p_V, rmse_p_CO = [], [], []
        with torch.no_grad():
            for t in range(n_pairs):
                x = torch.from_numpy(inp[t]).unsqueeze(0).to(device)
                y_true = tgt[t + 1]
                y_pred = model(x).cpu().numpy()[0]
                y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
                rel.append(rel_l2(y_pred_clipped, y_true))
                rT, rV, rC = rmse_per_channel_normalized(y_pred_clipped, y_true)
                rmse_n_T.append(rT); rmse_n_V.append(rV); rmse_n_CO.append(rC)
                pT, pV, pC = rmse_per_channel_physical(y_pred_clipped, y_true)
                rmse_p_T.append(pT); rmse_p_V.append(pV); rmse_p_CO.append(pC)
        out[key] = {
            "n_pairs": n_pairs,
            "rel_l2_mean": float(np.mean(rel)),
            "rel_l2_max": float(np.max(rel)),
            "rmse_norm_T": float(np.mean(rmse_n_T)),
            "rmse_norm_V": float(np.mean(rmse_n_V)),
            "rmse_norm_CO": float(np.mean(rmse_n_CO)),
            "rmse_C": float(np.mean(rmse_p_T)),
            "rmse_m": float(np.mean(rmse_p_V)),
            "rmse_ppm": float(np.mean(rmse_p_CO)),
        }
    return out


# ── Autoregressive 6-step rollout ─────────────────────────────────────────
def evaluate_autoregress(
    model: FireConvLSTM,
    h5: h5py.File,
    scenario_key: str,
    start_t_idx: int,
    n_steps: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    """6-step rollout from ``start_t_idx`` and per-step RelL2 vs ground truth."""
    inp = np.asarray(h5[scenario_key]["input"])    # (31, 5, X, Y, Z)
    tgt = np.asarray(h5[scenario_key]["target"])   # (31, 3, X, Y, Z)
    state = inp[start_t_idx].copy()                # (5, X, Y, Z)
    mask = state[3]                                # building mask channel
    metrics: List[Dict[str, float]] = []
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            x = torch.from_numpy(state).unsqueeze(0).to(device)
            y_pred = np.clip(model(x).cpu().numpy()[0], 0.0, 1.0)  # (3, X, Y, Z)
            t_idx = start_t_idx + step
            if t_idx >= tgt.shape[0]:
                break
            y_true = tgt[t_idx]
            metrics.append({
                "step": step,
                "t_idx": t_idx,
                "t_seconds": t_idx * DT_SLCF,
                "rel_l2": rel_l2(y_pred, y_true),
                "rmse_norm_T": float(np.sqrt(np.mean((y_pred[0] - y_true[0]) ** 2))),
                "rmse_C_T": float(np.sqrt(np.mean((
                    denormalize_temperature(y_pred[0]) - denormalize_temperature(y_true[0])
                ) ** 2))),
            })
            # Build next-step input: predicted T/V/CO + same mask + new time enc
            t_norm = (start_t_idx + step) * DT_SLCF / T_END_SECONDS
            state = np.zeros_like(state)
            state[:3] = y_pred
            state[3] = mask
            state[4] = np.full_like(mask, t_norm)
    return metrics


# ── Visualisation ──────────────────────────────────────────────────────────
def plot_prediction_panel(
    pred: np.ndarray, target: np.ndarray,
    scenario_key: str, t_idx: int,
    out_path: Path,
) -> None:
    """At z=1.75 m, show pred vs target vs |error| for T, V, CO."""
    z = 3  # z=1.75m
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    channel_names = ["T_norm", "V_norm", "CO_norm"]
    for c in range(3):
        p = pred[c, :, :, z].T
        t = target[c, :, :, z].T
        e = np.abs(p - t)
        for col, (img, ttl, cmap, vmin, vmax) in enumerate([
            (t, f"target {channel_names[c]}", "RdYlGn_r", 0, 1),
            (p, f"pred {channel_names[c]}",   "RdYlGn_r", 0, 1),
            (e, f"|err|",                     "magma",    0, max(0.05, float(e.max()))),
        ]):
            im = axes[c, col].imshow(
                img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[0, 30, 0, 20], aspect="equal",
            )
            axes[c, col].set_title(ttl, fontsize=10)
            plt.colorbar(im, ax=axes[c, col], fraction=0.04)
            axes[c, col].set_xticks([]); axes[c, col].set_yticks([])
    fig.suptitle(
        f"ConvLSTM single-step prediction  |  {scenario_key}  |  "
        f"t={(t_idx) * DT_SLCF:.0f}s → t={(t_idx+1) * DT_SLCF:.0f}s  (z=1.75m)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_loss_curve(loss_csv: Path, out_path: Path) -> None:
    epochs, losses = [], []
    with open(loss_csv, encoding="utf-8") as f:
        for line in f:
            e, l = line.strip().split(",")
            epochs.append(int(e)); losses.append(float(l))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, losses, lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (MSE, log scale)")
    ax.set_title(f"ConvLSTM training loss — final = {losses[-1]:.6f}")
    ax.grid(alpha=0.3, which="both")
    ax.axhline(losses[-1], color="red", lw=0.7, ls="--",
               label=f"final {losses[-1]:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_autoregress(
    metrics: List[Dict[str, float]], scenario_key: str, out_path: Path,
) -> None:
    if not metrics:
        return
    steps = [m["step"] for m in metrics]
    rel = [m["rel_l2"] for m in metrics]
    rmse_T = [m["rmse_C_T"] for m in metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(steps, rel, marker="o", lw=1.5)
    ax1.set_xlabel("autoregress step (10s each)")
    ax1.set_ylabel("Relative L2")
    ax1.set_title(f"Error compounding — {scenario_key}")
    ax1.grid(alpha=0.3)
    ax1.axhline(0.15, color="red", lw=0.8, ls="--", label="H2 target ≤ 15%")
    ax1.legend()
    ax2.plot(steps, rmse_T, marker="o", lw=1.5, color="tab:red")
    ax2.set_xlabel("autoregress step")
    ax2.set_ylabel("Temperature RMSE (°C)")
    ax2.set_title("Per-step T RMSE (physical units)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── Inference timing ──────────────────────────────────────────────────────
def measure_inference_time(
    model: FireConvLSTM, device: torch.device, n_runs: int = 50,
) -> Tuple[float, float]:
    """Average forward time + std (ms) for batch=1 input."""
    x = torch.rand(1, 5, *GRID_SHAPE, device=device)
    # warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/conv_lstm/best.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--output", type=Path, default=Path("figures/eval_convlstm"))
    parser.add_argument("--loss-csv", type=Path, default=Path("/tmp/loss.csv"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"checkpoint: {args.ckpt} ({args.ckpt.stat().st_size / 1024:.1f} KB)")

    # 0. Model
    model = load_model(args.ckpt, device)
    print(f"model params: {model.count_parameters():,}")

    # 1. Loss curve
    if args.loss_csv.exists():
        plot_loss_curve(args.loss_csv, args.output / "loss_curve.png")

    # 2. Single-step eval on diverse scenarios
    print("\n=== Single-step evaluation ===")
    eval_keys = [
        "scenario_000",  # 500kW @ loc 001 (Zone C 좌측)
        "scenario_007",  # 1000kW @ loc 001
        "scenario_014",  # 1500kW @ loc 001
        "scenario_011",  # 1000kW @ loc 005 (중앙 홀)
        "scenario_032",  # 1000kW H03 (held-out 위치 — D-024 후 train에 포함됨)
    ]
    with h5py.File(args.dataset, "r") as h5:
        results = evaluate_single_step(model, h5, eval_keys, device)
        for k, m in results.items():
            print(
                f"  {k}: rel_l2={m['rel_l2_mean']:.4f}±max{m['rel_l2_max']:.4f}  "
                f"RMSE(°C)={m['rmse_C']:.2f}  RMSE(m)={m['rmse_m']:.3f}  "
                f"RMSE(ppm)={m['rmse_ppm']:.2f}"
            )

        # 3. Visualisation — 2 scenarios × 2 timesteps
        print("\n=== Prediction visualisations ===")
        for k in ["scenario_014", "scenario_032"]:
            for t_idx in [10, 20]:  # t=100s, t=200s
                inp = np.asarray(h5[k]["input"][t_idx])
                tgt = np.asarray(h5[k]["target"][t_idx + 1])
                with torch.no_grad():
                    x = torch.from_numpy(inp).unsqueeze(0).to(device)
                    pred = np.clip(model(x).cpu().numpy()[0], 0.0, 1.0)
                plot_prediction_panel(
                    pred, tgt, k, t_idx,
                    args.output / f"pred_{k}_t{t_idx * 10}s.png",
                )

        # 4. Autoregressive rollout
        print("\n=== Autoregressive 6-step rollout ===")
        autoreg = evaluate_autoregress(
            model, h5, "scenario_014", start_t_idx=10, n_steps=6, device=device,
        )
        for m in autoreg:
            print(
                f"  step {m['step']} (t={m['t_seconds']:.0f}s): "
                f"rel_l2={m['rel_l2']:.4f}  RMSE_T={m['rmse_C_T']:.2f}°C"
            )
        plot_autoregress(autoreg, "scenario_014", args.output / "autoregress.png")

    # 5. Inference timing
    print("\n=== Inference timing ===")
    mean_ms, std_ms = measure_inference_time(model, device, n_runs=50)
    print(f"  forward (batch=1, {device}): {mean_ms:.2f} ± {std_ms:.2f} ms")
    # FDS wall-clock per scenario ≈ 23 min = 1.38e6 ms / 31 frames = ~44,500 ms per frame
    speedup = 44500.0 / mean_ms
    print(f"  vs FDS (~44,500 ms/frame on CPU cluster): ~{speedup:.0f}×")

    # 6. Save JSON summary
    summary = {
        "checkpoint": str(args.ckpt),
        "n_params": model.count_parameters(),
        "device": str(device),
        "inference_ms_mean": mean_ms,
        "inference_ms_std": std_ms,
        "speedup_vs_fds_estimated": float(speedup),
        "single_step_per_scenario": results,
        "autoregress_scenario_014": autoreg,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsummary → {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
