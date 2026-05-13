"""Track 1B 평가 — Sparse-input ConvLSTM 으로 60s 미래 예측 (보간 없음).

기존 sparse evaluation (`evaluate_sparse_sensing.py`, `evaluate_sparse_sensing_geodesic.py`)
는 sparse → interpolation → 기존 모델 의 cascade. 본 스크립트는:

* 16 sensor 위치 cell 만 측정값 유지 (다른 fluid cell 의 T/V/CO = 0)
* **보간 단계 없음** — 모델 자체가 sparse → dense 학습
* 같은 OOD 시나리오 + t₀=120s 로 비교

비교 baseline:
* Layer 2 (full SLCF + 기존 ConvLSTM): IoU 0.92 ideal
* Layer 4a (16 sensor + nearest):       IoU 0.28
* Layer 4b (16 sensor + linear):        IoU 0.19
* Layer 4d (16 sensor + geodesic IDW):  IoU 0.41
* **Layer 4e** (16 sensor + sparse retrain): IoU ???  ← 본 스크립트
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import (
    normalize_co, normalize_temperature, normalize_visibility,
)
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import (
    DT_SLCF, GRID_SHAPE, N_INPUT_CHANNELS, N_TIMESTEPS, T_END_SECONDS,
)
from src.shared.coordinates import cell_centres
from evaluate_t_locations import load_model, load_mask
from train_sparse_conv_lstm import (
    load_sensor_indices, make_sparse_indicator, sparsify_input,
)

SCEN_RE = re.compile(r"^sim_(?P<hrr>\d+)kw_(?P<area>\d+)m2_(?P<loc>T\d{2})$")
LOOKAHEAD_STEPS = 6


def autoregress_sparse(
    model: torch.nn.Module,
    initial_input_sparse: np.ndarray,   # (5, X, Y, Z) — already sparsified
    sparse_ind: np.ndarray,              # (X, Y, Z) bool
    t0_seconds: float,
    device: torch.device,
    n_steps: int = LOOKAHEAD_STEPS,
    resparsify: bool = False,
) -> np.ndarray:
    """Autoregress — input/output 모두 sparse 가정.

    옵션 `resparsify=True` 이면 매 step 의 모델 출력에서 sensor 위치 외 cell 의
    T/V/CO 를 0 으로 강제. 그러나 dense target 으로 학습된 모델이므로 기본은
    그대로 chaining.
    """
    state = initial_input_sparse.copy()
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
            if resparsify:
                not_sensor = ~sparse_ind
                for c in range(3):
                    state[c][not_sensor] = 0.0
            state[3] = mask_ch
            state[4] = np.full_like(mask_ch, t_next / T_END_SECONDS)
    return preds


def iou_rmse(pred_d: np.ndarray, true_d: np.ndarray, mask: np.ndarray,
              threshold: float = 0.5) -> Dict[str, float]:
    fluid = (mask > 0.5)
    fm = np.broadcast_to(fluid, pred_d.shape)
    p = (pred_d >= threshold); t = (true_d >= threshold)
    tp = float(np.sum(p & t & fm)); fp = float(np.sum(p & (~t) & fm))
    fn = float(np.sum((~p) & t & fm)); tn = float(np.sum((~p) & (~t) & fm))
    return {
        "iou": tp / (tp + fp + fn + 1e-9),
        "fnr": fn / (fn + tp + 1e-9),
        "rmse": float(np.sqrt(np.mean(
            (pred_d - true_d).astype(np.float64)[fm.reshape(pred_d.shape)] ** 2
        ))),
    }


def build_sparse_input(slices: Dict[str, np.ndarray],
                        mask: np.ndarray,
                        sparse_ind: np.ndarray) -> np.ndarray:
    """FDS truth → normalised + sparsified (31, 5, X, Y, Z) input tensor."""
    T = normalize_temperature(slices["temperature"]).astype(np.float32)
    V = normalize_visibility(slices["visibility"]).astype(np.float32)
    CO = normalize_co(slices["co"]).astype(np.float32)
    times = np.arange(N_TIMESTEPS) * DT_SLCF
    te = (times / T_END_SECONDS).astype(np.float32)
    expected = (N_TIMESTEPS, *GRID_SHAPE)
    mask_b = np.broadcast_to(mask.astype(np.float32)[None, :, :, :], expected).astype(np.float32)
    te_grid = np.broadcast_to(te[:, None, None, None], expected).astype(np.float32)
    inp = np.stack([T, V, CO, mask_b, te_grid], axis=1).astype(np.float32)
    # Sparsify T, V, CO channels
    not_sensor = ~sparse_ind
    for c in range(3):
        inp[:, c, not_sensor[..., 0]] = 0.0  # broadcast across batch dim if needed
    # Wait: not_sensor is (X, Y, Z) shape, we need to broadcast over time
    for c in range(3):
        for t in range(N_TIMESTEPS):
            inp[t, c][not_sensor] = 0.0
    return inp


def eval_scenario(scen_dir: Path, sparse_model: torch.nn.Module,
                   mask: np.ndarray, sparse_ind: np.ndarray,
                   t0_seconds: float, device: torch.device) -> Dict[str, Any]:
    name = scen_dir.name
    m = SCEN_RE.match(name)
    meta = {"name": name, "loc": m.group("loc"),
            "hrr_kw": int(m.group("hrr")), "area_m2": int(m.group("area")),
            "t0": t0_seconds}
    slices = extract_slices(scen_dir)
    truth_danger = compute_total_danger(
        slices["temperature"], slices["visibility"], slices["co"]).astype(np.float32)
    t0_idx = int(t0_seconds // DT_SLCF)
    if t0_idx + LOOKAHEAD_STEPS >= N_TIMESTEPS:
        return meta
    truth_window = truth_danger[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

    print(f"[scen] {name}")
    # Build sparse input
    inp = build_sparse_input(slices, mask, sparse_ind)
    # Autoregress 60s
    preds_norm = autoregress_sparse(
        sparse_model, inp[t0_idx], sparse_ind, t0_seconds, device,
    )
    times_arr = np.array([t0_seconds + (s + 1) * DT_SLCF for s in range(LOOKAHEAD_STEPS)])
    preds_danger = prediction_to_danger(preds_norm, times_arr)
    m6 = iou_rmse(preds_danger[5:6], truth_window[5:6], mask)
    m1 = iou_rmse(preds_danger[0:1], truth_window[0:1], mask)
    mall = iou_rmse(preds_danger, truth_window, mask)
    meta.update({
        "iou_step1": m1["iou"], "iou_step6": m6["iou"], "iou_all": mall["iou"],
        "rmse_step6": m6["rmse"], "fnr_step6": m6["fnr"],
    })
    return meta


def plot_comparison_full_stack(sparse_results: List[Dict[str, Any]],
                                out_path: Path) -> None:
    """Layer 별 비교 막대 — L2 / L4a-nearest / L4b-linear / L4d-geodesic / L4e-retrain.

    Hard-coded values from previous evaluations:
    """
    PREV = {
        "L2 (Full SLCF\nideal)":           0.92,
        "L4a (16 sensors\n+ nearest)":     0.28,
        "L4b (16 sensors\n+ linear)":      0.19,
        "L4c (16 sensors\n+ cubic)":       0.19,
        "L4d (16 sensors\n+ geodesic)":    0.41,
    }
    if sparse_results:
        l4e = np.mean([r["iou_step6"] for r in sparse_results if "iou_step6" in r])
    else:
        l4e = float("nan")
    keys = list(PREV.keys()) + ["L4e (16 sensors\n+ sparse retrain)"]
    vals = list(PREV.values()) + [l4e]
    colors = ["tab:green", "tab:gray", "tab:gray", "tab:gray", "tab:purple", "tab:red"]
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(keys)), vals, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel("IoU at t₀ + 60s  (ConvLSTM)")
    ax.set_title("Evaluation Layer L2 → L4 — IoU progression on OOD T01-T05 "
                 "(16 sensors, t₀ = 120s)")
    ax.axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_scenario(results: List[Dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    names = [r["name"].replace("sim_", "") for r in results]
    ious = [r.get("iou_step6", 0) for r in results]
    bars = ax.bar(names, ious, color="tab:purple")
    ax.set_ylabel("IoU at t₀ + 60s")
    ax.set_title("Sparse-input ConvLSTM (Track 1B) — per-scenario IoU @ t₀=120s+60s")
    ax.axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    ax.tick_params(axis="x", rotation=45)
    for b, v in zip(bars, ious):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=Path("checkpoints/conv_lstm_sparse/best.pt"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path, default=Path("configs/building.yaml"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/sparse_retrain"))
    parser.add_argument("--out-csv", type=Path,
                        default=Path("results/exp_sparse_retrain/comparison.csv"))
    parser.add_argument("--out-report", type=Path,
                        default=Path("docs/sparse_retrain_evaluation.md"))
    parser.add_argument("--t0", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[setup] device={device}")
    sparse_model = load_model(args.ckpt, device, "conv_lstm")
    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)
    print(f"[setup] {len(sensor_idxs)} sensors → {int(np.sum(sparse_ind))} cells")

    scens = sorted(d for d in args.raw_root.glob("sim_*_T*") if d.is_dir())
    results = []
    for scen in scens:
        try:
            r = eval_scenario(scen, sparse_model, mask, sparse_ind, args.t0, device)
            results.append(r)
        except Exception as e:
            print(f"[skip] {scen.name}: {e}")

    print("\n[agg]")
    if results:
        mean_iou = np.mean([r.get("iou_step6", 0) for r in results])
        mean_fnr = np.mean([r.get("fnr_step6", 0) for r in results])
        mean_rmse = np.mean([r.get("rmse_step6", 0) for r in results])
        print(f"  Mean IoU step 6 (60s):  {mean_iou:.3f}")
        print(f"  Mean FNR step 6:        {mean_fnr*100:.1f}%")
        print(f"  Mean RMSE step 6:       {mean_rmse:.3f}")

    print("[plot] full_stack_comparison.png")
    plot_comparison_full_stack(results, args.out_figures / "full_stack_comparison.png")
    print("[plot] per_scenario.png")
    plot_per_scenario(results, args.out_figures / "per_scenario.png")

    # CSV
    cols = ["name", "loc", "hrr_kw", "area_m2", "t0",
            "iou_step1", "iou_step6", "iou_all", "rmse_step6", "fnr_step6"]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, "") for c in cols])

    # Report
    lines = []
    lines.append("# Track 1B — Sparse-input ConvLSTM 재학습 평가\n")
    lines.append("> **목적**: 보간 단계 제거 — 모델 자체가 16 sensor 의 sparse signal "
                 "로부터 dense 60s 미래 예측을 직접 학습.\n")
    lines.append(f"> **체크포인트**: `{args.ckpt}`")
    lines.append(f"> **t₀ = {args.t0:.0f} s, lookahead 60 s, 16 sensors**\n")
    lines.append("\n## 1. 평균 결과 (13 OOD 시나리오)\n")
    if results:
        lines.append(f"- **IoU step 6 (60s 미래):** **{mean_iou:.3f}**")
        lines.append(f"- FNR step 6: {mean_fnr*100:.1f}%")
        lines.append(f"- RMSE step 6: {mean_rmse:.3f}")
        h5_pass = mean_iou >= 0.70
        lines.append(f"- H5 (≥ 0.70) 통과: {'✅ YES' if h5_pass else '❌ NO'}")

    lines.append("\n## 2. Layer-by-layer 비교\n")
    lines.append(f"![]({(args.out_figures / 'full_stack_comparison.png').as_posix()})\n")
    lines.append("\n## 3. 시나리오별 IoU\n")
    lines.append(f"![]({(args.out_figures / 'per_scenario.png').as_posix()})\n")
    lines.append("\n| 시나리오 | HRR | area | IoU step 1 | IoU step 6 | RMSE step 6 |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['hrr_kw']} kW | {r['area_m2']} m² | "
            f"{r.get('iou_step1', 0):.3f} | {r.get('iou_step6', 0):.3f} | "
            f"{r.get('rmse_step6', 0):.3f} |"
        )

    args.out_report.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[PASS]")
    print(f"  CSV: {args.out_csv}")
    print(f"  Report: {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
