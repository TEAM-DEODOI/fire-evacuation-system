"""Sparse-input FNO (6-channel) 평가 — 13 OOD 시나리오.

평가 모드:
1. **Re-sparsify chaining** (default True) — autoregress 시 매 step sensor 외
   T/V/CO 를 0 으로 강제 (L-013 fix 적용).
2. Sensor indicator (6번째 채널) — 모든 step 에서 그대로 유지 (mask 처럼).

산출물:
- figures/current/08_sparse_fno_v3/full_stack_comparison.png
- figures/current/08_sparse_fno_v3/per_scenario.png
- results/exp_sparse_fno_v3/comparison.csv
- docs/archive/auto_reports/sparse_fno_v3_evaluation.md
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import (
    normalize_co, normalize_temperature, normalize_visibility,
)
from src.models.fno_model import FNOFireModel
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS, T_END_SECONDS
from evaluate_t_locations import load_mask
from train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator

SCEN_RE = re.compile(r"^sim_(?P<hrr>\d+)kw_(?P<area>\d+)m2_(?P<loc>T\d{2})$")
LOOKAHEAD_STEPS = 6


def load_sparse_fno(ckpt: Path, device: torch.device,
                      hidden_channels=32, n_layers=4,
                      lifting_channels=128, projection_channels=128) -> FNOFireModel:
    model = FNOFireModel(
        n_modes=(12, 12, 4),
        in_channels=6,
        out_channels=3,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        lifting_channels=lifting_channels,
        projection_channels=projection_channels,
    )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if "_metadata" in state:
        try:
            state._metadata = state.pop("_metadata")
        except Exception:
            state.pop("_metadata", None)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def build_sparse_6ch_input(
    slices: Dict[str, np.ndarray],
    mask: np.ndarray,
    sparse_ind: np.ndarray,
) -> np.ndarray:
    """FDS truth → 6-channel sparse input tensor (31, 6, X, Y, Z)."""
    T = normalize_temperature(slices["temperature"]).astype(np.float32)
    V = normalize_visibility(slices["visibility"]).astype(np.float32)
    CO = normalize_co(slices["co"]).astype(np.float32)
    times = np.arange(N_TIMESTEPS) * DT_SLCF
    te = (times / T_END_SECONDS).astype(np.float32)
    expected = (N_TIMESTEPS, *GRID_SHAPE)
    mask_b = np.broadcast_to(mask.astype(np.float32)[None, :, :, :], expected).astype(np.float32)
    te_grid = np.broadcast_to(te[:, None, None, None], expected).astype(np.float32)
    sensor_b = np.broadcast_to(
        sparse_ind.astype(np.float32)[None, :, :, :], expected
    ).astype(np.float32)
    inp = np.stack([T, V, CO, mask_b, te_grid, sensor_b], axis=1).astype(np.float32)
    # Sparsify T/V/CO
    not_sensor = ~sparse_ind
    for c in range(3):
        for t in range(N_TIMESTEPS):
            inp[t, c][not_sensor] = 0.0
    return inp


def autoregress_sparse_fno(
    model: FNOFireModel,
    initial_input_6ch: np.ndarray,    # (6, X, Y, Z) sparse
    sparse_ind: np.ndarray,
    t0_seconds: float,
    device: torch.device,
    n_steps: int = LOOKAHEAD_STEPS,
    resparsify: bool = True,
) -> np.ndarray:
    """6-channel sparse FNO autoregress. Default resparsify=True (L-013 fix)."""
    state = initial_input_6ch.copy()
    mask_ch = state[3].copy()
    sensor_ch = state[5].copy()
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
            if resparsify:
                for c in range(3):
                    state[c][not_sensor] = 0.0
            state[3] = mask_ch
            state[4] = np.full_like(mask_ch, t_next / T_END_SECONDS)
            state[5] = sensor_ch
    return preds


def iou_rmse(pred_d, true_d, mask, threshold=0.5):
    fluid = (mask > 0.5)
    fm = np.broadcast_to(fluid, pred_d.shape)
    p = pred_d >= threshold; t = true_d >= threshold
    tp = float(np.sum(p & t & fm)); fp = float(np.sum(p & (~t) & fm))
    fn = float(np.sum((~p) & t & fm)); tn = float(np.sum((~p) & (~t) & fm))
    return {
        "iou": tp / (tp + fp + fn + 1e-9),
        "fnr": fn / (fn + tp + 1e-9),
        "rmse": float(np.sqrt(np.mean(
            (pred_d - true_d).astype(np.float64)[fm.reshape(pred_d.shape)] ** 2
        ))),
    }


def eval_scenario(scen_dir, model, mask, sparse_ind, t0_seconds, device,
                   resparsify=True):
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
    inp = build_sparse_6ch_input(slices, mask, sparse_ind)
    preds_norm = autoregress_sparse_fno(
        model, inp[t0_idx], sparse_ind, t0_seconds, device,
        resparsify=resparsify,
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


def plot_per_scenario(results, out_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    names = [r["name"].replace("sim_", "") for r in results]
    ious = [r.get("iou_step6", 0) for r in results]
    bars = ax.bar(names, ious, color=["tab:green" if v >= 0.7 else "tab:orange" for v in ious])
    ax.set_ylabel("IoU at t₀ + 60s")
    ax.set_title("Sparse-input FNO (6-channel) — per-scenario IoU @ t₀=120s+60s")
    ax.axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    ax.tick_params(axis="x", rotation=45)
    for b, v in zip(bars, ious):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_layer_stack(results, out_path):
    """L2 ideal + L4d/e/f + Sparse FNO 통합 비교."""
    PREV = {
        "L2 (Full SLCF\nideal)":              0.92,
        "L4d (39 sensor\n+ geodesic ConvLSTM)": 0.21,
        "L4d (39 sensor\n+ geodesic FNO no-PI)": 0.43,
        "L4e (Sparse-retrain\nConvLSTM + re-sparsify)": 0.58,
    }
    if results:
        cur = float(np.mean([r["iou_step6"] for r in results if "iou_step6" in r]))
    else:
        cur = float("nan")
    PREV["L4e* (Sparse FNO\n6-channel)"] = cur
    PREV["L4f (Tier 1 GNN\nbinary)"] = 0.90

    keys = list(PREV.keys())
    vals = list(PREV.values())
    colors = ["tab:green", "tab:gray", "tab:gray", "tab:purple",
              "tab:red", "gold"]
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(keys)), vals, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel("IoU at t₀ + 60s")
    ax.set_title("Evaluation layers — Sparse FNO (6-ch) result added")
    ax.axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=Path("checkpoints/fno_sparse_v3/best.pt"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path, default=Path("configs/building.yaml"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/current/08_sparse_fno_v3"))
    parser.add_argument("--out-csv", type=Path,
                        default=Path("results/exp_sparse_fno_v3/comparison.csv"))
    parser.add_argument("--out-report", type=Path,
                        default=Path("docs/archive/auto_reports/sparse_fno_v3_evaluation.md"))
    parser.add_argument("--t0", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-resparsify", action="store_true",
                        help="Disable re-sparsify chaining (for comparison)")
    args = parser.parse_args()

    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[setup] device={device}")
    model = load_sparse_fno(args.ckpt, device)
    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)
    resparsify = not args.no_resparsify
    print(f"[setup] {len(sensor_idxs)} sensors, resparsify={resparsify}")

    scens = sorted(d for d in args.raw_root.glob("sim_*_T*") if d.is_dir())
    results = []
    for scen in scens:
        try:
            r = eval_scenario(scen, model, mask, sparse_ind, args.t0, device,
                              resparsify=resparsify)
            results.append(r)
        except Exception as e:
            print(f"[skip] {scen.name}: {e}")

    if results:
        mean_iou = float(np.mean([r.get("iou_step6", 0) for r in results]))
        mean_fnr = float(np.mean([r.get("fnr_step6", 0) for r in results]))
        mean_rmse = float(np.mean([r.get("rmse_step6", 0) for r in results]))
        print(f"\n[agg]")
        print(f"  Mean IoU @ +60s: {mean_iou:.3f}")
        print(f"  Mean FNR @ +60s: {mean_fnr*100:.1f}%")
        print(f"  Mean RMSE step 6: {mean_rmse:.3f}")

    print("[plot] per_scenario.png")
    plot_per_scenario(results, args.out_figures / "per_scenario.png")
    print("[plot] full_stack_comparison.png")
    plot_layer_stack(results, args.out_figures / "full_stack_comparison.png")

    cols = ["name", "loc", "hrr_kw", "area_m2", "t0",
            "iou_step1", "iou_step6", "iou_all", "rmse_step6", "fnr_step6"]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, "") for c in cols])

    lines = [
        "# Sparse-input FNO (6-channel) 평가\n",
        f"> Checkpoint: `{args.ckpt}`",
        f"> t0 = {args.t0:.0f}s, lookahead 60s, 39 sensors",
        f"> re-sparsify chaining: {resparsify}\n",
        "## 1. 평균 결과 (13 OOD)\n",
        f"- Mean IoU @ +60s: **{mean_iou:.3f}**",
        f"- Mean FNR @ +60s: {mean_fnr*100:.1f}%",
        f"- Mean RMSE: {mean_rmse:.3f}",
        "",
        "## 2. 시나리오별 결과\n",
        "| 시나리오 | HRR | area | IoU step 1 | IoU step 6 | RMSE step 6 | FNR step 6 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['hrr_kw']} kW | {r['area_m2']} m² | "
            f"{r.get('iou_step1', 0):.3f} | {r.get('iou_step6', 0):.3f} | "
            f"{r.get('rmse_step6', 0):.3f} | {r.get('fnr_step6', 0)*100:.1f}% |"
        )
    args.out_report.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[PASS] outputs in {args.out_figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
