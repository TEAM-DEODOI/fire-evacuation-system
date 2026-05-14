"""Track 1A 검증 — Sparse intelligent sensors → interpolation → model.

본 시스템의 실 deployment 시나리오 검증:
1. 건물 16개 노드 (has_detector=True) 에 T/V/CO 감지기 설치 가정
2. z=1.75m (호흡고도) 에서 sparse measurement 추출 (16 points × 3 quantities)
3. scipy.interpolate.griddata 로 (60, 40) 평면 보간
4. z 축은 단순 broadcast (1.75m 측정값을 모든 z 에 복제)
5. 복원된 (5, 60, 40, 6) input → ConvLSTM/FNO autoregress → 60s 미래 예측
6. FDS truth 와 IoU/RMSE 비교

비교 baseline:
* Ideal: 진짜 SLCF input 사용 (= 기존 evaluate_t_locations.py 결과)
* Sparse-linear: 16 점 + linear interpolation
* Sparse-cubic:  16 점 + cubic interpolation
* Sparse-nearest: 16 점 + nearest-neighbor (baseline 하한)

출력 시각화: figures/sparse_sensing/
- per-scenario IoU bar chart
- 보간 quality 자체 (보간된 T 와 FDS T 의 RMSE) 도 같이 측정
- "interpolation error vs model output error" 분해

운용 시나리오: 화재가 충분히 진행된 t₀=120s 부터 시작 (cold-start 회피).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.interpolate import griddata

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import (
    normalize_co, normalize_temperature, normalize_visibility,
)
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import (
    CELL_SIZE_M, DT_SLCF, GRID_SHAPE, N_TIMESTEPS, T_END_SECONDS,
)
from src.shared.coordinates import cell_centres
from evaluate_t_locations import load_model, load_mask

SCEN_RE = re.compile(r"^sim_(?P<hrr>\d+)kw_(?P<area>\d+)m2_(?P<loc>T\d{2})$")
Z_BREATHING_M = 1.75
LOOKAHEAD_STEPS = 6


# ─── Sensor positions ──────────────────────────────────────────────────────
def load_sensor_positions(building_yaml: Path) -> List[Tuple[float, float]]:
    """16 detector 노드의 (x, y) 좌표만 반환 (z 는 무시)."""
    with building_yaml.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [(n["pos"][0], n["pos"][1])
            for n in cfg["nodes"] if n.get("has_detector")]


# ─── Sparse measurement + interpolation ────────────────────────────────────
def sparse_to_dense_frame(
    full_field: np.ndarray,   # (X, Y, Z) at one timestep
    sensor_xy: List[Tuple[float, float]],
    method: str = "linear",
) -> np.ndarray:
    """16 점 sample → (X, Y) 보간 → 모든 z 에 broadcast.

    Args:
        full_field: FDS truth (60, 40, 6), 한 frame 한 quantity (raw 물리 단위).
        sensor_xy: List of (x, y) world coordinates.
        method: scipy.griddata method — "linear", "cubic", "nearest".

    Returns:
        (60, 40, 6) 보간된 dense field. z 는 broadcast.
    """
    x_centres, y_centres, z_centres = cell_centres()  # (60,), (40,), (6,)
    # z=1.75m 에 가장 가까운 z index
    z_idx_breathing = int(np.argmin(np.abs(z_centres - Z_BREATHING_M)))

    # 1) Sample at sensor locations (breathing-z)
    sensor_values = []
    for (sx, sy) in sensor_xy:
        # Find nearest cell index
        ix = int(np.argmin(np.abs(x_centres - sx)))
        iy = int(np.argmin(np.abs(y_centres - sy)))
        sensor_values.append(full_field[ix, iy, z_idx_breathing])
    sensor_values = np.asarray(sensor_values, dtype=np.float32)

    # 2) Interpolate to 60×40 plane
    points = np.asarray(sensor_xy, dtype=np.float32)  # (16, 2)
    grid_x, grid_y = np.meshgrid(x_centres, y_centres, indexing="ij")  # (60, 40)
    interpolated_2d = griddata(
        points, sensor_values, (grid_x, grid_y),
        method=method, fill_value=sensor_values.mean(),
    )

    # 3) Broadcast to z (호흡고도 측정으로 가정 → 모든 z layer 에 복제)
    dense_3d = np.broadcast_to(interpolated_2d[:, :, None], GRID_SHAPE).astype(np.float32)
    return dense_3d


def sparse_to_dense_full_scenario(
    slices: Dict[str, np.ndarray],
    sensor_xy: List[Tuple[float, float]],
    method: str,
) -> Dict[str, np.ndarray]:
    """31 frame 모두에 대해 sparse → dense.

    Returns:
        {"temperature", "visibility", "co"} 각각 (31, 60, 40, 6) raw 단위.
    """
    out = {}
    for key in ("temperature", "visibility", "co"):
        full = slices[key]   # (31, 60, 40, 6)
        out[key] = np.stack([
            sparse_to_dense_frame(full[t], sensor_xy, method)
            for t in range(N_TIMESTEPS)
        ], axis=0).astype(np.float32)
    return out


def build_input_from_dense(
    dense_raw: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    """Dense raw fields → normalised (31, 5, 60, 40, 6) input tensor.

    `src.data_pipeline.normalize.build_input_tensor` 와 같은 결과를 생성.
    """
    T = normalize_temperature(dense_raw["temperature"]).astype(np.float32)
    V = normalize_visibility(dense_raw["visibility"]).astype(np.float32)
    CO = normalize_co(dense_raw["co"]).astype(np.float32)

    times = np.arange(N_TIMESTEPS) * DT_SLCF
    te = (times / T_END_SECONDS).astype(np.float32)  # (31,)

    expected = (N_TIMESTEPS, *GRID_SHAPE)
    mask_b = np.broadcast_to(mask.astype(np.float32)[None, :, :, :], expected).astype(np.float32)
    te_grid = np.broadcast_to(te[:, None, None, None], expected).astype(np.float32)

    return np.stack([T, V, CO, mask_b, te_grid], axis=1).astype(np.float32)


# ─── Autoregress ──────────────────────────────────────────────────────────
def autoregress(model: torch.nn.Module, initial_input: np.ndarray,
                t0_seconds: float, device: torch.device,
                n_steps: int = LOOKAHEAD_STEPS) -> np.ndarray:
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


# ─── Metrics ──────────────────────────────────────────────────────────────
def iou_rmse(pred_d: np.ndarray, true_d: np.ndarray, mask: np.ndarray,
              threshold: float = 0.5) -> Dict[str, float]:
    fluid = (mask > 0.5)
    fm = np.broadcast_to(fluid, pred_d.shape)
    p = (pred_d >= threshold)
    t = (true_d >= threshold)
    tp = float(np.sum(p & t & fm))
    fp = float(np.sum(p & (~t) & fm))
    fn = float(np.sum((~p) & t & fm))
    tn = float(np.sum((~p) & (~t) & fm))
    return {
        "iou": tp / (tp + fp + fn + 1e-9),
        "fnr": fn / (fn + tp + 1e-9),
        "rmse": float(np.sqrt(np.mean(
            (pred_d - true_d).astype(np.float64)[fm.reshape(pred_d.shape)] ** 2
        ))),
    }


def interp_quality(true_field: np.ndarray, interp_field: np.ndarray,
                    mask: np.ndarray) -> float:
    """보간된 field 와 진짜 field 의 RMSE (raw 단위, fluid cells)."""
    fluid = (mask > 0.5)
    diff = (true_field - interp_field)[..., fluid]
    return float(np.sqrt(np.mean(diff ** 2)))


# ─── Per-scenario ─────────────────────────────────────────────────────────
def eval_scenario(scen_dir: Path, models: Dict[str, torch.nn.Module],
                   mask: np.ndarray, sensor_xy: List[Tuple[float, float]],
                   methods: List[str], t0_seconds: float,
                   device: torch.device) -> Dict[str, Any]:
    name = scen_dir.name
    m = SCEN_RE.match(name)
    meta = {"name": name, "loc": m.group("loc"),
            "hrr_kw": int(m.group("hrr")), "area_m2": int(m.group("area")),
            "t0": t0_seconds}

    slices = extract_slices(scen_dir)
    truth_danger = compute_total_danger(
        slices["temperature"], slices["visibility"], slices["co"],
    ).astype(np.float32)

    t0_idx = int(t0_seconds // DT_SLCF)
    if t0_idx + LOOKAHEAD_STEPS >= N_TIMESTEPS:
        print(f"[skip] {name}: t0 too late")
        return meta
    truth_window = truth_danger[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

    print(f"[scen] {name}")
    # Each interpolation method
    for method in methods:
        dense = sparse_to_dense_full_scenario(slices, sensor_xy, method)
        # Interpolation quality at t0
        meta[f"interp_{method}_rmse_T"]  = interp_quality(slices["temperature"][t0_idx],
                                                          dense["temperature"][t0_idx], mask)
        meta[f"interp_{method}_rmse_V"]  = interp_quality(slices["visibility"][t0_idx],
                                                          dense["visibility"][t0_idx], mask)
        meta[f"interp_{method}_rmse_CO"] = interp_quality(slices["co"][t0_idx],
                                                          dense["co"][t0_idx], mask)
        # Build input tensor
        inp = build_input_from_dense(dense, mask)
        # Model autoregress
        for mname, model in models.items():
            preds_norm = autoregress(model, inp[t0_idx], t0_seconds, device)
            times_arr = np.array([t0_seconds + (s + 1) * DT_SLCF
                                   for s in range(LOOKAHEAD_STEPS)])
            preds_danger = prediction_to_danger(preds_norm, times_arr)
            m_step6 = iou_rmse(preds_danger[5:6], truth_window[5:6], mask)
            m_all = iou_rmse(preds_danger, truth_window, mask)
            meta[f"{method}__{mname}_iou_step6"]  = m_step6["iou"]
            meta[f"{method}__{mname}_iou_all"]    = m_all["iou"]
            meta[f"{method}__{mname}_rmse_step6"] = m_step6["rmse"]
            meta[f"{method}__{mname}_fnr_step6"]  = m_step6["fnr"]
    return meta


# ─── Plots ────────────────────────────────────────────────────────────────
def plot_method_comparison(results: List[Dict[str, Any]], methods: List[str],
                            models: List[str], out_path: Path) -> None:
    """각 방법별 평균 IoU step 6 막대 (모델별 그룹)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    width = 0.20
    x = np.arange(len(models))
    palette = {"linear": "tab:blue", "cubic": "tab:orange", "nearest": "tab:green"}
    for i, method in enumerate(methods):
        vals_iou = []
        vals_rmse = []
        for mname in models:
            key = f"{method}__{mname}_iou_step6"
            scen_vals = [r[key] for r in results if key in r and not np.isnan(r[key])]
            vals_iou.append(np.mean(scen_vals) if scen_vals else 0)
            key_r = f"{method}__{mname}_rmse_step6"
            scen_vals_r = [r[key_r] for r in results if key_r in r and not np.isnan(r[key_r])]
            vals_rmse.append(np.mean(scen_vals_r) if scen_vals_r else 0)
        axes[0].bar(x + (i - 1) * width, vals_iou, width,
                    label=f"sparse-{method}", color=palette.get(method, "gray"))
        axes[1].bar(x + (i - 1) * width, vals_rmse, width,
                    label=f"sparse-{method}", color=palette.get(method, "gray"))
    axes[0].set_xticks(x); axes[0].set_xticklabels(models)
    axes[0].set_ylabel("IoU at t₀+60s")
    axes[0].set_title("Sparse sensing → 60s prediction IoU\n(16 sensors, t₀ = 120s)")
    axes[0].axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    axes[0].grid(alpha=0.3, axis="y"); axes[0].legend(fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Risk-map RMSE at t₀+60s")
    axes[1].set_title("Sparse sensing → 60s prediction RMSE")
    axes[1].grid(alpha=0.3, axis="y"); axes[1].legend(fontsize=8)
    fig.suptitle("Track 1A — Sparse intelligent sensors + interpolation + model",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_interp_quality(results: List[Dict[str, Any]], methods: List[str],
                         out_path: Path) -> None:
    """보간 자체의 quality — RMSE in raw physical units."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    quantities = [("T", "°C"), ("V", "m"), ("CO", "ppm")]
    palette = {"linear": "tab:blue", "cubic": "tab:orange", "nearest": "tab:green"}
    for ax, (q, unit) in zip(axes, quantities):
        for method in methods:
            vals = [r[f"interp_{method}_rmse_{q}"] for r in results
                    if f"interp_{method}_rmse_{q}" in r]
            ax.hist(vals, bins=8, alpha=0.6, label=method, color=palette.get(method))
        ax.set_xlabel(f"Interp RMSE ({unit})")
        ax.set_ylabel("scenarios")
        ax.set_title(f"{q} — sparse-to-dense interpolation error")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("16-sensor interpolation quality (raw physical units, t₀=120s frame)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Snapshot figure (선택 시나리오 시각화) ─────────────────────────────────
def plot_sparse_snapshot(scen_dir: Path, sensor_xy: List[Tuple[float, float]],
                          models: Dict[str, torch.nn.Module], mask: np.ndarray,
                          t0_seconds: float, method: str, out_path: Path,
                          device: torch.device) -> None:
    """1 시나리오의 t₀+60s 시점 비교: FDS truth | sparse-interp-input | model pred."""
    slices = extract_slices(scen_dir)
    truth_danger = compute_total_danger(
        slices["temperature"], slices["visibility"], slices["co"],
    ).astype(np.float32)
    t0_idx = int(t0_seconds // DT_SLCF)
    truth_window = truth_danger[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

    dense = sparse_to_dense_full_scenario(slices, sensor_xy, method)
    inp = build_input_from_dense(dense, mask)

    z = 3  # z=1.75m
    n_cols = 4  # truth | interp-input @ t₀ | 3 model preds @ t₀+60s
    n_rows = 1
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Top row: t₀ — interpolated input vs FDS truth (temperature normalised)
    truth_T_t0 = normalize_temperature(slices["temperature"][t0_idx])
    interp_T_t0 = normalize_temperature(dense["temperature"][t0_idx])
    im00 = axes[0, 0].imshow(truth_T_t0[:, :, z].T, origin="lower", cmap="RdYlGn_r",
                              vmin=0, vmax=1, extent=[0, 30, 0, 20], aspect="equal")
    axes[0, 0].set_title(f"FDS truth T (normalised) @ t₀={t0_seconds:.0f}s", fontsize=10)
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.04)
    # Plot sensor positions
    for sx, sy in sensor_xy:
        axes[0, 0].plot(sx, sy, "ko", ms=5)
    im01 = axes[0, 1].imshow(interp_T_t0[:, :, z].T, origin="lower", cmap="RdYlGn_r",
                              vmin=0, vmax=1, extent=[0, 30, 0, 20], aspect="equal")
    axes[0, 1].set_title(f"sparse-{method} interpolated T @ t₀", fontsize=10)
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.04)
    for sx, sy in sensor_xy:
        axes[0, 1].plot(sx, sy, "ko", ms=5)
    err_T = np.abs(truth_T_t0[:, :, z] - interp_T_t0[:, :, z]).T
    im02 = axes[0, 2].imshow(err_T, origin="lower", cmap="magma",
                              vmin=0, vmax=max(0.1, float(err_T.max())),
                              extent=[0, 30, 0, 20], aspect="equal")
    axes[0, 2].set_title(f"|T interp error| @ t₀", fontsize=10)
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.04)

    # Bottom row: at t₀+60s — FDS truth danger | 3 model preds
    truth_d_60 = truth_window[5]
    im10 = axes[1, 0].imshow(truth_d_60[:, :, z].T, origin="lower", cmap="RdYlGn_r",
                              vmin=0, vmax=1, extent=[0, 30, 0, 20], aspect="equal")
    axes[1, 0].set_title(f"FDS truth danger @ t₀+60s", fontsize=10)
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.04)

    model_names = list(models.keys())
    for i, mname in enumerate(model_names[:2], start=1):
        preds_norm = autoregress(models[mname], inp[t0_idx], t0_seconds, device)
        times_arr = np.array([t0_seconds + (s + 1) * DT_SLCF for s in range(LOOKAHEAD_STEPS)])
        preds_d = prediction_to_danger(preds_norm, times_arr)
        ax = axes[1, i]
        im = ax.imshow(preds_d[5][:, :, z].T, origin="lower", cmap="RdYlGn_r",
                        vmin=0, vmax=1, extent=[0, 30, 0, 20], aspect="equal")
        ax.set_title(f"{mname} pred (sparse-{method}) @ t₀+60s", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"Track 1A snapshot — {scen_dir.name}, 16 sensors, {method} interpolation",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── CSV + markdown ───────────────────────────────────────────────────────
def write_csv(results: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    # Union of keys
    all_keys = sorted({k for r in results for k in r.keys()})
    base_keys = ["name", "loc", "hrr_kw", "area_m2", "t0"]
    other_keys = [k for k in all_keys if k not in base_keys]
    cols = base_keys + other_keys
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, "") for c in cols])


def write_report(results: List[Dict[str, Any]], methods: List[str],
                 model_names: List[str], fig_dir: Path,
                 out_path: Path) -> None:
    valid_results = [r for r in results if any(k.startswith("linear__") for k in r)]
    if not valid_results:
        return
    lines = []
    lines.append("# Track 1A — Sparse Intelligent Sensors 검증\n")
    lines.append("> **설정**: 16개 지능형 센서 (T/CO/V 감지) → spatial interpolation → "
                 "기존 모델 입력 → 60s 미래 예측.")
    lines.append(f"> **시작 시점**: t₀ = {valid_results[0]['t0']:.0f} s (mid-fire, cold-start 회피)")
    lines.append("> **건물**: 16 detector 노드 = `configs/building.yaml has_detector=True`\n")

    # Aggregate
    lines.append("\n## 1. 평균 결과 표\n")
    lines.append("| 방법 | 모델 | IoU step 6 | RMSE step 6 | FNR step 6 |")
    lines.append("|---|---|---|---|---|")
    summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        for mname in model_names:
            key_iou = f"{method}__{mname}_iou_step6"
            key_rmse = f"{method}__{mname}_rmse_step6"
            key_fnr = f"{method}__{mname}_fnr_step6"
            ious = [r[key_iou] for r in valid_results if key_iou in r]
            rmses = [r[key_rmse] for r in valid_results if key_rmse in r]
            fnrs = [r[key_fnr] for r in valid_results if key_fnr in r]
            if ious:
                m_iou = float(np.mean(ious))
                m_rmse = float(np.mean(rmses))
                m_fnr = float(np.mean(fnrs))
                summary[f"{method}__{mname}"] = {"iou": m_iou, "rmse": m_rmse, "fnr": m_fnr}
                lines.append(f"| sparse-{method} | {mname} | {m_iou:.3f} | "
                             f"{m_rmse:.3f} | {m_fnr*100:.1f}% |")

    # Interpolation quality
    lines.append("\n## 2. 보간 자체의 quality (raw 단위, t₀ frame)\n")
    lines.append("| 방법 | T RMSE (°C) | V RMSE (m) | CO RMSE (ppm) |")
    lines.append("|---|---|---|---|")
    for method in methods:
        vals_T = [r[f"interp_{method}_rmse_T"]  for r in valid_results
                  if f"interp_{method}_rmse_T" in r]
        vals_V = [r[f"interp_{method}_rmse_V"]  for r in valid_results
                  if f"interp_{method}_rmse_V" in r]
        vals_CO = [r[f"interp_{method}_rmse_CO"] for r in valid_results
                   if f"interp_{method}_rmse_CO" in r]
        if vals_T:
            lines.append(f"| sparse-{method} | {np.mean(vals_T):.2f} | "
                         f"{np.mean(vals_V):.2f} | {np.mean(vals_CO):.1f} |")

    lines.append("\n## 3. Figures\n")
    lines.append(f"![Method comparison]({(fig_dir / 'method_comparison.png').as_posix()})\n")
    lines.append(f"![Interp quality]({(fig_dir / 'interp_quality.png').as_posix()})\n")
    lines.append(f"![Snapshot]({(fig_dir / 'snapshot_T05_1500kw_linear.png').as_posix()})\n")

    # Compare with previous "ideal" results
    lines.append("\n## 4. ★ Ideal vs Sparse 비교 (핵심 결과)\n")
    lines.append(
        "이전 평가 (full SLCF input, t₀=120s) 결과 (`docs/cold_start_finding.md` §3):\n\n"
        "| 모델 | Ideal IoU step 6 (t₀=120s) |\n"
        "|---|---|\n"
        "| ConvLSTM | 0.92 |\n"
        "| FNO no-PI | 0.82 |\n"
        "| FNO PI | 0.89 |\n"
    )
    lines.append("\nTrack 1A (16 sensors, linear interp) 결과 — 위 표 §1 참조.\n")

    # Conclusion
    lines.append("\n## 5. 해석 / 결론\n")
    best = max(summary.items(), key=lambda kv: kv[1]["iou"]) if summary else None
    if best:
        bname, bv = best
        lines.append(
            f"- 최고 조합: **{bname}** (IoU {bv['iou']:.3f})  \n"
            f"- 이상적 full-SLCF 대비 손실: ConvLSTM 0.92 → "
            f"sparse {summary.get('linear__ConvLSTM', {'iou': 0})['iou']:.3f}  \n"
            f"- 16-sensor sparse 의 핵심 제약: spatial detail 손실. 보간이 부드러워 "
            f"실제 corridor 의 sharp 패턴 재현 못함.  \n"
        )
    lines.append(
        "\n### 권고\n"
        "- 16-sensor 결과가 H5 (≥ 0.70) 통과 못하면: sensor 개수 증가 ablation 필요\n"
        "- 또는 sparse-aware 모델 학습 (Track 1B) 검토\n"
        "- 또는 paper 에서 'high-end deployment 의 lower bound' 로 결과 명시\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path, default=Path("configs/building.yaml"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/sparse_sensing"))
    parser.add_argument("--out-csv", type=Path,
                        default=Path("results/exp_sparse_sensing/comparison.csv"))
    parser.add_argument("--out-report", type=Path,
                        default=Path("docs/sparse_sensing_evaluation.md"))
    parser.add_argument("--t0", type=float, default=120.0,
                        help="autoregress 시작 시점 (초, default mid-fire)")
    parser.add_argument("--methods", nargs="+",
                        default=["linear", "cubic", "nearest"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[setup] device={device}")
    models = {
        "ConvLSTM":  load_model(Path("checkpoints/conv_lstm/best.pt"),  device, "conv_lstm"),
        "FNO no-PI": load_model(Path("checkpoints/fno_no_pi/best.pt"),  device, "fno"),
        "FNO PI":    load_model(Path("checkpoints/fno_pi/best.pt"),     device, "fno"),
    }
    mask = load_mask(args.dataset)
    sensors = load_sensor_positions(args.building)
    print(f"[setup] {len(sensors)} sensors, methods={args.methods}, t0={args.t0}s\n")

    scens = sorted(d for d in args.raw_root.glob("sim_*_T*") if d.is_dir())
    results = []
    for scen in scens:
        try:
            r = eval_scenario(scen, models, mask, sensors, args.methods,
                              args.t0, device)
            results.append(r)
        except Exception as e:
            print(f"[skip] {scen.name}: {e}")

    # Per-method comparison figure
    print("[plot] method_comparison.png")
    plot_method_comparison(results, args.methods, list(models.keys()),
                           args.out_figures / "method_comparison.png")
    print("[plot] interp_quality.png")
    plot_interp_quality(results, args.methods,
                         args.out_figures / "interp_quality.png")
    # Snapshot for one scenario
    snap_scen = args.raw_root / "sim_1500kw_2m2_T05"
    if snap_scen.is_dir():
        print("[plot] snapshot_T05_1500kw_linear.png")
        plot_sparse_snapshot(snap_scen, sensors, models, mask, args.t0,
                              "linear",
                              args.out_figures / "snapshot_T05_1500kw_linear.png",
                              device)

    print("[csv]  comparison.csv")
    write_csv(results, args.out_csv)
    print("[doc]  sparse_sensing_evaluation.md")
    write_report(results, args.methods, list(models.keys()),
                 args.out_figures, args.out_report)
    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
