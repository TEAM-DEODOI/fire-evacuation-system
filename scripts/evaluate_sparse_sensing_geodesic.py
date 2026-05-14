"""Track 1A 확장 — Mask-aware geodesic interpolation.

기존 `evaluate_sparse_sensing.py` 의 단순 griddata (Euclidean) 와 달리,
**벽을 우회하는 geodesic 거리** 로 보간 가중치를 계산. 같은 방의 sensor 가
다른 방의 sensor 보다 더 강한 영향을 미치도록.

알고리즘:
1. 각 sensor 위치에서 BFS — fluid cell 만 따라 거리 누적 (벽을 건너지 못함)
2. 모든 fluid cell 에 대해 각 sensor 까지의 geodesic 거리 행렬 D ∈ ℝ^(N_fluid, 16)
3. IDW: weight[c, s] = 1 / (D[c, s]^p + eps), p=2
4. interpolated value[c] = Σ weight[c, s] · value[s] / Σ weight[c, s]
5. Solid cell 은 NaN 또는 0 으로 마킹

비교:
* baseline: scipy.griddata (linear, Euclidean) — 기존 결과
* proposed: geodesic IDW (mask-aware)

산출물:
- figures/sparse_sensing_geodesic/method_comparison.png
- figures/sparse_sensing_geodesic/geodesic_vs_linear_snapshot.png
- results/exp_sparse_sensing_geodesic/comparison.csv
- docs/sparse_sensing_geodesic_evaluation.md
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from src.shared.constants import CELL_SIZE_M, DT_SLCF, GRID_SHAPE, N_TIMESTEPS, T_END_SECONDS
from src.shared.coordinates import cell_centres
from evaluate_t_locations import load_model, load_mask

SCEN_RE = re.compile(r"^sim_(?P<hrr>\d+)kw_(?P<area>\d+)m2_(?P<loc>T\d{2})$")
Z_BREATHING_M = 1.75
LOOKAHEAD_STEPS = 6


# ─── Sensor positions ──────────────────────────────────────────────────────
def load_sensor_positions(
    building_yaml: Path, source: str = "d024",
) -> List[Tuple[float, float]]:
    """Load sensor (x, y) positions.

    Args:
        building_yaml: Path to ``configs/building.yaml`` (used only when
            ``source="building_yaml"``).
        source: ``"d024"`` (27 detectors from D-024) or ``"building_yaml"``
            (legacy 16 detectors with ``has_detector=True``).
    """
    if source == "d024":
        from src.tier1.detector_positions import ALL_DETECTORS
        return [(d.position[0], d.position[1]) for d in ALL_DETECTORS]
    elif source == "building_yaml":
        with building_yaml.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return [(n["pos"][0], n["pos"][1])
                for n in cfg["nodes"] if n.get("has_detector")]
    else:
        raise ValueError(f"unknown source: {source!r}")


def world_to_xy_idx(xy: Tuple[float, float]) -> Tuple[int, int]:
    x_c, y_c, _ = cell_centres()
    ix = int(np.argmin(np.abs(x_c - xy[0])))
    iy = int(np.argmin(np.abs(y_c - xy[1])))
    return ix, iy


# ─── Geodesic distance via BFS (mask-aware) ────────────────────────────────
def bfs_geodesic_distance(
    fluid_mask_2d: np.ndarray,    # (60, 40) bool — fluid=True
    source: Tuple[int, int],
) -> np.ndarray:
    """BFS — source 에서 모든 fluid cell 까지의 step 수.

    Args:
        fluid_mask_2d: (60, 40) bool. True = traversable.
        source: (ix, iy) starting cell.

    Returns:
        (60, 40) float — geodesic distance in *cell steps* (∞ for unreachable
        or solid cells).
    """
    nx, ny = fluid_mask_2d.shape
    dist = np.full((nx, ny), np.inf, dtype=np.float32)
    sx, sy = source
    if not fluid_mask_2d[sx, sy]:
        # sensor 위치가 solid 면, 가까운 fluid 로 snap
        # (실 운용에선 발생 안 해야 하지만 안전장치)
        for r in range(1, 5):
            best = None
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx_, ny_ = sx + dx, sy + dy
                    if 0 <= nx_ < nx and 0 <= ny_ < ny and fluid_mask_2d[nx_, ny_]:
                        if best is None or dx * dx + dy * dy < best[0]:
                            best = (dx * dx + dy * dy, nx_, ny_)
            if best is not None:
                sx, sy = best[1], best[2]
                break
        else:
            return dist  # all infinity
    dist[sx, sy] = 0.0
    q = deque([(sx, sy)])
    # 4-connectivity (벽 corner 통과 방지)
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        x, y = q.popleft()
        d = dist[x, y]
        for dx, dy in neighbours:
            nxn, nyn = x + dx, y + dy
            if 0 <= nxn < nx and 0 <= nyn < ny and fluid_mask_2d[nxn, nyn]:
                if dist[nxn, nyn] > d + 1:
                    dist[nxn, nyn] = d + 1
                    q.append((nxn, nyn))
    return dist


def precompute_geodesic_distances(
    mask: np.ndarray,
    sensor_xy: List[Tuple[float, float]],
    z_idx: int,
) -> np.ndarray:
    """모든 sensor 의 geodesic distance map 을 stack.

    Returns:
        (N_sensors, 60, 40) float — sensor s 에서 (ix, iy) 까지 거리.
    """
    fluid_2d = (mask[:, :, z_idx] > 0.5)
    dists = []
    for xy in sensor_xy:
        ix, iy = world_to_xy_idx(xy)
        d = bfs_geodesic_distance(fluid_2d, (ix, iy))
        dists.append(d)
    return np.stack(dists, axis=0)


def geodesic_idw_interpolate(
    sensor_values: np.ndarray,      # (N_sensors,) scalar 값
    geo_dist: np.ndarray,            # (N_sensors, nx, ny) geodesic distance
    p: float = 2.0,
    eps: float = 0.5,
) -> np.ndarray:
    """Geodesic IDW — sensor value 가중평균 (벽 우회 거리 기반).

    Returns:
        (nx, ny) float — interpolated values. Unreachable cell 은 NaN.
    """
    # weight = 1 / (d^p + eps)
    weights = 1.0 / (geo_dist ** p + eps)
    weights[~np.isfinite(geo_dist)] = 0.0   # unreachable cell 가중치 0
    # weighted sum
    num = np.sum(weights * sensor_values[:, None, None], axis=0)
    den = np.sum(weights, axis=0)
    out = np.full_like(num, np.nan, dtype=np.float32)
    valid = den > 0
    out[valid] = (num[valid] / den[valid]).astype(np.float32)
    return out


# ─── Sparse-to-dense per frame (geodesic) ──────────────────────────────────
def sparse_to_dense_frame_geodesic(
    full_field: np.ndarray,            # (60, 40, 6) raw
    sensor_xy: List[Tuple[float, float]],
    geo_dist: np.ndarray,               # (16, 60, 40)
    z_idx_breathing: int,
) -> np.ndarray:
    """1 frame, 1 quantity sparse → dense geodesic IDW."""
    # 1) sample at sensor locations (breathing-z)
    sensor_values = []
    for xy in sensor_xy:
        ix, iy = world_to_xy_idx(xy)
        sensor_values.append(full_field[ix, iy, z_idx_breathing])
    sensor_values = np.asarray(sensor_values, dtype=np.float32)
    # 2) geodesic IDW
    interp_2d = geodesic_idw_interpolate(sensor_values, geo_dist)
    # 3) NaN → mean fallback
    if np.any(np.isnan(interp_2d)):
        fill = float(np.nanmean(interp_2d))
        interp_2d = np.where(np.isnan(interp_2d), fill, interp_2d)
    # 4) z 축 broadcast
    return np.broadcast_to(interp_2d[:, :, None], GRID_SHAPE).astype(np.float32)


def sparse_to_dense_full_geodesic(
    slices: Dict[str, np.ndarray],
    sensor_xy: List[Tuple[float, float]],
    geo_dist: np.ndarray,
    z_idx_breathing: int,
) -> Dict[str, np.ndarray]:
    out = {}
    for key in ("temperature", "visibility", "co"):
        full = slices[key]
        out[key] = np.stack([
            sparse_to_dense_frame_geodesic(full[t], sensor_xy, geo_dist,
                                            z_idx_breathing)
            for t in range(N_TIMESTEPS)
        ], axis=0).astype(np.float32)
    return out


def sparse_to_dense_full_linear(
    slices: Dict[str, np.ndarray],
    sensor_xy: List[Tuple[float, float]],
    z_idx_breathing: int,
) -> Dict[str, np.ndarray]:
    """Baseline — scipy.griddata linear (Euclidean, mask-ignorant)."""
    x_c, y_c, _ = cell_centres()
    points = np.asarray(sensor_xy, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_c, y_c, indexing="ij")
    out = {}
    for key in ("temperature", "visibility", "co"):
        full = slices[key]
        frames = []
        for t in range(N_TIMESTEPS):
            vals = []
            for xy in sensor_xy:
                ix = int(np.argmin(np.abs(x_c - xy[0])))
                iy = int(np.argmin(np.abs(y_c - xy[1])))
                vals.append(full[t, ix, iy, z_idx_breathing])
            vals = np.asarray(vals, dtype=np.float32)
            interp_2d = griddata(points, vals, (grid_x, grid_y),
                                 method="linear", fill_value=float(vals.mean()))
            frames.append(np.broadcast_to(interp_2d[:, :, None], GRID_SHAPE).astype(np.float32))
        out[key] = np.stack(frames, axis=0)
    return out


def build_input_from_dense(dense_raw: Dict[str, np.ndarray],
                            mask: np.ndarray) -> np.ndarray:
    T = normalize_temperature(dense_raw["temperature"]).astype(np.float32)
    V = normalize_visibility(dense_raw["visibility"]).astype(np.float32)
    CO = normalize_co(dense_raw["co"]).astype(np.float32)
    times = np.arange(N_TIMESTEPS) * DT_SLCF
    te = (times / T_END_SECONDS).astype(np.float32)
    expected = (N_TIMESTEPS, *GRID_SHAPE)
    mask_b = np.broadcast_to(mask.astype(np.float32)[None, :, :, :], expected).astype(np.float32)
    te_grid = np.broadcast_to(te[:, None, None, None], expected).astype(np.float32)
    return np.stack([T, V, CO, mask_b, te_grid], axis=1).astype(np.float32)


# ─── Autoregress / metrics ─────────────────────────────────────────────────
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


def interp_quality(true_field: np.ndarray, interp_field: np.ndarray,
                    mask: np.ndarray) -> float:
    fluid = (mask > 0.5)
    diff = (true_field - interp_field)[..., fluid]
    return float(np.sqrt(np.mean(diff ** 2)))


# ─── Per-scenario eval ─────────────────────────────────────────────────────
def eval_scenario(scen_dir: Path, models: Dict[str, torch.nn.Module],
                   mask: np.ndarray, sensor_xy: List[Tuple[float, float]],
                   geo_dist: np.ndarray, z_idx_breathing: int,
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
    # method = "linear" (baseline) or "geodesic"
    method_fns = {
        "linear":   lambda: sparse_to_dense_full_linear(slices, sensor_xy, z_idx_breathing),
        "geodesic": lambda: sparse_to_dense_full_geodesic(slices, sensor_xy, geo_dist, z_idx_breathing),
    }
    for method, fn in method_fns.items():
        dense = fn()
        meta[f"interp_{method}_rmse_T"]  = interp_quality(slices["temperature"][t0_idx],
                                                          dense["temperature"][t0_idx], mask)
        meta[f"interp_{method}_rmse_V"]  = interp_quality(slices["visibility"][t0_idx],
                                                          dense["visibility"][t0_idx], mask)
        meta[f"interp_{method}_rmse_CO"] = interp_quality(slices["co"][t0_idx],
                                                          dense["co"][t0_idx], mask)
        inp = build_input_from_dense(dense, mask)
        for mname, model in models.items():
            preds_norm = autoregress(model, inp[t0_idx], t0_seconds, device)
            times_arr = np.array([t0_seconds + (s + 1) * DT_SLCF for s in range(LOOKAHEAD_STEPS)])
            preds_danger = prediction_to_danger(preds_norm, times_arr)
            m6 = iou_rmse(preds_danger[5:6], truth_window[5:6], mask)
            mall = iou_rmse(preds_danger, truth_window, mask)
            meta[f"{method}__{mname}_iou_step6"]  = m6["iou"]
            meta[f"{method}__{mname}_iou_all"]    = mall["iou"]
            meta[f"{method}__{mname}_rmse_step6"] = m6["rmse"]
            meta[f"{method}__{mname}_fnr_step6"]  = m6["fnr"]
    return meta


# ─── Plots ────────────────────────────────────────────────────────────────
def plot_comparison(results: List[Dict[str, Any]], models: List[str],
                     out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    width = 0.35
    x = np.arange(len(models))
    method_color = {"linear": "tab:gray", "geodesic": "tab:purple"}
    for i, method in enumerate(["linear", "geodesic"]):
        ious = []; rmses = []
        for mname in models:
            ki = f"{method}__{mname}_iou_step6"
            kr = f"{method}__{mname}_rmse_step6"
            iou_vals = [r[ki] for r in results if ki in r]
            rmse_vals = [r[kr] for r in results if kr in r]
            ious.append(np.mean(iou_vals) if iou_vals else 0)
            rmses.append(np.mean(rmse_vals) if rmse_vals else 0)
        label = f"sparse-{method}"
        axes[0].bar(x + (i - 0.5) * width, ious, width, label=label,
                    color=method_color[method])
        axes[1].bar(x + (i - 0.5) * width, rmses, width, label=label,
                    color=method_color[method])
        for j, v in enumerate(ious):
            axes[0].text(x[j] + (i - 0.5) * width, v, f"{v:.3f}",
                          ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models)
    axes[0].set_ylabel("IoU at t₀+60s")
    axes[0].set_title("Linear (Euclidean) vs Geodesic (mask-aware) IDW")
    axes[0].axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    axes[1].set_xticks(x); axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Risk-map RMSE at t₀+60s")
    axes[1].set_title("RMSE comparison")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    n_sensors_text = "n sensors"
    fig.suptitle(f"Track 1A.5 — Mask-aware geodesic interpolation\n"
                 f"t₀ = 120s, 13 OOD scenarios",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_interp_snapshot(scen_dir: Path, sensor_xy: List[Tuple[float, float]],
                          mask: np.ndarray, geo_dist: np.ndarray,
                          z_idx: int, t0_idx: int, out_path: Path) -> None:
    """Linear vs Geodesic 보간된 T 단면 비교."""
    slices = extract_slices(scen_dir)
    truth_T = normalize_temperature(slices["temperature"][t0_idx])
    dense_lin = sparse_to_dense_full_linear(slices, sensor_xy, z_idx)
    dense_geo = sparse_to_dense_full_geodesic(slices, sensor_xy, geo_dist, z_idx)
    interp_lin_T = normalize_temperature(dense_lin["temperature"][t0_idx])
    interp_geo_T = normalize_temperature(dense_geo["temperature"][t0_idx])

    z_view = z_idx
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ims = []
    for col, (img, ttl) in enumerate([
        (truth_T[:, :, z_view].T, "FDS truth T (norm)"),
        (interp_lin_T[:, :, z_view].T, "sparse-linear (Euclidean)"),
        (interp_geo_T[:, :, z_view].T, "sparse-geodesic (mask-aware)"),
    ]):
        im = axes[0, col].imshow(img, origin="lower", cmap="RdYlGn_r",
                                  vmin=0, vmax=1, extent=[0, 30, 0, 20], aspect="equal")
        axes[0, col].set_title(ttl, fontsize=11)
        plt.colorbar(im, ax=axes[0, col], fraction=0.04)
        for sx, sy in sensor_xy:
            axes[0, col].plot(sx, sy, "ko", ms=5)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        ims.append(im)

    for col, (img, ttl, vmax) in enumerate([
        (np.abs(truth_T[:, :, z_view] - interp_lin_T[:, :, z_view]).T,
         "|err linear|", 0.4),
        (np.abs(truth_T[:, :, z_view] - interp_geo_T[:, :, z_view]).T,
         "|err geodesic|", 0.4),
        (mask[:, :, z_view].T, "fluid mask (1=fluid)", 1.0),
    ]):
        im = axes[1, col].imshow(img, origin="lower",
                                  cmap="magma" if col < 2 else "gray",
                                  vmin=0, vmax=vmax,
                                  extent=[0, 30, 0, 20], aspect="equal")
        axes[1, col].set_title(ttl, fontsize=11)
        plt.colorbar(im, ax=axes[1, col], fraction=0.04)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    fig.suptitle(f"Geodesic vs Linear interpolation @ t₀={t0_idx * DT_SLCF:.0f}s  ({scen_dir.name})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Report ───────────────────────────────────────────────────────────────
def write_report(results: List[Dict[str, Any]], models: List[str],
                  fig_dir: Path, out_path: Path) -> None:
    valid = [r for r in results if any(k.startswith("linear__") for k in r)]
    if not valid:
        return
    lines = []
    lines.append("# Track 1A.5 — Mask-aware Geodesic Interpolation\n")
    lines.append("> **목적**: 단순 Euclidean 보간(`scipy.griddata`)이 벽을 무시하고 "
                 "다른 방의 sensor 값까지 사용하는 문제를 **geodesic IDW** 로 해결.\n")
    lines.append("> **알고리즘**: BFS 로 fluid cell 만 따라 distance 누적 → IDW(p=2).\n")
    lines.append("\n## 1. 평균 결과 (13 OOD 시나리오, t₀ = 120s)\n")
    lines.append("| 방법 | 모델 | IoU step 6 | RMSE step 6 | FNR step 6 |")
    lines.append("|---|---|---|---|---|")
    summary = {}
    for method in ("linear", "geodesic"):
        for mname in models:
            ki = f"{method}__{mname}_iou_step6"
            kr = f"{method}__{mname}_rmse_step6"
            kf = f"{method}__{mname}_fnr_step6"
            ious = [r[ki] for r in valid if ki in r]
            rmses = [r[kr] for r in valid if kr in r]
            fnrs  = [r[kf] for r in valid if kf in r]
            if ious:
                m_iou = float(np.mean(ious))
                m_rmse = float(np.mean(rmses))
                m_fnr = float(np.mean(fnrs))
                summary[(method, mname)] = m_iou
                tag = "sparse-linear" if method == "linear" else "**sparse-geodesic**"
                lines.append(f"| {tag} | {mname} | {m_iou:.3f} | "
                             f"{m_rmse:.3f} | {m_fnr*100:.1f}% |")

    lines.append("\n## 2. 보간 자체 quality (raw 단위, t₀=120s frame)\n")
    lines.append("| 방법 | T RMSE (°C) | V RMSE (m) | CO RMSE (ppm) |")
    lines.append("|---|---|---|---|")
    for method in ("linear", "geodesic"):
        vT  = [r[f"interp_{method}_rmse_T"]  for r in valid if f"interp_{method}_rmse_T" in r]
        vV  = [r[f"interp_{method}_rmse_V"]  for r in valid if f"interp_{method}_rmse_V" in r]
        vCO = [r[f"interp_{method}_rmse_CO"] for r in valid if f"interp_{method}_rmse_CO" in r]
        if vT:
            lines.append(f"| sparse-{method} | {np.mean(vT):.2f} | "
                         f"{np.mean(vV):.2f} | {np.mean(vCO):.1f} |")

    lines.append("\n## 3. Figures\n")
    lines.append(f"![]({(fig_dir / 'method_comparison_geodesic.png').as_posix()})\n")
    lines.append(f"![]({(fig_dir / 'snapshot_T05_geodesic.png').as_posix()})\n")

    # Improvement summary
    if summary:
        deltas = []
        for mname in models:
            if ("linear", mname) in summary and ("geodesic", mname) in summary:
                d = summary[("geodesic", mname)] - summary[("linear", mname)]
                deltas.append((mname, d))
        lines.append("\n## 4. 결론\n")
        for mname, d in deltas:
            sign = "+" if d >= 0 else ""
            lines.append(f"- {mname}: geodesic vs linear IoU 차이 = **{sign}{d:.3f}**  ")
        lines.append("")
        if any(d > 0.02 for _, d in deltas):
            lines.append("→ **Geodesic 가 명확한 개선 효과**. paper 에 mask-aware "
                         "interpolation 으로 채택 권장.")
        else:
            lines.append("→ Geodesic 의 개선은 marginal. 정보론적 bottleneck (16 sensor) "
                         "을 보간 알고리즘 만으로는 해결 불가. Track 1B (sparse-aware "
                         "모델 재학습) 필요.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(results: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    all_keys = sorted({k for r in results for k in r.keys()})
    base = ["name", "loc", "hrr_kw", "area_m2", "t0"]
    cols = base + [k for k in all_keys if k not in base]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, "") for c in cols])


# ─── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path, default=Path("configs/building.yaml"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/sparse_sensing_geodesic"))
    parser.add_argument("--out-csv", type=Path,
                        default=Path("results/exp_sparse_sensing_geodesic/comparison.csv"))
    parser.add_argument("--out-report", type=Path,
                        default=Path("docs/sparse_sensing_geodesic_evaluation.md"))
    parser.add_argument("--t0", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sensor-source", type=str, default="d024",
                        choices=["d024", "building_yaml"],
                        help="d024: 27 detectors from D-024 (default); "
                             "building_yaml: legacy 16 detectors")
    args = parser.parse_args()

    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[setup] device={device}")
    models = {
        "ConvLSTM":  load_model(Path("checkpoints/conv_lstm/best.pt"), device, "conv_lstm"),
        "FNO no-PI": load_model(Path("checkpoints/fno_no_pi/best.pt"), device, "fno"),
        "FNO PI":    load_model(Path("checkpoints/fno_pi/best.pt"), device, "fno"),
    }
    mask = load_mask(args.dataset)
    sensors = load_sensor_positions(args.building, source=args.sensor_source)
    print(f"[setup] {len(sensors)} sensors (source={args.sensor_source})")

    # Precompute geodesic distances (한 번만 — building 이 동일하므로 모든 시나리오 공유)
    _, _, z_c = cell_centres()
    z_idx_breathing = int(np.argmin(np.abs(z_c - Z_BREATHING_M)))
    print(f"[setup] computing geodesic distances at z={z_c[z_idx_breathing]:.2f}m ...")
    geo_dist = precompute_geodesic_distances(mask, sensors, z_idx_breathing)
    print(f"[setup] geo_dist shape={geo_dist.shape}, "
          f"max_finite={np.max(geo_dist[np.isfinite(geo_dist)]):.1f}, "
          f"#unreachable_cells={int(np.sum(~np.isfinite(geo_dist[0])))}")

    scens = sorted(d for d in args.raw_root.glob("sim_*_T*") if d.is_dir())
    results = []
    for scen in scens:
        try:
            r = eval_scenario(scen, models, mask, sensors, geo_dist,
                              z_idx_breathing, args.t0, device)
            results.append(r)
        except Exception as e:
            print(f"[skip] {scen.name}: {e}")

    print("[plot] method_comparison_geodesic.png")
    plot_comparison(results, list(models.keys()),
                    args.out_figures / "method_comparison_geodesic.png")
    snap_scen = args.raw_root / "sim_1500kw_2m2_T05"
    if snap_scen.is_dir():
        print("[plot] snapshot_T05_geodesic.png")
        plot_interp_snapshot(snap_scen, sensors, mask, geo_dist,
                              z_idx_breathing, int(args.t0 // DT_SLCF),
                              args.out_figures / "snapshot_T05_geodesic.png")
    print("[csv]")
    write_csv(results, args.out_csv)
    print("[doc]")
    write_report(results, list(models.keys()), args.out_figures, args.out_report)
    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
