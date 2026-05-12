"""평가용 T01-T05 OOD 시나리오 종합 평가 스크립트.

`data/raw/sim_*_T*/` 13 시나리오에 대해 ConvLSTM 의 일반화 성능을 측정한다.
훈련 데이터에 없는 새 화재 위치(T01-T05) × HRR(500/1000/1500 kW) × 면적(1/2 m²)
조합이라 OOD (Out-Of-Distribution) 평가의 성격이 있다.

수행 항목
---------
1. Single-step 정확도: RelL2 / RMSE(°C, m, ppm)  (30 pair × 13 scen = 390 추론)
2. 6-step autoregress rollout: 누적 오차 (H2 target 0.15 vs OOD 실측)
3. Risk Map IoU / FNR / FPR  (FDS truth ↔ ConvLSTM 예측, threshold=0.5)
4. Inference time (CPU/GPU 환경별 평균)
5. 시각화
   - 시나리오별 z=1.75 m T/V/CO 단면 비교 (frame 18, t=180s)
   - 시나리오별 autoregress 오차 곡선
   - 시나리오별 risk map 5-snapshot 모음 + animation GIF
   - 위치별/HRR별 RelL2 boxplot (집계)
6. 산출물
   - `results/eval_T01_T05/per_scenario_metrics.csv`
   - `results/eval_T01_T05/aggregated.csv`
   - `figures/eval_T01_T05/<scenario>/...`
   - `docs/eval_T01_T05_report.md`  ← 위치별 페이지 보고서
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import (
    build_input_tensor,
    build_target_tensor,
    normalize_scenario,
)
from src.models.conv_lstm_3d import FireConvLSTM
from src.risk_map.converter import prediction_to_danger
from src.risk_map.risk_map_class import StaticRiskMap
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS, T_END_SECONDS
from src.shared.normalization import (
    denormalize_co,
    denormalize_temperature,
    denormalize_visibility,
)

# ─── Constants ─────────────────────────────────────────────────────────────
DANGER_THRESHOLD: float = 0.5
"""IoU/FNR/FPR 평가용 위험-비위험 분류 임계값."""

Z_SLICE_IDX: int = 3
"""시각화용 z 인덱스 (≈ 1.75 m 호흡 영역)."""

SCEN_NAME_RE = re.compile(
    r"^sim_(?P<hrr>\d+)kw_(?P<area>\d+)m2_(?P<loc>T\d{2})$"
)


# ─── Model + mask loaders ──────────────────────────────────────────────────
def load_model(ckpt: Path, device: torch.device) -> FireConvLSTM:
    """ConvLSTM 체크포인트 로드 (`evaluate_convlstm.py` 와 동일 절차)."""
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
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


def load_mask(dataset_h5: Path) -> np.ndarray:
    """기존 `dataset.h5` 의 mask 재사용 (모든 시나리오 동일 건물)."""
    with h5py.File(dataset_h5, "r") as f:
        mask = np.asarray(f["mask"], dtype=np.float32)
    if mask.shape != GRID_SHAPE:
        raise ValueError(f"mask shape {mask.shape} != {GRID_SHAPE}")
    return mask


# ─── Scenario discovery + parsing ──────────────────────────────────────────
def discover_scenarios(raw_root: Path) -> List[Dict[str, Any]]:
    """`data/raw/sim_*_T*/` 디렉터리 탐색 + 메타데이터 파싱."""
    found = []
    for d in sorted(raw_root.glob("sim_*_T*")):
        if not d.is_dir():
            continue
        m = SCEN_NAME_RE.match(d.name)
        if not m:
            continue
        found.append({
            "name": d.name,
            "path": d,
            "hrr_kw": int(m.group("hrr")),
            "area_m2": int(m.group("area")),
            "loc": m.group("loc"),   # "T01" ~ "T05"
        })
    return found


# ─── Metric helpers ────────────────────────────────────────────────────────
def rel_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> float:
    """Relative L2 norm — ‖pred-target‖₂ / ‖target‖₂."""
    num = float(np.sqrt(np.mean((pred - target) ** 2)))
    den = float(np.sqrt(np.mean(target ** 2))) + eps
    return num / den


def rmse_normalized(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
    """채널별 RMSE in [0,1] domain. pred/target: (C=3, X, Y, Z)."""
    return (
        float(np.sqrt(np.mean((pred[0] - target[0]) ** 2))),
        float(np.sqrt(np.mean((pred[1] - target[1]) ** 2))),
        float(np.sqrt(np.mean((pred[2] - target[2]) ** 2))),
    )


def rmse_physical(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
    """채널별 RMSE in 물리 단위 (°C, m, ppm)."""
    T_p = denormalize_temperature(pred[0]); T_t = denormalize_temperature(target[0])
    V_p = denormalize_visibility(pred[1]);  V_t = denormalize_visibility(target[1])
    CO_p = denormalize_co(pred[2]);          CO_t = denormalize_co(target[2])
    return (
        float(np.sqrt(np.mean((T_p - T_t) ** 2))),
        float(np.sqrt(np.mean((V_p - V_t) ** 2))),
        float(np.sqrt(np.mean((CO_p - CO_t) ** 2))),
    )


def risk_confusion(
    pred_danger: np.ndarray,  # (T, X, Y, Z) ∈ [0, 1]
    true_danger: np.ndarray,  # (T, X, Y, Z) ∈ [0, 1]
    mask: np.ndarray,          # (X, Y, Z)  fluid=1.0
    threshold: float = DANGER_THRESHOLD,
) -> Dict[str, float]:
    """전체 시계열에 대한 IoU/FNR/FPR 집계 (fluid cell only)."""
    fluid_mask = (mask > 0.5)
    pred_pos = (pred_danger >= threshold)
    true_pos = (true_danger >= threshold)
    # broadcast fluid mask over time
    fm = np.broadcast_to(fluid_mask, pred_pos.shape)
    tp = float(np.sum(pred_pos & true_pos & fm))
    fp = float(np.sum(pred_pos & (~true_pos) & fm))
    fn = float(np.sum((~pred_pos) & true_pos & fm))
    tn = float(np.sum((~pred_pos) & (~true_pos) & fm))
    iou = tp / (tp + fp + fn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    return {"iou": iou, "fnr": fnr, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ─── Per-scenario evaluation ───────────────────────────────────────────────
def evaluate_scenario(
    scen: Dict[str, Any],
    model: FireConvLSTM,
    mask: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """단일 시나리오 평가. 메트릭 + tensor 반환."""
    print(f"[eval] {scen['name']} ... ", end="", flush=True)
    t_start = time.perf_counter()

    # 1) Raw → input/target tensor
    slices = extract_slices(scen["path"])
    norm = normalize_scenario(slices)
    inp = build_input_tensor(norm, mask, times=slices["times"])  # (31, 5, X, Y, Z)
    tgt = build_target_tensor(norm)                              # (31, 3, X, Y, Z)

    # 2) Single-step (30 pairs)
    n_pairs = inp.shape[0] - 1
    preds_single = np.zeros((n_pairs, 3, *GRID_SHAPE), dtype=np.float32)
    rel_list, rmseN_list, rmseP_list = [], [], []
    infer_times_ms = []
    with torch.no_grad():
        for t in range(n_pairs):
            x = torch.from_numpy(inp[t]).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            y = model(x)
            torch.cuda.synchronize() if device.type == "cuda" else None
            infer_times_ms.append((time.perf_counter() - t0) * 1000)
            y_pred = np.clip(y.cpu().numpy()[0], 0.0, 1.0)
            preds_single[t] = y_pred
            y_true = tgt[t + 1]
            rel_list.append(rel_l2(y_pred, y_true))
            rmseN_list.append(rmse_normalized(y_pred, y_true))
            rmseP_list.append(rmse_physical(y_pred, y_true))

    rmseN_arr = np.asarray(rmseN_list)  # (n_pairs, 3)
    rmseP_arr = np.asarray(rmseP_list)

    # 3) Autoregress (6 step from frame 0)
    auto_metrics: List[Dict[str, float]] = []
    state = inp[0].copy()
    mask_ch = state[3].copy()
    with torch.no_grad():
        for step in range(1, 7):
            x = torch.from_numpy(state).unsqueeze(0).to(device)
            y_pred = np.clip(model(x).cpu().numpy()[0], 0.0, 1.0)
            t_idx = step
            if t_idx >= tgt.shape[0]:
                break
            y_true = tgt[t_idx]
            auto_metrics.append({
                "step": step,
                "t_seconds": t_idx * DT_SLCF,
                "rel_l2": rel_l2(y_pred, y_true),
                "rmse_C_T": float(np.sqrt(np.mean((
                    denormalize_temperature(y_pred[0])
                    - denormalize_temperature(y_true[0])
                ) ** 2))),
            })
            t_norm = t_idx * DT_SLCF / T_END_SECONDS
            state = np.zeros_like(state)
            state[:3] = y_pred
            state[3] = mask_ch
            state[4] = np.full_like(mask_ch, t_norm)

    # 4) Risk Map (full 31-frame)
    times_full = np.arange(N_TIMESTEPS) * DT_SLCF  # (31,)
    # Truth danger from FDS slices directly (no normalization roundtrip)
    truth_danger = compute_total_danger(
        slices["temperature"], slices["visibility"], slices["co"],
    ).astype(np.float32)   # (31, X, Y, Z)

    # Pred danger from full single-step rollout
    # Frame 0 의 prediction 은 없음 → truth frame 0 그대로 사용 (입력 자체)
    pred_seq = np.concatenate([tgt[0:1], preds_single], axis=0)  # (31, 3, X, Y, Z)
    pred_danger = prediction_to_danger(pred_seq, times_full)

    risk_metrics_all = risk_confusion(pred_danger, truth_danger, mask)
    # Final-frame snapshot metrics
    risk_metrics_final = risk_confusion(
        pred_danger[-1:], truth_danger[-1:], mask,
    )

    elapsed = time.perf_counter() - t_start
    print(f"done ({elapsed:.1f}s)")

    return {
        "name": scen["name"],
        "loc": scen["loc"],
        "hrr_kw": scen["hrr_kw"],
        "area_m2": scen["area_m2"],
        # Single-step
        "rel_l2_mean":  float(np.mean(rel_list)),
        "rel_l2_max":   float(np.max(rel_list)),
        "rmse_norm_T":  float(np.mean(rmseN_arr[:, 0])),
        "rmse_norm_V":  float(np.mean(rmseN_arr[:, 1])),
        "rmse_norm_CO": float(np.mean(rmseN_arr[:, 2])),
        "rmse_C":       float(np.mean(rmseP_arr[:, 0])),
        "rmse_m":       float(np.mean(rmseP_arr[:, 1])),
        "rmse_ppm":     float(np.mean(rmseP_arr[:, 2])),
        # Autoregress
        "autoreg_rel_l2_1":  auto_metrics[0]["rel_l2"]  if len(auto_metrics) >= 1 else float("nan"),
        "autoreg_rel_l2_3":  auto_metrics[2]["rel_l2"]  if len(auto_metrics) >= 3 else float("nan"),
        "autoreg_rel_l2_6":  auto_metrics[5]["rel_l2"]  if len(auto_metrics) >= 6 else float("nan"),
        # Risk map
        "risk_iou_all":    risk_metrics_all["iou"],
        "risk_fnr_all":    risk_metrics_all["fnr"],
        "risk_fpr_all":    risk_metrics_all["fpr"],
        "risk_iou_final":  risk_metrics_final["iou"],
        "risk_fnr_final":  risk_metrics_final["fnr"],
        # Speed
        "infer_ms_mean":  float(np.mean(infer_times_ms)),
        "infer_ms_std":   float(np.std(infer_times_ms)),
        # Internal data for plots
        "_inp": inp,
        "_tgt": tgt,
        "_pred_seq": pred_seq,
        "_pred_danger": pred_danger,
        "_truth_danger": truth_danger,
        "_auto_metrics": auto_metrics,
    }


# ─── Plotters ──────────────────────────────────────────────────────────────
def plot_z_slice_panel(
    pred: np.ndarray, target: np.ndarray, name: str, t_idx: int,
    out_path: Path,
) -> None:
    """z=1.75 m 단면 T/V/CO 3-row × 3-col (target/pred/|err|)."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    ch_names = ["T_norm", "V_norm", "CO_norm"]
    for c in range(3):
        p = pred[c, :, :, Z_SLICE_IDX].T
        t = target[c, :, :, Z_SLICE_IDX].T
        e = np.abs(p - t)
        for col, (img, ttl, cmap, vmin, vmax) in enumerate([
            (t, f"target {ch_names[c]}", "RdYlGn_r", 0, 1),
            (p, f"pred {ch_names[c]}",   "RdYlGn_r", 0, 1),
            (e, f"|err|",                "magma",    0, max(0.05, float(e.max()))),
        ]):
            im = axes[c, col].imshow(
                img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[0, 30, 0, 20], aspect="equal",
            )
            axes[c, col].set_title(ttl, fontsize=10)
            plt.colorbar(im, ax=axes[c, col], fraction=0.04)
            axes[c, col].set_xticks([]); axes[c, col].set_yticks([])
    fig.suptitle(
        f"ConvLSTM single-step  |  {name}  |  t={t_idx * DT_SLCF:.0f}s → "
        f"t={(t_idx + 1) * DT_SLCF:.0f}s  (z=1.75m)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_autoregress_curve(
    auto_metrics: List[Dict[str, float]], name: str, out_path: Path,
) -> None:
    if not auto_metrics:
        return
    steps = [m["step"] for m in auto_metrics]
    rel = [m["rel_l2"] for m in auto_metrics]
    rmse_T = [m["rmse_C_T"] for m in auto_metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(steps, rel, marker="o", lw=1.5)
    ax1.set_xlabel("autoregress step (10s each)")
    ax1.set_ylabel("Relative L2")
    ax1.set_title(f"Error compounding — {name}")
    ax1.grid(alpha=0.3)
    ax1.axhline(0.15, color="red", lw=0.8, ls="--", label="H2 target ≤ 15%")
    ax1.legend()
    ax2.plot(steps, rmse_T, marker="o", lw=1.5, color="tab:red")
    ax2.set_xlabel("autoregress step")
    ax2.set_ylabel("Temperature RMSE (°C)")
    ax2.set_title("Per-step T RMSE")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_risk_snapshots(
    pred_danger: np.ndarray, true_danger: np.ndarray,
    name: str, out_path: Path,
) -> None:
    """5개 시점 (t=0, 60, 120, 180, 300 s) 의 z=1.75 m risk map 비교."""
    frame_indices = [0, 6, 12, 18, 30]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for col, fi in enumerate(frame_indices):
        if fi >= true_danger.shape[0]:
            continue
        t_img = true_danger[fi, :, :, Z_SLICE_IDX].T
        p_img = pred_danger[fi, :, :, Z_SLICE_IDX].T
        im1 = axes[0, col].imshow(
            t_img, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
            extent=[0, 30, 0, 20], aspect="equal",
        )
        axes[0, col].set_title(f"FDS truth  t={fi * DT_SLCF:.0f}s", fontsize=10)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        plt.colorbar(im1, ax=axes[0, col], fraction=0.04)
        im2 = axes[1, col].imshow(
            p_img, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
            extent=[0, 30, 0, 20], aspect="equal",
        )
        axes[1, col].set_title(f"ConvLSTM pred t={fi * DT_SLCF:.0f}s", fontsize=10)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        plt.colorbar(im2, ax=axes[1, col], fraction=0.04)
    fig.suptitle(
        f"Risk map (z=1.75 m, danger ∈ [0,1])  |  {name}", fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_risk_animation(
    pred_danger: np.ndarray, true_danger: np.ndarray,
    name: str, out_path: Path,
) -> None:
    """전체 31 프레임 risk map animation (GIF)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ims_t = axes[0].imshow(
        true_danger[0, :, :, Z_SLICE_IDX].T,
        origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
        extent=[0, 30, 0, 20], aspect="equal",
    )
    axes[0].set_title("FDS truth", fontsize=11)
    plt.colorbar(ims_t, ax=axes[0], fraction=0.04)
    ims_p = axes[1].imshow(
        pred_danger[0, :, :, Z_SLICE_IDX].T,
        origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
        extent=[0, 30, 0, 20], aspect="equal",
    )
    axes[1].set_title("ConvLSTM pred", fontsize=11)
    plt.colorbar(ims_p, ax=axes[1], fraction=0.04)
    title = fig.suptitle(f"{name}  t=0s", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    writer = PillowWriter(fps=3)
    with writer.saving(fig, str(out_path), dpi=90):
        for fi in range(true_danger.shape[0]):
            ims_t.set_data(true_danger[fi, :, :, Z_SLICE_IDX].T)
            ims_p.set_data(pred_danger[fi, :, :, Z_SLICE_IDX].T)
            title.set_text(f"{name}  t={fi * DT_SLCF:.0f}s")
            writer.grab_frame()
    plt.close(fig)


# ─── Aggregate plots ───────────────────────────────────────────────────────
def plot_aggregate_boxplots(
    results: List[Dict[str, Any]], out_path: Path,
) -> None:
    """위치별/HRR별 RelL2 + Risk IoU boxplot 4-panel."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) by location
    locs = sorted({r["loc"] for r in results})
    by_loc = {L: [r["rel_l2_mean"] for r in results if r["loc"] == L] for L in locs}
    axes[0, 0].boxplot([by_loc[L] for L in locs], labels=locs, showmeans=True)
    axes[0, 0].set_title("Single-step RelL2 by fire location")
    axes[0, 0].set_ylabel("RelL2")
    axes[0, 0].axhline(0.15, color="red", lw=0.8, ls="--", label="H2 ≤ 15%")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # 2) by HRR
    hrrs = sorted({r["hrr_kw"] for r in results})
    by_hrr = {h: [r["rel_l2_mean"] for r in results if r["hrr_kw"] == h] for h in hrrs}
    axes[0, 1].boxplot(
        [by_hrr[h] for h in hrrs], labels=[f"{h} kW" for h in hrrs], showmeans=True,
    )
    axes[0, 1].set_title("Single-step RelL2 by HRR")
    axes[0, 1].set_ylabel("RelL2")
    axes[0, 1].axhline(0.15, color="red", lw=0.8, ls="--")
    axes[0, 1].grid(alpha=0.3)

    # 3) risk IoU by location
    by_loc_iou = {L: [r["risk_iou_all"] for r in results if r["loc"] == L] for L in locs}
    axes[1, 0].boxplot([by_loc_iou[L] for L in locs], labels=locs, showmeans=True)
    axes[1, 0].set_title("Risk-map IoU by fire location")
    axes[1, 0].set_ylabel("IoU (threshold=0.5)")
    axes[1, 0].axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # 4) risk FNR by location
    by_loc_fnr = {L: [r["risk_fnr_all"] for r in results if r["loc"] == L] for L in locs}
    axes[1, 1].boxplot([by_loc_fnr[L] for L in locs], labels=locs, showmeans=True)
    axes[1, 1].set_title("Risk-map FNR by fire location")
    axes[1, 1].set_ylabel("FNR (false negative rate)")
    axes[1, 1].axhline(0.10, color="red", lw=0.8, ls="--", label="H4 < 10%")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        "ConvLSTM OOD evaluation — T01-T05 (13 scenarios)", fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── CSV + markdown writers ────────────────────────────────────────────────
SUMMARY_COLS = [
    "name", "loc", "hrr_kw", "area_m2",
    "rel_l2_mean", "rel_l2_max",
    "rmse_C", "rmse_m", "rmse_ppm",
    "autoreg_rel_l2_1", "autoreg_rel_l2_3", "autoreg_rel_l2_6",
    "risk_iou_all", "risk_fnr_all", "risk_fpr_all",
    "risk_iou_final", "risk_fnr_final",
    "infer_ms_mean", "infer_ms_std",
]


def write_csv(results: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLS)
        for r in results:
            w.writerow([r.get(c, "") for c in SUMMARY_COLS])


def write_aggregated_csv(results: List[Dict[str, Any]], out_path: Path) -> None:
    """위치별/HRR별 평균 CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # by location
    for loc in sorted({r["loc"] for r in results}):
        sub = [r for r in results if r["loc"] == loc]
        rows.append({
            "group": f"loc={loc}",
            "n": len(sub),
            "rel_l2_mean":     float(np.mean([r["rel_l2_mean"]     for r in sub])),
            "rmse_C":          float(np.mean([r["rmse_C"]          for r in sub])),
            "rmse_m":          float(np.mean([r["rmse_m"]          for r in sub])),
            "rmse_ppm":        float(np.mean([r["rmse_ppm"]        for r in sub])),
            "autoreg_rel_l2_6":float(np.mean([r["autoreg_rel_l2_6"]for r in sub])),
            "risk_iou_all":    float(np.mean([r["risk_iou_all"]    for r in sub])),
            "risk_fnr_all":    float(np.mean([r["risk_fnr_all"]    for r in sub])),
        })
    # by HRR
    for hrr in sorted({r["hrr_kw"] for r in results}):
        sub = [r for r in results if r["hrr_kw"] == hrr]
        rows.append({
            "group": f"hrr={hrr}kW",
            "n": len(sub),
            "rel_l2_mean":     float(np.mean([r["rel_l2_mean"]     for r in sub])),
            "rmse_C":          float(np.mean([r["rmse_C"]          for r in sub])),
            "rmse_m":          float(np.mean([r["rmse_m"]          for r in sub])),
            "rmse_ppm":        float(np.mean([r["rmse_ppm"]        for r in sub])),
            "autoreg_rel_l2_6":float(np.mean([r["autoreg_rel_l2_6"]for r in sub])),
            "risk_iou_all":    float(np.mean([r["risk_iou_all"]    for r in sub])),
            "risk_fnr_all":    float(np.mean([r["risk_fnr_all"]    for r in sub])),
        })
    # overall
    rows.append({
        "group": "ALL",
        "n": len(results),
        "rel_l2_mean":     float(np.mean([r["rel_l2_mean"]     for r in results])),
        "rmse_C":          float(np.mean([r["rmse_C"]          for r in results])),
        "rmse_m":          float(np.mean([r["rmse_m"]          for r in results])),
        "rmse_ppm":        float(np.mean([r["rmse_ppm"]        for r in results])),
        "autoreg_rel_l2_6":float(np.mean([r["autoreg_rel_l2_6"]for r in results])),
        "risk_iou_all":    float(np.mean([r["risk_iou_all"]    for r in results])),
        "risk_fnr_all":    float(np.mean([r["risk_fnr_all"]    for r in results])),
    })
    cols = ["group", "n", "rel_l2_mean", "rmse_C", "rmse_m", "rmse_ppm",
            "autoreg_rel_l2_6", "risk_iou_all", "risk_fnr_all"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def write_markdown_report(
    results: List[Dict[str, Any]],
    out_path: Path,
    fig_root: Path,
) -> None:
    """위치별 페이지 형태 markdown 보고서."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    locs = sorted({r["loc"] for r in results})
    overall_rel = np.mean([r["rel_l2_mean"] for r in results])
    overall_iou = np.mean([r["risk_iou_all"] for r in results])
    overall_fnr = np.mean([r["risk_fnr_all"] for r in results])
    overall_auto6 = np.mean([r["autoreg_rel_l2_6"] for r in results])
    infer = np.mean([r["infer_ms_mean"] for r in results])

    lines: List[str] = []
    lines.append("# ConvLSTM OOD 평가 — T01-T05 위치 보고서\n")
    lines.append(f"> **평가 시점**: 자동 생성  \n")
    lines.append(f"> **시나리오 수**: {len(results)}  \n")
    lines.append(f"> **모델**: ConvLSTM (`checkpoints/conv_lstm/best.pt`)  \n")
    lines.append(f"> **건물**: training 과 동일 (MESH/SLCF spec 일치) → **위치 OOD only**\n")
    lines.append("\n---\n")
    lines.append("## 0. 한 줄 요약\n")
    lines.append(
        f"새 5개 화재 위치(T01-T05) × 13개 HRR/면적 조합에 대해 ConvLSTM 의 "
        f"single-step RelL2 평균 **{overall_rel:.3f}**, 6-step autoregress RelL2 "
        f"**{overall_auto6:.3f}**, 위험도 맵 IoU **{overall_iou:.3f}** / FNR "
        f"**{overall_fnr * 100:.1f}%**, 추론 속도 **{infer:.1f} ms** "
        f"(CPU 기준).\n"
    )
    lines.append("\n## 1. 집계 결과\n")
    lines.append("![aggregated](" + str(fig_root / "aggregated_boxplots.png").replace("\\", "/") + ")\n")
    lines.append("\n### 1.1 전체 시나리오 단일 표\n")
    lines.append("| name | loc | HRR | area | RelL2 | RMSE °C | RMSE m | RMSE ppm | auto-6 | IoU | FNR | infer ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['loc']} | {r['hrr_kw']} | {r['area_m2']} m² | "
            f"{r['rel_l2_mean']:.3f} | {r['rmse_C']:.2f} | {r['rmse_m']:.2f} | "
            f"{r['rmse_ppm']:.1f} | {r['autoreg_rel_l2_6']:.3f} | "
            f"{r['risk_iou_all']:.3f} | {r['risk_fnr_all'] * 100:.1f}% | "
            f"{r['infer_ms_mean']:.1f} |"
        )
    lines.append("")
    lines.append("\n### 1.2 가설 검증 게이지\n")
    h2_pass = overall_rel <= 0.15
    h4_pass = overall_fnr < 0.10
    h5_pass = overall_iou >= 0.70
    lines.append("| 가설 | 목표 | OOD 측정 | 통과? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **H2** Single-step RelL2 | ≤ 0.15 | {overall_rel:.3f} | {'✅' if h2_pass else '❌'} |")
    lines.append(f"| **H4** Risk FNR | < 10% | {overall_fnr * 100:.1f}% | {'✅' if h4_pass else '❌'} |")
    lines.append(f"| **H5** Risk IoU | ≥ 0.70 | {overall_iou:.3f} | {'✅' if h5_pass else '❌'} |")
    lines.append("")
    # 위치별 페이지
    lines.append("\n---\n")
    lines.append("## 2. 위치별 상세 (T01-T05)\n")
    for loc in locs:
        sub = [r for r in results if r["loc"] == loc]
        lines.append(f"\n### 2.{locs.index(loc)+1} {loc}\n")
        lines.append("| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sub:
            snap = fig_root / r["name"] / "z_slice_t180.png"
            lines.append(
                f"| {r['name']} | {r['hrr_kw']} kW | {r['area_m2']} m² | "
                f"{r['rel_l2_mean']:.3f} | {r['autoreg_rel_l2_6']:.3f} | "
                f"{r['risk_iou_all']:.3f} | {r['risk_fnr_all'] * 100:.1f}% | "
                f"![]({str(snap).replace(chr(92), '/')}) |"
            )
        lines.append("")
        # 첫 시나리오의 risk snapshots / autoreg curve 노출
        first = sub[0]
        risk_img = fig_root / first["name"] / "risk_snapshots.png"
        auto_img = fig_root / first["name"] / "autoreg.png"
        gif_img = fig_root / first["name"] / "risk_animation.gif"
        lines.append(f"**Risk map snapshots** ({first['name']}):  \n")
        lines.append(f"![]({str(risk_img).replace(chr(92), '/')})\n")
        lines.append(f"**Autoregress error** ({first['name']}):  \n")
        lines.append(f"![]({str(auto_img).replace(chr(92), '/')})\n")
        lines.append(f"**Animation** ({first['name']}):  \n")
        lines.append(f"![]({str(gif_img).replace(chr(92), '/')})\n")
    lines.append("\n---\n")
    lines.append("## 3. 비교 — Training 평균 vs OOD 평균\n")
    lines.append(
        "참고: training (33 scen) 평가 결과는 handoff §3 의 ConvLSTM eval 수치.\n\n"
    )
    lines.append("| 항목 | Training (33) | OOD T01-T05 (13) |")
    lines.append("|---|---|---|")
    lines.append(f"| Single-step RelL2 | ≈ 0.115-0.158 | **{overall_rel:.3f}** |")
    lines.append(f"| Autoreg 6-step RelL2 | ≈ 0.093 | **{overall_auto6:.3f}** |")
    lines.append(f"| Risk IoU | ≈ 0.85 | **{overall_iou:.3f}** |")
    lines.append(f"| Risk FNR | ≈ 9.9% | **{overall_fnr * 100:.1f}%** |")
    lines.append(f"| Infer time (CPU) | ≈ 26.7 ms | **{infer:.1f} ms** |")
    lines.append("")
    lines.append("> 차이를 보면 **위치 일반화 능력**(H3 의 indirect 신호)을 가늠할 수 있음.\n")
    lines.append("> H3 의 정식 검증은 별도 OOD 시뮬 (Member A) 도착 후 FNO vs ConvLSTM 비교로 진행.\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/conv_lstm/best.pt"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--figures", type=Path, default=Path("figures/eval_T01_T05"))
    parser.add_argument("--results", type=Path, default=Path("results/eval_T01_T05"))
    parser.add_argument("--report",  type=Path, default=Path("docs/eval_T01_T05_report.md"))
    parser.add_argument("--no-gif", action="store_true", help="risk animation GIF skip (속도)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[setup] device={device}")

    if not args.ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset.h5 not found (for mask): {args.dataset}")

    args.figures.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading model from {args.ckpt}")
    model = load_model(args.ckpt, device)
    print(f"[setup] loading mask from {args.dataset}")
    mask = load_mask(args.dataset)

    scenarios = discover_scenarios(args.raw_root)
    print(f"[setup] found {len(scenarios)} T-location scenarios")
    if not scenarios:
        print("[FAIL] no scenarios found — check --raw-root")
        return 1

    results = []
    for scen in scenarios:
        try:
            r = evaluate_scenario(scen, model, mask, device)
        except Exception as e:
            print(f"[skip] {scen['name']}: {e}")
            continue
        # Per-scenario figures
        scen_fig_dir = args.figures / scen["name"]
        scen_fig_dir.mkdir(parents=True, exist_ok=True)
        # 1) z=1.75m slice at t=180s (mid-fire)
        plot_z_slice_panel(
            r["_pred_seq"][18], r["_tgt"][18],
            scen["name"], 18,
            scen_fig_dir / "z_slice_t180.png",
        )
        # 2) autoregress curve
        plot_autoregress_curve(
            r["_auto_metrics"], scen["name"],
            scen_fig_dir / "autoreg.png",
        )
        # 3) risk snapshots
        plot_risk_snapshots(
            r["_pred_danger"], r["_truth_danger"],
            scen["name"],
            scen_fig_dir / "risk_snapshots.png",
        )
        # 4) risk animation
        if not args.no_gif:
            plot_risk_animation(
                r["_pred_danger"], r["_truth_danger"],
                scen["name"],
                scen_fig_dir / "risk_animation.gif",
            )
        # strip internal data
        for k in list(r.keys()):
            if k.startswith("_"):
                del r[k]
        results.append(r)

    # Aggregated outputs
    print("[agg] writing CSV / aggregated CSV / boxplots / markdown report")
    write_csv(results, args.results / "per_scenario_metrics.csv")
    write_aggregated_csv(results, args.results / "aggregated.csv")
    plot_aggregate_boxplots(results, args.figures / "aggregated_boxplots.png")
    write_markdown_report(results, args.report, args.figures)

    # Also dump JSON for traceability
    with (args.results / "per_scenario_metrics.json").open("w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
                  f, indent=2, ensure_ascii=False)

    print(f"\n[PASS] evaluated {len(results)} scenarios")
    print(f"  CSV  : {args.results / 'per_scenario_metrics.csv'}")
    print(f"  Agg  : {args.results / 'aggregated.csv'}")
    print(f"  Plot : {args.figures / 'aggregated_boxplots.png'}")
    print(f"  Doc  : {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
