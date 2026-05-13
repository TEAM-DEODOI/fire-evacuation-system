"""ConvLSTM 예측 vs FDS 정답 risk map 비교 시각화 (발표용).

EXP-RISK-001 의 핵심 평가 방식 (teacher-forced single-step prediction)
으로 ConvLSTM 의 매 frame 예측을 정답과 비교한다:

* ``pred[0]    = target[0]`` (t=0 — ambient initial condition, 모델 호출 X)
* ``pred[t+1]  = model(input[t])`` for t=0…29

예측 출력을 :mod:`src.risk_map.converter.prediction_to_danger` 로 통과시켜
정답과 동일한 (T, 60, 40, 6) 위험도 grid 로 변환한 뒤,

1. **3-panel 동영상 GIF** — GT / ConvLSTM / |GT-Pred| at z=1.75m
2. **헤드라인 스냅샷 PNG** — t=50/150/300s 3개 시점
3. **per-frame 메트릭 곡선** — rel L2, IoU @ 0.5, FNR, FPR
4. **summary JSON** — frame별 + 평균 메트릭

위험도 임계값 0.5 (``TENABILITY.AGGREGATE_THRESHOLD``) 위 셀이 "위험 영역".
메트릭은 fluid cell (mask=1) 에 한정하여 계산해 wall 의 trivial-zero 가
값을 왜곡하지 않도록 함.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.conv_lstm_3d import FireConvLSTM
from src.risk_map.converter import prediction_to_danger
from src.shared.constants import (
    DOMAIN_SIZE_M,
    DT_SLCF,
    GRID_SHAPE,
    N_TIMESTEPS,
    TENABILITY,
)


# ─── Model loading ──────────────────────────────────────────────────────────
def load_model(ckpt: Path, device: torch.device) -> FireConvLSTM:
    # weights_only=False — own-trained checkpoint (PyTorch 2.6+ default change)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
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


# ─── Teacher-forced prediction sequence ─────────────────────────────────────
def teacher_forced_predict(
    model: FireConvLSTM,
    inp: np.ndarray,        # (31, 5, X, Y, Z)
    tgt: np.ndarray,        # (31, 3, X, Y, Z) — for t=0 seed
    device: torch.device,
) -> np.ndarray:
    """``pred[t+1] = model(input[t])`` for every t. ``pred[0] = target[0]``.

    이게 EXP-RISK-001 의 표준 평가 형태 — 매 step 의 입력이 정답이라
    autoregression 의 compounding error 가 배제된 단계별 정확도를 평가.
    """
    n = inp.shape[0]
    preds = np.zeros((n, 3, *GRID_SHAPE), dtype=np.float32)
    preds[0] = tgt[0]
    with torch.no_grad():
        for t in range(n - 1):
            x = torch.from_numpy(inp[t]).unsqueeze(0).to(device)
            y = np.clip(model(x).cpu().numpy()[0], 0.0, 1.0)
            preds[t + 1] = y
    return preds


# ─── Per-frame metrics (mask-restricted) ───────────────────────────────────
def _safe_div(num: float, den: float, eps: float = 1e-9) -> float:
    return num / (den + eps)


def per_frame_metrics(
    gt_danger: np.ndarray,    # (T, X, Y, Z)
    pred_danger: np.ndarray,  # (T, X, Y, Z)
    mask: np.ndarray,         # (X, Y, Z) — 1=fluid, 0=solid
    threshold: float = TENABILITY.AGGREGATE_THRESHOLD,
) -> List[Dict[str, float]]:
    """매 frame 의 rel L2 / IoU / FNR / FPR (fluid cells only)."""
    metrics: List[Dict[str, float]] = []
    fluid = mask.astype(bool)
    for t in range(gt_danger.shape[0]):
        gt = gt_danger[t][fluid]
        pr = pred_danger[t][fluid]

        # Relative L2
        num = float(np.sqrt(np.mean((gt - pr) ** 2)))
        den = float(np.sqrt(np.mean(gt ** 2)))
        rel = _safe_div(num, den)

        # Threshold-based confusion
        gt_pos = gt > threshold
        pr_pos = pr > threshold
        tp = int(np.sum(gt_pos & pr_pos))
        fp = int(np.sum(~gt_pos & pr_pos))
        fn = int(np.sum(gt_pos & ~pr_pos))
        tn = int(np.sum(~gt_pos & ~pr_pos))
        union = tp + fp + fn
        iou = _safe_div(tp, union)
        fnr = _safe_div(fn, tp + fn)  # 위험 셀 놓치는 비율
        fpr = _safe_div(fp, fp + tn)  # 안전 셀을 위험으로 잘못 보고

        metrics.append({
            "t_idx": t,
            "t_seconds": t * DT_SLCF,
            "rel_l2": rel,
            "iou_at_0.5": iou,
            "fnr": fnr,
            "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "gt_max": float(gt_danger[t].max()),
            "pred_max": float(pred_danger[t].max()),
        })
    return metrics


# ─── Visualisations ────────────────────────────────────────────────────────
def _format_title(scenario_id: str, t_s: float, m: Dict[str, float]) -> str:
    return (
        f"{scenario_id}  |  t = {t_s:.0f} s  |  "
        f"IoU={m['iou_at_0.5']:.2f}  FNR={m['fnr']:.2f}  "
        f"RelL2={m['rel_l2']:.3f}"
    )


def make_snapshot_panel(
    gt_danger: np.ndarray,
    pred_danger: np.ndarray,
    metrics: List[Dict[str, float]],
    scenario_id: str,
    out_path: Path,
    snapshot_t_idx: List[int] = (5, 15, 30),
    z_idx: int = 3,
) -> None:
    """3 시점 × 3 패널 (GT, Pred, |err|) 헤드라인 그리드."""
    rows = len(snapshot_t_idx)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.4 * rows))
    if rows == 1:
        axes = np.array([axes])
    lx, ly, _ = DOMAIN_SIZE_M

    for r, ti in enumerate(snapshot_t_idx):
        gt = gt_danger[ti, :, :, z_idx].T
        pr = pred_danger[ti, :, :, z_idx].T
        err = np.abs(gt - pr)
        m = metrics[ti]

        for c, (img, ttl, cmap, vmin, vmax) in enumerate([
            (gt,  "FDS ground truth",  "RdYlGn_r", 0.0, 1.0),
            (pr,  "ConvLSTM prediction", "RdYlGn_r", 0.0, 1.0),
            (err, "|GT - Pred|",       "magma",    0.0, max(0.05, float(err.max()))),
        ]):
            im = axes[r, c].imshow(
                img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[0, lx, 0, ly], aspect="equal",
            )
            axes[r, c].set_title(ttl, fontsize=10)
            plt.colorbar(im, ax=axes[r, c], fraction=0.04)
            if c == 0:
                axes[r, c].set_ylabel(
                    f"t = {m['t_seconds']:.0f}s\n"
                    f"IoU={m['iou_at_0.5']:.2f}  FNR={m['fnr']:.2f}",
                    fontsize=9,
                )
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    fig.suptitle(
        f"Risk map: FDS truth vs ConvLSTM prediction  |  {scenario_id}  |  "
        f"z = {0.25 + z_idx * 0.5:.2f} m",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def make_animation_gif(
    gt_danger: np.ndarray,
    pred_danger: np.ndarray,
    metrics: List[Dict[str, float]],
    scenario_id: str,
    out_path: Path,
    z_idx: int = 3,
    fps: int = 5,
) -> None:
    """31 frame side-by-side 애니메이션 GIF."""
    lx, ly, _ = DOMAIN_SIZE_M
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # initial frame
    gt0 = gt_danger[0, :, :, z_idx].T
    pr0 = pred_danger[0, :, :, z_idx].T
    err0 = np.abs(gt0 - pr0)

    im_gt = axes[0].imshow(
        gt0, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
        extent=[0, lx, 0, ly], aspect="equal",
    )
    im_pr = axes[1].imshow(
        pr0, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
        extent=[0, lx, 0, ly], aspect="equal",
    )
    # err 의 vmax 는 전체 시퀀스의 최대값으로 고정 — 시간에 따라 컬러바 안정
    err_vmax = float(np.abs(gt_danger - pred_danger).max())
    err_vmax = max(err_vmax, 0.05)
    im_err = axes[2].imshow(
        err0, origin="lower", cmap="magma", vmin=0, vmax=err_vmax,
        extent=[0, lx, 0, ly], aspect="equal",
    )
    axes[0].set_title("FDS ground truth")
    axes[1].set_title("ConvLSTM prediction")
    axes[2].set_title("|GT - Pred|")
    for ax in axes:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    plt.colorbar(im_gt, ax=axes[0], fraction=0.04)
    plt.colorbar(im_pr, ax=axes[1], fraction=0.04)
    plt.colorbar(im_err, ax=axes[2], fraction=0.04)

    suptitle = fig.suptitle(_format_title(scenario_id, 0.0, metrics[0]), fontsize=11)

    def update(frame_idx: int):
        gt = gt_danger[frame_idx, :, :, z_idx].T
        pr = pred_danger[frame_idx, :, :, z_idx].T
        err = np.abs(gt - pr)
        im_gt.set_array(gt)
        im_pr.set_array(pr)
        im_err.set_array(err)
        suptitle.set_text(
            _format_title(scenario_id, metrics[frame_idx]["t_seconds"], metrics[frame_idx])
        )
        return [im_gt, im_pr, im_err, suptitle]

    anim = animation.FuncAnimation(
        fig, update, frames=gt_danger.shape[0], interval=1000 // fps, blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(out_path, writer="pillow", fps=fps)
        print(f"  → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    except Exception as exc:  # pragma: no cover
        print(f"  GIF save failed: {exc}")
    plt.close(fig)


def plot_metrics_curve(
    metrics: List[Dict[str, float]],
    scenario_id: str,
    out_path: Path,
) -> None:
    """rel L2 / IoU / FNR / FPR 의 frame 추이."""
    ts = [m["t_seconds"] for m in metrics]
    rel = [m["rel_l2"] for m in metrics]
    iou = [m["iou_at_0.5"] for m in metrics]
    fnr = [m["fnr"] for m in metrics]
    fpr = [m["fpr"] for m in metrics]
    gt_max = [m["gt_max"] for m in metrics]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7))

    ax1.plot(ts, rel, lw=1.5, marker="o", ms=3)
    ax1.axhline(0.15, color="red", lw=0.7, ls="--", label="H2 target ≤ 15%")
    ax1.set_xlabel("t (s)"); ax1.set_ylabel("Relative L2 (mask-restricted)")
    ax1.set_title("Relative L2  |  ConvLSTM danger vs FDS danger")
    ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(ts, iou, lw=1.5, marker="o", ms=3, color="tab:green")
    ax2.axhline(0.7, color="red", lw=0.7, ls="--", label="H5 target ≥ 0.70")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("IoU @ threshold 0.5")
    ax2.set_title("Hazard-region IoU  (overlap of danger regions)")
    ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.3); ax2.legend()

    ax3.plot(ts, fnr, lw=1.5, marker="o", ms=3, color="tab:red", label="FNR (missed danger)")
    ax3.plot(ts, fpr, lw=1.5, marker="s", ms=3, color="tab:orange", label="FPR (false alarm)")
    ax3.axhline(0.10, color="red", lw=0.7, ls="--", label="H4 FNR target < 10%")
    ax3.set_xlabel("t (s)"); ax3.set_ylabel("rate")
    ax3.set_title("Confusion rates  @ threshold 0.5")
    ax3.set_ylim(0, max(0.3, max(fnr + fpr) * 1.1))
    ax3.grid(alpha=0.3); ax3.legend()

    ax4.plot(ts, gt_max, lw=1.5, marker="o", ms=3, color="tab:purple",
             label="GT max danger")
    pr_max = [m["pred_max"] for m in metrics]
    ax4.plot(ts, pr_max, lw=1.5, marker="s", ms=3, color="tab:blue",
             label="Pred max danger")
    ax4.set_xlabel("t (s)"); ax4.set_ylabel("max danger")
    ax4.set_title("Peak danger over time")
    ax4.set_ylim(0, 1.05); ax4.grid(alpha=0.3); ax4.legend()

    fig.suptitle(f"Per-frame risk-map metrics  |  {scenario_id}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", type=str, default="scenario_014",
        help="HDF5 group (e.g. scenario_014 = 1500kW@F1).",
    )
    parser.add_argument("--ckpt", type=Path,
                        default=Path("checkpoints/conv_lstm/best.pt"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/processed/dataset.h5"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: figures/risk_compare/<scenario>/")
    parser.add_argument("--snapshots", type=int, nargs="+", default=[5, 15, 30],
                        help="t_idx for the snapshot grid (default: 5, 15, 30 → 50/150/300s).")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    if args.output is None:
        args.output = Path("figures/risk_compare") / args.scenario
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"scenario: {args.scenario}")
    print(f"checkpoint: {args.ckpt}")
    print(f"output: {args.output}")

    # ── Model + data ─────────────────────────────────────────────────────
    model = load_model(args.ckpt, device)
    print(f"model params: {model.count_parameters():,}")

    with h5py.File(args.dataset, "r") as h5:
        if args.scenario not in h5:
            print(f"ERROR: scenario {args.scenario!r} not in {args.dataset}")
            return 1
        grp = h5[args.scenario]
        inp = np.asarray(grp["input"])
        tgt = np.asarray(grp["target"])
        mask = np.asarray(h5["mask"])
        original_id = grp.attrs.get("original_id", "?")
    print(f"scenario meta: original_id={original_id}, "
          f"input.shape={inp.shape}, mask solid={(mask==0).sum()}/{mask.size}")

    # ── Teacher-forced predictions ────────────────────────────────────────
    print("\n[1/4] teacher-forced prediction sequence...")
    preds = teacher_forced_predict(model, inp, tgt, device)
    print(f"  preds.shape={preds.shape}, "
          f"range=[{preds.min():.4f}, {preds.max():.4f}]")

    # ── Danger fields ─────────────────────────────────────────────────────
    print("\n[2/4] danger map conversion (denorm + tenability)...")
    times = np.arange(0, N_TIMESTEPS * DT_SLCF, DT_SLCF, dtype=np.float64)
    gt_danger = prediction_to_danger(tgt, times)
    pred_danger = prediction_to_danger(preds, times)
    print(f"  gt range [{gt_danger.min():.3f}, {gt_danger.max():.3f}],  "
          f"pred range [{pred_danger.min():.3f}, {pred_danger.max():.3f}]")

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\n[3/4] per-frame metrics...")
    metrics = per_frame_metrics(gt_danger, pred_danger, mask)
    rels = [m["rel_l2"] for m in metrics[1:]]   # skip t=0 (trivial identity)
    ious = [m["iou_at_0.5"] for m in metrics[1:]]
    fnrs = [m["fnr"] for m in metrics[1:]]
    fprs = [m["fpr"] for m in metrics[1:]]
    print(f"  rel_l2  mean={np.mean(rels):.4f}, max={np.max(rels):.4f}")
    print(f"  IoU @0.5 mean={np.mean(ious):.4f}, min={np.min(ious):.4f}")
    print(f"  FNR     mean={np.mean(fnrs):.4f}, max={np.max(fnrs):.4f}")
    print(f"  FPR     mean={np.mean(fprs):.4f}, max={np.max(fprs):.4f}")

    # ── Visualisations ────────────────────────────────────────────────────
    print("\n[4/4] visualisations...")
    make_snapshot_panel(
        gt_danger, pred_danger, metrics,
        args.scenario, args.output / "snapshots.png",
        snapshot_t_idx=list(args.snapshots),
    )
    make_animation_gif(
        gt_danger, pred_danger, metrics,
        args.scenario, args.output / "animation.gif", fps=args.fps,
    )
    plot_metrics_curve(metrics, args.scenario, args.output / "metrics.png")

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "scenario": args.scenario,
        "original_id": (
            original_id.decode() if isinstance(original_id, bytes) else str(original_id)
        ),
        "checkpoint": str(args.ckpt),
        "mode": "teacher-forced",
        "aggregate": {
            "rel_l2_mean": float(np.mean(rels)),
            "rel_l2_max": float(np.max(rels)),
            "iou_mean": float(np.mean(ious)),
            "iou_min": float(np.min(ious)),
            "fnr_mean": float(np.mean(fnrs)),
            "fnr_max": float(np.max(fnrs)),
            "fpr_mean": float(np.mean(fprs)),
            "fpr_max": float(np.max(fprs)),
        },
        "per_frame": metrics,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsummary → {args.output / 'summary.json'}")
    print(f"all artefacts in: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
