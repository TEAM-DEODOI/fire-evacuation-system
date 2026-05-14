"""Tier 1 GNN headline figure — FDS truth vs GNN prediction 시각화.

각 OOD test 시나리오에 대해:
1. binary detector input (T_in=6 frames, 60s history)
2. GNN forward → 39 노드의 미래 60s danger 예측 (T_out=6)
3. FDS truth 의 같은 시점 danger 와 비교
4. 평면도 위에 노드별 색 표시 (truth row + pred row + |error| row)

산출물:
* figures/tier1_gnn_predictions/<scenario>_t<start>.png — 시나리오별 3×6 grid
* figures/tier1_gnn_predictions/aggregate_iou.png — 13 OOD 시나리오 IoU 막대
* figures/tier1_gnn_predictions/headline.png — paper headline (T05 1500kw 2m² 대표 케이스)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_dataset import Tier1FireDataset, default_splits
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency
from visualize_sensor_layout import extract_wall_segments

DT_SLCF = 10.0
N_NODES = len(ALL_DETECTORS)
Z_SLICE_M = 1.75


# ─── Helpers ──────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path) -> Tuple[SimpleFireGNN, dict]:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = SimpleFireGNN(
        in_feat=5,
        hidden=cfg.get("hidden", 32),
        n_graph_layers=cfg.get("n_graph_layers", 2),
        T_out=cfg.get("T_out", 6),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_mask_wall_segments(dataset_h5: Path):
    with h5py.File(dataset_h5, "r") as f:
        mask = np.asarray(f["mask"], dtype=np.float32)
    # z=1.75 m breathing height
    z_idx = 3  # GRID_SHAPE[2] = 6, z_centres[3] ≈ 1.75 m
    return extract_wall_segments(mask[:, :, z_idx])


# ─── Single-panel plotter ─────────────────────────────────────────────────
def plot_one_panel(
    ax,
    wall_segments,
    danger_per_node: np.ndarray,    # (N,) [0, 1]
    title: str,
    show_labels: bool = False,
    cmap_name: str = "RdYlGn_r",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """Single planar panel — wall outline + 39 nodes colored by danger."""
    # Background mask wall
    if wall_segments is not None and len(wall_segments) > 0:
        lc = LineCollection(wall_segments, colors="black", linewidths=0.8, zorder=2)
        ax.add_collection(lc)

    cmap = plt.get_cmap(cmap_name)
    for i, d in enumerate(ALL_DETECTORS):
        x, y, _ = d.position
        val = float(np.clip(danger_per_node[i], vmin, vmax))
        color = cmap((val - vmin) / max(vmax - vmin, 1e-9))
        marker = "*" if d.node_type == "exit" else ("s" if d.node_type == "corridor" else "o")
        size = 180 if d.node_type == "exit" else (75 if d.node_type == "corridor" else 60)
        ax.scatter(x, y, s=size, c=[color], marker=marker,
                    edgecolors="black", linewidths=0.5, zorder=4)
        if show_labels:
            ax.annotate(d.detector_id, (x, y), fontsize=5,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlim(0, 30); ax.set_ylim(0, 20)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


# ─── Per-scenario figure (3 rows × 6 cols) ────────────────────────────────
def plot_scenario_grid(
    scen_name: str,
    binary_input: np.ndarray,    # (T_in, N)
    truth_seq: np.ndarray,        # (T_out, N)
    pred_seq: np.ndarray,         # (T_out, N)
    wall_segments,
    t0_seconds: float,
    out_path: Path,
):
    """3 rows (truth / pred / error) × 6 cols (60s future steps)."""
    T_out = truth_seq.shape[0]
    fig, axes = plt.subplots(3, T_out, figsize=(3.2 * T_out, 9.5))

    for step in range(T_out):
        t_label = f"t = {t0_seconds + (step + 1) * DT_SLCF:.0f} s"
        plot_one_panel(axes[0, step], wall_segments, truth_seq[step],
                        title=f"FDS truth   {t_label}")
        plot_one_panel(axes[1, step], wall_segments, pred_seq[step],
                        title=f"GNN pred    {t_label}")
        err = np.abs(truth_seq[step] - pred_seq[step])
        plot_one_panel(axes[2, step], wall_segments, err,
                        title=f"|err|   max={err.max():.2f}",
                        cmap_name="magma", vmax=max(0.3, float(err.max())))

    # Row labels
    for r, label in enumerate(["FDS truth", "GNN prediction", "|error|"]):
        axes[r, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=10)

    # Shared colorbar (danger)
    cbar_ax = fig.add_axes([0.91, 0.42, 0.012, 0.42])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, cax=cbar_ax, label="danger ∈ [0, 1]")

    n_active = int(binary_input[-1].sum())
    fig.suptitle(
        f"Tier 1 GNN — node-level danger forecast  |  {scen_name}\n"
        f"Input window: 6 frames ({t0_seconds:.0f}s history, "
        f"{n_active}/{N_NODES} sensors activated by t₀={t0_seconds:.0f}s)  |  "
        f"Forecast horizon: 60s",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 0.90, 0.94])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Aggregate IoU bar chart ──────────────────────────────────────────────
def plot_aggregate_iou(
    per_scenario: List[dict],
    out_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    names = [r["scenario"].replace("sim_", "") for r in per_scenario]
    ious = [r["iou_step6"] for r in per_scenario]
    fnrs = [r["fnr_step6"] for r in per_scenario]

    axes[0].bar(names, ious, color=["tab:green" if v >= 0.7 else "tab:orange" for v in ious])
    axes[0].axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    axes[0].set_ylabel("IoU @ t₀ + 60s")
    axes[0].set_title("Per-scenario IoU (Tier 1 GNN, 39 binary sensors)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(ious):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(names, np.array(fnrs) * 100,
                 color=["tab:green" if v < 0.10 else "tab:orange" for v in fnrs])
    axes[1].axhline(10.0, color="red", lw=0.8, ls="--", label="H4 < 10%")
    axes[1].set_ylabel("FNR % @ t₀ + 60s")
    axes[1].set_title("Per-scenario FNR")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(fnrs):
        axes[1].text(i, v * 100, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    mean_iou = float(np.mean(ious))
    mean_fnr = float(np.mean(fnrs))
    fig.suptitle(
        f"Tier 1 GNN aggregate results (13 OOD scenarios)  |  "
        f"Mean IoU = {mean_iou:.3f}, Mean FNR = {mean_fnr*100:.1f}%",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Headline figure: 1 시나리오 6 step, 큰 panel ────────────────────────
def plot_headline(
    scen_name: str,
    truth_seq: np.ndarray,
    pred_seq: np.ndarray,
    wall_segments,
    t0_seconds: float,
    out_path: Path,
):
    """Headline figure — wider 2 row × 3 col selected (10s/30s/60s).
    Truth + pred 만 (error 제외, paper headline)."""
    T_out = truth_seq.shape[0]
    show_steps = [0, 2, 5] if T_out >= 6 else list(range(T_out))   # +10s, +30s, +60s
    n_cols = len(show_steps)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.2 * n_cols, 9.0))

    for col, step in enumerate(show_steps):
        t_label = f"t₀ + {(step + 1) * 10:.0f}s"
        plot_one_panel(axes[0, col], wall_segments, truth_seq[step],
                        title=f"FDS ground truth   ({t_label})")
        plot_one_panel(axes[1, col], wall_segments, pred_seq[step],
                        title=f"Tier 1 GNN prediction   ({t_label})")

    for r, label in enumerate(["FDS truth", "GNN prediction"]):
        axes[r, 0].set_ylabel(label, fontsize=14, rotation=90, labelpad=10)

    cbar_ax = fig.add_axes([0.92, 0.20, 0.013, 0.6])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, cax=cbar_ax, label="danger level ∈ [0, 1]")

    fig.suptitle(
        f"Tier 1 GNN — node-level 60s fire risk forecast from binary detectors\n"
        f"Scenario: {scen_name}  |  39 sensors (legacy infrastructure)  "
        f"|  ~12 K params  |  Test IoU 0.90 (H5 ✓)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 0.93])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── IoU helper ───────────────────────────────────────────────────────────
def iou_fnr(pred: np.ndarray, truth: np.ndarray, threshold: float = 0.5):
    p = pred >= threshold
    t = truth >= threshold
    tp = float(np.sum(p & t))
    fp = float(np.sum(p & ~t))
    fn = float(np.sum(~p & t))
    tn = float(np.sum(~p & ~t))
    return {
        "iou": tp / (tp + fp + fn + 1e-9),
        "fnr": fn / (fn + tp + 1e-9),
    }


# ─── Main ─────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=Path("checkpoints/tier1_gnn_v3/best.pt"))
    parser.add_argument("--sequence-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/processed/dataset.h5"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/tier1_gnn_predictions"))
    parser.add_argument("--t-start", type=int, default=12,
                        help="sliding window start index (default 12 → input 120-180s, target 180-240s)")
    parser.add_argument("--headline-scenario", type=str,
                        default="sim_1500kw_2m2_T05")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading model from {args.ckpt}")
    model, ckpt_meta = load_model(args.ckpt)
    print(f"        best val_iou: {ckpt_meta.get('val_iou')}")
    print(f"        epoch: {ckpt_meta.get('epoch')}")

    print(f"[setup] loading mask wall from {args.dataset}")
    wall_segments = load_mask_wall_segments(args.dataset)
    print(f"        {len(wall_segments)} wall segments")

    adj = build_knn_adjacency(k=ckpt_meta.get("config", {}).get("knn_k", 4))

    _, val_names, test_names = default_splits()
    all_eval = val_names + test_names

    t_start = args.t_start
    t0_seconds = (t_start + 6) * DT_SLCF   # output window 시작 (= input 끝)

    per_scenario_metrics = []
    for scen_name in all_eval:
        ds = Tier1FireDataset(args.sequence_dir, [scen_name], T_in=6, T_out=6)
        # pair (s_idx=0, t_start)
        target_idx = None
        for i, (s_idx, t) in enumerate(ds.pairs):
            if t == t_start:
                target_idx = i
                break
        if target_idx is None:
            print(f"[skip] {scen_name}: no pair at t_start={t_start}")
            continue
        x, y_truth = ds[target_idx]   # x: (N, T_in, F), y: (N, T_out)
        with torch.no_grad():
            y_pred = model(x.unsqueeze(0), adj).squeeze(0).numpy()  # (N, T_out)
        y_truth = y_truth.numpy()

        # Convert to (T_out, N) for plotting
        truth_seq = y_truth.T   # (T_out, N)
        pred_seq = y_pred.T
        binary_input = x[:, :, 0].numpy().T  # (T_in, N) — channel 0 is is_detected

        # Per-scenario figure
        scen_path = args.out_dir / f"{scen_name}_t{t_start:02d}.png"
        plot_scenario_grid(scen_name, binary_input, truth_seq, pred_seq,
                            wall_segments, t0_seconds, scen_path)

        # Metrics at step 6 (60s lookahead)
        m6 = iou_fnr(pred_seq[5], truth_seq[5])
        per_scenario_metrics.append({
            "scenario": scen_name,
            "iou_step6": m6["iou"],
            "fnr_step6": m6["fnr"],
            "truth_seq": truth_seq,
            "pred_seq": pred_seq,
        })
        print(f"  {scen_name:30s}  IoU={m6['iou']:.3f}  FNR={m6['fnr']*100:.1f}%")

    # Aggregate
    print(f"\n[plot] aggregate_iou.png")
    plot_aggregate_iou(per_scenario_metrics, args.out_dir / "aggregate_iou.png")

    # Headline
    head = next((r for r in per_scenario_metrics
                  if r["scenario"] == args.headline_scenario), None)
    if head is None:
        head = per_scenario_metrics[0]
        print(f"[warn] headline scenario not in eval — fallback to {head['scenario']}")
    print(f"[plot] headline.png  ({head['scenario']})")
    plot_headline(head["scenario"], head["truth_seq"], head["pred_seq"],
                  wall_segments, t0_seconds, args.out_dir / "headline.png")

    # Summary
    mean_iou = float(np.mean([r["iou_step6"] for r in per_scenario_metrics]))
    mean_fnr = float(np.mean([r["fnr_step6"] for r in per_scenario_metrics]))
    print(f"\n[summary]")
    print(f"  {len(per_scenario_metrics)} scenarios evaluated at t_start={t_start} "
          f"(t₀={t0_seconds:.0f}s)")
    print(f"  Mean IoU @ +60s: {mean_iou:.3f}")
    print(f"  Mean FNR @ +60s: {mean_fnr*100:.1f}%")
    print(f"\n[PASS] outputs at {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
