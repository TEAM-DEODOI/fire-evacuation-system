"""Tier 1 GNN + Tier 2 FNO Ensemble — 학습 0, 즉시 검증.

Two-tier 시스템 통합:
- Tier 1 GNN (binary input, 39 nodes): IoU 0.904 (per-node)
- Tier 2 Sparse FNO 6-ch (continuous input, 60×40×6 cells): IoU 0.525 (per-cell)

Ensemble 절차:
1. Tier 1 forward → (T_out=6, N=39) node danger 시계열
2. Tier 1 출력을 cell-level 로 변환 — 각 cell 에서 k-nearest sensor 의 IDW
3. Tier 2 forward → (T_out=6, X, Y, Z) cell danger
4. Weighted average: w_tier1 * d_tier1_cell + w_tier2 * d_tier2

가중치 sweep: w_tier1 ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}.
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

from src.data_pipeline.fds_extractor import extract_slices
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS
from src.shared.coordinates import cell_centres
from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_dataset import Tier1FireDataset
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency
from evaluate_t_locations import load_mask, load_model
from evaluate_sparse_fno import (
    load_sparse_fno, build_sparse_6ch_input, autoregress_sparse_fno,
)
from evaluate_sparse_model import build_sparse_input, autoregress_sparse
from train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator

SCEN_RE = re.compile(r"^sim_(?P<hrr>\d+)kw_(?P<area>\d+)m2_(?P<loc>T\d{2})$")
LOOKAHEAD_STEPS = 6
T_IN = 6


# ─── Tier 1 node danger → cell-level conversion ───────────────────────────
def precompute_node_to_cell_weights(
    node_positions: List[Tuple[float, float, float]],
    k: int = 3,
    p: float = 2.0,
    sigma: float = 5.0,
    mask: np.ndarray = None,
    use_geodesic: bool = False,
    mask_aware: bool = False,
    adaptive_sigma: bool = False,
    sigma_floor: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """각 cell 에서 k-nearest node 의 IDW weight 계산.

    Args:
        use_geodesic: True 면 BFS geodesic distance (벽 우회), False 면 Euclidean.
        mask: (X, Y, Z) fluid/solid mask — geodesic 사용 시 필수.
        mask_aware: True 면 unreachable (BFS=inf) 노드의 weight 를 0 으로
            강제 (k 슬롯에 들어가도 영향 0). 또한 모든 노드가 unreachable
            인 cell (solid) 은 `np.ones(k)/k` fallback 대신 weight 0 유지.
        adaptive_sigma: True 면 cell-local σ 사용 — top-k 중 reachable 한
            가장 먼 노드 거리를 σ 로. 작은 방 cell 은 작은 σ (local 강조),
            큰 방 cell 은 큰 σ.
        sigma_floor: adaptive_sigma 시 minimum σ (m). 너무 작으면 단일
            노드만 dominate → 노이즈 발생.

    Returns:
        knn_idx: (X, Y, Z, k) — k-nearest node 인덱스
        knn_w:   (X, Y, Z, k) — normalized weights
    """
    x_c, y_c, z_c = cell_centres()
    nx, ny, nz = GRID_SHAPE
    node_xyz = np.array(node_positions)
    N = len(node_xyz)

    if use_geodesic:
        # Use scripts/evaluate_sparse_sensing_geodesic.py logic
        from evaluate_sparse_sensing_geodesic import (
            precompute_geodesic_distances, bfs_geodesic_distance,
        )
        # cell-size in m: 0.5
        # geodesic from each node → (N, X, Y) at z_breathing (z_idx=3)
        node_xy = [(p[0], p[1]) for p in node_positions]
        z_idx_breathing = 3
        geo_dist_2d = precompute_geodesic_distances(mask, node_xy, z_idx_breathing)
        # (N, nx, ny) in cell steps; convert to m
        geo_dist_2d_m = geo_dist_2d * 0.5
        # Broadcast over z (use same 2D distance for all z layers)
        # shape: (N, nx, ny, nz) — same value across z
        geo_dist = np.broadcast_to(
            geo_dist_2d_m[:, :, :, None], (N, nx, ny, nz)
        ).copy()
    else:
        # Euclidean
        cell_centres_grid = np.zeros((nx, ny, nz, 3), dtype=np.float32)
        cell_centres_grid[..., 0] = x_c[:, None, None]
        cell_centres_grid[..., 1] = y_c[None, :, None]
        cell_centres_grid[..., 2] = z_c[None, None, :]
        # (N, nx, ny, nz)
        diff = node_xyz[:, None, None, None, :] - cell_centres_grid[None, ...]
        geo_dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    knn_idx = np.zeros((nx, ny, nz, k), dtype=np.int32)
    knn_w = np.zeros((nx, ny, nz, k), dtype=np.float32)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                d = geo_dist[:, ix, iy, iz]
                # Replace inf (unreachable) with large value
                d_safe = np.where(np.isfinite(d), d, 1e6)
                top_k = np.argsort(d_safe)[:k]
                top_k_dists = d_safe[top_k]
                knn_idx[ix, iy, iz] = top_k

                # Reachable mask (top-k slot 별)
                reachable = top_k_dists < 1e5    # < 1e5 m = reachable

                # σ selection
                if adaptive_sigma:
                    valid_dists = top_k_dists[reachable]
                    if valid_dists.size > 0:
                        # k-th nearest reachable distance (with floor)
                        sigma_cell = max(sigma_floor, float(valid_dists[-1]))
                    else:
                        sigma_cell = sigma
                else:
                    sigma_cell = sigma

                w = np.exp(-top_k_dists ** 2 / (2 * sigma_cell ** 2))

                if mask_aware:
                    # Unreachable slot → weight 0
                    w = np.where(reachable, w, 0.0)
                    w_sum = w.sum()
                    if w_sum > 1e-9:
                        w = w / w_sum
                    else:
                        # Solid cell or fully unreachable — leave at 0
                        # (callers should multiply final danger map by fluid
                        # mask anyway; metrics are already mask-filtered).
                        w = np.zeros_like(w)
                else:
                    # Legacy behaviour: equal-weight fallback for solid cells
                    w_sum = w.sum()
                    w = w / (w_sum + 1e-9) if w_sum > 1e-9 else np.ones(k) / k

                knn_w[ix, iy, iz] = w
    return knn_idx, knn_w


def gnn_node_pred_to_cell_danger(
    gnn_pred: np.ndarray,    # (T_out, N) per-node danger
    knn_idx: np.ndarray,     # (X, Y, Z, k)
    knn_w: np.ndarray,       # (X, Y, Z, k)
) -> np.ndarray:
    """(T_out, N) → (T_out, X, Y, Z) by k-NN IDW."""
    T_out, N = gnn_pred.shape
    nx, ny, nz, k = knn_idx.shape
    # gather: (T_out, X, Y, Z, k) = gnn_pred[t, knn_idx]
    expanded = gnn_pred[:, knn_idx]   # (T_out, X, Y, Z, k)
    weighted = expanded * knn_w[None, ...]  # broadcast (T_out, X, Y, Z, k)
    cell_danger = weighted.sum(axis=-1)     # (T_out, X, Y, Z)
    return cell_danger.astype(np.float32)


# ─── Tier 1 GNN forward ───────────────────────────────────────────────────
def tier1_forward(
    scen_name: str,
    seq_dir: Path,
    gnn_model: SimpleFireGNN,
    adj: torch.Tensor,
    t_start: int,
    device: torch.device,
) -> np.ndarray:
    """Tier 1 GNN forward → (T_out, N=39) per-node danger."""
    ds = Tier1FireDataset(seq_dir, [scen_name], T_in=T_IN, T_out=LOOKAHEAD_STEPS)
    target_pair = None
    for i, (s_idx, t) in enumerate(ds.pairs):
        if t == t_start:
            target_pair = i
            break
    if target_pair is None:
        raise ValueError(f"No pair at t_start={t_start} for {scen_name}")
    x, _ = ds[target_pair]   # (N, T_in, F)
    with torch.no_grad():
        y_pred = gnn_model(x.unsqueeze(0), adj).squeeze(0).numpy()  # (N, T_out)
    return y_pred.T   # (T_out, N)


# ─── Tier 2 forward (FNO or ConvLSTM) ─────────────────────────────────────
def tier2_forward(
    scen_dir: Path,
    tier2_model,
    mask: np.ndarray,
    sparse_ind: np.ndarray,
    t0_seconds: float,
    device: torch.device,
    arch: str = "fno",         # "fno" (6-ch) or "conv_lstm" (5-ch)
) -> Tuple[np.ndarray, np.ndarray]:
    """Tier 2 sparse model forward → (T_out, X, Y, Z) cell danger + truth window."""
    slices = extract_slices(scen_dir)
    truth_danger = compute_total_danger(
        slices["temperature"], slices["visibility"], slices["co"],
    ).astype(np.float32)
    t0_idx = int(t0_seconds // DT_SLCF)
    truth_window = truth_danger[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

    if arch == "fno":
        inp = build_sparse_6ch_input(slices, mask, sparse_ind)
        preds_norm = autoregress_sparse_fno(
            tier2_model, inp[t0_idx], sparse_ind, t0_seconds, device,
            resparsify=True,
        )
    elif arch == "conv_lstm":
        inp = build_sparse_input(slices, mask, sparse_ind)   # (31, 5, ...)
        preds_norm = autoregress_sparse(
            tier2_model, inp[t0_idx], sparse_ind, t0_seconds, device,
            resparsify=True,
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")

    times_arr = np.array([t0_seconds + (s + 1) * DT_SLCF
                           for s in range(LOOKAHEAD_STEPS)])
    pred_danger = prediction_to_danger(preds_norm, times_arr)
    return pred_danger, truth_window


# ─── Ensemble + Metrics ───────────────────────────────────────────────────
def iou_rmse_fnr(pred, truth, mask, threshold=0.5):
    fluid = (mask > 0.5)
    fm = np.broadcast_to(fluid, pred.shape)
    p = pred >= threshold; t = truth >= threshold
    tp = float(np.sum(p & t & fm))
    fp = float(np.sum(p & (~t) & fm))
    fn = float(np.sum((~p) & t & fm))
    tn = float(np.sum((~p) & (~t) & fm))
    return {
        "iou": tp / (tp + fp + fn + 1e-9),
        "fnr": fn / (fn + tp + 1e-9),
        "rmse": float(np.sqrt(np.mean(
            (pred - truth).astype(np.float64)[fm.reshape(pred.shape)] ** 2
        ))),
    }


def eval_scenario_all_weights(
    scen_dir: Path,
    gnn_model, tier2_model,
    adj, mask, sparse_ind, knn_idx, knn_w,
    t_start: int, t0_seconds: float,
    weights_list: List[float],
    seq_dir: Path, device: torch.device,
    tier2_arch: str = "fno",
) -> List[Dict[str, Any]]:
    """모든 가중치에 대해 평가."""
    name = scen_dir.name

    # Tier 1 forward (한 번만)
    t1_node = tier1_forward(name, seq_dir, gnn_model, adj, t_start, device)  # (T_out, N)
    t1_cell = gnn_node_pred_to_cell_danger(t1_node, knn_idx, knn_w)           # (T_out, X, Y, Z)

    # Tier 2 forward (한 번만)
    t2_cell, truth = tier2_forward(scen_dir, tier2_model, mask, sparse_ind,
                                    t0_seconds, device, arch=tier2_arch)

    results = []
    for w_t1 in weights_list:
        ens = w_t1 * t1_cell + (1.0 - w_t1) * t2_cell
        m6 = iou_rmse_fnr(ens[5:6], truth[5:6], mask)
        results.append({
            "name": name, "w_tier1": w_t1,
            "iou_step6": m6["iou"], "fnr_step6": m6["fnr"], "rmse_step6": m6["rmse"],
        })
    # Tier 1 alone (cell-projected) — sanity check
    m6_t1 = iou_rmse_fnr(t1_cell[5:6], truth[5:6], mask)
    print(f"  [{name}] T1_cell IoU={m6_t1['iou']:.3f}  T2_cell IoU={iou_rmse_fnr(t2_cell[5:6], truth[5:6], mask)['iou']:.3f}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier1-ckpt", type=Path,
                        default=Path("checkpoints/tier1_gnn_v3/best.pt"))
    parser.add_argument("--tier2-ckpt", type=Path,
                        default=Path("checkpoints/fno_sparse_v3/best.pt"))
    parser.add_argument("--tier2-arch", type=str, default="fno",
                        choices=["fno", "conv_lstm"],
                        help="Tier 2 architecture (default fno = 6-ch sparse FNO)")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--seq-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path, default=Path("configs/building.yaml"))
    parser.add_argument("--out-figures", type=Path,
                        default=Path("figures/current/09_ensemble"))
    parser.add_argument("--out-csv", type=Path,
                        default=Path("results/exp_ensemble/comparison.csv"))
    parser.add_argument("--t0", type=float, default=120.0)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--knn-sigma", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--geodesic-projection", action="store_true",
                        help="Use BFS geodesic distance (wall-aware) instead of Euclidean for cell-projection")
    parser.add_argument("--mask-aware-projection", action="store_true",
                        help="Mask-aware k-NN: unreachable nodes weight=0, "
                             "and adaptive σ from k-th nearest reachable distance.")
    parser.add_argument("--sigma-floor", type=float, default=2.0,
                        help="Minimum σ when --mask-aware-projection is on.")
    args = parser.parse_args()

    args.out_figures.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[setup] loading models...")
    # Tier 1 GNN
    t1_ckpt = torch.load(args.tier1_ckpt, weights_only=False, map_location=device)
    cfg = t1_ckpt.get("config", {})
    gnn_model = SimpleFireGNN(
        in_feat=5,
        hidden=cfg.get("hidden", 32),
        n_graph_layers=cfg.get("n_graph_layers", 2),
        T_out=cfg.get("T_out", 6),
    )
    gnn_model.load_state_dict(t1_ckpt["model"])
    gnn_model.to(device).eval()
    adj = build_knn_adjacency(k=cfg.get("knn_k", 4))

    # Tier 2 model (FNO or ConvLSTM)
    if args.tier2_arch == "fno":
        tier2_model = load_sparse_fno(args.tier2_ckpt, device)
    else:
        tier2_model = load_model(args.tier2_ckpt, device, "conv_lstm")
    print(f"        tier2 arch: {args.tier2_arch}, ckpt: {args.tier2_ckpt}")

    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)

    # Precompute cell ← node mapping (한 번만)
    proj_name = "geodesic" if args.geodesic_projection else "Euclidean"
    if args.mask_aware_projection:
        proj_name += " + mask-aware k-NN + adaptive σ"
    print(f"[setup] precomputing {proj_name} k-NN cell-to-node weights (k={args.knn_k})...")
    node_positions = [d.position for d in ALL_DETECTORS]
    knn_idx, knn_w = precompute_node_to_cell_weights(
        node_positions, k=args.knn_k, sigma=args.knn_sigma,
        mask=mask, use_geodesic=args.geodesic_projection,
        mask_aware=args.mask_aware_projection,
        adaptive_sigma=args.mask_aware_projection,
        sigma_floor=args.sigma_floor,
    )
    print(f"        knn_idx {knn_idx.shape}, knn_w {knn_w.shape}")

    weights_list = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    t_start = int(args.t0 // DT_SLCF) - T_IN   # T_in=6 history before t0

    scens = sorted(d for d in args.raw_root.glob("sim_*_T*") if d.is_dir())
    all_results = []
    for scen in scens:
        try:
            rs = eval_scenario_all_weights(
                scen, gnn_model, tier2_model, adj, mask, sparse_ind,
                knn_idx, knn_w, t_start, args.t0, weights_list,
                args.seq_dir, device, tier2_arch=args.tier2_arch,
            )
            all_results.extend(rs)
        except Exception as e:
            print(f"[skip] {scen.name}: {e}")

    # Aggregate per weight
    print(f"\n[agg]")
    print(f"  {'w_tier1':>10} {'Mean IoU':>10} {'Mean FNR':>10} {'H5 pass':>10}")
    agg = []
    for w in weights_list:
        subset = [r for r in all_results if r["w_tier1"] == w]
        ious = [r["iou_step6"] for r in subset]
        fnrs = [r["fnr_step6"] for r in subset]
        n_pass = sum(1 for i in ious if i >= 0.7)
        agg.append({"w_tier1": w, "mean_iou": np.mean(ious),
                    "mean_fnr": np.mean(fnrs), "h5_pass": n_pass})
        print(f"  {w:>10.2f} {np.mean(ious):>10.3f} {np.mean(fnrs)*100:>9.1f}% {n_pass:>5d}/13")

    # Save CSV
    cols = ["w_tier1", "mean_iou", "mean_fnr", "h5_pass"]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in agg:
            w.writerow([r[c] for c in cols])
    # Per-scenario CSV
    per_scen_csv = args.out_csv.parent / "per_scenario.csv"
    with per_scen_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "w_tier1", "iou_step6", "fnr_step6", "rmse_step6"])
        for r in all_results:
            w.writerow([r["name"], r["w_tier1"], r["iou_step6"],
                        r["fnr_step6"], r["rmse_step6"]])

    # Plot weight sweep
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ws = [a["w_tier1"] for a in agg]
    axes[0].plot(ws, [a["mean_iou"] for a in agg], "o-", lw=2, color="tab:blue")
    axes[0].set_xlabel("w_tier1 (= 1 - w_tier2)")
    axes[0].set_ylabel("Mean IoU @ +60s")
    axes[0].axhline(0.70, color="red", lw=0.8, ls="--", label="H5 ≥ 0.70")
    axes[0].set_title("Ensemble weight sweep — Mean IoU")
    axes[0].grid(alpha=0.3); axes[0].legend()
    for w, a in zip(ws, agg):
        axes[0].text(w, a["mean_iou"], f"{a['mean_iou']:.3f}",
                      ha="center", va="bottom", fontsize=9)
    axes[1].plot(ws, [a["mean_fnr"] * 100 for a in agg], "s-", lw=2, color="tab:red")
    axes[1].set_xlabel("w_tier1")
    axes[1].set_ylabel("Mean FNR (%) @ +60s")
    axes[1].axhline(10, color="red", lw=0.8, ls="--", label="H4 < 10%")
    axes[1].set_title("Mean FNR")
    axes[1].grid(alpha=0.3); axes[1].legend()
    axes[2].plot(ws, [a["h5_pass"] for a in agg], "^-", lw=2, color="tab:green")
    axes[2].set_xlabel("w_tier1")
    axes[2].set_ylabel("# scenarios passing H5")
    axes[2].set_title("H5 pass count (max 13)")
    axes[2].grid(alpha=0.3); axes[2].set_ylim(-0.5, 13.5)
    fig.suptitle("Tier 1 GNN + Tier 2 Sparse FNO Ensemble — weight sweep",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out_figures / "weight_sweep.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {args.out_figures / 'weight_sweep.png'}")

    # Best weight
    best = max(agg, key=lambda a: a["mean_iou"])
    print(f"\n[BEST] w_tier1={best['w_tier1']:.2f} → "
          f"IoU={best['mean_iou']:.3f}, FNR={best['mean_fnr']*100:.1f}%, "
          f"H5 pass={best['h5_pass']}/13")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
