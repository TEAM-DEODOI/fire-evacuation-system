"""8-row 60s autoregress comparison — extends 6-model figure with Tier 1 GNN
cell-projected (geodesic) and 3-way Balanced ensemble (paper headline).

Rows (top to bottom):
1. FDS truth                                       (ground truth)
2. ConvLSTM (full input)                           — L2 upper bound (0.92 IoU)
3. FNO no-PI (full input)                          — L2 baseline
4. FNO PI  (full input)                            — L2 baseline
5. Sparse-ConvLSTM v3 (re-sparsify)                — L4e (0.581)
6. Sparse-FNO v3 (6-ch + sensor indicator)         — L4e' (0.525, FNR 10.4%)
7. Tier 1 GNN cell-projected (geodesic IDW, k=3)   — L4f cell (Tier 1 alone @ cell)
8. 3-way Balanced ensemble (0.5/0.25/0.25, geo)    — L4g ★★★ paper main (0.618, FNR 5.1%)

Output: figures/current/05_future_prediction/<scen>_grid_8row_t0_<NNN>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import (
    build_input_tensor, normalize_scenario,
)
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS
from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency

from evaluate_t_locations import load_model, load_mask
from evaluate_sparse_fno import (
    load_sparse_fno, build_sparse_6ch_input, autoregress_sparse_fno,
)
from evaluate_ensemble import (
    gnn_node_pred_to_cell_danger, precompute_node_to_cell_weights,
    tier1_forward,
)
from train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator
from visualize_60s_5model import (
    autoregress_full_input, autoregress_sparse_input, sparsify_initial_input,
)

Z_IDX = 3
LOOKAHEAD_STEPS = 6
T_IN = 6
# 3-way balanced weights (paper default — L4g)
W_GNN = 0.50
W_CONV = 0.25
W_FNO = 0.25


def plot_nrow(
    truth_danger: np.ndarray,
    preds_by_model: Dict[str, np.ndarray],
    scenario_name: str,
    t0_seconds: float,
    out_path: Path,
) -> None:
    n_cols = truth_danger.shape[0]
    rows = ["FDS truth"] + list(preds_by_model.keys())
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))

    for col in range(n_cols):
        t_label = f"t = {t0_seconds + (col + 1) * DT_SLCF:.0f} s"
        im = axes[0, col].imshow(
            truth_danger[col, :, :, Z_IDX].T,
            origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
            extent=[0, 30, 0, 20], aspect="equal",
        )
        axes[0, col].set_title(t_label, fontsize=10)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        for r, name in enumerate(preds_by_model.keys(), start=1):
            axes[r, col].imshow(
                preds_by_model[name][col, :, :, Z_IDX].T,
                origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
                extent=[0, 30, 0, 20], aspect="equal",
            )
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])

    # Bold + slightly larger labels for the two key rows (GNN-cell, Ensemble)
    for r, label in enumerate(rows):
        is_key = label.startswith("3-way") or label.startswith("Tier 1 GNN")
        axes[r, 0].set_ylabel(
            label, fontsize=12 if is_key else 11,
            rotation=90, labelpad=12,
            fontweight="bold",
            color="darkred" if is_key else "black",
        )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="danger ∈ [0, 1]")

    fig.suptitle(
        f"60 s future risk forecast — 8-row comparison  |  {scenario_name}\n"
        f"t0 = {t0_seconds:.0f} s, z = 1.75 m, autoregressive rollout (6 × 10 s)\n"
        f"Bottom two rows (red labels): Tier 1 GNN @ cell + 3-way Balanced Ensemble (paper main)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 0.95])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--seq-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path,
                        default=Path("configs/building.yaml"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/current/05_future_prediction"))
    parser.add_argument("--scenarios", type=str, nargs="+",
                        default=["sim_1500kw_2m2_T05",
                                 "sim_500kw_1m2_T01",
                                 "sim_1000kw_1m2_T03"])
    parser.add_argument("--t0", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--knn-sigma", type=float, default=5.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[setup] device={device}")

    # ── Models ────────────────────────────────────────────────────────────
    print(f"[setup] loading 3 full-input models + 2 sparse + GNN")
    full_models = {
        "ConvLSTM":  load_model(Path("checkpoints/conv_lstm/best.pt"),
                                 device, "conv_lstm"),
        "FNO no-PI": load_model(Path("checkpoints/fno_no_pi/best.pt"),
                                 device, "fno"),
        "FNO PI":    load_model(Path("checkpoints/fno_pi/best.pt"),
                                 device, "fno"),
    }
    sparse_conv = load_model(Path("checkpoints/conv_lstm_sparse_v3/best.pt"),
                              device, "conv_lstm")
    sparse_fno = load_sparse_fno(Path("checkpoints/fno_sparse_v3/best.pt"),
                                  device)

    gnn_ckpt = torch.load(Path("checkpoints/tier1_gnn_v3/best.pt"),
                           weights_only=False, map_location=device)
    cfg = gnn_ckpt.get("config", {})
    gnn_model = SimpleFireGNN(
        in_feat=5,
        hidden=cfg.get("hidden", 32),
        n_graph_layers=cfg.get("n_graph_layers", 2),
        T_out=cfg.get("T_out", 6),
    )
    gnn_model.load_state_dict(gnn_ckpt["model"])
    gnn_model.to(device).eval()
    adj = build_knn_adjacency(k=cfg.get("knn_k", 4))

    # ── Inputs/masks ──────────────────────────────────────────────────────
    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)

    print(f"[setup] precomputing geodesic k-NN cell→node weights (k={args.knn_k})")
    node_positions = [d.position for d in ALL_DETECTORS]
    knn_idx, knn_w = precompute_node_to_cell_weights(
        node_positions, k=args.knn_k, sigma=args.knn_sigma,
        mask=mask, use_geodesic=True,
    )

    t_start = int(args.t0 // DT_SLCF) - T_IN

    for sname in args.scenarios:
        sdir = args.raw_root / sname
        if not sdir.is_dir():
            print(f"[skip] {sname}: not a directory")
            continue
        print(f"[scen] {sname}")

        slices = extract_slices(sdir)
        norm = normalize_scenario(slices)
        inp = build_input_tensor(norm, mask, times=slices["times"])
        inp_6ch = build_sparse_6ch_input(slices, mask, sparse_ind)

        truth_danger = compute_total_danger(
            slices["temperature"], slices["visibility"], slices["co"],
        ).astype(np.float32)
        t0_idx = int(args.t0 // DT_SLCF)
        if t0_idx + LOOKAHEAD_STEPS >= N_TIMESTEPS:
            print(f"  skip: t0 too late")
            continue
        truth_window = truth_danger[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

        times_arr = np.array([args.t0 + (s + 1) * DT_SLCF
                                for s in range(LOOKAHEAD_STEPS)])

        preds: Dict[str, np.ndarray] = {}

        # 1–3) Full-input models
        for name, model in full_models.items():
            preds_norm = autoregress_full_input(model, inp[t0_idx],
                                                  args.t0, device)
            preds[name] = prediction_to_danger(preds_norm, times_arr)

        # 4) Sparse-ConvLSTM v3
        init_sparse = sparsify_initial_input(inp[t0_idx], sparse_ind)
        preds_norm = autoregress_sparse_input(sparse_conv, init_sparse,
                                                sparse_ind, args.t0, device)
        sparse_conv_danger = prediction_to_danger(preds_norm, times_arr)
        preds["Sparse-ConvLSTM (v3)"] = sparse_conv_danger

        # 5) Sparse-FNO v3 (6-ch)
        preds_norm = autoregress_sparse_fno(sparse_fno, inp_6ch[t0_idx],
                                              sparse_ind, args.t0, device,
                                              resparsify=True)
        sparse_fno_danger = prediction_to_danger(preds_norm, times_arr)
        preds["Sparse-FNO (6-ch v3)"] = sparse_fno_danger

        # 6) Tier 1 GNN cell-projected (geodesic IDW)
        t1_node = tier1_forward(sname, args.seq_dir, gnn_model, adj,
                                  t_start, device)               # (T_out, N)
        gnn_cell = gnn_node_pred_to_cell_danger(t1_node, knn_idx, knn_w)
        preds["Tier 1 GNN @ cell (geo)"] = gnn_cell

        # 7) 3-way Balanced ensemble (geodesic)
        ens = W_GNN * gnn_cell + W_CONV * sparse_conv_danger + W_FNO * sparse_fno_danger
        ens = np.clip(ens, 0.0, 1.0).astype(np.float32)
        preds[f"3-way Ensemble (w={W_GNN},{W_CONV},{W_FNO} geo) ★"] = ens

        # ─── Mask-out solid (non-fluid) cells for visual fairness ─────
        # Reason: GNN k-NN IDW does not respect the building mask. Geodesic
        # distance is infinite into solid cells → fallback to equal weight
        # → solid cells get the mean GNN node danger (~0.5). The 3-way
        # ensemble then renders solid regions as yellow (~0.25), while
        # other rows are trained with the mask channel and output ~0 there.
        # IoU/FNR metrics are already mask-filtered, so this only fixes
        # the visualization — solid → exactly 0.0 (deep green).
        fluid_mask = (mask > 0.5).astype(np.float32)         # (X, Y, Z)
        for k in list(preds.keys()):
            preds[k] = preds[k] * fluid_mask[None, ...]
        truth_window = truth_window * fluid_mask[None, ...]

        out_path = args.out_dir / f"{sname}_grid_8row_t0_{int(args.t0):03d}.png"
        plot_nrow(truth_window, preds, sname, args.t0, out_path)
        print(f"  -> {out_path}")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
