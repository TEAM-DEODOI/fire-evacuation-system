"""Precompute training data for the learned ensemble decoder (Option 2).

For each scenario in the 33 train + 13 OOD set, run GNN + Sparse-ConvLSTM +
Sparse-FNO forward at t0=120s and cache the cell-level predictions along
with FDS truth, mask, and cell coordinates.

Output: results/decoder_data/{scenario}.npz with arrays:
  gnn_cell      (T_out=6, X=60, Y=40, Z=6)  Tier 1 GNN node→cell (geodesic IDW)
  sparse_conv   (T_out=6, X=60, Y=40, Z=6)  Sparse-ConvLSTM v3 output
  sparse_fno    (T_out=6, X=60, Y=40, Z=6)  Sparse-FNO v3 (6-ch) output
  truth         (T_out=6, X=60, Y=40, Z=6)  FDS ground-truth cell danger
  mask          (X=60, Y=40, Z=6)             building fluid mask {0, 1}
  meta          dict: scenario name, t0, split (train/ood)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import build_input_tensor, normalize_scenario
from src.risk_map.converter import prediction_to_danger
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS
from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency

from evaluate_t_locations import load_mask, load_model
from evaluate_sparse_fno import (
    load_sparse_fno, build_sparse_6ch_input, autoregress_sparse_fno,
)
from evaluate_ensemble import (
    gnn_node_pred_to_cell_danger, precompute_node_to_cell_weights,
    tier1_forward,
)
from train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator
from visualize_60s_5model import (
    autoregress_sparse_input, sparsify_initial_input,
)

LOOKAHEAD_STEPS = 6
T_IN = 6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--seq-dir", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path,
                        default=Path("configs/building.yaml"))
    parser.add_argument("--gnn-ckpt", type=Path,
                        default=Path("checkpoints/tier1_gnn_v3/best.pt"))
    parser.add_argument("--conv-ckpt", type=Path,
                        default=Path("checkpoints/conv_lstm_sparse_v3/best.pt"))
    parser.add_argument("--fno-ckpt", type=Path,
                        default=Path("checkpoints/fno_sparse_v3/best.pt"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/decoder_data"))
    parser.add_argument("--t0", type=float, default=120.0)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--knn-sigma", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Models ────────────────────────────────────────────────────────────
    print(f"[setup] loading models")
    t1_ckpt = torch.load(args.gnn_ckpt, weights_only=False, map_location=device)
    cfg = t1_ckpt.get("config", {})
    gnn_model = SimpleFireGNN(
        in_feat=5, hidden=cfg.get("hidden", 32),
        n_graph_layers=cfg.get("n_graph_layers", 2),
        T_out=cfg.get("T_out", 6),
    )
    gnn_model.load_state_dict(t1_ckpt["model"])
    gnn_model.to(device).eval()
    adj = build_knn_adjacency(k=cfg.get("knn_k", 4))

    conv_model = load_model(args.conv_ckpt, device, "conv_lstm")
    fno_model = load_sparse_fno(args.fno_ckpt, device)

    mask = load_mask(args.dataset)
    sensor_idxs = load_sensor_indices(args.building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)

    print(f"[setup] precomputing geodesic k-NN weights")
    node_positions = [d.position for d in ALL_DETECTORS]
    knn_idx, knn_w = precompute_node_to_cell_weights(
        node_positions, k=args.knn_k, sigma=args.knn_sigma,
        mask=mask, use_geodesic=True,
    )

    # ── Identify scenarios (33 train + 13 OOD) ───────────────────────────
    all_dirs = sorted([d.name for d in args.raw_root.iterdir()
                       if d.is_dir() and d.name not in ("first_sim",)])
    # Train = s_000..s_032 (canonical D-024 names)
    # OOD   = sim_*_T01..T05 (13 scenarios)
    train_scens = [s for s in all_dirs if s.startswith("s_") and not s.startswith("sim_")]
    ood_scens   = [s for s in all_dirs if s.startswith("sim_")]
    print(f"[setup] {len(train_scens)} train + {len(ood_scens)} OOD scenarios")

    t0_idx = int(args.t0 // DT_SLCF)
    t_start = t0_idx - T_IN

    n_ok = 0
    for split, scen_names in [("train", train_scens), ("ood", ood_scens)]:
        for sname in scen_names:
            out_path = args.out_dir / f"{sname}.npz"
            if out_path.exists():
                print(f"  [skip-cached] {sname}")
                n_ok += 1
                continue

            sdir = args.raw_root / sname
            if not sdir.is_dir():
                print(f"  [skip-missing] {sname}")
                continue

            try:
                slices = extract_slices(sdir)
                norm = normalize_scenario(slices)
                inp = build_input_tensor(norm, mask, times=slices["times"])
                inp_6ch = build_sparse_6ch_input(slices, mask, sparse_ind)

                truth_danger = compute_total_danger(
                    slices["temperature"], slices["visibility"], slices["co"],
                ).astype(np.float32)
                if t0_idx + LOOKAHEAD_STEPS >= N_TIMESTEPS:
                    print(f"  [skip-t0late] {sname}")
                    continue
                truth_window = truth_danger[t0_idx + 1 : t0_idx + 1 + LOOKAHEAD_STEPS]

                times_arr = np.array(
                    [args.t0 + (s + 1) * DT_SLCF for s in range(LOOKAHEAD_STEPS)]
                )

                # Sparse ConvLSTM
                init_sparse = sparsify_initial_input(inp[t0_idx], sparse_ind)
                preds_norm = autoregress_sparse_input(
                    conv_model, init_sparse, sparse_ind, args.t0, device,
                )
                sparse_conv_danger = prediction_to_danger(preds_norm, times_arr)

                # Sparse FNO
                preds_norm = autoregress_sparse_fno(
                    fno_model, inp_6ch[t0_idx], sparse_ind, args.t0, device,
                    resparsify=True,
                )
                sparse_fno_danger = prediction_to_danger(preds_norm, times_arr)

                # Tier 1 GNN cell-projected (geodesic IDW)
                t1_node = tier1_forward(sname, args.seq_dir, gnn_model, adj,
                                          t_start, device)
                gnn_cell = gnn_node_pred_to_cell_danger(t1_node, knn_idx, knn_w)

                np.savez_compressed(
                    out_path,
                    gnn_cell=gnn_cell.astype(np.float32),
                    sparse_conv=sparse_conv_danger.astype(np.float32),
                    sparse_fno=sparse_fno_danger.astype(np.float32),
                    truth=truth_window.astype(np.float32),
                    mask=mask.astype(np.float32),
                    meta=np.array(
                        [sname, split, float(args.t0)], dtype=object
                    ),
                )
                n_ok += 1
                print(f"  [{split}] {sname}  -> {out_path.name}")

            except Exception as e:
                print(f"  [FAIL] {sname}: {e}")

    print(f"\n[summary] {n_ok}/{len(train_scens)+len(ood_scens)} scenarios cached -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
