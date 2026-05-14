# REPRODUCE — End-to-end Command Index

> This is the command-only recipe. For project context read `CLAUDE.md`,
> for results context read `docs/CURRENT_SESSION_STATE.md`, for decision
> rationale read `docs/decisions.md`.

All commands run from the repo root on Windows/PowerShell or
Linux/macOS with Python 3.10+. Adjust quoting accordingly.

---

## 0. Environment

```bash
# CPU is enough for everything except FNO training (RunPod A100 used).
python -m venv venv
. venv/bin/activate                    # or .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Tested with PyTorch 2.0+, neuraloperator, fdsreader, scipy, h5py,
matplotlib, NetworkX. No torch-geometric needed — Tier 1 GNN is
PyTorch-only (`SimpleFireGNN`).

---

## 1. Data

```bash
# Expected layout (gitignored — distribute via cloud):
data/raw/                  # 33 train (s_000..s_032) + 13 OOD (sim_*_T*)
data/processed/dataset.h5  # 33-scenario tensor cache, 221 MB

# Build the binary detector sequences (Tier 1 GNN input) for all 46 sims.
python scripts/build_detector_sequences.py
# -> results/detector_sequences/{name}.npz  (280 KB total)
```

---

## 2. Train surrogates

```bash
# Full-input models  (RunPod A100, ~30 min each)
python scripts/train_conv_lstm.py
python scripts/train_fno.py --pi-mode off  --out checkpoints/fno_no_pi/
python scripts/train_fno.py --pi-mode full --out checkpoints/fno_pi/

# Tier 1 GNN  (12 K params, CPU, ~5 min)
python scripts/train_tier1_gnn.py
# -> checkpoints/tier1_gnn_v3/best.pt   (val IoU 0.872, OOD IoU 0.904)

# Tier 2 sparse retrains  (D-042 re-sparsify fix is built in; was D-025 pre-merge)
python scripts/train_sparse_conv_lstm.py
# -> checkpoints/conv_lstm_sparse_v3/best.pt   (IoU 0.581 with re-sparsify)
python scripts/train_sparse_fno.py
# -> checkpoints/fno_sparse_v3/best.pt         (IoU 0.525, FNR 10.4%)
```

---

## 3. Learned ensemble decoder (L4h)

```bash
# Precompute forward outputs of the 3 surrogates on 33 train + 13 OOD
# at t0=120s.  (Cache ~15 MB, gitignored — regenerate as needed.)
python scripts/precompute_decoder_data.py

# Train decoder.  fn_weight=2.5 = paper default (D-045; was D-028 pre-merge).
python scripts/train_ensemble_decoder.py --fn-weight 2.5
# -> checkpoints/ensemble_decoder/best.pt        (IoU 0.733, FNR 11.5%)

# Sweep variants (paper ablation):
python scripts/train_ensemble_decoder.py --fn-weight 1.0 \
    --ckpt-dir checkpoints/ensemble_decoder_fn10 \
    --out-results results/exp_decoder_ensemble_fn10
python scripts/train_ensemble_decoder.py --fn-weight 4.0 \
    --ckpt-dir checkpoints/ensemble_decoder_fn40 \
    --out-results results/exp_decoder_ensemble_fn40
# fn=1.0 → IoU 0.727 / FNR 14.9% (BCE-only)
# fn=4.0 → IoU 0.718 / FNR 10.0% (safety-friendly, H4 pass)
```

---

## 4. Evaluation

```bash
# Layer comparisons (L1, L2, L3, L4 — produces "model_comparison.png")
python scripts/evaluate_t_locations.py                      # L1 teacher-forced
python scripts/visualize_60s_prediction.py                  # L2 full-SLCF rollout
python scripts/evaluate_detector_triggered.py               # L3 trigger-start
python scripts/evaluate_sparse_sensing_geodesic.py          # L4d sparse + geodesic
python scripts/evaluate_sparse_model.py     --resparsify    # L4e sparse-ConvLSTM
python scripts/evaluate_sparse_fno.py                       # L4e' sparse-FNO (resparsify=True default)
python scripts/visualize_tier1_predictions.py               # L4f GNN per-node + headline

# Ensemble grid searches
python scripts/evaluate_ensemble.py        --geodesic-projection         # L4g 2-way
python scripts/evaluate_ensemble_3way.py   --geodesic-projection         # L4g 3-way

# Learned-decoder vs hand-crafted comparison
python scripts/visualize_decoder_comparison.py
# -> figures/current/11_decoder_ensemble/  (frontier + per-scenario + fn-sweep)

# Multi-t0 robustness check (H6 prep)
python scripts/evaluate_decoder_multi_t0.py
# -> figures/current/12_decoder_multi_t0/  (verifies decoder stable on t0 ∈ [90, 210]s)

# H1 hypothesis verification
python scripts/measure_h1_inference.py
# -> figures/current/13_h1_speed/inference_latency.png
#    Full L4h pipeline = 456 ms / 3028× faster than FDS.

# 5-fold CV overfit check
python scripts/decoder_cv_overfit.py --epochs 20
# -> results/exp_decoder_cv/cv_summary.csv
```

---

## 5. Visualizations

```bash
# Paper figures (each script saves under figures/current/<NN>_*/):
python scripts/visualize_sensor_layout.py        # 01 — 39 sensor map
python scripts/visualize_l1_to_l4h_layers.py     # 02 — L1-L4h bars + staircase ★
python scripts/visualize_60s_5model.py           # 05 — 5-row 60 s rollout
python scripts/visualize_60s_6model.py           # 05 — 6-row (+ Sparse-FNO)
python scripts/visualize_60s_8row.py             # 05 — 8-row (+ GNN cell, hand ensemble)
python scripts/visualize_60s_9row.py             # 05 — 9-row (+ Learned Decoder ★ paper)
python scripts/visualize_tier1_gnn_alone.py      # 04 — Tier 1 GNN per-node alone
```

---

## 6. H6 path planning (next session)

```bash
# Pending modules — not yet implemented:
#   src/path_planning/edge_weights.py
#   src/path_planning/planners.py
#   src/path_planning/evacuation_sim.py
#   experiments/exp_path_001.py
# RiskMap adapter is already in place:
#   src/tier1/tier1_risk_map.py        (option α, per-node GNN)
#   src/tier1/ensemble_risk_map.py     (option β/γ, learned decoder)

# Future entry point will look like:
# python experiments/exp_path_001.py \
#     --scenarios sim_1500kw_2m2_T05 sim_500kw_1m2_T01 sim_1000kw_1m2_T03 \
#     --planners dijkstra static dynamic \
#     --risk-map decoder-fn25 \
#     --start-positions 8
```

---

## Key checkpoints (all whitelisted in `.gitignore`)

| Checkpoint | Size | Best metric |
|---|---|---|
| `checkpoints/conv_lstm/best.pt`              | 1.4 MB | train_loss 0.001 |
| `checkpoints/fno_no_pi/best.pt`              | 41 MB  | train_loss 0.0005 |
| `checkpoints/fno_pi/best.pt`                 | 41 MB  | train_loss 0.0005 |
| `checkpoints/tier1_gnn_v3/best.pt`           | 53 KB  | OOD IoU 0.904 |
| `checkpoints/conv_lstm_sparse_v3/best.pt`    | 1.4 MB | OOD IoU 0.581 |
| `checkpoints/fno_sparse_v3/best.pt`          | 14 MB  | OOD IoU 0.525 |
| `checkpoints/ensemble_decoder/best.pt`       | 12 KB  | OOD IoU 0.733 (fn=2.5 default) |
| `checkpoints/ensemble_decoder_fn10/best.pt`  | 12 KB  | OOD IoU 0.727 (BCE) |
| `checkpoints/ensemble_decoder_fn25/best.pt`  | 12 KB  | OOD IoU 0.733 (paper) |
| `checkpoints/ensemble_decoder_fn40/best.pt`  | 12 KB  | OOD IoU 0.718 (safety) |

Total: ~98 MB of tracked checkpoints. Everything else (FDS raw data,
processed dataset.h5, decoder data cache) is regenerated from these
plus the FDS scenarios.
