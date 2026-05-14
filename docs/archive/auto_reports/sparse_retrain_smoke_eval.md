# Track 1B — Sparse-input ConvLSTM 재학습 평가

> **목적**: 보간 단계 제거 — 모델 자체가 16 sensor 의 sparse signal 로부터 dense 60s 미래 예측을 직접 학습.

> **체크포인트**: `checkpoints\conv_lstm_sparse_smoke\best.pt`
> **t₀ = 120 s, lookahead 60 s, 16 sensors**


## 1. 평균 결과 (13 OOD 시나리오)

- **IoU step 6 (60s 미래):** **0.200**
- FNR step 6: 13.2%
- RMSE step 6: 0.493
- H5 (≥ 0.70) 통과: ❌ NO

## 2. Layer-by-layer 비교

![](figures/sparse_retrain_smoke/full_stack_comparison.png)


## 3. 시나리오별 IoU

![](figures/sparse_retrain_smoke/per_scenario.png)


| 시나리오 | HRR | area | IoU step 1 | IoU step 6 | RMSE step 6 |
|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | 1000 kW | 1 m² | 0.432 | 0.161 | 0.492 |
| sim_1000kw_1m2_T03 | 1000 kW | 1 m² | 0.253 | 0.163 | 0.502 |
| sim_1000kw_2m2_T01 | 1000 kW | 2 m² | 0.403 | 0.276 | 0.475 |
| sim_1000kw_2m2_T05 | 1000 kW | 2 m² | 0.338 | 0.327 | 0.467 |
| sim_1500kw_1m2_T02 | 1500 kW | 1 m² | 0.373 | 0.272 | 0.477 |
| sim_1500kw_1m2_T03 | 1500 kW | 1 m² | 0.290 | 0.220 | 0.485 |
| sim_1500kw_2m2_T05 | 1500 kW | 2 m² | 0.300 | 0.338 | 0.446 |
| sim_500kw_1m2_T01 | 500 kW | 1 m² | 0.324 | 0.066 | 0.524 |
| sim_500kw_1m2_T02 | 500 kW | 1 m² | 0.291 | 0.105 | 0.518 |
| sim_500kw_1m2_T03 | 500 kW | 1 m² | 0.215 | 0.077 | 0.539 |
| sim_500kw_1m2_T04 | 500 kW | 1 m² | 0.335 | 0.108 | 0.518 |
| sim_500kw_2m2_T02 | 500 kW | 2 m² | 0.377 | 0.226 | 0.486 |
| sim_500kw_2m2_T05 | 500 kW | 2 m² | 0.411 | 0.263 | 0.477 |