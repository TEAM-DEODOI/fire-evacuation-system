# Track 1B — Sparse-input ConvLSTM 재학습 평가

> **목적**: 보간 단계 제거 — 모델 자체가 16 sensor 의 sparse signal 로부터 dense 60s 미래 예측을 직접 학습.

> **체크포인트**: `checkpoints\conv_lstm_sparse_v3\best.pt`
> **t₀ = 120 s, lookahead 60 s, 16 sensors**


## 1. 평균 결과 (13 OOD 시나리오)

- **IoU step 6 (60s 미래):** **0.182**
- FNR step 6: 0.0%
- RMSE step 6: 0.708
- H5 (≥ 0.70) 통과: ❌ NO

## 2. Layer-by-layer 비교

![](figures/current/07_sparse_retrain_v3/full_stack_comparison.png)


## 3. 시나리오별 IoU

![](figures/current/07_sparse_retrain_v3/per_scenario.png)


| 시나리오 | HRR | area | IoU step 1 | IoU step 6 | RMSE step 6 |
|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | 1000 kW | 1 m² | 0.612 | 0.131 | 0.712 |
| sim_1000kw_1m2_T03 | 1000 kW | 1 m² | 0.567 | 0.130 | 0.725 |
| sim_1000kw_2m2_T01 | 1000 kW | 2 m² | 0.756 | 0.236 | 0.684 |
| sim_1000kw_2m2_T05 | 1000 kW | 2 m² | 0.751 | 0.352 | 0.661 |
| sim_1500kw_1m2_T02 | 1500 kW | 1 m² | 0.732 | 0.237 | 0.688 |
| sim_1500kw_1m2_T03 | 1500 kW | 1 m² | 0.649 | 0.183 | 0.701 |
| sim_1500kw_2m2_T05 | 1500 kW | 2 m² | 0.745 | 0.378 | 0.632 |
| sim_500kw_1m2_T01 | 500 kW | 1 m² | 0.486 | 0.053 | 0.751 |
| sim_500kw_1m2_T02 | 500 kW | 1 m² | 0.534 | 0.085 | 0.745 |
| sim_500kw_1m2_T03 | 500 kW | 1 m² | 0.409 | 0.064 | 0.766 |
| sim_500kw_1m2_T04 | 500 kW | 1 m² | 0.513 | 0.086 | 0.746 |
| sim_500kw_2m2_T02 | 500 kW | 2 m² | 0.669 | 0.189 | 0.703 |
| sim_500kw_2m2_T05 | 500 kW | 2 m² | 0.729 | 0.237 | 0.688 |