# Track 1B — Sparse-input ConvLSTM 재학습 평가

> **목적**: 보간 단계 제거 — 모델 자체가 16 sensor 의 sparse signal 로부터 dense 60s 미래 예측을 직접 학습.

> **체크포인트**: `checkpoints\conv_lstm_sparse_v3\best.pt`
> **t₀ = 120 s, lookahead 60 s, 16 sensors**


## 1. 평균 결과 (13 OOD 시나리오)

- **IoU step 6 (60s 미래):** **0.581**
- FNR step 6: 23.0%
- RMSE step 6: 0.120
- H5 (≥ 0.70) 통과: ❌ NO

## 2. Layer-by-layer 비교

![](figures/current/07_sparse_retrain_v3_resparsify/full_stack_comparison.png)


## 3. 시나리오별 IoU

![](figures/current/07_sparse_retrain_v3_resparsify/per_scenario.png)


| 시나리오 | HRR | area | IoU step 1 | IoU step 6 | RMSE step 6 |
|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | 1000 kW | 1 m² | 0.612 | 0.629 | 0.094 |
| sim_1000kw_1m2_T03 | 1000 kW | 1 m² | 0.567 | 0.505 | 0.133 |
| sim_1000kw_2m2_T01 | 1000 kW | 2 m² | 0.756 | 0.743 | 0.102 |
| sim_1000kw_2m2_T05 | 1000 kW | 2 m² | 0.751 | 0.715 | 0.137 |
| sim_1500kw_1m2_T02 | 1500 kW | 1 m² | 0.732 | 0.687 | 0.114 |
| sim_1500kw_1m2_T03 | 1500 kW | 1 m² | 0.649 | 0.595 | 0.131 |
| sim_1500kw_2m2_T05 | 1500 kW | 2 m² | 0.745 | 0.751 | 0.161 |
| sim_500kw_1m2_T01 | 500 kW | 1 m² | 0.486 | 0.385 | 0.109 |
| sim_500kw_1m2_T02 | 500 kW | 1 m² | 0.534 | 0.438 | 0.116 |
| sim_500kw_1m2_T03 | 500 kW | 1 m² | 0.409 | 0.346 | 0.151 |
| sim_500kw_1m2_T04 | 500 kW | 1 m² | 0.513 | 0.481 | 0.105 |
| sim_500kw_2m2_T02 | 500 kW | 2 m² | 0.669 | 0.606 | 0.108 |
| sim_500kw_2m2_T05 | 500 kW | 2 m² | 0.729 | 0.676 | 0.105 |