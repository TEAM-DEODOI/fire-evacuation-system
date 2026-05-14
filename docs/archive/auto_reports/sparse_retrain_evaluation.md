# Track 1B — Sparse-input ConvLSTM 재학습 평가

> **목적**: 보간 단계 제거 — 모델 자체가 16 sensor 의 sparse signal 로부터 dense 60s 미래 예측을 직접 학습.

> **체크포인트**: `checkpoints\conv_lstm_sparse\best.pt`
> **t₀ = 120 s, lookahead 60 s, 16 sensors**


## 1. 평균 결과 (13 OOD 시나리오)

- **IoU step 6 (60s 미래):** **0.071**
- FNR step 6: 83.4%
- RMSE step 6: 0.427
- H5 (≥ 0.70) 통과: ❌ NO

## 2. Layer-by-layer 비교

![](figures/sparse_retrain/full_stack_comparison.png)


## 3. 시나리오별 IoU

![](figures/sparse_retrain/per_scenario.png)


| 시나리오 | HRR | area | IoU step 1 | IoU step 6 | RMSE step 6 |
|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | 1000 kW | 1 m² | 0.409 | 0.100 | 0.411 |
| sim_1000kw_1m2_T03 | 1000 kW | 1 m² | 0.282 | 0.023 | 0.422 |
| sim_1000kw_2m2_T01 | 1000 kW | 2 m² | 0.563 | 0.108 | 0.434 |
| sim_1000kw_2m2_T05 | 1000 kW | 2 m² | 0.572 | 0.108 | 0.457 |
| sim_1500kw_1m2_T02 | 1500 kW | 1 m² | 0.519 | 0.092 | 0.435 |
| sim_1500kw_1m2_T03 | 1500 kW | 1 m² | 0.340 | 0.042 | 0.433 |
| sim_1500kw_2m2_T05 | 1500 kW | 2 m² | 0.513 | 0.108 | 0.462 |
| sim_500kw_1m2_T01 | 500 kW | 1 m² | 0.242 | 0.053 | 0.402 |
| sim_500kw_1m2_T02 | 500 kW | 1 m² | 0.269 | 0.030 | 0.413 |
| sim_500kw_1m2_T03 | 500 kW | 1 m² | 0.190 | 0.007 | 0.422 |
| sim_500kw_1m2_T04 | 500 kW | 1 m² | 0.307 | 0.064 | 0.402 |
| sim_500kw_2m2_T02 | 500 kW | 2 m² | 0.460 | 0.074 | 0.425 |
| sim_500kw_2m2_T05 | 500 kW | 2 m² | 0.573 | 0.113 | 0.428 |