# Track 1A — Sparse Intelligent Sensors 검증

> **설정**: 16개 지능형 센서 (T/CO/V 감지) → spatial interpolation → 기존 모델 입력 → 60s 미래 예측.
> **시작 시점**: t₀ = 120 s (mid-fire, cold-start 회피)
> **건물**: 16 detector 노드 = `configs/building.yaml has_detector=True`


## 1. 평균 결과 표

| 방법 | 모델 | IoU step 6 | RMSE step 6 | FNR step 6 |
|---|---|---|---|---|
| sparse-linear | ConvLSTM | 0.194 | 0.265 | 69.4% |
| sparse-linear | FNO no-PI | 0.171 | 0.231 | 80.7% |
| sparse-linear | FNO PI | 0.169 | 0.239 | 80.9% |
| sparse-cubic | ConvLSTM | 0.194 | 0.276 | 68.7% |
| sparse-cubic | FNO no-PI | 0.192 | 0.229 | 78.7% |
| sparse-cubic | FNO PI | 0.172 | 0.242 | 80.5% |
| sparse-nearest | ConvLSTM | 0.281 | 0.258 | 55.7% |
| sparse-nearest | FNO no-PI | 0.331 | 0.246 | 48.2% |
| sparse-nearest | FNO PI | 0.286 | 0.276 | 49.8% |

## 2. 보간 자체의 quality (raw 단위, t₀ frame)

| 방법 | T RMSE (°C) | V RMSE (m) | CO RMSE (ppm) |
|---|---|---|---|
| sparse-linear | 33.74 | 11.35 | 30.4 |
| sparse-cubic | 34.84 | 11.67 | 30.7 |
| sparse-nearest | 44.13 | 13.44 | 36.6 |

## 3. Figures

![Method comparison](figures/sparse_sensing/method_comparison.png)

![Interp quality](figures/sparse_sensing/interp_quality.png)

![Snapshot](figures/sparse_sensing/snapshot_T05_1500kw_linear.png)


## 4. ★ Ideal vs Sparse 비교 (핵심 결과)

이전 평가 (full SLCF input, t₀=120s) 결과 (`docs/cold_start_finding.md` §3):

| 모델 | Ideal IoU step 6 (t₀=120s) |
|---|---|
| ConvLSTM | 0.92 |
| FNO no-PI | 0.82 |
| FNO PI | 0.89 |


Track 1A (16 sensors, linear interp) 결과 — 위 표 §1 참조.


## 5. 해석 / 결론

- 최고 조합: **nearest__FNO no-PI** (IoU 0.331)  
- 이상적 full-SLCF 대비 손실: ConvLSTM 0.92 → sparse 0.194  
- 16-sensor sparse 의 핵심 제약: spatial detail 손실. 보간이 부드러워 실제 corridor 의 sharp 패턴 재현 못함.  


### 권고
- 16-sensor 결과가 H5 (≥ 0.70) 통과 못하면: sensor 개수 증가 ablation 필요
- 또는 sparse-aware 모델 학습 (Track 1B) 검토
- 또는 paper 에서 'high-end deployment 의 lower bound' 로 결과 명시
