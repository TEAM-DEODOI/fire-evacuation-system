# Track 1A.5 — Mask-aware Geodesic Interpolation

> **목적**: 단순 Euclidean 보간(`scipy.griddata`)이 벽을 무시하고 다른 방의 sensor 값까지 사용하는 문제를 **geodesic IDW** 로 해결.

> **알고리즘**: BFS 로 fluid cell 만 따라 distance 누적 → IDW(p=2).


## 1. 평균 결과 (13 OOD 시나리오, t₀ = 120s)

| 방법 | 모델 | IoU step 6 | RMSE step 6 | FNR step 6 |
|---|---|---|---|---|
| sparse-linear | ConvLSTM | 0.211 | 0.432 | 35.0% |
| sparse-linear | FNO no-PI | 0.351 | 0.268 | 44.3% |
| sparse-linear | FNO PI | 0.250 | 0.360 | 41.3% |
| **sparse-geodesic** | ConvLSTM | 0.212 | 0.441 | 28.2% |
| **sparse-geodesic** | FNO no-PI | 0.431 | 0.241 | 37.4% |
| **sparse-geodesic** | FNO PI | 0.317 | 0.330 | 34.6% |

## 2. 보간 자체 quality (raw 단위, t₀=120s frame)

| 방법 | T RMSE (°C) | V RMSE (m) | CO RMSE (ppm) |
|---|---|---|---|
| sparse-linear | 33.51 | 19.39 | 36.5 |
| sparse-geodesic | 28.19 | 18.85 | 29.4 |

## 3. Figures

![](figures/sparse_sensing_geodesic_v3/method_comparison_geodesic.png)

![](figures/sparse_sensing_geodesic_v3/snapshot_T05_geodesic.png)


## 4. 결론

- ConvLSTM: geodesic vs linear IoU 차이 = **+0.001**  
- FNO no-PI: geodesic vs linear IoU 차이 = **+0.080**  
- FNO PI: geodesic vs linear IoU 차이 = **+0.068**  

→ **Geodesic 가 명확한 개선 효과**. paper 에 mask-aware interpolation 으로 채택 권장.