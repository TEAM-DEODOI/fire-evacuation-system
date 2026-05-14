# Track 1A.5 — Mask-aware Geodesic Interpolation

> **목적**: 단순 Euclidean 보간(`scipy.griddata`)이 벽을 무시하고 다른 방의 sensor 값까지 사용하는 문제를 **geodesic IDW** 로 해결.

> **알고리즘**: BFS 로 fluid cell 만 따라 distance 누적 → IDW(p=2).


## 1. 평균 결과 (13 OOD 시나리오, t₀ = 120s)

| 방법 | 모델 | IoU step 6 | RMSE step 6 | FNR step 6 |
|---|---|---|---|---|
| sparse-linear | ConvLSTM | 0.230 | 0.309 | 52.1% |
| sparse-linear | FNO no-PI | 0.256 | 0.228 | 68.3% |
| sparse-linear | FNO PI | 0.232 | 0.257 | 68.7% |
| **sparse-geodesic** | ConvLSTM | 0.407 | 0.217 | 29.5% |
| **sparse-geodesic** | FNO no-PI | 0.313 | 0.185 | 63.6% |
| **sparse-geodesic** | FNO PI | 0.318 | 0.195 | 62.3% |

## 2. 보간 자체 quality (raw 단위, t₀=120s frame)

| 방법 | T RMSE (°C) | V RMSE (m) | CO RMSE (ppm) |
|---|---|---|---|
| sparse-linear | 30.18 | 13.93 | 29.5 |
| sparse-geodesic | 23.59 | 9.73 | 20.8 |

## 3. Figures

![](figures/sparse_sensing_geodesic_27/method_comparison_geodesic.png)

![](figures/sparse_sensing_geodesic_27/snapshot_T05_geodesic.png)


## 4. 결론

- ConvLSTM: geodesic vs linear IoU 차이 = **+0.177**  
- FNO no-PI: geodesic vs linear IoU 차이 = **+0.056**  
- FNO PI: geodesic vs linear IoU 차이 = **+0.085**  

→ **Geodesic 가 명확한 개선 효과**. paper 에 mask-aware interpolation 으로 채택 권장.