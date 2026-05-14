# Track 1A.5 — Mask-aware Geodesic Interpolation

> **목적**: 단순 Euclidean 보간(`scipy.griddata`)이 벽을 무시하고 다른 방의 sensor 값까지 사용하는 문제를 **geodesic IDW** 로 해결.

> **알고리즘**: BFS 로 fluid cell 만 따라 distance 누적 → IDW(p=2).


## 1. 평균 결과 (13 OOD 시나리오, t₀ = 120s)

| 방법 | 모델 | IoU step 6 | RMSE step 6 | FNR step 6 |
|---|---|---|---|---|
| sparse-linear | ConvLSTM | 0.194 | 0.265 | 69.4% |
| sparse-linear | FNO no-PI | 0.171 | 0.231 | 80.7% |
| sparse-linear | FNO PI | 0.169 | 0.239 | 80.9% |
| **sparse-geodesic** | ConvLSTM | 0.406 | 0.205 | 33.3% |
| **sparse-geodesic** | FNO no-PI | 0.375 | 0.174 | 55.3% |
| **sparse-geodesic** | FNO PI | 0.367 | 0.186 | 56.3% |

## 2. 보간 자체 quality (raw 단위, t₀=120s frame)

| 방법 | T RMSE (°C) | V RMSE (m) | CO RMSE (ppm) |
|---|---|---|---|
| sparse-linear | 33.74 | 11.35 | 30.4 |
| sparse-geodesic | 29.57 | 8.62 | 23.5 |

## 3. Figures

![](figures/sparse_sensing_geodesic/method_comparison_geodesic.png)

![](figures/sparse_sensing_geodesic/snapshot_T05_geodesic.png)


## 4. 결론

- ConvLSTM: geodesic vs linear IoU 차이 = **+0.213**  
- FNO no-PI: geodesic vs linear IoU 차이 = **+0.204**  
- FNO PI: geodesic vs linear IoU 차이 = **+0.197**  

→ **Geodesic 가 명확한 개선 효과**. paper 에 mask-aware interpolation 으로 채택 권장.