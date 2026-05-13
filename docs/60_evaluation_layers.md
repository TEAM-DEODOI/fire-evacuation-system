# 60 — Evaluation Layer Framework (L1-L4)

> 평가 가정의 변화에 따른 IoU 격차를 정량화. **Paper 의 핵심 contribution
> framework**.

---

## 1. 핵심 아이디어

같은 ConvLSTM 체크포인트에 **평가 가정만 바꿨을 뿐인데** 60s 미래 예측의 IoU 가
**0.21 ~ 0.92** 까지 변동.

이 격차 자체가 paper 의 contribution 영역.

---

## 2. Layer 정의

| Layer | 가정 | 사용 데이터 | 사용 모델 | IoU |
|---|---|---|---|---|
| **L1** | Teacher-forced single-step | 매 step FDS truth 입력 | ConvLSTM | 0.89 |
| **L2** | Full SLCF autoregress (ideal) | 14,400 cell 모두 측정 가정 | ConvLSTM | **0.92** |
| **L3** | Detector-triggered autoregress | full SLCF + 감지 시점부터 시작 | ConvLSTM | 0.53 |
| **L4a** | Sparse + nearest interp (16 sens) | nearest neighbor | ConvLSTM | 0.28 |
| **L4b** | Sparse + linear interp | scipy.griddata linear | ConvLSTM | 0.19 |
| **L4c** | Sparse + cubic interp | scipy.griddata cubic | ConvLSTM | 0.19 |
| **L4d (16)** | Sparse + geodesic IDW (16 sens) | BFS geodesic + IDW | ConvLSTM | 0.41 |
| **L4d (39)** | Sparse + geodesic IDW (39 sens) | BFS geodesic + IDW | ConvLSTM | 0.21 |
| **L4d (39)** | Sparse + geodesic IDW (39 sens) | BFS geodesic + IDW | **FNO no-PI** | **0.43** |
| L4e (no re-sparsify) | Sparse retrain ConvLSTM, naïve chaining | 모델 재학습 | Sparse ConvLSTM | 0.182 (FNR 0% conservative bias) |
| **L4e ★ (re-sparsify)** | **Sparse retrain + autoregress re-sparsify** | 매 step sensor 외 0 강제 | Sparse ConvLSTM | **0.581** (FNR 23%, 0/13 H5) |
| **L4e' ★ (FNO 6-ch)** | **Sparse FNO + sensor indicator channel + re-sparsify** | 6-channel + Fourier basis | Sparse FNO | **0.525** (FNR **10.4%**, **4/13 H5 통과**) |
| **L4g ★★ (Ensemble)** | **Tier 1 + Tier 2 weighted ensemble (w_t1=0.6)** | k-NN cell projection + 가중평균 | GNN + Sparse FNO | **0.576** (FNR **4.8%** ✅ H4) |
| **L4f** ★ | **Tier 1 GNN binary (39 nodes)** | D-023 trigger | **SimpleFireGNN** | **0.90** ★★ |

---

## 3. Layer 진행 다이어그램

```
   IoU @ t₀+60s on OOD T01-T05
   1.0 ─┐
        │ L2 ideal SLCF + ConvLSTM           ─── 0.92  (upper bound)
        │ ★ L4f Tier 1 GNN (binary, 39 nodes) ─── 0.90  ★★★ best deployable
        │
   0.7 ─┤                              ─── H5 (≥ 0.70)
        │
        │ L3 detector-triggered                   0.53
        │ L4d FNO no-PI geodesic IDW (39 sens)    0.43
        │ L4d ConvLSTM geodesic IDW (16 sens)     0.41
        │ FNO PI geodesic IDW (39 sens)           0.32
        │ L4a sparse + nearest interp             0.28
        │ L4d ConvLSTM geodesic IDW (39 sens)     0.21
        │ L4b/c sparse + linear/cubic             0.19
        │ L4e sparse-retrain ConvLSTM (50-ep)     0.18  (FNR 0% conservative!)
   0.0 ─┘
```

---

## 4. Layer 별 측정 방법

### L1 — Teacher-forced single-step

**가정**: 매 timestep `t` 에서 FDS truth `x[t]` 입력 → 모델 출력 `y[t+1]`.
자기 출력을 다시 입력으로 chaining ❌.

**측정**: `scripts/evaluate_t_locations.py`
**결과 figures**: `figures/legacy/eval_T01_T05/`

→ 모델의 **단일 step 표현력 측정**.

### L2 — Full SLCF autoregress (Ideal)

**가정**: t₀ 에 한 번 FDS truth `x[t₀]` 주입 → 자기 출력 chaining 으로 60s 진행.
입력은 14,400 cell 모두 측정 가정 (현실 X).

**측정**: `scripts/visualize_60s_prediction.py`
**결과 figures**: `figures/current/05_future_prediction/`

→ **Autoregress 안정성 측정** (cold-start 제외 시).
→ Cold-start (t₀=0) 시 모든 모델 fail; mid-fire (t₀≥60s) 시 0.92 도달.

### L3 — Detector-triggered autoregress

**가정**: 16 detector 중 첫 trigger 시점 (T > 60°C) 부터 L2 와 동일 진행.
입력은 여전히 full SLCF.

**측정**: `scripts/evaluate_detector_triggered.py`
**결과 figures**: `figures/current/06_detector_triggered/`

→ **시스템 워크플로우 (감지→예측) 의 정확도**.
→ 평균 IoU 0.53 — H5 미달.
→ 발견: trigger 시점 ≠ 좋은 예측 시작 시점 (early trigger 는 약한 신호).

### L4 — Sparse + Interpolation + 모델 (현실 deployment)

**가정**: 39 sensor 위치만 측정값, 나머지 0.
보간 (linear / cubic / geodesic IDW) 후 모델 forward.

**측정**: `scripts/evaluate_sparse_sensing_geodesic.py` (v3.3 39-sensor)
**결과 figures**: `figures/current/03_sparse_interpolation/`

→ 모든 조합이 H5 미달. **39 sensor 의 정보 bottleneck 한계** 입증.

### L4f — Tier 1 GNN (binary)

**가정**: 39 sensor 의 **binary on/off** 신호만 사용.
보간 단계 없음, 모델 직접 처리.

**측정**: `scripts/train_tier1_gnn.py` + `scripts/visualize_tier1_predictions.py`
**결과 figures**: `figures/current/04_tier1_gnn/`

→ **IoU 0.904** — 13/13 시나리오 H5 통과 + ideal 의 98% 도달.

---

## 5. 격차 분해 — 어디서 정확도가 손실되는가

L2 → L3 → L4 단계별 손실:

| 손실 단계 | 메커니즘 | IoU 손실 |
|---|---|---|
| L1 → L2 | Teacher-forced → autoregress 누적 오차 | -0.0 (놀랍게도 미미) |
| L2 → L3 | "감지 후 시작" 운용 제약 (약한 trigger 신호) | **-0.39** |
| L3 → L4 (continuous) | Full obs → 39 sensor + 보간 | **-0.30** |
| L4 (continuous) → **L4f** | **보간 → binary GNN** | **+0.47** ★ |

→ **두 가지 큰 격차** (L2→L3, L3→L4 continuous) 를 binary GNN (L4f) 가 우회.

---

## 6. Paper headline framing

```
                                              IoU @ t₀+60s
   L1  Teacher-forced single-step                0.89    (sandbox)
   L2  Full SLCF autoregress (ideal)             0.92    ◀── upper bound
─────────────────────────────────────────────────────────────────
   L4f Tier 1 GNN binary (D-023 trigger)         0.90  ★ best deployable
                                                       (ideal 의 98%)
─────────────────────────────────────────────────────────────────
   L4d Tier 2 sparse + geodesic + FNO no-PI      0.43
   L4d Tier 2 sparse + geodesic + ConvLSTM       0.21
   L4b/c Tier 2 sparse + linear/cubic            0.19
                                                 ───
                                                 H5 threshold = 0.70
```

**제목 한 줄**:
> *"Real-world deployment 의 IoU gap (0.92 → 0.21) 을 binary signal 의 GNN
> (12K params, IoU 0.90) 으로 회복."*

---

## 7. 학술적 시사점

1. **Phase-transition 특성**: 화재 spread 는 본질적으로 discrete trigger event.
   → binary 정보 손실 < continuous interpolation 손실.

2. **Spectral basis 호환성**: FNO 의 Fourier basis 가 dense smooth interpolation 과 매칭.
   → sparse regime 에서 H3 (FNO > ConvLSTM) 의 **부분 검증**.

3. **Inductive bias 의 가치**: GNN 의 graph + temporal RNN inductive bias 가 작은 모델 (12K)
   로도 큰 모델 (1.78M) 을 능가. **Capacity < Domain match**.

4. **Legacy infrastructure 의 latent value**: 추가 hardware 없이 IoU 0.90 도달.
   → 실 deployment 가능성 입증.

---

## 8. 평가 figure 인덱스

| Layer | Figure 경로 |
|---|---|
| L1 | `figures/legacy/eval_T01_T05/aggregated_boxplots.png` (16 sensor 기반) |
| L2 | `figures/current/05_future_prediction/sim_1500kw_2m2_T05_grid_t0_120.png` |
| L3 | `figures/current/06_detector_triggered/iou_per_scenario.png` |
| L4d | `figures/current/03_sparse_interpolation/method_comparison_geodesic.png` |
| **L4f** | **`figures/current/04_tier1_gnn/headline.png`** ★ |
| L1-L4 종합 비교 | `figures/current/02_l1_l4_layers/model_comparison.png` |
