# 40 — Tier 2: Continuous T/V/CO Models

> ConvLSTM / FNO no-PI / FNO PI 모델 명세. Full-SLCF (ideal) 와 Sparse-sensor
> (real deployment) 두 가지 deployment context.

---

## 1. Tier 2 시스템 개요

**입력**: 39 sensor 위치의 continuous T/V/CO 측정값 (또는 full SLCF training 시)
**출력**: (3, 60, 40, 6) per-cell future T/V/CO (10 s 단일 step, autoregress 6 회 → 60 s)
**Risk Map**: ConvLSTM/FNO 출력 → `prediction_to_danger` → StaticRiskMap

---

## 2. 학습 데이터

### 2.1 Training pairs (33 시나리오, 990 pair)

`data/processed/dataset.h5` 의 sliding pair:
- `(input[t], target[t+1])`: 한 frame → 다음 frame
- 각 scenario 30 pair, 33 scenarios → 990 pair total

### 2.2 정규화

| 채널 | Raw 단위 | Normalization |
|---|---|---|
| T | °C | `(T − 20) / 1180`, clip [0, 1] |
| V | m | `1 − clip(V / 30, 0, 1)` (inverse!) |
| CO | ppm | `log1p(CO) / log1p(5000)`, clip [0, 1] |

→ 모든 채널 [0, 1], **higher = more dangerous**.

---

## 3. 모델 아키텍처

### 3.1 ConvLSTM 3D

**구현**: `src/models/conv_lstm_3d.py`

```python
FireConvLSTM(
    in_channels=5,
    out_channels=3,
    hidden_dim=32,
    kernel_size=(3, 3, 3),
    num_layers=2,
)
# Parameters: 349,411
# Checkpoint size: 1.4 MB
```

3D conv + LSTM gate. Spatial 3D convolution (k=3×3×3) + temporal recurrence.

**Forward signature**:
```python
forward(x: (B, 5, 60, 40, 6)) -> (B, 3, 60, 40, 6)
```

### 3.2 FNO (Fourier Neural Operator)

**구현**: `src/models/fno_model.py` (wraps `neuralop.models.FNO`)

```python
FNOFireModel(
    n_modes=(12, 12, 4),         # Fourier truncation
    in_channels=5,
    out_channels=3,
    hidden_channels=32,
    n_layers=4,
    lifting_channels=128,
    projection_channels=128,
)
# Parameters: 1,780,000+
# Checkpoint size: 41 MB
```

Spectral basis (FFT) — 큰 receptive field, smooth global pattern 학습.

### 3.3 PI-FNO (Physics-Informed FNO)

**구현**: 같은 `FNOFireModel` + `src/models/pi_losses.py` (학습 시 추가 loss).

PI loss components (curriculum learning, weight ramp):
- `pde_heat`: heat diffusion residual (energy 보존)
- `pde_species`: species transport residual (CO/visibility)
- `boundary`: 벽 = no-flux 조건

`configs/pi_fno.yaml` 에 weight 명시 (data 1.0, pde_heat 0.1, pde_species 0.05, boundary 0.01).

---

## 4. 학습 결과 (33 시나리오, RunPod A100)

| 모델 | Train loss (epoch 99) | 학습 시간 | Best ckpt |
|---|---|---|---|
| ConvLSTM | 0.00104 | ~2시간 CPU | `checkpoints/conv_lstm/best.pt` |
| **FNO no-PI** | **0.00047** (50% lower!) | ~15분 A100 | `checkpoints/fno_no_pi/best.pt` |
| FNO PI | (epoch 99 ckpt) | ~15분 A100 | `checkpoints/fno_pi/best.pt` |

→ FNO 가 training distribution 에 더 잘 fit. 단 OOD generalization 은 ConvLSTM 이 우위 (H3 partial fail).

---

## 5. 평가 방식 — 2가지 Deployment Context

### 5.1 Full-SLCF (Ideal Upper Bound)

**가정**: 14,400 cell 모두 측정 가능 (현실 X, training 시 가정).
**평가**: `scripts/evaluate_t_locations.py` + `scripts/visualize_60s_prediction.py`

**결과** (13 OOD T01-T05, t₀=120s autoregress 60s):

| Model | IoU step 6 | RMSE °C | FNR step 6 |
|---|---|---|---|
| ConvLSTM | **0.92** | 5.68 | ~6% |
| FNO no-PI | 0.82 | 6.98 | ~7% |
| FNO PI | 0.89 | 6.86 | ~5% |

→ ConvLSTM 이 ideal regime 에서 최고. **H3 의 reverse** (FNO 가 generalize 더 잘 하지 못함).

### 5.2 Sparse-sensor (Real Deployment) — 39 sensors

**가정**: 39 sensor 위치만 측정값 보존, 나머지 0 → 보간 → 모델 forward.
**평가**: `scripts/evaluate_sparse_sensing_geodesic.py`

#### 보간 방법

| 방법 | 거리 metric | 벽 인식 |
|---|---|---|
| **Linear** (Euclidean) | scipy.griddata linear | ❌ 벽 무시 |
| **Geodesic IDW** | BFS on fluid cells | ✅ 벽 우회 |

#### 결과 (13 OOD, t₀=120s, +60s lookahead)

| 보간 | 모델 | IoU step 6 | RMSE step 6 | FNR step 6 |
|---|---|---|---|---|
| Linear | ConvLSTM | 0.211 | 0.432 | 38% |
| Linear | FNO no-PI | 0.351 | 0.268 | 41% |
| Linear | FNO PI | 0.250 | 0.362 | 41% |
| **Geodesic** | ConvLSTM | 0.212 | 0.443 | 25% |
| **Geodesic** | **FNO no-PI** ★ | **0.431** | **0.243** | 33% |
| **Geodesic** | FNO PI | 0.317 | 0.330 | 30% |

**핵심**:
- 13/13 시나리오 모든 조합이 H5 (0.70) 미달
- Best: FNO no-PI + geodesic = **0.431** (H5 마진 -0.27)
- **Geodesic 이 FNO 에서 +0.08 개선**, ConvLSTM 은 둔감 (+0.001)
- Linear → Geodesic: 보간 자체 RMSE °C 4° 감소

### 5.3 보간 시각화

`figures/current/03_sparse_interpolation/`:
- `method_comparison_geodesic.png` — Linear vs Geodesic 막대 비교
- `snapshot_T05_geodesic.png` — 한 시나리오의 truth vs linear interp vs geodesic interp + error

---

## 6. Sparse-input Retrain — L4e (Track 1B) ✅ 완료

**아이디어**: 모델 architecture 그대로, **input format 만 sparse 로 학습**.

**구현**: `scripts/train_sparse_conv_lstm.py`
- `SparseFireDataset`: dataset.h5 의 input 을 sparse 로 즉시 변환 (in-memory)
- 모델 (ConvLSTM) in_channels=5 그대로 (349K params)
- T/V/CO 채널을 39 sensor cell 만 nonzero
- Target 은 full dense → 모델이 sparse → dense 직접 학습

### 6.1 학습 결과 (39 sensor v3.3)

**체크포인트**: `checkpoints/conv_lstm_sparse_v3/best.pt`
- 50 epoch 학습, batch_size=4, warm-start from `conv_lstm/best.pt`
- Train MSE: 0.0187 → **0.0067** (수렴)
- Full-input ConvLSTM (0.001) 대비 6.7× — sparse 정보 손실 반영

### 6.2 OOD 평가 결과 (13 시나리오) — 두 가지 inference 방식

#### (a) Without re-sparsify chaining (default, conservative bias 발생)

| 메트릭 | Mean | 비교 |
|---|---|---|
| **IoU @ +60s** | **0.182** | L4d ConvLSTM geodesic (0.212) 보다 낮음 |
| **FNR @ +60s** | **0.0%** ★ | 모든 시나리오 0% (conservative bias) |
| RMSE step 6 | 0.708 | 높음 (over-prediction) |

#### (b) **With re-sparsify chaining ★** — autoregress distribution shift 해결

| 메트릭 | Mean | vs (a) |
|---|---|---|
| **IoU @ +60s** | **0.581** | **+0.40 (3.2× 향상)** |
| FNR @ +60s | 23.0% | +23%p (정상화) |
| **RMSE step 6** | **0.120** | -0.59 |

→ **Tier 2 sparse 의 새 best** (이전 FNO no-PI 0.43 능가). 단 여전히 H5 (0.70) 미달.

**왜 차이가 큰가** — Autoregress distribution shift:
- 학습: sparse input → dense target
- Inference (a): 자기 dense output 을 다시 입력 → 학습 분포 외 → drift 누적 → over-saturation
- Inference (b): 매 step 마다 sensor 외 cell 의 T/V/CO 를 0 으로 강제 → 학습 분포 유지
- → (b) 가 자연스러운 deployment 와도 일치 (매 10s 마다 sensor update)

**시나리오별** (`figures/current/07_sparse_retrain_v3/per_scenario.png`):
- Best: 1500kw_2m2_T05 (0.38), 1000kw_2m2_T05 (0.35)
- Worst: 500kw_1m2_T01 (0.05), 500kw_1m2_T03 (0.06)
- **13/13 H5 미달** (IoU 0.05-0.38 모두 < 0.70)

### 6.3 핵심 발견 — Conservative Bias

**FNR 0%** = 위험 영역을 한 번도 놓치지 않음.
**IoU 0.18** = 동시에 false positive 가 매우 많음 (over-prediction).

→ 모델이 sparse signal 만으로 "어디가 위험한지" 정확히 가리지 못하고
**모든 fluid cell 을 위험 (≥0.5) 으로 예측**.

**원인 분석**:
1. Sparse input 의 데이터 효율: 14,400 cell 중 234 만 nonzero (**1.6%**)
2. ConvLSTM 의 local 3D conv 가 sparse input 에 둔감 (dense 가정 architecture)
3. Training 시 dense target — sparse input 으로 정확히 매핑 어려움
4. **Safety-critical 측면에서는 valuable**: "위험 놓치지 않음" 보장.
   단 path planning 의 cost function 으로는 부적합 (모든 edge 가 위험으로 가중).

### 6.4 vs 보간 + 기존 학습 (L4d)

| 평가 | IoU | FNR | RMSE |
|---|---|---|---|
| L4d Sparse + Geodesic + **기존** ConvLSTM | 0.212 | 25% | 0.443 |
| L4e Sparse-retrain ConvLSTM (재학습) | 0.182 | **0.0%** | 0.708 |
| L4d Sparse + Geodesic + FNO no-PI (best Tier 2) | **0.431** | 33% | 0.243 |

→ **재학습이 IoU 측면에서는 손해**. FNO no-PI + geodesic 이 Tier 2 best.
→ **재학습의 가치는 FNR 0% (over-cautious bias)** — paper limitation 으로 명시.

### 6.5 시각화 — Conservative Bias 명확히 보임

`figures/current/05_future_prediction/<scenario>_grid_5model_t0_120.png` (3개):
5-row grid (FDS truth + ConvLSTM + FNO no-PI + FNO PI + **L4e Sparse ConvLSTM**)
× 6 col (t₀+10s ~ +60s).

**관찰** (re-sparsify 적용 후):
- Row 1-4 (full-input 모델들): truth 와 시각적으로 유사
- Row 5 (L4e Sparse ConvLSTM): truth 와 유사하게 회복 (L-013 fix 효과)

생성 스크립트: `scripts/visualize_60s_5model.py`

### 6.5b 6-row 통합 비교 (Sparse FNO 6-channel 추가)

`figures/current/05_future_prediction/<scenario>_grid_6model_t0_120.png` (3개):
6-row grid — 5-row + **L4e' Sparse FNO (6-channel)** 행 추가.

**관찰**:
- Row 6 (Sparse FNO 6-ch): building 구조 sharp 하게 보존, 시간 지나도 stable
- T05 1500kW 2m²: corridor 패턴 매우 정확 (4 시나리오 H5 통과)
- T01 500kW 1m²: fire spread 위치 잡지만 noisy

생성 스크립트: `scripts/visualize_60s_6model.py`

## 7. Sparse FNO (L4e') — 6-channel input + FNO architecture

**구현**: `scripts/train_sparse_fno.py` + `scripts/evaluate_sparse_fno.py`

### 7.1 두 가지 변경 통합

| 변경 | 목적 |
|---|---|
| **Input 5→6 채널**: sensor_indicator 추가 | 모델이 "어디가 measurement vs derived" 명시적으로 학습 |
| **Architecture: ConvLSTM → FNO** | Fourier basis 가 dense smooth pattern 과 호환 (L4d 입증) |

### 7.2 학습 (100 epoch, cold-start)

**체크포인트**: `checkpoints/fno_sparse_v3/best.pt` (14 MB, 1.79M params)
- Train MSE: 0.0348 → **0.00510** (수렴, 50 epoch 시점부터 plateau)
- 학습 시간: ~3시간 (RunPod 또는 사용자 환경)

### 7.3 OOD 평가 결과 (13 시나리오, t₀=120s, re-sparsify chaining)

| 메트릭 | Mean | vs L4e Sparse ConvLSTM |
|---|---|---|
| **IoU @ +60s** | **0.525** | -0.06 (살짝 낮음) |
| **FNR @ +60s** | **10.4%** ★ | -12.6%p (절반 이하) |
| RMSE step 6 | 0.156 | +0.04 |
| **H5 통과 시나리오** | **4/13** | (ConvLSTM 0/13) |

→ **4 시나리오에서 H5 (0.70) 통과**: 1000kw_2m2_T01/T05, 1500kw_2m2_T05, 500kw_2m2_T05
→ 강한 화재 + 큰 면적 (2 m²) 시나리오에서 우수

### 7.4 시나리오별 결과

| 시나리오 | HRR | area | IoU step 6 | FNR | H5? |
|---|---|---|---|---|---|
| **1000kw_2m2_T01** | 1000 | 2 m² | **0.754** | 8.2% | ✅ |
| **1000kw_2m2_T05** | 1000 | 2 m² | **0.754** | 24.0% | ✅ |
| **500kw_2m2_T05** | 500 | 2 m² | **0.745** | 9.4% | ✅ |
| **1500kw_2m2_T05** | 1500 | 2 m² | **0.715** | 28.4% | ✅ |
| 1500kw_1m2_T02 | 1500 | 1 m² | 0.689 | 12.6% | (close) |
| 500kw_2m2_T02 | 500 | 2 m² | 0.600 | 9.1% | ❌ |
| 1500kw_1m2_T03 | 1500 | 1 m² | 0.566 | 12.7% | ❌ |
| 1000kw_1m2_T01 | 1000 | 1 m² | 0.494 | 2.0% | ❌ |
| 1000kw_1m2_T03 | 1000 | 1 m² | 0.436 | 9.7% | ❌ |
| 500kw_1m2_T04 | 500 | 1 m² | 0.322 | 2.8% | ❌ |
| 500kw_1m2_T02 | 500 | 1 m² | 0.311 | 4.0% | ❌ |
| 500kw_1m2_T03 | 500 | 1 m² | 0.225 | 11.4% | ❌ |
| 500kw_1m2_T01 | 500 | 1 m² | 0.211 | 1.4% | ❌ |

→ 패턴: **면적 (1 m² vs 2 m²) 이 IoU 의 dominant factor**. HRR 보다 면적이 더 큰 영향.

### 7.5 Sparse FNO vs Sparse ConvLSTM 정밀 비교

| 측면 | Sparse ConvLSTM (L4e) | Sparse FNO (L4e') |
|---|---|---|
| Input channels | 5 | **6 (sensor indicator 추가)** |
| Architecture | 3D ConvLSTM | Fourier Neural Operator |
| Params | 349K | **1.79M (5×)** |
| Mean IoU | 0.581 | 0.525 |
| **Mean FNR** | 23.0% | **10.4%** ★ |
| H5 통과 시나리오 | 0/13 | **4/13** |
| 학습 epoch | 50 | 100 |
| 학습 시간 | ~2-3시간 CPU | ~3시간 RunPod (또는 CPU) |

**해석**:
- ConvLSTM IoU 가 살짝 높지만, **FNO 는 H5 임계를 *일부* 통과** (binary pass/fail 관점에서 우위)
- FNO 의 FNR 10.4% — H4 (< 10%) 임계 거의 통과
- **5× 큰 모델 capacity 가 sparse regime 에서 marginal gain**: 33 시나리오로는 FNO 의 풀 잠재력 발휘 어려움

### 7.6 결론 — Tier 2 sparse 의 두 가지 best

| 사용처 | 선택 |
|---|---|
| Best IoU | **L4e Sparse ConvLSTM + re-sparsify** (0.581) |
| Best FNR / 시나리오별 H5 통과 | **L4e' Sparse FNO 6-ch + re-sparsify** (10.4%, 4/13) |
| Best overall (모든 경우) | **L4f Tier 1 GNN** (0.904, 11/13 FNR<10%) |

Tier 1 GNN 은 여전히 단독 best. 단 Tier 2 sparse 도 paper 의 **충실한 비교군**으로 발전:
- Conservative bias (L-013) 발견 + 해결
- Sparse-aware 재학습으로 보간 baseline 의 2× 도달
- Sensor indicator channel 의 효과 정량화 (FNR 개선 → 안전한 운용)

### 6.6 결론

Sparse-input retrain 자체는 H5 (IoU ≥ 0.70) 달성 못 함. 단:
- **Conservative output 의 한 사례** — safety-critical regime 에서 valuable
- Sparse signal 의 fundamental information bottleneck 확인
- 5-row 비교 figure 가 paper 의 *limitation + safety discussion* 의 visual evidence
- **Tier 1 GNN (IoU 0.90) 이 여전히 best deployment 선택지**

---

## 7. 핵심 관찰 — Tier 2 의 한계

### 7.1 정보 bottleneck

39 sensor × 6 z = 234 cells / 14,400 = **1.6%** 만 nonzero T/V/CO.
보간 단계에서 정보 손실 발생 — 모델 capacity 보다 dominant.

### 7.2 ConvLSTM 의 둔감성

| Sensor 수 | ConvLSTM geodesic IoU |
|---|---|
| 16 | 0.41 |
| 27 | 0.41 |
| 39 | 0.21 |

→ 39 sensor 에서 오히려 IoU 하락. 새 위치 분포가 ConvLSTM 의 learned spatial pattern 과 mismatch.

### 7.3 FNO 가 sparse regime 에서 우위

| Sensor 수 | FNO no-PI geodesic IoU |
|---|---|
| 16 | 0.375 |
| 27 | 0.313 |
| **39** | **0.431** |

→ Fourier basis 가 더 dense 한 보간 결과를 자연스럽게 처리. H3 의 **부분 검증** (sparse regime 에서 FNO > ConvLSTM).

---

## 8. Tier 2 의 사용처

| 시나리오 | 권장 |
|---|---|
| Paper 의 ideal upper bound | ConvLSTM (IoU 0.92) |
| 39 sensor sparse deployment | **FNO no-PI + geodesic IDW** (IoU 0.43) |
| Drone 의 임의 위치 query | Tier 1 (per-node) + Tier 2 (cell) ensemble |
| 단독 추천 (대부분 시나리오) | **Tier 1 GNN** (Tier 2 sparse 보다 2.1× 우수) |

→ Tier 2 는 paper 의 *비교군* + drone 의 high-resolution 보강 용도. 단독 deployment 는 Tier 1 권장.
