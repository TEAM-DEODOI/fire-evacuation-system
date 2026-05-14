# 70 — Results Summary (모든 실험 종합)

> 모든 실험 결과를 한 문서로. 가설/모델/평가 layer 별 정리.

---

## 1. 한 줄 요약

> **Tier 1 GNN (binary signal, 12K params): IoU 0.904 — ideal upper bound 의 98%
> 달성, 39 화재 감지기 인프라만으로 즉시 배포 가능.**

---

## 2. 전체 결과 표 — Layer × Model

| Layer | Model | Mean IoU @ +60s | Mean FNR | RMSE step 6 | H5 | H4 |
|---|---|---|---|---|---|---|
| L1 (teacher-forced) | ConvLSTM | 0.89 | ~6% | — | ✅ | ✅ |
| L1 | FNO no-PI | 0.83 | ~7% | — | ✅ | ✅ |
| L1 | FNO PI | 0.84 | ~5% | — | ✅ | ✅ |
| L2 (full SLCF, mid-fire) | ConvLSTM | **0.92** | — | 0.08 | ✅ | ✅ |
| L2 | FNO no-PI | 0.82 | — | 0.10 | ✅ | ✅ |
| L2 | FNO PI | 0.89 | — | 0.09 | ✅ | ✅ |
| L3 (detector-triggered) | ConvLSTM | 0.53 | 37% | 0.13 | ❌ | ❌ |
| L3 | FNO no-PI | 0.55 | 42% | 0.11 | ❌ | ❌ |
| L3 | FNO PI | 0.48 | 49% | 0.13 | ❌ | ❌ |
| L4 sparse + linear (39) | ConvLSTM | 0.21 | 38% | 0.43 | ❌ | ❌ |
| L4 sparse + linear | FNO no-PI | 0.35 | 41% | 0.27 | ❌ | ❌ |
| L4 sparse + linear | FNO PI | 0.25 | 41% | 0.36 | ❌ | ❌ |
| L4 sparse + geodesic IDW (39) | ConvLSTM | 0.21 | 25% | 0.44 | ❌ | ❌ |
| L4 sparse + geodesic IDW | **FNO no-PI** | **0.43** | 33% | 0.24 | ❌ | ❌ |
| L4 sparse + geodesic IDW | FNO PI | 0.32 | 30% | 0.33 | ❌ | ❌ |
| L4e Sparse-retrain ConvLSTM (no re-sparsify) | Sparse ConvLSTM | 0.182 | 0.0% (conservative) | 0.71 | ❌ | ✅ |
| **L4e Sparse-retrain + re-sparsify ★** | **Sparse ConvLSTM** | **0.581** | 23.0% | 0.12 | ❌ | ❌ |
| **L4e' Sparse FNO 6-ch + re-sparsify ★** | **Sparse FNO** | **0.525** | **10.4%** ★ | 0.16 | ❌ (4/13 ✅) | ⚠ close |
| L4g 2-way Ensemble (GNN+FNO, w_t1=0.6) Euclidean | GNN + Sparse FNO | 0.576 | 4.8% ✅ | — | (4/13 ✅) | ✅ H4 |
| L4g 2-way Ensemble (GNN+FNO, w_t1=0.6) **Geodesic** | GNN + Sparse FNO | 0.569 | **4.2%** ✅ | — | (4/13 ✅) | ✅ H4 |
| L4g 2-way Ensemble (GNN+ConvLSTM, w_t1=0.4) | GNN + Sparse ConvLSTM | 0.619 | 15.0% | — | (4/13 ✅) | ❌ |
| **L4g 3-way Ensemble balanced (Euclid)** | **GNN + ConvLSTM + FNO (0.5/0.25/0.25)** | **0.621** | **6.4%** ✅ | — | **(5/13 ✅)** | **✅ H4** |
| **L4g 3-way Ensemble balanced (Geodesic) ★★★** | **GNN + ConvLSTM + FNO (0.5/0.25/0.25)** | **0.618** | **5.1%** ✅ | — | **(5/13 ✅)** | **✅ H4** |
| L4g 3-way Ensemble max IoU (Euclid) | GNN + ConvLSTM + FNO (0.4/0.45/0.15) | **0.625** | 10.8% | — | (4/13 ✅) | ⚠ close |
| L4g 3-way Ensemble max IoU (Geodesic) | GNN + ConvLSTM + FNO (0.4/0.6/0.0) | 0.624 | 14.1% | — | (4/13 ✅) | ❌ |
| **L4g 3-way Min FNR (Geodesic) ★★** | **GNN + ConvLSTM + FNO (0.6/0.1/0.3)** | 0.590 | **3.7%** ★ ✅ | — | (4/13 ✅) | **✅ H4** |
| **L4f Tier 1 GNN binary** | **SimpleFireGNN** | **0.904** ★ | **4.6%** | — | **✅** | **✅** |
| **L4h Learned Decoder fn=1.0** | PerCell MLP 1.4K params | 0.727 | 14.9% | — | (9/13 ✅) | ❌ |
| **L4h Learned Decoder fn=2.5 ★★★** | **PerCell MLP 1.4K params** | **0.733** | 11.5% | — | **(9/13 ✅)** | (8/13 ✅) |
| **L4h Learned Decoder fn=4.0** | PerCell MLP 1.4K params | 0.718 | **10.0%** | — | (8/13 ✅) | (8/13 ✅) |

---

## 3. 가설 검증 결과

| ID | 가설 | 목표 | 측정 | 통과? | 출처 |
|---|---|---|---|---|---|
| H1 | Speed ≥ 1000× FDS | < 50 ms | GNN 1,670,749×, Decoder 220,851×, **full L4h pipeline 3,028×** (456 ms) | ✅ | `figures/current/13_h1_speed/` |
| H2 | RelL2 ≤ 0.15 | training scenarios | **0.136** (ConvLSTM) | ✅ | `figures/legacy/eval_convlstm/` |
| H3 | FNO > ConvLSTM on OOD | EXP-FIRE-001 | full SLCF ❌, sparse 39 ✅ FNO 우위 | **⚠ 부분** | `figures/current/02_l1_l4_layers/` |
| H4 | Risk FNR < 10% | OOD | **GNN 4.6%** ★, Decoder fn=4.0 10.0%, ensemble 3.7-6.4% | ✅ | `figures/current/11_decoder_ensemble/` |
| H5 | Risk IoU ≥ 0.70 | OOD | **GNN 0.904** ★ (13/13), **Decoder fn=2.5 0.733** (9/13) | ✅ | `figures/current/04_tier1_gnn/`, `11_decoder_ensemble/` |
| H6 | Dynamic A* FED ≥ 30% ↓ | EXP-PATH-001 | path planning 미구현 | **🔜** | — |

### 3.1 H1 inference latency detail (single CPU core)

| Module | Params | Mean (ms) | std | Speedup vs FDS (23 min) |
|---|---|---|---|---|
| Tier 1 GNN (single forward) | 12 K | 0.83 | ±0.19 | **1,670,749×** |
| Sparse-ConvLSTM v3 (6-step rollout) | 349 K | 193.8 | ±12.0 | 7,122× |
| Sparse-FNO v3 (6-step rollout) | 1.79 M | 237.4 | ±3.1 | 5,813× |
| Learned Decoder (full grid forward) | 1.4 K | 6.25 | ±0.48 | **220,851×** |
| **Full L4h pipeline (end-to-end)** | 2.15 M | **456** | ±14 | **3,028×** |

All modules ≥ H1 threshold. Real-time H6 replan budget (~30 s) trivially satisfied.

### 3.2 Decoder robustness verifications (H6-prep)

| Check | Method | Result |
|---|---|---|
| **5-fold CV gap** | hold-out 7 train scenarios per fold, retrain decoder | mean train-OOD gap **-0.003** (std 0.025), OOD IoU std 0.008 → no overfit |
| **Multi-t₀ sweep** | trained at t₀=120s, eval on t₀ ∈ {60,90,120,150,180,210}s | t₀ ≥ 90s flat (IoU 0.726-0.736). t₀=150s best (IoU 0.736, FNR 9.9% — H4 pass). Cold-start t₀=60s fail (design boundary). |
| **Hand-engineered ceiling** | mask-aware k-NN + adaptive σ on hand-crafted ensemble | IoU 0.618 → 0.611 (Δ=-0.007). Hand projection plateaued — justifies learned decoder. |

---

## 4. Tier 1 vs Tier 2 직접 비교 (39 sensor, 동일 인프라)

| 측면 | Tier 2 (best: FNO no-PI + geodesic) | **Tier 1 GNN binary** |
|---|---|---|
| Mean IoU @ +60s | 0.431 | **0.904** ★ (+0.47) |
| Mean FNR | 33% | **4.6%** (-28%p) |
| H5 통과 시나리오 | **0/13** | **13/13** |
| H4 통과 시나리오 | 0/13 | 11/13 |
| 모델 params | 1.78M | **12K** (**150× smaller**) |
| 추론 시간 | ~27 ms | < 1 ms 예상 |
| 입력 신호 | 39 점 continuous T/V/CO | 39 점 binary on/off |
| 추가 단계 | spatial interpolation | 없음 (직접 사용) |
| 추가 hardware | 지능형 센서 설치 | **없음 (기존 화재 감지기)** |
| 출력 해상도 | (3, 60, 40, 6) per-cell | (39, 6) per-node |

→ **Tier 1 이 Tier 2 의 2.1배 정확 + 150× 가벼움 + 추가 hardware 0**.

---

## 5. 시나리오별 결과 (13 OOD)

### 5.1 Tier 1 GNN (best.pt @ epoch 25)

| 시나리오 | HRR | 면적 | IoU step 6 | FNR step 6 |
|---|---|---|---|---|
| sim_1500kw_2m2_T05 | 1500 | 2 m² | **1.000** ★ | 0.0% |
| sim_500kw_2m2_T02 | 500 | 2 m² | 0.969 | 0.0% |
| sim_1500kw_1m2_T02 | 1500 | 1 m² | 0.941 | 0.0% |
| sim_500kw_1m2_T01 | 500 | 1 m² | 0.933 | 0.0% |
| sim_1000kw_1m2_T01 | 1000 | 1 m² | 0.931 | 3.6% |
| sim_1000kw_2m2_T05 | 1000 | 2 m² | 0.923 | 7.7% |
| sim_1000kw_1m2_T03 | 1000 | 1 m² | 0.900 | 5.3% |
| sim_500kw_2m2_T05 | 500 | 2 m² | 0.909 | 3.2% |
| sim_500kw_1m2_T03 | 500 | 1 m² | 0.909 | 9.1% |
| sim_1000kw_2m2_T01 | 1000 | 2 m² | 0.886 | 11.4% ⚠ |
| sim_1500kw_1m2_T03 | 1500 | 1 m² | 0.857 | 14.3% ⚠ |
| sim_500kw_1m2_T02 | 500 | 1 m² | 0.810 | 5.6% |
| sim_500kw_1m2_T04 | 500 | 1 m² | 0.789 | 0.0% |
| **평균** | | | **0.904** | **4.6%** |

**13/13 H5 통과**, 2/13 만 H4 (FNR<10%) 미달.

### 5.2 Tier 2 (best: FNO no-PI + geodesic IDW)

| 시나리오 | IoU step 6 | FNR step 6 |
|---|---|---|
| sim_1000kw_1m2_T01 | 0.532 | 0.23 |
| sim_1500kw_2m2_T05 | 0.432 | 0.35 |
| sim_500kw_1m2_T01 | 0.429 | 0.39 |
| 평균 (13개) | **0.431** | **33%** |

→ **모두 H5 미달**.

---

## 5b. Sparse-retrain ConvLSTM (L4e) — 추가 발견

50-epoch warm-started, 39-sensor sparse input:

| 시나리오 | IoU step 6 | 비고 |
|---|---|---|
| 1500kw_2m2_T05 | 0.38 | best |
| 1000kw_2m2_T05 | 0.35 | |
| 1000kw_2m2_T01 | 0.24 | |
| 1500kw_1m2_T02 | 0.24 | |
| 500kw_2m2_T05 | 0.24 | |
| 500kw_2m2_T02 | 0.19 | |
| 1500kw_1m2_T03 | 0.18 | |
| 1000kw_1m2_T01 | 0.13 | |
| 1000kw_1m2_T03 | 0.13 | |
| 500kw_1m2_T04 | 0.09 | |
| 500kw_1m2_T02 | 0.08 | |
| 500kw_1m2_T03 | 0.06 | |
| 500kw_1m2_T01 | 0.05 | worst |
| **Mean** | **0.182** | All FNR 0% (conservative bias) |

**핵심 발견**:
- IoU 측면: 모든 시나리오 H5 (0.70) 미달
- FNR 측면: **모든 시나리오 0%** — 위험 영역 한 번도 놓치지 않음
- 즉 모델이 over-prediction (대부분 fluid cell → 위험)
- Safety-critical 측면에서는 *valuable conservative*, path planning cost 로는 부적합

**Tier 1 GNN (IoU 0.90, FNR 5%) 이 deployment 선택지로 우위 확정.**

상세: [`40_tier2_models_continuous.md`](40_tier2_models_continuous.md) §6.

---

## 6. 평면도 & Sensor 인프라

### 6.1 D-024 v3.3 — 39 sensors

| 영역 | 개수 |
|---|---|
| Zone B (남측) | 5 rooms |
| Zone A (사선 통로 안) | 5 rooms |
| North (북측 상단) | 4 rooms |
| Zone C (중간 row Y=15) | 5 rooms |
| Zone D (X=18) | 3 rooms |
| South corridor (Y=5) | 6 |
| East corridor (X=16.5) | 2 |
| Diagonal corridor | 3 |
| North corridor (Y=16) | 3 |
| Exits | 3 |
| **TOTAL** | **39** |

상세 좌표: [`30_sensor_infrastructure.md`](30_sensor_infrastructure.md)

### 6.2 D-023 — 트리거 모델

| 임계 | 값 | 근거 |
|---|---|---|
| Heat | T > 60 °C | NFPA 57°C + KOFEIS 70°C 중간값 |
| Smoke | V < 10 m | UL 268 13m 의 보수적 사용 |
| 조합 | OR (latched) | 실 감지기 모사 |
| CO | 제외 | UL 2034 누적 노출 임계 시뮬 시간 부족 |

---

## 7. 핵심 figures (Paper 용)

### Paper Figure 1 — Tier 1 GNN headline
**경로**: `figures/current/04_tier1_gnn/headline.png`
2 row (truth/pred) × 3 col (t₀+10s, +30s, +60s) of T05 1500kW 2m².
→ 시각적으로 GNN pred 와 FDS truth 거의 구분 불가.

### Paper Figure 2 — Per-scenario IoU
**경로**: `figures/current/04_tier1_gnn/aggregate_iou.png`
13 OOD 시나리오의 IoU + FNR 막대.

### Paper Figure 3 — L1-L4 Layer 비교
**경로**: `figures/current/02_l1_l4_layers/model_comparison.png`
3 모델 (ConvLSTM/FNO no-PI/PI) × 4 metrics.

### Paper Figure 4 — 39 sensor 평면도
**경로**: `figures/current/01_sensor_layout/sensor_layout.png`
색상별 sensor 종류 + 출구 + 벽.

### Paper Figure 5 — Geodesic vs Linear 보간
**경로**: `figures/current/03_sparse_interpolation/snapshot_T05_geodesic.png`
6-panel: truth + linear + geodesic + 2 error map + mask.

### Paper Figure 6 — 60s autoregress 4-model 비교
**경로**: `figures/current/05_future_prediction/sim_1500kw_2m2_T05_grid_t0_120.png`
4 row (truth/ConvLSTM/FNO no-PI/FNO PI) × 6 col (10s..60s).

### Paper Figure 7 — 60s autoregress **5-model** 비교 (L4e 포함)
**경로**: `figures/current/05_future_prediction/<scenario>_grid_5model_t0_120.png` (3 시나리오)
5 row (+ Sparse-ConvLSTM L4e) × 6 col.
→ **L4e 의 conservative bias** 시각적으로 명확히 보임 (시간 지나며 도메인 전체가 빨강 saturate).

---

## 8. 인프라 인덱스

### 8.1 학습된 체크포인트 (git 추적)

| 모델 | 경로 | Size | Best metric |
|---|---|---|---|
| ConvLSTM | `checkpoints/conv_lstm/best.pt` | 1.4 MB | train_loss 0.001 |
| FNO no-PI | `checkpoints/fno_no_pi/best.pt` | 41 MB | train_loss 0.0005 |
| FNO PI | `checkpoints/fno_pi/best.pt` | 41 MB | train_loss 0.0005 |
| **Tier 1 GNN v3** | **`checkpoints/tier1_gnn_v3/best.pt`** | **53 KB** | **OOD IoU 0.904** |
| Sparse-ConvLSTM v3 | `checkpoints/conv_lstm_sparse_v3/best.pt` | 1.4 MB | OOD IoU 0.581 (re-sparsify) |
| Sparse-FNO v3 (6-ch) | `checkpoints/fno_sparse_v3/best.pt` | 14 MB | OOD IoU 0.525 |
| **L4h Decoder fn=2.5 ★** | **`checkpoints/ensemble_decoder/best.pt`** | **12 KB** | **OOD IoU 0.733** (paper default) |
| L4h Decoder fn=1.0 (BCE) | `checkpoints/ensemble_decoder_fn10/best.pt` | 12 KB | OOD IoU 0.727 |
| L4h Decoder fn=2.5 | `checkpoints/ensemble_decoder_fn25/best.pt` | 12 KB | OOD IoU 0.733 |
| L4h Decoder fn=4.0 (safety) | `checkpoints/ensemble_decoder_fn40/best.pt` | 12 KB | OOD IoU 0.718, FNR 10.0% ✅H4 |

### 8.2 데이터 (gitignored, 외부 전송)

| 항목 | 크기 |
|---|---|
| `data/raw/*` (46 시나리오) | 5.7 GB |
| `data/processed/dataset.h5` | 221 MB |
| `results/detector_sequences/` (46 .npz) | 280 KB |

### 8.3 평가 결과 CSV

| 경로 | 내용 |
|---|---|
| `results/eval_T01_T05/` | L1 ConvLSTM teacher-forced |
| `results/eval_T01_T05_fno_no_pi/` | L1 FNO no-PI |
| `results/eval_T01_T05_fno_pi/` | L1 FNO PI |
| `results/exp_detector_triggered/` | L3 trigger-start |
| `results/exp_sparse_sensing/` | L4 linear/cubic/nearest |
| `results/exp_sparse_sensing_geodesic/` | L4 geodesic (16 sensor) |
| `results/exp_sparse_sensing_geodesic_27/` | L4 geodesic (27 sensor) |
| `results/exp_sparse_sensing_geodesic_v3/` | **L4 geodesic (39 sensor, current)** |
| `results/exp_fire_001/comparison.csv` | 3-model comparison |
| `results/detector_sequences/` | Tier 1 GNN 입력 |

---

## 9. Plan B 활성화 — Paper Reframing

**기존 Plan B** (CLAUDE.md): *"PI-FNO doesn't beat ConvLSTM → '30-scenario regime trade-offs' 로 reframe"*.

**우리의 갱신된 framing**:
> *"단일 39-detector 인프라 위에서 binary signal + lightweight GNN (12K params)
> 이 continuous signal + heavy models (1.78M) 을 2.1× 능가하며 ideal upper
> bound 의 98% 도달. Phase-transition 도메인에서는 inductive bias matching
> 이 capacity 보다 dominant."*

H3 실패 → 더 강한 framing (system-level deployment 가능성) 으로 reframe.
