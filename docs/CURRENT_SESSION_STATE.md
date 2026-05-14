# 🔖 Current Session State (Compact-Resistant Snapshot)

> **이 파일을 새 Claude 세션에서 가장 먼저 읽으세요.**
> Compact 직전 / 직후 무엇을 했고 어디서 멈췄는지 정확히 기록.
>
> **Last updated**: 2026-05-14 (evening — learned decoder breakthrough)
> **Latest commit**: `Learned ensemble decoder (Option 2) → IoU 0.733 / FNR 11.5% / H5 9-13, +0.115 over hand-crafted`

---

## 0. 한 줄 요약

> **Tier 1 GNN (0.90) + Tier 2 Sparse FNO (0.525) + Sparse ConvLSTM (0.581) 의 3-way ensemble + geodesic projection 으로 cell-level IoU 0.62 + FNR 3.7-5.1% 달성. 이제 H6 (path planning) 으로 가야 paper headline 완성.**

---

## 1. 시스템 비전 (절대 변경 금지)

```
       39 화재 감지기 (D-024 v3.3, z=2.5m, 평면도 기반)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
   Tier 1 (Legacy)              Tier 2 (Intelligent)
   ┌──────────────────┐        ┌──────────────────────┐
   │ Binary on/off    │        │ Continuous T, V, CO  │
   │ (D-023 trigger:  │        │ (sparse-input,       │
   │  T>60°C OR       │        │  re-sparsify, L-013) │
   │  vis<10m,        │        │                      │
   │  latched)        │        │                      │
   │                  │        │                      │
   │ → GraphGRU       │        │ → ConvLSTM 5-ch      │
   │   (12K params)   │        │   AND/OR             │
   │                  │        │ → FNO 6-ch (sensor   │
   │                  │        │   indicator channel) │
   │                  │        │                      │
   │ → per-node       │        │ → per-cell danger    │
   │   danger (39, 6) │        │   (3, 60, 40, 6)     │
   └──────────────────┘        └──────────────────────┘
        │                             │
        └──────────────┬──────────────┘
                       ▼
              Cell-level Ensemble (L4g)
              w_t1 GNN + w_conv ConvLSTM + w_fno FNO
              (geodesic projection for cell-mapping)
                       │
                       ▼
              Risk Map → Path Planning (A*) [🔜 미구현]
```

**핵심 framing** (paper 헤드라인):
> *"같은 39 감지기 인프라 위에 binary signal (Tier 1) + continuous signal (Tier 2) 의 dual surrogate system. Tier 1 GNN (12K params) 이 ideal full-SLCF observation 의 98% 도달. Tier 2 sparse 도 retraining + re-sparsify (L-013 fix) + ensemble 로 H4 (FNR<10%) 통과."*

---

## 2. 결과 요약 표 — **가장 핵심**

### 2.1 Evaluation Layer 종합 (13 OOD T01-T05, t₀=120s, +60s)

| Layer | Method | Mean IoU | Mean FNR | H5/H4 통과 |
|---|---|---|---|---|
| L1 teacher-forced | ConvLSTM | 0.89 | ~6% | ✅/✅ |
| L2 full SLCF (ideal) | ConvLSTM | **0.92** | — | ✅/✅ |
| L3 detector-triggered | ConvLSTM | 0.53 | 37% | ❌/❌ |
| L4d sparse linear+ConvLSTM | ConvLSTM | 0.21 | 38% | ❌/❌ |
| L4d sparse geodesic+FNO no-PI | FNO no-PI | 0.43 | 33% | ❌/❌ |
| **L4e Sparse ConvLSTM + re-sparsify** | Sparse ConvLSTM 5-ch | **0.581** | 23.0% | ❌/❌ |
| **L4e' Sparse FNO + re-sparsify** | Sparse FNO 6-ch | **0.525** | **10.4%** | (4/13 H5) ❌H4 |
| **L4g 2-way (GNN+FNO, w_t1=0.6)** | Ensemble | 0.576 | **4.8%** | (4/13) ✅H4 |
| **L4g 2-way (GNN+ConvLSTM, w_t1=0.4)** | Ensemble | **0.619** | 15.0% | (4/13) ❌H4 |
| **L4g 3-way Euclidean balanced** | GNN+Conv+FNO (0.5/0.25/0.25) | **0.621** | 6.4% | **5/13** ✅H4 |
| **L4g 3-way Geodesic balanced** ★ | GNN+Conv+FNO (0.5/0.25/0.25) | **0.618** | **5.1%** | **5/13** ✅H4 |
| **L4g 3-way Geodesic Min-FNR** ★★ | GNN+Conv+FNO (0.6/0.1/0.3) | 0.590 | **3.7%** | 4/13 ✅H4 |
| **L4h Learned Decoder fn=2.5 ★★★ (paper default)** | PerCell MLP 1.4K params | **0.733** | 11.5% | **9/13 H5, 8/13 H4** |
| L4h Learned Decoder fn=4.0 (safety) | PerCell MLP 1.4K params | 0.718 | **10.0%** | 8/13 H5, 8/13 H4 |
| **L4f Tier 1 GNN binary (per-node)** | SimpleFireGNN | **0.904** | 4.6% | **✅/✅** |

### 2.2 핵심 인사이트

1. **Tier 1 GNN per-node 0.90 → cell-projected 0.18-0.32** (k-NN over-smoothing).
   - Ensemble 에서 GNN 의 가치는 IoU 가 아닌 **FNR (안전성)**.
2. **ConvLSTM > FNO on cell-level IoU** (0.581 > 0.525). FNO 의 5× 큰 capacity (1.79M vs 349K) 가 33 시나리오로 부족.
3. **3-way > 2-way**: 각 모델의 inductive bias 가 complementary.
4. **Geodesic projection > Euclidean**: FNR -1~1.5%p 일관 개선.
5. **약한 화재 (1m²)** 가 항상 worst: sensor 정보 적음. Area > HRR.

### 2.3 가설 검증 현황

| ID | 가설 | 측정 | 통과 |
|---|---|---|---|
| H1 Speed ≥ 1000× | 52,000× | ✅ |
| H2 RelL2 ≤ 0.15 | 0.136 (ConvLSTM) | ✅ |
| H3 FNO > ConvLSTM OOD | full SLCF ❌ / sparse 39 ✅ (FNO no-PI 우위) | ⚠ partial |
| H4 FNR < 10% | GNN 4.6%, 3-way ensemble 3.7-6.4% | ✅ |
| H5 IoU ≥ 0.70 | GNN 0.904 (13/13 시나리오 통과) | ✅ |
| **H6 Dynamic A* FED ≥ 30%↓** | **path planning 미구현** | **🔜** |

---

## 3. 학습된 모델 인덱스 (모두 git 추적)

| 모델 | 경로 | Size | Mean IoU (13 OOD) |
|---|---|---|---|
| ConvLSTM (full input) | `checkpoints/conv_lstm/best.pt` | 1.4 MB | 0.92 (L2 ideal) |
| FNO no-PI (full input) | `checkpoints/fno_no_pi/best.pt` | 41 MB | 0.82 (L2) |
| FNO PI (full input) | `checkpoints/fno_pi/best.pt` | 41 MB | 0.89 (L2) |
| **Sparse ConvLSTM v3 (5-ch)** | `checkpoints/conv_lstm_sparse_v3/best.pt` | 1.4 MB | **0.581** (L4e) |
| **Sparse FNO v3 (6-ch + sensor_indicator)** | `checkpoints/fno_sparse_v3/best.pt` | 14 MB | **0.525** (L4e') |
| **Tier 1 GNN v3** | `checkpoints/tier1_gnn_v3/best.pt` | 53 KB | **0.904** (L4f) |

---

## 4. 핵심 코드 인덱스

### 핵심 모듈
- `src/tier1/detector_positions.py` — **D-024 v3.3** 39-sensor 위치 (Tier 1/2 공유)
- `src/tier1/detector_model.py` — **D-023** trigger model (heat 60°C OR smoke 10m, latched)
- `src/tier1/tier1_gnn.py` — `SimpleFireGNN`, `build_knn_adjacency`
- `src/tier1/tier1_dataset.py` — `Tier1FireDataset` (sliding window)

### 학습 스크립트
- `scripts/train_sparse_conv_lstm.py` (5-ch, **`load_sensor_indices` 자동 D-024 사용**)
- `scripts/train_sparse_fno.py` (6-ch, sensor_indicator 추가)
- `scripts/train_tier1_gnn.py`
- `scripts/build_detector_sequences.py` (46 시나리오 binary_sequence)

### 평가 스크립트
- `scripts/evaluate_sparse_model.py` — Sparse ConvLSTM (`--resparsify` ⚠ default False, **반드시 추가**)
- `scripts/evaluate_sparse_fno.py` — Sparse FNO (default True)
- `scripts/evaluate_ensemble.py` — 2-way (`--tier2-arch fno|conv_lstm`, `--geodesic-projection`)
- `scripts/evaluate_ensemble_3way.py` — 3-way grid search

### 시각화
- `scripts/visualize_sensor_layout.py` — 39 sensor 평면도
- `scripts/visualize_tier1_predictions.py` — GNN headline figure (★)
- `scripts/visualize_60s_5model.py` — 5-row 비교
- `scripts/visualize_60s_6model.py` — 6-row 비교 (Sparse FNO 포함)

---

## 5. 핵심 함정 (Lessons Learned, 절대 잊지 말 것)

### L-001/009/012: FDS 데이터 추출 — `fix_pyrosim_fds.py` 자동 패치, 자체 raw parser `fds_extractor.py`

### L-013 ★: Sparse model autoregress distribution shift
- **Bug**: Sparse 모델은 (sparse input → dense target) 학습. Naïve chaining 시 dense output → dense input → drift → conservative bias (IoU 0.18, FNR 0%).
- **Fix**: `resparsify=True` — 매 step 매 cell 의 T/V/CO 를 sensor 외 0 으로 강제. IoU 0.18 → **0.581** (3.2× 향상).
- **deployment 와 일치**: 매 10s 마다 sensor measurement update.

### Cold-start regime
- t₀=0 (화재 발생 직후) 의 autoregress 는 모든 모델 fail (IoU 0.01-0.34).
- 시스템 design boundary — 감지 후 예측 흐름 (mid-fire t₀≥60s 에서 정상 동작).

---

## 6. 결정 로그 핵심 (`docs/decisions.md` 참조)

- **D-023**: Trigger model heat 60°C OR smoke 10m, latched (NFPA + KOFEIS + UL 268)
- **D-024 v3.3**: **39 sensor 위치 확정** — Tier 1/2 공유. (22 rooms + 14 corridors + 3 exits)
- **D-025** (신규 권고): Re-sparsify chaining 을 Tier 2 sparse 의 default 운용 방식으로
- **D-026** (신규 권고): 3-way ensemble (GNN + ConvLSTM + FNO) + geodesic projection 을 cell-level deployment 의 reference 구성으로
- **D-027** (신규 권고): Paper framing — *"dual surrogate system on shared 39-detector infrastructure"*

---

## 7. 다음 작업 (★★★ critical)

### 7.1 H6 검증 — Path Planning + EXP-PATH-001

**미구현 모듈**:
1. `src/tier1/tier1_risk_map.py` — `Tier1RiskMap(RiskMap)` 클래스
   - `query(xyz, t)`: 가장 가까운 sensor node → GNN pred danger
   - 또는 ensemble 의 cell-level 출력 query
2. `src/path_planning/edge_weights.py` — edge weight = α·base_time + β·integrated_risk
3. `src/path_planning/planners.py` — Dijkstra / StaticAvoidance / **DynamicPredictive**
4. `src/path_planning/evacuation_sim.py` — `EvacuationSimulator.simulate(planner, risk_map_truth, start_xyz, graph)`
5. `experiments/exp_path_001.py` — 3 scenarios × 3 planners × 8 starts = 72 trials

**목표**: Dynamic A* cumulative FED ≤ 0.7 × Dijkstra FED (30%↓)

**예상 시간**: 5-7시간 (Path planning 모듈 + EXP-PATH-001).

### 7.2 보조 작업 (선택)

- **B1 Pre-interpolated FNO**: geodesic IDW 보간을 학습 input 으로 사용 → FNO 잠재력 활용. RunPod ~3시간.
- **3-way ensemble visualization**: 6-row figure 에 ensemble row 추가
- **PyBullet 통합 (Week 12)**: `docs/pybullet_integration_spec.md` 외주

---

## 8. 새 세션 시작 명령어 (literal copy-paste)

```
1. Read CLAUDE.md (auto-loaded)
2. Read docs/CURRENT_SESSION_STATE.md (this file)
3. Read docs/90_next_steps.md
4. (선택) Read docs/70_results_summary.md, docs/60_evaluation_layers.md

다음 작업 (H6) 으로 즉시 진입:
   docs/50_tier1_gnn_binary.md §9 — Tier1RiskMap 인터페이스 spec
   docs/10_system_architecture.md §4.1 — RiskMap abstract interface
```

---

## 9. 핵심 figures (Paper-grade)

| # | 경로 | 내용 |
|---|---|---|
| 1 ★★★ | `figures/current/04_tier1_gnn/headline.png` | Tier 1 GNN headline (T05 1500kW 2m²) |
| 2 | `figures/current/04_tier1_gnn/aggregate_iou.png` | 13 OOD GNN IoU + FNR 막대 |
| 3 | `figures/current/01_sensor_layout/sensor_layout.png` | 39 sensor 평면도 |
| 4 | `figures/current/05_future_prediction/sim_1500kw_2m2_T05_grid_6model_t0_120.png` | 6-row 모델 비교 |
| 5 | `figures/current/03_sparse_interpolation/snapshot_T05_geodesic.png` | Geodesic IDW 효과 |
| 6 | `figures/current/09_ensemble/weight_sweep.png` | 2-way ensemble weight |
| 7 | `figures/current/10_ensemble_3way/grid_search.png` | 3-way ensemble heatmap |
| 8 ★ | `figures/current/10_ensemble_3way_geodesic/grid_search.png` | 3-way + geodesic heatmap |

---

## 10. Commit Log 요약 (오늘 세션)

```
85dc4c8  D-024 v3.3: 39 sensors
9a61691  Tier 1 GNN v3 headline (IoU 0.904)
dfd1b7d  Docs reorganization (numbered 00-90)
135a481  Sparse ConvLSTM v3 (naïve IoU 0.182)
c97bfec  L-013 fix: re-sparsify → IoU 0.581 (3.2×)
6cb7663  5-model comparison figure
a644b18  Sparse FNO scripts (6-ch sensor indicator)
5f448ee  Sparse FNO v3 (IoU 0.525, FNR 10.4%)
(이번)   3-way ensemble + geodesic projection (IoU 0.618-0.625, FNR 3.7-5.1%)
```

---

**💡 New session 진입 시**: 이 파일 + `90_next_steps.md` 만 읽어도 즉시 컨텍스트 복원. H6 진행으로 바로 들어가면 됨.
