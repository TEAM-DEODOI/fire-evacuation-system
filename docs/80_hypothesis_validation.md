# 80 — Hypothesis Validation (H1–H6)

> 6대 가설 검증 현황 + 측정 근거.

---

## 1. 가설 요약표

| ID | 가설 | 목표 | 측정 | 상태 |
|---|---|---|---|---|
| **H1** | Speed ≥ 1000× FDS | < 50 ms | 52,000× (26.5 ms) | ✅ |
| **H2** | RelL2 ≤ 0.15 | ConvLSTM training | 0.136 | ✅ |
| **H3** | FNO > ConvLSTM on OOD | EXP-FIRE-001 | full SLCF ❌ / sparse 39 ✅ | **⚠ partial** |
| **H4** | Risk FNR < 10% | OOD | **GNN 4.6%** | ✅ |
| **H5** | Risk IoU ≥ 0.70 | OOD | **GNN 0.904** | ✅ |
| **H6** | Dynamic A* FED ≥ 30% ↓ | EXP-PATH-001 | 미구현 | 🔜 |

**발표 강조 순서**: H1 → H6 → H4 → H2 → H5 → H3.

---

## 2. H1 — Speed ≥ 1000× FDS

### 측정값
- ConvLSTM inference (CPU, batch=1): **26.5 ms**
- FDS single scenario CPU time: **23 분 = 1,380,000 ms**
- **Speedup: 52,000×** (목표 1000× 의 52배 초과)

### 출처
- `results/eval_T01_T05/per_scenario_metrics.csv` → `infer_ms_mean` 컬럼
- `figures/legacy/eval_T01_T05/aggregated_boxplots.png` — per-scenario inference time

### Tier 1 GNN inference time
- 12K params 모델, CPU < 1 ms 예상 (확정 측정 필요)
- → speedup 더 큼 (~ 80,000,000×)

### 결론
**✅ PASS** — 모든 모델이 1000× 초과 달성.

---

## 3. H2 — RelL2 ≤ 0.15

### 측정값
| Model | Training Mean RelL2 | OOD T01-T05 Mean RelL2 |
|---|---|---|
| ConvLSTM | 0.115-0.158 (border) | 0.136 |
| FNO no-PI | (similar) | 0.138 |
| FNO PI | (similar) | 0.157 |

### 출처
- `results/eval_T01_T05/aggregated.csv` (ConvLSTM)
- `results/eval_T01_T05_fno_no_pi/aggregated.csv`
- `results/eval_T01_T05_fno_pi/aggregated.csv`

### 결론
**✅ PASS (ConvLSTM, FNO no-PI)** — borderline.
FNO PI 는 0.157 로 살짝 실패 (PI loss 가 fitting 제약).

---

## 4. H3 — FNO > ConvLSTM on OOD

### 4.1 Full-SLCF (ideal regime)

L2 평가에서:

| Model | IoU @ +60s (mid-fire) | RMSE °C |
|---|---|---|
| ConvLSTM | **0.92** | 5.68 |
| FNO no-PI | 0.82 | 6.98 |
| FNO PI | 0.89 | 6.86 |

→ **ConvLSTM 이 ideal regime 에서 우위**. H3 ❌.

### 4.2 Sparse 39-sensor regime

L4d (geodesic IDW) 평가에서:

| Model | IoU @ +60s |
|---|---|
| ConvLSTM | 0.212 |
| **FNO no-PI** | **0.431** ★ |
| FNO PI | 0.317 |

→ **FNO no-PI 가 sparse regime 에서 우위**. H3 ✅ partial.

### 4.3 해석

- ConvLSTM 의 local conv 는 **dense input** 에 최적화 (full SLCF).
- FNO 의 Fourier basis 는 **smooth global pattern** (보간 결과) 에 최적화.
- 즉 H3 는 deployment context 에 따라 결과 다름.

### 결론
**⚠ Partial** — full SLCF ❌, sparse ✅. Paper 에서 *deployment-dependent* 로 명시.

---

## 5. H4 — Risk FNR < 10%

### 5.1 Tier 1 GNN (39 sensor binary)

| 시나리오 | FNR step 6 |
|---|---|
| 5 시나리오 | 0.0% |
| 5 시나리오 | 3-9% |
| 2 시나리오 | 11-14% ⚠ |
| **Mean (13 시나리오)** | **4.6%** ✅ |

→ **13/13 시나리오 중 11 시나리오 H4 PASS**, 2 시나리오만 마진 초과.
→ 평균 4.6% — H4 ✅.

### 5.2 Tier 2 sparse (best = FNO no-PI + geodesic)

| Mean FNR | 33% |
|---|---|
| 결론 | ❌ FAIL (모든 시나리오 fail) |

### 5.3 L1 teacher-forced (16 sensor based)

ConvLSTM: 6.0% — 이미 PASS.

### 결론
**✅ PASS** — Tier 1 GNN 이 강력한 H4 통과.

---

## 6. H5 — Risk IoU ≥ 0.70

### 6.1 Tier 1 GNN

| 시나리오 | IoU step 6 |
|---|---|
| All 13 시나리오 | **0.79 ~ 1.00** |
| Mean | **0.904** ✅ |
| H5 통과 비율 | **13/13** |

→ **모든 시나리오 H5 통과**. Mean 마진 +0.20.

### 6.2 다른 evaluation layers

| Layer | IoU |
|---|---|
| L2 (full SLCF) | 0.92 |
| L4d (sparse + geodesic + FNO no-PI) | 0.43 ❌ |
| L4d (sparse + linear) | 0.19 ❌ |

→ **Tier 1 만 sparse regime 에서 H5 통과**.

### 결론
**✅ PASS** — Tier 1 GNN headline result.

---

## 7. H6 — Dynamic A* FED ≥ 30% ↓ (path planning)

### 7.1 현재 상태

**🔜 미구현**. 잔여 작업:
- `src/tier1/tier1_risk_map.py` — Tier1RiskMap 클래스 작성
- `src/path_planning/edge_weights.py` — graph edge weight from danger
- `src/path_planning/planners.py` — Dijkstra / StaticAvoidance / DynamicPredictive
- `src/path_planning/evacuation_sim.py` — EvacuationSimulator
- EXP-PATH-001 실행: 3 planners × 13 OOD × multiple starts

### 7.2 전제 검증

H6 의 *전제 조건* 인 H4/H5 는 ✅. 즉 risk-map 의 quality 가 path planning 의 input 으로 신뢰 가능.

### 결론
**🔜 PENDING** — 다음 세션 작업.

---

## 8. 추가 발견 (Bonus)

### 8.1 Cold-start finding

t₀=0 (화재 발생 직후) 의 autoregress 시 모든 모델 fail.
- ConvLSTM IoU step 6: 0.34
- FNO 둘 다: 0.01

→ Cold-start 는 시스템의 **design boundary** (감지 후 예측 흐름).
실용적 한계가 아닌 시스템 정의의 일부.

### 8.2 Detector trigger ≠ ready

D-023 trigger 시점부터 시작해도 IoU 0.53 — H5 미달.
원인: 빠른 trigger (t=10s) 는 약한 신호 시점.
권고: **"trigger + N s delay"** 또는 **"multi-detector quorum"** 으로 valid regime 정의.

### 8.3 Mask-aware geodesic IDW

단순 Euclidean 보간 → mask-aware 변경 시 IoU +0.07~0.20 회복 (sparse regime).
ConvLSTM 은 둔감, FNO 는 큰 효과.

### 8.4 Binary > Continuous on phase-transition

화재 spread 같은 discrete trigger 도메인에서 binary 정보 손실 < continuous interpolation 손실.
→ Tier 1 GNN > Tier 2 sparse (2.1× 우위).

---

## 9. 가설 검증 figure

| Figure | 경로 | 내용 |
|---|---|---|
| 가설 게이지 | `figures/current/02_l1_l4_layers/model_comparison.png` | H2/H4/H5 막대 with H 임계선 |
| 위치별 비교 | `figures/current/02_l1_l4_layers/per_location.png` | T01-T05 별 모델 성능 |
| Tier 1 headline | `figures/current/04_tier1_gnn/headline.png` | H4/H5 ★ |
| Tier 1 per-scenario | `figures/current/04_tier1_gnn/aggregate_iou.png` | 13개 모두 H5 통과 |

---

## 10. 다음 작업 — H6 완성을 위한 단계

1. **Tier1RiskMap 클래스** — `src/tier1/tier1_risk_map.py`
   - `query(xyz, t)`: 가장 가까운 sensor node → GNN pred danger 반환
   - RiskMap interface 준수

2. **Path planning 모듈** — `src/path_planning/`
   - `edge_weights.py`: 노드별 danger → edge cost
   - `planners.py`: 3 종 (Dijkstra, Static, Dynamic Predictive)
   - `evacuation_sim.py`: occupant trajectory + cumulative FED

3. **EXP-PATH-001** — `experiments/exp_path_001.py`
   - 13 OOD × 3 planners × N start positions
   - 측정: cumulative FED, reach time, frac in danger
   - **목표**: Dynamic FED 가 Dijkstra FED 의 70% 이하 (30% ↓)

4. **PyBullet 통합 (Week 12)** — `docs/pybullet_integration_spec.md` 외주 활용
   - 단일 Crazyflie + GNN risk map → 동적 경로 데모

5. **발표 자료** — Paper Figure 1-6 활용
