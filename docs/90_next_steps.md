# 90 — Next Steps & Roadmap

> 잔여 작업, 우선순위, 새 세션 시작 시 진행 방법.
>
> **Last updated**: 2026-05-14 (H6-prep done).
> **새 세션 진입 시**: `CLAUDE.md` (auto-load) + `docs/CURRENT_SESSION_STATE.md`
> + 이 파일만 읽으면 됨.

---

## 1. 우선순위 매트릭스

| # | 작업 | 시간 | 의존성 | 가치 |
|---|---|---|---|---|
| **★★★** | **Path planning + EXP-PATH-001** (H6 검증) | 5-7시간 | 모든 RiskMap adapter ✅ | Paper 헤드라인 가설 |
| ★ | PyBullet Week 12 통합 (외주) | 외부 | URDF + RiskMap | 발표용 데모 영상 |
| ★ | 페이퍼 draft + 발표 슬라이드 | 다수 세션 | H6 결과 | 최종 deliverable |

### 완료된 작업

| Date | 작업 | Result | commit |
|---|---|---|---|
| 05-13 | Sparse ConvLSTM v3 (re-sparsify) | IoU 0.581 | c97bfec |
| 05-13 | Sparse FNO v3 (6-ch) | IoU 0.525, FNR 10.4% | 5f448ee |
| 05-14 | 2-way / 3-way ensemble (geodesic) | IoU 0.618, FNR 5.1% | cd2c07e |
| 05-14 | L4h Learned Decoder | **IoU 0.733**, FNR 11.5% (paper) | cc50777 |
| 05-14 | Decoder comparison figures (Pareto + per-scen) | — | e922ba9 |
| 05-14 | Multi-t₀ robustness (Step 1) | IoU stable [0.726, 0.736] for t₀ ∈ [90, 210]s | 5fe5c03 |
| 05-14 | EnsembleDecoderRiskMap (Step 2) | self-test 9/9 pass | d707e26 |
| 05-14 | H1 inference latency (Step 3) | **3,028× faster than FDS** | c29049c |
| 05-14 | 5-fold CV overfit (Step 4) | gap **-0.003** (no overfit) | 546c9cd |

---

## 2. ★★★ H6 검증 작업 세부 (가장 critical)

> **D-025 (2026-05-14) 적용**: H6 = "드론 swarm 안내 vs 고정 표지판 baseline",
> EXP-PATH-001 = 3 PyBullet 시나리오 비교. 구버전 §2 (3-planner 알고리즘
> 비교, 72 trial) 는 `docs/decisions.md::D-025` 의 사유로 폐기됨.

### 2.0 이미 준비된 RiskMap 어댑터 (D-046, was D-029)

| Tag | Class | Source | IoU | FNR | 용도 |
|---|---|---|---|---|---|
| α | `src/tier1/tier1_risk_map.py :: Tier1RiskMap` | per-node GNN nearest-node | 0.904 | 4.6% | ablation (S3 현재 default substitute) |
| **β ★** | `src/tier1/ensemble_risk_map.py :: EnsembleDecoderRiskMap` (fn=2.5) | cell-level decoder | **0.733** | 11.5% | **paper default — H5 9/13** |
| γ | `src/tier1/ensemble_risk_map.py :: EnsembleDecoderRiskMap` (fn=4.0) | cell-level decoder | 0.718 | **10.0%** | safety variant (H4 pass) |
| oracle | `src/risk_map/risk_map_class.py :: StaticRiskMap` | FDS truth | 1.0 | 0% | fairness baseline (S2 가 사용) |

→ EXP-PATH-001 4-RiskMap ablation 으로 confounder 분리.

### 2.1 현재 상태 (2026-05-14)

| 모듈 | 상태 |
|---|---|
| `src/path_planning/building_graph.py` | ✅ 완료 — D-026 cell-grid (60×40×6, fluid mask 위) |
| `src/path_planning/edge_weights.py` | ✅ 완료 — `EdgeWeightConfig` + N-sample integrated risk + `weight = base_cost·length + risk_scale·risk` + impassable filter |
| `src/path_planning/planners.py` | ✅ 완료 — **단일** `EvacuationPlanner` (weighted A* + replan, 3-class ABC 폐기 per D-025) |
| `src/path_planning/evacuation_sim.py` | ✅ 완료 — logical NumPy-only single-occupant sim (multi-agent PyBullet 시뮬은 `src/integration/`) |
| `src/tier1/tier1_risk_map.py` | ✅ 완료 — per-node GNN nearest-node adapter |
| `src/tier1/ensemble_risk_map.py` | ✅ 완료 — `EnsembleDecoderRiskMap` (D-045 decoder) |
| 자체 self-test | ✅ PASS — path_planning 4/4, Tier1RiskMap 9/9, EnsembleDecoderRiskMap 9/9 |
| EXP-PATH-001 large sweep | ✅ 완료 — 45 runs, H6 PASS (FED -98.7%) |

### 2.2 (이전 구현 단계 모두 완료 — 참고용 보존)

D-025 신버전 EXP-PATH-001 의 모든 모듈은 이미 구현 및 검증 완료. 아래는
구버전 작업 흐름 참고용 자료 (`src/integration/` skeleton 단계에서 사용됐던
설계 메모).

#### Step A: `src/tier1/tier1_risk_map.py` (Tier1RiskMap 클래스) — ✅ 완료

→ EXP-PATH-001 는 4개 RiskMap × 3 planner ablation 으로 confounder 분리.

class Tier1RiskMap(RiskMap):
    """GNN forward 결과 → RiskMap interface 어댑터.

    PyBullet 시나리오 S2/S3 모두에서 drone swarm 의 위험도 질의에 사용.
    매 30초마다 binary_sequence history 가 업데이트되면 GNN forward
    결과를 캐싱하고, query(xyz, t) 호출 시 가장 가까운 sensor node 의
    예측 danger 를 반환.
    """

    def __init__(
        self,
        gnn_model: SimpleFireGNN,
        binary_history: torch.Tensor,    # (T_in=6, 39)
        adj: torch.Tensor,
        t0_seconds: float,                # binary history 끝 시각
    ):
        ...

    def query(self, xyz, t=None):
        ...
```

**Test**: `__main__` self-test — synthetic 입력으로 query 동작 검증. ✅ 9/9 PASS.

#### Step B: `src/integration/` 신규 모듈 (Week 12) — ✅ 완료

| 파일 | 책임 |
|---|---|
| `urdf_builder.py` | STL → URDF 변환 + PyBullet 로딩 |
| `person_agent.py` | 단순화된 PersonAgent (1.2 m/s 등속, alive→evacuated/dead 3-state, 벽 충돌 회피) |
| `drone_swarm.py` | gym-pybullet-drones 다중 에이전트, Boids/APF 행동, 고립 PersonAgent 감지 → planner.plan 호출로 waypoint 제공 |
| `scene.py` | PyBullet 월드 셋업, 카메라, fire 시각화 |
| `scenarios/s1_fixed_sign.py` | 고정 표지판 baseline (각 출구 방향 화살표 사전 배치, person 이 RiskMap query 로 위험 회피하며 가장 가까운 미위험 표지로 이동) |
| `scenarios/s2_fds_swarm.py` | drone swarm + `StaticRiskMap.from_fds_dir(...)` |
| `scenarios/s3_fno_swarm.py` | drone swarm + `FNORiskMap` (또는 `Tier1RiskMap` 변형) |
| `metrics.py` | 5-metric 계산기: success_rate / mean_evac_time / danger_zone_exposure / casualty_rate / cumulative_FED |
| `run_exp_path_001.py` | 통합 entry — 3 시나리오 × N 시드 × 20 person 시뮬레이션 |

### 2.3 EXP-PATH-001 실행 (D-025 신버전, ✅ 완료)

```bash
python -m src.integration.run_exp_path_001 \
    --scenarios s1_fixed_sign s2_fds_swarm s3_fno_swarm \
    --fire-scenarios sim_1500kw_2m2_T05 sim_500kw_1m2_T01 sim_1000kw_1m2_T03 \
    --n-persons 20 \
    --n-seeds 5 \
    --output results/exp_path_001/
```

**총 trial**: 3 시나리오 × 3 화재 × 5 시드 = **45 PyBullet runs** (각 run 안에 20 person).

**측정** (per scenario × fire × seed):
- evacuation_success_rate ∈ [0, 1]
- mean_evacuation_time (s) — 도달한 person 한정 평균
- danger_zone_exposure_time (s) — risk ≥ 0.5 체류 시간 평균
- casualty_rate ∈ [0, 1] — dead status 비율
- cumulative_FED — person 별 누적 FED 평균 (H6 primary metric)

**가설 H6**: S2_FED ≤ 0.7 × S1_FED (drone swarm 이 baseline 대비 ≥30% FED 감소).
**결과**: S2/S1 = 0.0127, FED -98.7% → ✅ **H6 PASS**.
**가설 H5 보조 확인**: S2 vs S3 결과 차이 → risk map fidelity 가 path quality 로
transitive 하게 전달되는지. **결과**: S3 (Tier1RiskMap) FED Δ < 1e-7 vs S2 (FDS oracle) → ✅ transitive.

### 2.4 다음 ablation: 4-RiskMap EXP-PATH-001 (구 §2.2 의 변형)

D-046 의 4 RiskMap (α / β / γ / oracle) 을 모두 sweep 해서 EnsembleDecoderRiskMap
β/γ 가 path planning level 에서도 oracle 대비 충분한지 확인. 구버전 §2.2 의
3-planner ablation 은 D-025 로 폐기되었으나, RiskMap 축 ablation 은 여전히
유효함.

```bash
python experiments/exp_path_001_ablation.py \
    --scenarios sim_1500kw_2m2_T05 sim_500kw_1m2_T01 sim_1000kw_1m2_T03 \
    --risk-maps oracle decoder-fn25 decoder-fn40 tier1-gnn \
    --n-persons 20 --n-seeds 5 \
    --output results/exp_path_001_ablation/
```

**총 trial**: 3 시나리오 × 4 risk-maps × 5 시드 × 20 agents = 1200 person-runs.
H5 transitive 검증: decoder-fn25/fn40/tier1-gnn 모두 oracle 대비 FED Δ < 10× 면 통과.

### 2.4 시각화

- `figures/current/07_pybullet_h6/` (신규)
  - `exp_path_001/comparison.csv` — 45 run × 5 metric
  - `fed_boxplot.png` — 3 시나리오 × cumulative_FED 분포
  - `success_rate_bar.png` — 3 시나리오 × 화재 격자
  - `casualty_breakdown.png` — 시나리오별 dead/evacuated/alive 비율
  - `demo.mp4` (~90 s) — 한 화재 시나리오 위에서 S1 vs S2 vs S3 시네마틱

---

## 3. ✅ Sparse-input ConvLSTM 결과 — 완료 (2026-05-13)

학습 완료: `checkpoints/conv_lstm_sparse_v3/best.pt` (50 epoch, 1.4 MB).

**핵심 결과**:
- Mean IoU @ +60s: **0.182** (H5 미달)
- Mean FNR: **0.0%** (conservative bias — 모든 곳 위험 예측)
- Mean RMSE step 6: 0.708

**해석**: Sparse 정보 (1.6% nonzero) 만으로 학습한 결과, ConvLSTM 이 over-prediction 으로 수렴. 정확한 영역 식별 (IoU) 은 못 하지만 위험 영역 놓치지 않음 (FNR 0%). Safety-critical regime 에서 *valuable conservative bias*.

산출물:
- `figures/current/07_sparse_retrain_v3/full_stack_comparison.png` (Layer L2-L4 비교)
- `figures/current/07_sparse_retrain_v3/per_scenario.png`
- `results/exp_sparse_retrain_v3/comparison.csv`
- `docs/archive/auto_reports/sparse_retrain_v3_evaluation.md`

→ **Tier 1 GNN (IoU 0.90) 이 deployment 선택지로 확정**. L4e 의 IoU 낮지만 conservative bias 는 paper 의 *limitation + safety discussion* 으로 활용.

---

## 4. ★★ Tier 1 GNN H1 측정

```bash
python -c "
import torch, time
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency

model = SimpleFireGNN(in_feat=5, hidden=32, n_graph_layers=2, T_out=6)
model.load_state_dict(torch.load('checkpoints/tier1_gnn_v3/best.pt', weights_only=False)['model'])
model.eval()
adj = build_knn_adjacency(k=4)

x = torch.rand(1, 39, 6, 5)
# warmup
for _ in range(10): _ = model(x, adj)

times = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = model(x, adj)
    times.append((time.perf_counter() - t0) * 1000)

import numpy as np
print(f'GNN inference: {np.mean(times):.2f} +/- {np.std(times):.2f} ms')
"
```

→ 결과를 `docs/80_hypothesis_validation.md` 의 H1 섹션에 추가.

---

## 5. ★ PyBullet 통합 (Week 12 — 외주)

외주 명세: [`pybullet_integration_spec.md`](pybullet_integration_spec.md)

작업 흐름:
1. STL → URDF 변환 (Fusion2PyBullet 또는 trimesh + urdfpy)
2. 단일 Crazyflie drone, GNN-based risk map 사용
3. Dynamic planner 의 경로 따라 비행
4. `figures/current/08_pybullet_demo/demo.mp4` 산출 (~90 s)

내가 할 일:
- spec 명세 정확히 (✅ 이미 완료)
- Tier1RiskMap 인터페이스 안정화 (★★★ 작업 1 의 부산물)

---

## 6. ★ Tier 1 + Tier 2 Ensemble

**아이디어**: drone 의 임의 위치 query 시 두 시스템 결합.

```python
def query_ensemble(xyz, t, w_tier1=0.5, w_tier2=0.5):
    d1 = tier1_risk_map.query(xyz, t)
    d2 = tier2_risk_map.query(xyz, t)   # FNO no-PI from sparse interp
    return w_tier1 * d1 + w_tier2 * d2
```

**잠재 효과**:
- Tier 1 의 robust per-node prediction + Tier 2 의 spatial resolution
- Drone 이 노드 사이 임의 위치에 있을 때 Tier 2 의 interpolated value 활용

→ 검증: EXP-PATH-001 trial 3가지 (Tier 1 only / Tier 2 only / Ensemble) 비교.

---

## 7. ★ 페이퍼 draft 구조 (제안)

```
1. Introduction
   - Fire safety motivation
   - 기존 시스템의 한계 (정적 경로)
   - 우리의 contribution (Tier 1/2, evaluation layer framework)

2. Related Work
   - CFD surrogates (ConvLSTM, FNO, PI-FNO)
   - Graph-based fire prediction
   - Risk map + path planning
   - Fire safety standards (NFPA 72, ISO 13571, UL 268)

3. System Design
   - 3.1 Two-tier architecture (Tier 1 / Tier 2)
   - 3.2 39-detector infrastructure (D-024)
   - 3.3 D-023 trigger model
   - 3.4 ConvLSTM / FNO / PI-FNO architectures
   - 3.5 SimpleFireGNN architecture
   - 3.6 Geodesic IDW interpolation (mask-aware)
   - 3.7 Path planning (3 planners)

4. Experiments
   - 4.1 EXP-FIRE-001: model comparison (L1, L2)
   - 4.2 Evaluation layer framework (L1 → L4)
   - 4.3 EXP-RISK-001: H4/H5 verification
   - 4.4 EXP-PATH-001: H6 verification (Dynamic FED reduction)

5. Discussion
   - 5.1 Why Tier 1 beats Tier 2 sparse (phase-transition argument)
   - 5.2 Spectral basis vs local conv (sparse regime)
   - 5.3 Capacity vs domain match (12K params 의 효과)
   - 5.4 Deployment readiness (legacy infrastructure)

6. Limitations
   - Single floor, simulation only, idealized evacuees
   - Cold-start regime (design boundary)
   - 약한 화재 + far-from-detector 시나리오 (T01 500kW 등) 의 H4 marginal

7. Conclusion + Future Work
   - Multi-floor 확장
   - 실 화재 실험 데이터 검증
   - Reinforcement learning 기반 dynamic planner
```

**Paper Figures** (이미 준비됨):
1. **Tier 1 GNN headline** (`figures/current/04_tier1_gnn/headline.png`)
2. Per-scenario IoU/FNR (`figures/current/04_tier1_gnn/aggregate_iou.png`)
3. L1-L4 layer comparison (`figures/current/02_l1_l4_layers/model_comparison.png`)
4. 39 sensor 평면도 (`figures/current/01_sensor_layout/sensor_layout.png`)
5. Geodesic vs Linear interp (`figures/current/03_sparse_interpolation/snapshot_T05_geodesic.png`)
6. 60s autoregress (`figures/current/05_future_prediction/sim_1500kw_2m2_T05_grid_t0_120.png`)
7. (예정) Path planning EXP-PATH-001 box plot
8. (예정) PyBullet demo screenshot

---

## 8. 새 세션 시작 가이드

```
새 Claude Code 세션을 열 때:

1. CLAUDE.md  (auto-loaded, 헌장)
2. docs/README.md  (이 documentation 의 index)
3. docs/00_project_overview.md  (큰 그림 + 현재 상태)
4. docs/90_next_steps.md  (이 파일, 다음 작업)
5. docs/70_results_summary.md  (현재 결과)

이후 작업에 따라:
- H6 path planning → docs/50_tier1_gnn_binary.md §9 (Tier1RiskMap 인터페이스)
- 보간 추가 검증 → docs/40_tier2_models_continuous.md
- D-023/D-024 수정 → docs/30_sensor_infrastructure.md
- 새 가설 추가 → docs/80_hypothesis_validation.md
```

---

## 9. Plan B 활성화 (이미 적용)

- **원본**: PI-FNO doesn't beat ConvLSTM → "30-scenario regime trade-offs" 로 reframe
- **갱신**: H3 partial pass + Tier 1 GNN 발견 → **"단일 인프라, 두 가지 signal mode, binary 가 continuous 를 능가"** 로 더 강하게 reframe.

새 framing 의 contribution 이 원래 H3 보다 강력함.

---

## 10. 최종 deliverables 체크리스트

- [x] 33+13 시나리오 FDS 시뮬레이션 데이터
- [x] ConvLSTM training (33 시나리오)
- [x] FNO no-PI / FNO PI training (RunPod)
- [x] 39 sensor 위치 확정 (D-024 v3.3)
- [x] D-023 트리거 모델 + binary_sequence 생성 (46 시나리오)
- [x] Tier 1 GNN training (12K params, IoU 0.904)
- [x] Sparse + geodesic IDW + 3 모델 평가
- [x] Evaluation Layer L1-L4 framework
- [x] Sparse-input ConvLSTM (IoU 0.581 with re-sparsify, D-042)
- [x] Sparse-input FNO 6-ch (IoU 0.525, FNR 10.4%, D-042)
- [x] Hand-crafted 3-way ensemble + geodesic (IoU 0.618, FNR 5.1%, D-043)
- [x] **L4h Learned Decoder** (IoU **0.733** / FNR 11.5% paper, D-045)
- [x] **L4h fn-weight ablation** (fn=1.0/2.5/4.0)
- [x] Decoder Pareto frontier + per-scenario + sweep figures
- [x] **Multi-t₀ robustness check** (t₀ ∈ [90, 210]s stable)
- [x] **EnsembleDecoderRiskMap** (β/γ cell-level adapter, D-046)
- [x] **H1 latency verified** (full L4h 3,028× faster than FDS)
- [x] **5-fold CV** (mean gap -0.003, no overfit)
- [x] Paper Figure 1-8 + L1-L4h staircase
- [x] Documentation (CLAUDE.md, decisions, layer doc, results, this file)
- [x] REPRODUCE.md command index
- [ ] **Path planning + EXP-PATH-001** ★★★ ← 가장 중요 (5-7시간)
- [ ] PyBullet Week 12 통합 (외주)
- [ ] Paper draft
- [ ] 발표 슬라이드
- [ ] 코드 release (`v1.0-final` tag + `RELEASE.md`)
