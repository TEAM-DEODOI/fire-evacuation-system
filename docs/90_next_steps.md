# 90 — Next Steps & Roadmap

> 잔여 작업, 우선순위, 새 세션 시작 시 진행 방법.

---

## 1. 우선순위 매트릭스

| # | 작업 | 시간 | 의존성 | 가치 |
|---|---|---|---|---|
| **★★★** | **Tier1RiskMap + Path planning + EXP-PATH-001** (H6 검증) | 5-7시간 | 없음 | Paper 헤드라인 가설 |
| **★★** | Sparse-input ConvLSTM 결과 수집 + L4e 갱신 | 30분 (사용자 학습 끝나면) | 사용자 학습 결과 도착 | Layer 표 완성 |
| **★★** | Tier 1 GNN inference time 측정 (H1) | 30분 | GNN ckpt | H1 수치 확정 |
| ★ | PyBullet Week 12 통합 (외주) | 외부 | URDF 생성 | 발표용 데모 영상 |
| ★ | Tier 1 + Tier 2 ensemble (drone query) | 2-3시간 | Tier1RiskMap | 보조 contribution |
| ★ | 페이퍼 draft + 발표 슬라이드 | 다수 세션 | 위 작업 완료 | 최종 deliverable |

---

## 2. ★★★ H6 검증 작업 세부 (가장 critical)

### 2.1 모듈 작성

#### Step 1: `src/tier1/tier1_risk_map.py` (Tier1RiskMap 클래스)

```python
from src.risk_map.risk_map_class import RiskMap
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency
from src.tier1.detector_positions import ALL_DETECTORS

class Tier1RiskMap(RiskMap):
    """GNN forward 결과 → RiskMap interface 어댑터.
    
    use case:
        sim 시 매 30초마다 binary_sequence history 가 업데이트되면
        그걸로 forward 한 결과를 캐싱.
        query(xyz, t) 호출 시 가장 가까운 node 의 danger 반환.
    """
    
    def __init__(
        self,
        gnn_model: SimpleFireGNN,
        binary_history: torch.Tensor,    # (T_in=6, 39)
        adj: torch.Tensor,
        t0_seconds: float,                # binary history 끝 시각
    ):
        # 1) GNN forward → (39, T_out=6) pred danger
        # 2) sensor positions 캐시
        # 3) time mapping: t_query → step index in pred
    
    def query(self, xyz: np.ndarray, t: float | None = None) -> float | np.ndarray:
        # 1) 가장 가까운 sensor node 찾기 (KD-tree 또는 brute force)
        # 2) (t - self.t0_seconds) / 10 → step index
        # 3) clamp + return pred[node_idx, step_idx]
```

**Test**: `__main__` self-test — synthetic 입력으로 query 동작 검증.

#### Step 2: `src/path_planning/edge_weights.py`

```python
def compute_edge_weight(
    graph: nx.Graph,
    edge: tuple,            # (u, v)
    risk_map: RiskMap,
    t: float = 0.0,
    alpha: float = 1.0,
    beta: float = 50.0,
    n_samples: int = 5,
) -> float:
    """Edge weight = α · base_time + β · integrated_risk.
    
    Integrated risk: sample n_samples points along edge, query risk_map,
    take mean (or max for dynamic planner with lookahead).
    """
```

**Test**: 평면도 위 임의 edge 의 weight 계산 PASS.

#### Step 3: `src/path_planning/planners.py`

```python
class EvacuationPlanner(ABC):
    @abstractmethod
    def plan(self, start_xyz, risk_map, graph, t=0.0) -> list[np.ndarray]:
        """waypoint list from start to nearest exit. [] if no path."""

class DijkstraPlanner(EvacuationPlanner):
    """위험 무시, 최단 경로."""

class StaticAvoidancePlanner(EvacuationPlanner):
    """t=0 의 risk snapshot 으로 A*."""

class DynamicPredictivePlanner(EvacuationPlanner):
    """60s lookahead, 30s 마다 replan."""
```

#### Step 4: `src/path_planning/evacuation_sim.py`

```python
class EvacuationSimulator:
    def __init__(self, walking_speed_mps=1.5, dt=1.0):
        ...
    
    def simulate(
        self,
        planner: EvacuationPlanner,
        risk_map_truth: RiskMap,    # 항상 FDS-truth (fairness)
        start_xyz: np.ndarray,
        graph: nx.Graph,
    ) -> dict:
        # path: (N, 3) trajectory
        # cumulative_fed: (N,)
        # reach_time, frac_in_danger, final_fed, reached_exit
```

### 2.2 EXP-PATH-001 실행

```bash
python experiments/exp_path_001.py \
    --scenarios sim_1500kw_2m2_T05 sim_500kw_1m2_T01 sim_1000kw_1m2_T03 \
    --planners dijkstra static dynamic \
    --start-positions 8 \
    --output results/exp_path_001/
```

**총 trial**: 3 scenarios × 3 planners × 8 starts = **72 trials**.

**측정**:
- Mean cumulative FED per planner
- % failed evacuations (FED > 0.3)
- Mean reach time

**가설 H6**: Dynamic FED < Static FED < Dijkstra FED, **Dynamic FED ≤ 0.7 × Dijkstra FED**.

### 2.3 시각화

- `figures/current/07_path_planning/` (신규)
  - `exp_path_001/comparison.csv` — 72 trial 결과
  - `fed_boxplot.png` — 3 planner 별 cumulative FED 분포
  - `path_overlay.png` — 한 scenario 의 3 planner trajectory + risk heatmap

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
- [x] Paper Figure 1-6 (headline)
- [x] Documentation 정리 (이 파일들)
- [x] **Sparse-input ConvLSTM full training** ✅ (IoU 0.182, FNR 0% conservative bias)
- [ ] **Tier1RiskMap + path planning + EXP-PATH-001** ← 가장 중요
- [ ] **PyBullet Week 12 통합** (외주)
- [ ] Paper draft
- [ ] 발표 슬라이드
- [ ] 코드 release (`v1.0-final` tag + `RELEASE.md`)
