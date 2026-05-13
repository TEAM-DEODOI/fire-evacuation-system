# PyBullet 통합 모듈 명세서 (Member 외주용)

> ## ⚠️ STALE — 외주 전달 전 본문 전면 재작성 필요 (2026-05-14)
>
> 본 명세서는 **구버전 H6 정의** 기반으로 작성되었다. `D-025` (2026-05-14)
> 로 H6 가설과 EXP-PATH-001 설계가 다음과 같이 바뀌었다:
>
> | 차원 | 구버전 (본 문서) | 신버전 (`CLAUDE.md` + `D-025`) |
> |---|---|---|
> | H6 | "Dynamic A* FED ≥ 30% ↓ vs Static A*" | "**Drone swarm** 안내 FED ≥ 30% ↓ vs **fixed sign baseline**" |
> | 비교축 | 3 path planner (Dijkstra / Static / Dynamic) | 3 PyBullet 시나리오 (S1 fixed-sign / S2 FDS drone swarm / S3 PI-FNO drone swarm) |
> | 드론 | **단일** Crazyflie | **Swarm** (gym-pybullet-drones, Boids/APF) |
> | 인원 | 개념 없음 | **20 PersonAgent** (1.2 m/s, alive → evacuated/dead) |
> | 메트릭 | FED 중심 + D-022 4-metric | 5-metric: success_rate / mean_evac_time / danger_zone_exposure / casualty_rate / cumulative_FED |
> | H 매핑 | H6 only | **S1 vs S2 → H6**, **S2 vs S3 → H5** |
>
> **외주 전달 전 본문 §2–§5, §8 을 위 내용에 맞춰 재작성할 것.**
> 현재 본문은 single-drone + 3-planner 가정이라 외주자에게 잘못된 명세가
> 전달될 위험이 있다. §1 (프로젝트 컨텍스트), §3.1 (제공 자산), §6 (좌표·시간
> 함정), §7 (환경/의존성) 은 대부분 그대로 유효하다.
>
> 권위 있는 최신 정의: `CLAUDE.md` "EXP-PATH-001 — Three Scenarios" 절 +
> `docs/decisions.md::D-025`.
>
> ---
>
> **작성일**: 2026-05-13 (D-025 이전)
> **대상**: PyBullet 기반 대피 시뮬레이션을 담당할 팀원/협력자
> **위치**: `src/integration/` 디렉터리 (현재 비어 있음 — 본인이 신규 작성)
> **프로젝트 매뉴얼상 위치**: `docs/manual_v2.md` *Phase F — Week 12*
> **목표 산출물** *(구버전)*: H6 가설(Dynamic A* FED ≥ 30% ↓)을 시각적으로
> 입증하는 헤드라인 데모 영상 + 정량 비교 CSV
>
> 이 문서 하나만 읽으면 작업을 시작할 수 있도록 구성했다.
> 추가 디테일이 필요할 때만 `CLAUDE.md` / `docs/interface_contracts.md` /
> `docs/coordinate_convention.md` 를 참조하면 된다.

---

## 0. 한 줄 요약

> **건물 STL 을 URDF 로 변환하고, PyBullet 위에서 (가) 인원 대피 시뮬레이션과
> (나) 드론 보조 대피 시뮬레이션을 만든다.**
> 화재 위험도는 우리가 제공하는 `RiskMap.query(xyz, t) -> [0,1]` 인터페이스로만
> 질의한다. FDS / 딥러닝 모델 내부는 알 필요 없다.

다이어그램의 5단계와 본 명세서의 매핑:

| 다이어그램 단계 | 본 명세 |
|---|---|
| ① Fusion2PyBullet → STL→URDF, PyBullet Model | §4 Milestone 1 |
| ② 인원 대피 시뮬레이션 (Time Based, 고정 유동) | §4 Milestone 2 |
| ③ Risk map 연동 | §4 Milestone 3 |
| ④ 드론 활용 대피 시뮬레이션 (FDS, PI-FNO) | §4 Milestone 4 |
| ⑤ 결과 분석 | §4 Milestone 5 |

---

## 1. 프로젝트 컨텍스트 (필수 배경, 5분 분량)

### 1.1 프로젝트가 뭘 하는 시스템인가

14주 학부 캡스톤 — **능동형 화재대피 시스템**. 파이프라인은 다음과 같다.

```
FDS(화재 CFD) 시뮬 ──▶ ConvLSTM/PI-FNO 가 화재 확산 예측
                  ──▶ ISO-13571 기준 위험도 맵 변환
                  ──▶ 가중 A* 로 대피 경로 산출
                  ──▶ "정적 vs 동적 예측 기반" 경로의 누적 FED 비교  ← H6
                  ──▶ PyBullet 통합 데모로 시각화         ← 본 작업
```

**우리는 fire-detection 이 아닌 fire-response 시스템**이다. 즉
이미 화재 발생을 알고 있다는 가정하에, 더 안전한 경로를 능동적으로
제시한다. PyBullet 데모는 이 "능동 안내가 실제로 더 안전하다" 를
한눈에 보여주는 부분이다.

### 1.2 6개 핵심 가설 중 본 작업과 직결되는 것

| ID | 가설 | 본 작업 관련도 |
|---|---|---|
| **H6** | Dynamic A* 누적 FED ≥ 30% ↓ vs static | ★★★ 본 작업이 입증함 |
| H4 | Risk map FNR < 10% | 이미 입증 (ConvLSTM 9.9%) |
| H5 | Risk map IoU ≥ 0.70 | 이미 입증 (ConvLSTM 0.85) |

**발표 강조 순서**: H1 → H6 → H4 → H2 → H5 → H3.
즉 H6 가 발표 헤드라인이고, PyBullet 영상이 청중에게 가장 강한 인상을
주는 카드다.

### 1.3 건물 / 좌표 / 시간 — 절대 어기면 안 되는 규약

| 항목 | 값 | 비고 |
|---|---|---|
| 건물 | 단일층, 미로형 (3개 출구) | 다층 ❌ |
| 실제 STL 높이 | **3.2 m** | PyroSim 원본 그대로 |
| 학습용 격자(SLCF) | **60 × 40 × 6** cells, [0,30]×[0,20]×[0,3] m | 0~3 m 만 추출 |
| 셀 크기 | **0.5 m × 0.5 m × 0.5 m** | 절대 변경 금지 |
| 단위계 | **SI only** (m, s, °C, ppm) | mm ❌ |
| 좌표계 | **Z-up**, 원점 (0, 0, 0) | 모서리 원점 |
| 시뮬 시간 | 0 ~ 300 s, **10 s 간격** (31 프레임) | DT_SLCF=10 |
| 호흡 높이 | z = 1.5 m | 평가 기준 z |

> ⚠ 좌표 변환 버그는 본 프로젝트에서 가장 흔한 버그다. `CLAUDE.md` 와
> `docs/coordinate_convention.md` 의 PASS 기준을 반드시 검증할 것.

### 1.4 우리가 **하지 않는** 것 (스코프 확장 금지)

- 다층 건물
- 실시간 CFD
- 0.5 m 외 해상도
- 드론 군집 (단일 Crazyflie)
- 실제 화재 실험
- 환기 변경 (모든 시나리오: 양 끝 문 열림 고정)
- HCN/자극가스/복사열 (CO 단독)
- 실제 인간 행동 모델 (이상화된 보행자)
- 기존 소방 시스템 대체 (우리는 보조)

→ "이걸 추가하면 더 좋지 않을까?" 가 생각나면 **먼저 물어볼 것**.

---

## 2. 본 작업의 범위 (Scope)

### 2.1 해야 할 것

1. **STL → URDF 변환** (Fusion2PyBullet 또는 동등 도구)
2. **PyBullet 빌딩 모델 로딩** + collision/visual 확인
3. **인원(에버큐이) 대피 시뮬** — 이상화된 점-질량 에이전트
   - 다중 시작점, 가장 가까운 출구까지 경로 따라 이동
   - 보행속도 1.5 m/s (등속)
4. **Risk map 연동** — 위치/시간으로 위험도 질의, 누적 FED 계산
5. **드론 보조 대피 시뮬** — 동일 시나리오를 세 가지 플래너로 비교
   - Dijkstra (위험 무시 = baseline)
   - Static avoidance (현재 위험만 고려)
   - Dynamic predictive (60 s 미래 예측, 30 s 주기 재계획)
6. **결과 분석 + 시각화 영상 산출**
   - `results/exp_path_001/comparison.csv`
   - `figures/demo.mp4` (~90 s)
   - 위험도 floor heatmap + 경로 trail + FED 카운터 오버레이

### 2.2 하지 말 것

- FDS 시뮬 직접 돌리기 (우리가 .sf 파일 제공)
- 모델(ConvLSTM/FNO) 재학습 (체크포인트 제공)
- Risk map 계산 로직 재구현 (`RiskMap` 클래스 import 만)
- 새 경로 알고리즘 발명 (제공된 인터페이스 사용)
- 다층/swarm/ventilation 확장
- 보행자 군중 동력학 (Social Force Model 등) — 이상화로 충분

---

## 3. 내가 제공할 데이터 / 자산

### 3.1 즉시 제공 가능 (현재 레포에 존재)

| 자산 | 경로 | 형식 | 크기 |
|---|---|---|---|
| **건물 STL** | `data/raw/s_000/*.pyrogeom` (PyroSim 원본) | 자체 포맷 → STL 변환 필요 | 수 MB |
| **건물 그래프** | `configs/building.yaml` | YAML, 19노드/22엣지/3출구 | ~3 KB |
| **시나리오 메타** | `data/raw/scenario_config.json` | JSON, 33 시나리오 | ~20 KB |
| **FDS 슬라이스 원본** | `data/raw/s_000/ ~ s_032/*.sf` + `.smv` | FDS 바이너리 | 시나리오당 ~1-5 GB |
| **전처리된 데이터셋** | `data/processed/dataset.h5` | HDF5, (31, 5, 60, 40, 6) | ~3-5 GB |
| **건물 마스크** | `dataset.h5` 내부 `mask` 키 | (60, 40, 6) float32 | 작음 |
| **ConvLSTM 체크포인트** | `checkpoints/conv_lstm/best.pt` | PyTorch state_dict | 1.37 MB |
| **FNO no-PI 체크포인트** | `checkpoints/fno_no_pi/best.pt` (RunPod 학습 중) | PyTorch state_dict | ~7 MB |
| **FNO PI 체크포인트** | `checkpoints/fno_pi/best.pt` (예정) | PyTorch state_dict | ~7 MB |
| **레퍼런스 figure** | `figures/building_graph.png`, `figures/risk_compare/` | PNG/GIF | 참고용 |

**전송 방법**:
- 코드: git repo (private, push 권한 부여)
- 대용량 (FDS .sf, dataset.h5, checkpoints): Google Drive 공유 폴더
  또는 S3 사전 서명 URL (필요 시 발급)

### 3.2 본 작업이 의존하는 **코드 인터페이스** (수정 금지, import 만)

```python
# 모두 src/ 하위 — pip install -e . 후 그대로 사용 가능
from src.shared.constants import (
    GRID_SHAPE,          # (60, 40, 6)
    DOMAIN_SIZE_M,       # (30.0, 20.0, 3.0)
    DOMAIN_ORIGIN_M,     # (0.0, 0.0, 0.0)
    CELL_SIZE_M,         # 0.5
    T_END_SECONDS,       # 300.0
    N_TIMESTEPS,         # 31
    DT_SLCF,             # 10.0
)
from src.shared.coordinates import world_to_grid, grid_to_world, is_in_bounds
from src.shared.building import BuildingGraph    # 19노드 NetworkX wrapper
from src.risk_map.risk_map_class import (
    RiskMap,                # ABC
    StaticRiskMap,          # FDS 기반 정적 risk map
)
from src.risk_map.converter import model_output_to_risk_map
from src.risk_map.fed import accumulate_fed_co
```

핵심 인터페이스 한 번 더 (이거 하나만 외우면 됨):

```python
class RiskMap(ABC):
    def query(
        self,
        xyz: np.ndarray,            # (3,) 또는 (N, 3), 월드 m
        t: float | None = None,     # 초
    ) -> float | np.ndarray:
        """
        반환: danger ∈ [0, 1]   (높을수록 위험)
        Out-of-bounds (xyz) → 1.0
        Out-of-time-range (t > 300) → 1.0
        """
```

### 3.3 본 작업이 **신규로 만들 모듈** (`src/integration/` 하위)

| 모듈 | 책임 |
|---|---|
| `src/integration/urdf_builder.py` | STL → URDF 변환 + PyBullet 로딩 유틸 |
| `src/integration/evacuee_agent.py` | 점-질량 보행자 에이전트 (waypoint 추종) |
| `src/integration/drone_agent.py` | Crazyflie 드론 에이전트 (waypoint 추종) |
| `src/integration/scene.py` | PyBullet 월드 셋업, 카메라, 렌더링 |
| `src/integration/pybullet_demo.py` | 통합 엔트리포인트 (`run_demo()`) |
| `assets/building.urdf` | 산출 URDF 파일 |
| `assets/crazyflie.urdf` | `gym-pybullet-drones` 패키지에서 복사 |

> Path planning 모듈(`src/path_planning/edge_weights.py`, `planners.py`,
> `evacuation_sim.py`)은 **별도 작업으로 진행 중**이다. 본 작업은
> 이 모듈들의 인터페이스만 의존한다 (§5 참조). 만약 본 작업 시작 시점에
> 아직 작성되지 않았다면, 다음 합의된 시그니처에 맞춰 mock 으로
> 진행 가능하다.

### 3.4 데이터 흐름 (End-to-End)

> 이 절은 §3.1 (원본 자산) 과 §3.3 (신규 모듈) 을 **시간 흐름으로 연결**한다.
> "어떤 데이터로 무엇을 만들고, 그것을 위해 레포에 무엇이 있고 무엇을 추가로
> 만들어야 하는가" 를 한눈에 보기 위함.

#### 3.4.1 전체 파이프라인 다이어그램

```
 [원본 자산]                  [변환/모듈]                  [중간 산출물]                  [최종 산출물]
 ────────────────────────────────────────────────────────────────────────────────────────────────────────
 PyroSim .pyrogeom       ──▶  Fusion2PyBullet  ──▶   assets/building.stl  ──▶  assets/building.urdf
 (data/raw/s_*/*.pyrogeom)    또는 trimesh+urdfpy                                  │
                              ✱ 친구 작성                  ✱ 친구 산출            ✱ 친구 산출
                                                                                    │
 configs/building.yaml   ──▶  BuildingGraph         ──▶  nx.Graph(19노드)          │
 ✓ 이미 있음                  (src/shared/building.py)   ──▶  cell_to_node 매핑    │
                              ✓ 이미 있음                                          │
                                                                                    │
 data/raw/s_NNN/*.sf,.smv ──▶ fds_extractor + tenability+fed+aset ──▶ StaticRiskMap (FDS-truth)
 ✓ 33 시나리오 있음          (src/data_pipeline + src/risk_map)        ──▶  .query(xyz, t) ∈ [0,1]
                              ✓ 모두 이미 있음                                     │
                                                                                    │
 checkpoints/fno_pi/best.pt + data/processed/dataset.h5                            │  (선택)
 + initial frame                                                                   │
 ──▶ FNORiskMap / DynamicRiskMap (src/risk_map/converter.py + predictive.py)     ──┤
     ✓ 이미 있음 (FNO 체크포인트는 RunPod 학습 중)                                  │
                                                                                    ▼
                                                                          ┌────────────────────────┐
                                                                          │  Planner 3종           │
                              start_xyz ──────────────────────────────────│  Dijkstra / Static /    │
                              graph     ──────────────────────────────────│  Dynamic                │
                              risk_map  ──────────────────────────────────│  → list[waypoint xyz]   │
                                                                          │  (src/path_planning,     │
                                                                          │   Week 11 별도 작업)    │
                                                                          └─────────┬──────────────┘
                                                                                    │
                                                                                    ▼
                                                                          ┌────────────────────────┐
                                                                          │  EvacueeAgent           │
                                                                          │  DroneAgent             │
                                                                          │  Scene + 카메라          │
                                                                          │  ✱ 친구 작성             │
                                                                          └─────────┬──────────────┘
                                                                                    │
                                                                                    ▼
                                                                          trajectory dict
                                                                          (path_xyz, times,
                                                                           danger, co_ppm, fed, ...)
                                                                                    │
                                              ┌─────────────────────────────────────┤
                                              ▼                                     ▼
                                  PyBullet rendered frames           EvacuationSimulator 집계
                                  ──▶ figures/demo.mp4               (src/path_planning/evacuation_sim.py
                                  ✱ 친구 산출                          + src/risk_map/path_metrics.py)
                                                                       Week 11 ✓ path_metrics 있음
                                                                       Week 12 ✱ 친구가 simulator 연결
                                                                                    │
                                                                                    ▼
                                                                       results/exp_path_001/comparison.csv
                                                                       + figures/exp_path_001/fed_boxplot.png
                                                                       ✱ 친구 산출
```

범례
- `✓ 이미 있음` — 그대로 import / 사용
- `✱ 친구 작성` — 본 명세서 작업 범위
- 점선 분기 (Risk map 두 갈래) — FDS-truth 와 모델 예측 둘 다 같은 인터페이스. 본 작업은 **FDS-truth 만 써도 충분**, 모델 기반은 선택.

#### 3.4.2 단계별 입력 → 출력 매핑 표

| # | 단계 | 입력 (where) | 변환/사용 모듈 | 출력 | 레포 상태 |
|---|---|---|---|---|---|
| 1 | **STL 추출** | `data/raw/s_000/*.pyrogeom` (PyroSim 자체 포맷) | Fusion2PyBullet / PyroSim GUI export | `assets/building.stl` | ✱ **친구 작성** (원본은 있음, STL 변환은 1회성) |
| 2 | **URDF 작성** | `assets/building.stl` | trimesh + urdfpy 또는 수작업 XML | `assets/building.urdf` | ✱ **친구 작성** |
| 3 | **PyBullet 월드** | URDF + Crazyflie URDF | `pybullet.loadURDF` | sim world (in-memory) | ✱ **친구 작성** (`src/integration/scene.py`) |
| 4 | **건물 그래프 로드** | `configs/building.yaml` | `BuildingGraph.from_yaml(...)` | `nx.Graph` (19 node) | ✓ **있음** (`src/shared/building.py`) |
| 5 | **FDS truth RiskMap** | `data/raw/s_NNN/*.sf` + `.smv` (33 시나리오) | `StaticRiskMap.from_fds_dir(...)` | `RiskMap` 인스턴스 (query 가능) | ✓ **있음** (`src/risk_map/risk_map_class.py` + `src/data_pipeline/fds_extractor.py`) |
| 6 | **(옵션) 모델 RiskMap** | `dataset.h5` 의 initial frame + `checkpoints/fno_pi/best.pt` | `converter.model_output_to_risk_map` + `predictive.autoregress` | `FNORiskMap` / `DynamicRiskMap` | ✓ **converter/predictive 있음**, FNO ckpt 는 RunPod 학습 중 |
| 7 | **경로 계산 (3종)** | `start_xyz`, `risk_map`, `graph`, `t` | `DijkstraPlanner` / `StaticAvoidancePlanner` / `DynamicPredictivePlanner` | `list[waypoint xyz]` | ⚠ **Week 11 별도 작업** (없으면 mock 사용 — §5) |
| 8 | **EvacueeAgent 이동** | waypoints + walking_speed=1.5 m/s + dt=1.0 | `src/integration/evacuee_agent.py` | trajectory `path_xyz`, `times` | ✱ **친구 작성** |
| 9 | **DroneAgent 비행** | waypoints + flight_height=1.5~2.0 m | `src/integration/drone_agent.py` | drone trajectory | ✱ **친구 작성** |
| 10 | **위험도/CO 질의** | trajectory 의 매 점 + `RiskMap` (FDS-truth) | `risk_map.query(xyz, t)` (각 단계마다) | `danger_along_path`, `co_ppm_along_path` | ✓ **인터페이스 있음** (호출만) |
| 11 | **FED 누적** | `co_ppm_along_path`, `dt=1.0` | `accumulate_fed_co(...)` | `fed_cumulative`, `final_fed` | ✓ **있음** (`src/risk_map/fed.py`) |
| 12 | **path 4-metric** | trajectory + `RiskMap` (FDS-truth) | `path_metrics.py` (D-022 4-metric) | `peak_danger`, `time_in_hazard`, `aset_margin`, `fed_final` | ✓ **있음** (`src/risk_map/path_metrics.py`) |
| 13 | **EvacuationSimulator** | planner + risk_map_truth + start_xyz + graph | `EvacuationSimulator.simulate(...)` | trial dict (per-trial 결과) | ⚠ **Week 11**, 미완 시 친구가 §4-M5 형식으로 자체 wrapper 작성 |
| 14 | **렌더링** | PyBullet world + risk overlay + trajectory | `pybullet.getCameraImage` + `imageio` | per-frame PNG → `figures/demo.mp4` | ✱ **친구 작성** (`src/integration/pybullet_demo.py`) |
| 15 | **집계** | 45~72 trial dict | pandas concat + groupby | `results/exp_path_001/comparison.csv` + boxplot | ✱ **친구 작성** |
| 16 | **H6 판정** | comparison.csv | mean FED reduction 계산 | console 출력 + boxplot figure | ✱ **친구 작성** |

#### 3.4.3 "이미 있는 것" vs "친구가 만들 것" 한눈 요약

| 분류 | 항목 | 수정 가능 여부 |
|---|---|---|
| ✓ **그대로 import 만** | `BuildingGraph` (19노드), `StaticRiskMap`, `FNORiskMap`, `DynamicRiskMap`, `accumulate_fed_co`, `tenability`, `aset`, `path_metrics`, `converter`, `predictive`, `world_to_grid` / `grid_to_world`, `constants.*` | ❌ 수정 금지 |
| ✓ **데이터로 그대로 사용** | `configs/building.yaml`, `data/raw/s_NNN/` (33개), `data/processed/dataset.h5`, `checkpoints/conv_lstm/best.pt`, `checkpoints/fno_*/best.pt` | ❌ 수정 금지 |
| ⚠ **인터페이스만 의존, 본체는 별도 작업** | `EvacuationPlanner` 3종, `EvacuationSimulator` (Week 11 모듈) | 없으면 mock 으로 진행 (§5) |
| ✱ **친구가 신규 작성** | `assets/building.stl`, `assets/building.urdf`, `src/integration/urdf_builder.py`, `src/integration/scene.py`, `src/integration/evacuee_agent.py`, `src/integration/drone_agent.py`, `src/integration/pybullet_demo.py`, `results/exp_path_001/comparison.csv`, `figures/demo.mp4`, `figures/exp_path_001/fed_boxplot.png` | 자유 |

#### 3.4.4 데이터 흐름의 핵심 원칙 3가지

1. **`RiskMap.query` 는 모든 위험도 정보의 single source of truth.**
   - FDS, ConvLSTM, FNO 어떤 출처든 `RiskMap` 으로 wrap 된 뒤에는 동일하게 사용.
   - PyBullet 측에서 raw FDS 슬라이스를 직접 읽는 코드는 ❌ (계산 중복 + 좌표 버그 위험).
   - 단, *시각화* 용 floor heatmap 은 `risk_map.query` 를 (60, 40) 그리드 점들에 batch 호출해서 만든다.

2. **에버큐이가 "실제로 겪는 위험" 은 항상 FDS-truth 로 측정.**
   - Planner 가 무엇을 보든 (Dijkstra=아무것도/Static=t=0/Dynamic=60s 룩어헤드), 누적 FED 와 reach_time 은 `StaticRiskMap.from_fds_dir(scenario_X)` 에서 평가.
   - 이게 fair comparison 의 핵심이고, EXP-PATH-001 의 가설 H6 자체가 "planner 가 imperfect 한 정보로 결정해도 결과적으로 더 안전한가" 를 묻는 실험이다.

3. **PyroSim `.pyrogeom` 은 1회성으로 STL 만들고 끝.**
   - 33 시나리오 모두 같은 건물 → STL/URDF 는 1번만 만들면 된다.
   - 시나리오 차이는 화재 위치/HRR 이고, 건물 자체는 동일.
   - URDF 파일 1개 + 시나리오마다 `RiskMap` 인스턴스만 다르게 로드.

---

## 4. 마일스톤 (단계별 산출물)

### Milestone 1 — STL/URDF + 빈 PyBullet 월드 (~2일)

**할 일**:
- PyroSim `.pyrogeom` 또는 별도로 export 한 STL 을 URDF 로 변환
  (Fusion2PyBullet, 또는 trimesh+urdfpy 조합 가능)
- 단일 link, fixed base, collision/visual 모두 STL 메시 사용
- PyBullet 에서 로딩 후 카메라 회전하며 메시 깨짐 없는지 확인

**검증 조건**:
```python
import pybullet as p
p.connect(p.GUI)
p.loadURDF("assets/building.urdf", basePosition=[0, 0, 0], useFixedBase=True)
# 외관 확인 → 건물 윤곽이 [0,30]×[0,20]×[0,3.2] m 안에 들어와야 함
```

**자가 진단** (필수):
- 원점이 (0, 0, 0) 인가? (코너 정렬)
- Z 축이 위인가?
- 단위가 m 인가? (mm 로 export 되면 1/1000 스케일링)

### Milestone 2 — 인원 대피 시뮬 (Time-Based, 고정 유동) (~3일)

**할 일**:
- `EvacueeAgent` 클래스: 시작점(xyz) → 경로 waypoint 리스트 따라 등속 이동
- 보행속도 **1.5 m/s** (고정), dt=1.0 s 시뮬 스텝
- 시작점 5~8 개 미리 정의 (방 내부 임의 위치)
- 경로는 일단 **Dijkstra 결과 (위험 무시)** 만으로 시작
  (다이어그램의 "고정 유동" = Static Dijkstra 라고 해석)

**리턴 dict 표준 형식** (이후 단계들이 이걸 받아 시각화):
```python
{
    "path_xyz": np.ndarray,        # (N_steps, 3) 위치 trajectory
    "times":     np.ndarray,        # (N_steps,) 초
    "reach_time": float,            # 출구 도달 시각, 도달 못하면 inf
    "reached_exit": bool,
}
```

**시각화 산출물**:
- PyBullet GUI 내에서 빨간색 점/구가 path 따라 움직이는 GIF
- `figures/integration/m2_evacuee_trail.gif`

### Milestone 3 — Risk Map 연동 + FED 누적 (~2일)

**할 일**:
- `StaticRiskMap.from_fds_dir("data/raw/s_010")` 로 위험도 맵 로드
- Milestone 2 의 path 를 따라 매 스텝 위험도/CO 농도 질의
- 누적 FED 를 `accumulate_fed_co(co_ppm_along_path, dt_seconds=1.0)` 로 계산
- z = 1.5 m 슬라이스를 floor heatmap 으로 PyBullet 에 textured ground 또는 colored plane 으로 표시

**중요 — Risk map 과 PyBullet 좌표 일치**:
- Risk map 은 **[0, 30] × [0, 20] m** 영역만 정의됨
- 이 바깥은 `query` 가 1.0 (max danger) 반환 → 출구 노드 (e.g. (0,5)/(30,13)/(8,18)) 의
  바로 안쪽 셀 정도까지만 위험도 유효
- PyBullet 카메라/조명/grid 와 일치하도록 z 축 정렬 재확인

**검증 조건**:
- 화재 발생점 근방에서 query 값 > 0.8
- 멀리 떨어진 방 query 값 < 0.2
- 시간이 지날수록 같은 점의 danger 가 단조 증가 (또는 적어도 비감소 영역 존재)

**리턴 dict 확장**:
```python
{
    ...,                           # M2 의 키들
    "danger_along_path": np.ndarray,   # (N_steps,) ∈ [0, 1]
    "co_ppm_along_path": np.ndarray,   # (N_steps,) raw ppm
    "fed_cumulative":    np.ndarray,   # (N_steps,) ∈ [0, ∞)
    "final_fed":         float,
    "frac_in_danger":    float,        # danger > 0.5 인 step 비율
}
```

### Milestone 4 — 드론 보조 대피 시뮬 (3 plan 비교) (~4일)

**할 일**:
- Crazyflie URDF 로드 + 점-질량 근사 (실제 동역학 PID 까지는 불필요, 위치 보간으로 충분)
- 세 가지 planner 의 path 각각에 대해 동일 시나리오 / 동일 시작점으로 실행
- Risk map source 차이:

| Planner | 사용 RiskMap | 비고 |
|---|---|---|
| Dijkstra | (사용 안 함) | 길이만 |
| Static avoidance | `StaticRiskMap.from_fds_dir(...)` (t=0 snapshot) | 정적 |
| Dynamic predictive | `StaticRiskMap` (시간 의존 query) — 60 s 룩어헤드 | replan 30 s |

> ⚠ "Ground truth" 와 "모델 예측" 구분:
> - **에버큐이가 실제로 겪는 위험**: 항상 `FDSRiskMap` (= `StaticRiskMap.from_fds_dir(FDS)`) 으로 계산. 이건 fairness.
> - **Planner 가 보는 위험**: Static 은 t=0 의 FDS, Dynamic 은 (실제 데모에선) FNO 기반 `DynamicRiskMap`. 단, 본 명세에선 일단 **둘 다 FDS 사용** 으로 충분 — 우리는 plan 차이만 보고 싶음.
> 만약 PI-FNO 기반 dynamic 까지 보여주고 싶다면 §6 추가 옵션 참조.

**드론 행동**:
- 에버큐이보다 약간 앞서 (예: 2 m) 비행하며 다음 waypoint 를 시각적으로 안내
- 충돌 처리는 단순화 — collision filter 로 building mesh 와만 충돌하면 됨

**시각화 산출물**:
- 같은 시나리오에서 3 planner 의 결과를 좌/중/우 분할 화면 또는 3 연속 클립으로 비교
- 화면 오버레이: 시뮬레이션 시각, 현재 danger, 누적 FED, 남은 거리

### Milestone 5 — EXP-PATH-001 결과 분석 (~2일)

**할 일** — 매뉴얼 Phase F 의 EXP-PATH-001 절차 그대로:
- **시나리오 × 시작점 그리드 실험**: 3 시나리오 × 3 planner × 5~8 시작점 = 45~72 trial
- (3 시나리오는 추후 OOD 가 도착하면 그걸로, 지금은 train 33개에서 다양성 있는 3개 선택 — 예: HRR=500/1000/1500 각각 1개)
- trial 마다 `final_fed`, `reach_time`, `reached_exit`, `frac_in_danger` 기록

**산출 CSV** `results/exp_path_001/comparison.csv`:

| scenario_id | planner | start_id | final_fed | reach_time | reached_exit | frac_in_danger |
|---|---|---|---|---|---|---|
| s_005 | dijkstra | start_A | 0.42 | 38.0 | True | 0.34 |
| s_005 | static   | start_A | 0.28 | 42.0 | True | 0.21 |
| s_005 | dynamic  | start_A | 0.15 | 44.5 | True | 0.09 |
| ... | ... | ... | ... | ... | ... | ... |

**가설 검증 출력** (반드시 포함):
```
Mean final_fed  — Dijkstra: 0.XX, Static: 0.YY, Dynamic: 0.ZZ
Reduction (Dynamic vs Dijkstra): X.X%   ← H6 목표: ≥ 30%
% failed evacuations (FED > 0.3) per planner
Boxplot figure: figures/exp_path_001/fed_boxplot.png
```

**최종 데모 영상** `figures/demo.mp4` (~90 s):
- 1 시나리오, 1 시작점 골라서 시네마틱 영상
- 3 planner 를 화면 분할 또는 연속 컷
- 마지막 5 초에 FED 비교 막대그래프 페이드인

---

## 5. 의존 인터페이스 (`src/path_planning/` 가 별도 작업 중일 경우 대응)

본 작업은 다음 시그니처가 **이미 존재한다고 가정** 한다. 작업 시작 시점에
없다면 mock 으로 대체 후, 합의된 시그니처에 맞춰 추후 교체.

```python
# src/path_planning/planners.py
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from src.risk_map.risk_map_class import RiskMap

class EvacuationPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        start_xyz: np.ndarray,    # (3,) 월드 m
        risk_map: RiskMap,
        graph: nx.Graph,          # BuildingGraph.graph
        t: float = 0.0,
    ) -> list[np.ndarray]:
        """
        반환: waypoint 리스트. 각 원소 shape (3,) 월드 m.
              빈 리스트 = 경로 없음.
              첫 원소 = 시작점에 가까운 노드, 마지막 원소 = 출구 노드.
        """

class DijkstraPlanner(EvacuationPlanner): ...
class StaticAvoidancePlanner(EvacuationPlanner): ...
class DynamicPredictivePlanner(EvacuationPlanner): ...
```

```python
# src/path_planning/evacuation_sim.py
class EvacuationSimulator:
    def __init__(self, walking_speed_mps: float = 1.5, dt: float = 1.0): ...

    def simulate(
        self,
        planner: EvacuationPlanner,
        risk_map_truth: RiskMap,       # 항상 FDS 기반!
        start_xyz: np.ndarray,
        graph: nx.Graph,
    ) -> dict:
        """
        반환 dict 키: path, cumulative_fed, reach_time, frac_in_danger,
                      final_fed, reached_exit
        """
```

**Mock 으로 대체할 때** (path_planning 모듈 미완성 시): NetworkX 의
`nx.shortest_path` 만으로 Dijkstra mock 작성 후, Static/Dynamic 도 같은
경로를 우선 반환하도록 stub. 실 모듈 완성 후 교체.

---

## 6. 좌표·시간·단위 함정 모음 (필독)

### 6.1 SLCF 격자 ↔ 월드 좌표
- cell center 가 0.25, 0.75, 1.25, … (왼쪽 모서리 + 0.25 m)
- `world_to_grid(np.array([0.25, 0.25, 0.25])) → [0, 0, 0]`
- `world_to_grid(np.array([0.0, 0.0, 0.0]))` 은 -1 (경계 바깥)
- **반드시 `src.shared.coordinates` 의 헬퍼만 사용**, hard-code 금지

### 6.2 STL 높이 vs SLCF 높이
- STL 은 **3.2 m** 높이까지 있음 (실 건물)
- SLCF 는 **0 ~ 3 m** 만 추출 (학습용)
- → 1.5 m 호흡 높이는 STL 안에 들어 있고, risk map 으로도 질의 가능
- 드론 비행고도는 1.5 ~ 2.0 m 권장 (위험 측정 + 보행자 시야 보조)

### 6.3 시간축
- SLCF 프레임 인덱스 0 ~ 30 = 시간 0, 10, 20, …, 300 s
- 시뮬 dt = 1.0 s 로 잡고, risk_map.query(xyz, t) 의 t 가 frame 사이일 때는
  내부 interpolation 이 자동 처리 (걱정 안 해도 됨)

### 6.4 단위
- m, s, °C, ppm — **mm/cm/min/K 금지**
- STL/URDF export 시 mm 단위로 나가는 도구 많음 → URDF mesh tag `scale="0.001 0.001 0.001"` 필요 가능

### 6.5 자주 발생하는 버그 (lessons_learned.md 발췌)

| 증상 | 원인 | 대응 |
|---|---|---|
| risk_map.query 가 모두 1.0 | 좌표가 [0,30]×[0,20]×[0,3] 바깥 | URDF 위치/스케일 재확인 |
| URDF 가 PyBullet 에서 두께 0 | STL export 시 face inversion | trimesh.repair.fix_normals |
| 보행자가 벽 통과 | building URDF collision 메시 없음 | `<collision>` 태그 명시적으로 추가 |
| FED 가 항상 0 | CO ppm 이 raw 가 아닌 normalised 값 | `StaticRiskMap` 의 raw_co 채널 사용 |
| 시간이 다르면 danger 가 같음 | risk_map 인스턴스가 단일 프레임만 로드됨 | `from_fds_dir` 사용 (시계열 자동 로드) |

---

## 7. 환경 / 의존성

```
Python 3.10+
PyTorch 2.0+ (체크포인트 로드용; GPU 없어도 됨, CPU inference 충분)
fdsreader   (raw FDS 직접 안 읽어도 되면 생략 가능)
numpy, scipy, h5py, pyyaml, networkx
pybullet
gym-pybullet-drones   (Crazyflie URDF 용)
trimesh, urdfpy       (STL→URDF 변환용; Fusion2PyBullet 쓰면 불필요)
matplotlib            (figure 생성)
imageio[ffmpeg]       (mp4 렌더링)
```

설치:
```bash
git clone <repo>
cd fire-evacuation-system
pip install -e .
pip install pybullet gym-pybullet-drones trimesh urdfpy imageio[ffmpeg]
```

GPU 불필요 — PyBullet 통합은 CPU 만으로 충분. 단 ConvLSTM/FNO 로
`DynamicRiskMap` 까지 가려면 GPU 권장 (선택).

---

## 8. 결과물 제출 기준 (Definition of Done)

본 작업이 "완료" 로 판정되려면 다음이 모두 통과해야 한다.

### 8.1 코드 자가 검증

각 모듈은 `__main__` self-test 가 있어야 하고:
```bash
python -m src.integration.urdf_builder      # PASS
python -m src.integration.evacuee_agent     # PASS
python -m src.integration.drone_agent       # PASS
python -m src.integration.pybullet_demo --smoke   # PASS (5초 데모)
```

### 8.2 EXP-PATH-001 결과

- `results/exp_path_001/comparison.csv` 존재
- 최소 45 trial (3 시나리오 × 3 planner × 5 시작점) 행
- 집계 출력에 Dynamic vs Dijkstra FED reduction 명시
- → **H6 검증**: ≥ 30% 면 ✅, 미달이면 원인 분석 1 페이지 작성

### 8.3 영상

- `figures/demo.mp4` (~90 초, 1080p, 30fps 권장)
- 3 planner 가시적 비교 가능
- 자막 / 오버레이로 시간, danger, FED 표시

### 8.4 문서 갱신

- `docs/decisions.md` 에 본 작업 중 내린 설계 결정 (D-025~) 추가
- `docs/lessons_learned.md` 에 발생한 버그 + 해결법 추가 (L-013~)
- `CLAUDE.md` 의 "Current Project State" 갱신

### 8.5 Pull Request

- 브랜치: `feature/pybullet-integration`
- PR 본문에 §8.2 의 집계 표 붙여넣기
- 리뷰어: 본인 (Member D / 프로젝트 오너)

---

## 9. 일정 가이드 (참고)

| 주차 | 마일스톤 | 누적 예상 소요 |
|---|---|---|
| 1 주차 | M1 + M2 | ~5일 |
| 2 주차 | M3 + M4 시작 | ~7일 |
| 3 주차 | M4 마무리 + M5 | ~6일 |

총 약 2-3 주. 본인 페이스에 맞게 조정 가능하지만 **Week 12** 마감 (
14주 캡스톤 기준) 안에는 들어가야 한다.

---

## 10. 의사소통 약속

### 10.1 막혔을 때

> **"30분 이상 막히면 ASK"** — CLAUDE.md 의 원칙.
>
> 특히 다음은 본인 판단으로 결정하지 말고 물어볼 것:
> - 인터페이스 시그니처를 바꿔야 할 것 같을 때
> - 새 의존성 패키지 추가하고 싶을 때
> - 시뮬 파라미터(보행속도, dt, 시작점 좌표) 변경
> - 시각화 스타일/색상 결정

### 10.2 정기 동기화

- **주 1회 30분 짧은 sync** (체크포인트 영상 시연 + 다음 주 목표)
- 텍스트 채널에 매 마일스톤 완료 시 짧은 보고

### 10.3 코드 스타일 (CLAUDE.md §"Coding Conventions" 발췌)

- 모든 public 함수: type hint + docstring
- `pathlib.Path` (os.path 금지)
- `raise ValueError("clear message")` — silent failure 금지
- 절대 import: `from src.shared.constants import GRID_SHAPE`
- 한국어 주석 OK, 식별자(변수/함수명)는 영어
- 모든 계산 모듈에 `if __name__ == '__main__'` self-test + PASS/FAIL 출력

---

## 11. 자료 / 레퍼런스

### 본 레포 내 필수 참고
1. `CLAUDE.md` — 프로젝트 헌장
2. `docs/interface_contracts.md` §2 (RiskMap), §5 (PathPlanner), §6 (EvacuationSimulator)
3. `docs/coordinate_convention.md` — 좌표 규약
4. `docs/risk_indicators.md` — FED / tenability 공식
5. `docs/manual_v2.md` *Phase F (Week 12)* — 통합 데모 원안
6. `docs/handoff_2026_05_12.md` — 현재 진행 상황

### 외부 레퍼런스
- PyBullet Quickstart Guide
- `gym-pybullet-drones` GitHub (Crazyflie URDF 출처)
- Fusion2PyBullet (STL→URDF 변환 도구)
- ISO 13571:2012 §5–7 (FED 정의)
- SFPE Handbook 5th Ed. Ch. 63 (Purser & McAllister 2016)

---

## 12. FAQ (Anticipated)

**Q1. URDF 가 mm 단위로 나오는데 어떻게 해야 하나?**
A. URDF 의 `<mesh>` 태그에 `scale="0.001 0.001 0.001"` 추가. 그 후
`world_to_grid([15.0, 10.0, 1.5])` 가 (30, 20, 3) 근처 인덱스를 반환하는지로
검증.

**Q2. FNO 체크포인트로 DynamicRiskMap 만드는 게 헷갈린다.**
A. 본 작업은 일단 `StaticRiskMap.from_fds_dir` 만 써도 충분.
Dynamic-vs-Static 비교는 *planner 의 시간 인식* 차이이지 risk map 종류
차이가 아니다 (Milestone 4 의 ⚠ 박스 참조). PI-FNO 기반 dynamic 까지
하고 싶다면 추가 1주.

**Q3. 보행자 모델을 좀 더 사실적으로 하면 안 되나?**
A. 안 됨 (CLAUDE.md "What This Project Does NOT Do"). 점-질량 + 등속
보행으로 가설 H6 검증에 충분하고, 군중 동력학 도입은 스코프 폭발.
시간 남으면 ablation 으로 추가 가능 (논의 필수).

**Q4. PyBullet 대신 Unity/Unreal/Omniverse 쓰면 더 예쁠 것 같다.**
A. 거부. 매뉴얼에 PyBullet 명시 (CLAUDE.md Tech Stack §Wk 12). 데모용 가벼움이
선택 이유. 14주 안에 다른 엔진 학습 + 통합은 비현실적.

**Q5. 시작점은 어떻게 정하나?**
A. `BuildingGraph` 의 room 타입 노드 좌표를 그대로 쓰고, 거기서 반경
0.5 m 내 임의 점을 sample. 최소 5개. 한 시나리오 안에서는 시작점 고정.

**Q6. 화재 위치가 시나리오마다 다른데 화재 시각화도 PyBullet 에 띄워야 하나?**
A. 권장. `scenario_config.json` 의 `fire_loc` 좌표에 작은 빨간 구
(visual only, no collision) 를 띄우면 청중 이해도 ↑.

**Q7. 30%↓ 못 맞추면 어떻게 하나?**
A. 매뉴얼 Plan B (`docs/manual_v2.md` "Plan B Failure Scenarios"):
페이퍼 reframe — "30-scenario regime trade-offs" 라는 정직한 한계 논의로
전환. 본 작업 결과 자체는 그대로 유효. 본인 책임 아님.

---

## 13. 즉시 시작 체크리스트

작업 시작 시 다음을 순서대로 처리:

- [ ] git repo clone, `pip install -e .` 성공
- [ ] `python -c "from src.risk_map.risk_map_class import StaticRiskMap; print('OK')"` 통과
- [ ] `data/processed/dataset.h5` 다운로드 완료 (Drive 링크 별도 공유)
- [ ] `data/raw/s_010/` 다운로드 완료 (시나리오 1 개라도 일단)
- [ ] PyBullet GUI 가 본인 머신에서 정상 동작 (`pybullet.connect(p.GUI)`)
- [ ] 본 명세서 §1, §3, §5, §6 정독
- [ ] M1 착수 → 5일 내 첫 시연

---

> *명세서 버전 1.0 — 2026-05-13. 변경 시 `docs/decisions.md` 에 D-NNN 으로 기록.*
