# 10 — 시스템 아키텍처

> Tier 1 + Tier 2 의 모듈 구조, 데이터 흐름, 인터페이스 명세.

---

## 1. 모듈 레이어 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│  Application Layer  (실시간 사용)                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Path Planning (A*) — Tier1RiskMap or Tier2RiskMap query    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────┐
│  Risk Map Layer                │                                    │
│  ┌─────────────────────┐  ┌────┴──────────────┐                     │
│  │  Tier1RiskMap       │  │  StaticRiskMap    │  ← 기존 FDS truth   │
│  │  (39 node danger)   │  │  (60×40×6 grid)   │                     │
│  │  ★ 미구현            │  │  ✅ src/risk_map/  │                     │
│  └─────────────────────┘  └───────────────────┘                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────┐
│  Model Layer                   │                                    │
│  ┌─────────────────────┐  ┌────┴──────────────┐                     │
│  │  Tier 1: GraphGRU   │  │  Tier 2: ConvLSTM │                     │
│  │  ✅ src/tier1/       │  │  / FNO no-PI / PI │                     │
│  │  tier1_gnn.py       │  │  ✅ src/models/    │                     │
│  └─────────────────────┘  └───────────────────┘                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────┐
│  Data Preprocessing Layer      │                                    │
│  ┌──────────────────────────┐ │ ┌──────────────────────────────┐    │
│  │  Tier 1:                 │ │ │  Tier 2:                     │    │
│  │  detector_extractor.py   │ │ │  sparsify_input              │    │
│  │  (D-023 trigger)         │ │ │  (cell mask)                 │    │
│  │  build_binary_sequence   │ │ │  + geodesic IDW interp       │    │
│  └──────────────────────────┘ │ └──────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────┐
│  Raw Data Layer                │                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  FDS Simulation → SLCF slices → extract_slices              │    │
│  │  (T/V/CO at 31 frames × 60×40×6 grid, raw °C/m/ppm)         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 핵심 데이터 흐름 — Tier 1 (Binary GNN)

```
[Step 1] FDS truth
  src/data_pipeline/fds_extractor.py
  extract_slices(scenario_dir)
  → {"temperature": (31, 60, 40, 6) °C,
     "visibility":  (31, 60, 40, 6) m,
     "co":          (31, 60, 40, 6) ppm}

[Step 2] Detector trigger (D-023)
  src/tier1/detector_model.py
  extract_detector_events(temp, vis, detector_positions)
  → List[DetectorEvent]  (각 노드의 첫 trigger 시각)
  build_binary_sequence(events)
  → binary: (31, N=39) latched 0/1

[Step 3] Truth danger at sensor positions
  src/risk_map/tenability.py
  compute_total_danger(temp, vis, co)
  → (31, 60, 40, 6)
  scripts/build_detector_sequences.py
  node_danger_at_sensors(...)
  → node_danger: (31, 39) — truth target

[Step 4] Save .npz per scenario
  results/detector_sequences/<scenario>.npz
  {
    binary_sequence: (31, 39),
    node_danger:     (31, 39),  # GNN target
    activation_times: (39,),
    trigger_reasons:  (39,),
    detector_ids:     (39,),
  }

[Step 5] Sliding window dataset
  src/tier1/tier1_dataset.py
  Tier1FireDataset(seq_dir, T_in=6, T_out=6)
  → x: (N=39, T_in=6, F=5)  per sample
       F = [is_detected, det_time_norm, type_onehot × 3]
    y: (N=39, T_out=6)
       node danger over 60s future

[Step 6] Model forward
  src/tier1/tier1_gnn.py
  SimpleFireGNN(in_feat=5, hidden=32, T_out=6)
  forward(x, adj) — adj from build_knn_adjacency(k=4)
  → pred: (N=39, T_out=6) ∈ [0, 1]

[Step 7] (미구현) Tier1RiskMap.query(xyz, t)
  src/tier1/tier1_risk_map.py  ★
  - 가장 가까운 sensor node 찾기
  - 해당 노드의 pred danger 반환
  - A* edge weight 로 사용
```

---

## 3. 핵심 데이터 흐름 — Tier 2 (Continuous + 보간)

```
[Step 1-2] FDS truth + 정규화
  src/data_pipeline/fds_extractor.py + normalize.py
  → (31, 5, 60, 40, 6) normalized input
    channels: [T, V, CO, mask, time_enc]
    (training 시 dataset.h5 에 저장)

[Step 3] Sparse sampling (실 deployment 시 sensor 측정값)
  39 cell × 6 z-layer = 234 nonzero cells (1.6%)
  나머지 14,166 cells: T/V/CO = 0, mask + time_enc 는 그대로

[Step 4] Spatial interpolation (벽 인식)
  scripts/evaluate_sparse_sensing_geodesic.py
  geodesic_idw_interpolate(sensor_values, geo_dist, p=2)
  → 보간된 (60, 40) plane → broadcast over z → (60, 40, 6)
  ※ geodesic distance: BFS on fluid cells (벽 우회)
  ※ IDW weight: 1 / (d^p + eps)

[Step 5] 모델 forward (autoregress)
  ConvLSTM 또는 FNO no-PI/PI
  input: (5, 60, 40, 6) at t₀
  output: (3, 60, 40, 6) at t₀+10s
  자기 출력 chaining 으로 60s lookahead (6 step)

[Step 6] Risk map 생성
  src/risk_map/converter.py
  prediction_to_danger(pred, times)
  → (T_steps, 60, 40, 6) danger ∈ [0, 1]
  StaticRiskMap(danger_array, times)
    .query(xyz, t) → danger
```

---

## 4. 인터페이스 명세 (요약)

상세는 [`interface_contracts.md`](interface_contracts.md) 참조.

### 4.1 RiskMap 추상 인터페이스

```python
class RiskMap(ABC):
    @abstractmethod
    def query(
        self,
        xyz: np.ndarray,           # (3,) or (N, 3) world m
        t: float | None = None,    # seconds
    ) -> float | np.ndarray:
        """danger ∈ [0, 1].
        OOB → 1.0 (safety default).
        OOT → 1.0.
        """
```

구현체:
| Class | 입력 | 상태 |
|---|---|---|
| `StaticRiskMap` | FDS truth slices | ✅ `src/risk_map/risk_map_class.py` |
| `Tier1RiskMap` | GNN node danger | ★ 미구현 (`src/tier1/tier1_risk_map.py` 추후) |
| Tier 2 wrapper | ConvLSTM/FNO output | converter 활용 |

### 4.2 Tier 1 GNN forward 시그니처

```python
class SimpleFireGNN(nn.Module):
    def forward(
        self,
        x: torch.Tensor,         # (B, N=39, T_in=6, F=5)
        adj: torch.Tensor,       # (N, N) normalized adjacency
    ) -> torch.Tensor:           # (B, N, T_out=6) ∈ [0, 1]
```

### 4.3 ConvLSTM/FNO forward 시그니처

```python
class FireConvLSTM(nn.Module) / FNOFireModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,         # (B, 5, 60, 40, 6) ∈ [0, 1]
    ) -> torch.Tensor:           # (B, 3, 60, 40, 6) ∈ [0, 1]
```

---

## 5. 학습 / 평가 / 시각화 스크립트 인덱스

### 학습 (Training)

| 스크립트 | 입력 | 출력 |
|---|---|---|
| `src/training/train_conv_lstm.py` | dataset.h5 (33 train) | checkpoints/conv_lstm/best.pt |
| `src/training/train_fno.py` | dataset.h5 + `--use-pi` 옵션 | checkpoints/fno_{no_pi,pi}/best.pt |
| `scripts/train_sparse_conv_lstm.py` | dataset.h5 + 39 sensor mask | checkpoints/conv_lstm_sparse_v3/best.pt |
| `scripts/train_tier1_gnn.py` | results/detector_sequences/*.npz | checkpoints/tier1_gnn_v3/best.pt |

### 평가 (Evaluation)

| 스크립트 | 평가 대상 | 출력 |
|---|---|---|
| `scripts/evaluate_t_locations.py` | ConvLSTM/FNO on OOD T01-T05 | figures + CSV per model |
| `scripts/evaluate_detector_triggered.py` | trigger-based start (C1+D1) | figures + CSV |
| `scripts/evaluate_sparse_sensing_geodesic.py` | sparse interp + 3 models | figures + CSV |
| `scripts/evaluate_sparse_model.py` | sparse-retrained ConvLSTM | figures + CSV |
| `scripts/hypothesis_validation.py` | 3-model comparison | layer table + plots |

### 시각화 (Visualization)

| 스크립트 | 무엇을 |
|---|---|
| `scripts/visualize_sensor_layout.py` | 39 sensor 평면도 (Paper Figure A) |
| `scripts/visualize_tier1_predictions.py` | **Tier 1 GNN headline figure** (Paper Figure 1) |
| `scripts/visualize_60s_prediction.py` | autoregress 60s rollout (Paper Figure 추가) |
| `scripts/visualize_risk_comparison.py` | FDS vs ConvLSTM risk map 동영상 |
| `scripts/extract_stl_floor.py` | STL → 평면도 단면 추출 |

### 유틸 (Utility)

| 스크립트 | 무엇을 |
|---|---|
| `scripts/build_detector_sequences.py` | 46 시나리오 → binary_sequence.npz (Tier 1 입력 생성) |
| `scripts/generate_scenarios.py` | Jinja2 template → 33 .fds 파일 |
| `scripts/fix_pyrosim_fds.py` | PyroSim export 버그 자동 패치 |
| `scripts/_migrate_raw_to_canonical.py` | sim_* → s_NNN rename |

---

## 6. 의존성 & 환경

| Layer | 라이브러리 |
|---|---|
| 핵심 | Python 3.10+, PyTorch 2.0+, numpy, h5py |
| FDS 데이터 | `fdsreader` (단, OOB 버그 우회로 자체 raw parser `fds_extractor.py` 사용) |
| FNO | `neuraloperator` 2.0 (`pip install neuraloperator>=2.0.0`) |
| GNN | **PyTorch only** (PyG / PyG-Temporal 미사용 — Python 3.12 wheel 호환 회피) |
| STL | `trimesh` |
| 보간 | `scipy.interpolate.griddata` (linear), `BFS` (geodesic 자체 구현) |
| 시각화 | matplotlib (Agg backend), Pillow (GIF) |
| 학습 추적 | Weights & Biases (`wandb`) |

→ **PyG 의존성 없는 가벼운 환경**. 사용자 VS Code (Python 3.12 + PyTorch 2.11+cpu) 에서 바로 동작.
