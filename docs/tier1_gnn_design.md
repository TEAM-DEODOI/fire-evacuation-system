# Tier 1: GNN-Based Fire Spread Estimation from Binary Detectors

> 이 문서는 Tier 1 시스템의 설계, 학습 방법, 구현 가이드를 담습니다.
>
> **Tier 1이 필요한 이유**: Tier 2 (PI-FNO)는 연속 센서 데이터(온도장,
> CO 농도장)가 필요합니다. 실제 건물에는 그런 데이터가 없고 화재 감지기의
> **이진 신호** (작동/미작동)만 있습니다. Tier 1은 이 이진 신호만으로
> 구역별 위험도를 예측합니다.
>
> 관련 문서:
> - `CLAUDE.md` — 프로젝트 전체 컨텍스트
> - `docs/interface_contracts.md` — RiskMap 인터페이스
> - `docs/decisions.md` — D-020, D-021

---

## 목차

1. [개요 및 위치](#1-개요-및-위치)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [학습 데이터 구성](#3-학습-데이터-구성)
4. [모델 설계](#4-모델-설계)
5. [구현 가이드](#5-구현-가이드)
6. [RiskMap 통합](#6-riskmap-통합)
7. [Tier 1 → Tier 2 핸드오프](#7-tier-1--tier-2-핸드오프)
8. [실험 계획](#8-실험-계획)
9. [Plan B 및 우선순위](#9-plan-b-및-우선순위)

---

## 1. 개요 및 위치

### 전체 시스템에서 Tier 1의 위치

```
실제 배포 시나리오:

화재 감지기 (이진 신호)          연속 센서 (고급 설치)
       │                               │
       ▼                               ▼
  [Tier 1 GNN]                  [Tier 2 PI-FNO]
  구역별 위험도                  격자 전체 위험도
  (저해상도, 즉시)               (고해상도, 센서 필요)
       │                               │
       └───────────┬───────────────────┘
                   ▼
         [RiskMap 인터페이스]
         risk_map.query(xyz, t)
                   │
                   ▼
         [A* 경로 계획기]
                   │
                   ▼
              [대피 경로]
```

### 두 Tier의 비교

| 항목 | Tier 1 (GNN) | Tier 2 (PI-FNO) |
|------|-------------|-----------------|
| 입력 | 감지기 이진 신호 (0/1) | 격자 전체 T/V/CO 연속값 |
| 출력 해상도 | 노드 단위 (16~20개 구역) | 격자 단위 (60×40×6) |
| 추가 하드웨어 | 불필요 (기존 감지기 활용) | 연속 센서 필요 |
| 반응 속도 | 즉시 (첫 감지 신호에 반응) | 연속 데이터 축적 후 |
| 정밀도 | 낮음 (구역 수준) | 높음 (셀 수준) |
| 적용 대상 | 모든 건물 | 고급 센서 설치 건물 |

### 이 시스템이 해결하는 문제

Tier 2는 배포 시 다음 문제에 직면합니다:

**실제 배포 시 도메인 갭:**

```
학습 시 입력 (FDS):              배포 시 입력 (실제 건물):
(T, V, CO) 격자 전체   vs.       화재 감지기 이진 신호 몇 개
(60, 40, 6) = 14,400 셀          16~20개 점 데이터
```

Tier 1은 이 갭을 우회합니다. 학습도 FDS 데이터에서
추출한 이진 신호를 사용하므로 도메인 갭이 없습니다.

---

## 2. 시스템 아키텍처

### 데이터 흐름

```
FDS 시뮬레이션 출력 (30건)
         │
         ▼
[detector_extractor.py]
FDS DEVC CSV → 이진 감지 이벤트 시퀀스
         │
         ▼
[building_graph.py (재활용)]
NetworkX 그래프 (16~20 노드)
         │
         ▼
[tier1_dataset.py]
PyTorch Geometric 데이터 세트
  - 노드 특징: [is_detected, detection_time_norm, node_type_onehot]
  - 엣지 특징: [distance, width]
  - 레이블: 각 노드의 미래 위험도 시퀀스
         │
         ▼
[tier1_model.py]
A3T-GCN (Attention Temporal Graph Convolutional Network)
         │
         ▼
[tier1_risk_map.py]
Tier1RiskMap (RiskMap 상속)
  - query(xyz, t) → danger [0, 1]
```

### 그래프 구조

경로 계획에 사용하는 건물 그래프(`building_graph.py`)와 **동일한 그래프**를 사용합니다. 별도로 만들 필요 없습니다.

```
노드 (16~20개):
  - 방 중심점
  - 복도 구간 중점
  - 교차로
  - 출구

노드 속성:
  pos: (x, y, z) 미터
  type: 'room' | 'corridor' | 'intersection' | 'exit'
  is_exit: bool
  has_detector: bool      ← Tier 1 전용 추가 속성
  detector_id: str | None ← DEVC ID 매핑

엣지 속성:
  length: float (m)
  width: float  (m)
  base_time: float (s)
```

---

## 3. 학습 데이터 구성

### FDS에서 이진 감지 이벤트 추출

**기존 30건 FDS 시뮬레이션을 그대로 재활용합니다.** 추가 시뮬레이션 불필요.

두 가지 방법이 있습니다.

#### 방법 A: DEVC 추가 (권장, Member A에게 요청)

다음 FDS 시뮬레이션부터 `&DEVC`를 추가해서 각 노드 위치의 온도를 기록합니다.

```fortran
! 노드별 감지기 정의 (building.yaml의 노드 위치와 매핑)
! 높이 2.5m (천장 감지기 위치)
&DEVC XYZ=5.0,3.0,2.5,   QUANTITY='TEMPERATURE', ID='det_room_A' /
&DEVC XYZ=5.0,14.0,2.5,  QUANTITY='TEMPERATURE', ID='det_room_B' /
&DEVC XYZ=12.0,3.0,2.5,  QUANTITY='TEMPERATURE', ID='det_corridor_1' /
&DEVC XYZ=22.0,4.0,2.5,  QUANTITY='TEMPERATURE', ID='det_room_H' /
... (총 16~20개)
```

FDS 출력: `*_devc.csv` — 각 감지기의 시간별 온도 기록

#### 방법 B: SLCF 슬라이스에서 역산 (기존 30건에 적용 가능)

이미 생성된 30건에서 DEVC 없이 감지기 시뮬레이션:

```python
def extract_detector_events_from_slices(
    grid: np.ndarray,           # (31, 60, 40, 6) 온도
    coords: dict,               # fdsreader 좌표 딕셔너리
    detector_positions: list,   # [(x, y, z), ...] 노드별 위치 (m)
    threshold_celsius: float = 60.0,
) -> dict:
    """
    SLCF 온도 격자에서 감지기 이진 이벤트 추출.
    DEVC CSV 없어도 사용 가능.

    Returns:
        {
            'binary_sequence': np.ndarray (31, N_detectors),
                               0 = 미감지, 1 = 감지 (이후 계속 1)
            'activation_times': list[float | None],
                               None = 끝까지 미감지
        }
    """
    N = len(detector_positions)
    binary_seq = np.zeros((31, N), dtype=np.float32)
    activation_times = [None] * N

    x_coords = coords['x']  # (60,) 셀 중심 좌표
    y_coords = coords['y']  # (40,)
    z_coords = coords['z']  # (6,)

    for det_id, (dx, dy, dz) in enumerate(detector_positions):
        # 가장 가까운 격자 인덱스 (cell-centered 오프셋 포함)
        ix = int((dx - 0.25) / 0.5)
        iy = int((dy - 0.25) / 0.5)
        iz = int((dz - 0.25) / 0.5)

        # 범위 clamp
        ix = max(0, min(59, ix))
        iy = max(0, min(39, iy))
        iz = max(0, min(5, iz))

        activated = False
        for t_idx in range(31):
            temp = grid[t_idx, ix, iy, iz]
            if not activated and temp > threshold_celsius:
                activated = True
                activation_times[det_id] = t_idx * 10.0  # 초
            if activated:
                binary_seq[t_idx, det_id] = 1.0

    return {
        'binary_sequence': binary_seq,
        'activation_times': activation_times,
    }
```

**방법 B가 현실적입니다.** 기존 30건을 재처리하면 바로 학습 데이터가 생성됩니다.

### 노드별 레이블 생성

GNN의 예측 타겟: **각 노드의 미래 위험도 시퀀스**.

```python
def aggregate_risk_to_nodes(
    risk_grid: np.ndarray,      # (31, 60, 40, 6) 위험도
    node_cell_mapping: dict,    # {node_id: [(ix, iy, iz), ...]}
) -> np.ndarray:
    """
    격자 위험도 → 노드 위험도.

    각 노드가 차지하는 격자 셀들의 최대값을 노드 위험도로 사용.
    (평균보다 최대값이 안전 측면에서 보수적)

    Returns:
        (31, N_nodes) float32 ∈ [0, 1]
    """
    T, N = 31, len(node_cell_mapping)
    node_risks = np.zeros((T, N), dtype=np.float32)

    for node_id, cells in node_cell_mapping.items():
        for t in range(T):
            node_risks[t, node_id] = max(
                risk_grid[t, ix, iy, iz]
                for ix, iy, iz in cells
            )

    return node_risks
```

### 데이터 증강 (30건 → 150건)

감지기 오작동을 시뮬레이션해서 데이터를 5배 늘립니다.

```python
def augment_binary_sequence(binary_seq: np.ndarray) -> list:
    """
    단일 시나리오 → 5개 변형 생성.

    Args:
        binary_seq: (31, N_detectors) 원본 이진 시퀀스

    Returns:
        list of 5 augmented sequences, each (31, N_detectors)
    """
    augmented = [binary_seq.copy()]  # 원본

    # 1. 감지기 10% 고장 (무작위 미작동)
    noisy = binary_seq.copy()
    drop_mask = np.random.rand(binary_seq.shape[1]) < 0.1
    noisy[:, drop_mask] = 0
    augmented.append(noisy)

    # 2. 감지기 20% 고장
    noisy = binary_seq.copy()
    drop_mask = np.random.rand(binary_seq.shape[1]) < 0.2
    noisy[:, drop_mask] = 0
    augmented.append(noisy)

    # 3. 감지 지연 1스텝 (10초)
    delayed = np.zeros_like(binary_seq)
    delayed[1:] = binary_seq[:-1]
    augmented.append(delayed)

    # 4. 감지 지연 2스텝 (20초)
    delayed = np.zeros_like(binary_seq)
    delayed[2:] = binary_seq[:-2]
    augmented.append(delayed)

    return augmented  # 5개
```

**결과:** 30건 × 5 = 150개 학습 샘플.

---

## 4. 모델 설계

### 권장 모델: A3T-GCN

`torch_geometric_temporal` 라이브러리의 A3T-GCN을 사용합니다.
교통 흐름 예측에서 검증된 아키텍처로, 우리 문제와 구조가 동일합니다.

```
교통: 도로 그래프 + 속도 센서 시계열 → 미래 혼잡도
Tier 1: 건물 그래프 + 감지기 시계열 → 미래 구역 위험도
```

**설치:**
```bash
pip install torch-geometric torch-geometric-temporal
```

### 모델 코드

```python
# src/tier1/tier1_model.py

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import A3TGCN2


class Tier1FireGNN(nn.Module):
    """
    GNN-based fire risk predictor from binary detector signals.

    Input:
        x: (B, N_nodes, T_in, F_in)
            T_in = input window (e.g. 3 steps = 30s)
            F_in = node features:
                [is_detected, detection_time_norm, node_type_onehot × 4]
                = 6 features total

    Output:
        (B, N_nodes, T_out)
            T_out = prediction horizon (e.g. 6 steps = 60s)
            each value: danger ∈ [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 6,       # node feature dim
        out_channels: int = 32,     # hidden dim
        periods: int = 6,           # prediction steps (60s)
        batch_size: int = 4,
    ):
        super().__init__()

        self.a3tgcn = A3TGCN2(
            in_channels=in_channels,
            out_channels=out_channels,
            periods=periods,
            batch_size=batch_size,
        )

        # Prediction head: hidden → danger score per timestep
        self.output_head = nn.Sequential(
            nn.Linear(out_channels, 16),
            nn.ReLU(),
            nn.Linear(16, periods),
            nn.Sigmoid(),  # danger ∈ [0, 1]
        )

    def forward(
        self,
        x: torch.Tensor,        # (B, N, T_in, F)
        edge_index: torch.Tensor,  # (2, E)
        edge_weight: torch.Tensor, # (E,)
    ) -> torch.Tensor:
        """
        Returns:
            (B, N, T_out) — predicted danger per node per timestep
        """
        # A3TGCN2: (B, N, T_in, F) → (B, N, hidden)
        h = self.a3tgcn(x, edge_index, edge_weight)

        # (B, N, hidden) → (B, N, T_out)
        out = self.output_head(h)

        return out
```

### 하이퍼파라미터

```yaml
# configs/tier1_gnn.yaml

model:
  in_channels: 6          # [is_detected, det_time, node_type × 4]
  out_channels: 32        # hidden dim
  periods: 6              # 60s prediction horizon (6 × 10s)
  batch_size: 4

data:
  input_window: 3         # 과거 30초 (3스텝) 보고 미래 60초 예측
  detector_threshold_c: 60.0  # 감지기 작동 임계 온도

training:
  learning_rate: 1.0e-3
  weight_decay: 1.0e-5
  epochs: 100
  early_stopping_patience: 20
  loss: mse               # 위험도 회귀

augmentation:
  enabled: true
  fault_rates: [0.1, 0.2]      # 감지기 고장률
  delay_steps: [1, 2]          # 감지 지연 스텝
```

### 노드 특징 벡터 구성

각 노드의 입력 특징 (`F_in = 6`):

| Index | 특징 | 설명 |
|-------|------|------|
| 0 | `is_detected` | 현재 시점에서 감지기 작동 여부 (0 or 1) |
| 1 | `detection_time_norm` | 최초 감지 시각 / 300 (미감지시 0) |
| 2~5 | `node_type_onehot` | room/corridor/intersection/exit 원핫 |

```python
def build_node_features(
    binary_seq: np.ndarray,      # (31, N)
    activation_times: list,      # [float|None, ...]
    node_types: list,            # ['room', 'corridor', ...]
    t_current: int,              # 현재 시점 인덱스
    T_window: int = 3,           # 입력 윈도우 크기
) -> np.ndarray:
    """
    Returns: (N, T_window, F_in)
    """
    TYPE_MAP = {'room': 0, 'corridor': 1, 'intersection': 2, 'exit': 3}
    N = len(node_types)
    F = 6
    result = np.zeros((N, T_window, F), dtype=np.float32)

    for t_offset in range(T_window):
        t_idx = max(0, t_current - (T_window - 1 - t_offset))

        for n_id, n_type in enumerate(node_types):
            # 감지 여부
            result[n_id, t_offset, 0] = binary_seq[t_idx, n_id]

            # 최초 감지 시각
            act_t = activation_times[n_id]
            result[n_id, t_offset, 1] = (act_t / 300.0) if act_t is not None else 0.0

            # 노드 타입 원핫
            type_idx = TYPE_MAP.get(n_type, 0)
            result[n_id, t_offset, 2 + type_idx] = 1.0

    return result
```

---

## 5. 구현 가이드

### 디렉토리 구조

```
src/tier1/
├── __init__.py
├── detector_extractor.py    ← SLCF → 이진 감지 이벤트 추출
├── tier1_dataset.py         ← PyG Dataset 클래스
├── tier1_model.py           ← A3T-GCN 모델
├── train_tier1.py           ← 학습 진입점
└── tier1_risk_map.py        ← Tier1RiskMap (RiskMap 상속)

configs/
└── tier1_gnn.yaml           ← 하이퍼파라미터
```

### 작업 순서 (Member B 또는 C)

**Step 1: detector_extractor.py 구현 (반나절)**

입력: FDS SLCF 온도 격자 + 노드 좌표 목록
출력: 이진 감지 시퀀스 (31, N) + 감지 시각

자기 테스트: 화재원 근처 노드가 멀리 있는 노드보다 먼저 감지되는지 확인.

**Step 2: tier1_dataset.py 구현 (반나절)**

30건 × 5 증강 = 150 샘플 PyTorch Geometric Dataset.

자기 테스트: 1 배치 로드, 텐서 shape 확인.

**Step 3: tier1_model.py 구현 + forward pass 검증 (반나절)**

자기 테스트: 랜덤 입력 → forward pass → (B, N, 6) shape 확인.

**Step 4: train_tier1.py 구현 + 1 시나리오 과적합 테스트 (1일)**

1건 데이터로 100 에폭 학습 → val loss 0에 수렴해야 코드 정상.

**Step 5: tier1_risk_map.py 구현 (반나절)**

`Tier1RiskMap.query(xyz, t)` 경로 계획기와 동일한 인터페이스.

---

## 6. RiskMap 통합

가장 중요한 설계 원칙: **경로 계획기는 어떤 RiskMap인지 모른다.**

```python
# src/tier1/tier1_risk_map.py

import numpy as np
import scipy.interpolate
from src.risk_map.risk_map_class import RiskMap
from src.shared.building import load_building_graph

class Tier1RiskMap(RiskMap):
    """
    GNN 출력 기반 위험도 맵.

    GNN은 노드 단위 (16~20개) 위험도를 출력하고,
    이를 scipy.interpolate로 연속 공간으로 보간합니다.

    query(xyz, t) 인터페이스는 Tier2 (FDSRiskMap, FNORiskMap)와 동일.
    경로 계획기 코드 변경 없이 교체 가능.
    """

    def __init__(
        self,
        node_risks: np.ndarray,   # (T_out, N_nodes) ∈ [0, 1]
        node_positions: list,     # [(x, y, z), ...] 노드 좌표 (m)
        start_time: float = 0.0,  # 예측 시작 시각 (초)
        dt: float = 10.0,         # 시간 스텝 (초)
    ):
        self.node_risks = node_risks      # (T_out, N)
        self.node_positions = np.array(node_positions)  # (N, 3)
        self.start_time = start_time
        self.dt = dt
        self.T_out = node_risks.shape[0]
        self.t_max = start_time + (self.T_out - 1) * dt

    def query(
        self,
        xyz: np.ndarray,
        t: float | None = None,
    ) -> float | np.ndarray:
        """
        Returns danger ∈ [0, 1] at world coordinate xyz.

        Out-of-bounds → 1.0 (safety default).
        GNN은 노드 단위 예측이므로, 가장 가까운 노드의 위험도를 반환.
        """
        if t is None:
            t = self.t_max

        # 시간 인덱스 계산
        t_idx = int((t - self.start_time) / self.dt)
        t_idx = max(0, min(self.T_out - 1, t_idx))

        # 현재 시점 노드 위험도
        risks_at_t = self.node_risks[t_idx]  # (N,)

        if xyz.ndim == 1:
            return self._query_single(xyz, risks_at_t)
        else:
            return np.array([
                self._query_single(p, risks_at_t) for p in xyz
            ])

    def _query_single(
        self,
        xyz: np.ndarray,   # (3,)
        risks: np.ndarray, # (N,)
    ) -> float:
        """단일 점 쿼리: 가장 가까운 노드의 위험도 반환."""
        # 범위 체크
        if (xyz[0] < 0 or xyz[0] > 30 or
            xyz[1] < 0 or xyz[1] > 20 or
            xyz[2] < 0 or xyz[2] > 3):
            return 1.0

        # 가장 가까운 노드 (XY 거리만 사용, Z는 단일 층)
        dists = np.linalg.norm(
            self.node_positions[:, :2] - xyz[:2], axis=1
        )
        nearest_node = np.argmin(dists)
        return float(risks[nearest_node])

    @classmethod
    def from_model_output(
        cls,
        model_output: "torch.Tensor",   # (B, N, T_out)
        node_positions: list,
        start_time: float = 0.0,
    ) -> "Tier1RiskMap":
        """PyTorch 모델 출력에서 직접 생성."""
        import torch
        risks = model_output[0].detach().cpu().numpy()  # (N, T_out)
        risks_T = risks.T  # (T_out, N)
        return cls(risks_T, node_positions, start_time)
```

### 경로 계획기에서 Tier1RiskMap 사용

```python
# 기존 Tier 2 코드
tier2_map = FDSRiskMap.from_directory("data/raw/scenario_000")
path = planner.plan(start_xyz, tier2_map, graph, t=0.0)

# Tier 1으로 교체 — 코드 한 줄만 변경
tier1_map = Tier1RiskMap.from_model_output(model_output, node_positions)
path = planner.plan(start_xyz, tier1_map, graph, t=0.0)
```

---

## 7. Tier 1 → Tier 2 핸드오프

### 핵심 시나리오

```
t=0:  화재 감지기 3개 작동
        → Tier 1 즉시 활성화
        → 초기 대피 경로 제공 (구역 수준)

t=30s: 연속 센서 데이터 축적 (보간 초기화 완료)
        → Tier 2 활성화 준비

t=60s: Tier 2가 충분한 데이터 확보
        → Tier 1 → Tier 2 전환
        → 정밀한 격자 위험도 기반 경로 갱신
```

### 핸드오프 구현

```python
class AdaptiveRiskMap(RiskMap):
    """
    Tier 1 → Tier 2 자동 전환 맵.

    초기에는 Tier1RiskMap을 사용하다가,
    Tier 2가 준비되면 자동으로 FNORiskMap으로 전환.
    """

    def __init__(
        self,
        tier1_map: Tier1RiskMap,
        switch_time: float = 60.0,
    ):
        self.tier1_map = tier1_map
        self.tier2_map = None
        self.switch_time = switch_time
        self.active = "tier1"

    def activate_tier2(self, tier2_map: "FNORiskMap"):
        """Tier 2가 준비되면 호출."""
        self.tier2_map = tier2_map
        self.active = "tier2"

    def query(self, xyz, t=None):
        if self.active == "tier1" or self.tier2_map is None:
            return self.tier1_map.query(xyz, t)
        else:
            return self.tier2_map.query(xyz, t)
```

### 발표 메시지

> "기존 화재 감지기만 설치된 건물에서도 즉시 대피 경로를 제공합니다.
> 고급 연속 센서가 추가 설치된 경우, 60초 후 자동으로 더 정밀한
> AI 예측 기반 경로로 업그레이드됩니다."

---

## 8. 실험 계획

### EXP-TIER1-001: Tier 1 성능 단독 검증

| 항목 | 내용 |
|------|------|
| 목적 | 이진 감지기만으로 의미 있는 위험도 예측이 가능한가 |
| 비교 | 정적 대피 계획 vs Tier1 기반 동적 계획 |
| 데이터 | OOD 3건 (학습에 없는 화재 위치) |
| 지표 | 노드 위험도 예측 정확도 (MSE), 누적 FED, 대피 성공률 |
| 성공 기준 | Tier1 FED < 정적 FED (H6 보조 검증) |

### EXP-TIER1-002: Tier 1 vs Tier 2 비교

| 항목 | 내용 |
|------|------|
| 목적 | 추가 센서 없이 Tier 1이 얼마나 Tier 2에 근접하는가 |
| 비교 | Tier1RiskMap vs FNORiskMap (동일 시나리오) |
| 지표 | 경로 선택 일치율, 최종 FED 차이 |
| 의미 | "추가 하드웨어 없이 X% 성능" 메시지 |

### EXP-TIER1-003: 감지기 고장 강건성

| 항목 | 내용 |
|------|------|
| 목적 | 감지기 N%가 고장났을 때 얼마나 견디는가 |
| 변수 | 고장률 0%, 10%, 20%, 30% |
| 지표 | FED 증가율 |
| 발표 메시지 | "감지기 30% 고장 시에도 정상 경로의 X% 성능 유지" |

---

## 9. Plan B 및 우선순위

### 우선순위

Tier 1은 **EXP-FIRE-001, EXP-RISK-001, EXP-PATH-001 이후 작업**입니다.
주요 실험이 완료된 후 시간이 남을 때 진행합니다.

```
필수:
  EXP-FIRE-001 (Week 9)
  EXP-RISK-001 (Week 10)
  EXP-PATH-001 (Week 12)

선택 (시간 남으면):
  Tier 1 GNN (Week 11~12 병행 가능)
  EXP-TIER1-001, 002, 003
```

### 시간 없을 때 드랍 순서

1. EXP-TIER1-003 (감지기 고장 실험)
2. EXP-TIER1-002 (Tier 1 vs Tier 2 비교)
3. Tier 1 전체 — CLAUDE.md Plan B에 이미 기록됨

### 최소 구현으로 발표에 포함하려면

전체 구현이 어려우면 다음 최소 버전만으로도 발표에서 언급 가능합니다.

```python
class SimpleTier1RiskMap(RiskMap):
    """
    최소 버전: GNN 없이 규칙 기반.
    감지된 노드와 인접 노드를 즉시 위험으로 표시.
    """

    def __init__(self, binary_events: dict, graph: nx.Graph):
        self.detected_nodes = set(binary_events.keys())
        self.danger_zone = self.detected_nodes.copy()
        # 인접 노드도 위험 (BFS 1-hop)
        for node in self.detected_nodes:
            self.danger_zone.update(graph.neighbors(node))

    def query(self, xyz, t=None):
        nearest = self._nearest_node(xyz)
        return 0.9 if nearest in self.danger_zone else 0.1
```

이 버전은 GNN 학습 없이도 "감지기 신호 → 경로 계획"의 파이프라인을
시연할 수 있어 발표에서 Tier 1 개념 설명 데모로 사용 가능합니다.

---

## 관련 결정사항

`docs/decisions.md`에 추가할 항목:

```markdown
## D-020: Tier 1 이진 감지 이벤트 추출 방법

**Decision**: DEVC CSV보다 SLCF 슬라이스에서 역산.
기존 30건 재활용 가능, 추가 시뮬레이션 불필요.
Member A에게 향후 시뮬레이션에는 &DEVC 추가 요청.

## D-021: Tier 1 노드 위험도 집계 방법

**Decision**: 각 노드가 커버하는 격자 셀들의 최대값 사용.
평균보다 보수적이며 안전 측면에서 바람직.
FNR (위험 놓치는 비율) 최소화.
```

---

## 참고 자료

- **A3T-GCN 논문**: Zhu et al. 2020 — "A3T-GCN: Attention Temporal Graph
  Convolutional Network for Traffic Forecasting"
  (IEEE TITS, 교통 예측 → 우리 케이스에 직접 적용 가능)

- **torch_geometric_temporal**: 공식 문서 및 예제
  https://pytorch-geometric-temporal.readthedocs.io/

- **유사 연구**: GNN 기반 건물 화재 탐지
  - Khalid et al. 2021 — "Fire Detection using Graph Neural Networks"
  - Kim et al. 2023 — "Spatiotemporal GNN for Fire Spread Prediction"

- **건물 그래프**: `src/path_planning/building_graph.py`
  (Tier 1과 경로 계획기가 동일 그래프 공유)
