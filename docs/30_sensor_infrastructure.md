# 30 — Sensor Infrastructure (D-024 v3.3 + D-023)

> 39 sensor 위치 정의 + 트리거 모델. **Tier 1 과 Tier 2 가 공유**.

---

## 1. 핵심 결정 요약

| 결정 ID | 내용 |
|---|---|
| **D-024 v3.3** | 39 sensor 위치 확정 (평면도 기반) — Tier 1 & Tier 2 공유 |
| **D-023** | 트리거 모델: 열 (T > 60°C) OR 연기 (V < 10 m), latched, CO 제외 |

---

## 2. D-024 v3.3 — 39 Sensor 위치

**구현**: `src/tier1/detector_positions.py` (ALL_DETECTORS, 인덱스 순서 고정)

### 2.1 영역별 분포

| 영역 | 개수 | 위치 |
|---|---|---|
| Zone B (남측 방, Y=2.5) | 5 | (2.5, 6.5, 10, 14, 19) |
| Zone A (사선 통로 안 작은 방) | 5 | zone_a_1~5 |
| North (북측 상단 방) | 4 | north_room_2~5 |
| Zone C (중간 row, Y=15) | 5 | zone_c_mid_1~5 |
| Zone D (우측 X=18) | 3 | zone_d_1~3 |
| **rooms 합계** | **22** | |
| South corridor (Y=5) | 6 | south_corridor_1~6 |
| East corridor (X=16) | 2 | east_corridor_2, _3 |
| Diagonal corridor (Zone A) | 3 | diag_corridor_2~4 |
| North corridor (Y=16) | 3 | north_corridor_1~3 |
| **corridors 합계** | **14** | |
| Exits | 3 | exit_1_west, exit_2_north, exit_3_east |
| **TOTAL** | **39** | |

### 2.2 좌표 명세 (z = 2.5 m 천장)

**Zone B (남측, Y=2.5)**:
```
zone_b_1   (2.5, 2.5)   zone_b_2  (6.5, 2.5)   zone_b_3 (10.0, 2.5)
zone_b_4  (14.0, 2.5)   zone_b_5 (19.0, 2.5)
```

**Zone A (사선 통로 안)**:
```
zone_a_1   (1.5,  9.0)   zone_a_2 (4.0, 13.0)   zone_a_3 (5.0, 15.0)
zone_a_4   (7.5, 17.0)   zone_a_5 (10.5, 18.0)
```

**North rooms**:
```
north_room_2 (17.5, 18.5)   north_room_3 (21.5, 18.5)
north_room_4 (25.0, 18.0)   north_room_5 (28.0, 18.0)
```

**Zone C mid row (Y=15)**:
```
zone_c_mid_1 (18.0, 15)   zone_c_mid_2 (21.5, 15)   zone_c_mid_3 (24.0, 15)
zone_c_mid_4 (27.0, 15)   zone_c_mid_5 (29.0, 15)
```

**Zone D (X=18)**:
```
zone_d_1 (18.0, 12.5)   zone_d_2 (18.0, 9.5)   zone_d_3 (18.0, 7.0)
```

**Corridors**:
```
south_corridor_1 (2.5, 5)   ~6 (21.5, 5) — Y=5 균등
east_corridor_2 (16.5, 12.5)   east_corridor_3 (16.5, 9.5)
diag_corridor_2 (5.0, 11.5)   diag_corridor_3 (7.5, 14)   diag_corridor_4 (9.5, 16)
north_corridor_1 (21.5, 16)   north_corridor_2 (25, 16)   north_corridor_3 (28, 16)
```

**Exits**:
```
exit_1_west (2.5, 5.75)   exit_2_north (15.0, 16.75)   exit_3_east (22.0, 5.75)
```

### 2.3 모듈 인터페이스

```python
from src.tier1.detector_positions import (
    ALL_DETECTORS,                          # list[DetectorLocation] (39)
    get_detector_positions_legacy_format,   # [(id, (x, y, z)), ...]
    get_detector_by_id,
    get_detectors_by_area,                  # "zone_a" | "zone_b" | ... | "exit"
    detector_count_by_area,
)

@dataclass(frozen=True)
class DetectorLocation:
    detector_id: str
    position: tuple[float, float, float]    # (x, y, 2.5)
    node_type: Literal["room", "corridor", "exit"]
    area: str
    description: str
```

### 2.4 표준 준수

| 표준 | 우리 위치 | 일치도 |
|---|---|---|
| NFPA 72 spacing | 9 m 이내 권장 | ✅ 실측 최대 7 m |
| 설치 높이 | 천장 가까이 | ✅ z = 2.5 m |
| 방 1개당 감지기 | 1개 (작은 방 표준) | ✅ |

### 2.5 시각화

`figures/current/01_sensor_layout/`:
- `sensor_layout.png` — 39 sensor + 벽 + 출구
- `sensor_layout_annotated.png` — 모든 sensor ID 라벨 포함

생성 스크립트: `scripts/visualize_sensor_layout.py`

---

## 3. D-023 — 트리거 모델

**구현**: `src/tier1/detector_model.py`

### 3.1 트리거 조건

```python
HEAT_THRESHOLD_C  = 60.0   # NFPA 57°C + 한국 KOFEIS 70°C 중간값
SMOKE_THRESHOLD_M = 10.0   # UL 268 13m 의 보수적 사용

is_triggered(temperature_c, visibility_m) = 
    (temperature_c > 60.0) OR (visibility_m < 10.0)
```

**Latched 동작**: 한 번 활성화되면 시뮬레이션 끝까지 1.0 유지 (실제 감지기와 일치).

**CO 제외**: UL 2034 의 시간-가중 임계 (70 ppm × 60+분) 가 300s 시뮬 시간 안에 도달 안 함.

### 3.2 표준 근거

| 표준 | 임계값 | 본 모델 |
|---|---|---|
| NFPA 72 일반 (UL 521) | 57 °C | 60 °C (5% 보수적) |
| 한국 KOFEIS 0301 | 70 °C | 60 °C (14% 빠른 감지) |
| UL 268 7th | 1.5%/ft ≈ 13 m | 10 m (23% 보수적) |

### 3.3 모듈 인터페이스

```python
from src.tier1.detector_model import (
    DetectorEvent,                  # dataclass
    extract_detector_events,        # FDS slices → list[DetectorEvent]
    build_binary_sequence,          # events → (31, 39) float32
    detector_stats,                 # summary dict
    HEAT_THRESHOLD_C, SMOKE_THRESHOLD_M,
)

events = extract_detector_events(
    temperature_grid,        # (31, 60, 40, 6) °C
    visibility_grid,         # (31, 60, 40, 6) m
    detector_positions,      # from get_detector_positions_legacy_format()
)
# 각 event: detector_id, position, activation_frame, activation_time_s, trigger_reason

binary = build_binary_sequence(events)   # (31, 39) latched 0/1
```

### 3.4 트리거 통계 (46 시나리오)

각 시나리오마다 39 중 평균 30-39 sensor trigger:
- 강한 화재 (1500 kW): 39/39 (전체 활성)
- 약한 화재 (500 kW, 코너 위치): 34-38/39
- 평균 첫 trigger 시각: 41-106 초

대부분 smoke 가 먼저 trigger (90%+), heat 는 화재원 직상부만 (1-2 개).

저장 위치:
```
results/detector_sequences/<scenario>.npz
├── binary_sequence  (31, 39) — 0/1 latched
├── node_danger      (31, 39) — FDS truth 의 sensor 위치 danger (GNN target)
├── activation_times (39,)    — 첫 trigger 시각 (s) or -1
├── trigger_reasons  (39,)    — "heat"/"smoke"/"both"/"none"
└── detector_ids     (39,)    — sensor 이름 array
```

생성 스크립트: `scripts/build_detector_sequences.py`

---

## 4. TENABILITY vs D-023 임계값 — 명확한 구분

같은 60 °C 가 두 곳에 나타나지만 **의미가 다름**:

| 구분 | TENABILITY (사람 안전) | D-023 (감지기 작동) |
|---|---|---|
| 값 | `TENABILITY.T_DANGER_C = 60` | `HEAT_THRESHOLD_C = 60` |
| 의미 | 사람 노출 시 위험 시작 | 감지기 트리거 임계 |
| 사용처 | `src/risk_map/tenability.py` | `src/tier1/detector_model.py` |
| 적용 | continuous danger ∈ [0, 1] | binary on/off |

→ **두 임계값 체계는 독립**. 같은 60°C 는 우연한 일치.

---

## 5. Tier 1 (binary) 과 Tier 2 (continuous) 의 공유 인프라

**같은 39 sensor 위치를 두 가지 신호 모드로 사용**:

| Mode | 신호 | 모델 |
|---|---|---|
| Tier 1 | binary on/off (D-023 latched) | GraphGRU |
| Tier 2 | continuous T/V/CO 측정값 | sparse interp + ConvLSTM/FNO |

→ 페이퍼의 strong framing: *"단일 인프라, 두 가지 신호 처리 전략"*.
