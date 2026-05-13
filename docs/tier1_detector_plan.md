# Tier 1 GNN 감지기 모델 통합 계획

> 이 문서는 Tier 1 GNN의 핵심 입력인 **가상 화재 감지기 모델**의
> 통합 계획을 담습니다. 화재 감지기 표준 조사, D-023 결정,
> 실행 워크플로, 발표 통합 메시지까지.
>
> 관련 문서:
> - `docs/tier1_gnn_design.md` — Tier 1 GNN 전체 설계 (모델 아키텍처)
> - `docs/decisions.md` — D-023 결정 항목
> - `docs/risk_indicators.md` — TENABILITY (사람 안전) vs 감지기 임계값 구분
>
> 관련 작업 요청 파일:
> - `tier1_detector_task.md` — Claude Code 작업 요청 (실행)

---

## 목차

1. [핵심 결정 — D-023](#1-핵심-결정--d-023)
2. [화재 감지기 표준 조사](#2-화재-감지기-표준-조사)
3. [우리 시뮬레이션 환경의 한계](#3-우리-시뮬레이션-환경의-한계)
4. [감지기 모델 정의](#4-감지기-모델-정의)
5. [실행 워크플로](#5-실행-워크플로)
6. [first_sim 검증 결과 (예상)](#6-first_sim-검증-결과-예상)
7. [30건 시뮬레이션 적용 계획](#7-30건-시뮬레이션-적용-계획)
8. [Tier 1 GNN 통합](#8-tier-1-gnn-통합)
9. [발표 통합 메시지](#9-발표-통합-메시지)
10. [향후 작업 로드맵](#10-향후-작업-로드맵)

---

## 1. 핵심 결정 — D-023

### 요약

```
열 감지기 임계값:    온도 > 60°C
연기 감지기 임계값:  visibility < 10 m
CO 감지기:           제외 (시뮬레이션 시간 부족)
결합:                OR (둘 중 하나라도 만족 시 작동)
동작:                Latched (한번 활성 → 끝까지 유지)
```

### 결정 근거 한 줄 요약

> **NFPA 72 + 한국 KOFEIS 0301 + UL 268의 보수적 중간값. 우리 300초
> 시뮬레이션의 시간 제약과 화재 감지기 실제 작동 표준 사이의 균형점.**

상세 결정 기록: `docs/decisions.md`의 D-023.

---

## 2. 화재 감지기 표준 조사

### 2.1 열 감지기 (Heat Detector)

| 표준 | 작동 온도 | 적용 |
|---|---|---|
| NFPA 72 일반형 (UL 521) | **57°C (135°F)** | 미국 일반 |
| NFPA 72 고온형 | 90°C (194°F) | 미국 주방/보일러실 |
| 한국 KOFEIS 0301 일반형 | **70°C** | 한국 일반 |
| 한국 KOFEIS 0301 주방용 | 75~80°C | 한국 주방 |

**차동식 (Rate-of-Rise)**:
- 분당 8.3°C (15°F) 이상 온도 상승 시 작동
- 한국 차동식: 분당 6~8°C 상승

### 2.2 연기 감지기 (Smoke Detector)

**UL 268 7th edition 표준**:
- 일반 화재: 1.5%/ft obscuration 이전에 작동
- 격렬한 화재 (폴리우레탄 점화): 5%/ft 이전에 작동

**Visibility로 환산**:
```
obscuration → visibility (m)
1.5%/ft → 약 13 m
3.0%/ft → 약 6.5 m
5.0%/ft → 약 4 m
```

### 2.3 CO 감지기 (Carbon Monoxide Detector)

**UL 2034 표준 — 시간 가중 임계값**:

| CO 농도 | 작동 시간 |
|---|---|
| 70 ppm | 60~240 분 (1~4 시간) |
| 150 ppm | 10~50 분 |
| 400 ppm | 4~15 분 |
| 30 ppm 미만 | 30일까지 작동 안 함 |

**핵심**: 단일 임계값 아닌 시간-농도 곡선. **누적 노출**이 기준.

---

## 3. 우리 시뮬레이션 환경의 한계

### 3.1 first_sim 측정값

300초 시뮬레이션 + 1500 kW 화재 결과 (`first_sim/stats.json`):

| 측정 | 값 |
|---|---|
| Temperature max | 707.8 °C |
| Visibility min | 0.58 m |
| CO max | 422 ppm |
| 시뮬레이션 시간 | 300초 (5분) |

### 3.2 각 감지기의 적용 가능성 분석

| 감지기 | 임계값 | first_sim 도달? | 우리에 적합 |
|---|---|---|---|
| 열 (NFPA 57°C) | 57°C | ✓ ~50초에 도달 | ✓ |
| 열 (한국 70°C) | 70°C | ✓ ~50초에 도달 | ✓ |
| 차동식 | 분당 8.3°C↑ | △ 복잡 구현 | △ 옵션 |
| 연기 (UL 268 13m) | Vis < 13m | ✓ ~50초에 도달 | ✓ |
| CO (UL 2034) | 70ppm × 60+분 | ✗ 시간 부족 | ✗ |

**핵심 문제**:
CO 감지기는 누적 노출이 기준인데, 300초 시뮬레이션은 UL 2034 최단 요구
시간 (4분 = 240초)에 가까움. 학습 신호로 신뢰하기 어려움.

### 3.3 결론

→ **열 + 연기 OR 조합**이 우리 환경에 가장 적합.

---

## 4. 감지기 모델 정의

### 4.1 트리거 조건

```python
def is_triggered(temperature_c, visibility_m):
    """단일 시점에서 감지기 트리거 여부."""
    heat = temperature_c > 60.0   # D-023
    smoke = visibility_m < 10.0   # D-023
    return heat or smoke
```

### 4.2 Latched 동작

```python
def latched_sequence(temperature_series, visibility_series):
    """시간 시리즈 → 이진 시퀀스 (한번 켜지면 계속 1)."""
    triggered = False
    sequence = []
    for t, v in zip(temperature_series, visibility_series):
        if not triggered and is_triggered(t, v):
            triggered = True
        sequence.append(1 if triggered else 0)
    return sequence
```

**왜 Latched인가**:
- 실제 화재 감지기는 자동 reset 안 됨 (수동 reset 필요)
- 학습 데이터 일관성 (한번 감지된 화재가 "사라지지 않음")
- 모델이 "감지된 영역 + 시간 경과"로부터 위험도 추론

### 4.3 트리거 종류 분류

```
heat   = 열만 도달 (가장 빠른 트리거)
smoke  = 연기만 도달
both   = 동일 frame에 둘 다 도달
none   = 트리거 안 됨 (시뮬 끝까지)
```

이 분류는 **분석 + 발표용**. GNN 학습 입력에는 이진 (0/1)만 사용.

### 4.4 임계값 선택의 근거 정리

**열 60°C**:
- NFPA 57°C와 한국 70°C의 산술 평균 정도
- 보수적이면서 실용적
- TENABILITY.T_DANGER_C (60°C)와 우연히 같은 값 — **의미는 완전히 다름**

**연기 10m**:
- UL 268 13m보다 보수적 (확실한 위험 영역만 트리거)
- TENABILITY.V_SAFE_M (10m)과 우연히 같음 — 사람 안전 기준과 일치

**TENABILITY와 D-023 임계값의 의미 차이** (중요):

```
TENABILITY.T_DANGER_C = 60°C  → "사람이 60°C에서 위험"
HEAT_THRESHOLD_C     = 60°C  → "감지기가 60°C에서 작동"

같은 60°C 값이지만:
- TENABILITY: 노출 결과 (사람 안전 평가용)
- D-023: 감지 트리거 (감지기 작동 조건)
- 두 임계값 체계는 독립적
```

매뉴얼 D-023에 이 구분이 명시되어 있습니다.

---

## 5. 실행 워크플로

### 5.1 전체 데이터 흐름

```
1. FDS 시뮬레이션 (각 시나리오)
   └─→ data/raw/{scenario_id}/  (.smv, .sf 파일)

2. extract_slices()
   └─→ temperature_grid (31, 60, 40, 6)
       visibility_grid  (31, 60, 40, 6)

3. extract_detector_events(D-023 적용)
   └─→ List[DetectorEvent] (18개 감지기)

4. build_binary_sequence()
   └─→ (31, 18) 이진 시퀀스 → Tier 1 GNN 학습 입력

5. visualize_detectors (검증)
   └─→ figures/{scenario_id}/detectors/ (3장 PNG + JSON)
```

### 5.2 단일 시나리오 처리 시간

| 단계 | 예상 시간 |
|---|---|
| extract_slices | ~1초 |
| extract_detector_events | ~0.1초 |
| build_binary_sequence | <0.01초 |
| 시각화 3장 | ~3초 |

→ **시나리오당 5초 이내** 완료. 30건 일괄 처리 약 2.5분.

### 5.3 의존성

```
src/tier1/detector_model.py
  ├── src/shared/constants.py (GRID_SHAPE, N_TIMESTEPS)
  ├── src/shared/coordinates.py (world_to_grid)
  └── numpy

scripts/visualize_detectors.py
  ├── src/tier1/detector_model.py
  ├── src/data_pipeline/fds_extractor.py
  └── matplotlib
```

**새 라이브러리 추가 없음** — 기존 라이브러리만 사용.

---

## 6. first_sim 검증 결과 (예상)

### 6.1 18개 가상 감지기 위치

매뉴얼 `building.py` 그래프 노드와 매핑:

```
Zone A (좌상 사선):
  - zone_a_west    (3.0, 13.0, 2.5)
  - zone_a_center  (8.0, 14.0, 2.5)

Zone B (좌하):
  - zone_b_west    (4.0, 4.0, 2.5)
  - zone_b_center  (10.0, 3.0, 2.5)
  - zone_b_east    (18.0, 3.0, 2.5)

Zone C (우상):
  - zone_c_west    (15.0, 16.0, 2.5)
  - zone_c_center  (22.0, 16.0, 2.5)
  - zone_c_east    (28.0, 16.0, 2.5)

Zone D (우하):
  - zone_d_west    (24.0, 5.0, 2.5)
  - zone_d_center  (28.0, 5.0, 2.5)

중앙 홀:
  - hall_n         (15.0, 12.0, 2.5)
  - hall_s         (15.0, 7.0, 2.5)
  - hall_e         (20.0, 9.0, 2.5)
  - hall_w         (10.0, 9.0, 2.5)

교차로:
  - int_north      (12.0, 14.0, 2.5)
  - int_south      (15.0, 5.0, 2.5)

출구:
  - exit_west      (1.0, 5.0, 2.5)
  - exit_north     (8.0, 17.0, 2.5)
  - exit_east      (29.0, 13.0, 2.5)
```

모두 천장 가까이 (z=2.5m) — 실제 화재 감지기 설치 높이.

### 6.2 first_sim 예상 트리거 패턴

화재원 (18, 10) — first_sim의 화재 위치 기준:

| 그룹 | 감지기 | 예상 트리거 종류 | 예상 시간 |
|---|---|---|---|
| 화재원 인접 (~5m) | hall_e (20, 9) | heat 또는 both | 30~50초 |
| 화재원 인접 | hall_s (15, 7) | smoke 먼저 | 50~80초 |
| 중간 거리 (5~10m) | hall_n, hall_w, int_south | smoke | 80~150초 |
| 중간 거리 | zone_c_west, zone_c_center | smoke | 100~200초 |
| 멀리 (10m+) | zone_b, zone_a | smoke (천장 jet 도달 시) | 200~300초 |
| 매우 멀리 | zone_d, exit_west | never 또는 매우 늦음 | 280초 또는 - |

### 6.3 통계 예측

```
n_total: 19 (감지기)
n_triggered: 12~16 (예상)
n_heat_only: 1~2 (화재원 직상부만)
n_smoke_only: 10~14 (대부분)
n_both: 0~2 (화재원 매우 가까운 감지기)
n_never: 3~7 (Zone D, 출구 영역)

mean_activation_time_s: 약 120~180초
earliest_activation_s: 30~50초
latest_activation_s: 280~300초
```

이 패턴이 보이면 D-023 모델 정상 작동.

### 6.4 검증 합격 기준

다음 모두 만족해야 검증 합격:

- [ ] 화재원 인접 감지기 (hall_e, hall_s) 100초 이내 트리거
- [ ] 멀리 떨어진 출구 (exit_west) 트리거 안 되거나 매우 늦게
- [ ] 트리거 종류 분포가 합리적 (heat 1~2개, smoke 다수)
- [ ] 시간 분포가 연속적 (한 두 개만 동시 활성화 아님)
- [ ] OOB 위치 → trigger_reason="none"

---

## 7. 30건 시뮬레이션 적용 계획

### 7.1 일괄 처리 워크플로

30건 시뮬레이션 완료 후:

```bash
# 모든 시나리오에 D-023 모델 일괄 적용
for scenario in data/raw/s_*/; do
    python scripts/visualize_detectors.py "$scenario"
done

# 결과:
# figures/{scenario_id}/detectors/
#   ├── detectors_floorplan.png
#   ├── detectors_gantt.png
#   ├── detectors_summary.png
#   ├── binary_sequence.npy   ← Tier 1 GNN 학습 입력
#   └── events.json
```

### 7.2 다양성 검증

30건 시나리오의 binary_sequence.npy 다양성 확인:

```python
import numpy as np
from pathlib import Path

all_sequences = []
for scenario_dir in sorted(Path("figures").glob("s_*/detectors/")):
    seq = np.load(scenario_dir / "binary_sequence.npy")
    all_sequences.append(seq)
all_sequences = np.stack(all_sequences)  # (30, 31, 18)

# 검증 항목:
# 1. 각 시나리오마다 트리거 패턴이 다른가
# 2. 6개 화재 위치별로 명확히 다른 패턴인가
# 3. HRR 4단계 사이 점진적 차이가 있는가
# 4. OOD 3건이 학습 24건과 다른 패턴인가
```

### 7.3 학습 데이터 분할

매뉴얼 시나리오 분할 그대로 사용:

| 분할 | 시나리오 수 | binary_sequence 사용처 |
|---|---|---|
| Train | 24 | Tier 1 GNN 학습 |
| Val | 3 | Tier 1 GNN 검증 (HRR 보간) |
| Test OOD | 3 | EXP-TIER1-001 OOD 평가 |

---

## 8. Tier 1 GNN 통합

### 8.1 데이터 흐름 (GNN 학습)

```
30개 시나리오의 binary_sequence.npy
        │
        ▼
src/tier1/dataset.py (미작성, Week 11)
  - 30개 시퀀스 + 그래프 → PyTorch Geometric Dataset
  - 입력: (T_in=3, F=6) per node
    F = [is_detected, det_time_norm, node_type_onehot × 4]
  - 출력: (T_out=6,) per node, danger ∈ [0, 1]
        │
        ▼
src/tier1/tier1_model.py (미작성, Week 11)
  - A3T-GCN (torch_geometric_temporal)
  - 매뉴얼 사양: hidden=32, periods=6
        │
        ▼
src/tier1/tier1_risk_map.py (미작성, Week 12)
  - Tier1RiskMap (RiskMap 상속)
  - query(xyz, t) → danger
  - 경로 계획기와 동일 인터페이스
```

### 8.2 노드 특징 (F=6)

각 감지기/노드의 입력 특징:

| Index | 특징 | 출처 |
|---|---|---|
| 0 | is_detected | binary_sequence[t, n] |
| 1 | detection_time_norm | activation_time_s / 300 (미감지 시 0) |
| 2-5 | node_type_onehot | room / corridor / intersection / exit |

D-023 모델이 0, 1 번 특징을 직접 제공. 2-5번은 그래프 정의에서.

### 8.3 출력 (위험도 예측)

```
GNN 입력: 현재 + 과거 2 frame의 이진 감지기 신호
GNN 출력: 미래 6 frame (60초) 노드별 위험도
        │
        ▼
Tier1RiskMap.query(xyz, t)
  - 가장 가까운 노드 찾기
  - 해당 노드의 위험도 반환
```

### 8.4 Tier 2 PI-FNO와의 차이

| 측면 | Tier 1 GNN | Tier 2 PI-FNO |
|---|---|---|
| 입력 | 이진 감지기 (18개) | 연속 T/V/CO 격자 (60×40×6) |
| 추가 센서 | 불필요 (기존 활용) | 필요 (T/V/CO 측정) |
| 출력 해상도 | 노드 단위 (~18개) | 셀 단위 (14,400 cells) |
| 응답 속도 | < 1초 | 10초 (1 frame) |
| 정밀도 | 낮음 (영역 수준) | 높음 (셀 수준) |
| 적용 대상 | 모든 건물 | 고급 센서 설치 건물 |

발표 시 두 시스템의 **상호 보완성** 강조 가능.

---

## 9. 발표 통합 메시지

### 9.1 슬라이드 1: 문제 정의

> "기존 건물에는 화재 감지기만 있고 연속 센서는 없습니다.
> 이 제약 안에서 미래 위험을 예측할 수 있는가?"

### 9.2 슬라이드 2: 표준 준수 표

| 표준 | 우리 D-023 모델 | 일치도 |
|---|---|---|
| NFPA 72 일반형 (57°C) | 60°C | ✓ 5% 보수적 |
| 한국 KOFEIS 0301 (70°C) | 60°C | ✓ 14% 빠른 감지 |
| UL 268 7th (1.5%/ft = 13m) | 10m | ✓ 23% 보수적 |
| UL 2034 (CO) | 제외 | △ 시뮬레이션 시간 제약 |
| **추가 하드웨어 필요** | **없음** | ✓ 기존 인프라 활용 |

### 9.3 슬라이드 3: 18개 감지기 트리거 패턴

`figures/first_sim/detectors/detectors_gantt.png`

> "1500 kW 화재 발생 후 60초 안에 4개 감지기 활성.
> 이 4개 신호만으로 미래 위험 영역을 예측 가능."

### 9.4 슬라이드 4: Tier 1 vs Tier 2 vs 정적 비교

| 시스템 | 입력 | 출력 | 응답성 | 추가 비용 |
|---|---|---|---|---|
| 정적 알고리즘 | 없음 | 사전 정의 경로 | 0초 | 0 |
| **Tier 1 GNN (우리)** | **기존 감지기 18개** | **노드별 위험도** | **< 1초** | **0** |
| Tier 2 PI-FNO | 연속 센서 (T/V/CO) | 셀별 위험도 | 10초 | 고가 |

> "기존 인프라 활용으로 추가 비용 0, 그러나 안전성 X% 개선"

### 9.5 슬라이드 5: H6 가설 검증 (EXP-TIER1-002)

> "Tier 1만 사용 시 Tier 2 대비 안전성 차이는 단지 Y%.
> 즉, 추가 센서 없이도 거의 동등한 효과."

(Y% 값은 실험 후 결정. 예상 10~20%)

---

## 10. 향후 작업 로드맵

### 10.1 단기 (지금~1주)

```
[현재 작업] tier1_detector_task.md 실행 (Claude Code)
  └─→ src/tier1/detector_model.py + scripts/visualize_detectors.py
       + D-023 매뉴얼 추가 + tier1_gnn_design.md 업데이트

[병행] 30건 시뮬레이션 진행 중 (Member A)
```

### 10.2 중기 (Week 11)

```
30건 시뮬레이션 완료
  │
  ▼
[Member B] 30건에 D-023 모델 일괄 적용
  └─→ figures/s_*/detectors/binary_sequence.npy × 30

  │
  ▼
[Claude Code 작업] src/tier1/dataset.py 작성
  └─→ PyTorch Geometric Dataset
       이진 시퀀스 + 그래프 → 학습 페어

  │
  ▼
[Claude Code 작업] src/tier1/tier1_model.py 작성
  └─→ A3T-GCN 모델
       (torch_geometric_temporal 라이브러리)
```

### 10.3 장기 (Week 12)

```
[Claude Code 작업] src/tier1/train.py 작성
  └─→ 학습 24건 + 검증 3건 학습
       Wandb 로깅, checkpoint 저장

  │
  ▼
[Claude Code 작업] src/tier1/tier1_risk_map.py 작성
  └─→ Tier1RiskMap (RiskMap 상속)
       경로 계획기와 통합

  │
  ▼
EXP-TIER1-001 (Tier 1 성능 단독 검증)
EXP-TIER1-002 (Tier 1 vs Tier 2 비교)
EXP-TIER1-003 (감지기 고장 강건성)
```

### 10.4 발표 (Week 13~14)

```
- 슬라이드 1~5 (위 §9 발표 통합 메시지)
- 시각화 영상: 18개 감지기 활성화 + Tier 1 GNN 위험도 예측 동시 표시
- 정량 비교: 정적 vs Tier 1 vs Tier 2 메트릭 표
```

---

## 부록 A: 관련 파일 일람

### 신규 작성 (D-023 적용)

| 파일 | 역할 | 작성 시점 |
|---|---|---|
| `src/tier1/__init__.py` | 패키지 초기화 | tier1_detector_task.md |
| `src/tier1/detector_model.py` | 핵심 모델 | tier1_detector_task.md |
| `scripts/visualize_detectors.py` | 시각화 도구 | tier1_detector_task.md |
| `docs/tier1_detector_plan.md` | **이 문서** | - |

### 매뉴얼 업데이트

| 파일 | 추가 내용 |
|---|---|
| `docs/decisions.md` | D-023 항목 |
| `docs/tier1_gnn_design.md` | 감지기 모델 + 검증 결과 |

### 향후 작성 예정

| 파일 | 역할 | 작성 시점 |
|---|---|---|
| `src/tier1/dataset.py` | PyTorch Dataset | Week 11 |
| `src/tier1/tier1_model.py` | A3T-GCN 모델 | Week 11 |
| `src/tier1/train.py` | 학습 스크립트 | Week 11 |
| `src/tier1/tier1_risk_map.py` | Tier1RiskMap | Week 12 |
| `experiments/exp_tier1_001.py` | OOD 평가 | Week 12 |

---

## 부록 B: D-023 임계값과 TENABILITY의 명확한 구분

### 같은 숫자, 다른 의미

```
TENABILITY.T_DANGER_C = 60°C       (사람 안전 위험도 임계값)
HEAT_THRESHOLD_C      = 60°C       (감지기 작동 임계값)

같은 60°C 값이지만:

┌──────────────────────┬──────────────────────────┐
│ TENABILITY (사람용)    │ D-023 (감지기용)         │
├──────────────────────┼──────────────────────────┤
│ 위험도 d_T = 1.0 시점  │ 감지기 트리거 시점       │
│ 노출 결과 평가         │ 감지 작동 조건           │
│ src/risk_map/          │ src/tier1/               │
│ tenability.py          │ detector_model.py        │
│ compute_danger_T()     │ check_single_cell_trigger│
└──────────────────────┴──────────────────────────┘

같은 60°C가 우연히 일치한 것이고, 두 임계값 체계는 독립적.
```

### 코드에서의 구분

```python
# 사람 안전 (TENABILITY):
from src.shared.constants import TENABILITY
danger = compute_danger_temperature(T)  # TENABILITY.T_DANGER_C 사용
# → 사람이 노출되었을 때 위험도 [0, 1]

# 감지기 작동 (D-023):
from src.tier1.detector_model import HEAT_THRESHOLD_C
triggered = T > HEAT_THRESHOLD_C  # 60°C
# → 감지기가 작동하는지 (boolean)
```

매뉴얼 D-023의 "기존 결정과 관계" 섹션에 이 구분이 명시되어 있습니다.

---

## 부록 C: 차동식 감지기 도입 시 향후 확장

현재 D-023은 정온식 (Fixed Temperature) + 광전식 (Photoelectric)만 사용.
필요 시 차동식 (Rate-of-Rise) 추가 가능:

```python
def check_rate_of_rise(temperature_series, threshold_rate_c_per_min=8.3):
    """차동식 열 감지기 모사.
    
    NFPA 72 기준: 분당 8.3°C 이상 상승 시 작동.
    """
    # dt = 10초이므로 dt_min = 1/6분
    # 분당 8.3°C → 10초당 1.38°C
    rate = np.diff(temperature_series) / (10.0 / 60.0)  # °C/min
    return (rate > threshold_rate_c_per_min).any()
```

**도입 권장 시점**:
- 30건 검증 결과 다양성 부족 시
- EXP-TIER1-003 (감지기 고장 강건성)에서 추가 변형으로

**현재는 도입 안 함**:
- 학부 캡스톤 수준에서 복잡도 증가 부담
- 열 + 연기 OR 조합으로 충분한 학습 신호

---

## 문서 변경 이력

| 일자 | 변경 | 작성자 |
|---|---|---|
| 2025-XX-XX | 초기 작성 (D-023 결정 + 통합 계획) | 사용자 + Claude |
