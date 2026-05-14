# Claude Code 작업 요청 — Tier 1 GNN 화재 감지기 모델 정의 + D-023

> Tier 1 GNN의 감지기 트리거 조건을 화재안전 표준 (NFPA 72, UL 268)에
> 근거한 실제 감지기 모델로 정의. 단순 60°C 임계값 → 열+연기 OR 조합.
> 첫 시뮬레이션 데이터로 즉시 검증 가능.
>
> 새 Claude Code 세션에 통째로 붙여넣기.

---

## 사전 확인

- [ ] Day 1 모듈 (constants, normalization, coordinates) 완성 ✓
- [ ] `src/data_pipeline/fds_extractor.py` 완성 + 검증 통과 ✓
- [ ] `src/shared/building.py` 또는 그래프 정의 가능 ✓
- [ ] first_sim 시뮬레이션 데이터 (data/raw/first_sim/) ✓
- [ ] `docs/tier1_gnn_design.md` 존재 (이전 작업) ✓

---

## 배경 — 핵심 결정의 근거

### 실제 화재 감지기 표준

| 감지기 종류 | 표준 | 임계값 |
|---|---|---|
| 열 감지기 (NFPA 72 일반형) | UL 521 | 57°C (135°F) |
| 열 감지기 (한국 일반형) | KOFEIS 0301 | 70°C |
| 차동식 (Rate-of-Rise) | NFPA 72 | 분당 8.3°C 이상 |
| 연기 감지기 (UL 268) | UL 268 7th | 1.5%/ft obscuration ≈ Vis 13m |
| CO 감지기 (UL 2034) | UL 2034 | 70 ppm @ 60~240분 |

### 우리 시뮬레이션 (300초)에서의 적용 가능성

| 감지기 | 우리에 적합? | 이유 |
|---|---|---|
| 열 (57~70°C) | ✓ | 화재 부근 50초 안에 600°C 도달 |
| 연기 (Vis < 13m) | ✓ | 50~100초 안에 트리거 |
| 차동식 | △ | 구현 복잡, 옵션 |
| CO (70ppm × 60+분) | ✗ | 300초 시뮬레이션 시간 부족 |

**결정 (D-023)**:
- **열 감지기: 60°C** (NFPA 57°C와 한국 70°C 중간값)
- **연기 감지기: Visibility < 10m** (UL 268 13m의 보수적 사용)
- **OR 조합**: 둘 중 하나라도 만족 시 트리거
- **CO 감지기 제외**: 시뮬레이션 시간 부족

---

## 프롬프트 (여기부터 복사)

```
이번 작업은 Tier 1 GNN의 핵심 - 화재 감지기 트리거 모델 정의야.

매뉴얼 D-023 (신규)에 따라:
- 열 감지기: 온도 > 60°C
- 연기 감지기: visibility < 10m
- OR 조합 (둘 중 하나라도 만족 시 작동)
- CO 감지기 제외 (300초 시뮬레이션에서 UL 2034 임계값 도달 못함)

먼저 다음 컨텍스트 파일들을 읽어:

1. CLAUDE.md
2. docs/tier1_gnn_design.md (Tier 1 전체 설계)
3. docs/decisions.md (기존 결정들, D-023 추가 위치)
4. docs/risk_indicators.md (TENABILITY 임계값 참고)
5. src/data_pipeline/fds_extractor.py (extract_slices 사용)
6. src/shared/coordinates.py (world_to_grid)


====================================================================
작업 1: src/tier1/__init__.py + src/tier1/detector_model.py 신규
====================================================================

## 목적
화재 감지기 트리거 조건을 정의하고, FDS 슬라이스 데이터에서
가상 감지기 활성 이벤트를 추출하는 모듈.

## 파일 구조

```
src/tier1/
├── __init__.py
└── detector_model.py    ← 이번 작업
```

src/tier1/__init__.py는 비워두거나 단순 import만:

```python
"""Tier 1 GNN: binary detector signal-based fire risk estimation.

See docs/tier1_gnn_design.md for full design.
"""
```

## 시그니처

```python
# src/tier1/detector_model.py
"""Tier 1 GNN의 화재 감지기 트리거 모델.

매뉴얼 D-023에 따라 실제 화재 감지기 표준을 모사:
- 열 감지기 (NFPA 57°C + 한국 70°C 중간값): 60°C
- 연기 감지기 (UL 268 1.5%/ft ≈ 13m): 10m visibility (보수적)
- OR 조합: 둘 중 하나라도 만족 시 작동
- CO 감지기 제외: UL 2034 임계값 (70ppm × 60+분)이 300초 시뮬에서
  도달 안 함

배경:
실제 건물의 표준 화재 감지기는 보통 열 + 연기 복합 감지기.
일단 활성화되면 거의 다시 비활성화 안 됨 (latched). 우리 모델도 latch.

CO는 시뮬레이션 시간이 짧아 분석 가치 낮음 → 제외.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.shared.constants import GRID_SHAPE, N_TIMESTEPS
from src.shared.coordinates import world_to_grid


# === D-023: 감지기 트리거 임계값 ===
HEAT_THRESHOLD_C: float = 60.0   # 열 감지기 (°C)
SMOKE_THRESHOLD_M: float = 10.0  # 연기 감지기 (m)


@dataclass(frozen=True)
class DetectorEvent:
    """단일 감지기의 트리거 이벤트.

    Attributes:
        detector_id: 감지기 식별자 (예: "zone_a_center" 또는 노드 ID).
        position: (x, y, z) world 좌표 (m).
        activation_frame: 최초 트리거된 시간 frame 인덱스 (0~30). None이면 트리거 안 됨.
        activation_time_s: 최초 트리거된 시간 (초). None이면 트리거 안 됨.
        trigger_reason: "heat", "smoke", "both", "none" 중 하나.
            "heat" = 열 임계값만 도달
            "smoke" = 연기 임계값만 도달
            "both" = 동시 도달 (가장 빨리)
            "none" = 트리거 안 됨
    """
    detector_id: str
    position: tuple[float, float, float]
    activation_frame: Optional[int]
    activation_time_s: Optional[float]
    trigger_reason: str  # "heat" | "smoke" | "both" | "none"

    def is_activated(self) -> bool:
        return self.activation_frame is not None


def check_single_cell_trigger(
    temperature_c: np.ndarray,    # (T,) 시간별 온도
    visibility_m: np.ndarray,      # (T,) 시간별 시정거리
    heat_threshold_c: float = HEAT_THRESHOLD_C,
    smoke_threshold_m: float = SMOKE_THRESHOLD_M,
) -> tuple[Optional[int], str]:
    """단일 셀의 시간 시리즈에서 감지기 트리거 시점 결정.

    Args:
        temperature_c: (T,) 시간별 온도 (°C).
        visibility_m: (T,) 시간별 visibility (m).
        heat_threshold_c: 열 감지기 임계값. 기본 60°C (D-023).
        smoke_threshold_m: 연기 감지기 임계값. 기본 10m (D-023).

    Returns:
        (activation_frame, reason)
        - activation_frame: 최초 트리거 frame (0-indexed). None이면 트리거 안 됨.
        - reason: "heat", "smoke", "both", "none"

    Logic:
        OR 조합: 두 조건 중 하나라도 만족 시 트리거.
        가장 빨리 도달한 시점이 activation_frame.
        만약 동일 frame에서 둘 다 도달 → "both".
    """
    if temperature_c.shape != visibility_m.shape:
        raise ValueError(
            f"shape mismatch: T={temperature_c.shape}, V={visibility_m.shape}"
        )
    if temperature_c.ndim != 1:
        raise ValueError(
            f"both inputs must be 1-D, got {temperature_c.ndim}-D"
        )

    heat_triggered = temperature_c > heat_threshold_c
    smoke_triggered = visibility_m < smoke_threshold_m

    heat_first = np.argmax(heat_triggered) if heat_triggered.any() else -1
    smoke_first = np.argmax(smoke_triggered) if smoke_triggered.any() else -1

    if heat_first == -1 and smoke_first == -1:
        return None, "none"
    elif heat_first == -1:
        return int(smoke_first), "smoke"
    elif smoke_first == -1:
        return int(heat_first), "heat"
    elif heat_first == smoke_first:
        return int(heat_first), "both"
    elif heat_first < smoke_first:
        return int(heat_first), "heat"
    else:
        return int(smoke_first), "smoke"


def extract_detector_events(
    temperature_grid: np.ndarray,   # (T, X, Y, Z) °C
    visibility_grid: np.ndarray,    # (T, X, Y, Z) m
    detector_positions: list[tuple[str, tuple[float, float, float]]],
    dt_seconds: float = 10.0,
    heat_threshold_c: float = HEAT_THRESHOLD_C,
    smoke_threshold_m: float = SMOKE_THRESHOLD_M,
) -> list[DetectorEvent]:
    """FDS 슬라이스에서 가상 감지기 이벤트 추출.

    Args:
        temperature_grid: (T, 60, 40, 6) 온도 (°C).
        visibility_grid: (T, 60, 40, 6) visibility (m).
        detector_positions: [(detector_id, (x, y, z)), ...] world 좌표 (m).
        dt_seconds: 시간 frame 간격 (기본 10초).
        heat_threshold_c: 열 임계값. 기본 60°C.
        smoke_threshold_m: 연기 임계값. 기본 10m.

    Returns:
        list of DetectorEvent (모든 감지기에 대해 하나씩).

    Note:
        OOB 위치 (도메인 밖) 감지기는 trigger_reason="none"으로 반환.
    """
    T = temperature_grid.shape[0]
    if visibility_grid.shape != temperature_grid.shape:
        raise ValueError(
            f"T grid {temperature_grid.shape} != V grid {visibility_grid.shape}"
        )

    events = []
    for det_id, pos in detector_positions:
        # world → grid index
        idx = world_to_grid(np.asarray(pos))
        ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])

        # OOB 체크
        if (ix < 0 or iy < 0 or iz < 0
                or ix >= temperature_grid.shape[1]
                or iy >= temperature_grid.shape[2]
                or iz >= temperature_grid.shape[3]):
            events.append(DetectorEvent(
                detector_id=det_id,
                position=pos,
                activation_frame=None,
                activation_time_s=None,
                trigger_reason="none",
            ))
            continue

        # 시간 시리즈
        T_series = temperature_grid[:, ix, iy, iz]
        V_series = visibility_grid[:, ix, iy, iz]

        # 트리거 결정
        frame, reason = check_single_cell_trigger(
            T_series, V_series,
            heat_threshold_c=heat_threshold_c,
            smoke_threshold_m=smoke_threshold_m,
        )

        events.append(DetectorEvent(
            detector_id=det_id,
            position=pos,
            activation_frame=frame,
            activation_time_s=float(frame * dt_seconds) if frame is not None else None,
            trigger_reason=reason,
        ))

    return events


def build_binary_sequence(
    events: list[DetectorEvent],
    n_timesteps: int = N_TIMESTEPS,
) -> np.ndarray:
    """감지기 이벤트 → 이진 시퀀스 (Tier 1 GNN 학습 입력용).

    각 감지기에 대해 (T,) 이진 시퀀스:
    - activation_frame 이전: 0
    - activation_frame 이후 (포함): 1 (latched)

    Args:
        events: extract_detector_events 결과.
        n_timesteps: 시간 frame 수 (기본 31).

    Returns:
        (n_timesteps, n_detectors) float32 이진 행렬.
    """
    n_det = len(events)
    binary = np.zeros((n_timesteps, n_det), dtype=np.float32)

    for det_idx, event in enumerate(events):
        if event.activation_frame is not None:
            binary[event.activation_frame:, det_idx] = 1.0

    return binary


def detector_stats(events: list[DetectorEvent]) -> dict:
    """이벤트 리스트의 요약 통계.

    Returns:
        {
            'n_total': int,
            'n_triggered': int,
            'n_heat_only': int,
            'n_smoke_only': int,
            'n_both': int,
            'n_never': int,
            'mean_activation_time_s': float | None,
            'earliest_activation_s': float | None,
            'latest_activation_s': float | None,
        }
    """
    n_total = len(events)
    triggered = [e for e in events if e.is_activated()]
    heat_only = [e for e in events if e.trigger_reason == "heat"]
    smoke_only = [e for e in events if e.trigger_reason == "smoke"]
    both = [e for e in events if e.trigger_reason == "both"]
    never = [e for e in events if e.trigger_reason == "none"]

    activation_times = [e.activation_time_s for e in triggered]

    return {
        "n_total": n_total,
        "n_triggered": len(triggered),
        "n_heat_only": len(heat_only),
        "n_smoke_only": len(smoke_only),
        "n_both": len(both),
        "n_never": len(never),
        "mean_activation_time_s": (
            float(np.mean(activation_times)) if activation_times else None
        ),
        "earliest_activation_s": (
            float(min(activation_times)) if activation_times else None
        ),
        "latest_activation_s": (
            float(max(activation_times)) if activation_times else None
        ),
    }


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=" * 70)
    print("detector_model.py self-test")
    print("=" * 70)

    errors: list[str] = []

    # ── Test 1: check_single_cell_trigger 기본 케이스 ─────────────────
    print("\n[Test 1] check_single_cell_trigger 기본 케이스")

    # 케이스 A: 안전 (트리거 안 됨)
    T_safe = np.full(31, 20.0)
    V_safe = np.full(31, 30.0)
    frame, reason = check_single_cell_trigger(T_safe, V_safe)
    print(f"  Safe (20°C, 30m): frame={frame}, reason={reason}")
    if frame is not None or reason != "none":
        errors.append(f"safe case failed: {frame}, {reason}")

    # 케이스 B: 열만 트리거 (5번 frame부터 80°C, 시정 30m 유지)
    T_heat = np.array([20.0]*5 + [80.0]*26)
    V_clear = np.full(31, 30.0)
    frame, reason = check_single_cell_trigger(T_heat, V_clear)
    print(f"  Heat only (80°C from frame 5): frame={frame}, reason={reason}")
    if frame != 5 or reason != "heat":
        errors.append(f"heat-only failed: {frame}, {reason}")

    # 케이스 C: 연기만 트리거 (3번 frame부터 시정 5m, 온도 20°C 유지)
    T_cool = np.full(31, 20.0)
    V_smoke = np.array([30.0]*3 + [5.0]*28)
    frame, reason = check_single_cell_trigger(T_cool, V_smoke)
    print(f"  Smoke only (Vis 5m from frame 3): frame={frame}, reason={reason}")
    if frame != 3 or reason != "smoke":
        errors.append(f"smoke-only failed: {frame}, {reason}")

    # 케이스 D: 둘 다 동시 (8번 frame부터)
    T_both = np.array([20.0]*8 + [70.0]*23)
    V_both = np.array([30.0]*8 + [5.0]*23)
    frame, reason = check_single_cell_trigger(T_both, V_both)
    print(f"  Both at frame 8: frame={frame}, reason={reason}")
    if frame != 8 or reason != "both":
        errors.append(f"both-simultaneous failed: {frame}, {reason}")

    # 케이스 E: 연기 먼저 (3), 열 나중 (10) → smoke
    T_late = np.array([20.0]*10 + [80.0]*21)
    V_early = np.array([30.0]*3 + [5.0]*28)
    frame, reason = check_single_cell_trigger(T_late, V_early)
    print(f"  Smoke first (3) then heat (10): frame={frame}, reason={reason}")
    if frame != 3 or reason != "smoke":
        errors.append(f"smoke-first failed: {frame}, {reason}")

    # ── Test 2: extract_detector_events 합성 환경 ────────────────────
    print("\n[Test 2] extract_detector_events 합성 환경")

    # 31 timesteps, (60, 40, 6) grid
    # 화재가 (15, 10, 1.5)에서 t=50s (frame 5)에 발생
    T_grid = np.full((31, 60, 40, 6), 20.0, dtype=np.float32)
    V_grid = np.full((31, 60, 40, 6), 30.0, dtype=np.float32)

    # 화재원 셀: (30, 20, 3) — world (15.25, 10.25, 1.75)
    # frame 5부터 800°C, frame 7부터 시정 2m
    T_grid[5:, 28:33, 18:23, 2:4] = 800.0
    V_grid[7:, 28:33, 18:23, 2:4] = 2.0

    detectors = [
        ("near_fire", (15.0, 10.0, 1.75)),   # 화재원 가까움
        ("mid_distance", (10.0, 5.0, 1.75)),  # 중간 거리
        ("far_corner", (2.0, 2.0, 1.75)),     # 멀리
        ("oob_outside", (-5.0, 10.0, 1.5)),   # OOB
    ]

    events = extract_detector_events(T_grid, V_grid, detectors)
    print(f"  발견된 이벤트 수: {len(events)}")

    for e in events:
        time_str = f"{e.activation_time_s:.0f}s" if e.is_activated() else "never"
        print(f"  {e.detector_id:15s}  pos={e.position}  "
              f"trigger={e.trigger_reason:6s}  t={time_str}")

    # 검증
    near = next(e for e in events if e.detector_id == "near_fire")
    if near.trigger_reason != "heat":
        errors.append(f"near_fire trigger reason: {near.trigger_reason}")
    if near.activation_frame != 5:
        errors.append(f"near_fire frame: {near.activation_frame}")

    mid = next(e for e in events if e.detector_id == "mid_distance")
    if mid.trigger_reason != "none":
        errors.append(f"mid_distance should be none: {mid.trigger_reason}")

    oob = next(e for e in events if e.detector_id == "oob_outside")
    if oob.trigger_reason != "none":
        errors.append(f"oob should be none: {oob.trigger_reason}")

    # ── Test 3: build_binary_sequence ────────────────────────────────
    print("\n[Test 3] build_binary_sequence")
    binary = build_binary_sequence(events)
    print(f"  shape: {binary.shape}")
    if binary.shape != (31, 4):
        errors.append(f"binary shape: {binary.shape}")

    # near_fire (frame 5에 활성) → frame 0~4는 0, frame 5~30은 1
    near_idx = next(i for i, (det_id, _) in enumerate(detectors)
                    if det_id == "near_fire")
    if binary[4, near_idx] != 0:
        errors.append("near_fire frame 4 should be 0")
    if binary[5, near_idx] != 1:
        errors.append("near_fire frame 5 should be 1")
    if binary[30, near_idx] != 1:
        errors.append("near_fire latched 30 should be 1")
    print(f"  near_fire 시퀀스: ...{binary[3:8, near_idx]}...")
    print(f"  (frame 0~4=0, frame 5+=1 latched 확인)")

    # ── Test 4: detector_stats ───────────────────────────────────────
    print("\n[Test 4] detector_stats")
    stats = detector_stats(events)
    print(f"  통계: {stats}")
    if stats["n_total"] != 4:
        errors.append(f"n_total: {stats['n_total']}")
    if stats["n_triggered"] != 1:
        errors.append(f"n_triggered: {stats['n_triggered']}")

    # ── Test 5: first_sim 실데이터 (있으면) ─────────────────────────
    print("\n[Test 5] first_sim 실데이터 검증")
    real_dir = Path("data/raw/first_sim")
    if real_dir.is_dir():
        from src.data_pipeline.fds_extractor import extract_slices

        slices = extract_slices(real_dir)
        T_real = slices["temperature"]
        V_real = slices["visibility"]
        print(f"  T range: {T_real.min():.1f}~{T_real.max():.1f}°C")
        print(f"  V range: {V_real.min():.2f}~{V_real.max():.2f}m")

        # 가상 감지기 18개 (건물 각 영역 대표 위치)
        # 이 좌표는 src/shared/building.py의 노드와 매핑됨 (있으면)
        real_detectors = [
            ("zone_a_west", (3.0, 13.0, 2.5)),
            ("zone_a_center", (8.0, 14.0, 2.5)),
            ("zone_b_west", (4.0, 4.0, 2.5)),
            ("zone_b_center", (10.0, 3.0, 2.5)),
            ("zone_b_east", (18.0, 3.0, 2.5)),
            ("zone_c_west", (15.0, 16.0, 2.5)),
            ("zone_c_center", (22.0, 16.0, 2.5)),
            ("zone_c_east", (28.0, 16.0, 2.5)),
            ("zone_d_west", (24.0, 5.0, 2.5)),
            ("zone_d_center", (28.0, 5.0, 2.5)),
            ("hall_n", (15.0, 12.0, 2.5)),
            ("hall_s", (15.0, 7.0, 2.5)),
            ("hall_e", (20.0, 9.0, 2.5)),
            ("hall_w", (10.0, 9.0, 2.5)),
            ("int_north", (12.0, 14.0, 2.5)),
            ("int_south", (15.0, 5.0, 2.5)),
            ("exit_west", (1.0, 5.0, 2.5)),
            ("exit_north", (8.0, 17.0, 2.5)),
            ("exit_east", (29.0, 13.0, 2.5)),
        ]

        events_real = extract_detector_events(T_real, V_real, real_detectors)
        print(f"\n  18개 가상 감지기 (z=2.5m, 천장 가까이) 트리거 결과:")
        print(f"  {'detector':18s}  {'trigger':8s}  {'time':>6s}")
        print(f"  {'-'*18}  {'-'*8}  {'-'*6}")
        for e in events_real:
            time_str = f"{e.activation_time_s:.0f}s" if e.is_activated() else "never"
            print(f"  {e.detector_id:18s}  {e.trigger_reason:8s}  {time_str:>6s}")

        stats_real = detector_stats(events_real)
        print(f"\n  통계: {stats_real}")
        print(f"  → 화재원 (18, 10) 가까운 감지기들이 빠르게 트리거되어야 함")
    else:
        print("  SKIP: data/raw/first_sim/ 디렉토리 없음")

    # ── Test 6: 입력 validation ──────────────────────────────────────
    print("\n[Test 6] 입력 validation")
    try:
        check_single_cell_trigger(np.zeros(10), np.zeros(20))
    except ValueError:
        print("  ✓ shape mismatch raises")
    else:
        errors.append("shape mismatch 검증 안 됨")

    try:
        check_single_cell_trigger(np.zeros((5, 5)), np.zeros((5, 5)))
    except ValueError:
        print("  ✓ 2-D input raises")
    else:
        errors.append("2-D input 검증 안 됨")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nPASS: detector_model 검증 완료")
```


====================================================================
작업 2: scripts/visualize_detectors.py 신규 — first_sim 감지기 시각화
====================================================================

## 목적
first_sim 데이터에서 가상 감지기 트리거 패턴 시각화.
발표 자료 + 화재 확산 + 감지기 위치 검증 모두 가능.

## 시그니처

```python
# scripts/visualize_detectors.py
"""Tier 1 가상 감지기 트리거 패턴 시각화.

3장 figure 생성:
1. 평면도에 18개 감지기 + 트리거 시간 표시 (z=2.5m)
2. 각 감지기의 시간별 trigger 시퀀스 (gantt chart 스타일)
3. 통계 요약 (활성화 / 종류별)

사용:
    python scripts/visualize_detectors.py data/raw/first_sim/
    python scripts/visualize_detectors.py data/raw/first_sim/ --output figures/first_sim/detectors/
"""

import argparse
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.data_pipeline.fds_extractor import extract_slices
from src.tier1.detector_model import (
    extract_detector_events,
    build_binary_sequence,
    detector_stats,
    HEAT_THRESHOLD_C,
    SMOKE_THRESHOLD_M,
)


# 가상 감지기 위치 18개 (천장 가까이 z=2.5m)
# 매뉴얼의 건물 그래프 노드와 매핑됨
DEFAULT_DETECTORS = [
    ("zone_a_west", (3.0, 13.0, 2.5)),
    ("zone_a_center", (8.0, 14.0, 2.5)),
    ("zone_b_west", (4.0, 4.0, 2.5)),
    ("zone_b_center", (10.0, 3.0, 2.5)),
    ("zone_b_east", (18.0, 3.0, 2.5)),
    ("zone_c_west", (15.0, 16.0, 2.5)),
    ("zone_c_center", (22.0, 16.0, 2.5)),
    ("zone_c_east", (28.0, 16.0, 2.5)),
    ("zone_d_west", (24.0, 5.0, 2.5)),
    ("zone_d_center", (28.0, 5.0, 2.5)),
    ("hall_n", (15.0, 12.0, 2.5)),
    ("hall_s", (15.0, 7.0, 2.5)),
    ("hall_e", (20.0, 9.0, 2.5)),
    ("hall_w", (10.0, 9.0, 2.5)),
    ("int_north", (12.0, 14.0, 2.5)),
    ("int_south", (15.0, 5.0, 2.5)),
    ("exit_west", (1.0, 5.0, 2.5)),
    ("exit_north", (8.0, 17.0, 2.5)),
    ("exit_east", (29.0, 13.0, 2.5)),
]


def plot_floorplan(events, output_path):
    """평면도에 감지기 + 트리거 시간 표시."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 도메인 박스
    ax.set_xlim(-1, 31)
    ax.set_ylim(-1, 20)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Tier 1 Detectors — Activation Timeline\n"
        f"(Heat > {HEAT_THRESHOLD_C}°C OR Vis < {SMOKE_THRESHOLD_M}m)"
    )

    # 도메인 경계
    domain_rect = Rectangle((0, 0), 30, 20, fill=False,
                            edgecolor="black", lw=1, linestyle="--")
    ax.add_patch(domain_rect)

    # 색상 매핑: 트리거 시간 → cmap
    cmap = plt.get_cmap("RdYlGn_r")

    for e in events:
        x, y, _ = e.position
        if e.is_activated():
            # 활성 시간에 따라 색
            t_norm = e.activation_time_s / 300.0
            color = cmap(t_norm)
            label = f"{e.detector_id}\n{e.activation_time_s:.0f}s ({e.trigger_reason})"
        else:
            color = "lightgrey"
            label = f"{e.detector_id}\nnever"

        # 감지기 위치 점
        circle = Circle((x, y), 0.7, color=color, ec="black", lw=0.5)
        ax.add_patch(circle)
        ax.annotate(
            label, (x, y), fontsize=6, ha="center",
            va="center", color="black",
        )

    # 컬러바
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=300))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Activation time (s)", shrink=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path}")


def plot_gantt(binary_seq, events, output_path):
    """각 감지기의 시간별 상태 gantt chart 스타일.

    가로축: 시간 (0~300s)
    세로축: 감지기 (18개)
    색: 비활성=흰색, 활성=색 (트리거 종류별)
    """
    n_t, n_det = binary_seq.shape
    times = np.arange(n_t) * 10.0

    # 트리거 종류별 색
    reason_colors = {
        "heat": "#E24B4A",      # 빨강
        "smoke": "#5A85D8",     # 파랑
        "both": "#9B59B6",      # 보라
        "none": "#F5F4ED",      # 회색 (활성 안 됨)
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    for det_idx, event in enumerate(events):
        color = reason_colors.get(event.trigger_reason, "#888780")
        for t_idx in range(n_t):
            if binary_seq[t_idx, det_idx] > 0:
                ax.barh(
                    det_idx, 10.0, left=times[t_idx],
                    color=color, edgecolor="none",
                )

    ax.set_yticks(range(n_det))
    ax.set_yticklabels([e.detector_id for e in events], fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title("Detector Activation Timeline")
    ax.set_xlim(0, 310)
    ax.invert_yaxis()

    # 범례
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=reason_colors["heat"], label=f"Heat (T > {HEAT_THRESHOLD_C}°C)"),
        Patch(color=reason_colors["smoke"], label=f"Smoke (Vis < {SMOKE_THRESHOLD_M}m)"),
        Patch(color=reason_colors["both"], label="Both simultaneously"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path}")


def plot_summary_bar(stats, output_path):
    """통계 요약 막대 그래프."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Heat only", "Smoke only", "Both", "Never"]
    counts = [
        stats["n_heat_only"],
        stats["n_smoke_only"],
        stats["n_both"],
        stats["n_never"],
    ]
    colors = ["#E24B4A", "#5A85D8", "#9B59B6", "#B4B2A9"]

    bars = ax.bar(categories, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.annotate(
            str(count), (bar.get_x() + bar.get_width()/2, bar.get_height()),
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylabel("Number of detectors")
    ax.set_title(
        f"Detector Trigger Summary\n"
        f"({stats['n_triggered']}/{stats['n_total']} activated)"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fds_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--heat-threshold", type=float, default=HEAT_THRESHOLD_C,
        help=f"열 임계값 (°C), 기본 {HEAT_THRESHOLD_C}",
    )
    parser.add_argument(
        "--smoke-threshold", type=float, default=SMOKE_THRESHOLD_M,
        help=f"연기 임계값 (m), 기본 {SMOKE_THRESHOLD_M}",
    )
    args = parser.parse_args()

    if not args.fds_dir.exists():
        print(f"ERROR: {args.fds_dir} 존재하지 않음")
        return 1

    scenario_id = args.fds_dir.name
    output_dir = args.output or (Path("figures") / scenario_id / "detectors")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTier 1 detector visualization: {scenario_id}")
    print(f"  Heat threshold: {args.heat_threshold}°C")
    print(f"  Smoke threshold: {args.smoke_threshold}m")

    # 1. FDS 데이터 추출
    print("\n[1] FDS data extraction")
    slices = extract_slices(args.fds_dir)
    T_grid = slices["temperature"]
    V_grid = slices["visibility"]
    print(f"  T shape: {T_grid.shape}, range [{T_grid.min():.1f}, {T_grid.max():.1f}]°C")
    print(f"  V shape: {V_grid.shape}, range [{V_grid.min():.2f}, {V_grid.max():.2f}]m")

    # 2. 감지기 이벤트 추출
    print("\n[2] Extract detector events")
    events = extract_detector_events(
        T_grid, V_grid, DEFAULT_DETECTORS,
        heat_threshold_c=args.heat_threshold,
        smoke_threshold_m=args.smoke_threshold,
    )
    print(f"  Total detectors: {len(events)}")

    # 3. 이진 시퀀스 빌드
    print("\n[3] Build binary sequence")
    binary_seq = build_binary_sequence(events)
    print(f"  Shape: {binary_seq.shape}")
    print(f"  Final state (t=300s): {binary_seq[-1].sum():.0f} active out of {len(events)}")

    # 4. 통계
    stats = detector_stats(events)
    print(f"\n[4] Statistics:")
    for k, v in stats.items():
        if v is not None:
            print(f"  {k}: {v}")

    # 5. 시각화
    print("\n[5] Visualization")
    plot_floorplan(events, output_dir / "detectors_floorplan.png")
    plot_gantt(binary_seq, events, output_dir / "detectors_gantt.png")
    plot_summary_bar(stats, output_dir / "detectors_summary.png")

    # 6. 데이터 저장 (Tier 1 학습용)
    print("\n[6] Save data")
    np.save(output_dir / "binary_sequence.npy", binary_seq)
    with open(output_dir / "events.json", "w", encoding="utf-8") as f:
        events_dict = [
            {
                "detector_id": e.detector_id,
                "position": e.position,
                "activation_frame": e.activation_frame,
                "activation_time_s": e.activation_time_s,
                "trigger_reason": e.trigger_reason,
            }
            for e in events
        ]
        json.dump({
            "scenario_id": scenario_id,
            "heat_threshold_c": args.heat_threshold,
            "smoke_threshold_m": args.smoke_threshold,
            "events": events_dict,
            "stats": stats,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n생성 파일:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:30s}  {size_kb:>6.1f} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```


====================================================================
작업 3: docs/decisions.md 에 D-023 추가
====================================================================

기존 decisions.md 파일에 D-022 다음에 추가:

```markdown
## D-023: Tier 1 GNN 감지기 트리거 모델 정의

**Decision**:
Tier 1 GNN의 가상 화재 감지기 트리거 조건을 다음과 같이 정의:

- **열 감지기**: 온도 > **60°C**
- **연기 감지기**: visibility < **10 m**
- **OR 조합**: 둘 중 하나라도 만족 시 작동 (latched, 이후 비활성화 안 됨)
- **CO 감지기**: **제외** (300초 시뮬레이션에서 UL 2034 임계값 도달 안 함)

**Rationale**:

표준 근거:
- NFPA 72 일반형 열 감지기: 57°C (135°F)
- 한국 KOFEIS 0301 일반형: 70°C
- 60°C = 두 표준의 중간값 (보수적이면서 합리적)
- UL 268 7th: 1.5%/ft obscuration ≈ 13m visibility
- 10m는 보수적 선택 (확실히 위험한 영역만 트리거)

CO 제외 이유:
- UL 2034: 70 ppm × 60~240분, 150 ppm × 10~50분, 400 ppm × 4~15분
- 우리 first_sim에서 화재원 근처 최대 250 ppm 도달
- 그러나 300초 (5분)는 UL 2034의 최단 요구 시간 (4분)에 가까움
- 시뮬레이션 시간이 너무 짧아 신뢰성 있는 학습 신호 어려움
- 학습 데이터 다양성을 위해 열 + 연기로 충분

OR 조합 이유:
- 실제 건물은 보통 열 + 연기 복합 감지기 설치
- 화재 초기 (연기 우선) + 화재 정착 (열 우선) 모두 커버
- 단일 임계값보다 더 현실적인 학습 신호

Latched 동작:
- 한번 트리거된 감지기는 시뮬레이션 끝까지 활성 유지
- 실제 감지기와 일치 (수동 reset 필요)

**Discovered**:
2025년 first_sim 위험지도 분석 시 발견. 화재 감지기 표준 (NFPA 72, UL 268,
UL 2034) 조사 후 우리 300초 시뮬레이션에 적합한 감지기 모델 정의.

**Implementation**:
- 신규 모듈: src/tier1/detector_model.py
- 핵심 함수: check_single_cell_trigger, extract_detector_events
- 데이터클래스: DetectorEvent
- 시각화: scripts/visualize_detectors.py

**기존 결정과 관계**:
- D-019 (마스크 생성): 영향 없음
- TENABILITY 임계값 (위험도 계산용): 다름. TENABILITY는 사람 안전 임계값,
  D-023은 감지기 작동 임계값. 두 임계값 체계는 독립.

**Status**: Implemented. Tier 1 GNN 학습 데이터 생성에 사용.
```


====================================================================
작업 4: docs/tier1_gnn_design.md 업데이트
====================================================================

기존 tier1_gnn_design.md의 "## 3. 학습 데이터 구성" 섹션의 다음 부분을 찾기:

```python
def extract_detector_events_from_slices(
    grid: np.ndarray,
    coords: dict,
    detector_positions: list,
    threshold_celsius: float = 60.0,
) -> dict:
    ...
```

이 부분을 다음으로 교체:

```markdown
**감지기 트리거 모델 (D-023)**:

매뉴얼 D-023 결정에 따라 다음 모델 사용:
- 열 감지기: 온도 > 60°C
- 연기 감지기: visibility < 10m
- OR 조합 (둘 중 하나라도 만족 시 작동, latched)
- CO 감지기 제외 (시뮬레이션 시간 부족)

구현: `src/tier1/detector_model.py`의 `extract_detector_events()`:

```python
from src.tier1.detector_model import extract_detector_events

# FDS 슬라이스 데이터 + 감지기 좌표 → 이벤트 리스트
events = extract_detector_events(
    temperature_grid,    # (31, 60, 40, 6)
    visibility_grid,     # (31, 60, 40, 6)
    detector_positions,  # [(id, (x, y, z)), ...]
)

# 각 이벤트:
#   detector_id, position
#   activation_frame (None | 0~30)
#   activation_time_s (None | 0~300)
#   trigger_reason ("heat" | "smoke" | "both" | "none")
```
```

또한 같은 문서의 어딘가에 다음 섹션을 추가 (적절한 위치 판단):

```markdown
## 감지기 모델 검증 (first_sim 결과)

D-023 모델을 first_sim 시뮬레이션 데이터에 적용한 결과:

18개 가상 감지기 (천장 z=2.5m) 트리거 패턴:
- 화재원 (18, 10) 가까운 감지기: 30~60초에 활성
- 중간 거리 감지기: 80~150초에 활성
- 먼 거리 감지기: 200~300초 또는 활성 안 됨

시각화: `figures/first_sim/detectors/` (3장 figures):
- detectors_floorplan.png: 평면도에 시간 표시
- detectors_gantt.png: 시간별 활성화 시퀀스
- detectors_summary.png: 통계 막대 그래프

이 패턴이 Tier 1 GNN의 학습 신호. 30건 시뮬레이션 완료 후 모든 시나리오에
동일 모델 적용.
```


====================================================================
검증 및 보고
====================================================================

작업 후 다음 실행:

1. 단위 테스트:
   python -m src.tier1.detector_model
   → 6개 Test 모두 PASS 확인

2. first_sim 데이터로 시각화:
   python scripts/visualize_detectors.py data/raw/first_sim/
   → 3장 figure + binary_sequence.npy + events.json 생성

3. 매뉴얼 업데이트 확인:
   grep "D-023" docs/decisions.md
   grep "60°C\|HEAT_THRESHOLD" docs/tier1_gnn_design.md
   → 둘 다 결과 있어야 함

보고할 사항:
- 생성된 파일들 (모듈, 스크립트, 시각화) 경로 + 줄 수
- 6개 Test의 PASS/FAIL
- first_sim 18개 감지기 트리거 통계:
   * n_triggered (활성화된 수)
   * mean_activation_time_s (평균 활성 시간)
   * 트리거 종류 분포 (heat/smoke/both/never)
- detectors_floorplan.png 또는 detectors_gantt.png 직접 확인
- 발견한 이상점 (있으면)

## 금지 사항

- TENABILITY 임계값 (위험도용)과 D-023 임계값 (감지기용) 혼동 금지
  TENABILITY.T_DANGER_C (60°C) — 사람 안전 임계값
  HEAT_THRESHOLD_C (60°C) — 감지기 작동 임계값
  같은 60°C 값이지만 의미와 사용처가 완전히 다름
- 임계값을 임의 변경 금지 (D-023 정확히 사용: 60°C, 10m)
- CO 감지기 모델 추가 금지 (D-023 명시: 제외)
- Latched 동작 변경 금지 (한번 활성 → 끝까지 활성 유지)
- 새 의존성 추가 금지 — numpy, matplotlib만 사용

## 향후 작업 미리보기

이 모듈 완성 후 다음이 가능:

1. **30건 시뮬레이션 완료 시**:
   - 모든 시나리오에 동일 감지기 모델 적용
   - 30개 binary_sequence.npy 생성
   - Tier 1 GNN 학습 데이터 확보

2. **Tier 1 GNN 학습 데이터셋 빌드** (`src/tier1/dataset.py`, 미작성):
   - 각 시나리오의 이진 시퀀스 + 노드 그래프 → PyTorch Geometric Dataset
   - 학습 24건 / val 3건 / OOD 3건 분할

3. **A3T-GCN 모델 학습** (`src/tier1/tier1_model.py`, 미작성):
   - 매뉴얼 사양: in_channels=6, out_channels=32, periods=6
   - 이진 시퀀스 입력 → 노드별 미래 위험도 출력
```

---

## 사용 후 점검

Claude Code 작업 완료 후 다음을 확인하세요.

- [ ] `src/tier1/__init__.py` + `src/tier1/detector_model.py` 생성
- [ ] `python -m src.tier1.detector_model` 6개 Test PASS
- [ ] `scripts/visualize_detectors.py` 생성
- [ ] `figures/first_sim/detectors/` 3장 PNG + binary_sequence.npy + events.json
- [ ] `docs/decisions.md`에 D-023 추가
- [ ] `docs/tier1_gnn_design.md` 업데이트 (감지기 모델 + 검증 결과)

이상 통과하면 git commit:

```bash
git add src/tier1/ scripts/visualize_detectors.py docs/decisions.md docs/tier1_gnn_design.md
git commit -m "Tier 1 detector model: heat OR smoke (D-023)

- Heat threshold 60°C (NFPA 57 / KOFEIS 70 mid)
- Smoke threshold 10m visibility (UL 268 conservative)
- OR combination, latched
- CO excluded (UL 2034 not reachable in 300s sim)
- DetectorEvent dataclass + extract_detector_events()
- binary_sequence.npy for Tier 1 GNN training input
- 3 visualization figures (floorplan, gantt, summary)"
```

## 다음 단계 미리보기

이 작업 완료 후:

### 단기 (30건 시뮬레이션 완료 시):
모든 시나리오에 동일 모델 적용 → 30개 binary_sequence.npy 생성

### 중기 (Week 11):
`src/tier1/dataset.py` — PyTorch Geometric Dataset
`src/tier1/tier1_model.py` — A3T-GCN 모델

### 장기 (Week 12):
EXP-TIER1-001 (Tier 1 성능 단독 검증)
EXP-TIER1-002 (Tier 1 vs Tier 2 비교)
EXP-TIER1-003 (감지기 고장 강건성)

---

## 발표에서의 가치

D-023 모델의 발표 슬라이드 메시지:

**"기존 화재 감지기로 미래 30초 위험을 예측합니다."**

| 항목 | 우리 시스템 (Tier 1) |
|---|---|
| 입력 | 이진 감지기 신호 (열 60°C OR 연기 10m) |
| 추가 하드웨어 | **없음 (기존 감지기 활용)** |
| 출력 | 16~20개 영역별 위험도 |
| 지연 시간 | < 1초 |
| 표준 준수 | NFPA 72 + UL 268 |

이 메시지는 화재 안전 비전공자도 즉시 이해 가능하며, "실제 배포 가능성"
입증의 핵심.
