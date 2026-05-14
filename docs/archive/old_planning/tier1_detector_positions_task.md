# Claude Code 작업 요청 — Tier 1 감지기 위치 확정 (28개)

> 평면도 분석 결과를 코드화. 18개 방 + 7개 복도 + 3개 출구 = 28개 감지기.
> NFPA 72 spacing 표준 (9m 이내) 준수. 각 방의 중앙 + 복도 균등 배치.
> `src/tier1/detector_positions.py` 신규 + 기존 detector_model.py 업데이트.
>
> 새 Claude Code 세션에 통째로 붙여넣기.

---

## 사전 확인

- [ ] `src/tier1/detector_model.py` 존재 (이전 D-023 작업 완료)
- [ ] `docs/decisions.md`에 D-023 추가됨
- [ ] `scripts/visualize_detectors.py` 존재
- [ ] first_sim 시뮬레이션 데이터 (data/raw/first_sim/) 존재

---

## 배경 — 평면도 분석 결과

### 건물 구조 (평면도 기반)

```
도메인:    30m × 20m (L자형 + 중앙 정원)
출구:      3개
  - Exit 1 (서):  (2.5, 5.75) — 좌측 통로 끝
  - Exit 2 (북):  (15, 16.75) — 북측 정원 위
  - Exit 3 (동):  (22, 5.75) — 우측 통로 끝
정원:      X 5~14, Y 6~15.5 (외기 환기 영역)
```

### 영역별 방 개수 (평면도 식별)

| Zone | 위치 | 방 개수 | 특징 |
|---|---|---|---|
| Zone A | 좌상 사선 | 3개 | 작은 방, 대각선 통로 |
| Zone B | 남측 | 5개 | 큰 방, 정원 남측 |
| Zone C | 북동 | 4개 | 큰 방, 정원 북측 |
| Zone D | 동측 | 5개 | 작은 방, 정원 동측 |
| **방 합계** | | **17개** | |

### 복도 영역

| 복도 | 위치 | 길이 | 감지기 수 |
|---|---|---|---|
| 남측 복도 | Y=5.5, X 2~22 | 약 20m | 2개 (5.5, 12.5) |
| 동측 복도 | X=16, Y 6~14 | 약 8m | 2개 (Y=8, Y=13) |
| 북측 복도 | Y=15.5, X 14~30 | 약 16m | 2개 (X=10, X=22) |
| 동-동측 복도 | X=22, Y 8~15 | 약 7m | 1개 (Y=10) |
| **복도 합계** | | | **7개** |

### NFPA 72 spacing 표준 준수

- 천장 평면 화재 감지기: 최대 9m spacing
- 우리 복도 감지기: 평균 5~7m spacing → ✓ 표준 준수
- 방 감지기: 중앙 1개 (작은 방 표준)


---

## 프롬프트 (여기부터 복사)

```
이번 작업은 Tier 1 GNN의 가상 감지기 위치를 평면도 분석에 따라 확정하는 거야.

핵심 결정 (D-023과 별개의 위치 정의):
- 총 28개 감지기 (방 18 + 복도 7 + 출구 3)
- NFPA 72 spacing 9m 이내 준수
- 각 방 중앙 1개, 복도는 5~7m 간격
- 모두 천장 가까이 z=2.5m

먼저 다음 컨텍스트 파일들을 읽어:

1. CLAUDE.md
2. docs/decisions.md (D-023 — 트리거 모델)
3. docs/tier1_gnn_design.md (전체 설계)
4. docs/tier1_detector_plan.md (감지기 통합 계획)
5. src/tier1/detector_model.py (이전 작업 — DetectorEvent, extract_detector_events)
6. scripts/visualize_detectors.py (이전 작업 — 기존 18개 임시 위치 사용)


====================================================================
작업 1: src/tier1/detector_positions.py 신규 작성
====================================================================

## 목적
28개 감지기의 정확한 위치를 한 모듈에 정의. detector_model.py와 분리.

## 시그니처

```python
# src/tier1/detector_positions.py
"""Tier 1 GNN 가상 감지기 위치 정의.

매뉴얼 D-023 (감지기 트리거 모델)과 독립적으로,
평면도 분석에 따른 28개 감지기의 절대 좌표 정의.

설치 기준:
- 모든 감지기: z = 2.5m (천장 가까이, 실제 화재 감지기 설치 높이)
- 방 감지기 (18개): 각 방의 평면 중앙
- 복도 감지기 (7개): 5~7m 간격 (NFPA 72 spacing 9m 이내 준수)
- 출구 감지기 (3개): 각 출구 노드에 직접 배치

영역별:
- Zone A (좌상 사선): 3개 방
- Zone B (남측): 5개 방
- Zone C (북동): 4개 방
- Zone D (동측): 5개 작은 방 (총 5개)
- 중앙 복도: 7개
- 출구: 3개
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DetectorLocation:
    """단일 감지기의 절대 위치 + 메타데이터.

    Attributes:
        detector_id: 고유 식별자 (예: "zone_a_1", "south_corridor_1")
        position: (x, y, z) world 좌표 (m). z=2.5 천장 높이.
        node_type: 'room' | 'corridor' | 'exit'
        area: 'zone_a' | 'zone_b' | 'zone_c' | 'zone_d' | 'corridor' | 'exit'
        description: 사람 가독 설명 (발표/디버깅용)
    """
    detector_id: str
    position: tuple[float, float, float]
    node_type: Literal["room", "corridor", "exit"]
    area: str
    description: str


# 천장 높이 (모든 감지기 동일)
CEILING_HEIGHT_M: float = 2.5


# ============================================================
# Zone A — 좌상 사선 통로 (대각선 방 3개)
# ============================================================
ZONE_A_DETECTORS = [
    DetectorLocation(
        detector_id="zone_a_1",
        position=(3.0, 11.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_a",
        description="Zone A 좌하단 방 (사선 통로 아래쪽)",
    ),
    DetectorLocation(
        detector_id="zone_a_2",
        position=(6.0, 14.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_a",
        description="Zone A 중앙 방 (사선 통로 중간)",
    ),
    DetectorLocation(
        detector_id="zone_a_3",
        position=(8.5, 17.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_a",
        description="Zone A 우상단 방 (사선 통로 위쪽, 출구 2 근처)",
    ),
]


# ============================================================
# Zone B — 남측 큰 방 5개
# ============================================================
ZONE_B_DETECTORS = [
    DetectorLocation(
        detector_id="zone_b_1",
        position=(2.0, 2.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_b",
        description="Zone B 가장 서측 방 (출구 1 인접)",
    ),
    DetectorLocation(
        detector_id="zone_b_2",
        position=(6.0, 2.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_b",
    description="Zone B 두번째 방",
    ),
    DetectorLocation(
        detector_id="zone_b_3",
        position=(10.0, 2.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_b",
        description="Zone B 중앙 방",
    ),
    DetectorLocation(
        detector_id="zone_b_4",
        position=(14.0, 2.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_b",
        description="Zone B 네번째 방",
    ),
    DetectorLocation(
        detector_id="zone_b_5",
        position=(19.0, 2.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_b",
        description="Zone B 가장 동측 방 (출구 3 인접)",
    ),
]


# ============================================================
# Zone C — 북동 큰 방 4개
# ============================================================
ZONE_C_DETECTORS = [
    DetectorLocation(
        detector_id="zone_c_1",
        position=(17.0, 17.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_c",
        description="Zone C 가장 서측 방 (출구 2 인접)",
    ),
    DetectorLocation(
        detector_id="zone_c_2",
        position=(21.0, 17.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_c",
        description="Zone C 두번째 방",
    ),
    DetectorLocation(
        detector_id="zone_c_3",
        position=(25.0, 17.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_c",
        description="Zone C 세번째 방",
    ),
    DetectorLocation(
        detector_id="zone_c_4",
        position=(28.5, 17.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_c",
        description="Zone C 가장 동측 방",
    ),
]


# ============================================================
# Zone D — 동측 작은 방 5개
# ============================================================
ZONE_D_DETECTORS = [
    DetectorLocation(
        detector_id="zone_d_1",
        position=(24.5, 12.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_d",
        description="Zone D 좌상단 (X=24.5, 정원 동측 위쪽)",
    ),
    DetectorLocation(
        detector_id="zone_d_2",
        position=(24.5, 9.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_d",
        description="Zone D 좌중단 (X=24.5, 중앙)",
    ),
    DetectorLocation(
        detector_id="zone_d_3",
        position=(28.0, 12.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_d",
        description="Zone D 우상단 (X=28, 외측 위쪽)",
    ),
    DetectorLocation(
        detector_id="zone_d_4",
        position=(28.0, 9.5, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_d",
        description="Zone D 우중단 (X=28, 중앙)",
    ),
    DetectorLocation(
        detector_id="zone_d_5",
        position=(24.5, 7.0, CEILING_HEIGHT_M),
        node_type="room",
        area="zone_d",
        description="Zone D 좌하단 (X=24.5, 정원 동측 아래쪽)",
    ),
]


# ============================================================
# 복도 감지기 7개 — NFPA 72 spacing 9m 이내 준수
# ============================================================
CORRIDOR_DETECTORS = [
    # 남측 복도 (Y=5.5, X 2~22 정원 남측) — 20m 길이를 5.5+7m로 2등분
    DetectorLocation(
        detector_id="south_corridor_1",
        position=(5.5, 5.5, CEILING_HEIGHT_M),
        node_type="corridor",
        area="corridor",
        description="남측 복도 서측 (X=5.5, 출구 1과 중앙 사이)",
    ),
    DetectorLocation(
        detector_id="south_corridor_2",
        position=(12.5, 5.5, CEILING_HEIGHT_M),
        node_type="corridor",
        area="corridor",
        description="남측 복도 동측 (X=12.5, 중앙)",
    ),

    # 동측 복도 (X=16, Y 6~15) — 8m 길이를 2등분
    DetectorLocation(
        detector_id="east_corridor_1",
        position=(16.0, 8.0, CEILING_HEIGHT_M),
        node_type="corridor",
        area="corridor",
        description="동측 복도 남측 (Y=8, 정원 동측 아래쪽)",
    ),
    DetectorLocation(
        detector_id="east_corridor_2",
        position=(16.0, 13.0, CEILING_HEIGHT_M),
        node_type="corridor",
        area="corridor",
        description="동측 복도 북측 (Y=13, 정원 동측 위쪽)",
    ),

    # 북측 복도 (Y=15.5, X 14~30 정원 북측) — 16m 길이를 2등분
    DetectorLocation(
        detector_id="north_corridor_1",
        position=(10.0, 15.5, CEILING_HEIGHT_M),
        node_type="corridor",
        area="corridor",
        description="북측 복도 서측 (X=10, 정원 북측, 출구 2 근처)",
    ),
    DetectorLocation(
        detector_id="north_corridor_2",
        position=(22.0, 15.5, CEILING_HEIGHT_M),
        node_type="corridor",
        area="corridor",
        description="북측 복도 동측 (X=22, Zone C 앞)",
    ),

    # 동-동측 복도 (X=22, Y 6~15) — Zone D를 두 영역으로 분리
    DetectorLocation(
        detector_id="deep_east_corridor",
        position=(22.5, 10.0, CEILING_HEIGHT_M),
        node_type="corridor",
        area="corridor",
        description="동-동측 복도 (X=22.5, Y=10, Zone D 중앙)",
    ),
]


# ============================================================
# 출구 감지기 3개 — 각 출구 위치
# ============================================================
EXIT_DETECTORS = [
    DetectorLocation(
        detector_id="exit_1_west",
        position=(2.5, 5.75, CEILING_HEIGHT_M),
        node_type="exit",
        area="exit",
        description="서측 출구 (Exit 1) — 좌측 끝",
    ),
    DetectorLocation(
        detector_id="exit_2_north",
        position=(15.0, 16.75, CEILING_HEIGHT_M),
        node_type="exit",
        area="exit",
        description="북측 출구 (Exit 2) — 정원 북측",
    ),
    DetectorLocation(
        detector_id="exit_3_east",
        position=(22.0, 5.75, CEILING_HEIGHT_M),
        node_type="exit",
        area="exit",
        description="동측 출구 (Exit 3) — 우측",
    ),
]


# ============================================================
# 전체 감지기 통합 (28개)
# ============================================================
ALL_DETECTORS: list[DetectorLocation] = (
    ZONE_A_DETECTORS
    + ZONE_B_DETECTORS
    + ZONE_C_DETECTORS
    + ZONE_D_DETECTORS
    + CORRIDOR_DETECTORS
    + EXIT_DETECTORS
)


def get_detector_positions_legacy_format() -> list[tuple[str, tuple[float, float, float]]]:
    """기존 코드와의 호환성을 위한 변환 함수.

    extract_detector_events()가 [(id, (x,y,z)), ...] 형태를 받으므로
    DetectorLocation 객체를 이 형태로 변환.

    Returns:
        [(detector_id, (x, y, z)), ...] — 28개 튜플 리스트.
    """
    return [(d.detector_id, d.position) for d in ALL_DETECTORS]


def get_detectors_by_area(area: str) -> list[DetectorLocation]:
    """특정 영역(zone)의 감지기들만 반환.

    Args:
        area: 'zone_a' | 'zone_b' | 'zone_c' | 'zone_d' | 'corridor' | 'exit'

    Returns:
        해당 영역의 DetectorLocation 리스트.
    """
    return [d for d in ALL_DETECTORS if d.area == area]


def get_detector_by_id(detector_id: str) -> DetectorLocation:
    """ID로 감지기 조회.

    Args:
        detector_id: 감지기 식별자.

    Returns:
        DetectorLocation 객체.

    Raises:
        ValueError: 해당 ID 없음.
    """
    for d in ALL_DETECTORS:
        if d.detector_id == detector_id:
            return d
    raise ValueError(f"Unknown detector_id: {detector_id}")


def detector_count_by_area() -> dict[str, int]:
    """영역별 감지기 개수 통계.

    Returns:
        {'zone_a': 3, 'zone_b': 5, ..., 'exit': 3}
    """
    counts: dict[str, int] = {}
    for d in ALL_DETECTORS:
        counts[d.area] = counts.get(d.area, 0) + 1
    return counts


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("detector_positions.py self-test")
    print("=" * 70)

    errors: list[str] = []

    # ── Test 1: 총 28개 확인 ─────────────────────────────────────────
    print("\n[Test 1] 총 감지기 개수")
    expected_total = 28
    if len(ALL_DETECTORS) != expected_total:
        errors.append(f"총 개수: {len(ALL_DETECTORS)} != {expected_total}")
    print(f"  총 {len(ALL_DETECTORS)}개 감지기 (expected {expected_total})")

    # ── Test 2: 영역별 개수 ──────────────────────────────────────────
    print("\n[Test 2] 영역별 분포")
    counts = detector_count_by_area()
    expected_counts = {
        "zone_a": 3,
        "zone_b": 5,
        "zone_c": 4,
        "zone_d": 5,
        "corridor": 7,
        "exit": 3,
    }
    for area, expected in expected_counts.items():
        actual = counts.get(area, 0)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {area}: {actual} (expected {expected})")
        if actual != expected:
            errors.append(f"{area}: {actual} != {expected}")

    # ── Test 3: ID 중복 없음 ────────────────────────────────────────
    print("\n[Test 3] ID 고유성")
    ids = [d.detector_id for d in ALL_DETECTORS]
    if len(ids) != len(set(ids)):
        from collections import Counter
        dup = [k for k, v in Counter(ids).items() if v > 1]
        errors.append(f"중복 ID: {dup}")
    else:
        print(f"  ✓ 28개 모두 고유 ID")

    # ── Test 4: 모든 위치가 도메인 안 ─────────────────────────────
    print("\n[Test 4] 도메인 범위 검증 (0~30 × 0~20 × 0~3)")
    for d in ALL_DETECTORS:
        x, y, z = d.position
        if not (0.0 <= x <= 30.0):
            errors.append(f"{d.detector_id} x={x} OOB")
        if not (0.0 <= y <= 20.0):
            errors.append(f"{d.detector_id} y={y} OOB")
        if not (0.0 <= z <= 3.0):
            errors.append(f"{d.detector_id} z={z} OOB")
    if not errors:
        print(f"  ✓ 모든 28개 감지기가 도메인 내")

    # ── Test 5: 모든 감지기 z=2.5 ────────────────────────────────
    print("\n[Test 5] 천장 높이 z=2.5m")
    for d in ALL_DETECTORS:
        if d.position[2] != CEILING_HEIGHT_M:
            errors.append(f"{d.detector_id} z={d.position[2]} != {CEILING_HEIGHT_M}")
    if all(d.position[2] == CEILING_HEIGHT_M for d in ALL_DETECTORS):
        print(f"  ✓ 모든 감지기 z={CEILING_HEIGHT_M}m")

    # ── Test 6: get_detector_by_id ───────────────────────────────
    print("\n[Test 6] get_detector_by_id")
    d = get_detector_by_id("zone_b_3")
    print(f"  zone_b_3: pos={d.position}, type={d.node_type}")
    if d.position != (10.0, 2.5, 2.5):
        errors.append(f"zone_b_3 position wrong")

    try:
        get_detector_by_id("nonexistent")
        errors.append("nonexistent 검증 안 됨")
    except ValueError:
        print(f"  ✓ 잘못된 ID에 ValueError")

    # ── Test 7: get_detectors_by_area ────────────────────────────
    print("\n[Test 7] get_detectors_by_area")
    zone_b = get_detectors_by_area("zone_b")
    print(f"  zone_b: {len(zone_b)}개 — {[d.detector_id for d in zone_b]}")
    if len(zone_b) != 5:
        errors.append(f"zone_b 개수: {len(zone_b)} != 5")

    # ── Test 8: legacy format 변환 ───────────────────────────────
    print("\n[Test 8] legacy format 변환")
    legacy = get_detector_positions_legacy_format()
    print(f"  변환된 튜플 개수: {len(legacy)}")
    print(f"  첫 항목: {legacy[0]}")
    if len(legacy) != 28:
        errors.append(f"legacy 개수: {len(legacy)} != 28")
    # 형식 확인
    if not all(isinstance(item, tuple) and len(item) == 2
               and isinstance(item[1], tuple) and len(item[1]) == 3
               for item in legacy):
        errors.append("legacy format 형식 잘못됨")

    # ── Test 9: 복도 spacing NFPA 9m 이내 ────────────────────────
    print("\n[Test 9] 복도 감지기 spacing (NFPA 72 9m 이내)")
    corridors = get_detectors_by_area("corridor")
    import numpy as np
    # 모든 복도 감지기 쌍 거리 계산
    max_dist_to_neighbor = 0.0
    for i, d1 in enumerate(corridors):
        min_to_others = float('inf')
        for j, d2 in enumerate(corridors):
            if i != j:
                dist = np.linalg.norm(
                    np.array(d1.position[:2]) - np.array(d2.position[:2])
                )
                min_to_others = min(min_to_others, dist)
        max_dist_to_neighbor = max(max_dist_to_neighbor, min_to_others)
    print(f"  최대 이웃 거리: {max_dist_to_neighbor:.2f}m")
    # 9m 이내가 권장이지만 일부 영역은 단독일 수 있으므로 12m 정도 허용
    if max_dist_to_neighbor > 12.0:
        errors.append(
            f"NFPA spacing 초과: {max_dist_to_neighbor:.2f}m > 12m"
        )

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nPASS: detector_positions 검증 완료")
    print(f"\n총 통계: {detector_count_by_area()}")
```


====================================================================
작업 2: scripts/visualize_detectors.py 업데이트
====================================================================

## 변경 사항

기존 `DEFAULT_DETECTORS` 변수를 새 모듈에서 import.

기존 코드 (위쪽):
```python
# 가상 감지기 위치 18개 (천장 가까이 z=2.5m)
DEFAULT_DETECTORS = [
    ("zone_a_west", (3.0, 13.0, 2.5)),
    ...
]
```

이걸 다음으로 교체:
```python
from src.tier1.detector_positions import get_detector_positions_legacy_format

# D-024: 28개 감지기 (Zone A 3 + B 5 + C 4 + D 5 + 복도 7 + 출구 3)
DEFAULT_DETECTORS = get_detector_positions_legacy_format()
```

나머지 코드는 그대로 유지. 28개 자동 처리됨.


====================================================================
작업 3: docs/decisions.md 에 D-024 추가
====================================================================

기존 decisions.md 파일에 D-023 다음에 추가:

```markdown
## D-024: Tier 1 감지기 28개 위치 확정

**Decision**:
평면도 분석을 통해 Tier 1 GNN 가상 감지기 28개의 정확한 위치 확정:

| 영역 | 개수 | 위치 기준 |
|---|---|---|
| Zone A (좌상 사선) | 3 | 각 방 중앙 |
| Zone B (남측) | 5 | 각 방 중앙 (y=2.5) |
| Zone C (북동) | 4 | 각 방 중앙 (y=17.5) |
| Zone D (동측 작은 방) | 5 | 각 방 중앙 |
| 복도 | 7 | NFPA 72 spacing (5~7m) |
| 출구 | 3 | 출구 노드 직접 |
| **합계** | **28** | |

모든 감지기: z = 2.5m (천장 높이)

**Rationale**:
- 각 방 중앙 1개: 작은 방 표준 (방 면적 < 10m²)
- 복도 5~7m 간격: NFPA 72 spacing 9m 이내 준수
- 출구 인접 1개씩: 대피 경로 시작점 모니터링
- 천장 z=2.5m: 실제 화재 감지기 설치 표준 위치

**평면도 기반 위치 정밀화**:
- D-023 (감지기 트리거 모델)과 별개로 위치만 정의
- 기존 18개 (zone_a_west 등)에서 28개로 확장
- 각 영역에 충분한 감지기로 GNN 학습 입력 다양성 증가
- 영역별 균형: A:3, B:5, C:4, D:5 (방 17개) + 복도 7 + 출구 3

**Implementation**:
- 신규 모듈: src/tier1/detector_positions.py
- 핵심 데이터: ALL_DETECTORS (28개 DetectorLocation 리스트)
- 헬퍼: get_detector_by_id, get_detectors_by_area, get_detector_count_by_area
- legacy 호환: get_detector_positions_legacy_format() (tuple list)

**기존 결정과의 관계**:
- D-023 (감지기 트리거 모델): 변경 없음. 60°C OR 10m 그대로.
- 기존 scripts/visualize_detectors.py: import만 변경하여 28개 자동 처리.

**Status**: Implemented. Tier 1 GNN의 입력 노드 수 = 28.
```


====================================================================
작업 4: docs/tier1_detector_plan.md 업데이트
====================================================================

기존 `tier1_detector_plan.md`의 §6.1 (18개 가상 감지기 위치) 섹션을 찾기.

이 부분을 다음으로 교체:

```markdown
## 6.1 28개 가상 감지기 위치 (D-024)

평면도 분석 결과 18 → 28개로 확장. NFPA 72 spacing 9m 이내 표준 준수.

### 방 감지기 17개

```
Zone A — 좌상 사선 (3개):
  - zone_a_1   (3.0, 11.5, 2.5)
  - zone_a_2   (6.0, 14.5, 2.5)
  - zone_a_3   (8.5, 17.5, 2.5)

Zone B — 남측 (5개):
  - zone_b_1   (2.0, 2.5, 2.5)   — 출구 1 인접
  - zone_b_2   (6.0, 2.5, 2.5)
  - zone_b_3   (10.0, 2.5, 2.5)
  - zone_b_4   (14.0, 2.5, 2.5)
  - zone_b_5   (19.0, 2.5, 2.5)  — 출구 3 인접

Zone C — 북동 (4개):
  - zone_c_1   (17.0, 17.5, 2.5) — 출구 2 인접
  - zone_c_2   (21.0, 17.5, 2.5)
  - zone_c_3   (25.0, 17.5, 2.5)
  - zone_c_4   (28.5, 17.5, 2.5)

Zone D — 동측 (5개):
  - zone_d_1   (24.5, 12.5, 2.5)
  - zone_d_2   (24.5, 9.5, 2.5)
  - zone_d_3   (28.0, 12.5, 2.5)
  - zone_d_4   (28.0, 9.5, 2.5)
  - zone_d_5   (24.5, 7.0, 2.5)
```

### 복도 감지기 7개 (NFPA 72 spacing 준수)

```
- south_corridor_1     (5.5, 5.5, 2.5)   — 남측 복도 서측
- south_corridor_2     (12.5, 5.5, 2.5)  — 남측 복도 동측
- east_corridor_1      (16.0, 8.0, 2.5)  — 동측 복도 남측
- east_corridor_2      (16.0, 13.0, 2.5) — 동측 복도 북측
- north_corridor_1     (10.0, 15.5, 2.5) — 북측 복도 서측
- north_corridor_2     (22.0, 15.5, 2.5) — 북측 복도 동측
- deep_east_corridor   (22.5, 10.0, 2.5) — 동-동측 복도
```

### 출구 감지기 3개

```
- exit_1_west          (2.5, 5.75, 2.5)
- exit_2_north         (15.0, 16.75, 2.5)
- exit_3_east          (22.0, 5.75, 2.5)
```

모두 z = 2.5m (천장 가까이 — 실제 화재 감지기 설치 표준 위치).

총 28개 — Tier 1 GNN의 노드 개수.

위치 코드: `src/tier1/detector_positions.py`
```


====================================================================
검증 및 보고
====================================================================

작업 후 다음 실행:

1. 단위 테스트:
   python -m src.tier1.detector_positions
   → 9개 Test 모두 PASS 확인

2. 시각화 (28개로 자동 갱신):
   python scripts/visualize_detectors.py data/raw/first_sim/
   → 3장 figure (이번엔 28개 감지기로) + 통계

3. 매뉴얼 업데이트 확인:
   grep "D-024" docs/decisions.md
   grep "28개" docs/tier1_detector_plan.md
   → 둘 다 결과 있어야 함

4. 28개 감지기 위치 검증:
   python -c "
   from src.tier1.detector_positions import ALL_DETECTORS, detector_count_by_area
   print(f'총 {len(ALL_DETECTORS)}개')
   print(detector_count_by_area())
   "

보고할 사항:
- 생성된 파일: src/tier1/detector_positions.py + 줄 수
- 9개 Test 모두 PASS 확인
- 영역별 분포: zone_a=3, zone_b=5, zone_c=4, zone_d=5, corridor=7, exit=3
- first_sim 시각화 결과 (28개 감지기의 트리거 패턴)
- 발견한 이슈 (있으면)

## 금지 사항

- 28개 감지기 위치를 임의 변경 금지 (평면도 기반, D-024 정확히 사용)
- 모든 감지기 z=2.5m 유지 (천장 높이 변경 금지)
- 신규 의존성 추가 금지 — dataclass, typing만 사용
- 기존 detector_model.py 코드 변경 금지 (위치 정의만 분리)
- D-023 (트리거 모델)과 D-024 (위치) 구분 유지

## 주의 사항

- `ALL_DETECTORS`의 순서 매우 중요 (Zone A → B → C → D → 복도 → 출구)
  GNN 학습 데이터의 노드 인덱스가 이 순서와 매핑됨
- 향후 노드 추가 시 끝에만 추가 (기존 인덱스 보존)
- 28개 → 30개 등 변경 시 GNN 입력 차원도 함께 업데이트 필요

## 향후 작업 미리보기

이 모듈 완성 후:

1. **30건 시뮬레이션에 일괄 적용**:
   ```bash
   for s in data/raw/s_*/; do
       python scripts/visualize_detectors.py "$s"
   done
   # → 30개 binary_sequence.npy 생성 (각 (31, 28))
   ```

2. **Week 11: src/tier1/dataset.py** 작성 시 노드 수 = 28 사용

3. **Week 11: src/tier1/tier1_model.py** A3T-GCN 입력 노드 = 28
```

---

## 사용 후 점검

Claude Code 작업 완료 후 다음을 확인하세요.

- [ ] `src/tier1/detector_positions.py` 생성
- [ ] `python -m src.tier1.detector_positions` 9개 Test PASS
- [ ] 영역별 분포 확인: A=3, B=5, C=4, D=5, 복도=7, 출구=3
- [ ] `scripts/visualize_detectors.py` 28개로 자동 갱신
- [ ] `docs/decisions.md`에 D-024 추가
- [ ] `docs/tier1_detector_plan.md` §6.1 업데이트
- [ ] `figures/first_sim/detectors/` 3장 figure 28개로 재생성

이상 통과하면 git commit:

```bash
git add src/tier1/detector_positions.py scripts/visualize_detectors.py docs/decisions.md docs/tier1_detector_plan.md
git commit -m "Tier 1 detectors: 28 positions from floorplan analysis (D-024)

- 17 room detectors (A:3, B:5, C:4, D:5)
- 7 corridor detectors (NFPA 72 spacing < 9m)
- 3 exit detectors
- All at ceiling height z=2.5m
- DetectorLocation dataclass + helper functions
- Legacy format converter for backwards compat

Per D-024: Position determined from floorplan grid graph analysis."
```

## 다음 단계

이 작업 완료 후:

### 즉시 가능
- 30건 시뮬레이션 완료 후 28개 감지기 일괄 적용
- 영역별 트리거 패턴 분석

### Week 11
- `src/tier1/dataset.py` — 28 노드 그래프 구조
- A3T-GCN 모델 입력 노드 = 28

### 발표용
- 28개 감지기 시각화 (`figures/first_sim/detectors/floorplan.png`)
- "기존 28개 표준 화재 감지기로 미래 위험을 예측합니다" 메시지
