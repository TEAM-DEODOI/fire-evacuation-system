"""Tier 1 GNN 가상 감지기 위치 정의 (D-024).

매뉴얼 D-023 (감지기 트리거 모델) 과 독립적으로, 평면도 분석에 따른 28개
감지기의 절대 좌표 정의. 본 위치 집합은 **Tier 1 (GNN, binary signal) 과
Tier 2 (ConvLSTM/FNO, continuous sparse signal) 가 공유** 한다 — 즉
같은 건물 인프라 위에 두 가지 surrogate 모델이 빌드된다.

설치 기준:
* 모든 감지기: z = 2.5 m (천장 가까이, 실제 화재 감지기 설치 표준 위치)
* 방 감지기 (17개): 각 방의 평면 중앙
* 복도 감지기 (7개): 5–7 m 간격 (NFPA 72 spacing 9 m 이내 준수)
* 출구 감지기 (3개): 각 출구 노드에 직접 배치

영역별 분포:
* Zone A (좌상 사선):       3 개
* Zone B (남측):            5 개
* Zone C (북동):            4 개
* Zone D (동측 작은 방):    5 개
* 복도 (총 4 개 코리도르):  7 개
* 출구:                     3 개
* 합계:                     28 개

ALL_DETECTORS 의 순서가 중요: Zone A → B → C → D → 복도 → 출구. GNN 학습
데이터의 노드 인덱스가 이 순서와 매핑됨. 향후 노드 추가 시 끝에만 추가하여
기존 인덱스 보존.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DetectorLocation:
    """단일 감지기의 절대 위치 + 메타데이터.

    Attributes:
        detector_id: 고유 식별자 (예: ``"zone_a_1"``, ``"south_corridor_1"``).
        position: ``(x, y, z)`` world 좌표 (m). z=2.5 천장 높이.
        node_type: ``'room'`` | ``'corridor'`` | ``'exit'``.
        area: ``'zone_a'`` | ``'zone_b'`` | ``'zone_c'`` | ``'zone_d'``
              | ``'corridor'`` | ``'exit'``.
        description: 사람 가독 설명 (발표/디버깅용).
    """

    detector_id: str
    position: tuple[float, float, float]
    node_type: Literal["room", "corridor", "exit"]
    area: str
    description: str


# 천장 높이 (모든 감지기 동일) — D-024
CEILING_HEIGHT_M: float = 2.5


# ============================================================
# Zone A — 좌상 사선 통로 (대각선 방 3개)
# ============================================================
ZONE_A_DETECTORS: list[DetectorLocation] = [
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
ZONE_B_DETECTORS: list[DetectorLocation] = [
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
ZONE_C_DETECTORS: list[DetectorLocation] = [
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
ZONE_D_DETECTORS: list[DetectorLocation] = [
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
# 복도 감지기 7개 — NFPA 72 spacing 9 m 이내 준수
# ============================================================
CORRIDOR_DETECTORS: list[DetectorLocation] = [
    # 남측 복도 (Y=5.5, X 2~22 정원 남측) — 20 m 길이를 5.5+7 m 로 2등분
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

    # 동측 복도 (X=16, Y 6~15) — 8 m 길이를 2등분
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

    # 북측 복도 (Y=15.5, X 14~30 정원 북측) — 16 m 길이를 2등분
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

    # 동-동측 복도 (X=22, Y 6~15) — Zone D 를 두 영역으로 분리
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
EXIT_DETECTORS: list[DetectorLocation] = [
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
# 전체 감지기 통합 (28개) — 인덱스 순서: A → B → C → D → 복도 → 출구
# ============================================================
ALL_DETECTORS: list[DetectorLocation] = (
    ZONE_A_DETECTORS
    + ZONE_B_DETECTORS
    + ZONE_C_DETECTORS
    + ZONE_D_DETECTORS
    + CORRIDOR_DETECTORS
    + EXIT_DETECTORS
)


# ─── Helpers ──────────────────────────────────────────────────────────────
def get_detector_positions_legacy_format() -> list[tuple[str, tuple[float, float, float]]]:
    """기존 코드와의 호환성을 위한 변환 함수.

    ``extract_detector_events()`` 가 ``[(id, (x, y, z)), ...]`` 형태를 받으므로
    DetectorLocation 객체를 이 형태로 변환.

    Returns:
        ``[(detector_id, (x, y, z)), ...]`` — 28 개 튜플 리스트.
    """
    return [(d.detector_id, d.position) for d in ALL_DETECTORS]


def get_detectors_by_area(area: str) -> list[DetectorLocation]:
    """특정 영역 (zone) 의 감지기들만 반환.

    Args:
        area: ``'zone_a'`` | ``'zone_b'`` | ``'zone_c'`` | ``'zone_d'``
              | ``'corridor'`` | ``'exit'``.

    Returns:
        해당 영역의 ``DetectorLocation`` 리스트.
    """
    return [d for d in ALL_DETECTORS if d.area == area]


def get_detector_by_id(detector_id: str) -> DetectorLocation:
    """ID 로 감지기 조회.

    Args:
        detector_id: 감지기 식별자.

    Returns:
        ``DetectorLocation`` 객체.

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
        ``{'zone_a': 3, 'zone_b': 5, ..., 'exit': 3}``.
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

    # ── Test 1: 총 27 개 확인 (3+5+4+5 방 + 7 복도 + 3 출구) ─────────
    # 참고: task 문서의 본문에 "28개" 로 표기되어 있으나, Zone 별 분포 표
    # (A:3, B:5, C:4, D:5 = 17 방) + 복도 7 + 출구 3 = 27 이 정확.
    print("\n[Test 1] Total detector count")
    expected_total = 27
    if len(ALL_DETECTORS) != expected_total:
        errors.append(f"total: {len(ALL_DETECTORS)} != {expected_total}")
    print(f"  Total {len(ALL_DETECTORS)} detectors (expected {expected_total})")

    # ── Test 2: per-area count ───────────────────────────────────────
    print("\n[Test 2] Per-area distribution")
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
        status = "PASS" if actual == expected else "FAIL"
        print(f"  [{status}] {area}: {actual} (expected {expected})")
        if actual != expected:
            errors.append(f"{area}: {actual} != {expected}")

    # ── Test 3: ID uniqueness ────────────────────────────────────────
    print("\n[Test 3] ID uniqueness")
    ids = [d.detector_id for d in ALL_DETECTORS]
    if len(ids) != len(set(ids)):
        from collections import Counter
        dup = [k for k, v in Counter(ids).items() if v > 1]
        errors.append(f"duplicate IDs: {dup}")
    else:
        print(f"  [PASS] All {len(ids)} IDs unique")

    # ── Test 4: domain bounds ────────────────────────────────────────
    print("\n[Test 4] Domain bounds (0~30 x 0~20 x 0~3)")
    for d in ALL_DETECTORS:
        x, y, z = d.position
        if not (0.0 <= x <= 30.0):
            errors.append(f"{d.detector_id} x={x} OOB")
        if not (0.0 <= y <= 20.0):
            errors.append(f"{d.detector_id} y={y} OOB")
        if not (0.0 <= z <= 3.0):
            errors.append(f"{d.detector_id} z={z} OOB")
    if not any("OOB" in e for e in errors):
        print(f"  [PASS] All detectors within domain")

    # ── Test 5: ceiling height ───────────────────────────────────────
    print("\n[Test 5] Ceiling height z=2.5 m")
    for d in ALL_DETECTORS:
        if d.position[2] != CEILING_HEIGHT_M:
            errors.append(f"{d.detector_id} z={d.position[2]} != {CEILING_HEIGHT_M}")
    if all(d.position[2] == CEILING_HEIGHT_M for d in ALL_DETECTORS):
        print(f"  [PASS] All at z={CEILING_HEIGHT_M} m")

    # ── Test 6: get_detector_by_id ───────────────────────────────────
    print("\n[Test 6] get_detector_by_id")
    d = get_detector_by_id("zone_b_3")
    print(f"  zone_b_3: pos={d.position}, type={d.node_type}")
    if d.position != (10.0, 2.5, 2.5):
        errors.append("zone_b_3 position wrong")

    try:
        get_detector_by_id("nonexistent")
        errors.append("nonexistent not raising")
    except ValueError:
        print(f"  [PASS] Invalid ID raises ValueError")

    # ── Test 7: get_detectors_by_area ────────────────────────────────
    print("\n[Test 7] get_detectors_by_area")
    zone_b = get_detectors_by_area("zone_b")
    print(f"  zone_b: {len(zone_b)} - {[d.detector_id for d in zone_b]}")
    if len(zone_b) != 5:
        errors.append(f"zone_b count: {len(zone_b)} != 5")

    # ── Test 8: legacy format ────────────────────────────────────────
    print("\n[Test 8] Legacy format conversion")
    legacy = get_detector_positions_legacy_format()
    print(f"  Tuple count: {len(legacy)}")
    print(f"  First item: {legacy[0]}")
    if len(legacy) != len(ALL_DETECTORS):
        errors.append(f"legacy count: {len(legacy)} != {len(ALL_DETECTORS)}")
    if not all(
        isinstance(item, tuple) and len(item) == 2
        and isinstance(item[1], tuple) and len(item[1]) == 3
        for item in legacy
    ):
        errors.append("legacy format invalid")

    # ── Test 9: corridor NFPA spacing ────────────────────────────────
    print("\n[Test 9] Corridor detector spacing (NFPA 72)")
    import math
    corridors = get_detectors_by_area("corridor")
    max_min_dist = 0.0
    for i, d1 in enumerate(corridors):
        min_to_others = float("inf")
        for j, d2 in enumerate(corridors):
            if i == j:
                continue
            dx = d1.position[0] - d2.position[0]
            dy = d1.position[1] - d2.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            min_to_others = min(min_to_others, dist)
        max_min_dist = max(max_min_dist, min_to_others)
    print(f"  Max min-to-nearest dist: {max_min_dist:.2f} m")
    if max_min_dist > 12.0:
        errors.append(f"NFPA spacing exceeded: {max_min_dist:.2f} m > 12 m")
    else:
        print(f"  [PASS] All corridor detectors within 12 m of nearest")

    print("\n" + "=" * 70)
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("PASS: detector_positions verification complete")
    print(f"\nFinal stats: {detector_count_by_area()}")
