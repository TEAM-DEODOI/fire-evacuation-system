"""Tier 1 / Tier 2 공통 감지기 위치 정의 (D-024 v2 — 평면도 기반 정밀 산출).

사용자가 첨부한 평면도 사진에 직접 표시한 감지기 위치를 좌표화.

색상 정의 (사용자 사진):
* **빨간색 = 방 안 센서 (room)**
* **파란색 = 복도 센서 (corridor)**
* **별표 = 출구 (exit)**

분포 (39 개 total):
* Zone B (남측 방, Y=2.5):        5 rooms
* Zone A 사선 통로 안 작은 방:    5 rooms
* 북측 상단 작은 방 (NW + N):     5 rooms
* Zone C 중간 row (Y=14.5):       4 rooms
* 우측 중간 영역 (X=18):           3 rooms
* 남측 복도 (Y=5):                 6 corridors
* 동측 corridor (X=16):           3 corridors
* 사선 통로 안 복도:               5 corridors
* 출구:                           3 exits
* 합계:                          22 rooms + 14 corridors + 3 exits = 39

본 위치 집합은 Tier 1 (GNN, binary signal) 과 Tier 2 (ConvLSTM/FNO,
continuous signal) 가 공유. 모든 감지기 z = 2.5 m (천장 가까이).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DetectorLocation:
    """단일 감지기의 절대 위치 + 메타데이터."""

    detector_id: str
    position: tuple[float, float, float]
    node_type: Literal["room", "corridor", "exit"]
    area: str
    description: str


CEILING_HEIGHT_M: float = 2.5


# ============================================================
# Zone B — 남측 방 5개 (Y=2.5)
# ============================================================
ZONE_B_DETECTORS: list[DetectorLocation] = [
    DetectorLocation("zone_b_1", (2.5, 2.5, CEILING_HEIGHT_M), "room", "zone_b",
                     "Zone B 좌측 (출구 1 인접)"),
    DetectorLocation("zone_b_2", (6.5, 2.5, CEILING_HEIGHT_M), "room", "zone_b",
                     "Zone B 두번째"),
    DetectorLocation("zone_b_3", (10.0, 2.5, CEILING_HEIGHT_M), "room", "zone_b",
                     "Zone B 중앙"),
    DetectorLocation("zone_b_4", (14.0, 2.5, CEILING_HEIGHT_M), "room", "zone_b",
                     "Zone B 네번째"),
    DetectorLocation("zone_b_5", (19.0, 2.5, CEILING_HEIGHT_M), "room", "zone_b",
                     "Zone B 가장 동측 (출구 3 인접)"),
]


# ============================================================
# Zone A — 사선 통로 안의 작은 방 5개 (v3: zone_a_1 위치 보정)
# ============================================================
ZONE_A_DETECTORS: list[DetectorLocation] = [
    DetectorLocation("zone_a_1", (1.5, 9.0, CEILING_HEIGHT_M), "room", "zone_a",
                     "Zone A 좌하단 작은 방 — v3.3 (1.5, 9)"),
    DetectorLocation("zone_a_2", (4.0, 13.0, CEILING_HEIGHT_M), "room", "zone_a",
                     "Zone A 사선 통로 안 작은 방 1 — v3.1 옆 방으로 이동"),
    DetectorLocation("zone_a_3", (5.0, 15.0, CEILING_HEIGHT_M), "room", "zone_a",
                     "Zone A 사선 통로 안 작은 방 2"),
    DetectorLocation("zone_a_4", (7.5, 17.0, CEILING_HEIGHT_M), "room", "zone_a",
                     "Zone A 사선 통로 안 작은 방 3"),
    DetectorLocation("zone_a_5", (10.5, 18.0, CEILING_HEIGHT_M), "room", "zone_a",
                     "Zone A 사선 통로 우상단"),
]


# ============================================================
# 북측 상단 작은 방 — 4개 (v3: north_room_1 제거)
# ============================================================
NORTH_ROOMS_DETECTORS: list[DetectorLocation] = [
    DetectorLocation("north_room_2", (17.5, 18.5, CEILING_HEIGHT_M), "room", "north",
                     "북측 상단 작은 방 1 (Zone C 좌측) — v3.1 방 안으로"),
    DetectorLocation("north_room_3", (21.5, 18.5, CEILING_HEIGHT_M), "room", "north",
                     "북측 상단 작은 방 2 (Zone C 중앙)"),
    DetectorLocation("north_room_4", (25.0, 18.0, CEILING_HEIGHT_M), "room", "north",
                     "북측 상단 작은 방 3"),
    DetectorLocation("north_room_5", (28.0, 18.0, CEILING_HEIGHT_M), "room", "north",
                     "북측 상단 작은 방 4 (동측 끝)"),
]


# ============================================================
# Zone C — 중간 row (Y=15.0, 방 정중앙, v3.2 ↑ + 1개 추가)
# ============================================================
ZONE_C_DETECTORS: list[DetectorLocation] = [
    DetectorLocation("zone_c_mid_1", (18.0, 15.0, CEILING_HEIGHT_M), "room", "zone_c",
                     "Zone C 중간 좌측"),
    DetectorLocation("zone_c_mid_2", (21.5, 15.0, CEILING_HEIGHT_M), "room", "zone_c",
                     "Zone C 중간 left-center (방 정중앙)"),
    DetectorLocation("zone_c_mid_3", (24.0, 15.0, CEILING_HEIGHT_M), "room", "zone_c",
                     "Zone C 중간 right-center (v3.3 X-1)"),
    DetectorLocation("zone_c_mid_4", (27.0, 15.0, CEILING_HEIGHT_M), "room", "zone_c",
                     "Zone C 중간 우측 (v3.3 X-1)"),
    DetectorLocation("zone_c_mid_5", (29.0, 15.0, CEILING_HEIGHT_M), "room", "zone_c",
                     "Zone C 가장 우측 (v3.2 신규)"),
]


# ============================================================
# 우측 중간 영역 — 작은 방 3개 (X=18, 정원 동측 옆)
# ============================================================
ZONE_D_DETECTORS: list[DetectorLocation] = [
    DetectorLocation("zone_d_1", (18.0, 12.5, CEILING_HEIGHT_M), "room", "zone_d",
                     "우측 중간 작은 방 1"),
    DetectorLocation("zone_d_2", (18.0, 9.5, CEILING_HEIGHT_M), "room", "zone_d",
                     "우측 중간 작은 방 2"),
    DetectorLocation("zone_d_3", (18.0, 7.0, CEILING_HEIGHT_M), "room", "zone_d",
                     "우측 중간 작은 방 3"),
]


# ============================================================
# 복도 감지기 — v3: east_corridor_1 + diag_corridor_2/3/4/5 제거,
#                    east_corridor_2/3 아래로 이동, 북측 corridor 3개 + 사선 1개 추가
# ============================================================
CORRIDOR_DETECTORS: list[DetectorLocation] = [
    # 남측 복도 (Y=5)
    DetectorLocation("south_corridor_1", (2.5, 5.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "남측 복도 1 (좌측, 출구 1 인접)"),
    DetectorLocation("south_corridor_2", (6.5, 5.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "남측 복도 2"),
    DetectorLocation("south_corridor_3", (10.0, 5.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "남측 복도 3"),
    DetectorLocation("south_corridor_4", (15.5, 5.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "남측 복도 4 (중앙)"),
    DetectorLocation("south_corridor_5", (19.0, 5.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "남측 복도 5"),
    DetectorLocation("south_corridor_6", (21.5, 5.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "남측 복도 6 (우측, 출구 3 인접)"),

    # 동측 corridor (X=16.5, 세로) — v3 아래로 이동
    DetectorLocation("east_corridor_2", (16.5, 12.5, CEILING_HEIGHT_M), "corridor",
                     "corridor", "동측 corridor 중상 (v3 아래로 이동)"),
    DetectorLocation("east_corridor_3", (16.5, 9.5, CEILING_HEIGHT_M), "corridor",
                     "corridor", "동측 corridor 중하 (v3 아래로 이동)"),

    # 사선 통로 안 복도 (Zone A) — v3.3: diag_corridor_1 삭제, 2/3/4 평행이동
    DetectorLocation("diag_corridor_2", (5.0, 11.5, CEILING_HEIGHT_M), "corridor",
                     "corridor", "사선 통로 좌측 안쪽 복도 (v3.3 +2,-1)"),
    DetectorLocation("diag_corridor_3", (7.5, 14.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "사선 통로 중간 복도 (v3.3 +2,0)"),
    DetectorLocation("diag_corridor_4", (9.5, 16.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "사선 통로 우상단 복도 (v3.3 +2,0)"),

    # 북측 corridor (Zone C 위) — v3 신규
    DetectorLocation("north_corridor_1", (21.5, 16.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "북측 corridor 1 (Zone C 좌)"),
    DetectorLocation("north_corridor_2", (25.0, 16.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "북측 corridor 2 (Zone C 중앙)"),
    DetectorLocation("north_corridor_3", (28.0, 16.0, CEILING_HEIGHT_M), "corridor",
                     "corridor", "북측 corridor 3 (Zone C 우)"),
]


# ============================================================
# 출구 3개
# ============================================================
EXIT_DETECTORS: list[DetectorLocation] = [
    DetectorLocation("exit_1_west", (2.5, 5.75, CEILING_HEIGHT_M), "exit", "exit",
                     "서측 출구"),
    DetectorLocation("exit_2_north", (15.0, 16.75, CEILING_HEIGHT_M), "exit", "exit",
                     "북측 출구"),
    DetectorLocation("exit_3_east", (22.0, 5.75, CEILING_HEIGHT_M), "exit", "exit",
                     "동측 출구"),
]


# ============================================================
# 전체 통합 — 인덱스 순서 고정 (GNN 학습 데이터 매핑)
# ============================================================
ALL_DETECTORS: list[DetectorLocation] = (
    ZONE_B_DETECTORS
    + ZONE_A_DETECTORS
    + NORTH_ROOMS_DETECTORS
    + ZONE_C_DETECTORS
    + ZONE_D_DETECTORS
    + CORRIDOR_DETECTORS
    + EXIT_DETECTORS
)


# ─── Helpers ──────────────────────────────────────────────────────────────
def get_detector_positions_legacy_format() -> list[tuple[str, tuple[float, float, float]]]:
    """``[(detector_id, (x, y, z)), ...]`` 변환."""
    return [(d.detector_id, d.position) for d in ALL_DETECTORS]


def get_detectors_by_area(area: str) -> list[DetectorLocation]:
    return [d for d in ALL_DETECTORS if d.area == area]


def get_detector_by_id(detector_id: str) -> DetectorLocation:
    for d in ALL_DETECTORS:
        if d.detector_id == detector_id:
            return d
    raise ValueError(f"Unknown detector_id: {detector_id}")


def detector_count_by_area() -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in ALL_DETECTORS:
        counts[d.area] = counts.get(d.area, 0) + 1
    return counts


def detector_count_by_type() -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in ALL_DETECTORS:
        counts[d.node_type] = counts.get(d.node_type, 0) + 1
    return counts


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=" * 70)
    print("detector_positions.py (v2 / floorplan-based) self-test")
    print("=" * 70)
    errors: list[str] = []

    print("\n[Test 1] Total count")
    print(f"  Total: {len(ALL_DETECTORS)}")
    by_type = detector_count_by_type()
    print(f"  By type: {by_type}")
    by_area = detector_count_by_area()
    print(f"  By area: {by_area}")

    print("\n[Test 2] All IDs unique")
    ids = [d.detector_id for d in ALL_DETECTORS]
    if len(ids) != len(set(ids)):
        errors.append("duplicate IDs")
    else:
        print(f"  [PASS] {len(ids)} unique IDs")

    print("\n[Test 3] Domain bounds (0~30 x 0~20 x 0~3)")
    for d in ALL_DETECTORS:
        x, y, z = d.position
        if not (0.0 <= x <= 30.0 and 0.0 <= y <= 20.0 and 0.0 <= z <= 3.0):
            errors.append(f"{d.detector_id} OOB at {d.position}")
    if not errors:
        print(f"  [PASS] All in domain")

    print("\n[Test 4] Ceiling height z=2.5")
    for d in ALL_DETECTORS:
        if d.position[2] != CEILING_HEIGHT_M:
            errors.append(f"{d.detector_id} z={d.position[2]}")
    if not any("z=" in e for e in errors):
        print(f"  [PASS] All at z=2.5")

    print("\n[Test 5] Legacy format")
    legacy = get_detector_positions_legacy_format()
    print(f"  Tuple count: {len(legacy)}")
    if len(legacy) != len(ALL_DETECTORS):
        errors.append(f"legacy count mismatch")

    print("\n" + "=" * 70)
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("PASS")
    print(f"\nSummary: {len(ALL_DETECTORS)} detectors  "
          f"(rooms: {by_type.get('room', 0)}, "
          f"corridors: {by_type.get('corridor', 0)}, "
          f"exits: {by_type.get('exit', 0)})")
