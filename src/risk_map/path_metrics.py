"""H6 가설 검증을 위한 경로 안전성 메트릭 (D-022).

매뉴얼 D-022 (Phase 1 권장)에 따라 4개 메트릭으로 H6 가설
(동적 vs 정적 경로 알고리즘) 검증:

1. ``peak_danger``        — 경로상 최대 위험도 ∈ [0, 1]
2. ``time_in_hazard_s``   — 위험 영역(>0.5) 체류 시간 (s)
3. ``aset_margin_s``      — ASET - arrival_time 의 최소값 (양수 = 안전)
4. ``fed_final``          — 누적 FED (보조 지표, ``src/risk_map/fed.py`` 재사용)

각 함수는 pure NumPy. 의존성 최소화 (numpy + dataclass + math).

배경 (D-022 발견):
1500 kW 화재, 300초 시뮬레이션에서 누적 FED 최대값 = 0.043 (임계값 0.3 미달).
화재 크기 증가는 비현실적 + 30건 재시뮬레이션 비용 큼.
Peak Danger / Time-in-Hazard 는 시뮬레이션 시간과 무관 → 더 직접적 H6 메트릭.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from src.risk_map.fed import accumulate_fed_co
from src.risk_map.risk_map_class import RiskMap
from src.shared.constants import TENABILITY
from src.shared.coordinates import world_to_grid


@dataclass(frozen=True)
class PathSafetyMetrics:
    """경로 안전성 종합 평가 — H6 가설 검증용 (불변).

    Attributes:
        peak_danger: 경로상 마주친 최대 위험도 [0, 1].
            정적 알고리즘에서 더 크고 동적에서 더 작을 것으로 예상.
        time_in_hazard_s: 위험 영역(>0.5)에서 보낸 총 시간 (초).
            정적에서 더 크고 동적에서 더 작을 것으로 예상.
        aset_margin_s: ASET - arrival_time 의 최소값 (초).
            양수 = 안전 도달, 음수 = 도달 시 이미 위험.
        fed_final: 경로 종료 시점 누적 FED (D-008/D-009).
            300초 시뮬에서는 차이가 작을 수 있음.
        arrived_safely: 출구 도달 여부.
        travel_time_s: 출발 → 도달까지 총 시간 (초).
        n_path_points: 경로 sample 개수.
    """

    peak_danger: float
    time_in_hazard_s: float
    aset_margin_s: float
    fed_final: float
    arrived_safely: bool
    travel_time_s: float
    n_path_points: int

    def to_dict(self) -> dict:
        """JSON-friendly dict. 무한값은 ``None`` / ``"neg_inf"`` 로 직렬화."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None if math.isinf(v) and v > 0 else "neg_inf"
        return d

    def summary_line(self) -> str:
        """짧은 한 줄 요약 (발표 표 / 로그용)."""
        margin_str = (
            f"{self.aset_margin_s:+.1f}s"
            if math.isfinite(self.aset_margin_s) else "∞"
        )
        arrived = "OK" if self.arrived_safely else "FAIL"
        return (
            f"peak={self.peak_danger:.3f}  "
            f"hazard={self.time_in_hazard_s:.1f}s  "
            f"margin={margin_str}  "
            f"FED={self.fed_final:.4f}  "
            f"arrived={arrived}  "
            f"t={self.travel_time_s:.1f}s"
        )


# ─────────────────────────────────────────────────────────────────────────
# 메트릭 1: Peak Danger
# ─────────────────────────────────────────────────────────────────────────
def compute_peak_danger(
    path_xyz: np.ndarray,
    path_times: np.ndarray,
    risk_map: RiskMap,
) -> float:
    """경로상 마주친 최대 위험도.

    ``risk_map.query`` 가 OOB 입력에 1.0 을 반환하므로 OOB 점은 자동으로
    가장 위험한 값으로 간주된다 (interface_contracts.md §2 안전 규칙).
    빈 경로면 0.0.

    Args:
        path_xyz: ``(N, 3)`` 경로 좌표 (m).
        path_times: ``(N,)`` 각 점의 도달 시간 (s).
        risk_map: :class:`RiskMap` 구현체.

    Returns:
        최대 위험도 ∈ [0, 1].
    """
    if len(path_xyz) == 0:
        return 0.0

    arr = np.asarray(path_xyz, dtype=np.float64)
    ts = np.asarray(path_times, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"path_xyz must be (N, 3), got {arr.shape}")
    if ts.shape != (arr.shape[0],):
        raise ValueError(
            f"path_times shape {ts.shape} != (N={arr.shape[0]},)"
        )

    dangers = [float(risk_map.query(p, t=t)) for p, t in zip(arr, ts)]
    return max(dangers)


# ─────────────────────────────────────────────────────────────────────────
# 메트릭 2: Time in Hazardous Zone
# ─────────────────────────────────────────────────────────────────────────
def compute_time_in_hazard(
    path_xyz: np.ndarray,
    path_times: np.ndarray,
    risk_map: RiskMap,
    danger_threshold: float = TENABILITY.AGGREGATE_THRESHOLD,
) -> float:
    """위험 영역(danger > threshold)에서 보낸 누적 시간 (초).

    구간 ``[t_i, t_{i+1}]`` 에서 시작점 ``i`` 의 danger 가 ``threshold`` 를
    초과하면 ``dt = t_{i+1} - t_i`` 를 누적. 보수적 (살짝 과대평가) 추정.

    Args:
        path_xyz: ``(N, 3)``.
        path_times: ``(N,)``.
        risk_map: :class:`RiskMap`.
        danger_threshold: 기본 ``TENABILITY.AGGREGATE_THRESHOLD`` (0.5).

    Returns:
        위험 영역 체류 시간 (초). 0 이상.
    """
    if len(path_xyz) < 2:
        return 0.0

    arr = np.asarray(path_xyz, dtype=np.float64)
    ts = np.asarray(path_times, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"path_xyz must be (N, 3), got {arr.shape}")
    if ts.shape != (arr.shape[0],):
        raise ValueError(
            f"path_times {ts.shape} != (N={arr.shape[0]},)"
        )

    total = 0.0
    for i in range(len(arr) - 1):
        d_i = float(risk_map.query(arr[i], t=ts[i]))
        if d_i > danger_threshold:
            dt = float(ts[i + 1] - ts[i])
            total += max(dt, 0.0)
    return total


# ─────────────────────────────────────────────────────────────────────────
# 메트릭 3: ASET Margin
# ─────────────────────────────────────────────────────────────────────────
def compute_aset_margin(
    path_xyz: np.ndarray,
    path_times: np.ndarray,
    aset_map: np.ndarray,
) -> float:
    """경로상 최소 ASET 여유 시간.

    각 점에서 ``ASET[cell] - arrival_time`` 계산 후 최소값 반환.
    양수면 안전 도달, 음수면 그 점에 도달했을 때 이미 위험.

    OOB 점은 ``max(aset_map)`` (보수적 안전 측) 으로 처리.
    빈 경로면 ``+inf``.

    Args:
        path_xyz: ``(N, 3)``.
        path_times: ``(N,)``.
        aset_map: ``(60, 40, 6)`` ASET 맵 (`compute_aset_map` 결과).

    Returns:
        최소 여유 시간 (초).
    """
    if len(path_xyz) == 0:
        return float("inf")

    arr = np.asarray(path_xyz, dtype=np.float64)
    ts = np.asarray(path_times, dtype=np.float64)
    am = np.asarray(aset_map, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"path_xyz must be (N, 3), got {arr.shape}")
    if ts.shape != (arr.shape[0],):
        raise ValueError(
            f"path_times {ts.shape} != (N={arr.shape[0]},)"
        )
    if am.ndim != 3:
        raise ValueError(f"aset_map must be 3-D, got {am.shape}")

    aset_max = float(am.max())
    margins: list[float] = []
    for p, t in zip(arr, ts):
        idx = world_to_grid(np.asarray(p))
        ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
        if (
            ix < 0 or iy < 0 or iz < 0
            or ix >= am.shape[0]
            or iy >= am.shape[1]
            or iz >= am.shape[2]
        ):
            aset_at_point = aset_max  # OOB → 보수적 안전
        else:
            aset_at_point = float(am[ix, iy, iz])
        margins.append(aset_at_point - float(t))
    return min(margins)


# ─────────────────────────────────────────────────────────────────────────
# 메트릭 4: FED Final (보조 — fed.py 재사용)
# ─────────────────────────────────────────────────────────────────────────
def compute_fed_along_path(
    path_xyz: np.ndarray,
    path_times: np.ndarray,
    co_grid: np.ndarray,
    co_times: np.ndarray,
) -> float:
    """경로상 누적 CO-FED 의 종착값 (D-008 simplified form).

    각 경로 점에서 시간상 가장 가까운 ``co_times`` frame 의 cell-level CO 값을
    sample 한 뒤 :func:`accumulate_fed_co` 로 누적. 경로 sample 간격은
    균등 ``dt = (t_end - t_start) / (N - 1)`` 로 가정.

    Args:
        path_xyz: ``(N, 3)``.
        path_times: ``(N,)``.
        co_grid: ``(T, X, Y, Z)`` CO grid (ppm).
        co_times: ``(T,)`` frame 시각.

    Returns:
        경로 종료 시점 누적 FED. 빈/단일점 경로는 0.0.
    """
    if len(path_xyz) < 2:
        return 0.0

    arr = np.asarray(path_xyz, dtype=np.float64)
    ts = np.asarray(path_times, dtype=np.float64)
    co = np.asarray(co_grid, dtype=np.float32)
    ct = np.asarray(co_times, dtype=np.float64)

    co_along: list[float] = []
    for p, t in zip(arr, ts):
        t_idx = int(np.argmin(np.abs(ct - t)))
        t_idx = max(0, min(co.shape[0] - 1, t_idx))
        idx = world_to_grid(np.asarray(p))
        ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
        if (
            ix < 0 or iy < 0 or iz < 0
            or ix >= co.shape[1]
            or iy >= co.shape[2]
            or iz >= co.shape[3]
        ):
            co_along.append(0.0)  # OOB = building 밖 = CO 없음
        else:
            co_along.append(float(co[t_idx, ix, iy, iz]))

    co_arr = np.asarray(co_along)
    if len(co_arr) < 2:
        return 0.0
    dt_total = float(ts[-1] - ts[0])
    dt_step = dt_total / max(len(co_arr) - 1, 1)
    if dt_step <= 0:
        return 0.0
    fed = accumulate_fed_co(co_arr, dt_seconds=dt_step)
    return float(fed[-1])


# ─────────────────────────────────────────────────────────────────────────
# 통합 평가 함수
# ─────────────────────────────────────────────────────────────────────────
def evaluate_path_safety(
    path_xyz: np.ndarray,
    path_times: np.ndarray,
    risk_map: RiskMap,
    aset_map: np.ndarray,
    co_grid: np.ndarray,
    co_times: np.ndarray,
    exit_xyz: Optional[np.ndarray] = None,
    arrival_tolerance_m: float = 1.0,
) -> PathSafetyMetrics:
    """4 메트릭 + 도달 여부 + travel time 일괄 계산.

    Args:
        path_xyz: ``(N, 3)`` 경로.
        path_times: ``(N,)`` 도달 시간.
        risk_map: ``query(xyz, t)`` 가능한 :class:`RiskMap`.
        aset_map: ``(60, 40, 6)`` ASET 맵.
        co_grid: ``(T, X, Y, Z)`` CO grid (ppm).
        co_times: ``(T,)`` frame 시각.
        exit_xyz: ``(3,)`` 목표 출구. ``None`` 이면 ``arrived_safely=False``.
        arrival_tolerance_m: 출구 도달 판정 반경.

    Returns:
        :class:`PathSafetyMetrics`.
    """
    arr = np.asarray(path_xyz, dtype=np.float64)
    ts = np.asarray(path_times, dtype=np.float64)
    n = len(arr)

    peak = compute_peak_danger(arr, ts, risk_map)
    hazard_time = compute_time_in_hazard(arr, ts, risk_map)
    margin = compute_aset_margin(arr, ts, aset_map)
    fed = compute_fed_along_path(arr, ts, co_grid, co_times)

    travel = float(ts[-1] - ts[0]) if n >= 2 else 0.0

    arrived = False
    if exit_xyz is not None and n > 0:
        last = arr[-1]
        dist = float(np.linalg.norm(last - np.asarray(exit_xyz)))
        arrived = dist <= arrival_tolerance_m

    return PathSafetyMetrics(
        peak_danger=peak,
        time_in_hazard_s=hazard_time,
        aset_margin_s=margin,
        fed_final=fed,
        arrived_safely=arrived,
        travel_time_s=travel,
        n_path_points=n,
    )


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys

    print("=" * 60)
    print("path_metrics.py self-test")
    print("=" * 60)

    from src.risk_map.aset import compute_aset_map
    from src.risk_map.risk_map_class import StaticRiskMap
    from src.shared.constants import DT_SLCF, GRID_SHAPE

    errors: list[str] = []

    nx_, ny_, nz_ = GRID_SHAPE
    times = np.arange(0.0, 31 * DT_SLCF, DT_SLCF)
    danger = np.zeros((31, nx_, ny_, nz_), dtype=np.float32)
    danger[5:, 28:34, 18:24, 2:4] = 0.85  # 화재 영역 (t=50s ~)
    rm = StaticRiskMap(danger_array=danger, times=times)

    co_grid = np.zeros((31, nx_, ny_, nz_), dtype=np.float32)
    co_grid[5:, 28:34, 18:24, 2:4] = 800.0  # 화재 영역 CO 800 ppm

    aset_map = compute_aset_map(danger)

    # ── 1. Safe path (화재 회피, 출구 도달) ─────────────────────────────
    print("\n[1] 안전한 경로 (화재 영역 회피)")
    safe_path = np.array([
        [5.0, 5.0, 1.5],
        [3.0, 5.0, 1.5],
        [1.0, 5.0, 1.5],
        [0.0, 5.0, 1.5],
    ])
    safe_times = np.array([0.0, 10.0, 20.0, 30.0])
    m_safe = evaluate_path_safety(
        safe_path, safe_times, rm, aset_map, co_grid, times,
        exit_xyz=np.array([0.0, 5.0, 1.5]),
    )
    print(f"  {m_safe.summary_line()}")
    if m_safe.peak_danger > 0.1:
        errors.append(f"안전 경로 peak_danger 너무 큼: {m_safe.peak_danger}")
    if m_safe.time_in_hazard_s > 0:
        errors.append(f"안전 경로 hazard time 양수: {m_safe.time_in_hazard_s}")
    if not m_safe.arrived_safely:
        errors.append("안전 경로 도달 실패")
    if m_safe.aset_margin_s < 200:
        errors.append(f"안전 경로 aset_margin 너무 작음: {m_safe.aset_margin_s}")
    print("  PASS")

    # ── 2. Dangerous path (화재 통과) ──────────────────────────────────
    print("\n[2] 위험한 경로 (화재 통과)")
    danger_path = np.array([
        [10.0, 10.0, 1.5],
        [13.0, 10.0, 1.5],
        [16.0, 10.0, 1.5],  # 화재 영역
        [19.0, 10.0, 1.5],
    ])
    danger_times = np.array([60.0, 70.0, 80.0, 90.0])  # 화재 발생 후
    m_dang = evaluate_path_safety(
        danger_path, danger_times, rm, aset_map, co_grid, times,
        exit_xyz=np.array([19.0, 10.0, 1.5]),
    )
    print(f"  {m_dang.summary_line()}")
    if m_dang.peak_danger < 0.5:
        errors.append(f"위험 경로 peak_danger 작음: {m_dang.peak_danger}")
    if m_dang.time_in_hazard_s == 0:
        errors.append("위험 경로 hazard time이 0")
    print("  PASS")

    # ── 3. H6 비교: safe vs dangerous ─────────────────────────────────
    print("\n[3] H6 비교: 안전 vs 위험")
    print(
        f"  peak_danger     | safe={m_safe.peak_danger:.3f}  "
        f"danger={m_dang.peak_danger:.3f}  "
        f"Δ={m_dang.peak_danger - m_safe.peak_danger:+.3f}"
    )
    print(
        f"  time_in_hazard  | safe={m_safe.time_in_hazard_s:.1f}s  "
        f"danger={m_dang.time_in_hazard_s:.1f}s  "
        f"Δ={m_dang.time_in_hazard_s - m_safe.time_in_hazard_s:+.1f}s"
    )
    print(
        f"  aset_margin     | safe={m_safe.aset_margin_s:.1f}s  "
        f"danger={m_dang.aset_margin_s:.1f}s  "
        f"Δ={m_dang.aset_margin_s - m_safe.aset_margin_s:+.1f}s"
    )
    print(
        f"  fed_final       | safe={m_safe.fed_final:.4f}  "
        f"danger={m_dang.fed_final:.4f}  "
        f"Δ={m_dang.fed_final - m_safe.fed_final:+.4f}"
    )
    if m_dang.peak_danger <= m_safe.peak_danger:
        errors.append("위험 경로의 peak_danger가 안전 이하")
    if m_dang.time_in_hazard_s <= m_safe.time_in_hazard_s:
        errors.append("위험 경로의 hazard time이 안전 이하")
    if m_dang.aset_margin_s >= m_safe.aset_margin_s:
        errors.append("위험 경로의 aset_margin이 안전 이상")
    print("  PASS: 위험 경로가 모든 메트릭에서 더 위험 (H6 검증 가능)")

    # ── 4. 빈 경로 ────────────────────────────────────────────────────
    print("\n[4] 빈 경로 처리")
    m_empty = evaluate_path_safety(
        np.empty((0, 3)), np.empty(0), rm, aset_map, co_grid, times,
    )
    if m_empty.peak_danger != 0.0 or m_empty.time_in_hazard_s != 0.0:
        errors.append("빈 경로 메트릭이 0 아님")
    if not math.isinf(m_empty.aset_margin_s) or m_empty.aset_margin_s < 0:
        errors.append("빈 경로 aset_margin 이 +inf 아님")
    print(f"  PASS: {m_empty.summary_line()}")

    # ── 5. 단일 점 경로 ───────────────────────────────────────────────
    print("\n[5] 단일 점 경로")
    single_pt = np.array([[15.0, 10.0, 1.5]])
    single_t = np.array([100.0])
    m_single = evaluate_path_safety(
        single_pt, single_t, rm, aset_map, co_grid, times,
    )
    print(f"  PASS: {m_single.summary_line()}")
    if m_single.travel_time_s != 0.0:
        errors.append(f"단일점 travel_time {m_single.travel_time_s} != 0")
    if m_single.fed_final != 0.0:
        errors.append(f"단일점 fed_final {m_single.fed_final} != 0")

    # ── 6. JSON 직렬화 ─────────────────────────────────────────────────
    print("\n[6] to_dict 직렬화")
    d_dang = m_dang.to_dict()
    j = json.dumps(d_dang, ensure_ascii=False)
    print(f"  PASS: JSON head = {j[:120]}...")
    # 빈 경로 (aset_margin=inf) JSON serialise → None
    d_empty = m_empty.to_dict()
    if d_empty["aset_margin_s"] is not None:
        errors.append(
            f"inf aset_margin 직렬화 실패: {d_empty['aset_margin_s']}"
        )

    # ── 7. OOB 경로 점 ─────────────────────────────────────────────────
    print("\n[7] OOB 경로 점 처리 (도메인 밖)")
    oob_path = np.array([
        [-5.0, 10.0, 1.5],
        [15.0, 10.0, 1.5],
        [35.0, 10.0, 1.5],
    ])
    oob_times = np.array([0.0, 100.0, 200.0])
    m_oob = evaluate_path_safety(
        oob_path, oob_times, rm, aset_map, co_grid, times,
    )
    print(f"  peak={m_oob.peak_danger:.3f}  aset_margin={m_oob.aset_margin_s:.1f}s")
    if m_oob.peak_danger != 1.0:
        errors.append(f"OOB 경로의 peak_danger != 1.0: {m_oob.peak_danger}")
    print("  PASS: OOB → peak_danger = 1.0")

    # ── 8. 입력 validation ────────────────────────────────────────────
    print("\n[8] 입력 validation")
    try:
        compute_peak_danger(np.zeros((5, 4)), np.zeros(5), rm)
    except ValueError:
        print("  PASS: (N, 4) shape → ValueError")
    else:
        errors.append("잘못된 shape에서 ValueError 발생 안 함")
    try:
        compute_time_in_hazard(np.zeros((5, 3)), np.zeros(3), rm)
    except ValueError:
        print("  PASS: shape mismatch → ValueError")
    else:
        errors.append("shape mismatch에서 ValueError 발생 안 함")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nPASS: path_metrics 검증 완료")
