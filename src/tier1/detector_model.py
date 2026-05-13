"""Tier 1 GNN 화재 감지기 트리거 모델 (D-023).

매뉴얼 D-023에 따른 실제 화재 감지기 표준 모사:
* **열 감지기 (NFPA 57°C + 한국 70°C 중간값)**: 60°C
* **연기 감지기 (UL 268 1.5%/ft ≈ 13m)**: 10m visibility (보수적)
* **OR 조합**: 둘 중 하나라도 만족 시 작동 (latched)
* **CO 감지기 제외**: UL 2034 임계값이 300s 시뮬에서 도달 안 함

기존 ``detector_extractor.py`` 와의 차이:
* ``detector_extractor.py``: 온도 단독 임계 (단순)
* 본 모듈 (``detector_model.py``): 열 + 연기 OR (D-023, 표준 준수)

본 모듈 출력 ``binary_sequence`` 가 Tier 1 GNN 의 핵심 학습 입력.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.shared.constants import GRID_SHAPE, N_TIMESTEPS
from src.shared.coordinates import world_to_grid


# ─── D-023 임계값 ─────────────────────────────────────────────────────────
HEAT_THRESHOLD_C: float = 60.0
"""열 감지기 작동 온도 (D-023). NFPA 57°C + 한국 KOFEIS 70°C 중간값."""

SMOKE_THRESHOLD_M: float = 10.0
"""연기 감지기 작동 visibility (D-023). UL 268 13m 의 보수적 사용."""


# ─── DetectorEvent ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DetectorEvent:
    """단일 감지기의 트리거 이벤트.

    Attributes:
        detector_id: 감지기 식별자.
        position: ``(x, y, z)`` world 좌표 (m).
        activation_frame: 최초 트리거된 frame 인덱스 (0~30). None 이면 트리거 안 됨.
        activation_time_s: 최초 트리거 시각 (초). None 이면 트리거 안 됨.
        trigger_reason: ``"heat"`` | ``"smoke"`` | ``"both"`` | ``"none"``.
    """

    detector_id: str
    position: Tuple[float, float, float]
    activation_frame: Optional[int]
    activation_time_s: Optional[float]
    trigger_reason: str

    def is_activated(self) -> bool:
        return self.activation_frame is not None


# ─── Single-cell trigger ──────────────────────────────────────────────────
def check_single_cell_trigger(
    temperature_c: np.ndarray,    # (T,)
    visibility_m: np.ndarray,     # (T,)
    heat_threshold_c: float = HEAT_THRESHOLD_C,
    smoke_threshold_m: float = SMOKE_THRESHOLD_M,
) -> Tuple[Optional[int], str]:
    """단일 셀의 시간 시리즈에서 D-023 트리거 시점 결정.

    Args:
        temperature_c: ``(T,)`` 시간별 온도 (°C, raw).
        visibility_m: ``(T,)`` 시간별 visibility (m, raw).
        heat_threshold_c: D-023 = 60 °C.
        smoke_threshold_m: D-023 = 10 m.

    Returns:
        ``(activation_frame, reason)``.

        * ``activation_frame``: 최초 트리거 frame (0-indexed). ``None`` = 미트리거.
        * ``reason``: ``"heat"``, ``"smoke"``, ``"both"``, ``"none"``.

    Logic:
        OR 조합. 가장 먼저 도달한 쪽이 reason. 동일 frame 이면 "both".
    """
    if temperature_c.shape != visibility_m.shape:
        raise ValueError(
            f"shape mismatch: T={temperature_c.shape}, V={visibility_m.shape}"
        )
    if temperature_c.ndim != 1:
        raise ValueError(f"inputs must be 1-D, got {temperature_c.ndim}-D")

    heat_triggered = temperature_c > heat_threshold_c
    smoke_triggered = visibility_m < smoke_threshold_m

    heat_first = int(np.argmax(heat_triggered)) if heat_triggered.any() else -1
    smoke_first = int(np.argmax(smoke_triggered)) if smoke_triggered.any() else -1

    if heat_first == -1 and smoke_first == -1:
        return None, "none"
    if heat_first == -1:
        return smoke_first, "smoke"
    if smoke_first == -1:
        return heat_first, "heat"
    if heat_first == smoke_first:
        return heat_first, "both"
    if heat_first < smoke_first:
        return heat_first, "heat"
    return smoke_first, "smoke"


# ─── Multi-detector extraction ─────────────────────────────────────────────
def extract_detector_events(
    temperature_grid: np.ndarray,    # (T, X, Y, Z) °C
    visibility_grid: np.ndarray,     # (T, X, Y, Z) m
    detector_positions: List[Tuple[str, Tuple[float, float, float]]],
    dt_seconds: float = 10.0,
    heat_threshold_c: float = HEAT_THRESHOLD_C,
    smoke_threshold_m: float = SMOKE_THRESHOLD_M,
) -> List[DetectorEvent]:
    """FDS 슬라이스 (T grid + V grid) → 감지기 이벤트 리스트.

    Args:
        temperature_grid: ``(31, 60, 40, 6)`` °C.
        visibility_grid: ``(31, 60, 40, 6)`` m.
        detector_positions: ``[(id, (x, y, z)), ...]`` world 좌표.
        dt_seconds: frame 간격 (default 10 s).
        heat_threshold_c / smoke_threshold_m: D-023 임계값.

    Returns:
        ``DetectorEvent`` 리스트 (입력 감지기당 한 개).

    Note:
        OOB 위치는 ``trigger_reason="none"`` 으로 반환.
    """
    if visibility_grid.shape != temperature_grid.shape:
        raise ValueError(
            f"T grid {temperature_grid.shape} != V grid {visibility_grid.shape}"
        )

    events: List[DetectorEvent] = []
    for det_id, pos in detector_positions:
        idx = world_to_grid(np.asarray(pos))
        ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
        if (ix < 0 or iy < 0 or iz < 0
                or ix >= temperature_grid.shape[1]
                or iy >= temperature_grid.shape[2]
                or iz >= temperature_grid.shape[3]):
            events.append(DetectorEvent(
                detector_id=det_id, position=pos,
                activation_frame=None, activation_time_s=None,
                trigger_reason="none",
            ))
            continue

        T_series = temperature_grid[:, ix, iy, iz]
        V_series = visibility_grid[:, ix, iy, iz]
        frame, reason = check_single_cell_trigger(
            T_series, V_series, heat_threshold_c, smoke_threshold_m,
        )
        events.append(DetectorEvent(
            detector_id=det_id, position=pos,
            activation_frame=frame,
            activation_time_s=(float(frame * dt_seconds)
                                if frame is not None else None),
            trigger_reason=reason,
        ))
    return events


def build_binary_sequence(
    events: List[DetectorEvent],
    n_timesteps: int = N_TIMESTEPS,
) -> np.ndarray:
    """이벤트 리스트 → 이진 시퀀스 (Tier 1 GNN 학습 입력).

    각 감지기: latched (트리거 이후 끝까지 1).

    Returns:
        ``(n_timesteps, n_detectors)`` float32 (0.0 / 1.0).
    """
    binary = np.zeros((n_timesteps, len(events)), dtype=np.float32)
    for det_idx, event in enumerate(events):
        if event.activation_frame is not None:
            binary[event.activation_frame:, det_idx] = 1.0
    return binary


def detector_stats(events: List[DetectorEvent]) -> dict:
    """이벤트 리스트 요약 통계."""
    triggered = [e for e in events if e.is_activated()]
    heat_only = [e for e in events if e.trigger_reason == "heat"]
    smoke_only = [e for e in events if e.trigger_reason == "smoke"]
    both = [e for e in events if e.trigger_reason == "both"]
    never = [e for e in events if e.trigger_reason == "none"]
    times = [e.activation_time_s for e in triggered]
    return {
        "n_total": len(events),
        "n_triggered": len(triggered),
        "n_heat_only": len(heat_only),
        "n_smoke_only": len(smoke_only),
        "n_both": len(both),
        "n_never": len(never),
        "mean_activation_time_s": float(np.mean(times)) if times else None,
        "earliest_activation_s": float(min(times)) if times else None,
        "latest_activation_s": float(max(times)) if times else None,
    }


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=" * 70)
    print("detector_model.py (D-023) self-test")
    print("=" * 70)
    errors: List[str] = []

    # Test 1: 각 케이스 single-cell
    print("\n[Test 1] check_single_cell_trigger")
    # safe
    f, r = check_single_cell_trigger(np.full(31, 20.0), np.full(31, 30.0))
    if (f, r) != (None, "none"):
        errors.append(f"safe: {f},{r}")
    # heat only
    T_h = np.concatenate([np.full(5, 20.0), np.full(26, 80.0)])
    f, r = check_single_cell_trigger(T_h, np.full(31, 30.0))
    if (f, r) != (5, "heat"):
        errors.append(f"heat: {f},{r}")
    # smoke only
    V_s = np.concatenate([np.full(3, 30.0), np.full(28, 5.0)])
    f, r = check_single_cell_trigger(np.full(31, 20.0), V_s)
    if (f, r) != (3, "smoke"):
        errors.append(f"smoke: {f},{r}")
    # both
    T_b = np.concatenate([np.full(8, 20.0), np.full(23, 70.0)])
    V_b = np.concatenate([np.full(8, 30.0), np.full(23, 5.0)])
    f, r = check_single_cell_trigger(T_b, V_b)
    if (f, r) != (8, "both"):
        errors.append(f"both: {f},{r}")
    print(f"  [PASS] 4 cases")

    # Test 2: extract_detector_events
    print("\n[Test 2] extract_detector_events synthetic")
    T_grid = np.full((31, 60, 40, 6), 20.0, dtype=np.float32)
    V_grid = np.full((31, 60, 40, 6), 30.0, dtype=np.float32)
    T_grid[5:, 28:33, 18:23, 2:4] = 800.0
    V_grid[7:, 28:33, 18:23, 2:4] = 2.0
    detectors = [
        ("near_fire", (15.0, 10.0, 1.75)),
        ("far_corner", (2.0, 2.0, 1.75)),
        ("oob", (-5.0, 10.0, 1.5)),
    ]
    events = extract_detector_events(T_grid, V_grid, detectors)
    near = next(e for e in events if e.detector_id == "near_fire")
    if near.trigger_reason != "heat" or near.activation_frame != 5:
        errors.append(f"near: {near.trigger_reason}, {near.activation_frame}")
    oob = next(e for e in events if e.detector_id == "oob")
    if oob.trigger_reason != "none":
        errors.append(f"oob: {oob.trigger_reason}")
    print(f"  [PASS] {len(events)} detectors processed")

    # Test 3: build_binary_sequence
    print("\n[Test 3] build_binary_sequence")
    binary = build_binary_sequence(events)
    if binary.shape != (31, 3):
        errors.append(f"binary shape: {binary.shape}")
    near_idx = 0
    if binary[4, near_idx] != 0 or binary[5, near_idx] != 1 or binary[30, near_idx] != 1:
        errors.append("near latched wrong")
    print(f"  [PASS] shape {binary.shape}, latching verified")

    # Test 4: stats
    print("\n[Test 4] detector_stats")
    s = detector_stats(events)
    if s["n_total"] != 3 or s["n_triggered"] != 1:
        errors.append(f"stats: {s}")
    print(f"  stats: {s}")

    # Test 5: D-024 27 detectors on real first_sim
    print("\n[Test 5] D-024 27 detectors on first_sim (if available)")
    real_dir = Path("data/raw/first_sim")
    if real_dir.is_dir():
        try:
            from src.data_pipeline.fds_extractor import extract_slices
            from src.tier1.detector_positions import get_detector_positions_legacy_format
            slices = extract_slices(real_dir)
            d_pos = get_detector_positions_legacy_format()
            ev_real = extract_detector_events(
                slices["temperature"], slices["visibility"], d_pos,
            )
            s_real = detector_stats(ev_real)
            print(f"  Total: {s_real['n_total']}, triggered: {s_real['n_triggered']}")
            print(f"  by reason: heat={s_real['n_heat_only']} "
                  f"smoke={s_real['n_smoke_only']} both={s_real['n_both']} "
                  f"never={s_real['n_never']}")
            if s_real["mean_activation_time_s"] is not None:
                print(f"  mean trigger time: {s_real['mean_activation_time_s']:.0f}s")
        except Exception as e:
            print(f"  SKIP (real test): {e}")
    else:
        print("  SKIP: data/raw/first_sim/ not found")

    print("\n" + "=" * 70)
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("PASS: detector_model (D-023) verification complete")
