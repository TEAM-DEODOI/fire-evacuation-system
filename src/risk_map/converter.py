"""
Bridge from model output → :class:`StaticRiskMap`.

The model emits ``(T, 3, 60, 40, 6)`` *normalised* fields (channels:
``[T_norm, V_norm, CO_norm]``, all in ``[0, 1]``). This module:

1. Denormalises each channel back to physical units (°C, m, ppm) using
   the inverse functions in :mod:`src.shared.normalization`.
2. Applies the tenability functions
   (:mod:`src.risk_map.tenability`) per frame to obtain a
   ``(T, 60, 40, 6)`` aggregated danger field.
3. Wraps the danger field in a :class:`~src.risk_map.risk_map_class.StaticRiskMap`
   for the path-planning layer.

The same conversion works for FNO predictions and ConvLSTM outputs —
nothing here is model-specific.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from src.risk_map.risk_map_class import StaticRiskMap
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import GRID_SHAPE, N_OUTPUT_CHANNELS, TENABILITY
from src.shared.normalization import (
    denormalize_co,
    denormalize_temperature,
    denormalize_visibility,
)


def prediction_to_danger(
    model_output: np.ndarray,
    times: np.ndarray,
    weights: Tuple[float, float, float] = (
        TENABILITY.WEIGHT_T,
        TENABILITY.WEIGHT_V,
        TENABILITY.WEIGHT_CO,
    ),
) -> np.ndarray:
    """Denormalise a model output sequence and aggregate to a danger field.

    Args:
        model_output: ``(T, 3, 60, 40, 6)`` normalised prediction sequence
            (channels: ``[T_norm, V_norm, CO_norm]``).
        times: ``(T,)`` wall times (seconds). Used only to validate the
            leading dimension; not consumed by tenability.
        weights: ``(w_T, w_V, w_CO)`` aggregation weights.

    Returns:
        ``(T, 60, 40, 6)`` ``float32`` danger field in ``[0, 1]``.

    Raises:
        ValueError: If shapes disagree with the project conventions.
    """
    arr = np.asarray(model_output)
    if arr.ndim != 5:
        raise ValueError(
            f"model_output must be 5-D (T, C, X, Y, Z), got {arr.shape}"
        )
    if arr.shape[1] != N_OUTPUT_CHANNELS:
        raise ValueError(
            f"model_output channels {arr.shape[1]} != {N_OUTPUT_CHANNELS}"
        )
    if arr.shape[2:] != GRID_SHAPE:
        raise ValueError(
            f"spatial shape {arr.shape[2:]} != {GRID_SHAPE}"
        )
    t = np.asarray(times)
    if t.shape != (arr.shape[0],):
        raise ValueError(
            f"times shape {t.shape} != (T={arr.shape[0]},)"
        )

    # Denormalise per channel.
    t_celsius = denormalize_temperature(arr[:, 0])
    v_metres = denormalize_visibility(arr[:, 1])
    co_ppm = denormalize_co(arr[:, 2])

    return compute_total_danger(
        t_celsius, v_metres, co_ppm, weights=weights,
    ).astype(np.float32)


def build_static_risk_map(
    model_output: np.ndarray,
    times: np.ndarray,
    weights: Tuple[float, float, float] = (
        TENABILITY.WEIGHT_T,
        TENABILITY.WEIGHT_V,
        TENABILITY.WEIGHT_CO,
    ),
) -> StaticRiskMap:
    """Build a :class:`StaticRiskMap` from a model output sequence.

    Args:
        model_output: ``(T, 3, 60, 40, 6)`` normalised predictions.
        times: ``(T,)`` wall times.
        weights: Aggregation weights forwarded to :func:`prediction_to_danger`.

    Returns:
        Ready-to-query :class:`StaticRiskMap`.
    """
    danger = prediction_to_danger(model_output, times, weights)
    return StaticRiskMap(danger_array=danger, times=times)


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("converter.py self-test")
    print("=" * 60)

    errors: list[str] = []
    rng = np.random.default_rng(0)
    nx, ny, nz = GRID_SHAPE
    T = 31
    times = np.arange(0.0, T * 10.0, 10.0)

    # ── 1. prediction_to_danger shape + range ────────────────────────────
    print("\n[1] prediction_to_danger on random model output")
    model_out = rng.uniform(0.0, 1.0, (T, 3, nx, ny, nz)).astype(np.float32)
    danger = prediction_to_danger(model_out, times)
    print(f"  shape={danger.shape}, range=[{danger.min():.3f}, {danger.max():.3f}]")
    if danger.shape != (T, nx, ny, nz):
        errors.append(f"shape {danger.shape}")
    if danger.min() < 0.0 or danger.max() > 1.0:
        errors.append("out of [0, 1]")
    if danger.dtype != np.float32:
        errors.append(f"dtype {danger.dtype}")

    # ── 2. All-zero normalised output → low danger ─────────────────────
    print("\n[2] All-zero normalised output → low danger")
    # Channel 0 (T_norm=0) → T_raw=20°C → below safe → d=0
    # Channel 1 (V_norm=0) → V_raw=30m → above safe → d=0
    # Channel 2 (CO_norm=0) → CO_raw=0 → below safe → d=0
    # Total danger should be 0.
    zero_out = np.zeros((T, 3, nx, ny, nz), dtype=np.float32)
    d_zero = prediction_to_danger(zero_out, times)
    print(f"  range=[{d_zero.min():.6f}, {d_zero.max():.6f}]")
    if d_zero.max() > 1e-6:
        errors.append(f"zero-norm output should give zero danger, got {d_zero.max()}")

    # ── 3. All-one normalised output → high danger ─────────────────────
    print("\n[3] All-one normalised output → high danger")
    one_out = np.ones((T, 3, nx, ny, nz), dtype=np.float32)
    d_one = prediction_to_danger(one_out, times)
    print(f"  range=[{d_one.min():.4f}, {d_one.max():.4f}]")
    if d_one.min() < 0.99:
        errors.append(f"one-norm output should give max danger, got {d_one.min()}")

    # ── 4. build_static_risk_map → query smoke test ────────────────────
    print("\n[4] build_static_risk_map → query")
    rm = build_static_risk_map(model_out, times)
    v = rm.query(np.array([15.25, 10.25, 1.25]), t=100.0)
    print(f"  query((15.25, 10.25, 1.25), t=100) = {v:.4f}")
    if not (0.0 <= v <= 1.0):
        errors.append(f"converter→rm.query out of [0, 1]: {v}")

    # OOB still 1.0
    v_oob = rm.query(np.array([-5.0, 10.0, 1.5]), t=100.0)
    if v_oob != 1.0:
        errors.append(f"OOB query = {v_oob}")
    else:
        print("  ✓ OOB still returns 1.0")

    # ── 5. Input validation ────────────────────────────────────────────
    print("\n[5] Input validation")
    try:
        prediction_to_danger(np.zeros((T, 4, nx, ny, nz)), times)
    except ValueError:
        print("  PASS: wrong channel count raises")
    else:
        errors.append("expected ValueError on wrong channels")
    try:
        prediction_to_danger(np.zeros((10, 3, nx, ny, nz)), times)
    except ValueError:
        print("  PASS: T mismatch raises")
    else:
        errors.append("expected ValueError on T mismatch")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: converter validated")
