"""
Per-cell instantaneous danger scores from temperature / visibility / CO.

All thresholds are sourced from :data:`src.shared.constants.TENABILITY`
(ISO 13571:2012 + SFPE Handbook 5th ed. Ch. 63). See
``docs/risk_indicators.md`` §"Thresholds" and §"Aggregated Danger Score"
for the derivation.

Outputs are scalars in ``[0, 1]`` where ``1.0`` = maximally dangerous.
The functions are pure NumPy — they accept any-shape input and return
the same shape.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from src.shared.constants import TENABILITY


def compute_danger_temperature(t_celsius: np.ndarray) -> np.ndarray:
    """Per-cell temperature danger score.

    Formula: ``clip((T - T_SAFE) / (T_DANGER - T_SAFE), 0, 1)``.
    Defaults map ``30 °C → 0.0`` and ``60 °C → 1.0`` (saturating).
    """
    t = np.asarray(t_celsius, dtype=np.float64)
    span = TENABILITY.T_DANGER_C - TENABILITY.T_SAFE_C
    return np.clip((t - TENABILITY.T_SAFE_C) / span, 0.0, 1.0)


def compute_danger_visibility(v_metres: np.ndarray) -> np.ndarray:
    """Per-cell visibility danger score (inverse mapping).

    Formula: ``clip((V_SAFE - V) / (V_SAFE - V_DANGER), 0, 1)``.
    Defaults map ``10 m → 0.0`` and ``3 m → 1.0``.
    """
    v = np.asarray(v_metres, dtype=np.float64)
    span = TENABILITY.V_SAFE_M - TENABILITY.V_DANGER_M
    return np.clip((TENABILITY.V_SAFE_M - v) / span, 0.0, 1.0)


def compute_danger_co(co_ppm: np.ndarray) -> np.ndarray:
    """Per-cell instantaneous CO danger score.

    Formula: ``clip((CO - CO_SAFE) / (CO_DANGER - CO_SAFE), 0, 1)``.
    Defaults map ``100 ppm → 0.0`` and ``1400 ppm → 1.0``. Cumulative
    exposure is handled separately by
    :func:`src.risk_map.fed.accumulate_fed_co`.
    """
    co = np.asarray(co_ppm, dtype=np.float64)
    span = TENABILITY.CO_DANGER_PPM - TENABILITY.CO_SAFE_PPM
    return np.clip((co - TENABILITY.CO_SAFE_PPM) / span, 0.0, 1.0)


def compute_total_danger(
    t_celsius: np.ndarray,
    v_metres: np.ndarray,
    co_ppm: np.ndarray,
    weights: Tuple[float, float, float] = (
        TENABILITY.WEIGHT_T,
        TENABILITY.WEIGHT_V,
        TENABILITY.WEIGHT_CO,
    ),
) -> np.ndarray:
    """Aggregate the three indicators into a single danger field.

    Formula: ``clip(w_T·d_T + w_V·d_V + w_CO·d_CO, 0, 1)``.

    Args:
        t_celsius, v_metres, co_ppm: Raw fields in physical units.
        weights: ``(w_T, w_V, w_CO)``. Defaults to ``TENABILITY`` weights
            (0.4 / 0.4 / 0.2).
    """
    if len(weights) != 3:
        raise ValueError(f"weights must have 3 entries, got {len(weights)}")
    w_t, w_v, w_co = weights
    d_t = compute_danger_temperature(t_celsius)
    d_v = compute_danger_visibility(v_metres)
    d_co = compute_danger_co(co_ppm)
    return np.clip(w_t * d_t + w_v * d_v + w_co * d_co, 0.0, 1.0)


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("tenability.py self-test")
    print("=" * 60)

    errors: list[str] = []
    atol = 1e-9

    print("\n[1] Temperature danger thresholds")
    for T, expected in [(20.0, 0.0), (30.0, 0.0), (45.0, 0.5), (60.0, 1.0), (1200.0, 1.0)]:
        d = float(compute_danger_temperature(np.array([T]))[0])
        print(f"  T={T:>6.1f} °C → d={d:.3f}  (expected {expected})")
        if abs(d - expected) > atol:
            errors.append(f"T danger {T} → {d} != {expected}")

    print("\n[2] Visibility danger thresholds")
    for V, expected in [(30.0, 0.0), (10.0, 0.0), (6.5, 0.5), (3.0, 1.0), (0.0, 1.0)]:
        d = float(compute_danger_visibility(np.array([V]))[0])
        print(f"  V={V:>5.2f} m   → d={d:.3f}  (expected {expected})")
        if abs(d - expected) > atol:
            errors.append(f"V danger {V} → {d} != {expected}")

    print("\n[3] CO instantaneous danger thresholds")
    for CO, expected in [(0.0, 0.0), (100.0, 0.0), (750.0, 0.5), (1400.0, 1.0), (5000.0, 1.0)]:
        d = float(compute_danger_co(np.array([CO]))[0])
        print(f"  CO={CO:>6.1f} ppm → d={d:.3f}  (expected {expected})")
        if abs(d - expected) > atol:
            errors.append(f"CO danger {CO} → {d} != {expected}")

    print("\n[4] compute_total_danger")
    d = float(compute_total_danger(
        np.array([20.0]), np.array([30.0]), np.array([50.0]),
    )[0])
    print(f"  all-safe (20°C, 30m, 50ppm) → {d:.4f}  (expected 0)")
    if abs(d) > atol:
        errors.append(f"all-safe aggregate = {d}, expected 0")
    d = float(compute_total_danger(
        np.array([100.0]), np.array([0.0]), np.array([2000.0]),
    )[0])
    print(f"  all-danger (100°C, 0m, 2000ppm) → {d:.4f}  (expected 1)")
    if abs(d - 1.0) > atol:
        errors.append(f"all-danger aggregate = {d}, expected 1")
    d = float(compute_total_danger(
        np.array([60.0]), np.array([30.0]), np.array([50.0]),
        weights=(1.0, 0.0, 0.0),
    )[0])
    print(f"  T-only weight, T=60 → {d:.4f}  (expected 1)")
    if abs(d - 1.0) > atol:
        errors.append(f"T-only aggregate = {d}, expected 1")

    try:
        compute_total_danger(np.array([0.0]), np.array([0.0]), np.array([0.0]),
                             weights=(1.0, 0.0))
    except ValueError:
        print("  PASS: weights length validation raises")
    else:
        errors.append("expected ValueError on len(weights) != 3")

    print("\n[5] Shape preservation across (31, 60, 40, 6)")
    rng = np.random.default_rng(0)
    T_field = rng.uniform(20.0, 800.0, (31, 60, 40, 6))
    V_field = rng.uniform(0.0, 30.0, (31, 60, 40, 6))
    CO_field = rng.uniform(0.0, 5000.0, (31, 60, 40, 6))
    d_total = compute_total_danger(T_field, V_field, CO_field)
    print(f"  shape={d_total.shape}, range=[{d_total.min():.3f}, {d_total.max():.3f}]")
    if d_total.shape != (31, 60, 40, 6):
        errors.append(f"shape mismatch: {d_total.shape}")
    if d_total.min() < 0.0 or d_total.max() > 1.0:
        errors.append("aggregate out of [0, 1]")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: tenability validated")
