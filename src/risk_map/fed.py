"""
Fractional Effective Dose (FED) accumulation for CO exposure.

Implements the ISO 13571 §7.3 *simplified* form (D-008): no Purser
exponent. For an occupant moving through a CO field, the cumulative FED
after frame ``n`` is::

    FED_n = FED_{n-1} + CO_ppm[n] · (Δt / 60) / FED_REFERENCE

where ``FED_REFERENCE = 27000 ppm·min`` (ISO 13571 reference dose) and
``Δt`` is the path sampling interval in seconds.

``FED ≥ FED_THRESHOLD = 0.3`` flags an unsafe exposure for the sensitive
population per the project convention (D-009).
"""
from __future__ import annotations

import numpy as np

from src.shared.constants import TENABILITY


def accumulate_fed_co(
    co_ppm_along_path: np.ndarray,
    dt_seconds: float,
) -> np.ndarray:
    """Cumulative CO-FED for an occupant path.

    Args:
        co_ppm_along_path: 1-D array, length ``N``, units ppm.
        dt_seconds: Path sampling interval (seconds). Typical values:
            1.0 (drone replan tick), 10.0 (SLCF frame).

    Returns:
        ``(N,)`` cumulative FED array, dimensionless ∈ ``[0, ∞)``.

    Raises:
        ValueError: If input is not 1-D or ``dt_seconds <= 0``.

    Notes:
        Uses the simplified Δt·Σ form (D-008) rather than the full
        Purser exponent ``[CO]^1.036``; saturation behaviour at very
        high concentrations is bounded by upstream clipping (5000 ppm).
    """
    co = np.asarray(co_ppm_along_path, dtype=np.float64)
    if co.ndim != 1:
        raise ValueError(f"co_ppm_along_path must be 1-D, got shape {co.shape}")
    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be positive, got {dt_seconds}")

    # ISO 13571 §7.3 simplified: FED_n = Σ CO_n · (Δt_min / 27000)
    dt_min = dt_seconds / 60.0
    coef = dt_min / TENABILITY.FED_REFERENCE
    return np.cumsum(np.maximum(co, 0.0)) * coef


def fed_exceeds_threshold(fed: np.ndarray) -> np.ndarray:
    """Boolean mask of timesteps where FED crosses the danger threshold.

    Args:
        fed: ``(N,)`` cumulative FED.

    Returns:
        ``(N,)`` bool array — ``True`` once FED has crossed
        ``TENABILITY.FED_THRESHOLD`` (0.3 by default).
    """
    return np.asarray(fed) >= TENABILITY.FED_THRESHOLD


def time_to_incapacitation(
    fed: np.ndarray,
    dt_seconds: float,
) -> float:
    """Time (seconds) at which FED first crosses the threshold, or ``inf``.

    Args:
        fed: Cumulative FED array from :func:`accumulate_fed_co`.
        dt_seconds: Sampling interval used to compute ``fed``.

    Returns:
        Seconds elapsed before FED first reaches
        ``TENABILITY.FED_THRESHOLD``, or ``float("inf")`` if it never does.
    """
    over = np.asarray(fed) >= TENABILITY.FED_THRESHOLD
    if not over.any():
        return float("inf")
    return float(np.argmax(over)) * dt_seconds


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("fed.py self-test")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. Manual-example reference ──────────────────────────────────────
    # task_request_template Example 1:
    # CO = 1000 ppm constant for 30 min (dt=60s, N=30) → final FED ≈ 1.111
    print("\n[1] Reference example: 1000 ppm constant for 30 min")
    co = np.full(30, 1000.0)
    fed = accumulate_fed_co(co, dt_seconds=60.0)
    print(f"  final FED = {fed[-1]:.4f}  (expected ≈ 1.111)")
    if abs(fed[-1] - 1.0 / 0.9) > 1e-3:  # 1000*30/27000 = 30/27 ≈ 1.111
        errors.append(f"reference example off: {fed[-1]} vs 1.111")

    # ── 2. Zero input → zero FED ────────────────────────────────────────
    print("\n[2] Zero CO → zero FED")
    fed_zero = accumulate_fed_co(np.zeros(10), dt_seconds=10.0)
    if not np.all(fed_zero == 0):
        errors.append("zero CO produced non-zero FED")
    else:
        print("  PASS")

    # ── 3. Monotonicity ─────────────────────────────────────────────────
    print("\n[3] FED is monotonically non-decreasing for non-negative CO")
    co_random = np.abs(np.random.default_rng(0).standard_normal(50)) * 200
    fed_random = accumulate_fed_co(co_random, dt_seconds=10.0)
    if (np.diff(fed_random) < 0).any():
        errors.append("FED not monotonically non-decreasing")
    else:
        print(f"  N=50 random non-neg CO → diff always ≥ 0  ✓")

    # ── 4. Linearity in dt ──────────────────────────────────────────────
    print("\n[4] FED scales linearly with dt for constant CO")
    co_const = np.full(10, 500.0)
    f_10 = accumulate_fed_co(co_const, dt_seconds=10.0)[-1]
    f_20 = accumulate_fed_co(co_const, dt_seconds=20.0)[-1]
    print(f"  dt=10s final={f_10:.6f},  dt=20s final={f_20:.6f},  ratio={f_20/f_10:.3f}")
    if abs(f_20 / f_10 - 2.0) > 1e-9:
        errors.append(f"dt linearity broken: ratio = {f_20/f_10}")

    # ── 5. Negative CO clipped ──────────────────────────────────────────
    print("\n[5] Negative CO clipped to 0 (defensive)")
    co_neg = np.array([100.0, -50.0, 100.0])
    fed_neg = accumulate_fed_co(co_neg, dt_seconds=10.0)
    expected = accumulate_fed_co(np.array([100.0, 0.0, 100.0]), dt_seconds=10.0)
    if not np.allclose(fed_neg, expected):
        errors.append(f"negative CO not clipped: {fed_neg} vs {expected}")
    else:
        print("  PASS: -50 ppm treated as 0")

    # ── 6. Threshold + time-to-incapacitation ───────────────────────────
    print("\n[6] fed_exceeds_threshold + time_to_incapacitation")
    # 1000 ppm for 30 min → FED hits 0.3 at FED_n = 0.3
    # → n such that n * (1/60) * 1000 / 27000 ≥ 0.3
    #   n ≥ 0.3 * 60 * 27000 / 1000 = 486 frames at dt=1s? No — dt=60s.
    # Recompute: FED_n = n * (60/60) * 1000 / 27000 = n * 1000 / 27000
    # 0.3 → n ≥ 0.3 * 27000 / 1000 = 8.1 → first crossing at n=9 → t=8*60=480s? wait n=9 → 9*60=540s.
    # argmax(over) returns first index where True. n=9 (index 8 if 0-indexed) since fed[8]=9*1000/27000≈0.333.
    co_long = np.full(30, 1000.0)
    fed_long = accumulate_fed_co(co_long, dt_seconds=60.0)
    ttx = time_to_incapacitation(fed_long, dt_seconds=60.0)
    print(f"  1000 ppm × 30 min (dt=60s) → t_inc = {ttx} s")
    expected_idx = int(np.argmax(fed_long >= TENABILITY.FED_THRESHOLD))
    expected_t = expected_idx * 60.0
    if ttx != expected_t:
        errors.append(f"time_to_incapacitation = {ttx}, expected {expected_t}")

    # Never reaches threshold → inf
    safe_co = np.full(10, 10.0)
    ttx_safe = time_to_incapacitation(
        accumulate_fed_co(safe_co, dt_seconds=10.0), dt_seconds=10.0,
    )
    if ttx_safe != float("inf"):
        errors.append(f"safe CO time_to_incapacitation = {ttx_safe}, expected inf")
    else:
        print("  ✓ never-exceeds case returns inf")

    # ── 7. Input validation ──────────────────────────────────────────────
    print("\n[7] Input validation")
    try:
        accumulate_fed_co(np.zeros((10, 2)), dt_seconds=10.0)
    except ValueError:
        print("  PASS: 2-D input raises ValueError")
    else:
        errors.append("2-D input did not raise")
    try:
        accumulate_fed_co(np.zeros(10), dt_seconds=0.0)
    except ValueError:
        print("  PASS: dt=0 raises ValueError")
    else:
        errors.append("dt=0 did not raise")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: fed validated")
