"""
Bidirectional normalisation for fire field variables.

All inputs/outputs are normalised to ``[0, 1]`` where higher values
indicate greater danger. **Visibility uses an INVERSE mapping** so that
"low visibility → high danger" matches the convention used by the other
two channels.

Normalisation formulas (interface_contracts.md §1.1)::

    Temperature : (T - 20) / 1180             clipped to [0, 1]
    Visibility  : 1 - clip(V / 30, 0, 1)      INVERSE
    CO          : log1p(CO) / log1p(5000)     clipped to [0, 1]
    Time enc.   : t / 300                     linear (NOT sin — see D-???)

Each function accepts **either** a NumPy ``ndarray`` or a PyTorch ``Tensor``
and returns the same library (preserving dtype and device).
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover — torch is a hard dependency in production
    torch = None  # type: ignore[assignment]

from src.shared.constants import (
    CO_NORM_MAX_PPM,
    T_END_SECONDS,
    T_NORM_MIN_C,
    T_NORM_RANGE_C,
    T_MIN_SECONDS,
    V_NORM_MAX_M,
)

# ``ArrayLike`` covers both libraries. We avoid ``Union[np.ndarray, "torch.Tensor"]``
# at module level so the type alias survives when torch is not installed.
ArrayLike = Union[np.ndarray, "torch.Tensor"]  # type: ignore[name-defined]


# Pre-computed scalars
_LOG_CO_MAX_F: float = math.log1p(CO_NORM_MAX_PPM)


def _is_torch(x: object) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


# ─── Temperature ────────────────────────────────────────────────────────────
def normalize_temperature(t_celsius: ArrayLike) -> ArrayLike:
    """Normalise temperature (°C) → [0, 1].

    Formula: ``clip((T - T_NORM_MIN_C) / T_NORM_RANGE_C, 0, 1)``.
    Maps 20 °C → 0.0, 1200 °C → 1.0. Values outside the window are clipped.

    Accepts and returns numpy arrays *or* torch tensors of any shape.
    """
    if _is_torch(t_celsius):
        return torch.clamp(  # type: ignore[union-attr]
            (t_celsius - T_NORM_MIN_C) / T_NORM_RANGE_C, 0.0, 1.0
        )
    arr = np.asarray(t_celsius)
    return np.clip((arr - T_NORM_MIN_C) / T_NORM_RANGE_C, 0.0, 1.0)


def denormalize_temperature(t_norm: ArrayLike) -> ArrayLike:
    """Inverse of :func:`normalize_temperature`.

    Formula: ``t_norm * T_NORM_RANGE_C + T_NORM_MIN_C``.
    """
    return t_norm * T_NORM_RANGE_C + T_NORM_MIN_C


# ─── Visibility (INVERSE) ───────────────────────────────────────────────────
def normalize_visibility(v_metres: ArrayLike) -> ArrayLike:
    """Normalise visibility (m) → [0, 1] with **inverse** mapping.

    Formula: ``1 - clip(V / V_NORM_MAX_M, 0, 1)``.
    High raw visibility (safe) → low normalised value; 0 m → 1.0.

    Negative inputs are treated as 0 (worst-case visibility).
    """
    if _is_torch(v_metres):
        v = torch.clamp(v_metres, min=0.0)  # type: ignore[union-attr]
        return 1.0 - torch.clamp(v / V_NORM_MAX_M, 0.0, 1.0)
    arr = np.asarray(v_metres)
    v = np.maximum(arr, 0.0)
    return 1.0 - np.clip(v / V_NORM_MAX_M, 0.0, 1.0)


def denormalize_visibility(v_norm: ArrayLike) -> ArrayLike:
    """Inverse of :func:`normalize_visibility`.

    Formula: ``(1 - v_norm) * V_NORM_MAX_M``.
    """
    return (1.0 - v_norm) * V_NORM_MAX_M


# ─── CO ─────────────────────────────────────────────────────────────────────
def normalize_co(co_ppm: ArrayLike) -> ArrayLike:
    """Normalise CO concentration (ppm) → [0, 1] on a log scale.

    Formula: ``log1p(CO) / log1p(CO_NORM_MAX_PPM)``, clipped to [0, 1].
    Log scale compresses the long tail of high concentrations.
    """
    if _is_torch(co_ppm):
        co = torch.clamp(co_ppm, min=0.0)  # type: ignore[union-attr]
        return torch.clamp(torch.log1p(co) / _LOG_CO_MAX_F, 0.0, 1.0)
    arr = np.asarray(co_ppm)
    co = np.maximum(arr, 0.0)
    return np.clip(np.log1p(co) / _LOG_CO_MAX_F, 0.0, 1.0)


def denormalize_co(co_norm: ArrayLike) -> ArrayLike:
    """Inverse of :func:`normalize_co`.

    Formula: ``expm1(co_norm * log1p(CO_NORM_MAX_PPM))``.
    Input is clipped to [0, 1] first to avoid ``expm1`` overflow.
    """
    if _is_torch(co_norm):
        clipped = torch.clamp(co_norm, 0.0, 1.0)  # type: ignore[union-attr]
        return torch.expm1(clipped * _LOG_CO_MAX_F)
    arr = np.asarray(co_norm)
    clipped = np.clip(arr, 0.0, 1.0)
    return np.expm1(clipped * _LOG_CO_MAX_F)


# ─── Time encoding (linear) ────────────────────────────────────────────────
def compute_time_encoding(t_seconds: float) -> float:
    """Linear time encoding: ``t / T_END_SECONDS``.

    Returns a scalar in [0, 1]. Values outside [0, T_END] are clipped.

    Notes:
        Linear rather than ``sin(2π · t / T_END)`` — see
        ``docs/interface_contracts.md`` §1.1.
    """
    return float(
        max(0.0, min(1.0, (float(t_seconds) - T_MIN_SECONDS) / T_END_SECONDS))
    )


def decode_time_encoding(enc: float) -> float:
    """Inverse of :func:`compute_time_encoding`: ``enc * T_END_SECONDS``."""
    return float(enc) * T_END_SECONDS + T_MIN_SECONDS


# ─── Self-test (run with ``python -m src.shared.normalization``) ───────────
if __name__ == "__main__":
    print("=" * 60)
    print("normalization.py self-test")
    print("=" * 60)

    errors: list[str] = []
    rtol = 1e-9
    atol = 1e-9

    # ── 1. Round-trip identity (numpy) ─────────────────────────────────────
    print("\n[1] Round-trip identity (numpy)")
    test_T = np.array([20.0, 100.0, 600.0, 1200.0])
    rec_T = denormalize_temperature(normalize_temperature(test_T))
    if not np.allclose(rec_T, test_T, atol=atol):
        errors.append(f"T round-trip drift: {rec_T} vs {test_T}")
    test_V = np.array([0.0, 3.0, 10.0, 30.0])
    rec_V = denormalize_visibility(normalize_visibility(test_V))
    if not np.allclose(rec_V, test_V, atol=atol):
        errors.append(f"V round-trip drift: {rec_V} vs {test_V}")
    test_CO = np.array([0.0, 100.0, 1000.0, 5000.0])
    rec_CO = denormalize_co(normalize_co(test_CO))
    if not np.allclose(rec_CO, test_CO, atol=atol):
        errors.append(f"CO round-trip drift: {rec_CO} vs {test_CO}")
    print(f"  T: {test_T} → norm → denorm → {rec_T}")

    # ── 2. Boundary values ─────────────────────────────────────────────────
    print("\n[2] Boundary values")
    if abs(float(normalize_temperature(np.array([20.0]))[0])) > atol:
        errors.append("normalize_temperature(20) != 0")
    if abs(float(normalize_temperature(np.array([1200.0]))[0]) - 1.0) > atol:
        errors.append("normalize_temperature(1200) != 1")
    if abs(float(normalize_visibility(np.array([30.0]))[0])) > atol:
        errors.append("normalize_visibility(30) != 0  (high V → safe → low norm)")
    if abs(float(normalize_visibility(np.array([0.0]))[0]) - 1.0) > atol:
        errors.append("normalize_visibility(0) != 1   (zero V → danger → high norm)")
    if abs(float(normalize_co(np.array([0.0]))[0])) > atol:
        errors.append("normalize_co(0) != 0")
    if abs(float(normalize_co(np.array([CO_NORM_MAX_PPM]))[0]) - 1.0) > atol:
        errors.append("normalize_co(5000) != 1")

    # ── 3. Clipping ────────────────────────────────────────────────────────
    print("\n[3] Clipping out-of-range inputs")
    if float(normalize_temperature(np.array([-100.0]))[0]) != 0.0:
        errors.append("T below ambient should clip to 0")
    if float(normalize_temperature(np.array([5000.0]))[0]) != 1.0:
        errors.append("T above max should clip to 1")
    if float(normalize_visibility(np.array([100.0]))[0]) != 0.0:
        errors.append("V above 30 should clip to 0 (safe)")
    if float(normalize_visibility(np.array([-5.0]))[0]) != 1.0:
        errors.append("V negative should be 1 (dangerous)")
    if float(normalize_co(np.array([-100.0]))[0]) != 0.0:
        errors.append("CO negative should clip to 0")

    # ── 4. Torch / numpy parity ────────────────────────────────────────────
    print("\n[4] Numpy ↔ torch parity")
    if torch is None:
        print("  SKIP: torch not installed")
    else:
        vals = [20.0, 100.0, 600.0, 1200.0]
        np_arr = np.array(vals)
        t_arr = torch.tensor(vals, dtype=torch.float64)
        np_out = normalize_temperature(np_arr)
        t_out = normalize_temperature(t_arr)
        if not _is_torch(t_out):
            errors.append("torch input did not produce torch output (T)")
        elif not np.allclose(np_out, t_out.cpu().numpy(), atol=atol):
            errors.append(f"T parity: numpy={np_out} torch={t_out.cpu().numpy()}")

        np_out_v = normalize_visibility(np.array([0.0, 3.0, 30.0]))
        t_out_v = normalize_visibility(torch.tensor([0.0, 3.0, 30.0], dtype=torch.float64))
        if not _is_torch(t_out_v):
            errors.append("torch input did not produce torch output (V)")
        elif not np.allclose(np_out_v, t_out_v.cpu().numpy(), atol=atol):
            errors.append("V parity numpy vs torch differ")

        np_out_co = normalize_co(np.array([0.0, 1000.0, 5000.0]))
        t_out_co = normalize_co(torch.tensor([0.0, 1000.0, 5000.0], dtype=torch.float64))
        if not _is_torch(t_out_co):
            errors.append("torch input did not produce torch output (CO)")
        elif not np.allclose(np_out_co, t_out_co.cpu().numpy(), atol=atol):
            errors.append("CO parity numpy vs torch differ")

    # ── 5. Time encoding ───────────────────────────────────────────────────
    print("\n[5] Time encoding")
    if abs(compute_time_encoding(0.0)) > atol:
        errors.append(f"compute_time_encoding(0) = {compute_time_encoding(0.0)}, expected 0")
    if abs(compute_time_encoding(150.0) - 0.5) > atol:
        errors.append(f"compute_time_encoding(150) = {compute_time_encoding(150.0)}, expected 0.5")
    if abs(compute_time_encoding(300.0) - 1.0) > atol:
        errors.append(f"compute_time_encoding(300) = {compute_time_encoding(300.0)}, expected 1.0")
    if abs(compute_time_encoding(-50.0)) > atol:
        errors.append("negative time should clip to 0")
    if abs(compute_time_encoding(500.0) - 1.0) > atol:
        errors.append("time beyond T_END should clip to 1")
    if abs(decode_time_encoding(0.0)) > atol:
        errors.append("decode_time_encoding(0) != 0")
    if abs(decode_time_encoding(0.5) - 150.0) > atol:
        errors.append("decode_time_encoding(0.5) != 150")
    if abs(decode_time_encoding(1.0) - 300.0) > atol:
        errors.append("decode_time_encoding(1.0) != 300")

    # ── 6. Monotonicity ────────────────────────────────────────────────────
    print("\n[6] Monotonicity")
    t_grid = np.linspace(20.0, 1200.0, 100)
    n_t = normalize_temperature(t_grid)
    if (np.diff(n_t) < -atol).any():
        errors.append("temperature normalisation not non-decreasing")
    v_grid = np.linspace(0.0, 30.0, 100)
    n_v = normalize_visibility(v_grid)
    if (np.diff(n_v) > atol).any():
        errors.append("visibility normalisation not non-increasing (inverse mapping)")
    co_grid = np.linspace(0.0, 5000.0, 100)
    n_co = normalize_co(co_grid)
    if (np.diff(n_co) < -atol).any():
        errors.append("CO normalisation not non-decreasing")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: All normalizations validated")
