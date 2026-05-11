"""
Available Safe Egress Time (ASET) map computation.

For each SLCF cell ``(ix, iy, iz)``, ASET is the first time at which the
cell's aggregated danger first crosses ``danger_threshold``. Cells that
never exceed the threshold during the simulation get the sentinel value
``T_END_SECONDS`` (300 s).

See ``docs/risk_indicators.md`` §"ASET" for the spec.
"""
from __future__ import annotations

import numpy as np

from src.shared.constants import DT_SLCF, T_END_SECONDS, TENABILITY


def compute_aset_map(
    risk_grid_time: np.ndarray,
    danger_threshold: float = TENABILITY.AGGREGATE_THRESHOLD,
    dt_seconds: float = DT_SLCF,
    t_max_seconds: float = T_END_SECONDS,
) -> np.ndarray:
    """Per-cell first-exceedance time map.

    Args:
        risk_grid_time: Aggregated danger field, shape ``(T, X, Y, Z)``
            with values in ``[0, 1]``. Frame ``t_idx`` corresponds to
            wall time ``t_idx * dt_seconds``.
        danger_threshold: Threshold above which a cell is considered
            dangerous. Defaults to ``TENABILITY.AGGREGATE_THRESHOLD`` (0.5).
        dt_seconds: Time between frames. Defaults to ``DT_SLCF`` (10 s).
        t_max_seconds: Sentinel value for cells that never exceed the
            threshold. Defaults to ``T_END_SECONDS`` (300 s).

    Returns:
        ``(X, Y, Z)`` ``float32`` array of ASET values in seconds.

    Raises:
        ValueError: If ``risk_grid_time`` is not 4-D.
    """
    arr = np.asarray(risk_grid_time)
    if arr.ndim != 4:
        raise ValueError(
            f"risk_grid_time must be 4-D (T, X, Y, Z), got shape {arr.shape}"
        )
    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be positive, got {dt_seconds}")

    exceeded = arr > danger_threshold  # (T, X, Y, Z) bool
    # ``argmax`` returns the first True position. For all-False columns it
    # returns 0, which we mask out below.
    first_idx = np.argmax(exceeded, axis=0).astype(np.float32)
    never = ~exceeded.any(axis=0)
    aset = first_idx * float(dt_seconds)
    aset[never] = float(t_max_seconds)
    return aset


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("aset.py self-test")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. Single cell crossing at known time ───────────────────────────
    print("\n[1] Single cell crosses 0.5 at frame 5")
    T, X, Y, Z = 31, 2, 2, 2
    risk = np.zeros((T, X, Y, Z), dtype=np.float32)
    risk[5:, 0, 0, 0] = 0.9   # crosses at t=50s
    risk[10:, 1, 1, 1] = 0.6  # crosses at t=100s
    aset = compute_aset_map(risk)
    print(f"  cell (0,0,0): aset = {aset[0,0,0]} s  (expected 50)")
    print(f"  cell (1,1,1): aset = {aset[1,1,1]} s  (expected 100)")
    print(f"  cell (0,1,0): aset = {aset[0,1,0]} s  (expected 300 — never)")
    if aset[0, 0, 0] != 50.0:
        errors.append(f"crossing at idx 5: aset = {aset[0,0,0]} != 50")
    if aset[1, 1, 1] != 100.0:
        errors.append(f"crossing at idx 10: aset = {aset[1,1,1]} != 100")
    if aset[0, 1, 0] != T_END_SECONDS:
        errors.append(f"never-crossing cell: aset = {aset[0,1,0]} != {T_END_SECONDS}")

    # ── 2. Already-dangerous at t=0 → aset = 0 ──────────────────────────
    print("\n[2] Cell dangerous at t=0 → aset=0")
    risk2 = np.zeros((T, X, Y, Z), dtype=np.float32)
    risk2[:, 0, 0, 0] = 0.9
    aset2 = compute_aset_map(risk2)
    if aset2[0, 0, 0] != 0.0:
        errors.append(f"dangerous at t=0: aset = {aset2[0,0,0]} != 0")
    else:
        print("  PASS")

    # ── 3. Custom threshold ─────────────────────────────────────────────
    print("\n[3] Custom threshold")
    risk3 = np.zeros((T, 1, 1, 1), dtype=np.float32)
    risk3[5:, 0, 0, 0] = 0.4  # below default 0.5 but above 0.3
    aset_default = compute_aset_map(risk3)
    aset_03 = compute_aset_map(risk3, danger_threshold=0.3)
    print(f"  thr=0.5 (default): aset={aset_default[0,0,0]}  (expected 300)")
    print(f"  thr=0.3          : aset={aset_03[0,0,0]}  (expected 50)")
    if aset_default[0, 0, 0] != T_END_SECONDS:
        errors.append("default thr should miss 0.4 crossing")
    if aset_03[0, 0, 0] != 50.0:
        errors.append("custom thr 0.3 should catch 0.4 crossing")

    # ── 4. Project-scale shape ──────────────────────────────────────────
    print("\n[4] Project-scale (31, 60, 40, 6)")
    rng = np.random.default_rng(0)
    big = rng.uniform(0.0, 1.0, (31, 60, 40, 6)).astype(np.float32)
    aset_big = compute_aset_map(big)
    if aset_big.shape != (60, 40, 6):
        errors.append(f"big shape: {aset_big.shape}")
    if aset_big.min() < 0.0 or aset_big.max() > T_END_SECONDS:
        errors.append(f"big range: [{aset_big.min()}, {aset_big.max()}]")
    print(f"  shape={aset_big.shape}, "
          f"range=[{aset_big.min():.1f}, {aset_big.max():.1f}] s")

    # ── 5. Input validation ─────────────────────────────────────────────
    print("\n[5] Input validation")
    try:
        compute_aset_map(np.zeros((10, 10)))
    except ValueError:
        print("  PASS: 2-D input raises")
    else:
        errors.append("expected ValueError on 2-D input")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: aset validated")
