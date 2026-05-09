"""
Binary detector-event extraction from FDS slice files.

Implements ``docs/tier1_gnn_design.md`` §3 [방법 B]: simulate ceiling-mounted
heat detectors by sampling the FDS temperature SLCF at each detector's grid
cell and threshold-triggering at ``threshold_celsius``. No physical detector
hardware or FDS ``&DEVC`` line is required — the existing 30-scenario
SLCF data is sufficient.

Also provides :func:`augment_binary_sequence` for the 5× data augmentation
described in the design doc (random fault rates + detection delays).

This module is fully decoupled from the building graph — callers supply a
plain ``[(x, y, z), ...]`` list of detector world positions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.shared.constants import (
    CELL_SIZE_M,
    DT_SECONDS,
    GRID_SHAPE,
    TENABILITY,
    TIME_STEPS,
)


# Half-cell offset that maps world metres to a cell-centred SLCF index.
# Cell (0,0,0) has its centre at world (0.25, 0.25, 0.25). See L-004.
_HALF_CELL_M: float = CELL_SIZE_M / 2.0


def _world_to_cell_clamped(
    xyz: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """Convert a world coordinate (m) to the nearest SLCF cell index.

    Out-of-bounds coordinates are clamped to the nearest valid cell so a
    detector positioned just outside the SLCF region still produces a sensible
    sample (we treat slightly mis-placed detectors as "the closest interior
    cell").

    Args:
        xyz: ``(x, y, z)`` in metres.

    Returns:
        ``(ix, iy, iz)`` integer indices into a ``GRID_SHAPE`` array.
    """
    nx, ny, nz = GRID_SHAPE
    ix = int((xyz[0] - _HALF_CELL_M) / CELL_SIZE_M)
    iy = int((xyz[1] - _HALF_CELL_M) / CELL_SIZE_M)
    iz = int((xyz[2] - _HALF_CELL_M) / CELL_SIZE_M)
    return (
        max(0, min(nx - 1, ix)),
        max(0, min(ny - 1, iy)),
        max(0, min(nz - 1, iz)),
    )


def extract_detector_events_from_slices(
    grid: np.ndarray,
    detector_positions: Sequence[Tuple[float, float, float]],
    threshold_celsius: float = TENABILITY.T_DANGER_C,
) -> Dict[str, object]:
    """Convert an SLCF temperature grid into per-detector binary activation events.

    For each detector, this samples the temperature at the closest SLCF cell
    over all 31 frames and triggers on the first crossing of
    ``threshold_celsius``. Once triggered, the detector remains on (latching
    behaviour mirrors physical heat detectors).

    Args:
        grid: Temperature SLCF, shape ``(TIME_STEPS, nx, ny, nz)`` =
              ``(31, 60, 40, 6)``, units °C.
        detector_positions: List of ``(x, y, z)`` world metres, length
              ``N_detectors``.
        threshold_celsius: Activation threshold in °C. Defaults to the
              ISO 13571 danger temperature
              (:attr:`src.shared.constants.TENABILITY.T_DANGER_C` = 60.0).

    Returns:
        Dict with two keys:

        * ``"binary_sequence"`` — ``np.ndarray`` of shape
          ``(TIME_STEPS, N_detectors)`` and dtype ``float32``.
          ``1.0`` once the detector has triggered (and thereafter), ``0.0``
          before that.
        * ``"activation_times"`` — ``list[float | None]`` of length
          ``N_detectors``. The simulation time (seconds) at which each
          detector first triggered, or ``None`` if it never triggered
          within the simulation window.

    Raises:
        ValueError: If ``grid`` does not have the expected shape, or if
            ``detector_positions`` is empty.
    """
    expected = (TIME_STEPS, *GRID_SHAPE)
    if grid.shape != expected:
        raise ValueError(
            f"grid shape {grid.shape} != expected {expected}"
        )
    if len(detector_positions) == 0:
        raise ValueError("detector_positions must contain at least one detector")

    n_det = len(detector_positions)
    binary_seq = np.zeros((TIME_STEPS, n_det), dtype=np.float32)
    activation_times: List[Optional[float]] = [None] * n_det

    for det_id, pos in enumerate(detector_positions):
        ix, iy, iz = _world_to_cell_clamped(pos)
        cell_temp_series = grid[:, ix, iy, iz]  # shape (TIME_STEPS,)

        # First index where temperature exceeds threshold; -1 if never.
        triggered = np.flatnonzero(cell_temp_series > threshold_celsius)
        if triggered.size > 0:
            t_idx = int(triggered[0])
            activation_times[det_id] = float(t_idx * DT_SECONDS)
            binary_seq[t_idx:, det_id] = 1.0

    return {
        "binary_sequence": binary_seq,
        "activation_times": activation_times,
    }


def augment_binary_sequence(
    binary_seq: np.ndarray,
    fault_rates: Sequence[float] = (0.1, 0.2),
    delay_steps: Sequence[int] = (1, 2),
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Generate augmented copies of a binary detector sequence.

    Mirrors ``docs/tier1_gnn_design.md`` §3 augmentation. Default parameters
    yield 5 sequences (1 original + 2 fault-injected + 2 delayed), so 30
    scenarios → 150 training samples.

    Args:
        binary_seq: Original sequence, shape ``(TIME_STEPS, N_detectors)``,
                    values ∈ {0.0, 1.0}.
        fault_rates: Fraction of detectors to mute (force to 0 across all
                     timesteps) per fault-injection variant.
        delay_steps: Number of timesteps to shift each delay variant by;
                     activations earlier than ``delay`` are clipped.
        rng: Optional ``np.random.Generator`` for reproducibility.

    Returns:
        List of augmented sequences (the first entry is an unmodified copy
        of the input). Length = ``1 + len(fault_rates) + len(delay_steps)``.

    Raises:
        ValueError: If ``binary_seq`` is not 2-D or any rate is outside
            ``[0, 1]``, or any delay is negative.
    """
    if binary_seq.ndim != 2:
        raise ValueError(
            f"binary_seq must be 2-D (T, N_detectors), got shape {binary_seq.shape}"
        )
    if any(r < 0.0 or r > 1.0 for r in fault_rates):
        raise ValueError(f"fault_rates must lie in [0, 1], got {list(fault_rates)}")
    if any(d < 0 for d in delay_steps):
        raise ValueError(f"delay_steps must be non-negative, got {list(delay_steps)}")

    if rng is None:
        rng = np.random.default_rng()

    n_det = binary_seq.shape[1]
    augmented: List[np.ndarray] = [binary_seq.copy()]

    # Fault injection: mute a random subset of detectors entirely.
    for rate in fault_rates:
        noisy = binary_seq.copy()
        drop_mask = rng.random(n_det) < rate
        noisy[:, drop_mask] = 0.0
        augmented.append(noisy)

    # Detection delay: shift the activation pattern forward in time.
    for delay in delay_steps:
        delayed = np.zeros_like(binary_seq)
        if delay < binary_seq.shape[0]:
            delayed[delay:] = binary_seq[: binary_seq.shape[0] - delay]
        augmented.append(delayed)

    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run with ``python -m src.tier1.detector_extractor``).
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("detector_extractor.py self-test")
    print("=" * 60)

    rng = np.random.default_rng(0)
    nx, ny, nz = GRID_SHAPE
    errors: List[str] = []

    # ── Synthetic temperature grid: a fire at world (15, 10, 1.5) that grows
    # outward in time. Closest cell index = (29, 19, 2). ──────────────────────
    fire_xyz = (15.0, 10.0, 1.5)
    fire_ix, fire_iy, fire_iz = _world_to_cell_clamped(fire_xyz)

    grid = np.full((TIME_STEPS, nx, ny, nz), 22.0, dtype=np.float32)  # 22 °C ambient
    xx, yy, zz = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    cell_dist = np.sqrt(
        (xx - fire_ix) ** 2 + (yy - fire_iy) ** 2 + (zz - fire_iz) ** 2
    )
    # Fire ramps up linearly with time and decays with distance.
    for t_idx in range(TIME_STEPS):
        intensity = 1500.0 * (t_idx / (TIME_STEPS - 1))  # peak 1500 °C at t=300s
        grid[t_idx] = 22.0 + intensity * np.exp(-cell_dist / 6.0)

    # Detectors: one at fire, one mid-range (~7 m away), one far (~20 m away).
    detectors = [
        (15.0, 10.0, 2.5),   # near
        (22.0, 10.0, 2.5),   # mid
        (5.0, 2.0, 2.5),     # far
    ]

    # ── Test 1: extraction shape and types ──────────────────────────────────
    print("\n[Test 1] extract_detector_events_from_slices")
    out = extract_detector_events_from_slices(grid, detectors, threshold_celsius=60.0)
    bs = out["binary_sequence"]
    at = out["activation_times"]

    if bs.shape != (TIME_STEPS, 3):
        errors.append(f"binary_sequence shape {bs.shape} != (31, 3)")
    if bs.dtype != np.float32:
        errors.append(f"binary_sequence dtype {bs.dtype} != float32")
    if len(at) != 3:
        errors.append(f"activation_times length {len(at)} != 3")
    print(f"  binary_sequence.shape = {bs.shape}")
    print(f"  activation_times      = {at}")

    # ── Test 2: latching behaviour — once on, stays on ──────────────────────
    print("\n[Test 2] Latching: once triggered, stays on")
    for det_id in range(3):
        seq = bs[:, det_id]
        if at[det_id] is None:
            continue
        first_on = int(at[det_id] / DT_SECONDS)
        # Before activation must be all zero, after must be all one.
        if seq[:first_on].any():
            errors.append(f"detector {det_id}: non-zero before activation")
        if not seq[first_on:].all():
            errors.append(f"detector {det_id}: dropped after activation")
    print("  PASS: latching verified" if not errors else "  FAIL")

    # ── Test 3: ordering — near detector triggers before far ────────────────
    print("\n[Test 3] Activation ordering (near ≤ mid ≤ far)")
    near_t, mid_t, far_t = at[0], at[1], at[2]
    print(f"  near  = {near_t}")
    print(f"  mid   = {mid_t}")
    print(f"  far   = {far_t}")
    test3_errors: list[str] = []
    if near_t is None:
        test3_errors.append("near detector never triggered (threshold/grid mismatch?)")
    elif mid_t is not None and near_t > mid_t:
        test3_errors.append(f"near ({near_t}) should activate before mid ({mid_t})")
    if mid_t is not None and far_t is not None and mid_t > far_t:
        test3_errors.append(f"mid ({mid_t}) should activate before far ({far_t})")
    errors.extend(test3_errors)
    print("  PASS: ordering OK" if not test3_errors else "  FAIL: ordering wrong")

    # ── Test 4: input validation ────────────────────────────────────────────
    print("\n[Test 4] Input validation")
    try:
        extract_detector_events_from_slices(
            np.zeros((10, 60, 40, 6), dtype=np.float32),
            detectors,
        )
    except ValueError:
        print("  PASS: bad grid shape raises ValueError")
    else:
        errors.append("bad grid shape did not raise")

    try:
        extract_detector_events_from_slices(grid, [])
    except ValueError:
        print("  PASS: empty detector list raises ValueError")
    else:
        errors.append("empty detector list did not raise")

    # ── Test 5: augmentation count and shape ────────────────────────────────
    print("\n[Test 5] augment_binary_sequence")
    aug = augment_binary_sequence(bs, rng=rng)
    print(f"  variants generated = {len(aug)} (expect 1 + 2 + 2 = 5)")
    if len(aug) != 5:
        errors.append(f"augmentation count {len(aug)} != 5")
    for i, a in enumerate(aug):
        if a.shape != bs.shape:
            errors.append(f"variant {i} shape {a.shape} != {bs.shape}")

    # Original is unchanged.
    if not np.array_equal(aug[0], bs):
        errors.append("augmentation variant 0 should equal the input")

    # Delay variant: first ``delay`` rows must be zero.
    delay_variant = aug[3]  # 1 + len(fault_rates)=2  → index 3 = first delay
    if delay_variant[0].any():
        errors.append("delay-1 variant should have a zero row at t=0")
    print("  PASS: shapes / counts OK" if not errors else "  FAIL")

    # ── Verdict ─────────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("PASS")
    print("=" * 60)
