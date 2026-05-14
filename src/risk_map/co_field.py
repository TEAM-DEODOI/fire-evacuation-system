"""StaticCOField — raw CO concentration query at arbitrary ``(xyz, t)``.

Companion to :class:`~src.risk_map.risk_map_class.StaticRiskMap`. The risk
map aggregates T/V/CO into a single ``danger ∈ [0, 1]`` value via the
ISO 13571 tenability rules — useful for path planning and dynamic
threat assessment. But the CO-FED accumulator
(:func:`src.risk_map.fed.accumulate_fed_co`) needs the *raw CO ppm*
exposure history along an occupant's trajectory; that is what this
class provides.

Interface mirrors :class:`StaticRiskMap` for symmetry:

* :meth:`query(xyz, t)` — ppm scalar (or array for batch xyz)
* :meth:`from_fds_dir(dir)` — build from FDS slice files
* :meth:`from_npy(path)`    — load from cache
* :meth:`save(path)`        — write cache

The out-of-bounds policy differs from RiskMap intentionally:

* Outside the SLCF region  → **0 ppm** (atmosphere; no smoke).
* Before the first frame    → 0 ppm.
* After the last frame      → value at the last frame (smoke does not
  disappear instantly when our simulation stops).

These are the *safe* defaults for FED accumulation — never punish an
agent for being outside the building or after the recorded horizon
ends; agents that have evacuated already do not accumulate FED anyway.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.interpolate as si

from src.shared.constants import DOMAIN_SIZE_M, GRID_SHAPE
from src.shared.coordinates import cell_centres


_TOL: float = 1e-9


class StaticCOField:
    """RegularGridInterpolator-backed raw CO field over ``(t, x, y, z)``.

    Args:
        co_array: ``(n_frames, 60, 40, 6)`` raw CO concentration (ppm).
        times: ``(n_frames,)`` strictly ascending wall times (s).

    Raises:
        ValueError: If shapes or times are inconsistent.
    """

    def __init__(
        self,
        co_array: np.ndarray,
        times: np.ndarray,
    ) -> None:
        co = np.asarray(co_array, dtype=np.float32)
        ts = np.asarray(times, dtype=np.float64)

        if co.ndim != 4:
            raise ValueError(f"co_array must be 4-D, got shape {co.shape}")
        if co.shape[1:] != GRID_SHAPE:
            raise ValueError(
                f"co_array spatial shape {co.shape[1:]} != {GRID_SHAPE}"
            )
        if ts.shape != (co.shape[0],):
            raise ValueError(
                f"times shape {ts.shape} != (n_frames={co.shape[0]},)"
            )
        if not np.all(np.diff(ts) > 0):
            raise ValueError("times must be strictly increasing")

        x_centres, y_centres, z_centres = cell_centres()
        self.co_ppm = co
        self.times = ts
        self.start_time = float(ts[0])
        self.t_max = float(ts[-1])
        self.x = x_centres
        self.y = y_centres
        self.z = z_centres
        self.interp = si.RegularGridInterpolator(
            (ts, x_centres, y_centres, z_centres),
            co,
            method="linear",
            bounds_error=False,
            fill_value=0.0,  # outside region/time → no CO (atmosphere)
        )

    # ─── Public query ────────────────────────────────────────────────────
    def query(
        self,
        xyz: np.ndarray,
        t: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Return CO concentration (ppm) at ``xyz`` and time ``t``.

        Args:
            xyz: ``(3,)`` single point or ``(M, 3)`` batch in world metres.
            t: Wall time (s). ``None`` defaults to ``t_max``.

        Returns:
            Scalar (or ``(M,)`` array) of CO ppm ≥ 0.
        """
        arr = np.asarray(xyz, dtype=np.float64)
        if t is None:
            t = self.t_max

        # Clamp time to recorded horizon (see policy in module docstring).
        t_clamped = float(np.clip(t, self.start_time, self.t_max))

        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError(
                    f"single-point xyz must have 3 components, got {arr.shape}"
                )
            return self._query_single(arr, t_clamped)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return self._query_batch(arr, t_clamped)
        raise ValueError(
            f"xyz must have shape (3,) or (M, 3), got {arr.shape}"
        )

    # ─── Internals ───────────────────────────────────────────────────────
    def _in_bounds(self, xyz: np.ndarray) -> bool:
        lx, ly, lz = DOMAIN_SIZE_M
        return bool(
            -_TOL <= xyz[0] <= lx + _TOL
            and -_TOL <= xyz[1] <= ly + _TOL
            and -_TOL <= xyz[2] <= lz + _TOL
        )

    def _query_single(self, xyz: np.ndarray, t: float) -> float:
        if not self._in_bounds(xyz):
            return 0.0
        x = float(np.clip(xyz[0], self.x[0], self.x[-1]))
        y = float(np.clip(xyz[1], self.y[0], self.y[-1]))
        z = float(np.clip(xyz[2], self.z[0], self.z[-1]))
        pt = np.array([[t, x, y, z]], dtype=np.float64)
        return max(0.0, float(np.asarray(self.interp(pt)).item()))

    def _query_batch(self, xyz: np.ndarray, t: float) -> np.ndarray:
        lx, ly, lz = DOMAIN_SIZE_M
        in_bounds = (
            (xyz[:, 0] >= -_TOL) & (xyz[:, 0] <= lx + _TOL)
            & (xyz[:, 1] >= -_TOL) & (xyz[:, 1] <= ly + _TOL)
            & (xyz[:, 2] >= -_TOL) & (xyz[:, 2] <= lz + _TOL)
        )
        clamped = np.column_stack([
            np.clip(xyz[:, 0], self.x[0], self.x[-1]),
            np.clip(xyz[:, 1], self.y[0], self.y[-1]),
            np.clip(xyz[:, 2], self.z[0], self.z[-1]),
        ])
        ts = np.full(clamped.shape[0], t)
        pts = np.column_stack([ts, clamped])
        out = np.maximum(self.interp(pts), 0.0)
        out[~in_bounds] = 0.0
        return out.astype(np.float32)

    # ─── Factories ──────────────────────────────────────────────────────
    @classmethod
    def from_fds_dir(cls, fds_dir: Path) -> "StaticCOField":
        """Extract raw CO from an FDS scenario directory.

        Reuses :func:`src.data_pipeline.fds_extractor.extract_slices`,
        which already returns CO in ppm (after the ``× 1e6`` mol/mol →
        ppm conversion in the extractor pipeline).

        Args:
            fds_dir: FDS scenario directory.

        Returns:
            :class:`StaticCOField` populated from raw CO slice data.
        """
        from src.data_pipeline.fds_extractor import extract_slices
        slices = extract_slices(fds_dir)
        return cls(co_array=slices["co"], times=slices["times"])

    @classmethod
    def from_npy(cls, path: Path) -> "StaticCOField":
        """Load a previously-saved ``.npz`` (``co_ppm`` + ``times``)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CO npz not found: {path}")
        with np.load(path) as f:
            return cls(co_array=f["co_ppm"], times=f["times"])

    def save(self, path: Path) -> None:
        """Persist the CO grid + times to ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, co_ppm=self.co_ppm, times=self.times)


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import tempfile

    print("=" * 60)
    print("co_field.py self-test")
    print("=" * 60)

    errors: list[str] = []
    nx, ny, nz = GRID_SHAPE
    times = np.arange(0.0, 310.0, 10.0)  # 31 frames

    # Synthetic CO grid: 0 everywhere before t=50s, then 2000 ppm in a hot box.
    co = np.zeros((len(times), nx, ny, nz), dtype=np.float32)
    co[5:, 30:35, 20:25, 2:4] = 2000.0
    fld = StaticCOField(co_array=co, times=times)

    # ── 1. Inside hot box ────────────────────────────────────────────────
    print("\n[1] Point inside hot box at t=100s")
    v = fld.query(np.array([15.25, 10.25, 1.25]), t=100.0)
    print(f"  query = {v:.1f} ppm  (expected ~2000)")
    if abs(v - 2000.0) > 1.0:
        errors.append(f"hot-box query = {v}, expected 2000")

    # ── 2. Outside SLCF → 0 ppm (atmosphere) ────────────────────────────
    print("\n[2] OOB returns 0 ppm")
    v_oob = fld.query(np.array([-1.0, 10.0, 1.5]), t=100.0)
    if v_oob != 0.0:
        errors.append(f"OOB = {v_oob} ppm, expected 0")
    else:
        print("  PASS")

    # ── 3. t > t_max clamps to last frame ───────────────────────────────
    print("\n[3] t > t_max clamps")
    v_late = fld.query(np.array([15.25, 10.25, 1.25]), t=999.0)
    v_max = fld.query(np.array([15.25, 10.25, 1.25]), t=fld.t_max)
    if abs(v_late - v_max) > 1e-3:
        errors.append(f"t>t_max not clamped: {v_late} vs {v_max}")
    else:
        print(f"  PASS: t=999 ({v_late}) == t=t_max ({v_max})")

    # ── 4. Batch query ───────────────────────────────────────────────────
    print("\n[4] Batched query")
    pts = np.array([
        [15.25, 10.25, 1.25],      # hot
        [5.0, 5.0, 1.0],           # cold
        [-1.0, 10.0, 1.5],         # OOB
    ])
    out = fld.query(pts, t=100.0)
    print(f"  result = {out}")
    if not (out.shape == (3,) and abs(out[0] - 2000) < 1
            and out[1] == 0 and out[2] == 0):
        errors.append(f"batch = {out}")

    # ── 5. save / from_npy ───────────────────────────────────────────────
    print("\n[5] save / from_npy round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "co.npz"
        fld.save(p)
        f2 = StaticCOField.from_npy(p)
        a = fld.query(np.array([15.25, 10.25, 1.25]), t=100.0)
        b = f2.query(np.array([15.25, 10.25, 1.25]), t=100.0)
        if abs(a - b) > 1e-6:
            errors.append(f"round-trip: {a} vs {b}")
        else:
            print(f"  PASS: {a} == {b}")

    # ── 6. Negative CO clamped to 0 ──────────────────────────────────────
    print("\n[6] Negative CO from interp clamped to 0")
    co_neg = np.full((2, nx, ny, nz), -100.0, dtype=np.float32)
    fld_neg = StaticCOField(co_array=co_neg, times=np.array([0.0, 10.0]))
    v_neg = fld_neg.query(np.array([15.0, 10.0, 1.5]), t=5.0)
    if v_neg != 0.0:
        errors.append(f"negative not clamped: {v_neg}")
    else:
        print("  PASS")

    # ── 7. Input validation ──────────────────────────────────────────────
    print("\n[7] Input validation")
    try:
        StaticCOField(co_array=np.zeros((5, 10, 10, 10)), times=times[:5])
    except ValueError:
        print("  PASS: bad spatial shape -> ValueError")
    else:
        errors.append("bad spatial shape did not raise")
    try:
        fld.query(np.array([1.0, 2.0]))
    except ValueError:
        print("  PASS: (2,) xyz -> ValueError")
    else:
        errors.append("bad xyz shape did not raise")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: StaticCOField validated")
