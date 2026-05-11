"""
Abstract :class:`RiskMap` interface + ``StaticRiskMap`` implementation.

The :class:`RiskMap` ABC is the boundary contract between fire prediction
(FDS ground truth, FNO inference, dynamic predictive) and path planning.
See ``docs/interface_contracts.md`` §2.

:class:`StaticRiskMap` is the immutable, pre-computed concrete map. It
wraps a ``(T, 60, 40, 6)`` danger grid and a ``(T,)`` times array into
a ``scipy.interpolate.RegularGridInterpolator`` so ``query(xyz, t)``
returns a smoothly-interpolated danger scalar at any world coordinate
and simulation time.

Out-of-bounds rules (per the interface contract):
* ``xyz`` outside ``[0, 30] × [0, 20] × [0, 3]`` → ``1.0`` (max danger).
* ``t > t_max`` → ``1.0`` (the future is unknown → conservatively unsafe).
* ``t < start_time`` → value at ``start_time`` (clamped).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.interpolate as si

from src.shared.constants import DOMAIN_SIZE_M, DT_SLCF, GRID_SHAPE, TENABILITY
from src.shared.coordinates import cell_centres


class RiskMap(ABC):
    """Abstract danger-field representation consumed by path planning.

    Implementations must honour:
        * ``query`` returns a scalar in ``[0, 1]``;
        * out-of-bounds coordinates return ``1.0`` (max danger);
        * times beyond the available horizon return ``1.0``.
    """

    @abstractmethod
    def query(
        self,
        xyz: np.ndarray,
        t: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Return danger at world coordinate(s) ``xyz`` at time ``t``."""
        raise NotImplementedError


# Tiny tolerance for bounds comparisons so a query exactly on ``DOMAIN_SIZE_M``
# still counts as in-bounds.
_TOL: float = 1e-9


class StaticRiskMap(RiskMap):
    """RiskMap backed by a pre-computed ``(T, X, Y, Z)`` danger array.

    Args:
        danger_array: ``(n_frames, 60, 40, 6)``, values in ``[0, 1]``.
        times: ``(n_frames,)`` ascending wall times in seconds.

    Raises:
        ValueError: If ``danger_array`` has the wrong spatial shape,
            ``times`` does not match ``n_frames``, or times are not
            strictly increasing.
    """

    def __init__(
        self,
        danger_array: np.ndarray,
        times: np.ndarray,
    ) -> None:
        darr = np.asarray(danger_array, dtype=np.float32)
        ts = np.asarray(times, dtype=np.float64)

        if darr.ndim != 4:
            raise ValueError(
                f"danger_array must be 4-D, got shape {darr.shape}"
            )
        if darr.shape[1:] != GRID_SHAPE:
            raise ValueError(
                f"danger_array spatial shape {darr.shape[1:]} != {GRID_SHAPE}"
            )
        if ts.shape != (darr.shape[0],):
            raise ValueError(
                f"times shape {ts.shape} != (n_frames={darr.shape[0]},)"
            )
        if not np.all(np.diff(ts) > 0):
            raise ValueError("times must be strictly increasing")

        # Build the 4-D RegularGridInterpolator over (t, x, y, z). The
        # spatial axes use cell centres (0.25, 0.75, …) so interpolation
        # is consistent with the SLCF storage layout.
        x_centres, y_centres, z_centres = cell_centres()
        self.danger = darr
        self.times = ts
        self.start_time = float(ts[0])
        self.t_max = float(ts[-1])
        self.x = x_centres
        self.y = y_centres
        self.z = z_centres
        self.interp = si.RegularGridInterpolator(
            (ts, x_centres, y_centres, z_centres),
            darr,
            method="linear",
            bounds_error=False,
            fill_value=1.0,
        )

    # ─── RiskMap.query ──────────────────────────────────────────────────
    def query(
        self,
        xyz: np.ndarray,
        t: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Return danger at ``xyz`` at time ``t`` (defaults to ``t_max``)."""
        arr = np.asarray(xyz, dtype=np.float64)

        if t is None:
            t = self.t_max

        # Out-of-time-range → 1.0 across the board.
        if t > self.t_max + _TOL:
            if arr.ndim == 1:
                return 1.0
            return np.ones(arr.shape[0], dtype=np.float32)

        t_clamped = max(self.start_time, t)

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

    # ─── Internals ──────────────────────────────────────────────────────
    def _in_bounds(self, xyz: np.ndarray) -> bool:
        lx, ly, lz = DOMAIN_SIZE_M
        return bool(
            -_TOL <= xyz[0] <= lx + _TOL
            and -_TOL <= xyz[1] <= ly + _TOL
            and -_TOL <= xyz[2] <= lz + _TOL
        )

    def _query_single(self, xyz: np.ndarray, t: float) -> float:
        if not self._in_bounds(xyz):
            return 1.0
        # The RegularGridInterpolator is built on cell centres (0.25 …
        # 29.75 etc.) — coordinates inside the SLCF region but outside
        # that strict centre range (e.g. x=0.0 on the wall plane) would
        # otherwise be silently filled with ``fill_value=1.0``. Clamp to
        # the centre extents so boundary queries use the nearest cell.
        x = float(np.clip(xyz[0], self.x[0], self.x[-1]))
        y = float(np.clip(xyz[1], self.y[0], self.y[-1]))
        z = float(np.clip(xyz[2], self.z[0], self.z[-1]))
        # RegularGridInterpolator returns a 1-D array even for a single
        # point; ``.item()`` extracts the scalar regardless.
        pt = np.array([[t, x, y, z]], dtype=np.float64)
        return float(np.asarray(self.interp(pt)).item())

    def _query_batch(self, xyz: np.ndarray, t: float) -> np.ndarray:
        # Same boundary-clamp logic as the single-point path, vectorised.
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
        points = np.column_stack([ts, clamped])
        out = self.interp(points)
        out[~in_bounds] = 1.0
        return out.astype(np.float32)

    # ─── Factories ──────────────────────────────────────────────────────
    @classmethod
    def from_fds_dir(cls, fds_dir: Path) -> "StaticRiskMap":
        """Load an FDS scenario directory and build the ground-truth risk map.

        Reads the ``.sf`` slice files via
        :func:`src.data_pipeline.fds_extractor.extract_slices`, applies
        the tenability functions on the raw fields, and wraps the
        resulting ``(31, 60, 40, 6)`` aggregated danger grid.

        Args:
            fds_dir: FDS scenario directory containing ``.smv`` + ``.sf``.

        Returns:
            :class:`StaticRiskMap` populated from ground-truth data.
        """
        # Lazy import to avoid a hard dependency from this module on the
        # data pipeline (some callers only need ``query()`` over a
        # pre-computed danger array).
        from src.data_pipeline.fds_extractor import extract_slices
        from src.risk_map.tenability import compute_total_danger

        slices = extract_slices(fds_dir)
        danger = compute_total_danger(
            slices["temperature"],
            slices["visibility"],
            slices["co"],
        ).astype(np.float32)
        return cls(danger_array=danger, times=slices["times"])

    @classmethod
    def from_npy(cls, path: Path) -> "StaticRiskMap":
        """Load a previously-saved ``.npz`` archive (``danger`` + ``times``).

        Args:
            path: Path to ``.npz`` produced by :meth:`save`.

        Returns:
            :class:`StaticRiskMap` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If the archive is missing required keys.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"npz not found: {path}")
        with np.load(path) as f:
            return cls(danger_array=f["danger"], times=f["times"])

    def save(self, path: Path) -> None:
        """Persist the danger grid + times to ``.npz`` for fast re-loading."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, danger=self.danger, times=self.times)


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("risk_map_class.py self-test")
    print("=" * 60)

    errors: list[str] = []
    nx, ny, nz = GRID_SHAPE
    times = np.arange(0.0, 300.0 + DT_SLCF / 2.0, DT_SLCF)  # 31 frames

    # Build a synthetic danger grid with a localised hot zone.
    danger = np.zeros((len(times), nx, ny, nz), dtype=np.float32)
    danger[5:, 30:35, 20:25, 2:4] = 0.8  # fire from t=50s near (15, 10, 1.5)
    rm = StaticRiskMap(danger_array=danger, times=times)

    # ── 1. Single-point inside hot zone ─────────────────────────────────
    print("\n[1] Point inside hot zone at t=100s")
    d = rm.query(np.array([15.25, 10.25, 1.25]), t=100.0)
    print(f"  query((15.25, 10.25, 1.25), t=100) = {d:.4f}  (expected ~0.8)")
    if abs(d - 0.8) > 1e-3:
        errors.append(f"hot-zone query = {d}, expected ~0.8")

    # ── 2. Same point at t=0 (before fire) ──────────────────────────────
    print("\n[2] Same point at t=0 (before fire)")
    d0 = rm.query(np.array([15.25, 10.25, 1.25]), t=0.0)
    print(f"  query((15.25, 10.25, 1.25), t=0) = {d0:.4f}  (expected 0)")
    if d0 > 1e-6:
        errors.append(f"pre-fire query = {d0}, expected 0")

    # ── 3. Out-of-bounds ─────────────────────────────────────────────────
    print("\n[3] OOB queries return 1.0")
    for p in (np.array([-1.0, 10.0, 1.5]),
              np.array([35.0, 10.0, 1.5]),
              np.array([15.0, 25.0, 1.5]),
              np.array([15.0, 10.0, 4.0])):
        v = rm.query(p, t=100.0)
        if v != 1.0:
            errors.append(f"OOB {p.tolist()} → {v} != 1.0")
    print("  PASS")

    # ── 4. Out-of-time-range ────────────────────────────────────────────
    print("\n[4] t > t_max → 1.0")
    v = rm.query(np.array([15.0, 10.0, 1.5]), t=999.0)
    print(f"  query(in-bounds, t=999) = {v}  (expected 1.0)")
    if v != 1.0:
        errors.append(f"t>t_max single = {v}")
    v_batch = rm.query(
        np.array([[15.0, 10.0, 1.5], [5.0, 5.0, 1.0]]), t=999.0,
    )
    if not np.allclose(v_batch, 1.0):
        errors.append(f"t>t_max batch = {v_batch}")

    # ── 5. t < start_time clamps ────────────────────────────────────────
    print("\n[5] t < start_time clamps to start_time")
    v = rm.query(np.array([15.25, 10.25, 1.25]), t=-50.0)
    expected = rm.query(np.array([15.25, 10.25, 1.25]), t=0.0)
    if abs(v - expected) > 1e-6:
        errors.append(f"t<start_time clamp wrong: {v} vs {expected}")
    print(f"  query(..., t=-50) = {v} == query(..., t=0) = {expected}  ✓")

    # ── 6. Batch query ──────────────────────────────────────────────────
    print("\n[6] Batched query")
    pts = np.array([
        [15.25, 10.25, 1.25],    # hot zone → ~0.8
        [5.0, 5.0, 1.5],         # far from fire → 0
        [-1.0, 10.0, 1.5],       # OOB → 1.0
    ])
    out = rm.query(pts, t=100.0)
    print(f"  result = {out}")
    if not (out.shape == (3,) and abs(out[0] - 0.8) < 1e-3
            and out[1] < 1e-6 and out[2] == 1.0):
        errors.append(f"batch query wrong: {out}")

    # ── 7. t=None defaults to t_max ─────────────────────────────────────
    print("\n[7] t=None uses t_max")
    v = rm.query(np.array([15.25, 10.25, 1.25]), t=None)
    expected = rm.query(np.array([15.25, 10.25, 1.25]), t=rm.t_max)
    if abs(v - expected) > 1e-9:
        errors.append("t=None default broken")
    print(f"  query(..., t=None) = {v} == query(..., t={rm.t_max}) ✓")

    # ── 8. save / from_npy round-trip ───────────────────────────────────
    print("\n[8] save / from_npy round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "rm.npz"
        rm.save(npz_path)
        rm2 = StaticRiskMap.from_npy(npz_path)
        v_a = rm.query(np.array([15.25, 10.25, 1.25]), t=100.0)
        v_b = rm2.query(np.array([15.25, 10.25, 1.25]), t=100.0)
        if abs(v_a - v_b) > 1e-9:
            errors.append(f"round-trip drift: {v_a} vs {v_b}")
        print(f"  saved+loaded → identical query: {v_a} == {v_b}")

    # ── 9. Input validation ─────────────────────────────────────────────
    print("\n[9] Input validation")
    try:
        StaticRiskMap(danger_array=np.zeros((5, 10, 10, 10)), times=times)
    except ValueError:
        print("  PASS: bad spatial shape raises")
    else:
        errors.append("bad spatial shape did not raise")
    try:
        StaticRiskMap(danger_array=danger, times=times[:5])
    except ValueError:
        print("  PASS: times length mismatch raises")
    else:
        errors.append("times mismatch did not raise")

    # ── 10. Real first_sim if available ─────────────────────────────────
    print("\n[10] Real first_sim → from_fds_dir")
    real = Path("data/raw/first_sim")
    if real.is_dir():
        rm_real = StaticRiskMap.from_fds_dir(real)
        d_fire = rm_real.query(np.array([18.0, 10.0, 1.5]), t=150.0)
        d_far = rm_real.query(np.array([2.0, 18.0, 1.5]), t=150.0)
        print(f"  near fire (18, 10, 1.5) t=150: danger={d_fire:.3f}")
        print(f"  far from fire (2, 18, 1.5) t=150: danger={d_far:.3f}")
        if not (d_fire > d_far):
            errors.append(
                f"fire-near danger ({d_fire}) should exceed fire-far ({d_far})"
            )
    else:
        print("  SKIP: data/raw/first_sim not present")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: StaticRiskMap validated")
