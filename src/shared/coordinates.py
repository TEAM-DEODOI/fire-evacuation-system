"""
Coordinate utilities for the fire evacuation prediction system.

See ``docs/coordinate_convention.md`` for the full specification.

Convention summary
------------------
* World space:   metres, Z-up, origin ``(0, 0, 0)`` at one building corner.
* Grid indices:  integer ``(ix, iy, iz)``, 0-based, into the SLCF region.
* Cell-centered: cell ``(ix, iy, iz)`` has its centre at world coordinate
  ``((ix + 0.5) * CELL_SIZE_M, (iy + 0.5) * CELL_SIZE_M, (iz + 0.5) * CELL_SIZE_M)``,
  i.e. **0.25 m** for the corner cell (see L-004).
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from src.shared.constants import CELL_SIZE_M, DOMAIN_SIZE_M, GRID_SHAPE

# Half-cell offset for cell-centred indexing — see L-004.
_HALF_CELL: float = CELL_SIZE_M / 2.0  # 0.25 m

# Tolerance used when comparing coordinates against bounds — chosen well
# below half a cell so a point exactly on the open boundary still counts.
_TOL: float = 1e-9


# ─── World ↔ grid conversion ───────────────────────────────────────────────
def world_to_grid(xyz: np.ndarray) -> np.ndarray:
    """Convert world metres → grid cell indices (nearest "owner" cell).

    Args:
        xyz: ``(3,)`` single point or ``(N, 3)`` batch of world coordinates.

    Returns:
        Same shape as ``xyz`` but ``dtype=int``. Each axis-component is the
        cell index that "owns" the point, computed as
        ``ix = int((world_x - 0.25) / 0.5)`` per L-004. Out-of-bounds
        components return ``-1`` (per axis, so a point straddling the X
        boundary still preserves its valid Y/Z indices).

    Raises:
        ValueError: If the last axis of ``xyz`` is not 3.
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"xyz must end in dimension 3, got shape {arr.shape}")

    # int() truncates toward zero — matches the formula in the contract.
    raw = (arr - _HALF_CELL) / CELL_SIZE_M
    idx = raw.astype(np.int64)

    # Mark per-axis out-of-bounds with -1.
    bounds = np.asarray(GRID_SHAPE, dtype=np.int64)
    in_bounds = (idx >= 0) & (idx < bounds)
    idx = np.where(in_bounds, idx, -1)
    return idx


def grid_to_world(idx: np.ndarray) -> np.ndarray:
    """Convert grid indices → world-space cell-centre coordinates (m).

    Args:
        idx: ``(3,)`` or ``(N, 3)``, integer-valued.

    Returns:
        Same shape, ``dtype=float64`` — cell centres at
        ``(ix + 0.5) * CELL_SIZE_M`` etc.

    Raises:
        ValueError: If the last axis is not 3.
    """
    arr = np.asarray(idx, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"idx must end in dimension 3, got shape {arr.shape}")
    return arr * CELL_SIZE_M + _HALF_CELL


# ─── Bounds check ──────────────────────────────────────────────────────────
def is_in_bounds(xyz: np.ndarray) -> Union[bool, np.ndarray]:
    """Test whether world coordinates lie inside the SLCF region.

    SLCF: ``X ∈ [0, 30]``, ``Y ∈ [0, 20]``, ``Z ∈ [0, 3]`` — closed intervals.

    Args:
        xyz: ``(3,)`` or ``(N, 3)``.

    Returns:
        Scalar ``bool`` for a single point or ``(N,)`` bool array.

    Raises:
        ValueError: If the last axis is not 3.
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"xyz must end in dimension 3, got shape {arr.shape}")

    lx, ly, lz = DOMAIN_SIZE_M
    if arr.ndim == 1:
        return bool(
            (-_TOL <= arr[0] <= lx + _TOL)
            and (-_TOL <= arr[1] <= ly + _TOL)
            and (-_TOL <= arr[2] <= lz + _TOL)
        )
    return (
        (arr[..., 0] >= -_TOL) & (arr[..., 0] <= lx + _TOL)
        & (arr[..., 1] >= -_TOL) & (arr[..., 1] <= ly + _TOL)
        & (arr[..., 2] >= -_TOL) & (arr[..., 2] <= lz + _TOL)
    )


# ─── Cell-centre coordinate arrays ─────────────────────────────────────────
def cell_centres() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the SLCF cell-centre coordinate arrays.

    Used to construct ``scipy.interpolate.RegularGridInterpolator`` instances
    over the SLCF region.

    Returns:
        ``(x, y, z)`` triple of 1-D ``float64`` arrays with lengths
        ``GRID_SHAPE`` and values starting at 0.25 m, stepping by
        ``CELL_SIZE_M``.
    """
    nx, ny, nz = GRID_SHAPE
    x = np.arange(nx, dtype=np.float64) * CELL_SIZE_M + _HALF_CELL
    y = np.arange(ny, dtype=np.float64) * CELL_SIZE_M + _HALF_CELL
    z = np.arange(nz, dtype=np.float64) * CELL_SIZE_M + _HALF_CELL
    return x, y, z


# ─── fdsreader coords sanity check ─────────────────────────────────────────
def verify_fdsreader_coords(coords: dict) -> bool:
    """Verify a coords dict returned by ``fdsreader.to_global`` matches the spec.

    Args:
        coords: ``{"x": array, "y": array, "z": array}`` — the second return
            value of ``slc.to_global(return_coordinates=True)``.

    Returns:
        ``True`` if all of the following hold; otherwise ``False`` (with a
        diagnostic message printed to stdout):

        * each axis array has the SLCF cell count (60 / 40 / 6).
        * first and last entries match ``0.25`` and ``DOMAIN - 0.25``.
    """
    nx, ny, nz = GRID_SHAPE
    lx, ly, lz = DOMAIN_SIZE_M
    issues: list[str] = []

    for axis, expected_n, expected_last in (
        ("x", nx, lx - _HALF_CELL),
        ("y", ny, ly - _HALF_CELL),
        ("z", nz, lz - _HALF_CELL),
    ):
        if axis not in coords:
            issues.append(f"coords missing key {axis!r}")
            continue
        arr = np.asarray(coords[axis])
        if arr.shape != (expected_n,):
            issues.append(
                f"coords['{axis}'].shape = {arr.shape}, expected ({expected_n},)"
            )
            continue
        if abs(float(arr[0]) - _HALF_CELL) > 1e-3:
            issues.append(
                f"coords['{axis}'][0] = {arr[0]}, expected {_HALF_CELL}"
            )
        if abs(float(arr[-1]) - expected_last) > 1e-3:
            issues.append(
                f"coords['{axis}'][-1] = {arr[-1]}, expected {expected_last}"
            )

    if issues:
        print("verify_fdsreader_coords FAILED:")
        for msg in issues:
            print(f"  - {msg}")
        return False
    return True


# ─── Self-test (run with ``python -m src.shared.coordinates``) ────────────
if __name__ == "__main__":
    print("=" * 60)
    print("coordinates.py self-test")
    print("=" * 60)

    errors: list[str] = []
    nx, ny, nz = GRID_SHAPE

    # ── 1. Single-point conversions ────────────────────────────────────────
    print("\n[1] Single-point world ↔ grid conversion")
    cases: list[tuple[list[float], list[int]]] = [
        ([0.25, 0.25, 0.25], [0, 0, 0]),
        ([29.75, 19.75, 2.75], [nx - 1, ny - 1, nz - 1]),
        ([15.25, 9.75, 1.25], [30, 19, 2]),
    ]
    for world, expected in cases:
        got = world_to_grid(np.asarray(world))
        if got.tolist() != expected:
            errors.append(f"world_to_grid({world}) = {got.tolist()}, expected {expected}")
        else:
            print(f"  world_to_grid({world}) = {got.tolist()} OK")

    if grid_to_world(np.array([0, 0, 0])).tolist() != [_HALF_CELL] * 3:
        errors.append("grid_to_world([0,0,0]) wrong")
    if grid_to_world(np.array([nx - 1, ny - 1, nz - 1])).tolist() != [29.75, 19.75, 2.75]:
        errors.append("grid_to_world(last) wrong")

    # ── 2. Round-trip identity over the full grid ──────────────────────────
    print("\n[2] Round-trip identity grid → world → grid")
    fails = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                rt = world_to_grid(grid_to_world(np.array([ix, iy, iz])))
                if rt.tolist() != [ix, iy, iz]:
                    fails += 1
    if fails:
        errors.append(f"{fails} round-trip cells failed")
    else:
        print(f"  all {nx * ny * nz} cells round-tripped OK")

    # ── 3. Batch processing ────────────────────────────────────────────────
    print("\n[3] Batched (N, 3) inputs")
    batch = np.array(
        [
            [0.25, 0.25, 0.25],
            [29.75, 19.75, 2.75],
            [15.0, 10.0, 1.5],
        ]
    )
    idx_batch = world_to_grid(batch)
    if idx_batch.shape != (3, 3) or idx_batch.dtype.kind != "i":
        errors.append(f"batch shape/dtype wrong: {idx_batch.shape} {idx_batch.dtype}")
    print(f"  batch input shape {batch.shape} → output shape {idx_batch.shape}")
    world_batch = grid_to_world(idx_batch)
    print(f"  grid_to_world batch shape = {world_batch.shape}")

    # ── 4. Out-of-bounds ───────────────────────────────────────────────────
    print("\n[4] Out-of-bounds handling")
    oob_low = world_to_grid(np.array([-1.0, 5.0, 1.0]))
    if -1 not in oob_low.tolist():
        errors.append(f"world_to_grid([-1, 5, 1]) did not contain -1: {oob_low}")
    else:
        print(f"  world_to_grid([-1, 5, 1]) = {oob_low.tolist()} (contains -1)")

    oob_high = world_to_grid(np.array([100.0, 5.0, 1.0]))
    if -1 not in oob_high.tolist():
        errors.append(f"world_to_grid([100, 5, 1]) did not contain -1: {oob_high}")
    else:
        print(f"  world_to_grid([100, 5, 1]) = {oob_high.tolist()} (contains -1)")

    if not is_in_bounds(np.array([15.0, 10.0, 1.5])):
        errors.append("is_in_bounds([15, 10, 1.5]) should be True")
    if is_in_bounds(np.array([-1.0, 10.0, 1.5])):
        errors.append("is_in_bounds([-1, 10, 1.5]) should be False")
    if not is_in_bounds(np.array([0.0, 0.0, 0.0])):
        errors.append("is_in_bounds at corner (0,0,0) should be True")
    if not is_in_bounds(np.array([30.0, 20.0, 3.0])):
        errors.append("is_in_bounds at corner (30,20,3) should be True")

    # Batch bounds
    bb = is_in_bounds(
        np.array(
            [
                [15.0, 10.0, 1.5],
                [-1.0, 10.0, 1.5],
                [40.0, 10.0, 1.5],
            ]
        )
    )
    if not (bool(bb[0]) and not bool(bb[1]) and not bool(bb[2])):
        errors.append(f"batch is_in_bounds: {bb}")

    # ── 5. cell_centres ────────────────────────────────────────────────────
    print("\n[5] cell_centres()")
    x_arr, y_arr, z_arr = cell_centres()
    if x_arr.shape != (nx,) or y_arr.shape != (ny,) or z_arr.shape != (nz,):
        errors.append(
            f"cell_centres shapes ({x_arr.shape}, {y_arr.shape}, {z_arr.shape})"
        )
    if abs(float(x_arr[0]) - 0.25) > 1e-12 or abs(float(x_arr[-1]) - 29.75) > 1e-12:
        errors.append(f"x_arr endpoints: {x_arr[0]}, {x_arr[-1]}")
    if abs(float(y_arr[-1]) - 19.75) > 1e-12 or abs(float(z_arr[-1]) - 2.75) > 1e-12:
        errors.append(f"y/z endpoints: y={y_arr[-1]}, z={z_arr[-1]}")
    print(f"  x[0]={x_arr[0]}  x[-1]={x_arr[-1]}")
    print(f"  y[0]={y_arr[0]}  y[-1]={y_arr[-1]}")
    print(f"  z[0]={z_arr[0]}  z[-1]={z_arr[-1]}")

    # ── 6. verify_fdsreader_coords ─────────────────────────────────────────
    print("\n[6] verify_fdsreader_coords")
    good = {"x": x_arr, "y": y_arr, "z": z_arr}
    if not verify_fdsreader_coords(good):
        errors.append("verify_fdsreader_coords rejected a valid coords dict")
    bad = {"x": x_arr, "y": y_arr, "z": z_arr[:-1]}  # nz=5 → wrong
    if verify_fdsreader_coords(bad):
        errors.append("verify_fdsreader_coords accepted an invalid coords dict")
    print("  PASS: validator accepts good, rejects bad")

    # ── Verdict ────────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: All coordinate functions validated")
