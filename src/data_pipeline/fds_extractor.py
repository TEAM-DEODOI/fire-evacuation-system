"""
FDS slice-file extractor — raw ``.sf`` parser that bypasses ``fdsreader``.

Why this module exists
----------------------
``fdsreader`` raises a broadcast error on PyroSim's cell-centered SLCF
output because PyroSim writes node-based ``(62, 42, 8)`` data while
``fdsreader`` tries to compute a ``(61, 41, 7)`` cell-centered slice
inconsistently. ``L-001`` and ``L-009`` document the symptom; this
module is the workaround.

We parse the FORTRAN-unformatted ``.sf`` records ourselves, derive the
cell-centred field as the 8-vertex average per cell, slice down to the
project's ``(60, 40, 6)`` SLCF region (D-015) and align the time axis to
the canonical 31 frames. No external FDS library is imported.

Output of :func:`extract_slices` matches ``docs/interface_contracts.md``
§3.1 — see :func:`extract_slices` docstring for the exact dict layout.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.shared.constants import (
    DT_SLCF,
    GRID_SHAPE,
    N_TIMESTEPS,
    T_END_SECONDS,
    T_MIN_SECONDS,
)
from src.shared.coordinates import cell_centres

# Minimum node-shape needed to extract the SLCF region. PyroSim usually
# produces (62, 42, 8); we accept anything ≥ this (so the SLCF region
# fits) but flag larger shapes implicitly via the slice in
# :func:`extract_to_slcf_region`.
_MIN_NODE_SHAPE: Tuple[int, int, int] = (
    GRID_SHAPE[0] + 1,
    GRID_SHAPE[1] + 1,
    GRID_SHAPE[2] + 1,
)

# Frame-count safety threshold — way below the canonical 31. Anything
# smaller is almost certainly a truncated simulation rather than a
# downsampling decision.
_MIN_FRAMES_REQUIRED: int = 10

# Sane initial-temperature window. Real FDS scenarios start at TMPA
# (typically 20 °C). Anything below 0 °C or above 50 °C indicates a
# corrupt or mis-parsed file.
_INIT_TEMP_C_MIN: float = 0.0
_INIT_TEMP_C_MAX: float = 50.0


# ─── FORTRAN unformatted record helpers ────────────────────────────────────
def _read_record(f) -> Optional[bytes]:
    """Read one FORTRAN-unformatted record from ``f``.

    Layout: ``[uint32 size] [size bytes data] [uint32 size]`` — the size
    is repeated as a trailer so files can be read either direction.

    Returns:
        The payload bytes, or ``None`` on EOF.

    Raises:
        ValueError: If the record is truncated.
    """
    head = f.read(4)
    if len(head) == 0:
        return None
    if len(head) != 4:
        raise ValueError(f"truncated record header at offset {f.tell()}")
    (size,) = struct.unpack("<I", head)
    payload = f.read(size)
    if len(payload) != size:
        raise ValueError(
            f"truncated record payload: read {len(payload)} of {size} bytes"
        )
    tail = f.read(4)
    if len(tail) != 4:
        raise ValueError("truncated record trailer")
    return payload


# ─── .sf parsing ───────────────────────────────────────────────────────────
def parse_sf_file(sf_path: Path) -> Dict[str, Any]:
    """Raw-parse a single FDS ``.sf`` slice file (no fdsreader).

    Args:
        sf_path: Path to the slice file.

    Returns:
        ``{"quantity": str, "unit": str, "indices": (i1,i2,j1,j2,k1,k2),
        "node_shape": (nx, ny, nz), "times": (T,) float32,
        "node_data": (T, nx, ny, nz) float32}``.

    Raises:
        FileNotFoundError: If ``sf_path`` does not exist.
        ValueError: If the file is malformed, truncated, or has a
            record size inconsistent with its declared node shape.
    """
    sf_path = Path(sf_path)
    if not sf_path.exists():
        raise FileNotFoundError(f"slice file not found: {sf_path}")

    with sf_path.open("rb") as f:
        # ── Header (4 records) ─────────────────────────────────────────────
        q_bytes = _read_record(f)
        if q_bytes is None:
            raise ValueError(f"{sf_path}: empty file")
        s_bytes = _read_record(f)
        u_bytes = _read_record(f)
        i_bytes = _read_record(f)
        if any(r is None for r in (s_bytes, u_bytes, i_bytes)):
            raise ValueError(f"{sf_path}: truncated header")

        quantity = q_bytes.decode("ascii", errors="replace").strip()
        unit = u_bytes.decode("ascii", errors="replace").strip()

        if len(i_bytes) != 24:
            raise ValueError(
                f"{sf_path}: indices record size {len(i_bytes)} != 24 (6 × int32)"
            )
        i1, i2, j1, j2, k1, k2 = struct.unpack("<6i", i_bytes)
        nx_node = i2 - i1 + 1
        ny_node = j2 - j1 + 1
        nz_node = k2 - k1 + 1
        node_shape = (nx_node, ny_node, nz_node)
        if any(n <= 0 for n in node_shape):
            raise ValueError(f"{sf_path}: non-positive node_shape {node_shape}")

        expected_data_bytes = nx_node * ny_node * nz_node * 4  # float32

        # ── Frame loop ─────────────────────────────────────────────────────
        times: List[float] = []
        frames: List[np.ndarray] = []
        while True:
            t_record = _read_record(f)
            if t_record is None:
                break
            if len(t_record) != 4:
                raise ValueError(
                    f"{sf_path}: time record size {len(t_record)} != 4 "
                    f"(frame {len(times)})"
                )
            (t_value,) = struct.unpack("<f", t_record)

            d_record = _read_record(f)
            if d_record is None:
                raise ValueError(
                    f"{sf_path}: missing data record for time {t_value} "
                    f"(frame {len(times)})"
                )
            if len(d_record) != expected_data_bytes:
                raise ValueError(
                    f"{sf_path}: data record size {len(d_record)} != "
                    f"{expected_data_bytes} for node_shape {node_shape}"
                )

            grid = np.frombuffer(d_record, dtype="<f4").reshape(
                node_shape, order="F"
            )
            times.append(float(t_value))
            frames.append(np.ascontiguousarray(grid, dtype=np.float32))

    if not frames:
        raise ValueError(f"{sf_path}: no frames found after header")

    return {
        "quantity": quantity,
        "unit": unit,
        "indices": (i1, i2, j1, j2, k1, k2),
        "node_shape": node_shape,
        "times": np.asarray(times, dtype=np.float32),
        "node_data": np.stack(frames, axis=0),
    }


# ─── Node → cell-centred conversion (8-vertex average) ────────────────────
def node_to_cell_centered(node_data: np.ndarray) -> np.ndarray:
    """Convert node-based field to cell-centred field by 8-vertex averaging.

    Args:
        node_data: ``(T, nx, ny, nz)`` or ``(nx, ny, nz)``.

    Returns:
        Same rank with each spatial dim reduced by 1.

    Raises:
        ValueError: If ``node_data`` is not 3-D or 4-D.
    """
    if node_data.ndim == 4:
        return (
            node_data[:, :-1, :-1, :-1]
            + node_data[:, 1:, :-1, :-1]
            + node_data[:, :-1, 1:, :-1]
            + node_data[:, :-1, :-1, 1:]
            + node_data[:, 1:, 1:, :-1]
            + node_data[:, 1:, :-1, 1:]
            + node_data[:, :-1, 1:, 1:]
            + node_data[:, 1:, 1:, 1:]
        ) / 8.0
    if node_data.ndim == 3:
        return (
            node_data[:-1, :-1, :-1]
            + node_data[1:, :-1, :-1]
            + node_data[:-1, 1:, :-1]
            + node_data[:-1, :-1, 1:]
            + node_data[1:, 1:, :-1]
            + node_data[1:, :-1, 1:]
            + node_data[:-1, 1:, 1:]
            + node_data[1:, 1:, 1:]
        ) / 8.0
    raise ValueError(f"node_data must be 3-D or 4-D, got {node_data.ndim}-D")


# ─── SLCF region cropping (D-015) ──────────────────────────────────────────
def extract_to_slcf_region(cell_data: np.ndarray) -> np.ndarray:
    """Crop the cell-centred field down to ``(T, 60, 40, 6)``.

    PyroSim emits an extra cell in each dimension and stretches Z to 3.5 m
    even when the SLCF requests Z=3.0 m. We discard the over-shot and keep
    only the model-visible region.

    Args:
        cell_data: ``(T, nx_cell, ny_cell, nz_cell)``.

    Returns:
        ``(T, GRID_SHAPE[0], GRID_SHAPE[1], GRID_SHAPE[2])`` slice.

    Raises:
        ValueError: If ``cell_data`` is too small to contain the SLCF region.
    """
    nx, ny, nz = GRID_SHAPE
    if cell_data.ndim != 4:
        raise ValueError(
            f"cell_data must be 4-D (T, X, Y, Z), got shape {cell_data.shape}"
        )
    if cell_data.shape[1] < nx or cell_data.shape[2] < ny or cell_data.shape[3] < nz:
        raise ValueError(
            f"cell_data shape {cell_data.shape} too small. "
            f"Need at least (T, {nx}, {ny}, {nz})."
        )
    return cell_data[:, :nx, :ny, :nz]


# ─── Time alignment ────────────────────────────────────────────────────────
def align_times(
    raw_times: np.ndarray,
    target_times: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map raw FDS frame times → indices closest to the canonical grid.

    FDS uses adaptive CFL time-stepping so SLCF output times drift slightly
    off the nominal ``0, 10, 20, …, 300`` second schedule. We pick the
    nearest available frame per target time.

    Args:
        raw_times: ``(N_raw,)`` actual FDS output times.
        target_times: ``(N_target,)`` desired times. Defaults to
            ``np.arange(T_MIN_SECONDS, T_END_SECONDS + DT_SLCF/2, DT_SLCF)``
            (31 points: 0, 10, …, 300).

    Returns:
        ``(N_target,)`` int — index into ``raw_times`` for each target.
    """
    if target_times is None:
        target_times = np.arange(
            T_MIN_SECONDS, T_END_SECONDS + DT_SLCF / 2.0, DT_SLCF
        )
    raw = np.asarray(raw_times)
    indices = np.searchsorted(raw, np.asarray(target_times))
    return np.clip(indices, 0, len(raw) - 1).astype(np.int64)


# ─── .smv parsing ──────────────────────────────────────────────────────────
# SLCC / SLCF block: a header line whose tail carries six index values
# after an "&" marker, followed by four lines: filename, quantity name,
# short name, unit.
_SLC_HEADER_RE = re.compile(r"^\s*SLC[CF]\b")
_SLC_INDICES_RE = re.compile(
    r"&\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"
)


def parse_smv_for_slices(smv_path: Path) -> List[Dict[str, Any]]:
    """Extract slice-block metadata from an FDS ``.smv`` file.

    Args:
        smv_path: Path to ``*.smv``.

    Returns:
        List of dicts, one per SLCC/SLCF block, with keys
        ``sf_filename``, ``quantity``, ``short_name``, ``unit``,
        ``indices`` (6-tuple of int).

    Raises:
        FileNotFoundError: If ``smv_path`` does not exist.
        ValueError: If a slice block is truncated.
    """
    smv_path = Path(smv_path)
    if not smv_path.exists():
        raise FileNotFoundError(f".smv not found: {smv_path}")

    lines = smv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    slices: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _SLC_HEADER_RE.match(line):
            if i + 4 >= len(lines):
                raise ValueError(
                    f"{smv_path}: SLC[CF] block at line {i + 1} is truncated"
                )
            idx_match = _SLC_INDICES_RE.search(line)
            indices: Tuple[int, ...] = (
                tuple(int(x) for x in idx_match.groups()) if idx_match else (0,) * 6
            )

            sf_filename = lines[i + 1].strip()
            quantity = lines[i + 2].strip()
            short_name = lines[i + 3].strip()
            unit = lines[i + 4].strip()
            slices.append(
                {
                    "sf_filename": sf_filename,
                    "quantity": quantity,
                    "short_name": short_name,
                    "unit": unit,
                    "indices": indices,
                }
            )
            i += 5
        else:
            i += 1
    return slices


# ─── Quantity → kind classifier ────────────────────────────────────────────
def _classify_quantity(quantity: str, short_name: str) -> Optional[str]:
    """Map an FDS quantity-name string to one of ``temperature``/``visibility``/``co``.

    Returns ``None`` for slices we do not care about (e.g. velocity).
    """
    q = quantity.upper()
    s = short_name.upper()
    if "TEMPERATURE" in q:
        return "temperature"
    if "VISIBILITY" in q:
        return "visibility"
    # CO can appear as full "CARBON MONOXIDE VOLUME FRACTION" or as
    # "VOLUME FRACTION" with the short name carrying CO.
    if "CARBON MONOXIDE" in q or (("VOLUME FRACTION" in q) and ("CO" in s)):
        return "co"
    return None


# ─── Top-level extraction ──────────────────────────────────────────────────
def extract_slices(fds_dir: Path) -> Dict[str, Any]:
    """Extract the three project SLCFs from an FDS output directory.

    Pipeline: parse ``.smv`` → match files to T/V/CO by quantity name →
    parse each ``.sf`` → cell-centre via 8-vertex average → crop to
    ``(T, 60, 40, 6)`` → convert CO mol-fraction → ppm → align times to
    the 31-frame canonical grid.

    Args:
        fds_dir: Directory containing exactly one ``.smv`` plus its ``.sf``
            slice files.

    Returns:
        Dict matching ``docs/interface_contracts.md`` §3.1::

            {
                "temperature": (31, 60, 40, 6) float32, °C raw
                "visibility":  (31, 60, 40, 6) float32, m  raw
                "co":          (31, 60, 40, 6) float32, ppm
                "times":       (31,) float32, [0, 10, …, 300] s
                "coords": {
                    "x": (60,) float32 cell centres,
                    "y": (40,) float32,
                    "z": (6,)  float32,
                },
            }

    Raises:
        FileNotFoundError: If ``fds_dir`` or any expected file is missing.
        ValueError: If slice count != 3, node shape too small, frame count
            insufficient, or initial-temperature sanity check fails.
    """
    fds_dir = Path(fds_dir)
    if not fds_dir.is_dir():
        raise FileNotFoundError(f"fds_dir is not a directory: {fds_dir}")

    smv_files = sorted(fds_dir.glob("*.smv"))
    if not smv_files:
        raise FileNotFoundError(f"no .smv file in {fds_dir}")
    smv_path = smv_files[0]

    slice_metas = parse_smv_for_slices(smv_path)
    if not slice_metas:
        raise ValueError(f"{smv_path}: no SLCC/SLCF blocks found")

    by_kind: Dict[str, Dict[str, Any]] = {}
    for meta in slice_metas:
        kind = _classify_quantity(meta["quantity"], meta["short_name"])
        if kind is None or kind in by_kind:
            # Keep the first occurrence per kind; PyroSim sometimes emits
            # duplicates with the same quantity.
            continue
        by_kind[kind] = meta

    missing = [k for k in ("temperature", "visibility", "co") if k not in by_kind]
    if missing:
        raise ValueError(
            f"{smv_path}: could not map .sf files to {missing}. "
            f"Found quantities: {[m['quantity'] for m in slice_metas]}"
        )
    if len(by_kind) != 3:
        raise ValueError(
            f"{smv_path}: expected exactly 3 mapped slices (T, V, CO), got {len(by_kind)}"
        )

    # ── Parse the three .sf files ──────────────────────────────────────────
    parsed: Dict[str, Dict[str, Any]] = {}
    for kind, meta in by_kind.items():
        sf_path = fds_dir / meta["sf_filename"]
        parsed[kind] = parse_sf_file(sf_path)

    # ── Validation: node shape and frame count ────────────────────────────
    for kind, p in parsed.items():
        nshape = p["node_shape"]
        if any(n < req for n, req in zip(nshape, _MIN_NODE_SHAPE)):
            raise ValueError(
                f"{kind}: node_shape {nshape} < required minimum {_MIN_NODE_SHAPE}"
            )
        if len(p["times"]) < _MIN_FRAMES_REQUIRED:
            raise ValueError(
                f"{kind}: only {len(p['times'])} frames in .sf "
                f"(< {_MIN_FRAMES_REQUIRED}). Simulation may have stopped early."
            )

    # ── Cell-centre → SLCF crop → time-align ──────────────────────────────
    aligned: Dict[str, np.ndarray] = {}
    canonical_times: Optional[np.ndarray] = None

    for kind, p in parsed.items():
        cell = node_to_cell_centered(p["node_data"])         # (T, nx-1, ny-1, nz-1)
        slcf = extract_to_slcf_region(cell)                  # (T, 60, 40, 6)

        idx = align_times(p["times"])                        # (31,)
        if canonical_times is None:
            canonical_times = p["times"][idx].astype(np.float32)
        aligned[kind] = slcf[idx].astype(np.float32, copy=False)

    if canonical_times is None:
        raise RuntimeError("internal: canonical_times never populated")
    if len(canonical_times) != N_TIMESTEPS:
        raise ValueError(
            f"aligned time vector has {len(canonical_times)} entries, "
            f"expected {N_TIMESTEPS}"
        )

    # ── CO mol/mol → ppm ──────────────────────────────────────────────────
    aligned["co"] = aligned["co"] * 1.0e6

    # ── Sanity: initial temperature ────────────────────────────────────────
    t_init_mean = float(np.nanmean(aligned["temperature"][0]))
    if not (_INIT_TEMP_C_MIN <= t_init_mean <= _INIT_TEMP_C_MAX):
        raise ValueError(
            f"initial temperature mean {t_init_mean:.2f} °C outside "
            f"[{_INIT_TEMP_C_MIN}, {_INIT_TEMP_C_MAX}] — check FDS &MISC TMPA"
        )

    x_centres, y_centres, z_centres = cell_centres()
    return {
        "temperature": aligned["temperature"],
        "visibility": aligned["visibility"],
        "co": aligned["co"],
        "times": canonical_times,
        "coords": {
            "x": x_centres.astype(np.float32),
            "y": y_centres.astype(np.float32),
            "z": z_centres.astype(np.float32),
        },
    }


# ─── Convenience: scenario discovery ──────────────────────────────────────
def list_scenarios(raw_dir: Path) -> List[Path]:
    """Return sorted scenario subdirectories under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
    return sorted(d for d in raw_dir.iterdir() if d.is_dir())


# ─── Synthetic-data helpers for self-test ─────────────────────────────────
def _write_record(f, payload: bytes) -> None:
    """Write one FORTRAN-unformatted record (size + payload + size)."""
    n = len(payload)
    f.write(struct.pack("<I", n))
    f.write(payload)
    f.write(struct.pack("<I", n))


def _make_synthetic_sf(
    path: Path,
    node_shape: Tuple[int, int, int],
    n_frames: int,
    quantity: str,
    short_name: str,
    unit: str,
    fill_fn,
) -> None:
    """Write a self-consistent ``.sf`` file for unit testing.

    ``fill_fn(t_idx) -> np.ndarray`` produces the per-frame node-grid in
    physical units.
    """
    nx_, ny_, nz_ = node_shape
    with path.open("wb") as f:
        _write_record(f, quantity.ljust(30).encode("ascii")[:30])
        _write_record(f, short_name.ljust(30).encode("ascii")[:30])
        _write_record(f, unit.ljust(30).encode("ascii")[:30])
        _write_record(
            f, struct.pack("<6i", 0, nx_ - 1, 0, ny_ - 1, 0, nz_ - 1)
        )
        for t_idx in range(n_frames):
            t_val = t_idx * DT_SLCF
            _write_record(f, struct.pack("<f", float(t_val)))
            arr = fill_fn(t_idx).astype("<f4", copy=False)
            if arr.shape != node_shape:
                raise RuntimeError(
                    f"fill_fn returned shape {arr.shape}, expected {node_shape}"
                )
            # FDS data is FORTRAN column-major.
            _write_record(f, arr.tobytes(order="F"))


def _make_synthetic_smv(
    path: Path,
    entries: List[Tuple[str, str, str, str]],
    node_shape: Tuple[int, int, int],
) -> None:
    """Write a minimal ``.smv`` containing one SLCC block per entry.

    Each ``entries`` tuple is ``(sf_filename, quantity, short_name, unit)``.
    """
    nx_, ny_, nz_ = node_shape
    lines: List[str] = ["! synthetic SMV for fds_extractor self-test", ""]
    for slot, (fname, q, s, u) in enumerate(entries, start=1):
        lines.append(
            f"SLCC{slot:6d} # STRUCTURED &  0 {nx_ - 1}  0 {ny_ - 1}  0 {nz_ - 1} !"
        )
        lines.append(f" {fname}")
        lines.append(f" {q}")
        lines.append(f" {s}")
        lines.append(f" {u}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── Self-test (run with ``python -m src.data_pipeline.fds_extractor``) ────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("fds_extractor.py self-test")
    print("=" * 60)

    errors: List[str] = []

    # ── Test 1: node_to_cell_centered shape and arithmetic ───────────────
    print("\n[Test 1] node_to_cell_centered")
    node = np.ones((N_TIMESTEPS, 62, 42, 8), dtype=np.float32)
    cell = node_to_cell_centered(node)
    if cell.shape != (N_TIMESTEPS, 61, 41, 7):
        errors.append(f"shape wrong: {cell.shape}")
    elif not np.allclose(cell, 1.0):
        errors.append("uniform-1 input did not produce uniform-1 output")
    else:
        print(f"  shape {node.shape} → {cell.shape}, uniform value preserved")
    # 3-D variant
    cell3 = node_to_cell_centered(np.ones((62, 42, 8), dtype=np.float32))
    if cell3.shape != (61, 41, 7):
        errors.append(f"3-D variant shape wrong: {cell3.shape}")
    # 8-vertex averaging correctness: linear ramp in X → cell value = midpoint
    ramp = np.tile(
        np.arange(62, dtype=np.float32).reshape(62, 1, 1), (1, 42, 8)
    )
    cell_ramp = node_to_cell_centered(ramp)
    # Cell (i, j, k) averages node[i:i+2] → value = i + 0.5
    expected_first = 0.5
    expected_last = 60.5
    if abs(float(cell_ramp[0, 0, 0]) - expected_first) > 1e-5:
        errors.append(f"ramp cell[0,0,0] = {cell_ramp[0,0,0]} != {expected_first}")
    if abs(float(cell_ramp[-1, 0, 0]) - expected_last) > 1e-5:
        errors.append(f"ramp cell[-1,0,0] = {cell_ramp[-1,0,0]} != {expected_last}")

    # ── Test 2: extract_to_slcf_region ───────────────────────────────────
    print("\n[Test 2] extract_to_slcf_region")
    cell = np.random.rand(N_TIMESTEPS, 61, 41, 7).astype(np.float32)
    slcf = extract_to_slcf_region(cell)
    if slcf.shape != (N_TIMESTEPS, *GRID_SHAPE):
        errors.append(f"shape wrong: {slcf.shape}")
    elif not np.allclose(slcf[0, 0, 0, 0], cell[0, 0, 0, 0]):
        errors.append("first cell not preserved by crop")
    else:
        print(f"  shape {cell.shape} → {slcf.shape}")
    # ValueError on too-small input
    try:
        extract_to_slcf_region(np.zeros((10, 10, 10, 10), dtype=np.float32))
        errors.append("expected ValueError on too-small input")
    except ValueError:
        pass

    # ── Test 3: align_times ──────────────────────────────────────────────
    print("\n[Test 3] align_times")
    raw_t = np.array([0.0, 9.99, 20.01, 29.97, 40.05], dtype=np.float32)
    target = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    idx = align_times(raw_t, target)
    if idx.shape != (5,):
        errors.append(f"align_times shape: {idx.shape}")
    print(f"  raw → indices {idx.tolist()}")
    # Default target is the canonical 31-frame grid
    raw_long = np.arange(0.0, 300.0 + DT_SLCF, DT_SLCF, dtype=np.float32)
    idx_default = align_times(raw_long)
    if idx_default.shape != (N_TIMESTEPS,):
        errors.append(f"default target length: {idx_default.shape}")

    # ── Test 4: parse_sf_file round-trip with synthetic file ─────────────
    print("\n[Test 4] parse_sf_file round-trip on synthetic .sf")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        node_shape = (62, 42, 8)
        n_frames = 31

        def temp_fill(t_idx):
            # Ramp 20 °C → 200 °C linearly across frames, spatially uniform.
            base = 20.0 + (180.0 / max(1, n_frames - 1)) * t_idx
            return np.full(node_shape, base, dtype=np.float32)

        sf_path = tmp_dir / "synth_T.sf"
        _make_synthetic_sf(
            sf_path, node_shape, n_frames, "TEMPERATURE", "temp", "C", temp_fill
        )
        parsed = parse_sf_file(sf_path)
        if parsed["quantity"] != "TEMPERATURE":
            errors.append(f"quantity round-trip: {parsed['quantity']!r}")
        if parsed["node_shape"] != node_shape:
            errors.append(f"node_shape round-trip: {parsed['node_shape']}")
        if parsed["node_data"].shape != (n_frames, *node_shape):
            errors.append(f"node_data shape: {parsed['node_data'].shape}")
        if len(parsed["times"]) != n_frames:
            errors.append(f"times length: {len(parsed['times'])}")
        # First frame should be uniform 20 °C
        if not np.allclose(parsed["node_data"][0], 20.0):
            errors.append("initial frame not uniform 20 °C")
        print(f"  parsed quantity={parsed['quantity']!r} "
              f"node_shape={parsed['node_shape']} frames={len(parsed['times'])}")

        # ── Test 5: parse_smv_for_slices ───────────────────────────────
        print("\n[Test 5] parse_smv_for_slices")
        smv_path = tmp_dir / "scenario.smv"
        _make_synthetic_smv(
            smv_path,
            entries=[
                ("synth_T.sf", "TEMPERATURE", "temp", "C"),
                ("synth_V.sf", "SOOT VISIBILITY", "vis", "m"),
                ("synth_CO.sf", "CARBON MONOXIDE VOLUME FRACTION", "co", "mol/mol"),
            ],
            node_shape=node_shape,
        )
        metas = parse_smv_for_slices(smv_path)
        if len(metas) != 3:
            errors.append(f"parse_smv_for_slices returned {len(metas)} blocks")
        kinds = {_classify_quantity(m["quantity"], m["short_name"]) for m in metas}
        if kinds != {"temperature", "visibility", "co"}:
            errors.append(f"classifier kinds = {kinds}")
        for m in metas:
            print(f"  {m['sf_filename']:14s} {m['quantity']}")

        # ── Test 6: extract_slices end-to-end on synthetic directory ───
        print("\n[Test 6] extract_slices end-to-end")

        # Build T (uniform 22 °C → rising 100 °C at last frame, fire-like ramp)
        def fill_T(t_idx):
            grid = np.full(node_shape, 22.0, dtype=np.float32)
            # Ignite a hot patch after a few frames so the initial mean stays sane.
            if t_idx >= 3:
                grid[30:35, 20:25, 1:3] = 100.0 + 20.0 * (t_idx - 3)
            return grid

        # V: 30 m everywhere, dropping with fire
        def fill_V(t_idx):
            grid = np.full(node_shape, 30.0, dtype=np.float32)
            if t_idx >= 3:
                grid[30:35, 20:25, 1:3] = max(0.5, 30.0 - 5.0 * (t_idx - 3))
            return grid

        # CO: mol/mol (will be ×1e6 → ppm). 50 ppm baseline, 1500 ppm in fire.
        def fill_CO(t_idx):
            grid = np.full(node_shape, 50e-6, dtype=np.float32)
            if t_idx >= 3:
                grid[30:35, 20:25, 1:3] = 1500e-6
            return grid

        _make_synthetic_sf(tmp_dir / "synth_T.sf", node_shape, n_frames,
                           "TEMPERATURE", "temp", "C", fill_T)
        _make_synthetic_sf(tmp_dir / "synth_V.sf", node_shape, n_frames,
                           "SOOT VISIBILITY", "vis", "m", fill_V)
        _make_synthetic_sf(tmp_dir / "synth_CO.sf", node_shape, n_frames,
                           "CARBON MONOXIDE VOLUME FRACTION", "co", "mol/mol", fill_CO)

        out = extract_slices(tmp_dir)
        for kind in ("temperature", "visibility", "co"):
            shape = out[kind].shape
            if shape != (N_TIMESTEPS, *GRID_SHAPE):
                errors.append(f"{kind} shape {shape} != (31, 60, 40, 6)")
        if out["times"].shape != (N_TIMESTEPS,):
            errors.append(f"times shape: {out['times'].shape}")
        if out["coords"]["x"].shape != (GRID_SHAPE[0],):
            errors.append(f"coords x shape: {out['coords']['x'].shape}")
        # CO must now be in ppm (50 baseline → 50.0 after ×1e6)
        co0_mean = float(np.nanmean(out["co"][0]))
        if not (40.0 <= co0_mean <= 60.0):
            errors.append(f"CO baseline (ppm) after ×1e6 = {co0_mean}, expected ≈50")
        # Initial temperature must pass sanity check (we used 22 °C baseline)
        t_init = float(np.nanmean(out["temperature"][0]))
        if not (18.0 <= t_init <= 25.0):
            errors.append(f"T initial mean = {t_init}, expected ≈22")
        print(f"  T shape   = {out['temperature'].shape}, "
              f"T_init = {t_init:.2f} °C")
        print(f"  V shape   = {out['visibility'].shape}")
        print(f"  CO shape  = {out['co'].shape}, "
              f"CO_init = {co0_mean:.2f} ppm")
        print(f"  times     = {out['times'][0]:.1f} … {out['times'][-1]:.1f} s "
              f"({len(out['times'])} frames)")
        print(f"  coords    = x[{out['coords']['x'][0]:.2f}, "
              f"{out['coords']['x'][-1]:.2f}]  "
              f"y[{out['coords']['y'][0]:.2f}, "
              f"{out['coords']['y'][-1]:.2f}]  "
              f"z[{out['coords']['z'][0]:.2f}, "
              f"{out['coords']['z'][-1]:.2f}]")

    # ── Verdict ─────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: fds_extractor 모듈 검증 완료")
