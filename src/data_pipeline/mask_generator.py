"""
Build the building fluid/solid mask from FDS scenario output.

The mask is a ``(60, 40, 6)`` ``float32`` array where ``1.0`` marks fluid
cells (smoke/occupant accessible) and ``0.0`` marks solid cells (walls,
slabs, obstructions). Because the FDS building geometry is identical
across all 30 scenarios, the mask only needs to be generated once and is
re-used in :func:`src.data_pipeline.build_dataset.build`.

The primary source is the ``.fds`` input deck: each ``&OBST`` line with
non-zero Z extent is rasterised onto the SLCF grid. Zero-thickness
``&OBST`` (floor / ceiling visualisation slabs at ``z=0,0`` or ``z=3,3``)
are skipped — they are STL surface markers, not flow-blocking walls.

A secondary "data-only" method (:func:`generate_mask_from_slices`) infers
solidity from cells whose visibility never deviates from the ambient
30 m across the full simulation. It is a fallback when ``.fds`` parsing
fails (e.g. ``&GEOM`` STL-only decks with no ``&OBST`` decomposition).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.shared.constants import (
    CELL_SIZE_M,
    DOMAIN_SIZE_M,
    GRID_SHAPE,
    V_NORM_MAX_M,
)


# &OBST line capture — match the six XB coordinates regardless of formatting.
_OBST_XB_RE = re.compile(
    r"""
    &OBST\b                              # opening token
    [^/]*?                               # any attribute soup up to XB
    XB\s*=\s*
    ([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,
    \s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,
    \s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# OBSTs thinner than half a cell in Z are treated as STL surface markers
# rather than walls. Without this filter every floor / ceiling decomposed
# from the STL would erroneously eat up the bottom or top z-cell.
_MIN_Z_THICKNESS_M: float = CELL_SIZE_M / 2.0


# ─── Primary: .fds OBST → mask ───────────────────────────────────────────
def generate_mask_from_fds_file(fds_path: Path) -> np.ndarray:
    """Parse an FDS input deck and rasterise its walls into a ``(60, 40, 6)`` mask.

    Args:
        fds_path: Path to the ``.fds`` deck.

    Returns:
        ``(60, 40, 6)`` ``float32`` mask. ``1.0`` = fluid, ``0.0`` = solid.

    Raises:
        FileNotFoundError: If ``fds_path`` does not exist.
        ValueError: If the file contains no usable wall OBSTs (most likely
            because PyroSim emitted ``&GEOM`` instead). Callers may fall
            back to :func:`generate_mask_from_slices` in that case.
    """
    fds_path = Path(fds_path)
    if not fds_path.exists():
        raise FileNotFoundError(f".fds not found: {fds_path}")

    nx, ny, nz = GRID_SHAPE
    lx, ly, lz = DOMAIN_SIZE_M
    mask = np.ones((nx, ny, nz), dtype=np.float32)

    content = fds_path.read_text(encoding="utf-8", errors="replace")

    n_total = 0
    n_walls = 0
    n_thin_skipped = 0
    n_out_of_region = 0
    n_cells_marked_solid = 0

    for m in _OBST_XB_RE.finditer(content):
        n_total += 1
        x1, x2, y1, y2, z1, z2 = (float(v) for v in m.groups())

        # Sort the bounds in case PyroSim emitted them in reverse.
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if z1 > z2:
            z1, z2 = z2, z1

        # Zero-thickness in Z = floor / ceiling decomposition surface; skip.
        if (z2 - z1) < _MIN_Z_THICKNESS_M:
            n_thin_skipped += 1
            continue

        # Clip Z to the SLCF region (X, Y are handled below to preserve
        # thin-plate walls, which legitimately have ``x1 == x2`` or
        # ``y1 == y2`` and would otherwise be lost to the clip).
        z1c = max(z1, 0.0)
        z2c = min(z2, lz)
        if z2c <= z1c:
            n_out_of_region += 1
            continue

        # X axis — either a thin YZ-plane wall (``x1 == x2``) or an extended
        # slab. Thin plates occupy exactly one cell column; clamp the wall
        # to ``[0, lx]`` first so external-buffer plates collapse to a
        # single legal index.
        x_thin = (x2 - x1) < _MIN_Z_THICKNESS_M
        if x_thin:
            x_wall = min(max(x1, 0.0), lx)
            if not (0.0 <= x_wall <= lx):
                n_out_of_region += 1
                continue
            ix_lo = max(0, min(nx - 1, int(x_wall / CELL_SIZE_M)))
            ix_hi = ix_lo + 1
        else:
            x1c = max(x1, 0.0)
            x2c = min(x2, lx)
            if x2c <= x1c:
                n_out_of_region += 1
                continue
            ix_lo = max(0, int(np.floor(x1c / CELL_SIZE_M)))
            ix_hi = min(nx, int(np.ceil(x2c / CELL_SIZE_M)))

        # Y axis — symmetric handling for XZ-plane thin walls.
        y_thin = (y2 - y1) < _MIN_Z_THICKNESS_M
        if y_thin:
            y_wall = min(max(y1, 0.0), ly)
            if not (0.0 <= y_wall <= ly):
                n_out_of_region += 1
                continue
            iy_lo = max(0, min(ny - 1, int(y_wall / CELL_SIZE_M)))
            iy_hi = iy_lo + 1
        else:
            y1c = max(y1, 0.0)
            y2c = min(y2, ly)
            if y2c <= y1c:
                n_out_of_region += 1
                continue
            iy_lo = max(0, int(np.floor(y1c / CELL_SIZE_M)))
            iy_hi = min(ny, int(np.ceil(y2c / CELL_SIZE_M)))

        # Z axis — already clipped above.
        iz_lo = max(0, int(np.floor(z1c / CELL_SIZE_M)))
        iz_hi = min(nz, int(np.ceil(z2c / CELL_SIZE_M)))

        if ix_hi <= ix_lo or iy_hi <= iy_lo or iz_hi <= iz_lo:
            continue

        n_walls += 1
        before = float(mask[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi].sum())
        mask[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi] = 0.0
        after = float(mask[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi].sum())
        n_cells_marked_solid += int(before - after)

    if n_walls == 0:
        raise ValueError(
            f"{fds_path}: no wall-forming OBSTs found "
            f"(scanned {n_total} OBSTs, skipped {n_thin_skipped} thin "
            f"slabs, {n_out_of_region} outside SLCF region). "
            "Consider falling back to generate_mask_from_slices()."
        )

    print(
        f"  OBST scan: {n_total} total, {n_walls} walls used, "
        f"{n_thin_skipped} thin skipped, {n_out_of_region} OOR"
    )
    print(
        f"  solid cells: {int((mask == 0.0).sum())} / {mask.size} "
        f"({100 * (mask == 0.0).sum() / mask.size:.1f}%)"
    )
    return mask


# ─── Convenience: discover the .fds file in a scenario directory ─────────
def generate_mask_from_fds(fds_dir: Path) -> np.ndarray:
    """Find the ``.fds`` deck inside ``fds_dir`` and produce the mask."""
    fds_dir = Path(fds_dir)
    if not fds_dir.is_dir():
        raise FileNotFoundError(f"fds_dir is not a directory: {fds_dir}")
    fds_files = sorted(fds_dir.glob("*.fds"))
    if not fds_files:
        raise FileNotFoundError(f"no .fds deck in {fds_dir}")
    return generate_mask_from_fds_file(fds_files[0])


# ─── Fallback: visibility-stability heuristic ─────────────────────────────
def generate_mask_from_slices(
    slices: Dict[str, np.ndarray],
    tol_m: float = 0.5,
) -> np.ndarray:
    """Infer a mask from extracted SLCF data.

    A cell is marked solid if its visibility stays within ``tol_m`` metres
    of the ambient ``V_NORM_MAX_M`` (30 m) for every frame — i.e. no smoke
    ever reaches it. Mirrors the "unchanged from t=0 to t=300" heuristic
    used during ad-hoc inspection of first_sim.

    Args:
        slices: Output of ``extract_slices`` (uses ``"visibility"`` key).
        tol_m: How close to 30 m visibility must remain across all frames
            for the cell to be considered solid.

    Returns:
        ``(60, 40, 6)`` ``float32`` mask.

    Notes:
        This method has false positives: fluid cells far from the fire
        that smoke never reaches also stay at 30 m. Use ``.fds`` parsing
        whenever possible; reserve this for decks where ``&OBST`` is absent.
    """
    if "visibility" not in slices:
        raise ValueError("slices dict missing 'visibility' key")
    v = np.asarray(slices["visibility"])
    if v.ndim != 4:
        raise ValueError(f"visibility must be 4-D, got shape {v.shape}")

    stayed_clear = (np.abs(v - V_NORM_MAX_M) < tol_m).all(axis=0)
    mask = np.where(stayed_clear, 0.0, 1.0).astype(np.float32)
    return mask


# ─── Trivial default ─────────────────────────────────────────────────────
def default_open_floor_mask() -> np.ndarray:
    """All-fluid placeholder ``(60, 40, 6)`` mask of ones."""
    return np.ones(GRID_SHAPE, dtype=np.float32)


# ─── Optional: side-by-side visualisation against the summary slice ─────
def save_mask_preview(mask: np.ndarray, out_path: Path) -> None:
    """Write a quick PNG of the mask at three z-levels for visual review.

    Args:
        mask: ``(60, 40, 6)`` mask.
        out_path: Destination PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z_levels = [1, 3, 5]
    z_meters = [0.25 + iz * CELL_SIZE_M for iz in z_levels]

    fig, axes = plt.subplots(1, len(z_levels), figsize=(4 * len(z_levels), 4))
    for ax, iz, zm in zip(axes, z_levels, z_meters):
        ax.imshow(
            mask[:, :, iz].T,
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            extent=[0, DOMAIN_SIZE_M[0], 0, DOMAIN_SIZE_M[1]],
            aspect="equal",
        )
        ax.set_title(f"mask @ z={zm:.2f} m  (white=fluid, black=solid)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    fig.suptitle(
        f"Building mask — fluid ratio {(mask == 1.0).sum() / mask.size:.1%}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─── Self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("mask_generator.py self-test")
    print("=" * 60)

    errors: list[str] = []
    nx_, ny_, nz_ = GRID_SHAPE
    lx_, ly_, lz_ = DOMAIN_SIZE_M

    # ── 1. default_open_floor_mask ───────────────────────────────────────
    print("\n[1] default_open_floor_mask()")
    om = default_open_floor_mask()
    if om.shape != GRID_SHAPE or om.dtype != np.float32 or float(om.sum()) != om.size:
        errors.append("default mask wrong")
    print(f"  shape={om.shape}, dtype={om.dtype}, all-1: OK")

    # ── 2. Synthetic .fds → mask ────────────────────────────────────────
    print("\n[2] generate_mask_from_fds_file (synthetic deck)")
    with tempfile.TemporaryDirectory() as tmp:
        synth_fds = Path(tmp) / "synth.fds"
        # Two real walls + one floor slab (should be skipped) + one out-of-region
        # wall (should be skipped).
        synth_fds.write_text(
            "&HEAD CHID='X' /\n"
            "&OBST ID='floor',  XB=0.0,30.0,0.0,20.0,0.0,0.0 /\n"          # thin
            "&OBST ID='wall_y', XB=10.0,10.5, 0.0,20.0, 0.0,3.0 /\n"      # x=10..10.5 column
            "&OBST ID='wall_x', XB= 0.0,30.0, 5.0, 5.5, 0.0,3.0 /\n"      # y=5..5.5 row
            "&OBST ID='outside',XB=-5.0,-3.0, 0.0,20.0, 0.0,3.0 /\n"      # outside SLCF
            "&TAIL /\n",
            encoding="utf-8",
        )
        m = generate_mask_from_fds_file(synth_fds)
        # Column ix=20 (x=10..10.5) should be all solid except where row blocks already are
        col_solid = (m[20, :, :] == 0.0).all()
        row_solid = (m[:, 10, :] == 0.0).all()
        if not col_solid:
            errors.append("synthetic wall column ix=20 not fully solid")
        if not row_solid:
            errors.append("synthetic wall row iy=10 not fully solid")
        # Cells far from walls must remain fluid
        if m[0, 0, 0] != 1.0:
            errors.append("corner cell (0,0,0) should be fluid")
        print(f"  walls rasterised; corner (0,0,0) fluid: {m[0,0,0] == 1.0}")

    # ── 3. ValueError when no usable OBSTs ──────────────────────────────
    print("\n[3] .fds with only thin/OOR obstacles → ValueError")
    with tempfile.TemporaryDirectory() as tmp:
        bad_fds = Path(tmp) / "no_walls.fds"
        bad_fds.write_text(
            "&OBST XB=0.0,30.0,0.0,20.0,0.0,0.0 /\n"
            "&OBST XB=0.0,30.0,0.0,20.0,3.0,3.0 /\n",
            encoding="utf-8",
        )
        try:
            generate_mask_from_fds_file(bad_fds)
            errors.append("expected ValueError for no walls")
        except ValueError:
            print("  PASS: ValueError raised as expected")

    # ── 4. generate_mask_from_slices fallback ────────────────────────────
    print("\n[4] generate_mask_from_slices fallback")
    fake_v = np.full((31, nx_, ny_, nz_), V_NORM_MAX_M, dtype=np.float32)
    # Mark a fluid corridor: visibility drops at later frames
    fake_v[5:, 20:40, 10:30, :] = 5.0  # smoke reaches this region
    mask_from_data = generate_mask_from_slices({"visibility": fake_v})
    fluid_cells = int((mask_from_data == 1.0).sum())
    solid_cells = int((mask_from_data == 0.0).sum())
    print(f"  fluid={fluid_cells}, solid={solid_cells}")
    # Region where smoke reached should be marked fluid
    if mask_from_data[25, 20, 3] != 1.0:
        errors.append("smoky cell not marked fluid in fallback")

    # ── 5. Real first_sim deck if available ──────────────────────────────
    print("\n[5] Real first_sim deck (if present)")
    real_dir = Path("data/raw/first_sim")
    if real_dir.is_dir():
        real_mask = generate_mask_from_fds(real_dir)
        n_solid = int((real_mask == 0.0).sum())
        ratio = n_solid / real_mask.size
        print(f"  first_sim mask: solid {n_solid} / {real_mask.size}  ({ratio:.1%})")
        # Sanity bounds: a single-floor building should have *some* walls,
        # but not be entirely walled in.
        if not (0.0 < ratio < 0.9):
            errors.append(f"first_sim mask solid ratio {ratio} implausible")
        preview = Path("figures/first_sim/mask_preview.png")
        save_mask_preview(real_mask, preview)
        print(f"  preview saved → {preview}")
    else:
        print("  SKIP: data/raw/first_sim not present")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: mask_generator validated")
