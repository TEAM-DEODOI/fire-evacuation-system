"""Build the canonical *interior* mask (spawn-region union) from FDS risk maps.

Motivation
==========

The FDS fluid mask (``data/processed/dataset.h5::mask``) marks every cell
that smoke *can* flow through as ``True`` — this includes the outdoor
buffer around the building. Sampling spawn positions uniformly from the
fluid mask therefore places some agents outside the building, which is
visually wrong and physically meaningless for our evacuation experiment.

Per the user's design (2026-05-14): a cell is *interior* iff at least one
FDS scenario eventually fills it with non-trivial danger. Smoke escapes
to atmosphere through doors but never accumulates in open air, so the
union of "risk > ε at the final frame, across all scenarios" carves out
the building interior cleanly without any geometry knowledge.

What this script does
=====================

1. Iterate every FDS scenario in ``data/raw/`` (excluding ``first_sim``
   and the ``scenario_config.json`` file).
2. Load each scenario's :class:`~src.risk_map.risk_map_class.StaticRiskMap`
   from the cache when available, otherwise build it from FDS slice data
   and write the cache.
3. For each ``(60, 40, 6)`` danger grid take the final frame and OR-mask
   ``risk > eps`` (default ``eps=0.05``) into the running union.
4. Persist the union to ``data/processed/interior_mask.npz`` with keys
   ``mask`` (bool, ``(60, 40, 6)``) and ``meta`` (record array with
   per-scenario contribution counts).

The result is consumed by
:func:`src.path_planning.building_graph.load_interior_mask` and used by
:func:`src.integration.scenarios._common.spawn_agents` to restrict spawn
to indoor cells.

Usage::

    python scripts/build_interior_mask.py            # default eps=0.05
    python scripts/build_interior_mask.py --eps 0.02 # looser threshold
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.path_planning.building_graph import load_default_fluid_mask
from src.risk_map.risk_map_class import StaticRiskMap
from src.shared.constants import GRID_SHAPE


_RAW_DIR = Path("data/raw")
_CACHE_DIR = Path("results/cache/scenario_risk_maps")
_OUT_PATH = Path("data/processed/interior_mask.npz")
_EXCLUDE = {"first_sim", "scenario_config.json"}


def _discover_scenarios(raw_dir: Path) -> List[Path]:
    """Return every FDS scenario folder under ``raw_dir``."""
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and p.name not in _EXCLUDE
    )


def _load_or_build_risk_map(fds_dir: Path, *, verbose: bool) -> StaticRiskMap:
    """Load a cached :class:`StaticRiskMap` if present; else build + cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{fds_dir.name}.npz"
    if cache_path.exists():
        if verbose:
            print(f"  cache hit: {cache_path.name}")
        return StaticRiskMap.from_npy(cache_path)
    if verbose:
        print(f"  building from FDS (first call ~30 s)...")
    rm = StaticRiskMap.from_fds_dir(fds_dir)
    rm.save(cache_path)
    return rm


def build_interior_mask(
    raw_dir: Path = _RAW_DIR,
    eps: float = 0.05,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Compute the union interior mask over every FDS scenario.

    Args:
        raw_dir: Directory holding scenario subfolders (default ``data/raw``).
        eps: Risk threshold above which a cell counts as smoke-reached.
        verbose: Print per-scenario progress.

    Returns:
        ``(mask, contributions)`` where ``mask`` is a ``(60, 40, 6)`` bool
        union (``True`` = interior) and ``contributions`` is a list of
        ``(scenario_name, new_cells_added)`` tuples for diagnostics.
    """
    scenarios = _discover_scenarios(raw_dir)
    if not scenarios:
        raise RuntimeError(f"no scenarios discovered under {raw_dir}")

    union = np.zeros(GRID_SHAPE, dtype=bool)
    contributions: List[Tuple[str, int]] = []
    t0 = time.perf_counter()

    for i, fds_dir in enumerate(scenarios, start=1):
        if verbose:
            print(f"[{i:2d}/{len(scenarios)}] {fds_dir.name}")
        try:
            rm = _load_or_build_risk_map(fds_dir, verbose=verbose)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  SKIP ({exc.__class__.__name__}: {exc})")
            contributions.append((fds_dir.name, -1))
            continue

        # Use the last frame: smoke distribution at maximum spread.
        final = rm.danger[-1]  # (60, 40, 6)
        active = final > eps
        new_cells = int((active & ~union).sum())
        union |= active
        contributions.append((fds_dir.name, new_cells))
        if verbose:
            print(
                f"  active={int(active.sum()):5d}  "
                f"new={new_cells:4d}  union_total={int(union.sum()):5d}"
            )

    dt = time.perf_counter() - t0
    if verbose:
        print(
            f"\nRaw union built in {dt:.1f}s. "
            f"Raw cells: {int(union.sum())} / {union.size} "
            f"({union.mean():.1%})"
        )

    # AND with the fluid mask: RegularGridInterpolator leaks small values
    # into adjacent solid cells, which would let us spawn agents inside
    # walls. Restrict to truly-navigable fluid cells.
    fluid = load_default_fluid_mask()
    pre_and = int(union.sum())
    union = union & fluid
    if verbose:
        removed = pre_and - int(union.sum())
        print(
            f"After AND with fluid_mask: removed {removed} wall-bleed cells. "
            f"Interior cells: {int(union.sum())} ({union.mean():.1%})"
        )
    return union, contributions


def save_interior_mask(
    mask: np.ndarray,
    contributions: List[Tuple[str, int]],
    out_path: Path = _OUT_PATH,
    eps: float = 0.05,
) -> Path:
    """Persist ``mask`` + per-scenario contribution metadata to ``.npz``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = np.asarray([c[0] for c in contributions])
    counts = np.asarray([c[1] for c in contributions], dtype=np.int64)
    np.savez_compressed(
        out_path,
        mask=mask.astype(bool),
        scenario_names=names,
        new_cells=counts,
        eps=np.float32(eps),
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Risk threshold for 'smoke reached' (default 0.05).")
    parser.add_argument("--raw-dir", type=Path, default=_RAW_DIR)
    parser.add_argument("--out", type=Path, default=_OUT_PATH)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(f"build_interior_mask -- raw={args.raw_dir}  eps={args.eps}")
    print("=" * 70)

    if not args.raw_dir.exists():
        print(f"FAIL: raw dir missing: {args.raw_dir}", file=sys.stderr)
        return 1

    mask, contribs = build_interior_mask(
        raw_dir=args.raw_dir,
        eps=args.eps,
        verbose=not args.quiet,
    )
    out = save_interior_mask(mask, contribs, out_path=args.out, eps=args.eps)
    size_kb = out.stat().st_size / 1024
    print(f"\nwrote {out}  ({size_kb:.0f} KB)")

    # Layer-by-layer summary
    print("\nLayer-wise interior cell counts:")
    for k in range(mask.shape[2]):
        z_m = 0.25 + 0.5 * k
        print(f"  z={k} (z={z_m:.2f} m): {int(mask[:, :, k].sum()):4d} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
