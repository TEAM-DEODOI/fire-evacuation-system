"""Find FDS scenarios where fire spreads *evenly* across the interior
(opposite of :mod:`scripts.find_asymmetric_fires`).

Why we want this: with D-041 west-only exits, an asymmetric fire that
sits near the active exit is the easiest case for S1 to fail at — but
the most dramatic *drone* case is the opposite: fire fills the building
uniformly so every path to the single exit traverses risky cells, and
the planner has to choose the *least bad* corridor. That is where a
risk-aware drone earns its keep over a risk-blind sign.

Heuristic per scenario:

1. Load the cached :class:`StaticRiskMap`.
2. At the final frame, split the interior (z=3) into a 3×3 spatial grid
   over [0..30] × [0..20] m. For each cell of the grid measure mean
   ``danger``.
3. ``coverage`` = fraction of interior cells (intersected with
   :func:`load_interior_mask`) with danger > 0.30. A bigger fire.
4. ``uniformity`` = ``min(quadrant_mean) / max(quadrant_mean)`` over
   the 3×3 grid. 1.0 means perfectly even; closer to 0 means
   concentrated in one quadrant.
5. ``score`` = ``coverage × uniformity``. Higher = bigger and more
   evenly spread.

Run::

    python scripts/find_evenly_spread_fires.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.path_planning.building_graph import load_interior_mask
from src.risk_map.risk_map_class import StaticRiskMap
from src.shared.constants import CELL_SIZE_M, GRID_SHAPE


_CACHE = Path("results/cache/scenario_risk_maps")


def _sample_grid(rm: StaticRiskMap, t: float) -> np.ndarray:
    nx_, ny_, _ = GRID_SHAPE
    xs = 0.25 + CELL_SIZE_M * np.arange(nx_)
    ys = 0.25 + CELL_SIZE_M * np.arange(ny_)
    z = 0.25 + CELL_SIZE_M * 3
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([xx, yy, np.full_like(xx, z)], axis=-1).reshape(-1, 3)
    vals = np.asarray(rm.query(pts, t=t), dtype=np.float32).reshape(nx_, ny_)
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, default=_CACHE)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--coverage-thr", type=float, default=0.30,
                    help="Risk threshold counted as 'on fire' for coverage")
    args = ap.parse_args()

    interior = load_interior_mask()[:, :, 3]   # (60, 40) bool
    nx_, ny_ = interior.shape
    n_int = int(interior.sum())
    print(f"Interior cells (z=3): {n_int}")

    # 3x3 grid boundaries (x: 0..30, y: 0..20)
    bx = np.linspace(0, nx_, 4, dtype=int)   # [0, 20, 40, 60]
    by = np.linspace(0, ny_, 4, dtype=int)   # [0, 13/14, 26/27, 40]

    scored = []
    for npz in sorted(args.cache_dir.glob("*.npz")):
        try:
            rm = StaticRiskMap.from_npy(npz)
        except Exception as exc:
            continue
        vals = _sample_grid(rm, t=rm.t_max)
        vals = np.where(interior, vals, np.nan)
        # Coverage
        on_fire = (vals > args.coverage_thr)
        coverage = float(np.nansum(on_fire & interior)) / max(n_int, 1)
        # Quadrant means (3x3 = 9 quadrants)
        means = []
        for i in range(3):
            for j in range(3):
                block = vals[bx[i]:bx[i+1], by[j]:by[j+1]]
                m = np.nanmean(block) if np.any(~np.isnan(block)) else np.nan
                if not np.isnan(m):
                    means.append(float(m))
        if len(means) < 4:
            continue
        max_q = max(means)
        min_q = min(means)
        uniformity = (min_q / max_q) if max_q > 1e-6 else 0.0
        score = coverage * uniformity
        scored.append((
            npz.stem, score, coverage, uniformity,
            float(np.nanmean(vals)), max_q, min_q,
        ))

    scored.sort(key=lambda r: r[1], reverse=True)
    print()
    print(f"Top {args.top} by score = coverage x uniformity "
          f"(threshold {args.coverage_thr}):")
    print(
        f"  {'scenario':<30}  {'score':>5}  {'cover':>5}  {'unif':>5}  "
        f"{'mean':>5}  {'qmax':>5}  {'qmin':>5}"
    )
    for sid, score, cov, unif, m, qmx, qmn in scored[: args.top]:
        print(
            f"  {sid:<30}  {score:>5.3f}  {cov:>5.2f}  {unif:>5.2f}  "
            f"{m:>5.2f}  {qmx:>5.2f}  {qmn:>5.2f}"
        )

    print()
    print("Recommended (coverage >= 0.40, uniformity >= 0.50):")
    good = [r for r in scored if r[2] >= 0.40 and r[3] >= 0.50]
    for sid, score, cov, unif, *_ in good[:10]:
        print(f"  {sid:<30}  cover={cov:.2f}  unif={unif:.2f}")
    if not good:
        print("  (none — relax thresholds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
