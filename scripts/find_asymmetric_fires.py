"""Find FDS scenarios where fire / smoke is heavily concentrated near ONE
exit while the other exits stay safe.

Such scenarios are ideal to expose the S1 (risk-blind fixed-sign)
baseline: a person whose *nearest* exit is the one being engulfed will
march into the fire, while the S2/S3 drone swarm should detour them to
a safer exit.

Heuristic per scenario:

1. Load the cached :class:`StaticRiskMap` (``results/cache/scenario_risk_maps/<id>.npz``).
2. Sample the final-frame danger at each of the 3 canonical exits.
3. Score = ``max(exit_danger) - min(exit_danger)`` (large = one-sided).
4. Print top-K asymmetric scenarios with per-exit numbers.

A scenario is "useful" when:

* max ≥ 0.5   (the worst exit is genuinely hazardous)
* min ≤ 0.3   (at least one exit is genuinely safe)
* asymmetry ≥ 0.3

Run::

    python scripts/find_asymmetric_fires.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.integration.scenarios._common import exit_positions
from src.risk_map.risk_map_class import StaticRiskMap


_CACHE = Path("results/cache/scenario_risk_maps")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, default=_CACHE)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    exits = exit_positions()
    print(f"Probing {len(exits)} exits:")
    for i, ex in enumerate(exits):
        print(f"  exit_{i}: ({ex[0]:.2f}, {ex[1]:.2f}, {ex[2]:.2f})")

    scored: list[tuple[str, float, list[float], float]] = []
    for npz in sorted(args.cache_dir.glob("*.npz")):
        try:
            rm = StaticRiskMap.from_npy(npz)
        except Exception as exc:
            print(f"  SKIP {npz.name}: {exc.__class__.__name__}")
            continue
        # Sample final-frame danger at each exit (and a small 3x3 neighbourhood
        # around it to smooth interp noise).
        per_exit: list[float] = []
        for ex in exits:
            local: list[float] = []
            for dx in (-0.5, 0.0, 0.5):
                for dy in (-0.5, 0.0, 0.5):
                    p = np.array([ex[0] + dx, ex[1] + dy, ex[2]])
                    local.append(float(rm.query(p, t=rm.t_max)))
            per_exit.append(float(np.mean(local)))
        asym = max(per_exit) - min(per_exit)
        max_d = max(per_exit)
        scored.append((npz.stem, asym, per_exit, max_d))

    # Sort by asymmetry desc.
    scored.sort(key=lambda r: r[1], reverse=True)

    print()
    print(f"Top {args.top} by exit-asymmetry (max - min final-frame danger):")
    print(f"  {'scenario':<30}  {'asym':>5}  {'max':>5}  {'min':>5}  "
          f"{'west':>5}  {'north':>5}  {'east':>5}")
    for sid, asym, per_exit, max_d in scored[: args.top]:
        print(
            f"  {sid:<30}  {asym:>5.2f}  {max(per_exit):>5.2f}  "
            f"{min(per_exit):>5.2f}  "
            f"{per_exit[0]:>5.2f}  {per_exit[1]:>5.2f}  {per_exit[2]:>5.2f}"
        )

    print()
    print("Recommended (max>=0.5, min<=0.3, asym>=0.3):")
    good = [
        r for r in scored
        if r[3] >= 0.5 and min(r[2]) <= 0.3 and r[1] >= 0.3
    ]
    for sid, asym, per_exit, max_d in good[:10]:
        worst_idx = int(np.argmax(per_exit))
        safest_idx = int(np.argmin(per_exit))
        names = ["west", "north", "east"]
        print(
            f"  {sid:<30}  worst={names[worst_idx]}({per_exit[worst_idx]:.2f})  "
            f"safest={names[safest_idx]}({per_exit[safest_idx]:.2f})"
        )

    if not good:
        print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
