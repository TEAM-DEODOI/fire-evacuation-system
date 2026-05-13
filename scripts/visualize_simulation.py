"""End-to-end visualisation of EXP-PATH-001 scenarios.

Runs one trial each of S1 / S2 / S3 on a chosen fire scenario, captures
the per-tick state via :class:`SimulationRecorder`, and emits the
following figures under ``figures/sim/<fire_scenario_id>__seed<N>/``:

* ``s1_trajectories.png``  — top-down trails for S1 alone
* ``s2_trajectories.png``  — top-down trails for S2 alone
* ``s3_trajectories.png``  — top-down trails for S3 alone
* ``comparison.png``       — 3-panel S1 / S2 / S3 side-by-side
* ``s{1,2,3}.gif``         — animated GIF per scenario (optional via flag)

The simulation code in ``src/integration/scenarios/*`` was not modified
for this script -- it only consumes the optional ``recorder=`` keyword
that already exists on each ``run()``. Adding new figure types (heat
map difference, FED gradient, drone overlay, ...) extends *this script*
or :mod:`src.integration.renderer` only.

Usage::

    python scripts/visualize_simulation.py \\
        --fire sim_1500kw_2m2_T05 \\
        --seed 0 \\
        --n-persons 5 \\
        --t-end 200 \\
        --animate
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")

from src.integration.recorder import SimulationRecorder
from src.integration.renderer import (
    render_animation,
    render_comparison,
    render_trajectories,
)
from src.integration.scenarios import s1_fixed_sign, s2_fds_swarm, s3_fno_swarm
from src.integration.scenarios._common import load_truth_risk_map
from src.path_planning.building_graph import load_default_fluid_mask


_FIRE_DEFAULT = "sim_1500kw_2m2_T05"


def _run_with_recorder(
    label: str,
    runner,
    fire_scenario_id: str,
    fds_dir: Path,
    n_persons: int,
    seed: int,
    t_end_s: float,
    dt_s: float,
    **extra,
) -> SimulationRecorder:
    """Run one scenario with recording and return the populated recorder."""
    rec = SimulationRecorder(
        scenario_id=label,
        fire_scenario_id=fire_scenario_id,
        seed=seed,
    )
    t0 = time.perf_counter()
    runner(
        fire_scenario_id=fire_scenario_id,
        fds_dir=fds_dir,
        n_persons=n_persons,
        seed=seed,
        t_end_s=t_end_s,
        dt_s=dt_s,
        recorder=rec,
        **extra,
    )
    dt = time.perf_counter() - t0
    n_evac = sum(
        1 for aid in rec.agent_ids() if rec.final_status(aid) == "evacuated"
    )
    print(
        f"  {label:<16} {len(rec.frames):>4} frames  "
        f"{n_evac}/{len(rec.agent_ids())} evac  "
        f"({dt:.1f}s wall)"
    )
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run S1/S2/S3 once each and render trajectory figures "
            "(post-D-026 cell-grid). The simulation modules see only "
            "the optional recorder= kwarg -- no other coupling."
        )
    )
    parser.add_argument(
        "--fire", type=str, default=_FIRE_DEFAULT,
        help=f"FDS scenario folder name under data/raw/ (default {_FIRE_DEFAULT}).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for spawn positions.",
    )
    parser.add_argument("--n-persons", type=int, default=5)
    parser.add_argument("--t-end", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw"),
        help="Parent of FDS scenario folders.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default figures/sim/<fire>__seed<seed>/).",
    )
    parser.add_argument(
        "--no-risk-overlay", action="store_true",
        help="Disable risk-heatmap overlay (faster).",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Also write per-scenario GIF animation.",
    )
    parser.add_argument(
        "--fps", type=int, default=8,
        help="GIF framerate (only with --animate).",
    )
    args = parser.parse_args()

    fds_dir = args.raw_dir / args.fire
    if not fds_dir.exists():
        print(f"FAIL: FDS scenario dir not found: {fds_dir}", file=sys.stderr)
        return 1

    out_dir = args.output or (
        Path("figures/sim") / f"{args.fire}__seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(
        f"visualize_simulation -- fire={args.fire}  seed={args.seed}  "
        f"n_persons={args.n_persons}  t_end={args.t_end}s"
    )
    print("=" * 70)

    # ── Pre-load shared assets once (mask + truth risk map) ────────
    print("\n[setup] loading fluid mask + truth RiskMap")
    fluid_mask = load_default_fluid_mask()
    truth_rm = load_truth_risk_map(fds_dir, verbose=True)

    # ── Run 3 scenarios with recording ─────────────────────────────
    print("\n[run] executing 3 scenarios (each with recorder=...)")
    rec_s1 = _run_with_recorder(
        "S1_fixed_sign", s1_fixed_sign.run,
        fire_scenario_id=args.fire, fds_dir=fds_dir,
        n_persons=args.n_persons, seed=args.seed,
        t_end_s=args.t_end, dt_s=args.dt,
    )
    rec_s2 = _run_with_recorder(
        "S2_fds_swarm", s2_fds_swarm.run,
        fire_scenario_id=args.fire, fds_dir=fds_dir,
        n_persons=args.n_persons, seed=args.seed,
        t_end_s=args.t_end, dt_s=args.dt,
    )
    rec_s3 = _run_with_recorder(
        "S3_fno_swarm", s3_fno_swarm.run,
        fire_scenario_id=args.fire, fds_dir=fds_dir,
        n_persons=args.n_persons, seed=args.seed,
        t_end_s=args.t_end, dt_s=args.dt,
    )

    # ── Render figures ─────────────────────────────────────────────
    render_risk = None if args.no_risk_overlay else truth_rm
    risk_at_t = None if args.no_risk_overlay else args.t_end

    print(f"\n[render] writing to {out_dir}/")
    paths = []
    paths.append(render_trajectories(
        rec_s1, out_dir / "s1_trajectories.png",
        fluid_mask=fluid_mask, risk_map=render_risk, risk_at_t=risk_at_t,
    ))
    paths.append(render_trajectories(
        rec_s2, out_dir / "s2_trajectories.png",
        fluid_mask=fluid_mask, risk_map=render_risk, risk_at_t=risk_at_t,
    ))
    paths.append(render_trajectories(
        rec_s3, out_dir / "s3_trajectories.png",
        fluid_mask=fluid_mask, risk_map=render_risk, risk_at_t=risk_at_t,
    ))
    paths.append(render_comparison(
        [rec_s1, rec_s2, rec_s3], out_dir / "comparison.png",
        fluid_mask=fluid_mask, risk_map=render_risk, risk_at_t=risk_at_t,
    ))
    for p in paths:
        print(f"  wrote {p}  ({p.stat().st_size/1024:.0f} KB)")

    if args.animate:
        print(f"\n[animate] writing GIFs ({args.fps} fps, may take a minute)")
        for label, rec in (
            ("s1", rec_s1), ("s2", rec_s2), ("s3", rec_s3),
        ):
            t0 = time.perf_counter()
            gif = render_animation(
                rec, out_dir / f"{label}.gif",
                fluid_mask=fluid_mask,
                risk_map=render_risk,
                fps=args.fps,
            )
            dt = time.perf_counter() - t0
            print(
                f"  wrote {gif}  ({gif.stat().st_size/1024:.0f} KB, "
                f"{dt:.1f}s wall)"
            )

    print(f"\nDone -- see {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
