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
import math
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
    render_comparison_animation,
    render_trajectories,
)
from src.integration.scenarios import (
    s1_fixed_sign, s2_fds_swarm, s3_fno_swarm, s4_ensemble_swarm,
)
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
):
    """Run one scenario with recording. Returns (recorder, metrics)."""
    rec = SimulationRecorder(
        scenario_id=label,
        fire_scenario_id=fire_scenario_id,
        seed=seed,
    )
    t0 = time.perf_counter()
    metrics = runner(
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
    return rec, metrics


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
        "--t-start", type=float, default=0.0,
        help=(
            "Fire-clock offset (s). Simulation begins at this point in "
            "the fire's timeline so the smoke is already developed. "
            "Effective sim duration = t_end - t_start."
        ),
    )
    parser.add_argument(
        "--replan-period", type=float, default=None,
        help=(
            "Drone planner replan period (s). Default uses each "
            "scenario's own default (REPLAN_PERIOD_S = 30 s)."
        ),
    )
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
    replan_extras = (
        {"replan_period_s": float(args.replan_period)}
        if args.replan_period is not None else {}
    )
    print(
        f"\n[run] executing 4 scenarios (each with recorder=...)"
        f"  replan_period={'default' if not replan_extras else replan_extras['replan_period_s']}"
    )
    rec_s1, m_s1 = _run_with_recorder(
        "S1_fixed_sign", s1_fixed_sign.run,
        fire_scenario_id=args.fire, fds_dir=fds_dir,
        n_persons=args.n_persons, seed=args.seed,
        t_end_s=args.t_end, dt_s=args.dt,
        t_start_s=args.t_start,
    )
    rec_s2, m_s2 = _run_with_recorder(
        "S2_fds_swarm", s2_fds_swarm.run,
        fire_scenario_id=args.fire, fds_dir=fds_dir,
        n_persons=args.n_persons, seed=args.seed,
        t_end_s=args.t_end, dt_s=args.dt,
        t_start_s=args.t_start,
        **replan_extras,
    )
    rec_s3, m_s3 = _run_with_recorder(
        "S3_fno_swarm", s3_fno_swarm.run,
        fire_scenario_id=args.fire, fds_dir=fds_dir,
        n_persons=args.n_persons, seed=args.seed,
        t_end_s=args.t_end, dt_s=args.dt,
        t_start_s=args.t_start,
        **replan_extras,
    )
    rec_s4, m_s4 = _run_with_recorder(
        "S4_ensemble_swarm", s4_ensemble_swarm.run,
        fire_scenario_id=args.fire, fds_dir=fds_dir,
        n_persons=args.n_persons, seed=args.seed,
        t_end_s=args.t_end, dt_s=args.dt,
        t_start_s=args.t_start,
        **replan_extras,
    )

    # ── Metrics table ──────────────────────────────────────────────
    print("\n[metrics] EXP-PATH-001 5-metric summary")
    header = (
        f"  {'scenario':<20} {'evac %':>7} {'t_evac':>7} {'expos':>7} "
        f"{'dead %':>7} {'FED_mean':>10} {'FED_p90':>10} {'FED_max':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, m in (("S1_fixed_sign", m_s1),
                     ("S2_fds_swarm", m_s2),
                     ("S3_fno_swarm", m_s3),
                     ("S4_ensemble_swarm", m_s4)):
        evac_pct = m.evacuation_success_rate * 100.0
        t_evac = m.mean_evacuation_time_s
        t_str = f"{t_evac:7.2f}" if not math.isnan(t_evac) else f"{'nan':>7}"
        print(
            f"  {label:<20} {evac_pct:6.1f}% {t_str} "
            f"{m.danger_zone_exposure_time_s:6.2f}s "
            f"{m.casualty_rate*100:6.1f}% "
            f"{m.cumulative_fed:10.5f} "
            f"{m.p90_cumulative_fed:10.5f} "
            f"{m.max_cumulative_fed:10.5f}"
        )

    # ── H6 verdict (FED ratios vs S1) ───────────────────────────────
    if m_s1.cumulative_fed > 0:
        r_s2 = m_s2.cumulative_fed / m_s1.cumulative_fed
        r_s3 = m_s3.cumulative_fed / m_s1.cumulative_fed
        r_s4 = m_s4.cumulative_fed / m_s1.cumulative_fed
        print(
            f"\n  H6 ratios (target <= 0.7 = 30% FED reduction vs S1):\n"
            f"    S2/S1 = {r_s2:.3f}   "
            f"S3/S1 = {r_s3:.3f}   "
            f"S4/S1 = {r_s4:.3f}   "
            f"H6 pass = {(r_s2 <= 0.7) and (r_s3 <= 0.7) and (r_s4 <= 0.7)}"
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
    paths.append(render_trajectories(
        rec_s4, out_dir / "s4_trajectories.png",
        fluid_mask=fluid_mask, risk_map=render_risk, risk_at_t=risk_at_t,
    ))
    paths.append(render_comparison(
        [rec_s1, rec_s2, rec_s3, rec_s4], out_dir / "comparison.png",
        fluid_mask=fluid_mask, risk_map=render_risk, risk_at_t=risk_at_t,
    ))
    for p in paths:
        print(f"  wrote {p}  ({p.stat().st_size/1024:.0f} KB)")

    if args.animate:
        print(f"\n[animate] writing GIFs ({args.fps} fps, may take a minute)")
        for label, rec in (
            ("s1", rec_s1), ("s2", rec_s2),
            ("s3", rec_s3), ("s4", rec_s4),
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
        # 4-panel side-by-side GIF -- most informative for H6 reading.
        t0 = time.perf_counter()
        gif = render_comparison_animation(
            [rec_s1, rec_s2, rec_s3, rec_s4],
            out_dir / "comparison.gif",
            fluid_mask=fluid_mask,
            risk_map=render_risk,
            fps=args.fps,
            dt_s=args.dt,
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
