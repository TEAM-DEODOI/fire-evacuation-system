"""
Experiment: EXP-PATH-001 — three PyBullet evacuation scenarios.

> **Scope change 2026-05-14 (D-025)**: This script's *file name* still
> contains the legacy phrase "compare_paths" but the experiment no longer
> compares path-planning algorithms. After D-025 the comparison is between
> three end-to-end PyBullet scenarios — see ``docs/decisions.md::D-025``
> and ``CLAUDE.md`` *"EXP-PATH-001 — Three Scenarios"*. A rename to
> ``exp_path_001_three_scenarios.py`` is the recommended follow-up.

| Scenario | Guidance | RiskMap source | H roles |
|----------|----------|----------------|---------|
| S1 | Fixed-sign baseline (static signs, persons individually navigate) | FDS | H6 baseline |
| S2 | Drone swarm guidance via weighted A* waypoints | FDS | S1 vs S2 → H6 |
| S3 | Drone swarm guidance via weighted A* waypoints | PI-FNO | S2 vs S3 → H5 transitivity |

Each scenario runs with 20 simplified ``PersonAgent`` instances
(1.2 m/s, ``alive → evacuated / dead``) and reports five metrics
(`evacuation_success_rate`, `mean_evacuation_time`,
`danger_zone_exposure_time`, `casualty_rate`, `cumulative_FED`). The
**H6 primary metric** is mean cumulative FED — target reduction
``S2 ≤ 0.7 · S1``.

Status: **skeleton only**. Real implementation depends on the
``src/integration/`` modules (PyBullet env, PersonAgent, drone swarm),
which are still pending — see ``docs/90_next_steps.md`` §2.2 Step B.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def main(
    fire_scenarios: Sequence[str] = (
        "sim_1500kw_2m2_T05",
        "sim_500kw_1m2_T01",
        "sim_1000kw_1m2_T03",
    ),
    n_persons: int = 20,
    n_seeds: int = 5,
    fno_checkpoint: Path = Path("checkpoints/fno_pi/best.pt"),
    dataset_path: Path = Path("data/processed/dataset.h5"),
    output_dir: Path = Path("results/exp_path_001"),
) -> None:
    """Run the three-scenario H6 verification experiment.

    Args:
        fire_scenarios: FDS scenario IDs (one StaticRiskMap per scenario).
            Defaults sample HRR levels 500/1000/1500 kW.
        n_persons: Number of PersonAgent instances per PyBullet run.
        n_seeds: Number of stochastic seeds per (scenario × fire) cell.
            Total runs = 3 × len(fire_scenarios) × n_seeds.
        fno_checkpoint: PI-FNO weights for S3.
        dataset_path: Processed HDF5 (initial frames + masks).
        output_dir: Destination for ``comparison.csv`` plus per-scenario
            ``*.npz`` per-person trajectories.

    Raises:
        NotImplementedError: Until ``src/integration/`` is wired up.
    """
    raise NotImplementedError(
        "Week 12: depends on src/integration/{scene,person_agent,drone_swarm}. "
        "See docs/90_next_steps.md §2.2 Step B and docs/decisions.md::D-025."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "EXP-PATH-001: three-scenario PyBullet evacuation comparison "
            "(H6 + H5-transitive). See docs/decisions.md::D-025."
        )
    )
    parser.add_argument(
        "--fire-scenarios",
        nargs="+",
        default=[
            "sim_1500kw_2m2_T05",
            "sim_500kw_1m2_T01",
            "sim_1000kw_1m2_T03",
        ],
        help="FDS scenario IDs (one StaticRiskMap each).",
    )
    parser.add_argument("--n-persons", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--fno-checkpoint",
        type=Path,
        default=Path("checkpoints/fno_pi/best.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/dataset.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/exp_path_001"),
    )
    args = parser.parse_args()

    main(
        fire_scenarios=args.fire_scenarios,
        n_persons=args.n_persons,
        n_seeds=args.n_seeds,
        fno_checkpoint=args.fno_checkpoint,
        dataset_path=args.data,
        output_dir=args.output,
    )
