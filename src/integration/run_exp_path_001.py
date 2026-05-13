"""EXP-PATH-001 entry point — sweep 3 scenarios × N fires × M seeds (D-025).

This is the **PyBullet** experiment runner. It complements the
``experiments/exp_path_001_compare_paths.py`` skeleton at the project
root, which delegates here once ``src/integration/`` is fleshed out.

Sweep:

* 3 scenarios — S1 (fixed sign) / S2 (FDS swarm) / S3 (PI-FNO swarm).
* ``len(fire_scenarios)`` FDS folders under ``data/raw/`` (default 3).
* ``n_seeds`` random seeds per (scenario, fire) cell — controls
  PersonAgent start scatter.

Total PyBullet runs = ``3 · len(fire_scenarios) · n_seeds``. Each run
internally simulates ``n_persons`` agents (default 20).

Outputs:

* ``<output>/comparison.csv`` — one row per run via
  :func:`src.integration.metrics.write_metrics_csv`.
* Console: per-row summary + H6 verdict via
  :func:`src.integration.metrics.h6_verdict`.

**Status: skeleton.** The dispatch loop is implemented (delegates to
each scenario module) so that adding a working scenario module
immediately produces real rows in the CSV; until the scenarios are
implemented, each call raises NotImplementedError.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

from src.integration.metrics import (
    MetricsCollector,
    ScenarioMetrics,
    h6_verdict,
    write_metrics_csv,
)
from src.integration.scenarios import s1_fixed_sign, s2_fds_swarm, s3_fno_swarm


def run_sweep(
    fire_scenarios: Sequence[str],
    n_persons: int,
    n_seeds: int,
    fno_checkpoint: Path,
    dataset_path: Path,
    raw_dir: Path,
    output_dir: Path,
) -> List[ScenarioMetrics]:
    """Iterate (scenario, fire, seed) and collect :class:`ScenarioMetrics` rows.

    Args:
        fire_scenarios: FDS scenario folder names under ``raw_dir``.
        n_persons: PersonAgents per run.
        n_seeds: Seeds per (scenario, fire) cell.
        fno_checkpoint: PI-FNO weights (used only by S3).
        dataset_path: Processed HDF5 (used only by S3).
        raw_dir: Parent of FDS scenario folders.
        output_dir: Where to write ``comparison.csv`` (created if absent).

    Returns:
        Full list of rows collected.

    Raises:
        FileNotFoundError: If any FDS folder is missing.
        NotImplementedError: While individual scenario modules are
            still skeletons.
    """
    rows: List[ScenarioMetrics] = []
    for fire in fire_scenarios:
        fds_dir = raw_dir / fire
        if not fds_dir.exists():
            raise FileNotFoundError(f"FDS scenario dir not found: {fds_dir}")
        for seed in range(n_seeds):
            rows.append(
                s1_fixed_sign.run(
                    fire_scenario_id=fire,
                    fds_dir=fds_dir,
                    n_persons=n_persons,
                    seed=seed,
                )
            )
            rows.append(
                s2_fds_swarm.run(
                    fire_scenario_id=fire,
                    fds_dir=fds_dir,
                    n_persons=n_persons,
                    seed=seed,
                )
            )
            rows.append(
                s3_fno_swarm.run(
                    fire_scenario_id=fire,
                    fds_dir=fds_dir,
                    fno_checkpoint=fno_checkpoint,
                    dataset_path=dataset_path,
                    n_persons=n_persons,
                    seed=seed,
                )
            )
    write_metrics_csv(rows, output_dir / "comparison.csv")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EXP-PATH-001 PyBullet runner (D-025). Sweep 3 scenarios x N fires x M seeds."
    )
    parser.add_argument(
        "--fire-scenarios",
        nargs="+",
        default=[
            "sim_1500kw_2m2_T05",
            "sim_500kw_1m2_T01",
            "sim_1000kw_1m2_T03",
        ],
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
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/exp_path_001"),
    )
    args = parser.parse_args()

    rows = run_sweep(
        fire_scenarios=args.fire_scenarios,
        n_persons=args.n_persons,
        n_seeds=args.n_seeds,
        fno_checkpoint=args.fno_checkpoint,
        dataset_path=args.data,
        raw_dir=args.raw_dir,
        output_dir=args.output,
    )

    print(f"\nWrote {len(rows)} rows -> {args.output / 'comparison.csv'}")
    for r in rows:
        print("  " + r.summary_line())

    verdict = h6_verdict(rows)
    print("\nH6 verdict:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
