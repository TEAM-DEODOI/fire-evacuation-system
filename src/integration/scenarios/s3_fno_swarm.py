"""Scenario S3 — Drone swarm guided by PI-FNO predictions (D-025).

Identical to S2 except the RiskMap is :class:`FNORiskMap` (or, in the
Tier-1 paper-headline variant, :class:`Tier1RiskMap`) built from the
model's prediction of the scenario's initial frame, rather than from
FDS ground truth.

The occupant's experienced danger is still measured against
``StaticRiskMap.from_fds_dir(...)`` (the truth), so the only thing that
changes between S2 and S3 is what the *planner sees*. This is the
fairness setup from ``docs/interface_contracts.md`` §6.

**S2 vs S3 → H5 transitive** ("does risk-map fidelity translate to path
quality?"). If S3 ≈ S2, then a model that hits H5 on its own is also
deployment-ready as a planner input.

**Status: skeleton.** Run loop pending.
"""
from __future__ import annotations

from pathlib import Path

from src.integration.metrics import ScenarioMetrics


def run(
    fire_scenario_id: str,
    fds_dir: Path,
    fno_checkpoint: Path,
    dataset_path: Path,
    n_persons: int = 20,
    n_drones: int = 3,
    seed: int = 0,
    t_end_s: float = 300.0,
    dt_s: float = 1.0,
) -> ScenarioMetrics:
    """Execute one S3 run.

    Args:
        fire_scenario_id: FDS scenario folder name.
        fds_dir: Scenario dir (used only for ground-truth RiskMap).
        fno_checkpoint: PI-FNO weights.
        dataset_path: Processed HDF5 (initial frames + masks).
        n_persons: Number of PersonAgents. D-025 default 20.
        n_drones: Drone swarm size.
        seed: RNG seed.
        t_end_s: Max simulation time.
        dt_s: Outer-loop step.

    Returns:
        :class:`ScenarioMetrics` with scenario_id="S3_fno_swarm".

    Raises:
        NotImplementedError: Pending implementation.
    """
    raise NotImplementedError(
        "Week 12 M4/M5: load PI-FNO + initial frame -> FNORiskMap (planner side) "
        "+ StaticRiskMap.from_fds_dir (truth side) -> EvacuationPlanner "
        "-> DroneSwarm -> outer loop with split planner/truth maps -> metrics."
    )


if __name__ == "__main__":
    print("s3_fno_swarm.py - skeleton")
    print("SKIP")
