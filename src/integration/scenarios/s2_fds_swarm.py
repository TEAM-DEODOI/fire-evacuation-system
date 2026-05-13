"""Scenario S2 — Drone swarm guided by FDS-truth RiskMap (D-025).

Same Scene and 20 :class:`PersonAgent` as S1, but the waypoint provider
is :class:`~src.integration.drone_swarm.DroneSwarm` consuming a
:class:`StaticRiskMap` built from the scenario's FDS slices
(``StaticRiskMap.from_fds_dir(fds_dir)``). Drones replan every 30 s and
fly 2 m ahead of each shepherded person.

**S1 vs S2 → H6** ("dynamic guidance reduces FED ≥ 30 %").

**Status: skeleton.** Run loop pending.
"""
from __future__ import annotations

from pathlib import Path

from src.integration.metrics import ScenarioMetrics


def run(
    fire_scenario_id: str,
    fds_dir: Path,
    n_persons: int = 20,
    n_drones: int = 3,
    seed: int = 0,
    t_end_s: float = 300.0,
    dt_s: float = 1.0,
) -> ScenarioMetrics:
    """Execute one S2 run.

    Args:
        fire_scenario_id: FDS scenario folder name under ``data/raw/``.
        fds_dir: Absolute path to the scenario dir.
        n_persons: Number of PersonAgents. D-025 default 20.
        n_drones: Drone swarm size. Default 3.
        seed: RNG seed for starting positions.
        t_end_s: Maximum simulation time.
        dt_s: Outer-loop step.

    Returns:
        :class:`ScenarioMetrics` with scenario_id="S2_fds_swarm".

    Raises:
        NotImplementedError: Pending implementation.
    """
    raise NotImplementedError(
        "Week 12 M4/M5: Scene -> 20 PersonAgent -> StaticRiskMap.from_fds_dir "
        "-> EvacuationPlanner -> DroneSwarm(n_drones) -> outer loop -> metrics."
    )


if __name__ == "__main__":
    print("s2_fds_swarm.py - skeleton")
    print("SKIP")
