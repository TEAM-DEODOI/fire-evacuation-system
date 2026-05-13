"""Scenario S1 — Fixed-sign baseline (D-025 H6 baseline).

Setup:

* 20 :class:`~src.integration.person_agent.PersonAgent` spawned at
  random positions inside the building.
* Static "exit signs" pre-placed at corridor intersections that show
  the nearest exit assuming no fire.
* No drones. Persons individually query
  :class:`~src.risk_map.risk_map_class.RiskMap` (FDS truth here) to
  detect hazard and choose the nearest *non-hazardous* sign as their
  next waypoint.
* FED accumulation, status transitions, and metric collection happen
  identically to S2/S3 — only the WaypointProvider differs.

This is the **H6 baseline**: if drone-swarm guidance (S2) gives at
least a 30 % FED reduction relative to this, H6 passes.

**Status: skeleton.** Run loop pending.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.integration.metrics import ScenarioMetrics
from src.risk_map.risk_map_class import RiskMap


# ─── Fixed-sign provider ──────────────────────────────────────────────────
@dataclass
class FixedSignNetwork:
    """The static-sign equivalent of a :class:`WaypointProvider`.

    The network is a fixed list of sign positions and, for each sign, a
    list of candidate exit waypoints in priority order. When a person
    queries :meth:`next_waypoint`, the network returns the nearest sign
    whose pointed-at exit is currently non-hazardous (per the FDS truth
    :class:`RiskMap` it holds). If all signs are hazardous (worst case),
    returns the absolute-nearest exit and lets the person take its chances.
    """

    risk_map: RiskMap
    sign_positions: List[np.ndarray] = field(default_factory=list)
    sign_to_exit: dict = field(default_factory=dict)
    danger_threshold: float = 0.5

    def next_waypoint(
        self,
        agent_position: np.ndarray,
        t: float,
    ) -> Optional[np.ndarray]:
        """Implements :class:`WaypointProvider`.

        Raises:
            NotImplementedError: pending implementation.
        """
        raise NotImplementedError(
            "Week 12 M2/S1: nearest-sign + RiskMap-aware exit choice."
        )


# ─── Run loop ─────────────────────────────────────────────────────────────
def run(
    fire_scenario_id: str,
    fds_dir: Path,
    n_persons: int = 20,
    seed: int = 0,
    t_end_s: float = 300.0,
    dt_s: float = 1.0,
) -> ScenarioMetrics:
    """Execute one S1 run and return its :class:`ScenarioMetrics` row.

    Args:
        fire_scenario_id: FDS scenario folder name under ``data/raw/``.
        fds_dir: Absolute path to the scenario dir (FDS .sf/.smv files).
        n_persons: Number of PersonAgents. D-025 default 20.
        seed: RNG seed for starting positions.
        t_end_s: Maximum simulation time.
        dt_s: Outer-loop step.

    Returns:
        :class:`ScenarioMetrics` with scenario_id="S1_fixed_sign".

    Raises:
        NotImplementedError: Pending the next implementation commit.
    """
    raise NotImplementedError(
        "Week 12 M5: build Scene -> spawn 20 PersonAgent -> FixedSignNetwork "
        "-> outer loop with FED + status transitions -> aggregate via "
        "MetricsCollector -> return ScenarioMetrics."
    )


if __name__ == "__main__":
    print("s1_fixed_sign.py - skeleton")
    print("SKIP")
