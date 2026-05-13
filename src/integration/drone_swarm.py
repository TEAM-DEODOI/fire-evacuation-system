"""Drone swarm for S2/S3 active-guidance scenarios (D-025 Week 12).

Each drone is a Crazyflie quadrotor from ``gym-pybullet-drones``. The
swarm is a small population (~3-5 drones) that:

1. Detects isolated / lagging :class:`PersonAgent` instances via XY
   distance + status filter.
2. Computes a safe evacuation path for each shepherded agent using
   :class:`~src.path_planning.planners.EvacuationPlanner` with the
   scenario's :class:`RiskMap` (FDS-truth for S2, PI-FNO predictions
   for S3).
3. Flies ahead of the assigned agent (configurable lead distance,
   default 2 m, default altitude 1.8 m) to visually telegraph the next
   waypoint.
4. Coordinates with peers via a lightweight Boids/APF rule so two
   drones do not converge on the same agent.

Per :func:`WaypointProvider`, the swarm exposes
:meth:`DroneSwarm.next_waypoint(agent_position, t)` so persons can pull
the same way they would from a static sign network — the agent does
not need to know which scenario it is in.

**Status: skeleton.** Drone bodies + swarm coordination + waypoint
delivery to be implemented post Milestone 3 (URDF + scene + person agents
first). Interface frozen so scenarios can be drafted in parallel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx
import numpy as np

from src.path_planning.planners import EvacuationPlanner
from src.risk_map.risk_map_class import RiskMap


# ─── Config ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DroneSwarmConfig:
    """Top-level swarm hyperparameters.

    Attributes:
        n_drones: Number of drones in the swarm.
        flight_altitude_m: Constant Z for waypoint-following.
        lead_distance_m: How far ahead of the shepherded agent the drone
            flies (visual cue).
        max_speed_mps: Velocity cap for waypoint-following.
        assignment_radius_m: A drone considers any ALIVE agent within
            this XY radius for shepherding.
        replan_period_s: Cadence at which each shepherded agent's path
            is recomputed via :class:`EvacuationPlanner`.
        crazyflie_urdf: Optional path override; defaults to the URDF
            bundled with ``gym-pybullet-drones``.
    """

    n_drones: int = 3
    flight_altitude_m: float = 1.8
    lead_distance_m: float = 2.0
    max_speed_mps: float = 3.0
    assignment_radius_m: float = 8.0
    replan_period_s: float = 30.0
    crazyflie_urdf: Optional[str] = None


# ─── DroneSwarm ───────────────────────────────────────────────────────────
@dataclass
class DroneSwarm:
    """Multi-drone shepherd that fulfils the :class:`WaypointProvider` protocol.

    Holds one shared :class:`EvacuationPlanner` (the building graph is
    immutable), one shared :class:`RiskMap` (FDS for S2, PI-FNO for S3),
    and a per-agent assignment table. Per outer tick:

    1. Reassign drones to ALIVE agents (greedy nearest match).
    2. For each agent whose cached path has expired, re-plan.
    3. Step every drone one tick along its current waypoint.

    Attributes:
        client: PyBullet client id (set by :meth:`spawn`).
        config: :class:`DroneSwarmConfig`.
        planner: Shared :class:`EvacuationPlanner`.
        risk_map: Shared :class:`RiskMap`.
        graph: Building graph (passed for completeness — planner already
            holds it).
        drone_body_ids: PyBullet body ids in order.
        assignments: ``{agent_id: drone_idx}`` mapping.
    """

    config: DroneSwarmConfig
    planner: EvacuationPlanner
    risk_map: RiskMap
    graph: nx.Graph
    client: int = -1
    drone_body_ids: List[int] = field(default_factory=list)
    assignments: dict = field(default_factory=dict)
    _agent_paths: dict = field(default_factory=dict)
    _last_replan_t: dict = field(default_factory=dict)

    # ─── Lifecycle ─────────────────────────────────────────────────────
    def spawn(self, client: int, hover_positions: List[np.ndarray]) -> None:
        """Load ``n_drones`` Crazyflie URDFs at the given hover poses.

        Args:
            client: PyBullet client id.
            hover_positions: ``[xyz, ...]`` initial XYZ for each drone
                (length must equal ``config.n_drones``).

        Raises:
            NotImplementedError: Pending implementation.
            ValueError: If ``len(hover_positions) != config.n_drones``.
        """
        raise NotImplementedError(
            "Week 12 M4: load gym-pybullet-drones Crazyflie URDF n times."
        )

    def update(self, t: float, alive_agents: list) -> None:
        """Step the swarm one outer tick.

        Args:
            t: Simulation time (s).
            alive_agents: List of :class:`PersonAgent` with status ALIVE.

        Raises:
            NotImplementedError: Pending implementation.
        """
        raise NotImplementedError(
            "Week 12 M4: assignment, replan, kinematic step of drone."
        )

    # ─── WaypointProvider protocol ─────────────────────────────────────
    def next_waypoint(
        self,
        agent_position: np.ndarray,
        t: float,
    ) -> Optional[np.ndarray]:
        """Return the next path waypoint for the agent at ``agent_position``.

        If the agent has been assigned a drone with a cached path, return
        the next un-reached waypoint along that path. Otherwise return
        ``None`` (agent should stay put / fall back to sign logic).

        Raises:
            NotImplementedError: Pending implementation.
        """
        raise NotImplementedError(
            "Week 12 M4: look up agent assignment + serve cached waypoint."
        )


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("drone_swarm.py - skeleton (Week 12 M4 implementation pending)")
    print("SKIP")
