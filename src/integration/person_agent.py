"""Simplified PersonAgent for PyBullet evacuation scenarios (D-025 Week 12).

PersonAgent is intentionally minimal per ``CLAUDE.md`` *"What This Project
Does NOT Do"* — no panic-speed boost, no social-force model, no crowd
density effects. The agent moves at constant 1.2 m/s toward the current
waypoint (delivered by the surrounding scenario module), checks
PyBullet contact for wall avoidance, and exposes a 3-state status:

* ``alive``     — currently moving toward a waypoint.
* ``evacuated`` — reached an exit. Stops contributing to FED accumulation.
* ``dead``      — cumulative FED exceeded the ISO-13571 sensitive-population
                  threshold (0.3) OR experienced danger above
                  ``TENABILITY.AGGREGATE_THRESHOLD`` for longer than the
                  configured grace window. Stops moving.

The waypoint *source* is what differentiates scenarios:

* **S1 fixed-sign** — agent queries the static sign network for the
  nearest non-hazardous sign and uses its position as the next waypoint.
* **S2 / S3 drone swarm** — agent receives waypoints from
  :class:`DroneSwarm`, which runs weighted A* per affected agent.

This module owns *only* the agent's body + state machine. The waypoint
provider is injected at update time so the same agent class plugs into
all three scenarios.

**Status: skeleton.** Body / state machine / status transitions to be
implemented in the next commit. The interface is fixed (kept stable so
scenarios can be drafted in parallel).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

import numpy as np

from src.shared.constants import TENABILITY


# ─── Status enum ──────────────────────────────────────────────────────────
class PersonStatus(str, Enum):
    """3-state lifecycle per D-025."""

    ALIVE = "alive"
    EVACUATED = "evacuated"
    DEAD = "dead"


# ─── Waypoint provider protocol ───────────────────────────────────────────
class WaypointProvider(Protocol):
    """Anything that can hand a next waypoint to a PersonAgent.

    Implemented by both the fixed-sign network (S1) and the drone swarm
    (S2/S3). Returning ``None`` means "stay put" (e.g. all signs are
    hazardous, no drone is currently shepherding this agent).
    """

    def next_waypoint(
        self,
        agent_position: np.ndarray,
        t: float,
    ) -> Optional[np.ndarray]:
        ...


# ─── Config ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PersonAgentConfig:
    """Per-agent constants — fixed at construction.

    Attributes:
        walking_speed_mps: Constant speed. D-025 specifies 1.2 m/s; the
            logical sim used 1.5 m/s (D-017). PyBullet sim uses 1.2 m/s
            for D-025 alignment.
        body_radius_m: Capsule radius for wall collision.
        body_height_m: Capsule total height (head-of-shoulder ~1.5 m).
        fed_threshold: ``TENABILITY.FED_THRESHOLD`` (sensitive-pop).
        instant_death_danger: Danger value above which the agent
            transitions to ``dead`` immediately (e.g. trapped in
            flashover).
        arrival_tolerance_m: Distance to a waypoint that counts as reached.
    """

    walking_speed_mps: float = 1.2
    body_radius_m: float = 0.25
    body_height_m: float = 1.7
    fed_threshold: float = TENABILITY.FED_THRESHOLD  # 0.3
    instant_death_danger: float = 0.99
    arrival_tolerance_m: float = 0.5


# ─── PersonAgent ──────────────────────────────────────────────────────────
@dataclass
class PersonAgent:
    """One occupant in the PyBullet evacuation scenario.

    The agent body is a single capsule. Its kinematics are deliberately
    simple (kinematic motion, not full physics): each call to
    :meth:`update` translates the body toward the active waypoint by
    ``walking_speed_mps · dt``, after PyBullet collision queries veto
    moves that would penetrate the building mesh.

    Attributes:
        agent_id: Unique string identifier.
        body_id: PyBullet body id (assigned by :meth:`spawn`).
        position: ``(3,)`` current world position (m).
        status: :class:`PersonStatus`.
        cumulative_fed: Running FED accumulator (CO-derived; only the
            scenario module knows the CO grid, so it pushes updates).
        config: Per-agent constants.
    """

    agent_id: str
    body_id: int = -1
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    status: PersonStatus = PersonStatus.ALIVE
    cumulative_fed: float = 0.0
    config: PersonAgentConfig = field(default_factory=PersonAgentConfig)

    def spawn(self, client: int, start_xyz: np.ndarray) -> None:
        """Create the capsule body in PyBullet at ``start_xyz``.

        Args:
            client: PyBullet physics client id (from :class:`Scene`).
            start_xyz: Initial position (m).

        Raises:
            NotImplementedError: Pending the next implementation commit.
        """
        raise NotImplementedError(
            "Week 12 M2: build capsule via createCollisionShape + "
            "createMultiBody; store body id; set self.position."
        )

    def update(
        self,
        client: int,
        dt_s: float,
        provider: WaypointProvider,
        experienced_danger: float,
        experienced_co_ppm: float,
    ) -> None:
        """Advance the agent by one outer simulation tick.

        Steps:
        1. If ``status != ALIVE``, return (no motion, no FED accumulation).
        2. Accumulate FED from ``experienced_co_ppm`` over ``dt_s``.
        3. Transition to ``dead`` if FED ≥ threshold or
           ``experienced_danger ≥ instant_death_danger``.
        4. Ask ``provider`` for the next waypoint; if ``None``, stay put.
        5. Move toward the waypoint by ``walking_speed_mps · dt_s``,
           respecting PyBullet collision; transition to ``evacuated`` if
           waypoint is an exit and within ``arrival_tolerance_m``.

        Args:
            client: PyBullet physics client id.
            dt_s: Outer-loop step in seconds.
            provider: Source of the next waypoint.
            experienced_danger: ``risk_map_truth.query`` value at the
                agent's current position (passed by the scenario module).
            experienced_co_ppm: Raw CO concentration at the agent's
                position (also from the scenario module).

        Raises:
            NotImplementedError: Pending the next implementation commit.
        """
        raise NotImplementedError(
            "Week 12 M2: implement status machine + waypoint stepping + FED."
        )

    @property
    def alive(self) -> bool:
        return self.status == PersonStatus.ALIVE

    @property
    def evacuated(self) -> bool:
        return self.status == PersonStatus.EVACUATED

    @property
    def dead(self) -> bool:
        return self.status == PersonStatus.DEAD


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("person_agent.py - skeleton (Week 12 M2 implementation pending)")
    print("SKIP")
