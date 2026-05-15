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

**Status (2026-05-14, M2-full):**

* :meth:`PersonAgent.spawn` — **functional** (kinematic capsule body).
* :meth:`PersonAgent.step_toward` — **functional**. Atomic kinematic
  motion step with fluid-mask / PyBullet wall avoidance.
* :meth:`PersonAgent.accumulate_exposure` — **functional**. Per-step
  CO-FED accumulation + ``DEAD`` transitions (FED ≥ 0.3 OR danger ≥
  0.99). Decoupled from motion so scenarios choose the ordering.
* :meth:`PersonAgent.mark_evacuated` — convenience setter used by the
  scenario when the agent reaches an exit.

The original monolithic ``update()`` method is removed — scenarios call
``step_toward`` + ``accumulate_exposure`` + ``mark_evacuated`` directly
in their run loop, which makes ordering / replanning logic explicit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import pybullet as p

from src.shared.constants import CELL_SIZE_M, TENABILITY
from src.risk_map.fed import accumulate_fed_co


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

    walking_speed_mps: float = 0.5
    """Constant occupant speed. Lowered from 1.2 → 0.5 m/s (D-034,
    2026-05-14) so the simulation wall clock is on a similar scale to
    fire spread (~minutes). At 1.2 m/s every occupant evacuated in
    5–15 s on the small building, completely outrunning the fire and
    flattening the S1-vs-drone FED gap."""
    body_radius_m: float = 0.25
    body_height_m: float = 1.7
    fed_threshold: float = TENABILITY.FED_THRESHOLD  # 0.3
    instant_death_danger: float = 0.99
    arrival_tolerance_m: float = 0.5
    follow_range_m: float = 8.0
    """How far the agent can see a GUIDING drone (D-030). Slightly larger
    than the drone's sense_range_m so the drone can lead a 1.2 m/s person
    without breaking the chain on every tick."""
    exit_proximity_m: float = 1.0
    """D-051 (2026-05-14): when ANY exit is within this XY radius the
    agent ignores its current GUIDING drone and heads straight to that
    exit. Solves the "drone hovers around the exit" loop where the
    occupant orbits with the drone instead of stepping into the exit
    cell. Default 1.0 m = 2 cells at CELL_SIZE_M=0.5 m."""
    path_replan_period_s: float = 1.0
    """Cadence at which the agent re-plans its own BFS path to the
    current waypoint source (D-032, 2026-05-14). Forced replan also
    happens whenever the target cell changes."""
    z_layer: int = 3
    """Grid z-slice the agent navigates on (k=3 = breathing zone, z=1.75 m)."""


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
    following_drone_id: Optional[str] = None
    """When non-None, the agent currently sees this GUIDING drone and
    walks toward its position (D-030). Cleared when the drone moves out
    of ``config.follow_range_m`` or stops GUIDING."""
    nav_path: List[np.ndarray] = field(default_factory=list)
    """BFS path (cell-by-cell) from the agent's current cell to the
    currently-tracked waypoint source (an exit for S1, the followed
    drone for S2/S3). Each element is a (3,) world coordinate at the
    cell center. See :meth:`replan_path_to`. (D-032, 2026-05-14)"""
    nav_wp_idx: int = 0
    """Index of the next not-yet-reached waypoint in :attr:`nav_path`."""
    last_path_replan_t: float = -1e9
    last_path_target_cell: Optional[Tuple[int, int]] = None
    config: PersonAgentConfig = field(default_factory=PersonAgentConfig)

    def spawn(self, client: int, start_xyz: np.ndarray) -> None:
        """Create a kinematic capsule body at ``start_xyz``.

        The capsule is mass-0 (static in PyBullet's solver) but its pose
        is updated each tick via :meth:`step_toward` ->
        ``resetBasePositionAndOrientation``. PyBullet's collision
        detection still reports contacts against this body, which is
        what :meth:`step_toward` uses to veto wall-penetrating moves.

        Capsule geometry: total height = ``config.body_height_m``,
        radius = ``config.body_radius_m``. The cylindrical middle section
        has height ``body_height_m - 2 * body_radius_m``. The body is
        centred at ``start_xyz`` in XY and lifted so its bottom
        hemisphere just rests on ``z=0`` (regardless of the Z coordinate
        passed in -- we keep occupants on the floor).

        Args:
            client: PyBullet physics client id (from :class:`Scene`).
            start_xyz: Initial position (m). Only XY are honoured; Z is
                derived from capsule geometry.

        Raises:
            ValueError: If ``start_xyz`` is not a 3-D vector.
            RuntimeError: If the agent has already been spawned.
        """
        arr = np.asarray(start_xyz, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"start_xyz must be (3,), got {arr.shape}")
        if self.body_id >= 0:
            raise RuntimeError(
                f"PersonAgent {self.agent_id!r} already spawned (body_id={self.body_id})"
            )

        radius = float(self.config.body_radius_m)
        # PyBullet GEOM_CAPSULE: `height` is the cylindrical middle length,
        # not the total length (hemispherical caps add 2 * radius).
        cyl_h = max(0.01, float(self.config.body_height_m) - 2.0 * radius)
        # Bottom of capsule sits on z=0.
        center_z = cyl_h / 2.0 + radius

        col_shape = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=radius,
            height=cyl_h,
            physicsClientId=client,
        )
        vis_shape = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=radius,
            length=cyl_h,
            rgbaColor=[0.20, 0.55, 0.95, 1.0],
            physicsClientId=client,
        )
        self.body_id = p.createMultiBody(
            baseMass=0.0,  # static body — kinematic via resetBasePosition
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[float(arr[0]), float(arr[1]), center_z],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=client,
        )
        self.position = np.array([arr[0], arr[1], center_z], dtype=np.float64)

    def step_toward(
        self,
        client: int,
        target_xyz: np.ndarray,
        dt_s: float,
        *,
        obstacle_body_ids: Optional[List[int]] = None,
        fluid_mask: Optional[np.ndarray] = None,
        walking_speed_mps: Optional[float] = None,
    ) -> bool:
        """Advance one tick toward ``target_xyz`` with collision veto.

        Two collision-check modes (mutually exclusive):

        * **fluid_mask mode** (D-026, preferred for cell-grid planning):
          pass ``fluid_mask=(60, 40, 6)`` boolean. The candidate XY is
          mapped to a cell index via ``world_to_grid``; the move is
          allowed iff that cell is fluid (``True``). Bypasses PyBullet
          collision -- accurate for any STL because it matches the
          canonical risk-map discretisation.
        * **PyBullet mode** (legacy, M2-mini placeholder URDF):
          pass ``obstacle_body_ids=[scene.building_id]``. The candidate
          pose is tentatively applied and ``getClosestPoints(distance=0)``
          rejects penetrating moves. Subject to mesh-collision quirks
          on real STL geometry (L-015 + see D-026 rationale).

        Algorithm:
        1. Compute step distance = ``walking_speed * dt_s`` (capped at
           the remaining XY distance to ``target_xyz``).
        2. Compute candidate position.
        3. Run the selected collision check.
        4. If blocked: do **not** update ``self.position``; return
           ``False``.
        5. Otherwise: update ``self.position`` (and the PyBullet body
           pose if there is one), return ``True``.

        Args:
            client: PyBullet physics client id.
            target_xyz: Goal position. Only XY are used.
            dt_s: Outer-loop step in seconds.
            obstacle_body_ids: PyBullet body ids that should block motion
                (typically ``[scene.building_id]``). Mutually exclusive
                with ``fluid_mask``.
            fluid_mask: ``(60, 40, 6)`` boolean array where ``True`` =
                navigable. Preferred over ``obstacle_body_ids`` (D-026).
            walking_speed_mps: Override the agent's configured speed for
                this step. Defaults to ``config.walking_speed_mps``.

        Returns:
            ``True`` if the move was applied (or already at target).
            ``False`` if blocked.

        Raises:
            RuntimeError: If :meth:`spawn` has not been called.
            ValueError: If ``target_xyz`` is not 3-D, ``dt_s <= 0``,
                or both / neither of ``obstacle_body_ids`` and
                ``fluid_mask`` are provided.
        """
        if self.body_id < 0:
            raise RuntimeError(
                f"PersonAgent {self.agent_id!r} not spawned; call spawn() first"
            )
        target = np.asarray(target_xyz, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError(f"target_xyz must be (3,), got {target.shape}")
        if dt_s <= 0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}")
        if (fluid_mask is None) == (obstacle_body_ids is None):
            raise ValueError(
                "step_toward requires exactly one of fluid_mask or "
                "obstacle_body_ids"
            )

        speed = (
            float(walking_speed_mps)
            if walking_speed_mps is not None
            else float(self.config.walking_speed_mps)
        )
        step_dist = speed * float(dt_s)

        delta_xy = target[:2] - self.position[:2]
        remaining = float(np.linalg.norm(delta_xy))
        if remaining <= self.config.arrival_tolerance_m:
            return True  # already arrived
        direction = delta_xy / remaining
        move_len = min(step_dist, remaining)
        candidate = np.array(
            [
                self.position[0] + direction[0] * move_len,
                self.position[1] + direction[1] * move_len,
                self.position[2],  # keep Z (capsule on floor)
            ],
            dtype=np.float64,
        )

        # ── Mode A: fluid-mask collision (D-026, preferred) ─────────────
        if fluid_mask is not None:
            from src.shared.coordinates import world_to_grid
            cell = world_to_grid(np.asarray(candidate))
            ix, iy, iz = int(cell[0]), int(cell[1]), int(cell[2])
            nx_, ny_, nz_ = fluid_mask.shape
            in_bounds = 0 <= ix < nx_ and 0 <= iy < ny_ and 0 <= iz < nz_
            if not in_bounds or not bool(fluid_mask[ix, iy, iz]):
                return False
            # Move accepted: update body pose (kinematic) + cached position.
            p.resetBasePositionAndOrientation(
                self.body_id,
                candidate.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=client,
            )
            self.position = candidate
            return True

        # ── Mode B: PyBullet contact-based collision (legacy) ───────────
        # ``getContactPoints`` does not report mass-0 vs mass-0 (L-015).
        # Use ``getClosestPoints(distance=0.0)`` which probes the current
        # pose directly regardless of body mass.
        p.resetBasePositionAndOrientation(
            self.body_id,
            candidate.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=client,
        )
        for other in obstacle_body_ids:
            if other == self.body_id:
                continue
            overlaps = p.getClosestPoints(
                bodyA=self.body_id,
                bodyB=other,
                distance=0.0,
                physicsClientId=client,
            )
            if overlaps:
                p.resetBasePositionAndOrientation(
                    self.body_id,
                    self.position.tolist(),
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=client,
                )
                return False

        self.position = candidate
        return True

    def accumulate_exposure(
        self,
        experienced_co_ppm: float,
        experienced_danger: float,
        dt_s: float,
    ) -> bool:
        """Accumulate one step of CO exposure and apply the status machine.

        Called by each scenario's outer loop AFTER computing the agent's
        post-motion position's CO + danger. The motion step itself
        (:meth:`step_toward`) is independent so a scenario can choose
        either ordering (move-then-expose, expose-then-move).

        State transitions:
        * If ``status != ALIVE`` → no-op, returns the current status.
        * Else: accumulate ``CO_ppm * (dt/60) / FED_REFERENCE`` onto
          :attr:`cumulative_fed` (ISO 13571 §7.3 simplified per D-008).
        * If ``cumulative_fed >= fed_threshold`` (0.3) → ``DEAD``.
        * If ``experienced_danger >= instant_death_danger`` (0.99,
          flashover proxy) → ``DEAD``.

        ``EVACUATED`` is set by the scenario itself when the agent
        reaches an exit (see ``arrival_tolerance_m`` check in S1/S2/S3
        run loops). This method never transitions to evacuated.

        Args:
            experienced_co_ppm: Raw CO at the agent's current position
                (ppm). Negative values are clamped to 0.
            experienced_danger: ``risk_map_truth.query`` value at the
                same position. Used only for the instant-death check.
            dt_s: Step length (seconds). Must be > 0.

        Returns:
            ``True`` iff a transition to ``DEAD`` was applied in this
            step (so callers can record the time of death).

        Raises:
            ValueError: If ``dt_s <= 0``.
        """
        if dt_s <= 0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}")
        if self.status != PersonStatus.ALIVE:
            return False

        # Single-step FED increment (ISO 13571 simplified).
        # Re-use accumulate_fed_co to keep the formula in one place; a
        # 1-element array returns a 1-element cumulative sum which is
        # just this step's contribution.
        delta = float(
            accumulate_fed_co(
                np.array([max(0.0, float(experienced_co_ppm))]),
                dt_seconds=float(dt_s),
            )[0]
        )
        self.cumulative_fed += delta

        # Death checks.
        if self.cumulative_fed >= self.config.fed_threshold:
            self.status = PersonStatus.DEAD
            return True
        if float(experienced_danger) >= self.config.instant_death_danger:
            self.status = PersonStatus.DEAD
            return True
        return False

    def mark_evacuated(self) -> None:
        """Transition ``ALIVE → EVACUATED`` (idempotent; no-op otherwise)."""
        if self.status == PersonStatus.ALIVE:
            self.status = PersonStatus.EVACUATED

    # ─── Self path planning (D-032, 2026-05-14) ────────────────────────
    def replan_path_to(
        self,
        target_xyz: np.ndarray,
        fluid_mask: np.ndarray,
        t: float = 0.0,
    ) -> None:
        """Compute a wall-aware BFS path from the agent's current cell to
        the target cell. Result is cached in :attr:`nav_path`.

        The BFS runs on the **fluid_mask** z-slice (k=3 by default).
        Walls/solid cells block the search, so the resulting path never
        crosses a wall. Each waypoint is a cell center (0.5 m apart).

        Args:
            target_xyz: World XYZ of the destination (e.g. an exit
                position for S1, or a GUIDING drone's position for S2/S3).
            fluid_mask: ``(60, 40, 6)`` boolean. ``True`` = navigable.
            t: Current simulation time (recorded for replan throttling).
        """
        z_layer = int(self.config.z_layer)
        new_path = _bfs_grid_path(
            start_xyz=self.position,
            target_xyz=target_xyz,
            mask=fluid_mask,
            z_layer=z_layer,
        )
        self.nav_path = new_path
        self.nav_wp_idx = 1 if len(new_path) > 1 else 0
        self.last_path_replan_t = float(t)
        self.last_path_target_cell = (
            int(float(target_xyz[0]) / float(CELL_SIZE_M)),
            int(float(target_xyz[1]) / float(CELL_SIZE_M)),
        )

    def clear_path(self) -> None:
        """Drop the cached navigation path (call when no waypoint source
        is currently visible, e.g. no GUIDING drone in sight)."""
        self.nav_path = []
        self.nav_wp_idx = 0
        self.last_path_target_cell = None

    def step_along_path(
        self,
        client: int,
        dt_s: float,
        fluid_mask: np.ndarray,
        walking_speed_mps: Optional[float] = None,
    ) -> bool:
        """Advance along :attr:`nav_path` by up to ``speed * dt`` metres.

        Consumes path waypoints one at a time (cell-by-cell, 0.5 m
        spacing) so each sub-move is short enough that the candidate
        cell can be sanity-checked against ``fluid_mask`` — no chance of
        a single long step crossing through a wall, which is the core
        of the wall-crossing bug the previous ``step_toward`` had.

        Args:
            client: PyBullet physics client id.
            dt_s: Outer-loop step (s). Must be > 0.
            fluid_mask: ``(60, 40, 6)`` boolean. Each sub-move's
                destination cell is verified to be ``True``.
            walking_speed_mps: Optional speed override.

        Returns:
            True iff the agent moved at least once this tick.
        """
        if dt_s <= 0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}")
        if self.status != PersonStatus.ALIVE:
            return False
        if not self.nav_path:
            return False

        speed = (
            float(walking_speed_mps)
            if walking_speed_mps is not None
            else float(self.config.walking_speed_mps)
        )
        budget_m = speed * float(dt_s)
        moved = False
        nx_, ny_, nz_ = fluid_mask.shape
        z_layer = int(self.config.z_layer)

        while budget_m > 1e-6 and self.nav_wp_idx < len(self.nav_path):
            wp = np.asarray(
                self.nav_path[self.nav_wp_idx], dtype=np.float64
            ).reshape(3)
            delta_xy = wp[:2] - self.position[:2]
            dist = float(np.linalg.norm(delta_xy))
            if dist <= 1e-6:
                self.nav_wp_idx += 1
                continue
            move_len = min(budget_m, dist)
            direction = delta_xy / dist
            new_xy = self.position[:2] + direction * move_len
            cx = int(new_xy[0] / CELL_SIZE_M)
            cy = int(new_xy[1] / CELL_SIZE_M)
            # Cell-level fluid check (wall veto).
            if not (0 <= cx < nx_ and 0 <= cy < ny_):
                # Out of grid: stop, but do not invalidate path -- next
                # tick may re-plan from a valid position.
                break
            if not bool(fluid_mask[cx, cy, z_layer]):
                # Should not happen on a valid BFS path; abort safely.
                break
            self.position = np.array(
                [new_xy[0], new_xy[1], self.position[2]],
                dtype=np.float64,
            )
            moved = True
            budget_m -= move_len
            if move_len >= dist - 1e-6:
                # Exactly reached this waypoint -- advance to the next.
                self.nav_wp_idx += 1

        # Push the kinematic body pose once at end of tick.
        if moved and self.body_id >= 0:
            p.resetBasePositionAndOrientation(
                self.body_id,
                self.position.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=client,
            )
        return moved

    def nearest_exit_within_proximity(
        self,
        exits: List[np.ndarray],
    ) -> Optional[np.ndarray]:
        """D-051: return nearest exit within ``exit_proximity_m`` (XY) or None.

        Used by S2/S3 person loops as the **top priority** in their
        waypoint decision: if an exit is already within
        ``config.exit_proximity_m`` (default 1.0 m = 2 cells) the agent
        ignores its current GUIDING drone and walks straight into that
        exit cell. This prevents the "drone hovers around the exit"
        loop where the occupant orbits the drone instead of stepping
        into the exit tolerance radius.

        Args:
            exits: List of ``(3,)`` exit world positions.

        Returns:
            ``(3,)`` array of the chosen exit (Z preserved from the
            agent's current position) if any exit is within the
            proximity radius, else ``None``. Non-ALIVE agents always
            return ``None``.
        """
        if self.status != PersonStatus.ALIVE:
            return None
        proximity = float(self.config.exit_proximity_m)
        proximity_sq = proximity * proximity
        my_xy = self.position[:2]
        best: Optional[np.ndarray] = None
        best_d2 = proximity_sq
        for ex in exits:
            ex_arr = np.asarray(ex, dtype=np.float64).reshape(3)
            dx = float(ex_arr[0] - my_xy[0])
            dy = float(ex_arr[1] - my_xy[1])
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best = np.array(
                    [ex_arr[0], ex_arr[1], float(self.position[2])],
                    dtype=np.float64,
                )
        return best

    def scan_for_drones(self, drones) -> Optional[object]:
        """Find the nearest GUIDING drone within ``follow_range_m``.

        Per D-030 (user spec 2026-05-14): a PersonAgent has no global
        knowledge of the building or the fire. It only walks toward
        whatever drone is currently in its sight. This method updates
        :attr:`following_drone_id` to reflect the current visible drone
        (or clears it if none is in range) and returns the chosen drone
        object (or ``None``).

        Args:
            drones: List of :class:`DroneAgent`. Only drones whose
                ``.status`` is GUIDING (i.e. their attribute string
                equals ``"guiding"``) are considered visible — a drone
                in SEARCHING mode is "just patrolling", not actively
                guiding this agent.

        Returns:
            The closest GUIDING drone within ``follow_range_m`` of this
            agent's XY position, or ``None`` if no such drone exists.
        """
        if self.status != PersonStatus.ALIVE:
            self.following_drone_id = None
            return None
        follow_range = float(self.config.follow_range_m)
        best = None
        best_d2 = follow_range * follow_range
        my_xy = self.position[:2]
        for d in drones:
            # Use duck-typed status check to avoid a circular import.
            d_status = getattr(d, "status", None)
            if d_status is None or getattr(d_status, "value", str(d_status)) != "guiding":
                continue
            dxy = np.asarray(d.position, dtype=np.float64)[:2] - my_xy
            d2 = float(dxy[0] * dxy[0] + dxy[1] * dxy[1])
            if d2 <= best_d2:
                best_d2 = d2
                best = d
        self.following_drone_id = best.drone_id if best is not None else None
        return best

    @property
    def alive(self) -> bool:
        return self.status == PersonStatus.ALIVE

    @property
    def evacuated(self) -> bool:
        return self.status == PersonStatus.EVACUATED

    @property
    def dead(self) -> bool:
        return self.status == PersonStatus.DEAD


# ─── Module-private helpers ───────────────────────────────────────────────
def _bfs_grid_path(
    start_xyz: np.ndarray,
    target_xyz: np.ndarray,
    mask: np.ndarray,
    z_layer: int = 3,
) -> List[np.ndarray]:
    """4-connected BFS over a (nx, ny, nz) boolean mask at ``z_layer``.

    Returns a list of ``(3,)`` world-coordinate waypoints from the start
    cell to the target cell, inclusive. Every waypoint is the center of
    a mask-True cell, so an agent that follows the path step-by-step
    never enters a solid/wall cell.

    Snapping: if the start cell is solid (e.g. agent was clipped onto a
    wall by a previous bug), the BFS source is the nearest mask-True cell
    found by a small spiral. If the target cell is solid, the BFS target
    is the nearest mask-True cell to the target. If no path exists the
    function returns ``[]`` (caller should stay put / re-plan next tick).
    """
    nx_, ny_, nz_ = mask.shape
    if not (0 <= z_layer < nz_):
        return []
    layer = mask[:, :, z_layer]

    def _snap(x: float, y: float) -> Optional[Tuple[int, int]]:
        i0 = int(x / CELL_SIZE_M)
        j0 = int(y / CELL_SIZE_M)
        # Quick path.
        if (0 <= i0 < nx_ and 0 <= j0 < ny_ and layer[i0, j0]):
            return (i0, j0)
        # Spiral out a few rings looking for a fluid cell.
        for r in range(1, 6):
            for di in range(-r, r + 1):
                for dj in (-r, r):
                    ni, nj = i0 + di, j0 + dj
                    if 0 <= ni < nx_ and 0 <= nj < ny_ and layer[ni, nj]:
                        return (ni, nj)
                for dj in range(-r + 1, r):
                    ni = i0 + di
                    if di not in (-r, r):
                        continue  # only ring perimeter
                    if 0 <= ni < nx_ and 0 <= nj < ny_ and layer[ni, nj]:
                        return (ni, nj)
            for dj in range(-r, r + 1):
                for di in (-r, r):
                    ni, nj = i0 + di, j0 + dj
                    if 0 <= ni < nx_ and 0 <= nj < ny_ and layer[ni, nj]:
                        return (ni, nj)
        return None

    s_cell = _snap(float(start_xyz[0]), float(start_xyz[1]))
    g_cell = _snap(float(target_xyz[0]), float(target_xyz[1]))
    if s_cell is None or g_cell is None:
        return []
    if s_cell == g_cell:
        return [_cell_center_xyz(*s_cell, z_layer)]

    parent: Dict[Tuple[int, int], Tuple[int, int]] = {s_cell: s_cell}
    q: deque = deque([s_cell])
    found = False
    while q:
        c = q.popleft()
        if c == g_cell:
            found = True
            break
        ci, cj = c
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx_ and 0 <= nj < ny_):
                continue
            if not layer[ni, nj]:
                continue
            nbr = (ni, nj)
            if nbr in parent:
                continue
            parent[nbr] = c
            q.append(nbr)
    if not found:
        return []
    chain = [g_cell]
    while chain[-1] != s_cell:
        chain.append(parent[chain[-1]])
    chain.reverse()
    return [_cell_center_xyz(i, j, z_layer) for (i, j) in chain]


def _cell_center_xyz(i: int, j: int, z_layer: int) -> np.ndarray:
    return np.array([
        (i + 0.5) * CELL_SIZE_M,
        (j + 0.5) * CELL_SIZE_M,
        0.25 + CELL_SIZE_M * z_layer,
    ], dtype=np.float64)


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("=" * 60)
    print("person_agent.py self-test (M2-mini: spawn + step_toward)")
    print("=" * 60)

    errors: list[str] = []

    from src.integration.scene import Scene, SceneConfig

    placeholder = Path("assets/placeholder_building.urdf")
    if not placeholder.exists():
        print(f"FAIL: missing {placeholder}; run urdf_builder first")
        sys.exit(1)

    cfg = SceneConfig(
        connection_mode="DIRECT", building_urdf=placeholder
    )
    with Scene.create(cfg) as scene:
        obstacles = [scene.building_id]
        print(
            f"\nScene ready: client={scene.client} "
            f"plane_id={scene.plane_id} building_id={scene.building_id}"
        )

        # ── 1. Spawn validation ──────────────────────────────────────
        print("\n[1] Spawn rejects bad inputs")
        a = PersonAgent(agent_id="probe")
        try:
            a.spawn(scene.client, np.array([1.0, 2.0]))  # bad shape
        except ValueError:
            print("  PASS: (2,) -> ValueError")
        else:
            errors.append("(2,) start did not raise")
        a.spawn(scene.client, np.array([5.0, 13.0, 1.5]))
        if a.body_id < 0:
            errors.append(f"spawn left body_id={a.body_id}")
        # Capsule should rest on the floor: z = cyl_h/2 + radius
        cyl_h = a.config.body_height_m - 2 * a.config.body_radius_m
        exp_z = cyl_h / 2 + a.config.body_radius_m
        if abs(a.position[2] - exp_z) > 1e-6:
            errors.append(f"spawn z = {a.position[2]} != {exp_z}")
        print(
            f"  body_id={a.body_id} position={tuple(round(v, 3) for v in a.position)}"
        )

        # ── 2. Double spawn raises ───────────────────────────────────
        print("\n[2] Second spawn raises RuntimeError")
        try:
            a.spawn(scene.client, np.array([0.0, 0.0, 0.0]))
        except RuntimeError:
            print("  PASS")
        else:
            errors.append("double spawn did not raise")

        # ── 3. Clear-path agent reaches target ────────────────────────
        print("\n[3] Agent A: clear path (5, 13) -> (5, 18)")
        agent_a = PersonAgent(agent_id="A")
        agent_a.spawn(scene.client, np.array([5.0, 13.0, 1.5]))
        target_a = np.array([5.0, 18.0, agent_a.position[2]])
        a_start = agent_a.position.copy()
        moved_steps_a = 0
        blocked_a = 0
        for step in range(20):
            ok = agent_a.step_toward(
                scene.client, target_a, dt_s=1.0, obstacle_body_ids=obstacles
            )
            scene.step()
            if ok:
                moved_steps_a += 1
            else:
                blocked_a += 1
            dist = float(
                np.linalg.norm(agent_a.position[:2] - target_a[:2])
            )
            if dist <= agent_a.config.arrival_tolerance_m:
                break
        print(
            f"  end position = {tuple(round(v, 3) for v in agent_a.position)}  "
            f"steps={step + 1}  blocked={blocked_a}"
        )
        final_dist_a = float(
            np.linalg.norm(agent_a.position[:2] - target_a[:2])
        )
        if final_dist_a > agent_a.config.arrival_tolerance_m:
            errors.append(
                f"agent A did not arrive: dist={final_dist_a:.3f} m"
            )
        if blocked_a > 0:
            errors.append(
                f"agent A blocked {blocked_a} times on a clear path"
            )

        # Distance traveled approx 5 m at 1.2 m/s -> ~5 steps.
        traveled = float(
            np.linalg.norm(agent_a.position[:2] - a_start[:2])
        )
        if traveled < 4.5:
            errors.append(
                f"agent A traveled only {traveled:.2f} m (expected ~5)"
            )

        # ── 4. Wall-blocked agent stops before partition (x=15) ──────
        print("\n[4] Agent B: blocked by interior partition (10, 5) -> (20, 5)")
        agent_b = PersonAgent(agent_id="B")
        agent_b.spawn(scene.client, np.array([10.0, 5.0, 1.5]))
        target_b = np.array([20.0, 5.0, agent_b.position[2]])
        blocked_b = 0
        last_x = agent_b.position[0]
        for step in range(20):
            ok = agent_b.step_toward(
                scene.client, target_b, dt_s=1.0, obstacle_body_ids=obstacles
            )
            scene.step()
            if not ok:
                blocked_b += 1
                # Allow a couple of blocked ticks (boundary samples may
                # alternate) but if we've been stuck for 3 ticks, give up.
                if blocked_b >= 3:
                    break
            last_x = agent_b.position[0]
        print(
            f"  end position = {tuple(round(v, 3) for v in agent_b.position)}  "
            f"steps={step + 1}  blocked={blocked_b}"
        )

        # Partition centred at x=15, thickness 0.2 -> west face at x=14.9.
        # Capsule radius 0.25 -> agent should stop at x <= 14.65.
        if last_x >= 14.7:
            errors.append(
                f"agent B reached x={last_x:.3f} -- did not stop at partition"
            )
        if blocked_b == 0:
            errors.append("agent B never reported a blocked step")
        # Also confirm we did NOT reach the target.
        final_dist_b = float(
            np.linalg.norm(agent_b.position[:2] - target_b[:2])
        )
        if final_dist_b <= agent_b.config.arrival_tolerance_m:
            errors.append(
                f"agent B reached target dist={final_dist_b:.2f} -- collision veto broken"
            )

        # ── 5. step_toward rejects bad inputs ─────────────────────────
        print("\n[5] step_toward input validation")
        try:
            agent_b.step_toward(
                scene.client, np.array([1.0, 2.0]),  # bad shape
                dt_s=1.0,
                obstacle_body_ids=obstacles,
            )
        except ValueError:
            print("  PASS: (2,) target -> ValueError")
        else:
            errors.append("bad target shape did not raise")
        try:
            agent_b.step_toward(
                scene.client, target_b, dt_s=0.0, obstacle_body_ids=obstacles
            )
        except ValueError:
            print("  PASS: dt_s=0 -> ValueError")
        else:
            errors.append("dt_s=0 did not raise")

        # ── 6. Pre-spawn step_toward raises ──────────────────────────
        print("\n[6] step_toward before spawn raises")
        c = PersonAgent(agent_id="never_spawned")
        try:
            c.step_toward(
                scene.client, np.array([0.0, 0.0, 0.85]),
                dt_s=1.0,
                obstacle_body_ids=obstacles,
            )
        except RuntimeError:
            print("  PASS")
        else:
            errors.append("pre-spawn step_toward did not raise")

        # ── 7. Already-arrived agent returns True without moving ─────
        print("\n[7] Already-arrived agent: step_toward returns True, no move")
        d = PersonAgent(agent_id="atGoal")
        d.spawn(scene.client, np.array([7.0, 7.0, 1.5]))
        target_d = np.array([7.0, 7.0, d.position[2]])  # = current pos
        before = d.position.copy()
        ok = d.step_toward(
            scene.client, target_d, dt_s=1.0, obstacle_body_ids=obstacles
        )
        if not ok:
            errors.append("at-goal step_toward returned False")
        if not np.allclose(d.position, before):
            errors.append(
                f"at-goal step_toward moved: {before} -> {d.position}"
            )
        print(
            f"  ok={ok}  pos={tuple(round(v, 3) for v in d.position)}"
        )

    # ── 8. accumulate_exposure: zero CO → no change ───────────────
    print("\n[8] accumulate_exposure: zero CO leaves FED + status untouched")
    e = PersonAgent(agent_id="exposure_zero")
    e.spawn(Scene.create(SceneConfig(connection_mode="DIRECT", building_urdf=placeholder)).client,
            np.array([3.0, 3.0, 1.5]))
    died = e.accumulate_exposure(0.0, 0.0, dt_s=1.0)
    if died or e.cumulative_fed != 0.0 or e.status != PersonStatus.ALIVE:
        errors.append(
            f"zero CO changed state: fed={e.cumulative_fed} status={e.status} died={died}"
        )
    else:
        print(f"  PASS: fed={e.cumulative_fed} status={e.status.value}")

    # ── 9. accumulate_exposure: high CO → eventual DEAD ─────────────
    print("\n[9] accumulate_exposure: 5000 ppm steady → DEAD before 60 s")
    f = PersonAgent(agent_id="exposure_high")
    # Don't need a scene for this — body_id stays -1, accumulate_exposure
    # is body-independent.
    steps_to_dead = None
    for k in range(120):
        died = f.accumulate_exposure(5000.0, 0.0, dt_s=1.0)
        if died:
            steps_to_dead = k + 1
            break
    print(
        f"  steps_to_dead={steps_to_dead}  final_fed={f.cumulative_fed:.4f}  "
        f"status={f.status.value}"
    )
    # Expected: at 5000 ppm steady, FED reaches 0.3 after
    # 0.3 / (5000/60/27000) = 0.3 * 27000 * 60 / 5000 = 97.2 s. Allow ±5 s.
    if steps_to_dead is None or not (90 <= steps_to_dead <= 105):
        errors.append(
            f"high-CO death timing wrong: steps={steps_to_dead}"
        )
    if f.status != PersonStatus.DEAD:
        errors.append(f"high-CO did not transition to DEAD: {f.status}")

    # ── 10. accumulate_exposure: instant-death via danger ─────────
    print("\n[10] accumulate_exposure: danger ≥ 0.99 instant-kills")
    g = PersonAgent(agent_id="instant_death")
    died = g.accumulate_exposure(0.0, experienced_danger=1.0, dt_s=1.0)
    if not died or g.status != PersonStatus.DEAD:
        errors.append(f"instant-death not applied: died={died} status={g.status}")
    else:
        print(f"  PASS: died={died} status={g.status.value}")

    # ── 11. accumulate_exposure no-op once EVACUATED or DEAD ──────
    print("\n[11] accumulate_exposure on non-ALIVE agent is a no-op")
    h = PersonAgent(agent_id="evacuated_noop")
    h.mark_evacuated()
    fed_before = h.cumulative_fed
    died = h.accumulate_exposure(5000.0, 0.0, dt_s=1.0)
    if died or h.cumulative_fed != fed_before or h.status != PersonStatus.EVACUATED:
        errors.append(
            f"evacuated agent mutated by accumulate: fed={h.cumulative_fed} "
            f"status={h.status} died={died}"
        )
    else:
        print("  PASS")

    # ── 12. mark_evacuated idempotent ─────────────────────────────
    print("\n[12] mark_evacuated is idempotent")
    j = PersonAgent(agent_id="evac_idempotent")
    j.mark_evacuated(); j.mark_evacuated()
    if j.status != PersonStatus.EVACUATED:
        errors.append(f"second mark_evacuated broke status: {j.status}")
    # And it must NOT override DEAD.
    k = PersonAgent(agent_id="dead_first")
    k.accumulate_exposure(0.0, experienced_danger=1.0, dt_s=1.0)
    k.mark_evacuated()
    if k.status != PersonStatus.DEAD:
        errors.append(f"mark_evacuated overrode DEAD: {k.status}")
    else:
        print("  PASS")

    # ── 13. scan_for_drones picks nearest GUIDING in range ─────────
    print("\n[13] scan_for_drones picks nearest GUIDING drone in range")

    class _DummyDroneStatus:
        def __init__(self, v: str) -> None:
            self.value = v

    class _DummyDrone:
        def __init__(self, drone_id: str, pos, status_str: str) -> None:
            self.drone_id = drone_id
            self.position = np.asarray(pos, dtype=np.float64)
            self.status = _DummyDroneStatus(status_str)

    pp = PersonAgent(agent_id="scan_probe")
    pp.position = np.array([5.0, 5.0, 1.5])
    drones = [
        _DummyDrone("d_close_guide", [6.0, 5.0, 1.8], "guiding"),
        _DummyDrone("d_far_guide", [20.0, 5.0, 1.8], "guiding"),
        _DummyDrone("d_close_search", [4.5, 5.0, 1.8], "searching"),  # ignored
    ]
    picked = pp.scan_for_drones(drones)
    if picked is None or picked.drone_id != "d_close_guide":
        errors.append(
            f"expected d_close_guide, got "
            f"{None if picked is None else picked.drone_id}"
        )
    if pp.following_drone_id != "d_close_guide":
        errors.append(f"following_drone_id not set: {pp.following_drone_id}")
    else:
        print(f"  PASS: chose {pp.following_drone_id}")

    # ── 14. scan_for_drones returns None when out of range ─────────
    print("\n[14] No GUIDING drone in range -> None + clears following")
    pp.following_drone_id = "stale"
    far_drones = [_DummyDrone("far", [40.0, 40.0, 1.8], "guiding")]
    picked = pp.scan_for_drones(far_drones)
    if picked is not None:
        errors.append(f"expected None, got {picked.drone_id}")
    if pp.following_drone_id is not None:
        errors.append(f"following_drone_id not cleared: {pp.following_drone_id}")
    else:
        print("  PASS")

    # ── 15. scan_for_drones is a no-op for non-ALIVE ────────────────
    print("\n[15] scan_for_drones no-op for non-ALIVE")
    pp_evac = PersonAgent(agent_id="evac_scan")
    pp_evac.mark_evacuated()
    pp_evac.following_drone_id = "stale_d"
    drones_near = [_DummyDrone("close", [0.0, 0.0, 1.8], "guiding")]
    picked = pp_evac.scan_for_drones(drones_near)
    if picked is not None or pp_evac.following_drone_id is not None:
        errors.append(
            f"non-ALIVE scan mutated: "
            f"picked={picked} following={pp_evac.following_drone_id}"
        )
    else:
        print("  PASS")

    # ── Verdict ────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(
        "\nPASS: PersonAgent spawn + step_toward + accumulate_exposure + "
        "scan_for_drones validated"
    )
