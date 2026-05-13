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

**Status (2026-05-14, M2-mini):**

* :meth:`PersonAgent.spawn` — **functional** (kinematic capsule body).
* :meth:`PersonAgent.step_toward` — **functional**. Atomic kinematic
  motion step with PyBullet contact-based wall avoidance. Used by the
  full :meth:`update` and exercisable independently for testing.
* :meth:`PersonAgent.update` — still skeleton. Orchestrates status
  machine + FED accumulation + waypoint pulling; implemented in M2-full
  once the status transitions are nailed down.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol

import numpy as np
import pybullet as p

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

    # ── Verdict ────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: PersonAgent spawn + step_toward validated")
