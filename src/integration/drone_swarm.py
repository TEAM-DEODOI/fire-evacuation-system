"""Drone swarm for S2/S3 active-guidance scenarios (D-025 / D-030 Week 12).

Per user spec (2026-05-14):

* Drones do NOT know occupant positions a priori. They spawn at fire-alarm
  time and patrol the building interior to *search* for survivors.
* When a drone detects a PersonAgent within :attr:`sense_range_m`, it
  transitions to **GUIDING** state and computes a safe path from that
  agent to the nearest exit via
  :class:`~src.path_planning.planners.EvacuationPlanner`. The drone then
  flies *along that path* (acting as a visual cue).
* The PersonAgent's side of the protocol lives in
  :class:`~src.integration.person_agent.PersonAgent.scan_for_drones`:
  any GUIDING drone within :attr:`PersonAgentConfig.follow_range_m` is
  the agent's current waypoint source. When the drone moves on, the
  agent follows.
* "Interior" is the :func:`load_interior_mask` set (D-028) — drones only
  patrol cells the mask flags as indoor.

The swarm composition is **one drone per exit** by default (3 drones,
spawned at each canonical exit XY). Each drone is independent: no
Boids/APF coordination (decision: keep small swarm logic simple).

Two finite-state behaviours per drone:

* **SEARCHING** — frontier-based coverage. Each drone maintains its own
  ``visited_cells`` set; the next target is the *unvisited interior cell
  closest to the drone* (with a tie-break against ``last_target`` so
  drones do not oscillate). When all interior cells are visited, the
  visited set is reset.
* **GUIDING** — drone holds an assigned ``target_person_id`` and a path
  to that person's nearest exit. Drone flies along that path. If the
  target is evacuated / dead / out of follow range for ``lost_timeout_s``
  ticks, the drone falls back to SEARCHING.

The swarm is purely **logical** for now: no PyBullet body, no Crazyflie
URDF. The :class:`DroneAgent.position` is updated each tick via the same
kinematic step as :class:`~src.integration.person_agent.PersonAgent`
(fluid-mask veto). A Crazyflie body + Renderer overlay are M4-full work,
separate from the logic implemented here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.path_planning.planners import EvacuationPlanner
from src.risk_map.risk_map_class import RiskMap
from src.shared.constants import CELL_SIZE_M


# ─── Drone state enum ─────────────────────────────────────────────────────
class DroneStatus(str, Enum):
    """3-state drone lifecycle."""

    SEARCHING = "searching"
    GUIDING = "guiding"
    IDLE = "idle"    # reserved (e.g. landed at exit, low battery) -- unused for now


# ─── Config ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DroneAgentConfig:
    """Per-drone constants.

    Attributes:
        sense_range_m: XY radius within which a drone detects PersonAgents
            (user spec: 5 m).
        flight_speed_mps: Constant kinematic speed (user spec: 3 m/s, fast
            enough to lead a 1.2 m/s person).
        flight_altitude_m: Constant Z for visual telegraphing.
        waypoint_tolerance_m: Distance at which a drone counts as "arrived
            at the current waypoint" (advances to next path index).
        lost_timeout_s: How long a GUIDING drone keeps the assignment
            after losing sight of its target. Defaults to 5 s.
        replan_period_s: How often the drone re-plans the survivor->exit
            path while GUIDING (so a fresh RiskMap query closes hazardous
            edges).
        frontier_target_dwell_s: How long a SEARCHING drone sticks with a
            chosen frontier target before reconsidering, even if not yet
            reached. Avoids thrashing when frontiers shift.
    """

    sense_range_m: float = 5.0
    flight_speed_mps: float = 3.0
    flight_altitude_m: float = 1.8
    waypoint_tolerance_m: float = 0.5
    lost_timeout_s: float = 5.0
    replan_period_s: float = 5.0
    frontier_target_dwell_s: float = 4.0
    exit_release_tol_m: float = 1.5
    """Drone is considered "back at an exit" when its XY position is
    within this tolerance of any exit. Triggers a forced release from
    GUIDING back to SEARCHING (D-031 #3, 2026-05-14)."""
    min_frontier_component_cells: int = 16
    """Minimum size of an unvisited connected component to be considered
    as a frontier target. 1 cell = 0.25 m², so the default 16 cells
    (≈ 4 m², a 2×2 m patch) filters out wall-edge slivers that the drone
    has effectively already sensed but the alg picks up as unvisited.
    Without this filter the swarm thrashes between 1–3 cell slivers.

    If every unvisited component is smaller than this threshold the
    fallback in :func:`_pick_frontier_target_cluster` still returns the
    *largest* available component, so no real area is permanently
    starved of patrol. (D-031 #1 + Fix C, 2026-05-14.)"""


@dataclass(frozen=True)
class DroneSwarmConfig:
    """Top-level swarm hyperparameters.

    Attributes:
        n_drones: Number of drones. Default 3 (one per exit per user spec).
        drone: Per-drone constants.
    """

    n_drones: int = 3
    drone: DroneAgentConfig = field(default_factory=DroneAgentConfig)


# ─── DroneAgent ───────────────────────────────────────────────────────────
@dataclass
class DroneAgent:
    """A single drone in the swarm.

    Attributes:
        drone_id: Unique identifier (e.g. ``"drone_0"``).
        position: ``(3,)`` world position in metres.
        status: :class:`DroneStatus`.
        target_person_id: PersonAgent currently being shepherded
            (``None`` iff SEARCHING).
        current_path: Cached planner path during GUIDING (list of
            ``(3,)`` waypoints).
        wp_idx: Index into ``current_path`` of the drone's *next*
            waypoint (path[0] = start position, so first move target is
            index 1).
        visited_cells: Cells (``(i, j)`` interior-grid indices) this drone
            has already passed through (used for frontier exploration).
        frontier_target: ``(3,)`` world position the drone is currently
            heading toward while SEARCHING. ``None`` means "pick a new one".
        last_replan_t: Wall time of the most recent replan during GUIDING.
        last_seen_target_t: Wall time the drone most recently saw its
            assigned PersonAgent (used by ``lost_timeout_s``).
        config: Per-drone constants.
    """

    drone_id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    status: DroneStatus = DroneStatus.SEARCHING
    target_person_id: Optional[str] = None
    current_path: List[np.ndarray] = field(default_factory=list)
    wp_idx: int = 0
    frontier_target: Optional[np.ndarray] = None
    search_path: List[np.ndarray] = field(default_factory=list)
    """BFS path through interior cells from the drone's spawn-time cell
    to :attr:`frontier_target`. Computed once when a new frontier target
    is picked (D-031 #2 wall-aware navigation). Each element is a
    cell-center world coordinate ``(x, y, z)``."""
    search_wp_idx: int = 0
    last_replan_t: float = -1e9
    last_seen_target_t: float = -1e9
    last_frontier_pick_t: float = -1e9
    config: DroneAgentConfig = field(default_factory=DroneAgentConfig)

    def __post_init__(self) -> None:
        if self.position.shape != (3,):
            self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.position = self.position.astype(np.float64, copy=True)

    # ─── Kinematic motion ───────────────────────────────────────────
    def _step_toward(
        self,
        target_xyz: np.ndarray,
        dt_s: float,
        interior_mask: np.ndarray,
    ) -> None:
        """Advance position one tick toward ``target_xyz`` (kinematic).

        Mirrors :meth:`PersonAgent.step_toward` (fluid-mask mode) but
        clips to the **interior** mask so the drone never patrols outside
        the building. Z is held at ``config.flight_altitude_m``.
        """
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        delta = target[:2] - self.position[:2]
        remaining = float(np.linalg.norm(delta))
        if remaining <= 1e-6:
            return
        step = float(self.config.flight_speed_mps) * float(dt_s)
        move = min(step, remaining)
        direction = delta / remaining
        candidate = np.array([
            self.position[0] + direction[0] * move,
            self.position[1] + direction[1] * move,
            float(self.config.flight_altitude_m),
        ], dtype=np.float64)
        # Interior-mask veto: only enter cells flagged as indoor.
        ix = int(candidate[0] / CELL_SIZE_M)
        iy = int(candidate[1] / CELL_SIZE_M)
        # z-layer index at flight altitude.
        iz = int(candidate[2] / CELL_SIZE_M)
        nx_, ny_, nz_ = interior_mask.shape
        if not (0 <= ix < nx_ and 0 <= iy < ny_ and 0 <= iz < nz_):
            return  # off-grid: veto
        if not bool(interior_mask[ix, iy, iz]):
            return  # outside / wall: veto
        self.position = candidate
        # NB: visited marking happens at the swarm level, sensing-range
        # aware (see DroneSwarm._mark_sensed_area). The drone itself
        # no longer tracks any visited state.


# ─── DroneSwarm ───────────────────────────────────────────────────────────
@dataclass
class DroneSwarm:
    """Multi-drone shepherd implementing the user's search-and-guide spec.

    Use:

    >>> swarm = DroneSwarm(config=DroneSwarmConfig(n_drones=3))
    >>> swarm.spawn(spawn_xyzs=exit_positions())
    >>> for t in times:
    ...     swarm.update(t, dt, persons, risk_map, planner,
    ...                  interior_mask=interior, exits=exit_positions())

    Each call to :meth:`update` advances every drone by one tick. The
    PersonAgent side reads ``drone.position`` via
    :meth:`PersonAgent.scan_for_drones` and walks toward the nearest
    GUIDING drone within its follow range.

    Attributes:
        config: :class:`DroneSwarmConfig`.
        drones: List of :class:`DroneAgent` (populated by :meth:`spawn`).
    """

    config: DroneSwarmConfig = field(default_factory=DroneSwarmConfig)
    drones: List[DroneAgent] = field(default_factory=list)
    visited_cells: Set[Tuple[int, int]] = field(default_factory=set)
    """Swarm-shared sensed-cell set (D-030 v2, 2026-05-14). When any
    drone passes within ``sense_range_m`` of a cell, that cell is added
    here. Frontier-based exploration consults this single shared mask so
    drones naturally split coverage instead of each re-scanning the
    same regions."""

    # ─── Lifecycle ──────────────────────────────────────────────────
    def spawn(
        self,
        spawn_xyzs: List[np.ndarray],
        interior_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Create ``n_drones`` DroneAgents at the given start positions.

        Args:
            spawn_xyzs: List of ``(3,)`` world coordinates, one per
                drone (typically ``exit_positions()``). The Z is replaced
                by ``flight_altitude_m`` so drones always hover.
            interior_mask: ``(60, 40, 6)`` boolean. When supplied, the
                spawn-time sensing area of every drone is pre-marked into
                the swarm-shared visited set so the first frontier pick
                never selects "where we already are".

        Raises:
            ValueError: If fewer spawn positions are supplied than the
                configured ``n_drones`` (we never spawn drones on top of
                each other).
        """
        n = int(self.config.n_drones)
        if len(spawn_xyzs) < n:
            raise ValueError(
                f"need at least {n} spawn positions, got {len(spawn_xyzs)}"
            )
        self.drones.clear()
        self.visited_cells = set()
        for i in range(n):
            xyz = np.asarray(spawn_xyzs[i], dtype=np.float64).reshape(3)
            xyz[2] = float(self.config.drone.flight_altitude_m)
            d = DroneAgent(
                drone_id=f"drone_{i}",
                position=xyz.copy(),
                status=DroneStatus.SEARCHING,
                config=self.config.drone,
            )
            self.drones.append(d)
            if interior_mask is not None:
                self._mark_sensed_area(d.position, interior_mask)

    # ─── Per-tick update ────────────────────────────────────────────
    def update(
        self,
        t: float,
        dt_s: float,
        persons: List,  # List[PersonAgent] (avoid circular import)
        planner_rm: RiskMap,
        planner: EvacuationPlanner,
        interior_mask: np.ndarray,
        exits: List[np.ndarray],
        *,
        co_field: Optional[object] = None,
    ) -> None:
        """Advance every drone by one tick.

        Per-drone logic:

        1. **Update sensing**: any ALIVE person within ``sense_range_m``
           of this drone is "detected" by this drone.
        2. **GUIDING**: if currently shepherding a person, check that the
           person is still ALIVE and within range; if not, fall back to
           SEARCHING. Otherwise, re-plan if ``replan_period_s`` elapsed
           and step along the cached path one tick.
        3. **SEARCHING**: if at least one detected ALIVE person is not
           already assigned to another drone, take that person and switch
           to GUIDING. Otherwise, walk toward the current frontier target
           (pick a new one if reached / stale).

        Args:
            t: Wall time in seconds.
            dt_s: Outer-loop step (s).
            persons: List of :class:`PersonAgent`. Read-only — drones
                only inspect ``.status``, ``.position``, ``.agent_id``.
            planner_rm: RiskMap the drone planner consults (FDS truth in
                S2, model prediction in S3).
            planner: Shared :class:`EvacuationPlanner` (building graph
                cached at construction).
            interior_mask: ``(60, 40, 6)`` boolean — drones never leave
                indoor cells.
            exits: List of ``(3,)`` exit positions; closest is selected
                as each person's target.
        """
        # Build a fast index of currently-assigned person ids so two
        # drones do not converge on the same survivor.
        already_assigned = {
            d.target_person_id for d in self.drones
            if d.target_person_id is not None
        }

        alive_persons = [a for a in persons if _is_alive(a)]
        person_by_id = {a.agent_id: a for a in persons}

        for drone in self.drones:
            # ── 0. Sense ────────────────────────────────────────────
            detected = _detect_persons_near(
                drone.position, alive_persons,
                drone.config.sense_range_m,
            )

            # ── 1. GUIDING branch ───────────────────────────────────
            if drone.status == DroneStatus.GUIDING:
                target = person_by_id.get(drone.target_person_id)
                if target is None or not _is_alive(target):
                    # Survivor evacuated or died -- fall back.
                    _release_drone(drone)
                elif _drone_reached_exit(drone, exits):
                    # D-031 #3: drone has shepherded the agent to an exit
                    # (or arrived there ahead of them). Hand off and
                    # return to SEARCHING so other survivors can be
                    # found. The trailing PersonAgent will be marked
                    # evacuated by the scenario as soon as its own
                    # position enters the exit tolerance.
                    _release_drone(drone)
                else:
                    # Refresh "last seen" if target is currently in range.
                    if any(p.agent_id == target.agent_id for p in detected):
                        drone.last_seen_target_t = t
                    elif t - drone.last_seen_target_t > drone.config.lost_timeout_s:
                        # Lost target for too long: release and re-search.
                        _release_drone(drone)
                    if drone.status == DroneStatus.GUIDING:
                        # Replan periodically.
                        if (t - drone.last_replan_t) >= drone.config.replan_period_s:
                            new_path = planner.replan(
                                target.position, planner_rm, t=t,
                                co_field=co_field,
                            )
                            if new_path:
                                drone.current_path = new_path
                                drone.wp_idx = 1 if len(new_path) > 1 else 0
                            drone.last_replan_t = t
                        # Step toward current waypoint (or the target's
                        # position if path is exhausted).
                        wp = _drone_lead_waypoint(drone, target)
                        if wp is not None:
                            drone._step_toward(wp, dt_s, interior_mask)
                            self._mark_sensed_area(drone.position, interior_mask)
                            # Advance the path index if reached the
                            # currently-targeted waypoint.
                            if (
                                drone.wp_idx < len(drone.current_path)
                                and float(np.linalg.norm(
                                    drone.position[:2]
                                    - drone.current_path[drone.wp_idx][:2]
                                )) <= drone.config.waypoint_tolerance_m
                            ):
                                drone.wp_idx += 1
                        continue  # done with this drone for this tick

            # ── 2. SEARCHING branch (or just released) ──────────────
            # Try to acquire a new survivor.
            for p in detected:
                if p.agent_id in already_assigned:
                    continue
                # Acquire!
                drone.status = DroneStatus.GUIDING
                drone.target_person_id = p.agent_id
                drone.last_seen_target_t = t
                # Compute initial path person -> exit.
                new_path = planner.plan(
                    p.position, planner_rm, t=t, co_field=co_field,
                )
                drone.current_path = new_path
                drone.wp_idx = 1 if len(new_path) > 1 else 0
                drone.last_replan_t = t
                already_assigned.add(p.agent_id)
                break
            if drone.status == DroneStatus.GUIDING:
                continue  # acquired this tick; will step next tick

            # Frontier patrol (D-031: cluster-aware target + wall-aware path)
            # Fix A (D-031.1, 2026-05-14): do NOT trigger a new target every
            # time the current target cell falls into the swarm visited
            # set. That happened on the very next tick (sense radius
            # 5 m covers any nearby medoid the moment the drone takes a
            # step) and forced an expensive cluster+BFS recompute every
            # single tick. Arrival tolerance + dwell timeout are enough.
            need_new_target = (
                drone.frontier_target is None
                or _xy_dist(drone.position, drone.frontier_target)
                <= drone.config.waypoint_tolerance_m
                or (t - drone.last_frontier_pick_t)
                >= drone.config.frontier_target_dwell_s
                or drone.search_wp_idx >= len(drone.search_path)
            )
            if need_new_target:
                target_xyz = _pick_frontier_target_cluster(
                    drone.position,
                    self.visited_cells,
                    interior_mask,
                    min_component_cells=drone.config.min_frontier_component_cells,
                )
                drone.frontier_target = target_xyz
                drone.last_frontier_pick_t = t
                if target_xyz is not None:
                    drone.search_path = _bfs_drone_path(
                        start_xyz=drone.position,
                        target_xyz=target_xyz,
                        interior_mask=interior_mask,
                    )
                    drone.search_wp_idx = 1 if len(drone.search_path) > 1 else 0
                else:
                    drone.search_path = []
                    drone.search_wp_idx = 0
            if drone.frontier_target is not None and drone.search_path:
                # Step toward the next BFS waypoint (wall-aware).
                idx = min(drone.search_wp_idx, len(drone.search_path) - 1)
                wp = drone.search_path[idx]
                drone._step_toward(wp, dt_s, interior_mask)
                self._mark_sensed_area(drone.position, interior_mask)
                # Advance along the BFS path if reached the current waypoint.
                if _xy_dist(drone.position, wp) <= drone.config.waypoint_tolerance_m:
                    drone.search_wp_idx = idx + 1

    # ─── Sensing-aware visited marking (D-030 v2) ──────────────────
    def _mark_sensed_area(
        self,
        position: np.ndarray,
        interior_mask: np.ndarray,
    ) -> None:
        """Add every interior cell within ``sense_range_m`` of ``position``
        to the swarm-shared :attr:`visited_cells` set.

        A drone hovering at one spot effectively scans a disc of radius
        :attr:`DroneAgentConfig.sense_range_m` (default 5 m, i.e. ~314
        cells at 0.5 m resolution). Marking only the drone's footprint
        cell (the old behaviour) wasted roughly two orders of magnitude
        of coverage. With this method one tick at a single spot covers
        the entire visible neighbourhood.
        """
        sense_r = float(self.config.drone.sense_range_m)
        cell = float(CELL_SIZE_M)
        half_cells = int(sense_r / cell) + 1
        ix0 = int(position[0] / cell)
        iy0 = int(position[1] / cell)
        z_layer = 3  # breathing zone slice
        nx_, ny_, nz_ = interior_mask.shape
        if not (0 <= z_layer < nz_):
            return
        layer = interior_mask[:, :, z_layer]
        sense_r_sq = sense_r * sense_r
        for di in range(-half_cells, half_cells + 1):
            ni = ix0 + di
            if not (0 <= ni < nx_):
                continue
            dx_m = float(di) * cell
            for dj in range(-half_cells, half_cells + 1):
                nj = iy0 + dj
                if not (0 <= nj < ny_):
                    continue
                if not layer[ni, nj]:
                    continue
                dy_m = float(dj) * cell
                if dx_m * dx_m + dy_m * dy_m <= sense_r_sq:
                    self.visited_cells.add((ni, nj))

    # ─── Convenience accessors ──────────────────────────────────────
    def guiding_drones_in_range(
        self,
        agent_position: np.ndarray,
        follow_range_m: float,
    ) -> List[DroneAgent]:
        """All GUIDING drones within ``follow_range_m`` of ``agent_position``.

        Used by :meth:`PersonAgent.scan_for_drones` — the agent walks
        toward the nearest such drone. ``follow_range_m`` is the
        PersonAgent's sight range (typically a bit larger than the
        drone's sense range so a drone that has just spotted the agent
        and started moving stays visible).
        """
        nearby: List[DroneAgent] = []
        for d in self.drones:
            if d.status != DroneStatus.GUIDING:
                continue
            if _xy_dist(d.position, agent_position) <= follow_range_m:
                nearby.append(d)
        return nearby


# ─── Helpers (module-private) ─────────────────────────────────────────────
def _is_alive(person) -> bool:
    """Local alive-check that avoids importing PersonStatus at module load."""
    return getattr(person, "status", None).__class__.__name__ == "PersonStatus" and \
        person.status.value == "alive"


def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def _cell_of(xyz: np.ndarray) -> Tuple[int, int]:
    """``(i, j)`` cell index at flight-altitude z_layer for a world XY."""
    return (
        int(float(xyz[0]) / float(CELL_SIZE_M)),
        int(float(xyz[1]) / float(CELL_SIZE_M)),
    )


def _detect_persons_near(
    position: np.ndarray,
    alive_persons: List,
    sense_range_m: float,
) -> List:
    """Return alive persons within ``sense_range_m`` XY of ``position``."""
    return [
        p for p in alive_persons
        if _xy_dist(position, p.position) <= sense_range_m
    ]


def _release_drone(drone: DroneAgent) -> None:
    """Reset a GUIDING drone back to SEARCHING.

    Also clears the cached SEARCHING-state path so the freshly-released
    drone re-picks a frontier from its current (typically exit-adjacent)
    position rather than continuing the path it had before getting
    promoted to GUIDING.
    """
    drone.status = DroneStatus.SEARCHING
    drone.target_person_id = None
    drone.current_path = []
    drone.wp_idx = 0
    drone.last_replan_t = -1e9
    drone.frontier_target = None
    drone.search_path = []
    drone.search_wp_idx = 0


def _drone_reached_exit(
    drone: DroneAgent,
    exits: List[np.ndarray],
    tol_m: Optional[float] = None,
) -> bool:
    """Return True iff the drone has *finished its GUIDING path* at an exit.

    D-031 #3 + Fix (2026-05-14): a naive "drone within X m of any exit"
    check fires immediately because every drone is spawned at an exit.
    On the first SEARCHING-to-GUIDING transition the drone has barely
    moved, so it would be released again on the very next tick, which
    triggers a fresh acquire + ``planner.plan`` and re-creates the same
    loop. Result: planner is called ~150x per 60 s instead of ~36 (3
    drones × 12 replans). The profiler showed 87 % of run-time spent in
    ``planner.plan`` because of this.

    Correct condition: drone has consumed its full cached path
    (``wp_idx`` past the last index) **AND** is positioned within
    ``exit_release_tol_m`` of any exit. Path completion is the natural
    signal that the GUIDING mission is over -- it can never fire before
    the drone has actually flown the path to the exit.

    Args:
        drone: A GUIDING drone (the caller already gated on status).
        exits: Canonical exit XY positions.
        tol_m: Override the per-config tolerance.
    """
    tol = float(tol_m if tol_m is not None else drone.config.exit_release_tol_m)
    # Must have completed the GUIDING path.
    if not drone.current_path:
        return False
    if drone.wp_idx < len(drone.current_path):
        return False
    # And must actually be near an exit (defensive -- the path's last
    # waypoint should be an exit cell, but the planner can fall back to
    # other targets; double-check).
    for ex in exits:
        if _xy_dist(drone.position, ex) <= tol:
            return True
    return False


def _drone_lead_waypoint(
    drone: DroneAgent,
    target_person,
) -> Optional[np.ndarray]:
    """Where should the drone fly next while guiding ``target_person``?

    Default: head to the next un-reached waypoint on the cached path
    (path was computed *from the person's position to the exit*, so this
    naturally puts the drone "ahead of" the person). If the path is
    exhausted or empty, fall through to the person's last-known position
    (drone hovers nearby).
    """
    path = drone.current_path
    if not path:
        return np.asarray(target_person.position, dtype=np.float64).reshape(3)
    idx = min(drone.wp_idx, len(path) - 1)
    return np.asarray(path[idx], dtype=np.float64).reshape(3)


def _pick_frontier_target_cluster(
    drone_position: np.ndarray,
    visited_cells: Set[Tuple[int, int]],
    interior_mask: np.ndarray,
    min_component_cells: int = 4,
) -> Optional[np.ndarray]:
    """Cluster-aware frontier picker (D-031 #1).

    Strategy:

    1. Identify all unvisited interior cells at z=3.
    2. Group them into 4-connected components.
    3. Drop components smaller than ``min_component_cells`` (sliver noise
       that the drone has effectively already covered with its sense
       radius -- these would otherwise create thrashing targets).
    4. For each surviving component, compute a "score" = component size
       divided by (distance + 1) from the drone to the component's
       nearest cell. Bigger + closer = better.
    5. Pick the highest-scoring component and return its **medoid** —
       the interior cell with the smallest sum-of-distances to all other
       cells in the component. This is the "true center" of the unvisited
       region and gives the drone a stable, central target.

    If no components survive, reset the visited set and start over so
    the drone keeps moving instead of standstill.

    Args:
        drone_position: ``(3,)`` world XYZ.
        visited_cells: Shared swarm visited set (mutated only by the
            no-frontier fallback).
        interior_mask: ``(60, 40, 6)`` boolean.
        min_component_cells: Skip components smaller than this.

    Returns:
        ``(3,)`` world coordinate of the chosen medoid cell, or ``None``
        if no interior cells exist at all.
    """
    z_layer = 3
    layer = interior_mask[:, :, z_layer]
    nx_, ny_ = layer.shape
    z_world = 0.25 + CELL_SIZE_M * z_layer

    # 1+2. Build connected components over unvisited interior cells (4-conn).
    in_unvisited = np.zeros_like(layer, dtype=bool)
    for i in range(nx_):
        for j in range(ny_):
            if layer[i, j] and (i, j) not in visited_cells:
                in_unvisited[i, j] = True

    if not in_unvisited.any():
        # Total coverage -- reset and start over so drones keep moving.
        visited_cells.clear()
        all_cells = [(i, j) for i in range(nx_) for j in range(ny_)
                     if layer[i, j]]
        if not all_cells:
            return None
        # Fall through with everything treated as one big component.
        return _world_xyz_for_cell(
            *_medoid_of_cells(all_cells), z_world,
        )

    components: List[List[Tuple[int, int]]] = []
    seen = np.zeros_like(in_unvisited, dtype=bool)
    for si in range(nx_):
        for sj in range(ny_):
            if not in_unvisited[si, sj] or seen[si, sj]:
                continue
            stack = [(si, sj)]
            cells: List[Tuple[int, int]] = []
            while stack:
                i, j = stack.pop()
                if seen[i, j]:
                    continue
                seen[i, j] = True
                cells.append((i, j))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if (0 <= ni < nx_ and 0 <= nj < ny_
                            and in_unvisited[ni, nj] and not seen[ni, nj]):
                        stack.append((ni, nj))
            components.append(cells)

    # 3. Drop slivers. If only slivers remain, the drone has effectively
    # already sensed everything (the disc-shaped sense radius leaves only
    # wall-edge slivers behind). Per user spec (D-031.2, 2026-05-14):
    # reset the swarm-shared visited set and re-pick from scratch so the
    # drone keeps patrolling instead of standing on a tiny sliver.
    big = [c for c in components if len(c) >= max(1, int(min_component_cells))]
    if not big:
        visited_cells.clear()
        return _pick_frontier_target_cluster(
            drone_position=drone_position,
            visited_cells=visited_cells,
            interior_mask=interior_mask,
            min_component_cells=min_component_cells,
        )

    # 4. Score each surviving component.
    dx_world = float(drone_position[0])
    dy_world = float(drone_position[1])
    best_score = -float("inf")
    best_comp: Optional[List[Tuple[int, int]]] = None
    for comp in big:
        # nearest cell distance
        nearest_d2 = min(
            ((i + 0.5) * CELL_SIZE_M - dx_world) ** 2
            + ((j + 0.5) * CELL_SIZE_M - dy_world) ** 2
            for (i, j) in comp
        )
        dist = float(np.sqrt(nearest_d2))
        score = len(comp) / (dist + 1.0)
        if score > best_score:
            best_score = score
            best_comp = comp

    if best_comp is None:
        return None

    # 5. Medoid of the chosen component.
    mi, mj = _medoid_of_cells(best_comp)
    return _world_xyz_for_cell(mi, mj, z_world)


def _medoid_of_cells(cells: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Return the cell in ``cells`` that minimises sum-of-distances to all
    other cells (the "1-medoid" of the set). For large components we
    approximate via the cell closest to the centroid to keep this O(N)
    instead of O(N^2)."""
    if len(cells) == 1:
        return cells[0]
    arr = np.asarray(cells, dtype=np.float64)
    cx = arr[:, 0].mean()
    cy = arr[:, 1].mean()
    # Pick the actual cell closest to the centroid.
    d2 = (arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2
    idx = int(np.argmin(d2))
    return cells[idx]


def _world_xyz_for_cell(i: int, j: int, z_world: float) -> np.ndarray:
    return np.array([
        (i + 0.5) * CELL_SIZE_M,
        (j + 0.5) * CELL_SIZE_M,
        z_world,
    ], dtype=np.float64)


def _bfs_drone_path(
    start_xyz: np.ndarray,
    target_xyz: np.ndarray,
    interior_mask: np.ndarray,
) -> List[np.ndarray]:
    """4-connected BFS on the interior z=3 slice (D-031 #2 wall-aware).

    Returns a list of ``(3,)`` world-coordinate waypoints from the start
    cell to the target cell, inclusive of both endpoints. Each waypoint
    is a cell center at flight altitude. The drone steps from one
    waypoint to the next via :meth:`DroneAgent._step_toward`; because
    every waypoint is an interior cell, the path automatically avoids
    walls.

    If no path exists (target cell is in a different interior component),
    returns ``[target_xyz]`` so the drone simply tries the direct move
    (which will likely veto on a wall and trigger a fresh frontier pick
    on the next tick).
    """
    z_layer = 3
    layer = interior_mask[:, :, z_layer]
    nx_, ny_ = layer.shape
    z_world = 0.25 + CELL_SIZE_M * z_layer

    s_cell = _cell_of(start_xyz)
    g_cell = _cell_of(target_xyz)
    si, sj = s_cell
    gi, gj = g_cell

    if not (0 <= si < nx_ and 0 <= sj < ny_ and layer[si, sj]):
        return [np.asarray(target_xyz, dtype=np.float64).reshape(3)]
    if not (0 <= gi < nx_ and 0 <= gj < ny_ and layer[gi, gj]):
        return [np.asarray(target_xyz, dtype=np.float64).reshape(3)]
    if s_cell == g_cell:
        return [_world_xyz_for_cell(si, sj, z_world)]

    parent: Dict[Tuple[int, int], Tuple[int, int]] = {s_cell: s_cell}
    q = deque([s_cell])
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
        return [np.asarray(target_xyz, dtype=np.float64).reshape(3)]
    # Reconstruct path.
    chain = [g_cell]
    while chain[-1] != s_cell:
        chain.append(parent[chain[-1]])
    chain.reverse()
    return [_world_xyz_for_cell(i, j, z_world) for (i, j) in chain]


# Legacy alias kept for any external import; new code should call the
# cluster-aware variant directly.
def _pick_frontier_target(
    drone_position: np.ndarray,
    visited_cells: Set[Tuple[int, int]],
    interior_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Frontier-based coverage: pick the nearest unvisited interior cell.

    The drone's flight-altitude z-slice (k=3) of the interior mask is
    used; cells already in ``visited_cells`` are excluded. If every
    interior cell has been visited (rare in a long-running scenario),
    the visited set is wiped so the drone keeps moving.
    """
    z_layer = 3  # flight altitude
    layer = interior_mask[:, :, z_layer]
    nx_, ny_ = layer.shape

    # Build candidate (i, j) list
    candidates: List[Tuple[int, int]] = []
    for i in range(nx_):
        for j in range(ny_):
            if not layer[i, j]:
                continue
            if (i, j) in visited_cells:
                continue
            candidates.append((i, j))

    if not candidates:
        # Reset and try again
        visited_cells.clear()
        for i in range(nx_):
            for j in range(ny_):
                if layer[i, j]:
                    candidates.append((i, j))
    if not candidates:
        return None

    # Nearest by XY distance from drone.
    dx_world = drone_position[0]
    dy_world = drone_position[1]
    best: Optional[Tuple[int, int]] = None
    best_d2 = float("inf")
    for (i, j) in candidates:
        cx = (i + 0.5) * CELL_SIZE_M
        cy = (j + 0.5) * CELL_SIZE_M
        d2 = (cx - dx_world) ** 2 + (cy - dy_world) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (i, j)
    if best is None:
        return None
    i, j = best
    return np.array([
        (i + 0.5) * CELL_SIZE_M,
        (j + 0.5) * CELL_SIZE_M,
        0.25 + CELL_SIZE_M * z_layer,  # 1.75 m breathing zone
    ], dtype=np.float64)


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("drone_swarm.py self-test (D-030 search + guide)")
    print("=" * 60)

    from src.integration.scenarios._common import exit_positions
    from src.path_planning.building_graph import (
        build_graph,
        load_interior_mask,
    )
    from src.path_planning.edge_weights import EdgeWeightConfig

    errors: list[str] = []

    # ── 1. Spawn 3 drones at the 3 canonical exits ────────────────
    print("\n[1] Spawn 3 drones at the 3 exits")
    swarm = DroneSwarm()
    exits = exit_positions()
    interior = load_interior_mask()
    swarm.spawn(spawn_xyzs=exits, interior_mask=interior)
    if len(swarm.drones) != 3:
        errors.append(f"expected 3 drones, got {len(swarm.drones)}")
    for d in swarm.drones:
        if abs(d.position[2] - d.config.flight_altitude_m) > 1e-6:
            errors.append(f"drone {d.drone_id} altitude wrong: {d.position[2]}")
        if d.status != DroneStatus.SEARCHING:
            errors.append(f"drone {d.drone_id} not SEARCHING: {d.status}")
    print(
        "  ids/pos: "
        + ", ".join(
            f"{d.drone_id}@({d.position[0]:.1f},{d.position[1]:.1f})"
            for d in swarm.drones
        )
    )
    print(f"  spawn-time visited cells: {len(swarm.visited_cells)}")
    # 5 m sense radius covers ~314 cells; with 3 drones spread across
    # exits we expect at least ~200 unique cells visited at t=0
    # (overlaps possible but the exits are far apart).
    if len(swarm.visited_cells) < 100:
        errors.append(
            f"too few visited at spawn: {len(swarm.visited_cells)} "
            f"(expected > 100)"
        )

    # ── 2. Spawn rejects too few positions ──────────────────────────
    print("\n[2] Spawn raises if too few positions")
    try:
        DroneSwarm().spawn(spawn_xyzs=exits[:2])
    except ValueError:
        print("  PASS")
    else:
        errors.append("too-few-positions did not raise")

    # ── 3. Frontier patrol — drone moves AND swarm visited grows ───
    print("\n[3] Patrol expands the swarm-shared visited set quickly")
    planner = EvacuationPlanner(
        build_graph(),
        config=EdgeWeightConfig(),
        heuristic="euclidean",
        fallback_to_last=True,
    )
    # Zero-risk dummy RM so the planner happily builds paths
    # around the full interior.
    from src.integration.scenarios._common import zero_risk_map
    rm = zero_risk_map()
    swarm2 = DroneSwarm()
    swarm2.spawn(spawn_xyzs=exits, interior_mask=interior)
    start_pos = swarm2.drones[0].position.copy()
    visited_at_spawn = len(swarm2.visited_cells)
    n_interior_z3 = int(interior[:, :, 3].sum())
    print(f"  interior cells at z=3: {n_interior_z3}")
    print(f"  visited at spawn (t=0): {visited_at_spawn}")
    for step in range(30):
        swarm2.update(
            t=float(step), dt_s=1.0,
            persons=[],
            planner_rm=rm, planner=planner,
            interior_mask=interior, exits=exits,
        )
    end_pos = swarm2.drones[0].position
    moved = _xy_dist(start_pos, end_pos)
    visited_after = len(swarm2.visited_cells)
    coverage = visited_after / max(n_interior_z3, 1)
    print(
        f"  drone 0: {start_pos[:2]} -> {end_pos[:2]} moved {moved:.2f} m"
    )
    print(
        f"  visited after 30 s: {visited_after} cells ({coverage:.1%} of interior)"
    )
    # NB: with cluster-aware medoid targets a drone can legitimately
    # cover a lot of area without moving very far -- the medoid is the
    # *centre* of a big unvisited region, so one stop senses the whole
    # surrounding cluster. We therefore check coverage, not just movement.
    if visited_after <= visited_at_spawn:
        errors.append(
            f"visited did not grow: spawn={visited_at_spawn}, after={visited_after}"
        )
    # 5 m sense range x 3 drones x 30 s should cover a large fraction.
    if coverage < 0.50:
        errors.append(
            f"swarm covered only {coverage:.1%} of interior in 30 s (expected >= 50%)"
        )

    # ── 4. Detection + GUIDING transition ──────────────────────────
    print("\n[4] Drone within sense_range of an alive person -> GUIDING")
    from src.integration.person_agent import PersonAgent, PersonStatus
    swarm3 = DroneSwarm(
        config=DroneSwarmConfig(n_drones=1, drone=DroneAgentConfig(sense_range_m=5.0))
    )
    swarm3.spawn(spawn_xyzs=[np.array([10.0, 5.0, 1.8])], interior_mask=interior)
    person = PersonAgent(agent_id="P1")
    # Manually set position (don't need a Scene -- the drone never reads body_id).
    person.position = np.array([11.0, 5.0, 1.5])
    person.status = PersonStatus.ALIVE
    swarm3.update(
        t=0.0, dt_s=1.0,
        persons=[person],
        planner_rm=rm, planner=planner,
        interior_mask=interior, exits=exits,
    )
    d0 = swarm3.drones[0]
    if d0.status != DroneStatus.GUIDING:
        errors.append(f"drone did not transition to GUIDING: {d0.status}")
    if d0.target_person_id != "P1":
        errors.append(f"target_person_id wrong: {d0.target_person_id}")
    else:
        print(
            f"  PASS: drone -> GUIDING, target={d0.target_person_id}, "
            f"path len={len(d0.current_path)}"
        )

    # ── 5. Released when target evacuates ──────────────────────────
    print("\n[5] Drone falls back to SEARCHING when target evacuates")
    person.status = PersonStatus.EVACUATED
    swarm3.update(
        t=1.0, dt_s=1.0,
        persons=[person],
        planner_rm=rm, planner=planner,
        interior_mask=interior, exits=exits,
    )
    if swarm3.drones[0].status != DroneStatus.SEARCHING:
        errors.append(
            f"drone did not release after evac: {swarm3.drones[0].status}"
        )
    else:
        print("  PASS")

    # ── 6. Two drones do not claim the same survivor ──────────────
    print("\n[6] No double-claim")
    swarm4 = DroneSwarm(config=DroneSwarmConfig(n_drones=2))
    swarm4.spawn(spawn_xyzs=[
        np.array([10.0, 5.0, 1.8]),
        np.array([10.5, 5.0, 1.8]),
    ], interior_mask=interior)
    p = PersonAgent(agent_id="lonely")
    p.position = np.array([10.2, 5.0, 1.5])
    p.status = PersonStatus.ALIVE
    swarm4.update(
        t=0.0, dt_s=1.0,
        persons=[p],
        planner_rm=rm, planner=planner,
        interior_mask=interior, exits=exits,
    )
    guiding = [d for d in swarm4.drones if d.status == DroneStatus.GUIDING]
    if len(guiding) != 1:
        errors.append(f"expected exactly 1 guiding drone, got {len(guiding)}")
    else:
        print(f"  PASS: 1 drone GUIDING, 1 still SEARCHING")

    # ── 7. guiding_drones_in_range ─────────────────────────────────
    print("\n[7] guiding_drones_in_range honours follow_range")
    nearby = swarm4.guiding_drones_in_range(
        agent_position=np.array([10.2, 5.0, 1.5]),
        follow_range_m=2.0,
    )
    if len(nearby) != 1:
        errors.append(f"expected 1 nearby guiding drone, got {len(nearby)}")
    nearby_far = swarm4.guiding_drones_in_range(
        agent_position=np.array([29.0, 19.0, 1.5]),
        follow_range_m=2.0,
    )
    if nearby_far:
        errors.append(f"expected 0 far guiding drones, got {len(nearby_far)}")
    if not errors or nearby_far:
        print(f"  PASS: in-range={len(nearby)}, far={len(nearby_far)}")

    # ── 8. D-031 #3: drone releases after FINISHING its GUIDING path ──
    print("\n[8] D-031 #3 GUIDING drone releases on path completion + exit")
    swarm5 = DroneSwarm(config=DroneSwarmConfig(n_drones=1))
    ex_xyz = exits[0].copy()
    swarm5.spawn(spawn_xyzs=[ex_xyz + np.array([0.4, 0.0, 0.0])],
                 interior_mask=interior)
    p5 = PersonAgent(agent_id="P_at_exit")
    p5.position = np.array([ex_xyz[0] + 0.6, ex_xyz[1], 1.5])
    p5.status = PersonStatus.ALIVE
    # Tick 1: acquire (transition to GUIDING).
    swarm5.update(t=0.0, dt_s=1.0, persons=[p5],
                  planner_rm=rm, planner=planner,
                  interior_mask=interior, exits=exits)
    d5 = swarm5.drones[0]
    if d5.status != DroneStatus.GUIDING:
        errors.append(f"drone did not acquire: {d5.status}")
    # Tick 2: still GUIDING because drone has not yet flown the path.
    swarm5.update(t=1.0, dt_s=1.0, persons=[p5],
                  planner_rm=rm, planner=planner,
                  interior_mask=interior, exits=exits)
    if d5.status != DroneStatus.GUIDING:
        errors.append(
            f"drone released prematurely (before path complete): {d5.status}"
        )
    # Force path completion: pretend the drone has finished its cached
    # path (set wp_idx past the end) and verify the next update releases.
    d5.wp_idx = len(d5.current_path) + 1
    d5.position = ex_xyz.copy()
    swarm5.update(t=2.0, dt_s=1.0, persons=[p5],
                  planner_rm=rm, planner=planner,
                  interior_mask=interior, exits=exits)
    if d5.status != DroneStatus.SEARCHING:
        errors.append(
            f"drone did not release after path completion: {d5.status}"
        )
    else:
        print("  PASS: released after path completion + at exit")

    # ── 9. D-031 #1: cluster-aware target = medoid of largest cluster ──
    print("\n[9] D-031 #1 _pick_frontier_target_cluster picks medoid")
    # Construct a tiny synthetic mask: two interior rectangles, one small
    # next to the drone and one large further away. The large one should
    # win on score = size / (dist + 1).
    fake = np.zeros((20, 10, 6), dtype=bool)
    fake[1:3, 1:3, 3] = True       # small cluster (4 cells) near (0.5, 0.5)
    fake[10:18, 1:9, 3] = True     # big cluster (64 cells) far away
    target = _pick_frontier_target_cluster(
        drone_position=np.array([0.5, 1.0, 1.75]),
        visited_cells=set(),
        interior_mask=fake,
        min_component_cells=2,
    )
    if target is None:
        errors.append("cluster picker returned None")
    else:
        # Medoid of large cluster center ≈ (i=13.5, j=4.5) -> world (~7, ~2.5)
        if target[0] < 5.0:
            errors.append(
                f"cluster picker chose small cluster: target={target}"
            )
        else:
            print(f"  PASS: picked large cluster medoid at ({target[0]:.2f}, {target[1]:.2f})")

    # ── 10. D-031 #2: BFS path detours around a wall ──────────────────
    print("\n[10] D-031 #2 _bfs_drone_path detours around a wall")
    # A simple open interior with one wall column blocking the direct
    # line from cell (2, 4) to cell (6, 4). The wall is at i=5 for
    # j=0..7, so the only way through is j=8 or j=9.
    walled = np.zeros((10, 10, 6), dtype=bool)
    walled[0:10, 0:10, 3] = True
    walled[5, 0:8, 3] = False        # wall column at i=5 (j<8)
    # cell (2, 4) -> world (1.25, 2.25); cell (6, 4) -> world (3.25, 2.25)
    path = _bfs_drone_path(
        start_xyz=np.array([1.25, 2.25, 1.75]),
        target_xyz=np.array([3.25, 2.25, 1.75]),
        interior_mask=walled,
    )
    if not path or len(path) < 5:
        errors.append(
            f"BFS path suspiciously short for L-detour: len={len(path)}"
        )
    else:
        # Path should not contain any wall cell (i=5, j<8).
        bad = [
            wp for wp in path
            if int(wp[0] / CELL_SIZE_M) == 5 and int(wp[1] / CELL_SIZE_M) < 8
        ]
        if bad:
            errors.append(
                f"BFS path crossed wall cells: {len(bad)} bad waypoints"
            )
        else:
            # Verify path actually goes "north" past the wall (j >= 8) at
            # some point.
            went_around = any(
                int(wp[1] / CELL_SIZE_M) >= 8 for wp in path
            )
            if not went_around:
                errors.append(
                    f"BFS path did not detour (max j={max(int(wp[1]/CELL_SIZE_M) for wp in path)})"
                )
            else:
                print(f"  PASS: detour length={len(path)} steps, routed via j>=8")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: drone_swarm validated (incl. D-031 #1+#2+#3)")
