"""Simulation state recorder for offline visualisation.

Decouples visualisation from simulation: scenarios call
:meth:`SimulationRecorder.record` once per tick, downstream renderers
read a stable frame schema. The simulation never needs to know what
will eventually be plotted.

**Stability contract** (so future visualisation features don't force
edits to ``s1/s2/s3.run`` or :class:`PersonAgent`):

* :class:`AgentFrame` and :class:`SimulationFrame` add new state via
  ``extras: dict``. Existing fields are stable.
* :meth:`SimulationRecorder.record` keeps signature stable. New optional
  hooks land as keyword-only parameters with sensible defaults.
* Risk-grid capture is opt-in (off by default; one frame queries 2400
  cells, so capturing every tick is expensive).

Typical usage in a scenario::

    recorder = SimulationRecorder(scenario_id="S2_fds_swarm")
    ...
    for tick in outer_loop:
        ...
        if recorder is not None:
            recorder.record(t=t_now, agents=agents, risk_map=truth_rm,
                            arrived={a.agent_id for a in agents if ...})

Future renderers consume ``recorder.frames``; new fields ride on
``frame.extras`` / ``agent.extras``.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.shared.constants import CELL_SIZE_M, GRID_SHAPE


# ─── Frame dataclasses ────────────────────────────────────────────────────
@dataclass
class AgentFrame:
    """One agent's state at one tick. Extend via ``extras``."""

    agent_id: str
    pos: Tuple[float, float, float]
    status: str = "alive"            # alive / evacuated / dead
    arrived: bool = False
    extras: dict = field(default_factory=dict)


@dataclass
class DroneFrame:
    """One drone's state at one tick (D-030)."""

    drone_id: str
    pos: Tuple[float, float, float]
    status: str = "searching"        # searching / guiding / idle
    target_person_id: Optional[str] = None
    extras: dict = field(default_factory=dict)


@dataclass
class SimulationFrame:
    """All recorded state at one tick. Extend via ``extras``."""

    t: float
    scenario_id: str = ""
    fire_scenario_id: str = ""
    seed: int = 0
    agents: List[AgentFrame] = field(default_factory=list)
    drones: List[DroneFrame] = field(default_factory=list)
    risk_grid: Optional[np.ndarray] = None  # (60, 40) at z_layer if captured
    extras: dict = field(default_factory=dict)


# ─── Recorder ─────────────────────────────────────────────────────────────
@dataclass
class SimulationRecorder:
    """Append-only per-tick frame log.

    Attributes:
        scenario_id: Label propagated into each frame (e.g. ``"S2_fds_swarm"``).
        fire_scenario_id: FDS scenario name (e.g. ``"sim_1500kw_2m2_T05"``).
        seed: RNG seed used for spawn.
        capture_risk_grid: If True, every recorded frame samples the
            risk map on a (60, 40) grid at ``z_layer``. Expensive --
            default False; renderers may compute risk on-demand from a
            stored :class:`RiskMap`.
        z_layer: Risk-grid z-slice (matches D-026 breathing zone k=3).
        capture_every: Subsample N ticks (e.g. ``5`` records every 5th
            tick). Default ``1`` = every tick.
        frames: Accumulated :class:`SimulationFrame` list.
    """

    scenario_id: str = ""
    fire_scenario_id: str = ""
    seed: int = 0
    capture_risk_grid: bool = False
    z_layer: int = 3
    capture_every: int = 1
    frames: List[SimulationFrame] = field(default_factory=list)
    _tick_count: int = 0

    # ── Hot path ─────────────────────────────────────────────────────
    def record(
        self,
        *,
        t: float,
        agents: Iterable[Any],
        risk_map: Optional[Any] = None,
        arrived: Optional[Set[str]] = None,
        agent_extras_fn: Optional[Callable[[Any], dict]] = None,
        frame_extras: Optional[dict] = None,
        drones: Optional[Iterable[Any]] = None,
        drone_extras_fn: Optional[Callable[[Any], dict]] = None,
    ) -> None:
        """Capture one tick.

        Args:
            t: Simulation time (s).
            agents: Iterable of objects exposing ``agent_id`` and
                ``position`` (numpy array of length 3). Compatible with
                :class:`~src.integration.person_agent.PersonAgent` today
                and any future agent type that follows the same minimum
                duck-type.
            risk_map: Optional :class:`~src.risk_map.risk_map_class.RiskMap`.
                Used only if ``capture_risk_grid=True`` (off by default).
            arrived: Set of agent IDs that have evacuated by this tick.
                When provided, those agents' ``status`` is recorded as
                ``"evacuated"`` and ``arrived=True``.
            agent_extras_fn: Optional callable ``(agent) -> dict`` to
                attach extra per-agent telemetry (FED, danger, etc.) to
                ``AgentFrame.extras``. Returning ``None`` is treated as
                empty dict.
            frame_extras: Optional dict merged into ``SimulationFrame.extras``
                (drone positions, fire epicentre, etc.).
        """
        self._tick_count += 1
        if self.capture_every > 1 and (self._tick_count - 1) % self.capture_every != 0:
            return

        arrived_set: Set[str] = arrived if arrived is not None else set()
        agent_frames: List[AgentFrame] = []
        for a in agents:
            aid = getattr(a, "agent_id", "?")
            pos = getattr(a, "position", None)
            if pos is None:
                continue
            pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
            is_arrived = aid in arrived_set
            # Status priority order:
            # (1) ``arrived`` set wins (scenario-side truth; PersonAgent's
            #     internal status is not yet updated by the run loop --
            #     M2-full work).
            # (2) Otherwise read the agent's own ``status`` (enum or str).
            # (3) Fallback to "alive".
            if is_arrived:
                status_str = "evacuated"
            else:
                status = getattr(a, "status", None)
                if status is not None:
                    status_str = getattr(status, "value", str(status))
                else:
                    status_str = "alive"
            extras = {}
            if agent_extras_fn is not None:
                try:
                    extras = agent_extras_fn(a) or {}
                except Exception:  # noqa: BLE001
                    # Visualisation must never break the sim.
                    extras = {}
            agent_frames.append(AgentFrame(
                agent_id=aid,
                pos=pos_t,
                status=status_str,
                arrived=is_arrived,
                extras=dict(extras),
            ))

        # ── Drones (D-030) ─────────────────────────────────────────
        drone_frames: List[DroneFrame] = []
        if drones is not None:
            for d in drones:
                did = getattr(d, "drone_id", "?")
                pos = getattr(d, "position", None)
                if pos is None:
                    continue
                pos_t = (float(pos[0]), float(pos[1]), float(pos[2]))
                d_status = getattr(d, "status", None)
                status_str = (
                    getattr(d_status, "value", str(d_status))
                    if d_status is not None else "searching"
                )
                target_id = getattr(d, "target_person_id", None)
                extras: dict = {}
                if drone_extras_fn is not None:
                    try:
                        extras = drone_extras_fn(d) or {}
                    except Exception:  # noqa: BLE001
                        extras = {}
                drone_frames.append(DroneFrame(
                    drone_id=did,
                    pos=pos_t,
                    status=status_str,
                    target_person_id=target_id,
                    extras=dict(extras),
                ))

        risk_grid = None
        if self.capture_risk_grid and risk_map is not None:
            risk_grid = self._sample_risk_grid(risk_map, t)

        frame = SimulationFrame(
            t=float(t),
            scenario_id=self.scenario_id,
            fire_scenario_id=self.fire_scenario_id,
            seed=self.seed,
            agents=agent_frames,
            drones=drone_frames,
            risk_grid=risk_grid,
            extras=dict(frame_extras or {}),
        )
        self.frames.append(frame)

    # ── Risk grid sampler ────────────────────────────────────────────
    def _sample_risk_grid(self, risk_map: Any, t: float) -> np.ndarray:
        nx_, ny_, _ = GRID_SHAPE
        z = 0.25 + CELL_SIZE_M * self.z_layer
        # Batch query: (N, 3) of (x, y, z) for every cell centre.
        xs = 0.25 + CELL_SIZE_M * np.arange(nx_)
        ys = 0.25 + CELL_SIZE_M * np.arange(ny_)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")  # both (nx, ny)
        pts = np.stack([xx, yy, np.full_like(xx, z)], axis=-1).reshape(-1, 3)
        try:
            vals = np.asarray(risk_map.query(pts, t=t), dtype=np.float32)
        except Exception:  # noqa: BLE001
            # Fall back to per-cell loop if the RiskMap doesn't support
            # batch.
            vals = np.empty(pts.shape[0], dtype=np.float32)
            for k, p in enumerate(pts):
                vals[k] = float(risk_map.query(p, t=t))
        return vals.reshape(nx_, ny_)

    # ── Persistence ──────────────────────────────────────────────────
    def save_pickle(self, path: Path) -> Path:
        """Save the full frame log as a pickle. Round-trips losslessly."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load_pickle(cls, path: Path) -> "SimulationRecorder":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected SimulationRecorder, got {type(obj).__name__}")
        return obj

    def save_summary_json(self, path: Path) -> Path:
        """Save a JSON summary (no risk grids; agent extras serialised
        if JSON-compatible). Smaller and human-readable; use this for
        diffing / quick inspection."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "scenario_id": self.scenario_id,
            "fire_scenario_id": self.fire_scenario_id,
            "seed": self.seed,
            "n_frames": len(self.frames),
            "z_layer": self.z_layer,
            "capture_every": self.capture_every,
            "frames": [
                {
                    "t": fr.t,
                    "agents": [
                        {
                            "agent_id": ag.agent_id,
                            "pos": list(ag.pos),
                            "status": ag.status,
                            "arrived": ag.arrived,
                            "extras": _coerce_json(ag.extras),
                        }
                        for ag in fr.agents
                    ],
                    "extras": _coerce_json(fr.extras),
                }
                for fr in self.frames
            ],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        return path

    # ── Convenience views ────────────────────────────────────────────
    def agent_ids(self) -> List[str]:
        """All agent IDs that ever appeared in the log (in first-seen order)."""
        seen: List[str] = []
        seen_set: Set[str] = set()
        for fr in self.frames:
            for ag in fr.agents:
                if ag.agent_id not in seen_set:
                    seen.append(ag.agent_id)
                    seen_set.add(ag.agent_id)
        return seen

    def agent_trajectory(self, agent_id: str) -> np.ndarray:
        """Return ``(N, 3)`` array of positions for one agent. Empty if absent."""
        rows: List[Tuple[float, float, float]] = []
        for fr in self.frames:
            for ag in fr.agents:
                if ag.agent_id == agent_id:
                    rows.append(ag.pos)
                    break
        if not rows:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def agent_times(self, agent_id: str) -> np.ndarray:
        """Tick times where this agent had a recorded frame."""
        ts: List[float] = []
        for fr in self.frames:
            for ag in fr.agents:
                if ag.agent_id == agent_id:
                    ts.append(fr.t)
                    break
        return np.asarray(ts, dtype=np.float64)

    def final_status(self, agent_id: str) -> str:
        """Most recent recorded status (defaults to ``"unknown"``)."""
        for fr in reversed(self.frames):
            for ag in fr.agents:
                if ag.agent_id == agent_id:
                    return ag.status
        return "unknown"


def _coerce_json(obj):
    """Best-effort JSON coercion of arbitrary nested values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _coerce_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return repr(obj)


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import tempfile
    from types import SimpleNamespace

    print("=" * 60)
    print("recorder.py self-test")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. Basic record call with duck-typed agents ────────────────
    print("\n[1] record() with duck-typed agents (arrived wins over .status)")
    rec = SimulationRecorder(scenario_id="S_test", seed=0)
    # Mimic PersonAgent: has its own .status attribute (always "alive"
    # for now) -- the arrived= set should still drive the recorded
    # status, since the run-loop tracks arrival separately from the
    # PersonAgent state machine (which is M2-full work).
    fake_agents = [
        SimpleNamespace(
            agent_id="a", position=np.array([1.0, 2.0, 0.85]),
            status=SimpleNamespace(value="alive"),
        ),
        SimpleNamespace(
            agent_id="b", position=np.array([5.0, 6.0, 0.85]),
            status=SimpleNamespace(value="alive"),
        ),
    ]
    rec.record(t=0.0, agents=fake_agents)
    rec.record(t=1.0, agents=fake_agents, arrived={"a"})
    if len(rec.frames) != 2:
        errors.append(f"expected 2 frames, got {len(rec.frames)}")
    if rec.frames[1].agents[0].status != "evacuated":
        errors.append(
            f"arrived agent's status = {rec.frames[1].agents[0].status!r}, "
            f"expected 'evacuated' (arrived must override .status)"
        )
    if rec.frames[1].agents[0].arrived is not True:
        errors.append("arrived flag not set")
    if rec.frames[0].agents[0].status != "alive":
        errors.append("not-arrived agent should retain 'alive' status")
    print(
        f"  PASS: t=0 -> {rec.frames[0].agents[0].status}, "
        f"t=1 'a' arrived -> {rec.frames[1].agents[0].status}"
    )

    # ── 2. capture_every subsamples ────────────────────────────────
    print("\n[2] capture_every=3 subsamples")
    rec2 = SimulationRecorder(capture_every=3)
    for i in range(10):
        rec2.record(t=float(i), agents=fake_agents)
    # ticks 0, 3, 6, 9 -> 4 frames
    if len(rec2.frames) != 4:
        errors.append(f"expected 4 frames with capture_every=3, got {len(rec2.frames)}")
    else:
        print(f"  PASS: {len(rec2.frames)} frames")

    # ── 3. agent_extras_fn populates extras ────────────────────────
    print("\n[3] agent_extras_fn attaches extras")
    rec3 = SimulationRecorder()
    rec3.record(
        t=0.0, agents=fake_agents,
        agent_extras_fn=lambda a: {"id_upper": a.agent_id.upper()},
    )
    if rec3.frames[0].agents[0].extras.get("id_upper") != "A":
        errors.append("agent_extras_fn extras not captured")
    else:
        print("  PASS")

    # ── 4. agent_extras_fn errors don't break sim ──────────────────
    print("\n[4] agent_extras_fn exception is swallowed (sim must continue)")
    rec4 = SimulationRecorder()
    rec4.record(
        t=0.0, agents=fake_agents,
        agent_extras_fn=lambda a: 1 / 0,
    )
    if rec4.frames[0].agents[0].extras != {}:
        errors.append("failed agent_extras_fn left non-empty extras")
    else:
        print("  PASS (extras = {})")

    # ── 5. Trajectory view ─────────────────────────────────────────
    print("\n[5] agent_trajectory + agent_times views")
    traj = rec.agent_trajectory("a")
    times = rec.agent_times("a")
    if traj.shape != (2, 3):
        errors.append(f"trajectory shape {traj.shape} != (2, 3)")
    if not np.allclose(times, [0.0, 1.0]):
        errors.append(f"times = {times}, expected [0.0, 1.0]")
    print(f"  PASS: traj.shape={traj.shape}, times={times}")

    # ── 6. Risk-grid capture with a stub RiskMap ───────────────────
    print("\n[6] capture_risk_grid samples (60, 40) at z_layer=3")
    class StubRiskMap:
        def query(self, xyz, t=None):
            arr = np.asarray(xyz)
            if arr.ndim == 1:
                return 0.5
            return np.full(arr.shape[0], 0.5, dtype=np.float32)
    rec6 = SimulationRecorder(capture_risk_grid=True, z_layer=3)
    rec6.record(t=0.0, agents=fake_agents, risk_map=StubRiskMap())
    grid = rec6.frames[0].risk_grid
    if grid is None or grid.shape != (60, 40):
        errors.append(f"risk_grid shape {None if grid is None else grid.shape} != (60, 40)")
    elif not np.allclose(grid, 0.5):
        errors.append("risk_grid values wrong")
    else:
        print(f"  PASS: grid.shape={grid.shape}, mean={grid.mean():.2f}")

    # ── 7. Save / load pickle round-trip ───────────────────────────
    print("\n[7] save_pickle round-trip")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rec.pkl"
        rec.save_pickle(path)
        rec_loaded = SimulationRecorder.load_pickle(path)
        if len(rec_loaded.frames) != len(rec.frames):
            errors.append("pickle round-trip frame count mismatch")
        elif rec_loaded.scenario_id != "S_test":
            errors.append("pickle round-trip scenario_id lost")
        else:
            print("  PASS")

    # ── 8. JSON summary writes ──────────────────────────────────────
    print("\n[8] save_summary_json produces valid JSON")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rec.json"
        rec.save_summary_json(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload["n_frames"] != 2:
            errors.append("JSON n_frames mismatch")
        elif len(payload["frames"][0]["agents"]) != 2:
            errors.append("JSON agents count mismatch")
        else:
            print(f"  PASS: {payload['n_frames']} frames serialised")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: SimulationRecorder validated")
