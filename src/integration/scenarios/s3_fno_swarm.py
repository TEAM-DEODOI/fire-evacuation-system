"""Scenario S3 -- Drone swarm guided by model-predicted RiskMap (D-025).

Identical to :mod:`src.integration.scenarios.s2_fds_swarm` except the
*planner* consults a model-predicted :class:`RiskMap` rather than the
FDS truth. The occupant's experienced danger is still measured against
``StaticRiskMap.from_fds_dir(...)`` (the truth), so the only thing that
changes between S2 and S3 is **what the planner sees** -- the fairness
setup from ``docs/interface_contracts.md`` §6.

**S2 vs S3 -> H5 transitive** ("does risk-map fidelity translate to
path quality?"). If S3 ≈ S2, then a model that hits H5 on its own is
also deployment-ready as a planner input. If S3 is much worse, model
errors compound into unsafe paths.

**M5-mini status (2026-05-14):**

* ``scenario_id`` is ``"S3_fno_swarm"`` per CLAUDE.md canonical
  naming. The actual model-predicted RiskMap used is
  :class:`Tier1RiskMap` (the project's headline-IoU model). A
  follow-up M5-full commit will swap to a true PI-FNO
  ``FNORiskMap`` without changing the scenario_id or the rest of
  this file.
* The Tier1 GNN forward consumes
  ``results/detector_sequences/<fire_scenario_id>.npz`` plus the
  ``activation_times`` array, fed through the **same encoding** as
  ``Tier1FireDataset`` (training-eval parity, D-053-fix).
* Replan + truth-exposure logic is identical to S2.
* No casualty / FED metrics (M2-full / M3-full prerequisites).

**D-053-fix (2026-05-15) — three issues fixed at once:**

1. **Feature encoding.**  The legacy ``_binary_history_to_features``
   used a *window-relative* trigger-time encoding
   ``(t - first_active) / (T_in - 1)``, but the GNN was trained on
   *absolute* activation times normalised by 300 s
   (``act_times / T_END_SECONDS``).  We now mirror
   :class:`~src.tier1.tier1_dataset.Tier1FireDataset` exactly.
2. **Sliding window.**  The legacy build used the first six frames
   ``binary_seq[:6]`` (t = 0..50 s) for every replan, so it forecast
   t = 60..110 s once and froze.  Slow small fires (e.g. s_029) have
   almost no triggers in t = 0..50 s, so the GNN saw nearly empty
   input — completely off-distribution.  The new build slides the
   window to match the current fire clock: ``history = binary_seq[
   t0/dt - T_in : t0/dt]`` so the forecast covers ``[t0, t0+50 s]``.
3. **No refresh.**  The legacy run loop never rebuilt ``planner_rm``,
   so the swarm consulted a single 50 s forecast for the whole 300 s
   sim.  The loop now refreshes ``planner_rm`` every
   ``replan_period_s`` seconds of fire time (S4-style, D-046).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.integration.drone_swarm import DroneSwarm, DroneSwarmConfig, DroneAgentConfig
from src.integration.metrics import ScenarioMetrics
from src.integration.recorder import SimulationRecorder
from src.integration.scene import Scene, SceneConfig
from src.integration.person_agent import PersonStatus
from src.integration.scenarios._common import (
    BUILDING_URDF,
    PLACEHOLDER_URDF,
    building_urdf_path,
    drone_spawn_positions,
    exit_positions,
    load_truth_co_field,
    load_truth_risk_map,
    spawn_agents,
)
from src.path_planning.building_graph import (
    load_default_fluid_mask,
    load_interior_mask,
)
from src.path_planning.edge_weights import EdgeWeightConfig
from src.path_planning.planners import EvacuationPlanner
from src.risk_map.risk_map_class import RiskMap
from src.shared.constants import CELL_SIZE_M, DT_SLCF, REPLAN_PERIOD_S
from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_dataset import T_END_SECONDS
from src.tier1.tier1_gnn import N_NODES, SimpleFireGNN, build_knn_adjacency
from src.tier1.tier1_risk_map import Tier1RiskMap


_TIER1_CKPT = Path("checkpoints/tier1_gnn_v3/best.pt")
_DETECTOR_SEQ_DIR = Path("results/detector_sequences")
_T_IN = 6
_T_OUT = 6
_F_IN = 5  # [is_det, det_time_norm, room, corridor, exit]

# Valid GNN-forward fire-clock window. The training-time Tier1FireDataset
# enumerates t_start ∈ [0, 19] (inclusive) over a 31-frame sequence with
# T_in=T_out=6, so the earliest valid forecast t0 = T_in * dt = 60 s and
# the latest is (31 - T_out) * dt = 250 s. Outside this window we freeze
# planner_rm at the last successful build (S4-style behaviour, D-046).
_TIER1_MIN_T0_S = 60.0
_TIER1_MAX_T0_S = 250.0


# ─── Tier 1 planner RiskMap construction ──────────────────────────────────
@dataclass(frozen=True)
class _Tier1Artifacts:
    """One-time loaded forward-pass inputs for a given (scenario, ckpt)."""
    model: SimpleFireGNN
    adj: torch.Tensor
    binary_seq: np.ndarray          # (31, N_NODES) float32, latched 0/1
    act_times: np.ndarray           # (N_NODES,) float32, first-trigger time (s); -1 = never
    positions: List[Tuple[float, float, float]]


_TIER1_ART_CACHE: Dict[Tuple[str, str], _Tier1Artifacts] = {}


def _binary_history_to_features(
    binary_history: np.ndarray,
    act_times: np.ndarray,
) -> torch.Tensor:
    """Convert a ``(T_in, N)`` binary history slice + ``(N,)`` activation
    times to the GNN's ``(1, N, T_in, F=5)`` feature tensor.

    Encoding **must** match :class:`~src.tier1.tier1_dataset.Tier1FireDataset`
    (D-053-fix, 2026-05-15)::

        feat[:, t, 0] = is_detected (binary[t, n])
        feat[:, t, 1] = (act_times / T_END_SECONDS) * is_detected
                        ─ absolute, simulation-clock-normalised trigger time
                        ─ masked by current binary so unactivated nodes stay 0
        feat[:, t, 2:5] = node_type_onehot  (room / corridor / exit)

    The legacy implementation used a *window-relative*
    ``(t - first_active) / (T_in - 1)`` which never matched the training
    distribution and silently degraded every inference, especially on
    slow small fires where the trigger times within the window are
    nearly all zero.
    """
    t_in, n = binary_history.shape
    if n != N_NODES:
        raise ValueError(f"expected {N_NODES} sensors, got {n}")
    if act_times.shape != (N_NODES,):
        raise ValueError(
            f"act_times shape {act_times.shape} != ({N_NODES},)"
        )

    x = torch.zeros(1, n, t_in, _F_IN)

    types = ["room", "corridor", "exit"]
    for i, d in enumerate(ALL_DETECTORS):
        x[0, i, :, 2 + types.index(d.node_type)] = 1.0

    det_time_norm = np.where(
        act_times >= 0,
        np.clip(act_times / T_END_SECONDS, 0.0, 1.0),
        0.0,
    ).astype(np.float32)                                   # (N,)
    binary_t = torch.from_numpy(binary_history.astype(np.float32))   # (T_in, N)
    dtn_t = torch.from_numpy(det_time_norm)                          # (N,)
    for t in range(t_in):
        x[0, :, t, 0] = binary_t[t]
        x[0, :, t, 1] = dtn_t * binary_t[t]
    return x


def _load_tier1_artefacts(
    fire_scenario_id: str,
    *,
    ckpt_path: Optional[Path] = None,
    verbose: bool = True,
) -> Optional[_Tier1Artifacts]:
    """Load the GNN checkpoint + detector-sequence inputs once per run.

    Cached by ``(fire_scenario_id, ckpt_path)`` so repeated calls are
    free. Returns ``None`` if either the checkpoint or the detector
    sequence is missing on disk, or the load fails — the caller then
    keeps the FDS-truth fallback.
    """
    ckpt_p = Path(ckpt_path) if ckpt_path is not None else _TIER1_CKPT
    cache_key = (fire_scenario_id, str(ckpt_p))
    cached = _TIER1_ART_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not ckpt_p.exists():
        if verbose:
            print(
                f"  [planner_rm] Tier1 checkpoint missing ({ckpt_p}); "
                f"S3 planner reuses truth RiskMap"
            )
        return None

    seq_path = _DETECTOR_SEQ_DIR / f"{fire_scenario_id}.npz"
    if not seq_path.exists():
        if verbose:
            print(
                f"  [planner_rm] detector sequence missing ({seq_path}); "
                f"S3 planner reuses truth RiskMap"
            )
        return None

    try:
        ckpt = torch.load(ckpt_p, weights_only=False, map_location="cpu")
        cfg = ckpt.get("config", {})
        v5_cfg = ckpt.get("v5_config", {})  # D-053 architecture fallback

        def _cfg_get(key: str, default):
            if key in cfg:
                return cfg[key]
            if key in v5_cfg:
                return v5_cfg[key]
            return default

        model = SimpleFireGNN(
            in_feat=_F_IN,
            hidden=int(_cfg_get("hidden", 32)),
            n_graph_layers=int(_cfg_get("n_graph_layers", 2)),
            T_out=int(_cfg_get("T_out", _T_OUT)),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        adj = build_knn_adjacency(k=int(_cfg_get("knn_k", 4)))

        data = np.load(seq_path)
        binary_seq = data["binary_sequence"].astype(np.float32)   # (31, N)
        act_times = data["activation_times"].astype(np.float32)   # (N,)
        positions = [
            (d.position[0], d.position[1], 1.5) for d in ALL_DETECTORS
        ]

        art = _Tier1Artifacts(
            model=model,
            adj=adj,
            binary_seq=binary_seq,
            act_times=act_times,
            positions=positions,
        )
        _TIER1_ART_CACHE[cache_key] = art
        if verbose:
            print(
                f"  [planner_rm] Tier1 artefacts loaded "
                f"(ckpt={ckpt_p.parent.name}, "
                f"binary_seq={binary_seq.shape}, "
                f"act_times={act_times.shape})"
            )
        return art
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(
                f"  [planner_rm] artefact load failed "
                f"({exc.__class__.__name__}: {exc}); "
                f"S3 planner reuses truth RiskMap"
            )
        return None


def _build_tier1_rm(
    artefacts: _Tier1Artifacts,
    t0_s: float,
    *,
    fluid_mask: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Optional[Tier1RiskMap]:
    """Forward the GNN at fire-clock time ``t0_s`` and wrap as Tier1RiskMap.

    The 6-frame history fed to the GNN is ``binary_seq[t0/dt - T_in :
    t0/dt]``, matching the sliding window the trainer enumerated. The
    returned RiskMap covers ``[t0, t0 + (T_out - 1) * dt]`` = 50 s.

    Out-of-range ``t0_s`` (outside [60, 250] s) yields ``None`` so the
    caller can keep the last good build instead of spam-failing every
    replan tick.
    """
    try:
        t0_frame = int(round(float(t0_s) / DT_SLCF))
        t_start = t0_frame - _T_IN
        n_frames = artefacts.binary_seq.shape[0]
        if t_start < 0 or t_start + _T_IN > n_frames:
            if verbose:
                print(
                    f"  [planner_rm] t0={t0_s:.0f}s OOB for sliding window "
                    f"[{_TIER1_MIN_T0_S:.0f}, {_TIER1_MAX_T0_S:.0f}]s"
                )
            return None

        history = artefacts.binary_seq[t_start : t_start + _T_IN]   # (T_in, N)
        x = _binary_history_to_features(history, artefacts.act_times)
        with torch.no_grad():
            pred = artefacts.model(x, artefacts.adj)                # (1, N, T_out)

        rm = Tier1RiskMap.from_model_output(
            pred,
            artefacts.positions,
            batch_index=0,
            start_time=float(t0_s),
            dt=float(DT_SLCF),
            fluid_mask=fluid_mask,    # D-050: enables geodesic IDW
        )
        if verbose:
            mode = "geodesic-IDW" if fluid_mask is not None else "nearest-node"
            hist_lo = t_start * DT_SLCF
            hist_hi = (t_start + _T_IN - 1) * DT_SLCF
            print(
                f"  [planner_rm] refresh t0={t0_s:.0f}s "
                f"(history t={hist_lo:.0f}..{hist_hi:.0f}s, "
                f"forecast t={t0_s:.0f}..{rm.t_max:.0f}s, "
                f"interp={mode})"
            )
        return rm
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(
                f"  [planner_rm] forward FAILED at t0={t0_s:.0f}s "
                f"({exc.__class__.__name__}: {exc})"
            )
        return None


def _load_tier1_planner_rm(
    fire_scenario_id: str,
    fall_back: RiskMap,
    verbose: bool = True,
    *,
    fluid_mask: Optional[np.ndarray] = None,
    ckpt_path: Optional[Path] = None,
    t0_s: float = _TIER1_MIN_T0_S,
) -> RiskMap:
    """One-shot Tier1RiskMap at ``t0_s`` (default 60 s, the first valid
    window). Returns ``fall_back`` if any artefact is missing or the
    forward fails.

    Kept as a thin wrapper around :func:`_load_tier1_artefacts` +
    :func:`_build_tier1_rm` for backward compatibility — S4 still uses
    this to seed its initial planner_rm before its first decoder
    refresh. The S3 ``run()`` loop calls the two halves directly so it
    can refresh the GNN forward every ``replan_period_s`` seconds
    (D-053-fix).
    """
    art = _load_tier1_artefacts(
        fire_scenario_id, ckpt_path=ckpt_path, verbose=verbose,
    )
    if art is None:
        return fall_back
    rm = _build_tier1_rm(
        art, t0_s=t0_s, fluid_mask=fluid_mask, verbose=verbose,
    )
    return rm if rm is not None else fall_back


# ─── Local graph cache ────────────────────────────────────────────────────
_GRAPH_CACHE = None


def _building_graph():
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        from src.path_planning.building_graph import build_graph
        _GRAPH_CACHE = build_graph()
    return _GRAPH_CACHE


# ─── Run loop ─────────────────────────────────────────────────────────────
def run(
    fire_scenario_id: str,
    fds_dir: Path,
    fno_checkpoint: Path | None = None,   # accepted; M5-mini ignores
    dataset_path: Path | None = None,     # accepted; M5-mini ignores
    n_persons: int = 20,
    n_drones: int = 3,
    seed: int = 0,
    t_end_s: float = 300.0,
    dt_s: float = 1.0,
    replan_period_s: float = REPLAN_PERIOD_S,
    t_start_s: float = 0.0,
    *,
    recorder: Optional[SimulationRecorder] = None,
) -> ScenarioMetrics:
    """Execute one S3 run with model-predicted planner RiskMap.

    Args:
        fire_scenario_id: FDS scenario folder name. Also used to look
            up ``results/detector_sequences/<id>.npz`` for the GNN's
            input binary sequence.
        fds_dir: Path to the FDS scenario directory (truth RiskMap).
        fno_checkpoint: PI-FNO weights path. Accepted for forward
            compatibility; M5-mini uses Tier1 instead and ignores this.
        dataset_path: Processed HDF5 path. Accepted for forward
            compatibility; M5-mini ignores.
        n_persons: PersonAgents to spawn. Default 20 (capped at 10
            room nodes).
        n_drones: Drone swarm size. Accepted for forward compat;
            M5-mini does not spawn drone bodies.
        seed: RNG seed for start-position scatter.
        t_end_s: Maximum simulation time.
        dt_s: Outer-loop step.
        replan_period_s: Planner re-plan cadence (D-014: 30 s).

    Returns:
        :class:`ScenarioMetrics` with ``scenario_id="S3_fno_swarm"``.

    Raises:
        FileNotFoundError: If ``assets/placeholder_building.urdf`` is
            missing.
    """
    del fno_checkpoint, dataset_path  # M5-mini: not used.

    urdf = building_urdf_path()
    if not urdf.exists():
        raise FileNotFoundError(
            f"No building URDF on disk. Tried {BUILDING_URDF} (real STL) "
            f"and {PLACEHOLDER_URDF} (fallback). "
            f"Generate one: python -m src.integration.urdf_builder"
        )

    cfg = SceneConfig(
        connection_mode="DIRECT",
        building_urdf=urdf,
        dt_s=dt_s,
        draw_origin_axes=False,
    )
    scene = Scene.create(cfg)
    try:
        fluid_mask = load_default_fluid_mask()
        interior_mask = load_interior_mask()
        truth_rm = load_truth_risk_map(fds_dir, verbose=True)
        truth_co = load_truth_co_field(fds_dir, verbose=True)
        # D-053 (2026-05-15): S3 swapped from tier1_gnn_v3/best.pt to v6.
        # v6 is fine-tuned from v5 (h=48, graph_layers=3, 31K params) with
        # s_029-targeted refinement (s029_iou=1.0, val_iou=0.873).
        #
        # D-053-fix (2026-05-15): load artefacts ONCE, then re-forward the
        # GNN at the current fire clock every replan tick (S4-style).
        # planner_rm is seeded with truth_rm so the swarm has a safe RM
        # to consult before t_fire crosses _TIER1_MIN_T0_S (60 s).
        tier1_ckpt_path = Path("checkpoints/tier1_gnn_v6/best.pt")
        tier1_artefacts = _load_tier1_artefacts(
            fire_scenario_id,
            ckpt_path=tier1_ckpt_path,
            verbose=True,
        )
        planner_rm: RiskMap = truth_rm
        last_tier1_t0: Optional[float] = None
        tier1_n_success = 0
        tier1_n_fail = 0

        if tier1_artefacts is not None:
            # Warm-start: forward at the earliest valid t0 that is >= the
            # caller-supplied fire-clock offset, so the first sim tick
            # already sees a model RM (rather than waiting until t_fire
            # crosses 60 s).
            initial_t0 = max(float(t_start_s), _TIER1_MIN_T0_S)
            if initial_t0 <= _TIER1_MAX_T0_S:
                initial_rm = _build_tier1_rm(
                    tier1_artefacts,
                    t0_s=initial_t0,
                    fluid_mask=fluid_mask,
                    verbose=True,
                )
                if initial_rm is not None:
                    planner_rm = initial_rm
                    last_tier1_t0 = initial_t0
                    tier1_n_success += 1
                else:
                    tier1_n_fail += 1
        exits_xyz = exit_positions()
        agents = spawn_agents(
            scene, n_persons, seed,
            fluid_mask=fluid_mask,
            interior_mask=interior_mask,
        )
        n_actual = len(agents)

        # D-040: pure risk-based weighting (see S2). S3's planner_rm is
        # the model RiskMap, so the planner consults the model only —
        # no truth CO leakage, no proxy.
        # (D-054 FED-mode trial reverted 2026-05-15: see s2_fds_swarm.)
        planner = EvacuationPlanner(
            _building_graph(),
            config=EdgeWeightConfig(
                base_cost=1.0,
                risk_scale=10.0,
                risk_threshold=1.0,
                n_samples=5,
            ),
            heuristic="euclidean",
            fallback_to_last=True,
        )

        # Drone swarm. Identical to S2 EXCEPT the planner_rm is the
        # MODEL RiskMap (Tier1 GNN by default), not FDS truth. The
        # occupant still feels the FDS truth (truth_rm, truth_co).
        swarm = DroneSwarm(
            config=DroneSwarmConfig(
                n_drones=int(n_drones),
                drone=DroneAgentConfig(
                    replan_period_s=float(replan_period_s),
                ),
            ),
        )
        # D-041: drones still spawn at all 3 canonical exits.
        drone_spawn_xyz = drone_spawn_positions()
        spawn_seeds = [
            drone_spawn_xyz[i % len(drone_spawn_xyz)].copy()
            for i in range(int(n_drones))
        ]
        swarm.spawn(spawn_xyzs=spawn_seeds, interior_mask=interior_mask)

        arrived: Dict[str, float] = {}
        died_at: Dict[str, float] = {}
        exposure_s: Dict[str, float] = {a.agent_id: 0.0 for a in agents}

        # D-034: dual timelines (see s1/s2 for full comment).
        duration_s = max(0.0, float(t_end_s) - float(t_start_s))
        max_steps = int(math.ceil(duration_s / dt_s)) + 5
        DANGER_THRESHOLD = 0.5

        for _ in range(max_steps):
            t_sim = float(scene.t)
            if t_sim >= duration_s:
                break
            t_fire = t_sim + float(t_start_s)

            # 0. Tier1 GNN refresh tick (D-053-fix). Slide the 6-frame
            #    history to ``t_fire`` and re-forward the GNN so the
            #    planner sees a fresh 50 s lookahead. Inside the valid
            #    [60, 250] s window only; outside, freeze planner_rm at
            #    the last successful build.
            if (
                tier1_artefacts is not None
                and _TIER1_MIN_T0_S <= t_fire <= _TIER1_MAX_T0_S
                and (
                    last_tier1_t0 is None
                    or (t_fire - last_tier1_t0) >= float(replan_period_s)
                )
            ):
                first_attempt = last_tier1_t0 is None
                new_rm = _build_tier1_rm(
                    tier1_artefacts,
                    t0_s=t_fire,
                    fluid_mask=fluid_mask,
                    verbose=first_attempt,
                )
                if new_rm is not None:
                    planner_rm = new_rm
                    last_tier1_t0 = t_fire
                    tier1_n_success += 1
                else:
                    tier1_n_fail += 1

            # 1. Swarm advances using the MODEL RiskMap (planner_rm).
            #    D-040: risk-only planner — no CO field passed, so the
            #    swarm consults the model RiskMap exclusively (no truth
            #    leakage).
            swarm.update(
                t=t_fire,
                dt_s=float(dt_s),
                persons=agents,
                planner_rm=planner_rm,
                planner=planner,
                interior_mask=interior_mask,
                exits=exits_xyz,
            )

            # 2. Each ALIVE person scans for a GUIDING drone in sight
            #    and walks via self path planning (D-032). D-037 fallback:
            #    when no drone is in sight, behave like S1 (risk-blind
            #    nearest exit) so the occupant never just stands still.
            for agent in agents:
                if agent.status != PersonStatus.ALIVE:
                    continue
                # D-051: priority 0 — if an exit is already within
                # ``exit_proximity_m`` (default 1.0 m = 2 cells), ignore
                # the GUIDING drone and walk straight in. Otherwise fall
                # back to the original drone-follow / nearest-exit chain.
                near_exit_wp = agent.nearest_exit_within_proximity(exits_xyz)
                if near_exit_wp is not None:
                    waypoint = near_exit_wp
                else:
                    target_drone = agent.scan_for_drones(swarm.drones)
                    if target_drone is not None:
                        waypoint = np.array([
                            target_drone.position[0],
                            target_drone.position[1],
                            agent.position[2],
                        ], dtype=np.float64)
                    else:
                        waypoint = min(
                            exits_xyz,
                            key=lambda ex, _p=agent.position: float(
                                np.linalg.norm(np.asarray(ex)[:2] - _p[:2])
                            ),
                        )
                        waypoint = np.array([
                            waypoint[0], waypoint[1], agent.position[2],
                        ], dtype=np.float64)
                tgt_cell = (
                    int(waypoint[0] / CELL_SIZE_M),
                    int(waypoint[1] / CELL_SIZE_M),
                )
                if (
                    not agent.nav_path
                    or agent.last_path_target_cell != tgt_cell
                    or (t_sim - agent.last_path_replan_t)
                    >= agent.config.path_replan_period_s
                ):
                    agent.replan_path_to(waypoint, fluid_mask, t=t_sim)
                agent.step_along_path(scene.client, dt_s, fluid_mask)

                # 3. Truth-sampled danger + CO using the FIRE clock.
                danger = float(truth_rm.query(agent.position, t=t_fire))
                co_ppm = float(truth_co.query(agent.position, t=t_fire))
                if danger > DANGER_THRESHOLD:
                    exposure_s[agent.agent_id] += float(dt_s)

                died = agent.accumulate_exposure(
                    experienced_co_ppm=co_ppm,
                    experienced_danger=danger,
                    dt_s=float(dt_s),
                )
                if died:
                    died_at[agent.agent_id] = t_sim
                    continue

                for ex in exits_xyz:
                    if (
                        float(np.linalg.norm(agent.position[:2] - ex[:2]))
                        <= agent.config.arrival_tolerance_m
                    ):
                        arrived[agent.agent_id] = t_sim
                        agent.mark_evacuated()
                        break

            if recorder is not None:
                recorder.record(
                    t=t_fire,
                    agents=agents,
                    risk_map=truth_rm,
                    arrived=set(arrived.keys()),
                    drones=swarm.drones,
                    agent_extras_fn=lambda a, _es=exposure_s: {
                        "exposure_s": _es.get(a.agent_id, 0.0),
                        "cumulative_fed": float(a.cumulative_fed),
                        "following_drone_id": a.following_drone_id,
                    },
                )
            scene.step()
            if len(arrived) + len(died_at) == n_actual:
                break

        success_rate = (len(arrived) / n_actual) if n_actual else 0.0
        mean_evac_time = (
            float(np.mean(list(arrived.values())))
            if arrived else float("nan")
        )
        mean_exposure = (
            float(np.mean(list(exposure_s.values()))) if exposure_s else 0.0
        )
        casualty_rate = (len(died_at) / n_actual) if n_actual else 0.0
        feds = np.asarray([a.cumulative_fed for a in agents], dtype=np.float64)
        mean_fed = float(feds.mean()) if feds.size else 0.0
        max_fed = float(feds.max()) if feds.size else 0.0
        p90_fed = float(np.percentile(feds, 90)) if feds.size else 0.0

        if tier1_artefacts is not None:
            print(
                f"  [planner_rm] {tier1_n_success} refresh(es) ok, "
                f"{tier1_n_fail} failed; last t0={last_tier1_t0}"
            )

        return ScenarioMetrics(
            scenario_id="S3_fno_swarm",   # canonical name per CLAUDE.md
            fire_scenario_id=fire_scenario_id,
            seed=seed,
            n_persons=n_actual,
            evacuation_success_rate=success_rate,
            mean_evacuation_time_s=mean_evac_time,
            danger_zone_exposure_time_s=mean_exposure,
            casualty_rate=casualty_rate,
            cumulative_fed=mean_fed,
            max_cumulative_fed=max_fed,
            p90_cumulative_fed=p90_fed,
        )
    finally:
        scene.close()


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("s3_fno_swarm.py self-test (M5-mini, Tier1 as planner RM)")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. Zero-risk fallback smoke ────────────────────────────────
    print("\n[1] run() smoke: 5 agents, 60 s, no FDS (fallback to zero risk)")
    res_zero = run(
        fire_scenario_id="m5_mini_smoke",
        fds_dir=Path("data/raw/__no_such_scenario__"),
        n_persons=5,
        seed=0,
        t_end_s=60.0,
        dt_s=1.0,
    )
    print(f"  {res_zero.summary_line()}")
    if res_zero.scenario_id != "S3_fno_swarm":
        errors.append(f"scenario_id wrong: {res_zero.scenario_id}")
    if res_zero.danger_zone_exposure_time_s != 0.0:
        errors.append(
            f"zero-risk exposure should be 0, got {res_zero.danger_zone_exposure_time_s}"
        )

    # ── 2. Real FDS: S1 vs S2 vs S3 three-way comparison ───────────
    real_fds = Path("data/raw/sim_1500kw_2m2_T05")
    if real_fds.exists():
        print(
            "\n[2] S1 vs S2 vs S3 on sim_1500kw_2m2_T05, 5 agents, seed=0, t_end=200 s"
        )
        # S3 first (this module's run)
        res_s3 = run(
            fire_scenario_id="sim_1500kw_2m2_T05",
            fds_dir=real_fds,
            n_persons=5,
            seed=0,
            t_end_s=200.0,
            dt_s=1.0,
            replan_period_s=30.0,
        )
        print(f"  S3: {res_s3.summary_line()}")

        from src.integration.scenarios.s1_fixed_sign import run as s1_run
        from src.integration.scenarios.s2_fds_swarm import run as s2_run

        res_s2 = s2_run(
            fire_scenario_id="sim_1500kw_2m2_T05",
            fds_dir=real_fds,
            n_persons=5,
            seed=0,
            t_end_s=200.0,
            dt_s=1.0,
        )
        print(f"  S2: {res_s2.summary_line()}")

        res_s1 = s1_run(
            fire_scenario_id="sim_1500kw_2m2_T05",
            fds_dir=real_fds,
            n_persons=5,
            seed=0,
            t_end_s=200.0,
            dt_s=1.0,
        )
        print(f"  S1: {res_s1.summary_line()}")

        # H6 / H5 verdict via metrics.h6_verdict
        from src.integration.metrics import h6_verdict
        verdict = h6_verdict([res_s1, res_s2, res_s3])
        print(
            "\n  H6 / H5 numeric summary (mean_fed across rows -- still 0 in "
            "M5-mini, since FED needs M3-full CO loading):"
        )
        for k, v in verdict.items():
            print(f"    {k}: {v}")

        # Surface-level checks
        if res_s3.scenario_id != "S3_fno_swarm":
            errors.append("S3 scenario_id wrong")
        if res_s3.n_persons != res_s1.n_persons:
            errors.append(
                f"S1/S3 n_persons differ: {res_s1.n_persons} vs {res_s3.n_persons}"
            )

        print(
            "\n  diff summary on 5-agent / sim_1500kw_2m2_T05 / seed=0:"
        )
        print(
            f"    evac %     S1 {res_s1.evacuation_success_rate*100:5.1f} | "
            f"S2 {res_s2.evacuation_success_rate*100:5.1f} | "
            f"S3 {res_s3.evacuation_success_rate*100:5.1f}"
        )
        if not (
            math.isnan(res_s1.mean_evacuation_time_s)
            or math.isnan(res_s2.mean_evacuation_time_s)
            or math.isnan(res_s3.mean_evacuation_time_s)
        ):
            print(
                f"    mean t_evac S1 {res_s1.mean_evacuation_time_s:5.2f} | "
                f"S2 {res_s2.mean_evacuation_time_s:5.2f} | "
                f"S3 {res_s3.mean_evacuation_time_s:5.2f}"
            )
        print(
            f"    exposure   S1 {res_s1.danger_zone_exposure_time_s:5.2f} | "
            f"S2 {res_s2.danger_zone_exposure_time_s:5.2f} | "
            f"S3 {res_s3.danger_zone_exposure_time_s:5.2f}"
        )
    else:
        print(f"\n[2] SKIP: real FDS scenario {real_fds} not on disk")

    # ── 3. CSV round-trip ─────────────────────────────────────────
    print("\n[3] ScenarioMetrics.to_dict round-trip")
    d = res_zero.to_dict()
    if d["scenario_id"] != "S3_fno_swarm":
        errors.append("to_dict lost scenario_id")
    print(f"  keys = {sorted(d)}")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: s3_fno_swarm M5-mini validated")
