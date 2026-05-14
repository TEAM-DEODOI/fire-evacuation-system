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
  ``results/detector_sequences/<fire_scenario_id>.npz`` (the first
  ``T_in=6`` binary detection frames -> a 6-step prediction starting
  at t=60 s).
* Replan + truth-exposure logic is identical to S2.
* No casualty / FED metrics (M2-full / M3-full prerequisites).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

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
from src.tier1.tier1_gnn import N_NODES, SimpleFireGNN, build_knn_adjacency
from src.tier1.tier1_risk_map import Tier1RiskMap


_TIER1_CKPT = Path("checkpoints/tier1_gnn_v3/best.pt")
_DETECTOR_SEQ_DIR = Path("results/detector_sequences")
_T_IN = 6
_F_IN = 5  # [is_det, det_time_norm, room, corridor, exit]


# ─── Tier 1 planner RiskMap construction ──────────────────────────────────
def _binary_history_to_features(binary_history: np.ndarray) -> torch.Tensor:
    """Convert a ``(T_in, N)`` binary detection slice to the GNN's
    ``(1, N, T_in, F)`` input feature tensor.

    Feature layout (matches the training-time formatter in
    ``scripts/train_tier1_gnn.py`` and the smoke in
    ``scripts/smoke_tier1_pipeline.py``)::

        [is_det, det_time_norm, room_onehot, corridor_onehot, exit_onehot]
    """
    t_in, n = binary_history.shape
    assert n == N_NODES, f"expected {N_NODES} sensors, got {n}"
    x = torch.zeros(1, n, t_in, _F_IN)

    types = ["room", "corridor", "exit"]
    for i, d in enumerate(ALL_DETECTORS):
        x[0, i, :, 2 + types.index(d.node_type)] = 1.0

    first_active = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        nz = np.nonzero(binary_history[:, i])[0]
        if len(nz):
            first_active[i] = int(nz[0])

    for t in range(t_in):
        for i in range(n):
            if binary_history[t, i] > 0:
                x[0, i, t, 0] = 1.0
                if first_active[i] >= 0:
                    x[0, i, t, 1] = (t - first_active[i]) / max(t_in - 1, 1)
    return x


def _load_tier1_planner_rm(
    fire_scenario_id: str,
    fall_back: RiskMap,
    verbose: bool = True,
) -> RiskMap:
    """Build a :class:`Tier1RiskMap` from the GNN + detector sequence.

    Falls back to ``fall_back`` if any of the required artefacts are
    missing or the forward pass fails. This is exactly the
    "FDS-truth as proxy" S2-equivalent behaviour and produces a clear
    warning so the caller can tell apart "S3-as-intended" from
    "S3-degraded-to-S2".
    """
    if not _TIER1_CKPT.exists():
        if verbose:
            print(
                f"  [planner_rm] Tier1 checkpoint missing ({_TIER1_CKPT}); "
                f"S3 planner reuses truth RiskMap"
            )
        return fall_back

    seq_path = _DETECTOR_SEQ_DIR / f"{fire_scenario_id}.npz"
    if not seq_path.exists():
        if verbose:
            print(
                f"  [planner_rm] detector sequence missing ({seq_path}); "
                f"S3 planner reuses truth RiskMap"
            )
        return fall_back

    try:
        ckpt = torch.load(_TIER1_CKPT, weights_only=False, map_location="cpu")
        cfg = ckpt.get("config", {})
        model = SimpleFireGNN(
            in_feat=_F_IN,
            hidden=int(cfg.get("hidden", 32)),
            n_graph_layers=int(cfg.get("n_graph_layers", 2)),
            T_out=int(cfg.get("T_out", 6)),
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        adj = build_knn_adjacency(k=int(cfg.get("knn_k", 4)))

        data = np.load(seq_path)
        binary_seq = data["binary_sequence"]      # (31, 39)
        history = binary_seq[:_T_IN]              # first T_in frames (t = 0..50 s)
        if history.shape[0] < _T_IN:
            pad = np.zeros(
                (_T_IN - history.shape[0], N_NODES), dtype=np.float32
            )
            history = np.concatenate([pad, history], axis=0)

        x = _binary_history_to_features(history)
        with torch.no_grad():
            pred = model(x, adj)  # (1, N, T_out)

        positions = [
            (d.position[0], d.position[1], 1.5) for d in ALL_DETECTORS
        ]
        # The first T_in frames cover t=[0, (T_in-1)*dt]; the GNN's
        # prediction starts immediately after, at t = T_in * dt.
        start_time = float(_T_IN * DT_SLCF)
        rm = Tier1RiskMap.from_model_output(
            pred,
            positions,
            batch_index=0,
            start_time=start_time,
            dt=float(DT_SLCF),
        )
        if verbose:
            print(
                f"  [planner_rm] Tier1 GNN forward complete: "
                f"start_time={start_time:.0f}s  t_max={rm.t_max:.0f}s  "
                f"(history t=0..{(_T_IN - 1) * DT_SLCF:.0f}s)"
            )
        return rm
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(
                f"  [planner_rm] Tier1 build failed "
                f"({exc.__class__.__name__}: {exc}); "
                f"S3 planner reuses truth RiskMap"
            )
        return fall_back


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
        planner_rm = _load_tier1_planner_rm(
            fire_scenario_id, fall_back=truth_rm, verbose=True
        )
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
