"""Scenario S4 -- Drone swarm guided by L4h ensemble decoder RiskMap.

Identical to :mod:`src.integration.scenarios.s3_fno_swarm` except the
*planner* consults :class:`EnsembleDecoderRiskMap` -- the cell-level
learned ensemble (Tier1 GNN + Sparse-ConvLSTM + Sparse-FNO + PerCell
decoder, ``checkpoints/ensemble_decoder/best.pt``).

Decoder forward is rebuilt every ``replan_period_s`` seconds inside the
run loop so the planner always sees a freshly recomputed 60 s lookahead
window. If any required checkpoint is missing on disk the planner falls
back to :class:`Tier1RiskMap` (S3 behaviour) and the run continues.

The fire-clock-driven exposure / FED / casualty metrics are identical
to S2/S3 (M2-full + M3-full).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
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
from src.integration.scenarios.s3_fno_swarm import _load_tier1_planner_rm
from src.path_planning.building_graph import (
    load_default_fluid_mask,
    load_interior_mask,
)
from src.path_planning.edge_weights import EdgeWeightConfig
from src.path_planning.planners import EvacuationPlanner
from src.risk_map.risk_map_class import RiskMap
from src.shared.constants import CELL_SIZE_M, REPLAN_PERIOD_S
from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.ensemble_decoder import PerCellEnsembleDecoder
from src.tier1.ensemble_risk_map import EnsembleDecoderRiskMap
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency


_TIER1_CKPT = Path("checkpoints/tier1_gnn_v3/best.pt")
_DECODER_CKPT = Path("checkpoints/ensemble_decoder/best.pt")
_SPARSE_CONV_CKPT = Path("checkpoints/conv_lstm_sparse_v3/best.pt")
_SPARSE_FNO_CKPT = Path("checkpoints/fno_sparse_v3/best.pt")
_DATASET_PATH = Path("data/processed/dataset.h5")
_BUILDING_YAML = Path("configs/building.yaml")
_DETECTOR_SEQ_DIR = Path("results/detector_sequences")
_RAW_ROOT = Path("data/raw")

# Decoder valid t0 range. Tier1FireDataset enumerates 20 (t_start, scen)
# pairs per scenario -> t0 in [60, 250] s (inclusive). Outside the window
# we freeze the planner_rm at the last successful build instead of
# spam-failing every replan tick.
_DECODER_MIN_T0_S = 60.0
_DECODER_MAX_T0_S = 250.0


# ─── Decoder artefact bundle (loaded once per process) ────────────────────
@dataclass(frozen=True)
class _DecoderArtifacts:
    gnn_model: SimpleFireGNN
    adj: torch.Tensor
    sparse_conv_model: torch.nn.Module
    sparse_fno_model: torch.nn.Module
    decoder: PerCellEnsembleDecoder
    mask: np.ndarray
    sensor_indicator: np.ndarray
    knn_idx: np.ndarray
    knn_w: np.ndarray
    device: torch.device
    seq_dir: Path
    raw_root: Path


_ARTIFACTS_CACHE: Dict[Path, "_DecoderArtifacts"] = {}


def _ensure_scripts_on_path() -> None:
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _load_decoder_artifacts(
    decoder_ckpt: Path = _DECODER_CKPT,
    *,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> Optional[_DecoderArtifacts]:
    """Load the 4-model L4h stack + auxiliary tensors. ``None`` on failure."""
    if decoder_ckpt in _ARTIFACTS_CACHE:
        return _ARTIFACTS_CACHE[decoder_ckpt]

    required = {
        "decoder":     decoder_ckpt,
        "Tier1 GNN":   _TIER1_CKPT,
        "Sparse-Conv": _SPARSE_CONV_CKPT,
        "Sparse-FNO":  _SPARSE_FNO_CKPT,
        "dataset.h5":  _DATASET_PATH,
        "building.yaml": _BUILDING_YAML,
    }
    missing = [name for name, p in required.items() if not p.exists()]
    if missing:
        if verbose:
            print(
                f"  [decoder] artefacts missing ({', '.join(missing)}); "
                f"S4 falls back to Tier1RiskMap"
            )
        return None

    try:
        _ensure_scripts_on_path()
        from evaluate_t_locations import load_mask, load_model
        from evaluate_sparse_fno import load_sparse_fno
        from evaluate_ensemble import precompute_node_to_cell_weights
        from train_sparse_conv_lstm import (
            load_sensor_indices, make_sparse_indicator,
        )

        # Tier 1 GNN
        gnn_ckpt = torch.load(
            _TIER1_CKPT, weights_only=False, map_location=device
        )
        cfg = gnn_ckpt.get("config", {})
        gnn_model = SimpleFireGNN(
            in_feat=5,
            hidden=int(cfg.get("hidden", 32)),
            n_graph_layers=int(cfg.get("n_graph_layers", 2)),
            T_out=int(cfg.get("T_out", 6)),
        )
        gnn_model.load_state_dict(gnn_ckpt["model"])
        gnn_model.to(device).eval()
        adj = build_knn_adjacency(k=int(cfg.get("knn_k", 4)))

        # Tier 2 sparse pair
        sparse_conv = load_model(_SPARSE_CONV_CKPT, device, "conv_lstm")
        sparse_fno = load_sparse_fno(_SPARSE_FNO_CKPT, device)

        # PerCell decoder
        dec_ckpt = torch.load(
            decoder_ckpt, weights_only=False, map_location=device
        )
        dcfg = dec_ckpt["config"]
        decoder = PerCellEnsembleDecoder(
            hidden=int(dcfg["hidden"]),
            n_layers=int(dcfg["n_layers"]),
            dropout=float(dcfg.get("dropout", 0.0)),
        )
        decoder.load_state_dict(dec_ckpt["model"])
        decoder.to(device).eval()

        # Auxiliary tensors
        mask = load_mask(_DATASET_PATH)
        sensor_idxs = load_sensor_indices(_BUILDING_YAML)
        sensor_indicator = make_sparse_indicator(sensor_idxs, broadcast_z=True)
        node_positions = [d.position for d in ALL_DETECTORS]
        knn_idx, knn_w = precompute_node_to_cell_weights(
            node_positions, k=3, sigma=5.0, mask=mask, use_geodesic=True,
        )

        artefacts = _DecoderArtifacts(
            gnn_model=gnn_model,
            adj=adj,
            sparse_conv_model=sparse_conv,
            sparse_fno_model=sparse_fno,
            decoder=decoder,
            mask=mask,
            sensor_indicator=sensor_indicator,
            knn_idx=knn_idx,
            knn_w=knn_w,
            device=device,
            seq_dir=_DETECTOR_SEQ_DIR,
            raw_root=_RAW_ROOT,
        )
        _ARTIFACTS_CACHE[decoder_ckpt] = artefacts
        if verbose:
            print(
                f"  [decoder] L4h stack loaded "
                f"(ckpt={decoder_ckpt.parent.name}, device={device})"
            )
        return artefacts
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(
                f"  [decoder] artefact load failed "
                f"({exc.__class__.__name__}: {exc}); fallback to Tier1RiskMap"
            )
        return None


_DECODER_FAIL_SEEN: set = set()


def _build_decoder_rm(
    fire_scenario_id: str,
    t0: float,
    artefacts: _DecoderArtifacts,
    verbose: bool = False,
) -> Optional[EnsembleDecoderRiskMap]:
    """Run the L4h stack at (scenario, t0) and wrap as a RiskMap."""
    try:
        rm = EnsembleDecoderRiskMap.from_scenario(
            scen_name=fire_scenario_id,
            t0=float(t0),
            gnn_model=artefacts.gnn_model,
            adj=artefacts.adj,
            sparse_conv_model=artefacts.sparse_conv_model,
            sparse_fno_model=artefacts.sparse_fno_model,
            decoder=artefacts.decoder,
            mask=artefacts.mask,
            sensor_indicator=artefacts.sensor_indicator,
            knn_idx=artefacts.knn_idx,
            knn_w=artefacts.knn_w,
            seq_dir=artefacts.seq_dir,
            raw_root=artefacts.raw_root,
            device=artefacts.device,
        )
        if verbose:
            print(
                f"  [decoder] refresh t0={t0:.0f}s "
                f"covers [{rm.start_time:.0f}, {rm.t_max:.0f}]"
            )
        return rm
    except Exception as exc:  # noqa: BLE001
        key = (fire_scenario_id, exc.__class__.__name__, str(exc)[:80])
        if key not in _DECODER_FAIL_SEEN:
            _DECODER_FAIL_SEEN.add(key)
            import traceback as _tb
            print(
                f"  [decoder] forward FAILED at t0={t0:.0f}s "
                f"({exc.__class__.__name__}: {exc})"
            )
            tb_lines = _tb.format_exc().splitlines()
            for ln in tb_lines[-6:]:
                print(f"    {ln}")
        return None


# ─── Local graph cache (shared with S3 module) ────────────────────────────
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
    n_persons: int = 20,
    n_drones: int = 3,
    seed: int = 0,
    t_end_s: float = 300.0,
    dt_s: float = 1.0,
    replan_period_s: float = REPLAN_PERIOD_S,
    t_start_s: float = 0.0,
    *,
    recorder: Optional[SimulationRecorder] = None,
    decoder_ckpt: Optional[Path] = None,
) -> ScenarioMetrics:
    """Execute one S4 run with the L4h ensemble decoder as planner RiskMap.

    The decoder is rebuilt every ``replan_period_s`` seconds of fire
    time as long as ``t_fire`` stays inside ``[_DECODER_MIN_T0_S,
    _DECODER_MAX_T0_S]``. Outside that window the planner reuses the
    last successful build.
    """
    decoder_ckpt = decoder_ckpt or _DECODER_CKPT

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

        # D-046-style decoder loop. Decoder artefacts are loaded once and
        # cached at module scope. The "initial" planner_rm is the Tier 1
        # GNN forward — it covers the [60, 110] s window before the first
        # decoder refresh, and is also the fallback when any decoder
        # artefact is missing on disk.
        decoder_artefacts = _load_decoder_artifacts(
            decoder_ckpt=decoder_ckpt, verbose=True,
        )
        planner_rm: RiskMap = _load_tier1_planner_rm(
            fire_scenario_id, fall_back=truth_rm, verbose=True,
            fluid_mask=fluid_mask,
        )
        last_decoder_t0: Optional[float] = None
        decoder_n_success = 0
        decoder_n_fail = 0

        exits_xyz = exit_positions()
        agents = spawn_agents(
            scene, n_persons, seed,
            fluid_mask=fluid_mask,
            interior_mask=interior_mask,
        )
        n_actual = len(agents)

        # Same risk-only planner config as S3 (D-040).
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

        # Drone swarm. Identical to S3 except the planner_rm is the
        # ENSEMBLE DECODER, not the bare Tier1 GNN.
        swarm = DroneSwarm(
            config=DroneSwarmConfig(
                n_drones=int(n_drones),
                drone=DroneAgentConfig(
                    replan_period_s=float(replan_period_s),
                ),
            ),
        )
        drone_spawn_xyz = drone_spawn_positions()
        spawn_seeds = [
            drone_spawn_xyz[i % len(drone_spawn_xyz)].copy()
            for i in range(int(n_drones))
        ]
        swarm.spawn(spawn_xyzs=spawn_seeds, interior_mask=interior_mask)

        arrived: Dict[str, float] = {}
        died_at: Dict[str, float] = {}
        exposure_s: Dict[str, float] = {a.agent_id: 0.0 for a in agents}

        duration_s = max(0.0, float(t_end_s) - float(t_start_s))
        max_steps = int(math.ceil(duration_s / dt_s)) + 5
        DANGER_THRESHOLD = 0.5

        for _ in range(max_steps):
            t_sim = float(scene.t)
            if t_sim >= duration_s:
                break
            t_fire = t_sim + float(t_start_s)

            # 0. Decoder refresh tick. Only inside the dataset's valid
            #    t0 window; outside, freeze planner_rm at the last build.
            if (
                decoder_artefacts is not None
                and _DECODER_MIN_T0_S <= t_fire <= _DECODER_MAX_T0_S
                and (
                    last_decoder_t0 is None
                    or (t_fire - last_decoder_t0) >= float(replan_period_s)
                )
            ):
                first_attempt = last_decoder_t0 is None
                new_rm = _build_decoder_rm(
                    fire_scenario_id, t_fire, decoder_artefacts,
                    verbose=first_attempt,
                )
                if new_rm is not None:
                    planner_rm = new_rm
                    last_decoder_t0 = t_fire
                    decoder_n_success += 1
                else:
                    decoder_n_fail += 1

            # 1. Swarm advances using the MODEL RiskMap (planner_rm).
            swarm.update(
                t=t_fire,
                dt_s=float(dt_s),
                persons=agents,
                planner_rm=planner_rm,
                planner=planner,
                interior_mask=interior_mask,
                exits=exits_xyz,
            )

            # 2. Each ALIVE person follows priority chain (D-051):
            #    exit-proximity → drone in sight → nearest exit fallback.
            for agent in agents:
                if agent.status != PersonStatus.ALIVE:
                    continue
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

        if decoder_artefacts is not None:
            print(
                f"  [decoder] {decoder_n_success} refresh(es) ok, "
                f"{decoder_n_fail} failed; last t0={last_decoder_t0}"
            )

        return ScenarioMetrics(
            scenario_id="S4_ensemble_swarm",
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
    print("=" * 60)
    print("s4_ensemble_swarm.py self-test")
    print("=" * 60)
    res = run(
        fire_scenario_id="s_029",
        fds_dir=Path("data/raw/s_029"),
        n_persons=5,
        seed=0,
        t_end_s=200.0,
        dt_s=1.0,
        replan_period_s=30.0,
        t_start_s=60.0,
    )
    print(f"  {res.summary_line()}")
    print("\nPASS")
