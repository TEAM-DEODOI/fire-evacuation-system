"""Logical H6 mini-experiment: GNN-aware planning vs no-risk baseline.

This is a *pre-PyBullet* sanity check for the new (D-025) H6 hypothesis:
does drone-swarm-style active guidance reduce occupant risk exposure
relative to a baseline that has no fire awareness?

It is NOT the formal EXP-PATH-001 (which requires the multi-agent
PyBullet PersonAgent/drone-swarm pipeline in ``src/integration/``).
This script uses a logical NumPy-only single-occupant simulation so the
trend can be checked cheaply *before* committing to the Week-12
PyBullet build.

Setup:
* 3 OOD FDS scenarios (one each at 1500, 1000, 500 kW).
* 6 start points across the L-shape building.
* 2 strategies:
    - **S1 baseline** — planner sees ``zero_rm`` (every cell danger 0),
      so weighted A* reduces to a length-only shortest path. Models a
      person evacuating without any active fire information.
    - **S2 active**  — planner sees ``tier1_rm``, the Tier 1 GNN's
      per-node danger forecast built from the scenario's binary
      detector sequence. Replans every 30 s. Models the drone-swarm
      guidance in S2/S3 of the new EXP-PATH-001.

Fairness: regardless of what the *planner* sees, the occupant's
**experienced danger** comes from ``truth_rm`` — a Tier1RiskMap built
on the FDS-derived ``node_danger`` array stored in
``results/detector_sequences/<scenario>.npz``. This mirrors
``docs/interface_contracts.md`` §6 "risk_map_truth".

Output:
* ``results/logical_h6_mini/comparison.csv`` (36 trials).
* Console summary table of per-strategy means and the H6 logical proxy
  (peak_danger reduction ratio).

Caveats / known limitations of this proxy:
* Single occupant per trial; multi-person crowd effects ignored.
* Per-node truth (39 detector positions), not full SLCF grid truth.
* FED is not computed (would require raw CO grid; out of mini scope).
* GNN is forward-passed once with the first ``T_in=6`` frames of the
  binary sequence and the prediction is held constant through replans.
  A sliding-window refresh is the natural follow-up.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.path_planning.building_graph import build_graph
from src.path_planning.edge_weights import EdgeWeightConfig
from src.path_planning.evacuation_sim import simulate_evacuation
from src.path_planning.planners import EvacuationPlanner
from src.risk_map.risk_map_class import StaticRiskMap
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS
from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_gnn import N_NODES, SimpleFireGNN, build_knn_adjacency
from src.tier1.tier1_risk_map import Tier1RiskMap


# ─── Config ───────────────────────────────────────────────────────────────
CKPT_PATH = Path("checkpoints/tier1_gnn_v3/best.pt")
SEQ_DIR = Path("results/detector_sequences")
OUT_DIR = Path("results/logical_h6_mini")
OUT_CSV = OUT_DIR / "comparison.csv"

SCENARIOS: Tuple[str, ...] = (
    "sim_1500kw_2m2_T05",   # large fire — should make the proxy easy
    "sim_1000kw_1m2_T03",   # mid HRR small fire — typical
    "sim_500kw_1m2_T01",    # weak fire — hardest to detect / GNN may struggle
)

# Six start points: one per building zone + 2 in the central hall.
STARTS: Dict[str, np.ndarray] = {
    "zone_a_west":    np.array([3.0, 13.0, 1.5]),
    "zone_b_west":    np.array([4.0,  4.0, 1.5]),
    "zone_c_east":   np.array([28.0, 16.0, 1.5]),
    "zone_d_center": np.array([28.0,  5.0, 1.5]),
    "hall_n":        np.array([15.0, 12.0, 1.5]),
    "hall_w":        np.array([10.0,  9.0, 1.5]),
}

# Edge weight config — risk_threshold=0.95 keeps even mid-fire edges
# passable (otherwise no path exists in the small building).
EDGE_CFG = EdgeWeightConfig(
    base_cost=1.0, risk_scale=10.0, risk_threshold=0.95, n_samples=5
)


# ─── Model + adjacency loaded once ────────────────────────────────────────
def _load_gnn() -> Tuple[SimpleFireGNN, torch.Tensor]:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Tier 1 GNN checkpoint not found at {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = SimpleFireGNN(
        in_feat=5,
        hidden=int(cfg.get("hidden", 32)),
        n_graph_layers=int(cfg.get("n_graph_layers", 2)),
        T_out=int(cfg.get("T_out", 6)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    adj = build_knn_adjacency(k=int(cfg.get("knn_k", 4)))
    return model, adj


def _node_positions_xyz() -> List[Tuple[float, float, float]]:
    return [(d.position[0], d.position[1], 1.5) for d in ALL_DETECTORS]


def _binary_to_features(
    binary_history: np.ndarray,
) -> torch.Tensor:
    """Build the 5-channel GNN input from a ``(T_in, N)`` binary slice.

    Features per node per step::
        [is_det, det_time_norm, room_onehot, corridor_onehot, exit_onehot]
    """
    T_in, n = binary_history.shape
    assert n == N_NODES, f"expected {N_NODES} sensors, got {n}"
    x = torch.zeros(1, n, T_in, 5)

    types = ["room", "corridor", "exit"]
    for i, d in enumerate(ALL_DETECTORS):
        x[0, i, :, 2 + types.index(d.node_type)] = 1.0

    # det_time_norm: time since first activation for this sensor, normalised.
    first_active = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        nz = np.nonzero(binary_history[:, i])[0]
        if len(nz):
            first_active[i] = int(nz[0])

    for t in range(T_in):
        for i in range(n):
            if binary_history[t, i] > 0:
                x[0, i, t, 0] = 1.0
                if first_active[i] >= 0:
                    x[0, i, t, 1] = (t - first_active[i]) / max(T_in - 1, 1)
    return x


def _build_truth_rm(node_danger: np.ndarray) -> Tier1RiskMap:
    """Truth = FDS-derived per-node danger across all 31 frames."""
    return Tier1RiskMap(
        node_risks=node_danger.astype(np.float32),
        node_positions=_node_positions_xyz(),
        start_time=0.0,
        dt=float(DT_SLCF),
    )


def _build_tier1_rm(
    model: SimpleFireGNN,
    adj: torch.Tensor,
    binary_sequence: np.ndarray,
    t0_idx: int = 0,
    t_in: int = 6,
) -> Tier1RiskMap:
    """Forward the GNN on a window of the binary sequence."""
    history = binary_sequence[t0_idx : t0_idx + t_in]
    if history.shape[0] < t_in:
        # Pad with zeros at the front if the requested window precedes t=0.
        pad = np.zeros((t_in - history.shape[0], N_NODES), dtype=np.float32)
        history = np.concatenate([pad, history], axis=0)
    x = _binary_to_features(history)
    with torch.no_grad():
        pred = model(x, adj)  # (1, N, T_out)
    return Tier1RiskMap.from_model_output(
        pred,
        _node_positions_xyz(),
        batch_index=0,
        start_time=float(t0_idx * DT_SLCF + t_in * DT_SLCF),
        dt=float(DT_SLCF),
    )


def _zero_rm() -> StaticRiskMap:
    """All-safe risk map — planner sees nothing and falls back to length-only A*."""
    nx, ny, nz = GRID_SHAPE
    times = np.arange(0.0, N_TIMESTEPS * DT_SLCF, DT_SLCF)
    return StaticRiskMap(
        danger_array=np.zeros((N_TIMESTEPS, nx, ny, nz), dtype=np.float32),
        times=times,
    )


# ─── Trial dataclass ──────────────────────────────────────────────────────
@dataclass
class TrialRow:
    scenario: str
    start: str
    strategy: str
    arrived: bool
    exit_time: float
    n_replans: int
    peak_danger_truth: float
    time_in_hazard_truth: float
    travel_time_s: float


# ─── One trial ────────────────────────────────────────────────────────────
def _run_trial(
    scenario: str,
    start_name: str,
    start_xyz: np.ndarray,
    strategy: str,
    planner: EvacuationPlanner,
    planner_rm,
    truth_rm: Tier1RiskMap,
) -> TrialRow:
    result = simulate_evacuation(
        start_xyz=start_xyz,
        planner=planner,
        risk_map=planner_rm,
        replan_interval=30.0,
        t_end=200.0,
    )
    traj = result.trajectory_arr()
    times = result.times_arr()

    if len(traj) == 0:
        return TrialRow(
            scenario=scenario,
            start=start_name,
            strategy=strategy,
            arrived=False,
            exit_time=float("inf"),
            n_replans=result.n_replans,
            peak_danger_truth=0.0,
            time_in_hazard_truth=0.0,
            travel_time_s=0.0,
        )

    # Evaluate experienced danger under the FDS-truth map.
    truth_dangers = np.array(
        [float(truth_rm.query(p, t=float(t))) for p, t in zip(traj, times)]
    )
    peak_truth = float(truth_dangers.max())
    # Time in hazardous zone: sum dt where danger at start of step > 0.5.
    if len(times) >= 2:
        dts = np.diff(times)
        in_haz = (truth_dangers[:-1] > 0.5).astype(np.float64)
        time_in_hazard = float((in_haz * dts).sum())
    else:
        time_in_hazard = 0.0

    travel_time = float(times[-1] - times[0]) if len(times) >= 2 else 0.0

    return TrialRow(
        scenario=scenario,
        start=start_name,
        strategy=strategy,
        arrived=result.success,
        exit_time=result.exit_time,
        n_replans=result.n_replans,
        peak_danger_truth=peak_truth,
        time_in_hazard_truth=time_in_hazard,
        travel_time_s=travel_time,
    )


# ─── Entry ────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 60)
    print("logical_h6_mini  -  GNN-aware planning vs no-risk baseline")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Load Tier 1 GNN")
    model, adj = _load_gnn()
    print(f"  params={model.count_parameters()}  k_nn={adj.shape}")

    graph = build_graph()
    planner = EvacuationPlanner(graph, config=EDGE_CFG)
    zero_rm = _zero_rm()

    rows: List[TrialRow] = []

    for scenario in SCENARIOS:
        seq_path = SEQ_DIR / f"{scenario}.npz"
        if not seq_path.exists():
            print(f"\n[skip] {scenario}: {seq_path} not found")
            continue
        print(f"\n[2] Scenario {scenario}")
        data = np.load(seq_path)
        binary_seq = data["binary_sequence"]      # (31, 39)
        node_danger = data["node_danger"]          # (31, 39)
        print(
            f"  binary_seq shape={binary_seq.shape} "
            f"sum(triggered)={int(binary_seq[-1].sum())}/39 by t=300s"
        )
        print(
            f"  node_danger max={node_danger.max():.3f}  "
            f"mean@t=300={node_danger[-1].mean():.3f}"
        )

        truth_rm = _build_truth_rm(node_danger)
        tier1_rm = _build_tier1_rm(model, adj, binary_seq, t0_idx=0, t_in=6)

        for start_name, start_xyz in STARTS.items():
            for strategy, planner_rm in [
                ("S1_baseline", zero_rm),
                ("S2_tier1_active", tier1_rm),
            ]:
                row = _run_trial(
                    scenario, start_name, start_xyz, strategy,
                    planner, planner_rm, truth_rm,
                )
                rows.append(row)
                print(
                    f"    {strategy:>18}  {start_name:>14}  "
                    f"arr={'OK' if row.arrived else 'FAIL'}  "
                    f"t={row.exit_time:5.1f}s  "
                    f"peak={row.peak_danger_truth:.3f}  "
                    f"hazard={row.time_in_hazard_truth:.1f}s"
                )

    # ── CSV ──────────────────────────────────────────────────────────
    print(f"\n[3] Writing CSV: {OUT_CSV}")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario", "start", "strategy",
                "arrived", "exit_time_s", "n_replans",
                "peak_danger_truth", "time_in_hazard_truth_s",
                "travel_time_s",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.scenario, r.start, r.strategy,
                    r.arrived, f"{r.exit_time:.2f}", r.n_replans,
                    f"{r.peak_danger_truth:.4f}",
                    f"{r.time_in_hazard_truth:.2f}",
                    f"{r.travel_time_s:.2f}",
                ]
            )
    print(f"  wrote {len(rows)} rows")

    # ── Aggregate per strategy ──────────────────────────────────────
    print("\n[4] Aggregate summary (mean over all scenarios x starts)")
    by_strat: Dict[str, Dict[str, float]] = {}
    for strat in ("S1_baseline", "S2_tier1_active"):
        sub = [r for r in rows if r.strategy == strat]
        n = len(sub) or 1
        by_strat[strat] = {
            "n": float(len(sub)),
            "arrived_rate": sum(r.arrived for r in sub) / n,
            "mean_peak_truth": sum(r.peak_danger_truth for r in sub) / n,
            "mean_hazard_truth": sum(r.time_in_hazard_truth for r in sub) / n,
            "mean_travel_time": sum(r.travel_time_s for r in sub) / n,
        }
    print(
        f"  {'strategy':<20} {'n':>4} {'arrived%':>10} "
        f"{'peak_truth':>12} {'hazard_s':>10} {'travel_s':>10}"
    )
    for strat, m in by_strat.items():
        print(
            f"  {strat:<20} {int(m['n']):>4} {m['arrived_rate']*100:>9.1f}% "
            f"{m['mean_peak_truth']:>12.4f} "
            f"{m['mean_hazard_truth']:>10.2f} "
            f"{m['mean_travel_time']:>10.2f}"
        )

    if "S1_baseline" in by_strat and "S2_tier1_active" in by_strat:
        s1 = by_strat["S1_baseline"]
        s2 = by_strat["S2_tier1_active"]
        if s1["mean_peak_truth"] > 1e-6:
            ratio_peak = s2["mean_peak_truth"] / s1["mean_peak_truth"]
            print(
                f"\n  H6 logical proxy (mean peak_danger): "
                f"S2 / S1 = {ratio_peak:.3f}  "
                f"({'reduction' if ratio_peak < 1 else 'INCREASE'})"
            )
        if s1["mean_hazard_truth"] > 1e-6:
            ratio_haz = s2["mean_hazard_truth"] / s1["mean_hazard_truth"]
            print(
                f"  H6 logical proxy (mean time_in_hazard): "
                f"S2 / S1 = {ratio_haz:.3f}"
            )
        print(
            "\n  Note: this is a *logical proxy* of H6; the real "
            "EXP-PATH-001 (PyBullet, 20 persons, drone swarm) is the "
            "definitive test per D-025."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
