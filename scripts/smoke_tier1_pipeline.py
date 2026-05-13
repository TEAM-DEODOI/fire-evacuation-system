"""End-to-end integration smoke for the H6 logical pipeline.

Pipeline exercised:

    binary detector sequence (T_in=6, 39 sensors, one triggered)
        │   torch.no_grad
        ▼
    SimpleFireGNN.forward(x, adj) → (1, 39, 6)
        │   from_model_output
        ▼
    Tier1RiskMap.query(xyz, t) ∈ [0, 1]
        │
        ▼
    EvacuationPlanner.plan(start_xyz, risk_map, t) → waypoints
        │
        ▼
    simulate_evacuation(start_xyz, planner, risk_map) → EvacuationResult

This is *not* the actual EXP-PATH-001 (which requires PyBullet 20-person
swarm — see D-025). It is a logical sanity check that the GNN headline
checkpoint plugs into the new path-planning module via the
:class:`RiskMap` contract. A green run here is a prerequisite for
exposing :class:`Tier1RiskMap` to Week-12 drone-swarm code.

Run::

    python scripts/smoke_tier1_pipeline.py

PASS / FAIL printed on the last line.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from src.path_planning.building_graph import build_graph
from src.path_planning.edge_weights import EdgeWeightConfig
from src.path_planning.evacuation_sim import simulate_evacuation
from src.path_planning.planners import EvacuationPlanner
from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_gnn import N_NODES, SimpleFireGNN, build_knn_adjacency
from src.tier1.tier1_risk_map import Tier1RiskMap


CKPT_PATH = Path("checkpoints/tier1_gnn_v3/best.pt")
T_IN = 6
F_IN = 5  # in_feat: [is_det, det_time_norm, type_onehot × 3]


def _load_model() -> Tuple[SimpleFireGNN, torch.Tensor, dict]:
    """Load the headline Tier 1 GNN checkpoint + k-NN adjacency."""
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"checkpoint not found at {CKPT_PATH}. "
            f"Run scripts/train_tier1_gnn.py first."
        )
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = SimpleFireGNN(
        in_feat=F_IN,
        hidden=int(cfg.get("hidden", 32)),
        n_graph_layers=int(cfg.get("n_graph_layers", 2)),
        T_out=int(cfg.get("T_out", 6)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    adj = build_knn_adjacency(k=int(cfg.get("knn_k", 4)))
    return model, adj, ckpt


def _synthetic_input(
    triggered_idx: int,
    triggered_at_step: int = 2,
) -> torch.Tensor:
    """Build a ``(1, N, T_in, F)`` tensor with one sensor latched on.

    Feature layout follows ``src.tier1.tier1_gnn.build_node_type_onehot``
    plus the binary trigger channels::

        [is_det, det_time_norm, room, corridor, exit]
    """
    x = torch.zeros(1, N_NODES, T_IN, F_IN)

    # Type one-hot (constant across time).
    types = ["room", "corridor", "exit"]
    for i, d in enumerate(ALL_DETECTORS):
        x[0, i, :, 2 + types.index(d.node_type)] = 1.0

    # Trigger sensor `triggered_idx` from step `triggered_at_step` onward.
    for t in range(triggered_at_step, T_IN):
        x[0, triggered_idx, t, 0] = 1.0          # is_det = 1
        x[0, triggered_idx, t, 1] = (t - triggered_at_step) / max(T_IN - 1, 1)

    return x


def _node_positions_xyz() -> List[Tuple[float, float, float]]:
    """Detector world positions at breathing height z=1.5 m."""
    return [(d.position[0], d.position[1], 1.5) for d in ALL_DETECTORS]


def _detector_near(xyz: np.ndarray) -> int:
    """Pick the detector index closest to ``xyz`` (XY only)."""
    arr = np.asarray(xyz, dtype=np.float64)
    best = -1
    best_d2 = float("inf")
    for i, d in enumerate(ALL_DETECTORS):
        dx = d.position[0] - arr[0]
        dy = d.position[1] - arr[1]
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = i
    return best


def main() -> int:
    print("=" * 60)
    print("smoke_tier1_pipeline - end-to-end H6 logical sanity")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. Load GNN checkpoint ────────────────────────────────────────
    print("\n[1] Load Tier 1 GNN checkpoint")
    model, adj, ckpt = _load_model()
    print(
        f"  epoch={ckpt.get('epoch')} "
        f"val_iou={ckpt.get('val_iou'):.4f} "
        f"params={model.count_parameters()}"
    )

    # ── 2. Build synthetic detector trigger (sensor near central hall) ─
    print("\n[2] Synthetic detector trigger at central courtyard")
    central_xy = np.array([15.0, 10.0, 1.5])
    fire_det_idx = _detector_near(central_xy)
    print(
        f"  fire sensor idx = {fire_det_idx}  "
        f"({ALL_DETECTORS[fire_det_idx].detector_id} @ "
        f"{ALL_DETECTORS[fire_det_idx].position})"
    )
    x = _synthetic_input(fire_det_idx, triggered_at_step=2)

    # ── 3. GNN forward ─────────────────────────────────────────────────
    print("\n[3] GNN forward → Tier1RiskMap")
    with torch.no_grad():
        pred = model(x, adj)
    print(f"  forward output shape: {tuple(pred.shape)}")
    positions = _node_positions_xyz()
    rm = Tier1RiskMap.from_model_output(
        pred, positions, batch_index=0, start_time=0.0, dt=10.0
    )
    # Sanity: predicted danger near the fire sensor should rise over T_out steps.
    fp = ALL_DETECTORS[fire_det_idx].position
    near_fire = np.array([fp[0], fp[1], 1.5])
    d_first = float(rm.query(near_fire, t=0.0))
    d_last = float(rm.query(near_fire, t=rm.t_max))
    print(
        f"  danger near fire sensor: "
        f"t={rm.start_time:.0f}s {d_first:.3f} → "
        f"t={rm.t_max:.0f}s {d_last:.3f}"
    )
    # Soft check — we don't require monotonicity (GNN may be conservative),
    # but at least one of the two should be non-zero for a triggered sensor.
    if max(d_first, d_last) < 1e-3:
        errors.append("danger near triggered sensor is ~0 — GNN/adapter wiring broken")

    # ── 4. Plan from 3 corner starts via Tier1RiskMap ─────────────────
    print("\n[4] Plan from 3 starts using Tier1RiskMap")
    graph = build_graph()
    planner = EvacuationPlanner(
        graph,
        config=EdgeWeightConfig(
            base_cost=1.0, risk_scale=10.0, risk_threshold=0.95, n_samples=5
        ),
    )
    starts = {
        "zone_b_west":  np.array([4.0, 4.0, 1.5]),
        "zone_a_west":  np.array([3.0, 13.0, 1.5]),
        "zone_d_center": np.array([28.0, 5.0, 1.5]),
    }
    for label, s in starts.items():
        path = planner.plan(s, rm, t=20.0)
        if not path:
            errors.append(f"planner returned empty path from {label}")
            continue
        end_xy = path[-1][:2]
        print(
            f"  {label}  {len(path)} waypoints  "
            f"end ({end_xy[0]:.1f}, {end_xy[1]:.1f})"
        )

    # ── 5. Logical evacuation simulation from one start ────────────────
    print("\n[5] simulate_evacuation with Tier1RiskMap as the experienced map")
    result = simulate_evacuation(
        start_xyz=starts["zone_d_center"],
        planner=planner,
        risk_map=rm,
        replan_interval=30.0,
        t_end=180.0,
    )
    print(f"  {result.summary_line()}")
    if not result.success:
        errors.append(
            f"sim from zone_d_center failed (exit_time={result.exit_time})"
        )

    # ── 6. Confirm RiskMap contract compliance (boundary + time) ───────
    print("\n[6] RiskMap contract spot-checks")
    oob = float(rm.query(np.array([-5.0, 10.0, 1.5]), t=0.0))
    if oob != 1.0:
        errors.append(f"OOB query did not return 1.0: {oob}")
    over_t = float(rm.query(np.array([15.0, 10.0, 1.5]), t=999.0))
    if over_t != 1.0:
        errors.append(f"t > t_max did not return 1.0: {over_t}")
    print(f"  PASS: OOB={oob}, t>t_max={over_t}")

    # ── Verdict ────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nPASS: Tier1RiskMap fully integrated with path_planning pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
