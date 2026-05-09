"""
Tier 1 risk map: GNN node-level predictions exposed via the RiskMap contract.

Implements ``docs/tier1_gnn_design.md`` §6. The ``Tier1FireGNN`` outputs a
``(T_out, N_nodes)`` danger array; this module wraps that array (plus the
node world coordinates) into the abstract :class:`RiskMap` interface so the
A* path planner can swap between :class:`FDSRiskMap`, :class:`FNORiskMap`,
and :class:`Tier1RiskMap` with no code changes.

The design doc uses **nearest-node** lookup (not interpolation): the GNN
predicts at coarse zone level (16-20 nodes), so smoothly interpolating
between nodes would invent precision the model does not have. A drone at
position ``xyz`` is assigned the danger of the closest node in the XY
plane (Z is ignored — single floor).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from src.risk_map.risk_map_class import RiskMap
from src.shared.constants import DOMAIN_SIZE_M, DT_SECONDS


class Tier1RiskMap(RiskMap):
    """RiskMap backed by per-node GNN predictions.

    Args:
        node_risks: Predicted danger ∈ [0, 1], shape ``(T_out, N_nodes)``.
            Row ``t`` is the model's prediction for time ``start_time + t * dt``.
        node_positions: ``[(x, y, z), ...]`` world metres, length ``N_nodes``.
        start_time: Time (s) corresponding to row 0 of ``node_risks``.
            Defaults to 0.0.
        dt: Step size (s) between rows of ``node_risks``. Defaults to
            :data:`src.shared.constants.DT_SECONDS` (10.0).

    Raises:
        ValueError: If ``node_risks`` is not 2-D, ``node_positions`` length
            disagrees with ``node_risks.shape[1]``, or ``dt <= 0``.
    """

    def __init__(
        self,
        node_risks: np.ndarray,
        node_positions: Sequence[Tuple[float, float, float]],
        start_time: float = 0.0,
        dt: float = DT_SECONDS,
    ) -> None:
        node_risks = np.asarray(node_risks, dtype=np.float32)
        node_positions_arr = np.asarray(node_positions, dtype=np.float32)

        if node_risks.ndim != 2:
            raise ValueError(
                f"node_risks must be 2-D (T_out, N_nodes), got shape {node_risks.shape}"
            )
        if node_positions_arr.ndim != 2 or node_positions_arr.shape[1] != 3:
            raise ValueError(
                f"node_positions must be (N_nodes, 3), got shape {node_positions_arr.shape}"
            )
        n_nodes_risks = node_risks.shape[1]
        n_nodes_pos = node_positions_arr.shape[0]
        if n_nodes_risks != n_nodes_pos:
            raise ValueError(
                f"node count mismatch: node_risks has {n_nodes_risks}, "
                f"node_positions has {n_nodes_pos}"
            )
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.node_risks: np.ndarray = node_risks
        self.node_positions: np.ndarray = node_positions_arr
        self.start_time: float = float(start_time)
        self.dt: float = float(dt)
        self.t_out: int = node_risks.shape[0]
        self.t_max: float = self.start_time + (self.t_out - 1) * self.dt

    # ─── RiskMap.query ──────────────────────────────────────────────────────
    def query(
        self,
        xyz: np.ndarray,
        t: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Return danger ∈ [0, 1] at world coordinate(s) ``xyz``.

        Args:
            xyz: ``(3,)`` for a single point → returns ``float``.
                 ``(M, 3)`` for batched points → returns ``(M,)`` array.
                 Coordinates in world metres.
            t:   Query time (s). If ``None``, uses ``t_max``.

        Returns:
            Danger ∈ [0, 1]. Out-of-bounds points and times beyond ``t_max``
            return 1.0 (safety default per ``interface_contracts.md`` §2).
            Times before ``start_time`` are clamped to row 0.
        """
        xyz_arr = np.asarray(xyz, dtype=np.float32)

        # Out-of-time-range → max danger across all queried points.
        # (Below start_time we clamp; above t_max we treat as unknown future.)
        if t is None:
            t = self.t_max
        if t > self.t_max:
            if xyz_arr.ndim == 1:
                return 1.0
            return np.ones(xyz_arr.shape[0], dtype=np.float32)

        t_clamped = max(self.start_time, t)
        t_idx = int(round((t_clamped - self.start_time) / self.dt))
        t_idx = max(0, min(self.t_out - 1, t_idx))
        risks_at_t = self.node_risks[t_idx]  # (N_nodes,)

        if xyz_arr.ndim == 1:
            if xyz_arr.shape[0] != 3:
                raise ValueError(
                    f"single-point xyz must have 3 elements, got shape {xyz_arr.shape}"
                )
            return self._query_single(xyz_arr, risks_at_t)

        if xyz_arr.ndim == 2 and xyz_arr.shape[1] == 3:
            return self._query_batch(xyz_arr, risks_at_t)

        raise ValueError(
            f"xyz must have shape (3,) or (M, 3), got {xyz_arr.shape}"
        )

    # ─── Internals ──────────────────────────────────────────────────────────
    def _query_single(self, xyz: np.ndarray, risks: np.ndarray) -> float:
        """Single-point lookup: nearest XY node, with bounds check."""
        if not self._in_bounds(xyz):
            return 1.0
        nearest = int(
            np.argmin(np.linalg.norm(self.node_positions[:, :2] - xyz[:2], axis=1))
        )
        return float(risks[nearest])

    def _query_batch(self, xyz: np.ndarray, risks: np.ndarray) -> np.ndarray:
        """Batched lookup: ``xyz`` shape ``(M, 3)`` → ``(M,)`` danger array."""
        m = xyz.shape[0]
        out = np.ones(m, dtype=np.float32)  # default = max danger

        # In-bounds mask
        lx, ly, lz = DOMAIN_SIZE_M
        in_bounds = (
            (xyz[:, 0] >= 0) & (xyz[:, 0] <= lx)
            & (xyz[:, 1] >= 0) & (xyz[:, 1] <= ly)
            & (xyz[:, 2] >= 0) & (xyz[:, 2] <= lz)
        )
        if not in_bounds.any():
            return out

        valid_xyz = xyz[in_bounds]  # (K, 3)
        # Pairwise XY distances: (K, N_nodes)
        diff = valid_xyz[:, None, :2] - self.node_positions[None, :, :2]
        dists = np.linalg.norm(diff, axis=-1)
        nearest = np.argmin(dists, axis=1)  # (K,)
        out[in_bounds] = risks[nearest]
        return out

    def _in_bounds(self, xyz: np.ndarray) -> bool:
        """Return True if ``xyz`` lies inside the SLCF domain."""
        lx, ly, lz = DOMAIN_SIZE_M
        return bool(
            0.0 <= xyz[0] <= lx
            and 0.0 <= xyz[1] <= ly
            and 0.0 <= xyz[2] <= lz
        )

    # ─── Convenience constructor from a Tier1FireGNN output tensor ──────────
    @classmethod
    def from_model_output(
        cls,
        model_output: "object",
        node_positions: Sequence[Tuple[float, float, float]],
        batch_index: int = 0,
        start_time: float = 0.0,
        dt: float = DT_SECONDS,
    ) -> "Tier1RiskMap":
        """Build a :class:`Tier1RiskMap` directly from a GNN forward-pass tensor.

        Args:
            model_output: Tensor of shape ``(B, N, T_out)`` returned by
                :class:`src.tier1.tier1_model.Tier1FireGNN`. Imported lazily
                so this module doesn't depend on torch at import time.
            node_positions: ``[(x, y, z), ...]`` matching the GNN's node
                indexing.
            batch_index: Which sample of the batch to extract. Defaults to 0
                (typical inference case is batch size 1).
            start_time: Wall-clock simulation time at the first prediction
                step. Defaults to 0.0.
            dt: Step size (s). Defaults to :data:`DT_SECONDS`.

        Returns:
            :class:`Tier1RiskMap` populated with the chosen sample.
        """
        # Lazy import so callers that only consume pre-computed numpy arrays
        # don't need torch on the import path.
        import torch

        if not isinstance(model_output, torch.Tensor):
            raise TypeError(
                f"model_output must be a torch.Tensor, got {type(model_output).__name__}"
            )
        if model_output.dim() != 3:
            raise ValueError(
                f"model_output must be 3-D (B, N, T_out), got shape {tuple(model_output.shape)}"
            )

        # (N, T_out) for the chosen sample → transpose to (T_out, N).
        risks_n_t = model_output[batch_index].detach().cpu().numpy()
        risks_t_n = risks_n_t.T.astype(np.float32, copy=False)
        return cls(
            node_risks=risks_t_n,
            node_positions=node_positions,
            start_time=start_time,
            dt=dt,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run with ``python -m src.tier1.tier1_risk_map``).
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Tier1RiskMap self-test")
    print("=" * 60)

    errors: List[str] = []

    # 4-node toy graph at corners of a 20×10 region (Z=1.5 m breathing zone).
    nodes = [
        (5.0, 2.5, 1.5),    # node 0: bottom-left
        (25.0, 2.5, 1.5),   # node 1: bottom-right
        (5.0, 17.5, 1.5),   # node 2: top-left
        (25.0, 17.5, 1.5),  # node 3: top-right
    ]
    # T_out=6 frames. Risk pattern: node 0 dangerous from t=0, others ramp up.
    node_risks = np.array(
        [
            [0.9, 0.1, 0.1, 0.0],
            [0.9, 0.3, 0.2, 0.0],
            [0.9, 0.5, 0.4, 0.1],
            [0.9, 0.7, 0.6, 0.2],
            [0.9, 0.8, 0.7, 0.4],
            [0.9, 0.9, 0.8, 0.6],
        ],
        dtype=np.float32,
    )

    rm = Tier1RiskMap(node_risks, nodes, start_time=0.0, dt=10.0)
    print(f"  T_out={rm.t_out}, N_nodes={len(nodes)}, "
          f"t_max={rm.t_max}s, dt={rm.dt}s")

    # ── Test 1: query exactly at a known node ───────────────────────────────
    print("\n[Test 1] Query at node 0 position")
    d = rm.query(np.array([5.0, 2.5, 1.5]), t=0.0)
    print(f"  query((5, 2.5, 1.5), t=0.0) = {d}  (expected 0.9)")
    if abs(d - 0.9) > 1e-6:
        errors.append(f"node 0 lookup wrong: {d}")

    # ── Test 2: query near node 3 ───────────────────────────────────────────
    print("\n[Test 2] Query near node 3 at t=50s (t_idx=5)")
    d = rm.query(np.array([24.0, 17.0, 1.5]), t=50.0)
    print(f"  query((24, 17, 1.5), t=50) = {d}  (expected 0.6 from row 5)")
    if abs(d - 0.6) > 1e-6:
        errors.append(f"node 3 nearest lookup at t=50 wrong: {d}")

    # ── Test 3: batched query ───────────────────────────────────────────────
    print("\n[Test 3] Batched query (4 points, 1 each near each node)")
    pts = np.array([
        [5.0, 2.5, 1.5],
        [25.0, 2.5, 1.5],
        [5.0, 17.5, 1.5],
        [25.0, 17.5, 1.5],
    ])
    d_batch = rm.query(pts, t=20.0)  # row 2: [0.9, 0.5, 0.4, 0.1]
    print(f"  result = {d_batch}")
    expected = np.array([0.9, 0.5, 0.4, 0.1], dtype=np.float32)
    if not np.allclose(d_batch, expected, atol=1e-6):
        errors.append(f"batch query wrong: got {d_batch}, expected {expected}")

    # ── Test 4: out-of-bounds → 1.0 ─────────────────────────────────────────
    print("\n[Test 4] Out-of-bounds queries return 1.0")
    cases = [
        np.array([-1.0, 10.0, 1.5]),    # x < 0
        np.array([35.0, 10.0, 1.5]),    # x > 30
        np.array([15.0, -1.0, 1.5]),    # y < 0
        np.array([15.0, 25.0, 1.5]),    # y > 20
        np.array([15.0, 10.0, 4.0]),    # z > 3
    ]
    for p in cases:
        d = rm.query(p, t=0.0)
        if d != 1.0:
            errors.append(f"OOB at {p.tolist()} returned {d} != 1.0")
    print("  PASS" if not errors else "  FAIL")

    # ── Test 5: out-of-time → 1.0 ───────────────────────────────────────────
    print("\n[Test 5] t > t_max returns 1.0")
    d = rm.query(np.array([5.0, 2.5, 1.5]), t=999.0)
    print(f"  query(node0, t=999) = {d}  (expected 1.0)")
    if d != 1.0:
        errors.append(f"t > t_max returned {d} != 1.0")

    d_batch = rm.query(pts, t=999.0)
    if not np.allclose(d_batch, 1.0):
        errors.append(f"batch t > t_max wrong: {d_batch}")

    # ── Test 6: t=None defaults to t_max ────────────────────────────────────
    print("\n[Test 6] t=None uses t_max (last frame)")
    d = rm.query(np.array([25.0, 17.5, 1.5]), t=None)
    print(f"  query(node3, t=None) = {d}  (expected 0.6 = node_risks[5, 3])")
    if abs(d - 0.6) > 1e-6:
        errors.append(f"t=None default wrong: {d}")

    # ── Test 7: t < start_time clamps to row 0 ──────────────────────────────
    print("\n[Test 7] t < start_time clamps to row 0")
    d = rm.query(np.array([5.0, 2.5, 1.5]), t=-50.0)
    print(f"  query(node0, t=-50) = {d}  (expected 0.9 = row 0)")
    if abs(d - 0.9) > 1e-6:
        errors.append(f"t < start_time clamp wrong: {d}")

    # ── Test 8: input validation ────────────────────────────────────────────
    print("\n[Test 8] Constructor input validation")
    try:
        Tier1RiskMap(np.zeros(6), nodes)  # 1-D risks
    except ValueError:
        print("  PASS: 1-D node_risks raises ValueError")
    else:
        errors.append("1-D node_risks did not raise")

    try:
        Tier1RiskMap(node_risks, nodes[:3])  # mismatched node count
    except ValueError:
        print("  PASS: node count mismatch raises ValueError")
    else:
        errors.append("node count mismatch did not raise")

    try:
        Tier1RiskMap(node_risks, nodes, dt=0.0)
    except ValueError:
        print("  PASS: dt=0 raises ValueError")
    else:
        errors.append("dt=0 did not raise")

    # ── Test 9: from_model_output (skipped if torch missing) ────────────────
    print("\n[Test 9] from_model_output (torch path)")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  SKIP: torch not installed")
    else:
        import torch as _torch

        # Mock: (B=2, N=4, T_out=6). Take batch_index=0.
        mock = _torch.zeros(2, 4, 6)
        mock[0] = _torch.tensor([
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],   # node 0 — all 0.9
            [0.1, 0.3, 0.5, 0.7, 0.8, 0.9],   # node 1
            [0.1, 0.2, 0.4, 0.6, 0.7, 0.8],   # node 2
            [0.0, 0.0, 0.1, 0.2, 0.4, 0.6],   # node 3
        ])
        rm2 = Tier1RiskMap.from_model_output(mock, nodes, batch_index=0)
        d = rm2.query(np.array([5.0, 2.5, 1.5]), t=0.0)
        if abs(d - 0.9) > 1e-6:
            errors.append(f"from_model_output → query(node0, t=0) wrong: {d}")
        d = rm2.query(np.array([25.0, 17.5, 1.5]), t=50.0)  # node 3, row 5 → 0.6
        if abs(d - 0.6) > 1e-6:
            errors.append(f"from_model_output → query(node3, t=50) wrong: {d}")

        # Wrong tensor rank
        try:
            Tier1RiskMap.from_model_output(_torch.zeros(4, 6), nodes)
        except ValueError:
            print("  PASS: 2-D model_output raises ValueError")
        else:
            errors.append("from_model_output 2-D tensor did not raise")

        print("  PASS" if not errors else "  FAIL")

    # ── Verdict ─────────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("PASS")
    print("=" * 60)
