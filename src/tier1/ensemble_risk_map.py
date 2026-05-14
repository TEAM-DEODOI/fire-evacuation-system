"""EnsembleDecoderRiskMap — RiskMap adapter for the L4h learned ensemble
decoder (D-028).

This is the cell-level β option for H6 path planning. Compare to:
* :class:`src.tier1.tier1_risk_map.Tier1RiskMap` — per-node GNN nearest-node
  lookup (option α, IoU 0.904 / FNR 4.6% per-node, coarser cell coverage).
* :class:`src.risk_map.risk_map_class.StaticRiskMap` — FDS ground truth
  (oracle, used as fairness baseline in EXP-PATH-001).

The decoder produces a (T_out=6, 60, 40, 6) cell-level danger forecast for
a single "forecast moment" t0. EXP-PATH-001's
``DynamicPredictivePlanner`` replans every 30 s, so each replan tick
creates a fresh ``EnsembleDecoderRiskMap`` with the next t0 — the path
planner sees a sliding 60 s window of cell-level risk.

Out-of-range rules:
* xyz outside [0, 30] × [0, 20] × [0, 3]  → 1.0
* t  outside [t0, t0 + T_out * dt]          → 1.0
   (forecasts are strictly forward-looking; the past is not modelled).
* xyz inside solid mask                    → multiplied to 0 via mask
   pre-application (decoder already learns ~0 from the mask channel, but
   the multiply removes residual leakage from the IDW projection step).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.interpolate as si
import torch

from src.risk_map.risk_map_class import RiskMap, _TOL
from src.shared.constants import DOMAIN_SIZE_M, DT_SLCF, GRID_SHAPE
from src.shared.coordinates import cell_centres
from src.tier1.ensemble_decoder import (
    PerCellEnsembleDecoder, decoder_forward_grid,
)

LOOKAHEAD_STEPS = 6     # matches D-026/D-028 decoder T_out.


class EnsembleDecoderRiskMap(RiskMap):
    """Cell-level RiskMap from the learned ensemble decoder (L4h, β).

    Args:
        decoded: (T_out=6, 60, 40, 6) float32 — decoder cell danger.
        t0: forecast issue time (s). Valid query window: [t0, t0+T_out·dt].
        dt: forecast cadence (s), default DT_SLCF = 10.
        mask: optional (60, 40, 6) — multiplied into `decoded` so solid
            cells render as 0 (visualization fairness; metrics are already
            mask-filtered upstream).

    Raises:
        ValueError: on shape mismatches.
    """

    def __init__(
        self,
        decoded: np.ndarray,
        t0: float,
        dt: float = DT_SLCF,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        darr = np.asarray(decoded, dtype=np.float32)
        if darr.ndim != 4:
            raise ValueError(
                f"decoded must be 4-D (T, X, Y, Z), got {darr.shape}"
            )
        if darr.shape[1:] != GRID_SHAPE:
            raise ValueError(
                f"decoded spatial shape {darr.shape[1:]} != {GRID_SHAPE}"
            )
        T_out = darr.shape[0]

        if mask is not None:
            m = np.asarray(mask, dtype=np.float32)
            if m.shape != GRID_SHAPE:
                raise ValueError(
                    f"mask shape {m.shape} != {GRID_SHAPE}"
                )
            darr = darr * (m > 0.5).astype(np.float32)[None, ...]

        # Build the time grid. Frame i targets wall time t0 + (i+1)·dt;
        # prepending a synthetic "t = t0" frame == frame 0 makes queries
        # at the boundary natural without special-casing the interpolator.
        times = np.array(
            [t0 + (i + 1) * dt for i in range(T_out)], dtype=np.float64
        )
        times_full = np.concatenate([[t0], times])
        darr_full = np.concatenate([darr[:1], darr], axis=0)

        x_c, y_c, z_c = cell_centres()
        self.t0 = float(t0)
        self.dt = float(dt)
        self.T_out = T_out
        self.times = times_full
        self.start_time = self.t0
        self.t_max = float(times_full[-1])
        self.danger = darr_full
        self.x, self.y, self.z = x_c, y_c, z_c
        self.interp = si.RegularGridInterpolator(
            (times_full, x_c, y_c, z_c),
            darr_full,
            method="linear",
            bounds_error=False,
            fill_value=1.0,
        )

    # ─── RiskMap.query ──────────────────────────────────────────────────
    def query(
        self,
        xyz: np.ndarray,
        t: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        arr = np.asarray(xyz, dtype=np.float64)
        if t is None:
            t = self.t_max
        if t > self.t_max + _TOL or t < self.start_time - _TOL:
            if arr.ndim == 1:
                return 1.0
            return np.ones(arr.shape[0], dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError(
                    f"single-point xyz must have 3 components, got {arr.shape}"
                )
            return self._query_single(arr, t)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return self._query_batch(arr, t)
        raise ValueError(
            f"xyz must have shape (3,) or (M, 3), got {arr.shape}"
        )

    def _in_bounds(self, xyz: np.ndarray) -> bool:
        lx, ly, lz = DOMAIN_SIZE_M
        return bool(
            -_TOL <= xyz[0] <= lx + _TOL
            and -_TOL <= xyz[1] <= ly + _TOL
            and -_TOL <= xyz[2] <= lz + _TOL
        )

    def _query_single(self, xyz: np.ndarray, t: float) -> float:
        if not self._in_bounds(xyz):
            return 1.0
        x = float(np.clip(xyz[0], self.x[0], self.x[-1]))
        y = float(np.clip(xyz[1], self.y[0], self.y[-1]))
        z = float(np.clip(xyz[2], self.z[0], self.z[-1]))
        pt = np.array([[t, x, y, z]], dtype=np.float64)
        return float(np.asarray(self.interp(pt)).item())

    def _query_batch(self, xyz: np.ndarray, t: float) -> np.ndarray:
        lx, ly, lz = DOMAIN_SIZE_M
        in_bounds = (
            (xyz[:, 0] >= -_TOL) & (xyz[:, 0] <= lx + _TOL)
            & (xyz[:, 1] >= -_TOL) & (xyz[:, 1] <= ly + _TOL)
            & (xyz[:, 2] >= -_TOL) & (xyz[:, 2] <= lz + _TOL)
        )
        clamped = np.column_stack([
            np.clip(xyz[:, 0], self.x[0], self.x[-1]),
            np.clip(xyz[:, 1], self.y[0], self.y[-1]),
            np.clip(xyz[:, 2], self.z[0], self.z[-1]),
        ])
        ts = np.full(clamped.shape[0], t)
        points = np.column_stack([ts, clamped])
        out = self.interp(points)
        out[~in_bounds] = 1.0
        return out.astype(np.float32)

    # ─── End-to-end factory for H6 ──────────────────────────────────────
    @classmethod
    def from_scenario(
        cls,
        scen_name: str,
        t0: float,
        *,
        gnn_model,
        adj,
        sparse_conv_model,
        sparse_fno_model,
        decoder: PerCellEnsembleDecoder,
        mask: np.ndarray,
        sensor_indicator: np.ndarray,
        knn_idx: np.ndarray,
        knn_w: np.ndarray,
        seq_dir: Path,
        raw_root: Path,
        device: torch.device = torch.device("cpu"),
    ) -> "EnsembleDecoderRiskMap":
        """Run the full surrogate stack at (scen_name, t0) and wrap.

        Used by H6 EvacuationSimulator at each replan tick. Cost: ~1-2 s
        per call on CPU (dominated by the FDS extraction). For EXP-PATH-001
        the simulator caches the slice extraction across replan ticks.

        Note: forward helpers live in scripts/ for historical reasons;
        we lazy-import them rather than pulling them into src/ (the
        90_next_steps follow-up tracks the eventual refactor).
        """
        import sys
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from src.data_pipeline.fds_extractor import extract_slices
        from src.data_pipeline.normalize import (
            build_input_tensor, normalize_scenario,
        )
        from src.risk_map.converter import prediction_to_danger
        from evaluate_sparse_fno import (
            build_sparse_6ch_input, autoregress_sparse_fno,
        )
        from evaluate_ensemble import (
            gnn_node_pred_to_cell_danger, tier1_forward,
        )
        from visualize_60s_5model import (
            autoregress_sparse_input, sparsify_initial_input,
        )

        sdir = raw_root / scen_name
        slices = extract_slices(sdir)
        norm = normalize_scenario(slices)
        inp = build_input_tensor(norm, mask, times=slices["times"])
        inp_6ch = build_sparse_6ch_input(slices, mask, sensor_indicator)

        t0_idx = int(t0 // DT_SLCF)
        t_start = t0_idx - 6     # T_IN
        times_arr = np.array(
            [t0 + (s + 1) * DT_SLCF for s in range(LOOKAHEAD_STEPS)]
        )

        init_sparse = sparsify_initial_input(inp[t0_idx], sensor_indicator)
        preds_norm = autoregress_sparse_input(
            sparse_conv_model, init_sparse, sensor_indicator, t0, device,
        )
        sparse_conv_danger = prediction_to_danger(preds_norm, times_arr)

        preds_norm = autoregress_sparse_fno(
            sparse_fno_model, inp_6ch[t0_idx], sensor_indicator, t0, device,
            resparsify=True,
        )
        sparse_fno_danger = prediction_to_danger(preds_norm, times_arr)

        t1_node = tier1_forward(scen_name, seq_dir, gnn_model, adj,
                                  t_start, device)
        gnn_cell = gnn_node_pred_to_cell_danger(t1_node, knn_idx, knn_w)

        decoded = decoder_forward_grid(
            decoder, gnn_cell, sparse_conv_danger, sparse_fno_danger,
            mask, device=device,
        )
        return cls(decoded=decoded, t0=t0, mask=mask)


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EnsembleDecoderRiskMap self-test")
    print("=" * 60)

    nx, ny, nz = GRID_SHAPE
    T_out = LOOKAHEAD_STEPS

    # Synthetic decoder output — fire ball at (15, 10, 1.25) growing in radius
    decoded = np.zeros((T_out, nx, ny, nz), dtype=np.float32)
    for i in range(T_out):
        r = 3 + i
        for ix in range(max(30 - r, 0), min(30 + r, nx)):
            for iy in range(max(20 - r, 0), min(20 + r, ny)):
                if (ix - 30) ** 2 + (iy - 20) ** 2 <= r ** 2:
                    decoded[i, ix, iy, 2:4] = 0.85
    mask = np.ones(GRID_SHAPE, dtype=np.float32)
    mask[0:3, :, :] = 0.0     # carve out a "solid" strip on the west wall

    rm = EnsembleDecoderRiskMap(decoded=decoded, t0=120.0, mask=mask)
    errors: list[str] = []

    print("\n[1] inside fire @ t = t0 + 30s (=150s)")
    d = rm.query(np.array([15.0, 10.0, 1.25]), t=150.0)
    print(f"  query((15, 10, 1.25), t=150) = {d:.3f}  (expected ~0.85)")
    if abs(d - 0.85) > 0.05:
        errors.append(f"in-fire query = {d}")

    print("\n[2] outside fire blob @ t=150")
    d = rm.query(np.array([25.0, 18.0, 1.25]), t=150.0)
    print(f"  query((25, 18, 1.25), t=150) = {d:.3f}  (expected ~0)")
    if d > 0.05:
        errors.append(f"far-from-fire = {d}")

    print("\n[3] inside solid (x = 0.5, mask=0)")
    d = rm.query(np.array([0.5, 10.0, 1.25]), t=150.0)
    print(f"  query((0.5, 10, 1.25), t=150) = {d:.3f}  (expected ~0)")
    if d > 0.05:
        errors.append(f"solid-cell = {d}")

    print("\n[4] t > t_max → 1.0")
    d = rm.query(np.array([15.0, 10.0, 1.25]), t=999.0)
    if abs(d - 1.0) > 1e-6:
        errors.append(f"t > t_max single = {d}")
    print(f"  single point = {d}  OK")
    db = rm.query(np.array([[15.0, 10.0, 1.25], [5.0, 5.0, 1.5]]), t=999.0)
    if not np.allclose(db, 1.0):
        errors.append(f"t > t_max batch = {db}")
    print(f"  batch        = {db}  OK")

    print("\n[5] t < t0 → 1.0 (forecasts are forward-only)")
    d = rm.query(np.array([15.0, 10.0, 1.25]), t=60.0)
    if abs(d - 1.0) > 1e-6:
        errors.append(f"t < t0 = {d}")
    print(f"  query at t=60s (< t0=120s) = {d}  OK")

    print("\n[6] OOB space → 1.0")
    for p in (np.array([-1.0, 10.0, 1.5]),
              np.array([35.0, 10.0, 1.5]),
              np.array([15.0, 10.0, 4.0])):
        v = rm.query(p, t=150.0)
        if v != 1.0:
            errors.append(f"OOB {p.tolist()} → {v}")
    print("  PASS")

    print("\n[7] batch query mixing in-fire / safe / OOB")
    pts = np.array([
        [15.0, 10.0, 1.25],    # fire
        [5.0,  5.0,  1.5],     # safe
        [-1.0, 10.0, 1.5],     # OOB
    ])
    v = rm.query(pts, t=150.0)
    print(f"  result = {v}")
    if not (v.shape == (3,) and v[0] > 0.5 and v[1] < 0.1 and v[2] == 1.0):
        errors.append(f"batch wrong: {v}")

    print("\n[8] t = t0 exactly (boundary)")
    d_t0 = rm.query(np.array([15.0, 10.0, 1.25]), t=120.0)
    d_t0p10 = rm.query(np.array([15.0, 10.0, 1.25]), t=130.0)
    print(f"  query(t=t0=120) = {d_t0:.3f}, query(t=t0+10=130) = {d_t0p10:.3f}")

    print("\n[9] subclass of RiskMap")
    assert isinstance(rm, RiskMap)
    print("  PASS")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: EnsembleDecoderRiskMap validated")
