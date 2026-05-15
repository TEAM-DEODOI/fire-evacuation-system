"""Map a RiskMap's danger value back to a CO ppm proxy (D-049, S3 only).

The Tier 1 GNN (and any aggregated-danger surrogate) outputs a single
``danger`` channel in ``[0, 1]`` that combines temperature, visibility,
and CO via the tenability conversion in
:func:`~src.risk_map.tenability.tenability_to_danger`. The model does
not expose raw CO ppm.

The :func:`~src.path_planning.edge_weights.compute_edge_weights`
FED term needs raw CO ppm to compute the predicted FED contribution
of crossing an edge:

    edge_fed = avg_CO_ppm * (length / walking_speed / 60) / FED_REFERENCE

For S3 (model-only planner) we want to keep model-side fairness
intact -- the planner must not read the FDS truth's CO field, only
the model's danger field. This module provides a thin adapter that
**back-projects** the danger value into a CO ppm proxy::

    co_ppm = danger * co_ceiling_ppm

The default ``co_ceiling_ppm = 1400`` is the ISO 13571 instantaneous
danger threshold. The mapping is monotonic in danger, so A*'s
*ordering* of paths is preserved even though the absolute FED values
disagree with the truth's. That ordering is what matters for the
"choose the lowest-FED path" objective, so the proxy is sufficient
to drive S3's planner toward FED-minimising routes without leaking
the oracle CO field.

The adapter exposes the same ``query(xyz, t=...)`` signature as
:class:`~src.risk_map.co_field.StaticCOField`, so it can be passed
directly as the ``co_field`` argument of
:func:`~src.path_planning.edge_weights.compute_edge_weights` and
:meth:`~src.integration.drone_swarm.DroneSwarm.update`.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from src.risk_map.risk_map_class import RiskMap
from src.shared.constants import TENABILITY


class RiskMapCOProxy:
    """Adapter exposing a :class:`RiskMap` as a ``StaticCOField``-shaped object.

    Each ``query(xyz, t)`` call returns the underlying RiskMap's
    ``danger`` scaled by ``co_ceiling_ppm``. Out-of-bounds behaviour is
    inherited from the wrapped RiskMap (which conventionally returns
    ``1.0`` for OOB-XYZ -> proxy returns ``co_ceiling_ppm``).

    Args:
        risk_map: Any :class:`RiskMap` implementation (Tier1RiskMap,
            EnsembleDecoderRiskMap, StaticRiskMap, ...).
        co_ceiling_ppm: CO ppm value corresponding to ``danger = 1.0``.
            Default ``TENABILITY.CO_DANGER_PPM`` (1400 ppm, the ISO 13571
            instantaneous danger threshold). Lower values produce a
            milder proxy; higher values make the planner more averse
            to risky cells.
    """

    def __init__(
        self,
        risk_map: RiskMap,
        co_ceiling_ppm: float = float(TENABILITY.CO_DANGER_PPM),
    ) -> None:
        if co_ceiling_ppm <= 0:
            raise ValueError(
                f"co_ceiling_ppm must be > 0, got {co_ceiling_ppm}"
            )
        self.risk_map: RiskMap = risk_map
        self.co_ceiling_ppm: float = float(co_ceiling_ppm)

    # ── Pass-through metadata (so callers can introspect time range) ──
    @property
    def start_time(self) -> float:
        return float(getattr(self.risk_map, "start_time", 0.0))

    @property
    def t_max(self) -> float:
        return float(getattr(self.risk_map, "t_max", float("inf")))

    # ── StaticCOField-compatible query ────────────────────────────────
    def query(
        self,
        xyz: np.ndarray,
        t: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Return CO ppm proxy at ``xyz`` and time ``t``.

        Args:
            xyz: ``(3,)`` single point or ``(M, 3)`` batch.
            t: Wall time (s). Forwarded as-is to the underlying RiskMap;
                no extra clamping here (the wrapped object's own clamp
                policy applies -- e.g. Tier1RiskMap clamps to its
                ``[start_time, t_max]`` range per D-038).

        Returns:
            CO ppm value(s). Scalar if ``xyz`` is ``(3,)``, ``(M,)``
            array if ``xyz`` is ``(M, 3)``.
        """
        danger = self.risk_map.query(xyz, t=t)
        if isinstance(danger, np.ndarray):
            return danger.astype(np.float64) * self.co_ceiling_ppm
        return float(danger) * self.co_ceiling_ppm


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    from src.risk_map.risk_map_class import StaticRiskMap
    from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS

    print("=" * 60)
    print("co_proxy.py self-test (D-049)")
    print("=" * 60)

    errors: list[str] = []

    nx_, ny_, nz_ = GRID_SHAPE
    times = np.arange(0.0, N_TIMESTEPS * DT_SLCF, DT_SLCF)

    # Build a synthetic RiskMap with known danger pattern.
    danger = np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32)
    danger[:, 10:20, 10:20, :] = 0.5     # mild zone
    danger[:, 30:40, 10:20, :] = 0.9     # hot zone
    rm = StaticRiskMap(danger_array=danger, times=times)

    # ── 1. ceiling validation ──────────────────────────────────────────
    print("\n[1] co_ceiling_ppm validation")
    try:
        RiskMapCOProxy(rm, co_ceiling_ppm=-100.0)
    except ValueError:
        print("  PASS: negative ceiling -> ValueError")
    else:
        errors.append("negative ceiling did not raise")

    # ── 2. single-point query scales danger by ceiling ─────────────────
    print("\n[2] single-point query returns danger * ceiling")
    proxy = RiskMapCOProxy(rm, co_ceiling_ppm=1400.0)
    p_mild = np.array([7.5, 7.5, 1.5])     # cell (15, 15, 3) -> danger 0.5
    p_hot  = np.array([17.5, 7.5, 1.5])    # cell (35, 15, 3) -> danger 0.9
    p_safe = np.array([25.0, 5.0, 1.5])    # cell (50, 10, 3) -> danger 0.0

    co_mild = float(proxy.query(p_mild, t=100.0))
    co_hot  = float(proxy.query(p_hot,  t=100.0))
    co_safe = float(proxy.query(p_safe, t=100.0))
    expected_mild = 0.5 * 1400.0
    expected_hot  = 0.9 * 1400.0
    print(f"  mild  : danger 0.5  -> proxy {co_mild:7.1f} ppm  (expect {expected_mild:.1f})")
    print(f"  hot   : danger 0.9  -> proxy {co_hot:7.1f} ppm  (expect {expected_hot:.1f})")
    print(f"  safe  : danger 0.0  -> proxy {co_safe:7.1f} ppm  (expect 0.0)")
    if abs(co_mild - expected_mild) > 1e-3:
        errors.append(f"mild proxy off: {co_mild} vs {expected_mild}")
    if abs(co_hot - expected_hot) > 1e-3:
        errors.append(f"hot proxy off: {co_hot} vs {expected_hot}")
    if abs(co_safe - 0.0) > 1e-3:
        errors.append(f"safe proxy off: {co_safe}")

    # ── 3. batch query preserves vectorisation ─────────────────────────
    print("\n[3] batch query")
    pts = np.stack([p_mild, p_hot, p_safe])
    batch = proxy.query(pts, t=100.0)
    if not isinstance(batch, np.ndarray) or batch.shape != (3,):
        errors.append(f"batch shape wrong: {type(batch)}, {getattr(batch, 'shape', None)}")
    elif (
        abs(batch[0] - expected_mild) > 1e-3
        or abs(batch[1] - expected_hot)  > 1e-3
        or abs(batch[2] - 0.0)           > 1e-3
    ):
        errors.append(f"batch values wrong: {batch}")
    else:
        print(f"  PASS: batch = {batch}")

    # ── 4. monotonicity (proxy ordering matches risk ordering) ─────────
    print("\n[4] proxy is monotonic in underlying danger")
    if not (co_hot > co_mild > co_safe):
        errors.append(
            f"ordering broken: hot={co_hot}, mild={co_mild}, safe={co_safe}"
        )
    else:
        print(f"  PASS: hot({co_hot:.0f}) > mild({co_mild:.0f}) > safe({co_safe:.0f})")

    # ── 5. start_time / t_max passthrough ──────────────────────────────
    print("\n[5] start_time / t_max passthrough")
    if proxy.start_time != float(getattr(rm, "start_time", 0.0)):
        errors.append("start_time passthrough wrong")
    if proxy.t_max != float(getattr(rm, "t_max", float("inf"))):
        errors.append("t_max passthrough wrong")
    print(f"  PASS: start_time={proxy.start_time}, t_max={proxy.t_max}")

    # ── 6. lower ceiling produces lower ppm ────────────────────────────
    print("\n[6] custom co_ceiling_ppm")
    soft = RiskMapCOProxy(rm, co_ceiling_ppm=500.0)
    co_hot_soft = float(soft.query(p_hot, t=100.0))
    if abs(co_hot_soft - 0.9 * 500.0) > 1e-3:
        errors.append(f"custom ceiling off: {co_hot_soft}")
    else:
        print(f"  PASS: ceiling 500 -> {co_hot_soft:.1f} ppm")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: RiskMapCOProxy validated")
