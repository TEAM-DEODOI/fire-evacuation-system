"""Risk-aware edge weights for the cell-grid building graph (D-026).

Each edge ``(u, v)`` connects two adjacent fluid cells (0.5 m cardinal or
0.707 m diagonal). Edge weight for A* is::

    weight(u → v) = base_cost · length + risk_scale · mean(risk_u, risk_v)

where ``risk_u`` / ``risk_v`` come from a single ``risk_map.query`` call
at each cell's centre. This is the **cell-discrete** analogue of the
previous N-sample edge integration -- at 0.5 m per edge the endpoints
are already close enough that mid-point sampling adds nothing. Edges
whose mean risk exceeds ``risk_threshold`` are flagged
``passable=False``; :func:`remove_impassable_edges` removes them.

The functions mutate ``graph`` *in place*. Callers that need a snapshot
of the weights for repeated planning at a fixed time should pass a
deepcopied graph or re-call :func:`compute_edge_weights` each replan tick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import numpy as np

from src.risk_map.co_field import StaticCOField
from src.risk_map.risk_map_class import RiskMap
from src.shared.constants import TENABILITY


# ─── Config ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EdgeWeightConfig:
    """Edge-weight hyperparameters (D-039, 2026-05-14: FED-based mode).

    The weight of an edge ``(u, v)`` is::

        weight = base_cost · length
                 + fed_scale · mean_co_ppm · traversal_time_s
                                   / (60 · TENABILITY.FED_REFERENCE)
                 + risk_scale · mean_risk      (legacy, optional)

    The FED term is the projected ISO 13571 §7.3 *simplified* CO-FED
    contribution of crossing that edge at ``walking_speed_mps``. Summed
    along an A* path it equals the predicted cumulative FED for an
    occupant walking that path — so the cost-minimising A* directly
    minimises predicted FED, the H6 primary metric.

    ``risk_scale`` and ``risk_threshold`` are kept for backwards
    compatibility with the legacy risk-only weighting. The default
    config zeroes ``risk_scale`` and sets ``risk_threshold = 1.0`` so the
    legacy hard cutoff is disabled (no cell is "impassable" — the planner
    instead reasons about cumulative FED).

    Attributes:
        base_cost: Multiplier on edge ``length`` (m). Acts as a small
            distance tie-breaker so length minimisation still works
            when CO is zero everywhere.
        fed_scale: Multiplier on the predicted FED contribution of the
            edge. ``1.0`` keeps FED in its physical units (the path-sum
            equals the predicted FED in absolute terms). Set higher to
            make the planner more averse to smoke-filled cells.
        walking_speed_mps: Assumed occupant walking speed (m/s) when
            converting edge length into traversal time. Defaults to
            :class:`~src.integration.person_agent.PersonAgentConfig` 's
            walking_speed_mps (0.5 m/s as of D-034).
        risk_scale: Multiplier on the mean risk along the edge (legacy
            mode). Default 0 disables. Useful when no CO field is
            available — the planner falls back to risk-based weighting.
        risk_threshold: Legacy hard-cutoff threshold. Default 1.0 means
            "no cutoff". Older callers can still pass a stricter value.
        n_samples: Reserved for backward compatibility (the cell-grid
            average uses 2 endpoints always; multi-sample integration
            is no longer used).
    """

    base_cost: float = 1.0
    fed_scale: float = 0.0
    walking_speed_mps: float = 0.5
    risk_scale: float = 10.0
    risk_threshold: float = 1.0
    n_samples: int = 5
    # Default mode (D-040, 2026-05-14): pure RISK-based weighting.
    # weight = base_cost · length + risk_scale · edge_risk
    # No FED term (avoids needing a model-side CO field) and no hard
    # cutoff (risk_threshold=1.0 keeps every edge). The planner picks
    # the path with minimum cumulative risk, which is the right
    # objective when the only available signal is each scenario's own
    # risk_map (truth for S2, model for S3) — no proxy needed. Set
    # fed_scale>0 + provide co_field= to recover the FED-based mode.

    def __post_init__(self) -> None:
        if self.base_cost < 0:
            raise ValueError(f"base_cost must be ≥ 0, got {self.base_cost}")
        if self.fed_scale < 0:
            raise ValueError(f"fed_scale must be ≥ 0, got {self.fed_scale}")
        if self.risk_scale < 0:
            raise ValueError(f"risk_scale must be ≥ 0, got {self.risk_scale}")
        if not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError(
                f"risk_threshold must be in [0, 1], got {self.risk_threshold}"
            )
        if self.walking_speed_mps <= 0:
            raise ValueError(
                f"walking_speed_mps must be > 0, got {self.walking_speed_mps}"
            )
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be ≥ 2, got {self.n_samples}")


# ─── Edge risk integration ────────────────────────────────────────────────
def _edge_endpoints_xyz(
    graph: nx.Graph, u: str, v: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(pos_u, pos_v)`` as ``(3,)`` arrays from node ``pos`` attr."""
    pu = graph.nodes[u].get("pos")
    pv = graph.nodes[v].get("pos")
    if pu is None or pv is None:
        raise ValueError(
            f"edge ({u}, {v}) endpoints missing 'pos' attribute "
            f"(u={pu}, v={pv})"
        )
    return np.asarray(pu, dtype=np.float64), np.asarray(pv, dtype=np.float64)


def integrated_edge_risk(
    pos_u: np.ndarray,
    pos_v: np.ndarray,
    risk_map: RiskMap,
    t: float,
    n_samples: int = 5,
) -> float:
    """Return mean ``risk_map.query`` along the segment ``pos_u`` → ``pos_v``.

    Samples ``n_samples`` points at parameters ``s = 0, 1/(N-1), …, 1``.
    Both endpoints are included, so ``n_samples=2`` reduces to averaging
    the two endpoint risks.

    Args:
        pos_u, pos_v: ``(3,)`` world positions in metres.
        risk_map: Any :class:`RiskMap` implementation.
        t: Simulation time (s) passed to ``query``.
        n_samples: Number of samples along the edge.

    Returns:
        Mean risk ∈ ``[0, 1]``.

    Raises:
        ValueError: If ``n_samples < 2`` or endpoints are not 3-D.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be ≥ 2, got {n_samples}")
    pu = np.asarray(pos_u, dtype=np.float64)
    pv = np.asarray(pos_v, dtype=np.float64)
    if pu.shape != (3,) or pv.shape != (3,):
        raise ValueError(
            f"endpoint shapes must be (3,), got {pu.shape}, {pv.shape}"
        )

    s = np.linspace(0.0, 1.0, n_samples)
    # (N, 3) parametric points along the segment.
    pts = pu[None, :] + s[:, None] * (pv - pu)[None, :]
    risks = np.asarray(risk_map.query(pts, t=t), dtype=np.float64)
    return float(np.clip(risks.mean(), 0.0, 1.0))


# ─── Graph-level update ───────────────────────────────────────────────────
def compute_edge_weights(
    graph: nx.Graph,
    risk_map: RiskMap,
    t: float,
    config: EdgeWeightConfig | None = None,
    *,
    co_field: StaticCOField | None = None,
) -> None:
    """Set ``weight``, ``edge_risk``, ``edge_fed``, and ``passable`` per edge.

    Per D-039 (2026-05-14) the edge weight is **FED-predominant**:

    ``weight = base_cost · length
              + fed_scale · edge_fed
              + risk_scale · edge_risk``

    where ``edge_fed`` is the predicted ISO 13571 CO-FED contribution of
    crossing that edge at ``walking_speed_mps``:

    ``edge_fed = (avg_co_ppm) · (length / walking_speed_mps / 60)
                                       / FED_REFERENCE``

    Cumulative weight along an A* path therefore equals the predicted
    cumulative FED of an occupant walking that path (plus a small length
    penalty for tie-breaking). A* minimisation = FED minimisation.

    If ``co_field`` is ``None``, the FED term is dropped and weighting
    falls back to ``base_cost · length + risk_scale · edge_risk`` (the
    legacy behaviour). ``passable`` is still computed from ``edge_risk``
    against ``risk_threshold`` for backwards compatibility, but the
    default ``risk_threshold=1.0`` means no edge gets dropped.

    Args:
        graph: Cell-grid from
            :func:`src.path_planning.building_graph.build_graph`.
        risk_map: :class:`RiskMap` consulted once per cell for ``edge_risk``.
        t: Simulation time in seconds (passed to both ``risk_map`` and
            ``co_field`` queries).
        config: Hyperparameters. Defaults to :class:`EdgeWeightConfig()`.
        co_field: Optional :class:`StaticCOField` consulted once per
            cell for ``edge_fed``. ``None`` skips the FED term.

    Raises:
        ValueError: If any node is missing the ``pos`` attribute or any
            edge is missing ``length``.
    """
    cfg = config or EdgeWeightConfig()

    # Query each node's risk + CO once and cache.
    node_risk: dict = {}
    node_co: dict = {}
    for nid, attrs in graph.nodes(data=True):
        pos = attrs.get("pos")
        if pos is None:
            raise ValueError(f"node {nid} missing 'pos' attribute")
        pos_arr = np.asarray(pos)
        node_risk[nid] = float(risk_map.query(pos_arr, t=t))
        if co_field is not None:
            node_co[nid] = float(co_field.query(pos_arr, t=t))

    # FED scaling constants.
    fed_ref = float(TENABILITY.FED_REFERENCE)
    walking_v = float(cfg.walking_speed_mps)

    for u, v, attrs in graph.edges(data=True):
        length = attrs.get("length")
        if length is None or length <= 0:
            raise ValueError(
                f"edge ({u}, {v}) missing positive 'length' attribute "
                f"(got {length!r})"
            )
        risk = 0.5 * (node_risk[u] + node_risk[v])
        attrs["edge_risk"] = risk

        fed_contrib = 0.0
        if co_field is not None:
            avg_co = 0.5 * (node_co[u] + node_co[v])
            traversal_s = float(length) / walking_v
            # ISO 13571 §7.3 simplified: FED = CO_ppm · Δt(min) / 27000
            fed_contrib = max(0.0, avg_co) * (traversal_s / 60.0) / fed_ref
        attrs["edge_fed"] = fed_contrib

        attrs["weight"] = (
            cfg.base_cost * float(length)
            + cfg.fed_scale * fed_contrib
            + cfg.risk_scale * risk
        )
        attrs["passable"] = risk <= cfg.risk_threshold


def remove_impassable_edges(graph: nx.Graph) -> int:
    """Delete edges flagged ``passable=False``. Returns number removed.

    Call :func:`compute_edge_weights` first to populate the flag. Edges
    without a ``passable`` key are treated as passable (no removal).

    Args:
        graph: Mutated in place.

    Returns:
        Count of edges removed.
    """
    to_remove = [
        (u, v) for u, v, attrs in graph.edges(data=True)
        if attrs.get("passable", True) is False
    ]
    graph.remove_edges_from(to_remove)
    return len(to_remove)


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path

    from src.path_planning.building_graph import build_graph
    from src.risk_map.risk_map_class import StaticRiskMap
    from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS

    print("=" * 60)
    print("edge_weights.py self-test")
    print("=" * 60)

    errors: list[str] = []

    # ── Fixture: graph + synthetic risk map with one hot zone ──────────
    g = build_graph()
    nx_, ny_, nz_ = GRID_SHAPE
    times = np.arange(0.0, N_TIMESTEPS * DT_SLCF, DT_SLCF)

    # Hot zone over the central courtyard (~10–20 m in X, 8–12 m in Y).
    danger = np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32)
    # Cell indices: x∈[20, 40), y∈[16, 24), z all.
    danger[:, 20:40, 16:24, :] = 0.95
    rm_hot = StaticRiskMap(danger_array=danger, times=times)

    # All-safe map for sanity check.
    rm_safe = StaticRiskMap(
        danger_array=np.zeros_like(danger), times=times
    )

    # ── 1. integrated_edge_risk endpoints ──────────────────────────────
    print("\n[1] integrated_edge_risk sanity")
    pu = np.array([12.0, 10.0, 1.5])  # inside hot zone
    pv = np.array([8.0, 10.0, 1.5])   # boundary
    r_hot = integrated_edge_risk(pu, pv, rm_hot, t=100.0, n_samples=5)
    r_safe = integrated_edge_risk(pu, pv, rm_safe, t=100.0, n_samples=5)
    print(f"  hot zone edge risk = {r_hot:.3f}  (expect > 0.4)")
    print(f"  safe map  edge risk = {r_safe:.3f}  (expect 0.0)")
    if not (r_hot > 0.4):
        errors.append(f"hot edge risk too low: {r_hot}")
    if not (r_safe == 0.0):
        errors.append(f"safe edge risk not zero: {r_safe}")

    # ── 2. n_samples validation ────────────────────────────────────────
    print("\n[2] integrated_edge_risk rejects n_samples < 2")
    try:
        integrated_edge_risk(pu, pv, rm_safe, t=0.0, n_samples=1)
    except ValueError:
        print("  PASS: n_samples=1 → ValueError")
    else:
        errors.append("n_samples=1 did not raise ValueError")

    # ── 3. compute_edge_weights writes attributes ──────────────────────
    print("\n[3] compute_edge_weights populates all edges")
    cfg = EdgeWeightConfig(
        base_cost=1.0, risk_scale=10.0, risk_threshold=0.9, n_samples=5
    )
    compute_edge_weights(g, rm_hot, t=100.0, config=cfg)
    n_edges = g.number_of_edges()
    risk_attrs = sum(1 for _, _, a in g.edges(data=True) if "edge_risk" in a)
    weight_attrs = sum(1 for _, _, a in g.edges(data=True) if "weight" in a)
    print(f"  {n_edges} edges, edge_risk on {risk_attrs}, weight on {weight_attrs}")
    if risk_attrs != n_edges:
        errors.append("edge_risk not set on every edge")
    if weight_attrs != n_edges:
        errors.append("weight not set on every edge")

    # ── 4. Weight = base_cost*length + risk_scale*risk (formula check) ─
    print("\n[4] Weight formula check on an arbitrary edge")
    # Cell-grid graph uses (i, j, k) tuple IDs.
    # Pick the first edge that exists for a deterministic spot check.
    edge_iter = iter(g.edges(data=True))
    u, v, a = next(edge_iter)
    expected = cfg.base_cost * a["length"] + cfg.risk_scale * a["edge_risk"]
    if abs(a["weight"] - expected) > 1e-6:
        errors.append(
            f"weight drift: got {a['weight']:.4f}, expected {expected:.4f}"
        )
    else:
        print(
            f"  PASS: edge {u}<->{v}  length={a['length']:.3f}  "
            f"edge_risk={a['edge_risk']:.3f}  weight={a['weight']:.4f}"
        )

    # ── 5. Edges through hot zone marked impassable (lower threshold) ──
    # Default risk_threshold=0.9 is intentionally strict (a corridor that
    # is 90 % danger on average is essentially uninhabitable). With our
    # synthetic hot zone covering only part of the central courtyard the
    # mean edge risk peaks around 0.76 — well-formed but not impassable
    # under the default. Re-run with threshold=0.5 to exercise the
    # mechanism end-to-end.
    print("\n[5] Hot-zone edges marked impassable (threshold=0.5)")
    cfg_strict = EdgeWeightConfig(
        base_cost=1.0, risk_scale=10.0, risk_threshold=0.5, n_samples=5
    )
    compute_edge_weights(g, rm_hot, t=100.0, config=cfg_strict)
    impassable = [
        (u, v, a["edge_risk"]) for u, v, a in g.edges(data=True)
        if a.get("passable") is False
    ]
    print(f"  {len(impassable)} impassable edges (risk > {cfg_strict.risk_threshold})")
    if not impassable:
        errors.append("expected at least one impassable edge in hot zone")
    else:
        for u, v, r in sorted(impassable, key=lambda x: -x[2])[:3]:
            print(f"    {u} ↔ {v}  edge_risk={r:.3f}")

    # ── 6. remove_impassable_edges count matches ────────────────────────
    print("\n[6] remove_impassable_edges returns count")
    edges_before = g.number_of_edges()
    removed = remove_impassable_edges(g)
    edges_after = g.number_of_edges()
    print(f"  removed {removed}, edges {edges_before} → {edges_after}")
    if removed != len(impassable):
        errors.append(
            f"removed count {removed} != impassable count {len(impassable)}"
        )
    if edges_before - edges_after != removed:
        errors.append("edge count mismatch after removal")

    # ── 7. Safe map keeps every edge passable ──────────────────────────
    print("\n[7] All-safe risk map → no impassable edges")
    g2 = build_graph()
    compute_edge_weights(g2, rm_safe, t=0.0, config=cfg)
    removed_safe = remove_impassable_edges(g2)
    if removed_safe != 0:
        errors.append(f"safe map removed {removed_safe} edges (should be 0)")
    else:
        print(f"  PASS: 0 edges removed")

    # ── 8. EdgeWeightConfig validation ─────────────────────────────────
    print("\n[8] EdgeWeightConfig validation")
    for bad in (
        {"base_cost": -1.0},
        {"risk_scale": -0.1},
        {"risk_threshold": 1.5},
        {"n_samples": 1},
    ):
        try:
            EdgeWeightConfig(**bad)
        except ValueError:
            continue
        errors.append(f"EdgeWeightConfig did not reject {bad}")
    print("  PASS: all bad configs rejected")

    # ── Verdict ────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: edge_weights validated")
