"""Risk-aware edge weights for the building graph.

Each edge ``(u, v)`` carries a static ``length`` (metres). Edge weight for
A* is computed as::

    weight(u → v) = base_cost · length + risk_scale · integrated_risk(u, v, t)

where ``integrated_risk`` is the mean of ``risk_map.query`` evaluated at
``n_samples`` evenly-spaced points along the edge segment. An edge whose
integrated risk exceeds ``risk_threshold`` is flagged ``passable=False``;
:func:`remove_impassable_edges` deletes such edges so the planner cannot
route through hazardous corridors.

Sampling N=5 by default — small enough to stay cheap, dense enough to
catch a fire occupying part of a corridor. Numbers come from
``configs/path_planning.yaml`` and ``docs/interface_contracts.md`` §5.

The functions in this module mutate ``graph`` *in place*. Callers that
need a snapshot of the weights for repeated planning at a fixed time
should pass a deepcopied graph, or just re-call :func:`compute_edge_weights`
at each replan tick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import numpy as np

from src.risk_map.risk_map_class import RiskMap


# ─── Config ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EdgeWeightConfig:
    """Edge-weight hyperparameters.

    Defaults reproduce ``configs/path_planning.yaml::edge_weights``. They
    are intentionally redundant with the YAML so any caller can rely on
    the dataclass without parsing config.

    Attributes:
        base_cost: Multiplier on edge ``length`` (m). ``1.0`` → weight in
            time-equivalent units when divided by walking speed.
        risk_scale: Multiplier on the mean risk along the edge. Larger
            values make A* more averse to dangerous corridors.
        risk_threshold: Edges whose mean risk exceeds this are marked
            impassable.
        n_samples: Number of equally-spaced points (including both
            endpoints) used to integrate risk along the edge.
    """

    base_cost: float = 1.0
    risk_scale: float = 10.0
    risk_threshold: float = 0.9
    n_samples: int = 5

    def __post_init__(self) -> None:
        if self.base_cost < 0:
            raise ValueError(f"base_cost must be ≥ 0, got {self.base_cost}")
        if self.risk_scale < 0:
            raise ValueError(f"risk_scale must be ≥ 0, got {self.risk_scale}")
        if not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError(
                f"risk_threshold must be in [0, 1], got {self.risk_threshold}"
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
) -> None:
    """Set ``weight``, ``edge_risk``, and ``passable`` on every edge (in place).

    For each undirected edge ``(u, v)``:

    * ``edge_risk = integrated_edge_risk(pos_u, pos_v, risk_map, t, N)``
    * ``weight   = base_cost · length + risk_scale · edge_risk``
    * ``passable = edge_risk ≤ risk_threshold``

    The mutation is idempotent — repeated calls with the same risk map
    and ``t`` produce identical attributes.

    Args:
        graph: Building graph from :func:`src.path_planning.building_graph.build_graph`.
        risk_map: :class:`RiskMap` consulted at every sample point.
        t: Simulation time in seconds.
        config: Hyperparameters. Defaults to :class:`EdgeWeightConfig()`.

    Raises:
        ValueError: If any node is missing the ``pos`` attribute or any
            edge is missing ``length``.
    """
    cfg = config or EdgeWeightConfig()
    for u, v, attrs in graph.edges(data=True):
        length = attrs.get("length")
        if length is None or length <= 0:
            raise ValueError(
                f"edge ({u}, {v}) missing positive 'length' attribute "
                f"(got {length!r})"
            )
        pu, pv = _edge_endpoints_xyz(graph, u, v)
        risk = integrated_edge_risk(pu, pv, risk_map, t, cfg.n_samples)
        attrs["edge_risk"] = risk
        attrs["weight"] = cfg.base_cost * float(length) + cfg.risk_scale * risk
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
    print("\n[4] Weight formula check on hall_n ↔ hall_e edge")
    if g.has_edge("hall_n", "hall_e"):
        a = g["hall_n"]["hall_e"]
        expected = cfg.base_cost * a["length"] + cfg.risk_scale * a["edge_risk"]
        if abs(a["weight"] - expected) > 1e-6:
            errors.append(
                f"weight drift: got {a['weight']:.4f}, expected {expected:.4f}"
            )
        else:
            print(f"  PASS: weight = {a['weight']:.4f}, edge_risk={a['edge_risk']:.3f}")
    else:
        errors.append("expected edge hall_n ↔ hall_e missing")

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
