"""Shared helpers for ``src/integration/scenarios/*`` (D-025 Week 12).

The three scenarios (S1 fixed-sign, S2 FDS drone swarm, S3 PI-FNO
drone swarm) share four pieces of plumbing:

1. Path to the placeholder building URDF (the real STL → URDF flow
   lands in M1-full once ``trimesh`` is wired up).
2. A no-fire :class:`StaticRiskMap` (for the zero-risk fallback path
   when no FDS scenario is available).
3. A cache-aware FDS RiskMap loader.
4. Helpers that read the canonical building graph (room nodes, exits).

Extracting these here keeps the scenario files focused on the *guidance*
logic that actually differs between S1/S2/S3.

This module is private to the ``scenarios`` package — note the leading
underscore. Tests / external callers should go through the public
``run()`` function of each scenario.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.integration.person_agent import PersonAgent
from src.integration.scene import Scene
from src.path_planning.building_graph import build_graph, exit_nodes
from src.risk_map.risk_map_class import RiskMap, StaticRiskMap
from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS


# ─── Constants ────────────────────────────────────────────────────────────
BUILDING_URDF: Path = Path("assets/building.urdf")
"""Real-STL-derived building URDF (M1-full, 2026-05-14).

Generated once via ``build_building_urdf(assets/science_hall_lv5.stl, ...)``
and re-loaded by every scenario. Mesh tag references the STL with
``scale="0.001 0.001 0.001"`` (mm -> m per L-010).
"""

PLACEHOLDER_URDF: Path = Path("assets/placeholder_building.urdf")
"""Fallback 9-box L-shape URDF. Useful for unit tests that need fast
load + deterministic geometry, but does **not** match the
``shared/building.py`` 19-node graph topology -- some graph edges
cross its solid interior partitions. Production scenarios should
prefer :data:`BUILDING_URDF`.
"""

CACHE_DIR: Path = Path("results/cache/scenario_risk_maps")


def building_urdf_path(prefer_real: bool = True) -> Path:
    """Return the URDF a scenario should load.

    Default behaviour: prefer the real STL-derived URDF; fall back to
    the placeholder if it is missing. ``prefer_real=False`` forces the
    placeholder (used by tests that depend on its specific 9-box
    topology -- e.g. M2-mini partition collision).
    """
    if prefer_real and BUILDING_URDF.exists():
        return BUILDING_URDF
    return PLACEHOLDER_URDF


# ─── RiskMap helpers ──────────────────────────────────────────────────────
def zero_risk_map() -> StaticRiskMap:
    """All-safe risk map. Used when no FDS scenario is available."""
    nx_, ny_, nz_ = GRID_SHAPE
    times = np.arange(0.0, N_TIMESTEPS * DT_SLCF, DT_SLCF)
    return StaticRiskMap(
        danger_array=np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32),
        times=times,
    )


def load_truth_risk_map(fds_dir: Path, verbose: bool = True) -> RiskMap:
    """Build (or load from cache) the FDS-derived :class:`StaticRiskMap`.

    Cache location: ``CACHE_DIR / <fds_dir.name>.npz`` (~740 KB per
    scenario). First load via :meth:`StaticRiskMap.from_fds_dir` takes
    ~30 s; subsequent loads are <1 s.

    Falls back to :func:`zero_risk_map` on missing dir or fdsreader
    failure (with a clear stdout warning, never crashes).

    Args:
        fds_dir: Path to the FDS scenario directory.
        verbose: Print progress / cache hits.

    Returns:
        A :class:`RiskMap` (either the FDS-truth one, or the
        all-zero fallback).
    """
    cache_key = fds_dir.name if fds_dir.name else "_unnamed"
    cache_path = CACHE_DIR / f"{cache_key}.npz"

    if cache_path.exists():
        try:
            if verbose:
                print(f"  [risk_map] cache hit: {cache_path}")
            return StaticRiskMap.from_npy(cache_path)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(
                    f"  [risk_map] cache read failed ({cache_path}): {exc}; "
                    f"re-loading from FDS"
                )

    if not fds_dir.exists():
        if verbose:
            print(
                f"  [risk_map] FDS dir missing ({fds_dir}); "
                f"falling back to zero risk"
            )
        return zero_risk_map()

    try:
        if verbose:
            print(
                f"  [risk_map] loading FDS RiskMap from {fds_dir} "
                f"(first call: ~30 s)"
            )
        rm = StaticRiskMap.from_fds_dir(fds_dir)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(
                f"  [risk_map] FDS load failed "
                f"({exc.__class__.__name__}: {exc}); "
                f"falling back to zero risk"
            )
        return zero_risk_map()

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rm.save(cache_path)
        if verbose:
            size_kb = cache_path.stat().st_size / 1024
            print(f"  [risk_map] cached -> {cache_path} ({size_kb:.0f} KB)")
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [risk_map] cache write failed: {exc}")

    return rm


# ─── Graph + agents helpers ───────────────────────────────────────────────
def exit_positions() -> List[np.ndarray]:
    """All exit-node positions as (3,) world arrays at breathing height."""
    graph = build_graph()
    return [
        np.asarray(graph.nodes[nid]["pos"], dtype=np.float64)
        for nid in exit_nodes(graph)
    ]


def spawn_agents(
    scene: Scene,
    n_persons: int,
    seed: int,
) -> List[PersonAgent]:
    """Spawn up to ``n_persons`` agents at distinct room-node centres.

    Each agent's start XY is jittered uniformly within ±0.3 m of the
    canonical room-node position to avoid stacking on the same spot.
    The choice of which room nodes are picked is deterministic for a
    given ``seed``.
    """
    graph = build_graph()
    room_node_ids = [
        nid for nid, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "room"
    ]
    if not room_node_ids:
        raise RuntimeError("building graph has no room nodes")

    rng = np.random.default_rng(seed)
    n = min(n_persons, len(room_node_ids))
    chosen = list(rng.choice(room_node_ids, size=n, replace=False))

    agents: List[PersonAgent] = []
    for i, nid in enumerate(chosen):
        base = np.asarray(graph.nodes[nid]["pos"], dtype=np.float64)
        jitter = rng.uniform(-0.3, 0.3, size=2)
        start = np.array([base[0] + jitter[0], base[1] + jitter[1], base[2]])
        a = PersonAgent(agent_id=f"person_{i}_{nid}")
        a.spawn(scene.client, start)
        agents.append(a)
    return agents
