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
from typing import List, Optional, Tuple

import numpy as np

from src.integration.person_agent import PersonAgent
from src.integration.scene import Scene
from src.path_planning.building_graph import (
    build_graph,
    exit_nodes,
    fluid_cells_at,
    load_default_fluid_mask,
    load_interior_mask,
)
from src.risk_map.co_field import StaticCOField
from src.risk_map.risk_map_class import RiskMap, StaticRiskMap
from src.shared.constants import CELL_SIZE_M, DT_SLCF, GRID_SHAPE, N_TIMESTEPS


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
CO_CACHE_DIR: Path = Path("results/cache/scenario_co_fields")


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


def zero_co_field() -> StaticCOField:
    """All-zero CO field. Used as the fallback when no FDS scenario loads.

    With this field, :func:`src.risk_map.fed.accumulate_fed_co` returns 0
    for every step, so ``casualty_rate`` and ``cumulative_fed`` stay
    flat — matching the previous behaviour of S1/S2/S3 before CO was
    wired in.
    """
    nx_, ny_, nz_ = GRID_SHAPE
    times = np.arange(0.0, N_TIMESTEPS * DT_SLCF, DT_SLCF)
    return StaticCOField(
        co_array=np.zeros((N_TIMESTEPS, nx_, ny_, nz_), dtype=np.float32),
        times=times,
    )


def load_truth_co_field(fds_dir: Path, verbose: bool = True) -> StaticCOField:
    """Build (or load from cache) the FDS raw-CO field for FED accumulation.

    Mirrors :func:`load_truth_risk_map`: tries the npz cache first under
    ``CO_CACHE_DIR / <fds_dir.name>.npz``; falls back to building from
    FDS slices (~30 s) and writes the cache. On any failure returns the
    zero CO field so the scenario still completes — exposure / FED will
    just remain 0 for that run, with a clear stdout warning.

    Args:
        fds_dir: Path to the FDS scenario directory.
        verbose: Print progress / cache hits.

    Returns:
        A :class:`StaticCOField` (either FDS-truth or zero fallback).
    """
    cache_key = fds_dir.name if fds_dir.name else "_unnamed"
    cache_path = CO_CACHE_DIR / f"{cache_key}.npz"

    if cache_path.exists():
        try:
            if verbose:
                print(f"  [co_field] cache hit: {cache_path}")
            return StaticCOField.from_npy(cache_path)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(
                    f"  [co_field] cache read failed ({cache_path}): {exc}; "
                    f"re-loading from FDS"
                )

    if not fds_dir.exists():
        if verbose:
            print(
                f"  [co_field] FDS dir missing ({fds_dir}); "
                f"falling back to zero CO"
            )
        return zero_co_field()

    try:
        if verbose:
            print(
                f"  [co_field] loading FDS CO field from {fds_dir} "
                f"(first call: ~30 s)"
            )
        cf = StaticCOField.from_fds_dir(fds_dir)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(
                f"  [co_field] FDS load failed "
                f"({exc.__class__.__name__}: {exc}); falling back to zero CO"
            )
        return zero_co_field()

    try:
        CO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cf.save(cache_path)
        if verbose:
            size_kb = cache_path.stat().st_size / 1024
            print(f"  [co_field] cached -> {cache_path} ({size_kb:.0f} KB)")
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [co_field] cache write failed: {exc}")

    return cf


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
    *,
    fluid_mask: Optional[np.ndarray] = None,
    only_exit_reachable: bool = True,
    interior_mask: Optional[np.ndarray] = None,
) -> List[PersonAgent]:
    """Spawn agents at random *indoor* cells (D-026 cell grid).

    The spawn pool is the breathing-zone (``k=3``) slice of the
    **interior mask** — the union of cells reached by FDS smoke across
    all scenarios, ANDed with the fluid mask. This excludes outdoor
    fluid cells (open courtyard, building exterior) that the raw fluid
    mask alone would allow. When ``only_exit_reachable`` (default) the
    pool is further restricted to the connected component containing
    the three exit cells. The agent's XY is jittered uniformly within
    ``±max_jitter`` of the cell centre so the capsule stays inside the
    chosen cell.

    Args:
        scene: Active :class:`Scene`.
        n_persons: Max agents to spawn (capped at indoor-cell count).
        seed: RNG seed for cell selection and jitter.
        fluid_mask: ``(60, 40, 6)`` boolean mask. Loaded from
            ``data/processed/dataset.h5`` when ``None``. Still used to
            build the connectivity graph for exit-reachability.
        only_exit_reachable: If ``True``, restrict the spawn pool to
            cells in the same connected component as the 3 exits.
        interior_mask: ``(60, 40, 6)`` boolean indoor mask. Loaded from
            ``data/processed/interior_mask.npz`` when ``None``.

    Returns:
        List of :class:`PersonAgent`.

    Raises:
        RuntimeError: If no indoor cells are available.
    """
    if fluid_mask is None:
        fluid_mask = load_default_fluid_mask()
    if interior_mask is None:
        interior_mask = load_interior_mask()

    # Spawn pool: indoor fluid cells at the breathing-zone z-slice.
    layer = interior_mask[:, :, 3]
    cells = [(i, j) for i in range(layer.shape[0])
             for j in range(layer.shape[1]) if layer[i, j]]
    if not cells:
        raise RuntimeError("no indoor cells available for spawn")

    if only_exit_reachable:
        import networkx as nx
        graph = build_graph(fluid_mask=fluid_mask)
        exit_ids = exit_nodes(graph)
        if exit_ids:
            exit_component = nx.node_connected_component(graph, exit_ids[0])
            if all(e in exit_component for e in exit_ids):
                reachable = {(i, j) for (i, j, _k) in exit_component}
                cells = [c for c in cells if c in reachable]
                if not cells:
                    raise RuntimeError(
                        "no indoor cells are exit-reachable; "
                        "check interior_mask coverage"
                    )

    rng = np.random.default_rng(seed)
    n = min(n_persons, len(cells))
    indices = rng.choice(len(cells), size=n, replace=False)

    # Capsule radius 0.25 m in a 0.5 m cell -> max safe jitter 0.20 m.
    max_jitter = max(0.05, CELL_SIZE_M / 2 - 0.30)
    z_breathing = 0.25 + CELL_SIZE_M * 3  # 1.75 m

    agents: List[PersonAgent] = []
    for k_idx, cell_idx in enumerate(indices):
        i, j = cells[int(cell_idx)]
        cx = 0.25 + CELL_SIZE_M * i
        cy = 0.25 + CELL_SIZE_M * j
        jitter = rng.uniform(-max_jitter, max_jitter, size=2)
        start = np.array(
            [cx + jitter[0], cy + jitter[1], z_breathing],
            dtype=np.float64,
        )
        a = PersonAgent(agent_id=f"person_{k_idx}_cell_{i}_{j}")
        a.spawn(scene.client, start)
        agents.append(a)
    return agents
