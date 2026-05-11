"""
L-shaped building represented as a NetworkX graph.

All downstream modules (path planning, RiskMap variants, Tier 1 GNN,
evaluation) consume :class:`BuildingGraph` so the building topology is
defined in exactly one place. See ``docs/decisions.md`` (D-003, D-016)
for the maze + 3-exit decision, and Day 2 prompt §3 for the node-layout
guide.

Node count: 19 (within the 16–20 target). Exits: 3.

Layout (world coordinates, metres)::

           Y=20  ┌──exit_north(8,18)───────────────────┐
                 │                                     │
           Y=16  │  Zone A     ├ zone_c_west / center / east
                 │  (NW slab)  │   (north strip)
           Y=14  │  zone_a_*   ├──int_north
                 │             │
           Y=12  │             ├──hall_n
                 │             │     ╲
           Y=10  │             ├──hall_w / hall_e (central courtyard)
                 │             │     ╱
           Y= 7  │             ├──hall_s
                 │             │
           Y= 5  ├ exit_west(0,5)  ├──int_south─────── zone_d_west/center
                 │                 │                       Zone D (SE)
           Y= 3  │   Zone B (south strip)
                 │   zone_b_west / center / east
           Y= 0  │
                 └─────────────────────────────────exit_east(30,13)
                X=0                                          X=30

Only XY are meaningful for adjacency (single floor, Z fixed at 1.5 m
breathing height per :data:`src.shared.constants.DOMAIN_SIZE_M`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import networkx as nx
import numpy as np
import yaml

from src.shared.constants import DOMAIN_SIZE_M, GRID_SHAPE
from src.shared.coordinates import cell_centres

NodeType = Literal["room", "corridor", "intersection", "exit"]

_BREATHING_Z_M: float = 1.5
"""Reference Z (breathing zone) for nodes. Single floor — all nodes share this Z."""


# ─── Dataclasses ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BuildingNode:
    """Single graph node in the building topology."""

    id: str
    pos: Tuple[float, float, float]   # world metres
    node_type: NodeType
    is_exit: bool = False
    has_detector: bool = False        # Tier 1 GNN — non-exit nodes get detectors


@dataclass
class BuildingGraph:
    """NetworkX wrapper for the L-shaped building.

    Attributes:
        graph: ``nx.Graph`` whose node IDs are strings; node attributes
            duplicate the :class:`BuildingNode` fields and include a
            ``"node"`` key pointing back at the dataclass instance.
            Edge attributes carry ``length`` (m) and ``width`` (m).
        exits: List of :class:`BuildingNode` with ``is_exit=True`` in
            insertion order.
        cell_to_node: SLCF cell index → nearest-node id, by XY distance.
            Derived from node positions; recomputed on load and never
            written to YAML.
    """

    graph: nx.Graph
    exits: List[BuildingNode]
    cell_to_node: Dict[Tuple[int, int, int], str] = field(default_factory=dict)

    # ─── Lookup helpers ────────────────────────────────────────────────────
    def get_node(self, node_id: str) -> BuildingNode:
        """Return the :class:`BuildingNode` for ``node_id``.

        Raises:
            KeyError: If ``node_id`` is not in the graph.
        """
        if node_id not in self.graph:
            raise KeyError(f"unknown node id: {node_id!r}")
        return self.graph.nodes[node_id]["node"]

    def all_nodes(self) -> List[BuildingNode]:
        """Return every :class:`BuildingNode` in insertion order."""
        return [self.graph.nodes[nid]["node"] for nid in self.graph.nodes()]

    def nearest_node(self, xyz: np.ndarray) -> BuildingNode:
        """Return the closest node to ``xyz`` (XY distance, single floor)."""
        return self._nearest(xyz, candidates=self.all_nodes())

    def nearest_exit(self, xyz: np.ndarray) -> BuildingNode:
        """Return the closest exit to ``xyz``."""
        if not self.exits:
            raise ValueError("graph has no exits")
        return self._nearest(xyz, candidates=self.exits)

    def is_node_in_zone(self, node_id: str, zone: str) -> bool:
        """Test whether ``node_id`` belongs to ``zone``.

        Recognised zone labels: ``A``, ``B``, ``C``, ``D`` (uppercase),
        ``central_hall``, ``intersection``, ``exit``. Comparison is
        case-sensitive for the zone label.
        """
        if node_id not in self.graph:
            raise KeyError(f"unknown node id: {node_id!r}")
        return _zone_of(node_id) == zone

    # ─── Internals ─────────────────────────────────────────────────────────
    @staticmethod
    def _nearest(
        xyz: np.ndarray, candidates: List[BuildingNode]
    ) -> BuildingNode:
        arr = np.asarray(xyz, dtype=np.float64)
        if arr.shape[-1] != 3:
            raise ValueError(f"xyz must have 3 components, got shape {arr.shape}")
        target = arr[:2]  # XY only — single floor
        best: BuildingNode | None = None
        best_dist = math.inf
        for node in candidates:
            dx = node.pos[0] - target[0]
            dy = node.pos[1] - target[1]
            d = math.hypot(dx, dy)
            if d < best_dist:
                best_dist = d
                best = node
        assert best is not None, "candidates was empty"
        return best


# ─── Zone classification ───────────────────────────────────────────────────
def _zone_of(node_id: str) -> str:
    """Derive zone label from a node ID prefix."""
    if node_id.startswith("zone_a_"):
        return "A"
    if node_id.startswith("zone_b_"):
        return "B"
    if node_id.startswith("zone_c_"):
        return "C"
    if node_id.startswith("zone_d_"):
        return "D"
    if node_id.startswith("hall_"):
        return "central_hall"
    if node_id.startswith("int_"):
        return "intersection"
    if node_id.startswith("exit_"):
        return "exit"
    return "unknown"


# ─── Default L-shape layout ────────────────────────────────────────────────
# Node positions follow the Day-2 prompt guide. (x, y) are world metres;
# Z fixed at the breathing-zone reference.
_DEFAULT_NODES: List[Tuple[str, float, float, NodeType, bool]] = [
    # (id, x, y, type, is_exit)
    # Zone A — NW slab
    ("zone_a_west",   3.0,  13.0, "room", False),
    ("zone_a_center", 8.0,  14.0, "room", False),
    # Zone B — south strip
    ("zone_b_west",    4.0, 4.0, "room", False),
    ("zone_b_center", 10.0, 3.0, "room", False),
    ("zone_b_east",   18.0, 3.0, "room", False),
    # Zone C — north strip
    ("zone_c_west",   15.0, 16.0, "room", False),
    ("zone_c_center", 22.0, 16.0, "room", False),
    ("zone_c_east",   28.0, 16.0, "room", False),
    # Zone D — SE
    ("zone_d_west",   24.0, 5.0, "room", False),
    ("zone_d_center", 28.0, 5.0, "room", False),
    # Central hall edges (around the courtyard)
    ("hall_n", 15.0, 12.0, "corridor", False),
    ("hall_s", 15.0,  7.0, "corridor", False),
    ("hall_e", 20.0,  9.0, "corridor", False),
    ("hall_w", 10.0,  9.0, "corridor", False),
    # Intersections
    ("int_north", 12.0, 14.0, "intersection", False),
    ("int_south", 15.0,  5.0, "intersection", False),
    # Exits
    ("exit_west",  0.0,  5.0, "exit", True),
    ("exit_north", 8.0, 18.0, "exit", True),
    ("exit_east", 30.0, 13.0, "exit", True),
]

_DEFAULT_EDGES: List[Tuple[str, str]] = [
    # West exit → Zone B chain
    ("exit_west", "zone_b_west"),
    ("zone_b_west", "zone_b_center"),
    ("zone_b_center", "zone_b_east"),
    ("zone_b_east", "int_south"),
    # South intersection → hall + Zone D
    ("int_south", "hall_s"),
    ("int_south", "zone_d_west"),
    ("zone_d_west", "zone_d_center"),
    ("zone_d_west", "hall_e"),
    ("zone_d_center", "exit_east"),
    # Central courtyard ring (hall_n, hall_e, hall_s, hall_w)
    ("hall_n", "hall_e"),
    ("hall_e", "hall_s"),
    ("hall_s", "hall_w"),
    ("hall_w", "hall_n"),
    # Hall → north intersection
    ("hall_n", "int_north"),
    ("hall_w", "int_north"),
    # North intersection → Zone A and Zone C
    ("int_north", "zone_a_center"),
    ("int_north", "zone_c_west"),
    # Zone A chain → north exit
    ("zone_a_center", "zone_a_west"),
    ("zone_a_center", "exit_north"),
    # Zone C chain → east exit
    ("zone_c_west", "zone_c_center"),
    ("zone_c_center", "zone_c_east"),
    ("zone_c_east", "exit_east"),
]

_DEFAULT_EDGE_WIDTH_M: float = 1.5
_WALKING_SPEED_MPS_SENTINEL: float = 1.5  # only used in tests outside this module


def build_default_graph() -> BuildingGraph:
    """Construct the canonical L-shaped building graph.

    Every non-exit node receives ``has_detector=True`` so the Tier 1 GNN
    has the 16-detector setup described in
    ``docs/tier1_gnn_design.md``.

    Returns:
        A fully-populated :class:`BuildingGraph` — graph, exits list and
        ``cell_to_node`` mapping are all set.

    Raises:
        ValueError: If any default node lies outside the SLCF region —
            this should never happen but guards future edits.
    """
    lx, ly, _lz = DOMAIN_SIZE_M

    g = nx.Graph()
    exits: List[BuildingNode] = []

    for nid, x, y, ntype, is_exit in _DEFAULT_NODES:
        if not (0.0 <= x <= lx and 0.0 <= y <= ly):
            raise ValueError(
                f"node {nid} at ({x}, {y}) outside SLCF region "
                f"[0, {lx}] × [0, {ly}]"
            )
        node = BuildingNode(
            id=nid,
            pos=(float(x), float(y), _BREATHING_Z_M),
            node_type=ntype,
            is_exit=is_exit,
            has_detector=not is_exit,
        )
        g.add_node(
            nid,
            node=node,
            pos=node.pos,
            node_type=node.node_type,
            is_exit=node.is_exit,
            has_detector=node.has_detector,
        )
        if is_exit:
            exits.append(node)

    for src, dst in _DEFAULT_EDGES:
        if src not in g.nodes or dst not in g.nodes:
            raise ValueError(f"edge references unknown node: {src} ↔ {dst}")
        p1 = g.nodes[src]["pos"]
        p2 = g.nodes[dst]["pos"]
        length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        g.add_edge(src, dst, length=length, width=_DEFAULT_EDGE_WIDTH_M)

    bg = BuildingGraph(
        graph=g,
        exits=exits,
        cell_to_node=_compute_cell_to_node(g),
    )
    return bg


# ─── Cell-to-node mapping ──────────────────────────────────────────────────
def _compute_cell_to_node(g: nx.Graph) -> Dict[Tuple[int, int, int], str]:
    """For every SLCF cell, find the nearest graph node by XY distance.

    Single floor → Z is ignored when measuring proximity. The result is a
    plain dict so it survives YAML round-tripping if a future caller
    wants to persist it.
    """
    x_centres, y_centres, _z_centres = cell_centres()
    node_ids = list(g.nodes())
    node_xy = np.array(
        [[g.nodes[n]["pos"][0], g.nodes[n]["pos"][1]] for n in node_ids],
        dtype=np.float64,
    )
    nx_cells, ny_cells, nz_cells = GRID_SHAPE

    # (nx, ny, 1, 2) - (1, 1, N, 2) → (nx, ny, N) after norm
    xx, yy = np.meshgrid(x_centres, y_centres, indexing="ij")
    cell_xy = np.stack([xx, yy], axis=-1)  # (nx, ny, 2)
    diff = cell_xy[:, :, None, :] - node_xy[None, None, :, :]
    dists = np.linalg.norm(diff, axis=-1)   # (nx, ny, N)
    nearest = np.argmin(dists, axis=-1)     # (nx, ny)

    mapping: Dict[Tuple[int, int, int], str] = {}
    for ix in range(nx_cells):
        for iy in range(ny_cells):
            owner = node_ids[int(nearest[ix, iy])]
            for iz in range(nz_cells):
                mapping[(ix, iy, iz)] = owner
    return mapping


# ─── YAML persistence ──────────────────────────────────────────────────────
def save_graph(graph: BuildingGraph, yaml_path: Path) -> None:
    """Serialise ``graph`` to a YAML file (overwrites any existing file).

    The ``cell_to_node`` mapping is **not** persisted — it is a function
    of node positions and is recomputed by :func:`load_graph`.
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    nodes_payload: List[Dict[str, object]] = []
    for nid in graph.graph.nodes():
        n: BuildingNode = graph.graph.nodes[nid]["node"]
        nodes_payload.append(
            {
                "id": n.id,
                "pos": [float(n.pos[0]), float(n.pos[1]), float(n.pos[2])],
                "type": n.node_type,
                "is_exit": bool(n.is_exit),
                "has_detector": bool(n.has_detector),
            }
        )
    edges_payload: List[Dict[str, object]] = []
    for src, dst, attrs in graph.graph.edges(data=True):
        edges_payload.append(
            {
                "source": src,
                "target": dst,
                "length": float(attrs.get("length", 0.0)),
                "width": float(attrs.get("width", _DEFAULT_EDGE_WIDTH_M)),
            }
        )

    payload = {
        "version": 1,
        "nodes": nodes_payload,
        "edges": edges_payload,
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def load_graph(yaml_path: Path) -> BuildingGraph:
    """Reconstruct a :class:`BuildingGraph` from a YAML file.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        ValueError: If the YAML structure is malformed.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"building YAML not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict) or "nodes" not in payload or "edges" not in payload:
        raise ValueError(f"malformed building YAML at {yaml_path}: missing nodes/edges")

    lx, ly, _lz = DOMAIN_SIZE_M
    g = nx.Graph()
    exits: List[BuildingNode] = []
    for raw in payload["nodes"]:
        pos = tuple(raw["pos"])
        if len(pos) != 3:
            raise ValueError(f"node {raw.get('id')} has bad pos: {raw['pos']}")
        if not (0.0 <= pos[0] <= lx and 0.0 <= pos[1] <= ly):
            raise ValueError(
                f"node {raw['id']} at {pos} outside SLCF region"
            )
        node = BuildingNode(
            id=str(raw["id"]),
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
            node_type=str(raw["type"]),  # type: ignore[arg-type]
            is_exit=bool(raw.get("is_exit", False)),
            has_detector=bool(raw.get("has_detector", False)),
        )
        g.add_node(
            node.id,
            node=node,
            pos=node.pos,
            node_type=node.node_type,
            is_exit=node.is_exit,
            has_detector=node.has_detector,
        )
        if node.is_exit:
            exits.append(node)

    for raw in payload["edges"]:
        src = str(raw["source"])
        dst = str(raw["target"])
        if src not in g.nodes or dst not in g.nodes:
            raise ValueError(f"edge references unknown node(s): {src} ↔ {dst}")
        length = float(raw.get("length", 0.0))
        if length <= 0.0:
            # Recompute from positions if missing/zero in YAML.
            p1 = g.nodes[src]["pos"]
            p2 = g.nodes[dst]["pos"]
            length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        width = float(raw.get("width", _DEFAULT_EDGE_WIDTH_M))
        g.add_edge(src, dst, length=length, width=width)

    return BuildingGraph(
        graph=g,
        exits=exits,
        cell_to_node=_compute_cell_to_node(g),
    )


# ─── Visualisation ────────────────────────────────────────────────────────
def visualize_graph(graph: BuildingGraph, save_path: Path) -> None:
    """Render the graph as a 2-D floor plan PNG.

    Exits are green, rooms are blue, central-hall corridors are orange,
    intersections are red, edges are grey. The SLCF region rectangle is
    drawn for context.

    Args:
        graph: Graph to render.
        save_path: Destination PNG path. Parent directories are created.
    """
    # Lazy import so the module remains usable in headless environments
    # without matplotlib.
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    colour_by_type = {
        "exit": "#2ca02c",
        "room": "#1f77b4",
        "corridor": "#ff7f0e",
        "intersection": "#d62728",
    }

    lx, ly, _ = DOMAIN_SIZE_M
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.add_patch(
        Rectangle((0.0, 0.0), lx, ly, fill=False, edgecolor="black", lw=1.0)
    )

    # Edges first (so nodes draw on top)
    for src, dst in graph.graph.edges():
        p1 = graph.graph.nodes[src]["pos"]
        p2 = graph.graph.nodes[dst]["pos"]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", lw=1.2, zorder=1)

    # Nodes
    for nid in graph.graph.nodes():
        node: BuildingNode = graph.graph.nodes[nid]["node"]
        colour = colour_by_type.get(node.node_type, "#444444")
        ax.scatter(
            node.pos[0], node.pos[1],
            s=140, c=colour, edgecolor="black", linewidths=0.8, zorder=2,
        )
        ax.annotate(
            nid,
            (node.pos[0], node.pos[1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_xlim(-1.0, lx + 1.0)
    ax.set_ylim(-1.0, ly + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Building graph — {graph.graph.number_of_nodes()} nodes, "
        f"{graph.graph.number_of_edges()} edges, {len(graph.exits)} exits"
    )

    # Legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="w",
               markerfacecolor=col, markeredgecolor="black", markersize=10, label=lbl)
        for lbl, col in colour_by_type.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("building.py self-test")
    print("=" * 60)

    errors: List[str] = []

    # ── 1. build_default_graph ───────────────────────────────────────────
    print("\n[1] build_default_graph()")
    bg = build_default_graph()
    n_nodes = bg.graph.number_of_nodes()
    n_edges = bg.graph.number_of_edges()
    print(f"  nodes={n_nodes}, edges={n_edges}, exits={len(bg.exits)}")
    if not (16 <= n_nodes <= 20):
        errors.append(f"node count {n_nodes} outside [16, 20]")
    if len(bg.exits) != 3:
        errors.append(f"exit count {len(bg.exits)} != 3")
    if not all(e.is_exit for e in bg.exits):
        errors.append("non-exit node marked as exit")

    # ── 2. Connectivity ──────────────────────────────────────────────────
    print("\n[2] Graph connectivity")
    if not nx.is_connected(bg.graph):
        errors.append("graph not connected")
    else:
        print("  PASS: nx.is_connected")

    # ── 3. Reachability from every node to every exit ────────────────────
    print("\n[3] Every node reaches every exit")
    reach_fails = 0
    for nid in bg.graph.nodes():
        for ex in bg.exits:
            if not nx.has_path(bg.graph, nid, ex.id):
                reach_fails += 1
                errors.append(f"no path {nid} → {ex.id}")
    if reach_fails == 0:
        print("  PASS")

    # ── 4. Edge lengths positive ─────────────────────────────────────────
    print("\n[4] Edge lengths positive")
    for src, dst, attrs in bg.graph.edges(data=True):
        if attrs.get("length", 0.0) <= 0.0:
            errors.append(f"edge {src}↔{dst} has non-positive length")
        if attrs.get("width", 0.0) <= 0.0:
            errors.append(f"edge {src}↔{dst} has non-positive width")
    print("  PASS" if not errors else "  FAIL")

    # ── 5. nearest_node from (15, 10, 1.5) ───────────────────────────────
    print("\n[5] nearest_node((15, 10, 1.5))")
    nn = bg.nearest_node(np.array([15.0, 10.0, 1.5]))
    nn_dist = math.hypot(nn.pos[0] - 15.0, nn.pos[1] - 10.0)
    print(f"  → {nn.id} at {nn.pos}, dist={nn_dist:.2f} m")
    # hall_e (20,9), hall_n (15,12), hall_w (10,9), hall_s (15,7) → hall_n is dist 2.0
    if nn.id not in {"hall_n", "hall_s", "hall_e", "hall_w"}:
        errors.append(f"unexpected nearest_node: {nn.id}")

    # ── 6. nearest_exit from (15, 10, 1.5) ───────────────────────────────
    print("\n[6] nearest_exit((15, 10, 1.5))")
    ne = bg.nearest_exit(np.array([15.0, 10.0, 1.5]))
    ne_dist = math.hypot(ne.pos[0] - 15.0, ne.pos[1] - 10.0)
    print(f"  → {ne.id} at {ne.pos}, dist={ne_dist:.2f} m")
    if ne.id != "exit_north":
        errors.append(f"expected exit_north (dist 10.6 m), got {ne.id}")

    # ── 7. is_node_in_zone ───────────────────────────────────────────────
    print("\n[7] is_node_in_zone")
    if not bg.is_node_in_zone("zone_a_west", "A"):
        errors.append("zone_a_west should be in zone A")
    if not bg.is_node_in_zone("hall_n", "central_hall"):
        errors.append("hall_n should be in central_hall")
    if not bg.is_node_in_zone("exit_west", "exit"):
        errors.append("exit_west should be in zone 'exit'")
    if bg.is_node_in_zone("zone_a_west", "B"):
        errors.append("zone_a_west should NOT be in B")

    # ── 8. cell_to_node coverage ─────────────────────────────────────────
    print("\n[8] cell_to_node maps every SLCF cell")
    nx_cells, ny_cells, nz_cells = GRID_SHAPE
    expected_cells = nx_cells * ny_cells * nz_cells
    if len(bg.cell_to_node) != expected_cells:
        errors.append(
            f"cell_to_node has {len(bg.cell_to_node)} entries, expected {expected_cells}"
        )
    else:
        print(f"  {len(bg.cell_to_node)} entries OK")

    # ── 9. YAML round-trip ───────────────────────────────────────────────
    print("\n[9] YAML save → load round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "building.yaml"
        save_graph(bg, yaml_path)
        bg2 = load_graph(yaml_path)
        if bg2.graph.number_of_nodes() != n_nodes:
            errors.append("loaded graph has wrong node count")
        if bg2.graph.number_of_edges() != n_edges:
            errors.append("loaded graph has wrong edge count")
        if len(bg2.exits) != 3:
            errors.append("loaded graph lost exits")
        for src, dst, attrs in bg.graph.edges(data=True):
            if not bg2.graph.has_edge(src, dst):
                errors.append(f"loaded graph missing edge {src}↔{dst}")
                continue
            la = attrs["length"]
            lb = bg2.graph.edges[src, dst]["length"]
            if abs(la - lb) > 1e-6:
                errors.append(f"edge {src}↔{dst} length drift {la} vs {lb}")
        print("  PASS" if not errors else "  FAIL")

    # ── 10. Visualisation (optional but useful) ──────────────────────────
    print("\n[10] visualize_graph() → figures/building_graph.png")
    project_root = Path(__file__).resolve().parents[2]
    png_path = project_root / "figures" / "building_graph.png"
    try:
        visualize_graph(bg, png_path)
        if png_path.exists() and png_path.stat().st_size > 0:
            print(f"  saved {png_path} ({png_path.stat().st_size} bytes)")
        else:
            errors.append("PNG file empty after visualize_graph")
    except ImportError as exc:
        print(f"  SKIP: matplotlib unavailable ({exc})")

    # ── Verdict ──────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: Building graph validated")
