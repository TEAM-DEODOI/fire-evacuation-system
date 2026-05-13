"""Diagnose mismatch between assets/building.urdf and shared/building.py graph.

After the A2 transition to the real STL the EXP-PATH-001 sweep collapses
to ~0 % evacuation success rate across most rows. The hypothesis is that
the canonical 19-node building graph (positions hard-coded in
:mod:`src.shared.building`) does not match the actual STL geometry, so:

1. Some graph nodes land *inside* STL walls -> agents spawn already
   penetrating a wall, the very first ``step_toward`` call is vetoed
   and they stay there forever.
2. Some graph edges cut through walls -> the planner gives waypoints
   along edges that are not physically traversable.

This script probes both:

* For each graph node, place a kinematic capsule of the same dimensions
  as :class:`PersonAgent` at that XY (z chosen so it rests on the floor)
  and ask :func:`pybullet.getClosestPoints(distance=0.0)`. A non-empty
  result with negative ``distance`` means the capsule is overlapping
  the building mesh at that position.
* For each graph edge, sample 11 points uniformly along the segment
  (including both endpoints) and run the same probe. Report the number
  of penetrating samples per edge.

Outputs:

* Console text report (per-node + per-edge tables).
* ``figures/diagnostics/building_vs_graph.png`` -- top-down view of the
  STL footprint with nodes (green / orange / red) and edges (green / red).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

from src.path_planning.building_graph import build_graph, exit_nodes
from src.shared.building import BuildingNode  # for type hint


# ─── Probe params -- match PersonAgent capsule exactly ────────────────────
_RADIUS = 0.25
_CYL_H = 1.7 - 2 * _RADIUS  # 1.2 m
_Z_CENTER = _CYL_H / 2 + _RADIUS  # 0.85 m
_EDGE_SAMPLES = 11


def _connect_and_load(urdf: Path) -> Tuple[int, int]:
    """Connect PyBullet DIRECT mode and load the building URDF."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)
    building_id = p.loadURDF(
        str(urdf),
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
        physicsClientId=cid,
    )
    return cid, building_id


def _make_probe(cid: int) -> int:
    col = p.createCollisionShape(
        p.GEOM_CAPSULE, radius=_RADIUS, height=_CYL_H,
        physicsClientId=cid,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=[100.0, 100.0, _Z_CENTER],  # far away
        physicsClientId=cid,
    )


def _probe_at(
    cid: int, probe_id: int, building_id: int, xy: Tuple[float, float],
) -> float:
    """Move probe to (x, y, Z_CENTER) and return min distance to building.

    Negative = penetrating; 0 = touching; positive = clear.
    Returns +inf if no contact within 1 m (clear).
    """
    p.resetBasePositionAndOrientation(
        probe_id, [xy[0], xy[1], _Z_CENTER], [0, 0, 0, 1],
        physicsClientId=cid,
    )
    closest = p.getClosestPoints(
        bodyA=probe_id, bodyB=building_id, distance=1.0,
        physicsClientId=cid,
    )
    if not closest:
        return float("inf")
    return float(min(c[8] for c in closest))


def _classify_node(dist: float) -> str:
    if dist == float("inf") or dist > 0.05:
        return "free"          # clearly inside open space
    if dist > -1e-6:
        return "touching"      # grazing a wall
    return "embedded"          # capsule overlaps wall


_NODE_COLOR = {"free": "#2ca02c", "touching": "#ff7f0e", "embedded": "#d62728"}
_NODE_MARKER = {"free": "o", "touching": "s", "embedded": "X"}


# ─── Reports ──────────────────────────────────────────────────────────────
def diagnose_nodes(
    cid: int, probe_id: int, building_id: int, graph,
) -> Dict[str, Dict]:
    print("=" * 70)
    print("NODE PROBE -- capsule (r=0.25, h=1.7) at each graph node's XY")
    print("=" * 70)
    print(f"{'node_id':<22} {'pos (x,y)':<16} {'dist [m]':>10}  status")
    print("-" * 70)
    rows: Dict[str, Dict] = {}
    for nid, attrs in graph.nodes(data=True):
        pos = attrs["pos"]
        d = _probe_at(cid, probe_id, building_id, (pos[0], pos[1]))
        cls = _classify_node(d)
        dist_str = "+inf" if d == float("inf") else f"{d:+.3f}"
        print(f"{nid:<22} ({pos[0]:>5.1f},{pos[1]:>5.1f})  {dist_str:>10}  {cls}")
        rows[nid] = {"pos": (pos[0], pos[1]), "dist": d, "class": cls}
    n_free = sum(1 for r in rows.values() if r["class"] == "free")
    n_touch = sum(1 for r in rows.values() if r["class"] == "touching")
    n_embed = sum(1 for r in rows.values() if r["class"] == "embedded")
    print("-" * 70)
    print(f"Summary: free={n_free}  touching={n_touch}  embedded={n_embed}  "
          f"(total {len(rows)})")
    return rows


def diagnose_edges(
    cid: int, probe_id: int, building_id: int, graph, node_data,
) -> List[Dict]:
    print()
    print("=" * 70)
    print(f"EDGE PROBE -- {_EDGE_SAMPLES} samples per edge")
    print("=" * 70)
    print(f"{'edge':<40} {'penetrating':>12} {'min dist':>10}  status")
    print("-" * 70)
    rows: List[Dict] = []
    for u, v in graph.edges():
        pu = np.asarray(graph.nodes[u]["pos"][:2])
        pv = np.asarray(graph.nodes[v]["pos"][:2])
        dists: List[float] = []
        for s in np.linspace(0.0, 1.0, _EDGE_SAMPLES):
            pt = pu + s * (pv - pu)
            d = _probe_at(cid, probe_id, building_id, (float(pt[0]), float(pt[1])))
            dists.append(d)
        finite = [d for d in dists if d != float("inf")]
        min_d = min(finite) if finite else float("inf")
        n_penetrate = sum(1 for d in dists if d < -1e-6)
        status = "blocked" if n_penetrate > 0 else "clear"
        edge_label = f"{u} -- {v}"
        min_str = "+inf" if min_d == float("inf") else f"{min_d:+.3f}"
        print(f"{edge_label:<40} {n_penetrate}/{_EDGE_SAMPLES:>3}        "
              f"{min_str:>10}  {status}")
        rows.append({
            "u": u, "v": v, "pu": pu, "pv": pv,
            "n_penetrating": n_penetrate, "min_dist": min_d,
            "status": status,
        })
    n_clear = sum(1 for r in rows if r["status"] == "clear")
    n_block = sum(1 for r in rows if r["status"] == "blocked")
    print("-" * 70)
    print(f"Summary: clear={n_clear}  blocked={n_block}  (total {len(rows)})")
    return rows


# ─── Render ───────────────────────────────────────────────────────────────
def render(
    node_results: Dict[str, Dict],
    edge_results: List[Dict],
    stl_path: Path,
    out: Path,
) -> Path:
    """Top-down XY view: STL footprint outline + nodes + edges."""
    import trimesh
    mesh = trimesh.load(str(stl_path), force="mesh")
    # Apply same scaling as urdf_builder (mm -> m).
    raw_ext = float((mesh.bounds[1] - mesh.bounds[0]).max())
    s = 0.001 if raw_ext > 1e3 else 1.0
    verts = mesh.vertices * s   # (V, 3)
    faces = mesh.faces          # (F, 3)
    # Slice at z = _Z_CENTER (the breathing-zone height). Walls extend
    # vertically through 0..3.2 m so we just project ALL triangle edges
    # to XY; the resulting outline is the "wall map" at any horizontal slice.
    edges_xy: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for f in faces:
        for i in range(3):
            a = verts[f[i]][:2]
            b = verts[f[(i + 1) % 3]][:2]
            edges_xy.append(((float(a[0]), float(a[1])),
                             (float(b[0]), float(b[1]))))

    fig, ax = plt.subplots(figsize=(11.0, 7.0))

    # 1. building wall outline (light gray)
    from matplotlib.collections import LineCollection
    lc = LineCollection(edges_xy, colors="#999999", linewidths=0.4, alpha=0.7)
    ax.add_collection(lc)

    # 2. graph edges
    for e in edge_results:
        color = "#2ca02c" if e["status"] == "clear" else "#d62728"
        ax.plot(
            [e["pu"][0], e["pv"][0]], [e["pu"][1], e["pv"][1]],
            color=color, lw=2.0, alpha=0.85, zorder=3,
        )

    # 3. graph nodes
    for nid, r in node_results.items():
        ax.scatter(
            r["pos"][0], r["pos"][1],
            c=_NODE_COLOR[r["class"]],
            marker=_NODE_MARKER[r["class"]],
            s=180, edgecolor="black", linewidths=1.0, zorder=5,
        )
        ax.annotate(
            nid, r["pos"], textcoords="offset points", xytext=(6, 6),
            fontsize=7, zorder=6,
        )

    ax.set_xlim(-1.5, 31.5)
    ax.set_ylim(-1.5, 21.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"shared/building.py 19-node graph vs assets/building.urdf "
        f"(STL projected outline at z=1.5 m)\n"
        f"nodes: green=free, orange=touching, red=embedded   "
        f"edges: green=clear, red=blocked"
    )
    ax.grid(True, linestyle=":", alpha=0.4)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=_NODE_COLOR["free"], markeredgecolor="black",
               markersize=10, label="node: free"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor=_NODE_COLOR["touching"], markeredgecolor="black",
               markersize=10, label="node: touching wall"),
        Line2D([0], [0], marker="X", linestyle="None",
               markerfacecolor=_NODE_COLOR["embedded"], markeredgecolor="black",
               markersize=10, label="node: embedded in wall"),
        Line2D([0], [0], color="#2ca02c", lw=2.0, label="edge: clear"),
        Line2D([0], [0], color="#d62728", lw=2.0, label="edge: blocked"),
        Line2D([0], [0], color="#999999", lw=1.0, label="STL triangle edges"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    urdf = Path("assets/building.urdf")
    stl = Path("assets/science_hall_lv5.stl")
    if not urdf.exists():
        print(f"FAIL: {urdf} missing", file=sys.stderr); return 1
    if not stl.exists():
        print(f"FAIL: {stl} missing", file=sys.stderr); return 1

    cid, building_id = _connect_and_load(urdf)
    probe_id = _make_probe(cid)
    graph = build_graph()

    node_results = diagnose_nodes(cid, probe_id, building_id, graph)
    edge_results = diagnose_edges(cid, probe_id, building_id, graph, node_results)

    # Highlight exits.
    print()
    print("=" * 70)
    print("EXIT NODES (the targets every agent tries to reach)")
    print("=" * 70)
    for ex_id in exit_nodes(graph):
        r = node_results[ex_id]
        print(f"  {ex_id:<14} dist={r['dist']:+.3f}m  class={r['class']}")

    out_fig = Path("figures/diagnostics/building_vs_graph.png")
    render(node_results, edge_results, stl, out_fig)
    print(f"\nSaved diagnostic figure -> {out_fig}")

    p.disconnect(physicsClientId=cid)

    n_embed = sum(1 for r in node_results.values() if r["class"] == "embedded")
    n_block = sum(1 for r in edge_results if r["status"] == "blocked")
    print()
    print(f"Bottom line: {n_embed}/{len(node_results)} graph nodes are inside walls, "
          f"{n_block}/{len(edge_results)} graph edges cross walls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
