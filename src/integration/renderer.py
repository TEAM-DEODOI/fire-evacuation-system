"""Offline renderers for :class:`SimulationRecorder` logs.

The recorder writes a stable frame schema; renderers consume it. Adding
new visualisations (e.g. drone overlay, FED heatmap, 3D view) extends
*this* module only -- the simulation code and the recorder schema do
not need to change.

Currently provided:

* :func:`render_trajectories` -- top-down PNG of all agent trails with
  building footprint + exits + (optional) risk heatmap.
* :func:`render_snapshot` -- single PNG at one frame index.
* :func:`render_comparison` -- side-by-side panels (e.g. S1 / S2 / S3).
* :func:`render_animation` -- frame-by-frame GIF.

All renderers accept a precomputed :class:`SimulationRecorder` plus an
optional ``fluid_mask`` for the building footprint. Risk overlays are
optional and computed on-demand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.integration.recorder import SimulationRecorder
from src.shared.constants import CELL_SIZE_M, DOMAIN_SIZE_M, GRID_SHAPE


# ─── Constants ────────────────────────────────────────────────────────────
_STATUS_COLOR = {
    "alive": "#1f77b4",          # blue
    "evacuated": "#2ca02c",      # green
    "dead": "#d62728",           # red
    "unknown": "#888888",        # gray
}

_STATUS_MARKER = {
    "alive": "X",                # X if still inside building at end
    "evacuated": "s",            # square at exit
    "dead": "x",                 # x (lowercase) for casualty
    "unknown": "o",
}

_DRONE_STATUS_COLOR = {
    "searching": "#9467bd",      # purple (patrolling)
    "guiding":   "#ff7f0e",      # orange (active shepherd)
    "idle":      "#7f7f7f",      # grey
}


# ─── Helpers ──────────────────────────────────────────────────────────────
def _draw_building_footprint(
    ax,
    fluid_mask: Optional[np.ndarray] = None,
    z_layer: int = 3,
    alpha: float = 0.25,
) -> None:
    """Render the building's wall outline (top-down at ``z_layer``)."""
    if fluid_mask is None:
        from src.path_planning.building_graph import load_default_fluid_mask
        fluid_mask = load_default_fluid_mask()
    walls = 1.0 - fluid_mask[:, :, z_layer].astype(np.float32)
    lx, ly, _lz = DOMAIN_SIZE_M
    ax.imshow(
        walls.T,                  # transpose so (x, y) -> (column, row)
        cmap="Greys",
        alpha=alpha,
        extent=(0.0, lx, 0.0, ly),
        origin="lower",
        interpolation="nearest",
        zorder=0,
    )


def _draw_risk_overlay(
    ax,
    risk_grid: np.ndarray,
    alpha: float = 0.5,
    vmax: float = 1.0,
) -> None:
    """Render a (60, 40) risk grid as a heatmap overlay."""
    lx, ly, _lz = DOMAIN_SIZE_M
    cmap = LinearSegmentedColormap.from_list(
        "risk_overlay",
        [(1, 1, 1, 0), (1, 0.85, 0, 0.55), (0.85, 0, 0, 0.85)],
        N=256,
    )
    ax.imshow(
        risk_grid.T,
        cmap=cmap,
        alpha=alpha,
        extent=(0.0, lx, 0.0, ly),
        origin="lower",
        vmin=0.0, vmax=vmax,
        interpolation="bilinear",
        zorder=1,
    )


def _draw_exits(ax) -> None:
    """Mark the canonical 3 exits with green stars."""
    from src.integration.scenarios._common import exit_positions
    for ex in exit_positions():
        ax.plot(
            ex[0], ex[1], "*",
            color="#2ca02c",
            markersize=22,
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=6,
            label="_nolegend_",
        )


def _draw_drone_trails(
    ax,
    recorder: SimulationRecorder,
    line_alpha: float = 0.5,
    line_width: float = 1.0,
) -> None:
    """Draw each drone's trajectory + end marker (D-030)."""
    if not recorder.frames:
        return
    # Build per-drone trajectory by drone_id (first-seen order).
    drone_ids: List[str] = []
    seen: set = set()
    for fr in recorder.frames:
        for d in fr.drones:
            if d.drone_id not in seen:
                seen.add(d.drone_id)
                drone_ids.append(d.drone_id)
    for did in drone_ids:
        xs, ys, status_last = [], [], "searching"
        for fr in recorder.frames:
            for d in fr.drones:
                if d.drone_id == did:
                    xs.append(d.pos[0])
                    ys.append(d.pos[1])
                    status_last = d.status
                    break
        if not xs:
            continue
        # Trail (purple-ish, semi-transparent)
        ax.plot(
            xs, ys, "--",
            color=_DRONE_STATUS_COLOR.get("searching", "#888"),
            lw=line_width, alpha=line_alpha,
            zorder=2,
        )
        # End marker: triangle, coloured by final status
        ax.plot(
            xs[-1], ys[-1], "^",
            color=_DRONE_STATUS_COLOR.get(status_last, "#888"),
            markersize=11,
            markeredgecolor="black", markeredgewidth=0.9,
            zorder=5,
        )


def _draw_agent_trails(
    ax,
    recorder: SimulationRecorder,
    line_alpha: float = 0.6,
    line_width: float = 1.4,
) -> None:
    """Draw each agent's trajectory + start/end markers."""
    for aid in recorder.agent_ids():
        traj = recorder.agent_trajectory(aid)
        if traj.shape[0] == 0:
            continue
        status = recorder.final_status(aid)
        color = _STATUS_COLOR.get(status, _STATUS_COLOR["unknown"])
        marker = _STATUS_MARKER.get(status, "o")
        # Trail
        ax.plot(
            traj[:, 0], traj[:, 1], "-",
            color=color, lw=line_width, alpha=line_alpha,
            zorder=3,
        )
        # Start point (always black dot)
        ax.plot(
            traj[0, 0], traj[0, 1], "o",
            color="black", markersize=4, alpha=0.7, zorder=4,
        )
        # End point (status-coloured)
        ax.plot(
            traj[-1, 0], traj[-1, 1], marker,
            color=color, markersize=9,
            markeredgecolor="black", markeredgewidth=0.8,
            zorder=5,
        )


def _make_legend(ax) -> None:
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black",
               markersize=5, label="start"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor=_STATUS_COLOR["evacuated"], markeredgecolor="black",
               markersize=9, label="evacuated"),
        Line2D([0], [0], marker="X", linestyle="None",
               markerfacecolor=_STATUS_COLOR["alive"], markeredgecolor="black",
               markersize=9, label="alive (still inside)"),
        Line2D([0], [0], marker="^", linestyle="None",
               markerfacecolor=_DRONE_STATUS_COLOR["searching"],
               markeredgecolor="black",
               markersize=10, label="drone (searching)"),
        Line2D([0], [0], marker="^", linestyle="None",
               markerfacecolor=_DRONE_STATUS_COLOR["guiding"],
               markeredgecolor="black",
               markersize=10, label="drone (guiding)"),
        Line2D([0], [0], marker="*", linestyle="None",
               markerfacecolor=_STATUS_COLOR["evacuated"], markeredgecolor="black",
               markersize=13, label="exit"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)


def _style_axes(ax, title: str = "") -> None:
    lx, ly, _lz = DOMAIN_SIZE_M
    ax.set_xlim(-0.5, lx + 0.5)
    ax.set_ylim(-0.5, ly + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.3)


# ─── Public renderers ─────────────────────────────────────────────────────
def render_trajectories(
    recorder: SimulationRecorder,
    out_path: Path,
    *,
    fluid_mask: Optional[np.ndarray] = None,
    risk_map: Optional[object] = None,
    risk_at_t: Optional[float] = None,
    title: Optional[str] = None,
    dpi: int = 160,
) -> Path:
    """Top-down PNG of all agent trails over the recorded run.

    Args:
        recorder: Populated :class:`SimulationRecorder`.
        out_path: Destination PNG.
        fluid_mask: ``(60, 40, 6)`` boolean for the building footprint.
            Default loads the canonical mask.
        risk_map: Optional :class:`~src.risk_map.risk_map_class.RiskMap`
            for an overlay heatmap.
        risk_at_t: Time (s) at which to sample ``risk_map``. Defaults
            to the last frame's time when ``risk_map`` is given.
        title: Plot title.
        dpi: Output resolution.

    Returns:
        ``out_path``.
    """
    fig, ax = plt.subplots(figsize=(11.5, 7.5))

    _draw_building_footprint(ax, fluid_mask)
    if risk_map is not None:
        if risk_at_t is None and recorder.frames:
            risk_at_t = recorder.frames[-1].t
        if risk_at_t is not None:
            grid = _sample_risk_grid(risk_map, risk_at_t)
            _draw_risk_overlay(ax, grid)

    _draw_drone_trails(ax, recorder)
    _draw_agent_trails(ax, recorder)
    _draw_exits(ax)
    _make_legend(ax)

    if title is None:
        title = (
            f"{recorder.scenario_id}  "
            f"fire={recorder.fire_scenario_id}  "
            f"seed={recorder.seed}  "
            f"t=[{recorder.frames[0].t:.0f}, "
            f"{recorder.frames[-1].t:.0f}] s  "
            f"agents={len(recorder.agent_ids())}"
            if recorder.frames else recorder.scenario_id
        )
    _style_axes(ax, title=title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_snapshot(
    recorder: SimulationRecorder,
    frame_idx: int,
    out_path: Path,
    *,
    fluid_mask: Optional[np.ndarray] = None,
    risk_map: Optional[object] = None,
    show_trail_so_far: bool = True,
    title: Optional[str] = None,
    dpi: int = 160,
) -> Path:
    """Single-frame PNG at ``frame_idx`` (e.g. mid-evacuation snapshot)."""
    if not recorder.frames:
        raise ValueError("recorder has no frames")
    if not 0 <= frame_idx < len(recorder.frames):
        raise ValueError(
            f"frame_idx {frame_idx} outside [0, {len(recorder.frames)})"
        )
    frame = recorder.frames[frame_idx]

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    _draw_building_footprint(ax, fluid_mask)
    if risk_map is not None:
        grid = _sample_risk_grid(risk_map, frame.t)
        _draw_risk_overlay(ax, grid)
    elif frame.risk_grid is not None:
        _draw_risk_overlay(ax, frame.risk_grid)

    # Trails up to frame_idx (optional)
    if show_trail_so_far:
        for aid in recorder.agent_ids():
            traj = []
            for fr in recorder.frames[: frame_idx + 1]:
                for ag in fr.agents:
                    if ag.agent_id == aid:
                        traj.append(ag.pos)
                        break
            if not traj:
                continue
            arr = np.asarray(traj)
            status = frame.agents[0].status if frame.agents else "alive"
            # Determine this agent's status at this frame.
            for ag in frame.agents:
                if ag.agent_id == aid:
                    status = ag.status
                    break
            color = _STATUS_COLOR.get(status, _STATUS_COLOR["unknown"])
            ax.plot(arr[:, 0], arr[:, 1], "-", color=color, lw=1.2, alpha=0.55, zorder=3)

    # Current positions
    for ag in frame.agents:
        color = _STATUS_COLOR.get(ag.status, _STATUS_COLOR["unknown"])
        marker = _STATUS_MARKER.get(ag.status, "o")
        ax.plot(
            ag.pos[0], ag.pos[1], marker,
            color=color, markersize=10,
            markeredgecolor="black", markeredgewidth=0.8,
            zorder=5,
        )

    _draw_exits(ax)
    _make_legend(ax)
    if title is None:
        title = (
            f"{recorder.scenario_id}  fire={recorder.fire_scenario_id}  "
            f"t={frame.t:.0f} s  (frame {frame_idx + 1}/{len(recorder.frames)})"
        )
    _style_axes(ax, title=title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_comparison(
    recorders: Sequence[SimulationRecorder],
    out_path: Path,
    *,
    fluid_mask: Optional[np.ndarray] = None,
    risk_map: Optional[object] = None,
    risk_at_t: Optional[float] = None,
    suptitle: Optional[str] = None,
    dpi: int = 160,
) -> Path:
    """Side-by-side panels (e.g. S1 / S2 / S3 on one figure)."""
    if not recorders:
        raise ValueError("recorders must not be empty")
    n = len(recorders)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.5))
    if n == 1:
        axes = [axes]

    if risk_map is not None and risk_at_t is None:
        # Use the last common time across recorders for a consistent overlay.
        last_ts = [r.frames[-1].t for r in recorders if r.frames]
        if last_ts:
            risk_at_t = min(last_ts)
    risk_grid_cached: Optional[np.ndarray] = None
    if risk_map is not None and risk_at_t is not None:
        risk_grid_cached = _sample_risk_grid(risk_map, risk_at_t)

    for ax, rec in zip(axes, recorders):
        _draw_building_footprint(ax, fluid_mask)
        if risk_grid_cached is not None:
            _draw_risk_overlay(ax, risk_grid_cached)
        _draw_drone_trails(ax, rec)
        _draw_agent_trails(ax, rec)
        _draw_exits(ax)
        # Per-panel title with a one-line outcome summary.
        n_evac = sum(
            1 for aid in rec.agent_ids()
            if rec.final_status(aid) == "evacuated"
        )
        n_total = len(rec.agent_ids())
        title = (
            f"{rec.scenario_id}  "
            f"evac={n_evac}/{n_total}  "
            f"seed={rec.seed}"
        )
        _style_axes(ax, title=title)
    _make_legend(axes[-1])

    if suptitle is None and recorders[0].fire_scenario_id:
        suptitle = (
            f"fire={recorders[0].fire_scenario_id}  "
            f"risk overlay @ t={risk_at_t:.0f}s" if risk_grid_cached is not None
            else f"fire={recorders[0].fire_scenario_id}"
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, y=1.02)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_animation(
    recorder: SimulationRecorder,
    out_path: Path,
    *,
    fluid_mask: Optional[np.ndarray] = None,
    risk_map: Optional[object] = None,
    fps: int = 8,
    dpi: int = 100,
) -> Path:
    """Animated GIF of the recorder. Uses ``matplotlib.PillowWriter``.

    Args:
        recorder: Populated recorder.
        out_path: ``.gif`` (or other matplotlib-supported writer).
        fluid_mask: Optional building footprint mask.
        risk_map: Optional :class:`RiskMap` for per-frame heatmap; if
            ``recorder.capture_risk_grid`` is True the cached grid is
            used and ``risk_map`` is ignored.
        fps: Frames per second.
        dpi: Output DPI (smaller = faster + smaller file).
    """
    if not recorder.frames:
        raise ValueError("recorder has no frames")

    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    _draw_building_footprint(ax, fluid_mask)
    _draw_exits(ax)
    _style_axes(ax, title="")

    # Persistent agent markers we'll update per frame.
    agent_ids = recorder.agent_ids()
    markers = {}
    trails = {}
    for aid in agent_ids:
        (m,) = ax.plot([], [], "o", markersize=8, color=_STATUS_COLOR["alive"],
                       markeredgecolor="black", markeredgewidth=0.6, zorder=5)
        (l,) = ax.plot([], [], "-", color=_STATUS_COLOR["alive"], lw=1.2, alpha=0.5, zorder=3)
        markers[aid] = m
        trails[aid] = l

    # Persistent drone markers (D-030).
    drone_ids = sorted({d.drone_id for fr in recorder.frames for d in fr.drones})
    drone_markers: dict = {}
    drone_trails: dict = {}
    for did in drone_ids:
        (dm,) = ax.plot([], [], "^",
                        markersize=11,
                        color=_DRONE_STATUS_COLOR["searching"],
                        markeredgecolor="black", markeredgewidth=0.8,
                        zorder=5)
        (dl,) = ax.plot([], [], "--",
                        color=_DRONE_STATUS_COLOR["searching"],
                        lw=1.0, alpha=0.45, zorder=2)
        drone_markers[did] = dm
        drone_trails[did] = dl

    # Risk overlay artist (may be replaced each frame if heatmap dynamic).
    overlay_imgs: List = []

    def update(i: int):
        frame = recorder.frames[i]
        ax.set_title(
            f"{recorder.scenario_id}  fire={recorder.fire_scenario_id}  "
            f"t={frame.t:.1f}s  ({i + 1}/{len(recorder.frames)})",
            fontsize=11,
        )
        # Risk overlay (clear previous)
        for img in overlay_imgs:
            img.remove()
        overlay_imgs.clear()
        grid = frame.risk_grid
        if grid is None and risk_map is not None:
            grid = _sample_risk_grid(risk_map, frame.t)
        if grid is not None:
            lx, ly, _lz = DOMAIN_SIZE_M
            cmap = LinearSegmentedColormap.from_list(
                "risk_overlay",
                [(1, 1, 1, 0), (1, 0.85, 0, 0.55), (0.85, 0, 0, 0.85)],
                N=256,
            )
            img = ax.imshow(
                grid.T, cmap=cmap, alpha=0.55,
                extent=(0.0, lx, 0.0, ly), origin="lower",
                vmin=0.0, vmax=1.0, zorder=1,
            )
            overlay_imgs.append(img)
        # Agent markers + trails
        for ag in frame.agents:
            color = _STATUS_COLOR.get(ag.status, _STATUS_COLOR["unknown"])
            markers[ag.agent_id].set_data([ag.pos[0]], [ag.pos[1]])
            markers[ag.agent_id].set_color(color)
            # Update trail
            xs, ys = [], []
            for fr in recorder.frames[: i + 1]:
                for a2 in fr.agents:
                    if a2.agent_id == ag.agent_id:
                        xs.append(a2.pos[0])
                        ys.append(a2.pos[1])
                        break
            trails[ag.agent_id].set_data(xs, ys)
            trails[ag.agent_id].set_color(color)
        # Drone markers + trails
        for d in frame.drones:
            d_color = _DRONE_STATUS_COLOR.get(d.status, _DRONE_STATUS_COLOR["searching"])
            dm = drone_markers.get(d.drone_id)
            if dm is not None:
                dm.set_data([d.pos[0]], [d.pos[1]])
                dm.set_color(d_color)
            dxs, dys = [], []
            for fr in recorder.frames[: i + 1]:
                for d2 in fr.drones:
                    if d2.drone_id == d.drone_id:
                        dxs.append(d2.pos[0])
                        dys.append(d2.pos[1])
                        break
            dl = drone_trails.get(d.drone_id)
            if dl is not None:
                dl.set_data(dxs, dys)
                dl.set_color(d_color)
        artists = (
            list(markers.values()) + list(trails.values()) + overlay_imgs
            + list(drone_markers.values()) + list(drone_trails.values())
        )
        return artists

    ani = FuncAnimation(
        fig, update, frames=len(recorder.frames),
        interval=1000 / max(fps, 1), blit=False,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


def render_comparison_animation(
    recorders: Sequence[SimulationRecorder],
    out_path: Path,
    *,
    fluid_mask: Optional[np.ndarray] = None,
    risk_map: Optional[object] = None,
    fps: int = 8,
    dpi: int = 90,
    dt_s: float = 1.0,
) -> Path:
    """Side-by-side animated GIF: one panel per recorder, common timeline.

    Frames are sampled on a uniform time grid spanning
    ``[min(rec.frames[0].t), max(rec.frames[-1].t)]``. For each panel the
    closest recorder frame to the timeline tick is rendered; scenarios
    that finished early visually hold on their final state while others
    continue (so the H6 trade-off "S2 finished, S1 still struggling"
    reads instantly).

    Args:
        recorders: Two or more :class:`SimulationRecorder` instances.
            One panel per recorder, in order.
        out_path: Destination ``.gif``.
        fluid_mask: Optional ``(60, 40, 6)`` boolean for the building.
        risk_map: Optional :class:`RiskMap` -- when supplied, every
            frame's overlay is sampled at the panel's effective time
            so all panels share the same risk evolution.
        fps: GIF framerate.
        dpi: Output DPI.
        dt_s: Timeline step (seconds).
    """
    if not recorders:
        raise ValueError("recorders must not be empty")
    for rec in recorders:
        if not rec.frames:
            raise ValueError(
                f"recorder {rec.scenario_id!r} has no frames"
            )

    from matplotlib.animation import FuncAnimation, PillowWriter

    t_min = min(rec.frames[0].t for rec in recorders)
    t_max = max(rec.frames[-1].t for rec in recorders)
    times = np.arange(t_min, t_max + dt_s * 0.5, dt_s)

    n = len(recorders)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.0))
    if n == 1:
        axes = [axes]

    # Static layers: mask + exits + axes style (once per panel).
    for ax in axes:
        _draw_building_footprint(ax, fluid_mask)
        _draw_exits(ax)
        _style_axes(ax, title="")

    # Per-panel persistent artists.
    panel_state = []
    for rec, ax in zip(recorders, axes):
        markers: dict = {}
        trails: dict = {}
        for aid in rec.agent_ids():
            (m,) = ax.plot([], [], "o", markersize=8,
                           color=_STATUS_COLOR["alive"],
                           markeredgecolor="black", markeredgewidth=0.6, zorder=5)
            (l,) = ax.plot([], [], "-", color=_STATUS_COLOR["alive"],
                           lw=1.2, alpha=0.5, zorder=3)
            markers[aid] = m
            trails[aid] = l
        # Drone markers per panel
        drone_ids = sorted({d.drone_id for fr in rec.frames for d in fr.drones})
        d_markers: dict = {}
        d_trails: dict = {}
        for did in drone_ids:
            (dm,) = ax.plot([], [], "^", markersize=10,
                            color=_DRONE_STATUS_COLOR["searching"],
                            markeredgecolor="black", markeredgewidth=0.7,
                            zorder=5)
            (dl,) = ax.plot([], [], "--",
                            color=_DRONE_STATUS_COLOR["searching"],
                            lw=0.9, alpha=0.45, zorder=2)
            d_markers[did] = dm
            d_trails[did] = dl
        panel_state.append({
            "rec": rec,
            "ax": ax,
            "markers": markers,
            "trails": trails,
            "d_markers": d_markers,
            "d_trails": d_trails,
            "overlay": [],
            # Precomputed trajectory and per-frame index lookup.
            "frame_t": np.asarray([fr.t for fr in rec.frames]),
        })

    def _nearest_frame_idx(frame_ts: np.ndarray, t: float) -> int:
        # frame_ts is monotonically increasing.
        i = int(np.searchsorted(frame_ts, t))
        if i >= len(frame_ts):
            return len(frame_ts) - 1
        if i == 0:
            return 0
        # pick the closer of i-1 and i
        if abs(frame_ts[i] - t) < abs(frame_ts[i - 1] - t):
            return i
        return i - 1

    cmap = LinearSegmentedColormap.from_list(
        "risk_overlay",
        [(1, 1, 1, 0), (1, 0.85, 0, 0.55), (0.85, 0, 0, 0.85)],
        N=256,
    )

    def update(i: int):
        t = float(times[i])
        artists = []
        # Shared risk grid (sampled once per timeline tick if risk_map given).
        shared_grid = None
        if risk_map is not None:
            shared_grid = _sample_risk_grid(risk_map, t)
        for st in panel_state:
            rec = st["rec"]
            ax = st["ax"]
            idx = _nearest_frame_idx(st["frame_t"], t)
            frame = rec.frames[idx]
            # Risk overlay -- recorder-cached preferred, else shared sample.
            for img in st["overlay"]:
                img.remove()
            st["overlay"].clear()
            grid = frame.risk_grid if frame.risk_grid is not None else shared_grid
            if grid is not None:
                lx, ly, _lz = DOMAIN_SIZE_M
                img = ax.imshow(
                    grid.T, cmap=cmap, alpha=0.55,
                    extent=(0.0, lx, 0.0, ly), origin="lower",
                    vmin=0.0, vmax=1.0, zorder=1,
                )
                st["overlay"].append(img)
                artists.append(img)
            # Title with this panel's effective frame.
            n_evac_so_far = sum(
                1 for ag in frame.agents if ag.arrived or ag.status == "evacuated"
            )
            n_total = len(rec.agent_ids())
            ax.set_title(
                f"{rec.scenario_id}  t={frame.t:.0f}s  "
                f"evac={n_evac_so_far}/{n_total}",
                fontsize=10,
            )
            # Agents
            for ag in frame.agents:
                color = _STATUS_COLOR.get(ag.status, _STATUS_COLOR["unknown"])
                m = st["markers"].get(ag.agent_id)
                if m is not None:
                    m.set_data([ag.pos[0]], [ag.pos[1]])
                    m.set_color(color)
                    artists.append(m)
                # Trail (up to idx)
                xs, ys = [], []
                for fr2 in rec.frames[: idx + 1]:
                    for a2 in fr2.agents:
                        if a2.agent_id == ag.agent_id:
                            xs.append(a2.pos[0])
                            ys.append(a2.pos[1])
                            break
                l = st["trails"].get(ag.agent_id)
                if l is not None:
                    l.set_data(xs, ys)
                    l.set_color(color)
                    artists.append(l)
            # Drones (D-030)
            for d in frame.drones:
                d_color = _DRONE_STATUS_COLOR.get(
                    d.status, _DRONE_STATUS_COLOR["searching"]
                )
                dm = st["d_markers"].get(d.drone_id)
                if dm is not None:
                    dm.set_data([d.pos[0]], [d.pos[1]])
                    dm.set_color(d_color)
                    artists.append(dm)
                dxs, dys = [], []
                for fr2 in rec.frames[: idx + 1]:
                    for d2 in fr2.drones:
                        if d2.drone_id == d.drone_id:
                            dxs.append(d2.pos[0])
                            dys.append(d2.pos[1])
                            break
                dl = st["d_trails"].get(d.drone_id)
                if dl is not None:
                    dl.set_data(dxs, dys)
                    dl.set_color(d_color)
                    artists.append(dl)
        fig.suptitle(
            f"timeline t={t:.0f}s  (tick {i + 1}/{len(times)})",
            fontsize=12, y=1.005,
        )
        return artists

    ani = FuncAnimation(
        fig, update, frames=len(times),
        interval=1000 / max(fps, 1), blit=False,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


# ─── Risk-grid sampler (avoid circular import in recorder) ────────────────
def _sample_risk_grid(risk_map: object, t: float) -> np.ndarray:
    nx_, ny_, _ = GRID_SHAPE
    z = 0.25 + CELL_SIZE_M * 3  # k=3 breathing zone
    xs = 0.25 + CELL_SIZE_M * np.arange(nx_)
    ys = 0.25 + CELL_SIZE_M * np.arange(ny_)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([xx, yy, np.full_like(xx, z)], axis=-1).reshape(-1, 3)
    try:
        vals = np.asarray(risk_map.query(pts, t=t), dtype=np.float32)
    except Exception:  # noqa: BLE001
        vals = np.empty(pts.shape[0], dtype=np.float32)
        for k, p in enumerate(pts):
            vals[k] = float(risk_map.query(p, t=t))
    return vals.reshape(nx_, ny_)


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import tempfile
    from types import SimpleNamespace

    print("=" * 60)
    print("renderer.py self-test")
    print("=" * 60)

    errors: list[str] = []

    # Fabricate a small recorder with 2 agents, 10 frames.
    rec = SimulationRecorder(scenario_id="S_test", fire_scenario_id="fire_x", seed=42)
    for i in range(10):
        ags = [
            SimpleNamespace(
                agent_id="a",
                position=np.array([1.0 + 0.2 * i, 5.0, 0.85]),
            ),
            SimpleNamespace(
                agent_id="b",
                position=np.array([20.0 - 0.3 * i, 10.0 + 0.1 * i, 0.85]),
            ),
        ]
        rec.record(t=float(i), agents=ags,
                   arrived={"b"} if i >= 8 else None)

    # ── 1. render_trajectories produces a PNG ────────────────────
    print("\n[1] render_trajectories writes a PNG")
    with tempfile.TemporaryDirectory() as td:
        out = render_trajectories(
            rec, Path(td) / "traj.png",
        )
        size_kb = out.stat().st_size / 1024
        print(f"  wrote {out.name}  ({size_kb:.1f} KB)")
        if size_kb < 5:
            errors.append("PNG suspiciously small")

    # ── 2. render_snapshot ──────────────────────────────────────
    print("\n[2] render_snapshot at mid frame")
    with tempfile.TemporaryDirectory() as td:
        out = render_snapshot(rec, frame_idx=5, out_path=Path(td) / "snap.png")
        if not out.exists():
            errors.append("snapshot PNG missing")
        else:
            print(f"  PASS: wrote {out.name}")

    # ── 3. render_snapshot rejects bad index ────────────────────
    print("\n[3] render_snapshot rejects bad index")
    with tempfile.TemporaryDirectory() as td:
        try:
            render_snapshot(rec, frame_idx=999, out_path=Path(td) / "x.png")
        except ValueError:
            print("  PASS: out-of-range -> ValueError")
        else:
            errors.append("bad index did not raise")

    # ── 4. render_comparison (single panel = 1 recorder) ─────────
    print("\n[4] render_comparison with 1 recorder")
    with tempfile.TemporaryDirectory() as td:
        out = render_comparison([rec], Path(td) / "cmp.png")
        if not out.exists():
            errors.append("comparison PNG missing")
        else:
            print(f"  PASS: wrote {out.name}")

    # ── 5. render_comparison with 3 recorders ────────────────────
    print("\n[5] render_comparison with 3 recorders")
    rec2 = SimulationRecorder(scenario_id="S2_test", fire_scenario_id="fire_x", seed=42)
    rec3 = SimulationRecorder(scenario_id="S3_test", fire_scenario_id="fire_x", seed=42)
    for r in (rec2, rec3):
        for i in range(10):
            ags = [SimpleNamespace(agent_id="a", position=np.array([5.0, 5.0, 0.85]))]
            r.record(t=float(i), agents=ags)
    with tempfile.TemporaryDirectory() as td:
        out = render_comparison([rec, rec2, rec3], Path(td) / "cmp3.png")
        if not out.exists():
            errors.append("3-panel comparison PNG missing")
        else:
            print(f"  PASS: wrote {out.name}")

    # ── 6. render_animation produces a GIF ───────────────────────
    print("\n[6] render_animation produces a small GIF")
    with tempfile.TemporaryDirectory() as td:
        out = render_animation(
            rec, Path(td) / "anim.gif", fps=5, dpi=60,
        )
        if not out.exists():
            errors.append("animation GIF missing")
        else:
            size_kb = out.stat().st_size / 1024
            print(f"  PASS: wrote {out.name} ({size_kb:.1f} KB)")

    # ── 7. render_comparison_animation 3-panel GIF ────────────────
    print("\n[7] render_comparison_animation 3-panel side-by-side GIF")
    with tempfile.TemporaryDirectory() as td:
        out = render_comparison_animation(
            [rec, rec2, rec3], Path(td) / "cmp_anim.gif", fps=5, dpi=60,
        )
        if not out.exists():
            errors.append("comparison animation GIF missing")
        else:
            size_kb = out.stat().st_size / 1024
            print(f"  PASS: wrote {out.name} ({size_kb:.1f} KB)")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: renderer validated")
