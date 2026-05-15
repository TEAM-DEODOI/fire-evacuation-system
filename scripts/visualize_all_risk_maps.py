"""Batch-render a fire-spread GIF for every cached FDS scenario.

Outputs one GIF per scenario at ``figures/risk_maps/<scenario>.gif``.
Each frame is the breathing-zone (z=3, 1.75 m) risk-map slice at a single
sample time, overlaid on the building footprint (interior mask) — the
purpose is purely visual triage: which scenarios look interesting for
the EXP-PATH-001 study (asymmetric fires? quickly spreading? early
saturation? slow burns?).

Compute budget: each scenario has 31 frames (t = 0 .. 300 s, Δt = 10 s).
We allow downsampling via ``--frame-step`` to cut GIF size and rendering
time (default 1 → keep every frame). With ``--fps 3`` and
``--frame-step 1`` a GIF runs for ~10 s of wall time and ~10 s of
playback per scenario.

Run::

    python scripts/visualize_all_risk_maps.py
    python scripts/visualize_all_risk_maps.py --frame-step 2 --fps 3
    python scripts/visualize_all_risk_maps.py --only sim_1500kw_2m2_T05 s_028
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

from src.path_planning.building_graph import load_default_fluid_mask
from src.risk_map.risk_map_class import StaticRiskMap
from src.shared.constants import CELL_SIZE_M, DOMAIN_SIZE_M


_CACHE_DIR = Path("results/cache/scenario_risk_maps")
_OUT_DIR = Path("figures/risk_maps")


def _build_risk_cmap():
    return LinearSegmentedColormap.from_list(
        "risk_overlay",
        [(1, 1, 1, 0), (1, 0.85, 0, 0.55), (0.85, 0, 0, 0.85)],
        N=256,
    )


def _render_one(
    npz_path: Path,
    out_path: Path,
    *,
    fluid_mask: np.ndarray,
    fps: int,
    frame_step: int,
    dpi: int,
    z_layer: int = 3,
) -> None:
    """Render a single scenario's risk-map evolution as a GIF."""
    rm = StaticRiskMap.from_npy(npz_path)
    danger = rm.danger  # (T, 60, 40, 6)
    times = rm.times
    n_frames_total = danger.shape[0]
    sel = list(range(0, n_frames_total, max(1, int(frame_step))))
    # Always include the last frame so the GIF ends at the worst-case spread.
    if sel[-1] != n_frames_total - 1:
        sel.append(n_frames_total - 1)

    lx, ly, _lz = DOMAIN_SIZE_M
    walls = 1.0 - fluid_mask[:, :, z_layer].astype(np.float32)
    cmap = _build_risk_cmap()

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.imshow(
        walls.T,
        cmap="Greys",
        alpha=0.25,
        extent=(0.0, lx, 0.0, ly),
        origin="lower",
        interpolation="nearest",
        zorder=0,
    )
    risk_img = ax.imshow(
        danger[0, :, :, z_layer].T,
        cmap=cmap,
        alpha=0.85,
        extent=(0.0, lx, 0.0, ly),
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        zorder=1,
    )
    ax.set_xlim(-0.5, lx + 0.5)
    ax.set_ylim(-0.5, ly + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle=":", alpha=0.3)
    title = ax.set_title("", fontsize=10)
    plt.colorbar(risk_img, ax=ax, fraction=0.046, pad=0.04, label="danger")

    scenario_name = npz_path.stem

    def update(i: int):
        idx = sel[i]
        risk_img.set_data(danger[idx, :, :, z_layer].T)
        title.set_text(
            f"{scenario_name}  t={float(times[idx]):.0f}s  "
            f"(frame {idx + 1}/{n_frames_total})"
        )
        return [risk_img, title]

    ani = FuncAnimation(
        fig, update, frames=len(sel),
        interval=1000 / max(fps, 1), blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, default=_CACHE_DIR)
    ap.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    ap.add_argument("--fps", type=int, default=3,
                    help="Playback rate (frames per second).")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="Sample every Nth recorded frame (default 1 = all).")
    ap.add_argument("--dpi", type=int, default=90,
                    help="GIF resolution (lower = smaller file, faster).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Render only these scenario stems (omit ext).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip scenarios whose GIF already exists.")
    args = ap.parse_args()

    if not args.cache_dir.exists():
        print(f"FAIL: {args.cache_dir} missing", file=sys.stderr)
        return 1
    fluid_mask = load_default_fluid_mask()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cached: List[Path] = sorted(args.cache_dir.glob("*.npz"))
    if args.only:
        wanted = set(args.only)
        cached = [p for p in cached if p.stem in wanted]
    if not cached:
        print(f"No scenarios match.", file=sys.stderr)
        return 1

    print(
        f"Rendering {len(cached)} scenario(s) -> {args.out_dir}/<scenario>.gif "
        f"(fps={args.fps}, frame_step={args.frame_step}, dpi={args.dpi})"
    )

    t_start = time.perf_counter()
    for i, npz in enumerate(cached, start=1):
        out = args.out_dir / f"{npz.stem}.gif"
        if args.skip_existing and out.exists():
            print(f"  [{i:>2}/{len(cached)}] {npz.stem}  SKIP (exists)")
            continue
        t0 = time.perf_counter()
        try:
            _render_one(
                npz, out,
                fluid_mask=fluid_mask,
                fps=int(args.fps),
                frame_step=int(args.frame_step),
                dpi=int(args.dpi),
            )
            dt = time.perf_counter() - t0
            size_kb = out.stat().st_size / 1024
            print(
                f"  [{i:>2}/{len(cached)}] {npz.stem:<28}  "
                f"{size_kb:>5.0f} KB  ({dt:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:>2}/{len(cached)}] {npz.stem}  FAIL ({exc})")

    total_dt = time.perf_counter() - t_start
    print(f"\nTotal: {total_dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
