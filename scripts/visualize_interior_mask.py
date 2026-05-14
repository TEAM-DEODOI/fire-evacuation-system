"""Visualise the interior mask + emit an editable JSON for manual touch-ups.

Outputs (under ``figures/interior_mask/``):

* ``interior_mask_z3.png``     -- breathing-zone (k=3) slice with cell labels
* ``interior_mask_all_z.png``  -- 6-panel grid, one per z-layer
* ``interior_mask_overlay.png``-- interior on top of fluid mask
                                   (shows which fluid cells are outdoor)
* ``interior_mask_z3.json``    -- editable cell list (i, j) for manual edit

The PNG is high-res and labels every 5th cell so user can identify the
exact ``(i, j)`` of any cell they want to flip. Edit the JSON manually,
then run ``scripts/apply_interior_mask_edits.py`` to rebuild the npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.path_planning.building_graph import (
    load_default_fluid_mask,
    load_interior_mask,
)
from src.shared.constants import CELL_SIZE_M, DOMAIN_SIZE_M, GRID_SHAPE


_OUT_DIR = Path("figures/interior_mask")


def _draw_grid_with_cells(
    ax,
    mask: np.ndarray,
    *,
    title: str,
    label_every: int = 5,
) -> None:
    """Draw a (60, 40) mask with cell-index labels every ``label_every`` cells."""
    nx_, ny_ = mask.shape
    # Light grey for non-interior, blue for interior.
    img = np.zeros((nx_, ny_, 4), dtype=np.float32)
    img[..., :3] = np.where(mask[..., None],
                            np.array([0.30, 0.50, 0.85]),  # interior blue
                            np.array([0.92, 0.92, 0.92]))  # outdoor grey
    img[..., 3] = 1.0
    lx, ly, _ = DOMAIN_SIZE_M
    ax.imshow(
        np.transpose(img, (1, 0, 2)),
        extent=(0.0, lx, 0.0, ly),
        origin="lower",
        interpolation="nearest",
    )
    # Cell grid lines (every cell, very thin) + every-5 labels.
    for i in range(nx_ + 1):
        ax.axvline(i * CELL_SIZE_M, color="black", lw=0.18, alpha=0.18)
    for j in range(ny_ + 1):
        ax.axhline(j * CELL_SIZE_M, color="black", lw=0.18, alpha=0.18)
    # Bigger label every 5.
    for i in range(0, nx_, label_every):
        ax.text(
            i * CELL_SIZE_M + 0.25, -0.4, f"{i}",
            ha="center", va="top", fontsize=6, color="black",
        )
    for j in range(0, ny_, label_every):
        ax.text(
            -0.4, j * CELL_SIZE_M + 0.25, f"{j}",
            ha="right", va="center", fontsize=6, color="black",
        )
    # Cell coords inside each cell (only show on small subsets to avoid clutter).
    for i in range(0, nx_, label_every):
        for j in range(0, ny_, label_every):
            ax.plot(
                i * CELL_SIZE_M + 0.25,
                j * CELL_SIZE_M + 0.25,
                "+",
                color="black", markersize=4, alpha=0.6,
            )
    ax.set_xlim(-1.0, lx + 0.5)
    ax.set_ylim(-1.0, ly + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m) -- cell index = floor(x / 0.5)")
    ax.set_ylabel("Y (m) -- cell index = floor(y / 0.5)")
    ax.set_title(title, fontsize=11)


def _draw_overlay(
    ax,
    interior: np.ndarray,
    fluid: np.ndarray,
) -> None:
    """Show outdoor-fluid cells (red), interior (blue), wall (grey)."""
    nx_, ny_ = interior.shape
    img = np.zeros((nx_, ny_, 4), dtype=np.float32)
    for i in range(nx_):
        for j in range(ny_):
            if interior[i, j]:
                img[i, j] = (0.30, 0.50, 0.85, 1.0)        # interior blue
            elif fluid[i, j]:
                img[i, j] = (0.95, 0.30, 0.30, 1.0)        # outdoor red
            else:
                img[i, j] = (0.35, 0.35, 0.35, 1.0)        # wall grey
    lx, ly, _ = DOMAIN_SIZE_M
    ax.imshow(
        np.transpose(img, (1, 0, 2)),
        extent=(0.0, lx, 0.0, ly),
        origin="lower",
        interpolation="nearest",
    )
    for i in range(nx_ + 1):
        ax.axvline(i * CELL_SIZE_M, color="black", lw=0.15, alpha=0.18)
    for j in range(ny_ + 1):
        ax.axhline(j * CELL_SIZE_M, color="black", lw=0.15, alpha=0.18)
    for i in range(0, nx_, 5):
        ax.text(i * CELL_SIZE_M + 0.25, -0.4, f"{i}",
                ha="center", va="top", fontsize=6)
    for j in range(0, ny_, 5):
        ax.text(-0.4, j * CELL_SIZE_M + 0.25, f"{j}",
                ha="right", va="center", fontsize=6)
    ax.set_xlim(-1.0, lx + 0.5)
    ax.set_ylim(-1.0, ly + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        "Legend: BLUE = interior (spawnable), RED = fluid-but-outdoor "
        "(should NOT spawn), GREY = wall/solid",
        fontsize=10,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--z-layer", type=int, default=3,
                    help="Breathing-zone slice (default 3 = 1.75 m)")
    ap.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    args = ap.parse_args()

    interior = load_interior_mask()
    fluid = load_default_fluid_mask()
    nx_, ny_, nz_ = GRID_SHAPE

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Breathing-zone single panel ──────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 10))
    _draw_grid_with_cells(
        ax, interior[:, :, args.z_layer],
        title=(
            f"interior_mask  z_layer={args.z_layer}  "
            f"(z={0.25 + CELL_SIZE_M * args.z_layer:.2f} m)  "
            f"cells={int(interior[:,:,args.z_layer].sum())}  "
            f"(BLUE = spawnable)"
        ),
    )
    fig.tight_layout()
    p = args.out_dir / f"interior_mask_z{args.z_layer}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")

    # ── 2. All-z panel grid ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for k in range(nz_):
        ax = axes.flat[k]
        _draw_grid_with_cells(
            ax, interior[:, :, k],
            title=f"z={k}  (z={0.25 + CELL_SIZE_M * k:.2f} m)  "
                  f"cells={int(interior[:,:,k].sum())}",
            label_every=10,
        )
    fig.tight_layout()
    p = args.out_dir / "interior_mask_all_z.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")

    # ── 3. Overlay (interior vs fluid) at z_layer ──────────────────
    fig, ax = plt.subplots(figsize=(16, 10))
    _draw_overlay(ax, interior[:, :, args.z_layer], fluid[:, :, args.z_layer])
    fig.tight_layout()
    p = args.out_dir / "interior_mask_overlay.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")

    # ── 4. Editable JSON ───────────────────────────────────────────
    # List of cells that are NOT interior but are fluid (current "outdoor")
    # vs. cells that ARE interior (current "indoor"). User can move (i, j)
    # tuples between the two lists.
    layer_i = interior[:, :, args.z_layer]
    layer_f = fluid[:, :, args.z_layer]
    indoor_cells = [(int(i), int(j))
                    for i in range(nx_) for j in range(ny_)
                    if layer_i[i, j]]
    outdoor_fluid_cells = [(int(i), int(j))
                           for i in range(nx_) for j in range(ny_)
                           if (layer_f[i, j] and not layer_i[i, j])]
    p = args.out_dir / f"interior_mask_z{args.z_layer}.json"
    p.write_text(json.dumps({
        "z_layer": args.z_layer,
        "grid_shape_xy": [nx_, ny_],
        "cell_size_m": CELL_SIZE_M,
        "interior_cells": indoor_cells,
        "outdoor_fluid_cells": outdoor_fluid_cells,
        "_help": (
            "Move (i, j) tuples between 'interior_cells' (spawnable) and "
            "'outdoor_fluid_cells' (NOT spawnable). Cells in NEITHER list "
            "are walls (kept as-is). After editing, run: "
            "python scripts/apply_interior_mask_edits.py "
            f"{p.name}"
        ),
    }, indent=2))
    print(f"wrote {p}  (editable)")

    print()
    print(f"Layer-{args.z_layer} summary:")
    print(f"  interior cells:       {int(layer_i.sum())}")
    print(f"  fluid cells:          {int(layer_f.sum())}")
    print(f"  outdoor-fluid cells:  {int((layer_f & ~layer_i).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
