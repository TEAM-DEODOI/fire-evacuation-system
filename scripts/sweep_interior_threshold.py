"""Sweep the danger threshold over a single FDS scenario to find the
clean indoor mask.

Approach (user 2026-05-14):
    "RiskMap shows smoke distribution. Outdoor regions stay at zero
    risk because smoke disperses to atmosphere; indoor regions
    accumulate risk. So at the final frame, cells with risk > thr
    are indoor."

This script materialises that intuition by sweeping ``thr`` and rendering
side-by-side previews so the user can pick the sharpest separator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.integration.scenarios._common import load_truth_risk_map
from src.shared.constants import CELL_SIZE_M, DOMAIN_SIZE_M, GRID_SHAPE


_DEFAULT_FIRE = "sim_1500kw_2m2_T05"
_DEFAULT_Z = 3
_DEFAULT_T = 290.0
_OUT_PATH = Path("figures/interior_mask/threshold_sweep.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fire", type=str, default=_DEFAULT_FIRE,
                    help="FDS scenario folder under data/raw/.")
    ap.add_argument("--t", type=float, default=_DEFAULT_T,
                    help="Time (s) at which to sample the risk map.")
    ap.add_argument("--z-layer", type=int, default=_DEFAULT_Z,
                    help="Z-slice to use.")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.30, 0.50, 0.70])
    ap.add_argument("--out", type=Path, default=_OUT_PATH)
    args = ap.parse_args()

    fds_dir = Path("data/raw") / args.fire
    if not fds_dir.exists():
        print(f"FAIL: {fds_dir} missing", file=sys.stderr)
        return 1

    print(f"loading {args.fire} ...")
    rm = load_truth_risk_map(fds_dir, verbose=True)
    # Sample at the chosen z-slice across the full (60, 40) grid.
    nx_, ny_, _ = GRID_SHAPE
    xs = 0.25 + CELL_SIZE_M * np.arange(nx_)
    ys = 0.25 + CELL_SIZE_M * np.arange(ny_)
    z = 0.25 + CELL_SIZE_M * args.z_layer
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([xx, yy, np.full_like(xx, z)], axis=-1).reshape(-1, 3)
    vals = np.asarray(rm.query(pts, t=args.t), dtype=np.float32).reshape(nx_, ny_)
    print(
        f"  shape={vals.shape}  "
        f"min={vals.min():.3f}  max={vals.max():.3f}  "
        f"mean={vals.mean():.3f}"
    )

    # Build a panel for each threshold.
    thrs: List[float] = list(args.thresholds)
    n = len(thrs) + 1  # +1 for raw heatmap
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flat

    lx, ly, _ = DOMAIN_SIZE_M

    # Panel 0: continuous heatmap
    ax = axes[0]
    im = ax.imshow(
        vals.T, extent=(0.0, lx, 0.0, ly), origin="lower",
        cmap="hot_r", vmin=0.0, vmax=1.0, interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(
        f"continuous risk  fire={args.fire}\nt={args.t:.0f}s  z={args.z_layer}"
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # Panels 1..: binary mask at each threshold
    for k, thr in enumerate(thrs, start=1):
        mask = (vals > thr) & (vals < 1.0)  # exclude OOB sentinel
        ax = axes[k]
        ax.imshow(
            mask.T, extent=(0.0, lx, 0.0, ly), origin="lower",
            cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest",
        )
        ax.set_title(
            f"thr > {thr:.2f}  cells={int(mask.sum())}"
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # Hide spare panels
    for j in range(len(thrs) + 1, len(axes)):
        axes[j].axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
