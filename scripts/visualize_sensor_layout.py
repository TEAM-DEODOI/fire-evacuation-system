"""D-024 27 감지기 + 평면도 시각화.

사용자가 보여준 navigation graph figure 스타일로 27 감지기 + 3 출구 + 벽 표시.

생성 figure:
* figures/sensor_layout/sensor_layout.png
* figures/sensor_layout/sensor_layout_annotated.png  (감지기 ID 표시)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from src.shared.coordinates import cell_centres
from src.shared.constants import GRID_SHAPE, CELL_SIZE_M
from src.tier1.detector_positions import ALL_DETECTORS


# Z slice for visualization (~1.75 m breathing height)
Z_SLICE_M = 1.75


# ─── Wall segment extraction ───────────────────────────────────────────────
def extract_wall_segments(
    mask_2d: np.ndarray,
    origin: Tuple[float, float] = (0.0, 0.0),
    cell_size: float = CELL_SIZE_M,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Extract wall segments between fluid (1) and solid (0) cells.

    Args:
        mask_2d: (nx, ny) float/bool — 1.0 fluid, 0.0 solid.
        origin: world (x0, y0) of cell (0, 0) lower-left corner.
        cell_size: m.

    Returns:
        List of ((x1, y1), (x2, y2)) segments in world metres.
    """
    nx, ny = mask_2d.shape
    fluid = mask_2d > 0.5
    x0, y0 = origin
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    # Vertical walls (between i and i+1)
    for i in range(nx - 1):
        for j in range(ny):
            if fluid[i, j] != fluid[i + 1, j]:
                # wall at x = (i+1)*cs, between y=j*cs and y=(j+1)*cs
                x_w = x0 + (i + 1) * cell_size
                y_l = y0 + j * cell_size
                y_h = y0 + (j + 1) * cell_size
                segments.append(((x_w, y_l), (x_w, y_h)))

    # Horizontal walls (between j and j+1)
    for i in range(nx):
        for j in range(ny - 1):
            if fluid[i, j] != fluid[i, j + 1]:
                x_l = x0 + i * cell_size
                x_h = x0 + (i + 1) * cell_size
                y_w = y0 + (j + 1) * cell_size
                segments.append(((x_l, y_w), (x_h, y_w)))

    # Domain boundary — also draw if cell at edge is fluid (interior side)
    # We draw boundary as wall whether or not edge cell is fluid (to bound
    # the visualization), but only one segment per cell length.
    for i in range(nx):
        if fluid[i, 0]:
            x_l = x0 + i * cell_size
            x_h = x0 + (i + 1) * cell_size
            segments.append(((x_l, y0), (x_h, y0)))
        if fluid[i, ny - 1]:
            x_l = x0 + i * cell_size
            x_h = x0 + (i + 1) * cell_size
            y_top = y0 + ny * cell_size
            segments.append(((x_l, y_top), (x_h, y_top)))
    for j in range(ny):
        if fluid[0, j]:
            y_l = y0 + j * cell_size
            y_h = y0 + (j + 1) * cell_size
            segments.append(((x0, y_l), (x0, y_h)))
        if fluid[nx - 1, j]:
            y_l = y0 + j * cell_size
            y_h = y0 + (j + 1) * cell_size
            x_right = x0 + nx * cell_size
            segments.append(((x_right, y_l), (x_right, y_h)))

    return segments


def load_mask(dataset_h5: Path) -> np.ndarray:
    with h5py.File(dataset_h5, "r") as f:
        return np.asarray(f["mask"], dtype=np.float32)


# ─── Plotting ─────────────────────────────────────────────────────────────
def plot_sensor_layout(
    mask: np.ndarray,
    annotated: bool,
    out_path: Path,
) -> None:
    _, _, z_c = cell_centres()
    z_idx = int(np.argmin(np.abs(z_c - Z_SLICE_M)))
    mask_2d = mask[:, :, z_idx]
    segments = extract_wall_segments(mask_2d)

    fig, ax = plt.subplots(figsize=(14, 9))

    # Light grid cells (mimic the user's blue-grid look)
    ax.imshow(
        np.where(mask_2d > 0.5, 0.97, 0.85).T,
        origin="lower", extent=[0, 30, 0, 20], aspect="equal",
        cmap="Blues", vmin=0, vmax=1, alpha=0.5,
    )

    # Wall segments (black)
    lc = LineCollection(segments, colors="black", linewidths=1.3, zorder=3)
    ax.add_collection(lc)

    # Group sensors by area for color
    area_colors = {
        "zone_a": "tab:red",
        "zone_b": "tab:orange",
        "zone_c": "tab:green",
        "zone_d": "tab:purple",
        "corridor": "tab:blue",
        "exit": "gold",
    }
    area_markers = {
        "zone_a": "o", "zone_b": "o", "zone_c": "o", "zone_d": "o",
        "corridor": "s", "exit": "*",
    }
    area_labels: dict[str, bool] = {a: False for a in area_colors}

    for d in ALL_DETECTORS:
        x, y, _ = d.position
        c = area_colors.get(d.area, "black")
        m = area_markers.get(d.area, "o")
        size = 280 if d.area == "exit" else (110 if d.area == "corridor" else 90)
        label = d.area if not area_labels[d.area] else None
        area_labels[d.area] = True
        ax.scatter(
            x, y, s=size, c=c, marker=m,
            edgecolors="black", linewidths=1.0,
            zorder=5, label=label,
        )
        if annotated:
            ax.annotate(
                d.detector_id, (x, y),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, color="black",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="none", alpha=0.7),
                zorder=6,
            )

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 20)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"D-024 sensor layout @ z={Z_SLICE_M:.2f} m  |  "
        f"{len(ALL_DETECTORS)} detectors  (17 rooms + 7 corridors + 3 exits)"
    )
    # Legend in clean order
    handles, labels = ax.get_legend_handles_labels()
    order = ["zone_a", "zone_b", "zone_c", "zone_d", "corridor", "exit"]
    sorted_pairs = sorted(zip(labels, handles), key=lambda p: order.index(p[0]) if p[0] in order else 99)
    sorted_labels = [p[0] for p in sorted_pairs]
    sorted_handles = [p[1] for p in sorted_pairs]
    ax.legend(sorted_handles, sorted_labels, loc="lower right", framealpha=0.9)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--output", type=Path, default=Path("figures/sensor_layout"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    mask = load_mask(args.dataset)
    print(f"[setup] mask shape {mask.shape}")
    print(f"[setup] {len(ALL_DETECTORS)} detectors loaded from D-024")

    print("[plot] sensor_layout.png (plain)")
    plot_sensor_layout(mask, annotated=False, out_path=args.output / "sensor_layout.png")
    print("[plot] sensor_layout_annotated.png (with IDs)")
    plot_sensor_layout(mask, annotated=True, out_path=args.output / "sensor_layout_annotated.png")

    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
