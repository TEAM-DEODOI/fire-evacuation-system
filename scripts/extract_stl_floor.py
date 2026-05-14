"""STL → z=1.75m 평면 단면 추출 + 27 sensor 위치 후보 시각화.

목적: 사용자가 사진에 그린 파란/빨간 곡선 위치와 실제 STL wall 위치 비교 →
정확한 sensor 위치 산출.

워크플로:
1. STL 로드 (단위 mm → m 변환)
2. z=1.75m (breathing height) 평면 단면 추출 — wall trace
3. 새 sensor 위치 후보를 색깔별로 plot (파란=방, 빨강=복도, 노랑별=출구)
4. 사용자 검증용 figure 생성
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import h5py
from matplotlib.collections import LineCollection
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_sensor_layout import extract_wall_segments as extract_mask_segments


def load_stl_in_metres(stl_path: Path) -> trimesh.Trimesh:
    """STL 로드 + 단위 mm → m 변환 (bound 자동 감지)."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    # Auto-detect units: if bounds in 10000+ range, assume mm
    max_extent = mesh.extents.max()
    if max_extent > 1000:
        mesh.apply_scale(0.001)
        print(f"  unit: mm → m (scaled by 0.001)")
    print(f"  extents (m): {mesh.extents}")
    print(f"  bounds (m): min={mesh.bounds[0]}, max={mesh.bounds[1]}")
    return mesh


def extract_z_slice(mesh: trimesh.Trimesh, z: float) -> np.ndarray:
    """Get wall trace at z plane.

    Returns:
        ``(N, 2, 2)`` array of line segments in (x, y) plane.
    """
    plane_origin = np.array([0, 0, z])
    plane_normal = np.array([0, 0, 1])
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        return np.zeros((0, 2, 2))
    # to_2D returns Path2D
    path2d, _ = section.to_planar()
    segments = []
    for entity in path2d.entities:
        pts = path2d.vertices[entity.points]
        # break into consecutive segments
        for i in range(len(pts) - 1):
            segments.append([[pts[i, 0], pts[i, 1]],
                              [pts[i + 1, 0], pts[i + 1, 1]]])
    return np.array(segments)


def plot_floor_with_sensor_candidates(
    segments: np.ndarray,
    domain_x: Tuple[float, float],
    domain_y: Tuple[float, float],
    out_path: Path,
    mask_segments: np.ndarray | None = None,
) -> None:
    """Plot wall outline + sensor candidates color-coded.

    사용자 사진의 작은 파란/빨간 곡선 위치를 시각 추정해서 점으로 표시.
    Optionally overlay mask-derived wall segments (grey, thin) for comparison.
    """
    fig, ax = plt.subplots(figsize=(15, 9))

    # Mask wall segments (grey, thin) — for cross-check with STL
    if mask_segments is not None and len(mask_segments) > 0:
        lc_m = LineCollection(mask_segments, colors="grey", linewidths=0.6,
                               linestyles=":", zorder=2, alpha=0.5)
        ax.add_collection(lc_m)

    # STL wall segments (black, thick)
    if len(segments) > 0:
        lc = LineCollection(segments, colors="black", linewidths=1.4, zorder=3,
                             label="STL wall")
        ax.add_collection(lc)

    # ── Sensor candidates (estimated from user's photo annotations) ──────
    # Blue = room sensors (방 안)
    room_sensors = [
        # Zone A 사선 (좌상)
        (1.5, 9.3, "zone_a_room1"),
        (3.0, 12.5, "zone_a_room2"),
        (4.0, 13.0, "zone_a_room3"),
        (6.0, 13.0, "zone_a_room4"),
        (5.5, 16.5, "zone_a_room5"),
        (8.5, 17.5, "zone_a_room6"),
        # 북측 (Zone C 위쪽 + 가운데 상단 작은 방들)
        (11.5, 18.5, "north_room1"),
        (14.5, 18.5, "north_room2"),
        (17.5, 18.5, "north_room3"),
        # Zone C (북동 큰 방들)
        (21.0, 17.5, "zone_c_1"),
        (24.0, 17.5, "zone_c_2"),
        (27.0, 17.5, "zone_c_3"),
        (28.5, 18.5, "zone_c_4_corner"),
        (27.0, 14.5, "zone_c_east_mid"),
        # Zone B (남측)
        (2.5, 2.5, "zone_b_1"),
        (6.5, 2.5, "zone_b_2"),
        (10.0, 2.5, "zone_b_3"),
        (14.0, 2.5, "zone_b_4"),
        (19.0, 2.5, "zone_b_5"),
    ]
    # Red = corridor sensors (복도)
    corridor_sensors = [
        (8.0, 11.0, "corridor_a_diag_1"),
        (10.0, 14.5, "corridor_a_diag_2"),
        (15.5, 15.5, "north_corridor"),
        (16.0, 13.0, "east_corridor_upper"),
        (16.0, 8.0, "east_corridor_lower"),
        (16.0, 5.5, "south_corridor_east"),
        (10.5, 5.5, "south_corridor_mid"),
        (5.5, 5.5, "south_corridor_west"),
    ]
    # Gold star = exits
    exits = [
        (2.5, 5.75, "exit_west"),
        (15.0, 16.75, "exit_north"),
        (22.0, 5.75, "exit_east"),
    ]

    # Plot
    for x, y, name in room_sensors:
        ax.scatter(x, y, s=140, c="dodgerblue", edgecolors="black",
                    linewidths=1.0, zorder=5, marker="o",
                    label="room" if name == room_sensors[0][2] else None)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=6, color="black",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.6))
    for x, y, name in corridor_sensors:
        ax.scatter(x, y, s=140, c="tomato", edgecolors="black",
                    linewidths=1.0, zorder=5, marker="s",
                    label="corridor" if name == corridor_sensors[0][2] else None)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=6, color="black",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.6))
    for x, y, name in exits:
        ax.scatter(x, y, s=320, c="gold", edgecolors="black",
                    linewidths=1.0, zorder=6, marker="*",
                    label="exit" if name == exits[0][2] else None)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=6, color="black",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.6))

    ax.set_xlim(domain_x)
    ax.set_ylim(domain_y)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    n_total = len(room_sensors) + len(corridor_sensors) + len(exits)
    ax.set_title(
        f"NEW sensor candidates @ z=1.75 m  |  STL wall (black)  |  "
        f"{n_total} detectors  "
        f"({len(room_sensors)} rooms + {len(corridor_sensors)} corridors + {len(exits)} exits)"
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", type=Path, default=Path("assets/science_hall_lv5.stl"))
    parser.add_argument("--z", type=float, default=1.75)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/sensor_layout_v2"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] loading STL: {args.stl}")
    mesh = load_stl_in_metres(args.stl)
    print(f"[2] extracting z={args.z} m cross-section")
    segments = extract_z_slice(mesh, z=args.z)
    print(f"    STL segments: {len(segments)}")

    # Load mask wall segments (from dataset.h5, z=1.75m)
    mask_segments_arr = None
    if Path("data/processed/dataset.h5").exists():
        print(f"[3] loading mask wall (z~1.75) from dataset.h5")
        with h5py.File("data/processed/dataset.h5", "r") as f:
            mask = np.asarray(f["mask"], dtype=np.float32)
        # Use z_idx=3 (≈1.75 m breathing height)
        from src.shared.coordinates import cell_centres
        _, _, z_c = cell_centres()
        z_idx = int(np.argmin(np.abs(z_c - args.z)))
        mask_segs_raw = extract_mask_segments(mask[:, :, z_idx])
        # Convert tuple-of-tuples to (N, 2, 2) array
        mask_segments_arr = np.array([[list(p1), list(p2)] for p1, p2 in mask_segs_raw])
        print(f"    mask segments (z={z_c[z_idx]:.2f}m): {len(mask_segments_arr)}")

    print(f"[4] plotting sensor candidates + overlay")
    plot_floor_with_sensor_candidates(
        segments,
        domain_x=(0.0, 30.0), domain_y=(0.0, 20.0),
        out_path=args.out_dir / "stl_vs_mask_overlay.png",
        mask_segments=mask_segments_arr,
    )
    print("\n[PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
