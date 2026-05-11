"""Single-simulation validation script.

Runs three checks back-to-back against one FDS output directory:

1. ``extract_slices`` produces the canonical ``(31, 60, 40, 6)`` tensors
   with sane statistics (initial T ~ 20 °C, fire ignites, no NaN/Inf).
2. matplotlib renders PNG grids + GIF animations + a summary figure so a
   human can sanity-check fire spread visually.
3. Normalisation builds the ``(31, 5, 60, 40, 6)`` model-input tensor and
   confirms every channel sits in ``[0, 1]`` (including the inverse
   visibility mapping and the linear time-encoding).

This is the last gate before running all 30 simulations.

Usage::

    python scripts/validate_simulation.py data/raw/first_sim/
    python scripts/validate_simulation.py data/raw/first_sim/ --output figures/first_sim
    python scripts/validate_simulation.py data/raw/first_sim/ --no-viz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # headless backend — must precede pyplot import
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.data_pipeline.fds_extractor import extract_slices
from src.shared.constants import (
    DT_SLCF,
    GRID_SHAPE,
    N_TIMESTEPS,
    T_END_SECONDS,
)
from src.shared.normalization import (
    compute_time_encoding,
    normalize_co,
    normalize_temperature,
    normalize_visibility,
)


# ─────────────────────────────────────────────────────────────────────────
# Step 1: extract + verify
# ─────────────────────────────────────────────────────────────────────────
def step1_extract_and_verify(fds_dir: Path) -> Dict[str, Any]:
    """Run ``extract_slices`` and validate shapes, ranges, and ignition."""
    print("=" * 70)
    print("Step 1: 데이터 추출 + 검증")
    print("=" * 70)
    print(f"\n시뮬레이션 디렉토리: {fds_dir}")
    result = extract_slices(fds_dir)

    # Shape checks
    print("\n[Shape 검증]")
    expected_shape = (N_TIMESTEPS, *GRID_SHAPE)
    for key in ("temperature", "visibility", "co"):
        shape = result[key].shape
        status = "OK" if shape == expected_shape else "FAIL"
        print(f"  [{status}] {key}: {shape}  (expected {expected_shape})")
        assert shape == expected_shape, (
            f"{key} shape {shape} != expected {expected_shape}"
        )

    # Times
    times = result["times"]
    print("\n[Times 검증]")
    print(f"  times[0:5] = {times[:5]}")
    print(f"  times[-3:] = {times[-3:]}")
    print(f"  총 {len(times)}개")
    assert len(times) == N_TIMESTEPS, f"times length {len(times)} != {N_TIMESTEPS}"

    # Initial temperature
    T = result["temperature"]
    V = result["visibility"]
    CO = result["co"]
    t0_mean = float(T[0].mean())
    t0_max = float(T[0].max())
    print("\n[초기 온도 (t=0)]")
    print(f"  평균: {t0_mean:.2f} °C  (expected ~20)")
    print(f"  최대: {t0_max:.2f} °C   (expected ~20, 초기 균질)")
    if not (15.0 < t0_mean < 25.0):
        print("  ⚠ WARNING: 초기 온도가 비정상")

    # Fire spread over time
    print("\n[화재 확산 시간별 max 온도]")
    snapshot_indices = [0, 5, 10, 15, 20, 25, 30]
    for t_idx in snapshot_indices:
        max_t = float(T[t_idx].max())
        mean_t = float(T[t_idx].mean())
        print(
            f"  t={times[t_idx]:5.0f}s:  max={max_t:7.1f} °C   "
            f"mean={mean_t:6.2f} °C"
        )

    # NaN/Inf
    print("\n[NaN/Inf 체크]")
    for key in ("temperature", "visibility", "co"):
        arr = result[key]
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        status = "OK" if (n_nan == 0 and n_inf == 0) else "FAIL"
        print(f"  [{status}] {key}: NaN={n_nan}, Inf={n_inf}")
        assert n_nan == 0, f"{key} has {n_nan} NaN values"
        assert n_inf == 0, f"{key} has {n_inf} Inf values"

    # Value ranges
    print("\n[값 범위 (raw, 정규화 전)]")
    print(f"  Temperature: [{T.min():.2f}, {T.max():.2f}] °C")
    print(f"  Visibility:  [{V.min():.2f}, {V.max():.2f}] m")
    print(f"  CO:          [{CO.min():.4f}, {CO.max():.4f}] ppm")

    # Fire actually started?
    fire_started = bool(T.max() > 100.0)
    print(f"\n[화재 확산 여부]: {'화재 발생' if fire_started else '화재 안 남'}")
    assert fire_started, f"fire did not ignite: max T = {T.max():.2f} °C < 100"

    print("\n[OK] Step 1 PASS")
    return result


# ─────────────────────────────────────────────────────────────────────────
# Step 2: visualisation
# ─────────────────────────────────────────────────────────────────────────
def _plot_grid(
    field: np.ndarray,
    name: str,
    unit: str,
    cmap: str,
    times: np.ndarray,
    output_path: Path,
    snapshot_indices: list[int],
    z_levels: list[int],
    z_meters: list[float],
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
) -> None:
    """Render ``len(z_levels) × len(snapshot_indices)`` snapshot grid."""
    n_rows = len(z_levels)
    n_cols = len(snapshot_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    # Normalise indexing for the 1-row case so axes[r, c] always works.
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    if vmin is None:
        vmin = float(field.min())
    if vmax is None:
        vmax = float(field.max())

    norm = LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, vmin + 1e-3)) if log_scale else None

    im = None
    for r, iz in enumerate(z_levels):
        for c, ti in enumerate(snapshot_indices):
            ax = axes[r, c]
            slice_2d = field[ti, :, :, iz].T  # (40, 60) — Y axis as rows
            if norm is not None:
                im = ax.imshow(
                    slice_2d, origin="lower", cmap=cmap, norm=norm,
                    extent=[0, 30, 0, 20], aspect="equal",
                )
            else:
                im = ax.imshow(
                    slice_2d, origin="lower", cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    extent=[0, 30, 0, 20], aspect="equal",
                )
            if r == 0:
                ax.set_title(f"t={times[ti]:.0f}s", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"z={z_meters[r]:.2f}m", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=f"{name} ({unit})")
    fig.suptitle(f"{name} — 시간별 + 높이별", fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {output_path}")


def _make_gif(
    field: np.ndarray,
    name: str,
    unit: str,
    cmap: str,
    times: np.ndarray,
    output_path: Path,
    z_idx: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
) -> None:
    """Animate the 31-frame breathing-zone slice (z ≈ 1.75 m)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if vmin is None:
        vmin = float(field.min())
    if vmax is None:
        vmax = float(field.max())

    norm = LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, vmin + 1e-3)) if log_scale else None

    if norm is not None:
        im = ax.imshow(
            field[0, :, :, z_idx].T, origin="lower",
            cmap=cmap, norm=norm,
            extent=[0, 30, 0, 20], aspect="equal",
        )
    else:
        im = ax.imshow(
            field[0, :, :, z_idx].T, origin="lower",
            cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[0, 30, 0, 20], aspect="equal",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, label=f"{name} ({unit})")
    z_metres = 0.25 + z_idx * 0.5
    title = ax.set_title(f"{name} — t=0s (z={z_metres:.2f}m)")

    def _update(frame_idx: int):
        im.set_array(field[frame_idx, :, :, z_idx].T)
        title.set_text(f"{name} — t={times[frame_idx]:.0f}s (z={z_metres:.2f}m)")
        return [im, title]

    anim = animation.FuncAnimation(
        fig, _update, frames=N_TIMESTEPS, interval=200, blit=False
    )
    try:
        anim.save(output_path, writer="pillow", fps=5)
        print(f"  [OK] {output_path}")
    except Exception as exc:  # pragma: no cover — pillow rarely missing in CI
        print(f"  [WARN] GIF save failed: {exc}. PNG grids still produced.")
    plt.close(fig)


def step2_visualize(result: Dict[str, Any], output_dir: Path) -> None:
    """Render PNG grids, GIF animations, and a summary figure."""
    print("\n" + "=" * 70)
    print("Step 2: 시각화")
    print("=" * 70)
    output_dir.mkdir(parents=True, exist_ok=True)

    T = result["temperature"]
    V = result["visibility"]
    CO = result["co"]
    times = result["times"]

    snapshot_indices = [0, 5, 10, 15, 20, 25, 30]
    z_levels = [1, 3, 5]  # z-cells → world z = 0.75, 1.75, 2.75
    z_meters = [0.25 + iz * 0.5 for iz in z_levels]

    print("\n[PNG 그리드 생성]")
    _plot_grid(
        T, "Temperature", "°C", "inferno", times,
        output_dir / "grid_temperature.png",
        snapshot_indices, z_levels, z_meters,
        vmin=20.0, vmax=min(float(T.max()), 800.0),
    )
    _plot_grid(
        V, "Visibility", "m", "viridis_r", times,
        output_dir / "grid_visibility.png",
        snapshot_indices, z_levels, z_meters,
        vmin=0.0, vmax=30.0,
    )
    _plot_grid(
        CO, "CO", "ppm", "magma", times,
        output_dir / "grid_co.png",
        snapshot_indices, z_levels, z_meters,
        vmin=0.0, vmax=max(float(CO.max()), 10.0),
        log_scale=True,
    )

    print("\n[GIF 애니메이션 생성]")
    _make_gif(
        T, "Temperature", "°C", "inferno", times,
        output_dir / "animation_temperature.gif",
        vmin=20.0, vmax=min(float(T.max()), 800.0),
    )
    _make_gif(
        V, "Visibility", "m", "viridis_r", times,
        output_dir / "animation_visibility.gif",
        vmin=0.0, vmax=30.0,
    )
    _make_gif(
        CO, "CO", "ppm", "magma", times,
        output_dir / "animation_co.gif",
        vmin=0.0, vmax=max(float(CO.max()), 10.0),
        log_scale=True,
    )

    # Summary figure: T / V / CO at t=150s, z=1.75m
    print("\n[Summary figure]")
    t_summary = 15
    z_summary = 3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(
        T[t_summary, :, :, z_summary].T, origin="lower",
        cmap="inferno", vmin=20.0, vmax=min(float(T.max()), 800.0),
        extent=[0, 30, 0, 20], aspect="equal",
    )
    axes[0].set_title(f"Temperature (°C) @ t={times[t_summary]:.0f}s")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        V[t_summary, :, :, z_summary].T, origin="lower",
        cmap="viridis_r", vmin=0.0, vmax=30.0,
        extent=[0, 30, 0, 20], aspect="equal",
    )
    axes[1].set_title(f"Visibility (m) @ t={times[t_summary]:.0f}s")
    plt.colorbar(im1, ax=axes[1])

    co_max = float(CO.max())
    im2 = axes[2].imshow(
        CO[t_summary, :, :, z_summary].T, origin="lower",
        cmap="magma",
        norm=LogNorm(vmin=max(float(CO.min()), 1e-3), vmax=max(co_max, 10.0)),
        extent=[0, 30, 0, 20], aspect="equal",
    )
    axes[2].set_title(f"CO (ppm) @ t={times[t_summary]:.0f}s")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    plt.suptitle(f"Summary @ t={times[t_summary]:.0f}s, z={0.25 + z_summary * 0.5:.2f}m")
    plt.tight_layout()
    summary_path = output_dir / "summary.png"
    plt.savefig(summary_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {summary_path}")

    print("\n[OK] Step 2 PASS")


# ─────────────────────────────────────────────────────────────────────────
# Step 3: normalisation + model-input tensor
# ─────────────────────────────────────────────────────────────────────────
def step3_normalize_and_build_input(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise channels and stack the ``(31, 5, 60, 40, 6)`` model input."""
    print("\n" + "=" * 70)
    print("Step 3: 정규화 + 입력 텐서 빌드 + 검증")
    print("=" * 70)

    T = result["temperature"]
    V = result["visibility"]
    CO = result["co"]
    times = result["times"]

    print("\n[정규화 적용]")
    T_norm = normalize_temperature(T).astype(np.float32)
    V_norm = normalize_visibility(V).astype(np.float32)
    CO_norm = normalize_co(CO).astype(np.float32)

    for name, arr in (("T_norm", T_norm), ("V_norm", V_norm), ("CO_norm", CO_norm)):
        in_range = bool((arr.min() >= 0.0) and (arr.max() <= 1.0))
        status = "OK" if in_range else "FAIL"
        print(
            f"  [{status}] {name}: [{arr.min():.4f}, {arr.max():.4f}] "
            f"mean={arr.mean():.4f}"
        )
        assert in_range, f"{name} outside [0, 1]"

    # Visibility inverse check
    print("\n[Visibility inverse 매핑 검증]")
    v_max_idx = tuple(int(x) for x in np.unravel_index(V.argmax(), V.shape))
    print(f"  최대 raw V: {float(V[v_max_idx]):.2f} m at {v_max_idx}")
    print(f"  같은 위치 V_norm: {float(V_norm[v_max_idx]):.4f} (작아야 함, ~0)")
    assert V_norm[v_max_idx] < 0.5, "visibility inverse mapping appears broken"

    # Time encoding
    print("\n[시간 인코딩]")
    time_enc_per_frame = np.array(
        [compute_time_encoding(float(t)) for t in times], dtype=np.float32
    )
    print(f"  time_enc[0]  = {time_enc_per_frame[0]:.4f}  (expected 0.0)")
    print(f"  time_enc[15] = {time_enc_per_frame[15]:.4f}  (expected 0.5)")
    print(f"  time_enc[30] = {time_enc_per_frame[30]:.4f}  (expected 1.0)")
    assert time_enc_per_frame[0] == 0.0, "time encoding at t=0 not 0"
    assert abs(time_enc_per_frame[15] - 0.5) < 0.02, "time encoding at mid-frame off"
    assert time_enc_per_frame[30] == 1.0, "time encoding at last frame not 1"
    print("  [OK] linear t/300 confirmed")

    # Build (31, 5, 60, 40, 6)
    print("\n[입력 텐서 빌드 (placeholder mask)]")
    mask = np.ones(GRID_SHAPE, dtype=np.float32)
    mask_broadcast = np.broadcast_to(
        mask[None, :, :, :], (N_TIMESTEPS,) + GRID_SHAPE
    ).astype(np.float32)
    time_enc_grid = np.broadcast_to(
        time_enc_per_frame[:, None, None, None],
        (N_TIMESTEPS,) + GRID_SHAPE,
    ).astype(np.float32)

    input_tensor = np.stack(
        [T_norm, V_norm, CO_norm, mask_broadcast, time_enc_grid], axis=1,
    ).astype(np.float32)
    target_tensor = input_tensor[:, :3, :, :, :].copy()

    expected_in = (N_TIMESTEPS, 5, *GRID_SHAPE)
    expected_out = (N_TIMESTEPS, 3, *GRID_SHAPE)
    print(f"  [OK] input shape:  {input_tensor.shape}  (expected {expected_in})")
    print(f"  [OK] target shape: {target_tensor.shape}  (expected {expected_out})")
    print(f"  [OK] dtype: {input_tensor.dtype}")
    assert input_tensor.shape == expected_in, "input shape mismatch"
    assert target_tensor.shape == expected_out, "target shape mismatch"
    assert input_tensor.dtype == np.float32, "input dtype not float32"

    # Final range check
    print("\n[전체 범위 확인]")
    print(f"  input  : [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    print(f"  target : [{target_tensor.min():.4f}, {target_tensor.max():.4f}]")
    assert 0.0 <= input_tensor.min() and input_tensor.max() <= 1.0, (
        "input tensor outside [0, 1]"
    )

    # Per-channel stats
    print("\n[채널별 통계]")
    channel_names = ["T_norm", "V_norm", "CO_norm", "mask", "time_enc"]
    stats: Dict[str, Dict[str, float]] = {}
    for c_idx, name in enumerate(channel_names):
        ch = input_tensor[:, c_idx, :, :, :]
        entry = {
            "min": float(ch.min()),
            "max": float(ch.max()),
            "mean": float(ch.mean()),
            "std": float(ch.std()),
        }
        print(
            f"  ch{c_idx} {name:8s}: "
            f"min={entry['min']:.4f}, max={entry['max']:.4f}, "
            f"mean={entry['mean']:.4f}, std={entry['std']:.4f}"
        )
        stats[name] = entry

    print("\n[OK] Step 3 PASS")
    return {
        "input_tensor": input_tensor,
        "target_tensor": target_tensor,
        "stats": stats,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="단일 FDS 시뮬레이션 검증")
    parser.add_argument("fds_dir", type=Path, help="FDS 시뮬레이션 디렉토리")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="시각화 출력 디렉토리 (기본: figures/{시나리오ID})",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="시각화 단계 스킵 (빠른 검증)",
    )
    args = parser.parse_args()

    if not args.fds_dir.exists():
        print(f"ERROR: directory not found: {args.fds_dir}")
        return 1

    scenario_id = args.fds_dir.name
    output_dir = args.output or Path("figures") / scenario_id

    print(f"\n{'#' * 70}")
    print(f"# 시뮬레이션 검증: {scenario_id}")
    print(f"{'#' * 70}")

    result = step1_extract_and_verify(args.fds_dir)

    if not args.no_viz:
        step2_visualize(result, output_dir)

    norm_result = step3_normalize_and_build_input(result)

    print("\n" + "=" * 70)
    print(f"전체 검증 PASS: {scenario_id}")
    print("=" * 70)

    if not args.no_viz:
        print("\n생성된 파일:")
        print(f"  시각화: {output_dir}/")
        for f in sorted(output_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name} ({size_kb:.1f} KB)")

        stats_file = output_dir / "stats.json"
        with stats_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "scenario_id": scenario_id,
                    "channel_stats": norm_result["stats"],
                    "shape_input": list(norm_result["input_tensor"].shape),
                    "shape_target": list(norm_result["target_tensor"].shape),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  통계: {stats_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
