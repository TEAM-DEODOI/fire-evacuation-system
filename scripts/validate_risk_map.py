"""End-to-end risk-map validation on a single FDS scenario.

Pipeline:

1. ``extract_slices(fds_dir)`` → raw T, V, CO (31, 60, 40, 6).
2. ``compute_total_danger`` → aggregated danger grid (31, 60, 40, 6).
3. ``compute_aset_map`` → per-cell ASET map (60, 40, 6) in seconds.
4. ``StaticRiskMap`` interpolator → ``query(xyz, t)`` smoke test.
5. ``accumulate_fed_co`` → cumulative FED at two fixed observers.
6. matplotlib visualisations:

   * Danger field at z=1.75 m for 7 time snapshots.
   * ASET map at z=0.75 / 1.75 / 2.75 m.
   * FED accumulation curves for near-fire vs far-from-fire observers.

Usage::

    python scripts/validate_risk_map.py data/raw/first_sim/
    python scripts/validate_risk_map.py data/raw/first_sim/ --output figures/first_sim/risk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.data_pipeline.fds_extractor import extract_slices
from src.risk_map.aset import compute_aset_map
from src.risk_map.fed import (
    accumulate_fed_co,
    time_to_incapacitation,
)
from src.risk_map.risk_map_class import StaticRiskMap
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import (
    DOMAIN_SIZE_M,
    DT_SLCF,
    GRID_SHAPE,
    N_TIMESTEPS,
    T_END_SECONDS,
    TENABILITY,
)


# ─────────────────────────────────────────────────────────────────────────
# Step 1: extract + compute danger field
# ─────────────────────────────────────────────────────────────────────────
def step1_extract_and_danger(fds_dir: Path) -> Dict[str, Any]:
    """Extract slices, build danger grid + ASET map, build risk map."""
    print("=" * 70)
    print("Step 1: extract + compute danger grid")
    print("=" * 70)
    print(f"\nfds_dir: {fds_dir}")

    slices = extract_slices(fds_dir)
    T = slices["temperature"]
    V = slices["visibility"]
    CO = slices["co"]
    times = slices["times"]

    print(f"\n  T shape={T.shape}  range=[{T.min():.2f}, {T.max():.2f}] °C")
    print(f"  V shape={V.shape}  range=[{V.min():.2f}, {V.max():.2f}] m")
    print(f"  CO shape={CO.shape}  range=[{CO.min():.2f}, {CO.max():.2f}] ppm")

    print("\n  computing total danger (T/V/CO weights = "
          f"{TENABILITY.WEIGHT_T}/{TENABILITY.WEIGHT_V}/{TENABILITY.WEIGHT_CO})")
    danger = compute_total_danger(T, V, CO).astype(np.float32)
    assert danger.shape == (N_TIMESTEPS, *GRID_SHAPE), (
        f"danger shape {danger.shape}"
    )
    assert 0.0 <= danger.min() and danger.max() <= 1.0, (
        f"danger out of [0, 1]: [{danger.min()}, {danger.max()}]"
    )
    print(f"  danger shape={danger.shape}  "
          f"range=[{danger.min():.3f}, {danger.max():.3f}]")
    print(f"  danger at t=0  : max={danger[0].max():.4f}  mean={danger[0].mean():.4f}")
    print(f"  danger at t=150: max={danger[15].max():.4f}  mean={danger[15].mean():.4f}")
    print(f"  danger at t=300: max={danger[-1].max():.4f}  mean={danger[-1].mean():.4f}")

    print("\n  building ASET map (threshold=0.5)")
    aset = compute_aset_map(danger)
    n_safe_always = int((aset == T_END_SECONDS).sum())
    n_total = aset.size
    print(f"  ASET shape={aset.shape}  "
          f"range=[{aset.min():.1f}, {aset.max():.1f}] s")
    print(f"  cells never crossing 0.5: {n_safe_always}/{n_total} "
          f"({100 * n_safe_always / n_total:.1f}%)")

    print("\n  wrapping in StaticRiskMap")
    rm = StaticRiskMap(danger_array=danger, times=times)
    print(f"  start_time={rm.start_time}  t_max={rm.t_max}")

    print("\n[OK] Step 1 PASS")
    return {
        "slices": slices,
        "danger": danger,
        "aset": aset,
        "risk_map": rm,
        "times": times,
    }


# ─────────────────────────────────────────────────────────────────────────
# Step 2: StaticRiskMap.query smoke tests
# ─────────────────────────────────────────────────────────────────────────
def step2_query_tests(rm: StaticRiskMap) -> None:
    print("\n" + "=" * 70)
    print("Step 2: StaticRiskMap.query checks")
    print("=" * 70)

    # near-fire location (validate_simulation showed hot spot near (18, 10))
    near_fire = np.array([18.0, 10.0, 1.75])
    far_fire = np.array([2.0, 18.0, 1.75])
    oob = np.array([-1.0, 10.0, 1.75])

    for label, p in (("near fire", near_fire),
                      ("far from fire", far_fire),
                      ("out-of-bounds", oob)):
        for t in (0.0, 150.0, 300.0):
            v = rm.query(p, t=t)
            print(f"  {label:14s} @ t={t:5.0f}s → danger = {v:.4f}")

    # Batch query
    batch = np.stack([near_fire, far_fire, oob])
    out = rm.query(batch, t=150.0)
    print(f"\n  batch query at t=150: {out}")
    assert out[2] == 1.0, "OOB in batch did not return 1.0"

    # Beyond t_max
    v_inf = rm.query(near_fire, t=9999.0)
    assert v_inf == 1.0, "t > t_max did not return 1.0"
    print(f"  t > t_max → {v_inf}  (safety default)")

    print("\n[OK] Step 2 PASS")


# ─────────────────────────────────────────────────────────────────────────
# Step 3: FED accumulation at two observers
# ─────────────────────────────────────────────────────────────────────────
def step3_fed_curves(slices: Dict[str, np.ndarray]) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("Step 3: FED accumulation at fixed observers")
    print("=" * 70)

    CO = slices["co"]                             # (31, 60, 40, 6) ppm
    x_centres = np.linspace(0.25, 29.75, GRID_SHAPE[0])
    y_centres = np.linspace(0.25, 19.75, GRID_SHAPE[1])
    z_centres = np.linspace(0.25, 2.75, GRID_SHAPE[2])

    def _cell_of(x: float, y: float, z: float) -> tuple[int, int, int]:
        return (
            int(round((x - 0.25) / 0.5)),
            int(round((y - 0.25) / 0.5)),
            int(round((z - 0.25) / 0.5)),
        )

    observers = {
        "near_fire (18, 10, 1.75)": (18.0, 10.0, 1.75),
        "far_from_fire (2, 18, 1.75)": (2.0, 18.0, 1.75),
        "mid_corridor (15, 5, 1.75)": (15.0, 5.0, 1.75),
    }

    fed_curves: Dict[str, np.ndarray] = {}
    co_curves: Dict[str, np.ndarray] = {}
    for label, (x, y, z) in observers.items():
        ix, iy, iz = _cell_of(x, y, z)
        co_series = CO[:, ix, iy, iz]
        fed = accumulate_fed_co(co_series, dt_seconds=DT_SLCF)
        ttx = time_to_incapacitation(fed, dt_seconds=DT_SLCF)
        ttx_str = f"{ttx:.0f} s" if ttx != float("inf") else "∞ (safe)"
        print(
            f"  {label:36s}  "
            f"max CO={co_series.max():>7.1f} ppm  "
            f"final FED={fed[-1]:.4f}  "
            f"t_inc={ttx_str}"
        )
        co_curves[label] = co_series
        fed_curves[label] = fed

    print(f"\n  FED danger threshold: {TENABILITY.FED_THRESHOLD}")
    print("[OK] Step 3 PASS")
    return {"fed_curves": fed_curves, "co_curves": co_curves}


# ─────────────────────────────────────────────────────────────────────────
# Step 4: visualisation
# ─────────────────────────────────────────────────────────────────────────
def _plot_danger_panels(
    danger: np.ndarray,
    times: np.ndarray,
    out_path: Path,
) -> None:
    snap = [0, 5, 10, 15, 20, 25, 30]
    z_levels = [1, 3, 5]
    z_meters = [0.25 + iz * 0.5 for iz in z_levels]
    fig, axes = plt.subplots(
        len(z_levels), len(snap),
        figsize=(3 * len(snap), 2.5 * len(z_levels)),
    )
    if len(z_levels) == 1:
        axes = np.array([axes])
    im = None
    for r, iz in enumerate(z_levels):
        for c, ti in enumerate(snap):
            ax = axes[r, c]
            im = ax.imshow(
                danger[ti, :, :, iz].T,
                origin="lower", cmap="RdYlGn_r",
                vmin=0.0, vmax=1.0,
                extent=[0, DOMAIN_SIZE_M[0], 0, DOMAIN_SIZE_M[1]],
                aspect="equal",
            )
            if r == 0:
                ax.set_title(f"t={times[ti]:.0f}s", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"z={z_meters[r]:.2f}m", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="aggregated danger (0–1)")
    fig.suptitle("Aggregated danger (T/V/CO 0.4/0.4/0.2)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def _plot_aset(aset: np.ndarray, out_path: Path) -> None:
    z_levels = [1, 3, 5]
    z_meters = [0.25 + iz * 0.5 for iz in z_levels]
    fig, axes = plt.subplots(1, len(z_levels), figsize=(4 * len(z_levels), 4))
    im = None
    for ax, iz, zm in zip(axes, z_levels, z_meters):
        im = ax.imshow(
            aset[:, :, iz].T,
            origin="lower", cmap="viridis",
            vmin=0.0, vmax=T_END_SECONDS,
            extent=[0, DOMAIN_SIZE_M[0], 0, DOMAIN_SIZE_M[1]],
            aspect="equal",
        )
        ax.set_title(f"ASET @ z={zm:.2f}m")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    fig.colorbar(im, ax=axes, label="ASET (seconds, capped at T_END=300)")
    fig.suptitle(
        f"ASET map (threshold={TENABILITY.AGGREGATE_THRESHOLD})  "
        "— darker = sooner-dangerous", fontsize=11,
    )
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def _plot_fed_curves(
    times: np.ndarray,
    co_curves: Dict[str, np.ndarray],
    fed_curves: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, (ax_co, ax_fed) = plt.subplots(1, 2, figsize=(12, 4))
    colours = ["tab:red", "tab:blue", "tab:green"]
    for (label, co), (_, fed), c in zip(co_curves.items(), fed_curves.items(), colours):
        ax_co.plot(times, co, label=label, color=c)
        ax_fed.plot(times, fed, label=label, color=c)

    ax_co.set_xlabel("t (s)")
    ax_co.set_ylabel("CO (ppm)")
    ax_co.set_title("CO concentration at fixed observers")
    ax_co.legend(fontsize=8)
    ax_co.grid(alpha=0.3)

    ax_fed.set_xlabel("t (s)")
    ax_fed.set_ylabel("cumulative FED")
    ax_fed.axhline(
        TENABILITY.FED_THRESHOLD,
        color="black", lw=0.8, ls="--",
        label=f"FED threshold = {TENABILITY.FED_THRESHOLD}",
    )
    ax_fed.set_title("Cumulative FED (ISO 13571 §7.3)")
    ax_fed.legend(fontsize=8)
    ax_fed.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def step4_visualize(
    state: Dict[str, Any],
    fed: Dict[str, Any],
    output_dir: Path,
) -> None:
    print("\n" + "=" * 70)
    print("Step 4: visualisation")
    print("=" * 70)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_danger_panels(
        state["danger"], state["times"], output_dir / "danger_panels.png"
    )
    _plot_aset(state["aset"], output_dir / "aset_map.png")
    _plot_fed_curves(
        state["times"], fed["co_curves"], fed["fed_curves"],
        output_dir / "fed_curves.png",
    )
    print("\n[OK] Step 4 PASS")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Risk-map validation on one FDS scenario")
    parser.add_argument("fds_dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="figures output directory (default: figures/{scenario_id}/risk)",
    )
    args = parser.parse_args()

    if not args.fds_dir.exists():
        print(f"ERROR: directory not found: {args.fds_dir}")
        return 1

    scenario_id = args.fds_dir.name
    output_dir = args.output or (Path("figures") / scenario_id / "risk")

    print("\n" + "#" * 70)
    print(f"# Risk-map validation: {scenario_id}")
    print("#" * 70)

    state = step1_extract_and_danger(args.fds_dir)
    step2_query_tests(state["risk_map"])
    fed = step3_fed_curves(state["slices"])
    step4_visualize(state, fed, output_dir)

    # Persist summary stats for later comparison.
    stats_path = output_dir / "stats.json"
    payload = {
        "scenario_id": scenario_id,
        "danger_max": float(state["danger"].max()),
        "danger_mean_at_t_end": float(state["danger"][-1].mean()),
        "aset_min_s": float(state["aset"].min()),
        "aset_max_s": float(state["aset"].max()),
        "fraction_safe_at_t_end": float(
            (state["aset"] == T_END_SECONDS).sum() / state["aset"].size
        ),
        "fed_final": {k: float(v[-1]) for k, v in fed["fed_curves"].items()},
        "fed_threshold": TENABILITY.FED_THRESHOLD,
    }
    stats_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print(f"OVERALL PASS: {scenario_id}")
    print("=" * 70)
    print("\nartefacts:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:24s}  {size_kb:>6.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
