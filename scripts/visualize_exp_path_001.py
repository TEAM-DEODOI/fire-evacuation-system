"""EXP-PATH-001 comparison.csv -> 2x2 boxplot grid (paper Figure 4.4).

Reads ``results/exp_path_001/comparison.csv`` and renders one PNG with
four box plots (one per metric) overlaid by scenario:

* evacuation success rate (%)
* mean evacuation time (s)
* danger-zone exposure time (s)
* cumulative FED

Plus an annotation strip on top summarising:

* S2 vs S1 delta on success rate and mean exposure (the H6 trade-off)
* S2 ≡ S3 identity over every (fire, seed) row (H5 transitive)

Run::

    python scripts/visualize_exp_path_001.py

Defaults read from ``results/exp_path_001/comparison.csv`` and write
to ``figures/exp_path_001/comparison.png``.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Order of scenarios on the x-axis. Also defines color order.
_SCENARIO_ORDER = ["S1_fixed_sign", "S2_fds_swarm", "S3_fno_swarm"]
_SCENARIO_LABELS = ["S1\nfixed sign", "S2\nFDS swarm", "S3\nmodel swarm"]
_SCENARIO_COLORS = ["#888888", "#1f77b4", "#2ca02c"]   # gray / blue / green


def _boxplot(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str,
             y_is_percent: bool = False) -> None:
    """Render one metric as a boxplot, S1/S2/S3 grouped."""
    data = [
        df.loc[df["scenario_id"] == s, metric].values
        for s in _SCENARIO_ORDER
    ]
    if y_is_percent:
        data = [d * 100 for d in data]
    bp = ax.boxplot(
        data,
        labels=_SCENARIO_LABELS,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.4),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
    )
    for patch, c in zip(bp["boxes"], _SCENARIO_COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    # Overlay raw points (jittered) for transparency about n.
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        x_jit = np.random.RandomState(i).uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(
            [i + 1] * len(vals) + x_jit,
            vals,
            s=14,
            facecolor="black",
            alpha=0.45,
            zorder=3,
            edgecolor="none",
        )
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def _compute_deltas(df: pd.DataFrame) -> Dict[str, float]:
    """Mean per-(fire, seed) deltas between scenarios."""
    s1 = (
        df[df.scenario_id == "S1_fixed_sign"]
        .set_index(["fire_scenario_id", "seed"])
    )
    s2 = (
        df[df.scenario_id == "S2_fds_swarm"]
        .set_index(["fire_scenario_id", "seed"])
    )
    s3 = (
        df[df.scenario_id == "S3_fno_swarm"]
        .set_index(["fire_scenario_id", "seed"])
    )
    return {
        "s2_minus_s1_evac_pp": float(
            (s2.evacuation_success_rate - s1.evacuation_success_rate).mean()
            * 100
        ),
        "s2_minus_s1_t_evac_s": float(
            (s2.mean_evacuation_time_s - s1.mean_evacuation_time_s).mean()
        ),
        "s2_minus_s1_exposure_s": float(
            (s2.danger_zone_exposure_time_s - s1.danger_zone_exposure_time_s).mean()
        ),
        "s3_vs_s2_identical_rows": int(
            (
                (s2.evacuation_success_rate == s3.evacuation_success_rate)
                & np.isclose(s2.mean_evacuation_time_s, s3.mean_evacuation_time_s)
                & np.isclose(
                    s2.danger_zone_exposure_time_s,
                    s3.danger_zone_exposure_time_s,
                )
            ).sum()
        ),
        "n_rows_per_scenario": int(len(s2)),
    }


def render(csv_path: Path, out_path: Path, title: str | None = None) -> Path:
    """Read ``csv_path``, render the 2x2 boxplot, save to ``out_path``."""
    df = pd.read_csv(csv_path)
    missing = set(_SCENARIO_ORDER) - set(df.scenario_id.unique())
    if missing:
        raise ValueError(f"CSV missing scenarios: {missing}")

    deltas = _compute_deltas(df)
    n_rows = deltas["n_rows_per_scenario"]
    n_fires = df.fire_scenario_id.nunique()
    n_seeds = df.seed.nunique()
    n_persons = int(df.n_persons.max())

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    _boxplot(
        axes[0, 0], df,
        metric="evacuation_success_rate",
        title="Evacuation success rate",
        ylabel="%",
        y_is_percent=True,
    )
    axes[0, 0].set_ylim(0, 105)
    _boxplot(
        axes[0, 1], df,
        metric="mean_evacuation_time_s",
        title="Mean evacuation time",
        ylabel="s",
    )
    _boxplot(
        axes[1, 0], df,
        metric="danger_zone_exposure_time_s",
        title="Danger-zone exposure time (truth > 0.5)",
        ylabel="s",
    )
    _boxplot(
        axes[1, 1], df,
        metric="cumulative_fed",
        title="Cumulative FED (H6 primary metric)",
        ylabel="FED",
    )
    # FED panel: annotate that the metric is gated on M2-full / M3-full
    if df.cumulative_fed.abs().max() == 0.0:
        axes[1, 1].text(
            0.5, 0.5,
            "FED = 0 across all rows\nM2-full + M3-full prerequisite\n"
            "(PersonAgent CO-FED + raw CO grid)",
            ha="center", va="center", fontsize=10,
            transform=axes[1, 1].transAxes,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#fff7d6",
                edgecolor="#a07a00",
                linewidth=0.8,
            ),
        )

    # ── Supertitle + finding strip ─────────────────────────────────
    if title is None:
        title = (
            f"EXP-PATH-001 mini-sweep -- "
            f"{n_fires} fire scenarios x {n_seeds} seeds "
            f"x 3 guidance scenarios = {n_rows * 3} runs, "
            f"{n_persons} persons each"
        )
    fig.suptitle(title, fontsize=12.5, y=0.99)

    findings = (
        f"S2 vs S1:  evac %{deltas['s2_minus_s1_evac_pp']:+5.1f}p   "
        f"t_evac {deltas['s2_minus_s1_t_evac_s']:+5.2f}s   "
        f"exposure {deltas['s2_minus_s1_exposure_s']:+5.2f}s        |        "
        f"H5 transitive:  S3 == S2 in {deltas['s3_vs_s2_identical_rows']}/{n_rows} rows"
    )
    fig.text(0.5, 0.945, findings, ha="center", fontsize=10.5,
             color="#222222", family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the EXP-PATH-001 comparison.csv into a 2x2 boxplot."
    )
    parser.add_argument(
        "--csv", type=Path,
        default=Path("results/exp_path_001/comparison.csv"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("figures/exp_path_001/comparison.png"),
    )
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"FAIL: csv not found: {args.csv}", file=sys.stderr)
        return 1
    out = render(args.csv, args.out, title=args.title)
    size_kb = out.stat().st_size / 1024
    print(f"PASS: wrote {out} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
