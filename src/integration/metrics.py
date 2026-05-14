"""5-metric collector for EXP-PATH-001 (D-025 headline).

Per CLAUDE.md "EXP-PATH-001 — Three Scenarios", each PyBullet run
produces a row of five metrics per scenario × fire × seed:

1. ``evacuation_success_rate`` — fraction of persons whose status is
   ``evacuated`` at end of run.
2. ``mean_evacuation_time``    — mean reach-time across evacuated
   persons only (s). Undefined (NaN) if none evacuated.
3. ``danger_zone_exposure_time`` — mean per-person seconds spent in
   ``risk_map_truth.query(...) > 0.5``.
4. ``casualty_rate``           — fraction of persons whose status is
   ``dead`` at end of run.
5. ``cumulative_FED``          — mean per-person final cumulative FED.
   This is the **H6 primary metric** (S2_FED ≤ 0.7 · S1_FED is the
   ≥30 % reduction target).

The D-022 4-metric path-level set (``peak_danger``, ``time_in_hazard``,
``aset_margin``, ``fed_final``) remains useful as a *per-trajectory
diagnostic* — see :mod:`src.risk_map.path_metrics`. The integration
metrics here aggregate across the multi-agent population.

**Status: skeleton.** Aggregator + CSV writer pending. The dataclass
interface is stable so the scenario modules can already type-annotate
returns.
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ScenarioMetrics:
    """One row of the EXP-PATH-001 comparison table.

    Attributes:
        scenario_id: Logical scenario label (``"S1_fixed_sign"``,
            ``"S2_fds_swarm"``, ``"S3_fno_swarm"``).
        fire_scenario_id: FDS scenario name (e.g. ``"sim_1500kw_2m2_T05"``).
        seed: RNG seed used to scatter person start positions.
        n_persons: Number of persons in the run.
        evacuation_success_rate: ``evacuated / n_persons`` ∈ [0, 1].
        mean_evacuation_time_s: Mean exit time among evacuated persons.
            NaN if none evacuated.
        danger_zone_exposure_time_s: Mean per-person seconds with
            experienced danger > 0.5.
        casualty_rate: ``dead / n_persons`` ∈ [0, 1].
        cumulative_fed: Mean per-person final FED. H6 primary metric.
    """

    scenario_id: str
    fire_scenario_id: str
    seed: int
    n_persons: int
    evacuation_success_rate: float
    mean_evacuation_time_s: float
    danger_zone_exposure_time_s: float
    casualty_rate: float
    cumulative_fed: float
    max_cumulative_fed: float = 0.0
    """Worst-case (single-person max) FED at end of run (D-035, 2026-05-14).
    Captures the value that *matters most* for life-safety: the
    occupant with the highest FED. Drone swarms earn their value by
    pulling this maximum down even if the population mean changes
    little."""
    p90_cumulative_fed: float = 0.0
    """90th-percentile FED across the population. Less noisy than max
    for small N — at n_persons=20 this is the 2nd-worst person."""

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                d[k] = None
        return d

    def summary_line(self) -> str:
        """One-line console-friendly summary."""
        return (
            f"{self.scenario_id:<18} fire={self.fire_scenario_id:<22} "
            f"seed={self.seed:>2}  "
            f"evac={self.evacuation_success_rate*100:5.1f}%  "
            f"t_evac={self._fmt_nan(self.mean_evacuation_time_s, '.1f')}s  "
            f"exposure={self.danger_zone_exposure_time_s:5.1f}s  "
            f"dead={self.casualty_rate*100:5.1f}%  "
            f"FED(mean={self.cumulative_fed:.4f} "
            f"p90={self.p90_cumulative_fed:.4f} "
            f"max={self.max_cumulative_fed:.4f})"
        )

    @staticmethod
    def _fmt_nan(v: float, fmt: str) -> str:
        return "  NaN" if math.isnan(v) else f"{v:{fmt}}"


# ─── Collector ────────────────────────────────────────────────────────────
@dataclass
class MetricsCollector:
    """Per-run accumulator. Scenario modules push state at end of run.

    Typical usage::

        collector = MetricsCollector(scenario_id="S2_fds_swarm",
                                     fire_scenario_id="sim_1500kw_2m2_T05",
                                     seed=42)
        for person in persons:
            collector.add_person(
                evacuated=person.evacuated,
                dead=person.dead,
                exit_time=person.exit_time,
                exposure_time_s=person.exposure_time_s,
                cumulative_fed=person.cumulative_fed,
            )
        row = collector.finalize()
    """

    scenario_id: str
    fire_scenario_id: str
    seed: int
    _evacuated: List[bool] = field(default_factory=list)
    _dead: List[bool] = field(default_factory=list)
    _exit_times: List[float] = field(default_factory=list)
    _exposure: List[float] = field(default_factory=list)
    _feds: List[float] = field(default_factory=list)

    def add_person(
        self,
        evacuated: bool,
        dead: bool,
        exit_time: float,
        exposure_time_s: float,
        cumulative_fed: float,
    ) -> None:
        """Record one person's end-of-run state."""
        raise NotImplementedError(
            "Week 12 M5: append + light validation. Skeleton frozen for parallel work."
        )

    def finalize(self) -> ScenarioMetrics:
        """Compute the 5 aggregates and return a :class:`ScenarioMetrics`."""
        raise NotImplementedError(
            "Week 12 M5: aggregate via mean/fraction; NaN-handle empty buckets."
        )


# ─── CSV writer ───────────────────────────────────────────────────────────
def write_metrics_csv(rows: List[ScenarioMetrics], path: Path) -> None:
    """Dump a list of :class:`ScenarioMetrics` to ``path`` as CSV.

    Schema matches :class:`ScenarioMetrics` field order. Parent
    directories are created. Existing file is overwritten.

    Args:
        rows: Rows to write.
        path: Destination ``.csv``.
    """
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    field_order = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())


# ─── H6 verdict helper ────────────────────────────────────────────────────
def h6_verdict(rows: List[ScenarioMetrics]) -> dict:
    """Compute the H6 reduction ratio S2/S1 averaged over fire scenarios.

    Args:
        rows: All rows from a complete EXP-PATH-001 sweep.

    Returns:
        Dictionary with keys ``s1_mean_fed``, ``s2_mean_fed``,
        ``ratio`` (= s2/s1), ``passes_h6`` (= ratio ≤ 0.7), and
        ``s3_mean_fed`` if S3 rows present.

    Notes:
        Returns NaN for ``ratio`` when ``s1_mean_fed`` is zero.
    """
    def _mean_fed(label: str) -> float:
        feds = [r.cumulative_fed for r in rows if r.scenario_id == label]
        if not feds:
            return float("nan")
        return sum(feds) / len(feds)

    s1 = _mean_fed("S1_fixed_sign")
    s2 = _mean_fed("S2_fds_swarm")
    s3 = _mean_fed("S3_fno_swarm")
    ratio = s2 / s1 if (not math.isnan(s1) and s1 > 1e-9) else float("nan")
    passes = (not math.isnan(ratio)) and ratio <= 0.7
    out = {
        "s1_mean_fed": s1,
        "s2_mean_fed": s2,
        "ratio": ratio,
        "passes_h6": passes,
    }
    if not math.isnan(s3):
        out["s3_mean_fed"] = s3
    return out


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("metrics.py - skeleton (collector + finalize pending)")
    print("Probing dataclass shape and pure helpers...")
    sample = [
        ScenarioMetrics(
            scenario_id="S1_fixed_sign",
            fire_scenario_id="sim_1500kw_2m2_T05",
            seed=0,
            n_persons=20,
            evacuation_success_rate=0.85,
            mean_evacuation_time_s=42.3,
            danger_zone_exposure_time_s=12.7,
            casualty_rate=0.05,
            cumulative_fed=0.40,
        ),
        ScenarioMetrics(
            scenario_id="S2_fds_swarm",
            fire_scenario_id="sim_1500kw_2m2_T05",
            seed=0,
            n_persons=20,
            evacuation_success_rate=0.95,
            mean_evacuation_time_s=38.0,
            danger_zone_exposure_time_s=6.4,
            casualty_rate=0.00,
            cumulative_fed=0.24,
        ),
    ]
    for r in sample:
        print("  " + r.summary_line())
    verdict = h6_verdict(sample)
    print(f"  H6 verdict: {verdict}")
    if not verdict["passes_h6"]:
        # Sample numbers above are S2/S1 = 0.60 → passes.
        print("FAIL")
        raise SystemExit(1)

    # CSV round-trip
    out = Path("results/_smoke_metrics.csv")
    write_metrics_csv(sample, out)
    assert out.exists()
    print(f"  wrote sample CSV -> {out}")
    out.unlink()
    print("PASS (pure helpers verified; aggregator is still skeleton)")
