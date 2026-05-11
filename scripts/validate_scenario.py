"""
Validate one (or many) FDS scenario output directories.

Runs the post-simulation sanity checks called out in
``docs/manual_v2.md`` Week 5 and ``docs/lessons_learned.md`` (L-001, L-009):

1. ``fdsreader.Simulation`` loads the directory without raising.
2. Exactly three SLCF slices are present (Temperature / Visibility / CO).
3. Each ``slc.to_global(return_coordinates=True)`` call succeeds.
4. Each grid has shape ``(N_TIMESTEPS, 60, 40, 6)``.
5. The returned coords dict passes
   :func:`src.shared.coordinates.verify_fdsreader_coords`.
6. No NaN or Inf in any slice.
7. Mean initial temperature is in [18, 25] °C (typical TMPA=22 °C).
8. After ~30 s the maximum temperature has risen above 30 °C — i.e. the
   fire actually ignited.

The validator is structured so it can be exercised with synthetic
``ValidationResult`` instances even before any real FDS data lands;
``fdsreader`` is imported lazily so missing data does not break this file.

Usage::

    python scripts/validate_scenario.py data/raw/s_000/        # single
    python scripts/validate_scenario.py data/raw/              # all under tree
    python scripts/validate_scenario.py data/raw/ --json results.json
    python scripts/validate_scenario.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.shared.constants import DT_SLCF, GRID_SHAPE, N_TIMESTEPS
from src.shared.coordinates import verify_fdsreader_coords

_INIT_TEMP_MIN_C: float = 18.0
_INIT_TEMP_MAX_C: float = 25.0
_FIRE_RISE_C: float = 30.0
_FIRE_RISE_BY_S: float = 30.0


@dataclass
class ValidationResult:
    """Outcome of validating a single scenario directory."""

    scenario_id: str
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        """Record an error and flip ``passed`` to ``False``."""
        self.passed = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        """Record a non-fatal warning."""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation."""
        return asdict(self)


# ─── Validation primitives ─────────────────────────────────────────────────
def validate_scenario(fds_dir: Path) -> ValidationResult:
    """Validate a single scenario directory and return a structured result.

    Args:
        fds_dir: Directory containing the ``*.smv`` and ``*.sf`` outputs.

    Returns:
        :class:`ValidationResult`. ``passed=False`` indicates at least one
        check failed — see ``errors`` for diagnostics.
    """
    fds_dir = Path(fds_dir)
    result = ValidationResult(scenario_id=fds_dir.name)

    if not fds_dir.exists():
        result.fail(f"directory not found: {fds_dir}")
        return result
    if not fds_dir.is_dir():
        result.fail(f"not a directory: {fds_dir}")
        return result

    smv_files = sorted(fds_dir.glob("*.smv"))
    if not smv_files:
        result.fail(f"no .smv file in {fds_dir} — simulation may not have produced output")
        return result
    if len(smv_files) > 1:
        result.warn(
            f"multiple .smv files in {fds_dir.name}: {[s.name for s in smv_files]}"
        )

    # Lazy import so missing fdsreader doesn't break the module import path.
    try:
        import fdsreader  # type: ignore[import-not-found]
    except ImportError as exc:
        result.fail(f"fdsreader not installed: {exc}")
        return result

    try:
        sim = fdsreader.Simulation(str(fds_dir))
    except Exception as exc:  # broad on purpose — fdsreader exposes many failure modes
        result.fail(f"fdsreader.Simulation failed: {exc!r}")
        return result

    slices = list(sim.slices)
    result.info["n_slices"] = len(slices)
    if len(slices) != 3:
        result.fail(
            f"expected 3 SLCF (T/V/CO), found {len(slices)}: "
            f"{[s.quantity for s in slices]}"
        )
        return result

    # Match by quantity rather than positional order — PyroSim emits in
    # whatever order it likes.
    by_quantity = {str(getattr(s, "quantity", "")).upper(): s for s in slices}
    expected_quantities = ["TEMPERATURE", "SOOT VISIBILITY", "CARBON MONOXIDE VOLUME FRACTION"]
    grids: Dict[str, np.ndarray] = {}
    coords_first: Dict[str, np.ndarray] = {}
    for needle in expected_quantities:
        slc = next(
            (s for q, s in by_quantity.items() if needle in q.upper()),
            None,
        )
        if slc is None:
            result.fail(f"missing SLCF quantity '{needle}'")
            return result
        try:
            grid, coords = slc.to_global(return_coordinates=True)
        except Exception as exc:
            result.fail(
                f"to_global() failed for {needle}: {exc!r}. "
                "Common causes: VECTOR=.TRUE. (L-001) or SLCF Z=3.5 (L-009)."
            )
            return result

        # Time alignment per coordinate_convention.md ─ FDS times may drift
        # slightly off the canonical 0, 10, …, 300 s grid.
        try:
            times = np.asarray(slc.times, dtype=np.float64)
            target_times = np.arange(0.0, 300.0 + DT_SLCF / 2, DT_SLCF)
            indices = np.clip(
                np.searchsorted(times, target_times), 0, len(times) - 1
            )
            grid = np.asarray(grid)[indices]
        except Exception as exc:
            result.warn(f"time alignment fallback for {needle}: {exc!r}")
            grid = np.asarray(grid)

        expected_shape = (N_TIMESTEPS, *GRID_SHAPE)
        if grid.shape != expected_shape:
            result.fail(
                f"{needle}: shape {grid.shape} != expected {expected_shape}. "
                "Common causes: SLCF Z range != [0, 3] (L-009)."
            )
            return result
        grids[needle] = grid
        if not coords_first:
            coords_first = coords

    # ── Coordinate sanity ──────────────────────────────────────────────────
    if not verify_fdsreader_coords(coords_first):
        result.fail("verify_fdsreader_coords rejected the SLCF coordinate grid")
        return result

    # ── NaN / Inf ──────────────────────────────────────────────────────────
    for name, g in grids.items():
        if not np.isfinite(g).all():
            n_bad = int((~np.isfinite(g)).sum())
            result.fail(f"{name}: contains {n_bad} non-finite cells")

    # ── Initial temperature ────────────────────────────────────────────────
    temp_grid = grids["TEMPERATURE"]
    t_init_mean = float(np.nanmean(temp_grid[0]))
    result.info["t_init_mean_c"] = round(t_init_mean, 3)
    if not (_INIT_TEMP_MIN_C <= t_init_mean <= _INIT_TEMP_MAX_C):
        result.fail(
            f"initial temperature mean {t_init_mean:.2f} °C outside "
            f"[{_INIT_TEMP_MIN_C}, {_INIT_TEMP_MAX_C}] — check &MISC TMPA or initial conditions"
        )

    # ── Fire ignition check (max T at t=30s) ───────────────────────────────
    rise_idx = int(round(_FIRE_RISE_BY_S / DT_SLCF))
    rise_idx = min(rise_idx, N_TIMESTEPS - 1)
    t_30_max = float(np.nanmax(temp_grid[rise_idx]))
    result.info["t_max_at_30s_c"] = round(t_30_max, 3)
    if t_30_max < _FIRE_RISE_C:
        result.fail(
            f"max temperature at t={rise_idx * DT_SLCF:.0f}s = {t_30_max:.2f} °C — "
            "fire does not appear to have ignited"
        )

    result.info["grid_shape"] = list(temp_grid.shape)
    return result


def validate_all(raw_dir: Path) -> Dict[str, ValidationResult]:
    """Validate every direct child directory of ``raw_dir`` that contains a ``.smv``.

    Args:
        raw_dir: Directory whose children are scenario folders.

    Returns:
        ``{scenario_id: ValidationResult}`` in sorted order.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        empty = ValidationResult(scenario_id=str(raw_dir))
        empty.fail(f"raw_dir not found: {raw_dir}")
        return {empty.scenario_id: empty}

    scenario_dirs = sorted(
        d for d in raw_dir.iterdir()
        if d.is_dir() and any(d.glob("*.smv"))
    )
    if not scenario_dirs:
        empty = ValidationResult(scenario_id=str(raw_dir))
        empty.fail(f"no scenario subdirectories (with .smv) under {raw_dir}")
        return {empty.scenario_id: empty}

    return {d.name: validate_scenario(d) for d in scenario_dirs}


# ─── Reporting ────────────────────────────────────────────────────────────
def print_summary(results: Dict[str, ValidationResult]) -> None:
    """Print a one-line-per-scenario summary plus an aggregate footer."""
    if not results:
        print("(no scenarios to report)")
        return

    n_pass = 0
    n_fail = 0
    for sid, r in sorted(results.items()):
        status = "✓ PASS" if r.passed else "✗ FAIL"
        info_bits: List[str] = []
        if "grid_shape" in r.info:
            info_bits.append(f"shape={tuple(r.info['grid_shape'])}")
        if "t_init_mean_c" in r.info:
            info_bits.append(f"T_init={r.info['t_init_mean_c']:.1f}°C")
        info_str = "  ".join(info_bits)

        print(f"  {sid:14s}  {status}  {info_str}")
        for err in r.errors:
            print(f"      ERROR: {err}")
        for warn in r.warnings:
            print(f"      WARN : {warn}")

        if r.passed:
            n_pass += 1
        else:
            n_fail += 1

    total = n_pass + n_fail
    print("  " + "-" * 60)
    print(f"  Passed: {n_pass}/{total}")
    if n_fail:
        print(f"  Failed: {n_fail} scenarios — see errors above")


# ─── Self-test ─────────────────────────────────────────────────────────────
def _run_self_test() -> int:
    """Structural self-test — does not require any real FDS data."""
    print("=" * 60)
    print("validate_scenario.py self-test")
    print("=" * 60)

    errors: List[str] = []

    # ── 1. Non-existent path → passed=False ───────────────────────────────
    print("\n[1] Non-existent directory yields passed=False")
    bogus = validate_scenario(Path("/definitely/does/not/exist/__nope__"))
    print(f"  passed = {bogus.passed}  errors = {bogus.errors[:1]}")
    if bogus.passed:
        errors.append("expected passed=False for missing directory")
    if not bogus.errors:
        errors.append("expected at least one error for missing directory")

    # ── 2. Directory exists but no .smv → passed=False ─────────────────────
    print("\n[2] Directory without .smv yields passed=False")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        empty = validate_scenario(Path(tmp))
        print(f"  passed = {empty.passed}  errors = {empty.errors[:1]}")
        if empty.passed:
            errors.append("expected passed=False for empty directory")

    # ── 3. ValidationResult JSON round-trip ────────────────────────────────
    print("\n[3] ValidationResult JSON serialisation")
    r = ValidationResult(scenario_id="dummy")
    r.warn("informational only")
    r.info["t_init_mean_c"] = 21.7
    payload = json.dumps(r.to_dict())
    parsed = json.loads(payload)
    if parsed["scenario_id"] != "dummy":
        errors.append("scenario_id did not round-trip through JSON")
    if parsed["passed"] is not True:
        errors.append("default passed should be True")
    if parsed["warnings"] != ["informational only"]:
        errors.append("warnings did not round-trip")
    if parsed["info"]["t_init_mean_c"] != 21.7:
        errors.append("info dict did not round-trip")

    # ── 4. print_summary on empty dict ─────────────────────────────────────
    print("\n[4] print_summary handles an empty mapping")
    print_summary({})  # must not raise

    # ── 5. print_summary on a synthetic mixed result set ───────────────────
    print("\n[5] print_summary on synthetic results")
    good = ValidationResult(scenario_id="s_000")
    good.info["grid_shape"] = [31, 60, 40, 6]
    good.info["t_init_mean_c"] = 22.0
    bad = ValidationResult(scenario_id="s_001")
    bad.fail("shape mismatch (61,41,9)")
    print_summary({"s_000": good, "s_001": bad})

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nPASS: validator structure verified")
    return 0


# ─── CLI entry ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        help="Either a single scenario directory or the root containing many.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="If provided, write the per-scenario results dict to this JSON file.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the structural self-test and exit (no FDS data required).",
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(_run_self_test())
    if args.target is None:
        parser.error("target path is required (or use --self-test)")

    target = args.target
    if target.is_file() or any(target.glob("*.smv")):
        # Single scenario directory.
        results = {target.name: validate_scenario(target)}
    else:
        results = validate_all(target)

    print_summary(results)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {sid: r.to_dict() for sid, r in results.items()}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nresults JSON → {args.json}")

    n_failed = sum(1 for r in results.values() if not r.passed)
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
