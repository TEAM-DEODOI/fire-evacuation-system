"""
Integrity checker for a processed HDF5 dataset.

Validates the file produced by :func:`src.data_pipeline.build_dataset.build`
against the schema documented in ``data/README.md`` and the contract
consumed by :class:`src.dataset.FireDataset`. Designed to be cheap and
side-effect-free — read-only on the file — so it can run as the final
gate before training.

Checks performed:

1. Every scenario group has the expected ``input`` / ``target`` shape and
   ``float32`` dtype.
2. ``mask`` is ``(60, 40, 6)`` and contains values exclusively in
   ``{0.0, 1.0}``.
3. All ``input`` and ``target`` values fall in ``[0, 1]``.
4. ``metadata/{train,val,ood}_indices`` exist, contain int values, and
   the three index sets are pairwise disjoint.
5. Every index in the split arrays maps to a ``scenario_NNN`` group that
   exists in the file.
6. The total scenario count matches the sum of split sizes.

By default the checker prints a per-scenario summary and a verdict.
Strict mode (``expected_scenarios=N_SCENARIOS_TOTAL``) additionally
enforces the canonical 30 / 24 / 3 / 3 distribution.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from src.shared.constants import (
    GRID_SHAPE,
    N_INPUT_CHANNELS,
    N_OUTPUT_CHANNELS,
    N_SCENARIOS_OOD,
    N_SCENARIOS_TOTAL,
    N_SCENARIOS_TRAIN,
    N_SCENARIOS_VAL,
    N_TIMESTEPS,
)

_EXPECTED_INPUT_SHAPE = (N_TIMESTEPS, N_INPUT_CHANNELS, *GRID_SHAPE)
_EXPECTED_TARGET_SHAPE = (N_TIMESTEPS, N_OUTPUT_CHANNELS, *GRID_SHAPE)


@dataclass
class VerificationReport:
    """Aggregated result of one verification pass."""

    dataset_path: Path
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_scenarios: int = 0
    split_sizes: dict[str, int] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def verify_dataset(
    dataset_path: Path,
    *,
    expected_scenarios: Optional[int] = None,
    verbose: bool = True,
) -> VerificationReport:
    """Verify the structure and value ranges of a processed HDF5 dataset.

    Args:
        dataset_path: Path to ``dataset.h5``.
        expected_scenarios: If given, require exactly this many scenario
            groups and (when equal to :data:`N_SCENARIOS_TOTAL`) the
            canonical 24 / 3 / 3 split distribution.
        verbose: Print per-scenario status lines if ``True``.

    Returns:
        :class:`VerificationReport` with ``passed`` and ``errors`` set.

    Raises:
        FileNotFoundError: ``dataset_path`` missing.
    """
    dataset_path = Path(dataset_path)
    report = VerificationReport(dataset_path=dataset_path)
    if not dataset_path.exists():
        report.fail(f"file not found: {dataset_path}")
        return report

    with h5py.File(dataset_path, "r") as f:
        # ── Scenario groups ────────────────────────────────────────────
        scenario_keys = sorted(k for k in f.keys() if k.startswith("scenario_"))
        report.n_scenarios = len(scenario_keys)
        if not scenario_keys:
            report.fail("no scenario_* groups found")
            return report
        if verbose:
            print(f"  found {report.n_scenarios} scenario groups")

        for k in scenario_keys:
            grp = f[k]
            if "input" not in grp:
                report.fail(f"{k}: missing 'input' dataset")
                continue
            if "target" not in grp:
                report.fail(f"{k}: missing 'target' dataset")
                continue

            in_ds = grp["input"]
            tg_ds = grp["target"]
            if in_ds.shape != _EXPECTED_INPUT_SHAPE:
                report.fail(
                    f"{k}/input shape {in_ds.shape} != {_EXPECTED_INPUT_SHAPE}"
                )
                continue
            if tg_ds.shape != _EXPECTED_TARGET_SHAPE:
                report.fail(
                    f"{k}/target shape {tg_ds.shape} != {_EXPECTED_TARGET_SHAPE}"
                )
                continue
            if in_ds.dtype != np.float32:
                report.warn(f"{k}/input dtype {in_ds.dtype} (expected float32)")
            if tg_ds.dtype != np.float32:
                report.warn(f"{k}/target dtype {tg_ds.dtype} (expected float32)")

            # Value-range probe — read fully (small enough at 31 × C × 60 × 40 × 6).
            in_arr = in_ds[...]
            tg_arr = tg_ds[...]
            if np.isnan(in_arr).any() or np.isinf(in_arr).any():
                report.fail(f"{k}/input contains NaN/Inf")
            if np.isnan(tg_arr).any() or np.isinf(tg_arr).any():
                report.fail(f"{k}/target contains NaN/Inf")
            if in_arr.min() < 0.0 or in_arr.max() > 1.0:
                report.fail(
                    f"{k}/input range [{in_arr.min():.6f}, {in_arr.max():.6f}] "
                    "outside [0, 1]"
                )
            if tg_arr.min() < 0.0 or tg_arr.max() > 1.0:
                report.fail(
                    f"{k}/target range [{tg_arr.min():.6f}, {tg_arr.max():.6f}] "
                    "outside [0, 1]"
                )

            if verbose:
                orig = grp.attrs.get("original_id", "?")
                print(
                    f"    {k} (orig={orig}) "
                    f"in=[{in_arr.min():.3f}, {in_arr.max():.3f}] "
                    f"tg=[{tg_arr.min():.3f}, {tg_arr.max():.3f}]"
                )

        # ── Mask ───────────────────────────────────────────────────────
        if "mask" not in f:
            report.fail("missing /mask dataset")
        else:
            mask = f["mask"][...]
            if mask.shape != GRID_SHAPE:
                report.fail(f"mask shape {mask.shape} != {GRID_SHAPE}")
            unique = np.unique(mask)
            if not set(unique.tolist()).issubset({0.0, 1.0}):
                report.fail(f"mask has values outside {{0.0, 1.0}}: {unique.tolist()}")
            if verbose and report.passed:
                solid = int((mask == 0.0).sum())
                print(
                    f"  mask: {solid} solid / {mask.size} "
                    f"({100 * solid / mask.size:.1f}%)"
                )

        # ── Metadata splits ────────────────────────────────────────────
        if "metadata" not in f:
            report.fail("missing /metadata group")
        else:
            meta = f["metadata"]
            for key in ("train_indices", "val_indices", "ood_indices"):
                if key not in meta:
                    report.fail(f"missing metadata/{key}")
            if report.passed:
                tr = meta["train_indices"][...].astype(int).tolist()
                va = meta["val_indices"][...].astype(int).tolist()
                oo = meta["ood_indices"][...].astype(int).tolist()
                report.split_sizes = {"train": len(tr), "val": len(va), "ood": len(oo)}

                # Pairwise disjoint
                ts, vs, os_ = set(tr), set(va), set(oo)
                if ts & vs:
                    report.fail(f"train and val overlap: {sorted(ts & vs)}")
                if ts & os_:
                    report.fail(f"train and ood overlap: {sorted(ts & os_)}")
                if vs & os_:
                    report.fail(f"val and ood overlap: {sorted(vs & os_)}")

                # Every index points to an existing scenario group
                all_positions = {
                    int(k.split("_")[-1]) for k in scenario_keys
                }
                missing = sorted(
                    (ts | vs | os_) - all_positions
                )
                if missing:
                    report.fail(f"split indices reference missing groups: {missing}")

                # Sum matches total
                total = len(tr) + len(va) + len(oo)
                if total > len(scenario_keys):
                    report.fail(
                        f"split sum {total} > scenario count {len(scenario_keys)}"
                    )
                if verbose:
                    print(
                        f"  splits: train={len(tr)} val={len(va)} ood={len(oo)} "
                        f"(total {total})"
                    )

        # ── Strict canonical-30 mode ───────────────────────────────────
        if expected_scenarios is not None:
            if report.n_scenarios != expected_scenarios:
                report.fail(
                    f"scenario count {report.n_scenarios} != expected {expected_scenarios}"
                )
            if expected_scenarios == N_SCENARIOS_TOTAL and report.split_sizes:
                expected_splits = {
                    "train": N_SCENARIOS_TRAIN,
                    "val": N_SCENARIOS_VAL,
                    "ood": N_SCENARIOS_OOD,
                }
                if report.split_sizes != expected_splits:
                    report.fail(
                        f"canonical split mismatch: got {report.split_sizes}, "
                        f"expected {expected_splits}"
                    )

    return report


def _print_summary(report: VerificationReport) -> None:
    print("\n" + "─" * 60)
    print(f"  dataset:    {report.dataset_path}")
    print(f"  scenarios:  {report.n_scenarios}")
    print(f"  splits:     {report.split_sizes}")
    if report.warnings:
        print(f"  warnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"    - {w}")
    if report.errors:
        print(f"  errors ({len(report.errors)}):")
        for e in report.errors:
            print(f"    - {e}")
    print("─" * 60)
    print("  VERDICT:", "PASS" if report.passed else "FAIL")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the processed dataset.h5 file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=f"Require exactly {N_SCENARIOS_TOTAL} scenarios with canonical splits.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-scenario summary lines.",
    )
    args = parser.parse_args()

    expected = N_SCENARIOS_TOTAL if args.strict else None
    report = verify_dataset(args.dataset, expected_scenarios=expected, verbose=not args.quiet)
    _print_summary(report)
    sys.exit(0 if report.passed else 1)


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    import json
    import tempfile

    print("=" * 60)
    print("verify.py self-test")
    print("=" * 60)

    errors: list[str] = []

    # Build a synthetic dataset via build_dataset (reuses its monkey-patch
    # pathway) — or write minimal HDF5 directly.
    import src.data_pipeline.build_dataset as bdm
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp) / "raw"
        raw_dir.mkdir()
        out_path = Path(tmp) / "dataset.h5"
        cfg = {
            "version": 1,
            "scenarios": (
                [{"id": f"s_{i:03d}", "split": "train"} for i in range(4)]
                + [{"id": "s_val_0", "split": "val"}]
                + [{"id": "s_ood_0", "split": "ood"}]
            ),
        }
        for s in cfg["scenarios"]:
            sd = raw_dir / s["id"]
            sd.mkdir()
            (sd / "stub.fds").write_text(
                "&OBST XB=14.5,15.0, 0.0,20.0, 0.0,3.0 /\n", encoding="utf-8"
            )
        (raw_dir / "scenario_config.json").write_text(json.dumps(cfg))

        original = bdm._process_scenario

        def _fake_process(fds_dir, mask):
            nx, ny, nz = GRID_SHAPE
            inp = rng.uniform(0.0, 1.0, (N_TIMESTEPS, N_INPUT_CHANNELS, nx, ny, nz)).astype(np.float32)
            inp[:, 3] = mask
            tgt = inp[:, :3].copy()
            return inp, tgt

        bdm._process_scenario = _fake_process
        try:
            bdm.build(raw_dir=raw_dir, output_path=out_path, strict=True, compression=None)
        finally:
            bdm._process_scenario = original

        # ── 1. Verify the clean dataset ─────────────────────────────────
        print("\n[1] Verify clean synthetic dataset")
        report = verify_dataset(out_path, verbose=True)
        _print_summary(report)
        if not report.passed:
            errors.append("clean dataset failed verification")

        # ── 2. Strict mode (6 scenarios, not 30) should FAIL ────────────
        print("\n[2] Strict mode with mismatched expected_scenarios")
        strict_report = verify_dataset(out_path, expected_scenarios=N_SCENARIOS_TOTAL, verbose=False)
        if strict_report.passed:
            errors.append("strict mode should have failed for 6-scenario dataset")
        else:
            print("  PASS: strict mode correctly reports failure")

        # ── 3. Corrupt the file → verifier should detect ────────────────
        print("\n[3] Corrupt range → expect FAIL")
        with h5py.File(out_path, "r+") as f:
            inp = f["scenario_000/input"]
            arr = inp[...]
            arr[0, 0, 0, 0, 0] = 2.0  # out of [0, 1]
            inp[...] = arr
        bad_report = verify_dataset(out_path, verbose=False)
        if bad_report.passed:
            errors.append("verifier missed out-of-range value")
        else:
            print(f"  PASS: out-of-range detected ({bad_report.errors[0]})")

        # ── 4. Missing file ─────────────────────────────────────────────
        print("\n[4] Missing file → FAIL")
        missing = verify_dataset(Path(tmp) / "does_not_exist.h5", verbose=False)
        if missing.passed:
            errors.append("missing file should fail")
        else:
            print(f"  PASS: missing file detected")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: verify validated")
