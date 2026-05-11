"""
Orchestrate raw FDS scenarios → processed HDF5 dataset.

Pipeline per scenario:

1. ``extract_slices(fds_dir)`` — raw ``.sf`` parsing into
   ``(31, 60, 40, 6)`` raw fields (°C, m, ppm).
2. ``normalize_scenario`` + ``build_input_tensor`` +
   ``build_target_tensor`` — assemble ``(31, 5, 60, 40, 6)`` input and
   ``(31, 3, 60, 40, 6)`` target tensors normalised to ``[0, 1]``.
3. Write into a single HDF5 file with the schema documented in
   ``data/README.md`` and consumed by :class:`src.dataset.FireDataset`.

Output schema::

    /scenario_NNN/input    (31, 5, 60, 40, 6) float32
    /scenario_NNN/target   (31, 3, 60, 40, 6) float32
    /mask                  (60, 40, 6) float32
    /metadata/train_indices  int32
    /metadata/val_indices    int32
    /metadata/ood_indices    int32
    /metadata/scenario_ids   variable-length UTF-8 strings (original IDs
                             from scenario_config.json, indexed positionally)

The mask is derived once from the first scenario's ``.fds`` deck.

Two modes:

* ``strict=True`` (default): every scenario in ``scenario_config.json``
  must have a present, well-formed FDS output directory. Missing or
  unreadable scenarios raise ``ValueError``.
* ``strict=False``: missing scenarios are skipped with a warning, and
  the split indices are filtered to include only successfully-processed
  scenarios. Useful for partial development builds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.mask_generator import (
    default_open_floor_mask,
    generate_mask_from_fds,
)
from src.data_pipeline.normalize import (
    build_input_and_target,
)
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

# Default config path produced by ``scripts/generate_scenarios.py``.
_DEFAULT_CONFIG_NAME: str = "scenario_config.json"


# ─── Per-scenario processing ─────────────────────────────────────────────
def _process_scenario(
    fds_dir: Path,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract + normalise one scenario directory into ``(input, target)``."""
    slices = extract_slices(fds_dir)
    return build_input_and_target(slices, mask, times=slices["times"])


# ─── Split derivation ────────────────────────────────────────────────────
def _split_indices_from_config(
    scenarios: Sequence[Dict[str, Any]],
    successful_positions: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Map config split labels → HDF5 positional indices.

    ``successful_positions`` is the list of positions (in config order)
    that were written to HDF5. Returned arrays use those positional
    indices (i.e. they match the ``scenario_{NNN:03d}`` ordering in the
    HDF5 file).
    """
    out = {"train": [], "val": [], "ood": []}
    for pos in successful_positions:
        split = scenarios[pos].get("split", "train")
        if split not in out:
            raise ValueError(f"unknown split label: {split!r}")
        out[split].append(pos)
    return {k: np.asarray(v, dtype=np.int32) for k, v in out.items()}


# ─── HDF5 writer ─────────────────────────────────────────────────────────
def _write_hdf5(
    output_path: Path,
    inputs: Dict[int, np.ndarray],
    targets: Dict[int, np.ndarray],
    mask: np.ndarray,
    splits: Dict[str, np.ndarray],
    original_ids: Sequence[str],
    compression: Optional[str] = "gzip",
) -> None:
    """Write the full HDF5 file in one go.

    Args:
        output_path: Destination ``.h5`` path.
        inputs:  ``{position: (31, 5, 60, 40, 6) float32}``.
        targets: ``{position: (31, 3, 60, 40, 6) float32}``.
        mask:    ``(60, 40, 6) float32``.
        splits:  ``{"train": idx, "val": idx, "ood": idx}``.
        original_ids: Original scenario IDs (``s_000``, ``s_val_0``…) in
            config order, used to record provenance.
        compression: HDF5 compression filter (``"gzip"`` or ``None``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with h5py.File(output_path, "w") as f:
        for pos in sorted(inputs):
            grp = f.create_group(f"scenario_{pos:03d}")
            grp.create_dataset("input", data=inputs[pos], compression=compression)
            grp.create_dataset("target", data=targets[pos], compression=compression)
            grp.attrs["original_id"] = str(original_ids[pos])

        f.create_dataset("mask", data=mask.astype(np.float32), compression=compression)

        meta = f.create_group("metadata")
        meta.create_dataset("train_indices", data=splits["train"])
        meta.create_dataset("val_indices", data=splits["val"])
        meta.create_dataset("ood_indices", data=splits["ood"])
        # Variable-length UTF-8 strings for original IDs
        str_dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset(
            "scenario_ids",
            data=np.asarray(list(original_ids), dtype=object),
            dtype=str_dt,
        )
        meta.attrs["n_scenarios"] = int(len(inputs))


# ─── Top-level builder ───────────────────────────────────────────────────
def build(
    raw_dir: Path,
    output_path: Path,
    config_path: Optional[Path] = None,
    seed: int = 42,
    strict: bool = True,
    mask_fallback_all_ones: bool = False,
    compression: Optional[str] = "gzip",
) -> Dict[str, Any]:
    """Build the processed HDF5 dataset from raw FDS outputs.

    Args:
        raw_dir: Directory containing ``scenario_config.json`` and the per-
            scenario subdirectories (``s_000/``, ``s_val_0/`` …).
        output_path: Destination ``.h5`` path (parent created if absent).
        config_path: Override path to scenario config JSON. Defaults to
            ``raw_dir / "scenario_config.json"``.
        seed: Currently unused (split assignment comes from the config),
            kept for forward compatibility with synthetic randomised splits.
        strict: If ``True``, raise on any missing scenario directory or
            failed extraction. If ``False``, skip such scenarios and emit
            a warning.
        mask_fallback_all_ones: If the first scenario's ``.fds`` cannot be
            parsed for OBSTs, use the all-fluid mask instead of raising.
        compression: HDF5 compression filter.

    Returns:
        Summary dict::

            {
                "output_path": Path,
                "n_written": int,
                "n_skipped": int,
                "splits": {"train": n, "val": n, "ood": n},
                "mask_solid_ratio": float,
                "missing_ids": list[str],
            }

    Raises:
        FileNotFoundError: ``raw_dir`` or ``config_path`` missing.
        ValueError: ``strict=True`` and any scenario fails to process.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    cfg_path = Path(config_path) if config_path else raw_dir / _DEFAULT_CONFIG_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"scenario config not found: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    scenarios = list(cfg.get("scenarios", []))
    if not scenarios:
        raise ValueError(f"{cfg_path}: no scenarios listed")

    original_ids = [s["id"] for s in scenarios]
    print(f"\nbuild_dataset.build: {len(scenarios)} scenarios from {cfg_path}")

    # ── Mask: derive from the first available scenario's .fds ──────────
    mask: Optional[np.ndarray] = None
    for s in scenarios:
        sdir = raw_dir / s["id"]
        if not sdir.is_dir():
            continue
        try:
            mask = generate_mask_from_fds(sdir)
            print(f"  mask from {sdir.name}: "
                  f"{(mask == 0.0).sum()} solid / {mask.size} cells")
            break
        except (FileNotFoundError, ValueError) as exc:
            print(f"  mask attempt {sdir.name} failed: {exc}")
    if mask is None:
        if mask_fallback_all_ones:
            print("  WARN: no .fds-derived mask available; using all-ones fallback")
            mask = default_open_floor_mask()
        else:
            raise ValueError(
                "could not derive mask from any scenario; "
                "pass mask_fallback_all_ones=True to use a default all-fluid mask"
            )

    # ── Per-scenario extraction + normalisation ────────────────────────
    inputs: Dict[int, np.ndarray] = {}
    targets: Dict[int, np.ndarray] = {}
    missing: List[str] = []

    for pos, s in enumerate(scenarios):
        sdir = raw_dir / s["id"]
        if not sdir.is_dir():
            msg = f"scenario {s['id']!r} directory missing at {sdir}"
            if strict:
                raise ValueError(msg)
            print(f"  SKIP: {msg}")
            missing.append(s["id"])
            continue
        try:
            inp, tgt = _process_scenario(sdir, mask)
        except Exception as exc:  # broad on purpose — many possible failures
            msg = f"scenario {s['id']!r} failed: {exc!r}"
            if strict:
                raise ValueError(msg) from exc
            print(f"  SKIP: {msg}")
            missing.append(s["id"])
            continue

        if inp.shape != (N_TIMESTEPS, N_INPUT_CHANNELS, *GRID_SHAPE):
            raise ValueError(
                f"{s['id']}: input shape {inp.shape} != "
                f"({N_TIMESTEPS}, {N_INPUT_CHANNELS}, *GRID_SHAPE)"
            )
        if tgt.shape != (N_TIMESTEPS, N_OUTPUT_CHANNELS, *GRID_SHAPE):
            raise ValueError(
                f"{s['id']}: target shape {tgt.shape} != "
                f"({N_TIMESTEPS}, {N_OUTPUT_CHANNELS}, *GRID_SHAPE)"
            )

        inputs[pos] = inp
        targets[pos] = tgt
        print(f"  [{pos:3d}/{len(scenarios)}] {s['id']:10s} "
              f"split={s.get('split', '?'):5s} OK")

    if not inputs:
        raise ValueError("no scenarios successfully processed")

    successful = sorted(inputs)
    splits = _split_indices_from_config(scenarios, successful)

    # ── Write HDF5 ─────────────────────────────────────────────────────
    _write_hdf5(
        output_path=output_path,
        inputs=inputs,
        targets=targets,
        mask=mask,
        splits=splits,
        original_ids=original_ids,
        compression=compression,
    )

    summary = {
        "output_path": Path(output_path),
        "n_written": len(inputs),
        "n_skipped": len(missing),
        "splits": {k: int(v.size) for k, v in splits.items()},
        "mask_solid_ratio": float((mask == 0.0).sum() / mask.size),
        "missing_ids": missing,
    }
    print(
        f"\nwrote {summary['n_written']} scenarios "
        f"(train={summary['splits']['train']}, "
        f"val={summary['splits']['val']}, "
        f"ood={summary['splits']['ood']}) "
        f"→ {output_path}"
    )
    if missing:
        print(f"  skipped: {missing}")
    return summary


# ─── CLI ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing scenario_config.json + per-scenario subdirs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dataset.h5"),
        help="Destination .h5 file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override scenario_config.json path.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Lenient mode: skip missing/failing scenarios rather than aborting.",
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable HDF5 gzip compression (faster writes, larger file).",
    )
    args = parser.parse_args()

    summary = build(
        raw_dir=args.raw,
        output_path=args.output,
        config_path=args.config,
        strict=not args.allow_missing,
        compression=None if args.no_compression else "gzip",
    )
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import tempfile

    # Self-test mode is the default when run with no arguments — building
    # against real FDS data only makes sense once 30 scenarios exist.
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    print("=" * 60)
    print("build_dataset.py self-test (synthetic in-memory scenarios)")
    print("=" * 60)

    errors: list[str] = []
    rng = np.random.default_rng(0)
    nx, ny, nz = GRID_SHAPE

    # Synthesize 6 scenarios (4 train / 1 val / 1 ood) with random data
    # written through fake FDS directories. Because building the FDS .sf
    # files is heavy, monkey-patch ``_process_scenario`` to bypass extraction.
    n_test = 6
    cfg = {
        "version": 1,
        "total": n_test,
        "scenarios": (
            [{"id": f"s_{i:03d}", "split": "train"} for i in range(4)]
            + [{"id": "s_val_0", "split": "val"}]
            + [{"id": "s_ood_0", "split": "ood"}]
        ),
    }

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp) / "raw"
        raw_dir.mkdir()
        out_path = Path(tmp) / "dataset.h5"
        # Each scenario gets an empty directory + a stub .fds containing
        # a single wall so mask generation works without real FDS data.
        wall = (
            "&HEAD CHID='STUB' /\n"
            "&OBST XB=14.5,15.0, 0.0,20.0, 0.0,3.0 /\n"
            "&TAIL /\n"
        )
        for s in cfg["scenarios"]:
            sd = raw_dir / s["id"]
            sd.mkdir()
            (sd / "stub.fds").write_text(wall, encoding="utf-8")

        (raw_dir / _DEFAULT_CONFIG_NAME).write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Monkey-patch _process_scenario so we don't need real .sf files.
        import src.data_pipeline.build_dataset as bdm

        def _fake_process(fds_dir: Path, mask: np.ndarray):
            inp = rng.uniform(
                0.0, 1.0, (N_TIMESTEPS, N_INPUT_CHANNELS, nx, ny, nz)
            ).astype(np.float32)
            # Embed mask + linear time encoding so we can verify them.
            inp[:, 3, :, :, :] = mask
            te = np.linspace(0.0, 1.0, N_TIMESTEPS, dtype=np.float32)
            inp[:, 4, :, :, :] = te[:, None, None, None]
            tgt = inp[:, :3, :, :, :].copy()
            return inp, tgt

        original = bdm._process_scenario
        bdm._process_scenario = _fake_process
        try:
            summary = bdm.build(
                raw_dir=raw_dir,
                output_path=out_path,
                strict=True,
                compression=None,
            )
        finally:
            bdm._process_scenario = original

        # ── HDF5 inspection ─────────────────────────────────────────────
        with h5py.File(out_path, "r") as f:
            scenario_keys = sorted(k for k in f.keys() if k.startswith("scenario_"))
            if len(scenario_keys) != n_test:
                errors.append(f"expected {n_test} scenarios in HDF5, got {len(scenario_keys)}")
            for k in scenario_keys:
                inp_shape = f[k]["input"].shape
                tgt_shape = f[k]["target"].shape
                if inp_shape != (N_TIMESTEPS, N_INPUT_CHANNELS, nx, ny, nz):
                    errors.append(f"{k}/input shape {inp_shape}")
                if tgt_shape != (N_TIMESTEPS, N_OUTPUT_CHANNELS, nx, ny, nz):
                    errors.append(f"{k}/target shape {tgt_shape}")
            mask_shape = f["mask"].shape
            if mask_shape != GRID_SHAPE:
                errors.append(f"mask shape {mask_shape}")
            train_idx = f["metadata/train_indices"][...]
            val_idx = f["metadata/val_indices"][...]
            ood_idx = f["metadata/ood_indices"][...]
            if train_idx.tolist() != [0, 1, 2, 3]:
                errors.append(f"train indices {train_idx.tolist()}")
            if val_idx.tolist() != [4]:
                errors.append(f"val indices {val_idx.tolist()}")
            if ood_idx.tolist() != [5]:
                errors.append(f"ood indices {ood_idx.tolist()}")
            # Original IDs preserved
            ids = [s.decode() if isinstance(s, bytes) else s
                   for s in f["metadata/scenario_ids"][...]]
            if ids != [s["id"] for s in cfg["scenarios"]]:
                errors.append(f"scenario_ids drift: {ids}")
            # Mask should mark cell column ix=29 as solid (wall at x=14.5..15)
            mask_arr = f["mask"][...]
            if mask_arr[29, 10, 3] != 0.0:
                errors.append("mask wall ix=29 not solid")
            print(f"  HDF5 file size: {out_path.stat().st_size / 1024:.1f} KB")
            print(f"  scenarios:      {len(scenario_keys)}")
            print(f"  splits:         train={train_idx.tolist()} val={val_idx.tolist()} ood={ood_idx.tolist()}")
            print(f"  mask solid:     {(mask_arr == 0.0).sum()} cells")

        print(f"\n  summary: {summary}")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: build_dataset validated")
