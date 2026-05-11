"""
Generate a synthetic HDF5 dataset matching the production schema.

Used to exercise the training pipeline end-to-end before real FDS data
arrives. The generated file follows ``data/README.md`` exactly:

    /scenario_NNN/input    (31, 5, 60, 40, 6)  float32  ∈ [0, 1]
    /scenario_NNN/target   (31, 3, 60, 40, 6)  float32  ∈ [0, 1]
    /mask                  (60, 40, 6)         float32  (1=fluid, 0=solid)
    /metadata/train_indices  int32
    /metadata/val_indices    int32
    /metadata/ood_indices    int32

In ``learnable=True`` mode (default) the target is a slightly-noised copy
of the input's first three channels, so a model with sufficient capacity
can actually drive the loss down — useful for sanity checking that the
trainer / dataset / model pipeline composes correctly. Set
``learnable=False`` for pure i.i.d. noise.

Usage::

    python scripts/build_synthetic_dataset.py
    python scripts/build_synthetic_dataset.py --output data/processed/synthetic.h5 --n-scenarios 6
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

from src.shared.constants import (
    DT_SLCF,
    GRID_SHAPE,
    N_INPUT_CHANNELS,
    N_OUTPUT_CHANNELS,
    N_SCENARIOS_OOD,
    N_SCENARIOS_TOTAL,
    N_SCENARIOS_TRAIN,
    N_SCENARIOS_VAL,
    N_TIMESTEPS,
    T_END_SECONDS,
)


def _split_sizes(n_scenarios: int) -> Tuple[int, int, int]:
    """Pick split sizes for ``n_scenarios``.

    For the canonical 30-scenario case use the project constants
    (24/3/3). For smaller test datasets, scale down proportionally
    while guaranteeing each split has at least one scenario.
    """
    if n_scenarios == N_SCENARIOS_TOTAL:
        return N_SCENARIOS_TRAIN, N_SCENARIOS_VAL, N_SCENARIOS_OOD
    if n_scenarios < 3:
        raise ValueError(f"need at least 3 scenarios, got {n_scenarios}")
    val = max(1, n_scenarios // 10)
    ood = max(1, n_scenarios // 10)
    train = n_scenarios - val - ood
    if train < 1:
        raise ValueError(f"could not allocate train split for {n_scenarios} scenarios")
    return train, val, ood


def _build_mask(rng: np.random.Generator) -> np.ndarray:
    """Construct a fluid/solid mask for the synthetic file.

    Default = all fluid (real building mask is not yet known — depends on
    PyroSim maze). A couple of small "walls" are punched out so the mask
    actually exercises the masked-loss code path.
    """
    nx, ny, nz = GRID_SHAPE
    mask = np.ones((nx, ny, nz), dtype=np.float32)
    # Two small interior obstacles. Positions are arbitrary but stable
    # across runs (they don't depend on rng, so different scenarios share
    # the same geometry as in the real dataset).
    mask[10:20, 5:10, :] = 0.0
    mask[30:40, 30:35, :] = 0.0
    return mask


def _build_scenario(
    mask: np.ndarray,
    rng: np.random.Generator,
    learnable: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one ``(input, target)`` pair for a single scenario."""
    nx, ny, nz = GRID_SHAPE

    # Input: 5 channels [T, V, CO, mask, time_enc]
    inp = rng.uniform(0.0, 1.0, size=(N_TIMESTEPS, N_INPUT_CHANNELS, nx, ny, nz)).astype(np.float32)

    # Channel 3 = building mask (constant across time).
    inp[:, 3, :, :, :] = mask

    # Channel 4 = linear time encoding t/T_END (broadcast spatially).
    times = (np.arange(N_TIMESTEPS, dtype=np.float32) * DT_SLCF) / T_END_SECONDS
    inp[:, 4, :, :, :] = times[:, None, None, None]

    # Target: 3 channels [T, V, CO]
    if learnable:
        # Small perturbation of the input's first three channels — gives the
        # model an attainable target so over-fit tests actually converge.
        target = inp[:, :N_OUTPUT_CHANNELS, :, :, :].copy()
        target += 0.01 * rng.standard_normal(target.shape).astype(np.float32)
        np.clip(target, 0.0, 1.0, out=target)
    else:
        target = rng.uniform(0.0, 1.0, size=(N_TIMESTEPS, N_OUTPUT_CHANNELS, nx, ny, nz)).astype(np.float32)

    return inp, target


def build_synthetic_dataset(
    output_path: Path,
    n_scenarios: int = N_SCENARIOS_TOTAL,
    seed: int = 42,
    learnable: bool = True,
    compression: str = "gzip",
) -> None:
    """Write a synthetic HDF5 file at ``output_path``.

    Args:
        output_path: Destination ``.h5``. Parent directory created if needed.
        n_scenarios: Total scenarios (default 30, matching real dataset).
        seed: RNG seed for reproducibility.
        learnable: If True, target = input[:, :3] + small noise. Otherwise
            target is i.i.d. uniform.
        compression: HDF5 compression filter (``"gzip"`` or ``None``).

    Raises:
        ValueError: If ``n_scenarios < 3`` (cannot allocate three splits).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_train, n_val, n_ood = _split_sizes(n_scenarios)
    print(
        f"  scenarios = {n_scenarios}  (train/val/ood = {n_train}/{n_val}/{n_ood})"
    )
    print(f"  output    = {output_path}")
    print(f"  learnable = {learnable}")

    mask = _build_mask(rng)
    perm = rng.permutation(n_scenarios)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    ood_idx = perm[n_train + n_val :]

    with h5py.File(output_path, "w") as f:
        for sid in range(n_scenarios):
            inp, target = _build_scenario(mask, rng, learnable=learnable)
            grp = f.create_group(f"scenario_{sid:03d}")
            grp.create_dataset("input", data=inp, compression=compression)
            grp.create_dataset("target", data=target, compression=compression)

        f.create_dataset("mask", data=mask, compression=compression)

        meta = f.create_group("metadata")
        meta.create_dataset("train_indices", data=train_idx.astype(np.int32))
        meta.create_dataset("val_indices", data=val_idx.astype(np.int32))
        meta.create_dataset("ood_indices", data=ood_idx.astype(np.int32))
        meta.attrs["synthetic"] = True
        meta.attrs["learnable"] = learnable

    file_size_mb = output_path.stat().st_size / 1e6
    print(f"  wrote {file_size_mb:.1f} MB → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a synthetic HDF5 dataset for end-to-end pipeline testing."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/synthetic.h5"),
        help="Destination .h5 path.",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=N_SCENARIOS_TOTAL,
        help="Number of scenarios (default 30, matching the real dataset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--no-learnable",
        action="store_true",
        help="Disable the learnable signal — generate pure i.i.d. noise targets.",
    )
    args = parser.parse_args()

    build_synthetic_dataset(
        output_path=args.output,
        n_scenarios=args.n_scenarios,
        seed=args.seed,
        learnable=not args.no_learnable,
    )


if __name__ == "__main__":
    main()
