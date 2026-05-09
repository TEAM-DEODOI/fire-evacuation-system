"""
DataModule wrapping train / val / OOD :class:`FireDataset` instances.

Provides ``torch.utils.data.DataLoader`` objects with sensible defaults
for the A100-on-RunPod training environment.
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset.fire_dataset import FireDataset


class FireDataModule:
    """Manage the three DataLoaders (train / val / OOD).

    Args:
        dataset_path: Path to the processed HDF5 file.
        batch_size: Samples per batch. Default 4 — fits ``(B, 5, 60, 40, 6)``
            comfortably in 40 GB A100 memory.
        num_workers: DataLoader worker processes. Default 0 (single-process)
            for portability; bump on Linux training boxes.
        pin_memory: If True, pin host memory for faster GPU transfers.
        augment_train: Forwarded to :class:`FireDataset` for the train split.
        shuffle_train: Whether to shuffle the train DataLoader.

    Raises:
        FileNotFoundError: If ``dataset_path`` does not exist.
    """

    def __init__(
        self,
        dataset_path: Path,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        augment_train: bool = False,
        shuffle_train: bool = True,
    ) -> None:
        self.dataset_path: Path = Path(dataset_path)
        self.batch_size: int = int(batch_size)
        self.num_workers: int = int(num_workers)
        self.pin_memory: bool = bool(pin_memory)
        self.shuffle_train: bool = bool(shuffle_train)

        self.train_ds = FireDataset(self.dataset_path, split="train", augment=augment_train)
        self.val_ds = FireDataset(self.dataset_path, split="val", augment=False)
        self.ood_ds = FireDataset(self.dataset_path, split="ood", augment=False)

    def _make_loader(self, dataset: FireDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training DataLoader."""
        return self._make_loader(self.train_ds, shuffle=self.shuffle_train)

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader (not shuffled)."""
        return self._make_loader(self.val_ds, shuffle=False)

    def ood_dataloader(self) -> DataLoader:
        """OOD-evaluation DataLoader (not shuffled)."""
        return self._make_loader(self.ood_ds, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run with ``python -m src.dataset.data_module``).
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    import h5py
    import numpy as np

    from src.shared.constants import (
        GRID_SHAPE,
        N_INPUT_CHANNELS,
        N_OUTPUT_CHANNELS,
        TIME_STEPS,
    )

    print("=" * 60)
    print("FireDataModule self-test")
    print("=" * 60)

    errors: list[str] = []
    nx, ny, nz = GRID_SHAPE

    with tempfile.TemporaryDirectory() as tmp:
        h5_path = Path(tmp) / "synthetic.h5"
        rng = np.random.default_rng(0)
        with h5py.File(h5_path, "w") as f:
            for sid in range(6):
                grp = f.create_group(f"scenario_{sid:03d}")
                grp.create_dataset(
                    "input",
                    data=rng.uniform(0, 1, size=(TIME_STEPS, N_INPUT_CHANNELS, nx, ny, nz)).astype(np.float32),
                )
                grp.create_dataset(
                    "target",
                    data=rng.uniform(0, 1, size=(TIME_STEPS, N_OUTPUT_CHANNELS, nx, ny, nz)).astype(np.float32),
                )
            f.create_dataset("mask", data=np.ones(GRID_SHAPE, dtype=np.float32))
            meta = f.create_group("metadata")
            meta.create_dataset("train_indices", data=np.array([0, 1, 2, 3], dtype=np.int32))
            meta.create_dataset("val_indices", data=np.array([4], dtype=np.int32))
            meta.create_dataset("ood_indices", data=np.array([5], dtype=np.int32))

        dm = FireDataModule(h5_path, batch_size=4, num_workers=0)

        for name, loader_fn in [
            ("train", dm.train_dataloader),
            ("val", dm.val_dataloader),
            ("ood", dm.ood_dataloader),
        ]:
            loader = loader_fn()
            it = iter(loader)
            x, y = next(it)
            print(f"  {name}: batch x={tuple(x.shape)} y={tuple(y.shape)}")
            if x.shape[1:] != (N_INPUT_CHANNELS, nx, ny, nz):
                errors.append(f"{name} x channel/spatial mismatch")
            if y.shape[1:] != (N_OUTPUT_CHANNELS, nx, ny, nz):
                errors.append(f"{name} y channel/spatial mismatch")
            if x.shape[0] > 4:
                errors.append(f"{name} batch_size > 4")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("PASS")
    print("=" * 60)
