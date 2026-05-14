"""Sparse-input FNO 학습 — 6 channel input (sensor indicator 추가).

기존 ConvLSTM sparse retrain (`train_sparse_conv_lstm.py`) 결과:
- IoU @ +60s = 0.18 (naïve) → **0.58 (re-sparsify)**

목표:
- FNO 의 Fourier basis 가 sparse + dense smooth pattern 에 호환성 우수 (L4d 입증)
- + sensor indicator 6번째 채널로 "어디가 measurement vs derived" 명시
- → Tier 2 sparse 의 새 best 0.70+ 가능?

Input format (6 channel):
| Idx | 이름             | 값                                                   |
|-----|------------------|------------------------------------------------------|
|  0  | T_norm           | sparse: sensor cell only, else 0                     |
|  1  | V_norm           | sparse                                               |
|  2  | CO_norm          | sparse                                               |
|  3  | mask             | full (1.0 fluid / 0.0 solid)                         |
|  4  | time_enc         | full (broadcast)                                     |
|  5  | sensor_indicator | 1.0 if sensor cell else 0.0 (broadcast over time)    |

Usage:
    python scripts/train_sparse_fno.py --epochs 100 --batch-size 4 \
        --output checkpoints/fno_sparse_v3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models.fno_model import FNOFireModel
from src.shared.constants import GRID_SHAPE, N_TIMESTEPS
from train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator


def sparsify_input_6ch(
    inp: np.ndarray,                    # (5, X, Y, Z) original
    sparse_ind: np.ndarray,             # (X, Y, Z) bool
) -> np.ndarray:
    """5-channel input → 6-channel sparse-augmented.

    - Channels 0-2 (T/V/CO): sparse-only (sensor cell 외 0)
    - Channels 3-4 (mask/time): unchanged
    - Channel 5 (sensor_indicator): 1.0 where sensor, else 0
    """
    nx, ny, nz = inp.shape[1:]
    out = np.zeros((6, nx, ny, nz), dtype=np.float32)
    out[:5] = inp
    not_sensor = ~sparse_ind
    for c in range(3):
        out[c][not_sensor] = 0.0
    out[5] = sparse_ind.astype(np.float32)
    return out


# ─── Dataset ──────────────────────────────────────────────────────────────
class Sparse6chFireDataset(Dataset):
    """6-channel sparse-input dataset wrapper around dataset.h5."""

    def __init__(self, h5_path: Path, sparse_ind: np.ndarray, split: str = "train"):
        self.h5_path = h5_path
        self.sparse_ind = sparse_ind
        self.pairs: List[Tuple[str, int]] = []
        with h5py.File(self.h5_path, "r") as f:
            split_indices = np.asarray(f["metadata"][f"{split}_indices"])
            self.scenario_keys = [f"scenario_{i:03d}" for i in split_indices]
            for key in self.scenario_keys:
                n_pairs = f[key]["input"].shape[0] - 1
                for t in range(n_pairs):
                    self.pairs.append((key, t))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key, t = self.pairs[idx]
        with h5py.File(self.h5_path, "r") as f:
            inp_5ch = np.asarray(f[key]["input"][t])     # (5, X, Y, Z)
            tgt = np.asarray(f[key]["target"][t + 1])    # (3, X, Y, Z)
        inp_6ch = sparsify_input_6ch(inp_5ch, self.sparse_ind)
        return (torch.from_numpy(inp_6ch).float(),
                torch.from_numpy(tgt).float())


# ─── Training ─────────────────────────────────────────────────────────────
def train(args) -> None:
    device = torch.device(args.device)
    print(f"[setup] device={device}")

    sensor_idxs = load_sensor_indices(Path(args.building))
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)
    n_sparse = int(np.sum(sparse_ind))
    print(f"[setup] {len(sensor_idxs)} sensors → {n_sparse} cells "
          f"({100*n_sparse/np.prod(GRID_SHAPE):.1f}%)")

    ds = Sparse6chFireDataset(Path(args.dataset), sparse_ind, split="train")
    print(f"[setup] dataset pairs: {len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = FNOFireModel(
        n_modes=(12, 12, 4),
        in_channels=6,           # ★ 6-channel input (sensor indicator 포함)
        out_channels=3,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        lifting_channels=args.lifting_channels,
        projection_channels=args.projection_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] FNO 6-channel input, {n_params:,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output.mkdir(parents=True, exist_ok=True)
    loss_history: List[float] = []
    best_loss = float("inf")

    print(f"\n[train] {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for x, y in dl:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={mean_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), args.output / "best.pt")

    torch.save(model.state_dict(), args.output / "final.pt")
    np.savetxt(args.output / "loss_history.csv",
                np.array([[i + 1, l] for i, l in enumerate(loss_history)]),
                fmt="%d,%.8f", delimiter=",")
    print(f"\n[DONE] best loss: {best_loss:.6f}")
    print(f"  best.pt: {args.output / 'best.pt'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path, default=Path("configs/building.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/fno_sparse_v3"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lifting-channels", type=int, default=128)
    parser.add_argument("--projection-channels", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
