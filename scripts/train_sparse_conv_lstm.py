"""Track 1B — Sparse-input ConvLSTM 재학습.

Track 1A 의 발견: 16-sensor + 단순 보간 → IoU 0.19. mask-aware geodesic 으로
+0.21 개선 (0.41) 했으나 여전히 H5 (0.70) 미달.

→ **Fundamental fix**: 모델 자체를 sparse representation 으로 재학습.

설계:
* Model architecture: 기존 FireConvLSTM 그대로 (in_channels=5)
* Input format 변경:
    - T/V/CO 채널: 16 sensor 위치 cell 만 측정값, 다른 fluid cell 은 0
    - mask: 그대로 (벽 정보 유지)
    - time_enc: 그대로
* Dataset: 기존 dataset.h5 의 input 을 sparse 로 즉시 변환 (in-memory)
* Loss: 기존 그대로 (target 은 full dense — 즉 sparse → full reconstruction 학습)

이렇게 하면 모델이:
- "16 점만 보고 전체 dense 미래 예측" 을 직접 학습
- 보간 단계 자체 불필요
- 모델 내부에 amortized geodesic-aware 보간 + future prediction 둘 다 학습

Usage:
    python scripts/train_sparse_conv_lstm.py --epochs 50 --batch-size 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import Dataset, DataLoader

from src.models.conv_lstm_3d import FireConvLSTM
from src.shared.constants import GRID_SHAPE, N_TIMESTEPS
from src.shared.coordinates import cell_centres


# ─── Sparse mask generation ────────────────────────────────────────────────
def load_sensor_indices(building_yaml: Path) -> List[Tuple[int, int, int]]:
    """Sensor 위치를 (60, 40, 6) grid index 로 변환.

    Z 는 호흡고도 (1.75m → z_idx=3) 로 고정.
    """
    with building_yaml.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    x_c, y_c, z_c = cell_centres()
    z_idx = int(np.argmin(np.abs(z_c - 1.75)))
    sensors = []
    for n in cfg["nodes"]:
        if not n.get("has_detector"):
            continue
        sx, sy, _ = n["pos"]
        ix = int(np.argmin(np.abs(x_c - sx)))
        iy = int(np.argmin(np.abs(y_c - sy)))
        sensors.append((ix, iy, z_idx))
    return sensors


def make_sparse_indicator(sensor_idxs: List[Tuple[int, int, int]],
                           broadcast_z: bool = True) -> np.ndarray:
    """(60, 40, 6) bool — sensor 위치 True. broadcast_z 이면 z 전체 True."""
    ind = np.zeros(GRID_SHAPE, dtype=bool)
    for ix, iy, iz in sensor_idxs:
        if broadcast_z:
            ind[ix, iy, :] = True
        else:
            ind[ix, iy, iz] = True
    return ind


def sparsify_input(inp: np.ndarray,
                    sparse_ind: np.ndarray) -> np.ndarray:
    """Input (5, X, Y, Z) 의 T/V/CO 채널을 sparse 화.

    sensor 위치 외 cell 의 T/V/CO 값을 0 으로. mask, time_enc 그대로.

    Args:
        inp: (5, X, Y, Z) float32
        sparse_ind: (X, Y, Z) bool, sensor 위치 True
    """
    out = inp.copy()
    not_sensor = ~sparse_ind   # (X, Y, Z) bool
    # broadcast over channels 0..2
    for c in range(3):
        out[c][not_sensor] = 0.0
    return out


# ─── Dataset wrapper ───────────────────────────────────────────────────────
class SparseFireDataset(Dataset):
    """기존 dataset.h5 의 (input, target) pair 를 sparsify 해서 반환."""

    def __init__(self, h5_path: Path, sparse_ind: np.ndarray,
                 split: str = "train"):
        self.h5_path = h5_path
        self.sparse_ind = sparse_ind
        self.pairs: List[Tuple[int, int]] = []  # (scen_idx, t)
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
            inp = np.asarray(f[key]["input"][t])     # (5, X, Y, Z)
            tgt = np.asarray(f[key]["target"][t + 1]) # (3, X, Y, Z)
        sparse_inp = sparsify_input(inp, self.sparse_ind)
        return (torch.from_numpy(sparse_inp).float(),
                torch.from_numpy(tgt).float())


# ─── Training loop ─────────────────────────────────────────────────────────
def train(args) -> None:
    device = torch.device(args.device)
    print(f"[setup] device={device}")

    # 1) Load building, sensors, mask
    sensor_idxs = load_sensor_indices(Path(args.building))
    print(f"[setup] {len(sensor_idxs)} sensors → grid indices")
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)
    n_sparse_cells = int(np.sum(sparse_ind))
    print(f"[setup] sparse cells: {n_sparse_cells} / {np.prod(GRID_SHAPE)} "
          f"({100 * n_sparse_cells / np.prod(GRID_SHAPE):.1f}%)")

    # 2) Dataset / loader
    ds = SparseFireDataset(Path(args.dataset), sparse_ind, split="train")
    print(f"[setup] dataset pairs: {len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, pin_memory=False)

    # 3) Model
    model = FireConvLSTM(
        in_channels=5, out_channels=3, hidden_dim=32,
        kernel_size=(3, 3, 3), num_layers=2,
    ).to(device)
    if args.init_from is not None:
        sd = torch.load(args.init_from, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        if "_metadata" in sd:
            try:
                sd._metadata = sd.pop("_metadata")
            except Exception:
                sd.pop("_metadata", None)
        model.load_state_dict(sd)
        print(f"[setup] initialized from {args.init_from}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4) Training loop
    args.output.mkdir(parents=True, exist_ok=True)
    loss_history: List[float] = []
    best_loss = float("inf")
    print(f"\n[train] {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}\n")
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

    # Save final + loss curve
    torch.save(model.state_dict(), args.output / "final.pt")
    np.savetxt(args.output / "loss_history.csv",
                np.array([[i + 1, l] for i, l in enumerate(loss_history)]),
                fmt="%d,%.8f", delimiter=",")
    print(f"\n[DONE] best loss: {best_loss:.6f}")
    print(f"  best.pt: {args.output / 'best.pt'}")
    print(f"  loss curve: {args.output / 'loss_history.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.h5"))
    parser.add_argument("--building", type=Path, default=Path("configs/building.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/conv_lstm_sparse"))
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Initialise from existing ckpt (warm-start). "
                             "Default: cold start.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
