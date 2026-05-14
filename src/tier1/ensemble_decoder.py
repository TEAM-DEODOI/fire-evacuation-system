"""Learned ensemble decoder — per-cell MLP that combines the three
surrogate models (Tier 1 GNN node-projected + Tier 2 Sparse-ConvLSTM +
Tier 2 Sparse-FNO) into a single cell-level danger map.

Replaces the hand-crafted weighted average in `evaluate_ensemble_3way.py`
(which plateaued at IoU 0.618 / FNR 5.1% on the 13 OOD scenarios).

Architecture: simple MLP applied independently to every (cell, timestep)
pair. Spatial context is already encoded into the sparse-model outputs,
so adding spatial conv on top is unnecessary; per-cell MLP keeps the
parameter count tiny (~1 K) and trains in minutes.

Input features per cell-timestep (7-D):
  [gnn_cell, sparse_conv, sparse_fno, mask, x_norm, y_norm, z_norm, t_norm]
  where positions / time are normalized to [0, 1].

Output: cell danger ∈ [0, 1].

Loss: BCE on binary truth (danger ≥ 0.5).  Asymmetric option (FN penalty)
to bias towards safety (lower FNR).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ─── Constants ────────────────────────────────────────────────────────────
N_FEATURES = 8     # gnn, conv, fno, mask, x, y, z, t
N_LOOKAHEAD = 6
GRID_X, GRID_Y, GRID_Z = 60, 40, 6


# ─── Model ────────────────────────────────────────────────────────────────
class PerCellEnsembleDecoder(nn.Module):
    """Per-cell MLP — 8 input features → 1 scalar danger ∈ [0, 1].

    Tiny model (~1.2 K params) — trains in minutes on CPU.

    Args:
        hidden: hidden layer width (default 32).
        n_layers: number of hidden layers (default 2).
        dropout: dropout probability (default 0.0).
    """

    def __init__(self, hidden: int = 32, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = N_FEATURES
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., N_FEATURES). Returns (..., 1) ∈ [0, 1]."""
        return torch.sigmoid(self.net(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Feature builder ──────────────────────────────────────────────────────
def build_cell_features(
    gnn_cell: np.ndarray,        # (T, X, Y, Z)
    sparse_conv: np.ndarray,     # (T, X, Y, Z)
    sparse_fno: np.ndarray,      # (T, X, Y, Z)
    mask: np.ndarray,            # (X, Y, Z)
) -> np.ndarray:
    """Stack model outputs + position + time into (T, X, Y, Z, N_FEATURES).

    Position features are normalized to [0, 1].
    """
    T = gnn_cell.shape[0]
    X, Y, Z = mask.shape
    # Position grids (normalized)
    x_norm = np.linspace(0, 1, X, dtype=np.float32)
    y_norm = np.linspace(0, 1, Y, dtype=np.float32)
    z_norm = np.linspace(0, 1, Z, dtype=np.float32)
    xg, yg, zg = np.meshgrid(x_norm, y_norm, z_norm, indexing="ij")
    # broadcast over time
    xg_t = np.broadcast_to(xg, (T, X, Y, Z))
    yg_t = np.broadcast_to(yg, (T, X, Y, Z))
    zg_t = np.broadcast_to(zg, (T, X, Y, Z))
    mask_t = np.broadcast_to(mask[None, ...], (T, X, Y, Z))
    # Time grid
    t_norm = np.linspace(1 / N_LOOKAHEAD, 1.0, T, dtype=np.float32)
    t_grid = np.broadcast_to(
        t_norm[:, None, None, None], (T, X, Y, Z)
    )

    features = np.stack([
        gnn_cell, sparse_conv, sparse_fno, mask_t,
        xg_t, yg_t, zg_t, t_grid,
    ], axis=-1).astype(np.float32)
    return features    # (T, X, Y, Z, 8)


# ─── Dataset ──────────────────────────────────────────────────────────────
class DecoderDataset(Dataset):
    """Loads precomputed npz files into a flat cell-level training set.

    Each sample = one (cell, timestep) → 8-D feature + 1-D binary target.
    Only fluid cells (mask > 0.5) are kept.

    Args:
        npz_paths: list of npz files (one per scenario).
        threshold: danger threshold for binary target (default 0.5).
        fluid_only: True to drop solid cells.
    """

    def __init__(
        self,
        npz_paths: List[Path],
        threshold: float = 0.5,
        fluid_only: bool = True,
    ):
        feats_list = []
        targets_list = []
        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            gnn_cell = data["gnn_cell"]
            sparse_conv = data["sparse_conv"]
            sparse_fno = data["sparse_fno"]
            truth = data["truth"]
            mask = data["mask"]

            feats = build_cell_features(gnn_cell, sparse_conv, sparse_fno, mask)
            # (T, X, Y, Z, 8)
            target_bin = (truth >= threshold).astype(np.float32)
            target_cont = truth.astype(np.float32)

            if fluid_only:
                mask_t = np.broadcast_to(mask[None, ...], truth.shape)
                fluid_idx = mask_t > 0.5
                feats_flat = feats[fluid_idx]      # (n_samples, 8)
                tb_flat    = target_bin[fluid_idx] # (n_samples,)
                tc_flat    = target_cont[fluid_idx]
            else:
                feats_flat = feats.reshape(-1, feats.shape[-1])
                tb_flat = target_bin.reshape(-1)
                tc_flat = target_cont.reshape(-1)

            feats_list.append(feats_flat)
            # Stack binary + continuous target as 2-D for flexibility
            targets_list.append(np.stack([tb_flat, tc_flat], axis=-1))

        self.features = np.concatenate(feats_list, axis=0)
        self.targets  = np.concatenate(targets_list, axis=0)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.targets[idx]),
        )


# ─── Loss helpers ─────────────────────────────────────────────────────────
def asymmetric_bce_loss(
    pred: torch.Tensor,
    target_bin: torch.Tensor,
    fn_weight: float = 2.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Asymmetric BCE — heavier penalty for false negatives (missed danger).

    fn_weight > 1 ⇒ false negatives cost more ⇒ ↓FNR (at the cost of some FPR).
    Set to 1.0 for standard BCE.
    """
    p = pred.clamp(eps, 1 - eps)
    fn_term = -fn_weight * target_bin * torch.log(p)
    fp_term = -(1 - target_bin) * torch.log(1 - p)
    return (fn_term + fp_term).mean()


# ─── Forward over full grid (for evaluation/visualization) ────────────────
def decoder_forward_grid(
    decoder: PerCellEnsembleDecoder,
    gnn_cell: np.ndarray,
    sparse_conv: np.ndarray,
    sparse_fno: np.ndarray,
    mask: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Run decoder over the full (T, X, Y, Z) grid; returns same shape."""
    feats = build_cell_features(gnn_cell, sparse_conv, sparse_fno, mask)
    # (T, X, Y, Z, 8)
    flat = feats.reshape(-1, N_FEATURES)
    with torch.no_grad():
        out = decoder(torch.from_numpy(flat).to(device)).cpu().numpy().squeeze(-1)
    return out.reshape(feats.shape[:-1]).astype(np.float32)


if __name__ == "__main__":
    # Self-test
    decoder = PerCellEnsembleDecoder(hidden=32, n_layers=2)
    print(f"PerCellEnsembleDecoder: {decoder.n_params():,} params")
    x = torch.rand(4, N_FEATURES)
    y = decoder(x)
    assert y.shape == (4, 1)
    assert 0 <= y.min().item() <= y.max().item() <= 1
    print(f"  forward ok: in {x.shape} -> out {y.shape}, range [{y.min():.3f}, {y.max():.3f}]")

    # Feature builder self-test
    fake_gnn   = np.random.rand(N_LOOKAHEAD, GRID_X, GRID_Y, GRID_Z).astype(np.float32)
    fake_conv  = np.random.rand(N_LOOKAHEAD, GRID_X, GRID_Y, GRID_Z).astype(np.float32)
    fake_fno   = np.random.rand(N_LOOKAHEAD, GRID_X, GRID_Y, GRID_Z).astype(np.float32)
    fake_mask  = (np.random.rand(GRID_X, GRID_Y, GRID_Z) > 0.3).astype(np.float32)
    feats = build_cell_features(fake_gnn, fake_conv, fake_fno, fake_mask)
    assert feats.shape == (N_LOOKAHEAD, GRID_X, GRID_Y, GRID_Z, N_FEATURES)
    print(f"  feature builder ok: shape {feats.shape}")

    # Grid forward self-test
    out_grid = decoder_forward_grid(decoder, fake_gnn, fake_conv, fake_fno, fake_mask)
    assert out_grid.shape == (N_LOOKAHEAD, GRID_X, GRID_Y, GRID_Z)
    print(f"  grid forward ok: shape {out_grid.shape}")

    print("\n[PASS]")
