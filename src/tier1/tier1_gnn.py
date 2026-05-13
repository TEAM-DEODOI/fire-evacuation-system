"""Simple GNN for Tier 1 fire risk prediction (PyTorch only, no PyG dependency).

Architecture rationale:
* 27 node graph — too small to benefit from heavy spectral GNN libraries
* Avoid torch-geometric-temporal dependency (Python 3.12 wheel issues)
* End-to-end: binary sensor sequence → per-node future danger ∈ [0, 1]

Model components:
1. Node encoder: Linear(F → H)  — embed per-node feature at each timestep
2. Temporal encoder: GRU shared across nodes (per-node sequence)
3. Graph propagation: Adjacency-weighted message passing (k layers)
4. Output head: Linear(H → T_out) + Sigmoid

Tensor shapes (forward):
    x:           (B, N, T_in, F)        — node-feature sequences
    adj:         (N, N)                  — graph adjacency (symmetric, normalized)
    return:      (B, N, T_out)          — danger ∈ [0, 1] per node per step
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.tier1.detector_positions import ALL_DETECTORS

N_NODES = len(ALL_DETECTORS)   # 27


# ─── Adjacency construction ───────────────────────────────────────────────
def build_knn_adjacency(k: int = 4, sigma: float = 5.0) -> torch.Tensor:
    """k-NN graph from D-024 27 detector positions (Euclidean in 2D).

    Symmetric, Gaussian-weighted. Self-loops excluded.

    Args:
        k: number of nearest neighbors per node.
        sigma: Gaussian kernel bandwidth (m).

    Returns:
        ``(N, N)`` float32 adjacency. Symmetrically normalized
        ``D^{-1/2} A D^{-1/2}`` for GCN-style propagation.
    """
    positions = np.array([(d.position[0], d.position[1])
                           for d in ALL_DETECTORS])  # (N, 2)
    n = len(positions)
    # Pairwise Euclidean
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))            # (N, N)
    np.fill_diagonal(dist, np.inf)

    # k-NN selection
    adj = np.zeros((n, n), dtype=np.float32)
    nearest = np.argsort(dist, axis=1)[:, :k]
    for i in range(n):
        for j in nearest[i]:
            w = math.exp(-dist[i, j] ** 2 / (2 * sigma ** 2))
            adj[i, j] = w
    # Symmetrize
    adj = np.maximum(adj, adj.T)
    # Add self-loop
    adj += np.eye(n, dtype=np.float32)
    # Symmetric normalization: D^-1/2 A D^-1/2
    deg = adj.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg + 1e-9)
    adj_norm = adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return torch.from_numpy(adj_norm).float()


def build_node_type_onehot() -> torch.Tensor:
    """One-hot encoding of node type for 27 detectors.

    Categories: room, corridor, exit (3-dim).

    Returns:
        ``(N, 3)`` float32.
    """
    types = ["room", "corridor", "exit"]
    out = torch.zeros(N_NODES, 3, dtype=torch.float32)
    for i, d in enumerate(ALL_DETECTORS):
        out[i, types.index(d.node_type)] = 1.0
    return out


# ─── Model ─────────────────────────────────────────────────────────────────
class SimpleFireGNN(nn.Module):
    """27-node fire-risk GNN, PyTorch-only.

    Args:
        in_feat: per-node feature dim (default 5 = ``[is_det, det_time_norm,
                 type_onehot × 3]``).
        hidden: hidden dim of GRU and graph layers.
        n_graph_layers: number of message-passing iterations after GRU.
        T_out: prediction horizon (number of future steps).
    """

    def __init__(
        self,
        in_feat: int = 5,
        hidden: int = 32,
        n_graph_layers: int = 2,
        T_out: int = 6,
    ) -> None:
        super().__init__()
        self.in_feat = in_feat
        self.hidden = hidden
        self.T_out = T_out

        self.node_encoder = nn.Linear(in_feat, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.graph_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(n_graph_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, T_out),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,         # (B, N, T_in, F)
        adj: torch.Tensor,       # (N, N)
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must be 4-D (B, N, T, F), got {tuple(x.shape)}")
        B, N, T, F = x.shape
        if F != self.in_feat:
            raise ValueError(f"in_feat {F} != model.in_feat {self.in_feat}")

        # 1. Node + time encoding
        x_flat = x.reshape(B * N, T, F)
        h = self.node_encoder(x_flat)  # (B*N, T, H)

        # 2. Temporal GRU — take last hidden as node summary
        _, h_last = self.gru(h)        # h_last: (1, B*N, H)
        h_node = h_last.squeeze(0).reshape(B, N, self.hidden)  # (B, N, H)

        # 3. Graph propagation: A @ H, residual MLP
        for layer in self.graph_layers:
            # adj @ h_node (per batch)
            h_neigh = torch.einsum("nm,bmh->bnh", adj, h_node)
            h_node = h_node + layer(h_neigh)

        # 4. Output: per-node future danger sequence
        out = self.head(h_node)  # (B, N, T_out)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("tier1_gnn.py self-test")
    print("=" * 70)
    errors: list[str] = []

    # Test 1: adjacency
    print("\n[Test 1] build_knn_adjacency")
    adj = build_knn_adjacency(k=4)
    print(f"  shape: {tuple(adj.shape)}")
    if adj.shape != (N_NODES, N_NODES):
        errors.append(f"adj shape: {adj.shape}")
    # Symmetric
    if not torch.allclose(adj, adj.T, atol=1e-5):
        errors.append("adj not symmetric")
    print(f"  symmetric: PASS")
    # Diagonal nonzero (self-loops)
    if (adj.diag() <= 0).any():
        errors.append("missing self-loops")
    print(f"  diag mean: {adj.diag().mean():.3f}")
    # Sparsity
    n_nonzero = (adj > 0).sum().item()
    print(f"  nonzero entries: {n_nonzero} / {N_NODES * N_NODES} "
          f"({100 * n_nonzero / N_NODES**2:.1f}%)")

    # Test 2: node type one-hot
    print("\n[Test 2] build_node_type_onehot")
    types = build_node_type_onehot()
    print(f"  shape: {tuple(types.shape)}")
    # Row sums should all be 1
    if not torch.allclose(types.sum(dim=1), torch.ones(N_NODES)):
        errors.append("type onehot rows don't sum to 1")
    print(f"  type counts: room={(types[:, 0] > 0).sum()} "
          f"corridor={(types[:, 1] > 0).sum()} exit={(types[:, 2] > 0).sum()}")

    # Test 3: forward pass
    print("\n[Test 3] SimpleFireGNN forward")
    model = SimpleFireGNN(in_feat=5, hidden=32, n_graph_layers=2, T_out=6)
    print(f"  parameters: {model.count_parameters():,}")
    B, T_in = 2, 6
    x = torch.rand(B, N_NODES, T_in, 5)
    out = model(x, adj)
    print(f"  input  shape: {tuple(x.shape)}")
    print(f"  output shape: {tuple(out.shape)}")
    if out.shape != (B, N_NODES, 6):
        errors.append(f"output shape: {out.shape}")
    if out.min() < 0 or out.max() > 1:
        errors.append(f"output range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"  output range: [{out.min():.3f}, {out.max():.3f}]")

    # Test 4: backward pass (gradients flow)
    print("\n[Test 4] backward pass")
    target = torch.rand(B, N_NODES, 6)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()
    grad_norms = [p.grad.abs().mean().item() for p in model.parameters()
                  if p.grad is not None]
    print(f"  loss: {loss.item():.4f}")
    print(f"  mean grad mag (all params): {np.mean(grad_norms):.6f}")
    if all(g == 0 for g in grad_norms):
        errors.append("all gradients zero")

    print("\n" + "=" * 70)
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("PASS: tier1_gnn smoke test complete")
