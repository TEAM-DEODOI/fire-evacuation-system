"""
Tier 1 fire risk GNN.

Implements ``Tier1FireGNN`` exactly as specified in
``docs/tier1_gnn_design.md`` §4: an A3T-GCN backbone followed by a small
MLP head that emits a per-node danger sequence.

Input/Output
------------
Input  : ``x`` shape ``(B, N_nodes, T_in, F_in)`` — node-feature time-series
         ``edge_index`` shape ``(2, E)`` — graph topology (shared across batch)
         ``edge_weight`` shape ``(E,)``  — undirected edge weights
Output : ``(B, N_nodes, T_out)`` — predicted danger ∈ [0, 1] per node per step

By default ``T_in == T_out == periods`` (6 steps = 60 s) per the design doc.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_geometric_temporal.nn.recurrent import A3TGCN2
except ImportError as exc:  # pragma: no cover — surfaced at runtime
    A3TGCN2 = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    _IMPORT_ERROR = None


class Tier1FireGNN(nn.Module):
    """GNN-based fire risk predictor from binary detector signals.

    Wraps ``torch_geometric_temporal.nn.recurrent.A3TGCN2`` with a small MLP
    head that produces a per-node danger sequence.

    Args:
        in_channels: Per-node feature dimension. Default 6 →
            ``[is_detected, det_time_norm, type_onehot × 4]`` (see design doc §4).
        out_channels: A3T-GCN hidden dimension.
        periods: Length of the input/output sequences (``T_in == T_out``).
            6 corresponds to a 60 s window at 10 s steps.
        head_hidden: Hidden dimension of the prediction MLP head.

    Tensor shapes (forward):
        x:           ``(B, N, T_in, F_in)``     — node-feature sequences
        edge_index:  ``(2, E)``                 — graph topology
        edge_weight: ``(E,)``                   — edge weights
        return:      ``(B, N, T_out)``          — danger ∈ [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 32,
        periods: int = 6,
        head_hidden: int = 16,
    ) -> None:
        super().__init__()

        if A3TGCN2 is None:
            raise ImportError(
                "torch_geometric_temporal is not installed. "
                "Install with `pip install torch-geometric-temporal`."
            ) from _IMPORT_ERROR

        if in_channels < 1:
            raise ValueError(f"in_channels must be ≥ 1, got {in_channels}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be ≥ 1, got {out_channels}")
        if periods < 1:
            raise ValueError(f"periods must be ≥ 1, got {periods}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods

        # ``batch_size`` is required by A3TGCN2 only to allocate the recurrent
        # hidden state when ``H`` is not supplied. We always pass an explicit
        # ``H`` of the correct shape in :meth:`forward`, so the value here is
        # only a placeholder and the module is batch-size agnostic at runtime.
        self.a3tgcn = A3TGCN2(
            in_channels=in_channels,
            out_channels=out_channels,
            periods=periods,
            batch_size=1,
        )

        # Prediction head: hidden → danger score per timestep.
        self.output_head = nn.Sequential(
            nn.Linear(out_channels, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, periods),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-node danger sequences.

        Args:
            x: Node-feature sequences, shape ``(B, N, T_in, F_in)``.
            edge_index: Graph topology, shape ``(2, E)`` (LongTensor).
            edge_weight: Edge weights, shape ``(E,)`` (FloatTensor).

        Returns:
            Predicted danger scores, shape ``(B, N, T_out)``, values in [0, 1].

        Raises:
            ValueError: If ``x`` is not 4-D or its dimensions disagree with
                the model's configured ``in_channels`` / ``periods``.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4-D input (B, N, T, F), got shape {tuple(x.shape)}"
            )

        b, n, t_in, f_in = x.shape
        if f_in != self.in_channels:
            raise ValueError(
                f"in_channels mismatch: model={self.in_channels}, x.F={f_in}"
            )
        if t_in != self.periods:
            raise ValueError(
                f"periods mismatch: model={self.periods}, x.T={t_in}. "
                "T_in must equal model.periods (asymmetric T_in/T_out not "
                "supported by this wrapper yet)."
            )

        # A3TGCN2 expects (B, N, F, T); permute from our docstring layout (B, N, T, F).
        x_permuted = x.permute(0, 1, 3, 2).contiguous()

        # Provide an explicit zero hidden state shaped to the runtime batch
        # size. This bypasses A3TGCN2's internal ``batch_size`` placeholder
        # and keeps the module agnostic to the configured value.
        h0 = torch.zeros(b, n, self.out_channels, device=x.device, dtype=x.dtype)

        # A3TGCN2: (B, N, F, T) → (B, N, hidden)
        h = self.a3tgcn(x_permuted, edge_index, edge_weight, h0)

        # (B, N, hidden) → (B, N, periods)
        out = self.output_head(h)
        return out

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers used only by the self-test below.
# ─────────────────────────────────────────────────────────────────────────────

def _make_grid_edges(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a 4-connected undirected grid graph.

    Args:
        rows: Number of rows in the grid.
        cols: Number of columns.

    Returns:
        ``(edge_index, edge_weight)`` where ``edge_index`` has shape ``(2, 2E)``
        (each undirected edge stored in both directions) and ``edge_weight``
        is all-ones, shape ``(2E,)``.
    """
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            if c + 1 < cols:
                edges.append((n, n + 1))
            if r + 1 < rows:
                edges.append((n, n + cols))

    src = [a for a, _ in edges] + [b for _, b in edges]
    dst = [b for _, b in edges] + [a for a, _ in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run with ``python -m src.tier1.tier1_model``).
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Tier1FireGNN sanity check")
    print("=" * 60)

    if A3TGCN2 is None:
        print("\nFAIL: torch_geometric_temporal is not installed.")
        print(f"      Underlying error: {_IMPORT_ERROR}")
        print("      Run `pip install -r requirements.txt` first.")
        raise SystemExit(1)

    torch.manual_seed(0)

    # ── Test 1: forward pass shape on CPU ────────────────────────────────────
    print("\n[Test 1] Forward pass on CPU")
    rows, cols = 4, 4
    n_nodes = rows * cols  # 16
    in_channels, out_channels, periods = 6, 32, 6
    batch_size = 2

    model = Tier1FireGNN(
        in_channels=in_channels,
        out_channels=out_channels,
        periods=periods,
    )
    print(f"  Trainable parameters: {model.count_parameters():,}")

    edge_index, edge_weight = _make_grid_edges(rows, cols)
    x = torch.randn(batch_size, n_nodes, periods, in_channels)
    print(f"  x.shape           = {tuple(x.shape)}")
    print(f"  edge_index.shape  = {tuple(edge_index.shape)}")

    with torch.no_grad():
        y = model(x, edge_index, edge_weight)
    expected = (batch_size, n_nodes, periods)
    print(f"  y.shape           = {tuple(y.shape)}  (expected {expected})")
    assert y.shape == torch.Size(expected), f"output shape mismatch: {y.shape}"
    assert (y >= 0).all() and (y <= 1).all(), "output not in [0, 1]"
    print("  PASS: shape correct, output ∈ [0, 1]")

    # ── Test 2: gradient flow ────────────────────────────────────────────────
    print("\n[Test 2] Backward pass — gradient flow")
    target = torch.rand(batch_size, n_nodes, periods)
    loss_fn = nn.MSELoss()
    loss = loss_fn(model(x, edge_index, edge_weight), target)
    loss.backward()

    n_params = 0
    n_with_grad = 0
    for p in model.parameters():
        if p.requires_grad:
            n_params += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                n_with_grad += 1
    print(f"  loss              = {loss.item():.4f}")
    print(f"  params with grad  = {n_with_grad} / {n_params}")
    assert n_with_grad == n_params, "some parameters received no gradient"
    print("  PASS: all parameters received gradients")

    # ── Test 3: variable batch sizes ─────────────────────────────────────────
    print("\n[Test 3] Variable batch sizes")
    for bs in (1, 4):
        x_bs = torch.randn(bs, n_nodes, periods, in_channels)
        with torch.no_grad():
            y_bs = model(x_bs, edge_index, edge_weight)
        assert y_bs.shape == (bs, n_nodes, periods)
        print(f"  batch={bs}: y.shape = {tuple(y_bs.shape)} OK")

    # ── Test 4: GPU forward (if CUDA available) ──────────────────────────────
    if torch.cuda.is_available():
        print("\n[Test 4] GPU forward pass")
        m_gpu = Tier1FireGNN(in_channels, out_channels, periods).cuda()
        x_gpu = x.cuda()
        ei_gpu = edge_index.cuda()
        ew_gpu = edge_weight.cuda()
        with torch.no_grad():
            y_gpu = m_gpu(x_gpu, ei_gpu, ew_gpu)
        print(f"  y_gpu.device = {y_gpu.device}")
        print(f"  y_gpu.shape  = {tuple(y_gpu.shape)}")
        print("  PASS: GPU forward pass works")
    else:
        print("\n[Test 4] Skipped (no CUDA available)")

    print("\n" + "=" * 60)
    print("PASS")
    print("=" * 60)
