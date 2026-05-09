"""
Training entry point for ``Tier1FireGNN``.

Two modes:

* ``--smoke``: 1-batch over-fit sanity check. Generates random node features
  and a random target, then trains for a small number of epochs to confirm
  the loss collapses toward zero. **No real dataset required.** This is the
  Week 9 "Sanity check 2" pattern from ``docs/manual_v2.md``, applied to
  Tier 1.

* (default): Full training. Currently disabled because it requires the data
  pipeline modules (``detector_extractor.py``, ``tier1_dataset.py``) which
  depend on FDS data not yet available. Will be wired up in G3/G4.

Usage::

    python -m src.tier1.train_tier1 --smoke
    python -m src.tier1.train_tier1 --config configs/tier1_gnn.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> Dict[str, Any]:
    """Load a Tier 1 GNN YAML config.

    Args:
        path: Path to ``configs/tier1_gnn.yaml`` (or compatible).

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check: 1-batch over-fit
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(
    cfg: Dict[str, Any],
    n_epochs: int = 200,
    target_loss: float = 1e-3,
) -> None:
    """Over-fit a single random batch to verify gradient flow and convergence.

    Generates a random graph (4×4 grid, 16 nodes) and a random target tensor,
    then trains the model for ``n_epochs`` iterations. The final loss must
    drop below ``target_loss``; otherwise something is wired up wrong.

    Args:
        cfg: Parsed Tier 1 config.
        n_epochs: Number of optimisation steps to take on the single batch.
        target_loss: Pass threshold for the final loss.
    """
    import torch
    import torch.nn as nn

    from src.tier1.tier1_model import Tier1FireGNN, _make_grid_edges

    print("=" * 60)
    print("Tier 1 GNN — 1-batch over-fit sanity check")
    print("=" * 60)

    torch.manual_seed(42)

    in_channels = cfg["model"]["in_channels"]
    out_channels = cfg["model"]["out_channels"]
    periods = cfg["model"]["periods"]
    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    batch_size = int(cfg["training"]["batch_size"])

    # Placeholder graph: 4×4 grid until configs/building_graph.yaml lands.
    rows, cols = 4, 4
    n_nodes = rows * cols
    edge_index, edge_weight = _make_grid_edges(rows, cols)

    # Single fixed batch of inputs and targets.
    x = torch.randn(batch_size, n_nodes, periods, in_channels)
    y = torch.rand(batch_size, n_nodes, periods)  # targets in [0, 1]

    model = Tier1FireGNN(
        in_channels=in_channels,
        out_channels=out_channels,
        periods=periods,
    )
    print(f"  parameters         : {model.count_parameters():,}")
    print(f"  graph              : 4×4 grid, {n_nodes} nodes, "
          f"{edge_index.shape[1] // 2} undirected edges")
    print(f"  batch / periods    : {batch_size} / {periods}")
    print(f"  optimiser          : AdamW (lr={lr}, wd={weight_decay})")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    losses: list[float] = []
    for epoch in range(n_epochs):
        optim.zero_grad()
        pred = model(x, edge_index, edge_weight)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if epoch == 0 or (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  epoch {epoch + 1:4d}/{n_epochs}  loss = {loss.item():.6f}")

    final_loss = losses[-1]
    print(f"\n  initial loss : {losses[0]:.6f}")
    print(f"  final loss   : {final_loss:.6f}")
    print(f"  threshold    : {target_loss}")

    if final_loss > target_loss:
        print("\nFAIL: over-fit did not converge below threshold.")
        print("      Possible causes: lr too low, model bug, gradient blocked.")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("PASS")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Real training (deferred until G3/G4 land)
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Dict[str, Any]) -> None:
    """Full training loop. Not yet implemented.

    Will wire up:
        - ``detector_extractor.extract_detector_events_from_slices`` (G3)
        - ``tier1_dataset.Tier1FireDataset`` (G4)
        - ``Tier1FireGNN`` (G5, done)
        - W&B logging (G6)
        - Checkpointing
    """
    raise NotImplementedError(
        "Full training requires G3 (detector_extractor) and G4 (tier1_dataset). "
        "Run with --smoke for the 1-batch sanity check instead."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tier 1 fire-risk GNN")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tier1_gnn.yaml"),
        help="Path to tier1_gnn.yaml.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 1-batch over-fit sanity check (no real data needed).",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=200,
        help="Number of optimisation steps for the smoke test.",
    )
    parser.add_argument(
        "--smoke-target-loss",
        type=float,
        default=1e-3,
        help="Pass threshold for the smoke test final loss.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smoke:
        run_smoke_test(
            cfg,
            n_epochs=args.smoke_epochs,
            target_loss=args.smoke_target_loss,
        )
    else:
        train(cfg)


if __name__ == "__main__":
    main()
