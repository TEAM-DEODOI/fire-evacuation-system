"""
Training entry point for ``FireConvLSTM``.

Two modes:

* ``--smoke``: 1-batch over-fit using random tensors (no dataset required).
  Validates that model + optimiser + loss are wired correctly. Mirrors
  ``src.tier1.train_tier1 --smoke``.

* (default): Full training driven by ``configs/conv_lstm.yaml``. Loads the
  dataset (real FDS-derived HDF5 once available, or the synthetic file from
  ``scripts/build_synthetic_dataset.py``), builds a :class:`FireDataModule`
  + :class:`Trainer`, and saves the best checkpoint by validation loss.

Usage::

    python -m src.training.train_conv_lstm --smoke
    python scripts/build_synthetic_dataset.py
    python -m src.training.train_conv_lstm --data data/processed/synthetic.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> Dict[str, Any]:
    """Load ``configs/conv_lstm.yaml`` (or compatible).

    Args:
        path: Path to the YAML config.

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
# Smoke test: 1-batch over-fit on random tensors
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(
    cfg: Dict[str, Any],
    n_epochs: int = 200,
    target_loss: float = 1e-2,
) -> None:
    """Over-fit a single random batch to verify gradient flow.

    No HDF5 file or DataLoader is involved — the test generates one batch
    of random ``(B, 5, 60, 40, 6)`` input + ``(B, 3, 60, 40, 6)`` targets
    and trains the model on that single batch. The final loss must drop
    below ``target_loss``.

    Args:
        cfg: Parsed conv_lstm config.
        n_epochs: Number of optimisation steps on the single batch.
        target_loss: Pass threshold for the final loss.
    """
    from src.models.conv_lstm_3d import FireConvLSTM

    print("=" * 60)
    print("ConvLSTM — 1-batch over-fit sanity check")
    print("=" * 60)

    torch.manual_seed(42)

    m_cfg = cfg["model"]
    t_cfg = cfg["training"]

    model = FireConvLSTM(
        in_channels=int(m_cfg["in_channels"]),
        out_channels=int(m_cfg["out_channels"]),
        hidden_dim=int(m_cfg["hidden_dim"]),
        kernel_size=tuple(m_cfg["kernel_size"]),
        num_layers=int(m_cfg["num_layers"]),
    )
    print(f"  parameters : {model.count_parameters():,}")

    batch_size = int(t_cfg["batch_size"])
    x = torch.randn(batch_size, int(m_cfg["in_channels"]), 60, 40, 6)
    y = torch.rand(batch_size, int(m_cfg["out_channels"]), 60, 40, 6)
    print(f"  batch      : x={tuple(x.shape)}  y={tuple(y.shape)}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg["lr"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )
    print(f"  optimiser  : AdamW (lr={t_cfg['lr']}, wd={t_cfg['weight_decay']})")

    losses: list[float] = []
    for epoch in range(n_epochs):
        optim.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if epoch == 0 or (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  epoch {epoch + 1:4d}/{n_epochs}  loss = {loss.item():.6f}")

    final = losses[-1]
    print(f"\n  initial loss : {losses[0]:.6f}")
    print(f"  final loss   : {final:.6f}")
    print(f"  threshold    : {target_loss}")

    if final > target_loss:
        print("\nFAIL: over-fit did not converge below threshold.")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("PASS")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Full training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    config_path: Path = Path("configs/conv_lstm.yaml"),
    dataset_path: Path = Path("data/processed/dataset.h5"),
    checkpoint_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train ``FireConvLSTM`` against the processed HDF5 dataset.

    Args:
        config_path: Path to ``configs/conv_lstm.yaml``.
        dataset_path: Path to processed HDF5. Use the synthetic file from
            ``scripts/build_synthetic_dataset.py`` until real FDS data lands.
        checkpoint_dir: Where to save the best checkpoint. Defaults to the
            path in the YAML config.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        History dict from :meth:`Trainer.fit`.

    Raises:
        FileNotFoundError: If ``config_path`` or ``dataset_path`` is missing.
    """
    from src.dataset.data_module import FireDataModule
    from src.models.conv_lstm_3d import FireConvLSTM
    from src.training.trainer import Trainer

    cfg = load_config(config_path)

    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}.\n"
            "Hint: until real FDS data is processed, run\n"
            "  python scripts/build_synthetic_dataset.py "
            f"--output {dataset_path}"
        )

    if checkpoint_dir is None:
        checkpoint_dir = Path(cfg["checkpoints"]["dirpath"])
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    m_cfg = cfg["model"]
    t_cfg = cfg["training"]

    model = FireConvLSTM(
        in_channels=int(m_cfg["in_channels"]),
        out_channels=int(m_cfg["out_channels"]),
        hidden_dim=int(m_cfg["hidden_dim"]),
        kernel_size=tuple(m_cfg["kernel_size"]),
        num_layers=int(m_cfg["num_layers"]),
    )
    if resume_from is not None:
        state = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(state)
        print(f"resumed from {resume_from}")

    dm = FireDataModule(
        dataset_path=dataset_path,
        batch_size=int(t_cfg["batch_size"]),
        num_workers=0,  # safe default; bump on Linux training boxes
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg["lr"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )
    scheduler = None
    if str(t_cfg.get("scheduler", "")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(t_cfg["max_epochs"]),
            eta_min=float(t_cfg.get("eta_min", 1e-6)),
        )

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=int(t_cfg["max_epochs"]),
        grad_clip=float(t_cfg["grad_clip"]),
        scheduler=scheduler,
    )

    history = trainer.fit(
        dm.train_dataloader(),
        dm.val_dataloader(),
        save_path=checkpoint_dir / "best.pt",
    )
    print(f"\nbest checkpoint saved → {checkpoint_dir / 'best.pt'}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train FireConvLSTM")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/conv_lstm.yaml"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/dataset.h5"),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override checkpoints.dirpath from the config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 1-batch over-fit sanity check (no dataset required).",
    )
    parser.add_argument("--smoke-epochs", type=int, default=200)
    parser.add_argument("--smoke-target-loss", type=float, default=1e-2)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smoke:
        run_smoke_test(
            cfg,
            n_epochs=args.smoke_epochs,
            target_loss=args.smoke_target_loss,
        )
    else:
        train(
            config_path=args.config,
            dataset_path=args.data,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume,
        )


if __name__ == "__main__":
    main()
