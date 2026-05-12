"""
Generic training loop shared by ConvLSTM and PI-FNO.

Handles per-epoch training/validation, optional gradient clipping, optional
LR scheduler stepping (epoch-level), and best-checkpoint saving when a
``save_path`` is provided. Loss computation is delegated to the caller via
the ``loss_fn`` argument so the same Trainer drives both pure-MSE
(ConvLSTM baseline) and PI-loss (FNO) training.

Callbacks are accepted but treated as duck-typed plug-ins: any object with
``on_validation_end(epoch, metrics)`` or ``should_stop(epoch, metrics)`` is
invoked at the appropriate moment. No specific callback class is required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader


def _resolve_device(device: Optional[str]) -> str:
    """Pick a sensible default device when caller passes ``None``."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """Epoch-loop trainer for fire prediction models.

    Args:
        model: A ``FireConvLSTM`` or PI-FNO instance with a ``forward(x)``.
        optimizer: Configured PyTorch optimiser (e.g. ``AdamW``).
        loss_fn: ``loss_fn(pred, target) -> scalar tensor``.
        device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        max_epochs: Maximum training epochs.
        grad_clip: Gradient L2-norm clipping threshold. ``<=0`` disables.
        scheduler: Optional epoch-level LR scheduler.
        callbacks: Optional list of callback objects. Any object exposing
            ``on_validation_end(epoch, metrics)`` will be invoked after each
            validation pass; any exposing ``should_stop(epoch, metrics) ->
            bool`` is consulted to break the training loop early.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: Optional[str] = None,
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[Any]] = None,
        wandb_run: Optional[Any] = None,
    ) -> None:
        self.device: str = _resolve_device(device)
        self.model: torch.nn.Module = model.to(self.device)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_fn = loss_fn
        self.max_epochs: int = int(max_epochs)
        self.grad_clip: float = float(grad_clip)
        self.scheduler = scheduler
        self.callbacks: List[Any] = list(callbacks or [])
        # Optional Weights & Biases run handle. If provided, each epoch's
        # metrics are logged via ``wandb_run.log({...})``.
        self.wandb_run = wandb_run

    # ─── Public API ─────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_path: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Run training for up to ``max_epochs`` epochs.

        Args:
            train_loader: Iterates ``(x, y)`` pairs of normalised tensors.
            val_loader: Validation DataLoader. ``None`` or an empty loader
                (``len(loader.dataset) == 0``) disables the val pass —
                ``best`` checkpoint logic then falls back to train-loss
                improvement (D-024 mode).
            save_path: Optional ``.pt`` path for the best checkpoint.

        Returns:
            History dict ``{"train_losses": [...], "val_losses": [...]}``.
            ``val_losses`` is filled with ``nan`` when no val is supplied.
        """
        import math
        import time

        has_val = (
            val_loader is not None and len(val_loader.dataset) > 0  # type: ignore[arg-type]
        )
        history: Dict[str, List[float]] = {"train_losses": [], "val_losses": []}
        best_val: float = float("inf")
        best_train: float = float("inf")

        for epoch in range(self.max_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            history["train_losses"].append(train_loss)

            if has_val:
                val_loss = self.val_epoch(val_loader)  # type: ignore[arg-type]
            else:
                val_loss = float("nan")
            history["val_losses"].append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step()
            lr_now = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t0

            # Best-checkpoint logic: by val_loss when available, else train_loss.
            improved = False
            if save_path is not None:
                if has_val and val_loss < best_val:
                    best_val = val_loss
                    improved = True
                elif (not has_val) and train_loss < best_train:
                    best_train = train_loss
                    improved = True
                if improved:
                    save_path = Path(save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)

            metrics = {"train/loss": train_loss, "val/loss": val_loss}

            # ── wandb logging ────────────────────────────────────────────
            if self.wandb_run is not None:
                log_payload: Dict[str, Any] = {
                    "train/loss": train_loss,
                    "lr": lr_now,
                    "epoch_time_s": epoch_time,
                    "epoch": epoch,
                }
                if has_val:
                    log_payload["val/loss"] = val_loss
                if improved:
                    log_payload["best/improved"] = 1
                    log_payload["best/score"] = best_val if has_val else best_train
                try:
                    self.wandb_run.log(log_payload)
                except Exception as exc:  # pragma: no cover
                    print(f"  wandb.log failed: {exc}")

            for cb in self.callbacks:
                if hasattr(cb, "on_validation_end"):
                    cb.on_validation_end(epoch, metrics)

            stop = any(
                hasattr(cb, "should_stop") and cb.should_stop(epoch, metrics)
                for cb in self.callbacks
            )

            val_str = f"val={val_loss:.4f}" if has_val else "val=—"
            best_mark = "  *best*" if improved else ""
            print(
                f"Epoch {epoch + 1:3d}/{self.max_epochs}  "
                f"train={train_loss:.4f}  {val_str}  "
                f"lr={lr_now:.2e}  ({epoch_time:.1f}s){best_mark}"
            )
            if stop:
                print("  → early stopping triggered")
                break

        # Drop a final summary into wandb so the run page surfaces best metrics.
        if self.wandb_run is not None:
            try:
                self.wandb_run.summary["best/score"] = (
                    best_val if has_val else best_train
                )
                self.wandb_run.summary["epochs_completed"] = len(history["train_losses"])
            except Exception:  # pragma: no cover
                pass

        return history

    # ─── Per-epoch loops ────────────────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch. Returns the sample-weighted mean loss."""
        self.model.train()
        total = 0.0
        n_samples = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            bs = x.size(0)
            total += loss.item() * bs
            n_samples += bs
        return total / max(1, n_samples)

    def val_epoch(self, loader: DataLoader) -> float:
        """Run one validation epoch (no gradient computation)."""
        self.model.eval()
        total = 0.0
        n_samples = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                bs = x.size(0)
                total += loss.item() * bs
                n_samples += bs
        return total / max(1, n_samples)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run with ``python -m src.training.trainer``).
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Trainer self-test (overfit a tiny linear model)")
    print("=" * 60)

    torch.manual_seed(0)

    # Minimal dataset: 4 fixed (x, y) pairs. Linear model should over-fit
    # them in a few dozen epochs.
    class _ToyDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.x = torch.randn(4, 3)
            self.y = torch.randn(4, 1)

        def __len__(self) -> int:
            return self.x.shape[0]

        def __getitem__(self, i: int):
            return self.x[i], self.y[i]

    ds = _ToyDataset()
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = torch.nn.Linear(3, 1)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.functional.mse_loss

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "best.pt"
        tr = Trainer(
            model=model,
            optimizer=optim,
            loss_fn=loss_fn,
            device="cpu",
            max_epochs=200,
            grad_clip=1.0,
        )
        history = tr.fit(loader, loader, save_path=save_path)

        final_train = history["train_losses"][-1]
        final_val = history["val_losses"][-1]
        print(f"\nfinal train={final_train:.6f}  val={final_val:.6f}")
        if final_train > 1e-3:
            print("FAIL: did not converge (train loss too high)")
            raise SystemExit(1)
        if not save_path.exists():
            print("FAIL: best checkpoint was not saved")
            raise SystemExit(1)

    print("\n" + "=" * 60)
    print("PASS")
    print("=" * 60)
