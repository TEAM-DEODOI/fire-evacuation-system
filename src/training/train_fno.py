"""FNO 학습 스크립트.

매뉴얼 Section 4.3.3 + 4.4.5 / Day-9-Session-2:

* FNO no-PI (``--use-pi`` 없이): Week 8 학습.
* FNO full (``--use-pi``): Week 9 학습 + 4-stage curriculum.

Usage::

    # FNO no-PI 학습 (실데이터)
    python -m src.training.train_fno \\
        --dataset data/processed/dataset.h5 \\
        --output checkpoints/fno_no_pi/

    # FNO full 학습 (warm start from no-PI)
    python -m src.training.train_fno \\
        --dataset data/processed/dataset.h5 \\
        --use-pi \\
        --resume checkpoints/fno_no_pi/best.pt \\
        --output checkpoints/fno_pi/

    # Mock 데이터로 빠른 검증
    python -m src.training.train_fno --self-test
    python -m src.training.train_fno --self-test --use-pi

실제 HDF5 로딩은 별도 작업에서 추가될 예정 — 현재 ``--dataset`` 인자가
주어진 경우 ``NotImplementedError`` 를 발생시키고, mock 또는 self-test
경로만 동작합니다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.fno_model import FNOFireModel, build_default_fno
from src.models.pi_losses import CURRICULUM_STAGES, PIFNOLoss


def create_mock_dataset(
    n_scenarios: int = 4,
    n_timesteps: int = 31,
) -> Tuple[TensorDataset, torch.Tensor]:
    """Mock TensorDataset 생성. 실데이터와 같은 텐서 shape.

    각 시나리오에서 ``T - 1`` 페어 ``(input[t], target[t+1, :3])`` 를 만들어
    ``TensorDataset`` 으로 묶음. 마스크는 ``(60, 40, 6)`` all-fluid placeholder.

    Returns:
        ``(dataset, mask)``.
    """
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for _ in range(n_scenarios):
        all_input = torch.rand(n_timesteps, 5, 60, 40, 6)
        for t in range(n_timesteps - 1):
            inputs.append(all_input[t])
            targets.append(all_input[t + 1, :3])
    x = torch.stack(inputs)    # (N, 5, 60, 40, 6)
    y = torch.stack(targets)   # (N, 3, 60, 40, 6)
    mask = torch.ones(60, 40, 6)
    return TensorDataset(x, y), mask


def train_one_epoch(
    model: FNOFireModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Any,
    mask: torch.Tensor,
    device: torch.device,
    epoch: int,
    use_pi: bool,
) -> Dict[str, float]:
    """1 epoch 학습. 평균 손실 dict 반환."""
    model.train()
    epoch_stats: Dict[str, float] = {
        "total": 0.0, "data": 0.0, "boundary": 0.0,
        "mono": 0.0, "nonneg": 0.0, "pde": 0.0,
    }
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch_mask = mask.unsqueeze(0).unsqueeze(0).expand(
            x.size(0), 1, *mask.shape
        ).to(device)

        optimizer.zero_grad()
        pred = model(x)

        if use_pi:
            prev_TVC = x[:, :3, ...]
            total, ld = loss_fn(pred, y, prev_TVC, batch_mask)
            for k in ld:
                if k in epoch_stats:
                    epoch_stats[k] += ld[k]
        else:
            total = loss_fn(pred, y)
            epoch_stats["data"] += total.item()
            epoch_stats["total"] += total.item()

        total.backward()
        optimizer.step()
        n_batches += 1

    n = max(n_batches, 1)
    for k in epoch_stats:
        epoch_stats[k] /= n
    return epoch_stats


@torch.no_grad()
def validate(
    model: FNOFireModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """평균 validation MSE."""
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        total_loss += nn.functional.mse_loss(pred, y).item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None,
                        help="HDF5 데이터셋 경로. 없으면 mock.")
    parser.add_argument("--use-pi", action="store_true",
                        help="PI 손실 사용 (full 학습)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="이전 체크포인트 (warm start)")
    parser.add_argument("--output", type=Path,
                        default=Path("checkpoints/fno_test/"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--self-test", action="store_true",
                        help="Mock 데이터로 빠른 검증 (2 epoch만)")
    parser.add_argument(
        "--wandb",
        nargs="?",
        const="online",
        default=None,
        choices=("online", "offline", "disabled"),
        help=(
            "Enable Weights & Biases logging. Pass no value (bare --wandb) "
            "for online mode, or one of offline/disabled."
        ),
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Optional wandb run name.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === Dataset ===
    if args.self_test or args.dataset is None:
        print("Mock 데이터셋 생성 중...")
        dataset, mask = create_mock_dataset(n_scenarios=4)
        n_train = int(len(dataset) * 0.8)
        n_val = len(dataset) - n_train
        train_set, val_set = random_split(dataset, [n_train, n_val])
        epochs = 2 if args.self_test else args.epochs
    else:
        # 실데이터 HDF5 로딩 (D-024: val/ood 빈 split 허용)
        import h5py

        from src.dataset.data_module import FireDataModule

        if not args.dataset.exists():
            raise FileNotFoundError(
                f"dataset not found: {args.dataset}.\n"
                "Hint: run `python -m src.data_pipeline.build_dataset` first."
            )

        dm = FireDataModule(
            dataset_path=args.dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
        )
        train_set = dm.train_ds
        val_set = dm.val_ds if len(dm.val_ds) > 0 else None
        print(
            f"Train pairs: {len(train_set)}, "
            f"Val pairs: {'(empty)' if val_set is None else len(val_set)}"
        )

        # mask 는 HDF5 root /mask 에서 로드 (build_dataset.py 가 저장함)
        with h5py.File(args.dataset, "r") as f:
            mask = torch.from_numpy(f["mask"][...].astype("float32"))
        print(f"mask: {mask.shape}, solid={int((mask == 0).sum())}/{mask.numel()}")
        epochs = args.epochs

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    val_loader = (
        DataLoader(
            val_set, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
        )
        if val_set is not None else None
    )
    has_val = val_loader is not None
    val_count = len(val_set) if val_set is not None else 0
    print(f"Train: {len(train_set)}, Val: {val_count if has_val else '(empty)'}")

    # === Model ===
    model = build_default_fno().to(device)
    n_params = model.count_parameters()
    print(f"FNO 파라미터: {n_params:,}")

    if args.resume and args.resume.exists():
        print(f"체크포인트 로드: {args.resume}")
        # PyTorch 2.6+ defaults to weights_only=True, but our FNO checkpoint
        # carries module references (neuralop's GELU activation) that fail the
        # restricted unpickler. Loading our own trained checkpoint is safe.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model_state = ckpt["model"]
        # In PyTorch 2.6+ serialization, the OrderedDict's ``_metadata``
        # attribute (version-warning info) is sometimes round-tripped as a
        # regular key. Strip it before strict load. Re-attach as attribute so
        # ``load_state_dict`` still has access to deprecation metadata.
        if "_metadata" in model_state:
            try:
                model_state._metadata = model_state.pop("_metadata")
            except Exception:
                model_state.pop("_metadata", None)
        model.load_state_dict(model_state)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # === Loss ===
    if args.use_pi:
        loss_fn = PIFNOLoss().to(device)
        print("Loss: PIFNOLoss (curriculum)")
    else:
        loss_fn = nn.MSELoss()
        print("Loss: MSE only (no PI)")

    # === wandb (옵션) ===
    wandb_run = None
    if args.wandb is not None:
        from src.training.wandb_utils import init_wandb

        wandb_cfg = {
            "model": {
                "name": "FNOFireModel",
                "n_modes": (12, 12, 4),
                "in_channels": 5,
                "out_channels": 3,
                "hidden_channels": 32,
                "n_layers": 4,
                "lifting_channels": 128,
                "projection_channels": 128,
                "param_count": n_params,
            },
            "training": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "max_epochs": epochs,
                "use_pi": bool(args.use_pi),
            },
            "dataset": {
                "path": str(args.dataset) if args.dataset else "mock",
                "n_train_pairs": len(train_set),
                "n_val_pairs": val_count if has_val else 0,
            },
        }
        run_tags = ["fno", "PI" if args.use_pi else "no-PI"]
        if args.self_test:
            run_tags.append("self-test")
        wandb_run = init_wandb(
            project="fire-evacuation",
            config=wandb_cfg,
            run_name=args.wandb_name,
            mode=args.wandb,
            tags=run_tags,
            group="fno_pi" if args.use_pi else "fno_no_pi",
        )

    # === Train ===
    args.output.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        if args.use_pi:
            stage = loss_fn.set_stage_by_epoch(epoch)
            if epoch in (0, 30, 60, 80):
                print(f"\n>>> {stage.name} (epoch {epoch})")

        t_start = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, loss_fn, mask,
            device, epoch, args.use_pi,
        )
        val_loss = validate(model, val_loader, device) if has_val else float("nan")
        scheduler.step()
        t_elapsed = time.time() - t_start

        lr_now = scheduler.get_last_lr()[0]
        log = {
            "epoch": epoch,
            "train/loss": train_stats["total"],
            "val/loss": val_loss,
            "lr": lr_now,
            "epoch_time_s": t_elapsed,
        }
        if args.use_pi:
            for k in ("data", "boundary", "mono", "nonneg", "pde"):
                log[f"train/{k}"] = train_stats[k]
            log["stage/name"] = stage.name
            log["stage/idx"] = next(
                (i for i, s in enumerate(CURRICULUM_STAGES)
                 if s.epoch_start <= epoch < s.epoch_end),
                len(CURRICULUM_STAGES) - 1,
            )

        if epoch % 10 == 0 or epoch == epochs - 1 or args.self_test:
            val_str = f"val={val_loss:.6f}" if has_val else "val=—"
            print(
                f"Epoch {epoch:3d}: train={train_stats['total']:.6f} "
                f"{val_str} lr={lr_now:.2e} ({t_elapsed:.1f}s)"
            )

        if wandb_run is not None:
            wandb_run.log(log)

        # D-024: best by val when available, else by train loss.
        score = val_loss if has_val else train_stats["total"]
        if score < best_val_loss:
            best_val_loss = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_loss": train_stats["total"],
                },
                args.output / "best.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs - 1,
            "val_loss": val_loss,
        },
        args.output / "final.pt",
    )

    print(f"\n학습 완료. best val={best_val_loss:.6f}")
    print(f"저장: {args.output}/best.pt")

    if wandb_run is not None:
        try:
            wandb_run.summary["best/score"] = best_val_loss
            wandb_run.summary["epochs_completed"] = epochs
        except Exception:  # pragma: no cover
            pass
        from src.training.wandb_utils import finish_wandb
        finish_wandb(wandb_run)

    if args.self_test:
        print("\nself-test 결과:")
        print(f"  ✓ {epochs} epoch 완료")
        print(f"  ✓ best val loss: {best_val_loss:.6f}")
        if args.use_pi:
            print("  ✓ PI 손실 모드: stage 1 활성")
        else:
            print("  ✓ MSE only 모드")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
