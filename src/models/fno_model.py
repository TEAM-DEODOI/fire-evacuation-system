"""FNO (Fourier Neural Operator) model wrapper.

Wraps ``neuralop.models.FNO`` to match the project's (B, 5, 60, 40, 6) input
and (B, 3, 60, 40, 6) output interface, identical to ``FireConvLSTM`` so
training code can be shared.

PI losses are applied externally (see ``src/models/pi_losses.py``). This
module is the pure FNO; no physics-informed terms here.

Manual spec (Section 4.3.2 / Day-9-task-prompt):

* ``n_modes = (12, 12, 4)``
* ``in_channels = 5``, ``out_channels = 3``
* ``hidden_channels = 32``, ``n_layers = 4``
* ``lifting_channels = 128``, ``projection_channels = 128``
* Total parameters: ~2-4 M

API note: ``neuraloperator`` 2.0+ replaced the absolute
``lifting_channels`` / ``projection_channels`` arguments with the
``lifting_channel_ratio`` / ``projection_channel_ratio`` arguments. This
wrapper preserves the original absolute-value parameter names and
converts to ratios internally, so callers see no breaking change.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from neuralop.models import FNO

from src.shared.constants import (
    GRID_SHAPE,
    N_INPUT_CHANNELS,
    N_OUTPUT_CHANNELS,
)


class FNOFireModel(nn.Module):
    """FNO-based fire-spread predictor (no PI).

    ``neuralop.models.FNO`` wrapped to our data interface. PI losses are
    layered on top externally — this class is the pure FNO.

    Args:
        n_modes: Fourier truncation modes ``(mx, my, mz)``. Each entry
            must be strictly less than the matching ``GRID_SHAPE`` entry.
        in_channels: Input channels (5: T, V, CO, mask, time_enc).
        out_channels: Output channels (3: T, V, CO).
        hidden_channels: FNO internal width.
        n_layers: Number of FNO blocks.
        lifting_channels: Width of the lifting projection. Must be a
            positive multiple of ``hidden_channels`` so it converts
            cleanly to ``lifting_channel_ratio``.
        projection_channels: Width of the final projection. Same constraint
            as ``lifting_channels``.
        use_sigmoid_output: If ``True`` apply ``sigmoid`` to clamp output
            to ``[0, 1]`` (useful once PI losses are active and we need
            to guarantee normalised output).
    """

    def __init__(
        self,
        n_modes: Tuple[int, int, int] = (12, 12, 4),
        in_channels: int = N_INPUT_CHANNELS,
        out_channels: int = N_OUTPUT_CHANNELS,
        hidden_channels: int = 32,
        n_layers: int = 4,
        lifting_channels: int = 128,
        projection_channels: int = 128,
        use_sigmoid_output: bool = False,
    ) -> None:
        super().__init__()

        # Validate modes against the project grid.
        if len(n_modes) != 3:
            raise ValueError(f"n_modes must be 3-tuple, got {n_modes}")
        for axis, (m, g) in enumerate(zip(n_modes, GRID_SHAPE)):
            if m >= g:
                raise ValueError(
                    f"n_modes[{axis}]={m} must be < GRID_SHAPE[{axis}]={g}"
                )
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be >0, got {hidden_channels}")
        if lifting_channels <= 0 or projection_channels <= 0:
            raise ValueError("lifting/projection_channels must be positive")
        if lifting_channels % hidden_channels != 0:
            raise ValueError(
                f"lifting_channels={lifting_channels} must be a multiple of "
                f"hidden_channels={hidden_channels}"
            )
        if projection_channels % hidden_channels != 0:
            raise ValueError(
                f"projection_channels={projection_channels} must be a multiple of "
                f"hidden_channels={hidden_channels}"
            )

        self.n_modes = tuple(n_modes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.use_sigmoid_output = use_sigmoid_output

        # ``neuralop`` 2.0 takes channel *ratios* rather than absolute widths.
        lifting_ratio = lifting_channels // hidden_channels
        projection_ratio = projection_channels // hidden_channels

        self.fno = FNO(
            n_modes=self.n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_ratio,
            projection_channel_ratio=projection_ratio,
        )

        self.sigmoid = nn.Sigmoid() if use_sigmoid_output else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass.

        Args:
            x: ``(B, 5, 60, 40, 6)`` float, normalised to ``[0, 1]``.

        Returns:
            ``(B, 3, 60, 40, 6)`` next-timestep prediction (normalised).
        """
        if x.dim() != 5:
            raise ValueError(f"expected 5-D input (B,C,X,Y,Z), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"input channels {x.shape[1]} != {self.in_channels}"
            )
        out = self.fno(x)
        if self.sigmoid is not None:
            out = self.sigmoid(out)
        return out

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_default_fno() -> FNOFireModel:
    """Construct the canonical Section-4.3.2 FNO."""
    return FNOFireModel(
        n_modes=(12, 12, 4),
        in_channels=5,
        out_channels=3,
        hidden_channels=32,
        n_layers=4,
        lifting_channels=128,
        projection_channels=128,
    )


# ─── Self-test (run with ``python -m src.models.fno_model``) ─────────────
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("FNO 모델 검증 (mock 데이터)")
    print("=" * 70)

    # === Test 1: 모델 생성 + 파라미터 수 ============================
    print("\n[Test 1] 모델 생성")
    model = build_default_fno()
    n_params = model.count_parameters()
    print(f"  ✓ 모델 인스턴스 생성")
    print(f"  파라미터 수: {n_params:,} (예상 1.5M~4M)")
    assert 1_000_000 < n_params < 6_000_000, (
        f"파라미터 수가 예상 범위 밖: {n_params}"
    )
    print(f"  ✓ 파라미터 수 정상")

    # === Test 2: Forward pass ========================================
    print("\n[Test 2] Forward pass (랜덤 입력)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    model = model.to(device)

    batch_size = 2
    mock_input = torch.rand(batch_size, 5, 60, 40, 6, device=device, dtype=torch.float32)
    print(f"  입력 shape: {tuple(mock_input.shape)}")
    print(f"  입력 범위: [{mock_input.min():.4f}, {mock_input.max():.4f}]")

    model.eval()
    with torch.no_grad():
        start = time.time()
        output = model(mock_input)
        elapsed = time.time() - start

    print(f"  ✓ 출력 shape: {tuple(output.shape)}")
    expected_shape = (batch_size, 3, 60, 40, 6)
    assert tuple(output.shape) == expected_shape, (
        f"출력 shape mismatch: {output.shape} != {expected_shape}"
    )
    print(f"  ✓ forward 시간: {elapsed * 1000:.2f} ms ({batch_size} samples)")

    # === Test 3: Backward pass =======================================
    print("\n[Test 3] Backward pass")
    model.train()
    mock_target = torch.rand(batch_size, 3, 60, 40, 6, device=device, dtype=torch.float32)
    output = model(mock_input)
    loss = torch.nn.functional.mse_loss(output, mock_target)
    loss.backward()
    print(f"  ✓ loss: {loss.item():.4f}")

    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total_param = sum(1 for _ in model.parameters())
    print(f"  ✓ grad 있는 파라미터: {n_with_grad}/{n_total_param}")
    assert n_with_grad == n_total_param, "일부 파라미터가 grad 없음 — 모델 연결 문제"

    # === Test 4: 배치 크기별 메모리 ==================================
    print("\n[Test 4] 배치 크기별 메모리 사용량")
    if torch.cuda.is_available():
        for bs in (1, 2, 4, 8):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            mock_in = torch.rand(bs, 5, 60, 40, 6, device=device)
            mock_tgt = torch.rand(bs, 3, 60, 40, 6, device=device)
            model.zero_grad()
            out = model(mock_in)
            loss = torch.nn.functional.mse_loss(out, mock_tgt)
            loss.backward()
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  배치 크기 {bs}: peak memory = {mem_mb:.1f} MB")
            del mock_in, mock_tgt, out, loss
            torch.cuda.empty_cache()
    else:
        print("  (GPU 없음, 메모리 측정 스킵)")

    # === Test 5: Sigmoid 옵션 ========================================
    print("\n[Test 5] Sigmoid 출력 옵션")
    model_sig = FNOFireModel(
        n_modes=(12, 12, 4),
        in_channels=5,
        out_channels=3,
        hidden_channels=32,
        n_layers=4,
        use_sigmoid_output=True,
    ).to(device)
    with torch.no_grad():
        out_sig = model_sig(mock_input)
    print(f"  ✓ sigmoid 출력 범위: [{out_sig.min():.4f}, {out_sig.max():.4f}]")
    assert out_sig.min() >= 0.0 and out_sig.max() <= 1.0, "sigmoid 출력이 [0, 1] 밖"
    print(f"  ✓ sigmoid 출력이 [0, 1] 범위")

    # === Test 6: 1배치 과적합 ========================================
    print("\n[Test 6] 1배치 과적합 sanity check (30 epoch)")
    model = build_default_fno().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fixed_input = torch.rand(2, 5, 60, 40, 6, device=device)
    fixed_target = torch.rand(2, 3, 60, 40, 6, device=device)
    losses = []
    for epoch in range(30):
        optimizer.zero_grad()
        output = model(fixed_input)
        loss = torch.nn.functional.mse_loss(output, fixed_target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"  초기 loss: {losses[0]:.6f}")
    print(f"  10 epoch:  {losses[9]:.6f}")
    print(f"  30 epoch:  {losses[-1]:.6f}")
    assert losses[-1] < losses[0] * 0.5, (
        f"학습 안 됨: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
    print(f"  ✓ 학습 가능 확인 (loss {losses[0] / losses[-1]:.1f}배 감소)")

    print("\nPASS: FNO 모델 검증 완료")
    print("\n다음 단계: src/training/train_fno.py 작성 (Session 2)")
