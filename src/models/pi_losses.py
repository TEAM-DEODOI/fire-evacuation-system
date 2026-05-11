"""Physics-Informed FNO 손실 함수 5종 + 통합 클래스 (curriculum learning).

매뉴얼 Section 4.4.3 / Day-9-Session-2 사양:

1. ``data_loss``        — MSE (가중치 1.0, 항상 ON)
2. ``boundary_loss``    — 벽 내부 셀 값 0 강제 (가중치 0.1, Stage 1~)
3. ``monotonicity_loss``— 활성화된 영역의 급격한 감소 페널티 (0.05, Stage 2~)
4. ``nonnegative_loss`` — 출력 ≥ 0 강제 (0.01, Stage 3~)
5. ``pde_residual_loss``— 열확산 방정식 잔차 (0.01, Stage 4 optional)

Curriculum 4 stages (epoch 0–30 / 30–60 / 60–80 / 80–100). PDE stage가
발산하면 호출자가 Stage 3 모델로 폴백해야 함 — 매뉴얼 Section 4.4.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# 개별 손실 함수 5종
# ─────────────────────────────────────────────────────────────────────────
def data_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """단순 MSE.

    Args:
        pred: ``(B, 3, X, Y, Z)`` 정규화된 예측.
        target: ``(B, 3, X, Y, Z)`` 정규화된 정답.

    Returns:
        scalar loss.
    """
    return F.mse_loss(pred, target)


def boundary_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """벽 내부 셀 값이 0이 되도록 강제 — ``mean((pred * (1 - mask))²)``.

    Args:
        pred: ``(B, 3, X, Y, Z)``.
        mask: ``(B, 1, X, Y, Z)`` 또는 ``(B, X, Y, Z)``.
            ``1.0`` = 공기 셀, ``0.0`` = 벽.

    Returns:
        scalar loss.
    """
    if mask.dim() == 4:
        mask = mask.unsqueeze(1)  # (B, 1, X, Y, Z)
    wall_mask = 1.0 - mask
    wall_pred = pred * wall_mask
    return (wall_pred ** 2).mean()


def monotonicity_loss(
    pred: torch.Tensor,
    prev_input: torch.Tensor,
    activation_threshold: float = 0.3,
) -> torch.Tensor:
    """활성화된 영역(이전 위험도 > threshold)에서 ``pred < prev`` 페널티.

    Args:
        pred: ``(B, 3, X, Y, Z)`` t+1 예측.
        prev_input: ``(B, 3, X, Y, Z)`` t 입력의 T, V, CO 채널.
        activation_threshold: 활성화 임계값 (정규화된 위험도, 기본 0.3).

    Returns:
        scalar loss.
    """
    activated = (prev_input > activation_threshold).float()
    decrease = F.relu(prev_input - pred)
    return ((decrease * activated) ** 2).mean()


def nonnegative_loss(pred: torch.Tensor) -> torch.Tensor:
    """음수 출력 페널티 — ``mean(ReLU(-pred)²)``."""
    return (F.relu(-pred) ** 2).mean()


def laplacian_3d(field: torch.Tensor, dx: float = 0.5) -> torch.Tensor:
    """3D 라플라시안 (중심 유한차분, zero-padding 경계).

    Args:
        field: ``(B, C, X, Y, Z)``.
        dx: 격자 간격 (m). 기본 0.5 m (CELL_SIZE_M).

    Returns:
        같은 shape의 라플라시안. 외곽 셀은 0으로 패딩되어 계산됨.
    """
    pad = (1, 1, 1, 1, 1, 1)  # (Z_lo, Z_hi, Y_lo, Y_hi, X_lo, X_hi)
    fp = F.pad(field, pad, mode="constant", value=0.0)
    d2x = (
        fp[:, :, 2:, 1:-1, 1:-1]
        - 2 * fp[:, :, 1:-1, 1:-1, 1:-1]
        + fp[:, :, :-2, 1:-1, 1:-1]
    ) / (dx ** 2)
    d2y = (
        fp[:, :, 1:-1, 2:, 1:-1]
        - 2 * fp[:, :, 1:-1, 1:-1, 1:-1]
        + fp[:, :, 1:-1, :-2, 1:-1]
    ) / (dx ** 2)
    d2z = (
        fp[:, :, 1:-1, 1:-1, 2:]
        - 2 * fp[:, :, 1:-1, 1:-1, 1:-1]
        + fp[:, :, 1:-1, 1:-1, :-2]
    ) / (dx ** 2)
    return d2x + d2y + d2z


def pde_residual_loss(
    pred: torch.Tensor,
    prev_input: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.1,
    dt: float = 10.0,
    dx: float = 0.5,
    channel_idx: int = 0,
) -> torch.Tensor:
    """열확산 방정식 잔차 ``∂T/∂t = α∇²T`` 의 squared mean.

    Args:
        pred: ``(B, 3, X, Y, Z)`` t+1 예측 (정규화).
        prev_input: ``(B, 3, X, Y, Z)`` t 입력.
        mask: 공기 셀에만 PDE 적용 — ``(B, 1, X, Y, Z)`` 또는 ``(B, X, Y, Z)``.
        alpha: 열확산 계수 (정규화 좌표계). 학부생 수준: 0.01~1.0 grid search.
        dt: 시간 간격 (s).
        dx: 격자 간격 (m).
        channel_idx: T 채널 인덱스 (보통 0).

    Returns:
        scalar loss.
    """
    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    T_pred = pred[:, channel_idx:channel_idx + 1, ...]
    T_prev = prev_input[:, channel_idx:channel_idx + 1, ...]
    dT_dt = (T_pred - T_prev) / dt
    laplacian = laplacian_3d(T_prev, dx=dx)
    residual = (dT_dt - alpha * laplacian) * mask
    return (residual ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────
# 통합 손실 클래스 + Curriculum
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class PILossWeights:
    """5개 손실의 가중치. Curriculum 단계에 따라 동적 변경."""

    w_data: float = 1.0
    w_boundary: float = 0.1
    w_mono: float = 0.0       # Stage 2부터
    w_nonneg: float = 0.0     # Stage 3부터
    w_pde: float = 0.0        # Stage 4부터 (선택)


@dataclass
class CurriculumStage:
    """Curriculum 단계 정의 (매뉴얼 Section 4.4.2)."""

    epoch_start: int
    epoch_end: int
    weights: PILossWeights
    name: str


CURRICULUM_STAGES: Tuple[CurriculumStage, ...] = (
    CurriculumStage(
        0, 30,
        PILossWeights(w_data=1.0, w_boundary=0.1),
        "Stage 1: data + boundary",
    ),
    CurriculumStage(
        30, 60,
        PILossWeights(w_data=1.0, w_boundary=0.1, w_mono=0.05),
        "Stage 2: + monotonicity",
    ),
    CurriculumStage(
        60, 80,
        PILossWeights(w_data=1.0, w_boundary=0.1, w_mono=0.05, w_nonneg=0.01),
        "Stage 3: + nonnegative",
    ),
    CurriculumStage(
        80, 100,
        PILossWeights(
            w_data=1.0, w_boundary=0.1, w_mono=0.05,
            w_nonneg=0.01, w_pde=0.01,
        ),
        "Stage 4: + PDE (optional)",
    ),
)


class PIFNOLoss(nn.Module):
    """5개 PI 손실의 통합 클래스. 외부에서 ``set_stage_by_epoch`` 호출."""

    def __init__(
        self,
        weights: Optional[PILossWeights] = None,
        activation_threshold: float = 0.3,
        pde_alpha: float = 0.1,
        pde_dt: float = 10.0,
        pde_dx: float = 0.5,
    ) -> None:
        super().__init__()
        self.weights = weights if weights is not None else PILossWeights()
        self.activation_threshold = activation_threshold
        self.pde_alpha = pde_alpha
        self.pde_dt = pde_dt
        self.pde_dx = pde_dx

    def set_weights(self, weights: PILossWeights) -> None:
        """Curriculum 단계 변경 시 호출."""
        self.weights = weights

    def set_stage_by_epoch(self, epoch: int) -> CurriculumStage:
        """현재 epoch에 맞는 stage 자동 설정 + 반환."""
        for stage in CURRICULUM_STAGES:
            if stage.epoch_start <= epoch < stage.epoch_end:
                self.set_weights(stage.weights)
                return stage
        last = CURRICULUM_STAGES[-1]
        self.set_weights(last.weights)
        return last

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prev_input: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """5개 손실을 가중합. wandb-friendly dict 도 함께 반환."""
        w = self.weights
        loss_dict: dict = {}

        l_data = data_loss(pred, target)
        loss_dict["data"] = float(l_data.item())

        l_boundary = boundary_loss(pred, mask)
        loss_dict["boundary"] = float(l_boundary.item())

        l_mono = torch.tensor(0.0, device=pred.device)
        if w.w_mono > 0:
            l_mono = monotonicity_loss(pred, prev_input, self.activation_threshold)
        loss_dict["mono"] = float(l_mono.item())

        l_nonneg = torch.tensor(0.0, device=pred.device)
        if w.w_nonneg > 0:
            l_nonneg = nonnegative_loss(pred)
        loss_dict["nonneg"] = float(l_nonneg.item())

        l_pde = torch.tensor(0.0, device=pred.device)
        if w.w_pde > 0:
            l_pde = pde_residual_loss(
                pred, prev_input, mask,
                alpha=self.pde_alpha, dt=self.pde_dt, dx=self.pde_dx,
            )
        loss_dict["pde"] = float(l_pde.item())

        total = (
            w.w_data * l_data
            + w.w_boundary * l_boundary
            + w.w_mono * l_mono
            + w.w_nonneg * l_nonneg
            + w.w_pde * l_pde
        )
        loss_dict["total"] = float(total.item())
        return total, loss_dict


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("PI 손실 5종 단위 테스트")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    B, C, X, Y, Z = 2, 3, 60, 40, 6
    torch.manual_seed(0)
    pred = torch.rand(B, C, X, Y, Z, device=device, requires_grad=True)
    target = torch.rand(B, C, X, Y, Z, device=device)
    prev = torch.rand(B, C, X, Y, Z, device=device)
    mask = torch.ones(B, 1, X, Y, Z, device=device)
    mask[:, :, :5, :5, :] = 0.0  # carve a wall region

    # Test 1: data_loss ────────────────────────────────────────────────
    print("[Test 1] data_loss (MSE)")
    l = data_loss(pred, target)
    print(f"  ✓ scalar loss: {l.item():.6f}")
    assert l.dim() == 0

    # Test 2: boundary_loss ────────────────────────────────────────────
    print("[Test 2] boundary_loss")
    l = boundary_loss(pred, mask)
    print(f"  ✓ scalar loss: {l.item():.6f}")
    pred_zero_at_wall = pred.clone().detach()
    pred_zero_at_wall[:, :, :5, :5, :] = 0.0
    l_zero = boundary_loss(pred_zero_at_wall, mask)
    assert l_zero < 1e-6, f"벽 영역 0일 때 loss > 0: {l_zero}"
    print("  ✓ 벽 영역 0 강제 시 loss → 0 확인")

    # Test 3: monotonicity_loss ────────────────────────────────────────
    print("[Test 3] monotonicity_loss")
    l = monotonicity_loss(pred, prev)
    print(f"  ✓ scalar loss: {l.item():.6f}")
    l_same = monotonicity_loss(prev, prev)
    assert l_same < 1e-6, f"pred=prev일 때 loss > 0: {l_same}"
    print("  ✓ pred==prev일 때 loss → 0 확인")

    # Test 4: nonnegative_loss ─────────────────────────────────────────
    print("[Test 4] nonnegative_loss")
    pos_pred = torch.rand(B, C, X, Y, Z, device=device)
    l = nonnegative_loss(pos_pred)
    assert l < 1e-6, f"양수만일 때 loss > 0: {l}"
    print("  ✓ 양수만일 때 loss → 0 확인")
    neg_pred = -torch.ones(B, C, X, Y, Z, device=device)
    l_neg = nonnegative_loss(neg_pred)
    assert l_neg > 0.5, f"음수만일 때 loss={l_neg}, ≥ 0.5 기대"
    print("  ✓ 음수일 때 loss > 0 확인")

    # Test 5: laplacian_3d ─────────────────────────────────────────────
    print("[Test 5] laplacian_3d")
    const_field = torch.ones(B, C, X, Y, Z, device=device) * 5.0
    lap_const = laplacian_3d(const_field)
    interior = lap_const[:, :, 2:-2, 2:-2, 2:-2]
    assert interior.abs().max() < 1e-5, (
        f"상수 필드 라플라시안 0 아님: {interior.abs().max()}"
    )
    print("  ✓ 상수 필드 → 라플라시안 0 (내부) 확인")

    # Test 6: pde_residual_loss ────────────────────────────────────────
    print("[Test 6] pde_residual_loss")
    l = pde_residual_loss(pred, prev, mask)
    print(f"  ✓ scalar loss: {l.item():.6f}")
    assert l >= 0

    # Test 7: PIFNOLoss 통합 ───────────────────────────────────────────
    print("\n[Test 7] PIFNOLoss 통합")
    loss_fn = PIFNOLoss().to(device)

    print("\n  - Stage 1 (epoch=0):")
    stage = loss_fn.set_stage_by_epoch(0)
    print(f"    {stage.name}")
    print(
        f"    weights: data={loss_fn.weights.w_data}, "
        f"boundary={loss_fn.weights.w_boundary}, mono={loss_fn.weights.w_mono}"
    )
    total, ld = loss_fn(pred, target, prev, mask)
    print(f"    total={total.item():.6f}, dict={ld}")
    assert ld["mono"] == 0.0 and ld["nonneg"] == 0.0 and ld["pde"] == 0.0
    print("    ✓ Stage 1에서 mono, nonneg, pde 0")

    print("\n  - Stage 2 (epoch=30):")
    stage = loss_fn.set_stage_by_epoch(30)
    print(f"    {stage.name}")
    total, ld = loss_fn(pred, target, prev, mask)
    print(f"    total={total.item():.6f}, dict={ld}")
    assert ld["mono"] >= 0.0
    print("    ✓ Stage 2부터 mono 활성")

    print("\n  - Stage 3 (epoch=60):")
    loss_fn.set_stage_by_epoch(60)
    total, ld = loss_fn(pred, target, prev, mask)
    print(f"    weights: nonneg={loss_fn.weights.w_nonneg}")
    print(f"    dict={ld}")

    print("\n  - Stage 4 (epoch=80):")
    loss_fn.set_stage_by_epoch(80)
    total, ld = loss_fn(pred, target, prev, mask)
    print(f"    weights: pde={loss_fn.weights.w_pde}")
    print(f"    dict={ld}")

    # Test 8: backward 가능성 ──────────────────────────────────────────
    print("\n[Test 8] backward 가능 여부 (모든 stage)")
    for epoch in (0, 30, 60, 80):
        loss_fn.set_stage_by_epoch(epoch)
        pred_grad = torch.rand(B, C, X, Y, Z, device=device, requires_grad=True)
        total, _ = loss_fn(pred_grad, target, prev, mask)
        total.backward()
        assert pred_grad.grad is not None
        print(
            f"  ✓ Stage epoch={epoch}: backward OK, "
            f"grad norm={pred_grad.grad.norm().item():.4f}"
        )

    print("\nPASS: PI 손실 5종 + 통합 클래스 검증 완료")
