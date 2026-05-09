"""
Loss functions for fire prediction model training.

All losses operate on normalised tensors of shape ``(B, C, 60, 40, 6)`` and
return a scalar loss tensor. Optional ``mask`` arguments restrict the
loss to fluid (non-wall) cells, which is important once a real building
mask is available — otherwise the model is penalised for "wrong" values
inside walls where ground truth is undefined.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast a fluid/solid mask to the shape of ``target``.

    Accepts ``(X, Y, Z)``, ``(1, X, Y, Z)``, ``(B, 1, X, Y, Z)``, or already
    full ``(B, C, X, Y, Z)``. Returns a tensor with the same dtype and
    device as ``target``.
    """
    if mask.dim() == 3:
        mask = mask[None, None, :, :, :]
    elif mask.dim() == 4:
        mask = mask[None, :, :, :, :]
    elif mask.dim() != 5:
        raise ValueError(
            f"mask must be 3-D, 4-D, or 5-D, got shape {tuple(mask.shape)}"
        )
    mask = mask.to(dtype=target.dtype, device=target.device)
    return mask.expand_as(target)


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean squared error, optionally restricted to fluid cells.

    Args:
        pred: ``(B, C, X, Y, Z)``, normalised [0, 1].
        target: ``(B, C, X, Y, Z)``, normalised [0, 1].
        mask: Optional fluid mask. ``1.0`` = fluid (counted), ``0.0`` = solid
            (ignored). Accepted shapes: ``(X, Y, Z)``, ``(1, X, Y, Z)``,
            ``(B, 1, X, Y, Z)``, or full broadcast ``(B, C, X, Y, Z)``.

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If ``pred.shape != target.shape``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )

    if mask is None:
        return F.mse_loss(pred, target)

    mask_b = _expand_mask(mask, target)
    sq_err = (pred - target) ** 2
    denom = mask_b.sum().clamp(min=1.0)
    return (sq_err * mask_b).sum() / denom


def mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean absolute error, optionally restricted to fluid cells.

    Args:
        pred: ``(B, C, X, Y, Z)``.
        target: ``(B, C, X, Y, Z)``.
        mask: See :func:`mse_loss`.

    Returns:
        Scalar loss tensor.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )

    if mask is None:
        return F.l1_loss(pred, target)

    mask_b = _expand_mask(mask, target)
    abs_err = (pred - target).abs()
    denom = mask_b.sum().clamp(min=1.0)
    return (abs_err * mask_b).sum() / denom


def channel_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Iterable[float],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted sum of per-channel MSE losses.

    Args:
        pred: ``(B, C, X, Y, Z)``.
        target: ``(B, C, X, Y, Z)``.
        weights: Length-C iterable of per-channel weights. Need not sum to 1.
        mask: Optional fluid mask. Same conventions as :func:`mse_loss`.

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If ``len(weights) != pred.shape[1]``.
    """
    weights_t = torch.as_tensor(list(weights), dtype=pred.dtype, device=pred.device)
    n_channels = pred.shape[1]
    if weights_t.numel() != n_channels:
        raise ValueError(
            f"len(weights)={weights_t.numel()} != pred channels={n_channels}"
        )

    total: Optional[torch.Tensor] = None
    for c in range(n_channels):
        # Slice keeps the channel dim as 1 so mask broadcasting still works.
        per_ch = mse_loss(pred[:, c : c + 1], target[:, c : c + 1], mask)
        contribution = weights_t[c] * per_ch
        total = contribution if total is None else total + contribution
    assert total is not None  # n_channels >= 1
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run with ``python -m src.training.losses``).
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("losses.py self-test")
    print("=" * 60)

    torch.manual_seed(0)
    errors: list[str] = []

    # Shapes
    B, C_in, C_out, X, Y, Z = 2, 5, 3, 60, 40, 6
    pred = torch.rand(B, C_out, X, Y, Z)
    target = torch.rand(B, C_out, X, Y, Z)

    # ── Test 1: zero loss when pred == target ───────────────────────────────
    print("\n[Test 1] mse_loss(target, target) == 0")
    z = mse_loss(target.clone(), target).item()
    print(f"  loss = {z:.6e}")
    if z > 1e-12:
        errors.append(f"identity MSE != 0: {z}")

    # ── Test 2: known-value MSE ─────────────────────────────────────────────
    print("\n[Test 2] mse_loss((all-1), (all-0)) == 1.0")
    one = torch.ones(B, C_out, X, Y, Z)
    zero = torch.zeros(B, C_out, X, Y, Z)
    v = mse_loss(one, zero).item()
    print(f"  loss = {v:.6f}")
    if abs(v - 1.0) > 1e-6:
        errors.append(f"all-1 vs all-0 MSE = {v}, expected 1.0")

    # ── Test 3: masked MSE ignores solid cells ──────────────────────────────
    print("\n[Test 3] Masked MSE with half-fluid mask")
    mask = torch.zeros(X, Y, Z)
    mask[: X // 2] = 1.0  # left half = fluid, right half = solid
    # Make pred wrong only on the solid (masked-out) side; loss should be 0.
    pred_partial = target.clone()
    pred_partial[:, :, X // 2 :, :, :] += 5.0  # huge errors only in solid region
    masked = mse_loss(pred_partial, target, mask=mask).item()
    print(f"  masked loss (errors only in solid) = {masked:.6e}  (expected ≈ 0)")
    if masked > 1e-6:
        errors.append(f"masked MSE leaked solid-cell errors: {masked}")

    # And conversely, if errors are in the fluid region, loss > 0.
    pred_fluid_err = target.clone()
    pred_fluid_err[:, :, : X // 2, :, :] += 0.5
    masked_fluid = mse_loss(pred_fluid_err, target, mask=mask).item()
    print(f"  masked loss (errors in fluid)      = {masked_fluid:.6f}  (expected ≈ 0.25)")
    if abs(masked_fluid - 0.25) > 1e-3:
        errors.append(f"masked MSE on fluid: {masked_fluid}, expected 0.25")

    # ── Test 4: MAE basic ───────────────────────────────────────────────────
    print("\n[Test 4] mae_loss((all-1), (all-0)) == 1.0")
    v = mae_loss(one, zero).item()
    print(f"  loss = {v:.6f}")
    if abs(v - 1.0) > 1e-6:
        errors.append(f"MAE all-1 vs all-0 = {v}")

    # ── Test 5: channel_weighted_loss reduces to MSE when uniform weights ──
    print("\n[Test 5] channel_weighted_loss with weights=[1,1,1] equals 3 × MSE per channel")
    plain = mse_loss(pred, target).item()
    cw = channel_weighted_loss(pred, target, weights=[1.0, 1.0, 1.0]).item()
    # plain MSE = (sum sq_err over all channels) / (B*C*X*Y*Z)
    # channel-weighted = sum over c of MSE(c) where MSE(c) = (sum sq_err over c) / (B*X*Y*Z)
    # so cw = C * plain
    expected = C_out * plain
    print(f"  plain MSE × C = {expected:.6f},  channel_weighted = {cw:.6f}")
    if abs(cw - expected) > 1e-5:
        errors.append(f"channel-weighted: got {cw}, expected {expected}")

    # ── Test 6: weight-length mismatch raises ───────────────────────────────
    print("\n[Test 6] Mismatched weight length raises ValueError")
    try:
        channel_weighted_loss(pred, target, weights=[1.0, 1.0])
    except ValueError:
        print("  PASS: ValueError raised")
    else:
        errors.append("mismatched weight length did not raise")

    # ── Test 7: gradients flow ──────────────────────────────────────────────
    print("\n[Test 7] Gradient flow")
    p = torch.rand(B, C_out, X, Y, Z, requires_grad=True)
    loss = mse_loss(p, target, mask=mask)
    loss.backward()
    if p.grad is None or p.grad.abs().sum().item() == 0.0:
        errors.append("no gradient on pred")
    else:
        print(f"  grad norm = {p.grad.norm().item():.4f}")
        print("  PASS")

    # ── Verdict ─────────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("PASS")
    print("=" * 60)
