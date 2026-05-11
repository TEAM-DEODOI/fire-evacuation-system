"""
Apply channel-wise normalisation and assemble model-input/target tensors.

This wraps the per-element functions in ``src.shared.normalization`` and
the broadcasting glue required to go from ``extract_slices`` output
(raw physical units) to the canonical training-tensor shapes:

* input  ``(31, 5, 60, 40, 6)`` — channels ``[T_norm, V_norm, CO_norm, mask, time_enc]``
* target ``(31, 3, 60, 40, 6)`` — channels ``[T_norm, V_norm, CO_norm]``

All channels live in ``[0, 1]``. The visibility channel uses the inverse
mapping ``1 - V/30`` (low visibility → high danger) per
``docs/interface_contracts.md`` §1.1.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.shared.constants import (
    DT_SLCF,
    GRID_SHAPE,
    N_INPUT_CHANNELS,
    N_OUTPUT_CHANNELS,
    N_TIMESTEPS,
    T_MIN_SECONDS,
)
from src.shared.normalization import (
    compute_time_encoding,
    normalize_co,
    normalize_temperature,
    normalize_visibility,
)

# Both interface_contracts.md ("co_ppm") and fds_extractor ("co") appear in
# the codebase. ``normalize_scenario`` accepts either spelling for the CO key.
_CO_KEYS: tuple[str, ...] = ("co", "co_ppm")


def _resolve_co(raw_fields: Dict[str, np.ndarray]) -> np.ndarray:
    """Return the CO array regardless of which key the upstream code used."""
    for k in _CO_KEYS:
        if k in raw_fields:
            return raw_fields[k]
    raise ValueError(
        f"raw_fields missing CO data — expected one of {_CO_KEYS}, "
        f"got keys {sorted(raw_fields.keys())}"
    )


def _expected_field_shape() -> tuple[int, int, int, int]:
    return (N_TIMESTEPS, *GRID_SHAPE)


def normalize_scenario(
    raw_fields: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Normalise the three raw fields of a single scenario.

    Args:
        raw_fields: ``{"temperature": (31,60,40,6), "visibility": (...),
            "co" or "co_ppm": (...)}`` — raw physical units (°C, m, ppm).

    Returns:
        Dict with keys ``"temperature"``, ``"visibility"``, ``"co"``. Each
        array has the same shape as the input but values normalised to
        ``[0, 1]`` (``float32``).

    Raises:
        ValueError: If any required field is missing or has wrong shape.
    """
    expected = _expected_field_shape()

    if "temperature" not in raw_fields:
        raise ValueError("raw_fields missing 'temperature'")
    if "visibility" not in raw_fields:
        raise ValueError("raw_fields missing 'visibility'")

    T = np.asarray(raw_fields["temperature"])
    V = np.asarray(raw_fields["visibility"])
    CO = np.asarray(_resolve_co(raw_fields))

    for name, arr in (("temperature", T), ("visibility", V), ("co", CO)):
        if arr.shape != expected:
            raise ValueError(
                f"{name} shape {arr.shape} != expected {expected}"
            )

    return {
        "temperature": normalize_temperature(T).astype(np.float32, copy=False),
        "visibility": normalize_visibility(V).astype(np.float32, copy=False),
        "co": normalize_co(CO).astype(np.float32, copy=False),
    }


def _time_encoding_per_frame(
    times: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the per-frame time encoding ``(N_TIMESTEPS,)`` in ``[0, 1]``."""
    if times is None:
        times = np.arange(
            T_MIN_SECONDS, T_MIN_SECONDS + N_TIMESTEPS * DT_SLCF, DT_SLCF,
        )
    times = np.asarray(times, dtype=np.float64)
    if times.shape != (N_TIMESTEPS,):
        raise ValueError(
            f"times must have shape ({N_TIMESTEPS},), got {times.shape}"
        )
    return np.array(
        [compute_time_encoding(float(t)) for t in times], dtype=np.float32
    )


def build_input_tensor(
    normalised: Dict[str, np.ndarray],
    mask: np.ndarray,
    times: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stack the five input channels into ``(31, 5, 60, 40, 6)``.

    Args:
        normalised: Output of :func:`normalize_scenario`.
        mask: ``(60, 40, 6)`` fluid (1.0) / solid (0.0) mask.
        times: Optional ``(31,)`` real-frame times for time encoding. If
            ``None`` the canonical schedule (0, 10, …, 300 s) is used.

    Returns:
        ``(31, 5, 60, 40, 6)`` ``float32``, values in ``[0, 1]``.

    Raises:
        ValueError: If any input has a wrong shape.
    """
    expected_field = _expected_field_shape()
    for k in ("temperature", "visibility", "co"):
        arr = normalised[k]
        if arr.shape != expected_field:
            raise ValueError(
                f"normalised['{k}'] shape {arr.shape} != expected {expected_field}"
            )
    if mask.shape != GRID_SHAPE:
        raise ValueError(f"mask shape {mask.shape} != {GRID_SHAPE}")

    mask_b = np.broadcast_to(
        mask.astype(np.float32, copy=False)[None, :, :, :], expected_field,
    ).astype(np.float32)

    te = _time_encoding_per_frame(times)
    te_grid = np.broadcast_to(
        te[:, None, None, None], expected_field,
    ).astype(np.float32)

    out = np.stack(
        [
            normalised["temperature"],
            normalised["visibility"],
            normalised["co"],
            mask_b,
            te_grid,
        ],
        axis=1,
    ).astype(np.float32)

    expected_out = (N_TIMESTEPS, N_INPUT_CHANNELS, *GRID_SHAPE)
    if out.shape != expected_out:
        raise RuntimeError(
            f"internal: input tensor shape {out.shape} != {expected_out}"
        )
    return out


def build_target_tensor(normalised: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack the three target channels into ``(31, 3, 60, 40, 6)``.

    Target = normalised ``[T, V, CO]`` (mask and time encoding are not
    predicted — they are known inputs).
    """
    expected_field = _expected_field_shape()
    for k in ("temperature", "visibility", "co"):
        arr = normalised[k]
        if arr.shape != expected_field:
            raise ValueError(
                f"normalised['{k}'] shape {arr.shape} != expected {expected_field}"
            )
    out = np.stack(
        [normalised["temperature"], normalised["visibility"], normalised["co"]],
        axis=1,
    ).astype(np.float32)
    expected_out = (N_TIMESTEPS, N_OUTPUT_CHANNELS, *GRID_SHAPE)
    if out.shape != expected_out:
        raise RuntimeError(
            f"internal: target tensor shape {out.shape} != {expected_out}"
        )
    return out


def build_input_and_target(
    raw_fields: Dict[str, np.ndarray],
    mask: np.ndarray,
    times: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot: raw scenario → ``(input, target)`` tensor pair."""
    normalised = normalize_scenario(raw_fields)
    inp = build_input_tensor(normalised, mask, times)
    tgt = build_target_tensor(normalised)
    return inp, tgt


# ─── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path

    print("=" * 60)
    print("normalize.py self-test")
    print("=" * 60)

    errors: list[str] = []
    rng = np.random.default_rng(0)
    nx, ny, nz = GRID_SHAPE
    expected_field = (N_TIMESTEPS, nx, ny, nz)

    # ── 1. normalize_scenario shape + range ─────────────────────────────
    print("\n[1] normalize_scenario on synthetic raw fields")
    raw = {
        "temperature": rng.uniform(20.0, 800.0, expected_field).astype(np.float32),
        "visibility":  rng.uniform(0.0, 30.0,  expected_field).astype(np.float32),
        "co":          rng.uniform(0.0, 3000.0, expected_field).astype(np.float32),
    }
    norm = normalize_scenario(raw)
    for k in ("temperature", "visibility", "co"):
        arr = norm[k]
        if arr.shape != expected_field:
            errors.append(f"{k} shape {arr.shape}")
        if arr.dtype != np.float32:
            errors.append(f"{k} dtype {arr.dtype}")
        if not (arr.min() >= 0.0 and arr.max() <= 1.0):
            errors.append(f"{k} out of [0, 1]: [{arr.min()}, {arr.max()}]")
        print(f"  {k}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")

    # Inverse visibility check
    raw_vis_30 = raw["visibility"][0, 0, 0, 0]
    v_norm_at = norm["visibility"][0, 0, 0, 0]
    print(f"  Visibility inverse check: V[0,0,0,0]={raw_vis_30:.2f}m → V_norm={v_norm_at:.4f}")

    # Accept 'co_ppm' alias
    raw_alias = dict(raw)
    raw_alias["co_ppm"] = raw_alias.pop("co")
    norm_alias = normalize_scenario(raw_alias)
    if not np.allclose(norm_alias["co"], norm["co"]):
        errors.append("'co_ppm' alias should produce same output as 'co'")
    else:
        print("  'co_ppm' alias key accepted")

    # ── 2. build_input_tensor ───────────────────────────────────────────
    print("\n[2] build_input_tensor")
    mask = np.ones(GRID_SHAPE, dtype=np.float32)
    mask[10:15, 5:10, :] = 0.0  # carve a wall
    inp = build_input_tensor(norm, mask)
    expected_in = (N_TIMESTEPS, N_INPUT_CHANNELS, nx, ny, nz)
    if inp.shape != expected_in:
        errors.append(f"input shape {inp.shape} != {expected_in}")
    if inp.dtype != np.float32:
        errors.append(f"input dtype {inp.dtype}")
    if not (inp.min() >= 0.0 and inp.max() <= 1.0):
        errors.append(f"input out of [0, 1]")
    print(f"  shape={inp.shape}, dtype={inp.dtype}, range=[{inp.min():.4f}, {inp.max():.4f}]")
    # mask channel preserved (broadcast across time)
    if not np.array_equal(inp[0, 3], mask):
        errors.append("mask channel not preserved at frame 0")
    if not np.array_equal(inp[-1, 3], mask):
        errors.append("mask channel not preserved at last frame")
    # time encoding spans 0 → 1
    te_first = float(inp[0, 4, 0, 0, 0])
    te_last = float(inp[-1, 4, 0, 0, 0])
    if abs(te_first) > 1e-6:
        errors.append(f"time_enc[0] = {te_first}, expected 0")
    if abs(te_last - 1.0) > 1e-6:
        errors.append(f"time_enc[-1] = {te_last}, expected 1")
    print(f"  time_enc[0]={te_first:.4f}  time_enc[-1]={te_last:.4f}")

    # ── 3. build_target_tensor ──────────────────────────────────────────
    print("\n[3] build_target_tensor")
    tgt = build_target_tensor(norm)
    expected_tg = (N_TIMESTEPS, N_OUTPUT_CHANNELS, nx, ny, nz)
    if tgt.shape != expected_tg:
        errors.append(f"target shape {tgt.shape} != {expected_tg}")
    if not (tgt.min() >= 0.0 and tgt.max() <= 1.0):
        errors.append("target out of [0, 1]")
    # First three channels of input should equal target
    if not np.array_equal(inp[:, :3], tgt):
        errors.append("input[:, :3] != target")
    print(f"  shape={tgt.shape}, range=[{tgt.min():.4f}, {tgt.max():.4f}]")
    print(f"  input[:, :3] == target: True")

    # ── 4. build_input_and_target one-shot ──────────────────────────────
    print("\n[4] build_input_and_target (one-shot)")
    inp2, tgt2 = build_input_and_target(raw, mask)
    if not np.array_equal(inp2, inp):
        errors.append("one-shot input differs from two-step")
    if not np.array_equal(tgt2, tgt):
        errors.append("one-shot target differs from two-step")
    print("  one-shot matches two-step")

    # ── 5. Shape-mismatch errors ────────────────────────────────────────
    print("\n[5] Input-validation errors")
    try:
        normalize_scenario({"temperature": np.zeros((10, 5)), "visibility": raw["visibility"], "co": raw["co"]})
        errors.append("expected ValueError on bad T shape")
    except ValueError:
        print("  PASS: bad shape raises ValueError")
    try:
        normalize_scenario({"temperature": raw["temperature"], "visibility": raw["visibility"]})
        errors.append("expected ValueError on missing CO")
    except ValueError:
        print("  PASS: missing CO raises ValueError")

    # ── 6. Real first_sim end-to-end (if available) ─────────────────────
    print("\n[6] Real first_sim end-to-end")
    real_dir = Path("data/raw/first_sim")
    if real_dir.is_dir():
        from src.data_pipeline.fds_extractor import extract_slices
        from src.data_pipeline.mask_generator import generate_mask_from_fds

        slices = extract_slices(real_dir)
        real_mask = generate_mask_from_fds(real_dir)
        real_inp, real_tgt = build_input_and_target(
            slices, real_mask, times=slices["times"]
        )
        print(f"  input  shape: {real_inp.shape}  range=[{real_inp.min():.4f}, {real_inp.max():.4f}]")
        print(f"  target shape: {real_tgt.shape}  range=[{real_tgt.min():.4f}, {real_tgt.max():.4f}]")
        if real_inp.shape != (N_TIMESTEPS, 5, *GRID_SHAPE):
            errors.append("real input shape mismatch")
        if not (real_inp.min() >= 0.0 and real_inp.max() <= 1.0):
            errors.append("real input out of [0, 1]")
        # Mask channel should equal real_mask broadcast across time
        if not np.array_equal(real_inp[0, 3], real_mask):
            errors.append("real mask channel not equal to mask")
    else:
        print("  SKIP: data/raw/first_sim not present")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("\nPASS: normalize validated")
