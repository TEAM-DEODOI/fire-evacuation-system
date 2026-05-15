"""Render a model-vs-truth 2-panel GIF over the **full** scenario timeline.

The decoder needs a 6-frame history (60 s), so for t < 60 s the model
panel shows a "warming up" placeholder. From t = 60 s to t = 250 s the
script runs one decoder forward per 10 s slot and shows the freshest
``lookahead step 0`` prediction. For t > 250 s the dataset has no more
pairs (Tier1FireDataset only enumerates t_start ∈ [0, 19]) so the
panel freezes on the last successful forward.

Output: ``figures/decoder_pred/<scenario>__full.gif``

Run::

    python scripts/visualize_decoder_prediction.py --scenario s_029
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

from src.integration.scenarios.s3_fno_swarm import (
    _DECODER_CKPT_DEFAULT,
    _load_decoder_artifacts,
    _build_decoder_rm,
)
from src.path_planning.building_graph import load_default_fluid_mask
from src.risk_map.risk_map_class import StaticRiskMap
from src.shared.constants import CELL_SIZE_M, DOMAIN_SIZE_M


def _risk_cmap():
    return LinearSegmentedColormap.from_list(
        "risk_overlay",
        [(1, 1, 1, 0), (1, 0.85, 0, 0.55), (0.85, 0, 0, 0.85)],
        N=256,
    )


_DECODER_MIN_T0_S = 60.0
_DECODER_MAX_T0_S = 250.0
_DT_S = 10.0   # FDS frame interval (matches DT_SLCF)


def _render(
    scenario: str,
    decoder_ckpt: Path,
    fps: int,
    dpi: int,
    z_layer: int,
    out_path: Path,
) -> None:
    # 1) Load truth risk map.
    truth_npz = Path(f"results/cache/scenario_risk_maps/{scenario}.npz")
    if not truth_npz.exists():
        raise FileNotFoundError(f"truth cache missing: {truth_npz}")
    truth_rm = StaticRiskMap.from_npy(truth_npz)
    truth_times = truth_rm.times       # 31 frames at 10 s intervals
    truth_danger = truth_rm.danger     # (31, 60, 40, 6)

    # 2) Decoder forwards — one per 10 s slot in [60, 250].
    print(f"[decoder] loading artefacts (ckpt={decoder_ckpt})")
    artefacts = _load_decoder_artifacts(decoder_ckpt=decoder_ckpt, verbose=True)
    if artefacts is None:
        raise RuntimeError(
            "decoder artefacts unavailable — required checkpoints / data missing"
        )

    frame_t = list(truth_times.astype(float))   # full 31-frame timeline 0..300
    n_frames = len(frame_t)
    model_slices: List[np.ndarray] = [None] * n_frames    # type: ignore[assignment]
    truth_slices: List[np.ndarray] = [
        truth_danger[i, :, :, z_layer] for i in range(n_frames)
    ]

    n_ok = 0
    n_fail = 0
    last_good: np.ndarray = None  # type: ignore[assignment]
    cold_zero = np.zeros_like(truth_slices[0])
    for i, t in enumerate(frame_t):
        if t < _DECODER_MIN_T0_S:
            # Cold-start: not enough history yet.
            model_slices[i] = cold_zero
            continue
        if t > _DECODER_MAX_T0_S:
            # Out-of-dataset: hold last good prediction.
            model_slices[i] = last_good if last_good is not None else cold_zero
            continue
        rm = _build_decoder_rm(scenario, float(t), artefacts, verbose=False)
        if rm is None:
            n_fail += 1
            model_slices[i] = last_good if last_good is not None else cold_zero
            continue
        # rm.danger has shape (T_full, X, Y, Z); the first frame (lookahead
        # step 0) is the prediction *at* t0 itself — the value the planner
        # would consult for "now".
        slice_now = rm.danger[0, :, :, z_layer]
        model_slices[i] = slice_now
        last_good = slice_now
        n_ok += 1
        if n_ok <= 3 or n_ok == 20:
            print(
                f"  [decoder] frame {i:>2} t={t:6.1f}s   ok ({n_ok} so far)"
            )

    print(f"[decoder] decoder forwards: {n_ok} ok, {n_fail} failed")

    # 4) 2-panel animation.
    fluid = load_default_fluid_mask()
    walls = 1.0 - fluid[:, :, z_layer].astype(np.float32)
    lx, ly, _lz = DOMAIN_SIZE_M
    cmap = _risk_cmap()

    t_out = n_frames
    fig, (ax_m, ax_t) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, ttl in ((ax_m, f"Model (decoder fn=2.5)  {scenario}"),
                    (ax_t, f"Truth (FDS)  {scenario}")):
        ax.imshow(
            walls.T, cmap="Greys", alpha=0.25,
            extent=(0.0, lx, 0.0, ly), origin="lower",
            interpolation="nearest", zorder=0,
        )
        ax.set_xlim(-0.5, lx + 0.5)
        ax.set_ylim(-0.5, ly + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_title(ttl, fontsize=11)

    m_img = ax_m.imshow(
        model_slices[0].T, cmap=cmap, alpha=0.85,
        extent=(0.0, lx, 0.0, ly), origin="lower",
        vmin=0.0, vmax=1.0, interpolation="bilinear", zorder=1,
    )
    t_img = ax_t.imshow(
        truth_slices[0].T, cmap=cmap, alpha=0.85,
        extent=(0.0, lx, 0.0, ly), origin="lower",
        vmin=0.0, vmax=1.0, interpolation="bilinear", zorder=1,
    )
    plt.colorbar(m_img, ax=ax_m, fraction=0.046, pad=0.04, label="danger")
    plt.colorbar(t_img, ax=ax_t, fraction=0.046, pad=0.04, label="danger")
    sup = fig.suptitle("", fontsize=12)

    def update(i: int):
        m_img.set_data(model_slices[i].T)
        t_img.set_data(truth_slices[i].T)
        wall_t = frame_t[i]
        if wall_t < _DECODER_MIN_T0_S:
            model_status = f"warm-up (need {_DECODER_MIN_T0_S:.0f}s history)"
        elif wall_t > _DECODER_MAX_T0_S:
            model_status = "post-dataset (hold last)"
        else:
            model_status = "live decoder forward"
        sup.set_text(
            f"frame {i + 1}/{t_out}  t={wall_t:.0f}s  "
            f"|  model: {model_status}  "
            f"|  z={z_layer} (z={0.25 + CELL_SIZE_M * z_layer:.2f} m)"
        )
        return [m_img, t_img, sup]

    ani = FuncAnimation(
        fig, update, frames=t_out,
        interval=1000 / max(fps, 1), blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    fig.tight_layout()
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="s_029")
    ap.add_argument("--decoder-ckpt", type=Path, default=_DECODER_CKPT_DEFAULT)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--z-layer", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=Path("figures/decoder_pred"))
    args = ap.parse_args()

    out_path = args.out_dir / f"{args.scenario}__full.gif"
    _render(
        scenario=args.scenario,
        decoder_ckpt=args.decoder_ckpt,
        fps=int(args.fps),
        dpi=int(args.dpi),
        z_layer=int(args.z_layer),
        out_path=out_path,
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
