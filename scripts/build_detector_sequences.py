"""모든 시나리오 → 27 sensor binary_sequence (Tier 1 GNN 학습 입력).

각 raw 시나리오에서:
1. ``extract_slices()`` 로 SLCF 추출
2. D-023 트리거 모델 적용 (열 60°C OR 연기 10m, latched)
3. ``build_binary_sequence()`` → ``(31, 27)`` float32
4. ``compute_total_danger()`` 로 node-level truth danger ((31, 27)) 도 같이 저장

저장 형식: ``results/detector_sequences/<scenario>.npz``
  - ``binary_sequence``: ``(31, 27)`` float32, sensor binary (latched)
  - ``node_danger``: ``(31, 27)`` float32, FDS truth danger at each sensor (target)
  - ``activation_times``: ``(27,)`` float32, first-trigger time (s) or -1 if never
  - ``trigger_reasons``: list of str (heat/smoke/both/none)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.data_pipeline.fds_extractor import extract_slices
from src.risk_map.tenability import compute_total_danger
from src.shared.constants import GRID_SHAPE
from src.shared.coordinates import world_to_grid
from src.tier1.detector_model import (
    build_binary_sequence, detector_stats, extract_detector_events,
)
from src.tier1.detector_positions import (
    ALL_DETECTORS, get_detector_positions_legacy_format,
)


def node_danger_at_sensors(
    truth_danger: np.ndarray,    # (T, X, Y, Z)
    sensor_positions: list,
) -> np.ndarray:
    """각 sensor 위치의 danger 시계열 추출.

    Returns:
        ``(T, n_sensors)`` float32.
    """
    n_t = truth_danger.shape[0]
    n_s = len(sensor_positions)
    out = np.zeros((n_t, n_s), dtype=np.float32)
    for s_idx, pos in enumerate(sensor_positions):
        idx = world_to_grid(np.asarray(pos))
        ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
        if (0 <= ix < truth_danger.shape[1] and 0 <= iy < truth_danger.shape[2]
                and 0 <= iz < truth_danger.shape[3]):
            out[:, s_idx] = truth_danger[:, ix, iy, iz]
        # else: stays 0 (OOB sensor)
    return out


def process_scenario(scen_dir: Path, out_dir: Path) -> dict:
    """1 시나리오 처리. 통계 dict 반환."""
    name = scen_dir.name
    slices = extract_slices(scen_dir)

    # D-023 트리거 적용
    d_pos_legacy = get_detector_positions_legacy_format()
    sensor_positions = [pos for _, pos in d_pos_legacy]
    events = extract_detector_events(
        slices["temperature"], slices["visibility"], d_pos_legacy,
    )
    binary = build_binary_sequence(events)  # (31, 27)
    stats = detector_stats(events)

    # Node-level truth danger
    truth_d = compute_total_danger(
        slices["temperature"], slices["visibility"], slices["co"],
    ).astype(np.float32)
    node_d = node_danger_at_sensors(truth_d, sensor_positions)  # (31, 27)

    activation_times = np.array(
        [e.activation_time_s if e.activation_time_s is not None else -1.0
         for e in events],
        dtype=np.float32,
    )
    trigger_reasons = np.array([e.trigger_reason for e in events])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{name}.npz",
        binary_sequence=binary,
        node_danger=node_d,
        activation_times=activation_times,
        trigger_reasons=trigger_reasons,
        detector_ids=np.array([d.detector_id for d in ALL_DETECTORS]),
    )
    return {"name": name, **stats}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/detector_sequences"))
    parser.add_argument("--pattern", type=str, default="*",
                        help='glob pattern under raw-root (e.g. "s_*", "sim_*_T*")')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scens = []
    for p in sorted(args.raw_root.glob(args.pattern)):
        if not p.is_dir():
            continue
        if p.name == "first_sim":
            continue
        scens.append(p)
    print(f"[setup] {len(scens)} scenarios discovered")

    all_stats = []
    for s in scens:
        try:
            stats = process_scenario(s, args.output)
            all_stats.append(stats)
            print(f"  {s.name:30s} triggered={stats['n_triggered']:2d}/27  "
                  f"heat={stats['n_heat_only']:2d} smoke={stats['n_smoke_only']:2d} "
                  f"both={stats['n_both']:1d} never={stats['n_never']:2d}  "
                  f"mean={stats.get('mean_activation_time_s', 0):.0f}s")
        except Exception as e:
            print(f"  {s.name}: FAIL — {e}")

    print(f"\n[PASS] {len(all_stats)} sequences saved to {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
