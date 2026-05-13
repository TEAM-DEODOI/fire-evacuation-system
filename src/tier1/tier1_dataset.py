"""Tier 1 GNN Dataset — binary detector sequences → future node danger.

Loads pre-built ``.npz`` files from ``results/detector_sequences/`` (each
representing one scenario) and exposes sliding windows for training.

Per-scenario file structure (from ``scripts/build_detector_sequences.py``):
    * ``binary_sequence``: (31, 27) latched binary detection sequence
    * ``node_danger``:     (31, 27) FDS truth danger at each sensor
    * ``activation_times``: (27,) first-trigger time (s) per sensor
    * ``trigger_reasons``:  list of str

Sliding window:
    * Input  window length: ``T_in``  (default 6 = 60 s history)
    * Output window length: ``T_out`` (default 6 = 60 s lookahead)
    * Pairs:  (binary[t : t+T_in], node_danger[t+T_in : t+T_in+T_out])
    * Total per scenario: 31 - T_in - T_out + 1 = 20 (with default T=6)

Per-node feature (5-dim):
    * is_detected:       binary[t, n]
    * det_time_norm:     activation_time / 300  (clamped to [0, 1])
    * type_onehot × 3:   room / corridor / exit
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_gnn import build_node_type_onehot

N_NODES = len(ALL_DETECTORS)
T_END_SECONDS = 300.0
N_TIMESTEPS = 31


class Tier1FireDataset(Dataset):
    """Sliding-window dataset over multiple detector-sequence files.

    Args:
        sequence_dir: directory containing ``<scenario>.npz`` files.
        scenario_names: list of scenario names to include (without ``.npz``).
        T_in: input window length.
        T_out: output (target) window length.

    __getitem__ returns:
        x: ``(N, T_in, F=5)`` float32 — per-node feature sequence
        y: ``(N, T_out)`` float32 — target danger sequence
    """

    def __init__(
        self,
        sequence_dir: Path,
        scenario_names: List[str],
        T_in: int = 6,
        T_out: int = 6,
    ) -> None:
        self.sequence_dir = Path(sequence_dir)
        self.T_in = T_in
        self.T_out = T_out
        self.node_type_onehot = build_node_type_onehot().numpy()  # (N, 3)

        # Validate + cache all scenarios in memory (each is tiny)
        self.scenarios: List[dict] = []
        for name in scenario_names:
            p = self.sequence_dir / f"{name}.npz"
            if not p.exists():
                raise FileNotFoundError(f"missing: {p}")
            data = np.load(p, allow_pickle=False)
            binary = data["binary_sequence"]    # (31, 27)
            danger = data["node_danger"]         # (31, 27)
            act_times = data["activation_times"] # (27,)
            assert binary.shape == (N_TIMESTEPS, N_NODES), binary.shape
            assert danger.shape == (N_TIMESTEPS, N_NODES), danger.shape
            self.scenarios.append({
                "name": name,
                "binary": binary.astype(np.float32),
                "danger": danger.astype(np.float32),
                "act_times": act_times.astype(np.float32),
            })

        # Build (scen_idx, t_start) pairs
        self.pairs: List[Tuple[int, int]] = []
        n_pairs_per = N_TIMESTEPS - T_in - T_out + 1
        if n_pairs_per <= 0:
            raise ValueError(f"T_in+T_out={T_in+T_out} > {N_TIMESTEPS}")
        for s_idx in range(len(self.scenarios)):
            for t in range(n_pairs_per):
                self.pairs.append((s_idx, t))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_idx, t = self.pairs[idx]
        scen = self.scenarios[s_idx]
        binary = scen["binary"][t : t + self.T_in]    # (T_in, N)
        danger = scen["danger"][
            t + self.T_in : t + self.T_in + self.T_out
        ]                                              # (T_out, N)
        act_times = scen["act_times"]                  # (N,)
        # det_time_norm at each timestep: clamp((t_now - activation) / T_END, 0, 1)
        # Actually simpler: detection happened iff binary==1; encode activation_time_norm
        # as a per-node constant over the input window (lookup), masked by current binary.
        # For simplicity: include normalized activation time only when activated.
        det_time_norm = np.where(
            act_times >= 0,
            np.clip(act_times / T_END_SECONDS, 0.0, 1.0),
            0.0,
        ).astype(np.float32)  # (N,)
        # Build per-(t, n) feature: [is_det, det_time_norm, type_onehot × 3]
        x = np.zeros((N_NODES, self.T_in, 5), dtype=np.float32)
        for t_idx in range(self.T_in):
            x[:, t_idx, 0] = binary[t_idx]           # is_detected
            x[:, t_idx, 1] = det_time_norm * binary[t_idx]  # masked
            x[:, t_idx, 2:5] = self.node_type_onehot
        # Target: (N, T_out)  — transpose from (T_out, N)
        y = danger.T.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def default_splits() -> Tuple[List[str], List[str], List[str]]:
    """Default train / val / test splits.

    * Train: 33 training scenarios (s_000 ~ s_032)
    * Val:   3 of the 13 OOD scenarios (held out for early stopping)
    * Test:  remaining 10 OOD scenarios

    Returns:
        (train_names, val_names, test_names)
    """
    train = [f"s_{i:03d}" for i in range(33)]
    ood_all = [
        "sim_1000kw_1m2_T01", "sim_1000kw_1m2_T03", "sim_1000kw_2m2_T01",
        "sim_1000kw_2m2_T05", "sim_1500kw_1m2_T02", "sim_1500kw_1m2_T03",
        "sim_1500kw_2m2_T05", "sim_500kw_1m2_T01", "sim_500kw_1m2_T02",
        "sim_500kw_1m2_T03", "sim_500kw_1m2_T04", "sim_500kw_2m2_T02",
        "sim_500kw_2m2_T05",
    ]
    # First 3 as val (mixed HRR / location coverage)
    val = ood_all[:3]
    test = ood_all[3:]
    return train, val, test


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=" * 70)
    print("tier1_dataset.py self-test")
    print("=" * 70)

    seq_dir = Path("results/detector_sequences")
    if not seq_dir.exists():
        print(f"FAIL: {seq_dir} not found. Run scripts/build_detector_sequences.py first.")
        sys.exit(1)

    train, val, test = default_splits()
    print(f"\n[setup] train={len(train)}, val={len(val)}, test={len(test)}")

    ds = Tier1FireDataset(seq_dir, train, T_in=6, T_out=6)
    print(f"[train] {len(ds.scenarios)} scenarios, {len(ds)} pairs total")

    # Sample one
    x, y = ds[0]
    print(f"  x shape: {tuple(x.shape)}, dtype: {x.dtype}")
    print(f"  y shape: {tuple(y.shape)}, dtype: {y.dtype}")
    print(f"  x range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    print(f"  y range: [{y.min().item():.3f}, {y.max().item():.3f}]")

    errors = []
    if x.shape != (N_NODES, 6, 5):
        errors.append(f"x shape: {x.shape}")
    if y.shape != (N_NODES, 6):
        errors.append(f"y shape: {y.shape}")

    # DataLoader smoke
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    bx, by = next(iter(dl))
    print(f"\n[loader] batch x: {tuple(bx.shape)}, y: {tuple(by.shape)}")
    if bx.shape != (4, N_NODES, 6, 5):
        errors.append(f"batch x: {bx.shape}")

    print("\n" + "=" * 70)
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("PASS: tier1_dataset")
