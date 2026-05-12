"""One-shot migration: original PyroSim dir names → canonical s_000 etc.

Builds the scenario_config.json + renames data/raw subdirectories
following Option A from the design discussion (D-023):

* train (21) — _001…_007 × {500, 1000, 1500} kW
* val   ( 6) — _008, _009 × {500, 1000, 1500} kW
* ood   ( 6) — H01–H03 × {500, 1000} kW (no 1500 kW H variants)

Original directory names are recorded as ``original_id`` in
``scenario_config.json`` so the rename is reversible.

Idempotent: if a canonical-named directory already exists for a scenario
slot, that slot is skipped (assumed already migrated).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ─── Fire location table (extracted from .fds files; (x_centre, y_centre)) ──
# Index ↔ canonical location ID (loc_id). Same coordinates for all HRRs.
_LOC_TABLE: Dict[str, Tuple[float, float, str]] = {
    "001": (11.5, 18.0, "Zone_C_좌측"),
    "002": (19.0, 18.0, "Zone_C_중앙"),
    "003": (21.5, 15.0, "Zone_C_D_경계"),
    "004": (18.5, 12.5, "중앙홀_북측"),
    "005": (18.5, 9.5,  "중앙홀_헤드라인"),
    "006": (15.5, 2.5,  "Zone_B_중앙"),
    "007": (2.5,  2.5,  "Zone_B_좌측"),
    "008": (1.0,  9.5,  "서측_출구"),
    "009": (6.0,  15.5, "Zone_A_중앙"),
    "H01": (18.5, 15.0, "OOD_중앙홀_동측"),
    "H02": (6.0,  5.0,  "OOD_Zone_B_좌측"),
    "H03": (8.0,  14.5, "OOD_Zone_A_좌측"),
}

# (HRR_kw, suffix) tuples in canonical order
_HRRS_KW: Tuple[int, ...] = (500, 1000, 1500)


def _build_mapping() -> List[Dict[str, Any]]:
    """Return the 33-scenario list with canonical IDs + metadata + original_id."""
    scenarios: List[Dict[str, Any]] = []
    train_locs = [f"{i:03d}" for i in range(1, 8)]   # 001..007 (7)
    val_locs = [f"{i:03d}" for i in range(8, 10)]    # 008, 009 (2)
    ood_locs = ["H01", "H02", "H03"]                 # H01..H03

    # train (21 = 7 locs × 3 HRRs)
    sid = 0
    for hrr in _HRRS_KW:
        for loc in train_locs:
            x, y, zone = _LOC_TABLE[loc]
            scenarios.append({
                "id": f"s_{sid:03d}",
                "original_id": f"sim_{hrr}kw_1m2_{loc}",
                "loc_id": loc,
                "fire_x": x,
                "fire_y": y,
                "hrr_kw": hrr,
                "split": "train",
                "zone": zone,
            })
            sid += 1

    # val (6 = 2 held-out locs × 3 HRRs)
    val_idx = 0
    for hrr in _HRRS_KW:
        for loc in val_locs:
            x, y, zone = _LOC_TABLE[loc]
            scenarios.append({
                "id": f"s_val_{val_idx}",
                "original_id": f"sim_{hrr}kw_1m2_{loc}",
                "loc_id": loc,
                "fire_x": x,
                "fire_y": y,
                "hrr_kw": hrr,
                "split": "val",
                "zone": zone,
            })
            val_idx += 1

    # ood (6 = 3 H locs × {500, 1000} kW — no 1500kW H)
    ood_idx = 0
    for hrr in (500, 1000):
        for loc in ood_locs:
            x, y, zone = _LOC_TABLE[loc]
            scenarios.append({
                "id": f"s_ood_{ood_idx}",
                "original_id": f"sim_{hrr}kw_1m2_{loc}",
                "loc_id": loc,
                "fire_x": x,
                "fire_y": y,
                "hrr_kw": hrr,
                "split": "ood",
                "zone": zone,
                "intent": f"H location {loc} at {hrr}kW — OOD generalization",
            })
            ood_idx += 1

    return scenarios


def migrate(raw_dir: Path) -> None:
    scenarios = _build_mapping()
    print(f"Migration plan: {len(scenarios)} scenarios")
    n_train = sum(1 for s in scenarios if s["split"] == "train")
    n_val   = sum(1 for s in scenarios if s["split"] == "val")
    n_ood   = sum(1 for s in scenarios if s["split"] == "ood")
    print(f"  splits: train={n_train}  val={n_val}  ood={n_ood}")

    # 1) Physical rename
    renamed = 0
    skipped_already = 0
    missing = 0
    for s in scenarios:
        src = raw_dir / s["original_id"]
        dst = raw_dir / s["id"]
        if dst.is_dir():
            skipped_already += 1
            continue
        if not src.is_dir():
            print(f"  MISSING: {src.name} (would map to {s['id']})")
            missing += 1
            continue
        src.rename(dst)
        renamed += 1
    print(
        f"  rename: {renamed} done, {skipped_already} already, "
        f"{missing} missing"
    )

    if missing:
        print("\nFAIL: some source directories missing — aborting before JSON write")
        sys.exit(1)

    # 2) Write scenario_config.json
    config_path = raw_dir / "scenario_config.json"
    config = {
        "version": 1,
        "total": len(scenarios),
        "scenarios": scenarios,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  wrote {config_path}")


if __name__ == "__main__":
    raw = Path("data/raw")
    if not raw.is_dir():
        print(f"ERROR: {raw} not found")
        sys.exit(1)
    migrate(raw)
    print("\nDONE")
