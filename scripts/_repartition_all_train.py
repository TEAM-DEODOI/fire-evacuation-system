"""One-shot: 33 시나리오 전체를 train 으로 재배치.

D-023 의 Option A (train 21 / val 6 / ood 6) 를 폐기하고, 사용자가 결정한
**Option A' (train 33 / val 0 / ood 0)** 로 전환. val/ood 슬롯은 추후 별도
시뮬레이션이 추가될 때 사용.

작업:
1. ``s_val_*`` (6개), ``s_ood_*`` (6개) → ``s_021``…``s_032`` 로 rename.
2. ``scenario_config.json`` 재작성 — 모든 33 항목의 split 을 ``"train"`` 으로.
   ``original_id`` 와 fire 좌표 / HRR / zone 정보는 그대로 보존.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    raw = Path("data/raw")
    cfg_path = raw / "scenario_config.json"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found")
        return 1

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    scenarios = cfg["scenarios"]

    # ── 1) physical rename: s_val_* / s_ood_* → s_021…s_032 ────────────────
    train_max = max(
        (int(s["id"].split("_")[1]) for s in scenarios if s["split"] == "train"),
        default=-1,
    )
    next_idx = train_max + 1  # 21
    rename_map: dict[str, str] = {}
    for s in scenarios:
        if s["split"] == "train":
            continue
        new_id = f"s_{next_idx:03d}"
        rename_map[s["id"]] = new_id
        next_idx += 1

    print(f"Rename plan ({len(rename_map)} dirs):")
    for old, new in rename_map.items():
        print(f"  {old:10s} → {new}")

    renamed = 0
    for old, new in rename_map.items():
        src = raw / old
        dst = raw / new
        if dst.is_dir():
            print(f"  SKIP: {new} already exists")
            continue
        if not src.is_dir():
            print(f"  MISSING: {src}")
            continue
        src.rename(dst)
        renamed += 1
    print(f"  → {renamed} renamed")

    # ── 2) Update scenario_config.json ─────────────────────────────────────
    new_scenarios = []
    for s in scenarios:
        new_id = rename_map.get(s["id"], s["id"])
        s2 = dict(s)
        s2["id"] = new_id
        s2["split"] = "train"
        # Drop OOD-only fields if present
        s2.pop("intent", None)
        new_scenarios.append(s2)

    # Sort by canonical s_NNN order for readability
    new_scenarios.sort(key=lambda s: int(s["id"].split("_")[1]))

    new_cfg = {
        "version": 1,
        "total": len(new_scenarios),
        "scenarios": new_scenarios,
    }
    cfg_path.write_text(
        json.dumps(new_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  rewrote {cfg_path}")

    # Summary
    splits: dict[str, int] = {}
    for s in new_scenarios:
        splits[s["split"]] = splits.get(s["split"], 0) + 1
    print(f"\nsplits now: {splits}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
