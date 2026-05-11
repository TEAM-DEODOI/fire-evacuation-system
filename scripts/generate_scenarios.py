"""
Generate the 30 FDS scenario decks for the project from a Jinja2 template.

Layout produced under ``output_dir`` (defaults to ``data/raw/``)::

    output_dir/
    ├── s_000/s_000.fds                ← train (24 scenarios)
    ├── s_001/s_001.fds
    ├── ...
    ├── s_val_0/s_val_0.fds            ← val (3)
    ├── s_val_1/s_val_1.fds
    ├── s_val_2/s_val_2.fds
    ├── s_ood_0/s_ood_0.fds            ← OOD (3)
    ├── s_ood_1/s_ood_1.fds
    ├── s_ood_2/s_ood_2.fds
    └── scenario_config.json            ← full metadata

Fire locations and HRR levels come from the Day-2 prompt (locked in
``FIRE_LOCATIONS_TRAIN``, ``HRR_TRAIN``, ``SCENARIOS_VAL``,
``SCENARIOS_OOD`` below) and must match the PyroSim layout. Do NOT
introduce new positions or HRR levels here — adjust the prompt and
``docs/decisions.md`` first.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Template

from src.shared.constants import DOMAIN_SIZE_M

# ─── Locked scenario inputs (Day 2 prompt) ─────────────────────────────────
# (id_short, x_m, y_m, zone_label)
FIRE_LOCATIONS_TRAIN: List[tuple[str, float, float, str]] = [
    ("F1", 3.0, 12.0, "Zone_A_좌측"),
    ("F2", 8.0, 14.0, "Zone_A_중앙"),
    ("F3", 16.0, 16.0, "Zone_C_좌측"),
    ("F4", 20.0, 9.0, "Zone_C_D_경계"),
    ("F5", 4.0, 2.0, "Zone_B_좌측"),
    ("F6", 15.0, 2.0, "Zone_B_중앙"),
]

HRR_TRAIN: List[int] = [500, 1000, 1500, 2000]  # kW (D-012)

SCENARIOS_VAL: List[Dict[str, Any]] = [
    {"id": "s_val_0", "loc_id": "F1", "x": 3.0, "y": 12.0, "hrr": 800},
    {"id": "s_val_1", "loc_id": "F4", "x": 20.0, "y": 9.0, "hrr": 1200},
    {"id": "s_val_2", "loc_id": "F6", "x": 15.0, "y": 2.0, "hrr": 1700},
]

SCENARIOS_OOD: List[Dict[str, Any]] = [
    {
        "id": "s_ood_0",
        "loc_id": "O1",
        "x": 24.0,
        "y": 3.0,
        "hrr": 1000,
        "intent": "Zone D 미학습 영역",
    },
    {
        "id": "s_ood_1",
        "loc_id": "O2",
        "x": 15.0,
        "y": 9.0,
        "hrr": 1000,
        "intent": "중앙 홀 정중앙 (발표 헤드라인)",
    },
    {
        "id": "s_ood_2",
        "loc_id": "O3",
        "x": 28.0,
        "y": 16.0,
        "hrr": 1500,
        "intent": "외삽 검증",
    },
]

# Fire source is a 1 m × 1 m × 1 m OBST → top face area is 1 m², so
# HRRPUA (kW/m²) is numerically equal to the scenario HRR in kW.
_FIRE_HALF_M: float = 0.5

# Project root used to resolve default paths when this script is run via
# ``python scripts/generate_scenarios.py``.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_TEMPLATE: Path = _PROJECT_ROOT / "fds_templates" / "scenario_template.fds.j2"


# ─── Scenario assembly ────────────────────────────────────────────────────
def build_scenarios() -> List[Dict[str, Any]]:
    """Build the canonical 30-scenario metadata list.

    Returns:
        List of scenario dicts ordered ``train (24) → val (3) → ood (3)``.
        Each dict has keys: ``id``, ``loc_id``, ``fire_x``, ``fire_y``,
        ``hrr_kw``, ``split``, ``zone``. OOD entries additionally carry
        ``intent``.
    """
    scenarios: List[Dict[str, Any]] = []

    # ── train: 6 locations × 4 HRR levels = 24 ─────────────────────────────
    for loc_idx, (loc_id, x, y, zone) in enumerate(FIRE_LOCATIONS_TRAIN):
        for hrr_idx, hrr in enumerate(HRR_TRAIN):
            sid_num = loc_idx * len(HRR_TRAIN) + hrr_idx
            scenarios.append(
                {
                    "id": f"s_{sid_num:03d}",
                    "loc_id": loc_id,
                    "fire_x": float(x),
                    "fire_y": float(y),
                    "hrr_kw": int(hrr),
                    "split": "train",
                    "zone": zone,
                }
            )

    # ── val: 3 hand-picked HRR-interpolation scenarios ─────────────────────
    for entry in SCENARIOS_VAL:
        loc = next(loc for loc in FIRE_LOCATIONS_TRAIN if loc[0] == entry["loc_id"])
        scenarios.append(
            {
                "id": entry["id"],
                "loc_id": entry["loc_id"],
                "fire_x": float(entry["x"]),
                "fire_y": float(entry["y"]),
                "hrr_kw": int(entry["hrr"]),
                "split": "val",
                "zone": loc[3],
            }
        )

    # ── ood: 3 new locations ───────────────────────────────────────────────
    for entry in SCENARIOS_OOD:
        scenarios.append(
            {
                "id": entry["id"],
                "loc_id": entry["loc_id"],
                "fire_x": float(entry["x"]),
                "fire_y": float(entry["y"]),
                "hrr_kw": int(entry["hrr"]),
                "split": "ood",
                "zone": "OOD",
                "intent": entry["intent"],
            }
        )

    return scenarios


# ─── Template rendering ───────────────────────────────────────────────────
def render_fds(scenario: Dict[str, Any], template_path: Path) -> str:
    """Render one ``.fds`` file from a Jinja2 template.

    Args:
        scenario: A single entry from :func:`build_scenarios`.
        template_path: Path to ``scenario_template.fds.j2``.

    Returns:
        Rendered ``.fds`` content as a string.

    Raises:
        FileNotFoundError: If ``template_path`` does not exist.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"template not found: {template_path}")

    template = Template(template_path.read_text(encoding="utf-8"))

    fire_x = float(scenario["fire_x"])
    fire_y = float(scenario["fire_y"])

    context = {
        "scenario_id": scenario["id"],
        "hrrpua": int(scenario["hrr_kw"]),  # HRR (kW) == HRRPUA (kW/m²) for 1 m² top face
        "fire_x1": round(fire_x - _FIRE_HALF_M, 3),
        "fire_x2": round(fire_x + _FIRE_HALF_M, 3),
        "fire_y1": round(fire_y - _FIRE_HALF_M, 3),
        "fire_y2": round(fire_y + _FIRE_HALF_M, 3),
    }
    return template.render(**context)


def generate_all(template_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Write all 30 ``.fds`` files plus ``scenario_config.json`` under ``output_dir``.

    Args:
        template_path: Jinja2 template path.
        output_dir: Destination root (e.g. ``data/raw/``). Created if absent.

    Returns:
        The scenarios list (same as :func:`build_scenarios`) for callers that
        want to chain follow-up work.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios()
    for scenario in scenarios:
        scenario_dir = output_dir / scenario["id"]
        scenario_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_fds(scenario, template_path)
        (scenario_dir / f"{scenario['id']}.fds").write_text(rendered, encoding="utf-8")

    config = {"version": 1, "total": len(scenarios), "scenarios": scenarios}
    (output_dir / "scenario_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scenarios


# ─── Self-test ─────────────────────────────────────────────────────────────
def _run_self_test(template_path: Path) -> int:
    """Built-in test that exercises the full generation pipeline."""
    print("=" * 60)
    print("generate_scenarios.py self-test")
    print("=" * 60)

    errors: List[str] = []

    # ── 1. Count + split distribution ──────────────────────────────────────
    print("\n[1] build_scenarios() count + split distribution")
    scenarios = build_scenarios()
    print(f"  total = {len(scenarios)}")
    if len(scenarios) != 30:
        errors.append(f"expected 30 scenarios, got {len(scenarios)}")
    split_counts = {"train": 0, "val": 0, "ood": 0}
    for s in scenarios:
        split_counts[s["split"]] = split_counts.get(s["split"], 0) + 1
    print(f"  split = {split_counts}")
    if split_counts != {"train": 24, "val": 3, "ood": 3}:
        errors.append(f"split distribution wrong: {split_counts}")

    # ── 2. Fire positions inside SLCF [0, 30] × [0, 20] ────────────────────
    print("\n[2] All fire positions inside SLCF region")
    lx, ly, _ = DOMAIN_SIZE_M
    for s in scenarios:
        if not (0.0 <= s["fire_x"] <= lx and 0.0 <= s["fire_y"] <= ly):
            errors.append(
                f"{s['id']}: fire ({s['fire_x']}, {s['fire_y']}) outside SLCF"
            )

    # ── 3. HRR values come from the allowed set ────────────────────────────
    print("\n[3] HRR set restricted to project-defined values")
    allowed_hrr = set(HRR_TRAIN) | {e["hrr"] for e in SCENARIOS_VAL} | {e["hrr"] for e in SCENARIOS_OOD}
    print(f"  allowed = {sorted(allowed_hrr)}")
    for s in scenarios:
        if s["hrr_kw"] not in allowed_hrr:
            errors.append(f"{s['id']}: HRR {s['hrr_kw']} not in allowed set")

    # ── 4. Train locations are mutually separated ≥ 3 m ────────────────────
    print("\n[4] Training fire locations ≥ 3 m apart pairwise")
    locs = [(x, y) for _id, x, y, _z in FIRE_LOCATIONS_TRAIN]
    for i in range(len(locs)):
        for j in range(i + 1, len(locs)):
            d = math.hypot(locs[i][0] - locs[j][0], locs[i][1] - locs[j][1])
            if d < 3.0:
                errors.append(
                    f"train F{i + 1}↔F{j + 1} distance {d:.2f} m < 3 m"
                )

    # ── 5. OOD locations ≥ 3 m from every training location ────────────────
    print("\n[5] OOD locations ≥ 3 m from every training location")
    for e in SCENARIOS_OOD:
        for tx, ty in locs:
            d = math.hypot(e["x"] - tx, e["y"] - ty)
            if d < 3.0:
                errors.append(
                    f"OOD {e['id']} too close to train ({tx},{ty}): {d:.2f} m"
                )

    # ── 6. generate_all() into a temporary directory ───────────────────────
    print("\n[6] generate_all() into temp directory")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        out_scenarios = generate_all(template_path, tmp_dir)
        fds_files = sorted(tmp_dir.rglob("*.fds"))
        cfg_file = tmp_dir / "scenario_config.json"
        print(f"  .fds files written = {len(fds_files)}")
        print(f"  scenario_config.json = {cfg_file.exists()}")
        if len(fds_files) != 30:
            errors.append(f"expected 30 .fds files, got {len(fds_files)}")
        if not cfg_file.exists():
            errors.append("scenario_config.json missing")
        else:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            if cfg.get("total") != 30 or len(cfg.get("scenarios", [])) != 30:
                errors.append("scenario_config.json totals wrong")
            if cfg.get("version") != 1:
                errors.append("scenario_config.json version wrong")

        # ── 7. Spot-check one rendered .fds for correct substitution ───────
        print("\n[7] Spot-check rendered .fds substitution")
        sample = out_scenarios[0]              # s_000 (F1, 500 kW)
        sample_fds = (tmp_dir / sample["id"] / f"{sample['id']}.fds").read_text(encoding="utf-8")
        for needle, label in [
            (f"CHID='{sample['id']}'", "CHID"),
            (f"HRRPUA={sample['hrr_kw']}", "HRRPUA"),
            ("XB=0.0,30.0, 0.0,20.0, 0.0,3.0", "SLCF XB Z=3.0"),
        ]:
            if needle not in sample_fds:
                errors.append(f"{sample['id']}.fds missing '{label}': {needle}")
        # Fire OBST line must have the correct half-cell coordinates.
        fx1 = round(sample["fire_x"] - _FIRE_HALF_M, 3)
        fy1 = round(sample["fire_y"] - _FIRE_HALF_M, 3)
        if f"XB={fx1}" not in sample_fds:
            errors.append(f"{sample['id']}.fds OBST x1={fx1} not found")
        if f",{fy1}," not in sample_fds and f" {fy1}," not in sample_fds:
            errors.append(f"{sample['id']}.fds OBST y1={fy1} not found")
        # VECTOR=.TRUE. must not appear on any non-comment line (L-001).
        for ln in sample_fds.splitlines():
            stripped = ln.lstrip()
            if stripped.startswith("!"):
                continue  # FDS comment — informational only
            if "VECTOR=.TRUE." in stripped.upper().replace(" ", ""):
                errors.append(
                    f"{sample['id']}.fds has VECTOR=.TRUE. on a non-comment line: {ln!r}"
                )
                break

    # ── Verdict ────────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nPASS: 30 scenarios generated and validated")
    return 0


# ─── CLI entry ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--template",
        type=Path,
        default=_DEFAULT_TEMPLATE,
        help="Path to the Jinja2 template (default: fds_templates/scenario_template.fds.j2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Output directory under which scenario subfolders are created.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in test with a temporary output directory.",
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(_run_self_test(args.template))

    scenarios = generate_all(args.template, args.output)
    print(f"Generated {len(scenarios)} scenarios under {args.output}")
    print(f"  metadata → {args.output / 'scenario_config.json'}")


if __name__ == "__main__":
    main()
