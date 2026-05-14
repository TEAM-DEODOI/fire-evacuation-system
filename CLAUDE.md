# Fire Evacuation Prediction System — Project Context

> **Read this file first.** It is the single source of truth for project
> constraints, conventions, scope, and team workflow. Never violate the
> hard constraints below.
>
> When you need detail, refer to:
> - `docs/interface_contracts.md` — exact function signatures
> - `docs/coordinate_convention.md` — coordinate system rules
> - `docs/risk_indicators.md` — tenability thresholds + ISO/SFPE citations
> - `docs/manual_v2.md` — 14-week schedule and rationale
> - `docs/decisions.md` — log of past decisions and reasons
> - `docs/lessons_learned.md` — concrete bugs encountered, do not repeat
> - `docs/task_request_template.md` — standard format for task delegation
>
> Read these explicitly when relevant. Do **not** auto-load them every session.

---

## What This Project Is

14-week undergraduate engineering capstone competition entry: an
**active fire-response system**.

End-to-end pipeline:

```
FDS simulation data
    ↓
ConvLSTM and PI-FNO models predict fire spread
(Temperature, Visibility, CO over time)
    ↓
ISO-13571-based risk map conversion
(Tenability thresholds + cumulative FED)
    ↓
Weighted A* on building graph
(Static baseline vs Dynamic predictive)
    ↓
Path safety validation
(EXP-PATH-001: cumulative FED reduction)
    ↓
PyBullet integrated demo (Week 12)
(Drone swarm guides evacuees along planned paths — three scenarios compared)
```

---

## Six Core Hypotheses (Drives All Work)

Every line of code in this project exists to validate one of these hypotheses.

| ID | Hypothesis | Target | Experiment |
|----|-----------|--------|------------|
| H1 | **Speed**: PI-FNO inference is ≥1000× faster than FDS | <50ms vs FDS minutes | EXP-FIRE-001 |
| H2 | **Accuracy**: Relative L2 ≤ 15% on training scenarios | Rel L2 < 15% | EXP-FIRE-001 |
| H3 | **Generalization**: PI-FNO outperforms ConvLSTM on OOD scenarios | OOD Rel L2 lower | EXP-FIRE-001 |
| H4 | **Risk map quality**: Risk map FNR < 10% (no false safe predictions) | FNR < 10% | EXP-RISK-001 |
| H5 | **Risk map fidelity**: PI-FNO risk map IoU ≥ 0.7 vs FDS ground truth | IoU ≥ 0.7 @ 0.7 threshold | EXP-RISK-001 |
| H6 | **System effectiveness**: Dynamic drone-swarm guidance reduces cumulative FED ≥ 30% vs fixed sign baseline | FED reduction ≥ 30% | EXP-PATH-001 |

**Presentation emphasis order**: H1 → H6 → H4 → H2 → H5 → H3.

H6 is the strongest card: fixed-sign vs drone-swarm FED comparison is
intuitive, visually striking, and immediately resonates with fire safety
engineering judges. H4/H5 provide the scientific grounding that the risk
map driving H6 is trustworthy.

---

## Hard Constraints — NEVER Violate

### Geometry

| Parameter | Value |
|-----------|-------|
| Building | Single floor, complex maze layout (multiple rooms, intersections, central courtyard) |
| **Real STL building** | Up to 3.2 m height (preserved as-is in PyroSim) |
| **FDS MESH** (computational domain) | **100 × 80 × 8 cells** over **[−10, 40] × [−10, 30] × [0, 4] m** |
| **SLCF region** (learnable, model-visible) | **60 × 40 × 6 cells** over **[0, 30] × [0, 20] × [0, 3] m** |
| Cell resolution | **0.5 m × 0.5 m × 0.5 m** |
| External buffer | 10 m on −X, +X, −Y, +Y for ventilation boundaries |

**Critical distinction:** The MESH is what FDS simulates (with 10 m buffer).
The SLCF is what models ingest. The buffer exists purely for boundary
conditions and is invisible to ML.

**Real STL height 3.2 m, but SLCF extracts only 0~3 m.** This preserves
the physical building shape while keeping the learnable grid at 6 z-cells
(matching `GRID_SHAPE`). Top 0.2 m is the hottest smoke layer but does not
affect breathing-zone analysis (1.5 m). See decision D-015.

### Time

| Parameter | Value |
|-----------|-------|
| Simulation duration | 0–300 s |
| Time steps | **31 frames** at **10 s** intervals |
| FDS `DT_SLCF` | 10.0 |
| Model prediction step | 10 s (single autoregressive step) |
| Predictive horizon | 60 s (6 autoregressive steps) |

### Data

| Parameter | Value |
|-----------|-------|
| Total scenarios | **30** (24 train / 3 val / 3 OOD) |
| Fire HRR variations | 500, 1000, 1500, 2000 kW |
| Fire location variations | 6 distinct locations across the maze |
| Ventilation | All scenarios: both end doors open (fixed) |
| Single-scenario CPU time | ~23 minutes (validated empirically) |

### Compute

| Parameter | Value |
|-----------|-------|
| Training GPU | NVIDIA A100 40 GB on RunPod |
| Coordinate system | Metres, Z-up, world origin at corner (0, 0, 0) |
| Units | **SI only** — m, s, °C, ppm. NEVER mm. |

---

## Team Roles (Who Does What)

The project is a 4-person team with parallel workstreams.

| Member | Primary responsibility | Modules owned |
|--------|----------------------|---------------|
| **A** | Building modeling, FDS simulations, scenario design | PyroSim, FDS configs, scenario_config.json |
| **B** | Model training (ConvLSTM, PI-FNO) | `src/models/`, `src/training/` |
| **C** | Data pipeline, dataset, normalization | `src/shared/`, `src/data_pipeline/`, `src/dataset/` |
| **D** | Evaluation, integration, visualization | `src/evaluation/`, `src/integration/`, `src/visualization/`, paper |

Risk map and path planning (Weeks 10–11) are shared by B and C as the
team's joint workload.

**Claude Code is delegated to**: code implementation under all members'
guidance. Claude Code does NOT make scope decisions, hyperparameter
choices, or scientific judgment calls — those go to humans.

---

## Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| Deep learning | Python 3.10+, PyTorch 2.0+, CUDA 11.8+ |
| Neural operator | `neuraloperator` (FNO) |
| FDS data | `fdsreader` |
| Graph / routing | `NetworkX` |
| Interpolation | `scipy.interpolate.RegularGridInterpolator` |
| Drone sim (Wk 12) | `pybullet`, `gym-pybullet-drones` (drone swarm, multiple agents) |
| Experiment tracking | Weights & Biases |
| Testing | `pytest` |

---

## Tensor Conventions — CRITICAL

All model code must use these shapes:

```
Model input  : (B, 5, 60, 40, 6)   channels → [T, V, CO, mask, time_enc]
Model output : (B, 3, 60, 40, 6)   channels → [T, V, CO]
```

- All channels normalized to **[0, 1]**
- Convention: **higher value = more dangerous** (channels 0–2)
- Visibility is **INVERSE-mapped**: low visibility → high value
- Time encoding broadcast spatially (constant across grid for each frame)

Single-timestep usage: input is one frame, output is one frame at t+10s.
For 60s horizon, autoregress 6 times (chain output → input).

See `docs/interface_contracts.md` for exact normalisation formulas.

---

## RiskMap Interface (CRITICAL)

```python
from abc import ABC, abstractmethod
import numpy as np

class RiskMap(ABC):
    """Abstract risk map.

    Three concrete implementations live in src/risk_map/.
    Both data_pipeline (validation) and PyBullet integration use this same
    query() interface. Code that consumes risk maps does not need to know
    which implementation is in use.
    """

    @abstractmethod
    def query(
        self,
        xyz: np.ndarray,           # shape (3,) or (N, 3) in world metres
        t: float | None = None,    # simulation time in seconds
    ) -> float | np.ndarray:
        """Returns danger ∈ [0, 1].

        Out-of-bounds → 1.0  (safety default — drone going outside building
                              should be flagged dangerous).
        Out-of-time-range → 1.0
        """
```

Three concrete implementations:

| Class | Used by | Purpose |
|-------|---------|---------|
| `FDSRiskMap` | data_pipeline, validation | Ground truth from FDS data |
| `FNORiskMap` | evaluation | PI-FNO inference results |
| `DynamicRiskMap` | PyBullet demo (Week 12) | Time-evolving live map |

---

## Tenability Thresholds (ISO 13571 + SFPE Handbook)

| Indicator | Safe | Danger |
|-----------|------|--------|
| Temperature | 30 °C | 60 °C (humid air) |
| Visibility | 10 m | 3 m |
| CO instantaneous | 100 ppm | 1400 ppm |
| FED (CO cumulative) | — | 0.3 (sensitive population) |

Sources: ISO 13571:2012 §5–7; SFPE Handbook 5th Ed. Ch. 63 (Purser & McAllister 2016).
See `docs/risk_indicators.md` for full derivation and Korean glossary.

---

## What This Project Does NOT Do

Do **not** add these features, even if they seem natural extensions.
Each was explicitly excluded for scope or feasibility reasons (see
`docs/decisions.md` for rationales):

- Multi-floor buildings (single floor only)
- Real-time CFD (FDS pre-computed only)
- Mesh resolution other than 0.5 m
- Real fire experiments (simulation only)
- Ventilation variation (all scenarios: both doors open)
- HCN, irritant gas, or radiant heat FED (CO only)
- **Real human behaviour modelling** — PersonAgent uses a simplified movement
  model only: constant speed (1.2 m/s), wall collision avoidance via PyBullet
  contact, status transitions (alive → evacuated / dead). No panic speed
  increase, no social force, no crowd density effects.
- Replacing existing fire safety systems (we are auxiliary)

---

## FDS Input File Conventions — LESSONS LEARNED

These rules have been validated by trial and error. Violating them breaks
the data pipeline:

1. **NEVER set `VECTOR=.TRUE.` on `&SLCF`.**
   Combination of `VECTOR=.TRUE.` and `CELL_CENTERED=.TRUE.` causes
   `fdsreader` to fail with broadcast errors during slice loading.
   Use scalar slices only (3 slices: T, V, CO). See L-001.

2. **SLCF Z range MUST be exactly `0.0, 3.0`** (not 3.2 or 3.5).
   PyroSim may auto-set this to 3.5 to accommodate STL height 3.2 m.
   Manual fix required. See L-009 and D-015.

3. **Always use `CELL_CENTERED=.TRUE.`** for SLCF data we feed into models.

4. **`DT_SLCF=10.0`** to align with model time step.

5. **SLCF `XB` must extract** the learnable region [0, 30] × [0, 20] × [0, 3]
   even when the MESH is larger.

6. Three SLCF only: `TEMPERATURE`, `SOOT VISIBILITY` (or `VISIBILITY`),
   `VOLUME FRACTION` (with `SPEC_ID='CARBON MONOXIDE'`).

Example correct SLCF:
```
&SLCF QUANTITY='TEMPERATURE',
      CELL_CENTERED=.TRUE.,
      ID='Temperature',
      XB=0.0,30.0, 0.0,20.0, 0.0,3.0/
```

Wrong (causes fdsreader failure):
```
&SLCF QUANTITY='TEMPERATURE',
      VECTOR=.TRUE.,            ← REMOVE THIS LINE
      CELL_CENTERED=.TRUE.,
      ID='Temperature',
      XB=0.0,30.0, 0.0,20.0, 0.0,3.5/   ← MUST BE 3.0
```

---

## fdsreader Standard Pattern

This is the canonical way to load FDS data. Use verbatim in any module.

```python
import fdsreader
from pathlib import Path

sim = fdsreader.Simulation(str(Path(fds_dir)))

# Filter by quantity name (matches the SLCF QUANTITY in .fds file)
temp_slc = sim.slices.filter_by_quantity("TEMPERATURE")[0]
vis_slc  = sim.slices.filter_by_quantity("SOOT VISIBILITY")[0]
co_slc   = sim.slices.filter_by_quantity("CARBON MONOXIDE VOLUME FRACTION")[0]

# Returns (T, nx, ny, nz) array AND coordinate dict in world metres
grid, coords = temp_slc.to_global(return_coordinates=True)

# coords['x'] = [0.25, 0.75, ..., 29.75]   ← cell centres in world metres
# Because origin is aligned, these ARE world coordinates. No transformation.
```

For pre-conditions (CELL_CENTERED required, VECTOR forbidden, etc.) see
`docs/coordinate_convention.md`.

---

## Coding Conventions

### Required for all public code

- **Type hints** on every function signature
- **Docstrings** on every public function (Google or NumPy style)
- **Tensor shapes documented** in docstrings: `Args: x: shape (B, 5, 60, 40, 6)`
- **`pathlib.Path`** for file paths, never `os.path.join`
- **`raise ValueError("clear message")`** — never silent failures
- **`if __name__ == '__main__'`** self-test block in computational modules,
  printing PASS/FAIL clearly
- **Absolute imports**: `from src.shared.constants import GRID_SHAPE`
- **YAML** for config files (loaded via `dataclass` or Pydantic)
- **Korean comments OK**, but identifiers must be English

### Forbidden

- Black/Ruff configuration (skip — keep tooling minimal)
- `from foo import *` (always explicit)
- Configuration as raw dicts in production code (use dataclasses)
- Silent exception handling
- Hard-coded numerical constants (use `src/shared/constants.py`)

### Module structure

Every computational module must have:

1. Module docstring (purpose, author, related week in manual)
2. Imports
3. Type-hinted functions/classes with docstrings
4. `if __name__ == '__main__'` self-test that prints `PASS` or `FAIL`

---

## 14-Week Schedule Reference

| Week | Module | Owner |
|------|--------|-------|
| 1–2  | Environment setup, building modeling (PyroSim) | A |
| 3–5  | FDS scenario generation (30 runs) | A |
| 6    | Data pipeline: extraction, normalisation, masks | C |
| 7    | ConvLSTM baseline + over-fit test | B |
| 8    | PI-FNO no-PI version | B |
| 9    | PI-FNO full + EXP-FIRE-001 | B + D |
| 10   | Risk map module + EXP-RISK-001 | B + C |
| 11   | Path planning + ablations | B + C |
| 12   | EXP-PATH-001 + PyBullet integration | D + all |
| 13   | Visualisation, slides | D |
| 14   | Paper draft, code cleanup | all |

See `docs/manual_v2.md` for week-by-week details, deliverables, and
validation criteria.

---

## Three Critical Experiments

The project is structured around three named experiments. Each produces a
clear table for the final paper and presentation.

| Experiment | Compares | Output |
|------------|----------|--------|
| **EXP-FIRE-001** | ConvLSTM vs FNO no-PI vs FNO full | RMSE/SSIM table, OOD generalization |
| **EXP-RISK-001** | FDS ground truth vs PI-FNO predicted risk maps | IoU at 0.3/0.5/0.7, FNR, FPR |
| **EXP-PATH-001** | Three evacuation scenarios compared on cumulative FED, success rate, mean evacuation time | FED reduction, casualty rate |

### EXP-PATH-001 — Three Scenarios

All three scenarios share the same PyBullet building, the same 20 person
agents (simplified movement model — see NOT DO section), and the same
DynamicRiskMap interface.  The only differences are the guidance system and
the risk-map source.

| Scenario | Guidance system | RiskMap source | H6 role |
|----------|----------------|---------------|---------|
| **S1 — Fixed sign baseline** | Static guidance signs pre-placed in building; persons individually detect danger (query RiskMap) and navigate toward nearest non-hazardous sign | FDS | Baseline |
| **S2 — FDS drone swarm** | Drone swarm (gym-pybullet-drones, Boids/APF) detects isolated persons and guides them via dynamic waypoints computed by weighted A* | FDS | Experimental |
| **S3 — PI-FNO drone swarm** | Same drone swarm, same A* | PI-FNO | Experimental |

Comparing S1 vs S2 validates H6 (system effectiveness).
Comparing S2 vs S3 validates H5 (risk map fidelity translates to guidance quality).

**Metrics collected per scenario** (by `src/evaluation/metrics.py`):
- `evacuation_success_rate` — fraction evacuated within 300 s
- `mean_evacuation_time` — mean time to exit for evacuated persons (s)
- `danger_zone_exposure_time` — mean cumulative seconds in risk ≥ 0.5 zone
- `casualty_rate` — fraction with status `dead`
- `cumulative_FED` — mean cumulative FED per person (H6 primary metric)

Plus three **ablations** (Week 11):
- PI loss components contribution
- Training data size vs performance curve
- Model size variation (optional)

---

## Plan B — Failure Scenarios

Anticipate these. Each has a documented response in `docs/manual_v2.md`.

| Failure | Response |
|---------|----------|
| FDS scenarios don't finish in Week 5 | Reduce to 20 scenarios, T_END to 180 s |
| PI-FNO doesn't beat ConvLSTM | Refocus paper on "30-scenario regime trade-offs" |
| RunPod cost exceeds budget | Switch to Spot only, drop Ablation 3, drop PyBullet demo |
| Coordinate system bugs | Re-validate via `docs/coordinate_convention.md` checklist |
| Team member overload | Cut PyBullet demo, Ablation 3, Tier 1 GNN; preserve EXP-* |

---

## Workflow Rules

### Module development cycle

1. **Implement** with type hints and docstrings
2. **Self-test** via `__main__` block: `python -m src.module.path`
3. **Human review**: file diff and explanation
4. **Unit test** in `tests/` using pytest
5. **Commit** with clear message

### Decision-making

- **When in doubt about scope, ASK** before expanding
- **Single-purpose functions** — no "while we're at it" features
- **Reference checks before improvement**: confirm interface change is
  necessary before modifying contracts
- **Log major decisions** to `docs/decisions.md` after agreement
- **Log new bugs** to `docs/lessons_learned.md` after fixing

### Communicating with this project

- Read `CLAUDE.md` first (you are doing this now)
- Read explicitly referenced docs as needed
- Never auto-load all docs — preserve context budget
- Follow the 8-section task request template (see `docs/task_request_template.md`)

---

## Where Things Go

```
src/shared/        — coordinates, constants, building, normalisation
src/data_pipeline/ — FDS .smv/.sf → .npz conversion (Week 6)
src/dataset/       — PyTorch Dataset and DataLoaders (Week 6)
src/models/        — ConvLSTM, PI-FNO, base classes, losses (Week 7-9)
src/training/      — training loops, callbacks (Week 7-9)
src/evaluation/    — metrics, model comparison (Week 9-10)
src/risk_map/      — risk map conversion, ASET, FED, predictive (Week 10)
src/path_planning/ — graph, A*, evacuation simulator (Week 11)
src/integration/   — PyBullet demo: env_setup, person agents, drone swarm, scenarios, metrics (Week 12)
src/visualization/ — plots, animations (Week 13)
configs/           — YAML hyperparameters per module
experiments/       — executable scripts (exp_*, ablation_*)
notebooks/         — exploratory analysis only (not production code)
tests/             — pytest unit tests mirroring src/ structure
docs/              — project documentation (this is the heart)
checkpoints/       — trained models (.gitignore)
results/           — experiment outputs (CSV, JSON)
figures/           — paper/slide figures
```

---

## Key File Locations

| What | Where |
|------|-------|
| All numerical constants | `src/shared/constants.py` |
| Normalization functions | `src/shared/normalization.py` |
| Coordinate utilities | `src/shared/coordinates.py` |
| Building geometry | `src/shared/building.py` |
| RiskMap abstract base | `src/risk_map/risk_map_class.py` |
| Interface contracts | `docs/interface_contracts.md` |
| Coordinate conventions | `docs/coordinate_convention.md` |
| Tenability reference | `docs/risk_indicators.md` |
| 14-week manual | `docs/manual_v2.md` |
| Decision log | `docs/decisions.md` |
| Lessons learned | `docs/lessons_learned.md` |
| Task request template | `docs/task_request_template.md` |

---

## Current Project State (2026-05-14)

> **Resume here**: `docs/README.md` (index) → `docs/00_project_overview.md`
> → `docs/70_results_summary.md` (모든 결과) → `docs/90_next_steps.md`
> (다음 작업). 과거 handoff (`docs/archive/old_planning/handoff_2026_05_12.md`)
> 는 archive 로 이동.

**Completed**:
- Tier 0/1/2 (foundation, simulation tools, data extraction) — all ✅
- Data pipeline: 33+13-scenario dataset (33 train + 13 OOD from Member A's
  T01–T05 sims — val/ood blocker 해소)
- Risk Map module (tenability / FED / ASET / StaticRiskMap / converter /
  path_metrics) with D-022 4-metric H6 evaluation
- **ConvLSTM 학습 + 평가 완료**: RelL2 0.136 / IoU 0.887 / FNR 6.0% on 13 OOD
- **FNO no-PI + FNO full-PI 학습 완료 (RunPod A100)** — `checkpoints/fno_{no_pi,pi}/best.pt`
- **EXP-FIRE-001 3-model OOD 비교 완료** (`results/exp_fire_001/comparison.csv`)
  - ConvLSTM 0.136 / FNO no-PI 0.138 / FNO PI 0.157 (RelL2)
- **D-024 v3.3 — 39 sensor 인프라**: 22 rooms + 14 corridors + 3 exits
- **D-023 trigger model**: T>60°C OR V<10m, 46 시나리오 `binary_sequence` 생성
- **Tier 1 GNN (SimpleFireGNN, 12K params) 학습 + 평가**
  - **IoU 0.904 / FNR 4.6% on 13 OOD (13/13 H5 통과, 11/13 H4 통과)** ★
  - Tier 2 best (FNO no-PI + geodesic) 대비 IoU 2.1× / 150× smaller
- **L1–L4 evaluation layer framework** + mask-aware geodesic IDW interpolation
- **Sparse-input ConvLSTM (L4e)**: IoU 0.182 conservative bias →
  re-sparsify fix → IoU 0.581
- Documentation 재구성 — numbered `00_*`~`90_*` + `docs/archive/`
- **Path planning 모듈 (`src/path_planning/`) — 완료 (2026-05-14)**
  - `building_graph.py` thin adapter over `shared/building.py` 19-node graph
  - `edge_weights.py` weighted A* edges with N-sample integrated risk + impassable filter
  - `planners.py` single `EvacuationPlanner` (3-class ABC 폐기 per D-025)
  - `evacuation_sim.py` NumPy-only logical single-occupant trial
  - Self-test 4/4 PASS; integration smoke `scripts/smoke_tier1_pipeline.py` PASS
- **Tier1RiskMap 어댑터 검증 완료** — `src/tier1/tier1_risk_map.py` 9/9 self-test PASS,
  헤드라인 체크포인트와 shape 호환 확인
- **PyBullet integration `src/integration/` — D-025 9-module skeleton landed**
  - Functional: `scene.py` (placeholder URDF loads, 8/8 PASS), `urdf_builder.py`
    (`build_placeholder_urdf` 9-box L-shape, 6/6 PASS), `person_agent.py`
    (M2-mini: spawn + kinematic walk + `getClosestPoints` wall veto, 7/7 PASS),
    `metrics.py` (5-metric dataclass + CSV + `h6_verdict`), all 3 scenarios
    (`s1_fixed_sign` M3-α, `s2_fds_swarm` M4-mini, `s3_fno_swarm` M5-mini)
  - Skeleton: `drone_swarm.py` (Crazyflie + Boids/APF: M4-full), full STL→URDF
    in `urdf_builder.build_building_urdf` (trimesh 미설치: M1-full)
- **`interior_mask` 도입 (D-028, 2026-05-14)** — fluid mask 만으로
  PersonAgent 가 건물 외부에 spawn 되는 버그 (L-016) 를 해결. 46 FDS
  시나리오의 final-frame `risk > 0.6` cell 의 union (∩ fluid mask) 을
  `data/processed/interior_mask.npz` 에 저장; z=3 에 701 spawnable cells.
  `scripts/build_interior_mask.py` + `scripts/visualize_interior_mask.py`
  + `scripts/sweep_interior_threshold.py` 로 재생성 / 진단 가능.
- **PersonAgent FED 활성화 — M2-full + M3-full 완료 (D-029, 2026-05-14)** —
  새 `StaticCOField` (`src/risk_map/co_field.py`) 가 raw CO ppm grid 를
  query 하고, `PersonAgent.accumulate_exposure(co, danger, dt)` 가
  ISO 13571 §7.3 simplified FED 를 누적 + `DEAD` 전이. 3 시나리오
  step loop 모두 통합. CO 캐시 `results/cache/scenario_co_fields/`.
- **EXP-PATH-001 large sweep 완료 (D-025 default scale)** —
  `results/exp_path_001/comparison.csv` (45 rows = 3 fires × 5 seeds × 3 scenarios,
  **20 agents each × 300 s**, 실 STL 빌딩) +
  `figures/exp_path_001/comparison.png`
  - S1 fixed-sign: evac 54.7% (40-65% spread), 1500kW exposure 55-96s,
    FED mean 0.0018 (max 0.0043)
  - S2 FDS swarm:  evac 100.0% (15/15 rows), exposure ≤ 0.9s, t_evac ~11.4s,
    FED mean 0.0000230
  - S3 model swarm (Tier1RiskMap): S2 와 거의 동일 (FED Δ < 1e-7,
    H5 transitive 유지)
  - **H6 PASS**: FED ratio S2/S1 = 0.0127 (S2 가 S1 대비 **-98.7%**,
    목표 ≤ 0.7 충족). +45.3%p evac + exposure -24.5s 동반.
  - casualty_rate=0 (300 s 에서도 FED 0.3 threshold 미도달; 작은 빌딩
    + dt_SLCF=10s 기반 보간된 CO ppm 이 ~100-500 ppm 수준이라 그 시간
    안에 누적 부족. **H6 primary metric 은 통과** 했으므로 sufficient.).

**Hypothesis state (2026-05-14)**:
- H1 ✅ — 52,000× speedup (GNN ~26 ms vs FDS ~23 min)
- H2 ✅ — ConvLSTM RelL2 0.136 (FNO PI 0.157 marginal)
- H3 ⚠ partial — full SLCF에서 FNO < ConvLSTM, sparse 39 sensor regime
  에서 FNO no-PI + geodesic 이 ConvLSTM 우위
- H4 ✅ — Tier 1 GNN FNR 4.6%
- H5 ✅ — Tier 1 GNN IoU 0.904 (13/13 OOD); EXP-PATH-001 mini-sweep 에서
  path-level transitive 유지 (S3 FED Δ from S2 < 1e-7)
- **H6 ✅ — drone swarm 이 fixed sign 대비 FED -98.7%** (목표 ≥ 30%
  감소 충족). 20-agent × 300 s × 5 seeds × 3 fires = 45 runs.
  S1 mean FED 0.00181 vs S2 0.0000230, ratio 0.0127.

**Plan B activated** (paper reframing):
> 단일 39-detector 인프라 위에서 binary signal + 12K-param GNN 이
> continuous signal + 1.78M-param 모델을 2.1× 능가. Phase-transition
> 도메인에서는 inductive bias matching 이 capacity 보다 dominant.

**Recent decisions**:
- D-022: H6 metrics = peak_danger / time_in_hazard / aset_margin / fed_final
  (per-trajectory diagnostic; EXP-PATH-001 헤드라인은 D-025 의 5-metric)
- D-023: 30 → 33 scenarios, 4 → 3 HRR levels (500/1000/1500 kW)
- D-024: all 33 → train; val/ood Member A 의 T01–T05 13건으로 충당
- D-024 v3.3: floorplan-based 39 sensors (22 rooms + 14 corridors + 3 exits)
  *— 두 D-024 항목 번호 충돌 정리 필요*
- **D-025 (2026-05-14): H6 재정의 + drone swarm 도입 (D-013 반전)** —
  EXP-PATH-001 을 3 PyBullet 시나리오 비교 (S1 fixed-sign / S2 FDS swarm /
  S3 model swarm) 로 전환, 5-metric (evac_success_rate / mean_evac_time /
  danger_zone_exposure / casualty_rate / cumulative_FED)
- **D-026 (2026-05-14): Path-planning graph = fluid-cell grid (60×40×6)** —
  building.py 19-node 그래프 폐기, FDS fluid mask 에서 파생된 cell grid
  로 통일 (`src/path_planning/building_graph.py::build_graph`).
- **D-028 (2026-05-14): spawn pool = `interior_mask`** — fluid mask 대신
  46 FDS 시나리오 final-frame `risk>0.6` union ∩ fluid 를 사용
  (`scripts/build_interior_mask.py`, `data/processed/interior_mask.npz`).
  사용자 통찰: open space 는 연기 분산으로 risk 누적 안 됨 → high-risk
  cell = 내부.
- **D-029 (2026-05-14): PersonAgent FED 활성화 (M2-full + M3-full)** —
  새 `StaticCOField` 클래스로 raw CO ppm 을 query, `accumulate_exposure`
  메서드가 ISO 13571 FED 누적 + `DEAD` 전이. EXP-PATH-001 의 H6 primary
  metric 측정 가능 → **H6 PASS** (FED -98.8%).

**Recent lessons** (`docs/lessons_learned.md`):
- L-014: conda-forge pybullet 설치가 pip numpy 를 깨뜨림 — 결정적 복구 절차
  (pybullet=3.25=py311hbc92ba2_3 build 명시 + `pip install --force-reinstall
  --no-deps numpy==1.26.4`)
- L-015: PyBullet `getContactPoints` 가 mass=0 body 끼리 비어 있음 →
  `getClosestPoints(distance=0.0)` 사용 (kinematic agent + 정적 obstacle)
- L-016: fluid mask 만으로는 PersonAgent 가 건물 외부 (open courtyard /
  외부 yard / building 사이 strip) 에 spawn — open door 로 외부 fluid 와
  내부 fluid 가 단일 component 로 묶여 `only_exit_reachable` 필터 무효.
  `interior_mask` (FDS final-frame `risk>0.6` union ∩ fluid) 로 해결.
- L-017: `StaticRiskMap` 은 T/V/CO aggregated danger 만 보유 — FED 누적
  엔 raw CO ppm 이 필요. 별도 `StaticCOField` 클래스 도입 (대칭 API +
  OOB 시 0 ppm safe default). RiskMap 의 abstraction 보존.

**Next priority** (`docs/90_next_steps.md`):
1. ★★★ **Paper draft + 발표 슬라이드** — H6 PASS 확정 (large sweep 45
   runs). Figure 4.4 (`figures/exp_path_001/comparison.png`) 가 primary
   visual. H1/H2/H4/H5/H6 모두 PASS, paper 본격 진입 가능.
2. ★★ M4-full Crazyflie URDF + drone-swarm 실제 비행 — 현재 guidance
   로직만 동작; 시각화 임팩트 + 발표 영상 자료용.
3. ★ Tier 1 GNN inference time 정확 측정 (H1 수치 확정 — 현재 CLAUDE.md
   "52,000×" 는 추정값).
4. ★ M5-full 실제 FNORiskMap (현재 S3 는 Tier1RiskMap substitute) —
   진짜 PI-FNO 와 Tier1 둘 다 H5 transitive 통과시키는지 확인.
5. ★ A2 (trimesh + 실 STL → URDF) — 이미 완료 (`assets/building.urdf`),
   별도 작업 없음.

**Active blockers**:
- (모두 해소) FED activation, path planning, Tier1RiskMap, val/ood,
  interior spawn — 전부 closed. H1/H2/H3-partial/H4/H5/H6 hypothesis state
  안정화 + paper 단계 진입 준비 완료.

---

## Quick Reference for Claude Code Sessions

When you start a new session and don't know what to do:

1. Read this file (you are here)
2. Check **"Current Project State"** above for last status
3. Check `docs/manual_v2.md` for week-by-week schedule
4. If a specific task is requested, follow `docs/task_request_template.md`
5. Before writing code, confirm tensor shapes in `docs/interface_contracts.md`

When you finish a task:

1. Run the self-test (`__main__` block prints PASS)
2. Run pytest for relevant test file
3. Update **"Current Project State"** above with new status
4. If a new bug was found and fixed, append to `docs/lessons_learned.md`
5. If a new design decision was made, append to `docs/decisions.md`
