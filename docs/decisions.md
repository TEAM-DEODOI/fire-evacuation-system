# Decision Log

> Each major project decision logged with date, decision, alternatives,
> and rationale. **Append-only** — past decisions are not edited.
>
> When adding: assign next D-NNN number, fill the four fields, commit.

---

## D-001: Single floor, not multi-floor

**Date**: Project inception (Week 0)
**Decision**: Building is a single floor only.
**Alternatives**: Multi-floor with stairwells; high-rise.
**Rationale**: Multi-floor doubles or triples computational cost in FDS,
introduces stairwell evacuation modelling, and risks scope blow-up.
Single floor is sufficient to demonstrate the dynamic-vs-static
path-planning value proposition.

---

## D-002: Mesh resolution 0.5 m

**Date**: Week 0
**Decision**: Cell size 0.5 m × 0.5 m × 0.5 m → 60 × 40 × 6 SLCF grid.
**Alternatives**: 0.2 m (150 × 100 × 15 cells) for higher fidelity.
**Rationale**: 0.2 m would make FDS scenarios take 8–10× longer to run
(~40+ CPU-hours per scenario). 30 scenarios at that rate would
consume the entire RunPod budget. 0.5 m is sufficient for spatial
features at human-evacuation scale (room widths, corridor widths).

---

## D-003: Maze-style building, not simple rectangle

**Date**: Week 3
**Decision**: Maze-style layout with multiple rooms, intersections,
central courtyard, and 3 exits (NE end, SW end, mid-side).
**Alternatives**: Single corridor with fire at one end.
**Rationale**: A simple rectangle makes Dijkstra near-optimal — defeats
the purpose of comparing dynamic vs static planners. Maze structure
ensures Dijkstra and dynamic planners produce visibly different
paths under fire spread. 3 exits provide path diversity (asymmetric
distance from any fire location).

---

## D-004: SLCF region (60×40×6) ≠ FDS MESH (100×80×8)

**Date**: Week 3 (after fdsreader bug discovery)
**Decision**: FDS MESH includes 10 m external buffer for ventilation
boundary conditions; SLCF extracts only the building footprint.
**Alternatives**: Make FDS MESH = SLCF region (no buffer).
**Rationale**: Without external buffer, FDS treats building edges as
hard walls, distorting smoke/heat transport near doors. With buffer,
ventilation boundaries are physically reasonable. The model only
sees the SLCF region — buffer is invisible to ML.

---

## D-005: Three indicators (T, V, CO), not full toxic suite

**Date**: Week 1
**Decision**: ML predicts only Temperature, Visibility, CO.
**Alternatives**: Add HCN, irritant gases, radiant heat flux.
**Rationale**: ISO 13571 §5–7 designates these three as the dominant
indicators for typical residential fires. Adding more channels increases
model complexity without proportional benefit. Documented in
`risk_indicators.md` as conservative simplification.

---

## D-006: Visibility uses inverse normalisation

**Date**: Week 2
**Decision**: `V_norm = 1 − clip(V/30, 0, 1)`. Higher value = more dangerous.
**Alternatives**: Direct normalisation, or treat visibility as separate
sign convention.
**Rationale**: Keeps all 3 model output channels with the same
"high = dangerous" semantics, simplifying loss functions and risk
score computation. Documented prominently in `interface_contracts.md`.

---

## D-007: ConvLSTM as baseline, PI-FNO as primary

**Date**: Week 1
**Decision**: Train ConvLSTM (3D extension) as baseline; PI-FNO as
primary model.
**Alternatives**: Use only PI-FNO (less work but no comparison baseline).
**Rationale**: A baseline is necessary to demonstrate "PI-FNO beats X"
in EXP-FIRE-001. ConvLSTM is a natural choice because it's the
prior-art deep-learning sequence-prediction architecture for grid data.

---

## D-008: FED constant 27000 (not Purser exponent)

**Date**: Week 4
**Decision**: Use simplified FED formula
`FED = (Δt_min/27000) · Σ CO_ppm` instead of Purser's
`FED = Σ ([CO]^1.036) · Δt / C_t`.
**Alternatives**: Full Purser formula with exponent.
**Rationale**: Simplification preserves the linear-time-integral
structure that allows efficient path-integrated FED computation.
The exponent matters most at very high concentrations, which we
already saturate via clipping. ISO 13571 §7.3 references the
27000 ppm·min reference dose.

---

## D-009: FED threshold 0.3 (sensitive population)

**Date**: Week 4
**Decision**: `FED_THRESHOLD = 0.3` for the danger criterion.
**Alternatives**: 1.0 (healthy adults).
**Rationale**: Real evacuations include elderly, children, people with
respiratory conditions. ISO 13571 §7.3 explicitly recommends 0.3 for
sensitive populations. NFPA 130 (transit) also uses 0.3.

---

## D-010: VECTOR=.TRUE. forbidden on SLCF

**Date**: Week 5 (after fdsreader bug)
**Decision**: All SLCF lines must NOT have `VECTOR=.TRUE.`.
**Alternatives**: Use `VECTOR=.TRUE.` for vector field outputs (wind, etc.)
**Rationale**: `VECTOR=.TRUE.` + `CELL_CENTERED=.TRUE.` causes
`fdsreader` to fail with array broadcast errors on slice loading.
This was discovered the hard way during initial data validation.
Logged separately in `lessons_learned.md` as L-001.

---

## D-011: Training data 30 scenarios, not 100+

**Date**: Project inception
**Decision**: 24 train / 3 val / 3 OOD = 30 total.
**Alternatives**: 100+ scenarios for stronger generalisation guarantees.
**Rationale**: At ~25 minutes per scenario × 30 = 12.5 hours total.
RunPod budget supports this comfortably. 100+ would consume budget
without proportional benefit at student-team timeline. Standard
practice in surrogate ML for CFD literature.

---

## D-012: HRR variation 4 levels, not continuous

**Date**: Week 4
**Decision**: HRR ∈ {500, 1000, 1500, 2000} kW.
**Alternatives**: Continuous HRR sampling.
**Rationale**: Discrete levels simplify scenario indexing and OOD
construction. The 4 levels span a factor of 4× in fire intensity,
which is enough to test if the model interpolates within and
extrapolates beyond.

---

## D-013: Single Crazyflie drone for PyBullet, not swarm

**Date**: Week 1
**Decision**: PyBullet integration uses 1 drone.
**Alternatives**: Multi-drone swarm.
**Rationale**: Swarm coordination is a separate research problem.
A single drone is sufficient to demonstrate end-to-end use of the
risk map for evacuation guidance.

---

## D-014: Replan period 30 s for dynamic planner

**Date**: Week 11 (proposed)
**Decision**: `DynamicPredictivePlanner` re-plans every 30 s with 60 s lookahead.
**Alternatives**: Replan every step; replan never.
**Rationale**: Replan-every-step is computationally wasteful (PI-FNO
inference at 10 Hz adds up) and doesn't change the path much.
Replan never reduces dynamic planner to static planner. 30 s aligns
with 60 s prediction horizon (replan halfway through horizon).

---

## D-015: STL building height 3.2 m preserved, SLCF Z = 0–3 m only

**Date**: Week 5 (after STL inspection)
**Decision**: Real STL building can be up to 3.2 m tall; SLCF extraction
window is Z = 0 to 3 m only (6 cells × 0.5 m).
**Alternatives**:
- (A) Rescale STL to 3.0 m height → loses architectural realism
- (B) SLCF Z = 0 to 3.5 m (7 cells) → fdsreader broadcast error
- (C) SLCF Z = 0 to 4 m (8 cells, matches MESH) → larger grid (60, 40, 8),
  changes all interface contracts, breaks (60, 40, 6) convention
- (D) **Selected**: STL 3.2 m physical, SLCF 3.0 m model-visible

**Rationale**: Decision D minimizes interface changes (everything
stays at `(60, 40, 6)`) while preserving physical realism of the building.
The 3.0–3.2 m sliver is the hottest smoke layer but does not affect
breathing-zone (1.5 m) safety analysis. Documented in
`coordinate_convention.md` and `lessons_learned.md` (L-009).

---

## D-016: Three exits in maze layout, asymmetric placement

**Date**: Week 3 (refining D-003)
**Decision**: 3 exits, placed at NE end, SW end, and one mid-side.
**Alternatives**: 2 exits (one each end), 4 exits (more symmetric).
**Rationale**: 3 exits provide enough path diversity that fire location
strongly affects optimal exit choice (good for EXP-PATH-001). 4 exits
would make most fire scenarios trivially solvable. Asymmetric placement
ensures the fire-aware planner has different optimal paths from
different starting positions, demonstrating its value.

---

## D-017: Walking speed 1.5 m/s for evacuation simulation

**Date**: Week 11 (proposed)
**Decision**: `EvacuationSimulator.walking_speed_mps = 1.5`.
**Alternatives**: 1.0 (slow), 2.0 (running).
**Rationale**: 1.5 m/s is the SFPE Handbook standard for unimpeded
adult walking speed in fire egress. Reducing to 1.0 m/s would model
panic conditions; 2.0 m/s is over-fast for crowded conditions.

---

## D-018: PI loss 4-stage curriculum

**Date**: Week 8 (proposed)
**Decision**: PI loss components introduced in 4 stages over 100 epochs:
- Stage 1 (epochs 0–25): MSE only
- Stage 2 (25–50): + mass conservation (CO)
- Stage 3 (50–75): + heat diffusion residual
- Stage 4 (75–100): + tenability boundary

**Alternatives**: All-on from start; or simpler 2-stage curriculum.

**Rationale**: PI-FNO training is unstable when all loss terms are
on simultaneously, especially with random initialization. The data
loss must dominate early to bootstrap learning. Curriculum learning
is standard practice in physics-informed deep learning literature.
The progression mass → heat → boundary roughly orders these terms by
stability (mass conservation is most reliable; heat residual depends
on quality of T predictions; tenability is most "downstream").

---

## D-022: H6 가설 메트릭 확장 (FED 단독 → 4개 메트릭 조합)

**Decision**:
H6 가설 (동적 vs 정적 경로 알고리즘 비교) 검증 메트릭을 누적 FED 단독에서
4개 메트릭 조합으로 확장. FED는 보조 지표로 유지.

| 메트릭 | 정의 | H6 가시성 |
|---|---|---|
| peak_danger | 경로상 최대 위험도 [0, 1] | ★★★★★ |
| time_in_hazard_s | 위험 영역(>0.5) 체류 시간 (초) | ★★★★★ |
| aset_margin_s | ASET 안전 여유 시간 (초) | ★★★★★ |
| fed_final | 누적 FED (보조) | ★★☆☆☆ |

**Rationale**:
- 1500 kW 화재, 300초 시뮬레이션에서 누적 FED 최대값 = 0.043 (임계값 0.3 미달)
- 시뮬레이션 시간 (300초)이 FED 누적 시간 척도 (30분~1시간) 대비 짧음
- 화재 크기 증가는 비현실적 + 30건 재시뮬레이션 비용 큼
- Peak Danger / Time-in-Hazard는 시뮬레이션 시간 길이와 무관
- 동적 알고리즘의 본질적 가치 (미래 위험 회피)를 더 직접 측정

**Discovered**:
2025년 first_sim 위험지도 분석 시 발견. validate_risk_map.py 출력에서
3개 관측점 모두 FED < 0.043 < 0.3 (임계값).

**Implementation**:
- 신규 모듈: src/risk_map/path_metrics.py
- 핵심 클래스: PathSafetyMetrics (dataclass)
- 통합 함수: evaluate_path_safety()
- 기존 D-008 (FED simplified), D-009 (FED threshold 0.3) 변경 없음

**Status**: Implemented. EXP-PATH-001 (Week 12)에서 활용 예정.

---

## D-023: 시나리오 30 → 33 + 4 HRR → 3 HRR (D-011/D-012 개정)

**Date**: 2026-05-12 (실 데이터 도착 시점)
**Decision**: 본 학습에 사용할 시나리오를 30 (4 HRR × 6 location) 에서
**33 (3 HRR × 9 location + 3 H location × 2 HRR)** 로 변경. HRR 레벨도
{500, 1000, 1500, 2000} 에서 **{500, 1000, 1500}** 으로 축소.

**구성**:

| Split | 카운트 | 구성 |
|---|---|---|
| train | 21 | _001~_007 × 3 HRR (7 location × 3) |
| val | 6 | _008, _009 × 3 HRR (2 held-out location × 3) |
| ood | 6 | H01~H03 × 2 HRR (3 new location × 500/1000kW — 1500kW H 없음) |

**Alternatives**:
- 매뉴얼 spec 그대로 (4 HRR × 6 loc = 30): 2000kW 시뮬레이션 추가 필요. 비용 큼.
- HRR 1개 더 (2000kW): 9개 location × 2000kW = 9건 추가 비용.
- 21건 (1000/1500kW 만): 학습 데이터 적음, HRR 다양성 약함.

**Rationale**:
- Member A 가 4 HRR 대신 3 HRR (500/1000/1500) 로 데이터셋 구성. 각 HRR에서 9개
  location + 1000/500kW 에서만 3개 추가 H location.
- 결과적으로 spatial diversity 가 매뉴얼 (6 loc) 대비 더 풍부 (총 12 unique location).
  Path planning H6 가설 (EXP-PATH-001) 에서 generalization 평가에 더 유리.
- HRR 2000kW 부재는 EXP-FIRE-001 의 HRR-extrapolation 평가가 좁아지지만, EXP-PATH-001
  헤드라인엔 영향 없음.
- 1500kW H 부재 (OOD가 500/1000kW 한정) 는 비대칭이지만 generalization 검증엔 충분.

**Implementation**:
- `src/shared/constants.py`: `N_SCENARIOS_TOTAL = 33`, train/val/ood = 21/6/6.
  `HRR_LEVELS_KW = (500, 1000, 1500)`.
- `tests/test_constants.py`: 위 값 검증으로 업데이트.
- `data/raw/` 의 33개 디렉토리 (`sim_500kw_1m2_001` 등) 를 canonical 이름
  (`s_000`~`s_020`, `s_val_0`~`s_val_5`, `s_ood_0`~`s_ood_5`) 으로 rename.
  원래 이름은 `scenario_config.json` 의 `original_id` 필드에 보존.
- D-011 (총 30 시나리오), D-012 (4 HRR 레벨) 는 D-023 으로 개정됨.

**Status**: Implemented. ConvLSTM/PI-FNO 학습은 이 33-시나리오 dataset.h5 에서 진행.

---

## D-024: 33건 전부 train, val/ood 는 별도 시뮬레이션으로 보충

**Date**: 2026-05-12
**Decision**: D-023 의 21/6/6 split 을 폐기. 현재 보유한 **33건 전부 train**
으로 사용하여 학습 데이터 최대화. val 과 ood 는 추후 별도 시뮬레이션 (예:
2000kW HRR, 1500kW H location 등) 으로 채움.

**Alternatives**:
- D-023 그대로 (21 train / 6 val / 6 ood): 학습 데이터 손실, val·ood 분포가
  여전히 학습 분포와 같은 시뮬레이션 배치에서 나옴.
- Train/val 임의 split (예: 27/6): 통계 noise 큼, 적은 데이터로 monitoring
  의 가치 작음.

**Rationale**:
- 33건은 매뉴얼 spec (30) 보다 약간 많지만 ML 학습 기준으론 여전히 적음.
  6건을 val/ood로 떼는 것보다 33건 전부 학습이 robust generalization 에 유리.
- val/ood 가 추후 추가될 때 학습 분포 *밖* 의 시나리오 (다른 HRR, 다른 화재
  위치, 다른 빌딩 변형 등) 가 들어와야 H6/EXP-FIRE-001 evaluation 이 의미를
  가짐. 같은 batch 에서 떼낸 hold-out 은 그 의미가 약함.
- 단점: 학습 중 best-checkpoint monitoring 신호 없음. 대신 **train loss 기반**
  으로 best 저장 (또는 매 epoch fixed checkpoint 옵션 사용).

**Implementation**:
- `data/raw/` 디렉토리: `s_val_*`, `s_ood_*` (12개) → `s_021`–`s_032` 로 rename.
  모든 33 디렉토리가 ``s_000``–``s_032`` 의 균일 명명.
- `scenario_config.json`: 33 항목 모두 `"split": "train"`.
- `src/shared/constants.py`: `N_SCENARIOS_TRAIN = 33`, `N_SCENARIOS_VAL = 0`,
  `N_SCENARIOS_OOD = 0`.
- `src/dataset/fire_dataset.py`: 빈 split 허용 (empty Dataset 반환).
- `src/training/trainer.py`: val_loader 가 비었거나 ``None`` 이면 val pass
  스킵, train loss 기준으로 best checkpoint 저장.

**Status**: Implemented. 추후 val/ood 시뮬레이션 추가 시 scenario_config.json
에 새 항목 추가 + 재빌드만 하면 됨.

---

## How to Add a Decision

When making a major scope or interface decision:

1. Write a new section labeled `D-NNN`.
2. Date, Decision (one line), Alternatives, Rationale.
3. Keep concise — 3–5 sentences for rationale.
4. Update `CLAUDE.md` constraints if the decision changes them.
5. Commit with message `decisions: D-NNN - <one-line summary>`.

Do not edit past decisions; if a decision is reversed, write a new
entry explaining the reversal.
