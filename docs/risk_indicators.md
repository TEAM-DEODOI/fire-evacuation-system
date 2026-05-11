# Risk Indicators and Tenability Thresholds

> Reference document for tenability analysis.
> All thresholds are drawn from **ISO 13571:2012** and the
> **SFPE Handbook of Fire Protection Engineering** (5th ed.).
>
> See `CLAUDE.md` for project context, `interface_contracts.md` for module APIs.

---

## Korean Glossary (한국어 용어 정리)

| English | 한국어 | 설명 |
|---------|-------|------|
| Tenability | 거주가능성 / 인명안전조건 | 사람이 노출되어도 안전한 환경 |
| Fractional Effective Dose (FED) | 누적유효노출량 | 시간에 걸친 독성 누적량, ISO 13571 핵심 지표 |
| Available Safe Egress Time (ASET) | 가용피난시간 | 위험해질 때까지 남은 시간 |
| Required Safe Egress Time (RSET) | 요구피난시간 | 출구까지 도달하는 데 걸리는 시간 |
| Wayfinding | 피난동선 인식 / 길 찾기 | 가시거리가 결정 |
| Incapacitation | 행동불능 | FED = 1.0 시점에서 50% 의식상실 |
| Smoke layer | 연기층 | 천장 부근 가장 뜨거운 층 |
| Heat flux | 열복사량 | 본 프로젝트에서는 다루지 않음 |
| Hyperventilation | 과호흡 | 본 프로젝트에서는 다루지 않음 |

---

## Overview

Three instantaneous indicators and one cumulative indicator are used:

| Indicator | Symbol | Type | Source |
|-----------|--------|------|--------|
| Temperature | T | Instantaneous | ISO 13571 §5.2 |
| Visibility | V | Instantaneous | ISO 13571 §5.4 |
| CO concentration | [CO] | Instantaneous | SFPE Handbook Table 2-6.3 |
| Fractional Effective Dose | FED_CO | Cumulative | ISO 13571 §7.3 |

---

## Thresholds

### Temperature

| Level | °C | Notes |
|-------|----|-------|
| Safe | ≤ 30 °C | Sustained exposure negligible risk |
| Danger | ≥ 60 °C | Threshold for humid-air convective heating causing incapacitation (ISO 13571 Table 1) |

Normalisation (used by models):

```
T_norm = (T_celsius − 20) / 1180,   clipped to [0, 1]
```

Range: 20 °C → 0.0; 1200 °C → 1.0. Domain motivated by FDS output range.

Danger score (used by risk map):

```
d_T = clip((T − 30) / (60 − 30), 0, 1)
```

This is a linear ramp: 30 °C → 0.0; 60 °C → 1.0; saturates beyond.

### Visibility

| Level | m | Notes |
|-------|---|-------|
| Safe | ≥ 10 m | Occupants can locate exit signs unaided |
| Danger | ≤ 3 m | Wayfinding severely impaired; disorientation likely |

References: ISO 13571 §5.4; SFPE Handbook Ch. 2-6 (Purser).

**Inverse mapping**: unlike T and CO, *higher* raw visibility means
*safer*. The normalised value is therefore inverse:

```
V_norm = 1 − clip(V / 30, 0, 1)
```

Range: 30+ m → 0.0; 0 m → 1.0.

Danger score:

```
d_V = clip((10 − V) / (10 − 3), 0, 1)
```

This is a linear ramp: 10 m → 0.0; 3 m → 1.0; saturates beyond.

### CO Concentration (Instantaneous)

| Level | ppm | Notes |
|-------|-----|-------|
| Safe | ≤ 100 ppm | Slight effect on sensitive individuals after prolonged exposure |
| Danger | ≥ 1400 ppm | Headache, dizziness, incapacitation risk within minutes (SFPE Table 2-6.3) |

Normalisation (used by models):

```
CO_norm = log1p(CO_ppm) / log1p(5000),   clipped to [0, 1]
```

Log scale chosen because CO hazard is non-linear and spans several
orders of magnitude.

Danger score:

```
d_CO_inst = clip((CO − 100) / (1400 − 100), 0, 1)
```

---

## Fractional Effective Dose (FED) — CO

The FED accounts for **time-integrated exposure** and is the appropriate
indicator for predicting incapacitation along an evacuation path.

### Formula (ISO 13571 §7.3, simplified)

For a single occupant moving through the field:

```
FED_CO(t) = ∫₀ᵗ [CO](τ) / 27000 dτ          (with [CO] in ppm·min)
```

Equivalently in discrete time with frame interval Δt = 10 s:

```
FED_CO(t_n) = (Δt_min / 27000) · Σ_{k=0..n} CO_ppm[k]
            = (1/6 / 27000) · Σ_{k=0..n} CO_ppm[k]
```

where `Δt_min = 10 / 60 = 1/6` minutes.

### Why 27000?

The constant **27000 ppm·min** is the ISO 13571 reference dose for
incapacitation in an average healthy adult. This corresponds to:

```
FED = 1.0 ⟹ ~50% of healthy adults incapacitated.
FED = 0.3 ⟹ ~10% of sensitive individuals incapacitated.
```

We use the simplified form (without the Purser exponent `n = 1.036`)
because:

1. The exponent matters most at very high concentrations, which we
   already saturate via clipping.
2. The simplified form preserves the linear-time-integral structure
   that lets us compute FED along a path with low overhead.
3. Documented as decision D-008 with rationale.

### Threshold for this project

| Population | FED threshold |
|------------|---------------|
| Normal (healthy adults) | 1.0 |
| **Sensitive (used in this project)** | **0.3** |

The sensitive-population threshold (FED = 0.3) is specified in
**ISO 13571:2012 §7.3** and is adopted here because real evacuations
include elderly, children, and individuals with respiratory conditions.

### Path-integrated FED

For a moving occupant whose position at time `t` is `x(t)`:

```
FED_path = (Δt_min / 27000) · Σ_t CO_ppm(x(t), t)
```

This is the metric used by EXP-PATH-001 to compare static vs dynamic
evacuation paths.

---

## ASET (Available Safe Egress Time)

For each cell `(x, y, z)` in the SLCF region:

```
ASET(x, y, z) = first time t* such that d_total(t*, x, y, z) > danger_threshold
              = T_max  if cell never becomes dangerous
```

where `danger_threshold` is typically 0.5 (configurable in `risk_map.yaml`).

ASET maps are computed once per scenario and used to:
- Identify the spatial extent of safe regions over time
- Compare across model predictions (EXP-RISK-001)
- Provide bound on path planning horizon

```python
def compute_aset_map(risk_grid_time: np.ndarray,
                     danger_threshold: float = 0.5) -> np.ndarray:
    """Returns (60, 40, 6) ASET map in seconds."""
    T = risk_grid_time.shape[0]  # 31
    aset = np.full(risk_grid_time.shape[1:], 300.0)  # default = T_END
    for t_idx in range(T):
        t_seconds = t_idx * 10.0
        cells_dangerous = risk_grid_time[t_idx] > danger_threshold
        new_dangers = cells_dangerous & (aset == 300.0)
        aset[new_dangers] = t_seconds
    return aset
```

---

## Aggregated Danger Score

The risk map combines per-cell instantaneous indicators into a single
scalar `d ∈ [0, 1]`:

```
d_total = clip(0.4 · d_T + 0.4 · d_V + 0.2 · d_CO_inst, 0, 1)
```

Default weights (modifiable in `configs/risk_map.yaml`):

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Temperature (d_T) | **0.4** | Direct life-safety, fast-acting |
| Visibility (d_V) | **0.4** | Wayfinding-critical, drives decisions |
| CO instantaneous (d_CO_inst) | **0.2** | Slow accumulation; cumulative FED handles this better |

Cumulative FED is **separately tracked**, not folded into `d_total`,
because it is path-dependent and applies per occupant rather than
per cell.

---

## H6 가설 검증 메트릭 (D-022 적용)

H6: 동적 예측 위험가중 경로(우리 방법)는 정적 경로(베이스라인)보다 안전하다.

**Phase 1 권장 — 4개 메트릭 조합** (Week 12 EXP-PATH-001에서 사용):

| # | 메트릭 | 정의 | 단위 | 권장 |
|---|---|---|---|---|
| 1 | peak_danger | 경로상 최대 위험도 (TENABILITY 가중) | [0, 1] | ★★★★★ |
| 2 | time_in_hazard_s | 위험 영역 (danger > 0.5) 체류 시간 | 초 | ★★★★★ |
| 3 | aset_margin_s | ASET - arrival_time 최소값 | 초 | ★★★★★ |
| 4 | fed_final | 누적 FED (ISO 13571 simplified) | dim'less | ★★☆☆☆ |

**해석**:
- peak_danger: 0에 가까울수록 안전, 1에 가까울수록 위험.
  정적(베이스라인) 0.85+, 동적(우리 방법) < 0.5 예상.
- time_in_hazard_s: 0초가 이상적. 정적은 위험 영역을 통과하므로 양수,
  동적은 회피로 0에 가까움.
- aset_margin_s: 양수 = 안전 도달, 음수 = 도달 시 이미 위험.
  최소값을 사용하여 가장 위험한 순간 측정.
- fed_final: D-008/D-009 그대로. 단 300초 시뮬레이션에서는 임계값 0.3
  도달 어려움 — 보조 지표로 활용.

**왜 FED 단독으로 부족한가**:
1500 kW 화재 + 300초 시뮬레이션에서 누적 FED 최대값 = 0.043
(임계값 0.3의 14%). FED는 누적 노출 지표라 시뮬레이션 시간 (5분)이
짧으면 자연히 차이 작음. 화재 크기 증가는 비현실적 + 재시뮬 비용 큼.
Peak Danger / Time-in-Hazard는 시간 척도와 무관 → 더 직접적 H6 검증.

**구현 위치**:
- 모듈: `src/risk_map/path_metrics.py`
- 데이터클래스: `PathSafetyMetrics` (불변)
- 통합 함수: `evaluate_path_safety()`
- 의존: `RiskMap.query`, `src/risk_map/aset.py`, `src/risk_map/fed.py`

**EXP-PATH-001 발표 자료 형식**:

```
시나리오: OOD-1 (중앙 홀 화재, 1500 kW)

알고리즘     Peak Danger  Time-in-Hazard  ASET Margin  FED Final  Arrived?
─────────────────────────────────────────────────────────────────────────
정적 (D)     0.87         42 s            -18 s        0.052      ✓
정적 (S)     0.74         28 s             -8 s        0.048      ✓
동적 (우리)  0.41          8 s            +52 s        0.041      ✓
─────────────────────────────────────────────────────────────────────────
개선         53% ↓        81% ↓           +70 s        21% ↓
```

---

## Implementation Reference

Core functions (Week 10):

```python
from src.shared.constants import TENABILITY
from src.risk_map.tenability import (
    compute_danger_temperature,
    compute_danger_visibility,
    compute_danger_co,
    compute_total_danger,
)
from src.risk_map.fed import accumulate_fed_co
from src.risk_map.aset import compute_aset_map
```

The `TENABILITY` dataclass exports:

```python
TENABILITY.T_SAFE_C        = 30.0
TENABILITY.T_DANGER_C      = 60.0
TENABILITY.V_SAFE_M        = 10.0
TENABILITY.V_DANGER_M      = 3.0
TENABILITY.CO_SAFE_PPM     = 100.0
TENABILITY.CO_DANGER_PPM   = 1400.0
TENABILITY.FED_REFERENCE   = 27000.0   # ppm·min
TENABILITY.FED_THRESHOLD   = 0.3       # sensitive population
TENABILITY.WEIGHT_T        = 0.4
TENABILITY.WEIGHT_V        = 0.4
TENABILITY.WEIGHT_CO       = 0.2
TENABILITY.AGGREGATE_THRESHOLD = 0.5    # for ASET map
```

---

## Visualisation Defaults

For consistency in plots (Week 13):

| Quantity | Colormap | Range |
|----------|----------|-------|
| Temperature (raw) | `inferno` | 20 to 600 °C |
| Visibility (raw) | `viridis_r` | 0 to 30 m |
| CO (raw) | `magma` | 0 to 5000 ppm |
| Danger score | `RdYlGn_r` | 0.0 to 1.0 |
| ASET (seconds) | `viridis` | 0 to 300 s |

---

## Out-of-Scope Hazards

The following are **explicitly excluded** from this project (see
`docs/decisions.md` for rationales):

- **HCN** (hydrogen cyanide) — no FDS data available
- **Irritant gases** (acrolein, HCl) — out of scope
- **Radiant heat flux** — only convective temperature considered
- **Multi-component FED** — CO only
- **Hyperventilation due to elevated CO₂** — not modelled

These omissions make our tenability estimates **conservative under-estimates**
in some respects (we may miss some hazards) and **conservative over-estimates**
in others (we use sensitive-population thresholds). The net effect is
documented to be small (within 15%) for typical residential fire scenarios
per Purser's review work.

---

## Sources

1. **ISO 13571:2012** — *Life-threatening components of fire — Guidelines
   for the estimation of time available for escape using fire data.*
2. **SFPE Handbook of Fire Protection Engineering, 5th ed.**, Ch. 63,
   "Combustion Toxicity," Purser & McAllister, 2016.
3. **NFPA 130** for transit applications (referenced for FED 0.3 sensitive
   threshold confirmation).
4. **Purser, D. A.** "Toxicity assessment of combustion products."
   SFPE Handbook (3rd ed.), Ch. 2-6, 2002.
5. **RWTH Aachen Fire Simulation Lecture Notes** —
   https://firedynamics.github.io/LectureFireSimulation/ — worked
   examples of ASET/RSET in FDS contexts.

---

## Worked Example

For a single FDS scenario:

```python
from src.data_pipeline.fds_extractor import extract_slices
from src.risk_map.tenability import compute_total_danger
from src.risk_map.aset import compute_aset_map

# 1. Extract raw fields from FDS output
slices = extract_slices(Path("data/raw/scenario_000"))
T = slices['temperature']    # (31, 60, 40, 6) °C
V = slices['visibility']     # (31, 60, 40, 6) m
CO = slices['co_ppm']        # (31, 60, 40, 6) ppm

# 2. Compute danger fields
d_T = (T - 30) / 30
d_T = np.clip(d_T, 0, 1)

d_V = (10 - V) / 7
d_V = np.clip(d_V, 0, 1)

d_CO = (CO - 100) / 1300
d_CO = np.clip(d_CO, 0, 1)

# 3. Aggregate
risk = 0.4 * d_T + 0.4 * d_V + 0.2 * d_CO   # (31, 60, 40, 6)

# 4. ASET map
aset = compute_aset_map(risk, danger_threshold=0.5)  # (60, 40, 6)

# 5. Path-integrated FED for a hypothetical occupant
from src.risk_map.fed import accumulate_fed_co
co_at_path = ...  # CO values along the occupant's actual path
fed_seq = accumulate_fed_co(co_at_path, dt_seconds=1.0)
print(f"Final FED: {fed_seq[-1]:.3f} (threshold 0.3)")
```
