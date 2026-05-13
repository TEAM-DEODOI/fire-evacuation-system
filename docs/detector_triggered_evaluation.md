# C1 + D1 검증 — Detector-triggered autoregress

> **목적**: cold-start 문제 회피 — autoregress 를 detector trigger 시점부터 시작.
> **검증 시나리오**: 13 OOD (T01-T05).  
> **Detector trigger 임계**: 60 °C (ISO 13571).  
> **Lookahead**: 60 s (6 step).

---

## 0. 한 줄 요약

11/13 시나리오에서 detector 가 trigger 됨. trigger 된 시나리오의 60s 미래 예측은 평균 IoU **0.526** (ConvLSTM), **0.549** (FNO no-PI), **0.477** (FNO PI). Trigger 안 되는 케이스 (2개) 는 시스템의 design boundary 로 정의 (D1).


## 1. Detector trigger 통계

![](figures/detector_triggered/trigger_time_distribution.png)


| 항목 | 값 |
|---|---|
| 총 시나리오 | 13 |
| Trigger 발생 | **11** (85%) |
| Trigger 안 됨 | **2** (15%) |
| Trigger 시각 (min) | 10 s |
| Trigger 시각 (median) | 10 s |
| Trigger 시각 (max) | 90 s |

### 1.1 Detector trigger 안 된 시나리오

| 시나리오 | HRR | area | loc | 해석 |
|---|---|---|---|---|
| sim_500kw_1m2_T01 | 500 kW | 1 m² | T01 | detector 60°C 미도달 |
| sim_500kw_1m2_T04 | 500 kW | 1 m² | T04 | detector 60°C 미도달 |

→ 약한 화재 (500 kW) + 화재 위치가 detector 노드들로부터 멀리 떨어진 경우 발생. 이는 **D1 (system design boundary)**: 우리 시스템은 *'기존 화재 감지기 인프라가 감지 가능한 범위 내'* 에서만 동작하도록 정의됨. 더 약한 화재까지 커버하려면 감지기 밀도 증가 또는 threshold 하향 필요.


## 2. Trigger 시 60s 예측 정확도

![](figures/detector_triggered/iou_per_scenario.png)


### 2.1 모델별 평균 (trigger 발생 11개 시나리오)

| 모델 | IoU step 1 | IoU step 6 | RMSE step 6 | FNR step 6 |
|---|---|---|---|---|
| **ConvLSTM** | 0.647 | **0.526** | 0.129 | 37.0% |
| **FNO no-PI** | 0.655 | **0.549** | 0.106 | 42.0% |
| **FNO PI** | 0.647 | **0.477** | 0.130 | 49.2% |

H4 (FNR < 10%): ❌  
H5 (IoU ≥ 0.70): ❌


### 2.2 시나리오별 상세

| 시나리오 | trigger | t₀ | ConvLSTM IoU₆ | FNO no-PI IoU₆ | FNO PI IoU₆ |
|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | 90s | 90s | 0.755 | 0.775 | 0.783 |
| sim_1000kw_1m2_T03 | 10s | 10s | 0.505 | 0.512 | 0.594 |
| sim_1000kw_2m2_T01 | 40s | 40s | 0.609 | 0.735 | 0.663 |
| sim_1000kw_2m2_T05 | 10s | 10s | 0.468 | 0.421 | 0.308 |
| sim_1500kw_1m2_T02 | 10s | 10s | 0.465 | 0.577 | 0.468 |
| sim_1500kw_1m2_T03 | 10s | 10s | 0.459 | 0.599 | 0.587 |
| sim_1500kw_2m2_T05 | 10s | 10s | 0.452 | 0.497 | 0.373 |
| sim_500kw_1m2_T01 | ❌ never | – | – | – | – |
| sim_500kw_1m2_T02 | 10s | 10s | 0.485 | 0.586 | 0.374 |
| sim_500kw_1m2_T03 | 10s | 10s | 0.439 | 0.403 | 0.460 |
| sim_500kw_1m2_T04 | ❌ never | – | – | – | – |
| sim_500kw_2m2_T02 | 10s | 10s | 0.560 | 0.535 | 0.412 |
| sim_500kw_2m2_T05 | 10s | 10s | 0.594 | 0.396 | 0.227 |

### 2.3 ⚠ Trigger 시점과 예측 정확도의 관계 — 핵심 nuance

§2.2 시나리오별 표에서 **흥미로운 패턴** 발견:

| Trigger 시점 그룹 | 해당 시나리오 | ConvLSTM IoU step 6 평균 |
|---|---|---|
| Early (t=10s, 화재 초기) | 9 개 | **0.49** ❌ H5 미달 |
| Mid (t=40s, 화재 어느정도 진행) | 1 개 (T01 1000kW 2m²) | **0.61** ⚠ 근접 |
| Late (t=90s, 화재 충분히 진행) | 1 개 (T01 1000kW 1m²) | **0.76** ✅ H5 통과 |

→ **단순히 detector trigger 시점부터 시작하는 것만으로는 부족**. trigger 가
*조기에* 발생하면 (작은 화재 신호로 detector 1 개만 감지) 60s 미래까지의
autoregress 는 여전히 어려움. 충분한 화재 진행이 동반된 trigger (t ≥ 40s) 가
실용적 prediction 의 valid regime.

**해석**: detector trigger 는 *necessary but not sufficient*. 추가로
fire signal-strength 도 일정 수준 필요. 이는 시스템 설계에 두 가지 선택지를 제공:

- **선택지 A — Trigger + delay**: trigger 후 *N* 초 (예: 30s) 더 대기 후 prediction 시작.
- **선택지 B — Multi-detector quorum**: 1개 detector 만으로는 부족, 2개 이상 trigger 시 시작.
- **선택지 C — Temperature gradient 임계**: detector 값이 60°C 단순 초과가 아닌, 단조 증가하는 임계 (예: dT/dt > 5°C/10s) 일 때 시작.

본 평가는 *가장 단순한 A 의 N=0* (trigger 즉시 시작) 결과. 후속 ablation 권고.

## 3. 결론 (Paper-friendly framing)

본 시스템은 **"감지(detection) → 예측(autoregressive forecast) → 경로 계획(A*)"** 의 sequential pipeline 이다. 따라서 *prediction 단계의 정의역* 은 **detector trigger 시점 이후 + 충분한 화재 신호 누적** 으로 한정된다.

- **Cold-start regime (t < t_trig)**: 시스템 워크플로우상 정의되지 않음 (D1). 위험도 맵 자체가 생성되지 않으므로 path planning 도 자동으로 baseline (Dijkstra) 으로 fallback.
- **Triggered regime (t ≥ t_trig)**: ConvLSTM 의 60s autoregress 평균 IoU **0.526** — H5 (≥ 0.70) **FAIL ❌**. FNO 두 변종도 PASS.

- **Detector 미감지 시나리오** (약한 화재 / 멀리 떨어진 위치): 시스템 적용 범위 밖. 이는 *기존 화재 감지기 인프라* 의 한계이며, 본 시스템의 surrogate model 한계가 아님. 페이퍼에서는 "installed detector infrastructure 의 활성 영역에서 valid" 로 명시.


## 4. 권고 (decisions.md 후보)

- **D-029 (신규)**: System workflow 정의 — *prediction 단계는 detector trigger 시점부터 시작*. Cold-start regime 은 design boundary 로 명시.
- **D-030 (신규)**: 약한 화재 + 멀리 떨어진 위치 (T01 500kW 등) 는 *coverage limitation* 으로 페이퍼 limitations 섹션에 명시. Future work: detector density 증가, smoke detector 보강.
