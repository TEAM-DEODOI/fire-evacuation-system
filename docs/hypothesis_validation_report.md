# 가설 검증 종합 보고서 (H1-H6)

> **평가 시점**: 자동 생성  
> **평가 도메인**: OOD T01-T05 (13 scenarios, 새 화재 위치 5개)  
> **모델 후보**: ConvLSTM / FNO no-PI / FNO PI  
> **건물**: training 과 동일 → **위치 OOD only**

---

## 0. 한 줄 요약

3 모델 비교에서 **ConvLSTM 이 single-step 정확도/Risk Map 분류 모두 우위**. FNO no-PI 는 비슷한 수준, FNO PI 는 physics loss 가 fitting 을 살짝 제약함. H1/H4/H5 통과 (모델 무관), **H3(FNO > ConvLSTM) ❌ 실패** → 매뉴얼 Plan B 의 'PI-FNO doesn't beat ConvLSTM' 시그널.


## 1. 가설 검증 게이지 (best-of-3 기준)

| ID | 가설 | 목표 | 측정 (best model) | 통과 |
|---|---|---|---|---|
| **H1** | Speed ≥ 1000× FDS | < 50 ms | **26.5 ms** (52033×) | ✅ |
| **H2** | Single-step RelL2 ≤ 0.15 | OOD T01-T05 | **0.136** (ConvLSTM) | ✅ |
| **H3** | FNO > ConvLSTM on OOD | OOD T01-T05 | **ConvLSTM 0.136 < FNO no-PI 0.138 < FNO PI 0.157** | ❌ |
| **H4** | Risk FNR < 10% | OOD T01-T05 | **6.0%** (ConvLSTM) | ✅ |
| **H5** | Risk IoU ≥ 0.70 | OOD T01-T05 | **0.887** (ConvLSTM) | ✅ |
| **H6** | Dynamic A* FED ≥ 30% ↓ | EXP-PATH-001 | ⚠ **path_planning 모듈 미작성** | 🔜 보류 |


## 2. EXP-FIRE-001 — 3 모델 비교 (OOD T01-T05)

![](figures/hypothesis_validation/model_comparison.png)


### 2.1 핵심 메트릭 표 (13 시나리오 평균)

| Model | Single RelL2 | RMSE °C | RMSE m | RMSE ppm | Autoreg-6 | Risk IoU | Risk FNR | Infer ms |
|---|---|---|---|---|---|---|---|---|
| **ConvLSTM** | 0.136 | 5.68 | 1.34 | 4.26 | 0.882 | 0.887 | 6.0% | 26.5 |
| **FNO no-PI** | 0.138 | 6.98 | 1.39 | 4.61 | 0.770 | 0.828 | 7.2% | 27.1 |
| **FNO PI** | 0.157 | 6.86 | 1.67 | 7.38 | 0.818 | 0.836 | 6.9% | 30.3 |


### 2.2 위치별 모델 비교

![](figures/hypothesis_validation/per_location.png)


### 2.3 H3 가설 결론

- **H3 ❌**: FNO 두 변종 모두 ConvLSTM 을 **이기지 못함**.  
- single-step RelL2 ordering: **ConvLSTM < FNO no-PI < FNO PI**  
- risk-map IoU ordering: **ConvLSTM > FNO PI > FNO no-PI**  
- 유일한 FNO 우위 항목은 **6-step autoregress** — Fourier 도메인의 spatial smoothing 이 누적 오차를 둔화시킨 효과로 추정. 단 path planning lookahead 가 30 s 이내라면 큰 차별점 아님.


#### 왜 FNO 가 ConvLSTM 을 못 이겼는가 (해석)
1. **데이터 regime 영향**: 33 시나리오는 작은 데이터셋. FNO 의 spectral inductive bias 는 풍부한 데이터에서 우위를 보이지만, 33 샘플로는 ConvLSTM 의 local convolution 이 더 효율적으로 작은 패턴(벽/문 인근 smoke 흐름) 을 학습.
2. **위치 OOD 의 본질**: 새 위치 T01-T05 도 같은 격자/벽 구조 안에서 정의됨. 즉 진짜 'distribution shift' 라기보다 'spatial covariate shift'. ConvLSTM 의 spatial locality 가 이 종류 shift 에 더 강함.
3. **PI loss 의 역설**: FNO PI 의 RelL2 (0.157) > FNO no-PI (0.138). PI loss 가 data fitting 을 살짝 제약하면서, 위치 OOD 의 noise 가 큰 영역에서 도리어 generalization 을 도움 — 단, OOD 데이터의 신호 자체가 강해서 fitting 의 손해가 더 컸음.



## 3. 가설별 상세


### H1 — Speed ≥ 1000× FDS  ✅

- 모델 평균 추론 시간: ConvLSTM 26.5 ms / FNO no-PI 27.1 ms / FNO PI 30.3 ms  
- FDS 단일 시나리오 ~23 min → speedup **52,033×**  
- 목표 1000× 를 모든 모델이 **52배 이상 초과 달성**.


### H2 — Single-step RelL2 ≤ 0.15  ⚠ borderline

- OOD 평균 RelL2: ConvLSTM **0.136** / FNO no-PI **0.138** / FNO PI **0.157**  
- ConvLSTM 과 FNO no-PI 는 통과, FNO PI 는 마진 작은 실패  
- 단 시나리오별로 보면 ConvLSTM 도 500kW T01/T03 에서 0.158 / 0.159 로 임계 근방 → training distribution 의 신호/노이즈 비가 낮은 케이스에서 borderline.


### H3 — FNO > ConvLSTM on OOD  ❌ NOT VALIDATED

위 §2.3 참조. **Plan B 적용 권고**:

> *기존 매뉴얼 Plan B (CLAUDE.md L391):*
> *'PI-FNO doesn't beat ConvLSTM → 페이퍼를 "30-scenario regime trade-offs" 로 reframe.'*

**제안 reframing**: 
- 페이퍼 contribution 을 'PI-FNO 의 우위' 가 아니라 **'fire-evac 도메인에 맞는 surrogate model 선택의 데이터-regime 의존성'** 으로 재정의.
- ConvLSTM 을 default 추천하고 PI-FNO 는 large-data regime (≥ 200 scen) 가설로 future work 명시.


### H4 — Risk FNR < 10%  ✅

- OOD FNR: ConvLSTM **6.0%** / FNO no-PI 7.2% / FNO PI 6.9%  
- 모두 통과. 가장 보수적인 ConvLSTM 이 위험 영역 탐지율이 가장 높음.


### H5 — Risk IoU ≥ 0.70  ✅

- OOD IoU: ConvLSTM **0.887** / FNO no-PI 0.828 / FNO PI 0.836  
- 모두 통과. 목표 0.70 대비 모든 모델이 0.13+ 마진 확보.


### H6 — Dynamic A* FED ≥ 30% ↓  🔜 보류

- **path_planning 모듈 미작성** (Week 11). `src/path_planning/edge_weights.py`, `planners.py`, `evacuation_sim.py` 구현 필요.  
- 본 평가에서 검증한 것: H6 의 *전제 조건* — risk-map 의 quality (H4 + H5) 가 OOD 에서도 견고. 즉 planner 입력 신호는 신뢰 가능.  
- **OOD autoregress 6-step 의 큰 폭증 (0.88) 은 H6 설계에 중요한 제약**: Dynamic Predictive planner 의 lookahead 를 60 s 가 아닌 **20-30 s** 로 잡아야 안전. observation refresh 주기도 30 s 권장.



## 4. 결정 권고 (decisions.md 후보)

- **D-025 권고**: 주력 모델을 **ConvLSTM 으로 확정**. PI-FNO 는 ablation 후보로 페이퍼에 포함하되 main result 의 default 가 아님.
- **D-026 권고**: Dynamic risk-map 의 **lookahead 를 60 s → 30 s 단축**. OOD autoregress 누적 오차 (0.77-0.88) 가 60 s 시점에서 너무 큼.
- **D-027 권고**: 페이퍼 framing 을 *'spectral vs local surrogate trade-off in 33-scenario regime'* 로 reframe (Plan B 활용).



## 5. 잔여 작업

- [ ] **path_planning 모듈** (Week 11) — H6 직접 검증을 위한 EXP-PATH-001
- [ ] **PyBullet 통합** (Week 12) — `docs/pybullet_integration_spec.md` 참조
- [ ] **Tier 1 GNN** — 매뉴얼 Plan B 의 drop 후보, 시간 남으면
- [ ] **Member A val/ood 시뮬**: 향후 새 건물/HRR 범위 OOD 확보 시 H3 재평가
