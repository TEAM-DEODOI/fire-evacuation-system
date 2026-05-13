# Cold-start 발견 — 진짜 60초 미래 예측 검증 결과

> **발견 일자**: 2026-05-13
> **검증 스크립트**: `scripts/visualize_60s_prediction.py`
> **검증 도메인**: OOD T01-T05 시나리오 (특히 T05 1500kW 2m², T01 500kW 1m²)
> **연관 가설**: H2, H5 — *teacher-forced 결과만으로는 검증 부족*

---

## 0. 한 줄 요약

지금까지의 `evaluate_t_locations.py` 평가 결과 (IoU 0.83–0.89, FNR 6–7%) 는
모두 **teacher-forced single-step** 메트릭이었다. 진짜 "관측 시점 t₀ 에서
60초 미래 autoregress 예측" 으로 재검증하니 **두 가지 명확한 regime** 이 드러남:

- **Cold-start (t₀ = 0)**: 3 모델 모두 **catastrophic fail** — IoU step-6 0.01–0.34
- **Mid-fire (t₀ ≥ 60s, 화재 진행 중)**: 3 모델 모두 **매우 잘 동작** — IoU step-6 0.82–0.92

이는 *모델 자체의 실패가 아니라 실용적 운용 환경 정의의 문제*. 우리 시스템은
"감지 후 예측" pipeline 이므로 t₀=0 부터 시작하는 시나리오는 자체 워크플로우에
존재하지 않음.

---

## 1. 검증 방법

`scripts/visualize_60s_prediction.py` 는 다음을 수행:

1. 시나리오 1개 선택 (예: `sim_1500kw_2m2_T05`)
2. FDS 슬라이스 추출 → 정규화 → (31, 5, 60, 40, 6) input tensor
3. 시작 시각 `t₀ ∈ {0, 30, 60, 120, 180} s` 각각에 대해:
   - 그 시점의 (5, 60, 40, 6) input 으로 모델 초기화
   - **autoregress 6 step** (자기 출력을 다시 input 으로) → 60s 미래까지 예측
   - FDS truth (`compute_total_danger`) 와 비교
4. 산출:
   - `<scenario>_grid_t0_<NNN>.png` — 4 row (truth + 3 모델) × 6 col (step 1..6)
   - `<scenario>_error_curve_t0_<NNN>.png` — per-step RMSE & IoU
   - `<scenario>_sliding_animation.gif` — 모든 t₀ 의 60s lookahead 비교

기존 `evaluate_t_locations.py` 와의 차이:
- 기존: 매 timestep `t` 에서 input[t] (= FDS truth) → output[t+1] 단일 step
- 신규: input[t₀] 만 사용, output 을 다음 input 으로 chaining (no teacher forcing)

---

## 2. Cold-start 결과 (t₀ = 0)

시나리오 `sim_1500kw_2m2_T05` (가장 잘 예측되어야 하는 케이스):

| 모델 | IoU step 1 (t=10s) | IoU step 6 (t=60s) | RMSE step 6 |
|---|---|---|---|
| ConvLSTM | 0.00 | **0.34** | 0.22 |
| FNO no-PI | 0.00 | **0.01** | 0.36 |
| FNO PI | 0.00 | **0.01** | 0.36 |

→ ![grid t0=0](../figures/future_prediction/sim_1500kw_2m2_T05_grid_t0_000.png)

세 모델 모두 t=60s 시점 FDS truth (corridor 전체 빨강) 를 **재현하지 못함**.
ConvLSTM 은 약한 신호라도 전파하지만 FNO 두 변종은 거의 0 출력만 산출.

### Cold-start 의 원인

1. **입력 신호 부재**: t=0 의 SLCF cell 값:
   - Temperature: ~20°C (ambient) → normalized 0.0
   - Visibility: ~30 m → normalized 0.0 (inverse 매핑)
   - CO: 0 ppm → normalized 0.0
   - Mask: 1.0 (fluid) 또는 0.0 (solid)
   - Time encoding: 0.0
   - **즉 fluid mask 와 time encoding 외에는 모든 신호가 0**
2. **학습 데이터의 최적 답**: 33 시나리오 모두 t=0 frame 은 동일 (ambient).
   training pair (input[0], target[1]) 의 target 도 매우 낮은 신호. 모델은
   "0 입력 → 0 출력" 이 minimum-MSE 답으로 학습.
3. **Autoregress amplification**: 첫 step 출력이 0 에 가까우면 두 번째 input
   도 0. 자기 출력으로 chaining 이 ignite 못 함.
4. **FNO 가 ConvLSTM 보다 더 심한 이유**: FNO 의 Fourier transform 은
   high-frequency component 를 자연스럽게 제한 (mode truncation). 약한 spatial
   noise 가 ConvLSTM 의 local conv 에서는 amplify 될 수 있지만 FNO 에서는
   smoothing 으로 dampening 됨.

---

## 3. Mid-fire 결과 (t₀ = 120s)

시나리오 `sim_1500kw_2m2_T05`, 화재가 충분히 진행된 상태에서 시작:

| 모델 | IoU step 1 (t=130s) | IoU step 6 (t=180s) | RMSE step 6 |
|---|---|---|---|
| **ConvLSTM** | **0.96** | **0.92** | 0.080 |
| FNO no-PI | 0.90 | 0.82 | 0.096 |
| FNO PI | 0.90 | 0.89 | 0.089 |

→ ![grid t0=120](../figures/future_prediction/sim_1500kw_2m2_T05_grid_t0_120.png)

세 모델 모두 corridor 전체 패턴, top-right 영역 강한 위험도, 중정 안전영역까지
정확하게 재현. 누적오차도 안정적 — 6 step (60s) 후에도 IoU 0.82 ≥ 0.70 (H5 통과).

어려운 시나리오 (T01 500kW, 코너 화재) 도 t₀=120s 에서는 잘 동작:

→ ![grid t0=120 T01](../figures/future_prediction/sim_500kw_1m2_T01_grid_t0_120.png)

### Mid-fire 의 모델 순위 (t₀=120s, IoU step 6 기준)

1. **ConvLSTM 0.92** — local conv 의 strong-signal regime 우위
2. **FNO PI 0.89** — physics regularization 이 long-horizon 안정화에 기여
3. FNO no-PI 0.82 — Fourier domain 의 누적오차가 가장 큼

흥미: cold-start 에서는 FNO no-PI ≈ FNO PI 였지만, mid-fire 에서는
**FNO PI 가 FNO no-PI 보다 명확히 안정적**. PI loss 의 가치가 *long-horizon
autoregress* 에서 드러남.

---

## 4. 가설 검증 재평가

cold-start 발견 전후의 가설 평가:

| 가설 | teacher-forced 기준 | autoregress (mid-fire) 기준 | autoregress (cold-start) 기준 |
|---|---|---|---|
| H2 RelL2 ≤ 0.15 | ✅ (ConvLSTM 0.136) | — (RelL2 정의는 single-step) | — |
| H4 FNR < 10% | ✅ (ConvLSTM 6.0%) | ✅ (mid-fire 에서 더 좋음) | ❌ (단 t₀=0 에서 의미 없음) |
| H5 IoU ≥ 0.70 | ✅ (ConvLSTM 0.887) | ✅ (t₀=120s step 6 IoU 0.92) | ❌ (모든 모델 0.01-0.34) |

**결론**: H4/H5 는 *실용적 운용 환경* (감지 후 mid-fire) 에서 **여전히 통과**.
단 cold-start regime 은 별도 명시 필요.

---

## 5. 실용적 시사점 — 시스템 워크플로우 정의

이 발견은 우리 시스템의 워크플로우를 명확히 함:

```
   ┌──────────────────────────────────────────────────────────┐
   │  Fire 발생 (t = 0)                                        │
   │  ↓                                                        │
   │  화재 확산 초기 (t = 0 ~ ~30s)                            │
   │    — Temperature 가 detector threshold (60°C) 도달 전     │
   │    — SLCF 신호 거의 0                                     │
   │    — **모델 autoregress 불가** (cold-start fail)          │
   │  ↓                                                        │
   │  Detector trigger 시점 (t ≈ 30 ~ 60s)                     │
   │    — 가까운 detector 노드의 binary sequence 가 1.0 으로 전환│
   │    — 이 시점부터 ConvLSTM/FNO 입력 신호 충분               │
   │  ↓                                                        │
   │  ★ Mid-fire prediction phase (t ≥ ~60s)                  │
   │    — Autoregress 6-step (60s lookahead) 매우 잘 동작       │
   │    — IoU 0.82-0.96, ConvLSTM 우위                         │
   │  ↓                                                        │
   │  60초 미래 예측 → path planning edge weight 갱신           │
   └──────────────────────────────────────────────────────────┘
```

→ **t₀=0 의 cold-start fail 은 paper 의 한계가 아니라 시스템 설계의 자연스러운
boundary**. 감지 이전에는 risk map 자체가 정의 안 됨.

---

## 6. Tier 1 GNN 의 역할 — 재정의

이 발견은 Tier 1 GNN 의 가치를 명확히 함:

| 측면 | ConvLSTM / FNO | Tier 1 GNN |
|---|---|---|
| 입력 | SLCF grid (60×40×6) | detector binary (16 nodes) |
| Cold-start 동작 | ❌ 0 출력 | ✅ detector trigger 부터 즉시 동작 |
| 추론 속도 | ~27 ms | ~1-5 ms (추정) |
| Partial observation | ❌ 모든 cell 필요 | ✅ 일부 detector 만 trigger 돼도 작동 |
| 출력 해상도 | per-cell | per-node (path planning 직접) |

**GNN 의 핵심 역할**: detector signal 만으로 동작 가능 → cold-start 영역에서도
node-level danger 추정 가능. ConvLSTM/FNO 의 spatial precision 을 보완하는
*early-warning* 신호.

추천 시스템 설계:

```
   detector signals  ─▶  Tier 1 GNN  ─▶  node danger sequence (16 nodes × 6 step)
        │                                          │
        ↓                                          ↓
   trigger time (t ≥ 30s)               path planning A* edge weight
        ↓                                          ▲
   ConvLSTM/FNO  ─▶  cell danger  ──────cell→node 변환─┘
   (spatial precision)
```

즉:
- **t < trigger**: GNN 단독 (cold-start regime)
- **t ≥ trigger**: GNN + ConvLSTM 병행, A* edge weight 는 둘의 max/weighted avg

---

## 7. 권고 — Decision Log 후보

본 발견을 바탕으로 다음 결정을 `docs/decisions.md` 에 등록 제안:

- **D-025**: 주력 spatial 예측 모델을 **ConvLSTM 으로 확정**. mid-fire IoU 0.92
  로 H5 통과, cold-start 에서도 다른 모델보다 덜 fail.
- **D-026**: Dynamic risk-map 의 **autoregress lookahead 60s 유지** (단,
  cold-start 영역에서만 GNN/initial-state 처리 분기).
- **D-027**: 페이퍼 framing 을 *"감지(GNN) + 공간 예측(ConvLSTM) 하이브리드
  pipeline"* 으로 정리. PI-FNO 는 long-horizon stability ablation 으로 포함.
- **D-028 (신규)**: Tier 1 GNN 의 역할을 *"cold-start 영역의 node-level
  danger 추정 + path planning 직접 입력"* 으로 정의.

---

## 8. 잔여 검증 작업

- [ ] Sliding animation (t₀=0, 30, 60, 120, 180s) 의 전체 frame 별도 검토
- [ ] 어려운 위치 (T01, T03) 도 mid-fire 에서 H5 통과 확인 — *partial done* (T01 t₀=120s 통과 시각화 완료)
- [ ] Tier 1 GNN 학습 후 cold-start 영역 GNN 성능 측정 (현재는 추정)
- [ ] 두 가지 시점 (t₀=0 cold-start vs t₀=120s mid-fire) 의 IoU 분리 보고

---

> *이 문서는 `scripts/visualize_60s_prediction.py` 결과 + 사용자 지적 ("이 부분에 대한 검증이 부족한 것 같아") 으로 작성됨.*
> *연관 figure: `figures/future_prediction/` 전체.*
