# 평가 Layer L1-L4 — Ideal 부터 실 Deployment 까지

> **문서 목적**: 본 프로젝트의 평가가 **4단계 layer** 로 점진적으로 현실화됨을
> 명시. 각 layer 의 결과 격차 자체가 paper 의 핵심 contribution.
> **작성일**: 2026-05-13
> **참조**: `docs/cold_start_finding.md`, `docs/detector_triggered_evaluation.md`,
> `docs/sparse_sensing_evaluation.md`, `docs/hypothesis_validation_report.md`

---

## 0. 한 줄 요약

같은 ConvLSTM 체크포인트에 대해 **평가 가정만 바꿨을 뿐인데** 60s 미래 예측의
IoU 가 **0.92 → 0.28 로 70% 폭락**. 이 격차가 우리 시스템의 진짜 challenge
이자 paper 의 메인 contribution 영역.

---

## 1. 평가 가정의 4단계

| Layer | 평가 가정 | 입력 | ConvLSTM IoU step 6 (t₀=120s) | 위치 |
|---|---|---|---|---|
| **L1** | Teacher-forced single-step | 매 step FDS truth 주입 | **0.89** | `docs/eval_T01_T05_report.md` |
| **L2** | Full SLCF autoregress (ideal) | 60×40×6 = 14,400 cell 모두 측정 가정 | **0.92** | `docs/cold_start_finding.md` |
| **L3** | Detector-triggered autoregress | full SLCF + "감지 시점부터 시작" 로직 | **0.53** | `docs/detector_triggered_evaluation.md` |
| **L4** | **16-sensor sparse + 보간** | 16 지능형 센서 → griddata 보간 → 60×40×6 grid | **0.28** | `docs/sparse_sensing_evaluation.md` |
| **L4f** | **27-sensor binary + Tier 1 GNN** | 27 detector binary on/off (D-023 trigger) → GraphGRU → per-node danger | **0.821** ✅ | `docs/sparse_retrain_evaluation.md` + `checkpoints/tier1_gnn/` |

```
   IoU step 6 (60s 미래 예측)
   1.0 ─┐
        │                    L2: full SLCF (ideal upper bound)
   0.9 ─┤   ╭─────────────╮  ──────── 0.92
        │   │     L1      │   ╭───────╮
   0.8 ─┤   │ teacher-     │   │  L2   │
        │   │ forced 0.89  │   │ 0.92  │
   0.7 ─┤   ╰─────────────╯   ╰───────╯  ─────── H5 (0.70) 임계
        │                                ╭──────╮
   0.5 ─┤                                │  L3  │ 0.53
        │                                │ trig │
   0.3 ─┤                                ╰──────╯ ╭──────╮
        │                                          │  L4  │ 0.28
   0.0 ─┘                                          │sparse│
                                                   ╰──────╯
       L1 (sandbox) → L2 (ideal) → L3 (trigger) → L4 (실 deployment)
```

---

## 2. 각 Layer 가 측정하는 것

### L1 — Teacher-forced single-step (Sandbox)
- 매 timestep `t` 에서 FDS truth `x[t]` 입력 → 모델 출력 `y[t+1]`
- 자기 출력을 다시 입력으로 chaining ❌
- → **모델의 단일 step 표현력 측정**. autoregress 불안정성 무관.
- **결론**: ConvLSTM/FNO 모두 H5 통과. 모델 자체는 화재 패턴 학습 가능.

### L2 — Full SLCF autoregress (Ideal upper bound)
- t₀ 에 한 번 FDS truth `x[t₀]` 주입 → 자기 출력 chaining 으로 60s 진행
- 입력은 14,400 cell 모두 측정 가정 (현실에 없음)
- → **autoregress 안정성 측정** (cold-start 제외 시)
- **결론**: ConvLSTM 0.92 통과. FNO no-PI 0.82, FNO PI 0.89 도 통과. *단 t₀=0 (cold-start) 은 fail*.

### L3 — Detector-triggered autoregress
- 16 detector 중 첫 trigger 시점 (T > 60°C) 부터 L2 와 동일 진행
- 입력은 여전히 full SLCF (= 이상적 가정)
- → **시스템 워크플로우 정의 (감지→예측) 의 정확도**
- **결론**:
  - 11/13 시나리오 trigger 발생, 2개 미발생 (coverage limitation)
  - Trigger 시점 = 좋은 예측 시작 시점 ❌. 빠른 trigger (t=10s) 는 약한 신호.
  - 평균 IoU 0.53 — H5 미달.
  - → "trigger + delay" 또는 "multi-detector quorum" 필요.

### L4 — 16-sensor sparse + 보간 (실 deployment)
- t₀=120s 에서 시작 (mid-fire, cold-start 회피)
- 입력: **16개 위치의 T/V/CO 만 측정** (지능형 센서 가정)
- spatial interpolation (linear/cubic/nearest) → (60, 40) 평면 복원
- z 축은 broadcast (호흡고도 측정만 가정)
- → **실제 deployment 의 성능 측정**
- **결론**:
  - 모든 모델 × 모든 보간 방법이 H5 한참 미달 (IoU 0.17-0.33)
  - 보간 자체가 oversmoothing → sharp fire-front 손실
  - 놀라운 발견: **nearest > linear/cubic** (sharp gradient 보존)
  - → 16-sensor + 단순 보간 만으로는 H5 통과 불가

---

## 3. 격차 분해 — 어디서 정확도가 손실되는가

L2 → L3 → L4 의 단계별 손실:

| 손실 단계 | 메커니즘 | IoU 손실 |
|---|---|---|
| L1 → L2 | Teacher-forced → autoregress 누적 오차 | -0.0 (놀랍게도 미미) |
| L2 → L3 | "감지 후 시작" 운용 제약 (약한 trigger 신호) | **-0.39** |
| L3 → L4 | Full obs → 16 sensor + 보간 | **-0.25** |

**가장 큰 격차는 L2→L3**: trigger 시점이 화재 진행 정도와 다르다는 점. 단순한
solution: trigger + N s wait, 또는 quorum-based trigger.

**두 번째로 큰 격차는 L3→L4**: 보간 자체의 oversmoothing. 가능한 fix:
- Mask-aware / wall-respecting interpolation (geodesic 거리 기반)
- Sensor density 증가 (16 → 32 → 60)
- **Sparse-aware end-to-end 모델 재학습 (Track 1B)**

---

## 4. Paper Headline 형성

본 발견은 페이퍼의 한 챕터 또는 메인 결과로 자연스럽게 정리됨:

> **"Surrogate fire-prediction models exhibit dramatic accuracy degradation under
> realistic sensor sparsity. We propose <method> to recover usable IoU while
> preserving the model's spectral/temporal expressivity."**

논문 결과 표 (proposed):

```
Table N: Evaluation layers and IoU at t₀+60s (T01-T05 OOD scenarios).

Layer  Assumption                                  ConvLSTM  FNO no-PI  FNO PI
─────  ──────────────────────────────────────────  ────────  ─────────  ──────
L1     Teacher-forced single-step                  0.89      0.83       0.84
L2     Full SLCF autoregress (ideal)               0.92      0.82       0.89
L3     Detector-triggered (avg)                    0.53      0.55       0.48
L4a    16 sensors + nearest interp                 0.28      0.33       0.29
L4b    16 sensors + linear interp                  0.19      0.17       0.17
L4c    16 sensors + cubic interp                   0.19      0.19       0.17
L4d    [PROPOSED] mask-aware geodesic interp       ?         ?          ?
L4e    [PROPOSED] sparse-input retrained model     ?         ?          ?
```

→ **L4d / L4e 가 우리가 다음에 채워야 할 표**.

---

## 5. 시사점 — 다음 작업 우선순위

다음 두 가지가 paper 의 메인 contribution 후보:

### 5.1 Mask-aware interpolation (학습 ❌, 가장 빠름)
- 보간을 wall-respecting 으로 (geodesic distance, 또는 building topology 인식)
- **장점**: 학습 불필요, 즉시 평가 가능. 보수적 baseline 으로도 유용.
- **단점**: 16 sensor 의 fundamental information bottleneck 못 넘음 (정보론적 한계)

### 5.2 Sparse-input end-to-end 학습 (Track 1B)
- 기존 dataset.h5 의 input 을 sparse representation 으로 변환
- 모델 재학습 — sparse signal 로부터 dense pattern 학습
- **장점**: 진짜 fundamental fix. 보간 단계 자체 제거.
- **단점**: 학습 cost (RunPod 또는 CPU 2-3시간)

### 5.3 Sensor density ablation
- 16 → 32 → 60 → all-cell 의 IoU 그래프
- **장점**: paper headline figure 가 됨 ("sensor density vs prediction accuracy")
- **단점**: 32/60 sensor 의 실 deployment 비용 분석 필요

---

## 6. 누적 진행 상황

| 작업 | 상태 | 산출물 |
|---|---|---|
| L1 평가 | ✅ | `figures/eval_T01_T05/` |
| L2 평가 | ✅ | `figures/future_prediction/` |
| L3 평가 | ✅ | `figures/detector_triggered/` |
| **L4a/b/c** | **✅ 이번 세션** | `figures/sparse_sensing/` |
| **L4d (mask-aware)** | **🔜 진행 중** | `scripts/evaluate_sparse_sensing_geodesic.py` 예정 |
| **L4e (sparse-input retrain)** | **🔜 진행 중** | `scripts/build_sparse_dataset.py` + retrain |
| Sensor density ablation | ⬜ 다음 세션 | — |
| GNN (Track 2, binary deployment) | ⬜ 별개 트랙 | `src/tier1/` (80%) |
| Path planning + H6 검증 | ⬜ 차기 세션 | — |

---

> *본 문서는 사용자 (2026-05-13 세션) 의 지적 — "지금까지 평가가 ideal sensor
> 가정 위에 있었고, 실 deployment 검증이 빠져있다" — 을 명시적 framing 으로 정리.*
