# 세션 종합 — 2026-05-13

> **세션 시작**: handoff_2026_05_12.md 읽고 시작
> **세션 종료**: Track 1B 부분 학습 + L1-L4 evaluation layer 정립
> **commit 수**: 8 개
> **핵심 산출물**: 평가 layer 정의 + sparse sensor 검증 + paper headline figure

---

## 0. 세션 한 줄 요약

처음에는 "FNO no-PI/PI 검증 + GNN + 가설 검증" 으로 시작했지만, 세션 중간에
사용자의 **근본적 재정의** 가 들어와 프로젝트의 진짜 evaluation framework
(L1-L4 layered evaluation) 가 정립됨. **paper headline figure 완성**: 동일
ConvLSTM 이 평가 가정 변경만으로 IoU 0.92 → 0.20 까지 변동.

---

## 1. 세션 흐름 — 8 commit 진화

| Commit | 작업 | 발견 |
|---|---|---|
| 43bb07a | ConvLSTM OOD 평가 (T01-T05, 13 시나리오) | IoU 0.89 (teacher-forced) |
| 3d1fba1 | FNO no-PI/PI 평가 + H1-H6 검증 | **H3 ❌** (FNO 가 ConvLSTM 못 이김), Plan B 트리거 |
| d84d9ac | 진짜 60s autoregress 시각화 | **Cold-start (t₀=0) vs Mid-fire (t₀=120s)** 두 regime 발견 |
| (commit) | C1+D1 — detector-triggered eval | Trigger 시점 ≠ 좋은 예측 시작 시점 |
| (commit) | Track 1A — 16-sensor sparse + 보간 | **IoU 0.92 → 0.20 — 5배 폭락** |
| (commit) | Track 1A.5 — mask-aware geodesic IDW | **IoU 0.20 → 0.41 — 2배 회복** |
| (commit) | Track 1B 스크립트 + 5-epoch smoke 학습 | IoU 0.20 (재학습 잠재력 검증) |
| (commit) | 본 보고서 + 세션 마무리 | 다음 세션 plan |

---

## 2. 핵심 발견 — Evaluation Layer L1-L4

같은 ConvLSTM 체크포인트에 **평가 가정만 변경** 했을 때 IoU 변화:

| Layer | 평가 가정 | ConvLSTM IoU step 6 | 의미 |
|---|---|---|---|
| **L1** | Teacher-forced single-step | **0.89** | sandbox upper bound |
| **L2** | Full SLCF autoregress (ideal) | **0.92** | 14400 cell 측정 가정 — 비현실 |
| **L3** | Detector-triggered autoregress | **0.53** | trigger 시점 효과만 |
| **L4a** | 16 sensors + nearest | **0.28** | naive sparse |
| **L4b** | 16 sensors + linear | **0.19** | over-smoothing |
| **L4c** | 16 sensors + cubic | **0.19** | over-smoothing |
| **L4d** | 16 sensors + geodesic IDW (mask-aware) | **0.41** | wall-respecting 보간 |
| **L4e** | 16 sensors + sparse-input retrain (5-epoch preliminary) | **0.20** | full 학습 후 잠재력 미확정 |

이 단조 감소 curve **자체가 paper 의 메인 contribution 영역**.

`figures/sparse_retrain_smoke/full_stack_comparison.png` 가 headline figure
의 골격.

---

## 3. 시스템 비전 재정의 (사용자 입력)

세션 중반에 사용자가 명확히 한 시스템 비전:

### Track 1 (High-end Building): 지능형 센서 건물
- 각 방/복도 특정 지점에 **T + CO 감지기**
- Sparse measurements (~16-19 points) → spatial interpolation 또는 직접 sparse-aware 모델
- 본 세션의 Track 1A/1A.5/1B 가 여기 해당

### Track 2 (Low-end Building): 기존 화재 감지기
- 각 방에 **binary on/off 감지기만** (60°C threshold)
- Tier 1 GNN 으로 binary signal → node-level danger → path planning
- **별개 deployment context**, 본 세션에서는 보류

이전에 내가 GNN 을 "cold-start 해결책" 으로 자동 연결했던 건 잘못된 가정.
GNN 의 진짜 역할은 *legacy infrastructure 호환성*.

---

## 4. 가설 검증 현황

| ID | 가설 | L2 (ideal) | L4d (geodesic 16sens) | 결론 |
|---|---|---|---|---|
| **H1** | Speed ≥ 1000× FDS | ✅ 52000× | ✅ (모델 동일) | PASS |
| **H2** | RelL2 ≤ 0.15 | ✅ 0.136 | — (계산 안 함) | PASS |
| **H3** | FNO > ConvLSTM on OOD | ❌ | ❌ | Plan B 트리거 |
| **H4** | Risk FNR < 10% | ✅ 6.0% | ⚠ ~60%+ | 운용환경 따라 |
| **H5** | Risk IoU ≥ 0.70 | ✅ 0.92 | ❌ 0.41 | **운용환경 따라** |
| **H6** | Dynamic A* FED ≥ 30% ↓ | 🔜 | 🔜 | path_planning 필요 |

**핵심 nuance**: H4/H5 의 통과 여부가 평가 layer 에 따라 다름. *Ideal 에서는 통과,
실 deployment 에서는 미달*. paper 에서 이 distinction 을 명확히 해야 함.

---

## 5. 산출물 인덱스 (오늘 생성)

### 스크립트
- `scripts/evaluate_t_locations.py` (확장: `--model-type` 추가, FNO 지원)
- `scripts/hypothesis_validation.py`
- `scripts/visualize_60s_prediction.py`
- `scripts/evaluate_detector_triggered.py`
- `scripts/evaluate_sparse_sensing.py`
- `scripts/evaluate_sparse_sensing_geodesic.py`
- `scripts/train_sparse_conv_lstm.py` (Track 1B)
- `scripts/evaluate_sparse_model.py` (Track 1B)

### 보고서 (docs/)
- `docs/eval_T01_T05_report.md` (ConvLSTM OOD)
- `docs/eval_T01_T05_fno_no_pi_report.md`
- `docs/eval_T01_T05_fno_pi_report.md`
- `docs/hypothesis_validation_report.md`
- `docs/cold_start_finding.md`
- `docs/detector_triggered_evaluation.md`
- `docs/sparse_sensing_evaluation.md` (Track 1A 자동 생성)
- `docs/sparse_sensing_geodesic_evaluation.md`
- **`docs/evaluation_layers.md` (L1-L4 framing — 핵심)**
- `docs/sparse_retrain_smoke_eval.md` (Track 1B preliminary)
- `docs/sparse_retrain_evaluation.md` (Track 1B partial, 신뢰도 낮음)
- `docs/pybullet_integration_spec.md` (이전 세션부터)

### 결과 CSV (results/)
- `results/eval_T01_T05/`, `eval_T01_T05_fno_no_pi/`, `eval_T01_T05_fno_pi/`
- `results/exp_fire_001/comparison.csv`
- `results/exp_detector_triggered/comparison.csv`
- `results/exp_sparse_sensing/comparison.csv`
- `results/exp_sparse_sensing_geodesic/comparison.csv`
- `results/exp_sparse_retrain_smoke/comparison.csv`

### 핵심 Figures
- `figures/eval_T01_T05/aggregated_boxplots.png` (L1)
- `figures/future_prediction/sim_*/grid_t0_*.png` (L2, cold-start/mid-fire)
- `figures/detector_triggered/iou_per_scenario.png` (L3)
- `figures/sparse_sensing/method_comparison.png` (L4a-c)
- `figures/sparse_sensing_geodesic/{method_comparison_geodesic,snapshot_T05_geodesic}.png` (L4d) ★
- **`figures/sparse_retrain_smoke/full_stack_comparison.png` (paper headline 후보)** ★★

### 체크포인트 (git 화이트리스트)
- `checkpoints/conv_lstm/best.pt` (1.4 MB)
- `checkpoints/fno_no_pi/best.pt` (41 MB)
- `checkpoints/fno_pi/best.pt` (41 MB)
- `checkpoints/conv_lstm_sparse_smoke/best.pt` (1.4 MB, 5-epoch)
- `checkpoints/conv_lstm_sparse/best.pt` (1.4 MB, ~6-epoch partial — unstable)

---

## 6. 다음 세션 — 명확한 plan

### 6.1 즉시 작업 (Track 1B 완료)

```bash
# Full 50-100 epoch 학습 (CPU 50-100분 또는 RunPod 15-30분)
python scripts/train_sparse_conv_lstm.py \
    --epochs 50 \
    --batch-size 4 \
    --init-from checkpoints/conv_lstm/best.pt \
    --output checkpoints/conv_lstm_sparse

# 평가
python scripts/evaluate_sparse_model.py \
    --ckpt checkpoints/conv_lstm_sparse/best.pt
```

### 6.2 후속 — Sensor density ablation
- 8 / 16 / 32 / 60 / all 센서 비교
- "sensor density vs IoU" 곡선 → paper figure 2

### 6.3 Track 2 — GNN (별개 트랙)
- `tier1_dataset.py` 작성
- `train_tier1.train()` 구현
- `torch-geometric-temporal` 의존성 설치 (Python 3.12 호환 확인)
- binary detector 만으로 학습 + 평가
- **target deployment**: 기존 화재 감지기만 있는 building

### 6.4 H6 검증 — Path planning
- `src/path_planning/` 모듈 작성 (3 planners, evacuation_sim)
- EXP-PATH-001 — Dynamic A* vs Static FED reduction

### 6.5 PyBullet 통합 (Week 12)
- `docs/pybullet_integration_spec.md` 의 spec 사용
- 외주 또는 자체 진행

---

## 7. 사용자 의도 변화 요약

이 세션은 **사용자가 작업을 끌어준 세션**. 초기 의도("FNO+GNN+가설검증") 가
중간에 두 번 재정의됨:

1. **재정의 1**: "60초 미래 예측 진짜 검증 부족" → autoregress 시각화 →
   cold-start finding
2. **재정의 2**: "지능형 센서 가정, sparse → interpolation → 모델" 이
   진짜 시스템 비전 → Track 1A/1A.5/1B 트랙 정립

이런 재정의가 paper 의 contribution 을 fundamentally 더 강하게 만들었음.
초기 "FNO 우위 입증" (실패한 가설) 에서 "sparse sensor regime trade-off"
(robust 한 발견) 로 reframe.

---

## 8. Plan B 정식 활성화

**원 매뉴얼 Plan B (CLAUDE.md L391)**:
> *"PI-FNO doesn't beat ConvLSTM → 페이퍼를 '30-scenario regime trade-offs' 로 reframe"*

→ ✅ 활성. 단 reframe 의 정확한 방향은 *evaluation layer 의 sparse-deployment
gap* 으로. 더 강한 contribution.

새 페이퍼 outline (제안):
1. Intro: fire-evac surrogate models 의 promise + gap
2. Method: ConvLSTM/FNO/PI-FNO + tenability + path planning + **3 sensor regimes (full / sparse intelligent / binary GNN)**
3. EXP-FIRE-001: model comparison on full SLCF (H1, H2, H3)
4. **EXP-SPARSE-001 (NEW)**: L1-L4 evaluation layer gap quantification
5. EXP-RISK-001: risk map fidelity (H4, H5) per layer
6. EXP-PATH-001: dynamic A* FED reduction (H6)
7. Discussion: 어떤 deployment context 에서 어떤 모델/방법이 유효한가
8. Limitations: coverage (detector-out-of-range), cold-start, regime gaps

---

> *세션은 사용자의 두 번의 정확한 재정의로 본 프로젝트의 진짜 contribution 영역을 발견. Plan B 가 paper 를 더 강하게 만들어줌.*

---

## 9. 세션 끝 보강 — Tier 1/2 통합 framing + D-024 27 감지기 인프라

세션 마무리 직후 사용자가 3 개의 외부 계획 문서 전달:
- `docs/tier1_detector_plan.md` — Tier 1 GNN 의 감지기 모델 통합 계획
- `docs/tier1_detector_task.md` — D-023 트리거 모델 (다음 세션 task)
- `docs/tier1_detector_positions_task.md` — D-024 감지기 위치 (이번 세션 구현)

### 9.1 핵심 명확화 — Tier 1 / Tier 2 공유 인프라

사용자: *"여기서 정의한 센서의 위치는 tier 2 에서도 그대로 쓸거야"*

이전 세션 framing 정정:
- **이전**: Track 1 (intelligent) vs Track 2 (legacy) = 별개 deployment
- **수정**: **같은 27개 감지기 인프라, 두 가지 신호 모드**
  - Tier 1 (GNN): binary on/off (D-023 트리거)
  - Tier 2 (ConvLSTM/FNO): continuous T/V/CO

### 9.2 D-024 구현 (본 세션 추가 작업)

- `src/tier1/detector_positions.py` 작성 (27개 감지기)
- Self-test 9개 모두 PASS
- 영역 분포: Zone A=3, B=5, C=4, D=5, 복도=7, 출구=3
- 모두 z=2.5m, NFPA 72 spacing max 7m
- `docs/decisions.md` 에 D-024 추가
- 문서 vs 구현 불일치 해결: task §1 "28개" 는 typo, **실제 27개** (Zone 표 정확)

### 9.3 본 세션 sparse 평가의 재해석

지금까지의 sparse 평가는 **16개 임시 위치** 기반. **다음 세션에서 D-024 27개로 재평가 필요**:

| Layer | 16-sensor 결과 | 27-sensor 예상 |
|---|---|---|
| L4d (geodesic) | 0.41 | **↑ 예상** (sensor 11개 추가, 복도/출구 cover) |
| L4e (sparse retrain) | 0.20 (5ep) | **↑ 예상** + full 학습 |

### 9.4 다음 세션 plan — 업데이트

1. **D-023 구현** (`src/tier1/detector_model.py` + `scripts/visualize_detectors.py`) — `docs/tier1_detector_task.md` 명세
2. **first_sim + T01-T05 에 D-023 적용** — binary_sequence.npy 생성
3. **27-sensor sparse 재평가** — Track 1A/1A.5/1B 재실행
4. **Tier 1 GNN 학습** — `src/tier1/dataset.py` + `train_tier1.train()` 구현
5. **Tier 1 vs Tier 2 비교** — paper headline 표
6. **Path planning + H6** — `src/path_planning/` + EXP-PATH-001

### 9.5 갱신된 Paper Framing

> **"단일 건물 인프라 (27개 표준 화재 감지기) 위에 두 가지 surrogate
> fire-prediction 시스템 제안 — Tier 1 (binary signal → graph network,
> legacy 호환) + Tier 2 (continuous signal → spatial neural operator,
> intelligent sensors). 두 시스템 모두 NFPA 72 / UL 268 / KOFEIS 0301
> 표준의 보수적 임계값 사용, 추가 하드웨어 없이 기존 건물에 즉시 적용
> 가능. Evaluation layers L1→L4 가 'ideal upper bound vs realistic
> deployment' 격차를 정량화."**

이 framing 이 H1/H2/H4/H5 통과 + H3 실패 (Plan B) + cold-start +
sparse gap 발견 + 27-sensor 통합 인프라 모두를 하나로 묶음.
