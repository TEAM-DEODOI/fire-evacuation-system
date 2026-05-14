# 00 — 프로젝트 청사진

> **목적**: 이 프로젝트의 *why*, *what*, *how* 를 한 문서로. 새 세션에서
> 시작할 때 이 문서 + `README.md` 만으로 전체 그림 파악 가능.

---

## 1. 대회 컨텍스트

**14주 학부 캡스톤 — 능동형 화재대피 시스템 (Fire-Evacuation Response System)**

### 1.1 우리가 만드는 것

화재 발생 → **미래 위험 지도 예측 (60s lookahead)** → 동적 경로 안내.

**기존 시스템과의 차이**:
- 기존: 화재 감지 후 *정적* 대피 경로 (사전 정의)
- 우리: 화재 *확산 예측* 으로 *동적* 경로 재계산

### 1.2 deployment 가정

- **단일층** 미로형 건물 (Science Hall 평면도, 30 m × 20 m, 단일 화재 위치)
- **27 detectors**: 기존 건물의 표준 화재 감지기 (NFPA 72 / KOFEIS 0301 / UL 268 표준)
- **선택적 업그레이드**: 지능형 T/V/CO 센서 (Tier 2)
- 모든 산출물은 **추가 hardware 없이 즉시 배포 가능** 해야 함

---

## 2. 시스템 비전 — Two-Tier 구조

**같은 39 화재 감지기 인프라**에 두 가지 신호 모드:

```
       39 화재 감지기 (D-024 v3.3 위치 확정)
       z = 2.5 m 천장, NFPA 72 spacing 준수
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
   Tier 1 (Legacy)              Tier 2 (Intelligent)
   ┌──────────────────┐        ┌──────────────────┐
   │ Binary on/off    │        │ Continuous       │
   │ (D-023 trigger:  │        │  T, V, CO        │
   │  T>60°C OR       │        │  측정값          │
   │  vis<10m,        │        │                  │
   │  latched)        │        │                  │
   │                  │        │                  │
   │ → GraphGRU       │        │ → 벽 인식 보간   │
   │   (12K params)   │        │ → ConvLSTM/FNO   │
   │                  │        │   (350K-1.78M)   │
   │                  │        │                  │
   │ → per-node       │        │ → per-cell       │
   │   danger (39 nodes)       │   danger         │
   │   x 60s seq      │        │   (60x40x6)      │
   └──────────────────┘        └──────────────────┘
        │                             │
        └──────────────┬──────────────┘
                       ▼
              Risk Map → Path Planning (A*)
                       │
                       ▼
                대피 경로 안내
```

### 2.1 Tier 1 vs Tier 2 비교

| 측면 | Tier 1 (Binary GNN) | Tier 2 (Continuous + ConvLSTM/FNO) |
|---|---|---|
| 입력 | 39 nodes × 6 frame × binary | 60×40×6 grid (sparse cell 1.6%) |
| 출력 | (39, 6) per-node danger | (3, 60, 40, 6) per-cell |
| Params | **12K** | 350K (ConvLSTM) / 1.78M (FNO) |
| 추론 시간 | < 1 ms 예상 | ~ 27-30 ms |
| **IoU @ +60s** | **0.904** ✅ | 0.21-0.43 (보간/모델 따라) |
| 추가 hardware | **0 (기존 감지기)** | T/V/CO 측정 센서 설치 필요 |
| 사용 범위 | 모든 건물 (legacy 호환) | 신축/지능형 센서 건물 |

→ **Tier 1 이 Tier 2 보다 2.1배 정확 + 30배 가벼움**. Paper headline.

---

## 3. 6대 가설 (Hypotheses)

모든 작업의 정량적 검증 기준.

| ID | 가설 | 측정 | 현재 |
|---|---|---|---|
| **H1** | Speed ≥ 1000× FDS | < 50 ms | ✅ 52,000× (ConvLSTM 26.5 ms vs FDS 23 min) |
| **H2** | RelL2 ≤ 15% | training 시나리오 | ✅ 0.136 (ConvLSTM) |
| **H3** | FNO > ConvLSTM on OOD | EXP-FIRE-001 OOD | ⚠ full SLCF ❌, sparse 39 ✅ |
| **H4** | Risk FNR < 10% | OOD | ✅ **4.6%** (Tier 1 GNN) |
| **H5** | Risk IoU ≥ 0.70 | OOD | ✅ **0.904** (Tier 1 GNN) |
| **H6** | Dynamic A* FED ≥ 30% ↓ | EXP-PATH-001 | 🔜 path planning 미구현 |

**발표 강조 순서**: H1 → H6 → H4 → H2 → H5 → H3.

상세는 [`80_hypothesis_validation.md`](80_hypothesis_validation.md) 참조.

---

## 4. 핵심 발견 (Findings) — Paper Contributions

### C1. Evaluation Layer Framework (L1-L4)

평가 가정만 바꿔도 IoU 0.20 ~ 0.92 격차 발생.

| Layer | 가정 | ConvLSTM IoU |
|---|---|---|
| L1 | Teacher-forced single-step | 0.89 |
| L2 | Full SLCF autoregress (ideal upper bound) | **0.92** |
| L3 | Detector-triggered autoregress | 0.53 |
| L4 | Sparse + interp + 모델 (현실) | **0.21–0.43** |

→ 이상-현실 격차 = paper 의 핵심 contribution. [`60_evaluation_layers.md`](60_evaluation_layers.md) 참조.

### C2. Binary GNN > Continuous + 보간

같은 39 sensor 인프라에서:
- Tier 2 best (FNO no-PI + geodesic IDW): **IoU 0.431**
- **Tier 1 GNN (binary signal)**: **IoU 0.904** ← **2.1× 우위**

→ 화재 감지의 phase-transition 특성에서 binary 정보 손실 < 보간 정보 손실.

### C3. Mask-aware Geodesic Interpolation

단순 Euclidean 보간 (`scipy.griddata linear`) 대비 BFS-geodesic IDW (벽 인식)
가 sparse regime 에서 IoU +0.20 회복:
- Linear + ConvLSTM (16 sensor): 0.19
- **Geodesic + ConvLSTM**: 0.41

→ 벽을 무시한 보간이 정보를 다른 방으로 leak. 벽 인식 보간 = 자연스러운 fix.

### C4. Legacy Infrastructure Latent Value

추가 hardware 없이 IoU 0.90 도달 — ideal upper bound (0.92) 의 **98%**.

→ "기존 건물에 즉시 배포 가능" 한 실용적 시스템.

---

## 5. 전체 워크플로 — 데이터 흐름

```
[FDS 시뮬레이션]
  data/raw/s_000~s_032/ (33 training)
  data/raw/sim_*_T01..T05/ (13 OOD)
                │
                ▼
[데이터 추출 + 정규화]
  src/data_pipeline/extract_slices()  → (31, 60, 40, 6) T/V/CO
  src/data_pipeline/normalize()       → [0, 1] 정규화
                │
                ▼
[Tier 분기]
   ┌────────────────────────┬────────────────────────┐
   │                        │                        │
   ▼                        ▼                        ▼
[Tier 1 입력]           [Tier 2 입력 A]         [Tier 2 입력 B]
extract_detector_events  sparse: 39 cell 만      sparse → interp
(D-023 trigger)         non-zero, 나머지 0       (linear/cubic/
                                                  geodesic IDW)
   │                        │                        │
   ▼                        ▼                        ▼
[Tier 1 모델]           [Tier 2 모델 A]         [Tier 2 모델 B]
SimpleFireGNN           Sparse-input ConvLSTM   ConvLSTM/FNO
(k-NN graph + GRU       (재학습)                (기존 학습)
 + graph propagation)
                                                ※ FNO no-PI 가 best
                                                  on 39 sensor
   │                        │                        │
   ▼                        ▼                        ▼
[Tier 1 출력]           [Tier 2 출력]          [Tier 2 출력]
(39, T_out=6)           (3, 60, 40, 6)         (3, 60, 40, 6)
node-level danger        autoregress 6 step    autoregress 6 step
                        → 60s 미래             → 60s 미래
   │                        │                        │
   └────────────────────────┴────────────────────────┘
                            │
                            ▼
              [Risk Map → A* Path Planning]
                            │
                            ▼
                     동적 대피 경로
```

상세는 [`10_system_architecture.md`](10_system_architecture.md) 참조.

---

## 6. 핵심 디렉터리 인덱스

| 폴더 | 내용 |
|---|---|
| `src/shared/` | constants, coordinates, normalization, building graph |
| `src/data_pipeline/` | FDS extractor, mask generator, normalize, build_dataset |
| `src/risk_map/` | tenability, FED, ASET, StaticRiskMap, converter |
| `src/models/` | conv_lstm_3d, fno_model, pi_losses |
| `src/tier1/` | **detector_positions, detector_model, tier1_gnn, tier1_dataset** |
| `src/training/` | train_conv_lstm, train_fno, trainer, wandb_utils |
| `scripts/` | 평가/시각화/학습 entry points (15+ 스크립트) |
| `configs/` | YAML hyperparams (conv_lstm, pi_fno, tier1_gnn, building, risk_map) |
| `checkpoints/` | 학습된 모델 (`.gitignore` 화이트리스트로 4개 best.pt 추적) |
| `data/raw/` | FDS 출력 (46 시나리오, gitignored, 5.7 GB) |
| `data/processed/` | dataset.h5 (gitignored, 221 MB) |
| `figures/current/` | **paper-grade figures** (sensor layout, GNN, sparse, etc.) |
| `figures/legacy/` | 옛 figure (참고용) |
| `results/` | 실험별 CSV/JSON 결과 |

---

## 7. 다음 작업 우선순위

1. **★★★ H6 검증** — Tier 1 GNN 의 node danger 출력 → `Tier1RiskMap.query()`
   → 가중 A* edge weight → EXP-PATH-001 → Dynamic FED ≥ 30% ↓ 검증
2. **★★ Tier 1+2 ensemble** — drone position 임의 query 시 두 출력 결합 (Tier 1 가중 + Tier 2 보강)
3. **★ Sparse ConvLSTM 결과 정리** — 사용자 학습 결과 도착 시 L4e 갱신
4. **★ PyBullet 통합** — `pybullet_integration_spec.md` 외주 spec 활용
5. 발표 자료 + 페이퍼 draft (`figures/current/04_tier1_gnn/headline.png` = Figure 1)

상세는 [`90_next_steps.md`](90_next_steps.md) 참조.
