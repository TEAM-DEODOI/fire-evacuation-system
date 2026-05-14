# Fire Evacuation Prediction System — Documentation Index

> **🎯 진입점 문서.** 새 세션에서 시작할 때 이 파일을 먼저 읽으세요.
> 모든 docs 가 여기서 시작합니다.
>
> **마지막 업데이트**: 2026-05-13

---

## 📂 문서 구조

### 0. 큰 청사진 (필독)

| 파일 | 내용 |
|---|---|
| [`00_project_overview.md`](00_project_overview.md) | 프로젝트 목적 + 대회 컨텍스트 + 시스템 비전 |
| [`10_system_architecture.md`](10_system_architecture.md) | Tier 1/Tier 2 이중 시스템 + 데이터 흐름 도식 |

### 1. 인프라 & 데이터

| 파일 | 내용 |
|---|---|
| [`20_dataset_and_simulations.md`](20_dataset_and_simulations.md) | FDS 시뮬레이션 33+13 시나리오, 데이터셋 명세 |
| [`30_sensor_infrastructure.md`](30_sensor_infrastructure.md) | D-024 v3.3 **39 sensors** + D-023 트리거 모델 |

### 2. 모델 명세

| 파일 | 내용 |
|---|---|
| [`40_tier2_models_continuous.md`](40_tier2_models_continuous.md) | ConvLSTM / FNO no-PI / FNO PI (continuous T/V/CO) |
| [`50_tier1_gnn_binary.md`](50_tier1_gnn_binary.md) | Tier 1 GNN (binary detector signal) |

### 3. 평가 & 결과

| 파일 | 내용 |
|---|---|
| [`60_evaluation_layers.md`](60_evaluation_layers.md) | **L1-L4 평가 framework** (paper 핵심) |
| [`70_results_summary.md`](70_results_summary.md) | **모든 실험 결과 종합표** |
| [`80_hypothesis_validation.md`](80_hypothesis_validation.md) | H1-H6 가설 검증 현황 |

### 4. 미래 작업

| 파일 | 내용 |
|---|---|
| [`90_next_steps.md`](90_next_steps.md) | 잔여 작업 (Path planning, PyBullet, 발표) |
| [`pybullet_integration_spec.md`](pybullet_integration_spec.md) | PyBullet Week 12 외주 명세서 |

### 5. 레퍼런스 (Reference)

| 파일 | 내용 |
|---|---|
| [`decisions.md`](decisions.md) | 모든 결정 로그 (D-001 ~ D-024 v3.3) |
| [`interface_contracts.md`](interface_contracts.md) | 모듈 간 인터페이스 시그니처 |
| [`coordinate_convention.md`](coordinate_convention.md) | 좌표계 규약 (world m, Z-up) |
| [`risk_indicators.md`](risk_indicators.md) | ISO 13571 tenability + FED |
| [`lessons_learned.md`](lessons_learned.md) | 디버깅 로그 + 함정 모음 |

### 6. Archive (옛 자료)

| 폴더 | 내용 |
|---|---|
| [`archive/old_planning/`](archive/old_planning/) | 옛 계획 문서 (handoff, session summary, 14주 매뉴얼 등) |
| [`archive/auto_reports/`](archive/auto_reports/) | 스크립트 자동 생성 보고서 (참고용) |

---

## 🚀 빠른 시작 (새 Claude 세션)

```
1. CLAUDE.md  (auto-loaded, 프로젝트 헌장)
2. docs/README.md  (이 파일)
3. docs/00_project_overview.md  (목적 + 비전)
4. docs/70_results_summary.md  (현재까지 결과)
5. docs/90_next_steps.md  (다음 작업)
```

---

## 🏗️ 시스템 한 줄 요약

> **39 화재 감지기 인프라 위에 두 가지 surrogate fire-prediction 시스템 —
> Tier 1 (binary signal → 12K GNN, IoU 0.90) + Tier 2 (continuous signal →
> ConvLSTM/FNO + 보간, IoU 0.21-0.43). 추가 hardware 없이 기존 건물에
> 즉시 배포 가능. Ideal full-observation upper bound (0.92) 의 98% 달성.**

---

## 📊 현재 상태 (2026-05-13)

| 가설 | 측정값 | 통과 |
|---|---|---|
| H1 Speed ≥ 1000× FDS | 52,000× (ConvLSTM 26.5 ms) | ✅ |
| H2 RelL2 ≤ 0.15 | 0.136 (ConvLSTM training) | ✅ |
| H3 FNO > ConvLSTM OOD | full SLCF ❌, sparse 39 ✅ FNO no-PI 우위 | ⚠ 부분 |
| H4 Risk FNR < 10% | **GNN 4.6%** | ✅ |
| H5 Risk IoU ≥ 0.70 | **GNN 0.904** (13/13 시나리오 PASS) | ✅ |
| H6 Dynamic A* FED ≥ 30%↓ | path planning 미구현 | 🔜 |

---

## 🎬 다음 단계

1. **H6 검증** — Tier 1 GNN → `Tier1RiskMap.query()` → A* path planning
2. **Sparse-input ConvLSTM 학습 결과 수집** (사용자 진행 중)
3. **PyBullet 통합** (Week 12, [`pybullet_integration_spec.md`](pybullet_integration_spec.md) 외주)
4. **발표 자료 제작** — `figures/current/04_tier1_gnn/headline.png` 가 paper Figure 1
