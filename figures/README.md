# Figures — 인덱스

> Paper-grade figures (`current/`) 와 옛 figures (`legacy/`) 분리.

---

## 📂 `current/` — Paper-grade figures (현재 시스템)

| 폴더 | 용도 | 핵심 파일 |
|---|---|---|
| `01_sensor_layout/` | 39 sensor 평면도 | `sensor_layout.png`, `sensor_layout_annotated.png` |
| `02_l1_l4_layers/` | L1-L4 평가 layer 비교 | `model_comparison.png`, `per_location.png` |
| `03_sparse_interpolation/` | Sparse + geodesic IDW (39 sensor) | `method_comparison_geodesic.png`, `snapshot_T05_geodesic.png` |
| **`04_tier1_gnn/`** | **Tier 1 GNN 결과 (paper headline)** | **`headline.png`**, `aggregate_iou.png`, `sim_*_t12.png` (13개) |
| `05_future_prediction/` | 60s autoregress (L2 ideal) | `sim_*_grid_t0_*.png` (cold-start vs mid-fire) |
| `06_detector_triggered/` | L3 detector-triggered eval | `iou_per_scenario.png`, `trigger_time_distribution.png` |
| `07_sparse_retrain_v3/` | L4e Sparse-retrain ConvLSTM | `full_stack_comparison.png`, `per_scenario.png` |

---

## 🎬 Paper Figures (Top 6)

| # | Figure | 경로 | 내용 |
|---|---|---|---|
| **1** | **Tier 1 GNN headline** | `current/04_tier1_gnn/headline.png` | T05 1500kW 2m² 의 FDS vs GNN 시각 비교 (페이퍼 핵심) |
| 2 | Per-scenario IoU/FNR | `current/04_tier1_gnn/aggregate_iou.png` | 13 OOD 시나리오 모두 H5 통과 |
| 3 | L1-L4 layer 비교 | `current/02_l1_l4_layers/model_comparison.png` | 3 모델 × 4 metrics |
| 4 | 39 sensor layout | `current/01_sensor_layout/sensor_layout.png` | 평면도 + 색상별 sensor 종류 |
| 5 | Geodesic vs Linear | `current/03_sparse_interpolation/snapshot_T05_geodesic.png` | Mask-aware 보간 효과 |
| 6 | 60s autoregress | `current/05_future_prediction/sim_1500kw_2m2_T05_grid_t0_120.png` | 3 모델 × 6 step |

---

## 📦 `legacy/` — 옛 figures (참고용)

이전 sensor 위치 (16, 27 sensor 버전) 또는 발견 단계별 figures.
새 평가는 모두 `current/` 의 39-sensor v3.3 기반.

| 폴더 | 옛 시스템 |
|---|---|
| `eval_convlstm/` | 초기 ConvLSTM 학습 결과 |
| `eval_T01_T05/` | 16-sensor 기반 OOD eval (ConvLSTM) |
| `eval_T01_T05_fno_no_pi/`, `eval_T01_T05_fno_pi/` | FNO 변종 16-sensor eval |
| `first_sim/` | 초기 single-scenario 검증 |
| `risk_compare/` | FDS vs ConvLSTM risk map 초기 비교 |
| `sensor_layout_v2/` | STL + mask overlay (정보 비교용) |
| `sim_1000kw_1m2_001/` | 초기 single-scenario figure |
| `sparse_retrain/`, `sparse_retrain_smoke/` | Sparse-input retrain preliminary |
| `sparse_sensing/` | 16-sensor linear/cubic/nearest |
| `sparse_sensing_geodesic/` | 16-sensor geodesic |
| `sparse_sensing_geodesic_27/` | 27-sensor geodesic (D-024 v1) |

---

## 🔄 새 figure 생성 방법

| Figure 종류 | 스크립트 |
|---|---|
| 39 sensor 평면도 | `python scripts/visualize_sensor_layout.py` |
| Tier 1 GNN headline | `python scripts/visualize_tier1_predictions.py` |
| Sparse + geodesic | `python scripts/evaluate_sparse_sensing_geodesic.py --sensor-source d024` |
| 60s autoregress | `python scripts/visualize_60s_prediction.py` |
| L1-L4 비교 | `python scripts/hypothesis_validation.py` |

→ 모두 `figures/current/<subdir>/` 에 자동 저장.

---

## ⚠ Git ignore 정책

| 패턴 | 처리 |
|---|---|
| `figures/**/*.png`, `*.pdf`, `*.svg` | ✅ track (root 의 화이트리스트) |
| `figures/**/*.gif`, `*.mp4` | ❌ ignore (history 부풀음 방지) |
| `figures/legacy/` | 일부 디렉터리 ignore 룰 잔존 (eval_convlstm, risk_compare, etc.) |

GIF / 동영상은 외부 호스팅 권장 (Drive, YouTube).
