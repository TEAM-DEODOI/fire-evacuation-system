# Archive — 옛 자료

> 현재 문서 (`docs/*.md` numbered 00-90 + reference) 로 대체된 옛 자료들.
> 참고용으로 보존. 새 세션에서 직접 읽을 필요 없음.

---

## old_planning/

세션별 planning 문서, 초기 매뉴얼.

| 파일 | 대체 |
|---|---|
| `handoff_2026_05_12.md` | `docs/README.md` + `docs/00_project_overview.md` |
| `session_2026_05_13_summary.md` | `docs/70_results_summary.md` |
| `manual_v2.md` | `docs/00_project_overview.md` (compact) + `docs/90_next_steps.md` |
| `task_request_template.md` | (보존) — Claude Code 작업 요청 양식 |
| `tier1_gnn_design.md` | `docs/50_tier1_gnn_binary.md` (구현 후 갱신) |
| `tier1_detector_plan.md` | `docs/30_sensor_infrastructure.md` |
| `tier1_detector_task.md` | `docs/30_sensor_infrastructure.md` (D-023) |
| `tier1_detector_positions_task.md` | `docs/30_sensor_infrastructure.md` (D-024) |

## auto_reports/

스크립트 자동 생성된 평가 보고서. 정량 결과는 `docs/70_results_summary.md` 에 집계됨.

| 파일 | 출처 스크립트 | 핵심 결과 |
|---|---|---|
| `eval_T01_T05_report.md` | `evaluate_t_locations.py` (ConvLSTM) | L1 ConvLSTM 평균 IoU 0.89 |
| `eval_T01_T05_fno_no_pi_report.md` | (FNO no-PI) | L1 0.83 |
| `eval_T01_T05_fno_pi_report.md` | (FNO PI) | L1 0.84 |
| `hypothesis_validation_report.md` | `hypothesis_validation.py` | H1-H6 (구버전) |
| `cold_start_finding.md` | (수동) | Cold-start vs mid-fire 발견 |
| `detector_triggered_evaluation.md` | `evaluate_detector_triggered.py` | L3 mean 0.53 |
| `sparse_sensing_evaluation.md` | `evaluate_sparse_sensing.py` | L4 (16 sensor) linear/cubic/nearest |
| `sparse_sensing_geodesic_evaluation.md` | (16 sensor geodesic) | L4d (16 sensor) 0.41 |
| `sparse_sensing_geodesic_27_evaluation.md` | (27 sensor) | L4d (27 sensor) 0.41 |
| `sparse_sensing_geodesic_v3_evaluation.md` | (39 sensor, **current**) | L4d (39 sensor) 0.43 (FNO no-PI) |
| `sparse_retrain_evaluation.md` | `evaluate_sparse_model.py` | L4e preliminary 0.07 (학습 중단) |
| `sparse_retrain_smoke_eval.md` | (5-ep smoke) | L4e smoke 0.20 |

---

## 참고: 옛 → 새 매핑

```
old_planning/handoff_2026_05_12.md
  → 00_project_overview.md (헌장)
  + 70_results_summary.md (결과)
  + 90_next_steps.md (다음 작업)

old_planning/manual_v2.md (14주 계획)
  → 00_project_overview.md §7 (다음 작업 우선순위)
  + 90_next_steps.md

old_planning/tier1_gnn_design.md
  → 50_tier1_gnn_binary.md (구현된 실제 명세)

old_planning/tier1_detector_plan.md, _task.md, _positions_task.md
  → 30_sensor_infrastructure.md (D-023 + D-024 통합)

auto_reports/*.md (12개)
  → 70_results_summary.md (정량 집계)
  + 80_hypothesis_validation.md (가설별)
  + 60_evaluation_layers.md (L1-L4)
```

새 자료가 어떤 옛 자료를 어떻게 대체했는지 추적용.
