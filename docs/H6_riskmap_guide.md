# H6 RiskMap 사용 가이드 — 5 후보 통합 매뉴얼

> H6 (EXP-PATH-001) 에서 사용할 5가지 cell-level RiskMap 후보 별
> ckpt 파일, forward stack, `RiskMap(xyz, t)` 인터페이스로 wrap 하는 방법.
>
> **읽는 순서**: §1 (개요) → §2 (네가 쓰는 후보의 ckpt) → §3 (해당 후보 wrap 코드)
> → §6 (EXP-PATH-001 통합).
>
> **선행 자료**: `docs/decisions.md` D-026 (hand-crafted), D-028 (learned
> decoder), D-029 (H6 RiskMap source 결정). `src/tier1/ensemble_risk_map.py`
> (④, ⑤ 이미 wrap 완료).

---

## 1. 5 후보 한눈에

| # | RiskMap 후보 | OOD IoU | OOD FNR | 통합 방식 | wrap 추가 코드 |
|---|---|---|---|---|---|
| ① | Sparse-ConvLSTM v3 단독 | 0.581 | 23.0% | single surrogate | ~30 줄 helper |
| ② | Sparse-FNO v3 단독 | 0.525 | **10.4%** | single surrogate | ~30 줄 helper |
| ③ | Hand-crafted 3-way Balanced | 0.618 | **5.1%** | 고정 가중평균 (D-026) | ~30 줄 helper |
| **④ ★** | **Learned Decoder fn=2.5** | **0.733** | 11.5% | per-cell MLP (D-028, paper main) | **0 줄** ✓ |
| ⑤ | Learned Decoder fn=4.0 | 0.718 | **10.0%** | per-cell MLP, safety variant | **0 줄** ✓ |
| oracle | FDS truth | 1.0 | 0% | ground truth | **0 줄** ✓ |

→ H6 ablation 추천 = ②, ③, ④, oracle (4-way staircase narrative).

`RiskMap` 인터페이스 contract (모든 후보 공통):
```python
risk_map.query(xyz: np.ndarray, t: float | None) -> float | np.ndarray
# xyz: (3,) or (M, 3)  in world metres
# t:   simulation time (s).  None → uses t_max
# 반환: danger ∈ [0, 1].
# 규칙:  - 공간 OOB / 시간 OOB → 1.0 (max danger, 보수적)
#       - mask=0 (solid) cell → ~0 (decoder/StaticRiskMap 자체 처리)
```

---

## 2. ckpt 파일 매핑

```
checkpoints/
├── tier1_gnn_v3/best.pt              ← ③, ④, ⑤ 공통 base (53 KB)
├── conv_lstm_sparse_v3/best.pt       ← ①, ③, ④, ⑤ 공통 base (1.4 MB)
├── fno_sparse_v3/best.pt             ← ②, ③, ④, ⑤ 공통 base (14 MB)
├── ensemble_decoder/best.pt          ← ④ 만 (paper default, = fn=2.5 복사본, 8 KB)
├── ensemble_decoder_fn25/best.pt     ← ④ 명시적 (8 KB)
├── ensemble_decoder_fn40/best.pt     ← ⑤ (8 KB)
└── ensemble_decoder_fn10/best.pt     ← (paper ablation only, H6 불사용, 8 KB)
```

후보별 필요 ckpt 정리:

| # | 후보 | 필요한 ckpt |
|---|---|---|
| ① | Sparse-ConvLSTM 단독 | `conv_lstm_sparse_v3/best.pt` |
| ② | Sparse-FNO 단독 | `fno_sparse_v3/best.pt` |
| ③ | Hand-crafted 3-way | `tier1_gnn_v3` + `conv_lstm_sparse_v3` + `fno_sparse_v3` |
| ④ ★ | Decoder fn=2.5 | 위 3개 + `ensemble_decoder/best.pt` (또는 `ensemble_decoder_fn25/best.pt`) |
| ⑤ | Decoder fn=4.0 | 위 3개 + `ensemble_decoder_fn40/best.pt` |

모든 ckpt 가 `.gitignore` 의 whitelist 에 포함 → `git clone` 으로 자동 받음.

---

## 3. 후보별 forward + RiskMap wrap 코드

### ① Sparse-ConvLSTM v3 단독

```python
# scripts 의 helper 재사용
import torch, numpy as np
from src.data_pipeline.fds_extractor import extract_slices
from src.data_pipeline.normalize import build_input_tensor, normalize_scenario
from src.risk_map.converter import prediction_to_danger
from src.risk_map.risk_map_class import StaticRiskMap
from src.shared.constants import DT_SLCF
from scripts.evaluate_t_locations import load_mask, load_model
from scripts.train_sparse_conv_lstm import load_sensor_indices, make_sparse_indicator
from scripts.visualize_60s_5model import (
    autoregress_sparse_input, sparsify_initial_input,
)

LOOKAHEAD_STEPS = 6


def make_sparse_conv_risk_map(
    scen_name: str, t0: float, *,
    raw_root="data/raw", dataset_h5="data/processed/dataset.h5",
    building="configs/building.yaml", device="cpu",
):
    """① Sparse-ConvLSTM 단독 → StaticRiskMap."""
    device = torch.device(device)
    mask = load_mask(dataset_h5)
    sensor_idxs = load_sensor_indices(building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)

    model = load_model("checkpoints/conv_lstm_sparse_v3/best.pt",
                       device, "conv_lstm")

    slices = extract_slices(f"{raw_root}/{scen_name}")
    norm = normalize_scenario(slices)
    inp = build_input_tensor(norm, mask, times=slices["times"])
    t0_idx = int(t0 // DT_SLCF)

    init_sparse = sparsify_initial_input(inp[t0_idx], sparse_ind)
    preds_norm = autoregress_sparse_input(
        model, init_sparse, sparse_ind, t0, device,
    )
    times_arr = np.array([t0 + (s + 1) * DT_SLCF
                            for s in range(LOOKAHEAD_STEPS)])
    cell_danger = prediction_to_danger(preds_norm, times_arr)
    cell_danger = cell_danger * (mask > 0.5).astype(np.float32)[None, ...]
    return StaticRiskMap(danger_array=cell_danger, times=times_arr)
```

### ② Sparse-FNO v3 단독

```python
from scripts.evaluate_sparse_fno import (
    load_sparse_fno, build_sparse_6ch_input, autoregress_sparse_fno,
)


def make_sparse_fno_risk_map(
    scen_name: str, t0: float, *,
    raw_root="data/raw", dataset_h5="data/processed/dataset.h5",
    building="configs/building.yaml", device="cpu",
):
    """② Sparse-FNO 단독 → StaticRiskMap."""
    device = torch.device(device)
    mask = load_mask(dataset_h5)
    sensor_idxs = load_sensor_indices(building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)

    model = load_sparse_fno("checkpoints/fno_sparse_v3/best.pt", device)

    slices = extract_slices(f"{raw_root}/{scen_name}")
    inp_6ch = build_sparse_6ch_input(slices, mask, sparse_ind)
    t0_idx = int(t0 // DT_SLCF)

    preds_norm = autoregress_sparse_fno(
        model, inp_6ch[t0_idx], sparse_ind, t0, device, resparsify=True,
    )
    times_arr = np.array([t0 + (s + 1) * DT_SLCF
                            for s in range(LOOKAHEAD_STEPS)])
    cell_danger = prediction_to_danger(preds_norm, times_arr)
    cell_danger = cell_danger * (mask > 0.5).astype(np.float32)[None, ...]
    return StaticRiskMap(danger_array=cell_danger, times=times_arr)
```

### ③ Hand-crafted 3-way Balanced (D-026)

```python
from src.tier1.detector_positions import ALL_DETECTORS
from src.tier1.tier1_gnn import SimpleFireGNN, build_knn_adjacency
from scripts.evaluate_ensemble import (
    gnn_node_pred_to_cell_danger, precompute_node_to_cell_weights,
    tier1_forward,
)


def make_handcrafted_risk_map(
    scen_name: str, t0: float, *,
    raw_root="data/raw", seq_dir="results/detector_sequences",
    dataset_h5="data/processed/dataset.h5",
    building="configs/building.yaml", device="cpu",
    w_t1: float = 0.5, w_conv: float = 0.25, w_fno: float = 0.25,
):
    """③ Hand-crafted 3-way Balanced (D-026, geodesic) → StaticRiskMap.

    cell_danger = w_t1 · gnn_cell + w_conv · sparse_conv + w_fno · sparse_fno
    (default = 0.5 / 0.25 / 0.25 from D-026 grid search)
    """
    device = torch.device(device)
    mask = load_mask(dataset_h5)
    sensor_idxs = load_sensor_indices(building)
    sparse_ind = make_sparse_indicator(sensor_idxs, broadcast_z=True)

    # 1) Tier 1 GNN
    gnn_ckpt = torch.load("checkpoints/tier1_gnn_v3/best.pt",
                            weights_only=False, map_location=device)
    cfg = gnn_ckpt["config"]
    gnn = SimpleFireGNN(in_feat=5, hidden=cfg.get("hidden", 32),
                          n_graph_layers=cfg.get("n_graph_layers", 2),
                          T_out=cfg.get("T_out", 6))
    gnn.load_state_dict(gnn_ckpt["model"])
    gnn.to(device).eval()
    adj = build_knn_adjacency(k=cfg.get("knn_k", 4))

    # 2) Sparse models
    sparse_conv = load_model("checkpoints/conv_lstm_sparse_v3/best.pt",
                              device, "conv_lstm")
    sparse_fno = load_sparse_fno("checkpoints/fno_sparse_v3/best.pt", device)

    # 3) Geodesic IDW precompute
    node_positions = [d.position for d in ALL_DETECTORS]
    knn_idx, knn_w = precompute_node_to_cell_weights(
        node_positions, k=3, sigma=5.0, mask=mask, use_geodesic=True,
    )

    # 4) Forward all 3
    slices = extract_slices(f"{raw_root}/{scen_name}")
    norm = normalize_scenario(slices)
    inp = build_input_tensor(norm, mask, times=slices["times"])
    inp_6ch = build_sparse_6ch_input(slices, mask, sparse_ind)
    t0_idx = int(t0 // DT_SLCF)
    t_start = t0_idx - 6
    times_arr = np.array([t0 + (s + 1) * DT_SLCF
                            for s in range(LOOKAHEAD_STEPS)])

    init_sparse = sparsify_initial_input(inp[t0_idx], sparse_ind)
    pc_norm = autoregress_sparse_input(sparse_conv, init_sparse,
                                          sparse_ind, t0, device)
    sparse_conv_danger = prediction_to_danger(pc_norm, times_arr)

    pf_norm = autoregress_sparse_fno(sparse_fno, inp_6ch[t0_idx],
                                        sparse_ind, t0, device,
                                        resparsify=True)
    sparse_fno_danger = prediction_to_danger(pf_norm, times_arr)

    t1_node = tier1_forward(scen_name, seq_dir, gnn, adj, t_start, device)
    gnn_cell = gnn_node_pred_to_cell_danger(t1_node, knn_idx, knn_w)

    # 5) Weighted average (D-026 default 0.5 / 0.25 / 0.25)
    cell_danger = (
        w_t1 * gnn_cell + w_conv * sparse_conv_danger + w_fno * sparse_fno_danger
    )
    cell_danger = np.clip(cell_danger, 0.0, 1.0).astype(np.float32)
    cell_danger = cell_danger * (mask > 0.5).astype(np.float32)[None, ...]
    return StaticRiskMap(danger_array=cell_danger, times=times_arr)
```

### ④ ★ Learned Decoder fn=2.5 (paper main) — 추가 코드 0 줄

이미 `src/tier1/ensemble_risk_map.py` 에 완비. 호출만:

```python
from src.tier1.ensemble_risk_map import EnsembleDecoderRiskMap
from src.tier1.ensemble_decoder import PerCellEnsembleDecoder

# (위 ③ 와 동일하게 base 3개 모델 + knn_idx/knn_w 로딩 후)
dec_ckpt = torch.load("checkpoints/ensemble_decoder/best.pt",
                        weights_only=False, map_location=device)
decoder = PerCellEnsembleDecoder(
    hidden=dec_ckpt["config"]["hidden"],
    n_layers=dec_ckpt["config"]["n_layers"],
)
decoder.load_state_dict(dec_ckpt["model"])
decoder.to(device).eval()

risk_map = EnsembleDecoderRiskMap.from_scenario(
    scen_name="sim_1500kw_2m2_T05",
    t0=120.0,
    gnn_model=gnn, adj=adj,
    sparse_conv_model=sparse_conv,
    sparse_fno_model=sparse_fno,
    decoder=decoder,
    mask=mask, sensor_indicator=sparse_ind,
    knn_idx=knn_idx, knn_w=knn_w,
    seq_dir=Path("results/detector_sequences"),
    raw_root=Path("data/raw"),
    device=device,
)

# query
danger = risk_map.query(np.array([15.0, 10.0, 1.5]), t=150.0)  # → 0.85 등
```

### ⑤ Learned Decoder fn=4.0 (safety variant) — ckpt 경로만 다름

```python
dec_ckpt = torch.load("checkpoints/ensemble_decoder_fn40/best.pt",
                        weights_only=False, map_location=device)
# 나머지는 ④ 와 동일
```

### Oracle (FDS truth) — fairness baseline

```python
from src.risk_map.risk_map_class import StaticRiskMap

oracle_map = StaticRiskMap.from_fds_dir(Path("data/raw") / scen_name)
# 시간 범위 [0, 300]s 의 full FDS truth danger
```

---

## 4. 통합된 RiskMap factory (권장)

`src/path_planning/risk_map_factory.py` 라는 작은 dispatcher 를 만들면 H6
실험 코드가 깔끔해진다 (선택사항):

```python
# src/path_planning/risk_map_factory.py
from pathlib import Path
from src.risk_map.risk_map_class import StaticRiskMap


def make_risk_map(tag: str, scen_name: str, t0: float, **shared_kwargs):
    """tag ∈ {'sparse-conv','sparse-fno','hand-crafted',
                  'decoder-fn25','decoder-fn40','oracle'}"""
    if tag == "sparse-conv":
        return make_sparse_conv_risk_map(scen_name, t0, **shared_kwargs)
    if tag == "sparse-fno":
        return make_sparse_fno_risk_map(scen_name, t0, **shared_kwargs)
    if tag == "hand-crafted":
        return make_handcrafted_risk_map(scen_name, t0, **shared_kwargs)
    if tag.startswith("decoder-"):
        ckpt = {
            "decoder-fn25": "checkpoints/ensemble_decoder/best.pt",
            "decoder-fn40": "checkpoints/ensemble_decoder_fn40/best.pt",
        }[tag]
        return _make_decoder_risk_map(scen_name, t0, ckpt, **shared_kwargs)
    if tag == "oracle":
        return StaticRiskMap.from_fds_dir(
            Path(shared_kwargs.get("raw_root", "data/raw")) / scen_name
        )
    raise ValueError(f"unknown tag {tag}")
```

---

## 5. 추천 H6 ablation (4-way staircase)

`docs/decisions.md` D-029 + `docs/90_next_steps.md` §2.2 의 결정:

```
EXP-PATH-001:  3 scenarios × 3 planners × 4 risk-maps × 8 starts = 288 trials
```

| RiskMap tag | 후보 | 역할 |
|---|---|---|
| `sparse-fno`     | ② | naive Tier 2 surrogate baseline |
| `hand-crafted`   | ③ | engineered ensemble (D-026) |
| `decoder-fn25` ★ | ④ | paper main contribution |
| `oracle`         | — | fairness ceiling |

각 trial 의 결과 표 column = 위 4개 RiskMap. paper §4 의 한 표.

추가로 비교하려면 (시간 여유 시):
- ⑤ `decoder-fn40` — safety variant; FNR vs IoU trade-off 강조
- ① `sparse-conv` — conservative bias narrative

---

## 6. EXP-PATH-001 entry point 명세

`experiments/exp_path_001.py` 작성 시 권장 CLI:

```bash
python experiments/exp_path_001.py \
    --scenarios sim_1500kw_2m2_T05 sim_500kw_1m2_T01 sim_1000kw_1m2_T03 \
    --planners dijkstra static dynamic \
    --risk-maps sparse-fno hand-crafted decoder-fn25 oracle \
    --start-positions 8 \
    --output results/exp_path_001/
```

각 trial 의 4-metric 측정 (D-022):
- `peak_danger`     — 경로 상 최대 위험도 ∈ [0, 1]
- `time_in_hazard_s` — danger > 0.5 체류 시간 (초)
- `aset_margin_s`   — ASET 안전 여유 시간
- `fed_final`       — 누적 FED (보조)

가설 **H6**:
> Dynamic A* (with risk_map=decoder-fn25) 의 평균 FED ≤ 0.7 × Dijkstra
> 의 평균 FED (≥ 30% 감소)

추가로 자연스럽게 검증되는 narrative:
- "어떤 surrogate 가 oracle 의 몇 % 회복?" → `decoder-fn25 / oracle` 비율
- "Learned vs hand-crafted 의 dynamic-planning 효과 차이?" →
  `decoder-fn25 - hand-crafted` 의 FED 감소율 차이

---

## 7. 함정 / Gotchas

1. **`t_start` 인덱싱** — Tier 1 GNN 의 input window 는 [t_start, t_start+5]
   (6 frame history). t₀=120s 이면 t_start = 120/10 - 6 = 6.
   다른 t₀ 일반화는 `t_start = int(t0 // DT_SLCF) - 6`.

   GNN 입력은 binary 신호만이 아니라 **5-D feature** ((N=39, T_in=6, F=5)):
   * channel 0 `is_detected` — binary {0, 1}, D-023 trigger latched
   * channel 1 `det_time_norm` — continuous [0, 1], `activation_time / 300s`
   * channel 2-4 `type_onehot` — static {room, corridor, exit}

   세 종류 모두 기존 화재감지기 인프라에서 얻을 수 있는 정보 (트리거 +
   타임스탬프 + 설계도) — "추가 센서 hardware 0" framing 유효.
   `Tier1FireDataset.__getitem__` 이 5-D 구성을 자동 처리한다.

2. **Cold-start 회피** — t₀ < 90s 영역에서는 decoder IoU 가 0.662 까지
   떨어진다 (multi-t₀ 검증, commit `5fe5c03`). H6 evacuation simulator 는
   감지기 trigger 시점부터 시작 (D-023) — t₀ ≥ 90s 보장.

3. **Geodesic 캐시** — `precompute_node_to_cell_weights(...)` 는 한 번만
   호출. EvacuationSimulator 의 각 replan tick 에서 재호출 금지 (수십 초
   비용). 시뮬레이터 생성자에서 한 번 캐시.

4. **Mask multiply** — ③, ④, ⑤ 의 cell_danger 에 fluid mask 곱해야 시각화
   일관성 유지. `cell_danger * (mask > 0.5)[None, ...]`. Metric 계산은 어차피
   mask-filter 되니 영향 없음.

5. **Re-sparsify** — `autoregress_sparse_input` / `autoregress_sparse_fno`
   는 `resparsify=True` 가 default. 명시 안 해도 OK 이지만 명시적으로
   적어두면 의도 명확 (D-025, L-013).

6. **GNN ckpt config 의 `knn_k`** — adj graph 빌드 시 `cfg.get("knn_k", 4)`
   사용. ckpt 의 config 가 default 면 k=4. 4가 maintainer 결정.

---

## 8. 작동 확인 (H6 시작 전)

```bash
# 1. 모든 ckpt 로드 + self-test
python -m src.tier1.tier1_risk_map           # α (per-node GNN, ablation)
python -m src.tier1.ensemble_risk_map        # ④/⑤ cell-level
python -m src.risk_map.risk_map_class        # oracle
python -m src.tier1.ensemble_decoder         # decoder model

# 2. 한 시나리오에서 ④ paper main 호출 가능?
python scripts/measure_h1_inference.py       # 전체 stack timing 출력 확인
# 기대 출력: Full L4h pipeline ~456ms, speedup ~3,028x
```

위 4개 self-test 가 PASS 면 H6 의 RiskMap 인프라 모두 동작 보장. Path
planning 모듈 (edge_weights, planners, evacuation_sim) 만 작성하면 됨.

---

## 9. 관련 문서

| 주제 | 위치 |
|---|---|
| RiskMap source 결정 근거 | `docs/decisions.md` D-029 |
| Learned decoder 학습 디테일 | `docs/decisions.md` D-028 |
| Hand-crafted ensemble 디테일 | `docs/decisions.md` D-026 |
| Re-sparsify chaining (sparse 모델 사용 시) | `docs/decisions.md` D-025 |
| H6-prep 검증 (multi-t₀, CV, H1) | `docs/CURRENT_SESSION_STATE.md` §2.4 |
| 다음 5-7시간 step-by-step | `docs/90_next_steps.md` §2 |
| 전체 재현 명령 | `REPRODUCE.md` |
