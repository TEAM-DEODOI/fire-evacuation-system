# 50 — Tier 1 GNN (Binary Detector → Graph Prediction)

> Tier 1 시스템 — 39 화재 감지기의 binary signal 만으로 60초 미래 노드별
> 위험도 예측. **Paper의 핵심 contribution**.

---

## 1. 시스템 개요

**입력**: 39 nodes × 6 frame × 5 feature
**출력**: 39 nodes × 6 frame (60s) danger ∈ [0, 1]
**Params**: 12,006
**Test IoU**: **0.904** (13 OOD 시나리오 평균, H5 ✅)

추가 hardware 없이 기존 화재 감지기만으로 ideal upper bound (0.92) 의 98% 도달.

---

## 2. 입력 — Node Feature

각 sensor 노드의 5 feature × 6 timestep history:

| Feature | 의미 | 값 |
|---|---|---|
| 0 | `is_detected` | D-023 binary 0/1 (latched) |
| 1 | `det_time_norm` | 활성화 시각 정규화 `t_act / 300`, masked by is_detected |
| 2 | `type_onehot[0]` | 1 if room else 0 |
| 3 | `type_onehot[1]` | 1 if corridor else 0 |
| 4 | `type_onehot[2]` | 1 if exit else 0 |

**입력 텐서 shape**: `(B, N=39, T_in=6, F=5)`

**추가 입력**: 인접 행렬 `adj` shape `(39, 39)`, k-NN graph (k=4), Gaussian kernel, 대칭정규화.

---

## 3. 모델 아키텍처 — SimpleFireGNN

**구현**: `src/tier1/tier1_gnn.py` (**PyTorch only, no PyG dependency**)

```python
class SimpleFireGNN(nn.Module):
    def __init__(self, in_feat=5, hidden=32, n_graph_layers=2, T_out=6):
        # 1) Node + time encoder
        self.node_encoder = nn.Linear(in_feat, hidden)
        
        # 2) Temporal GRU (shared across nodes)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        
        # 3) Graph propagation (k 회 message passing)
        self.graph_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_graph_layers)
        ])
        
        # 4) Output head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, T_out),
            nn.Sigmoid(),
        )

    def forward(self, x, adj):
        # x: (B, N, T_in, F=5)
        # 1) Encode per (node, time)
        h = self.node_encoder(x.reshape(B*N, T, F))
        
        # 2) GRU temporal — take last hidden as node summary
        _, h_last = self.gru(h)
        h_node = h_last.squeeze(0).reshape(B, N, hidden)  # (B, N, H)
        
        # 3) Graph propagation — A @ H, residual MLP
        for layer in self.graph_layers:
            h_neigh = torch.einsum("nm,bmh->bnh", adj, h_node)
            h_node = h_node + layer(h_neigh)
        
        # 4) Output — per-node future danger
        return self.head(h_node)  # (B, N, T_out) ∈ [0, 1]
```

### 3.1 아키텍처 결정

| 구성 요소 | 선택 | 이유 |
|---|---|---|
| GNN library | **PyTorch only** | PyG/PyG-Temporal Python 3.12 wheel 호환 회피 |
| Temporal | GRU (shared) | 단순, 작은 input window (T=6) 에 적합 |
| Graph operator | Adjacency-weighted message passing | GCN style, simple |
| Layer 수 | 2 | 39 노드 작아서 적당 |
| Output activation | Sigmoid | danger ∈ [0, 1] 강제 |

### 3.2 Adjacency 구성

`build_knn_adjacency(k=4, sigma=5.0)`:
1. 39 sensor 의 (x, y) Euclidean distance matrix
2. k=4 nearest neighbors 선택
3. Gaussian weight: `exp(-d² / 2σ²)`, σ=5m
4. Self-loop 추가 (대각 1)
5. 대칭정규화: `D^(-1/2) A D^(-1/2)`

→ 인접 sensor 간 message passing, 멀리 떨어진 sensor 영향 작음.

---

## 4. 학습 데이터 — `Tier1FireDataset`

**구현**: `src/tier1/tier1_dataset.py`

### 4.1 데이터 소스

`results/detector_sequences/<scenario>.npz` (46 시나리오, `scripts/build_detector_sequences.py` 로 생성).

각 .npz 내용:
| 필드 | shape | 의미 |
|---|---|---|
| `binary_sequence` | (31, 39) | D-023 trigger latched 0/1 |
| `node_danger` | (31, 39) | FDS truth 의 sensor 위치 danger (**GNN target**) |
| `activation_times` | (39,) | 첫 trigger 시각 (s) or -1 |
| `trigger_reasons` | (39,) | "heat"/"smoke"/"both"/"none" |
| `detector_ids` | (39,) | sensor 이름 |

### 4.2 Sliding window

- T_in = 6 (input 60s history)
- T_out = 6 (output 60s future)
- 1 시나리오당 31 - 12 + 1 = **20 pair**

**Split** (default):
| Split | 시나리오 | Pair 수 |
|---|---|---|
| Train | s_000 ~ s_032 (33개) | 660 |
| Val | sim_*_T* 처음 3개 (sim_1000kw_*) | 60 |
| Test | sim_*_T* 나머지 10개 | 200 |

### 4.3 `__getitem__` 반환

```python
x: (N=39, T_in=6, F=5) torch.float32
y: (N=39, T_out=6) torch.float32
```

---

## 5. 학습 결과 (Tier 1 GNN v3)

**체크포인트**: `checkpoints/tier1_gnn_v3/best.pt`
**학습 시간**: ~5-10분 CPU
**총 params**: 12,006

### 5.1 학습 곡선

`checkpoints/tier1_gnn_v3/loss_curve.png`:
- Train MSE: 0.0385 (epoch 1) → 0.0051 (epoch 100)
- Best val IoU: **0.872** at epoch 25 (early best, ~5 epoch 만에 사실상 수렴)
- 학습 안정적, 약간의 overfitting 후 plateau

### 5.2 평가 결과 (13 OOD 시나리오)

| 메트릭 | Mean | Best | Worst | H5/H4 |
|---|---|---|---|---|
| **IoU @ +60s** | **0.904** | 1.000 (T05 1500kW 1m²) | 0.789 (T04 500kW 1m²) | **13/13 ✅** |
| **FNR @ +60s** | **4.6%** | 0% (5 시나리오) | 14.3% (T03 1500kW 1m²) | **11/13 ✅** |

### 5.3 시나리오별 결과

| 시나리오 | IoU step 6 | FNR step 6 |
|---|---|---|
| sim_1500kw_2m2_T05 | **1.000** ★ | 0.0% |
| sim_500kw_2m2_T02 | 0.969 | 0.0% |
| sim_1500kw_1m2_T02 | 0.941 | 0.0% |
| sim_500kw_1m2_T01 | 0.933 | 0.0% |
| sim_1000kw_1m2_T01 | 0.931 | 3.6% |
| sim_1000kw_2m2_T05 | 0.923 | 7.7% |
| sim_1000kw_1m2_T03 | 0.900 | 5.3% |
| sim_500kw_2m2_T05 | 0.909 | 3.2% |
| sim_500kw_1m2_T03 | 0.909 | 9.1% |
| sim_1000kw_2m2_T01 | 0.886 | 11.4% ⚠ |
| sim_1500kw_1m2_T03 | 0.857 | 14.3% ⚠ |
| sim_500kw_1m2_T02 | 0.810 | 5.6% |
| sim_500kw_1m2_T04 | 0.789 | 0.0% |

→ **모든 시나리오 H5 (0.70) 통과**. H4 (FNR < 10%) 는 2 시나리오 살짝 미달.

---

## 6. 시각화 — Paper Headline Figures

### 6.1 `figures/current/04_tier1_gnn/headline.png`

**Paper Figure 1**: T05 1500kW 2m² 시나리오의 FDS truth vs GNN prediction.
- 2 row (truth / pred) × 3 col (t₀+10s / +30s / +60s)
- 39 노드의 색상 매핑이 truth 와 시각적으로 거의 동일
- **메시지**: *"12K params 모델이 binary signal 만으로 ground truth 와 거의 일치"*

### 6.2 `figures/current/04_tier1_gnn/aggregate_iou.png`

**Paper Figure 2**: 13 OOD 시나리오의 IoU + FNR 막대 (per-scenario).
- 모두 H5 (0.70) 위
- 2 개 시나리오만 FNR 10%+ (T03 1500kW 1m², T01 1000kW 2m²)

### 6.3 시나리오별 detail figures

`figures/current/04_tier1_gnn/sim_*_t12.png` (13개):
- 3 row (truth / pred / |error|) × 6 col (60s timeline)
- 각 시나리오의 시간별 진행 + error map

생성 스크립트: `scripts/visualize_tier1_predictions.py`

---

## 7. Tier 1 GNN 이 Tier 2 보다 우수한 이유

### 7.1 정보 손실 비교

| 단계 | Tier 1 | Tier 2 (sparse) |
|---|---|---|
| Sensor 측정 | 39 nodes binary | 39 nodes continuous (3 channels) |
| 변환 | (없음) | sparse → dense interpolation |
| **정보 손실** | **0 (binary 그대로)** | **큼 (over-smoothing)** |

### 7.2 화재의 phase-transition 특성

화재 spread 는 본질적으로 **discrete trigger event** (감지/미감지).
- Binary 가 이 phase-transition 을 정확히 표현
- Continuous interpolation 은 sharp transition 을 over-smooth → 정보 손실
- GNN message passing 이 인접 노드 trigger 패턴을 자연스럽게 propagate

### 7.3 모델 capacity vs domain match

| 모델 | params | IoU |
|---|---|---|
| FNO no-PI (1.78M) + geodesic IDW | 1.78M | 0.43 |
| **GNN (12K)** | **12K** | **0.90** |

→ **150× 작은 모델이 2× 정확** — domain matching (graph + temporal RNN inductive bias) 이 capacity 보다 중요.

---

## 8. 모듈 인터페이스

```python
# Adjacency
from src.tier1.tier1_gnn import build_knn_adjacency, build_node_type_onehot
adj = build_knn_adjacency(k=4)              # (39, 39) torch.float32
types = build_node_type_onehot()            # (39, 3)

# Dataset
from src.tier1.tier1_dataset import Tier1FireDataset, default_splits
train_names, val_names, test_names = default_splits()
ds = Tier1FireDataset(
    sequence_dir=Path("results/detector_sequences"),
    scenario_names=train_names, T_in=6, T_out=6,
)

# Model
from src.tier1.tier1_gnn import SimpleFireGNN
model = SimpleFireGNN(in_feat=5, hidden=32, n_graph_layers=2, T_out=6)

# Training (한 줄)
# python scripts/train_tier1_gnn.py --epochs 100 --batch-size 8 --output checkpoints/tier1_gnn_v3
```

---

## 9. 다음 작업 — Tier1RiskMap

**미구현**: `src/tier1/tier1_risk_map.py`

설계:
```python
class Tier1RiskMap(RiskMap):
    """GNN 출력 → RiskMap 인터페이스 어댑터."""
    
    def __init__(self, gnn_model, current_binary_history, adj, t0):
        # 현재까지의 binary history 로 GNN forward
        self.gnn_pred = gnn_model(...)  # (39, T_out=6)
        self.t0 = t0
        self.node_positions = [d.position for d in ALL_DETECTORS]
    
    def query(self, xyz, t=None):
        # 1) 가장 가까운 sensor node 찾기
        # 2) (t - t0) / 10 → step index
        # 3) self.gnn_pred[node_idx, step_idx] 반환
```

→ H6 검증 (`EXP-PATH-001`) 을 위해 다음 세션 작업.
