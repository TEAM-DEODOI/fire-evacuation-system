# 20 — Dataset & FDS Simulations

> FDS 시뮬레이션 셋업, 시나리오 분할, dataset.h5 명세.

---

## 1. FDS 시뮬레이션 셋업 (모든 시나리오 공통)

### 1.1 Geometry — Hard Constraints

| 항목 | 값 |
|---|---|
| Building | 단일층, Science Hall LV5 평면도 (미로형) |
| STL 원본 | `assets/science_hall_lv5.stl` (48 KB, 472 vertex, 972 face) |
| Real STL 높이 | 3.2 m (PyroSim 원본) |
| **FDS MESH** | 100 × 80 × 8 cells, XB = [−10, 40] × [−10, 30] × [0, 4] m |
| **SLCF region** | **60 × 40 × 6 cells**, XB = [0, 30] × [0, 20] × [0, 3] m |
| Cell resolution | **0.5 m × 0.5 m × 0.5 m** |
| External buffer | 10 m on −X/+X/−Y/+Y (ventilation 경계) |

→ **MESH** = FDS 가 계산하는 영역 (큰 buffer 포함).
**SLCF** = 모델 학습/평가용 (60×40×6 = 14,400 cells).

### 1.2 Time

| 항목 | 값 |
|---|---|
| 시뮬레이션 duration | 0–300 s (5분) |
| Frame 수 | **31** frames (0, 10, 20, …, 300 s) |
| DT_SLCF | 10.0 s |
| Single-step prediction | 10 s |
| Lookahead horizon | 60 s (6 autoregress steps) |
| Single-scenario CPU time | ~23 min |

### 1.3 SLCF 출력 채널 (3개)

| 채널 | QUANTITY | 단위 |
|---|---|---|
| Temperature | `TEMPERATURE` | °C |
| Visibility | `SOOT VISIBILITY` 또는 `VISIBILITY` | m |
| CO | `VOLUME FRACTION` w/ `SPEC_ID='CARBON MONOXIDE'` | (mol fraction → ppm 변환) |

⚠ `VECTOR=.TRUE.` 사용 금지 (L-001 버그). `CELL_CENTERED=.TRUE.` 필수.

---

## 2. 시나리오 구성

### 2.1 Training set: 33 시나리오 (s_000 ~ s_032)

`data/raw/scenario_config.json` 의 canonical 메타데이터:

| 그룹 | 구성 | 개수 |
|---|---|---|
| 9 standard locations × 3 HRR (500/1000/1500 kW) | s_000 ~ s_026 | 27 |
| 3 H locations × 2 HRR (500/1000 kW, 1500 kW 부재) | s_027 ~ s_032 | 6 |
| **합계** | | **33** |

**Split**: D-024 결정에 따라 **모두 train**. val/ood 는 별도 (T01-T05).

### 2.2 OOD evaluation set: 13 시나리오 (T01 ~ T05)

`data/raw/sim_*_T*` — 평가용으로 별도 생성, **위치 OOD** (training 의 9 locations 외).

위치 좌표 (각 시나리오의 fire location):

| 위치 | (x, y) | 시나리오 |
|---|---|---|
| **T01** | (28.5, 18.0) — 우상단 코너 | 500/1000(1m²)/1000(2m²) — 3개 |
| **T02** | (18.5, 7.5) — 중앙 하단 | 500/500(2m²)/1500 — 3개 |
| **T03** | (10.5, 2.5) — 좌하단 | 500/1000/1500 — 3개 |
| **T04** | (8.0, 17.0) — 좌상단 | 500 — 1개 |
| **T05** | (15.0, 16.0) — 중정 북측 | 1000/1500/1000(2m²)/1500(2m²) — 4개 |

**총 13 시나리오**, **건물 geometry 동일** (training 과 같은 MESH/SLCF).

### 2.3 시나리오 매트릭스

| HRR \ 위치 | T01 | T02 | T03 | T04 | T05 |
|---|---|---|---|---|---|
| 500 kW × 1m² | ✓ | ✓ | ✓ | ✓ | — |
| 500 kW × 2m² | — | ✓ | — | — | ✓ |
| 1000 kW × 1m² | ✓ | — | ✓ | — | — |
| 1000 kW × 2m² | ✓ | — | — | — | ✓ |
| 1500 kW × 1m² | — | ✓ | ✓ | — | — |
| 1500 kW × 2m² | — | — | — | — | ✓ |

---

## 3. 데이터셋 — `data/processed/dataset.h5`

`scripts/run_data_pipeline.sh` 로 빌드 (`src/data_pipeline/build_dataset.py`).

### 3.1 HDF5 구조

```
dataset.h5  (221 MB, gitignored)
├── scenarios/
│   ├── scenario_000/
│   │   ├── input          (31, 5, 60, 40, 6) float32 normalized
│   │   └── target         (31, 3, 60, 40, 6) float32 normalized
│   ├── scenario_001/
│   ├── ...
│   └── scenario_032/         # 33 scenarios total (D-024 all-train)
├── mask                       (60, 40, 6) float32
└── metadata/
    ├── train_indices          (33,) int32
    ├── val_indices            (0,)   int32  # D-024 empty
    ├── ood_indices            (0,)   int32  # OOD 는 별도 data/raw/sim_*_T*
    └── scenario_ids           list of str
```

### 3.2 채널 layout

**Input (5 channels)**:
| Idx | 이름 | 정규화 |
|---|---|---|
| 0 | Temperature | `(T − 20) / 1180`, clip [0, 1] |
| 1 | Visibility | `1 − clip(V / 30, 0, 1)` (INVERSE) |
| 2 | CO | `log1p(CO) / log1p(5000)`, clip [0, 1] |
| 3 | mask | 1.0 fluid / 0.0 solid |
| 4 | time_enc | `t / 300` (broadcast 공간 전체) |

**Target (3 channels)**: [T, V, CO] — input 의 [0,1,2] 와 동일.

**Convention**: 모든 채널 [0, 1], **higher = more dangerous**.

### 3.3 데이터 통계 (33 train)

| 항목 | 값 |
|---|---|
| Pairs | 990 (33 시나리오 × 30 pair) |
| Mask solid 비율 | 24.3% (3,494 / 14,400 cells) |
| 모든 split | "train" (D-024) |

---

## 4. OOD 시나리오 — Raw FDS

13 OOD 시나리오는 dataset.h5 에 포함되지 않음. 평가 시:

```python
from src.data_pipeline.fds_extractor import extract_slices
slices = extract_slices(Path("data/raw/sim_1500kw_2m2_T05"))
# {"temperature": (31, 60, 40, 6), "visibility": ..., "co": ...}
```

→ 평가 스크립트들이 raw 디렉터리를 직접 읽음.

---

## 5. 데이터 인덱스 — 디렉터리

| 경로 | 내용 | 크기 |
|---|---|---|
| `data/raw/scenario_config.json` | 33 시나리오 메타 (gitignored exception) | 20 KB |
| `data/raw/s_NNN/` (33 dirs) | training FDS outputs | 시나리오당 ~150 MB |
| `data/raw/sim_*_T*/` (13 dirs) | OOD FDS outputs | 시나리오당 ~150 MB |
| `data/raw/first_sim/` | 초기 검증용 (legacy) | ~150 MB |
| `data/processed/dataset.h5` | 33 training pairs | 221 MB |
| `results/detector_sequences/<scenario>.npz` (46 files) | Tier 1 GNN 입력 | 각 ~6 KB |

총 raw FDS data: **5.7 GB** (gitignored, Drive/S3 별도 전송 필요).

---

## 6. 데이터 생성 절차 (재현)

```bash
# Step 1: FDS 시나리오 .fds 파일 생성
python scripts/generate_scenarios.py
# → data/raw/scenario_config.json + .fds template instances

# Step 2: PyroSim export 후 버그 패치
python scripts/fix_pyrosim_fds.py data/raw/s_000

# Step 3: FDS 실행 (각 시나리오 ~23 min CPU)
fds  data/raw/s_000/sim_*.fds   # ... 33회

# Step 4: dataset.h5 빌드
python -m src.data_pipeline.build_dataset \
    --raw data/raw --output data/processed/dataset.h5

# Step 5: Tier 1 binary sequences
python scripts/build_detector_sequences.py
# → results/detector_sequences/*.npz (46 files: 33 train + 13 OOD)
```

---

## 7. 알려진 함정 (Lessons)

| L# | 함정 | 대응 |
|---|---|---|
| L-001 | `VECTOR=.TRUE.` on SLCF → fdsreader 깨짐 | `fix_pyrosim_fds.py` 자동 제거 |
| L-009 | PyroSim 자동 Z=3.5 → broadcast error | 동일 스크립트로 Z=3.0 수정 |
| L-012 | cell-centered SLCF 와 fdsreader 비호환 | 자체 raw `.sf` parser (`fds_extractor.py`) |
| L-013+ | (cold-start, detector trigger, sparse gap 등 추가 발견 — `lessons_learned.md` 참조) |
