# ConvLSTM OOD 평가 — T01-T05 위치 보고서

> **평가 시점**: 자동 생성  

> **시나리오 수**: 13  

> **모델**: ConvLSTM (`checkpoints/conv_lstm/best.pt`)  

> **건물**: training 과 동일 (MESH/SLCF spec 일치) → **위치 OOD only**


---

## 0. 한 줄 요약

새 5개 화재 위치(T01-T05) × 13개 HRR/면적 조합에 대해 ConvLSTM 의 single-step RelL2 평균 **0.136**, 6-step autoregress RelL2 **0.882**, 위험도 맵 IoU **0.887** / FNR **6.0%**, 추론 속도 **26.5 ms** (CPU 기준).


## 1. 집계 결과

![aggregated](figures/eval_T01_T05/aggregated_boxplots.png)


### 1.1 전체 시나리오 단일 표

| name | loc | HRR | area | RelL2 | RMSE °C | RMSE m | RMSE ppm | auto-6 | IoU | FNR | infer ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | T01 | 1000 | 1 m² | 0.142 | 5.55 | 1.27 | 5.5 | 0.959 | 0.865 | 6.3% | 26.7 |
| sim_1000kw_1m2_T03 | T03 | 1000 | 1 m² | 0.144 | 6.44 | 1.38 | 5.0 | 1.096 | 0.892 | 5.1% | 26.6 |
| sim_1000kw_2m2_T01 | T01 | 1000 | 2 m² | 0.139 | 7.48 | 1.52 | 7.0 | 0.677 | 0.893 | 5.8% | 26.4 |
| sim_1000kw_2m2_T05 | T05 | 1000 | 2 m² | 0.118 | 7.40 | 1.53 | 4.7 | 0.432 | 0.933 | 5.1% | 26.4 |
| sim_1500kw_1m2_T02 | T02 | 1500 | 1 m² | 0.124 | 6.58 | 1.37 | 4.6 | 0.512 | 0.915 | 4.4% | 26.7 |
| sim_1500kw_1m2_T03 | T03 | 1500 | 1 m² | 0.139 | 7.08 | 1.50 | 5.2 | 0.889 | 0.894 | 6.0% | 26.2 |
| sim_1500kw_2m2_T05 | T05 | 1500 | 2 m² | 0.118 | 10.31 | 1.67 | 7.3 | 0.410 | 0.943 | 4.8% | 26.4 |
| sim_500kw_1m2_T01 | T01 | 500 | 1 m² | 0.158 | 3.92 | 1.22 | 3.9 | 1.426 | 0.790 | 6.3% | 26.2 |
| sim_500kw_1m2_T02 | T02 | 500 | 1 m² | 0.138 | 2.82 | 1.07 | 1.5 | 1.119 | 0.874 | 8.5% | 27.7 |
| sim_500kw_1m2_T03 | T03 | 500 | 1 m² | 0.159 | 4.06 | 1.21 | 3.0 | 1.581 | 0.875 | 6.8% | 26.1 |
| sim_500kw_1m2_T04 | T04 | 500 | 1 m² | 0.141 | 2.96 | 1.15 | 2.0 | 1.081 | 0.863 | 6.2% | 26.4 |
| sim_500kw_2m2_T02 | T02 | 500 | 2 m² | 0.127 | 4.58 | 1.22 | 3.1 | 0.676 | 0.893 | 5.6% | 26.5 |
| sim_500kw_2m2_T05 | T05 | 500 | 2 m² | 0.122 | 4.70 | 1.31 | 2.6 | 0.607 | 0.898 | 7.0% | 26.6 |


### 1.2 가설 검증 게이지

| 가설 | 목표 | OOD 측정 | 통과? |
|---|---|---|---|
| **H2** Single-step RelL2 | ≤ 0.15 | 0.136 | ✅ |
| **H4** Risk FNR | < 10% | 6.0% | ✅ |
| **H5** Risk IoU | ≥ 0.70 | 0.887 | ✅ |


---

## 2. 위치별 상세 (T01-T05)


### 2.1 T01

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | 1000 kW | 1 m² | 0.142 | 0.959 | 0.865 | 6.3% | ![](figures/eval_T01_T05/sim_1000kw_1m2_T01/z_slice_t180.png) |
| sim_1000kw_2m2_T01 | 1000 kW | 2 m² | 0.139 | 0.677 | 0.893 | 5.8% | ![](figures/eval_T01_T05/sim_1000kw_2m2_T01/z_slice_t180.png) |
| sim_500kw_1m2_T01 | 500 kW | 1 m² | 0.158 | 1.426 | 0.790 | 6.3% | ![](figures/eval_T01_T05/sim_500kw_1m2_T01/z_slice_t180.png) |

**Risk map snapshots** (sim_1000kw_1m2_T01):  

![](figures/eval_T01_T05/sim_1000kw_1m2_T01/risk_snapshots.png)

**Autoregress error** (sim_1000kw_1m2_T01):  

![](figures/eval_T01_T05/sim_1000kw_1m2_T01/autoreg.png)

**Animation** (sim_1000kw_1m2_T01):  

![](figures/eval_T01_T05/sim_1000kw_1m2_T01/risk_animation.gif)


### 2.2 T02

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1500kw_1m2_T02 | 1500 kW | 1 m² | 0.124 | 0.512 | 0.915 | 4.4% | ![](figures/eval_T01_T05/sim_1500kw_1m2_T02/z_slice_t180.png) |
| sim_500kw_1m2_T02 | 500 kW | 1 m² | 0.138 | 1.119 | 0.874 | 8.5% | ![](figures/eval_T01_T05/sim_500kw_1m2_T02/z_slice_t180.png) |
| sim_500kw_2m2_T02 | 500 kW | 2 m² | 0.127 | 0.676 | 0.893 | 5.6% | ![](figures/eval_T01_T05/sim_500kw_2m2_T02/z_slice_t180.png) |

**Risk map snapshots** (sim_1500kw_1m2_T02):  

![](figures/eval_T01_T05/sim_1500kw_1m2_T02/risk_snapshots.png)

**Autoregress error** (sim_1500kw_1m2_T02):  

![](figures/eval_T01_T05/sim_1500kw_1m2_T02/autoreg.png)

**Animation** (sim_1500kw_1m2_T02):  

![](figures/eval_T01_T05/sim_1500kw_1m2_T02/risk_animation.gif)


### 2.3 T03

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1000kw_1m2_T03 | 1000 kW | 1 m² | 0.144 | 1.096 | 0.892 | 5.1% | ![](figures/eval_T01_T05/sim_1000kw_1m2_T03/z_slice_t180.png) |
| sim_1500kw_1m2_T03 | 1500 kW | 1 m² | 0.139 | 0.889 | 0.894 | 6.0% | ![](figures/eval_T01_T05/sim_1500kw_1m2_T03/z_slice_t180.png) |
| sim_500kw_1m2_T03 | 500 kW | 1 m² | 0.159 | 1.581 | 0.875 | 6.8% | ![](figures/eval_T01_T05/sim_500kw_1m2_T03/z_slice_t180.png) |

**Risk map snapshots** (sim_1000kw_1m2_T03):  

![](figures/eval_T01_T05/sim_1000kw_1m2_T03/risk_snapshots.png)

**Autoregress error** (sim_1000kw_1m2_T03):  

![](figures/eval_T01_T05/sim_1000kw_1m2_T03/autoreg.png)

**Animation** (sim_1000kw_1m2_T03):  

![](figures/eval_T01_T05/sim_1000kw_1m2_T03/risk_animation.gif)


### 2.4 T04

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_500kw_1m2_T04 | 500 kW | 1 m² | 0.141 | 1.081 | 0.863 | 6.2% | ![](figures/eval_T01_T05/sim_500kw_1m2_T04/z_slice_t180.png) |

**Risk map snapshots** (sim_500kw_1m2_T04):  

![](figures/eval_T01_T05/sim_500kw_1m2_T04/risk_snapshots.png)

**Autoregress error** (sim_500kw_1m2_T04):  

![](figures/eval_T01_T05/sim_500kw_1m2_T04/autoreg.png)

**Animation** (sim_500kw_1m2_T04):  

![](figures/eval_T01_T05/sim_500kw_1m2_T04/risk_animation.gif)


### 2.5 T05

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1000kw_2m2_T05 | 1000 kW | 2 m² | 0.118 | 0.432 | 0.933 | 5.1% | ![](figures/eval_T01_T05/sim_1000kw_2m2_T05/z_slice_t180.png) |
| sim_1500kw_2m2_T05 | 1500 kW | 2 m² | 0.118 | 0.410 | 0.943 | 4.8% | ![](figures/eval_T01_T05/sim_1500kw_2m2_T05/z_slice_t180.png) |
| sim_500kw_2m2_T05 | 500 kW | 2 m² | 0.122 | 0.607 | 0.898 | 7.0% | ![](figures/eval_T01_T05/sim_500kw_2m2_T05/z_slice_t180.png) |

**Risk map snapshots** (sim_1000kw_2m2_T05):  

![](figures/eval_T01_T05/sim_1000kw_2m2_T05/risk_snapshots.png)

**Autoregress error** (sim_1000kw_2m2_T05):  

![](figures/eval_T01_T05/sim_1000kw_2m2_T05/autoreg.png)

**Animation** (sim_1000kw_2m2_T05):  

![](figures/eval_T01_T05/sim_1000kw_2m2_T05/risk_animation.gif)


---

## 3. 비교 — Training 평균 vs OOD 평균

참고: training (33 scen) 평가 결과는 handoff §3 의 ConvLSTM eval 수치.


| 항목 | Training (33) | OOD T01-T05 (13) |
|---|---|---|
| Single-step RelL2 | ≈ 0.115-0.158 | **0.136** |
| Autoreg 6-step RelL2 | ≈ 0.093 | **0.882** |
| Risk IoU | ≈ 0.85 | **0.887** |
| Risk FNR | ≈ 9.9% | **6.0%** |
| Infer time (CPU) | ≈ 26.7 ms | **26.5 ms** |

> 차이를 보면 **위치 일반화 능력**(H3 의 indirect 신호)을 가늠할 수 있음.

> H3 의 정식 검증은 별도 OOD 시뮬 (Member A) 도착 후 FNO vs ConvLSTM 비교로 진행.
