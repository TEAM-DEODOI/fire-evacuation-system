# ConvLSTM OOD 평가 — T01-T05 위치 보고서

> **평가 시점**: 자동 생성  

> **시나리오 수**: 13  

> **모델**: ConvLSTM (`checkpoints/conv_lstm/best.pt`)  

> **건물**: training 과 동일 (MESH/SLCF spec 일치) → **위치 OOD only**


---

## 0. 한 줄 요약

새 5개 화재 위치(T01-T05) × 13개 HRR/면적 조합에 대해 ConvLSTM 의 single-step RelL2 평균 **0.138**, 6-step autoregress RelL2 **0.770**, 위험도 맵 IoU **0.828** / FNR **7.2%**, 추론 속도 **27.1 ms** (CPU 기준).


## 1. 집계 결과

![aggregated](figures/eval_T01_T05_fno_no_pi/aggregated_boxplots.png)


### 1.1 전체 시나리오 단일 표

| name | loc | HRR | area | RelL2 | RMSE °C | RMSE m | RMSE ppm | auto-6 | IoU | FNR | infer ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | T01 | 1000 | 1 m² | 0.143 | 6.92 | 1.28 | 5.0 | 0.707 | 0.776 | 3.7% | 25.1 |
| sim_1000kw_1m2_T03 | T03 | 1000 | 1 m² | 0.161 | 7.68 | 1.59 | 7.0 | 0.807 | 0.821 | 3.1% | 25.3 |
| sim_1000kw_2m2_T01 | T01 | 1000 | 2 m² | 0.137 | 10.69 | 1.49 | 8.3 | 0.767 | 0.887 | 5.8% | 25.6 |
| sim_1000kw_2m2_T05 | T05 | 1000 | 2 m² | 0.117 | 9.70 | 1.57 | 5.0 | 0.822 | 0.869 | 12.4% | 24.9 |
| sim_1500kw_1m2_T02 | T02 | 1500 | 1 m² | 0.120 | 6.55 | 1.36 | 4.5 | 0.800 | 0.893 | 6.7% | 25.8 |
| sim_1500kw_1m2_T03 | T03 | 1500 | 1 m² | 0.147 | 8.05 | 1.61 | 6.3 | 0.803 | 0.857 | 6.3% | 24.5 |
| sim_1500kw_2m2_T05 | T05 | 1500 | 2 m² | 0.116 | 13.66 | 1.69 | 7.8 | 0.842 | 0.902 | 9.4% | 25.7 |
| sim_500kw_1m2_T01 | T01 | 500 | 1 m² | 0.167 | 5.13 | 1.29 | 3.3 | 0.706 | 0.621 | 2.6% | 30.0 |
| sim_500kw_1m2_T02 | T02 | 500 | 1 m² | 0.135 | 3.16 | 1.10 | 1.5 | 0.707 | 0.839 | 8.8% | 28.5 |
| sim_500kw_1m2_T03 | T03 | 500 | 1 m² | 0.180 | 5.34 | 1.46 | 3.9 | 0.851 | 0.779 | 3.8% | 29.5 |
| sim_500kw_1m2_T04 | T04 | 500 | 1 m² | 0.127 | 3.34 | 1.05 | 1.7 | 0.658 | 0.830 | 5.8% | 29.6 |
| sim_500kw_2m2_T02 | T02 | 500 | 2 m² | 0.122 | 4.72 | 1.22 | 3.0 | 0.769 | 0.875 | 8.0% | 28.5 |
| sim_500kw_2m2_T05 | T05 | 500 | 2 m² | 0.119 | 5.79 | 1.33 | 2.7 | 0.773 | 0.816 | 16.5% | 29.5 |


### 1.2 가설 검증 게이지

| 가설 | 목표 | OOD 측정 | 통과? |
|---|---|---|---|
| **H2** Single-step RelL2 | ≤ 0.15 | 0.138 | ✅ |
| **H4** Risk FNR | < 10% | 7.2% | ✅ |
| **H5** Risk IoU | ≥ 0.70 | 0.828 | ✅ |


---

## 2. 위치별 상세 (T01-T05)


### 2.1 T01

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | 1000 kW | 1 m² | 0.143 | 0.707 | 0.776 | 3.7% | ![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_1m2_T01/z_slice_t180.png) |
| sim_1000kw_2m2_T01 | 1000 kW | 2 m² | 0.137 | 0.767 | 0.887 | 5.8% | ![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_2m2_T01/z_slice_t180.png) |
| sim_500kw_1m2_T01 | 500 kW | 1 m² | 0.167 | 0.706 | 0.621 | 2.6% | ![](figures/eval_T01_T05_fno_no_pi/sim_500kw_1m2_T01/z_slice_t180.png) |

**Risk map snapshots** (sim_1000kw_1m2_T01):  

![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_1m2_T01/risk_snapshots.png)

**Autoregress error** (sim_1000kw_1m2_T01):  

![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_1m2_T01/autoreg.png)

**Animation** (sim_1000kw_1m2_T01):  

![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_1m2_T01/risk_animation.gif)


### 2.2 T02

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1500kw_1m2_T02 | 1500 kW | 1 m² | 0.120 | 0.800 | 0.893 | 6.7% | ![](figures/eval_T01_T05_fno_no_pi/sim_1500kw_1m2_T02/z_slice_t180.png) |
| sim_500kw_1m2_T02 | 500 kW | 1 m² | 0.135 | 0.707 | 0.839 | 8.8% | ![](figures/eval_T01_T05_fno_no_pi/sim_500kw_1m2_T02/z_slice_t180.png) |
| sim_500kw_2m2_T02 | 500 kW | 2 m² | 0.122 | 0.769 | 0.875 | 8.0% | ![](figures/eval_T01_T05_fno_no_pi/sim_500kw_2m2_T02/z_slice_t180.png) |

**Risk map snapshots** (sim_1500kw_1m2_T02):  

![](figures/eval_T01_T05_fno_no_pi/sim_1500kw_1m2_T02/risk_snapshots.png)

**Autoregress error** (sim_1500kw_1m2_T02):  

![](figures/eval_T01_T05_fno_no_pi/sim_1500kw_1m2_T02/autoreg.png)

**Animation** (sim_1500kw_1m2_T02):  

![](figures/eval_T01_T05_fno_no_pi/sim_1500kw_1m2_T02/risk_animation.gif)


### 2.3 T03

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1000kw_1m2_T03 | 1000 kW | 1 m² | 0.161 | 0.807 | 0.821 | 3.1% | ![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_1m2_T03/z_slice_t180.png) |
| sim_1500kw_1m2_T03 | 1500 kW | 1 m² | 0.147 | 0.803 | 0.857 | 6.3% | ![](figures/eval_T01_T05_fno_no_pi/sim_1500kw_1m2_T03/z_slice_t180.png) |
| sim_500kw_1m2_T03 | 500 kW | 1 m² | 0.180 | 0.851 | 0.779 | 3.8% | ![](figures/eval_T01_T05_fno_no_pi/sim_500kw_1m2_T03/z_slice_t180.png) |

**Risk map snapshots** (sim_1000kw_1m2_T03):  

![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_1m2_T03/risk_snapshots.png)

**Autoregress error** (sim_1000kw_1m2_T03):  

![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_1m2_T03/autoreg.png)

**Animation** (sim_1000kw_1m2_T03):  

![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_1m2_T03/risk_animation.gif)


### 2.4 T04

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_500kw_1m2_T04 | 500 kW | 1 m² | 0.127 | 0.658 | 0.830 | 5.8% | ![](figures/eval_T01_T05_fno_no_pi/sim_500kw_1m2_T04/z_slice_t180.png) |

**Risk map snapshots** (sim_500kw_1m2_T04):  

![](figures/eval_T01_T05_fno_no_pi/sim_500kw_1m2_T04/risk_snapshots.png)

**Autoregress error** (sim_500kw_1m2_T04):  

![](figures/eval_T01_T05_fno_no_pi/sim_500kw_1m2_T04/autoreg.png)

**Animation** (sim_500kw_1m2_T04):  

![](figures/eval_T01_T05_fno_no_pi/sim_500kw_1m2_T04/risk_animation.gif)


### 2.5 T05

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1000kw_2m2_T05 | 1000 kW | 2 m² | 0.117 | 0.822 | 0.869 | 12.4% | ![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_2m2_T05/z_slice_t180.png) |
| sim_1500kw_2m2_T05 | 1500 kW | 2 m² | 0.116 | 0.842 | 0.902 | 9.4% | ![](figures/eval_T01_T05_fno_no_pi/sim_1500kw_2m2_T05/z_slice_t180.png) |
| sim_500kw_2m2_T05 | 500 kW | 2 m² | 0.119 | 0.773 | 0.816 | 16.5% | ![](figures/eval_T01_T05_fno_no_pi/sim_500kw_2m2_T05/z_slice_t180.png) |

**Risk map snapshots** (sim_1000kw_2m2_T05):  

![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_2m2_T05/risk_snapshots.png)

**Autoregress error** (sim_1000kw_2m2_T05):  

![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_2m2_T05/autoreg.png)

**Animation** (sim_1000kw_2m2_T05):  

![](figures/eval_T01_T05_fno_no_pi/sim_1000kw_2m2_T05/risk_animation.gif)


---

## 3. 비교 — Training 평균 vs OOD 평균

참고: training (33 scen) 평가 결과는 handoff §3 의 ConvLSTM eval 수치.


| 항목 | Training (33) | OOD T01-T05 (13) |
|---|---|---|
| Single-step RelL2 | ≈ 0.115-0.158 | **0.138** |
| Autoreg 6-step RelL2 | ≈ 0.093 | **0.770** |
| Risk IoU | ≈ 0.85 | **0.828** |
| Risk FNR | ≈ 9.9% | **7.2%** |
| Infer time (CPU) | ≈ 26.7 ms | **27.1 ms** |

> 차이를 보면 **위치 일반화 능력**(H3 의 indirect 신호)을 가늠할 수 있음.

> H3 의 정식 검증은 별도 OOD 시뮬 (Member A) 도착 후 FNO vs ConvLSTM 비교로 진행.
