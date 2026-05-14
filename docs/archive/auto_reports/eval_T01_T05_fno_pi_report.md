# ConvLSTM OOD 평가 — T01-T05 위치 보고서

> **평가 시점**: 자동 생성  

> **시나리오 수**: 13  

> **모델**: ConvLSTM (`checkpoints/conv_lstm/best.pt`)  

> **건물**: training 과 동일 (MESH/SLCF spec 일치) → **위치 OOD only**


---

## 0. 한 줄 요약

새 5개 화재 위치(T01-T05) × 13개 HRR/면적 조합에 대해 ConvLSTM 의 single-step RelL2 평균 **0.157**, 6-step autoregress RelL2 **0.818**, 위험도 맵 IoU **0.836** / FNR **6.9%**, 추론 속도 **30.3 ms** (CPU 기준).


## 1. 집계 결과

![aggregated](figures/eval_T01_T05_fno_pi/aggregated_boxplots.png)


### 1.1 전체 시나리오 단일 표

| name | loc | HRR | area | RelL2 | RMSE °C | RMSE m | RMSE ppm | auto-6 | IoU | FNR | infer ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | T01 | 1000 | 1 m² | 0.161 | 6.64 | 1.52 | 9.0 | 0.775 | 0.796 | 3.0% | 25.3 |
| sim_1000kw_1m2_T03 | T03 | 1000 | 1 m² | 0.178 | 7.09 | 1.84 | 8.9 | 0.833 | 0.810 | 2.9% | 25.0 |
| sim_1000kw_2m2_T01 | T01 | 1000 | 2 m² | 0.155 | 10.11 | 1.76 | 11.7 | 0.822 | 0.887 | 5.1% | 25.9 |
| sim_1000kw_2m2_T05 | T05 | 1000 | 2 m² | 0.137 | 9.60 | 1.91 | 8.8 | 0.866 | 0.883 | 10.9% | 31.0 |
| sim_1500kw_1m2_T02 | T02 | 1500 | 1 m² | 0.141 | 7.46 | 1.67 | 8.0 | 0.847 | 0.884 | 7.0% | 30.6 |
| sim_1500kw_1m2_T03 | T03 | 1500 | 1 m² | 0.167 | 7.78 | 1.91 | 8.9 | 0.837 | 0.853 | 5.9% | 31.8 |
| sim_1500kw_2m2_T05 | T05 | 1500 | 2 m² | 0.136 | 13.45 | 2.05 | 13.2 | 0.880 | 0.907 | 8.8% | 31.0 |
| sim_500kw_1m2_T01 | T01 | 500 | 1 m² | 0.184 | 4.68 | 1.51 | 6.5 | 0.750 | 0.664 | 3.5% | 30.1 |
| sim_500kw_1m2_T02 | T02 | 500 | 1 m² | 0.158 | 3.30 | 1.37 | 2.7 | 0.757 | 0.843 | 8.5% | 32.8 |
| sim_500kw_1m2_T03 | T03 | 500 | 1 m² | 0.198 | 4.80 | 1.68 | 5.0 | 0.865 | 0.795 | 4.2% | 33.4 |
| sim_500kw_1m2_T04 | T04 | 500 | 1 m² | 0.146 | 3.40 | 1.29 | 3.4 | 0.750 | 0.847 | 6.6% | 32.4 |
| sim_500kw_2m2_T02 | T02 | 500 | 2 m² | 0.143 | 5.12 | 1.51 | 5.2 | 0.819 | 0.865 | 8.2% | 31.9 |
| sim_500kw_2m2_T05 | T05 | 500 | 2 m² | 0.140 | 5.74 | 1.63 | 4.6 | 0.828 | 0.835 | 15.0% | 33.1 |


### 1.2 가설 검증 게이지

| 가설 | 목표 | OOD 측정 | 통과? |
|---|---|---|---|
| **H2** Single-step RelL2 | ≤ 0.15 | 0.157 | ❌ |
| **H4** Risk FNR | < 10% | 6.9% | ✅ |
| **H5** Risk IoU | ≥ 0.70 | 0.836 | ✅ |


---

## 2. 위치별 상세 (T01-T05)


### 2.1 T01

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1000kw_1m2_T01 | 1000 kW | 1 m² | 0.161 | 0.775 | 0.796 | 3.0% | ![](figures/eval_T01_T05_fno_pi/sim_1000kw_1m2_T01/z_slice_t180.png) |
| sim_1000kw_2m2_T01 | 1000 kW | 2 m² | 0.155 | 0.822 | 0.887 | 5.1% | ![](figures/eval_T01_T05_fno_pi/sim_1000kw_2m2_T01/z_slice_t180.png) |
| sim_500kw_1m2_T01 | 500 kW | 1 m² | 0.184 | 0.750 | 0.664 | 3.5% | ![](figures/eval_T01_T05_fno_pi/sim_500kw_1m2_T01/z_slice_t180.png) |

**Risk map snapshots** (sim_1000kw_1m2_T01):  

![](figures/eval_T01_T05_fno_pi/sim_1000kw_1m2_T01/risk_snapshots.png)

**Autoregress error** (sim_1000kw_1m2_T01):  

![](figures/eval_T01_T05_fno_pi/sim_1000kw_1m2_T01/autoreg.png)

**Animation** (sim_1000kw_1m2_T01):  

![](figures/eval_T01_T05_fno_pi/sim_1000kw_1m2_T01/risk_animation.gif)


### 2.2 T02

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1500kw_1m2_T02 | 1500 kW | 1 m² | 0.141 | 0.847 | 0.884 | 7.0% | ![](figures/eval_T01_T05_fno_pi/sim_1500kw_1m2_T02/z_slice_t180.png) |
| sim_500kw_1m2_T02 | 500 kW | 1 m² | 0.158 | 0.757 | 0.843 | 8.5% | ![](figures/eval_T01_T05_fno_pi/sim_500kw_1m2_T02/z_slice_t180.png) |
| sim_500kw_2m2_T02 | 500 kW | 2 m² | 0.143 | 0.819 | 0.865 | 8.2% | ![](figures/eval_T01_T05_fno_pi/sim_500kw_2m2_T02/z_slice_t180.png) |

**Risk map snapshots** (sim_1500kw_1m2_T02):  

![](figures/eval_T01_T05_fno_pi/sim_1500kw_1m2_T02/risk_snapshots.png)

**Autoregress error** (sim_1500kw_1m2_T02):  

![](figures/eval_T01_T05_fno_pi/sim_1500kw_1m2_T02/autoreg.png)

**Animation** (sim_1500kw_1m2_T02):  

![](figures/eval_T01_T05_fno_pi/sim_1500kw_1m2_T02/risk_animation.gif)


### 2.3 T03

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1000kw_1m2_T03 | 1000 kW | 1 m² | 0.178 | 0.833 | 0.810 | 2.9% | ![](figures/eval_T01_T05_fno_pi/sim_1000kw_1m2_T03/z_slice_t180.png) |
| sim_1500kw_1m2_T03 | 1500 kW | 1 m² | 0.167 | 0.837 | 0.853 | 5.9% | ![](figures/eval_T01_T05_fno_pi/sim_1500kw_1m2_T03/z_slice_t180.png) |
| sim_500kw_1m2_T03 | 500 kW | 1 m² | 0.198 | 0.865 | 0.795 | 4.2% | ![](figures/eval_T01_T05_fno_pi/sim_500kw_1m2_T03/z_slice_t180.png) |

**Risk map snapshots** (sim_1000kw_1m2_T03):  

![](figures/eval_T01_T05_fno_pi/sim_1000kw_1m2_T03/risk_snapshots.png)

**Autoregress error** (sim_1000kw_1m2_T03):  

![](figures/eval_T01_T05_fno_pi/sim_1000kw_1m2_T03/autoreg.png)

**Animation** (sim_1000kw_1m2_T03):  

![](figures/eval_T01_T05_fno_pi/sim_1000kw_1m2_T03/risk_animation.gif)


### 2.4 T04

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_500kw_1m2_T04 | 500 kW | 1 m² | 0.146 | 0.750 | 0.847 | 6.6% | ![](figures/eval_T01_T05_fno_pi/sim_500kw_1m2_T04/z_slice_t180.png) |

**Risk map snapshots** (sim_500kw_1m2_T04):  

![](figures/eval_T01_T05_fno_pi/sim_500kw_1m2_T04/risk_snapshots.png)

**Autoregress error** (sim_500kw_1m2_T04):  

![](figures/eval_T01_T05_fno_pi/sim_500kw_1m2_T04/autoreg.png)

**Animation** (sim_500kw_1m2_T04):  

![](figures/eval_T01_T05_fno_pi/sim_500kw_1m2_T04/risk_animation.gif)


### 2.5 T05

| 시나리오 | HRR | area | RelL2 | auto-6 | IoU | FNR | snapshot |
|---|---|---|---|---|---|---|---|
| sim_1000kw_2m2_T05 | 1000 kW | 2 m² | 0.137 | 0.866 | 0.883 | 10.9% | ![](figures/eval_T01_T05_fno_pi/sim_1000kw_2m2_T05/z_slice_t180.png) |
| sim_1500kw_2m2_T05 | 1500 kW | 2 m² | 0.136 | 0.880 | 0.907 | 8.8% | ![](figures/eval_T01_T05_fno_pi/sim_1500kw_2m2_T05/z_slice_t180.png) |
| sim_500kw_2m2_T05 | 500 kW | 2 m² | 0.140 | 0.828 | 0.835 | 15.0% | ![](figures/eval_T01_T05_fno_pi/sim_500kw_2m2_T05/z_slice_t180.png) |

**Risk map snapshots** (sim_1000kw_2m2_T05):  

![](figures/eval_T01_T05_fno_pi/sim_1000kw_2m2_T05/risk_snapshots.png)

**Autoregress error** (sim_1000kw_2m2_T05):  

![](figures/eval_T01_T05_fno_pi/sim_1000kw_2m2_T05/autoreg.png)

**Animation** (sim_1000kw_2m2_T05):  

![](figures/eval_T01_T05_fno_pi/sim_1000kw_2m2_T05/risk_animation.gif)


---

## 3. 비교 — Training 평균 vs OOD 평균

참고: training (33 scen) 평가 결과는 handoff §3 의 ConvLSTM eval 수치.


| 항목 | Training (33) | OOD T01-T05 (13) |
|---|---|---|
| Single-step RelL2 | ≈ 0.115-0.158 | **0.157** |
| Autoreg 6-step RelL2 | ≈ 0.093 | **0.818** |
| Risk IoU | ≈ 0.85 | **0.836** |
| Risk FNR | ≈ 9.9% | **6.9%** |
| Infer time (CPU) | ≈ 26.7 ms | **30.3 ms** |

> 차이를 보면 **위치 일반화 능력**(H3 의 indirect 신호)을 가늠할 수 있음.

> H3 의 정식 검증은 별도 OOD 시뮬 (Member A) 도착 후 FNO vs ConvLSTM 비교로 진행.
