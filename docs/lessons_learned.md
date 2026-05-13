# Lessons Learned

> Concrete bugs, gotchas, and surprises encountered during the project.
> Each entry documents the symptom, root cause, and fix to prevent
> repeat occurrences.
>
> **Append-only**. When new bugs are found and fixed, add `L-NNN` entries.

---

## L-001: fdsreader broadcast error on `VECTOR=.TRUE.` slices

**Symptom**:

```
ValueError: could not broadcast input array from shape (61,41,8)
into shape (61,41,7)
```

When calling `slc.to_global(return_coordinates=True)` on a SLCF that
was generated with both `VECTOR=.TRUE.` and `CELL_CENTERED=.TRUE.`.

**Root cause**:

`fdsreader` cannot correctly compute the cell-centered slice dimensions
when `VECTOR=.TRUE.` is set on the FDS SLCF. There appears to be an
off-by-one or stride mismatch internally. This is a known limitation
of the `fdsreader` library and not a bug in our code.

**Fix**:

Always omit `VECTOR=.TRUE.` from SLCF lines used by ML pipeline.
We only need scalar fields (T, V, CO), not vector fields like velocity.

```fortran
# DO NOT USE:
&SLCF QUANTITY='TEMPERATURE',
      VECTOR=.TRUE.,            ! <-- this line breaks fdsreader
      CELL_CENTERED=.TRUE.,
      ID='Temperature', XB=...

# USE:
&SLCF QUANTITY='TEMPERATURE',
      CELL_CENTERED=.TRUE.,
      ID='Temperature',
      XB=0.0,30.0, 0.0,20.0, 0.0,3.0/
```

**Status**: Fixed. Documented in `CLAUDE.md` and `coordinate_convention.md`.

---

## L-002: FDS MESH must be larger than SLCF region

**Symptom**:

When MESH `XB` was set to (0, 30, 0, 20, 0, 3), smoke and heat near the
end-exit doors behaved oddly — temperatures dropped sharply right at
the door, gradients looked wrong.

**Root cause**:

FDS treats MESH boundaries as walls unless explicitly told otherwise.
A door defined right at the MESH edge has nowhere for smoke to escape.

**Fix**:

Extend MESH 10 m beyond building footprint on each side:
- MESH XB: `[-10, 40] × [-10, 30] × [0, 4]`
- SLCF XB: `[0, 30] × [0, 20] × [0, 3]`  (building only)
- Models train on the SLCF region; the buffer is invisible to ML.

**Status**: Fixed. Documented in `CLAUDE.md` and `coordinate_convention.md`.

---

## L-003: Visibility direction is opposite of T and CO

**Symptom**:

Initial training run had loss decreasing for T and CO channels but
*increasing* for visibility. Later, peak danger error gave wrong
direction in the test set.

**Root cause**:

Raw visibility in metres: **higher value is safer** (you can see further).
Raw temperature: **higher value is more dangerous**.
If both are normalised the same way, the loss function is fighting itself.

**Fix**:

Normalise visibility with **inverse mapping**:

```python
V_norm = 1 - np.clip(V_metres / 30.0, 0, 1)
```

So high raw visibility (safe) maps to low normalised value (≈ 0.0),
matching the convention "high normalised = dangerous" used for T and CO.

**Status**: Fixed. Documented prominently in `CLAUDE.md`,
`coordinate_convention.md`, and `interface_contracts.md`.

---

## L-004: Cell-centered indexing offset by 0.25 m, not 0

**Symptom**:

Drone positions queried at world coordinate (0, 0, 0) returned `nan`
or 1.0 (out-of-bounds), even though that's the corner of the building.

**Root cause**:

Cell (0, 0, 0) of the SLCF grid has its **centre** at world (0.25, 0.25, 0.25),
not at (0, 0, 0). Any query at (0, 0, 0) is technically outside the
last-cell-bounds of the interpolator, so the `bounds_error=False`
fallback kicks in.

**Fix**:

When converting world → grid index, account for the cell-centered offset:

```python
# WRONG:
ix = int(world_x / 0.5)

# CORRECT:
ix = int((world_x - 0.25) / 0.5)
```

Use `scipy.interpolate.RegularGridInterpolator` constructed from
the actual cell centres returned by `fdsreader.to_global(return_coordinates=True)`,
which automatically handles the offset.

**Status**: Documented. Pattern enshrined in `coordinate_convention.md`.

---

## L-005: HDF5 mask should be float, not bool

**Symptom**:

When loading the building mask via `h5py`, broadcasting it into a
PyTorch tensor produced `dtype=torch.uint8` and silently ruined
gradient flow when used as input.

**Root cause**:

PyTorch tensors with `uint8` dtype cannot be cast to `float32` via
`.float()` if the source HDF5 dataset was stored as `bool`. The result
is technically valid but loses gradients.

**Fix**:

Save the mask as `np.float32` directly:

```python
mask = mask_array.astype(np.float32)  # 0.0 or 1.0
hf.create_dataset("mask", data=mask)
```

Then reading it produces `float32` tensors directly.

**Status**: Captured as a convention in `interface_contracts.md`.

---

## L-006: PyTorch DataLoader with num_workers > 0 needs `if __name__ == '__main__'`

**Symptom**:

On Windows or in some Linux configurations, training with
`DataLoader(num_workers=4)` resulted in:

```
RuntimeError: An attempt has been made to start a new process before
the current process has finished its bootstrapping phase.
```

**Root cause**:

DataLoader workers are spawned via `multiprocessing.spawn` on Windows.
Without `if __name__ == '__main__'`, the child process re-imports the
main script, recursively spawning workers.

**Fix**:

Always wrap training entry points:

```python
if __name__ == '__main__':
    train_main()
```

This is already required by our coding conventions, so it's
double-enforced.

**Status**: Fixed by convention, validated in `train_conv_lstm.py` template.

---

## L-007: `to_global()` time alignment is approximate

**Symptom**:

`temp_slc.times` returned values like `[0.0, 10.001, 19.998, 30.002, ...]`
rather than exactly `[0, 10, 20, 30, ...]`. Indexing by computed time
(e.g., `slc[15]` for t=150) returned the wrong frame in some cases.

**Root cause**:

FDS time stepping is adaptive (CFL-controlled). `DT_SLCF` controls
output frequency but actual output times can drift slightly.

**Fix**:

Use `np.searchsorted` to find the closest output index:

```python
target_times = np.arange(0, 301, 10)  # [0, 10, ..., 300]
indices = np.searchsorted(temp_slc.times, target_times)
indices = np.clip(indices, 0, len(temp_slc.times) - 1)
grid_aligned = grid[indices]  # shape (31, 60, 40, 6)
```

**Status**: Documented in `coordinate_convention.md` `to_global()` pattern.

---

## L-008: PyBullet drone collision with floor

**Symptom** (anticipated for Week 12):

Drone falls through the floor at simulation start.

**Likely root cause**:

The Crazyflie URDF has a small collision radius. PyBullet's default
contact margin is 0.001 m. If drone spawn height is < 0.001 m, it can
phase through the floor.

**Planned fix**:

Spawn drone at z = 0.5 m (above the floor by 0.5 m) and explicitly
set its mass to a small value. Use `setAdditionalSearchPath` to load
URDF correctly.

**Status**: Open. Will be addressed in Week 12 when integration begins.

---

## L-009: SLCF Z range must be exactly (0, 3), not (0, 3.5)

**Symptom**:

```
ValueError: could not broadcast input array from shape (61,41,9)
into shape (61,41,8)
```

Plus `n_t` reported as 6 instead of 31 — silent data loss.

**Root cause**:

PyroSim auto-set SLCF Z range to `0.0, 3.5` (rounded up to accommodate
STL height of 3.2 m). With 0.5 m resolution, this produces 9 z-cells,
but our `(60, 40, 6)` interface expects 6 z-cells. The mismatch causes
fdsreader to silently lose time frames and then raise a broadcast error.

**Fix**:

Manually set SLCF Z range to exactly `0.0, 3.0` in PyroSim:

```fortran
&SLCF QUANTITY='TEMPERATURE',
      CELL_CENTERED=.TRUE.,
      ID='Temperature',
      XB=0.0,30.0, 0.0,20.0, 0.0,3.0/      ← must be 3.0, not 3.5
```

This must be done for all 3 SLCF slices (Temperature, Visibility, CO).

**Why we keep STL height at 3.2 m**: The physical building shape is
preserved. Only the SLCF extraction window is clamped to 0–3 m. The
0.2 m above is the hottest smoke layer but not relevant to occupant
breathing-zone analysis. See decision D-015.

**Status**: Fixed. Documented in `CLAUDE.md` "FDS Input File Conventions"
section, `coordinate_convention.md`, and decision D-015.

---

## L-010: STL files in millimetres, not metres

**Symptom**:

When importing the STL into PyroSim, building bounding box was reported
as 30,000 m × 18,270 m × 3,200 m (instead of 30 m × 18.27 m × 3.2 m).
FDS simulation ran but produced nonsensical fire spread.

**Root cause**:

The STL was created in CAD software using millimetres as the default unit.
PyroSim defaults to metres, so the values are interpreted directly without
unit conversion.

**Fix**:

Two options:

**(A) PyroSim direct fix**: When importing STL, set the import unit to
"Millimeter" so PyroSim performs the 1/1000 scale conversion.

**(B) Pre-convert STL**: Use the conversion script in
`scripts/convert_stl_units.py` (uses `numpy-stl`):

```python
import numpy as np
from stl import mesh

m = mesh.Mesh.from_file("SCIENCE_HALL_LV5.stl")
m.vectors *= 0.001  # mm → m
m.translate([-x_min, -y_min, -z_min])  # translate to origin
m.save("SCIENCE_HALL_FIXED.stl")
```

Always verify after fix:

```python
print(f"X range: {m.vectors[:, :, 0].min():.2f} to {m.vectors[:, :, 0].max():.2f}")
# Expected: 0.0 to ~30.0
```

**Status**: Fixed for current STL. Future STLs from CAD must be checked
for units before PyroSim import.

---

## L-011: PI-FNO autoregression error compounds

**Symptom** (anticipated for Week 9):

After 3-4 autoregressive steps, predictions drift away from physical
plausibility — temperatures may go negative, CO concentrations may
oscillate.

**Root cause**:

Each prediction step has small errors. Feeding these errors back as
input to the next step compounds them. Beyond 3-4 steps (30-40 s),
the model is operating outside its training distribution.

**Fix** (planned):

1. Limit autoregressive horizon to ~3 steps for direct use, 6 steps
   only for the dynamic risk map (where it's only used for path
   planning, not safety-critical evacuation decisions).
2. Add monotonicity constraint on tenability boundary (Stage 4 PI loss)
   to discourage non-physical oscillations.
3. Re-clip outputs to [0, 1] explicitly between steps.
4. Document predictions beyond 30s as "best-effort estimates" rather
   than reliable guidance.

**Status**: Planned. Will be addressed in Week 9 when PI-FNO is trained
and Week 10 when DynamicRiskMap is built.

---

## L-012: PyroSim cell-centered SLCF is incompatible with fdsreader

**Symptom**:

```
ValueError: could not broadcast input array from shape (61,41,8)
into shape (61,41,7)
```

The error appears when ``fdsreader.Slice.to_global(return_coordinates=True)``
is called on a cell-centered SLCF, even when the FDS input deck sets
``XB=...,0.0,3.0/`` (Z=3.0). This is *distinct* from L-009 — fixing the
``.fds`` deck with ``scripts/fix_pyrosim_fds.py`` does not eliminate the
problem when the source of the slice is PyroSim.

**Root cause**:

PyroSim writes cell-centered SLCF outputs as **node-based** ``(62, 42, 8)``
arrays inside the ``.sf`` file and additionally stretches the Z axis to
3.5 m (so cell shape becomes ``(61, 41, 7)`` rather than the canonical
``(60, 40, 6)``). ``fdsreader`` then expects a ``(61, 41, 7)`` payload
but receives ``(61, 41, 8)`` after its own cell-centering pass,
producing the broadcast error.

**Fix**:

Bypass ``fdsreader``. ``src/data_pipeline/fds_extractor.py`` parses the
``.sf`` FORTRAN-unformatted records directly, applies 8-vertex averaging
to convert node → cell-centered, then crops the result to the
``(60, 40, 6)`` SLCF region per D-015. The same module also converts
CO from FDS ``VOLUME FRACTION`` (mol/mol) to ppm and aligns the time
axis to the canonical 31 frames.

**Status**: Fixed by custom parser. Applies to all 30 scenarios. No
``fdsreader`` import remains in ``fds_extractor.py``; the dependency
stays in ``requirements.txt`` only for any future, non-cell-centered
diagnostic use.

---

## L-013: Sparse-input model autoregress distribution shift

**Symptom**: Sparse-input ConvLSTM (L4e, 39-sensor) 의 60s autoregress 가
t₀+10s 에서는 정확하다가 시간 진행할수록 도메인 전체가 빨강 (danger ≥ 0.5) 으로
saturate. IoU 0.18, FNR 0% (모든 cell 위험 예측).

**Root cause**: 모델은 **(sparse input → dense target)** 매핑으로 학습됐는데,
naïve autoregress 시 **(dense output → 다음 dense output)** 으로 chaining.
즉 inference 시 학습 분포 밖의 input distribution 으로 진입 → drift 누적 →
"everywhere dangerous" 가 minimum-MSE local optimum 으로 수렴.

**Fix**: `autoregress_sparse(..., resparsify=True)` — 매 step 모델 출력의
sensor 위치 외 cell T/V/CO 채널을 0 으로 강제. 결과: IoU 0.182 → **0.581**
(3.2× 향상). 실제 deployment 와도 일치 (매 10s 마다 sensor measurement update).

**Status**: Fixed via inference-time re-sparsify. `evaluate_sparse_model.py`
의 `--resparsify` flag 가 default 로 False 였음 — 이게 conservative bias 의
숨은 원인. `visualize_60s_5model.py` 도 동일하게 갱신.

**Cross-ref**: `docs/40_tier2_models_continuous.md §6.2`, `docs/decisions.md D-025`.

---

## L-014: conda-forge pybullet 설치가 pip numpy 를 깨뜨림 (numpy ABI 충돌)

**Symptom**: `conda install -n <env> -c conda-forge pybullet` 직후 모든
numpy 호출이 즉시 crash. 증상은 두 단계:

1. 단순 `import numpy as np; np.zeros(3).sum()` 가 exit code
   `0xC0000409` (`STATUS_STACK_BUFFER_OVERRUN`) 로 종료. 출력은 없음.
2. `import pybullet` 시 banner `pybullet build time: Oct 21 2025 ...` 출력 후
   `AttributeError: module 'numpy._globals' has no attribute
   '_signature_descriptor'` + `ImportError: numpy._core.multiarray failed
   to import` 발생.

**Root cause**: conda-forge 의 최신 `pybullet 3.25 py311h*_5` / `_4` 빌드는
**numpy ≥ 2.0** ABI 로 컴파일됨. 우리 env 는 `requirements.txt` 의 핀
(`torch 2.0.1` → `numpy<2`) 때문에 numpy **1.26.4** 가 pip 로 설치돼 있음.
conda 의 dependency solver 가 pybullet 설치 시 numpy 라이브러리 파일을
부분적으로 덮어쓰지만 metadata 는 `pypi_0 pypi` 로 남겨, 결과적으로 numpy
설치가 **불완전·일관성 깨진 상태**가 됨. numpy 자체 import 도 crash.

torch 2.0.1 (2023-06) 은 numpy 2.0 (2024-06) 이전 릴리스라 numpy 2.x 와는
호환 불가 — env-wide 로 numpy 를 2.x 로 올리는 옵션은 ML 파이프라인 전체가
망가지므로 차단됨.

**Fix** (재현 가능한 절차):

```bash
# 1) numpy 1.x ABI 로 빌드된 더 오래된 pybullet 빌드 명시 (_3 또는 그 이전)
conda install -n fire-evac -c conda-forge --override-channels \
    pybullet=3.25=py311hbc92ba2_3 -y

# 2) conda 가 부분 덮어쓴 pip numpy 를 강제 재설치 → ABI 복구
<env-python> -m pip install --force-reinstall --no-deps numpy==1.26.4

# 3) 검증
<env-python> -c "import numpy as np; print(np.zeros(3).sum())"
<env-python> -c "import pybullet as p; p.connect(p.DIRECT); p.disconnect()"
```

`conda search -c conda-forge pybullet=3.25=py311* --info` 로 빌드별 numpy
범위를 직접 확인 가능. numpy<2.0a0 으로 빌드된 hash 만 사용할 것:
`_0 / _1 / _2 / _3` ✅, `_4 / _5` ❌.

**Status**: Fixed for fire-evac env (2026-05-14). Conda+pip mixing 의
일반적 위험으로, 향후 pybullet upgrade 시 동일 절차 적용.

requirements.txt 의 `pybullet==3.2.6` 핀은 Windows Python 3.11 wheel 부재로
pip 설치 불가 — conda-forge `pybullet=3.25=py311hbc92ba2_3` 가 사실상의
대체. 다음 requirements.txt 정리 시 반영 권장.

**Cross-ref**: `docs/decisions.md D-025` (Week 12 PyBullet scope),
`requirements.txt` (`pybullet==3.2.6` 핀 stale), `docs/pybullet_integration_spec.md §7`
(환경/의존성 절 — 갱신 필요).

---

## How to Add a Lesson

When you encounter and fix a bug worth remembering:

1. Add a new section labeled `L-NNN`.
2. **Symptom** (literal error or behaviour observed).
3. **Root cause** (why it happened).
4. **Fix** (the actual change that resolved it).
5. **Status** (Fixed / Open / Workaround / Planned).
6. Cross-reference docs that were updated.

Keep entries concise — 3–8 sentences each. Append-only.
