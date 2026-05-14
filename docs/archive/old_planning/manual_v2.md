# 14-Week Project Manual v2

> Active fire-response system: PI-FNO + Risk Map + Path Planning.
> Engineering capstone competition entry.
>
> This manual is the execution plan. `CLAUDE.md` is the constraints.
> Read `CLAUDE.md` first if you haven't.

---

## Table of Contents

1. [Project goals and hypotheses](#part-0-project-goals-and-hypotheses)
2. [Phase A: Foundation (Weeks 1–5)](#phase-a-foundation-weeks-15)
3. [Phase B: Data Pipeline (Week 6)](#phase-b-data-pipeline-week-6)
4. [Phase C: ConvLSTM Baseline (Week 7)](#phase-c-convlstm-baseline-week-7)
5. [Phase D: PI-FNO Training (Weeks 8–9)](#phase-d-pi-fno-training-weeks-89)
6. [Phase E: Risk Map and Path Planning (Weeks 10–11)](#phase-e-risk-map-and-path-planning-weeks-1011)
7. [Phase F: Integration and EXP-PATH-001 (Week 12)](#phase-f-integration-and-exp-path-001-week-12)
8. [Phase G: Reporting (Weeks 13–14)](#phase-g-reporting-weeks-1314)
9. [Risk register and Plan B](#risk-register-and-plan-b)
10. [References by section](#references-by-section)

---

## Part 0: Project Goals and Hypotheses

### What we are building

End-to-end pipeline for predicting fire spread and computing safer
evacuation paths. Six core hypotheses drive every line of code:

| ID | Hypothesis | Validated by |
|----|-----------|--------------|
| H1 | PI-FNO inference is ≥1000× faster than FDS | Wall-clock measurement |
| H2 | Relative L2 ≤ 15% on training scenarios | EXP-FIRE-001 (val set) |
| H3 | PI-FNO outperforms ConvLSTM on OOD | EXP-FIRE-001 (OOD set) |
| H4 | Risk map FNR <10% | EXP-RISK-001 |
| H5 | Risk map IoU ≥0.7 vs FDS ground truth | EXP-RISK-001 |
| H6 | Dynamic A* reduces FED ≥30% vs static | EXP-PATH-001 |

**Presentation order**: H1 → H6 → H4 → H2 → H5 → H3

H6 is the strongest card. Static-vs-dynamic FED comparison is intuitive
and immediately resonates with fire safety judges.

### Three named experiments

| Experiment | Compares | Metrics | Output |
|-----------|----------|---------|--------|
| **EXP-FIRE-001** | ConvLSTM vs FNO no-PI vs FNO full | RMSE, SSIM per channel; OOD generalization | `results/exp_fire_001/comparison.csv` |
| **EXP-RISK-001** | FDS ground truth vs PI-FNO risk maps | IoU @ 0.3/0.5/0.7, FNR, FPR, ASET RMSE | `results/exp_risk_001/comparison.csv` |
| **EXP-PATH-001** | Dijkstra / Static / Dynamic planners | Cumulative FED, reach time, path % through danger | `results/exp_path_001/comparison.csv` |

Plus three ablations (Week 11):
- PI loss components contribution (Stage 1/2/3/4)
- Training data size vs performance curve
- Model size variation (optional, depends on time)

### Team roles

| Member | Owns |
|--------|------|
| **A** | PyroSim modeling, FDS simulations, scenario configs |
| **B** | ConvLSTM and PI-FNO training, hyperparameters |
| **C** | Data pipeline, dataset, normalization, coordinate utilities |
| **D** | Evaluation, integration demo, visualization, paper |

Risk map and path planning (Weeks 10–11) are joint B+C work.

---

## Phase A: Foundation (Weeks 1–5)

### Week 1: Environment Setup

**Goal**: All infrastructure in place. Single FDS scenario validates.

**Tasks**:
- Set up Python venv, RunPod GPU instance, W&B project, GitHub repo
- Use Claude Code with the skeleton-creation prompt to scaffold the project
- Verify FDS + Smokeview installation
- PyroSim license verified
- Run validate-existing test cases (FDS bundled examples)

**Owner**: All members in parallel.

### Week 2: Coordinate System and FDS Template

**Goal**: Coordinate conventions locked. Dummy scenario verifies fdsreader works.

**Tasks**:
- Lock down `docs/coordinate_convention.md` with team's signature
- Implement `src/shared/coordinates.py` with full unit tests
- Implement `src/shared/constants.py`
- Verify FDS `&MESH` and SLCF `XB` produce expected `(60, 40, 6)` shape
- Create FDS template `template.fds` with placeholders for scenario variables
- Test fdsreader on a single dummy scenario: must return shape `(31, 60, 40, 6)`

**FDS template requirements**:
```
&HEAD CHID='SCENARIO_ID' /
&MESH ID='Mesh01', IJK=100,80,8, XB=-10.0,40.0,-10.0,30.0,0.0,4.0/
&TIME T_END=300.0 /
&DUMP DT_RESTART=300.0, DT_SLCF=10.0 /
&REAC ID='SFPE WOOD_OAK' ... /

! Building geometry imported from STL
&GEOM SURF_ID='WALL', BNDF_GEOM=.TRUE., ID='Building',
      BINARY_FILE='SCIENCE_HALL_v2.stl' /

! Three SLCF — NEVER add VECTOR=.TRUE.
&SLCF QUANTITY='TEMPERATURE', CELL_CENTERED=.TRUE.,
      ID='Temperature', XB=0.0,30.0, 0.0,20.0, 0.0,3.0 /
&SLCF QUANTITY='SOOT VISIBILITY', CELL_CENTERED=.TRUE.,
      ID='Visibility', XB=0.0,30.0, 0.0,20.0, 0.0,3.0 /
&SLCF QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON MONOXIDE',
      CELL_CENTERED=.TRUE., ID='CO',
      XB=0.0,30.0, 0.0,20.0, 0.0,3.0 /

! Fire source — placeholder, varies per scenario
&SURF ID='BURNER', HRRPUA={HRR_PER_AREA}, COLOR='RED' /
&OBST XB={FIRE_X1},{FIRE_X2}, {FIRE_Y1},{FIRE_Y2}, 0.0,1.0,
      SURF_ID='BURNER' /

! Two doors as ventilation openings
&VENT MB='YMIN', SURF_ID='OPEN' /
&VENT MB='YMAX', SURF_ID='OPEN' /

&TAIL /
```

**Critical**: SLCF Z range is `0.0,3.0` exactly. Not 3.2, not 3.5.
See L-009. STL building height up to 3.2 m is preserved physically;
SLCF only extracts the model-visible region.

**Owner**: A (FDS template), C (coordinates and constants).

### Week 3: Building Geometry and Graph

**Goal**: Maze-style building modeled in PyroSim. Building graph drafted
for path planning.

**Tasks**:
- Build complex maze layout in PyroSim:
  - Multiple rooms (12–16 total)
  - Multiple intersections
  - Central courtyard or hall
  - Two end exits + at least one mid-side exit (3 exits total)
  - Internal door connections — designed for non-trivial path planning
  - Designed so Dijkstra ≠ optimal for fire-aware routing
- STL must be in metres, origin at `(0, 0, 0)`
- Export `.fds` template with placeholders
- **Building graph drafted** (NetworkX, 16–20 nodes) in parallel:
  - Owner B drafts node positions (rooms, intersections, exits)
  - Owner C verifies coordinates match STL
- Test fdsreader on single template scenario

**Critical lessons from earlier mistakes**:
- L-001: Never use `VECTOR=.TRUE.` on `&SLCF`
- L-002: MESH must be larger than building footprint (10 m external buffer)
- L-009: SLCF Z range exactly `0.0, 3.0`, not 3.5

**Owner**: A (PyroSim), B (graph design), C (coord verification).

### Weeks 4–5: FDS Scenario Generation

**Goal**: 30 validated FDS scenarios on disk.

**Plan**:
- 30 scenarios = 4 HRR levels × 6 fire locations + variation
- HRR: 500, 1000, 1500, 2000 kW
- Fire locations: 6 distinct cells across the maze
- Ventilation: both end doors open (fixed across all scenarios)
- Run on RunPod CPU instances or local cluster (FDS is CPU-bound)
- Estimated wall time: ~23 minutes per scenario, ~12 hours total
  (validated empirically from `0507_v1` test run)

**Per-scenario validation** (DO NOT skip):
1. `.smv` file exists
2. 3 `.sf` files: temperature, visibility, CO
3. `fdsreader` loads cleanly: `to_global(return_coordinates=True)` succeeds
4. Output shape is `(31, 60, 40, 6)`
5. Initial temperature ≈ 22 °C across grid (TMPA setting)
6. Fire source location shows temperature rise within 30 s
7. NaN/Inf check passes

**Scenario configuration**:
Save `data/scenario_config.json` with structure:
```json
{
  "scenarios": [
    {
      "id": "scenario_000",
      "fire_xy": [11.5, 4.5],
      "hrr_kw": 500,
      "ventilation": "both_open",
      "split": "train"
    },
    ...
  ]
}
```

**Owner**: A (simulations), C (validation script).

---

## Phase B: Data Pipeline (Week 6)

**Goal**: Convert raw FDS output → trainable `.npz` and HDF5.
End of week: 720 training pairs accessible via DataLoader.

### Modules to implement

#### 1. `src/data_pipeline/fds_extractor.py`

```python
def extract_slices(fds_dir: Path) -> dict[str, np.ndarray]:
    """Extract Temperature, Visibility, CO slices from FDS output.

    Args:
        fds_dir: directory containing .smv and .sf files

    Returns:
        dict with keys:
          'temperature': (31, 60, 40, 6) float32, °C raw
          'visibility':  (31, 60, 40, 6) float32, m raw
          'co_ppm':      (31, 60, 40, 6) float32, ppm raw
          'coords':      dict {'x', 'y', 'z'} cell centers in metres
          'times':       (31,) float32, [0, 10, ..., 300] s
    """
```

**Implementation pattern** (use the fdsreader standard, see CLAUDE.md):
```python
sim = fdsreader.Simulation(str(fds_dir))
temp_slc = sim.slices.filter_by_quantity("TEMPERATURE")[0]
grid, coords = temp_slc.to_global(return_coordinates=True)

# Time alignment (FDS times may not be exactly 10s apart)
target_times = np.arange(0, 301, 10)
indices = np.searchsorted(temp_slc.times, target_times)
indices = np.clip(indices, 0, len(temp_slc.times) - 1)
grid_aligned = grid[indices]
```

**Self-test**: Run on one validated scenario. Print shape (must be `(31, 60, 40, 6)`),
print initial temperature mean (must be ≈22 °C), check NaN/Inf.

#### 2. `src/data_pipeline/normalize.py`

Bidirectional normalization (forward and inverse).

```python
def normalize_temperature(t_celsius: np.ndarray) -> np.ndarray:
    """T → [0, 1]: clip((T - 20) / 1180, 0, 1)"""

def denormalize_temperature(t_norm: np.ndarray) -> np.ndarray:
    """[0, 1] → T: t_norm * 1180 + 20"""

# Same for visibility (INVERSE mapping) and CO (log scale)
```

**Self-test**: Round-trip identity.
`denormalize(normalize(x)) ≈ x` within 1e-5 tolerance.

#### 3. `src/data_pipeline/mask_generator.py`

```python
def generate_mask(building_geom: dict | Path) -> np.ndarray:
    """Generate (60, 40, 6) building mask from PyroSim geometry.

    Returns:
        float32 array, 1.0 = fluid (open) cell, 0.0 = solid (wall)

    Mask is identical across all scenarios since geometry is fixed.
    Save once to data/building_mask.npz.
    """
```

**Self-test**: Verify mask sum is plausible (most cells should be fluid).

#### 4. `src/dataset/fire_dataset.py`

```python
class FireDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: Path, split: str): ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, y) where:
          x: (5, 60, 40, 6) — input at time t (normalized)
          y: (3, 60, 40, 6) — target at time t+1 (normalized)
        """
```

Pairs are `(t, t+1)`. For 24 train scenarios × 30 pairs each = 720 pairs.

**Self-test**: Load 1 batch, verify shapes, verify all values in [0, 1].

#### 5. `src/data_pipeline/build_dataset.py`

Top-level script: process all 30 scenarios → `dataset.h5`.

```python
def build(scenario_dirs: list[Path], output_h5: Path) -> None:
    """Per-scenario pipeline:
    1. extract_slices(fds_dir)
    2. normalize each channel
    3. apply mask
    4. add time encoding
    5. save into HDF5 with scenario metadata

    HDF5 structure:
      /scenarios/scenario_000/input  (31, 5, 60, 40, 6)
      /scenarios/scenario_000/target (31, 3, 60, 40, 6)
      /mask                          (60, 40, 6)
      /metadata                      JSON
    """
```

### Validation

End of Week 6, the following must work:

```python
ds = FireDataset("data/processed/dataset.h5", split="train")
assert len(ds) == 720
x, y = ds[0]
assert x.shape == (5, 60, 40, 6)
assert y.shape == (3, 60, 40, 6)
assert (x >= 0).all() and (x <= 1).all()
```

**Owner**: C primarily, B reviews dataset interface.

---

## Phase C: ConvLSTM Baseline (Week 7)

**Goal**: Trained ConvLSTM model. Establishes the baseline for PI-FNO comparison.

### Reference repository

Use `ndrplz/ConvLSTM_pytorch` as the starting point. It is a 2D ConvLSTM
implementation. We extend to 3D by replacing `Conv2d` with `Conv3d` and
adapting hidden state shapes.

**URL**: `https://github.com/ndrplz/ConvLSTM_pytorch`

The 3D-extended version has already been validated by the user
(`conv_lstm_3d.py` self-test passed: 349,411 parameters, gradient flow OK).

### Modules to implement

#### 1. `src/models/conv_lstm_3d.py`

Already implemented and verified by the user. Lives in repo. Treats
input `(B, 5, 60, 40, 6)` → `(B, 3, 60, 40, 6)`.

#### 2. `src/training/trainer.py`

Generic training loop, model-agnostic.

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.optim = AdamW(model.parameters(), lr=config.lr,
                          weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optim, T_max=config.epochs)
        self.criterion = nn.MSELoss()

    def train_epoch(self): ...
    def validate(self): ...
    def fit(self):
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            wandb.log({...})
            if val_loss < self.best_val:
                self.save_checkpoint('best.pt')
```

Hyperparameters (from `configs/conv_lstm.yaml`):
- learning_rate: 1e-3
- weight_decay: 1e-5
- batch_size: 4 (A100 40 GB safe)
- epochs: 100
- early_stopping_patience: 15
- gradient_clip: 1.0
- scheduler: CosineAnnealingLR

#### 3. `src/training/train_conv_lstm.py`

Entry point script.

```python
@click.command()
@click.option('--config', default='configs/conv_lstm.yaml')
def main(config):
    cfg = load_config(config)
    model = FireConvLSTM(...)
    train_loader = DataLoader(FireDataset(..., split='train'), ...)
    val_loader   = DataLoader(FireDataset(..., split='val'),   ...)
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit()
```

### Critical sanity checks before full training

#### Sanity check 1: Forward and backward pass
Random input, verify shape, verify gradients flow. Already done.

#### Sanity check 2: Single-batch over-fit
Take 1 batch, train 100 steps, loss must converge to near zero.
**This is the most important check.** If this fails, model has a bug.

#### Sanity check 3: Memory and timing
On A100 40 GB with batch_size=4, must not OOM.
Time per epoch should be 1–2 minutes for full dataset (720 pairs).

### Validation targets
- val Rel L2 < 30% (on val set, training-distribution scenarios)
- test_ood Rel L2 measured (this is the baseline PI-FNO must beat)

**Owner**: B.

---

## Phase D: PI-FNO Training (Weeks 8–9)

### Week 8: PI-FNO without physics loss

**Goal**: FNO model trained with MSE only. Compare to ConvLSTM —
isolates "what does the FNO architecture buy us?"

**Reference**:
- Library: `neuraloperator` (PyTorch Ecosystem)
- 3D application reference: `CUBELeonwang/FNO-3DUM`
  - Building and Environment 2024 paper code
  - 3D urban microclimate FNO implementation
  - Direct template for our `train_fno.py`
- PI-FNO original: `neuraloperator/physics_informed`

**FNO hyperparameters** (initial):
- n_modes = (12, 12, 4) — Fourier modes per axis
- hidden_channels = 32
- n_layers = 4
- lifting_channels = 128
- projection_channels = 128
- Total parameters: ~2-4M

**Memory consideration**: FNO uses more memory than ConvLSTM.
May need to reduce batch_size from 4 to 2.

#### `src/models/fno.py`

```python
from neuralop.models import FNO

class FirePIFNO(nn.Module):
    """Wraps neuraloperator FNO for our (B, 5, 60, 40, 6) shape."""

    def __init__(self, config):
        self.fno = FNO(
            n_modes=config.n_modes,
            in_channels=5,
            out_channels=3,
            hidden_channels=config.hidden_channels,
            n_layers=config.n_layers,
            lifting_channels=config.lifting_channels,
            projection_channels=config.projection_channels,
        )

    def forward(self, x):
        # Add input lifting if needed
        return self.fno(x)
```

#### Training procedure (Week 8)
- Same dataset, same train/val split as ConvLSTM
- Same Trainer class, just swap the model
- MSE loss only (no PI yet)
- Save best checkpoint to `checkpoints/fno_no_pi/best.pt`

#### Go/No-Go decision at end of Week 8
- If FNO no-PI clearly outperforms ConvLSTM on OOD → proceed to Week 9 PI loss
- If FNO no-PI ≈ ConvLSTM → spend 1 extra day tuning FNO modes/channels
- If FNO no-PI clearly underperforms → activate Plan B (see risk register)

### Week 9: PI-FNO full + EXP-FIRE-001

**Goal**: Add physics-informed loss using curriculum learning.
Run EXP-FIRE-001 to compare all three models.

**PI loss components** (added in stages):

| Stage | Epochs | New loss component | Coefficient |
|-------|--------|-------------------|-------------|
| 1 | 0–25 | Data MSE only | λ_data = 1.0 |
| 2 | 25–50 | + Mass conservation (CO) | λ_mass = 0.001 |
| 3 | 50–75 | + Heat diffusion residual (rough) | λ_heat = 0.001 |
| 4 | 75–100 | + Tenability boundary (<safe stays safe) | λ_bound = 0.0001 |

**Curriculum-increase λ** within each stage to avoid loss explosion.

#### `src/training/losses.py`

```python
class PIFNOLoss(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_mass=0.0,
                 lambda_heat=0.0, lambda_bound=0.0):
        ...

    def forward(self, pred, target, x_input):
        loss_data = F.mse_loss(pred, target)

        # Mass conservation: ∂[CO]/∂t + ∇·(v[CO]) ≈ 0
        # Approximated as: per-cell change should be small unless
        # adjacent cells have different concentrations
        loss_mass = self.compute_mass_residual(pred, x_input)

        loss_heat = self.compute_heat_residual(pred, x_input)
        loss_bound = self.compute_tenability_boundary(pred)

        return (self.lambda_data * loss_data
              + self.lambda_mass * loss_mass
              + self.lambda_heat * loss_heat
              + self.lambda_bound * loss_bound)
```

Owner B should consult `neuraloperator/physics_informed` for the exact
PDE residual computation patterns.

#### EXP-FIRE-001 (end of Week 9)

Compare three models on val and OOD test sets:

| Model | Train Rel L2 | Val Rel L2 | OOD Rel L2 | RMSE per ch | SSIM |
|-------|-------------|-----------|------------|-------------|------|
| ConvLSTM | ? | ? | ? | ? | ? |
| FNO no-PI | ? | ? | ? | ? | ? |
| FNO full | ? | ? | ? | ? | ? |

Save: `results/exp_fire_001/comparison.csv`. This is **the** main result
table for the paper.

**Owner**: B (training), D (evaluation).

---

## Phase E: Risk Map and Path Planning (Weeks 10–11)

### Week 10: Risk Map Module

**Goal**: PI-FNO predictions → ISO-13571-based risk maps.
Validate via EXP-RISK-001.

**Pre-study**: All team members read these (1–2 days):
- ISO 13571:2012 §5–7 (tenability thresholds)
- SFPE Handbook 5th Ed. Ch. 63 (Purser & McAllister 2016, FED formula)
- RWTH Fire Simulation Lecture Notes (ASET calculation worked example)
  - URL: `https://firedynamics.github.io/LectureFireSimulation/`

#### Modules to implement

##### 1. `src/risk_map/tenability.py`

Per-cell instantaneous danger score from T, V, CO.

```python
def compute_danger_temperature(t_celsius: np.ndarray) -> np.ndarray:
    """T danger: clip((T - 30) / (60 - 30), 0, 1)
    Returns (..., spatial) array in [0, 1].
    """

def compute_danger_visibility(v_metres: np.ndarray) -> np.ndarray:
    """V danger: clip((10 - V) / (10 - 3), 0, 1)
    Returns (..., spatial) array in [0, 1].
    Lower V = higher danger.
    """

def compute_danger_co(co_ppm: np.ndarray) -> np.ndarray:
    """CO instantaneous danger: clip((CO - 100) / (1400 - 100), 0, 1)"""

def compute_total_danger(t, v, co, weights=(0.4, 0.4, 0.2)) -> np.ndarray:
    """Aggregate: w_T * d_T + w_V * d_V + w_CO * d_CO, clipped to [0, 1]"""
```

##### 2. `src/risk_map/fed.py`

Cumulative FED for a path-following occupant.

```python
def accumulate_fed_co(co_ppm_along_path: np.ndarray,
                     dt_seconds: float) -> np.ndarray:
    """ISO 13571 simplified form:
    FED_n = FED_{n-1} + CO_n * (dt/60) / 27000

    Args:
        co_ppm_along_path: (N,) ppm sequence
        dt_seconds: time step (typically 1.0 for drone replan interval)

    Returns:
        (N,) cumulative FED ∈ [0, ∞)
    """
```

**Note**: We use simplified form (no Purser exponent). See decision D-008.

##### 3. `src/risk_map/aset.py`

ASET map: per-cell time until that cell becomes dangerous.

```python
def compute_aset_map(risk_grid_time: np.ndarray,
                    danger_threshold: float = 0.5) -> np.ndarray:
    """Compute ASET map.

    Args:
        risk_grid_time: (T, 60, 40, 6) total danger over time
        danger_threshold: cell is dangerous when risk > threshold

    Returns:
        (60, 40, 6) ASET in seconds.
        For each cell, time at which danger first exceeds threshold.
        Cells that never become dangerous get max time (300 s).
    """
```

##### 4. `src/risk_map/risk_map_class.py`

Three concrete RiskMap implementations sharing one interface.

```python
from abc import ABC, abstractmethod
import scipy.interpolate as si

class RiskMap(ABC):
    @abstractmethod
    def query(self, xyz, t=None) -> float | np.ndarray:
        """Returns danger ∈ [0, 1]. See CLAUDE.md for full contract."""

class FDSRiskMap(RiskMap):
    """Loaded from a single FDS scenario's pre-computed risk grid."""
    def __init__(self, fds_dir: Path):
        slices = extract_slices(fds_dir)
        risk_grid = compute_total_danger(...)  # (31, 60, 40, 6)
        # Build interpolator
        x = np.linspace(0.25, 29.75, 60)
        y = np.linspace(0.25, 19.75, 40)
        z = np.linspace(0.25, 2.75, 6)
        t = np.linspace(0, 300, 31)
        self.interp = si.RegularGridInterpolator(
            (t, x, y, z), risk_grid,
            method='linear', bounds_error=False, fill_value=1.0,
        )

    def query(self, xyz, t=None):
        if t is None:
            t = self.t_max  # or current sim time
        if xyz.ndim == 1:
            return float(self.interp(np.array([t, *xyz])))
        else:
            ts = np.full(len(xyz), t)
            return self.interp(np.column_stack([ts, xyz]))


class FNORiskMap(RiskMap):
    """From PI-FNO inference."""
    def __init__(self, model, initial_state):
        # autoregress for 31 steps to get (31, 3, 60, 40, 6) prediction
        # convert to (31, 60, 40, 6) total danger
        # build interpolator like FDSRiskMap
        ...


class DynamicRiskMap(RiskMap):
    """Time-evolving live map.
    Maintains a rolling buffer of model predictions.
    Re-runs PI-FNO every 30 s (predictive horizon = 60 s).
    """

    def __init__(self, model, initial_observed):
        self.model = model
        self.cached_prediction = None  # (T, 60, 40, 6)
        self.last_prediction_t = -1
        ...

    def update(self, observed_state, current_t):
        """Called every 30 s by integration loop."""
        # Run model.forward 6 times autoregressively
        # Cache new (T, 60, 40, 6) starting from current_t
        ...

    def query(self, xyz, t):
        # Use cached prediction interpolated at xyz, t
        ...
```

##### 5. `src/risk_map/predictive.py`

Helper for autoregressive prediction.

```python
def autoregress(model, initial_state: torch.Tensor,
                n_steps: int = 6) -> torch.Tensor:
    """Run model n_steps times, chaining output → input.

    Returns: (n_steps, 3, 60, 40, 6) predictions.

    The model takes (B, 5, 60, 40, 6) → (B, 3, 60, 40, 6).
    For chaining, we keep mask channel constant and increment time encoding.
    """
```

#### EXP-RISK-001 validation

Compare FDS ground truth risk map vs PI-FNO risk map on val + OOD scenarios.

**Metrics** (per scenario, then aggregated):
- Risk region IoU @ 0.3 threshold
- Risk region IoU @ 0.5 threshold
- Risk region IoU @ 0.7 threshold
- False Negative Rate (most important — missed dangers)
- False Positive Rate
- Risk grid RMSE
- ASET map RMSE (in seconds)

**Success targets**:
- IoU @ 0.5 > 0.6
- FNR < 10% (safety-critical)
- FPR < 30%
- ASET RMSE < 30 s

Save: `results/exp_risk_001/comparison.csv`.

**Owner**: B + C (joint).

### Week 11: Path Planning + Ablations

**Goal**: Three planners implemented and compared. Ablation studies done.

#### Modules to implement

##### 1. `src/path_planning/building_graph.py`

Build NetworkX graph from PyroSim geometry.

```python
def build_graph(building_yaml: Path) -> nx.Graph:
    """Build undirected graph with 16-20 nodes.

    Node attributes:
      pos: (x, y, z) world metres
      type: 'room' | 'corridor' | 'intersection' | 'exit'
      is_exit: bool

    Edge attributes:
      length: m
      width: m (for bottleneck modeling)
      base_time: s (length / 1.5 m/s walking speed)
    """
```

##### 2. `src/path_planning/edge_weights.py`

Convert risk map → A* edge weights.

```python
def compute_edge_weight(graph: nx.Graph, edge: tuple,
                       risk_map: RiskMap, t: float,
                       alpha: float = 1.0,
                       beta: float = 50.0) -> float:
    """Edge weight = alpha * base_time + beta * risk_along_edge.

    Risk along edge is integrated by sampling 5 points between endpoints.
    """
```

##### 3. `src/path_planning/planners.py`

Three planner implementations.

```python
class DijkstraPlanner:
    """Shortest path. Ignores risk."""

    def plan(self, start_xyz, risk_map, graph, t=0.0):
        # Find nearest graph node to start_xyz
        # Run nx.shortest_path with weight='length'
        # Return list of node positions
        ...

class StaticAvoidancePlanner:
    """A* with current risk only. No replanning."""

    def plan(self, start_xyz, risk_map, graph, t=0.0):
        # Compute edge weights using risk_map at time t
        # Run nx.astar_path with weight=edge_weight, heuristic=euclidean
        # Plan once and never replan
        ...

class DynamicPredictivePlanner:
    """A* with 60s predictive risk. Replans every 30 s."""

    def plan(self, start_xyz, risk_map, graph, t=0.0,
             replan_every: float = 30.0):
        # Plan initially, then at t + 30, t + 60, etc.
        # risk_map should be a DynamicRiskMap that updates internally
        # Returns list of waypoints with timing
        ...
```

##### 4. `src/path_planning/evacuation_sim.py`

Simulate occupant moving along planned path, record FED accumulation.

```python
class EvacuationSimulator:
    def __init__(self, walking_speed_mps=1.5,
                 dt_seconds=1.0):
        ...

    def simulate(self, planner, risk_map_truth: FDSRiskMap,
                 start_xyz: np.ndarray) -> dict:
        """Simulate one occupant's evacuation.

        Args:
            planner: a Planner instance
            risk_map_truth: FDS ground truth — what the occupant
                actually experiences
            start_xyz: starting position

        Returns:
            {
                'path': list of positions over time,
                'cumulative_fed': (T,) FED at each time step,
                'reach_time': time to exit (or inf if didn't make it),
                'frac_in_danger': fraction of time in cells with d > 0.5,
            }
        """
```

#### Ablations (parallel with planning work)

**Ablation 1: PI loss components**
- Train 4 models: Stage 1, Stage 2, Stage 3, Stage 4
- Show OOD performance increases through stages

**Ablation 2: Training data size**
- Train PI-FNO with 10, 15, 20, 24 scenarios
- Plot OOD Rel L2 vs n_train
- Demonstrates "we are still in the data-scaling regime"

**Ablation 3 (optional)**: Model size
- Vary hidden_channels: 16, 32, 64
- Show 32 is the sweet spot for 30 scenarios

Save all to `results/ablations/`.

**Owner**: B (planning), D (ablations).

---

## Phase F: Integration and EXP-PATH-001 (Week 12)

### Goal
Run EXP-PATH-001 (the H6 hypothesis test). Build PyBullet integrated demo.

### EXP-PATH-001

Compare three planners on multiple OOD scenarios.

| Planner | Risk awareness | Replan |
|---------|---------------|--------|
| Dijkstra | None | Never |
| Static avoidance | Current snapshot | Never |
| Dynamic predictive | 60 s predictive | Every 30 s |

For each scenario × planner × starting position (3 OOD × 3 planners × 8 starts = 72 trials):
- Run EvacuationSimulator
- Record cumulative FED, reach time, frac time in danger

**Aggregate metrics**:
- Mean cumulative FED per planner
- % of trials with FED > 0.3 (failed evacuations)
- Mean reach time per planner

**Hypothesis H6**: Dynamic FED < Static FED < Dijkstra FED, and Dynamic
FED is at least 30% lower than Dijkstra FED.

Save: `results/exp_path_001/comparison.csv` + boxplot figure.

This is the **headline result** of the paper.

### PyBullet Integrated Demo

#### `src/integration/pybullet_demo.py`

```python
def run_demo(scenario_id: str, planner_type: str = 'dynamic',
             output_video: Path = Path('figures/demo.mp4')):
    """Build the integrated demo:

    1. Load building URDF (from STL conversion)
    2. Spawn single Crazyflie drone at building entrance
    3. Initialize DynamicRiskMap with FDS ground truth
       (drone responds to actual fire conditions)
    4. Run planner to find path to exit
    5. Drone follows path, querying risk_map continuously
    6. Render to video with risk overlay (z=1.5 m slice as colored ground)
    7. Show drone trail and current FED accumulation
    """
```

**Demo duration**: ~90 s (single scenario, simplified to make 1 clean video).

**Visualization layers**:
- Building geometry from URDF
- Drone (Crazyflie sphere)
- Risk overlay: 2D heatmap on the floor (colored by danger at z=1.5 m)
- Drone trail with FED accumulating
- Text overlay: simulation time, current danger at drone, path remaining

**Owner**: D primarily, A helps with URDF.

---

## Phase G: Reporting (Weeks 13–14)

### Week 13: Visualisation

#### `src/visualization/`

- `plot_fire_evolution.py` — animate T/V/CO over time (per-channel GIF)
- `plot_risk_map.py` — color-coded danger overlay at z=1.5 m
- `plot_paths.py` — overlay 3 planner outputs on building floor plan
- `plot_metrics.py` — bar charts for EXP-FIRE-001 results
- `plot_fed_curves.py` — cumulative FED over time, per planner

Save figures into `figures/` for paper and slides.

### Week 14: Paper + Slides

**Paper structure** (typical 8-page conference format):

1. **Introduction** — fire safety motivation, why ML for surrogate, our claim
2. **Related work** — CFD surrogates, ConvLSTM, FNO, fire safety standards
3. **Method**
   - 3.1 Data: FDS + scenario design
   - 3.2 Models: ConvLSTM3D and PI-FNO architectures
   - 3.3 Risk map: ISO 13571 + FED
   - 3.4 Planning: weighted A* on building graph
4. **Experiments**
   - 4.1 EXP-FIRE-001: model comparison
   - 4.2 EXP-RISK-001: risk map fidelity
   - 4.3 EXP-PATH-001: path safety (H6)
   - 4.4 Ablations
5. **Discussion** — limitations, scope, generalization claims
6. **Conclusion** — summary, future work

**Slides** (12-min talk):
- 2 slides: motivation + claim
- 2 slides: method overview (architecture diagram)
- 1 slide: experimental setup
- 3 slides: EXP-FIRE-001 (1 main, 2 supporting)
- 2 slides: EXP-PATH-001 (the headline)
- 1 slide: PyBullet demo video
- 1 slide: limitations + future work

**Code release**: tag `v1.0-final` and write `RELEASE.md` with reproduction instructions.

**Owner**: D leads, all members contribute.

---

## Risk Register and Plan B

| Risk | Mitigation |
|------|-----------|
| FDS scenario fails fdsreader load | Validate every scenario at end of Week 5 — do not wait until Week 6 |
| Single FDS scenario takes >12 hours | Reduce SLCF dump frequency, T_END to 180 s |
| ConvLSTM doesn't learn (loss flat) | Verify input is normalized, mask is float not bool, run 1-batch over-fit test |
| PI-FNO doesn't beat ConvLSTM | Refocus paper on "30-scenario regime trade-offs" rather than absolute superiority |
| RunPod cost exceeds budget | Switch to Spot only, drop Ablation 3, drop PyBullet demo |
| PI loss explodes | Drop to Stage 2 only, document as "PDE residual ineffective at our resolution" |
| Path planner suggests through walls | Verify graph connectivity carefully, A* respects edge dictionary |
| PyBullet drone falls through floor | Set collision filter, treat drone as point mass |
| Coordinate misalignment | Use the 5-step verification checklist in `coordinate_convention.md` |
| Risk map has high FNR | Lower threshold from 0.5 to 0.4 in path planner; keep FDS ground truth as fallback in demo |

### When things go really wrong

**Drop, in priority order**:
1. Ablation 3 (model size)
2. Tier 1 GNN exploration (mentioned in earlier conversations)
3. PyBullet integration demo
4. EXP-RISK-001 (could be folded into EXP-FIRE-001 evaluation)

**Cannot drop**:
- EXP-FIRE-001 (model comparison)
- EXP-PATH-001 (the headline H6 result)
- Working ConvLSTM and one FNO variant
- 30 FDS scenarios (or 20 in extreme reduction)

---

## References by Section

### Phase A: FDS / PyroSim
- ISO 13571:2012 — fire toxicity standard
- FDS User Guide (NIST) — `&MESH`, `&SLCF`, `&OBST`, `&VENT` reference
- PyroSim manual — STL import, scenario configs

### Phase B: Data pipeline
- `fdsreader` documentation: https://github.com/FireDynamics/fdsreader
- HDF5 best practices for scientific data

### Phase C: ConvLSTM
- Shi et al. 2015 — Convolutional LSTM Network: A Machine Learning Approach
  for Precipitation Nowcasting (NIPS)
- Repo: `ndrplz/ConvLSTM_pytorch` (2D, we extend to 3D)

### Phase D: PI-FNO
- Li et al. 2020 — Fourier Neural Operator for Parametric PDE (ICLR 2021)
- Library: `neuraloperator` (Python package)
- Repo: `CUBELeonwang/FNO-3DUM` — 3D urban microclimate FNO (Building and
  Environment 2024)
- Repo: `neuraloperator/physics_informed` — PI-FNO patterns

### Phase E: Risk + Path
- ISO 13571:2012 §5–7 — tenability thresholds
- SFPE Handbook 5th Ed. Ch. 63 (Purser & McAllister 2016) — FED formulas
- RWTH Fire Simulation Lecture Notes — ASET worked examples
  https://firedynamics.github.io/LectureFireSimulation/
- NetworkX A* documentation

### Phase F: PyBullet
- `pybullet` documentation — physics simulation
- `gym-pybullet-drones` — Crazyflie URDF and dynamics
- STL → URDF conversion guides

### Phase G: Paper
- Engineering capstone competition guidelines (specific to your competition)
- LaTeX templates for conference papers (IEEE, ACM, etc.)
