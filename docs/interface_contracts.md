# Interface Contracts

> Module-level interface specifications. Each section defines the **exact**
> tensors, types, and conventions that modules must honour. Changes here
> require updating all consuming modules and bumping the version notice
> at the bottom.
>
> See `CLAUDE.md` for project context, `coordinate_convention.md` for
> coordinate handling.

---

## 1. Model Input / Output

### 1.1 Input tensor

```
Shape  : (B, 5, 60, 40, 6)
Dtype  : torch.float32
Range  : [0, 1]   (all channels)
Device : CUDA (training) or CPU (inference)
```

Channel layout (dimension 1):

| Index | Name | Raw unit | Normalisation |
|-------|------|----------|---------------|
| 0 | Temperature | °C | `(T − 20) / 1180`, clipped to [0, 1] |
| 1 | Visibility | m | `1 − clip(V / 30, 0, 1)`  ← **INVERSE** mapping |
| 2 | CO | ppm | `log1p(CO) / log1p(5000)`, clipped to [0, 1] |
| 3 | Mask | — | 1.0 = fluid (free) cell, 0.0 = solid (wall) |
| 4 | Time encoding | — | `t / 300`, broadcast to grid |

**Convention**: higher value = more dangerous for channels 0–2.

The time encoding is the simulation time normalised by `T_END = 300 s`:
a linear ramp from 0.0 at `t = 0` to 1.0 at `t = 300 s`. It is broadcast
as a constant value across the spatial grid for each frame, so every
cell shares the same time channel value at a given timestep.

### 1.2 Output tensor

```
Shape  : (B, 3, 60, 40, 6)
Dtype  : torch.float32
Range  : [0, 1]
```

Channel layout (dimension 1):

| Index | Name | Same normalisation as input |
|-------|------|-----------------------------|
| 0 | Temperature | Yes |
| 1 | Visibility | Yes (inverse) |
| 2 | CO | Yes |

Mask and time encoding are not predicted (mask is constant, time is known).

### 1.3 Single-step prediction and autoregression

The model predicts **one step ahead**: input is frame `t`, output is
frame `t + 10s`. For 60 s horizon, autoregress 6 times.

**Autoregression pattern**:
```python
def autoregress(model, initial_state, n_steps=6, dt=10.0):
    """
    Args:
        initial_state: (5, 60, 40, 6) — full input including mask, time_enc
        n_steps: number of 10s steps to predict (max 6 for 60s horizon)
        dt: step duration in seconds (always 10.0 in our project)

    Returns:
        (n_steps, 3, 60, 40, 6) predictions
    """
    state = initial_state.clone()
    predictions = []
    t0 = compute_time_from_encoding(state[4, 0, 0, 0])

    for step in range(n_steps):
        with torch.no_grad():
            next_pred = model(state.unsqueeze(0)).squeeze(0)  # (3, 60, 40, 6)
        predictions.append(next_pred)

        # Build next input: predicted T,V,CO + same mask + new time
        new_state = torch.zeros_like(state)
        new_state[:3] = next_pred
        new_state[3] = state[3]  # mask unchanged
        new_t = t0 + (step + 1) * dt
        new_state[4] = compute_time_encoding(new_t)
        state = new_state

    return torch.stack(predictions)  # (n_steps, 3, 60, 40, 6)
```

**Important caveat**: error compounds with autoregression. Quality
typically degrades after ~3 steps (30 s) unless model is very accurate.
This is expected; risk map module handles this by using a 60 s horizon
with caution.

---

## 2. RiskMap Contract

```python
from abc import ABC, abstractmethod
import numpy as np

class RiskMap(ABC):
    """Risk map abstract interface.

    Three concrete implementations exist:
    - FDSRiskMap     : ground truth, used for training/validation
    - FNORiskMap     : PI-FNO inference results, used for evaluation
    - DynamicRiskMap : time-evolving live map, used in PyBullet demo
    """

    @abstractmethod
    def query(
        self,
        xyz: np.ndarray,
        t: float | None = None,
    ) -> float | np.ndarray:
        """Return danger level at world coordinate(s) xyz in metres.

        Args:
            xyz: shape (3,) for single point → returns float.
                 shape (N, 3) for batch     → returns (N,) array.
                 Coordinates in world metres (Z-up, origin at (0, 0, 0)).
            t:   Query time in seconds. If None, uses the most recent
                 available frame.

        Returns:
            Danger level ∈ [0, 1].

        Out-of-bounds rules (return 1.0 — max danger):
            - x < 0 or x > 30
            - y < 0 or y > 20
            - z < 0 or z > 3
            - t < 0 or t > available prediction horizon

        Out-of-time-range:
            - t < 0       → returns risk at t=0
            - t > horizon → returns 1.0 (safety default)
        """
```

### Concrete implementations

| Class | Source | Used in |
|-------|--------|---------|
| `FDSRiskMap` | Pre-computed FDS slices | EXP-RISK-001 ground truth |
| `FNORiskMap` | PI-FNO model inference | EXP-RISK-001 evaluation |
| `DynamicRiskMap` | Live PI-FNO + time advance | Week 12 PyBullet demo |

All three implement the same `query()` signature. Code that consumes
risk maps does not need to know which type is in use.

### `FDSRiskMap` construction

```python
class FDSRiskMap(RiskMap):
    @classmethod
    def from_directory(cls, fds_dir: Path) -> "FDSRiskMap":
        """Load FDS scenario, compute risk grid, build interpolator.

        Internal:
            risk_grid = compute_total_danger(temp, vis, co)  # (31, 60, 40, 6)
            self.interp = scipy.interpolate.RegularGridInterpolator(
                (times, x_coords, y_coords, z_coords), risk_grid,
                method='linear',
                bounds_error=False,
                fill_value=1.0,  # safety default
            )
        """

    def query(self, xyz, t=None):
        if t is None:
            t = self.t_max
        if xyz.ndim == 1:
            point = np.array([t, xyz[0], xyz[1], xyz[2]])
            return float(self.interp(point))
        else:
            ts = np.full(len(xyz), t)
            points = np.column_stack([ts, xyz])
            return self.interp(points)
```

### `DynamicRiskMap` lifecycle

```python
class DynamicRiskMap(RiskMap):
    def __init__(self, model, initial_state):
        self.model = model
        self.cached_predictions = None  # (T_cached, 60, 40, 6)
        self.cache_start_t = 0.0
        # ... initial cache from initial_state autoregressed 6 steps

    def update(self, observed_state, current_t):
        """Called every 30 s by integration loop.

        Args:
            observed_state: (5, 60, 40, 6) current observation
                            (in PyBullet demo, this is FDS ground truth)
            current_t: seconds
        """
        # Autoregress 6 steps from observed_state
        new_predictions = autoregress(self.model, observed_state, n_steps=6)
        self.cached_predictions = new_predictions
        self.cache_start_t = current_t

    def query(self, xyz, t):
        # Index into cached_predictions
        # Out-of-cache → return 1.0
        ...
```

### Standard usage pattern

```python
# In data pipeline:
fds_map = FDSRiskMap.from_directory("data/raw/scenario_000")
danger = fds_map.query(np.array([15.0, 10.0, 1.5]), t=120.0)

# In PyBullet demo:
dyn_map = DynamicRiskMap(model_checkpoint, initial_state)
while sim_running:
    drone_pos, _ = pybullet.getBasePositionAndOrientation(drone_id)
    danger = dyn_map.query(np.array(drone_pos), t=current_t)

    # Periodic update of the cache
    if current_t - dyn_map.cache_start_t >= 30.0:
        dyn_map.update(get_current_observed_state(), current_t)
```

---

## 3. Data Pipeline

### 3.1 `extract_slices(fds_dir)`

```python
def extract_slices(fds_dir: Path) -> dict[str, np.ndarray]:
    """Extract Temperature, Visibility, CO slices from FDS output.

    Returns dict with keys:
        'temperature': (31, 60, 40, 6) float32, °C raw (un-normalised)
        'visibility':  (31, 60, 40, 6) float32, m raw
        'co_ppm':      (31, 60, 40, 6) float32, ppm raw
        'coords': dict
            'x': (60,) float32, cell centres in metres
            'y': (40,) float32
            'z': (6,) float32
        'times': (31,) float32, seconds [0, 10, ..., 300]
    """
```

Raises `ValueError` if:
- `fdsreader` cannot load `.smv` (typically VECTOR=.TRUE. issue)
- Slice shape doesn't match (60, 40, 6) (typically SLCF Z range issue)
- Time count doesn't match 31 (typically DT_SLCF mismatch or
  fdsreader silently dropping frames due to broadcast errors)

### 3.2 `generate_mask`

```python
def generate_mask(building_geom: dict | Path) -> np.ndarray:
    """Generate (60, 40, 6) building mask from PyroSim geometry.

    Returns:
        float32 array, 1.0 = fluid (free) cell, 0.0 = solid (wall)

    The mask is identical across all scenarios since geometry is fixed.
    Save once to `data/building_mask.npz` and load thereafter.
    """
```

### 3.3 `build_dataset`

```python
def build(scenario_dirs: list[Path], output_h5: Path,
          metadata: list[dict]) -> None:
    """Process all 30 scenarios → HDF5 dataset.

    Args:
        scenario_dirs: list of directory paths
        output_h5:     destination HDF5 file
        metadata:      list of scenario configs (matching scenario_config.json)
    """
```

Output HDF5 structure:
```
dataset.h5
├── scenarios/
│   ├── scenario_000/
│   │   ├── input          (31, 5, 60, 40, 6)  normalised float32
│   │   └── target         (31, 3, 60, 40, 6)  normalised float32
│   ├── scenario_001/
│   └── ...
├── mask                   (60, 40, 6)
└── metadata               JSON-encoded scenario configs
```

---

## 4. FireDataset

```python
class FireDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: Path,
        split: Literal["train", "val", "test_ood"],
    ): ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: shape (5, 60, 40, 6)  — input at time t   (normalised)
            y: shape (3, 60, 40, 6)  — target at time t+1 (normalised)
        """

    def __len__(self) -> int:
        # Total pairs = sum over scenarios of (n_frames - 1)
        # For 24 train scenarios × 30 pairs/scenario = 720 pairs
```

DataLoader configuration recommendations (in `configs/conv_lstm.yaml`):
```yaml
batch_size: 4
num_workers: 4
pin_memory: true
shuffle: true       # for training only
drop_last: false
```

---

## 5. PathPlanner Contract

```python
class EvacuationPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        start_xyz: np.ndarray,
        risk_map: RiskMap,
        graph: nx.Graph,
        t: float = 0.0,
    ) -> list[np.ndarray]:
        """Compute safest path from start to nearest exit.

        Args:
            start_xyz: shape (3,) — occupant world position in metres.
            risk_map:  any RiskMap implementation.
            graph:     building NetworkX graph (16–20 nodes).
            t:         simulation time in seconds (for dynamic planners).

        Returns:
            Ordered list of waypoints (each shape (3,) world metres),
            from start to exit (inclusive).
            Returns empty list if no path exists.
        """
```

### Three concrete implementations

```python
class DijkstraPlanner(EvacuationPlanner):
    """Shortest path. Ignores risk."""
    def plan(self, start_xyz, risk_map, graph, t=0.0):
        # Find nearest graph node to start_xyz
        # nx.shortest_path(graph, source=start_node, target=exit_node, weight='length')
        # Pick exit minimizing length
        ...


class StaticAvoidancePlanner(EvacuationPlanner):
    """A* with current risk only. No replanning."""
    def plan(self, start_xyz, risk_map, graph, t=0.0):
        # Compute edge weights once using risk_map.query at time t
        # weight = alpha * length + beta * integrated_risk
        # nx.astar_path with euclidean heuristic
        ...


class DynamicPredictivePlanner(EvacuationPlanner):
    """A* with 60s predictive risk. Replans every 30 s."""

    def __init__(self, alpha=1.0, beta=50.0, replan_period=30.0,
                 lookahead=60.0):
        self.alpha = alpha
        self.beta = beta
        self.replan_period = replan_period
        self.lookahead = lookahead

    def plan(self, start_xyz, risk_map, graph, t=0.0):
        # Initial plan using risk at (t, t+lookahead]
        # Returns waypoints with embedded "replan_at_t" markers
        ...
```

### Building graph structure

NetworkX undirected graph with:

**Nodes** (16–20 total):
```python
node_attrs = {
    'pos': (x: float, y: float, z: float),  # metres
    'type': Literal['room', 'corridor', 'intersection', 'exit'],
    'is_exit': bool,
}
```

**Edges**:
```python
edge_attrs = {
    'length': float,   # m
    'width': float,    # m, for bottleneck modeling
    'base_time': float,  # length / 1.5 (walking speed in m/s)
}
```

### Edge weight computation

```python
def compute_edge_weight(graph, edge, risk_map, t,
                       alpha=1.0, beta=50.0,
                       n_samples=5) -> float:
    """Edge weight = alpha * base_time + beta * integrated_risk

    Risk is integrated by sampling n_samples points along the edge,
    querying risk_map at each, and taking the mean.

    For dynamic planner with lookahead, query at multiple times and
    take max (the worst risk during the planning horizon).
    """
```

---

## 6. EvacuationSimulator (for EXP-PATH-001)

```python
class EvacuationSimulator:
    def __init__(self, walking_speed_mps=1.5, dt=1.0):
        ...

    def simulate(
        self,
        planner: EvacuationPlanner,
        risk_map_truth: FDSRiskMap,  # ground truth, what occupant experiences
        start_xyz: np.ndarray,
        graph: nx.Graph,
    ) -> dict:
        """Simulate one occupant evacuation.

        Returns:
            'path': (N, 3) positions over time
            'cumulative_fed': (N,) FED at each step
            'reach_time': time to reach exit (or float('inf'))
            'frac_in_danger': fraction of steps with d > 0.5
            'final_fed': scalar
            'reached_exit': bool
        """
```

**Note on risk_map_truth**: Even when the planner uses a `DynamicRiskMap`
based on PI-FNO predictions, the **simulator** uses `FDSRiskMap` (ground
truth) to compute what the occupant actually experiences. This is the
correct fairness setup: the planner has imperfect knowledge, but the
"reality" comes from FDS.

---

## 7. Evaluation Metrics

All metrics operate on **normalised** tensors:

| Metric | Function | Output shape |
|--------|----------|--------------|
| RMSE per channel | `metrics.rmse_per_channel` | (3,) |
| MAE per channel | `metrics.mae_per_channel` | (3,) |
| SSIM per channel | `metrics.ssim_per_channel` | (3,) |
| Peak danger error | `metrics.peak_danger_error` | scalar |
| FED error along path | `metrics.fed_error` | scalar |
| Reach time | `metrics.reach_time` | scalar (s) |
| Risk IoU @ threshold | `metrics.risk_iou` | scalar |
| FNR (false negative rate) | `metrics.false_negative_rate` | scalar |
| FPR (false positive rate) | `metrics.false_positive_rate` | scalar |

### Comparison reports

| Experiment | Compares | Output |
|------------|----------|--------|
| EXP-FIRE-001 | ConvLSTM vs PI-FNO no-PI vs PI-FNO full | `results/exp_fire_001/comparison.csv` |
| EXP-RISK-001 | FDS ground truth vs PI-FNO predicted risk maps | `results/exp_risk_001/comparison.csv` |
| EXP-PATH-001 | Dijkstra vs Static vs Dynamic paths | `results/exp_path_001/comparison.csv` |

---

## 8. Configuration File Conventions

All configs are YAML, loaded via dataclass.

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    early_stopping_patience: int
    gradient_clip: float
    # …
```

Config file paths follow the naming convention:

```
configs/
├── building.yaml          # Geometry, graph nodes, edges, exits
├── data.yaml              # FDS paths, normalization settings, splits
├── conv_lstm.yaml         # ConvLSTM hyperparams
├── pi_fno.yaml            # PI-FNO hyperparams + PI loss schedule
├── risk_map.yaml          # Tenability weights, FED parameters
└── path_planning.yaml     # alpha, beta, replan period, lookahead
```

Each config file has a `version` field at top:
```yaml
version: 1
# Parameters below this point...
```

When changing a config in incompatible ways, bump the version and update
loaders to handle migration.

---

## 9. Versioning Policy

| Interface | Stable after |
|-----------|--------------|
| Tensor shapes (input/output) | Week 8 |
| RiskMap.query signature | Week 12 |
| FireDataset return format | Week 8 |
| PathPlanner.plan signature | Week 11 |
| Evaluation metric outputs | Week 14 |

**Before changing a "stable" interface**, log the proposal in
`docs/decisions.md` and discuss before touching code.

---

## 10. Common Pitfalls

When integrating modules, the following errors are most common:

| Pitfall | Detection | Fix |
|---------|-----------|-----|
| Visibility forgotten to inverse-map | Loss decreases for T/CO but increases for V | Ensure `1 - clip(V/30, 0, 1)` on raw V |
| Mask dtype is bool not float | Silent loss of gradients | Cast to float32 in dataset loader |
| Time encoding not broadcast spatially | Shape mismatch in conv | `time_enc.expand(60, 40, 6)` per frame |
| RiskMap returns NaN for valid coord | Out-of-bounds default not set | Verify `bounds_error=False, fill_value=1.0` |
| Planner returns empty list | Exit unreachable due to risk threshold | Lower risk threshold; verify graph connectivity |
| FDS shape (61, 41, 7) instead of (60, 40, 6) | SLCF VECTOR=.TRUE. or wrong Z range | Remove VECTOR=.TRUE., set Z to 0,3 (not 3.5) |

When in doubt, trace shapes and units explicitly:
```python
print(f"x.shape = {x.shape}, x.dtype = {x.dtype}, "
      f"x.min = {x.min().item():.3f}, x.max = {x.max().item():.3f}")
```
