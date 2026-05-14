# Claude Code Task Request Template

> Standard format for delegating a single module or function to Claude Code.
> Following this template increases success rate and reduces back-and-forth.
>
> **You** (the human) write tasks using this template. Claude Code reads
> the task and implements. Then you verify.

---

## Why a Template

Claude Code is a strong *implementer* but works best when given:

1. **Clear goal** — what you want
2. **Bounded context** — what to read, what to ignore
3. **Concrete interface** — exact signature, types, shapes
4. **Verification command** — how to know it works
5. **Forbidden actions** — common mistakes to avoid

This template provides all five.

---

## The 8-Section Template

Copy this block and fill in for each new task.

```markdown
# Task: <short descriptive title>

## 1. Goal
<one-sentence description of what the function or module does>

## 2. Context to Read
- `CLAUDE.md` (auto-loaded; you've already read it)
- `docs/<specific files relevant to this task>`
- `<any existing code files that this depends on>`

## 3. What to Implement
**File**: `<exact path, e.g., src/risk_map/fed.py>`

**Function signature**(s):
```python
def my_function(
    arg1: type1,
    arg2: type2,
) -> return_type:
    """One-line docstring."""
```

**Behaviour**: <2-4 sentences describing what it computes and why>

**Inputs**: <shapes, dtypes, ranges, units>
**Outputs**: <shapes, dtypes, ranges, units>

## 4. Interface Contract
- Must follow `docs/interface_contracts.md` for any tensor/dict shapes
- Must follow `docs/coordinate_convention.md` for any geometry
- Must use `src/shared/constants.py` for any numerical thresholds
- Must NOT modify any other module

## 5. Verification
After implementation, verify by running:

```bash
python -m src.<path.to.module>
```

This command should print `PASS` and exit 0. The `__main__` block must:
1. Load test data from <where>
2. Compute <what>
3. Verify <criterion>
4. Print clear pass/fail messages

Also add a unit test:

```bash
pytest tests/test_<module>.py -v
```

## 6. Code Style
- Type hints on every public function
- Docstrings (Google or NumPy style)
- pathlib.Path for paths
- raise ValueError with clear message; never silent failure
- No `from foo import *`
- No hard-coded numbers — use constants

## 7. Forbidden
- Do NOT modify other modules unless explicitly asked.
- Do NOT add dependencies without confirming first.
- Do NOT skip the `__main__` self-test block.
- Do NOT change the function signature without proposing first.

## 8. Reporting
After implementation:
1. Show me the full file content.
2. Show the output of `python -m <module>` (must be PASS).
3. Show the output of `pytest tests/test_<module>.py` (must be all PASS).
4. List any new dependencies added (should be ZERO normally).
```

---

## Worked Examples (Project-Specific)

These are real task requests for our fire evacuation system. Use them
as templates for similar tasks.

### Example 1: FED accumulation function

```markdown
# Task: Implement FED accumulation function (Week 10, Member B+C)

## 1. Goal
Compute path-integrated cumulative FED (Fractional Effective Dose)
from CO concentration along an evacuee's path.

## 2. Context to Read
- `CLAUDE.md`
- `docs/risk_indicators.md` (FED formula and threshold, especially
  the simplified ISO 13571 form)
- `docs/decisions.md` (D-008 explains why we use simplified form, not Purser)
- `docs/interface_contracts.md` (RiskMap interface for context)

## 3. What to Implement
**File**: `src/risk_map/fed.py`

**Function signature**:
```python
def accumulate_fed_co(
    co_ppm_along_path: np.ndarray,    # shape (N,) in ppm
    dt_seconds: float,                # time step between path points
) -> np.ndarray:
    """Compute cumulative FED at each path point."""
```

**Behaviour**: For each step along an evacuation path, compute the
accumulated FED following ISO 13571 §7.3 simplified form:
`FED_n = FED_{n-1} + CO_n · (dt/60) / 27000`

**Inputs**:
- `co_ppm_along_path`: 1D array, length N, units ppm
- `dt_seconds`: scalar, typically 1.0 (drone replan interval)

**Outputs**:
- shape (N,) cumulative FED values, dimensionless ∈ [0, ∞)

## 4. Interface Contract
- Use `TENABILITY.FED_REFERENCE = 27000` from `src/shared/constants.py`
- No external state; pure function

## 5. Verification
```bash
python -m src.risk_map.fed
```

Self-test: With constant `[CO] = 1000 ppm` for 30 minutes (`dt = 60s`,
N = 30), expected FED at end ≈ 1000 × 30 / 27000 = 1.111. Test must
verify within 0.001 of expected.

Also: `pytest tests/test_risk_map.py -v`

## 6. Code Style
Per CLAUDE.md.

## 7. Forbidden
- Do NOT use Purser exponent 1.036 (we use simplified form per
  decision D-008).
- Do NOT use any path-finding logic here — this function is purely
  for FED accumulation given a sequence of CO values.

## 8. Reporting
Show full file, `python -m` output, pytest output.
```

---

### Example 2: ConvLSTM training script

```markdown
# Task: Implement ConvLSTM training entry point (Week 7, Member B)

## 1. Goal
Single-command training script that loads `configs/conv_lstm.yaml`,
trains the FireConvLSTM model from `src/models/conv_lstm_3d.py` on
the dataset built in Week 6, logs to W&B, and saves the best checkpoint.

## 2. Context to Read
- `CLAUDE.md`
- `docs/manual_v2.md` Section "Phase C: ConvLSTM Baseline (Week 7)"
- `docs/interface_contracts.md` (FireDataset interface, Trainer pattern)
- `src/models/conv_lstm_3d.py` (model exists, do not modify)
- `src/dataset/fire_dataset.py` (dataset exists, do not modify)
- `src/training/trainer.py` (generic trainer, do not modify)

## 3. What to Implement
**File**: `src/training/train_conv_lstm.py`

**Function signature**:
```python
@click.command()
@click.option('--config', default='configs/conv_lstm.yaml')
@click.option('--wandb-project', default='fire-evacuation')
@click.option('--checkpoint-dir', default='checkpoints/conv_lstm')
def train_conv_lstm(config: str, wandb_project: str, checkpoint_dir: str):
    """Entry point: train ConvLSTM from config."""
```

**Behaviour**:
1. Load config YAML into a dataclass
2. Build FireDataset for train and val splits
3. Instantiate FireConvLSTM model
4. Initialize W&B run with config
5. Run Trainer.fit()
6. Save best checkpoint to checkpoint_dir/best.pt

## 4. Interface Contract
- Must use existing `Trainer` class from `src/training/trainer.py`
- Must use existing `FireDataset` and `FireConvLSTM`
- Must NOT introduce a custom training loop

## 5. Verification
```bash
python -m src.training.train_conv_lstm --config configs/conv_lstm_smoke.yaml
```

The smoke config has epochs=2, fast for quick verification.
Expected: model trains 2 epochs without errors, saves a checkpoint,
W&B run completes (or test mode if WANDB_MODE=disabled).

## 6. Code Style
Per CLAUDE.md.

## 7. Forbidden
- Do NOT modify Trainer, FireDataset, or FireConvLSTM
- Do NOT add new hyperparameters not listed in conv_lstm.yaml
- Do NOT install new packages

## 8. Reporting
Show file, run output (smoke test PASS), and the saved checkpoint
file size.
```

---

### Example 3: Coordinate utility

```markdown
# Task: Implement world↔grid coordinate utilities (Week 2, Member C)

## 1. Goal
Pure functions for converting between world metres and SLCF grid
indices, with clear handling of out-of-bounds.

## 2. Context to Read
- `CLAUDE.md`
- `docs/coordinate_convention.md` (cell-centered indexing rules,
  the 0.25 m offset)
- `docs/lessons_learned.md` (L-004 explains the offset bug)

## 3. What to Implement
**File**: `src/shared/coordinates.py`

**Function signatures**:
```python
def world_to_grid(xyz: np.ndarray) -> np.ndarray:
    """Convert world coords to grid indices.

    Args:
        xyz: shape (3,) or (N, 3), world metres
    Returns:
        same shape, dtype int. Out-of-bounds → -1.
    """

def grid_to_world(idx: np.ndarray) -> np.ndarray:
    """Convert grid indices to cell centre world coords.

    Args:
        idx: shape (3,) or (N, 3), int
    Returns:
        same shape, dtype float, world metres at cell centre
    """

def is_in_bounds(xyz: np.ndarray) -> bool | np.ndarray:
    """Return True for points in [0,30] x [0,20] x [0,3]."""

def cell_centres() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_60, y_40, z_6) cell centre coordinate arrays."""
```

## 4. Interface Contract
- Use constants from `src/shared/constants.py` (no hard-coded numbers)
- Pure functions, no side effects
- Handle both single point and batch inputs

## 5. Verification
```bash
python -m src.shared.coordinates
```

Self-test:
- world_to_grid([0.25, 0.25, 0.25]) → [0, 0, 0]
- world_to_grid([0.0, 0.0, 0.0]) → [-1, -1, -1] or similar (boundary)
- grid_to_world([0, 0, 0]) → [0.25, 0.25, 0.25]
- Roundtrip: grid_to_world(world_to_grid(x)) ≈ x for in-bounds x

## 6. Code Style
Per CLAUDE.md.

## 7. Forbidden
- Do NOT use hard-coded grid sizes (60, 40, 6) — use GRID_SHAPE constant
- Do NOT use os.path or sys

## 8. Reporting
Show full file, `python -m` PASS output, pytest test_coordinates.py output.
```

---

## Anti-Patterns to Avoid

### Anti-pattern 1: "Implement the entire risk map module"

Too broad. Break into:
- `tenability.py` (instantaneous danger)
- `fed.py` (cumulative)
- `aset.py` (ASET map)
- `risk_map_class.py` (abstract + 3 concrete)
- `predictive.py` (autoregress for 60s horizon)

Request each separately.

### Anti-pattern 2: "Make it generic / reusable"

Avoid. Specify the **exact** signature for our use case. We can
generalise later if needed.

### Anti-pattern 3: "Look at the codebase and figure out the best design"

Avoid. Even if Claude Code can do this, the result is unpredictable
and may conflict with our locked interfaces. Always specify the
signature.

### Anti-pattern 4: Skipping verification

Always include section 5. Without it, "implementation complete" can
mean code that doesn't even import correctly.

### Anti-pattern 5: "Refactor X for cleaner code"

Avoid mid-project. Refactoring during active development creates merge
conflicts and breaks downstream. Schedule refactoring for Week 14
(code cleanup) only.

---

## Workflow Recommendation

1. Identify the next module from `docs/manual_v2.md` schedule.
2. Copy this template, fill in 8 sections.
3. Send to Claude Code.
4. Review the output.
5. Run the verification commands yourself.
6. If clean, commit.
7. If issues, refine the task description and send back focused on
   the specific problem.

This typically yields a working module in 1–2 iterations.

---

## After Each Task

Update the project state:

1. If the task introduced a new design decision, append to
   `docs/decisions.md`.
2. If the task uncovered a new bug, append to `docs/lessons_learned.md`.
3. Update **"Current Project State"** in `CLAUDE.md` with the latest status.
4. Commit with a descriptive message.

---

## See Also

- `CLAUDE.md` — project context (auto-loaded)
- `docs/interface_contracts.md` — locked interfaces
- `docs/coordinate_convention.md` — coordinate system rules
- `docs/manual_v2.md` — week-by-week schedule
- `docs/decisions.md` — past decisions
- `docs/lessons_learned.md` — known bugs to avoid
- `docs/risk_indicators.md` — tenability and FED reference
