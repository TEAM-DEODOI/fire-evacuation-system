"""
Project-wide numerical constants for the fire evacuation prediction system.

Every value here is anchored in ``CLAUDE.md`` (Hard Constraints),
``docs/coordinate_convention.md`` (grid / MESH), ``docs/interface_contracts.md``
(normalisation), ``docs/risk_indicators.md`` (tenability thresholds), or a
decision log entry (D-008, D-012, D-014, D-015, D-017). No value in this
file is invented — modify it only if the upstream document changes first.

All other modules MUST import constants from here rather than hard-coding
numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

# ─── Grid dimensions (SLCF region — model-visible) ──────────────────────────
# CLAUDE.md "Hard Constraints" + docs/coordinate_convention.md §"Grid 2"
GRID_SHAPE: tuple[int, int, int] = (60, 40, 6)
"""(nx, ny, nz) — SLCF cells along each axis."""

GRID_SHAPE_BATCH: tuple[int, int, int, int] = (5, 60, 40, 6)
"""(C, nx, ny, nz) for a single model input frame: [T, V, CO, mask, time_enc]."""

GRID_SHAPE_OUTPUT: tuple[int, int, int, int] = (3, 60, 40, 6)
"""(C, nx, ny, nz) for a single model output frame: [T, V, CO]."""

CELL_SIZE_M: float = 0.5
"""Cubic cell side length in metres. NEVER change to 0.2 m (see D-002)."""

DOMAIN_SIZE_M: tuple[float, float, float] = (30.0, 20.0, 3.0)
"""(Lx, Ly, Lz) of the SLCF domain in metres."""

DOMAIN_ORIGIN_M: tuple[float, float, float] = (0.0, 0.0, 0.0)
"""SLCF domain origin in world coordinates."""

# ─── MESH dimensions (FDS computational domain, includes external buffer) ───
# coordinate_convention.md §"Grid 1"
MESH_SHAPE: tuple[int, int, int] = (100, 80, 8)
"""(IJK) for the FDS &MESH line. Wider than SLCF for ventilation buffer."""

MESH_EXTENT_M: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-10.0, 40.0),
    (-10.0, 30.0),
    (0.0, 4.0),
)
"""((x_min, x_max), (y_min, y_max), (z_min, z_max)) of the FDS &MESH."""

# ─── Time ───────────────────────────────────────────────────────────────────
# CLAUDE.md "Hard Constraints" → Time
T_MIN_SECONDS: float = 0.0
T_END_SECONDS: float = 300.0
N_TIMESTEPS: int = 31
"""Number of SLCF frames including t=0 (0, 10, 20, …, 300 s)."""

DT_SLCF: float = 10.0
"""Time step between consecutive SLCF frames in seconds (FDS DT_SLCF)."""

# ─── Channels ───────────────────────────────────────────────────────────────
N_INPUT_CHANNELS: int = 5
N_OUTPUT_CHANNELS: int = 3

CHANNEL_NAMES_INPUT: tuple[str, str, str, str, str] = (
    "temperature",
    "visibility",
    "co",
    "mask",
    "time_enc",
)
"""Channel layout for the model's 5-channel input frames."""

CHANNEL_NAMES_OUTPUT: tuple[str, str, str] = (
    "temperature",
    "visibility",
    "co",
)
"""Channel layout for the model's 3-channel output frames."""

# ─── Normalisation reference values (interface_contracts.md §1.1) ───────────
T_NORM_MIN_C: float = 20.0
"""Lower bound of the temperature normalisation: 20 °C → 0.0."""

T_NORM_RANGE_C: float = 1180.0
"""Width of the temperature normalisation: T_norm = (T - 20) / 1180."""

V_NORM_MAX_M: float = 30.0
"""Upper bound of visibility normalisation: V_norm = 1 - clip(V / 30, 0, 1)."""

CO_NORM_MAX_PPM: float = 5000.0
"""Upper bound of CO normalisation: CO_norm = log1p(CO) / log1p(5000)."""

# ─── Scenario split (CLAUDE.md "Hard Constraints" → Data + D-023/D-024) ────
# D-023: production data uses 3 HRR levels (500/1000/1500 kW) × 9 standard
#        locations + 3 held-out locations at the lower two HRRs. Total = 33.
# D-024: all 33 → train. val/ood reserved for future simulations.
N_SCENARIOS_TOTAL: int = 33
N_SCENARIOS_TRAIN: int = 33
N_SCENARIOS_VAL: int = 0
N_SCENARIOS_OOD: int = 0

# ─── Tenability thresholds (risk_indicators.md TENABILITY table) ────────────
@dataclass(frozen=True)
class TenabilityConstants:
    """Tenability thresholds from ISO 13571:2012 and SFPE Handbook 5th ed.

    Frozen dataclass — mutation raises ``FrozenInstanceError``.
    """

    # Temperature (°C)
    T_SAFE_C: float = 30.0
    T_DANGER_C: float = 60.0

    # Visibility (m) — higher is safer, so safe > danger
    V_SAFE_M: float = 10.0
    V_DANGER_M: float = 3.0

    # CO instantaneous concentration (ppm)
    CO_SAFE_PPM: float = 100.0
    CO_DANGER_PPM: float = 1400.0

    # Fractional Effective Dose (cumulative) — ISO 13571 §7.3, D-008/D-009
    FED_REFERENCE: float = 27000.0  # ppm·min reference dose
    FED_THRESHOLD: float = 0.3      # sensitive population

    # Aggregate-danger weights (risk_indicators.md §"Aggregated Danger Score")
    WEIGHT_T: float = 0.4
    WEIGHT_V: float = 0.4
    WEIGHT_CO: float = 0.2

    AGGREGATE_THRESHOLD: float = 0.5
    """Per-cell danger threshold used to compute the ASET map."""


TENABILITY = TenabilityConstants()
"""Singleton tenability thresholds. Access as ``TENABILITY.T_SAFE_C`` etc."""

# ─── HRR scenario levels (D-012 → revised by D-023) ────────────────────────
# Production data was generated at 3 HRR levels rather than the original 4.
HRR_LEVELS_KW: tuple[int, int, int] = (500, 1000, 1500)

# ─── Evacuation parameters (D-014, D-017) ───────────────────────────────────
WALKING_SPEED_MPS: float = 1.5
"""Adult unimpeded walking speed (SFPE Handbook standard, D-017)."""

REPLAN_PERIOD_S: float = 30.0
"""Dynamic-planner re-plan cadence (D-014)."""

LOOKAHEAD_S: float = 60.0
"""Predictive horizon for the dynamic planner (6 × DT_SLCF, D-014)."""

# ─── Derived (do not modify) ────────────────────────────────────────────────
N_CELLS: int = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
"""Total SLCF cells — must equal 14 400."""


# ─── Self-test (run with ``python -m src.shared.constants``) ───────────────
if __name__ == "__main__":
    from dataclasses import FrozenInstanceError

    print("=" * 60)
    print("constants.py self-test")
    print("=" * 60)

    errors: list[str] = []

    # ── Type checks ────────────────────────────────────────────────────────
    type_checks: list[tuple[str, object, type]] = [
        ("GRID_SHAPE", GRID_SHAPE, tuple),
        ("GRID_SHAPE_BATCH", GRID_SHAPE_BATCH, tuple),
        ("GRID_SHAPE_OUTPUT", GRID_SHAPE_OUTPUT, tuple),
        ("CELL_SIZE_M", CELL_SIZE_M, float),
        ("DOMAIN_SIZE_M", DOMAIN_SIZE_M, tuple),
        ("DOMAIN_ORIGIN_M", DOMAIN_ORIGIN_M, tuple),
        ("MESH_SHAPE", MESH_SHAPE, tuple),
        ("MESH_EXTENT_M", MESH_EXTENT_M, tuple),
        ("T_END_SECONDS", T_END_SECONDS, float),
        ("N_TIMESTEPS", N_TIMESTEPS, int),
        ("DT_SLCF", DT_SLCF, float),
        ("T_MIN_SECONDS", T_MIN_SECONDS, float),
        ("N_INPUT_CHANNELS", N_INPUT_CHANNELS, int),
        ("N_OUTPUT_CHANNELS", N_OUTPUT_CHANNELS, int),
        ("CHANNEL_NAMES_INPUT", CHANNEL_NAMES_INPUT, tuple),
        ("CHANNEL_NAMES_OUTPUT", CHANNEL_NAMES_OUTPUT, tuple),
        ("T_NORM_MIN_C", T_NORM_MIN_C, float),
        ("T_NORM_RANGE_C", T_NORM_RANGE_C, float),
        ("V_NORM_MAX_M", V_NORM_MAX_M, float),
        ("CO_NORM_MAX_PPM", CO_NORM_MAX_PPM, float),
        ("HRR_LEVELS_KW", HRR_LEVELS_KW, tuple),
        ("WALKING_SPEED_MPS", WALKING_SPEED_MPS, float),
        ("REPLAN_PERIOD_S", REPLAN_PERIOD_S, float),
        ("LOOKAHEAD_S", LOOKAHEAD_S, float),
        ("N_CELLS", N_CELLS, int),
    ]
    for name, value, expected_type in type_checks:
        if not isinstance(value, expected_type):
            errors.append(f"{name} is {type(value).__name__}, expected {expected_type.__name__}")

    # ── GRID consistency: product = 14 400 ─────────────────────────────────
    if N_CELLS != 14400:
        errors.append(f"N_CELLS={N_CELLS}, expected 14 400")
    if GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2] != 14400:
        errors.append("GRID_SHAPE product != 14 400")

    # ── DOMAIN_SIZE_M / CELL_SIZE_M == GRID_SHAPE ──────────────────────────
    for label, dom, n in zip(("X", "Y", "Z"), DOMAIN_SIZE_M, GRID_SHAPE):
        if round(dom / CELL_SIZE_M) != n:
            errors.append(f"{label}: {dom} / {CELL_SIZE_M} != {n}")

    # ── Channel-name lengths match channel counts ──────────────────────────
    if len(CHANNEL_NAMES_INPUT) != N_INPUT_CHANNELS:
        errors.append("CHANNEL_NAMES_INPUT length != N_INPUT_CHANNELS")
    if len(CHANNEL_NAMES_OUTPUT) != N_OUTPUT_CHANNELS:
        errors.append("CHANNEL_NAMES_OUTPUT length != N_OUTPUT_CHANNELS")

    # ── Time consistency ───────────────────────────────────────────────────
    if N_TIMESTEPS != int(T_END_SECONDS / DT_SLCF) + 1:
        errors.append(
            f"N_TIMESTEPS={N_TIMESTEPS} != T_END / DT + 1 = {int(T_END_SECONDS / DT_SLCF) + 1}"
        )

    # ── Scenario split sums to total ───────────────────────────────────────
    if N_SCENARIOS_TRAIN + N_SCENARIOS_VAL + N_SCENARIOS_OOD != N_SCENARIOS_TOTAL:
        errors.append("scenario split does not sum to N_SCENARIOS_TOTAL")

    # ── TENABILITY immutability + value sanity ─────────────────────────────
    try:
        TENABILITY.T_SAFE_C = 999.0  # type: ignore[misc]
        errors.append("TENABILITY is NOT frozen — assignment succeeded")
    except FrozenInstanceError:
        pass  # expected

    if not (TENABILITY.T_SAFE_C < TENABILITY.T_DANGER_C):
        errors.append("TENABILITY.T_SAFE_C must be < T_DANGER_C")
    if not (TENABILITY.V_SAFE_M > TENABILITY.V_DANGER_M):
        errors.append("TENABILITY.V_SAFE_M must be > V_DANGER_M (higher V = safer)")
    if not (TENABILITY.CO_SAFE_PPM < TENABILITY.CO_DANGER_PPM):
        errors.append("TENABILITY.CO_SAFE_PPM must be < CO_DANGER_PPM")
    if not (0.0 < TENABILITY.FED_THRESHOLD < 1.0):
        errors.append("TENABILITY.FED_THRESHOLD outside (0, 1)")
    if abs(TENABILITY.WEIGHT_T + TENABILITY.WEIGHT_V + TENABILITY.WEIGHT_CO - 1.0) > 1e-9:
        errors.append("TENABILITY weights must sum to 1.0")

    # ── Print + verdict ────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print(f"  GRID_SHAPE     : {GRID_SHAPE}  (× = {N_CELLS})")
    print(f"  DOMAIN_SIZE_M  : {DOMAIN_SIZE_M}")
    print(f"  MESH_SHAPE     : {MESH_SHAPE}")
    print(f"  TIME           : {N_TIMESTEPS} frames × {DT_SLCF}s = {T_END_SECONDS}s")
    print(f"  CHANNELS       : in={N_INPUT_CHANNELS} ({CHANNEL_NAMES_INPUT})")
    print(f"                   out={N_OUTPUT_CHANNELS} ({CHANNEL_NAMES_OUTPUT})")
    print(f"  TENABILITY     : T[{TENABILITY.T_SAFE_C},{TENABILITY.T_DANGER_C}]°C  "
          f"V[{TENABILITY.V_DANGER_M},{TENABILITY.V_SAFE_M}]m  "
          f"CO[{TENABILITY.CO_SAFE_PPM},{TENABILITY.CO_DANGER_PPM}]ppm")
    print(f"                   weights T/V/CO = {TENABILITY.WEIGHT_T}/{TENABILITY.WEIGHT_V}/{TENABILITY.WEIGHT_CO}")
    print(f"                   FED ref={TENABILITY.FED_REFERENCE} threshold={TENABILITY.FED_THRESHOLD}")
    print(f"  HRR_LEVELS_KW  : {HRR_LEVELS_KW}")
    print(f"  EVACUATION     : walking={WALKING_SPEED_MPS} m/s  replan={REPLAN_PERIOD_S}s  lookahead={LOOKAHEAD_S}s")
    print("\nPASS: All constants validated")
