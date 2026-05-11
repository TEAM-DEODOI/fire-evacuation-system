"""Shared constants, normalization, and geometry utilities."""
from src.shared.constants import (
    CELL_SIZE_M,
    DOMAIN_SIZE_M,
    DT_SLCF,
    GRID_SHAPE,
    N_CELLS,
    N_INPUT_CHANNELS,
    N_OUTPUT_CHANNELS,
    N_TIMESTEPS,
    T_END_SECONDS,
    TENABILITY,
)
from src.shared.normalization import (
    compute_time_encoding,
    decode_time_encoding,
    denormalize_co,
    denormalize_temperature,
    denormalize_visibility,
    normalize_co,
    normalize_temperature,
    normalize_visibility,
)

__all__ = [
    "GRID_SHAPE",
    "DOMAIN_SIZE_M",
    "CELL_SIZE_M",
    "N_TIMESTEPS",
    "DT_SLCF",
    "T_END_SECONDS",
    "N_INPUT_CHANNELS",
    "N_OUTPUT_CHANNELS",
    "TENABILITY",
    "N_CELLS",
    "normalize_temperature",
    "denormalize_temperature",
    "normalize_visibility",
    "denormalize_visibility",
    "normalize_co",
    "denormalize_co",
    "compute_time_encoding",
    "decode_time_encoding",
]
