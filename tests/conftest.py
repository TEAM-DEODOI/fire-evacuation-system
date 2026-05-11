"""
Shared pytest fixtures for the fire evacuation system test suite.

Import fixtures by name in any test file — pytest discovers them automatically.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.shared.constants import (
    GRID_SHAPE,
    N_INPUT_CHANNELS,
    N_OUTPUT_CHANNELS,
    N_TIMESTEPS,
)


@pytest.fixture
def grid_shape() -> tuple[int, int, int]:
    """Return the canonical grid shape (60, 40, 6)."""
    return GRID_SHAPE


@pytest.fixture
def zero_grid() -> np.ndarray:
    """A (60, 40, 6) float32 array of zeros."""
    return np.zeros(GRID_SHAPE, dtype=np.float32)


@pytest.fixture
def ones_grid() -> np.ndarray:
    """A (60, 40, 6) float32 array of ones."""
    return np.ones(GRID_SHAPE, dtype=np.float32)


@pytest.fixture
def random_grid(rng: np.random.Generator) -> np.ndarray:
    """A (60, 40, 6) float32 array of random values in [0, 1]."""
    return rng.random(GRID_SHAPE).astype(np.float32)


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def fake_input_tensor() -> np.ndarray:
    """Single-sample model input: (5, 60, 40, 6), random values in [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((N_INPUT_CHANNELS, *GRID_SHAPE)).astype(np.float32)


@pytest.fixture
def fake_output_tensor() -> np.ndarray:
    """Single-sample model output: (3, 60, 40, 6), random values in [0, 1]."""
    rng = np.random.default_rng(1)
    return rng.random((N_OUTPUT_CHANNELS, *GRID_SHAPE)).astype(np.float32)


@pytest.fixture
def fake_batch_input() -> np.ndarray:
    """Batch model input: (4, 5, 60, 40, 6), random values in [0, 1]."""
    rng = np.random.default_rng(2)
    return rng.random((4, N_INPUT_CHANNELS, *GRID_SHAPE)).astype(np.float32)


@pytest.fixture
def fake_batch_output() -> np.ndarray:
    """Batch model output: (4, 3, 60, 40, 6), random values in [0, 1]."""
    rng = np.random.default_rng(3)
    return rng.random((4, N_OUTPUT_CHANNELS, *GRID_SHAPE)).astype(np.float32)


@pytest.fixture
def fake_scenario() -> dict[str, np.ndarray]:
    """A complete fake FDS scenario with raw (un-normalised) fields.

    Returns:
        Dict with keys "temperature", "visibility", "co_ppm",
        each of shape (N_TIMESTEPS, 60, 40, 6).
        Values are in physical units (°C, m, ppm).
    """
    rng = np.random.default_rng(4)
    shape = (N_TIMESTEPS, *GRID_SHAPE)
    return {
        "temperature": rng.uniform(20.0, 800.0, shape).astype(np.float32),
        "visibility": rng.uniform(0.0, 30.0, shape).astype(np.float32),
        "co_ppm": rng.uniform(0.0, 3000.0, shape).astype(np.float32),
    }
