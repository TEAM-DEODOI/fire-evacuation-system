"""
Tests for src/shared/constants.py — Day 1 foundation, must pass.
"""
import pytest

from src.shared.constants import (
    CELL_SIZE_M,
    CHANNEL_NAMES_INPUT,
    CHANNEL_NAMES_OUTPUT,
    CO_NORM_MAX_PPM,
    DOMAIN_ORIGIN_M,
    DOMAIN_SIZE_M,
    DT_SLCF,
    GRID_SHAPE,
    GRID_SHAPE_BATCH,
    GRID_SHAPE_OUTPUT,
    HRR_LEVELS_KW,
    LOOKAHEAD_S,
    MESH_EXTENT_M,
    MESH_SHAPE,
    N_CELLS,
    N_INPUT_CHANNELS,
    N_OUTPUT_CHANNELS,
    N_SCENARIOS_OOD,
    N_SCENARIOS_TOTAL,
    N_SCENARIOS_TRAIN,
    N_SCENARIOS_VAL,
    N_TIMESTEPS,
    REPLAN_PERIOD_S,
    T_END_SECONDS,
    T_MIN_SECONDS,
    T_NORM_MIN_C,
    T_NORM_RANGE_C,
    TENABILITY,
    TenabilityConstants,
    V_NORM_MAX_M,
    WALKING_SPEED_MPS,
)


class TestGridShape:
    def test_shape_values(self) -> None:
        assert GRID_SHAPE == (60, 40, 6)

    def test_shape_is_tuple(self) -> None:
        assert isinstance(GRID_SHAPE, tuple)
        assert len(GRID_SHAPE) == 3

    def test_grid_product_equals_14400(self) -> None:
        nx, ny, nz = GRID_SHAPE
        assert nx * ny * nz == 14400

    def test_n_cells_equals_14400(self) -> None:
        assert N_CELLS == 14400

    def test_input_batch_shape(self) -> None:
        assert GRID_SHAPE_BATCH == (5, 60, 40, 6)

    def test_output_batch_shape(self) -> None:
        assert GRID_SHAPE_OUTPUT == (3, 60, 40, 6)


class TestDomainConsistency:
    def test_domain_size(self) -> None:
        assert DOMAIN_SIZE_M == (30.0, 20.0, 3.0)

    def test_domain_origin(self) -> None:
        assert DOMAIN_ORIGIN_M == (0.0, 0.0, 0.0)

    def test_cell_size(self) -> None:
        assert abs(CELL_SIZE_M - 0.5) < 1e-12

    def test_domain_divided_by_cell_equals_grid(self) -> None:
        for dom, n in zip(DOMAIN_SIZE_M, GRID_SHAPE):
            assert abs(dom / CELL_SIZE_M - n) < 1e-9


class TestMesh:
    def test_mesh_shape(self) -> None:
        assert MESH_SHAPE == (100, 80, 8)

    def test_mesh_extent(self) -> None:
        assert MESH_EXTENT_M == ((-10.0, 40.0), (-10.0, 30.0), (0.0, 4.0))

    def test_mesh_extent_consistent_with_shape(self) -> None:
        for (lo, hi), n in zip(MESH_EXTENT_M, MESH_SHAPE):
            assert abs((hi - lo) / CELL_SIZE_M - n) < 1e-9


class TestTimeConstants:
    def test_n_timesteps(self) -> None:
        assert N_TIMESTEPS == 31

    def test_dt_slcf(self) -> None:
        assert abs(DT_SLCF - 10.0) < 1e-12

    def test_t_end(self) -> None:
        assert abs(T_END_SECONDS - 300.0) < 1e-12

    def test_t_min(self) -> None:
        assert T_MIN_SECONDS == 0.0

    def test_time_consistency(self) -> None:
        assert N_TIMESTEPS == int(T_END_SECONDS / DT_SLCF) + 1

    def test_frames_span_full_duration(self) -> None:
        last = (N_TIMESTEPS - 1) * DT_SLCF
        assert abs(last - T_END_SECONDS) < 1e-9


class TestChannels:
    def test_input_channels(self) -> None:
        assert N_INPUT_CHANNELS == 5

    def test_output_channels(self) -> None:
        assert N_OUTPUT_CHANNELS == 3

    def test_input_channel_names(self) -> None:
        assert CHANNEL_NAMES_INPUT == (
            "temperature",
            "visibility",
            "co",
            "mask",
            "time_enc",
        )

    def test_output_channel_names(self) -> None:
        assert CHANNEL_NAMES_OUTPUT == ("temperature", "visibility", "co")

    def test_channel_count_matches_names(self) -> None:
        assert len(CHANNEL_NAMES_INPUT) == N_INPUT_CHANNELS
        assert len(CHANNEL_NAMES_OUTPUT) == N_OUTPUT_CHANNELS


class TestScenarioSplit:
    def test_total(self) -> None:
        assert N_SCENARIOS_TOTAL == 30

    def test_splits_sum_to_total(self) -> None:
        assert (
            N_SCENARIOS_TRAIN + N_SCENARIOS_VAL + N_SCENARIOS_OOD
            == N_SCENARIOS_TOTAL
        )

    def test_train_is_largest(self) -> None:
        assert N_SCENARIOS_TRAIN > N_SCENARIOS_VAL
        assert N_SCENARIOS_TRAIN > N_SCENARIOS_OOD


class TestNormalizationReferences:
    def test_t_norm_min(self) -> None:
        assert abs(T_NORM_MIN_C - 20.0) < 1e-12

    def test_t_norm_range(self) -> None:
        assert abs(T_NORM_RANGE_C - 1180.0) < 1e-12

    def test_t_norm_upper_bound(self) -> None:
        """T_NORM_MIN_C + T_NORM_RANGE_C should map to 1.0 (≡ 1200 °C)."""
        assert abs((T_NORM_MIN_C + T_NORM_RANGE_C) - 1200.0) < 1e-12

    def test_v_norm_max(self) -> None:
        assert abs(V_NORM_MAX_M - 30.0) < 1e-12

    def test_co_norm_max(self) -> None:
        assert abs(CO_NORM_MAX_PPM - 5000.0) < 1e-12


class TestTenability:
    def test_temperature_ordering(self) -> None:
        assert TENABILITY.T_SAFE_C < TENABILITY.T_DANGER_C

    def test_visibility_ordering(self) -> None:
        assert TENABILITY.V_SAFE_M > TENABILITY.V_DANGER_M

    def test_co_ordering(self) -> None:
        assert TENABILITY.CO_SAFE_PPM < TENABILITY.CO_DANGER_PPM

    def test_fed_threshold_in_range(self) -> None:
        assert 0.0 < TENABILITY.FED_THRESHOLD < 1.0

    def test_tenability_values(self) -> None:
        assert abs(TENABILITY.T_SAFE_C - 30.0) < 1e-9
        assert abs(TENABILITY.T_DANGER_C - 60.0) < 1e-9
        assert abs(TENABILITY.V_SAFE_M - 10.0) < 1e-9
        assert abs(TENABILITY.V_DANGER_M - 3.0) < 1e-9
        assert abs(TENABILITY.CO_SAFE_PPM - 100.0) < 1e-9
        assert abs(TENABILITY.CO_DANGER_PPM - 1400.0) < 1e-9
        assert abs(TENABILITY.FED_REFERENCE - 27000.0) < 1e-9
        assert abs(TENABILITY.FED_THRESHOLD - 0.3) < 1e-9
        assert abs(TENABILITY.WEIGHT_T - 0.4) < 1e-9
        assert abs(TENABILITY.WEIGHT_V - 0.4) < 1e-9
        assert abs(TENABILITY.WEIGHT_CO - 0.2) < 1e-9
        assert abs(TENABILITY.AGGREGATE_THRESHOLD - 0.5) < 1e-9

    def test_weights_sum_to_one(self) -> None:
        s = TENABILITY.WEIGHT_T + TENABILITY.WEIGHT_V + TENABILITY.WEIGHT_CO
        assert abs(s - 1.0) < 1e-9

    def test_tenability_is_singleton_instance(self) -> None:
        assert isinstance(TENABILITY, TenabilityConstants)

    def test_tenability_is_immutable(self) -> None:
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            TENABILITY.T_SAFE_C = 999.0  # type: ignore[misc]


class TestEvacuation:
    def test_walking_speed(self) -> None:
        assert abs(WALKING_SPEED_MPS - 1.5) < 1e-12

    def test_replan_period(self) -> None:
        assert abs(REPLAN_PERIOD_S - 30.0) < 1e-12

    def test_lookahead(self) -> None:
        assert abs(LOOKAHEAD_S - 60.0) < 1e-12

    def test_lookahead_is_six_steps(self) -> None:
        """LOOKAHEAD_S should be 6 SLCF steps (60 s = 6 × 10 s)."""
        assert abs(LOOKAHEAD_S / DT_SLCF - 6.0) < 1e-9


class TestHRRLevels:
    def test_hrr_levels(self) -> None:
        assert HRR_LEVELS_KW == (500, 1000, 1500, 2000)

    def test_hrr_levels_are_increasing(self) -> None:
        levels = list(HRR_LEVELS_KW)
        assert levels == sorted(levels)
