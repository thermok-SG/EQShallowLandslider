"""Tests for the Iverson (2000) transient infiltration hydrology option."""

import numpy as np
import pytest
from landlab import RasterModelGrid

from components.hydrology import (
    IversonWetnessCoupler,
    create_wetness_coupler,
    pressure_response,
)


def make_grid(shape=(5, 7), spacing=10.0, soil_depth=1.0, slope=0.2):
    grid = RasterModelGrid(shape, xy_spacing=spacing)
    surface = grid.add_zeros("topographic__elevation", at="node")
    surface[:] = np.tan(slope) * grid.x_of_node
    grid.add_full("soil__depth", soil_depth, at="node")
    grid.add_field(
        "bedrock__elevation", surface - soil_depth, at="node", copy=True
    )
    return grid


def test_pressure_response_matches_equation_27e_for_scalar_and_array():
    expected_at_one = np.exp(-1.0) / np.sqrt(np.pi) - 0.15729920705028513
    assert pressure_response(0.0) == pytest.approx(0.0)
    assert pressure_response(1.0) == pytest.approx(expected_at_one)
    np.testing.assert_allclose(
        pressure_response([0.0, 1.0, 10.0]),
        [0.0, 0.0502545416600122, 0.9596214126967848],
    )


def test_initial_profile_maps_pressure_head_to_existing_wetness_contract():
    grid = make_grid(soil_depth=2.0, slope=0.0)
    coupling = IversonWetnessCoupler(
        grid,
        initial_water_table_depth=1.5,
        slope_angle=0.0,
    )

    np.testing.assert_allclose(coupling.raw_pressure_head, 0.5)
    np.testing.assert_allclose(coupling.pressure_head, 0.5)
    np.testing.assert_allclose(coupling.relative_wetness, 0.25)
    np.testing.assert_allclose(coupling.saturated_thickness, 0.5)


def test_custom_evaluation_depth_still_preserves_pressure_in_stability_column():
    grid = make_grid(soil_depth=2.0, slope=0.0)
    coupling = IversonWetnessCoupler(
        grid,
        failure_depth=1.0,
        initial_water_table_depth=0.5,
        slope_angle=0.0,
    )

    np.testing.assert_allclose(coupling.pressure_head, 0.5)
    np.testing.assert_allclose(coupling.relative_wetness, 0.25)
    np.testing.assert_allclose(
        coupling.relative_wetness * grid.at_node["soil__depth"],
        coupling.pressure_head,
    )


def test_one_pulse_and_recession_match_analytical_superposition():
    slope = 0.2
    grid = make_grid(soil_depth=1.0, slope=slope)
    diffusivity = 1.0 / (4.0 * np.cos(slope) ** 2)
    coupling = IversonWetnessCoupler(
        grid,
        hydraulic_diffusivity=diffusivity,
        vertical_hydraulic_conductivity=2.0,
        initial_water_table_depth="soil__depth",
        slope_angle=slope,
    )

    coupling.run_one_step(1.0, infiltration_rate=1.0)
    expected_storm = 0.5 * pressure_response(1.0)
    np.testing.assert_allclose(coupling.raw_pressure_head, expected_storm)
    np.testing.assert_allclose(coupling.relative_wetness, expected_storm)

    coupling.run_one_step(1.0, infiltration_rate=0.0)
    expected_recovery = 0.5 * (
        pressure_response(2.0) - pressure_response(1.0)
    )
    np.testing.assert_allclose(coupling.raw_pressure_head, expected_recovery)
    np.testing.assert_allclose(coupling.relative_wetness, expected_recovery)
    assert coupling.pulse_count == 1


def test_adjacent_equal_intervals_are_merged_without_changing_response():
    grid = make_grid(slope=0.0)
    coupling = IversonWetnessCoupler(
        grid,
        hydraulic_diffusivity=0.25,
        vertical_hydraulic_conductivity=1.0,
        slope_angle=0.0,
    )
    coupling.run_one_step(1.0, infiltration_rate=0.4)
    coupling.run_one_step(1.0, infiltration_rate=0.4)

    assert coupling.pulse_count == 1
    np.testing.assert_allclose(
        coupling.raw_pressure_head,
        0.4 * pressure_response(2.0),
    )


def test_beta_line_suction_and_zero_depth_are_mapped_safely():
    grid = make_grid(soil_depth=1.0, slope=0.0)
    bare = int(grid.core_nodes[0])
    grid.at_node["soil__depth"][bare] = 0.0
    coupling = IversonWetnessCoupler(
        grid,
        hydraulic_diffusivity=1.0,
        vertical_hydraulic_conductivity=1.0,
        initial_water_table_depth=2.0,
        slope_angle=0.0,
    )
    assert np.all(coupling.raw_pressure_head[grid.at_node["soil__depth"] > 0] < 0)
    assert np.all(coupling.relative_wetness == 0.0)

    coupling.run_one_step(1.0e6, infiltration_rate=1.0)
    active = grid.at_node["soil__depth"] > 0
    np.testing.assert_allclose(coupling.pressure_head[active], 1.0)
    np.testing.assert_allclose(coupling.relative_wetness[active], 1.0)
    assert coupling.relative_wetness[bare] == 0.0


def test_infiltration_is_limited_by_conductivity_and_excess_is_runoff():
    grid = make_grid()
    coupling = IversonWetnessCoupler(
        grid,
        vertical_hydraulic_conductivity=2.0e-6,
        infiltration_rate=5.0e-6,
        slope_angle=0.0,
    )
    np.testing.assert_allclose(coupling.surface_runoff_rate, 3.0e-6)
    np.testing.assert_allclose(coupling.update_runoff_field(1.0e-6), 4.0e-6)


def test_factory_selects_iverson_model():
    coupling = create_wetness_coupler(
        make_grid(),
        "iverson",
        slope_angle=0.2,
    )
    assert isinstance(coupling, IversonWetnessCoupler)


def test_geometry_change_requires_explicit_history_reset():
    grid = make_grid()
    coupling = IversonWetnessCoupler(grid, slope_angle=0.2)
    coupling.run_one_step(60.0, infiltration_rate=1.0e-6)
    grid.at_node["soil__depth"][grid.core_nodes[0]] = 0.5

    with pytest.raises(RuntimeError, match="geometry changed"):
        coupling.run_one_step(60.0)
    with pytest.raises(RuntimeError, match="reset_history=True"):
        coupling.reconcile_after_terrain_change()

    coupling.reconcile_after_terrain_change(reset_history=True)
    assert coupling.elapsed_time == 0.0
    assert coupling.pulse_count == 0


def test_diffusivity_change_cannot_rewrite_existing_pulse_history():
    grid = make_grid()
    diffusivity = grid.add_full("soil__hydraulic_diffusivity", 1.0e-4, at="node")
    coupling = IversonWetnessCoupler(
        grid,
        hydraulic_diffusivity="soil__hydraulic_diffusivity",
        slope_angle=0.2,
    )
    coupling.run_one_step(60.0, infiltration_rate=1.0e-6)
    diffusivity[grid.core_nodes[0]] *= 2.0

    with pytest.raises(RuntimeError, match="pressure-history parameter"):
        coupling.run_one_step(60.0)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("hydraulic_diffusivity", 0.0, "hydraulic_diffusivity"),
        (
            "vertical_hydraulic_conductivity",
            0.0,
            "vertical_hydraulic_conductivity",
        ),
        ("infiltration_rate", -1.0e-7, "infiltration_rate"),
        ("initial_water_table_depth", -1.0, "initial_water_table_depth"),
        ("steady_infiltration_ratio", 2.0, "steady_infiltration_ratio"),
        ("failure_depth", -1.0, "failure_depth"),
        ("slope_angle", np.pi / 2.0, "slope_angle"),
    ],
)
def test_invalid_inputs_are_rejected(keyword, value, message):
    kwargs = {"slope_angle": 0.2}
    kwargs[keyword] = value
    with pytest.raises(ValueError, match=message):
        IversonWetnessCoupler(make_grid(), **kwargs)
