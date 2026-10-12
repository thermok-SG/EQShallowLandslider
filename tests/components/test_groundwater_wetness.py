"""Tests for transient GDP-to-ShallowLandslider hydrologic coupling."""

import numpy as np
import pytest
from landlab import RasterModelGrid

from components.hydrology import GroundwaterWetnessCoupler
from components.shallow_landslider import ShallowLandslider


def make_aquifer_grid(shape=(5, 7), spacing=10.0, soil_depth=1.0):
    grid = RasterModelGrid(shape, xy_spacing=spacing)
    bedrock = grid.add_zeros("bedrock__elevation", at="node")
    bedrock[:] = 0.01 * grid.x_of_node
    soil = grid.add_full("soil__depth", soil_depth, at="node")
    surface = grid.add_zeros("topographic__elevation", at="node")
    surface[:] = bedrock + soil
    return grid


def test_initial_wetness_is_saturated_thickness_fraction():
    grid = make_aquifer_grid()
    coupling = GroundwaterWetnessCoupler(
        grid,
        initial_relative_wetness=0.25,
        recharge_rate=0.0,
    )

    np.testing.assert_allclose(coupling.saturated_thickness[grid.core_nodes], 0.25)
    np.testing.assert_allclose(
        grid.at_node["soil__relative_wetness"][grid.core_nodes], 0.25
    )
    np.testing.assert_allclose(
        grid.at_node["water_table__elevation"],
        grid.at_node["aquifer_base__elevation"] + grid.at_node["aquifer__thickness"],
    )


def test_recharge_raises_transient_relative_wetness():
    grid = make_aquifer_grid()
    coupling = GroundwaterWetnessCoupler(
        grid,
        hydraulic_conductivity=1.0e-6,
        porosity=0.25,
        initial_relative_wetness=0.1,
    )
    before = coupling.relative_wetness.copy()

    coupling.run_one_step(6.0 * 3600.0, recharge_rate=2.0e-7)

    assert np.all(coupling.relative_wetness[grid.core_nodes] > before[grid.core_nodes])
    assert coupling.groundwater.number_of_substeps >= 1


def test_shallow_landslider_reads_each_new_groundwater_state():
    grid = make_aquifer_grid()
    coupling = GroundwaterWetnessCoupler(
        grid,
        hydraulic_conductivity=1.0e-7,
        porosity=0.25,
        initial_relative_wetness=0.05,
    )
    landslider = ShallowLandslider(
        grid,
        cohesion_eff=100.0,
        angle_int_frict=30.0,
        wetness_source="field",
        selection_method="all",
        split_by_width_config=None,
    )
    landslider.run_one_step()
    dry_fos = landslider.results["factor_of_safety"].copy()

    coupling.run_one_step(24.0 * 3600.0, recharge_rate=2.0e-6)
    landslider.run_one_step()

    np.testing.assert_allclose(
        landslider.results["relative_wetness"], coupling.relative_wetness
    )
    assert np.all(
        landslider.results["factor_of_safety"][grid.core_nodes]
        < dry_fos[grid.core_nodes]
    )


def test_zero_soil_nodes_are_dry_and_receive_no_recharge():
    grid = make_aquifer_grid()
    bare = int(grid.core_nodes[0])
    grid.at_node["soil__depth"][bare] = 0.0
    grid.at_node["topographic__elevation"][bare] = grid.at_node["bedrock__elevation"][
        bare
    ]

    coupling = GroundwaterWetnessCoupler(
        grid,
        recharge_rate=1.0e-6,
        initial_relative_wetness=0.5,
    )

    assert coupling.relative_wetness[bare] == 0.0
    assert coupling.groundwater.recharge[bare] == 0.0
    assert np.isclose(
        grid.at_node["topographic__elevation"][bare]
        - grid.at_node["aquifer_base__elevation"][bare],
        coupling.minimum_aquifer_thickness,
    )


def test_runoff_transfer_combines_gdp_and_external_surface_runoff():
    grid = make_aquifer_grid()
    coupling = GroundwaterWetnessCoupler(grid, initial_relative_wetness=1.0)
    coupling.run_one_step(3600.0, recharge_rate=1.0e-7)
    additional = np.full(grid.number_of_nodes, 2.0e-8)

    routed = coupling.update_runoff_field(additional)

    assert routed is grid.at_node["water__unit_flux_in"]
    np.testing.assert_allclose(routed, coupling.surface_runoff_rate + additional)


def test_terrain_change_requires_explicit_groundwater_reconciliation():
    grid = make_aquifer_grid()
    coupling = GroundwaterWetnessCoupler(
        grid,
        porosity=0.25,
        initial_relative_wetness=0.8,
    )
    eroded = int(grid.core_nodes[len(grid.core_nodes) // 2])
    grid.at_node["soil__depth"][eroded] = 0.2
    grid.at_node["topographic__elevation"][eroded] = (
        grid.at_node["bedrock__elevation"][eroded] + 0.2
    )

    with pytest.raises(RuntimeError, match="reconcile_after_terrain_change"):
        coupling.run_one_step(60.0)

    excess = coupling.reconcile_after_terrain_change()

    assert np.isclose(coupling.saturated_thickness[eroded], 0.2)
    assert np.isclose(excess[eroded], 0.25 * (0.8 - 0.2))
    assert np.isclose(coupling.relative_wetness[eroded], 1.0)
    coupling.run_one_step(60.0)


def test_deposition_adds_dry_capacity_without_creating_groundwater():
    grid = make_aquifer_grid()
    coupling = GroundwaterWetnessCoupler(
        grid,
        initial_relative_wetness=0.4,
    )
    deposited = int(grid.core_nodes[0])
    grid.at_node["soil__depth"][deposited] = 2.0
    grid.at_node["topographic__elevation"][deposited] = (
        grid.at_node["bedrock__elevation"][deposited] + 2.0
    )

    coupling.reconcile_after_terrain_change()

    assert np.isclose(coupling.saturated_thickness[deposited], 0.4)
    assert np.isclose(coupling.relative_wetness[deposited], 0.2)
    assert coupling.geomorphic_water_excess[deposited] == 0.0


@pytest.mark.parametrize("invalid_recharge", [-1.0e-8, np.nan])
def test_invalid_recharge_is_rejected(invalid_recharge):
    grid = make_aquifer_grid()
    coupling = GroundwaterWetnessCoupler(grid)

    with pytest.raises(ValueError, match="recharge_rate"):
        coupling.set_recharge_rate(invalid_recharge)
