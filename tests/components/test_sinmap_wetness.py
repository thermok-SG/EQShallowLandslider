"""Tests for the SINMAP-style equilibrium wetness option."""

import numpy as np
import pytest
from landlab import RasterModelGrid

from components.hydrology import (
    GroundwaterWetnessCoupler,
    SinmapWetnessCoupler,
    create_wetness_coupler,
)


def make_grid(shape=(5, 7), spacing=10.0, soil_depth=2.0):
    grid = RasterModelGrid(shape, xy_spacing=spacing)
    surface = grid.add_zeros("topographic__elevation", at="node")
    surface[:] = np.tan(0.2) * grid.x_of_node
    grid.add_full("soil__depth", soil_depth, at="node")
    grid.add_field(
        "bedrock__elevation", surface - soil_depth, at="node", copy=True
    )
    return grid


def test_equilibrium_wetness_matches_sinmap_index():
    grid = make_grid()
    recharge = 2.0e-7
    transmissivity = 4.0e-5
    area = 25.0

    coupling = SinmapWetnessCoupler(
        grid,
        transmissivity=transmissivity,
        recharge_rate=recharge,
        specific_contributing_area=area,
        slope_angle=0.2,
    )

    expected = (recharge / transmissivity) * (area / np.sin(0.2))
    np.testing.assert_allclose(coupling.relative_wetness, expected)
    np.testing.assert_allclose(coupling.saturated_thickness, expected * 2.0)


def test_wetness_is_capped_at_saturation_and_zero_without_soil():
    grid = make_grid()
    bare = int(grid.core_nodes[0])
    grid.at_node["soil__depth"][bare] = 0.0

    coupling = SinmapWetnessCoupler(
        grid,
        transmissivity=1.0e-6,
        recharge_rate=1.0e-5,
        specific_contributing_area=100.0,
        slope_angle=0.1,
    )

    active = grid.at_node["soil__depth"] > 0.0
    assert np.all(coupling.relative_wetness[active] == 1.0)
    assert coupling.relative_wetness[bare] == 0.0


def test_recharge_change_is_instantaneous_and_has_no_recession_memory():
    grid = make_grid()
    coupling = SinmapWetnessCoupler(
        grid,
        transmissivity=1.0e-4,
        specific_contributing_area=20.0,
        slope_angle=0.3,
    )

    coupling.run_one_step(3600.0, recharge_rate=1.0e-7)
    assert np.all(coupling.relative_wetness > 0.0)

    coupling.run_one_step(1.0, recharge_rate=0.0)
    assert np.all(coupling.relative_wetness == 0.0)


def test_default_area_uses_raster_drainage_area_per_cell_width():
    grid = make_grid(spacing=10.0)
    drainage = grid.add_full("drainage_area", 300.0, at="node")
    coupling = SinmapWetnessCoupler(
        grid,
        transmissivity=2.0e-4,
        recharge_rate=1.0e-7,
        slope_angle=0.25,
    )

    expected = (1.0e-7 / 2.0e-4) * ((drainage / grid.dx) / np.sin(0.25))
    np.testing.assert_allclose(coupling.relative_wetness, expected)


def test_factory_selects_hydrology_model():
    sinmap_grid = make_grid()
    sinmap = create_wetness_coupler(
        sinmap_grid,
        "sinmap",
        transmissivity=1.0e-4,
        specific_contributing_area=20.0,
        slope_angle=0.2,
    )
    assert isinstance(sinmap, SinmapWetnessCoupler)

    gdp_grid = make_grid()
    gdp = create_wetness_coupler(gdp_grid, "gdp")
    assert isinstance(gdp, GroundwaterWetnessCoupler)

    with pytest.raises(ValueError, match="gdp.*sinmap.*iverson"):
        create_wetness_coupler(make_grid(), "bucket")


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("transmissivity", 0.0, "transmissivity"),
        ("recharge_rate", -1.0e-7, "recharge_rate"),
        ("specific_contributing_area", -1.0, "specific_contributing_area"),
        ("slope_angle", np.pi / 2.0, "slope_angle"),
    ],
)
def test_invalid_inputs_are_rejected(keyword, value, message):
    kwargs = {
        "transmissivity": 1.0e-4,
        "recharge_rate": 1.0e-7,
        "specific_contributing_area": 20.0,
        "slope_angle": 0.2,
    }
    kwargs[keyword] = value

    with pytest.raises(ValueError, match=message):
        SinmapWetnessCoupler(make_grid(), **kwargs)
