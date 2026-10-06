from pathlib import Path

import numpy as np
import pytest
from landlab import RasterModelGrid

from run_landslide_model_cli import (
    configured_pga,
    load_pga_raster,
    load_config,
    prepare_config,
    validate_execution_mode,
)


CONFIG_PATH = Path(__file__).parents[1] / "ShallowLandslider_config.yaml"


def minimal_config():
    return {
        "dem_path": "dem.asc",
        "chunking": {"mode": "auto", "tile_size": [100, 200]},
        "soil_params": {},
        "pga": {},
        "simulation": {},
        "flow_params": {},
        "outputs": {},
    }


def test_distributed_example_yaml_is_valid():
    config = prepare_config(load_config(CONFIG_PATH))
    assert config["chunking"]["mode"] == "auto"
    assert config["simulation"]["custom_proportion"] is None


def test_legacy_chunking_flag_and_cli_override_are_supported():
    config = minimal_config()
    config["chunking"] = {"enable_auto": False}
    assert prepare_config(config)["chunking"]["mode"] == "never"
    assert prepare_config(config, "always")["chunking"]["mode"] == "always"


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("chunking", "tile_size", [0, 10], "tile_size"),
        ("pga", "distribution", "triangle", "pga.distribution"),
        ("simulation", "custom_proportion", 0, "custom_proportion"),
        ("simulation", "selection_method", "unknown", "selection_method"),
        ("outputs", "zarr_chunks", [1024], "zarr_chunks"),
    ],
)
def test_invalid_options_fail_during_validation(section, key, value, message):
    config = minimal_config()
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        prepare_config(config)


def test_runout_flag_dependencies_are_validated():
    config = minimal_config()
    config["simulation"]["enable_runout"] = True
    with pytest.raises(ValueError, match="compute_displacement and update_soil"):
        prepare_config(config)


def test_raster_soil_requires_a_path():
    config = minimal_config()
    config["soil_params"]["distribution"] = "raster"

    with pytest.raises(ValueError, match="soil_depth_path"):
        prepare_config(config)

    config["soil_params"]["soil_depth_path"] = "soil.asc"
    assert prepare_config(config)["soil_params"]["distribution"] == "raster"


def test_raster_pga_requires_a_horizontal_path():
    config = minimal_config()
    config["pga"]["distribution"] = "raster"

    with pytest.raises(ValueError, match="horizontal_path"):
        prepare_config(config)

    config["pga"]["horizontal_path"] = "pga.npy"
    assert prepare_config(config)["pga"]["distribution"] == "raster"


def test_pga_vertical_raster_path_must_be_a_string():
    config = minimal_config()
    config["pga"]["vertical_path"] = 42

    with pytest.raises(ValueError, match="vertical_path"):
        prepare_config(config)


def test_pga_raster_row_order_is_validated():
    config = minimal_config()
    config["pga"]["row_order"] = "sideways"

    with pytest.raises(ValueError, match="row_order"):
        prepare_config(config)


def test_configured_pga_loads_numpy_raster_and_derives_vertical(tmp_path):
    grid = RasterModelGrid((3, 4), xy_spacing=30)
    grid.add_zeros("topographic__elevation", at="node")
    nodata = grid.add_zeros("nodata__mask", at="node", dtype=bool)
    nodata[2] = True
    values = np.linspace(0.1, 0.6, grid.number_of_nodes, dtype="float32").reshape(
        grid.shape
    )
    path = tmp_path / "pga.npy"
    np.save(path, values)

    horizontal, vertical = configured_pga(
        grid,
        {
            "distribution": "raster",
            "horizontal_path": str(path),
            "vertical_to_horizontal_ratio": 0.4,
        },
        default_seed=1,
    )

    assert np.allclose(horizontal[~nodata], values.ravel()[~nodata])
    assert np.allclose(vertical[~nodata], 0.4 * horizontal[~nodata])
    assert np.isnan(horizontal[2])
    assert np.isnan(vertical[2])
    assert horizontal.dtype == np.float32
    assert vertical.dtype == np.float32


def test_numpy_pga_raster_shape_must_match_grid(tmp_path):
    grid = RasterModelGrid((3, 4), xy_spacing=30)
    path = tmp_path / "pga.npy"
    np.save(path, np.ones((2, 4), dtype="float32"))

    with pytest.raises(ValueError, match="does not match DEM shape"):
        load_pga_raster(path, grid)


def test_numpy_pga_raster_can_convert_north_first_rows_to_landlab_order(tmp_path):
    grid = RasterModelGrid((3, 4), xy_spacing=30)
    north_first = np.arange(12, dtype="float32").reshape(grid.shape)
    path = tmp_path / "pga.npy"
    np.save(path, north_first)

    values = load_pga_raster(path, grid, row_order="north_to_south")

    assert np.array_equal(values, np.flipud(north_first))


def test_invalid_drainage_relationship_is_rejected():
    config = minimal_config()
    config["soil_params"]["drainage_relationship"] = "sideways"

    with pytest.raises(ValueError, match="drainage_relationship"):
        prepare_config(config)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("P0", 0.0), ("h_star", 0.0), ("D", -1.0), ("eps", 0.0)],
)
def test_piecewise_curvature_parameters_are_validated(parameter, value):
    config = minimal_config()
    config["soil_params"].update(
        {"distribution": "curvature", "relationship": "piecewise", parameter: value}
    )

    with pytest.raises(ValueError, match=parameter):
        prepare_config(config)


def test_runout_rejects_single_flow_hill_metric():
    config = minimal_config()
    config["simulation"].update(
        {
            "compute_displacement": True,
            "enable_runout": True,
            "update_soil": True,
        }
    )
    config["flow_params"].update(
        {"enable": True, "separate_hill_flow": True, "hill_flow_metric": "D8"}
    )

    with pytest.raises(ValueError, match="multiple-flow"):
        prepare_config(config)


def test_chunked_mode_rejects_global_only_features():
    config = prepare_config(minimal_config())
    config["soil_params"]["distribution"] = "drainage_area"
    with pytest.raises(ValueError, match="not supported in chunked mode"):
        validate_execution_mode(config, use_chunking=True)


def test_configured_pga_honours_center_seed_and_nodata():
    grid = RasterModelGrid((5, 6), xy_spacing=30)
    grid.add_zeros("topographic__elevation", at="node")
    nodata = grid.add_zeros("nodata__mask", at="node", dtype=bool)
    nodata[7] = True
    options = {
        "horizontal_max": 0.6,
        "vertical_max": 0.2,
        "distribution": "circular",
        "center": [2, 3],
        "random_center": False,
        "seed": 123,
    }

    horizontal, vertical = configured_pga(grid, options, default_seed=999)

    center_node = grid.grid_coords_to_node_id(2, 3)
    assert np.isclose(horizontal[center_node], 0.6)
    assert np.isclose(vertical[center_node], 0.2)
    assert np.isnan(horizontal[7])
    assert np.isnan(vertical[7])


def test_vertical_pga_can_be_derived_from_horizontal_pga_ratio():
    config = minimal_config()
    config["pga"].update(
        {
            "horizontal_max": 0.7,
            "vertical_max": 99.0,
            "vertical_to_horizontal_ratio": 0.4,
        }
    )

    prepared = prepare_config(config)

    assert np.isclose(prepared["pga"]["vertical_max"], 0.28)


@pytest.mark.parametrize("ratio", [-0.1, float("nan")])
def test_invalid_vertical_to_horizontal_ratio_is_rejected(ratio):
    config = minimal_config()
    config["pga"]["vertical_to_horizontal_ratio"] = ratio
    with pytest.raises(ValueError, match="vertical_to_horizontal_ratio"):
        prepare_config(config)
