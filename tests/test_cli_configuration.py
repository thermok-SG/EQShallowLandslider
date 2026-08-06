from pathlib import Path

import numpy as np
import pytest
from landlab import RasterModelGrid

from run_landslide_model_cli import (
    configured_pga,
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
