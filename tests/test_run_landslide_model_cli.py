import numpy as np
import pytest
import yaml
from landlab import RasterModelGrid

import run_landslide_model_cli as cli


def test_fill_nodata_for_terrain_handles_nan_heavy_edge_tile():
    elevation = np.full((8, 8), np.nan)
    elevation[5:, 5:] = np.arange(9, dtype=float).reshape(3, 3)

    filled, mask = cli.fill_nodata_for_terrain(
        elevation, np.zeros_like(elevation, dtype=bool)
    )

    assert mask.sum() == 55
    assert np.isfinite(filled).all()
    assert np.array_equal(filled[~mask], elevation[~mask])
    assert np.all(filled[mask] >= elevation[~mask].min())
    assert np.all(filled[mask] <= elevation[~mask].max())


def test_curvature_soil_handles_nan_heavy_edge_tile():
    elevation = np.full((8, 8), np.nan)
    elevation[4:, 4:] = np.array(
        [
            [10.0, 10.2, 10.6, 11.1],
            [10.1, 10.5, 11.0, 11.8],
            [10.4, 10.9, 11.7, 12.6],
            [10.8, 11.5, 12.4, 13.5],
        ]
    )
    filled, mask = cli.fill_nodata_for_terrain(
        elevation, np.zeros_like(elevation, dtype=bool)
    )
    grid = RasterModelGrid(elevation.shape)
    grid.add_field("topographic__elevation", filled.ravel(), at="node")
    grid.status_at_node[mask.ravel()] = grid.BC_NODE_IS_CLOSED

    soil = cli.apply_configured_soil_depth(
        grid,
        {
            "distribution": "curvature",
            "relationship": "linear_std_local",
            "window": 5,
        },
    )
    soil[mask.ravel()] = 0.0

    assert np.isfinite(soil).all()
    assert np.all(soil[mask.ravel()] == 0.0)


def test_fill_nodata_for_terrain_marks_all_nodata_tile_for_skipping():
    elevation = np.full((5, 6), np.nan)

    filled, mask = cli.fill_nodata_for_terrain(
        elevation, np.ones_like(elevation, dtype=bool)
    )

    assert mask.all()
    assert np.all(filled == 0.0)


def test_apply_configured_soil_depth_forwards_curvature_options(monkeypatch):
    captured = {}

    def fake_apply(grid, **kwargs):
        captured.update(kwargs)
        return "soil"

    monkeypatch.setattr(cli, "apply_soil_depth", fake_apply)

    result = cli.apply_configured_soil_depth(
        object(), {"distribution": "curvature", "window": 9, "scale": 0.02}
    )

    assert result == "soil"
    assert captured["relationship"] == "linear_std_local"
    assert captured["window"] == 9
    assert captured["scale"] == 0.02
    assert captured["plot"] is False


@pytest.mark.parametrize(
    ("soil_cfg", "expected"),
    [
        ({"distribution": "uniform"}, 0),
        ({"distribution": "curvature", "relationship": "linear"}, 1),
        (
            {
                "distribution": "curvature",
                "relationship": "linear_std_local",
                "window": 5,
            },
            3,
        ),
        (
            {
                "distribution": "mean_elev_curv",
                "relationship": "linear_std_local",
                "window": 9,
            },
            5,
        ),
    ],
)
def test_required_curvature_overlap(soil_cfg, expected):
    assert cli.required_curvature_overlap(soil_cfg) == expected


def test_required_curvature_overlap_rejects_invalid_window():
    with pytest.raises(ValueError, match="window"):
        cli.required_curvature_overlap(
            {
                "distribution": "curvature",
                "relationship": "linear_std_local",
                "window": 0,
            }
        )


def test_forced_chunked_cli_writes_complete_acceleration_outputs(tmp_path, monkeypatch):
    """Exercise YAML parsing, NaN-edge curvature tiles, selection, and saving."""
    shape = (12, 13)
    rows, cols = np.indices(shape)
    elevation = 1000.0 + rows**2 + 0.25 * cols**2
    elevation = elevation.astype(float)
    elevation[:2, :] = -9999
    elevation[:, :2] = -9999
    elevation[-1, -5:] = -9999
    dem_path = tmp_path / "nan_edges.asc"
    header = (
        f"ncols {shape[1]}\n"
        f"nrows {shape[0]}\n"
        "xllcorner 0\n"
        "yllcorner 0\n"
        "cellsize 1\n"
        "NODATA_value -9999\n"
    )
    dem_path.write_text(
        header + "\n".join(" ".join(map(str, row)) for row in elevation),
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs"
    config = {
        "dem_path": str(dem_path),
        "dem_type": "SRTMGL1",
        "output_dir": str(output_dir),
        "smooth_num": 0,
        "random_seed": 7,
        "save_pickle": False,
        "flow_params": {"enable": False},
        "chunking": {
            "mode": "always",
            "threshold_cells": 1,
            "tile_size": [5, 6],
            "overlap": 2,
        },
        "soil_params": {
            "cohesion_eff": 1000,
            "angle_int_frict": 30,
            "submerged_soil_proportion": 0.4,
            "max_soil_depth": 1.5,
            "distribution": "curvature",
            "relationship": "linear_std_local",
            "window": 3,
        },
        "pga": {
            "horizontal_max": 0.8,
            "vertical_max": 0.1,
            "distribution": "circular",
            "center": [6, 6],
        },
        "simulation": {
            "selection_method": "probabilistic",
            "proportion_method": "conservative",
            "custom_proportion": None,
            "handle_small": "keep",
            "aspect_interval": 20,
            "compute_displacement": False,
            "enable_runout": False,
            "update_soil": False,
            "n_jobs": 1,
        },
        "split_by_width": {"enabled": False},
        "outputs": {
            "write_parquet": False,
            "write_zarr": False,
            "write_npy_fallback": True,
            "zarr_chunks": [8, 8],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["run_landslide_model_cli.py", "--config", str(config_path)]
    )

    cli.main()

    run_dir = next(path for path in output_dir.iterdir() if path.is_dir())
    rasters = run_dir / "rasters"
    critical = np.load(rasters / "critical_acceleration.npy")
    driving = np.load(rasters / "driving_acceleration.npy")
    nodata = np.load(rasters / "nodata_mask.npy")
    assert critical.shape == shape
    assert driving.shape == shape
    assert np.isfinite(critical[~nodata]).any()
    assert np.isfinite(driving[~nodata]).any()
    assert np.isnan(critical[nodata]).all()
