import numpy as np
import pytest
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
