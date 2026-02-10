# tests/utilities/test_utilities.py

import numpy as np
import pandas as pd
import pytest
from landlab import RasterModelGrid
import utilities as util


def make_grid(ny=5, nx=5, spacing=10.0):
    mg = RasterModelGrid((ny, nx), xy_spacing=spacing)
    z = mg.add_zeros("topographic__elevation", at="node")
    z[:] = np.linspace(0, ny * nx - 1, ny * nx)
    return mg


def test_apply_soil_depth_uniform():
    mg = make_grid()
    soil = util.apply_soil_depth(
        mg, max_soil_depth=1.2, distribution="uniform"
    )
    assert "soil__depth" in mg.at_node
    assert np.allclose(soil[mg.core_nodes], 1.2)
    assert np.allclose(mg.at_node["soil__depth"][mg.core_nodes], 1.2)
    
def test_apply_soil_depth_elevation_linear():
    mg = make_grid()
    z = mg.at_node["topographic__elevation"]
    z[:] = np.linspace(0, 100, mg.number_of_nodes)

    soil = util.apply_soil_depth(
        mg, distribution="elevation", relationship="linear", max_soil_depth=2.0
    )

    # Higher elevation → lower soil depth
    assert soil[z.argmax()] < soil[z.argmin()]
    assert np.all(soil >= 0)

def test_apply_soil_depth_curvature_linear():
    mg = make_grid()
    # create synthetic curvature field
    curv = np.linspace(-0.01, 0.01, mg.number_of_nodes)
    mg.add_field("curvature", curv, at="node")

    soil = util.apply_soil_depth(
        mg, distribution="curvature", relationship="linear", max_soil_depth=1.5
    )

    assert soil.shape == (mg.number_of_nodes,)
    assert np.nanmin(soil) >= 0
    assert np.nanmax(soil) <= 1.5


def test_fit_bivariate_kde_raises_on_nonpositive_log():
    df = pd.DataFrame({"length_m": [1, 2, 3], "width_m": [0, 1, 2]})
    with pytest.raises(ValueError):
        util.fit_bivariate_kde(
            df, "length_m", "width_m",
            log_transform=True, plot_results=False
        )

def test_fit_bivariate_kde_with_categories():
    df = pd.DataFrame({
        "length_m": [1, 2, 3, 4],
        "width_m": [1, 2, 1, 2],
        "zone": ["A", "A", "B", "B"]
    })

    kde, info = util.fit_bivariate_kde(
        df, "length_m", "width_m", category_col="zone",
        plot_results=False
    )

    assert "by_category" in kde
    assert set(kde["by_category"].keys()) == {"A", "B"}
    assert "x_bounds" in info and "y_bounds" in info

def test_generate_acceleration_grid_reproducibility():
    mg = make_grid()
    h2, v2 = util.generate_acceleration_grid(
        mg, horizontal_max=0.3, vertical_max=0.15,
        distribution="circular", seed=42
    )
    h3, v3 = util.generate_acceleration_grid(
        mg, horizontal_max=0.3, vertical_max=0.15,
        distribution="circular", seed=42
    )
    assert np.allclose(h2, h3)
    assert np.allclose(v2, v3)


def test_pickle_or_not_to_pickle_roundtrip(tmp_path, monkeypatch):
    f1 = tmp_path / "areas.csv"
    f2 = tmp_path / "zonal.csv"
    f3 = tmp_path / "clipped.csv"

    pd.DataFrame({"length_m": [1, 2, 3], "width_m": [1, 2, 3]}).to_csv(f1, index=False)
    pd.DataFrame({"Area": [1000, 1200, 800]}).to_csv(f2, index=False)
    pd.DataFrame({"Area": [900, 950, 910]}).to_csv(f3, index=False)

    def fake_kde(*args, **kwargs):
        return ({"overall": None}, {"x_bounds": (0, 1), "y_bounds": (0, 1)})

    monkeypatch.setattr(util, "fit_bivariate_kde", fake_kde)

    bundle = util.pickle_or_not_to_pickle(
        {"file1": str(f1), "file2": str(f2), "file3": str(f3)},
        pickle_path=str(tmp_path / "md.pkl"),
    )
    assert "measured_data" in bundle
    assert "kde_data" in bundle

    bundle2 = util.pickle_or_not_to_pickle(
        {"file1": str(f1), "file2": str(f2), "file3": str(f3)},
        pickle_path=str(tmp_path / "md.pkl"),
    )
    assert bundle2["kde_data"] == bundle["kde_data"]


def test_calculate_terrain_attribute_with_richdem():
    pytest.importorskip("richdem")
    mg = make_grid()
    z = mg.at_node["topographic__elevation"]
    z[:] = z + 0.01 * np.sin(np.arange(z.size))

    result = util.calculate_terrain_attribute(
        mg, "topographic__elevation",
        attrib="planform_curvature", out_field="curvature"
    )

    assert "curvature" in mg.at_node
    assert result.shape == (mg.number_of_nodes,)

def test_calculate_terrain_attribute_nodata():
    pytest.importorskip("richdem")
    mg = make_grid()
    z = mg.at_node["topographic__elevation"]
    z[:] = np.arange(z.size)
    z[5] = np.nan  # inject nodata

    result = util.calculate_terrain_attribute(
        mg, "topographic__elevation", attrib="slope_riserun", out_field="slope"
    )

    assert np.isnan(result[5])  # nodata preserved
    assert "slope" in mg.at_node