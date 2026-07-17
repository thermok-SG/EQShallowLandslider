# tests/utilities/test_utilities.py

import numpy as np
import pandas as pd
import pytest
import shutil
from pathlib import Path
from landlab import RasterModelGrid
from utils import utilities as util


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

    # Higher core-node elevation -> lower soil depth.
    core = mg.core_nodes
    assert soil[core[np.argmax(z[core])]] < soil[core[np.argmin(z[core])]]
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


def test_apply_soil_depth_accepts_constant_curvature():
    mg = make_grid()
    mg.add_field("curvature", np.zeros(mg.number_of_nodes), at="node")

    soil = util.apply_soil_depth(
        mg,
        distribution="curvature",
        relationship="linear_std_local",
        max_soil_depth=0.8,
    )

    assert np.allclose(soil[mg.core_nodes], 0.4)


def test_curvature_soil_respects_configured_maximum():
    mg = make_grid()
    mg.add_field(
        "curvature", np.linspace(-10.0, 10.0, mg.number_of_nodes), at="node"
    )

    soil = util.apply_soil_depth(
        mg,
        distribution="curvature",
        relationship="linear",
        a=5.0,
        b=0.0,
        max_soil_depth=0.8,
    )

    assert np.allclose(soil[mg.core_nodes], 0.8)


def test_fit_bivariate_kde_raises_on_nonpositive_log():
    df = pd.DataFrame({"length_m": [1, 2, 3], "width_m": [0, 1, 2]})
    with pytest.raises(ValueError):
        util.fit_bivariate_kde(
            df, "length_m", "width_m",
            log_transform=True, plot_results=False
        )

def test_fit_bivariate_kde_with_categories():
    df = pd.DataFrame({
        "length_m": [1, 2, 3, 4, 5, 6, 2, 3, 5, 7, 8, 10],
        "width_m": [1, 2, 1, 3, 2, 4, 1.5, 2.5, 2, 4, 3.5, 5],
        "zone": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"]
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


def test_pickle_or_not_to_pickle_roundtrip(monkeypatch):
    scratch = Path("tests") / "_scratch_pickle_roundtrip"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()

    try:
        f1 = scratch / "areas.csv"
        f2 = scratch / "zonal.csv"
        f3 = scratch / "clipped.csv"

        pd.DataFrame({"length_m": [1, 2, 3], "width_m": [1, 2, 3]}).to_csv(f1, index=False)
        pd.DataFrame({"Area_m2": [1000, 1200, 800]}).to_csv(f2, index=False)
        pd.DataFrame({"Area_m2": [900, 950, 910]}).to_csv(f3, index=False)

        def fake_kde(*args, **kwargs):
            return ({"overall": None}, {"x_bounds": (0, 1), "y_bounds": (0, 1)})

        monkeypatch.setattr(util, "fit_bivariate_kde", fake_kde)

        bundle = util.pickle_or_not_to_pickle(
            {"file1": str(f1), "file2": str(f2), "file3": str(f3)},
            pickle_path=str(scratch / "md.pkl"),
        )
        assert "measured_data" in bundle
        assert "kde_data" in bundle

        bundle2 = util.pickle_or_not_to_pickle(
            {"file1": str(f1), "file2": str(f2), "file3": str(f3)},
            pickle_path=str(scratch / "md.pkl"),
        )
        assert bundle2["kde_data"] == bundle["kde_data"]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


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


def test_plot_comparison_panels_accepts_observed_column_aliases():
    import matplotlib.pyplot as plt

    observed = pd.DataFrame(
        {
            "area_m2": [1000.0, 1500.0, 2200.0],
            "Elevation_mean": [100.0, 120.0, 130.0],
            "Mean_slope": [25.0, 30.0, 35.0],
        }
    )
    model = pd.DataFrame(
        {
            "area": [1100.0, 1800.0, 2100.0],
            "median_elevation": [105.0, 125.0, 128.0],
            "median_slope": [24.0, 31.0, 33.0],
        }
    )

    util.plot_comparison_panels_with_ecdf(observed, model)
    plt.close("all")
