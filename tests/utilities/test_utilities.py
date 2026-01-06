
import numpy as np
import pandas as pd
import pytest
from landlab import RasterModelGrid

from helper_functions import (
    apply_soil_depth,
    fit_bivariate_kde,
    generate_acceleration_grid,
    pickle_or_not_to_pickle,
    calculate_terrain_attribute,
)


def make_grid(ny=5, nx=5, spacing=10.0):
    mg = RasterModelGrid((ny, nx), xy_spacing=spacing)
    z = mg.add_zeros("topographic__elevation", at="node")
    z[:] = np.linspace(0, ny * nx - 1, ny * nx)
    return mg


def test_apply_soil_depth_uniform():
    mg = make_grid()
    soil = apply_soil_depth(mg, max_soil_depth=1.2, distribution="uniform", verbose=False)
    assert "soil__depth" in mg.at_node
    assert np.allclose(soil[mg.core_nodes], 1.2)
    assert np.allclose(mg.at_node["soil__depth"][mg.core_nodes], 1.2)


def test_fit_bivariate_kde_raises_on_nonpositive_log():
    df = pd.DataFrame({"length_m": [1.0, 2.0, 3.0], "width_m": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError):
        fit_bivariate_kde(df, "length_m", "width_m", log_transform=True, plot_results=False)


def test_generate_acceleration_grid_uniform_and_circular_reproducible():
    mg = make_grid()
    h1, v1 = generate_acceleration_grid(
        mg, horizontal_max=0.2, vertical_max=0.1, distribution="uniform"
    )
    assert h1.shape == (mg.number_of_nodes,)
    assert v1.shape == (mg.number_of_nodes,)
    h2, v2 = generate_acceleration_grid(
        mg, horizontal_max=0.3, vertical_max=0.15, distribution="circular", seed=42
    )
    h3, v3 = generate_acceleration_grid(
        mg, horizontal_max=0.3, vertical_max=0.15, distribution="circular", seed=42
    )
    assert np.allclose(h2, h3)
    assert np.allclose(v2, v3)



def test_pickle_or_not_to_pickle_roundtrip(tmp_path, monkeypatch):
    import helper_functions.utilities as hf_utils
    import pandas as pd
    import numpy as np

    # Prepare simple CSVs
    f1 = tmp_path / "areas.csv"
    f2 = tmp_path / "zonal.csv"
    f3 = tmp_path / "clipped.csv"
    pd.DataFrame({"length_m": [1, 2, 3], "width_m": [1, 2, 3]}).to_csv(f1, index=False)
    pd.DataFrame({"Area": [1000, 1200, 800]}).to_csv(f2, index=False)
    pd.DataFrame({"Area": [900, 950, 910]}).to_csv(f3, index=False)

    # Monkeypatch the function where it is actually used
    def fake_kde(*args, **kwargs):
        return ({"overall": None}, {"x_bounds": (0, 1), "y_bounds": (0, 1)})

    monkeypatch.setattr(hf_utils, "fit_bivariate_kde", fake_kde, raising=True)
    # Alternatively:
    # monkeypatch.setattr("helper_functions.utilities.fit_bivariate_kde", fake_kde, raising=True)

    bundle = pickle_or_not_to_pickle(
        {"file1": str(f1), "file2": str(f2), "file3": str(f3)},
        pickle_path=str(tmp_path / "md.pkl"),
    )
    assert "measured_data" in bundle
    assert "kde_data" in bundle

    bundle2 = pickle_or_not_to_pickle(
        {"file1": str(f1), "file2": str(f2), "file3": str(f3)},
        pickle_path=str(tmp_path / "md.pkl"),
    )

    # Both should be the trivial dict from the fake, so equality holds
    assert bundle2["kde_data"] == bundle["kde_data"]



def test_calculate_terrain_attribute_with_richdem():
    pytest.importorskip("richdem")
    mg = make_grid()
    z = mg.at_node["topographic__elevation"]
    z[:] = z[:] + 0.01 * np.sin(np.arange(z.size))
    result = calculate_terrain_attribute(
        mg, "topographic__elevation", attrib="planform_curvature", out_field="curvature"
    )
    assert "curvature" in mg.at_node
    assert result.shape == (mg.number_of_nodes,)
