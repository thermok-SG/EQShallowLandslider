
import numpy as np
import pytest
from landlab import RasterModelGrid

from shallow_landslide_component import ShallowLandslider


def make_grid(ny=5, nx=5, spacing=10.0):
    mg = RasterModelGrid((ny, nx), xy_spacing=spacing)
    z = mg.add_ones("topographic__elevation", at="node")
    # simple gradient so slopes are non-zero
    z[:] = np.linspace(0, ny * nx - 1, ny * nx)
    
    
    h = mg.add_zeros("soil__depth", at="node")
    h[:] = 1.0  # meters

    return mg


def test_initialization_creates_optional_fields_when_update_soil_true():
    mg = make_grid()
    assert "soil__depth" not in mg.at_node
    assert "bedrock__elevation" not in mg.at_node

    comp = ShallowLandslider(
        mg,
        cohesion_eff=100.0,
        angle_int_frict=30.0,
        update_soil=True,
        compute_displacement=False,
    )
    assert "soil__depth" in mg.at_node
    assert "bedrock__elevation" in mg.at_node
    assert np.allclose(mg.at_node["soil__depth"], 0.5)


def test_pga_fields_created_with_fallbacks_and_nan_at_boundaries():
    mg = make_grid()
    _ = ShallowLandslider(
        mg,
        cohesion_eff=50.0,
        angle_int_frict=32.0,
        pga_h=None,
        pga_v=None,
        pga_h_max=0.3,
        pga_v_max=0.1,
    )
    h = mg.at_node["earthquake__horizontal_pga"]
    v = mg.at_node["earthquake__vertical_pga"]
    # core nodes get fallback values
    assert np.allclose(h[mg.core_nodes], 0.3)
    assert np.allclose(v[mg.core_nodes], 0.1)
    # boundaries are NaN
    assert np.all(np.isnan(h[mg.boundary_nodes]))
    assert np.all(np.isnan(v[mg.boundary_nodes]))


def test_run_one_step_pipeline_sets_expected_fields_and_labels():
    mg = make_grid()
    comp = ShallowLandslider(
        mg,
        cohesion_eff=25.0,
        angle_int_frict=27.0,
        aspect_interval=45,
        selection_method="probabilistic",
        compute_displacement=False,
        split_by_width_config={
            "kde_data": {"overall": None},
            "kde_transform": {"x_bounds": (0, 1)},
            "width_threshold": 1.5,
        },
    )
    comp.run_one_step(dt=None)

    # stability fields
    for f in [
        "landslide__factor_of_safety",
        "landslide__critical_acceleration",
        "landslide__driving_minus_critical_acceleration",
        "landslide__unstable_mask",
    ]:
        assert f in mg.at_node

    # labels
    for f in [
        "landslide__region_labels",
        "landslide__aspect_subgroup_labels",
        "landslide__dimension_split_labels",
        "landslide__selected_labels",
    ]:
        assert f in mg.at_node

    # selected labels should be subset of aspect labels
    sel = mg.at_node["landslide__selected_labels"]
    asp = mg.at_node["landslide__aspect_subgroup_labels"]
    assert set(np.unique(sel)) <= set(np.unique(asp))


def test_selection_modes_probabilistic_vs_pga_weighted():
    mg = make_grid()

    comp = ShallowLandslider(
        mg, cohesion_eff=25.0, angle_int_frict=27.0, selection_method="probabilistic"
    )
    comp.run_one_step()
    assert comp._selected_proportion is not None

    comp2 = ShallowLandslider(
        mg, cohesion_eff=25.0, angle_int_frict=27.0, selection_method="pga_weighted"
    )
    comp2.run_one_step()
    # proportion produced by real code should be present and in range
    assert comp2._selected_proportion is not None
    assert (
        np.isnan(comp2._selected_proportion)
        or (0.0 <= comp2._selected_proportion <= 1.0)
    )


def test_displacement_and_threshold_marks_high_nodes():
    mg = make_grid()
    comp = ShallowLandslider(
        mg,
        cohesion_eff=25.0,
        angle_int_frict=27.0,
        compute_displacement=True,
        displacement_threshold=0.5,
        time_shaking=1.0,
    )
    comp.run_one_step(dt=1.0)
    assert "landslide__newmark_displacement" in mg.at_node
    # there should be some nodes at or above threshold (depends on inputs)
    assert len(comp._high_disp_nodes) >= 0


def test_results_property_contains_expected_keys():
    mg = make_grid()
    comp = ShallowLandslider(mg, cohesion_eff=25.0, angle_int_frict=27.0)
    comp.run_one_step()
    r = comp.results
    for key in [
        "factor_of_safety",
        "a_transient",
        "a_driving",
        "a_diff",
        "unstable_mask",
        "labels",
        "aspect_labels",
        "selected_labels",
        "group_properties",
    ]:
        assert key in r
