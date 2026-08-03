# tests/components/test_shallow_landslider.py

import numpy as np
import pytest
from landlab import RasterModelGrid
from landlab.components import PriorityFloodFlowRouter
from landlab.field.errors import FieldError
from components.shallow_landslider import ShallowLandslider, ShallowLandslideRunout


def make_grid(ny=5, nx=5, spacing=10.0, add_soil=False):
    mg = RasterModelGrid((ny, nx), xy_spacing=spacing)
    z = mg.add_zeros("topographic__elevation", at="node")
    z[:] = np.linspace(0, ny * nx - 1, ny * nx)
    if add_soil:
        soil = mg.add_zeros("soil__depth", at="node")
        soil[:] = 0.5
    return mg


def make_runout_grid(ny=8, nx=8, spacing=10.0):
    mg = RasterModelGrid((ny, nx), xy_spacing=spacing)
    z = mg.add_zeros("topographic__elevation", at="node")
    rows = np.repeat(np.arange(ny, 0, -1), nx)
    cols = np.tile(np.arange(nx), ny)
    z[:] = rows * spacing + 0.1 * cols

    soil = mg.add_zeros("soil__depth", at="node")
    soil[:] = 1.0

    router = PriorityFloodFlowRouter(
        mg,
        flow_metric="D8",
        separate_hill_flow=True,
        hill_flow_metric="Quinn",
        depression_handler="fill",
        update_hill_depressions=True,
        accumulate_flow=True,
    )
    router.run_one_step()
    return mg


def test_initialization_creates_optional_fields():
    mg = make_grid()
    assert "soil__depth" not in mg.at_node

    comp = ShallowLandslider(
        mg,
        cohesion_eff=100,
        angle_int_frict=30,
        update_soil=True,
    )

    assert "soil__depth" in mg.at_node
    assert "bedrock__elevation" in mg.at_node
    assert np.allclose(mg.at_node["soil__depth"], 0.5)


def test_pga_fields_created_correctly():
    mg = make_grid(add_soil=True)

    _ = ShallowLandslider(
        mg,
        cohesion_eff=50,
        angle_int_frict=32,
        pga_h=None,
        pga_v=None,
        pga_h_max=0.3,
        pga_v_max=0.1,
    )

    h = mg.at_node["earthquake__horizontal_pga"]
    v = mg.at_node["earthquake__vertical_pga"]

    assert np.allclose(h[mg.core_nodes], 0.3)
    assert np.allclose(v[mg.core_nodes], 0.1)
    assert np.all(np.isnan(h[mg.boundary_nodes]))
    assert np.all(np.isnan(v[mg.boundary_nodes]))


def test_run_one_step_pipeline():
    mg = make_grid(add_soil=True)

    comp = ShallowLandslider(
        mg,
        cohesion_eff=25.0,
        angle_int_frict=27.0,
        aspect_interval=45,
        selection_method="probabilistic",
        split_by_width_config={
            "kde_data": {"overall": None},
            "kde_transform": {"x_bounds": (0, 1)},
            "width_threshold": 1.5,
        },
    )

    comp.run_one_step()

    expected_fields = [
        "landslide__factor_of_safety",
        "landslide__critical_acceleration",
        "landslide__driving_minus_critical_acceleration",
        "landslide__unstable_mask",
        "landslide__region_labels",
        "landslide__aspect_subgroup_labels",
        "landslide__selected_labels",
    ]
    for f in expected_fields:
        assert f in mg.at_node

    sel = mg.at_node["landslide__selected_labels"]
    asp = mg.at_node["landslide__aspect_subgroup_labels"]
    assert set(np.unique(sel)) <= set(np.unique(asp))


def test_run_one_step_without_measured_width_data():
    mg = make_grid(add_soil=True)

    comp = ShallowLandslider(
        mg,
        cohesion_eff=25.0,
        angle_int_frict=27.0,
        split_by_width_config=None,
    )

    comp.run_one_step()

    assert comp.results["split_labels"] is None
    assert "landslide__dimension_split_labels" not in mg.at_node
    assert "landslide__aspect_subgroup_labels" in mg.at_node
    assert "landslide__selected_labels" in mg.at_node


def test_runout_updates_soil_depth_when_enabled():
    mg = make_runout_grid()
    soil_before = mg.at_node["soil__depth"].copy()

    comp = ShallowLandslider(
        mg,
        cohesion_eff=1.0,
        angle_int_frict=10.0,
        pga_h=2.0,
        pga_v=0.0,
        compute_displacement=True,
        time_shaking=10.0,
        displacement_threshold=0.0,
        enable_runout=True,
        update_soil=True,
        split_by_width_config=None,
    )

    comp.run_one_step()

    assert "landslide__newmark_displacement" in mg.at_node
    assert np.any(mg.at_node["landslide__selected_labels"] > 0)
    assert np.any(np.abs(mg.at_node["soil__depth"] - soil_before) > 0.0)
    assert hasattr(comp._runout, "_last_erosion")
    assert "landslide__erosion" in mg.at_node
    assert "landslide__deposition" in mg.at_node
    assert "landslide__soil_depth_change" in mg.at_node
    np.testing.assert_allclose(
        mg.at_node["landslide__soil_depth_change"],
        mg.at_node["landslide__deposition"] - mg.at_node["landslide__erosion"],
    )
    assert hasattr(comp._runout, "_last_deposition")
    assert np.isclose(
        np.sum(comp._runout._last_erosion),
        np.sum(comp._runout._last_deposition),
    )
    assert np.all(mg.at_node["soil__depth"] >= 0)
    runout_results = comp.results["runout"]
    np.testing.assert_array_equal(
        runout_results["failed_nodes"], comp.results["high_displacement_nodes"]
    )
    assert all(
        path[0] in set(runout_results["failed_nodes"])
        for path in runout_results["paths"]
    )
    assert set(np.flatnonzero(runout_results["erosion"])) <= set(
        runout_results["failed_nodes"]
    )
    assert len(runout_results["paths"]) == len(
        runout_results["path_proportions"]
    )


def test_runout_restores_source_to_endpoint_transport():
    mg = make_runout_grid()
    runout = ShallowLandslideRunout(mg)
    soil = np.ones(mg.number_of_nodes)

    updated, erosion, deposition = runout._update_soil_depth(
        paths=[(27, 20, 13), (27, 26, 25)],
        proportions=[0.5, 0.5],
        soil_depth=soil,
    )

    assert erosion[27] == 1.0
    assert deposition[13] == 0.5
    assert deposition[25] == 0.5
    assert erosion[20] == 0.0
    assert deposition[20] == 0.0
    assert erosion[26] == 0.0
    assert deposition[26] == 0.0
    assert updated[27] == 0.0
    assert updated[20] == 1.0
    assert updated[26] == 1.0
    assert updated[13] == 1.5
    assert updated[25] == 1.5
    assert np.sum(erosion) == np.sum(deposition) == 1.0


def test_runout_rejects_single_flow_hill_routing():
    mg = make_runout_grid()
    node_count = mg.number_of_nodes
    mg.at_node["hill_flow__receiver_node"] = np.arange(node_count)
    mg.at_node["hill_flow__receiver_proportions"] = np.ones(node_count)

    with pytest.raises(ValueError, match="multiflow hill routing"):
        ShallowLandslideRunout(mg)


def test_runout_requires_hill_flow_routing_fields():
    mg = make_grid(ny=8, nx=8, add_soil=True)

    with pytest.raises(FieldError, match="hill_flow__receiver_node"):
        ShallowLandslider(
            mg,
            cohesion_eff=1.0,
            angle_int_frict=10.0,
            pga_h=2.0,
            pga_v=0.0,
            compute_displacement=True,
            enable_runout=True,
            update_soil=True,
            split_by_width_config=None,
        )


def test_selection_modes_probabilistic_vs_pga_weighted():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(
        mg, cohesion_eff=25.0, angle_int_frict=27.0,
        selection_method="probabilistic"
    )
    comp.run_one_step()
    assert comp._selected_proportion is not None

    comp2 = ShallowLandslider(
        mg, cohesion_eff=25.0, angle_int_frict=27.0,
        selection_method="pga_weighted"
    )
    comp2.run_one_step()
    assert comp2._selected_proportion is not None
    assert (0.0 <= comp2._selected_proportion <= 1.0) or np.isnan(comp2._selected_proportion)


def test_results_property_contains_expected_keys():
    mg = make_grid(add_soil=True)
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

def test_aspect_labels_refine_region_labels():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(mg, cohesion_eff=15, angle_int_frict=30)
    comp.run_one_step()

    reg = mg.at_node["landslide__region_labels"]
    asp = mg.at_node["landslide__aspect_subgroup_labels"]

    # Aspect subgrouping should not introduce labels with no parent region.
    assert set(np.unique(asp)) - {0} <= set(np.unique(reg)) - {0}
    
def test_dimension_split_labels_consistency():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(
        mg, cohesion_eff=20, angle_int_frict=30,
        split_by_width_config={"kde_data": {"overall": None}, "kde_transform": {}, "width_threshold": 2.0},
    )
    comp.run_one_step()

    asp = mg.at_node["landslide__aspect_subgroup_labels"]
    dim = mg.at_node["landslide__dimension_split_labels"]

    # Split labels should refine but never contradict aspect groups
    assert set(np.unique(dim)) - {0} <= set(np.unique(asp)) - {0}
