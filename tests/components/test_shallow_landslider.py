# tests/components/test_shallow_landslider.py

import numpy as np
import pytest
from landlab import RasterModelGrid
from landlab.components import PriorityFloodFlowRouter
from landlab.field.errors import FieldError
from components.shallow_landslider import ShallowLandslider, ShallowLandslideRunout
from components.shallow_landslider.shallow_landslide_component import _regionprops


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
    np.testing.assert_allclose(
        mg.at_node["bedrock__elevation"],
        mg.at_node["topographic__elevation"] - mg.at_node["soil__depth"],
    )


def test_omitted_pga_defaults_to_zero():
    mg = make_grid(add_soil=True)

    _ = ShallowLandslider(
        mg,
        cohesion_eff=50,
        angle_int_frict=32,
        pga_h=None,
        pga_v=None,
    )

    h = mg.at_node["earthquake__horizontal_pga"]
    v = mg.at_node["earthquake__vertical_pga"]

    assert np.allclose(h[mg.core_nodes], 0.0)
    assert np.allclose(v[mg.core_nodes], 0.0)
    assert np.all(np.isnan(h[mg.boundary_nodes]))
    assert np.all(np.isnan(v[mg.boundary_nodes]))


def test_initialization_only_creates_public_output_field():
    mg = make_grid(add_soil=True)
    mg.add_zeros("bedrock__elevation", at="node")
    mg.add_zeros("earthquake__horizontal_pga", at="node")
    mg.add_zeros("earthquake__vertical_pga", at="node")
    fields_before = set(mg.at_node)

    ShallowLandslider(mg)

    assert set(mg.at_node) - fields_before == {"landslide__selected_labels"}


def test_regionprops_matches_expected_geometry():
    labels = np.zeros((5, 6), dtype=int)
    labels[1:3, 2:5] = 2

    (region,) = _regionprops(labels)

    assert region.label == 2
    assert region.bbox == (1, 2, 3, 5)
    assert region.area == 6.0
    assert np.allclose(region.centroid, (1.5, 3.0))
    assert np.isclose(region.axis_major_length, 4.0 * np.sqrt(2.0 / 3.0))
    assert np.isclose(region.axis_minor_length, 2.0)
    assert np.isclose(region.orientation, np.pi / 2.0)
    assert np.isclose(region.eccentricity, np.sqrt(5.0 / 8.0))


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

    assert {name for name in mg.at_node if name.startswith("landslide__")} == {
        "landslide__selected_labels"
    }
    sel = mg.at_node["landslide__selected_labels"]
    asp = comp.results["aspect_labels"]
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
    assert comp.results["aspect_labels"] is not None
    assert "landslide__selected_labels" in mg.at_node


def test_all_selection_retains_every_candidate_group():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1.0,
        angle_int_frict=10.0,
        selection_method="all",
        split_by_width_config=None,
    )

    comp.run_one_step()

    candidates = comp.results["aspect_labels"]
    np.testing.assert_array_equal(comp.results["selected_labels"], candidates)
    expected_proportion = 1.0 if np.any(candidates > 0) else 0.0
    assert comp.results["selected_proportion"] == expected_proportion


def test_hydrologic_selection_weight_inverts_candidate_critical_wetness():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1.0,
        angle_int_frict=10.0,
        selection_method="hydrologic",
        split_by_width_config=None,
    )
    labels = np.zeros(mg.shape, dtype=int)
    labels[1, 1:4] = 1
    critical = np.full(mg.number_of_nodes, np.nan)
    candidate_nodes = np.flatnonzero(labels.ravel() > 0)
    critical[candidate_nodes] = [0.2, 0.5, 0.8]

    weights, metadata = comp._generate_hydrologic_selection_weight(
        labels, critical
    )

    np.testing.assert_allclose(weights.ravel()[candidate_nodes], [1.0, 0.5, 0.0])
    assert np.all(weights[labels == 0] == 0.0)
    assert metadata["minimum_critical_wetness"] == pytest.approx(0.2)
    assert metadata["maximum_critical_wetness"] == pytest.approx(0.8)
    assert metadata["calibrated_probability"] is False


def test_hydrologic_selection_weight_uses_equal_weights_for_constant_range():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(mg, selection_method="hydrologic")
    labels = np.zeros(mg.shape, dtype=int)
    labels[1:3, 1:3] = 1
    critical = np.zeros(mg.number_of_nodes)
    critical[labels.ravel() > 0] = 0.4

    weights, metadata = comp._generate_hydrologic_selection_weight(
        labels, critical
    )

    assert np.all(weights[labels > 0] == 1.0)
    assert metadata["performed"] is False
    assert "same critical wetness" in metadata["reason"]


def test_hydrologic_selection_runs_at_zero_pga_and_exposes_weight():
    mg = make_grid(add_soil=True)
    mg.add_full("soil__relative_wetness", 1.0, at="node")
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1.0,
        angle_int_frict=10.0,
        wetness_source="field",
        selection_method="hydrologic",
        random_seed=42,
        split_by_width_config=None,
    )

    comp.run_one_step()

    candidates = comp.results["aspect_labels"] > 0
    weights = comp.results["selection_weight"]
    assert np.all(weights[~candidates] == 0.0)
    assert np.all((weights[candidates] >= 0.0) & (weights[candidates] <= 1.0))
    assert comp.results["selection_metadata"]["weight_type"] == (
        "inverted_critical_relative_wetness"
    )
    assert comp.results["selection_metadata"]["calibrated_probability"] is False


def test_hydrologic_runout_updates_landscape_and_requires_external_refresh():
    mg = make_runout_grid()
    wetness = mg.add_full("soil__relative_wetness", 1.0, at="node")
    elevation_before = mg.at_node["topographic__elevation"].copy()

    comp = ShallowLandslider(
        mg,
        cohesion_eff=1.0,
        angle_int_frict=10.0,
        wetness_source="field",
        selection_method="all",
        compute_displacement=False,
        enable_runout=True,
        update_soil=True,
        runout_source_mode="selected",
        runout_distance=20.0,
        evolve_landscape=True,
        split_by_width_config=None,
    )

    comp.run_one_step()

    assert comp.results["newmark"] is None
    assert comp.terrain_dirty
    assert comp.results["terrain_revision"] == 1
    assert np.any(mg.at_node["topographic__elevation"] != elevation_before)
    np.testing.assert_allclose(
        mg.at_node["topographic__elevation"],
        mg.at_node["bedrock__elevation"] + mg.at_node["soil__depth"],
    )
    expected_sources = np.flatnonzero(
        (comp.results["selected_labels"] > 0) & comp.results["unstable_mask"]
    )
    np.testing.assert_array_equal(
        comp.results["runout"]["source_nodes"], expected_sources
    )
    assert np.isclose(
        np.sum(comp.results["runout"]["erosion"]),
        np.sum(comp.results["runout"]["deposition"]),
    )

    with pytest.raises(RuntimeError, match="Rerun flow routing"):
        comp.run_one_step()

    # The driver is responsible for updating routing and wetness after terrain
    # changes. Wetness is kept saturated here to isolate landscape feedback.
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
    wetness[:] = 1.0
    first_fos = comp.results["factor_of_safety"].copy()
    comp.acknowledge_external_state_updated()
    assert not comp.terrain_dirty
    comp.run_one_step()
    assert not np.allclose(
        comp.results["factor_of_safety"][mg.core_nodes],
        first_fos[mg.core_nodes],
        equal_nan=True,
    )


def test_selected_runout_distance_can_come_from_node_field():
    mg = make_runout_grid()
    mg.add_full("landslide__runout_distance", 25.0, at="node")
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1.0,
        angle_int_frict=10.0,
        selection_method="all",
        enable_runout=True,
        update_soil=True,
        runout_source_mode="selected",
        split_by_width_config=None,
    )

    comp.run_one_step()

    assert np.all(
        mg.at_node["landslide__runout_distance"][
            comp.results["runout"]["source_nodes"]
        ]
        == 25.0
    )


def test_selected_runout_requires_distance_provider():
    mg = make_runout_grid()

    with pytest.raises(ValueError, match="requires runout_distance"):
        ShallowLandslider(
            mg,
            enable_runout=True,
            update_soil=True,
            runout_source_mode="selected",
        )


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

    assert comp.results["newmark"] is not None
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
        runout_results["source_nodes"], comp.results["high_displacement_nodes"]
    )
    assert all(
        path[0] in set(runout_results["source_nodes"])
        for path in runout_results["paths"]
    )
    assert set(np.flatnonzero(runout_results["erosion"])) <= set(
        runout_results["source_nodes"]
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


def test_runout_accepts_trigger_neutral_sources_and_source_distances():
    mg = make_runout_grid()
    runout = ShallowLandslideRunout(mg)

    runout.run_one_step(
        source_nodes=np.array([27, 28]),
        runout_distance=np.array([20.0, 30.0]),
    )

    np.testing.assert_array_equal(runout.results["source_nodes"], [27, 28])
    assert np.sum(runout.results["erosion"]) > 0.0
    assert np.isclose(
        np.sum(runout.results["erosion"]),
        np.sum(runout.results["deposition"]),
    )


def test_runout_retains_failed_nodes_keyword_as_alias():
    mg = make_runout_grid()
    runout = ShallowLandslideRunout(mg)

    runout.run_one_step(failed_nodes=np.array([27]), runout_distance=20.0)

    np.testing.assert_array_equal(runout.results["source_nodes"], [27])
    assert runout.results["failed_nodes"] is runout.results["source_nodes"]


@pytest.mark.parametrize(
    ("source_nodes", "runout_distance", "error", "message"),
    [
        ([27.0], 20.0, TypeError, "integer node IDs"),
        (np.array([999]), 20.0, ValueError, "outside the grid"),
        (np.array([27, 27]), 20.0, ValueError, "duplicate"),
        (np.array([27]), -1.0, ValueError, "cannot be negative"),
        (np.array([27]), np.array([1.0, 2.0]), ValueError, "per source"),
    ],
)
def test_runout_validates_trigger_neutral_inputs(
    source_nodes, runout_distance, error, message
):
    mg = make_runout_grid()
    runout = ShallowLandslideRunout(mg)

    with pytest.raises(error, match=message):
        runout.run_one_step(
            source_nodes=source_nodes,
            runout_distance=runout_distance,
        )


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
        "filled_labels",
        "hole_fill_mask",
        "aspect_labels",
        "selected_labels",
        "group_properties",
    ]:
        assert key in r

def test_aspect_labels_refine_region_labels():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(mg, cohesion_eff=15, angle_int_frict=30)
    comp.run_one_step()

    reg = comp.results["filled_labels"]
    asp = comp.results["aspect_labels"]

    # Aspect subgrouping should not introduce labels with no parent region.
    assert set(np.unique(asp)) - {0} <= set(np.unique(reg)) - {0}
    
def test_dimension_split_labels_consistency():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(
        mg, cohesion_eff=20, angle_int_frict=30,
        split_by_width_config={"kde_data": {"overall": None}, "kde_transform": {}, "width_threshold": 2.0},
    )
    comp.run_one_step()

    asp = comp.results["aspect_labels"]
    dim = comp.results["split_labels"]

    # Split labels should refine but never contradict aspect groups
    assert set(np.unique(dim)) - {0} <= set(np.unique(asp)) - {0}
